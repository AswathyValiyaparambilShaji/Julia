using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base = cfg["base_path"]
base2 = cfg["base_path_nt"]


for d in ["xflux","yflux","zflux"]
    mkpath(joinpath(base2, d))
end


NX, NY = 288, 468
minlat, maxlat = 24.0, 31.91
minlon, maxlon = 193.0, 199.0
lat = range(minlat, maxlat, length=NY)
lon = range(minlon, maxlon, length=NX)


buf = 3
tx, ty = 47, 66
nx = tx + 2*buf
ny = ty + 2*buf
nz = 88


kz = 1
dt = 25
dto = 144
Tts = 366192
nt = div(Tts, dto)


# 3-day and weekly parameters
nt_chunk = 72
n_chunks = div(nt, nt_chunk)
wk_start, wk_end = 1249, 1416


thk = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)
g = 9.8


T1, T2, delt, N = 10.2, 32.2, 1.0, 4
fcutlow, fcuthigh = 1 / T2, 1 / T1
fnq = 1 / delt
bpf = digitalfilter(Bandpass(fcutlow, fcuthigh), Butterworth(N); fs = fnq)


for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]


        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)


        hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))


        rho = Float64.(open(joinpath(base,"Density", "rho_in_$suffix.bin"), "r") do io
            raw_bytes = read(io, nx * ny * nz * nt * sizeof(Float64))
            reshape(reinterpret(Float64, raw_bytes), nx, ny, nz, nt)
        end)


        DRFfull = hFacC .* DRF3d
        z = cumsum(DRFfull, dims=3)
        zz = cat(zeros(nx, ny, 1), z; dims=3)
        za = -0.5 .* (zz[:, :, 1:end-1] .+ zz[:, :, 2:end])
        depth = sum(DRFfull, dims=3)
        DRFfull[hFacC .== 0] .= 0.0


        fu = Float64.(open(joinpath(base2, "UVW_NT", "fu_nt_$suffix.bin"), "r") do io
            raw_bytes = read(io, nx * ny * nz * nt * sizeof(Float32))
            reshape(reinterpret(Float32, raw_bytes), nx, ny, nz, nt)
        end)
        fv = Float64.(open(joinpath(base2, "UVW_NT", "fv_nt_$suffix.bin"), "r") do io
            raw_bytes = read(io, nx * ny * nz * nt * sizeof(Float32))
            reshape(reinterpret(Float32, raw_bytes), nx, ny, nz, nt)
        end)
        fw = Float64.(open(joinpath(base2, "UVW_NT", "fw_nt_$suffix.bin"), "r") do io
            raw_bytes = read(io, nx * ny * nz * nt * sizeof(Float32))
            reshape(reinterpret(Float32, raw_bytes), nx, ny, nz, nt)
        end)


        fr = bandpassfilter(rho, T1, T2, delt, N, nt)
        pres  = g .* cumsum(fr .* DRFfull, dims=3)
        pfz   = cat(zeros(nx, ny, 1, nt), pres; dims=3)
        pc_3d = 0.5 .* (pfz[:, :, 1:end-1, :] .+ pfz[:, :, 2:end, :])
        pa    = sum(pc_3d .* DRFfull, dims=3) ./ depth
        pp_3d = pc_3d .- pa


        mask4D = reshape(hFacC .== 0, nx, ny, nz, 1)
        pp_3d[repeat(mask4D, 1, 1, 1, size(pp_3d, 4))] .= 0.0


        ucA_3d = sum(fu .* DRFfull, dims=3) ./ depth
        up_3d  = fu .- ucA_3d
        up_3d[repeat(mask4D, 1, 1, 1, size(up_3d, 4))] .= 0.0


        vcA_3d = sum(fv .* DRFfull, dims=3) ./ depth
        vp_3d  = fv .- vcA_3d
        vp_3d[repeat(mask4D, 1, 1, 1, size(vp_3d, 4))] .= 0.0


        wcA_3d = sum(fw .* DRFfull, dims=3) ./ depth
        wp_3d  = fw .- wcA_3d
        wp_3d[repeat(mask4D, 1, 1, 1, size(wp_3d, 4))] .= 0.0


        xflx_3d = up_3d .* pp_3d
        yflx_3d = vp_3d .* pp_3d
        zflx_3d = wp_3d .* pp_3d


       
        open(joinpath(base2, "xflux", "xflx_$suffix.bin"), "w") do io; write(io, Float32.(xflx_3d)); end
        open(joinpath(base2, "yflux", "yflx_$suffix.bin"), "w") do io; write(io, Float32.(yflx_3d)); end
        open(joinpath(base2, "zflux", "zflx_$suffix.bin"), "w") do io; write(io, Float32.(zflx_3d)); end

        fu = fv =fw = vcA_3d =ucA_3d = wcA_3d = nothing; GC.gc()


        println("Completed tile: $suffix")
    end
end




