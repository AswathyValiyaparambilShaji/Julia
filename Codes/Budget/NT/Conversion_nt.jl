using DSP, MAT, Statistics, Printf, LinearAlgebra, TOML, Dates


include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base  = cfg["base_path"]
base2 = cfg["base_path_nt"]


for d in ["Conv", "Conv_3day", "Conv_wkly2"]
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
dto = 144
Tts = 366192
nt  = div(Tts, dto)
nt_chunk = 72
n_chunks = div(nt, nt_chunk)
ring_steps = nt_chunk
t_safe_start = ring_steps + 1              # first valid step (1801)
t_safe_end   = nt - ring_steps             # last  valid step (nt-1800)


# Safe 3-day chunks: only keep chunks that fall entirely within the safe range
safe_chunks = [c for c in 1:n_chunks
               if (c-1)*nt_chunk + 1 >= t_safe_start &&
                  c*nt_chunk          <= t_safe_end]



# Weekly window from date

t_origin   = DateTime(2012, 3, 1, 0, 0, 0)
t_wk_start = DateTime(2012,  5, 4, 0, 0, 0)
t_wk_end   = DateTime(2012, 5, 18, 18, 0, 0)
wk_start  = Int(Dates.Hour(t_wk_start - t_origin).value) + 1
wk_end    = Int(Dates.Hour(t_wk_end   - t_origin).value) + 1

thk   = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF   = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)
g = 9.8


T1, T2, delt, N = 10.2, 32.2, 1.0, 4


Threads.@threads for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix  = @sprintf("%02dx%02d_%d", xn, yn, buf)
        suffix2 = @sprintf("%02dx%02d_%d", xn, yn, buf-2)
        println("Starting tile: $suffix")


        rho = Float64.(open(joinpath(base, "Density", "rho_in_$suffix.bin"), "r") do io
            reshape(reinterpret(Float64, read(io, nx*ny*nz*nt*sizeof(Float64))), nx, ny, nz, nt)
        end)


        hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
        DRFfull = hFacC .* DRF3d
        depth   = sum(DRFfull, dims=3)
        DRFfull[hFacC .== 0] .= 0.0


        fu = Float64.(open(joinpath(base2, "UVW_NT", "fu_nt_$suffix.bin"), "r") do io
            reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
        end)
        fv = Float64.(open(joinpath(base2, "UVW_NT", "fv_nt_$suffix.bin"), "r") do io
            reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
        end)


        fr = bandpassfilter(rho, T1, T2, delt, N, nt)
        rho = nothing; GC.gc()


        UDA = dropdims(sum(fu .* DRFfull, dims=3) ./ depth; dims=3)
        VDA = dropdims(sum(fv .* DRFfull, dims=3) ./ depth; dims=3)
        fu = nothing; fv = nothing; GC.gc()


        pres  = g .* cumsum(fr .* DRFfull, dims=3)
        fr    = nothing; GC.gc()
        pfz   = cat(zeros(nx, ny, 1, nt), pres; dims=3)
        pres  = nothing; GC.gc()
        pc_3d = 0.5 .* (pfz[:, :, 1:end-1, :] .+ pfz[:, :, 2:end, :])
        pfz   = nothing; GC.gc()
        pa    = sum(pc_3d .* DRFfull, dims=3) ./ depth
        pp_3d = pc_3d .- pa
        pc_3d = nothing; pa = nothing; GC.gc()


        dx = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
        dy = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))


        H    = depth
        pb   = pp_3d[:, :, end, :]
        pp_3d = nothing; GC.gc()


        dHdx = (H[3:nx,   :] .- H[1:nx-2, :]) ./ (dx[2:nx-1, :] .+ dx[1:nx-2, :])
        dHdy = (H[:, 3:ny] .- H[:, 1:ny-2]) ./ (dy[:, 1:ny-2] .+ dy[:, 2:ny-1])


        W1 = .-(UDA[2:end-1, :,      :] .* dHdx)
        W2 = .-(VDA[:,       2:end-1, :] .* dHdy)
        UDA = nothing; VDA = nothing; dHdx = nothing; dHdy = nothing; GC.gc()


        w = W1[:, 2:end-1, :] .+ W2[2:end-1, :, :]
        W1 = nothing; W2 = nothing; GC.gc()


        c = pb[2:end-1, 2:end-1, :] .* w
        pb = nothing; w = nothing; GC.gc()


        open(joinpath(base2, "Conv", "Conv_nt_$suffix2.bin"), "w") do io
            write(io, Float32.(dropdims(mean(c[:, :, t_safe_start:t_safe_end], dims=3), dims=3)))
        end


        Conv_3day = zeros(Float32, nx-2, ny-2, length(safe_chunks))
        for (i, ch) in enumerate(safe_chunks)
            t1 = (ch-1)*nt_chunk + 1
            t2 = ch*nt_chunk
            Conv_3day[:, :, i] = Float32.(dropdims(mean(c[:, :, t1:t2], dims=3), dims=3))
        end
        open(joinpath(base2, "Conv_3day", "Conv_3day_nt_$suffix2.bin"), "w") do io
            write(io, Conv_3day)
        end
        Conv_3day = nothing; GC.gc()


        open(joinpath(base2, "Conv_wkly2", "Conv_wkly_nt_$suffix2.bin"), "w") do io
            write(io, Float32.(dropdims(mean(c[:, :, wk_start:wk_end], dims=3), dims=3)))
        end


        c = nothing; GC.gc()
        println("Completed tile: $suffix")
    end
end




