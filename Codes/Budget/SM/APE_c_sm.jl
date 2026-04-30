using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML


include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter


config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg  = TOML.parsefile(config_file)
base  = cfg["base_path"]
base2 = cfg["base_path2"]


NX, NY = 288, 468
buf = 3
tx, ty = 47, 66
nx = tx + 2*buf
ny = ty + 2*buf
nz = 88


dt  = 25
dto = 144
Tts = 366192
nt  = div(Tts, dto)
ts      = 72
nt_avg  = div(nt, ts)


thk   = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF   = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)
g     = 9.8
rho0  = 1027.5


T1, T2, delt, N = 9.0, 15.0, 1.0, 4
N2_threshold = 1.0e-8


mkpath(joinpath(base2, "APE"))


for xn in cfg["xn_start"]:cfg["xn_end"]
for yn in cfg["yn_start"]:cfg["yn_end"]


    suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
    println("Processing tile: $suffix")


    hFacC   = read_bin(joinpath(base, "hFacC", "hFacC_$suffix.bin"), (nx, ny, nz))
    DRFfull = hFacC .* DRF3d
    DRFfull[hFacC .== 0] .= 0.0
    mask3D  = hFacC .== 0


    rho = open(joinpath(base, "Density", "rho_in_$suffix.bin"), "r") do io
        Float64.(reshape(reinterpret(Float64, read(io, nx*ny*nz*nt*sizeof(Float64))), nx, ny, nz, nt))
    end
    rho_prime = bandpassfilter(rho, T1, T2, delt, N, nt)
    rho = nothing; GC.gc()
    rho_prime[repeat(mask3D, 1, 1, 1, nt)] .= 0.0


    N2_phase = Float64.(open(joinpath(base, "3day_mean", "N2", "N2_3day_$suffix.bin"), "r") do io
        raw = read(io, nx * ny * nz * nt_avg * sizeof(Float32))
        reshape(reinterpret(Float32, raw), nx, ny, nz, nt_avg)
    end)


    N2_adjusted = zeros(Float64, nx, ny, nz+1, nt_avg)
    N2_adjusted[:, :, 1,    :] = N2_phase[:, :, 1,      :]
    N2_adjusted[:, :, 2:nz, :] = N2_phase[:, :, 1:nz-1, :]
    N2_adjusted[:, :, nz+1, :] = N2_phase[:, :, nz-1,   :]


    k_last_full = zeros(Int, nx, ny)
    for j in 1:ny, i in 1:nx
        for k in nz:-1:1
            if hFacC[i, j, k] >= 1.0
                k_last_full[i, j] = k
                break
            end
        end
    end


    for j in 1:ny, i in 1:nx
        kf = k_last_full[i, j]
        if kf > 0
            N2_adjusted[i, j, kf+1, :] .= N2_phase[i, j, kf-1, :] # k+1 because of the concatination of adition surface grid
        end
    end


    N2_center = zeros(Float64, nx, ny, nz, nt_avg)
    for k in 1:nz
        N2_center[:, :, k, :] .= 0.5 .* (N2_adjusted[:, :, k, :] .+ N2_adjusted[:, :, k+1, :])
    end
    N2_adjusted = nothing
    N2_phase    = nothing


    N2_center[N2_center .< N2_threshold] .= N2_threshold
    println("  N2 range: ", extrema(N2_center))


    APE = fill(NaN, nx, ny, nz, nt)
    for t in 1:nt_avg
        n2_t   = N2_center[:, :, :, t]
        tstart = (t - 1) * ts + 1
        tend   = min(t * ts, nt)
        for tt in tstart:tend
            APE[:, :, :, tt] .= (g^2 .* rho_prime[:, :, :, tt].^2) ./ (2.0 .* rho0 .* n2_t)
        end
    end
    rho_prime = N2_center = nothing; GC.gc()


    println("  APE range: ", extrema(filter(isfinite, APE)))
println(APE[1:10,1:10,10,10])

    open(joinpath(base2, "APE", "APE_tc_sm_$suffix.bin"), "w") do io
        write(io, Float32.(APE))
    end
    APE = nothing; GC.gc()


    println("Completed tile: $suffix\n")
end
end




