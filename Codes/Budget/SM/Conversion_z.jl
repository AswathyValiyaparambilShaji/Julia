using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base  = cfg["base_path"]
base2 = cfg["base_path2"]


use_3day = true


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


kz  = 1
dt  = 25
dto = 144
Tts = 366192
nt  = div(Tts, dto)
nt3 = div(nt, 3*24)


thk   = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF   = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)
g = 9.8


T1, T2, delt, N = 9.0, 15.0, 1.0, 4
fcutlow, fcuthigh = 1 / T2, 1 / T1
fnq = 1 / delt
bpf = digitalfilter(Bandpass(fcutlow, fcuthigh), Butterworth(N); fs = fnq)


if use_3day


    println("Computing -rho'gW conversion for $nt3 3-day periods")
    mkpath(joinpath(base2, "Conv_3day"))


    for xn in cfg["xn_start"]:cfg["xn_end"]
        for yn in cfg["yn_start"]:cfg["yn_end"]
            suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)


            rho = Float64.(open(joinpath(base, "Density", "rho_in_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * nt * sizeof(Float64)
                reshape(reinterpret(Float64, read(io, nbytes)), nx, ny, nz, nt)
            end)


            hFacC   = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
            DRFfull = hFacC .* DRF3d
            depth   = sum(DRFfull, dims=3)                         # (nx, ny, 1)
            DRFfull[hFacC .== 0] .= 0.0


            dx = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
            dy = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))


            fu = Float64.(open(joinpath(base2, "UVW_F", "fu_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * nt * sizeof(Float32)
                reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nz, nt)
            end)


            fv = Float64.(open(joinpath(base2, "UVW_F", "fv_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * nt * sizeof(Float32)
                reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nz, nt)
            end)


            fr = bandpassfilter(rho, T1, T2, delt, N, nt)


            # Depth-averaged velocities
            UDA = dropdims(sum(fu .* DRFfull, dims=3) ./ depth; dims=3)   # (nx, ny, nt)
            VDA = dropdims(sum(fv .* DRFfull, dims=3) ./ depth; dims=3)   # (nx, ny, nt)


            # z = height above bottom at each level (nx, ny, nz)
            # cumsum from bottom (k=nz) upward to k=1
            z_from_bottom = cumsum(DRFfull[:, :, end:-1:1], dims=3)[:, :, end:-1:1]


            # d = total depth (nx, ny)
            d = dropdims(depth, dims=3)                                    # (nx, ny)


            # Term 1: -div(d * UDA)  at interior points (nx-2, ny-2, nt)
            dU = d .* UDA                                                  # (nx, ny, nt)
            dV = d .* VDA                                                  # (nx, ny, nt)


            term1 = .-(
                (dU[3:nx, 2:ny-1, :] .- dU[1:nx-2, 2:ny-1, :]) ./
                (dx[2:nx-1, 2:ny-1] .+ dx[1:nx-2, 2:ny-1]) .+
                (dV[2:nx-1, 3:ny,  :] .- dV[2:nx-1, 1:ny-2, :]) ./
                (dy[2:nx-1, 2:ny-1] .+ dy[2:nx-1, 1:ny-2])
            )                                                              # (nx-2, ny-2, nt)


            # Term 2: -z * div(UDA)  at interior points
            divUDA = (
                (UDA[3:nx, 2:ny-1, :] .- UDA[1:nx-2, 2:ny-1, :]) ./
                (dx[2:nx-1, 2:ny-1] .+ dx[1:nx-2, 2:ny-1]) .+
                (VDA[2:nx-1, 3:ny,  :] .- VDA[2:nx-1, 1:ny-2, :]) ./
                (dy[2:nx-1, 2:ny-1] .+ dy[2:nx-1, 1:ny-2])
            )                                                              # (nx-2, ny-2, nt)


            # z at interior points (nx-2, ny-2, nz)
            z_int = z_from_bottom[2:nx-1, 2:ny-1, :]


            # W(z) = -div(d*U) - z*div(U)  →  (nx-2, ny-2, nz, nt)
            Wz = reshape(term1,   nx-2, ny-2, 1,  nt) .-
                 reshape(z_int,   nx-2, ny-2, nz, 1)  .*
                 reshape(divUDA,  nx-2, ny-2, 1,  nt)


            # Conversion C(z) = -rho'(z) * g * W(z)
            rho_int  = fr[2:nx-1, 2:ny-1, :, :]                          # (nx-2, ny-2, nz, nt)
            Cz       = .-rho_int .* g .* Wz                               # (nx-2, ny-2, nz, nt)


            DRFint   = DRFfull[2:nx-1, 2:ny-1, :]
            DRFint4d = reshape(DRFint, nx-2, ny-2, nz, 1)
            depth_int = sum(DRFint4d, dims=3)                             # (nx-2, ny-2, 1, 1)


            Ca_full  = dropdims(sum(Cz .* DRFint4d, dims=3) ./ depth_int, dims=3)  # (nx-2, ny-2, nt)


            ca_3day       = zeros(nx-2, ny-2, nt3)
            hrs_per_chunk = 3 * 24
            for t in 1:nt3
                t_start = (t-1) * hrs_per_chunk + 1
                t_end   = min(t * hrs_per_chunk, nt)
                ca_3day[:, :, t] .= mean(Ca_full[:, :, t_start:t_end], dims=3)
            end


            suffix2 = @sprintf("%02dx%02d_%d", xn, yn, buf-2)
            open(joinpath(base2, "Conv_3day", "Conv_3day_$suffix2.bin"), "w") do io
                write(io, Float32.(ca_3day))
            end


            println("  Completed tile: $suffix")
        end
    end


    println("Completed -rho'gW conversion for $nt3 3-day periods")


else


    println("Computing -rho'gW conversion for full time average")
    mkpath(joinpath(base2, "Conv"))


    for xn in cfg["xn_start"]:cfg["xn_end"]
        for yn in cfg["yn_start"]:cfg["yn_end"]
            suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)


            rho = Float64.(open(joinpath(base, "Density", "rho_in_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * nt * sizeof(Float64)
                reshape(reinterpret(Float64, read(io, nbytes)), nx, ny, nz, nt)
            end)


            hFacC   = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
            DRFfull = hFacC .* DRF3d
            depth   = sum(DRFfull, dims=3)
            DRFfull[hFacC .== 0] .= 0.0


            dx = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
            dy = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))


            fu = Float64.(open(joinpath(base2, "UVW_F", "fu_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * nt * sizeof(Float32)
                reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nz, nt)
            end)


            fv = Float64.(open(joinpath(base2, "UVW_F", "fv_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * nt * sizeof(Float32)
                reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nz, nt)
            end)


            fr = bandpassfilter(rho, T1, T2, delt, N, nt)


            # Depth-averaged velocities
            UDA = dropdims(sum(fu .* DRFfull, dims=3) ./ depth; dims=3)
            VDA = dropdims(sum(fv .* DRFfull, dims=3) ./ depth; dims=3)


            # z = height above bottom at each level
            z_from_bottom = cumsum(DRFfull[:, :, end:-1:1], dims=3)[:, :, end:-1:1]


            d  = dropdims(depth, dims=3)
            dU = d .* UDA
            dV = d .* VDA


            term1 = .-(
                (dU[3:nx, 2:ny-1, :] .- dU[1:nx-2, 2:ny-1, :]) ./
                (dx[2:nx-1, 2:ny-1] .+ dx[1:nx-2, 2:ny-1]) .+
                (dV[2:nx-1, 3:ny,  :] .- dV[2:nx-1, 1:ny-2, :]) ./
                (dy[2:nx-1, 2:ny-1] .+ dy[2:nx-1, 1:ny-2])
            )


            divUDA = (
                (UDA[3:nx, 2:ny-1, :] .- UDA[1:nx-2, 2:ny-1, :]) ./
                (dx[2:nx-1, 2:ny-1] .+ dx[1:nx-2, 2:ny-1]) .+
                (VDA[2:nx-1, 3:ny,  :] .- VDA[2:nx-1, 1:ny-2, :]) ./
                (dy[2:nx-1, 2:ny-1] .+ dy[2:nx-1, 1:ny-2])
            )


            z_int = z_from_bottom[2:nx-1, 2:ny-1, :]


            Wz = reshape(term1,  nx-2, ny-2, 1,  nt) .-
                 reshape(z_int,  nx-2, ny-2, nz, 1)  .*
                 reshape(divUDA, nx-2, ny-2, 1,  nt)


            rho_int   = fr[2:nx-1, 2:ny-1, :, :]
            Cz        = .-rho_int .* g .* Wz


            DRFint    = DRFfull[2:nx-1, 2:ny-1, :]
            DRFint4d  = reshape(DRFint, nx-2, ny-2, nz, 1)
            depth_int = sum(DRFint4d, dims=3)


            Ca_full = dropdims(sum(Cz .* DRFint4d, dims=3) ./ depth_int, dims=3)
            ca      = dropdims(mean(Ca_full; dims=3); dims=3)


            suffix2 = @sprintf("%02dx%02d_%d", xn, yn, buf-2)
            open(joinpath(base2, "Conv", "Conv_$suffix2.bin"), "w") do io
                write(io, Float32.(ca))
            end


            println("  Completed tile: $suffix")
        end
    end


    println("Completed -rho'gW conversion for full time average")


end
 
using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML
using CairoMakie, SparseArrays


if !isdefined(Main, :FluxUtils)
    include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
end
using .FluxUtils: read_bin


config_file = get(ENV, "JULIA_CONFIG",
              joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)


base  = cfg["base_path"]
base2 = cfg["base_path2"]


NX, NY = 288, 468
minlat, maxlat = 24.0, 31.91
minlon, maxlon = 193.0, 199.0
lat = range(minlat, maxlat, length=NY)
lon = range(minlon, maxlon, length=NX)


buf = 3
tx, ty = 47, 66
nx = tx + 2*buf
ny = ty + 2*buf
nxi = nx - 2
nyi = ny - 2
nt3 = div(div(366192, 144), 3*24)


Conv_full = fill(NaN, NX, NY)


for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]


        suffix2 = @sprintf("%02dx%02d_%d", xn, yn, buf-2)


        # Read 3D (nxi, nyi, nt3) then average over time to get 2D (nxi, nyi)
        ca_3day = open(joinpath(base2, "Conv_3day", "Conv_3day_$suffix2.bin"), "r") do io
            nbytes = nxi * nyi * nt3 * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nxi, nyi, nt3)
        end


        conv_mean = dropdims(mean(ca_3day, dims=3), dims=3)   # (nxi, nyi) — now 2D

        xs = (xn - 1) * tx + 1
        xe = xs + tx + (2 * buf) - 1
        ys = (yn - 1) * ty + 1
        ye = ys + ty + (2 * buf) - 1

        

        Conv_full[xs+2:xe-2, ys+2:ye-2].= conv_mean[2:end-1, 2:end-1]   # strip buffer → (tx, ty)


        println("Completed tile $suffix2")
    end
end


println("\nConv_full range: $(minimum(filter(!isnan, Conv_full))) to $(maximum(filter(!isnan, Conv_full)))")


fig = Figure(size=(1000, 800))


ax = Axis(fig[1, 1],
        title="Depth-Integrated Time-Averaged Conversion (-ρ'gW)",
        xlabel="Longitude [°]",
        ylabel="Latitude [°]")


hm = CairoMakie.heatmap!(ax, lon, lat, Conv_full;
                       interpolate=false,
                       colormap=Reverse(:RdBu),
                       colorrange=(-0.05, 0.05))


Colorbar(fig[1, 2], hm, label="Conversion [W/m²]")


display(fig)


FIGDIR = cfg["fig_base"]
save(joinpath(FIGDIR, "Conv_rhogW_v1.png"), fig)




