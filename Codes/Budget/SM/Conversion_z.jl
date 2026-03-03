using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML


if !isdefined(Main, :FluxUtils)
    include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
    using .FluxUtils: read_bin, bandpassfilter
end


config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg  = TOML.parsefile(config_file)
base  = cfg["base_path"]
base2 = cfg["base_path2"]


# --- Toggle ---
use_3day = true   # true → 3-day chunks;  false → full time average


# --- Domain ---
NX, NY = 288, 468


# --- Tile & grid ---
buf    = 3
tx, ty = 47, 66
nx     = tx + 2 * buf
ny     = ty + 2 * buf
nz     = 88


# --- Time ---
dt   = 25
dto  = 144
Tts  = 366192
nt   = div(Tts, dto)
nt3  = div(nt, 3 * 24)


# --- Thickness ---
thk    = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF    = thk[1:nz]
DRF3d  = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)


# --- Constants ---
g = 9.8


# --- Filter parameters ---
T1, T2, delt, N = 9.0, 15.0, 1.0, 4


# ============================================================================
# MAIN WORKFLOW
# ============================================================================


if use_3day


    println("Computing C(z) [Option 1 W(z)] — 3-day chunks, nchunks = $nt3")
    mkpath(joinpath(base2, "Conv_z_3day"))
    mkpath(joinpath(base2, "Conv_3day"))


    for xn in cfg["xn_start"]:cfg["xn_end"]
        for yn in cfg["yn_start"]:cfg["yn_end"]


            suffix  = @sprintf("%02dx%02d_%d", xn, yn, buf)
            suffix2 = @sprintf("%02dx%02d_%d", xn, yn, buf - 2)


            # --- Read density ---
            rho = Float64.(open(joinpath(base, "Density", "rho_in_$suffix.bin"), "r") do io
                reshape(reinterpret(Float64, read(io, nx * ny * nz * nt * sizeof(Float64))), nx, ny, nz, nt)
            end)


            # --- Grid masks & thickness ---
            hFacC   = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
            DRFfull = hFacC .* DRF3d
            depth   = sum(DRFfull, dims=3)          # (nx, ny, 1)  total depth H
            DRFfull[hFacC .== 0] .= 0.0


            # --- Read filtered velocities ---
            fu = Float64.(open(joinpath(base2, "UVW_F", "fu_$suffix.bin"), "r") do io
                reshape(reinterpret(Float32, read(io, nx * ny * nz * nt * sizeof(Float32))), nx, ny, nz, nt)
            end)


            fv = Float64.(open(joinpath(base2, "UVW_F", "fv_$suffix.bin"), "r") do io
                reshape(reinterpret(Float32, read(io, nx * ny * nz * nt * sizeof(Float32))), nx, ny, nz, nt)
            end)


            # --- Grid spacings ---
            dx = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
            dy = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))


            # --- Bandpass filter density → ρ' ---
            fr = bandpassfilter(rho, T1, T2, delt, N, nt)   # (nx, ny, nz, nt)


            # ----------------------------------------------------------------
            # DEPTH-AVERAGED VELOCITIES
            # UDA, VDA : (nx, ny, nt)
            # ----------------------------------------------------------------
            UDA = dropdims(sum(fu .* DRFfull, dims=3) ./ depth; dims=3)
            VDA = dropdims(sum(fv .* DRFfull, dims=3) ./ depth; dims=3)


            # ----------------------------------------------------------------
            # z and d for W(z) = -∇H·(dUH) - z ∇H·UH
            #
            #   d = total depth           (nx, ny, 1)
            #   z = cumsum DRF top→down   (nx, ny, nz)
            #       at level k this is depth from surface to mid-level k
            # ----------------------------------------------------------------
            d     = depth                                      # (nx, ny, 1)
            z     = cumsum(DRFfull, dims=3)                   # (nx, ny, nz)


            # ----------------------------------------------------------------
            # TERM 1 : -∇H · (d · UH)    shape (nx-2, ny-2, nt)
            # ----------------------------------------------------------------
            dU = dropdims(d, dims=3) .* UDA                   # (nx, ny, nt)
            dV = dropdims(d, dims=3) .* VDA


            term1 = .-(
                (dU[3:nx,   2:ny-1, :] .- dU[1:nx-2, 2:ny-1, :]) ./
                (dx[2:nx-1, 2:ny-1]    .+ dx[1:nx-2, 2:ny-1])     .+
                (dV[2:nx-1, 3:ny,   :] .- dV[2:nx-1, 1:ny-2, :]) ./
                (dy[2:nx-1, 2:ny-1]    .+ dy[2:nx-1, 1:ny-2])
            )                                                  # (nx-2, ny-2, nt)


            # ----------------------------------------------------------------
            # TERM 2 : -z · ∇H · UH      shape (nx-2, ny-2, nz, nt)
            # ----------------------------------------------------------------
            divUDA = (
                (UDA[3:nx,   2:ny-1, :] .- UDA[1:nx-2, 2:ny-1, :]) ./
                (dx[2:nx-1,  2:ny-1]    .+ dx[1:nx-2,  2:ny-1])     .+
                (VDA[2:nx-1, 3:ny,   :] .- VDA[2:nx-1, 1:ny-2, :]) ./
                (dy[2:nx-1,  2:ny-1]    .+ dy[2:nx-1,  1:ny-2])
            )                                                  # (nx-2, ny-2, nt)


            z_int = z[2:nx-1, 2:ny-1, :]                      # (nx-2, ny-2, nz)


            # W(z) = term1 - z * divUDA   broadcasting to (nx-2, ny-2, nz, nt)
            Wz = reshape(term1,  nx-2, ny-2, 1,  nt) .-
                 reshape(z_int,  nx-2, ny-2, nz, 1)  .*
                 reshape(divUDA, nx-2, ny-2, 1,  nt)  # (nx-2, ny-2, nz, nt)


            # ----------------------------------------------------------------
            # CONVERSION  C(z,t) = ρ'(z,t) · g · W(z,t)
            # sign follows paper: C = ρ'gW  (positive = generation)
            # ----------------------------------------------------------------
            rho_int = fr[2:nx-1, 2:ny-1, :, :]               # (nx-2, ny-2, nz, nt)
            Cz      = -rho_int .* g .* Wz                      # (nx-2, ny-2, nz, nt)


            # DRFfull at interior for depth integration
            DRFint   = DRFfull[2:nx-1, 2:ny-1, :]             # (nx-2, ny-2, nz)
            DRFint4d = reshape(DRFint, nx-2, ny-2, nz, 1)


            # Depth-integrated C [W/m²] : (nx-2, ny-2, nt)
            Ca_full = dropdims(
                sum(Cz .* DRFint4d, dims=3),
                dims=3
            )


            # ----------------------------------------------------------------
            # 3-DAY AVERAGING
            # ----------------------------------------------------------------
            hrs_per_chunk = 3 * 24


            # C(z) per 3-day chunk : (nx-2, ny-2, nz, nt3)
            Cz_3day  = zeros(Float32, nx-2, ny-2, nz,  nt3)
            # Depth-averaged C per 3-day chunk : (nx-2, ny-2, nt3)
            Ca_3day  = zeros(Float32, nx-2, ny-2, nt3)


            for t in 1:nt3
                t_start = (t - 1) * hrs_per_chunk + 1
                t_end   = min(t * hrs_per_chunk, nt)
                Cz_3day[:, :, :, t] .= mean(Cz[:, :, :, t_start:t_end],  dims=4)
                Ca_3day[:, :,    t] .= mean(Ca_full[:, :, t_start:t_end], dims=3)
            end


            # Save C(z) 3-day  — shape (nx-2, ny-2, nz, nt3)
            open(joinpath(base2, "Conv_z_3day", "Conv_z_3day_$suffix2.bin"), "w") do io
                write(io, Cz_3day)
            end


            # Save depth-integrated C 3-day [W/m²] — shape (nx-2, ny-2, nt3)
            open(joinpath(base2, "Conv_3day", "Conv_3day_z_$suffix2.bin"), "w") do io
                write(io, Ca_3day)
            end


            println("  Completed tile: $suffix (3-day)")
        end
    end


    println("Done — 3-day C(z) conversion saved.")


else


    println("Computing C(z) [Option 1 W(z)] — full time average")
    mkpath(joinpath(base2, "Conv_z"))
    mkpath(joinpath(base2, "Conv"))


    for xn in cfg["xn_start"]:cfg["xn_end"]
        for yn in cfg["yn_start"]:cfg["yn_end"]


            suffix  = @sprintf("%02dx%02d_%d", xn, yn, buf)
            suffix2 = @sprintf("%02dx%02d_%d", xn, yn, buf - 2)


            # --- Read density ---
            rho = Float64.(open(joinpath(base, "Density", "rho_in_$suffix.bin"), "r") do io
                reshape(reinterpret(Float64, read(io, nx * ny * nz * nt * sizeof(Float64))), nx, ny, nz, nt)
            end)


            # --- Grid masks & thickness ---
            hFacC   = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
            DRFfull = hFacC .* DRF3d
            depth   = sum(DRFfull, dims=3)
            DRFfull[hFacC .== 0] .= 0.0


            # --- Read filtered velocities ---
            fu = Float64.(open(joinpath(base2, "UVW_F", "fu_$suffix.bin"), "r") do io
                reshape(reinterpret(Float32, read(io, nx * ny * nz * nt * sizeof(Float32))), nx, ny, nz, nt)
            end)


            fv = Float64.(open(joinpath(base2, "UVW_F", "fv_$suffix.bin"), "r") do io
                reshape(reinterpret(Float32, read(io, nx * ny * nz * nt * sizeof(Float32))), nx, ny, nz, nt)
            end)


            # --- Grid spacings ---
            dx = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
            dy = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))


            # --- Bandpass filter density → ρ' ---
            fr = bandpassfilter(rho, T1, T2, delt, N, nt)


            # --- Depth-averaged velocities ---
            UDA = dropdims(sum(fu .* DRFfull, dims=3) ./ depth; dims=3)
            VDA = dropdims(sum(fv .* DRFfull, dims=3) ./ depth; dims=3)


            # --- z and d ---
            d   = depth
            z   = cumsum(DRFfull, dims=3)


            # --- Term 1 : -∇H·(d·UH) ---
            dU = dropdims(d, dims=3) .* UDA
            dV = dropdims(d, dims=3) .* VDA


            term1 = .-(
                (dU[3:nx,   2:ny-1, :] .- dU[1:nx-2, 2:ny-1, :]) ./
                (dx[2:nx-1, 2:ny-1]    .+ dx[1:nx-2, 2:ny-1])     .+
                (dV[2:nx-1, 3:ny,   :] .- dV[2:nx-1, 1:ny-2, :]) ./
                (dy[2:nx-1, 2:ny-1]    .+ dy[2:nx-1, 1:ny-2])
            )


            # --- Term 2 : -z·∇H·UH ---
            divUDA = (
                (UDA[3:nx,   2:ny-1, :] .- UDA[1:nx-2, 2:ny-1, :]) ./
                (dx[2:nx-1,  2:ny-1]    .+ dx[1:nx-2,  2:ny-1])     .+
                (VDA[2:nx-1, 3:ny,   :] .- VDA[2:nx-1, 1:ny-2, :]) ./
                (dy[2:nx-1,  2:ny-1]    .+ dy[2:nx-1,  1:ny-2])
            )


            z_int = z[2:nx-1, 2:ny-1, :]


            Wz = reshape(term1,  nx-2, ny-2, 1,  nt) .-
                 reshape(z_int,  nx-2, ny-2, nz, 1)  .*
                 reshape(divUDA, nx-2, ny-2, 1,  nt)


            # --- Conversion C(z,t) = ρ'gW ---
            rho_int = fr[2:nx-1, 2:ny-1, :, :]
            Cz      = rho_int .* g .* Wz


            DRFint    = DRFfull[2:nx-1, 2:ny-1, :]
            DRFint4d  = reshape(DRFint, nx-2, ny-2, nz, 1)


            # Depth-integrated C [W/m²] : (nx-2, ny-2, nt)
            Ca_full = dropdims(
                sum(Cz .* DRFint4d, dims=3),
                dims=3
            )


            # Full time average
            # C(z) time-mean : (nx-2, ny-2, nz)
            Cz_mean = dropdims(mean(Cz,      dims=4), dims=4)
            Ca_mean = dropdims(mean(Ca_full, dims=3), dims=3)


            

            # Save depth-integrated C full mean [W/m²] — shape (nx-2, ny-2)
            open(joinpath(base2, "Conv", "Conv_$suffix2.bin"), "w") do io
                write(io, Float32.(Ca_mean))
            end


            println("  Completed tile: $suffix (full time average)")
        end
    end


    println("Done — full time average C(z) conversion saved.")


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
        ca_3day = open(joinpath(base2, "Conv_3day", "Conv_3day_z_$suffix2.bin"), "r") do io
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




