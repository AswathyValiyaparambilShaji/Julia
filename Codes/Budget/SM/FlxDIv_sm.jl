using DSP, MAT, Statistics, Printf,  FilePathsBase, LinearAlgebra, TOML
#using CairoMakie, SparseArrays


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base = cfg["base_path"]
base2 = cfg["base_path2"]


# --- TIME AVERAGING CONFIGURATION ---
# Set to true for 3-day averaging, false for full time period averaging
use_3day = true # Change this to true for 3-day averaging


# --- Domain & grid ---
NX, NY = 288, 468
minlat, maxlat = 24.0, 31.91
minlon, maxlon = 193.0, 199.0
lat = range(minlat, maxlat, length=NY)
lon = range(minlon, maxlon, length=NX)


# --- Tile & time ---
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
nt3 = div(nt, 3*24)  # Number of 3-day periods


# --- Thickness & constants ---
thk = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)
g = 9.8


# --- Filter (915 day band, 1 step sampling here) ---
T1, T2, delt, N = 9.0, 15.0, 1.0, 4
fcutlow, fcuthigh = 1 / T2, 1 / T1
fnq = 1 / delt
bpf = digitalfilter(Bandpass(fcutlow, fcuthigh), Butterworth(N); fs = fnq)


# ============================================================================
# MAIN WORKFLOW SPLIT: 3-DAY vs FULL TIME AVERAGE
# ============================================================================


if use_3day
    # ========================================================================
    # 3-DAY FLUX DIVERGENCE WORKFLOW
    # ========================================================================
    println("Computing flux divergence for $nt3 3-day periods")
    
    mkpath(joinpath(base2, "FDiv_3day"))
    
    for xn in cfg["xn_start"]:cfg["xn_end"]
        for yn in cfg["yn_start"]:cfg["yn_end"]
            suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
            
            # --- Read files ---
            hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
            
            # Read 3-day averaged flux files (4D: nx, ny, nz, nt3)
            fx = Float64.(open(joinpath(base2, "xflux", "xflx_3day_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * nt3 * sizeof(Float32)
                raw_bytes = read(io, nbytes)
                raw_data = reinterpret(Float32, raw_bytes)
                reshaped_data = reshape(raw_data, nx, ny, nz, nt3)
            end)
            
            fy = Float64.(open(joinpath(base2, "yflux", "yflx_3day_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * nt3 * sizeof(Float32)
                raw_bytes = read(io, nbytes)
                raw_data = reinterpret(Float32, raw_bytes)
                reshaped_data = reshape(raw_data, nx, ny, nz, nt3)
            end)
            
            dx = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
            dy = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))
            
            # DRFfull
            DRFfull = hFacC .* DRF3d
            z1 = cumsum(DRFfull, dims=3)
            depth = sum(DRFfull, dims=3)
            DRFfull[hFacC .== 0] .= 0.0
            
            # Depth integration for each time period
            fxX = sum(fx .* DRFfull, dims=3)  # Result: (nx, ny, 1, nt3)
            fyY = sum(fy .* DRFfull, dims=3)
            
            # Compute divergence for each 3-day period
            flxD = zeros(nx-2, ny-2, nt3)
            
            for t in 1:nt3
                for i in 2:(nx-2)
                    for j in 2:(ny-2)
                        dF_x_dx = (fxX[i+1, j, 1, t] - fxX[i-1, j, 1, t]) / (dx[i, j] + dx[i-1, j])
                        dF_y_dy = (fyY[i, j+1, 1, t] - fyY[i, j-1, 1, t]) / (dy[i, j] + dy[i, j-1])
                        flxD[i, j, t] = dF_x_dx + dF_y_dy
                    end
                end
            end
            
            # Save 3-day divergence
            suffix2 = @sprintf("%02dx%02d_%d", xn, yn, buf-2)
            open(joinpath(base2, "FDiv_3day", "FDiv_3day_$suffix2.bin"), "w") do io
                write(io, Float32.(flxD))
            end
            
            println("  Completed tile: $suffix (3-day divergence)")
        end
    end
    
    println("Completed flux divergence for $nt3 3-day periods")
    
else
    # ========================================================================
    # FULL TIME AVERAGE FLUX DIVERGENCE WORKFLOW
    # ========================================================================
    println("Computing flux divergence for full time average")
    
    mkpath(joinpath(base2, "FDiv"))
    
    for xn in cfg["xn_start"]:cfg["xn_end"]
        for yn in cfg["yn_start"]:cfg["yn_end"]
            suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
            
            # --- Read files ---
            hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
            
            # Read full time averaged flux files (3D: nx, ny, nz)
            fx = Float64.(open(joinpath(base2, "xflux", "xflx_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * sizeof(Float32)
                raw_bytes = read(io, nbytes)
                raw_data = reinterpret(Float32, raw_bytes)
                reshaped_data = reshape(raw_data, nx, ny, nz)
            end)
            
            fy = Float64.(open(joinpath(base2, "yflux", "yflx_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * sizeof(Float32)
                raw_bytes = read(io, nbytes)
                raw_data = reinterpret(Float32, raw_bytes)
                reshaped_data = reshape(raw_data, nx, ny, nz)
            end)
            
            dx = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
            dy = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))
            
            # DRFfull
            DRFfull = hFacC .* DRF3d
            z1 = cumsum(DRFfull, dims=3)
            depth = sum(DRFfull, dims=3)
            DRFfull[hFacC .== 0] .= 0.0
            
            # Depth integration (single time snapshot)
            fxX = sum(fx .* DRFfull, dims=3)  # Result: (nx, ny, 1)
            fyY = sum(fy .* DRFfull, dims=3)
            
            # Compute divergence
            flxD = zeros(nx-2, ny-2)
            
            for i in 2:(nx-2)
                for j in 2:(ny-2)
                    dF_x_dx = (fxX[i+1, j] - fxX[i-1, j]) / (dx[i, j] + dx[i-1, j])
                    dF_y_dy = (fyY[i, j+1] - fyY[i, j-1]) / (dy[i, j] + dy[i, j-1])
                    flxD[i, j] = dF_x_dx + dF_y_dy
                end
            end
            
            # Save divergence
            suffix2 = @sprintf("%02dx%02d_%d", xn, yn, buf-2)
            open(joinpath(base2, "FDiv", "FDiv_$suffix2.bin"), "w") do io
                write(io, Float32.(flxD))
            end
            
            println("  Completed tile: $suffix")
        end
    end
    
    println("Completed flux divergence for full time average")
    
end




