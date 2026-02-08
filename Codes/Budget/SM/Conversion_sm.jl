using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML
#using CairoMakie, SparseArrays


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base = cfg["base_path"]
base2 = cfg["base_path2"]


# --- TIME AVERAGING CONFIGURATION ---
# Set to true for 3-day averaging, false for full time period averaging
use_3day = true  # Change this to true for 3-day averaging


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
    # 3-DAY CONVERSION WORKFLOW
    # ========================================================================
    println("Computing conversion for $nt3 3-day periods")
    
    mkpath(joinpath(base2, "Conv_3day"))
    
    for xn in cfg["xn_start"]:cfg["xn_end"]
        for yn in cfg["yn_start"]:cfg["yn_end"]
            suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
            
            # --- Read fields ---
            rho = Float64.(open(joinpath(base, "Density", "rho_in_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * nt * sizeof(Float64)
                raw_bytes = read(io, nbytes)
                raw_data = reinterpret(Float64, raw_bytes)
                reshaped_data = reshape(raw_data, nx, ny, nz, nt)
            end)
            
            hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
            
            DRFfull = hFacC .* DRF3d
            z = cumsum(DRFfull, dims=3)
            depth = sum(DRFfull, dims=3)
            DRFfull[hFacC .== 0] .= 0.0
            
            fu = Float64.(open(joinpath(base2, "UVW_F", "fu_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * nt * sizeof(Float32)
                raw_bytes = read(io, nbytes)
                raw_data = reinterpret(Float32, raw_bytes)
                reshaped_data = reshape(raw_data, nx, ny, nz, nt)
            end)
            
            fv = Float64.(open(joinpath(base2, "UVW_F", "fv_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * nt * sizeof(Float32)
                raw_bytes = read(io, nbytes)
                raw_data = reinterpret(Float32, raw_bytes)
                reshaped_data = reshape(raw_data, nx, ny, nz, nt)
            end)
            
            fr = bandpassfilter(rho, T1, T2, delt, N, nt)
            
            UDA = dropdims(sum(fu .* DRFfull, dims=3) ./ depth; dims=3)
            VDA = dropdims(sum(fv .* DRFfull, dims=3) ./ depth; dims=3)
            
            # --- Pressure & perturbations ---
            pres  = g .* cumsum(fr .* DRFfull, dims=3)
            pfz   = cat(zeros(nx, ny, 1, nt), pres; dims=3)
            pc_3d = 0.5 .* (pfz[:, :, 1:end-1, :] .+ pfz[:, :, 2:end, :])
            pa    = sum(pc_3d .* DRFfull, dims=3) ./ depth
            pp_3d = pc_3d .- pa
            
            dx = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
            dy = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))
            
            H = depth
            pb = pp_3d[:, :, end, :]  # Bottom pressure (nx, ny, nt)
            
            dHdx = zeros(nx-2, ny)
            dHdx[:, :] .= (H[3:nx, :] .- H[1:nx-2, :]) ./ (dx[2:nx-1, :] .+ dx[1:nx-2, :])
            
            dHdy = zeros(nx, ny-2)
            dHdy[:, :] .= (H[:, 3:ny] .- H[:, 1:ny-2]) ./ (dy[:, 1:ny-2] .+ dy[:, 2:ny-1])
            
            W1 = .-(UDA[2:end-1, :, :] .* dHdx)
            W2 = .-(VDA[:, 2:end-1, :] .* dHdy)
            
            w1c = W1[:, 2:end-1, :]
            w2c = W2[2:end-1, :, :]
            w   = w1c .+ w2c
            
            # Compute conversion for each 3-day period
            c = pb[2:end-1, 2:end-1, :] .* w  # (nx-2, ny-2, nt)
            
            # Average over 3-day chunks
            ca_3day = zeros(nx-2, ny-2, nt3)
            hrs_per_chunk = 3 * 24
            
            for t in 1:nt3
                t_start = (t-1) * hrs_per_chunk + 1
                t_end = min(t * hrs_per_chunk, nt)
                ca_3day[:, :, t] .= mean(c[:, :, t_start:t_end], dims=3)
            end
            
            # Save 3-day conversion
            suffix2 = @sprintf("%02dx%02d_%d", xn, yn, buf-2)
            open(joinpath(base2, "Conv_3day", "Conv_3day_$suffix2.bin"), "w") do io
                write(io, Float32.(ca_3day))
            end
            
            println("  Completed tile: $suffix (3-day conversion)")
        end
    end
    
    println("Completed conversion for $nt3 3-day periods")
    
else
    # ========================================================================
    # FULL TIME AVERAGE CONVERSION WORKFLOW
    # ========================================================================
    println("Computing conversion for full time average")
    
    mkpath(joinpath(base2, "Conv"))
    
    for xn in cfg["xn_start"]:cfg["xn_end"]
        for yn in cfg["yn_start"]:cfg["yn_end"]
            suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
            
            # --- Read fields ---
            rho = Float64.(open(joinpath(base, "Density", "rho_in_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * nt * sizeof(Float64)
                raw_bytes = read(io, nbytes)
                raw_data = reinterpret(Float64, raw_bytes)
                reshaped_data = reshape(raw_data, nx, ny, nz, nt)
            end)
            
            hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
            
            DRFfull = hFacC .* DRF3d
            z = cumsum(DRFfull, dims=3)
            depth = sum(DRFfull, dims=3)
            DRFfull[hFacC .== 0] .= 0.0
            
            fu = Float64.(open(joinpath(base2, "UVW_F", "fu_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * nt * sizeof(Float32)
                raw_bytes = read(io, nbytes)
                raw_data = reinterpret(Float32, raw_bytes)
                reshaped_data = reshape(raw_data, nx, ny, nz, nt)
            end)
            
            fv = Float64.(open(joinpath(base2, "UVW_F", "fv_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * nt * sizeof(Float32)
                raw_bytes = read(io, nbytes)
                raw_data = reinterpret(Float32, raw_bytes)
                reshaped_data = reshape(raw_data, nx, ny, nz, nt)
            end)
            
            fr = bandpassfilter(rho, T1, T2, delt, N, nt)
            
            UDA = dropdims(sum(fu .* DRFfull, dims=3) ./ depth; dims=3)
            VDA = dropdims(sum(fv .* DRFfull, dims=3) ./ depth; dims=3)
            
            # --- Pressure & perturbations ---
            pres  = g .* cumsum(fr .* DRFfull, dims=3)
            pfz   = cat(zeros(nx, ny, 1, nt), pres; dims=3)
            pc_3d = 0.5 .* (pfz[:, :, 1:end-1, :] .+ pfz[:, :, 2:end, :])
            pa    = sum(pc_3d .* DRFfull, dims=3) ./ depth
            pp_3d = pc_3d .- pa
            
            dx = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
            dy = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))
            
            H = depth
            pb = pp_3d[:, :, end, :]  # Bottom pressure (nx, ny, nt)
            
            dHdx = zeros(nx-2, ny)
            dHdx[:, :] .= (H[3:nx, :] .- H[1:nx-2, :]) ./ (dx[2:nx-1, :] .+ dx[1:nx-2, :])
            
            dHdy = zeros(nx, ny-2)
            dHdy[:, :] .= (H[:, 3:ny] .- H[:, 1:ny-2]) ./ (dy[:, 1:ny-2] .+ dy[:, 2:ny-1])
            
            W1 = .-(UDA[2:end-1, :, :] .* dHdx)
            W2 = .-(VDA[:, 2:end-1, :] .* dHdy)
            
            w1c = W1[:, 2:end-1, :]
            w2c = W2[2:end-1, :, :]
            w   = w1c .+ w2c
            
            # Compute conversion and average over full time period
            c  = pb[2:end-1, 2:end-1, :] .* w
            ca = dropdims(mean(c; dims=3); dims=3)  # (nx-2, ny-2)
            
            # Save conversion
            suffix2 = @sprintf("%02dx%02d_%d", xn, yn, buf-2)
            open(joinpath(base2, "Conv", "Conv_$suffix2.bin"), "w") do io
                write(io, Float32.(ca))
            end
            
            println("  Completed tile: $suffix")
        end
    end
    
    println("Completed conversion for full time average")
    
end
 


