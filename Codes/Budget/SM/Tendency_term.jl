using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML
using CairoMakie, SparseArrays


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin


config_file = get(ENV, "JULIA_CONFIG",
             joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)


base  = cfg["base_path"]
base2 = cfg["base_path2"]


# --- TIME AVERAGING CONFIGURATION ---
# Set to true for 3-day averaging, false for full time period averaging
use_3day = false  # Change this to true for 3-day averaging


NX, NY = 288, 468
minlat, maxlat = 24.0, 31.91
minlon, maxlon = 193.0, 199.0
lat = range(minlat, maxlat, length=NY)
lon = range(minlon, maxlon, length=NX)


# --- Tile & time parameters ---
buf = 3
tx, ty = 47, 66
nx = tx + 2*buf
ny = ty + 2*buf
nz = 88
dt = 25  # time step in seconds
dto = 144
Tts = 366192
nt = div(Tts, dto)
ts = 72  # timesteps per 3-day period
nt_avg = div(nt, ts)  # number of 3-day periods
nt3 = div(nt, 3*24)  # Number of 3-day periods


# --- Thickness & constants ---
thk = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)


rho0 = 999.8


# ============================================================================
# MAIN WORKFLOW SPLIT: 3-DAY vs FULL TIME AVERAGE
# ============================================================================


if use_3day
    # ========================================================================
    # 3-DAY ENERGY TENDENCY WORKFLOW
    # ========================================================================
    println("Starting energy tendency calculation for $nt3 3-day periods...")
    
    mkpath(joinpath(base2, "TE_t_3day"))
    
    for xn in cfg["xn_start"]:cfg["xn_end"]
        for yn in cfg["yn_start"]:cfg["yn_end"]
            suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
            
            println("\n--- Processing tile: $suffix (3-day) ---")
            
            # --- Read APE ---
            APE = Float64.(open(joinpath(base2, "APE", "APE_t_sm_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * nt * sizeof(Float32)
                reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nz, nt)
            end)
            
            # --- Read KE ---
            ke_t = Float64.(open(joinpath(base2, "KE", "ke_t_sm_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * nt * sizeof(Float32)
                raw_bytes = read(io, nbytes)
                raw_data = reinterpret(Float32, raw_bytes)
                reshape(raw_data, nx, ny, nz, nt)
            end)
            
            # --- Calculate total energy ---
            TE = APE .+ ke_t
            
            # --- Calculate ∂E/∂t using centered differences ---
            dEdt = zeros(Float64, nx, ny, nz, nt)
            dt_output = dt * dto  # = 3600 seconds = 1 hour
            
            # Forward difference for first time step
            dEdt[:, :, :, 1] = (TE[:, :, :, 2] .- TE[:, :, :, 1]) ./ dt_output
            
            # Centered differences for interior points
            for t in 2:nt-1
                dEdt[:, :, :, t] = (TE[:, :, :, t+1] .- TE[:, :, :, t-1]) ./ (2 * dt_output)
            end
            
            # Backward difference for last time step
            dEdt[:, :, :, nt] = (TE[:, :, :, nt] .- TE[:, :, :, nt-1]) ./ dt_output
            
            # --- Replace NaN with zero ---
            dEdt[isnan.(dEdt)] .= 0.0
            
            # --- Depth integrate ---
            dEdt_depth_int = zeros(Float64, nx, ny, nt)
            for t in 1:nt
                for k in 1:nz
                    dEdt_depth_int[:, :, t] .+= dEdt[:, :, k, t] .* DRF[k]
                end
            end
            
            # --- Calculate 3-day averages ---
            println("Calculating 3-day averages...")
            dEdt_3day = zeros(Float64, nx, ny, nt3)
            hrs_per_chunk = 3 * 24
            
            for t in 1:nt3
                t_start = (t-1) * hrs_per_chunk + 1
                t_end = min(t * hrs_per_chunk, nt)
                
                # Average over this 3-day period
                dEdt_3day[:, :, t] = mean(dEdt_depth_int[:, :, t_start:t_end], dims=3)
            end
            
            println("3-day averaging complete")
            
            # --- Save output ---
            output_dir = joinpath(base2, "TE_t_3day")
            
            # Save 3-day averaged energy tendency
            open(joinpath(output_dir, "te_t_3day_$suffix.bin"), "w") do io
                write(io, Float32.(dEdt_3day))
            end
            
            println("Completed tile: $suffix")
            println("Output saved to $output_dir")
        end
    end
    
    println("\n=== All tiles processed successfully (3-day) ===")
    
else
    # ========================================================================
    # FULL TIME AVERAGE ENERGY TENDENCY WORKFLOW
    # ========================================================================
    println("Starting energy tendency calculation for full time average...")
    
    mkpath(joinpath(base2, "TE_t"))
    
    for xn in cfg["xn_start"]:cfg["xn_end"]
        for yn in cfg["yn_start"]:cfg["yn_end"]
            suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
            
            println("\n--- Processing tile: $suffix ---")
            
            # --- Read APE ---
            APE = Float64.(open(joinpath(base2, "APE", "APE_t_sm_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * nt * sizeof(Float32)
                reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nz, nt)
            end)
            
            # --- Read KE ---
            ke_t = Float64.(open(joinpath(base2, "KE", "ke_t_sm_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * nt * sizeof(Float32)
                raw_bytes = read(io, nbytes)
                raw_data = reinterpret(Float32, raw_bytes)
                reshape(raw_data, nx, ny, nz, nt)
            end)
            
            # --- Calculate total energy ---
            TE = APE .+ ke_t
            
            # --- Calculate ∂E/∂t using centered differences ---
            dEdt = zeros(Float64, nx, ny, nz, nt)
            dt_output = dt * dto  # = 3600 seconds = 1 hour
            
            # Forward difference for first time step
            dEdt[:, :, :, 1] = (TE[:, :, :, 2] .- TE[:, :, :, 1]) ./ dt_output
            
            # Centered differences for interior points
            for t in 2:nt-1
                dEdt[:, :, :, t] = (TE[:, :, :, t+1] .- TE[:, :, :, t-1]) ./ (2 * dt_output)
            end
            
            # Backward difference for last time step
            dEdt[:, :, :, nt] = (TE[:, :, :, nt] .- TE[:, :, :, nt-1]) ./ dt_output
            
            # --- Replace NaN with zero ---
            dEdt[isnan.(dEdt)] .= 0.0
            
            # --- Depth integrate ---
            dEdt_depth_int = zeros(Float64, nx, ny, nt)
            for t in 1:nt
                for k in 1:nz
                    dEdt_depth_int[:, :, t] .+= dEdt[:, :, k, t] .* DRF[k]
                end
            end
            
            # --- Time average ---
            dEdt_time_avg = zeros(Float64, nx, ny)
            dEdt_time_avg[:, :] = mean(dEdt_depth_int[:, :, :], dims=3)
            
            # --- Save output ---
            output_dir = joinpath(base2, "TE_t")
            
            # Save time-averaged, depth-integrated tendency
            open(joinpath(output_dir, "te_t_mean_$suffix.bin"), "w") do io
                write(io, Float32.(dEdt_time_avg))
            end
            
            println("Completed tile: $suffix")
            println("Output saved to $output_dir")
        end
    end
    
    println("\n=== All tiles processed successfully ===")
    
end




