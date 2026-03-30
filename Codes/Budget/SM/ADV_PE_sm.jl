using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML


include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter


config_file = get(ENV, "JULIA_CONFIG",
             joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)


base  = cfg["base_path"]
base2 = cfg["base_path2"]




# --- TIME MODE CONFIGURATION ---
# Options:
#   "3day"   -> PE flux for each 3-day period
#   "weekly" -> PE flux mean over Apr 22 00:00 - Apr 28 23:00
#   "full"   -> PE flux mean over full time record
time_mode = "3day"   # <-- change to "3day", "weekly", or "full"




# --- Domain & grid ---
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
dt = 25
dto = 144
Tts = 366192
nt = div(Tts, dto)
ts = 72                  # timesteps per 3-day period
nt_avg = div(nt, ts)     # number of 3-day periods
nt3 = div(nt, 3*24)      # number of 3-day periods




# -------------------------------------------------------------------------
# Weekly window: April 22 00:00:00 to April 28 23:00:00, 2012
#   Time series starts 2012-03-01T00:00:00, delta_t = 1 hour
#   March = 31 days = 744 hours
#   Apr 22 00:00 = hour 744 + (22-1)*24 = 1248  -> index 1248 + 1 = 1249
#   Apr 28 23:00 = hour 744 +  28 *24-1 = 1415  -> index 1415 + 1 = 1416
#   nt_week = 1416 - 1249 + 1 = 168  (7 days x 24 hrs)
# -------------------------------------------------------------------------
hour_apr22_start = 31*24 + (22-1)*24       # = 1248
hour_apr28_end   = 31*24 +  28   *24 - 1   # = 1415
idx_start        = hour_apr22_start + 1    # = 1249  (1-based)
idx_end          = hour_apr28_end   + 1    # = 1416  (1-based)
nt_week          = idx_end - idx_start + 1 # = 168




# --- Thickness & constants ---
thk = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)


rho0 = 999.8




# ============================================================================
# MAIN WORKFLOW
# ============================================================================


if time_mode == "3day"
    # ========================================================================
    # 3-DAY PE ADVECTIVE FLUX WORKFLOW
    # ========================================================================
    println("Starting PE flux calculation for $nt3 3-day periods...")


    mkpath(joinpath(base2, "U_PE_3dayold"))


    for xn in cfg["xn_start"]:cfg["xn_end"]
        for yn in cfg["yn_start"]:cfg["yn_end"]
            suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)


            println("\n--- Processing tile: $suffix (3-day) ---")


            # --- Read grid metrics ---
            hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
            dx = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
            dy = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))


            # --- Read U and V (3-day averaged) ---
            U = Float64.(open(joinpath(base, "3day_mean", "U", "ucc_3day_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * nt_avg * sizeof(Float32)
                raw_bytes = read(io, nbytes)
                raw_data = reinterpret(Float32, raw_bytes)
                reshape(raw_data, nx, ny, nz, nt_avg)
            end)


            V = Float64.(open(joinpath(base, "3day_mean", "V", "vcc_3day_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * nt_avg * sizeof(Float32)
                raw_bytes = read(io, nbytes)
                raw_data = reinterpret(Float32, raw_bytes)
                reshape(raw_data, nx, ny, nz, nt_avg)
            end)


            # --- Read PE (full temporal resolution) ---
            pe = Float64.(open(joinpath(base2, "pe", "pe_t_sm_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * nt * sizeof(Float32)
                raw_bytes = read(io, nbytes)
                raw_data = reinterpret(Float32, raw_bytes)
                reshape(raw_data, nx, ny, nz, nt)
            end)


            # --- Read N2 (3-day averaged) ---
            N2 = Float64.(open(joinpath(base, "3day_mean", "N2", "N2_3day_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * nt_avg * sizeof(Float32)
                raw_bytes = read(io, nbytes)
                raw_data = reinterpret(Float32, raw_bytes)
                reshape(raw_data, nx, ny, nz, nt_avg)
            end)


            # --- Calculate grid metrics ---
            DRFfull = hFacC .* DRF3d
            DRFfull[hFacC .== 0] .= 0.0


            # --- Adjust N2 to nz+1 levels (interfaces) then back to centers ---
            N2_adjusted = zeros(Float64, nx, ny, nz+1, nt_avg)
            N2_adjusted[:, :, 1, :]      = N2[:, :, 1, :]
            N2_adjusted[:, :, 2:nz, :]   = N2[:, :, 1:nz-1, :]
            N2_adjusted[:, :, nz+1, :]   = N2[:, :, nz, :]


            N2_center = zeros(Float64, nx, ny, nz, nt_avg)
            for k in 1:nz
                N2_center[:, :, k, :] = (N2_adjusted[:, :, k, :] .+ N2_adjusted[:, :, k+1, :]) ./ 2.0
            end


            # --- Filter out anomalously low N2 values ---
            N2_threshold = 1.0e-8
            println("Tile $suffix:")
            println("  Using physical N2 threshold: $N2_threshold")


            n_filtered = sum(N2_center .< N2_threshold)
            n_total = length(N2_center)
            println("  Filtering $(n_filtered) values out of $(n_total) ($(round(100*n_filtered/n_total, digits=2))%)")


            N2_center[N2_center .< N2_threshold] .= N2_threshold
            println("  After filtering - N2 range: ", extrema(N2_center))


            # --- Calculate PE gradients (vectorized) ---
            println("Calculating PE gradients...")
            pe_x = zeros(Float64, nx, ny, nz, nt)
            pe_y = zeros(Float64, nx, ny, nz, nt)


            dx_avg = dx[2:end-1, :] .+ dx[1:end-2, :]
            pe_x[2:end-1, :, :, :] = (pe[3:end, :, :, :] .- pe[1:end-2, :, :, :]) ./
                                      reshape(dx_avg, nx-2, ny, 1, 1)


            dy_avg = dy[:, 2:end-1] .+ dy[:, 1:end-2]
            pe_y[:, 2:end-1, :, :] = (pe[:, 3:end, :, :] .- pe[:, 1:end-2, :, :]) ./
                                      reshape(dy_avg, nx, ny-2, 1, 1)


            println("Gradients calculated")


            # --- Calculate advective PE flux for each 3-day period ---
            println("Calculating advective PE flux for 3-day periods...")
            U_PE_3day = zeros(Float64, nx, ny, nt3)
            hrs_per_chunk = 3 * 24


            for t in 1:nt3
                t_start = (t-1) * hrs_per_chunk + 1
                t_end   = min(t * hrs_per_chunk, nt)


                U_PE_temp = zeros(Float64, nx, ny, t_end - t_start + 1)


                for idx in 1:(t_end - t_start + 1)
                    t_actual = t_start + idx - 1
                    t_avg    = min(div(t_actual - 1, ts) + 1, nt_avg)


                    u_avg  = @view U[:, :, :, t_avg]
                    v_avg  = @view V[:, :, :, t_avg]
                    n2_avg = @view N2_center[:, :, :, t_avg]
                    pe_x_t = @view pe_x[:, :, :, t_actual]
                    pe_y_t = @view pe_y[:, :, :, t_actual]


                    temp1 = u_avg .* pe_x_t ./ n2_avg
                    temp2 = v_avg .* pe_y_t ./ n2_avg


                    temp1[isnan.(temp1)] .= 0.0
                    temp2[isnan.(temp2)] .= 0.0


                    U_PE_temp[:, :, idx] = rho0 .* dropdims(sum((temp1 .+ temp2) .* DRFfull, dims=3), dims=3)
                end


                U_PE_3day[:, :, t] = mean(U_PE_temp, dims=3)
            end


            println("Flux calculation complete")


            output_dir = joinpath(base2, "U_PE_3dayold")
            open(joinpath(output_dir, "u_pe_3day_$suffix.bin"), "w") do io
                write(io, Float32.(U_PE_3day))
            end


            println("Completed tile: $suffix")
            println("Output saved to $output_dir")
        end
    end


    println("\n=== All tiles processed successfully (3-day) ===")




elseif time_mode == "weekly"
    # ========================================================================
    # WEEKLY PE ADVECTIVE FLUX WORKFLOW  (Apr 22 00:00 - Apr 28 23:00)
    # ========================================================================
    println("Starting PE flux calculation for weekly window Apr 22-28 ($nt_week hourly snapshots)...")


    mkpath(joinpath(base2, "U_PE_weeklyold"))


    for xn in cfg["xn_start"]:cfg["xn_end"]
        for yn in cfg["yn_start"]:cfg["yn_end"]
            suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)


            println("\n--- Processing tile: $suffix (weekly) ---")


            # --- Read grid metrics ---
            hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
            dx = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
            dy = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))


            # --- Read U and V (3-day averaged) ---
            U = Float64.(open(joinpath(base, "3day_mean", "U", "ucc_3day_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * nt_avg * sizeof(Float32)
                raw_bytes = read(io, nbytes)
                raw_data = reinterpret(Float32, raw_bytes)
                reshape(raw_data, nx, ny, nz, nt_avg)
            end)


            V = Float64.(open(joinpath(base, "3day_mean", "V", "vcc_3day_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * nt_avg * sizeof(Float32)
                raw_bytes = read(io, nbytes)
                raw_data = reinterpret(Float32, raw_bytes)
                reshape(raw_data, nx, ny, nz, nt_avg)
            end)


            # --- Read full PE then subset to weekly window ---
            pe = Float64.(open(joinpath(base2, "pe", "pe_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * nt * sizeof(Float32)
                raw_bytes = read(io, nbytes)
                raw_data = reinterpret(Float32, raw_bytes)
                reshape(raw_data, nx, ny, nz, nt)
            end)[:, :, :, idx_start:idx_end]


            # --- Read N2 (3-day averaged) ---
            N2 = Float64.(open(joinpath(base, "3day_mean", "N2", "N2_3day_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * nt_avg * sizeof(Float32)
                raw_bytes = read(io, nbytes)
                raw_data = reinterpret(Float32, raw_bytes)
                reshape(raw_data, nx, ny, nz, nt_avg)
            end)


            # --- Calculate grid metrics ---
            DRFfull = hFacC .* DRF3d
            DRFfull[hFacC .== 0] .= 0.0


            # --- Adjust N2 to nz+1 levels (interfaces) then back to centers ---
            N2_adjusted = zeros(Float64, nx, ny, nz+1, nt_avg)
            N2_adjusted[:, :, 1, :]      = N2[:, :, 1, :]
            N2_adjusted[:, :, 2:nz, :]   = N2[:, :, 1:nz-1, :]
            N2_adjusted[:, :, nz+1, :]   = N2[:, :, nz, :]


            N2_center = zeros(Float64, nx, ny, nz, nt_avg)
            for k in 1:nz
                N2_center[:, :, k, :] = (N2_adjusted[:, :, k, :] .+ N2_adjusted[:, :, k+1, :]) ./ 2.0
            end


            # --- Filter out anomalously low N2 values ---
            N2_threshold = 1.0e-8
            println("Tile $suffix:")
            println("  Using physical N2 threshold: $N2_threshold")


            n_filtered = sum(N2_center .< N2_threshold)
            n_total = length(N2_center)
            println("  Filtering $(n_filtered) values out of $(n_total) ($(round(100*n_filtered/n_total, digits=2))%)")


            N2_center[N2_center .< N2_threshold] .= N2_threshold
            println("  After filtering - N2 range: ", extrema(N2_center))


            # --- Calculate PE gradients over weekly window ---
            println("Calculating PE gradients...")
            pe_x = zeros(Float64, nx, ny, nz, nt_week)
            pe_y = zeros(Float64, nx, ny, nz, nt_week)


            dx_avg = dx[2:end-1, :] .+ dx[1:end-2, :]
            pe_x[2:end-1, :, :, :] = (pe[3:end, :, :, :] .- pe[1:end-2, :, :, :]) ./
                                      reshape(dx_avg, nx-2, ny, 1, 1)


            dy_avg = dy[:, 2:end-1] .+ dy[:, 1:end-2]
            pe_y[:, 2:end-1, :, :] = (pe[:, 3:end, :, :] .- pe[:, 1:end-2, :, :]) ./
                                      reshape(dy_avg, nx, ny-2, 1, 1)


            println("Gradients calculated")


            # --- Calculate advective PE flux for each hourly step in window ---
            println("Calculating advective PE flux over weekly window...")
            U_PE = zeros(Float64, nx, ny, nt_week)


            for idx in 1:nt_week
                t_actual = idx_start + idx - 1
                t_avg    = min(div(t_actual - 1, ts) + 1, nt_avg)


                u_avg  = @view U[:, :, :, t_avg]
                v_avg  = @view V[:, :, :, t_avg]
                n2_avg = @view N2_center[:, :, :, t_avg]
                pe_x_t = @view pe_x[:, :, :, idx]
                pe_y_t = @view pe_y[:, :, :, idx]


                temp1 = u_avg .* pe_x_t ./ n2_avg
                temp2 = v_avg .* pe_y_t ./ n2_avg


                temp1[isnan.(temp1)] .= 0.0
                temp2[isnan.(temp2)] .= 0.0


                U_PE[:, :, idx] = rho0 .* dropdims(sum((temp1 .+ temp2) .* DRFfull, dims=3), dims=3)
            end


            println("Flux calculation complete")


            # --- Time average over weekly window ---
            u_pe_mean = dropdims(mean(U_PE, dims=3), dims=3)   # (nx, ny)


            output_dir = joinpath(base2, "U_PE_weeklyold")
            open(joinpath(output_dir, "u_pe_weekly_$suffix.bin"), "w") do io
                write(io, Float32.(u_pe_mean))
            end


            println("Completed tile: $suffix")
            println("Output saved to $output_dir")
        end
    end


    println("\n=== All tiles processed successfully (weekly) ===")




elseif time_mode == "full"
    # ========================================================================
    # FULL TIME AVERAGE PE ADVECTIVE FLUX WORKFLOW
    # ========================================================================
    println("Starting PE flux calculation for full time average...")


    mkpath(joinpath(base2, "U_PE_old"))


    for xn in cfg["xn_start"]:cfg["xn_end"]
        for yn in cfg["yn_start"]:cfg["yn_end"]
            suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)


            println("\n--- Processing tile: $suffix ---")


            # --- Read grid metrics ---
            hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
            dx = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
            dy = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))


            # --- Read U and V (3-day averaged) ---
            U = Float64.(open(joinpath(base, "3day_mean", "U", "ucc_3day_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * nt_avg * sizeof(Float32)
                raw_bytes = read(io, nbytes)
                raw_data = reinterpret(Float32, raw_bytes)
                reshape(raw_data, nx, ny, nz, nt_avg)
            end)


            V = Float64.(open(joinpath(base, "3day_mean", "V", "vcc_3day_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * nt_avg * sizeof(Float32)
                raw_bytes = read(io, nbytes)
                raw_data = reinterpret(Float32, raw_bytes)
                reshape(raw_data, nx, ny, nz, nt_avg)
            end)


            # --- Read PE (full temporal resolution) ---
            pe = Float64.(open(joinpath(base2, "pe", "pe_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * nt * sizeof(Float32)
                raw_bytes = read(io, nbytes)
                raw_data = reinterpret(Float32, raw_bytes)
                reshape(raw_data, nx, ny, nz, nt)
            end)


            # --- Read N2 (3-day averaged) ---
            N2 = Float64.(open(joinpath(base, "3day_mean", "N2", "N2_3day_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * nt_avg * sizeof(Float32)
                raw_bytes = read(io, nbytes)
                raw_data = reinterpret(Float32, raw_bytes)
                reshape(raw_data, nx, ny, nz, nt_avg)
            end)


            # --- Calculate grid metrics ---
            DRFfull = hFacC .* DRF3d
            DRFfull[hFacC .== 0] .= 0.0


            # --- Adjust N2 to nz+1 levels (interfaces) then back to centers ---
            N2_adjusted = zeros(Float64, nx, ny, nz+1, nt_avg)
            N2_adjusted[:, :, 1, :]      = N2[:, :, 1, :]
            N2_adjusted[:, :, 2:nz, :]   = N2[:, :, 1:nz-1, :]
            N2_adjusted[:, :, nz+1, :]   = N2[:, :, nz, :]


            N2_center = zeros(Float64, nx, ny, nz, nt_avg)
            for k in 1:nz
                N2_center[:, :, k, :] = (N2_adjusted[:, :, k, :] .+ N2_adjusted[:, :, k+1, :]) ./ 2.0
            end


            # --- Filter out anomalously low N2 values ---
            N2_threshold = 1.0e-8
            println("Tile $suffix:")
            println("  Using physical N2 threshold: $N2_threshold")


            n_filtered = sum(N2_center .< N2_threshold)
            n_total = length(N2_center)
            println("  Filtering $(n_filtered) values out of $(n_total) ($(round(100*n_filtered/n_total, digits=2))%)")


            N2_center[N2_center .< N2_threshold] .= N2_threshold
            println("  After filtering - N2 range: ", extrema(N2_center))


            # --- Calculate PE gradients (vectorized) ---
            println("Calculating PE gradients...")
            pe_x = zeros(Float64, nx, ny, nz, nt)
            pe_y = zeros(Float64, nx, ny, nz, nt)


            dx_avg = dx[2:end-1, :] .+ dx[1:end-2, :]
            pe_x[2:end-1, :, :, :] = (pe[3:end, :, :, :] .- pe[1:end-2, :, :, :]) ./
                                      reshape(dx_avg, nx-2, ny, 1, 1)


            dy_avg = dy[:, 2:end-1] .+ dy[:, 1:end-2]
            pe_y[:, 2:end-1, :, :] = (pe[:, 3:end, :, :] .- pe[:, 1:end-2, :, :]) ./
                                      reshape(dy_avg, nx, ny-2, 1, 1)


            println("Gradients calculated")


            # --- Initialize output: depth-integrated flux at each timestep ---
            U_PE = zeros(Float64, nx, ny, nt)


            # --- Calculate advective PE flux for each timestep ---
            println("Calculating advective PE flux...")
            for t in 1:nt
                t_avg  = min(div(t - 1, ts) + 1, nt_avg)


                u_avg  = @view U[:, :, :, t_avg]
                v_avg  = @view V[:, :, :, t_avg]
                n2_avg = @view N2_center[:, :, :, t_avg]
                pe_x_t = @view pe_x[:, :, :, t]
                pe_y_t = @view pe_y[:, :, :, t]


                temp1 = u_avg .* pe_x_t ./ n2_avg
                temp2 = v_avg .* pe_y_t ./ n2_avg


                temp1[isnan.(temp1)] .= 0.0
                temp2[isnan.(temp2)] .= 0.0


                U_PE[:, :, t] = rho0 .* dropdims(sum((temp1 .+ temp2) .* DRFfull, dims=3), dims=3)
            end


            println("Flux calculation complete")


            # --- Time average ---
            u_pe_mean = dropdims(mean(U_PE, dims=3), dims=3)   # (nx, ny)


            output_dir = joinpath(base2, "U_PE_old")
            open(joinpath(output_dir, "u_pe_mean_$suffix.bin"), "w") do io
                write(io, Float32.(u_pe_mean))
            end


            println("Completed tile: $suffix")
            println("Output saved to $output_dir")
        end
    end


    println("\n=== All tiles processed successfully ===")


else
    error("Unknown time_mode '$time_mode'. Choose \"3day\", \"weekly\", or \"full\".")


end




