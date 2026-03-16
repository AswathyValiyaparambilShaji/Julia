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
#   "3day"   -> KE flux for each 3-day period
#   "weekly" -> KE flux mean over Apr 22 00:00 - Apr 28 23:00
#   "full"   -> KE flux mean over full time record
time_mode = "full"   # <-- change to "3day", "weekly", or "full"




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
    # 3-DAY KE ADVECTIVE FLUX WORKFLOW
    # ========================================================================
    println("Starting KE flux calculation for $nt3 3-day periods...")


    mkpath(joinpath(base2, "U_KE_3day"))


    for xn in cfg["xn_start"]:cfg["xn_end"]
        for yn in cfg["yn_start"]:cfg["yn_end"]
            suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)


            println("\n--- Processing tile: $suffix (3-day) ---")


            # --- Read grid metrics ---
            hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
            dx = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
            dy = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))


            # --- Read velocity fields (3-day averaged) ---
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


            # --- Read kinetic energy (full temporal resolution) ---
            ke_t = Float64.(open(joinpath(base2, "KE", "ke_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * nt * sizeof(Float32)
                raw_bytes = read(io, nbytes)
                raw_data = reinterpret(Float32, raw_bytes)
                reshape(raw_data, nx, ny, nz, nt)
            end)


            # --- Calculate cell thicknesses ---
            DRFfull = hFacC .* DRF3d
            DRFfull[hFacC .== 0] .= 0.0


            # --- Calculate KE gradients (vectorized) ---
            println("Calculating KE gradients...")
            ke_x = zeros(Float64, nx, ny, nz, nt)
            ke_y = zeros(Float64, nx, ny, nz, nt)


            dx_avg = dx[2:end-1, :] .+ dx[1:end-2, :]
            ke_x[2:end-1, :, :, :] = (ke_t[3:end, :, :, :] .- ke_t[1:end-2, :, :, :]) ./
                                      reshape(dx_avg, nx-2, ny, 1, 1)


            dy_avg = dy[:, 2:end-1] .+ dy[:, 1:end-2]
            ke_y[:, 2:end-1, :, :] = (ke_t[:, 3:end, :, :] .- ke_t[:, 1:end-2, :, :]) ./
                                      reshape(dy_avg, nx, ny-2, 1, 1)


            println("Gradients calculated")


            # --- Calculate advective KE flux for each 3-day period ---
            println("Calculating advective KE flux for 3-day periods...")
            U_KE_3day = zeros(Float64, nx, ny, nt3)
            hrs_per_chunk = 3 * 24


            for t in 1:nt3
                t_start = (t-1) * hrs_per_chunk + 1
                t_end   = min(t * hrs_per_chunk, nt)


                U_KE_temp = zeros(Float64, nx, ny, t_end - t_start + 1)


                for idx in 1:(t_end - t_start + 1)
                    t_actual = t_start + idx - 1
                    t_avg    = min(div(t_actual - 1, ts) + 1, nt_avg)


                    u_avg  = @view U[:, :, :, t_avg]
                    v_avg  = @view V[:, :, :, t_avg]
                    ke_x_t = @view ke_x[:, :, :, t_actual]
                    ke_y_t = @view ke_y[:, :, :, t_actual]


                    temp1 = u_avg .* ke_x_t
                    temp2 = v_avg .* ke_y_t


                    U_KE_temp[:, :, idx] = dropdims(sum((temp1 .+ temp2) .* DRFfull, dims=3), dims=3)
                end


                U_KE_3day[:, :, t] = mean(U_KE_temp, dims=3)
            end


            println("Flux calculation complete")


            output_dir = joinpath(base2, "U_KE_3day")
            open(joinpath(output_dir, "u_ke_3day_$suffix.bin"), "w") do io
                write(io, Float32.(U_KE_3day))
            end


            println("Completed tile: $suffix")
            println("Output saved to $output_dir")
        end
    end


    println("\n=== All tiles processed successfully (3-day) ===")




elseif time_mode == "weekly"
    # ========================================================================
    # WEEKLY KE ADVECTIVE FLUX WORKFLOW  (Apr 22 00:00 - Apr 28 23:00)
    # ========================================================================
    println("Starting KE flux calculation for weekly window Apr 22-28 ($nt_week hourly snapshots)...")


    mkpath(joinpath(base2, "U_KE_weekly"))


    for xn in cfg["xn_start"]:cfg["xn_end"]
        for yn in cfg["yn_start"]:cfg["yn_end"]
            suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)


            println("\n--- Processing tile: $suffix (weekly) ---")


            # --- Read grid metrics ---
            hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
            dx = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
            dy = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))


            # --- Read velocity fields (3-day averaged) ---
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


            # --- Read full ke_t then subset to weekly window ---
            ke_t = Float64.(open(joinpath(base2, "KE", "ke_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * nt * sizeof(Float32)
                raw_bytes = read(io, nbytes)
                raw_data = reinterpret(Float32, raw_bytes)
                reshape(raw_data, nx, ny, nz, nt)
            end)[:, :, :, idx_start:idx_end]


            # --- Calculate cell thicknesses ---
            DRFfull = hFacC .* DRF3d
            DRFfull[hFacC .== 0] .= 0.0


            # --- Calculate KE gradients over weekly window ---
            println("Calculating KE gradients...")
            ke_x = zeros(Float64, nx, ny, nz, nt_week)
            ke_y = zeros(Float64, nx, ny, nz, nt_week)


            dx_avg = dx[2:end-1, :] .+ dx[1:end-2, :]
            ke_x[2:end-1, :, :, :] = (ke_t[3:end, :, :, :] .- ke_t[1:end-2, :, :, :]) ./
                                      reshape(dx_avg, nx-2, ny, 1, 1)


            dy_avg = dy[:, 2:end-1] .+ dy[:, 1:end-2]
            ke_y[:, 2:end-1, :, :] = (ke_t[:, 3:end, :, :] .- ke_t[:, 1:end-2, :, :]) ./
                                      reshape(dy_avg, nx, ny-2, 1, 1)


            println("Gradients calculated")


            # --- Calculate advective KE flux for each hourly step in window ---
            println("Calculating advective KE flux over weekly window...")
            U_KE = zeros(Float64, nx, ny, nt_week)


            for idx in 1:nt_week
                t_actual = idx_start + idx - 1
                t_avg    = min(div(t_actual - 1, ts) + 1, nt_avg)


                u_avg  = @view U[:, :, :, t_avg]
                v_avg  = @view V[:, :, :, t_avg]
                ke_x_t = @view ke_x[:, :, :, idx]
                ke_y_t = @view ke_y[:, :, :, idx]


                temp1 = u_avg .* ke_x_t
                temp2 = v_avg .* ke_y_t


                U_KE[:, :, idx] = dropdims(sum((temp1 .+ temp2) .* DRFfull, dims=3), dims=3)
            end


            println("Flux calculation complete")


            # --- Time average over weekly window ---
            u_ke_mean = dropdims(mean(U_KE, dims=3), dims=3)   # (nx, ny)


            output_dir = joinpath(base2, "U_KE_weekly")
            open(joinpath(output_dir, "u_ke_weekly_$suffix.bin"), "w") do io
                write(io, Float32.(u_ke_mean))
            end


            println("Completed tile: $suffix")
            println("Output saved to $output_dir")
        end
    end


    println("\n=== All tiles processed successfully (weekly) ===")




elseif time_mode == "full"
    # ========================================================================
    # FULL TIME AVERAGE KE ADVECTIVE FLUX WORKFLOW
    # ========================================================================
    println("Starting KE flux calculation for full time average...")


    mkpath(joinpath(base2, "U_KE"))


    for xn in cfg["xn_start"]:cfg["xn_end"]
        for yn in cfg["yn_start"]:cfg["yn_end"]
            suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)


            println("\n--- Processing tile: $suffix ---")


            # --- Read grid metrics ---
            hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
            dx = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
            dy = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))


            # --- Read velocity fields (3-day averaged) ---
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


            # --- Read kinetic energy (full temporal resolution) ---
            ke_t = Float64.(open(joinpath(base2, "KE", "ke_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * nt * sizeof(Float32)
                raw_bytes = read(io, nbytes)
                raw_data = reinterpret(Float32, raw_bytes)
                reshape(raw_data, nx, ny, nz, nt)
            end)


            # --- Calculate cell thicknesses ---
            DRFfull = hFacC .* DRF3d
            DRFfull[hFacC .== 0] .= 0.0


            # --- Calculate KE gradients (vectorized) ---
            println("Calculating KE gradients...")
            ke_x = zeros(Float64, nx, ny, nz, nt)
            ke_y = zeros(Float64, nx, ny, nz, nt)


            dx_avg = dx[2:end-1, :] .+ dx[1:end-2, :]
            ke_x[2:end-1, :, :, :] = (ke_t[3:end, :, :, :] .- ke_t[1:end-2, :, :, :]) ./
                                      reshape(dx_avg, nx-2, ny, 1, 1)


            dy_avg = dy[:, 2:end-1] .+ dy[:, 1:end-2]
            ke_y[:, 2:end-1, :, :] = (ke_t[:, 3:end, :, :] .- ke_t[:, 1:end-2, :, :]) ./
                                      reshape(dy_avg, nx, ny-2, 1, 1)


            println("Gradients calculated")


            # --- Initialize output array ---
            U_KE = zeros(Float64, nx, ny, nt)


            # --- Calculate advective KE flux for each timestep ---
            println("Calculating advective KE flux...")
            for t in 1:nt
                t_avg  = min(div(t - 1, ts) + 1, nt_avg)


                u_avg  = @view U[:, :, :, t_avg]
                v_avg  = @view V[:, :, :, t_avg]
                ke_x_t = @view ke_x[:, :, :, t]
                ke_y_t = @view ke_y[:, :, :, t]


                temp1 = u_avg .* ke_x_t
                temp2 = v_avg .* ke_y_t


                U_KE[:, :, t] = dropdims(sum((temp1 .+ temp2) .* DRFfull, dims=3), dims=3)
            end


            println("Flux calculation complete")


            # --- Time average ---
            u_ke_mean = dropdims(mean(U_KE, dims=3), dims=3)   # (nx, ny)


            output_dir = joinpath(base2, "U_KE")
            open(joinpath(output_dir, "u_ke_mean_$suffix.bin"), "w") do io
                write(io, Float32.(u_ke_mean))
            end


            println("Completed tile: $suffix")
            println("Output saved to $output_dir")
        end
    end


    println("\n=== All tiles processed successfully ===")


else
    error("Unknown time_mode '$time_mode'. Choose \"3day\", \"weekly\", or \"full\".")


end




