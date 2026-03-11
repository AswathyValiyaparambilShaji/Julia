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
#   "3day"   -> vertical shear production for each 3-day period
#   "weekly" -> vertical shear production mean over Apr 22 00:00 - Apr 28 23:00
#   "full"   -> vertical shear production mean over full time record
time_mode = "weekly"   # <-- change to "3day", "weekly", or "full"




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


@printf("Weekly window: Apr 22 00:00 - Apr 28 23:00  ->  indices %d:%d  (%d hourly snapshots)\n",
        idx_start, idx_end, nt_week)




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
    # 3-DAY VERTICAL SHEAR PRODUCTION WORKFLOW
    # ========================================================================
    println("Starting vertical shear production calculation for $nt3 3-day periods...")


    mkpath(joinpath(base2, "SP_V_3day"))


    for xn in cfg["xn_start"]:cfg["xn_end"]
        for yn in cfg["yn_start"]:cfg["yn_end"]
            suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)


            println("\n--- Processing tile: $suffix (3-day) ---")


            # --- Read grid metrics ---
            hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))


            # --- Read mean velocity fields (3-day averaged) ---
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


            # --- Read fluctuating velocities ---
            fu = Float64.(open(joinpath(base2, "UVW_F", "fu_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * nt * sizeof(Float32)
                raw_bytes = read(io, nbytes)
                raw_data = reinterpret(Float32, raw_bytes)
                reshape(raw_data, nx, ny, nz, nt)
            end)


            fv = Float64.(open(joinpath(base2, "UVW_F", "fv_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * nt * sizeof(Float32)
                raw_bytes = read(io, nbytes)
                raw_data = reinterpret(Float32, raw_bytes)
                reshape(raw_data, nx, ny, nz, nt)
            end)


            fw = Float64.(open(joinpath(base2, "UVW_F", "fw_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * nt * sizeof(Float32)
                raw_bytes = read(io, nbytes)
                raw_data = reinterpret(Float32, raw_bytes)
                reshape(raw_data, nx, ny, nz, nt)
            end)


            # --- Calculate cell thicknesses ---
            DRFfull = hFacC .* DRF3d
            DRFfull[hFacC .== 0] .= 0.0


            # --- Calculate vertical gradients of mean velocities: ∂U/∂z, ∂V/∂z ---
            println("Calculating vertical gradients of mean velocities...")
            U_z = zeros(Float64, nx, ny, nz, nt_avg)
            V_z = zeros(Float64, nx, ny, nz, nt_avg)


            for t_avg in 1:nt_avg
                for k in 2:nz-1
                    dz = DRF[k-1]/2.0 + DRF[k] + DRF[k+1]/2.0
                    U_z[:, :, k, t_avg] = (U[:, :, k-1, t_avg] .- U[:, :, k+1, t_avg]) ./ dz
                    V_z[:, :, k, t_avg] = (V[:, :, k-1, t_avg] .- V[:, :, k+1, t_avg]) ./ dz
                end
            end


            println("Vertical gradients calculated")


            # --- Calculate vertical shear production for each 3-day period ---
            println("Calculating vertical shear production for 3-day periods...")
            SP_V_3day = zeros(Float64, nx, ny, nt3)
            hrs_per_chunk = 3 * 24


            for t in 1:nt3
                t_start = (t-1) * hrs_per_chunk + 1
                t_end   = min(t * hrs_per_chunk, nt)


                sp_v_temp = zeros(Float64, nx, ny, t_end - t_start + 1)


                for idx in 1:(t_end - t_start + 1)
                    t_actual = t_start + idx - 1
                    t_avg    = min(div(t_actual - 1, ts) + 1, nt_avg)


                    U_z_t = @view U_z[:, :, :, t_avg]
                    V_z_t = @view V_z[:, :, :, t_avg]
                    ut    = @view fu[:, :, :, t_actual]
                    vt    = @view fv[:, :, :, t_actual]
                    wt    = @view fw[:, :, :, t_actual]


                    temp1 = wt .* ut .* U_z_t .* DRFfull
                    temp2 = wt .* vt .* V_z_t .* DRFfull


                    sp_v_temp[:, :, idx] = -rho0 .* dropdims(sum((temp1 .+ temp2), dims=3), dims=3)
                end


                SP_V_3day[:, :, t] = mean(sp_v_temp, dims=3)
            end


            println("Vertical shear production calculation complete")


            output_dir = joinpath(base2, "SP_V_3day")
            open(joinpath(output_dir, "sp_v_3day_$suffix.bin"), "w") do io
                write(io, Float32.(SP_V_3day))
            end


            println("Completed tile: $suffix")
            println("Output saved to $output_dir")
        end
    end


    println("\n=== All tiles processed successfully (3-day) ===")




elseif time_mode == "weekly"
    # ========================================================================
    # WEEKLY VERTICAL SHEAR PRODUCTION WORKFLOW  (Apr 22 00:00 - Apr 28 23:00)
    # ========================================================================
    println("Starting vertical shear production calculation for weekly window Apr 22-28 ($nt_week hourly snapshots)...")


    mkpath(joinpath(base2, "SP_V_weekly"))


    for xn in cfg["xn_start"]:cfg["xn_end"]
        for yn in cfg["yn_start"]:cfg["yn_end"]
            suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)


            println("\n--- Processing tile: $suffix (weekly) ---")


            # --- Read grid metrics ---
            hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))


            # --- Read mean velocity fields (3-day averaged, full record) ---
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


            # --- Read full fluctuating velocities then subset to weekly window ---
            fu = Float64.(open(joinpath(base2, "UVW_F", "fu_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * nt * sizeof(Float32)
                raw_bytes = read(io, nbytes)
                raw_data = reinterpret(Float32, raw_bytes)
                reshape(raw_data, nx, ny, nz, nt)
            end)[:, :, :, idx_start:idx_end]


            fv = Float64.(open(joinpath(base2, "UVW_F", "fv_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * nt * sizeof(Float32)
                raw_bytes = read(io, nbytes)
                raw_data = reinterpret(Float32, raw_bytes)
                reshape(raw_data, nx, ny, nz, nt)
            end)[:, :, :, idx_start:idx_end]


            fw = Float64.(open(joinpath(base2, "UVW_F", "fw_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * nt * sizeof(Float32)
                raw_bytes = read(io, nbytes)
                raw_data = reinterpret(Float32, raw_bytes)
                reshape(raw_data, nx, ny, nz, nt)
            end)[:, :, :, idx_start:idx_end]


            # --- Calculate cell thicknesses ---
            DRFfull = hFacC .* DRF3d
            DRFfull[hFacC .== 0] .= 0.0


            # --- Calculate vertical gradients of mean velocities: ∂U/∂z, ∂V/∂z ---
            println("Calculating vertical gradients of mean velocities...")
            U_z = zeros(Float64, nx, ny, nz, nt_avg)
            V_z = zeros(Float64, nx, ny, nz, nt_avg)


            for t_avg in 1:nt_avg
                for k in 2:nz-1
                    dz = DRF[k-1]/2.0 + DRF[k] + DRF[k+1]/2.0
                    U_z[:, :, k, t_avg] = (U[:, :, k-1, t_avg] .- U[:, :, k+1, t_avg]) ./ dz
                    V_z[:, :, k, t_avg] = (V[:, :, k-1, t_avg] .- V[:, :, k+1, t_avg]) ./ dz
                end
            end


            println("Vertical gradients calculated")


            # --- Calculate vertical shear production for each hourly step in window ---
            println("Calculating vertical shear production over weekly window...")
            sp_v = zeros(Float64, nx, ny, nt_week)


            for idx in 1:nt_week
                t_actual = idx_start + idx - 1
                t_avg    = min(div(t_actual - 1, ts) + 1, nt_avg)


                U_z_t = @view U_z[:, :, :, t_avg]
                V_z_t = @view V_z[:, :, :, t_avg]
                ut    = @view fu[:, :, :, idx]
                vt    = @view fv[:, :, :, idx]
                wt    = @view fw[:, :, :, idx]


                temp1 = wt .* ut .* U_z_t .* DRFfull
                temp2 = wt .* vt .* V_z_t .* DRFfull


                sp_v[:, :, idx] = -rho0 .* dropdims(sum((temp1 .+ temp2), dims=3), dims=3)
            end


            println("Vertical shear production calculation complete")


            # --- Time average over weekly window ---
            SP_V = dropdims(mean(sp_v, dims=3), dims=3)   # (nx, ny)


            output_dir = joinpath(base2, "SP_V_weekly")
            open(joinpath(output_dir, "sp_v_weekly_$suffix.bin"), "w") do io
                write(io, Float32.(SP_V))
            end


            println("Completed tile: $suffix")
            println("Output saved to $output_dir")
        end
    end


    println("\n=== All tiles processed successfully (weekly) ===")




elseif time_mode == "full"
    # ========================================================================
    # FULL TIME AVERAGE VERTICAL SHEAR PRODUCTION WORKFLOW
    # ========================================================================
    println("Starting vertical shear production calculation for full time average...")


    mkpath(joinpath(base2, "SP_V"))


    for xn in cfg["xn_start"]:cfg["xn_end"]
        for yn in cfg["yn_start"]:cfg["yn_end"]
            suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)


            println("\n--- Processing tile: $suffix ---")


            # --- Read grid metrics ---
            hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))


            # --- Read mean velocity fields (3-day averaged) ---
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


            # --- Read fluctuating velocities ---
            fu = Float64.(open(joinpath(base2, "UVW_F", "fu_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * nt * sizeof(Float32)
                raw_bytes = read(io, nbytes)
                raw_data = reinterpret(Float32, raw_bytes)
                reshape(raw_data, nx, ny, nz, nt)
            end)


            fv = Float64.(open(joinpath(base2, "UVW_F", "fv_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * nt * sizeof(Float32)
                raw_bytes = read(io, nbytes)
                raw_data = reinterpret(Float32, raw_bytes)
                reshape(raw_data, nx, ny, nz, nt)
            end)


            fw = Float64.(open(joinpath(base2, "UVW_F", "fw_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * nt * sizeof(Float32)
                raw_bytes = read(io, nbytes)
                raw_data = reinterpret(Float32, raw_bytes)
                reshape(raw_data, nx, ny, nz, nt)
            end)


            # --- Calculate cell thicknesses ---
            DRFfull = hFacC .* DRF3d
            DRFfull[hFacC .== 0] .= 0.0


            # --- Calculate vertical gradients of mean velocities: ∂U/∂z, ∂V/∂z ---
            println("Calculating vertical gradients of mean velocities...")
            U_z = zeros(Float64, nx, ny, nz, nt_avg)
            V_z = zeros(Float64, nx, ny, nz, nt_avg)


            for t_avg in 1:nt_avg
                for k in 2:nz-1
                    dz = DRF[k-1]/2.0 + DRF[k] + DRF[k+1]/2.0
                    U_z[:, :, k, t_avg] = (U[:, :, k-1, t_avg] .- U[:, :, k+1, t_avg]) ./ dz
                    V_z[:, :, k, t_avg] = (V[:, :, k-1, t_avg] .- V[:, :, k+1, t_avg]) ./ dz
                end
            end


            println("Vertical gradients calculated")


            # --- Initialize output array ---
            sp_v = zeros(Float64, nx, ny, nt)


            # --- Calculate vertical shear production for each timestep ---
            println("Calculating vertical shear production...")
            for t in 1:nt
                t_avg = min(div(t - 1, ts) + 1, nt_avg)


                U_z_t = @view U_z[:, :, :, t_avg]
                V_z_t = @view V_z[:, :, :, t_avg]
                ut    = @view fu[:, :, :, t]
                vt    = @view fv[:, :, :, t]
                wt    = @view fw[:, :, :, t]


                temp1 = wt .* ut .* U_z_t .* DRFfull
                temp2 = wt .* vt .* V_z_t .* DRFfull


                sp_v[:, :, t] = -rho0 .* dropdims(sum((temp1 .+ temp2), dims=3), dims=3)
            end


            println("Vertical shear production calculation complete")


            # --- Time average ---
            SP_V = dropdims(mean(sp_v, dims=3), dims=3)


            output_dir = joinpath(base2, "SP_V")
            open(joinpath(output_dir, "sp_v_mean_$suffix.bin"), "w") do io
                write(io, Float32.(SP_V))
            end


            println("Completed tile: $suffix")
            println("Output saved to $output_dir")
        end
    end


    println("\n=== All tiles processed successfully ===")


else
    error("Unknown time_mode '$time_mode'. Choose \"3day\", \"weekly\", or \"full\".")


end




