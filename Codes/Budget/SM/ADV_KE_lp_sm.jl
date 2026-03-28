using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML


include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin


config_file = get(ENV, "JULIA_CONFIG",
            joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)


base  = cfg["base_path"]
base2 = cfg["base_path2"]


# --- TIME MODE CONFIGURATION ---
# Options:
#   "3day"       -> depth-averaged KE flux averaged over each 3-day period
#   "weekly"     -> depth-averaged KE flux mean over Apr 22 00:00 - Apr 28 23:00
#   "full"       -> depth-averaged KE flux mean over full time record
#   "timeseries" -> depth-averaged KE flux saved at every timestep (nx, ny, nt)
time_mode = "full"   # <-- change to "3day", "weekly", "full", or "timeseries"


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
dto    = 144
Tts    = 366192
nt     = div(Tts, dto)
ts     = 72
nt_avg = div(nt, ts)
nt3    = div(nt, 3*24)


# -------------------------------------------------------------------------
# Weekly window: April 22 00:00:00 to April 28 23:00:00, 2012
#   Time series starts 2012-03-01T00:00:00, delta_t = 1 hour
#   March = 31 days = 744 hours
#   Apr 22 00:00 = hour 744 + (22-1)*24 = 1248  -> index 1249 (1-based)
#   Apr 28 23:00 = hour 744 +  28 *24-1 = 1415  -> index 1416 (1-based)
#   nt_week = 1416 - 1249 + 1 = 168  (7 days x 24 hrs)
# -------------------------------------------------------------------------
hour_apr22_start = 31*24 + (22-1)*24
hour_apr28_end   = 31*24 +  28   *24 - 1
idx_start        = hour_apr22_start + 1
idx_end          = hour_apr28_end   + 1
nt_week          = idx_end - idx_start + 1


# --- Thickness & constants ---
thk   = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF   = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)
rho0  = 999.8


# --- Output directories ---
mkpath(joinpath(base2, "U_KE_3day"))
mkpath(joinpath(base2, "U_KE_weekly"))
mkpath(joinpath(base2, "U_KE"))
mkpath(joinpath(base2, "U_KE_timeseries"))


# ============================================================================
# MAIN WORKFLOW
# ============================================================================


for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
        println("\n--- Processing tile: $suffix ($time_mode) ---")


        # --- Read grid metrics ---
        hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
        dx    = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
        dy    = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))


        # --- Read low-pass filtered U, V ---
        u_lp = Float64.(open(joinpath(base2, "UVW_LP", "u_lp_$suffix.bin"), "r") do io
            nbytes = nx * ny * nz * nt * sizeof(Float32)
            raw_bytes = read(io, nbytes)
            raw_data = reinterpret(Float32, raw_bytes)
            reshape(raw_data, nx, ny, nz, nt)
        end)


        v_lp = Float64.(open(joinpath(base2, "UVW_LP", "v_lp_$suffix.bin"), "r") do io
            nbytes = nx * ny * nz * nt * sizeof(Float32)
            raw_bytes = read(io, nbytes)
            raw_data = reinterpret(Float32, raw_bytes)
            reshape(raw_data, nx, ny, nz, nt)
        end)


        # --- Read kinetic energy ---
        ke_t = Float64.(open(joinpath(base2, "KE", "ke_t_sm_$suffix.bin"), "r") do io
            nbytes = nx * ny * nz * nt * sizeof(Float32)
            raw_bytes = read(io, nbytes)
            raw_data = reinterpret(Float32, raw_bytes)
            reshape(raw_data, nx, ny, nz, nt)
        end)


        # --- Cell thicknesses ---
        DRFfull = hFacC .* DRF3d
        DRFfull[hFacC .== 0] .= 0.0


        # --- KE gradients ---
        ke_x = zeros(Float64, nx, ny, nz, nt)
        ke_y = zeros(Float64, nx, ny, nz, nt)


        dx_avg = dx[2:end-1, :] .+ dx[1:end-2, :]
        ke_x[2:end-1, :, :, :] = (ke_t[3:end, :, :, :] .- ke_t[1:end-2, :, :, :]) ./
                                  reshape(dx_avg, nx-2, ny, 1, 1)


        dy_avg = dy[:, 2:end-1] .+ dy[:, 1:end-2]
        ke_y[:, 2:end-1, :, :] = (ke_t[:, 3:end, :, :] .- ke_t[:, 1:end-2, :, :]) ./
                                  reshape(dy_avg, nx, ny-2, 1, 1)




        # ---------------------------------------------------------------
        # Compute depth-averaged U_KE at every timestep -> (nx, ny, nt)
        # u_lp, v_lp, ke_x, ke_y all (nx, ny, nz, nt) Float64
        # DRFfull (nx, ny, nz) broadcasts over nt automatically
        # Kept in Float64 throughout — cast only at save
        # ---------------------------------------------------------------
        U_KE_ts = dropdims(
            sum((u_lp .* ke_x .+ v_lp .* ke_y) .* DRFfull, dims=3), dims=3)


        # ---------------------------------------------------------------
        # Save based on time_mode — cast to Float32 only here
        # ---------------------------------------------------------------
        if time_mode == "timeseries"
            # --- Save full time series (nx, ny, nt) ---
            open(joinpath(base2, "U_KE_timeseries", "u_ke_ts_$suffix.bin"), "w") do io
                write(io, Float32.(U_KE_ts))
            end
            println("  Saved: u_ke_ts_$suffix.bin  shape=(nx, ny, nt)")


        elseif time_mode == "3day"
            # --- Average into nt3 3-day chunks -> (nx, ny, nt3) ---
            hrs_per_chunk = 3 * 24
            U_KE_3day = zeros(Float64, nx, ny, nt3)
            for t in 1:nt3
                t_start = (t-1) * hrs_per_chunk + 1
                t_end   = min(t * hrs_per_chunk, nt)
                U_KE_3day[:, :, t] = mean(U_KE_ts[:, :, t_start:t_end], dims=3)
            end
            println("  U_KE_3day range: ", extrema(filter(isfinite, U_KE_3day)))
            open(joinpath(base2, "U_KE_3day", "u_ke_3day_$suffix.bin"), "w") do io
                write(io, Float32.(U_KE_3day))
            end
            println("  Saved: u_ke_3day_$suffix.bin  shape=(nx, ny, nt3)")


        elseif time_mode == "weekly"
            # --- Subset to weekly window then time-average -> (nx, ny) ---
            u_ke_weekly = dropdims(
                mean(U_KE_ts[:, :, idx_start:idx_end], dims=3), dims=3)
            println("  u_ke_weekly range: ", extrema(filter(isfinite, u_ke_weekly)))
            open(joinpath(base2, "U_KE_weekly", "u_ke_weekly_$suffix.bin"), "w") do io
                write(io, Float32.(u_ke_weekly))
            end
            println("  Saved: u_ke_weekly_$suffix.bin  shape=(nx, ny)")


        elseif time_mode == "full"
            # --- Time-average over full record -> (nx, ny) ---
            u_ke_mean = dropdims(mean(U_KE_ts, dims=3), dims=3)
            println("  u_ke_mean range: ", extrema(filter(isfinite, u_ke_mean)))
            open(joinpath(base2, "U_KE", "u_ke_mean_$suffix.bin"), "w") do io
                write(io, Float32.(u_ke_mean))
            end
            println("  Saved: u_ke_mean_$suffix.bin  shape=(nx, ny)")


        else
            error("Unknown time_mode '$time_mode'. Choose \"3day\", \"weekly\", \"full\", or \"timeseries\".")
        end


        println("Completed tile: $suffix")
    end
end


println("\n=== All tiles processed successfully ($time_mode) ===")




