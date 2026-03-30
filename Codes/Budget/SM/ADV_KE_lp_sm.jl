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


println("=== Starting advective KE flux | mode: $time_mode ===")


for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
        println("Processing tile: $suffix")


        # --- Read grid metrics ---
        hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
        dx    = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
        dy    = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))


        # --- Cell thicknesses ---
        DRFfull = hFacC .* DRF3d          # (nx, ny, nz)
        DRFfull[hFacC .== 0] .= 0.0
        hFacC = nothing
        GC.gc()


        # --- Pre-compute dx_avg, dy_avg as Float64 ---
        dx_avg = Float64.(dx[2:end-1, :] .+ dx[1:end-2, :])   # (nx-2, ny)
        dy_avg = Float64.(dy[:, 2:end-1] .+ dy[:, 1:end-2])   # (nx, ny-2)
        dx = nothing
        dy = nothing
        GC.gc()


        # --- Read low-pass filtered U, V as Float64 ---
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


        # --- Read kinetic energy as Float64 ---
        ke_t = Float64.(open(joinpath(base2, "KE", "ke_t_sm_$suffix.bin"), "r") do io
            nbytes = nx * ny * nz * nt * sizeof(Float32)
            raw_bytes = read(io, nbytes)
            raw_data = reinterpret(Float32, raw_bytes)
            reshape(raw_data, nx, ny, nz, nt)
        end)


        # ---------------------------------------------------------------
        # Accumulate depth-integrated U_KE level by level -> (nx, ny, nt)
        # This avoids allocating full (nx, ny, nz, nt) ke_x and ke_y arrays.
        # At each level k only (nx, ny, nt) slices are in memory.
        # Peak simultaneous large arrays: u_lp + v_lp + ke_t + U_KE_ts
        # ---------------------------------------------------------------
        U_KE_ts = zeros(Float64, nx, ny, nt)


        for k in 1:nz
            drf_k = DRFfull[:, :, k]           # (nx, ny)


            # x-gradient at level k
            ke_x_k = zeros(Float64, nx, ny, nt)
            ke_x_k[2:end-1, :, :] = (ke_t[3:end, :, k, :] .- ke_t[1:end-2, :, k, :]) ./
                                     reshape(dx_avg, nx-2, ny, 1)


            # y-gradient at level k
            ke_y_k = zeros(Float64, nx, ny, nt)
            ke_y_k[:, 2:end-1, :] = (ke_t[:, 3:end, k, :] .- ke_t[:, 1:end-2, k, :]) ./
                                     reshape(dy_avg, nx, ny-2, 1)


            # Accumulate depth-integrated flux
            U_KE_ts .+= (u_lp[:, :, k, :] .* ke_x_k .+
                         v_lp[:, :, k, :] .* ke_y_k) .* reshape(drf_k, nx, ny, 1)
        end


        # --- Free all large arrays ---
        u_lp    = nothing
        v_lp    = nothing
        ke_t    = nothing
        DRFfull = nothing
        dx_avg  = nothing
        dy_avg  = nothing
        GC.gc()


        # ---------------------------------------------------------------
        # Save based on time_mode — cast to Float32 only here
        # ---------------------------------------------------------------
        if time_mode == "timeseries"
            # --- Save full time series (nx, ny, nt) ---
            open(joinpath(base2, "U_KE_timeseries", "u_ke_ts_$suffix.bin"), "w") do io
                write(io, Float32.(U_KE_ts))
            end
            println("  Saved: u_ke_ts_$suffix.bin  shape=(nx, ny, nt)")
            U_KE_ts = nothing
            GC.gc()


        elseif time_mode == "3day"
            # --- Average into nt3 3-day chunks -> (nx, ny, nt3) ---
            hrs_per_chunk = 3 * 24
            U_KE_3day = zeros(Float64, nx, ny, nt3)
            for t in 1:nt3
                t_start = (t-1) * hrs_per_chunk + 1
                t_end   = min(t * hrs_per_chunk, nt)
                U_KE_3day[:, :, t] = mean(U_KE_ts[:, :, t_start:t_end], dims=3)
            end
            U_KE_ts = nothing
            GC.gc()
            println("  U_KE_3day range: ", extrema(filter(isfinite, U_KE_3day)))
            open(joinpath(base2, "U_KE_3day", "u_ke_3day_$suffix.bin"), "w") do io
                write(io, Float32.(U_KE_3day))
            end
            println("  Saved: u_ke_3day_$suffix.bin  shape=(nx, ny, nt3)")
            U_KE_3day = nothing
            GC.gc()


        elseif time_mode == "weekly"
            # --- Subset to weekly window then time-average -> (nx, ny) ---
            u_ke_weekly = dropdims(
                mean(U_KE_ts[:, :, idx_start:idx_end], dims=3), dims=3)
            U_KE_ts = nothing
            GC.gc()
            println("  u_ke_weekly range: ", extrema(filter(isfinite, u_ke_weekly)))
            open(joinpath(base2, "U_KE_weekly", "u_ke_weekly_$suffix.bin"), "w") do io
                write(io, Float32.(u_ke_weekly))
            end
            println("  Saved: u_ke_weekly_$suffix.bin  shape=(nx, ny)")
            u_ke_weekly = nothing
            GC.gc()


        elseif time_mode == "full"
            # --- Time-average over full record -> (nx, ny) ---
            u_ke_mean = dropdims(mean(U_KE_ts, dims=3), dims=3)
            U_KE_ts = nothing
            GC.gc()
            println("  u_ke_mean range: ", extrema(filter(isfinite, u_ke_mean)))
            open(joinpath(base2, "U_KE", "u_ke_mean_$suffix.bin"), "w") do io
                write(io, Float32.(u_ke_mean))
            end
            println("  Saved: u_ke_mean_$suffix.bin  shape=(nx, ny)")
            u_ke_mean = nothing
            GC.gc()


        else
            error("Unknown time_mode '$time_mode'. Choose \"3day\", \"weekly\", \"full\", or \"timeseries\".")
        end


        println("Completed tile: $suffix")
    end
end


println("\n=== All tiles processed successfully ($time_mode) ===")




