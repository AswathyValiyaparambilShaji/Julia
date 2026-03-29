using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML


include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "functions", "butter_filters.jl"))
using .FluxUtils: read_bin, bandpassfilter


config_file = get(ENV, "JULIA_CONFIG",
            joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)


base  = cfg["base_path"]
base2 = cfg["base_path2"]


# --- TIME MODE CONFIGURATION ---
# Options:
#   "3day"       -> horizontal shear production averaged over each 3-day period
#   "weekly"     -> horizontal shear production mean over Apr 22 00:00 - Apr 28 23:00
#   "full"       -> horizontal shear production mean over full time record
#   "timeseries" -> horizontal shear production saved at every timestep (nx, ny, nt)
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
dt     = 25
delt   = 1.0    # hourly sampling [hr] — used for filtering
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
#   Apr 22 00:00 = hour 744 + (22-1)*24 = 1248  -> index 1248 + 1 = 1249
#   Apr 28 23:00 = hour 744 +  28 *24-1 = 1415  -> index 1415 + 1 = 1416
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
mkpath(joinpath(base2, "SP_H_3day"))
mkpath(joinpath(base2, "SP_H_weekly"))
mkpath(joinpath(base2, "SP_H"))
mkpath(joinpath(base2, "SP_H_timeseries"))


# ============================================================================
# MAIN WORKFLOW
# ============================================================================


println("=== Starting horizontal shear production | mode: $time_mode ===")


for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
        println("Processing tile: $suffix")


        # --- Read grid metrics ---
        hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
        dx    = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
        dy    = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))


        # --- Cell thicknesses ---
        DRFfull = hFacC .* DRF3d
        DRFfull[hFacC .== 0] .= 0.0
        hFacC = nothing
        GC.gc()


        # ---------------------------------------------------------------
        # Read low-pass filtered U, V (full nt resolution)
        # ---------------------------------------------------------------
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


        # ---------------------------------------------------------------
        # Compute horizontal gradients of low-pass U, V — vectorized
        # over full nt, no time loop
        # ---------------------------------------------------------------
        dx_avg = dx[2:end-1, :] .+ dx[1:end-2, :]   # (nx-2, ny)
        dy_avg = dy[:, 2:end-1] .+ dy[:, 1:end-2]   # (nx, ny-2)


        U_x = zeros(Float64, nx, ny, nz, nt)
        U_x[2:end-1, :, :, :] = (u_lp[3:end, :, :, :] .- u_lp[1:end-2, :, :, :]) ./
                                  reshape(dx_avg, nx-2, ny, 1, 1)


        U_y = zeros(Float64, nx, ny, nz, nt)
        U_y[:, 2:end-1, :, :] = (u_lp[:, 3:end, :, :] .- u_lp[:, 1:end-2, :, :]) ./
                                  reshape(dy_avg, nx, ny-2, 1, 1)


        u_lp = nothing
        GC.gc()


        V_x = zeros(Float64, nx, ny, nz, nt)
        V_x[2:end-1, :, :, :] = (v_lp[3:end, :, :, :] .- v_lp[1:end-2, :, :, :]) ./
                                  reshape(dx_avg, nx-2, ny, 1, 1)


        V_y = zeros(Float64, nx, ny, nz, nt)
        V_y[:, 2:end-1, :, :] = (v_lp[:, 3:end, :, :] .- v_lp[:, 1:end-2, :, :]) ./
                                  reshape(dy_avg, nx, ny-2, 1, 1)


        v_lp   = nothing
        dx     = nothing
        dy     = nothing
        dx_avg = nothing
        dy_avg = nothing
        GC.gc()


        # ---------------------------------------------------------------
        # Read fluctuating velocities
        # ---------------------------------------------------------------
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


        # ---------------------------------------------------------------
        # Compute depth-averaged SP_H at every timestep -> (nx, ny, nt)
        # All arrays full nt — direct index, no t_avg lookup
        # Float64 throughout — cast only at save
        # ---------------------------------------------------------------
        SP_H_ts = -rho0 .* dropdims(
            sum((fu .* fu .* U_x .+
                 fu .* fv .* U_y .+
                 fu .* fv .* V_x .+
                 fv .* fv .* V_y) .* DRFfull, dims=3), dims=3)


        fu      = nothing
        fv      = nothing
        U_x     = nothing
        U_y     = nothing
        V_x     = nothing
        V_y     = nothing
        DRFfull = nothing
        GC.gc()


        # ---------------------------------------------------------------
        # Save based on time_mode — cast to Float32 only here
        # ---------------------------------------------------------------
        if time_mode == "timeseries"
            open(joinpath(base2, "SP_H_timeseries", "sp_h_ts_$suffix.bin"), "w") do io
                write(io, Float32.(SP_H_ts))
            end
            println("  Saved: sp_h_ts_$suffix.bin  shape=(nx, ny, nt)")
            SP_H_ts = nothing
            GC.gc()


        elseif time_mode == "3day"
            hrs_per_chunk = 3 * 24
            SP_H_3day = zeros(Float64, nx, ny, nt3)
            for t in 1:nt3
                t_start = (t-1) * hrs_per_chunk + 1
                t_end   = min(t * hrs_per_chunk, nt)
                SP_H_3day[:, :, t] = mean(SP_H_ts[:, :, t_start:t_end], dims=3)
            end
            SP_H_ts = nothing
            GC.gc()
            println("  SP_H_3day range: ", extrema(filter(isfinite, SP_H_3day)))
            open(joinpath(base2, "SP_H_3day", "sp_h_3day_$suffix.bin"), "w") do io
                write(io, Float32.(SP_H_3day))
            end
            println("  Saved: sp_h_3day_$suffix.bin  shape=(nx, ny, nt3)")
            SP_H_3day = nothing
            GC.gc()


        elseif time_mode == "weekly"
            sp_h_weekly = dropdims(
                mean(SP_H_ts[:, :, idx_start:idx_end], dims=3), dims=3)
            SP_H_ts = nothing
            GC.gc()
            println("  sp_h_weekly range: ", extrema(filter(isfinite, sp_h_weekly)))
            open(joinpath(base2, "SP_H_weekly", "sp_h_weekly_$suffix.bin"), "w") do io
                write(io, Float32.(sp_h_weekly))
            end
            println("  Saved: sp_h_weekly_$suffix.bin  shape=(nx, ny)")
            sp_h_weekly = nothing
            GC.gc()


        elseif time_mode == "full"
            sp_h_mean = dropdims(mean(SP_H_ts, dims=3), dims=3)
            SP_H_ts = nothing
            GC.gc()
            println("  sp_h_mean range: ", extrema(filter(isfinite, sp_h_mean)))
            open(joinpath(base2, "SP_H", "sp_h_mean_$suffix.bin"), "w") do io
                write(io, Float32.(sp_h_mean))
            end
            println("  Saved: sp_h_mean_$suffix.bin  shape=(nx, ny)")
            sp_h_mean = nothing
            GC.gc()


        else
            error("Unknown time_mode '$time_mode'. Choose \"3day\", \"weekly\", \"full\", or \"timeseries\".")
        end


        println("Completed tile: $suffix")
    end
end


println("\n=== All tiles processed successfully ($time_mode) ===")




