using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML, Impute


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
#   "3day"       -> buoyancy production averaged over each 3-day period
#   "weekly"     -> buoyancy production mean over Apr 22 00:00 - Apr 28 23:00
#   "full"       -> buoyancy production mean over full time record
#   "timeseries" -> buoyancy production saved at every timestep (nx, ny, nt)
time_mode = "3day"   # <-- change to "3day", "weekly", "full", or "timeseries"


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
delt   = 1.0
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
g     = 9.8


# --- N2 threshold ---
N2_threshold = 1.0e-8


# --- Output directories ---
mkpath(joinpath(base2, "BP_3day"))
mkpath(joinpath(base2, "BP_weekly"))
mkpath(joinpath(base2, "BP"))
mkpath(joinpath(base2, "BP_timeseries"))


# ============================================================================
# TILE FUNCTION — all allocations live here and are freed on return
# ============================================================================


function process_tile(suffix, base, base2, nx, ny, nz, nt, nt3,
                      DRF3d, idx_start, idx_end, rho0, g, N2_threshold,
                      delt, time_mode)


    # --- Read grid metrics ---
    hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
    dx    = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
    dy    = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))


    # --- Cell thicknesses ---
    DRFfull = hFacC .* DRF3d
    DRFfull[hFacC .== 0] .= 0.0


    # ---------------------------------------------------------------
    # Read full rho, compute B at every timestep (vectorized),
    # low-pass filter B, then free rho
    # ---------------------------------------------------------------
    rho = Float64.(open(joinpath(base, "Density", "rho_in_$suffix.bin"), "r") do io
        nbytes = nx * ny * nz * nt * sizeof(Float64)
        raw_bytes = read(io, nbytes)
        raw_data = reinterpret(Float64, raw_bytes)
        reshape(raw_data, nx, ny, nz, nt)
    end)


    # --- Mask rho to NaN where hFacC is zero ---
    for k in 1:nz
        mask = hFacC[:, :, k] .== 0
        rho[mask, k, :] .= NaN
    end


    # --- Compute B at every timestep — fully vectorized, no time loop ---
    B_raw = -g .* (rho .- rho0) ./ rho0    # (nx, ny, nz, nt)


    rho = nothing
    GC.gc()


    # --- Low-pass filter B (36 hr cutoff, order 4) ---
    B_2d    = reshape(permutedims(B_raw, (4,1,2,3)), nt, nx*ny*nz)
    B_raw   = nothing
    GC.gc()


    B_filt_2d = lowhighpass_butter(B_2d, 36.0, delt, 4, "low")
    B_2d      = nothing
    GC.gc()


    B_lp = permutedims(reshape(B_filt_2d, nt, nx, ny, nz), (2,3,4,1))  # (nx, ny, nz, nt)
    B_filt_2d = nothing
    GC.gc()


    # ---------------------------------------------------------------
    # Read raw N2, low-pass filter (36 hr cutoff, order 4),
    # adjust to centers, apply threshold, Impute
    # ---------------------------------------------------------------
    N2_raw = Float64.(read_bin(joinpath(base, "N2", "N2_$suffix.bin"),
                               (nx, ny, nz, nt)))

    nan_mask = isnan.(N2_raw)
    N2_raw[nan_mask] .= 0.0
    N2_2d = reshape(permutedims(N2_raw, (4,1,2,3)), nt, nx*ny*nz)
    N2_raw = nothing
    GC.gc()


    N2_filt_2d = lowhighpass_butter(N2_2d, 36.0, delt, 4, "low")
    N2_2d      = nothing
    GC.gc()


    N2_filt = permutedims(reshape(N2_filt_2d, nt, nx, ny, nz), (2,3,4,1))  # (nx, ny, nz, nt)
    N2_filt_2d = nothing
    GC.gc()
    N2_filt[nan_mask] .= NaN
    nan_mask = nothing; GC.gc()

    # --- Adjust N2 to nz+1 levels (interfaces) then average to centers ---
    N2_adjusted = zeros(Float64, nx, ny, nz+1, nt)
    N2_adjusted[:, :, 1,    :] = N2_filt[:, :, 1,      :]
    N2_adjusted[:, :, 2:nz, :] = N2_filt[:, :, 1:nz-1, :]
    N2_adjusted[:, :, nz+1, :] = N2_filt[:, :, nz-1,   :]
    N2_filt = nothing
    GC.gc()


    N2_center = zeros(Float64, nx, ny, nz, nt)
    for k in 1:nz
        N2_center[:, :, k, :] .= 0.5 .* (N2_adjusted[:, :, k,  :] .+
                                           N2_adjusted[:, :, k+1,:])
    end
    N2_adjusted = nothing
    GC.gc()


    # --- Apply threshold and Impute ---
    N2_center[N2_center .< N2_threshold] .= NaN
    for i in 1:nx, j in 1:ny, t in 1:nt
        N2_center[i, j, :, t] = Impute.interp(N2_center[i, j, :, t])
    end


    # ---------------------------------------------------------------
    # Compute B gradients — vectorized over full nt, no time loop
    # ---------------------------------------------------------------
    dx_avg = dx[2:end-1, :] .+ dx[1:end-2, :]   # (nx-2, ny)
    dy_avg = dy[:, 2:end-1] .+ dy[:, 1:end-2]   # (nx, ny-2)


    B_x = zeros(Float64, nx, ny, nz, nt)
    B_x[2:end-1, :, :, :] = (B_lp[3:end, :, :, :] .- B_lp[1:end-2, :, :, :]) ./
                              reshape(dx_avg, nx-2, ny, 1, 1)


    B_y = zeros(Float64, nx, ny, nz, nt)
    B_y[:, 2:end-1, :, :] = (B_lp[:, 3:end, :, :] .- B_lp[:, 1:end-2, :, :]) ./
                              reshape(dy_avg, nx, ny-2, 1, 1)


    B_lp   = nothing
    dx     = nothing
    dy     = nothing
    dx_avg = nothing
    dy_avg = nothing
    GC.gc()


    # --- Apply hFacC mask to B gradients ---
    for k in 1:nz, j in 2:ny-1, i in 2:nx-1
        if hFacC[i-1,j,k] != 1 || hFacC[i,j,k] != 1 || hFacC[i+1,j,k] != 1
            B_x[i, j, k, :] .= NaN
        end
        if hFacC[i,j-1,k] != 1 || hFacC[i,j,k] != 1 || hFacC[i,j+1,k] != 1
            B_y[i, j, k, :] .= NaN
        end
    end


    hFacC = nothing
    GC.gc()


    # ---------------------------------------------------------------
    # Read fluctuating buoyancy and velocities (bandpassed)
    # ---------------------------------------------------------------
    b = Float64.(open(joinpath(base2, "b", "b_t_sm_$suffix.bin"), "r") do io
        raw = read(io, nx * ny * nz * nt * sizeof(Float32))
        reshape(reinterpret(Float32, raw), nx, ny, nz, nt)
    end)


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
    # Compute depth-averaged BP at every timestep -> (nx, ny, nt)
    # All arrays full nt — direct index, no t_avg lookup
    # Float64 throughout — cast only at save
    # ---------------------------------------------------------------
    temp = (b ./ N2_center) .* (fu .* B_x .+ fv .* B_y)
    temp[isnan.(temp)] .= 0.0
    BP_ts = -rho0 .* dropdims(sum(temp .* DRFfull, dims=3), dims=3)


    b         = nothing
    fu        = nothing
    fv        = nothing
    N2_center = nothing
    B_x       = nothing
    B_y       = nothing
    DRFfull   = nothing
    temp      = nothing
    GC.gc()


    # ---------------------------------------------------------------
    # Save based on time_mode — cast to Float32 only here
    # ---------------------------------------------------------------
    if time_mode == "timeseries"
        open(joinpath(base2, "BP_timeseries", "bp_ts_$suffix.bin"), "w") do io
            write(io, Float32.(BP_ts))
        end
        println("  Saved: bp_ts_$suffix.bin  shape=(nx, ny, nt)")
        BP_ts = nothing
        GC.gc()


    elseif time_mode == "3day"
        hrs_per_chunk = 3 * 24
        BP_3day = zeros(Float64, nx, ny, nt3)
        for t in 1:nt3
            t_start = (t-1) * hrs_per_chunk + 1
            t_end   = min(t * hrs_per_chunk, nt)
            BP_3day[:, :, t] = mean(BP_ts[:, :, t_start:t_end], dims=3)
        end
        BP_ts = nothing
        GC.gc()
        println("  BP_3day range: ", extrema(filter(isfinite, BP_3day)))
        open(joinpath(base2, "BP_3day", "bp_3day_$suffix.bin"), "w") do io
            write(io, Float32.(BP_3day))
        end
        println("  Saved: bp_3day_$suffix.bin  shape=(nx, ny, nt3)")
        BP_3day = nothing
        GC.gc()


    elseif time_mode == "weekly"
        bp_weekly = dropdims(
            mean(BP_ts[:, :, idx_start:idx_end], dims=3), dims=3)
        BP_ts = nothing
        GC.gc()
        println("  bp_weekly range: ", extrema(filter(isfinite, bp_weekly)))
        open(joinpath(base2, "BP_weekly", "bp_weekly_$suffix.bin"), "w") do io
            write(io, Float32.(bp_weekly))
        end
        println("  Saved: bp_weekly_$suffix.bin  shape=(nx, ny)")
        bp_weekly = nothing
        GC.gc()


    elseif time_mode == "full"
        bp_mean = dropdims(mean(BP_ts, dims=3), dims=3)
        BP_ts = nothing
        GC.gc()
        println("  bp_mean range: ", extrema(filter(isfinite, bp_mean)))
        open(joinpath(base2, "BP", "bp_mean_$suffix.bin"), "w") do io
            write(io, Float32.(bp_mean))
        end
        println("  Saved: bp_mean_$suffix.bin  shape=(nx, ny)")
        bp_mean = nothing
        GC.gc()


    else
        error("Unknown time_mode '$time_mode'. Choose \"3day\", \"weekly\", \"full\", or \"timeseries\".")
    end


end


# ============================================================================
# MAIN LOOP — just calls the function per tile
# ============================================================================


println("=== Starting buoyancy production | mode: $time_mode ===")


for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
        println("Processing tile: $suffix")
        process_tile(suffix, base, base2, nx, ny, nz, nt, nt3,
                     DRF3d, idx_start, idx_end, rho0, g, N2_threshold,
                     delt, time_mode)
        GC.gc()
        println("Completed tile: $suffix")
    end
end


println("\n=== All tiles processed successfully ($time_mode) ===")




