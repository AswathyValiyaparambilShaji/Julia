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
#   "3day"       -> depth-averaged PE flux averaged over each 3-day period
#   "weekly"     -> depth-averaged PE flux mean over Apr 22 00:00 - Apr 28 23:00
#   "full"       -> depth-averaged PE flux mean over full time record
#   "timeseries" -> depth-averaged PE flux saved at every timestep (nx, ny, nt)
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
dt = 1        # hourly output, dt = 1 hr
dto = 144
Tts = 366192
nt  = div(Tts, dto)
ts  = 72
nt_avg = div(nt, ts)
nt3 = div(nt, 3*24)


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


# --- N2 threshold ---
N2_threshold = 1.0e-8


# --- Output directories ---
mkpath(joinpath(base2, "U_PE_3day"))
mkpath(joinpath(base2, "U_PE_weekly"))
mkpath(joinpath(base2, "U_PE"))
mkpath(joinpath(base2, "U_PE_timeseries"))


# ============================================================================
# MAIN WORKFLOW
# ============================================================================


println("=== Starting advective PE flux | mode: $time_mode ===")


for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
        println("Processing tile: $suffix")


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


        # --- Read PE (full temporal resolution) ---
        pe = Float64.(open(joinpath(base2, "pe", "pe_tn_sm_$suffix.bin"), "r") do io
            nbytes = nx * ny * nz * nt * sizeof(Float32)
            raw_bytes = read(io, nbytes)
            raw_data = reinterpret(Float32, raw_bytes)
            reshape(raw_data, nx, ny, nz, nt)
        end)


        # --- Read raw N2, low-pass filter (36 hr cutoff, order 4) ---
        N2_raw = Float64.(read_bin(joinpath(base, "N2", "N2_$suffix.bin"),
                                   (nx, ny, nz, nt)))


        N2_2d = permutedims(N2_raw, (4, 1, 2, 3))
        N2_2d = reshape(N2_2d, nt, nx*ny*nz)


        N2_filt_2d = lowhighpass_butter(N2_2d, 36.0, dt, 4, "low")


        N2_filt = reshape(N2_filt_2d, nt, nx, ny, nz)
        N2_filt = permutedims(N2_filt, (2, 3, 4, 1))


        # --- Adjust N2 to nz+1 levels (interfaces) then average to centers ---
        N2_adjusted = zeros(Float64, nx, ny, nz+1, nt)
        N2_adjusted[:, :, 1,    :] = N2_filt[:, :, 1,    :]
        N2_adjusted[:, :, 2:nz, :] = N2_filt[:, :, 1:nz-1, :]
        N2_adjusted[:, :, nz+1, :] = N2_filt[:, :, nz-1, :]


        N2_center = zeros(Float64, nx, ny, nz, nt)
        for k in 1:nz
            N2_center[:, :, k, :] .=
                0.5 .* (N2_adjusted[:, :, k,   :] .+
                        N2_adjusted[:, :, k+1,  :])
        end


        # --- Apply physical threshold ---
        N2_center[N2_center .< N2_threshold] .= N2_threshold


        # --- Cell thicknesses ---
        DRFfull = hFacC .* DRF3d
        DRFfull[hFacC .== 0] .= 0.0


        # --- PE gradients ---
        pe_x = zeros(Float64, nx, ny, nz, nt)
        pe_y = zeros(Float64, nx, ny, nz, nt)


        dx_avg = dx[2:end-1, :] .+ dx[1:end-2, :]
        pe_x[2:end-1, :, :, :] = (pe[3:end, :, :, :] .- pe[1:end-2, :, :, :]) ./
                                  reshape(dx_avg, nx-2, ny, 1, 1)


        dy_avg = dy[:, 2:end-1] .+ dy[:, 1:end-2]
        pe_y[:, 2:end-1, :, :] = (pe[:, 3:end, :, :] .- pe[:, 1:end-2, :, :]) ./
                                  reshape(dy_avg, nx, ny-2, 1, 1)


        # --- Depth-averaged advective PE flux -> (nx, ny, nt) ---
        temp = (u_lp .* pe_x .+ v_lp .* pe_y) ./ N2_center
        temp[isnan.(temp)] .= 0.0
        U_PE_ts = rho0 .* dropdims(sum(temp .* DRFfull, dims=3), dims=3)


        # --- Save based on time_mode — cast to Float32 only here ---
        if time_mode == "timeseries"
            open(joinpath(base2, "U_PE_timeseries", "u_pe_ts_$suffix.bin"), "w") do io
                write(io, Float32.(U_PE_ts))
            end


        elseif time_mode == "3day"
            hrs_per_chunk = 3 * 24
            U_PE_3day = zeros(Float64, nx, ny, nt3)
            for t in 1:nt3
                t_start = (t-1) * hrs_per_chunk + 1
                t_end   = min(t * hrs_per_chunk, nt)
                U_PE_3day[:, :, t] = mean(U_PE_ts[:, :, t_start:t_end], dims=3)
            end
            open(joinpath(base2, "U_PE_3day", "u_pe_3day_$suffix.bin"), "w") do io
                write(io, Float32.(U_PE_3day))
            end


        elseif time_mode == "weekly"
            u_pe_weekly = dropdims(
                mean(U_PE_ts[:, :, idx_start:idx_end], dims=3), dims=3)
            open(joinpath(base2, "U_PE_weekly", "u_pe_weekly_$suffix.bin"), "w") do io
                write(io, Float32.(u_pe_weekly))
            end


        elseif time_mode == "full"
            u_pe_mean = dropdims(mean(U_PE_ts, dims=3), dims=3)
            open(joinpath(base2, "U_PE", "u_pe_mean_$suffix.bin"), "w") do io
                write(io, Float32.(u_pe_mean))
            end


        else
            error("Unknown time_mode '$time_mode'. Choose \"3day\", \"weekly\", \"full\", or \"timeseries\".")
        end


        println("  Completed: $suffix")
    end
end


println("\n=== All tiles processed successfully ($time_mode) ===")




