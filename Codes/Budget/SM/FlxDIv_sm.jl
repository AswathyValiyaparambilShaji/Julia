using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML
#using CairoMakie, SparseArrays


include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter


config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base  = cfg["base_path"]
base2 = cfg["base_path2"]




# --- TIME AVERAGING CONFIGURATION ---
# Options:
#   "3day"   -> divergence for each 3-day averaged bin over the full record
#   "full"   -> divergence of the mean over all nt snapshots
#   "weekly" -> divergence of the mean over Apr 22-28 snapshots only
time_mode = "weekly"   # <-- change to "3day", "full", or "weekly"




# --- Domain & grid ---
NX, NY = 288, 468
minlat, maxlat = 24.0, 31.91
minlon, maxlon = 193.0, 199.0
lat = range(minlat, maxlat, length=NY)
lon = range(minlon, maxlon, length=NX)




# --- Tile & time ---
buf  = 3
tx, ty = 47, 66
nx = tx + 2*buf
ny = ty + 2*buf
nz = 88


kz  = 1
dt  = 25
dto = 144                    # model output interval in hours
Tts = 366192                 # total time steps (hours)
nt  = div(Tts, dto)          # total number of snapshots in raw flux file
nt3 = div(nt, 3*24)          # number of 3-day bins




# -------------------------------------------------------------------------
# Weekly window: April 22-28 2012
#   Series starts 2012-03-01T00:00:00, one snapshot every dto=144 hours
#   March = 31 days = 744 h  ->  April 1 starts at snapshot index:
#     div(744, dto) + 1 = div(744,144) + 1 = 6
#   April 22 starts at hour  744 + (22-1)*24 = 1248
#   April 28 ends   at hour  744 +  28  *24  = 1416
#
#   Raw snapshot index (1-based, each snapshot = dto=144 h):
#     Apr 22 -> div(1248, 144) + 1 = 9
#     Apr 28 -> div(1416, 144)     = 9   (snapshot 9 covers hours 1152-1295)
#              div(1416, 144) + 1  = 10  (snapshot 10 covers hours 1296-1439)
#   -> use snapshots 9 and 10 to cover the full Apr 22-28 window
# -------------------------------------------------------------------------
hour_apr22     = 31*24 + (22-1)*24           # = 1248
hour_apr28     = 31*24 +  28   *24           # = 1416
idx_week_start = div(hour_apr22, dto) + 1    # = 9
idx_week_end   = div(hour_apr28, dto) + 1    # = 10
idx_week_range = idx_week_start:idx_week_end
n_week         = length(idx_week_range)


@printf("Weekly window: Apr 22-28  ->  raw snapshot indices %d:%d  (%d snapshots x %d h = %d h)\n",
        first(idx_week_range), last(idx_week_range), n_week, dto, n_week*dto)




# --- Thickness & constants ---
thk   = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF   = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)
g = 9.8




# --- Bandpass filter (9-15 day band, 1-step sampling) ---
T1, T2, delt, N = 9.0, 15.0, 1.0, 4
fcutlow, fcuthigh = 1/T2, 1/T1
fnq = 1/delt
bpf = digitalfilter(Bandpass(fcutlow, fcuthigh), Butterworth(N); fs=fnq)




# ============================================================================
# MAIN WORKFLOW
# All three modes read the same raw flux file (nx x ny x nz x nt).
# Averaging over the time dimension is done here in the script.
# ============================================================================


if time_mode == "3day"
    println("Computing flux divergence for $nt3 3-day bins")
    mkpath(joinpath(base2, "FDiv_3day"))
elseif time_mode == "full"
    println("Computing flux divergence for full-record time mean ($nt snapshots)")
    mkpath(joinpath(base2, "FDiv"))
elseif time_mode == "weekly"
    println("Computing flux divergence for weekly mean Apr 22-28 " *
            "(raw snapshot indices $(first(idx_week_range)):$(last(idx_week_range)))")
    mkpath(joinpath(base2, "FDiv_weekly"))
else
    error("Unknown time_mode '$time_mode'. Choose \"3day\", \"full\", or \"weekly\".")
end




for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)


        # --- Read grid & mask ---
        hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
        dx    = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"),     (nx, ny))
        dy    = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"),     (nx, ny))


        DRFfull = hFacC .* DRF3d
        DRFfull[hFacC .== 0] .= 0.0


        # --- Read raw 4D flux file (nx x ny x nz x nt) -- same for all modes ---
        fx = Float64.(open(joinpath(base2, "xflux", "xflx_$suffix.bin"), "r") do io
            raw = reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32)))
            reshape(raw, nx, ny, nz, nt)
        end)


        fy = Float64.(open(joinpath(base2, "yflux", "yflx_$suffix.bin"), "r") do io
            raw = reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32)))
            reshape(raw, nx, ny, nz, nt)
        end)


        # --- Depth-integrate all snapshots -> (nx, ny, 1, nt) ---
        fxX = sum(fx .* DRFfull, dims=3)
        fyY = sum(fy .* DRFfull, dims=3)


        suffix2 = @sprintf("%02dx%02d_%d", xn, yn, buf-2)


        if time_mode == "3day"
            # -----------------------------------------------------------------
            # Average into nt3 bins of 3 days (3*24/dto snapshots per bin)
            # Output: (nx-2, ny-2, nt3)
            # -----------------------------------------------------------------
            nper = div(3*24, dto)   # number of raw snapshots per 3-day bin
            flxD = zeros(nx-2, ny-2, nt3)


            for t in 1:nt3
                t1 = (t-1)*nper + 1
                t2 = t*nper
                fxX_bin = mean(fxX[:,:,:,t1:t2], dims=4)  # (nx, ny, 1, 1)
                fyY_bin = mean(fyY[:,:,:,t1:t2], dims=4)


                for i in 2:(nx-2), j in 2:(ny-2)
                    dF_x_dx = (fxX_bin[i+1,j,1,1] - fxX_bin[i-1,j,1,1]) / (dx[i,j] + dx[i-1,j])
                    dF_y_dy = (fyY_bin[i,j+1,1,1] - fyY_bin[i,j-1,1,1]) / (dy[i,j] + dy[i,j-1])
                    flxD[i,j,t] = dF_x_dx + dF_y_dy
                end
            end


            open(joinpath(base2, "FDiv_3day", "FDiv_3day_$suffix2.bin"), "w") do io
                write(io, Float32.(flxD))
            end


        elseif time_mode == "full"
            # -----------------------------------------------------------------
            # Mean over all nt snapshots -> output is (nx-2, ny-2)
            # -----------------------------------------------------------------
            fxX_mean = mean(fxX, dims=4)   # (nx, ny, 1, 1)
            fyY_mean = mean(fyY, dims=4)


            flxD = zeros(nx-2, ny-2)


            for i in 2:(nx-2), j in 2:(ny-2)
                dF_x_dx = (fxX_mean[i+1,j,1,1] - fxX_mean[i-1,j,1,1]) / (dx[i,j] + dx[i-1,j])
                dF_y_dy = (fyY_mean[i,j+1,1,1] - fyY_mean[i,j-1,1,1]) / (dy[i,j] + dy[i,j-1])
                flxD[i,j] = dF_x_dx + dF_y_dy
            end


            open(joinpath(base2, "FDiv", "FDiv_$suffix2.bin"), "w") do io
                write(io, Float32.(flxD))
            end


        elseif time_mode == "weekly"
            # -----------------------------------------------------------------
            # Mean over Apr 22-28 snapshots only -> output is (nx-2, ny-2)
            # -----------------------------------------------------------------
            fxX_mean = mean(fxX[:,:,:,idx_week_range], dims=4)   # (nx, ny, 1, 1)
            fyY_mean = mean(fyY[:,:,:,idx_week_range], dims=4)


            flxD = zeros(nx-2, ny-2)


            for i in 2:(nx-2), j in 2:(ny-2)
                dF_x_dx = (fxX_mean[i+1,j,1,1] - fxX_mean[i-1,j,1,1]) / (dx[i,j] + dx[i-1,j])
                dF_y_dy = (fyY_mean[i,j+1,1,1] - fyY_mean[i,j-1,1,1]) / (dy[i,j] + dy[i,j-1])
                flxD[i,j] = dF_x_dx + dF_y_dy
            end


            open(joinpath(base2, "FDiv_weekly", "FDiv_weekly_$suffix2.bin"), "w") do io
                write(io, Float32.(flxD))
            end
        end


        println("  Completed tile: $suffix ($time_mode)")
    end
end


println("Completed flux divergence -- mode: $time_mode")




