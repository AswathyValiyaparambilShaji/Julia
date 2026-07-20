using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML
#using CairoMakie, SparseArrays


include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base  = cfg["base_path"]
base2 = cfg["base_path2"]


# --- Domain & grid ---
NX, NY = 288, 468
minlat, maxlat = 24.0, 31.91
minlon, maxlon = 193.0, 199.0
lat = range(minlat, maxlat, length=NY)
lon = range(minlon, maxlon, length=NX)


# --- Tile & time ---
buf = 3
tx, ty = 47, 66
nx = tx + 2*buf
ny = ty + 2*buf
nz = 88


kz  = 1
dt  = 25
dto = 144
Tts = 366192
nt  = div(Tts, dto)
nt3 = div(nt, 3*24)   # number of 3-day periods
hrs_per_chunk = 3 * 24

nt_chunk = 72
n_chunks = div(nt, nt_chunk)
ring_steps = nt_chunk
t_safe_start = ring_steps + 1              # first valid step (1801)
t_safe_end   = nt - ring_steps             # last  valid step (nt-1800)


# Safe 3-day chunks: only keep chunks that fall entirely within the safe range
safe_chunks = [c for c in 1:n_chunks
               if (c-1)*nt_chunk + 1 >= t_safe_start &&
                  c*nt_chunk          <= t_safe_end]



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
thk   = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF   = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)
g = 9.8


# --- Filter (9-15 day band, 1 step sampling) ---
T1, T2, delt, N = 9.0, 15.0, 1.0, 4
fcutlow, fcuthigh = 1/T2, 1/T1
fnq = 1/delt
bpf = digitalfilter(Bandpass(fcutlow, fcuthigh), Butterworth(N); fs=fnq)


# ============================================================================
# MAIN WORKFLOW — single pass over all timesteps, averaged three ways at the end
# ============================================================================
println("Computing conversion (full timestep pass, then 3day/weekly/full averaging)")


mkpath(joinpath(base2, "Conv_3day"))
mkpath(joinpath(base2, "Conv_weekly"))
mkpath(joinpath(base2, "Conv"))



Threads.@threads for xn in cfg["xn_start"]:cfg["xn_end"]
   for yn in cfg["yn_start"]:cfg["yn_end"]
       suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)


       # --- Read fields (full time series) ---
       rho = Float64.(open(joinpath(base, "Density", "rho_in_$suffix.bin"), "r") do io
           nbytes = nx * ny * nz * nt * sizeof(Float64)
           raw_bytes = read(io, nbytes)
           raw_data = reinterpret(Float64, raw_bytes)
           reshaped_data = reshape(raw_data, nx, ny, nz, nt)
       end)


       hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))


       DRFfull = hFacC .* DRF3d
       z     = cumsum(DRFfull, dims=3)
       depth = sum(DRFfull, dims=3)
       DRFfull[hFacC .== 0] .= 0.0


       fu = Float64.(open(joinpath(base2, "UVW_F", "fu_$suffix.bin"), "r") do io
           nbytes = nx * ny * nz * nt * sizeof(Float32)
           raw_bytes = read(io, nbytes)
           raw_data = reinterpret(Float32, raw_bytes)
           reshaped_data = reshape(raw_data, nx, ny, nz, nt)
       end)


       fv = Float64.(open(joinpath(base2, "UVW_F", "fv_$suffix.bin"), "r") do io
           nbytes = nx * ny * nz * nt * sizeof(Float32)
           raw_bytes = read(io, nbytes)
           raw_data = reinterpret(Float32, raw_bytes)
           reshaped_data = reshape(raw_data, nx, ny, nz, nt)
       end)


       fr = bandpassfilter(rho, T1, T2, delt, N, nt)
       rho = nothing; GC.gc()


       UDA = dropdims(sum(fu .* DRFfull, dims=3) ./ depth; dims=3)
       VDA = dropdims(sum(fv .* DRFfull, dims=3) ./ depth; dims=3)
       fu = nothing; fv = nothing; GC.gc()


       # --- Pressure & perturbations ---
       pres  = g .* cumsum(fr .* DRFfull, dims=3)
       fr = nothing; GC.gc()
       pfz   = cat(zeros(nx, ny, 1, nt), pres; dims=3)
       pres = nothing; GC.gc()
       pc_3d = 0.5 .* (pfz[:, :, 1:end-1, :] .+ pfz[:, :, 2:end, :])
       pfz = nothing; GC.gc()
       pa    = sum(pc_3d .* DRFfull, dims=3) ./ depth
       pp_3d = pc_3d .- pa
       pc_3d = nothing; pa = nothing; GC.gc()


       dx = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
       dy = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))


       H  = depth
       pb = pp_3d[:, :, end, :]   # bottom pressure (nx, ny, nt)
       println(pp_3d[12,12,:,10])
       pp_3d = nothing; GC.gc()


       dHdx = zeros(nx-2, ny)
       dHdx[:, :] .= (H[3:nx, :] .- H[1:nx-2, :]) ./ (dx[2:nx-1, :] .+ dx[1:nx-2, :])


       dHdy = zeros(nx, ny-2)
       dHdy[:, :] .= (H[:, 3:ny] .- H[:, 1:ny-2]) ./ (dy[:, 1:ny-2] .+ dy[:, 2:ny-1])


       W1 = .-(UDA[2:end-1, :, :] .* dHdx)
       W2 = .-(VDA[:, 2:end-1, :] .* dHdy)
       UDA = nothing; VDA = nothing; GC.gc()


       w1c = W1[:, 2:end-1, :]
       w2c = W2[2:end-1, :, :]
       w   = w1c .+ w2c
       W1 = nothing; W2 = nothing; w1c = nothing; w2c = nothing; GC.gc()


       # Conversion time series (nx-2, ny-2, nt)
       c = pb[2:end-1, 2:end-1, :] .* w
       pb = nothing; w = nothing; GC.gc()


       suffix2 = @sprintf("%02dx%02d_%d", xn, yn, buf-2)


       # ====================================================================
       # Average the single conversion time series three different ways
       # ====================================================================


       # --- Full time-mean ---
       ca_full = dropdims(mean(c; dims=3); dims=3)   # (nx-2, ny-2)
       open(joinpath(base2, "Conv", "Conv_$suffix2.bin"), "w") do io
           write(io, Float32.(ca_full))
       end


       # --- 3-day chunk means ---
       ca_3day = zeros(nx-2, ny-2, nt3)
       for (i, ch) in enumerate(safe_chunks)
            t1 = (ch-1)*nt_chunk + 1
            t2 = ch*nt_chunk
            ca_3day[:, :, i] = Float32.(dropdims(mean(c[:, :, t1:t2], dims=3), dims=3))
        end
       open(joinpath(base2, "Conv_3day", "Conv_3day_$suffix2.bin"), "w") do io
           write(io, Float32.(ca_3day))
       end


       # --- Weekly window mean (Apr 22-28) ---
       ca_weekly = dropdims(mean(c[:, :, idx_start:idx_end]; dims=3); dims=3)
       open(joinpath(base2, "Conv_weekly", "Conv_weekly_$suffix2.bin"), "w") do io
           write(io, Float32.(ca_weekly))
       end


       c = nothing; GC.gc()


       println("  Completed tile: $suffix (3day + weekly + full conversion)")
   end
end


println("Completed conversion for 3-day periods, weekly window, and full time average")




