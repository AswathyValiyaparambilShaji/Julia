using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML
using CairoMakie, SparseArrays


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin


config_file = get(ENV, "JULIA_CONFIG",
            joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)


base  = cfg["base_path"]
base2 = cfg["base_path2"]


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
dt = 25  # time step in seconds
dto = 144
Tts = 366192
nt = div(Tts, dto)
ts = 72  # timesteps per 3-day period
nt_avg = div(nt, ts)  # number of 3-day periods
nt3 = div(nt, 3*24)  # Number of 3-day periods
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


# --- Thickness & constants ---
thk = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)


rho0 = 1027.5


# ============================================================================
# MAIN WORKFLOW — single pass over all timesteps, averaged three ways at the end
# ============================================================================
println("Starting energy tendency calculation (full timestep pass, then 3day/weekly/full averaging)...")


mkpath(joinpath(base2, "TE_t_3day"))
mkpath(joinpath(base2, "TE_t_weekly"))
mkpath(joinpath(base2, "TE_t"))



Threads.@threads for xn in cfg["xn_start"]:cfg["xn_end"]
   for yn in cfg["yn_start"]:cfg["yn_end"]
       suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)


       println("\n--- Processing tile: $suffix ---")


       # --- Read APE ---
       APE = Float64.(open(joinpath(base2, "APE", "APE_t_sm_$suffix.bin"), "r") do io
           nbytes = nx * ny * nz * nt * sizeof(Float32)
           reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nz, nt)
       end)


       # --- Read KE ---
       ke_t = Float64.(open(joinpath(base2, "KE", "ke_t_sm_$suffix.bin"), "r") do io
           nbytes = nx * ny * nz * nt * sizeof(Float32)
           raw_bytes = read(io, nbytes)
           raw_data = reinterpret(Float32, raw_bytes)
           reshape(raw_data, nx, ny, nz, nt)
       end)


       # --- Calculate total energy ---
       TE = APE .+ ke_t
       APE = nothing; ke_t = nothing; GC.gc()


       # --- Calculate ∂E/∂t using centered differences ---
       dEdt = zeros(Float64, nx, ny, nz, nt)
       dt_output = dt * dto  # = 3600 seconds = 1 hour


       # Forward difference for first time step
       dEdt[:, :, :, 1] = (TE[:, :, :, 2] .- TE[:, :, :, 1]) ./ dt_output


       # Centered differences for interior points
       for t in 2:nt-1
           dEdt[:, :, :, t] = (TE[:, :, :, t+1] .- TE[:, :, :, t-1]) ./ (2 * dt_output)
       end


       # Backward difference for last time step
       dEdt[:, :, :, nt] = (TE[:, :, :, nt] .- TE[:, :, :, nt-1]) ./ dt_output


       TE = nothing; GC.gc()


       # --- Replace NaN with zero ---
       dEdt[isnan.(dEdt)] .= 0.0


       # --- Depth integrate ---
       dEdt_depth_int = zeros(Float64, nx, ny, nt)
       for t in 1:nt
           for k in 1:nz
               dEdt_depth_int[:, :, t] .+= dEdt[:, :, k, t] .* DRF[k]
           end
       end


       dEdt = nothing; GC.gc()


       # ====================================================================
       # Average the single dEdt_depth_int array three different ways
       # ====================================================================


       # --- Full time-mean ---
       dEdt_time_avg = dropdims(mean(dEdt_depth_int, dims=3), dims=3)
       output_dir_full = joinpath(base2, "TE_t")
       open(joinpath(output_dir_full, "te_t_mean_$suffix.bin"), "w") do io
           write(io, Float32.(dEdt_time_avg))
       end


       # --- 3-day chunk means ---
       dEdt_3day = zeros(Float64, nx, ny, nt3)
       for (i, c) in enumerate(safe_chunks)
            t1 = (c-1)*nt_chunk + 1
            t2 = c*nt_chunk
            dEdt_3day[:, :, i] = Float32.(dropdims(
                mean(dEdt_depth_int[:, :, t1:t2], dims=3), dims=3))
        end
       output_dir_3day = joinpath(base2, "TE_t_3day")
       open(joinpath(output_dir_3day, "te_t_3day_$suffix.bin"), "w") do io
           write(io, Float32.(dEdt_3day))
       end


       # --- Weekly window mean (Apr 22-28) ---
       dEdt_weekly = dropdims(mean(dEdt_depth_int[:, :, idx_start:idx_end], dims=3), dims=3)
       output_dir_weekly = joinpath(base2, "TE_t_weekly")
       open(joinpath(output_dir_weekly, "te_t_weekly_$suffix.bin"), "w") do io
           write(io, Float32.(dEdt_weekly))
       end


       dEdt_depth_int = nothing; GC.gc()


       println("Completed tile: $suffix")
       println("Output saved to $output_dir_full, $output_dir_3day, $output_dir_weekly")
   end
end


println("\n=== All tiles processed successfully (3day + weekly + full) ===")




