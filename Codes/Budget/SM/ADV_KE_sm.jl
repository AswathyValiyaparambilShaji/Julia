using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML


include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter


config_file = get(ENV, "JULIA_CONFIG",
            joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)


base  = cfg["base_path"]
base2 = cfg["base_path2"]


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
println("Starting KE flux calculation (full timestep pass, then 3day/weekly/full averaging)...")


mkpath(joinpath(base2, "U_KE_3day"))
mkpath(joinpath(base2, "U_KE_weekly"))
mkpath(joinpath(base2, "U_KE"))



Threads.@threads for xn in cfg["xn_start"]:cfg["xn_end"]
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
       ke_t = Float64.(open(joinpath(base2, "KE", "ke_t_sm_$suffix.bin"), "r") do io
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


       ke_t = nothing; GC.gc()


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


       U = nothing; V = nothing; ke_x = nothing; ke_y = nothing; GC.gc()


       # ====================================================================
       # Average the single U_KE array three different ways
       # ====================================================================


       # --- Full time-mean ---
       u_ke_full = dropdims(mean(U_KE, dims=3), dims=3)   # (nx, ny)
       output_dir_full = joinpath(base2, "U_KE")
       open(joinpath(output_dir_full, "u_ke_mean_$suffix.bin"), "w") do io
           write(io, Float32.(u_ke_full))
       end


       # --- 3-day chunk means ---
       U_KE_3day = zeros(Float64, nx, ny, nt3)
       for (i, c) in enumerate(safe_chunks)
            t1 = (c-1)*nt_chunk + 1
            t2 = c*nt_chunk
            U_KE_3day[:, :, i] = Float32.(dropdims(mean(U_KE[:, :, t1:t2], dims=3), dims=3))
        end
       output_dir_3day = joinpath(base2, "U_KE_3day")
       open(joinpath(output_dir_3day, "u_ke_3day_$suffix.bin"), "w") do io
           write(io, Float32.(U_KE_3day))
       end


       # --- Weekly window mean (Apr 22-28) ---
       u_ke_weekly = dropdims(mean(U_KE[:, :, idx_start:idx_end], dims=3), dims=3)
       output_dir_weekly = joinpath(base2, "U_KE_weekly")
       open(joinpath(output_dir_weekly, "u_ke_weekly_$suffix.bin"), "w") do io
           write(io, Float32.(u_ke_weekly))
       end


       U_KE = nothing; GC.gc()


       println("Completed tile: $suffix")
       println("Output saved to $output_dir_full, $output_dir_3day, $output_dir_weekly")
   end
end


println("\n=== All tiles processed successfully (3day + weekly + full) ===")




