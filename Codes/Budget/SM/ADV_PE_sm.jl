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
println("Starting PE flux calculation (full timestep pass, then 3day/weekly/full averaging)...")


mkpath(joinpath(base2, "U_PE_3day"))
mkpath(joinpath(base2, "U_PE_weekly"))
mkpath(joinpath(base2, "U_PE"))



Threads.@threads for xn in cfg["xn_start"]:cfg["xn_end"]
   for yn in cfg["yn_start"]:cfg["yn_end"]
       suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)


       println("\n--- Processing tile: $suffix ---")


       # --- Read grid metrics ---
       hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
       dx = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
       dy = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))


       # --- Read U and V (3-day averaged) ---
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


       # --- Read PE (full temporal resolution) ---
       pe = Float64.(open(joinpath(base2, "pe", "pe_t_sm_$suffix.bin"), "r") do io
           nbytes = nx * ny * nz * nt * sizeof(Float32)
           raw_bytes = read(io, nbytes)
           raw_data = reinterpret(Float32, raw_bytes)
           reshape(raw_data, nx, ny, nz, nt)
       end)


       # --- Read N2 (3-day averaged) ---
       N2 = Float64.(open(joinpath(base, "3day_mean", "N2", "N2_3day_$suffix.bin"), "r") do io
           nbytes = nx * ny * nz * nt_avg * sizeof(Float32)
           raw_bytes = read(io, nbytes)
           raw_data = reinterpret(Float32, raw_bytes)
           reshape(raw_data, nx, ny, nz, nt_avg)
       end)


       # --- Calculate grid metrics ---
       DRFfull = hFacC .* DRF3d
       DRFfull[hFacC .== 0] .= 0.0


       # --- Adjust N2 to nz+1 levels (interfaces) then back to centers ---
       N2_adjusted = zeros(Float64, nx, ny, nz+1, nt_avg)
       N2_adjusted[:, :, 1, :]      = N2[:, :, 1, :]
       N2_adjusted[:, :, 2:nz, :]   = N2[:, :, 1:nz-1, :]
       N2_adjusted[:, :, nz+1, :]   = N2[:, :, nz, :]
       k_last_full = zeros(Int, nx, ny)
       for j in 1:ny, i in 1:nx
           for k in nz:-1:1
               if hFacC[i, j, k] >= 1.0
                   k_last_full[i, j] = k
                   break
               end
           end
       end


       for j in 1:ny, i in 1:nx
           kf = k_last_full[i, j]
           if kf > 0
               N2_adjusted[i, j, kf+1, :] .= N2[i, j, kf-1, :] # k+1 because of the concatination of adition surface grid
           end
       end


       N2_center = zeros(Float64, nx, ny, nz, nt_avg)
       for k in 1:nz
           N2_center[:, :, k, :] .= 0.5 .* (N2_adjusted[:, :, k, :] .+ N2_adjusted[:, :, k+1, :])
       end
       N2_adjusted = nothing
       N2    = nothing


       # --- Filter out anomalously low N2 values ---
       N2_threshold = 1.0e-8


       n_filtered = sum(N2_center .< N2_threshold)
       n_total = length(N2_center)


       N2_center[N2_center .< N2_threshold] .= N2_threshold


       # --- Calculate PE gradients (vectorized) ---
       println("Calculating PE gradients...")
       pe_x = zeros(Float64, nx, ny, nz, nt)
       pe_y = zeros(Float64, nx, ny, nz, nt)


       dx_avg = dx[2:end-1, :] .+ dx[1:end-2, :]
       pe_x[2:end-1, :, :, :] = (pe[3:end, :, :, :] .- pe[1:end-2, :, :, :]) ./
                                 reshape(dx_avg, nx-2, ny, 1, 1)


       dy_avg = dy[:, 2:end-1] .+ dy[:, 1:end-2]
       pe_y[:, 2:end-1, :, :] = (pe[:, 3:end, :, :] .- pe[:, 1:end-2, :, :]) ./
                                 reshape(dy_avg, nx, ny-2, 1, 1)


       pe = nothing; GC.gc()


       println("Gradients calculated")


       # --- Initialize output: depth-integrated flux at each timestep ---
       U_PE = zeros(Float64, nx, ny, nt)


       # --- Calculate advective PE flux for each timestep ---
       println("Calculating advective PE flux...")
       for t in 1:nt
           t_avg  = min(div(t - 1, ts) + 1, nt_avg)


           u_avg  = @view U[:, :, :, t_avg]
           v_avg  = @view V[:, :, :, t_avg]
           n2_avg = @view N2_center[:, :, :, t_avg]
           pe_x_t = @view pe_x[:, :, :, t]
           pe_y_t = @view pe_y[:, :, :, t]


           temp1 = u_avg .* pe_x_t ./ n2_avg
           temp2 = v_avg .* pe_y_t ./ n2_avg


           temp1[isnan.(temp1)] .= 0.0
           temp2[isnan.(temp2)] .= 0.0


           U_PE[:, :, t] = rho0 .* dropdims(sum((temp1 .+ temp2) .* DRFfull, dims=3), dims=3)
       end


       println("Flux calculation complete")


       U = nothing; V = nothing; pe_x = nothing; pe_y = nothing
       N2_center = nothing; GC.gc()


       # ====================================================================
       # Average the single U_PE array three different ways
       # ====================================================================


       # --- Full time-mean ---
       u_pe_full = dropdims(mean(U_PE, dims=3), dims=3)   # (nx, ny)
       output_dir_full = joinpath(base2, "U_PE")
       open(joinpath(output_dir_full, "u_pe_mean_$suffix.bin"), "w") do io
           write(io, Float32.(u_pe_full))
       end


       # --- 3-day chunk means ---
       U_PE_3day = zeros(Float64, nx, ny, nt3)
       for (i, c) in enumerate(safe_chunks)
            t1 = (c-1)*nt_chunk + 1
            t2 = c*nt_chunk
            U_PE_3day[:, :, i] = Float32.(dropdims(mean(U_PE[:, :, t1:t2], dims=3), dims=3))
       end
       output_dir_3day = joinpath(base2, "U_PE_3day")
       open(joinpath(output_dir_3day, "u_pe_3day_$suffix.bin"), "w") do io
           write(io, Float32.(U_PE_3day))
       end


       # --- Weekly window mean (Apr 22-28) ---
       u_pe_weekly = dropdims(mean(U_PE[:, :, idx_start:idx_end], dims=3), dims=3)
       output_dir_weekly = joinpath(base2, "U_PE_weekly")
       open(joinpath(output_dir_weekly, "u_pe_weekly_$suffix.bin"), "w") do io
           write(io, Float32.(u_pe_weekly))
       end


       U_PE = nothing; GC.gc()


       println("Completed tile: $suffix")
       println("Output saved to $output_dir_full, $output_dir_3day, $output_dir_weekly")
   end
end


println("\n=== All tiles processed successfully (3day + weekly + full) ===")




