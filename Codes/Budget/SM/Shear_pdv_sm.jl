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
println("Starting vertical shear production calculation (full timestep pass, then 3day/weekly/full averaging)...")


mkpath(joinpath(base2, "SP_V_3day"))
mkpath(joinpath(base2, "SP_V_weekly"))
mkpath(joinpath(base2, "SP_V_bt"))


for xn in cfg["xn_start"]:cfg["xn_end"]
   for yn in cfg["yn_start"]:cfg["yn_end"]
       suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)


       println("\n--- Processing tile: $suffix ---")


       # --- Read grid metrics ---
       hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))


       # --- Read mean velocity fields (3-day averaged) ---
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


       # --- Read fluctuating velocities ---
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


       fw = Float64.(open(joinpath(base2, "UVW_F", "fw_$suffix.bin"), "r") do io
           nbytes = nx * ny * nz * nt * sizeof(Float32)
           raw_bytes = read(io, nbytes)
           raw_data = reinterpret(Float32, raw_bytes)
           reshape(raw_data, nx, ny, nz, nt)
       end)


       # --- Calculate cell thicknesses ---
       DRFfull = hFacC .* DRF3d
       depth   = sum(DRFfull, dims=3)


       DRFfull[hFacC .== 0] .= 0.0


       mask3D  = hFacC .== 0                           # (nx, ny, nz) Bool — reuse for masking
       ucA    = sum(fu .* DRFfull, dims=3) ./ depth    # (nx, ny, 1, nt) barotropic
       up_3d  = fu .- ucA
       up_3d[repeat(mask3D, 1, 1, 1, nt)] .= 0.0
       fu = ucA = nothing; GC.gc()
       vcA    = sum(fv .* DRFfull, dims=3) ./ depth
       vp_3d  = fv .- vcA
       vp_3d[repeat(mask3D, 1, 1, 1, nt)] .= 0.0
       fv = vcA = nothing; GC.gc()
       #wcA    = sum(fw .* DRFfull, dims=3) ./ depth
       wp_3d  = fw #.- wcA
       wp_3d[repeat(mask3D, 1, 1, 1, nt)] .= 0.0
       fw =  nothing; GC.gc()


       # --- Calculate vertical gradients of mean velocities: ∂U/∂z, ∂V/∂z ---
       println("Calculating vertical gradients of mean velocities...")
       U_z = zeros(Float64, nx, ny, nz, nt_avg)
       V_z = zeros(Float64, nx, ny, nz, nt_avg)


       for t_avg in 1:nt_avg
           for k in 2:nz-1
               dz = DRF[k-1]/2.0 + DRF[k] + DRF[k+1]/2.0
               U_z[:, :, k, t_avg] = (U[:, :, k-1, t_avg] .- U[:, :, k+1, t_avg]) ./ dz
               V_z[:, :, k, t_avg] = (V[:, :, k-1, t_avg] .- V[:, :, k+1, t_avg]) ./ dz
           end
       end


       println("Vertical gradients calculated")


       # --- Initialize output array ---
       sp_v = zeros(Float64, nx, ny, nt)


       # --- Calculate vertical shear production for each timestep ---
       println("Calculating vertical shear production...")
       for t in 1:nt
           t_avg = min(div(t - 1, ts) + 1, nt_avg)


           U_z_t = @view U_z[:, :, :, t_avg]
           V_z_t = @view V_z[:, :, :, t_avg]
           ut    = @view up_3d[:, :, :, t]
           vt    = @view vp_3d[:, :, :, t]
           wt    = @view wp_3d[:, :, :, t]


           temp1 = wt .* ut .* U_z_t .* DRFfull
           temp2 = wt .* vt .* V_z_t .* DRFfull


           sp_v[:, :, t] = -rho0 .* dropdims(sum((temp1 .+ temp2), dims=3), dims=3)
       end


       println("Vertical shear production calculation complete")


       U = nothing; V = nothing; up_3d = nothing; vp_3d = nothing; wp_3d = nothing
       U_z = nothing; V_z = nothing; GC.gc()


       # ====================================================================
       # Average the single sp_v array three different ways
       # ====================================================================


       # --- Full time-mean ---
       SP_V_full = dropdims(mean(sp_v, dims=3), dims=3)
       println(SP_V_full[20,1:10])
       output_dir_full = joinpath(base2, "SP_V_bt")
       open(joinpath(output_dir_full, "sp_v_mean_$suffix.bin"), "w") do io
           write(io, Float32.(SP_V_full))
       end


       # --- 3-day chunk means ---
       SP_V_3day = zeros(Float64, nx, ny, nt3)
       for (i, c) in enumerate(safe_chunks)
            t1 = (c-1)*nt_chunk + 1
            t2 = c*nt_chunk
            SP_V_3day[:, :, i] = Float32.(dropdims(mean(sp_v[:, :, t1:t2], dims=3), dims=3))
        end
       output_dir_3day = joinpath(base2, "SP_V_3day")
       open(joinpath(output_dir_3day, "sp_v_3day_$suffix.bin"), "w") do io
           write(io, Float32.(SP_V_3day))
       end


       # --- Weekly window mean (Apr 22-28) ---
       SP_V_weekly = dropdims(mean(sp_v[:, :, idx_start:idx_end], dims=3), dims=3)
       output_dir_weekly = joinpath(base2, "SP_V_weekly")
       open(joinpath(output_dir_weekly, "sp_v_weekly_$suffix.bin"), "w") do io
           write(io, Float32.(SP_V_weekly))
       end


       sp_v = nothing; GC.gc()


       println("Completed tile: $suffix")
       println("Output saved to $output_dir_full, $output_dir_3day, $output_dir_weekly")
   end
end


println("\n=== All tiles processed successfully (3day + weekly + full) ===")




