using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML, Impute


include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter


config_file = get(ENV, "JULIA_CONFIG",
            joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)


base  = cfg["base_path"]
base2 = cfg["base_path2"]


#-0.0026407592217563463, 0.001007435484580632


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
ts = 72
nt_avg = div(nt, ts)
nt3 = div(nt, 3*24)
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


# --- Weekly window ---
hour_apr22_start = 31*24 + (22-1)*24
hour_apr28_end   = 31*24 +  28   *24 - 1
idx_start        = hour_apr22_start + 1
idx_end          = hour_apr28_end   + 1
nt_week          = idx_end - idx_start + 1


# --- Thickness & constants ---
thk  = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF  = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)
rho0 = 1027.5
g    = 9.8


# ============================================================================
# Helper: compute horizontal gradients of IT buoyancy at a single timestep
# Returns b_IT_x, b_IT_y each of size (nx, ny, nz)
# ============================================================================
function compute_IT_buoyancy_gradients(b_t, dx, dy, hFacC, nx, ny, nz)
   b_x = fill(NaN, nx, ny, nz)
   b_y = fill(NaN, nx, ny, nz)


   for k in 1:nz
       b_x[2:end-1, :, k] .= (b_t[3:end, :, k] .- b_t[1:end-2, :, k]) ./
                               (dx[2:end-1, :] .+ dx[1:end-2, :])
       b_y[:, 2:end-1, k] .= (b_t[:, 3:end, k] .- b_t[:, 1:end-2, k]) ./
                               (dy[:, 2:end-1] .+ dy[:, 1:end-2])
   end


   # Mask near boundaries — consistent with your BP code
   for k in 1:nz, j in 2:ny-1, i in 2:nx-1
       if hFacC[i-1,j,k] != 1 || hFacC[i,j,k] != 1 || hFacC[i+1,j,k] != 1
           b_x[i, j, k] = NaN
       end
       if hFacC[i,j-1,k] != 1 || hFacC[i,j,k] != 1 || hFacC[i,j+1,k] != 1
           b_y[i, j, k] = NaN
       end
   end


   return b_x, b_y
end


# ============================================================================
# MAIN WORKFLOW — single pass over all timesteps, averaged three ways at the end
# ============================================================================


mkpath(joinpath(base2, "G_buoy_3day"))
mkpath(joinpath(base2, "G_buoy_weekly"))
mkpath(joinpath(base2, "G_buoy_full"))


println("Starting G_buoy calculation (full timestep pass, then 3day/weekly/full averaging)...")



Threads.@threads for xn in cfg["xn_start"]:cfg["xn_end"]
   for yn in cfg["yn_start"]:cfg["yn_end"]
       suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
       println("\n--- Processing tile: $suffix ---")


       hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
       dx    = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
       dy    = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))


       # --- Background N2 (3-day averaged) ---
       N2_phase = Float64.(open(joinpath(base, "3day_mean", "N2", "N2_3day_$suffix.bin"), "r") do io
           raw = read(io, nx * ny * nz * nt_avg * sizeof(Float32))
           reshape(reinterpret(Float32, raw), nx, ny, nz, nt_avg)
       end)


       N2_adjusted = zeros(Float64, nx, ny, nz+1, nt_avg)
       N2_adjusted[:, :, 1,    :] = N2_phase[:, :, 1,      :]
       N2_adjusted[:, :, 2:nz, :] = N2_phase[:, :, 1:nz-1, :]
       N2_adjusted[:, :, nz+1, :] = N2_phase[:, :, nz-1,   :]


       N2_center = zeros(Float64, nx, ny, nz, nt_avg)
       for k in 1:nz
           N2_center[:, :, k, :] .= 0.5 .* (N2_adjusted[:, :, k, :] .+ N2_adjusted[:, :, k+1, :])
       end


       N2_threshold = 1.0e-8
       N2_center[N2_center .< N2_threshold] .= NaN


       for i in 1:nx, j in 1:ny, t in 1:nt_avg
           N2_center[i, j, :, t] = Impute.interp(N2_center[i, j, :, t])
       end


       # --- IT buoyancy (differentiated field) ---
       b_IT = Float64.(open(joinpath(base2, "b", "b_t_sm_$suffix.bin"), "r") do io
           raw = read(io, nx * ny * nz * nt * sizeof(Float32))
           reshape(reinterpret(Float32, raw), nx, ny, nz, nt)
       end)


       # --- NIW buoyancy (owned field) ---
       b_NIW = Float64.(open(joinpath(base2, "b_NIW", "b_niw_$suffix.bin"), "r") do io
           raw = read(io, nx * ny * nz * nt * sizeof(Float32))
           reshape(reinterpret(Float32, raw), nx, ny, nz, nt)
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


       DRFfull = hFacC .* DRF3d
       depth   = sum(DRFfull, dims=3)
       DRFfull[hFacC .== 0] .= 0.0
       mask3D  = hFacC .== 0
      
       ucA    = sum(fu .* DRFfull, dims=3) ./ depth    # (nx, ny, 1, nt) barotropic
       up_3d  = fu .- ucA
       up_3d[repeat(mask3D, 1, 1, 1, nt)] .= 0.0
       fu = ucA = nothing; GC.gc()


       vcA    = sum(fv .* DRFfull, dims=3) ./ depth
       vp_3d  = fv .- vcA
       vp_3d[repeat(mask3D, 1, 1, 1, nt)] .= 0.0
       fv = vcA = nothing; GC.gc()


       g_buoy = zeros(Float64, nx, ny, nt)


       println("Calculating G_buoy for each timestep...")
       for t in 1:nt
           t_avg = min(div(t - 1, ts) + 1, nt_avg)


           b_NIW_t = @view b_NIW[:, :, :, t]
           b_NIW_x, b_NIW_y = compute_IT_buoyancy_gradients(b_NIW_t, dx, dy, hFacC, nx, ny, nz)


           n2_val = @view N2_center[:, :, :, t_avg]
           b_t    = @view b_IT[:, :, :, t]
           ut     = @view up_3d[:, :, :, t]
           vt     = @view vp_3d[:, :, :, t]
           # G_buoy = -(b_NIW/N2)(u_NIW*∂b_IT/∂x + v_NIW*∂b_IT/∂y) * DRF
           temp1 = (b_t ./ n2_val) .* ut .* b_NIW_x .* DRFfull
           temp2 = (b_t ./ n2_val) .* vt .* b_NIW_y .* DRFfull


           temp1[isnan.(temp1)] .= 0.0
           temp2[isnan.(temp2)] .= 0.0


           g_buoy[:, :, t] = -rho0 .* dropdims(sum(temp1 .+ temp2, dims=3), dims=3)
       end


       b_IT = nothing; b_NIW = nothing; up_3d = nothing; vp_3d = nothing
       N2_center = nothing; GC.gc()


       # ====================================================================
       # Average the single g_buoy array three different ways
       # ====================================================================


       # --- Full time-mean ---
       G_buoy_full = dropdims(mean(g_buoy, dims=3), dims=3)
       println("  G_buoy (full) range: $(extrema(G_buoy_full[isfinite.(G_buoy_full)]))")
       open(joinpath(base2, "G_buoy_full", "g_buoy_mean_$suffix.bin"), "w") do io
           write(io, Float32.(G_buoy_full))
       end


       # --- 3-day chunk means ---
       G_buoy_3day = zeros(Float64, nx, ny, nt3)
       for (i, c) in enumerate(safe_chunks)
            t1 = (c-1)*nt_chunk + 1
            t2 = c*nt_chunk
            G_buoy_3day[:, :, i] = Float32.(dropdims(mean(g_buoy[:, :, t1:t2], dims=3), dims=3))
        end
       println("  G_buoy (3day) range: $(extrema(G_buoy_3day[isfinite.(G_buoy_3day)]))")
       open(joinpath(base2, "G_buoy_3day", "g_buoy_3day_$suffix.bin"), "w") do io
           write(io, Float32.(G_buoy_3day))
       end


       # --- Weekly window mean (Apr 22-28) ---
       G_buoy_weekly = dropdims(mean(g_buoy[:, :, idx_start:idx_end], dims=3), dims=3)
       println("  G_buoy (weekly) range: $(extrema(G_buoy_weekly[isfinite.(G_buoy_weekly)]))")
       open(joinpath(base2, "G_buoy_weekly", "g_buoy_weekly_$suffix.bin"), "w") do io
           write(io, Float32.(G_buoy_weekly))
       end


       g_buoy = nothing; GC.gc()
       println("Completed tile: $suffix")
   end
end


println("\n=== All tiles processed successfully (3day + weekly + full) ===")




