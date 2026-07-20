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


# ============================================================================
# Helper: compute horizontal gradients at a single timestep
# Returns fu_x, fu_y, fv_x, fv_y each of size (nx, ny, nz)
# ============================================================================
function compute_IT_gradients(fu_t, fv_t, dx, dy, nx, ny, nz)
   fu_x = zeros(Float64, nx, ny, nz)
   fu_y = zeros(Float64, nx, ny, nz)
   fv_x = zeros(Float64, nx, ny, nz)
   fv_y = zeros(Float64, nx, ny, nz)


   dx_avg = dx[2:end-1, :] .+ dx[1:end-2, :]
   fu_x[2:end-1, :, :] = (fu_t[3:end, :, :] .- fu_t[1:end-2, :, :]) ./
                           reshape(dx_avg, nx-2, ny, 1)
   fv_x[2:end-1, :, :] = (fv_t[3:end, :, :] .- fv_t[1:end-2, :, :]) ./
                           reshape(dx_avg, nx-2, ny, 1)


   dy_avg = dy[:, 2:end-1] .+ dy[:, 1:end-2]
   fu_y[:, 2:end-1, :] = (fu_t[:, 3:end, :] .- fu_t[:, 1:end-2, :]) ./
                           reshape(dy_avg, nx, ny-2, 1)
   fv_y[:, 2:end-1, :] = (fv_t[:, 3:end, :] .- fv_t[:, 1:end-2, :]) ./
                           reshape(dy_avg, nx, ny-2, 1)


   return fu_x, fu_y, fv_x, fv_y
end


# ============================================================================
# MAIN WORKFLOW — single pass over all timesteps, averaged three ways at the end
# ============================================================================
println("Starting G_vel calculation (full timestep pass, then 3day/weekly/full averaging)...")


mkpath(joinpath(base2, "G_vel_H_3day"))
mkpath(joinpath(base2, "G_vel_H_weekly"))
mkpath(joinpath(base2, "G_vel_H_full"))



Threads.@threads for xn in cfg["xn_start"]:cfg["xn_end"]
   for yn in cfg["yn_start"]:cfg["yn_end"]
       suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
       println("\n--- Processing tile: $suffix ---")


       hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
       dx    = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
       dy    = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))


       fu = Float64.(open(joinpath(base2, "UVW_F", "fu_$suffix.bin"), "r") do io
           reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
       end)


       fv = Float64.(open(joinpath(base2, "UVW_F", "fv_$suffix.bin"), "r") do io
           reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
       end)


       fu_niw = Float64.(open(joinpath(base2, "UVW_NIW", "fu_niw_$suffix.bin"), "r") do io
           reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
       end)


       fv_niw = Float64.(open(joinpath(base2, "UVW_NIW", "fv_niw_$suffix.bin"), "r") do io
           reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
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


       g_vel = zeros(Float64, nx, ny, nt)


       println("Calculating G_vel for each timestep...")
       for t in 1:nt
           fu_n = @view fu_niw[:, :, :, t]
           fv_n = @view fv_niw[:, :, :, t]
           fu_nx, fu_ny, fv_nx, fv_ny = compute_IT_gradients(fu_n, fv_n, dx, dy, nx, ny, nz)


           ut = @view up_3d[:, :, :, t]
           vt = @view vp_3d[:, :, :, t]


           # G_vel = -rho0 * [us*(us*∂ut/∂x + vs*∂ut/∂y) + vs*(us*∂vt/∂x + vs*∂vt/∂y)] * DRF
           temp1 = ut .* ut .* fu_nx .* DRFfull
           temp2 = ut .* vt .* fu_ny .* DRFfull
           temp3 = vt .* ut .* fv_nx .* DRFfull
           temp4 = vt .* vt .* fv_ny .* DRFfull


           g_vel[:, :, t] = -rho0 .* dropdims(sum((temp1 .+ temp2 .+ temp3 .+ temp4), dims=3), dims=3)
       end


       fu_niw = nothing; fv_niw = nothing; up_3d = nothing; vp_3d = nothing; GC.gc()


       # ====================================================================
       # Average the single g_vel array three different ways
       # ====================================================================


       # --- Full time-mean ---
       G_vel_full = dropdims(mean(g_vel, dims=3), dims=3)
       open(joinpath(base2, "G_vel_H_full", "g_vel_mean_$suffix.bin"), "w") do io
           write(io, Float32.(G_vel_full))
       end


       # --- 3-day chunk means ---
       G_vel_3day = zeros(Float64, nx, ny, nt3)
       for (i, c) in enumerate(safe_chunks)
            t1 = (c-1)*nt_chunk + 1
            t2 = c*nt_chunk
            G_vel_3day[:, :, i] = Float32.(dropdims(mean(g_vel[:, :, t1:t2], dims=3), dims=3))
        end
       open(joinpath(base2, "G_vel_H_3day", "g_vel_3day_$suffix.bin"), "w") do io
           write(io, Float32.(G_vel_3day))
       end


       # --- Weekly window mean (Apr 22-28) ---
       G_vel_weekly = dropdims(mean(g_vel[:, :, idx_start:idx_end], dims=3), dims=3)
       open(joinpath(base2, "G_vel_H_weekly", "g_vel_weekly_$suffix.bin"), "w") do io
           write(io, Float32.(G_vel_weekly))
       end


       g_vel = nothing; GC.gc()


       println("Completed tile: $suffix")
   end
end


println("\n=== All tiles processed successfully (3day + weekly + full) ===")




