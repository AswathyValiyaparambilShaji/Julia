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
nt_ = div(nt, 3*24)

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
function compute_IT_gradients(fu_n, fv_n,  nx, ny,nz)
   U_z = zeros(Float64, nx, ny, nz)
   V_z = zeros(Float64, nx, ny, nz)
   


    for k in 2:nz-1
        dz = DRF[k-1]/2.0 + DRF[k] + DRF[k+1]/2.0
        U_z[:, :, k] = (fu_n[ :, :, k-1] .- fu_n[ :, :, k+1]) ./dz
        V_z[:, :, k] = (fv_n[:, :, k-1] .- fv_n[ :, :, k+1]) ./dz
    end


   return U_z, V_z
end



# ============================================================================
# MAIN WORKFLOW — single pass over all timesteps, averaged three ways at the end
# ============================================================================


mkpath(joinpath(base2, "G_vel_V_3day"))
mkpath(joinpath(base2, "G_vel_V_weekly"))
mkpath(joinpath(base2, "G_vel_V_full"))


println("Starting G_buoy calculation (full timestep pass, then 3day/weekly/full averaging)...")


Threads.@threads for xn in cfg["xn_start"]:cfg["xn_end"]
   for yn in cfg["yn_start"]:cfg["yn_end"]
       suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
       println("\n--- Processing tile: $suffix ---")


       hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
       dx    = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
       dy    = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))

        fu_niw = Float64.(open(joinpath(base2, "UVW_NIW", "fu_niw_$suffix.bin"), "r") do io
           reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
        end)


        fv_niw = Float64.(open(joinpath(base2, "UVW_NIW", "fv_niw_$suffix.bin"), "r") do io
           reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
        end)

        fu = Float64.(open(joinpath(base2, "UVW_F", "fu_$suffix.bin"), "r") do io
           reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
        end)


        fv = Float64.(open(joinpath(base2, "UVW_F", "fv_$suffix.bin"), "r") do io
           reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
        end)

        fw = Float64.(open(joinpath(base2, "UVW_F", "fw_$suffix.bin"), "r") do io
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

       #vcA    = sum(fv .* DRFfull, dims=3) ./ depth
       wp_3d  = fw #.- vcA
       wp_3d[repeat(mask3D, 1, 1, 1, nt)] .= 0.0
       fw = nothing; GC.gc()

       g_velv = zeros(Float64, nx, ny, nt)


       println("Calculating G_vel for each timestep...")
       for t in 1:nt
           fu_n = @view fu_niw[:, :, :, t]
           fv_n = @view fv_niw[:, :, :, t]
           fu_nz, fv_nz = compute_IT_gradients(fu_n, fv_n, nx, ny, nz)


           ut = @view up_3d[:, :, :, t]
           vt = @view vp_3d[:, :, :, t]
           wt = @view wp_3d[:, :, :, t]


           # G_vel = -rho0 * [us*(us*∂ut/∂x + vs*∂ut/∂y) + vs*(us*∂vt/∂x + vs*∂vt/∂y)] * DRF
           temp1 = wt .* ut .* fu_nz .* DRFfull
           temp2 = wt .* vt .* fv_nz .* DRFfull
          


           g_velv[:, :, t] = -rho0 .* dropdims(sum((temp1 .+ temp2), dims=3), dims=3)
       end
        up_3d = nothing; vp_3d = nothing
        wp_3d = nothing; GC.gc()


       # ====================================================================
       # Average the single g_buoy array three different ways
       # ====================================================================


       # --- Full time-mean ---
       G_velv_full = dropdims(mean(g_velv, dims=3), dims=3)
       open(joinpath(base2, "G_vel_V_full", "g_velv_mean_$suffix.bin"), "w") do io
           write(io, Float32.(G_velv_full))
       end


       # --- 3-day chunk means ---
       G_vel_V_3day = zeros(Float64, nx, ny, nt3)
       for (i, c) in enumerate(safe_chunks)
            t1 = (c-1)*nt_chunk + 1
            t2 = c*nt_chunk
            G_vel_V_3day[:, :, i] = Float32.(dropdims(mean(g_velv[:, :, t1:t2], dims=3), dims=3))
        end
       open(joinpath(base2, "G_vel_V_3day", "g_vel_V_3day_$suffix.bin"), "w") do io
           write(io, Float32.(G_vel_V_3day))
       end


       # --- Weekly window mean (Apr 22-28) ---
       G_vel_V_weekly = dropdims(mean(g_velv[:, :, idx_start:idx_end], dims=3), dims=3)
       open(joinpath(base2, "G_vel_V_weekly", "g_vel_V_weekly_$suffix.bin"), "w") do io
           write(io, Float32.(G_vel_V_weekly))
       end


       g_buoy = nothing; GC.gc()
       println("Completed tile: $suffix")
   end
end


println("\n=== All tiles processed successfully (3day + weekly + full) ===")




