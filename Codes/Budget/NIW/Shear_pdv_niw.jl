using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML

include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter

config_file = get(ENV, "JULIA_CONFIG",
             joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)

base  = cfg["base_path"]
base2 = cfg["base_path2"]

# --- TIME MODE CONFIGURATION ---
# Options: "3day", "weekly", "full"
time_mode = "3day"

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

# --- Weekly window ---
hour_apr22_start = 31*24 + (22-1)*24
hour_apr28_end   = 31*24 +  28   *24 - 1
idx_start        = hour_apr22_start + 1
idx_end          = hour_apr28_end   + 1
nt_week          = idx_end - idx_start + 1

@printf("Weekly window: Apr 22 00:00 - Apr 28 23:00  ->  indices %d:%d  (%d hourly snapshots)\n",
        idx_start, idx_end, nt_week)

# --- Thickness & constants ---
thk  = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF  = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)
rho0 = 999.8

# ============================================================================
# Helper: compute vertical gradients of IT velocities at a single timestep
# ∂u_IT/∂z and ∂v_IT/∂z   same finite difference scheme as your SP_V code
# ============================================================================
function compute_IT_vertical_gradients(fu_t, fv_t, DRF, nx, ny, nz)
    fu_z = zeros(Float64, nx, ny, nz)
    fv_z = zeros(Float64, nx, ny, nz)

    for k in 2:nz-1
        dz = DRF[k-1]/2.0 + DRF[k] + DRF[k+1]/2.0
        fu_z[:, :, k] = (fu_t[:, :, k-1] .- fu_t[:, :, k+1]) ./ dz
        fv_z[:, :, k] = (fv_t[:, :, k-1] .- fv_t[:, :, k+1]) ./ dz
    end

    return fu_z, fv_z
end

# ============================================================================
# MAIN WORKFLOW
# ============================================================================

if time_mode == "3day"
    println("Starting G_vel_V calculation for $nt3 3-day periods...")
    mkpath(joinpath(base2, "G_vel_V_3day"))

    for xn in cfg["xn_start"]:cfg["xn_end"]
        for yn in cfg["yn_start"]:cfg["yn_end"]
            suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
            println("\n--- Processing tile: $suffix (3-day) ---")

            hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))

            # IT-filtered velocities (differentiated field  vertical gradients taken here)
            fu = Float64.(open(joinpath(base2, "UVW_F", "fu_$suffix.bin"), "r") do io
                reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
            end)
            fv = Float64.(open(joinpath(base2, "UVW_F", "fv_$suffix.bin"), "r") do io
                reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
            end)

            # NIW-filtered velocities (owned field  appears twice)
            fu_niw = Float64.(open(joinpath(base2, "UVW_NIW", "fu_niw_$suffix.bin"), "r") do io
                reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
            end)
            fv_niw = Float64.(open(joinpath(base2, "UVW_NIW", "fv_niw_$suffix.bin"), "r") do io
                reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
            end)
            fw_niw = Float64.(open(joinpath(base2, "UVW_NIW", "fw_niw_$suffix.bin"), "r") do io
                reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
            end)

            DRFfull = hFacC .* DRF3d
            DRFfull[hFacC .== 0] .= 0.0

            G_vel_V_3day = zeros(Float64, nx, ny, nt3)
            hrs_per_chunk = 3 * 24

            for t in 1:nt3
                t_start = (t-1) * hrs_per_chunk + 1
                t_end   = min(t * hrs_per_chunk, nt)

                g_vel_v_temp = zeros(Float64, nx, ny, t_end - t_start + 1)

                for idx in 1:(t_end - t_start + 1)
                    t_actual = t_start + idx - 1

                    # IT vertical gradients at this timestep
                    fu_t = @view fu[:, :, :, t_actual]
                    fv_t = @view fv[:, :, :, t_actual]
                    fu_z, fv_z = compute_IT_vertical_gradients(fu_t, fv_t, DRF, nx, ny, nz)

                    # NIW owned fields
                    us  = @view fu_niw[:, :, :, t_actual]
                    vs  = @view fv_niw[:, :, :, t_actual]
                    ws  = @view fw_niw[:, :, :, t_actual]

                    # G_vel_V = -rho0 * [ws*us*∂u_IT/∂z + ws*vs*∂v_IT/∂z] * DRF
                    temp1 = ws .* us .* fu_z .* DRFfull
                    temp2 = ws .* vs .* fv_z .* DRFfull

                    g_vel_v_temp[:, :, idx] = -rho0 .* dropdims(sum((temp1 .+ temp2), dims=3), dims=3)
                end

                G_vel_V_3day[:, :, t] = mean(g_vel_v_temp, dims=3)
            end

            open(joinpath(base2, "G_vel_V_3day", "g_vel_v_3day_$suffix.bin"), "w") do io
                write(io, Float32.(G_vel_V_3day))
            end
            println("Completed tile: $suffix")
        end
    end
    println("\n=== All tiles processed successfully (3-day) ===")

elseif time_mode == "weekly"
    println("Starting G_vel_V calculation for weekly window Apr 22-28...")
    mkpath(joinpath(base2, "G_vel_V_weekly"))

    for xn in cfg["xn_start"]:cfg["xn_end"]
        for yn in cfg["yn_start"]:cfg["yn_end"]
            suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
            println("\n--- Processing tile: $suffix (weekly) ---")

            hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))

            fu = Float64.(open(joinpath(base2, "UVW_F", "fu_$suffix.bin"), "r") do io
                reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
            end)[:, :, :, idx_start:idx_end]

            fv = Float64.(open(joinpath(base2, "UVW_F", "fv_$suffix.bin"), "r") do io
                reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
            end)[:, :, :, idx_start:idx_end]

            fu_niw = Float64.(open(joinpath(base2, "UVW_NIW", "fu_niw_$suffix.bin"), "r") do io
                reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
            end)[:, :, :, idx_start:idx_end]

            fv_niw = Float64.(open(joinpath(base2, "UVW_NIW", "fv_niw_$suffix.bin"), "r") do io
                reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
            end)[:, :, :, idx_start:idx_end]

            fw_niw = Float64.(open(joinpath(base2, "UVW_NIW", "fw_niw_$suffix.bin"), "r") do io
                reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
            end)[:, :, :, idx_start:idx_end]

            DRFfull = hFacC .* DRF3d
            DRFfull[hFacC .== 0] .= 0.0

            g_vel_v = zeros(Float64, nx, ny, nt_week)

            for idx in 1:nt_week
                fu_t = @view fu[:, :, :, idx]
                fv_t = @view fv[:, :, :, idx]
                fu_z, fv_z = compute_IT_vertical_gradients(fu_t, fv_t, DRF, nx, ny, nz)

                us  = @view fu_niw[:, :, :, idx]
                vs  = @view fv_niw[:, :, :, idx]
                ws  = @view fw_niw[:, :, :, idx]

                temp1 = ws .* us .* fu_z .* DRFfull
                temp2 = ws .* vs .* fv_z .* DRFfull

                g_vel_v[:, :, idx] = -rho0 .* dropdims(sum((temp1 .+ temp2), dims=3), dims=3)
            end

            G_vel_V = dropdims(mean(g_vel_v, dims=3), dims=3)

            open(joinpath(base2, "G_vel_V_weekly", "g_vel_v_weekly_$suffix.bin"), "w") do io
                write(io, Float32.(G_vel_V))
            end
            println("Completed tile: $suffix")
        end
    end
    println("\n=== All tiles processed successfully (weekly) ===")

elseif time_mode == "full"
    println("Starting G_vel_V calculation for full time average...")
    mkpath(joinpath(base2, "G_vel_V_full"))

    for xn in cfg["xn_start"]:cfg["xn_end"]
        for yn in cfg["yn_start"]:cfg["yn_end"]
            suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
            println("\n--- Processing tile: $suffix (full) ---")

            hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))

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
            fw_niw = Float64.(open(joinpath(base2, "UVW_NIW", "fw_niw_$suffix.bin"), "r") do io
                reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
            end)

            DRFfull = hFacC .* DRF3d
            DRFfull[hFacC .== 0] .= 0.0

            g_vel_v = zeros(Float64, nx, ny, nt)

            println("Calculating G_vel_V for each timestep...")
            for t in 1:nt
                fu_t = @view fu[:, :, :, t]
                fv_t = @view fv[:, :, :, t]
                fu_z, fv_z = compute_IT_vertical_gradients(fu_t, fv_t, DRF, nx, ny, nz)

                us  = @view fu_niw[:, :, :, t]
                vs  = @view fv_niw[:, :, :, t]
                ws  = @view fw_niw[:, :, :, t]

                temp1 = ws .* us .* fu_z .* DRFfull
                temp2 = ws .* vs .* fv_z .* DRFfull

                g_vel_v[:, :, t] = -rho0 .* dropdims(sum((temp1 .+ temp2), dims=3), dims=3)
            end

            G_vel_V = dropdims(mean(g_vel_v, dims=3), dims=3)

            open(joinpath(base2, "G_vel_V_full", "g_vel_v_mean_$suffix.bin"), "w") do io
                write(io, Float32.(G_vel_V))
            end
            println("Completed tile: $suffix")
        end
    end
    println("\n=== All tiles processed successfully (full) ===")

else
    error("Unknown time_mode '$time_mode'. Choose \"3day\", \"weekly\", or \"full\".")
end