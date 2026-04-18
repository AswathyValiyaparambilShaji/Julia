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
time_mode = "full"

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
# MAIN WORKFLOW
# ============================================================================

if time_mode == "3day"
    println("Starting G_vel calculation for $nt3 3-day periods...")
    mkpath(joinpath(base2, "G_vel_3day"))

    for xn in cfg["xn_start"]:cfg["xn_end"]
        for yn in cfg["yn_start"]:cfg["yn_end"]
            suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
            println("\n--- Processing tile: $suffix (3-day) ---")

            hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
            dx    = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
            dy    = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))

            # IT-filtered velocities (differentiated field)
            fu = Float64.(open(joinpath(base2, "UVW_F", "fu_$suffix.bin"), "r") do io
                reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
            end)
            fv = Float64.(open(joinpath(base2, "UVW_F", "fv_$suffix.bin"), "r") do io
                reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
            end)

            # NIW-filtered velocities (owned field, appears twice)
            fu_niw = Float64.(open(joinpath(base2, "UVW_NIW", "fu_niw_$suffix.bin"), "r") do io
                reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
            end)
            fv_niw = Float64.(open(joinpath(base2, "UVW_NIW", "fv_niw_$suffix.bin"), "r") do io
                reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
            end)

            DRFfull = hFacC .* DRF3d
            DRFfull[hFacC .== 0] .= 0.0

            G_vel_3day = zeros(Float64, nx, ny, nt3)
            hrs_per_chunk = 3 * 24

            for t in 1:nt3
                t_start = (t-1) * hrs_per_chunk + 1
                t_end   = min(t * hrs_per_chunk, nt)

                g_vel_temp = zeros(Float64, nx, ny, t_end - t_start + 1)

                for idx in 1:(t_end - t_start + 1)
                    t_actual = t_start + idx - 1

                    # IT gradients computed at each timestep
                    fu_t = @view fu[:, :, :, t_actual]
                    fv_t = @view fv[:, :, :, t_actual]
                    fu_x, fu_y, fv_x, fv_y = compute_IT_gradients(fu_t, fv_t, dx, dy, nx, ny, nz)

                    # NIW velocities (owned field)
                    us = @view fu_niw[:, :, :, t_actual]
                    vs = @view fv_niw[:, :, :, t_actual]

                    # G_vel = -rho0 * [us*(us*∂ut/∂x + vs*∂ut/∂y) + vs*(us*∂vt/∂x + vs*∂vt/∂y)] * DRF
                    temp1 = us .* us .* fu_x .* DRFfull
                    temp2 = us .* vs .* fu_y .* DRFfull
                    temp3 = vs .* us .* fv_x .* DRFfull
                    temp4 = vs .* vs .* fv_y .* DRFfull

                    g_vel_temp[:, :, idx] = -rho0 .* dropdims(sum((temp1 .+ temp2 .+ temp3 .+ temp4), dims=3), dims=3)
                end

                G_vel_3day[:, :, t] = mean(g_vel_temp, dims=3)
            end

            open(joinpath(base2, "G_vel_3day", "g_vel_3day_$suffix.bin"), "w") do io
                write(io, Float32.(G_vel_3day))
            end
            println("Completed tile: $suffix")
        end
    end
    println("\n=== All tiles processed successfully (3-day) ===")

elseif time_mode == "weekly"
    println("Starting G_vel calculation for weekly window Apr 22-28...")
    mkpath(joinpath(base2, "G_vel_weekly"))

    for xn in cfg["xn_start"]:cfg["xn_end"]
        for yn in cfg["yn_start"]:cfg["yn_end"]
            suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
            println("\n--- Processing tile: $suffix (weekly) ---")

            hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
            dx    = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
            dy    = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))

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

            DRFfull = hFacC .* DRF3d
            DRFfull[hFacC .== 0] .= 0.0

            g_vel = zeros(Float64, nx, ny, nt_week)

            for idx in 1:nt_week
                fu_t = @view fu[:, :, :, idx]
                fv_t = @view fv[:, :, :, idx]
                fu_x, fu_y, fv_x, fv_y = compute_IT_gradients(fu_t, fv_t, dx, dy, nx, ny, nz)

                us = @view fu_niw[:, :, :, idx]
                vs = @view fv_niw[:, :, :, idx]

                temp1 = us .* us .* fu_x .* DRFfull
                temp2 = us .* vs .* fu_y .* DRFfull
                temp3 = vs .* us .* fv_x .* DRFfull
                temp4 = vs .* vs .* fv_y .* DRFfull

                g_vel[:, :, idx] = -rho0 .* dropdims(sum((temp1 .+ temp2 .+ temp3 .+ temp4), dims=3), dims=3)
            end

            G_vel = dropdims(mean(g_vel, dims=3), dims=3)

            open(joinpath(base2, "G_vel_weekly", "g_vel_weekly_$suffix.bin"), "w") do io
                write(io, Float32.(G_vel))
            end
            println("Completed tile: $suffix")
        end
    end
    println("\n=== All tiles processed successfully (weekly) ===")

elseif time_mode == "full"
    println("Starting G_vel calculation for full time average...")
    mkpath(joinpath(base2, "G_vel_full"))

    for xn in cfg["xn_start"]:cfg["xn_end"]
        for yn in cfg["yn_start"]:cfg["yn_end"]
            suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
            println("\n--- Processing tile: $suffix (full) ---")

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

            DRFfull = hFacC .* DRF3d
            DRFfull[hFacC .== 0] .= 0.0

            g_vel = zeros(Float64, nx, ny, nt)

            println("Calculating G_vel for each timestep...")
            for t in 1:nt
                fu_t = @view fu[:, :, :, t]
                fv_t = @view fv[:, :, :, t]
                fu_x, fu_y, fv_x, fv_y = compute_IT_gradients(fu_t, fv_t, dx, dy, nx, ny, nz)

                us = @view fu_niw[:, :, :, t]
                vs = @view fv_niw[:, :, :, t]

                temp1 = us .* us .* fu_x .* DRFfull
                temp2 = us .* vs .* fu_y .* DRFfull
                temp3 = vs .* us .* fv_x .* DRFfull
                temp4 = vs .* vs .* fv_y .* DRFfull

                g_vel[:, :, t] = -rho0 .* dropdims(sum((temp1 .+ temp2 .+ temp3 .+ temp4), dims=3), dims=3)
            end

            G_vel = dropdims(mean(g_vel, dims=3), dims=3)

            open(joinpath(base2, "G_vel_full", "g_vel_mean_$suffix.bin"), "w") do io
                write(io, Float32.(G_vel))
            end
            println("Completed tile: $suffix")
        end
    end
    println("\n=== All tiles processed successfully (full) ===")

else
    error("Unknown time_mode '$time_mode'. Choose \"3day\", \"weekly\", or \"full\".")
end