using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML, Impute


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
# MAIN WORKFLOW
# ============================================================================


if time_mode == "3day"
    println("Starting G_buoy calculation for $nt3 3-day periods...")
    mkpath(joinpath(base2, "G_buoy_3day"))


    for xn in cfg["xn_start"]:cfg["xn_end"]
        for yn in cfg["yn_start"]:cfg["yn_end"]
            suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
            println("\n--- Processing tile: $suffix (3-day) ---")


            hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
            dx    = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
            dy    = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))


            # --- Background N2 (3-day averaged) — same as your BP code ---
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


            # --- NIW buoyancy (owned field — appears twice) ---
            b_NIW = Float64.(open(joinpath(base2, "b_NIW", "b_niw_$suffix.bin"), "r") do io
                raw = read(io, nx * ny * nz * nt * sizeof(Float32))
                reshape(reinterpret(Float32, raw), nx, ny, nz, nt)
            end)


            # --- NIW velocities (owned field) ---
            fu_niw = Float64.(open(joinpath(base2, "UVW_NIW", "fu_niw_$suffix.bin"), "r") do io
                reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
            end)
            fv_niw = Float64.(open(joinpath(base2, "UVW_NIW", "fv_niw_$suffix.bin"), "r") do io
                reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
            end)


            DRFfull = hFacC .* DRF3d
            DRFfull[hFacC .== 0] .= 0.0


            G_buoy_3day = zeros(Float64, nx, ny, nt3)
            hrs_per_chunk = 3 * 24


            for t in 1:nt3
                t_start = (t-1) * hrs_per_chunk + 1
                t_end   = min(t * hrs_per_chunk, nt)


                g_buoy_temp = zeros(Float64, nx, ny, t_end - t_start + 1)


                for idx in 1:(t_end - t_start + 1)
                    t_actual = t_start + idx - 1
                    t_avg    = min(div(t_actual - 1, ts) + 1, nt_avg)


                    # IT buoyancy gradients at this timestep
                    b_IT_t = @view b_IT[:, :, :, t_actual]
                    b_IT_x, b_IT_y = compute_IT_buoyancy_gradients(b_IT_t, dx, dy, hFacC, nx, ny, nz)


                    # Background N2
                    n2_val = @view N2_center[:, :, :, t_avg]


                    # NIW owned fields
                    b_s  = @view b_NIW[:, :, :, t_actual]
                    us   = @view fu_niw[:, :, :, t_actual]
                    vs   = @view fv_niw[:, :, :, t_actual]


                    # G_buoy = -(b_NIW/N2)(u_NIW*∂b_IT/∂x + v_NIW*∂b_IT/∂y) * DRF
                    temp1 = (b_s ./ n2_val) .* us .* b_IT_x .* DRFfull
                    temp2 = (b_s ./ n2_val) .* vs .* b_IT_y .* DRFfull


                    temp1[isnan.(temp1)] .= 0.0
                    temp2[isnan.(temp2)] .= 0.0


                    g_buoy_temp[:, :, idx] = -rho0 .* dropdims(sum(temp1 .+ temp2, dims=3), dims=3)
                end


                G_buoy_3day[:, :, t] = mean(g_buoy_temp, dims=3)
            end


            println("  G_buoy range: $(extrema(G_buoy_3day[isfinite.(G_buoy_3day)]))")


            open(joinpath(base2, "G_buoy_3day", "g_buoy_3day_$suffix.bin"), "w") do io
                write(io, Float32.(G_buoy_3day))
            end
            println("Completed tile: $suffix")
        end
    end
    println("\n=== All tiles processed successfully (3-day) ===")


elseif time_mode == "weekly"
    println("Starting G_buoy calculation for weekly window Apr 22-28...")
    mkpath(joinpath(base2, "G_buoy_weekly"))


    for xn in cfg["xn_start"]:cfg["xn_end"]
        for yn in cfg["yn_start"]:cfg["yn_end"]
            suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
            println("\n--- Processing tile: $suffix (weekly) ---")


            hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
            dx    = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
            dy    = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))


            # --- Background N2 (3-day averaged, full record) ---
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


            # --- IT buoyancy (differentiated field) — subset to weekly ---
            b_IT = Float64.(open(joinpath(base2, "b", "b_t_sm_$suffix.bin"), "r") do io
                raw = read(io, nx * ny * nz * nt * sizeof(Float32))
                reshape(reinterpret(Float32, raw), nx, ny, nz, nt)
            end)[:, :, :, idx_start:idx_end]


            # --- NIW buoyancy (owned field) — subset to weekly ---
            b_NIW = Float64.(open(joinpath(base2, "b_NIW", "b_niw_$suffix.bin"), "r") do io
                raw = read(io, nx * ny * nz * nt * sizeof(Float32))
                reshape(reinterpret(Float32, raw), nx, ny, nz, nt)
            end)[:, :, :, idx_start:idx_end]


            fu_niw = Float64.(open(joinpath(base2, "UVW_NIW", "fu_niw_$suffix.bin"), "r") do io
                reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
            end)[:, :, :, idx_start:idx_end]


            fv_niw = Float64.(open(joinpath(base2, "UVW_NIW", "fv_niw_$suffix.bin"), "r") do io
                reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
            end)[:, :, :, idx_start:idx_end]


            DRFfull = hFacC .* DRF3d
            DRFfull[hFacC .== 0] .= 0.0


            g_buoy = zeros(Float64, nx, ny, nt_week)


            for idx in 1:nt_week
                t_actual = idx_start + idx - 1
                t_avg    = min(div(t_actual - 1, ts) + 1, nt_avg)


                b_IT_t = @view b_IT[:, :, :, idx]
                b_IT_x, b_IT_y = compute_IT_buoyancy_gradients(b_IT_t, dx, dy, hFacC, nx, ny, nz)


                n2_val = @view N2_center[:, :, :, t_avg]
                b_s    = @view b_NIW[:, :, :, idx]
                us     = @view fu_niw[:, :, :, idx]
                vs     = @view fv_niw[:, :, :, idx]


                temp1 = (b_s ./ n2_val) .* us .* b_IT_x .* DRFfull
                temp2 = (b_s ./ n2_val) .* vs .* b_IT_y .* DRFfull


                temp1[isnan.(temp1)] .= 0.0
                temp2[isnan.(temp2)] .= 0.0


                g_buoy[:, :, idx] = -rho0 .* dropdims(sum(temp1 .+ temp2, dims=3), dims=3)
            end


            G_buoy = dropdims(mean(g_buoy, dims=3), dims=3)
            println("  G_buoy range: $(extrema(G_buoy[isfinite.(G_buoy)]))")


            open(joinpath(base2, "G_buoy_weekly", "g_buoy_weekly_$suffix.bin"), "w") do io
                write(io, Float32.(G_buoy))
            end
            println("Completed tile: $suffix")
        end
    end
    println("\n=== All tiles processed successfully (weekly) ===")


elseif time_mode == "full"
    println("Starting G_buoy calculation for full time average...")
    mkpath(joinpath(base2, "G_buoy_full"))


    for xn in cfg["xn_start"]:cfg["xn_end"]
        for yn in cfg["yn_start"]:cfg["yn_end"]
            suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
            println("\n--- Processing tile: $suffix (full) ---")


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


            fu_niw = Float64.(open(joinpath(base2, "UVW_NIW", "fu_niw_$suffix.bin"), "r") do io
                reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
            end)
            fv_niw = Float64.(open(joinpath(base2, "UVW_NIW", "fv_niw_$suffix.bin"), "r") do io
                reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
            end)


            DRFfull = hFacC .* DRF3d
            DRFfull[hFacC .== 0] .= 0.0


            g_buoy = zeros(Float64, nx, ny, nt)


            println("Calculating G_buoy for each timestep...")
            for t in 1:nt
                t_avg = min(div(t - 1, ts) + 1, nt_avg)


                b_IT_t = @view b_IT[:, :, :, t]
                b_IT_x, b_IT_y = compute_IT_buoyancy_gradients(b_IT_t, dx, dy, hFacC, nx, ny, nz)


                n2_val = @view N2_center[:, :, :, t_avg]
                b_s    = @view b_NIW[:, :, :, t]
                us     = @view fu_niw[:, :, :, t]
                vs     = @view fv_niw[:, :, :, t]


                temp1 = (b_s ./ n2_val) .* us .* b_IT_x .* DRFfull
                temp2 = (b_s ./ n2_val) .* vs .* b_IT_y .* DRFfull


                temp1[isnan.(temp1)] .= 0.0
                temp2[isnan.(temp2)] .= 0.0


                g_buoy[:, :, t] = -rho0 .* dropdims(sum(temp1 .+ temp2, dims=3), dims=3)
            end


            G_buoy = dropdims(mean(g_buoy, dims=3), dims=3)
            println("  G_buoy range: $(extrema(G_buoy[isfinite.(G_buoy)]))")


            open(joinpath(base2, "G_buoy_full", "g_buoy_mean_$suffix.bin"), "w") do io
                write(io, Float32.(G_buoy))
            end
            println("Completed tile: $suffix")
        end
    end
    println("\n=== All tiles processed successfully (full) ===")


else
    error("Unknown time_mode '$time_mode'. Choose \"3day\", \"weekly\", or \"full\".")
end




