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
nt_avg = div(nt, ts)     # number of 3-day averaged mean fields
nt3 = div(nt, 3*24)      # number of 3-day periods
# --- Thickness & constants ---
thk = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)
rho0 = 1027.5
g    = 9.8
mkpath(joinpath(base2, "BC"))
mkpath(joinpath(base2, "BC","BP_3day"))
mkpath(joinpath(base2, "BC","BP"))

for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
        println("\n--- Processing tile: $suffix ---")
        # --- Read grid metrics ---
        hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
        dx = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
        dy = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))
        # --- Read density field and mask dry cells ---
        rho = Float64.(open(joinpath(base, "Density", "rho_in_$suffix.bin"), "r") do io
            raw_data = reinterpret(Float64, read(io, nx * ny * nz * nt * sizeof(Float64)))
            reshape(raw_data, nx, ny, nz, nt)
        end)
        for t in 1:nt
            for k in 1:nz
                mask = hFacC[:, :, k] .== 0
                rho[mask, k, t] .= NaN
            end
        end
        # --- Read N2 (3-day averaged) ---
        N2_phase = Float64.(open(joinpath(base, "3day_mean", "N2", "N2_3day_$suffix.bin"), "r") do io
            raw_data = reinterpret(Float32, read(io, nx * ny * nz * nt_avg * sizeof(Float32)))
            reshape(raw_data, nx, ny, nz, nt_avg)
        end)
        # --- Read buoyancy fluctuations ---
        b = Float64.(open(joinpath(base2, "b", "b_t_sm_$suffix.bin"), "r") do io
            raw_data = reinterpret(Float32, read(io, nx * ny * nz * nt * sizeof(Float32)))
            reshape(raw_data, nx, ny, nz, nt)
        end)
        # --- Read fluctuating velocities ---
        UF = Float64.(open(joinpath(base2, "UVW_F", "fu_$suffix.bin"), "r") do io
            raw_data = reinterpret(Float32, read(io, nx * ny * nz * nt * sizeof(Float32)))
            reshape(raw_data, nx, ny, nz, nt)
        end)
        VF = Float64.(open(joinpath(base2, "UVW_F", "fv_$suffix.bin"), "r") do io
            raw_data = reinterpret(Float32, read(io, nx * ny * nz * nt * sizeof(Float32)))
            reshape(raw_data, nx, ny, nz, nt)
        end)
        # --- Compute baroclinic (barotropic-removed) fluctuating velocities ---
        DRFfull = hFacC .* DRF3d
        depth   = sum(DRFfull, dims=3)
        DRFfull[hFacC .== 0] .= 0.0
        mask3D  = hFacC .== 0
        ucA    = sum(UF .* DRFfull, dims=3) ./ depth
        up_3d  = UF .- ucA
        up_3d[repeat(mask3D, 1, 1, 1, nt)] .= 0.0
        UF = ucA = nothing; GC.gc()
        vcA    = sum(VF .* DRFfull, dims=3) ./ depth
        vp_3d  = VF .- vcA
        vp_3d[repeat(mask3D, 1, 1, 1, nt)] .= 0.0
        VF = vcA = nothing; GC.gc()
        # --- Adjust N2 to interfaces ---
        println("Computing N2 at cell centers...")
        N2_adjusted = zeros(Float64, nx, ny, nz+1, nt_avg)
        N2_adjusted[:, :, 1,   :] = N2_phase[:, :, 1,   :]
        N2_adjusted[:, :, 2:nz,:] = N2_phase[:, :, 1:nz-1, :]
        N2_adjusted[:, :, nz+1,:] = N2_phase[:, :, nz-1, :]
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
                N2_adjusted[i, j, kf+1, :] .= N2_phase[i, j, kf-1, :] # k+1 because of the concatenation of addition surface grid
            end
        end
        N2_center = zeros(Float64, nx, ny, nz, nt_avg)
        for k in 1:nz
            N2_center[:, :, k, :] .= 0.5 .* (N2_adjusted[:, :, k, :] .+ N2_adjusted[:, :, k+1, :])
        end
        N2_adjusted = nothing
        N2_phase    = nothing
        # --- Filter out anomalously low N2 values ---
        N2_threshold = 1.0e-8
        N2_center[N2_center .< N2_threshold] .= N2_threshold
        # --- Calculate 3-day averaged mean buoyancy field B ---
        println("Calculating mean buoyancy field...")
        B = zeros(Float64, nx, ny, nz, nt_avg)
        for i in 1:nt_avg
            t_start = (i-1) * ts + 1
            t_end   = min(i * ts, nt)
            for x in 1:nx, y in 1:ny, z in 1:nz
                rho_slice = rho[x, y, z, t_start:t_end]
                valid_rho = rho_slice[isfinite.(rho_slice)]
                if length(valid_rho) > 0
                    B[x, y, z, i] = -g * (mean(valid_rho) - rho0) / rho0
                else
                    B[x, y, z, i] = NaN
                end
            end
        end
        rho = nothing; GC.gc()
        # --- Calculate mean buoyancy gradients: ∂B/∂x, ∂B/∂y ---
        println("Calculating buoyancy gradients...")
        B_x = fill(NaN, nx, ny, nz, nt_avg)
        B_y = fill(NaN, nx, ny, nz, nt_avg)
        for t in 1:nt_avg
            for k in 1:nz
                B_x[2:end-1, :, k, t] .= (B[3:end, :, k, t] .- B[1:end-2, :, k, t]) ./
                                          (dx[2:end-1, :] .+ dx[1:end-2, :])
                B_y[:, 2:end-1, k, t] .= (B[:, 3:end, k, t] .- B[:, 1:end-2, k, t]) ./
                                          (dy[:, 2:end-1] .+ dy[:, 1:end-2])
            end
        end
        for t in 1:nt_avg, k in 1:nz, j in 2:ny-1, i in 2:nx-1
            if hFacC[i-1,j,k] != 1 || hFacC[i,j,k] != 1 || hFacC[i+1,j,k] != 1
                B_x[i, j, k, t] = NaN
            end
            if hFacC[i,j-1,k] != 1 || hFacC[i,j,k] != 1 || hFacC[i,j+1,k] != 1
                B_y[i, j, k, t] = NaN
            end
        end
        B = nothing; GC.gc()
        println("Gradients calculated")
        # --- Calculate buoyancy production at every timestep ---
        println("Calculating buoyancy production...")
        bp      = zeros(Float64, nx, ny, nt)
        BP_3day = zeros(Float64, nx, ny, nt3)
        for t in 1:nt
            t_avg  = min(div(t - 1, ts) + 1, nt_avg)
            n2_val = @view N2_center[:, :, :, t_avg]
            B_x_t  = @view B_x[:, :, :, t_avg]
            B_y_t  = @view B_y[:, :, :, t_avg]
            b_t    = @view b[:, :, :, t]
            ut     = @view up_3d[:, :, :, t]
            vt     = @view vp_3d[:, :, :, t]
            temp1 = (b_t ./ n2_val) .* ut .* B_x_t .* DRFfull
            temp2 = (b_t ./ n2_val) .* vt .* B_y_t .* DRFfull
            temp1[isnan.(temp1)] .= 0.0
            temp2[isnan.(temp2)] .= 0.0
            bp[:, :, t] = -rho0 .* dropdims(sum(temp1 .+ temp2, dims=3), dims=3)
        end
        up_3d = vp_3d = b = N2_center = B_x = B_y = nothing; GC.gc()
        println("Buoyancy production calculation complete")
        # --- Full-record time average ---
        BP = dropdims(mean(bp, dims=3), dims=3)
        println("  BP range: $(extrema(BP[isfinite.(BP)]))")
        open(joinpath(base2, "BC", "bp_mean_$suffix.bin"), "w") do io
            write(io, Float32.(BP))
        end
        BP = nothing; GC.gc()
        # --- 3-day averages ---
        hrs_per_chunk = 3 * 24
        for t in 1:nt3
            t_start = (t-1) * hrs_per_chunk + 1
            t_end   = min(t * hrs_per_chunk, nt)
            BP_3day[:, :, t] = mean(bp[:, :, t_start:t_end], dims=3)
        end
        bp = nothing; GC.gc()
        open(joinpath(base2, "BC", "bp_3day_$suffix.bin"), "w") do io
            write(io, Float32.(BP_3day))
        end
        BP_3day = nothing; GC.gc()
        println("Completed tile: $suffix")
        println("Outputs saved to BC/")
    end
end
println("\n=== All tiles processed successfully ===")




