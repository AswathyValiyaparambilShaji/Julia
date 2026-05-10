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
mkpath(joinpath(base2, "BC"))
mkpath(joinpath(base2, "BC", "U_PE"))
mkpath(joinpath(base2, "BC", "U_PE_3day"))
for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
        println("\n--- Processing tile: $suffix ---")
        # --- Read grid metrics ---
        hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
        dx = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
        dy = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))
        # --- Read 3-day averaged mean velocity fields ---
        UF = Float64.(open(joinpath(base, "3day_mean", "U", "ucc_3day_$suffix.bin"), "r") do io
            raw_data = reinterpret(Float32, read(io, nx * ny * nz * nt_avg * sizeof(Float32)))
            reshape(raw_data, nx, ny, nz, nt_avg)
        end)
        VF = Float64.(open(joinpath(base, "3day_mean", "V", "vcc_3day_$suffix.bin"), "r") do io
            raw_data = reinterpret(Float32, read(io, nx * ny * nz * nt_avg * sizeof(Float32)))
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
            raw_data = reinterpret(Float32, read(io, nx * ny * nz * nt_avg * sizeof(Float32)))
            reshape(raw_data, nx, ny, nz, nt_avg)
        end)
        # --- Compute baroclinic (barotropic-removed) mean velocities ---
        DRFfull = hFacC .* DRF3d
        depth   = sum(DRFfull, dims=3)
        DRFfull[hFacC .== 0] .= 0.0
        mask3D  = hFacC .== 0
        ucA = sum(UF .* DRFfull, dims=3) ./ depth
        U   = UF .- ucA
        U[repeat(mask3D, 1, 1, 1, nt_avg)] .= 0.0
        UF = ucA = nothing; GC.gc()
        vcA = sum(VF .* DRFfull, dims=3) ./ depth
        V   = VF .- vcA
        V[repeat(mask3D, 1, 1, 1, nt_avg)] .= 0.0
        VF = vcA = nothing; GC.gc()
        # --- Adjust N2 to nz+1 levels (interfaces) then back to centers ---
        println("Computing N2 at cell centers...")
        N2_adjusted = zeros(Float64, nx, ny, nz+1, nt_avg)
        N2_adjusted[:, :, 1,   :]  = N2[:, :, 1,      :]
        N2_adjusted[:, :, 2:nz,:] = N2[:, :, 1:nz-1, :]
        N2_adjusted[:, :, nz+1,:] = N2[:, :, nz,     :]
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
                N2_adjusted[i, j, kf+1, :] .= N2[i, j, kf-1, :] # k+1 because of the concatenation of addition surface grid
            end
        end
        N2_center = zeros(Float64, nx, ny, nz, nt_avg)
        for k in 1:nz
            N2_center[:, :, k, :] .= 0.5 .* (N2_adjusted[:, :, k, :] .+ N2_adjusted[:, :, k+1, :])
        end
        N2_adjusted = nothing
        N2 = nothing
        # --- Filter out anomalously low N2 values ---
        N2_threshold = 1.0e-8
        N2_center[N2_center .< N2_threshold] .= N2_threshold
        # --- Calculate PE gradients (vectorized over full record) ---
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
        # --- Calculate advective PE flux at every timestep ---
        println("Calculating advective PE flux...")
        u_pe      = zeros(Float64, nx, ny, nt)
        U_PE_3day = zeros(Float64, nx, ny, nt3)
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
            u_pe[:, :, t] = rho0 .* dropdims(sum((temp1 .+ temp2) .* DRFfull, dims=3), dims=3)
        end
        U = V = N2_center = pe_x = pe_y = nothing; GC.gc()
        println("Flux calculation complete")
        # --- Full-record time average ---
        U_PE = dropdims(mean(u_pe, dims=3), dims=3)
        open(joinpath(base2, "BC", "U_PE", "u_pe_mean_$suffix.bin"), "w") do io
            write(io, Float32.(U_PE))
        end
        U_PE = nothing; GC.gc()
        # --- 3-day averages ---
        hrs_per_chunk = 3 * 24
        for t in 1:nt3
            t_start = (t-1) * hrs_per_chunk + 1
            t_end   = min(t * hrs_per_chunk, nt)
            U_PE_3day[:, :, t] = mean(u_pe[:, :, t_start:t_end], dims=3)
        end
        u_pe = nothing; GC.gc()
        open(joinpath(base2, "BC", "U_PE_3day", "u_pe_3day_$suffix.bin"), "w") do io
            write(io, Float32.(U_PE_3day))
        end
        U_PE_3day = nothing; GC.gc()
        println("Completed tile: $suffix")
        println("Outputs saved to BC/U_PE/ and BC/U_PE_3day/")
    end
end
println("\n=== All tiles processed successfully ===")



