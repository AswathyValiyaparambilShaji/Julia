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
mkpath(joinpath(base2, "BC", "U_KE"))
mkpath(joinpath(base2, "BC", "U_KE_3day"))
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
        # --- Read KE fluctuations (full temporal resolution) ---
        ke = Float64.(open(joinpath(base2, "KE", "ke_t_sm_$suffix.bin"), "r") do io
            raw_data = reinterpret(Float32, read(io, nx * ny * nz * nt * sizeof(Float32)))
            reshape(raw_data, nx, ny, nz, nt)
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
        # --- Calculate KE gradients ---
        println("Calculating KE gradients...")
        ke_x = zeros(Float64, nx, ny, nz, nt)
        ke_y = zeros(Float64, nx, ny, nz, nt)
        dx_avg = dx[2:end-1, :] .+ dx[1:end-2, :]
        ke_x[2:end-1, :, :, :] = (ke[3:end, :, :, :] .- ke[1:end-2, :, :, :]) ./
                                   reshape(dx_avg, nx-2, ny, 1, 1)
        dy_avg = dy[:, 2:end-1] .+ dy[:, 1:end-2]
        ke_y[:, 2:end-1, :, :] = (ke[:, 3:end, :, :] .- ke[:, 1:end-2, :, :]) ./
                                   reshape(dy_avg, nx, ny-2, 1, 1)
        ke = nothing; GC.gc()
        println("Gradients calculated")
        # --- Calculate advective KE flux at every timestep ---
        println("Calculating advective KE flux...")
        u_ke      = zeros(Float64, nx, ny, nt)
        U_KE_3day = zeros(Float64, nx, ny, nt3)
        for t in 1:nt
            t_avg  = min(div(t - 1, ts) + 1, nt_avg)
            u_avg  = @view U[:, :, :, t_avg]
            v_avg  = @view V[:, :, :, t_avg]
            ke_x_t = @view ke_x[:, :, :, t]
            ke_y_t = @view ke_y[:, :, :, t]
            temp1 = u_avg .* ke_x_t
            temp2 = v_avg .* ke_y_t
            temp1[isnan.(temp1)] .= 0.0
            temp2[isnan.(temp2)] .= 0.0
            u_ke[:, :, t] =  dropdims(sum((temp1 .+ temp2) .* DRFfull, dims=3), dims=3)
        end
        U = V = ke_x = ke_y = nothing; GC.gc()
        println("Advective KE flux calculation complete")
        # --- Full-record time average ---
        U_KE = dropdims(mean(u_ke, dims=3), dims=3)
        println(U_KE[20, 1:10])
        open(joinpath(base2, "BC", "U_KE", "u_ke_mean_$suffix.bin"), "w") do io
            write(io, Float32.(U_KE))
        end
        U_KE = nothing; GC.gc()
        # --- 3-day averages ---
        hrs_per_chunk = 3 * 24
        for t in 1:nt3
            t_start = (t-1) * hrs_per_chunk + 1
            t_end   = min(t * hrs_per_chunk, nt)
            U_KE_3day[:, :, t] = mean(u_ke[:, :, t_start:t_end], dims=3)
        end
        u_ke = nothing; GC.gc()
        open(joinpath(base2, "BC", "U_KE_3day", "u_ke_3day_$suffix.bin"), "w") do io
            write(io, Float32.(U_KE_3day))
        end
        U_KE_3day = nothing; GC.gc()
        println("Completed tile: $suffix")
        println("Outputs saved to BC/U_KE/ and BC/U_KE_3day/")
    end
end
println("\n=== All tiles processed successfully ===")




