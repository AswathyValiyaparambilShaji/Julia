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
mkpath(joinpath(base2, "BC","SP_V"))
mkpath(joinpath(base2, "BC", "SP_V_3day"))
for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
        println("\n--- Processing tile: $suffix ---")
        # --- Read grid metrics ---
        hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
        # --- Read 3-day averaged mean velocity fields ---
        UF = Float64.(open(joinpath(base, "3day_mean", "U", "ucc_3day_$suffix.bin"), "r") do io
            raw_data = reinterpret(Float32, read(io, nx * ny * nz * nt_avg * sizeof(Float32)))
            reshape(raw_data, nx, ny, nz, nt_avg)
        end)
        VF = Float64.(open(joinpath(base, "3day_mean", "V", "vcc_3day_$suffix.bin"), "r") do io
            raw_data = reinterpret(Float32, read(io, nx * ny * nz * nt_avg * sizeof(Float32)))
            reshape(raw_data, nx, ny, nz, nt_avg)
        end)
        # --- Read fluctuating velocities ---
        fu = Float64.(open(joinpath(base2, "UVW_F", "fu_$suffix.bin"), "r") do io
            raw_data = reinterpret(Float32, read(io, nx * ny * nz * nt * sizeof(Float32)))
            reshape(raw_data, nx, ny, nz, nt)
        end)
        fv = Float64.(open(joinpath(base2, "UVW_F", "fv_$suffix.bin"), "r") do io
            raw_data = reinterpret(Float32, read(io, nx * ny * nz * nt * sizeof(Float32)))
            reshape(raw_data, nx, ny, nz, nt)
        end)
        fw = Float64.(open(joinpath(base2, "UVW_F", "fw_$suffix.bin"), "r") do io
            raw_data = reinterpret(Float32, read(io, nx * ny * nz * nt * sizeof(Float32)))
            reshape(raw_data, nx, ny, nz, nt)
        end)
        # --- Compute baroclinic (barotropic-removed) fluctuating velocities ---
        DRFfull = hFacC .* DRF3d
        depth   = sum(DRFfull, dims=3)
        DRFfull[hFacC .== 0] .= 0.0
        mask3D  = hFacC .== 0
        ucA    = sum(fu .* DRFfull, dims=3) ./ depth
        up_3d  = fu .- ucA
        up_3d[repeat(mask3D, 1, 1, 1, nt)] .= 0.0
        fu = ucA = nothing; GC.gc()
        vcA    = sum(fv .* DRFfull, dims=3) ./ depth
        vp_3d  = fv .- vcA
        vp_3d[repeat(mask3D, 1, 1, 1, nt)] .= 0.0
        fv = vcA = nothing; GC.gc()
        wcA    = sum(fw .* DRFfull, dims=3) ./ depth
        wp_3d  = fw .- wcA
        wp_3d[repeat(mask3D, 1, 1, 1, nt)] .= 0.0
        fw = wcA = nothing; GC.gc()
        # --- Vertical gradients of mean velocities: ∂U/∂z, ∂V/∂z ---
        println("Calculating vertical gradients of mean velocities...")

        # ----- Baroclinic Mean Flow --------
        U = UF .- sum(UF .* DRFfull, dims=3) ./ depth
        V = VF .- sum(VF .* DRFfull, dims=3) ./ depth

        U[repeat(mask3D, 1, 1, 1, nt_avg)] .= 0.0
        V[repeat(mask3D, 1, 1, 1, nt_avg)] .= 0.0

        U_z = zeros(Float64, nx, ny, nz, nt_avg)
        V_z = zeros(Float64, nx, ny, nz, nt_avg)

        for t_avg in 1:nt_avg
            for k in 2:nz-1
                dz = DRF[k-1]/2.0 + DRF[k] + DRF[k+1]/2.0
                U_z[:, :, k, t_avg] = (U[:, :, k-1, t_avg] .- U[:, :, k+1, t_avg]) ./ dz
                V_z[:, :, k, t_avg] = (V[:, :, k-1, t_avg] .- V[:, :, k+1, t_avg]) ./ dz
            end
        end
        # --- Adjust N2 to interfaces ---
        #U_z[:, :, 1,   :] = U_z[:, :, 2,   :]
        #U_z[:, :, nz,  :] = U_z[:, :, nz-1, :]
        #V_z[:, :, 1,   :] = V_z[:, :, 2,   :]
        #V_z[:, :, nz,  :] = V_z[:, :, nz-1, :]

        U_z[repeat(mask3D, 1, 1, 1, nt_avg)] .= 0.0
        V_z[repeat(mask3D, 1, 1, 1, nt_avg)] .= 0.0

        U = V = nothing; GC.gc()
        println("Vertical gradients calculated")
        # --- Calculate vertical shear production at every timestep ---
        println("Calculating vertical shear production...")
        sp_v     = zeros(Float64, nx, ny, nt)
        SP_V_3day = zeros(Float64, nx, ny, nt3)
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
        up_3d = vp_3d = wp_3d = U_z = V_z = nothing; GC.gc()
        println("Vertical shear production calculation complete")
        # --- Full-record time average ---
        SP_V = dropdims(mean(sp_v, dims=3), dims=3)
        println(SP_V[20, 1:10])
        open(joinpath(base2, "BC","SP_V", "sp_v_mean_$suffix.bin"), "w") do io
            write(io, Float32.(SP_V))
        end
        SP_V = nothing; GC.gc()
        # --- 3-day averages ---
        hrs_per_chunk = 3 * 24
        for t in 1:nt3
            t_start = (t-1) * hrs_per_chunk + 1
            t_end   = min(t * hrs_per_chunk, nt)
            SP_V_3day[:, :, t] = mean(sp_v[:, :, t_start:t_end], dims=3)
        end
        sp_v = nothing; GC.gc()
        open(joinpath(base2,"BC", "SP_V_3day", "sp_v_3day_$suffix.bin"), "w") do io
            write(io, Float32.(SP_V_3day))
        end
        SP_V_3day = nothing; GC.gc()
        println("Completed tile: $suffix")
        println("Outputs saved to SP_V/ and SP_V_3day/")
    end
end
println("\n=== All tiles processed successfully ===")




