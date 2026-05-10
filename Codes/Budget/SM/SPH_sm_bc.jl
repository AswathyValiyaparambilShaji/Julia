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
mkpath(joinpath(base2, "BC","SP_H"))
mkpath(joinpath(base2, "BC","SP_H_3day"))
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
        # --- Read fluctuating velocities ---
        fu = Float64.(open(joinpath(base2, "UVW_F", "fu_$suffix.bin"), "r") do io
            raw_data = reinterpret(Float32, read(io, nx * ny * nz * nt * sizeof(Float32)))
            reshape(raw_data, nx, ny, nz, nt)
        end)
        fv = Float64.(open(joinpath(base2, "UVW_F", "fv_$suffix.bin"), "r") do io
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
        # --- Horizontal gradients of mean velocities: ∂U/∂x, ∂U/∂y, ∂V/∂x, ∂V/∂y ---
        println("Calculating horizontal gradients of mean velocities...")
        U_x = zeros(Float64, nx, ny, nz, nt_avg)
        U_y = zeros(Float64, nx, ny, nz, nt_avg)
        V_x = zeros(Float64, nx, ny, nz, nt_avg)
        V_y = zeros(Float64, nx, ny, nz, nt_avg)

        # ----- Baroclinic Mean Flow --------
        U = UF .- sum(UF .* DRFfull, dims=3) ./ depth
        V = VF .- sum(VF .* DRFfull, dims=3) ./ depth

        U[repeat(mask3D, 1, 1, 1, nt_avg)] .= 0.0
        V[repeat(mask3D, 1, 1, 1, nt_avg)] .= 0.0        

        dx_avg = dx[2:end-1, :] .+ dx[1:end-2, :]
        U_x[2:end-1, :, :, :] = (U[3:end, :, :, :] .- U[1:end-2, :, :, :]) ./
                                  reshape(dx_avg, nx-2, ny, 1, 1)
        V_x[2:end-1, :, :, :] = (V[3:end, :, :, :] .- V[1:end-2, :, :, :]) ./
                                  reshape(dx_avg, nx-2, ny, 1, 1)
        dy_avg = dy[:, 2:end-1] .+ dy[:, 1:end-2]
        U_y[:, 2:end-1, :, :] = (U[:, 3:end, :, :] .- U[:, 1:end-2, :, :]) ./
                                  reshape(dy_avg, nx, ny-2, 1, 1)
        V_y[:, 2:end-1, :, :] = (V[:, 3:end, :, :] .- V[:, 1:end-2, :, :]) ./
                                  reshape(dy_avg, nx, ny-2, 1, 1)
        U = V = nothing; GC.gc()

        println("Horizontal gradients calculated")
        # --- Calculate horizontal shear production at every timestep ---
        println("Calculating horizontal shear production...")
        sp_h      = zeros(Float64, nx, ny, nt)
        SP_H_3day = zeros(Float64, nx, ny, nt3)
        for t in 1:nt
            t_avg = min(div(t - 1, ts) + 1, nt_avg)
            U_x_t = @view U_x[:, :, :, t_avg]
            U_y_t = @view U_y[:, :, :, t_avg]
            V_x_t = @view V_x[:, :, :, t_avg]
            V_y_t = @view V_y[:, :, :, t_avg]
            ut    = @view up_3d[:, :, :, t]
            vt    = @view vp_3d[:, :, :, t]
            temp1 = ut .* ut .* U_x_t .* DRFfull
            temp2 = ut .* vt .* U_y_t .* DRFfull
            temp3 = ut .* vt .* V_x_t .* DRFfull
            temp4 = vt .* vt .* V_y_t .* DRFfull
            sp_h[:, :, t] = -rho0 .* dropdims(sum((temp1 .+ temp2 .+ temp3 .+ temp4), dims=3), dims=3)
        end
        up_3d = vp_3d = U_x = U_y = V_x = V_y = nothing; GC.gc()
        println("Horizontal shear production calculation complete")
        # --- Full-record time average ---
        SP_H = dropdims(mean(sp_h, dims=3), dims=3)
        println(SP_H[20, 1:10])
        open(joinpath(base2, "BC","SP_H", "sp_h_mean_$suffix.bin"), "w") do io
            write(io, Float32.(SP_H))
        end
        SP_H = nothing; GC.gc()
        # --- 3-day averages ---
        hrs_per_chunk = 3 * 24
        for t in 1:nt3
            t_start = (t-1) * hrs_per_chunk + 1
            t_end   = min(t * hrs_per_chunk, nt)
            SP_H_3day[:, :, t] = mean(sp_h[:, :, t_start:t_end], dims=3)
        end
        sp_h = nothing; GC.gc()
        open(joinpath(base2, "BC","SP_H_3day", "sp_h_3day_$suffix.bin"), "w") do io
            write(io, Float32.(SP_H_3day))
        end
        SP_H_3day = nothing; GC.gc()
        println("Completed tile: $suffix")
        println("Outputs saved to SP_H/ and SP_H_3day/")
    end
end
println("\n=== All tiles processed successfully ===")




