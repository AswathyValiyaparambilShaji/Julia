using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML ,CairoMakie



include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin
include(joinpath(@__DIR__, "..","..","..", "functions", "coriolis_frequency.jl"))


config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base = cfg["base_path"]


# --- Domain & grid ---
NX, NY = 288, 468
minlat, maxlat = 24.0, 31.91
minlon, maxlon = 193.0, 199.0
lat = range(minlat, maxlat, length=NY)
lon = range(minlon, maxlon, length=NX)


# --- Tile & time ---
buf = 3
tx, ty = 47, 66
nx = tx + 2*buf
ny = ty + 2*buf
nz = 88
dt = 25
dto = 144
Tts = 366192
nt = div(Tts, dto)
timesteps_per_3days = 72
nt_avg = div(nt, timesteps_per_3days)


# reference density
rho0 = 999.8


for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]


        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
        println("Processing tile: $suffix")


        # --- Read grid spacing ---
        dx = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
        dy = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))


        # --- Read velocity fields (3-day averaged) ---
        U = Float64.(open(joinpath(base,"3day_mean", "U", "ucc_3day_$suffix.bin"), "r") do io
            nbytes = nx * ny * nz * nt_avg * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nz, nt_avg)
        end)
      
        V = Float64.(open(joinpath(base,"3day_mean","V", "vcc_3day_$suffix.bin"), "r") do io
            nbytes = nx * ny * nz * nt_avg * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nz, nt_avg)
        end)
        
        U_y = zeros(Float64, nx, ny, 1, nt_avg)
        V_x = zeros(Float64, nx, ny, 1, nt_avg)


        # X-gradient: ∂V/∂x (central difference)
        dx_avg = dx[2:end-1, :] .+ dx[1:end-2, :]
        V_x[2:end-1, :, 1, :] = (V[3:end, :, 1, :] .- V[1:end-2, :, 1, :]) ./
                                  reshape(dx_avg, nx-2, ny, 1, 1)
       
        # Y-gradient: ∂U/∂y (central difference)
        dy_avg = dy[:, 2:end-1] .+ dy[:, 1:end-2]
        U_y[:, 2:end-1, 1, :] = (U[:, 3:end, 1, :] .- U[:, 1:end-2, 1, :]) ./
                                  reshape(dy_avg, nx, ny-2, 1, 1)


        # Calculate vorticity (correct sign)
        ζ = V_x - U_y
        VT = ζ[:,:,1,:]
        
        # Normalize by Coriolis frequency
        for j in 1:ny
            global_j = (yn - 1) * ty + j - buf
            if global_j >= 1 && global_j <= NY
                f_rad_s = coriolis_frequency(lat[global_j])
                #f_cpd = f_rad_s * 3600 *24/ (2π)
                VT[:, j, :] ./= f_rad_s
            end
        end
        # --- Save file ---
        outfile = joinpath(base,"3day_mean", "Vorticity", "Vorticity_3day_$suffix.bin")
        mkpath(dirname(outfile))
        open(outfile, "w") do io
            write(io, Float32.(VT))
        end
      
        println("Completed tile: $suffix")
    end
end



