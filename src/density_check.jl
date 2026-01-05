#=using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML
using CairoMakie


include(joinpath(@__DIR__,"..", "functions", "densjmd95.jl"))
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)


base  = cfg["base_path"]
base2 = cfg["base_path2"]


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


dt  = 25
dto = 144
Tts = 366192
nt  = div(Tts, dto)


ts      = 72      # 3-day window
nt_avg = div(nt, ts)


# --- Thickness & constants ---
thk = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)


# Initialize variables outside the loop
rho_insitu = nothing
z_center = nothing
DRFfull = nothing


for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]


        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
        
        # --- Read fields ---
        Salt = open(joinpath(base, "3day_mean", "Salt", "salt_3day_$suffix.bin"), "r") do io
            nbytes = nx * ny * nz * nt_avg * sizeof(Float32)
            raw_bytes = read(io, nbytes)
            raw_data = reinterpret(Float32, raw_bytes)
            reshaped_data = reshape(raw_data, nx, ny, nz, nt_avg)
            convert(Array{Float64,4}, reshaped_data)
        end


        Theta = open(joinpath(base, "3day_mean", "Theta", "theta_3day_$suffix.bin"), "r") do io
            nbytes = nx * ny * nz * nt_avg * sizeof(Float32)
            raw_bytes = read(io, nbytes)
            raw_data = reinterpret(Float32, raw_bytes)
            reshaped_data = reshape(raw_data, nx, ny, nz, nt_avg)
            convert(Array{Float64,4}, reshaped_data)
        end


        hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))


        # --- Calculate depth at cell centers ---
        DRFfull = hFacC .* DRF3d
        DRFfull[hFacC .== 0] .= NaN


        # Cumulative depth (absolute value, positive downward)
        z = cumsum(DRFfull, dims=3)
        zz= cat(zeros(nx, ny, 1),z; dims=3)

        # Adjust to get depth at cell center (mid-point of each cell)
        z_center = -0.5 .* (zz[:, :, 1:end-1] .+ zz[:, :, 2:end])



        # --- Calculate in-situ density at cell centers ---
        println("Calculating in-situ density at cell centers...")
        rho_insitu = zeros(Float64, nx, ny, nz, nt_avg)


        for t in 1:nt_avg
            S_t = Salt[:, :, :, t]
            T_t = Theta[:, :, :, t]
            
            # Calculate density at cell centers using depth at cell centers
            rho_insitu[:, :, :, t] = densjmd95(S_t, T_t, -z_center)
        end


        println("In-situ density calculation complete.")
    end
end


# --- Select a point (i, j) ---
i = 25  # x-index
j = 25  # y-index
t = 1   # time index


# --- Extract profiles at this point ---
rho_profile = rho_insitu[i, j, :, t]
z_profile = z_center[i, j, :]


# --- Calculate pressure at this point ---
g = 9.81  # gravitational acceleration (m/s²)
pres_profile = g .* cumsum(rho_profile .* DRFfull[i, j, :])


# --- Plot ---
fig = Figure(size = (1200, 800))


ax1 = Axis(fig[1, 1],
           xlabel = "Density (kg/m³)",
           ylabel = "Depth (m)",
           title = "In-situ Density Profile")
           #limits=(nothing, nothing,-4135,4))
lines!(ax1, rho_profile, z_profile, linewidth = 2)


ax2 = Axis(fig[1, 2],
           xlabel = "Pressure (Pa)",
           ylabel = "Depth (m)",
           title = "Pressure Profile")
           #limits=(nothing, nothing,-4135,4))
lines!(ax2, pres_profile, z_profile, linewidth = 2)


display(fig)




println("z_centers",pres_profile)=#

using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML
using CairoMakie


include(joinpath(@__DIR__,"..", "functions", "densjmd95.jl"))
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)


base  = cfg["base_path"]
base2 = cfg["base_path2"]


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


dt  = 25
dto = 144
Tts = 366192
nt  = div(Tts, dto)


ts      = 72      # 3-day window
nt_avg = div(nt, ts)


# --- Thickness & constants ---
thk = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)


# Initialize variables outside the loop
rho_insitu = nothing
z_center = nothing
DRFfull = nothing


for xn in cfg["xn_start"]:cfg["xn_end"]
   for yn in cfg["yn_start"]:cfg["yn_end"]


       suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
      
       # --- Read fields ---
       Salt = open(joinpath(base, "3day_mean", "Salt", "salt_3day_$suffix.bin"), "r") do io
           nbytes = nx * ny * nz * nt_avg * sizeof(Float32)
           raw_bytes = read(io, nbytes)
           raw_data = reinterpret(Float32, raw_bytes)
           reshaped_data = reshape(raw_data, nx, ny, nz, nt_avg)
           convert(Array{Float64,4}, reshaped_data)
       end


       Theta = open(joinpath(base, "3day_mean", "Theta", "theta_3day_$suffix.bin"), "r") do io
           nbytes = nx * ny * nz * nt_avg * sizeof(Float32)
           raw_bytes = read(io, nbytes)
           raw_data = reinterpret(Float32, raw_bytes)
           reshaped_data = reshape(raw_data, nx, ny, nz, nt_avg)
           convert(Array{Float64,4}, reshaped_data)
       end


       hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))


       # --- Calculate depth at cell centers ---
       DRFfull = hFacC .* DRF3d
       DRFfull[hFacC .== 0] .= NaN


       # Cumulative depth (absolute value, positive downward)
       z = cumsum(DRFfull, dims=3)
       zz = cat(zeros(nx, ny, 1), z; dims=3)


       # Adjust to get depth at cell center (mid-point of each cell)
       z_center = -0.5 .* (zz[:, :, 1:end-1] .+ zz[:, :, 2:end])


       # --- Calculate in-situ density at cell centers ---
       println("Calculating in-situ density at cell centers...")
       rho_insitu = zeros(Float64, nx, ny, nz, nt_avg)


       for t in 1:nt_avg
           S_t = Salt[:, :, :, t]
           T_t = Theta[:, :, :, t]
          
           # Calculate density at cell centers using depth at cell centers
           rho_insitu[:, :, :, t] = densjmd95(S_t, T_t, -z_center)
       end


       println("In-situ density calculation complete.")
   end
end


# --- Select a point (i, j) ---
i = 25  # x-index
j = 25  # y-index
t = 1   # time index


# --- Extract profiles at this point ---
rho_profile = rho_insitu[i, j, :, t]
z_center_profile = z_center[i, j, :]
dz_profile = DRFfull[i, j, :]  # Partial cell thickness at each level


# --- Calculate pressure at cell centers ---
g = 9.81  # gravitational acceleration (m/s²)


# Pressure at cell center k represents the pressure at that depth
# It's calculated by accumulating pressure from cells above
pres_center = zeros(nz)


for k in 1:nz
    if !isnan(rho_profile[k]) && !isnan(dz_profile[k])
        if k == 1
            # For the first cell, pressure at center is due to half the cell thickness
            pres_center[k] = rho_profile[k] * g * (0.5 * dz_profile[k])
        else
            # For subsequent cells:
            # Pressure = pressure from all cells above + half of current cell
            pressure_from_above = sum([rho_profile[kk] * g * dz_profile[kk] 
                                      for kk in 1:(k-1) if !isnan(rho_profile[kk]) && !isnan(dz_profile[kk])])
            pres_center[k] = pressure_from_above + rho_profile[k] * g * (0.5 * dz_profile[k])
        end
    else
        pres_center[k] = NaN
    end
end


# --- Plot ---
fig = Figure(size = (1200, 800))


ax1 = Axis(fig[1, 1],
          xlabel = "Density (kg/m³)",
          ylabel = "Depth (m)",
          title = "In-situ Density Profile")
lines!(ax1, rho_profile, z_center_profile, linewidth = 2, label = "Density at cell centers")
axislegend(ax1, position = :lb)


ax2 = Axis(fig[1, 2],
          xlabel = "Pressure (Pa)",
          ylabel = "Depth (m)",
          title = "Pressure Profile")
lines!(ax2, pres_center, z_center_profile, linewidth = 2, label = "Pressure at cell centers", color = :blue)
axislegend(ax2, position = :lb)


display(fig)


println("Pressure at cell centers: ", pres_center)
println("Depth at cell centers: ", z_center_profile)
println("Partial cell thicknesses: ", dz_profile)




