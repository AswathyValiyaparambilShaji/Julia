using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML
#using CairoMakie, SparseArrays


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base = cfg["bp_box27b"]


# --- Domain & grid ---
# --- Domain & grid of 27b ---
NX, NY = 1056, 1026 
minlat, maxlat = -60.0, -48.0
minlon, maxlon = 142.0208530805687, 163.9791469194313
lat = range(minlat, maxlat, length=NY)
lon = range(minlon, maxlon, length=NX)
NZ = 173

# --- Tile & time ---
buf = 3
tx, ty = 150, 146
nx = tx + 2*buf
ny = ty + 2*buf
nz = 170
kz = 1
nt = 558
timesteps_per_3days = 72  # 3 timesteps = 72 hours
nt_avg = div(nt, timesteps_per_3days)  



# Now parallelize over ALL 42 tiles
mkpath(joinpath(base,"3day_mean"))

# Create output directories if they don't exist
mkpath(joinpath(base, "3day_mean", "U"))
mkpath(joinpath(base,"3day_mean", "V"))
mkpath(joinpath(base,"3day_mean", "Salt"))
mkpath(joinpath(base,"3day_mean", "Theta"))
for xn in cfg["xn_start"]:cfg["xn_e27b"]
    for yn in cfg["yn_start"]:cfg["yn_e27b"]


        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)

        
        # --- Read fields ---
        
        U     = read_bin(joinpath(base, "U/U_v2_$suffix.bin"),   (nx, ny, nz, nt))
        V     = read_bin(joinpath(base, "V/V_v2_$suffix.bin"),   (nx, ny, nz, nt))
        Salt     = read_bin(joinpath(base, "Salt/Salt_v2_$suffix.bin"),   (nx, ny, nz, nt))
        Theta     = read_bin(joinpath(base, "Theta/Theta_v2_$suffix.bin"),   (nx, ny, nz, nt))

        # C-grid to centers
        uc = 0.5 .* (U[1:end-1, :, :, :] .+ U[2:end,   :, :, :])
        vc = 0.5 .* (V[:, 1:end-1, :, :] .+ V[:, 2:end, :, :])

        ucc = cat(uc, zeros(1, ny, nz, nt); dims=1)
        vcc = cat(vc, zeros(nx, 1, nz, nt); dims=2)

        # --- 3-day averaging ---
        ucc_3day = zeros(Float32, nx, ny, nz, nt_avg)
        vcc_3day = zeros(Float32, nx, ny, nz, nt_avg)
        salt_3day = zeros(Float32, nx, ny, nz, nt_avg)
        theta_3day = zeros(Float32, nx, ny, nz, nt_avg)
        
        for i in 1:nt_avg
            t_start = (i-1) * timesteps_per_3days + 1
            t_end = min(i * timesteps_per_3days, nt)
            #println("Start :",t_start)
            #println(t_end)
            
            # Average over 3-day window
            ucc_3day[:, :, :, i] = mean(ucc[:, :, :, t_start:t_end], dims=4)[:, :, :, 1]
            vcc_3day[:, :, :, i] = mean(vcc[:, :, :, t_start:t_end], dims=4)[:, :, :, 1]
            salt_3day[:, :, :, i] = mean(Salt[:, :, :, t_start:t_end], dims=4)[:, :, :, 1]
            theta_3day[:, :, :, i] = mean(Theta[:, :, :, t_start:t_end], dims=4)[:, :, :, 1]
        end
        
        # --- Save to binary files ---
        ucc_file = joinpath(base, "3day_mean","U/ucc_3day_$suffix.bin")
        vcc_file = joinpath(base, "3day_mean","V/vcc_3day_$suffix.bin")
        salt_file = joinpath(base, "3day_mean","Salt/salt_3day_$suffix.bin")
        theta_file = joinpath(base,"3day_mean", "Theta/theta_3day_$suffix.bin")
       
        write(ucc_file, Float32.(ucc_3day))
        write(vcc_file, Float32.(vcc_3day))
        write(salt_file, Float32.(salt_3day))
        write(theta_file, Float32.(theta_3day))


        # Clear memory
        U = nothing
        V = nothing
        ucc = nothing
        vcc = nothing
        ucc_3day = nothing
        vcc_3day = nothing
        Salt = nothing
        salt_3day = nothing
        theta_3day = nothing
        Theta = nothing
        GC.gc()




    end
end 
