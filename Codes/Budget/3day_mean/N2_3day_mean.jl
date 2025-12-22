using DSP, MAT, Statistics, Printf, Plots, FilePathsBase, LinearAlgebra, TOML
#using CairoMakie, SparseArrays

include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
include(joinpath(@__DIR__, "..","..","..", "functions", "densjmd95.jl"))

using .FluxUtils: read_bin, bandpassfilter
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

kz = 1
dt = 25
dto = 144
Tts = 366192
nt = div(Tts, dto)
timesteps_per_3days = 72  # 3 timesteps = 72 hours
nt_avg = div(nt, timesteps_per_3days)  

# --- Thickness & constants ---
thk = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)
g = 9.8

# reference density
rho0 = 999.8

# --- Filter (915 day band, 1 step sampling here) ---
T1, T2, delt, N = 9.0, 15.0, 1.0, 4
fcutlow, fcuthigh = 1 / T2, 1 / T1
fnq = 1 / delt
bpf = digitalfilter(Bandpass(fcutlow, fcuthigh), Butterworth(N); fs = fnq)


# Now parallelize over ALL 42 tiles
mkpath(joinpath(base,"3day_mean"))

# Create output directories if they don't exist

mkpath(joinpath(base,"3day_mean", "N2"))
for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]

        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
        
        println("Processing tile: $suffix")


        
        # --- Read fields ---
        Salt = open(joinpath(base,"3day_mean", "Salt", "salt_3day_$suffix.bin"), "r") do io
            # Calculate the number of bytes needed
            nbytes = nx * ny * nz *nt_avg * sizeof(Float32)
            # Read the raw bytes
            raw_bytes = read(io, nbytes)
            # Reinterpret as Float64 array and reshape
            raw_data = reinterpret(Float32, raw_bytes)
            reshaped_data = reshape(raw_data, nx, ny,nz ,nt_avg)
            convert(Array{Float64,4}, reshaped_data)
        end


        Theta = open(joinpath(base,"3day_mean", "Theta", "theta_3day_$suffix.bin"), "r") do io
            # Calculate the number of bytes needed
            nbytes = nx * ny * nz *nt_avg * sizeof(Float32)
            # Read the raw bytes
            raw_bytes = read(io, nbytes)
            # Reinterpret as Float64 array and reshape
            raw_data = reinterpret(Float32, raw_bytes)
            reshaped_data = reshape(raw_data, nx, ny, nz, nt_avg)
            convert(Array{Float64,4}, reshaped_data)
        end
        hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
        
        # --- Calculate depth and pressure ---
        DRFfull = hFacC .* DRF3d
        DRFfull[hFacC .== 0] .= 0.0
        
        z = cumsum(DRFfull, dims=3)
        zz= cat(zeros(nx, ny, 1),z; dims=3)

        p = copy(z)  
        
        za = -0.5 .* (zz[:, :, 1:end-1] .+ zz[:, :, 2:end])
        
        za_4d = repeat(zz, 1, 1, 1, nt)
        zc = za[:, :, 2:end] .- za[:, :, 1:end-1]  # (nx, ny, nz-1)
        Δz = cat(zc, zeros(nx, ny, 1); dims=3)
        
        # --- Initialize N2 array ---
        #mask_3d = (Δz .> 0) .& (hFacC[:, :, 1:end-1] .> 0) .& (hFacC[:, :, 2:end] .> 0)
        N2 = zeros(Float64, nx, ny, nz, nt_avg)
        
        # --- Process each time step separately to avoid broadcasting issues ---
        println("Calculating N² for each time step...")
        for t in 1:nt_avg
        
            S_t = Salt[:, :, :, t]
            T_t = Theta[:, :, :, t]
            
            S_upper = S_t[:, :, 1:end-1]     
            T_upper = T_t[:, :, 1:end-1]     
            
            S_lower = S_t[:, :, 2:end]       
            T_lower = T_t[:, :, 2:end]       
            
            rho_upper = densjmd95(S_upper, T_upper, z[:, :, 1:end-1])
            rho_lower = densjmd95(S_lower, T_lower, z[:, :, 1:end-1])
            
            
            rc = rho_lower .- rho_upper
            Δρ = cat(rc, zeros(nx, ny, 1); dims=3)
            
            N2_t = zeros(Float64, nx, ny, nz)
            #N2_t[mask_3d] = -(g ./ rho0) .* (Δρ[mask_3d] ./ Δz[mask_3d])
            N2_t = -(g ./ rho0) .* (Δρ ./ Δz)
            # Store in 4D array
            N2[:, :, :, t] = N2_t
        end
        


        # --- Save file ---
        outfile = joinpath(base,"3day_mean", "N2", "N2_3day_$suffix.bin")
        open(outfile, "w") do io
            write(io, N2)
        end
        
        println("Completed tile: $suffix")

    end
end

println("\nAll tiles processed successfully!")

