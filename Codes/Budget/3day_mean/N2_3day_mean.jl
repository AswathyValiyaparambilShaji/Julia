using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML
using Impute


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
timesteps_per_3days = 72
nt_avg = div(nt, timesteps_per_3days) 


# --- Thickness & constants ---
thk = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)
g = 9.8
rho0 = 999.8


# Create output directory
mkpath(joinpath(base,"3day_mean", "N2"))


for xn in cfg["xn_start"]:cfg["xn_end"]
   for yn in cfg["yn_start"]:cfg["yn_end"]


       suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
      
       println("Processing tile: $suffix")
      
       # --- Read fields ---
       Salt = Float64.(open(joinpath(base,"3day_mean", "Salt", "salt_3day_$suffix.bin"), "r") do io
           nbytes = nx * ny * nz *nt_avg * sizeof(Float32)
           raw_bytes = read(io, nbytes)
           raw_data = reinterpret(Float32, raw_bytes)
           reshaped_data = reshape(raw_data, nx, ny, nz, nt_avg)
           convert(Array{Float64,4}, reshaped_data)
       end)


       Theta = open(joinpath(base,"3day_mean", "Theta", "theta_3day_$suffix.bin"), "r") do io
           nbytes = nx * ny * nz *nt_avg * sizeof(Float32)
           raw_bytes = read(io, nbytes)
           raw_data = reinterpret(Float32, raw_bytes)
           reshaped_data = reshape(raw_data, nx, ny, nz, nt_avg)
           convert(Array{Float64,4}, reshaped_data)
       end
       
       hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
      
       # --- Calculate depths ---
       DRFfull = hFacC .* DRF3d
       DRFfull[hFacC .== 0] .= 0.0
      
       z_cumsum = cumsum(DRFfull, dims=3)
       zz = cat(zeros(nx, ny, 1), z_cumsum; dims=3)  # (nx, ny, nz+1)
       z_centers = -0.5 .* (zz[:, :, 1:end-1] .+ zz[:, :, 2:end])  # (nx, ny, nz)
       
       z_interfaces = -zz[:, :, 2:end-1]  # (nx, ny, nz-1)
       Δz = z_centers[:, :, 2:end] .- z_centers[:, :, 1:end-1]  # (nx, ny, nz-1)
      
       # --- Initialize N2 array ---
       N2 = zeros(Float64, nx, ny, nz, nt_avg)
      
       # --- Calculate N² at interfaces ---
       println("Calculating N² at interfaces...")
       for t in 1:nt_avg
           S_t = Salt[:, :, :, t]
           T_t = Theta[:, :, :, t]
          
           S_upper = S_t[:, :, 1:end-1]    
           T_upper = T_t[:, :, 1:end-1]    
          
           S_lower = S_t[:, :, 2:end]      
           T_lower = T_t[:, :, 2:end]      
          
           # CRITICAL: Reference both densities to the INTERFACE depth
           rho_upper = densjmd95(S_upper, T_upper, z_interfaces)
           rho_lower = densjmd95(S_lower, T_lower, z_interfaces)
          
           # Calculate density difference
           Δρ = rho_lower .- rho_upper
          
           # Calculate N² at interfaces (nz-1 values)
           N2_interfaces = -(g / rho0) .* (Δρ ./ Δz)
           
           # Store in array (put interface values at lower cell centers)
           N2[:, :, 2:end, t] = N2_interfaces
           
           # Handle the first level
           N2[:, :, 1, t] = N2_interfaces[:, :, 1]
       end
      
       # --- Set negative values to NaN ---
       println("Setting negative values to NaN...")
       N2[N2 .< 0] .= NaN
       
       n_nan_before = sum(isnan.(N2))
       println("  Number of NaN values before filling: $n_nan_before")
       
       # --- Fill NaN values using Impute.jl ---
       println("Filling NaN values with nearest neighbor...")
       
       # Fill along depth dimension (dim=3) for each (i,j,t) profile
       # Using nearest neighbor (forward fill then backward fill)
       for t in 1:nt_avg
           for j in 1:ny
               for i in 1:nx
                   profile = N2[i, j, :, t]
                   if any(isnan, profile)
                       # Forward fill (carry last observation forward)
                       profile_ff = Impute.locf(profile)
                       # Backward fill (carry next observation backward) 
                       profile_filled = Impute.nocb(profile_ff)
                       N2[i, j, :, t] = profile_filled
                   end
               end
           end
       end
       
       n_nan_after = sum(isnan.(N2))
       n_negative_after = sum(N2 .< 0)
       println("  Number of NaN values after filling: $n_nan_after")
       println("  Number of negative values after filling: $n_negative_after")
       println("  N² range: ", extrema(filter(isfinite, N2)))


       # --- Save file ---
       outfile = joinpath(base,"3day_mean", "N2", "N2_3day_$suffix.bin")
       open(outfile, "w") do io
           write(io, Float32.(N2))
       end
      
       println("Completed tile: $suffix")
   end
end


println("\nAll tiles processed successfully!")




