using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML
#using CairoMakie, SparseArrays

include(joinpath(@__DIR__,  "..", "..", "functions", "FluxUtils.jl"))
include(joinpath(@__DIR__, "..","..", "functions", "densjmd95.jl"))

using .FluxUtils: read_bin, bandpassfilter
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base = cfg["bp_box28"]


# --- Domain & grid ---
NX, NY = 384, 336
NZ = 173
minlat, maxlat = -24.5, -18.5
minlon, maxlon = 337.5, 345.4791122715405
lat = range(minlat, maxlat, length=NY)
lon = range(minlon, maxlon, length=NX)

# --- Tile & time ---
buf = 3
tx, ty = 54, 66
nx = tx + 2*buf
ny = ty + 2*buf
nz = 168

kz = 1

nt = 558

# --- Thickness & constants ---
thk =(open(joinpath(base, "hFacC",  "delR.bin"), "r") do io
                raw = read(io,  NZ * sizeof(Float32))
                ntoh.(reshape(reinterpret(Float32, raw), NZ))
            end)

DRF  = thk[1:nz]
sum(thk)
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)
g = 9.81

# Now parallelize over ALL 42 tiles
mkpath(joinpath(base,"Density"))

# Create output directories if they don't exist
za = zeros(Float64, nx,ny,nz)
rho = zeros(Float64, nx, ny, nz, nt)
for xn in cfg["xn_start"]:cfg["xn_e28"]
    for yn in cfg["yn_start"]:cfg["yn_e28"]

        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
        
        println("Processing tile: $suffix")



        
        # --- Read fields ---
        Salt = open(joinpath(base,"Salt", "Salt_v2_$suffix.bin"), "r") do io
            # Calculate the number of bytes needed
            nbytes = nx * ny * nz *nt * sizeof(Float32)
            # Read the raw bytes
            raw_bytes = read(io, nbytes)
            # Reinterpret as Float64 array and reshape
            raw_data = reinterpret(Float32, raw_bytes)
            reshaped_data = reshape(raw_data, nx, ny,nz ,nt)
            convert(Array{Float64,4}, reshaped_data)
        end


        Theta = open(joinpath(base, "Theta", "Theta_v2_$suffix.bin"), "r") do io
            # Calculate the number of bytes needed
            nbytes = nx * ny * nz *nt * sizeof(Float32)
            # Read the raw bytes
            raw_bytes = read(io, nbytes)
            # Reinterpret as Float64 array and reshape
            raw_data = reinterpret(Float32, raw_bytes)
            reshaped_data = reshape(raw_data, nx, ny, nz, nt)
            convert(Array{Float64,4}, reshaped_data)
        end
        hFacC = read_bin(joinpath(base, "hFacC/hFacC_v2_$suffix.bin"), (nx, ny, nz))
        
        # --- Calculate depth and pressure ---
        DRFfull = hFacC .* DRF3d
        DRFfull[hFacC .== 0] .= 0.0
        
        z = cumsum(DRFfull, dims=3)
        zz= cat(zeros(nx, ny, 1),z; dims=3)

        p = copy(z)  
        
        za = -0.5 .* (zz[:, :, 1:end-1] .+ zz[:, :, 2:end])
        #rho = zeros(Float64, nx, ny, nz, nt)
        for t in 1:nt
        
            S_t = Salt[:, :, :, t]
            T_t = Theta[:, :, :, t]
            
            rho1 = densjmd95(S_t, T_t,-za)
            
            rho[:, :, :, t] = rho1
        end
        

        # --- Save file ---
        outfile = joinpath(base,"Density", "rho_in_$suffix.bin")
        open(outfile, "w") do io
            write(io, rho)
        end
        
        println("Completed tile: $suffix")

    end
end

println("\nAll tiles processed successfully!")
