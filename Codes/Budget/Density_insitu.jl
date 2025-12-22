using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML
#using CairoMakie, SparseArrays

include(joinpath(@__DIR__, "..","..", "functions", "FluxUtils.jl"))
include(joinpath(@__DIR__, "..","..", "functions", "densjmd95.jl"))

using .FluxUtils: read_bin, bandpassfilter
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..","..", "config", "run_debug.toml"))
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

# --- Thickness & constants ---
thk = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)
g = 9.8

# Now parallelize over ALL 42 tiles
mkpath(joinpath(base,"Density"))

# Create output directories if they don't exist

for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]

        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
        
        println("Processing tile: $suffix")



        
        # --- Read fields ---
        Salt = open(joinpath(base,"Salt", "Salt_$suffix.bin"), "r") do io
            # Calculate the number of bytes needed
            nbytes = nx * ny * nz *nt * sizeof(Float32)
            # Read the raw bytes
            raw_bytes = read(io, nbytes)
            # Reinterpret as Float64 array and reshape
            raw_data = reinterpret(Float32, raw_bytes)
            reshaped_data = reshape(raw_data, nx, ny,nz ,nt)
            convert(Array{Float64,4}, reshaped_data)
        end


        Theta = open(joinpath(base, "Theta", "Theta_$suffix.bin"), "r") do io
            # Calculate the number of bytes needed
            nbytes = nx * ny * nz *nt * sizeof(Float32)
            # Read the raw bytes
            raw_bytes = read(io, nbytes)
            # Reinterpret as Float64 array and reshape
            raw_data = reinterpret(Float32, raw_bytes)
            reshaped_data = reshape(raw_data, nx, ny, nz, nt)
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
        rho = zeros(Float64, nx, ny, nz, nt)
        for t in 1:nt
        
            S_t = Salt[:, :, :, t]
            T_t = Theta[:, :, :, t]
            
            rho1 = densjmd95(S_t, T_t, za)
            
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

