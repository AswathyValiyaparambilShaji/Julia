using DSP, MAT, Statistics, Printf,  FilePathsBase, LinearAlgebra, TOML
#using CairoMakie, SparseArrays

include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base = cfg["base_path"]
base2 = cfg["base_path2"]

mkpath(joinpath(base2,"xflux"))
mkpath(joinpath(base2, "yflux"))
mkpath(joinpath(base2, "zflux"))

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

# --- Filter (915 day band, 1 step sampling here) ---
T1, T2, delt, N = 9.0, 15.0, 1.0, 4
fcutlow, fcuthigh = 1 / T2, 1 / T1
fnq = 1 / delt
bpf = digitalfilter(Bandpass(fcutlow, fcuthigh), Butterworth(N); fs = fnq)


# Now parallelize over ALL 42 tiles
mkpath(joinpath(base2, "FDiv"))

for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)


        # --- File / tile info ---
        hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz));


        fx = open(joinpath(base2, "xflux", "xflx_$suffix.bin"), "r") do io
            # Calculate the number of bytes needed
            nbytes = nx * ny * nz * sizeof(Float64)
            # Read the raw bytes
            raw_bytes = read(io, nbytes)
            # Reinterpret as Float64 array and reshape
            raw_data = reinterpret(Float64, raw_bytes)
            reshaped_data = reshape(raw_data, nx, ny, nz)
        end


        fy = open(joinpath(base2, "yflux", "yflx_$suffix.bin"), "r") do io
            # Calculate the number of bytes needed
            nbytes = nx * ny * nz * sizeof(Float64)
            # Read the raw bytes
            raw_bytes = read(io, nbytes)
            # Reinterpret as Float64 array and reshape
            raw_data = reinterpret(Float64, raw_bytes)
            reshaped_data = reshape(raw_data, nx, ny, nz)
        end
#=
        fz = open(joinpath(base, "zflux", "zflx_$suffix.bin"), "r") do io
            # Calculate the number of bytes needed
            nbytes = nx * ny * nz * sizeof(Float64)
            # Read the raw bytes
            raw_bytes = read(io, nbytes)
            # Reinterpret as Float64 array and reshape
            raw_data = reinterpret(Float64, raw_bytes)
            reshaped_data = reshape(raw_data, nx, ny, nz)
        end
  =#      
        dx = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"),   (nx, ny))
        dy = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"),   (nx, ny))
        # DRFfull

        DRFfull = hFacC .* DRF3d
        z1 = cumsum(DRFfull, dims=3)
        depth = sum(DRFfull, dims=3)
        DRFfull[hFacC .== 0] .= 0.0
        fxX = sum(fx .* DRFfull, dims=3)  # Integrate over depth (z-axis)
        fyY = sum(fy .* DRFfull, dims=3)
        flxD = zeros(nx-2, ny-2)

        for i in 2:(nx-2)
            for j in 2:(ny-2)
                    dF_x_dx = (fxX[i+1,j] - fxX[i-1,j]) / (dx[i,j] + dx[i-1,j])
                    dF_y_dy = (fyY[i,j+1] - fyY[i,j-1]) / (dy[i,j] + dy[i,j-1])

                    flxD[i,j] = dF_x_dx + dF_y_dy 
                    
                
            end
        end
        
        suffix2 = @sprintf("%02dx%02d_%d", xn, yn, buf-2)

        open(joinpath(base2, "FDiv", "FDiv_$suffix2.bin"), "w") do io; write(io, flxD); end
        
    end
end
