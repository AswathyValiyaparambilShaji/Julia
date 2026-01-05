using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML
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

#reference density
rho0 = 999.8


# Now parallelize over ALL 42 tiles

mkpath(joinpath(base2, "KE"))
mkpath(joinpath(base2,"b"))

for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]

      suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)

    
        # --- Read fields ---
        rho = Float64.(open(joinpath(base,"Density", "rho_in_$suffix.bin"), "r") do io
            # Calculate the number of bytes needed
            nbytes = nx * ny * nz *nt * sizeof(Float64)
            # Read the raw bytes
            raw_bytes = read(io, nbytes)
            # Reinterpret as Float64 array and reshape
            raw_data = reinterpret(Float64, raw_bytes)
            reshaped_data = reshape(raw_data, nx, ny,nz ,nt)
        end)
        
        hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))

        DRFfull = hFacC .* DRF3d
        z = cumsum(DRFfull, dims=3)
        depth = sum(DRFfull, dims=3)
        DRFfull[hFacC .== 0] .= 0.0

        fu = open(joinpath(base, "SM","UVW_F", "fu_$suffix.bin"), "r") do io
            # Calculate the number of bytes needed
            nbytes = nx * ny * nz *nt * sizeof(Float64)
            # Read the raw bytes
            raw_bytes = read(io, nbytes)
            # Reinterpret as Float64 array and reshape
            raw_data = reinterpret(Float64, raw_bytes)
            reshaped_data = reshape(raw_data, nx, ny,nz ,nt)
        end


        fv = open(joinpath(base,"SM", "UVW_F", "fv_$suffix.bin"), "r") do io
            # Calculate the number of bytes needed
            nbytes = nx * ny * nz *nt * sizeof(Float64)
            # Read the raw bytes
            raw_bytes = read(io, nbytes)
            # Reinterpret as Float64 array and reshape
            raw_data = reinterpret(Float64, raw_bytes)
            reshaped_data = reshape(raw_data, nx, ny, nz, nt)
        end

    # --- Bandpass filter (time is last dim) ---
    fr = bandpassfilter(rho, T1, T2, delt,N,nt)

    # --- KE ---
     
    ke = 0.5 .* rho0 .* (fu.^2+ fv.^2)

    # --- b --- 

    b = (-1 .* g .*(fr))./(rho0)

    
    # --- Save KE & b ---
        

        open(joinpath(base2, "KE", "ke_t_sm_$suffix.bin"), "w") do io
            write(io, ke)
        end#

        open(joinpath(base2, "b", "b_t_sm_$suffix.bin"), "w") do io
            write(io, b)
        end
        
        println("Completed tile: $suffix")
end


end