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

# --- Filter (915 day band, 1 step sampling here) ---
T1, T2, delt, N = 9.0, 15.0, 1.0, 4
fcutlow, fcuthigh = 1 / T2, 1 / T1
fnq = 1 / delt
bpf = digitalfilter(Bandpass(fcutlow, fcuthigh), Butterworth(N); fs = fnq)


# Now parallelize over ALL 42 tiles
mkpath(joinpath(base2, "Conv"))


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

        fu = Float64.(open(joinpath(base2, "UVW_F", "fu_$suffix.bin"), "r") do io
            # Calculate the number of bytes needed
            nbytes = nx * ny * nz *nt * sizeof(Float32)
            # Read the raw bytes
            raw_bytes = read(io, nbytes)
            # Reinterpret as Float64 array and reshape
            raw_data = reinterpret(Float32, raw_bytes)
            reshaped_data = reshape(raw_data, nx, ny,nz ,nt)
        end)


        fv = Float64.(open(joinpath(base2, "UVW_F", "fv_$suffix.bin"), "r") do io
            # Calculate the number of bytes needed
            nbytes = nx * ny * nz *nt * sizeof(Float32)
            # Read the raw bytes
            raw_bytes = read(io, nbytes)
            # Reinterpret as Float64 array and reshape
            raw_data = reinterpret(Float32, raw_bytes)
            reshaped_data = reshape(raw_data, nx, ny, nz, nt)
        end)
        fr = bandpassfilter(rho, T1, T2, delt,N,nt)
        
        UDA = dropdims(sum(fu.*  DRFfull, dims=3)./ depth;dims=3)
        VDA = dropdims(sum(fv.*  DRFfull, dims=3)./ depth;dims=3);
        # --- Pressure & perturbations ---
        pres  = g .* cumsum(fr .* DRFfull, dims=3)
        pfz   = cat(zeros(nx, ny, 1, nt), pres; dims=3)
        pc_3d = 0.5 .* (pfz[:, :, 1:end-1, :] .+ pfz[:, :, 2:end, :])
        pa    = sum(pc_3d .* DRFfull, dims=3) ./ depth
        pp_3d = pc_3d .- pa;

        dx = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"),   (nx, ny))
        dy = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"),   (nx, ny));


        H = depth;
        # ---- Bottom level of pc_3d: p_b (nx, ny, nt)
        pb = pp_3d[:, :, end, :];

        dHdx = zeros(nx-2, ny)
        dHdx[:, :] .= (H[3:nx, :] .- H[1:nx-2, :]) ./ (dx[2:nx-1, :] .+ dx[1:nx-2, :])

        dHdy = zeros(nx, ny-2)
        dHdy[:, :] .= (H[:, 3:ny] .- H[:, 1:ny-2]) ./ (dy[:, 1:ny-2] .+ dy[:, 2:ny-1])

        # Match shapes by expanding dHdx/dHdy across time (3rd dim)
        W1 = .-(UDA[2:end-1, :, :] .* dHdx)   
        W2 = .-(VDA[:, 2:end-1, :] .* dHdy) 

        # ---- Common interior region aligned to (nx-1, ny-1, nt)
        w1c = W1[:, 2:end-1, :]    
        w2c = W2[2:end-1, :, :]     
        w   = w1c .+ w2c            

        # ---- Multiply by p_b interior and average over time
        c  = pb[2:end-1, 2:end-1, :] .* w         
        ca = dropdims(mean(c; dims=3); dims=3)     

        suffix2 = @sprintf("%02dx%02d_%d", xn, yn, buf-2)

        open(joinpath(base2, "Conv", "Conv_$suffix2.bin"), "w") do io; write(io, Float32.(ca)); end
    end
end