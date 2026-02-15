using Printf, FilePathsBase, TOML, NCDatasets


# Include FluxUtils.jl
include(joinpath(@__DIR__, "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin_be


# Read the configuration file for paths
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base = cfg["base_path"]


# --- Grid parameters ---
NX, NY = 288, 468


# Tiling parameters
buf = 3
tx, ty = 47, 66
nx = tx + 2 * buf
ny = ty + 2 * buf
dto = 144
Tts = 366192
nt = div(Tts, dto)

ds = NCDataset(joinpath(base , "Siva_Diss","TotDiss_band1.nc"))
println(ds)


# --- Thickness & constants ---
thk = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)




rho0 = 999.8


hFacC_full = zeros(NX, NY, nz)
    for xn in cfg["xn_start"]:cfg["xn_end"]
        for yn in cfg["yn_start"]:cfg["yn_end"]
            suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
            
            println("\n--- Processing tile: $suffix ---")
            
            # --- Read grid metrics ---
            hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
            # Calculate tile positions in global grid
            xs = (xn - 1) * tx + 1
            xe = xs + tx + (2 * buf) - 1
            ys = (yn - 1) * ty + 1
            ye = ys + ty + (2 * buf) - 1
            
            hFacC_full[xs+2:xe-2, ys+2:ye-2, :] .= hFacC[buf:nx-buf+1, buf:ny-buf+1, :]
        end
    end
