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

# Initialize arrays to store the full flux data
tfx = zeros(NX, NY)  # For storing the xflux for the entire region
tfy = zeros(NX, NY)

for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)


        hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))

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

        # --- DRFfull and Flux Calculations ---
        DRFfull = hFacC .* DRF3d
        fxX = sum(fx .* DRFfull, dims=3)  # Integrate over depth (z-axis)
        fyY = sum(fy .* DRFfull, dims=3)  # Integrate over depth (z-axis)

        # --- Assign the flux data to the full flux arrays ---
        # Define the tile region (using indices from the loop)
        xs = (xn - 1) * tx + 1  
        xe = xs + tx + (2 * buf) - 1  
        
        ys = (yn - 1) * ty + 1  
        ye = ys + ty + (2 * buf) - 1  
        

        xsf = 2;
        xef = tx + (2*buf) - 1
        ysf = 2;
        yef = ty + (2*buf) - 1
        
        # Assign the flux data to the correct region in the full flux arrays
        tfx[xs+1:xe-1, ys+1:ye-1] .= fxX[xsf:xef,ysf:yef]
        tfy[xs+1:xe-1, ys+1:ye-1] .= fyY[xsf:xef,ysf:yef]
    end
end
println(tfx[1:tx,10])
# ABSOLUTE FLUX of MITgcm data for BOX56
fm = sqrt.(tfx.^2 + tfy.^2)


using CairoMakie
# --- Config ---
FIGDIR        = cfg["fig_base"]
HEAT_CBAR_MAX = 0.30        # kW/m
QUIVER_STEP   = 20          # grid stride for arrows
ARROW_SCALEUP = 5.0
DO_TRANSPOSE  = false       # set true if your image appears rotated/flipped

isdir(FIGDIR) || mkpath(FIGDIR)

# --- Prep (units & orientation) ---
fm_kW = fm ./ 1000          # convert to kW/m for display

# Optionally transpose all 2D fields to match Makie’s x=lon, y=lat expectation
F  = DO_TRANSPOSE ? fm_kW' : fm_kW
Ux = DO_TRANSPOSE ? tfx'   : tfx
Uy = DO_TRANSPOSE ? tfy'   : tfy

# lon/lat should align with F’s axes
LON = lon
LAT = lat

# -------------------------------
# Figure 1: Heatmap + Quiver
# -------------------------------
fig1 = Figure(resolution = (700, 600))
ax1  = Axis(fig1[1, 1],
    title  = "MITgcm Flux (with quiver)",
    xlabel = "Longitude [°]",
    ylabel = "Latitude [°]",
)

ax1.limits[] = (193.0,194.2,24.0, 25.4)

hm1 = CairoMakie.heatmap!(ax1, LON, LAT, F;
    interpolate = false,
    colorrange  = (0, HEAT_CBAR_MAX),
    colormap    = :jet,
)

# Build quiver subsample
pos = Point2f[]
vec = Vec2f[]
NX, NY = size(F)
for i in 1:QUIVER_STEP:NX, j in 1:QUIVER_STEP:NY
    u = Ux[i, j]; v = Uy[i, j]; m = F[i, j]
    if isfinite(u) && isfinite(v) && isfinite(m)
        push!(pos, Point2f(Float32(LON[i]), Float32(LAT[j])))
        push!(vec, Vec2f(Float32(u), Float32(v)))
    end
end

#=if !isempty(vec)
    # scale arrows to look nice relative to degrees grid
    maxmag = maximum(norm, vec)
    cell_x = (maximum(LON) - minimum(LON)) / NX
    cell_y = (maximum(LAT) - minimum(LAT)) / NY
    target = 5f0 * Float32(min(cell_x, cell_y))
    scale  = maxmag == 0 ? 1f0 : (target / maxmag) * Float32(ARROW_SCALEUP)

    arrows!(ax1, pos, scale .* vec; color=:black, shaftwidth=1.5f0, tipwidth=8f0, tiplength=8f0)
end
=#

Colorbar(fig1[1, 2], hm1, label = "Flux (kW/m)")
save(joinpath(FIGDIR, "Flux_with_quiver_NIW2_v3.png"), fig1)

display(fig1)