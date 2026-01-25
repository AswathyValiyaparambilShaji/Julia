using DSP, MAT, Statistics, Printf,  FilePathsBase, LinearAlgebra, TOML, CairoMakie
include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin

# Load configuration
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
nt_avg = 35  # Total timesteps


# Choose which timestep to plot
timestep = 1


# Initialize full domain array for this timestep
ζ_full = zeros(Float64, NX, NY,nt_avg)


for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        
        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
        
        # Read vorticity tile
        infile = joinpath(base, "3day_mean", "Vorticity", "Vorticity_3day_$suffix.bin")
        VT = open(infile, "r") do io
            nbytes = nx * ny * nt_avg * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt_avg)
        end
        # Calculate tile positions
       xs = (xn - 1) * tx + 1 
       xe = xs + tx + (2 * buf) - 1 
       ys = (yn - 1) * ty + 1 
       ye = ys + ty + (2 * buf) - 1 


       # Interior indices (removing buffer)
       xsf = 2
       xef = tx + (2*buf) - 1
       ysf = 2
       yef = ty + (2*buf) - 1
        
        # Copy tile data to full domain (all nx, ny for this tile)
        ζ_full[xs+1:xe-1, ys+1:ye-1,:] = VT[xsf:xef, ysf:yef,:]
    end
end

zf = ζ_full[ :,:,1]
# Create figure
fig = Figure(resolution=(1000, 800))
ax = Axis(fig[1, 1],
    xlabel = "Longitude [°]",
    ylabel = "Latitude [°]",
    title = "Normalized Vorticity ")

hm = CairoMakie.heatmap!(ax, lon, lat, zf,
    interpolate = false,colorrange=(-0.5, 0.5), colormap=Reverse(:RdBu))


Colorbar(fig[1, 2], hm, label = "ζ/f")


display(fig)


# Save figure
FIGDIR        = cfg["fig_base"]
save(joinpath(FIGDIR, "Vorticity_v1.png"), fig)

