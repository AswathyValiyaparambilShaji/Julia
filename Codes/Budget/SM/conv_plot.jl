using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML, CairoMakie


include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter


# Load configuration
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg   = TOML.parsefile(config_file)
base  = cfg["base_path"]
base2 = cfg["base_path2"]


# --- Domain & grid ---
NX, NY = 288, 468
minlat, maxlat = 24.0, 31.91
minlon, maxlon = 193.0, 199.0
lat = range(minlat, maxlat, length=NY)
lon = range(minlon, maxlon, length=NX)


# --- Tile ---
buf    = 3
tx, ty = 47, 66
nx     = tx + 2 * buf
ny     = ty + 2 * buf


# ============================================================================
# ASSEMBLE GLOBAL MAP FROM TILES
# ============================================================================


Conv_z = zeros(NX, NY)


for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]


        suffix2 = @sprintf("%02dx%02d_%d", xn, yn, buf - 2)
        fpath   = joinpath(base2, "Conv", "Conv_$suffix2.bin")


    

        # Read tile (nx-2) x (ny-2) Float32
        C = Float64.(open(fpath, "r") do io
            nbytes   = (nx - 2) * (ny - 2) * sizeof(Float32)
            raw_data = reinterpret(Float32, read(io, nbytes))
            reshape(raw_data, nx - 2, ny - 2)
        end)


        # Tile position in global grid
        xs = (xn - 1) * tx + 1
        xe = xs + tx + (2 * buf) - 1
        ys = (yn - 1) * ty + 1
        ye = ys + ty + (2 * buf) - 1


        # Place interior (strip one buffer cell each side)
        Conv_z[xs+2:xe-2, ys+2:ye-2] .= C[2:end-1, 2:end-1]


    end
end


# ============================================================================
# PLOT
# ============================================================================


fig = Figure(resolution = (700, 500))
println(Conv_z[10,10])

ax = Axis(fig[1, 1],
    title   = "Vertical Conversion Cz  (W/m²)",
    xlabel  = "Longitude [°]",
    ylabel  = "Latitude [°]")
ax.limits[] = ((minimum(lon), maximum(lon)), (minimum(lat), maximum(lat)))


hm = CairoMakie.heatmap!(ax, lon, lat, Conv_z;
    interpolate = false,
    colorrange  = (-0.050, 0.050),
    colormap    = Reverse(:RdBu))


Colorbar(fig[1, 2], hm, label = "W/m²")


display(fig)


FIGDIR = cfg["fig_base"]
mkpath(FIGDIR)
save(joinpath(FIGDIR, "ConvZ_map_v1.png"), fig)
println("Figure saved: $(joinpath(FIGDIR, "ConvZ_map_v1.png"))")




