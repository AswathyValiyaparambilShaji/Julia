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
Ah0    = zeros(NX, NY)


for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]


        suffix2 = @sprintf("%02dx%02d_%d", xn, yn, buf - 2)


        conv_path = joinpath(base2, "Conv_z_dI", "Conv_z_$suffix2.bin")
        ah0_path  = joinpath(base2, "Ah0_dI",    "Ah0_$suffix2.bin")


        missing_files = filter(!isfile, [conv_path, ah0_path])
        if !isempty(missing_files)
            @warn "Skipping tile $suffix2 — missing files:" missing_files
            continue
        end


        # --- Read Conv tile ---
        C = Float64.(open(conv_path, "r") do io
            nbytes = (nx - 2) * (ny - 2) * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx - 2, ny - 2)
        end)


        # --- Read Ah0 tile ---
        A = Float64.(open(ah0_path, "r") do io
            nbytes = (nx - 2) * (ny - 2) * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx - 2, ny - 2)
        end)


        # --- Tile position in global grid ---
        xs = (xn - 1) * tx + 1
        xe = xs + tx + (2 * buf) - 1
        ys = (yn - 1) * ty + 1
        ye = ys + ty + (2 * buf) - 1


        # --- Place interior (strip one buffer cell each side) ---
        Conv_z[xs+2:xe-2, ys+2:ye-2] .= C[2:end-1, 2:end-1]
        Ah0[xs+2:xe-2,    ys+2:ye-2] .= A[2:end-1, 2:end-1]


    end
end


# --- Net conversion: Conv_z + Ah0  (Bt-Bc Conversion = ρ′gW + Ah0) ---
Conv_net = Conv_z .+ Ah0


# ============================================================================
# PLOT  — three panels: Conv_z | Ah0 | Conv_net  +  one shared colorbar
# ============================================================================


clim = (-0.050, 0.050)
cmap = :bwr
lonv = collect(lon)
latv = collect(lat)


fig = Figure(size = (1600, 500))


titles = ["Vertical Conversion Cz  (W/m²)",
          "Barotropic Reynolds Stress Ah0  (W/m²)",
          "Net Conversion  Cz + Ah0  (W/m²)"]
fields = [Conv_z, Ah0, Conv_net]


for (col, (title, field)) in enumerate(zip(titles, fields))
    ax = Axis(fig[1, col],
        title  = title,
        xlabel = "Longitude [°]",
        ylabel = col == 1 ? "Latitude [°]" : "")
    ax.limits[] = ((minimum(lon), maximum(lon)), (minimum(lat), maximum(lat)))


    CairoMakie.heatmap!(ax, lonv, latv, field;
        interpolate = false,
        colorrange  = clim,
        colormap    = cmap)
end


# Single shared colorbar built directly from colorrange + colormap
# — does not depend on the heatmap object at all
Colorbar(fig[1, 4];
    colormap   = cmap,
    colorrange = clim,
    label      = "W/m²")


println("Conv_z   sample: $(Conv_z[10,10])")
println("Ah0      sample: $(Ah0[10,10])")
println("Conv_net sample: $(Conv_net[10,10])")


display(fig)


FIGDIR = cfg["fig_base"]
mkpath(FIGDIR)
save(joinpath(FIGDIR, "ConvZ_Ah0_map.png"), fig)
println("Figure saved: $(joinpath(FIGDIR, "ConvZ_Ah0_map.png"))")




