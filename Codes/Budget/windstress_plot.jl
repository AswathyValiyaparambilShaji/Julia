using Printf, FilePathsBase, TOML, CairoMakie, Statistics, LinearAlgebra


include(joinpath(@__DIR__, "..", "..","functions", "FluxUtils.jl"))
using .FluxUtils: read_bin


config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..",  "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base = cfg["base_path"]
base2 = cfg["base_path2"]


# --- Grid parameters ---
NX, NY = 288, 468
minlat, maxlat = 24.0, 31.91
minlon, maxlon = 193.0, 199.0
lat = range(minlat, maxlat, length=NY)
lon = range(minlon, maxlon, length=NX)


# Tiling parameters
buf = 3
tx, ty = 47, 66
nx = tx + 2 * buf
ny = ty + 2 * buf
dt = 25
dto = 144
Tts = 366192
nt = div(Tts, dto)


println("Total time steps: $nt")


# Initialize 3D arrays
TauX_all = zeros(NX, NY, nt)
TauY_all = zeros(NX, NY, nt)


# Load and process tiles
for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
        
        println("Reading tile $suffix...")
        
        # Read entire tile file (all time steps)
        taux = Float64.(open(joinpath(base, "Windstress", "taux_$suffix.bin"), "r") do io
            nbytes = nx * ny * nt * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt)
        end)
        
        tauy = Float64.(open(joinpath(base, "Windstress", "tauy_$suffix.bin"), "r") do io
            nbytes = nx * ny * nt * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt)
        end)
        
        # Center from Arakawa C-grid - ALL TIME STEPS AT ONCE
        taux_ext = zeros(nx+1, ny, nt)
        taux_ext[1:nx, :, :] .= taux
        taux_ext[end, :, :] .= taux[end, :, :]
        
        tauy_ext = zeros(nx, ny+1, nt)
        tauy_ext[:, 1:ny, :] .= tauy
        tauy_ext[:, end, :] .= tauy[:, end, :]
        
        # Average to centers
        taux_c = 0.5 .* (taux_ext[1:end-1, :, :] .+ taux_ext[2:end, :, :])
        tauy_c = 0.5 .* (tauy_ext[:, 1:end-1, :] .+ tauy_ext[:, 2:end, :])
        
        # Extract interior (remove buffer)
        taux_int = taux_c[buf+1:nx-buf, buf+1:ny-buf, :]
        tauy_int = tauy_c[buf+1:nx-buf, buf+1:ny-buf, :]
        
        # Calculate tile position in global grid
        xs = (xn - 1) * tx + 1
        xe = xs + tx - 1
        ys = (yn - 1) * ty + 1
        ye = ys + ty - 1
        
        # Assign to global arrays
        TauX_all[xs:xe, ys:ye, :] .= taux_int
        TauY_all[xs:xe, ys:ye, :] .= tauy_int
        
        println("  Completed $suffix")
    end
end




# ============================================================================
# TIME AVERAGE
# ============================================================================


println("\nCalculating time average over $nt time steps...")


# Time average along dimension 3
TauX_mean = mean(TauX_all, dims=3)[:, :, 1]
TauY_mean = mean(TauY_all, dims=3)[:, :, 1]




# Calculate magnitude
Tau_mag = sqrt.(TauX_mean.^2 .+ TauY_mean.^2)
println("  Magnitude: min=$(minimum(Tau_mag)), max=$(maximum(Tau_mag))")


# ============================================================================
# CREATE SINGLE PLOT
# ============================================================================


println("\nCreating time-averaged wind stress plot...")


FIGDIR = cfg["fig_base"]
mkpath(FIGDIR)


# Arrow parameters
QUIVER_STEP = 8
ARROW_LENGTH_SCALE = 0.3  # Fraction of grid spacing
ARROW_HEAD_SIZE = 8
ARROW_LINE_WIDTH = 1.5


fig = Figure(size=(900, 700))
ax = Axis(fig[1, 1],
    title = "Wind Stress",
    xlabel = "Longitude [°]",
    ylabel = "Latitude [°]",
    ylabelsize = 20,
    xlabelsize = 20,
    titlesize = 24
)


# Heatmap
hm = heatmap!(ax, lon, lat, Tau_mag,
    colorrange = (0, maximum(Tau_mag)),
    colormap = :Spectral_9
)


# Arrows with adaptive scaling
pos = Point2f[]
vec = Vec2f[]


for i in 1:QUIVER_STEP:NX, j in 1:QUIVER_STEP:NY
    u, v = TauX_mean[i, j], TauY_mean[i, j]
    if isfinite(u) && isfinite(v)
        push!(pos, Point2f(lon[i], lat[j]))
        push!(vec, Vec2f(u, v))
    end
end


if !isempty(vec)
    maxm = maximum(norm, vec)
    if maxm > 0
        # Adaptive scaling based on grid spacing
        cell_x = (maximum(lon) - minimum(lon)) / NX
        cell_y = (maximum(lat) - minimum(lat)) / NY
        target_length = min(cell_x, cell_y) * QUIVER_STEP * ARROW_LENGTH_SCALE
        
        scale = target_length / maxm
        
        arrows!(ax, pos, scale .* vec,
            color = :black, 
            arrowsize = ARROW_HEAD_SIZE,
            linewidth = ARROW_LINE_WIDTH
        )
        
        
    end
end


# Colorbar
Colorbar(fig[1, 2], hm, label = "[N/m²]") #,vertical=true, labelrotation = 90, valign =:top, halign =:center)


# Display
display(fig)


# Save
output_file = joinpath(FIGDIR, "WindStress_TimeAverage.png")
save(output_file, fig)


println("\nPlot saved: $output_file")
println("Done!")




