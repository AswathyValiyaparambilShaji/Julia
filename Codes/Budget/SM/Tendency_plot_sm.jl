using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML
using CairoMakie




config_file = get(ENV, "JULIA_CONFIG",
              joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)




base  = cfg["base_path"]
base2 = cfg["base_path2"]




NX, NY = 288, 468
minlat, maxlat = 24.0, 31.91
minlon, maxlon = 193.0, 199.0
lat = range(minlat, maxlat, length=NY)
lon = range(minlon, maxlon, length=NX)




buf = 3
tx, ty = 47, 66
nx = tx + 2*buf
ny = ty + 2*buf




TE_TENDENCY_full = fill(NaN, NX, NY)




# ==========================================================
# ========== BUILD ENERGY TENDENCY MAP =====================
# ==========================================================




for xn in cfg["xn_start"]:cfg["xn_end"]
   for yn in cfg["yn_start"]:cfg["yn_end"]




       suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)




       # Read time-averaged energy tendency
       te_mean = Float64.(open(joinpath(base2, "TE_t", "te_t_mean_$suffix.bin"), "r") do io
           nbytes = nx * ny * sizeof(Float32)
           reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
       end)




       xs = (xn - 1) * tx + 1
       xe = xs + tx - 1
       ys = (yn - 1) * ty + 1
       ye = ys + ty - 1




       te_interior = te_mean[buf+1:nx-buf, buf+1:ny-buf]




       TE_TENDENCY_full[xs:xe, ys:ye] .= te_interior




       println("Loaded tile $suffix")
   end
end




println("\nTE_TENDENCY_full range: $(minimum(skipmissing(TE_TENDENCY_full))) to $(maximum(skipmissing(TE_TENDENCY_full)))")




# ==========================================================
# =================== VISUALIZATION ========================
# ==========================================================




fig = Figure(resolution = (1000, 800))




ax = Axis(fig[1, 1],
        title="Depth-Integrated Time-Averaged Energy Tendency (∂E/∂t)",
        xlabel="Longitude [°]",
        ylabel="Latitude [°]")
ax.limits[] = (193.0, 194.2, 24.0, 25.4)





hm = CairoMakie.heatmap!(ax, lon, lat, TE_TENDENCY_full;
                       interpolate=false,
                        colormap=Reverse(:RdBu),
                        colorrange=(-0.05, 0.05))


Colorbar(fig[1, 2], hm, label="Buoyancy Production [W/m²]")


display(fig)




# Save figure
FIGDIR = cfg["fig_base"]
save(joinpath(FIGDIR, "TE_tendency_production_v1.png"), fig)




println("\nFigure saved to: $(joinpath(FIGDIR, "TE_tendency_production_v1.png"))")




