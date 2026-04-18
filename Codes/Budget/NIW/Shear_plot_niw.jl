using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML
using CairoMakie, SparseArrays
include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin

config_file = get(ENV, "JULIA_CONFIG",
              joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
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

# ==========================================================
# ======= INITIALIZE FULL-DOMAIN ARRAYS ====================
# ==========================================================
G_vel_H_full = fill(NaN, NX, NY)
G_vel_V_full = fill(NaN, NX, NY)

# ==========================================================
# ======= BUILD G HORIZONTAL AND VERTICAL MAPS =============
# ==========================================================
for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)

        # --- Read G horizontal shear (IT → NIW) ---
        g_vel_h = Float64.(open(joinpath(base2, "G_vel_full", "g_vel_mean_$suffix.bin"), "r") do io
            nbytes = nx * ny * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
        end)

        # --- Read G vertical shear (IT → NIW) ---
        g_vel_v = Float64.(open(joinpath(base2, "G_vel_V_full", "g_vel_v_mean_$suffix.bin"), "r") do io
            nbytes = nx * ny * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
        end)

        xs = (xn - 1) * tx + 1
        xe = xs + tx - 1
        ys = (yn - 1) * ty + 1
        ye = ys + ty - 1

        G_vel_H_full[xs:xe, ys:ye] .= g_vel_h[buf+1:nx-buf, buf+1:ny-buf]
        G_vel_V_full[xs:xe, ys:ye] .= g_vel_v[buf+1:nx-buf, buf+1:ny-buf]

        println("Completed tile $suffix")
    end
end

println("\nG_vel_H range: $(minimum(filter(!isnan, G_vel_H_full))) to $(maximum(filter(!isnan, G_vel_H_full)))")
println("G_vel_V range: $(minimum(filter(!isnan, G_vel_V_full))) to $(maximum(filter(!isnan, G_vel_V_full)))")

# ==========================================================
# =================== VISUALIZATION ========================
# ==========================================================

# --- Compute symmetric color range for each panel ---
clim_H = maximum(abs.(filter(!isnan, G_vel_H_full)))
clim_V = maximum(abs.(filter(!isnan, G_vel_V_full)))

# Optional: cap extremes for better visualization
clim_H = min(clim_H, 0.01)
clim_V = min(clim_V, 0.01)

fig = Figure(size=(1800, 800))

# --- Panel 1: G Horizontal Shear ---
ax1 = Axis(fig[1, 1],
    title="G: IT→NIW Horizontal Shear (Velocity)",
    xlabel="Longitude [°]",
    ylabel="Latitude [°]")

hm1 = CairoMakie.heatmap!(ax1, lon, lat, G_vel_H_full;
    interpolate=false,
    colormap=Reverse(:RdBu),
    colorrange=(-0.00005, 0.00005))

Colorbar(fig[1, 2], hm1, label="G Horizontal [W/m²]")

# --- Panel 2: G Vertical Shear ---
ax2 = Axis(fig[1, 3],
    title="G: IT→NIW Vertical Shear (Velocity)",
    xlabel="Longitude [°]",
    ylabel="Latitude [°]")

hm2 = CairoMakie.heatmap!(ax2, lon, lat, G_vel_V_full;
    interpolate=false,
    colormap=Reverse(:RdBu),
    colorrange=(-0.00005, 0.00005))

Colorbar(fig[1, 4], hm2, label="G Vertical [W/m²]")

display(fig)

# --- Save ---
FIGDIR = cfg["fig_base"]
save(joinpath(FIGDIR, "G_vel_H_V_comparison.png"), fig)
println("Figure saved.")