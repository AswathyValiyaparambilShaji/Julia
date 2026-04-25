using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML
using CairoMakie


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
G_buoy_full = fill(NaN, NX, NY)


# ==========================================================
# ======= BUILD G_BUOY MAP =================================
# ==========================================================
for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)


        g_buoy = Float64.(open(joinpath(base2, "G_buoy_full", "g_buoy_mean_$suffix.bin"), "r") do io
            nbytes = nx * ny * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
        end)


        xs = (xn - 1) * tx + 1
        xe = xs + tx - 1
        ys = (yn - 1) * ty + 1
        ye = ys + ty - 1


        G_buoy_full[xs:xe, ys:ye] .= g_buoy[buf+1:nx-buf, buf+1:ny-buf]


        println("Completed tile $suffix")
    end
end


println("\nG_buoy range: $(minimum(filter(!isnan, G_buoy_full))) to $(maximum(filter(!isnan, G_buoy_full)))")


# ==========================================================
# =================== VISUALIZATION ========================
# ==========================================================


clim_buoy = maximum(abs.(filter(!isnan, G_buoy_full)))
clim_buoy = min(clim_buoy, 0.01)   # cap — adjust after seeing printed range


fig = Figure(resolution=(600, 800))


ax = Axis(fig[1, 1],
    title="G: IT→NIW Buoyancy Production (Full Time Average)",
    xlabel="Longitude [°]",
    ylabel="Latitude [°]")


hm = CairoMakie.heatmap!(ax, lon, lat, G_buoy_full;
    interpolate=false,
    colormap=Reverse(:RdBu),
    colorrange=(-0.02, 0.02))


Colorbar(fig[1, 2], hm, label="G Buoyancy [W/m²]")


display(fig)


FIGDIR = cfg["fig_base"]
save(joinpath(FIGDIR, "G_buoy_full.png"), fig)
println("G_buoy figure saved.")

#=
# ==========================================================
# ======= COMBINED 3-PANEL PLOT (vel_H + vel_V + buoy) =====
# ==========================================================
# Optionally load the G_vel fields too for a combined summary figure


G_vel_H_full = fill(NaN, NX, NY)
G_vel_V_full = fill(NaN, NX, NY)


for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)


        g_vel_h = Float64.(open(joinpath(base2, "G_vel_full", "g_vel_mean_$suffix.bin"), "r") do io
            reshape(reinterpret(Float32, read(io, nx*ny*sizeof(Float32))), nx, ny)
        end)


        g_vel_v = Float64.(open(joinpath(base2, "G_vel_V_full", "g_vel_v_mean_$suffix.bin"), "r") do io
            reshape(reinterpret(Float32, read(io, nx*ny*sizeof(Float32))), nx, ny)
        end)


        xs = (xn - 1) * tx + 1
        xe = xs + tx - 1
        ys = (yn - 1) * ty + 1
        ye = ys + ty - 1


        G_vel_H_full[xs:xe, ys:ye] .= g_vel_h[buf+1:nx-buf, buf+1:ny-buf]
        G_vel_V_full[xs:xe, ys:ye] .= g_vel_v[buf+1:nx-buf, buf+1:ny-buf]
    end
end


# Compute G_total = G_vel_H + G_vel_V + G_buoy
G_total_full = G_vel_H_full .+ G_vel_V_full .+ G_buoy_full


println("\nG_vel_H range: $(minimum(filter(!isnan, G_vel_H_full))) to $(maximum(filter(!isnan, G_vel_H_full)))")
println("G_vel_V range: $(minimum(filter(!isnan, G_vel_V_full))) to $(maximum(filter(!isnan, G_vel_V_full)))")
println("G_buoy  range: $(minimum(filter(!isnan, G_buoy_full))) to $(maximum(filter(!isnan, G_buoy_full)))")
println("G_total range: $(minimum(filter(!isnan, G_total_full))) to $(maximum(filter(!isnan, G_total_full)))")


# --- Symmetric color limits per panel ---
clim_H     = min(maximum(abs.(filter(!isnan, G_vel_H_full))), 0.01)
clim_V     = min(maximum(abs.(filter(!isnan, G_vel_V_full))), 0.01)
clim_buoy  = min(maximum(abs.(filter(!isnan, G_buoy_full))),  0.01)
clim_total = min(maximum(abs.(filter(!isnan, G_total_full))), 0.01)


fig2 = Figure(size=(2200, 800))


# Panel 1: G_vel_H
ax1 = Axis(fig2[1, 1],
    title="G: Horizontal Shear (IT→NIW)",
    xlabel="Longitude [°]", ylabel="Latitude [°]")
hm1 = CairoMakie.heatmap!(ax1, lon, lat, G_vel_H_full;
    interpolate=false, colormap=Reverse(:RdBu),
    colorrange=(-clim_H, clim_H))
Colorbar(fig2[1, 2], hm1, label="[W/m²]")


# Panel 2: G_vel_V
ax2 = Axis(fig2[1, 3],
    title="G: Vertical Shear (IT→NIW)",
    xlabel="Longitude [°]", ylabel="Latitude [°]")
hm2 = CairoMakie.heatmap!(ax2, lon, lat, G_vel_V_full;
    interpolate=false, colormap=Reverse(:RdBu),
    colorrange=(-clim_V, clim_V))
Colorbar(fig2[1, 4], hm2, label="[W/m²]")


# Panel 3: G_buoy
ax3 = Axis(fig2[1, 5],
    title="G: Buoyancy Production (IT→NIW)",
    xlabel="Longitude [°]", ylabel="Latitude [°]")
hm3 = CairoMakie.heatmap!(ax3, lon, lat, G_buoy_full;
    interpolate=false, colormap=Reverse(:RdBu),
    colorrange=(-clim_buoy, clim_buoy))
Colorbar(fig2[1, 6], hm3, label="[W/m²]")


# Panel 4: G_total
ax4 = Axis(fig2[1, 7],
    title="G Total = G_H + G_V + G_buoy (IT→NIW)",
    xlabel="Longitude [°]", ylabel="Latitude [°]")
hm4 = CairoMakie.heatmap!(ax4, lon, lat, G_total_full;
    interpolate=false, colormap=Reverse(:RdBu),
    colorrange=(-clim_total, clim_total))
Colorbar(fig2[1, 8], hm4, label="[W/m²]")


display(fig2)
save(joinpath(FIGDIR, "G_all_components_full.png"), fig2)
println("Combined G figure saved.")

=#


