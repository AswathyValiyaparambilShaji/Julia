using MAT, TOML, CairoMakie, NCDatasets


config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg  = TOML.parsefile(config_file)
base = cfg["base_path"]


# --- Domain (just for axis limits) ---
minlat, maxlat = 24.0, 31.91
minlon, maxlon = 193.0, 199.0


# --- Plot settings ---
FIGDIR = cfg["fig_base"]
mkpath(FIGDIR)


# ════════════════════════════════════════════════════════════════════════
# 1) Mooring locations
# ════════════════════════════════════════════════════════════════════════
mooring_ids = [82, 83, 84, 85]
target_lats = [25.4891, 27.7690, 28.8995, 30.1312]
target_lons = [194.8451, 196.0301, 196.5105, 197.1154]
iwap_idx    = [1, 2, 3, 4]
n_points    = length(iwap_idx)


# ════════════════════════════════════════════════════════════════════════
# 2) IWAP observed flux, per mode (no summing)
# ════════════════════════════════════════════════════════════════════════
file3path = "/home/aswathy/mnt/data/aswathy/Mooring_Data/Flux_mooring_timeseries_ALL_IWAP.mat"
f3 = matopen(file3path)
lato3 = vec(read(f3, "lato"))
lono3 = vec(read(f3, "lono"))
Fuo3  = read(f3, "Fuo")   # (n_moorings, n_modes)
Fvo3  = read(f3, "Fvo")   # (n_moorings, n_modes)
println(lato3)
println(lono3)
println(Fuo3[:,2])
println(Fvo3[:,2])
for name in keys(f3)
    data = read(f3, name)
    println("Variable: ", name)
    println("  Type: ", typeof(data))
    println("  Size: ", size(data))
    println()
end
close(f3)

Fu_iwap_mode1 = [Fuo3[iwap_idx[p], 1] for p in 1:n_points]
Fv_iwap_mode1 = [Fvo3[iwap_idx[p], 1] for p in 1:n_points]
Fu_iwap_mode2 = [Fuo3[iwap_idx[p], 2] for p in 1:n_points]
Fv_iwap_mode2 = [Fvo3[iwap_idx[p], 2] for p in 1:n_points]

# ════════════════════════════════════════════════════════════════════════
# 3) Model flux, per mode (depth-integrated, kW/m)
#= ════════════════════════════════════════════════════════════════════════
model_file = joinpath(base, "SM", "MooringModalFlux", "MooringModalFlux_Box56_IWAP.nc")


lat_model, lon_model, uflux_model_modes, vflux_model_modes =
  NCDataset(model_file, "r") do ds
      (Array(ds["lat"][:]), Array(ds["lon"][:]),
       Array(ds["uflux_depth_int"][:, :]),   # (station, mode)
       Array(ds["vflux_depth_int"][:, :]))
  end


Fu_model_mode1 = Float64.(uflux_model_modes[:, 1])
Fv_model_mode1 = Float64.(vflux_model_modes[:, 1])
Fu_model_mode2 = Float64.(uflux_model_modes[:, 2])
Fv_model_mode2 = Float64.(vflux_model_modes[:, 2])


for p in 1:n_points
  dlat = abs(lat_model[p] - target_lats[p])
  dlon = abs(lon_model[p] - target_lons[p])
  if dlat > 0.01 || dlon > 0.01
      @warn "Model file station $p lat/lon ($(lat_model[p]), $(lon_model[p])) " *
            "does not match expected mooring $(mooring_ids[p])"
  end
end

=#
# ════════════════════════════════════════════════════════════════════════
# 4) Two-panel figure: Mode 1 (left) and Mode 2 (right), mooring vectors ONLY
#    -- no background flux field / heatmap
# ════════════════════════════════════════════════════════════════════════
fig = Figure(resolution = (1200, 800))


scale_x0 = minlon + 0.4
scale_y0 = maxlat - 0.4




# reference value for each panel's legend = the actual max flux magnitude
# present in that panel (IWAP or model, whichever is larger)
mag1_iwap  = sqrt.(Fu_iwap_mode1.^2  .+ Fv_iwap_mode1.^2)
#mag1_model = sqrt.(Fu_model_mode1.^2 .+ Fv_model_mode1.^2)
#scale_ref_kWm1 = maximum(vcat(mag1_iwap, mag1_model))
scale_ref_kWm1 = 2.0


mag2_iwap  = sqrt.(Fu_iwap_mode2.^2  .+ Fv_iwap_mode2.^2)
#mag2_model = sqrt.(Fu_model_mode2.^2 .+ Fv_model_mode2.^2)
#scale_ref_kWm2 = maximum(vcat(mag2_iwap, mag2_model))
scale_ref_kWm2 = 0.1


scale_mode1 = (target / (scale_ref_kWm1)) * (ARROW_SCALEUP)   # kW/m -> degrees
scale_mode2 = (target / (scale_ref_kWm2)) * (ARROW_SCALEUP)   # kW/m -> degrees

mooring_pos = Point2f.((target_lons), (target_lats))


# --- Panel 1: Mode 1 ---
ax1 = Axis(fig[1, 1],
     aspect = DataAspect(),
    title      = "Mode 1 flux (mooring vs model)",
    xlabel     = "Longitude [°]",
    ylabel     = "Latitude [°]",
    xlabelsize = 18,
    ylabelsize = 18,
    titlesize  = 16)
ax1.limits[] = ((minlon, maxlon), (minlat, maxlat))


iwap_vecs1  = Vec2f.((Fu_iwap_mode1 .* scale_mode1), (Fv_iwap_mode1 .* scale_mode1))
#model_vecs1 = Vec2f.(Float32.(Fu_model_mode1 .* scale_mode1), Float32.(Fv_model_mode1 .* scale_mode1))


arrows!(ax1, mooring_pos, iwap_vecs1;  color = :black,  arrowsize = 7, linewidth = 3)
#arrows!(ax1, mooring_pos, model_vecs1; color = :magenta, arrowsize = 7, linewidth = 3)


# --- scale legend arrow for panel 1 (length = actual max flux magnitude in this panel) ---
scale_len1 = (scale_ref_kWm1 * scale_mode1)
lines!(ax1, [scale_x0, scale_x0 + scale_len1], [scale_y0, scale_y0];
       color = :black, linewidth = 2.5)
arrows!(ax1, [Point2f(scale_x0, scale_y0)], [Vec2f(scale_len1, 0f0)];
        color = :black, arrowsize = 7, linewidth = 2.5)
text!(ax1, scale_x0, scale_y0 - 0.25; text = "$(round(scale_ref_kWm1, digits=2)) kW/m",
      fontsize = 11, color = :black)


# --- Panel 2: Mode 2 ---
ax2 = Axis(fig[1, 2],
     aspect = DataAspect(),
    title      = "Mode 2 flux (mooring vs model)",
    xlabel     = "Longitude [°]",
    ylabel     = "Latitude [°]",
    xlabelsize = 18,
    ylabelsize = 18,
    titlesize  = 16)
ax2.limits[] = ((minlon, maxlon), (minlat, maxlat))


iwap_vecs2  = Vec2f.((Fu_iwap_mode2 .* scale_mode2), (Fv_iwap_mode2 .* scale_mode2))
#model_vecs2 = Vec2f.(Float32.(Fu_model_mode2 .* scale_mode2), Float32.(Fv_model_mode2 .* scale_mode2))


arrows!(ax2, mooring_pos, iwap_vecs2;  color = :black,  arrowsize = 7, linewidth = 3)
#arrows!(ax2, mooring_pos, model_vecs2; color = :magenta, arrowsize = 7, linewidth = 3)


# --- scale legend arrow for panel 2 (length = actual max flux magnitude in this panel) ---
scale_len2 = (scale_ref_kWm2 * scale_mode2)
lines!(ax2, [scale_x0, scale_x0 + scale_len2], [scale_y0, scale_y0];
       color = :black, linewidth = 2.5)
arrows!(ax2, [Point2f(scale_x0, scale_y0)], [Vec2f(scale_len2, 0f0)];
        color = :black, arrowsize = 7, linewidth = 2.5)
text!(ax2, scale_x0, scale_y0 - 0.25; text = "$(round(scale_ref_kWm2, digits=2)) kW/m",
      fontsize = 11, color = :black)


# shared legend for arrow colors (IWAP vs model)
elem_iwap  = LineElement(color = :black,  linewidth = 3)
#elem_model = LineElement(color = :magenta, linewidth = 3)
Legend(fig[2, 1:2], [elem_iwap],
      ["IWAP (observed)"],
      orientation = :horizontal, tellwidth = false)


display(fig)


png_file = joinpath(FIGDIR, "Mooring_modes1_2_only.png")
save(png_file, fig)
println("Saved: $png_file")




