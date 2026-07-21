using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML, CairoMakie, NCDatasets


include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin


config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg   = TOML.parsefile(config_file)
base  = cfg["base_path"]
base2 = cfg["base_path2"]


# --- Domain ---
NX, NY   = 288, 468
minlat, maxlat = 24.0, 31.91
minlon, maxlon = 193.0, 199.0
lat = range(minlat, maxlat, length=NY)
lon = range(minlon, maxlon, length=NX)


# --- Tile ---
buf    = 3
tx, ty = 47, 66
nx, ny = tx + 2*buf, ty + 2*buf
nz     = 88


# --- Plot settings ---
FIGDIR        = cfg["fig_base"]
HEAT_CBAR_MAX = 15
QUIVER_STEP   = 20
ARROW_SCALEUP = 5.0
mkpath(FIGDIR)


# ════════════════════════════════════════════════════════════════════════
# 1) Assemble full-domain background flux field from tiles (unchanged)
# ════════════════════════════════════════════════════════════════════════
tfx = zeros(NX, NY)
tfy = zeros(NX, NY)


for xn in cfg["xn_start"]:cfg["xn_end"]
   for yn in cfg["yn_start"]:cfg["yn_end"]


       suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)


       fxvi = Float64.(open(joinpath(base2, "xflux_corr", "xflx_$suffix.bin"), "r") do io
           raw = read(io, nx*ny*sizeof(Float32))
           reshape(reinterpret(Float32, raw), nx, ny)
       end)


       fyvi = Float64.(open(joinpath(base2, "yflux_corr", "yflx_$suffix.bin"), "r") do io
           raw = read(io, nx*ny*sizeof(Float32))
           reshape(reinterpret(Float32, raw), nx, ny)
       end)


       xs = (xn-1)*tx + 1;  xe = xs + tx - 1
       ys = (yn-1)*ty + 1;  ye = ys + ty - 1
       xsf = buf+1;          xef = buf+tx
       ysf = buf+1;          yef = buf+ty


       tfx[xs:xe, ys:ye] .= fxvi[xsf:xef, ysf:yef]
       tfy[xs:xe, ys:ye] .= fyvi[xsf:xef, ysf:yef]


   end
end


fm    = sqrt.(tfx.^2 .+ tfy.^2)
fm_kW = fm ./ 1000                  # W/m -> kW/m


# background quiver arrow positions/vectors (subsampled, full-domain field)
pos    = Point2f[]
arrvec = Vec2f[]
for i in 1:QUIVER_STEP:NX, j in 1:QUIVER_STEP:NY
   u = tfx[i, j]; v = tfy[i, j]; m = fm_kW[i, j]
   if isfinite(u) && isfinite(v) && isfinite(m)
       push!(pos,    Point2f(Float32(lon[i]), Float32(lat[j])))
       push!(arrvec, Vec2f(Float32(u), Float32(v)))
   end
end


cell_x = (maxlon - minlon) / NX
cell_y = (maxlat - minlat) / NY
maxmag = isempty(arrvec) ? 1f0 : maximum(norm, arrvec)
target = 5f0 * Float32(min(cell_x, cell_y))
scale  = maxmag == 0 ? 1f0 : (target / maxmag) * Float32(ARROW_SCALEUP)   # kW/m -> degrees, shared by everything below


# ════════════════════════════════════════════════════════════════════════
# 2) Mooring locations
# ════════════════════════════════════════════════════════════════════════
mooring_ids = [82, 83, 84, 85]
target_lats = [25.4891, 27.7690, 28.8995, 30.1312]
target_lons = [194.8451, 196.0301, 196.5105, 197.1154]
iwap_idx    = [1, 2, 3, 4]
n_points    = length(iwap_idx)


# ════════════════════════════════════════════════════════════════════════
# 3) IWAP observed flux, per mode (no summing)
# ════════════════════════════════════════════════════════════════════════
file3path = "/nobackup/avaliyap/Box56/Mooring_Data/Flux_mooring_timeseries_ALL_IWAP.mat"
f3 = matopen(file3path)
lato3 = vec(read(f3, "lato"))
lono3 = vec(read(f3, "lono"))
Fuo3  = read(f3, "Fuo")   # (n_moorings, n_modes)
Fvo3  = read(f3, "Fvo")   # (n_moorings, n_modes)
close(f3)


Fu_iwap_mode1 = [Fuo3[iwap_idx[p], 1] for p in 1:n_points]
Fv_iwap_mode1 = [Fvo3[iwap_idx[p], 1] for p in 1:n_points]
Fu_iwap_mode2 = [Fuo3[iwap_idx[p], 2] for p in 1:n_points]
Fv_iwap_mode2 = [Fvo3[iwap_idx[p], 2] for p in 1:n_points]


for p in 1:n_points
   idx = iwap_idx[p]
   dlat = abs(lato3[idx] - target_lats[p])
   dlon = abs(lono3[idx] - target_lons[p])
   if dlat > 0.01 || dlon > 0.01
       @warn "Mooring $(mooring_ids[p]): IWAP file lat/lon ($(lato3[idx]), $(lono3[idx])) " *
             "does not closely match expected ($(target_lats[p]), $(target_lons[p]))"
   end
end


# ════════════════════════════════════════════════════════════════════════
# 4) Model flux, per mode (depth-integrated, kW/m)
# ════════════════════════════════════════════════════════════════════════
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


# ════════════════════════════════════════════════════════════════════════
# 5) Two-panel figure: Mode 1 (left) and Mode 2 (right), same background
# ════════════════════════════════════════════════════════════════════════
fig = Figure(resolution = (1300, 650))


scale_ref_kWm = 5.0   # reference arrow length shown in the scale legend, in kW/m
scale_x0 = minlon + 0.4
scale_y0 = maxlat - 0.4


function plot_panel!(fig_pos, mode_num, Fu_iwap, Fv_iwap, Fu_model, Fv_model)
   ax = Axis(fig_pos,
       title      = "Mode $mode_num flux (mooring vs model) over corrected total flux",
       xlabel     = "Longitude [°]",
       ylabel     = "Latitude [°]",
       xlabelsize = 18,
       ylabelsize = 18,
       titlesize  = 16)
   ax.limits[] = ((minlon, maxlon), (minlat, maxlat))


   hm = CairoMakie.heatmap!(ax, lon, lat, fm_kW,
       interpolate = false,
       colorrange  = (0, HEAT_CBAR_MAX),
       colormap    = :Spectral_9)


   # background full-domain flux arrows (context)
   if !isempty(arrvec)
       arrows!(ax, pos, scale .* arrvec, color=:gray30, arrowsize=6, linewidth=1.0)
   end


   # mooring arrows -- same "scale" factor as background, so lengths are directly comparable
   iwap_vecs  = Vec2f.(Float32.(Fu_iwap .* scale), Float32.(Fv_iwap .* scale))
   model_vecs = Vec2f.(Float32.(Fu_model .* scale), Float32.(Fv_model .* scale))
   mooring_pos = Point2f.(Float32.(target_lons), Float32.(target_lats))


   arrows!(ax, mooring_pos, iwap_vecs;  color = :steelblue,  arrowsize = 14, linewidth = 2.5)
   arrows!(ax, mooring_pos, model_vecs; color = :darkorange, arrowsize = 14, linewidth = 2.5)


   scatter!(ax, target_lons, target_lats; color = :black, markersize = 6)
   for p in 1:n_points
       text!(ax, target_lons[p], target_lats[p]; text = "M$(mooring_ids[p])",
             offset = (6, 6), fontsize = 10, color = :black)
   end


   # --- scale legend arrow (reference magnitude, same scale factor) ---
   arrows!(ax, [Point2f(scale_x0, scale_y0)], [Vec2f(Float32(scale_ref_kWm * scale), 0f0)];
           color = :black, arrowsize = 12, linewidth = 2.5)
   text!(ax, scale_x0, scale_y0 - 0.25; text = "$(scale_ref_kWm) kW/m",
         fontsize = 11, color = :black)


   return hm
end


hm1 = plot_panel!(fig[1, 1], 1, Fu_iwap_mode1, Fv_iwap_mode1, Fu_model_mode1, Fv_model_mode1)
hm2 = plot_panel!(fig[1, 2], 2, Fu_iwap_mode2, Fv_iwap_mode2, Fu_model_mode2, Fv_model_mode2)


Colorbar(fig[1, 3], hm1, label = "(kW/m)")


# shared legend for arrow colors (IWAP vs model)
elem_iwap  = LineElement(color = :steelblue,  linewidth = 3)
elem_model = LineElement(color = :darkorange, linewidth = 3)
Legend(fig[2, 1:3], [elem_iwap, elem_model],
       ["IWAP (observed)", "Model (reconstructed)"],
       orientation = :horizontal, tellwidth = false)


display(fig)


png_file = joinpath(FIGDIR, "Flux_corr_modes1_2_moorings.png")
save(png_file, fig)
println("Saved: $png_file")




