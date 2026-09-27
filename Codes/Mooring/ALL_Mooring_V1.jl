# =============================================================================
# Model (MITgcm) vs mooring modal energy flux -- REGIONAL comparison
#
#  * Uses only two mooring files: ALL and IWAP (ALL_OLD is dropped).
#  * Matching priority: IWAP > ALL.
#  * ALL-matched stations are grouped into N_REGIONS (=6) geographic regions
#    (automatic k-means on the sphere, or manual lon/lat boxes);
#    IWAP-matched stations form a 7th region of their own.
#  * For every region and for mode 1 and mode 2 -> one figure (7 x 2 = 14),
#    each with 3 panels:
#       (a) zoomed location map of the region (land + labeled stations)
#       (b) bar charts: |F| model vs mooring (top), direction 0-360 deg (bottom)
#       (c) zoomed flux-arrow map: model (red) vs mooring (blue) arrows
#  * Plus one overview world map showing all 7 regions.
# =============================================================================

using NCDatasets, MAT, Statistics, CairoMakie, GeoMakie, Printf, LinearAlgebra

# ----------------------------------------------------------------------------
# 0) PATHS & SETTINGS -- EDIT THESE
# ----------------------------------------------------------------------------
model_flux_ncfile = "/home/aswathy/mnt/data/aswathy/MITgcm_NAS/Moorings/Mooring_modal_fluxes.nc"
file1path = "/home/aswathy/mnt/data/aswathy/Mooring_Data/Flux_mooring_timeseries_ALL.mat"
file3path = "/home/aswathy/mnt/data/aswathy/Mooring_Data/Flux_mooring_timeseries_ALL_IWAP.mat"

FIGDIR = "/home/aswathy/mnt/data/aswathy/Mooring_Data/Regional_figs/"
mkpath(FIGDIR)

MATCH_TOL_DEG = 0.05     # lat/lon matching tolerance (degrees)
THETA_ROT     = pi / 3   # 60 deg, rotation of the IWAP MOORING fluxes (model is not rotated)

N_REGIONS   = 6          # number of regions for the ALL-matched stations
REGION_MODE = :auto      # :auto  -> k-means clustering of station positions
                        # :manual -> use REGION_BOXES below


# Only used when REGION_MODE == :manual. Longitudes in -180..180.
# name => (lonmin, lonmax, latmin, latmax)
REGION_BOXES = [
   "Region 1" => (-180.0, -150.0,  10.0,  30.0),
   "Region 2" => ( 110.0,  125.0,  15.0,  25.0),
   "Region 3" => (-80.0,  -60.0,   20.0,  45.0),
   "Region 4" => (-40.0,  -10.0,   30.0,  60.0),
   "Region 5" => ( 130.0,  150.0,  20.0,  40.0),
   "Region 6" => ( 40.0,   80.0,  -10.0,  25.0),
]

COL_MODEL = :crimson
COL_OBS   = :steelblue

# ---- layout settings (all figures are auto-trimmed to these sizes) ----
MAP_W  = 560            # width  (px) of each zoomed map panel (a) and (c)
MAP_H  = 460            # height (px) of the panel row
BAR_W_PER_STATION = 34  # width (px) per station in the bar panel (b)
BAR_W_MIN = 420         # minimum width (px) of the bar panel
OVERVIEW_W = 1300       # width (px) of the overview maps
BOX_PAD_DEG = 1.5       # padding of region boxes on the overview maps (auto-shrunk to avoid overlaps)

# ----------------------------------------------------------------------------
# 1) LOAD MODEL FLUXES (kW/m, depth-integrated, time-averaged)
# ----------------------------------------------------------------------------
ds = NCDataset(model_flux_ncfile, "r")
lon_m = Float64.(Array(ds["lon"]))
lat_m = Float64.(Array(ds["lat"]))
Fu1_m = Float64.(Array(ds["Fu_mode1"])); Fv1_m = Float64.(Array(ds["Fv_mode1"]))
Fu2_m = Float64.(Array(ds["Fu_mode2"])); Fv2_m = Float64.(Array(ds["Fv_mode2"]))
close(ds)
N_model = length(lat_m)
println("Loaded $N_model model stations from $model_flux_ncfile")


# ----------------------------------------------------------------------------
# 2) LOAD OBSERVED MOORINGS (ALL and IWAP only)
# ----------------------------------------------------------------------------
function load_mooring_mat(path)
   f = matopen(path)
   lato = vec(read(f, "lato"))
   lono = vec(read(f, "lono"))
   Fuo  = read(f, "Fuo")
   Fvo  = read(f, "Fvo")
   close(f)
   if ndims(Fuo) == 3            # (mooring, mode, time) -> NaN-aware time mean
       n1, n2, _ = size(Fuo)
       Fuo2d = fill(NaN, n1, n2); Fvo2d = fill(NaN, n1, n2)
       for i in 1:n1, j in 1:n2
           vu = filter(!isnan, @view Fuo[i, j, :])
           vv = filter(!isnan, @view Fvo[i, j, :])
           Fuo2d[i, j] = isempty(vu) ? NaN : mean(vu)
           Fvo2d[i, j] = isempty(vv) ? NaN : mean(vv)
       end
       Fuo, Fvo = Fuo2d, Fvo2d
   end
   return Float64.(lato), Float64.(lono), Float64.(Fuo), Float64.(Fvo)
end


lato1, lono1, Fuo1, Fvo1 = load_mooring_mat(file1path)   # ALL
lato3, lono3, Fuo3, Fvo3 = load_mooring_mat(file3path)   # IWAP

# rotate the IWAP mooring fluxes by 60 deg (all moorings, both modes)
Fuo3, Fvo3 = cos(THETA_ROT) .* Fuo3 .- sin(THETA_ROT) .* Fvo3,
            sin(THETA_ROT) .* Fuo3 .+ cos(THETA_ROT) .* Fvo3
println("ALL:  $(length(lato1)) moorings")
println("IWAP: $(length(lato3)) moorings")

# ----------------------------------------------------------------------------
# 3) MATCH EVERY MOORING -> ITS MODEL STATION
#    Every ALL and IWAP mooring is kept. Moorings at the same site share
#    the same model station (e.g. M2 and M7 both use one station).
#    After this section, index p refers to a MOORING ENTRY (not a model
#    station), and the model arrays are re-indexed so that lat_m[p],
#    Fu1_m[p], ... are the values of the model station of mooring entry p.
# ----------------------------------------------------------------------------
lon_m360 = mod.(lon_m, 360)


function nearest_station(lat0, lon0; tol = MATCH_TOL_DEG)
   d2 = (lat_m .- lat0) .^ 2 .+ (lon_m360 .- mod(lon0, 360)) .^ 2
   j = argmin(d2)
   return sqrt(d2[j]) <= tol ? j : nothing
end

struct MatchInfo
   dataset::Symbol   # :IWAP or :ALL
   idx::Int          # mooring number in that .mat file
end

entry_station = Int[]          # model station used by each mooring entry
matches       = MatchInfo[]    # which mooring each entry is


for (dsname, lats, lons) in ((:IWAP, lato3, lono3), (:ALL, lato1, lono1))
   for i in eachindex(lats)
       p = nearest_station(lats[i], lons[i])
       if p === nothing
           println("  dsname#i (lat=(lats[i]),lon=(lons[i])): no model station within tolerance -> skipped")
           continue
       end
       push!(entry_station, p)
       push!(matches, MatchInfo(dsname, i))
   end
end
# re-index the model arrays: one entry per mooring
N_stations = N_model
lat_m = lat_m[entry_station];  lon_m = lon_m[entry_station]
Fu1_m = Fu1_m[entry_station];  Fv1_m = Fv1_m[entry_station]
Fu2_m = Fu2_m[entry_station];  Fv2_m = Fv2_m[entry_station]
N_model = length(matches)

p_iwap = [p for p in 1:N_model if matches[p].dataset == :IWAP]
p_all  = [p for p in 1:N_model if matches[p].dataset == :ALL]
p_none = Int[]
println("\nMoorings compared: IWAP $(length(p_iwap)) / $(length(lato3)),  ALL $(length(p_all)) / $(length(lato1))")
println("Model stations not used by any ALL/IWAP mooring: ", setdiff(1:N_stations, entry_station))

station_label(m::MatchInfo) = m.dataset == :IWAP ? "IWAP$(m.idx)" :
                             m.dataset == :ALL  ? "M$(m.idx)" : "?"
label_of(p) = station_label(matches[p])

for s in unique(entry_station)
   who = [p for p in 1:N_model if entry_station[p] == s]
   length(who) > 1 && println("  model station $s shared by: ", join(label_of.(who), ", "))
end

# ----------------------------------------------------------------------------
# 4) FLUXES USED FOR COMPARISON (model is NOT rotated; IWAP moorings were
#    already rotated right after loading, in section 2)
# ----------------------------------------------------------------------------
model_uv(p, mode) = mode == 1 ? (Fu1_m[p], Fv1_m[p]) : (Fu2_m[p], Fv2_m[p])

function obs_uv(p, mode)
   m = matches[p]
   m.dataset == :IWAP && return Fuo3[m.idx, mode], Fvo3[m.idx, mode]
   m.dataset == :ALL  && return Fuo1[m.idx, mode], Fvo1[m.idx, mode]
   return NaN, NaN
end

direction360(u, v) = mod(atand(v, u), 360.0)   # 0 = east, counter-clockwise

# ----------------------------------------------------------------------------
# 5) DEFINE THE REGIONS
# ----------------------------------------------------------------------------
lon_m_geo = mod.(lon_m .+ 180, 360) .- 180     # -180..180 for mapping/grouping

# k-means on the unit sphere (dateline-safe), deterministic farthest-point init
function kmeans_sphere(lat, lon, k; niter = 300)
   n = length(lat)
   X = hcat(cosd.(lat) .* cosd.(lon), cosd.(lat) .* sind.(lon), sind.(lat))
   k = min(k, n)
   C = zeros(k, 3)
   mu = vec(mean(X, dims = 1))
   C[1, :] = X[argmax([norm(X[i, :] .- mu) for i in 1:n]), :]
   for j in 2:k
       d = [minimum(norm(X[i, :] .- C[l, :]) for l in 1:j-1) for i in 1:n]
       C[j, :] = X[argmax(d), :]
   end
   assign = zeros(Int, n)
   for _ in 1:niter
       new = [argmin([norm(X[i, :] .- C[l, :]) for l in 1:k]) for i in 1:n]
       new == assign && break
       assign = new
       for l in 1:k
           idx = findall(==(l), assign)
           isempty(idx) || (C[l, :] = vec(mean(X[idx, :], dims = 1)))
       end
   end
   return assign
end

fmt_lat(y) = @sprintf("%.1f°%s", abs(y), y >= 0 ? "N" : "S")
fmt_lon(x) = @sprintf("%.1f°%s", abs(x), x >= 0 ? "E" : "W")

tight_box(g) = (minimum(lon_m_geo[g]), maximum(lon_m_geo[g]), minimum(lat_m[g]), maximum(lat_m[g]))
inbox(x, y, b) = b[1] <= x <= b[2] && b[3] <= y <= b[4]
boxes_overlap(a, b) = !(a[2] < b[1] || b[2] < a[1] || a[4] < b[3] || b[4] < a[3])

# k-means groups by distance, so one group's bounding box can swallow a
# station of a neighbouring group. Fix: any station that lies inside another
# region's box is handed to that region (nearest centroid if several).
# Moving a station never enlarges any box, and each station moves at most
# once, so this always terminates.
function untangle_regions!(groups)
   moved = Set{Int}()
   while true
       changed = false
       boxes = [tight_box(g) for g in groups]
       for a in eachindex(groups), p in copy(groups[a])
           (p in moved || length(groups[a]) == 1) && continue
           hosts = [b for b in eachindex(groups) if b != a && inbox(lon_m_geo[p], lat_m[p], boxes[b])]
           isempty(hosts) && continue
           dist(h) = hypot(lon_m_geo[p] - mean(lon_m_geo[groups[h]]), lat_m[p] - mean(lat_m[groups[h]]))
           b = hosts[argmin(dist.(hosts))]
           filter!(!=(p), groups[a]); push!(groups[b], p); push!(moved, p)
           println("  untangle: moved $(label_of(p)) into the group whose box contained it")
           changed = true
           break
       end
       changed || break
   end
   return groups
end

regions = Tuple{String, String, Vector{Int}}[]   # (short name, long title, stations)


if REGION_MODE == :auto
   a = kmeans_sphere(lat_m[p_all], lon_m_geo[p_all], N_REGIONS)
   groups = [p_all[a .== l] for l in 1:maximum(a)]
   filter!(!isempty, groups)
   untangle_regions!(groups)
   filter!(!isempty, groups)
   tb = [tight_box(g) for g in groups]
   for i in eachindex(tb), j in i+1:length(tb)
       boxes_overlap(tb[i], tb[j]) &&
           @warn "Two auto regions still have overlapping boxes (no station is inside the wrong box). " *
                 "If this looks bad, switch to REGION_MODE = :manual."
   end
   sort!(groups, by = g -> mean(lon_m_geo[g]))            # west -> east
   for (r, g) in enumerate(groups)
       title = "Region $r($(fmt_lat(mean(lat_m[g]))), $(fmt_lon(mean(lon_m_geo[g]))))"
       push!(regions, ("Region$r", title, g))
   end
else
   for (name, (x0, x1, y0, y1)) in REGION_BOXES
       g = [p for p in p_all if x0 <= lon_m_geo[p] <= x1 && y0 <= lat_m[p] <= y1]
       push!(regions, (replace(name, " " => ""), name, g))
   end
   assigned = reduce(vcat, [r[3] for r in regions]; init = Int[])
   leftover = setdiff(p_all, assigned)
   if !isempty(leftover)
       @warn "$(length(leftover)) ALL-matched station(s) fall outside every REGION_BOX and are not plotted: " *
             join(label_of.(leftover), ", ")
   end
end
push!(regions, ("IWAP", "IWAP (mooring rotated 60°)", p_iwap))


println("\nRegion membership:")
for (short, title, g) in regions
   println("  $title  -> $(length(g)) stations: ", join(label_of.(g), ", "))
end


# ----------------------------------------------------------------------------
# 6) PLOTTING HELPERS
# ----------------------------------------------------------------------------
add_land!(ax) = poly!(ax, GeoMakie.land(); color = :lightgray, strokecolor = :gray40, strokewidth = 0.5)

function region_limits(ps; minpad = 1.0)
   x = lon_m_geo[ps]; y = lat_m[ps]
   dx = maximum(x) - minimum(x); dy = maximum(y) - minimum(y)
   px = max(0.2 * dx, minpad);   py = max(0.2 * dy, minpad)
   return (minimum(x) - px, maximum(x) + px, max(minimum(y) - py, -89.0), min(maximum(y) + py, 89.0))
end

# Arrows drawn as line segments (shaft + 2 head strokes), in lon/lat units.
function arrow_xy(x, y, u, v, scale; headfrac = 0.3, headang = 25.0)
   xs = Float64[]; ys = Float64[]
   for i in eachindex(x)
       (isnan(u[i]) || isnan(v[i])) && continue
       dx = u[i] * scale; dy = v[i] * scale
       L = hypot(dx, dy); L == 0 && continue
       xe = x[i] + dx; ye = y[i] + dy
       th = atan(dy, dx); hl = headfrac * L
       h1 = th + pi - deg2rad(headang); h2 = th + pi + deg2rad(headang)
       append!(xs, [x[i], xe, NaN, xe, xe + hl * cos(h1), NaN, xe, xe + hl * cos(h2), NaN])
       append!(ys, [y[i], ye, NaN, ye, ye + hl * sin(h1), NaN, ye, ye + hl * sin(h2), NaN])
   end
   return xs, ys
end


function nice_ref(x)                     # round reference arrow to 1, 2 or 5 x 10^n
   x <= 0 && return 1.0
   e = floor(log10(x)); f = x / 10^e
   return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * 10^e
end


panel_label!(pos, s) = Label(pos[1, 1, TopLeft()], s; fontsize = 22, font = :bold,
                            padding = (0, 10, 8, 0), halign = :right)


# Expand map limits so their width/height ratio matches the panel -> the map
# fills its panel completely (no empty bands left/right or above/below).
function fit_aspect(l, aspect)
   x0, x1, y0, y1 = l
   w = x1 - x0; h = y1 - y0
   if w / h < aspect
       e = (aspect * h - w) / 2; x0 -= e; x1 += e
   else
       e = (w / aspect - h) / 2; y0 -= e; y1 += e
   end
   return (max(x0, -180.0), min(x1, 180.0), max(y0, -90.0), min(y1, 90.0))
end

grow(b, d) = (b[1] - d, b[2] + d, b[3] - d, b[4] + d)

# Region boxes for the overview maps: tight around each region's stations,
# padded by BOX_PAD_DEG, but the padding is shrunk wherever it would make two
# boxes overlap or pull another region's station inside a box.
function display_boxes(groups; pad = BOX_PAD_DEG)
   tb = [tight_box(g) for g in groups]
   pads = fill(Float64(pad), length(tb))
   for _ in 1:30
       ok = true
       for i in eachindex(tb), j in eachindex(tb)
           i == j && continue
           bi = grow(tb[i], pads[i])
           foreign = any(inbox(lon_m_geo[p], lat_m[p], bi) for p in groups[j])
           clash   = boxes_overlap(bi, grow(tb[j], pads[j])) && !boxes_overlap(tb[i], tb[j])
           if (foreign && pads[i] > 1e-3) || clash
               pads[i] /= 2
               clash && (pads[j] /= 2)
               ok = false
           end
       end
       ok && break
   end
   return [grow(tb[i], pads[i]) for i in eachindex(tb)]
end

draw_box!(ax, b; kw...) = lines!(ax, [b[1], b[2], b[2], b[1], b[1]], [b[3], b[3], b[4], b[4], b[3]]; kw...)

# Overview GeoAxis zoomed to all matched stations, sized to fill its panel
function overview_axis(figpos, fig, title; pad = 5.0)
   allp = vcat(p_all, p_iwap)
   l = (minimum(lon_m_geo[allp]) - pad, maximum(lon_m_geo[allp]) + pad,
        minimum(lat_m[allp]) - pad,     maximum(lat_m[allp]) + pad)
   l = fit_aspect(l, 2.0)
   ax = GeoAxis(figpos; dest = "+proj=eqc", limits = l, title = title)
   colsize!(fig.layout, 1, Fixed(OVERVIEW_W))
   rowsize!(fig.layout, 1, Fixed(OVERVIEW_W * (l[4] - l[3]) / (l[2] - l[1])))
   return ax
end

# ----------------------------------------------------------------------------
# 7) OVERVIEW MAP: all regions (incl. IWAP) with boxes
# ----------------------------------------------------------------------------
reg_cols = Makie.wong_colors()
nonempty = [r for r in eachindex(regions) if !isempty(regions[r][3])]


# --- boxes: tight around each region + padding (never less than MINPAD_OV) ---
MINPAD_OV = 1.7                                   # minimum space (deg) around points
tb_ov   = [tight_box(regions[r][3]) for r in nonempty]
pads_ov = fill(Float64(BOX_PAD_DEG), length(tb_ov))
for _ in 1:30
   ok = true
   for i in eachindex(tb_ov), j in eachindex(tb_ov)
       i == j && continue
       bi = grow(tb_ov[i], pads_ov[i])
       # foreign points pulled in by the padding (ignore ones already inside the tight box)
       foreign = any(inbox(lon_m_geo[p], lat_m[p], bi) &&
                     !inbox(lon_m_geo[p], lat_m[p], tb_ov[i]) for p in regions[nonempty[j]][3])
       clash = boxes_overlap(bi, grow(tb_ov[j], pads_ov[j])) && !boxes_overlap(tb_ov[i], tb_ov[j])
       if (foreign || clash) && pads_ov[i] > MINPAD_OV
           pads_ov[i] = max(pads_ov[i] / 2, MINPAD_OV)
           clash && (pads_ov[j] = max(pads_ov[j] / 2, MINPAD_OV))
           ok = false
       end
   end
   ok && break
end
boxes_all = Dict(zip(nonempty, [grow(tb_ov[i], pads_ov[i]) for i in eachindex(tb_ov)]))


fig_ov = Figure(figure_padding = 12)
ax_ov = overview_axis(fig_ov[1, 1], fig_ov,
                    "Matched mooring stations grouped into $(length(regions)) regions")
add_land!(ax_ov)
for r in nonempty
  short, title, g = regions[r]
  c = reg_cols[mod1(r, length(reg_cols))]
  draw_box!(ax_ov, boxes_all[r]; color = c, linewidth = 1.5)
  scatter!(ax_ov, lon_m_geo[g], lat_m[g]; color = c, markersize = 9,
           strokecolor = :black, strokewidth = 0.4,
           marker = short == "IWAP" ? :diamond : :circle, label = title)
end
Legend(fig_ov[2, 1], ax_ov; orientation = :horizontal, nbanks = 2,
     tellwidth = false, framevisible = false)
resize_to_layout!(fig_ov)
display(fig_ov)




save(joinpath(FIGDIR, "Regions_overview_map.png"), fig_ov)
println("\nSaved: ", joinpath(FIGDIR, "Regions_overview_map.png"))



#=---------------------------------------------------------------------------
# 7b) REGION-BOX MAP: boxes around the ALL-station regions only;
#     IWAP stations get a distinct marker and NO box
# ----------------------------------------------------------------------------
box_r = [r for r in nonempty if regions[r][1] != "IWAP"]
boxes_reg = Dict(zip(box_r, display_boxes([regions[r][3] for r in box_r])))

fig_box = Figure(figure_padding = 12)
ax_box = overview_axis(fig_box[1, 1], fig_box,
                      "Mooring regions used for the model–mooring comparison")
add_land!(ax_box)

leg_elems = Any[]; leg_labels = String[]
for r in box_r
   short, title, g = regions[r]
   c = reg_cols[mod1(r, length(reg_cols))]
   b = boxes_reg[r]
   draw_box!(ax_box, b; color = c, linewidth = 2)
   scatter!(ax_box, lon_m_geo[g], lat_m[g]; color = c, markersize = 8,
            strokecolor = :black, strokewidth = 0.4)
   text!(ax_box, b[1], b[4]; text = "R$r", fontsize = 13, font = :bold, color = c,
         align = (:left, :bottom), offset = (2, 2))
   push!(leg_elems, [LineElement(color = c, linewidth = 2),
                     MarkerElement(color = c, marker = :circle, markersize = 8,
                                   strokecolor = :black, strokewidth = 0.4)])
   push!(leg_labels, "title [(length(g))]")
end

if !isempty(p_iwap)
   scatter!(ax_box, lon_m_geo[p_iwap], lat_m[p_iwap]; color = :black,
            marker = :star5, markersize = 16, strokecolor = :white, strokewidth = 0.6)
   push!(leg_elems, MarkerElement(color = :black, marker = :star5, markersize = 16,
                                  strokecolor = :white, strokewidth = 0.6))
   push!(leg_labels, "IWAP moorings  [$(length(p_iwap))]")
end

Legend(fig_box[2, 1], leg_elems, leg_labels;
      orientation = :horizontal, nbanks = 2, tellwidth = false, framevisible = false)
resize_to_layout!(fig_box)
display(fig_box)

box_png = joinpath(FIGDIR, "Regions_boxes_map.png")
save(box_png, fig_box)
println("Saved: $box_png")

=#
# ----------------------------------------------------------------------------
# 8) 14 FIGURES: each region x mode 1, 2 -- panels (a), (b), (c)
#    Every mooring gets its own bar pair. Moorings at the same site
#    (e.g. M2 and M7, different deployments) share the model value but keep
#    their own observed value; they are placed side by side and shaded grey.
# ----------------------------------------------------------------------------
println("\nTotal moorings in figures: ALL $(length(p_all)), IWAP $(length(p_iwap))")


for (short, title, g0) in regions, mode in 1:2
  if isempty(g0)
      @warn "$title has no stations -- skipping mode $mode figure."
      continue
  end
  # west -> east; moorings at the same site end up next to each other
  g = sort(g0, by = p -> (lon_m_geo[p], lat_m[p], matches[p].idx))
  n = length(g)
  labs = label_of.(g)
  x = lon_m_geo[g]; y = lat_m[g]


  # groups of moorings sharing one site (same model station)
  sites  = unique(entry_station[g])
  shared = [findall(==(s), entry_station[g]) for s in sites]
  shared = filter(k -> length(k) > 1, shared)


  # one label per site on the maps, e.g. "M2/M7"
  ukeys = unique(collect(zip(x, y)))
  ux = [k[1] for k in ukeys]; uy = [k[2] for k in ukeys]
  ulabs = [join(labs[(x .== k[1]) .& (y .== k[2])], "/") for k in ukeys]


  muv = [model_uv(p, mode) for p in g]; ouv = [obs_uv(p, mode) for p in g]
  mu = first.(muv); mv = last.(muv); ou = first.(ouv); ov = last.(ouv)
  mag_m = hypot.(mu, mv);            mag_o = hypot.(ou, ov)
  dir_m = direction360.(mu, mv);     dir_o = direction360.(ou, ov)


  # ---- region statistics (only stations with valid model AND obs) ----
  valid = .!isnan.(mag_m) .& .!isnan.(mag_o)
  nv = count(valid)
  if nv > 0
      pct  = mean(100 .* (mag_m[valid] .- mag_o[valid]) ./ mag_o[valid])
      dang = mean(abs.(mod.(dir_m[valid] .- dir_o[valid] .+ 180, 360) .- 180))
      stat_txt = ""#@sprintf("mean Δ|F| = %.1f%%,  mean |Δθ| = %.1f°  (%d/%d valid)", pct, dang, nv, n)
  else
      stat_txt = "no stations with valid model + obs data"
  end
  println("\ntitle (n moorings at $(length(sites)) sites): ", join(labs, ", "))
  for k in shared
      println("   same site: ", join(labs[k], ", "))
  end
  for k in 1:n
      (isnan(mag_m[k]) || isnan(mag_o[k])) &&
          println("   NaN flux at (labs[k])(model=(mag_m[k]), obs=$(mag_o[k]))")
  end
  lims  = fit_aspect(region_limits(g), MAP_W / MAP_H)   # map fills its panel
  wbars = max(BAR_W_MIN, BAR_W_PER_STATION * n)
  fig = Figure(figure_padding = 12)
  Label(fig[1, 1:3], "$title — Mode $mode", fontsize = 24, font = :bold)




  # ---------------- (a) zoomed location map ----------------
  ax_a = GeoAxis(fig[2, 1]; dest = "+proj=eqc", limits = lims,
                 title = "Moorings: $n  (sites: $(length(sites)))")
  add_land!(ax_a)
  scatter!(ax_a, ux, uy; color = short == "IWAP" ? :firebrick : :steelblue,
           markersize = 12, strokecolor = :black, strokewidth = 0.5)
  text!(ax_a, ux, uy; text = ulabs, fontsize = 10, offset = (5, 5), color = :black)
  panel_label!(fig[2, 1], "(a)")


  # ---------------- (b) magnitude + direction bars ----------------
  gb = fig[2, 2] = GridLayout()
  xs  = vcat(1:n, 1:n)
  grp = vcat(fill(1, n), fill(2, n))
  cols = [k == 1 ? COL_MODEL : COL_OBS for k in grp]


  ax_mag = Axis(gb[1, 1]; ylabel = "|F| (kW/m)", title = stat_txt,
                xticks = (1:n, labs), limits = (nothing, (0, nothing)))
  ax_dir = Axis(gb[2, 1]; ylabel = "Direction (°)", xlabel = "mooring station",
                xticks = (1:n, labs), xticklabelrotation = pi / 3,
                yticks = 0:90:360, limits = (nothing, (0, 360)))


  # grey band behind moorings that share a site
  for k in shared
      vspan!(ax_mag, minimum(k) - 0.5, maximum(k) + 0.5; color = (:gray, 0.18))
      vspan!(ax_dir, minimum(k) - 0.5, maximum(k) + 0.5; color = (:gray, 0.18))
  end


  barplot!(ax_mag, xs, vcat(mag_m, mag_o); dodge = grp, color = cols)
  hidexdecorations!(ax_mag; grid = false, ticks = false)
  barplot!(ax_dir, xs, vcat(dir_m, dir_o); dodge = grp, color = cols)
  linkxaxes!(ax_mag, ax_dir)
  rowgap!(gb, 6)
  panel_label!(gb, "(b)")


  # ---------------- (c) flux arrow comparison map ----------------
  ax_c = GeoAxis(fig[2, 3]; dest = "+proj=eqc", limits = lims,
                 title = "Flux vectors: model vs mooring")
  add_land!(ax_c)
  allmag = filter(!isnan, vcat(mag_m, mag_o))
  maxmag = isempty(allmag) ? 1.0 : maximum(allmag)
  span   = min(lims[2] - lims[1], lims[4] - lims[3])
  scale  = 0.25 * span / maxmag                       # longest arrow = 25% of map
  ax_m, ay_m = arrow_xy(x, y, mu, mv, scale)
  ax_o, ay_o = arrow_xy(x, y, ou, ov, scale)          # one blue arrow per mooring
  lines!(ax_c, ax_o, ay_o; color = COL_OBS,   linewidth = 2.2)
  lines!(ax_c, ax_m, ay_m; color = COL_MODEL, linewidth = 2.2)
  scatter!(ax_c, ux, uy; color = :black, markersize = 5)
  text!(ax_c, ux, uy; text = ulabs, fontsize = 9, offset = (-5, -12), color = :gray20)
  # reference arrow (bottom-left corner)
  ref = nice_ref(0.5 * maxmag)
  rx = lims[1] + 0.06 * (lims[2] - lims[1]); ry = lims[3] + 0.08 * (lims[4] - lims[3])
  rxs, rys = arrow_xy([rx], [ry], [ref], [0.0], scale)
  lines!(ax_c, rxs, rys; color = :black, linewidth = 2.2)
  text!(ax_c, rx, ry; text = @sprintf("%g kW/m", ref), fontsize = 11, offset = (0, 6))
  panel_label!(fig[2, 3], "(c)")


  # ---------------- legend & tight layout ----------------
  leg_el  = [PolyElement(color = COL_MODEL), PolyElement(color = COL_OBS)]
  leg_lab = ["Model (MITgcm)", "Mooring (obs)" * (short == "IWAP" ? ", rotated 60°" : "")]
  if !isempty(shared)
      push!(leg_el, PolyElement(color = (:gray, 0.18)))
      push!(leg_lab, "same site, different deployments")
  end
  Legend(fig[3, 1:3], leg_el, leg_lab;
         orientation = :horizontal, tellwidth = false, framevisible = false)


  colsize!(fig.layout, 1, Fixed(MAP_W))
  colsize!(fig.layout, 2, Fixed(wbars))
  colsize!(fig.layout, 3, Fixed(MAP_W))
  rowsize!(fig.layout, 2, Fixed(MAP_H))
  colgap!(fig.layout, 35)
  rowgap!(fig.layout, 8)
  resize_to_layout!(fig)                               # trim all empty space


  display(fig)
  outpng = joinpath(FIGDIR, "(short)mode(mode).png")
  save(outpng, fig)
  println("Saved: $outpng")
end


println("\nDone.")




