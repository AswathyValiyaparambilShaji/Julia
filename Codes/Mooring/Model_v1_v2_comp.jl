# =============================================================================
# Model version 1 vs model version 2 modal energy flux -- REGIONAL comparison
#
#  Same layout as the model-vs-mooring script, but the two things compared are
#  the two MODEL flux files (v1 and v2). The mooring .mat files are only used
#  to decide WHERE to compare and HOW to label/group the stations, so the
#  regions and labels (M1, M2, ..., IWAP1, ...) are identical to your
#  model-vs-mooring figures.
#
#  * Every ALL and IWAP mooring position is matched to its nearest v1 station
#    AND its nearest v2 station (within MATCH_TOL_DEG).
#  * Matching priority / region logic is unchanged: ALL-matched positions are
#    grouped into N_REGIONS regions (k-means or manual boxes), IWAP positions
#    form their own region.
#  * For every region and mode 1, 2 -> one figure with 3 panels:
#       (a) zoomed location map of the region
#       (b) bar charts: |F| v1 vs v2 (top), direction 0-360 deg (bottom)
#       (c) zoomed flux-arrow map: v1 (red) vs v2 (green)
#  * Plus: overview map of regions, a v1-vs-v2 scatter summary figure, and a
#    CSV table with all station values.
#  * No rotation is applied to either model (both are true east/north).
# =============================================================================


using NCDatasets, MAT, Statistics, CairoMakie, GeoMakie, Printf, LinearAlgebra


# ----------------------------------------------------------------------------
# 0) PATHS & SETTINGS -- EDIT THESE
# ----------------------------------------------------------------------------
model_v1_ncfile = "/home/aswathy/mnt/data/aswathy/MITgcm_NAS/Moorings/Mooring_modal_fluxes_v2n.nc"
model_v2_ncfile = "/home/aswathy/mnt/data/aswathy/MITgcm_NAS/Moorings/Mooring_modal_fluxes_v2.nc"


# mooring files: used ONLY for station positions, labels and regions
file1path = "/home/aswathy/mnt/data/aswathy/Mooring_Data/Flux_mooring_timeseries_ALL.mat"
file3path = "/home/aswathy/mnt/data/aswathy/Mooring_Data/Flux_mooring_timeseries_ALL_IWAP.mat"

FIGDIR = "/home/aswathy/mnt/data/aswathy/Mooring_Data/Regional_figs_v1_vs_v2/"
mkpath(FIGDIR)


LABEL_V1 = "Model v1"
LABEL_V2 = "Model v2"


MATCH_TOL_DEG = 0.05     # lat/lon matching tolerance (degrees) mooring -> model station


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


COL_V1 = :crimson
COL_V2 = :seagreen


SHOW_STATS = true        # print mean Δ|F| and mean |Δθ| (v2 relative to v1) above panel (b)


# ---- layout settings (all figures are auto-trimmed to these sizes) ----
MAP_W  = 560            # width  (px) of each zoomed map panel (a) and (c)
MAP_H  = 460            # height (px) of the panel row
BAR_W_PER_STATION = 34  # width (px) per station in the bar panel (b)
BAR_W_MIN = 420         # minimum width (px) of the bar panel
OVERVIEW_W = 1300       # width (px) of the overview maps
BOX_PAD_DEG = 1.5       # padding of region boxes on the overview maps


# ----------------------------------------------------------------------------
# 1) LOAD BOTH MODEL FLUX FILES (kW/m, depth-integrated, time-averaged)
# ----------------------------------------------------------------------------
function load_model_flux(path)
  ds = NCDataset(path, "r")
  lat = Float64.(Array(ds["lat"]))
  lon = Float64.(Array(ds["lon"]))
  Fu1 = Float64.(Array(ds["Fu_mode1"])); Fv1 = Float64.(Array(ds["Fv_mode1"]))
  Fu2 = Float64.(Array(ds["Fu_mode2"])); Fv2 = Float64.(Array(ds["Fv_mode2"]))
  close(ds)
  return (; lat, lon, lon360 = mod.(lon, 360), Fu1, Fv1, Fu2, Fv2, n = length(lat))
end


M1 = load_model_flux(model_v1_ncfile)
M2 = load_model_flux(model_v2_ncfile)
println("Loaded $(M1.n) stations from $LABEL_V1: $model_v1_ncfile")
println("Loaded $(M2.n) stations from $LABEL_V2: $model_v2_ncfile")


# ----------------------------------------------------------------------------
# 2) LOAD MOORING POSITIONS (ALL and IWAP) -- positions only, fluxes unused
# ----------------------------------------------------------------------------
function load_mooring_pos(path)
  f = matopen(path)
  lato = vec(read(f, "lato"))
  lono = vec(read(f, "lono"))
  close(f)
  return Float64.(lato), Float64.(lono)
end


lato1, lono1 = load_mooring_pos(file1path)   # ALL
lato3, lono3 = load_mooring_pos(file3path)   # IWAP
println("ALL:  $(length(lato1)) mooring positions")
println("IWAP: $(length(lato3)) mooring positions")


# ----------------------------------------------------------------------------
# 3) MATCH EVERY MOORING POSITION -> NEAREST v1 STATION AND NEAREST v2 STATION
#    After this section, index p refers to a MOORING ENTRY.
#    st1[p] / st2[p] = matched station in v1 / v2 (0 = no station within tol).
# ----------------------------------------------------------------------------
function nearest_station(M, lat0, lon0; tol = MATCH_TOL_DEG)
  d2 = (M.lat .- lat0) .^ 2 .+ (M.lon360 .- mod(lon0, 360)) .^ 2
  j = argmin(d2)
  return sqrt(d2[j]) <= tol ? j : 0
end


struct MatchInfo
  dataset::Symbol   # :IWAP or :ALL
  idx::Int          # mooring number in that .mat file
end


matches = MatchInfo[]
st1 = Int[]; st2 = Int[]
lat_e = Float64[]; lon_e = Float64[]      # plotting position of each entry


for (dsname, lats, lons) in ((:IWAP, lato3, lono3), (:ALL, lato1, lono1))
  for i in eachindex(lats)
      j1 = nearest_station(M1, lats[i], lons[i])
      j2 = nearest_station(M2, lats[i], lons[i])
      if j1 == 0 && j2 == 0
          println("  (dsname)#(i) (lat=(lats[i]),lon=(lons[i])): no v1 or v2 station within tolerance -> skipped")
          continue
      end
      j1 == 0 && println("  (dsname)#(i): no $LABEL_V1 station within tolerance (v1 shown as NaN)")
      j2 == 0 && println("  (dsname)#(i): no $LABEL_V2 station within tolerance (v2 shown as NaN)")
      push!(matches, MatchInfo(dsname, i)); push!(st1, j1); push!(st2, j2)
      # plot at the v1 station position (v2 if v1 missing) so moorings at the
      # same site share exactly one dot on the maps
      if j1 > 0
          push!(lat_e, M1.lat[j1]); push!(lon_e, M1.lon[j1])
      else
          push!(lat_e, M2.lat[j2]); push!(lon_e, M2.lon[j2])
      end
  end
end
N_entry = length(matches)
lon_e_geo = mod.(lon_e .+ 180, 360) .- 180     # -180..180 for mapping/grouping


p_iwap = [p for p in 1:N_entry if matches[p].dataset == :IWAP]
p_all  = [p for p in 1:N_entry if matches[p].dataset == :ALL]
println("\nEntries compared: IWAP $(length(p_iwap)) / $(length(lato3)),  ALL $(length(p_all)) / $(length(lato1))")
println("  with both v1 and v2 matched: $(count((st1 .> 0) .& (st2 .> 0))) / $N_entry")
println("$LABEL_V1 stations not used: ", setdiff(1:M1.n, st1))
println("$LABEL_V2 stations not used: ", setdiff(1:M2.n, st2))


# Sanity check: distance between the matched v1 and v2 stations
for p in 1:N_entry
  (st1[p] > 0 && st2[p] > 0) || continue
  d = hypot(M1.lat[st1[p]] - M2.lat[st2[p]], M1.lon360[st1[p]] - M2.lon360[st2[p]])
  d > MATCH_TOL_DEG && println("  NOTE: entry $p: v1 and v2 stations are $(round(d, digits=3))° apart")
end


station_label(m::MatchInfo) = m.dataset == :IWAP ? "IWAP$(m.idx)" :
                              m.dataset == :ALL  ? "M$(m.idx)" : "?"
label_of(p) = station_label(matches[p])


site_key = [(st1[p], st2[p]) for p in 1:N_entry]   # same site = same v1 AND v2 station
for s in unique(site_key)
  who = [p for p in 1:N_entry if site_key[p] == s]
  length(who) > 1 && println("  site $s shared by: ", join(label_of.(who), ", "))
end


# ----------------------------------------------------------------------------
# 4) FLUXES USED FOR COMPARISON (no rotation for either model)
# ----------------------------------------------------------------------------
function model_uv(M, j, mode)
  j == 0 && return NaN, NaN
  return mode == 1 ? (M.Fu1[j], M.Fv1[j]) : (M.Fu2[j], M.Fv2[j])
end
v1_uv(p, mode) = model_uv(M1, st1[p], mode)
v2_uv(p, mode) = model_uv(M2, st2[p], mode)


direction360(u, v) = mod(atand(v, u), 360.0)   # 0 = east, counter-clockwise
angdiff(a, b) = mod(a - b + 180, 360) - 180     # signed a-b in -180..180


# ----------------------------------------------------------------------------
# 5) DEFINE THE REGIONS (same algorithm as the model-vs-mooring script)
# ----------------------------------------------------------------------------
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


tight_box(g) = (minimum(lon_e_geo[g]), maximum(lon_e_geo[g]), minimum(lat_e[g]), maximum(lat_e[g]))
inbox(x, y, b) = b[1] <= x <= b[2] && b[3] <= y <= b[4]
boxes_overlap(a, b) = !(a[2] < b[1] || b[2] < a[1] || a[4] < b[3] || b[4] < a[3])


function untangle_regions!(groups)
  moved = Set{Int}()
  while true
      changed = false
      boxes = [tight_box(g) for g in groups]
      for a in eachindex(groups), p in copy(groups[a])
          (p in moved || length(groups[a]) == 1) && continue
          hosts = [b for b in eachindex(groups) if b != a && inbox(lon_e_geo[p], lat_e[p], boxes[b])]
          isempty(hosts) && continue
          dist(h) = hypot(lon_e_geo[p] - mean(lon_e_geo[groups[h]]), lat_e[p] - mean(lat_e[groups[h]]))
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


regions = Tuple{String, String, Vector{Int}}[]   # (short name, long title, entries)


if REGION_MODE == :auto
  a = kmeans_sphere(lat_e[p_all], lon_e_geo[p_all], N_REGIONS)
  groups = [p_all[a .== l] for l in 1:maximum(a)]
  filter!(!isempty, groups)
  untangle_regions!(groups)
  filter!(!isempty, groups)
  tb = [tight_box(g) for g in groups]
  for i in eachindex(tb), j in i+1:length(tb)
      boxes_overlap(tb[i], tb[j]) &&
          @warn "Two auto regions still have overlapping boxes. If this looks bad, switch to REGION_MODE = :manual."
  end
  sort!(groups, by = g -> mean(lon_e_geo[g]))            # west -> east
  for (r, g) in enumerate(groups)
      title = "Region $r ($(fmt_lat(mean(lat_e[g]))), $(fmt_lon(mean(lon_e_geo[g]))))"
      push!(regions, ("Region$r", title, g))
  end
else
  for (name, (x0, x1, y0, y1)) in REGION_BOXES
      g = [p for p in p_all if x0 <= lon_e_geo[p] <= x1 && y0 <= lat_e[p] <= y1]
      push!(regions, (replace(name, " " => ""), name, g))
  end
  assigned = reduce(vcat, [r[3] for r in regions]; init = Int[])
  leftover = setdiff(p_all, assigned)
  if !isempty(leftover)
      @warn "$(length(leftover)) ALL station(s) fall outside every REGION_BOX and are not plotted: " *
            join(label_of.(leftover), ", ")
  end
end
push!(regions, ("IWAP", "IWAP", p_iwap))


region_of = fill("", N_entry)
println("\nRegion membership:")
for (short, title, g) in regions
  println("  $title  -> $(length(g)) stations: ", join(label_of.(g), ", "))
  region_of[g] .= short
end


# ----------------------------------------------------------------------------
# 6) PLOTTING HELPERS (unchanged)
# ----------------------------------------------------------------------------
add_land!(ax) = poly!(ax, GeoMakie.land(); color = :lightgray, strokecolor = :gray40, strokewidth = 0.5)


function region_limits(ps; minpad = 1.0)
  x = lon_e_geo[ps]; y = lat_e[ps]
  dx = maximum(x) - minimum(x); dy = maximum(y) - minimum(y)
  px = max(0.2 * dx, minpad);   py = max(0.2 * dy, minpad)
  return (minimum(x) - px, maximum(x) + px, max(minimum(y) - py, -89.0), min(maximum(y) + py, 89.0))
end


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


function nice_ref(x)
  x <= 0 && return 1.0
  e = floor(log10(x)); f = x / 10^e
  return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * 10^e
end


panel_label!(pos, s) = Label(pos[1, 1, TopLeft()], s; fontsize = 22, font = :bold,
                             padding = (0, 10, 8, 0), halign = :right)


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
draw_box!(ax, b; kw...) = lines!(ax, [b[1], b[2], b[2], b[1], b[1]], [b[3], b[3], b[4], b[4], b[3]]; kw...)


function overview_axis(figpos, fig, title; pad = 5.0)
  allp = vcat(p_all, p_iwap)
  l = (minimum(lon_e_geo[allp]) - pad, maximum(lon_e_geo[allp]) + pad,
       minimum(lat_e[allp]) - pad,     maximum(lat_e[allp]) + pad)
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


MINPAD_OV = 1.7
tb_ov   = [tight_box(regions[r][3]) for r in nonempty]
pads_ov = fill(Float64(BOX_PAD_DEG), length(tb_ov))
for _ in 1:30
  ok = true
  for i in eachindex(tb_ov), j in eachindex(tb_ov)
      i == j && continue
      bi = grow(tb_ov[i], pads_ov[i])
      foreign = any(inbox(lon_e_geo[p], lat_e[p], bi) &&
                    !inbox(lon_e_geo[p], lat_e[p], tb_ov[i]) for p in regions[nonempty[j]][3])
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
                      "$LABEL_V1 vs $LABEL_V2: stations grouped into $(length(regions)) regions")
add_land!(ax_ov)
for r in nonempty
  short, title, g = regions[r]
  c = reg_cols[mod1(r, length(reg_cols))]
  draw_box!(ax_ov, boxes_all[r]; color = c, linewidth = 1.5)
  scatter!(ax_ov, lon_e_geo[g], lat_e[g]; color = c, markersize = 9,
           strokecolor = :black, strokewidth = 0.4,
           marker = short == "IWAP" ? :diamond : :circle, label = title)
end
Legend(fig_ov[2, 1], ax_ov; orientation = :horizontal, nbanks = 2,
       tellwidth = false, framevisible = false)
resize_to_layout!(fig_ov)
display(fig_ov)
save(joinpath(FIGDIR, "Regions_overview_map_v1_vs_v2.png"), fig_ov)
println("\nSaved: ", joinpath(FIGDIR, "Regions_overview_map_v1_vs_v2.png"))


# ----------------------------------------------------------------------------
# 8) 14 FIGURES: each region x mode 1, 2 -- panels (a), (b), (c)
#    Moorings at the same site (same v1 and v2 station) have identical model
#    values; they are still shown side by side and shaded grey so the labels
#    match the model-vs-mooring figures.
# ----------------------------------------------------------------------------
for (short, title, g0) in regions, mode in 1:2
  if isempty(g0)
      @warn "$title has no stations -- skipping mode $mode figure."
      continue
  end
  g = sort(g0, by = p -> (lon_e_geo[p], lat_e[p], matches[p].idx))
  n = length(g)
  labs = label_of.(g)
  x = lon_e_geo[g]; y = lat_e[g]


  sites  = unique(site_key[g])
  shared = [findall(==(s), site_key[g]) for s in sites]
  shared = filter(k -> length(k) > 1, shared)


  ukeys = unique(collect(zip(x, y)))
  ux = [k[1] for k in ukeys]; uy = [k[2] for k in ukeys]
  ulabs = [join(labs[(x .== k[1]) .& (y .== k[2])], "/") for k in ukeys]


  uv1 = [v1_uv(p, mode) for p in g]; uv2 = [v2_uv(p, mode) for p in g]
  u1 = first.(uv1); v1 = last.(uv1); u2 = first.(uv2); v2 = last.(uv2)
  mag_1 = hypot.(u1, v1);          mag_2 = hypot.(u2, v2)
  dir_1 = direction360.(u1, v1);   dir_2 = direction360.(u2, v2)


  # ---- region statistics: v2 relative to v1 (only entries valid in both) ----
  valid = .!isnan.(mag_1) .& .!isnan.(mag_2) .& (mag_1 .> 0)
  nv = count(valid)
  if nv > 0
      pct  = mean(100 .* (mag_2[valid] .- mag_1[valid]) ./ mag_1[valid])
      dang = mean(abs.(angdiff.(dir_2[valid], dir_1[valid])))
      stat_txt = ""#    @sprintf("v2 vs v1: mean Δ|F| = %+.1f%%,  mean |Δθ| = %.1f°  (%d/%d valid)", pct, dang, nv, n) : ""
  else
      stat_txt = "no stations with valid v1 + v2 data"
  end
  println("\n$title — mode mode(n moorings at $(length(sites)) sites): ", join(labs, ", "))
  println("   ", stat_txt)
  for k in shared
      println("   same site: ", join(labs[k], ", "))
  end
  for k in 1:n
      (isnan(mag_1[k]) || isnan(mag_2[k])) &&
          println("   NaN flux at (labs[k])(v1=(mag_1[k]), v2=$(mag_2[k]))")
  end


  lims  = fit_aspect(region_limits(g), MAP_W / MAP_H)
  wbars = max(BAR_W_MIN, BAR_W_PER_STATION * n)
  fig = Figure(figure_padding = 12)
  Label(fig[1, 1:3], "$title — Mode $mode: $LABEL_V1 vs $LABEL_V2", fontsize = 24, font = :bold)


  # ---------------- (a) zoomed location map ----------------
  ax_a = GeoAxis(fig[2, 1]; dest = "+proj=eqc", limits = lims,
                 title = "Stations: $n  (sites: $(length(sites)))")
  add_land!(ax_a)
  scatter!(ax_a, ux, uy; color = short == "IWAP" ? :firebrick : :steelblue,
           markersize = 12, strokecolor = :black, strokewidth = 0.5)
  text!(ax_a, ux, uy; text = ulabs, fontsize = 10, offset = (5, 5), color = :black)
  panel_label!(fig[2, 1], "(a)")


  # ---------------- (b) magnitude + direction bars ----------------
  gb = fig[2, 2] = GridLayout()
  xs   = vcat(1:n, 1:n)
  grp  = vcat(fill(1, n), fill(2, n))
  cols = [k == 1 ? COL_V1 : COL_V2 for k in grp]


  ax_mag = Axis(gb[1, 1]; ylabel = "|F| (kW/m)", title = stat_txt,
                xticks = (1:n, labs), limits = (nothing, (0, nothing)))
  ax_dir = Axis(gb[2, 1]; ylabel = "Direction (°)", xlabel = "mooring station",
                xticks = (1:n, labs), xticklabelrotation = pi / 3,
                yticks = 0:90:360, limits = (nothing, (0, 360)))


  for k in shared
      vspan!(ax_mag, minimum(k) - 0.5, maximum(k) + 0.5; color = (:gray, 0.18))
      vspan!(ax_dir, minimum(k) - 0.5, maximum(k) + 0.5; color = (:gray, 0.18))
  end


  barplot!(ax_mag, xs, vcat(mag_1, mag_2); dodge = grp, color = cols)
  hidexdecorations!(ax_mag; grid = false, ticks = false)
  barplot!(ax_dir, xs, vcat(dir_1, dir_2); dodge = grp, color = cols)
  linkxaxes!(ax_mag, ax_dir)
  rowgap!(gb, 6)
  panel_label!(gb, "(b)")


  # ---------------- (c) flux arrow comparison map ----------------
  ax_c = GeoAxis(fig[2, 3]; dest = "+proj=eqc", limits = lims,
                 title = "Flux vectors: $LABEL_V1 vs $LABEL_V2")
  add_land!(ax_c)
  allmag = filter(!isnan, vcat(mag_1, mag_2))
  maxmag = isempty(allmag) || maximum(allmag) == 0 ? 1.0 : maximum(allmag)
  span   = min(lims[2] - lims[1], lims[4] - lims[3])
  scale  = 0.25 * span / maxmag
  a2x, a2y = arrow_xy(x, y, u2, v2, scale)
  a1x, a1y = arrow_xy(x, y, u1, v1, scale)
  lines!(ax_c, a2x, a2y; color = COL_V2, linewidth = 2.2)
  lines!(ax_c, a1x, a1y; color = COL_V1, linewidth = 2.2)
  scatter!(ax_c, ux, uy; color = :black, markersize = 5)
  text!(ax_c, ux, uy; text = ulabs, fontsize = 9, offset = (-5, -12), color = :gray20)
  ref = nice_ref(0.5 * maxmag)
  rx = lims[1] + 0.06 * (lims[2] - lims[1]); ry = lims[3] + 0.08 * (lims[4] - lims[3])
  rxs, rys = arrow_xy([rx], [ry], [ref], [0.0], scale)
  lines!(ax_c, rxs, rys; color = :black, linewidth = 2.2)
  text!(ax_c, rx, ry; text = @sprintf("%g kW/m", ref), fontsize = 11, offset = (0, 6))
  panel_label!(fig[2, 3], "(c)")


  # ---------------- legend & tight layout ----------------
  leg_el  = [PolyElement(color = COL_V1), PolyElement(color = COL_V2)]
  leg_lab = [LABEL_V1, LABEL_V2]
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
  resize_to_layout!(fig)


  display(fig)
  outpng = joinpath(FIGDIR, "$(short)_mode$(mode)_v1_vs_v2.png")
  save(outpng, fig)
  println("Saved: $outpng")
end


# ----------------------------------------------------------------------------
# 9) SUMMARY FIGURE: v2 vs v1 for every station, both modes
#    left: |F| (log-log, 1:1 line)   right: direction (with 1:1 line)
#    one point per SITE (duplicates at the same site are identical)
# ----------------------------------------------------------------------------
site_first = [findfirst(==(s), site_key) for s in unique(site_key)]
fig_s = Figure(size = (1100, 950), figure_padding = 14)
Label(fig_s[0, 1:2], "$LABEL_V2 vs $LABEL_V1 at all matched sites", fontsize = 22, font = :bold)


for mode in 1:2
  m1 = [hypot(v1_uv(p, mode)...) for p in site_first]
  m2 = [hypot(v2_uv(p, mode)...) for p in site_first]
  d1 = [direction360(v1_uv(p, mode)...) for p in site_first]
  d2 = [direction360(v2_uv(p, mode)...) for p in site_first]
  ok = .!isnan.(m1) .& .!isnan.(m2) .& (m1 .> 0) .& (m2 .> 0)


  axm = Axis(fig_s[mode, 1]; xscale = log10, yscale = log10,
             xlabel = "LABELV1|F|(kW/m)",ylabel="LABEL_V2 |F| (kW/m)",
             title = "Mode $mode magnitude", aspect = 1)
  axd = Axis(fig_s[mode, 2]; xlabel = "LABELV1direction(°)",ylabel="LABEL_V2 direction (°)",
             title = "Mode $mode direction", aspect = 1,
             xticks = 0:90:360, yticks = 0:90:360, limits = ((0, 360), (0, 360)))


  if count(ok) > 0
      lo = minimum(vcat(m1[ok], m2[ok])) / 1.5; hi = maximum(vcat(m1[ok], m2[ok])) * 1.5
      lines!(axm, [lo, hi], [lo, hi]; color = :gray50, linestyle = :dash)
      lines!(axd, [0, 360], [0, 360]; color = :gray50, linestyle = :dash)
      for (r, (short, title, g)) in enumerate(regions)
          sel = [k for k in eachindex(site_first) if ok[k] && region_of[site_first[k]] == short]
          isempty(sel) && continue
          c = reg_cols[mod1(r, length(reg_cols))]
          mk = short == "IWAP" ? :diamond : :circle
          scatter!(axm, m1[sel], m2[sel]; color = c, marker = mk, markersize = 10,
                   strokecolor = :black, strokewidth = 0.4, label = title)
          scatter!(axd, d1[sel], d2[sel]; color = c, marker = mk, markersize = 10,
                   strokecolor = :black, strokewidth = 0.4)
      end
      r_log = cor(log10.(m1[ok]), log10.(m2[ok]))
      ratio = exp(mean(log.(m2[ok] ./ m1[ok])))
      dth   = mean(abs.(angdiff.(d2[ok], d1[ok])))
      text!(axm, 0.03, 0.97; space = :relative, align = (:left, :top), fontsize = 12,
            text = @sprintf("n = %d\nr(log) = %.2f\ngeo-mean v2/v1 = %.2f", count(ok), r_log, ratio))
      text!(axd, 0.03, 0.97; space = :relative, align = (:left, :top), fontsize = 12,
            text = @sprintf("mean |Δθ| = %.1f°", dth))
      @printf("\nMode %d overall (%d sites): r(log|F|) = %.3f, geo-mean v2/v1 = %.3f, mean |Δθ| = %.1f°\n",
              mode, count(ok), r_log, ratio, dth)
  end
  mode == 1 && Legend(fig_s[3, 1:2], axm; orientation = :horizontal, nbanks = 2,
                      tellwidth = false, framevisible = false)
end
display(fig_s)
summary_png = joinpath(FIGDIR, "Summary_scatter_v1_vs_v2.png")
save(summary_png, fig_s)
println("Saved: $summary_png")


# ----------------------------------------------------------------------------
# 10) CSV TABLE: one row per mooring entry and mode
# ----------------------------------------------------------------------------
csvfile = joinpath(FIGDIR, "v1_vs_v2_station_table.csv")
open(csvfile, "w") do io
  println(io, "label,dataset,region,v1_station,v2_station,lat,lon,mode,",
              "Fu_v1,Fv_v1,mag_v1,dir_v1,Fu_v2,Fv_v2,mag_v2,dir_v2,pct_diff_mag,dtheta_deg")
  for p in 1:N_entry, mode in 1:2
      (a1, b1) = v1_uv(p, mode); (a2, b2) = v2_uv(p, mode)
      m1 = hypot(a1, b1); m2 = hypot(a2, b2)
      d1 = direction360(a1, b1); d2 = direction360(a2, b2)
      pct = m1 > 0 ? 100 * (m2 - m1) / m1 : NaN
      @printf(io, "%s,%s,%s,%d,%d,%.5f,%.5f,%d,%.6g,%.6g,%.6g,%.2f,%.6g,%.6g,%.6g,%.2f,%.2f,%.2f\n",
              label_of(p), matches[p].dataset, region_of[p], st1[p], st2[p],
              lat_e[p], lon_e_geo[p], mode, a1, b1, m1, d1, a2, b2, m2, d2, pct, angdiff(d2, d1))
  end
end
println("Saved: $csvfile")


println("\nDone.")




