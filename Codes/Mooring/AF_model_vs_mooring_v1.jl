# =============================================================================
# Model (MITgcm) vs mooring modal energy flux -- REGIONAL comparison
#                        *** Matthew Alford M2 mooring data ***
#
#  * Reads Alford's intrfreq2_M2.nc (per-mode fluxes um_kwm / vm_kwm and the
#    mode-sum fluxes umtot / vmtot with Matthew's corrections applied).
#  * Matches every MITgcm point-flux station to the nearest Alford mooring
#    (dateline-safe, within MATCH_TOL_DEG). No IWAP / no rotation here.
#  * Matched stations are grouped into N_REGIONS geographic regions
#    (automatic k-means on the sphere, or manual lon/lat boxes).
#  * Global overview maps:
#       - all Alford moorings (matched ones coloured by region, unmatched grey)
#       - region boxes map
#  * For every region and every mode in MODES -> one figure with 3 panels:
#       (a) zoomed location map of the region (land + labeled stations)
#       (b) bar charts: |F| model vs mooring (top), direction 0-360 deg (bottom)
#       (c) zoomed flux-arrow map: model (red) vs mooring (blue) arrows
#  * A CSV table with every station / mode comparison.
# =============================================================================


using NCDatasets, Statistics, CairoMakie, GeoMakie, Printf, LinearAlgebra, DelimitedFiles


# ----------------------------------------------------------------------------
# 0) PATHS & SETTINGS -- EDIT THESE
# ----------------------------------------------------------------------------
# MITgcm fluxes extracted at the Alford mooring locations
# (must contain lon, lat, Fu_mode1, Fv_mode1, Fu_mode2, Fv_mode2 in kW/m)
model_flux_ncfile = "/home/aswathy/mnt/data/aswathy/MITgcm_NAS/Moorings/Mooring_modal_fluxes.nc"
alford_ncfile     = "/home/aswathy/Downloads/intrfreq2_M2.nc"


FIGDIR = "/home/aswathy/mnt/data/aswathy/Mooring_Data/Regional_figs_Alford/"
mkpath(FIGDIR)


MATCH_TOL_DEG = 0.05     # lat/lon matching tolerance (degrees)
APPLY_ALFORD_CORRECTIONS = true   # Matthew's corrections to umtot/vmtot (mode-sum only)


# Modes to compare: 1 = mode 1, 2 = mode 2, 0 = mode 1+2 total
# (for 0, the mooring uses the corrected umtot/vmtot; the model uses Fu1+Fu2)
MODES = (1, 2, 0)


N_REGIONS   = 6          # number of regions
REGION_MODE = :auto      # :auto   -> k-means clustering of station positions
                         # :manual -> use REGION_BOXES below


# Only used when REGION_MODE == :manual. Longitudes in -180..180.
# name => (lonmin, lonmax, latmin, latmax)
REGION_BOXES = [
    "Hawaii"            => (-165.0, -150.0,  15.0,  30.0),
    "Luzon / SCS"       => ( 110.0,  125.0,  15.0,  25.0),
    "NW Atlantic"       => ( -80.0,  -40.0,  20.0,  50.0),
    "NE Atlantic"       => ( -40.0,  -5.0,   30.0,  65.0),
    "NW Pacific"        => ( 125.0,  160.0,  20.0,  50.0),
    "Southern / Tasman" => ( 140.0,  180.0, -60.0, -20.0),
]


COL_MODEL = :crimson
COL_OBS   = :steelblue


# ---- layout settings (all figures are auto-trimmed to these sizes) ----
MAP_W  = 560            # width  (px) of each zoomed map panel (a) and (c)
MAP_H  = 460            # height (px) of the panel row
BAR_W_PER_STATION = 34  # width (px) per station in the bar panel (b)
BAR_W_MIN = 420         # minimum width (px) of the bar panel
OVERVIEW_W = 1400       # width (px) of the global overview maps
GLOBAL_LIMS = (-180.0, 180.0, -80.0, 70.0)
BOX_PAD_DEG = 1.5       # padding of region boxes on the overview maps (auto-shrunk to avoid overlaps)


# ----------------------------------------------------------------------------
# 1) LOAD MODEL FLUXES (kW/m, depth-integrated, time-averaged)
# ----------------------------------------------------------------------------
tofloat(a) = Float64.(coalesce.(Array(a), NaN))    # Missing -> NaN


ds = NCDataset(model_flux_ncfile, "r")
lon_m = tofloat(ds["lon"])
lat_m = tofloat(ds["lat"])
Fu1_m = tofloat(ds["Fu_mode1"]); Fv1_m = tofloat(ds["Fv_mode1"])
Fu2_m = tofloat(ds["Fu_mode2"]); Fv2_m = tofloat(ds["Fv_mode2"])
close(ds)
N_model = length(lat_m)
println("Loaded $N_model model stations from $model_flux_ncfile")


# ----------------------------------------------------------------------------
# 2) LOAD ALFORD MOORINGS
#    um_kwm, vm_kwm : (mooring, mode)   umtot, vmtot : (mooring)  [kW/m]
# ----------------------------------------------------------------------------
dsa = NCDataset(alford_ncfile, "r")
lat_a  = tofloat(dsa["lat"][:])
lon_a  = tofloat(dsa["lon"][:])
um_a   = tofloat(dsa["um_kwm"][:, :])
vm_a   = tofloat(dsa["vm_kwm"][:, :])
umtot_a = tofloat(dsa["umtot_kwm"][:])
vmtot_a = tofloat(dsa["vmtot_kwm"][:])
close(dsa)
N_alf = length(lat_a)
println("Alford: $N_alf moorings")


if APPLY_ALFORD_CORRECTIONS                     # Matthew's corrections (mode-sum)
    umtot_a[10] = 0.6423; vmtot_a[10] = 0.3952
    umtot_a[57] = 0.587;  vmtot_a[57] = 0.3030
    umtot_a[58] = 0.6767; vmtot_a[58] = 0.1443
    umtot_a[59] = 0.1878; vmtot_a[59] = 0.1880
    umtot_a[60] = 0.46;   vmtot_a[60] = 0.37
    println("Applied Alford corrections to umtot/vmtot at moorings 10, 57-60 " *
            "(per-mode um_kwm/vm_kwm are NOT corrected).")
end


wrap180(x) = mod(x + 180, 360) - 180
lon_a_geo = wrap180.(lon_a)                     # -180..180 for mapping
lon_m_geo = wrap180.(lon_m)


# ----------------------------------------------------------------------------
# 3) MATCH MODEL STATIONS -> ALFORD MOORINGS (dateline-safe)
# ----------------------------------------------------------------------------
function nearest_match(lat0, lon0, lats, lons; tol = MATCH_TOL_DEG)
    (isnan(lat0) || isnan(lon0)) && return nothing
    d2 = (lats .- lat0) .^ 2 .+ wrap180.(lons .- lon0) .^ 2
    d2 = replace(d2, NaN => Inf)
    j = argmin(d2)
    return sqrt(d2[j]) <= tol ? j : nothing
end


match_idx = Vector{Int}(undef, N_model)          # 0 = unmatched
for p in 1:N_model
    j = nearest_match(lat_m[p], lon_m[p], lat_a, lon_a)
    match_idx[p] = j === nothing ? 0 : j
end


p_all  = [p for p in 1:N_model if match_idx[p] > 0]
p_none = [p for p in 1:N_model if match_idx[p] == 0]
println("\nMatched to Alford: $(length(p_all)),  unmatched model stations: $(length(p_none))")
for p in p_none
    println("  unmatched model station p:lat=(lat_m[p]), lon=$(lon_m[p])")
end


# duplicates (several model stations on the same mooring)
for j in unique(match_idx[p_all])
    ps = [p for p in p_all if match_idx[p] == j]
    length(ps) > 1 && @warn "Alford mooring A$j matched by several model stations: $ps"
end


a_matched = falses(N_alf); a_matched[match_idx[p_all]] .= true
println("Alford moorings without a model station: $(count(.!a_matched)) -> ",
        join(["A$j" for j in findall(.!a_matched)], ", "))


label_of(p) = match_idx[p] > 0 ? "A$(match_idx[p])" : "?"


# ----------------------------------------------------------------------------
# 4) FLUXES USED FOR COMPARISON
# ----------------------------------------------------------------------------
function model_uv(p, mode)
    mode == 1 && return Fu1_m[p], Fv1_m[p]
    mode == 2 && return Fu2_m[p], Fv2_m[p]
    return Fu1_m[p] + Fu2_m[p], Fv1_m[p] + Fv2_m[p]          # mode 0 = total
end


function obs_uv(p, mode)
    j = match_idx[p]
    j == 0 && return NaN, NaN
    mode == 0 && return umtot_a[j], vmtot_a[j]
    return um_a[j, mode], vm_a[j, mode]
end


mode_title(mode) = mode == 0 ? "Mode 1+2 (total)" : "Mode $mode"
mode_tag(mode)   = mode == 0 ? "12" : string(mode)


direction360(u, v) = mod(atand(v, u), 360.0)   # 0 = east, counter-clockwise


# ----------------------------------------------------------------------------
# 5) DEFINE THE REGIONS
# ----------------------------------------------------------------------------
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


# Any station lying inside another region's box is handed to that region
# (nearest centroid if several). Each station moves at most once -> terminates.
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
        push!(regions, (replace(name, r"[^A-Za-z0-9]" => ""), name, g))
    end
    assigned = reduce(vcat, [r[3] for r in regions]; init = Int[])
    leftover = setdiff(p_all, assigned)
    if !isempty(leftover)
        @warn "$(length(leftover)) matched station(s) fall outside every REGION_BOX and are not plotted: " *
              join(label_of.(leftover), ", ")
    end
end


# regions straddling the dateline cannot be drawn with -180..180 limits
for (short, title, g) in regions
    !isempty(g) && maximum(lon_m_geo[g]) - minimum(lon_m_geo[g]) > 180 &&
        @warn "$title spans the dateline; its zoomed maps will be very wide. Consider :manual boxes."
end


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


# Expand map limits so their width/height ratio matches the panel
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


# Region boxes for the overview maps: padded by BOX_PAD_DEG, shrunk wherever
# the padding would make two boxes overlap or swallow another region's station.
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


# Global GeoAxis sized to fill its panel
function global_axis(figpos, fig, title)
    l = GLOBAL_LIMS
    ax = GeoAxis(figpos; dest = "+proj=eqc", limits = l, title = title,
                 xgridvisible = false, ygridvisible = false)
    colsize!(fig.layout, 1, Fixed(OVERVIEW_W))
    rowsize!(fig.layout, 1, Fixed(OVERVIEW_W * (l[4] - l[3]) / (l[2] - l[1])))
    return ax
end


# ----------------------------------------------------------------------------
# 7) GLOBAL OVERVIEW MAP: all Alford moorings, matched ones coloured by region
# ----------------------------------------------------------------------------
reg_cols = Makie.wong_colors()
regcol(r) = reg_cols[mod1(r, length(reg_cols))]
nonempty = [r for r in eachindex(regions) if !isempty(regions[r][3])]


fig_ov = Figure(figure_padding = 12)
ax_ov = global_axis(fig_ov[1, 1], fig_ov,
                    "Alford M2 moorings ($N_alf) and MITgcm matches grouped into $(length(nonempty)) regions")
add_land!(ax_ov)
if any(.!a_matched)
    scatter!(ax_ov, lon_a_geo[.!a_matched], lat_a[.!a_matched]; color = :white, marker = :xcross,
             markersize = 9, strokecolor = :gray30, strokewidth = 1.0,
             label = "Alford, no model match [$(count(.!a_matched))]")
end
for r in nonempty
    short, title, g = regions[r]
    scatter!(ax_ov, lon_m_geo[g], lat_m[g]; color = regcol(r), markersize = 10,
             strokecolor = :black, strokewidth = 0.5, label = "title[(length(g))]")
end
Legend(fig_ov[2, 1], ax_ov; orientation = :horizontal, nbanks = 3,
       tellwidth = false, framevisible = false)
resize_to_layout!(fig_ov)
display(fig_ov)


ov_png = joinpath(FIGDIR, "Alford_overview_map.png")
save(ov_png, fig_ov)
println("\nSaved: $ov_png")


# ----------------------------------------------------------------------------
# 7b) GLOBAL REGION-BOX MAP
# ----------------------------------------------------------------------------
boxes_reg = Dict(zip(nonempty, display_boxes([regions[r][3] for r in nonempty])))


fig_box = Figure(figure_padding = 12)
ax_box = global_axis(fig_box[1, 1], fig_box,
                     "Regions used for the MITgcm – Alford mooring comparison")
add_land!(ax_box)


leg_elems = Any[]; leg_labels = String[]
for r in nonempty
    short, title, g = regions[r]
    c = regcol(r)
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


Legend(fig_box[2, 1], leg_elems, leg_labels;
       orientation = :horizontal, nbanks = 3, tellwidth = false, framevisible = false)
resize_to_layout!(fig_box)
display(fig_box)


box_png = joinpath(FIGDIR, "Alford_regions_boxes_map.png")
save(box_png, fig_box)
println("Saved: $box_png")


# ----------------------------------------------------------------------------
# 8) REGION x MODE FIGURES -- panels (a), (b), (c)
# ----------------------------------------------------------------------------
csv_rows = Any[["region" "station" "model_idx" "lat" "lon" "mode" "absF_model_kWm" "absF_obs_kWm" "dir_model_deg" "dir_obs_deg" "pct_diff"]]


for (short, title, g) in regions, mode in MODES
    if isempty(g)
        @warn "$title has no stations -- skipping $(mode_title(mode)) figure."
        continue
    end
    n = length(g)
    labs = label_of.(g)
    x = lon_m_geo[g]; y = lat_m[g]


    muv = [model_uv(p, mode) for p in g]; ouv = [obs_uv(p, mode) for p in g]
    mu = first.(muv); mv = last.(muv); ou = first.(ouv); ov = last.(ouv)
    mag_m = hypot.(mu, mv);            mag_o = hypot.(ou, ov)
    dir_m = direction360.(mu, mv);     dir_o = direction360.(ou, ov)


    for k in 1:n
        pd = 100 * (mag_m[k] - mag_o[k]) / mag_o[k]
        push!(csv_rows, [title labs[k] g[k] y[k] x[k] mode_title(mode) mag_m[k] mag_o[k] dir_m[k] dir_o[k] pd])
    end


    # ---- region statistics (only stations with valid model AND obs) ----
    valid = .!isnan.(mag_m) .& .!isnan.(mag_o) .& (mag_o .> 0)
    nv = count(valid)
    if nv > 0
        pct  = mean(100 .* (mag_m[valid] .- mag_o[valid]) ./ mag_o[valid])
        dang = mean(abs.(mod.(dir_m[valid] .- dir_o[valid] .+ 180, 360) .- 180))
        stat_txt = ""#@sprintf("mean Δ|F| = %.1f%%,  mean |Δθ| = %.1f°  (%d/%d valid)", pct, dang, nv, n)
    else
        stat_txt = "no stations with valid model + obs data"
    end
    println("\n$title — $(mode_title(mode))")
    for k in 1:n
        (isnan(mag_m[k]) || isnan(mag_o[k])) &&
            println("   NaN flux at (labs[k])(model=(mag_m[k]), obs=$(mag_o[k]))")
    end


    lims  = fit_aspect(region_limits(g), MAP_W / MAP_H)
    wbars = max(BAR_W_MIN, BAR_W_PER_STATION * n)


    fig = Figure(figure_padding = 12)
    Label(fig[1, 1:3], "$title — $(mode_title(mode))", fontsize = 24, font = :bold)


    # ---------------- (a) zoomed location map ----------------
    ax_a = GeoAxis(fig[2, 1]; dest = "+proj=eqc", limits = lims,
                   title = "Station locations ($n)")
    add_land!(ax_a)
    scatter!(ax_a, x, y; color = :steelblue, markersize = 12,
             strokecolor = :black, strokewidth = 0.5)
    text!(ax_a, x, y; text = labs, fontsize = 10, offset = (5, 5), color = :black)
    panel_label!(fig[2, 1], "(a)")


    # ---------------- (b) magnitude + direction bars ----------------
    gb = fig[2, 2] = GridLayout()
    xs  = vcat(1:n, 1:n)
    grp = vcat(fill(1, n), fill(2, n))
    cols = [k == 1 ? COL_MODEL : COL_OBS for k in grp]


    ax_mag = Axis(gb[1, 1]; ylabel = "|F| (kW/m)", title = stat_txt,
                  xticks = (1:n, labs), limits = (nothing, (0, nothing)))
    barplot!(ax_mag, xs, vcat(mag_m, mag_o); dodge = grp, color = cols)
    hidexdecorations!(ax_mag; grid = false, ticks = false)


    ax_dir = Axis(gb[2, 1]; ylabel = "Direction (°)", xlabel = "mooring station",
                  xticks = (1:n, labs), xticklabelrotation = pi / 3,
                  yticks = 0:90:360, limits = (nothing, (0, 360)))
    barplot!(ax_dir, xs, vcat(dir_m, dir_o); dodge = grp, color = cols)
    linkxaxes!(ax_mag, ax_dir)
    rowgap!(gb, 6)
    panel_label!(gb, "(b)")


    # ---------------- (c) flux arrow comparison map ----------------
    ax_c = GeoAxis(fig[2, 3]; dest = "+proj=eqc", limits = lims,
                   title = "Flux vectors: model vs mooring")
    add_land!(ax_c)
    allmag = filter(!isnan, vcat(mag_m, mag_o))
    maxmag = isempty(allmag) || maximum(allmag) == 0 ? 1.0 : maximum(allmag)
    span   = min(lims[2] - lims[1], lims[4] - lims[3])
    scale  = 0.25 * span / maxmag                       # longest arrow = 25% of map
    ax_m, ay_m = arrow_xy(x, y, mu, mv, scale)
    ax_o, ay_o = arrow_xy(x, y, ou, ov, scale)
    lines!(ax_c, ax_o, ay_o; color = COL_OBS,   linewidth = 2.2)
    lines!(ax_c, ax_m, ay_m; color = COL_MODEL, linewidth = 2.2)
    scatter!(ax_c, x, y; color = :black, markersize = 5)
    text!(ax_c, x, y; text = labs, fontsize = 9, offset = (-5, -12), color = :gray20)
    ref = nice_ref(0.5 * maxmag)                         # reference arrow, bottom-left
    rx = lims[1] + 0.06 * (lims[2] - lims[1]); ry = lims[3] + 0.08 * (lims[4] - lims[3])
    rxs, rys = arrow_xy([rx], [ry], [ref], [0.0], scale)
    lines!(ax_c, rxs, rys; color = :black, linewidth = 2.2)
    text!(ax_c, rx, ry; text = @sprintf("%g kW/m", ref), fontsize = 11, offset = (0, 6))
    panel_label!(fig[2, 3], "(c)")


    # ---------------- legend & tight layout ----------------
    Legend(fig[3, 1:3],
           [PolyElement(color = COL_MODEL), PolyElement(color = COL_OBS)],
           ["Model (MITgcm)", "Mooring (Alford)" * (mode == 0 ? ", corrected mode-sum" : "")];
           orientation = :horizontal, tellwidth = false, framevisible = false)


    colsize!(fig.layout, 1, Fixed(MAP_W))
    colsize!(fig.layout, 2, Fixed(wbars))
    colsize!(fig.layout, 3, Fixed(MAP_W))
    rowsize!(fig.layout, 2, Fixed(MAP_H))
    colgap!(fig.layout, 35)
    rowgap!(fig.layout, 8)
    resize_to_layout!(fig)


    display(fig)
    outpng = joinpath(FIGDIR, "Alford_(short)mode(mode_tag(mode)).png")
    save(outpng, fig)
    println("Saved: $outpng")
end


# ----------------------------------------------------------------------------
# 9) COMPARISON TABLE
# ----------------------------------------------------------------------------
csv_file = joinpath(FIGDIR, "Alford_model_vs_mooring_fluxes.csv")
writedlm(csv_file, reduce(vcat, csv_rows), ',')
println("\nSaved: $csv_file")


println("\nDone.")


