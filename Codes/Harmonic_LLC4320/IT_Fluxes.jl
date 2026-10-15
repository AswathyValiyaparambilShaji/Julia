#=
IT_fluxes_simple.jl
-----------------------------------------------------------------------------
M2 internal-tide SSH amplitude with model (red) and mooring (blue) modal
energy-flux arrows on top.


  Figure 1  global internal-tide amplitude, moorings, region boxes
  Figure 2  one panel per region (ALL moorings), one figure per mode
  Figure 3  IWAP moorings, one panel per mode


Regions, boxes and arrows follow the model-mooring regional comparison script:
  * every arrow starts at the MODEL STATION position of its mooring
  * moorings at the same site (e.g. M2 and M7) share one model station, so the
    red model arrow is drawn ONCE per station; each mooring keeps its own blue arrow
  * panel limits = moorings + 20 % padding (at least 1 deg), stretched to the
    panel shape, so the map fills the panel (no white space)


No "$" string interpolation is used anywhere, so the text cannot get
garbled when the code is copied.
Run the whole file top to bottom.
-----------------------------------------------------------------------------
=#


using NCDatasets, MAT, Statistics, CairoMakie, Printf, LinearAlgebra


# =============================== SETTINGS ===================================
IT_FILE    = "/home/aswathy/mnt/data/aswathy/MITgcm_NAS/Moorings/M2_internal_tide_L150km.nc"
MODEL_FILE = "/home/aswathy/mnt/data/aswathy/MITgcm_NAS/Moorings/Mooring_modal_fluxes_v2n.nc"
ALL_FILE   = "/home/aswathy/mnt/data/aswathy/Mooring_Data/Flux_mooring_timeseries_ALL.mat"
IWAP_FILE  = "/home/aswathy/mnt/data/aswathy/Mooring_Data/Flux_mooring_timeseries_ALL_IWAP.mat"
FIGDIR     = "/home/aswathy/mnt/data/aswathy/Mooring_Data/IT_flux_figs/"
mkpath(FIGDIR)


IT_VAR   = "amp_it"   # background field: "amp_it", "amp_bt" or "amp_tot"
TAG      = "L150km"   # added to the figure file names
MODES    = (1, 2)     # flux modes to plot
TOL      = 0.05       # max distance (deg) between a mooring and its model station
ROT_IWAP = 60.0       # rotate the IWAP mooring fluxes by this many degrees
CMAX     = 5.0        # colour limit of the background [cm]


# ---- regions ----
REGION_MODE = :auto   # :auto   -> k-means grouping of the ALL moorings (as in the comparison script)
                      # :manual -> use REGION_BOXES below
N_REGIONS   = 6       # number of groups in :auto mode
BOX_PAD     = 1.5     # padding (deg) of the boxes on the global map
MIN_PAD     = 0.5     # box padding is shrunk (not below this) where boxes would overlap


# only used when REGION_MODE = :manual; name => (lon_min, lon_max, lat_min, lat_max), lon -180..180
REGION_BOXES = [
    "Region 1" => (-180.0, -150.0,  10.0, 30.0),
    "Region 2" => ( 110.0,  125.0,  15.0, 25.0),
]


# ---- panel size (px) ----
MAP_W = 560
MAP_H = 460


# ---- colours ----
DARK_RED  = RGBf(0.169, 0.000, 0.020)
CMAP_REG  = cgrad([RGBf(1.000, 1.000, 1.000), RGBf(0.996, 0.910, 0.784), RGBf(0.992, 0.733, 0.518),
                   RGBf(0.890, 0.290, 0.200), RGBf(0.600, 0.000, 0.051), DARK_RED])
LAND      = :gray55
COL_MODEL = :crimson
COL_OBS   = :royalblue3
# =============================================================================




# ----------------------------------------------------------------------------
# 1) MODEL FLUXES (one row per model station, one column per mode)
# ----------------------------------------------------------------------------
ds   = NCDataset(MODEL_FILE)
mlat = Float64.(Array(ds["lat"]))
mlon = mod.(Float64.(Array(ds["lon"])) .+ 180, 360) .- 180          # -180..180
mFu  = hcat(Float64.(Array(ds["Fu_mode1"])), Float64.(Array(ds["Fu_mode2"])))
mFv  = hcat(Float64.(Array(ds["Fv_mode1"])), Float64.(Array(ds["Fv_mode2"])))
close(ds)
println("Model stations: ", length(mlat))




# ----------------------------------------------------------------------------
# 2) MOORING FLUXES
# ----------------------------------------------------------------------------
function time_mean(A)                      # (mooring, mode, time) -> NaN-aware time mean
    out = fill(NaN, size(A, 1), size(A, 2))
    for i in axes(A, 1), j in axes(A, 2)
        v = filter(!isnan, A[i, j, :])
        isempty(v) || (out[i, j] = mean(v))
    end
    return out
end


function load_moorings(path)
    f   = matopen(path)
    lat = Float64.(vec(read(f, "lato")))
    lon = Float64.(vec(read(f, "lono")))
    Fu  = Float64.(read(f, "Fuo"))
    Fv  = Float64.(read(f, "Fvo"))
    close(f)
    if ndims(Fu) == 3
        Fu, Fv = time_mean(Fu), time_mean(Fv)
    end
    return lat, lon, Fu, Fv
end


lat_a, lon_a, Fu_a, Fv_a = load_moorings(ALL_FILE)
lat_i, lon_i, Fu_i, Fv_i = load_moorings(IWAP_FILE)


# rotate the IWAP mooring fluxes (the model is not rotated)
c, s = cosd(ROT_IWAP), sind(ROT_IWAP)
Fu_i, Fv_i = c .* Fu_i .- s .* Fv_i, s .* Fu_i .+ c .* Fv_i




# ----------------------------------------------------------------------------
# 3) PAIR EACH MOORING WITH ITS MODEL STATION
#    Each entry of `sites` is one mooring. lon/lat are the MODEL STATION
#    position (as in the comparison script); `station` is the station number,
#    shared by moorings at the same site.
# ----------------------------------------------------------------------------
sites = NamedTuple[]


function add_sites!(group, prefix, lat, lon, Fu, Fv)
    for i in eachindex(lat)
        d = hypot.(mlat .- lat[i], mod.(mlon, 360) .- mod(lon[i], 360))
        j = argmin(d)
        if d[j] > TOL
            println("  ", prefix, i, ": no model station within ", TOL, " deg -> skipped")
            continue
        end
        push!(sites, (group = group, label = string(prefix, i), station = j,
                      lon = mlon[j], lat = mlat[j],
                      obs_u = Fu[i, :], obs_v = Fv[i, :],
                      mod_u = mFu[j, :], mod_v = mFv[j, :]))
    end
end


add_sites!(:IWAP, "IWAP", lat_i, lon_i, Fu_i, Fv_i)
add_sites!(:ALL,  "M",    lat_a, lon_a, Fu_a, Fv_a)


all_sites  = filter(p -> p.group == :ALL,  sites)
iwap_sites = filter(p -> p.group == :IWAP, sites)
println("Matched: ALL ", length(all_sites), " / ", length(lat_a),
        ",  IWAP ", length(iwap_sites), " / ", length(lat_i))




# ----------------------------------------------------------------------------
# 4) REGIONS (same method as the comparison script)
# ----------------------------------------------------------------------------
col(s, f)           = Float64[getfield(p, f) for p in s]
fmt_lat(y)          = @sprintf("%.1f°%s", abs(y), y >= 0 ? "N" : "S")
fmt_lon(x)          = @sprintf("%.1f°%s", abs(x), x >= 0 ? "E" : "W")
inbox(x, y, b)      = b[1] <= x <= b[2] && b[3] <= y <= b[4]
boxes_overlap(a, b) = !(a[2] < b[1] || b[2] < a[1] || a[4] < b[3] || b[4] < a[3])
grow(b, d)          = (b[1] - d, b[2] + d, b[3] - d, b[4] + d)
tight_box(s)        = (minimum(col(s, :lon)), maximum(col(s, :lon)),
                       minimum(col(s, :lat)), maximum(col(s, :lat)))


alon = col(all_sites, :lon)
alat = col(all_sites, :lat)
tbox(g) = (minimum(alon[g]), maximum(alon[g]), minimum(alat[g]), maximum(alat[g]))


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


# a mooring lying inside another group's box is handed to that group
function untangle_regions!(groups)
    moved = Set{Int}()
    while true
        changed = false
        boxes = [tbox(g) for g in groups]
        for a in eachindex(groups), p in copy(groups[a])
            (p in moved || length(groups[a]) == 1) && continue
            hosts = [b for b in eachindex(groups) if b != a && inbox(alon[p], alat[p], boxes[b])]
            isempty(hosts) && continue
            dist(h) = hypot(alon[p] - mean(alon[groups[h]]), alat[p] - mean(alat[groups[h]]))
            b = hosts[argmin(dist.(hosts))]
            filter!(!=(p), groups[a]); push!(groups[b], p); push!(moved, p)
            changed = true
            break
        end
        changed || break
    end
    return groups
end


# regions = list of (name, box drawn on the global map, moorings)
regions = Tuple{String, NTuple{4, Float64}, Vector}[]


if REGION_MODE == :auto
    a = kmeans_sphere(alat, alon, N_REGIONS)
    groups = [findall(==(l), a) for l in 1:maximum(a)]
    filter!(!isempty, groups); untangle_regions!(groups); filter!(!isempty, groups)
    sort!(groups, by = g -> mean(alon[g]))                         # west -> east


    # global-map boxes: tight + padding, padding shrunk where boxes would overlap
    tb   = [tbox(g) for g in groups]
    pads = fill(Float64(BOX_PAD), length(tb))
    for _ in 1:30
        ok = true
        for i in eachindex(tb), j in eachindex(tb)
            i == j && continue
            bi      = grow(tb[i], pads[i])
            foreign = any(inbox(alon[p], alat[p], bi) && !inbox(alon[p], alat[p], tb[i]) for p in groups[j])
            clash   = boxes_overlap(bi, grow(tb[j], pads[j])) && !boxes_overlap(tb[i], tb[j])
            if (foreign || clash) && pads[i] > MIN_PAD
                pads[i] = max(pads[i] / 2, MIN_PAD)
                clash && (pads[j] = max(pads[j] / 2, MIN_PAD))
                ok = false
            end
        end
        ok && break
    end
    for (i, g) in enumerate(groups)
        name = string(fmt_lat(mean(alat[g])), ", ", fmt_lon(mean(alon[g])))
        push!(regions, (name, grow(tb[i], pads[i]), all_sites[g]))
    end
else
    for (name, b) in REGION_BOXES
        s_in = filter(p -> inbox(p.lon, p.lat, b), all_sites)
        isempty(s_in) ? println("  ", name, " contains no moorings -> dropped") :
                        push!(regions, (name, Float64.(b), s_in))
    end
end


println("\nRegions:")
for (k, (name, b, s_r)) in enumerate(regions)
    println("  R", k, "  ", name, "  -> ", length(s_r), " moorings: ", join([p.label for p in s_r], ", "))
end


IWAP_BOX = isempty(iwap_sites) ? nothing : grow(tight_box(iwap_sites), BOX_PAD)




# ----------------------------------------------------------------------------
# 5) INTERNAL-TIDE BACKGROUND (only the needed box is read)
# ----------------------------------------------------------------------------
ds_it  = NCDataset(IT_FILE)
it_lon = Float64.(ds_it["lon"][:])        # -180..180, increasing
it_lat = Float64.(ds_it["lat"][:])
it_var = variable(ds_it, IT_VAR)          # NaN on land


function read_it(x0, x1, y0, y1; maxpts = 1500)
    I = findall(x -> x0 <= x <= x1, it_lon)
    J = findall(y -> y0 <= y <= y1, it_lat)
    (isempty(I) || isempty(J)) && return Float64[], Float64[], zeros(Float32, 0, 0)
    step = max(1, cld(max(length(I), length(J)), maxpts))
    I = first(I):step:last(I)
    J = first(J):step:last(J)
    return it_lon[I], it_lat[J], 100 .* Float32.(it_var[I, J])
end




# ----------------------------------------------------------------------------
# 6) PLOTTING HELPERS (arrows, limits and scaling as in the comparison script)
# ----------------------------------------------------------------------------
# panel limits: moorings + 20 % padding (min 1 deg)
function region_limits(s; minpad = 1.0)
    x = col(s, :lon); y = col(s, :lat)
    px = max(0.2 * (maximum(x) - minimum(x)), minpad)
    py = max(0.2 * (maximum(y) - minimum(y)), minpad)
    return (minimum(x) - px, maximum(x) + px, max(minimum(y) - py, -89.0), min(maximum(y) + py, 89.0))
end


# stretch the limits to the panel shape so the map fills the panel
function fit_aspect(l, aspect)
    x0, x1, y0, y1 = l
    w = x1 - x0; h = y1 - y0
    if w / h < aspect
        e = (aspect * h - w) / 2; x0 -= e; x1 += e
    else
        e = (w / aspect - h) / 2; y0 -= e; y1 += e
    end
    return (x0, x1, y0, y1)
end


panel_limits(s) = fit_aspect(region_limits(s), MAP_W / MAP_H)


# arrows as line segments (shaft + 2 head strokes), in lon/lat units
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


function nice_ref(x)                       # 1, 2 or 5 x 10^n
    x <= 0 && return 1.0
    e = floor(log10(x)); f = x / 10^e
    return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * 10^e
end


function draw_box!(ax, b, label, color)
    lines!(ax, [b[1], b[2], b[2], b[1], b[1]], [b[3], b[3], b[4], b[4], b[3]];
           color = color, linewidth = 1.5)
    text!(ax, b[1], b[4]; text = label, color = color, font = :bold, fontsize = 15,
          align = (:left, :bottom), offset = (2, 2))
end


# One map panel: background + one blue arrow per mooring + one red arrow per model station
function flux_map!(pos, s, mode, lims; title = "")
    ax = Axis(pos; title = title, width = MAP_W, height = MAP_H, limits = lims,
              xlabel = "Longitude (°)", ylabel = "Latitude (°)")
    lon, lat, A = read_it(lims...)
    isempty(A) || heatmap!(ax, lon, lat, A; colormap = CMAP_REG, colorrange = (0, CMAX),
                           highclip = DARK_RED, nan_color = LAND, rasterize = true)
    isempty(s) && return ax


    # moorings (blue): one arrow each, from its model station
    x  = col(s, :lon);  y  = col(s, :lat)
    ou = Float64[p.obs_u[mode] for p in s];  ov = Float64[p.obs_v[mode] for p in s]


    # model (red): one arrow per station, even if several moorings share it
    st = unique(p -> p.station, s)
    xm = col(st, :lon); ym = col(st, :lat)
    mu = Float64[p.mod_u[mode] for p in st]; mv = Float64[p.mod_v[mode] for p in st]


    # longest arrow = 25 % of the panel's shorter side
    mags   = filter(isfinite, vcat(hypot.(mu, mv), hypot.(ou, ov)))
    maxmag = isempty(mags) ? 1.0 : maximum(mags)
    scale  = 0.25 * min(lims[2] - lims[1], lims[4] - lims[3]) / maxmag


    xo, yo = arrow_xy(x, y, ou, ov, scale)
    isempty(xo) || lines!(ax, xo, yo; color = COL_OBS, linewidth = 2.2)
    xmo, ymo = arrow_xy(xm, ym, mu, mv, scale)
    isempty(xmo) || lines!(ax, xmo, ymo; color = COL_MODEL, linewidth = 2.2)
    scatter!(ax, xm, ym; color = :black, markersize = 5)


    # reference arrow (bottom-left)
    ref = nice_ref(0.5 * maxmag)
    rx = lims[1] + 0.06 * (lims[2] - lims[1]); ry = lims[3] + 0.08 * (lims[4] - lims[3])
    rxs, rys = arrow_xy([rx], [ry], [ref], [0.0], scale)
    lines!(ax, rxs, rys; color = :black, linewidth = 2.2)
    text!(ax, rx, ry; text = @sprintf("%g kW/m", ref), fontsize = 11, offset = (0, 6))
    return ax
end


add_legend!(pos, obs_note = "") = Legend(pos,
    [LineElement(color = COL_MODEL, linewidth = 3), LineElement(color = COL_OBS, linewidth = 3)],
    ["Model (MITgcm)", string("Mooring (obs)", obs_note)];
    orientation = :horizontal, framevisible = false, tellwidth = false)


add_colorbar!(pos) = Colorbar(pos; colormap = CMAP_REG, limits = (0, CMAX), highclip = DARK_RED,
                              label = "M2 internal-tide SSH amplitude (cm)")




# ----------------------------------------------------------------------------
# 7) FIGURE 1: global map + moorings + region boxes
# ----------------------------------------------------------------------------
println("\nReading global field ...")
glon, glat, G = read_it(-180, 180, -90, 90; maxpts = 4000)
ylo, yhi = max(-80.0, glat[1]), glat[end]
GW = 1500


fig = Figure(fontsize = 18, figure_padding = 12)
ax  = Axis(fig[1, 1]; title = "M2 internal-tide SSH amplitude and mooring locations",
           width = GW, height = round(Int, GW * (yhi - ylo) / 360),
           xlabel = "Longitude (°)", ylabel = "Latitude (°)",
           limits = (-180, 180, ylo, yhi), xticks = -180:60:180, yticks = -60:30:60)
hm = heatmap!(ax, glon, glat, G; colormap = :magma, colorrange = (0, CMAX),
              highclip = :white, nan_color = LAND, rasterize = true)


for (k, (name, b, s_r)) in enumerate(regions)
    draw_box!(ax, b, string("R", k), :cyan)
end
IWAP_BOX === nothing || draw_box!(ax, IWAP_BOX, "IWAP", :lime)


st_all  = unique(p -> p.station, all_sites)
st_iwap = unique(p -> p.station, iwap_sites)
sa = scatter!(ax, col(st_all, :lon), col(st_all, :lat); color = :white,
              markersize = 9, strokecolor = :black, strokewidth = 1)
si = scatter!(ax, col(st_iwap, :lon), col(st_iwap, :lat); color = :lime, marker = :star5,
              markersize = 14, strokecolor = :black, strokewidth = 0.8)


Colorbar(fig[1, 2], hm; label = "Amplitude (cm)")
Legend(fig[2, 1], [sa, si],
       [string("ALL moorings [", length(all_sites), "]"),
        string("IWAP moorings [", length(iwap_sites), "]")];
       orientation = :horizontal, framevisible = false, tellwidth = false)
resize_to_layout!(fig)
save(joinpath(FIGDIR, string("IT_global_moorings_", TAG, ".png")), fig; px_per_unit = 2)
display(fig)
G = nothing; GC.gc()




# ----------------------------------------------------------------------------
# 8) FIGURE 2: one panel per region, one figure per mode
# ----------------------------------------------------------------------------
nreg = length(regions)
ncol = min(3, nreg)
nrow = cld(nreg, ncol)


for mode in MODES
    fig = Figure(fontsize = 16, figure_padding = 12)
    Label(fig[0, 1:ncol], string("Mode-", mode, " semidiurnal energy flux on M2 internal-tide SSH amplitude"),
          fontsize = 22, font = :bold)
    for (k, (name, b, s_r)) in enumerate(regions)
        row, column = fldmod1(k, ncol)
        title = string("(", 'a' + k - 1, ") R", k, ": ", name, "  [", length(s_r), " moorings]")
        flux_map!(fig[row, column], s_r, mode, panel_limits(s_r); title = title)
    end
    add_colorbar!(fig[1:nrow, ncol + 1])
    add_legend!(fig[nrow + 1, 1:ncol])
    colgap!(fig.layout, 25); rowgap!(fig.layout, 15)
    resize_to_layout!(fig)
    save(joinpath(FIGDIR, string("IT_flux_regions_mode", mode, "_", TAG, ".png")), fig; px_per_unit = 2)
    display(fig)
end




# ----------------------------------------------------------------------------
# 9) FIGURE 3: IWAP, one panel per mode
# ----------------------------------------------------------------------------
if !isempty(iwap_sites)
    nm   = length(MODES)
    lims = panel_limits(iwap_sites)
    fig  = Figure(fontsize = 16, figure_padding = 12)
    Label(fig[0, 1:nm], "IWAP: semidiurnal energy flux on M2 internal-tide SSH amplitude",
          fontsize = 22, font = :bold)
    for (k, mode) in enumerate(MODES)
        title = string("(", 'a' + k - 1, ") Mode ", mode, "  [", length(iwap_sites), " moorings]")
        flux_map!(fig[1, k], iwap_sites, mode, lims; title = title)
    end
    add_colorbar!(fig[1, nm + 1])
    add_legend!(fig[2, 1:nm], string(", rotated ", ROT_IWAP, "°"))
    colgap!(fig.layout, 25)
    resize_to_layout!(fig)
    save(joinpath(FIGDIR, string("IT_flux_IWAP_", TAG, ".png")), fig; px_per_unit = 2)
    display(fig)
else
    println("No IWAP moorings matched -> figure 3 skipped")
end


close(ds_it)
println("Done.")


