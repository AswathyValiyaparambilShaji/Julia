#=
plot_IT_moorings_fluxes.jl
-----------------------------------------------------------------------------
Internal-tide SSH amplitude (Gaussian high-pass, from M2_internal_tide.jl)
with mooring locations and semidiurnal modal energy-flux vectors on top.


Figures (one set per mode in MODES):
  1. IT_global_moorings_<TAG>.png
       global M2 internal-tide amplitude, ALL moorings (circles), IWAP moorings
       (stars), and the boxes R1..Rn that the regional panels show
  2. IT_flux_regions_mode<m>_<TAG>.png
       one figure, one subplot per ALL region (6 by default):
       IT amplitude background + model (red) and mooring (blue) flux vectors
  3. IT_flux_IWAP_<TAG>.png
       same idea for the IWAP moorings, one subplot per mode


The IT field is read straight from the NetCDF in pieces, so the full
17280 x 12960 array is never held in memory:
  - global map  : block-averaged F_GLOBAL x F_GLOBAL (default 4 -> 4320 x 3240)
  - regional map: full 1/48° resolution, block-averaged only if a panel would
                  have more than MAX_PTS_REG points along one side
Land (NaN in the file) is drawn grey, so no coastline package is needed.
-----------------------------------------------------------------------------
=#


using NCDatasets, MAT, Statistics, CairoMakie, Printf, LinearAlgebra


# ============================ 0) PATHS & SETTINGS ============================
IT_NCFILE         = "/home/aswathy/mnt/data/aswathy/MITgcm_NAS/Moorings/M2_internal_tide_L150km.nc"  # <- EDIT
model_flux_ncfile = "/home/aswathy/mnt/data/aswathy/MITgcm_NAS/Moorings/Mooring_modal_fluxes_v2n.nc"
file1path         = "/home/aswathy/mnt/data/aswathy/Mooring_Data/Flux_mooring_timeseries_ALL.mat"
file3path         = "/home/aswathy/mnt/data/aswathy/Mooring_Data/Flux_mooring_timeseries_ALL_IWAP.mat"


FIGDIR = "/home/aswathy/mnt/data/aswathy/Mooring_Data/IT_flux_figs/"
mkpath(FIGDIR)


IT_VAR = "amp_it"       # variable plotted in the background ("amp_it", "amp_bt", "amp_tot")
TAG    = "L150km"       # only used in figure names


MATCH_TOL_DEG = 0.05    # lat/lon matching tolerance (degrees)
THETA_ROT     = pi / 3  # 60 deg rotation of the IWAP MOORING fluxes (model not rotated)


N_REGIONS   = 6         # number of regions for the ALL-matched stations
REGION_MODE = :auto     # :auto -> k-means of station positions, :manual -> REGION_BOXES
REGION_BOXES = [        # only for :manual; name => (lonmin, lonmax, latmin, latmax), lon -180..180
    "Region 1" => (-180.0, -150.0,  10.0,  30.0),
    "Region 2" => ( 110.0,  125.0,  15.0,  25.0),
    "Region 3" => (-80.0,  -60.0,   20.0,  45.0),
    "Region 4" => (-40.0,  -10.0,   30.0,  60.0),
    "Region 5" => ( 130.0,  150.0,  20.0,  40.0),
    "Region 6" => ( 40.0,   80.0,  -10.0,  25.0),
]


MODES = (1, 2)          # flux modes to plot


# ---- background field ----
F_GLOBAL      = 4       # block average for the global map (1 = every grid point, very slow)
MAX_PTS_REG   = 1500    # max points along one side of a regional panel
IT_MAX_GLOBAL = 5.0     # colour limit, global map [cm]
IT_MAX_REG    = 5.0     # colour limit, regional panels [cm] (shared by all panels)


AMP_COLORS = [RGBf(1.000, 1.000, 1.000), RGBf(0.996, 0.910, 0.784), RGBf(0.992, 0.733, 0.518),
              RGBf(0.890, 0.290, 0.200), RGBf(0.600, 0.000, 0.051), RGBf(0.169, 0.000, 0.020)]
CMAP_GLOBAL = :magma
CMAP_REG    = cgrad(AMP_COLORS)      # light background so the arrows stay visible
LAND_COLOR  = :gray55


# ---- regional panels ----
REG_PAD_DEG  = 2.0      # minimum padding (deg) around the moorings of a region
PANEL_ASPECT = 1.3      # lon/lat ratio of each regional panel (all panels same shape)
ARROW_FRAC   = 0.25     # longest arrow = this fraction of the panel's shorter side
COL_MODEL    = :crimson
COL_OBS      = :royalblue3
# ==============================================================================


letters = 'a':'z'
fmt_lat(y) = @sprintf("%.1f°%s", abs(y), y >= 0 ? "N" : "S")
fmt_lon(x) = @sprintf("%.1f°%s", abs(x), x >= 0 ? "E" : "W")


# ----------------------------------------------------------------------------
# 1) INTERNAL-TIDE FIELD (read in pieces from the NetCDF)
# ----------------------------------------------------------------------------
ds_it  = NCDataset(IT_NCFILE, "r")
LON_IT = Float64.(collect(ds_it["lon"][:]))      # -180..180, increasing
LAT_IT = Float64.(collect(ds_it["lat"][:]))      # increasing
VAR_IT = variable(ds_it, IT_VAR)                 # raw variable: NaN on land, no Missing
println("IT field: ", IT_VAR, "  ", size(VAR_IT), "  lon ", LON_IT[1], "..", LON_IT[end],
        "  lat ", round(LAT_IT[1], digits = 2), "..", round(LAT_IT[end], digits = 2))


"NaN-aware f×f block mean (block kept if at least half of it is ocean)."
function blockmean_nan(A::AbstractMatrix, f::Int)
    f == 1 && return Float32.(A)
    m1, m2 = size(A) .÷ f
    out = fill(NaN32, m1, m2)
    Threads.@threads for J in 1:m2
        for I in 1:m1
            s = 0.0f0; w = 0
            @inbounds for j in (J-1)*f+1:J*f, i in (I-1)*f+1:I*f
                a = Float32(A[i, j])
                if isfinite(a); s += a; w += 1; end
            end
            out[I, J] = 2w >= f * f ? s / w : NaN32
        end
    end
    return out
end
bm_vec(v, f) = f == 1 ? v : [mean(@view v[(k-1)*f+1:k*f]) for k in 1:length(v)÷f]


"Global field [cm], block averaged f×f, read in latitude strips."
function read_it_global(f)
    n1, n2 = size(VAR_IT)
    m1, m2 = n1 ÷ f, n2 ÷ f
    rows = f * max(1, 1620 ÷ f)                  # lat rows per read, multiple of f
    out = Matrix{Float32}(undef, m1, m2)
    for j0 in 1:rows:m2*f
        j1 = min(j0 + rows - 1, m2 * f)
        A = VAR_IT[1:m1*f, j0:j1]
        out[:, (j0-1)÷f+1:j1÷f] = blockmean_nan(A, f)
    end
    return 100 .* out, bm_vec(LON_IT, f), bm_vec(LAT_IT, f)
end


"Regional field [cm] inside (x0, x1, y0, y1), full resolution unless too large."
function read_it_region(x0, x1, y0, y1)
    I = searchsortedfirst(LON_IT, x0):searchsortedlast(LON_IT, x1)
    J = searchsortedfirst(LAT_IT, y0):searchsortedlast(LAT_IT, y1)
    (isempty(I) || isempty(J)) && return Matrix{Float32}(undef, 0, 0), Float64[], Float64[]
    f = max(1, cld(max(length(I), length(J)), MAX_PTS_REG))
    A = blockmean_nan(VAR_IT[I, J], f)
    return 100 .* A, bm_vec(LON_IT[I], f), bm_vec(LAT_IT[J], f)
end


# ----------------------------------------------------------------------------
# 2) MODEL FLUXES AND MOORINGS (same as the mooring comparison script)
# ----------------------------------------------------------------------------
ds = NCDataset(model_flux_ncfile, "r")
lon_m = Float64.(Array(ds["lon"])); lat_m = Float64.(Array(ds["lat"]))
Fu1_m = Float64.(Array(ds["Fu_mode1"])); Fv1_m = Float64.(Array(ds["Fv_mode1"]))
Fu2_m = Float64.(Array(ds["Fu_mode2"])); Fv2_m = Float64.(Array(ds["Fv_mode2"]))
close(ds)
println("Loaded $(length(lat_m)) model stations")


function load_mooring_mat(path)
    f = matopen(path)
    lato = vec(read(f, "lato")); lono = vec(read(f, "lono"))
    Fuo = read(f, "Fuo");        Fvo = read(f, "Fvo")
    close(f)
    if ndims(Fuo) == 3                     # (mooring, mode, time) -> NaN-aware time mean
        n1, n2, _ = size(Fuo)
        Fu2 = fill(NaN, n1, n2); Fv2 = fill(NaN, n1, n2)
        for i in 1:n1, j in 1:n2
            vu = filter(!isnan, @view Fuo[i, j, :]); vv = filter(!isnan, @view Fvo[i, j, :])
            Fu2[i, j] = isempty(vu) ? NaN : mean(vu)
            Fv2[i, j] = isempty(vv) ? NaN : mean(vv)
        end
        Fuo, Fvo = Fu2, Fv2
    end
    return Float64.(lato), Float64.(lono), Float64.(Fuo), Float64.(Fvo)
end


lato1, lono1, Fuo1, Fvo1 = load_mooring_mat(file1path)   # ALL
lato3, lono3, Fuo3, Fvo3 = load_mooring_mat(file3path)   # IWAP
Fuo3, Fvo3 = cos(THETA_ROT) .* Fuo3 .- sin(THETA_ROT) .* Fvo3,
             sin(THETA_ROT) .* Fuo3 .+ cos(THETA_ROT) .* Fvo3
println("ALL: $(length(lato1)) moorings,  IWAP: $(length(lato3)) moorings")


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


entry_station = Int[]; matches = MatchInfo[]
for (dsname, lats, lons) in ((:IWAP, lato3, lono3), (:ALL, lato1, lono1))
    for i in eachindex(lats)
        p = nearest_station(lats[i], lons[i])
        if p === nothing
            println("  dsname#i (lat=(lats[i]),lon=(lons[i])): no model station within tolerance -> skipped")
            continue
        end
        push!(entry_station, p); push!(matches, MatchInfo(dsname, i))
    end
end
lat_m = lat_m[entry_station];  lon_m = lon_m[entry_station]
Fu1_m = Fu1_m[entry_station];  Fv1_m = Fv1_m[entry_station]
Fu2_m = Fu2_m[entry_station];  Fv2_m = Fv2_m[entry_station]
N_model = length(matches)


p_iwap = [p for p in 1:N_model if matches[p].dataset == :IWAP]
p_all  = [p for p in 1:N_model if matches[p].dataset == :ALL]
println("Moorings matched: IWAP $(length(p_iwap)) / $(length(lato3)),  ALL $(length(p_all)) / $(length(lato1))")


function label_of(p)
    m = matches[p]
    if m.dataset == :IWAP
        return string("IWAP", m.idx)
    else
        return string("M", m.idx)
    end
end




model_uv(p, mode) = mode == 1 ? (Fu1_m[p], Fv1_m[p]) : (Fu2_m[p], Fv2_m[p])
function obs_uv(p, mode)
    m = matches[p]
    m.dataset == :IWAP && return Fuo3[m.idx, mode], Fvo3[m.idx, mode]
    return Fuo1[m.idx, mode], Fvo1[m.idx, mode]
end


lon_m_geo = mod.(lon_m .+ 180, 360) .- 180     # -180..180


# ----------------------------------------------------------------------------
# 3) REGIONS (same k-means / manual logic as before)
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


tight_box(g) = (minimum(lon_m_geo[g]), maximum(lon_m_geo[g]), minimum(lat_m[g]), maximum(lat_m[g]))
inbox(x, y, b) = b[1] <= x <= b[2] && b[3] <= y <= b[4]


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
            changed = true
            break
        end
        changed || break
    end
    return groups
end


regions = Tuple{String, Vector{Int}}[]          # (title, mooring entries), ALL only
if REGION_MODE == :auto
    a = kmeans_sphere(lat_m[p_all], lon_m_geo[p_all], N_REGIONS)
    groups = [p_all[a .== l] for l in 1:maximum(a)]
    filter!(!isempty, groups); untangle_regions!(groups); filter!(!isempty, groups)
    sort!(groups, by = g -> mean(lon_m_geo[g]))                  # west -> east
    for g in groups
        push!(regions, ("$(fmt_lat(mean(lat_m[g]))), $(fmt_lon(mean(lon_m_geo[g])))", g))
    end
else
    for (name, (x0, x1, y0, y1)) in REGION_BOXES
        g = [p for p in p_all if x0 <= lon_m_geo[p] <= x1 && y0 <= lat_m[p] <= y1]
        isempty(g) ? (@warn "$name contains no matched moorings -- dropped") : push!(regions, (name, g))
    end
end
println("\nRegions:")
for (r, (t, g)) in enumerate(regions)
    println("  R$r  $t  -> $(length(g)) moorings: ", join(label_of.(g), ", "))
end


# ----------------------------------------------------------------------------
# 4) PLOTTING HELPERS
# ----------------------------------------------------------------------------
function region_limits(ps; minpad = REG_PAD_DEG)
    x = lon_m_geo[ps]; y = lat_m[ps]
    px = max(0.2 * (maximum(x) - minimum(x)), minpad)
    py = max(0.2 * (maximum(y) - minimum(y)), minpad)
    return (minimum(x) - px, maximum(x) + px, max(minimum(y) - py, -89.0), min(maximum(y) + py, 89.0))
end


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


"Arrows with a white halo so they stay visible on any background colour."
function draw_arrows!(ax, x, y, u, v, scale, color; lw = 2.4)
    xs, ys = arrow_xy(x, y, u, v, scale)
    isempty(xs) && return
    lines!(ax, xs, ys; color = :white, linewidth = lw + 2.5)
    lines!(ax, xs, ys; color = color, linewidth = lw)
end


function nice_ref(x)
    x <= 0 && return 1.0
    e = floor(log10(x)); f = x / 10^e
    return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * 10^e
end


draw_box!(ax, b; kw...) = lines!(ax, [b[1], b[2], b[2], b[1], b[1]], [b[3], b[3], b[4], b[4], b[3]]; kw...)


"""
One map panel: IT amplitude background, model + mooring flux vectors,
mooring sites with labels, reference arrow. Each panel has its own arrow scale.
"""
function flux_panel!(pos, g, mode, lims; title = "")
    gs = sort(g, by = p -> (lon_m_geo[p], lat_m[p], matches[p].idx))
    x = lon_m_geo[gs]; y = lat_m[gs]; labs = label_of.(gs)
    muv = model_uv.(gs, mode); ouv = obs_uv.(gs, mode)
    mu = first.(muv); mv = last.(muv); ou = first.(ouv); ov = last.(ouv)


    ax = Axis(pos; title = title, aspect = DataAspect(), limits = lims,
              xlabel = "Longitude (°)", ylabel = "Latitude (°)")
    F, xr, yr = read_it_region(lims...)
    isempty(F) || heatmap!(ax, xr, yr, F; colormap = CMAP_REG, colorrange = (0, IT_MAX_REG),
                           highclip = AMP_COLORS[end], nan_color = LAND_COLOR, rasterize = true)


    allmag = filter(isfinite, vcat(hypot.(mu, mv), hypot.(ou, ov)))
    maxmag = isempty(allmag) ? 1.0 : maximum(allmag)
    span   = min(lims[2] - lims[1], lims[4] - lims[3])
    scale  = ARROW_FRAC * span / maxmag
    draw_arrows!(ax, x, y, ou, ov, scale, COL_OBS)
    draw_arrows!(ax, x, y, mu, mv, scale, COL_MODEL)


    # one marker + label per site (e.g. "M2/M7")
    ukeys = unique(collect(zip(x, y)))
    ux = [k[1] for k in ukeys]; uy = [k[2] for k in ukeys]
    ulabs = [join(labs[(x .== k[1]) .& (y .== k[2])], "/") for k in ukeys]
    scatter!(ax, ux, uy; color = :black, markersize = 7, strokecolor = :white, strokewidth = 1)
    text!(ax, ux, uy; text = ulabs, fontsize = 10, offset = (5, 5), color = :black,
          glowwidth = 3, glowcolor = (:white, 0.85))


    # reference arrow (bottom-left)
    ref = nice_ref(0.5 * maxmag)
    rx = lims[1] + 0.06 * (lims[2] - lims[1]); ry = lims[3] + 0.07 * (lims[4] - lims[3])
    rxs, rys = arrow_xy([rx], [ry], [ref], [0.0], scale)
    lines!(ax, rxs, rys; color = :white, linewidth = 4.9)
    lines!(ax, rxs, rys; color = :black, linewidth = 2.4)
    text!(ax, rx, ry; text = @sprintf("%g kW/m", ref), fontsize = 11, offset = (0, 6),
          glowwidth = 3, glowcolor = (:white, 0.85))
    return ax
end


flux_legend!(pos, mode_txt; iwap = false) = Legend(pos,
    [LineElement(color = COL_MODEL, linewidth = 3), LineElement(color = COL_OBS, linewidth = 3)],
    ["Model (MITgcm) $mode_txt", "Mooring (obs)" * (iwap ? ", rotated 60°" : "")];
    orientation = :horizontal, framevisible = false, tellwidth = false)


it_colorbar!(pos) = Colorbar(pos; colormap = CMAP_REG, limits = (0, IT_MAX_REG),
                             highclip = AMP_COLORS[end], label = "M2 internal-tide SSH amplitude (cm)")


# panel limits (also drawn as boxes on the global map)
region_lims = [fit_aspect(region_limits(g), PANEL_ASPECT) for (_, g) in regions]
iwap_lims   = isempty(p_iwap) ? nothing : region_limits(p_iwap)


# ----------------------------------------------------------------------------
# 5) FIGURE 1: global internal-tide map + moorings
# ----------------------------------------------------------------------------
println("\nReading global IT field (block average (FGLOBAL)x(F_GLOBAL)) ...")
Fg, xg, yg = read_it_global(F_GLOBAL)


fig = Figure(size = (1800, 950), fontsize = 18)
ax = Axis(fig[1, 1]; title = "M2 internal-tide SSH amplitude and mooring locations",
          xlabel = "Longitude (°)", ylabel = "Latitude (°)", aspect = DataAspect(),
          limits = (-180, 180, max(-80, yg[1]), yg[end]),
          xticks = -180:60:180, yticks = -60:30:60)
hm = heatmap!(ax, xg, yg, Fg; colormap = CMAP_GLOBAL, colorrange = (0, IT_MAX_GLOBAL),
              highclip = :white, nan_color = LAND_COLOR, rasterize = true)
for (r, b) in enumerate(region_lims)
    draw_box!(ax, b; color = :cyan, linewidth = 1.5)
    text!(ax, b[1], b[4]; text = "R$r", color = :cyan, font = :bold, fontsize = 15,
          align = (:left, :bottom), offset = (2, 2))
end
if iwap_lims !== nothing
    draw_box!(ax, iwap_lims; color = :lime, linewidth = 1.5)
    text!(ax, iwap_lims[1], iwap_lims[4]; text = "IWAP", color = :lime, font = :bold, fontsize = 15,
          align = (:left, :bottom), offset = (2, 2))
end
sc_all = scatter!(ax, lon_m_geo[p_all], lat_m[p_all]; color = :white, marker = :circle,
                  markersize = 9, strokecolor = :black, strokewidth = 1)
sc_iw  = scatter!(ax, lon_m_geo[p_iwap], lat_m[p_iwap]; color = :lime, marker = :star5,
                  markersize = 14, strokecolor = :black, strokewidth = 0.8)
Colorbar(fig[1, 2], hm; label = "Amplitude (cm)")
Legend(fig[2, 1], [sc_all, sc_iw],
       ["ALL moorings [(length(pall))]","IWAPmoorings[(length(p_iwap))]"];
       orientation = :horizontal, framevisible = false, tellwidth = false)
resize_to_layout!(fig)
display(fig)
f1 = joinpath(FIGDIR, "IT_global_moorings_$(TAG).png")
save(f1, fig; px_per_unit = 2)
println("Saved: $f1")
Fg = nothing; GC.gc()


# ----------------------------------------------------------------------------
# 6) FIGURE 2: all ALL regions in one figure (one per mode)
# ----------------------------------------------------------------------------
nreg = length(regions)
ncol = min(3, nreg); nrow = cld(nreg, ncol)
for mode in MODES
    fig = Figure(size = (620ncol + 160, 560nrow + 160), fontsize = 16)
    Label(fig[0, 1:ncol], "Mode-$mode semidiurnal energy flux on M2 internal-tide SSH amplitude",
          fontsize = 22, font = :bold)
    for (r, (title, g)) in enumerate(regions)
        i, j = fldmod1(r, ncol)
        flux_panel!(fig[i, j], g, mode, region_lims[r];
                    title = "((letters[$r]))Rr: title [(length($g))]",r,g)
    end
    it_colorbar!(fig[1:nrow, ncol+1])
    flux_legend!(fig[nrow+1, 1:ncol], "mode $mode")
    colgap!(fig.layout, 20); rowgap!(fig.layout, 12)
    resize_to_layout!(fig)
    out = joinpath(FIGDIR, "IT_flux_regions_mode$(mode)_$(TAG).png")
    save(out, fig; px_per_unit = 2)
    println("Saved: $out")
    display(fig)
end


# ----------------------------------------------------------------------------
# 7) FIGURE 3: IWAP, one panel per mode
# ----------------------------------------------------------------------------
if iwap_lims !== nothing
    nm = length(MODES)
    fig = Figure(size = (700nm + 160, 900), fontsize = 16)
    Label(fig[0, 1:nm], "IWAP: semidiurnal energy flux on M2 internal-tide SSH amplitude",
          fontsize = 22, font = :bold)
    for (k, mode) in enumerate(MODES)
        flux_panel!(fig[1, k], p_iwap, mode, iwap_lims;
                    title = "($(letters[k])) Mode mode [(length(p_iwap)) moorings]")
    end
    it_colorbar!(fig[1, nm+1])
    flux_legend!(fig[2, 1:nm], "modes as titled"; iwap = true)
    colgap!(fig.layout, 20)
    resize_to_layout!(fig)
    out = joinpath(FIGDIR, "IT_flux_IWAP_$(TAG).png")
    save(out, fig; px_per_unit = 2)
    println("Saved: $out")
else
    @warn "No IWAP moorings matched -- IWAP figure skipped."
end


close(ds_it)
println("\nDone.")




