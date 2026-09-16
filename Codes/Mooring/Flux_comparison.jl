# ============================================================================
# Compare MITgcm modal baroclinic fluxes (88 mooring-colocated stations,
# saved on the HPC to Mooring_modal_fluxes.nc) against observed mooring
# fluxes from three .mat datasets (ALL, ALL_OLD, IWAP), for mode 1 and
# mode 2, by magnitude and by direction.
#
# MATCHING: each of the model's stations is matched by lat/lon against all
# three .mat mooring datasets, tried in priority order IWAP > ALL > ALL_OLD:
#   1) IWAP first, because it's the dataset that triggers the 60 deg
#      rotation below.
#   2) ALL next. ALL is treated as the up-to-date version of ALL_OLD (same
#      naming, "OLD" suffix implies superseded), so if a model station is
#      within tolerance of BOTH ALL and ALL_OLD at the same location, ALL's
#      flux is what gets plotted -- ALL_OLD is not used for that station,
#      and no averaging/blending of the two happens.
#   3) ALL_OLD is only used as a fallback, for a station that has no IWAP
#      or ALL mooring nearby but does have an ALL_OLD one.
# If you'd rather prefer ALL_OLD over ALL for overlapping locations, or see
# both plotted side by side, tell me and I'll change the priority order --
# right now it's IWAP > ALL > ALL_OLD, in that order, with no blending.
# Every ambiguous case (a station within tolerance of more than one
# dataset) is also printed at runtime so you can see exactly which stations
# it affected and which dataset won.
#
# LABELING: a station matched to IWAP is labeled "IWAP<n>", to ALL is
# labeled "MP<n>", and to ALL_OLD (fallback only) is labeled "OLD<n>" --
# <n> is that mooring's own index in its source file (not the model
# station number), so the same name identifies the same physical mooring
# on both the bar charts and the position map.
#
# ROTATION: per Zhao et al. (2010, p.3) / your existing 4-point script, a
# 60 deg rotation is needed to align the MITgcm and IWAP coordinate frames.
# As you specified: the MODEL flux (not the observed/mooring flux) is what
# gets rotated here, and ONLY for stations matched to an IWAP mooring.
# Stations matched to ALL or ALL_OLD are compared unrotated on both sides.
# Observed (mooring) fluxes are NEVER rotated, in any of the three datasets.
# ============================================================================


using NCDatasets, MAT, Statistics, CairoMakie, GeoMakie, GeoMakie.GeoJSON, Printf


# ----------------------------------------------------------------------------
# 0) PATHS -- EDIT THESE FOR YOUR SERVER
# ----------------------------------------------------------------------------
# This is the file produced by the HPC script you pasted (Mooring_modal_fluxes.nc,
# station-only: lat, lon, Fu_mode1, Fv_mode1, Fu_mode2, Fv_mode2). On the HPC it
# lived at /nobackup/avaliyap/V2/Moorings/Mooring_modal_fluxes.nc -- it will be
# somewhere else on your server, so point this at wherever you copied it to:
model_flux_ncfile = "/home/aswathy/mnt/data/aswathy/MITgcm_NAS/Moorings/Mooring_modal_fluxes.nc"   # <-- UPDATE THIS


file1path = "/home/aswathy/mnt/data/aswathy/Mooring_Data/Flux_mooring_timeseries_ALL.mat"
file2path = "/home/aswathy/mnt/data/aswathy/Mooring_Data/Flux_mooring_timeseries_ALL_OLD.mat"
file3path = "/home/aswathy/mnt/data/aswathy/Mooring_Data/Flux_mooring_timeseries_ALL_IWAP.mat"


FIGDIR = "/home/aswathy/mnt/data/aswathy/Mooring_Data/"   # <-- point this at wherever you want the PNGs saved
mkpath(FIGDIR)


MATCH_TOL_DEG = 0.05     # lat/lon matching tolerance (degrees), nearest-neighbor within this radius
THETA_ROT     = pi / 3   # 60 deg, IWAP-only model rotation (Zhao et al. 2010)


# ----------------------------------------------------------------------------
# 1) LOAD MODEL FLUXES (N stations, modes 1 & 2, already depth-integrated
#    and time-averaged -- kW/m)
# ----------------------------------------------------------------------------
ds = NCDataset(model_flux_ncfile, "r")
lon_m = Array(ds["lon"])
lat_m = Array(ds["lat"])
Fu1_m = Array(ds["Fu_mode1"])
Fv1_m = Array(ds["Fv_mode1"])
Fu2_m = Array(ds["Fu_mode2"])
Fv2_m = Array(ds["Fv_mode2"])
close(ds)
N_model = length(lat_m)
println("Loaded $N_model model stations from $model_flux_ncfile")


# ----------------------------------------------------------------------------
# 2) LOAD THE THREE OBSERVED MOORING DATASETS
# ----------------------------------------------------------------------------
function load_mooring_mat(path)
    f = matopen(path)
    lato = vec(read(f, "lato"))
    lono = vec(read(f, "lono"))
    Fuo  = read(f, "Fuo")
    Fvo  = read(f, "Fvo")
    close(f)
    # Some of these files store (mooring, mode) already time-averaged;
    # others store (mooring, mode, time) and need averaging over time.
    # Handle either shape automatically -- and average NaN-aware, since
    # these raw time series have data gaps: a plain `mean(dims=3)` would
    # turn ANY single missing timestep into a NaN for that whole mooring.
    if ndims(Fuo) == 3
        n1, n2, _ = size(Fuo)
        Fuo2d = fill(NaN, n1, n2)
        Fvo2d = fill(NaN, n1, n2)
        for i in 1:n1, j in 1:n2
            vu = filter(!isnan, @view Fuo[i, j, :])
            vv = filter(!isnan, @view Fvo[i, j, :])
            Fuo2d[i, j] = isempty(vu) ? NaN : mean(vu)
            Fvo2d[i, j] = isempty(vv) ? NaN : mean(vv)
        end
        Fuo, Fvo = Fuo2d, Fvo2d
    end
    return lato, lono, Fuo, Fvo   # Fuo/Fvo now (n_mooring, n_mode)
end


lato1, lono1, Fuo1, Fvo1 = load_mooring_mat(file1path)   # ALL
lato2, lono2, Fuo2, Fvo2 = load_mooring_mat(file2path)   # ALL_OLD
lato3, lono3, Fuo3, Fvo3 = load_mooring_mat(file3path)   # IWAP
println("ALL:     $(length(lato1)) moorings")
println("ALL_OLD: $(length(lato2)) moorings")
println("IWAP:    $(length(lato3)) moorings")


# ----------------------------------------------------------------------------
# 3) MATCH EACH MODEL STATION TO A MOORING, PRIORITY IWAP > ALL > ALL_OLD
# ----------------------------------------------------------------------------
function nearest_match(lat0, lon0, lats, lons; tol = MATCH_TOL_DEG)
    d2 = (lats .- lat0) .^ 2 .+ (lons .- lon0) .^ 2
    j = argmin(d2)
    return sqrt(d2[j]) <= tol ? j : nothing
end


struct MatchInfo
    dataset::Symbol   # :IWAP, :ALL, :ALL_OLD, or :none
    idx::Int          # index into that dataset's mooring list (0 if :none)
end


matches = Vector{MatchInfo}(undef, N_model)
n_ambiguous = 0
for p in 1:N_model
    j_iwap   = nearest_match(lat_m[p], lon_m[p], lato3, lono3)
    j_all    = nearest_match(lat_m[p], lon_m[p], lato1, lono1)
    j_allold = nearest_match(lat_m[p], lon_m[p], lato2, lono2)


    n_hits = count(!isnothing, (j_iwap, j_all, j_allold))
    if n_hits > 1
        n_ambiguous += 1
        winner = j_iwap !== nothing ? :IWAP : :ALL
        @warn "Model station $p (lat=$(lat_m[p]), lon=$(lon_m[p])) matched more than one mooring dataset within $(MATCH_TOL_DEG)°; using $winner (priority order IWAP > ALL > ALL_OLD)."
    end


    if j_iwap !== nothing
        matches[p] = MatchInfo(:IWAP, j_iwap)
    elseif j_all !== nothing
        matches[p] = MatchInfo(:ALL, j_all)
    elseif j_allold !== nothing
        matches[p] = MatchInfo(:ALL_OLD, j_allold)
    else
        matches[p] = MatchInfo(:none, 0)
    end
end


n_iwap   = count(m -> m.dataset == :IWAP, matches)
n_all    = count(m -> m.dataset == :ALL, matches)
n_allold = count(m -> m.dataset == :ALL_OLD, matches)
n_none   = count(m -> m.dataset == :none, matches)
println("\nMatch summary, counted from the MODEL side ($N_model model stations, tol = $(MATCH_TOL_DEG)°):")
println("  matched to IWAP:    $n_iwap")
println("  matched to ALL:     $n_all")
println("  matched to ALL_OLD: $n_allold  (fallback only -- used when a station had no IWAP or ALL match)")
println("  unmatched:          $n_none  (these model stations are NOT plotted anywhere below)")
println("  ambiguous (matched more than one dataset): $n_ambiguous")
println("  => total plotted in the bar charts: $(n_iwap + n_all + n_allold) model stations")


# Same numbers, counted from the MOORING side -- different denominator,
# often confused with the model-side count above. This is "how many of
# each source file's own moorings got used by some model station."
println("\nSame matches, counted from the MOORING side:")
println("  ALL file:     $n_all of $(length(lato1)) moorings were matched to a model station ($(length(lato1) - n_all) were not)")
println("  ALL_OLD file: $n_allold of $(length(lato2)) moorings were matched to a model station ($(length(lato2) - n_allold) were not; note ALL_OLD is only ever used as a fallback, so this count also excludes any ALL_OLD mooring whose location was already claimed by ALL)")
println("  IWAP file:    $n_iwap of $(length(lato3)) moorings were matched to a model station ($(length(lato3) - n_iwap) were not)")


# ---- DIAGNOSTIC: name and locate every unmatched station explicitly ----
if n_none > 0
    println("\nUnmatched station(s) (no mooring within $(MATCH_TOL_DEG)° in any of the 3 files):")
    for p in 1:N_model
        if matches[p].dataset == :none
            println("  station $p: lat=$(lat_m[p]), lon=$(lon_m[p])")
        end
    end
end


# ----------------------------------------------------------------------------
# 4) ROTATE MODEL FLUXES 60° -- ONLY FOR STATIONS MATCHED TO IWAP
#    (observed/mooring fluxes are never rotated, in any dataset)
# ----------------------------------------------------------------------------
function rotate_flux(Fu, Fv, theta)
    R = [cos(theta) -sin(theta); sin(theta) cos(theta)]
    v = R * [Fu; Fv]
    return v[1], v[2]
end


Fu1_m_cmp = copy(Fu1_m); Fv1_m_cmp = copy(Fv1_m)   # "cmp" = as used for comparison
Fu2_m_cmp = copy(Fu2_m); Fv2_m_cmp = copy(Fv2_m)
for p in 1:N_model
    if matches[p].dataset == :IWAP
        Fu1_m_cmp[p], Fv1_m_cmp[p] = rotate_flux(Fu1_m[p], Fv1_m[p], THETA_ROT)
        Fu2_m_cmp[p], Fv2_m_cmp[p] = rotate_flux(Fu2_m[p], Fv2_m[p], THETA_ROT)
    end
end


# ----------------------------------------------------------------------------
# 5) BUILD MATCHED MODEL/OBS PAIRS + LABELS, FOR MODE 1 AND MODE 2
# ----------------------------------------------------------------------------
matched_p = [p for p in 1:N_model if matches[p].dataset != :none]
n_matched = length(matched_p)
println("\n$n_matched / $N_model model stations matched to a mooring (IWAP, ALL, or ALL_OLD).")


# ---- DIAGNOSTIC: flag matched stations whose MODEL flux is NaN --------
# These stay in matched_p (matching is purely lat/lon-based), but a NaN
# bar height means CairoMakie silently fails to draw that bar -- so this
# is the other common way a station "disappears" from the plot even
# though it's still present in matched_p and gets an x-tick label.
nan_stations = [p for p in matched_p if isnan(Fu1_m[p]) || isnan(Fv1_m[p]) ||
                                          isnan(Fu2_m[p]) || isnan(Fv2_m[p])]
if !isempty(nan_stations)
    println("\nMatched station(s) with NaN model flux (bar will not render for these):")
    for p in nan_stations
        println("  station $p: lat=$(lat_m[p]), lon=$(lon_m[p]), matched dataset=$(matches[p].dataset)")
    end
end


function obs_flux(m::MatchInfo, mode::Int)
    if m.dataset == :IWAP
        return Fuo3[m.idx, mode], Fvo3[m.idx, mode]
    elseif m.dataset == :ALL
        return Fuo1[m.idx, mode], Fvo1[m.idx, mode]
    elseif m.dataset == :ALL_OLD
        return Fuo2[m.idx, mode], Fvo2[m.idx, mode]
    else
        return NaN, NaN
    end
end


# Label = "IWAP<idx>" / "MP<idx>" / "OLD<idx>" depending on which dataset
# this station matched, where <idx> is that mooring's own index within its
# source file (not the model station number p) -- this way the same name
# identifies the same physical mooring on both the bar chart and the map.
function station_label(m::MatchInfo)
    m.dataset == :IWAP    && return "IWAP$(m.idx)"
    m.dataset == :ALL     && return "MP$(m.idx)"
    m.dataset == :ALL_OLD && return "OLD$(m.idx)"
    return "?"
end
labels = [station_label(matches[p]) for p in matched_p]


# ----------------------------------------------------------------------------
# 6) POSITION MAP -- matched stations, colored by source dataset, labeled
#    with the exact same names used on the bar charts below. Same land-
#    shading approach (GeoMakie land polygons on a GeoAxis) as your
#    mooring-location-overlap script, added here with minimal changes to
#    everything else in this file.
# ----------------------------------------------------------------------------
function add_land!(ax)
    land = GeoMakie.land()
    poly!(ax, land, color = :lightgray, strokecolor = :gray40, strokewidth = 0.5)
end


function add_box!(ax, xlims, ylims; color = :black, linewidth = 1.5)
    x0, x1 = xlims
    y0, y1 = ylims
    lines!(ax, [x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color = color, linewidth = linewidth)
end


# GeoMakie's land polygons (and your other script's map) use -180..180
# longitude; lon_m from the model .nc file is 0..360 (MITgcm convention).
# This wrapped copy is used ONLY for this map's x-coordinates -- lon_m
# itself, and everything else in the script (matching, rotation, flux
# values), is untouched.
lon_m_geo = mod.(lon_m .+ 180, 360) .- 180


lon_pad = 0.1 * (maximum(lon_m_geo[matched_p]) - minimum(lon_m_geo[matched_p]) + 1e-6)
lat_pad = 0.1 * (maximum(lat_m[matched_p]) - minimum(lat_m[matched_p]) + 1e-6)
xlims_map = (minimum(lon_m_geo[matched_p]) - lon_pad, maximum(lon_m_geo[matched_p]) + lon_pad)
ylims_map = (minimum(lat_m[matched_p]) - lat_pad, maximum(lat_m[matched_p]) + lat_pad)


fig_map = Figure(resolution = (1000, 800))
ax_map = GeoAxis(fig_map[1, 1],
    dest   = "+proj=eqc",
    xlabel = "Longitude [°]", ylabel = "Latitude [°]",
    title  = "Matched mooring locations ($n_matched of $N_model model stations)",
    limits = (xlims_map[1], xlims_map[2], ylims_map[1], ylims_map[2]))


add_land!(ax_map)


is_iwap   = [matches[p].dataset == :IWAP for p in matched_p]
is_all    = [matches[p].dataset == :ALL for p in matched_p]
is_allold = [matches[p].dataset == :ALL_OLD for p in matched_p]


scatter!(ax_map, lon_m_geo[matched_p][is_iwap], lat_m[matched_p][is_iwap];
    color = :firebrick, markersize = 12, strokecolor = :black, strokewidth = 0.5, label = "IWAP")
scatter!(ax_map, lon_m_geo[matched_p][is_all], lat_m[matched_p][is_all];
    color = :steelblue, markersize = 12, strokecolor = :black, strokewidth = 0.5, label = "ALL")
scatter!(ax_map, lon_m_geo[matched_p][is_allold], lat_m[matched_p][is_allold];
    color = :seagreen, markersize = 12, marker = :utriangle, strokecolor = :black, strokewidth = 0.5, label = "ALL_OLD (fallback)")


for (k, p) in enumerate(matched_p)
    text!(ax_map, lon_m_geo[p] + 0.02, lat_m[p] + 0.02; text = labels[k], fontsize = 9, color = :black)
end


#add_box!(ax_map, xlims_map, ylims_map)


axislegend(ax_map, position = :rb)
display(fig_map)
map_png = joinpath(FIGDIR, "Matched_mooring_locations_map.png")
save(map_png, fig_map)
println("Saved: $map_png")


# ----------------------------------------------------------------------------
# 7) BAR CHARTS -- MODE 1 AND MODE 2
# ----------------------------------------------------------------------------
for mode in 1:2
    Fu_model = mode == 1 ? Fu1_m_cmp : Fu2_m_cmp
    Fv_model = mode == 1 ? Fv1_m_cmp : Fv2_m_cmp


    model_u = [Fu_model[p] for p in matched_p]
    model_v = [Fv_model[p] for p in matched_p]
    obs_uv  = [obs_flux(matches[p], mode) for p in matched_p]
    obs_u   = [x[1] for x in obs_uv]
    obs_v   = [x[2] for x in obs_uv]


    mag_model = sqrt.(model_u .^ 2 .+ model_v .^ 2)
    mag_obs   = sqrt.(obs_u .^ 2 .+ obs_v .^ 2)
    dir_model = atand.(model_v, model_u)      # bearing, degrees, -180..180
    dir_obs   = atand.(obs_v, obs_u)


    # ---- DIAGNOSTIC: any matched station still NaN on the obs side? ----
    # (e.g. a mooring with zero valid timesteps in its raw time series --
    # real, not a bug -- vs. everything else, which is now NaN-aware.)
    obs_nan_p = [matched_p[k] for k in 1:n_matched if isnan(mag_obs[k])]
    if !isempty(obs_nan_p)
        println("\nMode $mode: matched station(s) with NaN OBSERVED flux (no valid timesteps):")
        for p in obs_nan_p
            println("  station $p ($(station_label(matches[p]))): dataset=$(matches[p].dataset)")
        end
    end


    # ---- quick text summary -- computed only over stations with BOTH a
    #      valid model and a valid obs value, so one NaN mooring can't
    #      blank out the whole mean the way it did before ----
    valid = .!isnan.(mag_model) .& .!isnan.(mag_obs)
    n_valid = count(valid)
    pct_diff = 100 .* (mag_model[valid] .- mag_obs[valid]) ./ mag_obs[valid]
    ang_diff = mod.(dir_model[valid] .- dir_obs[valid] .+ 180, 360) .- 180
    @printf("\n== Mode %d (%d / %d matched stations have valid model+obs data) ==\n", mode, n_valid, n_matched)
    @printf("  mean |F_model| = %.3f kW/m,  mean |F_obs| = %.3f kW/m\n", mean(mag_model[valid]), mean(mag_obs[valid]))
    @printf("  mean %% diff (model vs obs)  = %.1f%%\n", mean(pct_diff))
    @printf("  mean |angle diff|           = %.1f deg\n", mean(abs.(ang_diff)))


    # ---- figure: 2 rows x 1 col, magnitude on top, direction below,
    #      model & mooring as a paired (dodged) bar per station ----
    fig = Figure(resolution = (max(1200, 22 * n_matched), 800))


    ax_mag = Axis(fig[1, 1],
        title = "Mode $mode flux magnitude: model vs mooring " *
                "($n_matched of $N_model model stations matched -- $n_all ALL, $n_allold ALL_OLD fallback, $n_iwap IWAP)",
        ylabel = "|F| (kW/m)", xticklabelrotation = pi / 3)
    xs   = vcat(1:n_matched, 1:n_matched)
    grp  = vcat(fill(1, n_matched), fill(2, n_matched))
    cols = [g == 1 ? :crimson : :steelblue for g in grp]
    barplot!(ax_mag, xs, vcat(mag_model, mag_obs); dodge = grp, color = cols)
    ax_mag.xticks = (1:n_matched, labels)


    ax_dir = Axis(fig[2, 1],
        title = "Mode $mode flux direction: model vs mooring",
        ylabel = "Angle(deg)", xlabel = "mooring station",
        xticklabelrotation = pi / 3)
    barplot!(ax_dir, xs, vcat(dir_model, dir_obs); dodge = grp, color = cols)
    ax_dir.xticks = (1:n_matched, labels)


    elem_model = PolyElement(color = :crimson)
    elem_obs   = PolyElement(color = :steelblue)
    Legend(fig[3, 1], [elem_model, elem_obs], ["Model (MITgcm)", "Mooring (obs)"],
        orientation = :horizontal, tellwidth = false)


    display(fig)
    outpng = joinpath(FIGDIR, "Model_vs_Mooring_mode$(mode)_allmatched.png")
    save(outpng, fig)
    println("Saved: $outpng")
end




