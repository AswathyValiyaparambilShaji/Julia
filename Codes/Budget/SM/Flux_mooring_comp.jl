using MAT
using NCDatasets
using CairoMakie
using Statistics
using TOML


include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin


config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg   = TOML.parsefile(config_file)
base  = cfg["base_path"]
base2 = cfg["base_path2"]


# ════════════════════════════════════════════════════════════════════════
# 1) IWAP mooring flux file
#    Fuo/Fvo are (n_moorings=6, n_modes=5) -- already time-averaged,
#    per-mode baroclinic energy flux in kW/m. Sum across the 5 modes
#    to get the total flux per mooring (same "sum the 5 modes" step
#    the model script does with uflux_modes/vflux_modes).
# ════════════════════════════════════════════════════════════════════════
file3path = "/nobackup/avaliyap/Box56/Mooring_Data/Flux_mooring_timeseries_ALL_IWAP.mat"


f3 = matopen(file3path)
lato3  = vec(read(f3, "lato"))
lono3  = vec(read(f3, "lono"))
depth3 = vec(read(f3, "depth"))
Fuo3   = read(f3, "Fuo")   # (6, 5)
Fvo3   = read(f3, "Fvo")   # (6, 5)
close(f3)


n_moorings = length(lato3)
println("Fuo size: ", size(Fuo3), "   Fvo size: ", size(Fvo3), "   n moorings: ", n_moorings)


if size(Fuo3, 1) != n_moorings
    error("Fuo's first dimension ($(size(Fuo3,1))) doesn't match n_moorings ($n_moorings) -- " *
          "check whether Fuo is actually (mode, mooring) instead of (mooring, mode)")
end


Fu_iwap_all = zeros(n_moorings)
Fv_iwap_all = zeros(n_moorings)
for m in 1:n_moorings
    Fu_iwap_all[m] = sum(skipmissing(replace(vec(Fuo3[m, :]), NaN => missing)))
    Fv_iwap_all[m] = sum(skipmissing(replace(vec(Fvo3[m, :]), NaN => missing)))
end


# ════════════════════════════════════════════════════════════════════════
# 2) Model-computed modal flux (from the mooring_modal_flux.jl output)
# ════════════════════════════════════════════════════════════════════════
model_file = joinpath(base, "SM", "MooringModalFlux", "MooringModalFlux_Box56_IWAP.nc")


lat_model, lon_model, tile_model, uflux_model_kwm, vflux_model_kwm =
    NCDataset(model_file, "r") do ds
        (Array(ds["lat"][:]), Array(ds["lon"][:]), Array(ds["tile"][:]),
         Array(ds["uflux_depth_int"][:]), Array(ds["vflux_depth_int"][:]))   # already kW/m
    end


# ════════════════════════════════════════════════════════════════════════
# 3) The 4 Box-56 moorings, with their IWAP index (1-based, into lato3)
# ════════════════════════════════════════════════════════════════════════
mooring_ids = [82, 83, 84, 85]
target_lats = [25.4891, 27.7690, 28.8995, 30.1312]
target_lons = [194.8451, 196.0301, 196.5105, 197.1154]
iwap_idx    = [1, 2, 3, 4]

#=
target_lats = [24.4891] #27.7690, 28.8995, 30.1312]
target_lons = [193.451]=#
n_points = length(iwap_idx)


# --- sanity check: IWAP lat/lon vs expected mooring locations ---
for p in 1:n_points
    idx = iwap_idx[p]
    dlat = abs(lato3[idx] - target_lats[p])
    dlon = abs(lono3[idx] - target_lons[p])
    if dlat > 0.01 || dlon > 0.01
        @warn "Mooring $(mooring_ids[p]): IWAP file lat/lon (($(lato3[idx])), $(lono3[idx])) " *
              "does not closely match expected (($(target_lats[p])), $(target_lons[p])) -- check iwap_idx mapping"
    end
end

# --- sanity check: model file's stations line up with the same 4 moorings, same order ---
for p in 1:n_points
    dlat = abs(lat_model[p] - target_lats[p])
    dlon = abs(lon_model[p] - target_lons[p])
    if dlat > 0.01 || dlon > 0.01
        @warn "Model file station $p lat/lon (($(lat_model[p])), $(lon_model[p])) " *
              "does not match expected mooring $(mooring_ids[p]) -- check station ordering"
    end
end
#


Fu_iwap  = [Fu_iwap_all[iwap_idx[p]] for p in 1:n_points]
Fv_iwap  = [Fv_iwap_all[iwap_idx[p]] for p in 1:n_points]
Fu_model = Float64.(uflux_model_kwm)
Fv_model = Float64.(vflux_model_kwm)

#
for p in 1:n_points
    println("Mooring $(mooring_ids[p]): IWAP u=$(Fu_iwap[p]) kW/m | Model u=$(Fu_model[p]) kW/m   ||   " *
            "IWAP v=$(Fv_iwap[p]) kW/m | Model v=$(Fv_model[p]) kW/m")
end
#

# ════════════════════════════════════════════════════════════════════════
# 4) Two-panel comparison: u-flux (left), v-flux (right), IWAP vs Model
# ════════════════════════════════════════════════════════════════════════
station_labels = ["Mooring $(id)" for id in mooring_ids]
x = 1:n_points


fig = Figure(size = (1100, 500))


# --- u-flux panel ---
ax_u = Axis(fig[1, 1],
    xlabel = "Station",
    ylabel = "u-flux (kW/m)",
    title  = "Eastward baroclinic energy flux",
    xticks = (x, station_labels),
    xgridvisible = false,
    ygridvisible = false,
)
ax_u.topspinevisible = true
ax_u.rightspinevisible = true


bar_x_u = vcat(x, x)
bar_h_u = vcat(Fu_iwap, Fu_model)
bar_dodge_u = vcat(fill(1, n_points), fill(2, n_points))
barplot!(ax_u, bar_x_u, bar_h_u, dodge = bar_dodge_u,
         color = map(d -> d == 1 ? :steelblue : :darkorange, bar_dodge_u))


# --- v-flux panel ---
ax_v = Axis(fig[1, 2],
    xlabel = "Station",
    ylabel = "v-flux (kW/m)",
    title  = "Northward baroclinic energy flux",
    xticks = (x, station_labels),
    xgridvisible = false,
    ygridvisible = false,
)
ax_v.topspinevisible = true
ax_v.rightspinevisible = true


bar_x_v = vcat(x, x)
bar_h_v = vcat(Fv_iwap, Fv_model)
bar_dodge_v = vcat(fill(1, n_points), fill(2, n_points))
barplot!(ax_v, bar_x_v, bar_h_v, dodge = bar_dodge_v,
         color = map(d -> d == 1 ? :steelblue : :darkorange, bar_dodge_v))


# shared legend
elem_iwap  = PolyElement(color = :steelblue)
elem_model = PolyElement(color = :darkorange)
Legend(fig[2, 1:2], [elem_iwap, elem_model], ["IWAP (observed)", "Model (5-mode reconstruction)"],
       orientation = :horizontal, tellwidth = false)


FIGDIR = cfg["fig_base"]
mkpath(FIGDIR)
save(joinpath(FIGDIR, "iwap_vs_model_flux_comparison.png"), fig)
display(fig)
println("\nSaved comparison figure to iwap_vs_model_flux_comparison.png")




using NCDatasets
ds = NCDataset(model_file, "r")
println(ds)
println("lat: ", ds["lat"][:])
close(ds)




