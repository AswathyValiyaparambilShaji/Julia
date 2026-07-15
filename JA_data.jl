using MAT
using CairoMakie
using GeoMakie
using GeoMakie.GeoJSON
using Statistics


# ════════════════════════════════════════════════════════════════════════
# 1) Load lat/lon (+ flux) from the three mooring files
# ════════════════════════════════════════════════════════════════════════
file1path = "/home/aswathy/mnt/data/aswathy/Mooring_Data/Flux_mooring_timeseries_ALL.mat"
file2path = "/home/aswathy/mnt/data/aswathy/Mooring_Data/Flux_mooring_timeseries_ALL_OLD.mat"
file3path = "/home/aswathy/mnt/data/aswathy/Mooring_Data/Flux_mooring_timeseries_ALL_IWAP.mat"


f1 = matopen(file1path)
lato1 = vec(read(f1, "lato")); lono1 = vec(read(f1, "lono"))
Fuo1  = read(f1, "Fuo");       Fvo1  = read(f1, "Fvo")
close(f1)


f2 = matopen(file2path)
lato2 = vec(read(f2, "lato")); lono2 = vec(read(f2, "lono"))
Fuo2  = read(f2, "Fuo");       Fvo2  = read(f2, "Fvo")
close(f2)


f3 = matopen(file3path)
lato3 = vec(read(f3, "lato")); lono3 = vec(read(f3, "lono"))
Fuo3  = read(f3, "Fuo");       Fvo3  = read(f3, "Fvo")
close(f3)


# ════════════════════════════════════════════════════════════════════════
# 2) Print lat/lon pairs for each file
# ════════════════════════════════════════════════════════════════════════
println("=== File 1 (ALL): $(length(lato1)) moorings ===")
for i in eachindex(lato1)
    println("  Mooring $i: (lat=$(lato1[i]), lon=$(lono1[i]))")
end


println("\n=== File 2 (ALL_OLD): $(length(lato2)) moorings ===")
for i in eachindex(lato2)
    println("  Mooring $i: (lat=$(lato2[i]), lon=$(lono2[i]))")
end


println("\n=== File 3 (ALL_IWAP): $(length(lato3)) moorings ===")
for i in eachindex(lato3)
    println("  Mooring $i: (lat=$(lato3[i]), lon=$(lono3[i]))")
end


# ════════════════════════════════════════════════════════════════════════
# 2b) Quick overlap check: are the "same" moorings shared across files?
#     Two moorings are called a "match" if lat & lon agree within `tol` deg.
# ════════════════════════════════════════════════════════════════════════
function find_matches(lat_a, lon_a, lat_b, lon_b; tol = 0.05)
    matches = Tuple{Int,Int}[]
    for i in eachindex(lat_a), j in eachindex(lat_b)
        if isapprox(lat_a[i], lat_b[j]; atol=tol) && isapprox(lon_a[i], lon_b[j]; atol=tol)
            push!(matches, (i, j))
        end
    end
    return matches
end


m12 = find_matches(lato1, lono1, lato2, lono2)
m13 = find_matches(lato1, lono1, lato3, lono3)
m23 = find_matches(lato2, lono2, lato3, lono3)


println("\n=== Location overlap (tolerance = 0.05°) ===")
println("File1 <-> File2 : $(length(m12)) matching moorings out of $(length(lato1))/$(length(lato2))")
println("File1 <-> File3 : $(length(m13)) matching moorings out of $(length(lato1))/$(length(lato3))")
println("File2 <-> File3 : $(length(m23)) matching moorings out of $(length(lato2))/$(length(lato3))")


# ════════════════════════════════════════════════════════════════════════
# 3) Separate mooring-location maps, one figure per file
# ════════════════════════════════════════════════════════════════════════
function add_land!(ax)
    land = GeoMakie.land()
    poly!(ax, land, color = :lightgray, strokecolor = :gray40, strokewidth = 0.5)
end


# Manually draw a closed rectangle around the axis limits. This guarantees a
# full border on all four sides (top/bottom/left/right) regardless of how
# the underlying Axis/GeoAxis spine attributes render.
function add_box!(ax, xlims, ylims; color = :black, linewidth = 1.5)
    x0, x1 = xlims
    y0, y1 = ylims
    lines!(ax, [x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
        color = color, linewidth = linewidth)
end


function plot_locations(lono, lato, title_str, outname; color = :steelblue)
    nmoor = length(lato)
    xlims = (-180, 180)
    ylims = (-80, 70)
    fig = Figure(size = (900, 500))
    ax = GeoAxis(fig[1,1],
        dest   = "+proj=eqc",
        xlabel = "Longitude (°E)",
        ylabel = "Latitude (°N)",
        title  = "$title_str ($nmoor moorings)",
        limits = (xlims[1], xlims[2], ylims[1], ylims[2]),
        xgridvisible = false,
        ygridvisible = false)


    add_land!(ax)


    scatter!(ax, lono, lato,
        color       = color,
        markersize  = 8,
        strokecolor = :black,
        strokewidth = 0.8)


    add_box!(ax, xlims, ylims)


    save(outname, fig)
    display(fig)
    println("Saved: $outname")
    return fig
end


println("\n--- Building individual location figures ---")
fig_loc1 = plot_locations(lono1, lato1, "File 1 (ALL): Mooring Locations",
                           "figure_file1_locations.png", color = :steelblue)
fig_loc2 = plot_locations(lono2, lato2, "File 2 (ALL_OLD): Mooring Locations",
                           "figure_file2_locations.png", color = :seagreen)
fig_loc3 = plot_locations(lono3, lato3, "File 3 (ALL_IWAP): Mooring Locations",
                           "figure_file3_locations.png", color = :firebrick)


# ════════════════════════════════════════════════════════════════════════
# 3b) Combined figure: all three files' moorings overlaid on one map
# ════════════════════════════════════════════════════════════════════════
println("\n--- Building combined location figure ---")
xlims_c = (-180, 180)
ylims_c = (-80, 70)


fig_all = Figure(size = (1000, 550))
ax_all = GeoAxis(fig_all[1,1],
    dest   = "+proj=eqc",
    xlabel = "Longitude (°E)",
    ylabel = "Latitude (°N)",
    title  = "All Mooring Locations (File 1, File 2, File 3)",
    limits = (xlims_c[1], xlims_c[2], ylims_c[1], ylims_c[2]),
    xgridvisible = false,
    ygridvisible = false)


add_land!(ax_all)


sc1 = scatter!(ax_all, lono1, lato1, color = :steelblue, markersize = 10,
    strokecolor = :black, strokewidth = 0.8, marker = :circle)
sc2 = scatter!(ax_all, lono2, lato2, color = :seagreen, markersize = 10,
    strokecolor = :black, strokewidth = 0.8, marker = :utriangle)
sc3 = scatter!(ax_all, lono3, lato3, color = :firebrick, markersize = 12,
    strokecolor = :black, strokewidth = 0.8, marker = :star5)


add_box!(ax_all, xlims_c, ylims_c)


Legend(fig_all[1,2],
    [sc1, sc2, sc3],
    ["File 1 (ALL, n=$(length(lato1)))",
     "File 2 (ALL_OLD, n=$(length(lato2)))",
     "File 3 (ALL_IWAP, n=$(length(lato3)))"])


save("figure_all_locations_combined.png", fig_all)
display(fig_all)
println("Saved: figure_all_locations_combined.png")


# ════════════════════════════════════════════════════════════════════════
# 4) Time series of flux (Mode 1 & Mode 2) for one mooring, as 2 subplots
#    (u in top panel, v in bottom panel). No date/time vector is stored in
#    the .mat file, so the x-axis is simply the sample index.
#    Change `mooring_idx` below to inspect a different mooring (1..79).
# ════════════════════════════════════════════════════════════════════════
mooring_idx = 1
ntime = size(Fuo1, 3)
t = 1:ntime


u_mode1 = Fuo1[mooring_idx, 1, :]
u_mode2 = Fuo1[mooring_idx, 2, :]
v_mode1 = Fvo1[mooring_idx, 1, :]
v_mode2 = Fvo1[mooring_idx, 2, :]


fig_ts = Figure(size = (1100, 700))


ax_u = Axis(fig_ts[1,1],
    xlabel = "Time index",
    ylabel = "Fu (kW/m)",
    title  = "Zonal Energy Flux (Fu) — Mooring $mooring_idx",
    xgridvisible      = false,
    ygridvisible      = false,
    topspinevisible   = true,
    rightspinevisible = true,
    bottomspinevisible = true,
    leftspinevisible  = true,
    spinewidth        = 1.5)


lines!(ax_u, t, u_mode1, color = :steelblue, linewidth = 1.2, label = "Mode 1")
lines!(ax_u, t, u_mode2, color = :firebrick, linewidth = 1.2, label = "Mode 2")
axislegend(ax_u, position = :rt)


ax_v = Axis(fig_ts[2,1],
    xlabel = "Time index",
    ylabel = "Fv (kW/m)",
    title  = "Meridional Energy Flux (Fv) — Mooring $mooring_idx",
    xgridvisible      = false,
    ygridvisible      = false,
    topspinevisible   = true,
    rightspinevisible = true,
    bottomspinevisible = true,
    leftspinevisible  = true,
    spinewidth        = 1.5)


lines!(ax_v, t, v_mode1, color = :steelblue, linewidth = 1.2, label = "Mode 1")
lines!(ax_v, t, v_mode2, color = :firebrick, linewidth = 1.2, label = "Mode 2")
axislegend(ax_v, position = :rt)


linkxaxes!(ax_u, ax_v)


save("figure_flux_timeseries_mooring$(mooring_idx).png", fig_ts)
display(fig_ts)


println("\nAll figures saved successfully.")




