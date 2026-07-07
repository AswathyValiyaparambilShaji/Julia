using MAT

file1 = matopen("/home/aswathy/mnt/data/aswathy/Mooring_Data/Flux_mooring_timeseries_ALL.mat")
file2 = matopen("/home/aswathy/mnt/data/aswathy/Mooring_Data/Flux_mooring_timeseries_ALL_OLD.mat")
file3 = matopen("/home/aswathy/mnt/data/aswathy/Mooring_Data/Flux_mooring_timeseries_ALL_IWAP.mat")
file4 = matopen("/home/aswathy/mnt/data/aswathy/Mooring_Data/Alford_latlon_fromFluxTimeSeries.mat")
file5 = matopen("/home/aswathy/mnt/data/aswathy/Mooring_Data/Alford_latlon_fromFluxTimeSeries_IWAP.mat")
file6 = matopen("/home/aswathy/mnt/data/aswathy/Mooring_Data/Alford_latlon_fromFluxTimeSeries_OLD.mat")


varnames = keys(file1)
println(varnames)


data = Dict(name => read(file1, name) for name in varnames)
close(file1)


for (k, v) in data
    println(k, " => ", typeof(v), "  size: ", size(v))
end


### File 2 ###

        println(" ###################    File 2 ###################")


varnames = keys(file2)
println(varnames)


data = Dict(name => read(file2, name) for name in varnames)
close(file2)


for (k, v) in data
    println(k, " => ", typeof(v), "  size: ", size(v))
end

### File 3 ###
        println(" ###################    File 3 ###################")

varnames = keys(file3)
println(varnames)


data = Dict(name => read(file3, name) for name in varnames)
close(file3)


for (k, v) in data
    println(k, " => ", typeof(v), "  size: ", size(v))
end

### File 4 ###
        println(" ###################    File 4 ###################")

varnames = keys(file4)
println(varnames)


data = Dict(name => read(file4, name) for name in varnames)
close(file4)


for (k, v) in data
    println(k, " => ", typeof(v), "  size: ", size(v))
end

### File 5 ###
        println(" ###################    File 5 ###################")

varnames = keys(file5)
println(varnames)


data = Dict(name => read(file5, name) for name in varnames)
close(file5)


for (k, v) in data
    println(k, " => ", typeof(v), "  size: ", size(v))
end

### File 6 ###
        println(" ###################    File 6 ###################")

varnames = keys(file6)
println(varnames)


data = Dict(name => read(file6, name) for name in varnames)
close(file6)


for (k, v) in data
    println(k, " => ", typeof(v), "  size: ", size(v))
end





using MAT
using CairoMakie
using GeoMakie
using GeoMakie.GeoJSON
using Statistics


# ── Choose which file to load ──────────────────────────────────────────────
# Options (swap path as needed):
#   Flux_mooring_timeseries_ALL.mat       -> 79 moorings, 2 modes, time series
#   Flux_mooring_timeseries_ALL_OLD.mat   -> 88 moorings, 2 modes, time series
#   Flux_mooring_timeseries_ALL_IWAP.mat  -> 6 moorings, 5 modes, NO time dim
#                                            (already averaged - see note below)


filepath = "/home/aswathy/mnt/data/aswathy/Mooring_Data/Flux_mooring_timeseries_ALL.mat"


file = matopen(filepath)
lato = vec(read(file, "lato"))
lono = vec(read(file, "lono"))
Fuo  = read(file, "Fuo")
Fvo  = read(file, "Fvo")
close(file)


# ── Handle both time-series files (3D) and the IWAP file (2D, no time) ────
if ndims(Fuo) == 3
    nmoor, nmode, ntime = size(Fuo)
    println("Moorings: $nmoor, Modes: $nmode, Time steps: $ntime")
    # Time-mean flux per mooring per mode
    um_kwm = dropdims(mean(Fuo, dims=3), dims=3)   # (nmoor, nmode)
    vm_kwm = dropdims(mean(Fvo, dims=3), dims=3)   # (nmoor, nmode)
else
    nmoor, nmode = size(Fuo)
    println("Moorings: $nmoor, Modes: $nmode (no time dimension - already averaged)")
    um_kwm = Fuo
    vm_kwm = Fvo
end


# Sum across modes to get total flux vector, then magnitude
umtot = dropdims(sum(um_kwm, dims=2), dims=2)
vmtot = dropdims(sum(vm_kwm, dims=2), dims=2)
flux_mag = sqrt.(umtot.^2 .+ vmtot.^2)


# ── Helper function: add land polygons to any GeoAxis ─────────────────────
function add_land!(ax)
    land = GeoMakie.land()
    poly!(ax, land, color=:lightgray, strokecolor=:gray40, strokewidth=0.5)
end


# ── Figure 1: Mooring locations ────────────────────────────────────────────
fig1 = Figure(size=(900, 500))
ax1 = GeoAxis(fig1[1,1],
    dest   = "+proj=eqc",
    xlabel = "Longitude (°E)",
    ylabel = "Latitude (°N)",
    title  = "Figure 1: Mooring Locations ($nmoor moorings)",
    limits = (-180, 180, -80, 70),
    xgridvisible = false,
    ygridvisible = false)


add_land!(ax1)


scatter!(ax1, lono, lato,
    color       = :steelblue,
    markersize  = 8,
    strokecolor = :black,
    strokewidth = 0.8)


save("figure1_mooring_locations.png", fig1)
display(fig1)


# ── Figure 2: Flux magnitude colored scatter ───────────────────────────────
fig2 = Figure(size=(900, 500))
ax2 = GeoAxis(fig2[1,1],
    dest   = "+proj=eqc",
    xlabel = "Longitude (°E)",
    ylabel = "Latitude (°N)",
    title  = "Figure 2: Time-Mean Energy Flux Magnitude (Sum of Modes)",
    limits = (-180, 180, -80, 70),
    xgridvisible = false,
    ygridvisible = false)


add_land!(ax2)


sc2 = scatter!(ax2, lono, lato,
    color       = flux_mag,
    colormap    = :viridis,
    markersize  = 10,
    strokecolor = :black,
    strokewidth = 0.5)


Colorbar(fig2[1,2], sc2, label = "Flux Magnitude (kW/m)")


save("figure2_flux_magnitude.png", fig2)
display(fig2)


# ── Figure 3: Per-mode flux vectors, one panel per mode ────────────────────
fig3   = Figure(size=(800*nmode, 500))
scale  = 2.0


for mode in 1:nmode
    ax = GeoAxis(fig3[1, mode],
        dest   = "+proj=eqc",
        xlabel = "Longitude (°E)",
        ylabel = "Latitude (°N)",
        title  = "Mode $mode Energy Flux Vectors",
        limits = (-180, 180, -80, 70),
        xgridvisible = false,
        ygridvisible = false)


    add_land!(ax)


    scatter!(ax, lono, lato,
        color      = :gray60,
        markersize = 5)


    for j in 1:nmoor
        arrows!(ax,
            [lono[j]], [lato[j]],
            [um_kwm[j, mode] * scale],
            [vm_kwm[j, mode] * scale],
            color     = :firebrick,
            arrowsize = 8,
            linewidth = 1.2)
    end
end


save("figure3_modal_flux_vectors.png", fig3)
display(fig3)


println("All figures saved successfully.")


