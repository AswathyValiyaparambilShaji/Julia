using NCDatasets
using CairoMakie
using GeoMakie
using GeoMakie.GeoJSON




# Open the .nc file
ds = Dataset("/home/aswathy/Downloads/intrfreq2_M2.nc", "r")
println(ds)
#println(keys(ds))
#println(ds.dim)




# ── Read variables ─────────────────────────────────────────────────────────────
lat    = ds["lat"][:]
lon    = ds["lon"][:]
umtot  = ds["umtot_kwm"][:]
vmtot  = ds["vmtot_kwm"][:]
um_kwm = ds["um_kwm"][:, :]
vm_kwm = ds["vm_kwm"][:, :]
KEint  = ds["KEint"][:, :]
PEint  = ds["PEint"][:, :]
Eint   = ds["Eint"][:, :]




close(ds)




# ── Apply Matthew's corrections ────────────────────────────────────────────────
umtot[10] = 0.6423;  vmtot[10] = 0.3952
umtot[57] = 0.587;   vmtot[57] = 0.3030
umtot[58] = 0.6767;  vmtot[58] = 0.1443
umtot[59] = 0.1878;  vmtot[59] = 0.1880
umtot[60] = 0.46;    vmtot[60] = 0.37




# ── Derived quantities ─────────────────────────────────────────────────────────
flux_mag = sqrt.((um_kwm[:,1] .+ um_kwm[:,2]).^2 .+ (vm_kwm[:,1] .+ vm_kwm[:,2]).^2)
E_total  = Eint[:,1] .+ Eint[:,2]


# Guard against log10(0) or negative values before log-scaling
flux_mag_safe = max.(flux_mag, eps())




# ── Helper function: add land polygons to any GeoAxis ─────────────────────────
function add_land!(ax)
   land = GeoMakie.land()
   poly!(ax, land,
       color       = :lightgray,
       strokecolor = :gray40,
       strokewidth = 0.5)
end




# ── Helper function: common axis decoration kwargs ─────────────────────────────
# GeoAxis has no spine attributes, so this only keeps ticks/labels visible
# even with the interior gridlines turned off.
axis_decorations = (
    xgridvisible        = false,
    ygridvisible         = false,
    xticksvisible        = true,
    yticksvisible        = true,
    xticklabelsvisible   = true,
    yticklabelsvisible   = true,
    xlabelvisible        = true,
    ylabelvisible        = true,
)




# ── Helper function: draw a visible rectangular border around the map ─────────
# GeoAxis has no spine/frame attribute, so we draw the border explicitly as a
# closed line at the plot's lon/lat limits.
function add_border!(ax; lonlims=(-180, 180), latlims=(-80, 70))
    lo1, lo2 = lonlims
    la1, la2 = latlims
    box_lon = [lo1, lo2, lo2, lo1, lo1]
    box_lat = [la1, la1, la2, la2, la1]
    lines!(ax, box_lon, box_lat, color = :black, linewidth = 1.2)
end




# ── Figure 1: Mooring locations ────────────────────────────────────────────────
fig1 = Figure(size=(900, 500))
ax1  = GeoAxis(fig1[1,1],
   dest   = "+proj=eqc",
   xlabel = "Longitude (°E)",
   ylabel = "Latitude (°N)",
   title  = "Figure 1: Mooring Locations (82 moorings)",
   limits = (-180, 180, -80, 70);
   axis_decorations...)




add_land!(ax1)
add_border!(ax1)




scatter!(ax1, lon, lat,
   color       = :steelblue,
   markersize  = 8,
   strokecolor = :black,
   strokewidth = 0.8)




save("figure1_mooring_locations.png", fig1)
display(fig1)




# ── Figure 2: Flux magnitude colored scatter (LOG SCALE) ───────────────────────
fig2 = Figure(size=(900, 500))
ax2  = GeoAxis(fig2[1,1],
   dest   = "+proj=eqc",
   xlabel = "Longitude (°E)",
   ylabel = "Latitude (°N)",
   title  = "Figure 2: M2 Baroclinic Energy Flux Magnitude (Mode 1 + Mode 2)",
   limits = (-180, 180, -80, 70);
   axis_decorations...)




add_land!(ax2)
add_border!(ax2)




sc2 = scatter!(ax2, lon, lat,
   color       = flux_mag_safe,
   colormap    = :viridis,
   colorscale  = log10,
   markersize  = 10,
   strokecolor = :black,
   strokewidth = 0.5)




Colorbar(fig2[1,2], sc2, label = "Flux Magnitude (kW/m, log scale)", scale = log10)




save("figure2_flux_magnitude.png", fig2)
display(fig2)




# ── Figure 3: Mode 1 and Mode 2 flux vectors side by side ─────────────────────
fig3  = Figure(size=(1600, 500))
scale = 2.0




for (i, mode) in enumerate([1, 2])
   ax = GeoAxis(fig3[1, i],
       dest   = "+proj=eqc",
       xlabel = "Longitude (°E)",
       ylabel = "Latitude (°N)",
       title  = "Figure 3$(i==1 ? "a" : "b"): Mode $mode M2 Energy Flux Vectors",
       limits = (-180, 180, -80, 70);
       axis_decorations...)




   add_land!(ax)
   add_border!(ax)




   scatter!(ax, lon, lat,
       color      = :gray60,
       markersize = 5)




   for j in 1:length(lon)
       arrows!(ax,
           [lon[j]], [lat[j]],
           [um_kwm[j, mode] * scale],
           [vm_kwm[j, mode] * scale],
           color     = :firebrick,
           arrowsize = 8,
           linewidth = 1.2)
   end
end




save("figure3_modal_flux_vectors.png", fig3)
display(fig3)




# ── Figure 4: Total energy colored scatter ─────────────────────────────────────
fig4 = Figure(size=(900, 500))
ax4  = GeoAxis(fig4[1,1],
   dest   = "+proj=eqc",
   xlabel = "Longitude (°E)",
   ylabel = "Latitude (°N)",
   title  = "Figure 4: Total Baroclinic Energy (Mode 1 + Mode 2)",
   limits = (-180, 180, -80, 70);
   axis_decorations...)




add_land!(ax4)
add_border!(ax4)




sc4 = scatter!(ax4, lon, lat,
   color       = E_total,
   colormap    = :plasma,
   markersize  = 10,
   strokecolor = :black,
   strokewidth = 0.5)




Colorbar(fig4[1,2], sc4, label = "Total Energy (J/m²)")




save("figure4_total_energy.png", fig4)
display(fig4)




println("All figures saved successfully.")




