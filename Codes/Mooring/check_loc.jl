using MAT, CSV, DataFrames, CairoMakie


df   = CSV.read("mnt/data/aswathy/MITgcm_NAS/Moorings/unique_mooring_locations_v2.csv", DataFrame)
vars = matread("mnt/data/aswathy/MITgcm_NAS/MooringLocations.mat")


lat_csv = df.lat
lon_csv = mod.(df.lon, 360)
lat_mat = vec(vars["lat"])[1:88]
lon_mat = mod.(vec(vars["lon"])[1:88], 360)


# ── Check: does each (lat, lon) pair match? ──
tol = 0.05   # degrees
for i in 1:88
    ok = abs(lat_csv[i] - lat_mat[i]) < tol && abs(lon_csv[i] - lon_mat[i]) < tol
    println(i, "  CSV (", lat_csv[i], ", ", lon_csv[i], ")   MAT (", lat_mat[i], ", ", lon_mat[i], ")   ", ok ? "match" : "NO MATCH")
end


# ── Plot: black = CSV, yellow = .mat ──
regions = [(125,160,-55,45), (190,240,-5,50), (280,318,20,60),
           (328,355,20,56),  (310,350,-72,-15), (0,62,-60,-25)]


fig = Figure(size=(1500, 900))
for (k, r) in enumerate(regions)
    ax = Axis(fig[(k-1)÷3+1, (k-1)%3+1], limits=r)
    scatter!(ax, lon_csv, lat_csv, color=:black,  markersize=16)
    scatter!(ax, lon_mat, lat_mat, color=:yellow, markersize=6)
end
#save("mooring_check.png", fig)
display(fig)



# ======================== ========================= ============================ =============================


using MAT, NCDatasets, CairoMakie


# ── ALL.mat mooring locations ──
f = matopen("/home/aswathy/mnt/data/aswathy/Mooring_Data/Flux_mooring_timeseries_ALL.mat")
lat_all = vec(read(f, "lato"))
lon_all = mod.(vec(read(f, "lono")), 360)
close(f)


# ── Model flux station locations ──
ds = Dataset("/home/aswathy/mnt/data/aswathy/MITgcm_NAS/Moorings/Mooring_modal_fluxes.nc")
lat_mod = vec(ds["lat"][1:88])          # change "lat"/"lon" if named differently
lon_mod = mod.(vec(ds["lon"][1:88]), 360)
close(ds)


# ── Plot: black = ALL.mat, yellow = model ──
regions = [(125,160,-55,45), (190,240,-5,50), (280,318,20,60),
           (328,355,20,56),  (310,350,-72,-15), (0,62,-60,-25)]


fig2 = Figure(size=(1500, 900))
for (k, r) in enumerate(regions)
    ax = Axis(fig2[(k-1)÷3+1, (k-1)%3+1], limits=r)
    scatter!(ax, lon_all, lat_all, color=:black,  markersize=28)
    scatter!(ax, lon_mod, lat_mod, color=:orange, markersize=10)
    text!(ax, lon_all, lat_all, text=string.(1:length(lat_all)),
          fontsize=9, offset=(6,4))      # ALL.mat mooring numbers
end
save("ALL_vs_model_locations.png", fig2)
display(fig2)





