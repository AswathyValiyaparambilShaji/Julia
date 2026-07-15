
using MAT
using NCDatasets


# ════════════════════════════════════════════════════════════════════════
# 1) File paths — edit these if your paths differ
# ════════════════════════════════════════════════════════════════════════
file1path = "/home/aswathy/mnt/data/aswathy/Mooring_Data/Flux_mooring_timeseries_ALL.mat"
file2path = "/home/aswathy/mnt/data/aswathy/Mooring_Data/Flux_mooring_timeseries_ALL_OLD.mat"
file3path = "/home/aswathy/mnt/data/aswathy/Mooring_Data/Flux_mooring_timeseries_ALL_IWAP.mat"
ncpath    = "/home/aswathy/Downloads/intrfreq2_M2.nc"


# ════════════════════════════════════════════════════════════════════════
# 2) Read lat/lon from the three .mat files
# ════════════════════════════════════════════════════════════════════════
f1 = matopen(file1path)
lato1 = vec(read(f1, "lato")); lono1 = vec(read(f1, "lono"))
close(f1)


f2 = matopen(file2path)
lato2 = vec(read(f2, "lato")); lono2 = vec(read(f2, "lono"))
close(f2)


f3 = matopen(file3path)
lato3 = vec(read(f3, "lato")); lono3 = vec(read(f3, "lono"))
close(f3)


# ════════════════════════════════════════════════════════════════════════
# 3) Read lat/lon from the .nc file
# ════════════════════════════════════════════════════════════════════════
ds = Dataset(ncpath, "r")
lat_nc = vec(ds["lat"][:])
lon_nc = vec(ds["lon"][:])
close(ds)
lon_nc = mod.(lon_nc,360)


# ════════════════════════════════════════════════════════════════════════
# 4) Collect everything into one master list, tagging where each point
#    came from and its original index within that source.
# ════════════════════════════════════════════════════════════════════════
struct MooringPoint
    source::String
    orig_idx::Int
    lat::Float64
    lon::Float64
end


all_points = MooringPoint[]


for i in eachindex(lato1)
    push!(all_points, MooringPoint("ALL",      i, lato1[i], lono1[i]))
end
for i in eachindex(lato2)
    push!(all_points, MooringPoint("ALL_OLD",  i, lato2[i], lono2[i]))
end
for i in eachindex(lato3)
    push!(all_points, MooringPoint("ALL_IWAP", i, lato3[i], lono3[i]))
end
for i in eachindex(lat_nc)
    push!(all_points, MooringPoint("NC_intrfreq2_M2", i, lat_nc[i], lon_nc[i]))
end


println("Total points collected (before dedup): $(length(all_points))")


# ════════════════════════════════════════════════════════════════════════
# 5) De-duplicate based on lat/lon proximity within a tolerance.
#    Keeps the FIRST occurrence encountered (in the order pushed above:
#    ALL -> ALL_OLD -> ALL_IWAP -> NC file) and records which other
#    sources/indices also matched that same physical mooring.
# ════════════════════════════════════════════════════════════════════════
tol = 0.05  # degrees; adjust if your "same mooring" tolerance differs


unique_points  = MooringPoint[]      # one representative point per unique mooring
duplicate_info = String[]            # human-readable list of matches for that mooring


for p in all_points
    matched_idx = 0
    for (k, u) in enumerate(unique_points)
        if isapprox(p.lat, u.lat; atol=tol) && isapprox(p.lon, u.lon; atol=tol)
            matched_idx = k
            break
        end
    end


    if matched_idx == 0
        push!(unique_points, p)
        push!(duplicate_info, "$(p.source)#$(p.orig_idx)")
    else
        duplicate_info[matched_idx] *= "; $(p.source)#$(p.orig_idx)"
    end
end

println("Unique mooring locations found: $(length(unique_points))")
println("Duplicates removed: $(length(all_points) - length(unique_points))")


# ════════════════════════════════════════════════════════════════════════
# 6) Write the unique moorings to CSV
#    Columns: mooring_id, lat, lon, first_source, first_source_index, matched_from
# ════════════════════════════════════════════════════════════════════════
outpath = "unique_mooring_locations.csv"


open(outpath, "w") do io
    println(io, "mooring_id,lat,lon,first_source,first_source_index,matched_from")
    for (k, p) in enumerate(unique_points)
        println(io, "$k,$(p.lat),$(p.lon),$(p.source),$(p.orig_idx),\"$(duplicate_info[k])\"")
    end
end
println("\n=== File 1 (ALL): $(length(lato1)) moorings ===")
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
println("\n=== NC file (intrfreq2_M2): $(length(lat_nc)) points ===")
for i in eachindex(lat_nc)
    println("  Point $i: (lat=$(lat_nc[i]), lon=$(lon_nc[i]))")
end


println("\n=== Unique mooring locations: $(length(unique_points)) ===")
for (k, p) in enumerate(unique_points)
    println("  Mooring $k: (lat=$(p.lat), lon=$(p.lon)) — first seen in $(p.source)#$(p.orig_idx)")
end




println("Saved unique mooring locations to: $outpath")




