using CSV
using DataFrames


# ════════════════════════════════════════════════════════════════════════
# 1) Path to the CSV produced by the earlier mooring-dedup script
# ════════════════════════════════════════════════════════════════════════
csvpath = "Mooring_locations_updated.csv"


# ════════════════════════════════════════════════════════════════════════
# 2) Box 56 limits — EDIT THESE to the actual bounds you have for Box 56
# ════════════════════════════════════════════════════════════════════════
box_id     = 56

box_minlat = 24.0  # <-- set actual min lat for Box 56
box_maxlat = 31.91   # <-- set actual max lat for Box 56
box_minlon = 193.0   # <-- set actual min lon for Box 56 (use 0-360 convention if needed)
box_maxlon = 199.0   # <-- set actual max lon for Box 56


# ════════════════════════════════════════════════════════════════════════
# 3) Load the unique mooring locations
# ════════════════════════════════════════════════════════════════════════
df = CSV.read(csvpath, DataFrame)


println("Loaded $(nrow(df)) unique mooring locations from $csvpath")
println("Box $box_id limits: lat [$box_minlat, $box_maxlat], lon [$box_minlon, $box_maxlon]")


# ════════════════════════════════════════════════════════════════════════
# 4) Find moorings that fall exactly inside Box 56
# ════════════════════════════════════════════════════════════════════════
inside_mask = (df.lat .>= box_minlat) .& (df.lat .<= box_maxlat) .&
              (df.lon .>= box_minlon) .& (df.lon .<= box_maxlon)


inside_df = df[inside_mask, :]


println("\n=== Moorings found INSIDE Box $box_id: $(nrow(inside_df)) ===")
for row in eachrow(inside_df)
    println("  Mooring $(row.mooring_id): (lat=$(row.lat), lon=$(row.lon)) ",
            "from $(row.first_source)#$(row.first_source_index) ",
            "[matched_from: $(row.matched_from)]")
end


# ════════════════════════════════════════════════════════════════════════
# 5) If nothing is exactly inside, report the nearest mooring(s) to the
#    box (measured as distance to the nearest edge/corner of the box)
# ════════════════════════════════════════════════════════════════════════
function dist_to_box(lat, lon, minlat, maxlat, minlon, maxlon)
    dlat = lat < minlat ? (minlat - lat) : (lat > maxlat ? (lat - maxlat) : 0.0)
    dlon = lon < minlon ? (minlon - lon) : (lon > maxlon ? (lon - maxlon) : 0.0)
    return sqrt(dlat^2 + dlon^2)
end


if nrow(inside_df) == 0
    println("\nNo moorings fall exactly inside Box $box_id.")
    println("Showing nearest mooring(s) to Box $box_id instead:")


    distances = [dist_to_box(row.lat, row.lon, box_minlat, box_maxlat, box_minlon, box_maxlon)
                 for row in eachrow(df)]


    df_with_dist = copy(df)
    df_with_dist.dist_to_box = distances
    sort!(df_with_dist, :dist_to_box)


    n_nearest = min(5, nrow(df_with_dist))  # show top 5 nearest
    for row in eachrow(df_with_dist[1:n_nearest, :])
        println("  Mooring $(row.mooring_id): (lat=$(row.lat), lon=$(row.lon)) ",
                "from $(row.first_source)#$(row.first_source_index), ",
                "distance to box = $(round(row.dist_to_box, digits=4)) deg")
    end
end




