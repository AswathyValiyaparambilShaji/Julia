using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML
using CairoMakie, SparseArrays


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin


config_file = get(ENV, "JULIA_CONFIG",
                joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)


base  = cfg["base_path"]
base2 = cfg["base_path2"]


NX, NY = 288, 468
minlat, maxlat = 24.0, 31.91
minlon, maxlon = 193.0, 199.0
lat = range(minlat, maxlat, length=NY)
lon = range(minlon, maxlon, length=NX)


buf = 3
tx, ty = 47, 66
nx = tx + 2*buf
ny = ty + 2*buf
nz = 88


dt  = 25
dto = 144
Tts = 366192
nt  = div(Tts, dto)
ts  = 72
nt_avg = div(nt, ts)


thk = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)


APE_full = fill(NaN, NX, NY)


# ==========================================================
# ============ BUILD DEPTH-INTEGRATED APE MAP ==============
# ==========================================================


for xn in cfg["xn_start"]:cfg["xn_end"]
  for yn in cfg["yn_start"]:cfg["yn_end"]


      suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)


      hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"),
                       (nx, ny, nz))


      DRFfull = DRF3d .* hFacC
      DRFfull[hFacC .== 0] .= 0


      APE = Float64.(open(joinpath(base2, "APE", "APE_t_sm_$suffix.bin"), "r") do io
          nbytes = nx * ny * nz * nt * sizeof(Float32)
          reshape(reinterpret(Float32, read(io, nbytes)),
                  nx, ny, nz, nt)
      end)


      # Time average ignoring NaN
      APE_clean = replace(APE, NaN => 0.0)
      APE_sum = sum(APE_clean, dims=4)
      APE_count = sum(.!isnan.(APE), dims=4)


      aped = zeros(Float64, nx, ny, nz)
      for i in 1:nx, j in 1:ny, k in 1:nz
          if APE_count[i,j,k,1] > 0
              aped[i,j,k] = APE_sum[i,j,k,1] / APE_count[i,j,k,1]
          else
              aped[i,j,k] = NaN
          end
      end


      # Depth integrate
      ape = zeros(Float64, nx, ny)
      for i in 1:nx, j in 1:ny
          weighted = aped[i,j,:] .* DRF3d[i,j,:]
          ape[i,j] = sum(weighted[.!isnan.(weighted)])
      end


      xs = (xn - 1) * tx + 1
      xe = xs + tx - 1
      ys = (yn - 1) * ty + 1
      ye = ys + ty - 1


      ape_interior = ape[buf+1:nx-buf, buf+1:ny-buf]


      APE_full[xs:xe, ys:ye] .= ape_interior


      println("Completed tile $suffix")
  end
end


println("\nAPE_full range: $(minimum(skipmissing(APE_full))) to $(maximum(skipmissing(APE_full)))")


# ==========================================================
# ============ DETECT OUTLIERS WITH SMALL DIFFERENCES ======
# ==========================================================


"""
Compute local statistics for sensitive outlier detection
"""
function compute_local_stats(data::Matrix{Float64}, i::Int, j::Int, radius::Int=2)
    neighbors = Float64[]
    
    # Get neighborhood (excluding center)
    for di in -radius:radius, dj in -radius:radius
        if di == 0 && dj == 0
            continue
        end
        
        ni, nj = i + di, j + dj
        
        if 1 <= ni <= size(data, 1) && 1 <= nj <= size(data, 2)
            val = data[ni, nj]
            if isfinite(val)
                push!(neighbors, val)
            end
        end
    end
    
    if length(neighbors) < 5
        return NaN, NaN, NaN, NaN, NaN
    end
    
    center_val = data[i, j]
    neighbor_median = median(neighbors)
    neighbor_mean = mean(neighbors)
    neighbor_std = std(neighbors)
    
    # Absolute difference from mean
    abs_diff = abs(center_val - neighbor_mean)
    
    # Relative difference from median (percentage)
    rel_diff = abs(center_val - neighbor_median) / (abs(neighbor_median) + 1e-10)
    
    # Z-score relative to neighbors
    z_score = (center_val - neighbor_mean) / (neighbor_std + 1e-10)
    
    return abs_diff, rel_diff, z_score, neighbor_mean, neighbor_median
end


# ==========================================================
# ============ FIND OUTLIER POINTS =========================
# ==========================================================


println("\n" * "="^70)
println("DETECTING OUTLIER POINTS (SENSITIVE)")
println("="^70)


outliers = []  # Store outlier information


# MUCH MORE SENSITIVE CRITERIA
ABS_DIFF_THRESHOLD = 100.0      # Absolute difference > 100 J/m²
REL_DIFF_THRESHOLD = 0.2        # Relative difference > 20%
Z_THRESHOLD = 1.5               # Z-score > 1.5 (much lower than before)


for i in 5:NX-4
    for j in 5:NY-4
        
        if !isfinite(APE_full[i, j])
            continue
        end
        
        abs_diff, rel_diff, z_score, neighbor_mean, neighbor_median = 
            compute_local_stats(APE_full, i, j, 2)
        
        if !isfinite(z_score)
            continue
        end
        
        # Detect outliers based on multiple criteria
        is_outlier = false
        outlier_reason = ""
        
        # Check absolute difference
        if abs_diff > ABS_DIFF_THRESHOLD
            is_outlier = true
            outlier_reason *= "abs_diff "
        end
        
        # Check relative difference
        if rel_diff > REL_DIFF_THRESHOLD
            is_outlier = true
            outlier_reason *= "rel_diff "
        end
        
        # Check z-score
        if abs(z_score) > Z_THRESHOLD
            is_outlier = true
            outlier_reason *= "z_score "
        end
        
        if is_outlier
            outlier_type = z_score > 0 ? "HIGH" : "LOW"
            push!(outliers, (
                i=i, j=j, 
                lon=lon[i], lat=lat[j],
                APE=APE_full[i,j],
                abs_diff=abs_diff,
                rel_diff=rel_diff,
                z_score=z_score,
                neighbor_mean=neighbor_mean,
                neighbor_median=neighbor_median,
                type=outlier_type,
                reason=outlier_reason
            ))
        end
    end
end


# Sort by absolute difference
sort!(outliers, by = x -> x.abs_diff, rev=true)


# ==========================================================
# ============ REPORT OUTLIERS =============================
# ==========================================================


println("\nDetection criteria:")
println("  Absolute difference > $ABS_DIFF_THRESHOLD J/m²")
println("  Relative difference > $(REL_DIFF_THRESHOLD*100)%")
println("  Z-score > $Z_THRESHOLD")


println("\n" * "="^70)
println("OUTLIERS DETECTED (top 50):")
println("="^70)
println("   i    j     Lon      Lat        APE     Abs_Diff  Rel_Diff  Z-score  Neighbor_med  Type  Reason")
println("-"^110)


for pt in outliers[1:min(50, end)]
    @printf("%4d %4d  %7.3f  %7.3f  %10.1f  %8.1f  %7.2f%%  %7.2f  %10.1f   %4s  %s\n",
            pt.i, pt.j, pt.lon, pt.lat, pt.APE, 
            pt.abs_diff, pt.rel_diff*100, pt.z_score,
            pt.neighbor_median, pt.type, pt.reason)
end


high_outliers = filter(x -> x.type == "HIGH", outliers)
low_outliers = filter(x -> x.type == "LOW", outliers)


println("\n📊 SUMMARY:")
println("   Total outliers detected: ", length(outliers))
println("   High outliers: ", length(high_outliers))
println("   Low outliers: ", length(low_outliers))


# ==========================================================
# =========== COMPUTE SMOOTHNESS MAP =======================
# ==========================================================


println("\n" * "="^70)
println("COMPUTING SPATIAL SMOOTHNESS MAP")
println("="^70)


smoothness_map = fill(NaN, NX, NY)


for i in 5:NX-4
    for j in 5:NY-4
        if !isfinite(APE_full[i, j])
            continue
        end
        
        abs_diff, rel_diff, z_score, _, _ = compute_local_stats(APE_full, i, j, 2)
        
        # Use absolute difference as smoothness metric
        if isfinite(abs_diff)
            smoothness_map[i, j] = abs_diff
        end
    end
end


println("Smoothness map computed (represents local variability)")


# ==========================================================
# ============ VISUALIZATION ===============================
# ==========================================================


fig = Figure(size=(1800, 1200))


# Plot 1: APE map with outliers marked
ax1 = Axis(fig[1, 1],
         title="Depth-Integrated Time-Averaged APE",
         xlabel="Longitude [°]",
         ylabel="Latitude [°]")


hm1 = CairoMakie.heatmap!(ax1, lon, lat, APE_full;
                        interpolate=false,
                        colormap=:jet,
                        colorrange=(0, 5000))


Colorbar(fig[1, 2], hm1, label="APE [J/m²]")


# Mark high outliers
if !isempty(high_outliers)
    high_lons = [x.lon for x in high_outliers]
    high_lats = [x.lat for x in high_outliers]
    scatter!(ax1, high_lons, high_lats, 
            color=:red, marker='○', markersize=10, 
            strokecolor=:white, strokewidth=2,
            label="High outliers")
end


# Mark low outliers
if !isempty(low_outliers)
    low_lons = [x.lon for x in low_outliers]
    low_lats = [x.lat for x in low_outliers]
    scatter!(ax1, low_lons, low_lats,
            color=:cyan, marker='○', markersize=10,
            strokecolor=:white, strokewidth=2,
            label="Low outliers")
end


axislegend(ax1, position=:lt)


# Plot 2: Smoothness/Variability map
ax2 = Axis(fig[1, 3],
         title="Local Variability (Abs Diff from Neighbors)",
         xlabel="Longitude [°]",
         ylabel="Latitude [°]")


hm2 = CairoMakie.heatmap!(ax2, lon, lat, smoothness_map;
                        interpolate=false,
                        colormap=:hot,
                        colorrange=(0, 500))


Colorbar(fig[1, 4], hm2, label="Absolute Difference [J/m²]")


# Plot 3: Zoomed view of highest variability region
if !isempty(outliers)
    # Find region with most outliers
    outlier_lons = [x.lon for x in outliers]
    outlier_lats = [x.lat for x in outliers]
    
    center_lon = median(outlier_lons)
    center_lat = median(outlier_lats)
    
    zoom_range_lon = 1.0  # degrees
    zoom_range_lat = 1.0  # degrees
    
    ax3 = Axis(fig[2, 1:2],
             title="Zoomed View - High Variability Region",
             xlabel="Longitude [°]",
             ylabel="Latitude [°]")
    
    ax3.limits = (center_lon - zoom_range_lon, center_lon + zoom_range_lon,
                  center_lat - zoom_range_lat, center_lat + zoom_range_lat)
    
    hm3 = CairoMakie.heatmap!(ax3, lon, lat, APE_full;
                            interpolate=false,
                            colormap=:jet,
                            colorrange=(0, 5000))
    
    # Mark outliers in zoomed view
    scatter!(ax3, outlier_lons, outlier_lats,
            color=:black, marker='x', markersize=12,
            strokewidth=3)
    
    Colorbar(fig[2, 3], hm3, label="APE [J/m²]")
end


display(fig)


# ==========================================================
# ============ DETAILED ANALYSIS OF TOP OUTLIERS ===========
# ==========================================================


if !isempty(outliers)
    println("\n" * "="^70)
    println("DETAILED VIEW OF TOP 10 OUTLIERS")
    println("="^70)
    
    for (idx, pt) in enumerate(outliers[1:min(10, end)])
        println("\n--- Outlier #$idx ---")
        println("Location: (i=$(pt.i), j=$(pt.j)) → Lon=$(round(pt.lon, digits=3))°, Lat=$(round(pt.lat, digits=3))°")
        println("APE value: $(round(pt.APE, digits=1)) J/m²")
        println("Absolute difference: $(round(pt.abs_diff, digits=1)) J/m²")
        println("Relative difference: $(round(pt.rel_diff*100, digits=1))%")
        println("Z-score: $(round(pt.z_score, digits=2))")
        println("Neighbor median: $(round(pt.neighbor_median, digits=1)) J/m²")
        println("Type: $(pt.type)")
        println("Triggered by: $(pt.reason)")
        
        # Show 5x5 neighborhood
        println("\n5x5 Neighborhood APE values:")
        for dj in -2:2
            row_str = ""
            for di in -2:2
                ni, nj = pt.i + di, pt.j + dj
                if 1 <= ni <= NX && 1 <= nj <= NY
                    val = APE_full[ni, nj]
                    if di == 0 && dj == 0
                        row_str *= @sprintf(" [%7.0f]", val)  # Mark center
                    elseif isfinite(val)
                        row_str *= @sprintf("  %7.0f ", val)
                    else
                        row_str *= "      NaN "
                    end
                end
            end
            println(row_str)
        end
    end
end




