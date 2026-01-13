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
# ============ DETECT OUTLIERS USING LOCAL STATISTICS =====
# ==========================================================


"""
Compute local statistics for outlier detection
"""
function compute_local_stats(data::Matrix{Float64}, i::Int, j::Int, radius::Int=3)
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
        return NaN, NaN, NaN, NaN
    end
    
    center_val = data[i, j]
    neighbor_median = median(neighbors)
    neighbor_mean = mean(neighbors)
    neighbor_std = std(neighbors)
    
    # Z-score relative to neighbors
    z_score = (center_val - neighbor_mean) / (neighbor_std + 1e-10)
    
    # Ratio relative to median
    ratio = center_val / (neighbor_median + 1e-10)
    
    return z_score, ratio, neighbor_mean, neighbor_median
end


# ==========================================================
# ============ FIND OUTLIER POINTS =========================
# ==========================================================


println("\n" * "="^70)
println("DETECTING OUTLIER POINTS")
println("="^70)


outliers = []  # Store (i, j, APE_value, z_score, ratio, type)


# Define outlier criteria
Z_THRESHOLD_HIGH = 5.0    # Points much higher than neighbors
Z_THRESHOLD_LOW = -5.0    # Points much lower than neighbors
RATIO_THRESHOLD_HIGH = 3.0  # Points 3x higher than neighbors
RATIO_THRESHOLD_LOW = 0.3   # Points 1/3 of neighbors


for i in 5:NX-4
    for j in 5:NY-4
        
        if !isfinite(APE_full[i, j])
            continue
        end
        
        z_score, ratio, neighbor_mean, neighbor_median = compute_local_stats(APE_full, i, j, 3)
        
        if !isfinite(z_score)
            continue
        end
        
        # Detect high outliers
        if z_score > Z_THRESHOLD_HIGH || ratio > RATIO_THRESHOLD_HIGH
            push!(outliers, (i=i, j=j, lon=lon[i], lat=lat[j], 
                           APE=APE_full[i,j], z_score=z_score, ratio=ratio,
                           neighbor_median=neighbor_median, type="HIGH"))
        end
        
        # Detect low outliers (anomalously low points)
        if z_score < Z_THRESHOLD_LOW || ratio < RATIO_THRESHOLD_LOW
            push!(outliers, (i=i, j=j, lon=lon[i], lat=lat[j],
                           APE=APE_full[i,j], z_score=z_score, ratio=ratio,
                           neighbor_median=neighbor_median, type="LOW"))
        end
    end
end


# Sort by absolute z-score
sort!(outliers, by = x -> abs(x.z_score), rev=true)


# ==========================================================
# ============ REPORT OUTLIERS =============================
# ==========================================================


println("\n🔴 HIGH OUTLIERS (top 30):")
println("   i    j     Lon      Lat        APE      Z-score   Ratio  Neighbor_med  Type")
println("-"^95)
high_outliers = filter(x -> x.type == "HIGH", outliers)
for pt in high_outliers[1:min(30, end)]
    @printf("%4d %4d  %7.3f  %7.3f  %10.1f  %8.2f  %6.2f  %10.1f   %s\n",
            pt.i, pt.j, pt.lon, pt.lat, pt.APE, pt.z_score, pt.ratio, 
            pt.neighbor_median, pt.type)
end


println("\n🔵 LOW OUTLIERS (top 30):")
println("   i    j     Lon      Lat        APE      Z-score   Ratio  Neighbor_med  Type")
println("-"^95)
low_outliers = filter(x -> x.type == "LOW", outliers)
for pt in low_outliers[1:min(30, end)]
    @printf("%4d %4d  %7.3f  %7.3f  %10.1f  %8.2f  %6.2f  %10.1f   %s\n",
            pt.i, pt.j, pt.lon, pt.lat, pt.APE, pt.z_score, pt.ratio,
            pt.neighbor_median, pt.type)
end


println("\n📊 SUMMARY:")
println("   Total high outliers: ", length(high_outliers))
println("   Total low outliers: ", length(low_outliers))
println("   Total outliers: ", length(outliers))


# ==========================================================
# ============ VISUALIZATION WITH OUTLIERS MARKED ==========
# ==========================================================


fig = Figure(size=(1200, 900))


ax = Axis(fig[1, 1],
         title="Depth-Integrated Time-Averaged APE with Outliers",
         xlabel="Longitude [°]",
         ylabel="Latitude [°]")


hm = CairoMakie.heatmap!(ax, lon, lat, APE_full;
                        interpolate=false,
                        colormap=:jet,
                        colorrange=(0, 5000))


Colorbar(fig[1, 2], hm, label="APE [J/m²]")


# Mark high outliers with red circles
if !isempty(high_outliers)
    high_lons = [x.lon for x in high_outliers[1:min(50, end)]]
    high_lats = [x.lat for x in high_outliers[1:min(50, end)]]
    scatter!(ax, high_lons, high_lats, 
            color=:red, marker='○', markersize=12, 
            strokecolor=:white, strokewidth=2,
            label="High outliers")
end


# Mark low outliers with blue circles
if !isempty(low_outliers)
    low_lons = [x.lon for x in low_outliers[1:min(50, end)]]
    low_lats = [x.lat for x in low_outliers[1:min(50, end)]]
    scatter!(ax, low_lons, low_lats,
            color=:blue, marker='○', markersize=12,
            strokecolor=:white, strokewidth=2,
            label="Low outliers")
end


#axislegend(ax, position=:lt)


display(fig)


# ==========================================================
# ============ ZOOM IN ON SPECIFIC OUTLIERS ================
# ==========================================================


if !isempty(outliers)
    println("\n" * "="^70)
    println("DETAILED VIEW OF TOP 5 OUTLIERS")
    println("="^70)
    
    for (idx, pt) in enumerate(outliers[1:min(5, end)])
        println("\n--- Outlier #$idx ---")
        println("Location: (i=$(pt.i), j=$(pt.j)) → Lon=$(round(pt.lon, digits=3))°, Lat=$(round(pt.lat, digits=3))°")
        println("APE value: $(round(pt.APE, digits=1)) J/m²")
        println("Z-score: $(round(pt.z_score, digits=2))")
        println("Ratio to neighbors: $(round(pt.ratio, digits=2))")
        println("Neighbor median: $(round(pt.neighbor_median, digits=1)) J/m²")
        println("Type: $(pt.type)")
        
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




