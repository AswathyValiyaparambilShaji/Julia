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


       APE = open(joinpath(base2, "APE", "APE_t_sm_$suffix.bin"), "r") do io
           nbytes = nx * ny * nz * nt * sizeof(Float64)
           reshape(reinterpret(Float64, read(io, nbytes)),
                   nx, ny, nz, nt)
       end


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
# ============ ANOMALY DETECTION FUNCTIONS =================
# ==========================================================


"""
Compute local anomaly for 2D field using neighborhood statistics
"""
function compute_2d_anomaly(data::Matrix{Float64}, i::Int, j::Int, radius::Int=3)
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
        return NaN, NaN, NaN
    end
    
    center_val = data[i, j]
    neighbor_median = median(neighbors)
    neighbor_std = std(neighbors)
    
    if abs(neighbor_median) < 1e-10
        return NaN, NaN, NaN
    end
    
    anomaly_ratio = (center_val - neighbor_median) / abs(neighbor_median)
    z_score = (center_val - neighbor_median) / (neighbor_std + 1e-10)
    
    return anomaly_ratio, z_score, neighbor_median
end


# ==========================================================
# ================ FIND ANOMALOUS PATCHES ==================
# ==========================================================


println("\n" * "="^70)
println("DETECTING ANOMALOUS PATCHES IN DEPTH-INTEGRATED APE")
println("="^70)


high_anomalies = []  # (i, j, APE_value, anomaly_ratio, z_score, neighbor_median)
low_anomalies = []


# Scan the domain
for i in 5:NX-4
    for j in 5:NY-4
        
        if !isfinite(APE_full[i, j])
            continue
        end
        
        anom_ratio, z_score, neighbor_med = compute_2d_anomaly(APE_full, i, j, 3)
        
        if !isfinite(anom_ratio)
            continue
        end
        
        # High anomalies: much higher than surroundings
        if anom_ratio > 3.0 || z_score > 5.0
            push!(high_anomalies, (i, j, APE_full[i,j], anom_ratio, z_score, neighbor_med))
        end
        
        # Low anomalies: much lower than surroundings
        if anom_ratio < -0.7 || z_score < -3.0
            push!(low_anomalies, (i, j, APE_full[i,j], anom_ratio, z_score, neighbor_med))
        end
    end
end


# Sort by anomaly magnitude
sort!(high_anomalies, by = x -> x[5], rev=true)  # by z-score
sort!(low_anomalies, by = x -> x[5])


# ==========================================================
# ====================== REPORT ============================
# ==========================================================


println("\n🔴 HIGH APE PATCHES (top 30):")
println("   i    j     Lon      Lat        APE       Ratio    Z-score  Neighbor_med")
println("-"^85)
for (i, j, ape_val, anom, z, nmed) in high_anomalies[1:min(30, end)]
    @printf("%4d %4d  %7.3f  %7.3f  %10.1f  %7.2fx  %7.2f  %10.1f\n",
            i, j, lon[i], lat[j], ape_val, anom, z, nmed)
end


println("\n🔵 LOW APE PATCHES (top 30):")
println("   i    j     Lon      Lat        APE       Ratio    Z-score  Neighbor_med")
println("-"^85)
for (i, j, ape_val, anom, z, nmed) in low_anomalies[1:min(30, end)]
    @printf("%4d %4d  %7.3f  %7.3f  %10.1f  %7.2fx  %7.2f  %10.1f\n",
            i, j, lon[i], lat[j], ape_val, anom, z, nmed)
end


println("\n📊 SUMMARY:")
println("   Total high patches (>3x or z>5): ", length(high_anomalies))
println("   Total low patches (<-0.7x or z<-3): ", length(low_anomalies))


# ==========================================================
# =================== VISUALIZATION ========================
# ==========================================================


# Create anomaly map
anomaly_map = fill(NaN, NX, NY)
for i in 5:NX-4, j in 5:NY-4
    if isfinite(APE_full[i,j])
        anom_ratio, _, _ = compute_2d_anomaly(APE_full, i, j, 3)
        anomaly_map[i,j] = anom_ratio
    end
end


fig = Figure(size=(1400, 500))


# Plot 1: Original APE
ax1 = Axis(fig[1, 1],
          title="Depth-Integrated Time-Averaged APE",
          xlabel="Longitude [°]",
          ylabel="Latitude [°]",        
          )
ax1.limits[] = (193.0,194.2,24.0, 25.4)#((minimum(lon), maximum(lon)), (minimum(lat), maximum(lat)))


hm1 = CairoMakie.heatmap!(ax1, lon, lat, APE_full;
                         interpolate=false,
                         colormap=:jet,
                         colorrange=(0, 5000))


Colorbar(fig[1, 2], hm1, label="APE [J/m²]")


# Mark high anomalies
if !isempty(high_anomalies)
    high_lons = [lon[x[1]] for x in high_anomalies[1:min(20, end)]]
    high_lats = [lat[x[2]] for x in high_anomalies[1:min(20, end)]]
    scatter!(ax1, high_lons, high_lats, color=:black, marker='□', markersize=8)
end


# Mark low anomalies
if !isempty(low_anomalies)
    low_lons = [lon[x[1]] for x in low_anomalies[1:min(20, end)]]
    low_lats = [lat[x[2]] for x in low_anomalies[1:min(20, end)]]
    scatter!(ax1, low_lons, low_lats, color=:white, marker='○', markersize=8)
end
 

# Plot 2: Anomaly ratio map
ax2 = Axis(fig[1, 3],
          title="APE Anomaly Ratio (vs neighbors)",
          xlabel="Longitude [°]",
          ylabel="Latitude [°]",
          )
ax2.limits[] = (193.0,194.2,24.0, 25.4)#((minimum(lon), maximum(lon)), (minimum(lat), maximum(lat)))



hm2 = CairoMakie.heatmap!(ax2, lon, lat, anomaly_map;
                         interpolate=false,
                         colormap=:RdBu_11,
                         colorrange=(-5, 5))


Colorbar(fig[1, 4], hm2, label="Anomaly Ratio")


display(fig)


# ==========================================================
# ============ DETAILED ANALYSIS OF TOP ANOMALY ============
# ==========================================================


if !isempty(high_anomalies)
    test_i, test_j = high_anomalies[1][1], high_anomalies[1][2]
    test_lon = lon[test_i]
    test_lat = lat[test_j]
    
    println("\n" * "="^70)
    println("DETAILED ANALYSIS OF HIGHEST ANOMALY")
    println("="^70)
    println("Location: Lon=$(round(test_lon, digits=3))°, Lat=$(round(test_lat, digits=3))°")
    println("Grid index: (i=$test_i, j=$test_j)")
    println("APE value: $(round(APE_full[test_i, test_j], digits=1)) J/m²")
    
    # Find tile
    test_xn = div(test_i - 1, tx) + 1
    test_yn = div(test_j - 1, ty) + 1
    test_suffix = @sprintf("%02dx%02d_%d", test_xn, test_yn, buf)
    
    local_i = mod(test_i - 1, tx) + 1 + buf
    local_j = mod(test_j - 1, ty) + 1 + buf
    
    println("Tile: $test_suffix, Local indices: (i=$local_i, j=$local_j)")
    
    # Extract 5x5 neighborhood for inspection
    println("\n5x5 Neighborhood APE values:")
    for dj in -2:2
        row_str = ""
        for di in -2:2
            ni, nj = test_i + di, test_j + dj
            if 1 <= ni <= NX && 1 <= nj <= NY
                val = APE_full[ni, nj]
                if isfinite(val)
                    row_str *= @sprintf("%8.0f ", val)
                else
                    row_str *= "     NaN "
                end
            end
        end
        println(row_str)
    end
end




