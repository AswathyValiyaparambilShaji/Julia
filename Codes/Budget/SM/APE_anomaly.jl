using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter


config_file = get(ENV, "JULIA_CONFIG",
                 joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)


base  = cfg["base_path"]
base2 = cfg["base_path2"]


# --- Tile & time ---
buf = 3
tx, ty = 47, 66
nx = tx + 2*buf
ny = ty + 2*buf
nz = 88


dt  = 25
dto = 144
Tts = 366192
nt  = div(Tts, dto)


ts      = 72      # 3-day window
nt_avg = div(nt, ts)


"""
Compute local anomaly: how much does this point differ from its neighbors?
Returns the ratio: (value - median_of_neighbors) / median_of_neighbors
"""
function compute_local_anomaly(data::Array{Float64, 3}, i::Int, j::Int, k::Int)
    neighbors = Float64[]
    
    # Get 3x3 neighborhood (excluding center)
    for di in -1:1, dj in -1:1
        if di == 0 && dj == 0
            continue  # skip center point
        end
        
        ni, nj = i + di, j + dj
        
        # Check bounds
        if 1 <= ni <= size(data, 1) && 1 <= nj <= size(data, 2)
            val = data[ni, nj, k]
            if isfinite(val)
                push!(neighbors, val)
            end
        end
    end
    
    if length(neighbors) < 3  # need at least 3 neighbors
        return NaN
    end
    
    center_val = data[i, j, k]
    neighbor_median = median(neighbors)
    
    if abs(neighbor_median) < 1e-10  # avoid division by tiny numbers
        return NaN
    end
    
    anomaly = (center_val - neighbor_median) / abs(neighbor_median)
    return anomaly
end


# ==========================================================
# =================== ANALYZE ANOMALIES ====================
# ==========================================================


for xn in cfg["xn_start"]:cfg["xn_end"]
   for yn in cfg["yn_start"]:cfg["yn_end"]


       suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
       
       println("\n" * "="^60)
       println("Analyzing tile: $suffix")
       println("="^60)


       # --- Read APE ---
       APE = open(joinpath(base2, "APE_no_reg", "APE_t_sm_$suffix.bin"), "r") do io
           raw = read(io, nx * ny * nz * nt * sizeof(Float64))
           reshape(reinterpret(Float64, raw), nx, ny, nz, nt)
       end
       
       # --- Read N2 ---
       N2_phase = open(joinpath(base,"3day_mean","N2","N2_3day_$suffix.bin"), "r") do io
           raw = read(io, nx * ny * nz * nt_avg * sizeof(Float64))
           reshape(reinterpret(Float64, raw), nx, ny, nz, nt_avg)
       end
       
       # --- Read b ---
       b = open(joinpath(base2, "b", "b_t_sm_$suffix.bin"), "r") do io
           raw = read(io, nx * ny * nz * nt * sizeof(Float64))
           reshape(reinterpret(Float64, raw), nx, ny, nz, nt)
       end
       
       # --- Read hFacC ---
       hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))


       # Analyze a snapshot (let's use t=1 as example)
       t = 1
       APE_snap = APE[:, :, :, t]
       b_snap = b[:, :, :, t]
       N2_snap = N2_phase[:, :, :, 1]  # corresponding 3-day mean
       
       # Store anomalies
       high_anomalies = []  # (i, j, k, APE_value, anomaly_ratio, N2, b)
       low_anomalies = []
       
       # Scan through the domain
       for k in 1:nz
           for i in 2:nx-1  # avoid edges
               for j in 2:ny-1
                   
                   if hFacC[i, j, k] == 0 || !isfinite(APE_snap[i, j, k])
                       continue
                   end
                   
                   anomaly = compute_local_anomaly(APE_snap, i, j, k)
                   
                   if !isfinite(anomaly)
                       continue
                   end
                   
                   # Find high anomalies (point much higher than neighbors)
                   if anomaly > 5.0  # 5x higher than median of neighbors
                       push!(high_anomalies, (i, j, k, APE_snap[i,j,k], anomaly, 
                                             N2_snap[i,j,k], b_snap[i,j,k]))
                   end
                   
                   # Find low anomalies (point much lower than neighbors)
                   if anomaly < -0.8  # 80% lower than median of neighbors
                       push!(low_anomalies, (i, j, k, APE_snap[i,j,k], anomaly,
                                            N2_snap[i,j,k], b_snap[i,j,k]))
                   end
               end
           end
       end
       
       # Sort by anomaly magnitude
       sort!(high_anomalies, by = x -> x[5], rev=true)
       sort!(low_anomalies, by = x -> x[5])
       
       # Report findings
       println("\n🔴 HIGH APE ANOMALIES (top 20):")
       println("   i    j    k        APE          Anomaly      N²            b")
       println("-"^75)
       for (idx, (i, j, k, ape, anom, n2, b_val)) in enumerate(high_anomalies[1:min(20, end)])
           @printf("%4d %4d %4d  %12.3e  %8.2fx  %12.3e  %12.3e\n", 
                   i, j, k, ape, anom, n2, b_val)
       end
       
       println("\n🔵 LOW APE ANOMALIES (top 20):")
       println("   i    j    k        APE          Anomaly      N²            b")
       println("-"^75)
       for (idx, (i, j, k, ape, anom, n2, b_val)) in enumerate(low_anomalies[1:min(20, end)])
           @printf("%4d %4d %4d  %12.3e  %8.2fx  %12.3e  %12.3e\n", 
                   i, j, k, ape, anom, n2, b_val)
       end
       
       println("\n📊 SUMMARY:")
       println("   Total high anomalies (>5x neighbors): ", length(high_anomalies))
       println("   Total low anomalies (<-0.8x neighbors): ", length(low_anomalies))
       
       # Check what's causing the high anomalies
       if !isempty(high_anomalies)
           n2_values = [x[6] for x in high_anomalies[1:min(100, end)]]
           b_values = [x[7] for x in high_anomalies[1:min(100, end)]]
           println("\n   In top 100 high anomalies:")
           println("      N² range: ", extrema(n2_values))
           println("      b range: ", extrema(b_values))
           println("      Negative N² count: ", sum(n2_values .< 0))
           println("      Very small N² (|N²| < 1e-5) count: ", sum(abs.(n2_values) .< 1e-5))
       end
       
       println("\nCompleted analysis for tile: $suffix")
   end
end


