using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin


config_file = get(ENV, "JULIA_CONFIG",
                joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)


base  = cfg["base_path"]


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


# Storage for analysis
N2_threshold = 1e-8


# Store information about problematic points
problematic_points = []


println("="^70)
println("IDENTIFYING POINTS WITH N² < $(N2_threshold)")
println("="^70)


# ==========================================================
# ====== SCAN ALL TILES AND IDENTIFY LOW N2 POINTS ========
# ==========================================================


for xn in cfg["xn_start"]:cfg["xn_end"]
  for yn in cfg["yn_start"]:cfg["yn_end"]


      suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
      println("\nProcessing tile $suffix...")


      # --- Read N2 ---
      N2_phase = Float64.(open(joinpath(base,"3day_mean","N2","N2_3day_$suffix.bin"), "r") do io
          raw = read(io, nx * ny * nz * nt_avg * sizeof(Float32))
          reshape(reinterpret(Float32, raw), nx, ny, nz, nt_avg)
      end)


      # --- Adjust N2 to interfaces ---
      N2_adjusted = zeros(Float64, nx, ny, nz+1, nt_avg)
      N2_adjusted[:, :, 1,   :] = N2_phase[:, :, 1,   :]
      N2_adjusted[:, :, 2:nz,:] = N2_phase[:, :, 1:nz-1, :]
      N2_adjusted[:, :, nz+1,:] = N2_phase[:, :, nz-1, :]


      # --- Average to cell centers ---
      N2_center = zeros(Float64, nx, ny, nz, nt_avg)
      for k in 1:nz
          N2_center[:, :, k, :] .=
              0.5 .* (N2_adjusted[:, :, k, :] .+
                      N2_adjusted[:, :, k+1, :])
      end


      # Read hFacC to identify water vs land
      hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"),
                       (nx, ny, nz))


      # Calculate global indices
      xs = (xn - 1) * tx + 1
      xe = xs + tx - 1
      ys = (yn - 1) * ty + 1
      ye = ys + ty - 1


      # Check interior points only (exclude buffer)
      for i_local in (buf+1):(nx-buf)
          for j_local in (buf+1):(ny-buf)
              
              # Global indices
              i_global = xs + (i_local - buf - 1)
              j_global = ys + (j_local - buf - 1)
              
              # Check if this is a water column
              if any(hFacC[i_local, j_local, :] .> 0)
                  
                  # Get all N² values at this location
                  n2_column = N2_center[i_local, j_local, :, :]
                  valid_n2 = filter(isfinite, n2_column)
                  
                  if !isempty(valid_n2)
                      min_n2 = minimum(valid_n2)
                      mean_n2 = mean(valid_n2)
                      median_n2 = median(valid_n2)
                      
                      # Count how many times N² < threshold
                      low_n2_count = sum(valid_n2 .< N2_threshold)
                      low_n2_percent = 100 * low_n2_count / length(valid_n2)
                      
                      # If minimum N² is below threshold, record it
                      if min_n2 < N2_threshold
                          push!(problematic_points, (
                              i = i_global,
                              j = j_global,
                              lon = lon[i_global],
                              lat = lat[j_global],
                              min_n2 = min_n2,
                              mean_n2 = mean_n2,
                              median_n2 = median_n2,
                              low_n2_count = low_n2_count,
                              low_n2_percent = low_n2_percent,
                              total_points = length(valid_n2),
                              tile = suffix
                          ))
                      end
                  end
              end
          end
      end
  end
end


# ==========================================================
# ====== PRINT RESULTS =====================================
# ==========================================================


println("\n" * "="^70)
println("RESULTS: POINTS WITH MIN N² < $(N2_threshold)")
println("="^70)
println("\nTotal points found: $(length(problematic_points))")


if !isempty(problematic_points)
    
    # Sort by minimum N² (most problematic first)
    sort!(problematic_points, by = x -> x.min_n2)
    
    println("\n" * "="^70)
    println("ALL PROBLEMATIC POINTS (sorted by min N²)")
    println("="^70)
    println("   i    j   Lon      Lat      Min N²      Mean N²     Median N²   %<$(N2_threshold)  Tile")
    println("-"^100)
    
    for pt in problematic_points
        @printf("%4d %4d  %7.3f  %7.3f  %.3e  %.3e  %.3e  %6.1f%%  %s\n",
                pt.i, pt.j, pt.lon, pt.lat, 
                pt.min_n2, pt.mean_n2, pt.median_n2,
                pt.low_n2_percent, pt.tile)
    end
    
    # Summary statistics
    println("\n" * "="^70)
    println("SUMMARY STATISTICS")
    println("="^70)
    
    min_n2_values = [pt.min_n2 for pt in problematic_points]
    mean_n2_values = [pt.mean_n2 for pt in problematic_points]
    median_n2_values = [pt.median_n2 for pt in problematic_points]
    percent_low = [pt.low_n2_percent for pt in problematic_points]
    
    println(@sprintf("Min N² range: %.3e to %.3e", minimum(min_n2_values), maximum(min_n2_values)))
    println(@sprintf("Mean N² range: %.3e to %.3e", minimum(mean_n2_values), maximum(mean_n2_values)))
    println(@sprintf("Median N² range: %.3e to %.3e", minimum(median_n2_values), maximum(median_n2_values)))
    println(@sprintf("Percent with N² < threshold: %.1f%% to %.1f%%", 
             minimum(percent_low), maximum(percent_low)))
    
    # Create a convenient list for manual plotting
    println("\n" * "="^70)
    println("FOR MANUAL PLOTTING - COPY THESE (i,j) PAIRS:")
    println("="^70)
    println("# Format: (i, j, lon, lat)")
    
    for (idx, pt) in enumerate(problematic_points)
        println(@sprintf("Point %3d: i=%3d, j=%3d  # Lon=%.3f, Lat=%.3f, MinN2=%.2e", 
                        idx, pt.i, pt.j, pt.lon, pt.lat, pt.min_n2))
    end
    
    # Also print as Julia arrays for easy copy-paste
    println("\n" * "="^70)
    println("AS JULIA ARRAYS (for easy copy-paste):")
    println("="^70)
    
    i_list = [pt.i for pt in problematic_points]
    j_list = [pt.j for pt in problematic_points]
    lon_list = [pt.lon for pt in problematic_points]
    lat_list = [pt.lat for pt in problematic_points]
    
    println("i_points = ", i_list)
    println("\nj_points = ", j_list)
    println("\nlon_points = ", lon_list)
    println("\nlat_points = ", lat_list)
    
    # Group by severity
    println("\n" * "="^70)
    println("GROUPED BY SEVERITY")
    println("="^70)
    
    extreme = filter(pt -> pt.low_n2_percent > 50, problematic_points)
    moderate = filter(pt -> 10 < pt.low_n2_percent <= 50, problematic_points)
    occasional = filter(pt -> pt.low_n2_percent <= 10, problematic_points)
    
    println(@sprintf("Extreme (>50%% of time with low N²): %d points", length(extreme)))
    if !isempty(extreme)
        for pt in extreme
            @printf("  i=%3d, j=%3d (%.3f°, %.3f°) - %.1f%% low N²\n",
                    pt.i, pt.j, pt.lon, pt.lat, pt.low_n2_percent)
        end
    end
    
    println(@sprintf("\nModerate (10-50%% of time with low N²): %d points", length(moderate)))
    if !isempty(moderate)
        for pt in moderate[1:min(10, end)]
            @printf("  i=%3d, j=%3d (%.3f°, %.3f°) - %.1f%% low N²\n",
                    pt.i, pt.j, pt.lon, pt.lat, pt.low_n2_percent)
        end
        if length(moderate) > 10
            println("  ... and $(length(moderate)-10) more")
        end
    end
    
    println(@sprintf("\nOccasional (<10%% of time with low N²): %d points", length(occasional)))
    
else
    println("\nNo points found with N² < $(N2_threshold)!")
end


println("\n" * "="^70)
println("Analysis complete!")
println("="^70)




