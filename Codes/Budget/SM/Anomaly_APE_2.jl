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


rho0 = 999.8


APE_full = fill(NaN, NX, NY)
MIN_N2_full = fill(NaN, NX, NY)  # Track minimum N2 at each point


# ==========================================================
# ====== BUILD APE MAP AND TRACK MINIMUM N2 ================
# ==========================================================


println("Building APE map and tracking minimum N2 values...")


for xn in cfg["xn_start"]:cfg["xn_end"]
  for yn in cfg["yn_start"]:cfg["yn_end"]


      suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)


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


      hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"),
                       (nx, ny, nz))


      DRFfull = DRF3d .* hFacC
      DRFfull[hFacC .== 0] .= 0


      APE = Float64.(open(joinpath(base2, "APE", "APE_t_sm_$suffix.bin"), "r") do io
          nbytes = nx * ny * nz * nt * sizeof(Float32)
          reshape(reinterpret(Float32, read(io, nbytes)),
                  nx, ny, nz, nt)
      end)


      # Time average APE ignoring NaN
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


      # Depth integrate APE
      ape = zeros(Float64, nx, ny)
      min_n2 = fill(Inf, nx, ny)  # Track minimum N2 in water column
      
      for i in 1:nx, j in 1:ny
          weighted = aped[i,j,:] .* DRF3d[i,j,:]
          ape[i,j] = sum(weighted[.!isnan.(weighted)])
          
          # Find minimum N2 in the water column for this location
          n2_column = N2_center[i, j, :, :]
          valid_n2 = filter(isfinite, n2_column)
          if !isempty(valid_n2)
              min_n2[i,j] = minimum(valid_n2)
          end
      end


      xs = (xn - 1) * tx + 1
      xe = xs + tx - 1
      ys = (yn - 1) * ty + 1
      ye = ys + ty - 1


      ape_interior = ape[buf+1:nx-buf, buf+1:ny-buf]
      min_n2_interior = min_n2[buf+1:nx-buf, buf+1:ny-buf]


      APE_full[xs:xe, ys:ye] .= ape_interior
      MIN_N2_full[xs:xe, ys:ye] .= min_n2_interior


      println("Completed tile $suffix")
  end
end


println("\nAPE_full range: $(minimum(skipmissing(APE_full))) to $(maximum(skipmissing(APE_full)))")
println("MIN_N2_full range: $(minimum(skipmissing(MIN_N2_full))) to $(maximum(skipmissing(MIN_N2_full)))")


# ==========================================================
# ====== ANALYZE CORRELATION BETWEEN LOW N2 AND APE ========
# ==========================================================


println("\n" * "="^70)
println("ANALYZING LOW N2 VALUES AND THEIR IMPACT ON APE")
println("="^70)


# Find points with different N2 thresholds
n2_thresholds = [1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4]


low_n2_points = Dict()


for thresh in n2_thresholds
    mask = MIN_N2_full .< thresh
    n_points = sum(mask)
    
    if n_points > 0
        ape_at_low_n2 = APE_full[mask]
        ape_mean = mean(filter(isfinite, ape_at_low_n2))
        ape_max = maximum(filter(isfinite, ape_at_low_n2))
        
        println(@sprintf("\nN2 < %.0e: %d points", thresh, n_points))
        println(@sprintf("  APE mean: %.2f J/m²", ape_mean))
        println(@sprintf("  APE max:  %.2f J/m²", ape_max))
        
        # Store points for this threshold
        low_n2_points[thresh] = findall(mask)
    end
end


# ==========================================================
# ====== IDENTIFY SPECIFIC PROBLEMATIC POINTS ==============
# ==========================================================


println("\n" * "="^70)
println("POINTS WITH N2 < 1e-6 (YOUR OBSERVED THRESHOLD)")
println("="^70)


thresh_target = 1e-6
mask_target = MIN_N2_full .< thresh_target
problematic_points = findall(mask_target)


println("\nFound $(length(problematic_points)) points with N2 < $(thresh_target)")


if length(problematic_points) > 0
    # Create list with details
    point_details = []
    for idx in problematic_points
        i, j = Tuple(idx)
        push!(point_details, (
            i=i, j=j,
            lon=lon[i], lat=lat[j],
            APE=APE_full[i,j],
            min_N2=MIN_N2_full[i,j]
        ))
    end
    
    # Sort by APE value
    sort!(point_details, by = x -> x.APE, rev=true)
    
    println("\nTop 50 points by APE value:")
    println("   i    j     Lon      Lat        APE      Min N2")
    println("-"^70)
    for pt in point_details[1:min(50, end)]
        @printf("%4d %4d  %7.3f  %7.3f  %10.1f  %.3e\n",
                pt.i, pt.j, pt.lon, pt.lat, pt.APE, pt.min_N2)
    end
end


# ==========================================================
# ====== VISUALIZATION =====================================
# ==========================================================


fig = Figure(size=(1800, 800))


# Plot 1: APE map
ax1 = Axis(fig[1, 1],
         title="Depth-Integrated Time-Averaged APE",
         xlabel="Longitude [°]",
         ylabel="Latitude [°]")


hm1 = CairoMakie.heatmap!(ax1, lon, lat, APE_full;
                        interpolate=false,
                        colormap=:jet,
                        colorrange=(0, 5000))


Colorbar(fig[1, 2], hm1, label="APE [J/m²]")


# Mark points with N2 < 1e-6
if !isempty(problematic_points)
    prob_i = [p[1] for p in problematic_points]
    prob_j = [p[2] for p in problematic_points]
    prob_lons = lon[prob_i]
    prob_lats = lat[prob_j]
    
    scatter!(ax1, prob_lons, prob_lats,
            color=:black, marker='x', markersize=8,
            strokewidth=2, label="N2 < 1e-6")
    
    axislegend(ax1, position=:lt)
end


# Plot 2: Minimum N2 map (log scale)
ax2 = Axis(fig[1, 3],
         title="Minimum N2 in Water Column",
         xlabel="Longitude [°]",
         ylabel="Latitude [°]")


# Replace Inf with NaN for plotting
MIN_N2_plot = copy(MIN_N2_full)
MIN_N2_plot[isinf.(MIN_N2_plot)] .= NaN


# Take log10 for better visualization
MIN_N2_log = log10.(MIN_N2_plot)


hm2 = CairoMakie.heatmap!(ax2, lon, lat, MIN_N2_log;
                        interpolate=false,
                        colormap=:viridis)


Colorbar(fig[1, 4], hm2, label="log10(Min N2) [s⁻²]")


# Mark same points
if !isempty(problematic_points)
    scatter!(ax2, prob_lons, prob_lats,
            color=:red, marker='x', markersize=8,
            strokewidth=2, label="N2 < 1e-6")
    
    axislegend(ax2, position=:lt)
end


display(fig)


# ==========================================================
# ====== RECOMMENDATION ====================================
# ==========================================================


println("\n" * "="^70)
println("RECOMMENDATION")
println("="^70)


if !isempty(problematic_points)
    n_problematic = length(problematic_points)
    total_points = sum(isfinite.(APE_full))
    percentage = 100 * n_problematic / total_points
    
    println(@sprintf("\nYou have %d points (%.2f%%) with N2 < 1e-6", n_problematic, percentage))
    println("\nThese are the 'dots' you see in your figure.")
    println("\nOptions:")
    println("  1. Use N2 threshold of 1e-6 in your filtering code")
    println("  2. Use spatial smoothing to blend these points with neighbors")
    println("  3. Accept them as-is if they represent real physical features")
    
    # Calculate what threshold would catch most of them
    all_min_n2 = filter(isfinite, MIN_N2_full)
    percentiles = [99, 95, 90, 80, 70, 60, 50]
    println("\nN2 percentile thresholds:")
    for p in percentiles
        thresh_p = quantile(all_min_n2, (100-p)/100)
        println(@sprintf("  Bottom %2d%%: N2 < %.3e", 100-p, thresh_p))
    end
end




