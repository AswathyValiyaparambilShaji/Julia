using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML
using CairoMakie


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter


config_file = get(ENV, "JULIA_CONFIG",
                joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)


base  = cfg["base_path"]
base2 = cfg["base_path2"]


# --- Domain & grid ---
NX, NY = 288, 468
minlat, maxlat = 24.0, 31.91
minlon, maxlon = 193.0, 199.0
lat = range(minlat, maxlat, length=NY)
lon = range(minlon, maxlon, length=NX)


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


# --- Thickness & constants ---
thk = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)


rho0 = 999.8


# ==========================================================
# ========== DIAGNOSTIC: FIND PROBLEMATIC N2 VALUES ========
# ==========================================================


println("="^70)
println("DIAGNOSTIC: Finding N2 values causing extreme APE")
println("="^70)


all_N2_values = Float64[]
all_b_values = Float64[]
all_APE_values = Float64[]
extreme_cases = []  # Store (N2, b, APE, location info)


for xn in cfg["xn_start"]:cfg["xn_end"]
  for yn in cfg["yn_start"]:cfg["yn_end"]


      suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
      println("\nProcessing tile $suffix...")


      # --- Read N2 (3-day averaged) ---
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


      # --- Read hFacC ---
      hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"),
                       (nx, ny, nz))


      # --- Read buoyancy ---
      b = Float64.(open(joinpath(base2, "b", "b_t_sm_$suffix.bin"), "r") do io
          raw = read(io, nx * ny * nz * nt * sizeof(Float32))
          reshape(reinterpret(Float32, raw), nx, ny, nz, nt)
      end)


      # --- Compute APE and collect data ---
      for t in 1:nt_avg
          n2_val = N2_center[:, :, :, t]
        
          tstart = (t - 1) * ts + 1
          tend   = (t - 1) * ts + ts
        
          for tt in tstart:tend
              b_tt = b[:, :, :, tt]
              ape_tt = 0.5 .* rho0 .* (b_tt.^2 ./ n2_val)
              
              # Collect all finite values
              for i in 1:nx, j in 1:ny, k in 1:nz
                  if isfinite(n2_val[i,j,k]) && isfinite(b_tt[i,j,k]) && isfinite(ape_tt[i,j,k])
                      push!(all_N2_values, n2_val[i,j,k])
                      push!(all_b_values, b_tt[i,j,k])
                      push!(all_APE_values, ape_tt[i,j,k])
                      
                      # Flag extreme APE values
                      if ape_tt[i,j,k] > 1e6  # Adjust threshold as needed
                          # Calculate global position
                          global_i = (xn - 1) * tx + i - buf
                          global_j = (yn - 1) * ty + j - buf
                          
                          if 1 <= global_i <= NX && 1 <= global_j <= NY
                              push!(extreme_cases, (
                                  N2=n2_val[i,j,k],
                                  b=b_tt[i,j,k],
                                  APE=ape_tt[i,j,k],
                                  tile=suffix,
                                  i=i, j=j, k=k,
                                  lon=lon[global_i],
                                  lat=lat[global_j],
                                  depth=k
                              ))
                          end
                      end
                  end
              end
          end
      end
  end
end


# ==========================================================
# =================== ANALYSIS & REPORTING =================
# ==========================================================


println("\n" * "="^70)
println("ANALYSIS RESULTS")
println("="^70)


# Overall statistics
println("\nOverall Statistics:")
println("  Total data points: ", length(all_N2_values))
println("  N2 range: ", extrema(all_N2_values))
println("  b range: ", extrema(all_b_values))
println("  APE range: ", extrema(all_APE_values))


# N2 percentiles
n2_percentiles = [0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
println("\nN2 Percentiles:")
for p in n2_percentiles
    val = quantile(all_N2_values, p/100)
    println(@sprintf("  %5.1f%%: %.6e", p, val))
end


# Find correlation between low N2 and high APE
println("\nAPE statistics for different N2 ranges:")
n2_thresholds = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
for thresh in n2_thresholds
    mask = all_N2_values .< thresh
    if sum(mask) > 0
        ape_subset = all_APE_values[mask]
        println(@sprintf("  N2 < %.0e: %d points, APE mean=%.2e, max=%.2e", 
                thresh, sum(mask), mean(ape_subset), maximum(ape_subset)))
    end
end


# Report extreme cases
println("\n" * "="^70)
println("TOP 50 EXTREME APE VALUES AND THEIR N2")
println("="^70)
sort!(extreme_cases, by = x -> x.APE, rev=true)
println("   Tile      i   j   k    Lon      Lat     Depth      N2          b           APE")
println("-"^100)
for case in extreme_cases[1:min(50, end)]
    println(@sprintf("%s  %3d %3d %3d  %7.3f  %7.3f   %3d   %.3e  %.3e  %.3e",
            case.tile, case.i, case.j, case.k, case.lon, case.lat, case.depth,
            case.N2, case.b, case.APE))
end


# ==========================================================
# =================== VISUALIZATIONS =======================
# ==========================================================


fig = Figure(size=(1400, 1000))


# 1. N2 histogram (log scale)
ax1 = Axis(fig[1, 1], 
          xlabel="N2 [s⁻²]", 
          ylabel="Count",
          title="N2 Distribution (log scale)",
          xscale=log10)
hist!(ax1, all_N2_values[all_N2_values .> 0], bins=100)


# 2. APE vs N2 scatter (log-log)
ax2 = Axis(fig[1, 2],
          xlabel="N2 [s⁻²]",
          ylabel="APE [J/m³]",
          title="APE vs N2",
          xscale=log10,
          yscale=log10)
scatter!(ax2, all_N2_values, all_APE_values, 
         markersize=2, alpha=0.1, color=:blue)


# 3. APE histogram (log scale)
ax3 = Axis(fig[2, 1],
          xlabel="APE [J/m³]",
          ylabel="Count",
          title="APE Distribution (log scale)",
          xscale=log10)
hist!(ax3, all_APE_values[all_APE_values .> 0], bins=100)


# 4. b² vs N2 (to see the direct relationship)
b_squared = all_b_values.^2
ax4 = Axis(fig[2, 2],
          xlabel="N2 [s⁻²]",
          ylabel="b² [s⁻⁴]",
          title="b² vs N2",
          xscale=log10,
          yscale=log10)
scatter!(ax4, all_N2_values, b_squared,
         markersize=2, alpha=0.1, color=:red)


display(fig)


# ==========================================================
# =============== SUGGEST THRESHOLD ========================
# ==========================================================


println("\n" * "="^70)
println("THRESHOLD SUGGESTIONS")
println("="^70)


# Find N2 value where APE becomes unreasonably large
ape_threshold = 1e5  # You can adjust this
problematic_N2 = all_N2_values[all_APE_values .> ape_threshold]
if !isempty(problematic_N2)
    suggested_threshold = maximum(problematic_N2)
    println(@sprintf("\nTo limit APE < %.0e, consider filtering N2 < %.3e", 
            ape_threshold, suggested_threshold))
    println(@sprintf("This would affect %.2f%% of data points",
            100 * sum(all_N2_values .< suggested_threshold) / length(all_N2_values)))
else
    println("\nNo problematic N2 values found for APE threshold of ", ape_threshold)
end

