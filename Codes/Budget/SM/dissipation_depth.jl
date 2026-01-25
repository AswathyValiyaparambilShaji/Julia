using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML, CairoMakie
include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin


# Load configuration
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base = cfg["base_path"]
base2 = cfg["base_path2"]


# --- Domain & grid ---
NX, NY = 288, 468
minlat, maxlat = 24.0, 31.91
minlon, maxlon = 193.0, 199.0
lat = range(minlat, maxlat, length=NY)
lon = range(minlon, maxlon, length=NX)


# --- Tile parameters ---
buf = 3
tx, ty = 47, 66
nx = tx + 2*buf
ny = ty + 2*buf


# --- Vertical levels ---
nz = 88


kz = 1
dt = 25
dto = 144
Tts = 366192
nt = div(Tts, dto)


# --- Thickness & constants ---
thk = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)
g = 9.8


# Initialize global arrays
Conv = zeros(NX, NY)
FDiv = zeros(NX, NY)
U_KE_full = zeros(NX, NY)
U_PE_full = zeros(NX, NY)
SP_H_full = zeros(NX, NY)
SP_V_full = zeros(NX, NY)
BP_full = zeros(NX, NY)
∇H = zeros(NX, NY)
FH = zeros(NX, NY)
RAC = zeros(NX, NY)  # Grid cell area
ET_full = zeros(NX, NY)


println("Loading energy budget terms...")


# ==========================================================
# ============ LOAD ALL TERMS ==============================
# ==========================================================


for xn in cfg["xn_start"]:cfg["xn_end"]
   for yn in cfg["yn_start"]:cfg["yn_end"]
       suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
       suffix2 = @sprintf("%02dx%02d_%d", xn, yn, buf-2)
       hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))


       DRFfull = hFacC .* DRF3d
       z = cumsum(DRFfull, dims=3)
       depth = sum(DRFfull, dims=3)
       DRFfull[hFacC .== 0] .= 0.0
       
       # Convert to negative depth (oceanographic convention)
       depth = -depth


       # --- Read Flux Divergence ---
       fxD = Float64.(open(joinpath(base2, "FDiv", "FDiv_$(suffix2).bin"), "r") do io
           nbytes = (nx-2) * (ny-2) * sizeof(Float32)
           raw_bytes = read(io, nbytes)
           raw_data = reinterpret(Float32, raw_bytes)
           reshape(raw_data, nx-2, ny-2)
       end)
       
       # --- Read Conversion ---
       C = Float64.(open(joinpath(base2, "Conv", "Conv_$(suffix2).bin"), "r") do io
           nbytes = (nx-2) * (ny-2) * sizeof(Float32)
           raw_bytes = read(io, nbytes)
           raw_data = reinterpret(Float32, raw_bytes)
           reshape(raw_data, nx-2, ny-2)
       end)
       
       # --- Read KE Advection ---
       u_ke_mean = Float64.(open(joinpath(base2, "U_KE", "u_ke_mean_$suffix.bin"), "r") do io
           nbytes = nx * ny * sizeof(Float32)
           reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
       end)
       
       # --- Read PE Advection ---
       u_pe_mean = Float64.(open(joinpath(base2, "U_PE", "u_pe_mean_$suffix.bin"), "r") do io
           nbytes = nx * ny * sizeof(Float32)
           reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
       end)
       
       # --- Read Shear Production ---
       sp_h_mean = Float64.(open(joinpath(base2, "SP_H", "sp_h_mean_$suffix.bin"), "r") do io
           nbytes = nx * ny * sizeof(Float32)
           reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
       end)


       sp_v_mean = Float64.(open(joinpath(base2, "SP_V", "sp_v_mean_$suffix.bin"), "r") do io
           nbytes = nx * ny * sizeof(Float32)
           reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
       end)


       # --- Read Buoyancy Production ---
       bp_mean = Float64.(open(joinpath(base2, "BP", "bp_mean_$suffix.bin"), "r") do io
           nbytes = nx * ny * sizeof(Float32)
           reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
       end)
       # Read time-averaged energy tendency
       te_mean = Float64.(open(joinpath(base2, "TE_t", "te_t_mean_$suffix.bin"), "r") do io
           nbytes = nx * ny * sizeof(Float32)
           reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
       end)

       dx = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
       dy = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))
       
       # Calculate grid cell area
       rac = dx .* dy


       H = depth


       # Horizontal gradients for roughness
       dHdx = zeros(nx, ny)
       dHdx[2:end-1, :] .= (H[3:end, :] .- H[1:end-2, :]) ./ (dx[2:end-1, :] .+ dx[3:end, :])


       dHdy = zeros(nx, ny)
       dHdy[:, 2:end-1] .= (H[:, 3:end] .- H[:, 1:end-2]) ./ (dy[:, 2:end-1] .+ dy[:, 3:end])


       # Gradient magnitude = topographic slope (roughness measure)
       gh = sqrt.(dHdx.^2 .+ dHdy.^2)


       # Calculate tile positions in global grid
       xs = (xn - 1) * tx + 1
       xe = xs + tx + (2 * buf) - 1
       ys = (yn - 1) * ty + 1
       ye = ys + ty + (2 * buf) - 1
       
       # Update global arrays (remove buffer zones)
       Conv[xs+2:xe-2, ys+2:ye-2] .= C[2:end-1, 2:end-1]
       FDiv[xs+2:xe-2, ys+2:ye-2] .= fxD[2:end-1, 2:end-1]
       
       # Extract interior regions
       u_ke_interior = u_ke_mean[buf:nx-buf+1, buf:ny-buf+1]
       u_pe_interior = u_pe_mean[buf:nx-buf+1, buf:ny-buf+1]
       sp_h_interior = sp_h_mean[buf:nx-buf+1, buf:ny-buf+1]
       sp_v_interior = sp_v_mean[buf:nx-buf+1, buf:ny-buf+1]
       bp_interior = bp_mean[buf:nx-buf+1, buf:ny-buf+1]
       gh_interior = gh[buf:nx-buf+1, buf:ny-buf+1]
       H_interior = H[buf:nx-buf+1, buf:ny-buf+1]
       rac_interior = rac[buf:nx-buf+1, buf:ny-buf+1]
       te_interior = te_mean[buf:nx-buf+1, buf:ny-buf+1]
       
       # Assign to global arrays
       U_KE_full[xs+2:xe-2, ys+2:ye-2] .= u_ke_interior
       U_PE_full[xs+2:xe-2, ys+2:ye-2] .= u_pe_interior
       SP_H_full[xs+2:xe-2, ys+2:ye-2] .= sp_h_interior
       SP_V_full[xs+2:xe-2, ys+2:ye-2] .= sp_v_interior
       BP_full[xs+2:xe-2, ys+2:ye-2] .= bp_interior
       ∇H[xs+2:xe-2, ys+2:ye-2] .= gh_interior
       FH[xs+2:xe-2, ys+2:ye-2] .= H_interior
       RAC[xs+2:xe-2, ys+2:ye-2] .= rac_interior
       ET_full[xs+2:xe-2, ys+2:ye-2] .= te_interior

       println("Completed tile $suffix")
   end
end


# Total energy fluxes (Flux Divergence + Advective fluxes)
TotalFlux = FDiv .+ U_KE_full .+ U_PE_full
MF = U_KE_full .+ U_PE_full .+ SP_H_full.+SP_V_full .+ BP_full
A = U_KE_full .+ U_PE_full
PS = SP_H_full.+SP_V_full

Residual = (Conv .- TotalFlux .+ SP_H_full.+SP_V_full .+ BP_full.+ET_full)


# ==========================================================
# ============ BINNING BY DEPTH AND ROUGHNESS ==============
# ==========================================================


println("\nCreating bins...")


# Get actual depth range (now negative)
ocean_mask = FH .< 0
max_depth = minimum(FH[ocean_mask])  # Most negative (deepest)
min_depth = maximum(FH[ocean_mask])  # Least negative (shallowest)


println("Depth range: $(max_depth) to $(min_depth) m")


# Define depth bins: starting at -5500 m with 750 m bin size
depth_bin_edges = collect(-6000:1000:0)
n_depth_bins = length(depth_bin_edges) - 1


println("Depth bins: $depth_bin_edges")


# Get roughness range
max_rough = maximum(∇H[ocean_mask])
min_rough = minimum(∇H[ocean_mask])


println("Roughness range: $(min_rough) to $(max_rough)")


# Define roughness bins: only 5 bins
roughness_bin_edges = collect(range(min_rough, max_rough, length=6))  # 6 edges = 5 bins
n_rough_bins = length(roughness_bin_edges) - 1


println("Roughness bins: $roughness_bin_edges")


# Initialize storage for area-weighted averaging
residual_by_depth = [Float64[] for _ in 1:n_depth_bins]
area_by_depth = [Float64[] for _ in 1:n_depth_bins]


residual_by_roughness = [Float64[] for _ in 1:n_rough_bins]
area_by_roughness = [Float64[] for _ in 1:n_rough_bins]


# Bin residuals by depth with area
println("\nBinning by depth...")
for i in 1:NX, j in 1:NY
   if ocean_mask[i,j]
       for k in 1:n_depth_bins
           if depth_bin_edges[k] <= FH[i,j] < depth_bin_edges[k+1]
               push!(residual_by_depth[k], Residual[i,j])
               push!(area_by_depth[k], RAC[i,j])
               break
           end
       end
   end
end


# Bin residuals by roughness with area
println("Binning by roughness...")
for i in 1:NX, j in 1:NY
   if ocean_mask[i,j]
       for k in 1:n_rough_bins
           if roughness_bin_edges[k] <= ∇H[i,j] < roughness_bin_edges[k+1]
               push!(residual_by_roughness[k], Residual[i,j])
               push!(area_by_roughness[k], RAC[i,j])
               break
           end
       end
   end
end


# Calculate area-weighted averages
println("\n" * "="^60)
println("AREA-WEIGHTED STATISTICS")
println("="^60)


# Depth bins
depth_bin_centers = [(depth_bin_edges[k] + depth_bin_edges[k+1])/2 for k in 1:n_depth_bins]
residual_AA_depth = zeros(n_depth_bins)
total_area_depth = zeros(n_depth_bins)


println("\nDepth bin statistics:")
for k in 1:n_depth_bins
   if !isempty(residual_by_depth[k])
       # Area-weighted average
       total_area = sum(area_by_depth[k])
       residual_AA_depth[k] = sum(residual_by_depth[k] .* area_by_depth[k]) / total_area
       total_area_depth[k] = total_area
       
       # Simple statistics
       mean_res = mean(residual_by_depth[k])
       std_res = std(residual_by_depth[k])
       n_points = length(residual_by_depth[k])
       
       println(@sprintf("  %.0f to %.0f m:", depth_bin_edges[k], depth_bin_edges[k+1]))
       println(@sprintf("    Area-weighted mean: %.2e W/m²", residual_AA_depth[k]))
       println(@sprintf("    Simple mean: %.2e W/m²", mean_res))
       println(@sprintf("    Std dev: %.2e W/m²", std_res))
       println(@sprintf("    Total area: %.2e m²", total_area))
       println(@sprintf("    N points: %d", n_points))
   end
end


# Roughness bins
roughness_bin_centers = [(roughness_bin_edges[k] + roughness_bin_edges[k+1])/2 for k in 1:n_rough_bins]
residual_AA_roughness = zeros(n_rough_bins)
total_area_roughness = zeros(n_rough_bins)


println("\nRoughness bin statistics:")
for k in 1:n_rough_bins
   if !isempty(residual_by_roughness[k])
       # Area-weighted average
       total_area = sum(area_by_roughness[k])
       residual_AA_roughness[k] = sum(residual_by_roughness[k] .* area_by_roughness[k]) / total_area
       total_area_roughness[k] = total_area
       
       # Simple statistics
       mean_res = mean(residual_by_roughness[k])
       std_res = std(residual_by_roughness[k])
       n_points = length(residual_by_roughness[k])
       
       println(@sprintf("  %.2e to %.2e:", roughness_bin_edges[k], roughness_bin_edges[k+1]))
       println(@sprintf("    Area-weighted mean: %.2e W/m²", residual_AA_roughness[k]))
       println(@sprintf("    Simple mean: %.2e W/m²", mean_res))
       println(@sprintf("    Std dev: %.2e W/m²", std_res))
       println(@sprintf("    Total area: %.2e m²", total_area))
       println(@sprintf("    N points: %d", n_points))
   end
end


# ==========================================================
# ==================== VISUALIZATIONS ======================
# ==========================================================


println("\nCreating visualizations...")


# Mask land for plotting
FH_plot = copy(FH)
FH_plot[.!ocean_mask] .= NaN


∇H_plot = copy(∇H)
∇H_plot[.!ocean_mask] .= NaN


# --- Figure 1: Bathymetry ---
fig_bathy = Figure(resolution=(1000, 800), fontsize=16)
ax_bathy = Axis(fig_bathy[1, 1],
   xlabel="Longitude (°E)",
   ylabel="Latitude (°N)",
   title="Bathymetry")


hm_bathy = heatmap!(ax_bathy, lon, lat, FH_plot,
   colormap=:deep,
   nan_color=:tan)
Colorbar(fig_bathy[1, 2], hm_bathy, label="Depth (m)")


display(fig_bathy)


# --- Figure 2: Seafloor Roughness ---
fig_rough = Figure(resolution=(1000, 800), fontsize=16)
ax_rough = Axis(fig_rough[1, 1],
   xlabel="Longitude (°)",
   ylabel="Latitude (°)",
   title="Seafloor Roughness")


hm_rough = heatmap!(ax_rough, lon, lat, ∇H_plot,
   colormap=:thermal,
   nan_color=:tan)
Colorbar(fig_rough[1, 2], hm_rough, label="|∇H|")


display(fig_rough)


# --- Figure 3: Bar Plots (Both together) ---
fig_bars = Figure(resolution=(1400, 600), fontsize=14)


# Panel 1: Depth vs Residual
ax_depth = Axis(fig_bars[1, 1],
   xlabel="Depth Range (m)",
   ylabel="Area-weighted Residual (W/m²)",
   title="Residual vs Depth")
   #,   xticklabelrotation=π/45)


# Create x-axis labels for depth bins
depth_labels = [@sprintf("%.0f to\n%.0f", depth_bin_edges[k], depth_bin_edges[k+1])
               for k in 1:n_depth_bins]


# Filter out empty bins
valid_depth = .!iszero.(residual_AA_depth)
x_depth = (1:n_depth_bins)[valid_depth]
y_depth = residual_AA_depth[valid_depth]
labels_depth = depth_labels[valid_depth]


# Color bars by sign
colors_depth = [val >= 0 ? :red : :blue for val in y_depth]


barplot!(ax_depth, x_depth, y_depth,
   color=colors_depth,
   strokecolor=:black,
   strokewidth=1,
   width = 0.5)
ax_depth.xticks = (x_depth, labels_depth)
hlines!(ax_depth, [0], color=:black, linewidth=1.5, linestyle=:dash)


# Panel 2: Roughness vs Residual
ax_rough_bar = Axis(fig_bars[1, 2],
   xlabel="Roughness Range (×10⁻³)",
   ylabel="Area-weighted Residual (W/m²)",
   title="Residual vs Roughness")#,    xticklabelrotation=π/4)


# Scale roughness bin edges for display
rs = 1000  # roughness scale factor
rbe_s = roughness_bin_edges .* rs  # scaled bin edges


# Create x-axis labels for roughness bins (scaled to whole numbers)
rl_s = [@sprintf("%d to\n%d", 
               round(Int, rbe_s[k]), 
               round(Int, rbe_s[k+1]))
               for k in 1:n_rough_bins]


# Filter out empty bins
valid_rough = .!iszero.(residual_AA_roughness)
x_rough = (1:n_rough_bins)[valid_rough]
y_rough = residual_AA_roughness[valid_rough]
labels_rough = rl_s[valid_rough]


# Color bars by sign
colors_rough = [val >= 0 ? :red : :blue for val in y_rough]


barplot!(ax_rough_bar, x_rough, y_rough,
   color=colors_rough,
   strokecolor=:black,
   strokewidth=1,
   width = 0.5)
ax_rough_bar.xticks = (x_rough, labels_rough)
hlines!(ax_rough_bar, [0], color=:black, linewidth=1.5, linestyle=:dash)


display(fig_bars)


# Save figures
FIGDIR = cfg["fig_base"]
save(joinpath(FIGDIR, "bathymetry.png"), fig_bathy, px_per_unit=2)
save(joinpath(FIGDIR, "seafloor_roughness.png"), fig_rough, px_per_unit=2)
save(joinpath(FIGDIR, "residual_bar_plots.png"), fig_bars, px_per_unit=2)


println("\nAll visualizations complete!")
println("Figures saved to: $FIGDIR")







