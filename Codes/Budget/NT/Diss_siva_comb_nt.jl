using Printf, Dates, MAT, FilePathsBase, TOML, NCDatasets, CairoMakie, Statistics


# Include FluxUtils.jl
include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin


# Read the configuration file for paths
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base = cfg["base_path"]
base2 = cfg["base_path_nt"]


# --- Grid parameters ---
NX, NY = 288, 468
nz = 88


# Domain
minlat, maxlat = 24.0, 31.91
minlon, maxlon = 193.0, 199.0
lat = range(minlat, maxlat, length=NY)
lon = range(minlon, maxlon, length=NX)


# Tiling parameters
buf = 3
tx, ty = 47, 66
nx = tx + 2 * buf
ny = ty + 2 * buf
dto = 144
Tts = 366192
nt  = div(Tts, dto)
nt3 = div(nt, 3*24)
nt_chunk = 72
n_chunks = div(nt, nt_chunk)
rho0 = 1027.5


# Depth threshold (in meters) -- used for the zonal-average panel
DEPTH_THRESHOLD = 3900.0


ring_steps = nt_chunk
t_safe_start = ring_steps + 1              # first valid step (1801)
t_safe_end   = nt - ring_steps             # last  valid step (nt-1800)


# Safe 3-day chunks: only keep chunks that fall entirely within the safe range
safe_chunks = [c for c in 1:n_chunks
              if (c-1)*nt_chunk + 1 >= t_safe_start &&
                 c*nt_chunk          <= t_safe_end]


# --- Thickness & constants ---
thk = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)


# ============================================================================
# PART 1: READ SIVA'S DISSIPATION FROM NetCDF
# ============================================================================
println("Reading Siva's dissipation from NetCDF...")


ds = NCDataset(joinpath(base, "Siva_Diss", "TotDiss_band1.nc"))
println(ds)


# Read F_band (z × y × x) and permute to (x × y × z)
F_band_nc = ds["F_band"][:, :, :]  # (88 × 467 × 287)
F_band = permutedims(F_band_nc, (3, 2, 1))  # Now (287 × 467 × 88)
println("F_band size: ", size(F_band))


close(ds)


# Pad F_band to match domain dimensions (keep NaN as is)
F_band_full = fill(NaN, NX, NY, nz)
F_band_full[1:287, 1:467, :] .= F_band


println("F_band_full size: ", size(F_band_full))


# ============================================================================
# PART 2: READ hFacC, DXC AND CALCULATE DEPTH-INTEGRATED SIVA DISSIPATION
# ============================================================================
println("\nReading hFacC/DXC and calculating depth-integrated Siva dissipation...")


hFacC_full = zeros(NX, NY, nz)
FH = zeros(NX, NY)
DXC = zeros(NX, NY)  # Zonal grid spacing (needed for the zonal-average panel)


for xn in cfg["xn_start"]:cfg["xn_end"]
   for yn in cfg["yn_start"]:cfg["yn_end"]
       suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
       println("Processing tile: $suffix")


       # Read hFacC
       hFacC = read_bin(joinpath(base, "hFacC", "hFacC_$suffix.bin"), (nx, ny, nz))


       # Calculate depth
       DRFfull = hFacC .* DRF3d
       depth = sum(DRFfull, dims=3)


       # Read grid cell spacing
       dx = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))


       # Calculate tile positions in global grid
       xs = (xn - 1) * tx + 1
       xe = xs + tx + (2 * buf) - 1
       ys = (yn - 1) * ty + 1
       ye = ys + ty + (2 * buf) - 1


       # Fill global arrays (remove buffer zones)
       hFacC_full[xs+2:xe-2, ys+2:ye-2, :] .= hFacC[buf:nx-buf+1, buf:ny-buf+1, :]
       FH[xs+2:xe-2, ys+2:ye-2] .= depth[buf:nx-buf+1, buf:ny-buf+1]
       DXC[xs+2:xe-2, ys+2:ye-2] .= dx[buf:nx-buf+1, buf:ny-buf+1]
   end
end


# Mask F_band with hFacC (set to 0 where hFacC == 0)
F_masked = copy(F_band_full)
F_masked[hFacC_full .== 0] .= 0.0
F_masked[isnan.(F_masked)] .= 0.0  # Handle any remaining NaNs


# Create 3D DRF array matching full domain
DRF3d_full = repeat(reshape(DRF, 1, 1, nz), NX, NY, 1)


# Vertical integration: sum(F * DRF * hFacC, dims=3)
Siva_Diss_integrated = dropdims(sum(F_masked .* DRF3d_full .* hFacC_full, dims=3), dims=3)


println("Siva dissipation integrated size: ", size(Siva_Diss_integrated))
println("Siva dissipation range: ", extrema(Siva_Diss_integrated))


# ============================================================================
# PART 3: CALCULATE ENERGY BUDGET DISSIPATION (RESIDUAL)
# ============================================================================
println("\nCalculating energy budget dissipation (residual)...")


# Initialize arrays for energy budget terms
Conv         = zeros(NX, NY)
FDiv         = zeros(NX, NY)
U_KE_full    = zeros(NX, NY)
U_PE_full    = zeros(NX, NY)
SP_H_full    = zeros(NX, NY)
SP_V_full    = zeros(NX, NY)
BP_full      = zeros(NX, NY)
ET_full      = zeros(NX, NY)
WPI_full     = zeros(NX, NY)


# Weekly window from date (kept for reference / potential sub-window analysis)
t_origin   = DateTime(2012, 3, 1, 0, 0, 0)
t_wk_start = DateTime(2012, 4, 22, 0, 0, 0)
t_wk_end   = DateTime(2012, 4, 28, 23, 0, 0)
wk_start   = Int(Dates.Hour(t_wk_start - t_origin).value) + 1
wk_end     = Int(Dates.Hour(t_wk_end   - t_origin).value) + 1


# Load energy budget data for all tiles
for xn in cfg["xn_start"]:cfg["xn_end"]
   for yn in cfg["yn_start"]:cfg["yn_end"]
       suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
       suffix2 = @sprintf("%02dx%02d_%d", xn, yn, buf-2)


       println("Loading energy budget for tile: $suffix")


       # --- Read Flux Divergence ---
       fxD = Float64.(open(joinpath(base2, "FDiv", "FDiv_nt_$suffix2.bin"), "r") do io
           nbytes = (nx-2) * (ny-2) * sizeof(Float32)
           raw_bytes = read(io, nbytes)
           raw_data = reinterpret(Float32, raw_bytes)
           reshape(raw_data, nx-2, ny-2)
       end)


       # --- Read Conversion ---
       C = Float64.(open(joinpath(base2, "Conv", "Conv_nt_$suffix2.bin"), "r") do io
           nbytes = (nx-2) * (ny-2) * sizeof(Float32)
           raw_bytes = read(io, nbytes)
           raw_data = reinterpret(Float32, raw_bytes)
           reshape(raw_data, nx-2, ny-2)
       end)


       # --- Read KE Advection ---
       u_ke_mean = Float64.(open(joinpath(base2, "U_KE", "u_ke_nt_$suffix.bin"), "r") do io
           nbytes = nx * ny * sizeof(Float32)
           reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
       end)


       # --- Read PE Advection ---
       u_pe_mean = Float64.(open(joinpath(base2, "U_PE", "u_pe_nt_$suffix.bin"), "r") do io
           nbytes = nx * ny * sizeof(Float32)
           reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
       end)


       # --- Read Shear Production ---
       sp_h_mean = Float64.(open(joinpath(base2, "SP_H", "sp_h_nt_$suffix.bin"), "r") do io
           nbytes = nx * ny * sizeof(Float32)
           reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
       end)


       # --- Read Energy Tendency ---
       te_mean = Float64.(open(joinpath(base2, "TE_t", "te_t_nt_$suffix.bin"), "r") do io
           nbytes = nx * ny * sizeof(Float32)
           reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
       end)


       # --- Read Vertical Shear Production ---
       sp_v_mean = Float64.(open(joinpath(base2, "SP_V", "sp_v_nt_$suffix.bin"), "r") do io
           nbytes = nx * ny * sizeof(Float32)
           reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
       end)


       # --- Read Buoyancy Production ---
       bp_mean = Float64.(open(joinpath(base2, "BP", "bp_nt_$suffix.bin"), "r") do io
           nbytes = nx * ny * sizeof(Float32)
           reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
       end)


       # --- Read Wind Power Input (with time dimension) ---
       wpi_tile = Float64.(open(joinpath(base2, "WindInput", "wpi_nt_$suffix.bin"), "r") do io
           nbytes = nx * ny * nt * sizeof(Float32)
           reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt)
       end)


       # Time average the WPI
       wpi_mean = mean(wpi_tile[:, :, t_safe_start:t_safe_end], dims=3)[:, :, 1]


       # --- Tile positions in global grid ---
       xs = (xn - 1) * tx + 1
       xe = xs + tx + (2 * buf) - 1
       ys = (yn - 1) * ty + 1
       ye = ys + ty + (2 * buf) - 1


       Conv[xs+2:xe-2, ys+2:ye-2] .= C[2:end-1, 2:end-1]
       FDiv[xs+2:xe-2, ys+2:ye-2] .= fxD[2:end-1, 2:end-1]


       U_KE_full[xs+2:xe-2,    ys+2:ye-2] .= u_ke_mean[buf:nx-buf+1, buf:ny-buf+1]
       U_PE_full[xs+2:xe-2,    ys+2:ye-2] .= u_pe_mean[buf:nx-buf+1, buf:ny-buf+1]
       SP_H_full[xs+2:xe-2,    ys+2:ye-2] .= sp_h_mean[buf:nx-buf+1, buf:ny-buf+1]
       SP_V_full[xs+2:xe-2,    ys+2:ye-2] .= sp_v_mean[buf:nx-buf+1, buf:ny-buf+1]
       BP_full[xs+2:xe-2,      ys+2:ye-2] .= bp_mean[buf:nx-buf+1,   buf:ny-buf+1]
       ET_full[xs+2:xe-2,      ys+2:ye-2] .= te_mean[buf:nx-buf+1,   buf:ny-buf+1]
       WPI_full[xs+2:xe-2,     ys+2:ye-2] .= wpi_mean[buf:nx-buf+1,  buf:ny-buf+1]


       println("Completed tile $suffix")
   end
end


println("\nCalculating derived terms...")


# Total energy fluxes (Flux Divergence + Advective fluxes)
TotalFlux = FDiv .+ U_KE_full .+ U_PE_full
MF        = U_KE_full .+ U_PE_full .+ SP_H_full .+ SP_V_full .+ BP_full
A         = U_KE_full .+ U_PE_full
PS        = SP_H_full .+ SP_V_full


# Residual dissipation -- G terms subtracted as energy lost from IT to NIW
Budget_Diss  = -(Conv .- TotalFlux .+ SP_H_full .+ SP_V_full .+ BP_full .+ WPI_full .- ET_full)


println("Budget dissipation range: ", extrema(Budget_Diss))


# ============================================================================
# PART 4: NORMALIZE FOR THE SPATIAL MAPS (panels a & b)
# ============================================================================
Siva_Diss_norm   = Siva_Diss_integrated ./ (FH) * 10^8
Budget_Diss_norm = (Budget_Diss ./ (rho0 .* FH)) * 10^8


# ============================================================================
# PART 5: DEPTH MASK + ZONAL AVERAGES (panel c)
# ============================================================================
println("\nCalculating zonal averages...")


# Create depth mask - TRUE where depth > 3900m
deep_mask = FH .> DEPTH_THRESHOLD


# Normalize dissipation fields (guarding against zero/invalid FH or DXC)
Siva_Diss_norm_zonal   = zeros(NX, NY)
Budget_Diss_norm_zonal = zeros(NX, NY)


valid_mask = (FH .> 0.0) .& (DXC .> 0.0)
Siva_Diss_norm_zonal[valid_mask]   = Siva_Diss_integrated[valid_mask] ./ FH[valid_mask]
Budget_Diss_norm_zonal[valid_mask] = Budget_Diss[valid_mask] ./ (rho0 .* FH[valid_mask])


# Compute zonal averages weighted by dx - ONLY for deep points
Siva_zonal = zeros(NY)
Budget_zonal = zeros(NY)


for j in 1:NY
   deep_points_at_lat = deep_mask[:, j]


   if sum(deep_points_at_lat) > 0
       total_dx = sum(DXC[deep_points_at_lat, j])
       Siva_zonal[j]   = sum(Siva_Diss_norm_zonal[deep_points_at_lat, j]   .* DXC[deep_points_at_lat, j]) / total_dx
       Budget_zonal[j] = sum(Budget_Diss_norm_zonal[deep_points_at_lat, j] .* DXC[deep_points_at_lat, j]) / total_dx
   else
       Siva_zonal[j] = NaN
       Budget_zonal[j] = NaN
   end
end


# Convert to 10^-8 W/kg
Siva_zonal_scaled   = Siva_zonal * 1e8
Budget_zonal_scaled = Budget_zonal * 1e8


# Smooth the Budget dissipation to reduce noise
function smooth_data(data, window=15)
   smoothed = copy(data)
   n = length(data)
   half_window = div(window, 2)


   for i in 1:n
       if isnan(data[i])
           continue
       end
       i_start = max(1, i - half_window)
       i_end = min(n, i + half_window)
       window_data = data[i_start:i_end]
       valid_data = filter(!isnan, window_data)
       if length(valid_data) > 0
           smoothed[i] = mean(valid_data)
       end
   end


   return smoothed
end


Budget_zonal_scaled_smooth = smooth_data(Budget_zonal_scaled, 15)


# Mask edges and latitudes beyond F_band data coverage
for j in 1:NY
   if j > 467 || j <= 2 || j >= NY-1
       Siva_zonal_scaled[j] = NaN
   end
end


# ============================================================================
# PART 6: COMBINED FIGURE -- (a) spectral map, (b) residual map, (c) zonal comparison
# ============================================================================
println("\nCreating combined comparison figure...")


FONT = "FreeSerif Bold"


fig = Figure(resolution = (1100, 600), figure_padding = (5, 5, 5, 5),
            fonts = (; regular = FONT))
crange = (-1.0, 1.0)
cmap = :bwr


# --- Panel (a): Siva Dissipation (spatial map) ---
ax1 = Axis(fig[1, 1],
   title="(a) Spectral Dissipation",
   xlabel="Longitude [°]",
   ylabel="Latitude [°]",
   ylabelsize=16,
   xlabelsize=16,
   titlesize=18,
   titlefont         = FONT,
   xlabelfont        = FONT,
   ylabelfont        = FONT,
   xticklabelfont    = FONT,
   yticklabelfont    = FONT,)
hm1 = heatmap!(ax1, lon, lat, Siva_Diss_norm,
   colormap=cmap, colorrange=crange)


# --- Panel (b): Energy Budget Dissipation (spatial map) ---
ax2 = Axis(fig[1, 2],
   title="(b) Residual Dissipation",
   xlabel="Longitude [°]",
   ylabel="",
   yticklabelsvisible=false,
   ylabelsize = 16,
   xlabelsize = 16,
   xticklabelsize    = 12,
   yticklabelsize    = 12,
   titlesize  = 18,
   titlefont         = FONT,
   xlabelfont        = FONT,
   ylabelfont        = FONT,
   xticklabelfont    = FONT,
   yticklabelfont    = FONT,)
hm2 = heatmap!(ax2, lon, lat, Budget_Diss_norm,
   colormap=cmap, colorrange=crange)


#Colorbar(fig[1, 3], hm2, label=rich("[x 10", superscript("-8"), " W/kg]"), labelsize = 14, ticklabelsize=12, width = 5)
Colorbar(fig[1, 3], hm2,
    ticklabelsize = 12,
    width = 15)   # a touch wider now that there's no side label eating space


Label(fig[1, 3, Top()],
    rich("[x 10", superscript("-8"), " W/kg]"),
    fontsize = 14,
    font = FONT,
    padding = (0, 0, 5, 0))   # (left, right, bottom, top) gap above the colorbar






# --- Panel (c): Zonal average comparison (line plot) ---
ax3 = Axis(fig[1, 4],
   title="(c) Zonal Average",
   xlabel=rich("Dissipation [×10", subscript("-8")," W/kg]"),
   ylabel="",
   yticklabelsvisible=false,
   ylabelsize = 16,
   xlabelsize = 16,
   titlesize = 18,
   xticklabelsize=12,
   yticklabelsize=12,
   titlefont         = FONT,
   xlabelfont        = FONT,
   ylabelfont        = FONT,
   xticklabelfont    = FONT,
   yticklabelfont    = FONT,
   )


lines!(ax3, Siva_zonal_scaled, lat,
   label="Spectral",
   color=:red,
   linewidth=2.5)


lines!(ax3, Budget_zonal_scaled_smooth, lat,
   label= "Residual",
   color=:blue,
   linewidth=2.5)


vlines!(ax3, [0], color=:gray, linestyle=:dash, linewidth=1)


axislegend(ax3, position=:lt, framevisible=true, labelsize=12)


# Keep all panels on the same latitude range for visual alignment
ylims!(ax1, minlat, maxlat)
ylims!(ax2, minlat, maxlat)
ylims!(ax3, minlat, maxlat)


colgap!(fig.layout, 1, 5)
colgap!(fig.layout, 2, 15)
colgap!(fig.layout, 3, 15)
colsize!(fig.layout, 1, Fixed(350))
colsize!(fig.layout, 2, Fixed(350))
colsize!(fig.layout, 4, Relative(0.25))


display(fig)


# Save figure
FIGDIR = cfg["fig_base"]
save(joinpath(FIGDIR, "Dissipation_Combined_nt_v1.png"), fig)
println("\nFigure saved: $(joinpath(FIGDIR, "Dissipation_Combined_nt_v1.png"))")




