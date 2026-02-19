using Printf, MAT, FilePathsBase, TOML, NCDatasets, CairoMakie, Statistics


# Include FluxUtils.jl
include(joinpath(@__DIR__, "..", "..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin


# Read the configuration file for paths
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base = cfg["base_path"]
base2 = cfg["base_path2"]


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


rho0 = 999.8


# Depth threshold (in meters)
DEPTH_THRESHOLD = 3900.0


# --- Thickness & constants ---
thk = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)


# ============================================================================
# PART 1: READ SIVA'S DISSIPATION FROM NetCDF
# ============================================================================


ds = NCDataset(joinpath(base, "Siva_Diss", "TotDiss_band1.nc"))


# Read F_band (z × y × x) and permute to (x × y × z)
F_band_nc = ds["F_band"][:, :, :]  # (88 × 467 × 287)
F_band = permutedims(F_band_nc, (3, 2, 1))  # Now (287 × 467 × 88)


close(ds)


# Pad F_band to match domain dimensions (keep NaN as is)
F_band_full = fill(NaN, NX, NY, nz)
F_band_full[1:287, 1:467, :] .= F_band


# ============================================================================
# PART 2: READ hFacC AND CALCULATE DEPTH-INTEGRATED SIVA DISSIPATION
# ============================================================================


hFacC_full = zeros(NX, NY, nz)
FH = zeros(NX, NY)
DXC = zeros(NX, NY)  # Zonal grid spacing


for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
        
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
        FH[xs+2:xe-2, ys+2:ye-2] .= depth[buf:nx-buf+1, buf:ny-buf+1, 1]
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


# ============================================================================
# PART 3: CALCULATE ENERGY BUDGET DISSIPATION (RESIDUAL)
# ============================================================================


# Initialize arrays for energy budget terms
Conv = zeros(NX, NY)
FDiv = zeros(NX, NY)
U_KE_full = zeros(NX, NY)
U_PE_full = zeros(NX, NY)
SP_H_full = zeros(NX, NY)
SP_V_full = zeros(NX, NY)
BP_full = zeros(NX, NY)
ET_full = zeros(NX, NY)


# Load energy budget data for all tiles
for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
        suffix2 = @sprintf("%02dx%02d_%d", xn, yn, buf-2)
        
        # Read energy budget terms
        fxD = Float64.(open(joinpath(base2, "FDiv", "FDiv_$(suffix2).bin"), "r") do io
            nbytes = (nx-2) * (ny-2) * sizeof(Float32)
            raw_bytes = read(io, nbytes)
            raw_data = reinterpret(Float32, raw_bytes)
            reshape(raw_data, nx-2, ny-2)
        end)
        
        C = Float64.(open(joinpath(base2, "Conv", "Conv_$(suffix2).bin"), "r") do io
            nbytes = (nx-2) * (ny-2) * sizeof(Float32)
            raw_bytes = read(io, nbytes)
            raw_data = reinterpret(Float32, raw_bytes)
            reshape(raw_data, nx-2, ny-2)
        end)
        
        u_ke_mean = Float64.(open(joinpath(base2, "U_KE", "u_ke_mean_$suffix.bin"), "r") do io
            nbytes = nx * ny * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
        end)
        
        u_pe_mean = Float64.(open(joinpath(base2, "U_PE", "u_pe_mean_$suffix.bin"), "r") do io
            nbytes = nx * ny * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
        end)
        
        sp_h_mean = Float64.(open(joinpath(base2, "SP_H", "sp_h_mean_$suffix.bin"), "r") do io
            nbytes = nx * ny * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
        end)
        
        sp_v_mean = Float64.(open(joinpath(base2, "SP_V", "sp_v_mean_$suffix.bin"), "r") do io
            nbytes = nx * ny * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
        end)
        
        bp_mean = Float64.(open(joinpath(base2, "BP", "bp_mean_$suffix.bin"), "r") do io
            nbytes = nx * ny * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
        end)
        
        te_mean = Float64.(open(joinpath(base2, "TE_t", "te_t_mean_$suffix.bin"), "r") do io
            nbytes = nx * ny * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
        end)
        
        # Calculate tile positions
        xs = (xn - 1) * tx + 1
        xe = xs + tx + (2 * buf) - 1
        ys = (yn - 1) * ty + 1
        ye = ys + ty + (2 * buf) - 1
        
        # Update global arrays (remove buffer zones)
        Conv[xs+2:xe-2, ys+2:ye-2] .= C[2:end-1, 2:end-1]
        FDiv[xs+2:xe-2, ys+2:ye-2] .= fxD[2:end-1, 2:end-1]
        U_KE_full[xs+2:xe-2, ys+2:ye-2] .= u_ke_mean[buf:nx-buf+1, buf:ny-buf+1]
        U_PE_full[xs+2:xe-2, ys+2:ye-2] .= u_pe_mean[buf:nx-buf+1, buf:ny-buf+1]
        SP_H_full[xs+2:xe-2, ys+2:ye-2] .= sp_h_mean[buf:nx-buf+1, buf:ny-buf+1]
        SP_V_full[xs+2:xe-2, ys+2:ye-2] .= sp_v_mean[buf:nx-buf+1, buf:ny-buf+1]
        BP_full[xs+2:xe-2, ys+2:ye-2] .= bp_mean[buf:nx-buf+1, buf:ny-buf+1]
        ET_full[xs+2:xe-2, ys+2:ye-2] .= te_mean[buf:nx-buf+1, buf:ny-buf+1]
    end
end


# Calculate energy budget dissipation (residual)
TotalFlux = FDiv .+ U_KE_full .+ U_PE_full
PS = SP_H_full .+ SP_V_full
Budget_Diss = -(Conv .- TotalFlux .+ PS .+ BP_full .- ET_full)


# ============================================================================
# PART 4: APPLY DEPTH MASK AND COMPUTE ZONAL AVERAGES
# ============================================================================


# Create depth mask - TRUE where depth > 3900m
deep_mask = FH .> DEPTH_THRESHOLD


# Normalize dissipation fields
Siva_Diss_norm = zeros(NX, NY)
Budget_Diss_norm = zeros(NX, NY)


valid_mask = (FH .> 0.0) .& (DXC .> 0.0)
Siva_Diss_norm[valid_mask] = Siva_Diss_integrated[valid_mask] ./ FH[valid_mask]
Budget_Diss_norm[valid_mask] = Budget_Diss[valid_mask] ./ (rho0 .* FH[valid_mask])


# Compute zonal averages weighted by dx - ONLY for deep points
Siva_zonal = zeros(NY)
Budget_zonal = zeros(NY)


for j in 1:NY
    # For this latitude, find points where depth > 3900m
    deep_points_at_lat = deep_mask[:, j]
    
    if sum(deep_points_at_lat) > 0
        # Zonal average weighted by dx (zonal grid spacing)
        total_dx = sum(DXC[deep_points_at_lat, j])
        Siva_zonal[j] = sum(Siva_Diss_norm[deep_points_at_lat, j] .* DXC[deep_points_at_lat, j]) / total_dx
        Budget_zonal[j] = sum(Budget_Diss_norm[deep_points_at_lat, j] .* DXC[deep_points_at_lat, j]) / total_dx
    else
        # If no deep points at this latitude, set to NaN
        Siva_zonal[j] = NaN
        Budget_zonal[j] = NaN
    end
end


# Convert to 10^-8 W/kg
Siva_zonal_scaled = Siva_zonal * 1e8
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
        # Get window indices
        i_start = max(1, i - half_window)
        i_end = min(n, i + half_window)
        
        # Calculate mean of non-NaN values in window
        window_data = data[i_start:i_end]
        valid_data = filter(!isnan, window_data)
        
        if length(valid_data) > 0
            smoothed[i] = mean(valid_data)
        end
    end
    
    return smoothed
end


Budget_zonal_scaled_smooth = smooth_data(Budget_zonal_scaled, 15)


# Set first and last values of Siva to NaN
Siva_zonal_scaled[1] = NaN
Siva_zonal_scaled[end] = NaN

# Set first and last values of Siva to NaN
Siva_zonal_scaled[2] = NaN
Siva_zonal_scaled[end-1] = NaN
# ============================================================================
# PART 5: PLOT ZONAL AVERAGES
# ============================================================================


fig = Figure(resolution=(800, 600))


ax = Axis(fig[1, 1],
    title="Zonal Average Dissipation ",
    xlabel="Dissipation [×10⁻⁸ W/kg]",
    ylabel="Latitude [°]",
    xlabelsize=16,
    ylabelsize=16,
    titlesize=18)


# Plot both dissipation profiles
lines!(ax, Siva_zonal_scaled, lat, 
    label="Direct Dissipation", 
    color=:red, 
    linewidth=2.5)



lines!(ax, Budget_zonal_scaled_smooth, lat, 
    label="Residual Dissipation", 
    color=:blue, 
    linewidth=2.5)



# Add zero reference line
vlines!(ax, [0], color=:gray, linestyle=:dash, linewidth=1)


# Add legend
axislegend(ax, position=:lt, framevisible=true, labelsize=14)


display(fig)


# Save figure
FIGDIR = cfg["fig_base"]
save(joinpath(FIGDIR, "Dissipation_Zonal_Deep.png"), fig)
println("Figure saved: $(joinpath(FIGDIR, "Dissipation_Zonal_Deep.png"))")





