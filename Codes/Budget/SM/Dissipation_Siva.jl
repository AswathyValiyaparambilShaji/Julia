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
# PART 2: READ hFacC AND CALCULATE DEPTH-INTEGRATED SIVA DISSIPATION
# ============================================================================
println("\nReading hFacC and calculating depth-integrated Siva dissipation...")


hFacC_full = zeros(NX, NY, nz)
FH = zeros(NX, NY)


for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
        println("Processing tile: $suffix")
        
        # Read hFacC
        hFacC = read_bin(joinpath(base, "hFacC", "hFacC_$suffix.bin"), (nx, ny, nz))
        
        # Calculate depth
        DRFfull = hFacC .* DRF3d
        depth = sum(DRFfull, dims=3)
        
        # Calculate tile positions in global grid
        xs = (xn - 1) * tx + 1
        xe = xs + tx + (2 * buf) - 1
        ys = (yn - 1) * ty + 1
        ye = ys + ty + (2 * buf) - 1
        
        # Fill global arrays (remove buffer zones)
        hFacC_full[xs+2:xe-2, ys+2:ye-2, :] .= hFacC[buf:nx-buf+1, buf:ny-buf+1, :]
        FH[xs+2:xe-2, ys+2:ye-2] .= depth[buf:nx-buf+1, buf:ny-buf+1]
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
        
        println("Loading energy budget for tile: $suffix")
        
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


println("Budget dissipation range: ", extrema(Budget_Diss))


# ============================================================================
# PART 4: PLOT BOTH DISSIPATIONS
# ============================================================================
println("\nCreating comparison plots...")


# Normalize for plotting
Siva_Diss_norm = Siva_Diss_integrated./(FH) * 10^8
Budget_Diss_norm = (Budget_Diss ./ (rho0 .* FH)) * 10^8



# Create comparison figure
fig = Figure(resolution=(800, 400))


crange = (-1.5, 1.5)
cmap = :bwr


# Plot 1: Siva Dissipation
ax1 = Axis(fig[1, 1],
    title="(a) Direct Dissipation",
    xlabel="Longitude [°]",
    ylabel="Latitude [°]",
    ylabelsize=16,
    xlabelsize=16,
    titlesize=18)
hm1 = heatmap!(ax1, lon, lat, Siva_Diss_norm, 
    colormap=cmap, colorrange=crange)


# Plot 2: Energy Budget Dissipation
ax2 = Axis(fig[1, 2],
    title="(b) Residual Dissipation",
    xlabel="Longitude [°]",
    ylabel="",
    yticklabelsvisible=false,
    ylabelsize=16,
    xlabelsize=16,
    titlesize=18)
hm2 = heatmap!(ax2, lon, lat, Budget_Diss_norm, 
    colormap=cmap, colorrange=crange)
Colorbar(fig[1, 3], hm2, label=rich("[x 10", superscript("-8"), " W/kg]"))


display(fig)


# Save figure
FIGDIR = cfg["fig_base"]
save(joinpath(FIGDIR, "Dissipation_Comparison.png"), fig)
println("\nFigure saved: $(joinpath(FIGDIR, "Dissipation_Comparison.png"))")




