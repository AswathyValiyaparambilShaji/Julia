using Printf, MAT, FilePathsBase, TOML, NCDatasets, CairoMakie


# Include FluxUtils.jl
include(joinpath(@__DIR__, "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin  # Changed from read_bin_be to read_bin


# Read the configuration file for paths
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base = cfg["base_path"]


# --- Grid parameters ---
NX, NY = 288, 468
nz = 88  # Total vertical levels in your grid


# Tiling parameters
buf = 3
tx, ty = 47, 66
nx = tx + 2 * buf
ny = ty + 2 * buf
dto = 144
Tts = 366192
nt = div(Tts, dto)


# --- Read F_band from NetCDF ---
ds = NCDataset(joinpath(base, "Siva_Diss", "TotDiss_band1.nc"))
println(ds)


# Read F_band (z × y × x) and permute to (x × y × z)
F_band_nc = ds["F_band"][:, :, :]  # This gives (88 × 467 × 287)
F_band = permutedims(F_band_nc, (3, 2, 1))  # Now (287 × 467 × 88)
println("F_band size: ", size(F_band))


close(ds)


# --- Thickness & constants ---
thk = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF = thk[1:nz]
rho0 = 999.8


# --- Pad F_band to match hFacC dimensions ---
# F_band is (287 × 467 × 88), need (288 × 468 × 90)
F_band_full = zeros(NX, NY, nz)
F_band_full[1:287, 1:467, 1:88] .= F_band
# Replace missing values with 0
F_band_full[isnan.(F_band_full)] .= 0.0


println("F_band_full size: ", size(F_band_full))


# --- Read hFacC ---
hFacC_full = zeros(NX, NY, nz)


# Create 3D DRF array for later use
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)


for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
        println("\n--- Processing tile: $suffix ---")
        
        # Read grid metrics using read_bin (not read_bin_be)
        hFacC = read_bin(joinpath(base, "hFacC", "hFacC_$suffix.bin"), (nx, ny, nz))
        
        # Calculate tile positions in global grid
        xs = (xn - 1) * tx + 1
        xe = xs + tx - 1
        ys = (yn - 1) * ty + 1
        ye = ys + ty - 1
        
        # Fill without buffer zones (matching your working code style)
        hFacC_full[xs:xe, ys:ye, :] .= hFacC[buf+1:nx-buf, buf+1:ny-buf, :]
    end
end


println("hFacC_full size: ", size(hFacC_full))


# --- Mask F_band with hFacC and integrate vertically ---
# Apply mask: only include points where hFacC > 0
F_masked = F_band_full .* (hFacC_full .> 0)


# Create 3D DRF array matching full domain
DRF3d_full = repeat(reshape(DRF, 1, 1, nz), NX, NY, 1)


# Vertical integration: sum(F * DRF * hFacC, dims=3)
F_integrated = dropdims(sum(F_masked .* DRF3d_full .* hFacC_full, dims=3), dims=3)


println("F_integrated size: ", size(F_integrated))
println("F_integrated range: ", extrema(F_integrated))


# --- Plot the integrated field using CairoMakie ---
fig = Figure(size=(1000, 800))
ax = Axis(fig[1, 1],
    title="Vertically Integrated F_band",
    xlabel="X",
    ylabel="Y",
    aspect=DataAspect())


# Create heatmap (transpose for correct orientation)
hm = heatmap!(ax, F_integrated', colormap=:viridis)


# Add colorbar
Colorbar(fig[1, 2], hm, label="Integrated Value")


# Save figure
save("F_band_integrated.png", fig)
println("\nPlot saved as F_band_integrated.png")


# Display figure (optional)
display(fig)




