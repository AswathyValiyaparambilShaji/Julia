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
rho0=999.8


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


println("\nCalculating derived terms...")


# Combined terms
A = U_KE_full .+ U_PE_full
PS = SP_H_full .+ SP_V_full


# Calculate residuals (unmasked)
Residual1 = Conv .- FDiv
Residual2 = Conv .- FDiv .- A
Residual3 = Conv .- FDiv .- A .+ PS
Residual4 = Conv .- FDiv .- A .+ PS .+ BP_full
Residual5 = Conv .- FDiv .- A .+ PS .+ BP_full .- ET_full


# ==========================================================
# ============ DEPTH MASKING AND AREA-WEIGHTED STATS =======
# ==========================================================


println("\n==========================================================")
println("Applying depth mask (>3900m)...")
println("==========================================================")


# Create mask for depths greater than 3900m
depth_mask = FH .> 3900.0


# Explicitly mask the grid cell areas
RAC_masked = RAC .* depth_mask


# Mask the residuals directly
Residual1_masked = Residual1 .* depth_mask
Residual2_masked = Residual2 .* depth_mask
Residual3_masked = Residual3 .* depth_mask
Residual4_masked = Residual4 .* depth_mask
Residual5_masked = Residual5 .* depth_mask


# Calculate total area of deep points only
total_area = sum(RAC_masked)


# Area-weighted mean = sum(value * area_masked) / total_area
aw_mean_residual1 = sum(Residual1_masked .* RAC_masked) / total_area
aw_mean_residual2 = sum(Residual2_masked .* RAC_masked) / total_area
aw_mean_residual3 = sum(Residual3_masked .* RAC_masked) / total_area
aw_mean_residual4 = sum(Residual4_masked .* RAC_masked) / total_area
aw_mean_residual5 = sum(Residual5_masked .* RAC_masked) / total_area


# Area-weighted standard deviation
# std = sqrt(sum((x - mean)^2 * area_masked) / total_area)
aw_std_residual1 = sqrt(sum(((Residual1_masked .- aw_mean_residual1).^2) .* RAC_masked) / total_area)
aw_std_residual2 = sqrt(sum(((Residual2_masked .- aw_mean_residual2).^2) .* RAC_masked) / total_area)
aw_std_residual3 = sqrt(sum(((Residual3_masked .- aw_mean_residual3).^2) .* RAC_masked) / total_area)
aw_std_residual4 = sqrt(sum(((Residual4_masked .- aw_mean_residual4).^2) .* RAC_masked) / total_area)
aw_std_residual5 = sqrt(sum(((Residual5_masked .- aw_mean_residual5).^2) .* RAC_masked) / total_area)


println("\n=== DEPTH > 3900m STATISTICS (Area-Weighted) ===")
println("\nArea-Weighted Means:")
println("  Residual1: $(aw_mean_residual1)")
println("  Residual2: $(aw_mean_residual2)")
println("  Residual3: $(aw_mean_residual3)")
println("  Residual4: $(aw_mean_residual4)")
println("  Residual5: $(aw_mean_residual5)")


println("\nArea-Weighted Standard Deviations:")
println("  Residual1: $(aw_std_residual1)")
println("  Residual2: $(aw_std_residual2)")
println("  Residual3: $(aw_std_residual3)")
println("  Residual4: $(aw_std_residual4)")
println("  Residual5: $(aw_std_residual5)")


println("\nMasking Statistics:")
println("  Total masked area: $(total_area) m²")
println("  Total domain area: $(sum(RAC)) m²")
println("  Number of deep points: $(sum(depth_mask))")
println("  Total number of points: $(NX*NY)")
println("  Percentage of points > 3900m: $(sum(depth_mask)/(NX*NY)*100)%")
println("  Percentage of area > 3900m: $(total_area/sum(RAC)*100)%")


# ==========================================================
# ============ BAR PLOT ====================================
# ==========================================================


FIGDIR = cfg["fig_base"]


# Create the plot
fig = Figure(resolution=(1200, 600))
ax = Axis(fig[1, 1],
    xlabel = "Residual Type",
    ylabel = "- D",
    title = "STD of Dissipation (Depth > 3900m, Area-Weighted)",
    xticks = (1:5, ["⟨C⟩-⟨∇·F⟩", "⟨C⟩-⟨∇·F⟩-⟨A⟩","⟨C⟩-⟨∇·F⟩-⟨A⟩+⟨SP⟩", "⟨C⟩-⟨∇·F⟩-⟨A⟩+⟨SP⟩+⟨BP⟩","⟨C⟩-⟨∇·F⟩-⟨A⟩+⟨SP⟩+⟨BP⟩-⟨∂E/∂t⟩"]),
    limits = (0.3, 5.7, 0, nothing),
    xticklabelsize = 14
)


# Data for bar plot - area-weighted std
std_values = [aw_std_residual1, aw_std_residual2, aw_std_residual3, aw_std_residual4, aw_std_residual5]


# Create bar plot
colors = [:coral, :seagreen, :goldenrod, :mediumpurple, :steelblue]
barplot!(ax, 1:5, std_values,
    color = colors,
    strokewidth = 2,
    strokecolor = :black,
    width = 0.6
)


# Add value labels on top of bars
for (i, val) in enumerate(std_values)
    text!(ax, i, val,
        text = @sprintf("%.2e", val),
        align = (:center, :bottom),
        fontsize = 14,
        offset = (0, 5)
    )
end


display(fig)
save(joinpath(FIGDIR, "Residual_std_Masked_AreaWeighted.png"), fig, px_per_unit=2)
println("\nSaved: Residual_std_Masked_AreaWeighted.png")


println("\n==========================================================")
println("Analysis complete! Plot saved to: $FIGDIR")
println("==========================================================")




