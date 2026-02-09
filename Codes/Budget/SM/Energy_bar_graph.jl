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


rho0 = 999.8
# --- Vertical levels ---
nz = 88


kz = 1
dt = 25
dto = 144
Tts = 366192
nt = div(Tts, dto)
nt3 = div(nt, 3*24)  # Number of 3-day periods


# --- Thickness & constants ---
thk = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)
g = 9.8


# Initialize global arrays for time-averaged fields
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


# Initialize 3D arrays for 3-day time series
Conv_3day = zeros(NX, NY, nt3)
FDiv_3day = zeros(NX, NY, nt3)
U_KE_3day = zeros(NX, NY, nt3)
U_PE_3day = zeros(NX, NY, nt3)
SP_H_3day = zeros(NX, NY, nt3)
SP_V_3day = zeros(NX, NY, nt3)
BP_3day = zeros(NX, NY, nt3)
ET_3day = zeros(NX, NY, nt3)


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
        
        # --- Read TIME-AVERAGED Flux Divergence ---
        fxD = Float64.(open(joinpath(base2, "FDiv", "FDiv_$(suffix2).bin"), "r") do io
            nbytes = (nx-2) * (ny-2) * sizeof(Float32)
            raw_bytes = read(io, nbytes)
            raw_data = reinterpret(Float32, raw_bytes)
            reshape(raw_data, nx-2, ny-2)
        end)
        
        # --- Read TIME-AVERAGED Conversion ---
        C = Float64.(open(joinpath(base2, "Conv", "Conv_$(suffix2).bin"), "r") do io
            nbytes = (nx-2) * (ny-2) * sizeof(Float32)
            raw_bytes = read(io, nbytes)
            raw_data = reinterpret(Float32, raw_bytes)
            reshape(raw_data, nx-2, ny-2)
        end)
        
        # --- Read TIME-AVERAGED KE Advection ---
        u_ke_mean = Float64.(open(joinpath(base2, "U_KE", "u_ke_mean_$suffix.bin"), "r") do io
            nbytes = nx * ny * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
        end)
        
        # --- Read TIME-AVERAGED PE Advection ---
        u_pe_mean = Float64.(open(joinpath(base2, "U_PE", "u_pe_mean_$suffix.bin"), "r") do io
            nbytes = nx * ny * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
        end)
        
        # --- Read TIME-AVERAGED Shear Production ---
        sp_h_mean = Float64.(open(joinpath(base2, "SP_H", "sp_h_mean_$suffix.bin"), "r") do io
            nbytes = nx * ny * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
        end)
        
        sp_v_mean = Float64.(open(joinpath(base2, "SP_V", "sp_v_mean_$suffix.bin"), "r") do io
            nbytes = nx * ny * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
        end)
        
        # --- Read TIME-AVERAGED Buoyancy Production ---
        bp_mean = Float64.(open(joinpath(base2, "BP", "bp_mean_$suffix.bin"), "r") do io
            nbytes = nx * ny * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
        end)
        
        # Read TIME-AVERAGED energy tendency
        te_mean = Float64.(open(joinpath(base2, "TE_t", "te_t_mean_$suffix.bin"), "r") do io
            nbytes = nx * ny * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
        end)
        
        # ==========================================================
        # ============ LOAD 3-DAY TIME SERIES ======================
        # ==========================================================
        
        # --- Read 3-DAY Flux Divergence ---
        fxD_3day_tile = Float64.(open(joinpath(base2, "FDiv_3day", "FDiv_3day_$(suffix2).bin"), "r") do io
            nbytes = (nx-2) * (ny-2) * nt3 * sizeof(Float32)
            raw_bytes = read(io, nbytes)
            raw_data = reinterpret(Float32, raw_bytes)
            reshape(raw_data, nx-2, ny-2, nt3)
        end)
        
        # --- Read 3-DAY Conversion ---
        C_3day_tile = Float64.(open(joinpath(base2, "Conv_3day", "Conv_3day_$(suffix2).bin"), "r") do io
            nbytes = (nx-2) * (ny-2) * nt3 * sizeof(Float32)
            raw_bytes = read(io, nbytes)
            raw_data = reinterpret(Float32, raw_bytes)
            reshape(raw_data, nx-2, ny-2, nt3)
        end)
        
        # --- Read 3-DAY KE Advection ---
        u_ke_3day_tile = Float64.(open(joinpath(base2, "U_KE_3day", "u_ke_3day_$suffix.bin"), "r") do io
            nbytes = nx * ny * nt3 * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt3)
        end)
        
        # --- Read 3-DAY PE Advection ---
        u_pe_3day_tile = Float64.(open(joinpath(base2, "U_PE_3day", "u_pe_3day_$suffix.bin"), "r") do io
            nbytes = nx * ny * nt3 * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt3)
        end)
        
        # --- Read 3-DAY Shear Production ---
        sp_h_3day_tile = Float64.(open(joinpath(base2, "SP_H_3day", "sp_h_3day_$suffix.bin"), "r") do io
            nbytes = nx * ny * nt3 * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt3)
        end)
        
        sp_v_3day_tile = Float64.(open(joinpath(base2, "SP_V_3day", "sp_v_3day_$suffix.bin"), "r") do io
            nbytes = nx * ny * nt3 * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt3)
        end)
        
        # --- Read 3-DAY Buoyancy Production ---
        bp_3day_tile = Float64.(open(joinpath(base2, "BP_3day", "bp_3day_$suffix.bin"), "r") do io
            nbytes = nx * ny * nt3 * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt3)
        end)
        
        # --- Read 3-DAY energy tendency ---
        te_3day_tile = Float64.(open(joinpath(base2, "TE_t_3day", "te_t_3day_$suffix.bin"), "r") do io
            nbytes = nx * ny * nt3 * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt3)
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
        
        # Update global arrays for TIME-AVERAGED fields (remove buffer zones)
        Conv[xs+2:xe-2, ys+2:ye-2] .= C[2:end-1, 2:end-1]
        FDiv[xs+2:xe-2, ys+2:ye-2] .= fxD[2:end-1, 2:end-1]
        
        # Extract interior regions for TIME-AVERAGED
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
        
        # Update global arrays for 3-DAY TIME SERIES
        Conv_3day[xs+2:xe-2, ys+2:ye-2, :] .= C_3day_tile[2:end-1, 2:end-1, :]
        FDiv_3day[xs+2:xe-2, ys+2:ye-2, :] .= fxD_3day_tile[2:end-1, 2:end-1, :]
        
        # Extract interior regions for 3-DAY
        U_KE_3day[xs+2:xe-2, ys+2:ye-2, :] .= u_ke_3day_tile[buf:nx-buf+1, buf:ny-buf+1, :]
        U_PE_3day[xs+2:xe-2, ys+2:ye-2, :] .= u_pe_3day_tile[buf:nx-buf+1, buf:ny-buf+1, :]
        SP_H_3day[xs+2:xe-2, ys+2:ye-2, :] .= sp_h_3day_tile[buf:nx-buf+1, buf:ny-buf+1, :]
        SP_V_3day[xs+2:xe-2, ys+2:ye-2, :] .= sp_v_3day_tile[buf:nx-buf+1, buf:ny-buf+1, :]
        BP_3day[xs+2:xe-2, ys+2:ye-2, :] .= bp_3day_tile[buf:nx-buf+1, buf:ny-buf+1, :]
        ET_3day[xs+2:xe-2, ys+2:ye-2, :] .= te_3day_tile[buf:nx-buf+1, buf:ny-buf+1, :]
        
        println("Completed tile $suffix")
    end
end


println("\nCalculating derived terms...")


# Total energy fluxes (Flux Divergence + Advective fluxes) - TIME-AVERAGED
TotalFlux = FDiv .+ U_KE_full .+ U_PE_full
A = U_KE_full .+ U_PE_full
PS = SP_H_full .+ SP_V_full


Residual = -(Conv .- TotalFlux .+ SP_H_full .+ SP_V_full .+ BP_full .- ET_full)
Residual2 = Conv .- FDiv


# Calculate spatial standard deviations
std_residual = std(Residual, corrected = false)
std_residual2 = std(Residual2, corrected = false)


println("\nStandard Deviations:")
println("  Residual:  $(std_residual)")
println("  Residual2: $(std_residual2)")


# ==========================================================
# ============ CONVERT TO W/kg (×10^-8) ====================
# ==========================================================


# Convert TIME-AVERAGED terms to W/kg in units of 10^-8
Conv_wkg = (Conv ./ (rho0 .* FH)) * 10^8
FDiv_wkg = (FDiv ./ (rho0 .* FH)) * 10^8
A_wkg = (A ./ (rho0 .* FH)) * 10^8
PS_wkg = (PS ./ (rho0 .* FH)) * 10^8
BP_wkg = (BP_full ./ (rho0 .* FH)) * 10^8
D_wkg = (Residual ./ (rho0 .* FH)) * 10^8


# Convert 3-DAY time series to W/kg
Conv_3day_wkg = zeros(NX, NY, nt3)
FDiv_3day_wkg = zeros(NX, NY, nt3)
A_3day_wkg = zeros(NX, NY, nt3)
PS_3day_wkg = zeros(NX, NY, nt3)
BP_3day_wkg = zeros(NX, NY, nt3)
D_3day_wkg = zeros(NX, NY, nt3)


for t in 1:nt3
    A_3day = U_KE_3day[:, :, t] .+ U_PE_3day[:, :, t]
    PS_3day = SP_H_3day[:, :, t] .+ SP_V_3day[:, :, t]
    TotalFlux_3day = FDiv_3day[:, :, t] .+ U_KE_3day[:, :, t] .+ U_PE_3day[:, :, t]
    Residual_3day = -(Conv_3day[:, :, t] .- TotalFlux_3day .+ SP_H_3day[:, :, t] .+ SP_V_3day[:, :, t] .+ BP_3day[:, :, t] .- ET_3day[:, :, t])
    
    Conv_3day_wkg[:, :, t] = (Conv_3day[:, :, t] ./ (rho0 .* FH)) * 10^8
    FDiv_3day_wkg[:, :, t] = (FDiv_3day[:, :, t] ./ (rho0 .* FH)) * 10^8
    A_3day_wkg[:, :, t] = (A_3day ./ (rho0 .* FH)) * 10^8
    PS_3day_wkg[:, :, t] = (PS_3day ./ (rho0 .* FH)) * 10^8
    BP_3day_wkg[:, :, t] = (BP_3day[:, :, t] ./ (rho0 .* FH)) * 10^8
    D_3day_wkg[:, :, t] = (Residual_3day ./ (rho0 .* FH)) * 10^8
end


# ==========================================================
# ============ AREA-WEIGHTED AVERAGING =====================
# ==========================================================


# Function for area-weighted mean (excluding NaN/Inf values)
function area_weighted_mean(field, area)
    valid_mask = .!(isnan.(field) .| isinf.(field))
    numerator = sum(field[valid_mask] .* area[valid_mask])
    denominator = sum(area[valid_mask])
    return numerator / denominator
end


# Function for temporal standard deviation of area-weighted means
function temporal_std_of_area_mean(field_3d, area)
    nt = size(field_3d, 3)
    means = zeros(nt)
    for t in 1:nt
        means[t] = area_weighted_mean(field_3d[:, :, t], area)
    end
    return std(means, corrected=false)
end


# Calculate area-weighted means (from time-averaged fields)
mean_Conv = area_weighted_mean(Conv_wkg, RAC)
mean_FDiv = -(area_weighted_mean(FDiv_wkg, RAC))
mean_A = -(area_weighted_mean(A_wkg, RAC))
mean_PS = area_weighted_mean(PS_wkg, RAC)
mean_BP = area_weighted_mean(BP_wkg, RAC)
mean_D = area_weighted_mean(D_wkg, RAC)


# Calculate temporal standard deviations (from 3-day time series)
std_Conv = temporal_std_of_area_mean(Conv_3day_wkg, RAC)
std_FDiv = temporal_std_of_area_mean(FDiv_3day_wkg, RAC)
std_A = temporal_std_of_area_mean(A_3day_wkg, RAC)
std_PS = temporal_std_of_area_mean(PS_3day_wkg, RAC)
std_BP = temporal_std_of_area_mean(BP_3day_wkg, RAC)
std_D = temporal_std_of_area_mean(D_3day_wkg, RAC)


f1 = -mean_FDiv/mean_Conv
f2 = (mean_PS+mean_BP)/mean_Conv
println("f1 = $f1")
println("f2 = $f2")


# ==========================================================
# ============ CREATE BARPLOT ==============================
# ==========================================================


fig_bar = Figure(resolution=(850, 500))


ax_bar = Axis(fig_bar[1, 1],
    xlabel = "Energy Budget Terms",
    ylabel = rich("[×10", superscript("-8"), " W/kg]"),
    title = "Area-Weighted Averaged Energy Budget with Temporal Std Dev",
)


# Data for barplot
terms = ["⟨C⟩", "-⟨∇·F⟩", "-⟨A⟩", "⟨Pₛ⟩", rich("⟨P",subscript("b"),"⟩"), "⟨D⟩"]
values = [mean_Conv, mean_FDiv, mean_A, mean_PS, mean_BP, mean_D]
errors = [std_Conv, std_FDiv, std_A, std_PS, std_BP, std_D]


# Create colors: positive = red, negative = blue
colors = [v >= 0 ? :red : :blue for v in values]


# Create barplot
barplot!(ax_bar, 1:length(terms), values,
    color = colors,
    strokecolor = :black,
    strokewidth = 1,
    width = 0.45
)


# Add error bars (now showing temporal variability)
errorbars!(ax_bar, 1:length(terms), values, errors,
    whiskerwidth = 10,
    color = :black,
    linewidth = 1.5
)


# Set x-axis labels
ax_bar.xticks = (1:length(terms), terms)


# Add a horizontal line at y=0
hlines!(ax_bar, [0], color = :black, linewidth = 1, linestyle = :dash)


display(fig_bar)


# Save barplot
FIGDIR = cfg["fig_base"]
save(joinpath(FIGDIR, "EnergyBudget_Barplot_with_temporal_std.png"), fig_bar)


println("\nBarplot saved to: $(joinpath(FIGDIR, "EnergyBudget_Barplot_with_temporal_std.png"))")




