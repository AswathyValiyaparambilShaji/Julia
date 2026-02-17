using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML, CairoMakie


include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin


# Load configuration
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
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


# Time parameters
dt = 25
dto = 144
Tts = 366192
nt = div(Tts, dto)


# Initialize global arrays
Conv = zeros(NX, NY)
FDiv = zeros(NX, NY)
U_KE_full = zeros(NX, NY)
U_PE_full = zeros(NX, NY)
SP_H_full = zeros(NX, NY)
SP_V_full = zeros(NX, NY)
BP_full = zeros(NX, NY)
ET_full = zeros(NX, NY)
WPI_full = zeros(NX, NY)  # Wind Power Input


println("Loading energy budget terms...")


# ==========================================================
# ============ LOAD ALL TERMS ==============================
# ==========================================================
for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
        suffix2 = @sprintf("%02dx%02d_%d", xn, yn, buf-2)
        
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
        
        # Read time-averaged energy tendency
        te_mean = Float64.(open(joinpath(base2, "TE_t", "te_t_mean_$suffix.bin"), "r") do io
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
        
        # --- Read Wind Power Input (with time dimension) ---
        wpi_tile = Float64.(open(joinpath(base2, "WindPowerInput", "wpi_$suffix.bin"), "r") do io
            nbytes = nx * ny * nt * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt)
        end)
        
        # Time average the WPI
        wpi_mean = mean(wpi_tile, dims=3)[:, :, 1]
        
        # Calculate tile positions in global grid (same for all terms)
        xs = (xn - 1) * tx + 1
        xe = xs + tx + (2 * buf) - 1
        ys = (yn - 1) * ty + 1
        ye = ys + ty + (2 * buf) - 1
        
        # Update global arrays (remove buffer zones - same indexing for all)
        Conv[xs+2:xe-2, ys+2:ye-2] .= C[2:end-1, 2:end-1]
        FDiv[xs+2:xe-2, ys+2:ye-2] .= fxD[2:end-1, 2:end-1]
        
        # Use same interior extraction for advection terms
        u_ke_interior = u_ke_mean[buf:nx-buf+1, buf:ny-buf+1]
        u_pe_interior = u_pe_mean[buf:nx-buf+1, buf:ny-buf+1]
        sp_h_interior = sp_h_mean[buf:nx-buf+1, buf:ny-buf+1]
        sp_v_interior = sp_v_mean[buf:nx-buf+1, buf:ny-buf+1]
        bp_interior = bp_mean[buf:nx-buf+1, buf:ny-buf+1]
        te_interior = te_mean[buf:nx-buf+1, buf:ny-buf+1]
        wpi_interior = wpi_mean[buf:nx-buf+1, buf:ny-buf+1]
        
        # Same tile positions as Conv and FDiv
        U_KE_full[xs+2:xe-2, ys+2:ye-2] .= u_ke_interior
        U_PE_full[xs+2:xe-2, ys+2:ye-2] .= u_pe_interior
        SP_H_full[xs+2:xe-2, ys+2:ye-2] .= sp_h_interior
        SP_V_full[xs+2:xe-2, ys+2:ye-2] .= sp_v_interior
        BP_full[xs+2:xe-2, ys+2:ye-2] .= bp_interior
        ET_full[xs+2:xe-2, ys+2:ye-2] .= te_interior
        WPI_full[xs+2:xe-2, ys+2:ye-2] .= wpi_interior
        
        println("Completed tile $suffix")
    end
end


println("\nCalculating derived terms...")


# Total energy fluxes (Flux Divergence + Advective fluxes)
TotalFlux = FDiv .+ U_KE_full .+ U_PE_full
MF = U_KE_full .+ U_PE_full .+ SP_H_full .+ SP_V_full .+ BP_full
A = U_KE_full .+ U_PE_full
PS = SP_H_full .+ SP_V_full


# Update residual to include WPI
Residual = -(Conv .- TotalFlux .+ SP_H_full .+ SP_V_full .+ BP_full .+ WPI_full .+ ET_full)
Residual2 = Conv .- FDiv


# Calculate spatial standard deviations
std_residual = std(Residual, corrected = false)
std_residual2 = std(Residual2, corrected = false)


println("\nStandard Deviations:")
println("  Residual:  $(std_residual)")
println("  Residual2: $(std_residual2)")


println("\nWind Power Input Statistics:")
println("  Mean: $(mean(WPI_full)) W/m²")
println("  Min:  $(minimum(WPI_full)) W/m²")
println("  Max:  $(maximum(WPI_full)) W/m²")


# Convert WPI to mW/m² for plotting only
WPI_plot = WPI_full .* 1000  # Convert to mW/m² (same as ×10^-3 W/m²)


println("  Mean: $(mean(WPI_plot)) mW/m²")
println("  Min:  $(minimum(WPI_plot)) mW/m²")
println("  Max:  $(maximum(WPI_plot)) mW/m²")


# ==========================================================
# =================== VISUALIZATION ========================
# ==========================================================


fig = Figure(resolution=(1400, 900))


# Color range for ALL plots (same for all terms including WPI)
crange = (-0.03, 0.03)
cmap = Reverse(:RdBu)


# Row 1, Column 1: Conversion
ax1 = Axis(fig[1, 1],
    title = "(a) ⟨C⟩",
    xlabel = "",
    xticklabelsvisible = false,
    ylabel = "Latitude [°]"
)
hm1 = heatmap!(ax1, lon, lat, Conv;
    interpolate = false,
    colorrange = crange,
    colormap = cmap)


# Row 1, Column 2: Flux Divergence
ax2 = Axis(fig[1, 2],
    title = "(b) ⟨∇·F⟩",
    xlabel = "",
    xticklabelsvisible = false,
    ylabel = "",
    yticklabelsvisible = false
)
hm2 = heatmap!(ax2, lon, lat, FDiv;
    interpolate = false,
    colorrange = crange,
    colormap = cmap)


# Row 1, Column 3: Advective fluxes
ax3 = Axis(fig[1, 3],
    title = "(c) ⟨A⟩",
    xlabel = "",
    xticklabelsvisible = false,
    ylabel = "",
    yticklabelsvisible = false
)
hm3 = heatmap!(ax3, lon, lat, A;
    interpolate = false,
    colorrange = crange,
    colormap = cmap)


# Row 1, Column 4: Wind Power Input (×10⁻³) - SAME colorrange
ax4 = Axis(fig[1, 4],
    title = rich("(d) ⟨WPI⟩ [×10", superscript("-3"), "]"),
    xlabel = "",
    xticklabelsvisible = false,
    ylabel = "",
    yticklabelsvisible = false
)
hm4 = heatmap!(ax4, lon, lat, WPI_plot;
    interpolate = false,
    colorrange = crange,  # Same range as others
    colormap = cmap)


# Row 2, Column 1: Shear Production
ax5 = Axis(fig[2, 1],
    title = "(e) ⟨Pₛ⟩",
    xlabel = "Longitude [°]",
    ylabel = "Latitude [°]"
)
hm5 = heatmap!(ax5, lon, lat, PS;
    interpolate = false,
    colorrange = crange,
    colormap = cmap)


# Row 2, Column 2: Buoyancy Production
ax6 = Axis(fig[2, 2],
    title = rich("(f) ⟨P", subscript("b"), "⟩"),
    xlabel = "Longitude [°]",
    ylabel = "",
    yticklabelsvisible = false
)
hm6 = heatmap!(ax6, lon, lat, BP_full;
    interpolate = false,
    colorrange = crange,
    colormap = cmap)


# Row 2, Column 3: Energy Tendency
ax7 = Axis(fig[2, 3],
    title = "(g) ⟨∂E/∂t⟩",
    xlabel = "Longitude [°]",
    ylabel = "",
    yticklabelsvisible = false
)
hm7 = heatmap!(ax7, lon, lat, ET_full;
    interpolate = false,
    colorrange = crange,
    colormap = cmap)


# Row 2, Column 4: Residual (Dissipation)
ax8 = Axis(fig[2, 4],
    title = "(h) ⟨D⟩",
    xlabel = "Longitude [°]",
    ylabel = "",
    yticklabelsvisible = false
)
hm8 = heatmap!(ax8, lon, lat, Residual;
    interpolate = false,
    colorrange = crange,
    colormap = cmap)


# Add shared colorbar for ALL terms (including WPI)
Colorbar(fig[1:2, 5], hm8, label = "[W/m²]")


display(fig)


# Save figure
FIGDIR = cfg["fig_base"]
save(joinpath(FIGDIR, "EnergyBudget_with_WPI.png"), fig)


println("\nFigure saved: $(joinpath(FIGDIR, "EnergyBudget_with_WPI.png"))")




