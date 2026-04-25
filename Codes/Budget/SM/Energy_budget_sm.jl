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
Conv         = zeros(NX, NY)
FDiv         = zeros(NX, NY)
U_KE_full    = zeros(NX, NY)
U_PE_full    = zeros(NX, NY)
SP_H_full    = zeros(NX, NY)
SP_V_full    = zeros(NX, NY)
BP_full      = zeros(NX, NY)
ET_full      = zeros(NX, NY)
WPI_full     = zeros(NX, NY)
G_vel_H_full = zeros(NX, NY)
G_vel_V_full = zeros(NX, NY)
G_buoy_full  = zeros(NX, NY)


println("Loading energy budget terms...")


# ==========================================================
# ============ LOAD ALL TERMS ==============================
# ==========================================================
for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix  = @sprintf("%02dx%02d_%d", xn, yn, buf)
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
        u_ke_mean = Float64.(open(joinpath(base2, "U_KE_old", "u_ke_mean_$suffix.bin"), "r") do io
            nbytes = nx * ny * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
        end)


        # --- Read PE Advection ---
        u_pe_mean = Float64.(open(joinpath(base2, "U_PE_old", "u_pe_mean_$suffix.bin"), "r") do io
            nbytes = nx * ny * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
        end)


        # --- Read Shear Production ---
        sp_h_mean = Float64.(open(joinpath(base2, "SP_H_old", "sp_h_mean_$suffix.bin"), "r") do io
            nbytes = nx * ny * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
        end)


        # --- Read Energy Tendency ---
        te_mean = Float64.(open(joinpath(base2, "TE_t", "te_t_mean_$suffix.bin"), "r") do io
            nbytes = nx * ny * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
        end)


        # --- Read Vertical Shear Production ---
        sp_v_mean = Float64.(open(joinpath(base2, "SP_V_old", "sp_v_mean_$suffix.bin"), "r") do io
            nbytes = nx * ny * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
        end)


        # --- Read Buoyancy Production ---
        bp_mean = Float64.(open(joinpath(base2, "BP_old", "bp_mean_$suffix.bin"), "r") do io
            nbytes = nx * ny * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
        end)


        # --- Read Wind Power Input (with time dimension) ---
        wpi_tile = Float64.(open(joinpath(base2, "WindPowerInput", "wpi_$suffix.bin"), "r") do io
            nbytes = nx * ny * nt * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt)
        end)


        # --- Read G horizontal shear (IT -> NIW) ---
        g_vel_h = Float64.(open(joinpath(base2, "G_vel_full", "g_vel_mean_$suffix.bin"), "r") do io
            nbytes = nx * ny * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
        end)


        # --- Read G vertical shear (IT -> NIW) ---
        g_vel_v = Float64.(open(joinpath(base2, "G_vel_V_full", "g_vel_v_mean_$suffix.bin"), "r") do io
            nbytes = nx * ny * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
        end)


        # --- Read G buoyancy (IT -> NIW) ---
        g_buoy = Float64.(open(joinpath(base2, "G_buoy_full", "g_buoy_mean_$suffix.bin"), "r") do io
            nbytes = nx * ny * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
        end)


        # Time average the WPI
        wpi_mean = mean(wpi_tile, dims=3)[:, :, 1]


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
        G_vel_H_full[xs+2:xe-2, ys+2:ye-2] .= g_vel_h[buf:nx-buf+1,  buf:ny-buf+1]
        G_vel_V_full[xs+2:xe-2, ys+2:ye-2] .= g_vel_v[buf:nx-buf+1,  buf:ny-buf+1]
        G_buoy_full[xs+2:xe-2,  ys+2:ye-2] .= g_buoy[buf:nx-buf+1,   buf:ny-buf+1]


        println("Completed tile $suffix")
    end
end


println("\nCalculating derived terms...")


# Total energy fluxes (Flux Divergence + Advective fluxes)
TotalFlux = FDiv .+ U_KE_full .+ U_PE_full
MF        = U_KE_full .+ U_PE_full .+ SP_H_full .+ SP_V_full .+ BP_full
A         = U_KE_full .+ U_PE_full
PS        = SP_H_full .+ SP_V_full
G_total   = G_vel_H_full .+ G_vel_V_full .+ G_buoy_full


# Residual dissipation -- G terms subtracted as energy lost from IT to NIW
Residual  = -(Conv .- TotalFlux .+ SP_H_full .+ SP_V_full .+ BP_full .+ WPI_full .- ET_full
              .+ G_vel_H_full .+ G_vel_V_full .+ G_buoy_full)
Residual2 = Conv .- FDiv


# Calculate spatial standard deviations
std_residual  = std(Residual,  corrected = false)
std_residual2 = std(Residual2, corrected = false)


# Convert WPI to mW/m2 for plotting only
WPI_plot = WPI_full .* 1000

# ==========================================================
# ============ SAVE DISSIPATION TO BINARY FILE =============
# ==========================================================


println("\nSaving dissipation field...")


DISS_DIR = joinpath(base2, "Dissipation")
mkpath(DISS_DIR)


open(joinpath(DISS_DIR, "dissipation_mean.bin"), "w") do io
    write(io, Float32.(Residual))
end


println("Dissipation saved to: $(joinpath(DISS_DIR, "dissipation_mean.bin"))")


# ==========================================================
# =================== VISUALIZATION ========================
# ==========================================================


fig = Figure(resolution=(1800, 800))


# Color ranges
crange  = (-0.03, 0.03)
crange2 = (-0.015, 0.015)
cmap = :bwr


# Row 1, Column 1: Conversion
ax1 = Axis(fig[1, 1],
    title = "(a) <C>",
    xlabel = "",
    xticklabelsvisible = false,
    ylabel = "Latitude [deg]"
)
hm1 = heatmap!(ax1, lon, lat, Conv;
    interpolate = false,
    colorrange = crange,
    colormap = cmap)


# Row 1, Column 2: Flux Divergence
ax2 = Axis(fig[1, 2],
    title = "(b) < ∇.F>",
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
    title = "(c) <A>",
    xlabel = "",
    xticklabelsvisible = false,
    ylabel = "",
    yticklabelsvisible = false
)
hm3 = heatmap!(ax3, lon, lat, A;
    interpolate = false,
    colorrange = crange,
    colormap = cmap)


# Row 1, Column 4: Dissipation
ax4 = Axis(fig[1, 4],
    title = "(d) <D>",
    xlabel = "",
    xticklabelsvisible = false,
    ylabel = "",
    yticklabelsvisible = false
)
hm4 = heatmap!(ax4, lon, lat, Residual;
    interpolate = false,
    colorrange = crange,
    colormap = cmap)


# Row 2, Column 1: Shear Production
ax5 = Axis(fig[2, 1],
    title = "(e) <Ps>",
    xlabel = "Longitude [deg]",
    ylabel = "Latitude [deg]"
)
hm5 = heatmap!(ax5, lon, lat, PS;
    interpolate = false,
    colorrange = crange2,
    colormap = cmap)


# Row 2, Column 2: Buoyancy Production
ax6 = Axis(fig[2, 2],
    title = "(f) <Pb>",
    xlabel = "Longitude [deg]",
    ylabel = "",
    yticklabelsvisible = false
)
hm6 = heatmap!(ax6, lon, lat, BP_full;
    interpolate = false,
    colorrange = crange2,
    colormap = cmap)


# Row 2, Column 3: Energy Tendency
ax7 = Axis(fig[2, 3],
    title = "(g) <dE/dt>",
    xlabel = "Longitude [deg]",
    ylabel = "",
    yticklabelsvisible = false
)
hm7 = heatmap!(ax7, lon, lat, ET_full;
    interpolate = false,
    colorrange = crange2,
    colormap = cmap)


#= Row 2, Column 4: Wind Power Input (x10^-3)
ax8 = Axis(fig[2, 4],
    title = "(h) <WPI> [x10^-3]",
    xlabel = "Longitude [deg]",
    ylabel = "",
    yticklabelsvisible = false
)
hm8 = heatmap!(ax8, lon, lat, WPI_plot;
    interpolate = false,
    colorrange = crange2,
    colormap = cmap)=#


# Row 2, Column 5: Total G transfer (IT -> NIW)
ax9 = Axis(fig[2, 4],
    title = "(i) <G>",
    xlabel = "Longitude [deg]",
    ylabel = "",
    yticklabelsvisible = false
)
hm9 = heatmap!(ax9, lon, lat, G_total;
    interpolate = false,
    colorrange = crange2,
    colormap = cmap)


# Add colorbars
Colorbar(fig[1, 6], hm4, label = "[W/m2]")
Colorbar(fig[2, 6], hm7, label = "[W/m2]")


display(fig)


# Save figure
FIGDIR = cfg["fig_base"]
save(joinpath(FIGDIR, "EnergyBudget_with_WPI_G_v6.png"), fig)


println("\nFigure saved: $(joinpath(FIGDIR, "EnergyBudget_with_WPI_G_v7.png"))")




