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


println("Computing area-averaged energy budget for $nt3 3-day periods...")


# Initialize 3D arrays for time-varying data
Conv = zeros(NX, NY, nt3)
FDiv = zeros(NX, NY, nt3)
U_KE_full = zeros(NX, NY, nt3)
U_PE_full = zeros(NX, NY, nt3)
SP_H_full = zeros(NX, NY, nt3)
SP_V_full = zeros(NX, NY, nt3)
BP_full = zeros(NX, NY, nt3)
ET_full = zeros(NX, NY, nt3)


# Static fields (same for all times)
∇H = zeros(NX, NY)
FH = zeros(NX, NY)
RAC = zeros(NX, NY)


# Load data for all tiles
for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
        suffix2 = @sprintf("%02dx%02d_%d", xn, yn, buf-2)
        
        hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
        
        DRFfull = hFacC .* DRF3d
        depth = sum(DRFfull, dims=3)
        DRFfull[hFacC .== 0] .= 0.0
        
        # --- Read 3-day data (4D or 3D with time) ---
        fxD = Float64.(open(joinpath(base2, "FDiv_3day", "FDiv_3day_$(suffix2).bin"), "r") do io
            nbytes = (nx-2) * (ny-2) * nt3 * sizeof(Float32)
            raw_bytes = read(io, nbytes)
            raw_data = reinterpret(Float32, raw_bytes)
            reshape(raw_data, nx-2, ny-2, nt3)
        end)
        
        C = Float64.(open(joinpath(base2, "Conv_3day", "Conv_3day_$(suffix2).bin"), "r") do io
            nbytes = (nx-2) * (ny-2) * nt3 * sizeof(Float32)
            raw_bytes = read(io, nbytes)
            raw_data = reinterpret(Float32, raw_bytes)
            reshape(raw_data, nx-2, ny-2, nt3)
        end)
        
        u_ke_3day = Float64.(open(joinpath(base2, "U_KE_3day", "u_ke_3day_$suffix.bin"), "r") do io
            nbytes = nx * ny * nt3 * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt3)
        end)
        
        u_pe_3day = Float64.(open(joinpath(base2, "U_PE_3day", "u_pe_3day_$suffix.bin"), "r") do io
            nbytes = nx * ny * nt3 * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt3)
        end)
        
        sp_h_3day = Float64.(open(joinpath(base2, "SP_H_3day", "sp_h_3day_$suffix.bin"), "r") do io
            nbytes = nx * ny * nt3 * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt3)
        end)
        
        sp_v_3day = Float64.(open(joinpath(base2, "SP_V_3day", "sp_v_3day_$suffix.bin"), "r") do io
            nbytes = nx * ny * nt3 * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt3)
        end)
        
        bp_3day = Float64.(open(joinpath(base2, "BP_3day", "bp_3day_$suffix.bin"), "r") do io
            nbytes = nx * ny * nt3 * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt3)
        end)
        
        te_3day = Float64.(open(joinpath(base2, "TE_t_3day", "te_t_3day_$suffix.bin"), "r") do io
            nbytes = nx * ny * nt3 * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt3)
        end)
        
        dx = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
        dy = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))
        rac = dx .* dy
        H = depth
        
        # Horizontal gradients for roughness
        dHdx = zeros(nx, ny)
        dHdx[2:end-1, :] .= (H[3:end, :] .- H[1:end-2, :]) ./ (dx[2:end-1, :] .+ dx[3:end, :])
        
        dHdy = zeros(nx, ny)
        dHdy[:, 2:end-1] .= (H[:, 3:end] .- H[:, 1:end-2]) ./ (dy[:, 2:end-1] .+ dy[:, 3:end])
        
        gh = sqrt.(dHdx.^2 .+ dHdy.^2)
        
        # Calculate tile positions in global grid
        xs = (xn - 1) * tx + 1
        xe = xs + tx + (2 * buf) - 1
        ys = (yn - 1) * ty + 1
        ye = ys + ty + (2 * buf) - 1
        
        # Update global arrays (remove buffer zones)
        Conv[xs+2:xe-2, ys+2:ye-2, :] .= C[2:end-1, 2:end-1, :]
        FDiv[xs+2:xe-2, ys+2:ye-2, :] .= fxD[2:end-1, 2:end-1, :]
        
        # Extract interior regions for each time
        U_KE_full[xs+2:xe-2, ys+2:ye-2, :] .= u_ke_3day[buf:nx-buf+1, buf:ny-buf+1, :]
        U_PE_full[xs+2:xe-2, ys+2:ye-2, :] .= u_pe_3day[buf:nx-buf+1, buf:ny-buf+1, :]
        SP_H_full[xs+2:xe-2, ys+2:ye-2, :] .= sp_h_3day[buf:nx-buf+1, buf:ny-buf+1, :]
        SP_V_full[xs+2:xe-2, ys+2:ye-2, :] .= sp_v_3day[buf:nx-buf+1, buf:ny-buf+1, :]
        BP_full[xs+2:xe-2, ys+2:ye-2, :] .= bp_3day[buf:nx-buf+1, buf:ny-buf+1, :]
        ET_full[xs+2:xe-2, ys+2:ye-2, :] .= te_3day[buf:nx-buf+1, buf:ny-buf+1, :]
        
        # Static fields (use first time slice or mean)
        ∇H[xs+2:xe-2, ys+2:ye-2] .= gh[buf:nx-buf+1, buf:ny-buf+1]
        FH[xs+2:xe-2, ys+2:ye-2] .= H[buf:nx-buf+1, buf:ny-buf+1]
        RAC[xs+2:xe-2, ys+2:ye-2] .= rac[buf:nx-buf+1, buf:ny-buf+1]
        
        println("Completed tile $suffix")
    end
end


println("\nComputing area-averaged budget terms...")


# Initialize arrays for area-averaged time series
Conv_avg = zeros(nt3)
FDiv_avg = zeros(nt3)
U_KE_avg = zeros(nt3)
U_PE_avg = zeros(nt3)
SP_H_avg = zeros(nt3)
SP_V_avg = zeros(nt3)
BP_avg = zeros(nt3)
ET_avg = zeros(nt3)
A_avg = zeros(nt3)
PS_avg = zeros(nt3)
TotalFlux_avg = zeros(nt3)
Residual_avg = zeros(nt3)


# Compute area-weighted averages for each time step
for t in 1:nt3
    # Normalize by depth and density for each term
    Conv_norm = Conv[:, :, t] ./ (rho0 .* FH)
    FDiv_norm = FDiv[:, :, t] ./ (rho0 .* FH)
    U_KE_norm = U_KE_full[:, :, t] ./ (rho0 .* FH)
    U_PE_norm = U_PE_full[:, :, t] ./ (rho0 .* FH)
    SP_H_norm = SP_H_full[:, :, t] ./ (rho0 .* FH)
    SP_V_norm = SP_V_full[:, :, t] ./ (rho0 .* FH)
    BP_norm = BP_full[:, :, t] ./ (rho0 .* FH)
    ET_norm = ET_full[:, :, t] ./ (rho0 .* FH)
    
    # Calculate derived terms
    A_norm = U_KE_norm .+ U_PE_norm
    PS_norm = SP_H_norm .+ SP_V_norm
    TotalFlux_norm = FDiv_norm .+ U_KE_norm .+ U_PE_norm
    Residual_norm = -(Conv_norm .- TotalFlux_norm .+ PS_norm .+ BP_norm .- ET_norm)
    
    # Area-weighted average (using RAC as area weights)
    total_area = sum(RAC)
    
    Conv_avg[t] = sum(Conv_norm .* RAC) / total_area
    FDiv_avg[t] = sum(FDiv_norm .* RAC) / total_area
    U_KE_avg[t] = sum(U_KE_norm .* RAC) / total_area
    U_PE_avg[t] = sum(U_PE_norm .* RAC) / total_area
    SP_H_avg[t] = sum(SP_H_norm .* RAC) / total_area
    SP_V_avg[t] = sum(SP_V_norm .* RAC) / total_area
    BP_avg[t] = sum(BP_norm .* RAC) / total_area
    ET_avg[t] = sum(ET_norm .* RAC) / total_area
    A_avg[t] = sum(A_norm .* RAC) / total_area
    PS_avg[t] = sum(PS_norm .* RAC) / total_area
    TotalFlux_avg[t] = sum(TotalFlux_norm .* RAC) / total_area
    Residual_avg[t] = sum(Residual_norm .* RAC) / total_area
    
    if t % 10 == 0
        println("  Processed $t/$nt3 time periods")
    end
end


# Create time axis (in days, assuming 3-day periods)
time_days = collect(1:nt3) .* 3


println("\nCreating time series plot...")


# Create figure with subplots
fig = Figure(resolution=(1400, 1000))


# Color scheme for different terms
colors = Dict(
    "Conv" => :red,
    "FDiv" => :blue,
    "Residual" => :black,
    "PS" => :green,
    "BP" => :orange,
    "A" => :purple,
    "ET" => :brown
)


# Plot 1: Main budget terms (larger scale)
ax1 = Axis(fig[1, 1],
    title="Area-Averaged Energy Budget Terms (3-day periods)",
    xlabel="Time [days]",
    ylabel="Energy Rate [×10⁻⁸ W/kg]",
    titlesize=20,
    xlabelsize=16,
    ylabelsize=16)


lines!(ax1, time_days, Conv_avg * 1e8, label="⟨C⟩ - Conversion", color=colors["Conv"], linewidth=2)
lines!(ax1, time_days, FDiv_avg * 1e8, label="⟨∇·F⟩ - Flux Divergence", color=colors["FDiv"], linewidth=2)
lines!(ax1, time_days, Residual_avg * 1e8, label="⟨D⟩ - Dissipation", color=colors["Residual"], linewidth=2)
hlines!(ax1, [0], color=:gray, linestyle=:dash, linewidth=1)


axislegend(ax1, position=:rt, framevisible=true, labelsize=14)


# Plot 2: Production and advection terms (smaller scale)
ax2 = Axis(fig[2, 1],
    title="Production and Advection Terms",
    xlabel="Time [days]",
    ylabel="Energy Rate [×10⁻⁸ W/kg]",
    titlesize=20,
    xlabelsize=16,
    ylabelsize=16)


lines!(ax2, time_days, PS_avg * 1e8, label="⟨Pₛ⟩ - Shear Production", color=colors["PS"], linewidth=2)
lines!(ax2, time_days, BP_avg * 1e8, label="⟨Pᵦ⟩ - Buoyancy Production", color=colors["BP"], linewidth=2)
lines!(ax2, time_days, A_avg * 1e8, label="⟨A⟩ - Advection", color=colors["A"], linewidth=2)
lines!(ax2, time_days, ET_avg * 1e8, label="⟨∂E/∂t⟩ - Tendency", color=colors["ET"], linewidth=2)
hlines!(ax2, [0], color=:gray, linestyle=:dash, linewidth=1)


axislegend(ax2, position=:rt, framevisible=true, labelsize=14)


# Plot 3: All terms together
ax3 = Axis(fig[3, 1],
    title="All Budget Terms Combined",
    xlabel="Time [days]",
    ylabel="Energy Rate [×10⁻⁸ W/kg]",
    titlesize=20,
    xlabelsize=16,
    ylabelsize=16)


lines!(ax3, time_days, Conv_avg * 1e8, label="⟨C⟩", color=colors["Conv"], linewidth=1.5)
lines!(ax3, time_days, FDiv_avg * 1e8, label="⟨∇·F⟩", color=colors["FDiv"], linewidth=1.5)
lines!(ax3, time_days, PS_avg * 1e8, label="⟨Pₛ⟩", color=colors["PS"], linewidth=1.5)
lines!(ax3, time_days, BP_avg * 1e8, label="⟨Pᵦ⟩", color=colors["BP"], linewidth=1.5)
lines!(ax3, time_days, A_avg * 1e8, label="⟨A⟩", color=colors["A"], linewidth=1.5)
lines!(ax3, time_days, Residual_avg * 1e8, label="⟨D⟩", color=colors["Residual"], linewidth=2)
hlines!(ax3, [0], color=:gray, linestyle=:dash, linewidth=1)


axislegend(ax3, position=:rt, framevisible=true, labelsize=12, nbanks=2)


display(fig)


# Save figure
FIGDIR = cfg["fig_base"]
mkpath(FIGDIR)
save(joinpath(FIGDIR, "EnergyBudget_TimeSeries_3day.png"), fig)
println("\nFigure saved: $(joinpath(FIGDIR, "EnergyBudget_TimeSeries_3day.png"))")


# Print statistics
println("\n=== TIME-AVERAGED STATISTICS (over all 3-day periods) ===")
println("Conversion (C):           $(mean(Conv_avg)*1e8) ± $(std(Conv_avg)*1e8) [×10⁻⁸ W/kg]")
println("Flux Divergence (∇·F):    $(mean(FDiv_avg)*1e8) ± $(std(FDiv_avg)*1e8) [×10⁻⁸ W/kg]")
println("Shear Production (Pₛ):    $(mean(PS_avg)*1e8) ± $(std(PS_avg)*1e8) [×10⁻⁸ W/kg]")
println("Buoyancy Production (Pᵦ): $(mean(BP_avg)*1e8) ± $(std(BP_avg)*1e8) [×10⁻⁸ W/kg]")
println("Advection (A):            $(mean(A_avg)*1e8) ± $(std(A_avg)*1e8) [×10⁻⁸ W/kg]")
println("Tendency (∂E/∂t):         $(mean(ET_avg)*1e8) ± $(std(ET_avg)*1e8) [×10⁻⁸ W/kg]")
println("Dissipation (D):          $(mean(Residual_avg)*1e8) ± $(std(Residual_avg)*1e8) [×10⁻⁸ W/kg]")


# Save data to file for further analysis
using DelimitedFiles
output_data = hcat(time_days, Conv_avg*1e8, FDiv_avg*1e8, PS_avg*1e8, BP_avg*1e8, 
                   A_avg*1e8, ET_avg*1e8, Residual_avg*1e8)
header = "Time[days] Conv FDiv PS BP A ET Dissipation [all in 1e-8 W/kg]"
writedlm(joinpath(FIGDIR, "energy_budget_timeseries_3day.txt"), 
         vcat(header, output_data), '\t')
println("\nData saved: $(joinpath(FIGDIR, "energy_budget_timeseries_3day.txt"))")


println("\nProcessing complete!")




