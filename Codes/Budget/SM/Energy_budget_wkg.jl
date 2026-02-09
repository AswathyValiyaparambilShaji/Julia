using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML, CairoMakie


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin


# Load configuration
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base = cfg["base_path"]
base2 = cfg["base_path2"]


# --- TIME AVERAGING CONFIGURATION ---
# Set to true for 3-day averaging, false for full time period averaging
use_3day = true  # Change this to true for 3-day averaging


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


# ============================================================================
# MAIN WORKFLOW SPLIT: 3-DAY vs FULL TIME AVERAGE
# ============================================================================


if use_3day
    # ========================================================================
    # 3-DAY ENERGY BUDGET WORKFLOW - CREATE FRAMES
    # ========================================================================
    println("Loading 3-day energy budget terms for $nt3 periods...")
    
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
    
    println("\nGenerating frames for $nt3 3-day periods...")
    
    # Create frames directory
    FIGDIR = cfg["fig_base"]
    frames_dir = joinpath(FIGDIR, "energy_budget_frames")
    mkpath(frames_dir)
    
    # Color ranges
    crange = (-1.2, 1.2)
    crange2 = (-0.2, 0.2)
    cmap = :bwr
    
    # Generate frame for each time period
    for t in 1:nt3
        println("  Generating frame $t/$nt3...")
        
        # Calculate derived terms for this time
        TotalFlux_t = FDiv[:, :, t] .+ U_KE_full[:, :, t] .+ U_PE_full[:, :, t]
        A_t = U_KE_full[:, :, t] .+ U_PE_full[:, :, t]
        PS_t = SP_H_full[:, :, t] .+ SP_V_full[:, :, t]
        Residual_t = -(Conv[:, :, t] .- TotalFlux_t .+ PS_t .+ BP_full[:, :, t] .- ET_full[:, :, t])
        
        # Create figure
        fig = Figure(resolution=(1200, 800))
        
        # Row 1, Column 1: Conversion
        ax1 = Axis(fig[1, 1],
            title="(a) ⟨C⟩ - Period $t",
            xlabel="",
            xticklabelsvisible=false,
            ylabel="Latitude [°]",
            ylabelsize = 19,
            titlesize=22)
        hm1 = heatmap!(ax1, lon, lat, (Conv[:, :, t]./(rho0.*FH))*10^8;
            interpolate=false, colorrange=crange, colormap=cmap)
        
        # Row 1, Column 2: Flux Divergence
        ax2 = Axis(fig[1, 2],
            title="(b) ⟨∇·F⟩",
            xlabel="",
            xticklabelsvisible=false,
            ylabel="",
            yticklabelsvisible=false,
            titlesize=22)
        hm2 = heatmap!(ax2, lon, lat, (FDiv[:, :, t]./(rho0.*FH))*10^8;
            interpolate=false, colorrange=crange, colormap=cmap)
        
        # Row 1, Column 3: Residual
        ax8 = Axis(fig[1, 3],
            title="(c) ⟨D⟩",
            xlabel="",
            xticklabelsvisible=false,
            ylabel="",
            yticklabelsvisible=false,
            titlesize=22)
        hm8 = heatmap!(ax8, lon, lat, (Residual_t./(rho0.*FH))*10^8;
            interpolate=false, colorrange=crange, colormap=cmap)
        
        # Row 2, Column 1: Shear Production
        ax5 = Axis(fig[2, 1],
            title="(d) ⟨Pₛ⟩",
            xlabel="Longitude [°]",
            ylabel="Latitude [°]",
            ylabelsize = 19,
            xlabelsize = 19,
            titlesize=22)
        hm5 = heatmap!(ax5, lon, lat, (PS_t./(rho0.*FH))*10^8;
            interpolate=false, colorrange=crange2, colormap=cmap)
        
        # Row 2, Column 2: Buoyancy Production
        ax6 = Axis(fig[2, 2],
            title=rich("(e) ⟨P",subscript("b"),"⟩"),
            xlabel="Longitude [°]",
            xlabelsize = 19,
            ylabel="",
            yticklabelsvisible=false,
            titlesize=22)
        hm6 = heatmap!(ax6, lon, lat, (BP_full[:, :, t]./(rho0.*FH))*10^8;
            interpolate=false, colorrange=crange2, colormap=cmap)
        
        # Row 2, Column 3: Advection
        ax3 = Axis(fig[2, 3],
            title="(f) ⟨A⟩",
            xlabel="Longitude [°]",
            yticklabelsvisible=false,
            xlabelsize = 19,
            titlesize=22)
        hm3 = heatmap!(ax3, lon, lat, (A_t./(rho0.*FH))*10^8;
            interpolate=false, colorrange=crange2, colormap=cmap)
        
        # Colorbars
        Colorbar(fig[1, 4], hm8, label=rich("[x 10",superscript("-8"),"W/kg]"))
        Colorbar(fig[2, 4], hm3, label=rich("[x 10",superscript("-8"),"W/kg]"))
        
        # Save frame
        frame_file = joinpath(frames_dir, @sprintf("energy_budget_frame_%03d.png", t))
        save(frame_file, fig)
    end
    
    println("\nAll $nt3 frames saved to: $frames_dir")
    println("\nTo create a movie, run:")
    println("ffmpeg -framerate 5 -i $frames_dir/energy_budget_frame_%03d.png -c:v libx264 -pix_fmt yuv420p $(joinpath(FIGDIR, "energy_budget_3day.mp4"))")
    
else
    # ========================================================================
    # FULL TIME AVERAGE ENERGY BUDGET WORKFLOW - SINGLE FIGURE
    # ========================================================================
    println("Loading full time average energy budget terms...")
    
    # Initialize 2D arrays
    Conv = zeros(NX, NY)
    FDiv = zeros(NX, NY)
    U_KE_full = zeros(NX, NY)
    U_PE_full = zeros(NX, NY)
    SP_H_full = zeros(NX, NY)
    SP_V_full = zeros(NX, NY)
    BP_full = zeros(NX, NY)
    ∇H = zeros(NX, NY)
    FH = zeros(NX, NY)
    RAC = zeros(NX, NY)
    ET_full = zeros(NX, NY)
    
    # Load data for all tiles
    for xn in cfg["xn_start"]:cfg["xn_end"]
        for yn in cfg["yn_start"]:cfg["yn_end"]
            suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
            suffix2 = @sprintf("%02dx%02d_%d", xn, yn, buf-2)
            hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
            
            DRFfull = hFacC .* DRF3d
            z = cumsum(DRFfull, dims=3)
            depth = sum(DRFfull, dims=3)
            DRFfull[hFacC .== 0] .= 0.0
            
            # --- Read full time average data ---
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
    
    # Total energy fluxes
    TotalFlux = FDiv .+ U_KE_full .+ U_PE_full
    MF = U_KE_full .+ U_PE_full .+ SP_H_full .+ SP_V_full .+ BP_full
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
    
    # Create single figure
    fig = Figure(resolution=(1200, 800))
    
    # Color range for plots
    crange = (-1.2, 1.2)
    crange2 = (-0.2, 0.2)
    cmap = :bwr
    
    # Row 1, Column 1: Conversion
    ax1 = Axis(fig[1, 1],
        title="(a) ⟨C⟩",
        xlabel="",
        xticklabelsvisible=false,
        ylabel="Latitude [°]",
        ylabelsize = 19,
        titlesize=22)
    hm1 = heatmap!(ax1, lon, lat, (Conv./(rho0.*FH))*10^8;
        interpolate=false, colorrange=crange, colormap=cmap)
    
    # Row 1, Column 2: Flux Divergence
    ax2 = Axis(fig[1, 2],
        title="(b) ⟨∇·F⟩",
        xlabel="",
        xticklabelsvisible=false,
        ylabel="",
        yticklabelsvisible=false,
        titlesize=22)
    hm2 = heatmap!(ax2, lon, lat, (FDiv./(rho0.*FH))*10^8;
        interpolate=false, colorrange=crange, colormap=cmap)
    
    # Row 1, Column 3: Residual
    ax8 = Axis(fig[1, 3],
        title="(c) ⟨D⟩",
        xlabel="",
        xticklabelsvisible=false,
        ylabel="",
        yticklabelsvisible=false,
        titlesize=22)
    hm8 = heatmap!(ax8, lon, lat, (Residual./(rho0.*FH))*10^8;
        interpolate=false, colorrange=crange, colormap=cmap)
    
    # Row 2, Column 1: Shear Production
    ax5 = Axis(fig[2, 1],
        title="(d) ⟨Pₛ⟩",
        xlabel="Longitude [°]",
        ylabel="Latitude [°]",
        ylabelsize = 19,
        xlabelsize = 19,
        titlesize=22)
    hm5 = heatmap!(ax5, lon, lat, (PS./(rho0.*FH))*10^8;
        interpolate=false, colorrange=crange2, colormap=cmap)
    
    # Row 2, Column 2: Buoyancy Production
    ax6 = Axis(fig[2, 2],
        title=rich("(e) ⟨P",subscript("b"),"⟩"),
        xlabel="Longitude [°]",
        xlabelsize = 19,
        ylabel="",
        yticklabelsvisible=false,
        titlesize=22)
    hm6 = heatmap!(ax6, lon, lat, (BP_full./(rho0.*FH))*10^8;
        interpolate=false, colorrange=crange2, colormap=cmap)
    
    # Row 2, Column 3: Advection
    ax3 = Axis(fig[2, 3],
        title="(f) ⟨A⟩",
        xlabel="Longitude [°]",
        yticklabelsvisible=false,
        xlabelsize = 19,
        titlesize=22)
    hm3 = heatmap!(ax3, lon, lat, (A./(rho0.*FH))*10^8;
        interpolate=false, colorrange=crange2, colormap=cmap)
    
    # Add shared colorbars
    Colorbar(fig[1, 4], hm8, label=rich("[x 10",superscript("-8"),"W/kg]"))
    Colorbar(fig[2, 4], hm3, label=rich("[x 10",superscript("-8"),"W/kg]"))
    
    display(fig)
    
    # Save figure
    FIGDIR = cfg["fig_base"]
    save(joinpath(FIGDIR, "EnergyBudget_Total_wkg_v3.png"), fig)
    println("\nFigure saved: $(joinpath(FIGDIR, "EnergyBudget_Total_wkg_v3.png"))")
    
end




