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


# Initialize global arrays
Conv = zeros(NX, NY)
FDiv = zeros(NX, NY)
U_KE_full = zeros(NX, NY)
U_PE_full = zeros(NX, NY)
SP_H_full = zeros(NX, NY)


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
        
        # Same tile positions as Conv and FDiv
        U_KE_full[xs+2:xe-2, ys+2:ye-2] .= u_ke_interior
        U_PE_full[xs+2:xe-2, ys+2:ye-2] .= u_pe_interior
        SP_H_full[xs+2:xe-2, ys+2:ye-2] .= sp_h_interior
        
        println("Completed tile $suffix")
    end
end


println("\nCalculating derived terms...")

# Total energy fluxes (Flux Divergence + Advective fluxes)
TotalFlux = FDiv .+ U_KE_full .+ U_PE_full .+ SP_H_full


# Calculate Residual: R = C - (FDiv + U_KE + U_PE)
Residual = Conv .- TotalFlux


# ==========================================================
# =================== VISUALIZATION ========================
# ==========================================================


fig = Figure(size=(1800, 1200))


# Color range for plots
crange = (-0.05, 0.05)
cmap = Reverse(:RdBu)


# Row 1, Column 1: Conversion
ax1 = Axis(fig[1, 1], 
          title="(a) Conversion ⟨C⟩", 
          xlabel="Longitude [°]", 
          ylabel="Latitude [°]")
hm1 = heatmap!(ax1, lon, lat, Conv; 
              interpolate=false, 
              colorrange=crange, 
              colormap=cmap)


# Row 1, Column 2: Flux Divergence (Eddy fluxes)
ax2 = Axis(fig[1, 2], 
          title="(b) Flux Divergence ⟨∇·F⟩", 
          xlabel="Longitude [°]", 
          ylabel="Latitude [°]")
hm2 = heatmap!(ax2, lon, lat, FDiv; 
              interpolate=false, 
              colorrange=crange, 
              colormap=cmap)


# Row 1, Column 3: Advective KE
ax3 = Axis(fig[1, 3], 
          title="(c) Advective KE Flux", 
          xlabel="Longitude [°]", 
          ylabel="Latitude [°]")
hm3 = heatmap!(ax3, lon, lat, U_KE_full; 
              interpolate=false, 
              colorrange=crange, 
              colormap=cmap)


# Row 2, Column 1: Advective PE
ax4 = Axis(fig[2, 1], 
          title="(d) Advective PE Flux", 
          xlabel="Longitude [°]", 
          ylabel="Latitude [°]")
hm4 = heatmap!(ax4, lon, lat, U_PE_full; 
              interpolate=false, 
              colorrange=crange, 
              colormap=cmap)


# Row 2, Column 2: Shear Production
ax5 = Axis(fig[2, 2], 
          title="(e) Shear Production P^s", 
          xlabel="Longitude [°]", 
          ylabel="Latitude [°]")
hm5 = heatmap!(ax5, lon, lat, SP_H_full; 
              interpolate=false, 
              colorrange=crange, 
              colormap=cmap)


# Row 2, Column 3: Residual
ax6 = Axis(fig[2, 3], 
          title="(f) Residual (C - FDiv - U_KE - U_PE - SP_H)", 
          xlabel="Longitude [°]", 
          ylabel="Latitude [°]")
hm6 = heatmap!(ax6, lon, lat, Residual; 
              interpolate=false, 
              colorrange=crange, 
              colormap=cmap)


# Add shared colorbar
Colorbar(fig[1:2, 4], hm6, label="Energy Flux [W/m²]")


display(fig)


# Save figure
FIGDIR = cfg["fig_base"]
save(joinpath(FIGDIR, "EnergyBudget_Complete_v2.png"), fig)


println("\n=== Energy budget visualization complete ===")
println("Saved to: $(joinpath(FIGDIR, "EnergyBudget_Complete_v1.png"))")




