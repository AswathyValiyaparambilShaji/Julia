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
SP_V_full = zeros(NX, NY)
BP_full = zeros(NX, NY)


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


      sp_v_mean = Float64.(open(joinpath(base2, "SP_V", "sp_v_mean_$suffix.bin"), "r") do io
          nbytes = nx * ny * sizeof(Float32)
          reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
      end)


      # --- Read Buoyancy Production ---
      bp_mean = Float64.(open(joinpath(base2, "BP", "bp_mean_$suffix.bin"), "r") do io
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
      sp_v_interior = sp_v_mean[buf:nx-buf+1, buf:ny-buf+1]
      bp_interior = bp_mean[buf:nx-buf+1, buf:ny-buf+1]
    
      # Same tile positions as Conv and FDiv
      U_KE_full[xs+2:xe-2, ys+2:ye-2] .= u_ke_interior
      U_PE_full[xs+2:xe-2, ys+2:ye-2] .= u_pe_interior
      SP_H_full[xs+2:xe-2, ys+2:ye-2] .= sp_h_interior
      SP_V_full[xs+2:xe-2, ys+2:ye-2] .= sp_v_interior
      BP_full[xs+2:xe-2, ys+2:ye-2] .= bp_interior 
    
      println("Completed tile $suffix")
  end
end


println("\nCalculating derived terms...")


# Total energy fluxes (Flux Divergence + Advective fluxes)
TotalFlux = FDiv .+ U_KE_full .+ U_PE_full .+ SP_H_full.+SP_V_full .+ BP_full
MF = U_KE_full .+ U_PE_full .+ SP_H_full.+SP_V_full .+ BP_full
A = U_KE_full .+ U_PE_full
PS = SP_H_full.+SP_V_full

Residual = Conv .- TotalFlux


# ==========================================================
# =================== VISUALIZATION ========================
# ==========================================================


fig = Figure(resolution=(1600, 400))

# Color range for plots
crange = (-0.02, 0.02)
cmap = Reverse(:RdBu)


# Row 1, Column 1: Conversion
ax1 = Axis(fig[1, 1],
        title="(a) Conversion ⟨C⟩",
        xlabel="",
        xticklabelsvisible=false,
        ylabel="Latitude [°]" 
        )
        #aspect=1)
hm1 = heatmap!(ax1, lon, lat, Conv;
            interpolate=false,
            colorrange=crange,
            colormap=cmap)


# Row 1, Column 2: Flux Divergence (Eddy fluxes)
ax2 = Axis(fig[1, 2],
        title="(b) Flux Divergence ⟨∇·F⟩",
        xlabel="",
        xticklabelsvisible=false,
        ylabel="",
        yticklabelsvisible=false, 
        )
        #aspect=1)
hm2 = heatmap!(ax2, lon, lat, FDiv;
            interpolate=false,
            colorrange=crange,
            colormap=cmap)


# Row 1, Column 3: Advective KE
ax3 = Axis(fig[1, 3],
        title="(c) ⟨A⟩+⟨PS⟩+⟨BP⟩",
        xlabel="",
        xticklabelsvisible=false,
        ylabel="",
        yticklabelsvisible=false 
        )
        #aspect=1)
hm3 = heatmap!(ax3, lon, lat, A;
            interpolate=false,
            colorrange=crange,
            colormap=cmap)


# Row 1, Column 4: Advective PE
ax4 = Axis(fig[1, 4],
        title="(d) Residual",
        xlabel="Longitude [°]",
        ylabel="",
        yticklabelsvisible=false )#aspect=1.2)
        #aspect=1)
hm4 = heatmap!(ax4, lon, lat,  Residual;
            interpolate=false,
            colorrange=crange,
            colormap=cmap)


# Add shared colorbar
Colorbar(fig[1, 5], hm4, label="Energy Flux [W/m²]")


display(fig)


# Save figure
FIGDIR = cfg["fig_base"]
save(joinpath(FIGDIR, "EnergyBudget_map2_v2.png"), fig)

