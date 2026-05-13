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
        u_ke_mean = Float64.(open(joinpath(base2, "BC","U_KE", "u_ke_mean_$suffix.bin"), "r") do io
            nbytes = nx * ny * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
        end)


        # --- Read PE Advection ---
        u_pe_mean = Float64.(open(joinpath(base2, "BC","U_PE", "u_pe_mean_$suffix.bin"), "r") do io
            nbytes = nx * ny * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
        end)


        # --- Read Shear Production ---
        sp_h_mean = Float64.(open(joinpath(base2, "BC","SP_H", "sp_h_mean_$suffix.bin"), "r") do io
            nbytes = nx * ny * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
        end)


        # --- Read Energy Tendency ---
        te_mean = Float64.(open(joinpath(base2, "TE_t", "te_t_mean_$suffix.bin"), "r") do io
            nbytes = nx * ny * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
        end)


        # --- Read Vertical Shear Production ---
        sp_v_mean = Float64.(open(joinpath(base2, "BC","SP_V", "sp_v_mean_$suffix.bin"), "r") do io
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
        end)#


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
        FH[xs+2:xe-2,     ys+2:ye-2] .= H[buf:nx-buf+1, buf:ny-buf+1]


        println("Completed tile $suffix")
    end
end

MF        = U_KE_full .+ U_PE_full .+ SP_H_full .+ SP_V_full .+ BP_full

# ── figure (unchanged except for the contour! lines below) ─────────────────
fig = Figure(resolution=(600, 800))
ax  = Axis(fig[1,1],
    title  = "Wave-mean flow interaction  (W/m²)",
    xlabel = "Longitude [°]",
    ylabel = "Latitude [°]")
ax.limits[] = ((minimum(lon), maximum(lon)), (minimum(lat), maximum(lat)))


hm = CairoMakie.heatmap!(ax, lon, lat, MF;
    interpolate = false,
    colorrange  = (-0.030, 0.030),
    colormap    = :bwr)


# --- bathymetry contours (thicker + labelled) ---
contour!(ax, lon, lat, FH;
    levels     = [500.0, 1000.0, 1500.0, 2000.0, 3000.0],
    color      = :black,
    linewidth  = 2,
    linestyle  = :solid,
    labels     = true,
    labelsize  = 18,
    labelcolor = :black)


Colorbar(fig[1,2], hm, label="W/m²")
display(fig)


FIGDIR = cfg["fig_base"]
mkpath(FIGDIR)
save(joinpath(FIGDIR, "WMF_map_v1_cpo.png"), fig)
println("Figure saved: $(joinpath(FIGDIR, "WMF_map_v1_cpo.png"))")




