using Printf, FilePathsBase, TOML, Statistics, LinearAlgebra


include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter


config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base = cfg["base_path"]
base2 = cfg["base_path2"]


# --- Grid parameters ---
NX, NY = 288, 468
NZ = 64  # Number of vertical levels (adjust as needed)
minlat, maxlat = 24.0, 31.91
minlon, maxlon = 193.0, 199.0
lat = range(minlat, maxlat, length=NY)
lon = range(minlon, maxlon, length=NX)


# Tiling parameters
buf = 3
tx, ty = 47, 66
nx = tx + 2 * buf
ny = ty + 2 * buf
nz = NZ
dt = 25  # time step in seconds
dto = 144  # output interval in time steps
Tts = 366192
nt = div(Tts, dto)


println("Total time steps: $nt")
println("Grid: $NX × $NY × $NZ")


# --- Filter parameters (9-15 hour band) ---
T1, T2, delt, N = 9.0, 15.0, 1.0, 4


# Create output directory for WPI tiles
OUTDIR = joinpath(base2, "WindPowerInput")
mkpath(OUTDIR)
println("\nOutput directory: $OUTDIR")


# ============================================================================
# READ, FILTER, AND PROCESS DATA TILE BY TILE
# ============================================================================


println("\n" * "="^80)
println("Processing tiles...")
println("="^80)


for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
        
        println("\n[Tile $suffix]")
        
        # --- Read Wind Stress ---
        taux = Float64.(open(joinpath(base, "Windstress", "taux_$suffix.bin"), "r") do io
            nbytes = nx * ny * nt * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt)
        end)
        
        tauy = Float64.(open(joinpath(base, "Windstress", "tauy_$suffix.bin"), "r") do io
            nbytes = nx * ny * nt * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt)
        end)
        
        
        # --- Read Filtered Velocities (fu, fv) - already at cell centers ---
        fu = Float64.(open(joinpath(base2, "UVW_F", "fu_$suffix.bin"), "r") do io
            nbytes = nx * ny * nz * nt * sizeof(Float32)
            raw_bytes = read(io, nbytes)
            raw_data = reinterpret(Float32, raw_bytes)
            reshape(raw_data, nx, ny, nz, nt)
        end)
        
        fv = Float64.(open(joinpath(base2, "UVW_F", "fv_$suffix.bin"), "r") do io
            nbytes = nx * ny * nz * nt * sizeof(Float32)
            raw_bytes = read(io, nbytes)
            raw_data = reinterpret(Float32, raw_bytes)
            reshape(raw_data, nx, ny, nz, nt)
        end)
        
        # ========================================================================
        # CONVERT WIND STRESS FROM ARAKAWA C-GRID TO CELL CENTERS
        # ========================================================================
        

        # --- Wind Stress to Centers ---
        taux_ext = zeros(nx+1, ny, nt)
        taux_ext[1:nx, :, :] .= taux
        taux_ext[end, :, :] .= taux[end, :, :]
        
        tauy_ext = zeros(nx, ny+1, nt)
        tauy_ext[:, 1:ny, :] .= tauy
        tauy_ext[:, end, :] .= tauy[:, end, :]
        
        taux_c = 0.5 .* (taux_ext[1:end-1, :, :] .+ taux_ext[2:end, :, :])
        tauy_c = 0.5 .* (tauy_ext[:, 1:end-1, :] .+ tauy_ext[:, 2:end, :])
        
        # ========================================================================
        # BANDPASS FILTER WIND STRESS (9-15 hour band)
        # ========================================================================
   
        tx_f = bandpassfilter(taux_c, T1, T2, delt, N, nt)
        ty_f = bandpassfilter(tauy_c, T1, T2, delt, N, nt)
        
        # ========================================================================
        # EXTRACT SURFACE VELOCITY (k=1, already filtered and at centers)
        # ========================================================================
        
        println("  Extracting surface velocities...")
        
        # fu and fv are already filtered and at cell centers: take surface velocity (k=1)
        fu_surf = fu[:, :, 1, :]  # Surface layer (k=1)
        fv_surf = fv[:, :, 1, :]  # Surface layer (k=1)
        
        # ========================================================================
        # CALCULATE WIND POWER INPUT
        # ========================================================================
        
        println("  Calculating wind power input...")
        
        # Wind Power Input = τ · u_surface
        # WPI = τx_filtered * u_filtered + τy_filtered * v_filtered [W/m²]
        WPI = tx_f .* fu_surf .+ ty_f .* fv_surf
        
        # ========================================================================
        # SAVE WIND POWER INPUT AS TILE (Float32)
        # ========================================================================
        
        println("  Saving WPI tile...")
        
        wpi_file = joinpath(OUTDIR, "wpi_$suffix.bin")
        open(wpi_file, "w") do io
            write(io, Float32.(WPI))
        end
        
        println("  ✓ Completed tile $suffix")
        
        # Free memory
        taux = nothing
        tauy = nothing
        fu = nothing
        fv = nothing
        tx_f = nothing
        ty_f = nothing
        fu_surf = nothing
        fv_surf = nothing
        WPI = nothing
        GC.gc()
    end
end



println("\nAll WPI tiles saved to: $OUTDIR")
println("\nDone!")




