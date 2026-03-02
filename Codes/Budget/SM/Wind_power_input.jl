using Printf, FilePathsBase, TOML, Statistics, LinearAlgebra


include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter


config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base = cfg["base_path"]
base2 = cfg["base_path2"]


# --- Grid parameters ---
NX, NY = 288, 468
NZ = 64
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
dt = 25
dto = 144
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
# DIAGNOSTIC STORAGE - accumulate across all tiles
# ============================================================================
diag_records = []   # will store NamedTuple for each tile


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


        # --- Read Filtered Velocities ---
        fu = Float64.(open(joinpath(base2, "UVW_F", "fu_$suffix.bin"), "r") do io
            nbytes = nx * ny * nz * nt * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nz, nt)
        end)


        fv = Float64.(open(joinpath(base2, "UVW_F", "fv_$suffix.bin"), "r") do io
            nbytes = nx * ny * nz * nt * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nz, nt)
        end)


        # --- Wind Stress to Centers ---
        taux_ext = zeros(nx+1, ny, nt)
        taux_ext[1:nx, :, :] .= taux
        taux_ext[end, :, :] .= taux[end, :, :]


        tauy_ext = zeros(nx, ny+1, nt)
        tauy_ext[:, 1:ny, :] .= tauy
        tauy_ext[:, end, :] .= tauy[:, end, :]


        taux_c = 0.5 .* (taux_ext[1:end-1, :, :] .+ taux_ext[2:end, :, :])
        tauy_c = 0.5 .* (tauy_ext[:, 1:end-1, :] .+ tauy_ext[:, 2:end, :])


        # --- Bandpass Filter Wind Stress ---
        tx_f = bandpassfilter(taux_c, T1, T2, delt, N, nt)
        ty_f = bandpassfilter(tauy_c, T1, T2, delt, N, nt)


        # --- Surface Velocity ---
        fu_surf = fu[:, :, 1, :]
        fv_surf = fv[:, :, 1, :]


        # --- WPI ---
        WPI = tx_f .* fu_surf .+ ty_f .* fv_surf


        # --- Store diagnostics for this tile (scalars only, no arrays) ---
        push!(diag_records, (
            suffix       = suffix,
            taux_rms     = sqrt(mean(taux_c.^2)),
            tauy_rms     = sqrt(mean(tauy_c.^2)),
            txf_rms      = sqrt(mean(tx_f.^2)),
            tyf_rms      = sqrt(mean(ty_f.^2)),
            filt_ratio   = sqrt(mean(tx_f.^2)) / sqrt(mean(taux_c.^2)),
            fu_rms       = sqrt(mean(fu_surf.^2)),
            fv_rms       = sqrt(mean(fv_surf.^2)),
            wpi_rms      = sqrt(mean(WPI.^2)),
            wpi_mean     = mean(WPI),
            wpi_max      = maximum(WPI),
            wpi_min      = minimum(WPI),
        ))


        println("  ✓ Tile $suffix done | WPI RMS = $(round(sqrt(mean(WPI.^2)), sigdigits=4)) W/m²")


        # --- Save ---
        wpi_file = joinpath(OUTDIR, "wpi_$suffix.bin")
        open(wpi_file, "w") do io
            write(io, Float32.(WPI))
        end


        # Free memory
        taux = nothing; tauy = nothing
        fu = nothing; fv = nothing
        tx_f = nothing; ty_f = nothing
        fu_surf = nothing; fv_surf = nothing
        WPI = nothing
        taux_c = nothing; tauy_c = nothing
        taux_ext = nothing; tauy_ext = nothing
        GC.gc()
    end
end


# ============================================================================
# PRINT FULL DIAGNOSTIC SUMMARY TABLE FOR ALL TILES
# ============================================================================
println("\n" * "="^80)
println("DIAGNOSTIC SUMMARY — ALL TILES")
println("="^80)
println(@sprintf("%-15s %10s %10s %10s %10s %10s %10s %10s %12s %12s",
    "Tile", "τx_rms", "τx_f_rms", "FiltRatio", "fu_rms", "fv_rms",
    "WPI_rms", "WPI_mean", "WPI_max", "WPI_min"))
println("-"^115)
for d in diag_records
    println(@sprintf("%-15s %10.4e %10.4e %10.4f %10.4e %10.4e %10.4e %12.4e %12.4e %12.4e",
        d.suffix,
        d.taux_rms, d.txf_rms, d.filt_ratio,
        d.fu_rms,   d.fv_rms,
        d.wpi_rms,  d.wpi_mean, d.wpi_max, d.wpi_min))
end
println("-"^115)


# --- Domain averages across all tiles ---
println("\nDomain averages across all $(length(diag_records)) tiles:")
println(@sprintf("  Mean τx_rms      : %.4e N/m²",  mean([d.taux_rms   for d in diag_records])))
println(@sprintf("  Mean τx_f_rms    : %.4e N/m²",  mean([d.txf_rms    for d in diag_records])))
println(@sprintf("  Mean filt ratio  : %.4f",        mean([d.filt_ratio for d in diag_records])))
println(@sprintf("  Mean fu_rms      : %.4e m/s",    mean([d.fu_rms     for d in diag_records])))
println(@sprintf("  Mean fv_rms      : %.4e m/s",    mean([d.fv_rms     for d in diag_records])))
println(@sprintf("  Mean WPI_rms     : %.4e W/m²",   mean([d.wpi_rms    for d in diag_records])))
println(@sprintf("  Mean WPI_mean    : %.4e W/m²",   mean([d.wpi_mean   for d in diag_records])))
println("="^80)


println("\nAll WPI tiles saved to: $OUTDIR")
println("\nDone!")




