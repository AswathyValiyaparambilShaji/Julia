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
rho0 = 999.8




# --- Thickness ---
thk  = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF  = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)




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


for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
        println("\n[Tile $suffix]")
        
        hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))

        DRFfull = hFacC .* DRF3d
        z = cumsum(DRFfull, dims=3)
        depth = sum(DRFfull, dims=3)
        DRFfull[hFacC .== 0] .= 0.0

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

        mask4D = reshape(hFacC .== 0, nx, ny, nz, 1)
 
        ucA_3d = sum(fu .* DRFfull, dims=3) ./ depth
        up_3d  = fu .- ucA_3d
        up_3d[repeat(mask4D, 1, 1, 1, size(up_3d, 4))] .= 0

        vcA_3d = sum(fv .* DRFfull, dims=3) ./ depth
        vp_3d  = fv .- vcA_3d
        vp_3d[repeat(mask4D, 1, 1, 1, size(vp_3d, 4))] .= 0 
        # --- Surface Velocity ---
        fu_surf = up_3d[:, :, 1, :]
        fv_surf = vp_3d[:, :, 1, :]


        # --- WPI ---
        WPI = tx_f .* fu_surf .+ ty_f .* fv_surf


        # --- Save ---
        wpi_file = joinpath(OUTDIR, "wpi_$suffix.bin")
        open(wpi_file, "w") do io
            write(io, Float32.(WPI))
        end


       
    end
end


println("\nAll WPI tiles saved to: $OUTDIR")
println("\nDone!")




