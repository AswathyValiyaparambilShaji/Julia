using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
include(joinpath(@__DIR__, "..","..","..", "functions", "butter_filters.jl"))
using .FluxUtils: read_bin, bandpassfilter


config_file = get(ENV, "JULIA_CONFIG",
            joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)


base  = cfg["base_path"]
base2 = cfg["base_path2"]


# --- Domain & grid ---
NX, NY = 288, 468
minlat, maxlat = 24.0, 31.91
minlon, maxlon = 193.0, 199.0
lat = range(minlat, maxlat, length=NY)
lon = range(minlon, maxlon, length=NX)


# --- Tile & time ---
buf = 3
tx, ty = 47, 66
nx = tx + 2*buf
ny = ty + 2*buf
nz = 88


dt  = 1       # hourly output, dt = 1 hr
dto = 144
Tts = 366192
nt  = div(Tts, dto)   # ~2543 hourly timesteps


# --- Thickness & constants ---
thk   = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF   = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)
rho0  = 999.8


# --- N2 threshold ---
N2_threshold = 1.0e-8


# --- Output directories ---
mkpath(joinpath(base2, "APE"))
mkpath(joinpath(base2, "pe"))


# ==========================================================
# ====================== MAIN LOOP =========================
# ==========================================================


for xn in cfg["xn_start"]:cfg["xn_end"]
for yn in cfg["yn_start"]:cfg["yn_end"]


    suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
    println("Processing tile: $suffix")


    # ----------------------------------------------------------
    # 1. Read raw hourly N2  (nx, ny, nz, nt)
    # ----------------------------------------------------------
    N2_raw = Float64.(read_bin(joinpath(base, "N2", "N2_$suffix.bin"),
                               (nx, ny, nz, nt)))
    println("  N2 raw range: ", extrema(filter(isfinite, N2_raw)))


    # ----------------------------------------------------------
    # 2. Low-pass filter N2 along time (36 hr cutoff, order 4)
    #    filtfilt operates along dim-1, so reshape to (nt, nx*ny*nz)
    # ----------------------------------------------------------
    println("  Low-pass filtering N2 (Tcut=36 hr)...")
    N2_2d = permutedims(N2_raw, (4, 1, 2, 3))          # (nt, nx, ny, nz)
    N2_2d = reshape(N2_2d, nt, nx*ny*nz)                # (nt, nx*ny*nz)


    N2_filt_2d = lowhighpass_butter(N2_2d, 36.0, dt, 4, "low")  # (nt, nx*ny*nz)


    N2_filt = reshape(N2_filt_2d, nt, nx, ny, nz)       # (nt, nx, ny, nz)
    N2_filt = permutedims(N2_filt, (2, 3, 4, 1))        # (nx, ny, nz, nt)


    # ----------------------------------------------------------
    # 3. Adjust filtered N2 to interfaces then average to centers
    #    (same approach as APE script for consistency)
    # ----------------------------------------------------------
    N2_adjusted = zeros(Float64, nx, ny, nz+1, nt)
    N2_adjusted[:, :, 1,      :] = N2_filt[:, :, 1,      :]
    N2_adjusted[:, :, 2:nz,   :] = N2_filt[:, :, 1:nz-1, :]
    N2_adjusted[:, :, nz+1,   :] = N2_filt[:, :, nz-1,   :]


    N2_center = zeros(Float64, nx, ny, nz, nt)
    for k in 1:nz
        N2_center[:, :, k, :] .=
            0.5 .* (N2_adjusted[:, :, k,   :] .+
                    N2_adjusted[:, :, k+1,  :])
    end


    # ----------------------------------------------------------
    # 4. Apply physical threshold
    # ----------------------------------------------------------
    n_filtered = sum(N2_center .< N2_threshold)
    n_total    = length(N2_center)
    println("  Filtering $(n_filtered) / $(n_total) values below threshold $(N2_threshold) ",
            "($(round(100*n_filtered/n_total, digits=2))%)")
    N2_center[N2_center .< N2_threshold] .= N2_threshold
    println("  N2_filtered range after threshold: ", extrema(N2_center))


    # ----------------------------------------------------------
    # 5. Read hFacC and buoyancy b
    # ----------------------------------------------------------
    hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))


    DRFfull = hFacC .* DRF3d
    DRFfull[hFacC .== 0] .= 0.0


    b = Float64.(open(joinpath(base2, "b", "b_t_sm_$suffix.bin"), "r") do io
        raw = read(io, nx * ny * nz * nt * sizeof(Float32))
        reshape(reinterpret(Float32, raw), nx, ny, nz, nt)
    end)


    # ----------------------------------------------------------
    # 6. Compute APE — fully vectorized, no time loop
    #    APE = 0.5 * rho0 * b² / N2   shape: (nx, ny, nz, nt)
    # ----------------------------------------------------------
    println("  Computing APE...")
    APE = 0.5 .* rho0 .* (b .^ 2) ./ N2_center


    println("  APE range:          ", extrema(filter(isfinite, APE)))
    println("  APE finite values:  ", sum(isfinite.(APE)))
    println("  APE Inf values:     ", sum(isinf.(APE)))
    println("  APE NaN values:     ", sum(isnan.(APE)))


    # --- pe (unchanged) ---
    pe = 0.5 .* b .^ 2


    # ----------------------------------------------------------
    # 7. Save
    # ----------------------------------------------------------
    open(joinpath(base2, "APE", "APE_tn_sm_$suffix.bin"), "w") do io
        write(io, Float32.(APE))
    end


    open(joinpath(base2, "pe", "pe_tn_sm_$suffix.bin"), "w") do io
        write(io, Float32.(pe))
    end


    println("  Completed tile: $suffix\n")
end
end


println("\nAll tiles processed successfully!")




