using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML
using CairoMakie


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
include(joinpath(@__DIR__, "..","..","..", "functions", "butter_filters.jl"))


using .FluxUtils: read_bin, bandpassfilter
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
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


dt  = 1
dto = 144
Tts = 366192
nt  = div(Tts, dto)


# --- Thickness & constants ---
thk   = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF   = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)
rho0  = 999.8


# --- Pick a single tile, point, and timestep to compare ---
xn_test = 1
yn_test = 1
ix      = 25    # local x index within tile (adjust as needed)
iy      = 33    # local y index within tile (adjust as needed)
it      = 500   # timestep index (adjust as needed)


suffix = @sprintf("%02dx%02d_%d", xn_test, yn_test, buf)
println("Comparing N2 profiles for tile: $suffix, ix=$ix, iy=$iy, it=$it")


# ==========================================================
# METHOD 1: Read instantaneous N2, then low-pass filter
# ==========================================================
println("\n--- Method 1: LP-filter instantaneous N2 ---")
N2_raw = Float64.(read_bin(joinpath(base, "N2", "N2_$suffix.bin"), (nx, ny, nz, nt)))
println("  N2 raw range: ", extrema(filter(isfinite, N2_raw)))
sizeof(N2_raw)
profile_m3 = N2_raw[ix, iy, :, it]                   # (nz,)

println("Handling NaNs before filtering...")
nan_mask = isnan.(N2_raw)
N2_raw[nan_mask] .= 0.0

println("  Low-pass filtering N2 (Tcut=36 hr)...")
N2_2d      = permutedims(N2_raw, (4, 1, 2, 3))          # (nt, nx, ny, nz)
N2_2d      = reshape(N2_2d, nt, nx*ny*nz)
N2_filt_2d = lowhighpass_butter(N2_2d, 36.0, dt, 4, "low")
N2_2d      = nothing; GC.gc()
N2_filt_m1 = reshape(N2_filt_2d, nt, nx, ny, nz)
N2_filt_2d = nothing; GC.gc()
N2_filt_m1 = permutedims(N2_filt_m1, (2, 3, 4, 1))      # (nx, ny, nz, nt)
N2_filt_m1[nan_mask] .= NaN
nan_mask = nothing; GC.gc()
profile_m1 = N2_filt_m1[ix, iy, :, it]                   # (nz,)
N2_raw = nothing; N2_filt_m1 = nothing; GC.gc()


# ==========================================================
# METHOD 2: Read N2 computed from LP-filtered density
# ==========================================================
println("\n--- Method 2: N2 from LP-filtered density ---")
N2_lpd = Float64.(read_bin(joinpath(base2, "N2_lpd", "N2_$suffix.bin"), (nx, ny, nz, nt)))
println("  N2_lpd range: ", extrema(filter(isfinite, N2_lpd)))


profile_m2 = N2_lpd[ix, iy, :, it]                       # (nz,)
N2_lpd = nothing; GC.gc()


# ==========================================================
# Depth axis from hFacC
# ==========================================================
hFacC   = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
DRFfull = hFacC .* DRF3d
DRFfull[hFacC .== 0] .= 0.0
z_cumsum  = cumsum(DRFfull, dims=3)
zz        = cat(zeros(nx, ny, 1), z_cumsum; dims=3)
z_centers = -0.5 .* (zz[:, :, 1:end-1] .+ zz[:, :, 2:end])  # (nx, ny, nz)
depth     = z_centers[ix, iy, :]                              # (nz,) negative meters


# ==========================================================
# Plot
# ==========================================================
fig = Figure(size=(500, 700))
ax  = Axis(fig[1, 1],
    xlabel = "N² (s⁻²)",
    ylabel = "Depth (m)",
    title  = @sprintf("N² profile comparison\ntile=%s, ix=%d, iy=%d, it=%d", suffix, ix, iy, it)
)

lines!(ax, profile_m3, depth, label="N² raw", color=:red,    linewidth=2)
lines!(ax, profile_m1, depth, label="LP-filter N² directly",      color=:steelblue, linewidth=2)
#lines!(ax, profile_m2, depth, label="N² from LP-filtered density", color=:tomato,    linewidth=2)


axislegend(ax, position=:rb)


outfile = joinpath(base2, "N2_profile_comparison_$(suffix)_it$(it).png")
save(outfile, fig)
display(fig)
println("\nSaved plot to: $outfile")




