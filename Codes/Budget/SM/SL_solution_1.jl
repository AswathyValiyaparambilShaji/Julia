using MAT, Statistics, Printf, LinearAlgebra, TOML
using Impute


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin


include(joinpath(@__DIR__, "..","..","..", "functions", "strum_liouville_noneqDZ_norm.jl"))


config_file = get(ENV, "JULIA_CONFIG",
    	joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)


base  = cfg["base_path"]
base2 = cfg["base_path2"]


# --- Grid ---
NX, NY = 288, 468
minlat, maxlat = 24.0, 31.91
minlon, maxlon = 193.0, 199.0
lat = range(minlat, maxlat, length=NY)
lon = range(minlon, maxlon, length=NX)


buf = 3
tx, ty = 47, 66
nx = tx + 2*buf
ny = ty + 2*buf
nz = 88


dto = 144
Tts = 366192
nt  = div(Tts, dto)
ts  = 72
nt_avg = div(nt, ts)


# --- Thickness (same DRF for every column, hFacC scales it) ---
thk = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF = Float64.(thk[1:nz])


# --- Wave parameters ---
om = 2π / (12.42 * 3600)


# ==========================================================
# CHOOSE POINT
# ==========================================================
target_lat = 24.5
target_lon = 193.9


i_pt = argmin(abs.(collect(lon) .- target_lon))
j_pt = argmin(abs.(collect(lat) .- target_lat))
println("Target lat=$target_lat lon=$target_lon")
println("Grid index i=$i_pt j=$j_pt")
println("Actual  lat=$(lat[j_pt]) lon=$(lon[i_pt])")


xn = cfg["xn_start"]
yn = cfg["yn_start"]
suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)


i_local = i_pt - (xn-1)*tx + buf
j_local = j_pt - (yn-1)*ty + buf
println("Local index in tile $suffix: i=$i_local j=$j_local")


# --- Read N2 and hFacC ---
N2_phase = Float64.(open(joinpath(base,"3day_mean","N2","N2_3day_$suffix.bin"), "r") do io
	raw = read(io, nx * ny * nz * nt_avg * sizeof(Float32))
	reshape(reinterpret(Float32, raw), nx, ny, nz, nt_avg)
end)


hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))


# ==========================================================
# BUILD A CLEAN, FULL, NaN-FREE PROFILE FOR THIS POINT
# ==========================================================
t = 1   # first 3-day window


hfac_col = hFacC[i_local, j_local, :]
N2_col   = N2_phase[i_local, j_local, :, t]   # full nz profile, may have leftover NaN


#println("\n--- Point i=$i_local j=$j_local t=$t ---")
#println("hFacC profile: ", hfac_col)
#println("N2 profile (raw, full column): ", N2_col)
#println("NaN count in raw full column: ", sum(isnan.(N2_col)))




x = replace(N2_col, NaN => missing)   # promotes to Vector{Union{Missing,Float64}}
x = Impute.locf(x)
x = Impute.nocb(x)
N2_col_filled = coalesce.(x, NaN)     # convert any leftover `missing` back to NaN
println(N2_col_filled)

n_nan_after = sum(isnan.(N2_col_filled))
if n_nan_after > 0
	error("Column i=$i_local j=$j_local t=$t is still entirely NaN after " *
      	"locf/nocb fill ($n_nan_after NaNs remain)  no valid data exists " *
      	"anywhere in this column to fill from.")
end
println("N2 profile (after nearest-neighbor fill): ", N2_col_filled)


# --- Ocean extent for this column ---
ocean_idx = findall(hfac_col .> 0)
k_top = ocean_idx[1]
k_bot = ocean_idx[end]
#k_top = 1
#k_bot = 88
#println("Ocean cell indices: ", ocean_idx)
#println("k_top = $k_top, k_bot = $k_bot")


# --- Faces (zf) for the actual wet part of the column ---
dz_col = (hfac_col .* DRF)[k_top:k_bot]
zf_col = -cumsum(dz_col)                 	# length M = k_bot - k_top + 1
M = length(zf_col)
#println("zf_col: ", zf_col)


# --- N2 at faces, keeping the FULL set of valid interior values ---
# Valid computed interface values run k_top:k_bot-1 (M-1 values); the very
# last slot (interface k_bot, adjacent to land) has no physically meaningful
# value, so it's padded with the last valid (deepest) value. Note the
# solver's matrix construction only ever uses the interior entries
# (positions 2:end-1), so this bottom placeholder is bookkeeping only
# it never enters the eigenvalue calculation.
N2_valid = N2_col_filled[k_top:k_bot-1]  	# M-1 real values, no dropped points
N2_faces =vcat(N2_valid , N2_valid[end])	# length M, matches zf_col
size(N2_valid)
size(N2_faces)
println(N2_faces)

@assert length(N2_faces) == length(zf_col) "N2_faces and zf_col length mismatch"
println("N2_faces (full profile, NaN-free): ", N2_faces)
println("length(N2_faces) = ", length(N2_faces))
println("length(zf_col)   = ", length(zf_col))


# ==========================================================
# CALL SL FUNCTION
# ==========================================================
f_pt = 2 * 7.2921e-5 * sin(deg2rad(target_lat))
println("\nf  = $f_pt rad/s")
println("om = $om rad/s")


println("\n--- Calling SL function ---")
k_sl, L_sl, C_sl, Cg_sl, Ce_sl, Weig_sl, Ueig_sl, Ueig2_sl =
	sturm_liouville_noneqDZ_norm(zf_col, N2_faces, f_pt, om, 0)


println("\nResults (first 5 modes):")
println("Mode | Ce (m/s) | Cg (m/s) | L (km)")
for n in 1:5
	println("  $n  |  $(round(Ce_sl[n], digits=3))  |  $(round(Cg_sl[n], digits=3))  |  $(round(L_sl[n]/1000, digits=1))")
end


