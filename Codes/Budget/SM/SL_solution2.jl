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


# --- Read N2 and hFacC (whole tile, all timesteps, once) ---
N2_phase = Float64.(open(joinpath(base,"3day_mean","N2","N2_3day_$suffix.bin"), "r") do io
    raw = read(io, nx * ny * nz * nt_avg * sizeof(Float32))
    reshape(reinterpret(Float32, raw), nx, ny, nz, nt_avg)
end)


hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))


# ==========================================================
# TIME-INDEPENDENT SETUP FOR THIS POINT
# (ocean extent and depth grid don't change with t, only N2 does)
# ==========================================================
hfac_col = hFacC[i_local, j_local, :]
println("\n--- Point i=$i_local j=$j_local ---")
println("hFacC profile: ", hfac_col)


ocean_idx = findall(hfac_col .> 0)
k_top = ocean_idx[1]
k_bot = ocean_idx[end]
println("Ocean cell indices: ", ocean_idx)
println("k_top = $k_top, k_bot = $k_bot")


# --- Faces (zf) for the actual wet part of the column ---
dz_col = (hfac_col .* DRF)[k_top:k_bot]
zf_col = -cumsum(dz_col)                    # length M = k_bot - k_top + 1
M = length(zf_col)
println("zf_col: ", zf_col)
size(zf_col)

f_pt = 2 * 7.2921e-5 * sin(deg2rad(target_lat))
println("\nf  = $f_pt rad/s")
println("om = $om rad/s")


# ==========================================================
# LOOP OVER ALL 3-DAY WINDOWS, SOLVE SL AT EACH
# ==========================================================
n_modes_keep = 5   # how many low modes to store/report


k_ts   = fill(NaN, nt_avg, n_modes_keep)
L_ts   = fill(NaN, nt_avg, n_modes_keep)
C_ts   = fill(NaN, nt_avg, n_modes_keep)
Cg_ts  = fill(NaN, nt_avg, n_modes_keep)
Ce_ts  = fill(NaN, nt_avg, n_modes_keep)


println("\n--- Looping over $nt_avg 3-day windows at point i=$i_local j=$j_local ---")
for t in 1:nt_avg


    N2_col = N2_phase[i_local, j_local, :, t]   # full nz profile, may have leftover NaN
    n_nan_raw = sum(isnan.(N2_col))


    # Same NaN-fill approach as your single-timestep version: convert
    # NaN -> missing so Impute.locf/nocb can act on it, then convert
    # any leftover missing back to NaN.
    x = replace(N2_col, NaN => missing)
    x = Impute.locf(x)
    x = Impute.nocb(x)
    N2_col_filled = coalesce.(x, NaN)


    n_nan_after = sum(isnan.(N2_col_filled))
    if n_nan_after > 0
        @warn "t=$t: column still has $n_nan_after NaN after locf/nocb fill — skipping this window"
        continue
    end


    # N2 at faces, keeping the FULL set of valid interior values (same
    # logic as the single-timestep version): valid computed interface
    # values run k_top:k_bot-1 (M-1 values); the last slot (interface
    # k_bot, adjacent to land) is padded with the last valid (deepest)
    # value. The solver only uses the interior entries (positions
    # 2:end-1), so this bottom placeholder is bookkeeping only.
    N2_valid = N2_col_filled[k_top:k_bot-1]
    N2_faces = vcat(N2_valid, N2_valid[end])
size(N2_valid)
size(N2_faces)

    @assert length(N2_faces) == length(zf_col) "t=$t: N2_faces/zf_col length mismatch"


    k_sl, L_sl, C_sl, Cg_sl, Ce_sl, Weig_sl, Ueig_sl, Ueig2_sl =
        sturm_liouville_noneqDZ_norm(zf_col, N2_faces, f_pt, om, 0)


    n_avail = min(n_modes_keep, length(Ce_sl))
    k_ts[t, 1:n_avail]  = k_sl[1:n_avail]
    L_ts[t, 1:n_avail]  = L_sl[1:n_avail]
    C_ts[t, 1:n_avail]  = C_sl[1:n_avail]
    Cg_ts[t, 1:n_avail] = Cg_sl[1:n_avail]
    Ce_ts[t, 1:n_avail] = Ce_sl[1:n_avail]


    if t == 1 || t % 20 == 0
        println("t=$t  (raw NaN=$n_nan_raw)  Ce[1:3]=$(round.(Ce_sl[1:3], digits=3))  " *
                "Cg[1:3]=$(round.(Cg_sl[1:3], digits=3))")
    end
end


println("\n--- Done. Mode-1 eigenspeed Ce over time (first 10 windows) ---")
println(round.(Ce_ts[1:min(10,nt_avg), 1], digits=3))


println("\n--- Done. Mode-1 group speed Cg over time (first 10 windows) ---")
println(round.(Cg_ts[1:min(10,nt_avg), 1], digits=3))


# --- Save time series for this point ---
outdir = joinpath(base, "3day_mean", "N2")
mkpath(outdir)
outfile_ts = joinpath(outdir, "SL_timeseries_$(suffix)_i$(i_local)_j$(j_local).mat")
matwrite(outfile_ts, Dict(
    "k"    => k_ts,
    "L"    => L_ts,
    "C"    => C_ts,
    "Cg"   => Cg_ts,
    "Ce"   => Ce_ts,
    "i_local" => i_local, "j_local" => j_local,
    "lat" => lat[j_pt], "lon" => lon[i_pt]
))
println("\nSaved time series to $outfile_ts")





