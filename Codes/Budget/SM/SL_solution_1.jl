using MAT, Statistics, Printf, LinearAlgebra, TOML


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


# --- Thickness ---
thk = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF = thk[1:nz]
dz  = Float64.(DRF)
zf  = vcat(0.0, -cumsum(dz))    # size (89,)


# --- Wave parameters ---
om = 2π / (12.42 * 3600)


# ==========================================================
# MOORING LOCATIONS
# lat 25.5N lon 194.9E
# lat 27.8N lon 196.0E
# lat 28.9N lon 196.5E
# lat 30.1N lon 197.1E
# ==========================================================


# pick mooring 83: lat=25.5, lon=194.9
target_lat = 24.5
target_lon = 193.9


# find closest grid index
i_pt = argmin(abs.(collect(lon) .- target_lon))
j_pt = argmin(abs.(collect(lat) .- target_lat))
println("Target lat=$target_lat lon=$target_lon")
println("Grid index i=$i_pt j=$j_pt")
println("Actual  lat=$(lat[j_pt]) lon=$(lon[i_pt])")


# which tile contains this point?
# tile index
xn = cfg["xn_start"]
yn = cfg["yn_start"]
suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)


# local index within tile
i_local = i_pt - (xn-1)*tx + buf
j_local = j_pt - (yn-1)*ty + buf
println("Local index in tile $suffix: i=$i_local j=$j_local")


# --- Read N2 ---
N2_phase = Float64.(open(joinpath(base,"3day_mean","N2","N2_3day_$suffix.bin"), "r") do io
    raw = read(io, nx * ny * nz * nt_avg * sizeof(Float32))
    reshape(reinterpret(Float32, raw), nx, ny, nz, nt_avg)
end)


# --- Read hFacC ---
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)
hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))


# ==========================================================
# DIAGNOSE THIS SINGLE POINT
# ==========================================================
t = 1   # first 3-day window


println("\n--- Diagnosing point i=$i_local j=$j_local t=$t ---")


# hFacC profile
hfac_col = hFacC[i_local, j_local, :]
println("\nhFacC profile: ", hfac_col)


# N2 profile
N2_col = N2_phase[i_local, j_local, :, t]
println("\nN2 profile: ", N2_col)


# how many NaN?
println("\nNumber of NaN in N2: ", sum(isnan.(N2_col)))
println("Number of valid N2:  ", sum(.!isnan.(N2_col)))
println("N2 range (non-NaN):  ", extrema(N2_col[.!isnan.(N2_col)]))


# ocean cells
ocean_idx = findall(hfac_col .> 0)
println("\nOcean cell indices: ", ocean_idx)
println("k_top = ", ocean_idx[1])
println("k_bot = ", ocean_idx[end])


k_top = ocean_idx[1]
k_bot = ocean_idx[end]


# actual dz for this column
DRFfull = hfac_col .* dz
DRFfull[hfac_col .== 0] .= 0.0
dz_col = DRFfull[k_top:k_bot]
zf_col =-cumsum(dz_col)
#zf_col = vcat(0.0, -cumsum(dz_col))
println("\nzf_col: ", zf_col)


# N2 for this column
N2_col_valid = N2_phase[i_local, j_local, k_top+1:k_bot-1, t]
println("\nN2 valid column (k_top:k_bot-1): ", N2_col_valid)
println("Any NaN? ", any(isnan.(N2_col_valid)))


# pad N2 to faces
N2_faces = vcat(N2_col_valid[1], N2_col_valid, N2_col_valid[end])
println("\nN2_faces (padded): ", N2_faces)
println("length(N2_faces) = ", length(N2_faces))
println("length(zf_col)   = ", length(zf_col))


# ==========================================================
# NOW CALL SL FUNCTION
# ==========================================================
f_pt = 2 * 7.2921e-5 * sin(deg2rad(target_lat))
println("\nf = $f_pt rad/s")
println("om = $om rad/s")


println("\n--- Calling SL function ---")
k_sl, L_sl, C_sl, Cg_sl, Ce_sl, Weig_sl, Ueig_sl, Ueig2_sl =
    sturm_liouville_noneqDZ_norm(zf_col, N2_faces, f_pt, om, 0)


println("\nResults (first 5 modes):")
println("Mode | Ce (m/s) | Cg (m/s) | L (km)")
for n in 1:5
    println("  $n  |  $(round(Ce_sl[n], digits=3))  |  $(round(Cg_sl[n], digits=3))  |  $(round(L_sl[n]/1000, digits=1))")
end




# change to problem point
i_local = 25
j_local = 35
t = 1

println("\n--- Diagnosing problem point i=$i_local j=$j_local t=$t ---")

hfac_col = hFacC[i_local, j_local, :]
N2_col   = N2_phase[i_local, j_local, :, t]

println("hFacC: ", hfac_col)
println("N2:    ", N2_col)
println("NaN count in N2: ", sum(isnan.(N2_col)))

ocean_idx = findall(hfac_col .> 0)
println("Ocean indices: ", ocean_idx)
zf_col =-cumsum(dz_col)

k_top = ocean_idx[1]
k_bot = ocean_idx[end]

N2_col_valid = N2_phase[i_local, j_local, k_top+1:k_bot-2, t]
println("N2 valid column: ", N2_col_valid)
println("Any NaN in valid column? ", any(isnan.(N2_col_valid)))
