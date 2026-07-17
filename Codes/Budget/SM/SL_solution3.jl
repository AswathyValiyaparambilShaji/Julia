using DSP, MAT, Statistics, Printf, LinearAlgebra, TOML, CairoMakie
using Impute


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter


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


kz = 1
dt = 25
dto = 144
Tts = 366192
nt  = div(Tts, dto)          # total hourly-ish baroclinic samples
ts  = 72                     # hourly samples per 3-day N2 window
nt_avg = div(nt, ts)         # number of 3-day N2 windows


# --- Thickness & constants ---
thk = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF = Float64.(thk[1:nz])
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)
g = 9.8


# --- Filter (9-15 day band, as in your flux script) ---
T1, T2, delt, N = 9.0, 15.0, 1.0, 4
fcutlow, fcuthigh = 1 / T2, 1 / T1
fnq = 1 / delt
bpf = digitalfilter(Bandpass(fcutlow, fcuthigh), Butterworth(N); fs = fnq)


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


xn = cfg["xn_start"]
yn = cfg["yn_start"]
suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)


i_local = i_pt - (xn-1)*tx + buf
j_local = j_pt - (yn-1)*ty + buf
println("Local index in tile $suffix: i=$i_local j=$j_local")


# ==========================================================
# READ BAROCLINIC FIELDS (whole tile, as in your flux script)
# ==========================================================
hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))


rho = Float64.(open(joinpath(base,"Density", "rho_in_$suffix.bin"), "r") do io
    nbytes = nx * ny * nz * nt * sizeof(Float64)
    raw_bytes = read(io, nbytes)
    raw_data = reinterpret(Float64, raw_bytes)
    reshape(raw_data, nx, ny, nz, nt)
end)


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


# --- Depths / masks (whole tile, same as your flux script) ---
DRFfull = hFacC .* DRF3d
depth = sum(DRFfull, dims=3)
DRFfull[hFacC .== 0] .= 0.0


# --- Bandpass filter density (tidal-band) ---
fr = bandpassfilter(rho, T1, T2, delt, N, nt)


# --- Perturbation velocities (same construction as your flux script) ---
mask4D = reshape(hFacC .== 0, nx, ny, nz, 1)


ucA_3d = sum(fu .* DRFfull, dims=3) ./ depth
up_3d  = fu .- ucA_3d
up_3d[repeat(mask4D, 1, 1, 1, size(up_3d, 4))] .= 0


vcA_3d = sum(fv .* DRFfull, dims=3) ./ depth
vp_3d  = fv .- vcA_3d
vp_3d[repeat(mask4D, 1, 1, 1, size(vp_3d, 4))] .= 0


# --- Extract the single point's profiles across ALL hourly timesteps ---
up_pt = up_3d[i_local, j_local, :, :]   # (nz, nt)
vp_pt = vp_3d[i_local, j_local, :, :]   # (nz, nt)


# ==========================================================
# READ N2 (3-day windows) AND COMPUTE THE TIME-MEAN PROFILE
# ==========================================================
N2_phase = Float64.(open(joinpath(base,"3day_mean","N2","N2_3day_$suffix.bin"), "r") do io
    raw = read(io, nx * ny * nz * nt_avg * sizeof(Float32))
    reshape(reinterpret(Float32, raw), nx, ny, nz, nt_avg)
end)


hfac_col = hFacC[i_local, j_local, :]
ocean_idx = findall(hfac_col .> 0)
k_top = ocean_idx[1]
k_bot = ocean_idx[end]
println("\nOcean cell indices: ", ocean_idx)
println("k_top = $k_top, k_bot = $k_bot")


# --- N2 time series at this point: fill NaN in each 3-day window first,
#     THEN average across windows to get one time-mean profile ---
N2_col_ts = N2_phase[i_local, j_local, :, :]   # (nz, nt_avg)
N2_col_ts_filled = similar(N2_col_ts)


n_bad_windows = 0
for t in 1:nt_avg
    col = N2_col_ts[:, t]
    x = replace(col, NaN => missing)
    x = Impute.locf(x)
    x = Impute.nocb(x)
    col_filled = coalesce.(x, NaN)
    if any(isnan, col_filled)
        n_bad_windows += 1
        col_filled .= NaN   # leave as NaN; excluded from the mean below via skipmissing-style handling
    end
    N2_col_ts_filled[:, t] = col_filled
end
if n_bad_windows > 0
    println("Note: $n_bad_windows of $nt_avg windows still had NaN after fill and are excluded from the time-mean.")
end


# time-mean across windows, ignoring any still-NaN windows per depth level
N2_mean_col = [mean(filter(!isnan, N2_col_ts_filled[k, :])) for k in 1:nz]
n_nan_mean = sum(isnan.(N2_mean_col))
if n_nan_mean > 0
    error("Time-mean N2 profile still has $n_nan_mean NaN depth level(s) — " *
          "every window was NaN at that depth, nothing to average.")
end
println("\nTime-mean N2 profile (full column): ", N2_mean_col)


# --- Faces (zf) for the actual wet part of the column ---
dz_col = (hfac_col .* DRF)[k_top:k_bot]
zf_col = -cumsum(dz_col)          # length M
M = length(zf_col)
H = sum(dz_col)


dz_vel = dz_col[2:end]            # length M-1, matches Ueig2's derivative-shortened grid
z_vel_centers = 0.5 .* (zf_col[1:end-1] .+ zf_col[2:end])   # length M-1


# --- N2 at faces, from the TIME-MEAN profile (same padding convention as before) ---
N2_valid = N2_mean_col[k_top:k_bot-1]
N2_faces = vcat(N2_valid, N2_valid[end])
@assert length(N2_faces) == length(zf_col) "N2_faces/zf_col length mismatch"


f_pt = 2 * 7.2921e-5 * sin(deg2rad(target_lat))
println("f  = $f_pt rad/s")
println("om = $om rad/s")
println(Ueig2_sl[:,1])

# ==========================================================
# SOLVE THE SL PROBLEM ONCE, USING THE TIME-MEAN N2
# ==========================================================
n_modes_keep = 25


k_sl, L_sl, C_sl, Cg_sl, Ce_sl, Weig_sl, Ueig_sl, Ueig2_sl =
    sturm_liouville_noneqDZ_norm(zf_col, N2_faces, f_pt, om, 0)


println("\nResults from the single (time-mean N2) SL solve, first $n_modes_keep modes:")
println("Mode | Ce (m/s) | Cg (m/s) | L (km)")
for n in 1:n_modes_keep
    println("  $n  |  $(round(Ce_sl[n], digits=3))  |  $(round(Cg_sl[n], digits=3))  |  $(round(L_sl[n]/1000, digits=1))")
end


# --- pick a depth to compare (meters, negative = below surface) ---
target_depth_m = -100.0
idx_vel = argmin(abs.(z_vel_centers .- target_depth_m))
println("\nComparing at depth ≈ $(round(z_vel_centers[idx_vel], digits=1)) m " *
        "(nearest to requested $target_depth_m m), grid index $idx_vel of $(M-1)")


abs_idx = k_top + idx_vel
Phi_here = Ueig2_sl[idx_vel, 1:n_modes_keep]   # mode-shape value at the chosen depth, per mode


# ==========================================================
# PROJECT EVERY HOURLY TIMESTEP ONTO THE FIXED MODE SHAPES,
# THEN SUM MODES 1-5 FOR THE RECONSTRUCTION
# ==========================================================
uhat = fill(NaN, nt, n_modes_keep)   # projected amplitude, per mode, all hourly steps
vhat = fill(NaN, nt, n_modes_keep)


u_recon_sum = fill(NaN, nt)   # sum of modes 1-5 reconstruction at the chosen depth
v_recon_sum = fill(NaN, nt)


u_actual = up_pt[abs_idx, :]   # actual BC signal at the chosen depth, all hourly steps
v_actual = vp_pt[abs_idx, :]


println("\n--- Projecting $nt hourly timesteps onto the fixed (time-mean) mode shapes ---")
for tau in 1:nt
    u_col = up_pt[k_top+1:k_bot, tau]
    v_col = vp_pt[k_top+1:k_bot, tau]


    u_sum = 0.0
    v_sum = 0.0                        
    for n in 1:n_modes_keep
        Phi_n = Ueig2_sl[:, n]
        uh = (1/H) * sum(Phi_n .* u_col .* dz_vel)
        vh = (1/H) * sum(Phi_n .* v_col .* dz_vel)
        uhat[tau, n] = uh
        vhat[tau, n] = vh
        u_sum += Phi_here[n] * uh
        v_sum += Phi_here[n] * vh
    end
    u_recon_sum[tau] = u_sum
    v_recon_sum[tau] = v_sum


    if tau == 1 || tau % 500 == 0
        println("tau=$tau done")
    end
end


# ==========================================================
# PLOT: actual BC vs sum of modes 1-5 reconstruction
# ==========================================================
t_days = (1:nt) ./ 24   # hourly samples -> days


n_days_show = 30
n_show = min(nt, n_days_show*24)
show_idx = 1:n_show


fig = Figure(size = (1000, 700))


ax1 = Axis(fig[1, 1], xlabel = "Time (days)", ylabel = "u' (m/s)",
           title = "Baroclinic u' vs sum of modes 1-25 (time-mean N2), depth ≈ $(round(z_vel_centers[idx_vel], digits=0)) m")
lines!(ax1, t_days[show_idx], u_actual[show_idx],     color = :black, label = "BC (full)")
lines!(ax1, t_days[show_idx], u_recon_sum[show_idx],  color = :red,   label = "Sum modes 1-5")
axislegend(ax1, position = :rt)


ax2 = Axis(fig[2, 1], xlabel = "Time (days)", ylabel = "v' (m/s)",
           title = "Baroclinic v' vs sum of modes 1-25 (time-mean N2), depth ≈ $(round(z_vel_centers[idx_vel], digits=0)) m")
lines!(ax2, t_days[show_idx], v_actual[show_idx],     color = :black, label = "BC (full)")
lines!(ax2, t_days[show_idx], v_recon_sum[show_idx],  color = :blue,  label = "Sum modes 1-5")
axislegend(ax2, position = :rt)


outdir = joinpath(base, "3day_mean", "N2")
mkpath(outdir)
figfile = joinpath(outdir, "SL_projection_meanN2_sum5modes_$(suffix)_i$(i_local)_j$(j_local).png")
#save(figfile, fig)
#println("\nSaved figure to $figfile")


fig

