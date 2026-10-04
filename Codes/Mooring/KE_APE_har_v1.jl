using Statistics, Printf, LinearAlgebra, TOML, NCDatasets, CairoMakie, Dates
include(joinpath(@__DIR__, "..", "..", "functions", "harmonic03.jl"))      # adjust path if needed
include(joinpath(@__DIR__, "..", "..", "functions", "densjmd95.jl"))
include(joinpath(@__DIR__, "..", "..", "functions", "strum_liouville_noneqDZ_norm.jl"))


config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "config", "run_debug.toml"))
cfg    = TOML.parsefile(config_file)
FIGDIR = cfg["fig_base_m"]


logfile = joinpath(FIGDIR, "run_log_M2harmonic.txt")
logio = open(logfile, "w")
redirect_stdout(logio)
redirect_stderr(logio)


g    = 9.81
rho0 = 1027.0
delt = 1.0                    # sampling interval [hours]
timesteps_per_3days = 72
NZ   = 173
n_modes_keep = 5


# ============================================================================
# LOAD DATA (unchanged)
# ============================================================================
mydir  = "/home/aswathy/mnt/data/aswathy/MITgcm_NAS/Moorings/"
ncfile = joinpath(mydir, "Moorings_88_timeseries.nc")
ds = NCDataset(ncfile, "r")
thk = (open(joinpath(mydir, "delR.bin"), "r") do io
          raw = read(io, NZ * sizeof(Float32))
          ntoh.(reshape(reinterpret(Float32, raw), NZ))
      end)
DRF = thk[1:NZ]


lon = Array(ds["lon"])
lat = Array(ds["lat"])
permute_to_std(x) = permutedims(Array(x), (2, 3, 1))      # -> (station, depth, time)
U     = Float64.(permute_to_std(ds["U_east"]))
V     = Float64.(permute_to_std(ds["V_north"]))
Salt  = Float64.(permute_to_std(ds["Salt"]))
Theta = Float64.(permute_to_std(ds["Theta"]))
hFacC = Float64.(Array(ds["hFacC"]))                      # (station, depth)


# ---- Time axis (needed for the harmonic fit) ----
time_raw = Array(ds["time"])                                  # Vector{DateTime}, hourly with gaps
xt = Dates.value.(time_raw .- time_raw[1]) ./ 86_400_000      # days since first sample
close(ds)
N_moor, nz, nt = size(U)
println("Loaded $N_moor mooring points, $nz levels, $nt timesteps.")


# ============================================================================
# STEP 0: CHECK MOORING LOCATIONS (unchanged)
# ============================================================================
fig0 = Figure(resolution=(800, 700))
ax0 = Axis(fig0[1, 1], xlabel="Longitude [°]", ylabel="Latitude [°]",
           title="Mooring locations check ($N_moor points)", aspect=DataAspect())
scatter!(ax0, lon, lat, color=:dodgerblue, markersize=10)
for p in 1:N_moor
    text!(ax0, lon[p] + 0.02, lat[p] + 0.02; text=string(p), fontsize=9, color=:black)
end
save(joinpath(FIGDIR, "mooring_locations_check.png"), fig0)


hFacC_moor = hFacC
mask2D  = hFacC_moor .== 0
DRFfull = hFacC_moor .* reshape(DRF, 1, nz)                # (N_moor, nz) cell thickness


# ============================================================================
# DENSITY (unchanged)
# ============================================================================
z  = cumsum(DRFfull, dims=2)
zz = cat(zeros(N_moor, 1), z; dims=2)
za = -0.5 .* (zz[:, 1:end-1] .+ zz[:, 2:end])
rho = zeros(Float64, N_moor, nz, nt)
for t in 1:nt
    rho[:, :, t] = densjmd95(Salt[:, :, t], Theta[:, :, t], -za)
end


# ============================================================================
# STEP 1: HARMONIC ANALYSIS -> complex M2 amplitude at every depth
#   harmonic03 fits  x(t) = A0 + Σ_j [a_j cos(ω_j t) + b_j sin(ω_j t)]
#   M2 part written as  Re[ Z e^{iωt} ]  with  Z = a − i b
#   |Z| = amplitude, phase lag = -angle(Z).  No time-series reconstruction needed.
#   A0 (time mean) is discarded -> this replaces the bandpass filter.
# ============================================================================
fq_all   = frequencies_L2()
freq_sel = [46]                          # M2 only (index 46 in the L2 table)
iM2      = 1                             # M2 is the only (first) constituent in the output
om       = fq_all[46] / 86400            # M2 frequency [rad/s]


# xt (days since first sample) was read from the file above
length(xt) == nt || error("time length $(length(xt)) ≠ nt = $nt")
dt_h = diff(xt) .* 24
println("Record length: $(round(xt[end] - xt[1], digits=2)) days; ",
        "sampling interval: (round(minimum(dth),digits=3))–(round(maximum(dt_h), digits=3)) h")
gaps = findall(dt_h .> delt + 1e-3)
println("Number of gaps in time axis: $(length(gaps)); missing samples ≈ ",
        round(Int, sum(dt_h[gaps] .- delt) / delt))
for k in gaps
    println("  gap: ", time_raw[k], " -> ", time_raw[k+1], "  ($(round(dt_h[k], digits=1)) h)")
end
# Gaps are fine for the harmonic fit: it uses the actual times in xt.


function m2_complex(X, xt, freq_sel, iM2)          # X: (nz, nt) -> Z: (nz)
    _, a, b, _, R2, _, _ = harmonic03(xt, X, freq_sel)
    Z = complex.(a[iM2, :], -b[iM2, :])
    Z[isnan.(Z)] .= 0                               # land / empty bins
    return Z, R2
end


Uc = zeros(ComplexF64, N_moor, nz)
Vc = zeros(ComplexF64, N_moor, nz)
Rc = zeros(ComplexF64, N_moor, nz)
R2_U = fill(NaN, N_moor)
for p in 1:N_moor
    Uc[p, :], r2 = m2_complex(U[p, :, :],   xt, freq_sel, iM2)
    Vc[p, :], _  = m2_complex(V[p, :, :],   xt, freq_sel, iM2)
    Rc[p, :], _  = m2_complex(rho[p, :, :], xt, freq_sel, iM2)
    r2g = filter(!isnan, r2)
    R2_U[p] = isempty(r2g) ? NaN : median(r2g)
end
println("Harmonic fit done. Median R² of U per mooring (M2 fit):")
println(round.(R2_U, digits=2))


# ============================================================================
# STEP 2: M2 PRESSURE AMPLITUDE from M2 density amplitude (hydrostatic)
#   same integration as before, applied to the complex amplitude (linear)
# ============================================================================
Pc = zeros(ComplexF64, N_moor, nz)
for p in 1:N_moor
    dz    = DRFfull[p, :]
    p_bot = g .* cumsum(Rc[p, :] .* dz)            # bottom face of each cell
    p_top = vcat(0.0 + 0im, p_bot[1:end-1])        # top face
    Pc[p, :] = 0.5 .* (p_top .+ p_bot)             # cell centre
end


# No depth-mean removal needed: the solver's Ueig = (dW/dz)/k with W = 0 at
# surface and bottom, so ∫φ_n dz = 0 exactly and the projection in STEP 5
# removes the barotropic (depth-uniform) part of u, v and p automatically.


# ============================================================================
# STEP 3: 3-DAY AVERAGING + N2 (time-mean N2 profile used in STEP 4)
#   Bins are defined by the actual time (xt, days), not by sample count,
#   so a bin never straddles a data gap. Empty bins (inside a gap) are dropped.
# ============================================================================
bin_id = floor.(Int, xt ./ 3) .+ 1
bins   = [findall(==(b), bin_id) for b in unique(bin_id)]
bins   = filter(ix -> length(ix) >= timesteps_per_3days ÷ 2, bins)   # need ≥ 1.5 days of data
nt_avg = length(bins)
salt_3day  = zeros(Float64, N_moor, nz, nt_avg)
theta_3day = zeros(Float64, N_moor, nz, nt_avg)
for (i, ix) in enumerate(bins)
    salt_3day[:, :, i]  = mean(Salt[:, :, ix],  dims=3)[:, :, 1]
    theta_3day[:, :, i] = mean(Theta[:, :, ix], dims=3)[:, :, 1]
end
println("3-day bins used for N²: $nt_avg")


zz2          = cat(zeros(N_moor, 1), cumsum(DRFfull, dims=2); dims=2)
z_centers    = -0.5 .* (zz2[:, 1:end-1] .+ zz2[:, 2:end])
z_interfaces = -zz2[:, 2:end-1]
dz_c         = z_centers[:, 2:end] .- z_centers[:, 1:end-1]


N2 = zeros(Float64, N_moor, nz, nt_avg)
for t in 1:nt_avg
    S_t = salt_3day[:, :, t];  T_t = theta_3day[:, :, t]
    rho_upper = densjmd95(S_t[:, 1:end-1], T_t[:, 1:end-1], z_interfaces)
    rho_lower = densjmd95(S_t[:, 2:end],   T_t[:, 2:end],   z_interfaces)
    N2[:, 1:end-1, t] = -(g / rho0) .* ((rho_lower .- rho_upper) ./ dz_c)
end
N2[N2 .< 0] .= NaN
N2[isnan.(N2)] .= 1e-10


# ============================================================================
# STEP 4: STURM-LIOUVILLE PER MOORING (om = M2 from the L2 table)
# ============================================================================
Ce_out   = fill(NaN, N_moor, n_modes_keep)
Cg_out   = fill(NaN, N_moor, n_modes_keep)
L_out    = fill(NaN, N_moor, n_modes_keep)
Ueig_out = fill(NaN, N_moor, nz, n_modes_keep)
Weig_out = fill(NaN, N_moor, nz, n_modes_keep)


for p in 1:N_moor
    f_pt = 2 * 7.2921e-5 * sin(deg2rad(lat[p]))
    hfac_col  = hFacC_moor[p, :]
    ocean_idx = findall(hfac_col .> 0)
    if isempty(ocean_idx)
        println("  mooring p/N_moor skipped (land)")
        continue
    end
    k_top, ibot = ocean_idx[1], ocean_idx[end]
    n_cells = ibot - k_top + 1


    N2_mean_col = [(v = filter(!isnan, N2[p, k, :]); isempty(v) ? 1e-10 : mean(v)) for k in 1:nz]
    dz_col   = (hfac_col .* DRF)[k_top:ibot]
    zf_col   = vcat(0.0, -cumsum(dz_col))
    N2_faces = vcat(1e-10, N2_mean_col[k_top:ibot])


    k_sl, L_sl, C_sl, Cg_sl, Ce_sl, Weig_sl, Ueig_sl, Ueig2_sl =
        sturm_liouville_noneqDZ_norm(zf_col, N2_faces, f_pt, om, 0)


    if size(Weig_sl, 1) != n_cells + 1 || size(Ueig2_sl, 1) != n_cells
        error("Mooring $p: unexpected eigenfunction sizes from solver.")
    end


    n_avail = min(n_modes_keep, length(Ce_sl))
    Ce_out[p, 1:n_avail] = Ce_sl[1:n_avail]
    Cg_out[p, 1:n_avail] = Cg_sl[1:n_avail]
    L_out[p, 1:n_avail]  = L_sl[1:n_avail]
    Ueig_out[p, k_top:ibot, 1:n_avail] = Ueig2_sl[:, 1:n_avail]
    Weig_out[p, k_top:ibot, 1:n_avail] = Weig_sl[2:end, 1:n_avail]
end


# Eigenspeed c_n and Coriolis at each mooring
Ω      = 7.2921e-5
f_moor = 2Ω .* sind.(lat)
cn_out = copy(Ce_out)        # solver: Ce = sqrt(1/eigenvalue) = eigenspeed c_n


# ============================================================================
# STEP 5: PROJECT M2 AMPLITUDE PROFILES ONTO MODES
#   Same formula as for the bandpass time series, but applied to ONE complex
#   number per depth instead of nt real numbers:
#     u_n = (1/H) ∫ u(z) φ_n(z) dz     (complex -> amplitude AND phase of mode n)
# ============================================================================
Un = fill(complex(NaN, NaN), N_moor, n_modes_keep)
Vn = fill(complex(NaN, NaN), N_moor, n_modes_keep)
Pn = fill(complex(NaN, NaN), N_moor, n_modes_keep)
Hm = fill(NaN, N_moor)
for p in 1:N_moor
    ocean_idx = findall(hFacC_moor[p, :] .> 0)
    isempty(ocean_idx) && continue
    k_top, ibot = ocean_idx[1], ocean_idx[end]
    Phi = Ueig_out[p, k_top:ibot, :]                 # (ncell, n_modes)
    any(isnan, Phi) && continue
    dz_col = (hFacC_moor[p, :] .* DRF)[k_top:ibot]
    H = sum(dz_col);  Hm[p] = H
    W = Phi .* dz_col
    Un[p, :] = transpose(W) * Uc[p, k_top:ibot] ./ H
    Vn[p, :] = transpose(W) * Vc[p, k_top:ibot] ./ H
    Pn[p, :] = transpose(W) * Pc[p, k_top:ibot] ./ H
end


# ============================================================================
# STEP 6: M2 MODAL ENERGY & FLUX  (time-mean; <cos²> = 1/2 gives the extra 1/2)
#   KE_n  = (rho0*H/4) * (|u_n|² + |v_n|²)        [J/m²]
#   APE_n = (H/4) * |p_n|² / (rho0 * c_n²)          [J/m²]
#   F_n   = (H/2) * Re(u_n * conj(p_n))            [W/m]
# ============================================================================
KE_M2 = (rho0 .* Hm ./ 4) .* (abs2.(Un) .+ abs2.(Vn)) ./ 1000      # kJ/m²  (N_moor, n_modes)
PE_M2 = (Hm ./ 4) .* abs2.(Pn) ./ (rho0 .* cn_out.^2) ./ 1000      # kJ/m²
E_M2  = KE_M2 .+ PE_M2
Fx_M2 = (Hm ./ 2) .* real.(Un .* conj.(Pn)) ./ 1000                # kW/m
Fy_M2 = (Hm ./ 2) .* real.(Vn .* conj.(Pn)) ./ 1000


# ============================================================================
# STEP 7: KE/APE vs FREE-WAVE THEORY  (ω²+f²)/(ω²−f²) with ω = M2
# ============================================================================
ratio_theory = (om^2 .+ f_moor.^2) ./ (om^2 .- f_moor.^2)
ratio_theory[abs.(f_moor) .>= om] .= NaN


ratio_obs = KE_M2 ./ PE_M2
ratio_obs[.!(KE_M2 .> 0) .| .!(PE_M2 .> 0)] .= NaN
dev = ratio_obs ./ ratio_theory                     # 1 = free wave


ratio_obs_sum = vec(sum(KE_M2, dims=2) ./ sum(PE_M2, dims=2))
dev_sum       = ratio_obs_sum ./ ratio_theory


println("\nM2 (harmonic) KE/APE")
println("Mooring |   lat  |  f/ω  | theory |  mode1  mode2  mode3 | mode-sum | E1 [kJ/m²]")
for p in 1:N_moor
    r = ratio_obs[p, 1:3]
    @printf("%7d | %6.2f | %5.3f | %6.3f | %6.2f %6.2f %6.2f | %8.2f | %8.3f\n",
            p, lat[p], f_moor[p]/om, ratio_theory[p], r..., ratio_obs_sum[p], E_M2[p, 1])
end


# ============================================================================
# OPTIONAL CHECK: rebuild mode-1 M2 time series at one mooring and use the
# bandpass-style formula mean((rho0*H/2)(u²+v²)). Should match KE_M2[p,1].
# ============================================================================
p_chk = findfirst(!isnan, Hm)
if !isnothing(p_chk)
    tsec = xt .* 86400
    u_t  = real.(Un[p_chk, 1] .* exp.(im .* om .* tsec))
    v_t  = real.(Vn[p_chk, 1] .* exp.(im .* om .* tsec))
    KE_ts = mean((rho0 * Hm[p_chk] / 2) .* (u_t.^2 .+ v_t.^2)) / 1000
    @printf("\nCheck mooring %d mode 1: KE from amplitudes = %.4f, from rebuilt series = %.4f kJ/m²\n",
            p_chk, KE_M2[p_chk, 1], KE_ts)
end


close(logio)




# ============================================================================
# STEP 8: PLOT KE/APE vs LATITUDE  (theory = line, moorings = points)
#   Moorings are plotted at their latitude, irrespective of longitude.
# ============================================================================
lat_line   = range(minimum(lat) - 0.5, maximum(lat) + 0.5, length=300)
f_line     = 2Ω .* sind.(lat_line)
ratio_line = (om^2 .+ f_line.^2) ./ (om^2 .- f_line.^2)
ratio_line[abs.(f_line) .>= om] .= NaN              # beyond M2 critical latitude


cols = Makie.wong_colors()
fig8 = Figure(resolution=(700, 800))
ax8  = Axis(fig8[1, 1], xlabel="KE / APE", ylabel="Latitude [°]",
            title="M2 KE/APE ratio vs latitude")
lines!(ax8, ratio_line, collect(lat_line), color=:black, linewidth=2.5,
       label="Theory (ω²+f²)/(ω²−f²)")
scatter!(ax8, ratio_obs[:, 1], lat, color=cols[1], marker=:circle,    markersize=9, label="Mode 1")
scatter!(ax8, ratio_obs[:, 2], lat, color=cols[2], marker=:utriangle, markersize=9, label="Mode 2")
#scatter!(ax8, ratio_obs_sum,   lat, color=cols[3], marker=:diamond,   markersize=9, label="Modes 1–$n_modes_keep combined")
         xlims!(ax8,0,20)
axislegend(ax8, position=:rb)
display(fig8)
# xlims!(ax8, 0, 5)                                 # uncomment if a few outliers squash the plot
save(joinpath(FIGDIR, "KE_APE_ratio_vs_lat_M2.png"), fig8)
println("Saved KE/APE vs latitude plot -> ", joinpath(FIGDIR, "KE_APE_ratio_vs_lat_M2.png"))



