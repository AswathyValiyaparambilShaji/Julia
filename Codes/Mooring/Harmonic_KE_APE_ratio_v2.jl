using DSP, Statistics, Printf, LinearAlgebra, TOML, NCDatasets, CairoMakie


include(joinpath(@__DIR__, "..", "..", "functions", "densjmd95.jl"))
include(joinpath(@__DIR__, "..", "..", "functions", "strum_liouville_noneqDZ_norm.jl"))
include(joinpath(@__DIR__, "..", "..", "functions", "harmonic03.jl"))
include(joinpath(@__DIR__, "..", "..", "functions", "coriolis_frequency.jl"))   # <- adjust path if needed


config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "config", "run_debug.toml"))
cfg    = TOML.parsefile(config_file)
FIGDIR = cfg["fig_base_m"]


g, rho0      = 9.81, 1027.0
NZ           = 173
n_modes_keep = 10
om           = 2π / (12.42 * 3600)   # M2 frequency [rad/s]
dt_hours     = 1.0                   # mooring sampling interval [hr]
#mydir        = "/home/aswathy/mnt/data/aswathy/MITgcm_NAS/Moorings/"
mydir        = "/nobackup/avaliyap/V2/Moorings/"


logfile = joinpath(FIGDIR, "run_log_KEAPE_harmonic.txt")
logio = open(logfile, "w")
redirect_stdout(logio)
redirect_stderr(logio)


try


# ---- load mooring data ----
combined_file = joinpath(mydir, "Moorings_88_combined.nc")   # <- adjust name if different
ds = NCDataset(combined_file, "r")


thk = (open(joinpath(mydir, "delR.bin"), "r") do io
    raw = read(io, NZ * sizeof(Float32))
    ntoh.(reshape(reinterpret(Float32, raw), NZ))
end)
DRF = thk[1:NZ]


lon = Array(ds["lon"])
lat = Array(ds["lat"])


permute_to_std(x) = permutedims(Array(x), (2, 3, 1))
U     = Float64.(permute_to_std(ds["U_east"]))
V     = Float64.(permute_to_std(ds["V_north"]))
Salt  = Float64.(permute_to_std(ds["Salt"]))
Theta = Float64.(permute_to_std(ds["Theta"]))
hFacC = Float64.(Array(ds["hFacC"]))
close(ds)


N_moor, nz, nt = size(U)
println("Loaded $N_moor mooring points, $nz levels, $nt timesteps.")


mask2D  = hFacC .== 0
DRFfull = hFacC .* reshape(DRF, 1, nz)
depth   = sum(DRFfull, dims=2)


# ---- density at every timestep ----
z  = cumsum(DRFfull, dims=2)
zz = cat(zeros(N_moor, 1), z; dims=2)
za = -0.5 .* (zz[:, 1:end-1] .+ zz[:, 2:end])


rho = zeros(Float64, N_moor, nz, nt)
for t in 1:nt
    rho[:, :, t] = densjmd95(Salt[:, :, t], Theta[:, :, t], -za)
end


# ---- time-averaged N2: 3-day mean S,T -> N2 per chunk -> mean over chunks ----
timesteps_per_3days = 72
nt_avg = div(nt, timesteps_per_3days)
salt_3day  = zeros(Float32, N_moor, nz, nt_avg)
theta_3day = zeros(Float32, N_moor, nz, nt_avg)
for i in 1:nt_avg
    t_start = (i - 1) * timesteps_per_3days + 1
    t_end   = min(i * timesteps_per_3days, nt)
    salt_3day[:, :, i]  = mean(Salt[:, :, t_start:t_end],  dims=3)[:, :, 1]
    theta_3day[:, :, i] = mean(Theta[:, :, t_start:t_end], dims=3)[:, :, 1]
end


z_cumsum     = cumsum(DRFfull, dims=2)
zz2          = cat(zeros(N_moor, 1), z_cumsum; dims=2)
z_interfaces = -zz2[:, 2:end-1]
z_centers    = -0.5 .* (zz2[:, 1:end-1] .+ zz2[:, 2:end])
dz           = z_centers[:, 2:end] .- z_centers[:, 1:end-1]


N2 = zeros(Float64, N_moor, nz, nt_avg)
for t in 1:nt_avg
    S_t = salt_3day[:, :, t]
    T_t = theta_3day[:, :, t]
    rho_upper = densjmd95(S_t[:, 1:end-1], T_t[:, 1:end-1], z_interfaces)
    rho_lower = densjmd95(S_t[:, 2:end],   T_t[:, 2:end],   z_interfaces)
    N2[:, 1:end-1, t] = -(g / rho0) .* ((rho_lower .- rho_upper) ./ dz)
end
N2[N2 .< 0] .= NaN
N2[isnan.(N2)] .= 1e-10


# ---- Sturm-Liouville solve per mooring, 10 modes, time-averaged N2 ----
Ueig_out    = fill(NaN, N_moor, nz, n_modes_keep)
Weig_out    = fill(NaN, N_moor, nz, n_modes_keep)
N2_bar      = fill(NaN, N_moor)   # depth+time averaged N2 -- APE denominator
n_avail_out = fill(0, N_moor)     # modes actually solved at each mooring


for p in 1:N_moor
    f_pt = coriolis_frequency(lat[p])
    hfac_col  = hFacC[p, :]
    ocean_idx = findall(hfac_col .> 0)
    if isempty(ocean_idx)
        println("  mooring $p/$N_moor skipped (no ocean cells)")
        continue
    end
    k_top, ibot = ocean_idx[1], ocean_idx[end]


    N2_mean_col = [ (v = filter(!isnan, N2[p, k, :]); isempty(v) ? 1e-10 : mean(v)) for k in 1:nz ]


    dz_col   = (hfac_col .* DRF)[k_top:ibot]
    zf_cells = cumsum(dz_col)
    N2_cells = N2_mean_col[k_top:ibot]
    zf_col   = vcat(0.0, -zf_cells)
    N2_faces = vcat(1e-10, N2_cells)


    N2_bar[p] = sum(N2_cells .* dz_col) / sum(dz_col)


    k_sl, L_sl, C_sl, Cg_sl, Ce_sl, Weig_sl, Ueig_sl, Ueig2_sl =
        sturm_liouville_noneqDZ_norm(zf_col, N2_faces, f_pt, om, 0)


    n_avail = min(n_modes_keep, length(Ce_sl))
    n_avail_out[p] = n_avail
    Ueig_out[p, k_top:ibot, 1:n_avail] = Ueig2_sl[:, 1:n_avail]
    Weig_out[p, k_top:ibot, 1:n_avail] = Weig_sl[2:end, 1:n_avail]


    println("  mooring $p/$N_moor solved ($n_avail modes)")
end
println("Solved Sturm-Liouville modes (up to n=$n_modes_keep) at $N_moor moorings.")


# ---- baroclinic, unfiltered perturbations (harmonic03 isolates M2 itself) ----
DRFfull_r = reshape(DRFfull, N_moor, nz, 1)
depth_r   = reshape(depth,   N_moor, 1, 1)


ucA_3d = sum(U .* DRFfull_r, dims=2) ./ depth_r
up_3d  = U .- ucA_3d
vcA_3d = sum(V .* DRFfull_r, dims=2) ./ depth_r
vp_3d  = V .- vcA_3d


rho_mean = mean(rho, dims=3)
bp_3d    = (-g / rho0) .* (rho .- rho_mean)   # buoyancy perturbation b' = -g*rho'/rho0


mask3D = repeat(reshape(mask2D, N_moor, nz, 1), 1, 1, nt)
up_3d[mask3D] .= 0
vp_3d[mask3D] .= 0
bp_3d[mask3D] .= 0


# ---- project u,v onto U-modes, b onto W-modes (only modes solved per mooring) ----
uhat_out = fill(NaN, N_moor, nt, n_modes_keep)
vhat_out = fill(NaN, N_moor, nt, n_modes_keep)
bhat_out = fill(NaN, N_moor, nt, n_modes_keep)


for p in 1:N_moor
    n_avail = n_avail_out[p]
    n_avail == 0 && continue


    hfac_col  = hFacC[p, :]
    ocean_idx = findall(hfac_col .> 0)
    k_top, ibot = ocean_idx[1], ocean_idx[end]


    Phi_all = @view Ueig_out[p, k_top:ibot, 1:n_avail]
    Psi_all = @view Weig_out[p, k_top:ibot, 1:n_avail]
    (any(isnan, Phi_all) || any(isnan, Psi_all)) && continue


    dz_col = (hfac_col .* DRF)[k_top:ibot]
    H = sum(dz_col)


    u_prof = @view up_3d[p, k_top:ibot, :]
    v_prof = @view vp_3d[p, k_top:ibot, :]
    b_prof = @view bp_3d[p, k_top:ibot, :]


    uhat_out[p, :, 1:n_avail] = (1 / H) .* (u_prof' * (Phi_all .* dz_col))
    vhat_out[p, :, 1:n_avail] = (1 / H) .* (v_prof' * (Phi_all .* dz_col))
    bhat_out[p, :, 1:n_avail] = (1 / H) .* (b_prof' * (Psi_all .* dz_col))
end
println("Modal projection complete.")


# ---- M2 harmonic fit of every modal time series (harmonic03) ----
t_days   = collect(0:nt-1) .* dt_hours ./ 24.0
freq_sel = [46]     # M2 in frequencies_L2()


AmpU = fill(NaN, N_moor, n_modes_keep)
AmpV = fill(NaN, N_moor, n_modes_keep)
AmpB = fill(NaN, N_moor, n_modes_keep)


for p in 1:N_moor
    n_avail_out[p] == 0 && continue


    u_modes = permutedims(@view uhat_out[p, :, :])
    v_modes = permutedims(@view vhat_out[p, :, :])
    b_modes = permutedims(@view bhat_out[p, :, :])


    _, au, bu, _, _, _, _ = harmonic03(t_days, u_modes, freq_sel)
    _, av, bv, _, _, _, _ = harmonic03(t_days, v_modes, freq_sel)
    _, ab, bb, _, _, _, _ = harmonic03(t_days, b_modes, freq_sel)


    AmpU[p, :] = sqrt.(vec(au).^2 .+ vec(bu).^2)
    AmpV[p, :] = sqrt.(vec(av).^2 .+ vec(bv).^2)
    AmpB[p, :] = sqrt.(vec(ab).^2 .+ vec(bb).^2)
end
println("M2 harmonic analysis complete.")


# ---- modal KE, APE, ratio: <x^2> = 0.5*Amp^2 for one harmonic ----
KE_out     = fill(NaN, N_moor, n_modes_keep)
APE_out    = fill(NaN, N_moor, n_modes_keep)
ratio_out  = fill(NaN, N_moor, n_modes_keep)
theory_out = fill(NaN, N_moor)


for p in 1:N_moor
    f_p = abs(coriolis_frequency(lat[p]))
    theory_out[p] = om > f_p ? (om^2 + f_p^2) / (om^2 - f_p^2) : NaN   # same for every mode
    for n in 1:n_modes_keep
        isnan(AmpU[p, n]) && continue
        KE_out[p, n]    = 0.5 * rho0 * 0.5 * (AmpU[p, n]^2 + AmpV[p, n]^2)
        APE_out[p, n]   = 0.5 * rho0 * 0.5 * (AmpB[p, n]^2) / N2_bar[p]
        ratio_out[p, n] = KE_out[p, n] / APE_out[p, n]
    end
end
println("Modal KE/APE ratio complete.")


# ---- save ----
outfile = joinpath(mydir, "Mooring_modal_KEAPE_harmonic.nc")
dso = NCDataset(outfile, "c")
defDim(dso, "station", N_moor)
defDim(dso, "mode", n_modes_keep)
v = defVar(dso, "lat", Float64, ("station",)); v[:] = lat
v = defVar(dso, "lon", Float64, ("station",)); v[:] = lon
v = defVar(dso, "N2_bar", Float64, ("station",)); v[:] = N2_bar
v = defVar(dso, "theoretical_ratio", Float64, ("station",)); v[:] = theory_out
v = defVar(dso, "KE_modal", Float64, ("station", "mode")); v[:] = KE_out
v = defVar(dso, "APE_modal", Float64, ("station", "mode")); v[:] = APE_out
v = defVar(dso, "ratio_modal", Float64, ("station", "mode")); v[:] = ratio_out
close(dso)
println("Saved -> $outfile")


# ---- plot: ratio vs mode across all moorings, vs theoretical range ----
modes = 1:n_modes_keep
ratio_mean = [ (vv = filter(!isnan, ratio_out[:, n]); isempty(vv) ? NaN : mean(vv)) for n in modes ]
ratio_std  = [ (vv = filter(!isnan, ratio_out[:, n]); isempty(vv) ? NaN : std(vv))  for n in modes ]
theory_valid = filter(!isnan, theory_out)
theory_lo, theory_hi = isempty(theory_valid) ? (NaN, NaN) : extrema(theory_valid)


fig = Figure(resolution = (800, 600), backgroundcolor = :white)
ax = Axis(fig[1, 1], xlabel = "Vertical mode", ylabel = "KE / APE",
    title = "Modal KE/APE ratio -- M2 harmonic fit, $N_moor moorings",
    xticks = collect(modes))
if !isnan(theory_lo)
    hspan!(ax, theory_lo, theory_hi, color = (:darkorange, 0.25), label = "theoretical range (M2)")
end
errorbars!(ax, collect(modes), ratio_mean, ratio_std, color = :seagreen, whiskerwidth = 8)
scatterlines!(ax, collect(modes), ratio_mean, color = :seagreen, linewidth = 2, markersize = 10,
    label = "observed (mean +/- 1 sigma)")
axislegend(ax, position = :rt)
display(fig)
outpng = joinpath(FIGDIR, "Mooring_modal_KEAPE_ratio_harmonic.png")
save(outpng, fig)
println("Saved -> $outpng")


catch e
    println(logio, "\n==================== UNCAUGHT ERROR ====================")
    println(logio, sprint(showerror, e, catch_backtrace()))
    println(logio, "==========================================================")
    flush(logio)
    rethrow()
finally
    flush(logio)
    close(logio)
end



