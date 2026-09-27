
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
mydir        = "/home/aswathy/mnt/data/aswathy/MITgcm_NAS/Moorings/"
#mydir        = "/nobackup/avaliyap/V2/Moorings/"


p = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1   # <- mooring index to process


logfile = joinpath(FIGDIR, "run_log_KEAPE_harmonic_mooring$(p).txt")
logio = open(logfile, "w")
redirect_stdout(logio)
redirect_stderr(logio)


try


# ---- open file, read only what's needed for mooring p ----
combined_file = joinpath(mydir, "Moorings_88_combined.nc")   # <- adjust name if different
ds = NCDataset(combined_file, "r")


thk = (open(joinpath(mydir, "delR.bin"), "r") do io
    raw = read(io, NZ * sizeof(Float32))
    ntoh.(reshape(reinterpret(Float32, raw), NZ))
end)
DRF = thk[1:NZ]


lat_p = ds["lat"][p]
lon_p = ds["lon"][p]
hfac_col = Float64.(Array(ds["hFacC"][p, :]))   # (nz,)


nz = length(hfac_col)
nt = size(ds["U_east"], 1)   # just reads the dimension length, not the data
println("Mooring $p: $nz levels, $nt timesteps.")


ocean_idx = findall(hfac_col .> 0)
isempty(ocean_idx) && error("Mooring $p has no ocean cells -- nothing to do.")
k_top, ibot = ocean_idx[1], ocean_idx[end]


# ---- read only this mooring's slice from disk: (nz, nt) ----
U_p     = Float64.(permutedims(Array(ds["U_east"][:, p, :])))
V_p     = Float64.(permutedims(Array(ds["V_north"][:, p, :])))
Salt_p  = Float64.(permutedims(Array(ds["Salt"][:, p, :])))
Theta_p = Float64.(permutedims(Array(ds["Theta"][:, p, :])))
close(ds)


DRFfull_p = hfac_col .* DRF
depth_p   = sum(DRFfull_p)


# ---- density at every timestep ----
z_p  = cumsum(DRFfull_p)
zz_p = vcat(0.0, z_p)
za_p = -0.5 .* (zz_p[1:end-1] .+ zz_p[2:end])


rho_p = zeros(Float64, nz, nt)
for t in 1:nt
    rho_p[:, t] = densjmd95(Salt_p[:, t], Theta_p[:, t], -za_p)
end


# ---- time-averaged N2: 3-day mean S,T -> N2 per chunk -> mean over chunks ----
timesteps_per_3days = 72
nt_avg = div(nt, timesteps_per_3days)
salt_3day_p  = zeros(Float64, nz, nt_avg)
theta_3day_p = zeros(Float64, nz, nt_avg)
for i in 1:nt_avg
    t_start = (i - 1) * timesteps_per_3days + 1
    t_end   = min(i * timesteps_per_3days, nt)
    salt_3day_p[:, i]  = mean(Salt_p[:, t_start:t_end],  dims=2)
    theta_3day_p[:, i] = mean(Theta_p[:, t_start:t_end], dims=2)
end


z_interfaces_p = -zz_p[2:end-1]
z_centers_p    = -0.5 .* (zz_p[1:end-1] .+ zz_p[2:end])
dz_p           = z_centers_p[2:end] .- z_centers_p[1:end-1]


N2_p = zeros(Float64, nz, nt_avg)
for t in 1:nt_avg
    S_t = salt_3day_p[:, t]
    T_t = theta_3day_p[:, t]
    rho_upper = densjmd95(S_t[1:end-1], T_t[1:end-1], z_interfaces_p)
    rho_lower = densjmd95(S_t[2:end],   T_t[2:end],   z_interfaces_p)
    N2_p[1:end-1, t] = -(g / rho0) .* ((rho_lower .- rho_upper) ./ dz_p)
end
N2_p[.!(N2_p .> 0)] .= 1e-10   # floor anything not strictly positive (negatives, NaN, exact zero)
N2_mean_col = vec(mean(N2_p, dims=2))


# ---- Sturm-Liouville solve, time-averaged N2, up to 10 modes ----
f_pt = coriolis_frequency(lat_p)
dz_col   = DRFfull_p[k_top:ibot]
zf_cells = cumsum(dz_col)
N2_cells = N2_mean_col[k_top:ibot]
zf_col   = vcat(0.0, -zf_cells)
N2_faces = vcat(1e-10, N2_cells)
N2_bar = sum(N2_cells .* dz_col) / sum(dz_col)


_, _, _, _, Ce_sl, Weig_sl, _, Ueig2_sl =
    sturm_liouville_noneqDZ_norm(zf_col, N2_faces, f_pt, om, 0)


n_avail = min(n_modes_keep, length(Ce_sl))
Phi_all = Ueig2_sl[:, 1:n_avail]
Psi_all = Weig_sl[2:end, 1:n_avail]
println("Solved $n_avail modes for mooring $p.")


# ---- baroclinic, unfiltered perturbations (harmonic03 isolates M2 itself) ----
ucA = sum(U_p .* DRFfull_p, dims=1) ./ depth_p
up  = U_p .- ucA
vcA = sum(V_p .* DRFfull_p, dims=1) ./ depth_p
vp  = V_p .- vcA


rho_mean_p = mean(rho_p, dims=2)
bp = (-g / rho0) .* (rho_p .- rho_mean_p)   # buoyancy perturbation b' = -g*rho'/rho0


dry = hfac_col .== 0
up[dry, :] .= 0
vp[dry, :] .= 0
bp[dry, :] .= 0


# ---- project u,v onto U-modes, b onto W-modes ----
H = sum(dz_col)
u_prof = up[k_top:ibot, :]
v_prof = vp[k_top:ibot, :]
b_prof = bp[k_top:ibot, :]


uhat = (1 / H) .* (u_prof' * (Phi_all .* dz_col))   # (nt, n_avail)
vhat = (1 / H) .* (v_prof' * (Phi_all .* dz_col))
bhat = (1 / H) .* (b_prof' * (Psi_all .* dz_col))


# ---- M2 harmonic fit of each modal time series ----
t_days   = collect(0:nt-1) .* dt_hours ./ 24.0
freq_sel = [46]     # M2 in frequencies_L2()


_, au, bu, _, _, _, _ = harmonic03(t_days, permutedims(uhat), freq_sel)
_, av, bv, _, _, _, _ = harmonic03(t_days, permutedims(vhat), freq_sel)
_, ab, bb, _, _, _, _ = harmonic03(t_days, permutedims(bhat), freq_sel)


AmpU = sqrt.(vec(au).^2 .+ vec(bu).^2)
AmpV = sqrt.(vec(av).^2 .+ vec(bv).^2)
AmpB = sqrt.(vec(ab).^2 .+ vec(bb).^2)


# ---- modal KE, APE, ratio: <x^2> = 0.5*Amp^2 for one harmonic ----
KE_out    = fill(NaN, n_modes_keep)
APE_out   = fill(NaN, n_modes_keep)
ratio_out = fill(NaN, n_modes_keep)
for n in 1:n_avail
    KE_out[n]    = 0.5 * rho0 * 0.5 * (AmpU[n]^2 + AmpV[n]^2)
    APE_out[n]   = 0.5 * rho0 * 0.5 * (AmpB[n]^2) / N2_bar
    ratio_out[n] = KE_out[n] / APE_out[n]
end


f_p = abs(f_pt)
theory_ratio = om > f_p ? (om^2 + f_p^2) / (om^2 - f_p^2) : NaN   # same value for every mode
println("Modal KE/APE ratio complete for mooring $p.")


# ---- save ----
outfile = joinpath(mydir, "Mooring_$(p)_modal_KEAPE_harmonic.nc")
dso = NCDataset(outfile, "c")
defDim(dso, "mode", n_modes_keep)
v = defVar(dso, "lat", Float64, ()); v[:] = lat_p
v = defVar(dso, "lon", Float64, ()); v[:] = lon_p
v = defVar(dso, "N2_bar", Float64, ()); v[:] = N2_bar
v = defVar(dso, "theoretical_ratio", Float64, ()); v[:] = theory_ratio
v = defVar(dso, "KE_modal", Float64, ("mode",)); v[:] = KE_out
v = defVar(dso, "APE_modal", Float64, ("mode",)); v[:] = APE_out
v = defVar(dso, "ratio_modal", Float64, ("mode",)); v[:] = ratio_out
close(dso)
println("Saved -> $outfile")


# ---- plot: ratio vs mode for this mooring, vs theoretical value ----
modes = 1:n_modes_keep
fig = Figure(resolution = (800, 600), backgroundcolor = :white)
ax = Axis(fig[1, 1], xlabel = "Vertical mode", ylabel = "KE / APE",
    title = "Modal KE/APE ratio -- mooring $p (M2 harmonic fit)",
    xticks = collect(modes))
if !isnan(theory_ratio)
    hlines!(ax, [theory_ratio], color = :darkorange, linestyle = :dash, label = "theoretical (M2)")
end
scatterlines!(ax, collect(modes), ratio_out, color = :seagreen, linewidth = 2, markersize = 10,
    label = "observed")
axislegend(ax, position = :rt)
display(fig)
outpng = joinpath(FIGDIR, "Mooring_$(p)_modal_KEAPE_ratio_harmonic.png")
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




