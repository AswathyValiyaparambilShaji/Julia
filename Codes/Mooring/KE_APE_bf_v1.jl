using DSP, Statistics, Printf, LinearAlgebra, TOML, NCDatasets, Impute, CairoMakie
include(joinpath(@__DIR__, "..", "..",  "functions", "FluxUtils.jl"))
using .FluxUtils: bandpassfilter
include(joinpath(@__DIR__, "..", "..", "functions", "densjmd95.jl"))
include(joinpath(@__DIR__, "..","..", "functions", "strum_liouville_noneqDZ_norm.jl"))


config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__,  "..", "..", "config", "run_debug.toml"))
cfg    = TOML.parsefile(config_file)
FIGDIR = cfg["fig_base_m"]


logfile = joinpath(FIGDIR, "run_log.txt")
logio = open(logfile, "w")
redirect_stdout(logio)
redirect_stderr(logio)   # also captures @warn and error messages


g    = 9.81
rho0 = 1027.0
T1, T2, delt, N = 9.0, 15.0, 1.0, 4
timesteps_per_3days = 72


NZ = 173


mydir  = "/home/aswathy/mnt/data/aswathy/MITgcm_NAS/Moorings/" # "/nobackup/avaliyap/V2/Moorings/"
ncfile = joinpath(mydir, "Moorings_88_timeseries.nc")
ds = NCDataset(ncfile, "r")
thk = (open(joinpath(mydir, "delR.bin"), "r") do io
           raw = read(io, NZ * sizeof(Float32))
           ntoh.(reshape(reinterpret(Float32, raw), NZ))
       end)


DRF = thk[1:NZ]
sum(thk)


lon = Array(ds["lon"])
lat = Array(ds["lat"])


# written as (time, station, depth); pipeline below needs (station, depth, time)
permute_to_std(x) = permutedims(Array(x), (2, 3, 1))
U     = Float64.(permute_to_std(ds["U_east"]))
V     = Float64.(permute_to_std(ds["V_north"]))
Salt  = Float64.(permute_to_std(ds["Salt"]))
Theta = Float64.(permute_to_std(ds["Theta"]))
hFacC = Float64.(Array(ds["hFacC"]))   # (station, depth)


close(ds)
N_moor, nz, nt = size(U)
println("Loaded $N_moor mooring points, $nz levels, $nt timesteps.")


# ============================================================================
# STEP 0: CHECK MOORING LOCATIONS
# ============================================================================
fig0 = Figure(resolution=(800, 700))
ax0 = Axis(fig0[1, 1], xlabel="Longitude [°]", ylabel="Latitude [°]",
           title="Mooring locations check ($N_moor points)", aspect=DataAspect())
scatter!(ax0, lon, lat, color=:dodgerblue, markersize=10)
for p in 1:N_moor
    text!(ax0, lon[p] + 0.02, lat[p] + 0.02; text=string(p), fontsize=9, color=:black)
end
loc_png = joinpath(FIGDIR, "mooring_locations_check.png")
save(loc_png, fig0)
println("Saved mooring location check -> $loc_png")


hFacC_moor = hFacC
mask2D  = hFacC_moor .== 0
DRFfull = hFacC_moor .* reshape(DRF, 1, nz)


# ============================================================================
# DEPTH / PRESSURE PROXY & DENSITY (densjmd95) AT MOORING POINTS
# ============================================================================
z  = cumsum(DRFfull, dims=2)
zz = cat(zeros(N_moor, 1), z; dims=2)
za = -0.5 .* (zz[:, 1:end-1] .+ zz[:, 2:end])
rho = zeros(Float64, N_moor, nz, nt)
for t in 1:nt
    S_t = Salt[:, :, t]
    T_t = Theta[:, :, t]
    rho1 = densjmd95(S_t, T_t, -za)
    rho[:, :, t] = rho1
end


# ============================================================================
# BANDPASS FILTER U, V, RHO (time is last dim)
# ============================================================================
fu = bandpassfilter(U,   T1, T2, delt, N, nt)
fv = bandpassfilter(V,   T1, T2, delt, N, nt)
fr = bandpassfilter(rho, T1, T2, delt, N, nt)


# ============================================================================
# BC PRESSURE PERTURBATION
# ============================================================================
depth = sum(DRFfull, dims=2)
DRFfull_r = reshape(DRFfull, N_moor, nz, 1)
depth_r   = reshape(depth,   N_moor, 1, 1)
mask3D = repeat(reshape(mask2D, N_moor, nz, 1), 1, 1, nt)


pres  = g .* cumsum(fr .* DRFfull_r, dims=2)
pfz   = cat(zeros(N_moor, 1, nt), pres; dims=2)
pc_3d = 0.5 .* (pfz[:, 1:end-1, :] .+ pfz[:, 2:end, :])
pa    = sum(pc_3d .* DRFfull_r, dims=2) ./ depth_r
pp_3d = pc_3d .- pa
pp_3d[mask3D] .= 0


println("\nSanity check -- depth-integrated pp_3d (should be ≈ 0):")
println(round.(sum(pp_3d .* DRFfull_r, dims=2) ./ depth_r, digits=4))


# ============================================================================
# BC VELOCITY PERTURBATIONS
# ============================================================================
ucA_3d = sum(fu .* DRFfull_r, dims=2) ./ depth_r
up_3d  = fu .- ucA_3d
up_3d[mask3D] .= 0
vcA_3d = sum(fv .* DRFfull_r, dims=2) ./ depth_r
vp_3d  = fv .- vcA_3d
vp_3d[mask3D] .= 0


println("\nSanity check -- depth-integrated up_3d (should be ≈ 0):")
println(round.(sum(up_3d .* DRFfull_r, dims=2) ./ depth_r, digits=4))
println("Sanity check -- depth-integrated vp_3d (should be ≈ 0):")
println(round.(sum(vp_3d .* DRFfull_r, dims=2) ./ depth_r, digits=4))


# ============================================================================
# BC FLUXES (undecomposed, for reference)
# ============================================================================
xflx_3d = up_3d .* pp_3d
yflx_3d = vp_3d .* pp_3d
Fu_b  = dropdims(sum(xflx_3d .* DRFfull_r, dims=2), dims=2)
Fv_b  = dropdims(sum(yflx_3d .* DRFfull_r, dims=2), dims=2)
Fu_bc = dropdims(mean(Fu_b, dims=2), dims=2) ./ 1000
Fv_bc = dropdims(mean(Fv_b, dims=2), dims=2) ./ 1000


# ============================================================================
# 3-DAY AVERAGING (for N2)
# ============================================================================
nt_avg = div(nt, timesteps_per_3days)
U_3day     = zeros(Float32, N_moor, nz, nt_avg)
V_3day     = zeros(Float32, N_moor, nz, nt_avg)
salt_3day  = zeros(Float32, N_moor, nz, nt_avg)
theta_3day = zeros(Float32, N_moor, nz, nt_avg)
for i in 1:nt_avg
    t_start = (i - 1) * timesteps_per_3days + 1
    t_end   = min(i * timesteps_per_3days, nt)
    U_3day[:, :, i]     = mean(U[:, :, t_start:t_end], dims=3)[:, :, 1]
    V_3day[:, :, i]     = mean(V[:, :, t_start:t_end], dims=3)[:, :, 1]
    salt_3day[:, :, i]  = mean(Salt[:, :, t_start:t_end], dims=3)[:, :, 1]
    theta_3day[:, :, i] = mean(Theta[:, :, t_start:t_end], dims=3)[:, :, 1]
end


# ============================================================================
# N2 CALCULATION AT MOORING POINTS
# ============================================================================
z_cumsum     = cumsum(DRFfull, dims=2)
zz2          = cat(zeros(N_moor, 1), z_cumsum; dims=2)
z_centers    = -0.5 .* (zz2[:, 1:end-1] .+ zz2[:, 2:end])
z_interfaces = -zz2[:, 2:end-1]
dz           = z_centers[:, 2:end] .- z_centers[:, 1:end-1]


N2 = zeros(Float64, N_moor, nz, nt_avg)
println("Calculating N² at interfaces...")
for t in 1:nt_avg
    S_t = salt_3day[:, :, t]
    T_t = theta_3day[:, :, t]
    rho_upper = densjmd95(S_t[:, 1:end-1], T_t[:, 1:end-1], z_interfaces)
    rho_lower = densjmd95(S_t[:, 2:end],   T_t[:, 2:end],   z_interfaces)
    drho = rho_lower .- rho_upper
    N2_interfaces = -(g / rho0) .* (drho ./ dz)
    N2[:, 1:end-1, t] = N2_interfaces
end


println("Setting negative values to NaN...")
N2[N2 .< 0] .= NaN
n_nan_before = sum(isnan.(N2))
println("  Number of NaN values before filling: $n_nan_before")
println("Filling NaN values with 1e-10...")
N2[isnan.(N2)] .= 1e-10
println("N2 calculation complete for all mooring points.")


# ============================================================================
# SOLVE STURM-LIOUVILLE EQUATION PER MOORING POINT
# ============================================================================
n_modes_keep = 5
om = 2π / (12.42 * 3600)
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
        println("  mooring point p/N_moor skipped (no ocean cells -- land/dry column)")
        continue
    end
    k_top = ocean_idx[1]
    ibot  = ocean_idx[end]
    n_cells = ibot - k_top + 1


    N2_mean_col = [ (v = filter(!isnan, N2[p, k, :]); isempty(v) ? 1e-10 : mean(v))
                    for k in 1:nz ]


    dz_col   = (hfac_col .* DRF)[k_top:ibot]
    zf_cells = cumsum(dz_col)
    N2_cells = N2_mean_col[k_top:ibot]


    zf_col   = vcat(0.0, -zf_cells)
    N2_faces = vcat(1e-10, N2_cells)


    k_sl, L_sl, C_sl, Cg_sl, Ce_sl, Weig_sl, Ueig_sl, Ueig2_sl =
        sturm_liouville_noneqDZ_norm(zf_col, N2_faces, f_pt, om, 0)


    if size(Weig_sl, 1) != n_cells + 1 || size(Ueig2_sl, 1) != n_cells
        error("Mooring point $p: solver returned Weig_sl with $(size(Weig_sl,1)) rows " *
              "(expected $(n_cells+1)) and Ueig2_sl with $(size(Ueig2_sl,1)) rows " *
              "(expected $n_cells). Check sturm_liouville_noneqDZ_norm's convention.")
    end


    n_avail = min(n_modes_keep, length(Ce_sl))
    Ce_out[p, 1:n_avail] = Ce_sl[1:n_avail]
    Cg_out[p, 1:n_avail] = Cg_sl[1:n_avail]
    L_out[p, 1:n_avail]  = L_sl[1:n_avail]


    Ueig_out[p, k_top:ibot, 1:n_avail] = Ueig2_sl[:, 1:n_avail]
    Weig_out[p, k_top:ibot, 1:n_avail] = Weig_sl[2:end, 1:n_avail]


    println("  mooring point p/N_moor solved")
end


# ============================================================================
# PROJECT BC VELOCITY & PRESSURE PERTURBATIONS ONTO HORIZONTAL EIGENMODES
# ============================================================================
uhat_out = fill(NaN, N_moor, nt, n_modes_keep)
vhat_out = fill(NaN, N_moor, nt, n_modes_keep)
phat_out = fill(NaN, N_moor, nt, n_modes_keep)
for p in 1:N_moor
    hfac_col = hFacC_moor[p, :]
    ocean_idx = findall(hfac_col .> 0)
    if isempty(ocean_idx)
        continue
    end
    k_top = ocean_idx[1]
    ibot  = ocean_idx[end]
    Phi_all = @view Ueig_out[p, k_top:ibot, :]
    if any(isnan, Phi_all)
        continue
    end
    dz_col = (hfac_col .* DRF)[k_top:ibot]
    H = sum(dz_col)
    u_prof = @view up_3d[p, k_top:ibot, :]
    v_prof = @view vp_3d[p, k_top:ibot, :]
    p_prof = @view pp_3d[p, k_top:ibot, :]
    W = Phi_all .* dz_col
    uhat_out[p, :, :] = (1/H) .* (u_prof' * W)
    vhat_out[p, :, :] = (1/H) .* (v_prof' * W)
    phat_out[p, :, :] = (1/H) .* (p_prof' * W)
end


# ============================================================================
# MODAL BC FLUXES (time-averaged, depth-integrated, kW/m) -- per mode
# ============================================================================
uflux_avg_out = fill(NaN, N_moor, n_modes_keep)
vflux_avg_out = fill(NaN, N_moor, n_modes_keep)
uflux_int_out = fill(NaN, N_moor, n_modes_keep)
vflux_int_out = fill(NaN, N_moor, n_modes_keep)
for p in 1:N_moor
    hfac_col = hFacC_moor[p, :]
    ocean_idx = findall(hfac_col .> 0)
    if isempty(ocean_idx)
        continue
    end
    k_top = ocean_idx[1]
    ibot  = ocean_idx[end]
    dz_col = (hfac_col .* DRF)[k_top:ibot]
    H = sum(dz_col)
    uhat = @view uhat_out[p, :, :]
    vhat = @view vhat_out[p, :, :]
    phat = @view phat_out[p, :, :]
    if any(isnan, uhat) || any(isnan, phat)
        continue
    end
    uflux_modes = uhat .* phat
    vflux_modes = vhat .* phat
    uflux_avg_modes = vec(mean(uflux_modes, dims=1))
    vflux_avg_modes = vec(mean(vflux_modes, dims=1))
    uflux_avg_out[p, :] = uflux_avg_modes
    vflux_avg_out[p, :] = vflux_avg_modes
    uflux_int_out[p, :] = uflux_avg_modes .* (H / 1000)
    vflux_int_out[p, :] = vflux_avg_modes .* (H / 1000)
end
println("Modal flux calculation complete for all $N_moor mooring points.")


# ============================================================================
# EIGENSPEED c_n FOR EQ. (10):  c_n = sqrt(ω² − f²) / k_n
# Uses only Ce_out and Cg_out from the solver. Hydrostatic rotating waves:
#   if Ce = eigenspeed c_n   →  Cg/Ce = sqrt(1 − f²/ω²)   → c_n = Ce
#   if Ce = phase speed ω/k  →  Cg/Ce = 1 − f²/ω²         → c_n = sqrt(Ce·Cg)
# The test below decides which convention the solver uses.
# ============================================================================
Ω      = 7.2921e-5
f_moor = 2Ω .* sind.(lat)                         # (N_moor)
s_rot  = sqrt.(max.(1 .- (f_moor ./ om).^2, 0))   # sqrt(1 − f²/ω²)


r_gc  = Cg_out ./ Ce_out
good  = .!isnan.(r_gc)
err_eig   = median(abs.(r_gc[good] .- repeat(s_rot,   1, n_modes_keep)[good]))
err_phase = median(abs.(r_gc[good] .- repeat(s_rot.^2, 1, n_modes_keep)[good]))


if err_eig <= err_phase
    cn_out = copy(Ce_out)
    println("\nEigenspeed: Ce_out IS the eigenspeed c_n (Cg/Ce ≈ sqrt(1−f²/ω²), err=$(round(err_eig, sigdigits=3)))")
else
    cn_out = sqrt.(Ce_out .* Cg_out)
    println("\nEigenspeed: Ce_out is the PHASE speed; using c_n = sqrt(Ce·Cg) (err=$(round(err_phase, sigdigits=3)))")
end
println("Mode-1 c_n (m/s), first 5 moorings: ", round.(cn_out[1:min(5, N_moor), 1], digits=3))


# Normalization check, Eq. (7): ∫φ_n² dz / H should be ≈ 1
println("Normalization ∫φ²dz/H (should be ≈ 1):")
for n in 1:n_modes_keep
    vals = Float64[]
    for p in 1:N_moor
        ocean_idx = findall(hFacC_moor[p, :] .> 0)
        isempty(ocean_idx) && continue
        k_top, ibot = ocean_idx[1], ocean_idx[end]
        dz_col = (hFacC_moor[p, :] .* DRF)[k_top:ibot]
        phi = Ueig_out[p, k_top:ibot, n]
        any(isnan, phi) && continue
        push!(vals, sum(phi.^2 .* dz_col) / sum(dz_col))
    end
    isempty(vals) || @printf("  mode %d: %.3f – %.3f\n", n, minimum(vals), maximum(vals))
end


# ============================================================================
# MODAL KINETIC & AVAILABLE POTENTIAL ENERGY  (Eq. 10)
#   KE_n = (rho0*H/2) * (u_n² + v_n²)         [J/m²]
#   PE_n = (H/2) * p_n² / (rho0 * c_n²)       [J/m²]
#   pp_3d = g∫ρ'dz → Pa, so no extra rho0 factor
# ============================================================================
nedge = 24                          # hours trimmed at each end (filter transients)
tt    = (1 + nedge):(nt - nedge)


KE_t_out   = fill(NaN, N_moor, nt, n_modes_keep)
PE_t_out   = fill(NaN, N_moor, nt, n_modes_keep)
KE_avg_out = fill(NaN, N_moor, n_modes_keep)     # kJ/m²
PE_avg_out = fill(NaN, N_moor, n_modes_keep)     # kJ/m²
E_avg_out  = fill(NaN, N_moor, n_modes_keep)     # kJ/m²


for p in 1:N_moor
    ocean_idx = findall(hFacC_moor[p, :] .> 0)
    isempty(ocean_idx) && continue
    k_top, ibot = ocean_idx[1], ocean_idx[end]
    dz_col = (hFacC_moor[p, :] .* DRF)[k_top:ibot]
    H = sum(dz_col)


    uhat = @view uhat_out[p, :, :]      # (nt, n_modes)
    vhat = @view vhat_out[p, :, :]
    phat = @view phat_out[p, :, :]
    c_n  = cn_out[p, :]                 # (n_modes)
    (any(isnan, uhat) || any(isnan, phat) || any(isnan, c_n)) && continue


    KE = (rho0 * H / 2) .* (uhat.^2 .+ vhat.^2)
    PE = (H / 2) .* phat.^2 ./ (rho0 .* reshape(c_n.^2, 1, :))


    KE_t_out[p, :, :] = KE
    PE_t_out[p, :, :] = PE
    KE_avg_out[p, :]  = vec(mean(KE[tt, :], dims=1)) ./ 1000
    PE_avg_out[p, :]  = vec(mean(PE[tt, :], dims=1)) ./ 1000
    E_avg_out[p, :]   = KE_avg_out[p, :] .+ PE_avg_out[p, :]
end
println("Modal KE/APE calculation complete for all $N_moor mooring points.")


# ============================================================================
# KE/APE RATIO vs FREE-WAVE CONSISTENCY RELATION
#   (KE/APE)_theory = (ω² + f²) / (ω² − f²)
# ============================================================================
ratio_theory = (om^2 .+ f_moor.^2) ./ (om^2 .- f_moor.^2)
ratio_theory[abs.(f_moor) .>= om] .= NaN          # beyond M2 critical latitude


ratio_obs = KE_avg_out ./ PE_avg_out              # ratio of time means (N_moor, n_modes)
ratio_obs[.!(KE_avg_out .> 0) .| .!(PE_avg_out .> 0)] .= NaN
dev = ratio_obs ./ ratio_theory                   # 1 = free wave; >1 KE excess; <1 APE excess


# depth mean removed + rigid lid → all modes are baroclinic
modes_bc     = 1:n_modes_keep
ratio_obs_bc = vec(sum(KE_avg_out[:, modes_bc], dims=2) ./ sum(PE_avg_out[:, modes_bc], dims=2))
dev_bc       = ratio_obs_bc ./ ratio_theory


println("\nMooring |   lat  |  f/ω  | theory |  mode1  mode2  mode3 | mode-sum")
for p in 1:N_moor
    r = ratio_obs[p, 1:min(3, n_modes_keep)]
    @printf("%7d | %6.2f | %5.3f | %6.3f | %6.2f %6.2f %6.2f | %6.2f\n",
            p, lat[p], f_moor[p]/om, ratio_theory[p], r..., ratio_obs_bc[p])
end


# ============================================================================
# FIGURE 1: one panel per mooring — KE/APE per mode vs theory
# ============================================================================
modes_plot = 1:n_modes_keep
ncol = ceil(Int, sqrt(N_moor)); nrow = ceil(Int, N_moor / ncol)
fig1 = Figure(size = (260ncol, 210nrow), fontsize = 10)
for p in 1:N_moor
    i, j = fldmod1(p, ncol)
    ax = Axis(fig1[i, j], title = @sprintf("M%d (%.2f°N)", p, lat[p]),
              xlabel = "mode", ylabel = "KE/APE", yscale = log10,
              xticks = collect(modes_plot))
    y = ratio_obs[p, modes_plot]
    if all(isnan, y)
        hidedecorations!(ax); continue
    end
    isnan(ratio_theory[p]) || hlines!(ax, [ratio_theory[p]], color = :black, linestyle = :dash)
    scatterlines!(ax, collect(modes_plot), y, color = :crimson, markersize = 7)
end
Legend(fig1[nrow + 1, 1:ncol],
       [LineElement(color = :black, linestyle = :dash),
        [LineElement(color = :crimson), MarkerElement(color = :crimson, marker = :circle)]],
       ["free wave (ω²+f²)/(ω²−f²)", "model ⟨KE⟩/⟨APE⟩"],
       orientation = :horizontal, framevisible = false)
save(joinpath(FIGDIR, "KE_APE_ratio_per_mooring.png"), fig1, px_per_unit = 2)

display(fig1)

# ============================================================================
# FIGURE 2: all moorings on one axis — modes 1–3, mode sum, theory
# ============================================================================
fig2 = Figure(size = (1100, 420))
ax2 = Axis(fig2[1, 1], xlabel = "mooring", ylabel = "⟨KE⟩ / ⟨APE⟩", yscale = log10)
lines!(ax2, 1:N_moor, ratio_theory, color = :black, linewidth = 2, linestyle = :dash,
       label = "(ω²+f²)/(ω²−f²)")
for (n, c) in zip(1:min(3, n_modes_keep), [:crimson, :royalblue, :seagreen])
    scatter!(ax2, 1:N_moor, ratio_obs[:, n], color = c, markersize = 7, label = "mode $n")
end
scatter!(ax2, 1:N_moor, ratio_obs_bc, color = :black, marker = :diamond, markersize = 8,
         label = "modes 1–$n_modes_keep")
axislegend(ax2, position = :rt, framevisible = false)
save(joinpath(FIGDIR, "KE_APE_ratio_all_moorings.png"), fig2, px_per_unit = 2)
display(fig2)


# ============================================================================
# FIGURE 3: maps of log10(model/theory) — mode 1 and mode sum
# ============================================================================
lim = 1.0     # ±1 = factor-of-10 departure
fig3 = Figure(size = (1200, 520))
for (k, (vals, ttl)) in enumerate([(dev[:, 1], "mode 1"), (dev_bc, "modes 1–$n_modes_keep")])
    ax = Axis(fig3[1, 2k - 1], xlabel = "Longitude [°]", ylabel = "Latitude [°]",
              title = "log₁₀[(KE/APE)_model / (KE/APE)_theory], $ttl", aspect = DataAspect())
    sc = scatter!(ax, lon, lat, color = log10.(vals), colormap = :balance,
                  colorrange = (-lim, lim), markersize = 13,
                  strokewidth = 0.5, strokecolor = :black, nan_color = :lightgray)
    Colorbar(fig3[1, 2k], sc, label = "red: KE excess · blue: APE excess")
end
save(joinpath(FIGDIR, "KE_APE_deviation_map.png"), fig3, px_per_unit = 2)
println("Saved KE/APE figures to $FIGDIR")
display(fig3)

close(logio)

# ============================================================================
# FIGURE 4: KE/APE — model vs free-wave theory, mode 1 and mode 2
# ============================================================================
moor_idx = 1:N_moor
fig4 = Figure(size = (1200, 750), fontsize = 13)

for (row, n) in enumerate(1:2)
    ax = Axis(fig4[row, 1],
              ylabel = "⟨KE⟩ / ⟨APE⟩",
              xlabel = row == 2 ? "Mooring number" : "",
              yscale = log10,
              xticks = 0:5:N_moor,
              title  = @sprintf("Mode %d   (median model/theory = %.2f)",
                                n, median(filter(!isnan, dev[:, n]))))

    # reference band: within a factor of 2 of theory
    band!(ax, collect(moor_idx), ratio_theory ./ 2, ratio_theory .* 2,
          color = (:gray, 0.15), label = "theory ×/÷ 2")

    # theoretical (ω² + f²)/(ω² − f²)
    lines!(ax, collect(moor_idx), ratio_theory, color = :black, linewidth = 2,
           linestyle = :dash, label = "(ω² + f²)/(ω² − f²)")

    # evaluated model ratio
    scatterlines!(ax, collect(moor_idx), ratio_obs[:, n],
                  color = row == 1 ? :crimson : :royalblue,
                  markersize = 8, linewidth = 1,
                  label = "model ⟨KE⟩/⟨APE⟩, mode $n")

    xlims!(ax, 0, N_moor + 1)
    row == 1 && hidexdecorations!(ax, grid = false, ticks = false)
    axislegend(ax, position = :rt, framevisible = false, labelsize = 11)
end

# latitude along the top for reference
ax_top = Axis(fig4[1, 1], xaxisposition = :top, xlabel = "Latitude [°N]",
              xticks = (collect(moor_idx)[1:5:end], [@sprintf("%.1f", lat[p]) for p in moor_idx[1:5:end]]))
hideydecorations!(ax_top); hidespines!(ax_top)
linkxaxes!(ax_top, contents(fig4[1, 1])[1])

rowgap!(fig4.layout, 10)
save(joinpath(FIGDIR, "KE_APE_mode1_mode2_vs_theory.png"), fig4, px_per_unit = 2)
println("Saved KE/APE mode 1 & 2 comparison -> ",
        joinpath(FIGDIR, "KE_APE_mode1_mode2_vs_theory.png"))



display(fig4)




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
            title="Semi Diurnal KE/APE ratio vs latitude")
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



