# ============================================================================
# compare_M2_flux_methods.jl
#
# Compare the depth-integrated baroclinic energy flux at every mooring
# computed in two ways:
#
#   (A) BANDPASS : 9–15 h Butterworth bandpass of u, v, ρ  → time series
#                  → remove depth mean → modal projection → F_n = H <u_n p_n>
#   (B) HARMONIC : harmonic fit (M2, optionally more constituents) of u, v, ρ
#                  → complex amplitudes → modal projection → F_n = (H/2) Re(u_n p_n*)
#
# To make the comparison clean, both methods use the SAME
#   * density (densjmd95), same hydrostatic pressure integration
#   * N² profile (gap-aware 3-day bins)
#   * frequency ω for the Sturm–Liouville problem
#   * vertical modes φ_n (solved once, used by both)
# so any difference comes only from how the tidal signal is extracted.
#
# Outputs (in FIGDIR):
#   run_log_flux_compare.txt        – log with per-mooring tables + summary stats
#   flux_compare_table.csv          – Fx, Fy per mooring for both methods (total + modes)
#   flux_compare_scatter.png        – harmonic vs bandpass (Fx, Fy, |F|), 1:1 line
#   flux_compare_map_total.png      – flux vectors at mooring locations (total BC)
#   flux_compare_map_mode1.png      – flux vectors at mooring locations (mode 1)
#   flux_compare_ratio_vs_lat.png   – |F_harm| / |F_bp| vs latitude
# ============================================================================
using DSP, Statistics, Printf, LinearAlgebra, TOML, NCDatasets, CairoMakie, Dates, DelimitedFiles
include(joinpath(@__DIR__, "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: bandpassfilter
include(joinpath(@__DIR__, "..", "..", "functions", "harmonic03.jl"))
include(joinpath(@__DIR__, "..", "..", "functions", "densjmd95.jl"))
include(joinpath(@__DIR__, "..", "..", "functions", "strum_liouville_noneqDZ_norm.jl"))


config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "config", "run_debug.toml"))
cfg    = TOML.parsefile(config_file)
FIGDIR = cfg["fig_base_m"]


logfile = joinpath(FIGDIR, "run_log_flux_compare.txt")
logio = open(logfile, "w")
redirect_stdout(logio)
redirect_stderr(logio)


# ============================================================================
# SETTINGS
# ============================================================================
g    = 9.81
rho0 = 1027.0
NZ   = 173
n_modes_keep = 5
n_modes_show = 3                     # modes shown in tables / scatter plots
timesteps_per_3days = 72


# Bandpass settings (same as the bandpass script)
T1, T2, delt, Nord = 9.0, 15.0, 1.0, 4
nedge = 24                           # hours trimmed at each end of the bandpassed
                                     # series (filter transients). Set 0 to
                                     # reproduce the original bandpass flux exactly.


# Harmonic settings
fq_all   = frequencies_L2()          # rad/day
iM2_tab  = 46                        # M2 index in the L2 table
freq_sel = [iM2_tab]                 # M2 only. To make the harmonic result comparable
                                     # to the whole 9–15 h band, add the S2, N2, K2
                                     # indices of the L2 table here; fluxes are then
                                     # summed over all selected constituents.
nc       = length(freq_sel)


# ONE frequency for the modes, used by BOTH methods
om = fq_all[iM2_tab] / 86400         # M2 [rad/s]
Ω  = 7.2921e-5


# ============================================================================
# LOAD DATA
# ============================================================================
mydir  = "/home/aswathy/mnt/data/aswathy/MITgcm_NAS/Moorings/"
ncfile = joinpath(mydir, "Moorings_88_timeseries.nc")
ds = NCDataset(ncfile, "r")
thk = open(joinpath(mydir, "delR.bin"), "r") do io
    raw = read(io, NZ * sizeof(Float32))
    ntoh.(reshape(reinterpret(Float32, raw), NZ))
end
DRF = Float64.(thk[1:NZ])


lon = Array(ds["lon"])
lat = Array(ds["lat"])
permute_to_std(x) = permutedims(Array(x), (2, 3, 1))      # -> (station, depth, time)
U     = Float64.(permute_to_std(ds["U_east"]))
V     = Float64.(permute_to_std(ds["V_north"]))
Salt  = Float64.(permute_to_std(ds["Salt"]))
Theta = Float64.(permute_to_std(ds["Theta"]))
hFacC = Float64.(Array(ds["hFacC"]))                      # (station, depth)
time_raw = Array(ds["time"])
close(ds)


xt = Dates.value.(time_raw .- time_raw[1]) ./ 86_400_000  # days since first sample
N_moor, nz, nt = size(U)
length(xt) == nt || error("time length $(length(xt)) ≠ nt = $nt")
println("Loaded $N_moor mooring points, $nz levels, $nt timesteps.")


dt_h = diff(xt) .* 24
gaps = findall(dt_h .> delt + 1e-3)
@printf("Record length: %.2f days; sampling interval %.3f–%.3f h; gaps: %d\n",
        xt[end] - xt[1], minimum(dt_h), maximum(dt_h), length(gaps))
if !isempty(gaps)
    println("  WARNING: the bandpass filter assumes uniform sampling and treats the")
    println("  record as contiguous across gaps; the harmonic fit uses the true times.")
    println("  Part of any difference near gaps comes from this.")
    for k in gaps
        println("    gap: ", time_raw[k], " -> ", time_raw[k+1], "  ($(round(dt_h[k], digits=1)) h)")
    end
end


hFacC_moor = hFacC
mask2D  = hFacC_moor .== 0
DRFfull = hFacC_moor .* reshape(DRF, 1, nz)               # (N_moor, nz)


# ocean column info per mooring
col_k  = Vector{Union{Nothing,UnitRange{Int}}}(nothing, N_moor)
col_dz = Vector{Vector{Float64}}(undef, N_moor)
Hm     = fill(NaN, N_moor)
for p in 1:N_moor
    idx = findall(hFacC_moor[p, :] .> 0)
    if isempty(idx)
        col_dz[p] = Float64[]
        continue
    end
    col_k[p]  = idx[1]:idx[end]
    col_dz[p] = DRFfull[p, idx[1]:idx[end]]
    Hm[p]     = sum(col_dz[p])
end


# ============================================================================
# DENSITY (shared)
# ============================================================================
z  = cumsum(DRFfull, dims=2)
zz = cat(zeros(N_moor, 1), z; dims=2)
za = -0.5 .* (zz[:, 1:end-1] .+ zz[:, 2:end])
rho = zeros(Float64, N_moor, nz, nt)
for t in 1:nt
    rho[:, :, t] = densjmd95(Salt[:, :, t], Theta[:, :, t], -za)
end


# ============================================================================
# N² (shared): gap-aware 3-day bins
# ============================================================================
bin_id = floor.(Int, xt ./ 3) .+ 1
bins   = [findall(==(b), bin_id) for b in unique(bin_id)]
bins   = filter(ix -> length(ix) >= timesteps_per_3days ÷ 2, bins)
nt_avg = length(bins)
salt_3day  = zeros(Float64, N_moor, nz, nt_avg)
theta_3day = zeros(Float64, N_moor, nz, nt_avg)
for (i, ix) in enumerate(bins)
    salt_3day[:, :, i]  = mean(Salt[:, :, ix],  dims=3)[:, :, 1]
    theta_3day[:, :, i] = mean(Theta[:, :, ix], dims=3)[:, :, 1]
end


z_centers    = za
z_interfaces = -zz[:, 2:end-1]
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
println("3-day bins used for N²: $nt_avg")


# ============================================================================
# VERTICAL MODES (shared, solved once)
# ============================================================================
Ce_out   = fill(NaN, N_moor, n_modes_keep)
Ueig_out = fill(NaN, N_moor, nz, n_modes_keep)
for p in 1:N_moor
    if col_k[p] === nothing
        println("  mooring p/N_moor skipped (land)")
        continue
    end
    kr = col_k[p]; k_top, ibot = first(kr), last(kr)
    f_pt = 2Ω * sind(lat[p])
    N2_mean_col = [(v = filter(!isnan, N2[p, k, :]); isempty(v) ? 1e-10 : mean(v)) for k in 1:nz]
    zf_col   = vcat(0.0, -cumsum(col_dz[p]))
    N2_faces = vcat(1e-10, N2_mean_col[k_top:ibot])


    k_sl, L_sl, C_sl, Cg_sl, Ce_sl, Weig_sl, Ueig_sl, Ueig2_sl =
        sturm_liouville_noneqDZ_norm(zf_col, N2_faces, f_pt, om, 0)
    size(Ueig2_sl, 1) == length(kr) || error("Mooring $p: unexpected Ueig2 size from solver.")


    n_avail = min(n_modes_keep, length(Ce_sl))
    Ce_out[p, 1:n_avail] = Ce_sl[1:n_avail]
    Ueig_out[p, kr, 1:n_avail] = Ueig2_sl[:, 1:n_avail]
end
println("Sturm–Liouville done (ω = M2 = $(round(2π/om/3600, digits=4)) h).")


# helper: modal projection  x_n = (1/H) ∫ x φ_n dz
#   X: (ncell) or (ncell, nt);  returns (n_modes) or (nt, n_modes)
project(Phi, dz, X, H) = transpose(transpose(Phi .* dz) * X) ./ H


# ============================================================================
# (A) BANDPASS METHOD
# ============================================================================
println("\n=== (A) BANDPASS (T1)–(T2) h ===")
fu = bandpassfilter(U,   T1, T2, delt, Nord, nt)
fv = bandpassfilter(V,   T1, T2, delt, Nord, nt)
fr = bandpassfilter(rho, T1, T2, delt, Nord, nt)


DRFfull_r = reshape(DRFfull, N_moor, nz, 1)
depth_r   = reshape(max.(sum(DRFfull, dims=2), eps()), N_moor, 1, 1)
mask3D    = repeat(reshape(mask2D, N_moor, nz, 1), 1, 1, nt)


pres  = g .* cumsum(fr .* DRFfull_r, dims=2)
pfz   = cat(zeros(N_moor, 1, nt), pres; dims=2)
pc_3d = 0.5 .* (pfz[:, 1:end-1, :] .+ pfz[:, 2:end, :])
pp_3d = pc_3d .- sum(pc_3d .* DRFfull_r, dims=2) ./ depth_r
up_3d = fu    .- sum(fu    .* DRFfull_r, dims=2) ./ depth_r
vp_3d = fv    .- sum(fv    .* DRFfull_r, dims=2) ./ depth_r
pp_3d[mask3D] .= 0;  up_3d[mask3D] .= 0;  vp_3d[mask3D] .= 0


tt = (1 + nedge):(nt - nedge)
println("Time window used for bandpass means: samples (first(tt))–(last(tt)) of $nt")


# total (undecomposed) BC flux, kW/m
Fx_bp_tot = vec(mean(dropdims(sum(up_3d .* pp_3d .* DRFfull_r, dims=2), dims=2)[:, tt], dims=2)) ./ 1000
Fy_bp_tot = vec(mean(dropdims(sum(vp_3d .* pp_3d .* DRFfull_r, dims=2), dims=2)[:, tt], dims=2)) ./ 1000


# modal flux, kW/m
Fx_bp = fill(NaN, N_moor, n_modes_keep)
Fy_bp = fill(NaN, N_moor, n_modes_keep)
for p in 1:N_moor
    col_k[p] === nothing && continue
    kr  = col_k[p]
    Phi = Ueig_out[p, kr, :]
    any(isnan, Phi) && continue
    dz, H = col_dz[p], Hm[p]
    uh = project(Phi, dz, up_3d[p, kr, :], H)       # (nt, n_modes)
    vh = project(Phi, dz, vp_3d[p, kr, :], H)
    ph = project(Phi, dz, pp_3d[p, kr, :], H)
    Fx_bp[p, :] = vec(mean(uh[tt, :] .* ph[tt, :], dims=1)) .* H ./ 1000
    Fy_bp[p, :] = vec(mean(vh[tt, :] .* ph[tt, :], dims=1)) .* H ./ 1000
end
println("Bandpass fluxes done.")


# ============================================================================
# (B) HARMONIC METHOD
# ============================================================================
println("\n=== (B) HARMONIC (constituents in L2 table: $freq_sel) ===")
# returns Z (nc, nz) with x(t) = Σ_j Re[Z_j e^{iω_j t}]
function harm_complex(X, xt, freq_sel)
    _, a, b, _, R2, _, _ = harmonic03(xt, X, freq_sel)
    Z = complex.(a, -b)
    Z[isnan.(Z)] .= 0
    return Z, R2
end


Uc = zeros(ComplexF64, nc, N_moor, nz)
Vc = zeros(ComplexF64, nc, N_moor, nz)
Rc = zeros(ComplexF64, nc, N_moor, nz)
R2_U = fill(NaN, N_moor)
for p in 1:N_moor
    Z, r2 = harm_complex(U[p, :, :],   xt, freq_sel);  Uc[:, p, :] = Z
    Z, _  = harm_complex(V[p, :, :],   xt, freq_sel);  Vc[:, p, :] = Z
    Z, _  = harm_complex(rho[p, :, :], xt, freq_sel);  Rc[:, p, :] = Z
    r2g = filter(!isnan, vec(r2))
    R2_U[p] = isempty(r2g) ? NaN : median(r2g)
end
println("Median R² of U fit per mooring: ", round.(R2_U, digits=2))


# pressure amplitude (hydrostatic, same integration as bandpass)
Pc = zeros(ComplexF64, nc, N_moor, nz)
for j in 1:nc, p in 1:N_moor
    p_bot = g .* cumsum(Rc[j, p, :] .* DRFfull[p, :])
    p_top = vcat(0.0 + 0im, p_bot[1:end-1])
    Pc[j, p, :] = 0.5 .* (p_top .+ p_bot)
end


Fx_hm_tot = fill(NaN, N_moor);  Fy_hm_tot = fill(NaN, N_moor)
Fx_hm = fill(NaN, N_moor, n_modes_keep)
Fy_hm = fill(NaN, N_moor, n_modes_keep)
for p in 1:N_moor
    col_k[p] === nothing && continue
    kr, dz, H = col_k[p], col_dz[p], Hm[p]
    Phi = Ueig_out[p, kr, :]
    fxt = 0.0; fyt = 0.0
    fxm = zeros(n_modes_keep); fym = zeros(n_modes_keep)
    for j in 1:nc
        u  = Uc[j, p, kr];  v = Vc[j, p, kr];  pr = Pc[j, p, kr]
        # total BC flux: remove depth mean (as in bandpass), then ½Re(u p*)
        up = u  .- sum(u  .* dz) / H
        vp = v  .- sum(v  .* dz) / H
        pp = pr .- sum(pr .* dz) / H
        fxt += sum(0.5 .* real.(up .* conj.(pp)) .* dz)
        fyt += sum(0.5 .* real.(vp .* conj.(pp)) .* dz)
        # modal flux
        if !any(isnan, Phi)
            un = project(Phi, dz, u,  H)
            vn = project(Phi, dz, v,  H)
            pn = project(Phi, dz, pr, H)
            fxm .+= (H / 2) .* real.(un .* conj.(pn))
            fym .+= (H / 2) .* real.(vn .* conj.(pn))
        end
    end
    Fx_hm_tot[p] = fxt / 1000;  Fy_hm_tot[p] = fyt / 1000
    if !any(isnan, Phi)
        Fx_hm[p, :] = fxm ./ 1000;  Fy_hm[p, :] = fym ./ 1000
    end
end
println("Harmonic fluxes done.")


# sum over modes 1..n_modes_keep
Fx_bp_msum = vec(sum(Fx_bp, dims=2));  Fy_bp_msum = vec(sum(Fy_bp, dims=2))
Fx_hm_msum = vec(sum(Fx_hm, dims=2));  Fy_hm_msum = vec(sum(Fy_hm, dims=2))


# ============================================================================
# COMPARISON
# ============================================================================
magn(x, y) = hypot.(x, y)
dirn(x, y) = atand.(y, x)                                   # deg, CCW from east
ddir(a, b) = mod.(a .- b .+ 180, 360) .- 180               # wrapped to [-180, 180)


# stats of harmonic (h) vs bandpass (b)
function cmp_stats(h, b)
    ok = .!isnan.(h) .& .!isnan.(b)
    n  = count(ok)
    n < 2 && return (n=n, bias=NaN, rmsd=NaN, r=NaN, slope=NaN)
    x, y = h[ok], b[ok]
    (n = n,
     bias  = mean(x .- y),
     rmsd  = sqrt(mean((x .- y).^2)),
     r     = cor(x, y),
     slope = dot(x, y) / dot(y, y))                         # harmonic ≈ slope × bandpass
end


cases = vcat([("Total BC", Fx_bp_tot, Fy_bp_tot, Fx_hm_tot, Fy_hm_tot),
              ("Modes 1–$n_modes_keep", Fx_bp_msum, Fy_bp_msum, Fx_hm_msum, Fy_hm_msum)],
             [("Mode $n", Fx_bp[:, n], Fy_bp[:, n], Fx_hm[:, n], Fy_hm[:, n]) for n in 1:n_modes_show])


println("\n" * "="^96)
println("SUMMARY: harmonic (H) vs bandpass (B) depth-integrated flux [kW/m]")
println("  slope = least-squares H ≈ slope·B through origin; ratio = median |F_H|/|F_B|;")
println("  |Δθ| = median absolute direction difference")
println("="^96)
@printf("%-12s | %-4s | %3s | %8s %8s %6s %6s | %7s %7s\n",
        "case", "comp", "n", "bias", "rmsd", "r", "slope", "ratio", "|Δθ|°")
for (name, bx, by, hx, hy) in cases
    mb, mh = magn(bx, by), magn(hx, hy)
    rat = mh ./ mb
    rat_g = filter(isfinite, rat)
    dth = abs.(ddir(dirn(hx, hy), dirn(bx, by)))
    dth_g = filter(!isnan, dth)
    medrat = isempty(rat_g) ? NaN : median(rat_g)
    medth  = isempty(dth_g) ? NaN : median(dth_g)
    for (comp, h, b) in (("Fx", hx, bx), ("Fy", hy, by), ("|F|", mh, mb))
        s = cmp_stats(h, b)
        if comp == "|F|"
            @printf("%-12s | %-4s | %3d | %8.3f %8.3f %6.3f %6.3f | %7.3f %7.1f\n",
                    name, comp, s.n, s.bias, s.rmsd, s.r, s.slope, medrat, medth)
        else
            @printf("%-12s | %-4s | %3d | %8.3f %8.3f %6.3f %6.3f |\n",
                    name, comp, s.n, s.bias, s.rmsd, s.r, s.slope)
        end
    end
    println("-"^96)
end


# per-mooring tables
for (name, bx, by, hx, hy) in cases[[1, 3]]                 # Total BC and Mode 1
    println("\nPer-mooring: $name   (kW/m, direction in ° CCW from east)")
    @printf("%4s %7s %7s | %8s %8s %8s %6s | %8s %8s %8s %6s | %6s %7s\n",
            "p", "lon", "lat", "Fx_B", "Fy_B", "|F|_B", "θ_B",
            "Fx_H", "Fy_H", "|F|_H", "θ_H", "H/B", "Δθ")
    for p in 1:N_moor
        mb, mh = hypot(bx[p], by[p]), hypot(hx[p], hy[p])
        θb, θh = atand(by[p], bx[p]), atand(hy[p], hx[p])
        @printf("%4d %7.2f %7.2f | %8.3f %8.3f %8.3f %6.0f | %8.3f %8.3f %8.3f %6.0f | %6.2f %7.0f\n",
                p, lon[p], lat[p], bx[p], by[p], mb, θb, hx[p], hy[p], mh, θh,
                mh / mb, mod(θh - θb + 180, 360) - 180)
    end
end


# ============================================================================
# CSV TABLE
# ============================================================================
header = ["p", "lon", "lat", "H_m", "R2_U_harm",
          "Fx_tot_bp", "Fy_tot_bp", "Fx_tot_hm", "Fy_tot_hm",
          "Fx_msum_bp", "Fy_msum_bp", "Fx_msum_hm", "Fy_msum_hm"]
cols = Any[collect(1:N_moor), lon, lat, Hm, R2_U,
           Fx_bp_tot, Fy_bp_tot, Fx_hm_tot, Fy_hm_tot,
           Fx_bp_msum, Fy_bp_msum, Fx_hm_msum, Fy_hm_msum]
for n in 1:n_modes_keep
    append!(header, ["Fx_m(n)bp","Fym(n)_bp", "Fx_m(n)hm","Fym(n)_hm"])
    append!(cols, [Fx_bp[:, n], Fy_bp[:, n], Fx_hm[:, n], Fy_hm[:, n]])
end
csvfile = joinpath(FIGDIR, "flux_compare_table.csv")
open(csvfile, "w") do io
    writedlm(io, permutedims(header), ',')
    writedlm(io, hcat(cols...), ',')
end
println("\nSaved table -> $csvfile")


# ============================================================================
# PLOTS
# ============================================================================
mc = Makie.wong_colors()


function one2one!(ax, b, h; color=:dodgerblue)
    ok = .!isnan.(b) .& .!isnan.(h)
    any(ok) || return
    lo = min(minimum(b[ok]), minimum(h[ok]), 0.0)
    hi = max(maximum(b[ok]), maximum(h[ok]), 0.0)
    pad = 0.05 * (hi - lo + eps())
    lines!(ax, [lo - pad, hi + pad], [lo - pad, hi + pad], color=:black, linestyle=:dash)
    scatter!(ax, b[ok], h[ok], color=color, markersize=7)
end


# --- scatter: rows = cases, columns = Fx, Fy, |F|
plot_cases = cases[[1, 3:end...]]                          # Total BC + modes 1..n_modes_show
fig1 = Figure(resolution=(1050, 300 * length(plot_cases)))
for (i, (name, bx, by, hx, hy)) in enumerate(plot_cases)
    for (j, (comp, b, h)) in enumerate((("Fx", bx, hx), ("Fy", by, hy),
                                        ("|F|", magn(bx, by), magn(hx, hy))))
        ax = Axis(fig1[i, j], xlabel="Bandpass $comp [kW/m]", ylabel="Harmonic $comp [kW/m]",
                  title="$name – $comp", aspect=1)
        one2one!(ax, b, h; color=mc[mod1(i, length(mc))])
    end
end
save(joinpath(FIGDIR, "flux_compare_scatter.png"), fig1)
display(fig1)


# --- maps of flux vectors
nan0(x) = ifelse(isnan(x), 0.0, x)
function flux_map(name, bx, by, hx, hy, fname)
    mags = filter(!isnan, vcat(magn(bx, by), magn(hx, hy)))
    Fmax = isempty(mags) ? 1.0 : maximum(mags)
    span = max(maximum(lon) - minimum(lon), maximum(lat) - minimum(lat), 0.5)
    ls   = 0.15 * span / Fmax                               # longest arrow = 15% of domain
    fig = Figure(resolution=(850, 750))
    ax  = Axis(fig[1, 1], xlabel="Longitude [°]", ylabel="Latitude [°]",
               title="$name flux: bandpass vs harmonic (max |F| = $(round(Fmax, digits=2)) kW/m)",
               aspect=DataAspect())
    scatter!(ax, lon, lat, color=:gray60, markersize=5)
    arrows!(ax, lon, lat, nan0.(bx) .* ls, nan0.(by) .* ls, color=mc[1], linewidth=2,
            arrowsize=8, label="Bandpass")
    arrows!(ax, lon, lat, nan0.(hx) .* ls, nan0.(hy) .* ls, color=mc[2], linewidth=2,
            arrowsize=8, label="Harmonic")
    axislegend(ax, position=:rb)
    display(fig)
    save(joinpath(FIGDIR, fname), fig)
end
flux_map("Total BC", Fx_bp_tot, Fy_bp_tot, Fx_hm_tot, Fy_hm_tot, "flux_compare_map_total.png")
flux_map("Mode 1",   Fx_bp[:, 1], Fy_bp[:, 1], Fx_hm[:, 1], Fy_hm[:, 1], "flux_compare_map_mode1.png")


# --- magnitude ratio vs latitude
fig3 = Figure(resolution=(700, 800))
ax3  = Axis(fig3[1, 1], xlabel="|F_harmonic| / |F_bandpass|", ylabel="Latitude [°]",
            title="Flux magnitude ratio vs latitude", xscale=log10)
vlines!(ax3, [1.0], color=:black, linestyle=:dash)
markers = [:circle, :utriangle, :diamond, :rect, :star5]
for (i, (name, bx, by, hx, hy)) in enumerate(plot_cases)
    r  = magn(hx, hy) ./ magn(bx, by)
    ok = isfinite.(r) .& (r .> 0)
    any(ok) && scatter!(ax3, r[ok], lat[ok], color=mc[mod1(i, length(mc))],
                        marker=markers[mod1(i, length(markers))], markersize=9, label=name)
end
axislegend(ax3, position=:rb)
display(fig3)
#ave(joinpath(FIGDIR, "flux_compare_ratio_vs_lat.png"), fig3)


println("Saved figures to $FIGDIR")
close(logio)




