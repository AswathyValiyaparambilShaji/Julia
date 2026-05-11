# ============================================================================
# beam_energy_budget.jl
# KE, APE, and KE/APE ratio for 20 beam stations from NC file
# Follows same computational style as the LLC4320 tile processing code
# Dims: U/V/rho/Theta/Salt → (beam_point=20, nz=88, nt=2543)
#       DRFfull             → (beam_point=20, nz=88)
#       DRF                 → (nz=88)
# ============================================================================


using NCDatasets, DSP, Statistics, CairoMakie, Printf
include(joinpath(@__DIR__, "..","..","..", "functions", "densjmd95.jl"))

# ============================================================================
# PATHS AND CONSTANTS
# ============================================================================
ncfile  = "/data3/aswathy/mnt/data/aswathy/MITgcm_NAS/figures/beam_UVrho.nc"
figfile = "mnt/data/aswathy/MITgcm_NAS/figures/beam_profiles.png"
t_plot  = 500


rho0   = 1027.5
g      = 9.81
ωM2    = 2π / (12.4206 * 3600.0)   # M2 angular frequency (rad/s)
Ω_rot  = 7.292115e-5               # Earth rotation rate (rad/s)


# filter parameters (same as tile code)
T1, T2, delt, N_ord = 9.0, 15.0, 1.0, 4
ts_3day = 72          # hourly timesteps per 3-day window





# ============================================================================
# CHECK — print file contents before anything else
# ============================================================================
println("\n========== NC FILE CHECK ==========")
NCDatasets.Dataset(ncfile, "r") do ds
    println("Dimensions:")
    for (k, v) in ds.dim
        @printf("  %-15s = %d\n", k, v)
    end
    println("\nVariables:")
    for vname in keys(ds)
        v    = ds[vname]
        dims = join(dimnames(v), " × ")
        T    = eltype(v)
        println("  $vname  [$T]  ($dims)")
        if ndims(v) <= 2
            data  = v[:]
            valid = filter(isfinite, Float64.(data[:]))
            if !isempty(valid)
                @printf("    range: %.4f  →  %.4f\n", minimum(valid), maximum(valid))
            end
        end
    end
    println("\nGlobal attributes:")
    for (k, v) in ds.attrib
        println("  $k = $v")
    end
end
println("====================================\n")




# ============================================================================
# READ
# ============================================================================
ds = NCDatasets.Dataset(ncfile, "r")


lons    = Float64.(ds["longitude"][:])       # (beam_point,)
lats    = Float64.(ds["latitude"][:])        # (beam_point,)
DRFfull = Float64.(ds["DRFfull"][:, :])      # (beam_point, nz)
DRF     = Float64.(ds["DRF"][:])             # (nz,)
U       = Float64.(ds["U"][:, :, :])         # (beam_point, nz, nt)
V       = Float64.(ds["V"][:, :, :])
rho     = Float64.(ds["rho_insitu"][:, :, :])
Theta   = Float64.(ds["Theta"][:, :, :])
Salt    = Float64.(ds["Salt"][:, :, :])


close(ds)


N_beam, nz, nt = size(U)
nt_avg = div(nt, ts_3day)                    # number of 3-day windows = 35
println("Loaded: N_beam=$N_beam  nz=$nz  nt=$nt  nt_avg=$nt_avg")
println("Plotting time index t=$t_plot  (of $nt total)\n")




# ============================================================================
# DEPTH COMPUTATION  — same logic as tile code, adapted for (beam_point, nz)
# ============================================================================
# DRFfull zeros mark dry cells (same as hFacC==0 in tile code)
mask2D  = (DRFfull .== 0.0)                  # (beam_point, nz)  true = dry


z_cumsum    = cumsum(DRFfull, dims=2)                                # (N_beam, nz)
zz          = cat(zeros(N_beam, 1), z_cumsum; dims=2)                # (N_beam, nz+1)
z_centers   = -0.5 .* (zz[:, 1:end-1] .+ zz[:, 2:end])             # (N_beam, nz)     negative = depth
z_interfaces = -zz[:, 2:end-1]                                       # (N_beam, nz-1)   interface depths
Δz          = z_centers[:, 2:end] .- z_centers[:, 1:end-1]          # (N_beam, nz-1)   spacing (negative)
depth       = sum(DRFfull, dims=2)                                    # (N_beam, 1)      total depth




# ============================================================================
# BANDPASS FILTER U, V, rho — same params as tile code (9-15 hr, order 4)
# ============================================================================
println("Bandpass filtering (9–15 hr) ...")
fcutlow  = 1.0 / T2
fcuthigh = 1.0 / T1
bpf = digitalfilter(Bandpass(fcutlow, fcuthigh), Butterworth(N_ord); fs = 1/delt)


fu = zeros(Float64, N_beam, nz, nt)
fv = zeros(Float64, N_beam, nz, nt)
fr = zeros(Float64, N_beam, nz, nt)


for i in 1:N_beam
    for k in 1:nz
        mask2D[i, k] && continue          # skip dry cells (same as hFacC==0)
        fu[i, k, :] = filtfilt(bpf, U[i, k, :])
        fv[i, k, :] = filtfilt(bpf, V[i, k, :])
        fr[i, k, :] = filtfilt(bpf, rho[i, k, :])
    end
end


# zero out dry cells (same as mask4D apply in tile code)
fu .*= reshape(.!mask2D, N_beam, nz, 1)
fv .*= reshape(.!mask2D, N_beam, nz, 1)
fr .*= reshape(.!mask2D, N_beam, nz, 1)




# ============================================================================
# PERTURBATION VELOCITIES — subtract depth-mean (same logic as tile code)
# ============================================================================
println("Computing perturbation velocities ...")
up_3d = zeros(Float64, N_beam, nz, nt)
vp_3d = zeros(Float64, N_beam, nz, nt)


for i in 1:N_beam
    H_i  = depth[i]                                          # scalar total depth
    dz_i = reshape(DRFfull[i, :], nz, 1)                    # (nz, 1)  for broadcast


    ucA = sum(fu[i, :, :] .* dz_i, dims=1) ./ H_i           # (1, nt) depth-mean
    vcA = sum(fv[i, :, :] .* dz_i, dims=1) ./ H_i


    up_3d[i, :, :] = fu[i, :, :] .- ucA                     # baroclinic u'
    vp_3d[i, :, :] = fv[i, :, :] .- vcA                     # baroclinic v'
end


up_3d .*= reshape(.!mask2D, N_beam, nz, 1)
vp_3d .*= reshape(.!mask2D, N_beam, nz, 1)




# ============================================================================
# KE — depth-integrated and 36-hr lowpass filtered
# ============================================================================
println("Computing KE ...")


# KE = 0.5 * rho0 * (u'^2 + v'^2) at each cell, then depth-integrate
KE_int = zeros(Float64, N_beam, nt)
for i in 1:N_beam
    dz_i   = reshape(DRFfull[i, :], nz, 1)
    ke_3d  = 0.5 .* rho0 .* (up_3d[i, :, :].^2 .+ vp_3d[i, :, :].^2)   # (nz, nt)
    KE_int[i, :] = vec(sum(ke_3d .* dz_i, dims=1))
end






# ============================================================================
# 3-DAY AVERAGING of U, V, Theta, Salt
# ============================================================================
println("3-day averaging ...")
theta_3day = zeros(Float64, N_beam, nz, nt_avg)
salt_3day  = zeros(Float64, N_beam, nz, nt_avg)


for i in 1:nt_avg
    t_start = (i-1) * ts_3day + 1
    t_end   = min(i * ts_3day, nt)
    theta_3day[:, :, i] = mean(Theta[:, :, t_start:t_end], dims=3)[:, :, 1]
    salt_3day[:, :, i]  = mean(Salt[:, :,  t_start:t_end], dims=3)[:, :, 1]
end




# ============================================================================
# N² — same method as tile code: densjmd95 at interfaces, store at cell centers
# ============================================================================
println("Computing N² ...")
N2 = zeros(Float64, N_beam, nz, nt_avg)


for t in 1:nt_avg
    S_t = salt_3day[:, :, t]          # (N_beam, nz)
    T_t = theta_3day[:, :, t]


    S_upper = S_t[:, 1:end-1]         # (N_beam, nz-1)
    T_upper = T_t[:, 1:end-1]
    S_lower = S_t[:, 2:end]
    T_lower = T_t[:, 2:end]


    # reference BOTH to interface depth (critical: same as tile code)
    rho_upper = densjmd95(S_upper, T_upper, z_interfaces)
    rho_lower = densjmd95(S_lower, T_lower, z_interfaces)


    Δρ = rho_lower .- rho_upper                              # (N_beam, nz-1)
    N2_interfaces = -(g / rho0) .* (Δρ ./ Δz)               # (N_beam, nz-1)


    # store at lower cell center (same indexing as tile code)
    N2[:, 1:end-1, t] = N2_interfaces
end




# ============================================================================
# BUOYANCY b AND APE — same formulation as tile code
# ============================================================================
println("Computing b and APE ...")
# b = -g/rho0 * rho' (bandpassed density perturbation)
b = (-g / rho0) .* fr
b .*= reshape(.!mask2D, N_beam, nz, 1)


# APE = 0.5 * rho0 * b² / N²  per 3-day window
APE = fill(NaN, N_beam, nz, nt)


for t in 1:nt_avg
    n2_val = N2[:, :, t]                                     # (N_beam, nz)
    tstart = (t-1) * ts_3day + 1
    tend   = min(t * ts_3day, nt)


    for tt in tstart:tend
        b_tt  = b[:, :, tt]                                  # (N_beam, nz)
        ape_tt = 0.5 .* rho0 .* (b_tt.^2 ./ n2_val)
        ape_tt[n2_val .<= 0.0]  .= NaN                       # unstable — same as tile code
        ape_tt[mask2D]          .= NaN
        APE[:, :, tt] = ape_tt
    end
end


# depth-integrate APE (NaN-safe, same approach as tile code)
APE_int = zeros(Float64, N_beam, nt)
for i in 1:N_beam
    for tt in 1:nt
        for k in 1:nz
            if !isnan(APE[i, k, tt])
                APE_int[i, tt] += APE[i, k, tt] * DRFfull[i, k]
            end
        end
    end
end




# ============================================================================
# KE/APE RATIO vs LATITUDE
# ============================================================================
println("Computing KE/APE ratio ...")
KE_mean  = vec(mean(KE_int,  dims=2))           # (N_beam,)  time-mean
APE_mean = vec(mean(APE_int, dims=2))


ratio_obs    = KE_mean  ./ APE_mean
fcor         = @. 2Ω_rot * sind(lats)
ratio_theory = @. (ωM2^2 + fcor^2) / (ωM2^2 - fcor^2)


println("\n=== KE/APE Station Summary ===")
@printf("%-5s  %-8s  %-12s  %-12s  %-8s  %-8s\n",
        "Stn","Lat(°N)","KE(J/m²)","APE(J/m²)","KE/APE","Theory")
for i in 1:N_beam
    @printf("%-5d  %-8.2f  %-12.3f  %-12.3f  %-8.3f  %-8.3f\n",
            i, lats[i], KE_mean[i], APE_mean[i], ratio_obs[i], ratio_theory[i])
end




# ============================================================================
# DEPTH AXIS for profile plots (cell-centre depths, positive-down for y-axis)
# ============================================================================
depth_c = -z_centers          # (N_beam, nz)  positive-down for plotting




# ============================================================================
# PLOT 1 — vertical profiles at t_plot  (your original plot)
# ============================================================================
U_t   = U[:, :, t_plot]
V_t   = V[:, :, t_plot]
rho_t = rho[:, :, t_plot]


colors = cgrad(:tab20, N_beam, categorical=true)


fig1 = Figure(size=(1300, 700))


ax_u = Axis(fig1[1, 1], xlabel="U (m/s)", ylabel="Depth (m)",
            title="Zonal velocity", yreversed=true)
ax_v = Axis(fig1[1, 2], xlabel="V (m/s)",
            title="Meridional velocity", yreversed=true)
ax_r = Axis(fig1[1, 3], xlabel="ρ (kg/m³)",
            title="In-situ density", yreversed=true)


for i in 1:N_beam
    label = @sprintf("%.2f°N, %.2f°E", lats[i], lons[i])
    col   = colors[i]
    d     = depth_c[i, :]
    lines!(ax_u, U_t[i, :],   d, color=col, linewidth=1.5, label=label)
    lines!(ax_v, V_t[i, :],   d, color=col, linewidth=1.5)
    lines!(ax_r, rho_t[i, :], d, color=col, linewidth=1.5)
    vlines!(ax_u, 0; color=:black, linewidth=0.5, linestyle=:dash)
    vlines!(ax_v, 0; color=:black, linewidth=0.5, linestyle=:dash)
end


xlims!(ax_r, 1020, 1060)
Legend(fig1[1, 4], ax_u, "Beam points", framevisible=true, labelsize=10)
Label(fig1[0, :], "Beam vertical profiles — time index t=$t_plot  (N_beam=$N_beam)",
      fontsize=14, font=:bold)

display(fig1)
println("Saved: $figfile")




# ============================================================================
# PLOT 2 — KE timeseries (all stations, 36-hr lowpass)
# ============================================================================
tt = collect(1:nt) ./ 24.0          # time axis in days


fig2 = Figure(size=(1000, 400))
ax_ke = Axis(fig2[1, 1], xlabel="Time (days)", ylabel="KE (J/m²)",
             title="D2 depth-integrated KE — 36-hr lowpass")
for i in 1:N_beam
    lines!(ax_ke, tt, KE_int[i, :], color=colors[i], linewidth=1.0,
           label=@sprintf("%.1f°N", lats[i]))
end
Legend(fig2[1, 2], ax_ke, "Stations", framevisible=true, labelsize=9)
display(fig2)




# ============================================================================
# PLOT 3 — KE/APE ratio vs latitude
# ============================================================================
fig3 = Figure(size=(500, 600))
ax_rat = Axis(fig3[1, 1], xlabel="KE / APE", ylabel="Latitude (°N)",
              title="KE/APE ratio vs latitude")
lines!(ax_rat,  ratio_theory, lats, color=:black, linestyle=:dash,
       linewidth=2, label="Theory (ω²+f²)/(ω²−f²)")
scatter!(ax_rat, ratio_obs, lats, color=:red, markersize=8, label="Observed")
axislegend(ax_rat, position=:lt)


display(fig1); display(fig2); display(fig3)
println("Done.")




