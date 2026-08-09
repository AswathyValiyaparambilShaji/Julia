using DSP, Statistics, Printf, LinearAlgebra, TOML, NCDatasets, Impute
include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: bandpassfilter
include(joinpath(@__DIR__, "..", "..", "..", "functions", "densjmd95.jl"))
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg  = TOML.parsefile(config_file)
base  = cfg["base_path"]
base2 = cfg["base_path2"]
g    = 9.81
rho0 = 1027.0
T1, T2, delt, N = 9.0, 15.0, 1.0, 4
timesteps_per_3days = 72


# ============================================================================
# READ MOORING NETCDF (point data: mooring_point, nz, nt)
# ============================================================================
ncfile = joinpath(base2, "mooring_UVrho.nc")
ds = NCDataset(ncfile, "r")
lon      = Array(ds["longitude"])
lat      = Array(ds["latitude"])
U        = Float64.(Array(ds["U"]))
V        = Float64.(Array(ds["V"]))
Salt     = Float64.(Array(ds["Salt"]))
Theta    = Float64.(Array(ds["Theta"]))
DRFfull  = Float64.(Array(ds["DRFfull"]))
DRF      = Float64.(Array(ds["DRF"]))
close(ds)
N_moor, nz, nt = size(U)


# ============================================================================
# RECONSTRUCT hFacC MASK AT MOORING POINTS (DRFfull = hFacC .* DRF)
# ============================================================================
hFacC_moor = DRFfull ./ reshape(DRF, 1, nz)
mask2D = hFacC_moor .== 0


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
# Check 
#println(round.(sum(pp_3d.*DRFfull_r,dims=2)./depth_r, digits=4))

# ============================================================================
# BC VELOCITY PERTURBATIONS
# ============================================================================
ucA_3d = sum(fu .* DRFfull_r, dims=2) ./ depth_r
up_3d  = fu .- ucA_3d
up_3d[mask3D] .= 0
vcA_3d = sum(fv .* DRFfull_r, dims=2) ./ depth_r
vp_3d  = fv .- vcA_3d
vp_3d[mask3D] .= 0
#println(round.(sum(up_3d.*DRFfull_r,dims=2)./depth_r, digits=4))
#println(round.(sum(vp_3d.*DRFfull_r,dims=2)./depth_r, digits=4))


# ============================================================================
# BC FLUXES
# ============================================================================
xflx_3d = up_3d .* pp_3d
yflx_3d = vp_3d .* pp_3d
size(xflx_3d)
Fu_b = dropdims(sum(xflx_3d .* DRFfull_r, dims=2),dims=2)    # (nx, ny, 1)
Fv_b = dropdims(sum(yflx_3d .* DRFfull_r, dims=2) ,dims=2)    # (nx, ny, 1)
Fu_bc= dropdims(mean(Fu_b,dims=2),dims=2)./1000
Fv_bc= dropdims(mean(Fv_b,dims=2),dims=2)./1000
size(Fu_bc)
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
n_modes_keep = 25
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
    if length(ocean_idx) < min_ocean_cells
        continue
    end
    k_top = ocean_idx[1]
    ibot  = ocean_idx[end]
    n_cells = ibot - k_top + 1


    # time-mean N2 per cell (nz points), NaN-safe
    N2_mean_col = [ (v = filter(!isnan, N2[p, k, :]); isempty(v) ? 1e-10 : mean(v))
                    for k in 1:nz ]


    dz_col = (hfac_col .* DRF)[k_top:ibot]        # n_cells
    zf_cells = cumsum(dz_col)                      # n_cells bottom-face depths
    N2_cells = N2_mean_col[k_top:ibot]              # n_cells values, matching zf_cells


    # prepend the surface: depth = 0, N2 = 1e-10 threshold
    zf_col   = vcat(0.0, -zf_cells)                 # n_cells+1
    N2_faces = vcat(1e-10, N2_cells)                # n_cells+1


    k_sl, L_sl, C_sl, Cg_sl, Ce_sl, Weig_sl, Ueig_sl, Ueig2_sl =
        sturm_liouville_noneqDZ_norm(zf_col, N2_faces, f_pt, om, 0)


    # expect n_cells+1 Weig (faces) and n_cells Ueig (cells)
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
    Weig_out[p, k_top:ibot, 1:n_avail] = Weig_sl[2:end, 1:n_avail]   # drop the surface face, keep the nz cell-bottom faces


    println("  mooring point $p/$N_moor solved")
end
        
# ============================================================================
# SEPARATE CHECK: mode 1 & mode 2 orthonormality (Ueig_out)
# ============================================================================
for p in 1:N_moor
    hfac_col = hFacC_moor[p, :]
    ocean_idx = findall(hfac_col .> 0)
    if length(ocean_idx) < min_ocean_cells
        continue
    end
    k_top = ocean_idx[1]
    ibot  = ocean_idx[end]
    dz_col = (hfac_col .* DRF)[k_top:ibot]
    H = sum(dz_col)


    U1 = @view Ueig_out[p, k_top:ibot, 1]
    U2 = @view Ueig_out[p, k_top:ibot, 2]
    if any(isnan, U1) || any(isnan, U2)
        continue
    end


    check_11 = sum(U1 .* U1 .* dz_col) / H   # should be ≈ 1
    check_22 = sum(U2 .* U2 .* dz_col) / H   # should be ≈ 1
    check_12 = sum(U1 .* U2 .* dz_col) / H   # should be ≈ 0


    println("Mooring point $p:")
    println("  mode1·mode1 = ", round(check_11, digits=4))
    println("  mode2·mode2 = ", round(check_22, digits=4))
    println("  mode1·mode2 = ", round(check_12, digits=6))
end





# ============================================================================
# SANITY CHECK: mode orthonormality
#   (1/H) ∫ Weig_m * Weig_n dz  ==  1 if m==n,  0 if m!=n
# ============================================================================
println("Checking mode orthonormality (ueig_out)...")


for p in 1:N_moor
    hfac_col = hFacC_moor[p, :]
    ocean_idx = findall(hfac_col .> 0)
    if length(ocean_idx) < min_ocean_cells
        continue
    end
    k_top = ocean_idx[1]
    ibot  = ocean_idx[end]


    dz_col = (hfac_col .* DRF)[k_top:ibot]
    H = sum(dz_col)


    Phi = @view Ueig_out[p, k_top:ibot, :]   # (n_cells, n_modes_keep)
    if any(isnan, Phi)
        continue
    end


    n_avail = size(Phi, 2)
    ortho_matrix = fill(NaN, n_avail, n_avail)
    for m in 1:n_avail, n in 1:n_avail
        ortho_matrix[m, n] = sum(Phi[:, m] .* Phi[:, n] .* dz_col) / H
    end


    println("Mooring point $p:")
    println("  diag (should be ≈ 1):    ", round.(diag(ortho_matrix), digits=4))
    off_diag_vals = [ortho_matrix[m, n] for m in 1:n_avail, n in 1:n_avail if m != n]
    println("  off-diag (should be ≈ 0): max abs = ", round(maximum(abs.(off_diag_vals)), digits=6))
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
   if length(ocean_idx) < min_ocean_cells
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




p       = 1        # mooring point
ts, te  = 100, 300  # time slice you want to show
z_idx   = 25         # which depth CELL (relative to k_top) to compare at
n_modes_sum = 1:5   # modes to sum


hfac_col = hFacC_moor[p, :]
ocean_idx = findall(hfac_col .> 0)
k_top = ocean_idx[1]
ibot  = ocean_idx[end]

Phi_all = Ueig_out[p, k_top:ibot, n_modes_sum]     # (n_cells, 5)
uhat_all = uhat_out[p, ts:te, n_modes_sum]          # (nt_slice, 5)


u_modal_sum = uhat_all * Phi_all[z_idx, :]          # (nt_slice,)


u_actual = up_3d[p,  z_idx , ts:te]


fig = Figure(resolution=(1000,400))
ax = Axis(fig[1,1], xlabel="time step", ylabel="u (m/s)",
          title="Mooring $p, depth cell $z_idx: modes 1-25 sum vs BC u ")
lines!(ax, ts:te, u_actual, color=:black, linewidth=2, label="BC u")
lines!(ax, ts:te, u_modal_sum, color=:red, linewidth=2, linestyle=:dash, label="modes 1 to 25")
axislegend(ax)
display(fig)

# reconstructing Pcolor plot

p = 1  # pick a mooring point
mode_n = 1


hfac_col = hFacC_moor[p, :]
ocean_idx = findall(hfac_col .> 0)
k_top = ocean_idx[1]
ibot  = ocean_idx[end]
dz_col = (hfac_col .* DRF)[k_top:ibot]
z_centers_col = -cumsum(dz_col) .+ dz_col./2   # depth of each cell center (adjust sign/offset as needed)


Ue_n = Ueig_out[p, k_top:ibot, mode_n]        # (n_cells,) vertical structure
We_n = Weig_out[p, k_top:ibot, mode_n]        # (n_cells,) vertical structure

uhat_n = uhat_out[p, :, mode_n]                # (nt,) time series amplitude


# reconstruct: (n_cells x nt)
u_reconstructed = Ue_n * uhat_n'


fig = Figure(resolution=(1000,500))
ax = Axis(fig[1,1], xlabel="time step", ylabel="depth (m)",
          title="Mode $mode_n reconstructed u, mooring point $p")
hm = heatmap!(ax, 1:size(u_reconstructed,2), z_centers_col, u_reconstructed'; colormap=:balance)
Colorbar(fig[1,2], hm)
display(fig)


fig = Figure(resolution=(400,600))
ax = Axis(fig[1,1], xlabel="U eigen ", ylabel="depth (m)",
          title="Mode 2 vertical structure, mooring point $p")
lines!(ax, Ue_n, z_centers_col)
vlines!(ax, [0.0], color=:gray, linestyle=:dash)  # zero-crossing reference line
display(fig)

fig = Figure(resolution=(400,600))
ax = Axis(fig[1,1], xlabel="W eigen ", ylabel="depth (m)",
          title="Mode 1 vertical structure, mooring point $p")
lines!(ax, We_n, z_centers_col)
vlines!(ax, [0.0], color=:gray, linestyle=:dash)  # zero-crossing reference line
display(fig)







# ============================================================================
# MODAL BC FLUXES (time-averaged and depth-integrated, kW/m)
# ============================================================================
uflux_avg_out = fill(NaN, N_moor, n_modes_keep)
vflux_avg_out = fill(NaN, N_moor, n_modes_keep)
uflux_int_out = fill(NaN, N_moor, n_modes_keep)
vflux_wm = fill(NaN, N_moor, n_modes_keep)
uflux_wm = fill(NaN, N_moor, n_modes_keep)
vflux_int_out = fill(NaN, N_moor, n_modes_keep)
for p in 1:N_moor
   hfac_col = hFacC_moor[p, :]
   ocean_idx = findall(hfac_col .> 0)
   if length(ocean_idx) < min_ocean_cells
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
   uflux_wm[p, :] = uflux_avg_modes .* (H)
   vflux_wm[p, :] = vflux_avg_modes .* (H)
end
println("Modal flux calculation complete for all mooring points.")




# ====================
# Plot
# ====================
using MAT, TOML, CairoMakie
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg  = TOML.parsefile(config_file)
base = cfg["base_path"]
minlat, maxlat = 24.0, 31.91
minlon, maxlon = 193.0, 199.0
FIGDIR = cfg["fig_base"]
mkpath(FIGDIR)
cell_x = (maxlon - minlon) / NX
cell_y = (maxlat - minlat) / NY
target = 5f0 * Float32(min(cell_x, cell_y))
ARROW_SCALEUP = 5.0


# ════════════════════════════════════════════════════════════════════════
# 1) Mooring locations (must match the order used in the mooring extraction /
#    modal flux pipeline, so uflux_int_out[p,:] lines up with mooring_ids[p])
# ════════════════════════════════════════════════════════════════════════
mooring_ids = [82, 83, 84, 85]
target_lats = [25.4891, 27.7690, 28.8995, 30.1312]
target_lons = [194.8451, 196.0301, 196.5105, 197.1154]
iwap_idx    = [1, 2, 3, 4]
n_points    = length(iwap_idx)


# ════════════════════════════════════════════════════════════════════════
# 2) IWAP observed flux, per mode (no summing)
# ════════════════════════════════════════════════════════════════════════
file3path = "/home/aswathy/mnt/data/aswathy/Mooring_Data/Flux_mooring_timeseries_ALL_IWAP.mat"
f3 = matopen(file3path)
lato3 = vec(read(f3, "lato"))
lono3 = vec(read(f3, "lono"))
Fuo3  = read(f3, "Fuo")
Fvo3  = read(f3, "Fvo")
close(f3)
Fu_iwap_mode1 = [Fuo3[iwap_idx[p], 1] for p in 1:n_points]
Fv_iwap_mode1 = [Fvo3[iwap_idx[p], 1] for p in 1:n_points]
Fu_iwap_mode2 = [Fuo3[iwap_idx[p], 2] for p in 1:n_points]
Fv_iwap_mode2 = [Fvo3[iwap_idx[p], 2] for p in 1:n_points]

#Fu_o = Float64.(sum(uflux_int_out,dims=2))
#Fv_o = Float64.(sum(vflux_int_out,dims=2))

# ════════════════════════════════════════════════════════════════════════
# 3) Model flux, per mode (depth-integrated, kW/m) -- taken directly from
#    uflux_int_out / vflux_int_out computed earlier in this session,
#    no NetCDF read needed since nothing was written to disk
# ════════════════════════════════════════════════════════════════════════
for p in 1:n_points
    dlat = abs(lat[p] - target_lats[p])
    dlon = abs(lon[p] - target_lons[p])
    if dlat > 0.01 || dlon > 0.01
        @warn "Mooring point $p lat/lon ($(lat[p]), $(lon[p])) does not match expected mooring $(mooring_ids[p])"
    end
end
Fu_model_mode1 = Float64.(uflux_int_out[:, 1])
Fv_model_mode1 = Float64.(vflux_int_out[:, 1])
Fu_model_mode2 = Float64.(uflux_int_out[:, 2])
Fv_model_mode2 = Float64.(vflux_int_out[:, 2])

Fu_m = dropdims(Float64.(sum(uflux_int_out,dims=2)),dims=2)
Fv_m = dropdims(Float64.(sum(vflux_int_out,dims=2)),dims=2)

# ════════════════════════════════════════════════════════════════════════
# 4) Two-panel figure: Mode 1 (left) and Mode 2 (right), IWAP + model vectors
# ════════════════════════════════════════════════════════════════════════
fig = Figure(resolution = (1800, 800))
scale_x0 = minlon + 0.4
scale_y0 = maxlat - 0.4
mag1_iwap  = sqrt.(Fu_iwap_mode1.^2  .+ Fv_iwap_mode1.^2)
mag1_model = sqrt.(Fu_model_mode1.^2 .+ Fv_model_mode1.^2)
scale_ref_kWm1 = 2.0
mag2_iwap  = sqrt.(Fu_iwap_mode2.^2  .+ Fv_iwap_mode2.^2)
mag2_model = sqrt.(Fu_model_mode2.^2 .+ Fv_model_mode2.^2)
scale_ref_kWm2 =0.2
scale_ref_kWm = 2.0

scale_mode1 = (target / (scale_ref_kWm1)) * (ARROW_SCALEUP)
scale_mode2 = (target / (scale_ref_kWm2)) * (ARROW_SCALEUP)
scale_mode = (target / (scale_ref_kWm)) * (ARROW_SCALEUP)

mooring_pos = Point2f.((target_lons), (target_lats))


# --- Panel 1: Mode 1 ---
ax1 = Axis(fig[1, 1],
    aspect = DataAspect(),
    title      = "Mode 1 flux (mooring vs model)",
    xlabel     = "Longitude [°]",
    ylabel     = "Latitude [°]",
    xlabelsize = 18,
    ylabelsize = 18,
    titlesize  = 16)
ax1.limits[] = ((minlon, maxlon), (minlat, maxlat))
iwap_vecs1  = Vec2f.((Fu_iwap_mode1 .* scale_mode1), (Fv_iwap_mode1 .* scale_mode1))
model_vecs1 = Vec2f.(Float32.(Fu_model_mode1 .* scale_mode1), Float32.(Fv_model_mode1 .* scale_mode1))
arrows!(ax1, mooring_pos, iwap_vecs1;  color = :black,   arrowsize = 7, linewidth = 3)
arrows!(ax1, mooring_pos, model_vecs1; color = :magenta, arrowsize = 7, linewidth = 3)
scale_len1 = (scale_ref_kWm1 * scale_mode1)
lines!(ax1, [scale_x0, scale_x0 + scale_len1], [scale_y0, scale_y0]; color = :black, linewidth = 2.5)
arrows!(ax1, [Point2f(scale_x0, scale_y0)], [Vec2f(scale_len1, 0f0)]; color = :black, arrowsize = 7, linewidth = 2.5)
text!(ax1, scale_x0, scale_y0 - 0.25; text = "$(round(scale_ref_kWm1, digits=2)) kW/m", fontsize = 11, color = :black)


# --- Panel 2: Mode 2 ---
ax2 = Axis(fig[1, 2],
    aspect = DataAspect(),
    title      = "Mode 2 flux (mooring vs model)",
    xlabel     = "Longitude [°]",
    ylabel     = " ",
    yticklabelsvisible = false,
    xlabelsize = 18,
    ylabelsize = 18,
    titlesize  = 16)
ax2.limits[] = ((minlon, maxlon), (minlat, maxlat))
iwap_vecs2  = Vec2f.((Fu_iwap_mode2 .* scale_mode2), (Fv_iwap_mode2 .* scale_mode2))
model_vecs2 = Vec2f.(Float32.(Fu_model_mode2 .* scale_mode2), Float32.(Fv_model_mode2 .* scale_mode2))
arrows!(ax2, mooring_pos, iwap_vecs2;  color = :black,   arrowsize = 7, linewidth = 3)
arrows!(ax2, mooring_pos, model_vecs2; color = :magenta, arrowsize = 7, linewidth = 3)
scale_len2 = (scale_ref_kWm2 * scale_mode2)
lines!(ax2, [scale_x0, scale_x0 + scale_len2], [scale_y0, scale_y0]; color = :black, linewidth = 2.5)
arrows!(ax2, [Point2f(scale_x0, scale_y0)], [Vec2f(scale_len2, 0f0)]; color = :black, arrowsize = 7, linewidth = 2.5)
text!(ax2, scale_x0, scale_y0 - 0.25; text = "$(round(scale_ref_kWm2, digits=2)) kW/m", fontsize = 11, color = :black)


# --- Shared legend for arrow colors (IWAP vs model) ---
elem_iwap  = LineElement(color = :black,   linewidth = 3)
elem_model = LineElement(color = :magenta, linewidth = 3)
Legend(fig[2, 1:2], [elem_iwap, elem_model], ["IWAP (observed)", "Model"], orientation = :horizontal, tellwidth = false)
colgap!(fig.layout,1,2)
# --- Panel 3: Modal sum ---
ax3 = Axis(fig[1, 3],
    aspect = DataAspect(),
    title      = "Modal sum Vs Baroclinic flux ",
    xlabel     = "Longitude [°]",
    ylabel     = " ",
    yticklabelsvisible = false,
    xlabelsize = 18,
    ylabelsize = 18,
    titlesize  = 16)
ax3.limits[] = ((minlon, maxlon), (minlat, maxlat))
model_vec = Vec2f.(Float32.(Fu_m .* scale_mode), Float32.(Fv_m .* scale_mode))
bc_vec = Vec2f.(Float32.(Fu_bc .* scale_mode), Float32.(Fv_bc .* scale_mode))
size(bc_vec)
arrows!(ax3, mooring_pos, bc_vec;  color = :blue,   arrowsize = 15, linewidth = 6)
arrows!(ax3, mooring_pos, model_vec; color = :magenta, arrowsize = 5, linewidth = 3)
scale_len = (scale_ref_kWm * scale_mode)
lines!(ax3, [scale_x0, scale_x0 + scale_len], [scale_y0, scale_y0]; color = :black, linewidth = 2.5)
arrows!(ax3, [Point2f(scale_x0, scale_y0)], [Vec2f(scale_len, 0f0)]; color = :black, arrowsize = 7, linewidth = 2.5)
text!(ax3, scale_x0, scale_y0 - 0.25; text = "$(round(scale_ref_kWm, digits=2)) kW/m", fontsize = 11, color = :black)
colgap!(fig.layout,2,2)
elem_bc  = LineElement(color = :blue,   linewidth = 3)
Legend(fig[2, 3], [elem_bc, elem_model], ["undecomposed", "decomposed"], orientation = :horizontal, tellwidth = false)
println(Fu_bc)
println()
display(fig)
png_file = joinpath(FIGDIR, "Mooring_modes1_2_only.png")
save(png_file, fig)
println("Saved: $png_file")


# ════════════════════════════════════════════════════════════════════════
# 5) Second figure: Mode 1 & Mode 2 flux, LINEAR arrow lengths,
#    moorings labeled MP1..MP4, blue = IWAP, red = model
# ════════════════════════════════════════════════════════════════════════
fig2 = Figure(resolution = (1200, 800))
scale_x0 = minlon + 0.4
scale_y0 = maxlat - 0.4
mag1_iwap  = sqrt.(Fu_iwap_mode1.^2  .+ Fv_iwap_mode1.^2)
mag1_model = sqrt.(Fu_model_mode1.^2 .+ Fv_model_mode1.^2)
scale_ref_kWm1 = 2.0
mag2_iwap  = sqrt.(Fu_iwap_mode2.^2  .+ Fv_iwap_mode2.^2)
mag2_model = sqrt.(Fu_model_mode2.^2 .+ Fv_model_mode2.^2)
scale_ref_kWm2 = 0.5
scale_ref_kWm  = 2.0




scale_mode1 = (target / (scale_ref_kWm1)) * (ARROW_SCALEUP)
scale_mode2 = (target / (scale_ref_kWm2)) * (ARROW_SCALEUP)
scale_mode  = (target / (scale_ref_kWm))  * (ARROW_SCALEUP)




mooring_pos = Point2f.((target_lons), (target_lats))




mp_labels = ["MP$(p)" for p in 1:n_points]




# Linear scaling: preserves direction and magnitude proportionally (no log)
function linear_scaled_vecs(Fu, Fv, scale)
   return Vec2f.(Float32.(Fu .* scale), Float32.(Fv .* scale))
end




# ════════════════════════════════════════════════════════════════════════
# Rotate observed (IWAP) fluxes by 60° before plotting/comparing
# (see Zhao et al., 2010, page 3) — matches MATLAB's theta = pi/3 rotation.
# Model fluxes are NOT rotated, exactly as in the MATLAB script.
# ════════════════════════════════════════════════════════════════════════
function rotate_flux(Fu, Fv, theta)
   R = [cos(theta) -sin(theta);
        sin(theta)  cos(theta)]
   Fu_rot = similar(Fu)
   Fv_rot = similar(Fv)
   for i in eachindex(Fu)
       v = R * [Fu[i]; Fv[i]]
       Fu_rot[i] = v[1]
       Fv_rot[i] = v[2]
   end
   return Fu_rot, Fv_rot
end




theta_rot = pi/3   # 60 degrees, matches MATLAB's theta




Fu_iwap_mode1_R, Fv_iwap_mode1_R = rotate_flux(Fu_iwap_mode1, Fv_iwap_mode1, theta_rot)
Fu_iwap_mode2_R, Fv_iwap_mode2_R = rotate_flux(Fu_iwap_mode2, Fv_iwap_mode2, theta_rot)




iwap_linvecs1  = linear_scaled_vecs(Fu_iwap_mode1_R,  Fv_iwap_mode1_R,  scale_mode1)
model_linvecs1 = linear_scaled_vecs(Fu_model_mode1, Fv_model_mode1, scale_mode1)
iwap_linvecs2  = linear_scaled_vecs(Fu_iwap_mode2_R,  Fv_iwap_mode2_R,  scale_mode2)
model_linvecs2 = linear_scaled_vecs(Fu_model_mode2, Fv_model_mode2, scale_mode2)




# Fu_model_mode1/2, Fv_model_mode1/2 are left untouched — matches MATLAB
# (unearest/vnearest, i.e. the model fluxes, are never rotated there)




scale_x0 = minlon + 0.4
scale_y0 = maxlat - 0.4




# --- Panel 1: Mode 1 (linear arrows) ---
ax1b = Axis(fig2[1, 1],
   title      = "Mode 1 flux (linear-scaled arrows)",
   xlabel     = "Longitude [°]",
   ylabel     = "Latitude [°]",
   xlabelsize = 18, ylabelsize = 18, titlesize = 16)
ax1b.limits[] = ((minlon, 200), (minlat, maxlat))
arrows!(ax1b, mooring_pos, iwap_linvecs1;  color = :blue, arrowsize = 10, linewidth = 3)
arrows!(ax1b, mooring_pos, model_linvecs1; color = :red,  arrowsize = 10, linewidth = 3)
for p in 1:n_points
   text!(ax1b, target_lons[p] + 0.05, target_lats[p] + 0.05;
         text = mp_labels[p], fontsize = 13, color = :black)
end
# scale bar
sl_1 = scale_ref_kWm1 * scale_mode1
arrows!(ax1b, [Point2f(scale_x0, scale_y0)], [Vec2f(sl_1, 0f0)]; color = :black, arrowsize = 7, linewidth = 2.5)
text!(ax1b, scale_x0, scale_y0 - 0.25; text = "$(scale_ref_kWm1) kW/m", fontsize = 11, color = :black)




# --- Panel 2: Mode 2 (linear arrows) ---
ax2b = Axis(fig2[1, 2],
   title      = "Mode 2 flux (linear-scaled arrows)",
   xlabel     = "Longitude [°]",
   ylabel     = " ",
   yticklabelsvisible = false,
   xlabelsize = 18, ylabelsize = 18, titlesize = 16)
ax2b.limits[] = ((minlon, 200), (minlat, maxlat))
arrows!(ax2b, mooring_pos, iwap_linvecs2;  color = :blue, arrowsize = 10, linewidth = 3)
arrows!(ax2b, mooring_pos, model_linvecs2; color = :red,  arrowsize = 10, linewidth = 3)
for p in 1:n_points
   text!(ax2b, target_lons[p] + 0.05, target_lats[p] + 0.05;
         text = mp_labels[p], fontsize = 13, color = :black)
end
# scale bar
sl_2 = scale_ref_kWm2 * scale_mode2
arrows!(ax2b, [Point2f(scale_x0, scale_y0)], [Vec2f(sl_2, 0f0)]; color = :black, arrowsize = 7, linewidth = 2.5)
text!(ax2b, scale_x0, scale_y0 - 0.25; text = "$(scale_ref_kWm2) kW/m", fontsize = 11, color = :black)




elem_iwap_r  = LineElement(color = :blue, linewidth = 3)
elem_model_b = LineElement(color = :red,  linewidth = 3)
Legend(fig2[2, 1:2], [elem_iwap_r, elem_model_b], ["IWAP (observed)", "Model"],
      orientation = :horizontal, tellwidth = false)




display(fig2)
png_file2 = joinpath(FIGDIR, "Mooring_modes1_2_linear.png")
save(png_file2, fig2)
println("Saved: $png_file2")




