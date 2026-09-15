using DSP, Statistics, Printf, LinearAlgebra, TOML, NCDatasets, Impute, CairoMakie
include(joinpath(@__DIR__, "..", "..",  "functions", "FluxUtils.jl"))
using .FluxUtils: bandpassfilter
include(joinpath(@__DIR__, "..", "..", "functions", "densjmd95.jl"))
include(joinpath(@__DIR__, "..","..", "functions", "strum_liouville_noneqDZ_norm.jl"))


config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
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
# --- Thickness & constants ---


g = 9.81


mydir  = "/nobackup/avaliyap/V2/Moorings/"   # matches build_mooring_netcdf.jl's mydir
ncfile = joinpath(mydir, "Moorings_88_timeseries.nc")
ds = NCDataset(ncfile, "r")
thk =(open(joinpath(mydir,   "delR.bin"), "r") do io
               raw = read(io,  NZ * sizeof(Float32))
               ntoh.(reshape(reinterpret(Float32, raw), NZ))
           end)


DRF  = thk[1:NZ]
sum(thk)
try


lon = Array(ds["lon"])
lat = Array(ds["lat"])




# written as (time, station, depth); pipeline below needs (station, depth, time)
permute_to_std(x) = permutedims(Array(x), (2, 3, 1))




U     = Float64.(permute_to_std(ds["U_east"]))
V     = Float64.(permute_to_std(ds["V_north"]))
Salt  = Float64.(permute_to_std(ds["Salt"]))
Theta = Float64.(permute_to_std(ds["Theta"]))
hFacC = Float64.(Array(ds["hFacC"]))   # (station, depth) -- already given, no reconstruction




close(ds)
N_moor, nz, nt = size(U)
println("Loaded $N_moor mooring points, $nz levels, $nt timesteps.")




# ============================================================================
# STEP 0 (NEW): CHECK MOORING LOCATIONS BEFORE DOING ANYTHING EXPENSIVE
# ============================================================================
fig0 = Figure(resolution=(800, 700))
ax0 = Axis(fig0[1, 1], xlabel="Longitude [°]", ylabel="Latitude [°]",
          title="Mooring locations check ($N_moor points)", aspect=DataAspect())
scatter!(ax0, lon, lat, color=:dodgerblue, markersize=10)
for p in 1:N_moor
   text!(ax0, lon[p] + 0.02, lat[p] + 0.02; text=string(p), fontsize=9, color=:black)
end
display(fig0)
loc_png = joinpath(FIGDIR, "mooring_locations_check.png")
save(loc_png, fig0)
println("Saved mooring location check -> $loc_png")
println("Review this BEFORE trusting the flux results below.")




hFacC_moor = hFacC
mask2D = hFacC_moor .== 0
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




# ============================================================================
# SANITY CHECK
# ============================================================================
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




# ============================================================================
# SANITY CHECK
# ============================================================================
println("\nSanity check -- depth-integrated up_3d (should be ≈ 0):")
println(round.(sum(up_3d .* DRFfull_r, dims=2) ./ depth_r, digits=4))
println("Sanity check -- depth-integrated vp_3d (should be ≈ 0):")
println(round.(sum(vp_3d .* DRFfull_r, dims=2) ./ depth_r, digits=4))




# ============================================================================
# BC FLUXES (undecomposed, for reference/sanity check against modal sum later)
# ============================================================================
xflx_3d = up_3d .* pp_3d
yflx_3d = vp_3d .* pp_3d
Fu_b = dropdims(sum(xflx_3d .* DRFfull_r, dims=2), dims=2)
Fv_b = dropdims(sum(yflx_3d .* DRFfull_r, dims=2), dims=2)
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
       println("  mooring point $p/$N_moor skipped (no ocean cells -- land/dry column)")
       continue
   end
   k_top = ocean_idx[1]
   ibot  = ocean_idx[end]
   n_cells = ibot - k_top + 1




   N2_mean_col = [ (v = filter(!isnan, N2[p, k, :]); isempty(v) ? 1e-10 : mean(v))
                   for k in 1:nz ]




   dz_col = (hfac_col .* DRF)[k_top:ibot]
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




   println("  mooring point $p/$N_moor solved")
end




# ============================================================================
# SANITY CHECK: mode 1 & mode 2 orthonormality (Ueig_out), per mooring point
#   (1/H) ∫ Φ1·Φ1 dz ≈ 1   (1/H) ∫ Φ2·Φ2 dz ≈ 1   (1/H) ∫ Φ1·Φ2 dz ≈ 0
# ============================================================================
println("\nChecking mode 1 & mode 2 orthonormality (Ueig_out)...")
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




   U1 = @view Ueig_out[p, k_top:ibot, 1]
   U2 = @view Ueig_out[p, k_top:ibot, 2]
   if any(isnan, U1) || any(isnan, U2)
       continue
   end




   check_11 = sum(U1 .* U1 .* dz_col) / H
   check_22 = sum(U2 .* U2 .* dz_col) / H
   check_12 = sum(U1 .* U2 .* dz_col) / H




   println("Mooring point $p:")
   println("  mode1·mode1 = ", round(check_11, digits=4))
   println("  mode2·mode2 = ", round(check_22, digits=4))
   println("  mode1·mode2 = ", round(check_12, digits=6))
end




# ============================================================================
# SANITY CHECK: full mode orthonormality matrix, all n_modes_keep modes
#   (1/H) ∫ Φm·Φn dz  ==  1 if m==n,  0 if m!=n
# ============================================================================
println("\nChecking full mode orthonormality matrix (Ueig_out)...")
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




   Phi = @view Ueig_out[p, k_top:ibot, :]
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
# This step was missing from the script you pasted -- restored from your
# earlier full version so uflux_int_out / vflux_int_out actually get computed.
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
# SAVE (NEW): mode 1 & mode 2 fluxes with mooring lat/lon to NetCDF
# Static per-station fields (already time-averaged, depth-integrated) --
# no time dimension needed.
# ============================================================================
flux_outfile = joinpath(mydir, "Mooring_modal_fluxes.nc")
dsflux = NCDataset(flux_outfile, "c")




defDim(dsflux, "station", N_moor)




v_flat = defVar(dsflux, "lat", Float64, ("station",))
v_flat.attrib["long_name"] = "latitude"
v_flat.attrib["units"] = "degrees_north"
v_flat[:] = lat




v_flon = defVar(dsflux, "lon", Float64, ("station",))
v_flon.attrib["long_name"] = "longitude"
v_flon.attrib["units"] = "degrees_east"
v_flon[:] = lon




v_fu1 = defVar(dsflux, "Fu_mode1", Float64, ("station",))
v_fu1.attrib["long_name"] = "mode 1 eastward depth-integrated baroclinic flux"
v_fu1.attrib["units"] = "kW m-1"
v_fu1[:] = uflux_int_out[:, 1]




v_fv1 = defVar(dsflux, "Fv_mode1", Float64, ("station",))
v_fv1.attrib["long_name"] = "mode 1 northward depth-integrated baroclinic flux"
v_fv1.attrib["units"] = "kW m-1"
v_fv1[:] = vflux_int_out[:, 1]




v_fu2 = defVar(dsflux, "Fu_mode2", Float64, ("station",))
v_fu2.attrib["long_name"] = "mode 2 eastward depth-integrated baroclinic flux"
v_fu2.attrib["units"] = "kW m-1"
v_fu2[:] = uflux_int_out[:, 2]




v_fv2 = defVar(dsflux, "Fv_mode2", Float64, ("station",))
v_fv2.attrib["long_name"] = "mode 2 northward depth-integrated baroclinic flux"
v_fv2.attrib["units"] = "kW m-1"
v_fv2[:] = vflux_int_out[:, 2]




close(dsflux)
println("Saved mode 1 & mode 2 fluxes with mooring lat/lon -> $flux_outfile")




# ============================================================================
# PLOT (NEW): mode 1 and mode 2 flux vectors at every mooring location.
# Auto-scales to whatever spatial extent / flux magnitude your 88 points
# actually have -- not hardcoded to the earlier 4-point IWAP subregion.
# ============================================================================
function plot_modal_flux_map(lon, lat, Fu, Fv, mode_num; figdir=FIGDIR)
   valid = .!isnan.(Fu) .& .!isnan.(Fv)
   n_valid = count(valid)
   if n_valid == 0
       @warn "Mode $mode_num: no valid (non-NaN) flux values to plot."
       return nothing
   end




   mag = sqrt.(Fu[valid].^2 .+ Fv[valid].^2)
   scale_ref = mean(mag) > 0 ? mean(mag) : 1.0




   lon_pad = 0.05 * (maximum(lon) - minimum(lon) + 1e-6)
   lat_pad = 0.05 * (maximum(lat) - minimum(lat) + 1e-6)
   xlims = (minimum(lon) - lon_pad, maximum(lon) + lon_pad)
   ylims = (minimum(lat) - lat_pad, maximum(lat) + lat_pad)




   target = 0.03 * max(xlims[2] - xlims[1], ylims[2] - ylims[1])
   scale = target / scale_ref




   fig = Figure(resolution=(900, 800))
   ax = Axis(fig[1, 1], aspect=DataAspect(), xlabel="Longitude [°]", ylabel="Latitude [°]",
             title="Mode $mode_num depth-integrated flux ($n_valid / $(length(lon)) moorings)")
   ax.limits[] = (xlims, ylims)




   pos  = Point2f.(lon[valid], lat[valid])
   vecs = Vec2f.(Float32.(Fu[valid] .* scale), Float32.(Fv[valid] .* scale))
   arrows!(ax, pos, vecs; color=:crimson, arrowsize=8, linewidth=2)
   scatter!(ax, lon[.!valid], lat[.!valid]; color=:gray, markersize=6)  # NaN points, greyed out




   sx0 = xlims[1] + 0.05 * (xlims[2] - xlims[1])
   sy0 = ylims[2] - 0.05 * (ylims[2] - ylims[1])
   sl  = scale_ref * scale
   arrows!(ax, [Point2f(sx0, sy0)], [Vec2f(sl, 0f0)]; color=:black, arrowsize=8, linewidth=2)
   text!(ax, sx0, sy0 - 0.02 * (ylims[2] - ylims[1]);
         text="$(round(scale_ref, digits=3)) kW/m", fontsize=11)




   display(fig)
   outpng = joinpath(figdir, "Mooring_flux_mode$(mode_num)_all$(length(lon)).png")
   save(outpng, fig)
   println("Saved: $outpng")
   return fig
end




plot_modal_flux_map(lon, lat, uflux_int_out[:, 1], vflux_int_out[:, 1], 1)
plot_modal_flux_map(lon, lat, uflux_int_out[:, 2], vflux_int_out[:, 2], 2)


catch e
    # Make sure a failure is actually visible in run_log.txt, instead of
    # vanishing when logio gets closed in `finally` below.
    println(logio, "\n==================== UNCAUGHT ERROR ====================")
    println(logio, sprint(showerror, e, catch_backtrace()))
    println(logio, "==========================================================")
    flush(logio)
    rethrow()
finally
    flush(logio)
    close(logio)
end


