using NCDatasets, CairoMakie, Printf, TOML


# ============================================================================
# CONFIG
# ============================================================================
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base2 = cfg["base_path2"]


ncfile  = joinpath(base2, "beam_UVrho.nc")
figfile = joinpath(base2, "beam_profiles.png")
t_plot  = 500          # time index to plot (change as needed)


# ============================================================================
# READ
# ============================================================================
ds = NCDatasets.Dataset(ncfile, "r")


lons    = ds["longitude"][:]          # (N_beam,)
lats    = ds["latitude"][:]           # (N_beam,)
U       = ds["U"][:, :, :]           # (N_beam, nz, nt)
V       = ds["V"][:, :, :]           # (N_beam, nz, nt)
rho     = ds["rho_insitu"][:, :, :]  # (N_beam, nz, nt)
DRFfull = ds["DRFfull"][:, :]        # (N_beam, nz)


close(ds)


N_beam, nz, nt = size(U)
println("Read: N_beam=$N_beam  nz=$nz  nt=$nt")
println("Plotting time index t=$t_plot")


# ============================================================================
# DEPTH AXIS — cell-centre depths from DRFfull for each beam point
# ============================================================================
depth_c = zeros(Float64, N_beam, nz)
for i in 1:N_beam
    cumthk = 0.0
    for k in 1:nz
        depth_c[i, k] = cumthk + 0.5 * DRFfull[i, k]
        cumthk        += DRFfull[i, k]
    end
end


# ============================================================================
# EXTRACT PROFILES AT t_plot
# ============================================================================
U_t   = U[:, :, t_plot]    # (N_beam, nz)
V_t   = V[:, :, t_plot]
rho_t = rho[:, :, t_plot]


# ============================================================================
# PLOT
# ============================================================================
colors = cgrad(:tab10, N_beam, categorical=true)


fig = Figure(size=(1100, 700))


ax_u = Axis(fig[1, 1],
    xlabel    = "U  (m/s)",
    ylabel    = "Depth  (m)",
    title     = "Zonal velocity",
    yreversed = true)


ax_v = Axis(fig[1, 2],
    xlabel    = "V  (m/s)",
    title     = "Meridional velocity",
    yreversed = true)


ax_r = Axis(fig[1, 3],
    xlabel    = "ρ  (kg/m³)",
    title     = "In-situ density",
    yreversed = true)


for i in 1:N_beam
    label = @sprintf("%.2f°N, %.2f°E", lats[i], lons[i])
    col   = colors[i]
    d     = depth_c[i, :]


    lines!(ax_u, U_t[i, :],   d, color=col, linewidth=1.5, label=label)
    lines!(ax_v, V_t[i, :],   d, color=col, linewidth=1.5)
    lines!(ax_r, rho_t[i, :],d,  color=col, linewidth=1.5)


    vlines!(ax_u, 0; color=:black, linewidth=0.5, linestyle=:dash)
    vlines!(ax_v, 0; color=:black, linewidth=0.5, linestyle=:dash)
end



xlims!(ax_r, 1020, 1060)


Legend(fig[1, 4], ax_u, "Beam points", framevisible=true, labelsize=11)


Label(fig[0, :],
    "Beam vertical profiles  —  time index t = $t_plot",
    fontsize=14, font=:bold)

display(fig)
save(figfile, fig, px_per_unit=2)
println("Saved: $figfile")




