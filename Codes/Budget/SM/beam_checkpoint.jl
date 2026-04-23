using NCDatasets, CairoMakie, Printf


# ============================================================================
# SET THIS to your locally downloaded NC file path
# ============================================================================
ncfile  = "mnt/data/aswathy/MITgcm_NAS/beam_UVrho.nc"   # ← change this
figfile = "mnt/data/aswathy/MITgcm_NAS/beam_profiles.png"           # ← change this
t_plot  = 500          # time index to plot


# ============================================================================
# CHECK — print file contents before plotting
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
        # print range for numeric scalars/vectors only
        if ndims(v) <= 2
            data = v[:]
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


lons    = ds["longitude"][:]
lats    = ds["latitude"][:]
U       = ds["U"][:, :, :]
V       = ds["V"][:, :, :]
rho     = ds["rho_insitu"][:, :, :]
DRFfull = ds["DRFfull"][:, :]


close(ds)


N_beam, nz, nt = size(U)
println("Loaded: N_beam=$N_beam  nz=$nz  nt=$nt")
println("Plotting time index t=$t_plot  (of $nt total)\n")


# ============================================================================
# DEPTH AXIS — cell-centre depths from DRFfull
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
U_t   = U[:, :, t_plot]
V_t   = V[:, :, t_plot]
rho_t = rho[:, :, t_plot]


# ============================================================================
# PLOT — N_beam colors cycled from :tab20
# ============================================================================
colors = cgrad(:tab20, N_beam, categorical=true)


fig = Figure(size=(1300, 700))


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
    lines!(ax_r, rho_t[i, :], d, color=col, linewidth=1.5)


    vlines!(ax_u, 0; color=:black, linewidth=0.5, linestyle=:dash)
    vlines!(ax_v, 0; color=:black, linewidth=0.5, linestyle=:dash)
end


xlims!(ax_r, 1020, 1060)


Legend(fig[1, 4], ax_u, "Beam points", framevisible=true, labelsize=10)


Label(fig[0, :],
    "Beam vertical profiles  —  time index t = $t_plot  (N_beam = $N_beam)",
    fontsize=14, font=:bold)

display(fig)
save(figfile, fig, px_per_unit=2)
println("Saved: $figfile")




