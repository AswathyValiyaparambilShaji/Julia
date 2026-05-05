

using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML, CairoMakie


include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter


config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg   = TOML.parsefile(config_file)
base  = cfg["base_path"]
base_sm = cfg["base_path2"]
base_nt = cfg["base_path_nt"]


buf      = 3
tx, ty   = 47, 66
nx, ny   = tx + 2*buf, ty + 2*buf
nz       = 88
dto      = 144
Tts      = 366192
nt       = div(Tts, dto)
hrs_3day = 72
nt_avg   = div(nt, hrs_3day)


ρ0 = 1027.5
g  = 9.8


thk   = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF   = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)


# ── Depth coordinate: cell-centre depth (positive downward, surface = 0) ──────
depth_m = cumsum(DRF) .- DRF ./ 2   # mid-point of each level [m]


# ── User settings: one tile, one point, one depth ────────────────────────────
xn = 1          # tile x index
yn = 1          # tile y index
ix = 12         # x point within tile interior (1:tx)
iy = 10         # y point within tile interior (1:ty)
iz = 10         # depth level to inspect in time-series panel (1:nz)
it = 144        # timestep for vertical profile panel
# ─────────────────────────────────────────────────────────────────────────────


suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)


# ── Load NIW+Tides (NT) ───────────────────────────────────────────────────────
fu_nt = Float64.(open(joinpath(base_nt, "UVW_NT", "fu_nt_$suffix.bin"), "r") do io
   raw_bytes = read(io, nx * ny * nz * nt * sizeof(Float32))
   reshape(reinterpret(Float32, raw_bytes), nx, ny, nz, nt)
end)
fv_nt = Float64.(open(joinpath(base_nt, "UVW_NT", "fv_nt_$suffix.bin"), "r") do io
   raw_bytes = read(io, nx * ny * nz * nt * sizeof(Float32))
   reshape(reinterpret(Float32, raw_bytes), nx, ny, nz, nt)
end)
fw_nt = Float64.(open(joinpath(base_nt, "UVW_NT", "fw_nt_$suffix.bin"), "r") do io
   raw_bytes = read(io, nx * ny * nz * nt * sizeof(Float32))
   reshape(reinterpret(Float32, raw_bytes), nx, ny, nz, nt)
end)


# ── Load Semi-diurnal (SM) ────────────────────────────────────────────────────
fu_sm = Float64.(open(joinpath(base_sm, "UVW_F", "fu_$suffix.bin"), "r") do io
   raw_bytes = read(io, nx * ny * nz * nt * sizeof(Float32))
   reshape(reinterpret(Float32, raw_bytes), nx, ny, nz, nt)
end)
fv_sm = Float64.(open(joinpath(base_sm, "UVW_F", "fv_$suffix.bin"), "r") do io
   raw_bytes = read(io, nx * ny * nz * nt * sizeof(Float32))
   reshape(reinterpret(Float32, raw_bytes), nx, ny, nz, nt)
end)
fw_sm = Float64.(open(joinpath(base_sm, "UVW_F", "fw_$suffix.bin"), "r") do io
   raw_bytes = read(io, nx * ny * nz * nt * sizeof(Float32))
   reshape(reinterpret(Float32, raw_bytes), nx, ny, nz, nt)
end)


# ── Diagnostic: time series + vertical profile at one point ──────────────────
px = ix + buf
py = iy + buf


time_hrs = (0:nt-1) .* (dto / 3600.0)   # hours
it_hr    = (it - 1) * dto / 3600.0      # hour of the profile snapshot


vars_nt = [fu_nt, fv_nt, fw_nt]
vars_sm = [fu_sm, fv_sm, fw_sm]
labels  = ["u", "v", "w"]
units   = ["m/s", "m/s", "m/s"]


fig = Figure(size = (1600, 900))


# ── Column headers ────────────────────────────────────────────────────────────
Label(fig[0, 1]; text = "Time series  (iz = $iz,  depth ≈ $(round(Int, depth_m[iz])) m)",
      fontsize = 14, font = :bold, tellwidth = false)
Label(fig[0, 2]; text = "Vertical profile  (it = $it,  t ≈ $(round(it_hr, digits=1)) hrs)",
      fontsize = 14, font = :bold, tellwidth = false)


for (k, (vnt, vsm, lbl, unt)) in enumerate(zip(vars_nt, vars_sm, labels, units))


    # ── Left panel: time series at fixed (ix, iy, iz) ────────────────────────
    ts_nt = vec(vnt[px, py, iz, :])
    ts_sm = vec(vsm[px, py, iz, :])


    ax_ts = Axis(fig[k, 1];
        title     = "$lbl  |  tile($xn,$yn)  ix=$ix  iy=$iy  iz=$iz",
        xlabel    = "Time (hrs)",
        ylabel    = "$lbl  [$unt]",
        titlesize = 12)


    lines!(ax_ts, time_hrs, ts_nt; color = :royalblue,  linewidth = 1.2, label = "NIW+Tides (NT)")
    lines!(ax_ts, time_hrs, ts_sm; color = :orangered,  linewidth = 1.2, label = "Semi-diurnal (SM)")
    # mark the profile timestep
    vlines!(ax_ts, [it_hr]; color = :black, linewidth = 1.0, linestyle = :dash, label = "Profile t")
    axislegend(ax_ts; position = :rt, labelsize = 10)


    # ── Right panel: vertical profile at fixed (ix, iy, it) ──────────────────
    prof_nt = vec(vnt[px, py, :, it])
    prof_sm = vec(vsm[px, py, :, it])


    ax_pf = Axis(fig[k, 2];
        title     = "$lbl profile  |  tile($xn,$yn)  ix=$ix  iy=$iy  it=$it",
        xlabel    = "$lbl  [$unt]",
        ylabel    = "Depth (m)",
        titlesize = 12,
        yreversed = true)          # surface at top


    lines!(ax_pf, prof_nt, depth_m; color = :royalblue, linewidth = 1.5, label = "NIW+Tides (NT)")
    lines!(ax_pf, prof_sm, depth_m; color = :orangered, linewidth = 1.5, label = "Semi-diurnal (SM)")
    vlines!(ax_pf, [0.0];          color = :grey70,     linewidth = 0.8, linestyle = :dash)
    hlines!(ax_pf, [depth_m[iz]];  color = :black,      linewidth = 0.8, linestyle = :dot,
            label = "iz=$iz")
    axislegend(ax_pf; position = :rb, labelsize = 10)
end


save("timeseries_profile_$(suffix)_ix$(ix)_iy$(iy)_iz$(iz)_it$(it).png", fig)
display(fig)
@printf("Saved → timeseries_profile_%s_ix%d_iy%d_iz%d_it%d.png\n", suffix, ix, iy, iz, it)
@printf("Point info: px=%d  py=%d  iz=%d  it=%d  nt=%d  dto=%ds\n", px, py, iz, it, nt, dto)
@printf("Profile snapshot: t ≈ %.1f hrs,  iz=%d depth ≈ %.0f m\n", it_hr, iz, depth_m[iz])


