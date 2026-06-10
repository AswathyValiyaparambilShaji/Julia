using Printf, TOML, CairoMakie


include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin


# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
config_file = get(ENV, "JULIA_CONFIG",
    joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg  = TOML.parsefile(config_file)
base = cfg["base_path"]


# ---------------------------------------------------------------
# Grid / tile parameters  (must match the filter script exactly)
# ---------------------------------------------------------------
buf  = 3
tx, ty = 47, 66
nx   = tx + 2*buf
ny   = ty + 2*buf
nz   = 88
dto  = 144
Tts  = 366192
nt   = div(Tts, dto)


# ---------------------------------------------------------------
# Which tile to inspect  ← change these to any tile you want
# ---------------------------------------------------------------
xn, yn = cfg["xn_start"], cfg["yn_start"]
suffix  = @sprintf("%02dx%02d_%d", xn, yn, buf)


# ---------------------------------------------------------------
# Which interior point to inspect  (1-based, within the padded tile)
# Centre of tile is the safest default.  Change iz for a deeper level.
# ---------------------------------------------------------------
ix, iy, iz = (nx ÷ 2) + 1, (ny ÷ 2) + 1, 1


@info "Inspecting tile $suffix  at grid point (ix=$ix, iy=$iy, iz=$iz)"


# ---------------------------------------------------------------
# Read RAW fields and replicate C-grid → centre interpolation
# ---------------------------------------------------------------
U_raw = read_bin(joinpath(base, "U", "U_$suffix.bin"),  (nx, ny, nz, nt))
V_raw = read_bin(joinpath(base, "V", "V_$suffix.bin"),  (nx, ny, nz, nt))
W_raw = read_bin(joinpath(base, "W", "W_$suffix.bin"),  (nx, ny, nz, nt))


uc_raw = 0.5 .* (U_raw[1:end-1, :, :, :] .+ U_raw[2:end,   :, :, :])
vc_raw = 0.5 .* (V_raw[:, 1:end-1, :, :] .+ V_raw[:, 2:end, :, :])
wc_raw = 0.5 .* (W_raw[:, :, 1:end-1, :] .+ W_raw[:, :, 2:end, :])


ucc_raw = cat(uc_raw, zeros(Float32, 1, ny, nz, nt); dims=1)
vcc_raw = cat(vc_raw, zeros(Float32, nx, 1, nz, nt); dims=2)
wcc_raw = cat(wc_raw, zeros(Float32, nx, ny, 1, nt); dims=3)


# ---------------------------------------------------------------
# Read FILTERED fields  (no filtering here — just read the saved bins)
# ---------------------------------------------------------------
fu = read_bin(joinpath(base, "NT", "UVW_NT", "fu_nt_$suffix.bin"), (nx, ny, nz, nt))
fv = read_bin(joinpath(base, "NT", "UVW_NT", "fv_nt_$suffix.bin"), (nx, ny, nz, nt))
fw = read_bin(joinpath(base, "NT", "UVW_NT", "fw_nt_$suffix.bin"), (nx, ny, nz, nt))


# ---------------------------------------------------------------
# Extract single time series at (ix, iy, iz)
# ---------------------------------------------------------------
u_ts  = vec(ucc_raw[ix, iy, iz, :])
v_ts  = vec(vcc_raw[ix, iy, iz, :])
w_ts  = vec(wcc_raw[ix, iy, iz, :])


fu_ts = vec(fu[ix, iy, iz, :])
fv_ts = vec(fv[ix, iy, iz, :])
fw_ts = vec(fw[ix, iy, iz, :])


t = 1:nt   # time index (each step = dto = 144 s)


# ---------------------------------------------------------------
# Ringing extent estimate
# N=4-tap Butterworth, longest period T2=32.2 hr → ~4×32 = ~129 steps
# ---------------------------------------------------------------
T2_steps = round(Int, 32.2)   # longest filter period in time steps (delt=1 hr here mapped to dto)
N_taps   = 4
ring_est = N_taps * T2_steps
@info "Estimated ringing zone: ~$ring_est time steps each end"


# ---------------------------------------------------------------
# Plot
# ---------------------------------------------------------------
fig = Figure(resolution=(1500, 950), fontsize=13)


vars   = [("U", u_ts, fu_ts), ("V", v_ts, fv_ts), ("W", w_ts, fw_ts)]
colors = (:steelblue, :firebrick)


for (row, (vname, raw, filt)) in enumerate(vars)
    ax = Axis(fig[row, 1],
        title  = "$vname — tile $suffix  (ix=$ix, iy=$iy, iz=$iz)",
        xlabel = "Time step  (×$(dto) s)",
        ylabel = "$vname  [m s⁻¹]")


    lines!(ax, t, raw;  color=colors[1], linewidth=0.7, label="Raw")
    lines!(ax, t, filt; color=colors[2], linewidth=1.0, label="NT filtered")


    vspan!(ax, 1,            ring_est; color=(:orange, 0.18), label="Est. ringing zone")
    vspan!(ax, nt-ring_est,  nt;       color=(:orange, 0.18))


    axislegend(ax; position=:rt, framevisible=false)
end


# --- Zoom: left end (U) ---
ax_zl = Axis(fig[1, 2],
    title  = "U — LEFT-end zoom  (first $(2*ring_est) steps)",
    xlabel = "Time step", ylabel = "U  [m s⁻¹]")
zl = 1:min(2*ring_est, nt)
lines!(ax_zl, zl, u_ts[zl];  color=colors[1], linewidth=0.8, label="Raw")
lines!(ax_zl, zl, fu_ts[zl]; color=colors[2], linewidth=1.0, label="Filtered")
vspan!(ax_zl, 1, ring_est; color=(:orange, 0.28), label="Est. ringing")
axislegend(ax_zl; position=:rt, framevisible=false)


# --- Zoom: right end (U) ---
ax_zr = Axis(fig[2, 2],
    title  = "U — RIGHT-end zoom  (last $(2*ring_est) steps)",
    xlabel = "Time step", ylabel = "U  [m s⁻¹]")
zr = max(1, nt-2*ring_est+1):nt
lines!(ax_zr, zr, u_ts[zr];  color=colors[1], linewidth=0.8, label="Raw")
lines!(ax_zr, zr, fu_ts[zr]; color=colors[2], linewidth=1.0, label="Filtered")
vspan!(ax_zr, nt-ring_est, nt; color=(:orange, 0.28), label="Est. ringing")
axislegend(ax_zr; position=:rt, framevisible=false)


# --- Residual panel: fu - u  (ringing shows up as spurious oscillations) ---
ax_d = Axis(fig[3, 2],
    title  = "Residual  (fu − u)  — ringing appears as spurious signal",
    xlabel = "Time step", ylabel = "Δ [m s⁻¹]")
lines!(ax_d, t, fu_ts .- u_ts; color=:darkgreen, linewidth=0.7)
vspan!(ax_d, 1,           ring_est; color=(:orange, 0.28), label="Est. ringing")
vspan!(ax_d, nt-ring_est, nt;       color=(:orange, 0.28))
axislegend(ax_d; position=:rt, framevisible=false)
display(fig)

# ---------------------------------------------------------------
# Save
# ---------------------------------------------------------------
outpath = joinpath(base, "NT", "ringing_check_$(suffix)_iz$(iz).png")
save(outpath, fig; px_per_unit=2)
@info "Saved → $outpath"


# ---------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------
println("\n====== Ringing diagnostic summary ======")
println("Tile              : $suffix")
println("Inspect point     : (ix=$ix, iy=$iy, iz=$iz)")
println("Total time steps  : $nt  ($(nt*dto/3600) hr)")
println("Est. ringing zone : $ring_est steps each end  ($(ring_est*dto/3600) hr)")
println("Safe interior     : steps $(ring_est+1)–$(nt-ring_est)  ($(nt-2*ring_est) steps)")
println("========================================\n")




