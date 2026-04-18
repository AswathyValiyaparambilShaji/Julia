using DSP, Statistics, Printf, CairoMakie, LinearAlgebra, TOML, FilePathsBase
include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin

config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base  = cfg["base_path"]
base2 = cfg["base_path2"]

# --- Domain & grid ---
buf = 3
tx, ty = 47, 66
nx = tx + 2*buf
ny = ty + 2*buf
nz = 88
dto  = 144
Tts  = 366192
nt   = div(Tts, dto)

timesteps_per_3days = 72
nt_avg = div(nt, timesteps_per_3days)

# --- Single tile & point to inspect ---
xn, yn = 1, 1                  # tile indices
ix, iy, iz = 25, 25, 10        # spatial point (interior, away from buffer)

suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
println("Inspecting tile: $suffix  |  point: (ix=$ix, iy=$iy, iz=$iz)")

# =========================================================
# 1. Raw U → C-grid interpolated to cell center
# =========================================================
println("Reading raw U...")
U_raw = Float64.(read_bin(joinpath(base, "U", "U_$suffix.bin"), (nx, ny, nz, nt)))

uc = 0.5 .* (U_raw[1:end-1, :, :, :] .+ U_raw[2:end, :, :, :])
ucc_raw = cat(uc, zeros(1, ny, nz, nt); dims=1)   # (nx, ny, nz, nt)
u_raw_ts = ucc_raw[ix, iy, iz, :]                 # length nt
U_raw = nothing; uc = nothing; ucc_raw = nothing; GC.gc()

# =========================================================
# 2. Low-pass filtered U  (same C-grid interp already applied)
# =========================================================
println("Reading LP-filtered U...")
u_lp_full = Float64.(read_bin(joinpath(base2, "UVW_LP", "u_lp_$suffix.bin"), (nx, ny, nz, nt)))
u_lp_ts = u_lp_full[ix, iy, iz, :]               # length nt
u_lp_full = nothing; GC.gc()

# =========================================================
# 3. 3-day mean U  (C-grid interp already applied)
# =========================================================
println("Reading 3-day mean U...")
u_3day_full = Float64.(read_bin(joinpath(base, "3day_mean", "U", "ucc_3day_$suffix.bin"), (nx, ny, nz, nt_avg)))
u_3day_ts = u_3day_full[ix, iy, iz, :]            # length nt_avg
u_3day_full = nothing; GC.gc()

# =========================================================
# 4. Time axes  [hours]
# =========================================================
t_hr   = collect(0:nt-1)           .* 1.0          # raw & LP: hourly
t_3day = collect(0:nt_avg-1)       .* 72.0 .+ 36.0 # 3-day mean: centred on each window

# =========================================================
# 5. Plot
# =========================================================
println("Plotting...")
fig = Figure(size = (1100, 500))
ax  = Axis(fig[1, 1],
    xlabel = "Time [hours]",
    ylabel = "u  [m/s]",
    title  = "U velocity comparison  tile $suffix, point (ix=$ix, iy=$iy, iz=$iz)")

lines!(ax, t_hr,   u_raw_ts;  color = :grey,      linewidth = 1.0, label = "Raw (C-grid centre)")
lines!(ax, t_hr,   u_lp_ts;   color = :steelblue,  linewidth = 1.8, label = "LP 36 hr")
scatter!(ax, t_3day, u_3day_ts; color = :crimson, markersize = 6,   label = "3-day mean")
lines!(ax,   t_3day, u_3day_ts; color = :crimson, linewidth = 1.4)

axislegend(ax; position = :rt)

save(joinpath(base2, "u_comparison_$(suffix)_ix$(ix)_iy$(iy)_iz$(iz).png"), fig)
println("Saved figure.")
display(fig)