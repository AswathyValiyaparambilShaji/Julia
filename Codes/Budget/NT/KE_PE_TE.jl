using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML, CairoMakie,Dates


include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin
include(joinpath(@__DIR__, "..", "..", "..", "functions", "coriolis_frequency.jl"))


# ============================================================
# NOTES ON WHAT WAS FIXED / ASSUMED (read before running)
# ============================================================
# 1. Your original global accumulators (Conv, FDiv, ET_full, ...) were
#    initialized as 2-D (NX, NY). But you assign 3-D tile data
#    (te_mean[...] -> nx,ny,nt3-2) into them, and later call
#    area_avg(F, ...) treating dim 3 as time. They MUST be 3-D
#    (NX, NY, nt3-2). Fixed below.
# 2. `nz`, `DRF3d`, `nt3`, `ts`, `ke_di`, `pe_di`, `te_mean`, `rho0`
#    were referenced but never defined/computed in your snippet.
#    I've defined them with reasonable, documented assumptions —
#    check the "ASSUMPTION" comments and adjust paths/values to match
#    your actual pipeline.
# 3. I only build KE, APE, and the Tendency (ET) term end-to-end,
#    per your request, using the same plotting theme as your
#    multi-term example.
# ============================================================


# Load configuration
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base  = cfg["base_path"]
base2 = cfg["base_path_nt"]


rho0 = get(cfg, "rho0", 1027.5)   # ASSUMPTION: reference density [kg/m^3], adjust/override in TOML as cfg["rho0"]


# --- Domain & grid ---
NX, NY = 288, 468
minlat, maxlat = 24.0, 31.91
minlon, maxlon = 193.0, 199.0
lat = range(minlat, maxlat, length=NY)
lon = range(minlon, maxlon, length=NX)


# --- Tile parameters ---
buf = 3
tx, ty = 47, 66
nx = tx + 2*buf
ny = ty + 2*buf
nz = 88

# --- Time parameters ---
dt  = 25        # model timestep [s]
dto = 144       # output cadence in timesteps
Tts = 366192    # total model timesteps
nt  = div(Tts, dto)               # number of raw output snapshots
ts  = 72                     # timesteps per 3-day period (3*24)
nt_avg = div(nt, ts)         # number of 3-day periods (same as nt3)
nt3 = div(nt, 3*24)
nt_chunk = 72
n_chunks = div(nt, nt_chunk)
rho0 = 1027.5

# --- Date axis ---
t_origin = DateTime(2012, 3, 4, 0, 0, 0)
# Each 3-day period: centre date at origin + (i-1)*3 days + 1.5 days offset
dates_3day = [t_origin + Day(3*(i-1)) + Hour(36) for i in 1:nt3-2]


# Convert dates → numeric (days since origin) for plotting, keep DateTime for labels
t_numeric  = [Dates.value(d - t_origin) / (1000*3600*24) for d in dates_3day]  # milliseconds→days


# Tick positions: every ~15 days (every 5th 3-day period), formatted as "Mon DD\nYYYY"
tick_every  = 5
tick_inds   = 1:tick_every:nt3-2
tick_vals   = t_numeric[tick_inds]
tick_labels = [Dates.format(dates_3day[i], "u dd\nyyyy") for i in tick_inds]
ring_steps = nt_chunk
t_safe_start = ring_steps + 1              # first valid step (1801)
t_safe_end   = nt - ring_steps             # last  valid step (nt-1800)


# Safe 3-day chunks: only keep chunks that fall entirely within the safe range
safe_chunks = [c for c in 1:n_chunks
               if (c-1)*nt_chunk + 1 >= t_safe_start &&
                  c*nt_chunk          <= t_safe_end]

# --- Thickness ---
thk  = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF  = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)


# --- Global accumulators (NOW 3-D: space + time) ---
KE_full = zeros(NX, NY, nt3-2)
PE_full = zeros(NX, NY, nt3-2)
ET_full = zeros(NX, NY, nt3-2)
FH      = zeros(NX, NY)   # total water column depth (for normalization), 2-D static field
RAC     = zeros(NX, NY)   # cell area, 2-D static field


println("Loading energy budget terms (KE, APE, Tendency)...")


# ==========================================================
# ============ LOAD ALL TERMS ==============================
# ==========================================================
for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix  = @sprintf("%02dx%02d_%d", xn, yn, buf)


        hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
        dx    = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
        dy    = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))


        DRFfull = hFacC .* DRF3d
        depth   = dropdims(sum(DRFfull, dims=3), dims=3)   # (nx,ny) water column depth
        DRFfull[hFacC .== 0] .= 0.0
        rac = dx .* dy


        # --- Read Energy Tendency (already 3-day averaged & depth-integrated on disk) ---
        te_3day = Float64.(open(joinpath(base2, "TE_t_3day", "te_t_3day_nt_$suffix.bin"), "r") do io
            nbytes = nx*ny*nt3*sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt3-2)
        end)
        te_mean = te_3day   # already 3-day mean, depth-integrated -> (nx,ny,nt3-2)


        # ---- Read KE (raw, depth-resolved, full time resolution) ----
        println("  Reading KE...")
        ke_raw = Float64.(open(joinpath(base2, "KE", "ke_t_nt_$suffix.bin"), "r") do io
            nbytes = nx * ny * nz * nt * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nz, nt)
        end)


        # ---- Read APE (raw, depth-resolved, full time resolution) ----
        ape_raw = Float64.(open(joinpath(base2, "APE", "APE_t_nt_$suffix.bin"), "r") do io
            nbytes = nx * ny * nz * nt * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nz, nt)
        end)

        ape_clean = replace(ape_raw, NaN => 0.0)
               DRFfull4 = reshape(DRFfull, nx, ny, nz, 1)

       pe_di     = dropdims(sum(ape_clean .* DRFfull4, dims=3), dims=3)  # nx x ny x nt

        # ---- Depth-integrate KE / APE using hFacC-weighted cell thickness ----
        ke_di = dropdims(sum(ke_raw .* reshape(DRFfull, nx, ny, nz, 1), dims=3), dims=3)   # (nx,ny,nt)


        # ---- 3-day mean of depth-integrated KE and APE ----
        ke_3day = zeros(nx, ny, nt3-2)
        pe_3day = zeros(nx, ny, nt3-2)
        for t in 2:nt3-1
            t_start = (t-1)*ts + 1
            t_end   = min(t*ts, nt)
            ke_3day[:, :, t-1] = mean(ke_di[:, :, t_start:t_end], dims=3)
            pe_3day[:, :, t-1] = mean(pe_di[:, :, t_start:t_end], dims=3)
        end


        # --- Tile positions in global grid ---
        xs = (xn - 1) * tx + 1
        xe = xs + tx + (2 * buf) - 1
        ys = (yn - 1) * ty + 1
        ye = ys + ty + (2 * buf) - 1


        KE_full[xs+2:xe-2, ys+2:ye-2, :] .= ke_3day[buf:nx-buf+1, buf:ny-buf+1, :]
        PE_full[xs+2:xe-2, ys+2:ye-2, :] .= pe_3day[buf:nx-buf+1, buf:ny-buf+1, :]
        ET_full[xs+2:xe-2, ys+2:ye-2, :] .= te_mean[buf:nx-buf+1, buf:ny-buf+1, :]


        FH[xs+2:xe-2, ys+2:ye-2]  .= depth[buf:nx-buf+1, buf:ny-buf+1]
        RAC[xs+2:xe-2, ys+2:ye-2] .= rac[buf:nx-buf+1, buf:ny-buf+1]


        println("Completed tile $suffix")
    end
end


# ==========================================================
# ============ MASKING, NORMALIZATION, AVERAGING ===========
# ==========================================================
valid_mask = (RAC .> 0.0) .& (FH .> 0.0)
println("\nValid points: $(sum(valid_mask)) / $(length(valid_mask))")
total_area = sum(RAC[valid_mask])


# helper: area-weighted average of a 3D (NX×NY×nt3-2) field over valid points
function area_avg(F, vmask, RAC, total_area)
    out = zeros(size(F, 3))
    for t in axes(F, 3)
        Ft = F[:, :, t]
        out[t] = sum(Ft[vmask] .* RAC[vmask]) / total_area
    end
    return out
end


# Normalise by ρ₀·H (→ W/kg for tendency/rate terms, J/kg for energy terms)
function norm_field(F, valid_mask, rho0, FH)
    Fn = zeros(size(F))
    for t in axes(F, 3)
        Fn[valid_mask, t] .= F[valid_mask, t] ./ (rho0 .* FH[valid_mask])
    end
    return Fn
end


KE_n = norm_field(KE_full, valid_mask, rho0, FH)
PE_n = norm_field(PE_full, valid_mask, rho0, FH)
ET_n = norm_field(ET_full, valid_mask, rho0, FH)


# Time series (area-weighted domain averages)
KE_avg = area_avg(KE_n, valid_mask, RAC, total_area)
PE_avg = area_avg(PE_n, valid_mask, RAC, total_area)
ET_avg = area_avg(ET_n, valid_mask, RAC, total_area)


# Time axis (3-day windows, starting at day 3)
time_days = collect(1:nt3-2) .* 3


# ============================================================
# Ticks — ASSUMPTION: you likely have real calendar-date ticks
# (tick_vals, tick_labels) defined elsewhere in your pipeline.
# If so, delete this block and reuse those instead.
# ============================================================
n_ticks     = min(8, length(time_days))
tick_idx    = round.(Int, range(1, length(time_days), length=n_ticks))
tick_vals   = time_days[tick_idx]
tick_labels = string.(tick_vals)


# ============================================================
# Shared theme
# ============================================================
FONT = "FreeSerif Bold"


c_ke = RGBf(0.10, 0.40, 0.75)   # steel blue   — kinetic energy
c_pe = RGBf(0.80, 0.40, 0.00)   # burnt amber  — available potential energy
c_et = RGBf(0.55, 0.40, 0.05)   # dark gold    — tendency


tick_col = RGBf(0.20, 0.20, 0.20)
grid_col = RGBAf(0.75, 0.75, 0.75, 0.6)


axis_theme = (
    backgroundcolor   = :white,
    xgridcolor        = grid_col,
    ygridcolor        = grid_col,
    xgridwidth        = 0.6,
    ygridwidth        = 0.6,
    xtickcolor        = tick_col,
    ytickcolor        = tick_col,
    xticklabelcolor   = tick_col,
    yticklabelcolor   = tick_col,
    xlabelcolor       = RGBf(0.10, 0.10, 0.10),
    ylabelcolor       = RGBf(0.10, 0.10, 0.10),
    titlecolor        = RGBf(0.05, 0.05, 0.05),
    titlesize         = 15,
    titlealign        = :left,
    xlabelsize        = 13,
    ylabelsize        = 13,
    xticklabelsize    = 10,
    yticklabelsize    = 11,
    spinewidth        = 0.8,
    topspinevisible   = false,
    rightspinevisible = false,
    leftspinecolor    = tick_col,
    bottomspinecolor  = tick_col,
    titlefont         = FONT,
    xlabelfont        = FONT,
    ylabelfont        = FONT,
    xticklabelfont    = FONT,
    yticklabelfont    = FONT,
    xticks            = (tick_vals, tick_labels),
)


leg_style = (
    framecolor      = RGBAf(0.3, 0.3, 0.3, 0.4),
    backgroundcolor = RGBAf(1.0, 1.0, 1.0, 0.85),
    labelcolor      = RGBf(0.10, 0.10, 0.10),
    labelsize       = 11,
    rowgap          = 3,
    patchsize       = (22, 2),
    nbanks          = 3,
    labelfont       = FONT,
)


sc = 1e8   # scale to match your ×10⁻⁸ convention


FIGDIR = cfg["fig_base"]
mkpath(FIGDIR)

fig3 = Figure(resolution=(700, 250), fontsize=14, backgroundcolor=:white,
             figure_padding=(5, 5, 5, 5), fonts=(; regular=FONT))


ax3 = Axis(fig3[1, 1];
   title  = "Domain-averaged KE, APE, and Tendency (3-day mean)",
   xlabel = "Time  [days]",
   ylabel = "Energy rate  [×10⁻⁸ W/kg]",
   axis_theme...)


ax3b = Axis(fig3[1, 1];
   yaxisposition   = :right,
   ylabel          = "Tendency  [×10⁻⁸ W/kg]",
   ylabelcolor     = c_et,
   yticklabelcolor = c_et,
   ytickcolor      = c_et,
   ylabelfont      = FONT,
   yticklabelfont  = FONT,
   ylabelsize      = 13,
   yticklabelsize  = 11,
   backgroundcolor = :transparent,
)
hidespines!(ax3b, :t, :b, :l)
hidexdecorations!(ax3b)
linkxaxes!(ax3, ax3b)


hlines!(ax3, [0.0]; color=RGBAf(0, 0, 0, 0.3), linewidth=0.8, linestyle=:dash)
l1 = lines!(ax3,  time_days, KE_avg .* sc; color=c_ke, linewidth=1.8)
l2 = lines!(ax3,  time_days, PE_avg .* sc; color=c_pe, linewidth=1.8)
l3 = lines!(ax3b, time_days, ET_avg .* sc; color=c_et, linewidth=2.0, linestyle=:dashdot)


axislegend(ax3, [l1, l2, l3], ["⟨KE⟩ ", "⟨APE⟩ ", "⟨∂E/∂t⟩ "]; position=:rt, leg_style...)


outpath3 = joinpath(FIGDIR, "KE_APE_Tendency_TimeSeries_3day_nt_v1.png")
save(outpath3, fig3, px_per_unit=2)
println("Figure saved → $outpath3")
display(fig3)




# ============================================================
# Time-mean KE and PE
# ============================================================
KE_tmean = dropdims(mean(KE_full, dims=3), dims=3)   # NX x NY
PE_tmean = dropdims(mean(PE_full, dims=3), dims=3)   # NX x NY

valid_mask = (RAC .> 0.0) .& (FH .> 0.0)
lat_vec = collect(lat)

# ============================================================
# 0.25° latitude binning  (area-weighted)
# ============================================================
bin_width   = 0.9
bin_edges   = collect(minlat : bin_width : maxlat)
bin_centers = bin_edges[1:end-1] .+ bin_width / 2
nbins       = length(bin_centers)

KE_binned = zeros(nbins)
PE_binned = zeros(nbins)

for b in 1:nbins
   lat_lo = bin_edges[b]
   lat_hi = bin_edges[b + 1]

   KE_sum = 0.0;  PE_sum = 0.0;  W_sum = 0.0
   for j in 1:NY
       lat_vec[j] < lat_lo  && continue
       lat_vec[j] >= lat_hi && continue   # half-open interval [lo, hi)

      for i in 1:NX
           valid_mask[i, j] || continue
           w       = RAC[i, j]
           KE_sum += KE_tmean[i, j] * w
           PE_sum += PE_tmean[i, j] * w
           W_sum  += w
       end
   end


   if W_sum > 0.0
       KE_binned[b] = KE_sum / W_sum
       PE_binned[b] = PE_sum / W_sum
   end
end


# KE/APE ratio from binned profiles
ratio_binned = KE_binned ./ PE_binned

# ============================================================
# NEW: Pointwise KE/APE ratio map and area-weighted std per bin
# ============================================================
ratio_map = fill(NaN, NX, NY)
for j in 1:NY, i in 1:NX
    if valid_mask[i, j] && PE_tmean[i, j] > 0.0
        ratio_map[i, j] = KE_tmean[i, j] / PE_tmean[i, j]
    end
end

ratio_std_binned = zeros(nbins)

for b in 1:nbins
    lat_lo = bin_edges[b]
    lat_hi = bin_edges[b + 1]
    μ      = ratio_binned[b]   # area-weighted mean already computed above

    var_sum = 0.0;  W_sum = 0.0

    for j in 1:NY
        lat_vec[j] < lat_lo  && continue
        lat_vec[j] >= lat_hi && continue


        for i in 1:NX
            valid_mask[i, j]       || continue
            isnan(ratio_map[i, j]) && continue
            w        = RAC[i, j]
            var_sum += w * (ratio_map[i, j] - μ)^2
            W_sum   += w
        end
    end


    if W_sum > 0.0
        ratio_std_binned[b] = sqrt(var_sum / W_sum)
    end
end
# ============================================================

# ============================================================
# Theoretical (ω²+f²)/(ω²-f²) for semidiurnal band (9–15 hr)
# evaluated at bin centres
# ============================================================
ω_M2 = 2π / (12.4206 * 3600)   # M2 semidiurnal [rad/s]
ω_lo = 2π / (32.2 * 3600)         # 0.8fhr band edge [rad/s]
ω_hi = 2π / (10.2  * 3600)         # 2.5fhr  band edge [rad/s]

theory_M2 = fill(NaN, nbins)
theory_lo = fill(NaN, nbins)
theory_hi = fill(NaN, nbins)

for b in 1:nbins
   f_b = abs(coriolis_frequency(bin_centers[b]))
   if ω_M2 > f_b
       theory_M2[b] = (ω_M2^2 + f_b^2) / (ω_M2^2 - f_b^2)
   end
   if ω_lo > f_b
       theory_lo[b] = (ω_lo^2 + f_b^2) / (ω_lo^2 - f_b^2)
   end
   if ω_hi > f_b
       theory_hi[b] = (ω_hi^2 + f_b^2) / (ω_hi^2 - f_b^2)
   end
end

# ============================================================
# Output directory
# ============================================================
mkpath(joinpath(base2, "EnergyRatio"))

# ============================================================
# PLOT 1: Binned KE and APE zonal profiles
# ============================================================
fig1 = Figure(resolution=(900, 560), backgroundcolor=:white)
ax1  = Axis(fig1[1, 1],
   title     = "Zonal-Mean, Time-Mean, Depth-Integrated KE & APE  (1° bins)",
   xlabel    = "Latitude (°N)",
   ylabel    = "Energy  (J m⁻²)",
   titlesize = 16, xlabelsize = 13, ylabelsize = 13)

lines!(ax1, bin_centers, KE_binned, linewidth=2.5, color=:royalblue,  label="KE")
lines!(ax1, bin_centers, PE_binned, linewidth=2.5, color=:darkorange, label="APE")
axislegend(ax1, position=:rt)

FIGDIR        = cfg["fig_base"]
save(joinpath(FIGDIR, "KE_APE_zonal_binned_nt_v2.png"), fig1)
println("Saved: KE_APE_zonal_binned_nt_v2.png")
display(fig1)

# ============================================================
# PLOT 2: Binned KE/APE ratio — latitude on Y-axis
#         observed + theoretical M2 curve + 9–15 hr shaded band
# ============================================================
fig2 = Figure(resolution=(600, 800), backgroundcolor=:white)
ax2  = Axis(fig2[1, 1],
   title     = "Zonal-Mean, Time-Mean, Depth-Integrated  KE / APE  (0.9° bins)",
   xlabel    = "KE / APE  (dimensionless)",
   ylabel    = "Latitude (°N)",
   titlesize = 14, xlabelsize = 12, ylabelsize = 12,
   limits    = ((0, 6), (minlat, maxlat)))

# ---- Shaded band between 9-hr and 15-hr theoretical curves ----
# direction=:y is required because latitude is on the Y-axis and
# ratio is on the X-axis: band! fills between x_left and x_right
# at each y value (latitude).
valid_band = isfinite.(theory_lo) .& isfinite.(theory_hi)
if any(valid_band)
   band!(ax2,
       bin_centers[valid_band],   # y values  (latitude)
       theory_lo[valid_band],     # x left  boundary (15-hr curve, larger ratio)
       theory_hi[valid_band],     # x right boundary (9-hr  curve, smaller ratio)
       direction = :y,            # <-- THIS was missing: fills along Y, spans X
       color = (:darkorange, 0.25))
end

# Theoretical M2 semidiurnal curve
lines!(ax2, theory_M2, bin_centers,
   color=:darkorange, linewidth=2.0, linestyle=:dash,
   label="(ω²+f²)/(ω²-f²)  M2")

# NEW: ±1σ shading around observed KE/APE ratio
band!(ax2,
    bin_centers,
    ratio_binned .- ratio_std_binned,   # x left  boundary
    ratio_binned .+ ratio_std_binned,   # x right boundary
    direction = :y,
    color = (:royalblue, 0.20),
    label = "KE / APE  ±1σ")

# Observed KE/APE ratio
lines!(ax2, ratio_binned, bin_centers,
   color=:royalblue, linewidth=2.5,
   label="KE / APE  (observed, 1° bins)")

# KE = APE reference line
vlines!(ax2, [1.0],
   color=:firebrick, linewidth=1.2, linestyle=:dot,
   label="KE = APE")

axislegend(ax2, position=:rt, labelsize=11)
# Save figure
FIGDIR        = cfg["fig_base"]
save(joinpath(FIGDIR, "KE_APE_ratio_zonal_binned_nt_v2.png"), fig2)
println("Saved: KE_APE_ratio_zonal_binned_nt_v2.png")
display(fig2)

println("\nDone!")





