using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML, CairoMakie, Dates
include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin


# Load configuration
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base  = cfg["base_path"]
base2 = cfg["base_path_nt"]


# --- Domain & grid ---
NX, NY = 288, 468
minlat, maxlat = 24.0, 31.91
minlon, maxlon = 193.0, 199.0
lat = range(minlat, maxlat, length=NY)
lon = range(minlon, maxlon, length=NX)


# --- Tile & time parameters ---
buf  = 3
tx, ty = 47, 66
nx = tx + 2*buf
ny = ty + 2*buf
nz = 88
dt  = 25
dto = 144
Tts = 366192
nt  = div(Tts, dto)
ts  = 72
nt_avg = div(nt, ts)
nt3 = div(nt, 3*24)


rho0 = 1027.5
DEPTH_THRESHOLD = 3000.0


# --- Date axis ---
t_origin   = DateTime(2012, 3, 1, 0, 0, 0)
dates_3day = [t_origin + Day(3*(i-1)) + Hour(36) for i in 1:nt3]
t_numeric  = [Dates.value(d - t_origin) / (1000*3600*24) for d in dates_3day]


tick_every  = 5
tick_inds   = 1:tick_every:nt3
tick_vals   = t_numeric[tick_inds]
tick_labels = [Dates.format(dates_3day[i], "u dd\nyyyy") for i in tick_inds]


# --- Thickness ---
thk   = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF   = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)
println("Computing area-averaged budget terms for $nt3 3-day periods...")


# --- Global arrays ---
Conv_full = zeros(NX, NY, nt3)
FDiv_full = zeros(NX, NY, nt3)
U_KE_full = zeros(NX, NY, nt3)
U_PE_full = zeros(NX, NY, nt3)
SP_H_full = zeros(NX, NY, nt3)
SP_V_full = zeros(NX, NY, nt3)
BP_full   = zeros(NX, NY, nt3)
ET_full   = zeros(NX, NY, nt3)
FH        = zeros(NX, NY)
RAC       = zeros(NX, NY)


# ============================================================
# Loop over tiles
# ============================================================
for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix  = @sprintf("%02dx%02d_%d", xn, yn, buf)
        suffix2 = @sprintf("%02dx%02d_%d", xn, yn, buf-2)
        println("\n--- Tile $suffix ---")


        hFacC   = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
        dx      = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
        dy      = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))
        DRFfull = hFacC .* DRF3d
        depth   = dropdims(sum(DRFfull, dims=3), dims=3)
        DRFfull[hFacC .== 0] .= 0.0
        rac = dx .* dy


        println("  Reading KE...")
        DRFfull4_f32 = Float32.(reshape(DRFfull, nx, ny, nz, 1))
        ke_di = open(joinpath(base2, "KE", "ke_t_nt_$suffix.bin"), "r") do io
            ke_raw = reshape(read!(io, Array{Float32}(undef, nx, ny, nz, nt)), nx, ny, nz, nt)
            Float64.(dropdims(sum(ke_raw .* DRFfull4_f32, dims=3), dims=3))
        end


        println("  Reading APE...")
        DRF3d4_f32 = Float32.(reshape(DRF3d, nx, ny, nz, 1))


        fxD = Float64.(open(joinpath(base2, "FDiv_3day", "FDiv_3day_nt_$(suffix2).bin"), "r") do io
            nbytes = (nx-2)*(ny-2)*nt3*sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx-2, ny-2, nt3)
        end)
        C = Float64.(open(joinpath(base2, "Conv_3day", "Conv_3day_nt_$(suffix2).bin"), "r") do io
            nbytes = (nx-2)*(ny-2)*nt3*sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx-2, ny-2, nt3)
        end)
        u_ke_3day = Float64.(open(joinpath(base2, "U_KE_3day", "u_ke_3day_nt_$suffix.bin"), "r") do io
            nbytes = nx*ny*nt3*sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt3)
        end)
        u_pe_3day = Float64.(open(joinpath(base2, "U_PE_3day", "u_pe_3day_nt_$suffix.bin"), "r") do io
            nbytes = nx*ny*nt3*sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt3)
        end)
        sp_h_3day = Float64.(open(joinpath(base2, "SP_H_3day", "sp_h_3day_nt_$suffix.bin"), "r") do io
            nbytes = nx*ny*nt3*sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt3)
        end)
        sp_v_3day = Float64.(open(joinpath(base2, "SP_V_3day", "sp_v_3day_nt_$suffix.bin"), "r") do io
            nbytes = nx*ny*nt3*sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt3)
        end)
        bp_3day = Float64.(open(joinpath(base2, "BP_3day", "bp_3day_nt_$suffix.bin"), "r") do io
            nbytes = nx*ny*nt3*sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt3)
        end)
        te_3day = Float64.(open(joinpath(base2, "TE_t_3day", "te_t_3day_nt_$suffix.bin"), "r") do io
            nbytes = nx*ny*nt3*sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt3)
        end)


        xs = (xn - 1) * tx + 1
        xe = xs + tx + (2 * buf) - 1
        ys = (yn - 1) * ty + 1
        ye = ys + ty + (2 * buf) - 1


        Conv_full[xs+2:xe-2, ys+2:ye-2, :] .= C[2:end-1, 2:end-1, :]
        FDiv_full[xs+2:xe-2, ys+2:ye-2, :] .= fxD[2:end-1, 2:end-1, :]
        U_KE_full[xs+2:xe-2, ys+2:ye-2, :] .= u_ke_3day[buf:nx-buf+1, buf:ny-buf+1, :]
        U_PE_full[xs+2:xe-2, ys+2:ye-2, :] .= u_pe_3day[buf:nx-buf+1, buf:ny-buf+1, :]
        SP_H_full[xs+2:xe-2, ys+2:ye-2, :] .= sp_h_3day[buf:nx-buf+1, buf:ny-buf+1, :]
        SP_V_full[xs+2:xe-2, ys+2:ye-2, :] .= sp_v_3day[buf:nx-buf+1, buf:ny-buf+1, :]
        BP_full[xs+2:xe-2, ys+2:ye-2, :]   .= bp_3day[buf:nx-buf+1, buf:ny-buf+1, :]
        ET_full[xs+2:xe-2, ys+2:ye-2, :]   .= te_3day[buf:nx-buf+1, buf:ny-buf+1, :]
        FH[xs+2:xe-2, ys+2:ye-2]  .= depth[buf:nx-buf+1, buf:ny-buf+1]
        RAC[xs+2:xe-2, ys+2:ye-2] .= rac[buf:nx-buf+1, buf:ny-buf+1]


        println("  Done.")
    end
end


# ============================================================
# Depth masks
# ============================================================
valid_mask   = (RAC .> 0.0) .& (FH .> 0.0)
shallow_mask = valid_mask .& (FH .<  DEPTH_THRESHOLD)
deep_mask    = valid_mask .& (FH .>= DEPTH_THRESHOLD)


println("\nValid points         : $(sum(valid_mask))")
println("Shallow (<$(Int(DEPTH_THRESHOLD)) m) : $(sum(shallow_mask))")
println("Deep    (>=$(Int(DEPTH_THRESHOLD)) m) : $(sum(deep_mask))")


total_area_shallow = sum(RAC[shallow_mask])
total_area_deep    = sum(RAC[deep_mask])


# ============================================================
# Helpers
# ============================================================
function area_avg(F, mask, RAC, total_area)
    out = zeros(size(F, 3))
    for t in axes(F, 3)
        Ft = F[:, :, t]
        out[t] = sum(Ft[mask] .* RAC[mask]) / total_area
    end
    return out
end


function norm_field(F, mask, FH)
    Fn = zeros(size(F))
    for t in axes(F, 3)
        Fn[mask, t] .= F[mask, t] ./ (rho0 .* FH[mask])
    end
    return Fn
end


function compute_avgs(mask, area)
    Conv_n = norm_field(Conv_full, mask, FH)
    FDiv_n = norm_field(FDiv_full, mask, FH)
    U_KE_n = norm_field(U_KE_full, mask, FH)
    U_PE_n = norm_field(U_PE_full, mask, FH)
    SP_H_n = norm_field(SP_H_full, mask, FH)
    SP_V_n = norm_field(SP_V_full, mask, FH)
    BP_n   = norm_field(BP_full,   mask, FH)
    ET_n   = norm_field(ET_full,   mask, FH)


    A_n         = U_KE_n .+ U_PE_n
    TotalFlux_n = FDiv_n .+ U_KE_n .+ U_PE_n
    PS_n        = SP_H_n .+ SP_V_n
    Residual_n  = -(Conv_n .- TotalFlux_n .+ PS_n .+ BP_n .- ET_n)


    return (
        Conv     = area_avg(Conv_n,     mask, RAC, area),
        FDiv     = area_avg(FDiv_n,     mask, RAC, area),
        SP_H     = area_avg(SP_H_n,     mask, RAC, area),
        SP_V     = area_avg(SP_V_n,     mask, RAC, area),
        BP       = area_avg(BP_n,       mask, RAC, area),
        A        = area_avg(A_n,        mask, RAC, area),
        ET       = area_avg(ET_n,       mask, RAC, area),
        Residual = area_avg(Residual_n, mask, RAC, area),
        PS       = area_avg(PS_n,       mask, RAC, area),
    )
end


println("\nComputing shallow averages...")
sh = compute_avgs(shallow_mask, total_area_shallow)


println("Computing deep averages...")
dp = compute_avgs(deep_mask, total_area_deep)


# ============================================================
# Compute ratios
# Wind Input (WI) = ET term
# Dissipation (D) = Residual
# q1 = Dissipation   / (C + WI)
# q2 = FDiv          / (C + WI)
# q3 = (Pb + Ps)     / (C + WI)
# ============================================================
function safe_ratio(num, denom; tol=1e-30)
    # Returns NaN where denominator is near zero to avoid blow-up
    return [abs(denom[t]) > tol ? num[t] / denom[t] : NaN for t in eachindex(denom)]
end


function compute_ratios(d)
    denom = d.Conv .+ d.ET                  # C + WI
    PS    = d.SP_H .+ d.SP_V               # Ps = SP_H + SP_V
    MF    = d.BP .+ PS                     # Mean flow: Pb + Ps


    q1 = safe_ratio(d.Residual, denom)     # Dissipation / (C + WI)
    q2 = safe_ratio(d.FDiv,     denom)     # Flux divergence / (C + WI)
    q3 = safe_ratio(MF,         denom)     # (Pb + Ps) / (C + WI)


    return (q1=q1, q2=q2, q3=q3)
end


sh_r = compute_ratios(sh)
dp_r = compute_ratios(dp)


# ============================================================
# Plot theme  (bold font throughout)
# ============================================================
FONT     = "FreeSerif Bold"
tick_col = RGBf(0.20, 0.20, 0.20)
grid_col = RGBAf(0.75, 0.75, 0.75, 0.6)


# Line colors for the 3 ratios
c_q1 = RGBf(0.80, 0.10, 0.10)   # red   — dissipation fraction
c_q2 = RGBf(0.10, 0.40, 0.75)   # blue  — flux divergence fraction
c_q3 = RGBf(0.10, 0.55, 0.25)   # green — mean flow fraction


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
    nbanks          = 1,
    labelfont       = FONT,
)


FIGDIR = cfg["fig_base"]
mkpath(FIGDIR)


# ============================================================
# Figure — two rows (shallow top, deep bottom)
# ============================================================
println("\nCreating ratio time-series figure...")


fig = Figure(resolution=(700, 500), fontsize=14, backgroundcolor=:white, figure_padding =(5,5,5,5),
             fonts=(; regular=FONT, bold=FONT))


function add_ratio_lines!(ax, r, t_numeric)
    hlines!(ax, [0.0]; color=RGBAf(0,0,0,0.3), linewidth=0.8, linestyle=:dash)
    hlines!(ax, [1.0]; color=RGBAf(0,0,0,0.2), linewidth=0.8, linestyle=:dot)
    lines!(ax, t_numeric, r.q1; label="q = ⟨D⟩ / (⟨C⟩ + ⟨WI⟩)  [Dissipation fraction]",   color=c_q1, linewidth=1.8)
    lines!(ax, t_numeric, r.q2; label="⟨∇·F⟩ / (⟨C⟩ + ⟨WI⟩)  [Flux divergence fraction]", color=c_q2, linewidth=1.8)
    lines!(ax, t_numeric, r.q3; label="(⟨Pᵦ⟩ + ⟨Pₛ⟩) / (⟨C⟩ + ⟨WI⟩)  [Mean flow fraction]", color=c_q3, linewidth=1.8)
end


# Row 1 — shallow
ax_sh = Axis(fig[1, 1];
    title  = "Shallow region  (H < $(Int(DEPTH_THRESHOLD)) m)",
    ylabel = "Fraction  [ ]",
    xticklabelsvisible = false,
    axis_theme...)


# Row 2 — deep
ax_dp = Axis(fig[2, 1];
    title  = "Deep region  (H ≥ $(Int(DEPTH_THRESHOLD)) m)",
    ylabel = "Fraction  [ ]",
    axis_theme...)


add_ratio_lines!(ax_sh, sh_r, t_numeric)
add_ratio_lines!(ax_dp, dp_r, t_numeric)


axislegend(ax_sh; position=:rt, leg_style...)
axislegend(ax_dp; position=:rt, leg_style...)


linkxaxes!(ax_sh, ax_dp)


rowgap!(fig.layout, 1, 8)


outpath = joinpath(FIGDIR, "Budget_Ratios_TimeSeries_DepthSplit.png")
save(outpath, fig, px_per_unit=2)
println("Figure saved → $outpath")
display(fig)




