using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML, CairoMakie


include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin


# Load configuration
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base  = cfg["base_path"]
base2 = cfg["base_path2"]


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
nt  = div(Tts, dto)          # total hourly timesteps
ts  = 72                     # timesteps per 3-day period (3*24)
nt_avg = div(nt, ts)         # number of 3-day periods (same as nt3)
nt3 = div(nt, 3*24)


rho0 = 999.8


# --- Thickness ---
thk  = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF  = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)
println("Computing area-averaged KE and PE for $nt3 3-day periods...")


# ============================================================
# Global accumulators (NX×NY grids)
# ============================================================
KE_full   = zeros(NX, NY, nt3)
PE_full   = zeros(NX, NY, nt3)


# Budget terms (from existing 3-day files)
Conv_full  = zeros(NX, NY, nt3)
FDiv_full  = zeros(NX, NY, nt3)
U_KE_full  = zeros(NX, NY, nt3)
U_PE_full  = zeros(NX, NY, nt3)
SP_H_full  = zeros(NX, NY, nt3)
SP_V_full  = zeros(NX, NY, nt3)
BP_full    = zeros(NX, NY, nt3)
ET_full    = zeros(NX, NY, nt3)
FH  = zeros(NX, NY)
RAC = zeros(NX, NY)


# ============================================================
# Loop over tiles
# ============================================================
for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix  = @sprintf("%02dx%02d_%d", xn, yn, buf)
        suffix2 = @sprintf("%02dx%02d_%d", xn, yn, buf-2)
        println("\n--- Tile $suffix ---")


        # ---- Grid metrics ----
        hFacC  = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
        dx     = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
        dy     = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))
        DRFfull = hFacC .* DRF3d
        depth   = dropdims(sum(DRFfull, dims=3), dims=3)
        DRFfull[hFacC .== 0] .= 0.0
        rac = dx .* dy


        # ---- KE: read as Float32, depth-integrate immediately, then widen ----
        # Avoids holding a Float64 copy of the full nx×ny×nz×nt array in memory.
        DRFfull4_f32 = Float32.(reshape(DRFfull, nx, ny, nz, 1))   # Float32 weights

        # ---- Budget terms (already 3-day averaged) ----
        fxD = Float64.(open(joinpath(base2, "FDiv_3day", "FDiv_3day_$(suffix2).bin"), "r") do io
            nbytes = (nx-2)*(ny-2)*nt3*sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx-2, ny-2, nt3)
        end)
        C = Float64.(open(joinpath(base2, "Conv_3day", "Conv_3day_$(suffix2).bin"), "r") do io
            nbytes = (nx-2)*(ny-2)*nt3*sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx-2, ny-2, nt3)
        end)
        u_ke_3day = Float64.(open(joinpath(base2, "U_KE_3day", "u_ke_3day_$suffix.bin"), "r") do io
            nbytes = nx*ny*nt3*sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt3)
        end)
        u_pe_3day = Float64.(open(joinpath(base2, "U_PE_3day", "u_pe_3day_$suffix.bin"), "r") do io
            nbytes = nx*ny*nt3*sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt3)
        end)
        sp_h_3day = Float64.(open(joinpath(base2, "SP_H_3day", "sp_h_3day_$suffix.bin"), "r") do io
            nbytes = nx*ny*nt3*sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt3)
        end)
        sp_v_3day = Float64.(open(joinpath(base2, "SP_V_3day", "sp_v_3day_$suffix.bin"), "r") do io
            nbytes = nx*ny*nt3*sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt3)
        end)
        bp_3day = Float64.(open(joinpath(base2, "BP_3day", "bp_3day_$suffix.bin"), "r") do io
            nbytes = nx*ny*nt3*sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt3)
        end)
        te_3day = Float64.(open(joinpath(base2, "TE_tn_3day", "te_tn_3day_$suffix.bin"), "r") do io
            nbytes = nx*ny*nt3*sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt3)
        end)


        # ---- Tile position in global grid ----
        xs = (xn - 1) * tx + 1
        xe = xs + tx + (2 * buf) - 1
        ys = (yn - 1) * ty + 1
        ye = ys + ty + (2 * buf) - 1


        # ---- Update global arrays (remove buffer zones) ----
        Conv_full[xs+2:xe-2, ys+2:ye-2, :] .= C[2:end-1, 2:end-1, :]
        FDiv_full[xs+2:xe-2, ys+2:ye-2, :] .= fxD[2:end-1, 2:end-1, :]


        U_KE_full[xs+2:xe-2, ys+2:ye-2, :] .= u_ke_3day[buf:nx-buf+1, buf:ny-buf+1, :]
        U_PE_full[xs+2:xe-2, ys+2:ye-2, :] .= u_pe_3day[buf:nx-buf+1, buf:ny-buf+1, :]
        SP_H_full[xs+2:xe-2, ys+2:ye-2, :] .= sp_h_3day[buf:nx-buf+1, buf:ny-buf+1, :]
        SP_V_full[xs+2:xe-2, ys+2:ye-2, :] .= sp_v_3day[buf:nx-buf+1, buf:ny-buf+1, :]
        BP_full[xs+2:xe-2, ys+2:ye-2, :]   .= bp_3day[buf:nx-buf+1, buf:ny-buf+1, :]
        ET_full[xs+2:xe-2, ys+2:ye-2, :]   .= te_3day[buf:nx-buf+1, buf:ny-buf+1, :]



        # Static fields
        FH[xs+2:xe-2, ys+2:ye-2]  .= depth[buf:nx-buf+1, buf:ny-buf+1]
        RAC[xs+2:xe-2, ys+2:ye-2] .= rac[buf:nx-buf+1, buf:ny-buf+1]


        println("  Done.")
    end
end


# ============================================================
# Area-weighted averages
# ============================================================
valid_mask = (RAC .> 0.0) .& (FH .> 0.0)
println("\nValid points: $(sum(valid_mask)) / $(length(valid_mask))")
total_area = sum(RAC[valid_mask])


# helper: area-weighted average of a 3D (NX×NY×nt3) field over valid points
function area_avg(F, vmask, RAC, total_area)
    out = zeros(size(F, 3))
    for t in axes(F, 3)
        Ft = F[:, :, t]
        out[t] = sum(Ft[vmask] .* RAC[vmask]) / total_area
    end
    return out
end


# Normalise budget terms by ρ₀·H (→ W/kg)
norm_field(F) = begin
    Fn = zeros(size(F))
    for t in axes(F, 3)
        Fn[valid_mask, t] .= F[valid_mask, t] ./ (rho0 .* FH[valid_mask])
    end
    Fn
end


Conv_n  = norm_field(Conv_full)
FDiv_n  = norm_field(FDiv_full)
U_KE_n  = norm_field(U_KE_full)
U_PE_n  = norm_field(U_PE_full)
SP_H_n  = norm_field(SP_H_full)
SP_V_n  = norm_field(SP_V_full)
BP_n    = norm_field(BP_full)
ET_n    = norm_field(ET_full)


# Derived budget terms
A_n          = U_KE_n .+ U_PE_n
PS_n         = SP_H_n .+ SP_V_n
TotalFlux_n  = FDiv_n .+ U_KE_n .+ U_PE_n
Residual_n   = -(Conv_n .- TotalFlux_n .+ PS_n .+ BP_n .- ET_n)
Residual_n1  = -(Conv_n .- TotalFlux_n .+ PS_n .+ BP_n)


# Time series (area-weighted)
Conv_avg      = area_avg(Conv_n,     valid_mask, RAC, total_area)
FDiv_avg      = area_avg(FDiv_n,     valid_mask, RAC, total_area)
PS_avg        = area_avg(PS_n,       valid_mask, RAC, total_area)
BP_avg        = area_avg(BP_n,       valid_mask, RAC, total_area)
A_avg         = area_avg(A_n,        valid_mask, RAC, total_area)
ET_avg        = area_avg(ET_n,       valid_mask, RAC, total_area)
Residual_avg  = area_avg(Residual_n, valid_mask, RAC, total_area)
Residual_avg1 = area_avg(Residual_n1,valid_mask, RAC, total_area)


# Time axis
time_days = collect(1:nt3) .* 3


# ============================================================
# Shared theme helpers
# ============================================================
c_conv = RGBf(0.80, 0.10, 0.10)   # deep red
c_fdiv = RGBf(0.10, 0.40, 0.75)   # steel blue
c_res  = RGBf(0.15, 0.15, 0.15)   # near-black (residual)
c_ps   = RGBf(0.10, 0.60, 0.30)   # forest green
c_bp   = RGBf(0.80, 0.40, 0.00)   # burnt amber
c_a    = RGBf(0.50, 0.15, 0.75)   # violet
c_et   = RGBf(0.55, 0.40, 0.05)   # dark gold


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
    xlabelsize        = 13,
    ylabelsize        = 13,
    xticklabelsize    = 11,
    yticklabelsize    = 11,
    spinewidth        = 0.8,
    topspinevisible   = false,
    rightspinevisible = false,
    leftspinecolor    = tick_col,
    bottomspinecolor  = tick_col,
)


sc = 1e8   # scale factor for display


leg_style = (
    framecolor      = RGBAf(0.3, 0.3, 0.3, 0.4),
    backgroundcolor = RGBAf(1.0, 1.0, 1.0, 0.85),
    labelcolor      = RGBf(0.10, 0.10, 0.10),
    labelsize       = 11,
    rowgap          = 3,
    patchsize       = (22, 2),
)


FIGDIR = cfg["fig_base"]
mkpath(FIGDIR)


# ============================================================
# Figure 1: Budget terms WITHOUT tendency (single panel, 1100×400)
# ============================================================
println("\nCreating Figure 1 (no tendency)...")
fig1 = Figure(resolution=(1100, 400), fontsize=14, backgroundcolor=:white)


ax1 = Axis(fig1[1, 1];
    title  = "All Budget Terms  (area-averaged, 3-day periods)",
    xlabel = "Time  [days]",
    ylabel = "Energy rate  [×10⁻⁸ W kg⁻¹]",
    axis_theme...)


hlines!(ax1, [0.0]; color=RGBAf(0,0,0,0.3), linewidth=0.8, linestyle=:dash)
lines!(ax1, time_days, Conv_avg      .* sc; label="⟨C⟩  Conversion",        color=c_conv, linewidth=1.8)
lines!(ax1, time_days, FDiv_avg      .* sc; label="⟨∇·F⟩  Flux divergence",  color=c_fdiv, linewidth=1.8)
lines!(ax1, time_days, PS_avg        .* sc; label="⟨Pₛ⟩  Shear production",  color=c_ps,   linewidth=1.8)
lines!(ax1, time_days, BP_avg        .* sc; label="⟨Pᵦ⟩  Buoyancy prod.",    color=c_bp,   linewidth=1.8)
lines!(ax1, time_days, A_avg         .* sc; label="⟨A⟩  Advection",          color=c_a,    linewidth=1.8)
#lines!(ax1, time_days, ET_avg        .* sc; label="⟨∂E/∂t⟩  Tendency",       color=c_et,   linewidth=2.0, linestyle=:dashdot)
lines!(ax1, time_days, Residual_avg1 .* sc; label="⟨R⟩  Residual (D)",       color=c_res,  linewidth=1.8)
axislegend(ax1; position=:rt, leg_style...)


outpath1 = joinpath(FIGDIR, "KE_PE_Budget_TimeSeries_3day_v5.png")
save(outpath1, fig1, px_per_unit=2)
println("Figure 1 saved → $outpath1")
display(fig1)


# ============================================================
# Figure 2: Budget terms WITH tendency + KE/APE panel (1100×800)
# ============================================================
println("\nCreating Figure 2 (with tendency )...")
fig2 = Figure(resolution=(1200, 400), fontsize=14, backgroundcolor=:white)


# --- Subplot 1: all budget terms including tendency ---
ax2a = Axis(fig2[1, 1];
    title  = "All Budget Terms  (area-averaged, 3-day periods)",
    xlabel = "Time  [days]",
    ylabel = "Energy rate  [×10⁻⁸ W kg⁻¹]",
    axis_theme...)


hlines!(ax2a, [0.0]; color=RGBAf(0,0,0,0.3), linewidth=0.8, linestyle=:dash)
lines!(ax2a, time_days, Conv_avg     .* sc; label="⟨C⟩  Conversion",        color=c_conv, linewidth=1.8)
lines!(ax2a, time_days, FDiv_avg     .* sc; label="⟨∇·F⟩  Flux divergence",  color=c_fdiv, linewidth=1.8)
lines!(ax2a, time_days, PS_avg       .* sc; label="⟨Pₛ⟩  Shear production",  color=c_ps,   linewidth=1.8)
lines!(ax2a, time_days, BP_avg       .* sc; label="⟨Pᵦ⟩  Buoyancy prod.",    color=c_bp,   linewidth=1.8)
lines!(ax2a, time_days, A_avg        .* sc; label="⟨A⟩  Advection",          color=c_a,    linewidth=1.8)
lines!(ax2a, time_days, ET_avg       .* sc; label="⟨∂E/∂t⟩  Tendency",       color=c_et,   linewidth=2.0, linestyle=:dashdot)
lines!(ax2a, time_days, Residual_avg .* sc; label="⟨R⟩  Residual (D)",       color=c_res,  linewidth=1.8)
axislegend(ax2a; position=:rt, leg_style...)



rowgap!(fig2.layout, 1, 24)


outpath2 = joinpath(FIGDIR, "Budget_TimeSeries_3day_wn_v4.png")
save(outpath2, fig2, px_per_unit=2)
println("Figure 2 saved → $outpath2")
display(fig2)




