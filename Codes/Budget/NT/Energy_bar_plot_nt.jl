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

# --- Thickness ---
thk   = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF   = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)
println("Computing area-averaged budget terms for $nt3 3-day periods...")

ring_steps = nt_chunk
t_safe_start = ring_steps + 1
t_safe_end   = nt - ring_steps


safe_chunks = [c for c in 1:n_chunks
              if (c-1)*nt_chunk + 1 >= t_safe_start &&
                 c*nt_chunk          <= t_safe_end]
@info "Safe 3-day chunks: $(length(safe_chunks)) of $n_chunks  (chunks $(safe_chunks[1])–$(safe_chunks[end]))"

# --- Global arrays ---
Conv_full = zeros(NX, NY, nt3-2)
FDiv_full = zeros(NX, NY, nt3-2)
U_KE_full = zeros(NX, NY, nt3-2)
U_PE_full = zeros(NX, NY, nt3-2)
SP_H_full = zeros(NX, NY, nt3-2)
SP_V_full = zeros(NX, NY, nt3-2)
BP_full   = zeros(NX, NY, nt3-2)
ET_full   = zeros(NX, NY, nt3-2)
WPI_full   = zeros(NX, NY, nt3-2)

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

       DRF3d4_f32 = Float32.(reshape(DRF3d, nx, ny, nz, 1))

       fxD = Float64.(open(joinpath(base2, "FDiv_3day", "FDiv_3day_nt_$(suffix2).bin"), "r") do io
           nbytes = (nx-2)*(ny-2)*nt3*sizeof(Float32)
           reshape(reinterpret(Float32, read(io, nbytes)), nx-2, ny-2, nt3-2)
       end)
       C = Float64.(open(joinpath(base2, "Conv_3day", "Conv_3day_nt_$(suffix2).bin"), "r") do io
           nbytes = (nx-2)*(ny-2)*nt3*sizeof(Float32)
           reshape(reinterpret(Float32, read(io, nbytes)), nx-2, ny-2, nt3-2)
       end)
       u_ke_3day = Float64.(open(joinpath(base2, "U_KE_3day", "u_ke_3day_nt_$suffix.bin"), "r") do io
           nbytes = nx*ny*nt3*sizeof(Float32)
           reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt3-2)
       end)
       u_pe_3day = Float64.(open(joinpath(base2, "U_PE_3day", "u_pe_3day_nt_$suffix.bin"), "r") do io
           nbytes = nx*ny*nt3*sizeof(Float32)
           reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt3-2)
       end)
       sp_h_3day = Float64.(open(joinpath(base2, "SP_H_3day", "sp_h_3day_nt_$suffix.bin"), "r") do io
           nbytes = nx*ny*nt3*sizeof(Float32)
           reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt3-2)
       end)
       sp_v_3day = Float64.(open(joinpath(base2, "SP_V_3day", "sp_v_3day_nt_$suffix.bin"), "r") do io
           nbytes = nx*ny*nt3*sizeof(Float32)
           reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt3-2)
       end)
       bp_3day = Float64.(open(joinpath(base2, "BP_3day", "bp_3day_nt_$suffix.bin"), "r") do io
           nbytes = nx*ny*nt3*sizeof(Float32)
           reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt3-2)
       end)
       te_3day = Float64.(open(joinpath(base2, "TE_t_3day", "te_t_3day_nt_$suffix.bin"), "r") do io
           nbytes = nx*ny*nt3*sizeof(Float32)
           reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt3-2)
       end)
       wpi_tile = Float64.(open(joinpath(base2, "WindInput", "wpi_nt_$suffix.bin"), "r") do io
           nbytes = nx * ny * nt * sizeof(Float32)
           reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt)
       end)

       xs = (xn - 1) * tx + 1
       xe = xs + tx + (2 * buf) - 1
       ys = (yn - 1) * ty + 1
       ye = ys + ty + (2 * buf) - 1
       wpi_3day = zeros(nx,ny,nt3-2)
       for (i, c) in enumerate(safe_chunks)
            t1 = (c-1)*nt_chunk + 1
           t2 = c*nt_chunk
           wpi_3day[:, :, i] = Float32.(dropdims(mean(wpi_tile[:, :, t1:t2], dims=3), dims=3))
       end
       Conv_full[xs+2:xe-2, ys+2:ye-2, :] .= C[2:end-1, 2:end-1, :]
       FDiv_full[xs+2:xe-2, ys+2:ye-2, :] .= fxD[2:end-1, 2:end-1, :]
       U_KE_full[xs+2:xe-2, ys+2:ye-2, :] .= u_ke_3day[buf:nx-buf+1, buf:ny-buf+1, :]
       U_PE_full[xs+2:xe-2, ys+2:ye-2, :] .= u_pe_3day[buf:nx-buf+1, buf:ny-buf+1, :]
       SP_H_full[xs+2:xe-2, ys+2:ye-2, :] .= sp_h_3day[buf:nx-buf+1, buf:ny-buf+1, :]
       SP_V_full[xs+2:xe-2, ys+2:ye-2, :] .= sp_v_3day[buf:nx-buf+1, buf:ny-buf+1, :]
       BP_full[xs+2:xe-2, ys+2:ye-2, :]   .= bp_3day[buf:nx-buf+1, buf:ny-buf+1, :]
       ET_full[xs+2:xe-2, ys+2:ye-2, :]   .= te_3day[buf:nx-buf+1, buf:ny-buf+1, :]
       WPI_full[xs+2:xe-2, ys+2:ye-2,:] .= wpi_3day[buf:nx-buf+1,  buf:ny-buf+1,:]
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

total_area_full    = sum(RAC[valid_mask])     # <-- CHANGED
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
   Conv_n = norm_field(Conv_full, mask, FH) .+ norm_field(WPI_full,   mask, FH)
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
   Residual_n  = -(Conv_n.- TotalFlux_n .+ PS_n .+ BP_n .- ET_n)

   return (
       Conv     = area_avg(Conv_n,     mask, RAC, area),
       FDiv     = area_avg(FDiv_n,     mask, RAC, area),
       SP_H     = area_avg(SP_H_n,     mask, RAC, area),
       SP_V     = area_avg(SP_V_n,     mask, RAC, area),
       BP       = area_avg(BP_n,       mask, RAC, area),
       A        = area_avg(A_n,        mask, RAC, area),
       ET       = area_avg(ET_n,       mask, RAC, area),
       Residual = area_avg(Residual_n, mask, RAC, area),
   )
end
println("\nComputing full area averages...")          # <-- CHANGED
fl = compute_avgs(valid_mask, total_area_full)        # <-- CHANGED
println("Computing deep averages...")
dp = compute_avgs(deep_mask, total_area_deep)

# ============================================================
# Plot theme
# ============================================================
FONT     = "FreeSerif Bold"
tick_col = RGBf(0.20, 0.20, 0.20)
grid_col = RGBAf(0.75, 0.75, 0.75, 0.6)
sc       = 1e8

FIGDIR = cfg["fig_base"]
mkpath(FIGDIR)
# ============================================================
# Time-average each term
# ============================================================
function time_mean_terms(d, sc)
   return (
       Conv     = mean(d.Conv)     * sc,
       FDiv     = mean(d.FDiv)     * sc,
       SP_H     = mean(d.SP_H)     * sc,
       SP_V     = mean(d.SP_V)     * sc,
       BP       = mean(d.BP)       * sc,
       A        = mean(d.A)        * sc,
       ET       = mean(d.ET)       * sc,
       Residual = mean(d.Residual) * sc,
   )
end
fl_mean = time_mean_terms(fl, sc)     # <-- CHANGED
dp_mean = time_mean_terms(dp, sc)
# ============================================================
# Labels and values — bottom to top order
# ============================================================
labels = [
   "⟨R⟩  Residual ",
   "⟨A⟩  Advection",
   "⟨Pᵦ⟩  Buoyancy prod.",
   "⟨Pₛᵛ⟩  Vert. shear prod.",
   "⟨Pₛᴴ⟩  Horiz. shear prod.",
   "⟨∂E/∂t⟩  Tendency",
   "⟨∇·F⟩  Flux divergence",
   "⟨C⟩  + ⟨WI⟩",
]
function extract_vals(d)
   return [
       d.Residual,
       d.A,
       d.BP,
       d.SP_V,
       d.SP_H,
       d.ET,
       d.FDiv,
       d.Conv,
   ]
end
fl_vals = extract_vals(fl_mean)     # <-- CHANGED
dp_vals = extract_vals(dp_mean)
n     = length(labels)
y_pos = collect(1:n)
c_pos = RGBf(0.75, 0.15, 0.15)
c_neg = RGBf(0.20, 0.40, 0.75)

fl_colors = [v >= 0 ? c_pos : c_neg for v in fl_vals]     # <-- CHANGED
dp_colors = [v >= 0 ? c_pos : c_neg for v in dp_vals]
# ============================================================
# Shared axis theme
# ============================================================
bar_theme = (
   backgroundcolor   = :white,
   xgridcolor        = grid_col,
   ygridcolor        = RGBAf(0, 0, 0, 0),
   xgridwidth        = 0.6,
   xtickcolor        = tick_col,
   ytickcolor        = RGBAf(0, 0, 0, 0),
   xticklabelcolor   = tick_col,
   yticklabelcolor   = RGBf(0.10, 0.10, 0.10),
   xlabelcolor       = RGBf(0.10, 0.10, 0.10),
   titlecolor        = RGBf(0.05, 0.05, 0.05),
   titlesize         = 14,
   titlealign        = :left,
   xlabelsize        = 12,
   yticklabelsize    = 11,
   xticklabelsize    = 10,
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
   yticks            = (y_pos, labels),
   ytickalign        = 1,
)
# ============================================================
# Figure — two panels side by side
# ============================================================
println("\nCreating bar plot of time-averaged budget terms...")

fig = Figure(resolution=(900, 480), backgroundcolor=:white,
            fonts=(; regular=FONT))
# Full area panel (left)                                      # <-- CHANGED
ax_fl = Axis(fig[1, 1];
   title  = "(a)  Full area",                                 # <-- CHANGED
   xlabel = "Energy rate  [×10⁻⁸ W/kg]",
   bar_theme...)
# Deep panel (right)
ax_dp = Axis(fig[1, 2];
   title  = "(b)  Deep region  (H ≥ $(Int(DEPTH_THRESHOLD)) m)",
   xlabel = "Energy rate  [×10⁻⁸ W/kg]",
   yticklabelsvisible = false,
   bar_theme...)

for (ax, vals, cols) in [(ax_fl, fl_vals, fl_colors), (ax_dp, dp_vals, dp_colors)]    # <-- CHANGED
   vlines!(ax, [0.0]; color=RGBAf(0, 0, 0, 0.4), linewidth=0.9, linestyle=:dash)

   barplot!(ax, y_pos, vals;
       direction   = :x,
       color       = cols,
       bar_labels  = [@sprintf("%.3f", v) for v in vals],
       label_size  = 10,
       label_font  = FONT,
       label_rotation = π/2,
       label_color = RGBf(0.15, 0.15, 0.15),
       gap         = 0.25,
   )
end

linkyaxes!(ax_fl, ax_dp)     # <-- CHANGED
colgap!(fig.layout, 1, 8)

outpath = joinpath(FIGDIR, "Budget_BarPlot_TimeAvg_FullAndDeep.png")     # <-- CHANGED
save(outpath, fig, px_per_unit=2)
println("Bar plot saved → $outpath")
display(fig)




