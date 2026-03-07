using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML, CairoMakie


if !isdefined(Main, :FluxUtils)
    include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
end
include(joinpath(@__DIR__, "..", "..", "..", "functions", "coriolis_frequency.jl"))

using .FluxUtils: read_bin


config_file = get(ENV, "JULIA_CONFIG",
    joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
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
buf = 3
tx, ty = 47, 66
nx = tx + 2*buf
ny = ty + 2*buf
nz = 88
dt  = 25
dto = 144
Tts = 366192
nt  = div(Tts, dto)
ts  = 72                  # timesteps per 3-day period
nt3 = div(nt, 3*24)
dt_output = dt * dto      # seconds per output interval = 3600 s


rho0 = 999.8


# --- Thickness ---
thk  = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF  = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)


# ============================================================
# Global accumulators
# ============================================================
KE_full  = zeros(NX, NY, nt3)
PE_full  = zeros(NX, NY, nt3)
TE_t_full = zeros(NX, NY, nt3)   # 3-day tendency of TE
FH       = zeros(NX, NY)
RAC      = zeros(NX, NY)


# ============================================================
# Tile loop
# ============================================================
for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
        println("\n--- Tile $suffix ---")


        # ---- Grid metrics (exactly as in reference code) ----
        hFacC   = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
        dx      = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
        dy      = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))
        DRFfull = hFacC .* DRF3d
        depth   = dropdims(sum(DRFfull, dims=3), dims=3)
        DRFfull[hFacC .== 0] .= 0.0
        rac     = dx .* dy


        # ---- Read KE (exactly as in reference code) ----
        println("  Reading KE...")
        ke_raw = Float64.(open(joinpath(base2, "KE", "ke_t_sm_$suffix.bin"), "r") do io
            nbytes = nx * ny * nz * nt * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nz, nt)
        end)


        # ---- Read APE (exactly as in reference code) ----
        println("  Reading APE...")
        ape_raw = Float64.(open(joinpath(base2, "APE", "APE_t_sm_$suffix.bin"), "r") do io
            nbytes = nx * ny * nz * nt * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nz, nt)
        end)


        # ---- Depth-integrate KE (weighted by DRFfull, as in reference) ----
        DRFfull4 = reshape(DRFfull, nx, ny, nz, 1)
        ke_di    = dropdims(sum(ke_raw .* DRFfull4, dims=3), dims=3)   # nx x ny x nt


        # ---- Depth-integrate APE (NaN-safe, using DRF3d, as in reference) ----
        DRF3d4    = reshape(DRF3d, nx, ny, nz, 1)
        ape_clean = replace(ape_raw, NaN => 0.0)
        pe_di     = dropdims(sum(ape_clean .* DRFfull4, dims=3), dims=3) # nx x ny x nt


        # ---- Total energy depth-integrated ----
        te_di = ke_di .+ pe_di   # nx x ny x nt


        # ---- 3-day mean of KE and PE (same as reference code) ----
        ke_3day = zeros(nx, ny, nt3)
        pe_3day = zeros(nx, ny, nt3)
        for t in 1:nt3
            t_start = (t-1)*ts + 1
            t_end   = min(t*ts, nt)
            ke_3day[:, :, t] = mean(ke_di[:, :, t_start:t_end], dims=3)
            pe_3day[:, :, t] = mean(pe_di[:, :, t_start:t_end], dims=3)
        end


        # ---- 3-day tendency of TE: (TE(t_end) - TE(t_start)) / DeltaT ----
        te_t_3day = zeros(nx, ny, nt3)
        hrs_per_chunk = 3 * 24   # 72 output steps
        for p in 1:nt3
            t_start = (p-1) * hrs_per_chunk + 1
            t_end   = min(p * hrs_per_chunk, nt)
            DeltaT  = (t_end - t_start) * dt_output   # [seconds]
            te_t_3day[:, :, p] = (te_di[:, :, t_end] .- te_di[:, :, t_start]) ./ DeltaT
        end


        # ---- Tile position in global grid (exactly as in reference) ----
        xs = (xn - 1) * tx + 1
        xe = xs + tx + (2 * buf) - 1
        ys = (yn - 1) * ty + 1
        ye = ys + ty + (2 * buf) - 1


        # ---- Place into global arrays (interior only, as in reference) ----
        KE_full[xs+2:xe-2,  ys+2:ye-2, :] .= ke_3day[buf:nx-buf+1,  buf:ny-buf+1, :]
        PE_full[xs+2:xe-2,  ys+2:ye-2, :] .= pe_3day[buf:nx-buf+1,  buf:ny-buf+1, :]
        TE_t_full[xs+2:xe-2, ys+2:ye-2, :] .= te_t_3day[buf:nx-buf+1, buf:ny-buf+1, :]
        FH[xs+2:xe-2,  ys+2:ye-2]          .= depth[buf:nx-buf+1,  buf:ny-buf+1]
        RAC[xs+2:xe-2, ys+2:ye-2]          .= rac[buf:nx-buf+1,    buf:ny-buf+1]


    end
end

KE_tmean = dropdims(mean(KE_full,  dims=3), dims=3)   # NX x NY
PE_tmean = dropdims(mean(PE_full,  dims=3), dims=3)   # NX x NY


valid_mask = (RAC .> 0.0) .& (FH .> 0.0)
println("\nValid points: $(sum(valid_mask)) / $(length(valid_mask))")


lat_vec = collect(lat)


KE_zonal  = zeros(NY)
PE_zonal  = zeros(NY)


for j in 1:NY
    idx = valid_mask[:, j]
    if any(idx)
        w          = RAC[idx, j]
        KE_zonal[j] = sum(KE_tmean[idx, j] .* w) / sum(w)
        PE_zonal[j] = sum(PE_tmean[idx, j] .* w) / sum(w)
    end
end


# KE/APE ratio
ratio_zonal = KE_zonal ./ PE_zonal


# ============================================================
# Theoretical (ω² + f²) / (ω² - f²) for semidiurnal band (9–15 hr)
# ω_M2 = M2 semidiurnal frequency (dominant); band edges at 9 hr and 15 hr
# f from coriolis_frequency() already in your FluxUtils.jl
# ============================================================
ω_M2 = 2π / (12.4206 * 3600)  # M2 semidiurnal [rad/s]
ω_lo = 2π / (15 * 3600)        # 15-hr edge [rad/s]
ω_hi = 2π / (9  * 3600)        # 9-hr edge  [rad/s]


theory_M2 = fill(NaN, NY)
theory_lo = fill(NaN, NY)
theory_hi = fill(NaN, NY)


for j in 1:NY
    f_j = abs(coriolis_frequency(lat_vec[j]))   # [rad/s]
    if ω_M2 > f_j
        theory_M2[j] = (ω_M2^2 + f_j^2) / (ω_M2^2 - f_j^2)
    end
    if ω_lo > f_j
        theory_lo[j] = (ω_lo^2 + f_j^2) / (ω_lo^2 - f_j^2)
    end
    if ω_hi > f_j
        theory_hi[j] = (ω_hi^2 + f_j^2) / (ω_hi^2 - f_j^2)
    end
end


# ============================================================
# PLOT 1: KE and APE zonal profiles
# ============================================================
mkpath(joinpath(base2, "EnergyRatio"))


fig1 = Figure(resolution=(900, 560), backgroundcolor=:white)
ax1  = Axis(fig1[1, 1],
    title     = "Zonal-Mean, Time-Mean, Depth-Integrated  KE & APE",
    xlabel    = "Latitude (°N)",
    ylabel    = "Energy  (J m⁻²)",
    titlesize = 16, xlabelsize = 13, ylabelsize = 13)


lines!(ax1, lat_vec, KE_zonal, linewidth=2.5, color=:royalblue,  label="KE")
lines!(ax1, lat_vec, PE_zonal, linewidth=2.5, color=:darkorange, label="APE")
axislegend(ax1, position=:rt)


save(joinpath(base2, "EnergyRatio", "KE_APE_zonal.png"), fig1)
println("Saved: KE_APE_zonal.png")
display(fig1)


# ============================================================
# PLOT 2: KE/APE ratio  — latitude on Y, ratio on X (0 to 20)
#         observed ratio + theoretical (ω²+f²)/(ω²-f²) overlaid
# ============================================================
fig2 = Figure(resolution=(600, 800), backgroundcolor=:white)
ax2  = Axis(fig2[1, 1],
    title     = "Zonal-Mean, Time-Mean, Depth-Integrated  KE / APE",
    xlabel    = "KE / APE  (dimensionless)",
    ylabel    = "Latitude (°N)",
    titlesize = 14, xlabelsize = 12, ylabelsize = 12,
    limits    = ((0, 20), (minlat, maxlat)))


# Shaded band between 9-hr and 15-hr theoretical curves
valid_band = isfinite.(theory_lo) .& isfinite.(theory_hi)
if any(valid_band)
    band!(ax2,
        lat_vec[valid_band],
        theory_lo[valid_band],
        theory_hi[valid_band],
        color = (:darkorange, 0.20))
end


# Theoretical M2 semidiurnal curve
lines!(ax2, theory_M2, lat_vec,
    color     = :darkorange,
    linewidth = 2.0,
    linestyle = :dash,
    label     = "(ω²+f²)/(ω²-f²)  M2")


# Observed KE/APE
lines!(ax2, ratio_zonal, lat_vec,
    color     = :royalblue,
    linewidth = 2.5,
    label     = "KE / APE  (observed)")


# KE = APE reference
vlines!(ax2, [1.0],
    color     = :firebrick,
    linewidth = 1.2,
    linestyle = :dot,
    label     = "KE = APE")


axislegend(ax2, position=:rt, labelsize=11)


save(joinpath(base2, "EnergyRatio", "KE_APE_ratio_zonal.png"), fig2)
println("Saved: KE_APE_ratio_zonal.png")
display(fig2)


println("\nDone!")




