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
nt_chunk = 72
n_chunks = div(nt, nt_chunk)
ring_steps = nt_chunk
t_safe_start = ring_steps + 1
t_safe_end   = nt - ring_steps




safe_chunks = [c for c in 1:n_chunks
             if (c-1)*nt_chunk + 1 >= t_safe_start &&
                c*nt_chunk          <= t_safe_end]
@info "Safe 3-day chunks: $(length(safe_chunks)) of $n_chunks  (chunks $(safe_chunks[1])–$(safe_chunks[end]))"


rho0 = 1027.5


# --- Date axis ---
t_origin   = DateTime(2012, 3, 4, 0, 0, 0)
dates_3day = [t_origin + Day(3*(i-1)) + Hour(36) for i in 1:nt3-2]
t_numeric  = [Dates.value(d - t_origin) / (1000*3600*24) for d in dates_3day]


tick_every  = 5
tick_inds   = 1:tick_every:nt3-2
tick_vals   = t_numeric[tick_inds]
tick_labels = [Dates.format(dates_3day[i], "u dd\nyyyy") for i in tick_inds]


# --- Thickness ---
thk   = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF   = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)
println("Computing area-averaged budget terms for $nt3 3-day periods...")


# --- Global arrays ---
Conv_full = zeros(NX, NY, nt3-2)
FDiv_full = zeros(NX, NY, nt3-2)
U_KE_full = zeros(NX, NY, nt3-2)
U_PE_full = zeros(NX, NY, nt3-2)
SP_H_full = zeros(NX, NY, nt3-2)
SP_V_full = zeros(NX, NY, nt3-2)
BP_full   = zeros(NX, NY, nt3-2)
ET_full   = zeros(NX, NY, nt3-2)
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


        DRFfull4_f32 = Float32.(reshape(DRFfull, nx, ny, nz, 1))
        
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
        WPI_full[xs+2:xe-2, ys+2:ye-2,:]   .= wpi_3day[buf:nx-buf+1, buf:ny-buf+1, :]
        FH[xs+2:xe-2, ys+2:ye-2]  .= depth[buf:nx-buf+1, buf:ny-buf+1]
        RAC[xs+2:xe-2, ys+2:ye-2] .= rac[buf:nx-buf+1, buf:ny-buf+1]


        println("  Done.")
    end
end


# ============================================================
    # Full domain mask (all valid points)
    # ============================================================
    valid_mask = (RAC .> 0.0) .& (FH .> 0.0)
    total_area = sum(RAC[valid_mask])


    println("\nTotal valid points : $(sum(valid_mask))")
    println("Total area         : $(total_area) m²")


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


    # ============================================================
    # Compute full-domain area-averaged terms
    # ============================================================
    println("\nNormalising fields...")
    Conv_n = norm_field(Conv_full, valid_mask, FH)
    FDiv_n = norm_field(FDiv_full, valid_mask, FH)
    U_KE_n = norm_field(U_KE_full, valid_mask, FH)
    U_PE_n = norm_field(U_PE_full, valid_mask, FH)
    SP_H_n = norm_field(SP_H_full, valid_mask, FH)
    SP_V_n = norm_field(SP_V_full, valid_mask, FH)
    BP_n   = norm_field(BP_full,   valid_mask, FH)
    ET_n   = norm_field(ET_full,   valid_mask, FH)
    WPI_n = norm_field(WPI_full,   valid_mask, FH)

    CWI = Conv_n .+ WPI_n
    A_n        = U_KE_n .+ U_PE_n
    PS_n       = SP_H_n .+ SP_V_n
    MF_n       = BP_n   .+ PS_n                          # mean flow = Pb + Ps
    Residual_n = -(Conv_n .+ WPI_n.- (FDiv_n .+ A_n) .+ PS_n .+ BP_n .- ET_n)


    println("Area-averaging...")
    Conv_avg     = area_avg(CWI,     valid_mask, RAC, total_area)
    FDiv_avg     = area_avg(FDiv_n,     valid_mask, RAC, total_area)
    ET_avg       = area_avg(ET_n,       valid_mask, RAC, total_area)
    Residual_avg = area_avg(Residual_n, valid_mask, RAC, total_area)
    MF_avg       = area_avg(MF_n,       valid_mask, RAC, total_area)


    # ============================================================
    # Compute ratios  — denominator: C + WI (ET)
    # NaN-safe mean: filter out NaN before calling mean()
    # ============================================================
    function safe_ratio(num, denom; tol=1e-30)
        return [abs(denom[t]) > tol ? num[t] / denom[t] : NaN for t in eachindex(denom)]
    end


    # NaN-safe mean (Julia has no nanmean — filter finite values)
    nanmean(x) = mean(filter(isfinite, x))


    denom = Conv_avg        # C + WI


    q1 = safe_ratio(Residual_avg, denom)   # Dissipation   / (C + WI)
    q2 = safe_ratio(FDiv_avg,     denom)   # Flux div      / (C + WI)
    q3 = safe_ratio(MF_avg,       denom)   # Mean flow     / (C + WI)


    # ============================================================
    # Print time-averaged percentages
    # ============================================================
    q1_pct = nanmean(q1) * 100
    q2_pct = nanmean(q2) * 100
    q3_pct = nanmean(q3) * 100


    println("\n========================================")
    println("  Time-averaged fractions (full domain)")
    println("========================================")
    @printf("  Dissipation  ⟨R⟩ / (⟨C⟩ + WI)           : %7.2f %%\n", q1_pct)
    @printf("  Flux div     ⟨∇·F⟩ / (⟨C⟩ + WI)         : %7.2f %%\n", q2_pct)
    @printf("  Mean flow    (⟨Pᵦ⟩+⟨Pₛ⟩) / (⟨C⟩ + WI)  : %7.2f %%\n", q3_pct)
    @printf("  Sum of three fractions                   : %7.2f %%\n", q1_pct + q2_pct + q3_pct)
    println("========================================\n")


    # ============================================================
    # Plot theme  (bold font throughout)
    # ============================================================
    FONT     = "FreeSerif Bold"
    tick_col = RGBf(0.20, 0.20, 0.20)
    grid_col = RGBAf(0.75, 0.75, 0.75, 0.6)


    c_q1 = RGBf(0.80, 0.10, 0.10)   # red   — dissipation
    c_q2 = RGBf(0.10, 0.40, 0.75)   # blue  — flux divergence
    c_q3 = RGBf(0.10, 0.55, 0.25)   # green — mean flow


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
    # Figure — single panel, full domain
    # ============================================================
    println("Creating ratio time-series figure (full domain)...")


    fig = Figure(resolution=(700, 320), fontsize=14, backgroundcolor=:white, figure_padding =(5,5,5,5),
                fonts=(; regular=FONT, bold=FONT))


    ax = Axis(fig[1, 1];
        title  = "Full domain  —  Energy fraction time series",
        ylabel = "Fraction  [ ]",
        axis_theme...)


    # Reference lines
    hlines!(ax, [0.0]; color=RGBAf(0,0,0,0.30), linewidth=0.8, linestyle=:dash)
    hlines!(ax, [1.0]; color=RGBAf(0,0,0,0.20), linewidth=0.8, linestyle=:dot)


    # Replace NaN with missing so CairoMakie skips gaps cleanly
    to_plot(v) = [isfinite(x) ? x : missing for x in v]


    lines!(ax, t_numeric, to_plot(q1);
        label     = @sprintf("⟨R⟩ / (⟨C⟩+WI)  [Dissipation]  —  mean = %.1f %%", q1_pct),
        color     = c_q1,
        linewidth = 1.8)


    lines!(ax, t_numeric, to_plot(q2);
        label     = @sprintf("⟨∇·F⟩ / (⟨C⟩+WI)  [Flux div]     —  mean = %.1f %%", q2_pct),
        color     = c_q2,
        linewidth = 1.8)


    lines!(ax, t_numeric, to_plot(q3);
        label     = @sprintf("(⟨Pᵦ⟩+⟨Pₛ⟩) / (⟨C⟩+WI)  [Mean flow]  —  mean = %.1f %%", q3_pct),
        color     = c_q3,
        linewidth = 1.8)


    axislegend(ax; position=:rt, leg_style...)


    outpath = joinpath(FIGDIR, "Budget_Ratios_FullDomain_TimeSeries.png")
    save(outpath, fig, px_per_unit=2)
    println("Figure saved → $outpath")
    display(fig)
#
