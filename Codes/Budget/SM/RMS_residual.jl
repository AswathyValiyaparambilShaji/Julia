using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML, CairoMakie


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin


config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base  = cfg["base_path"]
base2 = cfg["base_path2"]


NX, NY = 288, 468
minlat, maxlat = 24.0, 31.91
minlon, maxlon = 193.0, 199.0
lat = range(minlat, maxlat, length=NY)
lon = range(minlon, maxlon, length=NX)


buf = 3
tx, ty = 47, 66
nx = tx + 2*buf
ny = ty + 2*buf


rho0 = 999.8
nz   = 88


dt  = 25
dto = 144
Tts = 366192
nt  = div(Tts, dto)
nt3 = div(nt, 3*24)


thk   = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF   = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)


Conv_3day = zeros(NX, NY, nt3)
FDiv_3day = zeros(NX, NY, nt3)
U_KE_3day = zeros(NX, NY, nt3)
U_PE_3day = zeros(NX, NY, nt3)
SP_H_3day = zeros(NX, NY, nt3)
SP_V_3day = zeros(NX, NY, nt3)
BP_3day   = zeros(NX, NY, nt3)
ET_3day   = zeros(NX, NY, nt3)
FH  = zeros(NX, NY)
RAC = zeros(NX, NY)


println("Loading energy budget terms...")


for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix  = @sprintf("%02dx%02d_%d", xn, yn, buf)
        suffix2 = @sprintf("%02dx%02d_%d", xn, yn, buf-2)


        hFacC   = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
        dx      = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
        dy      = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))
        DRFfull = hFacC .* DRF3d
        depth   = dropdims(sum(DRFfull, dims=3), dims=3)
        DRFfull[hFacC .== 0] .= 0.0
        rac = dx .* dy


        fxD = Float64.(open(joinpath(base2, "FDiv_3day", "FDiv_3day_$(suffix2).bin"), "r") do io
            nbytes = (nx-2)*(ny-2)*nt3*sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx-2, ny-2, nt3)
        end)
        C = Float64.(open(joinpath(base2, "Conv_3day", "Conv_3day_$(suffix2).bin"), "r") do io
            nbytes = (nx-2)*(ny-2)*nt3*sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx-2, ny-2, nt3)
        end)
        u_ke_3day = Float64.(open(joinpath(base2, "U_KE_3dayold", "u_ke_3day_$suffix.bin"), "r") do io
            nbytes = nx*ny*nt3*sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt3)
        end)
        u_pe_3day = Float64.(open(joinpath(base2, "U_PE_3dayold", "u_pe_3day_$suffix.bin"), "r") do io
            nbytes = nx*ny*nt3*sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt3)
        end)
        sp_h_3day = Float64.(open(joinpath(base2, "SP_H_3dayold", "sp_h_3day_$suffix.bin"), "r") do io
            nbytes = nx*ny*nt3*sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt3)
        end)
        sp_v_3day = Float64.(open(joinpath(base2, "SP_V_3dayold", "sp_v_3day_$suffix.bin"), "r") do io
            nbytes = nx*ny*nt3*sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt3)
        end)
        bp_3day = Float64.(open(joinpath(base2, "BP3day_old", "bp_3day_$suffix.bin"), "r") do io
            nbytes = nx*ny*nt3*sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt3)
        end)
        te_3day = Float64.(open(joinpath(base2, "TE_t_3day", "te_t_3day_$suffix.bin"), "r") do io
            nbytes = nx*ny*nt3*sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt3)
        end)


        xs = (xn - 1) * tx + 1
        xe = xs + tx + (2 * buf) - 1
        ys = (yn - 1) * ty + 1
        ye = ys + ty + (2 * buf) - 1


        Conv_3day[xs+2:xe-2, ys+2:ye-2, :] .= C[2:end-1, 2:end-1, :]
        FDiv_3day[xs+2:xe-2, ys+2:ye-2, :] .= fxD[2:end-1, 2:end-1, :]
        U_KE_3day[xs+2:xe-2, ys+2:ye-2, :] .= u_ke_3day[buf:nx-buf+1, buf:ny-buf+1, :]
        U_PE_3day[xs+2:xe-2, ys+2:ye-2, :] .= u_pe_3day[buf:nx-buf+1, buf:ny-buf+1, :]
        SP_H_3day[xs+2:xe-2, ys+2:ye-2, :] .= sp_h_3day[buf:nx-buf+1, buf:ny-buf+1, :]
        SP_V_3day[xs+2:xe-2, ys+2:ye-2, :] .= sp_v_3day[buf:nx-buf+1, buf:ny-buf+1, :]
        BP_3day[xs+2:xe-2, ys+2:ye-2, :]   .= bp_3day[buf:nx-buf+1, buf:ny-buf+1, :]
        ET_3day[xs+2:xe-2, ys+2:ye-2, :]   .= te_3day[buf:nx-buf+1, buf:ny-buf+1, :]
        FH[xs+2:xe-2, ys+2:ye-2]           .= depth[buf:nx-buf+1, buf:ny-buf+1]
        RAC[xs+2:xe-2, ys+2:ye-2]          .= rac[buf:nx-buf+1, buf:ny-buf+1]


        println("Completed tile $suffix")
    end
end


# ==========================================================
# MASKS: full ocean and deep-only (H > 3000 m)
# ==========================================================


println("\nBuilding masks...")
full_mask = FH .> 0.0          # all wet ocean points
deep_mask = FH .> 3000.0       # deep points only


RAC_full = RAC .* full_mask
RAC_deep = RAC .* deep_mask


println("All ocean points : $(sum(full_mask)) / $(length(full_mask))")
println("Deep points >3000m: $(sum(deep_mask)) / $(length(full_mask))")


# ==========================================================
# HELPER: area-weighted RMS from zero  sqrt(Σ w·r²) / Σw)
# ==========================================================


function area_weighted_rms(field, area)
    valid = .!(isnan.(field) .| isinf.(field)) .& (area .> 0.0)
    w = area[valid]
    f = field[valid]
    return sqrt(sum(w .* f.^2) / sum(w))
end


# ==========================================================
# 3-DAY TIMESERIES: area-weighted RMS from zero
# ==========================================================


println("Calculating residual RMS timeseries...")


# Full-domain RMS
R1_full = zeros(nt3); R2_full = zeros(nt3); R3_full = zeros(nt3)
R4_full = zeros(nt3); R5_full = zeros(nt3)


# Deep-only RMS
R1_deep = zeros(nt3); R2_deep = zeros(nt3); R3_deep = zeros(nt3)
R4_deep = zeros(nt3); R5_deep = zeros(nt3)


for t in 1:nt3
    norm = rho0 .* FH


    C_tn  = Conv_3day[:,:,t] ./ norm
    Fd_tn = FDiv_3day[:,:,t] ./ norm
    A_tn  = (U_KE_3day[:,:,t] .+ U_PE_3day[:,:,t]) ./ norm
    PS_tn = (SP_H_3day[:,:,t] .+ SP_V_3day[:,:,t])  ./ norm
    BP_tn = BP_3day[:,:,t]   ./ norm
    ET_tn = ET_3day[:,:,t]   ./ norm


    r1 = -(C_tn .- Fd_tn)
    r2 = -(C_tn .- Fd_tn .- A_tn)
    r3 = -(C_tn .- Fd_tn .- A_tn .+ PS_tn)
    r4 = -(C_tn .- Fd_tn .- A_tn .+ PS_tn .+ BP_tn)
    r5 = -(C_tn .- Fd_tn .- A_tn .+ PS_tn .+ BP_tn .- ET_tn)


    for (r, rf_full, rf_deep) in zip(
            [r1, r2, r3, r4, r5],
            [R1_full, R2_full, R3_full, R4_full, R5_full],
            [R1_deep, R2_deep, R3_deep, R4_deep, R5_deep])
        rf_full[t] = area_weighted_rms(r .* full_mask, RAC_full)
        rf_deep[t] = area_weighted_rms(r .* deep_mask, RAC_deep)
    end
end


println("Done. nt3 = $nt3 periods")


# ==========================================================
# PLOT: two-panel — full domain (top) and deep only (bottom)
# ==========================================================


sc     = 1e8
t_axis = collect(1:nt3) .* 3


tick_col = RGBf(0.20, 0.20, 0.20)
grid_col = RGBAf(0.75, 0.75, 0.75, 0.6)


axis_theme = (
    backgroundcolor   = :white,
    xgridcolor        = grid_col,   ygridcolor        = grid_col,
    xgridwidth        = 0.6,        ygridwidth        = 0.6,
    xtickcolor        = tick_col,   ytickcolor        = tick_col,
    xticklabelcolor   = tick_col,   yticklabelcolor   = tick_col,
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


leg_style = (
    framecolor      = RGBAf(0.3, 0.3, 0.3, 0.4),
    backgroundcolor = RGBAf(1.0, 1.0, 1.0, 0.85),
    labelcolor      = RGBf(0.10, 0.10, 0.10),
    labelsize       = 11,
    rowgap          = 3,
    patchsize       = (22, 2),
)


cols = [
    RGBf(0.10, 0.40, 0.75),
    RGBf(0.80, 0.50, 0.00),
    RGBf(0.10, 0.60, 0.30),
    RGBf(0.60, 0.10, 0.70),
    RGBf(0.00, 0.00, 0.00),
]


labels = [
    "R1 = -(C - ∇·F)",
    "R2 = -(C - ∇·F - A)",
    "R3 = -(C - ∇·F - A + Pₛ)",
    "R4 = -(C - ∇·F - A + Pₛ + Pᵦ)",
    "R5 = -(C - ∇·F - A + Pₛ + Pᵦ - ∂E/∂t)",
]


full_series = [R1_full, R2_full, R3_full, R4_full, R5_full]
deep_series = [R1_deep, R2_deep, R3_deep, R4_deep, R5_deep]


fig = Figure(size=(1100, 750), fontsize=14, backgroundcolor=:white)


ax_full = Axis(fig[1, 1];
    title  = "RMS Residual — Full domain (all ocean)",
    xlabel = "Time  [days]",
    ylabel = "Area-weighted RMS  [×10⁻⁸ W kg⁻¹]",
    axis_theme...
)


ax_deep = Axis(fig[2, 1];
    title  = "RMS Residual — Deep points only (H > 3000 m)",
    xlabel = "Time  [days]",
    ylabel = "Area-weighted RMS  [×10⁻⁸ W kg⁻¹]",
    axis_theme...
)


for ax in (ax_full, ax_deep)
    hlines!(ax, [0.0]; color=RGBAf(0,0,0,0.3), linewidth=0.8, linestyle=:dash)
end


for (i, (c, lbl)) in enumerate(zip(cols, labels))
    lw = i == 5 ? 2.0 : 1.8
    lines!(ax_full, t_axis, full_series[i] .* sc; color=c, linewidth=lw, label=lbl)
    lines!(ax_deep, t_axis, deep_series[i] .* sc; color=c, linewidth=lw, label=lbl)
end


axislegend(ax_full; position=:rt, leg_style...)
axislegend(ax_deep; position=:rt, leg_style...)


rowgap!(fig.layout, 20)


FIGDIR = cfg["fig_base"]
mkpath(FIGDIR)
outpath = joinpath(FIGDIR, "Residual_RMS_Timeseries_3day_fullVsDeep_v2.png")
save(outpath, fig, px_per_unit=2)
println("\nFigure saved → $outpath")
display(fig)




