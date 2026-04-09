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
# DEPTH MASK: keep only points deeper than 3000 m
# ==========================================================


println("\nApplying depth mask (> 3000 m)...")
depth_mask = FH .> 3000.0
RAC_masked = RAC .* depth_mask
println("Deep points (>3000m): $(sum(depth_mask)) / $(length(depth_mask))")


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================


function area_weighted_mean(field, area)
    valid_mask = .!(isnan.(field) .| isinf.(field)) .& (area .> 0.0)
    return sum(field[valid_mask] .* area[valid_mask]) / sum(area[valid_mask])
end


function area_weighted_var(field, area, weighted_mean)
    valid_mask = .!(isnan.(field) .| isinf.(field)) .& (area .> 0.0)
    w = area[valid_mask]
    f = field[valid_mask]
    return sum(w .* (f .- weighted_mean).^2) / sum(w)
end


# ==========================================================
# 3-DAY TIMESERIES: mean + spatial variance (deep points only)
# ==========================================================


println("Calculating residual timeseries + spatial variance...")


R1_ts = zeros(nt3);  R1_var = zeros(nt3)
R2_ts = zeros(nt3);  R2_var = zeros(nt3)
R3_ts = zeros(nt3);  R3_var = zeros(nt3)
R4_ts = zeros(nt3);  R4_var = zeros(nt3)
R5_ts = zeros(nt3);  R5_var = zeros(nt3)


for t in 1:nt3
    norm  = rho0 .* FH


    C_tn  = Conv_3day[:,:,t] ./ norm
    Fd_tn = FDiv_3day[:,:,t] ./ norm
    A_tn  = (U_KE_3day[:,:,t] .+ U_PE_3day[:,:,t]) ./ norm
    PS_tn = (SP_H_3day[:,:,t] .+ SP_V_3day[:,:,t])  ./ norm
    BP_tn = BP_3day[:,:,t]   ./ norm
    ET_tn = ET_3day[:,:,t]   ./ norm


    # Compute residuals and apply depth mask explicitly
    r1 = -(C_tn .- Fd_tn)                                    .* depth_mask
    r2 = -(C_tn .- Fd_tn .- A_tn)                            .* depth_mask
    r3 = -(C_tn .- Fd_tn .- A_tn .+ PS_tn)                   .* depth_mask
    r4 = -(C_tn .- Fd_tn .- A_tn .+ PS_tn .+ BP_tn)          .* depth_mask
    r5 = -(C_tn .- Fd_tn .- A_tn .+ PS_tn .+ BP_tn .- ET_tn) .* depth_mask


    R1_ts[t] = area_weighted_mean(r1, RAC_masked);  R1_var[t] = area_weighted_var(r1, RAC_masked, R1_ts[t])
    R2_ts[t] = area_weighted_mean(r2, RAC_masked);  R2_var[t] = area_weighted_var(r2, RAC_masked, R2_ts[t])
    R3_ts[t] = area_weighted_mean(r3, RAC_masked);  R3_var[t] = area_weighted_var(r3, RAC_masked, R3_ts[t])
    R4_ts[t] = area_weighted_mean(r4, RAC_masked);  R4_var[t] = area_weighted_var(r4, RAC_masked, R4_ts[t])
    R5_ts[t] = area_weighted_mean(r5, RAC_masked);  R5_var[t] = area_weighted_var(r5, RAC_masked, R5_ts[t])
end


println("Done. nt3 = $nt3 periods")


# ==========================================================
# PLOT: spatial variance — 5 lines
# ==========================================================


sc2    = 1e16
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
    "Var(R1) = -(C - ∇·F)",
    "Var(R2) = -(C - ∇·F - A)",
    "Var(R3) = -(C - ∇·F - A + Pₛ)",
    "Var(R4) = -(C - ∇·F - A + Pₛ + Pᵦ)",
    "Var(R5) = -(C - ∇·F - A + Pₛ + Pᵦ - ∂E/∂t)",
]


vars = [R1_var, R2_var, R3_var, R4_var, R5_var]


fig_ts = Figure(resolution=(1100, 450), fontsize=14, backgroundcolor=:white)
ax_ts  = Axis(fig_ts[1, 1];
    title  = "Spatial Variance of Residuals — deep points only (H > 3000 m, 3-day averages)",
    xlabel = "Time  [days]",
    ylabel = "Spatial variance  [×10⁻¹⁶ W² kg⁻²]",
    axis_theme...
)


hlines!(ax_ts, [0.0]; color=RGBAf(0,0,0,0.3), linewidth=0.8, linestyle=:dash)


for (i, (v, c, lbl)) in enumerate(zip(vars, cols, labels))
    lw = i == 5 ? 2.0 : 1.8
    lines!(ax_ts, t_axis, v .* sc2; color=c, linewidth=lw, label=lbl)
end


axislegend(ax_ts; position=:rt, leg_style...)


FIGDIR = cfg["fig_base"]
mkpath(FIGDIR)
outpath = joinpath(FIGDIR, "Residual_Var_Timeseries_3day_deep_v3.png")
save(outpath, fig_ts, px_per_unit=2)
println("\nFigure saved → $outpath")
display(fig_ts)




