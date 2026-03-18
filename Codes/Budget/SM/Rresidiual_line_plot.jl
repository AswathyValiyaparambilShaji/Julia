

using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML, CairoMakie


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin


config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base  = cfg["base_path"]
base2 = cfg["base_path2"]


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


rho0 = 999.8
nz   = 88


dt  = 25
dto = 144
Tts = 366192
nt  = div(Tts, dto)
nt3 = div(nt, 3*24)


thk  = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF  = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)


# --- Global arrays ---
Conv_3day  = zeros(NX, NY, nt3)
FDiv_3day  = zeros(NX, NY, nt3)
U_KE_3day  = zeros(NX, NY, nt3)
U_PE_3day  = zeros(NX, NY, nt3)
SP_H_3day  = zeros(NX, NY, nt3)
SP_V_3day  = zeros(NX, NY, nt3)
BP_3day    = zeros(NX, NY, nt3)
ET_3day    = zeros(NX, NY, nt3)
FH  = zeros(NX, NY)
RAC = zeros(NX, NY)


println("Loading energy budget terms...")


# ==========================================================
# LOAD ALL TERMS
# ==========================================================


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


        te_3day = Float64.(open(joinpath(base2, "TE_t_3day", "te_t_3day_$suffix.bin"), "r") do io
            nbytes = nx*ny*nt3*sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt3)
        end)


        xs = (xn - 1) * tx + 1
        xe = xs + tx + (2 * buf) - 1
        ys = (yn - 1) * ty + 1
        ye = ys + ty + (2 * buf) - 1


        Conv_3day[xs+2:xe-2, ys+2:ye-2, :]  .= C[2:end-1, 2:end-1, :]
        FDiv_3day[xs+2:xe-2, ys+2:ye-2, :]  .= fxD[2:end-1, 2:end-1, :]
        U_KE_3day[xs+2:xe-2, ys+2:ye-2, :]  .= u_ke_3day[buf:nx-buf+1, buf:ny-buf+1, :]
        U_PE_3day[xs+2:xe-2, ys+2:ye-2, :]  .= u_pe_3day[buf:nx-buf+1, buf:ny-buf+1, :]
        SP_H_3day[xs+2:xe-2, ys+2:ye-2, :]  .= sp_h_3day[buf:nx-buf+1, buf:ny-buf+1, :]
        SP_V_3day[xs+2:xe-2, ys+2:ye-2, :]  .= sp_v_3day[buf:nx-buf+1, buf:ny-buf+1, :]
        BP_3day[xs+2:xe-2, ys+2:ye-2, :]    .= bp_3day[buf:nx-buf+1, buf:ny-buf+1, :]
        ET_3day[xs+2:xe-2, ys+2:ye-2, :]    .= te_3day[buf:nx-buf+1, buf:ny-buf+1, :]
        FH[xs+2:xe-2, ys+2:ye-2]            .= depth[buf:nx-buf+1, buf:ny-buf+1]
        RAC[xs+2:xe-2, ys+2:ye-2]           .= rac[buf:nx-buf+1, buf:ny-buf+1]


        println("Completed tile $suffix")
    end
end


println("\nCalculating residual timeseries...")


# ==========================================================
# AREA-WEIGHTED MEAN (valid points only)
# ==========================================================


function area_weighted_mean(field, area)
    valid_mask = .!(isnan.(field) .| isinf.(field)) .& (area .> 0.0)
    return sum(field[valid_mask] .* area[valid_mask]) / sum(area[valid_mask])
end


# ==========================================================
# 3-DAY TIMESERIES: normalize by ρ₀·H pointwise, then area-average
# ==========================================================


R1_ts = zeros(nt3)
R2_ts = zeros(nt3)
R3_ts = zeros(nt3)
R4_ts = zeros(nt3)
R5_ts = zeros(nt3)


for t in 1:nt3
    norm = rho0 .* FH   # spatially varying ρ₀·H(x,y)


    C_tn  = Conv_3day[:,:,t]  ./ norm
    Fd_tn = FDiv_3day[:,:,t]  ./ norm
    A_tn  = (U_KE_3day[:,:,t] .+ U_PE_3day[:,:,t]) ./ norm
    PS_tn = (SP_H_3day[:,:,t] .+ SP_V_3day[:,:,t])  ./ norm
    BP_tn = BP_3day[:,:,t]    ./ norm
    ET_tn = ET_3day[:,:,t]    ./ norm


    R1_ts[t] = area_weighted_mean(-(C_tn .- Fd_tn),                                      RAC)
    R2_ts[t] = area_weighted_mean(-(C_tn .- Fd_tn .- A_tn),                              RAC)
    R3_ts[t] = area_weighted_mean(-(C_tn .- Fd_tn .- A_tn .+ PS_tn),                     RAC)
    R4_ts[t] = area_weighted_mean(-(C_tn .- Fd_tn .- A_tn .+ PS_tn .+ BP_tn),            RAC)
    R5_ts[t] = area_weighted_mean(-(C_tn .- Fd_tn .- A_tn .+ PS_tn .+ BP_tn .- ET_tn),   RAC)
end


println("Residual timeseries computed (nt3 = $nt3 points)")


# ==========================================================
# PLOT
# ==========================================================


sc        = 1e8
t_axis    = collect(1:nt3) .* 3   # time in days


tick_col  = RGBf(0.20, 0.20, 0.20)
grid_col  = RGBAf(0.75, 0.75, 0.75, 0.6)


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


leg_style = (
    framecolor      = RGBAf(0.3, 0.3, 0.3, 0.4),
    backgroundcolor = RGBAf(1.0, 1.0, 1.0, 0.85),
    labelcolor      = RGBf(0.10, 0.10, 0.10),
    labelsize       = 11,
    rowgap          = 3,
    patchsize       = (22, 2),
)


fig_ts = Figure(resolution=(1100, 450), fontsize=14, backgroundcolor=:white)


ax_ts = Axis(fig_ts[1, 1];
    title  = "Cumulative Residual Timeseries (3-day averages, normalized by ρ₀H)",
    xlabel = "Time  [days]",
    ylabel = "Area-averaged residual  [×10⁻⁸ W kg⁻¹]",
    axis_theme...
)


hlines!(ax_ts, [0.0]; color=RGBAf(0,0,0,0.3), linewidth=0.8, linestyle=:dash)


lines!(ax_ts, t_axis, R1_ts .* sc; label="R1 = -(C - ∇·F)",                          color=RGBf(0.10,0.40,0.75), linewidth=1.8)
lines!(ax_ts, t_axis, R2_ts .* sc; label="R2 = -(C - ∇·F - A)",                      color=RGBf(0.80,0.50,0.00), linewidth=1.8)
lines!(ax_ts, t_axis, R3_ts .* sc; label="R3 = -(C - ∇·F - A + Pₛ)",                 color=RGBf(0.10,0.60,0.30), linewidth=1.8)
lines!(ax_ts, t_axis, R4_ts .* sc; label="R4 = -(C - ∇·F - A + Pₛ + Pᵦ)",           color=RGBf(0.60,0.10,0.70), linewidth=1.8)
lines!(ax_ts, t_axis, R5_ts .* sc; label="R5 = -(C - ∇·F - A + Pₛ + Pᵦ - ∂E/∂t)",  color=:black,               linewidth=2.0)


axislegend(ax_ts; position=:rt, leg_style...)


FIGDIR = cfg["fig_base"]
mkpath(FIGDIR)
outpath = joinpath(FIGDIR, "Residual_Timeseries_3day_normalized.png")
save(outpath, fig_ts, px_per_unit=2)
println("\nFigure saved → $outpath")
display(fig_ts)


