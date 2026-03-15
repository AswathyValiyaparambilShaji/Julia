using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML
using CairoMakie




if !isdefined(Main, :FluxUtils)
    include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
    using .FluxUtils: read_bin, bandpassfilter
end




config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg  = TOML.parsefile(config_file)
base  = cfg["base_path"]
base2 = cfg["base_path2"]




# --- Tile & grid ---
buf    = 3
tx, ty = 47, 66
nx     = tx + 2 * buf
ny     = ty + 2 * buf
nz     = 88


# --- Time ---
dto  = 144
Tts  = 366192
nt   = div(Tts, dto)


# --- Thickness ---
thk    = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF    = thk[1:nz]
DRF3d  = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)
rho0   = 998.0
g      = 9.8
T1, T2, delt, N = 9.0, 15.0, 1.0, 4

mkpath(joinpath(base2, "Conv_z_dI"))

# --- Single tile ---
xn = 1
yn = 1
suffix  = @sprintf("%02dx%02d_%d", xn, yn, buf)
suffix2 = @sprintf("%02dx%02d_%d", xn, yn, buf - 2)


# ============================================================
# READ CODE 2 FULL-RECORD CONVERSION  (time-averaged)
# ============================================================
C = Float64.(open(joinpath(base2, "Conv", "Conv_$suffix2.bin"), "r") do io
    nbytes = (nx-2) * (ny-2) * sizeof(Float32)
    reinterpret(Float32, read(io, nbytes))
end) |> x -> reshape(x, nx-2, ny-2)


println("Loaded Code 2 Conv_$suffix2.bin  size=$(size(C))  " *
        "min=$(round(minimum(C), sigdigits=4))  max=$(round(maximum(C), sigdigits=4))")


# --- Read density ---
rho = Float64.(open(joinpath(base, "Density", "rho_in_$suffix.bin"), "r") do io
    reshape(reinterpret(Float64, read(io, nx * ny * nz * nt * sizeof(Float64))), nx, ny, nz, nt)
end)


# --- Grid masks & thickness ---
hFacC   = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
DRFfull = hFacC .* DRF3d
depth   = sum(DRFfull, dims=3)           # (nx, ny, 1)
DRFfull[hFacC .== 0] .= 0.0


# --- Read filtered velocities ---
fu = Float64.(open(joinpath(base2, "UVW_F", "fu_$suffix.bin"), "r") do io
    reshape(reinterpret(Float32, read(io, nx * ny * nz * nt * sizeof(Float32))), nx, ny, nz, nt)
end)
fv = Float64.(open(joinpath(base2, "UVW_F", "fv_$suffix.bin"), "r") do io
    reshape(reinterpret(Float32, read(io, nx * ny * nz * nt * sizeof(Float32))), nx, ny, nz, nt)
end)

println(fu[19,35,2,20])

# --- Grid spacings ---
dx = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
dy = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))


# --- Bandpass filter ---
fr = bandpassfilter(rho, T1, T2, delt, N, nt)


# --- Depth-averaged velocities ---
UDA = dropdims(sum(fu .* DRFfull, dims=3) ./ depth; dims=3)   # (nx, ny, nt)
VDA = dropdims(sum(fv .* DRFfull, dims=3) ./ depth; dims=3)   # (nx, ny, nt)


# --- z and d ---
d   = depth                              # (nx, ny, 1)
z   = -cumsum(DRFfull, dims=3)           # (nx, ny, nz)  negative downward
d2d = dropdims(d, dims=3)               # (nx, ny)


# ============================================================
# EXPANDED FORM OF W(z)  [equation 5.42]
#
# Original:  W(z) = −∇·(d·U_H) − z·∇·U_H
# Expand first term using product rule:
#   −∇·(d·U_H) = −d·∇·U_H − U_H·∇d
# Therefore:
#   W(z) = −d·∇·U_H − U_H·∇d − z·∇·U_H
#         = −(z + d)·∇·U_H   − U_H·∇d
#
# This uses ONE divergence field (∇·U_H) and ONE depth-gradient
# (∇d), computed consistently with the same CFD stencil.
# ============================================================


# --- ∇·U_H  (CFD, cell-centred velocities) ---
divUDA = (
    (UDA[3:nx,   2:ny-1, :] .- UDA[1:nx-2, 2:ny-1, :]) ./
    (dx[2:nx-1,  2:ny-1]    .+ dx[1:nx-2,  2:ny-1])     .+
    (VDA[2:nx-1, 3:ny,   :] .- VDA[2:nx-1, 1:ny-2, :]) ./
    (dy[2:nx-1,  2:ny-1]    .+ dy[2:nx-1,  1:ny-2])
)   # (nx-2, ny-2, nt)


# --- ∇d  (depth gradient, time-independent, same CFD stencil) ---
dddx = (d2d[3:nx,   2:ny-1] .- d2d[1:nx-2, 2:ny-1]) ./
       (dx[2:nx-1, 2:ny-1]  .+ dx[1:nx-2,  2:ny-1])   # (nx-2, ny-2)


dddy = (d2d[2:nx-1, 3:ny  ] .- d2d[2:nx-1, 1:ny-2]) ./
       (dy[2:nx-1, 2:ny-1]  .+ dy[2:nx-1,  1:ny-2])   # (nx-2, ny-2)


# --- U_H·∇d  (barotropic flow advecting depth, time-varying) ---
UDA_int = UDA[2:nx-1, 2:ny-1, :]   # (nx-2, ny-2, nt)
VDA_int = VDA[2:nx-1, 2:ny-1, :]   # (nx-2, ny-2, nt)


UdotGradD = UDA_int .* reshape(dddx, nx-2, ny-2, 1) .+
            VDA_int .* reshape(dddy, nx-2, ny-2, 1)   # (nx-2, ny-2, nt)


# --- (z + d) at interior points ---
z_int = z[2:nx-1, 2:ny-1, :]                          # (nx-2, ny-2, nz)
d_int = d2d[2:nx-1, 2:ny-1]                           # (nx-2, ny-2)
z_p_d = z[:,:,:] .+ d

zpd = reshape(z_int, nx-2, ny-2, nz, 1) .+
      reshape(d_int, nx-2, ny-2, 1,  1)  # (z + d) >= 0  size: (nx-2, ny-2, nz, 1)
# --- W(z) = −(z+d)·∇·U_H − U_H·∇d ---
Wz = .- zpd .* reshape(divUDA,    nx-2, ny-2, 1,  nt) .-
        reshape(UdotGradD, nx-2, ny-2, 1, nt)
# broadcast: (nx-2, ny-2, nz, nt)

# --- Density perturbation (bandpass filtered) ---
rho_prime = fr
rho_int   = rho_prime[2:nx-1, 2:ny-1, :, :]   # (nx-2, ny-2, nz, nt)


# --- Conversion rate per depth level ---
# C = ρ′ · g · W  (no extra minus: sign is inside W via eq 5.42)
Cz = rho_int .* g .* Wz                        # (nx-2, ny-2, nz, nt)


# --- Depth integration ---
DRFint   = DRFfull[2:nx-1, 2:ny-1, :]          # (nx-2, ny-2, nz)
DRFint4d = reshape(DRFint, nx-2, ny-2, nz, 1)  # (nx-2, ny-2, nz, 1)


Ca_full = dropdims(sum(Cz .* DRFint4d, dims=3), dims=3)   # (nx-2, ny-2, nt)


# --- Time mean ---
ca = dropdims(mean(Ca_full; dims=3); dims=3)    # (nx-2, ny-2)


# --- Save ---
open(joinpath(base2, "Conv_z_dI", "Conv_z_$suffix2.bin"), "w") do io
    write(io, Float32.(ca))
end
println("Saved: Conv_z_$suffix2.bin")


# ============================================================
# POINT COMPARISON
# ============================================================
xi, yi = 35, 30


@printf("\n=== Point comparison at trimmed-grid index (%d, %d) ===\n", xi, yi)
@printf("  Code 1  Conv_z  : %+.6e  W/m²\n", ca[xi, yi])
@printf("  Code 2  Conv    : %+.6e  W/m²\n", C[xi, yi])
@printf("  Ratio C1/C2     : %+.4f\n",        ca[xi, yi] / C[xi, yi])
@printf("  Sign match      : %s\n",            sign(ca[xi, yi]) == sign(C[xi, yi]) ? "YES" : "NO")



# ============================================================
# PLOTS
# ============================================================


# --- Code 1: time-mean conversion ---
fig = Figure(resolution=(700, 500))
ax  = Axis(fig[1,1],
    title  = "DIV U $suffix2",
    xlabel = "x index",
    ylabel = "y index")
hm  = heatmap!(ax, dddx[1:end-1, 1:end-1], colormap=Reverse(:RdBu), colorrange=(-0.08, 0.08))
Colorbar(fig[1,2], hm, label="m")
display(fig)


# --- Code 2: time-mean conversion ---
fig2 = Figure(resolution=(700, 500))
ax2  = Axis(fig2[1,1],
    title  = "Code 2: Conv (time mean) - tile $suffix2",
    xlabel = "x index",
    ylabel = "y index")
hm2  = heatmap!(ax2, C, colormap=Reverse(:RdBu), colorrange=(-0.05, 0.05))
Colorbar(fig2[1,2], hm2, label="W/m²")
display(fig2)


# --- Vertical profile of Cz at a single point and time ---
fig3 = Figure(resolution=(400, 600))
ax3  = Axis(fig3[1,1],
    title  = "Vertical profile of Cz at (2,3), t=10",
    xlabel = "Conversion (W/m³)",
    ylabel = "Depth (m)")
lines!(ax3, Cz[2, 3, :, 10], z_int[2, 3, :],
    label     = "Cz",
    color     = :blue,
    linewidth = 2.5)
axislegend(ax3; position=:rt)
display(fig3)




