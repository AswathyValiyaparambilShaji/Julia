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
rho0=998.0

g = 9.8
T1, T2, delt, N = 9.0, 15.0, 1.0, 4


mkpath(joinpath(base2, "Conv_z_dI"))


# --- Single tile ---
xn = 1
yn = 1


suffix  = @sprintf("%02dx%02d_%d", xn, yn, buf)
suffix2 = @sprintf("%02dx%02d_%d", xn, yn, buf - 2)


# --- Read density ---
rho = Float64.(open(joinpath(base, "Density", "rho_in_$suffix.bin"), "r") do io
    reshape(reinterpret(Float64, read(io, nx * ny * nz * nt * sizeof(Float64))), nx, ny, nz, nt)
end)


# --- Grid masks & thickness ---
hFacC   = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
DRFfull = hFacC .* DRF3d
depth   = sum(DRFfull, dims=3)
DRFfull[hFacC .== 0] .= 0.0


# --- Read filtered velocities ---
fu = Float64.(open(joinpath(base2, "UVW_F", "fu_$suffix.bin"), "r") do io
    reshape(reinterpret(Float32, read(io, nx * ny * nz * nt * sizeof(Float32))), nx, ny, nz, nt)
end)


fv = Float64.(open(joinpath(base2, "UVW_F", "fv_$suffix.bin"), "r") do io
    reshape(reinterpret(Float32, read(io, nx * ny * nz * nt * sizeof(Float32))), nx, ny, nz, nt)
end)


# --- Grid spacings ---
dx = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
dy = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))


# --- Bandpass filter ---
fr = bandpassfilter(rho, T1, T2, delt, N, nt)


# --- Depth-averaged velocities ---
UDA = dropdims(sum(fu .* DRFfull, dims=3) ./ depth; dims=3)
VDA = dropdims(sum(fv .* DRFfull, dims=3) ./ depth; dims=3)


# --- z and d ---
d = depth
z = -cumsum(DRFfull, dims=3)


# --- Term 1 ---
dU = dropdims(d, dims=3) .* UDA
dV = dropdims(d, dims=3) .* VDA


term1 = .-(
    (dU[3:nx,   2:ny-1, :] .- dU[1:nx-2, 2:ny-1, :]) ./
    (dx[2:nx-1, 2:ny-1]    .+ dx[1:nx-2, 2:ny-1])     .+
    (dV[2:nx-1, 3:ny,   :] .- dV[2:nx-1, 1:ny-2, :]) ./
    (dy[2:nx-1, 2:ny-1]    .+ dy[2:nx-1, 1:ny-2])
)


# --- Term 2 ---
divUDA = (
    (UDA[3:nx,   2:ny-1, :] .- UDA[1:nx-2, 2:ny-1, :]) ./
    (dx[2:nx-1,  2:ny-1]    .+ dx[1:nx-2,  2:ny-1])     .+
    (VDA[2:nx-1, 3:ny,   :] .- VDA[2:nx-1, 1:ny-2, :]) ./
    (dy[2:nx-1,  2:ny-1]    .+ dy[2:nx-1,  1:ny-2])
)


z_int = z[2:nx-1, 2:ny-1, :]


# --- Wz ---
Wz = reshape(term1,  nx-2, ny-2, 1,  nt) .-
     reshape(z_int,  nx-2, ny-2, nz, 1)  .*
     reshape(divUDA, nx-2, ny-2, 1,  nt)
#size(Wz)
 # --- 4D land mask & 4D thickness/depth ---
        #mask4D    = reshape(hFacC .== 0, nx, ny, nz, 1)
        #DRFfull4D = repeat(DRFfull, 1, 1, 1, nt)       # (nx, ny, nz, nt)
        #depth4D   = repeat(depth,   1, 1, 1, nt)       # (nx, ny,  1, nt)

# --- Conversion ---
#rhoA_3d   = sum(fr .* DRFfull4D, dims=3) ./ depth4D  # barotropic rho (nx,ny,1,nt)
rho_prime = (fr) #.- rho0)./rho0                             # baroclinic rho'(nx,ny,nz,nt)
rho_int  = rho_prime[2:nx-1, 2:ny-1, :, :]
d_int = d[nx-2, ny-2,:]
sc = z_int ./ d_int
Cz       = -rho_int .* g .* Wz .*reshape(sc,  nx-2, ny-2, nz, 1)#./ reshape(d,  nx-2, ny-2, :, 1) # as scaling factor added -ve remove from front
size(Cz)
println(rho_prime[1:10])
println(fr[1:10])

DRFint   = DRFfull[2:nx-1, 2:ny-1, :]
DRFint4d = reshape(DRFint, nx-2, ny-2, nz, 1)


Ca_full = dropdims(sum(Cz .* DRFint4d, dims=3), dims=3)


# --- Time mean ---
ca = dropdims(mean(Ca_full; dims=3); dims=3)   # (nx-2, ny-2)


# --- Save ---
open(joinpath(base2, "Conv_z_dI", "Conv_z_$suffix2.bin"), "w") do io
    write(io, Float32.(ca))
end
println("Saved: Conv_z_$suffix2.bin")
# Read conversion field
            C = Float64.(open(joinpath(base2, "Conv", "Conv_$suffix2.bin"), "r") do io
                nbytes = (nx-2) * (ny-2) * sizeof(Float32)
                raw_bytes = read(io, nbytes)
                raw_data = reinterpret(Float32, raw_bytes)
                reshape(raw_data, nx-2, ny-2)
            end)


# ============================================================
# PLOT
# ============================================================
fig = Figure(resolution=(700, 500))
ax  = Axis(fig[1,1],
    title  = "Conversion (time mean) - tile $suffix2",
    xlabel = "x index",
    ylabel = "y index")
hm  = heatmap!(ax, ca[1:end-1,1:end-1], colormap=Reverse(:RdBu), colorrange =(-0.050,0.050))
Colorbar(fig[1,2], hm, label="W/m²")
display(fig)

fig = Figure(resolution=(700, 500))
ax  = Axis(fig[1,1],
    title  = "Conversion (time mean) - tile $suffix2",
    xlabel = "x index",
    ylabel = "y index")
hm  = heatmap!(ax, C, colormap=Reverse(:RdBu), colorrange =(-0.050,0.050))
Colorbar(fig[1,2], hm, label="W/m²")
display(fig)

fig2 = Figure(resolution=(400, 600))
ax  = Axis(fig2[1,1],
    xlabel = "x index",
    ylabel = "y index")
#=lines!(ax, fr[3,4,50:end,10], z[3,4,50:end], 
    label="conversion",
    color=:red,
    linewidth=2.5)=#
    lines!(ax, Cz[2,3,:,10], z_int[2,3,:], 
    label="conversion",
    color=:blue,
    linewidth=2.5)
    axislegend(ax; position=:rt)
display(fig2)



