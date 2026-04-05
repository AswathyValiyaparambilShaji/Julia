using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML, CairoMakie


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
include(joinpath(@__DIR__, "..","..","..", "functions", "butter_filters.jl"))
include(joinpath(@__DIR__, "..","..","..", "functions", "coriolis_frequency.jl"))
using .FluxUtils: read_bin, bandpassfilter


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


# --- Tile & time ---
buf = 3
tx, ty = 47, 66
nx = tx + 2*buf
ny = ty + 2*buf
nz = 88
dt  = 25
dto = 144
Tts = 366192
nt  = div(Tts, dto)


# reference density
rho0 = 999.8


# Snapshot time index to plot (change this to any timestep 1:nt)
t_snap = 1


# Accumulate surface vorticity snapshot for plotting
VT_global = zeros(Float64, NX, NY)


for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]


        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
        println("Processing tile: $suffix")


        # --- Read grid spacing ---
        dx = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
        dy = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))


        # --- Read full LP velocity fields (all nt timesteps) ---
        U = Float64.(open(joinpath(base2, "UVW_LP", "u_lp_$suffix.bin"), "r") do io
            nbytes = nx * ny * nz * nt * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nz, nt)
        end)


        V = Float64.(open(joinpath(base2, "UVW_LP", "v_lp_$suffix.bin"), "r") do io
            nbytes = nx * ny * nz * nt * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nz, nt)
        end)


        # --- Extract snapshot at t_snap (surface only) ---
        U_snap = U[:, :, 1, t_snap]   # (nx, ny)
        V_snap = V[:, :, 1, t_snap]   # (nx, ny)


        U = nothing; V = nothing; GC.gc()   # free full arrays immediately


        # --- Gradient arrays ---
        U_y = zeros(Float64, nx, ny)
        V_x = zeros(Float64, nx, ny)


        # X-gradient: ∂V/∂x (central difference)
        dx_avg = dx[2:end-1, :] .+ dx[1:end-2, :]
        V_x[2:end-1, :] = (V_snap[3:end, :] .- V_snap[1:end-2, :]) ./ dx_avg


        # Y-gradient: ∂U/∂y (central difference)
        dy_avg = dy[:, 2:end-1] .+ dy[:, 1:end-2]
        U_y[:, 2:end-1] = (U_snap[:, 3:end] .- U_snap[:, 1:end-2]) ./ dy_avg


        # --- Vorticity ---
        VT = V_x .- U_y   # (nx, ny)


        # --- Normalize by Coriolis frequency ---
        for j in 1:ny
            global_j = (yn - 1) * ty + j - buf
            if global_j >= 1 && global_j <= NY
                f_rad_s = coriolis_frequency(lat[global_j])
                VT[:, j] ./= f_rad_s
            end
        end


        # --- Tile index mapping ---
        xs = (xn - 1) * tx + 1
        xe = xs + tx + (2 * buf) - 1
        ys = (yn - 1) * ty + 1
        ye = ys + ty + (2 * buf) - 1


        xsf = 2;    xef = tx + (2*buf) - 1
        ysf = 2;    yef = ty + (2*buf) - 1


        VT_global[xs+1:xe-1, ys+1:ye-1] .= VT[xsf:xef, ysf:yef]


        println("Completed tile: $suffix")
    end
end


# --- Plot snapshot ---
plot_dir = joinpath(base, "Figures")
mkpath(plot_dir)


clim = maximum(abs.(VT_global[VT_global .!= 0]))


fig = Figure(size=(1000, 800))
ax  = Axis(fig[1, 1],
    xlabel  = "Longitude (°E)",
    ylabel  = "Latitude (°N)",
    title   = @sprintf("Surface Vorticity (ζ/f)"))


hm = CairoMakie.heatmap!(ax, lon, lat, VT_global,
    colormap   = :RdBu,
    colorrange = (-0.5, 0.5))


Colorbar(fig[1, 2], hm, label = "ζ/f")


display(fig)
save(joinpath(plot_dir, @sprintf("Vorticity_snap_%04d.png", t_snap)), fig)
println("Saved: $(joinpath(plot_dir, @sprintf("Vorticity_snap_%04d.png", t_snap)))")


# --- Movie code (commented out) ---
# for t in 1:nt
#     fig = Figure(size=(1000, 800))
#     ax  = Axis(fig[1, 1], xlabel="Longitude (°E)", ylabel="Latitude (°N)",
#                title=@sprintf("Surface Vorticity (ζ/f) - Hour %04d", t), aspect=DataAspect())
#     hm  = CairoMakie.heatmap!(ax, lon, lat, VT_global[:, :, t],
#                colormap=:RdBu, colorrange=(-clim, clim))
#     Colorbar(fig[1, 2], hm, label="ζ/f")
#     save(joinpath(plot_dir, @sprintf("frame_%04d.png", t)), fig)
#     if t % 100 == 0; println("  Completed $t / $nt"); end
# end
# println("ffmpeg -framerate 10 -i $(plot_dir)/frame_%04d.png -c:v libx264 -pix_fmt yuv420p $(joinpath(base, "Vorticity_movie.mp4"))")




