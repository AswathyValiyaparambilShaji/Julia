
using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
include(joinpath(@__DIR__, "..","..","..", "functions", "butter_filters.jl"))
using .FluxUtils: read_bin, bandpassfilter


config_file = get(ENV, "JULIA_CONFIG",joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
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


rho0 = 999.8
mkpath(joinpath(base, "Figures"))


# Initialize full-domain mean EKE array (surface only, time-averaged)
EKE_mean = zeros(Float64, NX, NY)


for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]


        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
        println("Processing tile: $suffix")


        # --- Read bandpassed velocity fields ---
        U_lp = Float64.(open(joinpath(base2, "UVW_LP", "u_lp_$suffix.bin"), "r") do io
            nbytes = nx * ny * nz * nt * sizeof(Float32)
            raw_bytes = read(io, nbytes)
            raw_data  = reinterpret(Float32, raw_bytes)
            reshape(raw_data, nx, ny, nz, nt)
        end)


        V_lp = Float64.(open(joinpath(base2, "UVW_LP", "v_lp_$suffix.bin"), "r") do io
            nbytes = nx * ny * nz * nt * sizeof(Float32)
            raw_bytes = read(io, nbytes)
            raw_data  = reinterpret(Float32, raw_bytes)
            reshape(raw_data, nx, ny, nz, nt)
        end)


        # --- EKE at surface (kz=1), then time-mean ---
        eke = 0.5 .* rho0 .* (U_lp[:, :, 1, :].^2 .+ V_lp[:, :, 1, :].^2)  # (nx, ny, nt)
        eke_mean = mean(eke, dims=3)[:, :, 1]                                 # (nx, ny)


        # --- Tile index mapping ---
        xs = (xn - 1) * tx + 1
        xe = xs + tx + (2 * buf) - 1


        ys = (yn - 1) * ty + 1
        ye = ys + ty + (2 * buf) - 1


        xsf = 2;        xef = tx + (2*buf) - 1
        ysf = 2;        yef = ty + (2*buf) - 1


        EKE_mean[xs+1:xe-1, ys+1:ye-1] .= eke_mean[xsf:xef, ysf:yef]


    end
end


# --- Plot mean EKE ---
using CairoMakie


plot_dir = joinpath(base, "Figures")


EKE_max = maximum(EKE_mean[EKE_mean .> 0])


fig = Figure(resolution=(700, 600))
ax  = Axis(fig[1, 1],
    xlabel  = "Longitude (°E)",
    ylabel  = "Latitude (°N)",
    title   = "Time-Mean Surface EKE",
)
    #aspect  = DataAspect())


hm = CairoMakie.heatmap!(ax, lon, lat, EKE_mean,
    colormap   = :jet,
    colorrange = (0, EKE_max))


Colorbar(fig[1, 2], hm, label = "EKE (J/m³)")

display(fig)
save(joinpath(plot_dir, "EKE_mean_surface.png"), fig)
println("Saved: $(joinpath(plot_dir, "EKE_mean_surface.png"))")


# --- Movie code (commented out) ---
# EKE_surface = EKE[:, :, 1, :]
# EKE_max     = maximum(EKE_surface[EKE_surface .> 0])
# for t in 1:nt
#     fig = Figure(size=(1000, 800))
#     ax  = Axis(fig[1, 1], xlabel="Longitude (°E)", ylabel="Latitude (°N)",
#                title=@sprintf("Surface EKE - Hour %04d", t), aspect=DataAspect())
#     hm  = CairoMakie.heatmap!(ax, lon, lat, EKE_surface[:, :, t],
#                colormap=:thermal, colorrange=(0, EKE_max))
#     Colorbar(fig[1, 2], hm, label="EKE (J/m³)")
#     save(joinpath(plot_dir, @sprintf("frame_%04d.png", t)), fig)
#     if t % 100 == 0; println("  Completed $t / $nt"); end
# end
# println("ffmpeg -framerate 10 -i $(plot_dir)/frame_%04d.png -c:v libx264 -pix_fmt yuv420p $(joinpath(base, "EKE_surface_movie.mp4"))")




