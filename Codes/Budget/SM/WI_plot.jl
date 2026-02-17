using Printf, FilePathsBase, Statistics, TOML
using CairoMakie


include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin


config_file = get(ENV, "JULIA_CONFIG",
              joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)


base  = cfg["base_path"]
base2 = cfg["base_path2"]


# --- Grid parameters ---
NX, NY = 288, 468
minlat, maxlat = 24.0, 31.91
minlon, maxlon = 193.0, 199.0
lat = range(minlat, maxlat, length=NY)
lon = range(minlon, maxlon, length=NX)


# Tiling parameters
buf = 3
tx, ty = 47, 66
nx = tx + 2 * buf
ny = ty + 2 * buf
dt = 25
dto = 144
Tts = 366192
nt = div(Tts, dto)


# Output directory where WPI tiles are saved
INDIR = joinpath(base2, "WindPowerInput")


# Initialize full global array (with time dimension)
WPI_full = fill(NaN, NX, NY, nt)


# ==========================================================
# ============ BUILD WPI MAP FROM TILES ====================
# ==========================================================


for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]


        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)


        # Read WPI tile (full time series: nx × ny × nt)
        wpi_tile = Float64.(open(joinpath(INDIR, "wpi_$suffix.bin"), "r") do io
            nbytes = nx * ny * nt * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt)
        end)


        # Extract interior (remove buffer)
        wpi_interior = wpi_tile[buf+1:nx-buf, buf+1:ny-buf, :]


        # Calculate tile position in global grid
        xs = (xn - 1) * tx + 1
        xe = xs + tx - 1
        ys = (yn - 1) * ty + 1
        ye = ys + ty - 1


        WPI_full[xs:xe, ys:ye, :] .= wpi_interior


        println("Completed tile $suffix")
    end
end


# ==========================================================
# ============ TIME AVERAGE OVER FULL GLOBAL ARRAY =========
# ==========================================================


println("\nCalculating time mean over full domain...")
WPI_mean = mean(WPI_full, dims=3)[:, :, 1]


println("WPI_mean range: $(minimum(filter(isfinite, WPI_mean))) to $(maximum(filter(isfinite, WPI_mean)))")


# ==========================================================
# =================== VISUALIZATION ========================
# ==========================================================


# Symmetric colorrange around zero
wpi_absmax = max(abs(minimum(filter(isfinite, WPI_mean))),
                 abs(maximum(filter(isfinite, WPI_mean))))


fig = Figure(size=(1000, 800))


ax = Axis(fig[1, 1],
    title = "Time-Averaged Wind Power Input (9-15 hr filtered)",
    xlabel = "Longitude [°]",
    ylabel = "Latitude [°]")


hm = CairoMakie.heatmap!(ax, lon, lat, WPI_mean;
    interpolate = false,
    colormap    = Reverse(:RdBu),
    colorrange  = (-wpi_absmax, wpi_absmax))


Colorbar(fig[1, 2], hm, label = "Wind Power Input [W/m²]")


display(fig)


# Save figure
FIGDIR = cfg["fig_base"]
mkpath(FIGDIR)
save(joinpath(FIGDIR, "WindPowerInput_9-15hr_filtered.png"), fig)


println("\nFigure saved to: $(joinpath(FIGDIR, "WindPowerInput_9-15hr_filtered.png"))")
println("\nDone!")




