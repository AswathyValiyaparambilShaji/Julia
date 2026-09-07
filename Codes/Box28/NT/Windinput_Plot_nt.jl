using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML
using CairoMakie, SparseArrays


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base = cfg["bp_box28"]
base2 = (joinpath(base, "NT"))       


# --- Domain & grid of 27b ---
NX, NY = 384, 336
minlat, maxlat = -24.5, -18.5
minlon, maxlon = 337.5, 345.4791122715405
lat = range(minlat, maxlat, length=NY)
lon = range(minlon, maxlon, length=NX)
NZ = 173

# --- Tile & time ---
buf = 3
tx, ty = 54, 66
nx = tx + 2*buf
ny = ty + 2*buf
nz = 168
kz = 1
nt = 558

# --- Thickness & constants ---
thk =(open(joinpath(base, "hFacC",  "delR.bin"), "r") do io
                raw = read(io,  NZ * sizeof(Float32))
                ntoh.(reshape(reinterpret(Float32, raw), NZ))
            end)

DRF  = thk[1:nz]
sum(thk)
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)
g = 9.81

# Output directory where WPI tiles are saved
INDIR = joinpath(base2, "WindInput")


# Initialize full global array (with time dimension)
WPI_full = fill(NaN, NX, NY, nt)


# ==========================================================
# ============ BUILD WPI MAP FROM TILES ====================
# ==========================================================


Threads.@threads for xn in cfg["xn_start"]:cfg["xn_e28"]
    for yn in cfg["yn_start"]:cfg["yn_e28"]



        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)


        # Read WPI tile (full time series: nx × ny × nt)
        wpi_tile = Float64.(open(joinpath(INDIR, "wpi_nt_$suffix.bin"), "r") do io
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

wpi_absmax = max(abs(minimum(filter(isfinite, WPI_mean))),
                 abs(maximum(filter(isfinite, WPI_mean))))


fig = Figure(size=(600, 700))


ax = Axis(fig[1, 1],
    title = "Time-Averaged Wind Input ",
    xlabel = "Longitude [°]",
    ylabel = "Latitude [°]")


hm = CairoMakie.heatmap!(ax, lon, lat, WPI_mean.*1000;
    interpolate = false,
    colormap    =:bwr,
    colorrange  = (-0.05, 0.05))


Colorbar(fig[1, 2], hm, label = "Wind Input [mW/m²]")


display(fig)


# Save figure
FIGDIR = cfg["fig_base_28"]
mkpath(FIGDIR)
save(joinpath(FIGDIR, "Windinput_NS_nt_V2.png"), fig)


println("\nFigure saved to: $(joinpath(FIGDIR, "Windinput_NS_nt_V2.png"))")
println("\nDone!")




