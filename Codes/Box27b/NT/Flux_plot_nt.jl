using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML
using CairoMakie, SparseArrays


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base = cfg["bp_box27b"]
base2 = (joinpath(base, "NT"))       


# --- Domain & grid ---
NX, NY = 1056, 1026 
minlat, maxlat = -60.0, -48.0
minlon, maxlon = 142.0208530805687, 163.9791469194313
lat = range(minlat, maxlat, length=NY)
lon = range(minlon, maxlon, length=NX)
NZ = 173

# --- Tile & time ---
buf = 3
tx, ty = 150, 146
nx = tx + 2*buf
ny = ty + 2*buf
nz = 170
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

# ============================================================================
# ASSEMBLE FULL DOMAIN FROM TILES
# Read 4D time series (nx, ny, nz, nt), time-average over dim=4, depth-integrate
# ============================================================================
println("Reading time series flux files and computing time + depth averages...")


tfx = zeros(NX, NY)
tfy = zeros(NX, NY)


for xn in cfg["xn_start"]:cfg["xn_e27b"]
    for yn in cfg["yn_start"]:cfg["yn_e7b"]


        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
        hFacC = read_bin(joinpath(base, "hFacC/hFacC_v2_$suffix.bin"), (nx, ny, nz))


        # Read 4D time series (nx, ny, nz, nt) — written as Float32
        fx = Float64.(open(joinpath(base2, "xflux", "xflx_$suffix.bin"), "r") do io
            raw_bytes = read(io, nx * ny * nz * nt * sizeof(Float32))
            reshape(reinterpret(Float32, raw_bytes), nx, ny, nz, nt)
        end)


        fy = Float64.(open(joinpath(base2, "yflux", "yflx_$suffix.bin"), "r") do io
            raw_bytes = read(io, nx * ny * nz * nt * sizeof(Float32))
            reshape(reinterpret(Float32, raw_bytes), nx, ny, nz, nt)
        end)


        # Time average over dim=4
        fx_tmean = mean(fx, dims=4)[:, :, :, 1]   # (nx, ny, nz)
        fy_tmean = mean(fy, dims=4)[:, :, :, 1]   # (nx, ny, nz)


        # Depth integrate
        DRFfull = hFacC .* DRF3d
        fxX = sum(fx_tmean .* DRFfull, dims=3)    # (nx, ny, 1)
        fyY = sum(fy_tmean .* DRFfull, dims=3)    # (nx, ny, 1)


        # Tile placement (trim buffer)
        xs  = (xn - 1) * tx + 1
        xe  = xs + tx + (2 * buf) - 1
        ys  = (yn - 1) * ty + 1
        ye  = ys + ty + (2 * buf) - 1
        xsf = 2
        xef = tx + (2*buf) - 1
        ysf = 2
        yef = ty + (2*buf) - 1


        tfx[xs+1:xe-1, ys+1:ye-1] .= fxX[xsf:xef, ysf:yef, 1]
        tfy[xs+1:xe-1, ys+1:ye-1] .= fyY[xsf:xef, ysf:yef, 1]


        fx = fy = fx_tmean = fy_tmean = nothing; GC.gc()
        println("Completed tile: $suffix")
    end
end


# ============================================================================
# PLOTTING  (identical method to reference code)
# ============================================================================
using CairoMakie


FIGDIR        = cfg["fig_base_27b"]
HEAT_CBAR_MAX = 15
QUIVER_STEP   = 20
ARROW_SCALEUP = 5.0
DO_TRANSPOSE  = false


mkpath(FIGDIR)


println("Creating flux map (full time average)...")


fm    = sqrt.(tfx.^2 .+ tfy.^2)
fm_kW = fm ./ 1000
F  = DO_TRANSPOSE ? fm_kW' : fm_kW
Ux = DO_TRANSPOSE ? tfx'   : tfx
Uy = DO_TRANSPOSE ? tfy'   : tfy


fig = Figure(resolution = (1200, 500))
ax  = Axis(fig[1, 1],
    title      = "MITgcm Perturbation Flux",
    xlabel     = "Longitude [°]",
    ylabel     = "Latitude [°]",
    ylabelsize = 22,
    xlabelsize = 22,
    titlesize  = 26)


hm = CairoMakie.heatmap!(ax, lon, lat, F,
    interpolate = false,
    colorrange  = (0, HEAT_CBAR_MAX),
    colormap    = :Spectral_9)


pos    = Point2f[]
arrvec = Vec2f[]
NX_local, NY_local = size(F)
for i in 1:QUIVER_STEP:NX_local, j in 1:QUIVER_STEP:NY_local
    u = Ux[i, j]; v = Uy[i, j]; m = F[i, j]
    if isfinite(u) && isfinite(v) && isfinite(m)
        push!(pos,    Point2f(Float32(lon[i]), Float32(lat[j])))
        push!(arrvec, Vec2f(Float32(u), Float32(v)))
    end
end


if !isempty(arrvec)
    maxmag = maximum(norm, arrvec)
    cell_x = (maximum(lon) - minimum(lon)) / NX_local
    cell_y = (maximum(lat) - minimum(lat)) / NY_local
    target = 5f0 * Float32(min(cell_x, cell_y))
    scale  = maxmag == 0 ? 1f0 : (target / maxmag) * Float32(ARROW_SCALEUP)
    arrows!(ax, pos, scale .* arrvec, color=:black, arrowsize=8, linewidth=1.5)
end


Colorbar(fig[1, 2], hm, label = "(kW/m)")


png_file = joinpath(FIGDIR, "Flux_perturbation_NS_timemean_V1.png")
save(png_file, fig)
display(fig)
println("PNG saved: $png_file")




