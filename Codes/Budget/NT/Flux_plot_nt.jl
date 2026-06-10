using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML


include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base  = cfg["base_path"]
base2 = cfg["base_path_nt"]


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


# --- Thickness ---
thk   = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF   = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)


# ============================================================================
# ASSEMBLE FULL DOMAIN FROM TILES
# Read 4D time series (nx, ny, nz, nt), time-average over dim=4, depth-integrate
# ============================================================================
println("Reading time series flux files and computing time + depth averages...")


tfx = zeros(NX, NY)
tfy = zeros(NX, NY)
FH = fill(NaN, NX, NY)


for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]


        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
        hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))


        # Read 4D time series (nx, ny, nz, nt) — written as Float32
        fx = Float64.(open(joinpath(base2, "xflux", "xflx_$suffix.bin"), "r") do io
            raw_bytes = read(io, nx * ny * nz * nt * sizeof(Float32))
            reshape(reinterpret(Float32, raw_bytes), nx, ny, nz, nt)
        end)


        fy = Float64.(open(joinpath(base2, "yflux", "yflx_$suffix.bin"), "r") do io
            raw_bytes = read(io, nx * ny * nz * nt * sizeof(Float32))
            reshape(reinterpret(Float32, raw_bytes), nx, ny, nz, nt)
        end)
    # --- depth from hFacC ---
        hFacC   = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
        DRFfull = hFacC .* DRF3d
        DRFfull[hFacC .== 0] .= 0.0
        depth   = dropdims(sum(DRFfull, dims=3), dims=3)
        DRFfull[hFacC .== 0] .= 0.0
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
        FH[xs+2:xe-2, ys+2:ye-2]  .= depth[buf:nx-buf+1, buf:ny-buf+1]

        fx = fy = fx_tmean = fy_tmean = nothing; GC.gc()
        println("Completed tile: $suffix")
    end
end


# ============================================================================
# PLOTTING  (identical method to reference code)
# ============================================================================
using CairoMakie


FIGDIR        = cfg["fig_base"]
HEAT_CBAR_MAX = 15
QUIVER_STEP   = 20
ARROW_SCALEUP = 5.0
DO_TRANSPOSE  = false

FONT = "FreeSerif Bold"

mkpath(FIGDIR)


println("Creating flux map (full time average)...")


fm    = sqrt.(tfx.^2 .+ tfy.^2)
fm_kW = fm ./ 1000
F  = DO_TRANSPOSE ? fm_kW' : fm_kW
Ux = DO_TRANSPOSE ? tfx'   : tfx
Uy = DO_TRANSPOSE ? tfy'   : tfy


fig = Figure(resolution = (350, 450), figure_padding =(5,5,5,5),
             fonts=(; regular=FONT))
ax  = Axis(fig[1, 1],
    title      = "MITgcm  Flux ",
    xlabel     = "Longitude [°]",
    ylabel     = "Latitude [°]",
    ylabelsize = 16,
    xlabelsize = 16,
    xticklabelsize    = 12,
    yticklabelsize    = 12,
    titlesize  = 18,
    titlefont         = FONT,
    xlabelfont        = FONT,
    ylabelfont        = FONT,
    xticklabelfont    = FONT,
    yticklabelfont    = FONT,
    )


hm = CairoMakie.heatmap!(ax, lon, lat, F,
    interpolate = false,
    colorrange  = (0, HEAT_CBAR_MAX),
    colormap    = cgrad(:Spectral_9,rev = true))


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

contour!(ax, lon, lat, FH;
    levels     = [500, 1500, 3000.0],
    color      = RGBf(0.25,0.25,0.25),
    linewidth  = 2,
    linestyle  = :solid,
    labels     = true,
    labelsize  = 10,
    labelfont = "FreeSerif",
    labelcolor = RGBf(0.25,0.25,0.25))
Colorbar(fig[1, 2], hm, label = "(kW/m)", labelsize = 14, ticklabelsize=12 , width = 5)

colgap!(fig.layout,1,5)
png_file = joinpath(FIGDIR, "Flux_perturbation_timemean_V4.png")
save(png_file, fig)
display(fig)
println("PNG saved: $png_file")




