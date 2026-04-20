using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML, CairoMakie


include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin


config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg   = TOML.parsefile(config_file)
base  = cfg["base_path"]
base2 = cfg["base_path2"]


# --- Domain ---
NX, NY   = 288, 468
minlat, maxlat = 24.0, 31.91
minlon, maxlon = 193.0, 199.0
lat = range(minlat, maxlat, length=NY)
lon = range(minlon, maxlon, length=NX)


# --- Tile ---
buf    = 3
tx, ty = 47, 66
nx, ny = tx + 2*buf, ty + 2*buf
nz     = 88


# --- Plot settings ---
FIGDIR        = cfg["fig_base"]
HEAT_CBAR_MAX = 15
QUIVER_STEP   = 20
ARROW_SCALEUP = 5.0
mkpath(FIGDIR)


# ── Assemble full domain from tiles (use pre-integrated vint files) ───────────
# xflx_vint and yflx_vint are already depth-integrated: shape (nx,ny,1,1)
tfx = zeros(NX, NY)
tfy = zeros(NX, NY)


for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]


        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)


        # read depth-integrated corrected fluxes (nx,ny,1,1)
        fxvi = Float64.(open(joinpath(base2, "xflux_corr", "xflx_$suffix.bin"), "r") do io
            raw = read(io, nx*ny*sizeof(Float32))
            reshape(reinterpret(Float32, raw), nx, ny)
        end)


        fyvi = Float64.(open(joinpath(base2, "yflux_corr", "yflx_$suffix.bin"), "r") do io
            raw = read(io, nx*ny*sizeof(Float32))
            reshape(reinterpret(Float32, raw), nx, ny)
        end)


        # tile index bounds (strip buffer)
        xs = (xn-1)*tx + 1;  xe = xs + tx - 1
        ys = (yn-1)*ty + 1;  ye = ys + ty - 1
        xsf = buf+1;          xef = buf+tx
        ysf = buf+1;          yef = buf+ty


        tfx[xs:xe, ys:ye] .= fxvi[xsf:xef, ysf:yef]
        tfy[xs:xe, ys:ye] .= fyvi[xsf:xef, ysf:yef]


    end
end


# ── Plot ──────────────────────────────────────────────────────────────────────
fm    = sqrt.(tfx.^2 .+ tfy.^2)
fm_kW = fm ./ 1000                  # W/m → kW/m


fig = Figure(resolution = (700, 600))
ax  = Axis(fig[1, 1],
    title      = "Corrected Internal Tide Flux (pint = p′ − pη)",
    xlabel     = "Longitude [°]",
    ylabel     = "Latitude [°]",
    xlabelsize = 22,
    ylabelsize = 22,
    titlesize  = 22)


hm = CairoMakie.heatmap!(ax, lon, lat, fm_kW,
    interpolate = false,
    colorrange  = (0, HEAT_CBAR_MAX),
    colormap    = :Spectral_9)


# quiver arrows
pos    = Point2f[]
arrvec = Vec2f[]
for i in 1:QUIVER_STEP:NX, j in 1:QUIVER_STEP:NY
    u = tfx[i, j]; v = tfy[i, j]; m = fm_kW[i, j]
    if isfinite(u) && isfinite(v) && isfinite(m)
        push!(pos,    Point2f(Float32(lon[i]), Float32(lat[j])))
        push!(arrvec, Vec2f(Float32(u), Float32(v)))
    end
end


if !isempty(arrvec)
    maxmag = maximum(norm, arrvec)
    cell_x = (maxlon - minlon) / NX
    cell_y = (maxlat - minlat) / NY
    target = 5f0 * Float32(min(cell_x, cell_y))
    scale  = maxmag == 0 ? 1f0 : (target / maxmag) * Float32(ARROW_SCALEUP)
    arrows!(ax, pos, scale .* arrvec, color=:black, arrowsize=8, linewidth=1.5)
end


Colorbar(fig[1, 2], hm, label = "(kW/m)")


png_file = joinpath(FIGDIR, "Flux_corr_full.png")
save(png_file, fig)
display(fig)
println("Saved: $png_file")




