using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML, CairoMakie


include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin


config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg   = TOML.parsefile(config_file)
base  = cfg["base_path"]
base2 = cfg["base_path2"]


NX, NY = 288, 468
minlat, maxlat = 24.0, 31.91
minlon, maxlon = 193.0, 199.0
lat = range(minlat, maxlat, length=NY)
lon = range(minlon, maxlon, length=NX)


buf    = 3
tx, ty = 47, 66
nx, ny = tx + 2*buf, ty + 2*buf
nz     = 88


thk   = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF   = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)


FIGDIR        = cfg["fig_base"]
HEAT_CBAR_MAX = 15
ARROW_SCALEUP = 5.0
DO_TRANSPOSE  = false


# 3 columns × 2 rows of arrows per tile; skip first column
QSTEP_X = div(tx, 3)    # ≈ 16
QSTEP_Y = div(ty, 2)    # = 33


mkpath(FIGDIR)


# ── Assemble both flux fields from tiles ─────────────────────────────────────
tfx_orig = zeros(NX, NY)
tfy_orig = zeros(NX, NY)
tfx_corr = zeros(NX, NY)
tfy_corr = zeros(NX, NY)


for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]


        suffix  = @sprintf("%02dx%02d_%d", xn, yn, buf)
        hFacC   = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
        DRFfull = hFacC .* DRF3d


        fx_o = Float64.(open(joinpath(base2, "xflux", "xflx_$suffix.bin"), "r") do io
            raw = read(io, nx*ny*nz*sizeof(Float32))
            reshape(reinterpret(Float32, raw), nx, ny, nz)
        end)
        fy_o = Float64.(open(joinpath(base2, "yflux", "yflx_$suffix.bin"), "r") do io
            raw = read(io, nx*ny*nz*sizeof(Float32))
            reshape(reinterpret(Float32, raw), nx, ny, nz)
        end)


        fx_c = Float64.(open(joinpath(base2, "xflux_corr", "xflx_$suffix.bin"), "r") do io
            raw = read(io, nx*ny*nz*sizeof(Float32))
            reshape(reinterpret(Float32, raw), nx, ny, nz)
        end)
        fy_c = Float64.(open(joinpath(base2, "yflux_corr", "yflx_$suffix.bin"), "r") do io
            raw = read(io, nx*ny*nz*sizeof(Float32))
            reshape(reinterpret(Float32, raw), nx, ny, nz)
        end)


        fxX_o = sum(fx_o .* DRFfull, dims=3)
        fyY_o = sum(fy_o .* DRFfull, dims=3)
        fxX_c = sum(fx_c .* DRFfull, dims=3)
        fyY_c = sum(fy_c .* DRFfull, dims=3)


        xs  = (xn-1)*tx + 1;  xe  = xs + tx - 1
        ys  = (yn-1)*ty + 1;  ye  = ys + ty - 1
        xsf = buf+1;           xef = buf+tx
        ysf = buf+1;           yef = buf+ty


        tfx_orig[xs:xe, ys:ye] .= fxX_o[xsf:xef, ysf:yef, 1]
        tfy_orig[xs:xe, ys:ye] .= fyY_o[xsf:xef, ysf:yef, 1]
        tfx_corr[xs:xe, ys:ye] .= fxX_c[xsf:xef, ysf:yef, 1]
        tfy_corr[xs:xe, ys:ye] .= fyY_c[xsf:xef, ysf:yef, 1]


    end
end


mag_orig = sqrt.(tfx_orig.^2 .+ tfy_orig.^2) ./ 1000
mag_corr = sqrt.(tfx_corr.^2 .+ tfy_corr.^2) ./ 1000
mag_diff = mag_corr .- mag_orig


println("Original  — min/max: $(extrema(mag_orig))")
println("Corrected — min/max: $(extrema(mag_corr))")
println("Difference— min/max: $(extrema(mag_diff))")


# ── Plot ──────────────────────────────────────────────────────────────────────
fig = Figure(resolution = (1800, 600))


# ── Panel 1: original ─────────────────────────────────────────────────────────
local F, Ux, Uy, pos, arrvec, NX_local, NY_local, maxmag, cell_x, cell_y, target, scale


F  = DO_TRANSPOSE ? mag_orig' : mag_orig
Ux = DO_TRANSPOSE ? tfx_orig' : tfx_orig
Uy = DO_TRANSPOSE ? tfy_orig' : tfy_orig


ax1 = Axis(fig[1,1],
    title      = "Original flux (no heaving correction)",
    xlabel     = "Longitude [°]", ylabel = "Latitude [°]",
    xlabelsize = 22, ylabelsize = 22, titlesize = 20)
hm1 = CairoMakie.heatmap!(ax1, lon, lat, F,
    interpolate = false, colorrange = (0, HEAT_CBAR_MAX), colormap = :Spectral_9)


pos    = Point2f[]
arrvec = Vec2f[]
NX_local, NY_local = size(F)
for i in (buf + 1 + QSTEP_X) : QSTEP_X : (NX_local - buf)
    for j in (buf + 1) : QSTEP_Y : (NY_local - buf)
        u = Ux[i, j]; v = Uy[i, j]; m = F[i, j]
        if isfinite(u) && isfinite(v) && isfinite(m) && m > 0.0
            push!(pos,    Point2f(Float32(lon[i]), Float32(lat[j])))
            push!(arrvec, Vec2f(Float32(u), Float32(v)))
        end
    end
end
if !isempty(arrvec)
    maxmag = maximum(norm, arrvec)
    cell_x = (maximum(lon) - minimum(lon)) / NX_local
    cell_y = (maximum(lat) - minimum(lat)) / NY_local
    target = 5f0 * Float32(min(cell_x, cell_y))
    scale  = maxmag == 0 ? 1f0 : (target / maxmag) * Float32(ARROW_SCALEUP)
    arrows!(ax1, pos, scale .* arrvec, color=:black, arrowsize=8, linewidth=1.5)
end
Colorbar(fig[1,2], hm1, label="(kW/m)")


# ── Panel 2: corrected ────────────────────────────────────────────────────────
F  = DO_TRANSPOSE ? mag_corr' : mag_corr
Ux = DO_TRANSPOSE ? tfx_corr' : tfx_corr
Uy = DO_TRANSPOSE ? tfy_corr' : tfy_corr


ax2 = Axis(fig[1,3],
    title      = "Corrected flux (with heaving correction)",
    xlabel     = "Longitude [°]", ylabel = "Latitude [°]",
    xlabelsize = 22, ylabelsize = 22, titlesize = 20)
hm2 = CairoMakie.heatmap!(ax2, lon, lat, F,
    interpolate = false, colorrange = (0, HEAT_CBAR_MAX), colormap = :Spectral_9)


pos    = Point2f[]
arrvec = Vec2f[]
NX_local, NY_local = size(F)
for i in (buf + 1 + QSTEP_X) : QSTEP_X : (NX_local - buf)
    for j in (buf + 1) : QSTEP_Y : (NY_local - buf)
        u = Ux[i, j]; v = Uy[i, j]; m = F[i, j]
        if isfinite(u) && isfinite(v) && isfinite(m) && m > 0.0
            push!(pos,    Point2f(Float32(lon[i]), Float32(lat[j])))
            push!(arrvec, Vec2f(Float32(u), Float32(v)))
        end
    end
end
if !isempty(arrvec)
    maxmag = maximum(norm, arrvec)
    cell_x = (maximum(lon) - minimum(lon)) / NX_local
    cell_y = (maximum(lat) - minimum(lat)) / NY_local
    target = 5f0 * Float32(min(cell_x, cell_y))
    scale  = maxmag == 0 ? 1f0 : (target / maxmag) * Float32(ARROW_SCALEUP)
    arrows!(ax2, pos, scale .* arrvec, color=:black, arrowsize=8, linewidth=1.5)
end
Colorbar(fig[1,4], hm2, label="(kW/m)")


# ── Panel 3: difference ───────────────────────────────────────────────────────
diff_max = maximum(abs.(mag_diff))
F = DO_TRANSPOSE ? mag_diff' : mag_diff


ax3 = Axis(fig[1,5],
    title      = "Difference (corrected − original)",
    xlabel     = "Longitude [°]", ylabel = "Latitude [°]",
    xlabelsize = 22, ylabelsize = 22, titlesize = 20)
hm3 = CairoMakie.heatmap!(ax3, lon, lat, F,
    interpolate = false, colorrange = (-diff_max, diff_max), colormap = :bwr)
Colorbar(fig[1,6], hm3, label="(kW/m)")


png_file = joinpath(FIGDIR, "Flux_heaving_comparison.png")
save(png_file, fig)
display(fig)
println("Saved: $png_file")




