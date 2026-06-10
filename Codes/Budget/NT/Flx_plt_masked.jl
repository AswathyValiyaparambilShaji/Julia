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


# --- Depth threshold ---
DEPTH_THRESHOLD = 3000.0


# ============================================================================
# ASSEMBLE FULL DOMAIN FROM TILES
# ============================================================================
println("Reading time series flux files and computing time + depth averages...")


tfx = zeros(NX, NY)
tfy = zeros(NX, NY)
FH  = zeros(NX, NY)   # full water column depth
RAC = zeros(NX, NY)   # cell area


for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]


        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
        hFacC  = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
        dx     = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
        dy     = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))


        DRFfull = hFacC .* DRF3d
        depth   = dropdims(sum(DRFfull, dims=3), dims=3)   # (nx, ny) total water depth
        DRFfull[hFacC .== 0] .= 0.0
        rac = dx .* dy


        # Read 4D time series (nx, ny, nz, nt)
        fx = Float64.(open(joinpath(base2, "xflux", "xflx_$suffix.bin"), "r") do io
            raw_bytes = read(io, nx * ny * nz * nt * sizeof(Float32))
            reshape(reinterpret(Float32, raw_bytes), nx, ny, nz, nt)
        end)


        fy = Float64.(open(joinpath(base2, "yflux", "yflx_$suffix.bin"), "r") do io
            raw_bytes = read(io, nx * ny * nz * nt * sizeof(Float32))
            reshape(reinterpret(Float32, raw_bytes), nx, ny, nz, nt)
        end)


        # Time average over dim=4
        fx_tmean = mean(fx, dims=4)[:, :, :, 1]
        fy_tmean = mean(fy, dims=4)[:, :, :, 1]


        # Depth integrate
        fxX = sum(fx_tmean .* DRFfull, dims=3)
        fyY = sum(fy_tmean .* DRFfull, dims=3)


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
        FH[ xs+1:xe-1, ys+1:ye-1] .= depth[xsf:xef, ysf:yef]
        RAC[xs+1:xe-1, ys+1:ye-1] .= rac[xsf:xef, ysf:yef]


        fx = fy = fx_tmean = fy_tmean = nothing; GC.gc()
        println("Completed tile: $suffix")
    end
end


# ============================================================================
# DEPTH MASKS
# ============================================================================
valid_mask   = (RAC .> 0.0) .& (FH .> 0.0)
shallow_mask = valid_mask .& (FH .<  DEPTH_THRESHOLD)   # H < 3000 m
deep_mask    = valid_mask .& (FH .>= DEPTH_THRESHOLD)   # H >= 3000 m


# Apply masks: NaN out regions that don't belong to each subplot
tfx_shallow = copy(tfx); tfy_shallow = copy(tfy)
tfx_deep    = copy(tfx); tfy_deep    = copy(tfy)


tfx_shallow[.!shallow_mask] .= NaN
tfy_shallow[.!shallow_mask] .= NaN
tfx_deep[   .!deep_mask]    .= NaN
tfy_deep[   .!deep_mask]    .= NaN


# ============================================================================
# PLOTTING — 1 figure, 2 subplots (shallow top, deep bottom)
# ============================================================================
using CairoMakie


FIGDIR        = cfg["fig_base"]
HEAT_CBAR_MAX = 15
QUIVER_STEP   = 20
ARROW_SCALEUP = 5.0


mkpath(FIGDIR)
println("Creating 2-panel depth-masked flux map...")


function flux_magnitude_kW(fx, fy)
    fm = sqrt.(fx.^2 .+ fy.^2) ./ 1000   # W/m -> kW/m
    fm[isnan.(fx)] .= NaN
    return fm
end


panels = [
    (tfx_shallow, tfy_shallow, flux_magnitude_kW(tfx_shallow, tfy_shallow),
     "Shallow region  (H < $(Int(DEPTH_THRESHOLD)) m)"),
    (tfx_deep,    tfy_deep,    flux_magnitude_kW(tfx_deep,    tfy_deep),
     "Deep region  (H ≥ $(Int(DEPTH_THRESHOLD)) m)"),
]


fig = Figure(resolution = (450, 1000))


for (idx, (Ux, Uy, F, title_str)) in enumerate(panels)


    ax = Axis(fig[idx, 1],
        title      = title_str,
        xlabel     = "Longitude [°]",
        ylabel     = "Latitude [°]",
        ylabelsize = 16,
        xlabelsize = 16,
        titlesize  = 18)


    hm = CairoMakie.heatmap!(ax, lon, lat, F,
        interpolate = false,
        colorrange  = (0, HEAT_CBAR_MAX),
        colormap    = cgrad(:Spectral_9,rev = true),
        nan_color   = :lightgray)


    # --- Quiver arrows ---
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


    Colorbar(fig[idx, 2], hm, label = "(kW/m)")
end


png_file = joinpath(FIGDIR, "Flux_perturbation_timemean_depth_masked.png")
save(png_file, fig)
display(fig)
println("PNG saved: $png_file")




