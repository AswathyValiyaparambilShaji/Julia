using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML, CairoMakie


include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter


# Load configuration
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg   = TOML.parsefile(config_file)
base  = cfg["base_path"]
base2 = cfg["base_path2"]


# --- Domain & grid ---
NX, NY = 288, 468
minlat, maxlat = 24.0, 31.91
minlon, maxlon = 193.0, 199.0
lat = range(minlat, maxlat, length=NY)
lon = range(minlon, maxlon, length=NX)


# --- Tile ---
buf    = 3
tx, ty = 47, 66
nx     = tx + 2 * buf
ny     = ty + 2 * buf
nz     = 88


# --- load DRF (needed for depth) ---
thk   = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF   = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)


# ============================================================================
# ASSEMBLE GLOBAL MAP FROM TILES
# ============================================================================


Conv_z = zeros(NX, NY)
FH     = fill(NaN, NX, NY)


for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]


        suffix  = @sprintf("%02dx%02d_%d", xn, yn, buf)
        suffix2 = @sprintf("%02dx%02d_%d", xn, yn, buf - 2)
        fpath   = joinpath(base2, "Conv", "Conv_$suffix2.bin")


        # Read tile (nx-2) x (ny-2) Float32
        C = Float64.(open(fpath, "r") do io
            nbytes   = (nx - 2) * (ny - 2) * sizeof(Float32)
            raw_data = reinterpret(Float32, read(io, nbytes))
            reshape(raw_data, nx - 2, ny - 2)
        end)


        # --- depth from hFacC ---
        hFacC   = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
        DRFfull = hFacC .* DRF3d
        DRFfull[hFacC .== 0] .= 0.0
        H = dropdims(sum(DRFfull, dims=3), dims=3)   # (nx, ny)


        # Tile position in global grid
        xs = (xn - 1) * tx + 1
        xe = xs + tx + (2 * buf) - 1
        ys = (yn - 1) * ty + 1
        ye = ys + ty + (2 * buf) - 1


        # FIX: strip 2 cells on each side (not 1) so exactly tx×ty cells are
        # placed per tile with no overlap → eliminates edge-artifact seams.
        # C has size (nx-2)×(ny-2) = 51×70; C[3:end-2, 3:end-2] = 47×66 = tx×ty ✓
        Conv_z[xs+3:xe-3, ys+3:ye-3] .= C[3:end-2, 3:end-2]


        # depth: H is (nx, ny) = 53×72; interior buf:nx-buf+1 = 3:51 = 49 cells,
        # shift by 1 to match the extra strip → buf+1:nx-buf = 4:50 = 47 = tx ✓
        FH[xs+3:xe-3, ys+3:ye-3] .= H[buf+1:nx-buf, buf+1:ny-buf]


        println("Completed tile $suffix")
    end
end


# ============================================================================
# PLOT
# ============================================================================


fig = Figure(resolution = (600, 800))
#println(Conv_z[10, 10])


ax = Axis(fig[1, 1],
    title   = "Vertical Conversion  (W/m²)",
    xlabel  = "Longitude [°]",
    ylabel  = "Latitude [°]")
ax.limits[] = ((minimum(lon), maximum(lon)), (minimum(lat), maximum(lat)))


hm = CairoMakie.heatmap!(ax, lon, lat, Conv_z;
    interpolate = false,
    colorrange  = (-0.030, 0.030),
    colormap    = :bwr)


# --- bathymetry contours ---
contour!(ax, lon, lat, FH;
    levels     = [100.0, 500.0, 1000.0, 3000.0],
    color      = :black,
    linewidth  = 2,
    linestyle  = :solid,
    labels     = true,
    labelsize  = 18,
    labelcolor = :black)


Colorbar(fig[1, 2], hm, label = "W/m²")


display(fig)


FIGDIR = cfg["fig_base"]
mkpath(FIGDIR)
save(joinpath(FIGDIR, "ConvZ_map_v1_cpo.png"), fig)
println("Figure saved: $(joinpath(FIGDIR, "ConvZ_map_v1_cpo.png"))")




