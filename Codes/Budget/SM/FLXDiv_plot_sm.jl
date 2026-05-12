using MAT, Statistics, Printf, LinearAlgebra, TOML, CairoMakie


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
nx     = tx + 2*buf
ny     = ty + 2*buf
nz     = 88


# --- load DRF (needed for depth) ---
thk  = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF  = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)


FDiv_z = zeros(NX, NY)
FH     = zeros(NX, NY)


for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix  = @sprintf("%02dx%02d_%d", xn, yn, buf)
        suffix2 = @sprintf("%02dx%02d_%d", xn, yn, buf-2)


        # --- FDiv (unchanged) ---
        fpath = joinpath(base2, "FDiv", "FDiv_$suffix2.bin")
        D = Float64.(open(fpath, "r") do io
            reshape(reinterpret(Float32, read(io, (nx-2)*(ny-2)*sizeof(Float32))), nx-2, ny-2)
        end)


        # --- depth from hFacC ---
        hFacC   = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
        DRFfull = hFacC .* DRF3d
        DRFfull[hFacC .== 0] .= 0.0
        H = dropdims(sum(DRFfull, dims=3), dims=3)   # (nx, ny)


        # --- tile positions (unchanged) ---
        xs = (xn-1)*tx + 1
        xe = xs + tx + (2*buf) - 1
        ys = (yn-1)*ty + 1
        ye = ys + ty + (2*buf) - 1


        FDiv_z[xs+2:xe-2, ys+2:ye-2] .= D[2:end-1, 2:end-1]
        FH[xs+2:xe-2,     ys+2:ye-2] .= H[buf:nx-buf+1, buf:ny-buf+1]


        println("Completed tile $suffix")
    end
end


# ── figure (unchanged except for the contour! lines below) ─────────────────
fig = Figure(resolution=(600, 800))
ax  = Axis(fig[1,1],
    title  = "Flux Divergence ∇·F  (W/m²)",
    xlabel = "Longitude [°]",
    ylabel = "Latitude [°]")
ax.limits[] = ((minimum(lon), maximum(lon)), (minimum(lat), maximum(lat)))


hm = CairoMakie.heatmap!(ax, lon, lat, FDiv_z;
    interpolate = false,
    colorrange  = (-0.030, 0.030),
    colormap    = :bwr)


# --- bathymetry contours (added) ---
contour!(ax, lon, lat, FH;
    levels     = [500.0, 1000.0, 2000.0, 3000.0, 4000.0],
    color      = :black,
    linewidth  = 0.8,
    linestyle  = :solid)


Colorbar(fig[1,2], hm, label="W/m²")
display(fig)


FIGDIR = cfg["fig_base"]
mkpath(FIGDIR)
save(joinpath(FIGDIR, "FDiv_map_v1_cpo.png"), fig)
println("Figure saved: $(joinpath(FIGDIR, "FDiv_map_v1_cpo.png"))")




