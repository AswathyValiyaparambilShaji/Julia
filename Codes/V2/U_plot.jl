using MAT, Printf, TOML, CairoMakie


include(joinpath(@__DIR__,  "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin


config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "config", "run_debug.toml"))
cfg     = TOML.parsefile(config_file)
#base    = cfg["base_path"]    # V1 base (for hFacC/bathymetry)
base = cfg["base_path_V2"] # V2 tile output root


# ── Grid ───────────────────────────────────────────────────────────────────────
NX, NY = 288, 468
minlat, maxlat = 24.0, 31.91
minlon, maxlon = 193.0, 199.0
lat = range(minlat, maxlat, length=NY)
lon = range(minlon, maxlon, length=NX)


# ── Tiling parameters ──────────────────────────────────────────────────────────
buf    = 3
tx, ty = 47, 66
nx = tx + 2*buf   # 53
ny = ty + 2*buf   # 72
nz  = 168
nt     = 558


# ── Bathymetry from V1 hFacC (same grid) ──────────────────────────────────────
#thk   = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
#DRF   = thk[1:nz]
#DRF3d = repeat(reshape(DRF, 1, 1, nz), nx_buf, ny_buf, 1)


# ── Output arrays ─────────────────────────────────────────────────────────────
U_mean = fill(NaN, NX, NY)




for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]


        suffix  = @sprintf("%02dx%02d_%d", xn, yn, buf)

        fpath = joinpath(base, "U", "U_v2_$suffix.bin")
        U = Float64.(open(fpath, "r") do io
            reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny,nz,nt)
        end)

        Us = U[:,:,1,:]
        Us_m = mean(Us,dims=3)

        # --- tile positions (unchanged) ---
        xn = 1
        yn=1
        xs = (xn-1)*tx + 1
        xe = xs + tx + (2*buf) - 1
        ys = (yn-1)*ty + 1
        ye = ys + ty + (2*buf) - 1
        


        # ── Assemble into global arrays (interior only, strip buffer) ─────────
        U_mean[xs+2:xe-2, ys+2:ye-2] .= Us_m[buf:nx-buf+1, buf:ny-buf+1]


    end
end


# ── Plot ───────────────────────────────────────────────────────────────────────
fig = Figure(resolution=(600, 800))
ax  = Axis(fig[1,1],
    title  = "Time-mean Surface U  (m/s)",
    xlabel = "Longitude [°]",
    ylabel = "Latitude [°]")
ax.limits[] = ((minimum(lon), maximum(lon)), (minimum(lat), maximum(lat)))


hm = CairoMakie.heatmap!(ax, lon, lat, U_mean;
    interpolate = false,
    colorrange  = (-0.5, 0.5),
    colormap    = :balance)





Colorbar(fig[1,2], hm, label="m/s")


FIGDIR = cfg["fig_base"]
mkpath(FIGDIR)
save(joinpath(FIGDIR, "U_mean_surface_v2.png"), fig)
println("Figure saved: $(joinpath(FIGDIR, "U_mean_surface_v2.png"))")
display(fig)




