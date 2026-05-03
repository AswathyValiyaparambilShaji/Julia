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


FDiv_z = zeros(NX, NY)


for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix2 = @sprintf("%02dx%02d_%d", xn, yn, buf-2)
        fpath   = joinpath(base2, "FDiv", "FDiv_$suffix2.bin")


        D = Float64.(open(fpath, "r") do io
            reshape(reinterpret(Float32, read(io, (nx-2)*(ny-2)*sizeof(Float32))), nx-2, ny-2)
        end)


        xs = (xn-1)*tx + 1
        xe = xs + tx + (2*buf) - 1
        ys = (yn-1)*ty + 1
        ye = ys + ty + (2*buf) - 1


        FDiv_z[xs+2:xe-2, ys+2:ye-2] .= D[2:end-1, 2:end-1]
    end
end


fig = Figure(size=(700, 500))
ax  = Axis(fig[1,1],
    title  = "Flux Divergence ∇·F  (W/m²)",
    xlabel = "Longitude [°]",
    ylabel = "Latitude [°]")
ax.limits[] = ((minimum(lon), maximum(lon)), (minimum(lat), maximum(lat)))


hm = CairoMakie.heatmap!(ax, lon, lat, FDiv_z;
    interpolate = false,
    colorrange  = (-0.050, 0.050),
    colormap    = Reverse(:RdBu))


Colorbar(fig[1,2], hm, label="W/m²")
display(fig)


FIGDIR = cfg["fig_base"]
mkpath(FIGDIR)
save(joinpath(FIGDIR, "FDiv_map_nt_V1.png"), fig)
println("Figure saved: $(joinpath(FIGDIR, "FDiv_map_nt_v1.png"))")




