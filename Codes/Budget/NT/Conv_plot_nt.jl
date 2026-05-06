using MAT, Statistics, Printf, LinearAlgebra, TOML, CairoMakie


include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin


config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg   = TOML.parsefile(config_file)
base  = cfg["base_path"]
base2 = cfg["base_path_nt"]


NX, NY = 288, 468
minlat, maxlat = 24.0, 31.91
minlon, maxlon = 193.0, 199.0
lat = range(minlat, maxlat, length=NY)
lon = range(minlon, maxlon, length=NX)


buf    = 3
tx, ty = 47, 66
nx     = tx + 2*buf
ny     = ty + 2*buf


Conv_z = zeros(NX, NY)


for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix2 = @sprintf("%02dx%02d_%d", xn, yn, buf-2)
        fpath   = joinpath(base2, "Conv", "Conv_nt_$suffix2.bin")


        C = Float64.(open(fpath, "r") do io
            reshape(reinterpret(Float32, read(io, (nx-2)*(ny-2)*sizeof(Float32))), nx-2, ny-2)
        end)


        xs = (xn-1)*tx + 1
        xe = xs + tx + (2*buf) - 1
        ys = (yn-1)*ty + 1
        ye = ys + ty + (2*buf) - 1


        Conv_z[xs+2:xe-2, ys+2:ye-2] .= C[2:end-1, 2:end-1]
    end
end


println("Conv_z range: ", extrema(Conv_z[Conv_z .!= 0]))


fig = Figure(resolution=(600, 800))
ax  = Axis(fig[1,1],
    title  = "Barotropic-to-Baroclinic Conversion  (W/m²)",
    xlabel = "Longitude [°]",
    ylabel = "Latitude [°]")
ax.limits[] = ((minimum(lon), maximum(lon)), (minimum(lat), maximum(lat)))


hm = CairoMakie.heatmap!(ax, lon, lat, Conv_z;
    interpolate = false,
    colorrange  = (-0.050, 0.050),
    colormap    = :bwr)


Colorbar(fig[1,2], hm, label="W/m²")
display(fig)


FIGDIR = cfg["fig_base"]
mkpath(FIGDIR)
save(joinpath(FIGDIR, "Conv_map.png"), fig)
println("Figure saved: $(joinpath(FIGDIR, "Conv_map.png"))")




