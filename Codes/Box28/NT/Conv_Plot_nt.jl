using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML
using CairoMakie, SparseArrays


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base = cfg["bp_box28"]
base2 = (joinpath(base, "NT"))       


# --- Domain & grid of 27b ---
NX, NY = 384, 336
minlat, maxlat = -24.5, -18.5
minlon, maxlon = 337.5, 345.4791122715405
lat = range(minlat, maxlat, length=NY)
lon = range(minlon, maxlon, length=NX)
NZ = 173

# --- Tile & time ---
buf = 3
tx, ty = 54, 66
nx = tx + 2*buf
ny = ty + 2*buf
nz = 168
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

Conv_z = zeros(NX, NY)


Threads.@threads for xn in cfg["xn_start"]:cfg["xn_e28"]
    for yn in cfg["yn_start"]:cfg["yn_e28"]
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


fig = Figure(resolution=(600, 700))
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


FIGDIR = cfg["fig_base_28"]
mkpath(FIGDIR)
save(joinpath(FIGDIR, "Conv_NS_nt_v1.png"), fig)
println("Figure saved: $(joinpath(FIGDIR, "Conv_NS_nt_v1.png"))")




