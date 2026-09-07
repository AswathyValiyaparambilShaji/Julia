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


SP_H_full = fill(NaN, NX, NY)


# ==========================================================
# =========== BUILD HORIZONTAL SHEAR PRODUCTION MAP ========
# ==========================================================



Threads.@threads for xn in cfg["xn_start"]:cfg["xn_e28"]
    for yn in cfg["yn_start"]:cfg["yn_e28"]


        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)


        # Read time-averaged horizontal shear production
        sp_h_mean = Float64.(open(joinpath(base2, "SP_H", "sp_h_nt_$suffix.bin"), "r") do io
            nbytes = nx * ny * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
        end)


        xs = (xn - 1) * tx + 1
        xe = xs + tx - 1
        ys = (yn - 1) * ty + 1
        ye = ys + ty - 1


        sp_h_interior = sp_h_mean[buf+1:nx-buf, buf+1:ny-buf]


        SP_H_full[xs:xe, ys:ye] .= sp_h_interior


        println("Completed tile $suffix")
    end
end


println("\nSP_H_full range: $(minimum(skipmissing(SP_H_full))) to $(maximum(skipmissing(SP_H_full)))")


# ==========================================================
# =================== VISUALIZATION ========================
# ==========================================================


fig = Figure(size=(600, 700))


ax = Axis(fig[1, 1],
        title=" Horizontal Shear Production",
        xlabel="Longitude [°]",
        ylabel="Latitude [°]")


hm = CairoMakie.heatmap!(ax, lon, lat, SP_H_full;
                       interpolate=false,
                       colormap=Reverse(:RdBu),
                       colorrange=(-0.015, 0.015))


Colorbar(fig[1, 2], hm, label="Horizontal Shear Production [W/m²]")


display(fig)


# Save figure
FIGDIR = cfg["fig_base_28"]
save(joinpath(FIGDIR, "SP_H_production_NS_nt_V1.png"), fig)





