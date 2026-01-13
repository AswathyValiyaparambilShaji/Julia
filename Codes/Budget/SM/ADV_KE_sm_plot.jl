using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML
using CairoMakie, SparseArrays


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin


config_file = get(ENV, "JULIA_CONFIG",
               joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)


base  = cfg["base_path"]
base2 = cfg["base_path2"]


NX, NY = 288, 468
minlat, maxlat = 24.0, 31.91
minlon, maxlon = 193.0, 199.0
lat = range(minlat, maxlat, length=NY)
lon = range(minlon, maxlon, length=NX)


buf = 3
tx, ty = 47, 66
nx = tx + 2*buf
ny = ty + 2*buf


U_KE_full = fill(NaN, NX, NY)


# ==========================================================
# ============ BUILD KE ADVECTION MAP ======================
# ==========================================================


for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]


        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)


        # Read time-averaged KE advection flux
        u_ke_mean = Float64.(open(joinpath(base2, "U_KE", "u_ke_mean_$suffix.bin"), "r") do io
            nbytes = nx * ny * sizeof(Float64)
            reshape(reinterpret(Float64, read(io, nbytes)), nx, ny)
        end)


        xs = (xn - 1) * tx + 1
        xe = xs + tx - 1
        ys = (yn - 1) * ty + 1
        ye = ys + ty - 1


        u_ke_interior = u_ke_mean[buf+1:nx-buf, buf+1:ny-buf]


        U_KE_full[xs:xe, ys:ye] .= u_ke_interior


        println("Completed tile $suffix")
    end
end


println("\nU_KE_full range: $(minimum(skipmissing(U_KE_full))) to $(maximum(skipmissing(U_KE_full)))")


# ==========================================================
# =================== VISUALIZATION ========================
# ==========================================================


fig = Figure(size=(1000, 800))


ax = Axis(fig[1, 1],
         title="Depth-Integrated Time-Averaged KE Advection",
         xlabel="Longitude [°]",
         ylabel="Latitude [°]")
#ax.limits[] = (193.0,194.2,24.0, 25.4)

hm = CairoMakie.heatmap!(ax, lon, lat, U_KE_full;
                        interpolate=false,
                        colormap=Reverse(:RdBu),
                        colorrange=(-0.05, 0.05))


Colorbar(fig[1, 2], hm, label="KE Advection [W/m²]")


display(fig)


# Save figure
FIGDIR = cfg["fig_base"]
save(joinpath(FIGDIR, "U_KE_advection_v1.png"), fig)




