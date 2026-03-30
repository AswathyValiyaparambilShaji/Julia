using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML
using CairoMakie, SparseArrays


include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin


config_file = get(ENV, "JULIA_CONFIG",
              joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
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


SP_V_full = fill(NaN, NX, NY)


# ==========================================================
# ============ BUILD VERTICAL SHEAR PRODUCTION MAP =========
# ==========================================================


for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]


        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)


        # Read time-averaged vertical shear production
        sp_v_mean = Float64.(open(joinpath(base2, "SP_V", "sp_v_mean_$suffix.bin"), "r") do io
            nbytes = nx * ny * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
        end)


        xs = (xn - 1) * tx + 1
        xe = xs + tx - 1
        ys = (yn - 1) * ty + 1
        ye = ys + ty - 1


        sp_v_interior = sp_v_mean[buf+1:nx-buf, buf+1:ny-buf]


        SP_V_full[xs:xe, ys:ye] .= sp_v_interior


        println("Completed tile $suffix")
    end
end


println("\nSP_V_full range: $(minimum(skipmissing(SP_V_full))) to $(maximum(skipmissing(SP_V_full)))")


# ==========================================================
# =================== VISUALIZATION ========================
# ==========================================================


fig = Figure(size=(1000, 800))


ax = Axis(fig[1, 1],
        title="Depth-Integrated Time-Averaged Vertical Shear Production",
        xlabel="Longitude [°]",
        ylabel="Latitude [°]")


hm = CairoMakie.heatmap!(ax, lon, lat, SP_V_full;
                       interpolate=false,
                       colormap=Reverse(:RdBu),
                       colorrange=(-0.01, 0.01))


Colorbar(fig[1, 2], hm, label="Vertical Shear Production [W/m²]")


display(fig)


# Save figure
FIGDIR = cfg["fig_base"]
save(joinpath(FIGDIR, "SP_V_production_v4_n.png"), fig)




