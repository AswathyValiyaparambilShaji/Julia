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


U_PE_full = fill(NaN, NX, NY)


# ==========================================================
# ============ BUILD PE ADVECTION MAP ======================
# ==========================================================


Threads.@threads for xn in cfg["xn_start"]:cfg["xn_e28"]
    for yn in cfg["yn_start"]:cfg["yn_e28"]


        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)


        # Read time-averaged PE advection flux
        u_pe_mean = Float64.(open(joinpath(base2, "U_PE", "u_pe_nt_$suffix.bin"), "r") do io
            nbytes = nx * ny * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
        end)


        xs = (xn - 1) * tx + 1
        xe = xs + tx - 1
        ys = (yn - 1) * ty + 1
        ye = ys + ty - 1


        u_pe_interior = u_pe_mean[buf+1:nx-buf, buf+1:ny-buf]


        U_PE_full[xs:xe, ys:ye] .= u_pe_interior


        println("Completed tile $suffix")
    end
end


println("\nU_PE_full range: $(minimum(skipmissing(U_PE_full))) to $(maximum(skipmissing(U_PE_full)))")


# ==========================================================
# =================== VISUALIZATION ========================
# ==========================================================


fig = Figure(size=(600, 700))


ax = Axis(fig[1, 1],
         title="PE Advection",
         xlabel="Longitude [°]",
         ylabel="Latitude [°]")
#ax.limits[] = (193.0,194.2,24.0, 25.4)

hm = CairoMakie.heatmap!(ax, lon, lat, U_PE_full;
                        interpolate=false,
                        colormap=Reverse(:RdBu),
                        colorrange=(-0.05, 0.05))


Colorbar(fig[1, 2], hm, label="PE Advection [W/m²]")


display(fig)


# Save figure
FIGDIR = cfg["fig_base_28"]
save(joinpath(FIGDIR, "U_PE_advection_NS_nt_V1.png"), fig)




