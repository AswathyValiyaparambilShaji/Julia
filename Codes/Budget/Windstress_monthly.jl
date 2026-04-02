using Printf, FilePathsBase, TOML, CairoMakie, Statistics, LinearAlgebra, Dates


include(joinpath(@__DIR__, "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin


config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base  = cfg["base_path"]
base2 = cfg["base_path2"]


# --- Grid parameters ---
NX, NY = 288, 468
minlat, maxlat = 24.0, 31.91
minlon, maxlon = 193.0, 199.0
lat = range(minlat, maxlat, length=NY)
lon = range(minlon, maxlon, length=NX)


# Tiling parameters
buf = 3
tx, ty = 47, 66
nx = tx + 2 * buf
ny = ty + 2 * buf
dt  = 25
dto = 144
Tts = 366192
nt  = div(Tts, dto)


println("Total time steps: $nt")


# Initialize 3D arrays
TauX_all = zeros(NX, NY, nt)
TauY_all = zeros(NX, NY, nt)


# Load and process tiles
for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
        println("Reading tile $suffix...")


        taux = Float64.(open(joinpath(base, "Windstress", "taux_$suffix.bin"), "r") do io
            nbytes = nx * ny * nt * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt)
        end)


        tauy = Float64.(open(joinpath(base, "Windstress", "tauy_$suffix.bin"), "r") do io
            nbytes = nx * ny * nt * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt)
        end)


        # Center from Arakawa C-grid
        taux_ext = zeros(nx+1, ny, nt)
        taux_ext[1:nx, :, :] .= taux
        taux_ext[end, :, :]  .= taux[end, :, :]


        tauy_ext = zeros(nx, ny+1, nt)
        tauy_ext[:, 1:ny, :] .= tauy
        tauy_ext[:, end, :]  .= tauy[:, end, :]


        taux_c = 0.5 .* (taux_ext[1:end-1, :, :] .+ taux_ext[2:end, :, :])
        tauy_c = 0.5 .* (tauy_ext[:, 1:end-1, :] .+ tauy_ext[:, 2:end, :])


        taux_int = taux_c[buf+1:nx-buf, buf+1:ny-buf, :]
        tauy_int = tauy_c[buf+1:nx-buf, buf+1:ny-buf, :]


        xs = (xn - 1) * tx + 1;  xe = xs + tx - 1
        ys = (yn - 1) * ty + 1;  ye = ys + ty - 1


        TauX_all[xs:xe, ys:ye, :] .= taux_int
        TauY_all[xs:xe, ys:ye, :] .= tauy_int


        println("  Completed $suffix")
    end
end


# Crop to valid region
valid_x  = (buf+1):(NX-buf)
valid_y  = (buf+1):(NY-buf)
lon_crop = lon[valid_x]
lat_crop = lat[valid_y]


# ============================================================================
# TIME AVERAGING BY MONTH
# ============================================================================


start_time = DateTime("2012-03-01T00:00:00")
timestamps = [start_time + Hour(t - 1) for t in 1:nt]


month_names = ["March", "April", "May", "June"]
month_nums  = [3, 4, 5, 6]


NX_crop = length(valid_x)
NY_crop = length(valid_y)


TauX_monthly = zeros(NX_crop, NY_crop, 4)
TauY_monthly = zeros(NX_crop, NY_crop, 4)
counts       = zeros(Int, 4)


for t in 1:nt
    m  = Dates.month(timestamps[t])
    mi = findfirst(==(m), month_nums)
    isnothing(mi) && continue
    TauX_monthly[:, :, mi] .+= TauX_all[valid_x, valid_y, t]
    TauY_monthly[:, :, mi] .+= TauY_all[valid_x, valid_y, t]
    counts[mi] += 1
end


for mi in 1:4
    TauX_monthly[:, :, mi] ./= counts[mi]
    TauY_monthly[:, :, mi] ./= counts[mi]
end


println("Steps per month: ", counts)


FIGDIR = cfg["fig_base"]
mkpath(FIGDIR)


# ============================================================================
# PLOT 1 — Wind Stress Magnitude
# ============================================================================


fig = Figure(resolution = (1200, 900))
for mi in 1:4
    row = (mi - 1) ÷ 2 + 1
    col = (mi - 1) % 2 + 1
    ax  = Axis(fig[row, col], title = month_names[mi],
               xlabel = "Longitude [°]", ylabel = "Latitude [°]")
    mag = sqrt.(TauX_monthly[:, :, mi].^2 .+ TauY_monthly[:, :, mi].^2)
    hm  = heatmap!(ax, lon_crop, lat_crop, mag,
                   colorrange = (0.0, 0.4), colormap = :jet)
    #Colorbar(fig[row, col+2], hm, label = "|τ| [N/m²]")
end
    Colorbar(fig[1:2, 3], hm, label = "|τ| [N/m²]")

save(joinpath(FIGDIR, "WindStress_Magnitude_monthly.png"), fig)
println("Saved WindStress_Magnitude_monthly.png")
display(fig)

# ============================================================================
# PLOT 2 — Zonal Wind Stress τx
# ============================================================================


fig = Figure(resolution = (1200, 900))
for mi in 1:4
    row = (mi - 1) ÷ 2 + 1
    col = (mi - 1) % 2 + 1
    ax  = Axis(fig[row, col], title = month_names[mi],
               xlabel = "Longitude [°]", ylabel = "Latitude [°]")
    hm  = heatmap!(ax, lon_crop, lat_crop, TauX_monthly[:, :, mi],
                   colorrange = (-0.2, 0.3), colormap = :jet)
    #Colorbar(fig[row, col+2], hm, label = "τx [N/m²]")
end
 Colorbar(fig[1:2, 3], hm, label = "τx [N/m²]")
save(joinpath(FIGDIR, "WindStress_TauX_monthly.png"), fig)
println("Saved WindStress_TauX_monthly.png")
display(fig)


# ============================================================================
# PLOT 3 — Meridional Wind Stress τy
# ============================================================================


fig = Figure(resolution = (1200, 900))
for mi in 1:4
    row = (mi - 1) ÷ 2 + 1
    col = (mi - 1) % 2 + 1
    ax  = Axis(fig[row, col], title = month_names[mi],
               xlabel = "Longitude [°]", ylabel = "Latitude [°]")
    hm  = heatmap!(ax, lon_crop, lat_crop, TauY_monthly[:, :, mi],
                   colorrange = (-0.2, 0.3), colormap = :jet)
    #Colorbar(fig[row, col+2], hm, label = "τy [N/m²]")
end
    Colorbar(fig[1:2, 3], hm, label = "τy [N/m²]")

save(joinpath(FIGDIR, "WindStress_TauY_monthly.png"), fig)
println("Saved WindStress_TauY_monthly.png")
display(fig)


println("Done!")




