using CairoMakie

include(joinpath(@__DIR__, "..","..", "functions", "FluxUtils.jl"))
include(joinpath(@__DIR__, "..","..", "functions", "densjmd95.jl"))

using .FluxUtils: read_bin, bandpassfilter
# ── File and dimensions ────────────────────────────────────────────────────────
fpath = "/data3/aswathy/mnt/data/aswathy/MITgcm_NAS/figures/U_288x468x168.20230501T000000"
NX, NY, NZ = 288, 468, 168


# ── Read ───────────────────────────────────────────────────────────────────────
#arr = read_bin(fpath, (NX * NY * NZ,))
U   = (open(fpath, "r") do io
            reshape(ntoh.(reinterpret(Float32, read(io, NX*NY*NZ*sizeof(Float32)))), NX, NY, NZ)
    end)

# ── Slice indices ──────────────────────────────────────────────────────────────
k0 = 1           # surface level for xy plot
j0 = NY ÷ 2      # middle row for xz plot
i0 = NX ÷ 2      # middle column for yz plot


xy = U[:, :, k0]
xz = U[:, j0, :]
yz = U[i0, :, :]


# Mask zeros (land/dry cells)
xy[xy .== 0f0] .= NaN
xz[xz .== 0f0] .= NaN
yz[yz .== 0f0] .= NaN


# ── Plot ───────────────────────────────────────────────────────────────────────
fig = Figure(resolution = (1200, 380))

cr(fld) = extrema(filter(isfinite, vec(fld)))

ax1 = Axis(fig[1, 1], title = "xy  (k=$k0, surface)", xlabel = "x", ylabel = "y")
hm1 = heatmap!(ax1, xy', colormap = :balance, colorrange=(-0.5,0.5))
Colorbar(fig[1, 2], hm1, label = "U (m/s)")


ax2 = Axis(fig[1, 3], title = "xz  (j=$j0)", xlabel = "x", ylabel = "z level")
hm2 = heatmap!(ax2, xz', colormap = :balance, colorrange=(-0.5,0.5))
Colorbar(fig[1, 4], hm2, label = "U (m/s)")


ax3 = Axis(fig[1, 5], title = "yz  (i=$i0)", xlabel = "y", ylabel = "z level")
hm3 = heatmap!(ax3, yz', colormap = :balance, colorrange=(-0.5,0.5))
Colorbar(fig[1, 6], hm3, label = "U (m/s)")


display(fig)




