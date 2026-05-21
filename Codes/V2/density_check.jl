using MAT, Printf, TOML, CairoMakie
include(joinpath(@__DIR__, "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin


config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "config", "run_debug.toml"))
cfg  = TOML.parsefile(config_file)
base = cfg["base_path_V2"]


# ── Grid ──────────────────────────────────────────────────────────────────────
NX, NY = 288, 468
minlat, maxlat = 24.0, 31.91
minlon, maxlon = 193.0, 199.0
lat = range(minlat, maxlat, length=NY)
lon = range(minlon, maxlon, length=NX)
NZ = 173


# ── Tiling parameters ─────────────────────────────────────────────────────────
buf    = 3
tx, ty = 47, 66
nx = tx + 2*buf   # 53
ny = ty + 2*buf   # 72
nz  = 168
nt  = 558


# ── Layer thicknesses ─────────────────────────────────────────────────────────
thk = open(joinpath(base, "hFacC", "delR.bin"), "r") do io
    raw = read(io, NZ * sizeof(Float32))
    ntoh.(reshape(reinterpret(Float32, raw), NZ))
end
DRF = thk[1:nz]                                   # 168 layer thicknesses [m]
z_mid = cumsum(DRF) .- DRF ./ 2                   # depth of layer centres [m]


# ── Snapshot to map ───────────────────────────────────────────────────────────
kz = 1    # surface level
it = 10   # time index


# ── Interior index range (strip buf cells each side) ──────────────────────────
# source:  buf+1 : nx-buf  →  4:50  →  tx=47 points
# dest:    xs : xs+tx-1
i_src = buf+1:nx-buf
j_src = buf+1:ny-buf


# ── Assemble surface density map ──────────────────────────────────────────────
D_surf = fill(NaN, NX, NY)
for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
        fpath  = joinpath(base, "Density", "rho_in_$suffix.bin")


        # BUG FIX: file was written as Float64 → must read as Float64
        rho = open(fpath, "r") do io
            reshape(reinterpret(Float64, read(io, nx*ny*nz*nt*sizeof(Float64))),
                    nx, ny, nz, nt)
        end


        # BUG FIX: global tile destinations (do NOT reassign xn/yn)
        xs = (xn-1)*tx + 1;  xe = xs + tx - 1   # length tx
        ys = (yn-1)*ty + 1;  ye = ys + ty - 1   # length ty


        D_surf[xs:xe, ys:ye] .= rho[i_src, j_src, kz, it]
    end
end
println("D_surf range: ", extrema(filter(!isnan, D_surf)))


# ── Map plot ──────────────────────────────────────────────────────────────────
fig = Figure(size=(700, 900))
ax  = Axis(fig[1,1],
    title  = "Surface Density  (kz=$kz, t=$it)",
    xlabel = "Longitude [°E]",
    ylabel = "Latitude [°N]")
hm = CairoMakie.heatmap!(ax, collect(lon), collect(lat), D_surf;
    interpolate = false,
    colorrange  = (1022.5, 1026.5),        # realistic open-Pacific surface range
    colormap    = :jet)
Colorbar(fig[1,2], hm; label="ρ  [kg m⁻³]")


FIGDIR = cfg["fig_base"]
mkpath(FIGDIR)
save(joinpath(FIGDIR, "D_surf_v2.png"), fig)
println("Saved map: $(joinpath(FIGDIR, "D_surf_v2.png"))")
display(fig)


# ── Density profiles at one point ─────────────────────────────────────────────
# Target: centre of the domain (global indices, 1-based)
xi_g = NX ÷ 2       # ≈ 144
yi_g = NY ÷ 2       # ≈ 234
lon_pt = lon[xi_g]
lat_pt = lat[yi_g]


# Which tile?
xn_pt = 1
yn_pt = 1
# Local index inside the tile array (offset by buf)
xi_loc =30     # in [buf+1, nx-buf]
yi_loc = 20


println(@sprintf("Profile point: (%.2f°E, %.2f°N)  → tile %dx%d  local (%d,%d)",
        lon_pt, lat_pt, xn_pt, yn_pt, xi_loc, yi_loc))


suffix_pt = @sprintf("%02dx%02d_%d", xn_pt, yn_pt, buf)
fpath_pt  = joinpath(base, "Density", "rho_in_$suffix_pt.bin")


rho_pt = open(fpath_pt, "r") do io
    reshape(reinterpret(Float64, read(io, nx*ny*nz*nt*sizeof(Float64))),
            nx, ny, nz, nt)
end

hFacC =(open(joinpath(base, "hFacC",  "hFacC_v2_$suffix_pt.bin"), "r") do io
                raw = read(io,  nx*ny*nz * sizeof(Float32))
                (reshape(reinterpret(Float32, raw), nx,ny,nz))
                end) 

# ── Profile figure ────────────────────────────────────────────────────────────
profile_times = [1, 100, 200, 300, 400, 500]
fig2 = Figure(size=(480, 700))
ax2  = Axis(fig2[1,1],
    title  = @sprintf("Density profile  (%.2f°E, %.2f°N)", lon_pt, lat_pt),
    xlabel = "ρ  [kg m⁻³]",
    ylabel = "Depth  [m]",
    yreversed = true)

hf = hFacC[xi_loc,yi_loc,:]
for it_p in profile_times
    prof = rho_pt[xi_loc, yi_loc, :, it_p]
    prof[hf .== 0] .= NaN
    lines!(ax2, prof, z_mid; label="t=$it_p")
end
axislegend(ax2; labelsize=11)


save(joinpath(FIGDIR, "D_profile_v2.png"), fig2)
println("Saved profile: $(joinpath(FIGDIR, "D_profile_v2.png"))")
display(fig2)




