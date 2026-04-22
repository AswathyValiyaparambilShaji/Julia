using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML


include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter


config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base  = cfg["base_path"]
base2 = cfg["base_path2"]


mkpath(joinpath(base2, "xflux"))
mkpath(joinpath(base2, "yflux"))


# --- Domain & grid ---
NX, NY = 288, 468
minlat, maxlat = 24.0, 31.91
minlon, maxlon = 193.0, 199.0
lat = range(minlat, maxlat, length=NY)
lon = range(minlon, maxlon, length=NX)


# --- Tile ---
buf = 3
tx, ty = 47, 66
nx = tx + 2*buf
ny = ty + 2*buf
nz = 88


# --- Thickness ---
thk   = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF   = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)


# ============================================================================
# ASSEMBLE FULL TIME-AVERAGED FLUX  (identical to time_mode="full" in ref code)
# ============================================================================
println("Reading full time-averaged flux files...")
tfx = zeros(NX, NY)
tfy = zeros(NX, NY)


for xn in cfg["xn_start"]:cfg["xn_end"]
	for yn in cfg["yn_start"]:cfg["yn_end"]
    	suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
    	hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))


    	fx = Float64.(open(joinpath(base2, "xflux", "xflx_$suffix.bin"), "r") do io
        	nbytes = nx * ny * nz * sizeof(Float32)
        	raw_bytes = read(io, nbytes)
        	raw_data  = reinterpret(Float32, raw_bytes)
        	reshape(raw_data, nx, ny, nz)
    	end)


    	fy = Float64.(open(joinpath(base2, "yflux", "yflx_$suffix.bin"), "r") do io
        	nbytes = nx * ny * nz * sizeof(Float32)
        	raw_bytes = read(io, nbytes)
        	raw_data  = reinterpret(Float32, raw_bytes)
        	reshape(raw_data, nx, ny, nz)
    	end)


    	DRFfull = hFacC .* DRF3d
    	fxX = sum(fx .* DRFfull, dims=3)
    	fyY = sum(fy .* DRFfull, dims=3)


    	xs  = (xn - 1) * tx + 1
    	xe  = xs + tx + (2 * buf) - 1
    	ys  = (yn - 1) * ty + 1
    	ye  = ys + ty + (2 * buf) - 1
    	xsf = 2
    	xef = tx + (2*buf) - 1
    	ysf = 2
    	yef = ty + (2*buf) - 1


    	tfx[xs+1:xe-1, ys+1:ye-1] .= fxX[xsf:xef, ysf:yef, 1]
    	tfy[xs+1:xe-1, ys+1:ye-1] .= fyY[xsf:xef, ysf:yef, 1]
	end
end


# ============================================================================
# DEFINE 10 EQUIDISTANT BEAM POINTS
# ============================================================================
beam_start_lon, beam_start_lat = 193.3, 24.2
beam_end_lon,   beam_end_lat   = 196.25, 27.5
N_beam = 10


beam_lons = collect(range(beam_start_lon, beam_end_lon, length=N_beam))
beam_lats = collect(range(beam_start_lat, beam_end_lat, length=N_beam))


println("\n========== BEAM POINTS ==========")
for i in 1:N_beam
	@printf("  Point %2d:  lon = %.4f°   lat = %.4f°\n", i, beam_lons[i], beam_lats[i])
end
println("=================================\n")


# ============================================================================
# DUMMY EXTENDED POINTS  adjust these coordinates and re-run to check visually
# ============================================================================
dummy_lons = [196.56, 196.62, 196.63, 196.65, 196.71,
          	197.10, 197.550, 197.95, 198.35, 198.70]
dummy_lats = [ 27.95,  28.50,  28.95,  29.35, 29.85, 
    30.20,    30.58,  30.85,  31.25,  31.55]
N_dummy = length(dummy_lons)


println("========== DUMMY EXTENDED POINTS ==========")
for i in 1:N_dummy
	@printf("  Dummy %2d:  lon = %.4f°   lat = %.4f°\n", i, dummy_lons[i], dummy_lats[i])
end
println("===========================================\n")


# ============================================================================
# PLOT   flux heatmap + beam points + dummy points
# ============================================================================
using CairoMakie


FIGDIR    	= cfg["fig_base"]
HEAT_CBAR_MAX = 15


mkpath(FIGDIR)


fm	= sqrt.(tfx.^2 .+ tfy.^2)
fm_kW = fm ./ 1000


fig = Figure(resolution = (700, 600))
ax  = Axis(fig[1, 1],
	title  	= "MITgcm Flux  Full Time Average with Beam Points",
	xlabel 	= "Longitude [°]",
	ylabel 	= "Latitude [°]",
	ylabelsize = 22,
	xlabelsize = 22,
	titlesize  = 24)


hm = CairoMakie.heatmap!(ax, lon, lat, fm_kW,
	interpolate = false,
	colorrange  = (0, HEAT_CBAR_MAX),
	colormap	= :Spectral_9)


# --- Original 10 equidistant beam points ---
lines!(ax, beam_lons, beam_lats,
	color=:white, linewidth=2, linestyle=:dash)


scatter!(ax, beam_lons, beam_lats,
	color=:white, markersize=12,
	strokewidth=1.5, strokecolor=:black)


for i in 1:N_beam
	text!(ax, beam_lons[i] + 0.04, beam_lats[i] + 0.04,
    	text=string(i), color=:white, fontsize=13, font=:bold)
end


# --- Dummy extended points (trial  adjust coordinates above) ---
lines!(ax, dummy_lons, dummy_lats,
	color=:yellow, linewidth=2, linestyle=:dash)


scatter!(ax, dummy_lons, dummy_lats,
	color=:yellow, markersize=12,
	strokewidth=1.5, strokecolor=:black)


for i in 1:N_dummy
	text!(ax, dummy_lons[i] + 0.04, dummy_lats[i] + 0.04,
    	text=string(N_beam + i), color=:yellow, fontsize=13, font=:bold)
end


Colorbar(fig[1, 2], hm, label="(kW/m)")


png_file = joinpath(FIGDIR, "Flux_beam_points.png")
save(png_file, fig)
display(fig)
println("Saved: $png_file")


