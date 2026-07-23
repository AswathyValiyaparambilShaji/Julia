using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML, CairoMakie, NCDatasets


include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin


config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg   = TOML.parsefile(config_file)
base  = cfg["base_path"]
base2 = cfg["base_path2"]


# --- Domain ---
NX, NY   = 288, 468
minlat, maxlat = 24.0, 31.91
minlon, maxlon = 193.0, 199.0
lat = range(minlat, maxlat, length=NY)
lon = range(minlon, maxlon, length=NX)


# --- Tile ---
buf    = 3
tx, ty = 47, 66
nx, ny = tx + 2*buf, ty + 2*buf
nz = 88
dto = 144
Tts = 366192
nt  = div(Tts, dto)
thk   = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF   = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)
g = 9.8
nt_chunk = 72
n_chunks = div(nt, nt_chunk)
ring_steps = nt_chunk
t_safe_start = ring_steps + 1              # first valid step (1801)
t_safe_end   = nt - ring_steps             # last  valid step (nt-1800)


# Safe 3-day chunks: only keep chunks that fall entirely within the safe range
safe_chunks = [c for c in 1:n_chunks
              if (c-1)*nt_chunk + 1 >= t_safe_start &&
                 c*nt_chunk          <= t_safe_end]


# --- Plot settings ---
FIGDIR        = cfg["fig_base"]
HEAT_CBAR_MAX = 15
QUIVER_STEP   = 20
ARROW_SCALEUP = 5.0
mkpath(FIGDIR)


# ════════════════════════════════════════════════════════════════════════
# 1) Assemble full-domain background flux field from tiles (unchanged)
# ════════════════════════════════════════════════════════════════════════
tfx = zeros(NX, NY)
tfy = zeros(NX, NY)


for xn in cfg["xn_start"]:cfg["xn_end"]
  for yn in cfg["yn_start"]:cfg["yn_end"]
      suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
       # Read 4D time series (nx, ny, nz, nt) — written as Float32
       fx = Float64.(open(joinpath(base2, "xflux", "xflx_$suffix.bin"), "r") do io
           raw_bytes = read(io, nx * ny * nz * nt * sizeof(Float32))
           reshape(reinterpret(Float32, raw_bytes), nx, ny, nz, nt)
       end)
       fy = Float64.(open(joinpath(base2, "yflux", "yflx_$suffix.bin"), "r") do io
           raw_bytes = read(io, nx * ny * nz * nt * sizeof(Float32))
           reshape(reinterpret(Float32, raw_bytes), nx, ny, nz, nt)
       end)
   # --- depth from hFacC ---
       hFacC   = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
       DRFfull = hFacC .* DRF3d
       DRFfull[hFacC .== 0] .= 0.0
       depth   = dropdims(sum(DRFfull, dims=3), dims=3)
       DRFfull[hFacC .== 0] .= 0.0
       # Time average over dim=4
       fx_tmean = mean(fx[:, :,:, t_safe_start:t_safe_end], dims=4)[:, :, :, 1]   # (nx, ny, nz)
       fy_tmean = mean(fy[:, :,:, t_safe_start:t_safe_end], dims=4)[:, :, :, 1]   # (nx, ny, nz)
       # Depth integrate
       DRFfull = hFacC .* DRF3d
       fxX = sum(fx_tmean .* DRFfull, dims=3)    # (nx, ny, 1)
       fyY = sum(fy_tmean .* DRFfull, dims=3)    # (nx, ny, 1)
      xs = (xn-1)*tx + 1;  xe = xs + tx - 1
      ys = (yn-1)*ty + 1;  ye = ys + ty - 1
      xsf = buf+1;          xef = buf+tx
      ysf = buf+1;          yef = buf+ty
      tfx[xs:xe, ys:ye] .= fxX[xsf:xef, ysf:yef]
      tfy[xs:xe, ys:ye] .= fyY[xsf:xef, ysf:yef]
  end
end


fm    = sqrt.(tfx.^2 .+ tfy.^2)
fm_kW = fm ./ 1000                  # W/m -> kW/m


# --- reference physical arrow length (in degrees) that the fixed kW/m
#     reference values below will be scaled to ---
cell_x = (maxlon - minlon) / NX
cell_y = (maxlat - minlat) / NY
target = 5f0 * Float32(min(cell_x, cell_y))


# ════════════════════════════════════════════════════════════════════════
# 2) Mooring locations
# ════════════════════════════════════════════════════════════════════════
mooring_ids = [82, 83, 84, 85]
target_lats = [25.4891, 27.7690, 28.8995, 30.1312]
target_lons = [194.8451, 196.0301, 196.5105, 197.1154]
iwap_idx    = [1, 2, 3, 4]
n_points    = length(iwap_idx)


# ════════════════════════════════════════════════════════════════════════
# 3) IWAP observed flux, per mode (no summing)
# ════════════════════════════════════════════════════════════════════════
file3path = "/nobackup/avaliyap/Box56/Mooring_Data/Flux_mooring_timeseries_ALL_IWAP.mat"
f3 = matopen(file3path)
lato3 = vec(read(f3, "lato"))
lono3 = vec(read(f3, "lono"))
Fuo3  = read(f3, "Fuo")   # (n_moorings, n_modes)
Fvo3  = read(f3, "Fvo")   # (n_moorings, n_modes)
close(f3)


Fu_iwap_mode1 = [Fuo3[iwap_idx[p], 1] for p in 1:n_points]
Fv_iwap_mode1 = [Fvo3[iwap_idx[p], 1] for p in 1:n_points]
Fu_iwap_mode2 = [Fuo3[iwap_idx[p], 2] for p in 1:n_points]
Fv_iwap_mode2 = [Fvo3[iwap_idx[p], 2] for p in 1:n_points]


for p in 1:n_points
  idx = iwap_idx[p]
  dlat = abs(lato3[idx] - target_lats[p])
  dlon = abs(lono3[idx] - target_lons[p])
  if dlat > 0.01 || dlon > 0.01
      @warn "Mooring $(mooring_ids[p]): IWAP file lat/lon ($(lato3[idx]), $(lono3[idx])) " *
            "does not closely match expected ($(target_lats[p]), $(target_lons[p]))"
  end
end


# ════════════════════════════════════════════════════════════════════════
# 4) Model flux, per mode (depth-integrated, kW/m)
# ════════════════════════════════════════════════════════════════════════
model_file = joinpath(base, "SM", "MooringModalFlux", "MooringModalFlux_Box56_IWAP.nc")


lat_model, lon_model, uflux_model_modes, vflux_model_modes =
  NCDataset(model_file, "r") do ds
      (Array(ds["lat"][:]), Array(ds["lon"][:]),
       Array(ds["uflux_depth_int"][:, :]),   # (station, mode)
       Array(ds["vflux_depth_int"][:, :]))
  end


Fu_model_mode1 = Float64.(uflux_model_modes[:, 1])
Fv_model_mode1 = Float64.(vflux_model_modes[:, 1])
Fu_model_mode2 = Float64.(uflux_model_modes[:, 2])
Fv_model_mode2 = Float64.(vflux_model_modes[:, 2])


FONT = "FreeSerif Bold"


# ════════════════════════════════════════════════════════════════════════
# 5) Two-panel figure: Mode 1 (left) and Mode 2 (right), same background
#    -- pcolor background only, NO background quiver arrows
#    -- each mode has its OWN arrow scale factor
# ════════════════════════════════════════════════════════════════════════
fig = Figure(resolution = (600, 400), figure_padding =(5,5,5,5),
             fonts=(; regular=FONT))


scale_x0 = minlon + 0.4
scale_y0 = maxlat - 0.4


# fixed reference values for each panel's legend -- these now ALSO drive the
# arrow scale factor below, so the plotted mooring arrows and the legend
# arrow are both scaled from the same fixed number, not independently
scale_ref_kWm1 = 2.0     # fixed reference for mode 1, in kW/m
scale_ref_kWm2 = 0.1    # fixed reference for mode 2, in kW/m


scale_mode1 = (target / Float32(scale_ref_kWm1)) * Float32(ARROW_SCALEUP)   # kW/m -> degrees
scale_mode2 = (target / Float32(scale_ref_kWm2)) * Float32(ARROW_SCALEUP)   # kW/m -> degrees


# --- Panel 1: Mode 1 ---
ax1 = Axis(fig[1, 1],
     aspect = DataAspect(),
    title      = "Mode 1 flux  ",
    xlabel     = "Longitude [°]",
    ylabel     = "Latitude [°]",
    xlabelsize = 12,
    ylabelsize = 12,
    titlesize  = 10,
    titlefont         = FONT,
    xlabelfont        = FONT,
    ylabelfont        = FONT,
    xticklabelfont    = FONT,
    yticklabelfont    = FONT,
    )
ax1.limits[] = ((minlon, maxlon), (minlat, maxlat))


hm1 = CairoMakie.heatmap!(ax1, lon, lat, fm_kW,
    interpolate = false,
    colorrange  = (0, HEAT_CBAR_MAX),
    colormap    = :Spectral_9)


iwap_vecs1  = Vec2f.(Float32.(Fu_iwap_mode1 .* scale_mode1), Float32.(Fv_iwap_mode1 .* scale_mode1))
model_vecs1 = Vec2f.(Float32.(Fu_model_mode1 .* scale_mode1), Float32.(Fv_model_mode1 .* scale_mode1))
mooring_pos = Point2f.(Float32.(target_lons), Float32.(target_lats))


arrows!(ax1, mooring_pos, iwap_vecs1;  color = :black,  arrowsize = 7, linewidth = 3)
arrows!(ax1, mooring_pos, model_vecs1; color = :magenta, arrowsize = 7, linewidth = 3)


#=scatter!(ax1, target_lons, target_lats; color = :black, markersize = 6)
for p in 1:n_points
    text!(ax1, target_lons[p], target_lats[p]; text = "M$(mooring_ids[p])",
          offset = (6, 6), fontsize = 10, color = :black)
end=#


# --- scale legend arrow for panel 1 (fixed reference: 2 kW/m) ---
scale_len1 = Float32(scale_ref_kWm1 * scale_mode1)
lines!(ax1, [scale_x0, scale_x0 + scale_len1], [scale_y0, scale_y0];
       color = :black, linewidth = 1.5)
arrows!(ax1, [Point2f(scale_x0, scale_y0)], [Vec2f(scale_len1, 0f0)];
        color = :black, arrowsize = 7, linewidth = 1.5)
text!(ax1, scale_x0, scale_y0 - 0.25; text = "$(round(scale_ref_kWm1, digits=2)) kW/m",
      fontsize = 6, color = :black)


# --- Panel 2: Mode 2 ---
ax2 = Axis(fig[1, 2],
     aspect = DataAspect(),
    title      = "Mode 2 flux  ",
    xlabel     = "Longitude [°]",
    ylabel     = " ",
    yticklabelsvisible = false,

    xlabelsize = 12,
    ylabelsize = 12,
    titlesize  = 10,
    titlefont         = FONT,
    xlabelfont        = FONT,
    ylabelfont        = FONT,
    xticklabelfont    = FONT,
    yticklabelfont    = FONT,
    )
ax2.limits[] = ((minlon, maxlon), (minlat, maxlat))


hm2 = CairoMakie.heatmap!(ax2, lon, lat, fm_kW,
    interpolate = false,
    colorrange  = (0, HEAT_CBAR_MAX),
    colormap    = :Spectral_9)


iwap_vecs2  = Vec2f.(Float32.(Fu_iwap_mode2 .* scale_mode2), Float32.(Fv_iwap_mode2 .* scale_mode2))
model_vecs2 = Vec2f.(Float32.(Fu_model_mode2 .* scale_mode2), Float32.(Fv_model_mode2 .* scale_mode2))


arrows!(ax2, mooring_pos, iwap_vecs2;  color = :black,  arrowsize = 7, linewidth = 3)
arrows!(ax2, mooring_pos, model_vecs2; color = :magenta, arrowsize = 7, linewidth = 3)


#=scatter!(ax2, target_lons, target_lats; color = :black, markersize = 6)
for p in 1:n_points
    text!(ax2, target_lons[p], target_lats[p]; text = "M$(mooring_ids[p])",
          offset = (6, 6), fontsize = 10, color = :black)
end=#


# --- scale legend arrow for panel 2 (fixed reference: 0.05 kW/m) ---
scale_len2 = Float32(scale_ref_kWm2 * scale_mode2)
lines!(ax2, [scale_x0, scale_x0 + scale_len2], [scale_y0, scale_y0];
       color = :black, linewidth =1.5)
arrows!(ax2, [Point2f(scale_x0, scale_y0)], [Vec2f(scale_len2, 0f0)];
        color = :black, arrowsize = 7, linewidth = 1.5)
text!(ax2, scale_x0, scale_y0 - 0.25; text = "$(round(scale_ref_kWm2, digits=2)) kW/m",
      fontsize = 6, color = :black)


Colorbar(fig[1, 3], hm1, label = "(kW/m)", labelsize = 9, ticklabelsize=7 , width = 4.5)


# shared legend for arrow colors (IWAP vs model)
elem_iwap  = LineElement(color = :black,  linewidth = 3)
elem_model = LineElement(color = :magenta, linewidth = 3)
Legend(fig[2, 1:3], [elem_iwap, elem_model],
      ["IWAP (observed)", "Model (reconstructed)"],
      orientation = :horizontal, tellwidth = false, labelfont = FONT)
colgap!(fig.layout,1,3)
colgap!(fig.layout,2,3)


display(fig)


png_file = joinpath(FIGDIR, "Flux_corr_modes1_2_moorings_V3.png")
save(png_file, fig)
println("Saved: $png_file")




