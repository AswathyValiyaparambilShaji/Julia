using MAT
using NCDatasets
using Statistics   # for `mean`
# ════════════════════════════════════════════════════════════════════════
# 1) File paths
# ════════════════════════════════════════════════════════════════════════
file1path = "/home/aswathy/mnt/data/aswathy/Mooring_Data/Flux_mooring_timeseries_ALL.mat"
ncpath    = "/home/aswathy/Downloads/intrfreq2_M2.nc"
# ════════════════════════════════════════════════════════════════════════
# 2) Read from ALL.mat
# ════════════════════════════════════════════════════════════════════════
f1 = matopen(file1path)
for name in keys(f1)
   data = read(f1, name)
   println("Variable: ", name)
   println("  Type: ", typeof(data))
   println("  Size: ", size(data))
   println()
end
lato1 = vec(read(f1, "lato")); lono1 = vec(read(f1, "lono"))
Fuo1  = read(f1, "Fuo");       Fvo1  = read(f1, "Fvo")
close(f1)
# ════════════════════════════════════════════════════════════════════════
# 3) Read from .nc file
# ════════════════════════════════════════════════════════════════════════
ds = Dataset(ncpath, "r")
lat_nc = vec(ds["lat"][:])
lon_nc = vec(ds["lon"][:])
lon_nc = mod.(lon_nc, 360)
um_kwm = ds["um_kwm"][:, :]
vm_kwm = ds["vm_kwm"][:, :]
close(ds)
# ════════════════════════════════════════════════════════════════════════
# 4) Match EVERY mooring to its NEAREST NC point — no mooring is dropped.
#    Previously moorings beyond `tol` degrees were excluded (j = 0). Now we
#    always take the closest NC index, and just flag whether it was within
#    tolerance or not, so every ALL.mat mooring makes it into the analysis.
# ════════════════════════════════════════════════════════════════════════
tol = 0.05  # degrees — used only to flag "close" vs "nearest but far" matches
function find_nearest_in_nc(lat0, lon0, lat_nc, lon_nc)
   lon0m = mod(lon0, 360)
   best_j = 1
   best_d = Inf
   for j in eachindex(lat_nc)
       d = sqrt((lat0 - lat_nc[j])^2 + (lon0m - lon_nc[j])^2)
       if d < best_d
           best_d = d
           best_j = j
       end
   end
   return best_j, best_d
end
matches = NamedTuple{(:mat_idx, :nc_idx, :dist, :within_tol), Tuple{Int,Int,Float64,Bool}}[]
for i in eachindex(lato1)
   j, d = find_nearest_in_nc(lato1[i], lono1[i], lat_nc, lon_nc)
   within = d <= tol
   push!(matches, (mat_idx=i, nc_idx=j, dist=d, within_tol=within))
   flag = within ? "within tol" : "NEAREST ONLY, beyond tol"
   println("  ALL mooring #$i (lat=$(lato1[i]), lon=$(mod(lono1[i],360)))  ->  NC #$j (lat=$(lat_nc[j]), lon=$(lon_nc[j]))  dist=$(round(d,digits=3))  [$flag]")
end
n_moor = length(matches)
println("\nTotal moorings included (all of them): $n_moor\n")
println("Fuo1/Fvo1 full size: ", size(Fuo1), " / ", size(Fvo1))
println("um_kwm/vm_kwm full size: ", size(um_kwm), " / ", size(vm_kwm))
println()
# ════════════════════════════════════════════════════════════════════════
# 5) Extract flux at each matched mooring (arrays now sized by n_moor)
# ════════════════════════════════════════════════════════════════════════
Fuo  = zeros(n_moor, 2)
Fvo  = zeros(n_moor, 2)
Fva  = zeros(n_moor, 2)
Fua  = zeros(n_moor, 2)
lona = zeros(n_moor)
lata = zeros(n_moor)
lono = zeros(n_moor)
lato = zeros(n_moor)
tt = 1
for m in matches
   i, j = m.mat_idx, m.nc_idx
   # Extract flux slices at this mooring — adjust indexing dimension if needed
   Fuo_i = Fuo1[i, :, :]
   n_nan = count(isnan, Fuo_i)
   println("Nan number", n_nan)   # assumes mooring is dim 1; flip to Fuo1[:, i] if dim 2
   Fuo_i[isnan.(Fuo_i)] .= 0
   Fuo[tt, :] = mean(Fuo_i, dims=2)
   Fvo_i = Fvo1[i, :, :]
   Fvo_i[isnan.(Fvo_i)] .= 0
   Fvo[tt, :] = mean(Fvo_i, dims=2)
   lona[tt] = lon_nc[j]
   lata[tt] = lat_nc[j]
   lono[tt] = lono1[i]
   lato[tt] = lato1[i]
   Fua[tt, :] = um_kwm[j, :]
   Fva[tt, :] = vm_kwm[j, :]
   tt = tt + 1
   println()
end
println("Fuo/Fvo full size: ", size(Fuo), " / ", size(Fvo))
println("Fua/Fva full size: ", size(Fua), " / ", size(Fva))
nn = 19
n1 = 11
#println(Fua[:, 1])
#println(Fuo[:, 1])
println(lata[n1:nn])
println(lato[n1:nn])
println(lona[n1:nn])
println(lono[n1:nn])
using CairoMakie   # switch to GLMakie if you prefer an interactive window
# ════════════════════════════════════════════════════════════════════════
# 6) Magnitudes for both datasets (needed for the magnitude scatter panels)
# ════════════════════════════════════════════════════════════════════════
mag_mode1_o = sqrt.(Fuo[:,1].^2 .+ Fvo[:,1].^2)   # Ansong, mode 1
mag_mode2_o = sqrt.(Fuo[:,2].^2 .+ Fvo[:,2].^2)   # Ansong, mode 2
mag_mode1_a = sqrt.(Fua[:,1].^2 .+ Fva[:,1].^2)   # Alford, mode 1
mag_mode2_a = sqrt.(Fua[:,2].^2 .+ Fva[:,2].^2)   # Alford, mode 2
# ════════════════════════════════════════════════════════════════════════
# ORIGINAL VECTOR-MAP COMPARISON (commented out — kept for reference)
# ════════════════════════════════════════════════════════════════════════
# # --- mooring positions (use the matched NC lon/lat, or swap to lono/lato if you prefer) ---
# mooring_pos = Point2f.(lono, lato)
# # --- axis limits with a bit of padding ---
# pad = 0.5
# minlon, maxlon = minimum(lono) - pad, maximum(lono) + pad
# minlat, maxlat = minimum(lato) - pad, maximum(lato) + pad
# # --- arrow scaling ---
# ARROW_SCALEUP = 5.0
# cell_x = (maxlon - minlon) / NX
# cell_y = (maxlat - minlat) / NY
# target = 5f0 * Float32(min(cell_x, cell_y))
# mag_mode1 = sqrt.(Fuo[:,1].^2 .+ Fvo[:,1].^2)
# mag_mode2 = sqrt.(Fuo[:,2].^2 .+ Fvo[:,2].^2)
# scale_ref_kWm1 = 0.5
# scale_ref_kWm2 = 0.05
# scale_mode1 = (target / scale_ref_kWm1) * ARROW_SCALEUP
# scale_mode2 = (target / scale_ref_kWm2) * ARROW_SCALEUP
# scale_x0 = 225#minlon - 3.4
# scale_y0 = 70#maxlat - 0.4
# fig = Figure(resolution = (1400, 800))
# # --- Panel 1: Mode 1 ---
# ax1 = Axis(fig[1, 1],
#    #aspect = DataAspect(),
#    title      = "Mode 1 flux (Ansong vs Alford)",
#    xticklabelsvisible = false,
#    xlabel     = "",
#    ylabel     = "Latitude [°]",
#    xlabelsize = 18,
#    ylabelsize = 18,
#    titlesize  = 16)
# ax1.limits[] = ((100, 320), (0, 80))
# vecs_o1 = Vec2f.(Fuo[n1:nn,1] .* scale_mode1, Fvo[n1:nn,1] .* scale_mode1)
# vecs_a1 = Vec2f.(Float32.(Fua[n1:nn,1] .* scale_mode1), Float32.(Fva[n1:nn,1] .* scale_mode1))
# arrows!(ax1, mooring_pos[n1:nn], vecs_o1; color = :black,   arrowsize = 12, linewidth = 7)
# arrows!(ax1, mooring_pos[n1:nn], vecs_a1; color = :magenta, arrowsize = 12, linewidth = 3)
# scale_len1 = scale_ref_kWm1 * scale_mode1
# lines!(ax1, [scale_x0, scale_x0 + scale_len1], [scale_y0, scale_y0]; color = :black, linewidth = 2.5)
# arrows!(ax1, [Point2f(scale_x0, scale_y0)], [Vec2f(scale_len1, 0f0)]; color = :black, arrowsize = 7, linewidth = 2.5)
# text!(ax1, scale_x0, scale_y0 -4; text = "$(scale_ref_kWm1) kW/m", fontsize = 14, color = :black)
# # --- Panel 2: Mode 2 ---
# ax2 = Axis(fig[2, 1],
#    #aspect = DataAspect(),
#    title      = "Mode 2 flux (Ansong vs Alford)",
#    xlabel     = "Longitude [°]",
#    ylabel     = " Latitude [°] ",
#    xlabelsize = 18,
#    ylabelsize = 18,
#    titlesize  = 16)
# ax2.limits[] = ((200, 360), (0, 80))
# vecs_o2 = Vec2f.(Fuo[n1:nn,2] .* scale_mode2, Fvo[n1:nn,2] .* scale_mode2)
# vecs_a2 = Vec2f.(Float32.(Fua[n1:nn,2] .* scale_mode2), Float32.(Fva[n1:nn,2] .* scale_mode2))
# arrows!(ax2, mooring_pos[n1:nn], vecs_o2; color = :black,   arrowsize = 12, linewidth = 7)
# arrows!(ax2, mooring_pos[n1:nn], vecs_a2; color = :magenta, arrowsize = 12, linewidth = 3)
# scale_len2 = scale_ref_kWm2 * scale_mode2
# lines!(ax2, [scale_x0, scale_x0 + scale_len2], [scale_y0, scale_y0]; color = :black, linewidth = 2.5)
# arrows!(ax2, [Point2f(scale_x0, scale_y0)], [Vec2f(scale_len2, 0f0)]; color = :black, arrowsize = 7, linewidth = 2.5)
# text!(ax2, scale_x0, scale_y0 - 4; text = "$(scale_ref_kWm2) kW/m", fontsize = 14, color = :black)
# # --- Shared legend ---
# elem_o = LineElement(color = :black,   linewidth = 3)
# elem_a = LineElement(color = :magenta, linewidth = 3)
# Legend(fig[3, 1], [elem_o, elem_a], ["Joseph Ansong ", "Mathew Alford "],
#       orientation = :horizontal, tellwidth = false)
# #colgap!(fig.layout, 1, 2)
# display(fig)
# png_file = joinpath(FIGDIR, "Mooring_Modes1_2_comparison.png")
# save(png_file, fig)
# println("Saved: $png_file")
# ════════════════════════════════════════════════════════════════════════
# 7) 6-panel scatter comparison (Ansong vs Alford) at matched moorings
#    Row 1: Xflux mode 1 | Yflux mode 1 | Magnitude mode 1
#    Row 2: Xflux mode 2 | Yflux mode 2 | Magnitude mode 2
#    Each point is one matched mooring location; x-axis = Ansong (ALL.mat),
#    y-axis = Alford (NC file); dashed line = 1:1 agreement.
# ════════════════════════════════════════════════════════════════════════
function add_scatter_panel!(fig, pos, xdata, ydata, label; color = :dodgerblue)
   ax = Axis(fig[pos...],
       title  = label,
       xlabel = "Ansong",
       ylabel = "Alford",
       aspect = 1,              # force a genuinely SQUARE box so the 1:1 line is a true diagonal
       titlesize = 15,
       xlabelsize = 14,
       ylabelsize = 14)
   scatter!(ax, xdata, ydata; color = color, markersize = 10)
   # 1:1 reference line spanning the data range (with small padding)
   lo = minimum(vcat(xdata, ydata))
   hi = maximum(vcat(xdata, ydata))
   padr = 0.05 * (hi - lo + eps())
   lims = (lo - padr, hi + padr)
   lines!(ax, [lims[1], lims[2]], [lims[1], lims[2]]; color = :black, linestyle = :dash, linewidth = 1.5)
   xlims!(ax, lims...)
   ylims!(ax, lims...)
   # --- summary statistics: correlation, RMSE, mean bias (Alford - Ansong), n ---
   r    = cor(xdata, ydata)
   rmse = sqrt(mean((ydata .- xdata).^2))
   bias = mean(ydata .- xdata)
   n    = length(xdata)
   stat_txt = "r = $(round(r, digits=2))\nRMSE = $(round(rmse, digits=3))\nbias = $(round(bias, digits=3))\nn = $n"
   text!(ax, lims[1] + 0.03*(lims[2]-lims[1]), lims[2] - 0.03*(lims[2]-lims[1]);
       text = stat_txt, align = (:left, :top), fontsize = 11, color = :black)
   return ax
end
fig2 = Figure(resolution = (1500, 950))
# Row 1 — Mode 1
add_scatter_panel!(fig2, (1, 1), Fuo[:,1], Fua[:,1], "Xflux — Mode 1"; color = :steelblue)
add_scatter_panel!(fig2, (1, 2), Fvo[:,1], Fva[:,1], "Yflux — Mode 1"; color = :seagreen)
add_scatter_panel!(fig2, (1, 3), mag_mode1_o, mag_mode1_a, "Magnitude — Mode 1"; color = :darkorange)
# Row 2 — Mode 2
add_scatter_panel!(fig2, (2, 1), Fuo[:,2], Fua[:,2], "Xflux — Mode 2"; color = :steelblue)
add_scatter_panel!(fig2, (2, 2), Fvo[:,2], Fva[:,2], "Yflux — Mode 2"; color = :seagreen)
add_scatter_panel!(fig2, (2, 3), mag_mode2_o, mag_mode2_a, "Magnitude — Mode 2"; color = :darkorange)
Label(fig2[0, 1:3], "Flux comparison at matched mooring locations (Ansong vs Alford)"; fontsize = 20)
display(fig2)
scatter_png = joinpath(FIGDIR, "Mooring_Flux_Scatter_Comparison.png")
save(scatter_png, fig2)
println("Saved: $scatter_png")
# ════════════════════════════════════════════════════════════════════════
# 8) Difference-based grouping: bin each mooring by |Alford - Ansong|
#    for Xflux and Yflux, mode 1 and mode 2, then plot how many moorings
#    fall into each difference range.
# ════════════════════════════════════════════════════════════════════════
# --- per-mooring differences (Alford - Ansong) ---
diff_x1 = Fua[:,1] .- Fuo[:,1]   # Xflux, mode 1
diff_y1 = Fva[:,1] .- Fvo[:,1]   # Yflux, mode 1
diff_x2 = Fua[:,2] .- Fuo[:,2]   # Xflux, mode 2
diff_y2 = Fva[:,2] .- Fvo[:,2]   # Yflux, mode 2
# --- bin edges for |difference| — EDIT THESE to suit your data's scale ---
edges_mode1 = [0.0, 0.05, 0.1, 0.2, Inf]   # for mode-1 variables (larger fluxes)
edges_mode2 = [0.0, 0.02, 0.05, 0.1, Inf]  # for mode-2 variables (smaller fluxes)
function bin_labels(edges)
   labs = String[]
   for k in 1:length(edges)-1
       lo, hi = edges[k], edges[k+1]
       push!(labs, isinf(hi) ? "$(lo)+" : "$(lo)–$(hi)")
   end
   return labs
end
# assign each mooring to a bin index (1..length(edges)-1) based on |diff|
function assign_bins(diffvec, edges)
   n = length(diffvec)
   binidx = zeros(Int, n)
   for i in 1:n
       a = abs(diffvec[i])
       for k in 1:length(edges)-1
           if edges[k] <= a < edges[k+1]
               binidx[i] = k
               break
           end
       end
       if binidx[i] == 0
           binidx[i] = length(edges) - 1   # falls in the open-ended last bin
       end
   end
   return binidx
end
bins_x1 = assign_bins(diff_x1, edges_mode1)
bins_y1 = assign_bins(diff_y1, edges_mode1)
bins_x2 = assign_bins(diff_x2, edges_mode2)
bins_y2 = assign_bins(diff_y2, edges_mode2)
labels_mode1 = bin_labels(edges_mode1)
labels_mode2 = bin_labels(edges_mode2)
# --- counts per bin for each variable ---
counts_x1 = [count(==(k), bins_x1) for k in 1:length(labels_mode1)]
counts_y1 = [count(==(k), bins_y1) for k in 1:length(labels_mode1)]
counts_x2 = [count(==(k), bins_x2) for k in 1:length(labels_mode2)]
counts_y2 = [count(==(k), bins_y2) for k in 1:length(labels_mode2)]
# --- print which mooring (lat/lon) falls in which bin, per variable ---
println("\n--- Xflux Mode 1 |Alford-Ansong| bin per mooring ---")
for i in 1:n_moor
   println("  lat=$(round(lata[i],digits=2)), lon=$(round(lona[i],digits=2)) -> $(labels_mode1[bins_x1[i]])  (diff=$(round(diff_x1[i],digits=3)))")
end
# --- grouped bar chart: 4 panels (Xflux m1, Yflux m1, Xflux m2, Yflux m2) ---
fig3 = Figure(resolution = (1300, 850))
ax_x1 = Axis(fig3[1,1], title = "Xflux Mode 1 — |diff| groups", xlabel = "Difference range (Alford−Ansong)",
   ylabel = "Number of moorings", xticks = (1:length(labels_mode1), labels_mode1), titlesize = 15)
barplot!(ax_x1, 1:length(labels_mode1), counts_x1; color = :steelblue)
ax_y1 = Axis(fig3[1,2], title = "Yflux Mode 1 — |diff| groups", xlabel = "Difference range (Alford−Ansong)",
   ylabel = "Number of moorings", xticks = (1:length(labels_mode1), labels_mode1), titlesize = 15)
barplot!(ax_y1, 1:length(labels_mode1), counts_y1; color = :seagreen)
ax_x2 = Axis(fig3[2,1], title = "Xflux Mode 2 — |diff| groups", xlabel = "Difference range (Alford−Ansong)",
   ylabel = "Number of moorings", xticks = (1:length(labels_mode2), labels_mode2), titlesize = 15)
barplot!(ax_x2, 1:length(labels_mode2), counts_x2; color = :steelblue)
ax_y2 = Axis(fig3[2,2], title = "Yflux Mode 2 — |diff| groups", xlabel = "Difference range (Alford−Ansong)",
   ylabel = "Number of moorings", xticks = (1:length(labels_mode2), labels_mode2), titlesize = 15)
barplot!(ax_y2, 1:length(labels_mode2), counts_y2; color = :seagreen)
Label(fig3[0, 1:2], "Mooring counts grouped by |Alford − Ansong| flux difference"; fontsize = 20)
display(fig3)
diffgroup_png = joinpath(FIGDIR, "Mooring_Flux_Difference_Groups.png")
save(diffgroup_png, fig3)
println("Saved: $diffgroup_png")
# ════════════════════════════════════════════════════════════════════════
# 9) ROTATION TEST: rotate one dataset's flux vectors by a small 5° angle
#    and re-check whether the scatter tightens onto the 1:1 line. Rotating
#    the Alford (NC-file) data here — flip the two arguments below to
#    rotate Ansong (Fuo, Fvo) instead if you want to test that direction.
# ════════════════════════════════════════════════════════════════════════
function rotate_flux(Fu, Fv, theta)
   R = [cos(theta) -sin(theta);
        sin(theta)  cos(theta)]
   Fu_rot = similar(Fu)
   Fv_rot = similar(Fv)
   for i in eachindex(Fu)
       v = R * [Fu[i]; Fv[i]]
       Fu_rot[i] = v[1]
       Fv_rot[i] = v[2]
   end
   return Fu_rot, Fv_rot
end
theta5 = deg2rad(25.0)   # small 5-degree rotation
Fua_rot = similar(Fua)
Fva_rot = similar(Fva)
Fua_rot[:,1], Fva_rot[:,1] = rotate_flux(Fua[:,1], Fva[:,1], theta5)   # mode 1
Fua_rot[:,2], Fva_rot[:,2] = rotate_flux(Fua[:,2], Fva[:,2], theta5)   # mode 2
mag_mode1_a_rot = sqrt.(Fua_rot[:,1].^2 .+ Fva_rot[:,1].^2)   # magnitude is rotation-invariant, kept for symmetry
mag_mode2_a_rot = sqrt.(Fua_rot[:,2].^2 .+ Fva_rot[:,2].^2)
# --- 6-panel scatter, same layout as fig2, but with rotated Alford data ---
fig4 = Figure(resolution = (1500, 950))
add_scatter_panel!(fig4, (1, 1), Fuo[:,1], Fua_rot[:,1], "Xflux — Mode 1 (15° rot)"; color = :steelblue)
add_scatter_panel!(fig4, (1, 2), Fvo[:,1], Fva_rot[:,1], "Yflux — Mode 1 (15° rot)"; color = :seagreen)
add_scatter_panel!(fig4, (1, 3), mag_mode1_o, mag_mode1_a_rot, "Magnitude — Mode 1 (15° rot)"; color = :darkorange)
add_scatter_panel!(fig4, (2, 1), Fuo[:,2], Fua_rot[:,2], "Xflux — Mode 2 (15° rot)"; color = :steelblue)
add_scatter_panel!(fig4, (2, 2), Fvo[:,2], Fva_rot[:,2], "Yflux — Mode 2 (15° rot)"; color = :seagreen)
add_scatter_panel!(fig4, (2, 3), mag_mode2_o, mag_mode2_a_rot, "Magnitude — Mode 2 (15° rot)"; color = :darkorange)
Label(fig4[0, 1:3], "Flux comparison after rotating Alford data by 15° (Ansong vs rotated Alford)"; fontsize = 20)
display(fig4)
scatter_rot_png = joinpath(FIGDIR, "Mooring_Flux_Scatter_Comparison_5deg_rotated.png")
save(scatter_rot_png, fig4)
println("Saved: $scatter_rot_png")
# ════════════════════════════════════════════════════════════════════════
# 10) Difference-group bar chart recomputed with the ROTATED Alford data
# ════════════════════════════════════════════════════════════════════════
diff_x1_rot = Fua_rot[:,1] .- Fuo[:,1]
diff_y1_rot = Fva_rot[:,1] .- Fvo[:,1]
diff_x2_rot = Fua_rot[:,2] .- Fuo[:,2]
diff_y2_rot = Fva_rot[:,2] .- Fvo[:,2]
bins_x1_rot = assign_bins(diff_x1_rot, edges_mode1)
bins_y1_rot = assign_bins(diff_y1_rot, edges_mode1)
bins_x2_rot = assign_bins(diff_x2_rot, edges_mode2)
bins_y2_rot = assign_bins(diff_y2_rot, edges_mode2)
counts_x1_rot = [count(==(k), bins_x1_rot) for k in 1:length(labels_mode1)]
counts_y1_rot = [count(==(k), bins_y1_rot) for k in 1:length(labels_mode1)]
counts_x2_rot = [count(==(k), bins_x2_rot) for k in 1:length(labels_mode2)]
counts_y2_rot = [count(==(k), bins_y2_rot) for k in 1:length(labels_mode2)]
fig5 = Figure(resolution = (1300, 850))
ax_x1r = Axis(fig5[1,1], title = "Xflux Mode 1 — |diff| groups (5° rot)", xlabel = "Difference range (Alford−Ansong)",
   ylabel = "Number of moorings", xticks = (1:length(labels_mode1), labels_mode1), titlesize = 15)
barplot!(ax_x1r, 1:length(labels_mode1), counts_x1_rot; color = :steelblue)
ax_y1r = Axis(fig5[1,2], title = "Yflux Mode 1 — |diff| groups (5° rot)", xlabel = "Difference range (Alford−Ansong)",
   ylabel = "Number of moorings", xticks = (1:length(labels_mode1), labels_mode1), titlesize = 15)
barplot!(ax_y1r, 1:length(labels_mode1), counts_y1_rot; color = :seagreen)
ax_x2r = Axis(fig5[2,1], title = "Xflux Mode 2 — |diff| groups (5° rot)", xlabel = "Difference range (Alford−Ansong)",
   ylabel = "Number of moorings", xticks = (1:length(labels_mode2), labels_mode2), titlesize = 15)
barplot!(ax_x2r, 1:length(labels_mode2), counts_x2_rot; color = :steelblue)
ax_y2r = Axis(fig5[2,2], title = "Yflux Mode 2 — |diff| groups (5° rot)", xlabel = "Difference range (Alford−Ansong)",
   ylabel = "Number of moorings", xticks = (1:length(labels_mode2), labels_mode2), titlesize = 15)
barplot!(ax_y2r, 1:length(labels_mode2), counts_y2_rot; color = :seagreen)
Label(fig5[0, 1:2], "Mooring counts grouped by |Alford − Ansong| flux difference (5° rotation applied)"; fontsize = 20)
display(fig5)
diffgroup_rot_png = joinpath(FIGDIR, "Mooring_Flux_Difference_Groups_5deg_rotated.png")
save(diffgroup_rot_png, fig5)
println("Saved: $diffgroup_rot_png")




