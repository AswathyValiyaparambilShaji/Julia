using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML
using CairoMakie


sci_formatter(x) = x -> @sprintf("%.1e", x)


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter


config_file = get(ENV, "JULIA_CONFIG",
                joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)


base  = cfg["base_path"]
base2 = cfg["base_path2"]


# --- Domain & grid ---
NX, NY = 288, 468
minlat, maxlat = 24.0, 31.91
minlon, maxlon = 193.0, 199.0
lat = range(minlat, maxlat, length=NY)
lon = range(minlon, maxlon, length=NX)


# --- Tile & time ---
buf = 3
tx, ty = 47, 66
nx = tx + 2*buf
ny = ty + 2*buf
nz = 88


dt  = 25
dto = 144
Tts = 366192
nt  = div(Tts, dto)


ts      = 72      # 3-day window
nt_avg = div(nt, ts)


# --- Thickness & constants ---
thk = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)


rho0 = 999.8


# --- Problematic points ---
# [i, j] format
high_point = [8, 65]       # High APE
low_pos_point = [7, 24]    # Low positive APE
low_neg_point = [21, 12]   # Low negative APE


points_list = [high_point, low_pos_point, low_neg_point]
point_labels = ["High APE (8,65)", "Low Positive APE (7,24)", "Low Negative APE (21,12)"]


# Select which tile to analyze (adjust based on your xn, yn)
xn = cfg["xn_start"]
yn = cfg["yn_start"]
suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)


# --- Read N2 (3-day averaged) ---
println("Reading N2...")
N2_phase = open(joinpath(base,"3day_mean","N2","N2_3day_$suffix.bin"), "r") do io
    raw = read(io, nx * ny * nz * nt_avg * sizeof(Float64))
    reshape(reinterpret(Float64, raw), nx, ny, nz, nt_avg)
end


# --- Adjust N2 to interfaces ---
N2_adjusted = zeros(Float64, nx, ny, nz+1, nt_avg)
N2_adjusted[:, :, 1,   :] = N2_phase[:, :, 1,   :]
N2_adjusted[:, :, 2:nz,:] = N2_phase[:, :, 1:nz-1, :]
N2_adjusted[:, :, nz+1,:] = N2_phase[:, :, nz-1, :]


# --- Average to cell centers ---
N2_center = zeros(Float64, nx, ny, nz, nt_avg)
for k in 1:nz
    N2_center[:, :, k, :] .= 0.5 .* (N2_adjusted[:, :, k, :] .+ N2_adjusted[:, :, k+1, :])
end


# --- Read hFacC ---
println("Reading hFacC...")
hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))


# --- Thickness ---
DRFfull = hFacC .* DRF3d
DRFfull[hFacC .== 0] .= NaN


# Calculate depth at cell centers
z = cumsum(DRFfull, dims=3)
zz = cat(zeros(nx, ny, 1), z; dims=3)
z_center = -0.5 .* (zz[:, :, 1:end-1] .+ zz[:, :, 2:end])


# --- Read buoyancy ---
println("Reading buoyancy...")
b = open(joinpath(base2, "b", "b_t_sm_$suffix.bin"), "r") do io
    raw = read(io, nx * ny * nz * nt * sizeof(Float64))
    reshape(reinterpret(Float64, raw), nx, ny, nz, nt)
end


# --- Read APE ---
println("Reading APE...")
APE = open(joinpath(base2, "APE", "APE_t_sm_$suffix.bin"), "r") do io
    raw = read(io, nx * ny * nz * nt * sizeof(Float64))
    reshape(reinterpret(Float64, raw), nx, ny, nz, nt)
end


# Select a time snapshot for analysis
t_snap = 1  # You can change this


# Map time index to N2 time index
t_n2 = div(t_snap - 1, ts) + 1


println("Creating figure...")


# --- Create the figure ---
fig = Figure(size = (2400, 1200)) #, figure_padding = (5,5,5,5))


# Remove all margins and adjust spacing
#fig.layout.padding = (0, 0, 0, 0)
#rowgap!(fig.layout, 10)
#colgap!(fig.layout, 10)


# Column titles
Label(fig[0, 1], "N²", fontsize = 20, font = :bold)
Label(fig[0, 2], "b²", fontsize = 20, font = :bold)
Label(fig[0, 3], "b²/N²", fontsize = 20, font = :bold)
Label(fig[0, 4], "APE", fontsize = 20, font = :bold)


# Colors for the 5 points
colors = [:red, :blue, :green, :orange, :purple]
point_names = ["Center", "Left (i-1)", "Right (i+1)", "Down (j-1)", "Up (j+1)"]


# Loop through the 3 problematic points
for (row_idx, (center_point, row_label)) in enumerate(zip(points_list, point_labels))
    i_c, j_c = center_point
    
    # Row label
    Label(fig[row_idx, 0], row_label, fontsize = 16, rotation = π/2)
    
    # Get 5 points: center and 4 neighbors
    neighbor_points = [
        [i_c, j_c],         # center
        [i_c-1, j_c],       # left
        [i_c+1, j_c],       # right
        [i_c, j_c-1],       # down
        [i_c, j_c+1]        # up
    ]
    
    # Column 1: N²
   
    

    ax1 = Axis(fig[row_idx, 1],
        xlabel = row_idx == 3 ? "N² (s⁻²)" : "",
        ylabel = "Depth (m)",
        xticklabelsvisible = (row_idx == 3),
        yticklabelsvisible = true,
        xtickformat = sci_formatter,
        xticks = WilkinsonTicks(4),
    )

    
    for (pt_idx, pt) in enumerate(neighbor_points)
        i, j = pt
        if i >= 1 && i <= nx && j >= 1 && j <= ny
            n2_prof = N2_center[i, j, :, t_n2]
            z_prof = z_center[i, j, :]
            lines!(ax1, n2_prof, z_prof, 
                  linewidth = 2, 
                  color = colors[pt_idx],
                  label = point_names[pt_idx])
        end
    end
    if row_idx == 1
        axislegend(ax1, position = :rb, framevisible = true)
    end
    
    # Column 2: b²
    

    ax2 = Axis(fig[row_idx, 2],
        xlabel = row_idx == 3 ? "b² (m² s⁻⁴)" : "",
        ylabel = "",
        xticklabelsvisible = (row_idx == 3),
        yticklabelsvisible = false,
        xtickformat = sci_formatter,
        xticks = WilkinsonTicks(4),
    )

    
    for (pt_idx, pt) in enumerate(neighbor_points)
        i, j = pt
        if i >= 1 && i <= nx && j >= 1 && j <= ny
            b_prof = b[i, j, :, t_snap]
            b2_prof = b_prof.^2
            z_prof = z_center[i, j, :]
            lines!(ax2, b2_prof, z_prof, 
                  linewidth = 2, 
                  color = colors[pt_idx])
        end
    end
    
    # Column 3: b²/N²
        ax3 = Axis(fig[row_idx, 3],
                xlabel = row_idx == 3 ? "b² / N² (m² s⁻²)" : "",
                ylabel = "",
                xticklabelsvisible = (row_idx == 3),
                yticklabelsvisible = false,
                xtickformat = sci_formatter,
                xticks = WilkinsonTicks(4),
            )

    
    for (pt_idx, pt) in enumerate(neighbor_points)
        i, j = pt
        if i >= 1 && i <= nx && j >= 1 && j <= ny
            b_prof = b[i, j, :, t_snap]
            n2_prof = N2_center[i, j, :, t_n2]
            ratio_prof = (b_prof.^2) ./ n2_prof
            z_prof = z_center[i, j, :]
            lines!(ax3, ratio_prof, z_prof, 
                  linewidth = 2, 
                  color = colors[pt_idx])
        end
    end
    
    # Column 4: APE
    ax4 = Axis(fig[row_idx, 4],
            xlabel = row_idx == 3 ? "APE (m² s⁻²)" : "",
            ylabel = "",
            xticklabelsvisible = (row_idx == 3),
            yticklabelsvisible = false,
            xtickformat = sci_formatter,
            xticks = WilkinsonTicks(4),
    )

    
    for (pt_idx, pt) in enumerate(neighbor_points)
        i, j = pt
        if i >= 1 && i <= nx && j >= 1 && j <= ny
            ape_prof = APE[i, j, :, t_snap]
            z_prof = z_center[i, j, :]
            lines!(ax4, ape_prof, z_prof, 
                  linewidth = 2, 
                  color = colors[pt_idx])
        end
    end
end

resize_to_layout!(fig)
display(fig)


# Save the figure
save(joinpath(base2, "APE_profiles_analysis.png"), fig, px_per_unit = 2)
println("Figure saved!")


# Print some diagnostics for each problematic point
for (center_point, label) in zip(points_list, point_labels)
    i_c, j_c = center_point
    println("\n=== $label ===")
    println("N² at center: ", extrema(filter(isfinite, N2_center[i_c, j_c, :, t_n2])))
    println("b² at center: ", extrema(filter(isfinite, b[i_c, j_c, :, t_snap].^2)))
    println("APE at center: ", extrema(filter(isfinite, APE[i_c, j_c, :, t_snap])))
    println("Depth-integrated APE: ", sum(filter(isfinite, APE[i_c, j_c, :, t_snap] .* DRFfull[i_c, j_c, :])))
end

using CairoMakie
using Printf


# Define points
points_list = [
   [8, 65],   # High APE
   [7, 24],   # Low Positive APE
   [21, 12]   # Low Negative APE
]
point_labels = ["High APE (8,65)", "Low Positive APE (7,24)", "Low Negative APE (21,12)"]


# Colors for the 5 points
colors = [:red, :blue, :green, :orange, :purple]
neighbor_names = ["Center", "Left (i-1)", "Right (i+1)", "Down (j-1)", "Up (j+1)"]


# Create figure with 1 row and 3 columns
fig = Figure(size = (1800, 600))


# Loop over the 3 points (3 columns)
for col_idx in 1:3
    i_c, j_c = points_list[col_idx]
    
    # Get 5 neighboring points
    neighbor_points = [
        [i_c, j_c],         # center
        [i_c-1, j_c],       # left
        [i_c+1, j_c],       # right
        [i_c, j_c-1],       # down
        [i_c, j_c+1]        # up
    ]
    
    # Create axis
    ax = Axis(fig[1, col_idx],
        xlabel = "N² (s⁻²)",
        ylabel = col_idx == 1 ? "Depth (m)" : "",
        title = point_labels[col_idx],
        yticklabelsvisible = (col_idx == 1)
    )
    
    # Plot all 5 profiles
    for (pt_idx, pt) in enumerate(neighbor_points)
        i, j = pt
        if i >= 1 && i <= nx && j >= 1 && j <= ny
            n2_prof = N2_center[i, j, :, t_n2]
            z_prof = z_center[i, j, :]
            lines!(ax, n2_prof, z_prof, 
                  linewidth = 2, 
                  color = colors[pt_idx],
                  label = neighbor_names[pt_idx])
        end
    end
    
    # Add legend only to first plot
    if col_idx == 1
        axislegend(ax, position = :rb, framevisible = true)
    end
end


display(fig)
save("N2_profiles_simple.png", fig, px_per_unit = 2)
println("Figure saved!")

using CairoMakie
using Printf


# Define points
points_list = [
   [25, 35],   # High APE
   [7, 24],   # Low Positive APE
   [21, 12]   # Low Negative APE
]
point_labels = ["High APE (8,65)", "Low Positive APE (7,24)", "Low Negative APE (21,12)"]


# Create figure with 1 row and 3 columns
fig = Figure(size = (1800, 600))


# Loop over the 3 points (3 columns)
for col_idx in 1:3
    i_c, j_c = points_list[col_idx]
    
    # Get N²_phase and N²_center profiles
    n2_phase_prof = N2_phase[i_c, j_c, :, t_n2]
    n2_center_prof = N2_center[i_c, j_c, :, t_n2]
    z_prof = z_center[i_c, j_c, :]
    
    # Create axis
    ax = Axis(fig[1, col_idx],
        xlabel = "N² (s⁻²)",
        ylabel = col_idx == 1 ? "Depth (m)" : "",
        title = point_labels[col_idx],
        yticklabelsvisible = (col_idx == 1)
    )
    
    # Plot both profiles
    lines!(ax, n2_phase_prof, z_prof, 
          linewidth = 2, 
          color = :blue,
          label = "N² phase")
    
    lines!(ax, n2_center_prof, z_prof, 
          linewidth = 2, 
          color = :red,
          label = "N² center")
    
    # Add legend only to first plot
    if col_idx == 1
        axislegend(ax, position = :rb, framevisible = true)
    end
end


display(fig)
save("N2_phase_vs_center.png", fig, px_per_unit = 2)
println("Figure saved!")







