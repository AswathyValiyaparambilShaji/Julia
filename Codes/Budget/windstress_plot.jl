using Printf, FilePathsBase, TOML, CairoMakie, Statistics, LinearAlgebra


# Include FluxUtils.jl
include(joinpath(@__DIR__, "..", "..","functions", "FluxUtils.jl"))
using .FluxUtils: read_bin


# Read the configuration file for paths
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..",  "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base = cfg["base_path"]
base2 = cfg["base_path2"]


# --- TIME AVERAGING CONFIGURATION ---
use_3day_movie = true  # Set to true for 3-day movie, false for single time average plot


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
dt = 25
dto = 144
Tts = 366192
nt = div(Tts, dto)
nt3 = div(nt, 3*24)  # Number of 3-day periods


println("Tile dimensions: nx=$nx, ny=$ny")
println("Total time steps: $nt")
println("Number of 3-day periods: $nt3")


# Check file size for first tile
test_file = joinpath(base, "Windstress", @sprintf("taux_%02dx%02d_%d.bin", cfg["xn_start"], cfg["yn_start"], buf))
if isfile(test_file)
    filesize_bytes = stat(test_file).size
    n_elements = filesize_bytes ÷ sizeof(Float32)
    expected_elements = nx * ny * nt
    println("\nFile check:")
    println("  File: $(basename(test_file))")
    println("  Size: $filesize_bytes bytes")
    println("  Elements in file: $n_elements")
    println("  Expected (nx×ny×nt): $expected_elements")
    
    if n_elements != expected_elements
        println("  WARNING: Size mismatch!")
        # Calculate what dimensions would work
        actual_nt = n_elements ÷ (nx * ny)
        println("  Actual time steps in file: $actual_nt")
        nt = actual_nt  # Update nt to match file
        nt3 = div(nt, 72)  # 3 days = 72 hours
        println("  Adjusted nt3 (3-day periods): $nt3")
    end
else
    println("Test file not found: $test_file")
end


# ============================================================================
# MAIN WORKFLOW: Load wind stress for all 3-day periods
# ============================================================================


if use_3day_movie
    println("\nLoading wind stress for $nt3 3-day periods...")
    
    # Initialize 3D arrays for all time periods
    TauX_3day = zeros(NX, NY, nt3)
    TauY_3day = zeros(NX, NY, nt3)
    
    for xn in cfg["xn_start"]:cfg["xn_end"]
        for yn in cfg["yn_start"]:cfg["yn_end"]
            suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
            
            println("Reading tile $suffix...")
            
            # Read all time steps for this tile
            taux_all = Float64.(open(joinpath(base, "Windstress", "taux_$suffix.bin"), "r") do io
                nbytes = nx * ny * nt * sizeof(Float32)
                raw_bytes = read(io, nbytes)
                raw_data = reinterpret(Float32, raw_bytes)
                reshape(raw_data, nx, ny, nt)
            end)
            
            tauy_all = Float64.(open(joinpath(base1, "Windstress", "tauy_$suffix.bin"), "r") do io
                nbytes = nx * ny * nt * sizeof(Float32)
                raw_bytes = read(io, nbytes)
                raw_data = reinterpret(Float32, raw_bytes)
                reshape(raw_data, nx, ny, nt)
            end)
            
            println("  Shape: $(size(taux_all))")
            
            # Compute 3-day averages
            for t3 in 1:nt3
                # Each 3-day period spans 72 time steps (3 days * 24 hours)
                t_start = (t3 - 1) * 72 + 1
                t_end = min(t3 * 72, nt)
                
                # Average over this 3-day period
                taux_tile = mean(taux_all[:, :, t_start:t_end], dims=3)[:, :, 1]
                tauy_tile = mean(tauy_all[:, :, t_start:t_end], dims=3)[:, :, 1]
                
                # Add last values for proper centering from Arakawa C-grid
                taux_extended = zeros(nx+1, ny)
                taux_extended[1:nx, :] .= taux_tile
                taux_extended[end, :] .= taux_tile[end, :]
                
                tauy_extended = zeros(nx, ny+1)
                tauy_extended[:, 1:ny] .= tauy_tile
                tauy_extended[:, end] .= tauy_tile[:, end]
                
                # Average to cell centers
                taux_centered = 0.5 .* (taux_extended[1:end-1, :] .+ taux_extended[2:end, :])
                tauy_centered = 0.5 .* (tauy_extended[:, 1:end-1] .+ tauy_extended[:, 2:end])
                
                # Calculate tile positions
                xs = (xn - 1) * tx + 1
                xe = xs + tx + (2 * buf) - 1
                ys = (yn - 1) * ty + 1
                ye = ys + ty + (2 * buf) - 1
                
                # Extract interior (remove buffer)
                taux_interior = taux_centered[buf+1:nx-buf, buf+1:ny-buf]
                tauy_interior = tauy_centered[buf+1:nx-buf, buf+1:ny-buf]
                
                # Assign to global arrays
                TauX_3day[xs+1:xe-buf-1, ys+1:ye-buf-1, t3] .= taux_interior
                TauY_3day[xs+1:xe-buf-1, ys+1:ye-buf-1, t3] .= tauy_interior
            end
            
            println("  Completed tile $suffix")
        end
    end
    
    # ============================================================================
    # CREATE MOVIE
    # ============================================================================
    
    println("\nCreating $nt3 frames for movie...")
    
    FIGDIR = cfg["fig_base"]
    mkpath(FIGDIR)
    frames_dir = joinpath(FIGDIR, "windstress_frames")
    mkpath(frames_dir)
    
    QUIVER_STEP = 8
    ARROW_SCALEUP = 5.0
    
    # Calculate global max for consistent color scale
    all_mags = [sqrt.(TauX_3day[:, :, t].^2 .+ TauY_3day[:, :, t].^2) for t in 1:nt3]
    HEAT_CBAR_MAX = maximum(maximum.(all_mags))
    
    println("Color range: 0 to $HEAT_CBAR_MAX N/m²")
    
    for t in 1:nt3
        # Calculate magnitude for this time period
        Tau_mag = sqrt.(TauX_3day[:, :, t].^2 .+ TauY_3day[:, :, t].^2)
        
        fig = Figure(size=(700, 600))
        ax = Axis(fig[1, 1],
            title = "Wind Stress - 3-Day Period $t/$nt3",
            xlabel = "Longitude [°]",
            ylabel = "Latitude [°]",
            ylabelsize = 22,
            xlabelsize = 22,
            titlesize = 26
        )
        
        # Heatmap of magnitude
        hm = heatmap!(ax, lon, lat, Tau_mag',
            interpolate = false,
            colorrange = (0, HEAT_CBAR_MAX),
            colormap = :thermal
        )
        
        # Add arrows
        pos = Point2f[]
        vec = Vec2f[]
        
        for i in 1:QUIVER_STEP:NX, j in 1:QUIVER_STEP:NY
            u = TauX_3day[i, j, t]
            v = TauY_3day[i, j, t]
            m = Tau_mag[i, j]
            if isfinite(u) && isfinite(v) && isfinite(m)
                push!(pos, Point2f(Float32(lon[i]), Float32(lat[j])))
                push!(vec, Vec2f(Float32(u), Float32(v)))
            end
        end
        
        if !isempty(vec)
            maxmag = maximum(norm, vec)
            cell_x = (maximum(lon) - minimum(lon)) / NX
            cell_y = (maximum(lat) - minimum(lat)) / NY
            target = 5f0 * Float32(min(cell_x, cell_y))
            scale = maxmag == 0 ? 1f0 : (target / maxmag) * Float32(ARROW_SCALEUP)
            arrows!(ax, pos, scale .* vec, color=:white, arrowsize=10, linewidth=1.5)
        end
        
        Colorbar(fig[1, 2], hm, label = "Wind Stress [N/m²]")
        
        save(joinpath(frames_dir, @sprintf("frame_%04d.png", t)), fig)
        
        if t % 5 == 0
            println("  Progress: $t/$nt3 frames")
        end
    end
    
    println("Frames saved to: $frames_dir")
    
    # Create video
    movie_file = joinpath(FIGDIR, "WindStress_3day_movie.mp4")
    input_pattern = joinpath(frames_dir, "frame_%04d.png")
    
    try
        run(`ffmpeg -y -framerate 5 -i $input_pattern -c:v libx264 -pix_fmt yuv420p -crf 23 $movie_file`)
        println("Video created: $movie_file")
        rm(frames_dir, recursive=true)
        println("Frames cleaned up")
    catch e
        println("\nffmpeg error:")
        println(e)
        println("\nFrames kept in: $frames_dir")
        println("Try running manually:")
        println("ffmpeg -y -framerate 5 -i $input_pattern -c:v libx264 -pix_fmt yuv420p $movie_file")
    end
    
else
    println("Set use_3day_movie = true to create movie")
end


println("\nDone!")




