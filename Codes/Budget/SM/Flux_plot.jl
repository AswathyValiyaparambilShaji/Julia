using DSP, MAT, Statistics, Printf,  FilePathsBase, LinearAlgebra, TOML
#using CairoMakie, SparseArrays




include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base = cfg["base_path"]
base2 = cfg["base_path2"]


# --- TIME AVERAGING CONFIGURATION ---
# Set to true for 3-day averaging, false for full time period averaging
use_3day = true  # Change this to true for 3-day averaging


mkpath(joinpath(base2,"xflux"))
mkpath(joinpath(base2, "yflux"))
mkpath(joinpath(base2, "zflux"))




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




kz = 1
dt = 25
dto = 144
Tts = 366192
nt = div(Tts, dto)
nt3 = div(nt, 3*24)  # Number of 3-day periods




# --- Thickness & constants ---
thk = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)
g = 9.8




# ============================================================================
# MAIN WORKFLOW SPLIT: 3-DAY vs FULL TIME AVERAGE
# ============================================================================


if use_3day
    # ========================================================================
    # 3-DAY AVERAGING WORKFLOW
    # ========================================================================
    println("Using 3-day averaged files with $nt3 time periods")
    
    # Initialize 3D arrays for time-varying 3-day averages
    tfx = zeros(NX, NY, nt3)
    tfy = zeros(NX, NY, nt3)
    
    for xn in cfg["xn_start"]:cfg["xn_end"]
        for yn in cfg["yn_start"]:cfg["yn_end"]
            suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
            
            hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
            
            # Read 3-day averaged files (4D: nx, ny, nz, nt3)
            fx = Float64.(open(joinpath(base2, "xflux", "xflx_3day_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * nt3 * sizeof(Float32)
                raw_bytes = read(io, nbytes)
                raw_data = reinterpret(Float32, raw_bytes)
                reshaped_data = reshape(raw_data, nx, ny, nz, nt3)
            end)
            
            fy = Float64.(open(joinpath(base2, "yflux", "yflx_3day_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * nt3 * sizeof(Float32)
                raw_bytes = read(io, nbytes)
                raw_data = reinterpret(Float32, raw_bytes)
                reshaped_data = reshape(raw_data, nx, ny, nz, nt3)
            end)
            
            # DRFfull and depth integration
            DRFfull = hFacC .* DRF3d
            fxX = sum(fx .* DRFfull, dims=3)  # Result: (nx, ny, 1, nt3)
            fyY = sum(fy .* DRFfull, dims=3)
            
            # Assign to full flux arrays
            xs = (xn - 1) * tx + 1 
            xe = xs + tx + (2 * buf) - 1 
            ys = (yn - 1) * ty + 1 
            ye = ys + ty + (2 * buf) - 1 
            
            xsf = 2
            xef = tx + (2*buf) - 1
            ysf = 2
            yef = ty + (2*buf) - 1
            
            tfx[xs+1:xe-1, ys+1:ye-1, :] .= fxX[xsf:xef, ysf:yef, 1, :]
            tfy[xs+1:xe-1, ys+1:ye-1, :] .= fyY[xsf:xef, ysf:yef, 1, :]
        end
    end
    
    println("Flux assembly complete for $nt3 3-day periods")
    
else
    # ========================================================================
    # FULL TIME AVERAGE WORKFLOW
    # ========================================================================
    println("Using full time averaged files")
    
    # Initialize 2D arrays for single time average
    tfx = zeros(NX, NY)
    tfy = zeros(NX, NY)
    
    for xn in cfg["xn_start"]:cfg["xn_end"]
        for yn in cfg["yn_start"]:cfg["yn_end"]
            suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
            
            hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
            
            # Read full time averaged files (3D: nx, ny, nz)
            fx = Float64.(open(joinpath(base2, "xflux", "xflx_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * sizeof(Float32)
                raw_bytes = read(io, nbytes)
                raw_data = reinterpret(Float32, raw_bytes)
                reshaped_data = reshape(raw_data, nx, ny, nz)
            end)
            
            fy = Float64.(open(joinpath(base2, "yflux", "yflx_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * sizeof(Float32)
                raw_bytes = read(io, nbytes)
                raw_data = reinterpret(Float32, raw_bytes)
                reshaped_data = reshape(raw_data, nx, ny, nz)
            end)
            
            # DRFfull and depth integration
            DRFfull = hFacC .* DRF3d
            fxX = sum(fx .* DRFfull, dims=3)  # Result: (nx, ny, 1)
            fyY = sum(fy .* DRFfull, dims=3)
            
            # Assign to full flux arrays
            xs = (xn - 1) * tx + 1 
            xe = xs + tx + (2 * buf) - 1 
            ys = (yn - 1) * ty + 1 
            ye = ys + ty + (2 * buf) - 1 
            
            xsf = 2
            xef = tx + (2*buf) - 1
            ysf = 2
            yef = ty + (2*buf) - 1
            
            tfx[xs+1:xe-1, ys+1:ye-1] .= fxX[xsf:xef, ysf:yef, 1]
            tfy[xs+1:xe-1, ys+1:ye-1] .= fyY[xsf:xef, ysf:yef, 1]
        end
    end
    
    println("Flux assembly complete")
    #println(tfx[1:tx, 10])
    
end




# ============================================================================
# PLOTTING
# ============================================================================
using CairoMakie


# --- Config ---
FIGDIR        = cfg["fig_base"]
HEAT_CBAR_MAX = 15      # kW/m
QUIVER_STEP   = 20      # grid stride for arrows
ARROW_SCALEUP = 5.0
DO_TRANSPOSE  = false   # set true if your image appears rotated/flipped


isdir(FIGDIR) || mkpath(FIGDIR)


if use_3day
    # ========================================================================
    # 3-DAY PLOTTING: Generate MOVIE from 3-day periods
    # ========================================================================
    println("Creating movie from $nt3 3-day periods...")
    
    # Create temporary directory for frames
    frames_dir = joinpath(FIGDIR, "temp_frames")
    mkpath(frames_dir)
    
    # Generate all frames
    for t in 1:nt3
        # ABSOLUTE FLUX
        fm = sqrt.(tfx[:, :, t].^2 .+ tfy[:, :, t].^2)
        
        # Prep (units & orientation)
        fm_kW = fm ./ 1000  # convert to kW/m
        F  = DO_TRANSPOSE ? fm_kW' : fm_kW
        Ux = DO_TRANSPOSE ? tfx[:, :, t]' : tfx[:, :, t]
        Uy = DO_TRANSPOSE ? tfy[:, :, t]' : tfy[:, :, t]
        
        LON = lon
        LAT = lat
        
        # Create figure
        fig = Figure(resolution = (700, 600))
        ax = Axis(fig[1, 1],
            title  = "MITgcm Flux - 3-Day Period $t/$nt3",
            xlabel = "Longitude [°]",
            ylabel = "Latitude [°]",
            ylabelsize = 22,
            xlabelsize = 22,
            titlesize = 26,
        )
        
        hm = CairoMakie.heatmap!(ax, LON, LAT, F;
            interpolate = false,
            colorrange  = (0, HEAT_CBAR_MAX),
            colormap    = :Spectral_9,
        )
        
        # Build quiver subsample
        pos = Point2f[]
        vec = Vec2f[]
        NX_local, NY_local = size(F)
        for i in 1:QUIVER_STEP:NX_local, j in 1:QUIVER_STEP:NY_local
            u = Ux[i, j]; v = Uy[i, j]; m = F[i, j]
            if isfinite(u) && isfinite(v) && isfinite(m)
                push!(pos, Point2f(Float32(LON[i]), Float32(LAT[j])))
                push!(vec, Vec2f(Float32(u), Float32(v)))
            end
        end
        
        if !isempty(vec)
            maxmag = maximum(norm, vec)
            cell_x = (maximum(LON) - minimum(LON)) / NX_local
            cell_y = (maximum(LAT) - minimum(LAT)) / NY_local
            target = 5f0 * Float32(min(cell_x, cell_y))
            scale  = maxmag == 0 ? 1f0 : (target / maxmag) * Float32(ARROW_SCALEUP)
            
            arrows!(ax, pos, scale .* vec; color=:black, shaftwidth=1.5f0, tipwidth=8f0, tiplength=8f0)
        end
        
        Colorbar(fig[1, 2], hm, label = "(kW/m)")
        
        # Save frame
        frame_file = joinpath(frames_dir, @sprintf("frame_%04d.png", t))
        save(frame_file, fig)
        println("  Generated frame $t/$nt3")
    end
    
    # Create movie using ffmpeg
    movie_file = joinpath(FIGDIR, "Flux_3day_movie.mp4")
    try
        run(`ffmpeg -y -framerate 5 -i $(frames_dir)/frame_%04d.png -c:v libx264 -pix_fmt yuv420p -crf 23 $movie_file`)
        println("\n✓ Movie saved: $movie_file")
        
        # Clean up frames
        rm(frames_dir, recursive=true)
        println("✓ Temporary frames cleaned up")
    catch e
        println("\n⚠ ffmpeg not available. Frames saved in: $frames_dir")
        println("  You can create the movie manually with:")
        println("  ffmpeg -framerate 5 -i $(frames_dir)/frame_%04d.png -c:v libx264 -pix_fmt yuv420p $movie_file")
    end
    
else
    # ========================================================================
    # FULL TIME AVERAGE PLOTTING: Single PNG figure
    # ========================================================================
    println("Creating single PNG figure for full time average...")
    
    # ABSOLUTE FLUX
    fm = sqrt.(tfx.^2 .+ tfy.^2)
    
    # Prep (units & orientation)
    fm_kW = fm ./ 1000  # convert to kW/m
    F  = DO_TRANSPOSE ? fm_kW' : fm_kW
    Ux = DO_TRANSPOSE ? tfx'   : tfx
    Uy = DO_TRANSPOSE ? tfy'   : tfy
    
    LON = lon
    LAT = lat
    
    # Create figure
    fig1 = Figure(resolution = (700, 600))
    ax1  = Axis(fig1[1, 1],
        title  = "MITgcm Flux ",
        xlabel = "Longitude [°]",
        ylabel = "Latitude [°]",
        ylabelsize = 22,
        xlabelsize = 22,
        titlesize = 26,
    )
    
    hm1 = CairoMakie.heatmap!(ax1, LON, LAT, F;
        interpolate = false,
        colorrange  = (0, HEAT_CBAR_MAX),
        colormap    = :Spectral_9,
    )
    
    # Build quiver subsample
    pos = Point2f[]
    vec = Vec2f[]
    NX_local, NY_local = size(F)
    for i in 1:QUIVER_STEP:NX_local, j in 1:QUIVER_STEP:NY_local
        u = Ux[i, j]; v = Uy[i, j]; m = F[i, j]
        if isfinite(u) && isfinite(v) && isfinite(m)
            push!(pos, Point2f(Float32(LON[i]), Float32(LAT[j])))
            push!(vec, Vec2f(Float32(u), Float32(v)))
        end
    end
    
    if !isempty(vec)
        maxmag = maximum(norm, vec)
        cell_x = (maximum(LON) - minimum(LON)) / NX_local
        cell_y = (maximum(LAT) - minimum(LAT)) / NY_local
        target = 5f0 * Float32(min(cell_x, cell_y))
        scale  = maxmag == 0 ? 1f0 : (target / maxmag) * Float32(ARROW_SCALEUP)
        
        arrows!(ax1, pos, scale .* vec; color=:black, shaftwidth=1.5f0, tipwidth=8f0, tiplength=8f0)
    end
    
    Colorbar(fig1[1, 2], hm1, label = "(kW/m)")
    
    # Save PNG
    png_file = joinpath(FIGDIR, "Flux_with_quiver_SM_v2.png")
    save(png_file, fig1)
    
    display(fig1)
    println("\n✓ PNG saved: $png_file")
    
end




