using Printf, FilePathsBase, TOML, CairoMakie, Statistics, LinearAlgebra


include(joinpath(@__DIR__, "..", "..","functions", "FluxUtils.jl"))
using .FluxUtils: read_bin


config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..",  "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base = cfg["base_path"]
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
dt = 25
dto = 144
Tts = 366192
nt = div(Tts, dto)


println("Total time steps: $nt")


# Initialize 3D arrays
TauX_all = zeros(NX, NY, nt)
TauY_all = zeros(NX, NY, nt)


# Load and process tiles
for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
        
        println("Reading tile $suffix...")
        
        # Read entire tile file (all time steps)
        taux = Float64.(open(joinpath(base, "Windstress", "taux_$suffix.bin"), "r") do io
            nbytes = nx * ny * nt * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt)
        end)
        
        tauy = Float64.(open(joinpath(base, "Windstress", "tauy_$suffix.bin"), "r") do io
            nbytes = nx * ny * nt * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt)
        end)
        
        # Center from Arakawa C-grid - ALL TIME STEPS AT ONCE
        taux_ext = zeros(nx+1, ny, nt)
        taux_ext[1:nx, :, :] .= taux
        taux_ext[end, :, :] .= taux[end, :, :]
        
        tauy_ext = zeros(nx, ny+1, nt)
        tauy_ext[:, 1:ny, :] .= tauy
        tauy_ext[:, end, :] .= tauy[:, end, :]
        
        # Average to centers
        taux_c = 0.5 .* (taux_ext[1:end-1, :, :] .+ taux_ext[2:end, :, :])
        tauy_c = 0.5 .* (tauy_ext[:, 1:end-1, :] .+ tauy_ext[:, 2:end, :])
        
        # Extract interior (remove buffer)
        taux_int = taux_c[buf+1:nx-buf, buf+1:ny-buf, :]
        tauy_int = tauy_c[buf+1:nx-buf, buf+1:ny-buf, :]
        
        # Calculate tile position in global grid
        xs = (xn - 1) * tx + 1
        xe = xs + tx - 1
        ys = (yn - 1) * ty + 1
        ye = ys + ty - 1
        
        # Assign to global arrays
        TauX_all[xs:xe, ys:ye, :] .= taux_int
        TauY_all[xs:xe, ys:ye, :] .= tauy_int
        
        println("  Completed $suffix")
    end
end

# Check global arrays
println("\n=== GLOBAL ARRAYS ===")
println("TauX_all:")
println("  Min: $(minimum(TauX_all))")
println("  Max: $(maximum(TauX_all))")
println("  Mean: $(mean(TauX_all))")
println("  Any NaN: $(any(isnan.(TauX_all)))")
println("  Any non-zero: $(any(TauX_all .!= 0))")

#println("  Any non-NaN: $(any(!isnan.(TauX_all )))")

println("\nTauY_all:")
println("  Min: $(minimum(TauY_all))")
println("  Max: $(maximum(TauY_all))")
println("  Mean: $(mean(TauY_all))")
println("  Any NaN: $(any(isnan.(TauY_all)))")
println("  Any non-zero: $(any(TauY_all .!= 0))")

# Check for NaN
println("\nData check:")
println("  TauX range: $(minimum(TauX_all)) to $(maximum(TauX_all))")
println("  TauY range: $(minimum(TauY_all)) to $(maximum(TauY_all))")
println("  Any NaN in TauX: $(any(isnan.(TauX_all)))")
println("  Any NaN in TauY: $(any(isnan.(TauY_all)))")


# Create frames
FIGDIR = cfg["fig_base"]
frames_dir = joinpath(FIGDIR, "windstress_frames")
mkpath(frames_dir)


println("\nCalculating color range...")
all_mag = sqrt.(TauX_all.^2 .+ TauY_all.^2)
CMAX = maximum(all_mag)
println("Color range: 0 to $CMAX N/m²")


QUIVER_STEP = 8
ARROW_SCALE = 5.0


println("\nCreating $nt frames...")


for t in 1:nt
    Tau_mag = all_mag[:, :, t]
    
    fig = Figure(size=(700, 600))
    ax = Axis(fig[1, 1],
        title = "Wind Stress - Step $t/$nt",
        xlabel = "Longitude [°]",
        ylabel = "Latitude [°]",
        ylabelsize = 22,
        xlabelsize = 22,
        titlesize = 26
    )
    
    heatmap!(ax, lon, lat, Tau_mag',
        colorrange = (0, CMAX),
        colormap = :thermal
    )
    
    # Arrows
    pos = Point2f[]
    vec = Vec2f[]
    for i in 1:QUIVER_STEP:NX, j in 1:QUIVER_STEP:NY
        u, v = TauX_all[i, j, t], TauY_all[i, j, t]
        if isfinite(u) && isfinite(v)
            push!(pos, Point2f(lon[i], lat[j]))
            push!(vec, Vec2f(u, v))
        end
    end
    
    if !isempty(vec)
        maxm = maximum(norm, vec)
        if maxm > 0
            arrows!(ax, pos, ARROW_SCALE .* vec ./ maxm, 
                color=:white, arrowsize=10, linewidth=1.5)
        end
    end
    
    Colorbar(fig[1, 2], label = "Wind Stress [N/m²]")
    save(joinpath(frames_dir, @sprintf("frame_%04d.png", t)), fig)
    
    t % 100 == 0 && println("  Frame $t/$nt")
end


# Create video
movie_file = joinpath(FIGDIR, "WindStress_movie.mp4")
input_pattern = joinpath(frames_dir, "frame_%04d.png")


try
    run(`ffmpeg -y -framerate 10 -i $input_pattern -c:v libx264 -pix_fmt yuv420p -crf 23 $movie_file`)
    println("\nVideo created: $movie_file")
    rm(frames_dir, recursive=true)
    println("Frames cleaned up")
catch e
    println("\nffmpeg error: $e")
    println("Frames kept in: $frames_dir")
end


println("\nDone!")




