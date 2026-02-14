using Printf, FilePathsBase, TOML


# Include FluxUtils.jl
include(joinpath(@__DIR__, "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin


# Read the configuration file for paths
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base = cfg["base_path"]


# Create output directory
output_dir = joinpath(base, "Windstress")
mkpath(output_dir)


# --- Grid parameters ---
NX, NY = 288, 468


# Tiling parameters
buf = 3
tx, ty = 47, 66
nx = tx + 2 * buf
ny = ty + 2 * buf
dto = 144
Tts = 366192
nt = div(Tts, dto)


println("Processing $nt time steps...")
println("Tile core: $tx × $ty, with buffer: $nx × $ny")
println("Output directory: $output_dir")


# --- Process time steps ---
for ts in 1:nt
    
    # Define time step suffix
    tt = (ts - 1) * dto
    suffix = @sprintf("%010d", tt + 597888)
    
    # --- Read the wind stress data ---
    taux_file = joinpath(base, "MIT_WS", "oceTAUX.$suffix.data")
    tauy_file = joinpath(base, "MIT_WS", "oceTAUY.$suffix.data")
    
    # Check if files exist
    if !isfile(taux_file)
        println("ERROR: Missing file at time step $ts: $taux_file")
        continue
    end
    if !isfile(tauy_file)
        println("ERROR: Missing file at time step $ts: $tauy_file")
        continue
    end
    
    # Read binary data
    taux = read_bin(taux_file, (NX, NY))
    tauy = read_bin(tauy_file, (NX, NY))
    
    # --- Tile data (following MATLAB logic) ---
    xn = 1
    for xs in (buf + 1):tx:(NX - buf)
        xe = xs + tx - 1
        xsb = xs - buf
        xeb = xe + buf
        
        yn = 1
        for ys in (buf + 1):ty:(NY - buf)
            ye = ys + ty - 1
            ysb = ys - buf
            yeb = ye + buf
            
            # Extract tile with buffer
            taux_blk = taux[xsb:xeb, ysb:yeb]
            tauy_blk = tauy[xsb:xeb, ysb:yeb]
            
            # Define output files
            tile_suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
            taux_tile_file = joinpath(output_dir, "taux_$tile_suffix.bin")
            tauy_tile_file = joinpath(output_dir, "tauy_$tile_suffix.bin")
            
            #=Append to files
            open(taux_tile_file, "a") do fid
                write(fid, Float32.(taux_blk))
            end
            open(tauy_tile_file, "a") do fid
                write(fid, Float32.(tauy_blk))
            end
            =#
            
            yn = yn + 1
        end
        xn = xn + 1
    end
    
    # Print progress
    if ts % 50 == 0 || ts == 1 || ts == nt
        println(taux_blk)
    end
end


println("\nWind stress tiling complete!")
println("Output location: $output_dir")




