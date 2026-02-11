using Printf, FilePathsBase, TOML#, JSON


# Include FluxUtils.jl for any additional utility functions
include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin


# Read the configuration file for paths
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base = cfg["base_path"]
base2 = cfg["base_path2"]
mkpath(joinpath(base2, "WindStress_F"))  # Directory for storing the wind stress tiles


# --- Grid parameters ---
NX, NY = 288, 468
minlat, maxlat = 24.0, 31.91
minlon, maxlon = 193.0, 199.0
lat = LinRange(minlat, maxlat, NY)
lon = LinRange(minlon, maxlon, NX)


# Tiling parameters
buf = 3
tx, ty = 47, 66
nx = tx + 2 * buf
ny = ty + 2 * buf
dt = 25
dto = 144
Tts = 366192
nt = div(Tts, dto)


println("Processing $nt time steps...")


# --- Process time steps ---
for ts in 1:nt
    
    # Define time step suffix
    tt = (ts - 1) * dto
    suffix = @sprintf("%010d", tt + 597888)
    
    # --- Read the wind stress data for `taux` and `tauy` ---
    # Corrected file names based on Python output
    taux_file = joinpath(base, "TauX", "oceTAUX.$suffix.data")
    tauy_file = joinpath(base, "TauY", "oceTAUY.$suffix.data")
    
    # Read binary data - 2D fields (NX, NY) for each time step
    taux = read_bin(taux_file, (NX, NY))  # taux is on U-grid (face in x-direction)
    tauy = read_bin(tauy_file, (NX, NY))  # tauy is on V-grid (face in y-direction)
    
    # --- Tile data and save ---
    for xn in cfg["xn_start"]:cfg["xn_end"]
        for yn in cfg["yn_start"]:cfg["yn_end"]
            
            tile_suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
            
            # Calculate tile indices (with buffer)
            x_start = (xn - 1) * tx - buf + 1
            x_end = xn * tx + buf
            y_start = (yn - 1) * ty - buf + 1
            y_end = yn * ty + buf
            
            # Ensure indices are within bounds
            x_start = max(1, x_start)
            x_end = min(NX, x_end)
            y_start = max(1, y_start)
            y_end = min(NY, y_end)
            
            # Extract tile subdomain
            taux_tile = taux[x_start:x_end, y_start:y_end]
            tauy_tile = tauy[x_start:x_end, y_start:y_end]
            
            # Define output file paths for the tiles
            taux_tile_file = joinpath(base2, "WindStress_F", "taux_$tile_suffix.bin")
            tauy_tile_file = joinpath(base2, "WindStress_F", "tauy_$tile_suffix.bin")
            
            # Append tile data to files
            open(taux_tile_file, "a") do io
                write(io, Float32.(taux_tile))
            end
            open(tauy_tile_file, "a") do io
                write(io, Float32.(tauy_tile))
            end
        end
    end
    
    if ts % 30 == 0 || ts == 1
        println("Progress: $ts/$nt - Time step: $suffix")
    end
end


println("Wind stress tiling complete!")
println("Created $(cfg["xn_end"] - cfg["xn_start"] + 1) × $(cfg["yn_end"] - cfg["yn_start"] + 1) = $((cfg["xn_end"] - cfg["xn_start"] + 1) * (cfg["yn_end"] - cfg["yn_start"] + 1)) tiles")




