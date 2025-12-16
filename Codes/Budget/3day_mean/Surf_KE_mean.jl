using DSP, MAT, Statistics, Printf, Plots, FilePathsBase, LinearAlgebra, TOML
#using CairoMakie, SparseArrays

include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
include(joinpath(@__DIR__, "..","..","..", "functions", "densjmd95.jl"))

using .FluxUtils: read_bin, bandpassfilter
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base = cfg["base_path"]


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
timesteps_per_3days = 72  # 3 timesteps = 72 hours
nt_avg = div(nt, timesteps_per_3days)  

# reference density
rho0 = 999.8
mkpath(joinpath(base,"Figures"))

# Initialize arrays for full domain
KE = zeros(Float64, NX, NY,1 , nt_avg  )  

for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]

        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
        
        println("Processing tile: $suffix")


        # --- Read velocity fields (3-day averaged) ---
        U = Float64.(open(joinpath(base,"3day_mean", "U", "ucc_3day_$suffix.bin"), "r") do io
            nbytes = nx * ny * nz * nt_avg * sizeof(Float32)
            raw_bytes = read(io, nbytes)
            raw_data = reinterpret(Float32, raw_bytes)
            reshape(raw_data, nx, ny, nz, nt_avg)
        end)
        
        V = Float64.(open(joinpath(base,"3day_mean","V", "vcc_3day_$suffix.bin"), "r") do io
            nbytes = nx * ny * nz * nt_avg * sizeof(Float32)
            raw_bytes = read(io, nbytes)
            raw_data = reinterpret(Float32, raw_bytes)
            reshape(raw_data, nx, ny, nz, nt_avg)
        end)


        # Surface KE Calculating in each time step
         ke =  0.5 .* rho0.* (U.^2 + V.^2)

        xs = (xn - 1) * tx + 1  
        xe = xs + tx + (2 * buf) - 1  
        
        ys = (yn - 1) * ty + 1  
        ye = ys + ty + (2 * buf) - 1  
        

        xsf = 2;
        xef = tx + (2*buf) - 1
        ysf = 2;
        yef = ty + (2*buf) - 1
        
        # Assign the flux data to the correct region in the full flux arrays
        KE[xs+1:xe-1, ys+1:ye-1,1,:] .= ke[xsf:xef,ysf:yef,1,:]
        
        # Save them as movie
    
    end
end

# Save them as movie --> Draw figures and make a movie

using CairoMakie, Printf


# Create output directory for frames
plot_dir = joinpath(base, "Figures")


# Extract surface KE (at kz=1)
KE_surface = KE[:, :, 1, :]  # (NX, NY, nt_avg)


# Get color range for consistent scaling
KE_max = maximum(KE_surface[KE_surface .> 0])


println("Creating frames for movie...")
println("Total frames: $nt_avg")


# Create individual frames
for t in 1:nt_avg
    fig = Figure(size=(1000, 800))
    ax = Axis(fig[1, 1],
        xlabel="Longitude (°E)",
        ylabel="Latitude (°N)",
        title="Surface KE - Day $(3*t)",
        aspect=DataAspect())
    
    hm = CairoMakie.heatmap!(ax, lon, lat, KE_surface[:, :, t],
        colormap=:thermal,
        colorrange=(0, KE_max))
    
    Colorbar(fig[1, 2], hm, label="KE (J/m³)")
    
    save(joinpath(plot_dir, @sprintf("frame_%04d.png", t)), fig)
    
    if t % 100 == 0
        println("  Completed $t / $nt_avg")
    end
end


println("\nFrames saved to: $plot_dir")
println("\nTo create movie, run in terminal:")
println("ffmpeg -framerate 10 -i $(plot_dir)/frame_%04d.png -c:v libx264 -pix_fmt yuv420p $(joinpath(base, "KE_surface_movie.mp4"))")

