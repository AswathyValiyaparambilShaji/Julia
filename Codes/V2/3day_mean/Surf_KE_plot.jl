using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML
#using CairoMakie, SparseArrays


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base = cfg["base_path_V2"]


# --- Domain & grid ---
NX, NY = 288, 468
minlat, maxlat = 24.0, 31.91
minlon, maxlon = 193.0, 199.0
lat = range(minlat, maxlat, length=NY)
lon = range(minlon, maxlon, length=NX)
NZ = 173

# --- Tile & time ---
buf = 3
tx, ty = 47, 66
nx = tx + 2*buf
ny = ty + 2*buf
nz = 168
kz = 1
nt = 558
timesteps_per_3days = 72  # 3 timesteps = 72 hours
nt_avg = div(nt, timesteps_per_3days)  

# reference density
rho0 = 1027.5

# Initialize arrays for full domain
KE = zeros(Float64, NX, NY, nt_avg  )  

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
         ke =  0.5 .* rho0 .* (U[:, :, 1, :].^2 .+ V[:, :, 1, :].^2)
        xs = (xn - 1) * tx + 1  
        xe = xs + tx + (2 * buf) - 1  
        
        ys = (yn - 1) * ty + 1  
        ye = ys + ty + (2 * buf) - 1  
        

        xsf = 2;
        xef = tx + (2*buf) - 1
        ysf = 2;
        yef = ty + (2*buf) - 1
        
        # Assign the flux data to the correct region in the full flux arrays
        KE[xs+1:xe-1, ys+1:ye-1,:] .= ke[xsf:xef,ysf:yef,:]
        
        # Save them as movie
    
    end
end

# Save them as movie --> Draw figures and make a movie

using CairoMakie, Printf

KE_10avg = zeros(Float64, NX, NY)


    
KE_10avg[:, :] = mean(KE[:, :, :], dims=3)
    
   
fig = Figure(size=(600, 800))
ax = Axis(fig[1, 1],
    xlabel="Longitude [°]",
    ylabel="Latitude [°]",
    title="EKE",
    titlesize=26,
    ylabelsize = 22,
    xlabelsize = 22,
    )


hm = CairoMakie.heatmap!(ax, lon, lat, KE_10avg,
    colormap=:jet,
    interpolate=false)


Colorbar(fig[1, 2], hm, label="[J/m³]")


display(fig)


# Save figure
FIGDIR = cfg["fig_base"]
save(joinpath(FIGDIR, "EKE_NS_v1.png"), fig)

#


