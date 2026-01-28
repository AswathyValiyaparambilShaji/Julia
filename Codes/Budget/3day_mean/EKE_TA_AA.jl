using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin


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
dt = 25
dto = 144
Tts = 366192
nt = div(Tts, dto)
timesteps_per_3days = 72
nt_avg = div(nt, timesteps_per_3days) 


# reference density
rho0 = 999.8

nt_10avg = div(nt_avg, 10)
# Initialize arrays for full domain
KE_surface = zeros(Float64, NX, NY, nt_avg)
DX_full = zeros(Float64, NX, NY)
DY_full = zeros(Float64, NX, NY)


for xn in cfg["xn_start"]:cfg["xn_end"]
   for yn in cfg["yn_start"]:cfg["yn_end"]


       suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
       println("Processing tile: $suffix")


       # --- Read grid spacing ---
       dx = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
       dy = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))


       # --- Read velocity fields (3-day averaged) ---
       U = Float64.(open(joinpath(base,"3day_mean", "U", "ucc_3day_$suffix.bin"), "r") do io
           nbytes = nx * ny * nz * nt_avg * sizeof(Float32)
           reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nz, nt_avg)
       end)
      
       V = Float64.(open(joinpath(base,"3day_mean","V", "vcc_3day_$suffix.bin"), "r") do io
           nbytes = nx * ny * nz * nt_avg * sizeof(Float32)
           reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nz, nt_avg)
       end)


       # Surface KE at k=1
       ke_surface = 0.5 .* rho0 .* (U[:, :, 1, :].^2 .+ V[:, :, 1, :].^2)


       # Calculate tile positions
       xs = (xn - 1) * tx + 1 
       xe = xs + tx + (2 * buf) - 1 
       ys = (yn - 1) * ty + 1 
       ye = ys + ty + (2 * buf) - 1 


       # Interior indices (removing buffer)
       xsf = 2
       xef = tx + (2*buf) - 1
       ysf = 2
       yef = ty + (2*buf) - 1
      
       # Assign to global arrays
       KE_surface[xs+1:xe-1, ys+1:ye-1, :] .= ke_surface[xsf:xef, ysf:yef, :]
       DX_full[xs+1:xe-1, ys+1:ye-1] .= dx[xsf:xef, ysf:yef]
       DY_full[xs+1:xe-1, ys+1:ye-1] .= dy[xsf:xef, ysf:yef]
   end
end

# --- 10-timestep averaging ---
println("\nPerforming 10-timestep averaging...")
KE_10avg = zeros(Float64, NX, NY)


    
    # Average over 10 timesteps
    KE_10avg[:, :] = mean(KE_surface[:, :, :], dims=3)
    
   
# --- Time average over all 10-timestep windows ---



#= --- Time average ---
println("\n10-timestep averaged surface KE calculated")

 KE_AA = zeros(Float64, nt_10avg)
# --- Weighted area average ---
dA = DX_full .* DY_full
for t in 1:nt_10avg
    KE_AA[t]= sum(KE_10avg[:,:,t] .* dA) / sum(dA)

end
println("\nWeighted area-averaged surface KE: $KE_AA J/m³")

println("Total area: $(sum(dA)/1e6) km²")

KE_time_avg = mean(KE_surface, dims=3)[:, :, 1]


println("\nTime-averaged surface KE calculated")
println("KE_time_avg range: $(extrema(KE_time_avg))")


# --- Weighted area average: sum(KE * dx * dy) / sum(dx * dy) ---
dA = DX_full .* DY_full  # Cell areas
KE_area_weighted = sum(KE_time_avg .* dA) / sum(dA)


println("\nWeighted area-averaged surface KE: $KE_area_weighted J/m³")
println("Total area: $(sum(dA)) m²")


# Optional: Save the time-averaged field
=#
using CairoMakie


fig = Figure(size=(700, 600))
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
save(joinpath(FIGDIR, "EKE_v3.png"), fig)

#


