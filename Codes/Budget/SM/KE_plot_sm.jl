using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML
using CairoMakie, SparseArrays


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin


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

kz = 1
dt = 25
dto = 144
Tts = 366192
nt = div(Tts, dto)

# --- Thickness & constants ---

thk = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)


g = 9.8



# Now parallelize over ALL 42 tiles
KE = zeros(NX,NY)
for xn in cfg["xn_start"]:cfg["xn_end"]
  for yn in cfg["yn_start"]:cfg["yn_end"]

    suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)

    
    # --- Read fields ---

      hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"),
                       (nx, ny, nz))
   
    DRFfull = hFacC .* DRF3d
    z = cumsum(DRFfull, dims=3)
    depth = sum(DRFfull, dims=3)
    DRFfull[hFacC .== 0] .= 0.0

 # --- Read KE ---
            ke_t = Float64.(open(joinpath(base2, "KE", "ke_t_sm_$suffix.bin"), "r") do io
                nbytes = nx * ny * nz * nt * sizeof(Float32)
                raw_bytes = read(io, nbytes)
                raw_data = reinterpret(Float32, raw_bytes)
                reshape(raw_data, nx, ny, nz, nt)
            end)


        #=fv = open(joinpath(base2, "UVW_F", "fv_$suffix.bin"), "r") do io
            # Calculate the number of bytes needed
            nbytes = nx * ny * nz * sizeof(Float64)
            # Read the raw bytes
            raw_bytes = read(io, nbytes)
            # Reinterpret as Float64 array and reshape
            raw_data = reinterpret(Float64, raw_bytes)
            reshaped_data = reshape(raw_data, nx, ny, nz)
        end
        =#

        ked = mean(ke_t, dims=4)
        ke = sum(ked .* DRFfull, dims=3)        # (nx,ny,1,1)

        # ---- Tile position in global grid ----
        xs = (xn - 1) * tx + 1
        xe = xs + tx + (2 * buf) - 1
        ys = (yn - 1) * ty + 1
        ye = ys + ty + (2 * buf) - 1


        #---- Place into global arrays (interior only) ----
        KE[xs+2:xe-2,   ys+2:ye-2] .= ke[buf:nx-buf+1,   buf:ny-buf+1]
        


    end
end
fig = Figure(resolution=(500, 400))

# --- Subplot 1: MITgcm Flux Heatmap + Quiver ---
ax1 = Axis(fig[1, 1], title="KE (kg/s²) ", xlabel="Longitude[°]", ylabel="Latitude[°]")
ax1.limits[] = ((minimum(lon), maximum(lon)), 
                (minimum(lat), maximum(lat)))
hm = CairoMakie.heatmap!(ax1, lon, lat, KE ; interpolate = false,colormap    = :jet,colorrange = (0,8000))


Colorbar(fig[1, 2], hm, label = " (kg/s²)")
display(fig)
FIGDIR        = cfg["fig_base"]
fgname = "KE_dI_v4.png"
save(joinpath(FIGDIR , fgname),fig)

