using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra
using CairoMakie, SparseArrays

include("/home3/avaliyap/Documents/julia/FluxUtils.jl")
using .FluxUtils: read_bin, bandpassfilter

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
thk = matread("/nobackup/avaliyap/Box56/hFacC/thk90.mat")["thk90"]
DRF = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)
g = 9.8


base = "/nobackup/avaliyap/Box56/"
base2 = "/nobackup/avaliyap/Box56/SM/"

# Now parallelize over ALL 42 tiles
KE = zeros(NX,NY)
for xn in 1:6
   for yn in 1:7

    suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)

    
    # --- Read fields ---
    hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
   
    DRFfull = hFacC .* DRF3d
    z = cumsum(DRFfull, dims=3)
    depth = sum(DRFfull, dims=3)
    DRFfull[hFacC .== 0] .= 0.0


    ke_t = open(joinpath(base2, "KE", "ke_t_sm_$suffix.bin"), "r") do io
            # Calculate the number of bytes needed
            nbytes = nx * ny * nz * nt * sizeof(Float64)
            # Read the raw bytes
            raw_bytes = read(io, nbytes)
            # Reinterpret as Float64 array and reshape
            raw_data = reinterpret(Float64, raw_bytes)
            reshaped_data = reshape(raw_data, nx, ny, nz, nt)
        end


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

        xs = (xn - 1) * tx + 1  
        xe = xs + tx + (2 * buf) - 1  
        
        ys = (yn - 1) * ty + 1  
        ye = ys + ty + (2 * buf) - 1  
        

        xsf = 2;
        xef = tx + (2*buf) - 1
        ysf = 2;
        yef = ty + (2*buf) - 1
        
        # Assign the flux data to the correct region in the full flux arrays
        KE[xs+1:xe-1, ys+1:ye-1] .= ke[xsf:xef,ysf:yef]
        


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

fgpathj = "/nobackup/avaliyap/Figure/Box56/"
fgname = "KE_dI_v3.png"
save(joinpath(fgpathj, fgname),fig)

