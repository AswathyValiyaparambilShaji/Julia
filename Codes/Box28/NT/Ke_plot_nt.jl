using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML
using CairoMakie, SparseArrays


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base = cfg["bp_box28"]
base2 = (joinpath(base, "NT"))       


# --- Domain & grid of 27b ---
NX, NY = 384, 336
minlat, maxlat = -24.5, -18.5
minlon, maxlon = 337.5, 345.4791122715405
lat = range(minlat, maxlat, length=NY)
lon = range(minlon, maxlon, length=NX)
NZ = 173

# --- Tile & time ---
buf = 3
tx, ty = 54, 66
nx = tx + 2*buf
ny = ty + 2*buf
nz = 168
kz = 1
nt = 558

# --- Thickness & constants ---
thk =(open(joinpath(base, "hFacC",  "delR.bin"), "r") do io
                raw = read(io,  NZ * sizeof(Float32))
                ntoh.(reshape(reinterpret(Float32, raw), NZ))
            end)

DRF  = thk[1:nz]
sum(thk)
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)
g = 9.81

# Now parallelize over ALL 42 tiles
KE = zeros(NX,NY)

Threads.@threads for xn in cfg["xn_start"]:cfg["xn_e28"]
    for yn in cfg["yn_start"]:cfg["yn_e28"]

    suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)

    
    # --- Read fields ---

      hFacC = read_bin(joinpath(base, "hFacC/hFacC_v2_$suffix.bin"),
                       (nx, ny, nz))
   
    DRFfull = hFacC .* DRF3d
    z = cumsum(DRFfull, dims=3)
    depth = sum(DRFfull, dims=3)
    DRFfull[hFacC .== 0] .= 0.0

 # ---- Read KE ----
        println("  Readig KE...")
        ke_raw = Float64.(open(joinpath(base2, "KE", "ke_t_nt_$suffix.bin"), "r") do io
            nbytes = nx * ny * nz * nt * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nz, nt)
        end)


        
        # ---- Depth-integrate KE (weighted by DRFfull) ----
        DRFfull4 = reshape(DRFfull, nx, ny, nz, 1)
        ke_di    = dropdims(sum(ke_raw .* DRFfull4, dims=3), dims=3)   # nx x ny x nt

        ke = mean(ke_di, dims =3)
        # ---- Tile position in global grid ----
        xs = (xn - 1) * tx + 1
        xe = xs + tx + (2 * buf) - 1
        ys = (yn - 1) * ty + 1
        ye = ys + ty + (2 * buf) - 1


        #---- Place into global arrays (interior only) ----
        KE[xs+2:xe-2,   ys+2:ye-2] .= ke[buf:nx-buf+1,   buf:ny-buf+1]
        


    end
end
fig = Figure(resolution=(600, 700))
# --- Subplot 1: MITgcm Flux Heatmap + Quiver ---
ax1 = Axis(fig[1, 1], title= rich("KE (KJ/m²"), xlabel="Longitude[°]", ylabel="Latitude[°]")
ax1.limits[] = ((minimum(lon), maximum(lon)), 
                (minimum(lat), maximum(lat)))
hm = CairoMakie.heatmap!(ax1, lon, lat, KE./1000;
                        interpolate=false,
                        colormap=:jet,
                        colorrange=(0, 15))



Colorbar(fig[1, 2], hm, label = " (kJ/m²)")
display(fig)
FIGDIR        = cfg["fig_base_28"]
fgname = "KE_NS_nt_v1.png"
save(joinpath(FIGDIR , fgname),fig)

