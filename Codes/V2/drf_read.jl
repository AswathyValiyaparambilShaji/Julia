using MAT, Printf, TOML, CairoMakie


include(joinpath(@__DIR__,  "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin

NZ =173
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "config", "run_debug.toml"))
cfg     = TOML.parsefile(config_file)
#base    = cfg["base_path"]    # V1 base (for hFacC/bathymetry)
base = cfg["base_path_V2"] # V2 tile output root
# ── Grid ───────────────────────────────────────────────────────────────────────
NX, NY = 288, 468
minlat, maxlat = 24.0, 31.91
minlon, maxlon = 193.0, 199.0
lat = range(minlat, maxlat, length=NY)
lon = range(minlon, maxlon, length=NX)


# ── Tiling parameters ──────────────────────────────────────────────────────────
buf    = 3
tx, ty = 47, 66
nx = tx + 2*buf   # 53
ny = ty + 2*buf   # 72
nz  = 168
nt     = 558


thk =(open(joinpath(base, "hFacC",  "delR.bin"), "r") do io
                raw = read(io,  NZ * sizeof(Float32))
                ntoh.(reshape(reinterpret(Float32, raw), NZ))
            end)

DRF  = thk[1:nz]
sum(thk)
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)

FH = fill(NaN, NX, NY)

for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]


        suffix  = @sprintf("%02dx%02d_%d", xn, yn, buf)
        hFacC =(open(joinpath(base, "hFacC",  "hFacC_v2_$suffix.bin"), "r") do io
                raw = read(io,  nx*ny*nz * sizeof(Float32))
                (reshape(reinterpret(Float32, raw), nx,ny,nz))
                end)  

        DRFfull = hFacC .* DRF3d
        DRFfull[hFacC .== 0] .= 0.0
        H = dropdims(sum(DRFfull, dims=3), dims=3) 
        FH[xs+2:xe-2,     ys+2:ye-2] .= H[buf:nx-buf+1, buf:ny-buf+1]

    end
end
println(FH[13,10])