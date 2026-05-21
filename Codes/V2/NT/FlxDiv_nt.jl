using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML
#using CairoMakie, SparseArrays


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base = cfg["base_path_V2"]
base2 = (joinpath(base, "NT"))       

#=for d in ["FDiv", "FDiv_3day", "FDiv_wkly"]
    mkpath(joinpath(base2, d))
end=#

for d in ["FDiv", "FDiv_3day"]
    mkpath(joinpath(base2, d))
end

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
nt_chunk = 72
n_chunks = div(nt,nt_chunk)
# --- Thickness & constants ---
thk =(open(joinpath(base, "hFacC",  "delR.bin"), "r") do io
                raw = read(io,  NZ * sizeof(Float32))
                ntoh.(reshape(reinterpret(Float32, raw), NZ))
            end)

DRF  = thk[1:nz]
sum(thk)
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)
g = 9.81

for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix  = @sprintf("%02dx%02d_%d", xn, yn, buf)
        suffix2 = @sprintf("%02dx%02d_%d", xn, yn, buf-2)
        println("Starting tile: $suffix")


        hFacC = read_bin(joinpath(base, "hFacC/hFacC_v2_$suffix.bin"), (nx, ny, nz))
        dx    = read_bin(joinpath(base, "DXC/DXC_v2_$suffix.bin"), (nx, ny))
        dy    = read_bin(joinpath(base, "DYC/DYC_v2_$suffix.bin"), (nx, ny))


        DRFfull = hFacC .* DRF3d
        DRFfull[hFacC .== 0] .= 0.0


        fx = Float64.(open(joinpath(base2, "xflux", "xflx_$suffix.bin"), "r") do io
            reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
        end)
        fy = Float64.(open(joinpath(base2, "yflux", "yflx_$suffix.bin"), "r") do io
            reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
        end)


        fxX = dropdims(sum(fx .* DRFfull, dims=3), dims=3)   # (nx, ny, nt)
        fyY = dropdims(sum(fy .* DRFfull, dims=3), dims=3)
        fx = nothing; fy = nothing; GC.gc()


        flxD = zeros(nx-2, ny-2, nt)
        for t in 1:nt
            for i in 2:(nx-2)
                for j in 2:(ny-2)
                    flxD[i, j, t] = (fxX[i+1, j, t] - fxX[i-1, j, t]) / (dx[i, j] + dx[i-1, j]) +
                                    (fyY[i, j+1, t] - fyY[i, j-1, t]) / (dy[i, j] + dy[i, j-1])
                end
            end
        end
        fxX = nothing; fyY = nothing; GC.gc()


        open(joinpath(base2, "FDiv", "FDiv_nt_$suffix2.bin"), "w") do io
            write(io, Float32.(mean(flxD, dims=3)))
        end


        FDiv_3day = zeros(Float32, nx-2, ny-2, n_chunks)
        for c in 1:n_chunks
            t1 = (c-1)*nt_chunk + 1
            t2 = c*nt_chunk
            FDiv_3day[:, :, c] = Float32.(mean(flxD[:, :, t1:t2], dims=3)[:, :, 1])
        end
        open(joinpath(base2, "FDiv_3day", "FDiv_3day_nt_$suffix2.bin"), "w") do io
            write(io, FDiv_3day)
        end
        FDiv_3day = nothing


        #=open(joinpath(base2, "FDiv_wkly", "FDiv_wkly_nt_$suffix2.bin"), "w") do io
            write(io, Float32.(mean(flxD[:, :, wk_start:wk_end], dims=3)))
        end=#


        flxD = nothing; GC.gc()
        println("Completed tile: $suffix")
    end
end


