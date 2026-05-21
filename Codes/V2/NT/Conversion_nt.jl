using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML
#using CairoMakie, SparseArrays


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base = cfg["base_path_V2"]
base2 = (joinpath(base, "NT"))    

#=for d in ["Conv", "Conv_3day", "Conv_wkly"]
    mkpath(joinpath(base2, d))
end=#
for d in ["Conv", "Conv_3day"]
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

T1, T2, delt, N = 10.2, 32.2, 1.0, 4


for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix  = @sprintf("%02dx%02d_%d", xn, yn, buf)
        suffix2 = @sprintf("%02dx%02d_%d", xn, yn, buf-2)
        println("Starting tile: $suffix")


        rho = Float64.(open(joinpath(base, "Density", "rho_in_$suffix.bin"), "r") do io
            reshape(reinterpret(Float64, read(io, nx*ny*nz*nt*sizeof(Float64))), nx, ny, nz, nt)
        end)


        hFacC = read_bin(joinpath(base, "hFacC/hFacC_v2_$suffix.bin"), (nx, ny, nz))
        DRFfull = hFacC .* DRF3d
        depth   = sum(DRFfull, dims=3)
        DRFfull[hFacC .== 0] .= 0.0


        fu = Float64.(open(joinpath(base2, "UVW_NT", "fu_nt_$suffix.bin"), "r") do io
            reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
        end)
        fv = Float64.(open(joinpath(base2, "UVW_NT", "fv_nt_$suffix.bin"), "r") do io
            reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
        end)


        fr = bandpassfilter(rho, T1, T2, delt, N, nt)
        rho = nothing; GC.gc()


        UDA = dropdims(sum(fu .* DRFfull, dims=3) ./ depth; dims=3)
        VDA = dropdims(sum(fv .* DRFfull, dims=3) ./ depth; dims=3)
        fu = nothing; fv = nothing; GC.gc()


        pres  = g .* cumsum(fr .* DRFfull, dims=3)
        fr    = nothing; GC.gc()
        pfz   = cat(zeros(nx, ny, 1, nt), pres; dims=3)
        pres  = nothing; GC.gc()
        pc_3d = 0.5 .* (pfz[:, :, 1:end-1, :] .+ pfz[:, :, 2:end, :])
        pfz   = nothing; GC.gc()
        pa    = sum(pc_3d .* DRFfull, dims=3) ./ depth
        pp_3d = pc_3d .- pa
        pc_3d = nothing; pa = nothing; GC.gc()


        dx = read_bin(joinpath(base, "DXC/DXC_v2_$suffix.bin"), (nx, ny))
        dy = read_bin(joinpath(base, "DYC/DYC_v2_$suffix.bin"), (nx, ny))


        H    = depth
        pb   = pp_3d[:, :, end, :]
        pp_3d = nothing; GC.gc()


        dHdx = (H[3:nx,   :] .- H[1:nx-2, :]) ./ (dx[2:nx-1, :] .+ dx[1:nx-2, :])
        dHdy = (H[:, 3:ny] .- H[:, 1:ny-2]) ./ (dy[:, 1:ny-2] .+ dy[:, 2:ny-1])


        W1 = .-(UDA[2:end-1, :,      :] .* dHdx)
        W2 = .-(VDA[:,       2:end-1, :] .* dHdy)
        UDA = nothing; VDA = nothing; dHdx = nothing; dHdy = nothing; GC.gc()


        w = W1[:, 2:end-1, :] .+ W2[2:end-1, :, :]
        W1 = nothing; W2 = nothing; GC.gc()


        c = pb[2:end-1, 2:end-1, :] .* w
        pb = nothing; w = nothing; GC.gc()


        open(joinpath(base2, "Conv", "Conv_nt_$suffix2.bin"), "w") do io
            write(io, Float32.(dropdims(mean(c, dims=3), dims=3)))
        end


        Conv_3day = zeros(Float32, nx-2, ny-2, n_chunks)
        for ch in 1:n_chunks
            t1 = (ch-1)*nt_chunk + 1
            t2 = ch*nt_chunk
            Conv_3day[:, :, ch] = Float32.(dropdims(mean(c[:, :, t1:t2], dims=3), dims=3))
        end
        open(joinpath(base2, "Conv_3day", "Conv_3day_nt_$suffix2.bin"), "w") do io
            write(io, Conv_3day)
        end
        Conv_3day = nothing; GC.gc()


        #=open(joinpath(base2, "Conv_wkly", "Conv_wkly_nt_$suffix2.bin"), "w") do io
            write(io, Float32.(dropdims(mean(c[:, :, wk_start:wk_end], dims=3), dims=3)))
        end=#


        c = nothing; GC.gc()
        println("Completed tile: $suffix")
    end
end




