using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML
#using CairoMakie, SparseArrays


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base = cfg["base_path_V2"]
base2 = (joinpath(base, "NT"))    


for d in ["SP_H", "SP_H_3day"]
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
g=9.81
rho0  = 1027.5


for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
        println("Starting tile: $suffix")


        hFacC = read_bin(joinpath(base, "hFacC/hFacC_v2_$suffix.bin"), (nx, ny, nz))
        dx    = read_bin(joinpath(base, "DXC/DXC_v2_$suffix.bin"), (nx, ny))
        dy    = read_bin(joinpath(base, "DYC/DYC_v2_$suffix.bin"), (nx, ny))


        DRFfull = hFacC .* DRF3d
        depth   = sum(DRFfull, dims=3)
        DRFfull[hFacC .== 0] .= 0.0
        mask3D  = hFacC .== 0


        U = Float64.(open(joinpath(base, "3day_mean", "U", "ucc_3day_$suffix.bin"), "r") do io
            reshape(reinterpret(Float32, read(io, nx*ny*nz*nt_avg*sizeof(Float32))), nx, ny, nz, nt_avg)
        end)
        V = Float64.(open(joinpath(base, "3day_mean", "V", "vcc_3day_$suffix.bin"), "r") do io
            reshape(reinterpret(Float32, read(io, nx*ny*nz*nt_avg*sizeof(Float32))), nx, ny, nz, nt_avg)
        end)


        fu = Float64.(open(joinpath(base2, "UVW_NT", "fu_nt_$suffix.bin"), "r") do io
            reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
        end)
        ucA   = sum(fu .* DRFfull, dims=3) ./ depth
        up_3d = fu .- ucA
        up_3d[repeat(mask3D, 1, 1, 1, nt)] .= 0.0
        fu = nothing; ucA = nothing; GC.gc()


        fv = Float64.(open(joinpath(base2, "UVW_NT", "fv_nt_$suffix.bin"), "r") do io
            reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
        end)
        vcA   = sum(fv .* DRFfull, dims=3) ./ depth
        vp_3d = fv .- vcA
        vp_3d[repeat(mask3D, 1, 1, 1, nt)] .= 0.0
        fv = nothing; vcA = nothing; GC.gc()


        U_x = zeros(Float64, nx, ny, nz, nt_avg)
        U_y = zeros(Float64, nx, ny, nz, nt_avg)
        V_x = zeros(Float64, nx, ny, nz, nt_avg)
        V_y = zeros(Float64, nx, ny, nz, nt_avg)
        dx_avg = dx[2:end-1, :] .+ dx[1:end-2, :]
        U_x[2:end-1, :, :, :] = (U[3:end, :, :, :] .- U[1:end-2, :, :, :]) ./
                                  reshape(dx_avg, nx-2, ny, 1, 1)
        V_x[2:end-1, :, :, :] = (V[3:end, :, :, :] .- V[1:end-2, :, :, :]) ./
                                  reshape(dx_avg, nx-2, ny, 1, 1)
        dy_avg = dy[:, 2:end-1] .+ dy[:, 1:end-2]
        U_y[:, 2:end-1, :, :] = (U[:, 3:end, :, :] .- U[:, 1:end-2, :, :]) ./
                                  reshape(dy_avg, nx, ny-2, 1, 1)
        V_y[:, 2:end-1, :, :] = (V[:, 3:end, :, :] .- V[:, 1:end-2, :, :]) ./
                                  reshape(dy_avg, nx, ny-2, 1, 1)
        U = nothing; V = nothing; GC.gc()


        sp_h = zeros(Float64, nx, ny, nt)
        for t in 1:nt
            t_avg = min(div(t-1, ts) + 1, nt_avg)
            U_x_t = @view U_x[:, :, :, t_avg]
            U_y_t = @view U_y[:, :, :, t_avg]
            V_x_t = @view V_x[:, :, :, t_avg]
            V_y_t = @view V_y[:, :, :, t_avg]
            ut    = @view up_3d[:, :, :, t]
            vt    = @view vp_3d[:, :, :, t]
            temp1 = ut .* ut .* U_x_t .* DRFfull
            temp2 = ut .* vt .* U_y_t .* DRFfull
            temp3 = ut .* vt .* V_x_t .* DRFfull
            temp4 = vt .* vt .* V_y_t .* DRFfull
            sp_h[:, :, t] = -rho0 .* dropdims(sum(temp1 .+ temp2 .+ temp3 .+ temp4, dims=3), dims=3)
        end
        up_3d = nothing; vp_3d = nothing
        U_x = nothing; U_y = nothing; V_x = nothing; V_y = nothing; GC.gc()


        open(joinpath(base2, "SP_H", "sp_h_nt_$suffix.bin"), "w") do io
            write(io, Float32.(dropdims(mean(sp_h, dims=3), dims=3)))
        end


        SP_H_3day = zeros(Float32, nx, ny, n_chunks)
        for c in 1:n_chunks
            t1 = (c-1)*nt_chunk + 1
            t2 = c*nt_chunk
            SP_H_3day[:, :, c] = Float32.(dropdims(mean(sp_h[:, :, t1:t2], dims=3), dims=3))
        end
        open(joinpath(base2, "SP_H_3day", "sp_h_3day_nt_$suffix.bin"), "w") do io
            write(io, SP_H_3day)
        end
        SP_H_3day = nothing; GC.gc()


        #=open(joinpath(base2, "SP_H_wkly", "sp_h_wkly_nt_$suffix.bin"), "w") do io
            write(io, Float32.(dropdims(mean(sp_h[:, :, wk_start:wk_end], dims=3), dims=3)))
        end=#


        sp_h = nothing; GC.gc()
        println("Completed tile: $suffix")
    end
end




