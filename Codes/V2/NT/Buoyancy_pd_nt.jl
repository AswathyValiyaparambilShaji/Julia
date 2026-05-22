using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML
#using CairoMakie, SparseArrays


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base = cfg["base_path_V2"]
base2 = (joinpath(base, "NT"))    


for d in ["BP", "BP_3day"]
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
ts = 72
nt_avg = div(nt, ts)
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
rho0  = 1027.5
g     = 9.8
N2_threshold = 1.0e-8


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


        rho = Float64.(open(joinpath(base, "Density", "rho_in_$suffix.bin"), "r") do io
            reshape(reinterpret(Float64, read(io, nx*ny*nz*nt*sizeof(Float64))), nx, ny, nz, nt)
        end)
        for t in 1:nt, k in 1:nz
            rho[hFacC[:,:,k] .== 0, k, t] .= NaN
        end


        N2_phase = Float64.(open(joinpath(base, "3day_mean", "N2", "N2_3day_$suffix.bin"), "r") do io
            reshape(reinterpret(Float32, read(io, nx*ny*nz*nt_avg*sizeof(Float32))), nx, ny, nz, nt_avg)
        end)


        b = Float64.(open(joinpath(base2, "b", "b_t_nt_$suffix.bin"), "r") do io
            reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
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


        N2_adjusted = zeros(Float64, nx, ny, nz+1, nt_avg)
        N2_adjusted[:, :, 1,    :] = N2_phase[:, :, 1,      :]
        N2_adjusted[:, :, 2:nz, :] = N2_phase[:, :, 1:nz-1, :]
        N2_adjusted[:, :, nz+1, :] = N2_phase[:, :, nz-1,   :]
        k_last_full = zeros(Int, nx, ny)
        for j in 1:ny, i in 1:nx
            for k in nz:-1:1
                if hFacC[i, j, k] >= 1.0
                    k_last_full[i, j] = k
                    break
                end
            end
        end
        for j in 1:ny, i in 1:nx
            kf = k_last_full[i, j]
            if kf > 0
                N2_adjusted[i, j, kf+1, :] .= N2_phase[i, j, kf-1, :]
            end
        end
        N2_center = zeros(Float64, nx, ny, nz, nt_avg)
        for k in 1:nz
            N2_center[:, :, k, :] .= 0.5 .* (N2_adjusted[:, :, k, :] .+ N2_adjusted[:, :, k+1, :])
        end
        N2_adjusted = nothing; N2_phase = nothing; GC.gc()
        N2_center[N2_center .< N2_threshold] .= N2_threshold


        B = zeros(Float64, nx, ny, nz, nt_avg)
        for i in 1:nt_avg
            t1 = (i-1)*ts + 1
            t2 = min(i*ts, nt)
            for z in 1:nz, y in 1:ny, x in 1:nx
                rho_slice = rho[x, y, z, t1:t2]
                valid_rho = rho_slice[isfinite.(rho_slice)]
                B[x, y, z, i] = length(valid_rho) > 0 ? -g * (mean(valid_rho) - rho0) / rho0 : NaN
            end
        end
        rho = nothing; GC.gc()


        B_x = fill(NaN, nx, ny, nz, nt_avg)
        B_y = fill(NaN, nx, ny, nz, nt_avg)
        for t in 1:nt_avg
            for k in 1:nz
                B_x[2:end-1, :, k, t] .= (B[3:end, :, k, t] .- B[1:end-2, :, k, t]) ./
                                          (dx[2:end-1, :] .+ dx[1:end-2, :])
                B_y[:, 2:end-1, k, t] .= (B[:, 3:end, k, t] .- B[:, 1:end-2, k, t]) ./
                                          (dy[:, 2:end-1] .+ dy[:, 1:end-2])
            end
        end
        for t in 1:nt_avg, k in 1:nz, j in 2:ny-1, i in 2:nx-1
            if hFacC[i-1,j,k] != 1 || hFacC[i,j,k] != 1 || hFacC[i+1,j,k] != 1
                B_x[i, j, k, t] = NaN
            end
            if hFacC[i,j-1,k] != 1 || hFacC[i,j,k] != 1 || hFacC[i,j+1,k] != 1
                B_y[i, j, k, t] = NaN
            end
        end
        B = nothing; GC.gc()


        bp = zeros(Float64, nx, ny, nt)
        for t in 1:nt
            t_avg  = min(div(t-1, ts) + 1, nt_avg)
            n2_val = @view N2_center[:, :, :, t_avg]
            B_x_t  = @view B_x[:, :, :, t_avg]
            B_y_t  = @view B_y[:, :, :, t_avg]
            b_t    = @view b[:, :, :, t]
            ut     = @view up_3d[:, :, :, t]
            vt     = @view vp_3d[:, :, :, t]
            temp1  = (b_t ./ n2_val) .* ut .* B_x_t .* DRFfull
            temp2  = (b_t ./ n2_val) .* vt .* B_y_t .* DRFfull
            temp1[isnan.(temp1)] .= 0.0
            temp2[isnan.(temp2)] .= 0.0
            bp[:, :, t] = -rho0 .* dropdims(sum(temp1 .+ temp2, dims=3), dims=3)
        end
        b = nothing; up_3d = nothing; vp_3d = nothing
        N2_center = nothing; B_x = nothing; B_y = nothing; GC.gc()


        open(joinpath(base2, "BP", "bp_nt_$suffix.bin"), "w") do io
            write(io, Float32.(dropdims(mean(bp, dims=3), dims=3)))
        end


        BP_3day = zeros(Float32, nx, ny, n_chunks)
        for c in 1:n_chunks
            t1 = (c-1)*nt_chunk + 1
            t2 = c*nt_chunk
            BP_3day[:, :, c] = Float32.(dropdims(mean(bp[:, :, t1:t2], dims=3), dims=3))
        end
        open(joinpath(base2, "BP_3day", "bp_3day_nt_$suffix.bin"), "w") do io
            write(io, BP_3day)
        end
        BP_3day = nothing; GC.gc()


        #=open(joinpath(base2, "BP_wkly", "bp_wkly_nt_$suffix.bin"), "w") do io
            write(io, Float32.(dropdims(mean(bp[:, :, wk_start:wk_end], dims=3), dims=3)))
        end=#


        bp = nothing; GC.gc()
        println("Completed tile: $suffix")
    end
end




