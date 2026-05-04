using DSP, MAT, Statistics, Printf, LinearAlgebra, TOML, Dates


include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base  = cfg["base_path"]
base2 = cfg["base_path_nt"]


for d in ["BP", "BP_3day", "BP_wkly"]
    mkpath(joinpath(base2, d))
end


NX, NY = 288, 468
minlat, maxlat = 24.0, 31.91
minlon, maxlon = 193.0, 199.0
buf = 3
tx, ty = 47, 66
nx = tx + 2*buf
ny = ty + 2*buf
nz = 88
dto = 144
Tts = 366192
nt = div(Tts, dto)
ts = 72
nt_avg = div(nt, ts)
nt_chunk = 72
n_chunks = div(nt, nt_chunk)


t_origin   = DateTime(2012, 3, 1, 0, 0, 0)
t_wk_start = DateTime(2012, 4, 22, 0, 0, 0)
t_wk_end   = DateTime(2012, 4, 28, 23, 0, 0)
wk_start   = Int(Dates.Hour(t_wk_start - t_origin).value) + 1
wk_end     = Int(Dates.Hour(t_wk_end   - t_origin).value) + 1


thk   = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF   = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)
rho0  = 1027.5
g     = 9.8
N2_threshold = 1.0e-8


for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
        println("Starting tile: $suffix")


        hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
        dx    = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
        dy    = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))


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


        b = Float64.(open(joinpath(base2, "b", "b_nt_$suffix.bin"), "r") do io
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


        open(joinpath(base2, "BP_wkly", "bp_wkly_nt_$suffix.bin"), "w") do io
            write(io, Float32.(dropdims(mean(bp[:, :, wk_start:wk_end], dims=3), dims=3)))
        end


        bp = nothing; GC.gc()
        println("Completed tile: $suffix")
    end
end




