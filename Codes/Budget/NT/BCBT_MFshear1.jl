using DSP, MAT, Statistics, Printf, LinearAlgebra, TOML, Dates


include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base  = cfg["base_path"]
base2 = cfg["base_path_nt"]


for d in ["SP1", "SP1_3day", "SP1_wkly2"]
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
t_wk_start = DateTime(2012,  5, 4, 0, 0, 0)
t_wk_end   = DateTime(2012, 5, 18, 18, 0, 0)
wk_start  = Int(Dates.Hour(t_wk_start - t_origin).value) + 1
wk_end    = Int(Dates.Hour(t_wk_end   - t_origin).value) + 1

thk   = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF   = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)
rho0  = 1027.5


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


        UF = Float64.(open(joinpath(base, "3day_mean", "U", "ucc_3day_$suffix.bin"), "r") do io
            reshape(reinterpret(Float32, read(io, nx*ny*nz*nt_avg*sizeof(Float32))), nx, ny, nz, nt_avg)
        end)
        VF = Float64.(open(joinpath(base, "3day_mean", "V", "vcc_3day_$suffix.bin"), "r") do io
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
         DRFfull = hFacC .* DRF3d
        depth   = sum(DRFfull, dims=3)
        DRFfull[hFacC .== 0] .= 0.0
        mask3D  = hFacC .== 0
        UcA    = sum(UF .* DRFfull, dims=3) ./ depth
        Up_3d  = UF .- UcA
        Up_3d[repeat(mask3D, 1, 1, 1, nt_avg)] .= 0.0
        UF = UcA = nothing; GC.gc()
        VcA    = sum(VF .* DRFfull, dims=3) ./ depth
        Vp_3d  = VF .- VcA
        Vp_3d[repeat(mask3D, 1, 1, 1, nt_avg)] .= 0.0
        VF = VcA = nothing; GC.gc()
        U_x = zeros(Float64, nx, ny, nz, nt_avg)
        U_y = zeros(Float64, nx, ny, nz, nt_avg)
        V_x = zeros(Float64, nx, ny, nz, nt_avg)
        V_y = zeros(Float64, nx, ny, nz, nt_avg)
        dx_avg = dx[2:end-1, :] .+ dx[1:end-2, :]
        U_x[2:end-1, :, :, :] = (Up_3d[3:end, :, :, :] .- Up_3d[1:end-2, :, :, :]) ./
                                  reshape(dx_avg, nx-2, ny, 1, 1)
        V_x[2:end-1, :, :, :] = (Vp_3d[3:end, :, :, :] .- Vp_3d[1:end-2, :, :, :]) ./
                                  reshape(dx_avg, nx-2, ny, 1, 1)
        dy_avg = dy[:, 2:end-1] .+ dy[:, 1:end-2]
        U_y[:, 2:end-1, :, :] = (Up_3d[:, 3:end, :, :] .- Up_3d[:, 1:end-2, :, :]) ./
                                  reshape(dy_avg, nx, ny-2, 1, 1)
        V_y[:, 2:end-1, :, :] = (Vp_3d[:, 3:end, :, :] .- Vp_3d[:, 1:end-2, :, :]) ./
                                  reshape(dy_avg, nx, ny-2, 1, 1)
        U = nothing; V = nothing; GC.gc()


        sp_h = zeros(Float64, nx, ny, nt)
        for t in 1:nt
            t_avg = min(div(t-1, ts) + 1, nt_avg)
            U_x_t = @view U_x[:, :, :, t_avg]
            U_y_t = @view U_y[:, :, :, t_avg]
            V_x_t = @view V_x[:, :, :, t_avg]
            V_y_t = @view V_y[:, :, :, t_avg]
            ubc    = @view up_3d[:, :, :, t]
            vbc    = @view vp_3d[:, :, :, t]
            ubt    = @view UcA[:, :, 1, t]
            vbt    = @view VcA[:, :, 1, t]
            temp1 = ubc .* ubt .* U_x_t .* DRFfull
            temp2 = ubc .* vbt .* U_y_t .* DRFfull
            temp3 = ubc .* vbt .* V_x_t .* DRFfull
            temp4 = vbc .* vbt .* V_y_t .* DRFfull
            sp_h[:, :, t] = -rho0 .* dropdims(sum(temp1 .+ temp2 .+ temp3 .+ temp4, dims=3), dims=3)
        end
        up_3d = nothing; vp_3d = nothing
        U_x = nothing; U_y = nothing; V_x = nothing; V_y = nothing; GC.gc()


        open(joinpath(base2, "SP1", "sp_h_nt_$suffix.bin"), "w") do io
            write(io, Float32.(dropdims(mean(sp_h, dims=3), dims=3)))
        end

#=
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


        open(joinpath(base2, "SP_H_wkly2", "sp_h_wkly_nt_$suffix.bin"), "w") do io
            write(io, Float32.(dropdims(mean(sp_h[:, :, wk_start:wk_end], dims=3), dims=3)))
        end

=#
        sp_h = nothing; GC.gc()
        println("Completed tile: $suffix")
    end
end




