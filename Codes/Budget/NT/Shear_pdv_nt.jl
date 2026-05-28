using DSP, MAT, Statistics, Printf, LinearAlgebra, TOML, Dates


include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base  = cfg["base_path"]
base2 = cfg["base_path_nt"]


for d in ["SP_V", "SP_V_3day", "SP_V_wkly"]
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


for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
        println("Starting tile: $suffix")


        hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))


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


        fw = Float64.(open(joinpath(base2, "UVW_NT", "fw_nt_$suffix.bin"), "r") do io
            reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
        end)
        wcA   = sum(fw .* DRFfull, dims=3) ./ depth
        wp_3d = fw .- wcA
        wp_3d[repeat(mask3D, 1, 1, 1, nt)] .= 0.0
        fw = nothing; wcA = nothing; GC.gc()


        U_z = zeros(Float64, nx, ny, nz, nt_avg)
        V_z = zeros(Float64, nx, ny, nz, nt_avg)
        for t_avg in 1:nt_avg
            for k in 2:nz-1
                dz = -(DRF[k-1]/2.0 + DRF[k] + DRF[k+1]/2.0)
                U_z[:, :, k, t_avg] = (U[:, :, k+1, t_avg] .- U[:, :, k-1, t_avg]) ./ dz
                V_z[:, :, k, t_avg] = (V[:, :, k+1, t_avg] .- V[:, :, k-1, t_avg]) ./ dz
            end
        end
        U = nothing; V = nothing; GC.gc()


        sp_v = zeros(Float64, nx, ny, nt)
        for t in 1:nt
            t_avg = min(div(t-1, ts) + 1, nt_avg)
            U_z_t = @view U_z[:, :, :, t_avg]
            V_z_t = @view V_z[:, :, :, t_avg]
            ut    = @view up_3d[:, :, :, t]
            vt    = @view vp_3d[:, :, :, t]
            wt    = @view wp_3d[:, :, :, t]
            temp1 = wt .* ut .* U_z_t .* DRFfull
            temp2 = wt .* vt .* V_z_t .* DRFfull
            sp_v[:, :, t] = -rho0 .* dropdims(sum(temp1 .+ temp2, dims=3), dims=3)
        end
        up_3d = nothing; vp_3d = nothing; wp_3d = nothing
        U_z = nothing; V_z = nothing; GC.gc()

        spm = dropdims(mean(sp_v, dims=3), dims=3)
        println(spm[20,1:10])
        open(joinpath(base2, "SP_V", "sp_v_nt_$suffix.bin"), "w") do io
            write(io, Float32.(spm))
        end


        SP_V_3day = zeros(Float32, nx, ny, n_chunks)
        for c in 1:n_chunks
            t1 = (c-1)*nt_chunk + 1
            t2 = c*nt_chunk
            SP_V_3day[:, :, c] = Float32.(dropdims(mean(sp_v[:, :, t1:t2], dims=3), dims=3))
        end
        open(joinpath(base2, "SP_V_3day", "sp_v_3day_nt_$suffix.bin"), "w") do io
            write(io, SP_V_3day)
        end
        SP_V_3day = nothing; GC.gc()


        open(joinpath(base2, "SP_V_wkly", "sp_v_wkly_nt_$suffix.bin"), "w") do io
            write(io, Float32.(dropdims(mean(sp_v[:, :, wk_start:wk_end], dims=3), dims=3)))
        end


        sp_v = nothing; GC.gc()
        println("Completed tile: $suffix")
    end
end




