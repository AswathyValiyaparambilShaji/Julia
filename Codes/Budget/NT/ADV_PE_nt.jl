using DSP, MAT, Statistics, Printf, LinearAlgebra, TOML, Dates


include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base  = cfg["base_path"]
base2 = cfg["base_path_nt"]


for d in ["U_PE", "U_PE_3day", "U_PE_wkly"]
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
N2_threshold = 1.0e-8


for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
        println("Starting tile: $suffix")


        hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
        dx    = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
        dy    = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))


        DRFfull = hFacC .* DRF3d
        DRFfull[hFacC .== 0] .= 0.0


        U = Float64.(open(joinpath(base, "3day_mean", "U", "ucc_3day_$suffix.bin"), "r") do io
            reshape(reinterpret(Float32, read(io, nx*ny*nz*nt_avg*sizeof(Float32))), nx, ny, nz, nt_avg)
        end)
        V = Float64.(open(joinpath(base, "3day_mean", "V", "vcc_3day_$suffix.bin"), "r") do io
            reshape(reinterpret(Float32, read(io, nx*ny*nz*nt_avg*sizeof(Float32))), nx, ny, nz, nt_avg)
        end)


        pe = Float64.(open(joinpath(base2, "pe", "pe_t_nt_$suffix.bin"), "r") do io
            reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
        end)


        N2 = Float64.(open(joinpath(base, "3day_mean", "N2", "N2_3day_$suffix.bin"), "r") do io
            reshape(reinterpret(Float32, read(io, nx*ny*nz*nt_avg*sizeof(Float32))), nx, ny, nz, nt_avg)
        end)


        N2_adjusted = zeros(Float64, nx, ny, nz+1, nt_avg)
        N2_adjusted[:, :, 1, :]    = N2[:, :, 1, :]
        N2_adjusted[:, :, 2:nz, :] = N2[:, :, 1:nz-1, :]
        N2_adjusted[:, :, nz+1, :] = N2[:, :, nz, :]
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
                N2_adjusted[i, j, kf+1, :] .= N2[i, j, kf-1, :]
            end
        end
        N2_center = zeros(Float64, nx, ny, nz, nt_avg)
        for k in 1:nz
            N2_center[:, :, k, :] .= 0.5 .* (N2_adjusted[:, :, k, :] .+ N2_adjusted[:, :, k+1, :])
        end
        N2_adjusted = nothing; N2 = nothing; GC.gc()
        N2_center[N2_center .< N2_threshold] .= N2_threshold


        pe_x = zeros(Float64, nx, ny, nz, nt)
        pe_y = zeros(Float64, nx, ny, nz, nt)
        dx_avg = dx[2:end-1, :] .+ dx[1:end-2, :]
        pe_x[2:end-1, :, :, :] = (pe[3:end, :, :, :] .- pe[1:end-2, :, :, :]) ./
                                   reshape(dx_avg, nx-2, ny, 1, 1)
        dy_avg = dy[:, 2:end-1] .+ dy[:, 1:end-2]
        pe_y[:, 2:end-1, :, :] = (pe[:, 3:end, :, :] .- pe[:, 1:end-2, :, :]) ./
                                   reshape(dy_avg, nx, ny-2, 1, 1)
        pe = nothing; GC.gc()


        U_PE = zeros(Float64, nx, ny, nt)
        for t in 1:nt
            t_avg  = min(div(t-1, ts) + 1, nt_avg)
            u_avg  = @view U[:, :, :, t_avg]
            v_avg  = @view V[:, :, :, t_avg]
            n2_avg = @view N2_center[:, :, :, t_avg]
            pe_x_t = @view pe_x[:, :, :, t]
            pe_y_t = @view pe_y[:, :, :, t]
            temp1  = u_avg .* pe_x_t ./ n2_avg
            temp2  = v_avg .* pe_y_t ./ n2_avg
            temp1[isnan.(temp1)] .= 0.0
            temp2[isnan.(temp2)] .= 0.0
            U_PE[:, :, t] = rho0 .* dropdims(sum((temp1 .+ temp2) .* DRFfull, dims=3), dims=3)
        end
        U = nothing; V = nothing; N2_center = nothing; pe_x = nothing; pe_y = nothing; GC.gc()


        open(joinpath(base2, "U_PE", "u_pe_nt_$suffix.bin"), "w") do io
            write(io, Float32.(dropdims(mean(U_PE, dims=3), dims=3)))
        end


        U_PE_3day = zeros(Float32, nx, ny, n_chunks)
        for c in 1:n_chunks
            t1 = (c-1)*nt_chunk + 1
            t2 = c*nt_chunk
            U_PE_3day[:, :, c] = Float32.(dropdims(mean(U_PE[:, :, t1:t2], dims=3), dims=3))
        end
        open(joinpath(base2, "U_PE_3day", "u_pe_3day_nt_$suffix.bin"), "w") do io
            write(io, U_PE_3day)
        end
        U_PE_3day = nothing; GC.gc()


        open(joinpath(base2, "U_PE_wkly", "u_pe_wkly_nt_$suffix.bin"), "w") do io
            write(io, Float32.(dropdims(mean(U_PE[:, :, wk_start:wk_end], dims=3), dims=3)))
        end


        U_PE = nothing; GC.gc()
        println("Completed tile: $suffix")
    end
end




