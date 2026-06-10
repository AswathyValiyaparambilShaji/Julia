using DSP, MAT, Statistics, Printf, LinearAlgebra, TOML, Dates


include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base  = cfg["base_path"]
base2 = cfg["base_path_nt"]


for d in ["U_KE", "U_KE_3day", "U_KE_wkly2"]
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
ring_steps = nt_chunk
t_safe_start = ring_steps + 1              # first valid step (1801)
t_safe_end   = nt - ring_steps             # last  valid step (nt-1800)


# Safe 3-day chunks: only keep chunks that fall entirely within the safe range
safe_chunks = [c for c in 1:n_chunks
               if (c-1)*nt_chunk + 1 >= t_safe_start &&
                  c*nt_chunk          <= t_safe_end]


t_origin   = DateTime(2012, 3, 1, 0, 0, 0)
t_wk_start = DateTime(2012,  5, 4, 0, 0, 0)
t_wk_end   = DateTime(2012, 5, 18, 18, 0, 0)
wk_start   = Int(Dates.Hour(t_wk_start - t_origin).value) + 1
wk_end     = Int(Dates.Hour(t_wk_end   - t_origin).value) + 1


thk   = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF   = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)


for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix  = @sprintf("%02dx%02d_%d", xn, yn, buf)
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


        ke_t = Float64.(open(joinpath(base2, "KE", "ke_t_nt_$suffix.bin"), "r") do io
            reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
        end)


        ke_x = zeros(Float64, nx, ny, nz, nt)
        ke_y = zeros(Float64, nx, ny, nz, nt)
        dx_avg = dx[2:end-1, :] .+ dx[1:end-2, :]
        ke_x[2:end-1, :, :, :] = (ke_t[3:end, :, :, :] .- ke_t[1:end-2, :, :, :]) ./
                                   reshape(dx_avg, nx-2, ny, 1, 1)
        dy_avg = dy[:, 2:end-1] .+ dy[:, 1:end-2]
        ke_y[:, 2:end-1, :, :] = (ke_t[:, 3:end, :, :] .- ke_t[:, 1:end-2, :, :]) ./
                                   reshape(dy_avg, nx, ny-2, 1, 1)
        ke_t = nothing; GC.gc()


        U_KE = zeros(Float64, nx, ny, nt)
        for t in 1:nt
            t_avg  = min(div(t-1, ts) + 1, nt_avg)
            u_avg  = @view U[:, :, :, t_avg]
            v_avg  = @view V[:, :, :, t_avg]
            ke_x_t = @view ke_x[:, :, :, t]
            ke_y_t = @view ke_y[:, :, :, t]
            U_KE[:, :, t] = dropdims(sum((u_avg .* ke_x_t .+ v_avg .* ke_y_t) .* DRFfull, dims=3), dims=3)
        end
        U = nothing; V = nothing; ke_x = nothing; ke_y = nothing; GC.gc()


        open(joinpath(base2, "U_KE", "u_ke_nt_$suffix.bin"), "w") do io
            write(io, Float32.(dropdims(mean(U_KE[:, :, t_safe_start:t_safe_end], dims=3), dims=3)))
        end


        U_KE_3day = zeros(Float32, nx, ny, nt3-2)
        for (i, c) in enumerate(safe_chunks)
            t1 = (c-1)*nt_chunk + 1
            t2 = c*nt_chunk
            U_KE_3day[:, :, i] = Float32.(dropdims(mean(U_KE[:, :, t1:t2], dims=3), dims=3))
        end
        open(joinpath(base2, "U_KE_3day", "u_ke_3day_nt_$suffix.bin"), "w") do io
            write(io, U_KE_3day)
        end
        U_KE_3day = nothing; GC.gc()


        open(joinpath(base2, "U_KE_wkly2", "u_ke_wkly_nt_$suffix.bin"), "w") do io
            write(io, Float32.(dropdims(mean(U_KE[:, :, wk_start:wk_end], dims=3), dims=3)))
        end


        U_KE = nothing; GC.gc()
        println("Completed tile: $suffix")
    end
end








