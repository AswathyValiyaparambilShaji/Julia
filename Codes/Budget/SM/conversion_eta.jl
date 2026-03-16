using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML


include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter


config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg   = TOML.parsefile(config_file)
base  = cfg["base_path"]
base2 = cfg["base_path2"]


# --- TIME MODE CONFIGURATION ---
# Options:
#   "3day"   -> conversion for each 3-day bin over full record (output has nt3 time steps)
#   "weekly" -> conversion mean over Apr 22 00:00 - Apr 28 23:00 (single output)
#   "full"   -> conversion mean over full record (single output)
time_mode = "full"   # <-- change to "3day", "weekly", or "full"


# --- Domain & grid ---
NX, NY = 288, 468
minlat, maxlat = 24.0, 31.91
minlon, maxlon = 193.0, 199.0
lat = range(minlat, maxlat, length=NY)
lon = range(minlon, maxlon, length=NX)


# --- Tile & time ---
buf = 3
tx, ty = 47, 66
nx = tx + 2*buf
ny = ty + 2*buf
nz = 88


dt  = 25
dto = 144
Tts = 366192
nt  = div(Tts, dto)
nt3 = div(nt, 3*24)


# -------------------------------------------------------------------------
# Weekly window: April 22 00:00 to April 28 23:00, 2012
# -------------------------------------------------------------------------
hour_apr22_start = 31*24 + (22-1)*24
hour_apr28_end   = 31*24 +  28   *24 - 1
idx_start        = hour_apr22_start + 1
idx_end          = hour_apr28_end   + 1
nt_week          = idx_end - idx_start + 1


@printf("Weekly window: Apr 22 00:00 - Apr 28 23:00  ->  indices %d:%d  (%d hourly snapshots)\n",
        idx_start, idx_end, nt_week)


# --- Thickness & constants ---
thk   = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF   = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)
g     = 9.8


# --- Filter parameters (9-15 day band, 1 hour sampling) ---
T1, T2, delt, N = 9.0, 15.0, 1.0, 4


# delt in seconds — used for ∂η/∂t finite differences
delt_s = delt * 3600.0


# ============================================================================
# MAIN WORKFLOW
# ============================================================================


if time_mode == "3day"
    println("Computing FULL conversion (Cb + Cs) for $nt3 3-day periods")
    mkpath(joinpath(base2, "Conv_eta_3day"))


    for xn in cfg["xn_start"]:cfg["xn_end"]
        for yn in cfg["yn_start"]:cfg["yn_end"]
            suffix  = @sprintf("%02dx%02d_%d", xn, yn, buf)
            suffix2 = @sprintf("%02dx%02d_%d", xn, yn, buf-2)


            # ---- Read fields (full time series) ----
            rho = Float64.(open(joinpath(base, "Density", "rho_in_$suffix.bin"), "r") do io
                reshape(reinterpret(Float64, read(io, nx*ny*nz*nt*sizeof(Float64))), nx, ny, nz, nt)
            end)


            hFacC   = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
            DRFfull = hFacC .* DRF3d
            depth   = sum(DRFfull, dims=3)
            DRFfull[hFacC .== 0] .= 0.0


            fu = Float64.(open(joinpath(base2, "UVW_F", "fu_$suffix.bin"), "r") do io
                reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
            end)


            fv = Float64.(open(joinpath(base2, "UVW_F", "fv_$suffix.bin"), "r") do io
                reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
            end)


            # eta is (nx, ny, nt) — bandpassfilter works on any ndims array
            # as long as time is the last dimension, which it is here
            eta_raw = Float64.(open(joinpath(base, "Eta", "Eta_$suffix.bin"), "r") do io
                reshape(reinterpret(Float32, read(io, nx*ny*nt*sizeof(Float32))), nx, ny, nt)
            end)


            dx = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
            dy = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))


            # ---- Filter rho (4D) and eta (3D) with the same bandpassfilter ----
            fr    = bandpassfilter(rho,     T1, T2, delt, N, nt)
            eta_f = bandpassfilter(eta_raw, T1, T2, delt, N, nt)


            # ---- Depth-averaged velocities ----
            UDA = dropdims(sum(fu .* DRFfull, dims=3) ./ depth; dims=3)  # (nx, ny, nt)
            VDA = dropdims(sum(fv .* DRFfull, dims=3) ./ depth; dims=3)


            # ---- Pressure perturbation p'' = p' − <p'> ----
            pres  = g .* cumsum(fr .* DRFfull, dims=3)
            pfz   = cat(zeros(nx, ny, 1, nt), pres; dims=3)
            pc_3d = 0.5 .* (pfz[:, :, 1:end-1, :] .+ pfz[:, :, 2:end, :])
            pa    = sum(pc_3d .* DRFfull, dims=3) ./ depth
            pp_3d = pc_3d .- pa


            ps = pp_3d[:, :, 1,   :]   # p'' at surface top layer  (nx, ny, nt)
            pb = pp_3d[:, :, end, :]   # p'' at bottom layer       (nx, ny, nt)


            # ---- ∂η/∂t : centred differences, one-sided at endpoints ----
            deta_dt = similar(eta_f)
            deta_dt[:, :, 2:end-1] .= (eta_f[:, :, 3:end] .- eta_f[:, :, 1:end-2]) ./ (2 * delt_s)
            deta_dt[:, :, 1]       .= (eta_f[:, :, 2]     .- eta_f[:, :, 1])        ./ delt_s
            deta_dt[:, :, end]     .= (eta_f[:, :, end]   .- eta_f[:, :, end-1])    ./ delt_s


            # ---- ∇η : centred differences onto staggered interior grid ----
            detadx = (eta_f[3:nx,   :, :] .- eta_f[1:nx-2, :, :]) ./
                     (dx[2:nx-1, :] .+ dx[1:nx-2, :])   # (nx-2, ny,   nt)
            detady = (eta_f[:, 3:ny, :] .- eta_f[:, 1:ny-2, :]) ./
                     (dy[:, 1:ny-2] .+ dy[:, 2:ny-1])   # (nx,   ny-2, nt)


            # ---- ∇d : time-independent ----
            d2d  = dropdims(depth, dims=3)
            dddx = (d2d[3:nx,   2:ny-1] .- d2d[1:nx-2, 2:ny-1]) ./
                   (dx[2:nx-1, 2:ny-1]  .+ dx[1:nx-2,  2:ny-1])   # (nx-2, ny-2)
            dddy = (d2d[2:nx-1, 3:ny  ] .- d2d[2:nx-1, 1:ny-2]) ./
                   (dy[2:nx-1, 2:ny-1]  .+ dy[2:nx-1,  1:ny-2])   # (nx-2, ny-2)


            # ---- Trim all fields to interior (nx-2, ny-2) ----
            UDA_int     = UDA[2:end-1, 2:end-1, :]       # (nx-2, ny-2, nt)
            VDA_int     = VDA[2:end-1, 2:end-1, :]
            pb_int      = pb[2:end-1,  2:end-1, :]
            ps_int      = ps[2:end-1,  2:end-1, :]
            deta_dt_int = deta_dt[2:end-1, 2:end-1, :]
            detadx_int  = detadx[:, 2:end-1, :]          # (nx-2, ny-2, nt)
            detady_int  = detady[2:end-1, :, :]          # (nx-2, ny-2, nt)


            # ====================================================================
            # Cb = −U_H · (p''_{−d} ∇d)                    [bottom, Eq. 16]
            # Cs = −(p''_η ∂η/∂t + U_H · (p''_η ∇η))      [surface, Eq. 16]
            # C  = Cb + Cs
            # ====================================================================
            Cb = .-(UDA_int .* reshape(dddx, nx-2, ny-2, 1) .+
                    VDA_int .* reshape(dddy, nx-2, ny-2, 1)) .* pb_int


            Cs = .-(ps_int .* deta_dt_int .+
                    UDA_int .* ps_int .* detadx_int .+
                    VDA_int .* ps_int .* detady_int)


            C_full = Cb .+ Cs   # (nx-2, ny-2, nt)


            # ---- Average into 3-day bins ----
            ca_3day       = zeros(nx-2, ny-2, nt3)
            hrs_per_chunk = 3 * 24
            for t in 1:nt3
                t_start = (t-1) * hrs_per_chunk + 1
                t_end   = min(t * hrs_per_chunk, nt)
                ca_3day[:, :, t] .= mean(C_full[:, :, t_start:t_end], dims=3)
            end


            open(joinpath(base2, "Conv_eta_3day", "Conv_eta_3day_$suffix2.bin"), "w") do io
                write(io, Float32.(ca_3day))
            end
            println("  Completed tile: $suffix (3-day, Cb+Cs)")
        end
    end
    println("Done: $nt3 3-day periods")




elseif time_mode == "weekly"
    println("Computing FULL conversion (Cb + Cs) for weekly window Apr 22-28 ($nt_week snapshots)")
    mkpath(joinpath(base2, "Conv_eta_weekly"))


    for xn in cfg["xn_start"]:cfg["xn_end"]
        for yn in cfg["yn_start"]:cfg["yn_end"]
            suffix  = @sprintf("%02dx%02d_%d", xn, yn, buf)
            suffix2 = @sprintf("%02dx%02d_%d", xn, yn, buf-2)


            # ---- Read & subset to weekly window ----
            rho = Float64.(open(joinpath(base, "Density", "rho_in_$suffix.bin"), "r") do io
                reshape(reinterpret(Float64, read(io, nx*ny*nz*nt*sizeof(Float64))), nx, ny, nz, nt)
            end)[:, :, :, idx_start:idx_end]


            hFacC   = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
            DRFfull = hFacC .* DRF3d
            depth   = sum(DRFfull, dims=3)
            DRFfull[hFacC .== 0] .= 0.0


            fu = Float64.(open(joinpath(base2, "UVW_F", "fu_$suffix.bin"), "r") do io
                reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
            end)[:, :, :, idx_start:idx_end]


            fv = Float64.(open(joinpath(base2, "UVW_F", "fv_$suffix.bin"), "r") do io
                reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
            end)[:, :, :, idx_start:idx_end]


            eta_raw = Float64.(open(joinpath(base, "Eta", "Eta_$suffix.bin"), "r") do io
                reshape(reinterpret(Float32, read(io, nx*ny*nt*sizeof(Float32))), nx, ny, nt)
            end)[:, :, idx_start:idx_end]


            dx = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
            dy = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))


            # ---- Filter ----
            fr    = bandpassfilter(rho,     T1, T2, delt, N, nt_week)
            eta_f = bandpassfilter(eta_raw, T1, T2, delt, N, nt_week)


            # ---- Depth-averaged velocities ----
            UDA = dropdims(sum(fu .* DRFfull, dims=3) ./ depth; dims=3)
            VDA = dropdims(sum(fv .* DRFfull, dims=3) ./ depth; dims=3)


            # ---- Pressure perturbation ----
            pres  = g .* cumsum(fr .* DRFfull, dims=3)
            pfz   = cat(zeros(nx, ny, 1, nt_week), pres; dims=3)
            pc_3d = 0.5 .* (pfz[:, :, 1:end-1, :] .+ pfz[:, :, 2:end, :])
            pa    = sum(pc_3d .* DRFfull, dims=3) ./ depth
            pp_3d = pc_3d .- pa


            ps = pp_3d[:, :, 1,   :]
            pb = pp_3d[:, :, end, :]


            # ---- ∂η/∂t ----
            deta_dt = similar(eta_f)
            deta_dt[:, :, 2:end-1] .= (eta_f[:, :, 3:end] .- eta_f[:, :, 1:end-2]) ./ (2 * delt_s)
            deta_dt[:, :, 1]       .= (eta_f[:, :, 2]     .- eta_f[:, :, 1])        ./ delt_s
            deta_dt[:, :, end]     .= (eta_f[:, :, end]   .- eta_f[:, :, end-1])    ./ delt_s


            # ---- ∇η ----
            detadx = (eta_f[3:nx,   :, :] .- eta_f[1:nx-2, :, :]) ./
                     (dx[2:nx-1, :] .+ dx[1:nx-2, :])   # (nx-2, ny,   nt_week)
            detady = (eta_f[:, 3:ny, :] .- eta_f[:, 1:ny-2, :]) ./
                     (dy[:, 1:ny-2] .+ dy[:, 2:ny-1])   # (nx,   ny-2, nt_week)


            # ---- ∇d ----
            d2d  = dropdims(depth, dims=3)
            dddx = (d2d[3:nx,   2:ny-1] .- d2d[1:nx-2, 2:ny-1]) ./
                   (dx[2:nx-1, 2:ny-1]  .+ dx[1:nx-2,  2:ny-1])
            dddy = (d2d[2:nx-1, 3:ny  ] .- d2d[2:nx-1, 1:ny-2]) ./
                   (dy[2:nx-1, 2:ny-1]  .+ dy[2:nx-1,  1:ny-2])


            # ---- Trim to interior ----
            UDA_int     = UDA[2:end-1, 2:end-1, :]
            VDA_int     = VDA[2:end-1, 2:end-1, :]
            pb_int      = pb[2:end-1,  2:end-1, :]
            ps_int      = ps[2:end-1,  2:end-1, :]
            deta_dt_int = deta_dt[2:end-1, 2:end-1, :]
            detadx_int  = detadx[:, 2:end-1, :]
            detady_int  = detady[2:end-1, :, :]


            # ---- Cb and Cs ----
            Cb = .-(UDA_int .* reshape(dddx, nx-2, ny-2, 1) .+
                    VDA_int .* reshape(dddy, nx-2, ny-2, 1)) .* pb_int


            Cs = .-(ps_int .* deta_dt_int .+
                    UDA_int .* ps_int .* detadx_int .+
                    VDA_int .* ps_int .* detady_int)


            C_full = Cb .+ Cs


            # ---- Weekly mean ----
            ca = dropdims(mean(C_full; dims=3); dims=3)


            open(joinpath(base2, "Conv_eta_weekly", "Conv_eta_weekly_$suffix2.bin"), "w") do io
                write(io, Float32.(ca))
            end
            println("  Completed tile: $suffix (weekly, Cb+Cs)")
        end
    end
    println("Done: weekly window Apr 22-28")




elseif time_mode == "full"
    println("Computing FULL conversion (Cb + Cs) for full time average")
    mkpath(joinpath(base2, "Conv_eta"))


    for xn in cfg["xn_start"]:cfg["xn_end"]
        for yn in cfg["yn_start"]:cfg["yn_end"]
            suffix  = @sprintf("%02dx%02d_%d", xn, yn, buf)
            suffix2 = @sprintf("%02dx%02d_%d", xn, yn, buf-2)


            rho = Float64.(open(joinpath(base, "Density", "rho_in_$suffix.bin"), "r") do io
                reshape(reinterpret(Float64, read(io, nx*ny*nz*nt*sizeof(Float64))), nx, ny, nz, nt)
            end)


            hFacC   = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
            DRFfull = hFacC .* DRF3d
            depth   = sum(DRFfull, dims=3)
            DRFfull[hFacC .== 0] .= 0.0


            fu = Float64.(open(joinpath(base2, "UVW_F", "fu_$suffix.bin"), "r") do io
                reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
            end)


            fv = Float64.(open(joinpath(base2, "UVW_F", "fv_$suffix.bin"), "r") do io
                reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
            end)


            eta_raw = Float64.(open(joinpath(base, "Eta", "Eta_$suffix.bin"), "r") do io
                reshape(reinterpret(Float32, read(io, nx*ny*nt*sizeof(Float32))), nx, ny, nt)
            end)


            dx = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
            dy = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))


            fr    = bandpassfilter(rho,     T1, T2, delt, N, nt)
            eta_f = bandpassfilter(eta_raw, T1, T2, delt, N, nt)


            UDA = dropdims(sum(fu .* DRFfull, dims=3) ./ depth; dims=3)
            VDA = dropdims(sum(fv .* DRFfull, dims=3) ./ depth; dims=3)


            pres  = g .* cumsum(fr .* DRFfull, dims=3)
            pfz   = cat(zeros(nx, ny, 1, nt), pres; dims=3)
            pc_3d = 0.5 .* (pfz[:, :, 1:end-1, :] .+ pfz[:, :, 2:end, :])
            pa    = sum(pc_3d .* DRFfull, dims=3) ./ depth
            pp_3d = pc_3d .- pa


            ps = pp_3d[:, :, 1,   :]
            pb = pp_3d[:, :, end, :]


            # ---- ∂η/∂t ----
            deta_dt = similar(eta_f)
            deta_dt[:, :, 2:end-1] .= (eta_f[:, :, 3:end] .- eta_f[:, :, 1:end-2]) ./ (2 * delt_s)
            deta_dt[:, :, 1]       .= (eta_f[:, :, 2]     .- eta_f[:, :, 1])        ./ delt_s
            deta_dt[:, :, end]     .= (eta_f[:, :, end]   .- eta_f[:, :, end-1])    ./ delt_s


            detadx = (eta_f[3:nx,   :, :] .- eta_f[1:nx-2, :, :]) ./
                     (dx[2:nx-1, :] .+ dx[1:nx-2, :])
            detady = (eta_f[:, 3:ny, :] .- eta_f[:, 1:ny-2, :]) ./
                     (dy[:, 1:ny-2] .+ dy[:, 2:ny-1])


            d2d  = dropdims(depth, dims=3)
            dddx = (d2d[3:nx,   2:ny-1] .- d2d[1:nx-2, 2:ny-1]) ./
                   (dx[2:nx-1, 2:ny-1]  .+ dx[1:nx-2,  2:ny-1])
            dddy = (d2d[2:nx-1, 3:ny  ] .- d2d[2:nx-1, 1:ny-2]) ./
                   (dy[2:nx-1, 2:ny-1]  .+ dy[2:nx-1,  1:ny-2])


            UDA_int     = UDA[2:end-1, 2:end-1, :]
            VDA_int     = VDA[2:end-1, 2:end-1, :]
            pb_int      = pb[2:end-1,  2:end-1, :]
            ps_int      = ps[2:end-1,  2:end-1, :]
            deta_dt_int = deta_dt[2:end-1, 2:end-1, :]
            detadx_int  = detadx[:, 2:end-1, :]
            detady_int  = detady[2:end-1, :, :]


            Cb = .-(UDA_int .* reshape(dddx, nx-2, ny-2, 1) .+
                    VDA_int .* reshape(dddy, nx-2, ny-2, 1)) .* pb_int


            Cs = .-(ps_int .* deta_dt_int .+
                    UDA_int .* ps_int .* detadx_int .+
                    VDA_int .* ps_int .* detady_int)


            C_full = Cb .+ Cs
            ca     = dropdims(mean(C_full; dims=3); dims=3)


            open(joinpath(base2, "Conv_eta", "Conv_eta_$suffix2.bin"), "w") do io
                write(io, Float32.(ca))
            end
            println("  Completed tile: $suffix (full avg, Cb+Cs)")
        end
    end
    println("Done: full time average")


else
    error("Unknown time_mode '$time_mode'. Choose \"3day\", \"weekly\", or \"full\".")
end




