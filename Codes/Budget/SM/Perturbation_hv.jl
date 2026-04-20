using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML


include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter


config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg   = TOML.parsefile(config_file)
base  = cfg["base_path"]
base2 = cfg["base_path2"]


mkpath(joinpath(base2, "xflux_corr"))
mkpath(joinpath(base2, "yflux_corr"))
mkpath(joinpath(base2, "zflux_corr"))


buf      = 3
tx, ty   = 47, 66
nx, ny   = tx + 2*buf, ty + 2*buf
nz       = 88
dto      = 144
Tts      = 366192
nt       = div(Tts, dto)
hrs_3day = 72
nt_avg   = div(nt, hrs_3day)


ρ0 = 999.8
g  = 9.8


thk   = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF   = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)


T1, T2, delt, N_filt = 9.0, 15.0, 1.0, 4




for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]


        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
        println("Processing tile: $suffix")


        # ── 1. Grid geometry ──────────────────────────────────────────────────
        hFacC   = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
        DRFfull = hFacC .* DRF3d
        DRFfull[hFacC .== 0] .= 0.0


        H     = sum(DRFfull, dims=3)
        DRF4d = reshape(DRFfull, nx, ny, nz, 1)
        H_4d  = reshape(H, nx, ny, 1, 1)


        # za (nx,ny,nz): cell-centre depths, positive downward
        z_edge = cat(zeros(nx, ny, 1), cumsum(DRFfull, dims=3); dims=3)
        za     = 0.5 .* (z_edge[:, :, 1:end-1] .+ z_edge[:, :, 2:end])


        mask3D       = hFacC .== 0
        mask4D_proto = reshape(mask3D, nx, ny, nz, 1)
        mask4D       = repeat(mask4D_proto, 1, 1, 1, nt)


        # ── 2. p′ = P - P_barotropic ──────────────────────────────────────────
        rho = Float64.(open(joinpath(base, "Density", "rho_in_$suffix.bin"), "r") do io
            raw = read(io, nx*ny*nz*nt*sizeof(Float64))
            reshape(reinterpret(Float64, raw), nx, ny, nz, nt)
        end)


        rhob  = bandpassfilter(rho, T1, T2, delt, N_filt, nt)
        rho   = nothing; GC.gc()


        pres  = g .* cumsum(rhob .* DRF4d, dims=3)
        pfz   = cat(zeros(nx, ny, 1, nt), pres; dims=3)
        pc    = 0.5 .* (pfz[:, :, 1:end-1, :] .+ pfz[:, :, 2:end, :])
        pres  = nothing; pfz = nothing; rhob = nothing; GC.gc()


        pp    = pc .- sum(pc .* DRF4d, dims=3) ./ H_4d
        pp[mask4D] .= 0.0
        pc    = nothing; GC.gc()


        # ── 3. Bandpass η (nx,ny,nt) ──────────────────────────────────────────
        eta_raw = Float64.(open(joinpath(base, "Eta", "Eta_$suffix.bin"), "r") do io
            raw = read(io, nx*ny*nt*sizeof(Float32))
            reshape(reinterpret(Float32, raw), nx, ny, nt)
        end)


        eta     = bandpassfilter(eta_raw, T1, T2, delt, N_filt, nt)
        eta_raw = nothing; GC.gc()


        # ── 4. N²: 3-day blocks → cell centres → expand to hourly ────────────
        N2p = Float64.(open(joinpath(base, "3day_mean", "N2",
                                     "N2_3day_$suffix.bin"), "r") do io
            raw = read(io, nx*ny*nz*nt_avg*sizeof(Float32))
            reshape(reinterpret(Float32, raw), nx, ny, nz, nt_avg)
        end)


        N2a = zeros(Float64, nx, ny, nz+1, nt_avg)
        N2a[:, :, 1,    :] .= N2p[:, :, 1,      :]
        N2a[:, :, 2:nz, :] .= N2p[:, :, 1:nz-1, :]
        N2a[:, :, nz+1, :] .= N2p[:, :, nz-1,   :]
        N2p = nothing


        N2c = zeros(Float64, nx, ny, nz, nt_avg)
        for k in 1:nz
            N2c[:, :, k, :] .= 0.5 .* (N2a[:, :, k, :] .+ N2a[:, :, k+1, :])
        end
        N2a = nothing; GC.gc()


        # clamp: must catch NaN explicitly — NaN < threshold = false in Julia
        N2thr = 1.0e-8
        n_nan = sum(isnan.(N2c))
        n_low = sum(N2c .< N2thr)
        println("  N2 NaN: $n_nan | below threshold: $n_low | before: $(extrema(filter(!isnan, N2c)))")
        N2c[isnan.(N2c) .| (N2c .< N2thr)] .= N2thr
        println("  N2 after clamp: $(extrema(N2c))")


        N2_4d = zeros(Float64, nx, ny, nz, nt)
        for b in 1:nt_avg
            ts = (b-1)*hrs_3day + 1
            te = min(b*hrs_3day, nt)
            N2_4d[:, :, :, ts:te] .= N2c[:, :, :, b:b]
        end
        N2c = nothing; GC.gc()


        # ── 5. ζbt = η(z+H)/H  [Eq. 5] ───────────────────────────────────────
        # +down convention: z+H = H-za
        zbt = reshape(eta, nx, ny, 1, nt) .*
              (H_4d .- reshape(za, nx, ny, nz, 1)) ./ H_4d
        zbt[mask4D] .= 0.0


        # ── 6. pη = ρ0·g·η − ∫ρ0·N²·ζbt dz′  [Eq. 6] ───────────────────────
        p_eta = ρ0 .* g .* reshape(eta, nx, ny, 1, nt) .-
                cumsum(ρ0 .* N2_4d .* zbt .* DRF4d, dims=3)
        p_eta[mask4D] .= 0.0
        zbt = nothing; N2_4d = nothing; eta = nothing; GC.gc()


        # ── 7. pint = p′ − pη  →  P - P_barotropic - P_heaving  [Eq. 4] ──────
        pint = pp .- p_eta
        pint[mask4D] .= 0.0
        pp = nothing; p_eta = nothing; GC.gc()


        # ── 8. Bandpassed velocities ──────────────────────────────────────────
        fu = Float64.(open(joinpath(base2, "UVW_F", "fu_$suffix.bin"), "r") do io
            raw = read(io, nx*ny*nz*nt*sizeof(Float32))
            reshape(reinterpret(Float32, raw), nx, ny, nz, nt)
        end)


        fv = Float64.(open(joinpath(base2, "UVW_F", "fv_$suffix.bin"), "r") do io
            raw = read(io, nx*ny*nz*nt*sizeof(Float32))
            reshape(reinterpret(Float32, raw), nx, ny, nz, nt)
        end)


        fw = Float64.(open(joinpath(base2, "UVW_F", "fw_$suffix.bin"), "r") do io
            raw = read(io, nx*ny*nz*nt*sizeof(Float32))
            reshape(reinterpret(Float32, raw), nx, ny, nz, nt)
        end)


        # ── 9. Baroclinic velocities (depth-mean removed) ─────────────────────
        up = fu .- sum(fu .* DRF4d, dims=3) ./ H_4d
        up[mask4D] .= 0.0
        fu = nothing; GC.gc()


        vp = fv .- sum(fv .* DRF4d, dims=3) ./ H_4d
        vp[mask4D] .= 0.0
        fv = nothing; GC.gc()


        wp = fw .- sum(fw .* DRF4d, dims=3) ./ H_4d
        wp[mask4D] .= 0.0
        fw = nothing; GC.gc()


        # ── 10. Time-mean pressure fluxes u′p′, v′p′, w′p′ ───────────────────
        xfm = mean(up .* pint, dims=4)
        up  = nothing; GC.gc()


        yfm = mean(vp .* pint, dims=4)
        vp  = nothing; GC.gc()


        zfm = mean(wp .* pint, dims=4)
        wp  = nothing; pint = nothing; GC.gc()


        # ── 11. Depth-integrated 2D maps ──────────────────────────────────────
        xvi = sum(xfm .* DRFfull, dims=3)
        yvi = sum(yfm .* DRFfull, dims=3)
        zvi = sum(zfm .* DRFfull, dims=3)


        # ── 12. Save 3D fluxes ────────────────────────────────────────────────
        open(joinpath(base2, "xflux_corr", "xflx_$suffix.bin"),         "w") do io; write(io, Float32.(xfm)); end
        open(joinpath(base2, "yflux_corr", "yflx_$suffix.bin"),         "w") do io; write(io, Float32.(yfm)); end
        open(joinpath(base2, "zflux_corr", "zflx_$suffix.bin"),         "w") do io; write(io, Float32.(zfm)); end



        println("  Saved: $suffix")


        xfm = nothing; yfm = nothing; zfm = nothing
        xvi = nothing; yvi = nothing; zvi = nothing
        GC.gc()


    end
end




