using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML, CairoMakie


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


# ── Global 2D arrays to accumulate tile results ───────────────────────────────
xn_range   = cfg["xn_start"]:cfg["xn_end"]
yn_range   = cfg["yn_start"]:cfg["yn_end"]
nx_global  = length(xn_range) * tx
ny_global  = length(yn_range) * ty


pp_global   = fill(NaN32, nx_global, ny_global)   # depth-int, time-mean p′
peta_global = fill(NaN32, nx_global, ny_global)   # depth-int, time-mean pη


for xn in xn_range
    for yn in yn_range


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


        # ── 3. Bandpass η ─────────────────────────────────────────────────────
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


        # ── 5. ζbt = η(z+H)/H ────────────────────────────────────────────────
        zbt = reshape(eta, nx, ny, 1, nt) .*
              (H_4d .- reshape(za, nx, ny, nz, 1)) ./ H_4d
        zbt[mask4D] .= 0.0


        # ── 6. pη = −∫ρ0·N²·ζbt dz′ ──────────────────────────────────────────
        p_eta = .- cumsum(ρ0 .* N2_4d .* zbt .* DRF4d, dims=3)
        p_eta[mask4D] .= 0.0
        zbt = nothing; N2_4d = nothing; eta = nothing; GC.gc()


        # ── 7. pint = p′ − pη ─────────────────────────────────────────────────
        pint = pp .- p_eta
        pint[mask4D] .= 0.0
        pp = nothing; p_eta = nothing; GC.gc()


        # ── 8. Depth-integrate & time-mean → 2D  (strip buffer) ──────────────
        # depth integral: sum(field * DRF, dims=3) → (nx,ny,1,nt)
        # time mean: mean over dim=4 → (nx,ny,1,1) → squeeze to (nx,ny)
        pp_2d   = Float32.(dropdims(mean(
                      sum(pp   .* DRF4d, dims=3), dims=4), dims=(3,4)))
        peta_2d = Float32.(dropdims(mean(
                      sum(p_eta_tile .* DRF4d, dims=3), dims=4), dims=(3,4)))

         #= Calculate tile positions in global grid
            xs = (xn - 1) * tx + 1
            xe = xs + tx + (2 * buf) - 1
            ys = (yn - 1) * ty + 1
            ye = ys + ty + (2 * buf) - 1


            # Update global arrays — full time dimension assigned
            Conv[xs+2:xe-2, ys+2:ye-2, :] .= C[2:end-1, 2:end-1, :]
            FDiv[xs+2:xe-2, ys+2:ye-2, :] .= fxD[2:end-1, 2:end-1, :]


            U_KE_full[xs+2:xe-2,    ys+2:ye-2, :] .= u_ke_3day[buf:nx-buf+1, buf:ny-buf+1, :] =#
        # strip buffer
        pp_int   = pp_2d[buf+1:end-buf,   buf+1:end-buf]
        peta_int = peta_2d[buf+1:end-buf, buf+1:end-buf]


        # place into global arrays
        xi = (xn - cfg["xn_start"]) * tx + 1
        yi = (yn - cfg["yn_start"]) * ty + 1
        pp_global[xi:xi+tx-1,   yi:yi+ty-1] = pp_int
        peta_global[xi:xi+tx-1, yi:yi+ty-1] = peta_int


        pp_2d = nothing; peta_2d = nothing
        pp_int = nothing; peta_int = nothing
        mask4D = nothing; GC.gc()


    end
end


# ── 9. Save global 2D fields ──────────────────────────────────────────────────
open(joinpath(base2, "pp_depthint_timemean.bin"), "w") do io
    write(io, pp_global)
end
open(joinpath(base2, "peta_depthint_timemean.bin"), "w") do io
    write(io, peta_global)
end
println("Saved global 2D fields.")


# ── 10. Plot comparison ───────────────────────────────────────────────────────
# Shared colour range: ±max of both fields (ignore NaN)
vmax_pp   = maximum(abs.(filter(!isnan, pp_global)))
vmax_peta = maximum(abs.(filter(!isnan, peta_global)))


fig = Figure(size = (1200, 500))


ax1 = Axis(fig[1, 1],
    title  = "Depth-integrated, Time-mean  p′  [Pa·m]",
    xlabel = "x tile index", ylabel = "y tile index")
hm1 = heatmap!(ax1, pp_global',
    colormap = :RdBu_r, colorrange = (-vmax_pp, vmax_pp))
Colorbar(fig[1, 2], hm1, label = "Pa·m")


ax2 = Axis(fig[1, 3],
    title  = "Depth-integrated, Time-mean  pη  [Pa·m]",
    xlabel = "x tile index", ylabel = "y tile index")
hm2 = heatmap!(ax2, peta_global',
    colormap = :RdBu_r, colorrange = (-vmax_peta, vmax_peta))
Colorbar(fig[1, 4], hm2, label = "Pa·m")


save(joinpath(base2, "pp_vs_peta_comparison.png"), fig)
println("Plot saved: pp_vs_peta_comparison.png")




