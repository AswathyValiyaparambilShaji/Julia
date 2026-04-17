using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra
using CairoMakie, SparseArrays

include("/home3/avaliyap/Documents/julia/FluxUtils.jl")
include("/home3/avaliyap/Documents/julia/functions/butter_filters.jl")
include("/home3/avaliyap/Documents/julia/functions/coriolis_frequency.jl")
using .FluxUtils: read_bin, bandpassfilter

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

kz = 1
dt = 25
dto = 144
Tts = 366192
nt = div(Tts, dto)

# --- Thickness & constants ---
thk = matread("/nobackup/avaliyap/Box56/hFacC/thk90.mat")["thk90"]
DRF = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)
g = 9.8

# --- Bandpass Filter Parameters ---
dth = 1.0      # Time step
N = 4          # Butterworth filter order

# --- File / tile info ---
base = "/nobackup/avaliyap/Box56/"
base2 = "/nobackup/avaliyap/Box56/NIW1/"

for xn in 1:6
    for yn in 1:7
        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)

        # --- Read fields ---
        rho = read_bin(joinpath(base, "rho/rho_$suffix.bin"), (nx, ny, nz, nt))
        rho[isnan.(rho)] .= 0
        hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
        U = read_bin(joinpath(base, "U/U_$suffix.bin"), (nx, ny, nz, nt))
        V = read_bin(joinpath(base, "V/V_$suffix.bin"), (nx, ny, nz, nt))
        W = read_bin(joinpath(base, "W/W_$suffix.bin"), (nx, ny, nz, nt))

        DRFfull = hFacC .* DRF3d
        z = cumsum(DRFfull, dims=3)
        depth = sum(DRFfull, dims=3)
        DRFfull[hFacC .== 0] .= 0.0

        # C-grid to centers
        uc = 0.5 .* (U[1:end-1, :, :, :] .+ U[2:end, :, :, :])
        vc = 0.5 .* (V[:, 1:end-1, :, :] .+ V[:, 2:end, :, :])
        wc = 0.5 .* (W[:, :, 1:end-1, :] .+ W[:, :, 2:end, :])

        ucc = cat(uc, zeros(1, ny, nz, nt); dims=1)
        vcc = cat(vc, zeros(nx, 1, nz, nt); dims=2)
        wcc = cat(wc, zeros(nx, ny, 1, nt); dims=3)
        # =========================
        # 1) Bring time to the first dim via a time loop
        # =========================
        T = Float32
        uf_tf = Array{T}(undef, nt, nx, ny, nz)
        vf_tf = Array{T}(undef, nt, nx, ny, nz)
        wf_tf = Array{T}(undef, nt, nx, ny, nz)
        rho_tf = Array{T}(undef, nt, nx, ny, nz)

        for t in 1:nt
            @views begin
                uf_tf[t, :, :, :] = ucc[:, :, :, t]
                vf_tf[t, :, :, :] = vcc[:, :, :, t]
                wf_tf[t, :, :, :] = wcc[:, :, :, t]
                rho_tf[t, :, :, :] = rho[:, :, :, t]
            end
        end

        # =========================
        # 2) Loop over latitude, filter each lat slice
        # =========================
        uf_tff = similar(uf_tf)
        vf_tff = similar(vf_tf)
        wf_tff = similar(wf_tf)
        rho_tff = similar(rho_tf)

        for j in 1:ny
            # Get current latitude value
            #current_lat = lat[yn + j - 1] # wrong
            
        
            f_rad_s = coriolis_frequency(current_lat)
            f_cph = f_rad_s * 3600 / (2π)  # Convert rad/s to cycles/hour
            
            # Calculate frequency limits in cph
            f_lower = 0.8 * f_cph  # Lower frequency limit (0.8 * f0)
            f_upper = 1/13.62 #1.2 * f_cph  # Upper frequency limit in cph 
            

                uf_tff[:, :, j, :] = uf_tf[:, :, j, :]
                vf_tff[:, :, j, :] = vf_tf[:, :, j, :]
                wf_tff[:, :, j, :] = wf_tf[:, :, j, :]
                rho_tff[:, :, j, :] = rho_tf[:, :, j, :]

            Tl = 1.0 / f_upper  # Shorter period = 13.21 hours (higher frequency) changed to 1.2f0
            Th = 1.0 / f_lower  # Longer period (lower frequency = 0.8*f)
            
            # Filter each longitude and depth point separately
            for i in 1:nx
                for k in 1:nz
                    # Extract time series at this (lon, lat, depth) point
                    uf_tff[:, i, j, k] = bandpass_butter(uf_tf[:, i, j, k], Tl, Th, dth, N)
                    vf_tff[:, i, j, k] = bandpass_butter(vf_tf[:, i, j, k], Tl, Th, dth, N)
                    wf_tff[:, i, j, k] = bandpass_butter(wf_tf[:, i, j, k], Tl, Th, dth, N)
                    rho_tff[:, i, j, k] = bandpass_butter(rho_tf[:, i, j, k], Tl, Th, dth, N)
                end
            end
        end

        # =========================
        # 3) Put time back to LAST dim
        # =========================
        ufl = similar(ucc)
        vfl = similar(vcc)
        wfl = similar(wcc)
        rhofl = similar(rho)

        for t in 1:nt
            @views begin
                ufl[:, :, :, t] = uf_tff[t, :, :, :]
                vfl[:, :, :, t] = vf_tff[t, :, :, :]
                wfl[:, :, :, t] = wf_tff[t, :, :, :]
                rhofl[:, :, :, t] = rho_tff[t, :, :, :]
            end
        end
            # --- Pressure & perturbations ---
        pres = g .* cumsum(rhofl .* DRFfull, dims=3)
        pfz = cat(zeros(nx, ny, 1, nt), pres; dims=3)
        pc_3d = 0.5 .* (pfz[:, :, 1:end-1, :] .+ pfz[:, :, 2:end, :])
        pa = sum(pc_3d .* DRFfull, dims=3) ./ depth
        pp_3d = pc_3d .- pa

        mask4D = reshape(hFacC .== 0, nx, ny, nz, 1)
        pp_3d[repeat(mask4D, 1, 1, 1, size(pp_3d, 4))] .= 0

        ucA_3d = sum(ufl .* DRFfull, dims=3) ./ depth
        up_3d = ufl .- ucA_3d
        up_3d[repeat(mask4D, 1, 1, 1, size(up_3d, 4))] .= 0

        vcA_3d = sum(vfl .* DRFfull, dims=3) ./ depth
        vp_3d = vfl .- vcA_3d
        vp_3d[repeat(mask4D, 1, 1, 1, size(vp_3d, 4))] .= 0

        wcA_3d = sum(wfl .* DRFfull, dims=3) ./ depth
        wp_3d = wfl .- wcA_3d
        wp_3d[repeat(mask4D, 1, 1, 1, size(wp_3d, 4))] .= 0

        # --- Fluxes (time-mean then vertical integrate) ---
        xflx_3d = up_3d .* pp_3d
        yflx_3d = vp_3d .* pp_3d
        zflx_3d = wp_3d .* pp_3d

        xfm_3d = mean(xflx_3d, dims=4)
        yfm_3d = mean(yflx_3d, dims=4)
        zfm_3d = mean(zflx_3d, dims=4)

        xfdm_3d = sum(xfm_3d .* DRFfull, dims=3)
        yfdm_3d = sum(yfm_3d .* DRFfull, dims=3)
        zfdm_3d = sum(zfm_3d .* DRFfull, dims=3)

        # --- Save flux means ---
        mkpath(joinpath(base,"NIW1"))
        base2 = "/nobackup/avaliyap/Box56/NIW1/"
        mkpath(joinpath(base2, "xflux"))
        mkpath(joinpath(base2, "yflux"))
        mkpath(joinpath(base2, "zflux"))
        open(joinpath(base2, "xflux", "xflx_niw1_$suffix.bin"), "w") do io
            write(io, xfm_3d)
        end
        open(joinpath(base2, "yflux", "yflx_niw1_$suffix.bin"), "w") do io
            write(io, yfm_3d)
        end
        open(joinpath(base2, "zflux", "zflx_niw1_$suffix.bin"), "w") do io
            write(io, zfm_3d)
        end

        println("Completed tile: $suffix")
    end
end

println("All tiles processed and saved.")

