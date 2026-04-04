using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML
include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter

config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg  = TOML.parsefile(config_file)
base  = cfg["base_path"]
base2 = cfg["base_path2"]

# --- Tile & grid ---
buf    = 3
tx, ty = 47, 66
nx     = tx + 2 * buf
ny     = ty + 2 * buf
nz     = 88

# --- Time ---
dto  = 144
Tts  = 366192
nt   = div(Tts, dto)

# --- Thickness ---
thk    = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF    = thk[1:nz]
DRF3d  = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)

rho0 = 998.0
g = 9.8
T1, T2, delt, N = 9.0, 15.0, 1.0, 4

mkpath(joinpath(base2, "Conv_z_dI"))
mkpath(joinpath(base2, "Ah0_dI"))

# ============================================================================
# MAIN LOOP OVER ALL TILES
# ============================================================================

for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]

        suffix  = @sprintf("%02dx%02d_%d", xn, yn, buf)
        suffix2 = @sprintf("%02dx%02d_%d", xn, yn, buf - 2)

        out_path  = joinpath(base2, "Conv_z_dI", "Conv_z_$suffix2.bin")
        Ah0_path  = joinpath(base2, "Ah0_dI",    "Ah0_$suffix2.bin")

        println("Processing tile: xn=$xn  yn=$yn  ($suffix)")

        # --- Check all required input files exist ---
        rho_path  = joinpath(base,  "Density",  "rho_in_$suffix.bin")
        hfac_path = joinpath(base,  "hFacC",    "hFacC_$suffix.bin")
        fu_path   = joinpath(base2, "UVW_F",    "fu_$suffix.bin")
        fv_path   = joinpath(base2, "UVW_F",    "fv_$suffix.bin")
        dx_path   = joinpath(base,  "DXC",      "DXC_$suffix.bin")
        dy_path   = joinpath(base,  "DYC",      "DYC_$suffix.bin")

        missing_files = filter(!isfile, [rho_path, hfac_path, fu_path, fv_path, dx_path, dy_path])
        if !isempty(missing_files)
            @warn "Skipping tile $suffix  missing files:" missing_files
            continue
        end

        # --- Read density ---
        rho = Float64.(open(rho_path, "r") do io
            reshape(reinterpret(Float64, read(io, nx * ny * nz * nt * sizeof(Float64))), nx, ny, nz, nt)
        end)

        # --- Grid masks & thickness ---
        hFacC   = read_bin(hfac_path, (nx, ny, nz))
        DRFfull = hFacC .* DRF3d
        depth   = sum(DRFfull, dims=3)
        DRFfull[hFacC .== 0] .= 0.0

        # --- Read filtered velocities ---
        fu = Float64.(open(fu_path, "r") do io
            reshape(reinterpret(Float32, read(io, nx * ny * nz * nt * sizeof(Float32))), nx, ny, nz, nt)
        end)

        fv = Float64.(open(fv_path, "r") do io
            reshape(reinterpret(Float32, read(io, nx * ny * nz * nt * sizeof(Float32))), nx, ny, nz, nt)
        end)

        # --- Grid spacings ---
        dx = read_bin(dx_path, (nx, ny))
        dy = read_bin(dy_path, (nx, ny))

        # --- Bandpass filter ---
        fr = bandpassfilter(rho, T1, T2, delt, N, nt)

        # --- Depth-averaged velocities ---
        UDA = dropdims(sum(fu .* DRFfull, dims=3) ./ depth; dims=3)
        VDA = dropdims(sum(fv .* DRFfull, dims=3) ./ depth; dims=3)

        # --- z and d ---
        d   = depth
        z   = -cumsum(DRFfull, dims=3)
        d2d = dropdims(d, dims=3)

        # --- Term 1: -div(d * U_DA) ---
        dU = dropdims(d, dims=3) .* UDA
        dV = dropdims(d, dims=3) .* VDA

        # --- Term 2: div(U_DA) ---
        divUDA = (
            (UDA[3:nx,   2:ny-1, :] .- UDA[1:nx-2, 2:ny-1, :]) ./
            (dx[2:nx-1,  2:ny-1]    .+ dx[1:nx-2,  2:ny-1])     .+
            (VDA[2:nx-1, 3:ny,   :] .- VDA[2:nx-1, 1:ny-2, :]) ./
            (dy[2:nx-1,  2:ny-1]    .+ dy[2:nx-1,  1:ny-2])
        )

        # --- ∇d (depth gradient, time-independent, same CFD stencil) ---
        dddx = (d2d[3:nx,   2:ny-1] .- d2d[1:nx-2, 2:ny-1]) ./
               (dx[2:nx-1, 2:ny-1]  .+ dx[1:nx-2,  2:ny-1])

        dddy = (d2d[2:nx-1, 3:ny  ] .- d2d[2:nx-1, 1:ny-2]) ./
               (dy[2:nx-1, 2:ny-1]  .+ dy[2:nx-1,  1:ny-2])

        # --- U_H·∇d (barotropic flow advecting depth, time-varying) ---
        UDA_int = UDA[2:nx-1, 2:ny-1, :]   # (nx-2, ny-2, nt)
        VDA_int = VDA[2:nx-1, 2:ny-1, :]   # (nx-2, ny-2, nt)

        UdotGradD = UDA_int .* reshape(dddx, nx-2, ny-2, 1) .+
                    VDA_int .* reshape(dddy, nx-2, ny-2, 1)  # (nx-2, ny-2, nt)

        z_int = z[2:nx-1, 2:ny-1, :]
        d_int = d2d[2:nx-1, 2:ny-1]

        zpd = reshape(z_int, nx-2, ny-2, nz, 1) .+
              reshape(d_int, nx-2, ny-2, 1,  1)  # (z + d) >= 0  size: (nx-2, ny-2, nz, 1)

        # --- W(z) = −(z+d)·∇·U_H − U_H·∇d ---
        Wz = .- zpd .* reshape(divUDA,    nx-2, ny-2, 1,  nt) .-
                       reshape(UdotGradD, nx-2, ny-2, 1, nt)

        # --- 4D arrays ---
        DRFfull4D = repeat(DRFfull, 1, 1, 1, nt)
        depth4D   = repeat(depth,   1, 1, 1, nt)

        # --- Baroclinic density anomaly ---
        rho_prime = fr
        rho_int   = rho_prime[2:nx-1, 2:ny-1, :, :]

        # --- Depth-scaled conversion ---
        Cz = rho_int .* g .* Wz

        # --- Depth-integrate ---
        DRFint   = DRFfull[2:nx-1, 2:ny-1, :]
        DRFint4d = reshape(DRFint, nx-2, ny-2, nz, 1)
        Ca_full  = dropdims(sum(Cz .* DRFint4d, dims=3), dims=3)

        # --- Time mean ---
        ca = dropdims(mean(Ca_full; dims=3); dims=3)

        # --- Save Conv ---
        open(out_path, "w") do io
            write(io, Float32.(ca))
        end
        println("  Saved: Conv_z_$suffix2.bin")

        # ── Ah0 = ρ0·(U·H·Ax + V·H·Ay) ──────────────────────────────────────
        # Depth-integrate baroclinic Reynolds stress products in a z-loop
        # to avoid full 4D allocations.  u' = fu − UDA,  v' = fv − VDA.
        uu = zeros(Float64, nx, ny, nt)
        vu = zeros(Float64, nx, ny, nt)
        uv = zeros(Float64, nx, ny, nt)
        vv = zeros(Float64, nx, ny, nt)
        for k in 1:nz
            u_bc_k = fu[:, :, k, :] .- UDA   # (nx, ny, nt)
            v_bc_k = fv[:, :, k, :] .- VDA   # (nx, ny, nt)
            drfk   = DRFfull[:, :, k]         # (nx, ny)  broadcasts over nt
            uu .+= u_bc_k .* u_bc_k .* drfk
            vu .+= v_bc_k .* u_bc_k .* drfk
            uv .+= u_bc_k .* v_bc_k .* drfk
            vv .+= v_bc_k .* v_bc_k .* drfk
        end

        # H·Ax = ∂/∂x(∫u'u' dz) + ∂/∂y(∫v'u' dz)
        HAx = (uu[3:nx,   2:ny-1, :] .- uu[1:nx-2, 2:ny-1, :]) ./
              (dx[2:nx-1, 2:ny-1]    .+ dx[1:nx-2, 2:ny-1])     .+
              (vu[2:nx-1, 3:ny,   :] .- vu[2:nx-1, 1:ny-2, :]) ./
              (dy[2:nx-1, 2:ny-1]    .+ dy[2:nx-1, 1:ny-2])

        # H·Ay = ∂/∂x(∫u'v' dz) + ∂/∂y(∫v'v' dz)
        HAy = (uv[3:nx,   2:ny-1, :] .- uv[1:nx-2, 2:ny-1, :]) ./
              (dx[2:nx-1, 2:ny-1]    .+ dx[1:nx-2, 2:ny-1])     .+
              (vv[2:nx-1, 3:ny,   :] .- vv[2:nx-1, 1:ny-2, :]) ./
              (dy[2:nx-1, 2:ny-1]    .+ dy[2:nx-1, 1:ny-2])

        # Ah0 = ρ0·(U·H·Ax + V·H·Ay), then time-mean
        Ah0_4d   = rho0 .* (UDA_int .* HAx .+ VDA_int .* HAy)
        Ah0_mean = dropdims(mean(Ah0_4d; dims=3); dims=3)
        println(Ah0_mean[1:10,1:10])
        # --- Save Ah0 ---
        open(Ah0_path, "w") do io
            write(io, Float32.(Ah0_mean))
        end
        println("  Saved: Ah0_$suffix2.bin")

        # --- Free memory ---
        uu = vu = uv = vv = HAx = HAy = Ah0_4d = nothing
        rho = fu = fv = fr = nothing
        GC.gc()

    end  # yn
end  # xn

println("All tiles done.")