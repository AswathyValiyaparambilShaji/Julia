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


# ============================================================================
# MAIN LOOP OVER ALL TILES
# ============================================================================


for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]


        suffix  = @sprintf("%02dx%02d_%d", xn, yn, buf)
        suffix2 = @sprintf("%02dx%02d_%d", xn, yn, buf - 2)


        out_path = joinpath(base2, "Conv_z_dI", "Conv_z_$suffix2.bin")


       

        println("Processing tile: xn=$xn  yn=$yn  ($suffix)")


        # --- Check all required input files exist ---
        rho_path = joinpath(base,  "Density",  "rho_in_$suffix.bin")
        hfac_path = joinpath(base, "hFacC",    "hFacC_$suffix.bin")
        fu_path  = joinpath(base2, "UVW_F",    "fu_$suffix.bin")
        fv_path  = joinpath(base2, "UVW_F",    "fv_$suffix.bin")
        dx_path  = joinpath(base,  "DXC",      "DXC_$suffix.bin")
        dy_path  = joinpath(base,  "DYC",      "DYC_$suffix.bin")


        missing_files = filter(!isfile, [rho_path, hfac_path, fu_path, fv_path, dx_path, dy_path])
        if !isempty(missing_files)
            @warn "Skipping tile $suffix — missing files:" missing_files
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
        d = depth
        z = -cumsum(DRFfull, dims=3)


        # --- Term 1: -div(d * U_DA) ---
        dU = dropdims(d, dims=3) .* UDA
        dV = dropdims(d, dims=3) .* VDA


        term1 = .-(
            (dU[3:nx,   2:ny-1, :] .- dU[1:nx-2, 2:ny-1, :]) ./
            (dx[2:nx-1, 2:ny-1]    .+ dx[1:nx-2, 2:ny-1])     .+
            (dV[2:nx-1, 3:ny,   :] .- dV[2:nx-1, 1:ny-2, :]) ./
            (dy[2:nx-1, 2:ny-1]    .+ dy[2:nx-1, 1:ny-2])
        )


        # --- Term 2: div(U_DA) ---
        divUDA = (
            (UDA[3:nx,   2:ny-1, :] .- UDA[1:nx-2, 2:ny-1, :]) ./
            (dx[2:nx-1,  2:ny-1]    .+ dx[1:nx-2,  2:ny-1])     .+
            (VDA[2:nx-1, 3:ny,   :] .- VDA[2:nx-1, 1:ny-2, :]) ./
            (dy[2:nx-1,  2:ny-1]    .+ dy[2:nx-1,  1:ny-2])
        )


        z_int = z[2:nx-1, 2:ny-1, :]


        # --- Wz: vertical velocity from continuity ---
        Wz = reshape(term1,  nx-2, ny-2, 1,  nt) .-
             reshape(z_int,  nx-2, ny-2, nz, 1)  .*
             reshape(divUDA, nx-2, ny-2, 1,  nt)


        # --- 4D arrays ---
        DRFfull4D = repeat(DRFfull, 1, 1, 1, nt)
        depth4D   = repeat(depth,   1, 1, 1, nt)


        # --- Baroclinic density anomaly ---
        rho_prime = fr .- rho0
        rho_int   = rho_prime[2:nx-1, 2:ny-1, :, :]


        # --- Depth-scaled conversion ---
        d_int = d[2:nx-1, 2:ny-1, :]
        sc    = z_int ./ d_int
        Cz    = rho_int .* g .* Wz .* reshape(sc, nx-2, ny-2, nz, 1)


        # --- Depth-integrate ---
        DRFint   = DRFfull[2:nx-1, 2:ny-1, :]
        DRFint4d = reshape(DRFint, nx-2, ny-2, nz, 1)
        Ca_full  = dropdims(sum(Cz .* DRFint4d, dims=3), dims=3)


        # --- Time mean ---
        ca = dropdims(mean(Ca_full; dims=3); dims=3)


        # --- Save ---
        open(out_path, "w") do io
            write(io, Float32.(ca))
        end
        println("  Saved: Conv_z_$suffix2.bin")


        # --- Free memory ---
        rho = fu = fv = fr = nothing
        GC.gc()


    end  # yn
end  # xn


println("All tiles done.")




