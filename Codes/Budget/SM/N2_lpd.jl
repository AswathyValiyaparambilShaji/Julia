using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML
using Impute


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
include(joinpath(@__DIR__, "..","..","..", "functions", "densjmd95.jl"))
include(joinpath(@__DIR__, "..","..","..", "functions", "butter_filters.jl"))


using .FluxUtils: read_bin, bandpassfilter
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base  = cfg["base_path"]
base2 = cfg["base_path2"]


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


dt  = 1
dto = 144
Tts = 366192
nt  = div(Tts, dto)


# --- Thickness & constants ---
thk   = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF   = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)
g     = 9.8
rho0  = 999.8


println("Total hourly timesteps: $nt")


mkpath(joinpath(base2, "N2_lpd"))


for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]


        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
        println("Processing tile: $suffix")


        # --- Read raw hourly fields ---
        Salt  = Float64.(read_bin(joinpath(base, "Salt/Salt_$suffix.bin"),  (nx, ny, nz, nt)))
        Theta = Float64.(read_bin(joinpath(base, "Theta/Theta_$suffix.bin"), (nx, ny, nz, nt)))


        hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))


        # --- Calculate depths ---
        DRFfull      = hFacC .* DRF3d
        DRFfull[hFacC .== 0] .= 0.0


        z_cumsum     = cumsum(DRFfull, dims=3)
        zz           = cat(zeros(nx, ny, 1), z_cumsum; dims=3)
        z_centers    = -0.5 .* (zz[:, :, 1:end-1] .+ zz[:, :, 2:end])    # (nx, ny, nz)
        z_interfaces = -zz[:, :, 2:end-1]                                  # (nx, ny, nz-1)
        Δz           = z_centers[:, :, 2:end] .- z_centers[:, :, 1:end-1] # (nx, ny, nz-1)


        # Broadcast interface depth over time: (nx, ny, nz-1, nt)
        z_int_4d = repeat(z_interfaces, outer=(1, 1, 1, nt))
        Δz_4d    = repeat(Δz,          outer=(1, 1, 1, nt))


        # --- Compute rho_upper and rho_lower at interfaces for all timesteps ---
        println("  Computing interface densities...")
        rho_upper = densjmd95(Salt[:, :, 1:end-1, :], Theta[:, :, 1:end-1, :], z_int_4d)  # (nx, ny, nz-1, nt)
        rho_lower = densjmd95(Salt[:, :, 2:end,   :], Theta[:, :, 2:end,   :], z_int_4d)  # (nx, ny, nz-1, nt)
        Salt = nothing; Theta = nothing; z_int_4d = nothing; GC.gc()


        # --- Low-pass filter rho_upper along time ---
        println("  Low-pass filtering interface densities (Tcut=36 hr)...")
        rho_upper_2d   = permutedims(rho_upper, (4, 1, 2, 3))
        rho_upper_2d   = reshape(rho_upper_2d, nt, nx*ny*(nz-1))
        rho_upper      = nothing; GC.gc()
        rho_upper_filt = lowhighpass_butter(rho_upper_2d, 36.0, dt, 4, "low")
        rho_upper_2d   = nothing; GC.gc()
        rho_upper_filt = reshape(rho_upper_filt, nt, nx, ny, nz-1)
        rho_upper_filt = permutedims(rho_upper_filt, (2, 3, 4, 1))    # (nx, ny, nz-1, nt)


        # --- Low-pass filter rho_lower along time ---
        rho_lower_2d   = permutedims(rho_lower, (4, 1, 2, 3))
        rho_lower_2d   = reshape(rho_lower_2d, nt, nx*ny*(nz-1))
        rho_lower      = nothing; GC.gc()
        rho_lower_filt = lowhighpass_butter(rho_lower_2d, 36.0, dt, 4, "low")
        rho_lower_2d   = nothing; GC.gc()
        rho_lower_filt = reshape(rho_lower_filt, nt, nx, ny, nz-1)
        rho_lower_filt = permutedims(rho_lower_filt, (2, 3, 4, 1))    # (nx, ny, nz-1, nt)


        # --- Compute N² from filtered interface densities ---
        println("  Computing N²...")
        Δρ             = rho_lower_filt .- rho_upper_filt
        rho_upper_filt = nothing; rho_lower_filt = nothing; GC.gc()


        N2_interfaces = -(g / rho0) .* (Δρ ./ Δz_4d)                 # (nx, ny, nz-1, nt)
        Δρ = nothing; Δz_4d = nothing; GC.gc()


        N2 = zeros(Float64, nx, ny, nz, nt)
        N2[:, :, 2:end, :] = N2_interfaces
        N2[:, :, 1,     :] = N2_interfaces[:, :, 1, :]
        N2_interfaces = nothing; GC.gc()


        # --- Set negative values to NaN ---
        println("  Setting negative values to NaN...")
        N2[N2 .< 0] .= NaN
        n_nan_before = sum(isnan.(N2))
        println("  NaN values before filling: $n_nan_before")


        # --- Fill NaN values (per profile — cannot be vectorized) ---
        println("  Filling NaN values...")
        for t in 1:nt
            for j in 1:ny
                for i in 1:nx
                    profile = N2[i, j, :, t]
                    if any(isnan, profile)
                        profile_ff     = Impute.locf(profile)
                        profile_filled = Impute.nocb(profile_ff)
                        N2[i, j, :, t] = profile_filled
                    end
                end
            end
        end


        n_nan_after      = sum(isnan.(N2))
        n_negative_after = sum(N2 .< 0)
        println("  NaN values after filling:      $n_nan_after")
        println("  Negative values after filling: $n_negative_after")
        println("  N² range: ", extrema(filter(isfinite, N2)))


        # --- Save ---
        outfile = joinpath(base2, "N2_lpd", "N2_$suffix.bin")
        open(outfile, "w") do io
            write(io, Float32.(N2))
        end


        println("Completed tile: $suffix")
    end
end

println("\nAll tiles processed successfully!")
