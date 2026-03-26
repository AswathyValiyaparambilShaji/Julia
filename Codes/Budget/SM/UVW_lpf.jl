using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML


include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
include(joinpath(@__DIR__, "..","..","..", "functions", "butter_filters.jl"))
using .FluxUtils: read_bin


config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base  = cfg["base_path"]
base2 = cfg["base_path2"]
mkpath(joinpath(base2, "UVW_LP"))


# --- Domain & grid ---
buf = 3
tx, ty = 47, 66
nx = tx + 2*buf
ny = ty + 2*buf
nz = 88


dto  = 144
Tts  = 366192
nt   = div(Tts, dto)


# --- Low-pass filter parameters ---
delt  = 1.0    # hourly sampling [hr]
Tcut  = 36.0   # low-pass cutoff [hr]
N_ord = 4


for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]


        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
        println("Processing tile: $suffix")


        # --- 1. Read raw U, V, W ---
        U = Float64.(read_bin(joinpath(base, "U", "U_$suffix.bin"), (nx, ny, nz, nt)))
        V = Float64.(read_bin(joinpath(base, "V", "V_$suffix.bin"), (nx, ny, nz, nt)))
        W = Float64.(read_bin(joinpath(base, "W", "W_$suffix.bin"), (nx, ny, nz, nt)))


        # --- 2. C-grid to cell centers ---
        uc = 0.5 .* (U[1:end-1, :, :, :] .+ U[2:end,   :, :, :])
        vc = 0.5 .* (V[:, 1:end-1, :, :] .+ V[:, 2:end, :, :])
        wc = 0.5 .* (W[:, :, 1:end-1, :] .+ W[:, :, 2:end, :])


        ucc = cat(uc, zeros(1,  ny, nz, nt); dims=1)
        vcc = cat(vc, zeros(nx, 1,  nz, nt); dims=2)
        wcc = cat(wc, zeros(nx, ny, 1,  nt); dims=3)


        # --- 3. Reshape to (nt, nx*ny*nz) --- same as N2 pattern ---
        u_2d = reshape(permutedims(ucc, (4,1,2,3)), nt, nx*ny*nz)
        v_2d = reshape(permutedims(vcc, (4,1,2,3)), nt, nx*ny*nz)
        w_2d = reshape(permutedims(wcc, (4,1,2,3)), nt, nx*ny*nz)


        # --- 4. Low-pass filter at 36 hr --- same as N2 ---
        println("  Low-pass filtering U (Tcut=$(Tcut) hr)...")
        u_lp_2d = lowhighpass_butter(u_2d, Tcut, delt, N_ord, "low")


        println("  Low-pass filtering V (Tcut=$(Tcut) hr)...")
        v_lp_2d = lowhighpass_butter(v_2d, Tcut, delt, N_ord, "low")


        println("  Low-pass filtering W (Tcut=$(Tcut) hr)...")
        w_lp_2d = lowhighpass_butter(w_2d, Tcut, delt, N_ord, "low")


        # --- 5. Reshape back to (nx, ny, nz, nt) --- same as N2 ---
        u_lp = permutedims(reshape(u_lp_2d, nt, nx, ny, nz), (2,3,4,1))
        v_lp = permutedims(reshape(v_lp_2d, nt, nx, ny, nz), (2,3,4,1))
        w_lp = permutedims(reshape(w_lp_2d, nt, nx, ny, nz), (2,3,4,1))


        println("  u_lp range: ", extrema(filter(isfinite, u_lp)))
        println("  v_lp range: ", extrema(filter(isfinite, v_lp)))
        println("  w_lp range: ", extrema(filter(isfinite, w_lp)))


        # --- 6. Save ---
        open(joinpath(base2, "UVW_LP", "u_lp_$suffix.bin"), "w") do io
            write(io, Float32.(u_lp))
        end
        open(joinpath(base2, "UVW_LP", "v_lp_$suffix.bin"), "w") do io
            write(io, Float32.(v_lp))
        end
        open(joinpath(base2, "UVW_LP", "w_lp_$suffix.bin"), "w") do io
            write(io, Float32.(w_lp))
        end


        println("Completed tile: $suffix")
    end
end




