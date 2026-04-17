using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML
include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter

config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base = cfg["base_path"]
base2 = cfg["base_path2"]
mkpath(joinpath(base2, "UVW_NIW"))

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
dt = 25
dto = 144
Tts = 366192
nt = div(Tts, dto)

# --- NIW Filter (29-44 hr band) ---
T1_niw, T2_niw, delt, N = 29.0, 44.0, 1.0, 4
fcutlow_niw  = 1 / T2_niw   # 1/44 cphr
fcuthigh_niw = 1 / T1_niw   # 1/29 cphr
fnq = 1 / delt
bpf_niw = digitalfilter(Bandpass(fcutlow_niw, fcuthigh_niw), Butterworth(N); fs = fnq)

# --- Parallelize over tiles ---
for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)

        # --- Read fields ---
        U = read_bin(joinpath(base, "U/U_$suffix.bin"), (nx, ny, nz, nt))
        V = read_bin(joinpath(base, "V/V_$suffix.bin"), (nx, ny, nz, nt))
        W = read_bin(joinpath(base, "W/W_$suffix.bin"), (nx, ny, nz, nt))

        # --- C-grid to centers ---
        uc = 0.5 .* (U[1:end-1, :, :, :] .+ U[2:end,   :, :, :])
        vc = 0.5 .* (V[:, 1:end-1, :, :] .+ V[:, 2:end, :, :])
        wc = 0.5 .* (W[:, :, 1:end-1, :] .+ W[:, :, 2:end, :])
        ucc = cat(uc, zeros(1, ny, nz, nt); dims=1)
        vcc = cat(vc, zeros(nx, 1, nz, nt); dims=2)
        wcc = cat(wc, zeros(nx, ny, 1, nt); dims=3)

        # --- NIW Bandpass filter (29-44 hr) ---
        fu_niw = bandpassfilter(ucc, T1_niw, T2_niw, delt, N, nt)
        fv_niw = bandpassfilter(vcc, T1_niw, T2_niw, delt, N, nt)
        fw_niw = bandpassfilter(wcc, T1_niw, T2_niw, delt, N, nt)

        # --- Save NIW filtered fields ---
        open(joinpath(base2, "UVW_NIW", "fu_niw_$suffix.bin"), "w") do io
            write(io, Float32.(fu_niw))
        end
        open(joinpath(base2, "UVW_NIW", "fv_niw_$suffix.bin"), "w") do io
            write(io, Float32.(fv_niw))
        end
        open(joinpath(base2, "UVW_NIW", "fw_niw_$suffix.bin"), "w") do io
            write(io, Float32.(fw_niw))
        end

        println("Completed NIW tile: $suffix")
    end
end