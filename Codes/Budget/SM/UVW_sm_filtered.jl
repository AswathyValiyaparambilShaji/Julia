using DSP, MAT, Statistics, Printf, Plots, FilePathsBase, LinearAlgebra, TOML
#using CairoMakie, SparseArrays

include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base = cfg["base_path"]
mkpath(joinpath(base,"SM_I"))
base2 = cfg["base_path2"]
mkpath(joinpath(base2, "UVW_F"))

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


# --- Filter (915 day band, 1 step sampling here) ---
T1, T2, delt, N = 9.0, 15.0, 1.0, 4
fcutlow, fcuthigh = 1 / T2, 1 / T1
fnq = 1 / delt
bpf = digitalfilter(Bandpass(fcutlow, fcuthigh), Butterworth(N); fs = fnq)


# Now parallelize over ALL 42 tiles


for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]


        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)

        
        # --- Read fields ---
        
        U     = read_bin(joinpath(base, "U/U_$suffix.bin"),   (nx, ny, nz, nt))
        V     = read_bin(joinpath(base, "V/V_$suffix.bin"),   (nx, ny, nz, nt))
        W     = read_bin(joinpath(base, "W/W_$suffix.bin"),   (nx, ny, nz, nt))


        # C-grid to centers
        uc = 0.5 .* (U[1:end-1, :, :, :] .+ U[2:end,   :, :, :])
        vc = 0.5 .* (V[:, 1:end-1, :, :] .+ V[:, 2:end, :, :])
        wc = 0.5 .* (W[:, :, 1:end-1, :] .+ W[:, :, 2:end,   :])

        ucc = cat(uc, zeros(1, ny, nz, nt); dims=1)
        vcc = cat(vc, zeros(nx, 1, nz, nt); dims=2)
        wcc = cat(wc, zeros(nx, ny, 1, nt); dims=3)

        # --- Bandpass filter (time is last dim) ---
        fu = bandpassfilter(ucc, T1, T2, delt,N,nt)
        fv = bandpassfilter(vcc, T1, T2, delt,N,nt)
        fw = bandpassfilter(wcc, T1, T2, delt,N,nt)

        
        # --- Save U, V , W filtered  ---


            open(joinpath(base2, "UVW_F", "fu_$suffix.bin"), "w") do io
                write(io, fu)
            end
            open(joinpath(base2, "UVW_F", "fv_$suffix.bin"), "w") do io
                write(io, fv)
            end
                
            open(joinpath(base2, "UVW_F", "fw_$suffix.bin"), "w") do io
                write(io, fw)
            end

            println("Completed tile: $suffix")
    end
end



