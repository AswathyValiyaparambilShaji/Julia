using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML
#using CairoMakie, SparseArrays


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base = cfg["base_path"]
mkpath(joinpath(base,"NT"))
base2 = cfg["base_path_nt"]
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


# --- Filter (10.2–32.2 hr broadband: 0.8f₀ to 2.5f₀ at mean lat 27.695°N) ---
T1, T2, delt, N = 10.2, 32.2, 1.0, 4
mkpath(joinpath(base, "NT"))
mkpath(joinpath(base, "NT","UVW_nt"))

# --- Loop over all tiles ---
for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]


        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)


        # --- Read fields ---
        U = read_bin(joinpath(base, "U/U_$suffix.bin"), (nx, ny, nz, nt))
        V = read_bin(joinpath(base, "V/V_$suffix.bin"), (nx, ny, nz, nt))
        W = read_bin(joinpath(base, "W/W_$suffix.bin"), (nx, ny, nz, nt))


        # C-grid to centers
        uc = 0.5 .* (U[1:end-1, :, :, :] .+ U[2:end,   :, :, :])
        vc = 0.5 .* (V[:, 1:end-1, :, :] .+ V[:, 2:end, :, :])
        wc = 0.5 .* (W[:, :, 1:end-1, :] .+ W[:, :, 2:end, :])


        ucc = cat(uc, zeros(1, ny, nz, nt); dims=1)
        vcc = cat(vc, zeros(nx, 1, nz, nt); dims=2)
        wcc = cat(wc, zeros(nx, ny, 1, nt); dims=3)


        # --- Bandpass filter (time is last dim) ---
        fu = bandpassfilter(ucc, T1, T2, delt, N, nt)
        fv = bandpassfilter(vcc, T1, T2, delt, N, nt)
        fw = bandpassfilter(wcc, T1, T2, delt, N, nt)


        # --- Save broadband filtered U, V, W ---
        open(joinpath(base,"NT", "UVW_NT", "fu_nt_$suffix.bin"), "w") do io
               write(io, Float32.(fu))
           end
           open(joinpath(base,"NT", "UVW_NT", "fv_nt_$suffix.bin"), "w") do io
               write(io, Float32.(fv))
           end
              
           open(joinpath(base,"NT", "UVW_NT", "fw_nt_$suffix.bin"), "w") do io
               write(io, Float32.(fw))
           end
 


        println("Completed tile: $suffix")
    end
end












