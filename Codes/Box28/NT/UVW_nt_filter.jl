using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML
#using CairoMakie, SparseArrays


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base = cfg["bp_box28"]
baseout = cfg["bp_box28_nt"]


# --- Domain & grid of 27b ---
NX, NY = 384, 336
minlat, maxlat = -24.5, -18.5
minlon, maxlon = 337.5, 345.4791122715405
lat = range(minlat, maxlat, length=NY)
lon = range(minlon, maxlon, length=NX)
NZ = 173

# --- Tile & time --- 
buf = 3
tx, ty = 54, 66
nx = tx + 2*buf
ny = ty + 2*buf
nz = 168



kz = 1
nt = 558
# --- Filter (10.2–32.2 hr broadband: 0.8f₀ to 2.5f₀ at mean lat 27.695°N) ---
T1, T2, delt, N = 10.2, 32.2, 1.0, 4
mkpath(joinpath(base, "NT"))
mkpath(joinpath(base, "NT","UVW_NT"))


# --- Loop over all tiles ---

Threads.@threads for xn in cfg["xn_start"]:cfg["xn_e28"]
    for yn in cfg["yn_start"]:cfg["yn_e28"]


        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)


        # --- Read fields ---
        U = Float64.(open(joinpath(base, "U","U_v2_$suffix.bin"), "r") do io
            reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny,nz,nt)
        end)#
        V = Float64.(open(joinpath(base, "V", "V_v2_$suffix.bin"), "r") do io
            reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny,nz,nt)
        end)
        W = Float64.(open(joinpath(base, "W","W_v2_$suffix.bin"), "r") do io
            reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny,nz,nt)
        end)


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












