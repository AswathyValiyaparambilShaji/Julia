using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML
#using CairoMakie, SparseArrays


include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter


config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base = cfg["bp_box27b"]
baseout = cfg["bp_box27b_nt"]


# --- Domain & grid of 27b ---
NX, NY = 1056, 1026
minlat, maxlat = -60.0, -48.0
minlon, maxlon = 142.0208530805687, 163.9791469194313
lat = range(minlat, maxlat, length=NY)
lon = range(minlon, maxlon, length=NX)


# --- Tile & time ---
buf = 3
tx, ty = 150, 146
nx = tx + 2 * buf
ny = ty + 2 * buf
NZ = 173
nz = 170
kz = 1
nt = 558


# --- Filter (10.2-32.2 hr broadband: 0.8f0 to 2.5f0 at mean lat 27.695 deg N) ---
T1, T2, delt, N = 10.2, 32.2, 1.0, 4


mkpath(joinpath(base, "NT"))
mkpath(joinpath(base, "NT", "UVW_NT"))


# ------------------------------------------------------------------
# Reads one component's raw Float32 tile, averages it from the C-grid
# face to the cell center along `pad_dim`, pads it back to the full
# (nx,ny,nz,nt) size with zeros, then bandpass-filters it.
#
# Kept entirely in Float32 and frees each large intermediate as soon
# as it's no longer needed, and only ONE component is ever resident
# at a time (see main loop) instead of U, V and W all at once.
# ------------------------------------------------------------------
function process_component(base::String, letter::String, suffix::String,
                            nx::Int, ny::Int, nz::Int, nt::Int,
                            pad_dim::Int,
                            T1, T2, delt, N)
    path_in = joinpath(base, letter, "$(letter)_v2_$suffix.bin")


    raw = open(path_in, "r") do io
        reshape(reinterpret(Float32, read(io, nx * ny * nz * nt * sizeof(Float32))), nx, ny, nz, nt)
    end


    centered = zeros(Float32, nx, ny, nz, nt)
    if pad_dim == 1
        @views centered[1:end-1, :, :, :] .= 0.5f0 .* (raw[1:end-1, :, :, :] .+ raw[2:end, :, :, :])
    elseif pad_dim == 2
        @views centered[:, 1:end-1, :, :] .= 0.5f0 .* (raw[:, 1:end-1, :, :] .+ raw[:, 2:end, :, :])
    elseif pad_dim == 3
        @views centered[:, :, 1:end-1, :] .= 0.5f0 .* (raw[:, :, 1:end-1, :] .+ raw[:, :, 2:end, :])
    else
        error("pad_dim must be 1, 2 or 3")
    end


    raw = nothing
    GC.gc()


    filtered = bandpassfilter(centered, T1, T2, delt, N, nt)


    centered = nothing
    GC.gc()


    return filtered
end


function write_component(base::String, suffix::String, tag::String, data)
    open(joinpath(base, "NT", "UVW_NT", "$(tag)_nt_$suffix.bin"), "w") do io
        write(io, Float32.(data))
    end
end


# --- Loop over all tiles ---
for xn in cfg["xn_start"]:cfg["xn_e27b"]
    for yn in cfg["yn_start"]:cfg["yn_e27b"]
        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)


        # U: average along dim 1 (x-faces -> centers)
        fu = process_component(base, "U", suffix, nx, ny, nz, nt, 1, T1, T2, delt, N)
        write_component(base, suffix, "fu", fu)
        fu = nothing
        GC.gc()


        # V: average along dim 2 (y-faces -> centers)
        fv = process_component(base, "V", suffix, nx, ny, nz, nt, 2, T1, T2, delt, N)
        write_component(base, suffix, "fv", fv)
        fv = nothing
        GC.gc()


        # W: average along dim 3 (z-faces -> centers)
        fw = process_component(base, "W", suffix, nx, ny, nz, nt, 3, T1, T2, delt, N)
        write_component(base, suffix, "fw", fw)
        fw = nothing
        GC.gc()


        println("Completed tile: $suffix")
    end
end




