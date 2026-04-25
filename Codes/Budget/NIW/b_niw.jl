using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter


config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base  = cfg["base_path"]
base2 = cfg["base_path2"]


mkpath(joinpath(base2, "b_NIW"))


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


dt  = 25
dto = 144
Tts = 366192
nt  = div(Tts, dto)


# --- Thickness & constants ---
thk   = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF   = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)
g     = 9.8
rho0  = 999.8


# --- NIW Filter parameters (+/-0.125fo band) ---
T1_niw, T2_niw, delt, N = 22.68,29.16, 1.0, 4


# --- Loop over tiles ---
for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]


        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
        println("Processing tile: $suffix")


        # --- Read density (Float64) ---
        rho = open(joinpath(base, "Density", "rho_in_$suffix.bin"), "r") do io
            Float64.(reshape(reinterpret(Float64, read(io, nx*ny*nz*nt*sizeof(Float64))), nx, ny, nz, nt))
        end


        # --- Read hFacC mask ---
        hFacC = read_bin(joinpath(base, "hFacC", "hFacC_$suffix.bin"), (nx, ny, nz))
        mask3D = hFacC .== 0


        # --- NIW bandpass filter density ---
        fr_niw = bandpassfilter(rho, T1_niw, T2_niw, delt, N, nt)
        rho = nothing; GC.gc()


        # --- NIW buoyancy perturbation b_NIW = -g * rho'_NIW / rho0 ---
        fr_niw[repeat(mask3D, 1, 1, 1, nt)] .= 0.0


        b_niw = (-g ./ rho0) .* fr_niw
        b_niw[repeat(mask3D, 1, 1, 1, nt)] .= 0.0
        fr_niw = nothing; GC.gc()


        # --- Save NIW buoyancy ---
        open(joinpath(base2, "b_NIW", "b_niw_$suffix.bin"), "w") do io
            write(io, Float32.(b_niw))
        end
        b_niw = nothing


        println("Completed tile: $suffix")
        GC.safepoint()
        GC.gc(true)
    end
end




