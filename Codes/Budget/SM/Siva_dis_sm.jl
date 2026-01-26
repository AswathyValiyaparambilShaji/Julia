    
    using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML


    include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
    using .FluxUtils: read_bin, bandpassfilter


    config_file = get(ENV, "JULIA_CONFIG",
                joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
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


    dt  = 25
    dto = 144
    Tts = 366192
    nt  = div(Tts, dto)


    ts      = 72      # 3-day window
    nt_avg = div(nt, ts)


    # --- Thickness & constants ---
    thk = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
    DRF = thk[1:nz]
    DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)


    rho0 = 999.8


println("Processing $nt time steps...")

raw_data = read(joinpath(base, "Dis", "Fs_TotDiss_band1.bin")) 
total_elements = length(raw_data) ÷ sizeof(Float64) 
nxny = NX*NY
println("Total Float64 elements in file: $total_elements")
println("NX * NY: $nxny")





Fm = Float64.(open(joinpath(base, "Dis", "Fmask_TotDiss_band1.bin"), "r") do io
            raw = read(io, NX * NY *  sizeof(Float32))
            reshape(reinterpret(Float64, raw), NX, NY)
        end)
Fs = Float64.(open(joinpath(base, "Dis", "Fs_TotDiss_band1.bin"), "r") do io
            raw = read(io, NX * NY *  sizeof(Float64))
            reshape(reinterpret(Float64, raw), NX, NY)
        end)