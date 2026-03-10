using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML, CairoMakie


include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter


config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg   = TOML.parsefile(config_file)
base  = cfg["base_path"]
base2 = cfg["base_path2"]


mkpath(joinpath(base2, "b"))


# --- Tile & time ---
buf    = 3
tx, ty = 47, 66
nx     = tx + 2 * buf
ny     = ty + 2 * buf
nz     = 88


dto = 144
Tts = 366192
nt  = div(Tts, dto)


# --- Thickness & constants ---
thk   = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF   = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)
g     = 9.8
rho0  = 999.8


# --- Filter parameters ---
T1, T2, delt, N = 9.0, 15.0, 1.0, 4


xn = 1
yn = 1


# for xn in cfg["xn_start"]:cfg["xn_end"]
#     for yn in cfg["yn_start"]:cfg["yn_end"]


        suffix   = @sprintf("%02dx%02d_%d", xn, yn, buf)
        out_path = joinpath(base2, "b", "b_$suffix.bin")


        println("Processing tile: $suffix")


        rho_path  = joinpath(base, "Density", "rho_in_$suffix.bin")
        hfac_path = joinpath(base, "hFacC",   "hFacC_$suffix.bin")


        missing_files = filter(!isfile, [rho_path, hfac_path])
        if !isempty(missing_files)
            @warn "Missing files:" missing_files
        end


        # --- Read density ---
        rho = open(rho_path, "r") do io
            nbytes = nx * ny * nz * nt * sizeof(Float64)
            Float64.(reshape(reinterpret(Float64, read(io, nbytes)), nx, ny, nz, nt))
        end
        println("rho loaded: size=$(size(rho)), range=[$(minimum(rho)), $(maximum(rho))]")


        # --- Grid mask & thickness ---
        hFacC   = read_bin(hfac_path, (nx, ny, nz))
        DRFfull = hFacC .* DRF3d
        depth   = sum(DRFfull, dims=3)
        DRFfull[hFacC .== 0] .= 0.0


        mask4D    = reshape(hFacC .== 0, nx, ny, nz, 1)
        DRFfull4D = repeat(DRFfull, 1, 1, 1, nt)
        depth4D   = repeat(depth,   1, 1, 1, nt)


        # --- Bandpass filter density ---
        fr = bandpassfilter(rho, T1, T2, delt, N, nt)
        println("fr loaded: size=$(size(fr)), range=[$(minimum(fr)), $(maximum(fr))]")


        # ================================================================
        # WAY 1 — baroclinic perturbation buoyancy
        #   rho' = fr - <fr>_z   (wave-band density minus its depth mean)
        #   b1   = -g/rho0 * rho'
        #   Enforces integral(b1 dz) = 0 — purely baroclinic
        # ================================================================
        #rhoA_3d   = sum(fr .* DRFfull4D, dims=3) ./ depth4D   # (nx,ny,1,nt)
        rho_prime = fr.- rhoA_3d                           # (nx,ny,nz,nt)
        rho_prime[repeat(mask4D, 1, 1, 1, nt)] .= 0.0
        b1 = (-g ./ rho0) .* fr
        b1[repeat(mask4D, 1, 1, 1, nt)] .= 0.0
        println("b1 (baroclinic): size=$(size(b1)), range=[$(minimum(b1)), $(maximum(b1))]")


        # ================================================================
        # WAY 2 — buoyancy from full density anomaly relative to rho0
        #   b2 = -g/rho0 * (rho - rho0)
        #   Uses raw (unfiltered) density; rho0 is reference value
        #   Includes both barotropic and baroclinic signal + all frequencies
        # ================================================================
        b2 = (-g ./ rho0) .* (rho .- rho0)

        b2[repeat(mask4D, 1, 1, 1, nt)] .= 0.0
        println("b2 (full anomaly): size=$(size(b2)), range=[$(minimum(b2)), $(maximum(b2))]")
        fb = bandpassfilter(b2, T1, T2, delt, N, nt)


        # ================================================================
        # PROFILE COMPARISON at a single point and time
        # Pick a wet point — adjust (ip, jp, tp) as needed
        # ================================================================
        ip, jp, tp = 19, 10, 10


        # Depth axis for plotting
        z_col = -cumsum(DRFfull[ip, jp, :])        # (nz,) negative downward


        b1_prof = b1[ip, jp, :, tp]
        b2_prof = fb[ip, jp, :, tp]


        fig = Figure(resolution = (500, 700))
        ax  = Axis(fig[1, 1],
            title   = "Buoyancy profile  (i=$ip, j=$jp, t=$tp)",
            xlabel  = "b  [m/s²]",
            ylabel  = "Depth [m]")


        lines!(ax, b1_prof, z_col,
            label     = "b1 = -g/ρ₀ · (fr  )   ",
            color     = :green,
            linewidth = 2)


        lines!(ax, b2_prof, z_col,
            label     = "b2 = -g/ρ₀ · (ρ − ρ₀)        [filtered b]",
            color     = :red,
            linewidth = 2,
            linestyle = :dash)


        axislegend(ax, position = :rt, labelsize = 10)
        display(fig)


        FIGDIR = cfg["fig_base"]
        mkpath(FIGDIR)
        save(joinpath(FIGDIR, "b_profile_comparison.png"), fig)
        println("Figure saved: b_profile_comparison.png")


        # ================================================================
        # Once you are happy with b1, uncomment to save:
        #= 
        open(out_path, "w") do io
            write(io, Float32.(b1))
        end
        println("Saved: b_$suffix.bin")
        =#
        # ================================================================


# end  # yn
# end  # xn


println("Done.")




