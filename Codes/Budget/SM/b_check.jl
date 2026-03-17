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
        rhoA = sum(fr .* DRFfull, dims=3) ./ depth
        rho_prime = fr .-rhoA
        println("fr loaded: size=$(size(fr)), range=[$(minimum(fr)), $(maximum(fr))]")


        # ================================================================
        # WAY 1 — buoyancy from raw bandpassed density (no depth-mean removal)
        #   b1 = -g/rho0 * fr
        #   This is b' as currently used in your budget
        #   NOT purely baroclinic — may contain small barotropic signal
        # ================================================================
        b1 = (-g ./ rho0) .* rho_prime
        b1[repeat(mask4D, 1, 1, 1, nt)] .= 0.0
        println("b1 (raw bandpassed): size=$(size(b1)), range=[$(minimum(b1)), $(maximum(b1))]")


        # ================================================================
        # WAY 2 — buoyancy from full density anomaly relative to rho0
        #   b2 = -g/rho0 * (rho - rho0)
        #   Uses raw (unfiltered) density; rho0 is reference value
        #   Includes both barotropic and baroclinic signal + all frequencies
        # ================================================================
        b2 = (-g ./ rho0) .* (rho .- rho0)
        b2[repeat(mask4D, 1, 1, 1, nt)] .= 0.0
        println("b2 (full anomaly): size=$(size(b2)), range=[$(minimum(b2)), $(maximum(b2))]")


        # --- Bandpass filter b2 to get fb (semidiurnal band, no depth-mean removal) ---
        fb = bandpassfilter(b2, T1, T2, delt, N, nt)
        println("fb (bandpassed b2): size=$(size(fb)), range=[$(minimum(fb)), $(maximum(fb))]")


        # ================================================================
        # PROFILE COMPARISON at a single point and time
        # ================================================================
        ip, jp, tp = 19, 10, 10


        # Depth axis for plotting
        z_col = -cumsum(DRFfull[ip, jp, :])        # (nz,) negative downward


        b1_prof = b1[ip, jp, :, tp]
        fb_prof = fb[ip, jp, :, tp]


        fig = Figure(resolution = (500, 700))
        ax  = Axis(fig[1, 1],
            title   = "Buoyancy profile  (i=$ip, j=$jp, t=$tp)",
            xlabel  = "b  [m/s²]",
            ylabel  = "Depth [m]")


        lines!(ax, b1_prof, z_col,
            label     = "b1 = -g/ρ₀ · fr  (raw bandpassed)",
            color     = :green,
            linewidth = 2)


        lines!(ax, fb_prof, z_col,
            label     = "fb = bandpassed(-g/ρ₀ · (ρ−ρ₀))",
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
        # DEPTH INTEGRATION DIAGNOSTIC
        # A purely baroclinic signal satisfies ∫b' dz = 0
        # We check this for fb (raw bandpassed b') at (ip, jp)
        # ================================================================


        drf_col = DRFfull[ip, jp, :]              # (nz,) cell thicknesses at this column
        H_col   = sum(drf_col)                    # total water depth at this point


        # --- Single point, single timestep ---
        fb_int_val = sum(b1_prof .* drf_col)      # scalar ∫fb dz at t=tp
        mean_abs   = mean(abs.(fb_prof))           # mean |fb| over depth
        fb_int     = fb_int_val / (H_col)   # normalized ratio — key diagnostic


        println("\n--- Depth Integration Diagnostic at (i=$ip, j=$jp, t=$tp) ---")
        println("  Total depth H          = $(round(H_col, digits=1)) m")
        println("  ∫fb dz                 = $(fb_int_val)  m²/s²")
        println("  Mean |fb|              = $(mean_abs)  m/s²")
        println("  fb_sum/H       = $(fb_int) ")
      
        # --- All timesteps at same (ip, jp) — time series of ∫fb dz ---
        fb_int_time = [sum(fb[ip, jp, :, t] .* drf_col) for t in 1:nt]   # (nt,)


        println("\n  Time series of ∫fb dz at (i=$ip, j=$jp):")
        println("  Mean   = $(mean(fb_int_time))  m²/s²")
        println("  Std    = $(std(fb_int_time))   m²/s²")
        println("  Max    = $(maximum(abs.(fb_int_time)))  m²/s²")
        println("  Mean normalized ratio over time = $(round(mean(abs.(fb_int_time)) / (mean_abs * H_col) * 100, digits=3)) %")


        # --- Plot 1: Time series of ∫fb dz ---
        fig2 = Figure(resolution = (700, 350))
        ax2  = Axis(fig2[1, 1],
            title  = "Depth-integrated b′  (i=$ip, j=$jp) — should be ≈ 0 if purely baroclinic",
            xlabel = "Timestep",
            ylabel = "∫b′ dz  [m²/s²]")


        lines!(ax2, 1:nt, fb_int_time,
            color     = :blue,
            linewidth = 1.5,
            label     = "∫fb dz")
        hlines!(ax2, [0.0],
            color     = :black,
            linestyle = :dash,
            linewidth = 1.0)


        axislegend(ax2, position = :rt)
        display(fig2)


        save(joinpath(FIGDIR, "b_depth_integral_timeseries.png"), fig2)
        println("Figure saved: b_depth_integral_timeseries.png")


        # --- Plot 2: Cumulative depth integral profile (how integral builds with depth) ---
        # This shows WHERE in the water column the non-zero contribution comes from
        fb_cumint_prof = cumsum(fb_prof .* drf_col)    # (nz,) cumulative ∫fb dz from surface


        fig3 = Figure(resolution = (500, 700))
        ax3  = Axis(fig3[1, 1],
            title  = "Cumulative ∫b′ dz profile  (i=$ip, j=$jp, t=$tp)",
            xlabel = "Cumulative ∫b′ dz  [m²/s²]",
            ylabel = "Depth [m]")


        lines!(ax3, fb_cumint_prof, z_col,
            color     = :blue,
            linewidth = 2,
            label     = "Cumulative ∫fb dz")
        vlines!(ax3, [0.0],
            color     = :black,
            linestyle = :dash,
            linewidth = 1.0)


        axislegend(ax3, position = :rt)
        display(fig3)


        save(joinpath(FIGDIR, "b_cumulative_integral_profile.png"), fig3)
        println("Figure saved: b_cumulative_integral_profile.png")


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


println("\nDone.")




