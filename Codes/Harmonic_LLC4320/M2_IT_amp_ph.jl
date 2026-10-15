#=
M2_amp_phase_filtered.jl
-----------------------------------------------------------------------------
Gaussian low-pass filter applied to the M2 SSH AMPLITUDE and PHASE SEPARATELY
(llc4320_v2 harmonic fit, D. Whitt), cutoff wavelength LAMBDA_C = 300 km.


   amplitude A(x,y)  [m]   -> ordinary 2-D Gaussian weighted average
                              A_f   = gaussfilt2d(A)
                              A_res = A - A_f


   phase g(x,y)      [deg] -> CIRCULAR Gaussian average (phase wraps 0/360,
                              so it cannot be averaged as a plain number:
                              the mean of 350° and 10° must be 0°, not 180°)
                              U     = exp(-i g)            unit vector, |U| = 1
                              U_f   = gaussfilt2d(U)
                              g_f   = -angle(U_f)          filtered phase
                              R     = |U_f|  (0..1)        phase coherence:
                                       1 = phase uniform inside the kernel,
                                       near 0 = phase changes a lot (amphidromes,
                                       strong internal-tide phase ripples)
                              g_res = wrap180(g - g_f)


IMPORTANT – what these fields are and are not
   A_f and g_f are a smoothed amplitude map and a smoothed phase map.
   They are NOT the barotropic tide, and A_res / g_res are NOT the
   internal-tide amplitude / phase. The tide signal is A·cos(ωt − g); it adds
   linearly only as the complex number Z = A·e^{−ig}. Separating barotropic and
   internal tide must be done on Z (see M2_internal_tide.jl). Use this script
   for smoothed amplitude/phase maps and as a diagnostic / comparison.


Output (full model resolution 17280 x 12960, 1/48°):
   <OUTDIR>/M2_amp_phase_filtered_L300km.nc
   <OUTDIR>/M2_amp_phase_filtered_L300km.log
   <OUTDIR>/M2_*_L300km.png
-----------------------------------------------------------------------------
=#


using Base.Threads, Statistics, Printf, Dates, Logging, CairoMakie, NCDatasets
include(joinpath(@__DIR__, "..", "..", "functions", "GaussianFilter2D.jl"))


# ============================ user settings ==================================
const OUTDIR    = "/nobackup/avaliyap/Figure/Harmonic_LLC4320/Amp_phase_filtered/"


const NX        = 4320
const COEFDIR   = "/nobackupp27/dbwhitt/llc_4320/OUT/proc/tidal_harmonic_fit/coefs"
const GRIDDIR   = "/nobackupp27/dbwhitt/llc_4320/grid_90x90x19493"
const CON       = "M2"
const AMPFILE   = joinpath(COEFDIR, "Eta_tide_amp_$(CON).data")
const PHAFILE   = joinpath(COEFDIR, "Eta_tide_pha_$(CON).data")
const DEPTHFILE = joinpath(GRIDDIR, "Depth.data")     # land = depth 0 (if missing: land = amp 0)


# filter: wavelengths LONGER than LAMBDA_C are kept in the filtered field
# (50 % amplitude response at exactly LAMBDA_C)
const LAMBDA_C  = 300e3     # [m]  = 300 km
const TRUNCATE  = 4.0       # kernel cut at ±4σ


# figures
const F_IMG      = 4        # global maps: 4x4 block average (17280x12960 -> 4320x3240)
const AMP_MAX    = 100.0    # colour limit amplitude / filtered amplitude [cm]
const ARES_MAX   = 5.0      # colour limit amplitude residual [cm]
const PRES_MAX   = 90.0     # colour limit phase residual [deg]
const LAND_COLOR = :gray55
# ==============================================================================


const R_EARTH = 6371e3
const TAG     = @sprintf("L%dkm", round(Int, LAMBDA_C / 1e3))
const OUTNC   = joinpath(OUTDIR, "$(CON)_amp_phase_filtered_$(TAG).nc")


# ============================ log file ========================================
mkpath(OUTDIR)
const LOGIO = open(joinpath(OUTDIR, "$(CON)_amp_phase_filtered_$(TAG).log"), "w")


"Write a time-stamped line to the log file (and to the terminal)."
function logmsg(parts...)
    line = string(Dates.format(now(), "HH:MM:SS"), "  ", parts...)
    println(LOGIO, line); flush(LOGIO)
    println(stdout, line)
end
global_logger(SimpleLogger(LOGIO))
logmsg("started on ", gethostname(), " with ", nthreads(), " threads")


# ----------------------------------------------------------------------------
# 1. llc reading and stitching (same as M2_internal_tide.jl)
# ----------------------------------------------------------------------------
"Read compact llc binary (big-endian Float32) -> nx × 13nx matrix."
function read_llc(fn::AbstractString; nx::Int = NX)
    expected = nx * 13nx * 4
    filesize(fn) == expected ||
        error("$fn is $(filesize(fn)) bytes, expected expected(compactllc(nx) format)")
    A = Matrix{Float32}(undef, nx, 13nx)
    read!(fn, A)
    A .= ntoh.(A)
    logmsg("read ", basename(fn), "  min/max = ", minimum(A), " / ", maximum(A))
    return A
end


"Facets 1,2 (nx × 3nx) and the sideways facets 4,5 (reshaped to 3nx × nx). Facet 3 = Arctic, unused."
function llc_facets(A; nx = size(A, 1))
    f1 = A[:, 1:3nx]
    f2 = A[:, 3nx+1:6nx]
    f4 = reshape(A[:, 7nx+1:10nx],  3nx, nx)
    f5 = reshape(A[:, 10nx+1:13nx], 3nx, nx)
    return f1, f2, f4, f5
end


valid_pt(x, y) = !(x == 0 && y == 0)          # blank land tiles have XC = YC = 0
wrap180(d) = mod(d + 180, 360) - 180


"Choose transpose/flip of a sideways facet so dim1 points east and dim2 north."
function find_orientation(xc4, yc4)
    candidates = (f -> permutedims(f),
                  f -> reverse(permutedims(f), dims = 1),
                  f -> reverse(permutedims(f), dims = 2),
                  f -> reverse(permutedims(f), dims = (1, 2)))
    for (k, g) in enumerate(candidates)
        x, y = g(xc4), g(yc4)
        east = 0; north = 0
        for j in 1:97:size(x, 2)-1, i in 1:97:size(x, 1)-1
            if valid_pt(x[i, j], y[i, j]) && valid_pt(x[i+1, j], y[i+1, j]) &&
               valid_pt(x[i, j+1], y[i, j+1])
                east  += sign(wrap180(x[i+1, j] - x[i, j]))
                north += sign(y[i, j+1] - y[i, j])
            end
        end
        if east > 0 && north > 0
            logmsg("  rotated facet orientation: option ", k)
            return g
        end
    end
    error("could not determine the orientation of a rotated facet")
end


"Facets 1,2,4,5 side by side -> 17280 × 12960 (dim1 = lon, dim2 = lat)."
function to_rect(A, g4, g5)
    f1, f2, f4, f5 = llc_facets(A)
    return vcat(f1, f2, g4(f4), g5(f5))
end


"Linear fill of NaNs in a monotonic coordinate vector (incl. ends)."
function fill_coord!(v)
    g = findall(!isnan, v)
    for k in eachindex(v)
        isnan(v[k]) || continue
        a, b = k < g[1]   ? (g[1], g[2]) :
               k > g[end] ? (g[end-1], g[end]) :
               (g[searchsortedlast(g, k)], g[searchsortedfirst(g, k)])
        v[k] = v[a] + (v[b] - v[a]) * (k - a) / (b - a)
    end
    return v
end


"1-D lon (unwrapped, continuous) and lat vectors of the stitched map."
function lonlat_vectors(XC, YC)
    n1, n2 = size(XC)
    lon = fill(NaN, n1); lat = fill(NaN, n2)
    @threads for i in 1:n1
        v = Float64[XC[i, j] for j in 1:16:n2 if valid_pt(XC[i, j], YC[i, j])]
        isempty(v) || (lon[i] = median(v))
    end
    @threads for j in 1:n2
        v = Float64[YC[i, j] for i in 1:16:n1 if valid_pt(XC[i, j], YC[i, j])]
        isempty(v) || (lat[j] = median(v))
    end
    g = findall(!isnan, lon)
    for k in 2:length(g)
        lon[g[k]] = lon[g[k-1]] + wrap180(lon[g[k]] - lon[g[k-1]])
    end
    fill_coord!(lon); fill_coord!(lat)
    return lon, lat
end


# ----------------------------------------------------------------------------
# 2. helpers
# ----------------------------------------------------------------------------
nanof(::Type{T}) where {T<:Real}    = T(NaN)
nanof(::Type{T}) where {T<:Complex} = T(NaN, NaN)
const CNaN = ComplexF32(NaN32, NaN32)


"Average over f×f blocks, ocean points only (block kept if ≥ half is ocean). f = 1 returns the field."
function block_mean(F::AbstractMatrix{T}, m::AbstractMatrix{Bool}, f::Int) where {T}
    f == 1 && return ifelse.(m, F, nanof(T))
    n1, n2 = size(F) .÷ f
    out = Matrix{T}(undef, n1, n2)
    @threads for J in 1:n2
        for I in 1:n1
            s = zero(T); w = 0
            @inbounds for j in (J-1)*f+1:J*f, i in (I-1)*f+1:I*f
                if m[i, j]; s += F[i, j]; w += 1; end
            end
            out[I, J] = 2w >= f * f ? s / w : nanof(T)
        end
    end
    return out
end
block_mean(v::AbstractVector, f::Int) = [mean(v[(k-1)*f+1:k*f]) for k in 1:length(v)÷f]


"Phase [deg] -> unit complex vector e^{-ig} (NaN on land)."
phasor(P, ocean) = ifelse.(ocean, ComplexF32.(cis.(-deg2rad.(P))), CNaN)


"Unit/complex vector -> phase in 0..360 deg (NaN stays NaN)."
phase_of(U) = Float32.(mod.(-rad2deg.(angle.(U)), 360))


"Block-average a PHASE field for plotting, circularly (via unit vectors)."
phase_block(P, ocean, f) = phase_of(block_mean(phasor(P, ocean), ocean, f))


mem() = @sprintf("%.1f GB", Sys.maxrss() / 1e9)


# ----------------------------------------------------------------------------
# 3. main
# ----------------------------------------------------------------------------
function main()
    t0 = time()
    σ = sigma_from_cutoff(LAMBDA_C)
    logmsg(@sprintf("filter: cutoff %.0f km -> Gaussian σ = %.1f km (FWHM %.1f km), cut at ±%.0fσ = ±%.0f km",
                    LAMBDA_C / 1e3, σ / 1e3, 2.3548σ / 1e3, TRUNCATE, TRUNCATE * σ / 1e3))
    for λ in (50e3, 100e3, 150e3, 200e3, 300e3, 500e3, 1000e3)
        logmsg(@sprintf("   wavelength %5.0f km : %5.1f %% kept in filtered field, %5.1f %% in residual",
                        λ / 1e3, 100gauss_response(λ, σ), 100(1 - gauss_response(λ, σ))))
    end


    # --- grid -----------------------------------------------------------------
    logmsg("STEP 1/5  grid")
    xc = read_llc(joinpath(GRIDDIR, "XC.data"))
    yc = read_llc(joinpath(GRIDDIR, "YC.data"))
    _, _, x4, x5 = llc_facets(xc); _, _, y4, y5 = llc_facets(yc)
    logmsg("  facet 4:"); g4 = find_orientation(x4, y4)
    logmsg("  facet 5:"); g5 = find_orientation(x5, y5)
    lonu, lat = lonlat_vectors(to_rect(xc, g4, g5), to_rect(yc, g4, g5))
    xc = yc = x4 = x5 = y4 = y5 = nothing; GC.gc()


    p   = sortperm(wrap180.(lonu))                 # lon -180 -> 180
    lon = wrap180.(lonu)[p]
    rect(A) = to_rect(A, g4, g5)[p, :]
    dlon = (lonu[end] - lonu[1]) / (length(lonu) - 1)
    n1, n2 = length(lon), length(lat)
    logmsg(@sprintf("  map %d x %d, lon %.3f .. %.3f, lat %.2f .. %.2f, Δlon = %.5f°",
                    n1, n2, lon[1], lon[end], lat[1], lat[end], dlon))


    dx = R_EARTH .* cosd.(lat) .* deg2rad(dlon)    # [m] grid spacing in x, per row
    y  = R_EARTH .* deg2rad.(lat)                  # [m] distance in y


    # --- data -----------------------------------------------------------------
    logmsg("STEP 2/5  reading ", CON, " amplitude / phase / depth")
    amp   = rect(read_llc(AMPFILE))
    pha   = rect(read_llc(PHAFILE))
    depth = isfile(DEPTHFILE) ? rect(read_llc(DEPTHFILE)) : nothing
    ocean = depth === nothing ? (amp .!= 0) : (depth .> 0)
    ocean .&= isfinite.(amp) .& isfinite.(pha)
    logmsg(depth === nothing ? "  Depth.data not found -> land = (amplitude == 0)" :
                               "  land = (depth == 0)")
    pmin, pmax = extrema(pha[ocean])
    if pmax <= 2π + 0.01
        logmsg("  WARNING: phase looks like radians -> converting to degrees")
        pha .= rad2deg.(pha)
    end
    pha .= mod.(pha, 360)
    amp[.!ocean] .= NaN32; pha[.!ocean] .= NaN32
    logmsg(@sprintf("  ocean %.1f %% of map, amp max %.3f m, phase %.1f .. %.1f deg",
                    100count(ocean) / length(ocean), maximum(amp[ocean]), extrema(pha[ocean])...))


    # --- filter amplitude (scalar) ----------------------------------------------
    logmsg("STEP 3/5  Gaussian filter: amplitude, then phase (circular)")
    tf = time()
    # passed as a complex array with zero imaginary part, so the exact same
    # gaussfilt2d call as in M2_internal_tide.jl is used; real part = filtered A
    Ac = ifelse.(ocean, ComplexF32.(amp), CNaN)
    Af = Float32.(ifelse.(ocean, real.(gaussfilt2d(Ac, ocean, dx, y, σ;
                                         truncate = TRUNCATE, periodic_x = true)), NaN32))
    Ac = nothing; GC.gc()
    Ares = amp .- Af
    logmsg(@sprintf("  amplitude done in %.1f min, memory %s", (time() - tf) / 60, mem()))


    # --- filter phase (circular, via unit vectors e^{-ig}) ----------------------
    tf = time()
    Uf = gaussfilt2d(phasor(pha, ocean), ocean, dx, y, σ; truncate = TRUNCATE, periodic_x = true)
    Pf = ifelse.(ocean, phase_of(Uf), NaN32)                 # filtered phase [deg]
    R  = Float32.(ifelse.(ocean, abs.(Uf), NaN32))           # phase coherence 0..1
    Uf = nothing; GC.gc()
    Pres = Float32.(wrap180.(pha .- Pf))                     # phase residual, -180..180
    logmsg(@sprintf("  phase done in %.1f min, memory %s", (time() - tf) / 60, mem()))


    logmsg(@sprintf("  RMS over ocean: A %.2f cm, A_filt %.2f cm, A_res %.2f cm",
                    100sqrt(mean(abs2, amp[ocean])), 100sqrt(mean(abs2, Af[ocean])),
                    100sqrt(mean(abs2, Ares[ocean]))))
    logmsg(@sprintf("  RMS phase residual %.1f deg, median phase coherence %.3f",
                    sqrt(mean(abs2, Pres[ocean])), median(R[ocean])))


    # --- save (full resolution) ---------------------------------------------------
    logmsg("STEP 4/5  saving full-resolution NetCDF ", OUTNC)
    isfile(OUTNC) && rm(OUTNC)
    NCDataset(OUTNC, "c") do ds
        defDim(ds, "lon", n1); defDim(ds, "lat", n2)
        vlon = defVar(ds, "lon", Float64, ("lon",), attrib = ["units" => "degrees_east"])
        vlat = defVar(ds, "lat", Float64, ("lat",), attrib = ["units" => "degrees_north"])
        vlon[:] = lon; vlat[:] = lat
        put(name, F, units, long) = begin
            v = defVar(ds, name, Float32, ("lon", "lat"); deflatelevel = 1, shuffle = true,
                       chunksizes = [n1 ÷ 16, n2 ÷ 16], fillvalue = NaN32,
                       attrib = ["units" => units, "long_name" => long])
            v[:, :] = F
            logmsg("  wrote ", name)
        end
        put("amp",        amp,  "m",   "$CON SSH amplitude (original)")
        put("pha",        pha,  "deg", "$CON SSH Greenwich phase lag (original)")
        put("amp_filt",   Af,   "m",   "$CON SSH amplitude, Gaussian low-pass of amplitude")
        put("pha_filt",   Pf,   "deg", "$CON SSH phase, circular Gaussian low-pass of phase")
        put("amp_res",    Ares, "m",   "amp - amp_filt (NOT internal-tide amplitude)")
        put("pha_res",    Pres, "deg", "wrap180(pha - pha_filt) (NOT internal-tide phase)")
        put("pha_coherence", R, "1",   "|Gaussian mean of exp(-i*pha)|, 1 = uniform phase within kernel")
        depth === nothing || put("depth", depth, "m", "model depth (0 = land)")
        ds.attrib["source"] = "D. Whitt llc4320_v2 utide fit to Eta, 2023-03-27 to 2024-01-30"
        ds.attrib["filter"] = @sprintf("2-D Gaussian low-pass applied separately to amplitude and to phase (phase averaged circularly via exp(-i*pha)); land excluded, periodic in lon; cutoff %.0f km (50%% response), sigma %.1f km, truncate %.0f sigma",
                                       LAMBDA_C / 1e3, σ / 1e3, TRUNCATE)
        ds.attrib["note"] = "amp_filt/pha_filt are smoothed maps, not a barotropic/internal-tide decomposition; that requires filtering the complex field amp*exp(-i*pha)"
        ds.attrib["created"] = string(now())
    end


    # --- global figures ---------------------------------------------------------
    logmsg("STEP 5/5  global figures ((FIMG)x(F_IMG) block average)")
    lon_i, lat_i = F_IMG == 1 ? (lon, lat) : (block_mean(lon, F_IMG), block_mean(lat, F_IMG))
    ylims = (max(-80, lat_i[1]), lat_i[end])
    axkw  = (xlabel = "Longitude (°)", ylabel = "Latitude (°)", limits = (-180, 180, ylims...),
             xticks = -180:60:180, yticks = -60:30:60)


    function save_map(name, title, F, cmap, crange, label)
        fig = Figure(size = (1800, 900), fontsize = 18)
        ax  = Axis(fig[1, 1]; title = title, axkw...)
        hm  = heatmap!(ax, lon_i, lat_i, F; colormap = cmap, colorrange = crange,
                       nan_color = LAND_COLOR, rasterize = true)
        Colorbar(fig[1, 2], hm; label = label)
        f = joinpath(OUTDIR, "$(CON)_$(name)_$(TAG).png"); save(f, fig; px_per_unit = 2)
        logmsg("  saved ", f)
    end


    L = round(Int, LAMBDA_C / 1e3)
    save_map("amp", "$CON SSH amplitude (original)",
             100 .* block_mean(amp, ocean, F_IMG), :magma, (0, AMP_MAX), "Amplitude (cm)")
    save_map("amp_filt", "CONSSHamplitude,Gaussianfiltered(L km)",
             100 .* block_mean(Af, ocean, F_IMG), :magma, (0, AMP_MAX), "Amplitude (cm)")
    save_map("amp_res", "CONamplituderesidual,A-Afilt(L km)",
             100 .* block_mean(Ares, ocean, F_IMG), :balance, (-ARES_MAX, ARES_MAX), "Amplitude (cm)")
    save_map("pha", "$CON SSH Greenwich phase (original)",
             phase_block(pha, ocean, F_IMG), :twilight, (0, 360), "Phase (°)")
    save_map("pha_filt", "CONSSHphase,circularGaussianfiltered(L km)",
             phase_block(Pf, ocean, F_IMG), :twilight, (0, 360), "Phase (°)")
    save_map("pha_res", "CONphaseresidual,wrap(g-gfilt)(L km)",
             Float32.(wrap180.(phase_block(Pres, ocean, F_IMG))), :balance,
             (-PRES_MAX, PRES_MAX), "Phase (°)")
    save_map("pha_coherence", "$CON phase coherence |⟨e^{−ig}⟩| within $L km kernel",
             block_mean(R, ocean, F_IMG), :viridis, (0, 1), "Coherence")


    logmsg(@sprintf("FINISHED OK in %.1f min, peak memory %s", (time() - t0) / 60, mem()))
end


main()
close(LOGIO)




