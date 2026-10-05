#=
M2_internal_tide.jl
-----------------------------------------------------------------------------
Separate the llc4320_v2 M2 SSH tide (Dan Whitt's utide fit to Eta,
27 Mar 2023 – 30 Jan 2024) into
    barotropic (large scales)  Z_bt = 2-D Gaussian low-pass of Z
    internal tide (small scales) Z_it = Z − Z_bt
and save + plot the result.


Steps
 1. Read XC/YC, find how the sideways facets are rotated, stitch facets
    1,2,4,5 into one 17280 x 12960 lon x lat map (same as plot_M2_global.jl).
 2. Re-order the columns so longitude runs -180 -> 180 (a simple circular
    shift; the map is periodic in longitude, so nothing is lost).
 3. Build the complex M2 amplitude  Z = A·exp(−i·g)  (A in m, g in degrees).
    The COMPLEX field is filtered, never amplitude and phase separately.
 4. Low-pass Z with gaussfilt2d (GaussianFilter2D.jl): σ in km, grid spacing
    dx = R·cos(lat)·Δlon per row, land excluded, periodic in longitude.
 5. Save everything to ONE NetCDF file on the plain lon/lat grid, so later you
    never have to deal with llc facets again (see "Reading the output" below).
 6. Plot: global internal-tide amplitude, global barotropic amplitude with
    co-phase lines (a check on the filter), and regional zooms.


Everything that is reported goes to the log file in OUTDIR (see LOGFILE).


Choosing the filter scale (LAMBDA_C)
  The low-pass keeps 50 % of the amplitude at wavelength LAMBDA_C.
  With LAMBDA_C = 300 km (σ ≈ 56 km):
     mode-1 M2 internal tide, λ ≈ 150 km : ~94 % ends up in Z_it
     λ ≈ 200 km                         : ~79 % ends up in Z_it
     barotropic scales, λ ≈ 1000 km     : ~6 % leaks into Z_it
  Barotropic leakage is largest on shelves/near coasts (large, short-scale
  barotropic tide) -> the plots grey out water shallower than DEPTH_MIN.
  Try 200 / 300 / 400 km and compare; the log prints these numbers for you.


Reading the output later (no facets, no rotation, just lon x lat)
  Julia :  using NCDatasets
           ds  = NCDataset("M2_internal_tide_L300km.nc")
           lon = ds["lon"][:];  lat = ds["lat"][:]
           Ait = ds["amp_it"][:, :]          # 17280 x 12960, NaN on land
           # a region only (fast):  i = findall(-165 .<= lon .<= -150); j = findall(15 .<= lat .<= 30)
           #                        Ait_reg = ds["amp_it"][i[1]:i[end], j[1]:j[end]]
  Python:  xarray.open_dataset("M2_internal_tide_L300km.nc")
  MATLAB:  ncread("M2_internal_tide_L300km.nc", "amp_it")
  Rebuild the complex field:  Z = amp .* cis.(-deg2rad.(pha))


Run (needs ~15–25 GB RAM; many threads help):
    julia -t 128 M2_internal_tide.jl          (or JULIA_NUM_THREADS in PBS)
Packages:  julia -e 'using Pkg; Pkg.add(["CairoMakie", "NCDatasets"])'
-----------------------------------------------------------------------------
=#


using Base.Threads, Statistics, Printf, Dates, Logging, CairoMakie, NCDatasets
include(joinpath(@__DIR__, "..", "..", "functions", "GaussianFilter2D.jl"))


# ============================ user settings ==================================
const OUTDIR    = "/nobackup/avaliyap/Figure/Harmonic_LLC4320/Internal_tide/"   # figures + log + NetCDF


const NX        = 4320
const COEFDIR   = "/nobackupp27/dbwhitt/llc_4320/OUT/proc/tidal_harmonic_fit/coefs"
const GRIDDIR   = "/nobackupp27/dbwhitt/llc_4320/grid_90x90x19493"
const CON       = "M2"
const AMPFILE   = joinpath(COEFDIR, "Eta_tide_amp_$(CON).data")
const PHAFILE   = joinpath(COEFDIR, "Eta_tide_pha_$(CON).data")
const DEPTHFILE = joinpath(GRIDDIR, "Depth.data")      # optional; skipped if missing


const LAMBDA_C  = 300e3     # cutoff wavelength [m] (low-pass keeps 50 % amplitude here)
const TRUNCATE  = 4.0       # kernel cut at ±4σ (as in gaussfilt.jl)
const DEPTH_MIN = 1000.0    # [m] grey out shallower water in the internal-tide plots (0 = off)


const F_IMG     = 4         # block averaging for global images (17280x12960 -> 4320x3240)
const F_CON     = 16        # block averaging for global co-phase lines
const DPHI      = 30.0      # co-phase line spacing [deg] (30° ≈ 1.03 h)
const IT_MAX    = 5.0       # colour limit internal-tide amplitude [cm]
const BT_MAX    = 100.0     # colour limit barotropic amplitude [cm]


# regional zooms at full resolution: (name, lon_min, lon_max, lat_min, lat_max)
const REGIONS = [
    ("Hawaii",        -165.0, -150.0, 15.0, 30.0),
    ("Luzon_Strait",   114.0,  128.0, 15.0, 25.0),
    ("Amazon_shelf",   -55.0,  -40.0,  0.0, 12.0),
]
# ==============================================================================


const R_EARTH = 6371e3
const T_M2    = 12.4206012          # hours
const SIGMA_NAME = @sprintf("L%dkm", round(Int, LAMBDA_C / 1e3))
const OUTNC   = joinpath(OUTDIR, "$(CON)_internal_tide_$(SIGMA_NAME).nc")


# ============================ log file ========================================
mkpath(OUTDIR)
const LOGFILE = joinpath(OUTDIR, "$(CON)_internal_tide_$(SIGMA_NAME).log")
const LOGIO   = open(LOGFILE, "w")


"Write a time-stamped line to the log file (and to the terminal)."
function logmsg(parts...)
    line = string(Dates.format(now(), "HH:MM:SS"), "  ", parts...)
    println(LOGIO, line); flush(LOGIO)
    println(stdout, line)
end
global_logger(SimpleLogger(LOGIO))
logmsg("started on host ", gethostname(), " with ", nthreads(), " threads, Julia ", VERSION)


# ----------------------------------------------------------------------------
# 1. llc reading and stitching (same as plot_M2_global.jl)
# ----------------------------------------------------------------------------
"Read compact llc binary (big-endian Float32) -> nx × 13nx matrix."
function read_llc(fn::AbstractString; nx::Int = NX)
    expected = nx * 13nx * 4
    filesize(fn) == expected ||
        error("$fn is $(filesize(fn)) bytes, expected expected(compactllc(nx) format)")
    A = Matrix{Float32}(undef, nx, 13nx)
    read!(fn, A)
    A .= ntoh.(A)
    logmsg("read ", basename(fn), "  size ", size(A), "  min/max = ", minimum(A), " / ", maximum(A))
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
# 2. plotting helpers
# ----------------------------------------------------------------------------
nanof(::Type{T}) where {T<:Real}    = T(NaN)
nanof(::Type{T}) where {T<:Complex} = T(NaN, NaN)


"Average over f×f blocks using only points where m is true (≥ half the block must be valid)."
function block_mean(F::AbstractMatrix{T}, m::AbstractMatrix{Bool}, f::Int) where {T}
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


"Co-phase lines of complex field Z every DPHI degrees (thick line = 0°)."
function cophase_lines!(ax, x, y, Z; color = :white)
    for θ in 0:DPHI:359.999
        W = Z .* cis(deg2rad(θ))
        F = Float32.(ifelse.(real.(W) .> 0, imag.(W), NaN))   # Im = 0 where Re > 0
        contour!(ax, x, y, F; levels = [0.0], color = θ == 0 ? color : (color, 0.7),
                 linewidth = θ == 0 ? 2.0 : 0.8)
    end
end


mem() = @sprintf("%.1f GB", Sys.maxrss() / 1e9)


# ----------------------------------------------------------------------------
# 3. main
# ----------------------------------------------------------------------------
function main()
    t0 = time()
    σ = sigma_from_cutoff(LAMBDA_C)
    logmsg(@sprintf("filter: cutoff λc = %.0f km -> σ = %.1f km (FWHM %.1f km), truncate = %.1fσ",
                    LAMBDA_C / 1e3, σ / 1e3, 2.3548σ / 1e3, TRUNCATE))
    for λ in (100e3, 150e3, 200e3, 300e3, 500e3, 1000e3, 2000e3)
        logmsg(@sprintf("   λ = %5.0f km : %5.1f %% in barotropic (low-pass), %5.1f %% in internal tide",
                        λ / 1e3, 100gauss_response(λ, σ), 100(1 - gauss_response(λ, σ))))
    end


    # --- grid -----------------------------------------------------------------
    logmsg("STEP 1/6  grid: orientation, lon/lat")
    xc = read_llc(joinpath(GRIDDIR, "XC.data"))
    yc = read_llc(joinpath(GRIDDIR, "YC.data"))
    _, _, x4, x5 = llc_facets(xc); _, _, y4, y5 = llc_facets(yc)
    logmsg("  facet 4:"); g4 = find_orientation(x4, y4)
    logmsg("  facet 5:"); g5 = find_orientation(x5, y5)
    lonu, lat = lonlat_vectors(to_rect(xc, g4, g5), to_rect(yc, g4, g5))
    xc = yc = x4 = x5 = y4 = y5 = nothing; GC.gc()


    # columns re-ordered so lon runs -180 -> 180 (circular shift of a periodic map)
    p   = sortperm(wrap180.(lonu))
    lon = wrap180.(lonu)[p]
    rect(A) = to_rect(A, g4, g5)[p, :]
    dlon = (lonu[end] - lonu[1]) / (length(lonu) - 1)
    n1, n2 = length(lon), length(lat)
    logmsg(@sprintf("  map %d x %d, lon %.3f .. %.3f (increasing: %s), lat %.2f .. %.2f, Δlon = %.5f° (1/48 = %.5f)",
                    n1, n2, lon[1], lon[end], all(diff(lon) .> 0), lat[1], lat[end], dlon, 1 / 48))


    # grid spacing for the filter
    dx = R_EARTH .* cosd.(lat) .* deg2rad(dlon)     # [m] per row (shrinks with latitude)
    y  = R_EARTH .* deg2rad.(lat)                   # [m] along-y distance (non-uniform rows OK)
    logmsg(@sprintf("  dx: %.2f km at equator row, %.2f km at the northern edge",
                    dx[argmin(abs.(lat))] / 1e3, dx[end] / 1e3))


    # --- data -----------------------------------------------------------------
    logmsg("STEP 2/6  reading ", CON, " amplitude / phase")
    amp = rect(read_llc(AMPFILE))
    pha = rect(read_llc(PHAFILE))
    ocean = (amp .!= 0) .& isfinite.(amp) .& isfinite.(pha)
    pmin, pmax = extrema(pha[ocean])
    logmsg(@sprintf("  ocean %.1f %%, amp max %.3f m, phase %.1f .. %.1f",
                    100count(ocean) / length(ocean), maximum(amp[ocean]), pmin, pmax))
    if pmax <= 2π + 0.01
        logmsg("  WARNING: phase looks like radians -> converting to degrees")
        pha .= rad2deg.(pha)
    end
    Z = ComplexF32.(amp .* cis.(-deg2rad.(pha)))    # η = Re{Z e^{iωt}} = A cos(ωt − g)
    Z[.!ocean] .= ComplexF32(NaN32, NaN32)
    amp = pha = nothing; GC.gc()


    depth = nothing
    if isfile(DEPTHFILE)
        depth = rect(read_llc(DEPTHFILE))
    else
        logmsg("  no Depth.data found -> no shallow-water masking, no depth in NetCDF")
    end
    logmsg("  memory so far: ", mem())


    # --- filter -----------------------------------------------------------------
    logmsg("STEP 3/6  2-D Gaussian low-pass (this is the slow step)")
    tf = time()
    Zbt = gaussfilt2d(Z, ocean, dx, y, σ; truncate = TRUNCATE, periodic_x = true)
    Zit = Z .- Zbt                                   # NaN on land (from Z and Zbt)
    logmsg(@sprintf("  filter done in %.1f min, memory %s", (time() - tf) / 60, mem()))
    deep = depth === nothing ? ocean : ocean .& (depth .>= DEPTH_MIN)
    logmsg(@sprintf("  RMS amplitude (depth ≥ %.0f m): total %.2f cm, barotropic %.2f cm, internal tide %.2f cm",
                    DEPTH_MIN, 100sqrt(mean(abs2, Z[deep])), 100sqrt(mean(abs2, Zbt[deep])),
                    100sqrt(mean(abs2, Zit[deep]))))


    # --- save -------------------------------------------------------------------
    logmsg("STEP 4/6  saving ", OUTNC)
    isfile(OUTNC) && rm(OUTNC)
    phase(z) = isnan(real(z)) ? NaN32 : Float32(mod(-rad2deg(angle(z)), 360))
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
        put("amp_tot", Float32.(abs.(Z)),   "m",   "$CON SSH amplitude, total")
        put("pha_tot", phase.(Z),           "deg", "$CON SSH Greenwich phase lag, total")
        put("amp_bt",  Float32.(abs.(Zbt)), "m",   "$CON SSH amplitude, Gaussian low-pass (barotropic)")
        put("pha_bt",  phase.(Zbt),         "deg", "$CON SSH phase, Gaussian low-pass (barotropic)")
        put("amp_it",  Float32.(abs.(Zit)), "m",   "$CON SSH amplitude, total - low-pass (internal tide)")
        put("pha_it",  phase.(Zit),         "deg", "$CON SSH phase, total - low-pass (internal tide)")
        depth === nothing || put("depth", depth, "m", "model depth")
        ds.attrib["source"]   = "D. Whitt llc4320_v2 utide 8-constituent fit to Eta, 2023-03-27 to 2024-01-30"
        ds.attrib["filter"]   = @sprintf("2-D Gaussian low-pass, land-renormalised, periodic in lon; sigma = %.1f km, cutoff (50%% amplitude) = %.0f km, truncate = %.1f sigma",
                                         σ / 1e3, LAMBDA_C / 1e3, TRUNCATE)
        ds.attrib["phase_convention"] = "eta(t) = amp*cos(omega*t - pha); complex Z = amp*exp(-i*pha)"
        ds.attrib["created"]  = string(now())
    end


    # --- global figures ---------------------------------------------------------
    logmsg("STEP 5/6  global figures")
    lon_i, lat_i = block_mean(lon, F_IMG), block_mean(lat, F_IMG)
    lon_c, lat_c = block_mean(lon, F_CON), block_mean(lat, F_CON)
    Ait_img = block_mean(Float32.(abs.(Zit)), deep, F_IMG)
    Abt_img = block_mean(Float32.(abs.(Zbt)), ocean, F_IMG)
    Zbt_con = block_mean(Zbt, ocean, F_CON)
    ylims = (max(-80, lat_i[1]), lat_i[end])
    axkw = (xlabel = "Longitude (°)", ylabel = "Latitude (°)", limits = (-180, 180, ylims...),
            xticks = -180:60:180, yticks = -60:30:60)


    fig = Figure(size = (1800, 900), fontsize = 18)
    ax = Axis(fig[1, 1]; title = @sprintf("%s internal-tide SSH amplitude (total − Gaussian low-pass, λc = %.0f km); grey: land or depth < %.0f m",
                                         CON, LAMBDA_C / 1e3, DEPTH_MIN), axkw...)
    hm = heatmap!(ax, lon_i, lat_i, 100 .* Ait_img; colormap = :magma, colorrange = (0, IT_MAX),
                  highclip = :white, nan_color = :gray80, rasterize = true)
    Colorbar(fig[1, 2], hm; label = "Amplitude (cm)")
    f = joinpath(OUTDIR, "$(CON)_IT_amp_global_$(SIGMA_NAME).png"); save(f, fig; px_per_unit = 2)
    logmsg("  saved ", f)


    fig = Figure(size = (1800, 900), fontsize = 18)
    ax = Axis(fig[1, 1]; title = @sprintf("%s barotropic (low-pass) amplitude, co-phase lines every %.0f°", CON, DPHI), axkw...)
    hm = heatmap!(ax, lon_i, lat_i, 100 .* Abt_img; colormap = :viridis, colorrange = (0, BT_MAX),
                  highclip = :yellow, nan_color = :gray80, rasterize = true)
    cophase_lines!(ax, lon_c, lat_c, Zbt_con)
    Colorbar(fig[1, 2], hm; label = "Amplitude (cm)")
    f = joinpath(OUTDIR, "$(CON)_BT_amp_cophase_global_$(SIGMA_NAME).png"); save(f, fig; px_per_unit = 2)
    logmsg("  saved ", f)


    # --- regional figures (full resolution) -------------------------------------
    logmsg("STEP 6/6  regional figures")
    for (name, x0, x1, y0, y1) in REGIONS
        I = findall(x -> x0 <= x <= x1, lon); J = findall(v -> y0 <= v <= y1, lat)
        (isempty(I) || isempty(J)) && (logmsg("  skip ", name, " (outside map)"); continue)
        I = I[1]:I[end]; J = J[1]:J[end]
        x, yy = lon[I], lat[J]
        zit = Zit[I, J]; zbt = Zbt[I, J]
        if depth !== nothing
            sh = depth[I, J] .< DEPTH_MIN
            zit[sh] .= ComplexF32(NaN32, NaN32)
        end
        fig = Figure(size = (1700, 650), fontsize = 18)
        a1 = Axis(fig[1, 1]; title = "$name: internal-tide amplitude (cm), lines: barotropic co-phase",
                  xlabel = "Longitude (°)", ylabel = "Latitude (°)", aspect = DataAspect())
        h1 = heatmap!(a1, x, yy, 100 .* abs.(zit); colormap = :magma, colorrange = (0, IT_MAX),
                      highclip = :white, nan_color = :gray80, rasterize = true)
        cophase_lines!(a1, x, yy, zbt; color = :cyan)
        Colorbar(fig[1, 2], h1)
        a2 = Axis(fig[1, 3]; title = "$name: Re{Z_it} snapshot (cm) — crests & beams",
                  xlabel = "Longitude (°)", aspect = DataAspect())
        h2 = heatmap!(a2, x, yy, 100 .* real.(zit); colormap = :balance,
                      colorrange = (-IT_MAX, IT_MAX), nan_color = :gray80, rasterize = true)
        Colorbar(fig[1, 4], h2)
        f = joinpath(OUTDIR, "$(CON)_IT_$(name)_$(SIGMA_NAME).png"); save(f, fig; px_per_unit = 2)
        logmsg("  saved ", f)
    end


    logmsg(@sprintf("FINISHED OK in %.1f min, peak memory %s", (time() - t0) / 60, mem()))
end


try
    main()
catch err
    logmsg("ERROR -- the script stopped. Full message and line numbers below:")
    logmsg(sprint(showerror, err, catch_backtrace()))
    close(LOGIO)
    exit(1)
end
close(LOGIO)




