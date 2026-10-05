#=
M2_internal_tide.jl
-----------------------------------------------------------------------------
M2 internal tide from the llc4320_v2 SSH harmonic fit (Dan Whitt, utide fit
to Eta, 27 Mar 2023 – 30 Jan 2024), by Gaussian filtering + subtraction:


    Z     = A·exp(−i·g)            complex M2 SSH amplitude (total)
    Z_bt  = gaussfilt2d(Z)         large scales  (> LAMBDA_C)  ≈ barotropic
    Z_it  = Z − Z_bt               small scales  (< LAMBDA_C)  ≈ internal tide


Same idea as box-averaging and subtracting; the Gaussian just gives nearer
points more weight (see GaussianFilter2D.jl).


Do we need to reconstruct SSH time series?  No.
    η(x,t) = A cos(ωt − g) = Re{Z·e^{iωt}}.  The filter is a weighted average,
    so filtering Z once gives exactly the same result as filtering the
    reconstructed SSH at every time step. To get the internal-tide SSH at any
    time from the saved file:
        η_it(t) = amp_it .* cos.(ω*t .- deg2rad.(pha_it)),  ω = 2π/(12.4206 h)
    The "Re{Z_it}" figure is that snapshot at t = 0 (Greenwich phase 0).


Output (all at FULL model resolution, 17280 x 12960, 1/48°):
    <OUTDIR>/M2_internal_tide_L150km.nc   amp/pha of total, barotropic, internal tide + depth
    <OUTDIR>/M2_internal_tide_L150km.log  progress and numbers to check
    <OUTDIR>/*.png                         figures


Reading the NetCDF later (plain lon x lat grid, no llc facets):
    using NCDatasets
    ds  = NCDataset("M2_internal_tide_L150km.nc")
    lon = ds["lon"][:]; lat = ds["lat"][:]          # lon -180..180, lat increasing
    Ait = ds["amp_it"][:, :]                        # [m], NaN on land
-----------------------------------------------------------------------------
=#


using Base.Threads, Statistics, Printf, Dates, Logging, CairoMakie, NCDatasets
include(joinpath(@__DIR__, "..", "..", "functions", "GaussianFilter2D.jl"))


# ============================ user settings ==================================
const OUTDIR    = "/nobackup/avaliyap/Figure/Harmonic_LLC4320/Internal_tide/"


const NX        = 4320
const COEFDIR   = "/nobackupp27/dbwhitt/llc_4320/OUT/proc/tidal_harmonic_fit/coefs"
const GRIDDIR   = "/nobackupp27/dbwhitt/llc_4320/grid_90x90x19493"
const CON       = "M2"
const AMPFILE   = joinpath(COEFDIR, "Eta_tide_amp_$(CON).data")
const PHAFILE   = joinpath(COEFDIR, "Eta_tide_pha_$(CON).data")
const DEPTHFILE = joinpath(GRIDDIR, "Depth.data")     # land = depth 0 (if missing: land = amp 0)


# filter: wavelengths LONGER than LAMBDA_C are removed (go to Z_bt).
# The split is gradual: at exactly LAMBDA_C, half the amplitude goes each way.
const LAMBDA_C  = 150e3     # [m]
const TRUNCATE  = 4.0       # kernel cut at ±4σ (as in gaussfilt.jl)


# figures
const F_IMG     = 4         # global maps: average 4x4 points (17280x12960 -> 4320x3240);
                            #   1 = every grid point (very large PNG, slow)
const F_CON     = 16        # global co-phase lines: 16x16 block average (smooth lines)
const DPHI      = 30.0      # co-phase line spacing [deg] (30° ≈ 1.03 h)
const IT_MAX    = 5.0       # colour limit internal-tide amplitude [cm]
const BT_MAX    = 100.0     # colour limit barotropic amplitude [cm]


# white (0) -> dark (max) colour map; reuse it in other scripts (e.g. mooring maps)
const AMP_COLORS = [RGBf(1.000, 1.000, 1.000), RGBf(0.996, 0.910, 0.784), RGBf(0.992, 0.733, 0.518),
                    RGBf(0.890, 0.290, 0.200), RGBf(0.600, 0.000, 0.051), RGBf(0.169, 0.000, 0.020)]
const CMAP_AMP   = cgrad(AMP_COLORS)          # white -> peach -> red -> almost black
const LAND_COLOR = :gray55


# regional zooms at full resolution: (name, lon_min, lon_max, lat_min, lat_max)
const REGIONS = [
    ("Hawaii",        -165.0, -150.0, 15.0, 30.0),
    ("Luzon_Strait",   114.0,  128.0, 15.0, 25.0),
    ("Amazon_shelf",   -55.0,  -40.0,  0.0, 12.0),
]
# moorings to mark on the maps: (name, lon, lat); leave empty for none
const MOORINGS = Tuple{String, Float64, Float64}[
    # ("M1", -158.0, 22.0),
]
# ==============================================================================


const R_EARTH = 6371e3
const TAG     = @sprintf("L%dkm", round(Int, LAMBDA_C / 1e3))
const OUTNC   = joinpath(OUTDIR, "$(CON)_internal_tide_$(TAG).nc")


# ============================ log file ========================================
mkpath(OUTDIR)
const LOGIO = open(joinpath(OUTDIR, "$(CON)_internal_tide_$(TAG).log"), "w")


"Write a time-stamped line to the log file (and to the terminal)."
function logmsg(parts...)
    line = string(Dates.format(now(), "HH:MM:SS"), "  ", parts...)
    println(LOGIO, line); flush(LOGIO)
    println(stdout, line)
end
global_logger(SimpleLogger(LOGIO))
logmsg("started on ", gethostname(), " with ", nthreads(), " threads")


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
# 2. plotting helpers
# ----------------------------------------------------------------------------
nanof(::Type{T}) where {T<:Real}    = T(NaN)
nanof(::Type{T}) where {T<:Complex} = T(NaN, NaN)


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


"Amplitude map, white (0) -> dark (cmax), land grey. Reusable for mooring maps."
amp_map!(ax, x, y, A_cm, cmax) =
    heatmap!(ax, x, y, A_cm; colormap = CMAP_AMP, colorrange = (0, cmax),
             highclip = AMP_COLORS[end], nan_color = LAND_COLOR, rasterize = true)


"Co-phase lines of complex field Z every DPHI degrees (thick line = 0°)."
function cophase_lines!(ax, x, y, Z; color = :dodgerblue)
    for θ in 0:DPHI:359.999
        W = Z .* cis(deg2rad(θ))
        F = Float32.(ifelse.(real.(W) .> 0, imag.(W), NaN))   # Im = 0 where Re > 0
        contour!(ax, x, y, F; levels = [0.0], color = θ == 0 ? color : (color, 0.7),
                 linewidth = θ == 0 ? 2.0 : 0.8)
    end
end


"Mark moorings (if any) with triangles and names."
function moorings!(ax)
    isempty(MOORINGS) && return
    scatter!(ax, [m[2] for m in MOORINGS], [m[3] for m in MOORINGS]; marker = :utriangle,
             markersize = 16, color = :cyan, strokecolor = :black, strokewidth = 1.2)
    text!(ax, [m[2] for m in MOORINGS], [m[3] for m in MOORINGS]; text = [m[1] for m in MOORINGS],
          offset = (8, 4), fontsize = 14, color = :black)
end


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
        logmsg(@sprintf("   wavelength %5.0f km : %5.1f %% kept in internal tide, %5.1f %% removed (barotropic)",
                        λ / 1e3, 100(1 - gauss_response(λ, σ)), 100gauss_response(λ, σ)))
    end


    # --- grid -----------------------------------------------------------------
    logmsg("STEP 1/6  grid")
    xc = read_llc(joinpath(GRIDDIR, "XC.data"))
    yc = read_llc(joinpath(GRIDDIR, "YC.data"))
    _, _, x4, x5 = llc_facets(xc); _, _, y4, y5 = llc_facets(yc)
    logmsg("  facet 4:"); g4 = find_orientation(x4, y4)
    logmsg("  facet 5:"); g5 = find_orientation(x5, y5)
    lonu, lat = lonlat_vectors(to_rect(xc, g4, g5), to_rect(yc, g4, g5))
    xc = yc = x4 = x5 = y4 = y5 = nothing; GC.gc()


    # columns re-ordered so lon runs -180 -> 180 (circular shift; map is periodic)
    p   = sortperm(wrap180.(lonu))
    lon = wrap180.(lonu)[p]
    rect(A) = to_rect(A, g4, g5)[p, :]
    dlon = (lonu[end] - lonu[1]) / (length(lonu) - 1)
    n1, n2 = length(lon), length(lat)
    logmsg(@sprintf("  map %d x %d, lon %.3f .. %.3f (increasing: %s), lat %.2f .. %.2f, Δlon = %.5f° (1/48 = %.5f)",
                    n1, n2, lon[1], lon[end], all(diff(lon) .> 0), lat[1], lat[end], dlon, 1 / 48))


    dx = R_EARTH .* cosd.(lat) .* deg2rad(dlon)     # [m] grid spacing in x, per row
    y  = R_EARTH .* deg2rad.(lat)                   # [m] distance in y
    logmsg(@sprintf("  dx = %.2f km at the equator, %.2f km at the northern edge",
                    dx[argmin(abs.(lat))] / 1e3, dx[end] / 1e3))


    # --- data -----------------------------------------------------------------
    logmsg("STEP 2/6  reading ", CON, " amplitude / phase / depth")
    amp = rect(read_llc(AMPFILE))
    pha = rect(read_llc(PHAFILE))
    depth = isfile(DEPTHFILE) ? rect(read_llc(DEPTHFILE)) : nothing
    ocean = depth === nothing ? (amp .!= 0) : (depth .> 0)          # land = depth 0
    ocean .&= isfinite.(amp) .& isfinite.(pha)
    logmsg(depth === nothing ? "  Depth.data not found -> land = (amplitude == 0)" :
                               "  land = (depth == 0)")
    pmin, pmax = extrema(pha[ocean])
    logmsg(@sprintf("  ocean %.1f %% of map, amp max %.3f m, phase %.1f .. %.1f",
                    100count(ocean) / length(ocean), maximum(amp[ocean]), pmin, pmax))
    if pmax <= 2π + 0.01
        logmsg("  WARNING: phase looks like radians -> converting to degrees")
        pha .= rad2deg.(pha)
    end
    Z = ComplexF32.(amp .* cis.(-deg2rad.(pha)))    # η = Re{Z e^{iωt}} = A cos(ωt − g)
    Z[.!ocean] .= ComplexF32(NaN32, NaN32)
    amp = pha = nothing; GC.gc()
    logmsg("  memory so far: ", mem())


    # --- filter -----------------------------------------------------------------
    logmsg("STEP 3/6  2-D Gaussian low-pass + subtraction")
    tf = time()
    Zbt = gaussfilt2d(Z, ocean, dx, y, σ; truncate = TRUNCATE, periodic_x = true)
    Zit = Z .- Zbt
    logmsg(@sprintf("  done in %.1f min, memory %s", (time() - tf) / 60, mem()))
    logmsg(@sprintf("  RMS amplitude over ocean: total %.2f cm, barotropic %.2f cm, internal tide %.2f cm",
                    100sqrt(mean(abs2, Z[ocean])), 100sqrt(mean(abs2, Zbt[ocean])),
                    100sqrt(mean(abs2, Zit[ocean]))))


    # --- save (full resolution) ---------------------------------------------------
    logmsg("STEP 4/6  saving full-resolution NetCDF ", OUTNC)
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
        put("amp_tot", abs.(Z),   "m",   "$CON SSH amplitude, total")
        put("pha_tot", phase.(Z), "deg", "$CON SSH Greenwich phase lag, total")
        put("amp_bt",  abs.(Zbt), "m",   "$CON SSH amplitude, Gaussian low-pass (barotropic)")
        put("pha_bt",  phase.(Zbt), "deg", "$CON SSH phase, Gaussian low-pass (barotropic)")
        put("amp_it",  abs.(Zit), "m",   "$CON SSH amplitude, total - low-pass (internal tide)")
        put("pha_it",  phase.(Zit), "deg", "$CON SSH phase, total - low-pass (internal tide)")
        depth === nothing || put("depth", depth, "m", "model depth (0 = land)")
        ds.attrib["source"] = "D. Whitt llc4320_v2 utide 8-constituent fit to Eta, 2023-03-27 to 2024-01-30"
        ds.attrib["filter"] = @sprintf("2-D Gaussian low-pass (land excluded, periodic in lon); cutoff %.0f km (50%% amplitude), sigma %.1f km, truncate %.0f sigma; internal tide = total - low-pass",
                                       LAMBDA_C / 1e3, σ / 1e3, TRUNCATE)
        ds.attrib["phase_convention"] = "eta(t) = amp*cos(omega*t - pha*pi/180)"
        ds.attrib["created"] = string(now())
    end


    # --- global figures ---------------------------------------------------------
    logmsg("STEP 5/6  global figures (", F_IMG == 1 ? "every grid point" : "(FIMG)x(F_IMG) block average", ")")
    lon_i, lat_i = F_IMG == 1 ? (lon, lat) : (block_mean(lon, F_IMG), block_mean(lat, F_IMG))
    lon_c, lat_c = block_mean(lon, F_CON), block_mean(lat, F_CON)
    Ait_img = block_mean(abs.(Zit), ocean, F_IMG)
    Abt_img = block_mean(abs.(Zbt), ocean, F_IMG)
    Zbt_con = block_mean(Zbt, ocean, F_CON)
    ylims = (max(-80, lat_i[1]), lat_i[end])
    figsize = F_IMG == 1 ? (n1 ÷ 4, n2 ÷ 8) : (1800, 900)      # bigger canvas if full resolution
    axkw = (xlabel = "Longitude (°)", ylabel = "Latitude (°)", limits = (-180, 180, ylims...),
            xticks = -180:60:180, yticks = -60:30:60)


    fig = Figure(size = figsize, fontsize = 18)
    ax = Axis(fig[1, 1]; title = @sprintf("%s internal-tide SSH amplitude ", CON), axkw...)
    hm = amp_map!(ax, lon_i, lat_i, 100 .* Ait_img, IT_MAX)
    moorings!(ax)
    Colorbar(fig[1, 2], hm; label = "Amplitude (cm)")
    f = joinpath(OUTDIR, "$(CON)_IT_amp_global_$(TAG).png"); save(f, fig; px_per_unit = 2)
    logmsg("  saved ", f)


    fig = Figure(size = figsize, fontsize = 18)
    ax = Axis(fig[1, 1]; title = @sprintf("%s barotropic (low-pass) SSH amplitude, co-phase lines every %.0f°", CON, DPHI), axkw...)
    hm = amp_map!(ax, lon_i, lat_i, 100 .* Abt_img, BT_MAX)
    cophase_lines!(ax, lon_c, lat_c, Zbt_con)
    Colorbar(fig[1, 2], hm; label = "Amplitude (cm)")
    f = joinpath(OUTDIR, "$(CON)_BT_amp_cophase_global_$(TAG).png"); save(f, fig; px_per_unit = 2)
    logmsg("  saved ", f)


    #= --- regional figures (every grid point) ------------------------------------
    logmsg("STEP 6/6  regional figures (full resolution)")
    for (name, x0, x1, y0, y1) in REGIONS
        I = findall(v -> x0 <= v <= x1, lon); J = findall(v -> y0 <= v <= y1, lat)
        (isempty(I) || isempty(J)) && (logmsg("  skip ", name); continue)
        I = I[1]:I[end]; J = J[1]:J[end]
        x, yy = lon[I], lat[J]
        zit = Zit[I, J]; zbt = Zbt[I, J]
        fig = Figure(size = (1700, 650), fontsize = 18)
        a1 = Axis(fig[1, 1]; title = "$name: internal-tide amplitude (cm), lines: barotropic co-phase",
                  xlabel = "Longitude (°)", ylabel = "Latitude (°)", aspect = DataAspect())
        h1 = amp_map!(a1, x, yy, 100 .* abs.(zit), IT_MAX)
        cophase_lines!(a1, x, yy, zbt)
        moorings!(a1)
        Colorbar(fig[1, 2], h1)
        a2 = Axis(fig[1, 3]; title = "$name: internal-tide SSH at t = 0, Re{Z_it} (cm)",
                  xlabel = "Longitude (°)", aspect = DataAspect())
        h2 = heatmap!(a2, x, yy, 100 .* real.(zit); colormap = :balance,
                      colorrange = (-IT_MAX, IT_MAX), nan_color = LAND_COLOR, rasterize = true)
        moorings!(a2)
        Colorbar(fig[1, 4], h2)
        f = joinpath(OUTDIR, "$(CON)_IT_$(name)_$(TAG).png"); save(f, fig; px_per_unit = 2)
        logmsg("  saved ", f)
    end
=#

    logmsg(@sprintf("FINISHED OK in %.1f min, peak memory %s", (time() - t0) / 60, mem()))
end


main()
close(LOGIO)




