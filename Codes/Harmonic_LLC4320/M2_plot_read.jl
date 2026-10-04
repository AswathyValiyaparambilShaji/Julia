#=
plot_M2_global.jl
-----------------------------------------------------------------------------
Global maps of the llc4320_v2 M2 SSH tide (Dan Whitt's utide fit to Eta,
27 Mar 2023 – 30 Jan 2024):


  figures/M2_amp_cophase_global.png : amplitude (colour) + co-phase lines
  figures/M2_phase_global.png       : Greenwich phase lag (cyclic colour map)


How it works
  1. Read the compact LLC binary files (big-endian Float32, 4320 x 56160).
     Each file holds ALL 13 faces packed one after another.
  2. Cut out the 4 lat-lon facets (skip the Arctic cap), rotate the two
     sideways facets, and put them side by side -> one 17280 x 12960 map.
  3. Average over blocks of grid points (block averaging) to make arrays a
     plotting library can handle (a 224-million-pixel image is far too big).
  4. Draw co-phase lines from a smoother, more heavily averaged complex field
     so they don't get crowded:
       - averaging the COMPLEX amplitude cancels the small-scale internal-tide
         wiggles and keeps the large-scale (barotropic) phase pattern;
       - lines are drawn every DPHI degrees (default 30° ≈ 1 h of M2).


Run on a NAS compute node (needs ~8–10 GB RAM), e.g.
    qsub -I -q devel -l select=1:ncpus=40:model=cas_ait -l walltime=1:00:00
    julia -t 40 plot_M2_global.jl


Packages:  julia> ] add CairoMakie
-----------------------------------------------------------------------------
=#


using Base.Threads, Statistics, Printf, Dates, Logging


# ============================ log file ========================================
# Everything the script reports (progress, numbers to check, warnings, and the
# full error message if it crashes) is written to   figures/plot_M2_global.log
# so you don't need to read the terminal. Open it with  `less`/`cat`, or copy it
# to your laptop, after the run.
const FIGDIR_ = "/nobackup/avaliyap/Figure/Harmonic_LLC4320/"
mkpath(FIGDIR_)
const LOGFILE = joinpath(FIGDIR_, "plot_M2_global.log")
const LOGIO   = open(LOGFILE, "w")


"Write a time-stamped line to the log file (and to the terminal)."
function logmsg(parts...)
    line = string(Dates.format(now(), "HH:MM:SS"), "  ", parts...)
    println(LOGIO, line); flush(LOGIO)       # flush: line is saved even if the job dies
    println(stdout, line)
end
global_logger(SimpleLogger(LOGIO))           # package warnings also go to the log file


logmsg("started on host ", gethostname(), " with ", nthreads(), " threads, Julia ", VERSION)


# Loading the plotting package is the most common first-run failure -> log it.
try
    @eval using CairoMakie
    logmsg("CairoMakie loaded")
catch err
    logmsg("ERROR: could not load CairoMakie. Install it once with:  julia -e 'using Pkg; Pkg.add(\"CairoMakie\")'")
    logmsg(sprint(showerror, err))
    close(LOGIO); exit(1)
end


# ============================ user settings ==================================
const NX       = 4320                                   # face size of llc4320
const COEFDIR  = "/nobackupp27/dbwhitt/llc_4320/OUT/proc/tidal_harmonic_fit/coefs"
const GRIDDIR  = "/nobackupp27/dbwhitt/llc_4320/grid_90x90x19493"
const AMPFILE  = joinpath(COEFDIR, "Eta_tide_amp_M2.data")   # amplitude [m]
const PHAFILE  = joinpath(COEFDIR, "Eta_tide_pha_M2.data")   # phase (expected degrees)
const FIGDIR   = FIGDIR_                                 # output folder (created above)


const F_IMG    = 4      # block size for the colour image  (17280x12960 -> 4320x3240)
const F_CON    = 16     # block size for the phase lines    (-> 1080x810, ~30–40 km blocks)
                        #   increase (e.g. 24 or 32) for even smoother, less crowded lines
const DPHI     = 30.0   # spacing of co-phase lines [deg]; 30° ≈ 1.03 h of M2
                        #   use 360*3/12.4206 ≈ 86.9 for "every 3 hours"
const AMP_MAX  = 100.0  # colour-scale limit for amplitude [cm]
# ==============================================================================




# ----------------------------------------------------------------------------
# 1. Reading the compact LLC binary file
# ----------------------------------------------------------------------------
"""
    read_llc(fn) -> Matrix{Float32} (nx × 13nx)


Same as Dan's MATLAB:  fread(fid,[nx ny],'float32') with 'ieee-be'.
Julia, like MATLAB/Fortran, stores arrays column-major, so the layout matches.
"""
function read_llc(fn::AbstractString; nx::Int = NX)
    expected = nx * 13nx * 4                       # bytes = points × 4 bytes
    filesize(fn) == expected ||
        error("$fn is $(filesize(fn)) bytes, expected expected(compactllc(nx) format)")
    A = Matrix{Float32}(undef, nx, 13nx)
    read!(fn, A)                                   # raw bytes -> Float32
    A .= ntoh.(A)                                  # big-endian -> this machine's byte order
    logmsg("read ", basename(fn), "  size ", size(A), "  min/max = ",
           minimum(A), " / ", maximum(A))
    return A
end


# ----------------------------------------------------------------------------
# 2. Cutting the file into facets and stitching a global lon x lat map
# ----------------------------------------------------------------------------
"""
Columns of the compact array:
    1      : 3nx   facet 1 (faces 1–3)   nx × 3nx, already lon × lat
    3nx+1  : 6nx   facet 2 (faces 4–6)   nx × 3nx, already lon × lat
    6nx+1  : 7nx   facet 3 (face 7)      Arctic cap -> not used here
    7nx+1  : 10nx  facet 4 (faces 8–10)  stored sideways: reshape to 3nx × nx
    10nx+1 : 13nx  facet 5 (faces 11–13) stored sideways: reshape to 3nx × nx
"""
function llc_facets(A; nx = size(A, 1))
    f1 = A[:, 1:3nx]
    f2 = A[:, 3nx+1:6nx]
    f4 = reshape(A[:, 7nx+1:10nx],  3nx, nx)
    f5 = reshape(A[:, 10nx+1:13nx], 3nx, nx)
    return f1, f2, f4, f5
end


# Blank (all-land) tiles were dropped from the run, so their XC = YC = 0.
valid_pt(x, y) = !(x == 0 && y == 0)
wrap180(d) = mod(d + 180, 360) - 180              # longitude difference in (-180,180]


"""
The sideways facets must be transposed and possibly flipped. Instead of
hard-coding it, try the 4 possibilities and keep the one in which longitude
increases along dim 1 (eastward) and latitude along dim 2 (northward).
"""
function find_orientation(xc4, yc4)
    candidates = (f -> permutedims(f),
                  f -> reverse(permutedims(f), dims = 1),
                  f -> reverse(permutedims(f), dims = 2),
                  f -> reverse(permutedims(f), dims = (1, 2)))
    for (k, g) in enumerate(candidates)
        x, y = g(xc4), g(yc4)
        east = 0; north = 0
        for j in 1:97:size(x, 2)-1, i in 1:97:size(x, 1)-1     # sparse sample is enough
            if valid_pt(x[i, j], y[i, j]) && valid_pt(x[i+1, j], y[i+1, j]) &&
               valid_pt(x[i, j+1], y[i, j+1])
                east  += sign(wrap180(x[i+1, j] - x[i, j]))
                north += sign(y[i, j+1] - y[i, j])
            end
        end
        if east > 0 && north > 0
            logmsg("  rotated facet orientation: option ", k, " (east score ", east,
                   ", north score ", north, ")")
            return g
        end
    end
    error("could not determine the orientation of a rotated facet")
end


"Facets 1, 2, 4, 5 side by side -> (4nx) × (3nx) = 17280 × 12960, dim1 = lon, dim2 = lat."
function to_rect(A, g4, g5)
    f1, f2, f4, f5 = llc_facets(A)
    return vcat(f1, f2, g4(f4), g5(f5))
end


# ----------------------------------------------------------------------------
# 3. 1-D lon / lat vectors of the stitched map
#    (in this part of the llc grid lon depends only on i, lat only on j)
# ----------------------------------------------------------------------------
"Linear fill of NaNs (incl. extrapolation at the ends) in a monotonic vector."
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


function lonlat_vectors(XC, YC)
    n1, n2 = size(XC)
    lon = fill(NaN, n1); lat = fill(NaN, n2)
    @threads for i in 1:n1                       # median over (sampled) valid points
        v = Float64[XC[i, j] for j in 1:16:n2 if valid_pt(XC[i, j], YC[i, j])]
        isempty(v) || (lon[i] = median(v))
    end
    @threads for j in 1:n2
        v = Float64[YC[i, j] for i in 1:16:n1 if valid_pt(XC[i, j], YC[i, j])]
        isempty(v) || (lat[j] = median(v))
    end
    # unwrap longitude (remove the 180 -> -180 jump) so it can be filled/averaged
    g = findall(!isnan, lon)
    for k in 2:length(g)
        lon[g[k]] = lon[g[k-1]] + wrap180(lon[g[k]] - lon[g[k-1]])
    end
    fill_coord!(lon)                             # columns that were entirely blank tiles
    fill_coord!(lat)                             # Antarctic-interior rows (all land)
    return lon, lat                              # lon is unwrapped (continuous)
end


# ----------------------------------------------------------------------------
# 4. Block averaging (ocean points only)
# ----------------------------------------------------------------------------
nanof(::Type{T}) where {T<:Real}    = T(NaN)
nanof(::Type{T}) where {T<:Complex} = T(NaN, NaN)


"""
Average F over f×f blocks using only ocean points (mask m). A block is kept if
at least half of it is ocean, otherwise NaN (shown as land).
"""
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


"Wrap lon to [-180,180) and reorder columns so lon increases (Makie needs monotonic axes)."
function sort_lon(lon, mats...)
    lw = wrap180.(lon)
    p = sortperm(lw)
    return lw[p], map(M -> M[p, :], mats)...
end


# ----------------------------------------------------------------------------
# 5. Main
# ----------------------------------------------------------------------------
mem() = @sprintf("%.1f GB", Sys.maxrss() / 1e9)      # peak memory used so far


function main()
    t0 = time()
    logmsg("output folder: ", abspath(FIGDIR))


    # --- grid: orientation of rotated facets + lon/lat vectors ---------------
    logmsg("STEP 1/5  reading grid (XC, YC)")
    xc = read_llc(joinpath(GRIDDIR, "XC.data"))
    yc = read_llc(joinpath(GRIDDIR, "YC.data"))
    _, _, x4, x5 = llc_facets(xc); _, _, y4, y5 = llc_facets(yc)
    logmsg("  facet 4:"); g4 = find_orientation(x4, y4)
    logmsg("  facet 5:"); g5 = find_orientation(x5, y5)
    lon, lat = lonlat_vectors(to_rect(xc, g4, g5), to_rect(yc, g4, g5))
    xc = yc = x4 = x5 = y4 = y5 = nothing; GC.gc()
    logmsg(@sprintf("  stitched map: %d lon x %d lat, lat range %.2f .. %.2f  (CHECK: ~-80 .. ~57-70)",
                    length(lon), length(lat), lat[1], lat[end]))
    logmsg("  memory so far: ", mem())


    # --- M2 amplitude and phase ----------------------------------------------
    logmsg("STEP 2/5  reading M2 amplitude and phase")
    amp = to_rect(read_llc(AMPFILE), g4, g5)          # [m]
    pha = to_rect(read_llc(PHAFILE), g4, g5)          # [deg] expected
    ocean = (amp .!= 0) .& isfinite.(amp) .& isfinite.(pha)
    pmin, pmax = extrema(pha[ocean])
    logmsg(@sprintf("  ocean points: %.1f %% of map", 100 * count(ocean) / length(ocean)))
    logmsg(@sprintf("  amplitude: mean %.3f m, max %.3f m  (CHECK: max ~1-5 m on shelves)",
                    mean(amp[ocean]), maximum(amp[ocean])))
    logmsg(@sprintf("  phase range: %.2f .. %.2f  (CHECK: 0..360 or -180..180 = degrees)", pmin, pmax))
    if pmax <= 2π + 0.01                               # safety check on units
        logmsg("  WARNING: phase looks like radians -> converting to degrees")
        pha .= rad2deg.(pha)
    end


    # complex amplitude: η(t) = A cos(ωt − g) = Re{Z e^{iωt}},  Z = A e^{−ig}
    Z = ComplexF32.(amp .* cis.(-deg2rad.(pha)))
    pha = nothing; GC.gc()
    logmsg("  memory so far: ", mem())


    # --- coarse fields for plotting ------------------------------------------
    logmsg("STEP 3/5  block averaging (image ", F_IMG, "x", F_IMG, ", lines ", F_CON, "x", F_CON, ")")
    A_img = block_mean(amp, ocean, F_IMG)              # mean amplitude (image)
    Z_img = block_mean(Z,   ocean, F_IMG)              # for the phase map
    Z_con = block_mean(Z,   ocean, F_CON)              # smoother, for the lines
    amp = Z = nothing; GC.gc()


    lon_img, A_img, Z_img = sort_lon(block_mean(lon, F_IMG), A_img, Z_img)
    lat_img               = block_mean(lat, F_IMG)
    lon_con, Z_con        = sort_lon(block_mean(lon, F_CON), Z_con)
    lat_con               = block_mean(lat, F_CON)
    logmsg("  image grid ", size(A_img), ", line grid ", size(Z_con),
           ", lon ", round(lon_img[1], digits = 2), " .. ", round(lon_img[end], digits = 2))


    G_img = mod.(-rad2deg.(angle.(Z_img)), 360)        # Greenwich phase lag [deg]
    ylims = (max(-80, lat_img[1]), lat_img[end])


    # --- Figure 1: amplitude + co-phase lines -----------------------------------
    logmsg("STEP 4/5  figure 1: amplitude + co-phase lines every ", DPHI, " deg")
    fig = Figure(size = (1800, 900), fontsize = 18)
    ax = Axis(fig[1, 1]; xlabel = "Longitude (°)", ylabel = "Latitude (°)",
              title = @sprintf("llc4320_v2 M2 SSH amplitude, co-phase lines every %.0f° (%.2f h); thick = 0°",
                               DPHI, DPHI / 360 * 12.4206),
              limits = (-180, 180, ylims...), xticks = -180:60:180, yticks = -60:30:60)
    hm = heatmap!(ax, lon_img, lat_img, 100 .* A_img; colormap = :viridis,
                  colorrange = (0, AMP_MAX), highclip = :yellow, nan_color = :gray80,
                  rasterize = true)
    # A co-phase line for phase θ is where Z e^{iθ} is real and positive:
    # zero contour of Im{Z e^{iθ}}, keeping only Re{Z e^{iθ}} > 0.
    # (Contouring the phase directly would draw a false line at the 360°→0° jump.)
    for θ in 0:DPHI:359.999
        W = Z_con .* cis(deg2rad(θ))
        F = Float32.(ifelse.(real.(W) .> 0, imag.(W), NaN))
        contour!(ax, lon_con, lat_con, F; levels = [0.0],
                 color = θ == 0 ? :white : (:white, 0.75),
                 linewidth = θ == 0 ? 2.0 : 0.8)
    end
    Colorbar(fig[1, 2], hm; label = "Amplitude (cm)")
    f1 = joinpath(FIGDIR, "M2_amp_cophase_global.png")
    save(f1, fig; px_per_unit = 2)
    logmsg("  saved ", f1)


    # --- Figure 2: phase map (cyclic colour map) -----------------------------
    logmsg("STEP 5/5  figure 2: phase map")
    fig2 = Figure(size = (1800, 900), fontsize = 18)
    ax2 = Axis(fig2[1, 1]; xlabel = "Longitude (°)", ylabel = "Latitude (°)",
               title = "llc4320_v2 M2 SSH Greenwich phase lag",
               limits = (-180, 180, ylims...), xticks = -180:60:180, yticks = -60:30:60)
    hm2 = heatmap!(ax2, lon_img, lat_img, G_img; colormap = :romaO,   # cyclic: 0° = 360°
                   colorrange = (0, 360), nan_color = :gray80, rasterize = true)
    Colorbar(fig2[1, 2], hm2; label = "Phase (°)", ticks = 0:60:360)
    f2 = joinpath(FIGDIR, "M2_phase_global.png")
    save(f2, fig2; px_per_unit = 2)
    logmsg("  saved ", f2)


    logmsg(@sprintf("FINISHED OK in %.1f min, peak memory %s", (time() - t0) / 60, mem()))
end


# Run, and if anything goes wrong write the full error + where it happened to the log.
try
    main()
catch err
    logmsg("ERROR -- the script stopped. Full message and line numbers below:")
    logmsg(sprint(showerror, err, catch_backtrace()))
    close(LOGIO)
    exit(1)
end
close(LOGIO)




