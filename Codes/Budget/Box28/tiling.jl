using Printf, TOML, Dates


config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "config", "run_debug.toml"))
cfg     = TOML.parsefile(config_file)
basein  = cfg["bp_box28_v2"]
baseout = cfg["bp_box28"]


# ── Fixed dimensions for THIS box ───────────────────────────────────────────────
# (388x337 box mentioned in the prompt is written here as 384x337 — fix if that
#  was a typo on either side)
NX, NY, NZ = 384, 337, 168


# ── Tiling parameters ──────────────────────────────────────────────────────────
buf = 3
# => 7 x 5 = 35 tiles, every single one exactly 60 x 72 x NZ. Only 1 row out of
# 337 in Y is left out of all tiling output entirely.
ntiles_x = 7
ntiles_y = 5


# ── Time info ────────────────────────────────────────────────────────────────
t_start = DateTime(2023, 5, 1, 0, 0, 0)
nt      = 558


# ── Variable lists ─────────────────────────────────────────────────────────────
vars_3d = ["V", "Salt", "Theta", "W"]
vars_2d = ["Eta", "oceTAUX", "oceTAUY"]


# ── Tile-bound computation (uniform tiles, drop any remainder) ─────────────────
# Splits `interior` (= N - 2*buf) into `ntiles` EQUAL-SIZE pieces of width
# `interior ÷ ntiles`. If `interior` isn't evenly divisible, the leftover
# `interior % ntiles` grid points are left uncovered at the far edge of the
# domain (reported via `dropped`) instead of making any tile a different size.
# Returns (bounds, dropped) where bounds is a Vector of (core_start, core_end)
# 1-based indices INTO THE INTERIOR (i.e. 1:interior).
function tile_core_bounds_uniform(interior::Int, ntiles::Int)
    tile_size = div(interior, ntiles)
    used      = tile_size * ntiles
    dropped   = interior - used
    bounds = Vector{Tuple{Int,Int}}(undef, ntiles)
    s = 1
    for i in 1:ntiles
        e = s + tile_size - 1
        bounds[i] = (s, e)
        s = e + 1
    end
    return bounds, dropped
end


# Maps a core bound (in interior-index space) to the buffered index range in
# full-grid space: [core_start, core_end + 2*buf] (mirrors the original
# xsb:xeb / ysb:yeb construction, including buf=0 exactly reproducing the
# no-buffer case).
buffered_range(core::Tuple{Int,Int}, buf::Int) = (core[1], core[2] + 2*buf)


xbounds, x_dropped = tile_core_bounds_uniform(NX - 2*buf, ntiles_x)
ybounds, y_dropped = tile_core_bounds_uniform(NY - 2*buf, ntiles_y)


println("x tiles: ", length(xbounds), " x uniform width ", xbounds[1][2]-xbounds[1][1]+1,
        " (buffered ", xbounds[1][2]-xbounds[1][1]+1+2*buf, "); dropped grid points: ", x_dropped)
println("y tiles: ", length(ybounds), " x uniform width ", ybounds[1][2]-ybounds[1][1]+1,
        " (buffered ", ybounds[1][2]-ybounds[1][1]+1+2*buf, "); dropped grid points: ", y_dropped)
if x_dropped > 0
    println("  -> x grid indices ", NX - x_dropped + 1, ":", NX, " are not covered by any tile")
end
if y_dropped > 0
    println("  -> y grid indices ", NY - y_dropped + 1, ":", NY, " are not covered by any tile")
end
println("total tiles: ", length(xbounds) * length(ybounds), " — every tile is exactly the same size")


# ── Readers ────────────────────────────────────────────────────────────────────
function read_3d(fpath)
    arr = Array{Float32}(undef, NX * NY * NZ)
    open(fpath, "r") do io; read!(io, arr); end
    arr .= ntoh.(arr)
    return reshape(arr, NX, NY, NZ)
end


function read_2d(fpath)
    arr = Array{Float32}(undef, NX * NY)
    open(fpath, "r") do io; read!(io, arr); end
    arr .= ntoh.(arr)
    return reshape(arr, NX, NY)
end


# ── Tiling kernel (now driven by the bounds lists, not a fixed stride) ─────────
function tile_and_append_3d!(fld, output_dir, varname, xbounds, ybounds, buf)
    for (xn, xc) in enumerate(xbounds)
        xsb, xeb = buffered_range(xc, buf)
        for (yn, yc) in enumerate(ybounds)
            ysb, yeb = buffered_range(yc, buf)
            blk = Float32.(fld[xsb:xeb, ysb:yeb, :])
            tile_file = joinpath(output_dir, @sprintf("%s_v2_%02dx%02d_%d.bin", varname, xn, yn, buf))
            open(tile_file, "a") do fid; write(fid, blk); end
        end
    end
end


function tile_and_append_2d!(fld, output_dir, varname, xbounds, ybounds, buf)
    for (xn, xc) in enumerate(xbounds)
        xsb, xeb = buffered_range(xc, buf)
        for (yn, yc) in enumerate(ybounds)
            ysb, yeb = buffered_range(yc, buf)
            blk = Float32.(fld[xsb:xeb, ysb:yeb])
            tile_file = joinpath(output_dir, @sprintf("%s_v2_%02dx%02d_%d.bin", varname, xn, yn, buf))
            open(tile_file, "a") do fid; write(fid, blk); end
        end
    end
end


# ── Process 3D variables ───────────────────────────────────────────────────────
for varname in vars_3d
    input_dir  = joinpath(basein, varname)
    output_dir = joinpath(baseout, varname)
    mkpath(output_dir)
    println("\n── $varname (3D) ──────────────────────────────────────────────")


    for ts in 1:nt
        dt    = t_start + Hour(ts - 1)
        dtstr = Dates.format(dt, "yyyymmddTHHMMSS")
        fpath = joinpath(input_dir, "$(varname)_$(NX)x(NY)x(NZ).$dtstr")


        if !isfile(fpath)
            println("Missing: $fpath — skipping")
            continue
        end


        fld = read_3d(fpath)
        tile_and_append_3d!(fld, output_dir, varname, xbounds, ybounds, buf)
        fld = nothing
        GC.gc()
    end
    println("$varname complete → $output_dir")
end


# ── Process 2D variables ───────────────────────────────────────────────────────
for varname in vars_2d
    input_dir  = joinpath(basein, varname)
    output_dir = joinpath(baseout, varname)
    mkpath(output_dir)
    println("\n── $varname (2D) ──────────────────────────────────────────────")


    for ts in 1:nt
        dt    = t_start + Hour(ts - 1)
        dtstr = Dates.format(dt, "yyyymmddTHHMMSS")
        fpath = joinpath(input_dir, "$(varname)_$(NX)x(NY).dtstr")


        if !isfile(fpath)
            println("Missing: $fpath — skipping")
            continue
        end


        fld = read_2d(fpath)
        tile_and_append_2d!(fld, output_dir, varname, xbounds, ybounds, buf)   # was tile_and_append_2! (undefined) in the original
        fld = nothing
        GC.gc()
    end
    println("$varname complete → $output_dir")
end


println("\nAll variables tiled → $baseout")




