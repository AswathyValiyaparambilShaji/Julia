using Printf, TOML, Dates


config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "config", "run_debug.toml"))
cfg     = TOML.parsefile(config_file)
basein  =  cfg["bp_box28_v2"]  # FIXED: was cfg["/nobackup/avaliyap/LLC4320_V2/Box27b/"] — that's a path, not a config key.
baseout = cfg["bp_box28"]      # <- verify these two key names against your run_debug.toml; rename if different.


# ── Fixed dimensions for THIS box ───────────────────────────────────────────────
NX, NY, NZ = 384, 337, 168


buf = 3
# 7 x 5 = 35 tiles, every tile exactly 60 x 72 x NZ (buffered).
#   x: 378 / 7 = 54 exactly -> nothing dropped
#   y: 331 / 5 = 66 remainder 1 -> 1 grid row dropped at the tail (y = 337)
ntiles_x = 7
ntiles_y = 5


# ── Variable lists ─────────────────────────────────────────────────────────────
vars_3d = ["hFacC_384x337x168"]
vars_2d = ["DXC", "DYC", "RAC","GEBCO2025_on_LLC4320_v16b"]


# ── Tile-bound computation (same function used for box27b's U/V/W/Salt/Theta) ──
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


buffered_range(core::Tuple{Int,Int}, buf::Int) = (core[1], core[2] + 2*buf)


xbounds, x_dropped = tile_core_bounds_uniform(NX - 2*buf, ntiles_x)
ybounds, y_dropped = tile_core_bounds_uniform(NY - 2*buf, ntiles_y)


println("x tiles: ", length(xbounds), " x uniform width ", xbounds[1][2]-xbounds[1][1]+1,
        " (buffered ", xbounds[1][2]-xbounds[1][1]+1+2*buf, "); dropped grid points: ", x_dropped)
println("y tiles: ", length(ybounds), " x uniform width ", ybounds[1][2]-ybounds[1][1]+1,
        " (buffered ", ybounds[1][2]-ybounds[1][1]+1+2*buf, "); dropped grid points: ", y_dropped)
println("total tiles: ", length(xbounds) * length(ybounds))


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


# ── Tiling kernel (driven by the bounds lists, NOT a fixed tx/ty stride) ───────
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
    input_dir  = joinpath(basein, "grid")
    output_dir = joinpath(baseout, varname)
    mkpath(output_dir)
    println("\n── $varname (3D) ──────────────────────────────────────────────")


    fpath = joinpath(input_dir, "$(varname)_$(NX)x$(NY)x$(NZ)")   # FIXED: was hardcoded "..._288x468x168"


    if !isfile(fpath)
        println("Missing: $fpath — skipping")
    else
        fld = read_3d(fpath)
        tile_and_append_3d!(fld, output_dir, varname, xbounds, ybounds, buf)
        fld = nothing
        GC.gc()
    end
    println("$varname complete → $output_dir")
end


# ── Process 2D variables ───────────────────────────────────────────────────────
for varname in vars_2d
    input_dir  = joinpath(basein, "grid")
    output_dir = joinpath(baseout, varname)
    mkpath(output_dir)
    println("\n── $varname (2D) ──────────────────────────────────────────────")


    fpath = joinpath(input_dir, "$(varname)_$(NX)x$(NY)")   # FIXED: was hardcoded "..._288x468"


    if !isfile(fpath)
        println("Missing: $fpath — skipping")
    else
        fld = read_2d(fpath)
        tile_and_append_2d!(fld, output_dir, varname, xbounds, ybounds, buf)
        fld = nothing
        GC.gc()
    end
    println("$varname complete → $output_dir")
end


println("\nAll variables tiled → $baseout")




