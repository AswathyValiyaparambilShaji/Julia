using Printf, TOML, Dates


config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "config", "run_debug.toml"))
cfg     = TOML.parsefile(config_file)
basein  = cfg["/nobackup/avaliyap/LLC4320_V2/Box27b/grid/"]   # <- same keys used in the fixed box28 U/V/W/Salt/Theta script
baseout = cfg["bp_box27b"]


# ── Fixed dimensions for THIS box ───────────────────────────────────────────────
NX, NY, NZ = 1056, 1030, 170


# ── Tiling parameters (must match the box28 U/V/W/Salt/Theta tiling exactly,
#    so grid tiles line up spatially with the state-variable tiles) ────────────
buf = 3
# 7 x 7 = 49 tiles, every tile exactly 156 x 152 x NZ (buffered).
#   x: 1050 / 7 = 150 exactly -> nothing dropped
#   y: 1024 / 7 = 146 remainder 2 -> dropped SYMMETRICALLY: 1 grid row at the
#      very start (y = 1) and 1 grid row at the very end (y = 1030)
ntiles_x = 7
ntiles_y = 7


# ── Variable lists ─────────────────────────────────────────────────────────────
vars_3d = ["hFacC_1056x1030x170"]
vars_2d = ["DXC", "DYC", "RAC","GEBCO2025_on_LLC4320_v16b"]


# ── Tile-bound computation — SYMMETRIC drop, same function used for box28's
#    U/V/W/Salt/Theta tiling (tile_1056x1030_7x7.jl) ────────────────────────────
function tile_core_bounds_symmetric(interior::Int, ntiles::Int)
    tile_size = div(interior, ntiles)
    used      = tile_size * ntiles
    drop      = interior - used
    front     = div(drop, 2)
    back      = drop - front
    bounds = Vector{Tuple{Int,Int}}(undef, ntiles)
    s = 1 + front
    for i in 1:ntiles
        e = s + tile_size - 1
        bounds[i] = (s, e)
        s = e + 1
    end
    return bounds, front, back
end


buffered_range(core::Tuple{Int,Int}, buf::Int) = (core[1], core[2] + 2*buf)


xbounds, x_front, x_back = tile_core_bounds_symmetric(NX - 2*buf, ntiles_x)
ybounds, y_front, y_back = tile_core_bounds_symmetric(NY - 2*buf, ntiles_y)


xw = xbounds[1][2] - xbounds[1][1] + 1
yw = ybounds[1][2] - ybounds[1][1] + 1
println("x tiles: $ntiles_x x uniform width $xw (buffered $(xw+2*buf)); dropped $(x_front) at start, $(x_back) at end")
println("y tiles: $ntiles_y x uniform width $yw (buffered $(yw+2*buf)); dropped $(y_front) at start, $(y_back) at end")
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




