using Printf, TOML, Dates


config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "config", "run_debug.toml"))
cfg     = TOML.parsefile(config_file)
basein  =  cfg["bp_box27b_v2"]
baseout = cfg["bp_box27b"]


# Fixed dimensions for THIS box 
NX, NY, NZ = 1056, 1030, 170


#  Tiling parameters 
buf = 3

# 

ntiles_x = 7
ntiles_y = 7


#  Time info 
t_start = DateTime(2023, 5, 1, 0, 0, 0)
nt      = 558


#  Variable lists 
vars_3d = ["V", "Salt", "Theta", "W","U"]
vars_2d = ["Eta", "oceTAUX", "oceTAUY"]

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
println("total tiles: ", length(xbounds) * length(ybounds), " — every tile is exactly the same size")


# ── Readers 
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


# ── Tiling kernel (driven by the bounds lists) 
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


# ── Process 3D variables 
for varname in vars_3d
    input_dir  = joinpath(basein, varname)
    output_dir = joinpath(baseout, varname)
    mkpath(output_dir)
    println("\n $varname (3D) ")


    for ts in 1:nt
        dt    = t_start + Hour(ts - 1)
        dtstr = Dates.format(dt, "yyyymmddTHHMMSS")
        fpath = joinpath(input_dir, "$(varname)_$(NX)x$(NY)x$(NZ).$dtstr")


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


# ── Process 2D variables 
for varname in vars_2d
    input_dir  = joinpath(basein, varname)
    output_dir = joinpath(baseout, varname)
    mkpath(output_dir)
    println("\n $varname (2D) ")


    for ts in 1:nt
        dt    = t_start + Hour(ts - 1)
        dtstr = Dates.format(dt, "yyyymmddTHHMMSS")
        fpath = joinpath(input_dir, "$(varname)_$(NX)x$(NY).$dtstr")


        if !isfile(fpath)
            println("Missing: $fpath — skipping")
            continue
        end


        fld = read_2d(fpath)
        tile_and_append_2d!(fld, output_dir, varname, xbounds, ybounds, buf)
        fld = nothing
        GC.gc()
    end
    println("$varname complete → $output_dir")
end


println("\nAll variables tiled → $baseout")




