using Printf, TOML, Dates


config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "config", "run_debug.toml"))
cfg     = TOML.parsefile(config_file)
basein  = cfg["base_pathv2"]
baseout = cfg["base_path_V2"]


# ── Fixed dimensions ───────────────────────────────────────────────────────────
NX, NY, NZ = 288, 468, 168


# ── Tiling parameters ──────────────────────────────────────────────────────────
buf    = 3
tx, ty = 47, 66




# ── Variable lists ─────────────────────────────────────────────────────────────
# (varname, filename_suffix, is_3d)
vars_3d = ["GEBCO2025_on_LLC4320_v16b"]
#vars_2d = ["DXC", "DYC", "RAC"]


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


# ── Tiling kernel ──────────────────────────────────────────────────────────────
function tile_and_append_3d!(fld, output_dir, varname)
    xn = 1
    for xs in (buf+1):tx:(NX-buf)
        xsb, xeb = xs - buf, xs + tx - 1 + buf
        yn = 1
        for ys in (buf+1):ty:(NY-buf)
            ysb, yeb = ys - buf, ys + ty - 1 + buf
            blk = Float32.(fld[xsb:xeb, ysb:yeb, :])
            tile_file = joinpath(output_dir, @sprintf("%s_v2_%02dx%02d_%d.bin", varname, xn, yn, buf))
            open(tile_file, "a") do fid; write(fid, blk); end
            yn += 1
        end
        xn += 1
    end
end


function tile_and_append_2d!(fld, output_dir, varname)
    xn = 1
    for xs in (buf+1):tx:(NX-buf)
        xsb, xeb = xs - buf, xs + tx - 1 + buf
        yn = 1
        for ys in (buf+1):ty:(NY-buf)
            ysb, yeb = ys - buf, ys + ty - 1 + buf
            blk = Float32.(fld[xsb:xeb, ysb:yeb])
            tile_file = joinpath(output_dir, @sprintf("%s_v2_%02dx%02d_%d.bin", varname, xn, yn, buf))
            open(tile_file, "a") do fid; write(fid, blk); end
            yn += 1
        end
        xn += 1
    end
end







# ── Process 3D variables ───────────────────────────────────────────────────────
for varname in vars_3d
    input_dir  = joinpath(basein, "grid")
    output_dir = joinpath(baseout, varname)
    mkpath(output_dir)
    println("\n── $varname (3D) ──────────────────────────────────────────────")


    #for ts in 1:nt
        #dt    = t_start + Hour(ts - 1)
        #dtstr = Dates.format(dt, "yyyymmddTHHMMSS")
        fpath = joinpath(input_dir, "$(varname)_288x468x168")


        if !isfile(fpath)
            println("Missing: $fpath — skipping")
            continue
        end


        fld = read_3d(fpath)
        tile_and_append_3d!(fld, output_dir, varname)
        fld = nothing
        GC.gc()
        #println("done: $ts / $nt  ($dtstr)")
    #end
    println("$varname complete → $output_dir")
end


#= ── Process 2D variables ───────────────────────────────────────────────────────
for varname in vars_2d
    input_dir  = joinpath(basein, "grid")
    output_dir = joinpath(baseout, varname)
    mkpath(output_dir)
    println("\n── $varname (2D) ──────────────────────────────────────────────")


    #for ts in 1:nt
       
        fpath = joinpath(input_dir, "$(varname)_288x468")


        if !isfile(fpath)
            println("Missing: $fpath — skipping")
            continue
        end


        fld = read_2d(fpath)
        tile_and_append_2d!(fld, output_dir, varname)
        fld = nothing
        GC.gc()
        #println("done: $ts / $nt  ($dtstr)")
    #end
    println("$varname complete → $output_dir")
end
=#

println("\nAll variables tiled → $baseout")




