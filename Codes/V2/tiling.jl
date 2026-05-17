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


# ── Paths ──────────────────────────────────────────────────────────────────────
input_dir  = joinpath(basein, "U")
output_dir = joinpath(baseout, "U")
mkpath(output_dir)


# ── Time info ─────────────────────────────────────────────────────────────────
# 2023-05-01 00:00 → 2023-05-24 06:00, hourly → 558 snapshots
t_start = DateTime(2023, 5, 1, 0, 0, 0)
nt      = 558


# ── Read one big-endian Float32 snapshot ──────────────────────────────────────
function read_snapshot(fpath)
    arr = Array{Float32}(undef, NX * NY * NZ)
    open(fpath, "r") do io; read!(io, arr); end
    arr .= ntoh.(arr)
    return reshape(arr, NX, NY, NZ)
end



# ── Process ────────────────────────────────────────────────────────────────────
for ts in 1:nt


    dt    = t_start + Hour(ts - 1)
    dtstr = Dates.format(dt, "yyyymmddTHHMMSS")
    fpath = joinpath(input_dir, "U_288x468x168.$dtstr")


    if !isfile(fpath)
        println("Missing: $fpath — skipping")
        continue
    end


    U = read_snapshot(fpath)


    xn = 1
    for xs in (buf+1):tx:(NX-buf)
        xsb, xeb = xs - buf, xs + tx - 1 + buf
        yn = 1
        for ys in (buf+1):ty:(NY-buf)
            ysb, yeb = ys - buf, ys + ty - 1 + buf


            blk = Float32.(U[xsb:xeb, ysb:yeb, :])


            tile_file = joinpath(output_dir, @sprintf("U_v2_%02dx%02d_%d.bin", xn, yn, buf))
            open(tile_file, "a") do fid
                write(fid, blk)
            end


            yn += 1
        end
        xn += 1
    end


    U = nothing
    GC.gc()
    println("done: $ts / $nt  ($dtstr)")
end


println("Tiling complete → $output_dir")




