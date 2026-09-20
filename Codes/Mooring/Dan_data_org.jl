using NCDatasets, Statistics, Printf


# Combines all site_00NN.nc files into one NetCDF shaped like the old
# Moorings_88_timeseries.nc: destaggers U/V, rotates to true east/north,
# keeps Theta/Salt at the center cell, stacks all sites together.


# ---- EDIT THESE ----
sites_dir = "/nobackupp27/dbwhitt/llc_4320/OUT/regions/moorings/sites/"
outfile   = "/nobackupp/avaliyap/V2/Moorings/Moorings_88_combined.nc"
logfile   = "/nobackupp/avaliyap/V2/Moorings/build_mooring_netcdf_v2_log.txt"
# ---------------------


mkpath(dirname(outfile))
mkpath(dirname(logfile))


function load_mooring_site(ncfile::AbstractString)
    ds = NCDataset(ncfile, "r")


    lat          = ds.attrib["lat"]
    lon          = ds.attrib["lon"]
    site_id      = ds.attrib["site"]
    face         = ds.attrib["face"]
    AngleCS      = ds.attrib["AngleCS"]
    AngleSN      = ds.attrib["AngleSN"]
    depth_bottom = ds.attrib["Depth"]


    RC  = Array(ds["RC"])
    RF  = Array(ds["RF"])
    DRF = Array(ds["DRF"])
    wet = Array(ds["wet"])


    time = Array(ds["time"])
    have = Array(ds["have"])


    U_raw     = Float64.(Array(ds["U"]))
    V_raw     = Float64.(Array(ds["V"]))
    Theta_raw = Float64.(Array(ds["Theta"]))
    Salt_raw  = Float64.(Array(ds["Salt"]))


    close(ds)


    jc, ic = 2, 2   # center of the 3x3 block


    # destagger: average U over cellcol (i), V over cellrow (j)
    U_center = dropdims(mean(U_raw[jc, :, :, :], dims=1), dims=1)
    V_center = dropdims(mean(V_raw[:, ic, :, :], dims=1), dims=1)


    # Theta/Salt: center cell only, not averaged
    Theta_center = Theta_raw[jc, ic, :, :]
    Salt_center  = Salt_raw[jc, ic, :, :]


    # rotate model-native (U,V) -> true (east,north)
    U_east  = U_center .* AngleCS .- V_center .* AngleSN
    V_north = U_center .* AngleSN .+ V_center .* AngleCS


    return (; site_id, lat, lon, face, AngleCS, AngleSN, depth_bottom,
            RC, RF, DRF, wet, time, have,
            U_east, V_north, Theta = Theta_center, Salt = Salt_center)
end


function site_number(fname)
    m = match(r"site_(\d+)\.nc$", fname)
    return m === nothing ? typemax(Int) : parse(Int, m.captures[1])
end


files = filter(f -> occursin(r"^site_\d+\.nc$", f), readdir(sites_dir))
sort!(files, by = site_number)
N_moor = length(files)


logio = open(logfile, "w")
redirect_stdout(logio)
redirect_stderr(logio)


try
    println("Found $N_moor site files in $sites_dir")
    N_moor == 0 && error("No site_*.nc files found in $sites_dir")


    first_site = load_mooring_site(joinpath(sites_dir, files[1]))
    nz   = length(first_site.RC)
    nzp1 = length(first_site.RF)
    nt   = length(first_site.time)
    DRF  = first_site.DRF
    RC   = first_site.RC
    RF   = first_site.RF


    println("nz=$nz, nz+1=$nzp1, nt=$nt (from $(files[1]))")


    lon_out     = fill(NaN, N_moor)
    lat_out     = fill(NaN, N_moor)
    AngleCS_out = fill(NaN, N_moor)
    AngleSN_out = fill(NaN, N_moor)
    face_out    = fill(0,   N_moor)
    site_id_out = fill(0,   N_moor)
    depth_out   = fill(NaN, N_moor)
    wet_out     = zeros(Int32, N_moor, nz)


    U_east_out  = fill(NaN32, nt, N_moor, nz)
    V_north_out = fill(NaN32, nt, N_moor, nz)
    Theta_out   = fill(NaN32, nt, N_moor, nz)
    Salt_out    = fill(NaN32, nt, N_moor, nz)


    for (p, fname) in enumerate(files)
        site = load_mooring_site(joinpath(sites_dir, fname))


        if length(site.time) != nt
            println("  WARNING site $p ($fname): time length $(length(site.time)) != $nt -- skipping")
            continue
        end
        if !isapprox(site.DRF, DRF; rtol=1e-6)
            println("  WARNING site $p ($fname): DRF differs from site 1")
        end


        lon_out[p]     = site.lon
        lat_out[p]     = site.lat
        AngleCS_out[p] = site.AngleCS
        AngleSN_out[p] = site.AngleSN
        face_out[p]    = site.face
        site_id_out[p] = site.site_id
        depth_out[p]   = site.depth_bottom
        wet_out[p, :]  = site.wet


        # (k,t) -> (t,k) for storage as (time, station, depth)
        U_east_out[:, p, :]  = Float32.(permutedims(site.U_east,  (2, 1)))
        V_north_out[:, p, :] = Float32.(permutedims(site.V_north, (2, 1)))
        Theta_out[:, p, :]   = Float32.(permutedims(site.Theta,   (2, 1)))
        Salt_out[:, p, :]    = Float32.(permutedims(site.Salt,    (2, 1)))


        println("  site $p/$N_moor solved ($fname): lat=$(site.lat), lon=$(site.lon), n_wet=$(sum(site.wet))")
        flush(logio)
    end


    isfile(outfile) && rm(outfile)
    ds_out = NCDataset(outfile, "c")


    defDim(ds_out, "time", nt)
    defDim(ds_out, "station", N_moor)
    defDim(ds_out, "depth", nz)
    defDim(ds_out, "depth_p1", nzp1)


    v = defVar(ds_out, "lon", Float64, ("station",)); v.attrib["units"] = "degrees_east"; v[:] = lon_out
    v = defVar(ds_out, "lat", Float64, ("station",)); v.attrib["units"] = "degrees_north"; v[:] = lat_out
    v = defVar(ds_out, "AngleCS", Float64, ("station",)); v[:] = AngleCS_out
    v = defVar(ds_out, "AngleSN", Float64, ("station",)); v[:] = AngleSN_out
    v = defVar(ds_out, "face", Int32, ("station",)); v[:] = face_out
    v = defVar(ds_out, "site_id", Int32, ("station",)); v[:] = site_id_out
    v = defVar(ds_out, "bottom_depth", Float64, ("station",)); v.attrib["units"] = "m"; v[:] = depth_out


    v = defVar(ds_out, "DRF", Float64, ("depth",)); v.attrib["units"] = "m"; v[:] = DRF
    v = defVar(ds_out, "RC", Float64, ("depth",));  v.attrib["units"] = "m"; v[:] = RC
    v = defVar(ds_out, "RF", Float64, ("depth_p1",)); v.attrib["units"] = "m"; v[:] = RF


    v = defVar(ds_out, "wet", Int32, ("station", "depth")); v[:, :] = wet_out


    v = defVar(ds_out, "hFacC", Float64, ("station", "depth"))
    v.attrib["note"] = "0/1 mask aliased from 'wet', not a fractional hFacC"
    v[:, :] = Float64.(wet_out)


    v = defVar(ds_out, "U_east", Float32, ("time", "station", "depth")); v.attrib["units"] = "m/s"; v[:, :, :] = U_east_out
    v = defVar(ds_out, "V_north", Float32, ("time", "station", "depth")); v.attrib["units"] = "m/s"; v[:, :, :] = V_north_out
    v = defVar(ds_out, "Theta", Float32, ("time", "station", "depth")); v[:, :, :] = Theta_out
    v = defVar(ds_out, "Salt", Float32, ("time", "station", "depth")); v[:, :, :] = Salt_out


    ds_out.attrib["note"] = "U/V destaggered and rotated to true east/north; Theta/Salt from center cell only"


    close(ds_out)


    println("\nSaved combined mooring NetCDF -> $outfile")
    println("Done.")


catch e
    println(logio, "\nERROR: ", sprint(showerror, e, catch_backtrace()))
    flush(logio)
    rethrow()
finally
    flush(logio)
    close(logio)
end




