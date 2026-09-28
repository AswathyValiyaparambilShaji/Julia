using NCDatasets, Statistics, Printf


# Combines all site_00NN.nc files into one NetCDF shaped like the old
# Moorings_88_timeseries.nc.
#
# Order of operations per site (this order matters):
#   1. Destagger: move U and V from their C-grid faces to the centre of the
#      middle cell of the 3x3 block (mean of the two faces of that cell).
#   2. Rotate: model-native (U,V) -> true (east,north) using AngleCS/AngleSN.
#      Rotation must come AFTER destaggering, because U and V live at
#      different points until they are both averaged to the cell centre,
#      and AngleCS/AngleSN are defined at the cell centre.
#   3. Theta/Salt: taken at the centre cell (already cell-centred, no averaging).
#   4. Missing snapshots (have==0) and dry levels (wet==0) -> NaN.
#
# Index layout in Julia (from the file): U[cellrow, cellcol, k, time]
#   cellcol = model i  (i-1, i, i+1)  -> U is staggered along cellcol
#   cellrow = model j  (j-1, j, j+1)  -> V is staggered along cellrow
# This holds on all faces (on faces 4,5 only the geographic direction of i/j
# differs, which the rotation handles).
# MITgcm C-grid: U(i,j) is on the WEST face of cell (i,j), V(i,j) on the SOUTH face.
#   centre cell = index 2, so:
#   U_centre = 0.5*(U[j=2, i=2] + U[j=2, i=3])   west + east face
#   V_centre = 0.5*(V[j=2, i=2] + V[j=3, i=2])   south + north face


# ---- EDIT THESE ----
sites_dir = "/nobackupp27/dbwhitt/llc_4320/OUT/regions/moorings/sites/"
outfile   = "/nobackup/avaliyap/V2/Moorings/Moorings_88_combined_v2n.nc"
logfile   = "/nobackup/avaliyap/V2/Moorings/build_mooring_netcdf_v2_logn.txt"
wanted_sites = 1:88   # site numbers to keep (e.g. [3, 7, 12, ...] for a custom list)
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
    edge_clamped = ds.attrib["edge_clamped"]


    RC    = Array(ds["RC"])
    RF    = Array(ds["RF"])
    DRF   = Array(ds["DRF"])
    wet   = Array(ds["wet"])
    hFacC = Float64.(Array(ds["hFacC"]))
    time  = Array(ds["time"])
    have  = Array(ds["have"])


    # dims: (cellrow=j, cellcol=i, k, time)
    U_raw     = Float64.(Array(ds["U"]))
    V_raw     = Float64.(Array(ds["V"]))
    Theta_raw = Float64.(Array(ds["Theta"]))
    Salt_raw  = Float64.(Array(ds["Salt"]))
    close(ds)


    jc, ic = 2, 2   # centre cell: cellrow (j) = 2, cellcol (i) = 2


    # 1) destagger to the centre of cell (jc, ic)   -> (k, time)
    U_center = 0.5 .* (U_raw[jc, ic, :, :] .+ U_raw[jc, ic+1, :, :])   # west + east face (along i)
    V_center = 0.5 .* (V_raw[jc, ic, :, :] .+ V_raw[jc+1, ic, :, :])   # south + north face (along j)


    # 2) rotate model-native (U,V) -> true (east,north), now that both are collocated
    U_east  = U_center .* AngleCS .- V_center .* AngleSN
    V_north = U_center .* AngleSN .+ V_center .* AngleCS


    # 3) tracers at the centre cell
    Theta_center = Theta_raw[jc, ic, :, :]
    Salt_center  = Salt_raw[jc, ic, :, :]


    # 4) mask missing snapshots and dry levels
    missing_t = have .== 0
    dry_k     = wet  .== 0
    for A in (U_east, V_north, Theta_center, Salt_center)
        A[:, missing_t] .= NaN
        A[dry_k, :]     .= NaN
    end


    return (; site_id, lat, lon, face, AngleCS, AngleSN, depth_bottom, edge_clamped,
            RC, RF, DRF, wet, hFacC, time, have,
            U_east, V_north, Theta = Theta_center, Salt = Salt_center)
end


function site_number(fname)
    m = match(r"site_(\d+)\.nc$", fname)
    return m === nothing ? typemax(Int) : parse(Int, m.captures[1])
end


all_files = filter(f -> occursin(r"^site_\d+\.nc$", f), readdir(sites_dir))
sort!(all_files, by = site_number)
files  = filter(f -> site_number(f) in wanted_sites, all_files)
N_moor = length(files)


logio = open(logfile, "w")
redirect_stdout(logio)
redirect_stderr(logio)


try
    println("Found $(length(all_files)) site files in $sites_dir; keeping $N_moor (wanted $(length(wanted_sites)))")
    N_moor == 0 && error("No site_*.nc files found in $sites_dir")
    N_moor == length(wanted_sites) ||
        error("Expected $(length(wanted_sites)) site files, found $N_moor -- missing: " *
              string(setdiff(collect(wanted_sites), site_number.(files))))


    first_site = load_mooring_site(joinpath(sites_dir, files[1]))
    nz   = length(first_site.RC)
    nzp1 = length(first_site.RF)
    nt   = length(first_site.time)
    DRF  = first_site.DRF
    RC   = first_site.RC
    RF   = first_site.RF
    time = first_site.time
    first_site = nothing
    println("nz=nz,nz+1=nzp1, nt=$nt (from $(files[1]))")


    lon_out     = fill(NaN, N_moor)
    lat_out     = fill(NaN, N_moor)
    AngleCS_out = fill(NaN, N_moor)
    AngleSN_out = fill(NaN, N_moor)
    face_out    = fill(Int32(0), N_moor)
    site_id_out = fill(Int32(0), N_moor)
    edge_out    = fill(Int32(0), N_moor)
    depth_out   = fill(NaN, N_moor)
    wet_out     = zeros(Int32, N_moor, nz)
    hFacC_out   = fill(NaN, N_moor, nz)
    have_out    = zeros(Int32, nt, N_moor)
    U_east_out  = fill(NaN32, nt, N_moor, nz)
    V_north_out = fill(NaN32, nt, N_moor, nz)
    Theta_out   = fill(NaN32, nt, N_moor, nz)
    Salt_out    = fill(NaN32, nt, N_moor, nz)


    for (p, fname) in enumerate(files)
        site = load_mooring_site(joinpath(sites_dir, fname))


        if length(site.time) != nt
            println("  WARNING site p(fname): time length $(length(site.time)) != $nt -- skipping")
            continue
        end
        site.time == time || println("  WARNING site p(fname): time axis differs from site 1")
        isapprox(site.DRF, DRF; rtol=1e-6) || println("  WARNING site p(fname): DRF differs from site 1")
        site.edge_clamped != 0 &&
            println("  WARNING site p(fname): edge_clamped=$(site.edge_clamped) -- 3x3 block may not be centred on the mooring")


        lon_out[p]     = site.lon
        lat_out[p]     = site.lat
        AngleCS_out[p] = site.AngleCS
        AngleSN_out[p] = site.AngleSN
        face_out[p]    = site.face
        site_id_out[p] = site.site_id
        edge_out[p]    = site.edge_clamped
        depth_out[p]   = site.depth_bottom
        wet_out[p, :]  = site.wet
        hFacC_out[p, :] = site.hFacC
        have_out[:, p] = site.have


        # (k,t) -> (t,k) for storage as (time, station, depth)
        U_east_out[:, p, :]  = Float32.(permutedims(site.U_east,  (2, 1)))
        V_north_out[:, p, :] = Float32.(permutedims(site.V_north, (2, 1)))
        Theta_out[:, p, :]   = Float32.(permutedims(site.Theta,   (2, 1)))
        Salt_out[:, p, :]    = Float32.(permutedims(site.Salt,    (2, 1)))


        nmis = count(==(0), site.have)
        nmis_wet_hfac = count((site.wet .!= 0) .!= (site.hFacC .> 0))
        println("  site p/N_moor done (fname):lat=(site.lat), lon=(site.lon),face=(site.face), " *
                "nwet=(sum(site.wet)),missingsnapshots=nmis, wet/hFacC mismatched levels=$nmis_wet_hfac")
        flush(logio)
    end


    isfile(outfile) && rm(outfile)
    ds_out = NCDataset(outfile, "c")
    defDim(ds_out, "time", nt)
    defDim(ds_out, "station", N_moor)
    defDim(ds_out, "depth", nz)
    defDim(ds_out, "depth_p1", nzp1)


    defVar(ds_out, "time", time, ("time",),
           attrib = ["units" => "hours since 2023-01-01 00:00:00"])
    v = defVar(ds_out, "lon", Float64, ("station",)); v.attrib["units"] = "degrees_east"; v[:] = lon_out
    v = defVar(ds_out, "lat", Float64, ("station",)); v.attrib["units"] = "degrees_north"; v[:] = lat_out
    v = defVar(ds_out, "AngleCS", Float64, ("station",)); v[:] = AngleCS_out
    v = defVar(ds_out, "AngleSN", Float64, ("station",)); v[:] = AngleSN_out
    v = defVar(ds_out, "face", Int32, ("station",)); v[:] = face_out
    v = defVar(ds_out, "site_id", Int32, ("station",)); v[:] = site_id_out
    v = defVar(ds_out, "edge_clamped", Int32, ("station",)); v[:] = edge_out
    v = defVar(ds_out, "bottom_depth", Float64, ("station",)); v.attrib["units"] = "m"; v[:] = depth_out
    v = defVar(ds_out, "DRF", Float64, ("depth",)); v.attrib["units"] = "m"; v[:] = DRF
    v = defVar(ds_out, "RC", Float64, ("depth",));  v.attrib["units"] = "m"; v[:] = RC
    v = defVar(ds_out, "RF", Float64, ("depth_p1",)); v.attrib["units"] = "m"; v[:] = RF


    v = defVar(ds_out, "wet", Int32, ("station", "depth"))
    v.attrib["note"] = "1 where centre Theta nonzero in any snapshot; use this as the mask"
    v[:, :] = wet_out
    v = defVar(ds_out, "hFacC", Float64, ("station", "depth"))
    v.attrib["note"] = "grid hFacC at centre cell (fractional); valid from ~Apr 2023 only; use 'wet' as mask"
    v[:, :] = hFacC_out
    v = defVar(ds_out, "have", Int32, ("time", "station"))
    v.attrib["note"] = "1 if the snapshot was extracted, 0 if missing (data set to NaN)"
    v[:, :] = have_out


    v = defVar(ds_out, "U_east", Float32, ("time", "station", "depth")); v.attrib["units"] = "m/s"; v[:, :, :] = U_east_out
    v = defVar(ds_out, "V_north", Float32, ("time", "station", "depth")); v.attrib["units"] = "m/s"; v[:, :, :] = V_north_out
    v = defVar(ds_out, "Theta", Float32, ("time", "station", "depth")); v[:, :, :] = Theta_out
    v = defVar(ds_out, "Salt", Float32, ("time", "station", "depth")); v[:, :, :] = Salt_out


    ds_out.attrib["note"] = "U/V destaggered to the centre cell (mean of its two C-grid faces), " *
                            "then rotated to true east/north with AngleCS/AngleSN; " *
                            "Theta/Salt from centre cell; missing snapshots and dry levels = NaN"
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




