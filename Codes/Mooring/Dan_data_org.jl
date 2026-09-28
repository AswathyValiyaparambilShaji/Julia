using NCDatasets, Statistics


# Rotate U,V to east/north at all 3x3 points, then average to the centre cell:
#   U_east  along x (cellcol, dim 2)
#   V_north along y (cellrow, dim 1)
# Arrays in Julia are (cellrow, cellcol, k, time).


sites_dir    = "/nobackupp27/dbwhitt/llc_4320/OUT/regions/moorings/sites/"
outfile      = "/nobackup/avaliyap/V2/Moorings/Moorings_88_combined_v2n.nc"
logfile      = "/nobackup/avaliyap/V2/Moorings/build_mooring_netcdf_v2_logn.txt"
wanted_sites = 1:88


mkpath(dirname(outfile))


function load_site(f)
    ds = NCDataset(f, "r")
    CS, SN = ds.attrib["AngleCS"], ds.attrib["AngleSN"]
    s = (site_id = ds.attrib["site"], lat = ds.attrib["lat"], lon = ds.attrib["lon"],
         face = ds.attrib["face"], CS = CS, SN = SN, depth = ds.attrib["Depth"],
         RC = Array(ds["RC"]), RF = Array(ds["RF"]), DRF = Array(ds["DRF"]),
         wet = Array(ds["wet"]), hFacC = Float64.(Array(ds["hFacC"])),
         time = Array(ds["time"]), have = Array(ds["have"]))
    U     = Float64.(Array(ds["U"]))
    V     = Float64.(Array(ds["V"]))
    Theta = Float64.(Array(ds["Theta"]))
    Salt  = Float64.(Array(ds["Salt"]))
    close(ds)


    # 1) rotate at every point
    Ue = U .* CS .- V .* SN
    Vn = U .* SN .+ V .* CS


    # 2) average to the centre cell (2,2)
    U_east  = 0.5 .* (Ue[2, 2, :, :] .+ Ue[2, 3, :, :])   # along x (cellcol)
    V_north = 0.5 .* (Vn[2, 2, :, :] .+ Vn[3, 2, :, :])   # along y (cellrow)
    T  = Theta[2, 2, :, :]
    Sa = Salt[2, 2, :, :]


    # missing snapshots and dry levels -> NaN
    for A in (U_east, V_north, T, Sa)
        A[:, s.have .== 0] .= NaN
        A[s.wet .== 0, :]  .= NaN
    end
    return merge(s, (U_east = U_east, V_north = V_north, Theta = T, Salt = Sa))
end


site_number(f) = parse(Int, match(r"site_(\d+)\.nc$", f).captures[1])
files = filter(f -> occursin(r"^site_\d+\.nc$", f), readdir(sites_dir))
files = sort(filter(f -> site_number(f) in wanted_sites, files), by = site_number)
N = length(files)
N == length(wanted_sites) || error("Expected $(length(wanted_sites)) sites, found $N")


logio = open(logfile, "w")


s1 = load_site(joinpath(sites_dir, files[1]))
nz, nt = length(s1.RC), length(s1.time)


lon = zeros(N); lat = zeros(N); CS = zeros(N); SN = zeros(N); depth = zeros(N)
face = zeros(Int32, N); site_id = zeros(Int32, N)
wet = zeros(Int32, N, nz); hFacC = zeros(N, nz); have = zeros(Int32, nt, N)
U_east = fill(NaN32, nt, N, nz); V_north = fill(NaN32, nt, N, nz)
Theta  = fill(NaN32, nt, N, nz); Salt    = fill(NaN32, nt, N, nz)


for (p, f) in enumerate(files)
    s = load_site(joinpath(sites_dir, f))
    length(s.time) == nt || (println(logio, "skip $f: time length $(length(s.time))"); continue)


    lon[p], lat[p], CS[p], SN[p], depth[p] = s.lon, s.lat, s.CS, s.SN, s.depth
    face[p], site_id[p] = s.face, s.site_id
    wet[p, :], hFacC[p, :], have[:, p] = s.wet, s.hFacC, s.have


    # (k,t) -> (t,k)
    U_east[:, p, :]  = Float32.(s.U_east')
    V_north[:, p, :] = Float32.(s.V_north')
    Theta[:, p, :]   = Float32.(s.Theta')
    Salt[:, p, :]    = Float32.(s.Salt')


    println(logio, "site p/N f:face=(s.face) lat=(s.lat)lon=(s.lon) CS=(s.CS)SN=(s.SN)")
    flush(logio)
end


isfile(outfile) && rm(outfile)
NCDataset(outfile, "c") do ds
    defDim(ds, "time", nt); defDim(ds, "station", N)
    defDim(ds, "depth", nz); defDim(ds, "depth_p1", nz + 1)


    defVar(ds, "time", s1.time, ("time",), attrib = ["units" => "hours since 2023-01-01 00:00:00"])
    defVar(ds, "lon", lon, ("station",), attrib = ["units" => "degrees_east"])
    defVar(ds, "lat", lat, ("station",), attrib = ["units" => "degrees_north"])
    defVar(ds, "AngleCS", CS, ("station",))
    defVar(ds, "AngleSN", SN, ("station",))
    defVar(ds, "face", face, ("station",))
    defVar(ds, "site_id", site_id, ("station",))
    defVar(ds, "bottom_depth", depth, ("station",), attrib = ["units" => "m"])
    defVar(ds, "RC", s1.RC, ("depth",), attrib = ["units" => "m"])
    defVar(ds, "DRF", s1.DRF, ("depth",), attrib = ["units" => "m"])
    defVar(ds, "RF", s1.RF, ("depth_p1",), attrib = ["units" => "m"])
    defVar(ds, "wet", wet, ("station", "depth"))
    defVar(ds, "hFacC", hFacC, ("station", "depth"))
    defVar(ds, "have", have, ("time", "station"))
    defVar(ds, "U_east", U_east, ("time", "station", "depth"), attrib = ["units" => "m/s"])
    defVar(ds, "V_north", V_north, ("time", "station", "depth"), attrib = ["units" => "m/s"])
    defVar(ds, "Theta", Theta, ("time", "station", "depth"))
    defVar(ds, "Salt", Salt, ("time", "station", "depth"))
    ds.attrib["note"] = "U,V rotated to east/north at all 3x3 points, then U_east averaged " *
                        "along cellcol and V_north along cellrow to the centre cell"
end


println(logio, "Saved -> $outfile")
close(logio)




