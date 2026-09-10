using MAT
using NCDatasets
using Dates
using Printf


# ==================================================================
# CONFIG -- adjust paths / variable folder names here if needed
# ==================================================================
moordir  = "/nobackup/kzhang/llc_4320/regions/Moorings/"
mydir  = "/nobackup/avaliyap/V2/Moorings/"

matfile  = joinpath(moordir, "MooringLocations.mat")
outfile  = joinpath(mydir, "Moorings_88_timeseries.nc")


n_moor = 88   # first 88 stations in MooringLocations.mat -- already verified


# Folder/prefix names for each raw variable, following the VAR/VAR_<nx>.<date>
# pattern confirmed for U and V. Theta/Salt are guessed by the same pattern --
# edit these two entries if your actual folder names differ (the script will
# error out clearly at startup if none of the candidates exist).
var_candidates = Dict(
    "U"     => ["U"],
    "V"     => ["V"],
    "Theta" => ["Theta", "T"],
    "Salt"  => ["Salt", "S"],
)


# ==================================================================
# 1) Load grid / station metadata
# ==================================================================
vars = matread(matfile)
lat     = vec(vars["lat"])[1:n_moor]
lon     = vec(vars["lon"])[1:n_moor]
AngleCS = vec(vars["AngleCS"])
AngleSN = vec(vars["AngleSN"])
RC      = vec(vars["RC"])          # nz-length vector, cell-center depth
nx      = Int(vars["nx"])          # total stations in the raw extraction, e.g. 1996
nz      = Int(vars["nz"])          # vertical levels


println("Loaded MooringLocations.mat: nx=$nx, nz=$nz, using first $n_moor stations.")


# ==================================================================
# 2) Resolve which folder/prefix each variable actually uses, and
#    find the set of timestamps common to ALL four variables
# ==================================================================
function resolve_vardir(moordir, candidates)
    for c in candidates
        d = joinpath(moordir, c)
        if isdir(d)
            return d, c
        end
    end
    error("Could not find a folder for any of $candidates under $moordir " *
          "-- edit var_candidates at the top of this script.")
end


vardirs  = Dict{String, Tuple{String,String}}()   # varname => (dir, prefix)
datesets = Dict{String, Vector{String}}()


for (vname, cands) in var_candidates
    d, prefix = resolve_vardir(moordir, cands)
    vardirs[vname] = (d, prefix)
    fpref = "$(prefix)_$(nx)."
    files = filter(f -> startswith(f, fpref), readdir(d))
    datesets[vname] = [f[(length(fpref)+1):end] for f in files]
    println("  $vname -> $d  ($(length(files)) files found)")
end


dates = sort(collect(intersect(datesets["U"], datesets["V"], datesets["Theta"], datesets["Salt"])))
println("Timestamps common to U, V, Theta, Salt: $(length(dates))")
if isempty(dates)
    error("No common timestamps found across variables -- check var_candidates paths above.")
end


# ==================================================================
# 3) Helpers
# ==================================================================
function read_llc_field(fname, nx, nz)
    raw  = reinterpret(UInt32, read(fname))
    data = reinterpret(Float32, ntoh.(raw))
    return reshape(data, nx, nz)   # (station, level) -- matches MATLAB's readbin(...,[nx nz])
end


function parse_llc_date(dte)
    # dte like "20230524T060000"
    return DateTime(dte, dateformat"yyyymmdd\THHMMSS")
end


# ==================================================================
# 4) Create the NetCDF file and its dimensions/variables up front.
#    "time" is unlimited, so we can append one slice at a time below
#    without holding every timestep in memory at once.
# ==================================================================
ds = NCDataset(outfile, "c")


defDim(ds, "station", n_moor)
defDim(ds, "depth", nz)
defDim(ds, "time", Inf)


v_lat = defVar(ds, "lat", Float64, ("station",))
v_lat.attrib["long_name"] = "latitude"
v_lat.attrib["units"] = "degrees_north"
v_lat[:] = lat


v_lon = defVar(ds, "lon", Float64, ("station",))
v_lon.attrib["long_name"] = "longitude"
v_lon.attrib["units"] = "degrees_east"
v_lon[:] = lon


v_depth = defVar(ds, "depth", Float64, ("depth",))
v_depth.attrib["long_name"] = "cell-center depth (MITgcm RC)"
v_depth.attrib["units"] = "m"
v_depth[:] = RC


v_time = defVar(ds, "time", Float64, ("time",))
v_time.attrib["long_name"] = "time"
v_time.attrib["units"] = "seconds since 1970-01-01T00:00:00"
v_time.attrib["calendar"] = "standard"


v_uE = defVar(ds, "U_east", Float32, ("time", "station", "depth"))
v_uE.attrib["long_name"] = "eastward velocity"
v_uE.attrib["units"] = "m s-1"


v_vN = defVar(ds, "V_north", Float32, ("time", "station", "depth"))
v_vN.attrib["long_name"] = "northward velocity"
v_vN.attrib["units"] = "m s-1"


v_th = defVar(ds, "Theta", Float32, ("time", "station", "depth"))
v_th.attrib["long_name"] = "potential temperature"
v_th.attrib["units"] = "degC"


v_sa = defVar(ds, "Salt", Float32, ("time", "station", "depth"))
v_sa.attrib["long_name"] = "salinity"
v_sa.attrib["units"] = "psu"


# ==================================================================
# 5) Loop over timestamps: read one snapshot of each variable,
#    rotate U/V, and append it as the next time slice.
#    Theta and Salt are scalars -- no rotation needed.
# ==================================================================
for (it, dte) in enumerate(dates)
    Ufile = joinpath(vardirs["U"][1],     "$(vardirs["U"][2])_$(nx).$(dte)")
    Vfile = joinpath(vardirs["V"][1],     "$(vardirs["V"][2])_$(nx).$(dte)")
    Tfile = joinpath(vardirs["Theta"][1], "$(vardirs["Theta"][2])_$(nx).$(dte)")
    Sfile = joinpath(vardirs["Salt"][1],  "$(vardirs["Salt"][2])_$(nx).$(dte)")


    U = read_llc_field(Ufile, nx, nz)
    V = read_llc_field(Vfile, nx, nz)
    T = read_llc_field(Tfile, nx, nz)
    S = read_llc_field(Sfile, nx, nz)


    U_east  = U .* AngleCS .- V .* AngleSN
    V_north = U .* AngleSN .+ V .* AngleCS


    v_time[it]     = datetime2unix(parse_llc_date(dte))
    v_uE[it, :, :] = U_east[1:n_moor, :]
    v_vN[it, :, :] = V_north[1:n_moor, :]
    v_th[it, :, :] = T[1:n_moor, :]
    v_sa[it, :, :] = S[1:n_moor, :]


    if it % 20 == 0 || it == length(dates)
        println("  wrote $it / $(length(dates))  ($dte)")
    end
end


close(ds)
println("\nDone. Wrote $(length(dates)) timesteps for $n_moor stations -> $outfile")




