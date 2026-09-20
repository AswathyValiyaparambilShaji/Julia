using NCDatasets


# ============================================================================
# Usage:
#   julia inspect_mooring_nc.jl /path/to/site_0001.nc
#   julia inspect_mooring_nc.jl /path/to/site_0001.nc /path/to/output_info.txt
#
# If the second argument (output txt path) is omitted, it defaults to the
# input filename with "_info.txt" appended, written in the current directory.
# ============================================================================


ncfile  = length(ARGS) >= 1 ? ARGS[1] : "/nobackupp27/dbwhitt/llc_4320/OUT/regions/moorings/sites/site_0001.nc"
outfile = length(ARGS) >= 2 ? ARGS[2] : splitext(basename(ncfile))[1] * "_info.txt"


function report(io, ds, label, candidates)
    println(io, "\n$label:")
    found_any = false
    for c in candidates
        if haskey(ds, c)
            v = ds[c]
            println(io, "  FOUND '$c'  size=$(size(v))  dims=$(dimnames(v))")
            found_any = true
        end
    end
    if !found_any
        println(io, "  none of $candidates present")
    end
end


function inspect_nc(ncfile::AbstractString, io::IO)
    println(io, "="^78)
    println(io, "FILE: $ncfile")
    println(io, "="^78)


    ds = NCDataset(ncfile, "r")


    # 1) NCDatasets' own structural summary
    println(io, "\n--- NCDatasets SUMMARY ---")
    show(io, ds)
    println(io)


    # 2) Global attributes
    println(io, "\n--- GLOBAL ATTRIBUTES ---")
    for (k, v) in ds.attrib
        println(io, "  $k = $v")
    end


    # 3) Dimensions
    println(io, "\n--- DIMENSIONS ---")
    for (name, len) in ds.dim
        println(io, "  $name => $len")
    end


    # 4) Every variable: dims, size, dtype, attributes
    println(io, "\n--- VARIABLES ---")
    for vname in keys(ds)
        v = ds[vname]
        println(io, "\n  $vname")
        println(io, "    dims  : ", dimnames(v))
        println(io, "    size  : ", size(v))
        println(io, "    eltype: ", eltype(v))
        if !isempty(v.attrib)
            println(io, "    attributes:")
            for (ak, av) in v.attrib
                println(io, "      $ak = $av")
            end
        end
    end


    # 5) Targeted checks
    println(io, "\n" * "="^78)
    println(io, "TARGETED CHECKS (for updating the extraction/diagnostic script)")
    println(io, "="^78)


    report(io, ds, "Velocity components", ["U", "V", "U_east", "V_north", "UVEL", "VVEL", "u", "v"])
    report(io, ds, "Rotation angles (model-relative U/V -> true east/north)",
           ["AngleCS", "AngleSN", "angleCS", "angleSN", "CS", "SN"])
    report(io, ds, "Horizontal coordinates", ["lon", "lat", "longitude", "latitude", "XC", "YC"])
    report(io, ds, "Vertical thickness / mask", ["hFacC", "DRF", "DRFfull", "Depth", "drF"])
    report(io, ds, "Salinity / temperature", ["Salt", "Theta", "S", "T", "salinity", "theta"])


    println(io, "\nDimensions of length 3 (possible 3x3 horizontal-neighbor axes):")
    found3 = false
    for (name, len) in ds.dim
        if len == 3
            println(io, "  $name = 3")
            found3 = true
        end
    end
    if !found3
        println(io, "  none found")
    end


    # 6) Tiny data peek
    println(io, "\n--- SAMPLE VALUES ---")
    for c in ["U", "U_east", "UVEL", "u"]
        if haskey(ds, c)
            v = ds[c]
            idx = ntuple(_ -> 1, ndims(v))
            try
                println(io, "  $c$(idx) = ", v[idx...])
            catch e
                println(io, "  could not sample $c: ", e)
            end
            break
        end
    end


    close(ds)
    println(io, "\nDone.")
end


open(outfile, "w") do io
    try
        inspect_nc(ncfile, io)
    catch e
        println(io, "\n==================== ERROR ====================")
        println(io, sprint(showerror, e, catch_backtrace()))
        flush(io)
        rethrow()
    end
end


println("Saved diagnostic output to: ", outfile)





