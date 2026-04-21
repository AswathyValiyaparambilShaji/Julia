using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML, NCDatasets


include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter


config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base  = cfg["base_path"]
base2 = cfg["base_path2"]


# --- Domain & grid ---
NX, NY = 288, 468
minlat, maxlat = 24.0, 31.91
minlon, maxlon = 193.0, 199.0
lat = collect(range(minlat, maxlat, length=NY))
lon = collect(range(minlon, maxlon, length=NX))


# --- Tile & time ---
buf = 3
tx, ty = 47, 66
nx = tx + 2*buf
ny = ty + 2*buf
nz = 88


dt  = 25
dto = 144
Tts = 366192
nt  = div(Tts, dto)


# --- Thickness ---
thk   = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF   = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)


# ============================================================================
# BEAM POINTS
# ============================================================================
beam_lons = [193.3000, 193.6278, 193.9556, 194.2833, 194.6111,
             194.9389, 195.2667, 195.5944, 195.9222, 196.2500]
beam_lats = [ 24.2000,  24.5667,  24.9333,  25.3000,  25.6667,
              26.0333,  26.4000,  26.7667,  27.1333,  27.5000]
N_beam = 10


# ============================================================================
# FIND NEAREST GRID INDICES FOR EACH BEAM POINT
# ============================================================================
beam_ix = [argmin(abs.(lon .- beam_lons[i])) for i in 1:N_beam]
beam_iy = [argmin(abs.(lat .- beam_lats[i])) for i in 1:N_beam]


println("========== BEAM POINT GRID INDICES ==========")
for i in 1:N_beam
    @printf("  Point %2d:  lon=%.4f (ix=%d, grid_lon=%.4f)   lat=%.4f (iy=%d, grid_lat=%.4f)\n",
        i, beam_lons[i], beam_ix[i], lon[beam_ix[i]],
           beam_lats[i], beam_iy[i], lat[beam_iy[i]])
end
println("==============================================\n")


# ============================================================================
# ALLOCATE BEAM ARRAYS ONLY  (N_beam, nz, nt) — no full-domain allocation
# ============================================================================
U_beam   = zeros(Float32, N_beam, nz, nt)
V_beam   = zeros(Float32, N_beam, nz, nt)
rho_beam = zeros(Float32, N_beam, nz, nt)
DRF_beam = zeros(Float32, N_beam, nz)


# ============================================================================
# TILE LOOP — extract beam points tile by tile (avoids full domain in memory)
# ============================================================================
println("Extracting beam points from tiles...")


for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]


        # Global index range of this tile's interior (matching full-domain write logic)
        xs = (xn - 1) * tx + 1;  xe = xs + tx + 2*buf - 1
        ys = (yn - 1) * ty + 1;  ye = ys + ty + 2*buf - 1


        # Check which beam points fall inside this tile's interior
        pts = [i for i in 1:N_beam
               if xs + 1 <= beam_ix[i] <= xe - 1 &&
                  ys + 1 <= beam_iy[i] <= ye - 1]


        isempty(pts) && continue   # skip tile if no beam points inside


        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
        println("  Tile $suffix  → beam points: $pts")


        # --- Read U, V (raw C-grid) ---
        U_raw = read_bin(joinpath(base, "U/U_$suffix.bin"), (nx, ny, nz, nt))
        V_raw = read_bin(joinpath(base, "V/V_$suffix.bin"), (nx, ny, nz, nt))


        # C-grid to cell centres
        uc = 0.5 .* (U_raw[1:end-1, :, :, :] .+ U_raw[2:end,   :, :, :])
        vc = 0.5 .* (V_raw[:, 1:end-1, :, :] .+ V_raw[:, 2:end, :, :])


        ucc = cat(uc, zeros(Float32, 1, ny, nz, nt); dims=1)
        vcc = cat(vc, zeros(Float32, nx, 1, nz, nt); dims=2)


        U_raw = nothing; V_raw = nothing; GC.gc()


        # --- Read rho ---
        rho = open(joinpath(base, "Density", "rho_in_$suffix.bin"), "r") do io
            arr = Array{Float64}(undef, nx, ny, nz, nt)
            read!(io, arr)
            arr
        end




        # --- hFacC & DRFfull ---
        hFacC   = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
        DRFfull = hFacC .* DRF3d    # (nx, ny, nz)
        for t in 1:nt
            rho1 = rho[:,:,:,t]
            rho1[hFacC .== 0] .= NaN
            rho[:,:,:,t] = rho1
        end

         


        # --- Extract each beam point from this tile ---
        # Local index: global ix → local lx = ix - xs + 1
        # (global xs+1 ↔ local xsf=2, consistent with reference slicing)
        for i in pts
            lx = beam_ix[i] - xs + 1
            ly = beam_iy[i] - ys + 1
            U_beam[i, :, :]   .= Float32.(ucc[lx, ly, :, :])
            V_beam[i, :, :]   .= Float32.(vcc[lx, ly, :, :])
            rho_beam[i, :, :] .= Float32.(rho[lx, ly, :, :])
            DRF_beam[i, :]    .= Float32.(DRFfull[lx, ly, :])
        end


        ucc = nothing; vcc = nothing; rho = nothing; hFacC = nothing; DRFfull = nothing
        GC.gc()


    end
end


println("Beam point extraction complete.\n")


# ============================================================================
# SAVE TO NetCDF
# ============================================================================
ncfile = joinpath(base2, "beam_UVrho.nc")
println("Writing NetCDF: $ncfile ...")


NCDatasets.Dataset(ncfile, "c") do ds


    defDim(ds, "beam_point", N_beam)
    defDim(ds, "nz",         nz)      # depth index only — no coordinate values
    defDim(ds, "nt",         nt)      # time index only  — no coordinate values


    defVar(ds, "longitude", beam_lons, ("beam_point",),
        attrib = Dict("units" => "degrees_east", "long_name" => "Beam point longitude"))


    defVar(ds, "latitude", beam_lats, ("beam_point",),
        attrib = Dict("units" => "degrees_north", "long_name" => "Beam point latitude"))


    defVar(ds, "U", U_beam, ("beam_point", "nz", "nt"),
        attrib = Dict("units" => "m/s", "long_name" => "Cell-centred zonal velocity"))


    defVar(ds, "V", V_beam, ("beam_point", "nz", "nt"),
        attrib = Dict("units" => "m/s", "long_name" => "Cell-centred meridional velocity"))


    defVar(ds, "rho_insitu", rho_beam, ("beam_point", "nz", "nt"),
        attrib = Dict("units" => "kg/m3", "long_name" => "In-situ density"))


    defVar(ds, "DRFfull", DRF_beam, ("beam_point", "nz"),
        attrib = Dict("units" => "m", "long_name" => "hFacC-weighted layer thickness at beam points"))


    ds.attrib["title"]      = "MITgcm beam extraction: U, V, rho_insitu, DRFfull"
    ds.attrib["beam_start"] = "lon=193.3000, lat=24.2000"
    ds.attrib["beam_end"]   = "lon=196.2500, lat=27.5000"
    ds.attrib["n_points"]   = N_beam
    ds.attrib["created_by"] = "beam_extract_nc.jl"


end


println("Done. NetCDF saved: $ncfile")




