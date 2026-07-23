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
# MOORING POINTS
# ============================================================================
target_lons = [194.8451, 196.0301, 196.5105, 197.1154]
target_lats = [25.4891, 27.7690, 28.8995, 30.1312]
N_moor = 4




# ============================================================================
# FIND NEAREST GRID INDICES FOR EACH MOORING POINT
# ============================================================================
moor_ix = [argmin(abs.(lon .- target_lons[i])) for i in 1:N_moor]
moor_iy = [argmin(abs.(lat .- target_lats[i])) for i in 1:N_moor]




println("========== MOORING POINT GRID INDICES ==========")
for i in 1:N_moor
   @printf("  Point %2d:  lon=%.4f (ix=%d, grid_lon=%.4f)   lat=%.4f (iy=%d, grid_lat=%.4f)\n",
   	i, target_lons[i], moor_ix[i], lon[moor_ix[i]],
      	target_lats[i], moor_iy[i], lat[moor_iy[i]])
end
println("==================================================\n")




# ============================================================================
# ALLOCATE MOORING ARRAYS ONLY  (N_moor, nz, nt)  no full-domain allocation
# ============================================================================
U_moor   = zeros(Float32, N_moor, nz, nt)
V_moor   = zeros(Float32, N_moor, nz, nt)
rho_moor = zeros(Float32, N_moor, nz, nt)
DRF_moor = zeros(Float32, N_moor, nz)
Theta_moor   = zeros(Float32, N_moor, nz, nt)
Salt_moor = zeros(Float32, N_moor, nz, nt)


# ============================================================================
# TILE LOOP  extract mooring points tile by tile (avoids full domain in memory)
# ============================================================================
println("Extracting mooring points from tiles...")




for xn in cfg["xn_start"]:cfg["xn_end"]
   for yn in cfg["yn_start"]:cfg["yn_end"]


   	# Global index range of this tile's interior (matching full-domain write logic)
   	xs = (xn - 1) * tx + 1;  xe = xs + tx + 2*buf - 1
   	ys = (yn - 1) * ty + 1;  ye = ys + ty + 2*buf - 1


   	# Check which mooring points fall inside this tile's interior
   	pts = [i for i in 1:N_moor
          	if xs + 1 <= moor_ix[i] <= xe - 1 &&
             	ys + 1 <= moor_iy[i] <= ye - 1]


   	isempty(pts) && continue   # skip tile if no mooring points inside


   	suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
   	println("  Tile $suffix  → mooring points: $pts")


   	# --- Read U, V (raw C-grid) ---
   	U_raw = read_bin(joinpath(base, "U/U_$suffix.bin"), (nx, ny, nz, nt))
   	V_raw = read_bin(joinpath(base, "V/V_$suffix.bin"), (nx, ny, nz, nt))
   	# --- Read T, S (raw C-grid) ---
   	Salt 	= read_bin(joinpath(base, "Salt/Salt_$suffix.bin"),   (nx, ny, nz, nt))
   	Theta 	= read_bin(joinpath(base, "Theta/Theta_$suffix.bin"),   (nx, ny, nz, nt))


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
   	DRFfull = hFacC .* DRF3d	# (nx, ny, nz)


   	for t in 1:nt    	# ← correct
       	rho1 = rho[:, :, :, t]
       	rho1[hFacC .== 0] .= NaN
       	rho[:, :, :, t] = rho1
   	end


   	# --- Extract each mooring point from this tile ---
   	# Local index: global ix → local lx = ix - xs + 1
   	# (global xs+1 ↔ local xsf=2, consistent with reference slicing)
   	for i in pts
       	lx = moor_ix[i] - xs + 1
       	ly = moor_iy[i] - ys + 1
       	println(lx)
       	println(ly)
       	U_moor[i, :, :]   .= Float32.(ucc[lx, ly, :, :])
       	V_moor[i, :, :]   .= Float32.(vcc[lx, ly, :, :])
       	rho_moor[i, :, :] .= Float32.(rho[lx, ly, :, :])
       	DRF_moor[i, :]	.= Float32.(DRFfull[lx, ly, :])
       	Salt_moor[i, :,:]	.= Float32.(Salt[lx, ly, :,:])
       	Theta_moor[i, :,:]   .= Float32.(Theta[lx, ly, :,:])
   	end


   	ucc = nothing; vcc = nothing; rho = nothing; hFacC = nothing; DRFfull = nothing
   	GC.gc()


   end
end




println("Mooring point extraction complete.\n")




# ============================================================================
# SAVE TO NetCDF
# ============================================================================
ncfile = joinpath(base2, "mooring_UVrho.nc")
println("Writing NetCDF: $ncfile ...")




NCDatasets.Dataset(ncfile, "c") do ds


   defDim(ds, "mooring_point", N_moor)
   defDim(ds, "nz",     	nz)  	# depth index only  no coordinate values
   defDim(ds, "nt",     	nt)  	# time index only   no coordinate values


   defVar(ds, "longitude", target_lons, ("mooring_point",),
   	attrib = Dict("units" => "degrees_east", "long_name" => "Mooring point longitude"))


   defVar(ds, "latitude", target_lats, ("mooring_point",),
   	attrib = Dict("units" => "degrees_north", "long_name" => "Mooring point latitude"))


   defVar(ds, "U", U_moor, ("mooring_point", "nz", "nt"),
   	attrib = Dict("units" => "m/s", "long_name" => "Cell-centred zonal velocity"))


   defVar(ds, "V", V_moor, ("mooring_point", "nz", "nt"),
   	attrib = Dict("units" => "m/s", "long_name" => "Cell-centred meridional velocity"))


   defVar(ds, "rho_insitu", rho_moor, ("mooring_point", "nz", "nt"),
   	attrib = Dict("units" => "kg/m3", "long_name" => "In-situ density"))


   defVar(ds, "DRFfull", DRF_moor, ("mooring_point", "nz"),
   	attrib = Dict("units" => "m", "long_name" => "hFacC-weighted layer thickness at mooring points"))


   defVar(ds, "DRF", Float32.(DRF), ("nz",),
   	attrib = Dict("units" => "m", "long_name" => "Nominal layer thickness (1D, hFacC-independent)"))


   defVar(ds, "Theta", Float32.(Theta_moor), ("mooring_point","nz","nt"))


   defVar(ds, "Salt", Float32.(Salt_moor), ("mooring_point","nz","nt"))


end




println("Done. NetCDF saved: $ncfile")




