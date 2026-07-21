using DSP, MAT, Statistics, Printf, LinearAlgebra, TOML
using NCDatasets


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter


config_file = get(ENV, "JULIA_CONFIG",
       joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)


base  = cfg["base_path"]
base2 = cfg["base_path2"]


# --- Full domain grid ---
NX_full, NY_full = 288, 468
minlat, maxlat = 24.0, 31.91
minlon, maxlon = 193.0, 199.0
lat_full = range(minlat, maxlat, length=NY_full)
lon_full = range(minlon, maxlon, length=NX_full)


# --- Tile geometry ---
buf = 3
tx, ty = 47, 66
nx = tx + 2*buf
ny = ty + 2*buf
nz = 88


dto = 144
Tts = 366192
nt  = div(Tts, dto)


thk = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF = Float64.(thk[1:nz])
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)
g = 9.8


# --- Filter (9-15 day band, as in your flux script) ---
T1, T2, delt, N = 9.0, 15.0, 1.0, 4


n_modes_keep = 5


modalamp_dir = joinpath(base, "3day_mean", "ModalAmplitudes")   # already-saved uhat/vhat (5 modes)
sl_dir       = joinpath(base, "3day_mean", "SLmodes")           # Ueig, needed only for pressure projection
out_dir      = joinpath(base, "SM", "MooringModalFlux")
mkpath(out_dir)
out_file = joinpath(out_dir, "MooringModalFlux_Box56_IWAP.nc")
isfile(out_file) && rm(out_file)


# Run with multiple Julia threads: `julia --threads auto script.jl`
BLAS.set_num_threads(1)
println("Julia threads available: ", Threads.nthreads())


# ==========================================================
# TARGET POINTS -- the 4 IWAP moorings found inside Box 56
# ==========================================================
mooring_ids = [82, 83, 84, 85]
target_lats = [25.4891, 27.7690, 28.8995, 30.1312]
target_lons = [194.8451, 196.0301, 196.5105, 197.1154]
iwap_idx    = [1, 2, 3, 4]
n_points = length(target_lats)


i_pt = [argmin(abs.(collect(lon_full) .- target_lons[p])) for p in 1:n_points]
j_pt = [argmin(abs.(collect(lat_full) .- target_lats[p])) for p in 1:n_points]
xn_of_point = [fld(i_pt[p]-1, tx) + 1 for p in 1:n_points]
yn_of_point = [fld(j_pt[p]-1, ty) + 1 for p in 1:n_points]
i_local_of_point = [i_pt[p] - (xn_of_point[p]-1)*tx + buf for p in 1:n_points]
j_local_of_point = [j_pt[p] - (yn_of_point[p]-1)*ty + buf for p in 1:n_points]


println("Point  id   lat       lon        tile        i_local  j_local")
for p in 1:n_points
   println("  $p   $(mooring_ids[p])   $(target_lats[p])   $(target_lons[p])   " *
           "$(xn_of_point[p])x$(yn_of_point[p])   $(i_local_of_point[p])  $(j_local_of_point[p])")
end


tile_groups = Dict{Tuple{Int,Int}, Vector{Int}}()
for p in 1:n_points
   key = (xn_of_point[p], yn_of_point[p])
   haskey(tile_groups, key) ? push!(tile_groups[key], p) : (tile_groups[key] = [p])
end
tile_keys = collect(keys(tile_groups))


lat_out       = copy(target_lats)
lon_out       = copy(target_lons)
tile_out      = fill("", n_points)
uflux_avg_out = fill(NaN, n_points, n_modes_keep)   # depth-avg, time-avg, per mode [W/m^2]
vflux_avg_out = fill(NaN, n_points, n_modes_keep)
uflux_int_out = fill(NaN, n_points, n_modes_keep)   # depth-integrated, time-avg, per mode [kW/m]
vflux_int_out = fill(NaN, n_points, n_modes_keep)


# NetCDF/HDF5 reads are not guaranteed thread-safe -- serialize all file I/O
# behind this lock, while the actual filtering/pressure-integration/matrix
# work (the expensive part) runs fully in parallel across tiles.
io_lock = ReentrantLock()


# ==========================================================
# ONE THREAD PER TILE THAT CONTAINS A MOORING
# (only rho + hFacC + Ueig are read; fu/fv are NOT re-read since
#  uhat/vhat are already saved in ModalAmp_$(suffix).nc)
# ==========================================================
@time Threads.@threads for ti in 1:length(tile_keys)
   xn, yn = tile_keys[ti]
   point_ids = tile_groups[(xn, yn)]
   suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
   println("\n=== [thread $(Threads.threadid())] Tile $suffix contains points: $point_ids ===")


   modalamp_file = joinpath(modalamp_dir, "ModalAmp_$(suffix).nc")
   sl_file       = joinpath(sl_dir, "SLmodes_$(suffix).nc")


   if !isfile(modalamp_file) || !isfile(sl_file)
       @warn "Tile $suffix: ModalAmp or SLmodes file missing -- skipping points $point_ids"
       continue
   end


   local uhat_tile, vhat_tile, Ueig_tile, hFacC, rho
   lock(io_lock) do
       uhat_tile, vhat_tile = NCDataset(modalamp_file, "r") do ds
           Array(ds["uhat"][:,:,:,:]), Array(ds["vhat"][:,:,:,:])   # (x,y,t,mode) -- already projected
       end
       Ueig_tile = NCDataset(sl_file, "r") do ds
           Array(ds["Ueig"][:,:,:,:])
       end
       hFacC = read_bin(joinpath(base, "hFacC", "hFacC_$suffix.bin"), (nx, ny, nz))
       rho = Float64.(open(joinpath(base, "Density", "rho_in_$suffix.bin"), "r") do io
           nbytes = nx * ny * nz * nt * sizeof(Float64)
           raw_bytes = read(io, nbytes)
           raw_data = reinterpret(Float64, raw_bytes)
           reshape(raw_data, nx, ny, nz, nt)
       end)
   end


   # --- Depths / masks (whole tile) ---
   DRFfull = hFacC .* DRF3d
   depth = sum(DRFfull, dims=3)
   DRFfull[hFacC .== 0] .= 0.0
   mask4D = reshape(hFacC .== 0, nx, ny, nz, 1)


   # --- Perturbation pressure only (u,v modal amplitudes already exist) ---
   fr    = bandpassfilter(rho, T1, T2, delt, N, nt)
   pres  = g .* cumsum(fr .* DRFfull, dims=3)
   pfz   = cat(zeros(nx, ny, 1, nt), pres; dims=3)
   pc_3d = 0.5 .* (pfz[:, :, 1:end-1, :] .+ pfz[:, :, 2:end, :])
   pa    = sum(pc_3d .* DRFfull, dims=3) ./ depth
   pp_3d = pc_3d .- pa
   pp_3d[repeat(mask4D, 1, 1, 1, size(pp_3d, 4))] .= 0


   for p in point_ids
       tile_out[p] = suffix
       i_local = i_local_of_point[p]
       j_local = j_local_of_point[p]


       hfac_col  = @view hFacC[i_local, j_local, :]
       ocean_idx = findall(hfac_col .> 0)
       if length(ocean_idx) < 4
           @warn "Point $p (mooring $(mooring_ids[p])): land/too shallow -- skipping"
           continue
       end
       k_top, k_bot = ocean_idx[1], ocean_idx[end]


       Phi_col = @view Ueig_tile[i_local, j_local, k_top:k_bot, :]
       if any(isnan, Phi_col)
           @warn "Point $p (mooring $(mooring_ids[p])): SL modes not solved here -- skipping"
           continue
       end


       dz_col = (hfac_col .* DRF)[k_top:k_bot]
       H = sum(dz_col)


       p_prof = @view pp_3d[i_local, j_local, k_top:k_bot, :]   # (M, nt)
       W      = Phi_col .* dz_col                                # (M, n_modes_keep)
       phat   = (1/H) .* (p_prof' * W)                           # (nt, n_modes_keep)


       # --- already-computed modal velocity amplitudes, no re-projection ---
       uhat = @view uhat_tile[i_local, j_local, :, :]   # (nt, n_modes_keep)
       vhat = @view vhat_tile[i_local, j_local, :, :]


       uflux_modes = uhat .* phat   # (nt, n_modes_keep) -- per-mode flux, NOT summed
       vflux_modes = vhat .* phat


       uflux_avg_modes = vec(mean(uflux_modes, dims=1))   # time-avg per mode, (n_modes_keep,)
       vflux_avg_modes = vec(mean(vflux_modes, dims=1))


       uflux_avg_out[p, :] = uflux_avg_modes
       vflux_avg_out[p, :] = vflux_avg_modes
       uflux_int_out[p, :] = uflux_avg_modes .* (H / 1000)
       vflux_int_out[p, :] = vflux_avg_modes .* (H / 1000)


       println("  Point $p (mooring $(mooring_ids[p])): H=$(round(H,digits=1)) m | " *
               "uflux_avg (per mode)=$(round.(uflux_avg_out[p, :], digits=4)) W/m^2 | " *
               "uflux_int (per mode)=$(round.(uflux_int_out[p, :], digits=4)) kW/m")
   end
end


# ==========================================================
# WRITE OUTPUT NETCDF -- results in ORIGINAL point order
# ==========================================================
ds = NCDataset(out_file, "c")
defDim(ds, "station", n_points)
defDim(ds, "mode", n_modes_keep)


lat_v = defVar(ds, "lat", Float64, ("station",)); lat_v[:] = lat_out
lat_v.attrib["long_name"] = "Mooring latitude"


lon_v = defVar(ds, "lon", Float64, ("station",)); lon_v[:] = lon_out
lon_v.attrib["long_name"] = "Mooring longitude"


tile_v = defVar(ds, "tile", String, ("station",)); tile_v[:] = tile_out
tile_v.attrib["long_name"] = "Tile suffix containing this mooring"


mode_v = defVar(ds, "mode", Int32, ("mode",)); mode_v[:] = collect(1:n_modes_keep)
mode_v.attrib["long_name"] = "Vertical mode number"


ufa_v = defVar(ds, "uflux_depth_avg", Float64, ("station","mode")); ufa_v[:,:] = uflux_avg_out
ufa_v.attrib["long_name"] = "Depth-averaged, time-averaged u-flux, per mode"
ufa_v.attrib["units"] = "W/m^2"


vfa_v = defVar(ds, "vflux_depth_avg", Float64, ("station","mode")); vfa_v[:,:] = vflux_avg_out
vfa_v.attrib["long_name"] = "Depth-averaged, time-averaged v-flux, per mode"
vfa_v.attrib["units"] = "W/m^2"


ufi_v = defVar(ds, "uflux_depth_int", Float64, ("station","mode")); ufi_v[:,:] = uflux_int_out
ufi_v.attrib["long_name"] = "Depth-integrated, time-averaged u-flux, per mode (for comparison to IWAP kW/m)"
ufi_v.attrib["units"] = "kW/m"


vfi_v = defVar(ds, "vflux_depth_int", Float64, ("station","mode")); vfi_v[:,:] = vflux_int_out
vfi_v.attrib["long_name"] = "Depth-integrated, time-averaged v-flux, per mode (for comparison to IWAP kW/m)"
vfi_v.attrib["units"] = "kW/m"


close(ds)
println("\nSaved $out_file")




