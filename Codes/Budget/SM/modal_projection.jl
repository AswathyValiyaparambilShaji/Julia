using DSP, MAT, Statistics, Printf, LinearAlgebra, TOML
using NCDatasets


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter


config_file = get(ENV, "JULIA_CONFIG",
       joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)


base  = cfg["base_path"]
base2 = cfg["base_path2"]


# --- Grid ---
buf = 3
tx, ty = 47, 66
nx = tx + 2*buf
ny = ty + 2*buf
nz = 88


dto = 144
Tts = 366192
nt  = div(Tts, dto)


# --- Thickness ---
thk = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF = Float64.(thk[1:nz])
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)
g = 9.8


# --- Filter (9-15 day band, as in your flux script) ---
T1, T2, delt, N = 9.0, 15.0, 1.0, 4
fcutlow, fcuthigh = 1 / T2, 1 / T1
fnq = 1 / delt
bpf = digitalfilter(Bandpass(fcutlow, fcuthigh), Butterworth(N); fs = fnq)


n_modes_keep = 5


sl_dir  = joinpath(base, "3day_mean", "SLmodes")
out_dir = joinpath(base, "3day_mean", "ModalAmplitudes")
mkpath(out_dir)


# Run with multiple Julia threads for this to help: `julia --threads auto script.jl`
# (or set the JULIA_NUM_THREADS environment variable before launching).
# We parallelize across grid columns below with Threads.@threads, so we keep
# BLAS itself single-threaded per call to avoid oversubscribing CPU cores.
BLAS.set_num_threads(1)
println("Julia threads available: ", Threads.nthreads())


@time begin
# ==========================================================
# LOOP OVER TILES
# ==========================================================
for xn in cfg["xn_start"]:cfg["xn_end"]
   for yn in cfg["yn_start"]:cfg["yn_end"]


       suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
       println("\n=== Projecting tile $suffix ===")


       # --- read the saved SL mode shapes for this tile ---
       sl_file = joinpath(sl_dir, "SLmodes_$(suffix).nc")
       Ueig_tile = NCDataset(sl_file, "r") do ds
           Array(ds["Ueig"][:,:,:,:])
       end
       Weig_tile = NCDataset(sl_file, "r") do ds
           Array(ds["Weig"][:,:,:,:])
       end


       # --- read baroclinic fields and rebuild u', v' (same as your flux script) ---
       hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
       #=rho = Float64.(open(joinpath(base,"Density", "rho_in_$suffix.bin"), "r") do io
           nbytes = nx * ny * nz * nt * sizeof(Float64)
           raw_bytes = read(io, nbytes)
           raw_data = reinterpret(Float64, raw_bytes)
           reshape(raw_data, nx, ny, nz, nt)
       end)=#
       fu = Float64.(open(joinpath(base2, "UVW_F", "fu_$suffix.bin"), "r") do io
           nbytes = nx * ny * nz * nt * sizeof(Float32)
           raw_bytes = read(io, nbytes)
           raw_data = reinterpret(Float32, raw_bytes)
           reshape(raw_data, nx, ny, nz, nt)
       end)
       fv = Float64.(open(joinpath(base2, "UVW_F", "fv_$suffix.bin"), "r") do io
           nbytes = nx * ny * nz * nt * sizeof(Float32)
           raw_bytes = read(io, nbytes)
           raw_data = reinterpret(Float32, raw_bytes)
           reshape(raw_data, nx, ny, nz, nt)
       end)
       fw = Float64.(open(joinpath(base2, "UVW_F", "fw_$suffix.bin"), "r") do io
           nbytes = nx * ny * nz * nt * sizeof(Float32)
           raw_bytes = read(io, nbytes)
           raw_data = reinterpret(Float32, raw_bytes)
           reshape(raw_data, nx, ny, nz, nt)
       end)


       DRFfull = hFacC .* DRF3d
       depth = sum(DRFfull, dims=3)
       DRFfull[hFacC .== 0] .= 0.0
       #=
       fr = bandpassfilter(rho, T1, T2, delt, N, nt)
=#
       mask4D = reshape(hFacC .== 0, nx, ny, nz, 1)

       ucA_3d = sum(fu .* DRFfull, dims=3) ./ depth
       up_3d  = fu .- ucA_3d
       up_3d[repeat(mask4D, 1, 1, 1, size(up_3d, 4))] .= 0


       vcA_3d = sum(fv .* DRFfull, dims=3) ./ depth
       vp_3d  = fv .- vcA_3d
       vp_3d[repeat(mask4D, 1, 1, 1, size(vp_3d, 4))] .= 0


       # vertical velocity: no barotropic component to remove (w = 0 at
       # top/bottom by construction), so we project the raw field
       wp_3d = copy(fw)
       wp_3d[repeat(mask4D, 1, 1, 1, size(wp_3d, 4))] .= 0


       # ==========================================================
       # CREATE OUTPUT NETCDF, WRITE ROW BY ROW AS WE GO
       # ==========================================================
       out_file = joinpath(out_dir, "ModalAmp_$(suffix).nc")
       isfile(out_file) && rm(out_file)


       ds_out = NCDataset(out_file, "c")
       defDim(ds_out, "x", nx)
       defDim(ds_out, "y", ny)
       defDim(ds_out, "t", nt)
       defDim(ds_out, "mode", n_modes_keep)


       uhat_var = defVar(ds_out, "uhat", Float64, ("x","y","t","mode"))
       uhat_var.attrib["long_name"] = "Projected horizontal modal amplitude (u)"
       vhat_var = defVar(ds_out, "vhat", Float64, ("x","y","t","mode"))
       vhat_var.attrib["long_name"] = "Projected horizontal modal amplitude (v)"
       what_var = defVar(ds_out, "what", Float64, ("x","y","t","mode"))
       what_var.attrib["long_name"] = "Projected vertical modal amplitude (w)"


       for j in 1:ny
           # buffer a whole row in memory, write once per row instead of
           # once per grid point (nx times fewer NetCDF write calls)
           uhat_row = fill(NaN, nx, nt, n_modes_keep)
           vhat_row = fill(NaN, nx, nt, n_modes_keep)
           what_row = fill(NaN, nx, nt, n_modes_keep)


           # each i is fully independent of every other i, so this loop
           # parallelizes across however many Julia threads are available
           Threads.@threads for i in 1:nx
               hfac_col = @view hFacC[i, j, :]
               ocean_idx = findall(hfac_col .> 0)
               if length(ocean_idx) < 4
                   continue   # land / too-shallow, leave as NaN in the output
               end


               k_top = ocean_idx[1]
               k_bot = ocean_idx[end]


               # Ueig's top layer has been padded (copied from the second
               # layer) at save time, so it's now defined on the full
               # k_top:k_bot column, same grid as Weig/w/hFacC — no more
               # M-1 staggering, so we slice from k_top like Weig does.
               Phi_all = @view Ueig_tile[i, j, k_top:k_bot, :]     # (M, n_modes_keep)
               if any(isnan, Phi_all)
                   continue   # this column wasn't solved in the SL step, skip
               end
               Wphi_all = @view Weig_tile[i, j, k_top:k_bot, :]    # (M, n_modes_keep)
               if any(isnan, Wphi_all)
                   continue   # vertical eigenfunction not solved for this column
               end


               dz_col = (hfac_col .* DRF)[k_top:k_bot]
               H = sum(dz_col)


               u_prof = @view up_3d[i, j, k_top:k_bot, :]   # (M, nt)
               v_prof = @view vp_3d[i, j, k_top:k_bot, :]
               w_prof = @view wp_3d[i, j, k_top:k_bot, :]   # (M, nt)


               # weight each mode shape by depth thickness ONCE, then do the
               # whole (all hours x all modes) projection as one matrix
               # multiply instead of nt*n_modes individual dot products
               W = Phi_all .* dz_col                        # (M, n_modes_keep)
               uhat_row[i, :, :] = (1/H) .* (u_prof' * W)   # (nt, M)*(M,n_modes) -> (nt, n_modes)
               vhat_row[i, :, :] = (1/H) .* (v_prof' * W)


               # vertical mode projection, weighted over the full k_top:k_bot
               # column (Weig is defined at the same levels as w)
               Ww = Wphi_all .* dz_col                       # (M, n_modes_keep)
               what_row[i, :, :] = (1/H) .* (w_prof' * Ww)  # (nt, M)*(M,n_modes) -> (nt, n_modes)
           end


           uhat_var[:, j, :, :] = uhat_row
           vhat_var[:, j, :, :] = vhat_row
           what_var[:, j, :, :] = what_row


           if j % 10 == 0
               println("  row j=$j/$ny projected")
           end
       end


       close(ds_out)
       println("Saved $out_file")
   end
end
end
println("\nAll tiles projected successfully!")




