using MAT, Statistics, Printf, LinearAlgebra, TOML
using Impute
using NCDatasets


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin


include(joinpath(@__DIR__, "..","..","..", "functions", "strum_liouville_noneqDZ_norm.jl"))


config_file = get(ENV, "JULIA_CONFIG",
        joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)


base  = cfg["base_path"]


# --- Grid ---
buf = 3
tx, ty = 47, 66
nx = tx + 2*buf
ny = ty + 2*buf
nz = 88


dto = 144
Tts = 366192
nt  = div(Tts, dto)
ts  = 72
nt_avg = div(nt, ts)


# --- Thickness ---
thk = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF = Float64.(thk[1:nz])


# --- Wave parameters ---
om = 2π / (12.42 * 3600)
n_modes_keep = 5
min_ocean_cells = 4


outdir = joinpath(base, "3day_mean", "SLmodes")
mkpath(outdir)


# lat grid, needed per-row for Coriolis f
minlat, maxlat = 24.0, 31.91
NY_full = 468
lat_full = range(minlat, maxlat, length=NY_full)

@time begin
# ==========================================================
# LOOP OVER TILES
# ==========================================================
for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]


        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
        println("\n=== Processing tile $suffix ===")


        hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))


        N2_phase = Float64.(open(joinpath(base,"3day_mean","N2","N2_3day_$suffix.bin"), "r") do io
            raw = read(io, nx * ny * nz * nt_avg * sizeof(Float32))
            reshape(reinterpret(Float32, raw), nx, ny, nz, nt_avg)
        end)


        Ce_tile   = fill(NaN, nx, ny, n_modes_keep)
        Cg_tile   = fill(NaN, nx, ny, n_modes_keep)
        L_tile    = fill(NaN, nx, ny, n_modes_keep)
        Ueig_tile = fill(NaN, nx, ny, nz, n_modes_keep)
        Weig_tile = fill(NaN, nx, ny, nz, n_modes_keep)


        for j in 1:ny
            j_global = clamp((yn-1)*ty + j - buf, 1, NY_full)
            f_pt = 2 * 7.2921e-5 * sin(deg2rad(lat_full[j_global]))


            for i in 1:nx


                hfac_col = hFacC[i, j, :]
                ocean_idx = findall(hfac_col .> 0)


                if length(ocean_idx) < min_ocean_cells
                    continue   # land / too-shallow column, leave as NaN
                end


                k_top = ocean_idx[1]
                k_bot = ocean_idx[end]


                # NaN-fill N2 in each 3-day window, then average across windows
                N2_col_ts = N2_phase[i, j, :, :]
                N2_col_ts_filled = similar(N2_col_ts)
                for t in 1:nt_avg
                    x = replace(N2_col_ts[:, t], NaN => missing)
                    N2_col_ts_filled[:, t] = coalesce.(Impute.nocb(Impute.locf(x)), NaN)
                end
                N2_mean_col = [mean(filter(!isnan, N2_col_ts_filled[k, :])) for k in 1:nz]


                # The per-window locf/nocb fill above only helps when a depth
                # level is NaN in SOME windows but valid in others. If the
                # surface (k_top) is NaN in EVERY window for this column,
                # nocb has nothing to backfill from, and the time-mean above
                # stays NaN. Fix that directly here: copy the surface value
                # from the layer immediately below it, same idea as the
                # nearest-neighbor fill already validated at the point level.
                if isnan(N2_mean_col[k_top])
                    N2_mean_col[k_top] = N2_mean_col[k_top+1]
                end


                dz_col = (hfac_col .* DRF)[k_top:k_bot]
                zf_col = -cumsum(dz_col)


                N2_valid = N2_mean_col[k_top:k_bot-1]
                N2_faces = vcat(N2_valid, N2_valid[end])


                k_sl, L_sl, C_sl, Cg_sl, Ce_sl, Weig_sl, Ueig_sl, Ueig2_sl =
                    sturm_liouville_noneqDZ_norm(zf_col, N2_faces, f_pt, om, 0)


                n_avail = min(n_modes_keep, length(Ce_sl))


                Ce_tile[i, j, 1:n_avail] = Ce_sl[1:n_avail]
                Cg_tile[i, j, 1:n_avail] = Cg_sl[1:n_avail]
                L_tile[i, j, 1:n_avail]  = L_sl[1:n_avail]


                # Ueig2 has one fewer point than Weig (it's a derivative of
                # Weig, so it has no value defined exactly at the surface
                # face k_top). Rather than leaving that slot NaN, extrapolate
                # it by duplicating the shallowest computed value (Ueig2[1,:])
                # into k_top too -- the same nearest-neighbor logic already
                # used to fix the surface N2 value above.
                Ueig_tile[i, j, k_top, 1:n_avail]         = Ueig2_sl[1, 1:n_avail]
                Ueig_tile[i, j, k_top+1:k_bot, 1:n_avail] = Ueig2_sl[:, 1:n_avail]


                Weig_tile[i, j, k_top:k_bot, 1:n_avail]   = Weig_sl[:, 1:n_avail]
            end


            if j % 10 == 0
                println("  row j=$j/$ny done")
            end
        end


        # ==========================================================
        # WRITE NETCDF FOR THIS TILE
        # ==========================================================
        ncfile = joinpath(outdir, "SLmodes_$(suffix).nc")
        isfile(ncfile) && rm(ncfile)


        NCDataset(ncfile, "c") do ds
            defDim(ds, "x", nx)
            defDim(ds, "y", ny)
            defDim(ds, "z", nz)
            defDim(ds, "mode", n_modes_keep)


            v = defVar(ds, "Ce", Float64, ("x","y","mode"))
            v.attrib["long_name"] = "Eigenspeed c_n (time-mean N2)"
            v.attrib["units"] = "m/s"
            v[:,:,:] = Ce_tile


            v = defVar(ds, "Cg", Float64, ("x","y","mode"))
            v.attrib["long_name"] = "Group speed c_g_n"
            v.attrib["units"] = "m/s"
            v[:,:,:] = Cg_tile


            v = defVar(ds, "L", Float64, ("x","y","mode"))
            v.attrib["long_name"] = "Horizontal wavelength"
            v.attrib["units"] = "m"
            v[:,:,:] = L_tile


            v = defVar(ds, "Ueig", Float64, ("x","y","z","mode"))
            v.attrib["long_name"] = "Normalized horizontal velocity eigenfunction (Phi_n)"
            v.attrib["note"] = "Defined at k_top+1:k_bot for each column; NaN elsewhere"
            v[:,:,:,:] = Ueig_tile


            v = defVar(ds, "Weig", Float64, ("x","y","z","mode"))
            v.attrib["long_name"] = "Vertical velocity eigenfunction (psi_n)"
            v.attrib["note"] = "Defined at k_top:k_bot for each column; NaN elsewhere"
            v[:,:,:,:] = Weig_tile


            ds.attrib["title"] = "Vertical mode SL solution using time-mean N2"
            ds.attrib["tile"] = suffix
            ds.attrib["n_modes_keep"] = n_modes_keep
        end


        println("Saved $ncfile")
    end
end
end

println("\nAll tiles processed successfully!")




