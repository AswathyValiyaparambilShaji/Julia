using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter


include(joinpath(@__DIR__, "..","..","..", "functions", "strum_liouville_noneqDZ_norm.jl"))


config_file = get(ENV, "JULIA_CONFIG",
            joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)


base  = cfg["base_path"]
base2 = cfg["base_path2"]


# --- Domain & grid ---
NX, NY = 288, 468
minlat, maxlat = 24.0, 31.91
minlon, maxlon = 193.0, 199.0
lat = range(minlat, maxlat, length=NY)
lon = range(minlon, maxlon, length=NX)


# --- Tile & time ---
buf = 3
tx, ty = 47, 66
nx = tx + 2*buf
ny = ty + 2*buf
nz = 88


dt  = 25
dto = 144
Tts = 366192
 

ts     = 72
nt_avg = div(nt, ts)


# --- Thickness & constants ---
thk = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)


rho0 = 1027.5


# --- face depths zf from DRF ---
# zf has size nz+1 = 89 (surface to bottom)
dz = Float64.(DRF)
zf = vcat(0.0, -cumsum(dz))    # size (89,)


# Your N2 is at interior faces zf[2:end-1], size 87
# N2_phase is stored as size (nx, ny, 88, nt_avg)
# where index 1:87 are valid interfaces, index 88 is empty
zf_N2 = zf[2:end-1]            # size 87 — matches valid N2 levels


# --- Wave parameters ---
om = 2π / (12.42 * 3600)       # M2 frequency


# --- Output directories ---
mkpath(joinpath(base2, "modes"))


# ==========================================================
# ====================== MAIN LOOP =========================
# ==========================================================


for xn in cfg["xn_start"]:cfg["xn_end"]
for yn in cfg["yn_start"]:cfg["yn_end"]


    suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)


    # --- Read N2 (3-day averaged, at interior faces) ---
    N2_phase = Float64.(open(joinpath(base,"3day_mean","N2","N2_3day_$suffix.bin"), "r") do io
        raw = read(io, nx * ny * nz * nt_avg * sizeof(Float32))
        reshape(reinterpret(Float32, raw), nx, ny, nz, nt_avg)
    end)


    # valid N2 is only first 87 levels (index 88 is empty)
    # N2_valid : (nx, ny, 87, nt_avg)
    N2_valid = N2_phase[:, :, 1:end-1, :]   # drop last empty level


    # --- Read hFacC ---
    hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"),
                    (nx, ny, nz))


    # --- Thickness ---
    DRFfull = hFacC .* DRF3d
    DRFfull[hFacC .== 0] .= 0.0


    # --- Find last valid ocean center index per point ---
    k_last_full = zeros(Int, nx, ny)
    for j in 1:ny, i in 1:nx
        for k in nz:-1:1
            if hFacC[i, j, k] >= 1.0
                k_last_full[i, j] = k
                break
            end
        end
    end


    # ==========================================================
    # ========= MODAL DECOMPOSITION LOOP ======================
    # ==========================================================


    Nmodes = 5


    Ueig2_all = zeros(Float32, nx, ny, 87,   Nmodes, nt_avg)
    Weig_all  = zeros(Float32, nx, ny, 87,   Nmodes, nt_avg)
    Ce_all    = zeros(Float32, nx, ny, Nmodes, nt_avg)
    Cg_all    = zeros(Float32, nx, ny, Nmodes, nt_avg)


    for t in 1:nt_avg
        for j in 1:ny, i in 1:nx


            # skip land points
            if k_last_full[i, j] == 0
                continue
            end


            # last valid interface index
            # k_last_full is last valid center → last valid interface is k_last_full
            # because interface k sits between center k-1 and center k
            kbot = min(k_last_full[i, j], 87)   # cap at 87 (valid N2 size)


            # extract valid N2 and matching zf for this column
            N2_faces = N2_valid[i, j, 2:kbot, t]   # size (kbot,)
            zf_local = zf_N2[1:kbot]                # matching face depths


            # need at least 3 points for eigen solver
            if length(N2_faces) < 3
                continue
            end


            # skip if any NaN or Inf
            if any(isnan.(N2_faces)) || any(isinf.(N2_faces))
                continue
            end


            # local latitude and Coriolis
            lat_ij = lat[j + (yn-1)*ty]
            f_ij   = 2 * 7.2921e-5 * sin(deg2rad(lat_ij))


            # call SL function
            try
                k_sl, L_sl, C_sl, Cg_sl, Ce_sl, Weig_sl, Ueig_sl, Ueig2_sl =
                    sturm_liouville_noneqDZ_norm(zf_local, N2_faces, f_ij, om, 0)


                Ce_all[i, j, :, t]               = Float32.(Ce_sl[1:Nmodes])
                Cg_all[i, j, :, t]               = Float32.(Cg_sl[1:Nmodes])
                Weig_all[i, j, 1:kbot, :, t]     = Float32.(Weig_sl[:, 1:Nmodes])
                Ueig2_all[i, j, 1:kbot-1, :, t]  = Float32.(Ueig2_sl[:, 1:Nmodes])
            catch e
                @warn "SL failed at i=$i j=$j t=$t: $e"
                continue
            end


        end
    end


    # ==========================================================
    # ========= SUMMARY ========================================
    # ==========================================================


    println("\n=== Tile $suffix Summary ===")


    n_success = sum(Ce_all[:, :, 1, 1] .!= 0)
    n_total   = nx * ny
    println("Solved: $n_success / $n_total points ($(round(100*n_success/n_total, digits=1))%)")


    n_failed = sum(Ce_all[:, :, 1, 1] .== 0)
    println("Skipped (land/bad N2): $n_failed points")


    println("\nCe (mode 1) stats for t=1:")
    Ce1   = Ce_all[:, :, 1, 1]
    valid = Ce1[Ce1 .!= 0]
    if length(valid) > 0
        println("  min = $(round(minimum(valid), digits=3)) m/s")
        println("  max = $(round(maximum(valid), digits=3)) m/s")
        println("  mean= $(round(mean(valid),    digits=3)) m/s")
    end


    i_check, j_check = 25, 30
    println("\nSingle point check (i=$i_check, j=$j_check):")
    for n in 1:Nmodes
        println("  Mode $n | Ce=$(round(Ce_all[i_check,j_check,n,1], digits=3)) m/s | Cg=$(round(Cg_all[i_check,j_check,n,1], digits=3)) m/s")
    end


    # --- Save ---
    open(joinpath(base2, "modes", "Ueig2_$suffix.bin"), "w") do io
        write(io, Ueig2_all)
    end
    open(joinpath(base2, "modes", "Weig_$suffix.bin"), "w") do io
        write(io, Weig_all)
    end
    open(joinpath(base2, "modes", "Ce_$suffix.bin"), "w") do io
        write(io, Ce_all)
    end
    open(joinpath(base2, "modes", "Cg_$suffix.bin"), "w") do io
        write(io, Cg_all)
    end


    println("Completed tile: $suffix")


end  # end yn
end  # end xn




