using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML
#using CairoMakie, SparseArrays


include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter


config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base  = cfg["base_path"]
base2 = cfg["base_path2"]


mkpath(joinpath(base2, "xflux"))
mkpath(joinpath(base2, "yflux"))
mkpath(joinpath(base2, "zflux"))


# --- Domain & grid ---
NX, NY = 288, 468
minlat, maxlat = 24.0, 31.91
minlon, maxlon = 193.0, 199.0
lat = range(minlat, maxlat, length=NY)
lon = range(minlon, maxlon, length=NX)


# --- Tile & time ---
buf  = 3
tx, ty = 47, 66
nx = tx + 2*buf
ny = ty + 2*buf
nz = 88


kz  = 1
dt  = 25
dto = 144          # model output interval in hours
Tts = 366192       # total time steps (hours)
nt  = div(Tts, dto)  # total number of snapshots in raw file


# -------------------------------------------------------------------------
# Weekly window: April 22 00:00:00 to April 28 23:00:00, 2012
#   Series starts 2012-03-01T00:00:00, one snapshot every dto=144 hours
#   March = 31 days = 744 h
#   Apr 22 00:00:00 = hour 744 + (22-1)*24 = 1248  -> snapshot div(1248,144)+1 = 9
#   Apr 28 23:00:00 = hour 744 +  28 *24-1 = 1415  -> snapshot div(1415,144)+1 = 10
#   So we read snapshots 9 and 10 (nt_week = 2)
# -------------------------------------------------------------------------
hour_apr22_start = 31*24 + (22-1)*24        # = 1248  (Apr 22 00:00)
hour_apr28_end   = 31*24 +  28   *24 - 1    # = 1415  (Apr 28 23:00)


idx_start = hour_apr22_start +1 # = 9
idx_end   = hour_apr28_end +1 # = 10
nt_week   = idx_end - idx_start + 1         # = 2

# --- Thickness & constants ---
thk   = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF   = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)
g = 9.8


# --- Filter (9-15 day band, 1 step sampling) ---
T1, T2, delt, N = 9.0, 15.0, 1.0, 4
fcutlow, fcuthigh = 1/T2, 1/T1
fnq = 1/delt
bpf = digitalfilter(Bandpass(fcutlow, fcuthigh), Butterworth(N); fs=fnq)




for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]


        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)


        # --- Read hFacC mask ---
        hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))


        # --- Read full rho (Float64, nx x ny x nz x nt) then subset to weekly window ---
        rho = Float64.(open(joinpath(base, "Density", "rho_in_$suffix.bin"), "r") do io
            nbytes = nx * ny * nz * nt * sizeof(Float64)
            raw_bytes = read(io, nbytes)
            raw_data = reinterpret(Float64, raw_bytes)
            reshape(raw_data, nx, ny, nz, nt)
        end)[:, :, :, idx_start:idx_end]   # subset to weekly window


        # --- Read full fu, fv, fw (Float32, nx x ny x nz x nt) then subset ---
        fu = Float64.(open(joinpath(base2, "UVW_F", "fu_$suffix.bin"), "r") do io
            nbytes = nx * ny * nz * nt * sizeof(Float32)
            raw_bytes = read(io, nbytes)
            raw_data = reinterpret(Float32, raw_bytes)
            reshape(raw_data, nx, ny, nz, nt)
        end)[:, :, :, idx_start:idx_end]


        fv = Float64.(open(joinpath(base2, "UVW_F", "fv_$suffix.bin"), "r") do io
            nbytes = nx * ny * nz * nt * sizeof(Float32)
            raw_bytes = read(io, nbytes)
            raw_data = reinterpret(Float32, raw_bytes)
            reshape(raw_data, nx, ny, nz, nt)
        end)[:, :, :, idx_start:idx_end]


        fw = Float64.(open(joinpath(base2, "UVW_F", "fw_$suffix.bin"), "r") do io
            nbytes = nx * ny * nz * nt * sizeof(Float32)
            raw_bytes = read(io, nbytes)
            raw_data = reinterpret(Float32, raw_bytes)
            reshape(raw_data, nx, ny, nz, nt)
        end)[:, :, :, idx_start:idx_end]


        # --- Depth geometry (same as upstream script) ---
        DRFfull = hFacC .* DRF3d
        z   = cumsum(DRFfull, dims=3)
        zz  = cat(zeros(nx, ny, 1), z; dims=3)
        za  = -0.5 .* (zz[:, :, 1:end-1] .+ zz[:, :, 2:end])
        depth = sum(DRFfull, dims=3)
        DRFfull[hFacC .== 0] .= 0.0


        # --- Bandpass filter rho (time is last dim) ---
        fr = bandpassfilter(rho, T1, T2, delt, N, nt_week)


        # --- Pressure perturbation (same as upstream script) ---
        pres  = g .* cumsum(fr .* DRFfull, dims=3)
        pfz   = cat(zeros(nx, ny, 1, nt_week), pres; dims=3)
        pc_3d = 0.5 .* (pfz[:, :, 1:end-1, :] .+ pfz[:, :, 2:end, :])
        pa    = sum(pc_3d .* DRFfull, dims=3) ./ depth
        pp_3d = pc_3d .- pa


        mask4D = reshape(hFacC .== 0, nx, ny, nz, 1)
        pp_3d[repeat(mask4D, 1, 1, 1, nt_week)] .= 0


        # --- Velocity perturbations (same as upstream script) ---
        ucA_3d = sum(fu .* DRFfull, dims=3) ./ depth
        up_3d  = fu .- ucA_3d
        up_3d[repeat(mask4D, 1, 1, 1, nt_week)] .= 0


        vcA_3d = sum(fv .* DRFfull, dims=3) ./ depth
        vp_3d  = fv .- vcA_3d
        vp_3d[repeat(mask4D, 1, 1, 1, nt_week)] .= 0


        wcA_3d = sum(fw .* DRFfull, dims=3) ./ depth
        wp_3d  = fw .- wcA_3d
        wp_3d[repeat(mask4D, 1, 1, 1, nt_week)] .= 0


        # --- Fluxes (same as upstream script) ---
        xflx_3d = up_3d .* pp_3d
        yflx_3d = vp_3d .* pp_3d
        zflx_3d = wp_3d .* pp_3d


        # --- Time average over the weekly window then save ---
        xfm_3d = mean(xflx_3d, dims=4)   # (nx, ny, nz, 1)
        yfm_3d = mean(yflx_3d, dims=4)
        zfm_3d = mean(zflx_3d, dims=4)


        open(joinpath(base2, "xflux", "xflx_weekly_$suffix.bin"), "w") do io
            write(io, Float32.(xfm_3d))
        end
        open(joinpath(base2, "yflux", "yflx_weekly_$suffix.bin"), "w") do io
            write(io, Float32.(yfm_3d))
        end
       


        println("  Completed tile: $suffix (weekly Apr 22-28, $nt_week snapshots averaged)")
    end
end


println("Completed weekly perturbation flux -- Apr 22 00:00 to Apr 28 23:00")




