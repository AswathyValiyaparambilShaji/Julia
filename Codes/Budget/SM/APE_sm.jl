using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter


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
nt  = div(Tts, dto)


ts      = 72      # 3-day window
nt_avg = div(nt, ts)


# --- Thickness & constants ---
thk = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)


rho0 = 999.8


# --- Output directories ---
mkpath(joinpath(base2, "APE"))
mkpath(joinpath(base2, "pe"))


# ==========================================================
# ====================== MAIN LOOP =========================
# ==========================================================


for xn in cfg["xn_start"]:cfg["xn_end"]
   for yn in cfg["yn_start"]:cfg["yn_end"]


       suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)


       # --- Read N2 (3-day averaged) ---
       N2_phase = Float64.(open(joinpath(base,"3day_mean","N2","N2_3day_$suffix.bin"), "r") do io
           raw = read(io, nx * ny * nz * nt_avg * sizeof(Float32))
           reshape(reinterpret(Float32, raw), nx, ny, nz, nt_avg)
       end)


       # --- Adjust N2 to interfaces ---
       N2_adjusted = zeros(Float64, nx, ny, nz+1, nt_avg)
       N2_adjusted[:, :, 1,   :] = N2_phase[:, :, 1,   :]
       N2_adjusted[:, :, 2:nz,:] = N2_phase[:, :, 1:nz-1, :]
       N2_adjusted[:, :, nz+1,:] = N2_phase[:, :, nz-1, :]


       # --- Average to cell centers ---
       N2_center = zeros(Float64, nx, ny, nz, nt_avg)
       for k in 1:nz
           N2_center[:, :, k, :] .=
               0.5 .* (N2_adjusted[:, :, k, :] .+
                       N2_adjusted[:, :, k+1, :])
       end


       # --- Read hFacC ---
       hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"),
                        (nx, ny, nz))


       # --- Thickness ---
       DRFfull = hFacC .* DRF3d
       DRFfull[hFacC .== 0] .= 0.0


       # --- Read buoyancy ---
       b = Float64.(open(joinpath(base2, "b", "b_t_sm_$suffix.bin"), "r") do io
           raw = read(io, nx * ny * nz * nt * sizeof(Float32))
           reshape(reinterpret(Float32, raw), nx, ny, nz, nt)
       end)


       # ==================================================
       # ======== NO REGULARIZATION - RAW APE =============
       # ==================================================
       
       APE = fill(NaN, nx, ny, nz, nt)


       for t in 1:nt_avg
           n2_val = N2_center[:, :, :, t]
          
           tstart = (t - 1) * ts + 1
           tend   = (t - 1) * ts + ts
          
           for tt in tstart:tend
               b_tt = b[:, :, :, tt]
              
               # Direct computation - no flooring, no explicit masking
               # APE = 0.5 * rho0 * (b^2 / N2)
               APE[:, :, :, tt] = 0.5 .* rho0 .* (b_tt.^2 ./ n2_val)
           end
       end


       # Print some diagnostics
       println("Tile $suffix:")
       println("  N2 range: ", extrema(filter(isfinite, N2_center)))
       println("  b range: ", extrema(filter(isfinite, b)))
       println("  APE range: ", extrema(filter(isfinite, APE)))
       println("  APE has ", sum(isfinite.(APE)), " finite values")
       println("  APE has ", sum(isinf.(APE)), " infinite values")
       println("  APE has ", sum(isnan.(APE)), " NaN values")


       # --- PE (unchanged) ---
       pe = 0.5 .* b.^2


       # --- Save ---
       open(joinpath(base2, "APE", "APE_t_sm_$suffix.bin"), "w") do io
           write(io, Float32.(APE))
       end


       open(joinpath(base2, "pe", "pe_t_sm_$suffix.bin"), "w") do io
           write(io, Float32.(pe))
       end


       println("Completed tile: $suffix")
   end
end




