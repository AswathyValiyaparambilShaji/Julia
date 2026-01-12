using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML
using CairoMakie, SparseArrays


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin


config_file = get(ENV, "JULIA_CONFIG",
                joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)


base  = cfg["base_path"]
base2 = cfg["base_path2"]


NX, NY = 288, 468
minlat, maxlat = 24.0, 31.91
minlon, maxlon = 193.0, 199.0
lat = range(minlat, maxlat, length=NY)
lon = range(minlon, maxlon, length=NX)


buf = 3
tx, ty = 47, 66
nx = tx + 2*buf
ny = ty + 2*buf
nz = 88


dt  = 25
dto = 144
Tts = 366192
nt  = div(Tts, dto)
ts  = 72
nt_avg = div(nt, ts)


thk = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)


APE_full = fill(NaN, NX, NY)


# ==========================================================
# ============ BUILD DEPTH-INTEGRATED APE MAP ==============
# ==========================================================


for xn in cfg["xn_start"]:cfg["xn_end"]
  for yn in cfg["yn_start"]:cfg["yn_end"]


      suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)


      hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"),
                       (nx, ny, nz))


      DRFfull = DRF3d .* hFacC
      DRFfull[hFacC .== 0] .= 0


      APE = Float64.(open(joinpath(base2, "APE", "APE_t_sm_$suffix.bin"), "r") do io
          nbytes = nx * ny * nz * nt * sizeof(Float32)
          reshape(reinterpret(Float32, read(io, nbytes)),
                  nx, ny, nz, nt)
      end)


      # Time average ignoring NaN
      APE_clean = replace(APE, NaN => 0.0)
      APE_sum = sum(APE_clean, dims=4)
      APE_count = sum(.!isnan.(APE), dims=4)


      aped = zeros(Float64, nx, ny, nz)
      for i in 1:nx, j in 1:ny, k in 1:nz
          if APE_count[i,j,k,1] > 0
              aped[i,j,k] = APE_sum[i,j,k,1] / APE_count[i,j,k,1]
          else
              aped[i,j,k] = NaN
          end
      end


      # Depth integrate
      ape = zeros(Float64, nx, ny)
      for i in 1:nx, j in 1:ny
          weighted = aped[i,j,:] .* DRF3d[i,j,:]
          ape[i,j] = sum(weighted[.!isnan.(weighted)])
      end


      xs = (xn - 1) * tx + 1
      xe = xs + tx - 1
      ys = (yn - 1) * ty + 1
      ye = ys + ty - 1


      ape_interior = ape[buf+1:nx-buf, buf+1:ny-buf]


      APE_full[xs:xe, ys:ye] .= ape_interior


      println("Completed tile $suffix")
  end
end


println("\nAPE_full range: $(minimum(skipmissing(APE_full))) to $(maximum(skipmissing(APE_full)))")


# ==========================================================
# =================== VISUALIZATION ========================
# ==========================================================


fig = Figure(size=(1000, 800))


ax = Axis(fig[1, 1],
         title="Depth-Integrated Time-Averaged APE",
         xlabel="Longitude [°]",
         ylabel="Latitude [°]")


hm = CairoMakie.heatmap!(ax, lon, lat, APE_full;
                        interpolate=false,
                        colormap=:jet,
                        colorrange=(0, 8000))


Colorbar(fig[1, 2], hm, label="APE [J/m²]")


display(fig)


# Save figure
FIGDIR        = cfg["fig_base"]
save(joinpath(FIGDIR, "APE_sm_v2.png"), fig)



