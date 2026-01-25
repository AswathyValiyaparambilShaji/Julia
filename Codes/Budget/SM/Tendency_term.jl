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


# --- Tile & time parameters ---
buf = 3
tx, ty = 47, 66
nx = tx + 2*buf
ny = ty + 2*buf
nz = 88
dt = 25  # time step in seconds
dto = 144
Tts = 366192
nt = div(Tts, dto)
ts = 72
nt_avg = div(nt, ts)




# --- Thickness & constants ---
thk = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)




rho0 = 999.8




for xn in cfg["xn_start"]:cfg["xn_end"]
   for yn in cfg["yn_start"]:cfg["yn_end"]




       suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)




       # --- Read APE ---
       APE = Float64.(open(joinpath(base2, "APE", "APE_t_sm_$suffix.bin"), "r") do io
          nbytes = nx * ny * nz * nt * sizeof(Float32)
          reshape(reinterpret(Float32, read(io, nbytes)),
                  nx, ny, nz, nt)
       end)


       # --- Read KE ---
       ke_t = Float64.(open(joinpath(base2, "KE", "ke_t_sm_$suffix.bin"), "r") do io
           nbytes = nx * ny * nz * nt * sizeof(Float32)
           raw_bytes = read(io, nbytes)
           raw_data = reinterpret(Float32, raw_bytes)
           reshape(raw_data, nx, ny, nz, nt)
       end)


       # --- Calculate total energy ---
       TE = APE .+ ke_t


       # --- Calculate ∂E/∂t using centered differences ---
       dEdt = zeros(Float64, nx, ny, nz, nt)


       # Time interval between saved snapshots (in seconds)
       dt_output = dt * dto  # = 25 * 144 = 3600 seconds = 1 hour


       # Forward difference for first time step
       dEdt[:, :, :, 1] = (TE[:, :, :, 2] .- TE[:, :, :, 1]) ./ dt_output


       # Centered differences for interior points
       for t in 2:nt-1
           dEdt[:, :, :, t] = (TE[:, :, :, t+1] .- TE[:, :, :, t-1]) ./ (2 * dt_output)
       end


       # Backward difference for last time step
       dEdt[:, :, :, nt] = (TE[:, :, :, nt] .- TE[:, :, :, nt-1]) ./ dt_output


       # --- Replace NaN with zero before depth integration ---
       dEdt[isnan.(dEdt)] .= 0.0


       # --- Depth integrate: multiply by DRF and sum over depth ---
       dEdt_depth_int = zeros(Float64, nx, ny, nt)
       for t in 1:nt
           for k in 1:nz
               dEdt_depth_int[:, :, t] .+= dEdt[:, :, k, t] .* DRF[k]
           end
       end


       # --- Time average: reshape and average over ts chunks ---
       dEdt_time_avg = zeros(Float64, nx, ny)
       
       dEdt_time_avg[:, :] = mean(dEdt_depth_int[:, :, :], dims=3)
      
       println(dEdt_time_avg[:,1])


       # --- Save output ---
       output_dir = joinpath(base2, "TE_t")
       mkpath(output_dir)
      
       # Save time-averaged, depth-integrated tendency
       open(joinpath(output_dir, "te_t_mean_$suffix.bin"), "w") do io
           write(io, Float32.(dEdt_time_avg))
       end
      
       println("Processed tile ($xn, $yn): saved $(sizeof(Float32.(dEdt_time_avg)) / 1024^2) MB")


   end
end




