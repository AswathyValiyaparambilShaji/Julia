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


# --- Tile & time parameters ---
buf = 3
tx, ty = 47, 66
nx = tx + 2*buf
ny = ty + 2*buf
nz = 88
dt = 25
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


# --- Output directories ---
mkpath(joinpath(base2, "SP_V"))


println("Starting vertical shear production calculation for 42 tiles...")


for xn in cfg["xn_start"]:cfg["xn_end"]
   for yn in cfg["yn_start"]:cfg["yn_end"]


       suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)


       println("\n--- Processing tile: $suffix ---")
      
       # --- Read grid metrics ---
       hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
      
       # --- Read mean velocity fields (3-day averaged) ---
       U = Float64.(open(joinpath(base,"3day_mean","U", "ucc_3day_$suffix.bin"), "r") do io
           nbytes = nx * ny * nz * nt_avg * sizeof(Float32)
           raw_bytes = read(io, nbytes)
           raw_data = reinterpret(Float32, raw_bytes)
           reshape(raw_data, nx, ny, nz, nt_avg)
       end)
      
       V = Float64.(open(joinpath(base, "3day_mean","V", "vcc_3day_$suffix.bin"), "r") do io
           nbytes = nx * ny * nz * nt_avg * sizeof(Float32)
           raw_bytes = read(io, nbytes)
           raw_data = reinterpret(Float32, raw_bytes)
           reshape(raw_data, nx, ny, nz, nt_avg)
       end)
      
       # --- Read fluctuating velocities ---
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
      
       # --- Calculate cell thicknesses ---
       DRFfull = hFacC .* DRF3d
       DRFfull[hFacC .== 0] .= 0.0
      
       # --- Calculate vertical gradients of mean velocities: ∂U/∂z, ∂V/∂z ---
       println("Calculating vertical gradients of mean velocities...")
       U_z = zeros(Float64, nx, ny, nz, nt_avg)
       V_z = zeros(Float64, nx, ny, nz, nt_avg)
          
       for t_avg in 1:nt_avg
           for k in 2:nz-1
               dz = DRF[k-1]/2.0 + DRF[k] + DRF[k+1]/2.0
               
               U_z[:, :, k, t_avg] = (U[:, :, k-1, t_avg] .- U[:, :, k+1, t_avg]) ./ dz
               V_z[:, :, k, t_avg] = (V[:, :, k-1, t_avg] .- V[:, :, k+1, t_avg]) ./ dz
           end          
           
        end
      
       println("Vertical gradients calculated")
      
       # --- Initialize output array ---
       sp_v = zeros(Float64, nx, ny, nt)
      
       # --- Calculate vertical shear production: P^s_v = -∫[w·u·∂U/∂z] dz ---
       println("Calculating vertical shear production...")
       for t in 1:nt
           # Map timestep to corresponding 3-day average period
           t_avg = min(div(t - 1, ts) + 1, nt_avg)
          
           # Get mean velocity vertical gradients at this period
           U_z_t = @view U_z[:, :, :, t_avg]
           V_z_t = @view V_z[:, :, :, t_avg]


           # Get fluctuating velocities at this timestep
           ut = @view fu[:, :, :, t]
           vt = @view fv[:, :, :, t]
           wt = @view fw[:, :, :, t]


           # Calculate vertical shear production terms:
           # w·u·∂U/∂z + w·v·∂V/∂z
           temp1 = wt .* ut .* U_z_t .* DRFfull   # w*u*∂U/∂z
           temp2 = wt .* vt .* V_z_t .* DRFfull   # w*v*∂V/∂z
          
           # Depth integrate with negative sign:
           sp_v[:, :, t] = -rho0 .* dropdims(sum((temp1 .+ temp2), dims=3), dims=3)
       end
      
       println("Vertical shear production calculation complete")
      
       # --- Time average ---
       SP_V = dropdims(mean(sp_v, dims=3), dims=3)
      
       # --- Save outputs ---
       output_dir = joinpath(base2, "SP_V")
       mkpath(output_dir)
      
       # Save time-averaged vertical shear production
       open(joinpath(output_dir, "sp_v_mean_$suffix.bin"), "w") do io
           write(io, Float32.(SP_V))
       end
      
       println("Completed tile: $suffix")
       println("Output saved to $output_dir")
   end
end


println("\n=== All 42 tiles processed successfully ===")




