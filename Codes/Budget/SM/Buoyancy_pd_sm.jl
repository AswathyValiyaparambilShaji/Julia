using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML, Impute


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
g = 9.8


# --- Output directories ---
mkpath(joinpath(base2, "BP"))


println("Starting buoyancy production calculation for 42 tiles...")


for xn in cfg["xn_start"]:cfg["xn_end"]
  for yn in cfg["yn_start"]:cfg["yn_end"]


      suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)


      println("\n--- Processing tile: $suffix ---")
    
      # --- Read grid metrics ---
      hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
      dx = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
      dy = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))
      
      # --- Read density field ---
      rho = Float64.(open(joinpath(base,"Density", "rho_in_$suffix.bin"), "r") do io
          nbytes = nx * ny * nz * nt * sizeof(Float64)
          raw_bytes = read(io, nbytes)
          raw_data = reinterpret(Float64, raw_bytes)
          reshape(raw_data, nx, ny, nz, nt)
      end)
      
      # --- Read N2 (3-day averaged) ---
      N2_phase = Float64.(open(joinpath(base,"3day_mean","N2","N2_3day_$suffix.bin"), "r") do io
          raw = read(io, nx * ny * nz * nt_avg * sizeof(Float32))
          reshape(reinterpret(Float32, raw), nx, ny, nz, nt_avg)
      end)


      # --- Read buoyancy fluctuations ---
      b = Float64.(open(joinpath(base2, "b", "b_t_sm_$suffix.bin"), "r") do io
          raw = read(io, nx * ny * nz * nt * sizeof(Float32))
          reshape(reinterpret(Float32, raw), nx, ny, nz, nt)
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


      # Filter N2: anything below threshold becomes NaN
      N2_threshold = 1.0e-8
      N2_center[N2_center .< N2_threshold] .= NaN
      
      # Linear interpolation to fill NaN values along vertical dimension
      println("  Interpolating N2 values...")
      for i in 1:nx, j in 1:ny, t in 1:nt_avg
          N2_center[i, j, :, t] = Impute.interp(N2_center[i, j, :, t])
      end
      
      println("  N2 range after interpolation: $(extrema(N2_center))")


      # --- Calculate cell thicknesses ---
      DRFfull = hFacC .* DRF3d
      DRFfull[hFacC .== 0] .= 0.0
      
      # --- Calculate 3-day averaged mean buoyancy field B ---
      println("Calculating mean buoyancy field...")
      B = zeros(Float64, nx, ny, nz, nt_avg)
      
      for i in 1:nt_avg
          t_start = (i-1) * ts + 1
          t_end = min(i * ts, nt)
          
          # Average density over 3-day window
          rho_avg = mean(rho[:, :, :, t_start:t_end], dims=4)[:, :, :, 1]
          
          # Calculate buoyancy: B = -g * (ρ - ρ₀) / ρ₀
          B[:, :, :, i] = -g .* (rho_avg .- rho0) ./ rho0
      end
    
      # --- Calculate mean buoyancy gradients: ∂B/∂x, ∂B/∂y ---
      println("Calculating mean buoyancy gradients...")
      B_x = zeros(Float64, nx, ny, nz, nt_avg)
      B_y = zeros(Float64, nx, ny, nz, nt_avg)
      
      # X-gradient: ∂B/∂x (central difference)
      dx_avg = dx[2:end-1, :] .+ dx[1:end-2, :]
      B_x[2:end-1, :, :, :] = (B[3:end, :, :, :] .- B[1:end-2, :, :, :]) ./
                                reshape(dx_avg, nx-2, ny, 1, 1)
      
      # Y-gradient: ∂B/∂y (central difference)
      dy_avg = dy[:, 2:end-1] .+ dy[:, 1:end-2]
      B_y[:, 2:end-1, :, :] = (B[:, 3:end, :, :] .- B[:, 1:end-2, :, :]) ./
                                reshape(dy_avg, nx, ny-2, 1, 1)
      
      println("Gradients calculated")
    
      # --- Initialize output array ---
      bp = zeros(Float64, nx, ny, nt)
    
      # --- Calculate buoyancy production: P^B = -ρ₀ ∫[(b/N²)·u·∇B] dz ---
      println("Calculating buoyancy production...")
      for t in 1:nt
          # Map timestep to corresponding 3-day average period
          t_avg = min(div(t - 1, ts) + 1, nt_avg)
        
          # Get mean fields at this period
          n2_val = @view N2_center[:, :, :, t_avg]
          B_x_t = @view B_x[:, :, :, t_avg]
          B_y_t = @view B_y[:, :, :, t_avg]


          # Get fluctuating fields at this timestep
          b_t = @view b[:, :, :, t]
          ut = @view fu[:, :, :, t]
          vt = @view fv[:, :, :, t]


          # Calculate buoyancy production terms:
          # (b/N²)·u·∇B = (b/N²)·u·∂B/∂x + (b/N²)·v·∂B/∂y
          temp1 = (b_t ./ n2_val) .* ut .* B_x_t .* DRFfull   # (b/N²)*u*∂B/∂x
          temp2 = (b_t ./ n2_val) .* vt .* B_y_t .* DRFfull   # (b/N²)*v*∂B/∂y
          
          # Handle any remaining NaN/Inf from division
          temp1[.!isfinite.(temp1)] .= 0.0
          temp2[.!isfinite.(temp2)] .= 0.0
        
          # Depth integrate with negative sign: P^B = -ρ₀ * ∫[...] dz
          bp[:, :, t] = -rho0 .* dropdims(sum(temp1 .+ temp2, dims=3), dims=3)
      end
    
      println("Buoyancy production calculation complete")
    
      # --- Time average ---
      BP = dropdims(mean(bp, dims=3), dims=3)
    
      println("  BP range: $(extrema(BP[isfinite.(BP)]))")
    
      # --- Save outputs ---
      output_dir = joinpath(base2, "BP")
      mkpath(output_dir)
    
      # Save time-averaged buoyancy production
      open(joinpath(output_dir, "bp_mean_$suffix.bin"), "w") do io
          write(io, Float32.(BP))
      end
    
      println("Completed tile: $suffix")
      println("Output saved to $output_dir")
  end
end


println("\n=== All 42 tiles processed successfully ===")


