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
ts = 72  # CRITICAL FIX: Was undefined, needed for t_avg calculation
nt_avg = div(nt, ts)


# --- Thickness & constants ---
thk = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)


rho0 = 999.8


# --- Output directories ---
mkpath(joinpath(base2, "ADV_KE"))

println("Starting KE flux calculation for 42 tiles...")

for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]


        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)

        println("\n--- Processing tile: $suffix ---")
        
        # --- Read grid metrics ---
        hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
        dx = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
        dy = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))
        
        # --- Read velocity fields (3-day averaged) ---
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
        
        # --- Read kinetic energy (full temporal resolution) ---
        ke_t = Float64.(open(joinpath(base2, "KE", "ke_t_sm_$suffix.bin"), "r") do io
            nbytes = nx * ny * nz * nt * sizeof(Float32)
            raw_bytes = read(io, nbytes)
            raw_data = reinterpret(Float32, raw_bytes)
            reshape(raw_data, nx, ny, nz, nt)
        end)
        
        # --- Calculate cell thicknesses ---
        DRFfull = hFacC .* DRF3d
        DRFfull[hFacC .== 0] .= 0.0
        
        # --- Calculate KE gradients (vectorized) ---
        println("Calculating KE gradients...")
        ke_x = zeros(Float64, nx, ny, nz, nt)
        ke_y = zeros(Float64, nx, ny, nz, nt)
        
        # X-gradient: ∂KE/∂x (central difference)
        dx_avg = dx[2:end-1, :] .+ dx[1:end-2, :]
        ke_x[2:end-1, :, :, :] = (ke_t[3:end, :, :, :] .- ke_t[1:end-2, :, :, :]) ./
                                  reshape(dx_avg, nx-2, ny, 1, 1)
        
        # Y-gradient: ∂KE/∂y (central difference)
        dy_avg = dy[:, 2:end-1] .+ dy[:, 1:end-2]
        ke_y[:, 2:end-1, :, :] = (ke_t[:, 3:end, :, :] .- ke_t[:, 1:end-2, :, :]) ./
                                  reshape(dy_avg, nx, ny-2, 1, 1)
        
        println("Gradients calculated")
        
        # --- Initialize output array ---
        U_KE = zeros(Float64, nx, ny, nt)
        
        # --- Calculate advective KE flux for each timestep ---
        println("Calculating advective KE flux...")
        for t in 1:nt
            # Map timestep to corresponding 3-day average period
            t_avg = min(div(t - 1, ts) + 1, nt_avg)
            
            # Get 3-day averaged velocities
            u_avg = @view U[:, :, :, t_avg]
            v_avg = @view V[:, :, :, t_avg]
            
            # Get KE gradients at this timestep
            ke_x_t = @view ke_x[:, :, :, t]
            ke_y_t = @view ke_y[:, :,:, t]
            
            # Calculate advective flux: U·∇KE
            temp1 = u_avg .* ke_x_t
            temp2 = v_avg .* ke_y_t
            
            # Handle infinities and NaNs
            # temp1[.!isfinite.(temp1)] .= 0.0
            # temp2[.!isfinite.(temp2)] .= 0.0
            
            # Depth integrate: ∫(U·∇KE) dz
            U_KE[:, :, t] = dropdims(sum((temp1 .+ temp2) .* DRFfull, dims=3), dims=3)
        end
        
        println("Flux calculation complete")
        
        # --- Time average ---
        u_ke_mean = dropdims(mean(U_KE, dims=3), dims=3)
        
        # --- Save outputs ---
        output_dir = joinpath(base2, "U_KE")
        mkpath(output_dir)
        
        # Save time-averaged flux
        open(joinpath(output_dir, "u_ke_mean_$suffix.bin"), "w") do io
            write(io, Float32.(u_ke_mean))
        end
        
        #= Save full time series
        open(joinpath(output_dir, "u_ke_timeseries_$suffix.bin"), "w") do io
            write(io, U_KE)
        end=#
        
        println("Completed tile: $suffix")
        println("Output saved to $output_dir")
    end
end

println("\n=== All 42 tiles processed successfully ===")




