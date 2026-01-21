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

println("Starting Shear Production (hor) calculation for 42 tiles...")

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
        
        
        fu = Float64.(open(joinpath(base2, "UVW_F", "fu_$suffix.bin"), "r") do io
            # Calculate the number of bytes needed
            nbytes = nx * ny * nz *nt * sizeof(Float32)
            # Read the raw bytes
            raw_bytes = read(io, nbytes)
            # Reinterpret as Float64 array and reshape
            raw_data = reinterpret(Float32, raw_bytes)
            reshaped_data = reshape(raw_data, nx, ny,nz ,nt)
        end)


        fv = Float64.(open(joinpath(base2, "UVW_F", "fv_$suffix.bin"), "r") do io
            # Calculate the number of bytes needed
            nbytes = nx * ny * nz *nt * sizeof(Float32)
            # Read the raw bytes
            raw_bytes = read(io, nbytes)
            # Reinterpret as Float64 array and reshape
            raw_data = reinterpret(Float32, raw_bytes)
            reshaped_data = reshape(raw_data, nx, ny, nz, nt)
        end)
        
        # --- Calculate cell thicknesses ---
        DRFfull = hFacC .* DRF3d
        DRFfull[hFacC .== 0] .= 0.0
        
        # --- Calculate KE gradients (vectorized) ---
        println("Calculating gradients...")
        U_x = zeros(Float64, nx, ny, nz, nt_avg)
        U_y = zeros(Float64, nx, ny, nz, nt_avg)

        V_x = zeros(Float64, nx, ny, nz, nt_avg)
        V_y = zeros(Float64, nx, ny, nz, nt_avg)
        
        # X-gradient: ∂U/∂x (central difference)
        dx_avg = dx[2:end-1, :] .+ dx[1:end-2, :]
        U_x[2:end-1, :, :, :] = (U[3:end, :, :, :] .- U[1:end-2, :, :, :]) ./
                                  reshape(dx_avg, nx-2, ny, 1, 1)
        V_x[2:end-1, :, :, :] = (V[3:end, :, :, :] .- V[1:end-2, :, :, :]) ./
                                  reshape(dx_avg, nx-2, ny, 1, 1)
        
        # Y-gradient: ∂U/∂y (central difference)
        dy_avg = dy[:, 2:end-1] .+ dy[:, 1:end-2]
        U_y[:, 2:end-1, :, :] = (U[:, 3:end, :, :] .- U[:, 1:end-2, :, :]) ./
                                  reshape(dy_avg, nx, ny-2, 1, 1)
        # Y-gradient: ∂V/∂y (central difference)
        V_y[:, 2:end-1, :, :] = (V[:, 3:end, :, :] .- V[:, 1:end-2, :, :]) ./
                                  reshape(dy_avg, nx, ny-2, 1, 1)
        
        println("Gradients calculated")
        
        # --- Initialize output array ---
        sp_h = zeros(Float64, nx, ny, nt)
        
        for t in 1:nt
            # Map timestep to corresponding 3-day average period
            t_avg = min(div(t - 1, ts) + 1, nt_avg)
            
            # Get gradients at this timestep
            U_x_t = @view U_x[:, :, :, t_avg]
            V_y_t = @view V_y[:, :, :, t_avg]
            V_x_t = @view V_x[:, :, :, t_avg]
            U_y_t = @view U_y[:, :, :, t_avg]

            ut = @view fu[:, :,:, t]
            vt = @view fv[:, :,:, t]

            # Calculate advective flux: 
            temp1 = ut .* ut .* U_x_t.* DRFfull
            temp2 = ut .* vt .* U_y_t.* DRFfull
            temp3 = ut .* vt .* V_x_t.* DRFfull
            temp4 = vt .* vt .* V_y_t.* DRFfull
           
            # Depth integrate: 
            sp_h[:, :, t] = -rho0 .*(dropdims(sum((temp1 .+ temp2 .+temp3 .+temp4) , dims=3), dims=3))
        end
        
        println("Flux calculation complete")
        
        # --- Time average ---
        SP_H = dropdims(mean(sp_h, dims=3), dims=3)
        
        # --- Save outputs ---
        output_dir = joinpath(base2, "SP_H")
        mkpath(output_dir)
        
        # Save time-averaged flux
        open(joinpath(output_dir, "sp_h_mean_$suffix.bin"), "w") do io
            write(io, Float32.(SP_H))
        end
                 
        println("Completed tile: $suffix")
        println("Output saved to $output_dir")
    end
end

println("\n=== All 42 tiles processed successfully ===")




