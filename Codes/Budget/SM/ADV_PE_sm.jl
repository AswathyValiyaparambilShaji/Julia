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
mkpath(joinpath(base2, "ADV_PE"))

println("Starting KE flux calculation for 42 tiles...")

for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]


        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)

        println("\n--- Processing tile: $suffix ---")
        # --- Read grid metrics ---
        hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
        dx = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
        dy = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))
        
        # --- Read U and V (3-day averaged) ---
        U = Float64.(open(joinpath(base,"3day_mean", "U", "ucc_3day_$suffix.bin"), "r") do io
            nbytes = nx * ny * nz * nt_avg * sizeof(Float32)
            raw_bytes = read(io, nbytes)
            raw_data = reinterpret(Float32, raw_bytes)
            reshape(raw_data, nx, ny, nz, nt_avg)
        end)
        
        V = Float64.(open(joinpath(base, "3day_mean", "V", "vcc_3day_$suffix.bin"), "r") do io
            nbytes = nx * ny * nz * nt_avg * sizeof(Float32)
            raw_bytes = read(io, nbytes)
            raw_data = reinterpret(Float32, raw_bytes)
            reshape(raw_data, nx, ny, nz, nt_avg)
        end)
        
        # --- Read PE (full temporal resolution) ---
        pe = Float64.(open(joinpath(base2, "pe", "pe_t_sm_$suffix.bin"), "r") do io
            nbytes = nx * ny * nz * nt * sizeof(Float32)
            raw_bytes = read(io, nbytes)
            raw_data = reinterpret(Float32, raw_bytes)
            reshape(raw_data, nx, ny, nz, nt)
        end)
        
        # --- Read N2 (3-day averaged) ---
        N2 = Float64.(open(joinpath(base, "3day_mean", "N2", "N2_3day_$suffix.bin"), "r") do io
            nbytes = nx * ny * nz * nt_avg * sizeof(Float32)
            raw_bytes = read(io, nbytes)
            raw_data = reinterpret(Float32, raw_bytes)
            reshape(raw_data, nx, ny, nz, nt_avg)
        end)
        
        # --- Calculate grid metrics ---
        DRFfull = hFacC .* DRF3d
        DRFfull[hFacC .== 0] .= 0.0
        
        # --- Adjust N2 to nz+1 levels (interfaces) then back to centers ---
        N2_adjusted = zeros(Float64, nx, ny, nz+1, nt_avg)
        N2_adjusted[:, :, 1, :] = N2[:, :, 1, :]  # FIX: Changed from N2_phase to N2
        N2_adjusted[:, :, 2:nz, :] = N2[:, :, 1:nz-1, :]  # FIX: Changed from N2_phase to N2
        N2_adjusted[:, :, nz+1, :] = N2[:, :, nz, :]  # FIX: Use nz instead of nz-1
        
        N2_center = zeros(Float64, nx, ny, nz, nt_avg)
        for k in 1:nz
            N2_center[:, :, k, :] = (N2_adjusted[:, :, k, :] .+ N2_adjusted[:, :, k+1, :]) ./ 2.0
        end

        # ==========================================================
        # ======== FILTER OUT ANOMALOUSLY LOW N2 VALUES ============
        # ==========================================================
        
        # Use physical threshold instead of statistical one
        # N2 < 1e-6 represents very weak stratification
        N2_threshold = 1.0e-8
        
        println("Tile $suffix:")
        println("  Using physical N2 threshold: $N2_threshold")
        
        # Count values that will be filtered
        n_filtered = sum(N2_center .< N2_threshold)
        n_total = length(N2_center)
        println("  Filtering $(n_filtered) values out of $(n_total) ($(round(100*n_filtered/n_total, digits=2))%)")
        
        # Replace low N2 values with threshold (conservative approach)
        # This avoids over-smoothing while preventing extreme APE values
        N2_center[N2_center .< N2_threshold] .= N2_threshold
        
        println("  After filtering - N2 range: ", extrema(N2_center))

        
        # --- Calculate PE gradients (fully vectorized) ---
        println("Calculating PE gradients...")
        pe_x = zeros(Float64, nx, ny, nz, nt)
        pe_y = zeros(Float64, nx, ny, nz, nt)
        
        # X-gradient (interior points only): ∂PE/∂x
        dx_avg = dx[2:end-1, :] .+ dx[1:end-2, :]
        pe_x[2:end-1, :, :, :] = (pe[3:end, :, :, :] .- pe[1:end-2, :, :, :]) ./ 
                                  reshape(dx_avg, nx-2, ny, 1, 1)
        
        # Y-gradient (interior points only): ∂PE/∂y
        dy_avg = dy[:, 2:end-1] .+ dy[:, 1:end-2]
        pe_y[:, 2:end-1, :, :] = (pe[:, 3:end, :, :] .- pe[:, 1:end-2, :, :]) ./ 
                                  reshape(dy_avg, nx, ny-2, 1, 1)
        
        println("Gradients calculated")

        # Need to mask N2 greater than 10^-8

        
        # --- Initialize output: depth-integrated flux at each timestep ---
        U_PE = zeros(Float64, nx, ny, nt)
        
        # --- Calculate advective PE flux for each timestep ---
        println("Calculating advective PE flux...")
        for t in 1:nt
            # Map timestep to corresponding 3-day average period
            t_avg = min(div(t - 1, ts) + 1, nt_avg)
            
            # Get 3-day averaged velocity and N2
            u_avg = @view U[:, :, :, t_avg]
            v_avg = @view V[:, :, :, t_avg]
            n2_avg = @view N2_center[:, :, :, t_avg]
            
            # Get PE gradients at this timestep
            pe_x_t = @view pe_x[:, :, :, t]
            pe_y_t = @view pe_y[:, :, :, t]
            
            # Calculate advective flux: u·∇PE / N²
            temp1 = u_avg .* pe_x_t ./ n2_avg
            temp2 = v_avg .* pe_y_t ./ n2_avg
            
            # Handle infinities and NaNs
            temp1[isnan.(temp1)] .= 0.0
            temp2[isnan.(temp2)] .= 0.0
            
            # Depth integrate: ρ₀ * ∫(u·∇PE / N²) dz
            U_PE[:, :, t] = rho0 .* dropdims(sum((temp1 .+ temp2) .* DRFfull, dims=3), dims=3)
        end
        
        println("Flux calculation complete")
        
        # --- Time average ---
        u_pe_mean = dropdims(mean(U_PE, dims=3), dims=3)
        
        # --- Save outputs ---
        output_dir = joinpath(base2, "U_PE")
        mkpath(output_dir)

        # Save time-averaged flux
        open(joinpath(base2, "U_PE", "u_pe_mean_$suffix.bin"), "w") do io
            write(io, Float32.(u_pe_mean))
        end
        
        #= Save full time series
        open(joinpath(base2, "U_PE", "u_pe_timeseries_$suffix.bin"), "w") do io
            write(io, U_PE)
        end=#
        
        println("Completed tile: $suffix")
        println("Output saved to $(joinpath(base2, "U_PE"))")
    end
end


println("\n === All tiles processed successfully ===")