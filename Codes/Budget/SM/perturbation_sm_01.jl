using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML
#using CairoMakie, SparseArrays


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base = cfg["base_path"]
base2 = cfg["base_path2"]


# --- TIME AVERAGING CONFIGURATION ---
# Set to true for 3-day averaging, false for full time period averaging
use_3day = true  # Change this to true for 3-day averaging


mkpath(joinpath(base2,"xflux"))
mkpath(joinpath(base2, "yflux"))
mkpath(joinpath(base2, "zflux"))


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


kz = 1
dt = 25
dto = 144
Tts = 366192
nt = div(Tts, dto)
nt3 = div(nt, 3*24)  # Number of timesteps for 3-day averaging


# --- Thickness & constants ---
thk = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)
g = 9.8


# --- Filter (915 day band, 1 step sampling here) ---
T1, T2, delt, N = 9.0, 15.0, 1.0, 4
fcutlow, fcuthigh = 1 / T2, 1 / T1
fnq = 1 / delt
bpf = digitalfilter(Bandpass(fcutlow, fcuthigh), Butterworth(N); fs = fnq)




# Now parallelize over ALL 42 tiles


for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]




        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)


    
        # --- Read fields ---
        #rho = read_bin(joinpath(base, "Density/rho_in_$suffix.bin"), (nx, ny, nz, nt))    # rho[isnan.(rho)] .= 0
        hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))


        rho = Float64.(open(joinpath(base,"Density", "rho_in_$suffix.bin"), "r") do io
            # Calculate the number of bytes needed
            nbytes = nx * ny * nz *nt * sizeof(Float64)
            # Read the raw bytes
            raw_bytes = read(io, nbytes)
            # Reinterpret as Float64 array and reshape
            raw_data = reinterpret(Float64, raw_bytes)
            reshaped_data = reshape(raw_data, nx, ny,nz ,nt)
        end)
        DRFfull = hFacC .* DRF3d
        z = cumsum(DRFfull, dims=3)
        zz= cat(zeros(nx, ny, 1),z; dims=3)
      
        za = -0.5 .* (zz[:, :, 1:end-1] .+ zz[:, :, 2:end]) 
        depth = sum(DRFfull, dims=3)
        DRFfull[hFacC .== 0] .= 0.0
        
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


        fw = Float64.(open(joinpath(base2, "UVW_F", "fw_$suffix.bin"), "r") do io
            # Calculate the number of bytes needed
            nbytes = nx * ny * nz *nt * sizeof(Float32)
            # Read the raw bytes
            raw_bytes = read(io, nbytes)
            # Reinterpret as Float64 array and reshape
            raw_data = reinterpret(Float32, raw_bytes)
            reshaped_data = reshape(raw_data, nx, ny, nz, nt)
        end)


        # --- Bandpass filter (time is last dim) ---
        fr = bandpassfilter(rho, T1, T2, delt,N,nt)
            # --- Pressure & perturbations ---
        pres  = g .* cumsum(fr .* DRFfull, dims=3)
        pfz   = cat(zeros(nx, ny, 1, nt), pres; dims=3)
        pc_3d = 0.5 .* (pfz[:, :, 1:end-1, :] .+ pfz[:, :, 2:end, :])
        pa    = sum(pc_3d .* DRFfull, dims=3) ./ depth
        pp_3d = pc_3d .- pa


        mask4D = reshape(hFacC .== 0, nx, ny, nz, 1)
        pp_3d[repeat(mask4D, 1, 1, 1, size(pp_3d, 4))] .= 0


        ucA_3d = sum(fu .* DRFfull, dims=3) ./ depth
        up_3d  = fu .- ucA_3d
        up_3d[repeat(mask4D, 1, 1, 1, size(up_3d, 4))] .= 0


        vcA_3d = sum(fv .* DRFfull, dims=3) ./ depth
        vp_3d  = fv .- vcA_3d
        vp_3d[repeat(mask4D, 1, 1, 1, size(vp_3d, 4))] .= 0 


        wcA_3d = sum(fw .* DRFfull, dims=3) ./ depth
        wp_3d  = fw .- wcA_3d
        wp_3d[repeat(mask4D, 1, 1, 1, size(wp_3d, 4))] .= 0


        # --- Fluxes ---
        xflx_3d = up_3d .* pp_3d
        yflx_3d = vp_3d .* pp_3d
        zflx_3d = wp_3d .* pp_3d


        # --- TIME AVERAGING: Full period or 3-day ---
        if !use_3day
            # Original behavior: time-mean over entire period then vertical integrate
            xfm_3d = mean(xflx_3d, dims=4)
            yfm_3d = mean(yflx_3d, dims=4)
            zfm_3d = mean(zflx_3d, dims=4)


            xfdm_3d = sum(xfm_3d .* DRFfull, dims=3)        # (nx,ny,1,1)
            yfdm_3d = sum(yfm_3d .* DRFfull, dims=3)
            zfdm_3d = sum(zfm_3d .* DRFfull, dims=3)


            hfdm_3d = sqrt.(xfdm_3d.^2 .+ yfdm_3d.^2)       # (nx,ny,1,1)
            
            # Save as before (single time snapshot)
            open(joinpath(base2, "xflux", "xflx_$suffix.bin"), "w") do io
                write(io, Float32.(xfm_3d))
            end
            open(joinpath(base2, "yflux", "yflx_$suffix.bin"), "w") do io
                write(io, Float32.(yfm_3d))
            end
            open(joinpath(base2, "zflux", "zflx_$suffix.bin"), "w") do io
                write(io, Float32.(zfm_3d))
            end
            
        else  # use_3day = true
            # 3-day averaging: compute rolling 3-day means, then vertical integrate
          
            # Pre-allocate arrays for 3-day averaged results
            xfm_3d_3day = zeros(Float32, nx, ny, nz, nt3)
            yfm_3d_3day = zeros(Float32, nx, ny, nz, nt3)
            zfm_3d_3day = zeros(Float32, nx, ny, nz, nt3)
            
            # Compute 3-day averages
            for t in 1:nt3
                t_start = (t-1) * 72 + 1
                t_end = min(t * 72, nt)
            
                
                xfm_3d_3day[:, :, :, t] .= mean(xflx_3d[:, :, :, t_start:t_end], dims=4)
                yfm_3d_3day[:, :, :, t] .= mean(yflx_3d[:, :, :, t_start:t_end], dims=4)
                zfm_3d_3day[:, :, :, t] .= mean(zflx_3d[:, :, :, t_start:t_end], dims=4)
            end
            
            # Save the 3-day averaged fluxes (before depth integration)
            open(joinpath(base2, "xflux", "xflx_3day_$suffix.bin"), "w") do io
                write(io, xfm_3d_3day)
            end
            open(joinpath(base2, "yflux", "yflx_3day_$suffix.bin"), "w") do io
                write(io, yfm_3d_3day)
            end
            open(joinpath(base2, "zflux", "zflx_3day_$suffix.bin"), "w") do io
                write(io, zfm_3d_3day)
            end
            
            println("  Saved 3-day averaged fluxes with $nt3 time windows")
        end
        
        println("Completed tile: $suffix")
    
    end
end


