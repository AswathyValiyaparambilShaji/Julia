using DSP, MAT, Statistics, Printf,  FilePathsBase, LinearAlgebra, TOML, CairoMakie


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter


# Load configuration
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base = cfg["base_path"]
base2 = cfg["base_path2"]


# Create output directories
mkpath(joinpath(base2,"xflux"))
mkpath(joinpath(base2, "yflux"))
mkpath(joinpath(base2, "zflux"))
mkpath(joinpath(base2, "FDiv"))


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


# Initialize global arrays
DS = zeros(NX, NY)
FDiv = zeros(NX, NY)
Conv = zeros(NX, NY)


# Process all tiles
for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
        suffix2 = @sprintf("%02dx%02d_%d", xn, yn, buf-2)

        #println(joinpath(base2, "FDiv", "FDiv_$(suffix2).bin"))
        # Read flux divergence field
        fxD = open(joinpath(base2, "FDiv", "FDiv_$(suffix2).bin"), "r") do io
            nbytes = (nx-2) * (ny-2) * sizeof(Float64)
            raw_bytes = read(io, nbytes)
            raw_data = reinterpret(Float64, raw_bytes)
            reshape(raw_data, nx-2, ny-2)
        end


        # Read conversion field
        C = open(joinpath(base2, "Conv", "Conv_$(suffix2).bin"), "r") do io
            nbytes = (nx-2) * (ny-2) * sizeof(Float64)
            raw_bytes = read(io, nbytes)
            raw_data = reinterpret(Float64, raw_bytes)
            reshape(raw_data, nx-2, ny-2)
        end
        
        # Calculate dissipation
        dispn = C .- fxD
        
        # Calculate tile positions in global grid
        xs = (xn - 1) * tx + 1  
        xe = xs + tx + (2 * buf) - 1  
        ys = (yn - 1) * ty + 1  
        ye = ys + ty + (2 * buf) - 1  
        
        # Update global arrays (remove buffer zones)
        DS[xs+2:xe-2, ys+2:ye-2] .= dispn[2:end-1, 2:end-1] 
        FDiv[xs+2:xe-2, ys+2:ye-2] .= fxD[2:end-1, 2:end-1] 
        Conv[xs+2:xe-2, ys+2:ye-2] .= C[2:end-1, 2:end-1] 
    end
end


# Create visualization
fig = Figure(resolution=(1200, 400))


# Subplot 1: Conversion
ax1 = Axis(fig[1, 1], title="Conversion  (W/m²) ", xlabel="Longitude[°]", ylabel="Latitude[°]")
ax1.limits[] =((minimum(lon), maximum(lon)), (minimum(lat), maximum(lat))) #(193.0,194.2,24.0, 25.4)
hm = CairoMakie.heatmap!(ax1, lon, lat, Conv; interpolate=false, colorrange=(-0.05, 0.05), colormap=Reverse(:RdBu))


# Subplot 2: Flux Divergence
ax2 = Axis(fig[1, 2], title="∇.F (W/m²) ", xlabel="Longitude[°]")
ax2.limits[] = ((minimum(lon), maximum(lon)), (minimum(lat), maximum(lat))) #(193.0,194.2,24.0, 25.4)
hm1 = CairoMakie.heatmap!(ax2, lon, lat, FDiv; interpolate=false, colorrange=(-0.05, 0.05), colormap=Reverse(:RdBu))


# Subplot 3: Dissipation
ax3 = Axis(fig[1, 3], title="Dissipation (W/m²) ", xlabel="Longitude[°]")
ax3.limits[] = ((minimum(lon), maximum(lon)), (minimum(lat), maximum(lat))) #(193.0,194.2,24.0, 25.4)
hm2 = CairoMakie.heatmap!(ax3, lon, lat, DS; interpolate=false, colorrange=(-0.05, 0.05), colormap=Reverse(:RdBu))


# Add colorbar and display
Colorbar(fig[1, 4], hm2, label=" (W/m²)")
display(fig)


# Save figure
FIGDIR        = cfg["fig_base"]
save(joinpath(FIGDIR, "CCDFull_SM_I_v1.png"), fig)




