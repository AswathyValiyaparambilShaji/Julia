using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML
using CairoMakie, SparseArrays


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base = cfg["base_path_V2"]
base2 = (joinpath(base, "NT"))       


# --- Domain & grid ---
NX, NY = 288, 468
minlat, maxlat = 24.0, 31.91
minlon, maxlon = 193.0, 199.0
lat = range(minlat, maxlat, length=NY)
lon = range(minlon, maxlon, length=NX)
NZ = 173



# --- Tile & time ---
buf = 3
tx, ty = 47, 66
nx = tx + 2*buf
ny = ty + 2*buf
nz = 168
kz = 1
nt = 558

# --- Thickness & constants ---
thk =(open(joinpath(base, "hFacC",  "delR.bin"), "r") do io
                raw = read(io,  NZ * sizeof(Float32))
                ntoh.(reshape(reinterpret(Float32, raw), NZ))
            end)

DRF  = thk[1:nz]
sum(thk)
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)
g = 9.81



g = 9.8




# --- Filter (9-15 day band, 1 step sampling) ---
T1, T2, delt, N = 9.0, 15.0, 1.0, 4
fcutlow, fcuthigh = 1/T2, 1/T1
fnq = 1/delt
bpf = digitalfilter(Bandpass(fcutlow, fcuthigh), Butterworth(N); fs=fnq)




# ============================================================================
# MAIN WORKFLOW
# ============================================================================


    # ========================================================================
    # FULL TIME AVERAGE
    # ========================================================================


    DS   = zeros(NX, NY)
    FDiv = zeros(NX, NY)
    Conv = zeros(NX, NY)


    for xn in cfg["xn_start"]:cfg["xn_end"]
        for yn in cfg["yn_start"]:cfg["yn_end"]
            suffix  = @sprintf("%02dx%02d_%d", xn, yn, buf)
            suffix2 = @sprintf("%02dx%02d_%d", xn, yn, buf-2)


            # Read flux divergence field
            fxD = Float64.(open(joinpath(base2, "FDiv", "FDiv_nt_$(suffix2).bin"), "r") do io
                nbytes = (nx-2) * (ny-2) * sizeof(Float32)
                raw_bytes = read(io, nbytes)
                raw_data = reinterpret(Float32, raw_bytes)
                reshape(raw_data, nx-2, ny-2)
            end)


            # Read conversion field
            C = Float64.(open(joinpath(base2, "Conv", "Conv_nt_$(suffix2).bin"), "r") do io
                nbytes = (nx-2) * (ny-2) * sizeof(Float32)
                raw_bytes = read(io, nbytes)
                raw_data = reinterpret(Float32, raw_bytes)
                reshape(raw_data, nx-2, ny-2)
            end)


            # Calculate dissipation
            dispn = -(C .- fxD)


            # Calculate tile positions in global grid
            xs = (xn - 1) * tx + 1
            xe = xs + tx + (2 * buf) - 1
            ys = (yn - 1) * ty + 1
            ye = ys + ty + (2 * buf) - 1


            # Update global arrays (remove buffer zones)
            DS[xs+2:xe-2,   ys+2:ye-2] .= -dispn[2:end-1, 2:end-1]
            FDiv[xs+2:xe-2, ys+2:ye-2] .= fxD[2:end-1,   2:end-1]
            Conv[xs+2:xe-2, ys+2:ye-2] .= C[2:end-1,     2:end-1]
        end
    end


    # Create visualization
    fig = Figure(resolution=(900, 400))


    ax1 = Axis(fig[1, 1], title="Conversion  (W/m²)", xlabel="Longitude[°]", ylabel="Latitude[°]")
    ax1.limits[] = ((minimum(lon), maximum(lon)), (minimum(lat), maximum(lat)))
    hm = CairoMakie.heatmap!(ax1, lon, lat, Conv; interpolate=false, colorrange=(-0.03, 0.03), colormap=:bwr)


    ax2 = Axis(fig[1, 2], title="∇.F (W/m²)", xlabel="Longitude[°]",yticklabelsvisible=false)
    ax2.limits[] = ((minimum(lon), maximum(lon)), (minimum(lat), maximum(lat)))
    hm1 = CairoMakie.heatmap!(ax2, lon, lat, FDiv; interpolate=false, colorrange=(-0.03, 0.03), colormap=:bwr)


    ax3 = Axis(fig[1, 3], title="Dissipation (W/m²)", xlabel="Longitude[°]",yticklabelsvisible=false)
    ax3.limits[] = ((minimum(lon), maximum(lon)), (minimum(lat), maximum(lat)))
    hm2 = CairoMakie.heatmap!(ax3, lon, lat, DS; interpolate=false, colorrange=(-0.03, 0.03), colormap=:bwr)


    Colorbar(fig[1, 4], hm2, label=" (W/m²)")
    display(fig)


    FIGDIR = cfg["fig_base"]
    save(joinpath(FIGDIR, "CCDFull_NS_nt_v1.png"), fig)
    println("Figure saved: $(joinpath(FIGDIR, "CCDFull_NS_nt_v1.png"))")








