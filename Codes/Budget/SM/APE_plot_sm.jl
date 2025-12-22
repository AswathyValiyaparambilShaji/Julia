using DSP, MAT, Statistics, Printf, Plots, FilePathsBase, LinearAlgebra, TOML
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


for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]


        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)


        hFacC = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"),
                         (nx, ny, nz))


        DRFfull = DRF3d .* hFacC
        DRFfull[hFacC .== 0] .= 0


        APE = open(joinpath(base2, "APE", "APE_t_sm_$suffix.bin"), "r") do io
            nbytes = nx * ny * nz * nt * sizeof(Float64)
            reshape(reinterpret(Float64, read(io, nbytes)),
                    nx, ny, nz, nt)
        end


# Time average ignoring NaN - replace NaN with 0 for sum
APE_clean = replace(APE, NaN => 0.0)
APE_sum = sum(APE_clean, dims=4)
APE_count = sum(.!isnan.(APE), dims=4)


# Avoid division by zero
aped = zeros(Float64, nx, ny, nz)
for i in 1:nx, j in 1:ny, k in 1:nz
    if APE_count[i,j,k,1] > 0
        aped[i,j,k] = APE_sum[i,j,k,1] / APE_count[i,j,k,1]
    else
        aped[i,j,k] = NaN
    end
end
        # Depth integrate ignoring NaN
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


println("APE_full range: ",
        minimum(skipmissing(APE_full)),
        " to ",
        maximum(skipmissing(APE_full)))


fig = Figure(size=(600, 500))


ax = Axis(fig[1, 1],
          title="Depth-Integrated Time-Averaged APE",
          xlabel="Longitude [°]",
          ylabel="Latitude [°]",
          limits=(193.0, 193.9, 24.0, 25.1))


hm = CairoMakie.heatmap!(ax, lon, lat, APE_full;
                         interpolate=false,
                         colormap=:jet,
                         colorrange=(0, 5000))


Colorbar(fig[1, 2], hm, label="APE [J/m²]")


display(fig)


# Find positions where APE_full > 10000
high_ape_mask = APE_full .> 10000
high_ape_indices = findall(high_ape_mask)


if !isempty(high_ape_indices)
    println("\nFound $(length(high_ape_indices)) locations with APE > 10000:")
    
    for idx in high_ape_indices[1:min(10, length(high_ape_indices))]  # Show first 10
        i, j = idx.I
        ape_val = APE_full[i, j]
        lat_val = lat[j]
        lon_val = lon[i]
        
        println("  Lon: $(round(lon_val, digits=3))°, Lat: $(round(lat_val, digits=3))°, APE: $(round(ape_val, digits=1)) J/m²")
        println("    Grid index: (i=$i, j=$j)")
    end
    
    # Pick one location for detailed analysis
    test_idx = high_ape_indices[1]
    test_i, test_j = test_idx.I
    test_lon = lon[test_i]
    test_lat = lat[test_j]
    
    println("\n=== Analyzing location: Lon=$(round(test_lon, digits=3))°, Lat=$(round(test_lat, digits=3))° ===")
    
    # Find which tile this belongs to
    test_xn = div(test_i - 1, tx) + 1
    test_yn = div(test_j - 1, ty) + 1
    test_suffix = @sprintf("%02dx%02d_%d", test_xn, test_yn, buf)
    
    # Local indices within tile (including buffer)
    local_i = mod(test_i - 1, tx) + 1 + buf
    local_j = mod(test_j - 1, ty) + 1 + buf
    
    println("Tile: $test_suffix, Local indices: (i=$local_i, j=$local_j)")
    
else
    println("\nNo locations found with APE > 10000")
    println("Maximum APE: $(maximum(skipmissing(APE_full)))")
end




    # Local indices within tile (including buffer)
    local_i = 41
    local_j = 19
    
    println("Tile: $test_suffix")
    println("Local indices in tile: i=$local_i, j=$local_j")
    
    # Read APE for this tile
    APE_tile = open(joinpath(base2, "APE", "APE_t_sm_$test_suffix.bin"), "r") do io
        nbytes = nx * ny * nz * nt * sizeof(Float64)
        reshape(reinterpret(Float64, read(io, nbytes)), nx, ny, nz, nt)
    end
    
    # Extract APE at this point: (nz, nt)
    ape_point = APE_tile[local_i, local_j, :, :]
    
    println("\nAPE at point statistics:")
    println("  Total elements: $(length(ape_point))")
    println("  Finite values: $(sum(isfinite.(ape_point)))")
    println("  NaN values: $(sum(isnan.(ape_point)))")
    if sum(isfinite.(ape_point)) > 0
        println("  Range: $(extrema(ape_point[isfinite.(ape_point)]))")
    end
        
    # ===== TIME AVERAGE AT THIS POINT =====
    # Replace NaN with 0 for summing
    ape_point_clean = replace(ape_point, NaN => 0.0)
    ape_point_sum = sum(ape_point_clean, dims=2)  # Sum over time (dim 2)
    ape_point_count = sum(.!isnan.(ape_point), dims=2)  # Count valid values

    ape_tavg = zeros(Float64, nz)
    for k in 1:nz
        if ape_point_count[k, 1] > 0
            ape_tavg[k] = ape_point_sum[k, 1] / ape_point_count[k, 1]
        else
            ape_tavg[k] = NaN
        end
    end



    # Select 5 time steps evenly distributed
    time_steps = round.(Int, range(1, nt, length=5))
    
    # Plot depth profiles
    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1],
              xlabel="APE [J/m³]",
              ylabel="Depth level",
              title="APE Depth Profiles at Lon=$(round(test_lon, digits=3))°, Lat=$(round(test_lat, digits=3))°",
              yreversed=true)
    
    colors = [:blue, :red, :green, :orange, :purple]
    
    for (idx, t) in enumerate(time_steps)
        ape_profile = ape_point[:, t]
        valid = isfinite.(ape_profile)
        
        if sum(valid) > 0
            lines!(ax, ape_profile[valid], (1:nz)[valid], 
                   label="t=$t", 
                   color=colors[idx],
                   linewidth=2)
        end
    end
    
    axislegend(ax, position=:rb)
   
    


    # ===== PLOT 2: TIME-AVERAGED =====
    ax2 = Axis(fig[1, 2],
               xlabel="APE [J/m³]",
               ylabel="Depth level",
               title="Time-Averaged APE Depth Profile\nLon=$(round(test_lon, digits=3))°, Lat=$(round(test_lat, digits=3))°",
               yreversed=true)
    
    valid = isfinite.(ape_tavg)
    if sum(valid) > 0
        lines!(ax2, ape_tavg[valid], (1:nz)[valid], 
               color=:black,
               linewidth=3)
    end
    
    display(fig)
    
