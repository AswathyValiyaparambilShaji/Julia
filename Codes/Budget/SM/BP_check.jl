using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML, Impute
using CairoMakie


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin


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


# Pick one tile to diagnose (middle of domain)
xn = div(cfg["xn_end"] - cfg["xn_start"], 2) + cfg["xn_start"]
yn = div(cfg["yn_end"] - cfg["yn_start"], 2) + cfg["yn_start"]
suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)


println("Diagnosing tile: $suffix")


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


# --- Mask rho to NaN where hFacC is zero ---
println("Masking density field to NaN where hFacC = 0...")
for t in 1:nt
    for k in 1:nz
        mask = hFacC[:, :, k] .== 0
        rho[mask, k, t] .= NaN
    end
end


println("Rho range after masking: $(extrema(rho[isfinite.(rho)]))")


# --- Calculate cell thicknesses ---
DRFfull = hFacC .* DRF3d
DRFfull[hFacC .== 0] .= 0.0


# --- Calculate 3-day averaged mean buoyancy field B ---
println("Calculating mean buoyancy field...")
B = zeros(Float64, nx, ny, nz, nt_avg)


for i in 1:nt_avg
    t_start = (i-1) * ts + 1
    t_end = min(i * ts, nt)
    
    # Average density over 3-day window, handling NaN
    for x in 1:nx, y in 1:ny, z in 1:nz
        rho_slice = rho[x, y, z, t_start:t_end]
        valid_rho = rho_slice[isfinite.(rho_slice)]
        
        if length(valid_rho) > 0
            rho_avg = mean(valid_rho)
            # Calculate buoyancy: B = -g * (ρ - ρ₀) / ρ₀
            B[x, y, z, i] = -g * (rho_avg - rho0) / rho0
        else
            B[x, y, z, i] = NaN
        end
    end
end


# --- Mask B to NaN where hFacC is zero ---
println("Masking B where hFacC = 0...")
for t in 1:nt_avg
    for k in 1:nz
        mask = hFacC[:, :, k] .== 0
        B[mask, k, t] .= NaN
    end
end


println("B range: $(extrema(B[isfinite.(B)]))")


# --- Calculate mean buoyancy gradients ---
println("Calculating buoyancy gradients...")
B_x = fill(NaN, nx, ny, nz, nt_avg)
B_y = fill(NaN, nx, ny, nz, nt_avg)


# X-gradient: ∂B/∂x
for t in 1:nt_avg
    for k in 1:nz
        B_x[2:end-1, :, k, t] .= (B[3:end, :, k, t] .- B[1:end-2, :, k, t]) ./ 
                                   (dx[2:end-1, :] .+ dx[1:end-2, :])
    end
end


# Y-gradient: ∂B/∂y
for t in 1:nt_avg
    for k in 1:nz
        B_y[:, 2:end-1, k, t] .= (B[:, 3:end, k, t] .- B[:, 1:end-2, k, t]) ./ 
                                   (dy[:, 2:end-1] .+ dy[:, 1:end-2])
    end
end


# --- Mask gradients to NaN where hFacC is not 1 ---
println("Masking gradients where hFacC != 1...")
for t in 1:nt_avg
    for k in 1:nz
        for j in 2:ny-1
            for i in 2:nx-1
                # Mask B_x if cell or x-neighbors are not fully valid
                if hFacC[i-1, j, k] != 1 || hFacC[i, j, k] != 1 || hFacC[i+1, j, k] != 1
                    B_x[i, j, k, t] = NaN
                end
                # Mask B_y if cell or y-neighbors are not fully valid
                if hFacC[i, j-1, k] != 1 || hFacC[i, j, k] != 1 || hFacC[i, j+1, k] != 1
                    B_y[i, j, k, t] = NaN
                end
            end
        end
    end
end


println("B_x range after masking: $(extrema(B_x[isfinite.(B_x)]))")
println("B_y range after masking: $(extrema(B_y[isfinite.(B_y)]))")


# ==========================================================
# =================== VISUALIZATIONS =======================
# ==========================================================


# Pick a time slice (middle of time series)
t_slice = div(nt_avg, 2)
# Pick a depth level (near surface, say k=10)
k_surface = 10
# Pick a point for vertical profile (middle of tile)
i_point = div(nx, 2)
j_point = div(ny, 2)


fig = Figure(size=(1800, 1200))


# --- Row 1: B at surface level ---
ax1 = Axis(fig[1, 1],
          title="B at surface (k=$k_surface, t=$t_slice)",
          xlabel="x index",
          ylabel="y index")
hm1 = heatmap!(ax1, B[:, :, k_surface, t_slice];
              colormap=:balance,
              interpolate=false)
Colorbar(fig[1, 2], hm1, label="B [m/s²]")


# --- Row 1: Depth-integrated B ---
# Before depth integration, convert NaN to zero
B_for_int = replace(B[:, :, :, t_slice], NaN => 0.0)
B_depth_int = dropdims(sum(B_for_int .* DRFfull, dims=3), dims=3)


ax2 = Axis(fig[1, 3],
          title="Depth-integrated B (t=$t_slice)",
          xlabel="x index",
          ylabel="y index")
hm2 = heatmap!(ax2, B_depth_int;
              colormap=:balance,
              interpolate=false)
Colorbar(fig[1, 4], hm2, label="∫B dz [m²/s²]")


# --- Row 2: B_x at surface ---
ax3 = Axis(fig[2, 1],
          title="∂B/∂x at surface (k=$k_surface, t=$t_slice)",
          xlabel="x index",
          ylabel="y index")
hm3 = heatmap!(ax3, B_x[:, :, k_surface, t_slice];
              colormap=Reverse(:RdBu),
              interpolate=false)
Colorbar(fig[2, 2], hm3, label="∂B/∂x [1/s²]")


# --- Row 2: B_y at surface ---
ax4 = Axis(fig[2, 3],
          title="∂B/∂y at surface (k=$k_surface, t=$t_slice)",
          xlabel="x index",
          ylabel="y index")
hm4 = heatmap!(ax4, B_y[:, :, k_surface, t_slice];
              colormap=Reverse(:RdBu),
              interpolate=false)
Colorbar(fig[2, 4], hm4, label="∂B/∂y [1/s²]")


# --- Row 3, Left: Depth-integrated B_x ---
B_x_for_int = replace(B_x[:, :, :, t_slice], NaN => 0.0)
B_x_depth_int = dropdims(sum(B_x_for_int .* DRFfull, dims=3), dims=3)


ax5 = Axis(fig[3, 1],
          title="Depth-integrated ∂B/∂x (t=$t_slice)",
          xlabel="x index",
          ylabel="y index")
hm5 = heatmap!(ax5, B_x_depth_int;
              colormap=Reverse(:RdBu),
              interpolate=false)
Colorbar(fig[3, 2], hm5, label="∫(∂B/∂x) dz")


# --- Row 3, Right: Depth-integrated B_y ---
B_y_for_int = replace(B_y[:, :, :, t_slice], NaN => 0.0)
B_y_depth_int = dropdims(sum(B_y_for_int .* DRFfull, dims=3), dims=3)


ax6 = Axis(fig[3, 3],
          title="Depth-integrated ∂B/∂y (t=$t_slice)",
          xlabel="x index",
          ylabel="y index")
hm6 = heatmap!(ax6, B_y_depth_int;
              colormap=Reverse(:RdBu),
              interpolate=false)
Colorbar(fig[3, 4], hm6, label="∫(∂B/∂y) dz")


display(fig)


# Save figure
FIGDIR = cfg["fig_base"]
save(joinpath(FIGDIR, "B_diagnostics_tile_$(suffix).png"), fig)


println("\n=== Diagnostics complete ===")
println("Tile: $suffix")
println("B statistics:")
println("  Mean: $(mean(B))")
println("  Std:  $(std(B))")
println("  Min:  $(minimum(B))")
println("  Max:  $(maximum(B))")
println("\nB_x statistics:")
println("  Mean: $(mean(B_x))")
println("  Std:  $(std(B_x))")
println("  Min:  $(minimum(B_x))")
println("  Max:  $(maximum(B_x))")
println("\nB_y statistics:")
println("  Mean: $(mean(B_y))")
println("  Std:  $(std(B_y))")
println("  Min:  $(minimum(B_y))")
println("  Max:  $(maximum(B_y))")




