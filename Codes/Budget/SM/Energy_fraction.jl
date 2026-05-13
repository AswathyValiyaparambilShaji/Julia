using DSP, Statistics, Printf, TOML, MAT


include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin


config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base  = cfg["base_path"]
base2 = cfg["base_path2"]


NX, NY = 288, 468
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
ts = 72                  # timesteps per 3-day period
nt_avg = div(nt, ts)     # number of 3-day periods
nt3 = div(nt, 3*24)      # number of 3-day periods



# --- Thickness & constants ---
thk = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)


rho0 = 1027.5


# Initialize global arrays
Conv         = zeros(NX, NY)
FDiv         = zeros(NX, NY)
U_KE_full    = zeros(NX, NY)
U_PE_full    = zeros(NX, NY)
SP_H_full    = zeros(NX, NY)
SP_V_full    = zeros(NX, NY)
BP_full      = zeros(NX, NY)
ET_full      = zeros(NX, NY)
WPI_full     = zeros(NX, NY)
G_vel_H_full = zeros(NX, NY)
G_vel_V_full = zeros(NX, NY)
G_buoy_full  = zeros(NX, NY)

println("Loading energy budget terms...")


for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix  = @sprintf("%02dx%02d_%d", xn, yn, buf)
        suffix2 = @sprintf("%02dx%02d_%d", xn, yn, buf-2)

        # --- depth from hFacC ---
        hFacC   = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
        DRFfull = hFacC .* DRF3d
        DRFfull[hFacC .== 0] .= 0.0
        H = dropdims(sum(DRFfull, dims=3), dims=3)   # (nx, ny)

        # Grid cell areas
        dx  = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
        dy  = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))
        rac = dx .* dy


        # Flux Divergence
        fxD = Float64.(open(joinpath(base2, "FDiv", "FDiv_$(suffix2).bin"), "r") do io
            reshape(reinterpret(Float32, read(io, (nx-2)*(ny-2)*sizeof(Float32))), nx-2, ny-2)
        end)


        # Conversion
        C = Float64.(open(joinpath(base2, "Conv", "Conv_$(suffix2).bin"), "r") do io
            reshape(reinterpret(Float32, read(io, (nx-2)*(ny-2)*sizeof(Float32))), nx-2, ny-2)
        end)


        # Horizontal Shear Production
        sp_h_mean = Float64.(open(joinpath(base2,"BC", "SP_H", "sp_h_mean_$suffix.bin"), "r") do io
            reshape(reinterpret(Float32, read(io, nx*ny*sizeof(Float32))), nx, ny)
        end)
        
        # --- Read KE Advection ---
        u_ke_mean = Float64.(open(joinpath(base2, "BC","U_KE", "u_ke_mean_$suffix.bin"), "r") do io
            nbytes = nx * ny * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
        end)


        # --- Read PE Advection ---
        u_pe_mean = Float64.(open(joinpath(base2, "BC","U_PE", "u_pe_mean_$suffix.bin"), "r") do io
            nbytes = nx * ny * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
        end)

        # Vertical Shear Production
        sp_v_mean = Float64.(open(joinpath(base2,"BC", "SP_V", "sp_v_mean_$suffix.bin"), "r") do io
            reshape(reinterpret(Float32, read(io, nx*ny*sizeof(Float32))), nx, ny)
        end)


        # Buoyancy Production
        bp_mean = Float64.(open(joinpath(base2, "BP", "bp_mean_$suffix.bin"), "r") do io
            reshape(reinterpret(Float32, read(io, nx*ny*sizeof(Float32))), nx, ny)
        end)

        # --- Read Wind Power Input (with time dimension) ---
        wpi_tile = Float64.(open(joinpath(base2, "WindPowerInput", "wpi_$suffix.bin"), "r") do io
            nbytes = nx * ny * nt * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny, nt)
        end)
        # --- Read Energy Tendency ---
        te_mean = Float64.(open(joinpath(base2, "TE_t", "te_t_mean_$suffix.bin"), "r") do io
            nbytes = nx * ny * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
        end)


        # Time average the WPI
        wpi_mean = mean(wpi_tile, dims=3)[:, :, 1]

        # Tile positions in global grid
        xs = (xn - 1) * tx + 1
        xe = xs + tx + 2*buf - 1
        ys = (yn - 1) * ty + 1
        ye = ys + ty + 2*buf - 1


        # FDiv and Conv (suffix2 files: strip 1 cell on each side)
        Conv[xs+2:xe-2, ys+2:ye-2] .= C[2:end-1, 2:end-1]
        FDiv[xs+2:xe-2, ys+2:ye-2] .= fxD[2:end-1, 2:end-1]


        # buf-stripped terms
        U_KE_full[xs+2:xe-2,    ys+2:ye-2] .= u_ke_mean[buf:nx-buf+1, buf:ny-buf+1]
        U_PE_full[xs+2:xe-2,    ys+2:ye-2] .= u_pe_mean[buf:nx-buf+1, buf:ny-buf+1]
        SP_H_full[xs+2:xe-2, ys+2:ye-2] .= sp_h_mean[buf:nx-buf+1, buf:ny-buf+1]
        SP_V_full[xs+2:xe-2, ys+2:ye-2] .= sp_v_mean[buf:nx-buf+1, buf:ny-buf+1]
        BP_full[xs+2:xe-2, ys+2:ye-2]   .= bp_mean[buf:nx-buf+1, buf:ny-buf+1]
        RAC[xs+2:xe-2, ys+2:ye-2]       .= rac[buf:nx-buf+1, buf:ny-buf+1]
        ET_full[xs+2:xe-2,      ys+2:ye-2] .= te_mean[buf:nx-buf+1,   buf:ny-buf+1]
        WPI_full[xs+2:xe-2,     ys+2:ye-2] .= wpi_mean[buf:nx-buf+1,  buf:ny-buf+1]
        FH[xs+2:xe-2,     ys+2:ye-2] .= H[buf:nx-buf+1, buf:ny-buf+1]



        println("  Completed tile $suffix")
    end
end

Residual  = -(Conv .- TotalFlux .+ SP_H_full .+ SP_V_full .+ BP_full .+ WPI_full .- ET_full)

# ==========================================================
# ============ AREA-WEIGHTED AVERAGES & FRACTIONS ==========
# ==========================================================


valid_mask = RAC .> 0.0
total_area = sum(RAC[valid_mask])


# Area-weighted mean: sum(F * dA) / sum(dA)
# Each grid point is weighted by its cell area (dx*dy),
# so larger cells contribute more to the domain average.
function area_weighted_mean(F::Array{Float64,2}, RAC::Array{Float64,2},
                             mask::BitMatrix, total_area::Float64)
    return sum(F[mask] .* RAC[mask]) / total_area
end


mean_C    = area_weighted_mean(Conv./ (rho0 .* FH),                   RAC, valid_mask, total_area)
mean_FDiv = area_weighted_mean(FDiv./ (rho0 .* FH),                   RAC, valid_mask, total_area)
mean_Ps   = area_weighted_mean((SP_H_full .+ SP_V_full)./ (rho0 .* FH), RAC, valid_mask, total_area)
mean_Pb   = area_weighted_mean(BP_full./ (rho0 .* FH),                RAC, valid_mask, total_area)
mean_KE   = area_weighted_mean(U_KE_full./ (rho0 .* FH),                RAC, valid_mask, total_area)
mean_PE   = area_weighted_mean(U_PE_full./ (rho0 .* FH),                RAC, valid_mask, total_area)
mean_R   = area_weighted_mean(Residual./ (rho0 .* FH),                RAC, valid_mask, total_area)

frac_FDiv = (mean_FDiv / mean_C) * 100.0
frac_PsPb = ((mean_Ps + mean_Pb) / mean_C) * 100.0
frac_A = ((mean_KE + mean_PE) / mean_C) * 100.0
frac_R = ((mean_R) / mean_C) * 100.0


println("\n=== Area-Weighted Energy Budget Fractions ===")
println(  "A fraction :", frac_A)
println("  ⟨∇·F⟩        fraction : ", frac_FDiv)
println("  ⟨Ps⟩ +⟨Pb⟩ fraction : ", frac_PsPb)
println("  ⟨R⟩   fraction : ", frac_R)
println()