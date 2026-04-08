using DSP, Statistics, Printf, TOML, MAT


include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin


config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base  = cfg["base_path"]
base2 = cfg["base_path2"]


NX, NY = 288, 468
buf = 3
tx, ty = 47, 66
nx = tx + 2*buf
ny = ty + 2*buf


# Global arrays
Conv      = zeros(NX, NY)
FDiv      = zeros(NX, NY)
SP_H_full = zeros(NX, NY)
SP_V_full = zeros(NX, NY)
BP_full   = zeros(NX, NY)
RAC       = zeros(NX, NY)   # actual cell areas from dx*dy


println("Loading energy budget terms...")


for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix  = @sprintf("%02dx%02d_%d", xn, yn, buf)
        suffix2 = @sprintf("%02dx%02d_%d", xn, yn, buf-2)


        # Grid cell areas
        dx  = read_bin(joinpath(base, "DXC/DXC_$suffix.bin"), (nx, ny))
        dy  = read_bin(joinpath(base, "DYC/DYC_$suffix.bin"), (nx, ny))
        rac = dx .* dy

# Avoids holding a Float64 copy of the full nx×ny×nz×nt array in memory.
# --- Read Flux Divergence ---
        fxD = Float64.(open(joinpath(base2, "FDiv", "FDiv_$(suffix2).bin"), "r") do io
            nbytes = (nx-2) * (ny-2) * sizeof(Float32)
            raw_bytes = read(io, nbytes)
            raw_data = reinterpret(Float32, raw_bytes)
            reshape(raw_data, nx-2, ny-2)
        end)
        
        # --- Read Conversion ---
        C = Float64.(open(joinpath(base2, "Conv", "Conv_$(suffix2).bin"), "r") do io
            nbytes = (nx-2) * (ny-2) * sizeof(Float32)
            raw_bytes = read(io, nbytes)
            raw_data = reinterpret(Float32, raw_bytes)
            reshape(raw_data, nx-2, ny-2)
        end)
        
        # --- Read KE Advection ---
        u_ke_mean = Float64.(open(joinpath(base2, "U_KE_old", "u_ke_mean_$suffix.bin"), "r") do io
            nbytes = nx * ny * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
        end)
        
        # --- Read PE Advection ---
        u_pe_mean = Float64.(open(joinpath(base2, "U_PE_old", "u_pe_mean_$suffix.bin"), "r") do io
            nbytes = nx * ny * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
        end)
        
        # --- Read Shear Production ---
        sp_h_mean = Float64.(open(joinpath(base2, "SP_H_old", "sp_h_mean_$suffix.bin"), "r") do io
            nbytes = nx * ny * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
        end)
        
        # Read time-averaged energy tendency
        te_mean = Float64.(open(joinpath(base2, "TE_t", "te_t_mean_$suffix.bin"), "r") do io
            nbytes = nx * ny * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
        end)
        
        sp_v_mean = Float64.(open(joinpath(base2, "SP_V_old", "sp_v_mean_$suffix.bin"), "r") do io
            nbytes = nx * ny * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
        end)
        
        # --- Read Buoyancy Production ---
        bp_mean = Float64.(open(joinpath(base2, "BP_old", "bp_mean_$suffix.bin"), "r") do io
            nbytes = nx * ny * sizeof(Float32)
            reshape(reinterpret(Float32, read(io, nbytes)), nx, ny)
        end)
        
        


        # ---- Tile position in global grid ----
        xs = (xn - 1) * tx + 1
        xe = xs + tx + (2 * buf) - 1
        ys = (yn - 1) * ty + 1
        ye = ys + ty + (2 * buf) - 1


        # ---- Update global arrays (remove buffer zones) ----
        Conv_full[xs+2:xe-2, ys+2:ye-2, :] .= C[2:end-1, 2:end-1, :]
        FDiv_full[xs+2:xe-2, ys+2:ye-2, :] .= fxD[2:end-1, 2:end-1, :]


        U_KE_full[xs+2:xe-2, ys+2:ye-2, :] .= u_ke_3day[buf:nx-buf+1, buf:ny-buf+1, :]
        U_PE_full[xs+2:xe-2, ys+2:ye-2, :] .= u_pe_3day[buf:nx-buf+1, buf:ny-buf+1, :]
        SP_H_full[xs+2:xe-2, ys+2:ye-2, :] .= sp_h_3day[buf:nx-buf+1, buf:ny-buf+1, :]
        SP_V_full[xs+2:xe-2, ys+2:ye-2, :] .= sp_v_3day[buf:nx-buf+1, buf:ny-buf+1, :]
        BP_full[xs+2:xe-2, ys+2:ye-2, :]   .= bp_3day[buf:nx-buf+1, buf:ny-buf+1, :]
        ET_full[xs+2:xe-2, ys+2:ye-2, :]   .= te_3day[buf:nx-buf+1, buf:ny-buf+1, :]
    

        # Static fields
        FH[xs+2:xe-2, ys+2:ye-2]  .= depth[buf:nx-buf+1, buf:ny-buf+1]
        RAC[xs+2:xe-2, ys+2:ye-2] .= rac[buf:nx-buf+1, buf:ny-buf+1]


        println("  Completed tile $suffix")
    end
end


# ==========================================================
# ============ AREA-WEIGHTED AVERAGES & FRACTIONS ==========
# ==========================================================


valid_mask = RAC .> 0.0
total_area = sum(RAC[valid_mask])


wmean(F) = sum(F[valid_mask] .* RAC[valid_mask]) / total_area


mean_C    = wmean(Conv)
mean_FDiv = wmean(FDiv)
mean_Ps   = wmean(SP_H_full .+ SP_V_full)
mean_Pb   = wmean(BP_full)


frac_FDiv = (mean_FDiv / mean_C) * 100.0
frac_PsPb = ((mean_Ps + mean_Pb) / mean_C) * 100.0


println("\n=== Area-Weighted Energy Budget Fractions ===")
println(@sprintf("  ⟨C⟩           = %.4e W/m²", mean_C))
println(@sprintf("  ⟨∇·F⟩        = %.4e W/m²", mean_FDiv))
println(@sprintf("  ⟨Ps⟩         = %.4e W/m²", mean_Ps))
println(@sprintf("  ⟨Pb⟩         = %.4e W/m²", mean_Pb))
println()
println(@sprintf("  ∇·F / C       = %.1f%%  (flux divergence fraction)", frac_FDiv))
println(@sprintf("  (Ps+Pb) / C   = %.1f%%  (wave–mean flow interaction fraction)", frac_PsPb))




