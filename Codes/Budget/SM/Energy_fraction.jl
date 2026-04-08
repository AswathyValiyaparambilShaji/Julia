using DSP, Statistics, Printf, TOML


include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin


# Load configuration
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base2 = cfg["base_path2"]


# --- Domain & grid ---
NX, NY = 288, 468


# --- Tile parameters ---
buf = 3
tx, ty = 47, 66
nx = tx + 2*buf
ny = ty + 2*buf


# Time parameters
dt = 25
dto = 144
Tts = 366192
nt = div(Tts, dto)


# Initialize global arrays
Conv     = zeros(NX, NY)
FDiv     = zeros(NX, NY)
SP_H_full = zeros(NX, NY)
SP_V_full = zeros(NX, NY)
BP_full  = zeros(NX, NY)


println("Loading energy budget terms...")


for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix  = @sprintf("%02dx%02d_%d", xn, yn, buf)
        suffix2 = @sprintf("%02dx%02d_%d", xn, yn, buf-2)


        # Flux Divergence (buf-2 sized, no extra buffer strip needed)
        fxD = Float64.(open(joinpath(base2, "FDiv", "FDiv_$(suffix2).bin"), "r") do io
            reshape(reinterpret(Float32, read(io, (nx-2)*(ny-2)*sizeof(Float32))), nx-2, ny-2)
        end)


        # Conversion
        C = Float64.(open(joinpath(base2, "Conv", "Conv_$(suffix2).bin"), "r") do io
            reshape(reinterpret(Float32, read(io, (nx-2)*(ny-2)*sizeof(Float32))), nx-2, ny-2)
        end)


        # Horizontal Shear Production
        sp_h_mean = Float64.(open(joinpath(base2, "SP_H_old", "sp_h_mean_$suffix.bin"), "r") do io
            reshape(reinterpret(Float32, read(io, nx*ny*sizeof(Float32))), nx, ny)
        end)


        # Vertical Shear Production
        sp_v_mean = Float64.(open(joinpath(base2, "SP_V_old", "sp_v_mean_$suffix.bin"), "r") do io
            reshape(reinterpret(Float32, read(io, nx*ny*sizeof(Float32))), nx, ny)
        end)


        # Buoyancy Production
        bp_mean = Float64.(open(joinpath(base2, "BP_old", "bp_mean_$suffix.bin"), "r") do io
            reshape(reinterpret(Float32, read(io, nx*ny*sizeof(Float32))), nx, ny)
        end)


        # Tile positions in global grid
        xs = (xn - 1) * tx + 1
        xe = xs + tx + 2*buf - 1
        ys = (yn - 1) * ty + 1
        ye = ys + ty + 2*buf - 1


        # FDiv and Conv already have the 1-cell interior strip removed (suffix2)
        Conv[xs+2:xe-2, ys+2:ye-2] .= C[2:end-1, 2:end-1]
        FDiv[xs+2:xe-2, ys+2:ye-2] .= fxD[2:end-1, 2:end-1]


        # Advection/shear/BP terms: strip buf on each side
        SP_H_full[xs+2:xe-2, ys+2:ye-2] .= sp_h_mean[buf:nx-buf+1, buf:ny-buf+1]
        SP_V_full[xs+2:xe-2, ys+2:ye-2] .= sp_v_mean[buf:nx-buf+1, buf:ny-buf+1]
        BP_full[xs+2:xe-2, ys+2:ye-2]   .= bp_mean[buf:nx-buf+1, buf:ny-buf+1]


        println("  Completed tile $suffix")
    end
end


# ==========================================================
# ============ AREA AVERAGES & FRACTIONS ===================
# ==========================================================


mean_C    = mean(Conv)
mean_FDiv = mean(FDiv)
mean_Ps   = mean(SP_H_full .+ SP_V_full)
mean_Pb   = mean(BP_full)


frac_FDiv    = (mean_FDiv / mean_C) * 100.0
frac_PsPb    = ((mean_Ps + mean_Pb) / mean_C) * 100.0


println("\n=== Area-Averaged Energy Budget Fractions ===")
println(@sprintf("  ⟨C⟩              = %.4e W/m²", mean_C))
println(@sprintf("  ⟨∇·F⟩           = %.4e W/m²", mean_FDiv))
println(@sprintf("  ⟨Ps⟩            = %.4e W/m²", mean_Ps))
println(@sprintf("  ⟨Pb⟩            = %.4e W/m²", mean_Pb))
println()
println(@sprintf("  ∇·F / C          = %.1f%%  (flux divergence fraction)", frac_FDiv))
println(@sprintf("  (Ps + Pb) / C    = %.1f%%  (wave–mean flow interaction fraction)", frac_PsPb))


