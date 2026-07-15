using MAT, Statistics, Printf, LinearAlgebra, TOML, Dates


include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base  = cfg["base_path"]
base2 = cfg["base_path_nt"]


for d in ["TE_t_fd", "TE_t_3day_fd", "TE_t_wkly2_fd",
          "TE_t_bulk", "TE_t_3day_bulk", "TE_t_wkly2_bulk"]
   mkpath(joinpath(base2, d))
end


NX, NY = 288, 468
minlat, maxlat = 24.0, 31.91
minlon, maxlon = 193.0, 199.0
buf = 3
tx, ty = 47, 66
nx = tx + 2*buf
ny = ty + 2*buf
nz = 88
dt  = 25
dto = 144
Tts = 366192
nt  = div(Tts, dto)
nt_chunk = 72
n_chunks = div(nt, nt_chunk)
dt_output = dt * dto


ring_steps = nt_chunk
t_safe_start = ring_steps + 1              # first valid step (1801)
t_safe_end   = nt - ring_steps             # last  valid step (nt-1800)


safe_chunks = [c for c in 1:n_chunks
              if (c-1)*nt_chunk + 1 >= t_safe_start &&
                 c*nt_chunk          <= t_safe_end]


t_origin   = DateTime(2012, 3, 1, 0, 0, 0)
t_wk_start = DateTime(2012,  5, 4, 0, 0, 0)
t_wk_end   = DateTime(2012, 5, 18, 18, 0, 0)
wk_start  = Int(Dates.Hour(t_wk_start - t_origin).value) + 1
wk_end    = Int(Dates.Hour(t_wk_end   - t_origin).value) + 1


thk   = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF   = thk[1:nz]


Threads.@threads for xn in cfg["xn_start"]:cfg["xn_end"]
   for yn in cfg["yn_start"]:cfg["yn_end"]
       suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
       println("Starting tile: $suffix")


       APE = Float64.(open(joinpath(base2, "APE", "APE_t_nt_$suffix.bin"), "r") do io
           reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
       end)
       ke_t = Float64.(open(joinpath(base2, "KE", "ke_t_nt_$suffix.bin"), "r") do io
           reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
       end)


       TE = APE .+ ke_t          # raw total energy, 4D (nx,ny,nz,nt)
       APE = nothing; ke_t = nothing; GC.gc()
       TE[isnan.(TE)] .=0.0

       # =================================================================
       # Depth-integrated raw energy (needed only for the BULK method)
       # =================================================================

       TE_depthint = zeros(Float64, nx, ny, nt)
       for t in 1:nt, k in 1:nz
           TE_depthint[:, :, t] .+= TE[:, :, k, t] .* DRF[k]
       end


       # =================================================================
       # FD METHOD: forward-difference pointwise tendency, then time-average
       # dEdt_fd[t] = (TE[t+1] - TE[t]) / dt_output,  valid for t = 1:nt-1
       # =================================================================
       dEdt_fd = zeros(Float64, nx, ny, nz, nt)
       for t in 1:nt-1
           dEdt_fd[:, :, :, t] = (TE[:, :, :, t+1] .- TE[:, :, :, t]) ./ dt_output
       end
       dEdt_fd[isnan.(dEdt_fd)] .= 0.0
       TE = nothing; GC.gc()


       dEdt_depthint_fd = zeros(Float64, nx, ny, nt)
       for t in 1:nt, k in 1:nz
           dEdt_depthint_fd[:, :, t] .+= dEdt_fd[:, :, k, t] .* DRF[k]
       end
       dEdt_fd = nothing; GC.gc()


       # --- full time average (safe interior; fd step t "belongs" to [t,t+1]) ---
       open(joinpath(base2, "TE_t_fd", "te_t_nt_$(suffix)_fd.bin"), "w") do io
           write(io, Float32.(dropdims(
               mean(dEdt_depthint_fd[:, :, t_safe_start:t_safe_end-1], dims=3), dims=3)))
       end


       # --- 3-day averages ---
       TE_3day_fd = zeros(Float32, nx, ny, length(safe_chunks))
       for (i, c) in enumerate(safe_chunks)
           t1 = (c-1)*nt_chunk + 1
           t2 = c*nt_chunk
           TE_3day_fd[:, :, i] = Float32.(dropdims(
               mean(dEdt_depthint_fd[:, :, t1:t2-1], dims=3), dims=3))
       end
       open(joinpath(base2, "TE_t_3day_fd", "te_t_3day_nt_$(suffix)_fd.bin"), "w") do io
           write(io, TE_3day_fd)
       end
       TE_3day_fd = nothing; GC.gc()


       # --- weekly average ---
       open(joinpath(base2, "TE_t_wkly2_fd", "te_t_wkly_nt_$(suffix)_fd.bin"), "w") do io
           write(io, Float32.(dropdims(
               mean(dEdt_depthint_fd[:, :, wk_start:wk_end-1], dims=3), dims=3)))
       end
       dEdt_depthint_fd = nothing; GC.gc()


       # =================================================================
       # BULK METHOD: dE/dt = (E[t2] - E[t1]) / (t2 - t1), one value per
       # window — computed directly from TE_depthint, no time-averaging
       # =================================================================


       # --- full range ---
       open(joinpath(base2, "TE_t_bulk", "te_t_nt_$(suffix)_bulk.bin"), "w") do io
           dE = TE_depthint[:, :, t_safe_end] .- TE_depthint[:, :, t_safe_start]
           dT = (t_safe_end - t_safe_start) * dt_output
           write(io, Float32.(dE ./ dT))
       end


       # --- 3-day windows ---
       TE_3day_bulk = zeros(Float32, nx, ny, length(safe_chunks))
       for (i, c) in enumerate(safe_chunks)
           t1 = (c-1)*nt_chunk + 1
           t2 = c*nt_chunk
           dE = TE_depthint[:, :, t2] .- TE_depthint[:, :, t1]
           dT = (t2 - t1) * dt_output
           TE_3day_bulk[:, :, i] = Float32.(dE ./ dT)
       end
       open(joinpath(base2, "TE_t_3day_bulk", "te_t_3day_nt_$(suffix)_bulk.bin"), "w") do io
           write(io, TE_3day_bulk)
       end
       TE_3day_bulk = nothing; GC.gc()


       # --- weekly window ---
       open(joinpath(base2, "TE_t_wkly2_bulk", "te_t_wkly_nt_$(suffix)_bulk.bin"), "w") do io
           dE = TE_depthint[:, :, wk_end] .- TE_depthint[:, :, wk_start]
           dT = (wk_end - wk_start) * dt_output
           write(io, Float32.(dE ./ dT))
       end


       TE_depthint = nothing; GC.gc()
       println("Completed tile: $suffix")
   end
end




