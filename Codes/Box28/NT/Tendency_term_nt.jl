using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML, Dates
#using CairoMakie, SparseArrays


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base = cfg["base_path_V2"]
base2 = (joinpath(base, "NT"))    
t_origin   = DateTime(2023, 5, 1, 0, 0, 0)
t_wk_start = DateTime(2023, 5, 4, 0, 0, 0)
t_wk_end   = DateTime(2023, 5, 18, 18, 0, 0)
wk_start   = Int(Dates.Hour(t_wk_start - t_origin).value) + 1
wk_end     = Int(Dates.Hour(t_wk_end   - t_origin).value) + 1

for d in ["TE_t", "TE_t_3day","TE_t_wkly"]
    mkpath(joinpath(base2, d))
end

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
nt_chunk = 72
n_chunks = div(nt,nt_chunk)
ts = 72
nt_avg = div(nt, ts)

# --- Thickness & constants ---
thk =(open(joinpath(base, "hFacC",  "delR.bin"), "r") do io
                raw = read(io,  NZ * sizeof(Float32))
                ntoh.(reshape(reinterpret(Float32, raw), NZ))
            end)

DRF  = thk[1:nz]
sum(thk)
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)
g = 9.81
rho0  = 1027.5
dt_output =3600

for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]
        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
        println("Starting tile: $suffix")


        APE = Float64.(open(joinpath(base2, "APE", "APE_t_nt_$suffix.bin"), "r") do io
            reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
        end)
        ke_t = Float64.(open(joinpath(base2, "KE", "ke_t_nt_$suffix.bin"), "r") do io
            reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt)
        end)


        TE = APE .+ ke_t
        APE = nothing; ke_t = nothing; GC.gc()


        dEdt = zeros(Float64, nx, ny, nz, nt)
        dEdt[:, :, :, 1]    = (TE[:, :, :, 2]  .- TE[:, :, :, 1])    ./ dt_output
        for t in 2:nt-1
            dEdt[:, :, :, t] = (TE[:, :, :, t+1] .- TE[:, :, :, t-1]) ./ (2 * dt_output)
        end
        dEdt[:, :, :, nt]   = (TE[:, :, :, nt] .- TE[:, :, :, nt-1]) ./ dt_output
        dEdt[isnan.(dEdt)] .= 0.0
        TE = nothing; GC.gc()


        dEdt_di = zeros(Float64, nx, ny, nt)
        for t in 1:nt, k in 1:nz
            dEdt_di[:, :, t] .+= dEdt[:, :, k, t] .* DRF[k]
        end
        dEdt = nothing; GC.gc()


        open(joinpath(base2, "TE_t", "te_t_nt_$suffix.bin"), "w") do io
            write(io, Float32.(dropdims(mean(dEdt_di, dims=3), dims=3)))
        end


        TE_3day = zeros(Float32, nx, ny, n_chunks)
        for c in 1:n_chunks
            t1 = (c-1)*nt_chunk + 1
            t2 = c*nt_chunk
            TE_3day[:, :, c] = Float32.(dropdims(mean(dEdt_di[:, :, t1:t2], dims=3), dims=3))
        end
        open(joinpath(base2, "TE_t_3day", "te_t_3day_nt_$suffix.bin"), "w") do io
            write(io, TE_3day)
        end
        TE_3day = nothing; GC.gc()


        open(joinpath(base2, "TE_t_wkly", "te_t_wkly_nt_$suffix.bin"), "w") do io
            write(io, Float32.(dropdims(mean(dEdt_di[:, :, wk_start:wk_end], dims=3), dims=3)))
        end


        dEdt_di = nothing; GC.gc()
        println("Completed tile: $suffix")
    end
end




