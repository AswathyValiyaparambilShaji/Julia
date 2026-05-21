using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base = cfg["base_path_V2"]
base2 = (joinpath(base, "NT")) 

mkpath(joinpath(base2, "KE"))
mkpath(joinpath(base2, "b"))


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
# --- Thickness & constants ---
thk =(open(joinpath(base, "hFacC",  "delR.bin"), "r") do io
                raw = read(io,  NZ * sizeof(Float32))
                ntoh.(reshape(reinterpret(Float32, raw), NZ))
            end)

DRF  = thk[1:nz]
sum(thk)
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)
g = 9.81

T1, T2, delt, N = 10.2, 32.2, 1.0, 4

rho0  = 1027.5

# --- Loop over tiles ---
for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]


        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
        println("Processing tile: $suffix")


        # --- Read density (Float64) ---
        rho = open(joinpath(base, "Density", "rho_in_$suffix.bin"), "r") do io
            Float64.(reshape(reinterpret(Float64, read(io, nx*ny*nz*nt*sizeof(Float64))), nx, ny, nz, nt))
        end


        # --- Read hFacC mask ---
        hFacC = read_bin(joinpath(base, "hFacC", "hFacC_v2_$suffix.bin"), (nx, ny, nz))


        # --- Build thickness arrays ---
        DRFfull = hFacC .* DRF3d
        depth   = sum(DRFfull, dims=3)
        DRFfull[hFacC .== 0] .= 0.0
        mask3D  = hFacC .== 0                           # (nx, ny, nz) Bool — reuse for masking


        # --- Bandpass filter density, free rho immediately ---
        fr  = bandpassfilter(rho, T1, T2, delt, N, nt)
        rho = nothing; GC.gc()


        # --- Read fu, compute baroclinic u', free fu ---
        fu     = open(joinpath(base2, "UVW_NT", "fu_nt_$suffix.bin"), "r") do io
            Float64.(reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt))
        end
        ucA    = sum(fu .* DRFfull, dims=3) ./ depth    # (nx, ny, 1, nt) barotropic
        up_3d  = fu .- ucA
        up_3d[repeat(mask3D, 1, 1, 1, nt)] .= 0.0
        fu = ucA = nothing; GC.gc()


        # --- Read fv, compute baroclinic v', free fv ---
        fv     = open(joinpath(base2, "UVW_NT", "fv_nt_$suffix.bin"), "r") do io
            Float64.(reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt))
        end
        vcA    = sum(fv .* DRFfull, dims=3) ./ depth
        vp_3d  = fv .- vcA
        vp_3d[repeat(mask3D, 1, 1, 1, nt)] .= 0.0
        fv = vcA = nothing; GC.gc()


        # --- Read fw, compute baroclinic w', free fw ---
        fw     = open(joinpath(base2, "UVW_NT", "fw_nt_$suffix.bin"), "r") do io
            Float64.(reshape(reinterpret(Float32, read(io, nx*ny*nz*nt*sizeof(Float32))), nx, ny, nz, nt))
        end
        wcA    = sum(fw .* DRFfull, dims=3) ./ depth
        wp_3d  = fw .- wcA
        wp_3d[repeat(mask3D, 1, 1, 1, nt)] .= 0.0
        fw = wcA = nothing; GC.gc()


        # --- Perturbation KE = 0.5 * rho0 * (u'^2 + v'^2 + w'^2) ---
        ke = 0.5 .* rho0 .* (up_3d.^2 .+ vp_3d.^2 .+ wp_3d.^2)
        ke[repeat(mask3D, 1, 1, 1, nt)] .= 0.0
        up_3d = vp_3d = wp_3d = nothing; GC.gc()


        #--- Baroclinic buoyancy b = -g * rho' / rho0 ---
        rho_prime = fr 
        rho_prime[repeat(mask3D, 1, 1, 1, nt)] .= 0.0
        fr = nothing


        b = (-g ./ rho0) .* rho_prime
        b[repeat(mask3D, 1, 1, 1, nt)] .= 0.0
        rho_prime = nothing; GC.gc()

#
        # --- Save outputs ---
        open(joinpath(base2, "KE", "ke_t_nt_$suffix.bin"), "w") do io
            write(io, Float32.(ke))
        end
        ke = nothing


        open(joinpath(base2, "b", "b_t_nt_$suffix.bin"), "w") do io
            write(io, Float32.(b))
        end
        b = nothing


        println("Completed tile: $suffix")
        GC.safepoint()
        GC.gc(true)
    end
end



