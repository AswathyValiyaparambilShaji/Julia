using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML


include(joinpath(@__DIR__, "..","..","..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter


config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..","..","..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base  = cfg["base_path"]
base2 = cfg["base_path2"]


mkpath(joinpath(base2, "KE"))
mkpath(joinpath(base2, "b"))


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


dt  = 25
dto = 144
Tts = 366192
nt  = div(Tts, dto)


# --- Thickness & constants ---
thk   = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF   = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)
g     = 9.8
rho0  = 999.8


# --- Filter parameters (9-15 day bandpass, 1 step sampling) ---
T1, T2, delt, N = 9.0, 15.0, 1.0, 4
fcutlow  = 1 / T2
fcuthigh = 1 / T1
fnq      = 1 / delt
bpf = digitalfilter(Bandpass(fcutlow, fcuthigh), Butterworth(N); fs=fnq)


# --- Loop over tiles ---
for xn in cfg["xn_start"]:cfg["xn_end"]
    for yn in cfg["yn_start"]:cfg["yn_end"]


        suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
        println("Processing tile: $suffix")


        # --- Read density (Float64) ---
        rho = open(joinpath(base, "Density", "rho_in_$suffix.bin"), "r") do io
            nbytes   = nx * ny * nz * nt * sizeof(Float64)
            raw_data = reinterpret(Float64, read(io, nbytes))
            Float64.(reshape(raw_data, nx, ny, nz, nt))
        end


        # --- Read hFacC mask ---
        hFacC = read_bin(joinpath(base, "hFacC", "hFacC_$suffix.bin"), (nx, ny, nz))


        # --- Build thickness arrays ---
        DRFfull = hFacC .* DRF3d                        # (nx, ny, nz)
        depth   = sum(DRFfull, dims=3)                  # (nx, ny, 1)
        DRFfull[hFacC .== 0] .= 0.0


        # --- 4D land mask & 4D thickness/depth ---
        mask4D    = reshape(hFacC .== 0, nx, ny, nz, 1)
        DRFfull4D = repeat(DRFfull, 1, 1, 1, nt)       # (nx, ny, nz, nt)
        depth4D   = repeat(depth,   1, 1, 1, nt)       # (nx, ny,  1, nt)


        # --- Read bandpass-filtered velocities (Float32 on disk) ---
        fu = open(joinpath(base2, "UVW_F", "fu_$suffix.bin"), "r") do io
            nbytes   = nx * ny * nz * nt * sizeof(Float32)
            raw_data = reinterpret(Float32, read(io, nbytes))
            Float64.(reshape(raw_data, nx, ny, nz, nt))
        end


        fv = open(joinpath(base2, "UVW_F", "fv_$suffix.bin"), "r") do io
            nbytes   = nx * ny * nz * nt * sizeof(Float32)
            raw_data = reinterpret(Float32, read(io, nbytes))
            Float64.(reshape(raw_data, nx, ny, nz, nt))
        end


        # --- Bandpass filter density ---
        fr = bandpassfilter(rho, T1, T2, delt, N, nt)


        # ----------------------------------------------------------------
        # Perturbation velocities (baroclinic = bandpassed - depth-mean)
        # Removes barotropic component, consistent with flux code
        # ∫u' dz = 0,  ∫v' dz = 0
        # ----------------------------------------------------------------
        ucA_3d = sum(fu .* DRFfull4D, dims=3) ./ depth4D   # barotropic u (nx,ny,1,nt)
        up_3d  = fu .- ucA_3d                               # baroclinic u'(nx,ny,nz,nt)
        up_3d[repeat(mask4D, 1, 1, 1, nt)] .= 0.0


        vcA_3d = sum(fv .* DRFfull4D, dims=3) ./ depth4D   # barotropic v (nx,ny,1,nt)
        vp_3d  = fv .- vcA_3d                               # baroclinic v'(nx,ny,nz,nt)
        vp_3d[repeat(mask4D, 1, 1, 1, nt)] .= 0.0


        # ----------------------------------------------------------------
        # Perturbation KE = 0.5 * rho0 * (u'^2 + v'^2)
        # Purely baroclinic, consistent with flux F = p'u'
        # ----------------------------------------------------------------
        ke = 0.5 .* rho0 .* (up_3d.^2 .+ vp_3d.^2)        # (nx, ny, nz, nt)
        ke[repeat(mask4D, 1, 1, 1, nt)] .= 0.0


        # ----------------------------------------------------------------
        # Baroclinic density perturbation
        # fr = bandpassed rho (wave band only)
        # subtract depth-mean to enforce ∫rho' dz = 0
        # same operation as velocity — removes barotropic density signal
        # Required for APE consistency in baroclinic IT energy equation:
        #   d/dt(KE + APE) = -∇·F + sources/sinks
        # ----------------------------------------------------------------
        rhoA_3d   = sum(fr .* DRFfull4D, dims=3) ./ depth4D  # barotropic rho (nx,ny,1,nt)
        rho_prime = fr .- rhoA_3d                             # baroclinic rho'(nx,ny,nz,nt)
        rho_prime[repeat(mask4D, 1, 1, 1, nt)] .= 0.0


        # --- Baroclinic buoyancy b = -g * rho' / rho0 ---
        b = (-g .* rho_prime) ./ rho0                         # (nx, ny, nz, nt)
        b[repeat(mask4D, 1, 1, 1, nt)] .= 0.0


        # --- Save outputs ---
        open(joinpath(base2, "KE", "ke_$suffix.bin"), "w") do io
            write(io, Float32.(ke))
        end


        open(joinpath(base2, "b", "b_$suffix.bin"), "w") do io
            write(io, Float32.(b))
        end


        println("Completed tile: $suffix")
    end
end




