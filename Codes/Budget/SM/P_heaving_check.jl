using DSP, MAT, Statistics, Printf, FilePathsBase, LinearAlgebra, TOML, CairoMakie


include(joinpath(@__DIR__, "..", "..", "..", "functions", "FluxUtils.jl"))
using .FluxUtils: read_bin, bandpassfilter


config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg   = TOML.parsefile(config_file)
base  = cfg["base_path"]
base2 = cfg["base_path2"]


buf      = 3
tx, ty   = 47, 66
nx, ny   = tx + 2*buf, ty + 2*buf
nz       = 88
dto      = 144
Tts      = 366192
nt       = div(Tts, dto)
hrs_3day = 72
nt_avg   = div(nt, hrs_3day)


ρ0 = 999.8
g  = 9.8


thk   = matread(joinpath(base, "hFacC", "thk90.mat"))["thk90"]
DRF   = thk[1:nz]
DRF3d = repeat(reshape(DRF, 1, 1, nz), nx, ny, 1)


T1, T2, delt, N_filt = 9.0, 15.0, 1.0, 4


# ── User settings: one tile, one point, one timestep ─────────────────────────
xn   = 1                   # tile x index
yn   = 1                   # tile y index
ix   = 12                  # x point within tile interior (1:tx)
iy   = 10                 # y point within tile interior (1:ty)
it   = 144                 # timestep to inspect
# ─────────────────────────────────────────────────────────────────────────────


suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
println("Single-tile profile check: $suffix  point ($ix,$iy)  timestep $it")


# ── 1. Grid geometry ──────────────────────────────────────────────────────────
hFacC   = read_bin(joinpath(base, "hFacC/hFacC_$suffix.bin"), (nx, ny, nz))
DRFfull = hFacC .* DRF3d
DRFfull[hFacC .== 0] .= 0.0


H     = sum(DRFfull, dims=3)
DRF4d = reshape(DRFfull, nx, ny, nz, 1)
H_4d  = reshape(H, nx, ny, 1, 1)


z_edge = cat(zeros(nx, ny, 1), cumsum(DRFfull, dims=3); dims=3)
za     = 0.5 .* (z_edge[:, :, 1:end-1] .+ z_edge[:, :, 2:end])


mask3D       = hFacC .== 0
mask4D_proto = reshape(mask3D, nx, ny, nz, 1)
mask4D       = repeat(mask4D_proto, 1, 1, 1, nt)


# ── 2. p′ ─────────────────────────────────────────────────────────────────────
rho = Float64.(open(joinpath(base, "Density", "rho_in_$suffix.bin"), "r") do io
    raw = read(io, nx*ny*nz*nt*sizeof(Float64))
    reshape(reinterpret(Float64, raw), nx, ny, nz, nt)
end)


rhob  = bandpassfilter(rho, T1, T2, delt, N_filt, nt)
rho   = nothing; GC.gc()


pres  = g .* cumsum(rhob .* DRF4d, dims=3)
pfz   = cat(zeros(nx, ny, 1, nt), pres; dims=3)
pc    = 0.5 .* (pfz[:, :, 1:end-1, :] .+ pfz[:, :, 2:end, :])
pres  = nothing; pfz = nothing; rhob = nothing; GC.gc()


pp    = pc .- sum(pc .* DRF4d, dims=3) ./ H_4d
pp[mask4D] .= 0.0
pc    = nothing; GC.gc()


# ── 3. Bandpass η ─────────────────────────────────────────────────────────────
eta_raw = Float64.(open(joinpath(base, "Eta", "Eta_$suffix.bin"), "r") do io
    raw = read(io, nx*ny*nt*sizeof(Float32))
    reshape(reinterpret(Float32, raw), nx, ny, nt)
end)


eta     = bandpassfilter(eta_raw, T1, T2, delt, N_filt, nt)
eta_raw = nothing; GC.gc()


# ── 4. N² ─────────────────────────────────────────────────────────────────────
N2p = Float64.(open(joinpath(base, "3day_mean", "N2",
                             "N2_3day_$suffix.bin"), "r") do io
    raw = read(io, nx*ny*nz*nt_avg*sizeof(Float32))
    reshape(reinterpret(Float32, raw), nx, ny, nz, nt_avg)
end)


N2a = zeros(Float64, nx, ny, nz+1, nt_avg)
N2a[:, :, 1,    :] .= N2p[:, :, 1,      :]
N2a[:, :, 2:nz, :] .= N2p[:, :, 1:nz-1, :]
N2a[:, :, nz+1, :] .= N2p[:, :, nz-1,   :]
N2p = nothing


N2c = zeros(Float64, nx, ny, nz, nt_avg)
for k in 1:nz
    N2c[:, :, k, :] .= 0.5 .* (N2a[:, :, k, :] .+ N2a[:, :, k+1, :])
end
N2a = nothing; GC.gc()


N2thr = 1.0e-8
N2c[isnan.(N2c) .| (N2c .< N2thr)] .= N2thr


N2_4d = zeros(Float64, nx, ny, nz, nt)
for b in 1:nt_avg
    ts = (b-1)*hrs_3day + 1
    te = min(b*hrs_3day, nt)
    N2_4d[:, :, :, ts:te] .= N2c[:, :, :, b:b]
end
N2c = nothing; GC.gc()


# ── 5. ζbt ────────────────────────────────────────────────────────────────────
zbt = reshape(eta, nx, ny, 1, nt) .*
      (H_4d .- reshape(za, nx, ny, nz, 1)) ./ H_4d
zbt[mask4D] .= 0.0


# ── 6. pη ─────────────────────────────────────────────────────────────────────
p_eta = .- cumsum(ρ0 .* N2_4d .* zbt .* DRF4d, dims=3)
p_eta[mask4D] .= 0.0
zbt = nothing; N2_4d = nothing; eta = nothing; GC.gc()


# ── 7. p_int = p′ − pη ───────────────────────────────────────────────────────
pint = pp .- p_eta


# ── 8. Extract profiles at (ix, iy) in tile interior, timestep it ────────────
# ix, iy are 1-based within the interior; shift to full-tile indices
gx = ix + buf
gy = iy + buf


# depth axis: positive downward [m]
depth_vec = za[gx, gy, :]          # nz


pp_prof   = pp[gx, gy, :, it]      # p′
peta_prof = p_eta[gx, gy, :, it]   # pη
pint_prof = pint[gx, gy, :, it]    # p′ − pη


# mask land values to NaN for clean plotting
land = mask3D[gx, gy, :]
pp_prof[land]   .= NaN
peta_prof[land] .= NaN
pint_prof[land] .= NaN



# ── 9. Plot ───────────────────────────────────────────────────────────────────
fig = Figure(size = (600, 700))


ax = Axis(fig[1, 1],
    title  = "Pressure profiles  ",
    xlabel = "Pressure  ",
    ylabel = "Depth  [m]",
    yreversed = true)


lines!(ax, pp_prof,   depth_vec, color = :red,   linewidth = 2,
       label = "p′")
lines!(ax, peta_prof, depth_vec, color = :blue,  linewidth = 2,
       label = "pη")
lines!(ax, pint_prof, depth_vec, color = :black, linewidth = 2,
       label = "p′ − pη")
vlines!(ax, [0.0], color = :gray, linestyle = :dash, linewidth = 1)
axislegend(ax, position = :rb)
display(fig)

save(joinpath(base2, "profile_pp_peta_tile$(suffix)_t$(it).png"), fig)
println("Profile plot saved.")




