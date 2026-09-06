using Printf, CairoMakie


fname = "/nobackup/kzhang/llc_4320/regions/Moorings/U/U_1996.20230524T060000"
nstat, nlev = 1996, 173


# Files are real*4, big-endian -- read raw bytes, byte-swap, then view as Float32
raw  = reinterpret(UInt32, read(fname))
data = reinterpret(Float32, ntoh.(raw))
u    = reshape(data, nlev, nstat)   # (levels, stations) -- try (nstat, nlev) if profiles look like noise


@printf("shape: %s\n", size(u))
@printf("min/max/mean: %.4f / %.4f / %.4f\n", minimum(u), maximum(u), sum(u) / length(u))


fig = Figure()
ax = Axis(fig[1, 1], xlabel="level index", ylabel="U (m/s)", title="U profile check")
lines!(ax, 1:nlev, u[:, 1],  label="station 1",  linewidth=2)
lines!(ax, 1:nlev, u[:, 88], label="station 88", linewidth=2)
axislegend(ax)


save("U_profile_check.png", fig)
println("Saved plot to U_profile_check.png")




