using Printf

fname = "/nobackup/kzhang/llc_4320/regions/Moorings/U/U_1996.20230524T060000"
nstat, nlev = 1996, 173

# Files are real*4, big-endian -- read raw bytes, byte-swap, then view as Float32
raw  = reinterpret(UInt32, read(fname))
data = reinterpret(Float32, ntoh.(raw))
u    = reshape(data, nlev, nstat)   # (levels, stations) -- try (nstat, nlev) if this looks wrong

@printf("shape: %s\n", size(u))
@printf("min/max/mean: %.4f / %.4f / %.4f\n", minimum(u), maximum(u), sum(u) / length(u))
println("station 1, first 5 levels: ", u[1:5, 1])
println("station 88, first 5 levels: ", u[1:5, 88])