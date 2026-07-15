using TOML, Printf, Statistics, CairoMakie


# -----------------------------------------------------------------------
# Config / dimensions (mirrors the processing script)
# -----------------------------------------------------------------------
config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base2 = cfg["base_path_nt"]


buf = 3
tx, ty = 47, 66
nx = tx + 2*buf
ny = ty + 2*buf


dt  = 25
dto = 144
Tts = 366192
nt  = div(Tts, dto)
nt_chunk = 72
n_chunks = div(nt, nt_chunk)


ring_steps   = nt_chunk
t_safe_start = ring_steps + 1
t_safe_end   = nt - ring_steps


safe_chunks = [c for c in 1:n_chunks
              if (c-1)*nt_chunk + 1 >= t_safe_start &&
                 c*nt_chunk          <= t_safe_end]
n_safe = length(safe_chunks)


# -----------------------------------------------------------------------
# Tile 1 = first (xn, yn) from config
# -----------------------------------------------------------------------
xn1 = cfg["xn_start"]
yn1 = cfg["yn_start"]
suffix = @sprintf("%02dx%02d_%d", xn1, yn1, buf)
println("Reading tile: $suffix")


# -----------------------------------------------------------------------
# File paths for the three 3-day methods
# -----------------------------------------------------------------------
path_cfd  = joinpath(base2, "TE_t_3day",     "te_t_3day_nt_$(suffix).bin")
path_fd   = joinpath(base2, "TE_t_3day_fd",  "te_t_3day_nt_$(suffix)_fd.bin")
path_bulk = joinpath(base2, "TE_t_3day_bulk","te_t_3day_nt_$(suffix)_bulk.bin")


function read_3day(path, nx, ny, n)
    open(path, "r") do io
        data = reinterpret(Float32, read(io, nx*ny*n*sizeof(Float32)))
        reshape(data, nx, ny, n)
    end
end


TE_cfd  = read_3day(path_cfd,  nx, ny, n_safe)
TE_fd   = read_3day(path_fd,   nx, ny, n_safe)
TE_bulk = read_3day(path_bulk, nx, ny, n_safe)


# -----------------------------------------------------------------------
# Reduce to a single time series: spatial mean over the interior
# (exclude the buffer rows/cols, since those are ghost/overlap cells)
# -----------------------------------------------------------------------
xr = (buf+1):(nx-buf)
yr = (buf+1):(ny-buf)


ts_cfd  = vec(mean(TE_cfd[xr, yr, :],  dims=(1,2)))
ts_fd   = vec(mean(TE_fd[xr, yr, :],   dims=(1,2)))
ts_bulk = vec(mean(TE_bulk[xr, yr, :], dims=(1,2)))
println(ts_bulk[1:10])

# x-axis: chunk index (or convert to days if you prefer)
chunk_idx = 1:n_safe


# -----------------------------------------------------------------------
# Plot with CairoMakie
# -----------------------------------------------------------------------
fig = Figure(size = (1000, 500))
ax = Axis(fig[1, 1],
    xlabel = "3-day chunk index",
    ylabel = "dE/dt (depth-integrated)",
    title  = "Tile $(suffix): 3-day tendency comparison")


lines!(ax, chunk_idx, ts_cfd,  label = "CFD (centered)", color = :steelblue)
lines!(ax, chunk_idx, ts_fd,   label = "FD (forward)",   color = :orangered)
lines!(ax, chunk_idx, ts_bulk, label = "Bulk endpoint",  color = :black, linestyle = :dash)


axislegend(ax, position = :rt)


outfile = joinpath(base2, "TE_t_3day_comparison_$(suffix).png")
save(outfile, fig)
println("Saved plot to: $outfile")


fig




