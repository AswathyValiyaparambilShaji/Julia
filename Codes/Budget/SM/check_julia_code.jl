using Printf, TOML


config_file = get(ENV, "JULIA_CONFIG", joinpath(@__DIR__, "..", "..", "..", "config", "run_debug.toml"))
cfg = TOML.parsefile(config_file)
base  = cfg["base_path"]
base2 = cfg["base_path2"]


buf = 3
tx, ty = 47, 66
nx = tx + 2*buf
ny = ty + 2*buf
nz = 88
dto = 144
Tts = 366192
nt = div(Tts, dto)


xn, yn = 1, 1
suffix = @sprintf("%02dx%02d_%d", xn, yn, buf)
sz = nx * ny * nz * nt


println("=== SYSTEM CHECK: $suffix ===")
println("base2: ", base2)
println()


for (label, path) in [
    ("fu",   joinpath(base2, "UVW_F", "fu_$suffix.bin")),
    ("fv",   joinpath(base2, "UVW_F", "fv_$suffix.bin")),
    ("fw",   joinpath(base2, "UVW_F", "fw_$suffix.bin")),
    ("KE",   joinpath(base2, "KE",    "ke_t_sm_$suffix.bin")),
    ("b_IT", joinpath(base2, "b",     "b_t_sm_$suffix.bin")),
]
    if !isfile(path)
        println("$label: FILE NOT FOUND → $path")
        continue
    end
    data = open(path, "r") do io
        reinterpret(Float32, read(io, sz * sizeof(Float32)))
    end
    finite_vals = filter(isfinite, data)
    println("$label:")
    println("  path:  $path")
    println("  size:  $(filesize(path)) bytes")
    println("  range: $(extrema(finite_vals))")
    println("  NaNs:  $(count(isnan, data))")
    println("  zeros: $(count(iszero, data))")
    println()
end




