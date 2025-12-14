module FluxUtils

using DSP

export read_bin, bandpassfilter

"""
    read_bin(path, dims::NTuple{N,Int})

Read a Float32 binary file at `path` and reshape to `dims`.
"""
function read_bin(filename::AbstractString, dims::NTuple{N,Int}) where {N}
    open(filename, "r") do io
        data = read!(io, Vector{Float32}(undef, prod(dims)))
        reshape(data, dims)
    end
end


"""
    bandpass_butter(y, Tl, Th, dt, N, nt)

Zero-phase Butterworth bandpass along the **last dimension** of `y`.

Args
- `y`  : vector / array with time in the last dim
- `Tl` : smallest period to keep
- `Th` : largest period to keep
- `dt` : sampling interval
- `N`  : Butterworth order
- `nt` : length of the time dimension (must equal size(y, end))

Returns an array with the same shape as `y`.
"""
function bandpassfilter(y, Tl, Th, dt, N, nt)
    @assert size(y, ndims(y)) == nt "nt must equal size(y, end)"
    fs = 1/dt
    f1, f2 = 1/Th, 1/Tl
    #@assert 0 < f1 < f2 < fs/2 "Require 1/Th < 1/Tl < fs/2"

    b = digitalfilter(Bandpass(f1, f2), Butterworth(N); fs=fs)

    nd  = ndims(y)
    perm = (nd, 1:nd-1...)                 # move time(last) -> first
    yp   = permutedims(y, perm)            # (nt, ...)

    f = x -> reshape(filtfilt(b, vec(x)), size(x))
    yf  = mapslices(f, yp; dims=1)         # filter along first axis

    permutedims(yf, invperm(perm))         # restore original axes
end

end # module







