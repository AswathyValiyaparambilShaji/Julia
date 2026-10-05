#=
GaussianFilter2D.jl
-----------------------------------------------------------------------------
2-D Gaussian smoothing (low-pass) for gridded fields, built on the same ideas
as gaussfilt.jl (M. Buijsman, USM DMS, 2026-08-07):


  * Gaussian kernel with standard deviation set by a PHYSICAL length (metres)
    (or by its full width at half maximum, FWHM, with  fwhm = true);
  * kernel truncated at ±truncate·σ (default 4σ, as in gaussfilt.jl);
  * the kernel is RENORMALISED over the points that actually exist, so the
    result is unbiased near edges. In 2-D, "points that don't exist" are both
    points outside the array AND land points (mask = false). This means the
    coast does not pull the smoothed field toward zero.


How the 2-D filter is done
  A 2-D Gaussian is separable:  G(x,y) = G(x)·G(y).
  So the 2-D filter = a 1-D Gaussian along x (every row), then a 1-D Gaussian
  along y (every column) — exactly gaussfilt.jl applied twice. This gives the
  same answer as a full 2-D kernel, at a tiny fraction of the cost.
  The renormalisation is done by filtering both  mask·Z  and  mask  and then
  dividing:  Zsmooth = G*(mask·Z) / G*(mask)   ("normalised convolution").


Grid spacing may vary:
  * dx can be different for every row (e.g. dx = R·cos(lat)·Δlon on a lon-lat grid),
  * y can be non-uniform (any increasing vector of distances, e.g. R·lat in radians).


Works for Real and Complex fields (complex: real and imaginary parts are
smoothed together — this is what you want for tidal harmonic amplitudes).
Multi-threaded: start Julia with  -t N  or set JULIA_NUM_THREADS.


Usage
    include("GaussianFilter2D.jl")
    Zs = gaussfilt2d(Z, mask, dx, y, σ)                    # σ in metres
    Zs = gaussfilt2d(Z, mask, dx, y, L; fwhm = true)       # L = FWHM in metres
    Zs = gaussfilt2d(Z, mask, dx, y, σ; periodic_x = true) # global, wraps in longitude
    Zs = gaussfilt2d(Z, 2e3, 2e3, 50e3)                    # uniform grid, mask = isfinite.(Z)


Helpers
    sigma_from_cutoff(λc)    σ for which the filter passes 50 % amplitude at wavelength λc
    gauss_response(λ, σ)     fraction of amplitude kept by the low-pass at wavelength λ
-----------------------------------------------------------------------------
=#


using Base.Threads


# NaN of the right type (complex NaN must have NaN in both parts)
_nan(::Type{T}) where {T<:Real}    = T(NaN)
_nan(::Type{T}) where {T<:Complex} = T(NaN, NaN)


"""
    gaussfilt2d(Z, mask, dx, y, L; fwhm=false, truncate=4.0, periodic_x=false)


Low-pass filter the matrix `Z` (dim 1 = x, dim 2 = y) with a 2-D Gaussian.


- `mask` : `true` where data are valid (ocean), `false` on land / missing.
- `dx`   : vector (length = size(Z,2)); grid spacing along x for each row j [m].
- `y`    : vector (length = size(Z,2)); increasing along-y distance of row j [m].
- `L`    : Gaussian standard deviation σ [m], or the FWHM if `fwhm = true`.
- `truncate`   : kernel cut at ±truncate·σ.
- `periodic_x` : `true` if dim 1 wraps around (global longitude).


Returns a matrix like `Z` (Float32 / ComplexF32) with `NaN` where `mask` is false.
"""
function gaussfilt2d(Z::AbstractMatrix{T}, mask::AbstractMatrix{Bool},
                     dx::AbstractVector{<:Real}, y::AbstractVector{<:Real}, L::Real;
                     fwhm::Bool = false, truncate::Real = 4.0,
                     periodic_x::Bool = false) where {T<:Number}
    n1, n2 = size(Z)
    size(mask) == size(Z) || throw(DimensionMismatch("mask must have the same size as Z"))
    length(dx) == n2      || throw(DimensionMismatch("dx must have length size(Z,2) = $n2"))
    length(y)  == n2      || throw(DimensionMismatch("y must have length size(Z,2) = $n2"))
    issorted(y)           || throw(ArgumentError("y must be increasing"))


    σ = fwhm ? L / (2 * sqrt(2 * log(2))) : Float64(L)     # FWHM -> σ (as in gaussfilt.jl)
    σ > 0 || return T <: Complex ? ComplexF32.(Z) : Float32.(Z)


    Acc = T <: Complex ? ComplexF64 : Float64   # accumulate in double precision
    Sto = T <: Complex ? ComplexF32 : Float32   # store intermediates in single precision


    # ---------------- pass 1: along x (each row j has its own dx) ----------------
    numx = Matrix{Sto}(undef, n1, n2)       # G_x * (mask·Z)
    denx = Matrix{Float32}(undef, n1, n2)   # G_x * mask
    @threads for j in 1:n2
        s  = σ / dx[j]                                  # σ in grid points for this row
        hw = ceil(Int, truncate * s)                    # half-width in points
        periodic_x && (hw = min(hw, (n1 - 1) ÷ 2))      # never wrap onto itself
        offs = -hw:hw
        k = [exp(-0.5 * (o / s)^2) for o in offs]       # unnormalised weights
        @inbounds for i in 1:n1
            num = zero(Acc); den = 0.0
            for (kk, o) in enumerate(offs)
                ii = i + o
                if periodic_x
                    ii = ii < 1 ? ii + n1 : (ii > n1 ? ii - n1 : ii)
                elseif !(1 <= ii <= n1)
                    continue                            # outside array -> skip (renormalise)
                end
                mask[ii, j] || continue                 # land -> skip (renormalise)
                num += k[kk] * Z[ii, j]
                den += k[kk]
            end
            numx[i, j] = num
            denx[i, j] = den
        end
    end


    # ---------------- pass 2: along y (non-uniform spacing allowed) ----------------
    out = Matrix{Sto}(undef, n1, n2)
    lim = truncate * σ
    @threads for j in 1:n2
        lo = searchsortedfirst(y, y[j] - lim)           # rows inside ±truncate·σ
        hi = searchsortedlast(y,  y[j] + lim)
        num = zeros(Acc, n1); den = zeros(Float64, n1)
        for jj in lo:hi
            w = exp(-0.5 * ((y[jj] - y[j]) / σ)^2)
            @inbounds @simd for i in 1:n1
                num[i] += w * numx[i, jj]
                den[i] += w * denx[i, jj]
            end
        end
        @inbounds for i in 1:n1
            out[i, j] = (mask[i, j] && den[i] > 0) ? Sto(num[i] / den[i]) : _nan(Sto)
        end
    end
    return out
end


"""
    gaussfilt2d(Z, dx, dy, L; kwargs...)


Uniform grid version: constant spacings `dx`, `dy` [m]; mask = isfinite.(Z).
"""
function gaussfilt2d(Z::AbstractMatrix, dx::Real, dy::Real, L::Real; kwargs...)
    n2 = size(Z, 2)
    mask = isfinite.(Z)
    return gaussfilt2d(Z, mask, fill(Float64(dx), n2), Float64(dy) .* (0:n2-1), L; kwargs...)
end


"""
    sigma_from_cutoff(λc)


Gaussian σ whose low-pass response is 0.5 (half amplitude) at wavelength `λc`:
response(λ) = exp(-2π²σ²/λ²)  ⇒  σ = λc·sqrt(ln2/2)/π ≈ 0.187·λc.
"""
sigma_from_cutoff(λc::Real) = λc * sqrt(log(2) / 2) / π


"""
    gauss_response(λ, σ)


Fraction of the amplitude of a wave with wavelength `λ` kept by the low-pass
(the high-pass, Z − lowpass(Z), keeps 1 − this).
"""
gauss_response(λ::Real, σ::Real) = exp(-2π^2 * σ^2 / λ^2)




