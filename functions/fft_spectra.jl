"""
    fft_spectra(t, y; tukeycf=0.5, numwin=1, linfit=true, prewhit=false)

Performs simple spectral analysis similar to the MATLAB implementation fft_spectra2.m.

# Arguments
- `t::Vector{Float64}`: Time vector
- `y::Vector{Float64}`: Data vector
- `tukeycf::Float64`: Tukey window parameter (1=hanning, 0=boxcar)
- `numwin::Int`: Number of overlapping windows (50% overlap)
- `linfit::Bool`: Remove linear trend
- `prewhit::Bool`: Apply prewhitening

# Returns
- `period::Vector{Float64}`
- `freq::Vector{Float64}`
- `power::Vector{Float64}`

# info
Maarten Buijsman, USM, 2025-8-4; converted from fft_spectra2.m using chatgpt
https://chatgpt.com/c/6890bc81-9f54-800d-910c-bae1637ec9ac
"""
# new (Julia/DSP idiom)
using DSP: tukey, hann
using FFTW
#window = tukeycf == 1 ? hann(nt) :
 #        tukeycf == 0 ? ones(nt) : tukey(nt, tukeycf)



function fft_spectra(t::AbstractVector{<:Real}, y::AbstractVector{<:Real};
    tukeycf::Float64=0.5, numwin::Int=1, linfit::Bool=true, prewhit::Bool=false)

    # Ensure even length
    if isodd(length(t))
        t = t[1:end-1]
        y = y[1:end-1]
    end

    dt = t[2] - t[1]
    nt1 = length(t)
    inw = floor(Int, nt1 ÷ (numwin + 1))  # half-length of each window

    # Window indices
    windows = []
    start_idx = 1
    for i in 1:numwin
        push!(windows, start_idx:(start_idx + 2*inw - 1))
        start_idx = i*inw + 1
    end

    powers = Float64[]
    power_matrix = []

    for idx in windows
        yi = y[idx]
        nt = length(yi)

        # Detrend if required
        # turn this into a function later
        if linfit
            x = collect(1:length(yi))
            X = hcat(x, ones(length(x)))
            β = X \ yi
            yi .-= X * β
        end

        # Remove mean
        yi .-= mean(yi)

        # Apply window
        window = tukeycf == 1 ? hanning(nt) :
                 tukeycf == 0 ? ones(nt) : tukeywin(nt, tukeycf)
        yi .= yi .* window

        # Prewhiten if requested
        if prewhit
            yi = diff(yi) ./ dt
            if isodd(length(yi))
                yi = yi[1:end-1]
            end
            nt = length(yi)
        end

        df = 1 / (dt * nt)
        freq = (1:nt ÷ 2) ./ (dt * nt)
        period = 1.0 ./ freq

        Y = fft(yi) .* dt
        deleteat!(Y, 1)   # remove sum value

        P2 = abs2.(Y)
        P1 = 2 .* P2[1:nt ÷ 2]

        if prewhit
            push!(power_matrix, (P1 .* df) ./ (freq .^ 2))
        else
            push!(power_matrix, P1 .* df)
        end
    end

    # Average power over windows
    power = reduce(+, power_matrix) ./ numwin

    # Define freq and period outside the loop
    nt = length(y[windows[1]])  # size of first window
    freq = (1:nt ÷ 2) ./ (dt * nt)
    period = 1.0 ./ freq

    return period, freq, power
end