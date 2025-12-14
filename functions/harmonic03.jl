using LinearAlgebra
using CairoMakie
using Statistics


"""
    harmonic03(xt, xv, freq_sel, cel_num=1, plot_fig=false, freq_dat="L2")


Least-squares harmonic analysis for a time series (per depth bin).


Model:
    z(t) = A0 + ∑_{j=1}^m [ a_j*cos(ω_j*t) + b_j*sin(ω_j*t) ]


Arguments
- xt::Vector{<:Real}          : time in days (length nt)
- xv::AbstractArray{<:Real}   : velocity series; size (nbins, nt) or a Vector length nt
- freq_sel::AbstractVector{<:Integer} : indices into the selected frequency table
- cel_num::Integer            : depth index to plot (default 1)
- plot_fig::Bool              : whether to display a plot for depth `cel_num`
- freq_dat::AbstractString    : which frequency table ("L2" supported here)


Returns
- Ao::Vector{Float64}                 : mean (per depth), length nbins
- a::Matrix{Float64}                  : cos-coeffs, size (m, nbins)
- b::Matrix{Float64}                  : sin-coeffs, size (m, nbins)
- fq2::Vector{Float64}                : selected ω (rad/day), length m
- coef_det::Vector{Float64}           : R² per depth, length nbins
- sigma::Vector{Float64}              : residual std per depth, length nbins
- fit::Matrix{Float64}                : reconstructed series, size (nbins, nt), NaN where input was NaN


Notes
- `xt` must be in days to match ω in rad/day (dimensionless arguments to sin/cos).
- If `xv` is a vector, it will be reshaped to (1, nt) automatically.
"""
function harmonic03(
    xt,
    xv,
    freq_sel::AbstractVector{<:Integer},
    cel_num::Integer = 1,
    plot_fig::Bool = false,
    freq_dat::AbstractString = "L2"
)
    # Normalize types/shapes
    xt = collect(float.(xt))
    if ndims(xv) == 1
        xv = reshape(float.(xv), 1, :)
    else
        xv = Array{Float64}(xv)
    end
    nbins, nt = size(xv)
    length(xt) == nt || throw(ArgumentError("length(xt)=$(length(xt)) must equal size(xv,2)=$nt"))


    # Frequencies (ω in rad/day)
    fq = get_frequencies(freq_dat)                # throws if unsupported
    fq2 = fq[freq_sel]
    m = length(fq2)
    ntide = 2*m + 1                                # [1, cos(ω1 t), sin(ω1 t), ..., cos(ωm t), sin(ωm t)]


    # Preallocations
    Ao       = fill(NaN, nbins)
    a        = fill(NaN, m, nbins)
    b        = fill(NaN, m, nbins)
    coef_det = fill(NaN, nbins)
    sigma    = fill(NaN, nbins)
    fit      = fill(NaN, nbins, nt)


    # Optional plotting scaffolding
    fig = nothing; ax = nothing
    if plot_fig
        fig = Figure(resolution = (900, 600))
        ax  = Axis(fig[1, 1], xlabel="Time (days)", ylabel="Velocity", title="Harmonic Analysis")
    end
    plot_idx = clamp(cel_num, 1, nbins)


    # Process each depth bin
    for i in 1:nbins
        xvrow = @view xv[i, :]
        good  = findall(!isnan, xvrow)             # indices where data exist
        if length(good) <= ntide
            # Not enough data points to estimate coefficients (needs > ntide)
            continue
        end


        xt2 = xt[good]
        y   = Float64.(xvrow[good])


        # Build design matrix A (length(xt2) × ntide)
        A = ones(length(xt2), ntide)
        for j in 1:m
            ω = fq2[j]
            A[:, 2*j]     .= cos.(ω .* xt2)
            A[:, 2*j + 1] .= sin.(ω .* xt2)
        end


        # Robust least-squares solution (A \ y) instead of manual SVD accumulation
        β = A \ y


        # Unpack coefficients
        Ao[i]   = β[1]
        a[:, i] = β[2:2:end]
        b[:, i] = β[3:2:end]


        # Fitted series on the non-NaN times
        yhat = A * β
        fit[i, good] = yhat


        # Goodness of fit
        ȳ   = mean(y)
        sst = sum((y .- ȳ).^2)
        sse = sum((y .- yhat).^2)
        coef_det[i] = sst > 0 ? 1 - sse/sst : NaN
        sigma[i]    = sqrt(sse / (length(y) - ntide))


        # Plot (only selected depth)
        if plot_fig && i == plot_idx
            CairoMakie.scatter!(ax, xt2, y, label="Data", markersize=6)
            lines!(ax, xt2, yhat, label="Fit", linewidth=2)
            axislegend(ax, position = :rb)
        end
    end


    if plot_fig && !isnothing(fig)
        display(fig)
    end


    return Ao, a, b, fq2, coef_det, sigma, fit
end


# ----------------------
# Frequency selection
# ----------------------
function get_frequencies(which::AbstractString)
    if which == "L2"
        return frequencies_L2()
    else
        throw(ArgumentError("Only freq_dat=\"L2\" is implemented in this file. Got freq_dat=$(which)."))
    end
end


"""
    frequencies_L2() -> Vector{Float64}


Return the L2 tidal angular frequencies in rad/day.
(Subset as provided; indices are 1-based and match the original list.
K1 is index 21, M2 is index 46, so `freq_sel = [21, 46]` selects K1 and M2.)
"""
function frequencies_L2()
    fq = Float64[]


    # num      rad/day             name           (deg/hr)
    push!(fq,  0.0172027749 )  # 1 Sa             0.0410686
    push!(fq,  0.0344055918 )  # 2 Ssa            0.0821373
    push!(fq,  0.1975102965 )  # 3 Msm            0.4715211
    push!(fq,  0.2280271411 )  # 4 Mm             0.5443747
    push!(fq,  0.4255374376 )  # 5 Msf            1.0158958
    push!(fq,  0.4599430294 )  # 6 Mf             1.0980331
    push!(fq,  0.6879701705 )  # 7 A7             1.6424078
    push!(fq,  5.1868805159 )  # 8 α1             12.3827651
    push!(fq,  5.3843908124 )  # 9 2Q1            12.8542862
    push!(fq,  5.4149076570 )  # 10 σ1             12.9271398
    push!(fq,  5.6124179535 )  # 11 Q1             13.3986609
    push!(fq,  5.6429347981 )  # 12 ρ1             13.4715145
    push!(fq,  5.8404450946 )  # 13 O1             13.9430356
    push!(fq,  5.8748506864 )  # 14 τ1             14.0251729
    push!(fq,  6.0379553911 )  # 15 β1             14.4145567
    push!(fq,  6.0723609410 )  # 16 M1/NO1         14.4966939
    push!(fq,  6.1028778275 )  # 17 χ1             14.5695476
    push!(fq,  6.2487805532 )  # 18 π1             14.9178647
    push!(fq,  6.2659825322 )  # 19 P1             14.9589314
    push!(fq,  6.2831853072 )  # 20 S1             15.0000000
    push!(fq,  6.3003880821 )  # 21 K1             15.0410686
    push!(fq,  6.3175900612 )  # 22 ψ1             15.0821353
    push!(fq,  6.3347936739 )  # 23 φ1             15.1232059
    push!(fq,  6.4978983786 )  # 24 θ1             15.5125897
    push!(fq,  6.5284152232 )  # 25 J1             15.5854433
    push!(fq,  6.6915199280 )  # 26 2PO1           15.9748271
    push!(fq,  6.7259255197 )  # 27 SO1            16.0569644
    push!(fq,  6.7603311115 )  # 28 OO1            16.1391017
    push!(fq,  6.9883582526 )  # 29 ϒ1/KQ1         16.6834764
    push!(fq,  11.2553527516 ) # 30 3MKS2          26.8701754
    push!(fq,  11.2592414569 ) # 31 2NS2           26.8794590
    push!(fq,  11.2897583015 ) # 32 3MS2           26.9523126
    push!(fq,  11.4567517534 ) # 33 OQ2            27.3509801
    push!(fq,  11.4872685980 ) # 34 ε2/MNS2        27.4238337
    push!(fq,  11.5177854426 ) # 35 2ML2S2         27.4966873
    push!(fq,  11.6464845975 ) # 36 2MS2K2         27.8039339
    push!(fq,  11.6808901474 ) # 37 NLK2/2MK2      27.8860711
    push!(fq,  11.6847788945 ) # 38 2N2            27.8953548
    push!(fq,  11.7152957391 ) # 39 μ2/2MS2        27.9682084
    push!(fq,  11.9128060356 ) # 40 N2             28.4397295
    push!(fq,  11.9433228802 ) # 41 ν2             28.5125831
    push!(fq,  11.9777284720 ) # 42 MKL2S2         28.5947204
    push!(fq,  12.1064275850 ) # 43 OP2            28.9019669
    push!(fq,  12.1236304018 ) # 44 MPS2           28.9430356
    push!(fq,  12.1236311977 ) # 45 H1             28.9430375
    push!(fq,  12.1408331767 ) # 46 M2             28.9841042
    push!(fq,  12.1580351558 ) # 47 H2             29.0251709
    push!(fq,  12.1580359936 ) # 48 MSP2           29.0251729
    push!(fq,  12.1752387685 ) # 49 MKS2           29.0662415
    push!(fq,  12.2096443603 ) # 50 M2(KS)2        29.1483788
    push!(fq,  12.3039378815 ) # 51 2SN(MK)2       29.3734880
    push!(fq,  12.3383434732 ) # 52 λ2             29.4556253
    push!(fq,  12.3688603179 ) # 53 L2/2MN2        29.5284789
    push!(fq,  12.5491686353 ) # 54 T2             29.9589333
    push!(fq,  12.5663706144 ) # 55 S2             30.0000000
    push!(fq,  12.5835725934 ) # 56 R2             30.0410667
    push!(fq,  12.6007762061 ) # 57 K2             30.0821373
    push!(fq,  12.7943977555 ) # 58 ζ/MSN2         30.5443747
    push!(fq,  12.8288033472 ) # 59 η/KJ2          30.6265120
    push!(fq,  12.8632089390 ) # 60 2KM(SN)2       30.7086493
    push!(fq,  12.9919080520 ) # 61 2SM2           31.0158958
    push!(fq,  13.0263136437 ) # 62 SKM2           31.0980331


    return fq
end


# -------------------------------------------------------------------
# Example (commented):
# xt = (1:32) ./ 24
# xv = [1.97, 1.46, 0.98, 0.73, 0.67, 0.82, 1.15, 1.58, 2.0, 2.33, 2.48, 2.43, 2.25, 2.02, 1.82, 1.72, 1.75, 1.91, 2.22, 2.54, 2.87, 3.1, 3.15, 2.94, 2.57, 2.06, 1.56, 1.13, 0.84, 0.73, 0.79, 1.07]
# Ao, a, b, fq2, R2, sigma, fit = harmonic03(xt, xv, [21, 46], 1, true, "L2")
# -------------------------------------------------------------------





