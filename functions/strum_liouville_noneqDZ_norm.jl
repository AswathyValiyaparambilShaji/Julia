"""
    k, L, C, Cg, Ce, Weig, Ueig, Ueig2 = 
    sturm_liouville_noneqDZ_norm(zf::Vector{Float64}, N2::Vector{Float64}, 
    f::Float64, om::Float64, nonhyd::Int; fillval::Float64 = 1e-12)

Solve for n eigenfunctions and eigenvalues for given vector N2[zf]; n = length(zf)-2. 

Returns wavenumber k, wavelength L, phase, group, and eigen speeds C, Cg, and Ce, 
and non-normalized vertical velocity eigenfunctions Weig at faces 
and horizontal velocty eigenfunction Ueig=1/k*dW/dz (Gerkema IW syllabus) 
at centers and normalized Ueig2 at centers

# Arguments    
- `zf::Vector{Float64}`: layer faces [m], can either surface to bottom (default)
                         (e.g., 0 to -H) or bottom to surface,

- `N2::Vector{Float64}`: Brunt-Väisälä frequency squared [rad^2/s^2] 
                         at layer faces zf,

- `f::Float64`: Coriolis frequency [rad/s],

- `om::Float64`: internal wave frequency [rad/s],

- `nonhyd::Int`: if 1, solve the non-hydrostatic Sturm-Liouville problem

- `fillval::Float64`: Default is 1e12; replace negative values 
                      in N(z) fill value

# info
Maarten Buijsman, Oladeji Siyanbola, USM, 2025-8-27. Based on Ashok & Bhaduria (2009). 
"""
function sturm_liouville_noneqDZ_norm(zf::Vector{Float64}, N2::Vector{Float64}, f::Float64, om::Float64, nonhyd::Int; fillval::Float64 = 1e-12)

    # code is set up for surface to bottom ordered profiles
    flipped = zf[1] > zf[end]  # if true, input is surface to bottom
    if !flipped
        zf = reverse(zf)
        N2 = reverse(N2)
    end

    dz = -diff(zf)
    H = sum(dz)
    N = length(dz)

    # Handle hydrostatic / nonhydrostatic modes
    if nonhyd == 1
        NN = clamp.(N2 .- om^2, fillval, Inf)
    else
        NN = clamp.(N2, fillval, Inf)
    end

    # Construct B matrix
    B = Diagonal(-NN[2:end-1])

    # Construct A matrix
    A = zeros(N - 1, N - 1)
    A[1,1] = -2 / (dz[1] * dz[2])
    A[1,2] = 2 / (dz[2] * (dz[1] + dz[2]))

    for i in 2:N-2
        A[i,i-1] =  2 / (dz[i] * (dz[i] + dz[i+1]))
        A[i,i]   = -2 / (dz[i] * dz[i+1])
        A[i,i+1] =  2 / (dz[i+1] * (dz[i] + dz[i+1]))
    end

    i = N - 1
    A[i,i-1] =  2 / (dz[i] * (dz[i] + dz[i+1]))
    A[i,i]   = -2 / (dz[i] * dz[i+1])

    # Solve eigenvalue problem
    invCe2, W1 = eigen(A, B)
    Ce2 = 1 ./ invCe2
    Ce = sqrt.(Ce2)
    idx = sortperm(Ce, rev=true)  # order large to small
    Ce = Ce[idx]
    W1 = W1[:, idx]

    # compute dispersion characteristics
    # println("f=",f)
    k = abs.(sqrt(om^2 - f^2) ./ Ce)
    C = om ./ k
    L = 2π ./ k
    Cg = Ce.^2 .* k ./ om

    # Compute vertical structure functions (Weig: faces)
    Weig = vcat(zeros(1, size(W1,2)), W1, zeros(1, size(W1,2)))

    # Compute horizontal eigenfunctions (Ueig: centers)
    # and normalize dw/dz by k as in Gerkema
    dWeig = Weig[2:end, :] .- Weig[1:end-1, :]
    dzu   = repeat(dz, 1, size(W1,2))
    kk    = repeat(transpose(k), length(dz),1)
    Ueig  = dWeig ./ dzu ./kk

    # normalize Ueig
    norm_factor = sqrt.(sum(Ueig.^2 .* dzu, dims=1) ./ H)
    norm_factor[norm_factor .== 0] .= Inf
    Ueig2 = Ueig ./ norm_factor

    # need to normalize Weig

    # set the correct sign near the surface (positive)
    # Weig (negative; see Gerkema IW syllabus)
    for i in 1:size(Weig,2)
        if Weig[2,i] > 0
            Weig[:,i] .*= -1
        end
    end

    #Ueig (positive)
    for i in 1:size(Ueig2,2)
        if Ueig2[1,i] < 0
            Ueig[:,i] .*= -1
            Ueig2[:,i] .*= -1            
        end
    end

    # Reverse output structure functions if input was reversed
    if !flipped
        Weig  = reverse(Weig, dims=1)
        Ueig  = reverse(Ueig, dims=1)
        Ueig2 = reverse(Ueig2, dims=1)        
    end

    return k, L, C, Cg, Ce, Weig, Ueig, Ueig2
end