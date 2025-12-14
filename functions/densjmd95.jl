"""
    densjmd95(s, t, p)

Density of Sea Water using Jackett and McDougall 1995 (JAOT 12) 
polynomial (modified UNESCO polynomial).

# Arguments
- `s`: salinity [psu (PSS-78)]
- `t`: potential temperature [degree C (IPTS-68)]
- `p`: pressure [dbar]

# Returns
- `ρ`: density [kg/m³]

# Example
```julia
s = 35.5  # PSU
t = 3.0   # degC
p = 3000.0  # dbar
ρ = densjmd95(s, t, p)
# Expected: ρ ≈ 1041.83267 kg/m³
```

# Author
Converted to Julia from MATLAB version by Martin Losch (2002-08-09)
"""
function densjmd95(s, t, p)
    
    # Convert pressure from dbar to bar
    p = 0.1 .* p
    
    # Coefficients for density of fresh water at p = 0
    eosJMDCFw = [
        999.842594,
        6.793952e-02,
        -9.095290e-03,
        1.001685e-04,
        -1.120083e-06,
        6.536332e-09
    ]
    
    # Coefficients for density of sea water at p = 0
    eosJMDCSw = [
        8.244930e-01,
        -4.089900e-03,
        7.643800e-05,
        -8.246700e-07,
        5.387500e-09,
        -5.724660e-03,
        1.022700e-04,
        -1.654600e-06,
        4.831400e-04
    ]
    
    # Check for negative salinity
    if any(s .< 0)
        @warn "Found negative salinity values, resetting them to NaN"
        s = replace(x -> x < 0 ? NaN : x, s)
    end
    
    # Calculate temperature powers
    t2 = t .^ 2
    t3 = t .^ 3
    t4 = t .^ 4
    
    # Calculate s^(3/2)
    s3o2 = s .* sqrt.(s)
    
    # Density of freshwater at the surface
    ρ = eosJMDCFw[1] .+
        eosJMDCFw[2] .* t .+
        eosJMDCFw[3] .* t2 .+
        eosJMDCFw[4] .* t3 .+
        eosJMDCFw[5] .* t4 .+
        eosJMDCFw[6] .* t4 .* t
    
    # Add sea water contribution
    ρ = ρ .+
        s .* (
            eosJMDCSw[1] .+
            eosJMDCSw[2] .* t .+
            eosJMDCSw[3] .* t2 .+
            eosJMDCSw[4] .* t3 .+
            eosJMDCSw[5] .* t4
        ) .+
        s3o2 .* (
            eosJMDCSw[6] .+
            eosJMDCSw[7] .* t .+
            eosJMDCSw[8] .* t2
        ) .+
        eosJMDCSw[9] .* s .^ 2
    
    # Apply pressure correction
    ρ = ρ ./ (1 .- p ./ bulkmodjmd95(s, t, p))
    
    return ρ
end


"""
    bulkmodjmd95(s, t, p)

Calculate secant bulk modulus of seawater using JMD95 equation of state.

# Arguments
- `s`: salinity [psu]
- `t`: potential temperature [degC]
- `p`: pressure [bar]

# Returns
- `bulkmod`: bulk modulus [bar]
"""
function bulkmodjmd95(s, t, p)
    
    # Coefficients for secant bulk modulus of fresh water at p = 0
    eosJMDCKFw = [
        1.965933e+04,
        1.444304e+02,
        -1.706103e+00,
        9.648704e-03,
        -4.190253e-05
    ]
    
    # Coefficients for secant bulk modulus of sea water at p = 0
    eosJMDCKSw = [
        5.284855e+01,
        -3.101089e-01,
        6.283263e-03,
        -5.084188e-05,
        3.886640e-01,
        9.085835e-03,
        -4.619924e-04
    ]
    
    # Coefficients for secant bulk modulus at pressure p
    eosJMDCKP = [
        3.186519e+00,
        2.212276e-02,
        -2.984642e-04,
        1.956415e-06,
        6.704388e-03,
        -1.847318e-04,
        2.059331e-07,
        1.480266e-04,
        2.102898e-04,
        -1.202016e-05,
        1.394680e-07,
        -2.040237e-06,
        6.128773e-08,
        6.207323e-10
    ]
    
    # Check for negative salinity
    if any(s .< 0)
        @warn "Found negative salinity values, resetting them to NaN"
        s = replace(x -> x < 0 ? NaN : x, s)
    end
    
    # Calculate powers
    t2 = t .^ 2
    t3 = t .^ 3
    t4 = t .^ 4
    s3o2 = s .* sqrt.(s)
    p2 = p .^ 2
    
    # Secant bulk modulus of fresh water at the surface
    bulkmod = eosJMDCKFw[1] .+
              eosJMDCKFw[2] .* t .+
              eosJMDCKFw[3] .* t2 .+
              eosJMDCKFw[4] .* t3 .+
              eosJMDCKFw[5] .* t4
    
    # Add sea water contribution at surface
    bulkmod = bulkmod .+
              s .* (
                  eosJMDCKSw[1] .+
                  eosJMDCKSw[2] .* t .+
                  eosJMDCKSw[3] .* t2 .+
                  eosJMDCKSw[4] .* t3
              ) .+
              s3o2 .* (
                  eosJMDCKSw[5] .+
                  eosJMDCKSw[6] .* t .+
                  eosJMDCKSw[7] .* t2
              )
    
    # Add pressure effects
    bulkmod = bulkmod .+
              p .* (
                  eosJMDCKP[1] .+
                  eosJMDCKP[2] .* t .+
                  eosJMDCKP[3] .* t2 .+
                  eosJMDCKP[4] .* t3
              ) .+
              p .* s .* (
                  eosJMDCKP[5] .+
                  eosJMDCKP[6] .* t .+
                  eosJMDCKP[7] .* t2
              ) .+
              p .* s3o2 .* eosJMDCKP[8] .+
              p2 .* (
                  eosJMDCKP[9] .+
                  eosJMDCKP[10] .* t .+
                  eosJMDCKP[11] .* t2
              ) .+
              p2 .* s .* (
                  eosJMDCKP[12] .+
                  eosJMDCKP[13] .* t .+
                  eosJMDCKP[14] .* t2
              )
    
    return bulkmod
end

