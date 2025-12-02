"""
    place_interior_knots(sojourns::AbstractVector{<:Real}, nknots::Integer; 
                         lower_bound::Real=0.0, upper_bound::Real=Inf)

Place interior knots at quantiles of sojourn times, following survextrap convention.

This function computes knot locations at evenly-spaced quantiles of the observed 
sojourn times. On the clock-reset (semi-Markov) scale, there may be ties in sojourn 
times. When quantile locations are tied, knots are equally spaced between unique 
quantile values to avoid degenerate spline bases.

# Arguments
- `sojourns`: Vector of sojourn times (reset scale)
- `nknots`: Number of interior knots to place (default: 10 following survextrap)
- `lower_bound`: Lower boundary knot (default: 0.0)
- `upper_bound`: Upper boundary knot (default: maximum sojourn)

# Returns
- Vector of interior knot locations (excluding boundary knots)

# Notes
- Quantile levels are spaced evenly on (0, 1): k/(nknots+1) for k = 1, ..., nknots
- When ties occur, knots are equally spaced between unique quantile values
- Returns empty vector if nknots ≤ 0 or insufficient unique sojourns

# References
- survextrap methods: https://chjackson.github.io/survextrap/articles/methods.html
- Tang et al. (2022): floor(n^(1/5)) interior knots at quantiles
"""
function place_interior_knots(sojourns::AbstractVector{<:Real}, nknots::Integer;
                              lower_bound::Real=0.0, upper_bound::Real=Inf)
    
    # Handle edge cases
    nknots <= 0 && return Float64[]
    isempty(sojourns) && return Float64[]
    
    # Use finite upper bound
    ub = isfinite(upper_bound) ? upper_bound : maximum(sojourns)
    lb = lower_bound
    
    # Compute quantile levels: evenly spaced on (0, 1)
    # Following Tang et al.: k/(nknots+1) for k = 1, ..., nknots
    quantile_levels = [(k / (nknots + 1)) for k in 1:nknots]
    
    # Get raw quantile locations
    raw_knots = quantile(sojourns, quantile_levels)
    
    # Handle ties: unique quantile values, then interpolate if needed
    unique_raw = unique(raw_knots)
    
    if length(unique_raw) == nknots
        # No ties - use raw quantiles directly
        knots = raw_knots
    elseif length(unique_raw) == 1
        # All quantiles are tied to single value - spread evenly in interval
        # This can happen with very sparse data
        mid = unique_raw[1]
        if mid > lb && mid < ub
            # Spread nknots evenly between lb and ub (interior only)
            knots = collect(range(lb + (ub - lb)/(nknots + 1), 
                                  stop=ub - (ub - lb)/(nknots + 1), 
                                  length=nknots))
        else
            knots = collect(range(lb + (ub - lb)/(nknots + 1),
                                  stop=ub - (ub - lb)/(nknots + 1),
                                  length=nknots))
        end
    else
        # Some ties - interpolate between unique values
        # Strategy: preserve unique quantile locations, fill gaps evenly
        knots = Float64[]
        sizehint!(knots, nknots)
        
        # Group consecutive raw_knots by value
        i = 1
        while i <= nknots
            current_val = raw_knots[i]
            
            # Count how many consecutive knots have same value
            j = i
            while j <= nknots && raw_knots[j] ≈ current_val
                j += 1
            end
            n_tied = j - i
            
            if n_tied == 1
                # No tie - use as-is
                push!(knots, current_val)
            else
                # Tied knots: determine interval for spacing
                left_bound = i == 1 ? lb : raw_knots[i - 1]
                right_bound = j > nknots ? ub : raw_knots[j]
                
                # Equally space n_tied knots between left and right bounds
                if left_bound < current_val && current_val < right_bound
                    # Current value is in interior - spread around it
                    spacing = (right_bound - left_bound) / (n_tied + 1)
                    for k in 1:n_tied
                        push!(knots, left_bound + k * spacing)
                    end
                else
                    # Degenerate case - just use current value
                    for _ in 1:n_tied
                        push!(knots, current_val)
                    end
                end
            end
            
            i = j
        end
        
        # Ensure uniqueness after interpolation
        unique!(sort!(knots))
        
        # If we lost some knots due to ties, fill with evenly spaced
        if length(knots) < nknots
            # Add more evenly spaced knots
            remaining = nknots - length(knots)
            gaps = Float64[]
            
            # Find largest gaps
            all_points = sort(vcat([lb], knots, [ub]))
            for i in 1:(length(all_points) - 1)
                push!(gaps, all_points[i + 1] - all_points[i])
            end
            
            # Add knots in largest gaps
            gap_order = sortperm(gaps, rev=true)
            for g in gap_order[1:min(remaining, length(gap_order))]
                new_knot = (all_points[g] + all_points[g + 1]) / 2
                push!(knots, new_knot)
            end
            
            sort!(knots)
        end
    end
    
    # Clamp to (lower_bound, upper_bound) - knots must be strictly interior
    filter!(k -> lb < k < ub, knots)
    
    return knots
end

"""
    default_nknots(n_observations::Integer)

Compute default number of interior knots following Tang et al. (2022) convention.

Uses floor(n^(1/5)) for sieve estimation. For typical survival data sizes:
- n = 100: 2-3 knots
- n = 500: 3-4 knots  
- n = 1000: 4 knots
- n = 10000: 6 knots

For small samples, returns at least 2 knots for flexibility.
survextrap uses a fixed 10 knots by default.
"""
function default_nknots(n_observations::Integer)
    n_observations <= 0 && return 0
    return max(2, floor(Int, n_observations^(1/5)))
end

# =============================================================================
# Legacy coefficient transformation functions
# =============================================================================
# These functions are retained for backward compatibility with any external
# code that may depend on them. New code should use _spline_ests2coefs and
# _spline_coefs2ests from modelgeneration.jl which work with RuntimeSplineHazard.

"""
    spline_ests2coefs(coefs, hazard; clamp_zeros = false)

Transform spline parameter estimates on their unrestricted estimation scale to coefficients.
"""
function spline_ests2coefs(ests, hazard; clamp_zeros = false)

    # transform
    if hazard.monotone == 0
        # just exponentiate
        coefs = exp.(ests)

    elseif hazard.monotone != 0

        # exponentiate 
        ests_nat = exp.(ests)
        coefs    = zeros(eltype(ests_nat), length(ests))
            
        # accumulate
        if length(coefs) > 1
            k = BSplineKit.order(hazard.hazsp)
            t = knots(hazard.hazsp.spline)
    
            for i in 2:length(coefs)
                coefs[i] = coefs[i-1] + ests_nat[i] * (t[i + k] - t[i]) / k
            end
        end

        # intercept
        coefs .+= ests_nat[1]

        # if monotone decreasing then reverse
        if hazard.monotone == -1
            reverse!(coefs)
        end
    end
    
    # clamp numerical zeros
    if clamp_zeros
        coefs[findall(isapprox.(coefs, 0.0; atol = sqrt(eps())))] .= zero(eltype(coefs))    
    end

    return coefs
end

"""
    spline_coefs2ests(ests, hazard; clamp_zeros = false)

Transform spline coefficients to unrestrected estimation scale parameters.
"""
function spline_coefs2ests(coefs, hazard; clamp_zeros = false)

    # transform
    if hazard.monotone == 0
        # just exponentiate
        ests_nat = coefs

    elseif hazard.monotone != 0

        # reverse if moonotone decreasing
        coefs_nat = deepcopy((hazard.monotone == 1) ? coefs : reverse(coefs))

        # initialize
        ests_nat = zeros(eltype(coefs), length(coefs_nat))
        
        # accumulate
        if length(coefs) > 1
            k = BSplineKit.order(hazard.hazsp)
            t = knots(hazard.hazsp.spline)
            
            for i in length(ests_nat):-1:2
                ests_nat[i] = (coefs_nat[i] - coefs_nat[i - 1]) * k / (t[i + k] - t[i])
            end
        end

        # intercept
        ests_nat[1] = coefs_nat[1]

    end
    
    # clamp numerical errors to zero
    if clamp_zeros
        ests_nat[findall(isapprox.(ests_nat, 0.0; atol = sqrt(eps())))] .= zero(eltype(ests_nat))
    end

    ests = log.(ests_nat)

    return ests
end

"""
    rectify_coefs!(ests, model)

Pass model estimates through spline coefficient transformations to remove numerical zeros.

For spline hazards, the I-spline transformation can accumulate numerical errors
during optimization. This function applies a round-trip transformation
(ests → coefs → ests) with zero-clamping to clean up near-zero values.
"""
function rectify_coefs!(ests, model)
    nested = VectorOfVectors(ests, get_elem_ptr(model.parameters))

    for i in eachindex(model.hazards)
        haz = model.hazards[i]
        if isa(haz, RuntimeSplineHazard)
            # New functional design - use basis stored in the hazard functions
            # We need to rebuild the basis from the knots
            nbasis = haz.npar_baseline
            basis = _rebuild_spline_basis(haz)
            
            # Round-trip transformation to clean up numerical zeros
            baseline_ests = nested[i][1:nbasis]
            coefs = _spline_ests2coefs(baseline_ests, basis, haz.monotone)
            
            # Clamp numerical zeros in coefficients
            coefs[findall(isapprox.(coefs, 0.0; atol = sqrt(eps())))] .= zero(eltype(coefs))
            
            # Transform back
            rectified_baseline = _spline_coefs2ests(coefs, basis, haz.monotone; clamp_zeros=true)
            
            # Combine with covariate parameters
            rectified = vcat(rectified_baseline, nested[i][nbasis+1:end])
            
            # Copy back to ests
            deepsetindex!(nested, rectified, i)
        end
    end
end

"""
    _rebuild_spline_basis(hazard::RuntimeSplineHazard)

Rebuild the BSpline basis from a RuntimeSplineHazard for coefficient transformations.
"""
function _rebuild_spline_basis(hazard::RuntimeSplineHazard)
    B = BSplineBasis(BSplineOrder(hazard.degree + 1), copy(hazard.knots))
    if (hazard.degree > 1) && hazard.natural_spline
        B = RecombinedBSplineBasis(B, Natural())
    end
    return B
end

"""
    remake_splines!(hazard::RuntimeSplineHazard, parameters)

No-op for RuntimeSplineHazard since splines are constructed on-the-fly during evaluation.
The functional design means no internal state needs updating.
"""
function remake_splines!(hazard::RuntimeSplineHazard, parameters)
    return nothing
end

"""
    set_riskperiod!(hazard::RuntimeSplineHazard)

No-op for RuntimeSplineHazard since risk period is handled dynamically during 
extrapolation evaluation in the hazard/cumhaz closures.
"""
function set_riskperiod!(hazard::RuntimeSplineHazard)
    return nothing
end

"""
    _update_spline_hazards!(hazards, pars)

Update all spline hazards with new parameters. This is a no-op for the current
functional spline design (splines are constructed on-the-fly), but provides a
single call site for spline parameter updates.

# Arguments
- `hazards`: Vector of hazard objects
- `pars`: Nested parameter vectors (log-scale for splines)

This helper reduces code duplication across likelihood, sampling, and fitting code.
"""
function _update_spline_hazards!(hazards::Vector{<:_Hazard}, pars)
    for i in eachindex(hazards)
        if isa(hazards[i], _SplineHazard)
            remake_splines!(hazards[i], pars[i])
            set_riskperiod!(hazards[i])
        end
    end
    return nothing
end
