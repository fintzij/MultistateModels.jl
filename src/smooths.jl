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
    nested = unflatten(model.parameters.reconstructor, ests)

    # Get hazard names in sorted order
    haznames_sorted = [hazname for (hazname, idx) in sort(collect(model.hazkeys), by = x -> x[2])]
    
    # Process each hazard
    for (i, hazname) in enumerate(haznames_sorted)
        haz = model.hazards[i]
        if isa(haz, RuntimeSplineHazard)
            # New functional design - use basis stored in the hazard functions
            # We need to rebuild the basis from the knots
            nbasis = haz.npar_baseline
            basis = _rebuild_spline_basis(haz)
            
            # Extract baseline parameters from nested structure
            hazard_params = nested[hazname]
            baseline_values = collect(values(hazard_params.baseline))
            
            # Round-trip transformation to clean up numerical zeros
            coefs = _spline_ests2coefs(baseline_values, basis, haz.monotone)
            
            # Clamp numerical zeros in coefficients
            coefs[findall(isapprox.(coefs, 0.0; atol = sqrt(eps())))] .= zero(eltype(coefs))
            
            # Transform back
            rectified_baseline = _spline_coefs2ests(coefs, basis, haz.monotone; clamp_zeros=true)
            
            # Update ests vector directly (need to find position in flat vector)
            # Compute offset for this hazard's parameters in flat vector
            offset = sum(model.hazards[j].npar_total for j in 1:(i-1); init=0)
            ests[offset+1:offset+nbasis] .= rectified_baseline
        end
    end
end

"""
    _rebuild_spline_basis(hazard::RuntimeSplineHazard)

Rebuild the BSpline basis from a RuntimeSplineHazard for coefficient transformations.

Handles natural splines and constant extrapolation boundary conditions:
- constant: enforces D¹=0 (Neumann BC) at boundaries for smooth constant extrapolation
- natural_spline: enforces D²=0 (natural spline) at boundaries
"""
function _rebuild_spline_basis(hazard::RuntimeSplineHazard)
    B = BSplineBasis(BSplineOrder(hazard.degree + 1), copy(hazard.knots))
    
    # Apply boundary conditions based on extrapolation method
    if hazard.extrapolation == "constant" && hazard.degree >= 2
        # constant: D¹=0 at both boundaries for C¹ continuity
        B = RecombinedBSplineBasis(B, Derivative(1))
    elseif (hazard.degree > 1) && hazard.natural_spline
        # Natural spline: D²=0 at boundaries
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
            # Note: For RuntimeSplineHazard, remake_splines! is a no-op
            remake_splines!(hazards[i], nothing)
            set_riskperiod!(hazards[i])
        end
    end
    return nothing
end

# =============================================================================
# Data-Driven Knot Placement from Sampled Paths
# =============================================================================

"""
    place_knots_from_paths!(target_model::MultistateProcess, 
                            fitted_model::MultistateProcess;
                            n_paths::Int = 1000,
                            n_knots::Union{Int, Nothing} = nothing,
                            quantile_probs::Union{Vector{Float64}, Nothing} = nothing,
                            min_ess::Int = 100)

Update spline hazards in `target_model` with knot locations computed from conditional 
sample paths drawn using `fitted_model` as the proposal distribution.

This function is useful for data-driven knot placement when you have:
1. A fitted Markov or phase-type surrogate model that approximates the true process
2. An unfitted model with spline hazards where you want optimal knot locations

The workflow:
1. Draw conditional sample paths from `fitted_model` using importance sampling
2. Extract transition times (sojourns) for each transition type from the sampled paths
3. Compute quantiles of the sojourn distributions to place interior knots
4. Rebuild `target_model` with updated spline hazards using the recommended knots

# Arguments
- `target_model::MultistateProcess`: Unfitted model with spline hazards to update (modified in place)
- `fitted_model::MultistateProcess`: Fitted Markov or phase-type model used to draw paths
- `n_paths::Int=1000`: Number of paths to sample per subject (if using fixed count)
- `n_knots::Union{Int,Nothing}=nothing`: Number of interior knots per transition. 
   If `nothing`, uses `default_nknots()` based on sample size.
- `quantile_probs::Union{Vector{Float64},Nothing}=nothing`: Custom quantile probabilities 
   for knot placement (e.g., `[0.25, 0.5, 0.75]`). If `nothing`, uses evenly-spaced quantiles.
- `min_ess::Int=100`: Minimum effective sample size for importance sampling

# Returns
- `Dict{Tuple{Int,Int}, Vector{Float64}}`: Dictionary mapping `(statefrom, stateto)` to 
   the computed interior knot locations for each transition

# Example
```julia
# Fit a Markov surrogate first
markov_fit = fit(markov_model)

# Create target model with spline hazards (knots will be replaced)
h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; degree=3, knots=nothing)
h21 = Hazard(@formula(0 ~ 1), "sp", 2, 1; degree=3, knots=nothing)
spline_model = multistatemodel(h12, h21; data=data)

# Place knots based on sampled paths
knot_locations = place_knots_from_paths!(spline_model, markov_fit; n_knots=5)

# Now fit the spline model
spline_fit = fit(spline_model)
```

# Notes
- Only updates hazards that are `SplineHazard` types in `target_model`
- Requires that `fitted_model` has been fitted and has a valid surrogate
- The transition matrix of both models must be compatible
- For transitions with no sampled events, falls back to evenly-spaced knots

See also: [`place_interior_knots`](@ref), [`draw_paths`](@ref), [`default_nknots`](@ref)
"""
function place_knots_from_paths!(target_model::MultistateProcess, 
                                  fitted_model::MultistateProcess;
                                  n_paths::Int = 1000,
                                  n_knots::Union{Int, Nothing} = nothing,
                                  quantile_probs::Union{Vector{Float64}, Nothing} = nothing,
                                  min_ess::Int = 100)
    
    # Validate inputs
    @assert size(target_model.tmat) == size(fitted_model.tmat) "Transition matrices must have same dimensions"
    
    # Check that fitted_model has been fitted:
    # - MultistateModelFitted: fitted via fit() - either exact data or MCEM with surrogate
    # - MultistateMarkovModel: can use draw_paths directly for exact data
    is_fitted = fitted_model isa MultistateModelFitted
    is_markov_unfitted = fitted_model isa MultistateMarkovModel || fitted_model isa MultistateMarkovModelCensored
    @assert is_fitted || is_markov_unfitted "fitted_model must be a MultistateModelFitted or Markov model"
    
    # Draw conditional sample paths
    path_result = draw_paths(fitted_model; min_ess=min_ess)
    
    # Extract paths - handle different return types
    all_paths = if path_result isa NamedTuple && haskey(path_result, :loglik)
        # Exact data case - extract paths from data
        extract_paths(fitted_model.data)
    else
        # Imputed paths case - flatten the subject-level vectors
        vcat(path_result...)
    end
    
    # Dictionary to store computed knots per transition
    knot_locations = Dict{Tuple{Int,Int}, Vector{Float64}}()
    
    # Process each spline hazard in target model
    for (idx, haz) in enumerate(target_model.hazards)
        if !(haz isa _SplineHazard)
            continue
        end
        
        statefrom = haz.statefrom
        stateto = haz.stateto
        
        # Extract sojourn times for this transition from sampled paths
        sojourns = extract_sojourns(statefrom, stateto, all_paths)
        
        if isempty(sojourns)
            @warn "No sampled transitions $(statefrom)→$(stateto); using evenly-spaced knots"
            # Fall back to evenly spaced knots within boundary
            bknots = haz.knots[1], haz.knots[end]
            nk = n_knots !== nothing ? n_knots : 2
            interior = collect(range(bknots[1] + (bknots[2] - bknots[1])/(nk + 1),
                                     stop=bknots[2] - (bknots[2] - bknots[1])/(nk + 1),
                                     length=nk))
            knot_locations[(statefrom, stateto)] = interior
            continue
        end
        
        # Determine number of knots
        nk = n_knots !== nothing ? n_knots : default_nknots(length(sojourns))
        
        # Compute knot locations
        if quantile_probs !== nothing
            # User-specified quantiles
            @assert length(quantile_probs) == nk "quantile_probs length must match n_knots"
            interior = quantile(sojourns, quantile_probs)
        else
            # Evenly-spaced quantiles via place_interior_knots
            lb = 0.0
            ub = maximum(sojourns)
            interior = place_interior_knots(sojourns, nk; lower_bound=lb, upper_bound=ub)
        end
        
        knot_locations[(statefrom, stateto)] = interior
        
        @info "Placed $(length(interior)) knots for $(statefrom)→$(stateto) at: $(round.(interior, digits=3))"
    end
    
    # Rebuild target model with new knot locations
    # This requires reconstructing the hazard specifications and calling multistatemodel again
    _rebuild_model_with_knots!(target_model, knot_locations)
    
    return knot_locations
end

"""
    _rebuild_model_with_knots!(model::MultistateProcess, knot_locations::Dict)

Internal helper to rebuild spline hazards with new knot locations.
Modifies the model's RuntimeSplineHazard objects in place.

Note: This is a simplified update that modifies the internal hazard structures.
For a complete rebuild, use `multistatemodel` with updated hazard specifications.
"""
function _rebuild_model_with_knots!(model::MultistateProcess, 
                                     knot_locations::Dict{Tuple{Int,Int}, Vector{Float64}})
    # For now, we update the knots field in RuntimeSplineHazard
    # A more complete solution would rebuild the entire spline basis
    
    for (idx, haz) in enumerate(model.hazards)
        if !(haz isa RuntimeSplineHazard)
            continue
        end
        
        key = (haz.statefrom, haz.stateto)
        if !haskey(knot_locations, key)
            continue
        end
        
        new_interior = knot_locations[key]
        
        # Get current boundary knots
        old_knots = haz.knots
        lb = old_knots[1]
        ub = old_knots[end]
        
        # Extend boundaries if needed to encompass new interior knots
        if !isempty(new_interior)
            lb = min(lb, minimum(new_interior) - 0.001)
            ub = max(ub, maximum(new_interior) + 0.001)
        end
        
        # New complete knot sequence
        new_knots = unique(sort([lb; new_interior; ub]))
        
        # Rebuild the spline basis with appropriate boundary conditions
        B = BSplineBasis(BSplineOrder(haz.degree + 1), copy(new_knots))
        if haz.extrapolation == "constant" && haz.degree >= 2
            # constant: D¹=0 at both boundaries for C¹ continuity
            B = RecombinedBSplineBasis(B, Derivative(1))
        elseif (haz.degree > 1) && haz.natural_spline
            # Natural spline: D²=0 at boundaries
            B = RecombinedBSplineBasis(B, Natural())
        end
        
        # Use the stored extrapolation method
        extrap_method = if haz.extrapolation == "linear"
            BSplineKit.SplineExtrapolations.Linear()
        else
            # "constant" uses Flat() - the smooth basis above ensures C¹ continuity
            BSplineKit.SplineExtrapolations.Flat()
        end
        
        nbasis = length(B)
        
        # Get linpred_effect from metadata (Symbol: :none, :linear, or :all)
        linpred_effect = haz.metadata.linpred_effect
        
        # Regenerate parameter names with correct count
        # Baseline spline parameters: hazname_sp1, hazname_sp2, etc.
        baseline_names = [Symbol(string(haz.hazname), "_sp", i) for i in 1:nbasis]
        # Keep covariate parameters as-is
        covar_pars = haz.has_covariates ? [Symbol(string(haz.hazname), "_", c) for c in haz.covar_names] : Symbol[]
        new_parnames = vcat(baseline_names, covar_pars)
        
        # Generate new hazard/cumhaz functions with correct nbasis
        hazard_fn, cumhaz_fn = _generate_spline_hazard_fns(
            B, extrap_method, haz.monotone, nbasis, new_parnames, linpred_effect
        )
        
        # Create new RuntimeSplineHazard with updated knots and parnames
        new_haz = RuntimeSplineHazard(
            haz.hazname,
            haz.statefrom,
            haz.stateto,
            haz.family,
            new_parnames,  # Updated parameter names
            nbasis,        # Updated npar_baseline
            nbasis + length(covar_pars),  # Updated npar_total
            hazard_fn,     # New function
            cumhaz_fn,     # New function
            haz.has_covariates,
            haz.covar_names,
            haz.degree,
            new_knots,     # Updated knots
            haz.natural_spline,
            haz.monotone,
            haz.extrapolation,
            haz.metadata,
            haz.shared_baseline_key
        )
        
        # Replace in model
        model.hazards[idx] = new_haz
    end
    
    return nothing
end
