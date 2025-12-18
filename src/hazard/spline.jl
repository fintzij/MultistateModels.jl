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
# Spline Knot Calibration Functions
# =============================================================================

"""
    calibrate_splines(model::MultistateProcess; 
                      quantiles=nothing, 
                      nknots=nothing,
                      n_paths::Int=1000,
                      min_ess::Int=100,
                      verbose::Bool=true) -> NamedTuple

Compute recommended knot locations for spline hazards based on transition times.

For exact data (obstype 1 or 3), uses observed sojourn times directly.
For panel data (obstype 2), fits a Markov surrogate, simulates sample paths,
and uses the simulated transition times to determine knot locations.

# Arguments
- `model::MultistateProcess`: An unfitted model with one or more spline hazards.
- `quantiles`: Interior knot quantile levels. Can be:
  - `Vector{Float64}`: Quantile levels (e.g., `[0.25, 0.5, 0.75]`) applied to all hazards
  - `NamedTuple`: Per-hazard quantiles, e.g., `(h12 = [0.25, 0.5], h23 = [0.33, 0.67])`
  - `nothing` (default): Use `nknots` to generate evenly-spaced quantiles
- `nknots`: Number of interior knots. Can be:
  - `Int`: Number of interior knots for all hazards
  - `NamedTuple`: Per-hazard counts, e.g., `(h12 = 3, h23 = 2)`
  - `nothing` (default): Use `floor(n^(1/5))`
- `n_paths::Int=1000`: Number of paths to sample per subject (panel data only)
- `min_ess::Int=100`: Minimum effective sample size for importance sampling (panel data only)
- `verbose::Bool=true`: Print info about knot placement

# Returns
A `NamedTuple` with one entry per spline hazard, each containing:
- `boundary_knots::Vector{Float64}`: `[0.0, max_sojourn]` for the transition
- `interior_knots::Vector{Float64}`: Recommended interior knot locations

# Errors
- `ArgumentError` if model is a `MultistateModelFitted`
- `ArgumentError` if model has no spline hazards
- `ArgumentError` if both `quantiles` and `nknots` are specified

# Notes
- For exact data: uses observed sojourn times directly (no surrogate fitting)
- For panel data: fits Markov surrogate, simulates paths, extracts transition times
- Lower boundary is always 0.0 (sojourns are non-negative)
- Upper boundary is the maximum observed/simulated sojourn time
- For ties, knots are spread evenly using [`place_interior_knots`](@ref)

# Example
```julia
h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; degree=3)
h23 = Hazard(@formula(0 ~ 1), "sp", 2, 3; degree=3)
model = multistatemodel(h12, h23; data=data)

# Auto-select nknots based on sample size
knots = calibrate_splines(model)

# Specify number of knots
knots = calibrate_splines(model; nknots=3)

# Different knots per hazard
knots = calibrate_splines(model; nknots=(h12=4, h23=2))

# Specify explicit quantiles
knots = calibrate_splines(model; quantiles=[0.2, 0.4, 0.6, 0.8])
```

See also: [`calibrate_splines!`](@ref), [`place_interior_knots`](@ref)
"""
function calibrate_splines(model::MultistateProcess;
                           quantiles::Union{Vector{Float64}, NamedTuple, Nothing}=nothing,
                           nknots::Union{Int, NamedTuple, Nothing}=nothing,
                           n_paths::Int=1000,
                           min_ess::Int=100,
                           verbose::Bool=true)
    
    # Validation
    if model isa MultistateModelFitted
        throw(ArgumentError("Cannot calibrate splines on a fitted model. Use an unfitted MultistateModel."))
    end
    
    if !isnothing(quantiles) && !isnothing(nknots)
        throw(ArgumentError("Specify either `quantiles` or `nknots`, not both."))
    end
    
    # Find spline hazards
    spline_indices = findall(h -> h isa RuntimeSplineHazard, model.hazards)
    if isempty(spline_indices)
        throw(ArgumentError("Model has no spline hazards to calibrate."))
    end
    
    # Determine if data is exact or panel
    has_exact_data = _has_exact_transitions(model)
    
    # Get sojourns by transition
    if has_exact_data
        # Use observed data directly
        verbose && @info "Using observed transition times for knot calibration"
        sojourns_by_transition = _extract_sojourns_from_data(model)
    else
        # Fit surrogate and simulate paths
        verbose && @info "Fitting Markov surrogate and simulating paths for knot calibration"
        sojourns_by_transition = _extract_sojourns_from_surrogate(model, n_paths, min_ess)
    end
    
    # Compute knots for each spline hazard
    results = Dict{Symbol, NamedTuple}()
    
    for idx in spline_indices
        haz = model.hazards[idx]
        hazname = haz.hazname
        key = (haz.statefrom, haz.stateto)
        
        # Get sojourns for this transition
        sojourns = get(sojourns_by_transition, key, Float64[])
        
        if isempty(sojourns)
            @warn "No transitions $(key[1])→$(key[2]) for hazard $hazname; using model boundaries"
            bknots = [haz.knots[1], haz.knots[end]]
            results[hazname] = (boundary_knots=bknots, interior_knots=Float64[])
            continue
        end
        
        # Determine number of knots
        nk = _get_nknots_for_hazard(hazname, nknots, length(sojourns))
        
        # Determine quantile levels or compute from nknots
        qlevels = _get_quantiles_for_hazard(hazname, quantiles, nk)
        
        # Compute boundary knots
        lb = 0.0  # Lower boundary is always 0 for sojourns
        ub = maximum(sojourns)
        
        # Compute interior knots
        if qlevels !== nothing
            # User-specified quantiles
            interior = quantile(sojourns, qlevels)
            interior = unique(interior)
        else
            # Use place_interior_knots for automatic placement with tie handling
            interior = place_interior_knots(sojourns, nk; lower_bound=lb, upper_bound=ub)
        end
        
        verbose && @info "Calibrated $(length(interior)) interior knots for $hazname at: $(round.(interior, digits=3))"
        results[hazname] = (boundary_knots=[lb, ub], interior_knots=interior)
    end
    
    return NamedTuple(results)
end

"""
    calibrate_splines!(model::MultistateProcess; quantiles=nothing, nknots=nothing, 
                       n_paths=1000, min_ess=100, verbose=true)

Compute and apply knot locations for spline hazards in-place.

Modifies the spline hazards in `model` to use the computed knot locations.
See [`calibrate_splines`](@ref) for argument details.

# Arguments
Same as [`calibrate_splines`](@ref).

# Returns
`NamedTuple` with computed knot locations (same as [`calibrate_splines`](@ref)).

# Notes
- Modifies `model.hazards` in place for all `RuntimeSplineHazard` hazards
- Also rebuilds the parameter structure to match the new number of basis functions
- After calling this, the model's spline hazards will have updated knots

# Example
```julia
model = multistatemodel(h12, h23; data=data)
knots = calibrate_splines!(model; nknots=5)  # Updates knots in-place
fitted = fit(model)  # Fit with calibrated knots
```

See also: [`calibrate_splines`](@ref)
"""
function calibrate_splines!(model::MultistateProcess;
                            quantiles::Union{Vector{Float64}, NamedTuple, Nothing}=nothing,
                            nknots::Union{Int, NamedTuple, Nothing}=nothing,
                            n_paths::Int=1000,
                            min_ess::Int=100,
                            verbose::Bool=true)
    
    # Get recommended knots (this also does validation)
    knots = calibrate_splines(model; quantiles=quantiles, nknots=nknots, 
                              n_paths=n_paths, min_ess=min_ess, verbose=verbose)
    
    # Build knot_locations dict for _rebuild_model_with_knots!
    knot_locations = Dict{Tuple{Int,Int}, Vector{Float64}}()
    for idx in findall(h -> h isa RuntimeSplineHazard, model.hazards)
        haz = model.hazards[idx]
        result = knots[haz.hazname]
        knot_locations[(haz.statefrom, haz.stateto)] = result.interior_knots
    end
    
    # Rebuild model with new knots
    _rebuild_model_with_knots!(model, knot_locations)
    
    # Update model parameters to match new hazard dimensions
    _rebuild_model_parameters!(model)
    
    return knots
end

"""
    _rebuild_model_with_knots!(model::MultistateProcess, knot_locations::Dict{Tuple{Int,Int}, Vector{Float64}})

Rebuild spline hazards in-place with new interior knot locations.

This function creates new `RuntimeSplineHazard` objects with updated knots
and replaces them in the model's hazard vector. The boundary knots are preserved
from the original hazards.

# Arguments
- `model::MultistateProcess`: The model to modify
- `knot_locations::Dict{Tuple{Int,Int}, Vector{Float64}}`: Map from (statefrom, stateto) 
  to new interior knot locations

# Notes
- Only modifies `RuntimeSplineHazard` entries
- Preserves all other hazard properties (degree, natural_spline, monotone, etc.)
- After calling this, `_rebuild_model_parameters!` should be called to update parameters
"""
function _rebuild_model_with_knots!(model::MultistateProcess, 
                                     knot_locations::Dict{Tuple{Int,Int}, Vector{Float64}})
    
    for (idx, haz) in enumerate(model.hazards)
        # Only process spline hazards
        haz isa RuntimeSplineHazard || continue
        
        key = (haz.statefrom, haz.stateto)
        haskey(knot_locations, key) || continue
        
        new_interior_knots = knot_locations[key]
        
        # Preserve boundary knots from existing hazard
        old_knots = haz.knots
        bknots = [old_knots[1], old_knots[end]]
        
        # Combine and sort knots
        allknots = unique(sort([bknots[1]; new_interior_knots; bknots[2]]))
        
        # Build new B-spline basis with same settings as original
        B = BSplineBasis(BSplineOrder(haz.degree + 1), copy(allknots))
        
        # Apply boundary conditions via basis recombination (same as original)
        use_constant = haz.extrapolation == "constant"
        if use_constant && (haz.degree >= 2)
            B = RecombinedBSplineBasis(B, Derivative(1))
        elseif (haz.degree > 1) && haz.natural_spline
            B = RecombinedBSplineBasis(B, Natural())
        end
        
        # Determine extrapolation method
        extrap_method = if haz.extrapolation == "linear"
            BSplineKit.SplineExtrapolations.Linear()
        else
            BSplineKit.SplineExtrapolations.Flat()
        end
        
        # Number of basis functions = number of spline coefficients
        nbasis = length(B)
        
        # Build new parameter names
        n_covar = length(haz.covar_names)
        baseline_names = [Symbol(string(haz.hazname), "_sp", i) for i in 1:nbasis]
        covar_parnames = [Symbol(string(haz.hazname), "_", cn) for cn in haz.covar_names]
        parnames = vcat(baseline_names, covar_parnames)
        npar_total = nbasis + n_covar
        
        # Generate new hazard and cumhaz functions
        hazard_fn, cumhaz_fn = _generate_spline_hazard_fns(
            B, extrap_method, haz.monotone, nbasis, parnames, haz.metadata.linpred_effect
        )
        
        # Create new RuntimeSplineHazard with updated knots
        new_haz = RuntimeSplineHazard(
            haz.hazname,
            haz.statefrom,
            haz.stateto,
            haz.family,
            parnames,
            nbasis,
            npar_total,
            hazard_fn,
            cumhaz_fn,
            haz.has_covariates,
            haz.covar_names,
            haz.degree,
            allknots,
            haz.natural_spline,
            haz.monotone,
            haz.extrapolation,
            haz.metadata,
            haz.shared_baseline_key,
        )
        
        # Replace in model
        model.hazards[idx] = new_haz
    end
    
    # Also update totalhazards if they reference spline hazards
    _rebuild_totalhazards!(model)
    
    return nothing
end

"""
    _rebuild_totalhazards!(model::MultistateProcess)

Rebuild the totalhazards structure after hazard modifications.
"""
function _rebuild_totalhazards!(model::MultistateProcess)
    # Rebuild totalhazards from scratch using current hazards
    model.totalhazards = build_totalhazards(model.hazards, model.tmat)
    return nothing
end

# Import _generate_spline_hazard_fns from modelgeneration - it's already available
# since smooths.jl is included after modelgeneration.jl

"""
    _extract_sojourns_from_data(model::MultistateProcess) -> Dict{Tuple{Int,Int}, Vector{Float64}}

Extract observed sojourn times for each transition from the model's data.

Returns dictionary mapping (statefrom, stateto) to vector of sojourn times.
Only includes exact transitions (obstype 1 or 3) where the transition actually occurred.
"""
function _extract_sojourns_from_data(model::MultistateProcess)
    data = model.data
    result = Dict{Tuple{Int,Int}, Vector{Float64}}()
    
    for row in eachrow(data)
        # Only exact transitions
        obstype = row.obstype
        if !(obstype == 1 || obstype == 3)
            continue
        end
        
        statefrom = row.statefrom
        stateto = row.stateto
        
        # Skip if no actual transition
        if statefrom == stateto
            continue
        end
        
        # Sojourn time
        sojourn = row.tstop - row.tstart
        
        key = (statefrom, stateto)
        if !haskey(result, key)
            result[key] = Float64[]
        end
        push!(result[key], sojourn)
    end
    
    return result
end

"""
    _has_exact_transitions(model::MultistateProcess) -> Bool

Check if the model data contains any exact transition observations (obstype 1 or 3).
"""
function _has_exact_transitions(model::MultistateProcess)
    data = model.data
    for row in eachrow(data)
        obstype = row.obstype
        if (obstype == 1 || obstype == 3) && row.statefrom != row.stateto
            return true
        end
    end
    return false
end

"""
    _extract_sojourns_from_surrogate(model::MultistateProcess, n_paths::Int, min_ess::Int) 
        -> Dict{Tuple{Int,Int}, Vector{Float64}}

Fit a Markov surrogate to the model, simulate sample paths, and extract sojourn times.

Used for panel data where exact transition times are not observed.
"""
function _extract_sojourns_from_surrogate(model::MultistateProcess, n_paths::Int, min_ess::Int)
    # Fit Markov surrogate
    surrogate_fitted = fit_surrogate(model; verbose=false)
    
    # Draw sample paths
    path_result = draw_paths(surrogate_fitted; min_ess=min_ess)
    
    # Extract paths - handle different return types from draw_paths
    # For panel data: returns NamedTuple with samplepaths::Vector{Vector{SamplePath}}
    # For exact data: returns NamedTuple with loglik (paths come from data)
    all_paths = if path_result isa NamedTuple && haskey(path_result, :samplepaths)
        # Panel data: NamedTuple with samplepaths field - flatten nested vectors
        SamplePath[p for subj_paths in path_result.samplepaths for p in subj_paths]
    elseif path_result isa NamedTuple && haskey(path_result, :loglik)
        # Exact data case (shouldn't happen here, but handle gracefully)
        extract_paths(surrogate_fitted.data)
    elseif path_result isa Vector{<:Vector}
        # Direct Vector{Vector{SamplePath}} - flatten
        SamplePath[p for subj_paths in path_result for p in subj_paths]
    else
        # Already a flat vector
        path_result
    end
    
    # Extract sojourns for each transition
    result = Dict{Tuple{Int,Int}, Vector{Float64}}()
    
    # Get all possible transitions from tmat
    tmat = model.tmat
    n_states = size(tmat, 1)
    for i in 1:n_states
        for j in 1:n_states
            if tmat[i, j] != 0
                sojourns = extract_sojourns(i, j, all_paths)
                if !isempty(sojourns)
                    result[(i, j)] = sojourns
                end
            end
        end
    end
    
    return result
end

"""
    _get_nknots_for_hazard(hazname::Symbol, nknots, n_obs::Int) -> Int

Determine number of interior knots for a specific hazard.
"""
function _get_nknots_for_hazard(hazname::Symbol, 
                                 nknots::Union{Int, NamedTuple, Nothing},
                                 n_obs::Int)
    if nknots === nothing
        return default_nknots(n_obs)
    elseif nknots isa Int
        return nknots
    else
        # NamedTuple - look up by hazard name
        return haskey(nknots, hazname) ? nknots[hazname] : default_nknots(n_obs)
    end
end

"""
    _get_quantiles_for_hazard(hazname::Symbol, quantiles, nknots::Int) -> Union{Vector{Float64}, Nothing}

Determine quantile levels for a specific hazard, or nothing if using automatic placement.
"""
function _get_quantiles_for_hazard(hazname::Symbol,
                                    quantiles::Union{Vector{Float64}, NamedTuple, Nothing},
                                    nknots::Int)
    if quantiles === nothing
        return nothing  # Use place_interior_knots for automatic placement
    elseif quantiles isa Vector{Float64}
        return quantiles
    else
        # NamedTuple - look up by hazard name
        return haskey(quantiles, hazname) ? quantiles[hazname] : nothing
    end
end

"""
    _rebuild_model_parameters!(model::MultistateProcess)

Rebuild model parameter structure after spline hazard modification.

When spline hazards are rebuilt with different knot counts, the number of
basis functions changes. This function rebuilds the parameter structure
to match the new hazard dimensions, using the same `rebuild_parameters` function
that `set_parameters!` uses to ensure consistency.

# Notes
- Parameters are initialized to zero on log scale (rates = 1.0)
- The resulting parameter structure includes a proper `reconstructor` field
  for AD-compatible flatten/unflatten operations
"""
function _rebuild_model_parameters!(model::MultistateProcess)
    # Build new parameter vectors (zeros on log scale) for each hazard
    new_param_vectors = Vector{Vector{Float64}}(undef, length(model.hazards))
    for i in eachindex(model.hazards)
        haz = model.hazards[i]
        new_param_vectors[i] = zeros(Float64, haz.npar_total)
    end
    
    # Use rebuild_parameters to create proper parameter structure with reconstructor
    model.parameters = rebuild_parameters(new_param_vectors, model)
    
    return nothing
end

"""
    _build_unflatten_function(hazards::Vector{<:_Hazard})

Build a function that converts flat parameter vector to nested structure.
"""
function _build_unflatten_function(hazards::Vector{<:_Hazard})
    # Precompute offsets and sizes
    offsets = Int[]
    sizes = Int[]
    baselines = Int[]
    haznames = Symbol[]
    
    offset = 0
    for haz in hazards
        push!(offsets, offset)
        push!(sizes, haz.npar_total)
        push!(baselines, haz.npar_baseline)
        push!(haznames, haz.hazname)
        offset += haz.npar_total
    end
    
    function unflatten(flat::AbstractVector)
        result = Dict{Symbol, NamedTuple}()
        for i in eachindex(hazards)
            start_idx = offsets[i] + 1
            end_idx = offsets[i] + sizes[i]
            haz_params = flat[start_idx:end_idx]
            baseline = haz_params[1:baselines[i]]
            covariates = sizes[i] > baselines[i] ? haz_params[(baselines[i]+1):end] : eltype(flat)[]
            result[haznames[i]] = (baseline=baseline, covariates=covariates)
        end
        return NamedTuple(result)
    end
    
    return unflatten
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
