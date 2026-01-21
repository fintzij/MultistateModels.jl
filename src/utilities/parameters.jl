# ============================================================================
# Parameter Manipulation Functions
# ============================================================================
# Functions for unflattening, setting, getting, and managing model parameters.
#
# PARAMETER SCALE CONVENTION (v0.3.0+):
# All parameters are stored on NATURAL scale. No transformations needed.
# Box constraints (lb/ub) enforce positivity instead of exp() transforms.
# ============================================================================

"""
    unflatten_parameters(flat_params::AbstractVector, model::MultistateProcess)

Unflatten parameter vector to nested NamedTuple structure.

This is the primary function for converting flat optimization parameters to the
nested NamedTuple format used by hazard evaluation functions.

# Arguments
- `flat_params`: Flat parameter vector (Float64 or Dual for AD)
- `model`: MultistateProcess model containing ReConstructor

# Returns
NamedTuple of nested parameters with structure:
- `(hazname = (baseline = (...), covariates = (...)), ...)`

# Example
```julia
flat_params = [0.5, 0.3]  # rate and covariate coefficient
# Returns: (h12 = (baseline = (h12_rate = 0.5,), covariates = (h12_x = 0.3,)),)
```
"""
function unflatten_parameters(flat_params::AbstractVector{T}, model::MultistateProcess) where {T<:Real}
    if T === Float64
        return unflatten(model.parameters.reconstructor, flat_params)
    else
        # For Dual or other types, use AD-compatible unflatten
        return unflattenAD(model.parameters.reconstructor, flat_params)
    end
end

"""
    update_pars_cache!(pars_cache, flat_params, hazards)

Update mutable parameter vectors in-place from flat parameter vector.
This avoids allocating new NamedTuples on each likelihood call.

For Float64 parameters only - AD types should use unflatten_parameters.

# Arguments
- `pars_cache::Vector{Vector{Float64}}`: Pre-allocated parameter vectors (one per hazard)
- `flat_params::Vector{Float64}`: Flat parameter vector (already on natural scale)
- `hazards::Vector{<:_Hazard}`: Hazard objects with npar_total info

# Note
As of v0.3.0, all parameters are stored on natural scale. No transformation needed.
"""
function update_pars_cache!(pars_cache::Vector{Vector{Float64}}, 
                            flat_params::AbstractVector{Float64}, 
                            hazards::Vector{<:_Hazard})
    offset = 0
    @inbounds for (h, hazard) in enumerate(hazards)
        npars = hazard.npar_total
        pars_h = pars_cache[h]
        
        # v0.3.0+: Copy all parameters directly (already on natural scale)
        for i in 1:npars
            pars_h[i] = flat_params[offset + i]
        end
        
        offset += npars
    end
    return nothing
end

"""
    compute_hazard_rates_cached!(rates_cache, pars_cache, covars_cache, hazards)

Compute hazard rates using cached parameters and covariates.
Updates rates_cache in-place for all (pattern, hazard) combinations.

# Arguments
- `rates_cache::Vector{Vector{Float64}}`: Pre-allocated rates [pattern][hazard]
- `pars_cache::Vector{Vector{Float64}}`: Cached parameter vectors [hazard]
- `covars_cache::Vector{Vector{NamedTuple}}`: Cached covariates [pattern][hazard]
- `hazards::Vector{<:_Hazard}`: Hazard objects

For Markov models, hazard rates don't depend on time, so we compute once per pattern.
"""
function compute_hazard_rates_cached!(rates_cache::Vector{Vector{Float64}},
                                      pars_cache::Vector{Vector{Float64}},
                                      covars_cache::Vector{Vector{NamedTuple}},
                                      hazards::Vector{<:_Hazard})
    npatterns = length(rates_cache)
    nhazards = length(hazards)
    
    @inbounds for p in 1:npatterns
        pattern_covars = covars_cache[p]
        pattern_rates = rates_cache[p]
        for h in 1:nhazards
            hazard = hazards[h]
            pars = pars_cache[h]
            covars = pattern_covars[h]
            # Evaluate hazard at t=0 (for Markov, rate is constant)
            pattern_rates[h] = _eval_hazard_from_vec(hazard, 0.0, pars, covars)
        end
    end
    return nothing
end

"""
    _eval_hazard_from_vec(hazard, t, pars_vec, covars)

Evaluate hazard using a flat parameter vector (natural scale) instead of NamedTuple.
This is the allocation-free path for cached parameters.
"""
@inline function _eval_hazard_from_vec(hazard::_Hazard, t::Real, 
                                        pars_vec::Vector{Float64}, 
                                        covars::NamedTuple)
    # Call the hazard with vector parameters directly
    return hazard(t, pars_vec, covars)
end

"""
    rebuild_parameters(new_param_vectors::Vector{Vector{Float64}}, model::MultistateProcess)

Rebuild the parameters structure from new parameter vectors.
v0.3.0+: Parameters are stored on NATURAL scale (rates, shapes as positive values).
Covariate coefficients are unconstrained.

# Arguments
- `new_param_vectors`: Vector of parameter vectors (one per hazard), on natural scale
- `model`: The model containing hazard info for npar_baseline

# Keyword Arguments
- `validate_bounds::Bool=true`: Whether to validate that parameters are within bounds

# Returns
- NamedTuple with flat, nested, and reconstructor fields

# Note
For phase-type models with shared hazards, `model.hazkeys` maps hazard names to 
*parameter* indices, not hazard indices. This function builds a reverse mapping
to find the correct hazard for each parameter index (which has the correct parnames).

Validates that new parameters are within model bounds before rebuilding.
"""
function rebuild_parameters(new_param_vectors::Vector{Vector{Float64}}, model::MultistateProcess; 
                            validate_bounds::Bool=true)
    # Build reverse mapping: parameter index -> hazard index
    # For phase-type models, multiple hazards may share the same parameter index,
    # but they all have the same parnames, so we just need one representative hazard.
    params_idx_to_hazard_idx = Dict{Int, Int}()
    for (haz_idx, h) in enumerate(model.hazards)
        params_idx = model.hazkeys[h.hazname]
        if !haskey(params_idx_to_hazard_idx, params_idx)
            params_idx_to_hazard_idx[params_idx] = haz_idx
        end
    end
    
    params_nested_pairs = [
        let haz_idx = params_idx_to_hazard_idx[params_idx]
            hazname => build_hazard_params(new_param_vectors[params_idx], model.hazards[haz_idx].parnames, model.hazards[haz_idx].npar_baseline, model.hazards[haz_idx].npar_total)
        end
        for (hazname, params_idx) in sort(collect(model.hazkeys), by = x -> x[2])
    ]
    
    params_nested = NamedTuple(params_nested_pairs)
    
    # Create ReConstructor for AD-compatible flatten/unflatten
    reconstructor = ReConstructor(params_nested, unflattentype=UnflattenFlexible())
    params_flat = flatten(reconstructor, params_nested)
    
    # Validate bounds before returning (v0.3.0+)
    if validate_bounds
        if hasproperty(model, :bounds) && !isnothing(model.bounds)
            validate_parameter_bounds!(params_flat, model)
        else
            @warn "Bounds validation skipped: model.bounds is nothing or missing. " *
                  "This can happen with manually constructed models or old serialized models. " *
                  "Consider rebuilding the model with multistatemodel() for full validation."
        end
    end
    
    # v0.3.0+: No separate .natural field - compute on-demand via accessors
    return (
        flat = params_flat,
        nested = params_nested,
        reconstructor = reconstructor
    )
end

# Tolerance for bound checking - accounts for solver precision issues
# Use relative tolerance for finite bounds, absolute for comparison with 0
# ATOL should be >= typical optimizer precision (Ipopt default tolerance is ~1e-8)
const BOUNDS_RTOL = 1e-6
const BOUNDS_ATOL = 1e-6

"""
    validate_parameter_bounds!(flat_params::AbstractVector, model::MultistateProcess; 
                               warn_only::Bool=false, rtol::Real=BOUNDS_RTOL, atol::Real=BOUNDS_ATOL)

Validate that parameters are within model bounds, with tolerance for numerical precision.

Uses a small tolerance to avoid false positives from optimizer precision issues.
A parameter `p` violates lower bound `lb` if `p < lb - max(atol, rtol * abs(lb))`.

# Arguments
- `flat_params`: Flat parameter vector to validate
- `model`: Model containing bounds
- `warn_only`: If true, issue warning instead of throwing error (default: false)
- `rtol`: Relative tolerance for bound comparison (default: 1e-6)
- `atol`: Absolute tolerance for bound comparison (default: 1e-10)

# Throws
- `ArgumentError` if any parameter violates bounds beyond tolerance (unless `warn_only=true`)
"""
function validate_parameter_bounds!(flat_params::AbstractVector, model::MultistateProcess; 
                                    warn_only::Bool=false, rtol::Real=BOUNDS_RTOL, atol::Real=BOUNDS_ATOL)
    lb = model.bounds.lb
    ub = model.bounds.ub
    
    # Check violations with tolerance
    # For lb: violation if p < lb - tol where tol = max(atol, rtol * |lb|)
    # For ub: violation if p > ub + tol where tol = max(atol, rtol * |ub|)
    violations_lb = Int[]
    violations_ub = Int[]
    
    for i in eachindex(flat_params)
        lb_tol = max(atol, rtol * abs(lb[i]))
        ub_tol = max(atol, rtol * abs(ub[i]))
        
        if isfinite(lb[i]) && flat_params[i] < lb[i] - lb_tol
            push!(violations_lb, i)
        end
        if isfinite(ub[i]) && flat_params[i] > ub[i] + ub_tol
            push!(violations_ub, i)
        end
    end
    
    if !isempty(violations_lb) || !isempty(violations_ub)
        parnames = reduce(vcat, [h.parnames for h in model.hazards])
        
        msg = "Parameter values violate bounds (beyond tolerance rtol=$rtol, atol=$atol):\n"
        
        for i in violations_lb
            pname = i <= length(parnames) ? string(parnames[i]) : "param[$i]"
            tol = max(atol, rtol * abs(lb[i]))
            msg *= "  $pname: value=$(flat_params[i]) < lb=$(lb[i]) (effective lb=$(lb[i] - tol))\n"
        end
        
        for i in violations_ub
            pname = i <= length(parnames) ? string(parnames[i]) : "param[$i]"
            tol = max(atol, rtol * abs(ub[i]))
            msg *= "  $pname: value=$(flat_params[i]) > ub=$(ub[i]) (effective ub=$(ub[i] + tol))\n"
        end
        
        if warn_only
            @warn msg
        else
            throw(ArgumentError(msg))
        end
    end
    
    return nothing
end

"""
    set_parameters!(model::MultistateProcess, newvalues::Vector{Vector{Float64}})

Set model parameters given a nested vector of values (one vector per hazard).
v0.3.0+: Values should be on NATURAL scale for baseline parameters (e.g., rate=0.5, shape=1.2).
Covariate coefficients are on their natural scale (unconstrained).

# Arguments
- `model::MultistateProcess`: The model to update
- `newvalues`: Nested vector with parameters for each hazard (natural scale for baseline)

# Note
Updates model.parameters with the new values and remakes spline hazards as needed.

# Example
```julia
# Set exponential rate to 0.5 (NOT log(0.5))
set_parameters!(model, [[0.5]])

# Set Weibull shape=1.5, scale=0.3, covariate effect=0.2
set_parameters!(model, [[1.5, 0.3, 0.2]])
```
"""
function set_parameters!(model::MultistateProcess, newvalues::Vector{Vector{Float64}})
    
    # Get number of hazards and expected parameter counts
    n_hazards = length(model.hazards)
    
    # check that we have the right number of parameters
    if length(newvalues) != n_hazards
        throw(ArgumentError("New values and model parameters are not of the same length. Expected $n_hazards, got $(length(newvalues))."))
    end

    for i in 1:n_hazards
        expected_len = model.hazards[i].npar_total
        if expected_len != length(newvalues[i])
            throw(ArgumentError("New values for hazard $i and model parameters for that hazard are not of the same length. " *
                               "Expected $expected_len, got $(length(newvalues[i]))."))
        end
    end
    
    # Rebuild parameters with bound validation
    model.parameters = rebuild_parameters(newvalues, model)
    
    return nothing
end

"""
    set_parameters!(model::MultistateProcess, newvalues::Tuple)

Set model parameters given a tuple of vectors parameterizing transition intensities.
v0.3.0+: Values should be on natural scale for baseline parameters (e.g., rates, shapes),
as-is for covariate coefficients.

Assigns new values by hazard index in order.
"""
function set_parameters!(model::MultistateProcess, newvalues::Tuple)
    # Get expected parameter counts from hazard objects
    n_hazards = length(model.hazards)
    
    # check that there is a vector of parameters for each cause-specific hazard
    if length(newvalues) != n_hazards
        throw(ArgumentError("Number of supplied parameter vectors not equal to number of transition intensities. Expected $n_hazards, got $(length(newvalues))."))
    end

    for i in eachindex(newvalues)
        # check that we have the right number of parameters
        expected_len = model.hazards[i].npar_total
        if expected_len != length(newvalues[i])
            throw(ArgumentError("New values and parameters for cause-specific hazard $i are not of the same length. " *
                               "Expected $expected_len, got $(length(newvalues[i]))."))
        end
    end
    
    # Convert tuple to vector and rebuild with bound validation
    new_param_vectors = [Vector{Float64}(collect(newvalues[i])) for i in 1:n_hazards]
    model.parameters = rebuild_parameters(new_param_vectors, model)
    
    return nothing
end

"""
    set_parameters!(model::MultistateProcess, newvalues::NamedTuple)

Set model parameters given a NamedTuple with hazard names as keys.
v0.3.0+: Values should be on natural scale for baseline parameters (e.g., rates, shapes),
as-is for covariate coefficients.

Assignment is made by matching tuple keys in `newvalues` to the key in `model.hazkeys`.
"""
function set_parameters!(model::MultistateProcess, newvalues::NamedTuple)
    # Validate parameter lengths using hazard info
    value_keys = keys(newvalues)

    for k in eachindex(value_keys)
        vind = value_keys[k]
        mind = model.hazkeys[vind]
        expected_len = model.hazards[mind].npar_total

        # check length of supplied parameters
        if length(newvalues[vind]) != expected_len
            throw(ArgumentError("The new parameter values for $vind are not the expected length. " *
                               "Expected $expected_len, got $(length(newvalues[vind]))."))
        end
    end
    
    # Rebuild full parameters, incorporating partial updates
    # Start with current values and overlay the new ones
    new_param_vectors = Vector{Vector{Float64}}(undef, length(model.hazards))
    for (hazname, idx) in model.hazkeys
        if hazname in value_keys
            new_param_vectors[idx] = Vector{Float64}(collect(newvalues[hazname]))
        else
            # Keep current values - extract from nested structure
            new_param_vectors[idx] = extract_params_vector(model.parameters.nested[idx])
        end
    end
    
    model.parameters = rebuild_parameters(new_param_vectors, model)
    
    return nothing
end

"""
    set_parameters!(model::MultistateProcess, h::Int64, newvalues::Vector{Float64})

Set parameters for a single hazard by hazard index.
v0.3.0+: Values should be on NATURAL scale for baseline parameters (e.g., rate=0.5).
Covariate coefficients are on their natural scale (unconstrained).

# Arguments
- `model::MultistateProcess`: The model to update
- `h::Int64`: Index of the hazard to update
- `newvalues::Vector{Float64}`: New parameter values (natural scale for baseline, as-is for covariates)

# Example
```julia
# Update hazard 1 with exponential rate = 0.5
set_parameters!(model, 1, [0.5])

# Update hazard with Weibull shape=2.0, scale=1.5, covariate effects
set_parameters!(model, 2, [2.0, 1.5, 0.3, -0.2])
```
"""
function set_parameters!(model::MultistateProcess, h::Int64, newvalues::Vector{Float64})
    n_hazards = length(model.hazards)
    
    # Check hazard index
    if h < 1 || h > n_hazards
        throw(BoundsError(model.hazards, h))
    end
    
    # Check parameter length
    expected_len = model.hazards[h].npar_total
    if length(newvalues) != expected_len
        throw(ArgumentError("New values length ($(length(newvalues))) does not match expected length ($expected_len) for hazard $h."))
    end
    
    # Build new parameter vectors (keep current for other hazards)
    new_param_vectors = Vector{Vector{Float64}}(undef, n_hazards)
    for idx in 1:n_hazards
        if idx == h
            new_param_vectors[idx] = Vector{Float64}(newvalues)
        else
            # Keep current values - extract from nested structure
            new_param_vectors[idx] = extract_params_vector(model.parameters.nested[idx])
        end
    end
    
    model.parameters = rebuild_parameters(new_param_vectors, model)
    
    return nothing
end

"""
    get_subjinds(data::DataFrame)

Return a vector with the row indices for each subject in the dataset.

Subject indices are assumed to be contiguous (each subject's rows are consecutive).
This is validated and a warning is issued if non-contiguous indices are detected,
as this may indicate data sorting issues that could affect likelihood computation.

# Returns
- `subjinds`: Vector of vectors, where `subjinds[i]` contains row indices for subject i
- `nsubj`: Number of unique subjects

# Warning
Issues a warning if any subject's row indices are non-contiguous (not consecutive),
which may indicate the data needs to be sorted by subject ID.
"""
function get_subjinds(data::DataFrame)

    # number of subjects
    ids = unique(data.id)
    nsubj = length(ids)

    # initialize vector of indices
    subjinds = Vector{Vector{Int64}}(undef, nsubj)

    # get indices for each subject
    for i in eachindex(ids)
        subjinds[i] = findall(x -> x == ids[i], data.id)
    end
    
    # Validate that indices are contiguous (consecutive)
    # Non-contiguous indices indicate unsorted data which can cause issues
    for i in eachindex(subjinds)
        inds = subjinds[i]
        if length(inds) > 1
            # Check if indices are consecutive: diff should be all 1s
            if any(diff(inds) .!= 1)
                @warn "Subject $(ids[i]) has non-contiguous row indices: $(inds). " *
                      "This may indicate the data is not sorted by subject ID. " *
                      "Consider sorting the data: sort!(data, :id) before model construction."
                break  # Only warn once to avoid flooding output
            end
        end
    end

    # return indices
    return subjinds, nsubj
end

# ============================================================================
# Parameter Construction Functions
# ============================================================================

"""
    build_hazard_params(params, parnames, npar_baseline, npar_total)

Create a NamedTuple with baseline and covariate parameters using named fields.

v0.3.0+: All parameters are stored on natural scale. No transformations are applied.
Baseline parameters are positive-constrained via Ipopt box constraints during optimization.

This function is type-generic to support ForwardDiff.Dual during automatic differentiation.

# Arguments
- `params`: Full parameter vector - natural scale for baseline, as-is for covariates
- `parnames`: Vector of parameter names (e.g., [:h12_shape, :h12_scale, :h12_age])
- `npar_baseline`: Number of baseline parameters
- `npar_total`: Total number of parameters

# Returns
- `NamedTuple`: `(baseline = (h12_shape=..., h12_scale=...), covariates = (h12_age=...,))` 
  or just `(baseline = (h12_shape=..., h12_scale=...),)` if no covariates

# Examples
```julia
# Weibull with covariates (natural scale)
params = build_hazard_params([1.5, 0.2, 0.3, 0.1], 
                             [:h12_shape, :h12_scale, :h12_age, :h12_sex], 2, 4)
# Returns: (baseline = (h12_shape = 1.5, h12_scale = 0.2), 
#           covariates = (h12_age = 0.3, h12_sex = 0.1))

# Exponential without covariates  
params = build_hazard_params([0.5], [:h12_rate], 1, 1)
# Returns: (baseline = (h12_rate = 0.5),)
```

# Note
All parameters are stored on natural scale (as expected by hazard functions).
Covariate coefficients are stored as-is (unconstrained).
"""
function build_hazard_params(params::AbstractVector{<:Real}, parnames::Vector{Symbol}, npar_baseline::Int, npar_total::Int)
    @assert length(params) == npar_total "Parameter vector length mismatch"
    @assert length(parnames) == npar_total "Parameter names length must match parameter vector length"
    @assert npar_baseline <= npar_total "Baseline parameters cannot exceed total parameters"
    
    # Extract baseline parameter names and values (natural scale)
    baseline_names = parnames[1:npar_baseline]
    baseline_values = params[1:npar_baseline]
    
    # Create NamedTuple for baseline with named fields
    baseline = NamedTuple{Tuple(baseline_names)}(baseline_values)
    
    # Handle covariates if present
    if npar_total > npar_baseline
        covar_names = parnames[(npar_baseline+1):npar_total]
        covar_values = params[(npar_baseline+1):npar_total]
        covariates = NamedTuple{Tuple(covar_names)}(covar_values)
        return (baseline = baseline, covariates = covariates)
    else
        return (baseline = baseline,)
    end
end

"""
    extract_baseline_values(hazard_params::NamedTuple)

Extract baseline parameter values as a vector from NamedTuple structure.

# Arguments
- `hazard_params`: NamedTuple with `baseline` field containing named parameters

# Returns
- Vector of baseline parameter values in order
"""
function extract_baseline_values(hazard_params::NamedTuple)
    return collect(values(hazard_params.baseline))
end

"""
    extract_covariate_values(hazard_params::NamedTuple)

Extract covariate coefficient values as a vector from NamedTuple structure.

# Arguments
- `hazard_params`: NamedTuple with optional `covariates` field

# Returns
- Vector of covariate coefficient values, or empty vector if no covariates
"""
function extract_covariate_values(hazard_params::NamedTuple)
    return haskey(hazard_params, :covariates) ? 
           collect(values(hazard_params.covariates)) : Float64[]
end

"""
    extract_params_vector(hazard_params)

Extract the full parameter vector from a hazard's NamedTuple params structure.
Returns values in the scale they are stored (natural scale after transformation).

# Arguments
- `hazard_params`: NamedTuple with `baseline` (named NamedTuple) and optionally `covariates` (named NamedTuple)

# Returns
- Vector of all parameter values in order (baseline values first, then covariate values)

# Note
After the unflatten transformation, baseline values are on natural scale (exp applied).
Covariate coefficients are unchanged (unconstrained).
"""
function extract_params_vector(hazard_params::NamedTuple)
    baseline_vals = collect(values(hazard_params.baseline))
    if haskey(hazard_params, :covariates)
        covar_vals = collect(values(hazard_params.covariates))
        return vcat(baseline_vals, covar_vals)
    else
        return baseline_vals
    end
end

"""
    extract_natural_vector(hazard_params, family::Symbol)

IDENTITY TRANSFORM (v0.3.0+): Extracts parameter values without transformation.

Previously applied exp() transforms to baseline parameters.
Now all parameters are stored on natural scale, so this just extracts and concatenates.

# Arguments
- `hazard_params`: NamedTuple with `baseline` (named NamedTuple) and optionally `covariates` (named NamedTuple)
- `family`: Hazard family (unused, kept for API compatibility)

# Returns
- Vector with baseline values followed by covariate coefficients (all unchanged)
"""
function extract_natural_vector(hazard_params::NamedTuple, family::Symbol)
    baseline_vals = collect(values(hazard_params.baseline))
    
    # v0.3.0+: All parameters on natural scale, no exp() transformation
    baseline_natural = baseline_vals
    
    if haskey(hazard_params, :covariates)
        covar_vals = collect(values(hazard_params.covariates))
        return vcat(baseline_natural, covar_vals)
    else
        return baseline_natural
    end
end

"""
    get_estimation_scale_params(parameters)

Extract nested parameters from a parameters structure (v0.3.0+: returns natural scale).

As of v0.3.0, there is no separate "estimation scale" - all parameters are stored on
natural scale. This function extracts the `nested` field for backward compatibility.

# Arguments
- `parameters`: A NamedTuple with `nested` field, or already-extracted nested parameters

# Returns
- `NamedTuple`: Parameters indexed by hazard name (on natural scale)

# Note
As of v0.3.0, this returns the same values as get_hazard_params (both on natural scale).
The function name is kept for backward compatibility with existing code.

# Example
```julia
# Get params (now on natural scale)
pars = get_estimation_scale_params(model.parameters)
# pars[:h12].baseline contains natural-scale values
```
"""
function get_estimation_scale_params(parameters)
    # If parameters has a 'nested' field, extract it; otherwise assume already extracted
    if hasfield(typeof(parameters), :nested)
        return parameters.nested
    else
        # Already nested parameters (e.g., passed directly from simulation internals)
        return parameters
    end
end

"""
    get_hazard_params(parameters, hazards)

Extract parameters for hazard function evaluation.

As of v0.3.0, all parameters are stored on natural scale, so this function
simply extracts the nested parameters without transformation.

# Arguments
- `parameters`: A NamedTuple with `nested` field, or already-extracted nested parameters
- `hazards`: Vector of hazard objects (kept for API compatibility)

# Returns
- `NamedTuple`: Parameters indexed by hazard name (on natural scale)

# Example
```julia
# Get params for hazard evaluation
haz_pars = get_hazard_params(model.parameters, model.hazards)
# haz_pars[:h12].baseline contains natural-scale values
# Pass to hazard: eval_hazard(hazard, t, haz_pars[:h12], covars)
```
"""
function get_hazard_params(parameters, hazards)
    # v0.3.0+: params already on natural scale, just extract
    return get_estimation_scale_params(parameters)
end

"""
    get_hazard_params_indexed(parameters, hazards)

Extract natural-scale parameters as a Tuple indexed by hazard position.

This is an optimized version of `get_hazard_params` that returns a Tuple instead
of a NamedTuple. When iterating over hazards by index (e.g., `for h in 1:nhazards`),
use `params_indexed[h]` instead of `params[hazard.hazname]` to avoid runtime
Symbol lookup overhead.

# Performance
For a NamedTuple with dynamic symbol access like `params[hazard.hazname]`, Julia
must perform a runtime lookup. With indexed access, `params_indexed[h]` compiles
to a direct tuple element access, which is ~5-10x faster per access.

# Arguments
- `parameters`: A NamedTuple with `nested` field, or already-extracted nested parameters
- `hazards`: Vector of hazard objects (used to determine transformation per family)

# Returns
- `Tuple`: Natural-scale parameters indexed by hazard position (1-based)

# Example
```julia
params_indexed = get_hazard_params_indexed(model.parameters, model.hazards)
# In hot loop:
for h in totalhazard.components
    hazard_pars = params_indexed[h]  # Fast indexed access
    cumhaz = eval_cumhaz(hazard, lb, ub, hazard_pars, covars)
end
```

See also: [`get_hazard_params`](@ref)
"""
@inline function get_hazard_params_indexed(parameters, hazards)
    params_named = get_hazard_params(parameters, hazards)
    # Convert NamedTuple to Tuple for indexed access
    # values() returns a Tuple of the NamedTuple values
    return values(params_named)
end

# AD-compatible version that preserves element types
@inline function get_hazard_params_indexed(parameters, hazards, ::Type{T}) where T
    # v0.3.0+: params already on natural scale, just extract and convert to Tuple
    est_params = get_estimation_scale_params(parameters)
    return values(est_params)
end
