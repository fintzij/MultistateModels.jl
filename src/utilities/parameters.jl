# ============================================================================
# Parameter Manipulation Functions
# ============================================================================
# Functions for unflattening, setting, getting, and managing model parameters.
# Handles AD-compatible transformations and cached parameter updates.
# ============================================================================

"""
    unflatten_natural(flat_params::AbstractVector, model::MultistateProcess)

Unflatten parameter vector to nested NamedTuple and transform to NATURAL SCALE.

This is the primary function for converting flat optimization parameters to the
nested NamedTuple format used by hazard evaluation functions. It applies the
appropriate scale transformation (exp for baseline parameters).

# What it does
1. Unflattens the flat parameter vector to nested NamedTuple structure
2. Transforms baseline parameters from estimation scale (log) to natural scale (exp)

# When to use
- Likelihood computation (hazard evaluation requires natural-scale parameters)
- Simulation (hazard rates must be positive)
- Any place that calls `eval_hazard`, `eval_cumhaz`, etc.

# Arguments
- `flat_params`: Flat parameter vector on estimation scale (Float64 or Dual)
- `model`: MultistateProcess model containing ReConstructor

# Returns
NamedTuple of nested parameters on NATURAL SCALE:
- Baseline parameters: exp() applied (positive values)
- Covariate coefficients: unchanged (already unconstrained)

# Note
Uses AD-compatible transformation that preserves ForwardDiff.Dual types.

# Example
```julia
# flat_params = [log(0.5), 0.3]  # log-rate and covariate coefficient
# Returns: (h12 = (baseline = (h12_Intercept = 0.5,), covariates = (h12_x = 0.3,)),)
#          ↑ exp(log(0.5)) = 0.5 on natural scale
```

See also: [`unflatten_estimation`](@ref)
"""
function unflatten_natural(flat_params::AbstractVector{T}, model::MultistateProcess) where {T<:Real}
    # Unflatten to estimation-scale nested structure
    if T === Float64
        params_estimation = unflatten(model.parameters.reconstructor, flat_params)
    else
        # For Dual or other types, use AD-compatible unflatten
        params_estimation = unflattenAD(model.parameters.reconstructor, flat_params)
    end
    
    # Transform baseline parameters to natural scale (exp applied where needed)
    # Pass hazards so we know which parameters need transformation
    return to_natural_scale(params_estimation, model.hazards, T)
end

"""
    update_pars_cache!(pars_cache, flat_params, hazards)

Update mutable parameter vectors in-place from flat parameter vector.
This avoids allocating new NamedTuples on each likelihood call.

For Float64 parameters only - AD types should use unflatten_natural.

# Arguments
- `pars_cache::Vector{Vector{Float64}}`: Pre-allocated parameter vectors (one per hazard)
- `flat_params::Vector{Float64}`: Flat parameter vector on estimation scale
- `hazards::Vector{<:_Hazard}`: Hazard objects with npar_total info

The function transforms baseline parameters to natural scale (exp) in-place.
"""
function update_pars_cache!(pars_cache::Vector{Vector{Float64}}, 
                            flat_params::AbstractVector{Float64}, 
                            hazards::Vector{<:_Hazard})
    offset = 0
    @inbounds for (h, hazard) in enumerate(hazards)
        npars = hazard.npar_total
        nbase = hazard.npar_baseline
        pars_h = pars_cache[h]
        
        # Copy baseline parameters with exp transformation (natural scale)
        for i in 1:nbase
            pars_h[i] = exp(flat_params[offset + i])
        end
        
        # Copy covariate parameters as-is
        for i in (nbase + 1):npars
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

# Backward compatibility aliases (deprecated)
# Use unflatten_natural instead
const unflatten_parameters = unflatten_natural
const safe_unflatten = unflatten_natural

"""
    unflatten_estimation(flat_params, model)

Unflatten parameter vector to nested NamedTuple structure on ESTIMATION SCALE.
Returns parameters with baseline on log scale (no exp() transformation applied).

This function is used when you need the raw estimation-scale parameters, such as:
- Spline remake code which needs log-scale coefficients
- Constraint checking before optimization
- Parameter initialization

Unlike `unflatten_natural`, this does NOT apply exp() to baseline parameters.

# Arguments
- `flat_params`: Flat parameter vector on estimation scale (Float64 or Dual)
- `model`: MultistateProcess model containing ReConstructor

# Returns
NamedTuple of nested parameters on ESTIMATION SCALE:
- Baseline parameters: log scale (as stored in optimization)
- Covariate coefficients: unchanged (already unconstrained)

# Example
```julia
# flat_params = [log(0.5), 0.3]  # log-rate and covariate coefficient
# Returns: (h12 = (baseline = (h12_Intercept = log(0.5),), covariates = (h12_x = 0.3,)),)
#          ↑ still on log scale
```

See also: [`unflatten_natural`](@ref)
"""
function unflatten_estimation(flat_params::AbstractVector{T}, model::MultistateProcess) where {T<:Real}
    if T === Float64
        return unflatten(model.parameters.reconstructor, flat_params)
    else
        return unflattenAD(model.parameters.reconstructor, flat_params)
    end
end

# Backward compatibility alias (deprecated)
# Use unflatten_estimation instead
const unflatten_to_estimation_scale = unflatten_estimation

"""
    rebuild_parameters(new_param_vectors::Vector{Vector{Float64}}, model::MultistateProcess)

Rebuild the parameters structure from new parameter vectors.
Parameters are stored on log scale for baseline, as-is for covariates.

# Arguments
- `new_param_vectors`: Vector of parameter vectors (one per hazard), on log scale for baseline
- `model`: The model containing hazard info for npar_baseline

# Returns
- NamedTuple with flat, nested, natural, and unflatten fields
"""
function rebuild_parameters(new_param_vectors::Vector{Vector{Float64}}, model::MultistateProcess)
    params_nested_pairs = [
        hazname => build_hazard_params(new_param_vectors[idx], model.hazards[idx].parnames, model.hazards[idx].npar_baseline, model.hazards[idx].npar_total)
        for (hazname, idx) in sort(collect(model.hazkeys), by = x -> x[2])
    ]
    
    params_nested = NamedTuple(params_nested_pairs)
    
    # Create ReConstructor for AD-compatible flatten/unflatten
    reconstructor = ReConstructor(params_nested, unflattentype=UnflattenFlexible())
    params_flat = flatten(reconstructor, params_nested)
    
    # Get natural scale parameters as flattened vectors per hazard (family-aware transformation)
    params_natural_pairs = [
        hazname => extract_natural_vector(params_nested[hazname], model.hazards[idx].family)
        for (hazname, idx) in sort(collect(model.hazkeys), by = x -> x[2])
    ]
    params_natural = NamedTuple(params_natural_pairs)
    
    return (
        flat = params_flat,
        nested = params_nested,
        natural = params_natural,
        reconstructor = reconstructor  # NEW: Store ReConstructor instead of unflatten_fn
    )
end

"""
    set_parameters!(model::MultistateProcess, newvalues::Vector{Vector{Float64}})

Set model parameters given a nested vector of values (one vector per hazard).
Values should be on log scale for baseline parameters, as-is for covariate coefficients.

# Arguments
- `model::MultistateProcess`: The model to update
- `newvalues`: Nested vector with parameters for each hazard

# Note
Updates model.parameters with the new values and remakes spline hazards as needed.
"""
function set_parameters!(model::MultistateProcess, newvalues::Vector{Vector{Float64}})
    
    # Get current natural-scale parameters
    current_natural = model.parameters.natural
    n_hazards = length(current_natural)
    
    # check that we have the right number of parameters
    if length(newvalues) != n_hazards
        error("New values and model parameters are not of the same length. Expected $n_hazards, got $(length(newvalues)).")
    end

    for i in 1:n_hazards
        if length(current_natural[i]) != length(newvalues[i])
            error("New values for hazard $i and model parameters for that hazard are not of the same length.")
        end

        # remake if a spline hazard
        if isa(model.hazards[i], _SplineHazard) 
            remake_splines!(model.hazards[i], newvalues[i])
            set_riskperiod!(model.hazards[i])
        end
    end
    
    # Rebuild parameters with proper constraints
    model.parameters = rebuild_parameters(newvalues, model)
    
    return nothing
end

"""
    set_parameters!(model::MultistateProcess, newvalues::Tuple)

Set model parameters given a tuple of vectors parameterizing transition intensities. 
Values should be on log scale for baseline parameters, as-is for covariate coefficients.

Assigns new values by hazard index in order.
"""
function set_parameters!(model::MultistateProcess, newvalues::Tuple)
    # Get current natural-scale parameters
    current_natural = model.parameters.natural
    n_hazards = length(current_natural)
    
    # check that there is a vector of parameters for each cause-specific hazard
    if length(newvalues) != n_hazards
        error("Number of supplied parameter vectors not equal to number of transition intensities. Expected $n_hazards, got $(length(newvalues)).")
    end

    for i in eachindex(newvalues)
        # check that we have the right number of parameters
        if length(current_natural[i]) != length(newvalues[i])
            error("New values and parameters for cause-specific hazard $i are not of the same length.")
        end
        
        # remake if a spline hazard
        if isa(model.hazards[i], _SplineHazard)
            remake_splines!(model.hazards[i], newvalues[i])
            set_riskperiod!(model.hazards[i])
        end
    end
    
    # Convert tuple to vector and rebuild
    new_param_vectors = [Vector{Float64}(collect(newvalues[i])) for i in 1:n_hazards]
    model.parameters = rebuild_parameters(new_param_vectors, model)
    
    return nothing
end

"""
    set_parameters!(model::MultistateProcess, newvalues::NamedTuple)

Set model parameters given a NamedTuple with hazard names as keys.
Values should be on log scale for baseline parameters, as-is for covariate coefficients.

Assignment is made by matching tuple keys in `newvalues` to the key in `model.hazkeys`.
"""
function set_parameters!(model::MultistateProcess, newvalues::NamedTuple)
    # Get current natural-scale parameters
    current_natural = model.parameters.natural
    value_keys = keys(newvalues)

    for k in eachindex(value_keys)
        vind = value_keys[k]
        mind = model.hazkeys[vind]

        # check length of supplied parameters
        if length(newvalues[vind]) != length(current_natural[mind])
            error("The new parameter values for $vind are not the expected length.")
        end

        # remake if a spline hazard
        if isa(model.hazards[mind], _SplineHazard)
            remake_splines!(model.hazards[mind], newvalues[vind])
            set_riskperiod!(model.hazards[mind])
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
Values should be on log scale for baseline parameters, as-is for covariate coefficients.

# Arguments
- `model::MultistateProcess`: The model to update
- `h::Int64`: Index of the hazard to update
- `newvalues::Vector{Float64}`: New parameter values (log scale for baseline, as-is for covariates)

# Example
```julia
# Update hazard 1 with new baseline parameter
set_parameters!(model, 1, [log(0.5)])

# Update hazard with baseline and covariate effects
set_parameters!(model, 2, [log(2.0), log(1.5), 0.3, -0.2])
```
"""
function set_parameters!(model::MultistateProcess, h::Int64, newvalues::Vector{Float64})
    current_natural = model.parameters.natural
    n_hazards = length(current_natural)
    
    # Check hazard index
    if h < 1 || h > n_hazards
        error("Hazard index $h is out of bounds. Model has $n_hazards hazards.")
    end
    
    # Check parameter length
    if length(newvalues) != length(current_natural[h])
        error("New values length ($(length(newvalues))) does not match expected length ($(length(current_natural[h]))) for hazard $h.")
    end
    
    # Remake splines if needed
    if isa(model.hazards[h], _SplineHazard)
        remake_splines!(model.hazards[h], newvalues)
        set_riskperiod!(model.hazards[h])
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

    # return indices
    return subjinds, nsubj
end

# ============================================================================
# Parameter Construction Functions
# ============================================================================

"""
    build_hazard_params(log_scale_params, parnames, npar_baseline, npar_total)

Create a NamedTuple with baseline and covariate parameters using named fields.

Parameters are stored on log scale for baseline, as-is for covariates.
No transformations are applied - hazard functions expect log scale.

This function is type-generic to support ForwardDiff.Dual during automatic differentiation.

# Arguments
- `log_scale_params`: Full parameter vector - log scale for baseline, as-is for covariates
- `parnames`: Vector of parameter names (e.g., [:h12_shape, :h12_scale, :h12_age])
- `npar_baseline`: Number of baseline parameters
- `npar_total`: Total number of parameters

# Returns
- `NamedTuple`: `(baseline = (h12_shape=..., h12_scale=...), covariates = (h12_age=...,))` 
  or just `(baseline = (h12_shape=..., h12_scale=...),)` if no covariates

# Examples
```julia
# Weibull with covariates
params = build_hazard_params([log(1.5), log(0.2), 0.3, 0.1], 
                             [:h12_shape, :h12_scale, :h12_age, :h12_sex], 2, 4)
# Returns: (baseline = (h12_shape = 0.405, h12_scale = -1.609), 
#           covariates = (h12_age = 0.3, h12_sex = 0.1))

# Exponential without covariates  
params = build_hazard_params([log(0.5)], [:h12_intercept], 1, 1)
# Returns: (baseline = (h12_intercept = -0.693),)
```

# Note
All parameters are stored on log scale for baseline (as expected by hazard functions).
Covariate coefficients are stored as-is (unconstrained).
"""
function build_hazard_params(log_scale_params::AbstractVector{<:Real}, parnames::Vector{Symbol}, npar_baseline::Int, npar_total::Int)
    @assert length(log_scale_params) == npar_total "Parameter vector length mismatch"
    @assert length(parnames) == npar_total "Parameter names length must match parameter vector length"
    @assert npar_baseline <= npar_total "Baseline parameters cannot exceed total parameters"
    
    # Extract baseline parameter names and values
    baseline_names = parnames[1:npar_baseline]
    baseline_values = log_scale_params[1:npar_baseline]
    
    # Create NamedTuple for baseline with named fields
    baseline = NamedTuple{Tuple(baseline_names)}(baseline_values)
    
    # Handle covariates if present
    if npar_total > npar_baseline
        covar_names = parnames[(npar_baseline+1):npar_total]
        covar_values = log_scale_params[(npar_baseline+1):npar_total]
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

Extract the natural-scale parameter vector from a hazard's NamedTuple params structure.
Applies appropriate transformation based on hazard family.

For Gompertz: shape is unconstrained (identity), rate is positive (exp)
For other families: all baseline parameters are positive (exp)

# Arguments
- `hazard_params`: NamedTuple with `baseline` (named NamedTuple) and optionally `covariates` (named NamedTuple)
- `family`: Hazard family (`:exp`, `:wei`, `:gom`, `:sp`)

# Returns
- Vector with natural-scale baseline values followed by covariate coefficients (as-is)
"""
function extract_natural_vector(hazard_params::NamedTuple, family::Symbol)
    baseline_vals = collect(values(hazard_params.baseline))
    
    if family == :gom
        # Gompertz: shape is unconstrained (first), rate is positive (second)
        baseline_natural = [i == 1 ? baseline_vals[i] : exp(baseline_vals[i]) for i in eachindex(baseline_vals)]
    else
        # All other families: all baseline params are positive
        baseline_natural = exp.(baseline_vals)
    end
    
    if haskey(hazard_params, :covariates)
        covar_vals = collect(values(hazard_params.covariates))
        return vcat(baseline_natural, covar_vals)
    else
        return baseline_natural
    end
end

# Backward-compatible version without family (assumes all positive)
function extract_natural_vector(hazard_params::NamedTuple)
    baseline_vals = collect(values(hazard_params.baseline))
    baseline_natural = exp.(baseline_vals)
    
    if haskey(hazard_params, :covariates)
        covar_vals = collect(values(hazard_params.covariates))
        return vcat(baseline_natural, covar_vals)
    else
        return baseline_natural
    end
end

"""
    get_estimation_scale_params(parameters)

Extract estimation-scale parameters from a parameters structure.
Returns parameters on LOG scale for baseline, as-is for covariates.

This function extracts the `nested` field from model.parameters, which stores
parameters on estimation scale (the scale used by optimizers).

# Arguments
- `parameters`: A NamedTuple with `nested` field, or already-extracted nested parameters

# Returns
- `NamedTuple`: Estimation-scale parameters indexed by hazard name

# Note
For hazard evaluation, use `get_hazard_params()` instead, which returns natural scale.
This function is primarily for:
- Storing/outputting parameter values
- Initialization routines
- Converting back from natural scale

# Example
```julia
# Get estimation-scale params (for storage/output)
est_pars = get_estimation_scale_params(model.parameters)
# est_pars[:h12].baseline contains log-scale values
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

Extract natural-scale parameters ready for hazard function evaluation.
Family-aware version that correctly transforms parameters based on hazard type.

For Gompertz: shape is kept as-is (unconstrained), rate is exp-transformed
For other families: all baseline parameters are exp-transformed

This is the primary function for getting parameters for hazard evaluation.
The returned NamedTuple can be passed directly to hazard functions.

# Arguments
- `parameters`: A NamedTuple with `nested` field, or already-extracted nested parameters
- `hazards`: Vector of hazard objects (used to determine transformation per family)

# Returns
- `NamedTuple`: Natural-scale parameters indexed by hazard name

# Example
```julia
# Get natural-scale params for hazard evaluation
haz_pars = get_hazard_params(model.parameters, model.hazards)
# haz_pars[:h12].baseline contains natural-scale values
# Pass to hazard: eval_hazard(hazard, t, haz_pars[:h12], covars)
```
"""
function get_hazard_params(parameters, hazards)
    # Get estimation-scale params first
    est_params = get_estimation_scale_params(parameters)
    # Transform to natural scale for hazard evaluation (family-aware)
    return to_natural_scale(est_params, hazards, Float64)
end

# Backward-compatible version without hazards (deprecated - assumes all positive)
# This version applies exp() to ALL baseline parameters, which is incorrect for Gompertz
function get_hazard_params(parameters)
    # Get estimation-scale params first
    est_params = get_estimation_scale_params(parameters)
    # Transform to natural scale for hazard evaluation (legacy: all exp)
    return to_natural_scale(est_params, Float64)
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
    # Get estimation-scale params first
    est_params = get_estimation_scale_params(parameters)
    # Transform to natural scale (family-aware)
    params_named = to_natural_scale(est_params, hazards, T)
    return values(params_named)
end
