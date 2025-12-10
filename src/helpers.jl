# ============================================================================
# ReConstructor: AD-compatible parameter flattening/unflattening
# Based on ModelWrappers.jl pattern - manual implementation to avoid dependencies
# ============================================================================

"""
Abstract type for flatten mode selection.
"""
abstract type FlattenTypes end

"""
    FlattenContinuous <: FlattenTypes

Flatten only continuous parameters (default behavior).
"""
struct FlattenContinuous <: FlattenTypes end

"""
    FlattenAll <: FlattenTypes

Flatten all parameters including integers.
"""
struct FlattenAll <: FlattenTypes end

"""
Abstract type for unflatten mode selection.
"""
abstract type UnflattenTypes end

"""
    UnflattenStrict <: UnflattenTypes

Type-stable unflatten that converts to original types.
Use for standard evaluation (non-AD contexts).
"""
struct UnflattenStrict <: UnflattenTypes end

"""
    UnflattenFlexible <: UnflattenTypes

Type-polymorphic unflatten that preserves input types.
Use for automatic differentiation to preserve Dual types.
"""
struct UnflattenFlexible <: UnflattenTypes end

"""
    FlattenDefault{F<:FlattenTypes, U<:UnflattenTypes}

Default settings for flatten/unflatten operations.

# Fields
- `flattentype::F`: Controls which parameters to flatten
- `unflattentype::U`: Controls type handling during unflatten
"""
struct FlattenDefault{F<:FlattenTypes, U<:UnflattenTypes}
    flattentype::F
    unflattentype::U
end

# Convenience constructors
FlattenDefault() = FlattenDefault(FlattenContinuous(), UnflattenStrict())
FlattenDefault(unflattentype::UnflattenTypes) = FlattenDefault(FlattenContinuous(), unflattentype)

"""
    ReConstructor{F,S,T,U,V}

Pre-computed flatten/unflatten closures with buffers for efficient parameter operations.

Stores four function variants:
- `flatten_strict`: Type-stable flatten (returns Vector{Float64})
- `flatten_flexible`: AD-compatible flatten (returns Vector{T})
- `unflatten_strict`: Type-stable unflatten (converts to original types)
- `unflatten_flexible`: Type-polymorphic unflatten (preserves Dual types for AD)

# Usage
```julia
# Create reconstructor
rc = ReConstructor(params)

# Standard usage (fast)
flat = flatten(rc, params)
reconstructed = unflatten(rc, flat)

# AD usage (preserves Dual types)
flat_dual = flattenAD(rc, params_dual)
reconstructed_dual = unflattenAD(rc, flat_dual)
```
"""
struct ReConstructor{F,S,T,U,V}
    default::FlattenDefault
    flatten_strict::F
    flatten_flexible::S
    unflatten_strict::T
    unflatten_flexible::U
    _buffer::V  # Pre-allocated buffer for intermediate operations
end

# ============================================================================
# Construction functions for different types
# ============================================================================

"""
    construct_flatten(output::Type{T}, flattentype::FlattenTypes, 
                     unflattentype::UnflattenTypes, x::Real)

Build flatten/unflatten closures for Real numbers.

Returns tuple of (flatten_function, unflatten_function).
"""
function construct_flatten(
    output::Type{T},
    flattentype::FlattenTypes,
    unflattentype::UnflattenTypes,
    x::Real
) where {T<:Real}
    
    # Flatten: Real → Vector{T}
    function flatten_to_Real(val::S) where {S<:Real}
        return T[val]
    end
    
    # Unflatten variants - use different names to avoid method overwriting
    if unflattentype isa UnflattenStrict
        # Strict: convert to original type (type-stable, breaks AD)
        unflatten_to_Real_strict(v::AbstractVector{S}) where {S<:Real} = convert(typeof(x), only(v))
        return flatten_to_Real, unflatten_to_Real_strict
    else  # UnflattenFlexible
        # Flexible: preserve input type (allows Dual types for AD)
        unflatten_to_Real_flexible(v::AbstractVector{S}) where {S<:Real} = only(v)
        return flatten_to_Real, unflatten_to_Real_flexible
    end
end

"""
    construct_flatten(output::Type{T}, flattentype::FlattenTypes,
                     unflattentype::UnflattenTypes, x::AbstractVector{<:Real})

Build flatten/unflatten closures for Vector of Real numbers.

Handles both FlattenContinuous and FlattenAll modes.
"""
function construct_flatten(
    output::Type{T},
    flattentype::FlattenTypes,
    unflattentype::UnflattenTypes,
    x::AbstractVector{<:Real}
) where {T<:Real}
    
    n = length(x)
    
    # Check if all elements are continuous (Float) or if we have integers
    is_continuous = all(xi -> xi isa AbstractFloat, x)
    
    # For FlattenContinuous mode with integer vectors, skip flattening
    if flattentype isa FlattenContinuous && !is_continuous
        # Return identity functions - don't flatten integer vectors
        flatten_Vector_skip(val) = T[]
        unflatten_Vector_skip(v) = x  # Return original vector
        return flatten_Vector_skip, unflatten_Vector_skip
    end
    
    # Flatten: Vector → concatenated Vector{T}
    function flatten_Vector(val::AbstractVector{S}) where {S<:Real}
        return convert(Vector{T}, val)
    end
    
    # Unflatten variants
    if unflattentype isa UnflattenStrict
        # Strict: convert to original element type (type-stable)
        function unflatten_Vector_strict(v::AbstractVector{S}) where {S<:Real}
            return convert(typeof(x), v[1:n])
        end
        return flatten_Vector, unflatten_Vector_strict
    else  # UnflattenFlexible
        # Flexible: preserve input element types (allows Dual for AD)
        function unflatten_Vector_flexible(v::AbstractVector{S}) where {S<:Real}
            # Use view to avoid allocation, return as vector of input type
            return collect(v[1:n])
        end
        return flatten_Vector, unflatten_Vector_flexible
    end
end

"""
    construct_flatten(output::Type{T}, flattentype::FlattenTypes,
                     unflattentype::UnflattenTypes, x::Tuple)

Build flatten/unflatten closures for Tuples (recursive construction).

This is the core of the recursive algorithm:
1. Build constructors for each element recursively
2. Flatten once to determine sizes
3. Compute cumulative sizes for indexing
4. Return composed closures that handle the full structure
"""
function construct_flatten(
    output::Type{T},
    flattentype::FlattenTypes,
    unflattentype::UnflattenTypes,
    x::Tuple
) where {T<:Real}
    
    # Step 1: Recursively build constructors for each element
    x_constructors = map(xᵢ -> construct_flatten(T, flattentype, unflattentype, xᵢ), x)
    _flatten = first.(x_constructors)
    _unflatten = last.(x_constructors)
    
    # Step 2: Flatten once to determine sizes
    x_vecs = map((flat, xᵢ) -> flat(xᵢ), _flatten, x)
    lengths = map(length, x_vecs)
    cumulative_sizes = cumsum(lengths)
    
    # Step 3: Build composed flatten function
    function flatten_Tuple(val::Tuple)
        mapped = map((flat, xᵢ) -> flat(xᵢ), _flatten, val)
        return isempty(mapped) ? T[] : reduce(vcat, mapped)
    end
    
    # Step 4: Build composed unflatten with proper indexing
    function unflatten_Tuple(v::AbstractVector{S}) where {S<:Real}
        return map(_unflatten, lengths, cumulative_sizes) do unflat, len, cumsize
            start_idx = cumsize - len + 1
            if len == 0
                # Empty vector case (e.g., integer vectors in FlattenContinuous mode)
                return unflat(S[])
            else
                return unflat(view(v, start_idx:cumsize))
            end
        end
    end
    
    return flatten_Tuple, unflatten_Tuple
end

"""
    construct_flatten(output::Type{T}, flattentype::FlattenTypes,
                     unflattentype::UnflattenTypes, x::NamedTuple)

Build flatten/unflatten closures for NamedTuples (recursive construction).

Key difference for AD compatibility:
- Strict mode: `typeof(x)(tuple)` - requires concrete types, breaks with Dual
- Flexible mode: `NamedTuple{names}(tuple)` - accepts any types, preserves Dual
"""
function construct_flatten(
    output::Type{T},
    flattentype::FlattenTypes,
    unflattentype::UnflattenTypes,
    x::NamedTuple
) where {T<:Real}
    
    names = keys(x)
    values_tuple = values(x)
    
    # Build constructor for the underlying tuple (recursive)
    flatten_tuple, unflatten_tuple = construct_flatten(T, flattentype, unflattentype, values_tuple)
    
    # Flatten: just use tuple flatten
    flatten_NamedTuple(val::NamedTuple) = flatten_tuple(values(val))
    
    # Unflatten: reconstruct NamedTuple with appropriate type handling
    if unflattentype isa UnflattenStrict
        # Strict: Use typed constructor (type-stable, requires concrete types)
        function unflatten_NamedTuple_strict(v::AbstractVector{S}) where {S<:Real}
            v_tuple = unflatten_tuple(v)
            return typeof(x)(v_tuple)  # Requires concrete types - breaks with Dual
        end
        return flatten_NamedTuple, unflatten_NamedTuple_strict
    else  # UnflattenFlexible
        # Flexible: Use generic constructor (preserves any types including Dual)
        function unflatten_NamedTuple_flexible(v::AbstractVector{S}) where {S<:Real}
            v_tuple = unflatten_tuple(v)
            return NamedTuple{names}(v_tuple)  # Generic - works with Dual!
        end
        return flatten_NamedTuple, unflatten_NamedTuple_flexible
    end
end

# ============================================================================
# ReConstructor builder and user API
# ============================================================================

"""
    ReConstructor(x; flattentype=FlattenContinuous(), unflattentype=UnflattenStrict())

Build a ReConstructor for the given parameter structure.

# Arguments
- `x`: Parameter structure (NamedTuple, Tuple, Vector, or Real)
- `flattentype`: FlattenContinuous() or FlattenAll()
- `unflattentype`: UnflattenStrict() or UnflattenFlexible()

# Returns
ReConstructor with pre-computed flatten/unflatten closures

# Example
```julia
params = (baseline = (shape = 1.5, scale = 0.2), covariates = (age = 0.3,))
rc = ReConstructor(params, unflattentype=UnflattenFlexible())

# Standard usage
flat = flatten(rc, params)

# AD usage (preserves Dual types)
using ForwardDiff
f(p) = sum(unflattenAD(rc, p).baseline)
grad = ForwardDiff.gradient(f, flat)
```
"""
function ReConstructor(
    x;
    flattentype::FlattenTypes = FlattenContinuous(),
    unflattentype::UnflattenTypes = UnflattenStrict()
)
    default = FlattenDefault(flattentype, unflattentype)
    
    # Build strict constructors (for standard usage)
    flatten_strict_fn, unflatten_strict_fn = construct_flatten(Float64, flattentype, UnflattenStrict(), x)
    
    # Build flexible constructors (for AD usage)
    flatten_flexible_fn, unflatten_flexible_fn = construct_flatten(Float64, flattentype, UnflattenFlexible(), x)
    
    # Pre-allocate buffer for intermediate operations
    flat_example = flatten_strict_fn(x)
    buffer = similar(flat_example)
    
    return ReConstructor(
        default,
        flatten_strict_fn,
        flatten_flexible_fn,
        unflatten_strict_fn,
        unflatten_flexible_fn,
        buffer
    )
end

"""
    flatten(rc::ReConstructor, x)

Flatten parameter structure to vector (type-stable, returns Vector{Float64}).

Use for standard parameter operations.
"""
flatten(rc::ReConstructor, x) = rc.flatten_strict(x)

"""
    unflatten(rc::ReConstructor, v::AbstractVector)

Unflatten vector to parameter structure (type-stable).

Use for standard parameter operations.
"""
unflatten(rc::ReConstructor, v::AbstractVector) = rc.unflatten_strict(v)

"""
    flattenAD(rc::ReConstructor, x)

Flatten parameter structure to vector (type-polymorphic).

Use when working with AD types (preserves Dual numbers).
"""
flattenAD(rc::ReConstructor, x) = rc.flatten_flexible(x)

"""
    unflattenAD(rc::ReConstructor, v::AbstractVector)

Unflatten vector to parameter structure (type-polymorphic).

Use for automatic differentiation - preserves Dual types.
"""
unflattenAD(rc::ReConstructor, v::AbstractVector) = rc.unflatten_flexible(v)

# ============================================================================
# Parameter unflattening with AD compatibility
# ============================================================================

"""
    unflatten_parameters(flat_params::AbstractVector, model::MultistateProcess)

Unflatten parameter vector and transform to natural scale for hazard evaluation.

This function:
1. Unflattens the flat parameter vector to nested NamedTuple structure
2. Transforms baseline parameters from estimation scale (log) to natural scale (exp)

The returned parameters are on NATURAL SCALE and ready for direct use in hazard functions.
Hazard functions no longer need to apply exp() internally.

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
"""
function unflatten_parameters(flat_params::AbstractVector{T}, model::MultistateProcess) where {T<:Real}
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

# Backward compatibility alias
const safe_unflatten = unflatten_parameters

"""
    unflatten_to_estimation_scale(flat_params, model)

Unflatten parameters WITHOUT applying natural-scale transformation.
Returns parameters on ESTIMATION SCALE (log for baseline, as-is for covariates).

This is used by spline remake code which needs log-scale coefficients.
Unlike `unflatten_parameters`, this does NOT apply exp() to baseline parameters.

# Arguments
- `flat_params`: Flat parameter vector on estimation scale
- `model`: MultistateProcess model containing ReConstructor

# Returns
NamedTuple of nested parameters on ESTIMATION SCALE (log for baseline)
"""
function unflatten_to_estimation_scale(flat_params::AbstractVector{T}, model::MultistateProcess) where {T<:Real}
    if T === Float64
        return unflatten(model.parameters.reconstructor, flat_params)
    else
        return unflattenAD(model.parameters.reconstructor, flat_params)
    end
end

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

#=============================================================================
PHASE 2: Parameter Update with ParameterHandling.jl
=============================================================================# 

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

# ============================================================================
# Parameter Scale Transformations
# ============================================================================
#
# PARAMETER SCALE CONVENTION:
#
# Estimation Scale (what optimizers see):
#   - Baseline parameters (intercept, shape, scale): log scale (unconstrained)
#   - Covariate coefficients: as-is (already unconstrained)
#
# Natural/Model Scale (what hazard functions use):
#   - Baseline parameters: natural scale (exp applied, positive)
#   - Covariate coefficients: as-is
#
# The transformation happens at the boundary:
#   - to_natural_scale: Called in unflatten_parameters (estimation → natural)
#   - to_estimation_scale: Called when storing/outputting parameters
#
# This design:
#   1. Avoids repeated exp() calls in hazard function hot loops
#   2. Makes hazard functions simpler (no internal transformations)
#   3. Centralizes transformation logic for extensibility (SODEN, etc.)
#
# ============================================================================

"""
    transform_baseline_to_natural(baseline::NamedTuple, family::Symbol, ::Type{T}) where T

Transform baseline parameters from estimation scale to natural scale.
Applies exp() to parameters that are constrained positive, identity to unconstrained.

For Gompertz: shape is unconstrained (identity), scale/rate is positive (exp)
For Weibull/Exponential: all baseline parameters are positive (exp)

# Arguments
- `baseline`: NamedTuple of baseline parameters on estimation scale
- `family`: Hazard family (`:exp`, `:wei`, `:gom`, `:sp`)
- `T`: Element type (Float64 or ForwardDiff.Dual)

# Returns
NamedTuple with same keys but values transformed appropriately
"""
@inline function transform_baseline_to_natural(baseline::NamedTuple, family::Symbol, ::Type{T}) where T
    if family == :sp
        # Splines: parameters stay on log scale - hazard function applies exp() internally
        # via _spline_ests2coefs, so no transformation here
        return baseline
    elseif family == :gom
        # Gompertz: shape is unconstrained (first param), rate is positive (second param)
        # Parameter order: [shape, scale/rate]
        ks = keys(baseline)
        vs = values(baseline)
        # First param (shape): identity transform
        # Second param (scale/rate): exp transform
        transformed_values = ntuple(i -> i == 1 ? vs[i] : exp(vs[i]), length(vs))
        return NamedTuple{ks}(transformed_values)
    else
        # All other families: all baseline parameters are positive (exp)
        transformed_values = map(v -> exp(v), values(baseline))
        return NamedTuple{keys(baseline)}(transformed_values)
    end
end

# Backward-compatible version without family (assumes all positive)
@inline function transform_baseline_to_natural(baseline::NamedTuple, ::Type{T}) where T
    transformed_values = map(v -> exp(v), values(baseline))
    return NamedTuple{keys(baseline)}(transformed_values)
end

"""
    transform_baseline_to_estimation(baseline::NamedTuple, family::Symbol)

Transform baseline parameters from natural scale to estimation scale.
Applies log() to positive-constrained parameters, identity to unconstrained.

For Gompertz: shape is unconstrained (identity), scale/rate is positive (log)
For Weibull/Exponential: all baseline parameters are positive (log)

# Arguments
- `baseline`: NamedTuple of baseline parameters on natural scale
- `family`: Hazard family (`:exp`, `:wei`, `:gom`, `:sp`)

# Returns
NamedTuple with same keys but values transformed appropriately
"""
@inline function transform_baseline_to_estimation(baseline::NamedTuple, family::Symbol)
    if family == :sp
        # Splines: parameters are already on log/estimation scale - hazard function 
        # applies exp() internally, so no transformation here
        return baseline
    elseif family == :gom
        # Gompertz: shape is unconstrained (first param), rate is positive (second param)
        ks = keys(baseline)
        vs = values(baseline)
        # First param (shape): identity transform
        # Second param (scale/rate): log transform
        transformed_values = ntuple(i -> i == 1 ? vs[i] : log(vs[i]), length(vs))
        return NamedTuple{ks}(transformed_values)
    else
        # All other families: all baseline parameters are positive (log)
        transformed_values = map(v -> log(v), values(baseline))
        return NamedTuple{keys(baseline)}(transformed_values)
    end
end

# Backward-compatible version without family (assumes all positive)
@inline function transform_baseline_to_estimation(baseline::NamedTuple)
    transformed_values = map(v -> log(v), values(baseline))
    return NamedTuple{keys(baseline)}(transformed_values)
end

"""
    to_natural_scale(params_nested::NamedTuple, hazards, ::Type{T}) where T

Transform nested parameters from estimation scale to natural scale.
Applies exp() to positive-constrained baseline parameters, identity to unconstrained.

# Arguments
- `params_nested`: NamedTuple of per-hazard parameter NamedTuples (estimation scale)
- `hazards`: Vector of hazard objects (used to determine transformation per family)
- `T`: Element type for AD compatibility

# Returns
NamedTuple of per-hazard parameter NamedTuples on natural scale

# Example
```julia
# Estimation scale: (h12 = (baseline = (h12_Intercept = -0.5,), covariates = (h12_x = 0.3,)))
# Natural scale:    (h12 = (baseline = (h12_Intercept = 0.606,), covariates = (h12_x = 0.3,)))
```
"""
function to_natural_scale(params_nested::NamedTuple, hazards, ::Type{T}) where T
    hazard_keys = keys(params_nested)
    transformed_hazards = map(enumerate(hazard_keys)) do (idx, hazname)
        hazard_params = params_nested[hazname]
        hazard = hazards[idx]
        family = hazard.family
        
        # Transform baseline to natural scale (family-aware)
        baseline_natural = transform_baseline_to_natural(hazard_params.baseline, family, T)
        
        # Keep covariates unchanged
        if haskey(hazard_params, :covariates)
            hazname => (baseline = baseline_natural, covariates = hazard_params.covariates)
        else
            hazname => (baseline = baseline_natural,)
        end
    end
    
    return NamedTuple(transformed_hazards)
end

# Backward-compatible version without hazards (assumes all positive)
function to_natural_scale(params_nested::NamedTuple, ::Type{T}) where T
    transformed_hazards = map(keys(params_nested)) do hazname
        hazard_params = params_nested[hazname]
        
        # Transform baseline to natural scale (legacy: all exp)
        baseline_natural = transform_baseline_to_natural(hazard_params.baseline, T)
        
        # Keep covariates unchanged
        if haskey(hazard_params, :covariates)
            hazname => (baseline = baseline_natural, covariates = hazard_params.covariates)
        else
            hazname => (baseline = baseline_natural,)
        end
    end
    
    return NamedTuple(transformed_hazards)
end

"""
    to_estimation_scale(params_nested::NamedTuple)

Transform nested parameters from natural scale to estimation scale.
Applies log() to all baseline parameters, leaves covariate coefficients unchanged.

# Arguments
- `params_nested`: NamedTuple of per-hazard parameter NamedTuples (natural scale)

# Returns
NamedTuple of per-hazard parameter NamedTuples on estimation scale

# Example
```julia
# Natural scale:    (h12 = (baseline = (h12_Intercept = 0.606,), covariates = (h12_x = 0.3,)))
# Estimation scale: (h12 = (baseline = (h12_Intercept = -0.5,), covariates = (h12_x = 0.3,)))
```
"""
function to_estimation_scale(params_nested::NamedTuple, hazards)
    hazard_keys = keys(params_nested)
    transformed_hazards = map(enumerate(hazard_keys)) do (idx, hazname)
        hazard_params = params_nested[hazname]
        hazard = hazards[idx]
        family = hazard.family
        
        # Transform baseline to estimation scale (family-aware)
        baseline_estimation = transform_baseline_to_estimation(hazard_params.baseline, family)
        
        # Keep covariates unchanged
        if haskey(hazard_params, :covariates)
            hazname => (baseline = baseline_estimation, covariates = hazard_params.covariates)
        else
            hazname => (baseline = baseline_estimation,)
        end
    end
    
    return NamedTuple(transformed_hazards)
end

# Backward-compatible version without hazards (assumes all positive/log)
function to_estimation_scale(params_nested::NamedTuple)
    transformed_hazards = map(keys(params_nested)) do hazname
        hazard_params = params_nested[hazname]
        
        # Transform baseline to estimation scale (legacy: all log)
        baseline_estimation = transform_baseline_to_estimation(hazard_params.baseline)
        
        # Keep covariates unchanged
        if haskey(hazard_params, :covariates)
            hazname => (baseline = baseline_estimation, covariates = hazard_params.covariates)
        else
            hazname => (baseline = baseline_estimation,)
        end
    end
    
    return NamedTuple(transformed_hazards)
end

"""
    set_parameters_flat!(model::MultistateProcess, flat_params::AbstractVector)

Set model parameters from a flat parameter vector (as used by optimizers).
The flat vector is unflattened using ParameterHandling's unflatten function.

# Arguments
- `model::MultistateProcess`: The model to update  
- `flat_params::AbstractVector`: Flat parameter vector (same format as parameters.flat)

# Note
This is the primary method for updating parameters during optimization.
Spline hazards are remade with the new parameters.
"""
function set_parameters_flat!(model::MultistateProcess, flat_params::AbstractVector)
    # Unflatten to get nested structure
    params_nested = unflatten(model.parameters.reconstructor, flat_params)
    
    # Get new flat version (reconstructor stays the same)
    new_flat = flatten(model.parameters.reconstructor, params_nested)
    
    # Update spline hazards if needed - extract log-scale params properly
    for i in eachindex(model.hazards)
        if isa(model.hazards[i], _SplineHazard)
            hazard_params = values(params_nested)[i]
            log_params = extract_params_vector(hazard_params)
            remake_splines!(model.hazards[i], log_params)
            set_riskperiod!(model.hazards[i])
        end
    end
    
    # Compute natural scale parameters (family-aware transformation)
    params_natural_pairs = [
        hazname => extract_natural_vector(params_nested[hazname], model.hazards[idx].family)
        for (hazname, idx) in sort(collect(model.hazkeys), by = x -> x[2])
    ]
    params_natural = NamedTuple(params_natural_pairs)
    
    # Update parameters (reconstructor stays the same)
    model.parameters = (
        flat = Vector{Float64}(flat_params),
        nested = params_nested,
        natural = params_natural,
        reconstructor = model.parameters.reconstructor
    )
    
    return nothing
end

#=============================================================================
PHASE 3: Parameter Getter Functions
=============================================================================# 

# Internal helpers - prefer get_parameters(model; scale=:flat/:nested/:natural) for new code
function get_parameters_flat(model::MultistateProcess)
    return model.parameters.flat
end

function get_parameters_nested(model::MultistateProcess)
    return model.parameters.nested
end

function get_parameters_natural(model::MultistateProcess)
    return model.parameters.natural
end

"""
    get_unflatten_fn(model::MultistateProcess)

Get the unflatten function to convert flat parameter vector back to NamedTuple.

This function is useful in optimization when you need to convert the flat
vector back to the structured NamedTuple representation.

# Returns
- `Function`: unflatten function that takes a flat vector and returns NamedTuple

# Example
```julia
unflatten = get_unflatten_fn(model)
flat_params = get_parameters_flat(model)
params_nested = unflatten(flat_params)
```

# See also
- [`get_parameters_flat`](@ref) - Get flat parameter vector
- [`get_parameters_nested`](@ref) - Get nested parameters
"""
function get_unflatten_fn(model::MultistateProcess)
    # Return a function that unflattens Float64 vectors
    return p -> unflatten(model.parameters.reconstructor, p)
end

"""
    get_parameters(model::MultistateProcess, h::Int64; scale::Symbol=:natural)

Get parameters for a specific hazard by index.

# Arguments
- `model::MultistateProcess`: The model
- `h::Int64`: Hazard index (1-based)
- `scale::Symbol`: One of `:natural`, `:nested`, or `:log` (default: `:natural`)

# Returns
- `Vector{Float64}` or `NamedTuple`: Parameters for the specified hazard

# Scales
- `:natural` - Natural scale (baseline/shape/scale are positive values)
- `:nested` - Nested NamedTuple with (baseline = [...], covariates = [...])
- `:log` - Log scale (baseline on log scale, covariates on natural scale)

# Examples
```julia
# Get natural scale parameters for hazard 1
params_nat = get_parameters(model, 1)  # Returns [2.0, 1.5, ...]

# Get log scale parameters
params_log = get_parameters(model, 1, scale=:log)  # Returns [log(2.0), log(1.5), ...]

# Get nested parameters
params_nested = get_parameters(model, 1, scale=:nested)
```

# See also
- [`get_parameters_flat`](@ref) - Get all parameters as flat vector
- [`get_parameters_natural`](@ref) - Get all parameters on natural scale
- [`set_parameters!`](@ref) - Set parameters for a hazard
"""
function get_parameters(model::MultistateProcess, h::Int64; scale::Symbol=:natural)
    # Validate hazard index using parameters structure
    n_hazards = length(model.parameters.natural)
    if h < 1 || h > n_hazards
        throw(BoundsError(1:n_hazards, h))
    end
    
    if scale == :log
        # Compute log-scale parameters from parameters.flat
        # Split flat vector into per-hazard vectors based on natural structure
        natural_vals = values(model.parameters.natural)
        block_sizes = [length(v) for v in natural_vals]
        offset = sum(block_sizes[1:h-1])
        return model.parameters.flat[(offset+1):(offset+block_sizes[h])]
    elseif scale == :natural || scale == :nested || scale == :transformed
        # Find hazard name from index
        hazname = nothing
        for (name, idx) in model.hazkeys
            if idx == h
                hazname = name
                break
            end
        end
        if hazname === nothing
            error("Could not find hazard name for index $h")
        end
        
        # Return appropriate representation
        if scale == :natural
            return model.parameters.natural[hazname]
        else  # :nested or :transformed (backward compat)
            return model.parameters.nested[hazname]
        end
    else
        throw(ArgumentError("scale must be :natural, :nested, or :log (got :$scale)"))
    end
end

"""
    check_data!(data::DataFrame, tmat::Matrix, emat::Matrix{<:Real}; verbose = true)

Validate a user-supplied data frame to ensure that it conforms to MultistateModels.jl requirements.
"""
function check_data!(data::DataFrame, tmat::Matrix, emat::Matrix{<:Real}; verbose = true)

    # validate column names and order
    if any(names(data)[1:6] .!== ["id", "tstart", "tstop", "statefrom", "stateto", "obstype"])
        error("The first 6 columns of the data should be 'id', 'tstart', 'tstop', 'statefrom', 'stateto', 'obstype'.")
    end

    # coerce id to Int64, times to Float64, states to Int64, obstype to Int64
    data.id        = convert(Vector{Int64},   data.id)
    data.tstart    = convert(Vector{Float64}, data.tstart)
    data.tstop     = convert(Vector{Float64}, data.tstop)
    data.obstype   = convert(Vector{Int64},   data.obstype)
    data.statefrom = convert(Vector{Union{Missing,Int64}}, data.statefrom)
    data.stateto   = convert(Vector{Union{Missing, Int64}}, data.stateto)        

    # verify that subject id's are (1, 2, ...)
    unique_id = unique(data.id)
    nsubj = length(unique_id)
    if any(unique_id .!= 1:nsubj)
        error("The subject id's should be 1, 2, 3, ... .")
    end

    # warn about individuals starting in absorbing states
    # check if there are any absorbing states
    absorbing = map(x -> all(x .== 0), eachrow(tmat))

    # look to see if any of the absorbing states are in statefrom
    if any(absorbing)
        which_absorbing = findall(absorbing .== true)
        abs_warn = any(map(x -> any(data.statefrom .== x), which_absorbing))

        if verbose && abs_warn
            @warn "The data contains contains observations where a subject originates in an absorbing state."
        end
    end

    # error if any tstart < tstop
    if any(data.tstart >= data.tstop)
        error("The data should not contain time intervals where tstart is greater than or equal to tstop.")
    end

    # within each subject's data, error if tstart or tstop are out of order or there are discontinuities given multiple time intervals
    for i in unique_id
        inds = findall(data.id .== i)

        # check sorting
        if(!issorted(data.tstart[inds]) || !issorted(data.tstop[inds])) 
            error("tstart and tstop must be sorted for each subject.")
        end
        
        # check for discontinuities
        if(length(inds) > 1)
            if(any(data.tstart[inds[Not(begin)]] .!= 
                    data.tstop[inds[Not(end)]]))
                error("Time intervals for subject $i contain discontinuities.")
            end
        end
    end

    # error if data includes states not in the unique states
    emat_ids = Int64.(emat[:,1])
    statespace = sort(vcat(0, collect(1:size(tmat,1)), emat_ids))
    allstates = sort(vcat(unique(data.stateto), unique(data.statefrom)))
    if !all(allstates .∈ Ref(statespace))
        error("Data contains states that are not in the state space.")
    end

    # warn if state labels are not contiguous (e.g., 1,2,4 instead of 1,2,3)
    # Model creation will fail if states are not 1,2,3,...,n
    observed_states = sort(unique(filter(!=(0), allstates)))  # Exclude state 0 (censoring)
    if !isempty(observed_states)
        expected_states = collect(1:maximum(observed_states))
        if observed_states != expected_states
            missing_states = setdiff(expected_states, observed_states)
            if verbose
                @warn "State labels are not contiguous. States $(missing_states) are missing. State labels must be 1, 2, ..., n for model creation to succeed."
            end
        end
    end

    # warning if tmat specifies an allowed transition for which no such transitions were observed in the data
    n_rs = compute_number_transitions(data, tmat)
    for r in 1:size(tmat)[1]
        for s in 1:size(tmat)[2]
            if verbose && tmat[r,s]!=0 && n_rs[r,s]==0 
                @warn "Data does not contain any transitions from state $r to state $s"
            end
        end
    end

    # check that obstype is one of the allowed censoring schemes
    if any(data.obstype .∉ Ref([1,2]))
        emat_id = Int64.(emat[:,1])
        if any(data.obstype .∉ Ref([[1,2]; emat_id]))
            error("obstype should be one of 1, 2, or a censoring id from emat.")
        end
    end

    # check that stateto is 0 when obstype is not 1 or 2
    for i in Base.OneTo(nrow(data))
        if (data.obstype[i] > 2) & (data.stateto[i] .!= 0)            
            error("When obstype>2, stateto should be 0.")
        end
    end

    # check that subjects start in an observed state (statefrom!=0)
    for subj in Base.OneTo(nsubj)
        datasubj = filter(:id => ==(subj), data)
        if datasubj.statefrom[1] == 0          
            error("Subject $subj should not start in state 0.")
        end
    end

    # check that there is no row for a subject after they hit an absorbing state

end

"""
    check_SubjectWeights(SubjectWeights::Vector{Float64}, data::DataFrame)

Check that subject-level weights are properly specified.
"""
function check_SubjectWeights(SubjectWeights::Vector{Float64}, data::DataFrame)
    
    # check that the number of subject weights is correct
    if length(SubjectWeights) != length(unique(data.id))
        error("The length of SubjectWeights is not equal to the number of subjects.")
    end

    # check that the subject weights are non-negative
    if any(SubjectWeights .<= 0)
        error("The elements of SubjectWeights should be positive.")
    end
end

"""
    check_ObservationWeights(ObservationWeights::Vector{Float64}, data::DataFrame)

Check that observation-level weights are properly specified.
"""
function check_ObservationWeights(ObservationWeights::Vector{Float64}, data::DataFrame)
    
    # check that the number of observation weights is correct
    if length(ObservationWeights) != nrow(data)
        error("The length of ObservationWeights ($(length(ObservationWeights))) is not equal to the number of observations ($(nrow(data))).")
    end

    # check that the observation weights are non-negative
    if any(ObservationWeights .<= 0)
        error("The elements of ObservationWeights should be positive.")
    end
end

"""
    check_weight_exclusivity(SubjectWeights, ObservationWeights, nsubj::Int64)

Check that SubjectWeights and ObservationWeights are mutually exclusive and handle defaults.
Returns (SubjectWeights, ObservationWeights) where at most one is non-nothing.
"""
function check_weight_exclusivity(SubjectWeights, ObservationWeights, nsubj::Int64)
    
    # Check mutual exclusivity
    if !isnothing(SubjectWeights) && !isnothing(ObservationWeights)
        error("SubjectWeights and ObservationWeights are mutually exclusive. Specify only one.")
    end
    
    # SubjectWeights must always be a Vector{Float64} for the model struct
    # Set to ones if not provided (whether ObservationWeights is used or not)
    if isnothing(SubjectWeights)
        SubjectWeights = ones(Float64, nsubj)
    end
    
    return (SubjectWeights, ObservationWeights)
end
"""
    check_CensoringPatterns(CensoringPatterns::Matrix, tmat::Matrix)

Validate a user-supplied censoring patterns matrix to ensure it conforms to MultistateModels.jl requirements.
Accepts both Int64 (binary 0/1) and Float64 (emission probabilities in [0,1]).
"""
function check_CensoringPatterns(CensoringPatterns::Matrix{T}, tmat::Matrix) where T <: Real
    
    nrow, ncol = size(CensoringPatterns)

    # check for empty
    if nrow == 0 | ncol < 2
        error("The matrix CensoringPatterns seems to be empty, while there are censored states.")
    end

    # censoring patterns must be labelled as 3, 4, ...
    if !all(CensoringPatterns[:,1] .== 3:(nrow+2))
        error("The first column of the matrix `CensoringPatterns` must be of the form (3, 4, ...) .")
    end

    # check that values are in [0, 1]
    if any(CensoringPatterns[:,2:ncol] .< 0) || any(CensoringPatterns[:,2:ncol] .> 1)
        error("Columns 2, 3, ... of CensoringPatterns must have values in [0, 1].")
    end

    # censoring patterns must indicate the presence/absence of each state
    n_states = size(tmat, 1)
    if ncol - 1 .!= n_states
        error("The multistate model contains $n_states states, but CensoringPatterns contains $(ncol-1) states.")
    end

    # censoring patterns must have at least one possible state
    for i in 1:nrow
        if all(CensoringPatterns[i,2:ncol] .== 0)
            error("Censoring pattern $i has no allowed state.")
        end
        if all(CensoringPatterns[i,2:ncol] .== 1)
            println("All states are allowed in censoring pattern $(2+i).")
        end
        if sum(CensoringPatterns[i,2:ncol] .> 0) .== 1
            println("Censoring pattern $i has only one allowed state; if these observations are not censored there is no need to use a censoring pattern.")
        end
    end
end

"""
    build_tpm_book(tmat::Matrix{Int64}, tpm_index::Vector{DataFrame})

Build container for holding transition probability matrices.
"""
function build_tpm_book(T::DataType, tmat::Matrix{Int64}, tpm_index::Vector{DataFrame})

    # build the TPM container
    nstates = size(tmat, 1)
    nmats   = map(x -> nrow(x), tpm_index) 
    book    = [[zeros(T, nstates, nstates) for j in 1:nmats[i]] for i in eachindex(tpm_index)]

    return book
end

"""
    build_hazmat_book(tmat::Matrix{Int64}, tpm_index::Vector{DataFrame})

Build container for holding transition intensity matrices.
"""
function build_hazmat_book(T::DataType, tmat::Matrix{Int64}, tpm_index::Vector{DataFrame})
    # Making this "type aware" by using T::DataType so that autodiff worksA
    # build the TPM container
    nstates = size(tmat, 1)
    nmats   = map(x -> nrow(x), tpm_index) 
    book    = [zeros(T, nstates, nstates) for j in eachindex(tpm_index)]

    return book
end

"""
    build_tpm_mapping(data::DataFrame)

Construct bookkeeping objects for transition probability matrices for time intervals over which a multistate Markov process is piecewise homogeneous. The first bookkeeping object is a data frame that 
"""
function build_tpm_mapping(data::DataFrame) 

    # maps each row in dataset to TPM
    # first col is covar combn, second is tpm index
    tpm_map = zeros(Int64, nrow(data), 2)

    # check if the data contains covariates
    if ncol(data) == 6 # no covariates
        
        # get intervals
        gaps = data.tstop - data.tstart

        # get unique start and stop
        ugaps = sort(unique(gaps))

        # for solving Kolmogorov equations - saveats
        tpm_index = 
            [DataFrame(tstart = 0,
                       tstop  = ugaps,
                       datind = 0),]

        # first instance of each interval in the data
        for i in Base.OneTo(nrow(tpm_index[1]))
            tpm_index[1].datind[i] = 
                findfirst(gaps .== tpm_index[1].tstop[i])
        end

        # match intervals to unique tpms
        tpm_map[:,1] .= 1
        for i in Base.OneTo(size(tpm_map, 1))
            tpm_map[i,2] = findfirst(ugaps .== gaps[i])
        end    

    else
        # get unique covariates
        covars = data[:,Not(1:6)]
        ucovars = unique(data[:,Not(1:6)])

        # get gap times
        gaps = data.tstop - data.tstart

        # initialize tpm_index
        tpm_index = [DataFrame() for i in 1:nrow(ucovars)]

        # for each set of unique covariates find gaps
        for k in Base.OneTo(nrow(ucovars))

            # get indices for rows that have the covars
            covinds = findall(map(x -> all(x == ucovars[k,:]), eachrow(covars)) .== 1)

            # find unique gaps 
            ugaps = sort(unique(gaps[covinds]))

            # fill in tpm_index
            tpm_index[k] = DataFrame(tstart = 0, tstop = ugaps, datind = 0)

            # first instance of each interval in the data
            for i in Base.OneTo(nrow(tpm_index[k]))
                tpm_index[k].datind[i] = 
                    covinds[findfirst(gaps[covinds] .== tpm_index[k].tstop[i])]
            end

            # fill out the tpm_map 
            # match intervals to unique tpms
            tpm_map[covinds, 1] .= k
            for i in eachindex(covinds)
                tpm_map[covinds[i],2] = findfirst(ugaps .== gaps[covinds[i]])
            end  
        end
    end

    # return objects
    return tpm_index, tpm_map
end

"""
    build_fbmats(model)

Build the forward recursion matrices.
"""
function build_fbmats(model)

    # get sizes of stuff
    n_states = size(model.tmat, 1)
    n_times = [sum(model.data.id .== s) for s in unique(model.data.id)]

    # create the forward matrices
    fbmats = [zeros(Float64, n_times[s], n_states, n_states) for s in eachindex(n_times)]

    return fbmats
end

"""
    collapse_data(data::DataFrame; SubjectWeights::Vector{Float64} = ones(unique(data.id)))

Collapse subjects to create an internal representation of a dataset and optionally recompute a vector of subject weights.
"""
function collapse_data(data::DataFrame; SubjectWeights::Vector{Float64} = ones(Float64, length(unique(data.id))))
    
    # find unique subjects
    ids = unique(data.id)
    _data = [DataFrame() for k in 1:length(ids)]
    for k in ids
        _data[k] = data[findall(data.id .== k),Not(:id)]
    end
    _DataCollapsed = unique(_data)

    # find the collapsed dataset for each individual
    inds = map(x -> findfirst(_DataCollapsed .== Ref(x)), _data)

    # tabulate the SubjectWeights
    SubjectWeightsCollapsed = map(x -> sum(SubjectWeights[findall(inds .== x)]), unique(inds))

    # add a fake id variable to the collapsed datasets (for purchasing alcohol)
    for k in 1:length(_DataCollapsed)
        insertcols!(_DataCollapsed[k], :tstart, :id => fill(k, nrow(_DataCollapsed[k])))
    end

    # vcat
    DataCollapsed = reduce(vcat, _DataCollapsed)
       
    return DataCollapsed, SubjectWeightsCollapsed
end

"""
    initialize_surrogate!(model::MultistateSemiMarkovProcess; surrogate_parameters = nothing, surrogate_constraints = nothing, crude_inits = true, verbose = true)

Populate the field for the markov surrogate in semi-Markov models.
If the model does not have a markovsurrogate, builds and fits one.
If it exists, updates its parameters with fitted values.
"""
function initialize_surrogate!(model::MultistateProcess; surrogate_parameters = nothing, surrogate_constraints = nothing, crude_inits = true, verbose = true)

    # fit surrogate model
    surrogate_fitted = fit_surrogate(model; surrogate_parameters=surrogate_parameters, surrogate_constraints=surrogate_constraints, crude_inits=crude_inits, verbose=verbose)

    # Update model's markovsurrogate with fitted surrogate
    if isnothing(model.markovsurrogate)
        # Create new surrogate
        model.markovsurrogate = MarkovSurrogate(surrogate_fitted.hazards, surrogate_fitted.parameters)
    else
        # Update existing surrogate parameters
        copyto!(model.markovsurrogate.parameters.flat, surrogate_fitted.parameters.flat)
    end
end

"""
    make_subjdat(path::SamplePath, subjectdata::SubDataFrame)

Create a DataFrame for a single subject from a SamplePath object and the original data for that subject.
"""
function make_subjdat(path::SamplePath, subjectdata::SubDataFrame) 

    # times when the likelihood needs to be evaluated
    if (ncol(subjectdata) > 6) & (nrow(subjectdata) > 1)
        
        # times when covariates change
        keepvec = findall(map(i -> !isequal(subjectdata[i-1, 7:end], subjectdata[i, 7:end]), 2:nrow(subjectdata))) .+ 1 

        # utimes is the times in the path (which includes first and last times from subjectdata), and tstart when covariates change
        utimes = sort(unique(vcat(path.times, subjectdata.tstart[keepvec])))

    else 
        utimes = path.times
    end

    # get indices in the data object that correspond to the unique times
    datinds = searchsortedlast.(Ref(subjectdata.tstart), utimes)

    # get indices in the path that correspond to the unique times
    pathinds = searchsortedlast.(Ref(path.times), utimes)

    # make subject data
    subjdat_lik = DataFrame(
        tstart = utimes[Not(end)],
        tstop  = utimes[Not(begin)],
        increment = diff(utimes),
        sojourn = 0.0,
        sojournind = pathinds[Not(end)],
        statefrom = path.states[pathinds[Not(end)]],
        stateto   = path.states[pathinds[Not(begin)]])

    # compute sojourns
    subjdat_gdf = groupby(subjdat_lik, :sojournind)
    for g in subjdat_gdf
        g.sojourn .= cumsum([0.0; g.increment[Not(end)]])
    end

    # remove sojournind
    select!(subjdat_lik, Not(:sojournind))

    # tack on covariates if any
    if ncol(subjectdata) > 6
        # get covariates at the relevant times
        covars = subjectdata[datinds[Not(end)], Not([:id, :tstart, :tstop, :statefrom, :stateto, :obstype])]
        # concatenate and return
        subjdat_lik = hcat(subjdat_lik, covars)
    end
    
    # output
    return subjdat_lik
end