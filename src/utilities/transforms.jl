# ============================================================================
# Parameter Scale Transformations
# ============================================================================
#
# PARAMETER SCALE CONVENTION (v0.3.0+):
#
# ALL parameters are stored on NATURAL scale. Box constraints handle non-negativity.
# There is NO separate "estimation scale" - all transform functions are now identity.
#
# This simplified design:
#   1. Enables truly quadratic penalties P(β) = (λ/2)βᵀSβ for PIJCV
#   2. Uses Ipopt box constraints (lb ≥ 0) instead of exp() transforms
#   3. Eliminates confusion about which scale parameters are on
#   4. Makes hazard functions simpler (no internal transformations)
#
# The transform functions below are now identity operations for backward compatibility
# with calling code. They will be removed in a future version.
#
# ============================================================================

"""
    transform_baseline_to_natural(baseline::NamedTuple, family::Symbol, ::Type{T}) where T

IDENTITY TRANSFORM (v0.3.0+): Returns baseline unchanged.

Previously transformed parameters from estimation scale to natural scale.
Now all parameters are stored on natural scale, so this is identity.

# Arguments
- `baseline`: NamedTuple of baseline parameters (already on natural scale)
- `family`: Hazard family (unused, kept for API compatibility)
- `T`: Element type (unused, kept for API compatibility)

# Returns
Unchanged baseline NamedTuple
"""
@inline function transform_baseline_to_natural(baseline::NamedTuple, family::Symbol, ::Type{T}) where T
    # v0.3.0+: All parameters on natural scale, identity transform
    return baseline
end

"""
    transform_baseline_to_estimation(baseline::NamedTuple, family::Symbol)

IDENTITY TRANSFORM (v0.3.0+): Returns baseline unchanged.

Previously transformed parameters from natural scale to estimation scale.
Now all parameters are stored on natural scale, so this is identity.

# Arguments
- `baseline`: NamedTuple of baseline parameters (on natural scale)
- `family`: Hazard family (unused, kept for API compatibility)

# Returns
Unchanged baseline NamedTuple
"""
@inline function transform_baseline_to_estimation(baseline::NamedTuple, family::Symbol)
    # v0.3.0+: All parameters on natural scale, identity transform
    return baseline
end

"""
    to_natural_scale(params_nested::NamedTuple, hazards, ::Type{T}) where T

IDENTITY TRANSFORM (v0.3.0+): Returns params_nested unchanged.

Previously transformed parameters from estimation scale to natural scale.
Now all parameters are stored on natural scale, so this is identity.

# Arguments
- `params_nested`: NamedTuple of per-hazard parameter NamedTuples (already on natural scale)
- `hazards`: Vector of hazard objects (kept for API compatibility)
- `T`: Element type for AD compatibility

# Returns
Unchanged NamedTuple of per-hazard parameter NamedTuples
"""
function to_natural_scale(params_nested::NamedTuple, hazards, ::Type{T}) where T
    hazard_keys = keys(params_nested)
    transformed_hazards = map(enumerate(hazard_keys)) do (idx, hazname)
        hazard_params = params_nested[hazname]
        
        # Find the hazard with the matching name
        # We cannot assume hazards[idx] corresponds to hazname because params_nested keys 
        # might be sorted differently than the hazards vector
        h_idx = findfirst(h -> h.hazname == hazname, hazards)
        if isnothing(h_idx)
            throw(ArgumentError("Hazard $hazname found in parameters but not in hazards vector. " *
                               "Check that parameter names match hazard definitions."))
        end
        hazard = hazards[h_idx]
        
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

"""
    to_estimation_scale(params_nested::NamedTuple, hazards)

IDENTITY TRANSFORM (v0.3.0+): Returns params_nested unchanged.

Previously transformed parameters from natural scale to estimation scale.
Now all parameters are stored on natural scale, so this is identity.

# Arguments
- `params_nested`: NamedTuple of per-hazard parameter NamedTuples (on natural scale)
- `hazards`: Vector of hazard objects (kept for API compatibility)

# Returns
Unchanged NamedTuple of per-hazard parameter NamedTuples
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
As of v0.3.0, all parameters are on natural scale.
"""
function set_parameters_flat!(model::MultistateProcess, flat_params::AbstractVector)
    # Unflatten to get nested structure
    params_nested = unflatten(model.parameters.reconstructor, flat_params)
    
    # Get new flat version (reconstructor stays the same)
    new_flat = flatten(model.parameters.reconstructor, params_nested)
    
    # Update spline hazards if needed - v0.3.0+: params are on natural scale
    for i in eachindex(model.hazards)
        if isa(model.hazards[i], _SplineHazard)
            hazard_params = values(params_nested)[i]
            params_vec = extract_params_vector(hazard_params)
            remake_splines!(model.hazards[i], params_vec)
            set_riskperiod!(model.hazards[i])
        end
    end
    
    # v0.3.0+: nested == natural (no transformation needed, but kept for API compatibility)
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

# ============================================================================
# Parameter Getter Functions
# ============================================================================

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
            throw(ArgumentError("Could not find hazard name for index $h. Available hazards: $(keys(model.hazkeys))"))
        end
        
        # Return appropriate representation
        if scale == :natural
            return model.parameters.natural[hazname]
        elseif scale == :nested
            return model.parameters.nested[hazname]
        else
            throw(ArgumentError("scale must be :natural or :nested (got :$scale)"))
        end
    else
        throw(ArgumentError("scale must be :natural, :nested, or :log (got :$scale)"))
    end
end
