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
    elseif family in (:exp, :wei)
        # Exponential and Weibull: all baseline parameters are positive (exp)
        transformed_values = map(v -> exp(v), values(baseline))
        return NamedTuple{keys(baseline)}(transformed_values)
    else
        throw(ArgumentError("Unknown hazard family :$family for baseline transformation. Expected one of :exp, :wei, :gom, :sp"))
    end
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
    elseif family in (:exp, :wei)
        # Exponential and Weibull: all baseline parameters are positive (log)
        transformed_values = map(v -> log(v), values(baseline))
        return NamedTuple{keys(baseline)}(transformed_values)
    else
        throw(ArgumentError("Unknown hazard family :$family for baseline transformation. Expected one of :exp, :wei, :gom, :sp"))
    end
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
        else  # :nested or :transformed (backward compat)
            return model.parameters.nested[hazname]
        end
    else
        throw(ArgumentError("scale must be :natural, :nested, or :log (got :$scale)"))
    end
end
