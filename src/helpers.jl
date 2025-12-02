"""
    set_parameters!(model::MultistateProcess, newvalues::Union{VectorOfVectors,Vector{Vector{Float64}}})

Set model parameters given a nested vector of values (one vector per hazard).
Values should be on log scale for positive-constrained parameters (baseline rates, shapes, scales).

# Arguments
- `model::MultistateProcess`: The model to update
- `newvalues`: Nested vector or VectorOfVectors with parameters for each hazard

# Note
Updates model.parameters with the new values and remakes spline hazards as needed.
"""
function set_parameters!(model::MultistateProcess, newvalues::Union{VectorOfVectors,Vector{Vector{Float64}}})
    
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
    
    # Rebuild parameters from the new values
    # newvalues are on log scale, so exp to get natural scale for safe_positive
    params_transformed_pairs = [
        hazname => safe_positive(exp.(Vector{Float64}(newvalues[idx])))
        for (hazname, idx) in sort(collect(model.hazkeys), by = x -> x[2])
    ]
    
    params_transformed = NamedTuple(params_transformed_pairs)
    params_flat, unflatten_fn = ParameterHandling.flatten(params_transformed)
    params_natural = ParameterHandling.value(params_transformed)
    
    model.parameters = (
        flat = params_flat,
        transformed = params_transformed,
        natural = params_natural,
        unflatten = unflatten_fn
    )
    
    return nothing
end

"""
    set_parameters!(model::MultistateProcess, newvalues::Tuple)

Set model parameters given a tuple of vectors parameterizing transition intensities. 
Values should be on log scale for positive-constrained parameters.

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
    
    # Rebuild parameters from the new values (log scale input -> natural scale via exp)
    params_transformed_pairs = [
        hazname => safe_positive(exp.(Vector{Float64}(collect(newvalues[idx]))))
        for (hazname, idx) in sort(collect(model.hazkeys), by = x -> x[2])
    ]
    
    params_transformed = NamedTuple(params_transformed_pairs)
    params_flat, unflatten_fn = ParameterHandling.flatten(params_transformed)
    params_natural = ParameterHandling.value(params_transformed)
    
    model.parameters = (
        flat = params_flat,
        transformed = params_transformed,
        natural = params_natural,
        unflatten = unflatten_fn
    )
    
    return nothing
end

"""
    set_parameters!(model::MultistateProcess, newvalues::NamedTuple)

Set model parameters given a NamedTuple with hazard names as keys.
Values should be on log scale for positive-constrained parameters.

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
            # Keep current values (need to convert from natural back to log scale)
            new_param_vectors[idx] = log.(Vector{Float64}(collect(current_natural[idx])))
        end
    end
    
    params_transformed_pairs = [
        hazname => safe_positive(exp.(new_param_vectors[idx]))
        for (hazname, idx) in sort(collect(model.hazkeys), by = x -> x[2])
    ]
    
    params_transformed = NamedTuple(params_transformed_pairs)
    params_flat, unflatten_fn = ParameterHandling.flatten(params_transformed)
    params_natural = ParameterHandling.value(params_transformed)
    
    model.parameters = (
        flat = params_flat,
        transformed = params_transformed,
        natural = params_natural,
        unflatten = unflatten_fn
    )
    
    return nothing
end

"""
    set_parameters!(model::MultistateProcess, h::Int64, newvalues::Vector{Float64})

Set parameters for a single hazard by hazard index.
Values should be on log scale for positive-constrained parameters.

# Arguments
- `model::MultistateProcess`: The model to update
- `h::Int64`: Index of the hazard to update
- `newvalues::Vector{Float64}`: New parameter values (log scale)

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
            # Keep current values (convert from natural back to log scale)
            new_param_vectors[idx] = log.(Vector{Float64}(collect(current_natural[idx])))
        end
    end
    
    params_transformed_pairs = [
        hazname => safe_positive(exp.(new_param_vectors[idx]))
        for (hazname, idx) in sort(collect(model.hazkeys), by = x -> x[2])
    ]
    
    params_transformed = NamedTuple(params_transformed_pairs)
    params_flat, unflatten_fn = ParameterHandling.flatten(params_transformed)
    params_natural = ParameterHandling.value(params_transformed)
    
    model.parameters = (
        flat = params_flat,
        transformed = params_transformed,
        natural = params_natural,
        unflatten = unflatten_fn
    )
    
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

# Minimum value for parameters to avoid ParameterHandling.positive() epsilon errors
# ParameterHandling uses ε = sqrt(eps(Float64)) ≈ 1.49e-8, so we use 1e-7 as floor
const PARAM_FLOOR = 1e-7

"""
    safe_positive(x::AbstractVector{<:Real})

Create a ParameterHandling.positive() transformation with a floor to prevent
"must be greater than ε" errors. Values below `PARAM_FLOOR` are clamped.

This handles edge cases where optimization produces very small parameter values,
particularly in phase-type models with sparse data for some transitions.
"""
function safe_positive(x::AbstractVector{<:Real})
    return ParameterHandling.positive(max.(x, PARAM_FLOOR))
end

"""
    get_log_scale_params(parameters)

Extract log-scale (unconstrained) parameters from a parameters structure.
This is the correct scale for passing to hazard functions.

Hazard functions expect log-scale parameters (baseline rates, shapes, scales)
because they apply exp() internally. The ParameterHandling `natural` field
contains natural-scale values (after exp transform), which should NOT be passed
to hazard functions directly.

# Arguments
- `parameters`: A NamedTuple with `transformed` field containing PositiveArray wrappers

# Returns
- `Tuple`: Log-scale parameters indexed by hazard number

# Example
```julia
# Get log-scale params for hazard evaluation
log_pars = get_log_scale_params(model.parameters)
# Pass to likelihood
ll = loglik(log_pars, path, model.hazards, model)
```
"""
function get_log_scale_params(parameters)
    # Extract unconstrained (log-scale) values from each PositiveArray in transformed
    return map(x -> x.unconstrained_value, values(parameters.transformed))
end

"""
    get_elem_ptr(parameters)

Get element pointer array for converting flat parameter vector to nested structure.
Returns a vector where elem_ptr[i]:elem_ptr[i+1]-1 gives indices for hazard i.

This replaces the old VectorOfVectors elem_ptr field with a computation from parameters.
"""
function get_elem_ptr(parameters)
    sizes = [length(v) for v in values(parameters.natural)]
    return cumsum([1; sizes])
end

"""
    nest_params(flat_params::AbstractVector, parameters)

Convert a flat parameter vector to a VectorOfVectors using parameters structure.
This is the **preferred AD-compatible method** for converting flat optimizer 
parameters to nested form for hazard evaluation.

# Arguments
- `flat_params::AbstractVector`: Flat parameter vector (from optimizer or model.parameters.flat)
- `parameters`: The model's parameters NamedTuple containing `natural` field

# Returns  
- `VectorOfVectors`: Nested view of parameters indexed by hazard number (1-based)

# AD Compatibility
This function works with ForwardDiff.Dual numbers because VectorOfVectors creates 
views without type conversion. Use this instead of `unflatten_to_tuple` for any
code that will be differentiated.

# Example
```julia
# In AD-compatible likelihood computation:
pars = nest_params(parameters, model.parameters)
haz_params = pars[1]  # Parameters for first hazard (log scale)
```

See also: [`get_log_scale_params`](@ref), [`unflatten_to_tuple`](@ref)
"""
function nest_params(flat_params::AbstractVector, parameters)
    elem_ptr = get_elem_ptr(parameters)
    return VectorOfVectors(flat_params, elem_ptr)
end

"""
    set_parameters_flat!(model::MultistateProcess, flat_params::AbstractVector)

Set model parameters from a flat parameter vector (as used by optimizers).
The flat vector is unflattened using the model's parameters.unflatten function.

# Arguments
- `model::MultistateProcess`: The model to update  
- `flat_params::AbstractVector`: Flat parameter vector (same format as parameters.flat)

# Note
This is the primary method for updating parameters during optimization.
Spline hazards are remade with the new parameters.
"""
function set_parameters_flat!(model::MultistateProcess, flat_params::AbstractVector)
    # Unflatten to get transformed structure
    params_transformed = model.parameters.unflatten(flat_params)
    params_natural = ParameterHandling.value(params_transformed)
    
    # Get new unflatten function (in case structure changed)
    new_flat, new_unflatten = ParameterHandling.flatten(params_transformed)
    
    # Update spline hazards if needed
    natural_tuple = values(params_natural)
    for i in eachindex(model.hazards)
        if isa(model.hazards[i], _SplineHazard)
            # Convert back to log scale for spline remake
            log_params = log.(Vector{Float64}(collect(natural_tuple[i])))
            remake_splines!(model.hazards[i], log_params)
            set_riskperiod!(model.hazards[i])
        end
    end
    
    # Update parameters
    model.parameters = (
        flat = Vector{Float64}(flat_params),
        transformed = params_transformed,
        natural = params_natural,
        unflatten = new_unflatten
    )
    
    return nothing
end

"""
    unflatten_to_tuple(parameters::AbstractVector, unflatten_fn)

!!! warning "Deprecated"
    This function is NOT compatible with ForwardDiff automatic differentiation
    because ParameterHandling's unflatten captures Float64 type constraints at
    construction time. Use `nest_params(parameters, model.parameters)` instead.

Convert flat parameter vector to indexable Tuple for integer indexing.
Returns a Tuple where `result[hazard_index]` gives parameters for that hazard.

# Arguments
- `parameters::AbstractVector`: Flat parameter vector (from optimizer or parameters.flat)
- `unflatten_fn`: The unflatten function from ParameterHandling.flatten()

# Returns
- `Tuple`: Parameters indexed by hazard number (1-based), compatible with pars[hazard_idx]

# AD Compatibility Warning
This function does NOT work with ForwardDiff.Dual numbers. ParameterHandling's
unflatten function captures type constraints (Float64) at construction, causing
MethodError when passed Dual numbers during differentiation.

For AD-compatible code, use `nest_params` which creates VectorOfVectors views:
```julia
# AD-compatible (RECOMMENDED):
pars = nest_params(parameters, model.parameters)

# NOT AD-compatible (DEPRECATED):
pars = unflatten_to_tuple(parameters, model.parameters.unflatten)
```

See also: [`nest_params`](@ref), [`get_log_scale_params`](@ref)
"""
function unflatten_to_tuple(parameters::AbstractVector, unflatten_fn)
    # Unflatten to get NamedTuple of PositiveArray wrappers
    transformed_nt = unflatten_fn(parameters)
    # Extract unconstrained (log-scale) values from each PositiveArray
    # This does NOT preserve ForwardDiff.Dual types - use nest_params instead!
    unconstrained = map(x -> x.unconstrained_value, values(transformed_nt))
    return unconstrained
end

#=============================================================================
PHASE 3: Parameter Getter Functions
=============================================================================# 

"""
    get_parameters_flat(model::MultistateProcess)

Get model parameters as a flat vector suitable for optimization.

This is the representation that optimization algorithms expect - a single
flat Vector{Float64} with all parameters concatenated.

# Returns
- `Vector{Float64}`: Flat parameter vector (transformed/log scale)

# Example
```julia
flat_params = get_parameters_flat(model)
# Use in optimizer
result = optimize(flat_params) do p
    # unflatten and update model
    objective_function(p, model)
end
```

# See also
- [`get_parameters_transformed`](@ref) - Get parameters with transformations as NamedTuple
- [`get_parameters_natural`](@ref) - Get parameters on natural scale as NamedTuple
- [`get_unflatten_fn`](@ref) - Get function to unflatten parameters
"""
function get_parameters_flat(model::MultistateProcess)
    return model.parameters.flat
end

"""
    get_parameters_transformed(model::MultistateProcess)

Get model parameters in transformed (log) scale as a NamedTuple.

Parameters are organized by hazard name with log transformations applied
for positive constraints (baseline, shape, scale). Covariate coefficients
remain on natural scale.

# Returns
- `NamedTuple`: Parameters by hazard name with transformations

# Example
```julia
params_trans = get_parameters_transformed(model)
# Access specific hazard
h12_trans = params_trans.h12  # e.g., positive([log(2.0)])
```

# See also
- [`get_parameters_flat`](@ref) - Get parameters as flat vector for optimization
- [`get_parameters_natural`](@ref) - Get parameters on natural scale
"""
function get_parameters_transformed(model::MultistateProcess)
    return model.parameters.transformed
end

"""
    get_parameters_natural(model::MultistateProcess)

Get model parameters on natural scale as a NamedTuple.

All positive constraints are inverted (exp applied to log scale parameters).
This is the "human-readable" representation where baseline rates, shapes, 
and scales are on their natural positive scale.

# Returns
- `NamedTuple`: Parameters by hazard name on natural scale

# Example
```julia
params_nat = get_parameters_natural(model)
# Access specific hazard - values are on natural scale
h12_baseline = params_nat.h12[1]  # e.g., 2.0 (not log(2.0))
```

# See also
- [`get_parameters_flat`](@ref) - Get parameters as flat vector
- [`get_parameters_transformed`](@ref) - Get parameters with transformations
"""
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
params_transformed = unflatten(flat_params)
params_natural = ParameterHandling.value(params_transformed)
```

# See also
- [`get_parameters_flat`](@ref) - Get flat parameter vector
- [`get_parameters_transformed`](@ref) - Get transformed parameters
"""
function get_unflatten_fn(model::MultistateProcess)
    return model.parameters.unflatten
end

"""
    get_parameters(model::MultistateProcess, h::Int64; scale::Symbol=:natural)

Get parameters for a specific hazard by index.

# Arguments
- `model::MultistateProcess`: The model
- `h::Int64`: Hazard index (1-based)
- `scale::Symbol`: One of `:natural`, `:transformed`, or `:log` (default: `:natural`)

# Returns
- `Vector{Float64}`: Parameters for the specified hazard

# Scales
- `:natural` - Natural scale (baseline/shape/scale are positive values)
- `:transformed` - Transformed scale (from ParameterHandling, includes wrapper)
- `:log` - Log scale (from model.parameters VectorOfVectors, backward compat)

# Examples
```julia
# Get natural scale parameters for hazard 1
params_nat = get_parameters(model, 1)  # Returns [2.0, 1.5, ...]

# Get log scale parameters (legacy representation)
params_log = get_parameters(model, 1, scale=:log)  # Returns [log(2.0), log(1.5), ...]

# Get transformed parameters (with ParameterHandling wrapper)
params_trans = get_parameters(model, 1, scale=:transformed)
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
    elseif scale == :natural || scale == :transformed
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
        else  # :transformed
            return model.parameters.transformed[hazname]
        end
    else
        throw(ArgumentError("scale must be :natural, :transformed, or :log (got :$scale)"))
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