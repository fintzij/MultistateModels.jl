# model_assembly.jl - Model component assembly and parameter building
#
# This file contains:
# - build_hazards: Orchestrates hazard building from specs
# - build_parameters: Creates parameter structure from hazard parameters
# - build_totalhazards: Creates total hazard objects per state
# - build_emat: Creates emission matrix for HMM
# - Helper functions for weight handling and validation

"""
        build_hazards(hazards::HazardFunction...; data::DataFrame, surrogate::Bool = false)

Instantiate runtime hazard objects from the symbolic specifications produced by
`Hazard` or `@hazard`. This attaches the relevant design matrices, builds the
baseline parameter containers, and returns everything needed by `multistatemodel`.

# Arguments
- `hazards`: one or more hazard specifications. Formula arguments are optional for
    intercept-only transitions, matching the `Hazard` constructor semantics.
- `data`: `DataFrame` containing the covariates referenced by the hazard formulas.
- `surrogate`: when `true`, force exponential baselines (useful for MCEM surrogates).

# Returns
1. `_hazards`: callable hazard objects used internally by the simulator/likelihood.
2. `parameters`: the ParameterHandling.jl structure (flat, natural, transforms, etc.).
3. `hazkeys`: dictionary mapping hazard names (e.g. `:h12`) to indices in `_hazards`.
"""
function build_hazards(hazards::HazardFunction...; data::DataFrame, surrogate = false)
    contexts = [_prepare_hazard_context(h, data; surrogate = surrogate) for h in hazards]

    _hazards = Vector{_Hazard}(undef, length(hazards))
    parameters_list = Vector{Vector{Float64}}()  # Collect parameters as nested vectors
    hazkeys = Dict{Symbol, Int64}()

    for (idx, ctx) in enumerate(contexts)
        hazkeys[ctx.hazname] = idx
        builder = get(_HAZARD_BUILDERS, ctx.family, nothing)
        builder === nothing && throw(ArgumentError("Unknown hazard family: $(ctx.family). Supported families: $(keys(_HAZARD_BUILDERS))"))
        hazard_struct, hazpars = builder(ctx)
        _hazards[idx] = hazard_struct
        push!(parameters_list, hazpars)
    end

    parameters = build_parameters(parameters_list, hazkeys, _hazards)
    return _hazards, parameters, hazkeys
end

"""
    build_parameters(parameters::Vector{Vector{Float64}}, hazkeys::Dict{Symbol, Int64}, hazards::Vector{<:_Hazard})

Create a parameters structure from nested parameter vectors.

Returns a NamedTuple with fields:
- `flat`: Vector{Float64} of all parameters (log scale for baseline, as-is for covariates)
- `nested`: NamedTuple of NamedTuples per hazard with `baseline` and optional `covariates` fields
- `unflatten`: Function to reconstruct nested structure from flat vector

Parameters are stored on natural scale (v0.3.0+). Box constraints enforce positivity.
"""
function build_parameters(parameters::Vector{Vector{Float64}}, hazkeys::Dict{Symbol, Int64}, hazards::Vector{<:_Hazard})
    # Build nested parameters structure per hazard
    params_nested_pairs = [
        begin
            # Robustly find hazard by name
            h_idx = findfirst(h -> h.hazname == hazname, hazards)
            if isnothing(h_idx)
                throw(ArgumentError("Hazard $hazname not found in hazards vector. Available: $(getfield.(hazards, :hazname))"))
            end
            hazard = hazards[h_idx]
            hazname => build_hazard_params(parameters[idx], hazard.parnames, hazard.npar_baseline, hazard.npar_total)
        end
        for (hazname, idx) in sort(collect(hazkeys), by = x -> x[2])
    ]
    params_nested = NamedTuple(params_nested_pairs)
    
    # Create ReConstructor for AD-compatible flatten/unflatten
    reconstructor = ReConstructor(params_nested, unflattentype=UnflattenFlexible())
    params_flat = flatten(reconstructor, params_nested)
    
    # Return structure (v0.3.0+: no separate .natural field, compute on-demand via accessors)
    return (
        flat = params_flat,
        nested = params_nested,
        reconstructor = reconstructor
    )
end

### Total hazards
"""
    build_totalhazards(_hazards, tmat)

 Return a vector of _TotalHazard objects for each origin state, which may be of subtype `_TotalHazardAbsorbing` or `_TotalHazardTransient`. 

 Accepts the internal array _hazards corresponding to allowable transitions, and the transition matrix tmat
"""
function build_totalhazards(_hazards, tmat)

    # initialize a vector for total hazards
    _totalhazards = Vector{_TotalHazard}(undef, size(tmat, 1))

    # populate the vector of total hazards
    for h in eachindex(_totalhazards) 
        if sum(tmat[h,:]) == 0
            _totalhazards[h] = 
                _TotalHazardAbsorbing()
        else
            _totalhazards[h] = 
                _TotalHazardTransient(tmat[h, findall(tmat[h,:] .!= 0)])
        end
    end

    return _totalhazards
end

"""
    build_emat(data::DataFrame, CensoringPatterns::Matrix{Float64}, EmissionMatrix::Union{Nothing, Matrix{Float64}}, tmat::Matrix{Int64})

Create the emission matrix used by the forward–backward routines. Each row
corresponds to an observation; columns correspond to latent states. The helper
marks which states are compatible with each observation or censoring code using
`CensoringPatterns` (if provided) and the transition structure `tmat`.

If `EmissionMatrix` is provided, it is used directly (allowing observation-specific emission probabilities).
Otherwise, the emission matrix is constructed from `CensoringPatterns` and observation types.

Values represent P(observation | state): 0 means impossible, 1 means certain, values in (0,1) 
represent soft evidence.
"""
function build_emat(data::DataFrame, CensoringPatterns::Matrix{Float64}, EmissionMatrix::Union{Nothing, Matrix{Float64}}, tmat::Matrix{Int64})
    
    n_obs = nrow(data)
    n_states = size(tmat, 1)
    
    # If EmissionMatrix is provided, validate and use it directly
    if !isnothing(EmissionMatrix)
        if size(EmissionMatrix) != (n_obs, n_states)
            throw(ArgumentError("EmissionMatrix must have dimensions ($(n_obs), $(n_states)), got $(size(EmissionMatrix))."))
        end
        if any(EmissionMatrix .< 0) || any(EmissionMatrix .> 1)
            throw(ArgumentError("EmissionMatrix values must be in [0, 1]."))
        end
        for i in 1:n_obs
            if all(EmissionMatrix[i,:] .== 0)
                throw(ArgumentError("EmissionMatrix row $i has no allowed states (all zeros)."))
            end
        end
        return EmissionMatrix
    end
    
    # Otherwise, build from CensoringPatterns
    emat = zeros(Float64, n_obs, n_states)

    for i in 1:n_obs
        if data.obstype[i] ∈ [OBSTYPE_EXACT, OBSTYPE_PANEL] # state known (exact or panel): stateto observed
            emat[i,data.stateto[i]] = 1.0
        elseif data.obstype[i] == 0 # state fully censored: all states possible
            emat[i,:] .= 1.0
        else # state partially censored (obstype > 2): use censoring pattern
            emat[i,:] .= CensoringPatterns[data.obstype[i] - CENSORING_OBSTYPE_OFFSET, 2:n_states+1]
        end 
    end

    return emat
end

# Weight handling utilities

@inline function _ensure_subject_weights(SubjectWeights, nsubj)
    return isnothing(SubjectWeights) ? ones(Float64, nsubj) : SubjectWeights
end

@inline function _ensure_observation_weights(ObservationWeights)
    return ObservationWeights  # nothing stays nothing, vector stays vector
end

@inline function _prepare_censoring_patterns(CensoringPatterns, n_states)
    return isnothing(CensoringPatterns) ? Matrix{Float64}(undef, 0, n_states) : Float64.(CensoringPatterns)
end

# Validation utilities
function _validate_inputs!(data::DataFrame,
                           tmat::Matrix{Int64},
                           CensoringPatterns::Matrix{Float64},
                           SubjectWeights,
                           ObservationWeights;
                           verbose::Bool,
                           phase_to_state::Union{Nothing, Vector{Int}} = nothing)
    check_data!(data, tmat, CensoringPatterns; verbose = verbose, phase_to_state = phase_to_state)
    
    # Validate weights
    if !isnothing(SubjectWeights)
        check_SubjectWeights(SubjectWeights, data)
    end
    if !isnothing(ObservationWeights)
        check_ObservationWeights(ObservationWeights, data)
    end
    
    if any(data.obstype .∉ Ref([1, 2]))
        check_CensoringPatterns(CensoringPatterns, tmat)
    end
end

# Model classification utilities

@inline function _observation_mode(data::DataFrame)
    if all(data.obstype .== 1)
        return :exact
    elseif all(data.obstype .∈ Ref([1, 2]))
        return :panel
    else
        return :censored
    end
end

@inline function _process_class(hazards::Vector{<:_Hazard})
    # Use the _is_markov_hazard helper for consistent classification
    # This correctly handles PhaseTypeCoxianHazard (Markov) and degree-0 splines
    return all(_is_markov_hazard.(hazards)) ? :markov : :semi_markov
end

@inline function _model_constructor(mode::Symbol, process::Symbol)
    # All unfitted models use MultistateModel now
    # Behavior is determined by content (hazards, observation type), not struct type
    return MultistateModel
end

function _assemble_model(mode::Symbol,
                         process::Symbol,
                         components::NamedTuple,
                         surrogate::Union{Nothing, MarkovSurrogate},
                         modelcall;
                         phasetype_surrogate::Union{Nothing, AbstractSurrogate} = nothing,
                         phasetype_expansion::Union{Nothing, PhaseTypeExpansion} = nothing)
    # Generate parameter bounds at model construction time
    # This ensures bounds are always available for fitting and parameter validation
    bounds = _generate_package_bounds_from_components(components.parameters.flat, components.hazards, components.hazkeys)
    
    # Single MultistateModel struct handles all cases
    return MultistateModel(
        components.data,
        components.parameters,
        bounds,
        components.hazards,
        components.totalhazards,
        components.tmat,
        components.emat,
        components.hazkeys,
        components.subjinds,
        components.SubjectWeights,
        components.ObservationWeights,
        components.CensoringPatterns,
        surrogate,
        phasetype_surrogate,
        modelcall,
        phasetype_expansion,
    )
end

"""
    _generate_package_bounds_from_components(flat_params, hazards, hazkeys) -> NamedTuple{(:lb, :ub)}

Generate package-level parameter bounds during model construction.
This is called by `_assemble_model` to initialize the bounds field.

Uses the same logic as `_generate_package_bounds` in bounds.jl but operates
on the components before the model struct is created.
"""
function _generate_package_bounds_from_components(flat_params::Vector{Float64}, 
                                                   hazards::Vector{<:_Hazard},
                                                   hazkeys::Dict{Symbol, Int64})
    n_params = length(flat_params)
    lb = fill(-Inf, n_params)
    ub = fill(Inf, n_params)
    
    # Lower bound for non-negative parameters
    const_nonneg_lb = 0.0
    
    # Iterate over hazards in parameter order
    param_offset = 0
    for (hazname, _) in sort(collect(hazkeys), by = x -> x[2])
        # Find hazard by name
        haz_idx = findfirst(h -> h.hazname == hazname, hazards)
        hazard = hazards[haz_idx]
        family = hazard.family
        n_baseline = hazard.npar_baseline
        n_total = hazard.npar_total
        
        # Set bounds for baseline parameters based on family
        baseline_lb = _get_baseline_lb_internal(family, n_baseline, const_nonneg_lb)
        lb[param_offset+1:param_offset+n_baseline] .= baseline_lb
        
        # Covariate coefficients are unconstrained (already initialized to -Inf/Inf)
        
        param_offset += n_total
    end
    
    return (lb = lb, ub = ub)
end

"""
    _get_baseline_lb_internal(family, n_baseline, nonneg_lb) -> Vector{Float64}

Get lower bounds for baseline parameters by hazard family.
Internal version used during model construction.
"""
function _get_baseline_lb_internal(family::Symbol, n_baseline::Int, nonneg_lb::Float64)
    if family == :exp
        return fill(nonneg_lb, n_baseline)
    elseif family == :wei
        return fill(nonneg_lb, n_baseline)
    elseif family == :gom
        # Gompertz: shape ∈ ℝ, rate ≥ 0
        if n_baseline == 2
            return [-Inf, nonneg_lb]
        else
            return fill(nonneg_lb, n_baseline)
        end
    elseif family == :sp
        return fill(nonneg_lb, n_baseline)
    elseif family == :pt
        return fill(nonneg_lb, n_baseline)
    else
        return fill(-Inf, n_baseline)
    end
end
