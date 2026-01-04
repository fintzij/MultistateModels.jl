# =============================================================================
# Likelihood Utilities: Common functions for likelihood computation
# =============================================================================
#
# Contents:
# - ForwardDiff Dual number handling (_unwrap_to_float)
# - Parameter preparation (prepare_parameters)
# - Time transform optimization (enabled check)
# - Single path likelihood (loglik, loglik_path)
#
# Split from loglik.jl for maintainability (January 2026)
# =============================================================================

# =============================================================================
# Helper Functions for ForwardDiff Dual Types
# =============================================================================

"""
    _unwrap_to_float(x)

Recursively unwrap ForwardDiff Dual numbers to their underlying Float64 value.
Handles nested Duals (e.g., during Hessian computation with forward-over-forward AD).
"""
_unwrap_to_float(x::Float64) = x
_unwrap_to_float(x::Real) = Float64(x)  # Handle other numeric types
_unwrap_to_float(x::ForwardDiff.Dual) = _unwrap_to_float(ForwardDiff.value(x))

# =============================================================================
# Parameter Preparation (dispatch-based normalization)
# =============================================================================

"""
    prepare_parameters(parameters, model::MultistateProcess)

Normalize parameter representations for downstream hazard calls.
Uses multiple dispatch to handle different parameter container types.

For flat vectors, this is equivalent to calling `unflatten_natural(p, model)`.

# Supported types
- `Tuple`: Nested parameters indexed by hazard number (returned as-is)
- `NamedTuple`: Parameters keyed by hazard name (returned as-is)
- `AbstractVector{<:AbstractVector}`: Already nested format (returned as-is)
- `AbstractVector{<:Real}`: Flat parameter vector (unflattened via `unflatten_natural`)

# Note on AD Compatibility
Uses `unflatten_natural` to handle both Float64 and ForwardDiff.Dual types correctly.

See also: [`unflatten_natural`](@ref), [`unflatten_estimation`](@ref)
"""
prepare_parameters(p::Tuple, ::MultistateProcess) = p
prepare_parameters(p::NamedTuple, ::MultistateProcess) = p
prepare_parameters(p::AbstractVector{<:AbstractVector}, ::MultistateProcess) = p

function prepare_parameters(p::AbstractVector{<:Real}, model::MultistateProcess)
    # Return NamedTuple indexed by hazard name (not Tuple of values)
    # Downstream code accesses parameters[hazard.hazname]
    return unflatten_natural(p, model)
end

# =============================================================================
# Exactly observed sample paths
# =============================================================================

@inline function _time_transform_enabled(totalhazard::_TotalHazard, hazards::Vector{<:_Hazard})
    if totalhazard isa _TotalHazardAbsorbing
        return false
    end

    for idx in totalhazard.components
        if hazards[idx].metadata.time_transform
            return true
        end
    end

    return false
end

"""
    loglik_path(pars, subjectdata::DataFrame, hazards::Vector{<:_Hazard}, totalhazards::Vector{<:_TotalHazard}, tmat::Array{Int,2})

Log-likelihood for a single sample path. The subject data is provided as a DataFrame with columns including:
- `sojourn`: Time spent in current state at start of interval
- `increment`: Time increment for this interval
- `statefrom`: State at start of interval
- `stateto`: State at end of interval
- Additional covariate columns

This function is called after converting a SamplePath object to DataFrame format using `make_subjdat()`.
"""
loglik_path = function(pars, subjectdata::DataFrame, hazards::Vector{<:_Hazard}, totalhazards::Vector{<:_TotalHazard}, tmat::Array{Int,2})

     # initialize log likelihood
     ll = 0.0

    tt_context = maybe_time_transform_context(pars, subjectdata, hazards)
 
     # recurse through the sample path
     for i in Base.OneTo(nrow(subjectdata))

        # accumulate survival probabilty
        origin_state = subjectdata.statefrom[i]
        use_transform = _time_transform_enabled(totalhazards[origin_state], hazards)
        
        # Use @view to avoid DataFrameRow allocation
        row_data = @view subjectdata[i, :]

        val = survprob(
            subjectdata.sojourn[i],
            subjectdata.sojourn[i] + subjectdata.increment[i],
            pars,
            row_data,
            totalhazards[origin_state],
            hazards;
            give_log = true,
            apply_transform = use_transform,
            cache_context = tt_context)
        
        ll += val
 
        # accumulate hazard if there is a transition
        if subjectdata.statefrom[i] != subjectdata.stateto[i]
            
            # index for transition
            transind = tmat[subjectdata.statefrom[i], subjectdata.stateto[i]]

            # log hazard at time of transition
            haz_value = eval_hazard(
                hazards[transind],
                subjectdata.sojourn[i] + subjectdata.increment[i],
                pars[transind],
                row_data;
                apply_transform = hazards[transind].metadata.time_transform,
                cache_context = tt_context,
                hazard_slot = transind)
            
            ll += log(haz_value)
        end
     end
 
     # unweighted loglikelihood
     return ll
end

"""
    loglik(parameters, path::SamplePath, hazards::Vector{<:_Hazard}, model::MultistateProcess)

Convenience wrapper that evaluates the log-likelihood of a single sample path using the
current model definition. Accepts various parameter container types (see `prepare_parameters`)
and uses `_compute_path_loglik_fused` for efficient, correct evaluation (including AFT+TVC).

# Arguments
- `parameters`: Tuple, NamedTuple, or flat AbstractVector (will be unflattened internally)
- `path::SamplePath`: The sample path to evaluate
- `hazards::Vector{<:_Hazard}`: Hazard functions
- `model::MultistateProcess`: Model containing unflatten function and structure
"""
function loglik(parameters, path::SamplePath, hazards::Vector{<:_Hazard}, model::MultistateProcess)
    pars = prepare_parameters(parameters, model)

    # Build minimal cache for this subject
    subj_inds = model.subjectindices[path.subj]
    subj_dat = view(model.data, subj_inds, :)
    
    # Construct SubjectCovarCache locally
    # Note: This is slightly inefficient compared to reusing a global cache,
    # but this function is typically used in contexts where we don't have the global cache handy.
    # For MCEM inner loops, we might want to optimize this further, but correctness comes first.
    # Convert names to Symbol for proper comparison (names() returns Vector{String})
    covar_cols = setdiff(Symbol.(names(model.data)), [:id, :tstart, :tstop, :statefrom, :stateto, :obstype])
    has_covars = !isempty(covar_cols)
    
    if has_covars
        covar_data = subj_dat[:, covar_cols]
        tstart = collect(subj_dat.tstart)
    else
        covar_data = DataFrame()
        tstart = Float64[]
    end
    
    subj_cache = SubjectCovarCache(tstart, covar_data)
    
    # Covar names per hazard
    covar_names_per_hazard = [
        hasfield(typeof(hazards[h]), :covar_names) ? 
            hazards[h].covar_names : 
            extract_covar_names(hazards[h].parnames)
        for h in eachindex(hazards)
    ]
    
    # Time transform context
    any_time_transform = any(h -> h.metadata.time_transform, hazards)
    tt_context = if any_time_transform
        sample_df = isempty(covar_data) ? nothing : covar_data[1:1, :]
        maybe_time_transform_context(pars, sample_df, hazards)
    else
        nothing
    end
    
    # Use the fused path likelihood computation which correctly handles AFT+TVC
    return _compute_path_loglik_fused(
        path, pars, hazards, model.totalhazards, model.tmat,
        subj_cache, covar_names_per_hazard, tt_context, Float64
    )
end

########################################################
### Batched (Hazard-Centric) Likelihood for Exact Data #
########################################################

