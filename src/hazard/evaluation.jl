# =============================================================================
# Hazard Evaluation Functions
# =============================================================================
#
# Callable interfaces for hazard structs and the unified eval_hazard/eval_cumhaz API.
#
# =============================================================================

#=============================================================================
Callable Hazard Interface
=============================================================================# 

"""
    (hazard::MarkovHazard)(t, pars, covars=Float64[])

Make MarkovHazard directly callable for hazard evaluation.
Returns hazard rate at time t (time parameter ignored for Markov processes).

Covariates can be passed as NamedTuple (cached) or DataFrameRow (direct view).
"""
function (hazard::MarkovHazard)(t::Real, pars::Union{AbstractVector, NamedTuple}, covars::Union{AbstractVector,CovariateData}=Float64[])
    return hazard.hazard_fn(t, pars, covars)
end

"""
    (hazard::SemiMarkovHazard)(t, pars, covars=Float64[])

Make SemiMarkovHazard directly callable for hazard evaluation.
Returns hazard rate at time t.

Covariates can be passed as NamedTuple (cached) or DataFrameRow (direct view).
"""
function (hazard::SemiMarkovHazard)(t::Real, pars::Union{AbstractVector, NamedTuple}, covars::Union{AbstractVector,CovariateData}=Float64[])
    return hazard.hazard_fn(t, pars, covars)
end

"""
    (hazard::_SplineHazard)(t, pars, covars=Float64[])

Make spline hazards (RuntimeSplineHazard) directly callable for hazard evaluation.
Returns hazard rate at time t.

Covariates can be passed as NamedTuple (cached) or DataFrameRow (direct view).
"""
function (hazard::_SplineHazard)(t::Real, pars::Union{AbstractVector, NamedTuple}, covars::Union{AbstractVector,CovariateData}=Float64[])
    return hazard.hazard_fn(t, pars, covars)
end

"""
    cumulative_hazard(hazard::Union{MarkovHazard,SemiMarkovHazard,_SplineHazard}, lb, ub, pars, covars=Float64[])

Compute cumulative hazard over interval [lb, ub].

Covariates can be passed as NamedTuple (cached) or DataFrameRow (direct view).
"""
function cumulative_hazard(hazard::Union{MarkovHazard,SemiMarkovHazard,_SplineHazard}, 
                          lb::Real, ub::Real, 
                          pars::Union{AbstractVector, NamedTuple}, 
                          covars::Union{AbstractVector,CovariateData}=Float64[])
    return hazard.cumhaz_fn(lb, ub, pars, covars)
end

#=============================================================================
Hazard Evaluation API
=============================================================================# 

"""
    eval_hazard(hazard, t, pars, covars; apply_transform=false, cache_context=nothing, hazard_slot=nothing)

Evaluate the hazard rate at time `t`. Returns the hazard on NATURAL scale (not log).

This is the primary interface for hazard evaluation. It handles:
- Direct evaluation via hazard_fn
- Time transform optimization (when apply_transform=true and hazard supports it)
- Caching for repeated evaluations with same linear predictor

# Arguments
- `hazard::_Hazard`: The hazard function struct
- `t::Real`: Time at which to evaluate hazard
- `pars::AbstractVector`: Parameters (log-scale for baseline, natural for covariates)
- `covars`: Covariate values - can be NamedTuple (cached) or DataFrameRow (direct view, zero-copy)
- `apply_transform::Bool=false`: Use time transform optimization if hazard supports it
- `cache_context::Union{Nothing,TimeTransformContext}=nothing`: Cache for repeated evaluations
- `hazard_slot::Union{Nothing,Int}=nothing`: Index of this hazard in the model

# Returns
- `Float64`: Hazard rate on natural scale
"""
@inline function eval_hazard(hazard::_Hazard, t::Real, pars::Union{AbstractVector, NamedTuple}, covars::CovariateData;
                             apply_transform::Bool = false,
                             cache_context::Union{Nothing,TimeTransformContext}=nothing,
                             hazard_slot::Union{Nothing,Int}=nothing)
    use_transform = apply_transform && hazard.metadata.time_transform
    use_transform || return hazard(t, pars, covars)

    _ensure_transform_supported(hazard)
    linpred = _linear_predictor(pars, covars, hazard)
    pars_vec = _time_transform_pars(pars)

    if cache_context === nothing || hazard_slot === nothing
        return _time_transform_hazard(hazard, pars_vec, t, linpred)
    end

    cache = _shared_or_local_cache(cache_context, hazard_slot, hazard)
    key = _hazard_cache_key(cache, linpred, t)
    return get!(cache.hazard_values, key) do
        _time_transform_hazard(hazard, pars_vec, t, linpred)
    end
end

"""
    eval_cumhaz(hazard, lb, ub, pars, covars; apply_transform=false, cache_context=nothing, hazard_slot=nothing)

Evaluate the cumulative hazard over interval [lb, ub]. Returns on NATURAL scale (not log).

This is the primary interface for cumulative hazard evaluation.

# Arguments
- `hazard::_Hazard`: The hazard function struct
- `lb::Real`: Lower bound of interval
- `ub::Real`: Upper bound of interval  
- `pars::AbstractVector`: Parameters (log-scale for baseline, natural for covariates)
- `covars`: Covariate values - can be NamedTuple (cached) or DataFrameRow (direct view, zero-copy)
- `apply_transform::Bool=false`: Use time transform optimization if hazard supports it
- `cache_context::Union{Nothing,TimeTransformContext}=nothing`: Cache for repeated evaluations
- `hazard_slot::Union{Nothing,Int}=nothing`: Index of this hazard in the model

# Returns
- `Float64`: Cumulative hazard on natural scale
"""
@inline function eval_cumhaz(hazard::_Hazard, lb::Real, ub::Real, pars::Union{AbstractVector, NamedTuple}, covars::CovariateData;
                             apply_transform::Bool = false,
                             cache_context::Union{Nothing,TimeTransformContext}=nothing,
                             hazard_slot::Union{Nothing,Int}=nothing)
    use_transform = apply_transform && hazard.metadata.time_transform
    use_transform || return cumulative_hazard(hazard, lb, ub, pars, covars)

    _ensure_transform_supported(hazard)
    linpred = _linear_predictor(pars, covars, hazard)
    pars_vec = _time_transform_pars(pars)

    if cache_context === nothing || hazard_slot === nothing
        return _time_transform_cumhaz(hazard, pars_vec, lb, ub, linpred)
    end

    cache = _shared_or_local_cache(cache_context, hazard_slot, hazard)
    key = _cumul_cache_key(cache, linpred, lb, ub)
    return get!(cache.cumulhaz_values, key) do
        _time_transform_cumhaz(hazard, pars_vec, lb, ub, linpred)
    end
end

# =============================================================================
# Convenience Methods (DataFrameRow covariate extraction)
# =============================================================================

"""
    eval_hazard(hazard, t, pars, subjdat::DataFrameRow; ...)

Convenience method that extracts covariates from DataFrameRow.
"""
@inline function eval_hazard(hazard::_Hazard, t::Real, pars::Union{AbstractVector, NamedTuple}, subjdat::DataFrameRow;
                             apply_transform::Bool = false,
                             cache_context::Union{Nothing,TimeTransformContext}=nothing,
                             hazard_slot::Union{Nothing,Int}=nothing)
    covars = extract_covariates_fast(subjdat, hazard.covar_names)
    return eval_hazard(hazard, t, pars, covars; 
                       apply_transform=apply_transform,
                       cache_context=cache_context, 
                       hazard_slot=hazard_slot)
end

"""
    eval_cumhaz(hazard, lb, ub, pars, subjdat::DataFrameRow; ...)

Convenience method that extracts covariates from DataFrameRow.
"""
@inline function eval_cumhaz(hazard::_Hazard, lb::Real, ub::Real, pars::Union{AbstractVector, NamedTuple}, subjdat::DataFrameRow;
                             apply_transform::Bool = false,
                             cache_context::Union{Nothing,TimeTransformContext}=nothing,
                             hazard_slot::Union{Nothing,Int}=nothing)
    covars = extract_covariates_fast(subjdat, hazard.covar_names)
    return eval_cumhaz(hazard, lb, ub, pars, covars;
                       apply_transform=apply_transform,
                       cache_context=cache_context,
                       hazard_slot=hazard_slot)
end
