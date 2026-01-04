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
    eval_hazard(hazard, t, pars, covars; apply_transform=false, cache_context=nothing, hazard_slot=nothing, use_effective_time=false)

Evaluate the hazard rate at time `t`. Returns the hazard on NATURAL scale (not log).

This is the primary interface for hazard evaluation. It handles:
- Direct evaluation via hazard_fn
- Time transform optimization (when apply_transform=true and hazard supports it)
- Caching for repeated evaluations with same linear predictor
- Effective time handling for AFT with time-varying covariates

# Arguments
- `hazard::_Hazard`: The hazard function struct
- `t::Real`: Time at which to evaluate hazard (effective time if `use_effective_time=true`)
- `pars::AbstractVector`: Parameters (log-scale for baseline, natural for covariates)
- `covars`: Covariate values - can be NamedTuple (cached) or DataFrameRow (direct view, zero-copy)
- `apply_transform::Bool=false`: Use time transform optimization if hazard supports it
- `cache_context::Union{Nothing,TimeTransformContext}=nothing`: Cache for repeated evaluations
- `hazard_slot::Union{Nothing,Int}=nothing`: Index of this hazard in the model
- `use_effective_time::Bool=false`: If true, `t` is treated as effective time (integrated history). 
  For AFT, this bypasses internal time scaling but applies the rate scaling factor.

# Returns
- `Float64`: Hazard rate on natural scale
"""
@inline function eval_hazard(hazard::_Hazard, t::Real, pars::Union{AbstractVector, NamedTuple}, covars::CovariateData;
                             apply_transform::Bool = false,
                             cache_context::Union{Nothing,TimeTransformContext}=nothing,
                             hazard_slot::Union{Nothing,Int}=nothing,
                             use_effective_time::Bool = false)
    use_transform = apply_transform && hazard.metadata.time_transform
    
    if !use_transform
        if use_effective_time && hazard.metadata.linpred_effect == :aft
            # For AFT with effective time τ, we need h₀(τ) × exp(-β'x).
            # The generated hazard_fn computes h(t|x) = h₀(t·e^{-β'x}) × e^{-β'x} = h₀(t) × t^(κ-1) × e^{-κβ'x}
            # which assumes clock time input. When we pass τ, this gives wrong result.
            # Instead: get h₀(τ) by passing zero covariates, then multiply by exp(-linpred).
            
            if isempty(hazard.covar_names)
                return hazard(t, pars, covars)
            else
                # Compute linear predictor before zeroing covariates
                linpred = _linear_predictor(pars, covars, hazard)
                
                # Zero covariates to get baseline hazard h₀(τ)
                zero_vals = zeros(Float64, length(hazard.covar_names))
                zero_covars = NamedTuple{Tuple(hazard.covar_names)}(zero_vals)
                
                # h(t|x) = h₀(τ) × exp(-β'x) for AFT with effective time
                return hazard(t, pars, zero_covars) * exp(-linpred)
            end
        else
            return hazard(t, pars, covars)
        end
    end

    _ensure_transform_supported(hazard)
    linpred = _linear_predictor(pars, covars, hazard)
    pars_vec = _time_transform_pars(pars)

    # If using effective time (AFT with TVC), we pass 0.0 as linpred to the time transform
    # to avoid double-scaling the time argument. However, for AFT, we must still apply
    # the rate scaling factor exp(-linpred).
    # For PH, effective time = clock time, so linpred is used as is (or 0 if we handled it outside? 
    # No, PH doesn't use effective time logic usually, but if it did, it would be same).
    
    # Actually, for AFT: h(t|x) = h0(tau) * exp(-beta*x).
    # If use_effective_time=true, t is tau. We want h0(t) * exp(-linpred).
    # _time_transform_hazard(..., t, 0.0) returns h0(t).
    # So we multiply by exp(-linpred).
    
    eff_linpred = (use_effective_time && hazard.metadata.linpred_effect == :aft) ? zero(linpred) : linpred

    val = if cache_context === nothing || hazard_slot === nothing
        _time_transform_hazard(hazard, pars_vec, t, eff_linpred)
    else
        cache = _shared_or_local_cache(cache_context, hazard_slot, hazard)
        key = _hazard_cache_key(cache, eff_linpred, t)
        get!(cache.hazard_values, key) do
            _time_transform_hazard(hazard, pars_vec, t, eff_linpred)
        end
    end
    
    if use_effective_time && hazard.metadata.linpred_effect == :aft
        return val * exp(-linpred)
    else
        return val
    end
end

"""
    eval_cumhaz(hazard, lb, ub, pars, covars; apply_transform=false, cache_context=nothing, hazard_slot=nothing, use_effective_time=false)

Evaluate the cumulative hazard over interval [lb, ub]. Returns on NATURAL scale (not log).

This is the primary interface for cumulative hazard evaluation.

# Arguments
- `hazard::_Hazard`: The hazard function struct
- `lb::Real`: Lower bound of interval (effective time if `use_effective_time=true`)
- `ub::Real`: Upper bound of interval (effective time if `use_effective_time=true`)
- `pars::AbstractVector`: Parameters (log-scale for baseline, natural for covariates)
- `covars`: Covariate values - can be NamedTuple (cached) or DataFrameRow (direct view, zero-copy)
- `apply_transform::Bool=false`: Use time transform optimization if hazard supports it
- `cache_context::Union{Nothing,TimeTransformContext}=nothing`: Cache for repeated evaluations
- `hazard_slot::Union{Nothing,Int}=nothing`: Index of this hazard in the model
- `use_effective_time::Bool=false`: If true, `lb` and `ub` are treated as effective times.
  For AFT, this bypasses internal time scaling.

# Returns
- `Float64`: Cumulative hazard on natural scale
"""
@inline function eval_cumhaz(hazard::_Hazard, lb::Real, ub::Real, pars::Union{AbstractVector, NamedTuple}, covars::CovariateData;
                             apply_transform::Bool = false,
                             cache_context::Union{Nothing,TimeTransformContext}=nothing,
                             hazard_slot::Union{Nothing,Int}=nothing,
                             use_effective_time::Bool = false)
    use_transform = apply_transform && hazard.metadata.time_transform
    
    if !use_transform
        if use_effective_time && hazard.metadata.linpred_effect == :aft
            # For AFT with effective time, we need to evaluate the baseline cumulative hazard H0(lb, ub).
            # The standard cumhaz_fn includes the covariate effect: H(t|x) = H0(t*exp(-beta*x)) * exp(-beta*x).
            # Since lb/ub are already effective times (tau = t*exp(-beta*x)), we want H0(tau).
            # We achieve this by passing zero covariates, which forces linear_pred = 0.
            # This results in H0(tau) * exp(0) = H0(tau).
            
            if isempty(hazard.covar_names)
                return cumulative_hazard(hazard, lb, ub, pars, covars)
            else
                # Construct NamedTuple of zeros matching covariate names
                # This is necessary because cumhaz_fn uses property access (covars.name)
                zero_vals = zeros(Float64, length(hazard.covar_names))
                zero_covars = NamedTuple{Tuple(hazard.covar_names)}(zero_vals)
                return cumulative_hazard(hazard, lb, ub, pars, zero_covars)
            end
        else
            return cumulative_hazard(hazard, lb, ub, pars, covars)
        end
    end

    _ensure_transform_supported(hazard)
    linpred = _linear_predictor(pars, covars, hazard)
    pars_vec = _time_transform_pars(pars)

    # If using effective time, we pass 0.0 as linpred to avoid scaling lb/ub again.
    # For AFT: H(t) = H0(tau). We want H0(ub) - H0(lb).
    # _time_transform_cumhaz(..., lb, ub, 0.0) returns H0(ub) - H0(lb).
    
    eff_linpred = (use_effective_time && hazard.metadata.linpred_effect == :aft) ? zero(linpred) : linpred

    if cache_context === nothing || hazard_slot === nothing
        return _time_transform_cumhaz(hazard, pars_vec, lb, ub, eff_linpred)
    end

    cache = _shared_or_local_cache(cache_context, hazard_slot, hazard)
    key = _cumul_cache_key(cache, eff_linpred, lb, ub)
    return get!(cache.cumulhaz_values, key) do
        _time_transform_cumhaz(hazard, pars_vec, lb, ub, eff_linpred)
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
