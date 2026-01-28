# =============================================================================
# Spline Evaluation with External Cache
# =============================================================================
#
# This file provides spline evaluation functions that use the external 
# HazardEvalCache system instead of closure-internal Ref{Any} caches.
#
# These functions are loaded after spline_builder.jl so they can use:
# - _spline_ests2coefs
# - _eval_cumhaz_with_extrap  
# - _eval_linear_pred_named
#
# =============================================================================

"""
    get_or_build_spline!(cache::HazardEvalCache{SplineEvalState}, hazard::RuntimeSplineHazard, 
                         pars::AbstractVector)

Get cached spline objects or build them if needed.
Returns (spline_ext, cumhaz_spline) tuple.

For Float64 parameters: uses and updates cache
For Dual parameters (AD): always rebuilds, never caches

This function is the central point for spline caching, replacing the
distributed Ref{Any} caches in spline closures.
"""
function get_or_build_spline!(cache::HazardEvalCache{SplineEvalState}, hazard::RuntimeSplineHazard,
                              pars::AbstractVector{T}) where T
    state = cache.state
    
    # For AD mode, always rebuild - never use cache
    if is_ad_mode(T)
        return _build_spline_objects(hazard, pars)
    end
    
    # Check if cache is valid
    if state.spline_ext !== nothing && !needs_rebuild(cache, pars)
        return (state.spline_ext, state.cumhaz_spline)
    end
    
    # Cache miss or invalidated - rebuild
    spline_ext, cumhaz_spline = _build_spline_objects(hazard, pars)
    
    # Update cache
    state.spline_ext = spline_ext
    state.cumhaz_spline = cumhaz_spline
    update_hash!(cache, pars)
    
    return (spline_ext, cumhaz_spline)
end

"""
    _build_spline_objects(hazard::RuntimeSplineHazard, pars::AbstractVector)

Build SplineExtrapolation and cumulative hazard spline from parameters.
This is the core spline construction that happens on cache miss.

Returns (spline_ext, cumhaz_spline) tuple.
"""
function _build_spline_objects(hazard::RuntimeSplineHazard, pars::AbstractVector{T}) where T
    nbasis = hazard.npar_baseline
    
    # Extract spline coefficients (first nbasis parameters)
    spline_coefs_vec = pars[1:nbasis]
    
    # Build basis from hazard's stored knots
    B = BSplineBasis(BSplineOrder(hazard.degree + 1), copy(hazard.knots))
    
    # Apply boundary conditions if needed
    use_constant = hazard.extrapolation == "constant"
    if use_constant && (hazard.degree >= 2)
        B = RecombinedBSplineBasis(B, (), Derivative(1))
    elseif (hazard.degree > 1) && hazard.natural_spline
        B = RecombinedBSplineBasis(B, (), Derivative(2))
    end
    
    # Determine extrapolation method
    extrap_method = if hazard.extrapolation == "linear"
        BSplineKit.SplineExtrapolations.Linear()
    else
        BSplineKit.SplineExtrapolations.Flat()
    end
    
    # Transform parameters to spline coefficients
    coefs = _spline_ests2coefs(spline_coefs_vec, B, hazard.monotone)
    
    # Build spline and extrapolation
    spline = Spline(B, coefs)
    spline_ext = SplineExtrapolation(spline, extrap_method)
    
    # Build cumulative hazard spline (integral)
    cumhaz_spline = integral(spline_ext.spline)
    
    return (spline_ext, cumhaz_spline)
end

"""
    eval_spline_hazard_cached(hazard::RuntimeSplineHazard, t::Real, pars, covars, spline_ext)

Evaluate spline hazard using pre-built spline objects.
This bypasses the closure's internal caching.
"""
function eval_spline_hazard_cached(hazard::RuntimeSplineHazard, t::Real, 
                                   pars::Union{AbstractVector, NamedTuple}, 
                                   covars, spline_ext)
    nbasis = hazard.npar_baseline
    has_covars = hazard.has_covariates
    effect = hazard.metadata.linpred_effect
    
    # Extract covariate parameters
    if pars isa AbstractVector
        covar_pars = has_covars ? pars[(nbasis+1):end] : Float64[]
    else
        covar_pars = has_covars ? pars.covariates : NamedTuple()
    end
    
    # Compute linear predictor
    n_covars = covars isa AbstractVector ? length(covars) : length(covars)
    if has_covars && n_covars > 0
        linear_pred = pars isa AbstractVector ? 
                      dot(collect(covars), covar_pars) : 
                      _eval_linear_pred_named(covar_pars, covars)
    else
        linear_pred = 0.0
    end
    
    # Evaluate hazard with covariate effect
    if effect == :aft
        scale = exp(-linear_pred)
        h0 = spline_ext(t * scale)
        return h0 * scale
    else
        h0 = spline_ext(t)
        return h0 * exp(linear_pred)
    end
end

"""
    eval_spline_cumhaz_cached(hazard::RuntimeSplineHazard, lb::Real, ub::Real, 
                              pars, covars, spline_ext, cumhaz_spline)

Evaluate spline cumulative hazard using pre-built spline objects.
This bypasses the closure's internal caching.
"""
function eval_spline_cumhaz_cached(hazard::RuntimeSplineHazard, lb::Real, ub::Real,
                                   pars::Union{AbstractVector, NamedTuple},
                                   covars, spline_ext, cumhaz_spline)
    nbasis = hazard.npar_baseline
    has_covars = hazard.has_covariates
    effect = hazard.metadata.linpred_effect
    
    # Extract covariate parameters
    if pars isa AbstractVector
        covar_pars = has_covars ? pars[(nbasis+1):end] : Float64[]
    else
        covar_pars = has_covars ? pars.covariates : NamedTuple()
    end
    
    # Compute linear predictor
    n_covars = covars isa AbstractVector ? length(covars) : length(covars)
    if n_covars > 0 && has_covars
        linear_pred = pars isa AbstractVector ? 
                      dot(collect(covars), covar_pars) : 
                      _eval_linear_pred_named(covar_pars, covars)
    else
        linear_pred = 0.0
    end
    
    # Evaluate cumulative hazard with covariate effect
    if effect == :aft
        scale = exp(-linear_pred)
        H0 = _eval_cumhaz_with_extrap(spline_ext, cumhaz_spline, lb * scale, ub * scale)
        return H0
    else
        H0 = _eval_cumhaz_with_extrap(spline_ext, cumhaz_spline, lb, ub)
        return H0 * exp(linear_pred)
    end
end

# =============================================================================
# Dispatch Helpers for Cached vs Standard Evaluation
# =============================================================================
# These functions provide a unified interface that dispatches to cached 
# evaluation for spline hazards (when cache is available) and standard 
# eval_hazard/eval_cumhaz for other hazard types.

"""
    eval_hazard_maybe_cached(hazard, t, pars, covars, flat_pars, hazard_cache;
                             apply_transform, cache_context, hazard_slot, use_effective_time)

Evaluate hazard, using cached spline if available for RuntimeSplineHazard.
Falls back to standard eval_hazard for non-spline hazards.

# Arguments  
- `hazard`: The hazard struct
- `t`: Time point
- `pars`: Parameters as nested NamedTuple (from unflatten)
- `covars`: Covariate values
- `flat_pars`: Flat parameter vector (needed for spline cache)
- `hazard_cache`: HazardEvalCache for this hazard (or nothing)
- Other kwargs: passed to eval_hazard

For spline hazards with valid cache:
- Gets/builds spline from cache using `get_or_build_spline!`
- Calls `eval_spline_hazard_cached` with pre-built spline

For non-spline hazards or when cache is nothing:
- Falls back to standard `eval_hazard`
"""
@inline function eval_hazard_maybe_cached(
    hazard::RuntimeSplineHazard, t::Real, 
    pars::NamedTuple, covars, 
    flat_pars::AbstractVector{T},
    hazard_cache::Union{Nothing, HazardEvalCache};
    apply_transform::Bool = false,
    cache_context = nothing,
    hazard_slot::Union{Nothing,Int} = nothing,
    use_effective_time::Bool = false
) where T
    # For AD mode or no cache, fall back to standard path
    if is_ad_mode(T) || hazard_cache === nothing || !(hazard_cache.state isa SplineEvalState)
        return eval_hazard(hazard, t, pars, covars;
                          apply_transform=apply_transform,
                          cache_context=cache_context,
                          hazard_slot=hazard_slot,
                          use_effective_time=use_effective_time)
    end
    
    # Get or build cached spline objects (type-stable path)
    hazard_pars_vec = if haskey(pars, hazard.hazname)
        pars_for_haz = pars[hazard.hazname]
        collect(Float64, values(pars_for_haz.baseline))
    else
        flat_pars[1:hazard.npar_total]
    end
    
    spline_ext, cumhaz_spline = get_or_build_spline!(hazard_cache, hazard, hazard_pars_vec)
    
    # Handle effective time for AFT
    if use_effective_time && hazard.metadata.linpred_effect == :aft
        # For AFT with effective time, need special handling
        # Get baseline h₀(τ) and multiply by exp(-linpred)
        if hazard.has_covariates
            linpred = _linear_predictor(pars, covars, hazard)
            h0 = spline_ext(t)
            return h0 * exp(-linpred)
        else
            return spline_ext(t)
        end
    end
    
    # Use cached evaluation
    return eval_spline_hazard_cached(hazard, t, pars, covars, spline_ext)
end

# Fallback for non-spline hazards
@inline function eval_hazard_maybe_cached(
    hazard::_Hazard, t::Real,
    pars::NamedTuple, covars,
    flat_pars::AbstractVector,
    hazard_cache::Union{Nothing, HazardEvalCache};
    apply_transform::Bool = false,
    cache_context = nothing,
    hazard_slot::Union{Nothing,Int} = nothing,
    use_effective_time::Bool = false
)
    return eval_hazard(hazard, t, pars, covars;
                      apply_transform=apply_transform,
                      cache_context=cache_context,
                      hazard_slot=hazard_slot,
                      use_effective_time=use_effective_time)
end

"""
    eval_cumhaz_maybe_cached(hazard, lb, ub, pars, covars, flat_pars, hazard_cache;
                              apply_transform, cache_context, hazard_slot, use_effective_time)

Evaluate cumulative hazard, using cached spline if available for RuntimeSplineHazard.
Falls back to standard eval_cumhaz for non-spline hazards.
"""
@inline function eval_cumhaz_maybe_cached(
    hazard::RuntimeSplineHazard, lb::Real, ub::Real,
    pars::NamedTuple, covars,
    flat_pars::AbstractVector{T},
    hazard_cache::Union{Nothing, HazardEvalCache};
    apply_transform::Bool = false,
    cache_context = nothing,
    hazard_slot::Union{Nothing,Int} = nothing,
    use_effective_time::Bool = false
) where T
    # For AD mode or no cache, fall back to standard path
    if is_ad_mode(T) || hazard_cache === nothing || !(hazard_cache.state isa SplineEvalState)
        return eval_cumhaz(hazard, lb, ub, pars, covars;
                          apply_transform=apply_transform,
                          cache_context=cache_context,
                          hazard_slot=hazard_slot,
                          use_effective_time=use_effective_time)
    end
    
    # Get or build cached spline objects
    hazard_pars_vec = if haskey(pars, hazard.hazname)
        pars_for_haz = pars[hazard.hazname]
        collect(Float64, values(pars_for_haz.baseline))
    else
        flat_pars[1:hazard.npar_total]
    end
    
    spline_ext, cumhaz_spline = get_or_build_spline!(hazard_cache, hazard, hazard_pars_vec)
    
    # Handle effective time for AFT
    if use_effective_time && hazard.metadata.linpred_effect == :aft
        # For AFT with effective time, lb and ub are already effective times
        # Just evaluate baseline cumhaz H₀(lb, ub)
        return _eval_cumhaz_with_extrap(spline_ext, cumhaz_spline, lb, ub)
    end
    
    # Use cached evaluation
    return eval_spline_cumhaz_cached(hazard, lb, ub, pars, covars, spline_ext, cumhaz_spline)
end

# Fallback for non-spline hazards
@inline function eval_cumhaz_maybe_cached(
    hazard::_Hazard, lb::Real, ub::Real,
    pars::NamedTuple, covars,
    flat_pars::AbstractVector,
    hazard_cache::Union{Nothing, HazardEvalCache};
    apply_transform::Bool = false,
    cache_context = nothing,
    hazard_slot::Union{Nothing,Int} = nothing,
    use_effective_time::Bool = false
)
    return eval_cumhaz(hazard, lb, ub, pars, covars;
                      apply_transform=apply_transform,
                      cache_context=cache_context,
                      hazard_slot=hazard_slot,
                      use_effective_time=use_effective_time)
end
