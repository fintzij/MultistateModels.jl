# =============================================================================
# Time Transform Functions
# =============================================================================
#
# Optimized hazard/cumhaz computation using time-transform factorization.
# For hazards h(t|x) = h₀(t) * g(β'x), we compute h₀(t) once and scale by g(β'x).
#
# =============================================================================

"""
    _ensure_transform_supported(hazard::_Hazard)

Verify that time transform optimization is supported for this hazard type.
Throws ArgumentError if hazard claims time_transform=true but type is unsupported.
"""
@inline function _ensure_transform_supported(hazard::_Hazard)
    if !(hazard isa Union{MarkovHazard, SemiMarkovHazard, RuntimeSplineHazard})
        throw(ArgumentError("Time transform not implemented for hazard type $(typeof(hazard)). Supported: MarkovHazard, SemiMarkovHazard, RuntimeSplineHazard"))
    end
end

"""
    _time_transform_cache(context::TimeTransformContext, hazard_slot::Int)

Get the cache for a specific hazard slot from the context.
Throws BoundsError if hazard_slot exceeds available caches.
"""
@inline function _time_transform_cache(context::TimeTransformContext{LinType,TimeType}, hazard_slot::Int) where {LinType,TimeType}
    if hazard_slot <= length(context.local_caches)
        return context.local_caches[hazard_slot]
    end
    throw(BoundsError(context.local_caches, hazard_slot))
end

"""
    _shared_or_local_cache(context::TimeTransformContext, hazard_slot::Int, hazard::_Hazard)

Get shared cache if hazard shares baseline with others, otherwise local cache.
"""
@inline function _shared_or_local_cache(context::TimeTransformContext{LinType,TimeType}, hazard_slot::Int, hazard::_Hazard) where {LinType,TimeType}
    shared_key = hazard.shared_baseline_key
    if isnothing(shared_key)
        return _time_transform_cache(context, hazard_slot)
    end
    return get!(context.shared_baselines.caches, shared_key) do
        TimeTransformCache(LinType, TimeType)
    end
end

"""
    _hazard_cache_key(cache, linpred, t)

Create cache key for hazard lookup.
"""
@inline function _hazard_cache_key(cache::TimeTransformCache{LinType,TimeType}, linpred, t) where {LinType,TimeType}
    TimeTransformHazardKey{LinType,TimeType}(convert(LinType, linpred), convert(TimeType, t))
end

"""
    _cumul_cache_key(cache, linpred, lb, ub)

Create cache key for cumulative hazard lookup.
"""
@inline function _cumul_cache_key(cache::TimeTransformCache{LinType,TimeType}, linpred, lb, ub) where {LinType,TimeType}
    TimeTransformCumulKey{LinType,TimeType}(convert(LinType, linpred), convert(TimeType, lb), convert(TimeType, ub))
end

# =============================================================================
# Markov Hazard Time Transforms
# =============================================================================

"""
    _time_transform_hazard(hazard::MarkovHazard, pars, t, linpred)

Compute hazard using time-transform factorization for Markov (exponential) hazard.
Formula: h(t|x) = rate * exp(β'x) for PH
"""
@inline function _time_transform_hazard(hazard::MarkovHazard, pars::AbstractVector, t::Real, linpred::Real)
    _ = t  # unused for exponential (time-homogeneous)
    # pars[1] is the natural-scale rate (already exp'd)
    rate = pars[1]
    if hazard.metadata.linpred_effect == :aft
        return rate * exp(-linpred)
    else
        return rate * exp(linpred)
    end
end

"""
    _time_transform_cumhaz(hazard::MarkovHazard, pars, lb, ub, linpred)

Compute cumulative hazard using time-transform for Markov hazard.
"""
@inline function _time_transform_cumhaz(hazard::MarkovHazard, pars::AbstractVector, lb::Real, ub::Real, linpred::Real)
    rate = _time_transform_hazard(hazard, pars, ub, linpred)
    return rate * (ub - lb)
end

# =============================================================================
# Weibull Time Transforms
# =============================================================================

"""
    _time_transform_hazard_weibull(pars, linpred, effect, t)

Weibull hazard with PH or AFT effect.
pars = [shape, scale] on NATURAL scale (already positive)
"""
@inline function _time_transform_hazard_weibull(pars::AbstractVector, linpred::Real, effect::Symbol, t::Real)
    # pars are on NATURAL scale (shape and scale already positive)
    shape, scale = pars[1], pars[2]
    # h(t) = shape * scale * t^(shape-1)
    haz = shape * scale
    if shape != 1.0
        haz *= t^(shape - 1)
    end
    if effect == :aft
        return haz * exp(-shape * linpred)
    else
        return haz * exp(linpred)
    end
end

"""
    _time_transform_cumhaz_weibull(pars, linpred, effect, lb, ub)

Weibull cumulative hazard with PH or AFT effect.
pars = [shape, scale] on NATURAL scale
"""
@inline function _time_transform_cumhaz_weibull(pars::AbstractVector, linpred::Real, effect::Symbol, lb::Real, ub::Real)
    # pars are on NATURAL scale
    shape, scale = pars[1], pars[2]
    base = scale * (ub^shape - lb^shape)
    if effect == :aft
        return base * exp(-shape * linpred)
    else
        return base * exp(linpred)
    end
end

# =============================================================================
# Gompertz Time Transforms
# =============================================================================

"""
    _gompertz_baseline_cumhaz(shape, rate, lb, ub)

Baseline Gompertz cumulative hazard: H₀(t) = (rate/shape) * (exp(shape*t) - 1)
Special case for shape ≈ 0: H₀(t) = rate * t (exponential)
"""
@inline function _gompertz_baseline_cumhaz(shape::Real, rate::Real, lb::Real, ub::Real)
    if abs(shape) < SHAPE_ZERO_TOL
        # Taylor expansion around shape = 0
        # (exp(shape*t) - 1)/shape = t + shape*t^2/2 + shape^2*t^3/6 + ...
        # H(lb, ub) = rate * [ (ub - lb) + shape/2 * (ub^2 - lb^2) + shape^2/6 * (ub^3 - lb^3) ]
        
        term1 = ub - lb
        term2 = (ub^2 - lb^2) / 2
        term3 = (ub^3 - lb^3) / 6
        
        return rate * (term1 + shape * term2 + shape^2 * term3)
    else
        # Gompertz: H(lb,ub) = (rate/shape) * (exp(shape*ub) - exp(shape*lb))
        return (rate / shape) * (exp(shape * ub) - exp(shape * lb))
    end
end

"""
    _time_transform_hazard_gompertz(pars, linpred, effect, t)

Gompertz hazard with PH or AFT effect.
pars = [shape, rate] on NATURAL scale (rate positive, shape unconstrained)
"""
@inline function _time_transform_hazard_gompertz(pars::AbstractVector, linpred::Real, effect::Symbol, t::Real)
    # pars are on NATURAL scale
    # pars[1] = shape (can be positive, negative, or zero)
    # pars[2] = rate (positive)
    shape, rate = pars[1], pars[2]
    if effect == :aft
        time_scale = exp(-linpred)
        scaled_shape = shape * time_scale
        return rate * exp(scaled_shape * t) * time_scale
    else
        return rate * exp(shape * t + linpred)
    end
end

"""
    _time_transform_cumhaz_gompertz(pars, linpred, effect, lb, ub)

Gompertz cumulative hazard with PH or AFT effect.
pars = [shape, rate] on NATURAL scale
"""
@inline function _time_transform_cumhaz_gompertz(pars::AbstractVector, linpred::Real, effect::Symbol, lb::Real, ub::Real)
    # pars are on NATURAL scale
    # flexsurv parameterization: H(t) = (rate/shape) * (exp(shape*t) - 1)
    shape, rate = pars[1], pars[2]
    if effect == :aft
        time_scale = exp(-linpred)
        scaled_shape = shape * time_scale
        scaled_rate = rate * time_scale
        return _gompertz_baseline_cumhaz(scaled_shape, scaled_rate, lb, ub)
    else
        base = _gompertz_baseline_cumhaz(shape, rate, lb, ub)
        return base * exp(linpred)
    end
end

# =============================================================================
# Semi-Markov Hazard Time Transform Dispatch
# =============================================================================

"""
    _time_transform_hazard(hazard::SemiMarkovHazard, pars, t, linpred)

Dispatch to appropriate time transform based on hazard family.
"""
@inline function _time_transform_hazard(hazard::SemiMarkovHazard, pars::AbstractVector, t::Real, linpred::Real)
    family = hazard.family
    effect = hazard.metadata.linpred_effect
    
    if family == :wei
        return _time_transform_hazard_weibull(pars, linpred, effect, t)
    elseif family == :gom
        return _time_transform_hazard_gompertz(pars, linpred, effect, t)
    else
        throw(ArgumentError("time_transform=true is not implemented for family $(family)"))
    end
end

"""
    _time_transform_cumhaz(hazard::SemiMarkovHazard, pars, lb, ub, linpred)

Dispatch to appropriate cumulative hazard time transform based on hazard family.
"""
@inline function _time_transform_cumhaz(hazard::SemiMarkovHazard, pars::AbstractVector, lb::Real, ub::Real, linpred::Real)
    family = hazard.family
    effect = hazard.metadata.linpred_effect
    
    if family == :wei
        return _time_transform_cumhaz_weibull(pars, linpred, effect, lb, ub)
    elseif family == :gom
        return _time_transform_cumhaz_gompertz(pars, linpred, effect, lb, ub)
    else
        throw(ArgumentError("time_transform=true is not implemented for family $(family)"))
    end
end

# =============================================================================
# Spline Hazard Time Transforms
# =============================================================================

"""
    _time_transform_hazard(hazard::RuntimeSplineHazard, pars, t, linpred)

Time transform for spline hazards.
For splines, h(t|x) = h₀(t) * exp(β'x) for PH.
"""
@inline function _time_transform_hazard(hazard::RuntimeSplineHazard, pars::AbstractVector, t::Real, linpred::Real)
    # The hazard_fn closure handles baseline spline evaluation
    # Linear predictor effect applied based on linpred_effect mode
    base_haz = hazard.hazard_fn(t, pars, NamedTuple())
    
    effect = hazard.metadata.linpred_effect
    if effect == :aft
        # AFT: h(t|x) = h₀(t * exp(-linpred)) * exp(-linpred)
        # Scale time by exp(-linpred) before evaluating baseline
        scale = exp(-linpred)
        base_haz = hazard.hazard_fn(t * scale, pars, NamedTuple())
        return base_haz * scale
    else
        # PH: h(t|x) = h₀(t) * exp(linpred)
        base_haz = hazard.hazard_fn(t, pars, NamedTuple())
        return base_haz * exp(linpred)
    end
end

"""
    _time_transform_cumhaz(hazard::RuntimeSplineHazard, pars, lb, ub, linpred)

Time transform for spline cumulative hazard.
"""
@inline function _time_transform_cumhaz(hazard::RuntimeSplineHazard, pars::AbstractVector, lb::Real, ub::Real, linpred::Real)
    # The cumhaz_fn closure handles baseline spline cumulative hazard
    
    effect = hazard.metadata.linpred_effect
    if effect == :aft
        # AFT: H(t|x) = H₀(t * exp(-linpred))
        # So H(ub|x) - H(lb|x) = H₀(ub * exp(-linpred)) - H₀(lb * exp(-linpred))
        scale = exp(-linpred)
        return hazard.cumhaz_fn(lb * scale, ub * scale, pars, NamedTuple())
    else
        # PH: H(t|x) = H₀(t) * exp(linpred)
        base_cumhaz = hazard.cumhaz_fn(lb, ub, pars, NamedTuple())
        return base_cumhaz * exp(linpred)
    end
end
