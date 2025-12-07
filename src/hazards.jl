#=============================================================================
PHASE 2: Runtime-Generated Hazard Functions

PARAMETER SCALE CONVENTION (Phase 2.5):
- All baseline/shape/scale parameters are stored and passed on LOG SCALE
- This matches the model.parameters storage convention (legacy compatibility)
- Covariate coefficients (β) remain on NATURAL SCALE
- Functions internally apply exp() transformations to return natural scale hazards
- ParameterHandling.jl integration (Phase 3) will handle transformations explicitly

Example:
  model.parameters = [[0.8], [log(2.5), log(1.5)]]  # Log scale storage
  hazard_fn(t, [0.8], []) returns exp(0.8) = 2.225...  # Natural scale output
=============================================================================# 

"""
    extract_covar_names(parnames::Vector{Symbol})

Extract covariate names from parameter names by removing hazard prefix and 
excluding baseline parameters (Intercept, shape, scale, and spline basis sp1, sp2, ...).

# Example
```julia
parnames = [:h12_Intercept, :h12_age, :h12_trt]
extract_covar_names(parnames)  # Returns [:age, :trt]

parnames_wei = [:h12_shape, :h12_scale, :h12_age]
extract_covar_names(parnames_wei)  # Returns [:age]

parnames_spline = [:h12_sp1, :h12_sp2, :h12_sp3, :h12_age]
extract_covar_names(parnames_spline)  # Returns [:age]
```
"""
function extract_covar_names(parnames::Vector{Symbol})
    covar_names = Symbol[]
    for pname in parnames
        pname_str = String(pname)
        # Skip baseline parameters (not covariates)
        # Exponential: "Intercept", Weibull/Gompertz: "shape" and "scale"
        # Spline: "sp1", "sp2", etc. (spline basis coefficients)
        if occursin("Intercept", pname_str) || occursin("shape", pname_str) || occursin("scale", pname_str)
            continue
        end
        # Remove hazard prefix (e.g., "h12_age" -> "age")
        covar_name = replace(pname_str, r"^h\d+_" => "")
        # Skip spline basis parameters (sp1, sp2, ...)
        if occursin(r"^sp\d+$", covar_name)
            continue
        end
        push!(covar_names, Symbol(covar_name))
    end
    return covar_names
end

"""
    extract_covariates(subjdat::DataFrameRow, parnames::Vector{Symbol})

Extract covariates from a DataFrame row as a NamedTuple, using parameter names to determine which columns to extract.

# Arguments
- `subjdat`: A DataFrameRow containing covariate values
- `parnames`: Vector of parameter names (e.g., [:h12_Intercept, :h12_age, :h12_trt])

# Returns
- Empty NamedTuple() if no covariates
- NamedTuple with covariate values otherwise (e.g., (age=50, trt=1))

# Example
```julia
row = DataFrame(id=1, tstart=0.0, tstop=10.0, statefrom=1, stateto=2, obstype=1, age=50, trt=1)[1, :]
parnames = [:h12_Intercept, :h12_age, :h12_trt]
extract_covariates(row, parnames)  # Returns (age=50, trt=1)
```
"""
function _lookup_covariate_value(subjdat::Union{DataFrameRow,DataFrame}, cname::Symbol)
    if hasproperty(subjdat, cname)
        return subjdat[cname]
    end

    cname_str = String(cname)
    if occursin("&", cname_str)
        parts = split(cname_str, "&")
        vals = (_lookup_covariate_value(subjdat, Symbol(strip(part))) for part in parts)
        return prod(vals)
    elseif occursin(":", cname_str)
        parts = split(cname_str, ":")
        vals = (_lookup_covariate_value(subjdat, Symbol(strip(part))) for part in parts)
        return prod(vals)
    else
        throw(ArgumentError("Covariate $(cname) is not present in the data row."))
    end
end

function extract_covariates(subjdat::Union{DataFrameRow,DataFrame}, parnames::Vector{Symbol})
    covar_names = extract_covar_names(parnames)
    
    if isempty(covar_names)
        return NamedTuple()
    end
    
    # Extract values from subjdat
    # Handle both DataFrameRow and DataFrame (for single-row DataFrame)
    if subjdat isa DataFrame
        @assert nrow(subjdat) == 1 "DataFrame must have exactly one row"
        subjdat_row = subjdat[1, :]
    else
        subjdat_row = subjdat
    end
    
    values = Tuple(_lookup_covariate_value(subjdat_row, cname) for cname in covar_names)
    return NamedTuple{Tuple(covar_names)}(values)
end

"""
    extract_covariates_fast(subjdat::DataFrameRow, covar_names::Vector{Symbol})

Extract covariate values into a NamedTuple for caching in hot loops.

For single evaluations, prefer passing the DataFrameRow directly to `eval_hazard`/`eval_cumhaz`
which avoids allocation. Use this function only when you need to cache covariate values
for repeated evaluation (e.g., in likelihood computation across many intervals).

# Arguments
- `subjdat::DataFrameRow`: current observation row
- `covar_names::Vector{Symbol}`: pre-extracted covariate names from hazard.covar_names

# Returns
- `NamedTuple`: covariate values keyed by name (allocates)

# Performance Note
- Single call: Pass DataFrameRow directly to `eval_hazard(haz, t, pars, row)` (zero-copy)
- Repeated calls: Cache with `covars = extract_covariates_fast(row, haz.covar_names)` then reuse
"""
@inline function extract_covariates_fast(subjdat::DataFrameRow, covar_names::Vector{Symbol})
    isempty(covar_names) && return NamedTuple()
    values = Tuple(_lookup_covariate_value(subjdat, cname) for cname in covar_names)
    return NamedTuple{Tuple(covar_names)}(values)
end

# Covariate data types: NamedTuple for cached values, DataFrameRow for direct access
const CovariateData = Union{NamedTuple, DataFrameRow}

@inline _covariate_entry(covars_cache::AbstractVector{<:NamedTuple}, hazard_slot::Int) = covars_cache[hazard_slot]
@inline _covariate_entry(covars_cache::DataFrameRow, ::Int) = covars_cache
@inline _covariate_entry(covars_cache, ::Int) = covars_cache

@inline function _linear_predictor(pars::AbstractVector, covars::CovariateData, hazard::_Hazard)
    hazard.has_covariates || return zero(eltype(pars))
    covar_names = hazard.covar_names  # Use pre-cached covar_names
    offset = hazard.npar_baseline
    linpred = zero(eltype(pars))
    for (i, cname) in enumerate(covar_names)
        val = getproperty(covars, cname)
        coeff = pars[offset + i]
        linpred += coeff * val
    end
    return linpred
end

@inline function _ensure_transform_supported(hazard::_Hazard)
    hazard.metadata.time_transform || return
    # Splines now support time transform
end

@inline function _time_transform_cache(context::TimeTransformContext{LinType,TimeType}, hazard_slot::Int) where {LinType,TimeType}
    cache = context.caches[hazard_slot]
    if cache === nothing
        cache = TimeTransformCache(LinType, TimeType)
        context.caches[hazard_slot] = cache
    end
    return cache
end

@inline function _shared_or_local_cache(context::TimeTransformContext{LinType,TimeType}, hazard_slot::Int, hazard::_Hazard) where {LinType,TimeType}
    key = hazard.shared_baseline_key
    if key === nothing
        return _time_transform_cache(context, hazard_slot)
    end

    return get!(context.shared_baselines.caches, key) do
        TimeTransformCache(LinType, TimeType)
    end
end

@inline function _hazard_cache_key(cache::TimeTransformCache{LinType,TimeType}, linpred, t) where {LinType,TimeType}
    return TimeTransformHazardKey{LinType,TimeType}(convert(LinType, linpred), convert(TimeType, t))
end

@inline function _cumul_cache_key(cache::TimeTransformCache{LinType,TimeType}, linpred, lb, ub) where {LinType,TimeType}
    return TimeTransformCumulKey{LinType,TimeType}(convert(LinType, linpred), convert(TimeType, lb), convert(TimeType, ub))
end

@inline function _time_transform_hazard(hazard::MarkovHazard, pars::AbstractVector, t::Real, linpred::Real)
    _ = t
    if hazard.metadata.linpred_effect == :aft
        return exp(pars[1] - linpred)
    else
        return exp(pars[1] + linpred)
    end
end

@inline function _time_transform_cumhaz(hazard::MarkovHazard, pars::AbstractVector, lb::Real, ub::Real, linpred::Real)
    rate = _time_transform_hazard(hazard, pars, ub, linpred)
    return rate * (ub - lb)
end

@inline function _time_transform_hazard_weibull(pars::AbstractVector, linpred::Real, effect::Symbol, t::Real)
    log_shape, log_scale = pars[1], pars[2]
    shape = exp(log_shape)
    scale = exp(log_scale)
    base = log(scale) + log(shape) + (shape - 1) * log(t)
    if effect == :aft
        return exp(base - shape * linpred)
    else
        return exp(base + linpred)
    end
end

@inline function _time_transform_cumhaz_weibull(pars::AbstractVector, linpred::Real, effect::Symbol, lb::Real, ub::Real)
    log_shape, log_scale = pars[1], pars[2]
    shape = exp(log_shape)
    scale = exp(log_scale)
    base = scale * (ub^shape - lb^shape)
    if effect == :aft
        return base * exp(-shape * linpred)
    else
        return base * exp(linpred)
    end
end

@inline function _gompertz_baseline_cumhaz(shape::Float64, scale::Float64, lb::Real, ub::Real)
    if abs(shape) < 1e-10
        return scale * (ub - lb)
    else
        return scale * (exp(shape * ub) - exp(shape * lb))
    end
end

@inline function _time_transform_hazard_gompertz(pars::AbstractVector, linpred::Real, effect::Symbol, t::Real)
    log_shape, log_scale = pars[1], pars[2]
    shape = exp(log_shape)
    scale = exp(log_scale)
    if effect == :aft
        time_scale = exp(-linpred)
        scaled_shape = shape * time_scale
        return exp(log(scale) + log(shape) + log(time_scale) + scaled_shape * t)
    else
        return exp(log_scale + log_shape + shape * t + linpred)
    end
end

@inline function _time_transform_cumhaz_gompertz(pars::AbstractVector, linpred::Real, effect::Symbol, lb::Real, ub::Real)
    log_shape, log_scale = pars[1], pars[2]
    shape = exp(log_shape)
    scale = exp(log_scale)
    if effect == :aft
        time_scale = exp(-linpred)
        scaled_shape = shape * time_scale
        if abs(scaled_shape) < 1e-10
            return scale * time_scale * (ub - lb)
        else
            return scale * (exp(scaled_shape * ub) - exp(scaled_shape * lb))
        end
    else
        base = _gompertz_baseline_cumhaz(shape, scale, lb, ub)
        return base * exp(linpred)
    end
end

@inline function _time_transform_hazard(hazard::SemiMarkovHazard, pars::AbstractVector, t::Real, linpred::Real)
    family = hazard.family
    if family == "wei"
        return _time_transform_hazard_weibull(pars, linpred, hazard.metadata.linpred_effect, t)
    elseif family == "gom"
        return _time_transform_hazard_gompertz(pars, linpred, hazard.metadata.linpred_effect, t)
    else
        throw(ArgumentError("time_transform=true is not implemented for family $(family)"))
    end
end

@inline function _time_transform_cumhaz(hazard::SemiMarkovHazard, pars::AbstractVector, lb::Real, ub::Real, linpred::Real)
    family = hazard.family
    if family == "wei"
        return _time_transform_cumhaz_weibull(pars, linpred, hazard.metadata.linpred_effect, lb, ub)
    elseif family == "gom"
        return _time_transform_cumhaz_gompertz(pars, linpred, hazard.metadata.linpred_effect, lb, ub)
    else
        throw(ArgumentError("time_transform=true is not implemented for family $(family)"))
    end
end

# Spline hazard time transform methods
# For splines, the baseline hazard α(t) is modeled by the spline, so we evaluate
# the stored hazard_fn and cumhaz_fn closures directly.

@inline function _time_transform_hazard(hazard::RuntimeSplineHazard, pars::AbstractVector, t::Real, linpred::Real)
    # The hazard_fn closure handles baseline spline evaluation
    # Linear predictor effect applied based on linpred_effect mode
    base_haz = hazard.hazard_fn(t, pars, NamedTuple())
    
    effect = hazard.metadata.linpred_effect
    if effect == :aft
        # AFT: h(t) * exp(-linpred) - time is scaled, hazard adjusted
        return base_haz * exp(-linpred)
    else
        # PH: h(t) * exp(linpred)
        return base_haz * exp(linpred)
    end
end

@inline function _time_transform_cumhaz(hazard::RuntimeSplineHazard, pars::AbstractVector, lb::Real, ub::Real, linpred::Real)
    # The cumhaz_fn closure handles baseline spline cumulative hazard
    base_cumhaz = hazard.cumhaz_fn(lb, ub, pars, NamedTuple())
    
    effect = hazard.metadata.linpred_effect
    if effect == :aft
        # AFT: H(t) * exp(-linpred)
        return base_cumhaz * exp(-linpred)
    else
        # PH: H(t) * exp(linpred)
        return base_cumhaz * exp(linpred)
    end
end

"""
    _build_linear_pred_expr(parnames, first_covar_index)

Construct an expression that evaluates the linear predictor `β'X` inside a
runtime-generated hazard function. Falls back to zero when no covariates are
present so analytic hazards without covariates keep working without special-casing.
"""
function _build_linear_pred_expr(parnames::Vector{Symbol}, first_covar_index::Int)
    covar_names = extract_covar_names(parnames)
    if isempty(covar_names)
        return :(zero(eltype(pars)))
    end

    terms = Any[]
    for (i, cname) in enumerate(covar_names)
        idx = first_covar_index + i - 1
        push!(terms, :(pars[$idx] * covars.$(cname)))
    end

    return length(terms) == 1 ? terms[1] : Expr(:call, :+, terms...)
end

"""
    generate_exponential_hazard(parnames::Vector{Symbol}, linpred_effect::Symbol)

Generate runtime functions for exponential hazards with optional PH/AFT covariate
effects controlled by `linpred_effect`.
"""
function generate_exponential_hazard(parnames::Vector{Symbol}, linpred_effect::Symbol)
    linear_pred_expr = _build_linear_pred_expr(parnames, 2)
    if linpred_effect == :ph
        hazard_fn = @RuntimeGeneratedFunction(:(
            function(t, pars, covars)
                linear_pred = $linear_pred_expr
                return exp(pars[1] + linear_pred)
            end
        ))

        cumhaz_fn = @RuntimeGeneratedFunction(:(
            function(lb, ub, pars, covars)
                linear_pred = $linear_pred_expr
                return exp(pars[1] + linear_pred) * (ub - lb)
            end
        ))
    elseif linpred_effect == :aft
        hazard_fn = @RuntimeGeneratedFunction(:(
            function(t, pars, covars)
                linear_pred = $linear_pred_expr
                return exp(pars[1] - linear_pred)
            end
        ))

        cumhaz_fn = @RuntimeGeneratedFunction(:(
            function(lb, ub, pars, covars)
                linear_pred = $linear_pred_expr
                return exp(pars[1] - linear_pred) * (ub - lb)
            end
        ))
    else
        error("Unsupported linpred_effect $(linpred_effect) for exponential hazard")
    end

    return hazard_fn, cumhaz_fn
end

"""
    generate_weibull_hazard(parnames::Vector{Symbol}, linpred_effect::Symbol)

Generate runtime functions for Weibull hazards, supporting PH or AFT covariate
effects.
"""
function generate_weibull_hazard(parnames::Vector{Symbol}, linpred_effect::Symbol)
    linear_pred_expr = _build_linear_pred_expr(parnames, 3)
    if linpred_effect == :ph
        hazard_fn = @RuntimeGeneratedFunction(:(
            function(t, pars, covars)
                log_shape, log_scale = pars[1], pars[2]
                shape = exp(log_shape)
                linear_pred = $linear_pred_expr
                
                log_haz = log_scale + log_shape + linear_pred
                if shape != 1.0
                    log_haz += (shape - 1) * log(t)
                end
                
                return exp(log_haz)
            end
        ))

        cumhaz_fn = @RuntimeGeneratedFunction(:(
            function(lb, ub, pars, covars)
                log_shape, log_scale = pars[1], pars[2]
                shape = exp(log_shape)
                scale = exp(log_scale)
                linear_pred = $linear_pred_expr
                return scale * exp(linear_pred) * (ub^shape - lb^shape)
            end
        ))
    elseif linpred_effect == :aft
        hazard_fn = @RuntimeGeneratedFunction(:(
            function(t, pars, covars)
                log_shape, log_scale = pars[1], pars[2]
                shape = exp(log_shape)
                linear_pred = $linear_pred_expr
                
                log_haz = log_scale + log_shape - shape * linear_pred
                if shape != 1.0
                    log_haz += (shape - 1) * log(t)
                end
                
                return exp(log_haz)
            end
        ))

        cumhaz_fn = @RuntimeGeneratedFunction(:(
            function(lb, ub, pars, covars)
                log_shape, log_scale = pars[1], pars[2]
                shape = exp(log_shape)
                scale = exp(log_scale)
                linear_pred = $linear_pred_expr
                return scale * exp(-shape * linear_pred) * (ub^shape - lb^shape)
            end
        ))
    else
        error("Unsupported linpred_effect $(linpred_effect) for Weibull hazard")
    end

    return hazard_fn, cumhaz_fn
end

"""
    generate_gompertz_hazard(parnames::Vector{Symbol}, linpred_effect::Symbol)

Generate runtime functions for Gompertz hazards with PH/AFT covariate handling.
"""
function generate_gompertz_hazard(parnames::Vector{Symbol}, linpred_effect::Symbol)
    linear_pred_expr = _build_linear_pred_expr(parnames, 3)
    if linpred_effect == :ph
        hazard_fn = @RuntimeGeneratedFunction(:(
            function(t, pars, covars)
                log_shape, log_scale = pars[1], pars[2]
                shape = exp(log_shape)
                linear_pred = $linear_pred_expr
                return exp(log_scale + log_shape + shape * t + linear_pred)
            end
        ))

        cumhaz_fn = @RuntimeGeneratedFunction(:(
            function(lb, ub, pars, covars)
                log_shape, log_scale = pars[1], pars[2]
                shape = exp(log_shape)
                scale = exp(log_scale)
                if abs(shape) < 1e-10
                    baseline_cumhaz = scale * (ub - lb)
                else
                    baseline_cumhaz = scale * (exp(shape * ub) - exp(shape * lb))
                end
                linear_pred = $linear_pred_expr
                return baseline_cumhaz * exp(linear_pred)
            end
        ))
    elseif linpred_effect == :aft
        hazard_fn = @RuntimeGeneratedFunction(:(
            function(t, pars, covars)
                log_shape, log_scale = pars[1], pars[2]
                shape = exp(log_shape)
                linear_pred = $linear_pred_expr
                time_scale = exp(-linear_pred)
                return exp(log_scale + log_shape + shape * (t * time_scale) - linear_pred)
            end
        ))

        cumhaz_fn = @RuntimeGeneratedFunction(:(
            function(lb, ub, pars, covars)
                log_shape, log_scale = pars[1], pars[2]
                shape = exp(log_shape)
                scale = exp(log_scale)
                linear_pred = $linear_pred_expr
                time_scale = exp(-linear_pred)
                scaled_shape = shape * time_scale
                if abs(scaled_shape) < 1e-10
                    baseline_cumhaz = scale * time_scale * (ub - lb)
                else
                    baseline_cumhaz = scale * (exp(scaled_shape * ub) - exp(scaled_shape * lb))
                end
                return baseline_cumhaz
            end
        ))
    else
        error("Unsupported linpred_effect $(linpred_effect) for Gompertz hazard")
    end

    return hazard_fn, cumhaz_fn
end

#=============================================================================
PHASE 2: Callable Hazard Interface
=============================================================================# 

"""
    (hazard::MarkovHazard)(t, pars, covars=Float64[])

Make MarkovHazard directly callable for hazard evaluation.
Returns hazard rate at time t (time parameter ignored for Markov processes).

Covariates can be passed as NamedTuple (cached) or DataFrameRow (direct view).
"""
function (hazard::MarkovHazard)(t::Real, pars::AbstractVector, covars::Union{AbstractVector,CovariateData}=Float64[])
    return hazard.hazard_fn(t, pars, covars)
end

"""
    (hazard::SemiMarkovHazard)(t, pars, covars=Float64[])

Make SemiMarkovHazard directly callable for hazard evaluation.
Returns hazard rate at time t.

Covariates can be passed as NamedTuple (cached) or DataFrameRow (direct view).
"""
function (hazard::SemiMarkovHazard)(t::Real, pars::AbstractVector, covars::Union{AbstractVector,CovariateData}=Float64[])
    return hazard.hazard_fn(t, pars, covars)
end

"""
    (hazard::_SplineHazard)(t, pars, covars=Float64[])

Make spline hazards (RuntimeSplineHazard) directly callable for hazard evaluation.
Returns hazard rate at time t.

Covariates can be passed as NamedTuple (cached) or DataFrameRow (direct view).
"""
function (hazard::_SplineHazard)(t::Real, pars::AbstractVector, covars::Union{AbstractVector,CovariateData}=Float64[])
    return hazard.hazard_fn(t, pars, covars)
end

"""
    cumulative_hazard(hazard::Union{MarkovHazard,SemiMarkovHazard,_SplineHazard}, lb, ub, pars, covars=Float64[])

Compute cumulative hazard over interval [lb, ub].

Covariates can be passed as NamedTuple (cached) or DataFrameRow (direct view).
"""
function cumulative_hazard(hazard::Union{MarkovHazard,SemiMarkovHazard,_SplineHazard}, 
                          lb::Real, ub::Real, 
                          pars::AbstractVector, 
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
@inline function eval_hazard(hazard::_Hazard, t::Real, pars::AbstractVector, covars::CovariateData;
                             apply_transform::Bool = false,
                             cache_context::Union{Nothing,TimeTransformContext}=nothing,
                             hazard_slot::Union{Nothing,Int}=nothing)
    use_transform = apply_transform && hazard.metadata.time_transform
    use_transform || return hazard(t, pars, covars)

    _ensure_transform_supported(hazard)
    linpred = _linear_predictor(pars, covars, hazard)

    if cache_context === nothing || hazard_slot === nothing
        return _time_transform_hazard(hazard, pars, t, linpred)
    end

    cache = _shared_or_local_cache(cache_context, hazard_slot, hazard)
    key = _hazard_cache_key(cache, linpred, t)
    return get!(cache.hazard_values, key) do
        _time_transform_hazard(hazard, pars, t, linpred)
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
@inline function eval_cumhaz(hazard::_Hazard, lb::Real, ub::Real, pars::AbstractVector, covars::CovariateData;
                             apply_transform::Bool = false,
                             cache_context::Union{Nothing,TimeTransformContext}=nothing,
                             hazard_slot::Union{Nothing,Int}=nothing)
    use_transform = apply_transform && hazard.metadata.time_transform
    use_transform || return cumulative_hazard(hazard, lb, ub, pars, covars)

    _ensure_transform_supported(hazard)
    linpred = _linear_predictor(pars, covars, hazard)

    if cache_context === nothing || hazard_slot === nothing
        return _time_transform_cumhaz(hazard, pars, lb, ub, linpred)
    end

    cache = _shared_or_local_cache(cache_context, hazard_slot, hazard)
    key = _cumul_cache_key(cache, linpred, lb, ub)
    return get!(cache.cumulhaz_values, key) do
        _time_transform_cumhaz(hazard, pars, lb, ub, linpred)
    end
end

# Convenience methods that extract covariates from DataFrameRow
@inline function eval_hazard(hazard::_Hazard, t::Real, pars::AbstractVector, subjdat::DataFrameRow;
                             apply_transform::Bool = false,
                             cache_context::Union{Nothing,TimeTransformContext}=nothing,
                             hazard_slot::Union{Nothing,Int}=nothing)
    covars = extract_covariates_fast(subjdat, hazard.covar_names)
    return eval_hazard(hazard, t, pars, covars; 
                       apply_transform=apply_transform,
                       cache_context=cache_context, 
                       hazard_slot=hazard_slot)
end

@inline function eval_cumhaz(hazard::_Hazard, lb::Real, ub::Real, pars::AbstractVector, subjdat::DataFrameRow;
                             apply_transform::Bool = false,
                             cache_context::Union{Nothing,TimeTransformContext}=nothing,
                             hazard_slot::Union{Nothing,Int}=nothing)
    covars = extract_covariates_fast(subjdat, hazard.covar_names)
    return eval_cumhaz(hazard, lb, ub, pars, covars;
                       apply_transform=apply_transform,
                       cache_context=cache_context,
                       hazard_slot=hazard_slot)
end

# =============================================================================
# Total Hazard and Survival Probability Functions
# =============================================================================
#
# These functions compute survival probabilities and total cumulative hazards
# by dispatching on the _TotalHazard type. They are used by:

# - loglik_path: For sample path likelihood computation
# - loglik_markov: For Markov model panel data likelihood
# - simulation.jl: For cumulative incidence calculations
# - next_state_probs: For transition probability computation
#
# =============================================================================

"""
    survprob(lb, ub, parameters, subjdat_row, _totalhazard::_TotalHazardTransient, _hazards; give_log = true) 

Return the survival probability over the interval [lb, ub].
"""
function survprob(lb, ub, parameters, subjdat_row, _totalhazard::_TotalHazardTransient, _hazards;
                   give_log = true,
                   apply_transform::Bool = false,
                   cache_context::Union{Nothing,TimeTransformContext}=nothing) 

    # log total cumulative hazard
    log_survprob = -total_cumulhaz(
        lb,
        ub,
        parameters,
        subjdat_row,
        _totalhazard,
        _hazards;
        give_log = false,
        apply_transform = apply_transform,
        cache_context = cache_context)

    # return survival probability or not
    give_log ? log_survprob : exp(log_survprob)
end

"""
    total_cumulhaz(lb, ub, parameters, subjdat_row, _totalhazard::_TotalHazardTransient, _hazards; give_log = true) 

Return the log-total cumulative hazard out of a transient state over the interval [lb, ub].
"""
function total_cumulhaz(lb, ub, parameters, subjdat_row, _totalhazard::_TotalHazardTransient, _hazards;
                        give_log = true,
                        apply_transform::Bool = false,
                        cache_context::Union{Nothing,TimeTransformContext}=nothing) 

    # log total cumulative hazard
    tot_haz = 0.0

    for x in _totalhazard.components
        tot_haz += eval_cumhaz(
            _hazards[x],
            lb,
            ub,
            parameters[x],
            subjdat_row;
            apply_transform = apply_transform,
            cache_context = cache_context,
            hazard_slot = x)
    end
    
    # return the log, or not
    give_log ? log(tot_haz) : tot_haz
end

function total_cumulhaz(lb, ub, parameters, covars_cache::AbstractVector{<:NamedTuple}, _totalhazard::_TotalHazardTransient, _hazards;
                        give_log = true,
                        apply_transform::Bool = false,
                        cache_context::Union{Nothing,TimeTransformContext}=nothing)
    tot_haz = 0.0

    for x in _totalhazard.components
        covars = _covariate_entry(covars_cache, x)
        tot_haz += eval_cumhaz(
            _hazards[x],
            lb,
            ub,
            parameters[x],
            covars;
            apply_transform = apply_transform,
            cache_context = cache_context,
            hazard_slot = x)
    end

    give_log ? log(tot_haz) : tot_haz
end

"""
    total_cumulhaz(lb, ub, parameters, subjdat_row, _totalhazard::_TotalHazardAbsorbing, _hazards; give_log = true) 

Return zero log-total cumulative hazard over the interval [lb, ub] as the current state is absorbing.
"""
function total_cumulhaz(lb, ub, parameters, subjdat_row, _totalhazard::_TotalHazardAbsorbing, _hazards;
                        give_log = true,
                        apply_transform::Bool = false,
                        cache_context::Union{Nothing,TimeTransformContext}=nothing) 

    # return 0 cumulative hazard
    give_log ? -Inf : 0

end

function total_cumulhaz(lb, ub, parameters, covars_cache::AbstractVector{<:NamedTuple}, _totalhazard::_TotalHazardAbsorbing, _hazards;
                        give_log = true,
                        apply_transform::Bool = false,
                        cache_context::Union{Nothing,TimeTransformContext}=nothing)
    give_log ? -Inf : 0
end

"""
    survprob(lb, ub, parameters, subjdat_row, _totalhazard::_TotalHazardAbsorbing, _hazards; give_log = true)

Return survival probability = 1.0 (log = 0.0) for absorbing states, since no transitions can occur.
"""
function survprob(lb, ub, parameters, subjdat_row, _totalhazard::_TotalHazardAbsorbing, _hazards;
                  give_log = true,
                  apply_transform::Bool = false,
                  cache_context::Union{Nothing,TimeTransformContext}=nothing)
    give_log ? 0.0 : 1.0
end

function survprob(lb, ub, parameters, covars_cache::AbstractVector{<:NamedTuple}, _totalhazard::_TotalHazardTransient, _hazards;
                  give_log = true,
                  apply_transform::Bool = false,
                  cache_context::Union{Nothing,TimeTransformContext}=nothing)
    log_survprob = -total_cumulhaz(
        lb,
        ub,
        parameters,
        covars_cache,
        _totalhazard,
        _hazards;
        give_log = false,
        apply_transform = apply_transform,
        cache_context = cache_context)

    give_log ? log_survprob : exp(log_survprob)
end

"""
    next_state_probs(t, scur, subjdat_row, parameters, hazards, totalhazards, tmat;
                     apply_transform = false,
                     cache_context = nothing)

Return a vector `ns_probs` with probabilities of transitioning to each state based on hazards from the current state.

# Arguments 
- `t`: time at which hazards should be calculated
- `scur`: current state
- `subjdat_row`: DataFrame row containing subject covariates
- `parameters`: vector of vectors of model parameters
- `hazards`: vector of cause-specific hazards
- `totalhazards`: vector of total hazards
- `tmat`: transition matrix
- `apply_transform`: pass `true` to allow Tang-enabled hazards to reuse cached trajectories
- `cache_context`: optional `TimeTransformContext` shared across hazards
"""
function _next_state_probs!(ns_probs::AbstractVector{Float64}, trans_inds::AbstractVector{Int}, t, scur, covars_cache, parameters, hazards, totalhazards;
                           apply_transform::Bool,
                           cache_context::Union{Nothing,TimeTransformContext})
    fill!(ns_probs, 0.0)
    isempty(trans_inds) && return ns_probs

    if length(trans_inds) == 1
        ns_probs[trans_inds[1]] = 1.0
        return ns_probs
    end

    # Compute log-hazards for softmax
    vals = map(totalhazards[scur].components) do x
        covars = _covariate_entry(covars_cache, x)
        haz = eval_hazard(hazards[x], t, parameters[x], covars;
                          apply_transform = apply_transform && hazards[x].metadata.time_transform,
                          cache_context = cache_context,
                          hazard_slot = x)
        log(haz)  # softmax expects log scale
    end
    ns_probs[trans_inds] = softmax(vals)

    local_probs = view(ns_probs, trans_inds)
    if all(isnan.(local_probs))
        local_probs .= 1 / length(trans_inds)
    elseif any(isnan.(local_probs))
        pisnan = findall(isnan.(local_probs))
        if length(pisnan) == 1
            local_probs[pisnan] = 1 - sum(local_probs[Not(pisnan)])
        else
            local_probs[pisnan] .= (1 - sum(local_probs[Not(pisnan)])) / length(pisnan)
        end
    end

    return ns_probs
end

function next_state_probs!(ns_probs::AbstractVector{Float64}, trans_inds::AbstractVector{Int}, t, scur, subjdat_row, parameters, hazards, totalhazards;
                           apply_transform::Bool = false,
                           cache_context::Union{Nothing,TimeTransformContext}=nothing)
    return _next_state_probs!(ns_probs, trans_inds, t, scur, subjdat_row, parameters, hazards, totalhazards;
        apply_transform = apply_transform,
        cache_context = cache_context)
end

function next_state_probs(t, scur, subjdat_row, parameters, hazards, totalhazards, tmat;
                          apply_transform::Bool = false,
                          cache_context::Union{Nothing,TimeTransformContext}=nothing)
    ns_probs = zeros(size(tmat, 2))
    trans_inds = findall(tmat[scur,:] .!= 0.0)
    return next_state_probs!(ns_probs, trans_inds, t, scur, subjdat_row, parameters, hazards, totalhazards;
        apply_transform = apply_transform,
        cache_context = cache_context)
end

########################################################
############# multistate markov process ################
###### transition intensities and probabilities ########
########################################################

"""
    compute_hazmat!(Q, parameters, hazards::Vector{T}, tpm_index::DataFrame, model_data::DataFrame) where T <: _Hazard

Fill in a matrix of transition intensities for a multistate Markov model (in-place version).
"""
function compute_hazmat!(Q, parameters, hazards::Vector{T}, tpm_index::DataFrame, model_data::DataFrame) where T <: _Hazard

    # Get the DataFrameRow for covariate extraction - ONLY ONCE
    @inbounds subjdat_row = model_data[tpm_index.datind[1], :]
    
    # Pre-extract covariates for each hazard using cached covar_names
    # This avoids regex parsing in extract_covariates on every call
    @inbounds for h in eachindex(hazards) 
        hazard = hazards[h]
        # Use extract_covariates_fast with pre-cached covar_names
        covars = extract_covariates_fast(subjdat_row, hazard.covar_names)
        Q[hazard.statefrom, hazard.stateto] = 
            eval_hazard(hazard, tpm_index.tstart[1], parameters[h], covars)
    end

    # set diagonal elements equal to the sum of off-diags
    Q[diagind(Q)] = -sum(Q, dims = 2)
end

"""
    compute_hazmat(T, n_states, parameters, hazards, tpm_index, model_data)

Construct transition intensity matrix Q for a multistate Markov model (non-mutating version).

Returns a fresh matrix without modifying any pre-allocated storage.
Compatible with reverse-mode AD (Enzyme, Zygote).

# Arguments
- `T`: Element type (e.g., Float64 or Dual)
- `n_states::Int`: Number of states in the model
- `parameters`: Nested parameters (tuple of vectors)
- `hazards`: Vector of hazard objects
- `tpm_index`: DataFrame with time and data indices
- `model_data`: Model data DataFrame
"""
function compute_hazmat(::Type{T}, n_states::Int, parameters, hazards::Vector{<:_Hazard}, 
                        tpm_index::DataFrame, model_data::DataFrame) where T
    # Get covariate row once
    subjdat_row = model_data[tpm_index.datind[1], :]
    
    # Build Q matrix functionally using comprehension
    # Start with zeros, then set off-diagonals
    Q = zeros(T, n_states, n_states)
    
    for h in eachindex(hazards)
        hazard = hazards[h]
        covars = extract_covariates_fast(subjdat_row, hazard.covar_names)
        rate = eval_hazard(hazard, tpm_index.tstart[1], parameters[h], covars)
        Q = setindex_immutable(Q, rate, hazard.statefrom, hazard.stateto)
    end
    
    # Set diagonal: each element is negative sum of its row (excluding diagonal)
    for i in 1:n_states
        row_sum = zero(T)
        for j in 1:n_states
            if i != j
                row_sum += Q[i, j]
            end
        end
        Q = setindex_immutable(Q, -row_sum, i, i)
    end
    
    return Q
end

"""
    setindex_immutable(A, val, i, j)

Return a new matrix with A[i,j] = val without mutating A.
This is the key primitive for reverse-mode AD compatibility.
"""
@inline function setindex_immutable(A::AbstractMatrix{T}, val::T, i::Int, j::Int) where T
    # Create a copy and set the value
    B = copy(A)
    B[i, j] = val
    return B
end

# More efficient version using ntuple for small matrices (avoids copy overhead)
@inline function setindex_immutable(A::AbstractMatrix{T}, val, i::Int, j::Int) where T
    B = copy(A)
    B[i, j] = convert(T, val)
    return B
end

"""
    compute_tmat!(P, Q, tpm_index::DataFrame, cache)

Calculate transition probability matrices for a multistate Markov process (in-place version). 
"""
function compute_tmat!(P, Q, tpm_index::DataFrame, cache)

    @inbounds for t in eachindex(P)
        copyto!(P[t], exponential!(Q * tpm_index.tstop[t], ExpMethodGeneric(), cache))
    end  
end

"""
    compute_tmat(Q, dt)

Compute transition probability matrix P = exp(Q * dt) without mutation.
Compatible with reverse-mode AD.

# Arguments
- `Q`: Transition intensity matrix
- `dt`: Time interval

# Returns
- `P`: Transition probability matrix
"""
function compute_tmat(Q::AbstractMatrix{T}, dt::Real) where T
    # Use ExponentialUtilities.exp_generic for AD-compatible matrix exponential
    # exp_generic handles arbitrary element types (including Dual numbers for ForwardDiff)
    Qt = Q * dt
    return ExponentialUtilities.exp_generic(Qt)
end


"""
    cumulative_incidence(t, model::MultistateProcess, subj::Int64=1)

Compute the cumulative incidence for each possible transition as a function of time since state entry. Assumes the subject starts their observation period at risk and saves cumulative incidence at the supplied vector of times, t.
"""
function cumulative_incidence(t, model::MultistateProcess, subj::Int64=1)

    # grab parameters, hazards and total hazards
    parameters   = model.parameters
    hazards      = model.hazards
    totalhazards = model.totalhazards

    # subject data
    subj_inds = model.subjectindices[subj]
    subj_dat  = view(model.data, subj_inds, :)

    # merge times with left endpoints of subject observation intervals
    subj_times = sort(unique([0.0; t]))

    # identify transient states
    transients = findall(isa.(totalhazards, _TotalHazardTransient))

    # identify which transient state to grab for each hazard (as transients[trans_inds[h]])
    trans_inds  = reduce(vcat, [i * ones(Int64, length(totalhazards[transients[i]].components)) for i in eachindex(transients)])

    # initialize cumulative incidence
    n_intervals = length(subj_times) - 1
    incidences  = zeros(Float64, n_intervals, length(hazards))
    survprobs   = ones(Float64, n_intervals, length(transients))

    # indices for starting cumulative incidence increments
    interval_inds = map(x -> searchsortedlast(subj_dat.tstart .- minimum(subj_dat.tstart), x), subj_times[Not(end)])

    # compute the survival probabilities to start each interval
    if n_intervals > 1
        for s in eachindex(transients)
            # initialize sprob and identify origin state
            sprob = 1.0
            statefrom = transients[s]

            # compute survival probabilities
            for i in 2:n_intervals
                survprobs[i,s] = sprob * survprob(subj_times[i-1], subj_times[i], parameters, subj_inds[interval_inds[i-1]], totalhazards[statefrom], hazards; give_log = false)
                sprob = survprobs[i,s]
            end
        end
    end
    
    # compute the cumulative incidence for each transition type
    for h in eachindex(hazards)
        # identify origin state
        statefrom = transients[trans_inds[h]]

        # compute incidences
        for r in 1:n_intervals
            subjdat_row = subj_dat[interval_inds[r], :]
            covars = extract_covariates_fast(subjdat_row, hazards[h].covar_names)
            incidences[r,h] = 
                survprobs[r,trans_inds[h]] * 
                quadgk(t -> (
                        eval_hazard(hazards[h], t, parameters[h], covars) * 
                        survprob(subj_times[r], t, parameters, subjdat_row, totalhazards[statefrom], hazards; give_log = false)), 
                        subj_times[r], subj_times[r + 1])[1]
        end        
    end

    # return cumulative incidences
    return cumsum(incidences; dims = 1)
end

"""
    cumulative_incidence(t, model::MultistateProcess, statefrom, subj::Int64=1)

Compute the cumulative incidence for each possible transition originating in `statefrom` as a function of time since state entry. Assumes the subject starts their observation period at risk and saves cumulative incidence at the supplied vector of times since state entry. This function is used internally.
"""
function cumulative_incidence(t, model::MultistateProcess, parameters, statefrom, subj::Int64=1)

    # get hazards
    hazards = model.hazards

    # get total hazards
    totalhazards = model.totalhazards

    # return zero if starting from absorbing state
    if isa(totalhazards[statefrom], _TotalHazardAbsorbing)
        return zeros(length(t))
    end

    # subject data
    subj_inds = model.subjectindices[subj]
    subj_dat  = view(model.data, subj_inds, :)

    # merge times with left endpoints of subject observation intervals
    subj_times = sort(unique([0.0; t]))

    # initialize cumulative incidence
    n_intervals = length(subj_times) - 1
    hazinds     = totalhazards[statefrom].components
    incidences  = zeros(Float64, n_intervals, length(hazinds))
    survprobs   = ones(Float64, n_intervals)

    # indices for starting cumulative incidence increments
    interval_inds = map(x -> searchsortedlast(subj_dat.tstart .- minimum(subj_dat.tstart), x), subj_times[Not(end)])

    # compute the survival probabilities to start each interval
    if n_intervals > 1

        # initialize sprob
        sprob = 1.0

        # compute survival probabilities
        for i in 2:n_intervals
            survprobs[i] = sprob * survprob(subj_times[i-1], subj_times[i], parameters, subj_inds[interval_inds[i-1]], totalhazards[statefrom], hazards; give_log = false)
            sprob = survprobs[i]
        end
    end
    
    # compute the cumulative incidence for each transition type
    for h in eachindex(hazinds)
        for r in 1:n_intervals
            subjdat_row = subj_dat[interval_inds[r], :]
            hazard = hazards[hazinds[h]]
            covars = extract_covariates_fast(subjdat_row, hazard.covar_names)
            incidences[r,h] = 
                survprobs[r] * 
                quadgk(t -> (
                        eval_hazard(hazard, t, parameters[hazinds[h]], covars) * 
                        survprob(subj_times[r], t, parameters, subjdat_row, totalhazards[statefrom], hazards; give_log = false)), 
                        subj_times[r], subj_times[r + 1])[1]
        end        
    end

    # return cumulative incidences
    return cumsum(incidences; dims = 1)
end

"""
    compute_hazard(t, model::MultistateProcess, hazard::Symbol)

Compute the hazard at times t. 

# Arguments
- t: time or vector of times. 
- model: MultistateProcess object. 
- hazard: Symbol specifying the hazard, e.g., :h12 for the hazard for transitioning from state 1 to state 2. 
- subj: subject id. 
"""
function compute_hazard(t, model::MultistateProcess, hazard::Symbol, subj::Int64 = 1)

    # get hazard index
    hazind = model.hazkeys[hazard]
    
    # get log-scale parameters for this hazard
    hazard_params = get_parameters(model, hazind, scale=:log)
    haz = model.hazards[hazind]

    # compute hazards
    hazards = zeros(Float64, length(t))
    for s in eachindex(t)
        # get row
        rowind = findlast((model.data.id .== subj) .& (model.data.tstart .<= t[s]))
        subjdat_row = model.data[rowind, :]
        covars = extract_covariates_fast(subjdat_row, haz.covar_names)

        # compute hazard
        hazards[s] = eval_hazard(haz, t[s], hazard_params, covars)
    end

    # return hazards
    return hazards
end

"""
    compute_cumulative_hazard(tstart, tstop, model::MultistateProcess, hazard::Symbol, subj::Int64=1)

Compute the cumulative hazard over [tstart,tstop]. 

# Arguments
- tstart: starting times
- tstop: stopping times
- model: MultistateProcess object. 
- hazard: Symbol specifying the hazard, e.g., :h12 for the hazard for transitioning from state 1 to state 2. 
- subj: subject id. 
"""
function compute_cumulative_hazard(tstart, tstop, model::MultistateProcess, hazard::Symbol, subj::Int64 = 1)

    # check bounds
    if (length(tstart) == length(tstop))
        # nothing to do
    elseif (length(tstart) == 1) & (length(tstop) != 1)
        tstart = rep(tstart, length(tstart))
    elseif (length(tstart) != 1) & (length(tstop) == 1)
        tstop = rep(tstop, length(tstart))
    else
        error("Lengths of tstart and tstop are not compatible.")
    end

    # get hazard index
    hazind = model.hazkeys[hazard]
    
    # get log-scale parameters for this hazard
    hazard_params = get_parameters(model, hazind, scale=:log)
    haz = model.hazards[hazind]

    # compute hazards
    cumulative_hazards = zeros(Float64, length(tstart))
    for s in eachindex(tstart)

        # find times between tstart and tstop
        times = [tstart[s]; model.data.tstart[findall((model.data.id .== subj) .& (model.data.tstart .> tstart[s]) .& (model.data.tstart .< tstop[s]))]; tstop[s]]

        # initialize cumulative hazard
        chaz = 0.0

        # accumulate
        for i in 1:(length(times) - 1)
            # get row
            rowind = findlast((model.data.id .== subj) .& (model.data.tstart .<= times[i]))
            subjdat_row = model.data[rowind, :]
            covars = extract_covariates_fast(subjdat_row, haz.covar_names)

            # compute cumulative hazard
            chaz += eval_cumhaz(haz, times[i], times[i+1], hazard_params, covars)
        end

        # save
        cumulative_hazards[s] = chaz
    end

    # return cumulative hazards
    return cumulative_hazards
end
