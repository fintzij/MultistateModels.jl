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

Extract covariate names from parameter names by removing hazard prefix and excluding Intercept/shape/scale.

# Example
```julia
parnames = [:h12_Intercept, :h12_age, :h12_trt]
extract_covar_names(parnames)  # Returns [:age, :trt]

parnames_wei = [:h12_shape, :h12_scale, :h12_age]
extract_covar_names(parnames_wei)  # Returns [:age]
```
"""
function extract_covar_names(parnames::Vector{Symbol})
    covar_names = Symbol[]
    for pname in parnames
        pname_str = String(pname)
        # Skip baseline parameters (not covariates)
        # Exponential: "Intercept", Weibull/Gompertz: "shape" and "scale"
        if occursin("Intercept", pname_str) || occursin("shape", pname_str) || occursin("scale", pname_str)
            continue
        end
        # Remove hazard prefix (e.g., "h12_age" -> "age")
        covar_name = replace(pname_str, r"^h\d+_" => "")
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

Fast covariate extraction using pre-computed covariate names from hazard struct.
Avoids regex parsing of parameter names on every call.

# Arguments
- `subjdat::DataFrameRow`: current observation row
- `covar_names::Vector{Symbol}`: pre-extracted covariate names from hazard.covar_names

# Returns
- `NamedTuple`: covariate values keyed by name
"""
@inline function extract_covariates_fast(subjdat::DataFrameRow, covar_names::Vector{Symbol})
    isempty(covar_names) && return NamedTuple()
    values = Tuple(_lookup_covariate_value(subjdat, cname) for cname in covar_names)
    return NamedTuple{Tuple(covar_names)}(values)
end

@inline _covariate_entry(covars_cache::AbstractVector{<:NamedTuple}, hazard_slot::Int) = covars_cache[hazard_slot]
@inline _covariate_entry(covars_cache, ::Int) = covars_cache

@inline function _call_haz_with_covars(t, parameters, covars_cache, hazard::_Hazard, hazard_slot;
                                       give_log = true,
                                       apply_transform::Bool = false,
                                       cache_context::Union{Nothing,TimeTransformContext}=nothing)
    covars = _covariate_entry(covars_cache, hazard_slot)
    return call_haz(
        t,
        parameters,
        covars,
        hazard;
        give_log = give_log,
        apply_transform = apply_transform,
        cache_context = cache_context,
        hazard_slot = hazard_slot)
end

@inline function _call_cumulhaz_with_covars(lb, ub, parameters, covars_cache, hazard::_Hazard, hazard_slot;
                                            give_log = true,
                                            apply_transform::Bool = false,
                                            cache_context::Union{Nothing,TimeTransformContext}=nothing)
    covars = _covariate_entry(covars_cache, hazard_slot)
    return call_cumulhaz(
        lb,
        ub,
        parameters,
        covars,
        hazard;
        give_log = give_log,
        apply_transform = apply_transform,
        cache_context = cache_context,
        hazard_slot = hazard_slot)
end

@inline function _linear_predictor(pars::AbstractVector, covars::NamedTuple, hazard::_Hazard)
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
    if hazard isa SplineHazard
        throw(ArgumentError("time_transform=true is not yet supported for spline hazards"))
    end
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
    return TangHazardKey{LinType,TimeType}(convert(LinType, linpred), convert(TimeType, t))
end

@inline function _cumul_cache_key(cache::TimeTransformCache{LinType,TimeType}, linpred, lb, ub) where {LinType,TimeType}
    return TangCumulKey{LinType,TimeType}(convert(LinType, linpred), convert(TimeType, lb), convert(TimeType, ub))
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
                return exp(log_scale + log_shape + (shape - 1) * log(t) + linear_pred)
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
                return exp(log_scale + log_shape + (shape - 1) * log(t) - shape * linear_pred)
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
"""
function (hazard::MarkovHazard)(t::Real, pars::AbstractVector, covars::Union{AbstractVector,NamedTuple}=Float64[])
    return hazard.hazard_fn(t, pars, covars)
end

"""
    (hazard::SemiMarkovHazard)(t, pars, covars=Float64[])

Make SemiMarkovHazard directly callable for hazard evaluation.
Returns hazard rate at time t.
"""
function (hazard::SemiMarkovHazard)(t::Real, pars::AbstractVector, covars::Union{AbstractVector,NamedTuple}=Float64[])
    return hazard.hazard_fn(t, pars, covars)
end

"""
    (hazard::SplineHazard)(t, pars, covars=Float64[])

Make SplineHazard directly callable for hazard evaluation.
Returns hazard rate at time t.
"""
function (hazard::SplineHazard)(t::Real, pars::AbstractVector, covars::Union{AbstractVector,NamedTuple}=Float64[])
    return hazard.hazard_fn(t, pars, covars)
end

"""
    cumulative_hazard(hazard::Union{MarkovHazard,SemiMarkovHazard,SplineHazard}, lb, ub, pars, covars=Float64[])

Compute cumulative hazard over interval [lb, ub].
"""
function cumulative_hazard(hazard::Union{MarkovHazard,SemiMarkovHazard,SplineHazard}, 
                          lb::Real, ub::Real, 
                          pars::AbstractVector, 
                          covars::Union{AbstractVector,NamedTuple}=Float64[])
    return hazard.cumhaz_fn(lb, ub, pars, covars)
end

#=============================================================================
PHASE 2: Backward Compatibility Layer
=============================================================================# 

"""
Backward compatibility: Make new hazard types work with old call_haz() dispatch.

For new hazard types (MarkovHazard, SemiMarkovHazard, SplineHazard), we extract covariates
by name using the parnames field and the provided subjdat row.

# Note on Interface:
- Old hazard types: Expect `rowind` and have `.data` field
- New hazard types: Expect `subjdat::DataFrameRow` and use `parnames` to extract covariates
- Both interfaces supported for backward compatibility
"""

function _maybe_transform_hazard(hazard::_Hazard, parameters, covars::NamedTuple, t::Real;
                                 apply_transform::Bool,
                                 cache_context::Union{Nothing,TimeTransformContext}=nothing,
                                 hazard_slot::Union{Nothing,Int}=nothing)
    use_transform = apply_transform && hazard.metadata.time_transform
    use_transform || return hazard(t, parameters, covars)

    _ensure_transform_supported(hazard)
    linpred = _linear_predictor(parameters, covars, hazard)

    if cache_context === nothing || hazard_slot === nothing
        return _time_transform_hazard(hazard, parameters, t, linpred)
    end

    cache = _shared_or_local_cache(cache_context, hazard_slot, hazard)
    key = _hazard_cache_key(cache, linpred, t)
    return get!(cache.hazard_values, key) do
        _time_transform_hazard(hazard, parameters, t, linpred)
    end
end

function _maybe_transform_cumulhaz(hazard::_Hazard, parameters, covars::NamedTuple, lb::Real, ub::Real;
                                   apply_transform::Bool,
                                   cache_context::Union{Nothing,TimeTransformContext}=nothing,
                                   hazard_slot::Union{Nothing,Int}=nothing)
    use_transform = apply_transform && hazard.metadata.time_transform
    use_transform || return cumulative_hazard(hazard, lb, ub, parameters, covars)

    _ensure_transform_supported(hazard)
    linpred = _linear_predictor(parameters, covars, hazard)

    if cache_context === nothing || hazard_slot === nothing
        return _time_transform_cumhaz(hazard, parameters, lb, ub, linpred)
    end

    cache = _shared_or_local_cache(cache_context, hazard_slot, hazard)
    key = _cumul_cache_key(cache, linpred, lb, ub)
    return get!(cache.cumulhaz_values, key) do
        _time_transform_cumhaz(hazard, parameters, lb, ub, linpred)
    end
end

# MarkovHazard backward compatibility (with subjdat)
function call_haz(t, parameters, covars::NamedTuple, hazard::MarkovHazard;
                  give_log = true,
                  apply_transform::Bool = false,
                  cache_context::Union{Nothing,TimeTransformContext}=nothing,
                  hazard_slot::Union{Nothing,Int}=nothing)
    value = _maybe_transform_hazard(
        hazard,
        parameters,
        covars,
        t;
        apply_transform = apply_transform,
        cache_context = cache_context,
        hazard_slot = hazard_slot)
    give_log ? log(value) : value
end

function call_haz(t, parameters, subjdat::Union{DataFrameRow,DataFrame}, hazard::MarkovHazard;
                  give_log = true,
                  apply_transform::Bool = false,
                  cache_context::Union{Nothing,TimeTransformContext}=nothing,
                  hazard_slot::Union{Nothing,Int}=nothing)
    covars = extract_covariates(subjdat, hazard.parnames)
    return call_haz(
        t,
        parameters,
        covars,
        hazard;
        give_log = give_log,
        apply_transform = apply_transform,
        cache_context = cache_context,
        hazard_slot = hazard_slot)
end

function call_cumulhaz(lb, ub, parameters, covars::NamedTuple, hazard::MarkovHazard;
                       give_log = true,
                       apply_transform::Bool = false,
                       cache_context::Union{Nothing,TimeTransformContext}=nothing,
                       hazard_slot::Union{Nothing,Int}=nothing)
    value = _maybe_transform_cumulhaz(
        hazard,
        parameters,
        covars,
        lb,
        ub;
        apply_transform = apply_transform,
        cache_context = cache_context,
        hazard_slot = hazard_slot)
    give_log ? log(value) : value
end

function call_cumulhaz(lb, ub, parameters, subjdat::Union{DataFrameRow,DataFrame}, hazard::MarkovHazard;
                       give_log = true,
                       apply_transform::Bool = false,
                       cache_context::Union{Nothing,TimeTransformContext}=nothing,
                       hazard_slot::Union{Nothing,Int}=nothing)
    covars = extract_covariates(subjdat, hazard.parnames)
    return call_cumulhaz(
        lb,
        ub,
        parameters,
        covars,
        hazard;
        give_log = give_log,
        apply_transform = apply_transform,
        cache_context = cache_context,
        hazard_slot = hazard_slot)
end

# SemiMarkovHazard backward compatibility (with subjdat)
function call_haz(t, parameters, covars::NamedTuple, hazard::SemiMarkovHazard;
                  give_log = true,
                  apply_transform::Bool = false,
                  cache_context::Union{Nothing,TimeTransformContext}=nothing,
                  hazard_slot::Union{Nothing,Int}=nothing)
    value = _maybe_transform_hazard(
        hazard,
        parameters,
        covars,
        t;
        apply_transform = apply_transform,
        cache_context = cache_context,
        hazard_slot = hazard_slot)
    give_log ? log(value) : value
end

function call_haz(t, parameters, subjdat::Union{DataFrameRow,DataFrame}, hazard::SemiMarkovHazard;
                  give_log = true,
                  apply_transform::Bool = false,
                  cache_context::Union{Nothing,TimeTransformContext}=nothing,
                  hazard_slot::Union{Nothing,Int}=nothing)
    covars = extract_covariates(subjdat, hazard.parnames)
    return call_haz(
        t,
        parameters,
        covars,
        hazard;
        give_log = give_log,
        apply_transform = apply_transform,
        cache_context = cache_context,
        hazard_slot = hazard_slot)
end

function call_cumulhaz(lb, ub, parameters, covars::NamedTuple, hazard::SemiMarkovHazard;
                       give_log = true,
                       apply_transform::Bool = false,
                       cache_context::Union{Nothing,TimeTransformContext}=nothing,
                       hazard_slot::Union{Nothing,Int}=nothing)
    value = _maybe_transform_cumulhaz(
        hazard,
        parameters,
        covars,
        lb,
        ub;
        apply_transform = apply_transform,
        cache_context = cache_context,
        hazard_slot = hazard_slot)
    give_log ? log(value) : value
end

function call_cumulhaz(lb, ub, parameters, subjdat::Union{DataFrameRow,DataFrame}, hazard::SemiMarkovHazard;
                       give_log = true,
                       apply_transform::Bool = false,
                       cache_context::Union{Nothing,TimeTransformContext}=nothing,
                       hazard_slot::Union{Nothing,Int}=nothing)
    covars = extract_covariates(subjdat, hazard.parnames)
    return call_cumulhaz(
        lb,
        ub,
        parameters,
        covars,
        hazard;
        give_log = give_log,
        apply_transform = apply_transform,
        cache_context = cache_context,
        hazard_slot = hazard_slot)
end

# SplineHazard backward compatibility (with subjdat)
function call_haz(t, parameters, covars::NamedTuple, hazard::SplineHazard;
                  give_log = true,
                  apply_transform::Bool = false,
                  cache_context::Union{Nothing,TimeTransformContext}=nothing,
                  hazard_slot::Union{Nothing,Int}=nothing)
    apply_transform && _ensure_transform_supported(hazard)
    value = hazard(t, parameters, covars)
    give_log ? log(value) : value
end

function call_haz(t, parameters, subjdat::Union{DataFrameRow,DataFrame}, hazard::SplineHazard;
                  give_log = true,
                  apply_transform::Bool = false,
                  cache_context::Union{Nothing,TimeTransformContext}=nothing,
                  hazard_slot::Union{Nothing,Int}=nothing)
    covars = extract_covariates(subjdat, hazard.parnames)
    return call_haz(
        t,
        parameters,
        covars,
        hazard;
        give_log = give_log,
        apply_transform = apply_transform,
        cache_context = cache_context,
        hazard_slot = hazard_slot)
end

function call_cumulhaz(lb, ub, parameters, covars::NamedTuple, hazard::SplineHazard;
                       give_log = true,
                       apply_transform::Bool = false,
                       cache_context::Union{Nothing,TimeTransformContext}=nothing,
                       hazard_slot::Union{Nothing,Int}=nothing)
    apply_transform && _ensure_transform_supported(hazard)
    value = cumulative_hazard(hazard, lb, ub, parameters, covars)
    give_log ? log(value) : value
end

function call_cumulhaz(lb, ub, parameters, subjdat::Union{DataFrameRow,DataFrame}, hazard::SplineHazard;
                       give_log = true,
                       apply_transform::Bool = false,
                       cache_context::Union{Nothing,TimeTransformContext}=nothing,
                       hazard_slot::Union{Nothing,Int}=nothing)
    covars = extract_covariates(subjdat, hazard.parnames)
    return call_cumulhaz(
        lb,
        ub,
        parameters,
        covars,
        hazard;
        give_log = give_log,
        apply_transform = apply_transform,
        cache_context = cache_context,
        hazard_slot = hazard_slot)
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
        tot_haz += call_cumulhaz(
            lb,
            ub,
            parameters[x],
            subjdat_row,
            _hazards[x];
            give_log = false,
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
        tot_haz += _call_cumulhaz_with_covars(
            lb,
            ub,
            parameters[x],
            covars_cache,
            _hazards[x],
            x;
            give_log = false,
            apply_transform = apply_transform,
            cache_context = cache_context)
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

    vals = map(x -> _call_haz_with_covars(
            t,
            parameters[x],
            covars_cache,
            hazards[x],
            x;
            apply_transform = apply_transform && hazards[x].metadata.time_transform,
            cache_context = cache_context),
        totalhazards[scur].components)
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

Fill in a matrix of transition intensities for a multistate Markov model.
"""
function compute_hazmat!(Q, parameters, hazards::Vector{T}, tpm_index::DataFrame, model_data::DataFrame) where T <: _Hazard

    # Get the DataFrameRow for covariate extraction
    subjdat_row = model_data[tpm_index.datind[1], :]
    
    # compute transition intensities
    for h in eachindex(hazards) 
        Q[hazards[h].statefrom, hazards[h].stateto] = 
            call_haz(
                tpm_index.tstart[1], 
                parameters[h],
                subjdat_row,
                hazards[h]; 
                give_log = false)
    end

    # set diagonal elements equal to the sum of off-diags
    Q[diagind(Q)] = -sum(Q, dims = 2)
end

"""
    compute_tmat!(P, Q, tpm_index::DataFrame, cache)

Calculate transition probability matrices for a multistate Markov process. 
"""
function compute_tmat!(P, Q, tpm_index::DataFrame, cache)

    for t in eachindex(P)
        copyto!(P[t], exponential!(Q * tpm_index.tstop[t], ExpMethodGeneric(), cache))
    end  
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
            incidences[r,h] = 
                survprobs[r,trans_inds[h]] * 
                quadgk(t -> (
                        call_haz(t, parameters[h], subjdat_row, hazards[h]; give_log = false) * 
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
            incidences[r,h] = 
                survprobs[r] * 
                quadgk(t -> (
                        call_haz(t, parameters[hazinds[h]], subj_inds[interval_inds[r]], hazards[hazinds[h]]; give_log = false) * 
                        survprob(subj_times[r], t, parameters, subj_inds[interval_inds[r]], totalhazards[statefrom], hazards; give_log = false)), 
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

    # compute hazards
    hazards = zeros(Float64, length(t))
    for s in eachindex(t)
        # get row index
        rowind = findlast((model.data.id .== subj) .& (model.data.tstart .<= t[s]))

        # compute hazard
        hazards[s] = call_haz(t[s], model.parameters[hazind], rowind, model.hazards[hazind]; give_log = false)
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

    # compute hazards
    cumulative_hazards = zeros(Float64, length(tstart))
    for s in eachindex(tstart)

        # find times between tstart and tstop
        times = [tstart[s]; model.data.tstart[findall((model.data.id .== subj) .& (model.data.tstart .> tstart[s]) .& (model.data.tstart .< tstop[s]))]; tstop[s]]

        # initialize cumulative hazard
        chaz = 0.0

        # accumulate
        for i in 1:(length(times) - 1)
            # get row index
            rowind = findlast((model.data.id .== subj) .& (model.data.tstart .<= times[i]))

            # compute hazard
            chaz += call_cumulhaz(times[i], times[i+1], model.parameters[hazind], rowind, model.hazards[hazind]; give_log = false)
        end

        # save
        cumulative_hazards[s] = chaz
    end

    # return cumulative hazards
    return cumulative_hazards
end
