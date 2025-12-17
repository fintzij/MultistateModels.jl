# =============================================================================
# Covariate Extraction and Linear Predictor Functions
# =============================================================================
#
# Functions for extracting covariates from data and computing linear predictors.
# These are used throughout hazard evaluation.
#
# =============================================================================

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
    _lookup_covariate_value(subjdat, cname)

Look up a covariate value from subject data, handling interaction terms.
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

"""
    extract_covariates(subjdat::Union{DataFrameRow,DataFrame}, parnames::Vector{Symbol})

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
function extract_covariates(subjdat::Union{DataFrameRow,DataFrame}, parnames::Vector{Symbol})
    covar_names = extract_covar_names(parnames)
    
    if isempty(covar_names)
        return NamedTuple()
    end
    
    # Build NamedTuple by extracting each covariate value
    vals = [_lookup_covariate_value(subjdat, cname) for cname in covar_names]
    return NamedTuple{Tuple(covar_names)}(Tuple(vals))
end

"""
    extract_covariates_fast(subjdat::DataFrameRow, covar_names::Vector{Symbol})

Fast covariate extraction using pre-cached covariate names (avoids regex parsing).
Returns empty NamedTuple if no covariates.

This is the preferred method in hot paths where covar_names has already been
extracted from parameter names (stored in hazard.covar_names).
"""
@inline function extract_covariates_fast(subjdat::DataFrameRow, covar_names::Vector{Symbol})
    isempty(covar_names) && return NamedTuple()
    vals = Tuple(_lookup_covariate_value(subjdat, cname) for cname in covar_names)
    return NamedTuple{Tuple(covar_names)}(vals)
end

# =============================================================================
# Linear Predictor Computation
# =============================================================================

# Union type for covariate data: either cached NamedTuple or direct DataFrame view
const CovariateData = Union{NamedTuple, DataFrameRow}

# Helpers for covariate entry lookup
@inline _covariate_entry(covars_cache::AbstractVector{<:NamedTuple}, hazard_slot::Int) = covars_cache[hazard_slot]
@inline _covariate_entry(covars_cache::DataFrameRow, ::Int) = covars_cache
@inline _covariate_entry(covars_cache, ::Int) = covars_cache

"""
    _linear_predictor(pars::AbstractVector, covars::CovariateData, hazard::_Hazard)

Compute linear predictor β'x from flat parameter vector and covariates.
Uses hazard.covar_names for covariate names and hazard.npar_baseline for offset.
"""
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

"""
    _linear_predictor(pars::NamedTuple, covars::CovariateData, hazard::_Hazard)

Compute linear predictor β'x from named parameters and covariates.
Handles nested NamedTuple structure (baseline and covariates fields).
"""
@inline function _linear_predictor(pars::NamedTuple, covars::CovariateData, hazard::_Hazard)
    hazard.has_covariates || return zero(Float64)
    covar_pars = pars.covariates
    covar_names = hazard.covar_names  # Use the actual covariate names from hazard
    linpred = zero(Float64)
    for cname in covar_names
        val = getproperty(covars, cname)
        # Find the coefficient - covar_pars keys are like h12_x, we need to match by suffix
        coeff = zero(Float64)
        for pname in keys(covar_pars)
            # Check if pname ends with _cname (e.g., h12_x ends with _x)
            pname_str = String(pname)
            cname_str = String(cname)
            if endswith(pname_str, "_" * cname_str) || endswith(pname_str, cname_str)
                coeff = getproperty(covar_pars, pname)
                break
            end
        end
        linpred += coeff * val
    end
    return linpred
end

# =============================================================================
# Parameter Vector Extraction for Time Transform
# =============================================================================

"""
    _time_transform_pars(pars)

Extract parameter vector suitable for time transform functions.
Handles both flat vectors and NamedTuple parameter formats.
"""
@inline function _time_transform_pars(pars::AbstractVector)
    return pars
end

@inline function _time_transform_pars(pars::NamedTuple)
    # For NamedTuple, extract baseline parameters as vector
    return collect(values(pars.baseline))
end

# =============================================================================
# Linear Predictor Expression Builder (for code generation)
# =============================================================================

"""
    _build_linear_pred_expr_named(parnames::Vector{Symbol})

Build linear predictor expression that accesses covariate coefficients by name.

Generates: covars.age * pars.covariates.h13_age + covars.sex * pars.covariates.h13_sex + ...

Note: - Covariate VALUES (covars.XXX) use names WITHOUT hazard prefix (from data)
      - Covariate COEFFICIENTS (pars.covariates.XXX) use parameter names WITH prefix
      - `parnames` contains the full parameter names (e.g., :h13_trt, :h13_age)
"""
function _build_linear_pred_expr_named(parnames::Vector{Symbol})
    # Get covariate names without prefix
    covar_names = extract_covar_names(parnames)
    
    if isempty(covar_names)
        return :(0.0)  # Return zero literal instead of zero(eltype(pars)) which fails for NamedTuples
    end
    
    # Build mapping from covariate name (no prefix) to parameter name (with prefix)
    covar_to_par = Dict{Symbol,Symbol}()
    for pname in parnames
        pname_str = String(pname)
        # Skip baseline parameters
        if occursin("Intercept", pname_str) || occursin("shape", pname_str) || 
           occursin("scale", pname_str) || occursin(r"^h\d+_sp\d+$", pname_str)
            continue
        end
        # Get covariate name (without prefix)
        covar_name = Symbol(replace(pname_str, r"^h\d+_" => ""))
        covar_to_par[covar_name] = pname
    end
    
    # Build sum of covars.name * pars.covariates.parname
    terms = Any[]
    for cname in covar_names
        parname = covar_to_par[cname]
        push!(terms, :(covars.$(cname) * pars.covariates.$(parname)))
    end
    
    return length(terms) == 1 ? terms[1] : Expr(:call, :+, terms...)
end
