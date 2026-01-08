# hazard_builders.jl - Internal hazard building infrastructure
#
# This file contains:
# - HazardBuildContext struct for passing build state
# - Hazard family registry (register_hazard_family!)
# - Parametric hazard builders (exponential, Weibull, Gompertz)
# - Common build utilities (_build_parametric_hazard_common, etc.)

# hazard build orchestration -------------------------------------------------

struct HazardBuildContext
    hazard::HazardFunction
    hazname::Symbol
    family::Symbol
    metadata::HazardMetadata
    rhs_names::Vector{String}
    shared_baseline_key::Union{Nothing,SharedBaselineKey}
    data::DataFrame  # For data-dependent operations like automatic knot placement
    hazschema::StatsModels.AbstractTerm  # Result of apply_schema (FormulaTerm <: AbstractTerm)
end

# Computed accessors for HazardBuildContext - avoids storing redundant data
_ncovar(ctx::HazardBuildContext) = max(length(ctx.rhs_names) - 1, 0)
_has_covariates(ctx::HazardBuildContext) = length(ctx.rhs_names) > 1

const _HAZARD_BUILDERS = Dict{Symbol,Function}()

function register_hazard_family!(name::Union{AbstractString,Symbol}, builder::Function)
    key = name isa Symbol ? name : Symbol(lowercase(String(name)))
    _HAZARD_BUILDERS[key] = builder
    return builder
end

function _hazard_rhs_names(hazschema)
    has_intercept = _hazard_formula_has_intercept(hazschema.rhs)
    coef_names = StatsModels.coefnames(hazschema.rhs)
    coef_vec = coef_names isa AbstractVector ? collect(coef_names) : [coef_names]
    return has_intercept ? coef_vec : vcat("(Intercept)", coef_vec)
end

@inline function _hazard_name(hazard::HazardFunction)
    return Symbol("h$(hazard.statefrom)$(hazard.stateto)")
end

function _prepare_hazard_context(hazard::HazardFunction,
                                 data::DataFrame;
                                 surrogate::Bool = false)
    # Note: `surrogate=true` builds a **Markov** surrogate (exponential hazards).
    # This is the first step in surrogate construction. For non-exponential hazards,
    # a PhaseTypeSurrogate is subsequently built FROM this MarkovSurrogate during
    # MCEM fitting (see fit_phasetype_surrogate in surrogate/markov.jl).
    #
    # The selection of Markov vs phase-type proposal happens in resolve_proposal_config()
    # based on needs_phasetype_proposal() which checks if hazards are non-exponential.
    schema = StatsModels.schema(hazard.hazard, data)
    hazschema = apply_schema(hazard.hazard, schema)
    modelcols(hazschema, data) # validate design matrix construction
    rhs_names = _hazard_rhs_names(hazschema)
    # Normalize family to Symbol (hazard.family is now Symbol)
    # For Markov surrogates, coerce to exponential regardless of original family
    runtime_family = surrogate ? :exp : hazard.family
    hazname = _hazard_name(hazard)
    shared_key = shared_baseline_key(hazard, runtime_family)
    return HazardBuildContext(
        hazard,
        hazname,
        runtime_family,
        hazard.metadata,
        rhs_names,
        shared_key,
        data,
        hazschema,
    )
end

@inline function _covariate_labels(rhs_names::Vector{String})
    if length(rhs_names) <= 1
        return String[]
    end
    return rhs_names[2:end]
end

@inline function _prefixed_symbols(hazname::Symbol, labels::Vector{String})
    prefix = string(hazname) * "_"
    # Covariate labels from StatsModels; baseline "(Intercept)" is handled separately
    # so we don't expect it here, but clean it just in case for robustness
    return Symbol.(prefix .* labels)
end

"""
    _build_parametric_hazard_common(ctx, baseline_names, generator_fn, hazard_type)

Common logic for building parametric hazard structs. Reduces duplication across
exponential, Weibull, and Gompertz builders.

Note: As of v0.3.0, initial parameters are on NATURAL scale, not log scale.
"""
function _build_parametric_hazard_common(
    ctx::HazardBuildContext,
    baseline_names::Vector{Symbol},
    generator_fn::Function,
    ::Type{HazType}
) where {HazType <: _Hazard}
    covar_pars = _semimarkov_covariate_parnames(ctx)
    parnames = vcat(baseline_names, covar_pars)
    hazard_fn, cumhaz_fn = generator_fn(parnames, ctx.metadata.linpred_effect)
    npar_baseline = length(baseline_names)
    ncovar = _ncovar(ctx)
    npar_total = npar_baseline + ncovar
    covar_names = Symbol.(_covariate_labels(ctx.rhs_names))
    smooth_info = _extract_smooth_info(ctx, parnames)
    
    haz_struct = HazType(
        ctx.hazname,
        ctx.hazard.statefrom,
        ctx.hazard.stateto,
        ctx.family,
        parnames,
        npar_baseline,
        npar_total,
        hazard_fn,
        cumhaz_fn,
        _has_covariates(ctx),
        covar_names,
        ctx.metadata,
        ctx.shared_baseline_key,
        smooth_info,
    )
    
    # v0.3.0+: Initialize parameters on NATURAL scale
    # - Covariate coefficients: 0.0 (no effect)
    # - Baseline: family-specific defaults for rate=1 behavior
    init_pars = zeros(Float64, npar_total)
    if ctx.family == :exp
        init_pars[1] = 1.0  # rate = 1.0
    elseif ctx.family == :wei
        init_pars[1] = 1.0  # shape = 1.0 (reduces to exponential)
        init_pars[2] = 1.0  # scale = 1.0
    elseif ctx.family == :gom
        init_pars[1] = 0.0  # shape = 0 (reduces to exponential: h(t) = rate * exp(0*t))
        init_pars[2] = 1.0  # rate = 1.0
    else
        # For unknown families, use 1.0 for all baseline params (conservative default)
        init_pars[1:npar_baseline] .= 1.0
    end
    
    return haz_struct, init_pars
end

function _build_exponential_hazard(ctx::HazardBuildContext)
    baseline = Symbol[Symbol(string(ctx.hazname), "_rate")]
    return _build_parametric_hazard_common(ctx, baseline, generate_exponential_hazard, MarkovHazard)
end

function _semimarkov_covariate_parnames(ctx::HazardBuildContext)
    covars = _covariate_labels(ctx.rhs_names)
    return _prefixed_symbols(ctx.hazname, covars)
end

"""
    _extract_smooth_info(ctx::HazardBuildContext, parnames::Vector{Symbol})

Extract information about smooth covariate terms (s(x), te(x,y)) from the hazard schema.
"""
function _extract_smooth_info(ctx::HazardBuildContext, parnames::Vector{Symbol})
    smooth_info = SmoothTermInfo[]
    rhs = ctx.hazschema.rhs
    
    # MatrixTerm contains all terms on the RHS
    terms = rhs isa StatsModels.MatrixTerm ? rhs.terms : (rhs,)
    
    for term in terms
        if term isa SmoothTerm
            # Find indices of coefficients for this term in parnames
            cnames = StatsModels.coefnames(term)
            prefixed_cnames = Symbol.(string(ctx.hazname) * "_" .* cnames)
            
            indices = [findfirst(==(p), parnames) for p in prefixed_cnames]
            if any(isnothing, indices)
                # This should not happen if parnames was built correctly
                throw(ArgumentError("Could not find parameter indices for smooth term $(term.label) in hazard $(ctx.hazname)"))
            end
            
            push!(smooth_info, SmoothTermInfo(indices, term.S, term.label))
        elseif term isa TensorProductTerm
            # Find indices of coefficients for tensor product term
            cnames = StatsModels.coefnames(term)
            prefixed_cnames = Symbol.(string(ctx.hazname) * "_" .* cnames)
            
            indices = [findfirst(==(p), parnames) for p in prefixed_cnames]
            if any(isnothing, indices)
                throw(ArgumentError("Could not find parameter indices for tensor product term $(term.label) in hazard $(ctx.hazname)"))
            end
            
            push!(smooth_info, SmoothTermInfo(indices, term.S, term.label))
        end
    end
    
    return smooth_info
end

function _build_weibull_hazard(ctx::HazardBuildContext)
    baseline = Symbol[Symbol(string(ctx.hazname), "_shape"), Symbol(string(ctx.hazname), "_scale")]
    return _build_parametric_hazard_common(ctx, baseline, generate_weibull_hazard, SemiMarkovHazard)
end

function _build_gompertz_hazard(ctx::HazardBuildContext)
    baseline = Symbol[Symbol(string(ctx.hazname), "_shape"), Symbol(string(ctx.hazname), "_scale")]
    return _build_parametric_hazard_common(ctx, baseline, generate_gompertz_hazard, SemiMarkovHazard)
end

# Register parametric hazard families
register_hazard_family!(:exp, _build_exponential_hazard)
register_hazard_family!(:wei, _build_weibull_hazard)
register_hazard_family!(:gom, _build_gompertz_hazard)
