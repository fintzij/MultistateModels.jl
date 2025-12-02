const _DEFAULT_HAZARD_FORMULA = StatsModels.@formula(0 ~ 1)

@inline function _hazard_formula_has_intercept(rhs_term)
    rhs_term isa StatsModels.ConstantTerm && return true
    rhs_term isa StatsModels.InterceptTerm && return true
    if rhs_term isa StatsModels.MatrixTerm
        return any(_hazard_formula_has_intercept, rhs_term.terms)
    end
    return false
end

"""
        Hazard(family::Union{AbstractString,Symbol}, statefrom::Integer, stateto::Integer; kwargs...)
        Hazard(hazard::StatsModels.FormulaTerm, family::Union{AbstractString,Symbol}, statefrom::Integer, stateto::Integer; kwargs...)

Construct a parametric or semi-parametric cause-specific hazard specification to be
consumed by `multistatemodel`. Provide a `StatsModels` formula only when the hazard
has covariates; when you omit it, the constructor automatically supplies the
intercept-only design `@formula(0 ~ 1)` so you never have to write `+ 1` yourself.

# Positional arguments
- `family`: "exp", "wei", "gom", or "sp" (string or symbol, case-insensitive).
- `statefrom` / `stateto`: integers describing the transition.
- `hazard` *(optional)*: a `StatsModels.FormulaTerm` describing covariates that act
    multiplicatively on the baseline hazard. Skip this argument for intercept-only hazards.

# Keyword arguments
- `degree`, `knots`, `boundaryknots`, `natural_spline`, `extrapolation`, `monotone`:
    spline controls used only when `family == "sp"`. See the BSplineKit docs for details.
- `monotone`: `0` (default) leaves the spline unconstrained, `1` enforces an increasing
    hazard, and `-1` enforces a decreasing hazard.
- `time_transform::Bool`: enable Tang-style shared-trajectory caching for this transition.
- `linpred_effect::Symbol`: `:ph` (default) for proportional hazards or `:aft` for
    accelerated-failure-time behaviour.

# Examples
```julia
julia> Hazard("exp", 1, 2)                      # intercept only
julia> Hazard(@formula(0 ~ age + trt), "wei", 1, 3)
julia> @hazard begin                             # macro front-end uses the same rules
                     family = :gom
                     transition = 2 => 4
                     formula = @formula(0 ~ stage)
             end
```
"""
function Hazard(
    hazard::StatsModels.FormulaTerm,
    family::Union{AbstractString,Symbol},
    statefrom::Int64,
    stateto::Int64;
    degree::Int64 = 3,
    knots::Union{Vector{Float64}, Float64, Nothing} = nothing,
    boundaryknots::Union{Vector{Float64}, Nothing} = nothing,
    natural_spline = true,
    extrapolation = "linear",
    monotone = 0,
    time_transform::Bool = false,
    linpred_effect::Symbol = :ph)

    # Input validation
    @assert statefrom > 0 "statefrom must be a positive integer, got $statefrom"
    @assert stateto > 0 "stateto must be a positive integer, got $stateto"
    @assert statefrom != stateto "statefrom and stateto must differ (got $statefrom → $stateto)"
    
    family_str = family isa String ? family : String(family)
    family_key = lowercase(family_str)
    
    valid_families = ("exp", "wei", "gom", "sp")
    @assert family_key in valid_families "family must be one of $valid_families, got \"$family_key\""
    
    if family_key == "sp"
        @assert degree >= 0 "spline degree must be non-negative, got $degree"
        @assert extrapolation in ("linear", "flat") "extrapolation must be \"linear\" or \"flat\", got \"$extrapolation\""
        @assert monotone in (-1, 0, 1) "monotone must be -1, 0, or 1, got $monotone"
    end
    
    @assert linpred_effect in (:ph, :aft) "linpred_effect must be :ph or :aft, got :$linpred_effect"
    
    metadata = HazardMetadata(time_transform = time_transform, linpred_effect = linpred_effect)

    if family_key != "sp"
        h = ParametricHazard(hazard, family_key, statefrom, stateto, metadata)
    else 
        if natural_spline & (monotone != 0)
            @info "Natural boundary conditions are not currently compatible with monotone splines. The restrictions on second derivatives at the spline boundaries will be removed."
            natural_spline = false
        end

        # change extrapolation to flat if degree = 0
        extrapolation = degree > 0 ? extrapolation : "flat"
        natural_spline = degree < 2 ? false : natural_spline

        h = SplineHazard(hazard, family_key, statefrom, stateto, degree, knots, boundaryknots, extrapolation, natural_spline, sign(monotone), metadata)
    end

    return h
end

function Hazard(
    family::Union{AbstractString,Symbol},
    statefrom::Integer,
    stateto::Integer;
    kwargs...)
    family_str = family isa String ? family : String(family)
    return Hazard(_DEFAULT_HAZARD_FORMULA, family_str, Int(statefrom), Int(stateto); kwargs...)
end

"""
    enumerate_hazards(hazards::HazardFunction...)

Standardise a collection of `Hazard`/`@hazard` definitions. The result is a
`DataFrame` with columns `statefrom`, `stateto`, `trans`, and `order`, sorted by
origin and destination so downstream helpers (e.g. `create_tmat`) can rely on a
stable ordering. Duplicate transitions (same origin/destination pair) raise an error.
"""
function enumerate_hazards(hazards::HazardFunction...)

    n_haz = length(hazards)

    # initialize state space information
    hazinfo = 
        DataFrames.DataFrame(
            statefrom = zeros(Int64, n_haz),
            stateto = zeros(Int64, n_haz),
            trans = zeros(Int64, n_haz),
            order = collect(1:n_haz))

    # grab the origin and destination states for each hazard
    for i in eachindex(hazards)
        hazinfo.statefrom[i] = hazards[i].statefrom
        hazinfo.stateto[i] = hazards[i].stateto
    end

    # check for duplicate transitions
    transition_pairs = [(hazinfo.statefrom[i], hazinfo.stateto[i]) for i in 1:n_haz]
    if length(unique(transition_pairs)) != n_haz
        duplicates = [tp for tp in unique(transition_pairs) if count(==(tp), transition_pairs) > 1]
        error("Duplicate transitions detected: $(duplicates). Each transition (statefrom → stateto) should be specified only once.")
    end

    # enumerate and sort hazards
    sort!(hazinfo, [:statefrom, :stateto])
    hazinfo[:,:trans] = collect(1:n_haz)

    # return the hazard information
    return hazinfo
end

"""
    create_tmat(hazinfo::DataFrame)

Create the familiar transition matrix that `multistatemodel` expects. Rows are
origin states, columns are destination states, zeros mark impossible transitions,
and the positive entries are the transition numbers assigned by `enumerate_hazards`.
"""
function create_tmat(hazinfo::DataFrame)
    
    # initialize the transition matrix
    statespace = sort(unique([hazinfo[:,:statefrom] hazinfo[:, :stateto]]))
    n_states = length(statespace)

    # initialize transition matrix
    tmat = zeros(Int64, n_states, n_states)

    for i in axes(hazinfo, 1)
        tmat[hazinfo.statefrom[i], hazinfo.stateto[i]] = 
            hazinfo.trans[i]
    end

    return tmat
end


# hazard build orchestration -------------------------------------------------

struct HazardBuildContext
    hazard::HazardFunction
    hazname::Symbol
    family::String
    metadata::HazardMetadata
    rhs_names::Vector{String}
    shared_baseline_key::Union{Nothing,SharedBaselineKey}
    data::DataFrame  # For data-dependent operations like automatic knot placement
end

# Computed accessors for HazardBuildContext - avoids storing redundant data
_ncovar(ctx::HazardBuildContext) = max(length(ctx.rhs_names) - 1, 0)
_has_covariates(ctx::HazardBuildContext) = length(ctx.rhs_names) > 1

const _HAZARD_BUILDERS = Dict{String,Function}()

function register_hazard_family!(name::AbstractString, builder::Function)
    _HAZARD_BUILDERS[lowercase(String(name))] = builder
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
    schema = StatsModels.schema(hazard.hazard, data)
    hazschema = apply_schema(hazard.hazard, schema)
    modelcols(hazschema, data) # validate design matrix construction
    rhs_names = _hazard_rhs_names(hazschema)
    runtime_family = surrogate ? "exp" : lowercase(hazard.family)
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
    clean = replace.(labels, "(Intercept)" => "Intercept")
    return Symbol.(prefix .* clean)
end

"""
    _build_parametric_hazard_common(ctx, baseline_names, generator_fn, hazard_type)

Common logic for building parametric hazard structs. Reduces duplication across
exponential, Weibull, and Gompertz builders.
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
    )
    return haz_struct, zeros(Float64, npar_total)
end

function _build_exponential_hazard(ctx::HazardBuildContext)
    baseline = Symbol[Symbol(string(ctx.hazname), "_Intercept")]
    return _build_parametric_hazard_common(ctx, baseline, generate_exponential_hazard, MarkovHazard)
end

function _baseline_parnames(hazname::Symbol, labels::Vector{String})
    return _prefixed_symbols(hazname, labels)
end

function _semimarkov_covariate_parnames(ctx::HazardBuildContext)
    covars = _covariate_labels(ctx.rhs_names)
    return _prefixed_symbols(ctx.hazname, covars)
end

function _build_weibull_hazard(ctx::HazardBuildContext)
    baseline = Symbol[Symbol(string(ctx.hazname), "_shape"), Symbol(string(ctx.hazname), "_scale")]
    return _build_parametric_hazard_common(ctx, baseline, generate_weibull_hazard, SemiMarkovHazard)
end

function _build_gompertz_hazard(ctx::HazardBuildContext)
    baseline = Symbol[Symbol(string(ctx.hazname), "_shape"), Symbol(string(ctx.hazname), "_scale")]
    return _build_parametric_hazard_common(ctx, baseline, generate_gompertz_hazard, SemiMarkovHazard)
end

"""
    _build_spline_hazard(ctx::HazardBuildContext)

Build a SplineHazard from the hazard specification context.

Uses BSplineKit to construct the spline basis at build time, then generates
runtime hazard/cumhaz functions that construct Spline objects on-the-fly from
the current parameters. This functional approach ensures AD compatibility.

The spline coefficients are parameterized on log scale for positivity.
For monotone splines (monotone != 0), an I-spline-like cumsum transformation
is applied via spline_ests2coefs().
"""
function _build_spline_hazard(ctx::HazardBuildContext)
    # Access the original SplineHazard (user-facing type) from context
    hazard = ctx.hazard::SplineHazard
    data = ctx.data
    
    # Covariate parameter names (if any)
    covar_pars = _semimarkov_covariate_parnames(ctx)
    covar_names = Symbol.(_covariate_labels(ctx.rhs_names))
    has_covars = !isempty(covar_names)
    
    # Extract sojourn times on the reset scale for this transition
    # Used for automatic boundary and knot placement
    samplepaths = extract_paths(data)
    sojourns_transition = extract_sojourns(hazard.statefrom, hazard.stateto, samplepaths)
    sojourns_stay = extract_sojourns(hazard.statefrom, hazard.statefrom, samplepaths)
    all_sojourns = vcat(sojourns_transition, sojourns_stay)
    
    # Determine boundary knots
    if hazard.boundaryknots === nothing
        if isempty(all_sojourns)
            # No observations for this transition - use timespan as fallback
            bknots = [0.0, maximum(data.tstop)]
        else
            bknots = [0.0, maximum(all_sojourns)]
        end
    else
        bknots = copy(hazard.boundaryknots)
    end
    
    # Determine interior knots - automatic placement if not specified
    if hazard.knots === nothing
        # Automatic knot placement using quantiles
        if isempty(sojourns_transition)
            # No observed transitions - use evenly spaced knots
            nknots = default_nknots(length(all_sojourns))
            if nknots > 0
                intknots = collect(range(bknots[1] + (bknots[2] - bknots[1])/(nknots + 1),
                                         stop=bknots[2] - (bknots[2] - bknots[1])/(nknots + 1),
                                         length=nknots))
            else
                intknots = Float64[]
            end
        else
            nknots = default_nknots(length(sojourns_transition))
            intknots = place_interior_knots(sojourns_transition, nknots;
                                           lower_bound=bknots[1], upper_bound=bknots[2])
        end
        
        if !isempty(intknots)
            @info "Auto-placed $(length(intknots)) interior knots for $(hazard.statefrom)→$(hazard.stateto) transition at: $(round.(intknots, digits=3))"
        end
    elseif hazard.knots isa Float64
        intknots = [hazard.knots]
    else
        intknots = copy(hazard.knots)
    end
    
    # Validate interior knots are within boundaries
    if !isempty(intknots)
        if any(intknots .< bknots[1]) || any(intknots .> bknots[2])
            @warn "Interior knots outside boundary knots detected. Adjusting boundaries."
            bknots[1] = min(bknots[1], minimum(intknots))
            bknots[2] = max(bknots[2], maximum(intknots))
        end
    end
    
    # Combine and sort knots
    allknots = unique(sort([bknots[1]; intknots; bknots[2]]))
    
    # Build B-spline basis
    B = BSplineBasis(BSplineOrder(hazard.degree + 1), copy(allknots))
    
    # Apply natural spline recombination if requested
    if (hazard.degree > 1) && hazard.natural_spline
        B = RecombinedBSplineBasis(B, Natural())
    end
    
    # Determine extrapolation method
    extrap_method = if hazard.extrapolation == "linear"
        BSplineKit.SplineExtrapolations.Linear()
    else
        BSplineKit.SplineExtrapolations.Flat()
    end
    
    # Number of basis functions = number of spline coefficients
    nbasis = length(B)
    
    # Build parameter names: spline coefficients + covariates
    baseline_names = [Symbol(string(ctx.hazname), "_sp", i) for i in 1:nbasis]
    parnames = vcat(baseline_names, covar_pars)
    npar_total = nbasis + length(covar_pars)
    
    # Generate runtime hazard and cumhaz functions
    hazard_fn, cumhaz_fn = _generate_spline_hazard_fns(
        B, extrap_method, hazard.monotone, nbasis, parnames, ctx.metadata.linpred_effect
    )
    
    # Build the internal RuntimeSplineHazard struct
    haz_struct = RuntimeSplineHazard(
        ctx.hazname,
        hazard.statefrom,
        hazard.stateto,
        ctx.family,
        parnames,
        nbasis,
        npar_total,
        hazard_fn,
        cumhaz_fn,
        has_covars,
        covar_names,
        hazard.degree,
        allknots,
        hazard.natural_spline,
        hazard.monotone,
        ctx.metadata,
        ctx.shared_baseline_key,
    )
    
    # Initialize parameters at zero (log scale → exp(0) = 1 for all coefficients)
    init_params = zeros(Float64, npar_total)
    
    return haz_struct, init_params
end

"""
    _generate_spline_hazard_fns(basis, extrap_method, monotone, nbasis, parnames, linpred_effect)

Generate runtime hazard and cumulative hazard functions for spline hazards.

Returns a tuple (hazard_fn, cumhaz_fn) where each function has signature:
- hazard_fn(t, pars, covars) -> Float64
- cumhaz_fn(lb, ub, pars, covars) -> Float64

The functions construct Spline objects on-the-fly from parameters, ensuring
AD compatibility by avoiding stored mutable state.
"""
function _generate_spline_hazard_fns(
    basis::Union{BSplineBasis, RecombinedBSplineBasis},
    extrap_method,
    monotone::Int,
    nbasis::Int,
    parnames::Vector{Symbol},
    linpred_effect::Symbol
)
    # Build linear predictor expression for covariates (if any)
    linear_pred_expr = _build_linear_pred_expr(parnames, nbasis + 1)
    
    # Capture spline infrastructure in closures
    # Note: basis and extrap_method are immutable, safe to capture
    
    if linpred_effect == :ph
        # Proportional hazards: h(t|x) = h0(t) * exp(β'x)
        hazard_fn = let B = basis, ext = extrap_method, mono = monotone, nb = nbasis
            function(t, pars, covars)
                # Transform parameters to spline coefficients
                coefs = _spline_ests2coefs(pars[1:nb], B, mono)
                # Build spline on-the-fly
                spline = Spline(B, coefs)
                spline_ext = SplineExtrapolation(spline, ext)
                # Evaluate baseline hazard
                h0 = spline_ext(t)
                # Apply covariate effect (linear predictor starts at nbasis+1)
                linear_pred = isempty(covars) ? 0.0 : _eval_linear_pred(pars, covars, nb + 1)
                return h0 * exp(linear_pred)
            end
        end
        
        cumhaz_fn = let B = basis, ext = extrap_method, mono = monotone, nb = nbasis
            function(lb, ub, pars, covars)
                # Transform parameters to spline coefficients
                coefs = _spline_ests2coefs(pars[1:nb], B, mono)
                # Build spline and its integral on-the-fly
                spline = Spline(B, coefs)
                spline_ext = SplineExtrapolation(spline, ext)
                cumhaz_spline = integral(spline_ext.spline)
                # Handle extrapolation for cumulative hazard
                # Note: integral() gives the antiderivative of the core spline
                # For flat extrapolation, we need special handling at boundaries
                H0 = _eval_cumhaz_with_extrap(spline_ext, cumhaz_spline, lb, ub)
                # Apply covariate effect
                linear_pred = isempty(covars) ? 0.0 : _eval_linear_pred(pars, covars, nb + 1)
                return H0 * exp(linear_pred)
            end
        end
    else
        error("Spline hazards currently only support proportional hazards (linpred_effect=:ph)")
    end
    
    return hazard_fn, cumhaz_fn
end

"""
    _spline_ests2coefs(ests, basis, monotone)

Transform spline parameter estimates (log scale) to spline coefficients.

For monotone == 0: simple exponentiation (positivity constraint)
For monotone == 1: I-spline-like cumulative sum (increasing hazard)
For monotone == -1: reverse cumulative sum (decreasing hazard)
"""
function _spline_ests2coefs(ests::AbstractVector{T}, basis, monotone::Int) where T
    if monotone == 0
        # Simple positivity: exp(θ)
        return exp.(ests)
    else
        # I-spline transformation for monotonicity
        ests_nat = exp.(ests)
        coefs = zeros(T, length(ests))
        
        if length(coefs) > 1
            k = BSplineKit.order(basis)
            t = BSplineKit.knots(basis)
            
            for i in 2:length(coefs)
                coefs[i] = coefs[i-1] + ests_nat[i] * (t[i + k] - t[i]) / k
            end
        end
        
        # Add intercept
        coefs .+= ests_nat[1]
        
        # Reverse for decreasing monotone
        if monotone == -1
            reverse!(coefs)
        end
        
        return coefs
    end
end

"""
    _spline_coefs2ests(coefs, basis, monotone; clamp_zeros=false)

Transform spline coefficients back to log-scale parameter estimates.
Inverse of `_spline_ests2coefs`.

For monotone == 0: simple log (inverse of exp)
For monotone == 1: difference transformation (inverse of cumsum)
For monotone == -1: reverse then difference
"""
function _spline_coefs2ests(coefs::AbstractVector{T}, basis, monotone::Int; clamp_zeros::Bool=false) where T
    if monotone == 0
        # Inverse of exp
        ests_nat = coefs
    else
        # Reverse if decreasing
        coefs_nat = monotone == 1 ? copy(coefs) : reverse(coefs)
        
        ests_nat = zeros(T, length(coefs_nat))
        
        if length(coefs) > 1
            k = BSplineKit.order(basis)
            t = BSplineKit.knots(basis)
            
            # Inverse of the cumsum: take differences
            for i in length(ests_nat):-1:2
                ests_nat[i] = (coefs_nat[i] - coefs_nat[i - 1]) * k / (t[i + k] - t[i])
            end
        end
        
        # Intercept
        ests_nat[1] = coefs_nat[1]
    end
    
    # Clamp numerical zeros
    if clamp_zeros
        ests_nat[findall(isapprox.(ests_nat, 0.0; atol = sqrt(eps())))] .= zero(T)
    end
    
    return log.(ests_nat)
end

"""
    _eval_linear_pred(pars, covars, start_idx)

Evaluate the linear predictor β'x starting from index start_idx in pars.
"""
function _eval_linear_pred(pars::AbstractVector, covars, start_idx::Int)
    result = zero(eltype(pars))
    if covars isa NamedTuple
        covar_vec = collect(values(covars))
    else
        covar_vec = covars
    end
    for (i, x) in enumerate(covar_vec)
        result += pars[start_idx + i - 1] * x
    end
    return result
end

"""
    _eval_cumhaz_with_extrap(spline_ext, cumhaz_spline, lb, ub)

Evaluate cumulative hazard over [lb, ub] with proper extrapolation handling.

For flat extrapolation: hazard is constant beyond boundaries, so cumulative
hazard grows linearly beyond the spline support.
For linear extrapolation: uses the derivative at boundaries.
"""
function _eval_cumhaz_with_extrap(spline_ext::SplineExtrapolation, cumhaz_spline, lb, ub)
    bounds = BSplineKit.boundaries(spline_ext.spline.basis)
    t_lo, t_hi = bounds
    
    # Simple case: both endpoints within spline support
    if lb >= t_lo && ub <= t_hi
        return cumhaz_spline(ub) - cumhaz_spline(lb)
    end
    
    # Handle extrapolation
    H_total = zero(eltype(coefficients(spline_ext.spline)))
    
    # Contribution from below lower boundary
    if lb < t_lo
        h_lo = spline_ext(t_lo)  # Hazard at boundary (uses extrapolation method)
        if spline_ext.method isa BSplineKit.SplineExtrapolations.Flat
            # Flat: constant hazard below boundary
            H_total += h_lo * (min(ub, t_lo) - lb)
        else
            # Linear: hazard changes linearly (but still use spline_ext for value)
            H_total += h_lo * (min(ub, t_lo) - lb)
        end
    end
    
    # Contribution within spline support
    actual_lb = max(lb, t_lo)
    actual_ub = min(ub, t_hi)
    if actual_ub > actual_lb
        H_total += cumhaz_spline(actual_ub) - cumhaz_spline(actual_lb)
    end
    
    # Contribution from above upper boundary
    if ub > t_hi
        h_hi = spline_ext(t_hi)
        if spline_ext.method isa BSplineKit.SplineExtrapolations.Flat
            # Flat: constant hazard above boundary
            H_total += h_hi * (ub - max(lb, t_hi))
        else
            H_total += h_hi * (ub - max(lb, t_hi))
        end
    end
    
    return H_total
end

register_hazard_family!("exp", _build_exponential_hazard)
register_hazard_family!("wei", _build_weibull_hazard)
register_hazard_family!("gom", _build_gompertz_hazard)
register_hazard_family!("sp", _build_spline_hazard)


# mutable structs

#=============================================================================
PHASE 2: Consolidated build_hazards with Runtime Functions
=============================================================================# 

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
        builder === nothing && error("Unknown hazard family: $(ctx.family)")
        hazard_struct, hazpars = builder(ctx)
        _hazards[idx] = hazard_struct
        push!(parameters_list, hazpars)
    end

    parameters = build_parameters(parameters_list, hazkeys)
    return _hazards, parameters, hazkeys
end

"""
    build_parameters(parameters::Vector{Vector{Float64}}, hazkeys::Dict{Symbol, Int64})

Create a ParameterHandling.jl structure from nested parameter vectors.

Returns a NamedTuple with fields:
- `flat`: Vector{Float64} of all parameters in flattened form (transformed scale)
- `transformed`: NamedTuple of positive() transformations for each hazard
- `natural`: NamedTuple of Vectors for each hazard (natural scale)
- `unflatten`: Function to reconstruct from flat vector

Parameters are stored on the log scale using ParameterHandling.positive().
"""
function build_parameters(parameters::Vector{Vector{Float64}}, hazkeys::Dict{Symbol, Int64})
    # Use safe_positive to prevent ParameterHandling epsilon errors
    # Input parameters are on log scale, exp to get natural scale
    params_transformed_pairs = [
        hazname => safe_positive(exp.(parameters[idx]))
        for (hazname, idx) in sort(collect(hazkeys), by = x -> x[2])
    ]

    params_transformed = NamedTuple(params_transformed_pairs)
    
    # Flatten to get the flat vector and unflatten function
    params_flat, unflatten_fn = ParameterHandling.flatten(params_transformed)
    
    # Get natural scale parameters
    params_natural = ParameterHandling.value(params_transformed)
    
    # Return nested structure
    return (
        flat = params_flat,
        transformed = params_transformed,
        natural = params_natural,
        unflatten = unflatten_fn
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
            error("EmissionMatrix must have dimensions ($(n_obs), $(n_states)), got $(size(EmissionMatrix)).")
        end
        if any(EmissionMatrix .< 0) || any(EmissionMatrix .> 1)
            error("EmissionMatrix values must be in [0, 1].")
        end
        for i in 1:n_obs
            if all(EmissionMatrix[i,:] .== 0)
                error("EmissionMatrix row $i has no allowed states (all zeros).")
            end
        end
        return EmissionMatrix
    end
    
    # Otherwise, build from CensoringPatterns
    emat = zeros(Float64, n_obs, n_states)

    for i in 1:n_obs
        if data.obstype[i] ∈ [1, 2] # observation not censored
            emat[i,data.stateto[i]] = 1.0
        elseif data.obstype[i] == 0 # observation censored, all states are possible
            emat[i,:] .= 1.0
        else
            emat[i,:] .= CensoringPatterns[data.obstype[i] - 2, 2:n_states+1]
        end 
    end

    return emat
end

@inline function _ensure_subject_weights(SubjectWeights, nsubj)
    return isnothing(SubjectWeights) ? ones(Float64, nsubj) : SubjectWeights
end

@inline function _ensure_observation_weights(ObservationWeights)
    return ObservationWeights  # nothing stays nothing, vector stays vector
end

@inline function _prepare_censoring_patterns(CensoringPatterns, n_states)
    return isnothing(CensoringPatterns) ? Matrix{Float64}(undef, 0, n_states) : Float64.(CensoringPatterns)
end

function _validate_inputs!(data::DataFrame,
                           tmat::Matrix{Int64},
                           CensoringPatterns::Matrix{Float64},
                           SubjectWeights,
                           ObservationWeights;
                           verbose::Bool)
    check_data!(data, tmat, CensoringPatterns; verbose = verbose)
    
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
    return all(isa.(hazards, _MarkovHazard)) ? :markov : :semi_markov
end

@inline function _model_constructor(mode::Symbol, process::Symbol)
    if mode == :exact
        return MultistateModel
    elseif mode == :panel
        return process == :markov ? MultistateMarkovModel : MultistateSemiMarkovModel
    elseif mode == :censored
        return process == :markov ? MultistateMarkovModelCensored : MultistateSemiMarkovModelCensored
    else
        error("Unknown observation mode $(mode)")
    end
end

function _assemble_model(mode::Symbol,
                         process::Symbol,
                         components::NamedTuple,
                         surrogate::Union{Nothing, MarkovSurrogate},
                         modelcall)
    ctor = _model_constructor(mode, process)
    return ctor(
        components.data,
        components.parameters,
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
        modelcall,
    )
end

"""
    multistatemodel(hazards::HazardFunction...; data::DataFrame, surrogate = :none, ...)

Construct a full multistate model from a collection of hazards defined via `Hazard`
or `@hazard`. Hazards without covariates can omit a `@formula` entirely; the helper
will insert the intercept-only design automatically as described in `Hazard`'s docs.

# Keywords
- `data`: long-format `DataFrame` with at least `:subject`, `:statefrom`, `:stateto`, `:time`, `:obstype`.
- `surrogate::Symbol = :none`: surrogate model for importance sampling in MCEM.
  - `:none` (default): no surrogate created (for Markov models or exact data)
  - `:markov`: create a Markov surrogate (required for semi-Markov MCEM fitting)
- `optimize_surrogate::Bool = false`: if `surrogate = :markov`, fit the surrogate via MLE at model creation time.
  If `false`, surrogate will be fitted when `fit()` is called (default behavior).
- `surrogate_constraints`: optional constraints for surrogate optimization (only used if `optimize_surrogate = true`).
- `SubjectWeights`: optional per-subject weights (length = number of subjects). Mutually exclusive with `ObservationWeights`.
- `ObservationWeights`: optional per-observation weights (length = number of rows in data). Mutually exclusive with `SubjectWeights`.
- `CensoringPatterns`: optional matrix describing which states are compatible with each censoring code. Values in [0,1].
- `EmissionMatrix`: optional matrix of emission probabilities (nrow(data) × nstates). Values are P(observation|state).
- `verbose`: print additional validation output.

# Examples
```julia
# Markov model (no surrogate needed)
model = multistatemodel(h12, h21; data = df)

# Semi-Markov model - surrogate will be created and fitted when fit() is called
model = multistatemodel(h12_wei, h21_wei; data = df, surrogate = :markov)

# Semi-Markov model - pre-fit surrogate at model creation time
model = multistatemodel(h12_wei, h21_wei; data = df, surrogate = :markov, optimize_surrogate = true)
```
"""
function multistatemodel(hazards::HazardFunction...; 
                        data::DataFrame, 
                        surrogate::Symbol = :none,
                        optimize_surrogate::Bool = false,
                        surrogate_constraints = nothing,
                        SubjectWeights::Union{Nothing,Vector{Float64}} = nothing, 
                        ObservationWeights::Union{Nothing,Vector{Float64}} = nothing,
                        CensoringPatterns::Union{Nothing,Matrix{<:Real}} = nothing, 
                        EmissionMatrix::Union{Nothing,Matrix{Float64}} = nothing,
                        verbose = false) 

    # Validate surrogate option
    if surrogate ∉ (:none, :markov)
        error("surrogate must be :none or :markov, got :$surrogate")
    end
    
    # Validate inputs
    isempty(hazards) && throw(ArgumentError("At least one hazard must be provided"))

    # catch the model call
    modelcall = (hazards = hazards, data = data, SubjectWeights = SubjectWeights, ObservationWeights = ObservationWeights, CensoringPatterns = CensoringPatterns, EmissionMatrix = EmissionMatrix)

    # get indices for each subject in the dataset
    subjinds, nsubj = get_subjinds(data)

    # enumerate the hazards and reorder 
    hazinfo = enumerate_hazards(hazards...)

    # reorder hazards and pop the order column in hazinfo
    hazards = hazards[hazinfo.order]
    select!(hazinfo, Not(:order))

    # compile matrix enumerating instantaneous state transitions
    tmat = create_tmat(hazinfo)

    # Handle weight exclusivity and defaults
    SubjectWeights, ObservationWeights = check_weight_exclusivity(SubjectWeights, ObservationWeights, nsubj)
    
    # Prepare patterns
    CensoringPatterns = _prepare_censoring_patterns(CensoringPatterns, size(tmat, 1))

    _validate_inputs!(data, tmat, CensoringPatterns, SubjectWeights, ObservationWeights; verbose = verbose)
    emat = build_emat(data, CensoringPatterns, EmissionMatrix, tmat)

    _hazards, parameters, hazkeys = build_hazards(hazards...; data = data, surrogate = false)
    _totalhazards = build_totalhazards(_hazards, tmat)

    # Build surrogate if requested
    if surrogate === :markov
        surrogate_haz, surrogate_pars_ph, _ = build_hazards(hazards...; data = data, surrogate = true)
        markov_surrogate = MarkovSurrogate(surrogate_haz, surrogate_pars_ph)
    else
        markov_surrogate = nothing
    end

    components = (
        data = data,
        parameters = parameters,
        hazards = _hazards,
        totalhazards = _totalhazards,
        tmat = tmat,
        emat = emat,
        hazkeys = hazkeys,
        subjinds = subjinds,
        SubjectWeights = SubjectWeights,
        ObservationWeights = ObservationWeights,
        CensoringPatterns = CensoringPatterns,
    )

    mode = _observation_mode(data)
    process = _process_class(_hazards)

    model = _assemble_model(mode, process, components, markov_surrogate, modelcall)
    
    # Optionally fit surrogate at model creation time
    if optimize_surrogate && surrogate === :markov
        if verbose
            println("Fitting Markov surrogate at model creation time...")
        end
        surrogate_fitted = fit_surrogate(model; surrogate_constraints = surrogate_constraints, verbose = verbose)
        model.markovsurrogate = MarkovSurrogate(surrogate_fitted.hazards, surrogate_fitted.parameters)
    end
    
    return model
end