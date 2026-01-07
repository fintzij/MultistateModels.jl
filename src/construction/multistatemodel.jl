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
- `family`: `:exp`, `:wei`, `:gom`, `:sp`, or `:pt` (symbol or string, case-insensitive).
  - `:exp`: Exponential (constant hazard)
  - `:wei`: Weibull
  - `:gom`: Gompertz
  - `:sp`: Spline (flexible baseline hazard)
  - `:pt`: Phase-type (Coxian) - flexible sojourn distribution via latent phases
- `statefrom` / `stateto`: integers describing the transition.
- `hazard` *(optional)*: a `StatsModels.FormulaTerm` describing covariates that act
    multiplicatively on the baseline hazard. Skip this argument for intercept-only hazards.

# Keyword arguments
- `degree`, `knots`, `boundaryknots`, `natural_spline`, `extrapolation`, `monotone`:
    spline controls used only when `family == :sp`. See the BSplineKit docs for details.
- `monotone`: `0` (default) leaves the spline unconstrained, `1` enforces an increasing
    hazard, and `-1` enforces a decreasing hazard.
- `n_phases::Int`: Number of Coxian phases (only for `family == :pt`, default: 2).
    Must be ≥ 1. **Prefer specifying `n_phases` at model construction** via 
    `multistatemodel(...; n_phases = Dict(state => k))`. When specified in Hazard(), 
    it serves as a fallback if not provided at model level.
- `time_transform::Bool`: enable time transformation shared-trajectory caching for this transition.
- `linpred_effect::Symbol`: `:ph` (default) for proportional hazards or `:aft` for
    accelerated-failure-time behaviour.

# Examples
```julia
julia> Hazard(:exp, 1, 2)                        # intercept only
julia> Hazard(@formula(0 ~ age + trt), :wei, 1, 3)
julia> Hazard(:pt, 1, 2)                         # phase-type (n_phases set at model level)
julia> Hazard(@formula(0 ~ age), :pt, 1, 2)      # phase-type with covariates
julia> @hazard begin                              # macro front-end uses the same rules
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
    extrapolation = "constant",
    monotone = 0,
    n_phases::Int = 2,
    coxian_structure::Symbol = :unstructured,
    time_transform::Bool = false,
    linpred_effect::Symbol = :ph)

    # Input validation - use ArgumentError for user-facing validation
    statefrom > 0 || throw(ArgumentError("statefrom must be a positive integer, got $statefrom"))
    stateto > 0 || throw(ArgumentError("stateto must be a positive integer, got $stateto"))
    statefrom != stateto || throw(ArgumentError("statefrom and stateto must differ (got $statefrom → $stateto)"))
    
    # Normalize family to Symbol (accept both String and Symbol for backward compatibility)
    family_key = family isa Symbol ? family : Symbol(lowercase(String(family)))
    
    valid_families = (:exp, :wei, :gom, :sp, :pt)
    family_key in valid_families || throw(ArgumentError("family must be one of $valid_families, got :$family_key"))
    
    if family_key == :sp
        degree >= 0 || throw(ArgumentError("spline degree must be non-negative, got $degree"))
        extrapolation in ("linear", "constant", "flat") || throw(ArgumentError("extrapolation must be \"linear\", \"constant\", or \"flat\", got \"$extrapolation\""))
        monotone in (-1, 0, 1) || throw(ArgumentError("monotone must be -1, 0, or 1, got $monotone"))
        # For degree < 2, constant extrapolation (C1 boundary) is impossible.
        # Default to flat extrapolation (C0 boundary) which is usually desired.
        if extrapolation == "constant" && degree < 2
            extrapolation = "flat"
        end
    end
    
    if family_key == :pt
        n_phases >= 1 || throw(ArgumentError("n_phases must be ≥ 1, got $n_phases"))
        coxian_structure in (:unstructured, :sctp) || throw(ArgumentError("coxian_structure must be :unstructured or :sctp, got :$coxian_structure"))
    end
    
    linpred_effect in (:ph, :aft) || throw(ArgumentError("linpred_effect must be :ph or :aft, got :$linpred_effect"))
    
    metadata = HazardMetadata(time_transform = time_transform, linpred_effect = linpred_effect)

    if family_key == :pt
        # Phase-type (Coxian) hazard
        h = PhaseTypeHazard(hazard, family_key, statefrom, stateto, n_phases, coxian_structure, metadata)
    elseif family_key != :sp
        h = ParametricHazard(hazard, family_key, statefrom, stateto, metadata)
    else 
        if natural_spline & (monotone != 0)
            @info "Natural boundary conditions are not currently compatible with monotone splines. The restrictions on second derivatives at the spline boundaries will be removed."
            natural_spline = false
        end

        # For degree < 2, constant extrapolation (C1 boundary) is impossible.
        # Fall back to flat extrapolation which is usually desired.
        if degree < 2 && extrapolation == "constant"
            extrapolation = "flat"
        end
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
    # Pass through to main constructor which handles String/Symbol normalization
    return Hazard(_DEFAULT_HAZARD_FORMULA, family, Int(statefrom), Int(stateto); kwargs...)
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
        throw(ArgumentError("Duplicate transitions detected: $(duplicates). Each transition (statefrom → stateto) should be specified only once."))
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
    schema = StatsModels.schema(hazard.hazard, data)
    hazschema = apply_schema(hazard.hazard, schema)
    modelcols(hazschema, data) # validate design matrix construction
    rhs_names = _hazard_rhs_names(hazschema)
    # Normalize family to Symbol (hazard.family is now Symbol)
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
    
    # Track whether user provided explicit boundary knots
    user_provided_boundaries = hazard.boundaryknots !== nothing
    
    # Validate interior knots are within boundaries
    # If user provided both boundaries and interior knots that exceed them, warn
    # If boundaries were inferred, silently extend to encompass interior knots
    if !isempty(intknots)
        needs_adjustment = any(intknots .< bknots[1]) || any(intknots .> bknots[2])
        if needs_adjustment
            if user_provided_boundaries
                @warn "Interior knots outside user-specified boundary knots. Adjusting boundaries."
            end
            # Silently extend boundaries if they were inferred
            bknots[1] = min(bknots[1], minimum(intknots))
            bknots[2] = max(bknots[2], maximum(intknots))
        end
    end
    
    # Combine and sort knots
    allknots = unique(sort([bknots[1]; intknots; bknots[2]]))
    
    # Build B-spline basis
    B = BSplineBasis(BSplineOrder(hazard.degree + 1), copy(allknots))
    
    # Determine if we need smooth constant extrapolation
    # "constant" enforces h'=0 at boundaries for C¹ continuity with flat extrapolation
    use_constant = hazard.extrapolation == "constant"
    
    # Apply boundary conditions via basis recombination
    if use_constant && (hazard.degree >= 2)
        # constant: enforce D¹=0 (Neumann BC) at both boundaries
        # This gives C¹ continuity when extending as constant beyond boundaries
        # The hazard approaches the boundary tangentially (zero slope), ensuring
        # a smooth transition to the constant extrapolation region.
        B = RecombinedBSplineBasis(B, Derivative(1))
    elseif (hazard.degree > 1) && hazard.natural_spline
        # Natural spline: D²=0 at boundaries only
        B = RecombinedBSplineBasis(B, Natural())
    end
    
    # Determine extrapolation method
    # "constant" uses Flat() with the Neumann BC basis above for C¹ continuity
    # "linear" uses Linear() for slope-based extrapolation
    extrap_method = if hazard.extrapolation == "linear"
        BSplineKit.SplineExtrapolations.Linear()
    else
        # "constant" uses Flat() - the smooth basis above ensures C¹ continuity
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
    smooth_info = _extract_smooth_info(ctx, parnames)
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
        hazard.extrapolation,
        ctx.metadata,
        ctx.shared_baseline_key,
        smooth_info,
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
    # Extract covariate names (if any)
    covar_names = extract_covar_names(parnames)
    has_covars = !isempty(covar_names)
    
    # Capture spline infrastructure in closures
    # Note: basis and extrap_method are immutable, safe to capture
    
    # Capture spline infrastructure in closures
    # Note: basis and extrap_method are immutable, safe to capture
    
    hazard_fn = let B = basis, ext = extrap_method, mono = monotone, nb = nbasis, has_cov = has_covars, effect = linpred_effect
        function(t, pars, covars)
            # Handle both vector and NamedTuple parameter formats
            if pars isa AbstractVector
                # Legacy vector format: first nbasis elements are spline coefficients
                spline_coefs_vec = pars[1:nb]
                covar_pars = has_cov ? pars[(nb+1):end] : Float64[]
            else
                # NamedTuple format
                spline_coefs_vec = collect(values(pars.baseline))
                covar_pars = has_cov ? pars.covariates : NamedTuple()
            end
            
            # Transform parameters to spline coefficients
            coefs = _spline_ests2coefs(spline_coefs_vec, B, mono)
            # Build spline on-the-fly
            spline = Spline(B, coefs)
            spline_ext = SplineExtrapolation(spline, ext)
            
            # Apply covariate effect - only if covariates actually provided
            n_covars = covars isa AbstractVector ? length(covars) : length(covars)
            if has_cov && n_covars > 0
                linear_pred = pars isa AbstractVector ? 
                              dot(collect(covars), covar_pars) : 
                              _eval_linear_pred_named(covar_pars, covars)
            else
                linear_pred = 0.0
            end

            if effect == :aft
                scale = exp(-linear_pred)
                h0 = spline_ext(t * scale)
                return h0 * scale
            else
                h0 = spline_ext(t)
                return h0 * exp(linear_pred)
            end
        end
    end
    
    cumhaz_fn = let B = basis, ext = extrap_method, mono = monotone, nb = nbasis, has_cov = has_covars, effect = linpred_effect
        function(lb, ub, pars, covars)
            # Handle both vector and NamedTuple parameter formats
            if pars isa AbstractVector
                # Legacy vector format
                spline_coefs_vec = pars[1:nb]
                covar_pars = has_cov ? pars[(nb+1):end] : Float64[]
            else
                # NamedTuple format
                spline_coefs_vec = collect(values(pars.baseline))
                covar_pars = has_cov ? pars.covariates : NamedTuple()
            end
            
            # Transform parameters to spline coefficients
            coefs = _spline_ests2coefs(spline_coefs_vec, B, mono)
            # Build spline and its integral on-the-fly
            spline = Spline(B, coefs)
            spline_ext = SplineExtrapolation(spline, ext)
            cumhaz_spline = integral(spline_ext.spline)
            
            # Apply covariate effect - only if covariates actually provided
            n_covars_provided = covars isa NamedTuple ? length(covars) : length(covars)
            if n_covars_provided > 0
                linear_pred = pars isa AbstractVector ? 
                              dot(collect(covars), covar_pars) : 
                              _eval_linear_pred_named(covar_pars, covars)
            else
                linear_pred = 0.0
            end

            if effect == :aft
                scale = exp(-linear_pred)
                H0 = _eval_cumhaz_with_extrap(spline_ext, cumhaz_spline, lb * scale, ub * scale)
                return H0
            else
                H0 = _eval_cumhaz_with_extrap(spline_ext, cumhaz_spline, lb, ub)
                return H0 * exp(linear_pred)
            end
        end
    end
    
    return hazard_fn, cumhaz_fn
end

"""
    _spline_ests2coefs(ests, basis, monotone)

Transform spline parameter estimates to spline coefficients.

For monotone == 0: identity (parameters ARE coefficients, non-negativity via box constraints)
For monotone == 1: I-spline-like cumulative sum (increasing hazard)
For monotone == -1: reverse cumulative sum (decreasing hazard)

Note: Parameters are now on NATURAL scale (not log). Box constraints ensure β ≥ 0.
"""
function _spline_ests2coefs(ests::AbstractVector{T}, basis, monotone::Int) where T
    if monotone == 0
        # Non-negative coefficients directly (box constraints ensure ≥ 0)
        return ests
    else
        # I-spline transformation for monotonicity
        # ests are non-negative increments (box constrained to ≥ 0)
        coefs = zeros(T, length(ests))
        
        if length(coefs) > 1
            k = BSplineKit.order(basis)
            t = BSplineKit.knots(basis)
            
            for i in 2:length(coefs)
                coefs[i] = coefs[i-1] + ests[i] * (t[i + k] - t[i]) / k
            end
        end
        
        # Add intercept
        coefs .+= ests[1]
        
        # Reverse for decreasing monotone
        if monotone == -1
            reverse!(coefs)
        end
        
        return coefs
    end
end

"""
    _spline_coefs2ests(coefs, basis, monotone; clamp_zeros=false)

Transform spline coefficients back to natural-scale parameter estimates.
Inverse of `_spline_ests2coefs`.

For monotone == 0: identity (coefficients ARE parameters)
For monotone == 1: difference transformation (inverse of cumsum)
For monotone == -1: reverse then difference

Note: Parameters are now on NATURAL scale (not log). Box constraints ensure β ≥ 0.
"""
function _spline_coefs2ests(coefs::AbstractVector{T}, basis, monotone::Int; clamp_zeros::Bool=false) where T
    if monotone == 0
        # Identity: coefficients are parameters directly
        return copy(coefs)
    else
        # Reverse if decreasing
        coefs_nat = monotone == 1 ? copy(coefs) : reverse(coefs)
        
        ests = zeros(T, length(coefs_nat))
        
        if length(coefs) > 1
            k = BSplineKit.order(basis)
            t = BSplineKit.knots(basis)
            
            # Inverse of the cumsum: take differences
            for i in length(ests):-1:2
                ests[i] = (coefs_nat[i] - coefs_nat[i - 1]) * k / (t[i + k] - t[i])
            end
        end
        
        # Intercept
        ests[1] = coefs_nat[1]
        
        # Clamp numerical zeros
        if clamp_zeros
            ests[findall(isapprox.(ests, 0.0; atol = sqrt(eps())))] .= zero(T)
        end
        
        return ests
    end
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
    _eval_linear_pred_named(pars_covariates::NamedTuple, covars::NamedTuple)

Evaluate the linear predictor β'x using named parameter and covariate access.
Assumes parameter names match covariate names (with hazard prefix stripped).

# Example
```julia
pars.covariates = (h12_age = 0.3, h12_sex = 0.1)
covars = (age = 50.0, sex = 1.0)
result = _eval_linear_pred_named(pars.covariates, covars)  # 0.3*50 + 0.1*1
```
"""
function _eval_linear_pred_named(pars_covariates::NamedTuple, covars::NamedTuple)
    result = 0.0
    for (pname, pval) in pairs(pars_covariates)
        # Extract covariate name by stripping hazard prefix (h12_age → age)
        cname_str = replace(String(pname), r"^h\d+_" => "")
        cname = Symbol(cname_str)
        if haskey(covars, cname)
            result += pval * getfield(covars, cname)
        else
            throw(ArgumentError("Covariate $cname not found in covars NamedTuple. Available: $(keys(covars))"))
        end
    end
    return result
end

"""
    _eval_cumhaz_with_extrap(spline_ext, cumhaz_spline, lb, ub)

Evaluate cumulative hazard over [lb, ub] with proper extrapolation handling.

For "constant" extrapolation: hazard is constant beyond boundaries (with C¹ smooth
transition via Neumann BC), so cumulative hazard grows linearly.

For "linear" extrapolation: hazard varies linearly beyond boundaries:
  h(t) = h(t_b) + h'(t_b) * (t - t_b)
so cumulative hazard is:
  H(a,b) = h(t_b) * Δt + h'(t_b) * Δt² / 2  (for t > t_hi)
  H(a,b) = h(t_b) * Δt - h'(t_b) * Δt² / 2  (for t < t_lo)
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
    is_linear = spline_ext.method isa BSplineKit.SplineExtrapolations.Linear
    
    # Contribution from below lower boundary
    if lb < t_lo
        h_lo = spline_ext.spline(t_lo)  # Hazard at lower boundary
        dt = min(ub, t_lo) - lb
        if is_linear
            # Linear extrapolation: h(t) = h(t_lo) + h'(t_lo) * (t - t_lo) for t < t_lo
            # ∫[lb, t_lo] h(t) dt = h(t_lo) * dt + h'(t_lo) * ∫[lb, t_lo] (t - t_lo) dt
            # The integral ∫[lb, t_lo] (t - t_lo) dt = -dt²/2 (since t - t_lo < 0 in this region)
            dh_lo = ForwardDiff.derivative(t -> spline_ext.spline(t), t_lo)
            H_total += h_lo * dt - dh_lo * dt^2 / 2
        else
            # Constant: hazard stays at boundary value (C¹ continuous for "constant" mode)
            H_total += h_lo * dt
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
        h_hi = spline_ext.spline(t_hi)  # Hazard at upper boundary
        dt = ub - max(lb, t_hi)
        if is_linear
            # Linear extrapolation: h(t) = h(t_hi) + h'(t_hi) * (t - t_hi) for t > t_hi
            # ∫[t_hi, ub] h(t) dt = h(t_hi) * dt + h'(t_hi) * dt²/2
            dh_hi = ForwardDiff.derivative(t -> spline_ext.spline(t), t_hi)
            H_total += h_hi * dt + dh_hi * dt^2 / 2
        else
            # Constant: hazard stays at boundary value (C¹ continuous for "constant" mode)
            H_total += h_hi * dt
        end
    end
    
    return H_total
end

register_hazard_family!(:exp, _build_exponential_hazard)
register_hazard_family!(:wei, _build_weibull_hazard)
register_hazard_family!(:gom, _build_gompertz_hazard)
register_hazard_family!(:sp, _build_spline_hazard)


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
        builder === nothing && throw(ArgumentError("Unknown hazard family: $(ctx.family). Supported families: $(keys(_HAZARD_BUILDERS))"))
        hazard_struct, hazpars = builder(ctx)
        _hazards[idx] = hazard_struct
        push!(parameters_list, hazpars)
    end

    parameters = build_parameters(parameters_list, hazkeys, _hazards)
    return _hazards, parameters, hazkeys
end

"""
    build_parameters(parameters::Vector{Vector{Float64}}, hazkeys::Dict{Symbol, Int64}, hazards::Vector{<:_Hazard})

Create a parameters structure from nested parameter vectors.

Returns a NamedTuple with fields:
- `flat`: Vector{Float64} of all parameters (log scale for baseline, as-is for covariates)
- `nested`: NamedTuple of NamedTuples per hazard with `baseline` and optional `covariates` fields
- `natural`: NamedTuple of Vectors for each hazard (natural scale for baseline, as-is for covariates)
- `unflatten`: Function to reconstruct nested structure from flat vector

Parameters are stored on log scale for baseline (rates, shapes, scales) because hazard 
functions expect log-scale and apply exp() internally. Covariate coefficients are unconstrained.
"""
function build_parameters(parameters::Vector{Vector{Float64}}, hazkeys::Dict{Symbol, Int64}, hazards::Vector{<:_Hazard})
    # Build nested parameters structure per hazard
    # Parameters are stored on log scale for baseline, as-is for covariates
    params_nested_pairs = [
        begin
            # Robustly find hazard by name
            h_idx = findfirst(h -> h.hazname == hazname, hazards)
            if isnothing(h_idx)
                throw(ArgumentError("Hazard $hazname not found in hazards vector. Available: $(getfield.(hazards, :hazname))"))
            end
            hazard = hazards[h_idx]
            hazname => build_hazard_params(parameters[idx], hazard.parnames, hazard.npar_baseline, hazard.npar_total)
        end
        for (hazname, idx) in sort(collect(hazkeys), by = x -> x[2])
    ]
    params_nested = NamedTuple(params_nested_pairs)
    
    # Create ReConstructor for AD-compatible flatten/unflatten
    reconstructor = ReConstructor(params_nested, unflattentype=UnflattenFlexible())
    params_flat = flatten(reconstructor, params_nested)
    
    # Get natural scale parameters (family-aware transformation)
    params_natural_pairs = [
        begin
            h_idx = findfirst(h -> h.hazname == hazname, hazards)
            hazard = hazards[h_idx]
            hazname => extract_natural_vector(params_nested[hazname], hazard.family)
        end
        for (hazname, idx) in sort(collect(hazkeys), by = x -> x[2])
    ]
    params_natural = NamedTuple(params_natural_pairs)
    
    # Return structure
    return (
        flat = params_flat,
        nested = params_nested,
        natural = params_natural,
        reconstructor = reconstructor  # NEW: Store ReConstructor instead of unflatten_fn
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
            throw(ArgumentError("EmissionMatrix must have dimensions ($(n_obs), $(n_states)), got $(size(EmissionMatrix))."))
        end
        if any(EmissionMatrix .< 0) || any(EmissionMatrix .> 1)
            throw(ArgumentError("EmissionMatrix values must be in [0, 1]."))
        end
        for i in 1:n_obs
            if all(EmissionMatrix[i,:] .== 0)
                throw(ArgumentError("EmissionMatrix row $i has no allowed states (all zeros)."))
            end
        end
        return EmissionMatrix
    end
    
    # Otherwise, build from CensoringPatterns
    emat = zeros(Float64, n_obs, n_states)

    for i in 1:n_obs
        if data.obstype[i] ∈ [1, 2] # state known (exact or panel): stateto observed
            emat[i,data.stateto[i]] = 1.0
        elseif data.obstype[i] == 0 # state fully censored: all states possible
            emat[i,:] .= 1.0
        else # state partially censored (obstype > 2): use censoring pattern
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
    # Use the _is_markov_hazard helper for consistent classification
    # This correctly handles PhaseTypeCoxianHazard (Markov) and degree-0 splines
    return all(_is_markov_hazard.(hazards)) ? :markov : :semi_markov
end

@inline function _model_constructor(mode::Symbol, process::Symbol)
    # All unfitted models use MultistateModel now
    # Behavior is determined by content (hazards, observation type), not struct type
    return MultistateModel
end

function _assemble_model(mode::Symbol,
                         process::Symbol,
                         components::NamedTuple,
                         surrogate::Union{Nothing, MarkovSurrogate},
                         modelcall;
                         phasetype_expansion::Union{Nothing, PhaseTypeExpansion} = nothing)
    # Single MultistateModel struct handles all cases
    return MultistateModel(
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
        phasetype_expansion,
    )
end

"""
    multistatemodel(hazards::HazardFunction...; data::DataFrame, surrogate = :none, ...)

Construct a full multistate model from a collection of hazards defined via `Hazard`
or `@hazard`. Hazards without covariates can omit a `@formula` entirely; the helper
will insert the intercept-only design automatically as described in `Hazard`'s docs.

# Keywords
- `data`: long-format `DataFrame` with at least `:subject`, `:statefrom`, `:stateto`, `:time`, `:obstype`.
- `constraints`: optional parameter constraints (see `make_constraints`). When provided at model creation,
  constraints are validated at parameter setting time and used as defaults in `fit()`.
- `initialize::Bool = true`: whether to initialize parameters at model creation. Uses `:crude` method
  for Markov and phase-type models (crude rates from data), `:surrogate` method for semi-Markov panel
  models (simulate paths, fit to exact data). If `false`, parameters remain at defaults (all rates = 1).
  Set to `false` if you want to manually set parameters before fitting.
- `surrogate::Symbol = :none`: surrogate model for importance sampling in MCEM.
  - `:none` (default): no surrogate created (for Markov models or exact data)
  - `:markov`: create a Markov surrogate (required for semi-Markov MCEM fitting)
- `fit_surrogate::Bool = true`: if `surrogate = :markov`, fit the surrogate via MLE at model creation time.
  If `false`, surrogate parameters remain at default values and will be fitted when `fit()` is called.
  Setting to `true` (default) is recommended as it avoids redundant fitting during initialization.
- `surrogate_constraints`: optional constraints for surrogate optimization (only used if `fit_surrogate = true`).
- `n_phases::Union{Nothing, Dict{Int,Int}} = nothing`: number of phases per state for phase-type hazards.
  Only states with `:pt` hazards should be specified. Example: `Dict(1 => 3, 2 => 2)` means state 1 has
  3 phases and state 2 has 2 phases. If a state has `:pt` hazards but is not in the dict, an error is thrown.
  If `n_phases[s] == 1`, the phase-type is coerced to exponential internally.
- `coxian_structure::Symbol = :unstructured`: constraint structure for phase-type hazards.
  - `:unstructured` (default): no constraints on progression and absorption rates
  - `:sctp`: Stationary Conditional Transition Probability - ensures P(r→s | transition out of r) is 
    constant over time. Automatically generates constraints: `h_{r_j,s} = τ_{r_j} × h_{r_1,s}` where
    τ parameters (`log_phase_rb`, `log_phase_rc`, ...) are shared across destinations and estimated.
- `SubjectWeights`: optional per-subject weights (length = number of subjects). Mutually exclusive with `ObservationWeights`.
- `ObservationWeights`: optional per-observation weights (length = number of rows in data). Mutually exclusive with `SubjectWeights`.
- `CensoringPatterns`: optional matrix describing which states are compatible with each censoring code. Values in [0,1].
- `EmissionMatrix`: optional matrix of emission probabilities (nrow(data) × nstates). Values are P(observation|state).
- `verbose`: print additional validation output.

# Examples
```julia
# Markov model (no surrogate needed)
model = multistatemodel(h12, h21; data = df)

# Model with parameter constraints
cons = make_constraints(
    cons = [:(log_λ_12 == log_λ_21)],  # Equal rates
    lcons = [0.0],
    ucons = [0.0]
)
model = multistatemodel(h12, h21; data = df, constraints = cons)

# Semi-Markov model - surrogate created and fitted at model creation time (default)
model = multistatemodel(h12_wei, h21_wei; data = df, surrogate = :markov)

# Semi-Markov model - defer surrogate fitting to fit() call
model = multistatemodel(h12_wei, h21_wei; data = df, surrogate = :markov, fit_surrogate = false)

# Model without automatic initialization (set parameters manually)
model = multistatemodel(h12, h21; data = df, initialize = false)
set_parameters!(model, (h12 = [log(0.5)], h21 = [log(0.3)]))

# Phase-type model with 3 phases on state 1
h12 = Hazard(@formula(0 ~ 1 + x), "pt", 1, 2)
h13 = Hazard(@formula(0 ~ 1 + x), "pt", 1, 3)
model = multistatemodel(h12, h13; data = df, n_phases = Dict(1 => 3))

# Phase-type model with SCTP constraints
model = multistatemodel(h12, h13; data = df, n_phases = Dict(1 => 3), coxian_structure = :sctp)
```
"""
function multistatemodel(hazards::HazardFunction...; 
                        data::DataFrame, 
                        constraints = nothing,
                        initialize::Bool = true,
                        surrogate::Symbol = :none,
                        fit_surrogate::Bool = true,
                        surrogate_constraints = nothing,
                        n_phases::Union{Nothing, Dict{Int,Int}} = nothing,
                        coxian_structure::Symbol = :unstructured,
                        SubjectWeights::Union{Nothing,Vector{Float64}} = nothing, 
                        ObservationWeights::Union{Nothing,Vector{Float64}} = nothing,
                        CensoringPatterns::Union{Nothing,Matrix{<:Real}} = nothing, 
                        EmissionMatrix::Union{Nothing,Matrix{Float64}} = nothing,
                        verbose = false) 

    # Validate surrogate option
    if surrogate ∉ (:none, :markov)
        throw(ArgumentError("surrogate must be :none or :markov, got :$surrogate"))
    end
    
    # Validate coxian_structure
    if coxian_structure ∉ (:unstructured, :sctp)
        throw(ArgumentError("coxian_structure must be :unstructured or :sctp, got :$coxian_structure"))
    end
    
    # Validate inputs
    isempty(hazards) && throw(ArgumentError("At least one hazard must be provided"))

    # Check for phase-type hazards and route accordingly
    if any(h -> h isa PhaseTypeHazard, hazards)
        return _build_phasetype_model_from_hazards(hazards;
            data = data,
            constraints = constraints,
            initialize = initialize,
            n_phases = n_phases,
            coxian_structure = coxian_structure,
            SubjectWeights = SubjectWeights,
            ObservationWeights = ObservationWeights,
            CensoringPatterns = CensoringPatterns,
            EmissionMatrix = EmissionMatrix,
            verbose = verbose
        )
    end
    
    # Validate n_phases not specified for non-phase-type models
    if !isnothing(n_phases) && !isempty(n_phases)
        throw(ArgumentError("n_phases specified but no :pt hazards found. n_phases only applies to phase-type models."))
    end

    # catch the model call (includes constraints for use by fit())
    modelcall = (hazards = hazards, data = data, constraints = constraints, SubjectWeights = SubjectWeights, ObservationWeights = ObservationWeights, CensoringPatterns = CensoringPatterns, EmissionMatrix = EmissionMatrix)

    # Expand smooth term basis columns into data (make a copy to avoid mutating user data)
    data = copy(data)
    expand_all_smooth_terms!(data, hazards)

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

    # Build surrogate if requested (initially unfitted)
    if surrogate === :markov
        surrogate_haz, surrogate_pars_ph, _ = build_hazards(hazards...; data = data, surrogate = true)
        markov_surrogate = MarkovSurrogate(surrogate_haz, surrogate_pars_ph; fitted=false)
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
    
    # Initialize parameters (default: true)
    # Uses :auto method which selects :crude for Markov/phase-type, :surrogate for semi-Markov
    if initialize
        initialize_parameters!(model; constraints = constraints)
    end
    
    # Fit surrogate at model creation time (default: true)
    if fit_surrogate && surrogate === :markov
        if verbose
            println("Fitting Markov surrogate at model creation time...")
        end
        fitted_surrogate = _fit_markov_surrogate(model; 
            surrogate_constraints = surrogate_constraints, 
            verbose = verbose)
        model.markovsurrogate = fitted_surrogate
    end
    
    return model
end