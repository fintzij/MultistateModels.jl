# hazard_constructors.jl - User-facing Hazard() constructor and input validation
#
# This file contains:
# - Hazard() constructor (main entry point for defining hazards)
# - Input validation for hazard specifications
# - enumerate_hazards() and create_tmat() for state space setup

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
- `coxian_structure::Symbol`: Coxian structure constraint for phase-type hazards
    (only for `family == :pt`, default: `:sctp`).
    - `:sctp` (default): SCTP (Stationary Conditional Transition Probability) constraint with
      eigenvalue ordering ν₁ ≤ ν₂ ≤ ... ≤ νₙ. Ensures P(destination | leaving) is constant
      across phases and provides identifiability through eigenvalue ordering.
    - `:unstructured`: All progression and exit rates are free parameters (not recommended,
      may have identifiability issues).
- `covariate_constraints::Symbol`: Controls covariate parameter sharing for phase-type
    hazards (only for `family == :pt`, default: `:homogeneous`).
    - `:homogeneous` (default): Destination-specific shared effects. All phases share the same
      covariate effect per destination (e.g., `h12_age`). Parameter count: p.
      **Recommended for better identifiability.**
    - `:unstructured`: Phase-specific covariate effects. Each phase has independent
      covariate parameters (e.g., `h12_a_age`, `h12_b_age`). Parameter count: n × p.
- `time_transform::Bool`: enable time transformation shared-trajectory caching for this transition.
- `linpred_effect::Symbol`: `:ph` (default) for proportional hazards or `:aft` for
    accelerated-failure-time behaviour.

# Examples
```julia
julia> Hazard(:exp, 1, 2)                        # intercept only
julia> Hazard(@formula(0 ~ age + trt), :wei, 1, 3)
julia> Hazard(:pt, 1, 2)                         # phase-type with defaults (sctp + homogeneous)
julia> Hazard(@formula(0 ~ age), :pt, 1, 2)      # phase-type with homogeneous covariates (default)
julia> Hazard(@formula(0 ~ age), :pt, 1, 2; covariate_constraints=:unstructured)  # phase-specific
julia> Hazard(:pt, 1, 2; coxian_structure=:unstructured)  # override to unstructured (not recommended)
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
    coxian_structure::Symbol = :sctp,
    covariate_constraints::Symbol = :homogeneous,
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
        coxian_structure in (:unstructured, :sctp) || 
            throw(ArgumentError("coxian_structure must be :unstructured or :sctp, got :$coxian_structure"))
        covariate_constraints in (:unstructured, :homogeneous) || throw(ArgumentError("covariate_constraints must be :unstructured or :homogeneous, got :$covariate_constraints"))
    end
    
    linpred_effect in (:ph, :aft) || throw(ArgumentError("linpred_effect must be :ph or :aft, got :$linpred_effect"))
    
    metadata = HazardMetadata(time_transform = time_transform, linpred_effect = linpred_effect)

    if family_key == :pt
        # Phase-type (Coxian) hazard
        h = PhaseTypeHazard(hazard, family_key, statefrom, stateto, n_phases, coxian_structure, covariate_constraints, metadata)
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
