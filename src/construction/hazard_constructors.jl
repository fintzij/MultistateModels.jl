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

# =============================================================================
# Error Message Helpers (M7_P1)
# =============================================================================

# Common typos and their corrections for hazard families
const _FAMILY_TYPO_MAP = Dict(
    :exponential => :exp,
    :expon => :exp,
    :weibull => :wei,
    :weib => :wei,
    :weibu => :wei,
    :gompertz => :gom,
    :gomp => :gom,
    :spline => :sp,
    :splines => :sp,
    :bspline => :sp,
    :phasetype => :pt,
    :phase_type => :pt,
    :coxian => :pt,
)

"""
    _suggest_family(family::Symbol) -> Vector{Symbol}

Suggest corrections for common typos in hazard family names.
Returns empty vector if no suggestion available.
"""
function _suggest_family(family::Symbol)
    # Check exact typo map
    haskey(_FAMILY_TYPO_MAP, family) && return [_FAMILY_TYPO_MAP[family]]
    
    # Check if it's a case variation
    family_lower = Symbol(lowercase(String(family)))
    haskey(_FAMILY_TYPO_MAP, family_lower) && return [_FAMILY_TYPO_MAP[family_lower]]
    
    # Check Levenshtein-like similarity (simple heuristic)
    valid = (:exp, :wei, :gom, :sp, :pt)
    family_str = String(family)
    for v in valid
        v_str = String(v)
        # If the family starts with the same letter and is short, suggest it
        if !isempty(family_str) && first(family_str) == first(v_str) && length(family_str) <= 6
            return [v]
        end
    end
    
    return Symbol[]
end

"""
    _suggest_extrapolation(extrap::AbstractString) -> Vector{String}

Suggest corrections for common typos in extrapolation method names.
"""
function _suggest_extrapolation(extrap::AbstractString)
    extrap_lower = lowercase(extrap)
    
    # Common typos
    typo_map = Dict(
        "const" => "constant",
        "c" => "constant", 
        "lin" => "linear",
        "l" => "linear",
        "f" => "flat",
        "zero" => "flat",
        "none" => "flat",
    )
    
    haskey(typo_map, extrap_lower) && return [typo_map[extrap_lower]]
    
    # Check prefixes
    for valid in ("constant", "linear", "flat")
        if startswith(valid, extrap_lower) && length(extrap_lower) >= 2
            return [valid]
        end
    end
    
    return String[]
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
- `extrapolation`: How the spline hazard behaves beyond boundary knots (default: `"constant"`).
    - `"constant"`: Hazard approaches boundary with zero slope (C¹ continuous) then extends flat.
      The Neumann boundary condition (h'=0) ensures smooth transition to constant extrapolation.
      Recommended for most applications.
    - `"flat"`: Hazard extends as constant beyond boundaries (C⁰ continuous only). The spline
      can have non-zero slope at boundaries, creating a potential kink. Use when full boundary
      flexibility is needed.
    - `"linear"`: Hazard extends linearly using the slope at the boundary.
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
    statefrom::Int,
    stateto::Int;
    degree::Int = 3,
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
    
    # Validate family with helpful error message (M7_P1)
    valid_families = (:exp, :wei, :gom, :sp, :pt)
    if family_key ∉ valid_families
        # Check for common typos and provide suggestions
        suggestions = _suggest_family(family_key)
        if !isempty(suggestions)
            throw(ArgumentError("Unknown hazard family :$family_key. Did you mean :$(suggestions[1])? Valid options: $valid_families"))
        else
            throw(ArgumentError("Unknown hazard family :$family_key. Valid options: $valid_families"))
        end
    end
    
    # Spline-specific validation (M5_P1)
    if family_key == :sp
        degree >= 0 || throw(ArgumentError("spline degree must be non-negative, got $degree. Use degree=3 for cubic splines (default)."))
        
        # Validate extrapolation
        if extrapolation ∉ ("linear", "constant", "flat")
            suggestions = _suggest_extrapolation(extrapolation)
            if !isempty(suggestions)
                throw(ArgumentError("Invalid extrapolation \"$extrapolation\". Did you mean \"$(suggestions[1])\"? Valid options: \"linear\", \"constant\", \"flat\""))
            else
                throw(ArgumentError("extrapolation must be \"linear\", \"constant\", or \"flat\", got \"$extrapolation\""))
            end
        end
        
        monotone in (-1, 0, 1) || throw(ArgumentError("monotone must be -1 (decreasing), 0 (unconstrained), or 1 (increasing), got $monotone"))
        
        # Validate knots if provided (empty vector is allowed - means boundary knots only)
        if !isnothing(knots) && knots isa Vector{Float64} && !isempty(knots)
            issorted(knots) || throw(ArgumentError("knots must be in strictly increasing order. Got: $knots"))
            allunique(knots) || throw(ArgumentError("knots must be distinct (no duplicates). Got: $knots"))
        end
        
        # Validate boundary knots if provided
        if !isnothing(boundaryknots)
            length(boundaryknots) == 2 || throw(ArgumentError("boundaryknots must have exactly 2 elements [lower, upper], got $(length(boundaryknots)) elements"))
            boundaryknots[1] < boundaryknots[2] || throw(ArgumentError("boundaryknots must be ordered [lower, upper] with lower < upper, got $(boundaryknots)"))
            
            # If both knots and boundaryknots provided, validate relationship
            if !isnothing(knots) && knots isa Vector{Float64} && !isempty(knots)
                if minimum(knots) < boundaryknots[1]
                    throw(ArgumentError("Interior knots must be within boundary knots. Knot $(minimum(knots)) < lower boundary $(boundaryknots[1])"))
                end
                if maximum(knots) > boundaryknots[2]
                    throw(ArgumentError("Interior knots must be within boundary knots. Knot $(maximum(knots)) > upper boundary $(boundaryknots[2])"))
                end
            end
        end
        
        # For degree < 2, constant extrapolation (C1 boundary) is impossible.
        # Default to flat extrapolation (C0 boundary) which is usually desired.
        if extrapolation == "constant" && degree < 2
            extrapolation = "flat"
        end
    end
    
    # Phase-type specific validation
    if family_key == :pt
        n_phases >= 1 || throw(ArgumentError("n_phases must be ≥ 1, got $n_phases. For single-phase, use :exp hazard instead."))
        # Valid structures: :unstructured (no constraints) or :sctp (eigenvalue ordering, increasing by default)
        # Note: :sctp_increasing and :sctp_decreasing were removed - :sctp now implies increasing eigenvalues
        if coxian_structure ∉ (:unstructured, :sctp)
            if coxian_structure in (:sctp_increasing, :sctp_decreasing)
                throw(ArgumentError("coxian_structure :$coxian_structure is deprecated. Use :sctp instead (implies eigenvalue ordering)."))
            else
                throw(ArgumentError("coxian_structure must be :unstructured or :sctp, got :$coxian_structure"))
            end
        end
        if covariate_constraints ∉ (:unstructured, :homogeneous)
            throw(ArgumentError("covariate_constraints must be :unstructured or :homogeneous, got :$covariate_constraints"))
        end
    end
    
    # Validate linpred_effect with helpful message
    if linpred_effect ∉ (:ph, :aft)
        throw(ArgumentError("linpred_effect must be :ph (proportional hazards) or :aft (accelerated failure time), got :$linpred_effect"))
    end
    
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
            statefrom = zeros(Int, n_haz),
            stateto = zeros(Int, n_haz),
            trans = zeros(Int, n_haz),
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
    tmat = zeros(Int, n_states, n_states)

    for i in axes(hazinfo, 1)
        tmat[hazinfo.statefrom[i], hazinfo.stateto[i]] = 
            hazinfo.trans[i]
    end

    return tmat
end
