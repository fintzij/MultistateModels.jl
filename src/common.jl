"""
    Abstract type for hazard functions. Subtypes are ParametricHazard or SplineHazard.
"""
abstract type HazardFunction end

"""
Abstract struct for internal _Hazard types.
"""
abstract type _Hazard end

"""
Abstract struct for internal Markov _Hazard types.
"""
abstract type _MarkovHazard <: _Hazard end

"""
Abstract struct for internal semi-Markov _Hazard types.
"""
abstract type _SemiMarkovHazard <: _Hazard end

"""
Abstract struct for internal spline _Hazard types.
"""
abstract type _SplineHazard <: _SemiMarkovHazard end

"""
Abstract type for total hazards.
"""
abstract type _TotalHazard end

# =============================================================================
# Hazard metadata shared across analytic families
# ============================================================================= #

const _VALID_LINPRED_EFFECTS = (:ph, :aft)

struct HazardMetadata
    time_transform::Bool
    linpred_effect::Symbol
    function HazardMetadata(time_transform::Bool, linpred_effect::Symbol)
        effect = Symbol(linpred_effect)
        effect in _VALID_LINPRED_EFFECTS ||
            throw(ArgumentError("linpred_effect must be one of $(_VALID_LINPRED_EFFECTS), got $(linpred_effect)."))
        return new(time_transform, effect)
    end
end

HazardMetadata(; time_transform::Bool = false, linpred_effect::Symbol = :ph) =
    HazardMetadata(time_transform, linpred_effect)

struct TimeTransformHazardKey{LinType,TimeType}
    linpred::LinType
    t::TimeType
end

struct TimeTransformCumulKey{LinType,TimeType}
    linpred::LinType
    lb::TimeType
    ub::TimeType
end

function Base.:(==)(a::TimeTransformHazardKey, b::TimeTransformHazardKey)
    return a.linpred == b.linpred && a.t == b.t
end

function Base.hash(a::TimeTransformHazardKey, h::UInt)
    return hash(a.t, hash(a.linpred, h))
end

function Base.:(==)(a::TimeTransformCumulKey, b::TimeTransformCumulKey)
    return a.linpred == b.linpred && a.lb == b.lb && a.ub == b.ub
end

function Base.hash(a::TimeTransformCumulKey, h::UInt)
    return hash(a.ub, hash(a.lb, hash(a.linpred, h)))
end

mutable struct TimeTransformCache{LinType,TimeType}
    hazard_values::Dict{TimeTransformHazardKey{LinType,TimeType}, LinType}
    cumulhaz_values::Dict{TimeTransformCumulKey{LinType,TimeType}, LinType}
end

function TimeTransformCache(::Type{LinType}, ::Type{TimeType}) where {LinType,TimeType}
    hazard_values = Dict{TimeTransformHazardKey{LinType,TimeType}, LinType}()
    cumulhaz_values = Dict{TimeTransformCumulKey{LinType,TimeType}, LinType}()
    return TimeTransformCache{LinType,TimeType}(hazard_values, cumulhaz_values)
end

"""
    SharedBaselineKey

Composite key describing a Tang-style shared baseline. Multiple hazards can share a
trajectory cache when both the origin state and the hashed baseline specification
match. The `baseline_signature` is a deterministic hash of spline degree, knot
locations, and any other parameters that influence α(t).
"""
struct SharedBaselineKey
    statefrom::Int
    baseline_signature::UInt64
end

"""
    SharedBaselineTable

Top-level registry for Tang shared trajectories. Maps `SharedBaselineKey`s to
`TimeTransformCache` instances that are shared by every hazard leaving a given
state when their baselines match.
"""
mutable struct SharedBaselineTable{LinType,TimeType}
    caches::Dict{SharedBaselineKey, TimeTransformCache{LinType,TimeType}}
end

function SharedBaselineTable(::Type{LinType}, ::Type{TimeType}) where {LinType,TimeType}
    caches = Dict{SharedBaselineKey, TimeTransformCache{LinType,TimeType}}()
    return SharedBaselineTable{LinType,TimeType}(caches)
end

mutable struct TimeTransformContext{LinType,TimeType}
    caches::Vector{Union{Nothing,TimeTransformCache{LinType,TimeType}}}
    shared_baselines::SharedBaselineTable{LinType,TimeType}
end

function TimeTransformContext(::Type{LinType}, ::Type{TimeType}, nhazards::Integer) where {LinType,TimeType}
    caches = Vector{Union{Nothing,TimeTransformCache{LinType,TimeType}}}(undef, nhazards)
    fill!(caches, nothing)
    shared_baselines = SharedBaselineTable(LinType, TimeType)
    return TimeTransformContext{LinType,TimeType}(caches, shared_baselines)
end

@inline function _time_column_eltype(time_data)
    time_data === nothing && return Float64
    if time_data isa AbstractVector
        return Base.nonmissingtype(eltype(time_data))
    else
        return Base.nonmissingtype(typeof(time_data))
    end
end

"""
    _param_scalar_eltype(pars)

Extract the scalar element type from nested parameter structures.
Handles NamedTuples (baseline/covariates structure), Vectors, and scalar values.

Returns `Float64` for nested NamedTuple structures (e.g., from get_hazard_params)
by recursively extracting the first numeric value's type.
"""
@inline function _param_scalar_eltype(pars)
    if pars isa AbstractVector
        return eltype(pars)
    elseif pars isa NamedTuple
        # Nested NamedTuple structure - extract first scalar
        first_val = first(values(pars))
        if first_val isa NamedTuple
            return _param_scalar_eltype(first_val)
        elseif first_val isa AbstractVector
            return eltype(first_val)
        else
            return typeof(first_val)
        end
    else
        return typeof(pars)
    end
end

"""
    maybe_time_transform_context(pars, subjectdata, hazards; time_column = :sojourn)

Return a `TimeTransformContext` when Tang transforms are enabled for at least one
hazard and caching remains active. Callers may override `time_column` when the
relevant observation durations live under a different column name. When no
hazards opt into time transforms (or caching is globally disabled), the
function returns `nothing` so downstream code can skip the shared-cache branch.
"""
function maybe_time_transform_context(pars,
                                      subjectdata,
                                      hazards::Vector{<:_Hazard};
                                      time_column::Symbol = :sojourn)
    _time_transform_cache_enabled() || return nothing
    idx = findfirst(h -> h.metadata.time_transform, hazards)
    idx === nothing && return nothing

    # Extract scalar element type from nested parameter structure
    lin_type = _param_scalar_eltype(pars[idx])
    time_data = (subjectdata !== nothing && hasproperty(subjectdata, time_column)) ?
        getproperty(subjectdata, time_column) : nothing
    time_type = _time_column_eltype(time_data)
    return TimeTransformContext(lin_type, time_type, length(hazards))
end

const _TIME_TRANSFORM_CACHE_ENABLED = Ref(true)

function enable_time_transform_cache!(flag::Bool)
    _TIME_TRANSFORM_CACHE_ENABLED[] = flag
    return _TIME_TRANSFORM_CACHE_ENABLED[]
end

_time_transform_cache_enabled() = _TIME_TRANSFORM_CACHE_ENABLED[]

# =============================================================================
# PHASE 2: Consolidated Hazard Types (3 types instead of 8)
# ============================================================================= #

"""
    MarkovHazard

Consolidated hazard type for Markov processes (time-homogeneous).
Supports exponential family hazards with optional covariates.

# Fields
- `hazname::Symbol`: Name identifier (e.g., :h12)
- `statefrom::Int64`: Origin state
- `stateto::Int64`: Destination state  
- `family::Symbol`: Distribution family (`:exp`)
- `parnames::Vector{Symbol}`: Parameter names
- `npar_baseline::Int64`: Number of baseline parameters (without covariates)
- `npar_total::Int64`: Total number of parameters (baseline + covariates)
- `hazard_fn`: Runtime-generated hazard function (t, pars, covars) -> Float64
- `cumhaz_fn`: Runtime-generated cumulative hazard function
- `has_covariates::Bool`: Whether covariates are present
- `covar_names::Vector{Symbol}`: Pre-extracted covariate names for fast lookup
- `metadata::HazardMetadata`: Tang/linpred metadata
- `shared_baseline_key::Union{Nothing,SharedBaselineKey}`: identifies Tang-sharable baselines
"""
struct MarkovHazard <: _MarkovHazard
    hazname::Symbol
    statefrom::Int64
    stateto::Int64
    family::Symbol
    parnames::Vector{Symbol}
    npar_baseline::Int64
    npar_total::Int64
    hazard_fn::Function
    cumhaz_fn::Function
    has_covariates::Bool
    covar_names::Vector{Symbol}
    metadata::HazardMetadata
    shared_baseline_key::Union{Nothing,SharedBaselineKey}
end

"""
    SemiMarkovHazard

Consolidated hazard type for semi-Markov processes (time-dependent).
Supports Weibull and Gompertz families with optional covariates.

# Fields
- `hazname::Symbol`: Name identifier (e.g., :h12)
- `statefrom::Int64`: Origin state
- `stateto::Int64`: Destination state
- `family::Symbol`: Distribution family (`:wei`, `:gom`)
- `parnames::Vector{Symbol}`: Parameter names
- `npar_baseline::Int64`: Number of baseline parameters (shape + scale)
- `npar_total::Int64`: Total number of parameters (baseline + covariates)
- `hazard_fn`: Runtime-generated hazard function (t, pars, covars) -> Float64
- `cumhaz_fn`: Runtime-generated cumulative hazard function
- `has_covariates::Bool`: Whether covariates are present
- `covar_names::Vector{Symbol}`: Pre-extracted covariate names for fast lookup
- `metadata::HazardMetadata`: Tang/linpred metadata
- `shared_baseline_key::Union{Nothing,SharedBaselineKey}`
"""
struct SemiMarkovHazard <: _SemiMarkovHazard
    hazname::Symbol
    statefrom::Int64
    stateto::Int64
    family::Symbol
    parnames::Vector{Symbol}
    npar_baseline::Int64
    npar_total::Int64
    hazard_fn::Function
    cumhaz_fn::Function
    has_covariates::Bool
    covar_names::Vector{Symbol}
    metadata::HazardMetadata
    shared_baseline_key::Union{Nothing,SharedBaselineKey}
end

"""
    RuntimeSplineHazard

Internal hazard type for spline-based hazards (runtime evaluation).
Uses B-spline basis functions for flexible baseline hazard.

This is the internal/runtime version - users specify hazards using
`SplineHazard <: HazardFunction` which is converted to this type
during model construction.

# Fields
- `hazname::Symbol`: Name identifier (e.g., :h12)
- `statefrom::Int64`: Origin state
- `stateto::Int64`: Destination state
- `family::Symbol`: Always `:sp` for splines
- `parnames::Vector{Symbol}`: Parameter names
- `npar_baseline::Int64`: Number of spline coefficients
- `npar_total::Int64`: Total number of parameters (spline + covariates)
- `hazard_fn`: Runtime-generated hazard function (t, pars, covars) -> Float64
- `cumhaz_fn`: Runtime-generated cumulative hazard function
- `has_covariates::Bool`: Whether covariates are present
- `covar_names::Vector{Symbol}`: Pre-extracted covariate names for fast lookup
- `degree::Int64`: Spline degree
- `knots::Vector{Float64}`: Knot locations
- `natural_spline::Bool`: Natural spline constraint
- `monotone::Int64`: Monotonicity constraint (0, -1, 1)
- `extrapolation::String`: Extrapolation method ("flat", "linear", or "survextrap")
- `metadata::HazardMetadata`: Tang/linpred metadata
- `shared_baseline_key::Union{Nothing,SharedBaselineKey}`
"""
struct RuntimeSplineHazard <: _SplineHazard
    hazname::Symbol
    statefrom::Int64
    stateto::Int64
    family::Symbol
    parnames::Vector{Symbol}
    npar_baseline::Int64
    npar_total::Int64
    hazard_fn::Function
    cumhaz_fn::Function
    has_covariates::Bool
    covar_names::Vector{Symbol}
    degree::Int64
    knots::Vector{Float64}
    natural_spline::Bool
    monotone::Int64
    extrapolation::String
    metadata::HazardMetadata
    shared_baseline_key::Union{Nothing,SharedBaselineKey}
end

"""
    PhaseTypeCoxianHazard <: _MarkovHazard

Runtime hazard type for phase-type (Coxian) transitions on the expanded state space.

This type represents the internal structure of a phase-type hazard after model
construction. It inherits from `_MarkovHazard` because the expanded state space
is Markovian (each phase transition is exponential).

# Coxian Structure

For a transition s → d with n phases, this hazard manages:
- Progression rates λ₁...λₙ₋₁ (between phases within origin state)
- Exit rates μ₁...μₙ (from each phase to destination state)

The expanded hazard from phase i has rate:
- λᵢ + μᵢ (for i < n)
- μₙ (for i = n, final phase)

# Fields

**Standard hazard fields:**
- `hazname::Symbol`: Name identifier (e.g., :h12)
- `statefrom::Int64`: Observed origin state
- `stateto::Int64`: Observed destination state
- `family::Symbol`: Always `:pt`
- `parnames::Vector{Symbol}`: Parameter names [λ₁...λₙ₋₁, μ₁...μₙ, covariates...]
- `npar_baseline::Int`: Baseline parameters (2n - 1)
- `npar_total::Int`: Total parameters (baseline + covariates)
- `hazard_fn::Function`: Total hazard out of current phase
- `cumhaz_fn::Function`: Cumulative hazard
- `has_covariates::Bool`: Whether covariates are present
- `covar_names::Vector{Symbol}`: Pre-extracted covariate names
- `metadata::HazardMetadata`: Tang/linpred metadata
- `shared_baseline_key`: Tang baseline sharing key

**Phase-type specific fields:**
- `n_phases::Int`: Number of Coxian phases
- `phase_index::Int`: Which phase this hazard represents (1 to n_phases)
- `is_progression::Bool`: True if this is a progression hazard (λ), false if exit (μ)
- `progression_param_indices::UnitRange{Int}`: Indices of λ parameters in parnames
- `exit_param_indices::UnitRange{Int}`: Indices of μ parameters in parnames

See also: [`PhaseTypeHazardSpec`](@ref), [`PhaseTypeModel`](@ref)
"""
struct PhaseTypeCoxianHazard <: _MarkovHazard
    hazname::Symbol
    statefrom::Int64                 # observed state from
    stateto::Int64                   # observed state to
    family::Symbol                   # :pt
    parnames::Vector{Symbol}         # [λ₁, ..., λₙ₋₁, μ₁, ..., μₙ, covariates...]
    npar_baseline::Int64             # 2n - 1
    npar_total::Int64                # baseline + covariates
    hazard_fn::Function              # hazard function (t, pars, covars) -> rate
    cumhaz_fn::Function              # cumulative hazard function
    has_covariates::Bool
    covar_names::Vector{Symbol}
    metadata::HazardMetadata
    shared_baseline_key::Union{Nothing, SharedBaselineKey}
    
    # Phase-type specific fields
    n_phases::Int                    # number of Coxian phases
    phase_index::Int                 # which phase this hazard represents (1..n_phases)
    is_progression::Bool             # true = progression (λ), false = exit (μ)
    progression_param_indices::UnitRange{Int}  # indices of λ params (1:n-1)
    exit_param_indices::UnitRange{Int}         # indices of μ params (n:2n-1)
end

# =============================================================================
# User-Facing Hazard Specification Types
# =============================================================================
#
# These types are used by users to specify hazards via the Hazard() and 
# @hazard macros. They are then processed by build_hazards() to create
# the internal _Hazard types used for computation.
#
# =============================================================================

"""
Abstract type for multistate process.
"""
abstract type MultistateProcess end

"""
    Abstract type for multistate Markov process.
"""
abstract type MultistateMarkovProcess <: MultistateProcess end

"""
    Abstract type for multistate semi-Markov process.
"""
abstract type MultistateSemiMarkovProcess <: MultistateProcess end

"""
    ParametricHazard(haz::StatsModels.FormulaTerm, family::Symbol, statefrom::Int64, stateto::Int64)

Specify a cause-specific baseline hazard. 

# Arguments
- `hazard`: regression formula for the (log) hazard, parsed using StatsModels.jl.
- `family`: parameterization for the baseline hazard, one of `:exp` for exponential, `:wei` for Weibull, `:gom` for Gompertz. 
- `statefrom`: state number for the origin state.
- `stateto`: state number for the destination state.
"""
struct ParametricHazard <: HazardFunction
    hazard::StatsModels.FormulaTerm   # StatsModels.jl formula
    family::Symbol     # one of :exp, :wei, :gom
    statefrom::Int64   # starting state number
    stateto::Int64     # destination state number
    metadata::HazardMetadata
end

"""
    SplineHazard(haz::StatsModels.FormulaTerm, family::Symbol, statefrom::Int64, stateto::Int64; df::Union{Int64,Nothing}, degree::Int64, knots::Union{Vector{Float64},Float64,Nothing}, boundaryknots::Union{Vector{Float64},Nothing}, extrapolation::String, natural_spline::Bool)

Specify a cause-specific baseline hazard. 

# Arguments
- `hazard`: regression formula for the (log) hazard, parsed using StatsModels.jl.
- `family`: `:sp` for splines for the baseline hazard.
- `statefrom`: state number for the origin state.
- `stateto`: state number for the destination state.
- `df`: Degrees of freedom.
- `degree`: Degree of the spline polynomial basis.
- `knots`: Vector of knot locations.
- `boundaryknots`: Vector of boundary knot locations
- `extrapolation`: Extrapolation method beyond boundary knots:
    - "constant" (default): Constant hazard beyond boundaries with C¹ continuity (smooth transition).
      Uses basis recombination to enforce h'=0 at boundaries (Neumann BC). Requires degree >= 2.
    - "linear": Linear extrapolation using derivative at boundary.
- `natural_spline`: Restrict the second derivative to zero at the boundaries (natural spline).
- `monotone`: 
"""
struct SplineHazard <: HazardFunction
    hazard::StatsModels.FormulaTerm   # StatsModels.jl formula
    family::Symbol     # :sp for splines
    statefrom::Int64   # starting state number
    stateto::Int64     # destination state number
    degree::Int64
    knots::Union{Nothing,Float64,Vector{Float64}}
    boundaryknots::Union{Nothing,Vector{Float64}}
    extrapolation::String
    natural_spline::Bool
    monotone::Int64
    metadata::HazardMetadata
end

"""
    PhaseTypeHazardSpec(hazard, family, statefrom, stateto, n_phases, metadata)

User-facing specification for a phase-type (Coxian) hazard.
Created by `Hazard(:pt, ...)` and converted to internal types during model construction.

A phase-type hazard models the sojourn time as absorption in a Coxian Markov chain
with `n_phases` latent phases. This provides a flexible family of distributions
(including exponential as n_phases=1) while maintaining Markovian structure.

# Parameterization (Coxian)

For a transition s → d with n phases, the parameter vector contains:
- λ₁, ..., λₙ₋₁: progression rates between phases (n-1 parameters)
- μ₁, ..., μₙ: exit rates to destination state (n parameters)

Total baseline parameters: 2n - 1

# Fields
- `hazard`: StatsModels.jl formula for covariates
- `family`: Always `:pt`
- `statefrom`: Origin state number
- `stateto`: Destination state number  
- `n_phases`: Number of Coxian phases (≥ 1)
- `structure`: Coxian structure constraint (`:unstructured`, `:allequal`, or `:prop_to_prog`)
- `metadata`: HazardMetadata for time_transform and linpred_effect

# Structures
- `:unstructured` (default): All λᵢ and μᵢ are free parameters
- `:allequal`: All λᵢ equal, all μᵢ equal (2 free parameters + covariates)
- `:prop_to_prog`: μᵢ = c × λᵢ for i < n (Titman-Sharples constraint)

# Example
```julia
# 3-phase Coxian hazard for transition 1 → 2
h = Hazard(@formula(0 ~ 1 + age), :pt, 1, 2; n_phases=3)

# With all-equal constraint (Erlang-like)
h = Hazard(:pt, 1, 2; n_phases=3, coxian_structure=:allequal)
```

See also: [`PhaseTypeCoxianHazard`](@ref), [`PhaseTypeModel`](@ref)
"""
struct PhaseTypeHazardSpec <: HazardFunction
    hazard::StatsModels.FormulaTerm   # StatsModels.jl formula
    family::Symbol                     # :pt
    statefrom::Int64
    stateto::Int64
    n_phases::Int                      # number of Coxian phases (≥1)
    structure::Symbol                  # :unstructured, :allequal, or :prop_to_prog
    metadata::HazardMetadata
    
    function PhaseTypeHazardSpec(hazard::StatsModels.FormulaTerm, family::Symbol,
                                  statefrom::Int64, stateto::Int64, n_phases::Int,
                                  structure::Symbol, metadata::HazardMetadata)
        family == :pt || throw(ArgumentError("PhaseTypeHazardSpec family must be :pt"))
        n_phases >= 1 || throw(ArgumentError("n_phases must be ≥ 1, got $n_phases"))
        structure in (:unstructured, :allequal, :prop_to_prog) ||
            throw(ArgumentError("structure must be :unstructured, :allequal, or :prop_to_prog, got :$structure"))
        new(hazard, family, statefrom, stateto, n_phases, structure, metadata)
    end
end

# -----------------------------------------------------------------------------
# Baseline signature helpers (Tang shared trajectories)
# -----------------------------------------------------------------------------

@inline _hashable_tuple(x::Nothing) = nothing
@inline _hashable_tuple(x::AbstractVector) = Tuple(x)
@inline _hashable_tuple(x) = x

baseline_signature(::HazardFunction, ::Symbol) = nothing

function baseline_signature(h::ParametricHazard, runtime_family::Symbol)
    parts = (:parametric, runtime_family)
    return UInt64(hash(parts))
end

function baseline_signature(h::SplineHazard, runtime_family::Symbol)
    if runtime_family != :sp
        return UInt64(hash((:parametric, runtime_family)))
    end
    knots_repr = _hashable_tuple(h.knots)
    boundary_repr = _hashable_tuple(h.boundaryknots)
    parts = (
        :spline,
        runtime_family,
        h.degree,
        knots_repr,
        boundary_repr,
        Symbol(h.extrapolation),
        h.natural_spline,
        h.monotone,
    )
    return UInt64(hash(parts))
end

shared_baseline_key(::HazardFunction, ::Symbol) = nothing

function shared_baseline_key(h::ParametricHazard, runtime_family::Symbol)
    h.metadata.time_transform || return nothing
    sig = baseline_signature(h, runtime_family)
    sig === nothing && return nothing
    return SharedBaselineKey(h.statefrom, sig)
end

function shared_baseline_key(h::SplineHazard, runtime_family::Symbol)
    h.metadata.time_transform || return nothing
    sig = baseline_signature(h, runtime_family)
    sig === nothing && return nothing
    return SharedBaselineKey(h.statefrom, sig)
end

function baseline_signature(h::PhaseTypeHazardSpec, runtime_family::Symbol)
    parts = (:phasetype, runtime_family, h.n_phases)
    return UInt64(hash(parts))
end

function shared_baseline_key(h::PhaseTypeHazardSpec, runtime_family::Symbol)
    h.metadata.time_transform || return nothing
    sig = baseline_signature(h, runtime_family)
    sig === nothing && return nothing
    return SharedBaselineKey(h.statefrom, sig)
end

# =============================================================================
# Hazard Classification Helpers
# =============================================================================

"""
    _is_markov_hazard(hazard::_Hazard) -> Bool

Check if a hazard is Markovian (time-homogeneous).
Returns true for `_MarkovHazard` subtypes and degree-0 splines.

This is the authoritative check for model classification - models with all
Markov hazards are classified as `:markov`, otherwise `:semi_markov`.

Note: `PhaseTypeCoxianHazard <: _MarkovHazard`, so phase-type hazards are
considered Markovian (the expanded state space is Markovian).
"""
@inline function _is_markov_hazard(hazard::_Hazard)
    # MarkovHazard and PhaseTypeCoxianHazard are Markovian
    hazard isa _MarkovHazard && return true
    
    # Degree-0 splines are piecewise constant (step hazards) - Markovian
    if hazard isa RuntimeSplineHazard
        return hazard.degree == 0
    end
    
    return false
end

"""
Total hazard for absorbing states, contains nothing as the total hazard is always zero.
"""
struct _TotalHazardAbsorbing <: _TotalHazard 
end

"""
Total hazard struct for transient states, contains the indices of cause-specific hazards that contribute to the total hazard. The components::Vector{Int64} are indices of Vector{_Hazard} when call_tothaz needs to extract the correct cause-specific hazards.
"""
struct _TotalHazardTransient <: _TotalHazard
    components::Vector{Int64}
end

"""
    MarkovSurrogate(hazards::Vector{_MarkovHazard}, parameters::NamedTuple; fitted::Bool=false)

Markov surrogate for importance sampling proposals in MCEM.
Uses ParameterHandling.jl for parameter management.

# Fields
- `hazards::Vector{_MarkovHazard}`: Exponential hazard functions for each transition
- `parameters::NamedTuple`: Parameter structure (flat, nested, natural, unflatten)
- `fitted::Bool`: Whether the surrogate parameters have been fitted via MLE.
  If `false`, parameters are default/placeholder values and the surrogate should be
  fitted before use in MCEM or importance sampling.

# Construction
```julia
# Unfitted surrogate (needs fitting before use)
surrogate = MarkovSurrogate(hazards, params)  # fitted=false by default

# Fitted surrogate
surrogate = MarkovSurrogate(hazards, params; fitted=true)
```

See also: [`set_surrogate!`](@ref)
"""
struct MarkovSurrogate
    hazards::Vector{_MarkovHazard}
    parameters::NamedTuple
    fitted::Bool
    
    # Inner constructor that accepts any vector of hazards (converts to _MarkovHazard)
    function MarkovSurrogate(hazards::Vector{<:_Hazard}, parameters::NamedTuple; fitted::Bool=false)
        # Verify all hazards are Markov-compatible
        for h in hazards
            h isa _MarkovHazard || throw(ArgumentError(
                "MarkovSurrogate requires all hazards to be Markov (exponential). Got $(typeof(h))."))
        end
        new(convert(Vector{_MarkovHazard}, hazards), parameters, fitted)
    end
end

"""
    SurrogateControl(model::MultistateProcess, statefrom, targets, uinds, ginds)

Struct containing objects for computing the discrepancy of a Markov surrogate.
"""
struct SurrogateControl
    model::MultistateProcess
    statefrom::Int64
    targets::Matrix{Float64}
    uinds::Vector{Union{Nothing, Int64}}
    ginds::Vector{Union{Nothing, Int64}}
end

"""
    MultistateModel

Struct that fully specifies a multistate process for simulation or inference, used in the case when sample paths are fully observed. 

# Fields
- `data::DataFrame`: Long-format dataset with observations
- `parameters::NamedTuple`: Parameter structure containing:
  - `flat::Vector{Float64}` - flat parameter vector for optimizer (log scale for baseline)
  - `nested::NamedTuple` - nested parameters by hazard name with baseline/covariates fields
  - `natural::NamedTuple` - natural scale parameters by hazard name
  - `unflatten::Function` - function to unflatten flat vector to nested structure
- `hazards::Vector{_Hazard}`: Cause-specific hazard functions
- Plus other model specification fields...
"""
mutable struct MultistateModel <: MultistateProcess
    data::DataFrame
    parameters::NamedTuple  # Sole parameter storage: (flat, nested, natural, unflatten)
    hazards::Vector{_Hazard}
    totalhazards::Vector{_TotalHazard}
    tmat::Matrix{Int64}
    emat::Matrix{Float64}
    hazkeys::Dict{Symbol, Int64}
    subjectindices::Vector{Vector{Int64}}
    SubjectWeights::Vector{Float64}
    ObservationWeights::Union{Nothing, Vector{Float64}}
    CensoringPatterns::Matrix{Float64}
    markovsurrogate::Union{Nothing, MarkovSurrogate}
    modelcall::NamedTuple
end

"""
    MultistateMarkovModel

Struct that fully specifies a multistate Markov process with no censored state, used with panel data.
Parameters are stored in `parameters` as (flat, nested, natural, unflatten).
"""
mutable struct MultistateMarkovModel <: MultistateMarkovProcess
    data::DataFrame
    parameters::NamedTuple  # Sole parameter storage: (flat, nested, natural, unflatten)
    hazards::Vector{_MarkovHazard}
    totalhazards::Vector{_TotalHazard}
    tmat::Matrix{Int64}
    emat::Matrix{Float64}
    hazkeys::Dict{Symbol, Int64}
    subjectindices::Vector{Vector{Int64}}
    SubjectWeights::Vector{Float64}
    ObservationWeights::Union{Nothing, Vector{Float64}}
    CensoringPatterns::Matrix{Float64}
    markovsurrogate::Union{Nothing, MarkovSurrogate}
    modelcall::NamedTuple
end

"""
    MultistateMarkovModelCensored

Struct that fully specifies a multistate Markov process with some censored states, used with panel data.
Parameters are stored in `parameters` as (flat, nested, natural, unflatten).
"""
mutable struct MultistateMarkovModelCensored <: MultistateMarkovProcess
    data::DataFrame
    parameters::NamedTuple  # Sole parameter storage: (flat, nested, natural, unflatten)
    hazards::Vector{_MarkovHazard}
    totalhazards::Vector{_TotalHazard}
    tmat::Matrix{Int64}
    emat::Matrix{Float64}
    hazkeys::Dict{Symbol, Int64}
    subjectindices::Vector{Vector{Int64}}
    SubjectWeights::Vector{Float64}
    ObservationWeights::Union{Nothing, Vector{Float64}}
    CensoringPatterns::Matrix{Float64}
    markovsurrogate::Union{Nothing, MarkovSurrogate}
    modelcall::NamedTuple
end

"""
    MultistateSemiMarkovModel

Struct that fully specifies a multistate semi-Markov process, used with exact death times.
Parameters are stored in `parameters` as (flat, nested, natural, unflatten).
"""
mutable struct MultistateSemiMarkovModel <: MultistateSemiMarkovProcess
    data::DataFrame
    parameters::NamedTuple  # Sole parameter storage: (flat, nested, natural, unflatten)
    hazards::Vector{_Hazard}  # Can contain both MarkovHazard and SemiMarkovHazard
    totalhazards::Vector{_TotalHazard}
    tmat::Matrix{Int64}
    emat::Matrix{Float64}
    hazkeys::Dict{Symbol, Int64}
    subjectindices::Vector{Vector{Int64}}
    SubjectWeights::Vector{Float64}
    ObservationWeights::Union{Nothing, Vector{Float64}}
    CensoringPatterns::Matrix{Float64}
    markovsurrogate::Union{Nothing, MarkovSurrogate}
    modelcall::NamedTuple
end

"""
    MultistateSemiMarkovModelCensored

Struct that fully specifies a multistate semi-Markov process with some censored states, used with panel data.
Parameters are stored in `parameters` as (flat, nested, natural, unflatten).
"""
mutable struct MultistateSemiMarkovModelCensored <: MultistateSemiMarkovProcess
    data::DataFrame
    parameters::NamedTuple  # Sole parameter storage: (flat, nested, natural, unflatten)
    hazards::Vector{_Hazard}  # Can contain both MarkovHazard and SemiMarkovHazard
    totalhazards::Vector{_TotalHazard}
    tmat::Matrix{Int64}
    emat::Matrix{Float64}
    hazkeys::Dict{Symbol, Int64}
    subjectindices::Vector{Vector{Int64}}
    SubjectWeights::Vector{Float64}
    ObservationWeights::Union{Nothing, Vector{Float64}}
    CensoringPatterns::Matrix{Float64}
    markovsurrogate::Union{Nothing, MarkovSurrogate}
    modelcall::NamedTuple
end

# =============================================================================
# Phase-Type Hazard Model (for :pt family hazards)
# =============================================================================

# Forward declaration for PhaseTypeMappings (defined in phasetype.jl)
# The actual struct is defined in phasetype.jl and included after common.jl

"""
    PhaseTypeModel <: MultistateMarkovProcess

A multistate model with phase-type (Coxian) hazards.

This is a wrapper around an expanded Markov model where states with `:pt` hazards
are split into multiple latent phases. The user interacts with the model in terms
of the original observed states and phase-type parameters (λ progression rates, 
μ exit rates), while internally the model operates on the expanded state space.

Phase-type models are classified as Markov processes (`<: MultistateMarkovProcess`)
because the expanded state space is Markovian - each phase transition follows an
exponential distribution.

# Fields

**Original (observed) state space:**
- `data::DataFrame`: Original data on observed states
- `parameters::NamedTuple`: Parameters in phase-type parameterization (λ, μ)
- `tmat::Matrix{Int64}`: Original transition matrix
- `hazards_spec::Vector{<:HazardFunction}`: Original user hazard specifications

**Expanded (internal) state space:**
- `expanded_data::DataFrame`: Data expanded to phase-level observations
- `expanded_parameters::NamedTuple`: Parameters for expanded Markov model
- `expanded_model::MultistateMarkovProcess`: The internal expanded Markov model
- `mappings::PhaseTypeMappings`: Bidirectional state space mappings

**Model metadata:**
- `totalhazards::Vector{_TotalHazard}`: Total hazards (on expanded space)
- `emat::Matrix{Float64}`: Emission matrix (for panel data)
- `hazkeys::Dict{Symbol, Int64}`: Hazard name → index mapping
- `subjectindices::Vector{Vector{Int64}}`: Subject data indices
- `SubjectWeights::Vector{Float64}`: Subject-level weights
- `ObservationWeights::Union{Nothing, Vector{Float64}}`: Observation weights
- `CensoringPatterns::Matrix{Float64}`: Censoring pattern matrix
- `markovsurrogate::Union{Nothing, MarkovSurrogate}`: Markov surrogate (if any)
- `modelcall::NamedTuple`: Model specification call

# Behavior

- **Fitting**: Uses the expanded Markov model internally; results are translated back
  to phase-type parameters
- **Simulation**: Can output paths on expanded or collapsed state space via `expanded` kwarg
- **Likelihood**: Computed on expanded state space using standard Markov likelihood

# Example

```julia
# Specify model with phase-type hazard
h12 = Hazard(@formula(0 ~ 1), :pt, 1, 2; n_phases=3)
h23 = Hazard(@formula(0 ~ 1), :exp, 2, 3)
model = multistatemodel(data, (h12, h23))

# Fit (internally uses expanded Markov model)
fitted = fit(model)

# Simulate (returns collapsed paths by default)
sim = simulate(model; paths=true)
sim_expanded = simulate(model; paths=true, expanded=true)
```

See also: [`PhaseTypeHazardSpec`](@ref), [`PhaseTypeMappings`](@ref), [`PhaseTypeFittedModel`](@ref)
"""
mutable struct PhaseTypeModel <: MultistateMarkovProcess
    # Expanded (internal) state space - these are standard fields for loglik compatibility
    # Note: `data`, `tmat`, `parameters` MUST contain expanded versions for loglik_markov
    data::DataFrame                  # Expanded data (phase-type internal states)
    tmat::Matrix{Int64}              # Expanded transition matrix
    parameters::NamedTuple           # Expanded parameters (for loglik_markov)
    expanded_model::Any              # MultistateMarkovProcess (expanded), may be nothing
    mappings::Any                    # PhaseTypeMappings (defined in phasetype.jl)
    
    # Original (observed) state space - user-facing
    original_data::DataFrame         # Original user data (observed states)
    original_tmat::Matrix{Int64}     # Original transition matrix
    original_parameters::NamedTuple  # Phase-type parameterization (λ, μ) for users
    hazards_spec::Vector{<:HazardFunction}  # Original user hazard specs
    
    # Standard model fields (on expanded space for internal operations)
    hazards::Vector{_MarkovHazard}   # Expanded hazards (all Markov)
    totalhazards::Vector{_TotalHazard}
    emat::Matrix{Float64}
    hazkeys::Dict{Symbol, Int64}
    subjectindices::Vector{Vector{Int64}}
    SubjectWeights::Vector{Float64}
    ObservationWeights::Union{Nothing, Vector{Float64}}
    CensoringPatterns::Matrix{Float64}
    markovsurrogate::Union{Nothing, MarkovSurrogate}
    modelcall::NamedTuple
end

"""
    MultistateModelFitted

Struct that fully specifies a fitted multistate model.
Parameters are stored in `parameters` as (flat, nested, natural, unflatten).
"""
mutable struct MultistateModelFitted <: MultistateProcess
    data::DataFrame
    parameters::NamedTuple  # Sole parameter storage: (flat, nested, natural, unflatten)
    loglik::NamedTuple
    vcov::Union{Nothing,Matrix{Float64}}
    ij_vcov::Union{Nothing,Matrix{Float64}}  # Infinitesimal jackknife variance-covariance
    jk_vcov::Union{Nothing,Matrix{Float64}}  # Jackknife variance-covariance
    subject_gradients::Union{Nothing,Matrix{Float64}}  # Subject-level score vectors (p × n)
    hazards::Vector{_Hazard}
    totalhazards::Vector{_TotalHazard}
    tmat::Matrix{Int64}
    emat::Matrix{Float64}
    hazkeys::Dict{Symbol, Int64}
    subjectindices::Vector{Vector{Int64}}
    SubjectWeights::Vector{Float64}
    ObservationWeights::Union{Nothing, Vector{Float64}}
    CensoringPatterns::Matrix{Float64}
    markovsurrogate::Union{Nothing, MarkovSurrogate}
    ConvergenceRecords::Union{Nothing, NamedTuple, Optim.OptimizationResults, Optim.MultivariateOptimizationResults}
    ProposedPaths::Union{Nothing, NamedTuple}
    modelcall::NamedTuple
end

"""
    SamplePath(subjID::Int64, times::Vector{Float64}, states::Vector{Int64})

Struct for storing a sample path, consists of subject identifier, jump times, state sequence.
"""
struct SamplePath
    subj::Int64
    times::Vector{Float64}
    states::Vector{Int64}
    SamplePath(subj, times, states) = length(times) != length(states) ? error("Number of times in a jump chain must equal the number of states.") : new(subj, times, states)
end

@inline function Base.:(==)(lhs::SamplePath, rhs::SamplePath)
    lhs.subj == rhs.subj || return false
    lhs.times == rhs.times || return false
    return lhs.states == rhs.states
end

@inline function Base.isequal(lhs::SamplePath, rhs::SamplePath)
    lhs.subj == rhs.subj || return false
    _vector_isequal(lhs.times, rhs.times) || return false
    return _vector_isequal(lhs.states, rhs.states)
end

@inline function Base.hash(path::SamplePath, h::UInt)
    h = hash(path.subj, h)
    h = hash(path.times, h)
    return hash(path.states, h)
end

@inline function _vector_isequal(x::Vector, y::Vector)
    length(x) == length(y) || return false
    for (xi, yi) in zip(x, y)
        isequal(xi, yi) || return false
    end
    return true
end

"""
    ExactData(model::MultistateProcess, samplepaths::Array{SamplePath})

Struct containing exactly observed sample paths and a model object. Used in fitting a multistate model to completely observed data.
"""
struct ExactData
    model::MultistateProcess
    paths::Array{SamplePath}
end

"""
    ExactDataAD(model::MultistateProcess, samplepaths::Array{SamplePath})

Struct containing exactly observed sample paths and a model object. Used in fitting a multistate model to completely observed data. Used for computing the variance-covariance matrix via autodiff.
"""
struct ExactDataAD
    path::Vector{SamplePath}
    samplingweight::Vector{Float64}
    hazards::Vector{<:_Hazard}
    model::MultistateProcess
end

"""
    MPanelDataColumnAccessor

Pre-extracted DataFrame columns for allocation-free access in hot paths.
Avoids DataFrame dispatch overhead by storing direct references to column vectors.

# Fields
- `tstart`, `tstop`: Time interval bounds
- `statefrom`, `stateto`: State transition indicators
- `obstype`: Observation type (1=exact, 2=panel, 3+=censored)
- `tpm_map_col1`: Covariate pattern index (from tpm_map matrix column 1)
- `tpm_map_col2`: TPM time index (from tpm_map matrix column 2)
"""
struct MPanelDataColumnAccessor
    tstart::Vector{Float64}
    tstop::Vector{Float64}
    statefrom::Vector{Int}
    stateto::Vector{Int}
    obstype::Vector{Int}
    tpm_map_col1::Vector{Int}  # Covariate pattern index
    tpm_map_col2::Vector{Int}  # TPM time interval index
end

function MPanelDataColumnAccessor(data::DataFrame, tpm_map::Matrix{Int64})
    MPanelDataColumnAccessor(
        data.tstart,
        data.tstop,
        data.statefrom,
        data.stateto,
        data.obstype,
        tpm_map[:, 1],  # Extract column 1 as vector
        tpm_map[:, 2]   # Extract column 2 as vector
    )
end

"""
    TPMCache

Mutable cache for pre-allocated arrays used in Markov likelihood computation.
Stores hazard matrix book, TPM book, matrix exponential cache, and work arrays.

This cache is used when parameters are Float64 to avoid repeated allocations.
When parameters are Dual (ForwardDiff), fresh arrays are allocated since the
element type must match the parameter type for AD compatibility.

# Fields
- `hazmat_book`: Pre-allocated hazard intensity matrices (one per covariate pattern)
- `tpm_book`: Pre-allocated transition probability matrices (nested: pattern × time interval)
- `exp_cache`: Pre-allocated workspace for matrix exponential computation
- `q_work`: Work matrix for forward algorithm (S × S)
- `lmat_work`: Work matrix for forward algorithm likelihood (S × max_obs+1)
- `pars_cache`: Mutable parameter vectors for in-place updates (avoids NamedTuple allocation)
- `covars_cache`: Pre-extracted covariates per unique covariate pattern
- `hazard_rates_cache`: Pre-computed hazard rates per (pattern, hazard) for Markov models
- `eigen_cache`: Cached eigendecompositions for fast matrix exponentials with multiple Δt
- `dt_values`: Unique Δt values per covariate pattern for batched matrix exp
"""
mutable struct TPMCache
    hazmat_book::Vector{Matrix{Float64}}
    tpm_book::Vector{Vector{Matrix{Float64}}}
    exp_cache::Any  # ExponentialUtilities cache (type varies)
    q_work::Matrix{Float64}
    lmat_work::Matrix{Float64}
    # New caches for additional optimizations
    pars_cache::Vector{Vector{Float64}}  # Mutable parameter vectors per hazard
    covars_cache::Vector{Vector{NamedTuple}}  # Pre-extracted covariates per pattern per hazard
    hazard_rates_cache::Vector{Vector{Float64}}  # Pre-computed rates per pattern per hazard
    # Eigendecomposition cache for batched matrix exponentials
    eigen_cache::Vector{Union{Nothing, Tuple{Matrix{Float64}, Vector{ComplexF64}, Matrix{ComplexF64}}}}
    dt_values::Vector{Vector{Float64}}  # Unique Δt values per pattern
end

function TPMCache(tmat::Matrix{Int64}, tpm_index::Vector{DataFrame}, 
                  model_data::DataFrame, hazards::Vector{<:_Hazard})
    nstates = size(tmat, 1)
    nmats = map(x -> nrow(x), tpm_index)
    nhazards = length(hazards)
    npatterns = length(tpm_index)
    
    # Pre-allocate hazmat_book (one matrix per covariate pattern)
    hazmat_book = [zeros(Float64, nstates, nstates) for _ in eachindex(tpm_index)]
    
    # Pre-allocate tpm_book (nested: one vector of matrices per covariate pattern)
    tpm_book = [[zeros(Float64, nstates, nstates) for _ in 1:nmats[i]] for i in eachindex(tpm_index)]
    
    # Pre-allocate matrix exponential cache
    exp_cache = ExponentialUtilities.alloc_mem(hazmat_book[1], ExpMethodGeneric())
    
    # Work arrays for forward algorithm
    q_work = zeros(Float64, nstates, nstates)
    
    # lmat_work sized for largest subject (estimate conservatively)
    max_obs_estimate = 100
    lmat_work = zeros(Float64, nstates, max_obs_estimate + 1)
    
    # Pre-allocate mutable parameter vectors (one per hazard)
    pars_cache = [zeros(Float64, h.npar_total) for h in hazards]
    
    # Pre-extract covariates for each (pattern, hazard) combination
    # This avoids repeated DataFrame lookups in compute_hazmat!
    covars_cache = Vector{Vector{NamedTuple}}(undef, npatterns)
    for p in 1:npatterns
        datind = tpm_index[p].datind[1]
        row = model_data[datind, :]
        covars_cache[p] = [_extract_covars_or_empty(row, h.covar_names) for h in hazards]
    end
    
    # Pre-allocate hazard rates cache (one rate per pattern per hazard)
    # Values will be computed when parameters are updated
    hazard_rates_cache = [zeros(Float64, nhazards) for _ in 1:npatterns]
    
    # Eigendecomposition cache - initially empty, populated on first likelihood call
    # Use eigendecomposition when npatterns == 1 && nmats[1] >= 2 (benefit threshold)
    eigen_cache = Vector{Union{Nothing, Tuple{Matrix{Float64}, Vector{ComplexF64}, Matrix{ComplexF64}}}}(undef, npatterns)
    fill!(eigen_cache, nothing)
    
    # Extract Δt values for each pattern
    dt_values = [collect(Float64, idx.tstop) for idx in tpm_index]
    
    TPMCache(hazmat_book, tpm_book, exp_cache, q_work, lmat_work,
             pars_cache, covars_cache, hazard_rates_cache, eigen_cache, dt_values)
end

# Helper to extract covariates or return empty NamedTuple
# Handles interaction terms (e.g., "trt & age") by returning empty NamedTuple
@inline function _extract_covars_or_empty(row::DataFrameRow, covar_names::Vector{Symbol})
    isempty(covar_names) && return NamedTuple()
    
    # Check if all covar_names are valid column names (no interaction terms)
    for cname in covar_names
        cname_str = String(cname)
        # Interaction terms contain " & " and aren't valid column names
        if occursin(" & ", cname_str) || !hasproperty(row, cname)
            return NamedTuple()  # Return empty - can't cache interaction terms
        end
    end
    
    values = Tuple(getproperty(row, cname) for cname in covar_names)
    return NamedTuple{Tuple(covar_names)}(values)
end

"""
    _covars_cache_valid(covars_cache, hazards) -> Bool

Check if the covars_cache has valid covariates for all hazards that need them.
Returns false if any hazard with covariates has an empty NamedTuple in the cache.
This happens when hazards have interaction terms that can't be pre-cached.
"""
@inline function _covars_cache_valid(covars_cache::Vector{Vector{NamedTuple}}, 
                                     hazards::Vector{<:_Hazard})
    isempty(covars_cache) && return false
    
    # Check first pattern (all patterns should have same structure)
    pattern_covars = covars_cache[1]
    
    for (h, covars) in zip(hazards, pattern_covars)
        # If hazard has covariates but cache is empty, it's not valid
        if h.has_covariates && isempty(covars)
            return false
        end
    end
    
    return true
end

# Backward-compatible constructor (without hazards)
function TPMCache(tmat::Matrix{Int64}, tpm_index::Vector{DataFrame})
    nstates = size(tmat, 1)
    nmats = map(x -> nrow(x), tpm_index)
    npatterns = length(tpm_index)
    
    hazmat_book = [zeros(Float64, nstates, nstates) for _ in eachindex(tpm_index)]
    tpm_book = [[zeros(Float64, nstates, nstates) for _ in 1:nmats[i]] for i in eachindex(tpm_index)]
    exp_cache = ExponentialUtilities.alloc_mem(hazmat_book[1], ExpMethodGeneric())
    q_work = zeros(Float64, nstates, nstates)
    lmat_work = zeros(Float64, nstates, 100 + 1)
    
    # Empty caches when hazards not provided (backward compatibility)
    pars_cache = Vector{Vector{Float64}}()
    covars_cache = Vector{Vector{NamedTuple}}()
    hazard_rates_cache = Vector{Vector{Float64}}()
    
    # Initialize eigen cache and dt_values
    eigen_cache = Vector{Union{Nothing, Tuple{Matrix{Float64}, Vector{ComplexF64}, Matrix{ComplexF64}}}}(undef, npatterns)
    fill!(eigen_cache, nothing)
    dt_values = [collect(Float64, idx.tstop) for idx in tpm_index]
    
    TPMCache(hazmat_book, tpm_book, exp_cache, q_work, lmat_work,
             pars_cache, covars_cache, hazard_rates_cache, eigen_cache, dt_values)
end

"""
    MPanelData(model::MultistateProcess, books::Tuple)

Struct containing panel data, a model object, and bookkeeping objects. Used in fitting a multistate Markov model to panel data.

# Fields
- `model`: The multistate model
- `books`: Bookkeeping tuple (tpm_index, tpm_map) from `build_tpm_mapping`
- `columns`: Pre-extracted DataFrame column accessors for allocation-free access

The `columns` field provides direct access to data columns without DataFrame dispatch
overhead, significantly reducing allocations in the likelihood hot path.
"""
struct MPanelData
    model::MultistateProcess
    books::Tuple # tpm_index and tpm_map, from build_tpm_containers
    columns::MPanelDataColumnAccessor  # Pre-extracted columns for fast access
    cache::TPMCache  # Pre-allocated arrays for Float64 likelihood evaluations
end

# Constructor with automatic column extraction and cache allocation
function MPanelData(model::MultistateProcess, books::Tuple)
    tpm_map = books[2]  # Extract tpm_map matrix from books tuple
    columns = MPanelDataColumnAccessor(model.data, tpm_map)
    # Use enhanced cache constructor with hazards for covariate pre-extraction
    cache = TPMCache(model.tmat, books[1], model.data, model.hazards)
    MPanelData(model, books, columns, cache)
end

"""
    SMPanelData(model::MultistateProcess
    paths::Vector{Vector{SamplePath}}
    ImportanceWeights::Vector{Vector{Float64}}

Struct containing panel data, a model object, and bookkeeping objects. Used in fitting a multistate semi-Markov model to panel data via MCEM.
"""
struct SMPanelData
    model::MultistateProcess
    paths::Vector{Vector{SamplePath}}
    ImportanceWeights::Vector{Vector{Float64}}
end

# =============================================================================
# Fused Likelihood Data Structures
# =============================================================================

"""
    LightweightInterval

Minimal representation of a likelihood interval without DataFrame overhead.
Used in optimized batched likelihood computation to avoid allocation.
"""
struct LightweightInterval
    lb::Float64           # Lower bound (sojourn time)
    ub::Float64           # Upper bound (sojourn + increment)
    statefrom::Int        # Origin state
    stateto::Int          # Destination state
    covar_row_idx::Int    # Index into subject's covariate data
end

"""
    SubjectCovarCache

Cached covariate data for a single subject, keyed by time intervals.
Used in fused likelihood to avoid repeated DataFrame lookups.
"""
struct SubjectCovarCache
    tstart::Vector{Float64}   # Start times for covariate intervals
    covar_data::DataFrame     # Covariate columns only (no id, tstart, tstop, etc.)
end

"""
    TVCIntervalWorkspace

Pre-allocated workspace for computing TVC intervals to reduce allocations.
Used in semi-Markov likelihood when time-varying covariates are present.
"""
mutable struct TVCIntervalWorkspace
    change_times::Vector{Float64}
    utimes::Vector{Float64}
    intervals::Vector{LightweightInterval}
    sojourns::Vector{Float64}
    pathinds::Vector{Int}
    datinds::Vector{Int}
    
    function TVCIntervalWorkspace(max_times::Int=200)
        new(
            Vector{Float64}(undef, max_times),
            Vector{Float64}(undef, max_times),
            Vector{LightweightInterval}(undef, max_times),
            Vector{Float64}(undef, max_times),
            Vector{Int}(undef, max_times),
            Vector{Int}(undef, max_times)
        )
    end
end

# Thread-local TVC workspace storage
const TVC_INTERVAL_WORKSPACES = Dict{Int, TVCIntervalWorkspace}()
const TVC_WORKSPACE_LOCK = ReentrantLock()

"""Get or create thread-local TVCIntervalWorkspace"""
function get_tvc_workspace()::TVCIntervalWorkspace
    tid = Threads.threadid()
    ws = get(TVC_INTERVAL_WORKSPACES, tid, nothing)
    if isnothing(ws)
        lock(TVC_WORKSPACE_LOCK) do
            ws = get!(TVC_INTERVAL_WORKSPACES, tid) do
                TVCIntervalWorkspace(200)
            end
        end
    end
    return ws
end

# =============================================================================
# AD Backend Selection
# =============================================================================

"""
    ADBackend

Abstract type for automatic differentiation backend selection.
Enables switching between ForwardDiff (forward-mode, mutation-tolerant) and 
Enzyme (reverse-mode, mutation-free) based on problem characteristics.
"""
abstract type ADBackend end

"""
    ForwardDiffBackend <: ADBackend

Use ForwardDiff.jl for automatic differentiation.

**Characteristics:**
- Forward-mode AD: O(n) cost where n = number of parameters
- Efficient for small to medium parameter counts (< ~100 params)
- Tolerates in-place mutation in the objective function
- Default choice for most multistate models

**When to use:**
- Models with few parameters (exponential, Weibull hazards)
- When the mutating likelihood implementation is preferred for speed
"""
struct ForwardDiffBackend <: ADBackend end

"""
    EnzymeBackend <: ADBackend

Use Enzyme.jl for automatic differentiation.

**Characteristics:**
- Reverse-mode AD: O(1) cost in parameters (scales with output size)  
- Efficient for large parameter counts (> ~100 params)
- Requires mutation-free objective function
- Uses `loglik_*_functional` variants internally

**When to use:**
- Models with many parameters (complex spline hazards, many covariates)
- Neural ODE hazards (future extension)
- When Hessian computation is also needed efficiently

**Requirements:**
Enzyme requires the likelihood function to be mutation-free. The package
automatically selects functional (non-mutating) likelihood implementations
when this backend is specified.

**Note:** Enzyme.jl Julia 1.12 support is experimental (as of Dec 2024).
For Julia 1.12, use ForwardDiffBackend or MooncakeBackend.
"""
struct EnzymeBackend <: ADBackend end

"""
    MooncakeBackend <: ADBackend

Use Mooncake.jl for automatic differentiation (reverse-mode).

**Characteristics:**
- Reverse-mode AD: O(1) cost in number of parameters
- Efficient for large parameter counts (> ~100 params)  
- Pure Julia, good version compatibility
- Supports mutation (unlike Zygote)

**Works well for:**
- Semi-Markov models (no matrix exponential in likelihood)
- Models with many parameters where reverse-mode efficiency matters

**Known limitation (as of Dec 2024):**
Does NOT work for Markov panel models. The matrix exponential computation
uses LAPACK.gebal! internally, which Mooncake cannot differentiate through.
ChainRules.jl has an rrule for exp(::Matrix), but that rule itself calls
LAPACK, so Mooncake still fails. Use `ForwardDiffBackend()` for Markov models.
"""
struct MooncakeBackend <: ADBackend end

"""
    default_ad_backend(n_params::Int; is_markov::Bool=false) -> ADBackend

Select default AD backend based on parameter count and model type.

# Arguments
- `n_params::Int`: Number of parameters in the model
- `is_markov::Bool=false`: Whether the model uses Markov panel likelihoods

# Returns
- `ADBackend`: ForwardDiff for Markov models, Mooncake for large non-Markov models

# Notes
ForwardDiff is used for Markov models because matrix exponential differentiation
requires forward-mode AD (Mooncake/Enzyme cannot differentiate LAPACK calls).
For non-Markov models with many parameters, Mooncake's reverse-mode is more efficient.
"""
function default_ad_backend(n_params::Int; is_markov::Bool=false)
    if is_markov
        # Markov models require ForwardDiff due to matrix exponential
        return ForwardDiffBackend()
    else
        # For non-Markov, use reverse-mode for large parameter counts
        return n_params < 100 ? ForwardDiffBackend() : MooncakeBackend()
    end
end

"""
    get_optimization_ad(backend::ADBackend)

Convert ADBackend to Optimization.jl AD specification.
"""
get_optimization_ad(::ForwardDiffBackend) = Optimization.AutoForwardDiff()
get_optimization_ad(::EnzymeBackend) = Optimization.AutoEnzyme()
get_optimization_ad(::MooncakeBackend) = Optimization.AutoMooncake()

# =============================================================================
# Threading Configuration
# =============================================================================
#
# Parallelization support for likelihood evaluation. Uses Julia's built-in 
# Threads.@threads with physical core detection to avoid hyperthreading overhead.
#
# Thread safety considerations:
# - Each subject/path computes an independent likelihood contribution
# - Thread-local accumulators are used for the final sum
# - Shared read-only data (TPM books, hazards) is accessed without locks
# - No mutation of shared state during parallel execution
#
# Usage:
#   fit(model; parallel=true)  # Enable parallel likelihood evaluation
#   fit(model; parallel=false) # Sequential evaluation (default for AD)
#   fit(model; nthreads=4)     # Use exactly 4 threads
#
# =============================================================================

"""
    get_physical_cores() -> Int

Detect the number of physical CPU cores (excluding hyperthreads).

Uses Sys.CPU_THREADS as total threads, then estimates physical cores by
dividing by 2 on systems that typically have 2 threads per core (Intel/AMD x86).
On ARM (Apple Silicon), threads typically equal physical cores.

Returns at least 1 to ensure valid thread count.
"""
function get_physical_cores()
    total_threads = Sys.CPU_THREADS
    # Heuristic: ARM typically doesn't hyperthread, x86 does
    # Check architecture via pointer size and platform hints
    if Sys.ARCH == :aarch64 || Sys.ARCH == :arm64
        # Apple Silicon and ARM: threads ≈ physical cores
        return max(1, total_threads)
    else
        # x86: assume 2 threads per core (hyperthreading)
        return max(1, total_threads ÷ 2)
    end
end

"""
    recommended_nthreads(; task_count::Int=0) -> Int

Recommend number of threads for parallel likelihood evaluation.

# Arguments
- `task_count::Int=0`: Number of parallel tasks (subjects/paths). If 0, ignored.

# Returns
Number of threads to use, considering:
1. Available Julia threads (Threads.nthreads())
2. Physical cores (to avoid hyperthreading overhead)
3. Task count (no benefit from more threads than tasks)

# Notes
- Returns min(available_threads, physical_cores, task_count)
- Leaves 1 core free for main thread if > 4 physical cores available
- Returns 1 if threading provides no benefit
"""
function recommended_nthreads(; task_count::Int=0)
    available = Threads.nthreads()
    physical = get_physical_cores()
    
    # Don't use more threads than physical cores
    n = min(available, physical)
    
    # Leave 1 core for main thread on larger systems
    if n > 4
        n = n - 1
    end
    
    # Don't use more threads than tasks
    if task_count > 0
        n = min(n, task_count)
    end
    
    # At least 1 thread
    return max(1, n)
end

"""
    ThreadingConfig

Configuration for parallel likelihood evaluation.

# Fields
- `enabled::Bool`: Whether parallelization is active
- `nthreads::Int`: Number of threads to use
- `min_batch_size::Int`: Minimum tasks per thread to justify overhead
"""
struct ThreadingConfig
    enabled::Bool
    nthreads::Int
    min_batch_size::Int
end

"""
    ThreadingConfig(; parallel=false, nthreads=nothing, min_batch_size=10)

Create threading configuration for likelihood evaluation.

# Arguments
- `parallel::Bool=false`: Enable parallel execution
- `nthreads::Union{Nothing,Int}=nothing`: Number of threads. If nothing, auto-detect.
- `min_batch_size::Int=10`: Minimum tasks per thread to justify threading overhead

# Notes
When `parallel=true` and `nthreads=nothing`, uses `recommended_nthreads()` for auto-detection.
"""
function ThreadingConfig(; parallel::Bool=false, nthreads::Union{Nothing,Int}=nothing, 
                          min_batch_size::Int=10)
    if !parallel
        return ThreadingConfig(false, 1, min_batch_size)
    end
    
    n = isnothing(nthreads) ? recommended_nthreads() : nthreads
    
    # Disable if only 1 thread available
    if n <= 1
        return ThreadingConfig(false, 1, min_batch_size)
    end
    
    return ThreadingConfig(true, n, min_batch_size)
end

"""
    should_parallelize(config::ThreadingConfig, task_count::Int) -> Bool

Determine whether to use parallel execution for given task count.

Returns true if:
1. Threading is enabled in config
2. Task count exceeds min_batch_size × nthreads threshold
"""
function should_parallelize(config::ThreadingConfig, task_count::Int)
    config.enabled || return false
    return task_count >= config.nthreads * config.min_batch_size
end

# Global threading configuration (can be overridden per-call)
const _GLOBAL_THREADING_CONFIG = Ref(ThreadingConfig(parallel=false))

"""
    set_threading_config!(; parallel=false, nthreads=nothing, min_batch_size=10)

Set global threading configuration for likelihood evaluation.

# Example
```julia
# Enable parallel likelihood evaluation globally
set_threading_config!(parallel=true)

# Use exactly 4 threads
set_threading_config!(parallel=true, nthreads=4)

# Disable parallelization
set_threading_config!(parallel=false)
```
"""
function set_threading_config!(; parallel::Bool=false, nthreads::Union{Nothing,Int}=nothing,
                                 min_batch_size::Int=10)
    _GLOBAL_THREADING_CONFIG[] = ThreadingConfig(parallel=parallel, nthreads=nthreads,
                                                  min_batch_size=min_batch_size)
    return _GLOBAL_THREADING_CONFIG[]
end

"""
    get_threading_config() -> ThreadingConfig

Get current global threading configuration.
"""
get_threading_config() = _GLOBAL_THREADING_CONFIG[]


