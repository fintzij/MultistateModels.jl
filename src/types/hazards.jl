# =============================================================================
# Hazard Type Definitions
# =============================================================================
# All hazard-related types: abstract types, internal runtime types, and user-facing specs.
# Also includes abstract MultistateProcess types (needed early in dependency chain).
# Organized by inheritance hierarchy, then by family within each category.
# =============================================================================

using StatsModels

# =============================================================================
# Abstract Model Types (needed early for surrogates dependency)
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

# =============================================================================
# Abstract Hazard Types
# =============================================================================

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
# Hazard Metadata
# =============================================================================

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

# =============================================================================
# Time Transform Cache Types (for Tang shared trajectories)
# =============================================================================

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

# Global enable/disable for time transform caching
const _TIME_TRANSFORM_CACHE_ENABLED = Ref(true)

function enable_time_transform_cache!(flag::Bool)
    _TIME_TRANSFORM_CACHE_ENABLED[] = flag
    return _TIME_TRANSFORM_CACHE_ENABLED[]
end

_time_transform_cache_enabled() = _TIME_TRANSFORM_CACHE_ENABLED[]

# =============================================================================
# Runtime Hazard Types (Internal)
# =============================================================================
# These types are created during model construction and used for computation.
# Users specify hazards using the user-facing types below.
# =============================================================================

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
# Total Hazard Types
# =============================================================================

"""
Total hazard for absorbing states, contains nothing as the total hazard is always zero.
"""
struct _TotalHazardAbsorbing <: _TotalHazard 
end

"""
Total hazard struct for transient states, contains the indices of cause-specific hazards 
that contribute to the total hazard. The components::Vector{Int64} are indices of 
Vector{_Hazard} when call_tothaz needs to extract the correct cause-specific hazards.
"""
struct _TotalHazardTransient <: _TotalHazard
    components::Vector{Int64}
end

# =============================================================================
# User-Facing Hazard Specification Types
# =============================================================================
# These types are used by users to specify hazards via the Hazard() and 
# @hazard macros. They are then processed by build_hazards() to create
# the internal _Hazard types used for computation.
# =============================================================================

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

Specify a cause-specific baseline hazard with spline basis.

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
    - "constant" (default): Constant hazard beyond boundaries with C¹ continuity.
    - "linear": Linear extrapolation using derivative at boundary.
- `natural_spline`: Restrict the second derivative to zero at the boundaries.
- `monotone`: Monotonicity constraint (0, -1, 1).
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

# =============================================================================
# Baseline Signature Helpers (Tang shared trajectories)
# =============================================================================

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

# =============================================================================
# Time Transform Context Helper
# =============================================================================

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
hazard and caching remains active. Returns `nothing` when no hazards opt into 
time transforms (or caching is globally disabled).
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
