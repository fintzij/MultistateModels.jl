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

struct TangHazardKey{LinType,TimeType}
    linpred::LinType
    t::TimeType
end

struct TangCumulKey{LinType,TimeType}
    linpred::LinType
    lb::TimeType
    ub::TimeType
end

function Base.:(==)(a::TangHazardKey, b::TangHazardKey)
    return a.linpred == b.linpred && a.t == b.t
end

function Base.hash(a::TangHazardKey, h::UInt)
    return hash(a.t, hash(a.linpred, h))
end

function Base.:(==)(a::TangCumulKey, b::TangCumulKey)
    return a.linpred == b.linpred && a.lb == b.lb && a.ub == b.ub
end

function Base.hash(a::TangCumulKey, h::UInt)
    return hash(a.ub, hash(a.lb, hash(a.linpred, h)))
end

mutable struct TimeTransformCache{LinType,TimeType}
    hazard_values::Dict{TangHazardKey{LinType,TimeType}, LinType}
    cumulhaz_values::Dict{TangCumulKey{LinType,TimeType}, LinType}
end

function TimeTransformCache(::Type{LinType}, ::Type{TimeType}) where {LinType,TimeType}
    hazard_values = Dict{TangHazardKey{LinType,TimeType}, LinType}()
    cumulhaz_values = Dict{TangCumulKey{LinType,TimeType}, LinType}()
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

    lin_type = eltype(pars[idx])
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
- `family::String`: Distribution family ("exp")
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
    family::String
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
- `family::String`: Distribution family ("wei", "gom")
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
    family::String
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
    SplineHazard

Consolidated hazard type for spline-based hazards.
Uses B-spline basis functions for flexible baseline hazard.

# Fields
- `hazname::Symbol`: Name identifier (e.g., :h12)
- `statefrom::Int64`: Origin state
- `stateto::Int64`: Destination state
- `family::String`: Always "sp" for splines
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
- `metadata::HazardMetadata`: Tang/linpred metadata
- `shared_baseline_key::Union{Nothing,SharedBaselineKey}`
"""
struct SplineHazard <: _SplineHazard
    hazname::Symbol
    statefrom::Int64
    stateto::Int64
    family::String
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
    metadata::HazardMetadata
    shared_baseline_key::Union{Nothing,SharedBaselineKey}
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
    ParametricHazard(haz::StatsModels.FormulaTerm, family::string, statefrom::Int64, stateto::Int64)

Specify a cause-specific baseline hazard. 

# Arguments
- `hazard`: regression formula for the (log) hazard, parsed using StatsModels.jl.
- `family`: parameterization for the baseline hazard, one of "exp" for exponential, "wei" for Weibull, "gom" for Gompert. 
- `statefrom`: state number for the origin state.
- `stateto`: state number for the destination state.
"""
struct ParametricHazard <: HazardFunction
    hazard::StatsModels.FormulaTerm   # StatsModels.jl formula
    family::String     # one of "exp", "wei", "gom"
    statefrom::Int64   # starting state number
    stateto::Int64     # destination state number
    metadata::HazardMetadata
end

"""
    SplineHazard(haz::StatsModels.FormulaTerm, family::string, statefrom::Int64, stateto::Int64; df::Union{Int64,Nothing}, degree::Int64, knots::Union{Vector{Float64},Float64,Nothing}, boundaryknots::Union{Vector{Float64},Nothing}, extrapolation::String, natural_spline::Bool)

Specify a cause-specific baseline hazard. 

# Arguments
- `hazard`: regression formula for the (log) hazard, parsed using StatsModels.jl.
- `family`: "sp" for splines for the baseline hazard.
- `statefrom`: state number for the origin state.
- `stateto`: state number for the destination state.
- `df`: Degrees of freedom.
- `degree`: Degree of the spline polynomial basis.
- `knots`: Vector of knot locations.
- `boundaryknots`: Vector of boundary knot locations
- `extrapolation`: Either "linear" or "flat"
- `natural_spline`: Restrict the second derivative to zero at the boundaries (natural spline).
- `monotone`: 
"""
struct SplineHazard <: HazardFunction
    hazard::StatsModels.FormulaTerm   # StatsModels.jl formula
    family::String     # "sp" for splines
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

# -----------------------------------------------------------------------------
# Baseline signature helpers (Tang shared trajectories)
# -----------------------------------------------------------------------------

@inline _hashable_tuple(x::Nothing) = nothing
@inline _hashable_tuple(x::AbstractVector) = Tuple(x)
@inline _hashable_tuple(x) = x

baseline_signature(::HazardFunction, ::AbstractString) = nothing

function baseline_signature(h::ParametricHazard, runtime_family::AbstractString)
    parts = (:parametric, Symbol(runtime_family))
    return UInt64(hash(parts))
end

function baseline_signature(h::SplineHazard, runtime_family::AbstractString)
    if runtime_family != "sp"
        return UInt64(hash((:parametric, Symbol(runtime_family))))
    end
    knots_repr = _hashable_tuple(h.knots)
    boundary_repr = _hashable_tuple(h.boundaryknots)
    parts = (
        :spline,
        Symbol(runtime_family),
        h.degree,
        knots_repr,
        boundary_repr,
        Symbol(h.extrapolation),
        h.natural_spline,
        h.monotone,
    )
    return UInt64(hash(parts))
end

shared_baseline_key(::HazardFunction, ::AbstractString) = nothing

function shared_baseline_key(h::ParametricHazard, runtime_family::AbstractString)
    h.metadata.time_transform || return nothing
    sig = baseline_signature(h, runtime_family)
    sig === nothing && return nothing
    return SharedBaselineKey(h.statefrom, sig)
end

function shared_baseline_key(h::SplineHazard, runtime_family::AbstractString)
    h.metadata.time_transform || return nothing
    sig = baseline_signature(h, runtime_family)
    sig === nothing && return nothing
    return SharedBaselineKey(h.statefrom, sig)
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
    MarkovSurrogate(hazards::Vector{_MarkovHazard}, parameters::VectorOfVectors)
"""
struct MarkovSurrogate
    hazards::Vector{_MarkovHazard}
    parameters::VectorOfVectors
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
    MultistateModel(data::DataFrame, parameters::VectorOfVectors,hazards::Vector{Union{_Exponential, _ExponentialPH}}, totalhazards::Vector{_TotalHazard},tmat::Matrix{Int64}, hazkeys::Dict{Symbol, Int64}, subjectindices::Vector{Vector{Int64}})

Struct that fully specifies a multistate process for simulation or inference, used in the case when sample paths are fully observed. 

# New ParameterHandling fields (Phase 1):
- `parameters_ph`: NamedTuple containing:
  - `flat::Vector{Float64}` - flat parameter vector for optimizer
  - `transformed` - ParameterHandling transformed parameters (with positive() etc.)
  - `natural::NamedTuple` - natural scale parameters by hazard name
  - `unflatten::Function` - function to unflatten flat vector

# Phase 3 Change: Now mutable to allow `parameters_ph` updates
"""
mutable struct MultistateModel <: MultistateProcess
    data::DataFrame
    parameters::VectorOfVectors  # Legacy - keep for Phase 1 compatibility
    parameters_ph::NamedTuple  # NEW: Nested ParameterHandling structure
    hazards::Vector{_Hazard}
    totalhazards::Vector{_TotalHazard}
    tmat::Matrix{Int64}
    emat::Matrix{Float64}
    hazkeys::Dict{Symbol, Int64}
    subjectindices::Vector{Vector{Int64}}
    SubjectWeights::Vector{Float64}
    ObservationWeights::Union{Nothing, Vector{Float64}}
    CensoringPatterns::Matrix{Float64}
    markovsurrogate::MarkovSurrogate
    modelcall::NamedTuple
end

"""
    MultistateMarkovModel(data::DataFrame, parameters::VectorOfVectors,hazards::Vector{Union{_Exponential, _ExponentialPH}}, totalhazards::Vector{_TotalHazard},tmat::Matrix{Int64}, hazkeys::Dict{Symbol, Int64}, subjectindices::Vector{Vector{Int64}})

Struct that fully specifies a multistate Markov process with no censored state, used with panel data.

# Phase 3 Change: Now mutable to allow `parameters_ph` updates
"""
mutable struct MultistateMarkovModel <: MultistateMarkovProcess
    data::DataFrame
    parameters::VectorOfVectors  # Legacy - keep for Phase 1 compatibility
    parameters_ph::NamedTuple  # Nested: (flat, transformed, natural, unflatten)
    hazards::Vector{_MarkovHazard}
    totalhazards::Vector{_TotalHazard}
    tmat::Matrix{Int64}
    emat::Matrix{Float64}
    hazkeys::Dict{Symbol, Int64}
    subjectindices::Vector{Vector{Int64}}
    SubjectWeights::Vector{Float64}
    ObservationWeights::Union{Nothing, Vector{Float64}}
    CensoringPatterns::Matrix{Float64}
    markovsurrogate::MarkovSurrogate
    modelcall::NamedTuple
end

"""
MultistateMarkovModelCensored(data::DataFrame, parameters::VectorOfVectors,hazards::Vector{_Hazard}, totalhazards::Vector{_TotalHazard},tmat::Matrix{Int64}, hazkeys::Dict{Symbol, Int64}, subjectindices::Vector{Vector{Int64}})

Struct that fully specifies a multistate Markov process with some censored states, used with panel data.

# Phase 3 Change: Now mutable to allow `parameters_ph` updates
"""
mutable struct MultistateMarkovModelCensored <: MultistateMarkovProcess
    data::DataFrame
    parameters::VectorOfVectors  # Legacy - keep for Phase 1 compatibility
    parameters_ph::NamedTuple  # Nested: (flat, transformed, natural, unflatten)
    hazards::Vector{_MarkovHazard}
    totalhazards::Vector{_TotalHazard}
    tmat::Matrix{Int64}
    emat::Matrix{Float64}
    hazkeys::Dict{Symbol, Int64}
    subjectindices::Vector{Vector{Int64}}
    SubjectWeights::Vector{Float64}
    ObservationWeights::Union{Nothing, Vector{Float64}}
    CensoringPatterns::Matrix{Float64}
    markovsurrogate::MarkovSurrogate
    modelcall::NamedTuple
end

"""
MultistateSemiMarkovModel(data::DataFrame, parameters::VectorOfVectors,hazards::Vector{_Hazard}, totalhazards::Vector{_TotalHazard},tmat::Matrix{Int64}, hazkeys::Dict{Symbol, Int64}, subjectindices::Vector{Vector{Int64}})

Struct that fully specifies a multistate semi-Markov process, used with exact death times.

# Phase 3 Change: Now mutable to allow `parameters_ph` updates
"""
mutable struct MultistateSemiMarkovModel <: MultistateSemiMarkovProcess
    data::DataFrame
    parameters::VectorOfVectors  # Legacy - keep for Phase 1 compatibility
    parameters_ph::NamedTuple  # Nested: (flat, transformed, natural, unflatten)
    hazards::Vector{_Hazard}  # Can contain both MarkovHazard and SemiMarkovHazard
    totalhazards::Vector{_TotalHazard}
    tmat::Matrix{Int64}
    emat::Matrix{Float64}
    hazkeys::Dict{Symbol, Int64}
    subjectindices::Vector{Vector{Int64}}
    SubjectWeights::Vector{Float64}
    ObservationWeights::Union{Nothing, Vector{Float64}}
    CensoringPatterns::Matrix{Float64}
    markovsurrogate::MarkovSurrogate
    modelcall::NamedTuple
end

"""
MultistateSemiMarkovModelCensored(data::DataFrame, parameters::VectorOfVectors,hazards::Vector{_Hazard}, totalhazards::Vector{_TotalHazard},tmat::Matrix{Int64}, hazkeys::Dict{Symbol, Int64}, subjectindices::Vector{Vector{Int64}})

Struct that fully specifies a multistate semi-Markov process with some censored states, used with panel data.

# Phase 3 Change: Now mutable to allow `parameters_ph` updates
"""
mutable struct MultistateSemiMarkovModelCensored <: MultistateSemiMarkovProcess
    data::DataFrame
    parameters::VectorOfVectors  # Legacy - keep for Phase 1 compatibility
    parameters_ph::NamedTuple  # Nested: (flat, transformed, natural, unflatten)
    hazards::Vector{_Hazard}  # Can contain both MarkovHazard and SemiMarkovHazard
    totalhazards::Vector{_TotalHazard}
    tmat::Matrix{Int64}
    emat::Matrix{Float64}
    hazkeys::Dict{Symbol, Int64}
    subjectindices::Vector{Vector{Int64}}
    SubjectWeights::Vector{Float64}
    ObservationWeights::Union{Nothing, Vector{Float64}}
    CensoringPatterns::Matrix{Float64}
    markovsurrogate::MarkovSurrogate
    modelcall::NamedTuple
end

"""
    MultistateModelFitted(data::DataFrame, parameters::VectorOfVectors, gradient::Vector{Float64}, hazards::Vector{_Hazard}, totalhazards::Vector{_TotalHazard},tmat::Matrix{Int64}, hazkeys::Dict{Symbol, Int64}, subjectindices::Vector{Vector{Int64}})

Struct that fully specifies a fitted multistate model.

# Phase 3 Change: Now mutable (though less critical for fitted models)
"""
mutable struct MultistateModelFitted <: MultistateProcess
    data::DataFrame
    parameters::VectorOfVectors  # Legacy - keep for Phase 1 compatibility
    parameters_ph::NamedTuple  # NEW Phase 3: (flat, transformed, natural, unflatten)
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
    markovsurrogate::MarkovSurrogate
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
    MPanelData(model::MultistateProcess, books::Tuple)

Struct containing panel data, a model object, and bookkeeping objects. Used in fitting a multistate Markov model to panel data.
"""
struct MPanelData
    model::MultistateProcess
    books::Tuple # tpm_index and tpm_map, from build_tpm_containers
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

