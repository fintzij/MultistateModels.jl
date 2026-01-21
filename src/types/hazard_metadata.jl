# =============================================================================
# Hazard Metadata and Time Transform Caches
# =============================================================================
#
# Supporting types for hazard configuration and Tang-style shared trajectory
# caching. These are internal types used during hazard evaluation.
#
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
# Time Transform Cache Types (Tang-style shared trajectories)
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

Composite key describing a shared baseline. Multiple hazards can share a
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

# =============================================================================
# Time Transform Helpers
# =============================================================================

@inline function _time_column_eltype(time_data)
    isnothing(time_data) && return Float64
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

# =============================================================================
# Optional TimeTransformContext Wrapper (M8_P1)
# =============================================================================

"""
    OptionalTimeTransformContext{LinType,TimeType}

Type-stable wrapper for optional TimeTransformContext.

This struct eliminates the Union{Nothing, TimeTransformContext{L,T}} return type
from `maybe_time_transform_context`, providing better type inference and
allowing the compiler to generate more efficient code.

# Fields
- `has_context::Bool`: Whether a context is present
- `context::TimeTransformContext{LinType,TimeType}`: The context (may be uninitialized if !has_context)

# Usage
```julia
opt_ctx = maybe_time_transform_context(pars, data, hazards)
if opt_ctx.has_context
    ctx = opt_ctx.context
    # use ctx...
end
```
"""
struct OptionalTimeTransformContext{LinType,TimeType}
    has_context::Bool
    context::TimeTransformContext{LinType,TimeType}
    
    # Constructor for "no context" case - uses placeholder types
    function OptionalTimeTransformContext{L,T}() where {L,T}
        # Create a minimal placeholder context (won't be used)
        placeholder = TimeTransformContext(L, T, 0)
        new{L,T}(false, placeholder)
    end
    
    # Constructor for "has context" case
    function OptionalTimeTransformContext(ctx::TimeTransformContext{L,T}) where {L,T}
        new{L,T}(true, ctx)
    end
end

# Default placeholder types for the "nothing" case
const _DEFAULT_OPTIONAL_CONTEXT = OptionalTimeTransformContext{Float64, Float64}()

"""
    maybe_time_transform_context(pars, subjectdata, hazards; time_column = :sojourn)

Return a `TimeTransformContext` when transforms are enabled for at least one
hazard and caching remains active. Callers may override `time_column` when the
relevant observation durations live under a different column name. When no
hazards opt into time transforms (or caching is globally disabled), the
function returns `nothing` so downstream code can skip the shared-cache branch.

# Type Stability Note
For type-stable code paths, consider using `maybe_time_transform_context_stable`
which returns an `OptionalTimeTransformContext` wrapper instead of `Union{Nothing, ...}`.
"""
function maybe_time_transform_context(pars,
                                      subjectdata,
                                      hazards::Vector{<:_Hazard};
                                      time_column::Symbol = :sojourn)
    _time_transform_cache_enabled() || return nothing
    idx = findfirst(h -> h.metadata.time_transform, hazards)
    isnothing(idx) && return nothing

    # Extract scalar element type from nested parameter structure
    lin_type = _param_scalar_eltype(pars[idx])
    time_data = (subjectdata !== nothing && hasproperty(subjectdata, time_column)) ?
        getproperty(subjectdata, time_column) : nothing
    time_type = _time_column_eltype(time_data)
    # Promote time_type to lin_type to handle AFT where time becomes Dual
    actual_time_type = promote_type(lin_type, time_type)
    return TimeTransformContext(lin_type, actual_time_type, length(hazards))
end

"""
    maybe_time_transform_context_stable(pars, subjectdata, hazards; time_column = :sojourn)

Type-stable version of `maybe_time_transform_context`.

Returns an `OptionalTimeTransformContext` wrapper that avoids the Union{Nothing, ...}
return type for better type inference in hot paths. Check `.has_context` before
accessing `.context`.

# Example
```julia
opt_ctx = maybe_time_transform_context_stable(pars, data, hazards)
if opt_ctx.has_context
    ctx = opt_ctx.context
    # use ctx in hot loop...
end
```
"""
function maybe_time_transform_context_stable(pars,
                                             subjectdata,
                                             hazards::Vector{<:_Hazard};
                                             time_column::Symbol = :sojourn)
    _time_transform_cache_enabled() || return _DEFAULT_OPTIONAL_CONTEXT
    idx = findfirst(h -> h.metadata.time_transform, hazards)
    isnothing(idx) && return _DEFAULT_OPTIONAL_CONTEXT

    # Extract scalar element type from nested parameter structure
    lin_type = _param_scalar_eltype(pars[idx])
    time_data = (subjectdata !== nothing && hasproperty(subjectdata, time_column)) ?
        getproperty(subjectdata, time_column) : nothing
    time_type = _time_column_eltype(time_data)
    actual_time_type = promote_type(lin_type, time_type)
    ctx = TimeTransformContext(lin_type, actual_time_type, length(hazards))
    return OptionalTimeTransformContext(ctx)
end

const _TIME_TRANSFORM_CACHE_ENABLED = Ref(true)

function enable_time_transform_cache!(flag::Bool)
    _TIME_TRANSFORM_CACHE_ENABLED[] = flag
    return _TIME_TRANSFORM_CACHE_ENABLED[]
end

_time_transform_cache_enabled() = _TIME_TRANSFORM_CACHE_ENABLED[]
