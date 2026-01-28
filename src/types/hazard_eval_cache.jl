# =============================================================================
# Hazard Evaluation Cache Types
# =============================================================================
#
# Unified caching infrastructure for hazard evaluation.
#
# Design principles:
# 1. Cache Float64 state, rebuild for Dual types (AD compatibility)
# 2. Type-stable design (no Ref{Any})
# 3. Thread-safe (no global mutable state)
# 4. Integrates with existing TimeTransformContext and SubjectCovarCache
#
# =============================================================================

# =============================================================================
# AD Mode Detection
# =============================================================================

"""
    is_ad_mode(::Type{T}) -> Bool

Detect if type T is an automatic differentiation Dual type.
Returns `true` for ForwardDiff.Dual, ReverseDiff.TrackedReal, etc.
Returns `false` for Float64 and other concrete numeric types.

Used to bypass caching during gradient/Hessian computation since
cached Dual objects would break the computation graph.
"""
@inline is_ad_mode(::Type{Float64}) = false
@inline is_ad_mode(::Type{Float32}) = false
@inline is_ad_mode(::Type{T}) where {T<:Integer} = false
@inline is_ad_mode(::Type{T}) where {T} = true  # Conservative: assume AD for unknown types

# Specific overloads for ForwardDiff (most common)
@inline is_ad_mode(::Type{<:ForwardDiff.Dual}) = true

"""
    is_ad_mode(x) -> Bool

Value-based version that dispatches on eltype.
"""
@inline is_ad_mode(x::AbstractArray) = is_ad_mode(eltype(x))
@inline is_ad_mode(x::Number) = is_ad_mode(typeof(x))

# =============================================================================
# Parameter Hashing
# =============================================================================

"""
    hash_parameters(pars::AbstractVector) -> UInt64

Compute a hash of parameter values for cache invalidation.
"""
@inline function hash_parameters(pars::AbstractVector{T}) where T
    return hash(pars)
end

"""
    hash_parameters(pars::NamedTuple) -> UInt64

Hash nested parameter NamedTuple by flattening to values.
"""
@inline function hash_parameters(pars::NamedTuple)
    return hash(values(pars))
end

# =============================================================================
# Hazard Evaluation State Types
# =============================================================================

"""
    HazardEvalState

Abstract type for hazard-specific cached evaluation state.
"""
abstract type HazardEvalState end

"""
    NoEvalState <: HazardEvalState

Marker type for hazards that don't need cached evaluation state.
Used for parametric hazards (exponential, Weibull, Gompertz).
"""
struct NoEvalState <: HazardEvalState end

"""
    SplineEvalState <: HazardEvalState

Cached spline objects for spline hazard evaluation.

# Fields
- `spline_ext`: Cached SplineExtrapolation object (or nothing if not built)
- `cumhaz_spline`: Cached integral of spline for cumulative hazard (or nothing)
"""
mutable struct SplineEvalState <: HazardEvalState
    spline_ext::Any      # Union{Nothing, SplineExtrapolation}
    cumhaz_spline::Any   # Union{Nothing, Spline} - the integral
end

SplineEvalState() = SplineEvalState(nothing, nothing)

"""
    reset!(state::SplineEvalState)

Clear cached spline objects, forcing rebuild on next evaluation.
"""
function reset!(state::SplineEvalState)
    state.spline_ext = nothing
    state.cumhaz_spline = nothing
    return state
end

reset!(::NoEvalState) = NoEvalState()

# =============================================================================
# Hazard Evaluation Cache
# =============================================================================

"""
    HazardEvalCache{S<:HazardEvalState}

Unified cache for a single hazard's evaluation state.
Tracks parameter hash for invalidation and holds hazard-specific state.

# Type Parameters
- `S`: Concrete HazardEvalState subtype (NoEvalState or SplineEvalState)

# Fields
- `pars_hash::UInt64`: Hash of parameters when state was built
- `state::S`: Hazard-specific cached state
"""
mutable struct HazardEvalCache{S<:HazardEvalState}
    pars_hash::UInt64
    state::S
end

"""
    HazardEvalCache(state::S) where S<:HazardEvalState

Create cache with zero hash (will rebuild on first use).
"""
HazardEvalCache(state::S) where {S<:HazardEvalState} = HazardEvalCache{S}(UInt64(0), state)

"""
    needs_rebuild(cache::HazardEvalCache, pars) -> Bool

Check if cache needs to be rebuilt due to parameter change.
Always returns true for AD mode (Dual types).
"""
@inline function needs_rebuild(cache::HazardEvalCache, pars)
    if is_ad_mode(pars)
        return true
    end
    return hash_parameters(pars) != cache.pars_hash
end

"""
    update_hash!(cache::HazardEvalCache, pars)

Update the parameter hash after rebuilding state.
Only updates for Float64 (non-AD) parameters.
"""
@inline function update_hash!(cache::HazardEvalCache, pars)
    if !is_ad_mode(pars)
        cache.pars_hash = hash_parameters(pars)
    end
    return cache
end

"""
    reset!(cache::HazardEvalCache)

Reset cache state and hash, forcing rebuild on next use.
"""
function reset!(cache::HazardEvalCache)
    cache.pars_hash = UInt64(0)
    reset!(cache.state)
    return cache
end

# =============================================================================
# Hazard Evaluation Context
# =============================================================================

"""
    HazardEvalContext

Container for all evaluation caches within a likelihood computation.

# Fields
- `hazard_caches::Vector{HazardEvalCache}`: One cache per hazard
- `subject_covars::Vector{SubjectCovarCache}`: Pre-built subject covariate data
- `covar_names_per_hazard::Vector{Vector{Symbol}}`: Covariate names for each hazard
- `pars_hash::UInt64`: Global parameter hash for bulk invalidation
"""
mutable struct HazardEvalContext
    hazard_caches::Vector{HazardEvalCache}
    subject_covars::Vector{SubjectCovarCache}
    covar_names_per_hazard::Vector{Vector{Symbol}}
    pars_hash::UInt64
end

"""
    HazardEvalContext(n_hazards::Int, n_subjects::Int)

Create empty context with pre-allocated vectors.
"""
function HazardEvalContext(n_hazards::Int, n_subjects::Int)
    hazard_caches = Vector{HazardEvalCache}(undef, n_hazards)
    subject_covars = Vector{SubjectCovarCache}(undef, n_subjects)
    covar_names = Vector{Vector{Symbol}}(undef, n_hazards)
    return HazardEvalContext(hazard_caches, subject_covars, covar_names, UInt64(0))
end

"""
    invalidate_if_needed!(ctx::HazardEvalContext, pars)

Check if parameters changed and invalidate all hazard caches if so.
Returns `true` if invalidation occurred.
"""
function invalidate_if_needed!(ctx::HazardEvalContext, pars)
    if is_ad_mode(pars)
        return true
    end
    
    new_hash = hash_parameters(pars)
    if new_hash != ctx.pars_hash
        for cache in ctx.hazard_caches
            reset!(cache)
        end
        ctx.pars_hash = new_hash
        return true
    end
    return false
end

"""
    get_hazard_cache(ctx::HazardEvalContext, hazard_idx::Int) -> HazardEvalCache
"""
@inline function get_hazard_cache(ctx::HazardEvalContext, hazard_idx::Int)
    return ctx.hazard_caches[hazard_idx]
end

"""
    get_subject_covar(ctx::HazardEvalContext, subject_idx::Int) -> SubjectCovarCache
"""
@inline function get_subject_covar(ctx::HazardEvalContext, subject_idx::Int)
    return ctx.subject_covars[subject_idx]
end

"""
    get_covar_names(ctx::HazardEvalContext, hazard_idx::Int) -> Vector{Symbol}
"""
@inline function get_covar_names(ctx::HazardEvalContext, hazard_idx::Int)
    return ctx.covar_names_per_hazard[hazard_idx]
end

# =============================================================================
# Factory Functions
# =============================================================================

"""
    create_hazard_cache(hazard::_Hazard) -> HazardEvalCache

Create appropriate cache type based on hazard family.
Spline hazards get SplineEvalState, parametric get NoEvalState.
"""
function create_hazard_cache(hazard::_Hazard)
    if hazard isa RuntimeSplineHazard
        return HazardEvalCache(SplineEvalState())
    else
        return HazardEvalCache(NoEvalState())
    end
end

"""
    create_hazard_caches(hazards::Vector{<:_Hazard}) -> Vector{HazardEvalCache}

Create caches for all hazards in a model.
"""
function create_hazard_caches(hazards::Vector{<:_Hazard})
    return [create_hazard_cache(h) for h in hazards]
end

"""
    build_hazard_eval_context(model::MultistateProcess) -> HazardEvalContext

Build complete evaluation context from a model.
Creates hazard caches, subject covariate caches, and extracts covariate names.

This is the main entry point - called once at the start of fitting and passed
to likelihood functions.

# Arguments
- `model`: MultistateProcess containing hazards and data

# Returns
HazardEvalContext ready for use in likelihood computation.

# Example
```julia
ctx = build_hazard_eval_context(model)
ll = loglik_exact(params, data; hazard_eval_ctx=ctx)
```
"""
function build_hazard_eval_context(model::MultistateProcess)
    hazards = model.hazards
    n_hazards = length(hazards)
    n_subjects = length(model.subjectindices)
    
    ctx = HazardEvalContext(n_hazards, n_subjects)
    
    # Create hazard caches
    for (i, hazard) in enumerate(hazards)
        ctx.hazard_caches[i] = create_hazard_cache(hazard)
    end
    
    # Build covariate names per hazard
    for (i, hazard) in enumerate(hazards)
        ctx.covar_names_per_hazard[i] = if hasfield(typeof(hazard), :covar_names)
            collect(hazard.covar_names)
        else
            extract_covar_names(hazard.parnames)
        end
    end
    
    # Build subject covariate caches (reuse existing function)
    ctx.subject_covars = build_subject_covar_cache(model)
    
    return ctx
end

"""
    build_hazard_eval_context(hazards::Vector{<:_Hazard}, samplepaths, data::DataFrame) -> HazardEvalContext

Build complete evaluation context for a model.
Creates hazard caches, subject covariate caches, and extracts covariate names.

# Arguments
- `hazards`: Vector of hazard structs from model
- `samplepaths`: Vector of SamplePath (for subject iteration)
- `data`: Original DataFrame with covariate data

# Returns
HazardEvalContext ready for use in likelihood computation.
"""
function build_hazard_eval_context(hazards::Vector{<:_Hazard}, samplepaths, data::DataFrame)
    n_hazards = length(hazards)
    n_subjects = length(samplepaths)
    
    ctx = HazardEvalContext(n_hazards, n_subjects)
    
    # Create hazard caches
    for (i, hazard) in enumerate(hazards)
        ctx.hazard_caches[i] = create_hazard_cache(hazard)
    end
    
    # Build covariate names per hazard
    for (i, hazard) in enumerate(hazards)
        ctx.covar_names_per_hazard[i] = collect(hazard.covar_names)
    end
    
    # Build subject covariate caches
    # Identify covariate columns (exclude standard data columns)
    covar_cols = setdiff(Symbol.(names(data)), [:id, :tstart, :tstop, :statefrom, :stateto, :obstype])
    has_covars = !isempty(covar_cols)
    
    for (i, path) in enumerate(samplepaths)
        subj_id = path.subj
        subj_rows = filter(row -> row.id == subj_id, data)
        
        if has_covars && nrow(subj_rows) > 0
            covar_data = subj_rows[:, covar_cols]
            tstart = collect(subj_rows.tstart)
        else
            covar_data = DataFrame()
            tstart = Float64[]
        end
        
        ctx.subject_covars[i] = SubjectCovarCache(tstart, covar_data)
    end    
    return ctx
end