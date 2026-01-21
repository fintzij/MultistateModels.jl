# =============================================================================
# Data Containers for Multistate Models
# =============================================================================
#
# Containers for observed data, sample paths, panel data, and likelihood caches.
# These are the "data" types that hold information for model fitting.
#
# =============================================================================

# =============================================================================
# Sample Paths
# =============================================================================

"""
    SamplePath(subj, times, states)

A sample path through a multistate model.

# Fields
- `subj::Int64`: Subject identifier
- `times::Vector{Float64}`: Transition times
- `states::Vector{Int64}`: State sequence

# Requirements
`length(times)` must equal `length(states)`.
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

# =============================================================================
# Exact Data (Fully Observed Paths)
# =============================================================================

"""
    ExactData(model::MultistateProcess, samplepaths::Array{SamplePath})

Struct containing exactly observed sample paths and a model object. 
Used in fitting a multistate model to completely observed data.
"""
struct ExactData
    model::MultistateProcess
    paths::Array{SamplePath}
end

"""
    ExactDataAD(model::MultistateProcess, samplepaths::Array{SamplePath})

Struct containing exactly observed sample paths and a model object. 
Used in fitting a multistate model to completely observed data. 
Used for computing the variance-covariance matrix via autodiff.
"""
struct ExactDataAD
    path::Vector{SamplePath}
    samplingweight::Vector{Float64}
    hazards::Vector{<:_Hazard}
    model::MultistateProcess
end

# =============================================================================
# Panel Data Column Accessors
# =============================================================================

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

# =============================================================================
# TPM Cache (Transition Probability Matrix Cache)
# =============================================================================

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
- `eigen_cache`: Cached eigendecompositions for fast matrix exponentials with multiple Δt (legacy)
- `schur_cache`: Workspace for Schur decomposition-based matrix exponentials
- `dt_values`: Unique Δt values per covariate pattern for batched matrix exp
"""

"""
    SchurCache

Pre-allocated workspace for Schur decomposition-based matrix exponentials.

Used by `compute_tmat_batched!` to efficiently compute exp(Q * Δt) for
multiple Δt values using a single Schur decomposition. The key optimization
is computing the Schur decomposition once and reusing it for all Δt values.

# Fields
- `Q_work`: Work matrix for in-place Schur decomposition
- `E_work`: Work matrix for triangular matrix exponential exp(T * Δt)
- `tmp_work`: Work matrix for intermediate multiplication U * E

# Performance
Provides speedup vs standard matrix exponential when computing multiple
Δt values for the same Q matrix (common in Markov likelihood).
Numerically stable for defective matrices (repeated eigenvalues).
"""
struct SchurCache
    Q_work::Matrix{Float64}   # Copy of Q for in-place schur!
    E_work::Matrix{Float64}   # exp(T * dt) where T is upper triangular
    tmp_work::Matrix{Float64} # Intermediate result U * E
end

function SchurCache(n::Int)
    SchurCache(
        Matrix{Float64}(undef, n, n),
        Matrix{Float64}(undef, n, n),
        Matrix{Float64}(undef, n, n)
    )
end

"""
    CachedSchurDecomposition

Pre-computed Schur decomposition for efficient matrix exponential computation.

Stores Q = U * T * U' where T is upper triangular, allowing efficient
computation of exp(Q * Δt) = U * exp(T * Δt) * U' for arbitrary Δt values.

# Fields
- `T::Matrix{Float64}`: Upper triangular Schur form
- `U::Matrix{Float64}`: Unitary matrix from Schur decomposition
- `E_work::Matrix{Float64}`: Pre-allocated workspace for exp(T * Δt)
- `tmp_work::Matrix{Float64}`: Pre-allocated workspace for U * exp(T * Δt)

# Performance
The Schur decomposition is O(n³) and computed once. Each subsequent
exp(Q * Δt) requires only exp(T * Δt) + two matrix multiplications,
which is faster than recomputing the full decomposition.
"""
struct CachedSchurDecomposition
    T::Matrix{Float64}        # Upper triangular Schur form
    U::Matrix{Float64}        # Unitary matrix
    E_work::Matrix{Float64}   # Workspace for exp(T * dt)
    tmp_work::Matrix{Float64} # Workspace for U * E
end

"""
    CachedSchurDecomposition(Q::Matrix{Float64})

Pre-compute and cache the Schur decomposition of Q.
"""
function CachedSchurDecomposition(Q::Matrix{Float64})
    n = size(Q, 1)
    F = schur(Q)
    CachedSchurDecomposition(
        copy(F.T),
        copy(F.Z),
        Matrix{Float64}(undef, n, n),
        Matrix{Float64}(undef, n, n)
    )
end

"""
    compute_tpm_from_schur!(P, cached::CachedSchurDecomposition, dt)

Compute P = exp(Q * dt) using cached Schur decomposition (in-place).

# Arguments
- `P::Matrix{Float64}`: Output matrix (modified in-place)
- `cached::CachedSchurDecomposition`: Pre-computed Schur decomposition
- `dt::Float64`: Time interval (must be non-negative)

# Returns
Nothing (modifies P in-place)

# Throws
- `AssertionError`: if `dt < 0`
"""
function compute_tpm_from_schur!(P::Matrix{Float64}, cached::CachedSchurDecomposition, dt::Float64)
    @assert dt >= 0.0 "Time interval dt must be non-negative, got dt=$dt"
    n = size(cached.T, 1)
    if dt == 0.0
        # Identity matrix
        fill!(P, 0.0)
        for i in 1:n
            P[i, i] = 1.0
        end
    else
        # exp(T * dt)
        cached.E_work .= exp(cached.T * dt)
        # P = U * exp(T*dt) * U'
        mul!(cached.tmp_work, cached.U, cached.E_work)
        mul!(P, cached.tmp_work, cached.U')
    end
    # Debug assertion: verify TPM row sums ≈ 1.0 (M11_P2)
    MSM_DEBUG_ASSERTIONS && @assert all(isapprox.(sum(P, dims=2), 1.0, atol=TPM_ROW_SUM_TOL)) "TPM row sums must be ≈ 1.0, got $(vec(sum(P, dims=2)))"
    return nothing
end

"""
    compute_tpm_from_schur(cached::CachedSchurDecomposition, dt)

Compute P = exp(Q * dt) using cached Schur decomposition.

# Arguments
- `cached::CachedSchurDecomposition`: Pre-computed Schur decomposition
- `dt::Float64`: Time interval (must be non-negative)

# Returns
- `Matrix{Float64}`: Transition probability matrix

# Throws
- `AssertionError`: if `dt < 0`
"""
function compute_tpm_from_schur(cached::CachedSchurDecomposition, dt::Float64)
    @assert dt >= 0.0 "Time interval dt must be non-negative, got dt=$dt"
    n = size(cached.T, 1)
    P = Matrix{Float64}(undef, n, n)
    compute_tpm_from_schur!(P, cached, dt)
    return P
end

# =============================================================================
# TPM Cache with Version Counter (M17_P2)
# =============================================================================

"""
    TPMCache

Mutable cache for pre-allocated arrays used in Markov likelihood computation.
Stores hazard matrix book, TPM book, matrix exponential cache, and work arrays.

# Cache Invalidation (M17_P2)

The cache includes a `version` counter that should be incremented whenever
hazard parameters change. Users of the cache should check `version` to detect
stale cached values:

```julia
# At likelihood computation start:
if cache.version != expected_version
    # Recompute TPMs
end

# After updating parameters:
cache.version += 1
```

The cache is automatically invalidated (version incremented) when parameters
are modified via `set_parameters!`.

# Fields
- `version::Int`: Version counter for cache invalidation
- `hazmat_book`: Pre-allocated hazard intensity matrices (one per covariate pattern)
- `tpm_book`: Pre-allocated transition probability matrices (nested: pattern × time interval)
- `exp_cache`: Pre-allocated workspace for matrix exponential computation
- `q_work`: Work matrix for forward algorithm (S × S)
- `lmat_work`: Work matrix for forward algorithm likelihood (S × max_obs+1)
- `pars_cache`: Mutable parameter vectors for in-place updates
- `covars_cache`: Pre-extracted covariates per unique covariate pattern
- `hazard_rates_cache`: Pre-computed hazard rates per (pattern, hazard)
- `eigen_cache`: Cached eigendecompositions (legacy)
- `schur_cache`: Workspace for Schur decomposition-based matrix exponentials
- `dt_values`: Unique Δt values per covariate pattern
"""
mutable struct TPMCache
    version::Int  # Cache invalidation counter (M17_P2)
    hazmat_book::Vector{Matrix{Float64}}
    tpm_book::Vector{Vector{Matrix{Float64}}}
    exp_cache::Nothing  # ExponentialUtilities cache for ExpMethodGeneric (always Nothing)
    q_work::Matrix{Float64}
    lmat_work::Matrix{Float64}
    # New caches for additional optimizations
    pars_cache::Vector{Vector{Float64}}  # Mutable parameter vectors per hazard
    covars_cache::Vector{Vector{NamedTuple}}  # Pre-extracted covariates per pattern per hazard
    hazard_rates_cache::Vector{Vector{Float64}}  # Pre-computed rates per pattern per hazard
    # Eigendecomposition cache for batched matrix exponentials (legacy, retained for compatibility)
    eigen_cache::Vector{Union{Nothing, Tuple{Matrix{Float64}, Vector{ComplexF64}, Matrix{ComplexF64}}}}
    # Schur decomposition workspace for fast batched matrix exponentials
    schur_cache::SchurCache
    dt_values::Vector{Vector{Float64}}  # Unique Δt values per pattern
end

"""
    invalidate_cache!(cache::TPMCache)

Increment the cache version to invalidate all cached TPM values.

Call this after modifying hazard parameters to ensure subsequent likelihood
computations recompute the transition probability matrices.

# Example
```julia
set_parameters!(model, new_params)
invalidate_cache!(model_data.cache)  # If using MPanelData
```
"""
function invalidate_cache!(cache::TPMCache)
    cache.version += 1
    return cache
end

"""
    is_cache_valid(cache::TPMCache, expected_version::Int) -> Bool

Check if the cache is valid for the expected version.

# Arguments
- `cache::TPMCache`: The TPM cache to check
- `expected_version::Int`: The version the caller expects

# Returns
- `true` if cache.version == expected_version
"""
@inline is_cache_valid(cache::TPMCache, expected_version::Int) = cache.version == expected_version

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
    
    # Schur decomposition workspace for batched matrix exponentials
    schur_cache = SchurCache(nstates)
    
    # Initialize with version 0
    TPMCache(0, hazmat_book, tpm_book, exp_cache, q_work, lmat_work,
             pars_cache, covars_cache, hazard_rates_cache, eigen_cache, schur_cache, dt_values)
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
    
    # Schur decomposition workspace
    schur_cache = SchurCache(nstates)
    
    # Initialize with version 0 (M17_P2)
    TPMCache(0, hazmat_book, tpm_book, exp_cache, q_work, lmat_work,
             pars_cache, covars_cache, hazard_rates_cache, eigen_cache, schur_cache, dt_values)
end

# =============================================================================
# Panel Data Containers
# =============================================================================

"""
    MPanelData(model::MultistateProcess, books::Tuple)

Struct containing panel data, a model object, and bookkeeping objects. 
Used in fitting a multistate Markov model to panel data.

# Fields
- `model`: The multistate model
- `books`: Bookkeeping tuple (tpm_index, tpm_map) from `build_tpm_mapping`
- `columns`: Pre-extracted DataFrame column accessors for allocation-free access
- `cache`: Pre-allocated arrays for Float64 likelihood evaluations

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
    SMPanelData(model::MultistateProcess, paths, ImportanceWeights)

Struct containing panel data, a model object, and bookkeeping objects. 
Used in fitting a multistate semi-Markov model to panel data via MCEM.

# Fields
- `model`: The multistate model
- `paths`: Vector of sampled paths per subject
- `ImportanceWeights`: Importance weights for sampled paths
"""
struct SMPanelData
    model::MultistateProcess
    paths::Vector{Vector{SamplePath}}
    ImportanceWeights::Vector{Vector{Float64}}
end

# =============================================================================
# Lightweight Interval for Fused Likelihood
# =============================================================================

"""
    LightweightInterval

Minimal representation of a likelihood interval without DataFrame overhead.
Used in optimized batched likelihood computation to avoid allocation.

# Fields
- `lb`: Lower bound (sojourn time)
- `ub`: Upper bound (sojourn + increment)
- `statefrom`: Origin state
- `stateto`: Destination state
- `covar_row_idx`: Index into subject's covariate data
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

# Fields
- `tstart`: Start times for covariate intervals
- `covar_data`: Covariate columns only (no id, tstart, tstop, etc.)
"""
struct SubjectCovarCache
    tstart::Vector{Float64}   # Start times for covariate intervals
    covar_data::DataFrame     # Covariate columns only (no id, tstart, tstop, etc.)
end

"""
    TVCIntervalWorkspace

Pre-allocated workspace for computing TVC intervals to reduce allocations.
Used in semi-Markov likelihood when time-varying covariates are present.

# Fields
- `change_times`: Pre-allocated buffer for covariate change times
- `utimes`: Pre-allocated buffer for unique times
- `intervals`: Pre-allocated buffer for intervals
- `sojourns`: Pre-allocated buffer for sojourn times
- `pathinds`: Pre-allocated buffer for path indices
- `datinds`: Pre-allocated buffer for data indices
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

"""
    get_tvc_workspace()::TVCIntervalWorkspace

Get or create thread-local TVCIntervalWorkspace.

Thread-safe: The entire get-or-create operation is inside the lock to prevent
TOCTOU race conditions when multiple threads initialize simultaneously.
"""
function get_tvc_workspace()::TVCIntervalWorkspace
    tid = Threads.threadid()
    # Entire get-or-create must be inside lock to avoid TOCTOU race (C7_P2 fix)
    lock(TVC_WORKSPACE_LOCK) do
        return get!(TVC_INTERVAL_WORKSPACES, tid) do
            TVCIntervalWorkspace(200)
        end
    end
end

"""
    clear_tvc_workspaces!()

Clear all thread-local TVC interval workspaces, freeing memory.

This function should be called to reclaim memory in long-running processes
after model fitting is complete. Thread-safe.

See also: [`clear_all_workspaces!`](@ref) for clearing all workspace types.
"""
function clear_tvc_workspaces!()
    lock(TVC_WORKSPACE_LOCK) do
        empty!(TVC_INTERVAL_WORKSPACES)
    end
    return nothing
end
