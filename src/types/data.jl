# =============================================================================
# Data Structure Type Definitions
# =============================================================================
# Types for sample paths, panel data, exact data, caches, and related structures.
# =============================================================================

using DataFrames
using ExponentialUtilities

# =============================================================================
# Sample Path Types
# =============================================================================

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

# =============================================================================
# Exact Data Types
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
Used for computing the variance-covariance matrix via autodiff.
"""
struct ExactDataAD
    path::Vector{SamplePath}
    samplingweight::Vector{Float64}
    hazards::Vector{<:_Hazard}
    model::MultistateProcess
end

# =============================================================================
# Panel Data Types
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
- `pars_cache`: Mutable parameter vectors for in-place updates
- `covars_cache`: Pre-extracted covariates per unique covariate pattern
- `hazard_rates_cache`: Pre-computed hazard rates per (pattern, hazard)
- `eigen_cache`: Cached eigendecompositions for fast matrix exponentials
- `dt_values`: Unique Δt values per covariate pattern
"""
mutable struct TPMCache
    hazmat_book::Vector{Matrix{Float64}}
    tpm_book::Vector{Vector{Matrix{Float64}}}
    exp_cache::Any  # ExponentialUtilities cache (type varies)
    q_work::Matrix{Float64}
    lmat_work::Matrix{Float64}
    pars_cache::Vector{Vector{Float64}}
    covars_cache::Vector{Vector{NamedTuple}}
    hazard_rates_cache::Vector{Vector{Float64}}
    eigen_cache::Vector{Union{Nothing, Tuple{Matrix{Float64}, Vector{ComplexF64}, Matrix{ComplexF64}}}}
    dt_values::Vector{Vector{Float64}}
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
    covars_cache = Vector{Vector{NamedTuple}}(undef, npatterns)
    for p in 1:npatterns
        datind = tpm_index[p].datind[1]
        row = model_data[datind, :]
        covars_cache[p] = [_extract_covars_or_empty(row, h.covar_names) for h in hazards]
    end
    
    # Pre-allocate hazard rates cache
    hazard_rates_cache = [zeros(Float64, nhazards) for _ in 1:npatterns]
    
    # Eigendecomposition cache - initially empty
    eigen_cache = Vector{Union{Nothing, Tuple{Matrix{Float64}, Vector{ComplexF64}, Matrix{ComplexF64}}}}(undef, npatterns)
    fill!(eigen_cache, nothing)
    
    # Extract Δt values for each pattern
    dt_values = [collect(Float64, idx.tstop) for idx in tpm_index]
    
    TPMCache(hazmat_book, tpm_book, exp_cache, q_work, lmat_work,
             pars_cache, covars_cache, hazard_rates_cache, eigen_cache, dt_values)
end

# Helper to extract covariates or return empty NamedTuple
@inline function _extract_covars_or_empty(row::DataFrameRow, covar_names::Vector{Symbol})
    isempty(covar_names) && return NamedTuple()
    
    for cname in covar_names
        cname_str = String(cname)
        if occursin(" & ", cname_str) || !hasproperty(row, cname)
            return NamedTuple()
        end
    end
    
    values = Tuple(getproperty(row, cname) for cname in covar_names)
    return NamedTuple{Tuple(covar_names)}(values)
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
    
    pars_cache = Vector{Vector{Float64}}()
    covars_cache = Vector{Vector{NamedTuple}}()
    hazard_rates_cache = Vector{Vector{Float64}}()
    
    eigen_cache = Vector{Union{Nothing, Tuple{Matrix{Float64}, Vector{ComplexF64}, Matrix{ComplexF64}}}}(undef, npatterns)
    fill!(eigen_cache, nothing)
    dt_values = [collect(Float64, idx.tstop) for idx in tpm_index]
    
    TPMCache(hazmat_book, tpm_book, exp_cache, q_work, lmat_work,
             pars_cache, covars_cache, hazard_rates_cache, eigen_cache, dt_values)
end

"""
    MPanelData(model::MultistateProcess, books::Tuple)

Struct containing panel data, a model object, and bookkeeping objects. 
Used in fitting a multistate Markov model to panel data.

# Fields
- `model`: The multistate model
- `books`: Bookkeeping tuple (tpm_index, tpm_map) from `build_tpm_mapping`
- `columns`: Pre-extracted DataFrame column accessors for allocation-free access
- `cache`: Pre-allocated arrays for Float64 likelihood evaluations
"""
struct MPanelData
    model::MultistateProcess
    books::Tuple
    columns::MPanelDataColumnAccessor
    cache::TPMCache
end

function MPanelData(model::MultistateProcess, books::Tuple)
    tpm_map = books[2]
    columns = MPanelDataColumnAccessor(model.data, tpm_map)
    cache = TPMCache(model.tmat, books[1], model.data, model.hazards)
    MPanelData(model, books, columns, cache)
end

"""
    SMPanelData(model::MultistateProcess, paths, ImportanceWeights)

Struct containing panel data, a model object, and bookkeeping objects. 
Used in fitting a multistate semi-Markov model to panel data via MCEM.
"""
struct SMPanelData
    model::MultistateProcess
    paths::Vector{Vector{SamplePath}}
    ImportanceWeights::Vector{Vector{Float64}}
end

# =============================================================================
# Fused Likelihood Data Structures
# =============================================================================
# NOTE: LightweightInterval, SubjectCovarCache, TVCIntervalWorkspace, and
# get_tvc_workspace() are defined in common.jl (the authoritative location).
# These were moved there for proper include order with likelihoods.jl.
# =============================================================================

