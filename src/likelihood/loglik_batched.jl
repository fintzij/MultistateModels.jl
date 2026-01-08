# =============================================================================
# Batched (Hazard-Centric) Likelihood Infrastructure
# =============================================================================
#
# Contents:
# - StackedHazardData: Batched data structure for likelihood
# - ExactDataAD: Container for AD-compatible exact data
# - Batched interval stacking and caching
#
# Split from loglik.jl for maintainability (January 2026)
# =============================================================================

# =============================================================================
# Cached path data for efficient batched likelihood
# =============================================================================
#
# DEVELOPER GUIDE: Batched Likelihood Infrastructure
# ===================================================
#
# This section contains infrastructure for efficient batched likelihood computation,
# particularly useful in MCEM where thousands of paths must be evaluated per iteration.
#
# PRIMARY IMPLEMENTATION:
# - loglik_exact: Fused per-interval computation with full AD support
# - loglik_semi_markov_batched!: For nested Vector{Vector{SamplePath}} (SMPanelData)
#
# Key Concepts:
#
# 1. PATH CACHING (CachedPathData)
#    - Pre-computes DataFrame representations of SamplePaths
#    - Eliminates O(n_hazards × n_paths) redundant `make_subjdat` calls
#    - Use: `cached_paths = cache_path_data(paths, model)`
#
# 2. INTERVAL STACKING (StackedHazardData)
#    - Reorganizes data from path-centric to hazard-centric layout
#    - Groups all intervals where a specific hazard contributes
#    - Enables vectorized evaluation within each hazard
#    - Used by: loglik_semi_markov_batched!
#
# 3. LINEAR PREDICTOR CACHING
#    - Pre-computes Xβ values when parameters are known
#    - Stored in StackedHazardData.linpreds
#    - Passed via `pars=pars` argument to stack_intervals_for_hazard
#
# Data Flow:
#
#   paths → cache_path_data → CachedPathData[]
#                              ↓
#                    stack_intervals_for_hazard (per hazard)
#                              ↓
#                    StackedHazardData (per hazard)
#
# Performance Notes:
# - Batched approach provides ~10-15% speedup on typical datasets
# - Speedup comes from reduced DataFrame creation and path iteration overhead
# - Memory usage is slightly lower due to pre-allocation
#
# =============================================================================

"""
    CachedPathData

Pre-computed DataFrame representation of a SamplePath for batched likelihood computation.

Caching path DataFrames upfront eliminates redundant `make_subjdat` calls that would
otherwise occur O(n_hazards × n_paths) times in the batched likelihood loop.

# Fields
- `subj::Int`: Subject index (same as path.subj)
- `df::DataFrame`: Pre-computed subject data from `make_subjdat(path, subj_dat)`
- `linpreds::Dict{Int,Vector{Float64}}`: Pre-computed linear predictors keyed by hazard index

# Linear Predictor Caching

For PH hazards, the linear predictor `exp(Xβ)` is constant across time for each interval.
Pre-computing these values once per hazard avoids redundant computation in the inner loop.

# Example

```julia
# Pre-allocate all path data once
cached_paths = Vector{CachedPathData}(undef, n_paths)
for (i, path) in enumerate(paths)
    subj_inds = model.subjectindices[path.subj]
    subj_dat = view(model.data, subj_inds, :)
    cached_paths[i] = CachedPathData(path.subj, make_subjdat(path, subj_dat), Dict{Int,Vector{Float64}}())
end

# Then use in stack_intervals_for_hazard
stacked = stack_intervals_for_hazard(h, cached_paths, model, hazards, totalhazards, tmat)
```

See also: [`stack_intervals_for_hazard`](@ref), [`loglik_exact`](@ref)
"""
struct CachedPathData
    subj::Int
    df::DataFrame
    linpreds::Dict{Int,Vector{Float64}}
end

# =============================================================================
# Batched data structures for hazard-centric computation
# =============================================================================

"""
    StackedHazardData

Pre-processed interval data for a single hazard, organized for batched computation.
Contains all intervals across all paths where this hazard contributes to the likelihood.

# Fields
- `lb::Vector{Float64}`: Lower bounds of intervals (sojourn times)
- `ub::Vector{Float64}`: Upper bounds of intervals (sojourn + increment)
- `covars::Vector{NamedTuple}`: Covariates for each interval
- `linpreds::Vector{Float64}`: Pre-computed linear predictors `Xβ` for each interval
- `path_idx::Vector{Int}`: Index into the paths array for accumulating results
- `is_transition::Vector{Bool}`: Whether this interval ends with a transition via this hazard
- `transition_times::Vector{Float64}`: Time of transition (only meaningful when is_transition is true)

# Linear Predictor Caching

For PH hazards, the linear predictor `exp(Xβ)` is constant across time for each interval.
For AFT hazards, the time rescaling factor depends on the linear predictor.
Pre-computing these once per interval avoids redundant computation in the inner loop.

Note: Linear predictors are only computed when `pars` are provided to 
`stack_intervals_for_hazard`. When `pars` is `nothing`, linpreds will be empty.
"""
struct StackedHazardData
    lb::Vector{Float64}
    ub::Vector{Float64}
    covars::Vector{NamedTuple}
    linpreds::Vector{Float64}
    path_idx::Vector{Int}
    is_transition::Vector{Bool}
    transition_times::Vector{Float64}
end

"""
    stack_intervals_for_hazard(hazard_idx::Int, cached_paths::Vector{CachedPathData}, 
                               hazards::Vector{<:_Hazard}, totalhazards::Vector{<:_TotalHazard}, 
                               tmat::Array{Int,2}; pars=nothing)

Pre-process all paths to extract intervals relevant to a specific hazard.
An interval is relevant if:
1. The origin state has this hazard as an exit transition (contributes to survival)
2. The transition at the end of the interval is via this hazard (contributes to transition probability)

This version accepts pre-cached path DataFrames to avoid redundant `make_subjdat` calls.

# Arguments
- `hazard_idx::Int`: Index of the hazard to stack intervals for
- `cached_paths::Vector{CachedPathData}`: Pre-computed path DataFrames
- `hazards::Vector{<:_Hazard}`: All hazards in the model
- `totalhazards::Vector{<:_TotalHazard}`: Total hazard structures by state
- `tmat::Array{Int,2}`: Transition matrix
- `pars`: Optional nested parameters. If provided, linear predictors are pre-computed.

# Returns
`StackedHazardData` with pre-computed intervals and (optionally) linear predictors.
"""
function stack_intervals_for_hazard(hazard_idx::Int, cached_paths::Vector{CachedPathData},
                                    hazards::Vector{<:_Hazard}, totalhazards::Vector{<:_TotalHazard}, 
                                    tmat::Array{Int,2}; pars=nothing)
    # Pre-compute which states have this hazard as an exit component
    n_states = size(tmat, 1)
    hazard_exits_state = falses(n_states)
    for s in 1:n_states
        if totalhazards[s] isa _TotalHazardTransient
            hazard_exits_state[s] = hazard_idx in totalhazards[s].components
        end
    end
    
    # Count intervals to pre-allocate
    n_intervals = 0
    for cpd in cached_paths
        for row_i in 1:nrow(cpd.df)
            if hazard_exits_state[cpd.df.statefrom[row_i]]
                n_intervals += 1
            end
        end
    end
    
    # Pre-allocate arrays
    lb = Vector{Float64}(undef, n_intervals)
    ub = Vector{Float64}(undef, n_intervals)
    covars = Vector{NamedTuple}(undef, n_intervals)
    linpreds = Vector{Float64}(undef, n_intervals)
    path_idx = Vector{Int}(undef, n_intervals)
    is_transition = Vector{Bool}(undef, n_intervals)
    transition_times = Vector{Float64}(undef, n_intervals)
    
    # Fill arrays
    interval_i = 0
    hazard = hazards[hazard_idx]
    covar_names = hasfield(typeof(hazard), :covar_names) ? 
                  hazard.covar_names : 
                  extract_covar_names(hazard.parnames)
    hazard_pars = pars === nothing ? nothing : pars[hazard_idx]
    
    for (path_i, cpd) in enumerate(cached_paths)
        subjdat_df = cpd.df
        
        for row_i in 1:nrow(subjdat_df)
            origin_state = subjdat_df.statefrom[row_i]
            if !hazard_exits_state[origin_state]
                continue
            end
            
            interval_i += 1
            # Use @view to avoid allocation when accessing row
            @views row = subjdat_df[row_i, :]
            
            lb[interval_i] = row.sojourn
            ub[interval_i] = row.sojourn + row.increment
            covar_tuple = extract_covariates_fast(row, covar_names)
            covars[interval_i] = covar_tuple
            path_idx[interval_i] = path_i
            
            # Pre-compute linear predictor if parameters provided
            if hazard_pars !== nothing
                linpreds[interval_i] = _linear_predictor(hazard_pars, covar_tuple, hazard)
            else
                linpreds[interval_i] = 0.0
            end
            
            # Check if transition via this hazard
            dest_state = subjdat_df.stateto[row_i]
            if origin_state != dest_state && tmat[origin_state, dest_state] == hazard_idx
                is_transition[interval_i] = true
                transition_times[interval_i] = ub[interval_i]
            else
                is_transition[interval_i] = false
                transition_times[interval_i] = 0.0  # placeholder
            end
        end
    end
    
    return StackedHazardData(lb, ub, covars, linpreds, path_idx, is_transition, transition_times)
end

# Legacy method for backward compatibility (converts paths to cached format internally)
function stack_intervals_for_hazard(hazard_idx::Int, paths::Vector{SamplePath}, model::MultistateProcess,
                                    hazards::Vector{<:_Hazard}, totalhazards::Vector{<:_TotalHazard}, 
                                    tmat::Array{Int,2}; pars=nothing)
    # Convert to cached format
    cached_paths = cache_path_data(paths, model)
    return stack_intervals_for_hazard(hazard_idx, cached_paths, hazards, totalhazards, tmat; pars=pars)
end

"""
    cache_path_data(paths::Vector{SamplePath}, model::MultistateProcess) -> Vector{CachedPathData}

Pre-allocate and cache DataFrame representations for all paths.

This function converts each SamplePath to its DataFrame representation once,
eliminating redundant `make_subjdat` calls in batched likelihood computation.

# Arguments
- `paths::Vector{SamplePath}`: Sample paths to cache
- `model::MultistateProcess`: Model containing subject data

# Returns
- `Vector{CachedPathData}`: Cached path data with pre-computed DataFrames

# Example
```julia
cached = cache_path_data(data.paths, data.model)
# Now use cached paths in batched operations
for h in 1:n_hazards
    stacked = stack_intervals_for_hazard(h, cached, hazards, totalhazards, tmat)
end
```
"""
function cache_path_data(paths::Vector{SamplePath}, model::MultistateProcess)
    n_paths = length(paths)
    cached = Vector{CachedPathData}(undef, n_paths)
    
    for (i, path) in enumerate(paths)
        subj_inds = model.subjectindices[path.subj]
        subj_dat = view(model.data, subj_inds, :)
        df = make_subjdat(path, subj_dat)
        cached[i] = CachedPathData(path.subj, df, Dict{Int,Vector{Float64}}())
    end
    
    return cached
end

########################################################
##################### Wrappers #########################
########################################################
