########################################################
### Likelihod functions for sample paths, tpms, etc. ###
########################################################

# =============================================================================
# Parameter Preparation (dispatch-based normalization)
# =============================================================================

"""
    prepare_parameters(parameters, model::MultistateProcess)

Normalize parameter representations for downstream hazard calls.
Uses multiple dispatch to handle different parameter container types.

# Supported types
- `Tuple`: Nested parameters indexed by hazard number (returned as-is)
- `NamedTuple`: Parameters keyed by hazard name (converted to values tuple)
- `VectorOfVectors`: Already nested AD-compatible format (returned as-is)
- `AbstractVector`: Flat parameter vector (nested via `nest_params`)
"""
prepare_parameters(p::Tuple, ::MultistateProcess) = p
prepare_parameters(p::NamedTuple, ::MultistateProcess) = values(p)
prepare_parameters(p::ArraysOfArrays.VectorOfVectors, ::MultistateProcess) = p
prepare_parameters(p::AbstractVector{<:AbstractVector}, ::MultistateProcess) = p
prepare_parameters(p::AbstractVector, model::MultistateProcess) = nest_params(p, model.parameters)

# =============================================================================
# Exactly observed sample paths
# =============================================================================

@inline function _time_transform_enabled(totalhazard::_TotalHazard, hazards::Vector{<:_Hazard})
    if totalhazard isa _TotalHazardAbsorbing
        return false
    end

    for idx in totalhazard.components
        if hazards[idx].metadata.time_transform
            return true
        end
    end

    return false
end

"""
    loglik_path(pars, subjectdata::DataFrame, hazards::Vector{<:_Hazard}, totalhazards::Vector{<:_TotalHazard}, tmat::Array{Int,2})

Log-likelihood for a single sample path. The subject data is provided as a DataFrame with columns including:
- `sojourn`: Time spent in current state at start of interval
- `increment`: Time increment for this interval
- `statefrom`: State at start of interval
- `stateto`: State at end of interval
- Additional covariate columns

This function is called after converting a SamplePath object to DataFrame format using `make_subjdat()`.
"""
loglik_path = function(pars, subjectdata::DataFrame, hazards::Vector{<:_Hazard}, totalhazards::Vector{<:_TotalHazard}, tmat::Array{Int,2})

     # initialize log likelihood
     ll = 0.0

    tt_context = maybe_time_transform_context(pars, subjectdata, hazards)
 
     # recurse through the sample path
     for i in Base.OneTo(nrow(subjectdata))

        # accumulate survival probabilty
        origin_state = subjectdata.statefrom[i]
        use_transform = _time_transform_enabled(totalhazards[origin_state], hazards)
        
        # Use @view to avoid DataFrameRow allocation
        row_data = @view subjectdata[i, :]

        ll += survprob(
            subjectdata.sojourn[i],
            subjectdata.sojourn[i] + subjectdata.increment[i],
            pars,
            row_data,
            totalhazards[origin_state],
            hazards;
            give_log = true,
            apply_transform = use_transform,
            cache_context = tt_context)
 
        # accumulate hazard if there is a transition
        if subjectdata.statefrom[i] != subjectdata.stateto[i]
            
            # index for transition
            transind = tmat[subjectdata.statefrom[i], subjectdata.stateto[i]]

            # log hazard at time of transition
            # Pass the DataFrame row for new hazard types (with name-based covariate matching)
            ll += call_haz(
                subjectdata.sojourn[i] + subjectdata.increment[i],
                pars[transind],
                row_data,
                hazards[transind];
                give_log = true,
                apply_transform = hazards[transind].metadata.time_transform,
                cache_context = tt_context,
                hazard_slot = transind)
        end
     end
 
     # unweighted loglikelihood
     return ll
end

"""
    loglik(parameters, path::SamplePath, hazards::Vector{<:_Hazard}, model::MultistateProcess)

Convenience wrapper that evaluates the log-likelihood of a single sample path using the
current model definition. Accepts various parameter container types (see `prepare_parameters`)
and reuses `loglik_path` for the heavy lifting.

# Arguments
- `parameters`: Tuple, NamedTuple, VectorOfVectors, or flat AbstractVector
- `path::SamplePath`: The sample path to evaluate
- `hazards::Vector{<:_Hazard}`: Hazard functions
- `model::MultistateProcess`: Model containing unflatten function and structure
"""
function loglik(parameters, path::SamplePath, hazards::Vector{<:_Hazard}, model::MultistateProcess)
    pars = prepare_parameters(parameters, model)

    # build a subject-level dataframe aligned with the sample path
    subj_inds = model.subjectindices[path.subj]
    subj_dat = view(model.data, subj_inds, :)
    subjdat_df = make_subjdat(path, subj_dat)

    return loglik_path(pars, subjdat_df, hazards, model.totalhazards, model.tmat)
end

########################################################
### Batched (Hazard-Centric) Likelihood for Exact Data #
########################################################

# =============================================================================
# Separability trait for hazard functions
# =============================================================================

"""
    is_separable(hazard::_Hazard) -> Bool

Determine whether a hazard has a separable form that admits an analytic cumulative hazard.

# Tang's Separability Condition

A hazard is separable if it can be written in the form:

```math
\\Lambda'(t | x) = \\alpha(t) \\cdot \\exp(x^\\top \\beta) \\cdot q(\\Lambda)
```

where:
- `α(t)` is the baseline hazard (depends only on time)
- `exp(x'β)` is the covariate effect (PH) or time-rescaling factor (AFT)
- `q(Λ)` depends only on the cumulative hazard (constant = 1 for PH models)

When `is_separable` returns `true`, the cumulative hazard can be computed analytically
using the closed-form `cumulative_hazard` function, avoiding expensive ODE solves.

# Current Implementation

All existing hazard types return `true`:
- `MarkovHazard`: Exponential family with constant baseline → separable
- `SemiMarkovHazard`: Weibull/Gompertz with known cumulative hazards → separable
- `RuntimeSplineHazard`: Spline baseline with known integral → separable

# Future Extensions

Future hazard types may return `false`:
- `:ode` hazards with non-separable RHS
- `:ode_neural` hazards where `f(t, Λ, x)` is a neural network

When `is_separable` returns `false`, the likelihood must use ODE solvers
(via DifferentialEquations.jl) to compute cumulative hazards numerically.

# Usage Sites

This trait is checked in several locations:
- `loglik_exact`: Use analytic cumhaz when separable
- `survprob`: Skip ODE solve when all exit hazards are separable
- `TimeTransformContext`: Separability is prerequisite for Tang-style caching
- `simulate`: Cumulative incidence/survival computations

# References

- Tang, W., He, K., Xu, G., & Zhu, J. (2022). Survival Analysis via Ordinary 
  Differential Equations. JASA. arXiv:2009.03449

See also: [`cumulative_hazard`](@ref), [`TimeTransformContext`](@ref)
"""
is_separable(::_Hazard) = true  # Default: all current hazards are separable

# Explicit dispatches for documentation clarity
is_separable(::MarkovHazard) = true
is_separable(::SemiMarkovHazard) = true
is_separable(::_SplineHazard) = true

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
# 4. ODE DATA CONVERSION (BatchedODEData)
#    - Converts NamedTuple covariates to Matrix format
#    - Matrix layout: (n_features × n_intervals) for Lux compatibility
#    - Use: `ode_data = to_batched_ode_data(stacked)`
#
# Data Flow:
#
#   paths → cache_path_data → CachedPathData[]
#                              ↓
#                    stack_intervals_for_hazard (per hazard)
#                              ↓
#                    StackedHazardData (per hazard)
#                              ↓
#                    [Optional] to_batched_ode_data (for neural hazards)
#                              ↓
#                    BatchedODEData (for ODE solves)
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
# Batched ODE-compatible data structures
# =============================================================================

"""
    BatchedODEData

Pre-processed interval data organized for batched ODE solves via `EnsembleProblem`.

This struct organizes integration limits and covariates in a layout compatible with
SciML's `EnsembleProblem` for future neural ODE hazards. The covariate matrix uses
Lux's preferred batch-last layout (features × n_intervals).

# Fields
- `tspans::Matrix{Float64}`: Integration limits, 2 × n_intervals (row 1 = lb, row 2 = ub)
- `covars::Matrix{Float64}`: Covariate values, n_features × n_intervals (Lux batch-last)
- `path_idx::Vector{Int}`: Maps each interval back to its source path for accumulation
- `is_transition::Vector{Bool}`: Whether interval ends with transition via this hazard
- `transition_times::Vector{Float64}`: Time of transition (when is_transition is true)

# Compatibility with SciML

For future ODE-based hazards, this struct enables:

```julia
# Construct EnsembleProblem with varying tspans
function prob_func(prob, i, repeat)
    remake(prob, tspan = (data.tspans[1,i], data.tspans[2,i]))
end
ensemble_prob = EnsembleProblem(ode_prob, prob_func = prob_func)
sol = solve(ensemble_prob, Tsit5(), EnsembleThreads(), trajectories = n_intervals)
```

# Covariate Layout

Covariates are stored with batch dimension last to match Lux conventions:
- Shape: (n_features, n_intervals) 
- Access: `covars[:, i]` gives feature vector for interval i

See also: [`StackedHazardData`](@ref), [`is_separable`](@ref)
"""
struct BatchedODEData
    tspans::Matrix{Float64}      # 2 × n_intervals (lb, ub)
    covars::Matrix{Float64}      # n_features × n_intervals (Lux batch-last)
    path_idx::Vector{Int}        # Maps intervals to paths
    is_transition::Vector{Bool}  # Transition indicator
    transition_times::Vector{Float64}  # Time of transition
end

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
    to_batched_ode_data(sd::StackedHazardData) -> BatchedODEData

Convert `StackedHazardData` to `BatchedODEData` format suitable for neural network hazards.

This function transforms the covariate representation from a vector of NamedTuples
(efficient for parametric hazards) to a matrix format (efficient for neural network
batched evaluation with Lux).

# Arguments
- `sd::StackedHazardData`: Pre-processed interval data with NamedTuple covariates

# Returns
- `BatchedODEData`: Same data with covariates in matrix format (n_features × n_intervals)

# Covariate Layout
The returned `covars` matrix has shape `(n_features, n_intervals)` with batch dimension
last to match Lux conventions. Feature order matches the order in the NamedTuples.

# Example
```julia
# After stacking intervals for a hazard
stacked = stack_intervals_for_hazard(h, cached_paths, hazards, totalhazards, tmat)

# Convert to ODE-ready format
ode_data = to_batched_ode_data(stacked)

# Use with neural network hazard
# predictions = nn_model(ode_data.covars, ps, st)
```

See also: [`StackedHazardData`](@ref), [`BatchedODEData`](@ref), [`is_separable`](@ref)
"""
function to_batched_ode_data(sd::StackedHazardData; use_views::Bool=false)
    n_intervals = length(sd.lb)
    
    if n_intervals == 0
        return BatchedODEData(
            zeros(2, 0),
            zeros(0, 0),
            Int[],
            Bool[],
            Float64[]
        )
    end
    
    # Determine number of features from first covariate tuple
    first_covar = sd.covars[1]
    n_features = length(first_covar)
    
    # Build time spans matrix (2 × n_intervals)
    # Note: tspans must be a Matrix, so we can't use views here
    tspans = Matrix{Float64}(undef, 2, n_intervals)
    for i in 1:n_intervals
        tspans[1, i] = sd.lb[i]
        tspans[2, i] = sd.ub[i]
    end
    
    # Build covariate matrix (n_features × n_intervals)
    if n_features == 0
        covars = zeros(0, n_intervals)
    else
        covars = Matrix{Float64}(undef, n_features, n_intervals)
        for i in 1:n_intervals
            covar_tuple = sd.covars[i]
            for (j, val) in enumerate(covar_tuple)
                covars[j, i] = Float64(val)
            end
        end
    end
    
    # For vectors, optionally use views to avoid allocation
    # SAFETY: Views are only safe if the source StackedHazardData remains in scope
    # during all operations using the BatchedODEData
    if use_views
        return BatchedODEData(
            tspans,
            covars,
            sd.path_idx,         # View (actually reference to same array)
            sd.is_transition,    # View
            sd.transition_times  # View
        )
    else
        return BatchedODEData(
            tspans,
            covars,
            copy(sd.path_idx),
            copy(sd.is_transition),
            copy(sd.transition_times)
        )
    end
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

# =============================================================================
# Generic dispatch methods for loglik and loglik!
# =============================================================================
#
# These dispatch methods provide a unified interface for likelihood computation
# across different data types. The MCEM algorithm calls loglik(p, data) and
# loglik!(p, logliks, data) generically, and these methods dispatch to the
# appropriate specialized implementations.
# =============================================================================

"""
    loglik(parameters, data::SMPanelData; neg=true, use_sampling_weight=true)

Dispatch method for semi-Markov panel data. Calls `loglik_semi_markov`.

See [`loglik_semi_markov`](@ref) for details.
"""
loglik(parameters, data::SMPanelData; neg=true, use_sampling_weight=true) = 
    loglik_semi_markov(parameters, data; neg=neg, use_sampling_weight=use_sampling_weight)

"""
    loglik!(parameters, logliks::Vector{}, data::SMPanelData)

In-place dispatch method for semi-Markov panel data. Calls `loglik_semi_markov!`.

See [`loglik_semi_markov!`](@ref) for details.
"""
loglik!(parameters, logliks::Vector{}, data::SMPanelData) = 
    loglik_semi_markov!(parameters, logliks, data)

"""
    loglik(parameters, data::MPanelData; neg=true, return_ll_subj=false)

Dispatch method for Markov panel data. Calls `loglik_markov`.

See [`loglik_markov`](@ref) for details.
"""
loglik(parameters, data::MPanelData; neg=true, return_ll_subj=false) = 
    loglik_markov(parameters, data; neg=neg, return_ll_subj=return_ll_subj)

"""
    loglik_AD(parameters, data::ExactDataAD; neg = true) 

Compute (negative) log-likelihood for a single sample path with AD support.

This function is used for per-path Fisher information computation in variance 
estimation, where we need to differentiate the likelihood of individual paths.
Unlike `loglik_exact` which operates on `ExactData` (multiple paths), this
operates on `ExactDataAD` containing a single path.

# Arguments
- `parameters`: Flat parameter vector
- `data::ExactDataAD`: Single path data container
- `neg::Bool=true`: Return negative log-likelihood

# Returns
Scalar (negative) log-likelihood for the single path, weighted by sampling weight.

See also: [`loglik_exact`](@ref), [`ExactDataAD`](@ref)
"""
function loglik_AD(parameters, data::ExactDataAD; neg = true)

    # nest parameters using VectorOfVectors (AD-compatible)
    pars = nest_params(parameters, data.model.parameters)

    # snag the hazards
    hazards = data.model.hazards

    # remake spline parameters and calculate risk periods
    # Note: parameters are already on log scale (from optimizer flat vector)
    for i in eachindex(hazards)
        if isa(hazards[i], _SplineHazard)
            # pars[i] is already log-scale, pass directly
            log_pars = Vector{Float64}(collect(pars[i]))
            remake_splines!(hazards[i], log_pars)
            set_riskperiod!(hazards[i])
        end
    end

    # send each element of samplepaths to loglik
    # Convert SamplePath to DataFrame using make_subjdat
    path = data.path[1]
    subj_inds = data.model.subjectindices[path.subj]
    subj_dat = view(data.model.data, subj_inds, :)
    subjdat_df = make_subjdat(path, subj_dat)
    ll = loglik_path(pars, subjdat_df, hazards, data.model.totalhazards, data.model.tmat) * data.samplingweight[1]

    neg ? -ll : ll
end

"""
    loglik(parameters, data::MPanelData; neg = true)

Return sum of (negative) log likelihood for a Markov model fit to panel and/or exact and/or censored data. 
"""
function loglik_markov(parameters, data::MPanelData; neg = true, return_ll_subj = false)

    # nest the model parameters using VectorOfVectors (AD-compatible)
    pars = nest_params(parameters, data.model.parameters)

    # build containers for transition intensity and prob mtcs
    hazmat_book = build_hazmat_book(eltype(parameters), data.model.tmat, data.books[1])
    tpm_book = build_tpm_book(eltype(parameters), data.model.tmat, data.books[1])

    # allocate memory for matrix exponential
    cache = ExponentialUtilities.alloc_mem(similar(hazmat_book[1]), ExpMethodGeneric())

    # Solve Kolmogorov equations for TPMs
    for t in eachindex(data.books[1])

        # compute the transition intensity matrix
        compute_hazmat!(
            hazmat_book[t],
            pars,
            data.model.hazards,
            data.books[1][t],
            data.model.data)

        # compute transition probability matrices
        compute_tmat!(
            tpm_book[t],
            hazmat_book[t],
            data.books[1][t],
            cache)
    end

    # number of subjects
    nsubj = length(data.model.subjectindices)

    # Element type for AD compatibility
    T = eltype(parameters)

    # accumulate the log likelihood
    ll = zero(T)

    # container for subject-level loglikelihood
    if return_ll_subj
        ll_subj = zeros(T, nsubj)
    end

    # number of states
    S = size(data.model.tmat, 1)

    # initialize Q 
    q = zeros(eltype(parameters), S, S)

    # for each subject, compute the likelihood contribution
    for subj in Base.OneTo(nsubj)

        # subject data
        subj_inds = data.model.subjectindices[subj]

        # check if observation weights are provided
        has_obs_weights = !isnothing(data.model.ObservationWeights)

        # Cache hazard covar_names to avoid repeated lookups
        hazards = data.model.hazards

        # no state is censored
        if all(data.model.data.obstype[subj_inds] .∈ Ref([1,2]))
            
            # subject contribution to the loglikelihood
            subj_ll = zero(T)

            # add the contribution of each observation
            @inbounds for i in subj_inds
                # get observation weight (default to 1.0)
                obs_weight = has_obs_weights ? data.model.ObservationWeights[i] : 1.0
                
                obstype_i = data.model.data.obstype[i]
                
                if obstype_i == 1 # exact data
                    # Use @view to avoid DataFrameRow allocation
                    row_data = @view data.model.data[i, :]
                    
                    statefrom_i = data.model.data.statefrom[i]
                    stateto_i = data.model.data.stateto[i]
                    dt = data.model.data.tstop[i] - data.model.data.tstart[i]
                    
                    obs_ll = survprob(
                        0,
                        dt,
                        pars,
                        row_data,
                        data.model.totalhazards[statefrom_i],
                        hazards;
                        give_log = true)
                                        
                    if statefrom_i != stateto_i # if there is a transition, add log hazard
                        trans_idx = data.model.tmat[statefrom_i, stateto_i]
                        obs_ll += call_haz(
                            dt,
                            pars[trans_idx],
                            row_data,
                            hazards[trans_idx];
                            give_log = true)
                    end
                    
                    subj_ll += obs_ll * obs_weight

                else # panel data (obstype == 2)
                    statefrom_i = data.model.data.statefrom[i]
                    stateto_i = data.model.data.stateto[i]
                    book_idx1 = data.books[2][i, 1]
                    book_idx2 = data.books[2][i, 2]
                    subj_ll += log(tpm_book[book_idx1][book_idx2][statefrom_i, stateto_i]) * obs_weight
                end
            end

        else
            # Forward algorithm for censored observations
            # NOTE: ObservationWeights are currently not supported for censored observations
            # (forward algorithm). Use SubjectWeights instead for weighted estimation.
            if has_obs_weights && any(data.model.ObservationWeights[subj_inds] .!= 1.0)
                @warn "ObservationWeights are not supported for censored observations (obstype > 2). Using unweighted likelihood for subject $subj."
            end
            
            # Pre-compute transient state transitions (avoid findall allocations in loop)
            tmat_cache = data.model.tmat
            transient_dests = [findall(tmat_cache[r,:] .!= 0) for r in 1:S]
            
            # initialize likelihood matrix
            lmat = zeros(eltype(parameters), S, length(subj_inds) + 1)
            @inbounds lmat[data.model.data.statefrom[subj_inds[1]], 1] = 1

            # initialize counter for likelihood matrix
            ind = 1

            # update the vector l
            @inbounds for i in subj_inds

                # increment counter for likelihood matrix
                ind += 1
                
                obstype_i = data.model.data.obstype[i]
                dt = data.model.data.tstop[i] - data.model.data.tstart[i]

                # compute q, the transition probability matrix
                if obstype_i != 1
                    # if panel data, simply grab q from tpm_book
                    book_idx1 = data.books[2][i, 1]
                    book_idx2 = data.books[2][i, 2]
                    copyto!(q, tpm_book[book_idx1][book_idx2])
                    
                else
                    # if exact data (obstype = 1), compute q by hand
                    # reset Q
                    fill!(q, -Inf)
                    
                    # Use @view for data row
                    row_data = @view data.model.data[i, :]
                    
                    # compute q(r,s)
                    for r in 1:S
                        if isa(data.model.totalhazards[r], _TotalHazardAbsorbing)
                            q[r,r] = 0.0
                        else
                            # survival probability - use pre-computed destinations
                            dest_states = transient_dests[r]
                            log_surv = survprob(0, dt, pars, row_data,
                                data.model.totalhazards[r], hazards; give_log = true)
                            for dest in dest_states
                                q[r, dest] = log_surv
                            end
                            
                            # hazard
                            for s in dest_states
                                trans_idx = tmat_cache[r, s]
                                q[r, s] += call_haz(dt, pars[trans_idx], row_data,
                                    hazards[trans_idx]; give_log = true)
                            end
                        end

                        # pedantic b/c of numerical error
                        q[r,r] = maximum([1 - exp(logsumexp(q[r, Not(r)])), eps()])
                        q[r,Not(r)] = exp.(q[r, Not(r)])               
                    end
                end # end-compute q

                # compute the set of possible "states to" and their emission probabilities
                # For exact observations (stateto > 0), only that state is possible with probability 1
                # For censored observations, use the emission matrix
                stateto_i = data.model.data.stateto[i]
                if stateto_i > 0
                    # Exact observation - only one state possible
                    for r in 1:S
                        lmat[stateto_i, ind] += q[r, stateto_i] * lmat[r, ind - 1]
                    end
                else
                    # Censored observation - weight by emission probabilities
                    for s in 1:S
                        emission_prob = data.model.emat[i, s]
                        if emission_prob > 0
                            for r in 1:S
                                lmat[s, ind] += q[r, s] * lmat[r, ind - 1] * emission_prob
                            end
                        end
                    end
                end
            end

            # log likelihood
            subj_ll=log(sum(lmat[:,size(lmat, 2)]))
        end

        if return_ll_subj
            # weighted subject loglikelihood
            ll_subj[subj] = subj_ll * data.model.SubjectWeights[subj]
        else
            # weighted loglikelihood
            ll += subj_ll * data.model.SubjectWeights[subj]
        end        
    end

    if return_ll_subj
        ll_subj
    else
        neg ? -ll : ll
    end
end

"""
    loglik(parameters, data::SMPanelData; neg = true)

Return sum of (negative) complete data log-likelihood terms in the Monte Carlo maximum likelihood algorithm for fitting a semi-Markov model to panel data.

This implementation uses the fused path-centric approach from `loglik_exact`, calling
`_compute_path_loglik_fused` directly to avoid DataFrame allocation overhead.
"""
function loglik_semi_markov(parameters, data::SMPanelData; neg = true, use_sampling_weight = true)

    # nest the model parameters using VectorOfVectors (AD-compatible)
    pars = nest_params(parameters, data.model.parameters)

    # snag the hazards and model components
    hazards = data.model.hazards
    totalhazards = data.model.totalhazards
    tmat = data.model.tmat
    n_hazards = length(hazards)

    # update spline hazards with current parameters (no-op for functional splines)
    _update_spline_hazards!(hazards, pars)

    # Build subject covariate cache (reusable across all paths)
    subject_covars = build_subject_covar_cache(data.model)
    
    # Element type for AD compatibility
    T = eltype(parameters)
    
    # Get covariate names for each hazard (precomputed)
    covar_names_per_hazard = [
        hasfield(typeof(hazards[h]), :covar_names) ? 
            hazards[h].covar_names : 
            extract_covar_names(hazards[h].parnames)
        for h in 1:n_hazards
    ]
    
    # Check if any hazard uses time transform and create context if needed
    any_time_transform = any(h -> h.metadata.time_transform, hazards)
    tt_context = if any_time_transform && !isempty(data.paths) && !isempty(data.paths[1])
        sample_subj = subject_covars[data.paths[1][1].subj]
        sample_df = isempty(sample_subj.covar_data) ? nothing : sample_subj.covar_data[1:1, :]
        maybe_time_transform_context(pars, sample_df, hazards)
    else
        nothing
    end

    # compute the semi-markov log-likelihoods using fused approach
    ll = zero(T)
    for i in eachindex(data.paths)
        lls = zero(T)
        for j in eachindex(data.paths[i])
            path = data.paths[i][j]
            subj_cache = subject_covars[path.subj]
            
            path_ll = _compute_path_loglik_fused(
                path, pars, hazards, totalhazards, tmat,
                subj_cache, covar_names_per_hazard, tt_context, T
            )
            lls += path_ll * data.ImportanceWeights[i][j]
        end
        if use_sampling_weight
            lls *= data.model.SubjectWeights[i]
        end
        ll += lls
    end

    # return the log-likelihood
    neg ? -ll : ll
end

"""
    loglik(parameters, data::SMPanelData)

Update log-likelihood for each individual and each path of panel data in a semi-Markov model.

This implementation uses the fused path-centric approach from `loglik_exact`, calling
`_compute_path_loglik_fused` directly to avoid DataFrame allocation overhead.

# Notes on future neural ODE compatibility:
When `is_separable(hazard) == false` for ODE-based hazards, the `call_cumulhaz` 
function in `_compute_path_loglik_fused` is the extension point where numerical 
ODE solvers would be invoked instead of analytic cumulative hazard formulas.
"""
function loglik_semi_markov!(parameters, logliks::Vector{}, data::SMPanelData)

    # nest the model parameters using VectorOfVectors (AD-compatible)
    pars = nest_params(parameters, data.model.parameters)

    # snag the hazards and model components
    hazards = data.model.hazards
    totalhazards = data.model.totalhazards
    tmat = data.model.tmat
    n_hazards = length(hazards)

    # remake spline parameters and calculate risk periods
    # Note: parameters are already on log scale (from optimizer flat vector)
    for i in eachindex(hazards)
        if isa(hazards[i], _SplineHazard)
            # pars[i] is already log-scale, pass directly
            log_pars = Vector{Float64}(collect(pars[i]))
            remake_splines!(hazards[i], log_pars)
            set_riskperiod!(hazards[i])
        end
    end

    # Build subject covariate cache (reusable across all paths)
    subject_covars = build_subject_covar_cache(data.model)
    
    # Element type for computation (Float64 for in-place version)
    T = Float64
    
    # Get covariate names for each hazard (precomputed)
    covar_names_per_hazard = [
        hasfield(typeof(hazards[h]), :covar_names) ? 
            hazards[h].covar_names : 
            extract_covar_names(hazards[h].parnames)
        for h in 1:n_hazards
    ]
    
    # Check if any hazard uses time transform and create context if needed
    any_time_transform = any(h -> h.metadata.time_transform, hazards)
    tt_context = if any_time_transform && !isempty(data.paths) && !isempty(data.paths[1])
        sample_subj = subject_covars[data.paths[1][1].subj]
        sample_df = isempty(sample_subj.covar_data) ? nothing : sample_subj.covar_data[1:1, :]
        maybe_time_transform_context(pars, sample_df, hazards)
    else
        nothing
    end

    # Compute log-likelihoods using fused path-centric approach
    for i in eachindex(data.paths)
        for j in eachindex(data.paths[i])
            path = data.paths[i][j]
            subj_cache = subject_covars[path.subj]
            
            logliks[i][j] = _compute_path_loglik_fused(
                path, pars, hazards, totalhazards, tmat,
                subj_cache, covar_names_per_hazard, tt_context, T
            )
        end
    end
end

"""
    loglik_semi_markov_batched!(parameters, logliks, data::SMPanelData)

Batched version of `loglik_semi_markov!` that computes all path log-likelihoods
using the batched hazard evaluation approach. This reduces redundant computations
when there are many paths per subject.

Arguments:
- `parameters`: Flat parameter vector
- `logliks`: Nested Vector{Vector{Float64}} to store log-likelihoods (modified in-place)
- `data`: SMPanelData containing the model and paths
"""
function loglik_semi_markov_batched!(parameters, logliks::Vector{Vector{Float64}}, data::SMPanelData)
    # Nest parameters using VectorOfVectors (AD-compatible)
    pars = nest_params(parameters, data.model.parameters)
    
    # Get hazards
    hazards = data.model.hazards
    n_hazards = length(hazards)
    
    # Remake spline parameters and calculate risk periods
    # Note: parameters are already on log scale (from optimizer flat vector)
    for i in eachindex(hazards)
        if isa(hazards[i], _SplineHazard)
            # pars[i] is already log-scale, pass directly
            log_pars = Vector{Float64}(collect(pars[i]))
            remake_splines!(hazards[i], log_pars)
            set_riskperiod!(hazards[i])
        end
    end
    
    # Flatten paths for batched processing
    # Build mapping from flat index to (subject_idx, path_idx)
    n_total_paths = sum(length(ps) for ps in data.paths)
    flat_paths = Vector{SamplePath}(undef, n_total_paths)
    path_mapping = Vector{Tuple{Int,Int}}(undef, n_total_paths)
    
    flat_idx = 0
    for i in eachindex(data.paths)
        for j in eachindex(data.paths[i])
            flat_idx += 1
            flat_paths[flat_idx] = data.paths[i][j]
            path_mapping[flat_idx] = (i, j)
        end
    end
    
    # Pre-cache all path DataFrames
    cached_paths = cache_path_data(flat_paths, data.model)
    
    # Create TimeTransformContext if needed
    sample_df = isempty(cached_paths) ? nothing : first(cached_paths).df
    tt_context = maybe_time_transform_context(pars, sample_df, hazards)
    
    # Initialize flat log-likelihoods
    ll_flat = zeros(n_total_paths)
    
    # Pre-process intervals for each hazard
    stacked_data = Vector{StackedHazardData}(undef, n_hazards)
    for h in 1:n_hazards
        stacked_data[h] = stack_intervals_for_hazard(
            h, cached_paths, hazards, data.model.totalhazards, data.model.tmat;
            pars=pars)
    end
    
    # Compute log-likelihoods in batched manner
    for h in 1:n_hazards
        sd = stacked_data[h]
        n_intervals = length(sd.lb)
        
        if n_intervals == 0
            continue
        end
        
        hazard = hazards[h]
        hazard_pars = pars[h]
        use_transform = hazard.metadata.time_transform
        
        for i in 1:n_intervals
            # Cumulative hazard contribution (survival component)
            # Note: use give_log=false to get cumhaz directly, not log(cumhaz)
            cumhaz = call_cumulhaz(
                sd.lb[i], sd.ub[i], hazard_pars, sd.covars[i], hazard;
                give_log = false,
                apply_transform = use_transform,
                cache_context = tt_context,
                hazard_slot = h)
            
            ll_flat[sd.path_idx[i]] -= cumhaz
            
            # Transition hazard
            if sd.is_transition[i]
                log_haz = call_haz(
                    sd.transition_times[i], hazard_pars, sd.covars[i], hazard;
                    give_log = true,
                    apply_transform = use_transform,
                    cache_context = tt_context,
                    hazard_slot = h)
                ll_flat[sd.path_idx[i]] += log_haz
            end
        end
    end
    
    # Map back to nested structure
    for k in 1:n_total_paths
        i, j = path_mapping[k]
        logliks[i][j] = ll_flat[k]
    end
end

# =============================================================================
# Fused Batched Likelihood Computation
# =============================================================================
#
# This section provides an optimized batched likelihood implementation that:
# 1. Avoids DataFrame allocation by computing intervals directly from SamplePath
# 2. Caches covariate lookups per subject (not per path)
# 3. Uses columnar storage for better cache locality
# 4. Pre-allocates all working memory for reuse across iterations
#
# Note: LightweightInterval and SubjectCovarCache structs are defined in common.jl
#
# =============================================================================

"""
    build_subject_covar_cache(model::MultistateProcess)

Build a cache of covariate data per subject.
This is called once per model and reused across all likelihood evaluations.
"""
function build_subject_covar_cache(model::MultistateProcess)
    n_subjects = length(model.subjectindices)
    covar_cols = setdiff(names(model.data), [:id, :tstart, :tstop, :statefrom, :stateto, :obstype])
    has_covars = !isempty(covar_cols)
    
    caches = Vector{SubjectCovarCache}(undef, n_subjects)
    
    for subj in 1:n_subjects
        subj_inds = model.subjectindices[subj]
        subj_data = view(model.data, subj_inds, :)
        
        if has_covars
            covar_data = subj_data[:, covar_cols]
            tstart = collect(subj_data.tstart)
        else
            covar_data = DataFrame()
            tstart = Float64[]
        end
        
        caches[subj] = SubjectCovarCache(tstart, covar_data)
    end
    
    return caches
end

"""
    compute_intervals_from_path(path::SamplePath, subject_covar::SubjectCovarCache)

Compute likelihood intervals directly from a SamplePath without creating a DataFrame.
Returns a vector of LightweightIntervals.

This implements the same logic as `make_subjdat` but avoids DataFrame allocation.
"""
function compute_intervals_from_path(path::SamplePath, subject_covar::SubjectCovarCache)
    # Determine evaluation times
    # For paths without covariates, use path times directly
    # For paths with covariates, merge path times with covariate change times
    
    n_transitions = length(path.times) - 1
    
    if isempty(subject_covar.covar_data) || nrow(subject_covar.covar_data) <= 1
        # No time-varying covariates - use path times directly
        intervals = Vector{LightweightInterval}(undef, n_transitions)
        
        sojourn = 0.0
        for i in 1:n_transitions
            increment = path.times[i+1] - path.times[i]
            intervals[i] = LightweightInterval(
                sojourn,
                sojourn + increment,
                path.states[i],
                path.states[i+1],
                1  # Single covariate row
            )
            
            # Reset sojourn if state changes (semi-Markov clock reset)
            if path.states[i] != path.states[i+1]
                sojourn = 0.0
            else
                sojourn += increment
            end
        end
        
        return intervals
    else
        # Time-varying covariates - need to split intervals at covariate change times
        # Find times where covariates change
        tstart = subject_covar.tstart
        covar_data = subject_covar.covar_data
        
        # Identify covariate change times (where row differs from previous)
        change_times = [tstart[1]]  # Always include first time
        for i in 2:length(tstart)
            if !isequal(covar_data[i-1, :], covar_data[i, :])
                push!(change_times, tstart[i])
            end
        end
        
        # Merge path times with covariate change times
        utimes = sort(unique(vcat(path.times, change_times)))
        
        # Filter to only times within path range
        filter!(t -> path.times[1] <= t <= path.times[end], utimes)
        
        n_intervals = length(utimes) - 1
        intervals = Vector{LightweightInterval}(undef, n_intervals)
        
        # Track sojourn per state visit
        pathinds = searchsortedlast.(Ref(path.times), utimes)
        datinds = searchsortedlast.(Ref(tstart), utimes)
        
        # Compute sojourns by grouping consecutive intervals in same state visit
        sojourns = zeros(n_intervals)
        current_sojourn = 0.0
        current_pathind = pathinds[1]
        
        for i in 1:n_intervals
            increment = utimes[i+1] - utimes[i]
            
            # Reset sojourn if we're in a new state visit
            if pathinds[i] != current_pathind
                current_sojourn = 0.0
                current_pathind = pathinds[i]
            end
            
            sojourns[i] = current_sojourn
            current_sojourn += increment
        end
        
        # Build intervals
        for i in 1:n_intervals
            increment = utimes[i+1] - utimes[i]
            intervals[i] = LightweightInterval(
                sojourns[i],
                sojourns[i] + increment,
                path.states[pathinds[i]],
                path.states[pathinds[i+1]],
                datinds[i]
            )
        end
        
        return intervals
    end
end

"""
    extract_covariates_lightweight(subject_covar::SubjectCovarCache, row_idx::Int, covar_names::Vector{Symbol})

Extract covariates from the subject cache without DataFrame row access overhead.
"""
@inline function extract_covariates_lightweight(subject_covar::SubjectCovarCache, row_idx::Int, covar_names::Vector{Symbol})
    isempty(covar_names) && return NamedTuple()
    
    # Clamp row_idx to valid range
    idx = clamp(row_idx, 1, max(1, nrow(subject_covar.covar_data)))
    
    if isempty(subject_covar.covar_data)
        return NamedTuple()
    end
    
    # Extract values for requested covariates
    values = Tuple(subject_covar.covar_data[idx, cname] for cname in covar_names)
    return NamedTuple{Tuple(covar_names)}(values)
end

"""
    loglik_exact(parameters, data::ExactData; neg=true, return_ll_subj=false)

Compute (negative) log-likelihood for exactly observed sample paths.

This is a fused batched implementation that processes intervals and computes hazards 
in a single pass, optimized for memory locality by processing all hazards per interval
rather than all intervals per hazard.

# Features
- ForwardDiff (forward-mode AD) via parametric element types
- Zygote/Enzyme (reverse-mode AD) via functional accumulation (no in-place mutation)
- Time transforms (Tang-style caching) when hazards have `time_transform=true`
- Time-varying covariates with proper sojourn tracking

# Arguments
- `parameters`: Flat parameter vector
- `data::ExactData`: Exact data containing model and sample paths
- `neg::Bool=true`: Return negative log-likelihood (for minimization)
- `return_ll_subj::Bool=false`: Return per-path log-likelihoods instead of sum

# Returns
- If `return_ll_subj=false`: Scalar (negative) log-likelihood
- If `return_ll_subj=true`: Vector of weighted per-path log-likelihoods
"""
function loglik_exact(parameters, data::ExactData; neg=true, return_ll_subj=false)
    # Nest parameters using VectorOfVectors - preserves dual number types (AD-compatible)
    pars = nest_params(parameters, data.model.parameters)
    
    # Get model components
    hazards = data.model.hazards
    totalhazards = data.model.totalhazards
    tmat = data.model.tmat
    n_hazards = length(hazards)
    n_paths = length(data.paths)
    
    # Remake spline parameters if needed
    # Note: parameters are already on log scale (from optimizer flat vector)
    for i in eachindex(hazards)
        if isa(hazards[i], _SplineHazard)
            # pars[i] is already log-scale, pass directly
            log_pars = Vector{Float64}(collect(pars[i]))
            remake_splines!(hazards[i], log_pars)
            set_riskperiod!(hazards[i])
        end
    end
    
    # Build subject covariate cache (this is type-stable, no parameters involved)
    subject_covars = build_subject_covar_cache(data.model)
    
    # Element type for AD compatibility (Float64 or Dual)
    T = eltype(parameters)
    
    # Get covariate names for each hazard (precomputed, doesn't depend on parameters)
    covar_names_per_hazard = [
        hasfield(typeof(hazards[h]), :covar_names) ? 
            hazards[h].covar_names : 
            extract_covar_names(hazards[h].parnames)
        for h in 1:n_hazards
    ]
    
    # Check if any hazard uses time transform
    any_time_transform = any(h -> h.metadata.time_transform, hazards)
    
    # Create TimeTransformContext if needed
    # We need a sample DataFrame for type inference - build minimal one
    tt_context = if any_time_transform && !isempty(data.paths)
        sample_subj = subject_covars[data.paths[1].subj]
        sample_df = isempty(sample_subj.covar_data) ? nothing : sample_subj.covar_data[1:1, :]
        maybe_time_transform_context(pars, sample_df, hazards)
    else
        nothing
    end
    
    # Subject weights (precomputed lookup for efficiency)
    subj_weights = data.model.SubjectWeights
    
    # Accumulate log-likelihoods functionally for reverse-mode AD compatibility
    # Using map instead of in-place mutation
    ll_paths = map(enumerate(data.paths)) do (path_idx, path)
        _compute_path_loglik_fused(
            path, pars, hazards, totalhazards, tmat, 
            subject_covars[path.subj], covar_names_per_hazard,
            tt_context, T
        )
    end
    
    # Convert to proper array type
    ll_array = collect(T, ll_paths)
    
    if return_ll_subj
        # Element-wise multiplication preserves AD types
        return ll_array .* [subj_weights[path.subj] for path in data.paths]
    else
        # Weighted sum
        ll = sum(ll_array[i] * subj_weights[data.paths[i].subj] for i in eachindex(data.paths))
        return neg ? -ll : ll
    end
end

"""
    _compute_path_loglik_fused(path, pars, hazards, totalhazards, tmat, 
                                subj_cache, covar_names_per_hazard, tt_context, T)

Compute log-likelihood for a single path. Extracted for functional style (reverse-mode AD).

This function is the core likelihood computation shared by `loglik_exact` and `loglik_semi_markov`.
It uses a path-centric approach that iterates over sojourn intervals in a sample path, 
accumulating log-survival contributions (via `call_cumulhaz`) and transition hazard 
contributions (via `call_haz`).

# Neural ODE Extension Point
The `call_cumulhaz` invocations are the extension points for neural ODE-based hazards.
When `is_separable(hazard) == false` for an ODE-based hazard, `call_cumulhaz` should be 
extended to invoke a numerical ODE solver (e.g., DifferentialEquations.jl) to compute 
the cumulative hazard as an integral: Λ(t₀, t₁) = ∫_{t₀}^{t₁} λ(s) ds.

For reverse-mode AD compatibility with neural ODEs:
- Use SciMLSensitivity.jl adjoints (BacksolveAdjoint, QuadratureAdjoint)
- Ensure the hazard's metadata declares supported adjoint methods
- The functional accumulation style in this function avoids in-place mutation 
  required for Zygote/Enzyme compatibility

See also: `loglik_exact`, `loglik_semi_markov`, `call_cumulhaz`
"""
function _compute_path_loglik_fused(
    path::SamplePath, 
    pars,  # Parameters as nested structure (from nest_params, VectorOfVectors or Tuple)
    hazards::Vector{<:_Hazard},
    totalhazards::Vector{<:_TotalHazard}, 
    tmat::Matrix{Int64},
    subj_cache::SubjectCovarCache, 
    covar_names_per_hazard::Vector{Vector{Symbol}},
    tt_context,
    ::Type{T}
) where T
    
    n_hazards = length(hazards)
    n_transitions = length(path.times) - 1
    
    # Initialize log-likelihood for this path
    ll = zero(T)
    
    # Check if we have time-varying covariates
    has_tvc = !isempty(subj_cache.covar_data) && nrow(subj_cache.covar_data) > 1
    
    if !has_tvc
        # Fast path: no time-varying covariates
        sojourn = 0.0
        
        for i in 1:n_transitions
            increment = path.times[i+1] - path.times[i]
            lb = sojourn
            ub = sojourn + increment
            statefrom = path.states[i]
            stateto = path.states[i+1]
            
            # Get total hazard for origin state
            tothaz = totalhazards[statefrom]
            
            if tothaz isa _TotalHazardTransient
                # Determine if time transform is enabled for this state
                use_transform = _time_transform_enabled(tothaz, hazards)
                
                # Accumulate cumulative hazards for all exit hazards
                for h in tothaz.components
                    hazard = hazards[h]
                    hazard_pars = pars[h]
                    
                    # Extract covariates
                    covars = extract_covariates_lightweight(subj_cache, 1, covar_names_per_hazard[h])
                    
                    # Cumulative hazard (use hazard-specific transform flag)
                    cumhaz = call_cumulhaz(
                        lb, ub, hazard_pars, covars, hazard;
                        give_log = false,
                        apply_transform = hazard.metadata.time_transform,
                        cache_context = tt_context,
                        hazard_slot = h)
                    
                    ll -= cumhaz
                end
                
                # Add transition hazard if transition occurred
                if statefrom != stateto
                    trans_h = tmat[statefrom, stateto]
                    hazard = hazards[trans_h]
                    hazard_pars = pars[trans_h]
                    covars = extract_covariates_lightweight(subj_cache, 1, covar_names_per_hazard[trans_h])
                    
                    log_haz = call_haz(
                        ub, hazard_pars, covars, hazard;
                        give_log = true,
                        apply_transform = hazard.metadata.time_transform,
                        cache_context = tt_context,
                        hazard_slot = trans_h)
                    
                    ll += log_haz
                end
            end
            
            # Update sojourn (reset on state change)
            sojourn = (statefrom != stateto) ? 0.0 : ub
        end
    else
        # Slow path: time-varying covariates (use full interval computation)
        intervals = compute_intervals_from_path(path, subj_cache)
        
        for interval in intervals
            tothaz = totalhazards[interval.statefrom]
            
            if tothaz isa _TotalHazardTransient
                for h in tothaz.components
                    hazard = hazards[h]
                    hazard_pars = pars[h]
                    covars = extract_covariates_lightweight(subj_cache, interval.covar_row_idx, covar_names_per_hazard[h])
                    
                    cumhaz = call_cumulhaz(
                        interval.lb, interval.ub, hazard_pars, covars, hazard;
                        give_log = false,
                        apply_transform = hazard.metadata.time_transform,
                        cache_context = tt_context,
                        hazard_slot = h)
                    
                    ll -= cumhaz
                end
                
                if interval.statefrom != interval.stateto
                    trans_h = tmat[interval.statefrom, interval.stateto]
                    hazard = hazards[trans_h]
                    hazard_pars = pars[trans_h]
                    covars = extract_covariates_lightweight(subj_cache, interval.covar_row_idx, covar_names_per_hazard[trans_h])
                    
                    log_haz = call_haz(
                        interval.ub, hazard_pars, covars, hazard;
                        give_log = true,
                        apply_transform = hazard.metadata.time_transform,
                        cache_context = tt_context,
                        hazard_slot = trans_h)
                    
                    ll += log_haz
                end
            end
        end
    end
    
    return ll
end
