########################################################
### Likelihod functions for sample paths, tpms, etc. ###
########################################################

# =============================================================================
# Helper Functions for ForwardDiff Dual Types
# =============================================================================

"""
    _unwrap_to_float(x)

Recursively unwrap ForwardDiff Dual numbers to their underlying Float64 value.
Handles nested Duals (e.g., during Hessian computation with forward-over-forward AD).
"""
_unwrap_to_float(x::Float64) = x
_unwrap_to_float(x::Real) = Float64(x)  # Handle other numeric types
_unwrap_to_float(x::ForwardDiff.Dual) = _unwrap_to_float(ForwardDiff.value(x))

# =============================================================================
# Parameter Preparation (dispatch-based normalization)
# =============================================================================

"""
    prepare_parameters(parameters, model::MultistateProcess)

Normalize parameter representations for downstream hazard calls.
Uses multiple dispatch to handle different parameter container types.

For flat vectors, this is equivalent to calling `unflatten_natural(p, model)`.

# Supported types
- `Tuple`: Nested parameters indexed by hazard number (returned as-is)
- `NamedTuple`: Parameters keyed by hazard name (returned as-is)
- `AbstractVector{<:AbstractVector}`: Already nested format (returned as-is)
- `AbstractVector{<:Real}`: Flat parameter vector (unflattened via `unflatten_natural`)

# Note on AD Compatibility
Uses `unflatten_natural` to handle both Float64 and ForwardDiff.Dual types correctly.

See also: [`unflatten_natural`](@ref), [`unflatten_estimation`](@ref)
"""
prepare_parameters(p::Tuple, ::MultistateProcess) = p
prepare_parameters(p::NamedTuple, ::MultistateProcess) = p
prepare_parameters(p::AbstractVector{<:AbstractVector}, ::MultistateProcess) = p

function prepare_parameters(p::AbstractVector{<:Real}, model::MultistateProcess)
    # Return NamedTuple indexed by hazard name (not Tuple of values)
    # Downstream code accesses parameters[hazard.hazname]
    return unflatten_natural(p, model)
end

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
            haz_value = eval_hazard(
                hazards[transind],
                subjectdata.sojourn[i] + subjectdata.increment[i],
                pars[transind],
                row_data;
                apply_transform = hazards[transind].metadata.time_transform,
                cache_context = tt_context,
                hazard_slot = transind)
            ll += log(haz_value)
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
- `parameters`: Tuple, NamedTuple, or flat AbstractVector (will be unflattened internally)
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

    # unflatten parameters to natural scale (AD-compatible)
    pars = unflatten_natural(parameters, data.model)

    # snag the hazards
    hazards = data.model.hazards

    # Remake spline parameters if needed
    # Note: For RuntimeSplineHazard, remake_splines! is a no-op
    for i in eachindex(hazards)
        if isa(hazards[i], _SplineHazard)
            remake_splines!(hazards[i], nothing)
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
    loglik_markov(parameters, data::MPanelData; neg=true, return_ll_subj=false, backend=nothing)

Unified Markov panel likelihood with automatic AD backend dispatch.

Return sum of (negative) log likelihood for a Markov model fit to panel and/or exact 
and/or censored data.

# Arguments
- `parameters`: Flat parameter vector
- `data::MPanelData`: Markov panel data container
- `neg::Bool=true`: Return negative log-likelihood
- `return_ll_subj::Bool=false`: Return vector of subject-level log-likelihoods
- `backend::Union{Nothing,ADBackend}=nothing`: AD backend for implementation dispatch
  - `nothing` or `ForwardDiffBackend()`: Use mutating implementation (default)
  - `EnzymeBackend()` or `MooncakeBackend()`: Use non-mutating functional implementation

# Implementation Details
- **Mutating implementation** (`_loglik_markov_mutating`): Uses in-place operations
  (`compute_hazmat!`, `compute_tmat!`) for efficiency. Compatible with ForwardDiff.
- **Functional implementation** (`_loglik_markov_functional`): Avoids all mutations
  for reverse-mode AD compatibility (Enzyme, Mooncake). ~10-20% slower due to allocations.

# Returns
- If `return_ll_subj=false`: Scalar (negative) log-likelihood
- If `return_ll_subj=true`: Vector of per-subject weighted log-likelihoods

# Note
For reverse-mode AD backends (Enzyme, Mooncake), `return_ll_subj` is not supported
and will raise an error.

See also: [`loglik_exact`](@ref), [`loglik_semi_markov`](@ref)
"""
function loglik_markov(parameters, data::MPanelData; 
                       neg = true, 
                       return_ll_subj = false,
                       backend::Union{Nothing, ADBackend} = nothing)
    # Dispatch based on backend type
    if isnothing(backend) || backend isa ForwardDiffBackend
        # Use mutating implementation (default, ForwardDiff-compatible)
        return _loglik_markov_mutating(parameters, data; neg=neg, return_ll_subj=return_ll_subj)
    elseif backend isa EnzymeBackend || backend isa MooncakeBackend
        # Use functional implementation (reverse-mode AD compatible)
        if return_ll_subj
            throw(ArgumentError("return_ll_subj=true is not supported with reverse-mode AD backends"))
        end
        return _loglik_markov_functional(parameters, data; neg=neg)
    else
        throw(ArgumentError("Unknown AD backend type: $(typeof(backend))"))
    end
end

"""
    _loglik_markov_mutating(parameters, data::MPanelData; neg=true, return_ll_subj=false)

Mutating implementation of Markov panel likelihood. Uses in-place operations
for efficiency. Compatible with ForwardDiff (forward-mode AD).

This is an internal function. Use `loglik_markov` for the public API.
"""
function _loglik_markov_mutating(parameters, data::MPanelData; neg = true, return_ll_subj = false)

    # Element type for AD compatibility
    T = eltype(parameters)

    # Pre-extracted column accessors for O(1) access (avoids DataFrame dispatch overhead)
    cols = data.columns
    
    # Check if we can use pre-computed hazard rates (Markov models only)
    # For Markov hazards, rates are time-invariant and can be computed once per likelihood call
    # But interaction terms (e.g., trt * age) can't be pre-cached, so check validity
    is_markov = data.model isa MultistateMarkovProcess
    can_use_rate_cache = is_markov && T === Float64 && 
                         !isempty(data.cache.hazard_rates_cache) &&
                         _covars_cache_valid(data.cache.covars_cache, data.model.hazards)

    # Use cached arrays for Float64, allocate fresh for Dual types (AD)
    if T === Float64
        # Reuse pre-allocated cache (zero-allocation path for non-AD evaluations)
        hazmat_book = data.cache.hazmat_book
        tpm_book = data.cache.tpm_book
        exp_cache = data.cache.exp_cache
        
        # Unflatten to NamedTuple structure (required by hazard functions)
        # NamedTuple allocations are minimal (~13) and mostly stack-allocated
        pars = unflatten_natural(parameters, data.model)
        
        # Pre-compute hazard rates once for all patterns (Markov models only)
        if can_use_rate_cache
            compute_hazard_rates!(data.cache.hazard_rates_cache, pars, 
                                  data.model.hazards, data.cache.covars_cache)
        end
        
        # Reset hazard matrices and compute TPMs
        # Use batched eigendecomposition approach when multiple Δt values exist
        @inbounds for t in eachindex(data.books[1])
            # Use cached rates path for Markov models
            if can_use_rate_cache
                compute_hazmat_from_rates!(
                    hazmat_book[t],
                    data.cache.hazard_rates_cache[t],
                    data.model.hazards)
            else
                fill!(hazmat_book[t], zero(Float64))
                compute_hazmat!(
                    hazmat_book[t],
                    pars,
                    data.model.hazards,
                    data.books[1][t],
                    data.model.data)
            end
            
            # Invalidate eigen cache when Q changes (new parameters)
            invalidate_eigen_cache!(data.cache, t)
            
            # Use batched approach when >= 2 unique Δt values (eigendecomp pays off)
            n_dt = length(data.cache.dt_values[t])
            if n_dt >= 2
                compute_tmat_batched!(
                    tpm_book[t],
                    hazmat_book[t],
                    data.cache.dt_values[t],
                    data.cache.eigen_cache,
                    t)
            else
                # Single Δt: use standard approach
                compute_tmat!(
                    tpm_book[t],
                    hazmat_book[t],
                    data.books[1][t],
                    exp_cache)
            end
        end
    else
        # Allocate fresh arrays for AD (element type must be Dual)
        pars = unflatten_natural(parameters, data.model)
        hazmat_book = build_hazmat_book(T, data.model.tmat, data.books[1])
        tpm_book = build_tpm_book(T, data.model.tmat, data.books[1])
        exp_cache = ExponentialUtilities.alloc_mem(similar(hazmat_book[1]), ExpMethodGeneric())
        
        # Solve Kolmogorov equations for TPMs
        @inbounds for t in eachindex(data.books[1])
            compute_hazmat!(
                hazmat_book[t],
                pars,
                data.model.hazards,
                data.books[1][t],
                data.model.data)
            compute_tmat!(
                tpm_book[t],
                hazmat_book[t],
                data.books[1][t],
                exp_cache)
        end
    end

    # number of subjects
    nsubj = length(data.model.subjectindices)

    # accumulate the log likelihood
    ll = zero(T)

    # container for subject-level loglikelihood
    if return_ll_subj
        ll_subj = zeros(T, nsubj)
    end

    # number of states
    S = size(data.model.tmat, 1)

    # initialize Q (use cached version for Float64, allocate for AD)
    if T === Float64
        q = data.cache.q_work
        fill!(q, zero(Float64))
    else
        q = zeros(T, S, S)
    end
    
    # check if observation weights are provided (once, outside loop)
    has_obs_weights = !isnothing(data.model.ObservationWeights)
    
    # Cache hazard covar_names to avoid repeated lookups (once, outside loop)
    hazards = data.model.hazards

    # for each subject, compute the likelihood contribution
    for subj in Base.OneTo(nsubj)

        # subject data
        subj_inds = data.model.subjectindices[subj]

        # Check if all observations are exact (1) or panel (2) - no censoring
        # Manual loop avoids broadcast allocations from all(... .∈ Ref([1,2]))
        all_exact_or_panel = true
        @inbounds for i in subj_inds
            if cols.obstype[i] > 2
                all_exact_or_panel = false
                break
            end
        end

        # no state is censored
        if all_exact_or_panel
            
            # subject contribution to the loglikelihood
            subj_ll = zero(T)

            # add the contribution of each observation
            @inbounds for i in subj_inds
                # get observation weight (default to 1.0)
                obs_weight = has_obs_weights ? data.model.ObservationWeights[i] : 1.0
                
                obstype_i = cols.obstype[i]
                
                if obstype_i == 1 # exact data
                    # Use @view to avoid DataFrameRow allocation
                    row_data = @view data.model.data[i, :]
                    
                    statefrom_i = cols.statefrom[i]
                    stateto_i = cols.stateto[i]
                    dt = cols.tstop[i] - cols.tstart[i]
                    
                    obs_ll = dt ≈ 0 ? 0 : survprob(
                        0,
                        dt,
                        pars,
                        row_data,
                        data.model.totalhazards[statefrom_i],
                        hazards;
                        give_log = true)
                                        
                    if statefrom_i != stateto_i # if there is a transition, add log hazard
                        trans_idx = data.model.tmat[statefrom_i, stateto_i]
                        hazard = hazards[trans_idx]
                        hazard_pars = pars[hazard.hazname]
                        haz_value = eval_hazard(hazard, dt, hazard_pars, row_data)
                        obs_ll += log(haz_value)
                    end
                    
                    subj_ll += obs_ll * obs_weight

                else # panel data (obstype == 2, since we're in all_exact_or_panel branch)
                    # Forward algorithm for single observation:
                    # L = Σ_s P(X=s | X_0=statefrom) * P(Y | X=s)
                    #   = Σ_s TPM[statefrom, s] * emat[i, s]
                    #
                    # When emat[i, stateto] = 1 and emat[i, other] = 0 (default for exact observations),
                    # this reduces to TPM[statefrom, stateto].
                    # When EmissionMatrix provides soft evidence, this properly weights all states.
                    statefrom_i = cols.statefrom[i]
                    book_idx1 = cols.tpm_map_col1[i]
                    book_idx2 = cols.tpm_map_col2[i]
                    tpm = tpm_book[book_idx1][book_idx2]
                    
                    # Compute likelihood: Σ_s P(transition to s) * P(observation | s)
                    prob_sum = zero(T)
                    for s in 1:S
                        emission_prob = data.model.emat[i, s]
                        if emission_prob > 0
                            prob_sum += tpm[statefrom_i, s] * emission_prob
                        end
                    end
                    subj_ll += log(prob_sum) * obs_weight
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
            
            # initialize likelihood matrix (use cache for Float64)
            nobs_subj = length(subj_inds)
            if T === Float64
                # Use cached lmat_work, resize if needed
                if size(data.cache.lmat_work, 2) < nobs_subj + 1
                    data.cache.lmat_work = zeros(Float64, S, nobs_subj + 1)
                end
                lmat = @view data.cache.lmat_work[:, 1:(nobs_subj + 1)]
                fill!(lmat, zero(Float64))
            else
                lmat = zeros(T, S, nobs_subj + 1)
            end
            @inbounds lmat[cols.statefrom[subj_inds[1]], 1] = 1

            # initialize counter for likelihood matrix
            ind = 1

            # update the vector l
            @inbounds for i in subj_inds

                # increment counter for likelihood matrix
                ind += 1
                
                obstype_i = cols.obstype[i]
                dt = cols.tstop[i] - cols.tstart[i]

                # compute q, the transition probability matrix
                if obstype_i != 1
                    # if panel data, simply grab q from tpm_book
                    book_idx1 = cols.tpm_map_col1[i]
                    book_idx2 = cols.tpm_map_col2[i]
                    copyto!(q, tpm_book[book_idx1][book_idx2])
                    
                else
                    # if exact data (obstype = 1), compute q by hand
                    # reset Q
                    fill!(q, -Inf)
                    
                    # Use @view for data row
                    row_data = @view data.model.data[i, :]
                    
                    if dt ≈ 0
                        # Instantaneous observation (dt=0): use hazard ratios
                        # This occurs in phase-type expanded data where transitions
                        # are recorded at a single point in time.
                        # q[r,s] = h(r,s) / Σ_k h(r,k) = conditional prob of destination s
                        fill!(q, zero(eltype(q)))
                        for r in 1:S
                            if isa(data.model.totalhazards[r], _TotalHazardAbsorbing)
                                q[r,r] = 1.0  # Absorbing state stays in itself
                            else
                                dest_states = transient_dests[r]
                                # Compute hazards for each destination
                                total_haz = zero(eltype(q))
                                for s in dest_states
                                    trans_idx = tmat_cache[r, s]
                                    hazard = hazards[trans_idx]
                                    hazard_pars = pars[hazard.hazname]
                                    haz_value = eval_hazard(hazard, dt, hazard_pars, row_data)
                                    q[r, s] = haz_value
                                    total_haz += haz_value
                                end
                                # Normalize to get conditional probabilities
                                if total_haz > 0
                                    for s in dest_states
                                        q[r, s] /= total_haz
                                    end
                                end
                                # For instantaneous obs, diagonal is 0 (transition definitely occurred)
                                q[r, r] = 0.0
                            end
                        end
                    else
                        # Regular exact observation (dt > 0)
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
                                    hazard = hazards[trans_idx]
                                    hazard_pars = pars[hazard.hazname]
                                    haz_value = eval_hazard(hazard, dt, hazard_pars, row_data)
                                    q[r, s] += log(haz_value)
                                end
                            end

                            # pedantic b/c of numerical error
                            q[r,r] = maximum([1 - exp(logsumexp(q[r, Not(r)])), eps()])
                            q[r,Not(r)] = exp.(q[r, Not(r)])               
                        end
                    end
                end # end-compute q

                # Forward algorithm update: α_i(s) = e_{is} * Σ_r Q_{rs} * α_{i-1}(r)
                # where e_{is} = P(Y_i | X_i = s) is the emission probability.
                #
                # The emission matrix is constructed by build_emat():
                # - For exact observations: e[i, stateto] = 1, e[i, other] = 0
                # - For censored observations: e[i, :] from CensoringPatterns
                # - For user-supplied EmissionMatrix: uses provided soft evidence
                #
                # This unified approach enables soft evidence for any observation type.
                for s in 1:S
                    emission_prob = data.model.emat[i, s]
                    if emission_prob > 0
                        for r in 1:S
                            lmat[s, ind] += q[r, s] * lmat[r, ind - 1] * emission_prob
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

# =============================================================================
# Non-Mutating Markov Likelihood (Reverse-Mode AD Compatible)
# =============================================================================

"""
    _loglik_markov_functional(parameters, data::MPanelData; neg=true)

Non-mutating implementation of Markov panel likelihood for reverse-mode AD (Enzyme, Mooncake).

This version avoids all in-place mutations:
- Uses `compute_hazmat` and `compute_tmat` instead of their `!` variants
- Accumulates log-likelihood functionally without pre-allocated containers
- Uses comprehensions and reductions instead of loops with mutation

Performance: ~10-20% slower than mutating version due to allocations,
but enables reverse-mode gradient computation which may be faster overall for large models.

This is an internal function. Use `loglik_markov(params, data; backend=EnzymeBackend())` 
for the public API.

# Arguments
- `parameters`: Flat parameter vector
- `data::MPanelData`: Markov panel data container
- `neg::Bool=true`: Return negative log-likelihood

# Returns
Scalar (negative) log-likelihood value.
"""
function _loglik_markov_functional(parameters, data::MPanelData; neg = true)
    # Unflatten parameters to natural scale (AD-compatible)
    pars = unflatten_natural(parameters, data.model)
    
    # Model components
    hazards = data.model.hazards
    tmat = data.model.tmat
    n_states = size(tmat, 1)
    T = eltype(parameters)
    
    # Pre-extracted column accessors for O(1) access (avoids DataFrame dispatch overhead)
    cols = data.columns
    
    # Build TPM book functionally (no mutation)
    # For each unique covariate/time pattern, compute Q then P = exp(Q*dt)
    tpm_dict = Dict{Tuple{Int,Int}, Matrix{T}}()
    for (t_idx, tpm_index_df) in enumerate(data.books[1])
        Q = compute_hazmat(T, n_states, pars, hazards, tpm_index_df, data.model.data)
        # Compute P for each time interval in this pattern
        for t in eachindex(tpm_index_df.tstop)
            dt = tpm_index_df.tstop[t]
            P = compute_tmat(Q, dt)
            # Store with composite key (could be optimized)
            tpm_dict[(t_idx, t)] = P
        end
    end
    
    # Accumulate log-likelihood functionally
    nsubj = length(data.model.subjectindices)
    has_obs_weights = !isnothing(data.model.ObservationWeights)
    
    # Compute subject contributions using map (no mutation)
    subj_contributions = map(1:nsubj) do subj
        subj_inds = data.model.subjectindices[subj]
        subj_weight = data.model.SubjectWeights[subj]
        
        # Check if any censored observations using direct column access
        # Manual loop avoids broadcast allocations from all(... .∈ Ref([1,2]))
        all_uncensored = true
        for i in subj_inds
            if cols.obstype[i] > 2
                all_uncensored = false
                break
            end
        end
        
        if all_uncensored
            # Simple case: no forward algorithm needed
            subj_ll = sum(subj_inds) do i
                obs_weight = has_obs_weights ? data.model.ObservationWeights[i] : one(T)
                obstype_i = cols.obstype[i]
                
                if obstype_i == 1  # exact data
                    row_data = @view data.model.data[i, :]
                    statefrom_i = cols.statefrom[i]
                    stateto_i = cols.stateto[i]
                    dt = cols.tstop[i] - cols.tstart[i]
                    
                    obs_ll = dt ≈ 0 ? 0 : survprob(zero(T), dt, pars, row_data, 
                                     data.model.totalhazards[statefrom_i], hazards; 
                                     give_log = true)
                    
                    if statefrom_i != stateto_i
                        trans_idx = tmat[statefrom_i, stateto_i]
                        hazard = hazards[trans_idx]
                        hazard_pars = pars[hazard.hazname]
                        haz_value = eval_hazard(hazard, dt, hazard_pars, row_data)
                        obs_ll += log(haz_value)
                    end
                    
                    obs_ll * obs_weight
                else  # panel data (obstype == 2, since we're in all_exact_or_panel branch)
                    # Forward algorithm for single observation:
                    # L = Σ_s P(X=s | X_0=statefrom) * P(Y | X=s)
                    #   = Σ_s TPM[statefrom, s] * emat[i, s]
                    statefrom_i = cols.statefrom[i]
                    book_idx1 = cols.tpm_map_col1[i]
                    book_idx2 = cols.tpm_map_col2[i]
                    P = tpm_dict[(book_idx1, book_idx2)]
                    
                    # Compute likelihood: Σ_s P(transition to s) * P(observation | s)
                    prob_sum = zero(T)
                    for s in 1:n_states
                        emission_prob = data.model.emat[i, s]
                        if emission_prob > zero(T)
                            prob_sum += P[statefrom_i, s] * emission_prob
                        end
                    end
                    log(prob_sum) * obs_weight
                end
            end
            
            subj_ll * subj_weight
        else
            # Forward algorithm for censored observations
            # Build log-likelihood via matrix-vector products
            subj_ll = _forward_algorithm_functional(
                subj_inds, pars, data, tpm_dict, T, n_states, hazards, tmat
            )
            subj_ll * subj_weight
        end
    end
    
    ll = sum(subj_contributions)
    return neg ? -ll : ll
end

# Backward compatibility alias
# Use loglik_markov(params, data; backend=EnzymeBackend()) instead
const loglik_markov_functional = _loglik_markov_functional

"""
    _forward_algorithm_functional(subj_inds, pars, data, tpm_dict, T, n_states, hazards, tmat)

Non-mutating forward algorithm for censored state observations.
Returns the log-likelihood contribution for one subject.
"""
function _forward_algorithm_functional(subj_inds, pars, data, tpm_dict, ::Type{T}, 
                                       n_states::Int, hazards, tmat) where T
    # Pre-extracted column accessors for O(1) access
    cols = data.columns
    
    # Initialize: probability vector for initial state
    init_state = cols.statefrom[subj_inds[1]]
    α = zeros(T, n_states)
    α = setindex_immutable_vec(α, one(T), init_state)
    
    # Forward pass: α[t+1] = α[t] * P[t] (with emission probabilities for censored states)
    for i in subj_inds
        obstype_i = cols.obstype[i]
        dt = cols.tstop[i] - cols.tstart[i]
        
        if obstype_i == 1
            # Exact data: compute transition probabilities directly
            row_data = @view data.model.data[i, :]
            statefrom = cols.statefrom[i]
            stateto = cols.stateto[i]
            
            # Survival probability + hazard
            log_surv = dt ≈ 0 ? 0 : survprob(zero(T), dt, pars, row_data,
                               data.model.totalhazards[statefrom], hazards; give_log = true)
            
            if statefrom != stateto
                trans_idx = tmat[statefrom, stateto]
                hazard = hazards[trans_idx]
                hazard_pars = pars[hazard.hazname]
                haz_value = eval_hazard(hazard, dt, hazard_pars, row_data)
                log_prob = log_surv + log(haz_value)
            else
                log_prob = log_surv
            end
            
            # Update probability vector: concentrate mass on observed state
            new_α = zeros(T, n_states)
            prob_from = α[statefrom]
            new_α = setindex_immutable_vec(new_α, prob_from * exp(log_prob), stateto)
            α = new_α
        else
            # Panel/censored data: multiply by TPM
            book_idx1 = cols.tpm_map_col1[i]
            book_idx2 = cols.tpm_map_col2[i]
            P = tpm_dict[(book_idx1, book_idx2)]
            
            # Matrix-vector product (non-mutating)
            α = P' * α  # transpose because α is a column vector
            
            # Apply emission probabilities: α_i(s) = e_{is} * α_i(s)
            # The emission matrix is constructed by build_emat():
            # - For exact observations: e[i, stateto] = 1, e[i, other] = 0
            # - For censored observations: e[i, :] from CensoringPatterns
            # - For user-supplied EmissionMatrix: uses provided soft evidence
            #
            # This unified approach enables soft evidence for any observation type.
            new_α = zeros(T, n_states)
            for s in 1:n_states
                emission_prob = data.model.emat[i, s]
                if emission_prob > zero(T)
                    new_α = setindex_immutable_vec(new_α, new_α[s] + α[s] * emission_prob, s)
                end
            end
            α = new_α
        end
    end
    
    # Log-likelihood is log of sum of final probabilities
    return log(sum(α))
end

"""
    setindex_immutable_vec(v, val, i)

Return a new vector with v[i] = val without mutating v.
"""
@inline function setindex_immutable_vec(v::AbstractVector{T}, val, i::Int) where T
    w = copy(v)
    w[i] = convert(T, val)
    return w
end

"""
    loglik_semi_markov(parameters, data::SMPanelData; neg=true, use_sampling_weight=true, parallel=false)

Compute importance-weighted log-likelihood for semi-Markov panel data (MCEM).

This function computes:
```math
Q(θ|θ') = Σᵢ SubjectWeights[i] × Σⱼ ImportanceWeights[i][j] × ℓᵢⱼ(θ)
```

where `ℓᵢⱼ` is the complete-data log-likelihood for path j of subject i.

# Arguments
- `parameters`: Flat parameter vector
- `data::SMPanelData`: Semi-Markov panel data with sample paths and importance weights
- `neg::Bool=true`: Return negative log-likelihood
- `use_sampling_weight::Bool=true`: Apply subject sampling weights
- `parallel::Bool=false`: Use multi-threaded parallel computation

# Parallel Execution
When `parallel=true` and `Threads.nthreads() > 1`, uses flat path-level parallelism
with `@threads :static`. This provides good load balance even when subjects have
highly variable numbers of paths (e.g., some subjects have 1 path, others have 500).

The flat-indexing approach maps each (subject, path) pair to a linear index,
ensuring work is distributed evenly across threads regardless of path distribution.

# Returns
Scalar (negative) log-likelihood value.

# See Also
- [`mcem_mll`](@ref): Uses this for MCEM objective computation
- [`loglik_semi_markov!`](@ref): In-place version for path log-likelihoods
"""
function loglik_semi_markov(parameters, data::SMPanelData; neg=true, use_sampling_weight=true, parallel=false)

    # Unflatten parameters to natural scale (AD-compatible)
    pars = unflatten_natural(parameters, data.model)

    # Get hazards and model components
    hazards = data.model.hazards
    totalhazards = data.model.totalhazards
    tmat = data.model.tmat
    n_hazards = length(hazards)
    nsubj = length(data.paths)

    # Update spline hazards with current parameters (no-op for functional splines)
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
    
    # Check if any hazard uses time transform
    any_time_transform = any(h -> h.metadata.time_transform, hazards)
    
    # Count total paths for parallel decision
    n_total_paths = sum(length(data.paths[i]) for i in eachindex(data.paths))
    use_parallel = parallel && Threads.nthreads() > 1 && n_total_paths >= 50
    
    if use_parallel
        # Flat path-level parallelism for load balance
        # Build flat index mapping: path k → (subject i, path j)
        path_to_subj = Vector{Int}(undef, n_total_paths)
        path_to_j = Vector{Int}(undef, n_total_paths)
        k = 1
        for i in eachindex(data.paths)
            for j in eachindex(data.paths[i])
                path_to_subj[k] = i
                path_to_j[k] = j
                k += 1
            end
        end
        
        # Pre-allocate flat log-likelihood array
        ll_flat = Vector{T}(undef, n_total_paths)
        
        # Parallel over flat path index
        Threads.@threads :static for k in 1:n_total_paths
            i = path_to_subj[k]
            j = path_to_j[k]
            path = data.paths[i][j]
            subj_cache = subject_covars[path.subj]
            
            # Thread-local TimeTransformContext
            tt_context = if any_time_transform
                sample_df = isempty(subj_cache.covar_data) ? nothing : subj_cache.covar_data[1:1, :]
                maybe_time_transform_context(pars, sample_df, hazards)
            else
                nothing
            end
            
            ll_flat[k] = _compute_path_loglik_fused(
                path, pars, hazards, totalhazards, tmat,
                subj_cache, covar_names_per_hazard, tt_context, T
            )
        end
        
        # Weighted reduction: reassemble by subject and apply importance weights
        ll = zero(T)
        k = 1
        for i in eachindex(data.paths)
            subj_ll = zero(T)
            for j in eachindex(data.paths[i])
                subj_ll += ll_flat[k] * data.ImportanceWeights[i][j]
                k += 1
            end
            if use_sampling_weight
                ll += subj_ll * data.model.SubjectWeights[i]
            else
                ll += subj_ll
            end
        end
    else
        # Sequential path: simpler, AD-compatible
        tt_context = if any_time_transform && !isempty(data.paths) && !isempty(data.paths[1])
            sample_subj = subject_covars[data.paths[1][1].subj]
            sample_df = isempty(sample_subj.covar_data) ? nothing : sample_subj.covar_data[1:1, :]
            maybe_time_transform_context(pars, sample_df, hazards)
        else
            nothing
        end

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
    end

    # Return the log-likelihood
    neg ? -ll : ll
end

"""
    loglik(parameters, data::SMPanelData)

Update log-likelihood for each individual and each path of panel data in a semi-Markov model.

This implementation uses the fused path-centric approach from `loglik_exact`, calling
`_compute_path_loglik_fused` directly to avoid DataFrame allocation overhead.

# Notes on future neural ODE compatibility:
When `is_separable(hazard) == false` for ODE-based hazards, the `eval_cumhaz` 
function in `_compute_path_loglik_fused` is the extension point where numerical 
ODE solvers would be invoked instead of analytic cumulative hazard formulas.
"""
function loglik_semi_markov!(parameters, logliks::Vector{}, data::SMPanelData)

    # Unflatten parameters to natural scale (AD-compatible)
    pars = unflatten_natural(parameters, data.model)

    # snag the hazards and model components
    hazards = data.model.hazards
    totalhazards = data.model.totalhazards
    tmat = data.model.tmat
    n_hazards = length(hazards)

    # Remake spline parameters if needed
    # Note: For RuntimeSplineHazard, remake_splines! is a no-op
    for i in eachindex(hazards)
        if isa(hazards[i], _SplineHazard)
            remake_splines!(hazards[i], nothing)
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
    # Unflatten parameters to natural scale (AD-compatible)
    pars = unflatten_natural(parameters, data.model)
    
    # Get hazards
    hazards = data.model.hazards
    n_hazards = length(hazards)
    
    # Remake spline parameters if needed
    # Note: For RuntimeSplineHazard, remake_splines! is a no-op
    for i in eachindex(hazards)
        if isa(hazards[i], _SplineHazard)
            remake_splines!(hazards[i], nothing)
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
        hazard_pars = pars[hazard.hazname]
        use_transform = hazard.metadata.time_transform
        
        for i in 1:n_intervals
            # Cumulative hazard contribution (survival component)
            cumhaz = eval_cumhaz(
                hazard, sd.lb[i], sd.ub[i], hazard_pars, sd.covars[i];
                apply_transform = use_transform,
                cache_context = tt_context,
                hazard_slot = h)
            
            ll_flat[sd.path_idx[i]] -= cumhaz
            
            # Transition hazard
            if sd.is_transition[i]
                haz_value = eval_hazard(
                    hazard, sd.transition_times[i], hazard_pars, sd.covars[i];
                    apply_transform = use_transform,
                    cache_context = tt_context,
                    hazard_slot = h)
                ll_flat[sd.path_idx[i]] += log(haz_value)
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

For repeated calls (e.g., MCEM), uses thread-local workspace to reduce allocations.
"""
function compute_intervals_from_path(path::SamplePath, subject_covar::SubjectCovarCache)
    # Get thread-local workspace for TVC case
    ws = get_tvc_workspace()
    return compute_intervals_from_path!(ws, path, subject_covar)
end

"""
    compute_intervals_from_path!(ws::TVCIntervalWorkspace, path::SamplePath, subject_covar::SubjectCovarCache)

Workspace-based version that minimizes allocations for repeated calls.
"""
function compute_intervals_from_path!(ws::TVCIntervalWorkspace, path::SamplePath, subject_covar::SubjectCovarCache)
    n_transitions = length(path.times) - 1
    
    if isempty(subject_covar.covar_data) || nrow(subject_covar.covar_data) <= 1
        # No time-varying covariates - use path times directly
        # Still need to allocate result, but workspace not needed
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
        # Time-varying covariates - use workspace to reduce allocations
        tstart = subject_covar.tstart
        covar_data = subject_covar.covar_data
        
        # Identify covariate change times using workspace
        n_change = 1
        @inbounds ws.change_times[1] = tstart[1]
        for i in 2:length(tstart)
            if !isequal(covar_data[i-1, :], covar_data[i, :])
                n_change += 1
                if n_change > length(ws.change_times)
                    resize!(ws.change_times, 2 * n_change)
                end
                ws.change_times[n_change] = tstart[i]
            end
        end
        
        # Merge path times with covariate change times into utimes
        # Collect all unique times in range
        n_utimes = 0
        path_start = path.times[1]
        path_end = path.times[end]
        
        # Add path times
        for t in path.times
            n_utimes += 1
            if n_utimes > length(ws.utimes)
                resize!(ws.utimes, 2 * n_utimes)
            end
            @inbounds ws.utimes[n_utimes] = t
        end
        
        # Add change times within range
        for i in 1:n_change
            t = ws.change_times[i]
            if path_start <= t <= path_end
                n_utimes += 1
                if n_utimes > length(ws.utimes)
                    resize!(ws.utimes, 2 * n_utimes)
                end
                @inbounds ws.utimes[n_utimes] = t
            end
        end
        
        # Sort and remove duplicates (in-place)
        sort!(@view(ws.utimes[1:n_utimes]))
        
        # Unique in-place
        j = 1
        @inbounds for i in 2:n_utimes
            if ws.utimes[i] != ws.utimes[j]
                j += 1
                ws.utimes[j] = ws.utimes[i]
            end
        end
        n_utimes = j
        
        n_intervals = n_utimes - 1
        
        # Ensure workspace capacity
        if n_intervals > length(ws.intervals)
            resize!(ws.intervals, max(n_intervals, 2 * length(ws.intervals)))
            resize!(ws.sojourns, length(ws.intervals))
            resize!(ws.pathinds, length(ws.intervals) + 1)
            resize!(ws.datinds, length(ws.intervals) + 1)
        end
        
        # Compute path and data indices
        @inbounds for i in 1:n_utimes
            ws.pathinds[i] = searchsortedlast(path.times, ws.utimes[i])
            ws.datinds[i] = searchsortedlast(tstart, ws.utimes[i])
        end
        
        # Compute sojourns
        current_sojourn = 0.0
        current_pathind = ws.pathinds[1]
        
        @inbounds for i in 1:n_intervals
            increment = ws.utimes[i+1] - ws.utimes[i]
            
            if ws.pathinds[i] != current_pathind
                current_sojourn = 0.0
                current_pathind = ws.pathinds[i]
            end
            
            ws.sojourns[i] = current_sojourn
            current_sojourn += increment
        end
        
        # Build intervals (need to allocate result vector)
        intervals = Vector{LightweightInterval}(undef, n_intervals)
        @inbounds for i in 1:n_intervals
            increment = ws.utimes[i+1] - ws.utimes[i]
            intervals[i] = LightweightInterval(
                ws.sojourns[i],
                ws.sojourns[i] + increment,
                path.states[ws.pathinds[i]],
                path.states[ws.pathinds[i+1]],
                ws.datinds[i]
            )
        end
        
        return intervals
    end
end

"""
    extract_covariates_lightweight(subject_covar::SubjectCovarCache, row_idx::Int, covar_names::Vector{Symbol})

Extract covariates from the subject cache without DataFrame row access overhead.

Handles interaction terms (e.g., `Symbol("trt & age")`) by computing the product
of their component values.
"""
@inline function extract_covariates_lightweight(subject_covar::SubjectCovarCache, row_idx::Int, covar_names::Vector{Symbol})
    isempty(covar_names) && return NamedTuple()
    
    # Clamp row_idx to valid range
    idx = clamp(row_idx, 1, max(1, nrow(subject_covar.covar_data)))
    
    if isempty(subject_covar.covar_data)
        return NamedTuple()
    end
    
    # Extract values for requested covariates, handling interaction terms
    values = Tuple(_lookup_covar_value_lightweight(subject_covar.covar_data, idx, cname) for cname in covar_names)
    return NamedTuple{Tuple(covar_names)}(values)
end

"""
    _lookup_covar_value_lightweight(covar_data::DataFrame, row_idx::Int, cname::Symbol)

Look up a covariate value, handling interaction terms (e.g., `:"trt & age"`).

For interaction terms, computes the product of the component values.
"""
@inline function _lookup_covar_value_lightweight(covar_data::DataFrame, row_idx::Int, cname::Symbol)
    # Direct column access for simple covariates
    if hasproperty(covar_data, cname)
        return covar_data[row_idx, cname]
    end
    
    # Handle interaction terms: "trt & age" or "trt:age"
    cname_str = String(cname)
    if occursin("&", cname_str)
        parts = split(cname_str, "&")
        return prod(_lookup_covar_value_lightweight(covar_data, row_idx, Symbol(strip(part))) for part in parts)
    elseif occursin(":", cname_str)
        parts = split(cname_str, ":")
        return prod(_lookup_covar_value_lightweight(covar_data, row_idx, Symbol(strip(part))) for part in parts)
    else
        throw(ArgumentError("column name :$cname not found in covariate data"))
    end
end

"""
    loglik_exact(parameters, data::ExactData; neg=true, return_ll_subj=false, parallel=false)

Compute log-likelihood for exact (fully observed) multistate data.

# Arguments
- `parameters`: Flat parameter vector
- `data::ExactData`: Exact data containing model and sample paths
- `neg::Bool=true`: Return negative log-likelihood
- `return_ll_subj::Bool=false`: Return per-path weighted log-likelihoods instead of scalar
- `parallel::Bool=false`: Use multi-threaded parallel computation

# Parallel Execution
When `parallel=true` and `Threads.nthreads() > 1`, uses `@threads :static` for 
path-level parallelism. This is beneficial when:
- Number of paths > 100
- Per-path computation cost > 10μs

Note: Parallel mode is NOT used during AD gradient computation (ForwardDiff uses
the sequential path). Use parallel for objective evaluation during line search.

# Returns
- If `return_ll_subj=false`: Scalar (negative) log-likelihood
- If `return_ll_subj=true`: Vector of per-path weighted log-likelihoods
"""
function loglik_exact(parameters, data::ExactData; neg=true, return_ll_subj=false, parallel=false)
    # Unflatten parameters to natural scale - preserves dual number types (AD-compatible)
    pars = unflatten_natural(parameters, data.model)
    
    # Get model components
    hazards = data.model.hazards
    totalhazards = data.model.totalhazards
    tmat = data.model.tmat
    n_hazards = length(hazards)
    n_paths = length(data.paths)
    
    # Remake spline parameters if needed
    # Note: For RuntimeSplineHazard (the current implementation), remake_splines! is a no-op
    # since splines are constructed on-the-fly during evaluation. We still call it for 
    # future-proofing if other spline implementations need parameter updates.
    for i in eachindex(hazards)
        if isa(hazards[i], _SplineHazard)
            # RuntimeSplineHazard.remake_splines! is a no-op, but call for extensibility
            remake_splines!(hazards[i], nothing)
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
    
    # Subject weights (precomputed lookup for efficiency)
    subj_weights = data.model.SubjectWeights
    
    # Parallel vs sequential execution
    use_parallel = parallel && Threads.nthreads() > 1 && n_paths >= 10
    
    if use_parallel
        # Parallel path: pre-allocate and use @threads :static
        ll_array = Vector{T}(undef, n_paths)
        
        Threads.@threads :static for path_idx in 1:n_paths
            path = data.paths[path_idx]
            
            # Thread-local TimeTransformContext (avoid cache sharing)
            tt_context = if any_time_transform
                sample_subj = subject_covars[path.subj]
                sample_df = isempty(sample_subj.covar_data) ? nothing : sample_subj.covar_data[1:1, :]
                maybe_time_transform_context(pars, sample_df, hazards)
            else
                nothing
            end
            
            ll_array[path_idx] = _compute_path_loglik_fused(
                path, pars, hazards, totalhazards, tmat,
                subject_covars[path.subj], covar_names_per_hazard,
                tt_context, T
            )
        end
    else
        # Sequential path: functional style for reverse-mode AD compatibility
        # Create TimeTransformContext once (shared across all paths)
        tt_context = if any_time_transform && !isempty(data.paths)
            sample_subj = subject_covars[data.paths[1].subj]
            sample_df = isempty(sample_subj.covar_data) ? nothing : sample_subj.covar_data[1:1, :]
            maybe_time_transform_context(pars, sample_df, hazards)
        else
            nothing
        end
        
        ll_paths = map(enumerate(data.paths)) do (path_idx, path)
            _compute_path_loglik_fused(
                path, pars, hazards, totalhazards, tmat, 
                subject_covars[path.subj], covar_names_per_hazard,
                tt_context, T
            )
        end
        ll_array = collect(T, ll_paths)
    end
    
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
accumulating log-survival contributions (via `eval_cumhaz`) and transition hazard 
contributions (via `eval_hazard`).

# Neural ODE Extension Point
The `eval_cumhaz` invocations are the extension points for neural ODE-based hazards.
When `is_separable(hazard) == false` for an ODE-based hazard, `eval_cumhaz` should be 
extended to invoke a numerical ODE solver (e.g., DifferentialEquations.jl) to compute 
the cumulative hazard as an integral: Λ(t₀, t₁) = ∫_{t₀}^{t₁} λ(s) ds.

For reverse-mode AD compatibility with neural ODEs:
- Use SciMLSensitivity.jl adjoints (BacksolveAdjoint, QuadratureAdjoint)
- Ensure the hazard's metadata declares supported adjoint methods
- The functional accumulation style in this function avoids in-place mutation 
  required for Zygote/Enzyme compatibility

See also: `loglik_exact`, `loglik_semi_markov`, `eval_cumhaz`
"""
function _compute_path_loglik_fused(
    path::SamplePath, 
    pars,  # Parameters as nested structure (from unflatten or Tuple)
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
    
    # Pre-extract hazard parameters by index to avoid repeated NamedTuple lookups
    # This converts dynamic symbol lookup to static tuple indexing
    pars_indexed = values(pars)  # Convert NamedTuple to Tuple for indexed access
    
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
                    hazard_pars = pars_indexed[h]  # Fast indexed access
                    
                    # Extract covariates
                    covars = extract_covariates_lightweight(subj_cache, 1, covar_names_per_hazard[h])
                    
                    # Cumulative hazard (use hazard-specific transform flag)
                    cumhaz = eval_cumhaz(
                        hazard, lb, ub, hazard_pars, covars;
                        apply_transform = hazard.metadata.time_transform,
                        cache_context = tt_context,
                        hazard_slot = h)
                    
                    ll -= cumhaz
                end
                
                # Add transition hazard if transition occurred
                if statefrom != stateto
                    trans_h = tmat[statefrom, stateto]
                    hazard = hazards[trans_h]
                    hazard_pars = pars_indexed[trans_h]  # Fast indexed access
                    covars = extract_covariates_lightweight(subj_cache, 1, covar_names_per_hazard[trans_h])
                    
                    haz_value = eval_hazard(
                        hazard, ub, hazard_pars, covars;
                        apply_transform = hazard.metadata.time_transform,
                        cache_context = tt_context,
                        hazard_slot = trans_h)
                    
                    ll += log(haz_value)
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
                    hazard_pars = pars_indexed[h]  # Fast indexed access
                    covars = extract_covariates_lightweight(subj_cache, interval.covar_row_idx, covar_names_per_hazard[h])
                    
                    cumhaz = eval_cumhaz(
                        hazard, interval.lb, interval.ub, hazard_pars, covars;
                        apply_transform = hazard.metadata.time_transform,
                        cache_context = tt_context,
                        hazard_slot = h)
                    
                    ll -= cumhaz
                end
                
                if interval.statefrom != interval.stateto
                    trans_h = tmat[interval.statefrom, interval.stateto]
                    hazard = hazards[trans_h]
                    hazard_pars = pars_indexed[trans_h]  # Fast indexed access
                    covars = extract_covariates_lightweight(subj_cache, interval.covar_row_idx, covar_names_per_hazard[trans_h])
                    
                    haz_value = eval_hazard(
                        hazard, interval.ub, hazard_pars, covars;
                        apply_transform = hazard.metadata.time_transform,
                        cache_context = tt_context,
                        hazard_slot = trans_h)
                    
                    ll += log(haz_value)
                end
            end
        end
    end
    
    return ll
end
