########################################################
### Likelihod functions for sample paths, tpms, etc. ###
########################################################

### Exactly observed sample paths ----------------------
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

        ll += survprob(
            subjectdata.sojourn[i],
            subjectdata.sojourn[i] + subjectdata.increment[i],
            pars,
            subjectdata[i, :],
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
                subjectdata[i, :],
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
current model definition. Accepts either a `VectorOfVectors` or flat parameter vector and
reuses `loglik_path` for the heavy lifting.
"""
function loglik(parameters, path::SamplePath, hazards::Vector{<:_Hazard}, model::MultistateProcess)

    # normalize parameter representation for downstream hazard calls
    pars = parameters isa VectorOfVectors ? parameters : VectorOfVectors(parameters, model.parameters.elem_ptr)

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
- `SplineHazard`: Spline baseline with known integral → separable

# Future Extensions

Future hazard types may return `false`:
- `:ode` hazards with non-separable RHS
- `:ode_neural` hazards where `f(t, Λ, x)` is a neural network

When `is_separable` returns `false`, the likelihood must use ODE solvers
(via DifferentialEquations.jl) to compute cumulative hazards numerically.

# Usage Sites

This trait is checked in several locations:
- `loglik_exact_batched`: Use analytic cumhaz when separable
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
is_separable(::SplineHazard) = true

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
#    - Use: `stacked = stack_intervals_for_hazard(h, cached_paths, hazards, ...)`
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
# 5. BATCHED FUNCTIONS
#    - loglik_exact_batched: For flat Vector{SamplePath} (ExactData)
#    - loglik_semi_markov_batched!: For nested Vector{Vector{SamplePath}} (SMPanelData)
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

See also: [`stack_intervals_for_hazard`](@ref), [`loglik_exact_batched`](@ref)
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

"""
    loglik_exact_batched(parameters, data::ExactData; neg=true, return_ll_subj=false)

Batched (hazard-centric) log-likelihood computation for exact data.
Instead of iterating path-by-path and calling survprob (which loops over hazards),
this version:
1. Pre-caches all path DataFrames once upfront (eliminating redundant make_subjdat calls)
2. Pre-processes all paths to extract intervals per hazard
3. Loops over hazards in the outer loop
4. Computes all cumulative hazards for a hazard at once
5. Accumulates contributions to path-level log-likelihoods

This approach is more efficient when:
- There are many paths (MCEM with many imputed paths)
- Hazard computations can be vectorized
- Memory is not the bottleneck (compute > memory)

# Separability

For separable hazards (where `is_separable(hazard) == true`), the cumulative hazard
is computed analytically using the closed-form `cumulative_hazard` function. This
applies to all current hazard types (Markov, SemiMarkov, Spline).

Future ODE-based hazards with `is_separable(hazard) == false` will use numerical
integration via DifferentialEquations.jl.

See also: [`is_separable`](@ref), [`cache_path_data`](@ref), [`StackedHazardData`](@ref)
"""
function loglik_exact_batched(parameters, data::ExactData; neg=true, return_ll_subj=false)
    # Nest parameters
    pars = VectorOfVectors(parameters, data.model.parameters.elem_ptr)
    
    # Get hazards
    hazards = data.model.hazards
    n_hazards = length(hazards)
    n_paths = length(data.paths)
    
    # Remake spline parameters and calculate risk periods
    for i in eachindex(hazards)
        if isa(hazards[i], _SplineHazard)
            remake_splines!(hazards[i], pars[i])
            set_riskperiod!(hazards[i])
        end
    end
    
    # Initialize path-level log-likelihoods
    ll_paths = zeros(n_paths)
    
    # Pre-cache all path DataFrames once (eliminates redundant make_subjdat calls)
    cached_paths = cache_path_data(data.paths, data.model)
    
    # Create TimeTransformContext if any hazard uses time_transform
    # Use first cached path's DataFrame to get time column type
    sample_df = isempty(cached_paths) ? nothing : first(cached_paths).df
    tt_context = maybe_time_transform_context(pars, sample_df, hazards)
    
    # Pre-process intervals for each hazard using cached paths
    # Pass pars to pre-compute linear predictors
    stacked_data = Vector{StackedHazardData}(undef, n_hazards)
    for h in 1:n_hazards
        stacked_data[h] = stack_intervals_for_hazard(
            h, cached_paths, hazards, data.model.totalhazards, data.model.tmat;
            pars=pars)
    end
    
    # Outer loop over hazards
    for h in 1:n_hazards
        sd = stacked_data[h]
        n_intervals = length(sd.lb)
        
        if n_intervals == 0
            continue
        end
        
        hazard = hazards[h]
        hazard_pars = pars[h]
        use_transform = hazard.metadata.time_transform
        
        # For separable hazards, use analytic cumulative hazard
        # Future ODE hazards with is_separable(hazard) == false will branch here
        
        # Compute cumulative hazards and hazard values for this hazard
        for i in 1:n_intervals
            # Cumulative hazard contribution (survival)
            log_cumhaz = call_cumulhaz(
                sd.lb[i], sd.ub[i], hazard_pars, sd.covars[i], hazard;
                give_log = true,
                apply_transform = use_transform,
                cache_context = tt_context,
                hazard_slot = h)
            
            # Subtract from log-likelihood (survival = exp(-cumhaz))
            ll_paths[sd.path_idx[i]] -= exp(log_cumhaz)
            
            # Add transition hazard if applicable
            if sd.is_transition[i]
                log_haz = call_haz(
                    sd.transition_times[i], hazard_pars, sd.covars[i], hazard;
                    give_log = true,
                    apply_transform = use_transform,
                    cache_context = tt_context,
                    hazard_slot = h)
                ll_paths[sd.path_idx[i]] += log_haz
            end
        end
    end
    
    # Apply subject weights - lookup by path's subject index
    subj_weights = data.model.SubjectWeights
    path_weights = [subj_weights[cpd.subj] for cpd in cached_paths]
    
    if return_ll_subj
        return ll_paths .* path_weights
    else
        ll = sum(ll_paths .* path_weights)
        return neg ? -ll : ll
    end
end

########################################################
##################### Wrappers #########################
########################################################

"""
    loglik(parameters, data::ExactData; neg = true) 

Return sum of (negative) log likelihoods for all sample paths. Use mapreduce() to call loglik() and sum the results. Each sample path object is `path::SamplePath` and contains the subject index and the jump chain. 
"""
function loglik_exact(parameters, data::ExactData; neg=true, return_ll_subj=false)

    # nest parameters
    pars = VectorOfVectors(parameters, data.model.parameters.elem_ptr)

    # snag the hazards
    hazards = data.model.hazards

    # remake spline parameters and calculate risk periods
    for i in eachindex(hazards)
        if isa(hazards[i], _SplineHazard)
            remake_splines!(hazards[i], pars[i])
            set_riskperiod!(hazards[i])
        end
    end

    if return_ll_subj
        # send each element of samplepaths to loglik
        # Convert SamplePath to DataFrame using make_subjdat
        map((path, w) -> begin
            subj_inds = data.model.subjectindices[path.subj]
            subj_dat = view(data.model.data, subj_inds, :)
            subjdat_df = make_subjdat(path, subj_dat)
            loglik_path(pars, subjdat_df, hazards, data.model.totalhazards, data.model.tmat) * w
        end, data.paths, data.model.SubjectWeights)
    else
        # Convert SamplePath to DataFrame using make_subjdat
        ll = mapreduce((path, w) -> begin
            subj_inds = data.model.subjectindices[path.subj]
            subj_dat = view(data.model.data, subj_inds, :)
            subjdat_df = make_subjdat(path, subj_dat)
            loglik_path(pars, subjdat_df, hazards, data.model.totalhazards, data.model.tmat) * w
        end, +, data.paths, data.model.SubjectWeights)
        neg ? -ll : ll
    end
end

"""
    loglik(parameters, data::ExactDataAD; neg = true) 

Return sum of (negative) log likelihoods for all sample paths. Use mapreduce() to call loglik() and sum the results. Each sample path object is `path::SamplePath` and contains the subject index and the jump chain. NOTE: Why is this different from loglik_exact?
"""
function loglik_AD(parameters, data::ExactDataAD; neg = true)

    # nest parameters
    pars = VectorOfVectors(parameters, data.model.parameters.elem_ptr)

    # snag the hazards
    hazards = data.model.hazards

    # remake spline parameters and calculate risk periods
    for i in eachindex(hazards)
        if isa(hazards[i], _SplineHazard)
            remake_splines!(hazards[i], pars[i])
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

    # nest the model parameters
    pars = VectorOfVectors(parameters, data.model.parameters.elem_ptr)

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

    # accumulate the log likelihood
    ll = 0.0

    # container for subject-level loglikelihood
    if return_ll_subj
        ll_subj = zeros(Float64, nsubj)
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

        # no state is censored
        if all(data.model.data.obstype[subj_inds] .∈ Ref([1,2]))
            
            # subject contribution to the loglikelihood
            subj_ll = 0.0

            # add the contribution of each observation
            for i in subj_inds
                # get observation weight (default to 1.0)
                obs_weight = has_obs_weights ? data.model.ObservationWeights[i] : 1.0
                
                # get data row for covariate access
                row_data = data.model.data[i, :]
                
                if data.model.data.obstype[i] == 1 # exact data
                    
                    obs_ll = survprob(
                        0,
                        data.model.data.tstop[i] - data.model.data.tstart[i],
                        pars,
                        row_data,
                        data.model.totalhazards[data.model.data.statefrom[i]],
                        data.model.hazards;
                        give_log = true)
                                        
                    if data.model.data.statefrom[i] != data.model.data.stateto[i] # if there is a transition, add log hazard
                        obs_ll += call_haz(
                            data.model.data.tstop[i] - data.model.data.tstart[i],
                            pars[data.model.tmat[data.model.data.statefrom[i], data.model.data.stateto[i]]],
                            row_data,
                            data.model.hazards[data.model.tmat[data.model.data.statefrom[i], data.model.data.stateto[i]]];
                            give_log = true)
                    end
                    
                    subj_ll += obs_ll * obs_weight

                else # panel data
                    subj_ll += log(tpm_book[data.books[2][i, 1]][data.books[2][i, 2]][data.model.data.statefrom[i], data.model.data.stateto[i]]) * obs_weight
                end
            end

        else
            # Forward algorithm for censored observations
            # NOTE: ObservationWeights are currently not supported for censored observations
            # (forward algorithm). Use SubjectWeights instead for weighted estimation.
            if has_obs_weights && any(data.model.ObservationWeights[subj_inds] .!= 1.0)
                @warn "ObservationWeights are not supported for censored observations (obstype > 2). Using unweighted likelihood for subject $subj."
            end
            
            # initialize likelihood matrix
            lmat = zeros(eltype(parameters), S, length(subj_inds) + 1)
            lmat[data.model.data.statefrom[subj_inds[1]], 1] = 1

            # initialize counter for likelihood matrix
            ind = 1

            # update the vector l
            for i in subj_inds

                # increment counter for likelihood matrix
                ind += 1

                # compute q, the transition probability matrix
                if data.model.data.obstype[i] != 1
                    # if panel data, simply grab q from tpm_book
                    copyto!(q, tpm_book[data.books[2][i, 1]][data.books[2][i, 2]])
                    
                else
                    # if exact data (obstype = 1), compute q by hand
                    # reset Q
                    fill!(q, -Inf)
                    
                    # compute q(r,s)
                    for r in 1:S
                        if isa(data.model.totalhazards[r], _TotalHazardAbsorbing)
                            q[r,r] = 0.0
                        else
                            # survival probability
                            q[r, findall(data.model.tmat[r,:] .!= 0)] .= survprob(
                                0,
                                data.model.data.tstop[i] - data.model.data.tstart[i],
                                pars,
                                i,
                                data.model.totalhazards[r],
                                data.model.hazards;
                                give_log = true) 
                            
                            # hazard
                            for s in 1:S
                                if (s != r) & (data.model.tmat[r,s] != 0)
                                    q[r, s] += call_haz(
                                        data.model.data.tstop[i] - data.model.data.tstart[i],
                                        pars[data.model.tmat[r, s]],
                                        i,
                                        data.model.hazards[data.model.tmat[r, s]];
                                        give_log = true)
                                end
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
                if data.model.data.stateto[i] > 0
                    # Exact observation - only one state possible
                    for r in 1:S
                        lmat[data.model.data.stateto[i], ind] += q[r, data.model.data.stateto[i]] * lmat[r, ind - 1]
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
"""
function loglik_semi_markov(parameters, data::SMPanelData; neg = true, use_sampling_weight = true)

    # nest the model parameters
    pars = VectorOfVectors(parameters, data.model.parameters.elem_ptr)

    # snag the hazards
    hazards = data.model.hazards

    # remake spline parameters and calculate risk periods
    for i in eachindex(hazards)
        if isa(hazards[i], _SplineHazard)
            remake_splines!(hazards[i], pars[i])
            set_riskperiod!(hazards[i])
        end
    end

    # compute the semi-markov log-likelihoods
    ll = 0.0
    for i in eachindex(data.paths)
        lls = 0.0
        for j in eachindex(data.paths[i])
            # mlm: function Q in the EM
            # Convert SamplePath to DataFrame using make_subjdat
            path = data.paths[i][j]
            subj_inds = data.model.subjectindices[path.subj]
            subj_dat = view(data.model.data, subj_inds, :)
            subjdat_df = make_subjdat(path, subj_dat)
            lls += loglik_path(pars, subjdat_df, hazards, data.model.totalhazards, data.model.tmat) * data.ImportanceWeights[i][j]
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
"""
function loglik_semi_markov!(parameters, logliks::Vector{}, data::SMPanelData)

    # nest the model parameters
    pars = VectorOfVectors(parameters, data.model.parameters.elem_ptr)

    # snag the hazards
    hazards = data.model.hazards

    # remake spline parameters and calculate risk periods
    for i in eachindex(hazards)
        if isa(hazards[i], _SplineHazard)
            remake_splines!(hazards[i], pars[i])
            set_riskperiod!(hazards[i])
        end
    end

    for i in eachindex(data.paths)
        for j in eachindex(data.paths[i])
            # Convert SamplePath to DataFrame using make_subjdat
            path = data.paths[i][j]
            subj_inds = data.model.subjectindices[path.subj]
            subj_dat = view(data.model.data, subj_inds, :)
            subjdat_df = make_subjdat(path, subj_dat)
            logliks[i][j] = loglik_path(pars, subjdat_df, hazards, data.model.totalhazards, data.model.tmat)
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
    # Nest parameters
    pars = VectorOfVectors(parameters, data.model.parameters.elem_ptr)
    
    # Get hazards
    hazards = data.model.hazards
    n_hazards = length(hazards)
    
    # Remake spline parameters and calculate risk periods
    for i in eachindex(hazards)
        if isa(hazards[i], _SplineHazard)
            remake_splines!(hazards[i], pars[i])
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
            # Cumulative hazard contribution
            log_cumhaz = call_cumulhaz(
                sd.lb[i], sd.ub[i], hazard_pars, sd.covars[i], hazard;
                give_log = true,
                apply_transform = use_transform,
                cache_context = tt_context,
                hazard_slot = h)
            
            ll_flat[sd.path_idx[i]] -= exp(log_cumhaz)
            
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
