# ============================================================================
# Simulation Module
# ============================================================================
# This module provides functions for simulating sample paths from multistate
# models. The main entry points are:
#   - simulate(): unified wrapper that can return data, paths, or both
#   - simulate_data(): convenience function for simulated datasets only
#   - simulate_paths(): convenience function for continuous paths only
#   - simulate_path(): simulates a single subject's sample path
#   - observe_path(): converts continuous path to discrete observations
#
# The simulation engine supports two strategies for locating jump times:
#   - CachedTransformStrategy: uses precomputed time transformations (faster)
#   - DirectTransformStrategy: computes cumulative hazards directly (baseline)
#
# Jump time root-finding uses NonlinearSolve.jl's ITP bracketing method.
# ============================================================================

using Random
using NonlinearSolve: IntervalNonlinearProblem, solve, ITP

# ----------------------------------------------------------------------------
# Type Definitions for Simulation Strategies
# ----------------------------------------------------------------------------

"""
    AbstractTransformStrategy

Abstract type for controlling how the simulator computes cumulative hazards
when locating jump times.
"""
abstract type AbstractTransformStrategy end

"""
    CachedTransformStrategy <: AbstractTransformStrategy

Uses precomputed time transformation caches for hazards that support them.
This is the default and recommended strategy for models with expensive
cumulative hazard computations (e.g., spline hazards).
"""
struct CachedTransformStrategy <: AbstractTransformStrategy end

"""
    DirectTransformStrategy <: AbstractTransformStrategy

Always computes cumulative hazards directly without caching.
Useful for regression testing or debugging.
"""
struct DirectTransformStrategy <: AbstractTransformStrategy end

# Legacy aliases for backward compatibility


# Strategy dispatch helpers
"""
    _use_time_transform(::CachedTransformStrategy) -> true
    _use_time_transform(::DirectTransformStrategy) -> false

Returns whether to use cached time transformations based on strategy type.
"""
_use_time_transform(::CachedTransformStrategy) = true
_use_time_transform(::DirectTransformStrategy) = false

"""
    AbstractJumpSolver

Abstract type for jump time root-finding algorithms.
"""
abstract type AbstractJumpSolver end

"""
    OptimJumpSolver <: AbstractJumpSolver

Uses Optim.jl's Brent method to locate jump times. This is the default solver.
Brent's method is robust and efficient for bounded univariate optimization.

# Fields
- `rel_tol::Float64`: Relative tolerance (default: sqrt(sqrt(eps())))
- `abs_tol::Float64`: Absolute tolerance (default: sqrt(eps()))
"""
struct OptimJumpSolver <: AbstractJumpSolver
    rel_tol::Float64
    abs_tol::Float64
    OptimJumpSolver(; rel_tol::Float64 = sqrt(sqrt(eps())), abs_tol::Float64 = sqrt(eps())) = new(rel_tol, abs_tol)
end

"""
    ExponentialJumpSolver <: AbstractJumpSolver

Uses closed-form inversion for exponential (Markov) hazards.

For states where all outgoing transitions have constant (time-invariant) hazard rates,
the jump time can be computed analytically without root-finding:

    t = -log(1 - (u - cuminc)/(1 - cuminc)) / total_rate

This solver is automatically selected when the simulation detects that all
hazards from the current state are Markov (exponential).

Falls back to OptimJumpSolver for non-exponential hazards.
"""
struct ExponentialJumpSolver <: AbstractJumpSolver
    fallback::OptimJumpSolver
    ExponentialJumpSolver(; fallback::OptimJumpSolver = OptimJumpSolver()) = new(fallback)
end

"""
    HybridJumpSolver <: AbstractJumpSolver

Automatically selects between exponential (closed-form) and ITP (root-finding)
based on whether the current state's hazards are all Markov (exponential).

This is the recommended solver for mixed models with both exponential and
non-exponential transitions.

# Fields  
- `exp_solver::ExponentialJumpSolver`: Solver for exponential hazards
- `itp_solver::OptimJumpSolver`: Solver for non-exponential hazards
"""
struct HybridJumpSolver <: AbstractJumpSolver
    exp_solver::ExponentialJumpSolver
    itp_solver::OptimJumpSolver
    HybridJumpSolver(; 
        exp_solver::ExponentialJumpSolver = ExponentialJumpSolver(),
        itp_solver::OptimJumpSolver = OptimJumpSolver()
    ) = new(exp_solver, itp_solver)
end

# ----------------------------------------------------------------------------
# Helper Functions for Exponential Hazard Detection
# ----------------------------------------------------------------------------

"""
    _is_state_all_markov(state::Int, totalhazard::_TotalHazardTransient, hazards::Vector{<:_Hazard}) -> Bool

Check if all outgoing hazards from a state are Markov (exponential/time-invariant).

Returns `true` if all component hazards are `_MarkovHazard` subtypes.
"""
@inline function _is_state_all_markov(totalhazard::_TotalHazardTransient, hazards::AbstractVector{<:_Hazard})
    @inbounds for idx in totalhazard.components
        hazards[idx] isa _MarkovHazard || return false
    end
    return true
end

"""
    _compute_total_markov_rate(params::Tuple, covars_cache::AbstractVector{<:NamedTuple},
                               totalhazard::_TotalHazardTransient, hazards::AbstractVector{<:_Hazard}) -> Float64

Compute the total hazard rate for a state where all hazards are Markov (exponential).

For Markov hazards, the rate is constant: h(t) = rate = baseline * exp(Xβ).
The total rate is the sum of all component hazard rates.

# Arguments
- `params::Tuple`: Indexed parameters (from `values(params_named)`)
- `covars_cache::AbstractVector{<:NamedTuple}`: Cached covariate values per hazard
- `totalhazard::_TotalHazardTransient`: Total hazard specification
- `hazards::AbstractVector{<:_Hazard}`: All hazard functions

# Returns
- `Float64`: Sum of all hazard rates from the current state
"""
@inline function _compute_total_markov_rate(params::Tuple, covars_cache::AbstractVector{<:NamedTuple},
                                            totalhazard::_TotalHazardTransient, hazards::AbstractVector{<:_Hazard})
    total_rate = 0.0
    @inbounds for idx in totalhazard.components
        hazard = hazards[idx]
        # For Markov hazards, eval at t=0 gives the constant rate
        total_rate += hazard(0.0, params[idx], covars_cache[idx])
    end
    return total_rate
end

"""
    _exponential_jump_time(u::Float64, cuminc::Float64, total_rate::Float64, interval_len::Float64) -> Float64

Compute jump time analytically for exponential hazards.

Solves: u = cuminc + (1 - cuminc) * (1 - exp(-total_rate * t))

# Arguments
- `u`: Uniform random variate (cumulative incidence target)
- `cuminc`: Current cumulative incidence
- `total_rate`: Sum of all hazard rates from current state
- `interval_len`: Maximum time in interval (upper bound)

# Returns
- `Float64`: Jump time (clamped to interval_len)
"""
@inline function _exponential_jump_time(u::Float64, cuminc::Float64, total_rate::Float64, interval_len::Float64)
    # Solve: u = cuminc + (1 - cuminc) * (1 - exp(-total_rate * t))
    # => (u - cuminc) / (1 - cuminc) = 1 - exp(-total_rate * t)
    # => exp(-total_rate * t) = 1 - (u - cuminc) / (1 - cuminc)
    # => t = -log(1 - (u - cuminc) / (1 - cuminc)) / total_rate
    
    remaining_survival = 1.0 - cuminc
    if remaining_survival <= _DELTA_U || total_rate <= 0.0
        return interval_len  # Edge case: return boundary
    end
    
    prob_increment = (u - cuminc) / remaining_survival
    if prob_increment >= 1.0 - _DELTA_U
        return interval_len  # Event at or beyond boundary
    end
    
    t = -log(1.0 - prob_increment) / total_rate
    return min(t, interval_len)  # Clamp to interval
end

# ----------------------------------------------------------------------------
# Helper Functions for Simulation Data Preparation
# ----------------------------------------------------------------------------

"""
    _prepare_simulation_data(model, newdata, tmax, autotmax)

Prepare data and subject indices for simulation based on user options.

Priority: newdata > tmax > autotmax > model.data as-is

# Returns
- `sim_data::DataFrame`: The data to use for simulation
- `sim_subjinds::Vector{Vector{Int64}}`: Subject indices for the simulation data
- `restore_needed::Bool`: Whether model.data was modified (needs restoration)
- `original_data::Union{Nothing,DataFrame}`: Original data if restoration needed
- `original_subjinds::Union{Nothing,Vector{Vector{Int64}}}`: Original indices if needed
"""
function _prepare_simulation_data(model::MultistateProcess, 
                                   newdata::Union{Nothing,DataFrame}, 
                                   tmax::Union{Nothing,Float64},
                                   autotmax::Bool)
    
    if !isnothing(newdata)
        # Validate newdata has same columns as model.data
        model_cols = Set(names(model.data))
        new_cols = Set(names(newdata))
        if model_cols != new_cols
            missing_cols = setdiff(model_cols, new_cols)
            extra_cols = setdiff(new_cols, model_cols)
            err_parts = String[]
            if !isempty(missing_cols)
                push!(err_parts, "missing columns: $(collect(missing_cols))")
            end
            if !isempty(extra_cols)
                push!(err_parts, "extra columns: $(collect(extra_cols))")
            end
            throw(ArgumentError("newdata columns must match model.data columns. " * join(err_parts, "; ")))
        end
        
        # Compute subject indices for newdata
        sim_subjinds, _ = get_subjinds(newdata)
        
        # Store original and swap
        original_data = model.data
        original_subjinds = model.subjectindices
        model.data = newdata
        model.subjectindices = sim_subjinds
        
        return newdata, sim_subjinds, true, original_data, original_subjinds
        
    elseif !isnothing(tmax)
        # Create a copy of the data with tstop set to tmax
        tmax > 0 || throw(ArgumentError("tmax must be positive, got $tmax"))
        
        sim_data = copy(model.data)
        sim_data.tstop .= tmax
        
        # Collapse to one row per subject (taking first row's covariates)
        sim_data = _collapse_to_single_interval(sim_data)
        sim_subjinds, _ = get_subjinds(sim_data)
        
        # Store original and swap
        original_data = model.data
        original_subjinds = model.subjectindices
        model.data = sim_data
        model.subjectindices = sim_subjinds
        
        return sim_data, sim_subjinds, true, original_data, original_subjinds
        
    elseif autotmax
        # Use maximum tstop from the data as implicit tmax
        implicit_tmax = maximum(model.data.tstop)
        
        sim_data = copy(model.data)
        sim_data.tstop .= implicit_tmax
        
        # Collapse to one row per subject
        sim_data = _collapse_to_single_interval(sim_data)
        sim_subjinds, _ = get_subjinds(sim_data)
        
        # Store original and swap
        original_data = model.data
        original_subjinds = model.subjectindices
        model.data = sim_data
        model.subjectindices = sim_subjinds
        
        return sim_data, sim_subjinds, true, original_data, original_subjinds
    else
        # Use model data as-is
        return model.data, model.subjectindices, false, nothing, nothing
    end
end

"""
    _collapse_to_single_interval(data::DataFrame)

Collapse multi-row per subject data to a single row per subject.
Takes the first row's covariates and statefrom, sets tstart=0.
"""
function _collapse_to_single_interval(data::DataFrame)
    # Get unique subjects in order
    ids = unique(data.id)
    
    # For each subject, take the first row and modify tstart
    collapsed_rows = DataFrame[]
    for id in ids
        subj_data = data[data.id .== id, :]
        first_row = DataFrame(subj_data[1:1, :])
        first_row.tstart .= 0.0
        # tstop is already set by caller
        push!(collapsed_rows, first_row)
    end
    
    return reduce(vcat, collapsed_rows)
end

"""
    _restore_model_data!(model, original_data, original_subjinds)

Restore the model's original data and subject indices after simulation.
"""
function _restore_model_data!(model::MultistateProcess, 
                               original_data::DataFrame, 
                               original_subjinds::Vector{Vector{Int64}})
    model.data = original_data
    model.subjectindices = original_subjinds
    return nothing
end

# ----------------------------------------------------------------------------
# Main Simulation Functions  
# ----------------------------------------------------------------------------

"""
    simulate(model::MultistateProcess; nsim = 1, data = true, paths = false,
             strategy = CachedTransformStrategy(), solver = OptimJumpSolver(),
             newdata = nothing, tmax = nothing, autotmax = true)

Simulate datasets and/or continuous-time sample paths from a multistate model.

This is the main simulation entry point. By default it returns discretely
observed data (subject to the model's observation scheme). Set `paths = true`
to also receive the underlying continuous-time sample paths.

# Arguments
- `model::MultistateProcess`: multistate model object created by `multistatemodel()`
- `nsim::Int`: number of simulations to run (default: 1)
- `data::Bool`: return discretely observed datasets (default: true)
- `paths::Bool`: return continuous-time sample paths (default: false)
- `strategy::AbstractTransformStrategy`: controls cumulative hazard computation
  - `CachedTransformStrategy()` (default): uses cached time transformations when available
  - `DirectTransformStrategy()`: always computes cumulative hazards directly
- `solver::AbstractJumpSolver`: controls jump time root-finding algorithm
  - `OptimJumpSolver(; rel_tol, abs_tol)` (default): Optim.jl's Brent method
- `newdata::Union{Nothing,DataFrame}`: Optional new data template for simulation.
  Must have the same columns as the original model data. Supersedes `tmax` and `autotmax`.
- `tmax::Union{Nothing,Float64}`: Optional maximum simulation time. When provided,
  all subjects are simulated over `[0, tmax]` using the first row's covariates.
  Supersedes `autotmax` but is superseded by `newdata`.
- `autotmax::Bool`: If `true` (default) and neither `newdata` nor `tmax` is provided,
  uses `maximum(data.tstop)` as the implicit tmax so all subjects have the same
  observation window. Set to `false` to use each subject's original observation times.
- `expanded::Bool`: For phase-type models, controls whether results use the expanded
  (phase-level) state space (`true`) or the original observed state space (`false`).
  Default is `false`. Has no effect for non-phase-type models.

# Returns
Depends on `data` and `paths` arguments:
- `data=true, paths=false`: `Vector{DataFrame}` of simulated datasets
- `data=false, paths=true`: `Vector{SamplePath}` of continuous-time paths
- `data=true, paths=true`: tuple of (datasets, paths)

# Example
```julia
model = multistatemodel(...)

# Get only observed data (default - uses autotmax=true)
datasets = simulate(model; nsim = 100)

# Simulate all subjects to time 10.0
datasets = simulate(model; nsim = 100, tmax = 10.0)

# Use subject-specific observation windows from the original data
datasets = simulate(model; nsim = 100, autotmax = false)

# Simulate using a new data template
new_template = DataFrame(id=1:500, tstart=0.0, tstop=20.0, ...)
datasets = simulate(model; nsim = 100, newdata = new_template)

# Get both data and underlying paths
data, paths = simulate(model; nsim = 100, data = true, paths = true)

# Get only continuous paths
paths = simulate(model; nsim = 100, data = false, paths = true)

# Use direct computation (no caching) for debugging
datasets = simulate(model; nsim = 10, strategy = DirectTransformStrategy())

# Use Optim.jl solver with custom tolerances
datasets = simulate(model; nsim = 10, solver = OptimJumpSolver(rel_tol = 1e-6))

# Per-transition observation types (some exact, some panel, some censored)
print_transition_map(model)  # See transition indices
obstype_map = Dict(1 => 1, 2 => 2, 3 => 3)  # trans 1 exact, 2 panel, 3 censored
datasets = simulate(model; nsim = 100, obstype_by_transition = obstype_map)
```

# Per-Transition Observation Types

Control observation type on a per-transition basis using `obstype_by_transition`:

```julia
print_transition_map(model)
# Output:
# Index | From → To
# ------|----------
#     1 | 1 → 2
#     2 | 1 → 3
#     3 | 2 → 3

obstype_map = Dict(1 => 1, 2 => 2, 3 => 3)
datasets = simulate(model; nsim=100, obstype_by_transition=obstype_map)
```

Observation type codes:
- `1`: Exact (transition time and states fully observed)
- `2`: Panel (only endpoint state at interval boundary observed)
- `3+`: Censored (endpoint state unknown/missing)

When transitions with different observation types occur in the same interval:
- Exact transitions emit rows with true transition times
- Non-exact transitions contribute to one interval-level row
- Interval-level row has `obstype = max(codes)` among non-exact transitions

See also: [`simulate_data`](@ref), [`simulate_paths`](@ref), [`simulate_path`](@ref),
[`enumerate_transitions`](@ref), [`print_transition_map`](@ref)
"""
function simulate(model::MultistateProcess; nsim = 1, data = true, paths = false, 
                  strategy::AbstractTransformStrategy = CachedTransformStrategy(),
                  solver::AbstractJumpSolver = OptimJumpSolver(),
                  newdata::Union{Nothing,DataFrame} = nothing,
                  tmax::Union{Nothing,Float64} = nothing,
                  autotmax::Bool = true,
                  expanded::Bool = false,
                  obstype_by_transition::Union{Nothing,Dict{Int,Int}} = nothing)

    # throw an error if neither paths nor data are asked for
    if paths == false && data == false
        throw(ArgumentError("Neither paths nor data requested. Set at least one of `paths=true` or `data=true`."))
    end

    # Validate per-transition observation parameters
    if obstype_by_transition !== nothing
        tmat = model.tmat
        trans_map = transition_index_map(tmat)
        validate_obstype_by_transition(obstype_by_transition, tmat)
    else
        trans_map = nothing
    end

    # Prepare simulation data (handles newdata, tmax, autotmax priority)
    sim_data, sim_subjinds, restore_needed, original_data, original_subjinds = 
        _prepare_simulation_data(model, newdata, tmax, autotmax)
    
    try
        # number of subjects (from prepared data)
        nsubj = length(sim_subjinds)

        # initialize array for simulated paths 
        if paths == true
            samplepaths = Array{SamplePath}(undef, nsubj, nsim)
        end

        # initialize container for simulated data  
        if data == true
            datasets = Array{DataFrame}(undef, nsubj, nsim)
        end 

        for i in Base.OneTo(nsim)
            for j in Base.OneTo(nsubj)
                
                # simulate a path for subject j
                samplepath = simulate_path(model, j; strategy = strategy, solver = solver)

                # save path if requested
                if paths == true
                    samplepaths[j, i] = samplepath
                end

                # observe path 
                if data == true
                    datasets[j, i] = observe_path(samplepath, model;
                                                   obstype_by_transition = obstype_by_transition,
                                                   trans_map = trans_map)
                end
            end
        end

        # vertically concatenate datasets into Vector{DataFrame}
        if data == true
            dat = [reduce(vcat, @view(datasets[:, i])) for i in 1:nsim]
        end

        # collect subject paths into Vector{Vector{SamplePath}}
        if paths == true
            trajectories = [collect(@view(samplepaths[:, i])) for i in 1:nsim]
        end

        # Collapse results if expanded=false and model is phase-type
        if !expanded && has_phasetype_expansion(model)
            mappings = model.phasetype_expansion.mappings
            if data == true
                dat = [_collapse_data(df, mappings) for df in dat]
            end
            if paths == true
                trajectories = [[_collapse_path(p, mappings) for p in pset] for pset in trajectories]
            end
        end

        # return paths and data
        if paths == false && data == true
            return dat
        elseif paths == true && data == true
            return dat, trajectories
        elseif paths == true && data == false
            return trajectories
        else
            @assert false "Internal error: unexpected combination of data=$data, paths=$paths"
        end
    finally
        # Restore original model data if we modified it
        if restore_needed
            _restore_model_data!(model, original_data, original_subjinds)
        end
    end
end

"""
    simulate_data(model::MultistateProcess; nsim = 1, 
                  strategy = CachedTransformStrategy(),
                  solver = OptimJumpSolver(), newdata = nothing, tmax = nothing, 
                  autotmax = true)

Simulate discretely observed datasets from a multistate model.

This is a convenience wrapper around `simulate()` that returns only the
observed data (no continuous-time paths). Equivalent to calling
`simulate(model; nsim=nsim, data=true, paths=false, ...)`.

# Arguments
- `model::MultistateProcess`: multistate model object created by `multistatemodel()`
- `nsim::Int`: number of datasets to simulate (default: 1)
- `strategy::AbstractTransformStrategy`: controls cumulative hazard computation (default: `CachedTransformStrategy()`)
- `solver::AbstractJumpSolver`: controls jump time root-finding (default: `OptimJumpSolver()`)
- `newdata::Union{Nothing,DataFrame}`: Optional new data template for simulation (default: nothing)
- `tmax::Union{Nothing,Float64}`: Optional maximum simulation time (default: nothing)
- `autotmax::Bool`: Use maximum tstop as implicit tmax (default: true)
- `expanded::Bool`: For phase-type models, use expanded state space (default: false)
- `obstype_by_transition::Union{Nothing,Dict{Int,Int}}`: Per-transition observation types (default: nothing)

# Returns
- `Vector{DataFrame}`: array of simulated datasets with dimensions (1, nsim)

# Example
```julia
model = multistatemodel(...)
datasets = simulate_data(model; nsim = 100)
datasets = simulate_data(model; nsim = 100, tmax = 15.0)

# Per-transition observation types
print_transition_map(model)  # See transition indices
obstype_map = Dict(1 => 1, 2 => 2)  # transition 1 exact, transition 2 panel
datasets = simulate_data(model; nsim = 100, obstype_by_transition = obstype_map)
```

See also: [`simulate`](@ref), [`simulate_paths`](@ref)
"""
function simulate_data(model::MultistateProcess;
                       nsim::Int = 1,
                       strategy::AbstractTransformStrategy = CachedTransformStrategy(),
                       solver::AbstractJumpSolver = OptimJumpSolver(),
                       newdata::Union{Nothing,DataFrame} = nothing,
                       tmax::Union{Nothing,Float64} = nothing,
                       autotmax::Bool = true,
                       expanded::Bool = false,
                       obstype_by_transition::Union{Nothing,Dict{Int,Int}} = nothing)
    return simulate(model; nsim = nsim, data = true, paths = false,
                    strategy = strategy, solver = solver,
                    newdata = newdata, tmax = tmax, autotmax = autotmax,
                    expanded = expanded,
                    obstype_by_transition = obstype_by_transition)
end

"""
    simulate_paths(model::MultistateProcess; nsim = 1, 
                   strategy = CachedTransformStrategy(),
                   solver = OptimJumpSolver(), newdata = nothing, tmax = nothing,
                   autotmax = true)

Simulate continuous-time sample paths from a multistate model.

This is a convenience wrapper around `simulate()` that returns only the
continuous-time sample paths (no discretely observed data). Equivalent to
calling `simulate(model; nsim=nsim, data=false, paths=true, ...)`.

# Arguments
- `model::MultistateProcess`: multistate model object created by `multistatemodel()`
- `nsim::Int`: number of path collections to simulate (default: 1)
- `strategy::AbstractTransformStrategy`: controls cumulative hazard computation (default: `CachedTransformStrategy()`)
- `solver::AbstractJumpSolver`: controls jump time root-finding (default: `OptimJumpSolver()`)
- `newdata::Union{Nothing,DataFrame}`: Optional new data template for simulation (default: nothing)
- `tmax::Union{Nothing,Float64}`: Optional maximum simulation time (default: nothing)
- `autotmax::Bool`: Use maximum tstop as implicit tmax (default: true)
- `expanded::Bool`: For phase-type models, use expanded state space (default: false)

# Returns
- `Matrix{SamplePath}`: array of sample paths with dimensions (nsubj, nsim)

# Example
```julia
model = multistatemodel(...)
paths = simulate_paths(model; nsim = 100)
paths = simulate_paths(model; nsim = 100, tmax = 15.0)
```

See also: [`simulate`](@ref), [`simulate_data`](@ref), [`simulate_path`](@ref)
"""
function simulate_paths(model::MultistateProcess;
                        nsim::Int = 1,
                        strategy::AbstractTransformStrategy = CachedTransformStrategy(),
                        solver::AbstractJumpSolver = OptimJumpSolver(),
                        newdata::Union{Nothing,DataFrame} = nothing,
                        tmax::Union{Nothing,Float64} = nothing,
                        autotmax::Bool = true,
                        expanded::Bool = false)
    return simulate(model; nsim = nsim, data = false, paths = true,
                    strategy = strategy, solver = solver,
                    newdata = newdata, tmax = tmax, autotmax = autotmax,
                    expanded = expanded)
end

"""
    _find_jump_time(solver::AbstractJumpSolver, gap_fn, lo, hi)

Dispatch to the appropriate jump time solver based on solver type.
"""
function _find_jump_time end

"""
    _find_jump_time(solver::OptimJumpSolver, gap_fn, lo, hi)

Find jump time using NonlinearSolve.jl's ITP bracketing method.

The ITP (Interpolate, Truncate, Project) algorithm is provably optimal for
worst-case performance and near-optimal for average-case, making it ideal
for root-finding in simulation where robustness and speed both matter.
"""
function _find_jump_time(solver::OptimJumpSolver, gap_fn, lo, hi)
    fa, fb = gap_fn(lo), gap_fn(hi)
    
    # Check for sign change (root exists in interval)
    if fb > 0 && fa < 0
        # Use NonlinearSolve's ITP bracketing solver
        prob = IntervalNonlinearProblem((u, p) -> gap_fn(u), (lo, hi))
        sol = solve(prob, ITP(); abstol=solver.abs_tol)
        return sol.u
    elseif fb <= 0
        # No root in interval - event happens at or after boundary
        return hi
    else
        # Both positive - shouldn't happen, return lo
        return lo
    end
end

"""
    _find_jump_time_exponential(u, cuminc, total_rate, interval_len)

Closed-form jump time computation for exponential hazards.

This is the core computation, called when we know all hazards are Markov.
"""
@inline function _find_jump_time_exponential(u::Float64, cuminc::Float64, total_rate::Float64, interval_len::Float64)
    return _exponential_jump_time(u, cuminc, total_rate, interval_len)
end

"""
    _materialize_covariates(row::DataFrameRow, hazards::AbstractVector{<:_Hazard})

Extract and cache covariate values for all hazards from a single data row.

This avoids repeated covariate extraction during the inner simulation loop.
Uses pre-computed covar_names from each hazard struct to avoid regex parsing.
Returns a vector of NamedTuples, one per hazard, containing the covariate
values needed for that hazard's computation.

# Arguments
- `row::DataFrameRow`: current observation row from subject's data
- `hazards::AbstractVector{<:_Hazard}`: model hazard specifications

# Returns
- `Vector{NamedTuple}`: cached covariate values indexed by hazard
"""
@inline function _materialize_covariates(row::DataFrameRow, hazards::AbstractVector{<:_Hazard})
    cache = Vector{NamedTuple}(undef, length(hazards))
    for (i, hazard) in enumerate(hazards)
        cache[i] = extract_covariates_fast(row, hazard.covar_names)
    end
    return cache
end

"""
    _sample_next_state(rng::AbstractRNG, probs::AbstractVector{Float64},
                       trans_inds::AbstractVector{Int})

Sample the next state from the conditional transition probability distribution.

Given the transition probabilities and the indices of reachable states from
the current state, samples one of those states using inverse-CDF sampling.

# Arguments
- `rng::AbstractRNG`: random number generator
- `probs::AbstractVector{Float64}`: transition probabilities (full state vector)
- `trans_inds::AbstractVector{Int}`: indices of states reachable from current state

# Returns
- `Int`: index of the sampled next state
"""
@inline function _sample_next_state(rng::AbstractRNG, probs::AbstractVector{Float64}, trans_inds::AbstractVector{Int})
    threshold = rand(rng)
    cumulative = 0.0
    for idx in trans_inds
        cumulative += probs[idx]
        if threshold <= cumulative
            return idx
        end
    end
    return trans_inds[end]
end

# Minimum cumulative incidence increment for numerical stability
const _DELTA_U = sqrt(eps())

"""
    simulate_path(model::MultistateProcess, subj::Int64;
                  strategy = CachedTransformStrategy(), solver = OptimJumpSolver(),
                  rng = Random.default_rng())

Simulate a single sample path for one subject.

This is the core simulation engine. It generates a continuous-time sample path
by repeatedly:
1. Sampling a cumulative incidence threshold u ~ Uniform(0,1)
2. Finding the time t where the cumulative incidence equals u
3. Sampling the next state from the conditional transition probabilities
4. Repeating until an absorbing state is reached or censoring occurs

# Arguments
- `model::MultistateProcess`: multistate model object
- `subj::Int64`: subject index (1-based)
- `strategy::AbstractTransformStrategy`: controls cumulative hazard computation
  - `CachedTransformStrategy()` (default): uses cached time transformations
  - `DirectTransformStrategy()`: computes cumulative hazards directly
- `solver::AbstractJumpSolver`: controls jump time root-finding
  - `OptimJumpSolver(; rel_tol, abs_tol)` (default): Optim.jl's Brent method
- `rng::AbstractRNG`: random number generator (default: task-local RNG)

# Returns
- `SamplePath`: struct containing subject ID, jump times, and state sequence

See also: [`simulate`](@ref), [`simulate_paths`](@ref)
"""
function simulate_path(model::MultistateProcess, subj::Int64;
                       strategy::AbstractTransformStrategy = CachedTransformStrategy(),
                       solver::AbstractJumpSolver = OptimJumpSolver(),
                       rng::AbstractRNG = Random.default_rng(),
                       expanded::Bool = true)

    # Validate inputs
    1 <= subj <= length(model.subjectindices) || 
        throw(ArgumentError("Subject index $subj out of range [1, $(length(model.subjectindices))]"))

    # Determine whether to use time transform caching based on strategy
    time_transform = _use_time_transform(strategy)

    # subject data
    subj_inds = model.subjectindices[subj]
    subj_dat  = view(model.data, subj_inds, :)

    # Get natural-scale parameters for hazard functions (family-aware)
    # Use indexed tuple for fast access in hot loops
    params_named = get_hazard_params(model.parameters, model.hazards)
    params = values(params_named)  # Convert to Tuple for indexed access

    # current index
    row = 1 # row in subject's data that is incremented
    ind = subj_inds[row] # index in complete dataset
    subjdat_row = subj_dat[row, :] # current DataFrameRow for covariate extraction
    covars_cache = _materialize_covariates(subjdat_row, model.hazards)


    # current state
    scur = subj_dat.statefrom[1]

    tt_context = time_transform ? maybe_time_transform_context(params_named, subj_dat, model.hazards; time_column = :tstop) : nothing

    # tcur and tstop
    tcur    = subj_dat.tstart[1]
    tstop   = subj_dat.tstop[1]

    # initialize time in state and cumulative incidence 
    # clock resets after each transition
    timeinstate = 0.0
    cuminc = 0.0

    # vector for next state transition probabilities
    nstates  = size(model.tmat, 2)
    ns_probs = zeros(Float64, nstates)
    transitions_by_state = [findall(model.tmat[row, :] .!= 0) for row in 1:nstates]

    # Pre-compute which states have all-Markov (exponential) outgoing hazards
    # This enables closed-form jump time computation for those states
    is_state_all_markov = Vector{Bool}(undef, nstates)
    @inbounds for s in 1:nstates
        if model.totalhazards[s] isa _TotalHazardTransient
            is_state_all_markov[s] = _is_state_all_markov(model.totalhazards[s], model.hazards)
        else
            is_state_all_markov[s] = false  # Absorbing states
        end
    end

    # initialize sample path
    times  = [tcur]; sizehint!(times, nstates * 2)
    states = [scur]; sizehint!(states, nstates * 2)

    # flag for whether to stop simulation
    # obviously don't simulate if the initial state is absorbing
    keep_going = isa(model.totalhazards[scur], _TotalHazardTransient)
    
    # sample the cumulative incidence if transient
    if keep_going
        u = max(rand(rng), _DELTA_U)
    end

    # simulate path
    while keep_going
        use_transform = time_transform && _time_transform_enabled(model.totalhazards[scur], model.hazards)
        
        # calculate event probability over the next interval
        interval_incid = (1 - cuminc) * (1 - survprob(
            timeinstate,
            timeinstate + tstop - tcur,
            params,
            covars_cache,
            model.totalhazards[scur],
            model.hazards;
            give_log = false,
            apply_transform = use_transform,
            cache_context = tt_context))

        # check if event happened in the interval
        if (u < (cuminc + interval_incid)) && (u >= cuminc)

            interval_len = tstop - tcur
            
            # Use closed-form for exponential hazards, root-finding otherwise
            if is_state_all_markov[scur]
                # Closed-form: compute total rate and use analytical formula
                total_rate = _compute_total_markov_rate(params, covars_cache, 
                                                        model.totalhazards[scur], model.hazards)
                timeincrement = _find_jump_time_exponential(u, cuminc, total_rate, interval_len)
            else
                # Non-exponential: use root-finding
                current_timeinstate = timeinstate
                gap_fn = t -> begin
                    surv = survprob(
                        current_timeinstate,
                        current_timeinstate + t,
                        params,
                        covars_cache,
                        model.totalhazards[scur],
                        model.hazards;
                        give_log = false,
                        apply_transform = use_transform,
                        cache_context = tt_context)
                    cuminc + (1 - cuminc) * (1 - surv) - u
                end
                timeincrement = _find_jump_time(solver, gap_fn, _DELTA_U, interval_len)
            end

            timeinstate += timeincrement

            # calculate next state transition probabilities 
            # next_state_probs!(ns_probs, timeinstate, scur, subjdat_row, model.parameters, model.hazards, model.totalhazards, model.tmat)

            trans_inds = transitions_by_state[scur]
            next_state_probs!(
                ns_probs,
                trans_inds,
                timeinstate,
                scur,
                covars_cache,
                params,
                model.hazards,
                model.totalhazards;
                apply_transform = use_transform,
                cache_context = tt_context)
            scur = _sample_next_state(rng, ns_probs, trans_inds)
            
            # increment time in state and cache the jump time and state
            tcur = times[end] + timeinstate
            push!(times, tcur)
            push!(states, scur)
                
            # check if the next state is transient
            keep_going = isa(model.totalhazards[scur], _TotalHazardTransient) 

            # draw new cumulative incidence, reset cuminc and time in state
            if keep_going
                u           = max(rand(rng), _DELTA_U) # sample cumulative incidence
                cuminc      = 0.0 # reset cuminc
                timeinstate = 0.0 # reset time in state
            end

        elseif u >= (cuminc + interval_incid) # no transition in interval

            # if you keep going do some bookkeeping
            if row != size(subj_dat, 1)  # no censoring

                # increment the time in state
                timeinstate += tstop - tcur

                # increment cumulative inicidence
                cuminc += interval_incid

                # increment the row indices and interval endpoints
                row  += 1
                ind  += 1
                subjdat_row = subj_dat[row, :]  # Update DataFrameRow for new interval
                covars_cache = _materialize_covariates(subjdat_row, model.hazards)
                tcur  = subj_dat.tstart[row]
                tstop = subj_dat.tstop[row]

            else # censoring, return current state and tmax
                # stop sampling
                keep_going = false

                # increment tcur 
                tcur = tstop

                # push the state and time at the right endpoint
                push!(times, tcur)
                push!(states, scur)     
            end
        end
    end

    path = SamplePath(subj, times, states)
    
    # Collapse to observed states if expanded=false and model is phase-type
    if !expanded && has_phasetype_expansion(model)
        mappings = model.phasetype_expansion.mappings
        path = _collapse_path(path, mappings)
    end
    
    return path
end


# NOTE: Phase-type model simulation methods (simulate(::PhaseTypeModel), etc.)
# have been removed as part of package streamlining.
# Phase-type hazards are now handled internally via MultistateModel with
# phasetype_expansion metadata. Standard simulate() handles expansion/collapse.

# ----------------------------------------------------------------------------
# Collapsing Functions: Expanded → Observed State Space
# ----------------------------------------------------------------------------
# These utilities are kept for internal phase-type expansion support.

"""
    _collapse_simulation_result(result, mappings, return_data, return_paths)

Collapse simulation results from expanded to observed state space.

Maps phase states back to observed states and merges consecutive intervals
in the same observed state.
"""
function _collapse_simulation_result(result, mappings, return_data::Bool, return_paths::Bool)
    if return_paths == false && return_data == true
        # result is Vector{DataFrame}
        return [_collapse_data(df, mappings) for df in result]
        
    elseif return_paths == true && return_data == true
        # result is (Vector{DataFrame}, Vector{Vector{SamplePath}})
        dat, trajectories = result
        collapsed_dat = [_collapse_data(df, mappings) for df in dat]
        collapsed_trajectories = [[_collapse_path(p, mappings) for p in paths] for paths in trajectories]
        return collapsed_dat, collapsed_trajectories
        
    elseif return_paths == true && return_data == false
        # result is Vector{Vector{SamplePath}}
        return [[_collapse_path(p, mappings) for p in paths] for paths in result]
        
    else
        @assert false "Internal error: unexpected combination of return_data=$return_data, return_paths=$return_paths"
    end
end

"""
    _collapse_path(path::SamplePath, mappings) -> SamplePath

Collapse a sample path from expanded phase-type states to observed states.

Consecutive phases in the same observed state are merged into a single interval.

# Example
Expanded path: states = [1, 2, 3, 5], times = [0, 0.5, 1.0, 1.5]
If phase_to_state = [1, 1, 2, 3, 3] (phases 1-2 are state 1, phase 3 is state 2, phases 4-5 are state 3)
Collapsed path: states = [1, 2, 3], times = [0, 1.0, 1.5]
"""
function _collapse_path(path::SamplePath, mappings)
    phase_to_state = mappings.phase_to_state
    
    # Map each state in path to observed state
    observed_states = [phase_to_state[s] for s in path.states]
    
    # Find indices where observed state changes (or first state)
    change_indices = Int[1]
    for i in 2:length(observed_states)
        if observed_states[i] != observed_states[i-1]
            push!(change_indices, i)
        end
    end
    
    # Extract times and states at change points
    collapsed_times = path.times[change_indices]
    collapsed_states = observed_states[change_indices]
    
    return SamplePath(path.subj, collapsed_times, collapsed_states)
end

"""
    _collapse_data(df::DataFrame, mappings) -> DataFrame

Collapse simulated data from expanded to observed state space.

Maps statefrom/stateto using phase_to_state mapping and merges consecutive
rows with the same observed state.
"""
function _collapse_data(df::DataFrame, mappings)
    if nrow(df) == 0
        return df
    end
    
    phase_to_state = mappings.phase_to_state
    
    # Map states to observed states, preserving missing values
    result = copy(df)
    result.statefrom = [ismissing(s) ? missing : phase_to_state[s] for s in df.statefrom]
    result.stateto = [ismissing(s) ? missing : phase_to_state[s] for s in df.stateto]
    
    # Merge consecutive rows with same observed statefrom and stateto
    # Group by subject first
    collapsed_rows = DataFrame[]
    
    for subj_df in groupby(result, :id)
        collapsed_subj = _collapse_subject_data(subj_df)
        push!(collapsed_rows, collapsed_subj)
    end
    
    return reduce(vcat, collapsed_rows)
end

"""
    _collapse_subject_data(subj_df::AbstractDataFrame) -> DataFrame

Collapse one subject's data by merging consecutive rows with same states.
"""
function _collapse_subject_data(subj_df::AbstractDataFrame)
    if nrow(subj_df) == 0
        return DataFrame(subj_df)
    end
    
    # Collect rows, merging when statefrom and stateto match
    rows = NamedTuple[]
    current_row = nothing
    
    for (i, row) in enumerate(eachrow(subj_df))
        if current_row === nothing
            current_row = (
                id = row.id,
                tstart = row.tstart,
                tstop = row.tstop,
                statefrom = row.statefrom,
                stateto = row.stateto,
                obstype = row.obstype
            )
        elseif row.statefrom == current_row.statefrom && row.stateto == current_row.stateto
            # Same transition, extend the interval
            current_row = (
                id = current_row.id,
                tstart = current_row.tstart,
                tstop = row.tstop,  # Extend to new tstop
                statefrom = current_row.statefrom,
                stateto = current_row.stateto,
                obstype = row.obstype  # Use latest obstype
            )
        else
            # Different transition, save current and start new
            push!(rows, current_row)
            current_row = (
                id = row.id,
                tstart = row.tstart,
                tstop = row.tstop,
                statefrom = row.statefrom,
                stateto = row.stateto,
                obstype = row.obstype
            )
        end
    end
    
    # Don't forget the last row
    if current_row !== nothing
        push!(rows, current_row)
    end
    
    return DataFrame(rows)
end

