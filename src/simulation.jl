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
# Jump time root-finding uses Optim.jl's Brent method (OptimJumpSolver).
# ============================================================================

using Random

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
const TangTransformStrategy = CachedTransformStrategy
const LegacyTransformStrategy = DirectTransformStrategy

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
    simulate(model::MultistateProcess; nsim = 1, data = true, paths = false,
             delta_u = sqrt(eps()), delta_t = sqrt(eps()),
             strategy = CachedTransformStrategy(), solver = OptimJumpSolver())

Simulate datasets and/or continuous-time sample paths from a multistate model.

This is the main simulation entry point. By default it returns discretely
observed data (subject to the model's observation scheme). Set `paths = true`
to also receive the underlying continuous-time sample paths.

# Arguments
- `model::MultistateProcess`: multistate model object created by `multistatemodel()`
- `nsim::Int`: number of simulations to run (default: 1)
- `data::Bool`: return discretely observed datasets (default: true)
- `paths::Bool`: return continuous-time sample paths (default: false)
- `delta_u::Float64`: minimum cumulative incidence increment per jump (default: sqrt(eps()))
- `delta_t::Float64`: minimum time increment per jump (default: sqrt(eps()))
- `strategy::AbstractTransformStrategy`: controls cumulative hazard computation
  - `CachedTransformStrategy()` (default): uses cached time transformations when available
  - `DirectTransformStrategy()`: always computes cumulative hazards directly
- `solver::AbstractJumpSolver`: controls jump time root-finding algorithm
  - `OptimJumpSolver(; rel_tol, abs_tol)` (default): Optim.jl's Brent method

# Returns
Depends on `data` and `paths` arguments:
- `data=true, paths=false`: `Vector{DataFrame}` of simulated datasets
- `data=false, paths=true`: `Vector{SamplePath}` of continuous-time paths
- `data=true, paths=true`: tuple of (datasets, paths)

# Example
```julia
model = multistatemodel(...)

# Get only observed data (default)
datasets = simulate(model; nsim = 100)

# Get both data and underlying paths
data, paths = simulate(model; nsim = 100, data = true, paths = true)

# Get only continuous paths
paths = simulate(model; nsim = 100, data = false, paths = true)

# Use direct computation (no caching) for debugging
datasets = simulate(model; nsim = 10, strategy = DirectTransformStrategy())

# Use Optim.jl solver with custom tolerances
datasets = simulate(model; nsim = 10, solver = OptimJumpSolver(rel_tol = 1e-6))
```

See also: [`simulate_data`](@ref), [`simulate_paths`](@ref), [`simulate_path`](@ref)
"""
function simulate(model::MultistateProcess; nsim = 1, data = true, paths = false, 
                  delta_u = sqrt(eps()), delta_t = sqrt(eps()), 
                  strategy::AbstractTransformStrategy = CachedTransformStrategy(),
                  solver::AbstractJumpSolver = OptimJumpSolver())

    # throw an error if neither paths nor data are asked for
    if paths == false && data == false
        error("Why are you calling `simulate` if you don't want sample paths or data? Stop wasting my time.")
    end

    # number of subjects
    nsubj = length(model.subjectindices)

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
            samplepath = simulate_path(model, j, delta_u, delta_t; strategy = strategy, solver = solver)

            # save path if requested
            if paths == true
                samplepaths[j, i] = samplepath
            end

            # observe path 
            if data == true
                datasets[j, i] = observe_path(samplepath, model)
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

    # return paths and data
    if paths == false && data == true
        return dat
    elseif paths == true && data == true
        return dat, trajectories
    elseif paths == true && data == false
        return trajectories
    else
        error("Internal error: unexpected combination of data=$data, paths=$paths")
    end
end

"""
    simulate_data(model::MultistateProcess; nsim = 1, delta_u = sqrt(eps()),
                  delta_t = sqrt(eps()), strategy = CachedTransformStrategy(),
                  solver = OptimJumpSolver())

Simulate discretely observed datasets from a multistate model.

This is a convenience wrapper around `simulate()` that returns only the
observed data (no continuous-time paths). Equivalent to calling
`simulate(model; nsim=nsim, data=true, paths=false, ...)`.

# Arguments
- `model::MultistateProcess`: multistate model object created by `multistatemodel()`
- `nsim::Int`: number of datasets to simulate (default: 1)
- `delta_u::Float64`: minimum cumulative incidence increment per jump (default: sqrt(eps()))
- `delta_t::Float64`: minimum time increment per jump (default: sqrt(eps()))
- `strategy::AbstractTransformStrategy`: controls cumulative hazard computation (default: `CachedTransformStrategy()`)
- `solver::AbstractJumpSolver`: controls jump time root-finding (default: `OptimJumpSolver()`)

# Returns
- `Vector{DataFrame}`: array of simulated datasets with dimensions (1, nsim)

# Example
```julia
model = multistatemodel(...)
datasets = simulate_data(model; nsim = 100)
```

See also: [`simulate`](@ref), [`simulate_paths`](@ref)
"""
function simulate_data(model::MultistateProcess;
                       nsim::Int = 1,
                       delta_u::Float64 = sqrt(eps()),
                       delta_t::Float64 = sqrt(eps()),
                       strategy::AbstractTransformStrategy = CachedTransformStrategy(),
                       solver::AbstractJumpSolver = OptimJumpSolver())
    return simulate(model; nsim = nsim, data = true, paths = false,
                    delta_u = delta_u, delta_t = delta_t, strategy = strategy, solver = solver)
end

"""
    simulate_paths(model::MultistateProcess; nsim = 1, delta_u = sqrt(eps()),
                   delta_t = sqrt(eps()), strategy = CachedTransformStrategy(),
                   solver = OptimJumpSolver())

Simulate continuous-time sample paths from a multistate model.

This is a convenience wrapper around `simulate()` that returns only the
continuous-time sample paths (no discretely observed data). Equivalent to
calling `simulate(model; nsim=nsim, data=false, paths=true, ...)`.

# Arguments
- `model::MultistateProcess`: multistate model object created by `multistatemodel()`
- `nsim::Int`: number of path collections to simulate (default: 1)
- `delta_u::Float64`: minimum cumulative incidence increment per jump (default: sqrt(eps()))
- `delta_t::Float64`: minimum time increment per jump (default: sqrt(eps()))
- `strategy::AbstractTransformStrategy`: controls cumulative hazard computation (default: `CachedTransformStrategy()`)
- `solver::AbstractJumpSolver`: controls jump time root-finding (default: `OptimJumpSolver()`)

# Returns
- `Matrix{SamplePath}`: array of sample paths with dimensions (nsubj, nsim)

# Example
```julia
model = multistatemodel(...)
paths = simulate_paths(model; nsim = 100)
```

See also: [`simulate`](@ref), [`simulate_data`](@ref), [`simulate_path`](@ref)
"""
function simulate_paths(model::MultistateProcess;
                        nsim::Int = 1,
                        delta_u::Float64 = sqrt(eps()),
                        delta_t::Float64 = sqrt(eps()),
                        strategy::AbstractTransformStrategy = CachedTransformStrategy(),
                        solver::AbstractJumpSolver = OptimJumpSolver())
    return simulate(model; nsim = nsim, data = false, paths = true,
                    delta_u = delta_u, delta_t = delta_t, strategy = strategy, solver = solver)
end

"""
    _find_jump_time(solver::AbstractJumpSolver, gap_fn, lo, hi, delta_t)

Dispatch to the appropriate jump time solver based on solver type.
"""
function _find_jump_time end

"""
    _find_jump_time(solver::OptimJumpSolver, gap_fn, lo, hi, delta_t)

Find jump time using Optim.jl's Brent method with parameters from solver.
"""
function _find_jump_time(solver::OptimJumpSolver, gap_fn, lo, hi, delta_t)
    # Optim.jl minimizes squared gap
    obj_fn = t -> gap_fn(t)^2
    result = Optim.optimize(obj_fn, lo, hi, Brent(); 
                            rel_tol = solver.rel_tol, abs_tol = solver.abs_tol)
    
    if Optim.converged(result)
        return result.minimizer
    else
        iters = Optim.iterations(result)
        error("simulate_path failed to locate jump time after $iters iterations on interval [$lo, $hi]")
    end
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

"""
    simulate_path(model::MultistateProcess, subj::Int64, delta_u, delta_t;
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
- `delta_u::Float64`: minimum cumulative incidence increment per jump
- `delta_t::Float64`: minimum time increment per jump
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
function simulate_path(model::MultistateProcess, subj::Int64, delta_u, delta_t;
                       strategy::AbstractTransformStrategy = CachedTransformStrategy(),
                       solver::AbstractJumpSolver = OptimJumpSolver(),
                       rng::AbstractRNG = Random.default_rng())

    # Validate inputs
    1 <= subj <= length(model.subjectindices) || 
        throw(ArgumentError("Subject index $subj out of range [1, $(length(model.subjectindices))]"))
    delta_u > 0 || throw(ArgumentError("delta_u must be positive, got $delta_u"))
    delta_t > 0 || throw(ArgumentError("delta_t must be positive, got $delta_t"))

    # Determine whether to use time transform caching based on strategy
    time_transform = _use_time_transform(strategy)

    # subject data
    subj_inds = model.subjectindices[subj]
    subj_dat  = view(model.data, subj_inds, :)

    # Get log-scale parameters for hazard functions
    params = get_log_scale_params(model.parameters)

    # current index
    row = 1 # row in subject's data that is incremented
    ind = subj_inds[row] # index in complete dataset
    subjdat_row = subj_dat[row, :] # current DataFrameRow for covariate extraction
    covars_cache = _materialize_covariates(subjdat_row, model.hazards)

    # current state
    scur = subj_dat.statefrom[1]

    tt_context = time_transform ? maybe_time_transform_context(params, subj_dat, model.hazards; time_column = :tstop) : nothing

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

    # initialize sample path
    times  = [tcur]; sizehint!(times, nstates * 2)
    states = [scur]; sizehint!(states, nstates * 2)

    # flag for whether to stop simulation
    # obviously don't simulate if the initial state is absorbing
    keep_going = isa(model.totalhazards[scur], _TotalHazardTransient)
    
    # sample the cumulative incidence if transient
    if keep_going
        u = max(rand(rng), delta_u)
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

            # Use solver dispatch to find jump time
            timeincrement = _find_jump_time(solver, gap_fn, delta_t, interval_len, delta_t)

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
                u           = max(rand(rng), delta_u) # sample cumulative incidence
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

    return SamplePath(subj, times, states)
end

