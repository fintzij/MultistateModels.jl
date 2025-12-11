# Part 2: Hot Paths Analysis

## Overview

This document details the critical execution paths in MultistateModels.jl. Understanding these paths is essential for targeted optimization.

---

## 2.1 Simulation Hot Path

### Entry Point: `simulate_path(model, subj)`

**Location**: `simulation.jl:550-720`

**Call graph**:
```
simulate_path(model, subj)
├── get_hazard_params(model.parameters, model.hazards)  # Parameter extraction
├── _materialize_covariates(subjdat_row, hazards)       # Covariate caching
├── maybe_time_transform_context(params, subj_dat, hazards)  # Cache setup
│
└── MAIN LOOP (while keep_going)
    ├── survprob(timeinstate, timeinstate + dt, params, covars, totalhazard, hazards)
    │   └── eval_cumhaz(hazard, lb, ub, params, covars)  # For each component hazard
    │
    ├── IF event in interval:
    │   ├── _find_jump_time(solver, gap_fn, lower, upper)  # ROOT FINDING
    │   │   └── solve(IntervalNonlinearProblem(gap_fn, bounds), ITP())
    │   │       └── gap_fn calls survprob repeatedly
    │   │
    │   ├── next_state_probs!(ns_probs, trans_inds, timeinstate, scur, covars, params, ...)
    │   │   └── eval_hazard(hazard, t, params, covars) for each exit hazard
    │   │
    │   └── _sample_next_state(rng, ns_probs, trans_inds)
    │
    └── ELSE: increment timeinstate, cuminc, move to next interval
```

### Critical Functions

#### `survprob` (hazards.jl:950-1050)
```julia
function survprob(lb, ub, pars, covars_cache, totalhazard::_TotalHazardTransient, hazards; 
                  give_log=false, apply_transform=false, cache_context=nothing)
    # Accumulate cumulative hazard over all component hazards
    cumhaz = zero(eltype(pars))
    for hazard_idx in totalhazard.components
        hazard = hazards[hazard_idx]
        hazard_pars = pars[hazard.hazname]
        covars = _covariate_entry(covars_cache, hazard_idx)
        
        if apply_transform && hazard.metadata.time_transform
            # Use cached time transform
            cumhaz += _cached_cumhaz(hazard, hazard_pars, covars, lb, ub, cache_context, hazard_idx)
        else
            cumhaz += eval_cumhaz(hazard, lb, ub, hazard_pars, covars)
        end
    end
    give_log ? -cumhaz : exp(-cumhaz)
end
```

**Optimization opportunities**:
1. `pars[hazard.hazname]` does Dict lookup - could pre-extract to vector
2. `_covariate_entry` dispatch on covars_cache type
3. Time transform caching may have overhead > benefit

#### `_find_jump_time` (simulation.jl:480-510)
```julia
function _find_jump_time(solver::OptimJumpSolver, gap_fn, lower, upper)
    prob = IntervalNonlinearProblem(gap_fn, (lower, upper))
    sol = solve(prob, ITP())
    return sol.u
end
```

**Optimization opportunities**:
1. Problem construction per call - could reuse with `remake`
2. For exponential hazards, closed-form: `t = -log(1-u) / rate`
3. ITP algorithm may be overkill for smooth monotonic functions

#### `eval_cumhaz` dispatch (hazards.jl:600-800)

Multiple methods for different hazard types:

```julia
# Exponential (Markov)
@inline function eval_cumhaz(hazard::MarkovHazard, lb, ub, pars, covars)
    rate = pars.baseline[1]  # Already on natural scale
    linpred = _linear_predictor(pars, covars, hazard)
    return rate * exp(linpred) * (ub - lb)
end

# Weibull (SemiMarkov)
@inline function eval_cumhaz(hazard::SemiMarkovHazard, lb, ub, pars, covars)
    shape, scale = pars.baseline[1], pars.baseline[2]  # Natural scale
    linpred = _linear_predictor(pars, covars, hazard)
    effect = hazard.metadata.linpred_effect
    # ... Weibull cumulative hazard formula
end

# Spline
@inline function eval_cumhaz(hazard::RuntimeSplineHazard, lb, ub, pars, covars)
    # Numerical integration of spline basis
    ...
end
```

**Optimization opportunities**:
1. `_linear_predictor` creates temporaries for covariate extraction
2. Spline cumulative hazard uses quadrature - could cache basis evaluations

---

## 2.2 MCEM Inference Hot Path

### Entry Point: `fit(model::MultistateSemiMarkovModel)`

**Location**: `modelfitting.jl:900-1400`

**Call graph**:
```
fit(model; maxiter, tol, ...)
├── SETUP
│   ├── build_tpm_mapping(model.data)
│   ├── build_hazmat_book, build_tpm_book
│   ├── compute_hazmat!, compute_tmat!  # Initial surrogate TPMs
│   └── [optional] fit_phasetype_surrogate(...)
│
└── MCEM LOOP (while !converged && iter < maxiter)
    │
    ├── E-STEP: DrawSamplePaths!(...)
    │   ├── Per subject:
    │   │   ├── ForwardFiltering!(fbmats, subj_dat, tpm_book, tpm_map, emat)
    │   │   └── draw_samplepath(subj, model, tpm_book, hazmat_book, ...)
    │   │       ├── BackwardSampling!(subj_dat, fbmats)
    │   │       └── sample_ecctmc!(times, states, P, Q, a, b, t0, t1)
    │   │
    │   ├── Compute target log-likelihoods:
    │   │   └── loglik(params_target, path, hazards_target, model)  # Per path
    │   │       └── loglik_path(pars, subjdat_df, hazards, totalhazards, tmat)
    │   │
    │   └── ComputeImportanceWeightsESS!(...)
    │       └── psis(logweights)  # Pareto smoothed importance sampling
    │
    ├── M-STEP: Optimization
    │   ├── loglik_semi_markov(params, data; neg=true)  # Objective
    │   │   └── Per subject, per path:
    │   │       loglik_path(...) * ImportanceWeight
    │   │
    │   └── solve(OptimizationProblem, Ipopt.Optimizer())
    │
    ├── Update surrogate (if EM_restart)
    │   └── compute_hazmat!, compute_tmat!
    │
    └── Convergence check
```

### Critical Functions in E-Step

#### `DrawSamplePaths!` (sampling.jl:1-200)
```julia
function DrawSamplePaths!(model; ess_target, ess_cur, samplepaths, 
                          loglik_surrog, loglik_target_prop, loglik_target_cur,
                          ImportanceWeights, tpm_book_surrogate, ...)
    
    for i in 1:nsubj
        # Forward filtering
        if any(subj_dat.obstype .∉ Ref([1,2]))
            ForwardFiltering!(fbmats[i], subj_dat, tpm_book, tpm_map, emat)
        end
        
        # Draw paths until ESS target met
        while ess_cur[i] < ess_target
            # Sample path
            path = draw_samplepath(i, model, tpm_book_surrogate, ...)
            push!(samplepaths[i], path)
            
            # Compute log-likelihoods
            loglik_target[i][end] = loglik(params_target, path, hazards_target, model)
            loglik_surrog[i][end] = loglik(params_surrog, path, hazards_surrog, model)
            
            # Update weights and ESS
            ComputeImportanceWeightsESS!(...)
        end
    end
end
```

**Optimization opportunities**:
1. `loglik(params, path, hazards, model)` called per path - could batch
2. `make_subjdat` called inside `loglik` - should cache in `CachedPathData`
3. Per-subject loop is embarrassingly parallel

#### `sample_ecctmc!` (sampling.jl:600-700)
```julia
function sample_ecctmc!(jumptimes, stateseq, P, Q, a, b, t0, t1)
    # Endpoint-conditioned CTMC sampling via uniformization
    nstates = size(Q, 1)
    T = t1 - t0
    m = maximum(abs.(diag(Q)))
    
    # R = I + Q/m (uniformized transition matrix)
    R = ElasticArray{Float64}(undef, nstates, nstates, 1)
    R[:,:,1] = diagm(ones(Float64, nstates)) + Q / m
    
    # Sample number of jumps via Poisson
    # Sample jump times uniformly
    # Sample states via backward sampling
    ...
end
```

**Optimization opportunities**:
1. `ElasticArray` allocation per call - could pre-allocate
2. `diagm(ones(...))` creates temporary - use `I` or pre-allocated identity
3. `R[:,:,1]^njumps` - matrix power is expensive, cache powers

#### `loglik_path` (likelihoods.jl:70-130)
```julia
function loglik_path(pars, subjectdata::DataFrame, hazards, totalhazards, tmat)
    ll = 0.0
    tt_context = maybe_time_transform_context(pars, subjectdata, hazards)
    
    for i in 1:nrow(subjectdata)
        origin_state = subjectdata.statefrom[i]
        row_data = @view subjectdata[i, :]
        
        # Survival contribution
        ll += survprob(sojourn, sojourn + increment, pars, row_data, 
                       totalhazards[origin_state], hazards; give_log=true)
        
        # Transition contribution (if state changed)
        if statefrom != stateto
            trans_idx = tmat[statefrom, stateto]
            hazard = hazards[trans_idx]
            ll += log(eval_hazard(hazard, sojourn + increment, pars[hazard.hazname], row_data))
        end
    end
    return ll
end
```

**Optimization opportunities**:
1. `nrow(subjectdata)` + DataFrame row access - use column vectors directly
2. `pars[hazard.hazname]` Dict lookup in inner loop
3. `maybe_time_transform_context` allocates per call

### Critical Functions in M-Step

#### `loglik_semi_markov` (likelihoods.jl:1200-1350)
```julia
function loglik_semi_markov(parameters, data::SMPanelData; neg=true, use_sampling_weight=true)
    pars = safe_unflatten(parameters, data.model)
    
    ll = 0.0
    for i in 1:nsubj
        subj_ll = 0.0
        for j in 1:length(data.paths[i])
            path = data.paths[i][j]
            weight = data.ImportanceWeights[i][j]
            
            # make_subjdat converts path to DataFrame
            subjdat_df = make_subjdat(path, subj_dat)
            path_ll = loglik_path(pars, subjdat_df, hazards, totalhazards, tmat)
            subj_ll += path_ll * weight
        end
        ll += subj_ll * SubjectWeight[i]
    end
    return neg ? -ll : ll
end
```

**Optimization opportunities**:
1. `make_subjdat` called for every path every M-step iteration - CACHE THIS
2. `safe_unflatten` called every objective evaluation - could cache structure
3. Inner loops are independent - parallelize

---

## 2.3 Markov Panel Likelihood Hot Path

### Entry Point: `loglik_markov(parameters, data::MPanelData)`

**Location**: `likelihoods.jl:740-960`

**Call graph**:
```
loglik_markov(parameters, data)
├── safe_unflatten(parameters, model)
├── build_hazmat_book, build_tpm_book  # Pre-allocated
├── ExponentialUtilities.alloc_mem(...)  # Matrix exp cache
│
├── FOR each unique (covariate, time) pattern:
│   ├── compute_hazmat!(hazmat_book[t], pars, hazards, tpm_index, data)
│   └── compute_tmat!(tpm_book[t], hazmat_book[t], tpm_index, cache)
│       └── exponential!(Q * dt, ExpMethodGeneric(), cache)
│
└── FOR each subject:
    ├── IF all observations exact or panel (no censoring):
    │   └── Sum log(P[statefrom, stateto]) from tpm_book
    │
    └── ELSE (censored observations):
        └── Forward algorithm with emission probabilities
```

### Critical Functions

#### `compute_hazmat!` (likelihoods.jl:1500-1600)
```julia
function compute_hazmat!(Q, pars, hazards, tpm_index, data)
    fill!(Q, 0.0)
    nstates = size(Q, 1)
    
    for r in 1:nstates
        for s in 1:nstates
            if tmat[r,s] != 0
                hazard_idx = tmat[r,s]
                hazard = hazards[hazard_idx]
                hazard_pars = pars[hazard.hazname]
                
                # Get representative row for this covariate pattern
                row_idx = tpm_index.datind[1]
                row_data = @view data[row_idx, :]
                
                Q[r,s] = eval_hazard(hazard, 0.0, hazard_pars, row_data)
            end
        end
        Q[r,r] = -sum(Q[r, :])  # Diagonal = negative row sum
    end
end
```

**Optimization opportunities**:
1. Could vectorize hazard evaluation
2. `@view data[row_idx, :]` still allocates DataFrameRow wrapper

#### `compute_tmat!` (likelihoods.jl:1600-1650)
```julia
function compute_tmat!(P, Q, tpm_index, cache)
    dt = tpm_index.tstop[1]  # Time interval
    copyto!(P, Q)
    P .*= dt
    exponential!(P, ExpMethodGeneric(), cache)  # P = exp(Q*dt)
end
```

**Optimization opportunities**:
1. For 2-state models, closed-form exponential
2. For 3-state with special structure, analytical formulas may exist

---

## 2.4 Allocation Hot Spots (Suspected)

| Location | Operation | Concern |
|----------|-----------|---------|
| `simulation.jl` | `IntervalNonlinearProblem` construction | Per-jump allocation |
| `sampling.jl` | `ElasticArray` in `sample_ecctmc!` | Per-sample allocation |
| `likelihoods.jl` | `make_subjdat` DataFrame creation | Per-path, per-iteration |
| `hazards.jl` | `_linear_predictor` tuple creation | Per-hazard-eval |
| `helpers.jl` | `unflattenAD` NamedTuple construction | Per-objective-eval |
| `common.jl` | `TimeTransformCache` Dict operations | Per-cached-eval |

---

## 2.5 Parallelization Opportunities

| Operation | Parallelizable? | Barrier |
|-----------|-----------------|---------|
| `simulate(; nsim=N)` | Yes (trivial) | None |
| E-step per-subject | Yes | RNG state management |
| `loglik_path` per-path | Yes | Accumulation into scalar |
| M-step optimization | No | Sequential algorithm |
| TPM computation per-pattern | Yes | None |

**Current parallel support**:
- `Threads.@threads` used in some places
- Not systematic

**Recommendation**: After profiling, add `@threads` to largest parallel regions.

---

## Action Items from Hot Path Analysis

| Item | Priority | Expected Impact |
|------|----------|-----------------|
| Profile `_find_jump_time` overhead | High | May reveal 20-50% simulation time |
| Cache `make_subjdat` results | High | Avoid repeated DataFrame construction |
| Profile `sample_ecctmc!` allocations | Medium | Memory pressure in MCEM |
| Pre-extract hazard params to vector | Medium | Avoid Dict lookup in inner loops |
| Benchmark TimeTransformCache | Medium | May remove overhead |
| Add systematic parallelization | Low | After serial optimization |
