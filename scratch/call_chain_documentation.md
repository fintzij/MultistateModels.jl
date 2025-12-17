# MultistateModels.jl - Comprehensive Call Chain Documentation

**Created:** 2025-12-18  
**Branch:** `package_streamlining`

This document maps the complete call chains for model generation, simulation, and inference, including triggers based on user options, data types, and model structure.

---

## Table of Contents

1. [Model Generation](#1-model-generation)
2. [Simulation](#2-simulation)
3. [Inference (Fitting)](#3-inference-fitting)
4. [Likelihood Computation](#4-likelihood-computation)
5. [Variance Estimation](#5-variance-estimation)
6. [Test Coverage Summary](#6-test-coverage-summary)

---

## 1. Model Generation

### Entry Points

| Function | File | Line | Purpose |
|----------|------|------|---------|
| `Hazard()` | construction/multistatemodel.jl | 1-150 | User API for hazard specification |
| `multistatemodel()` | construction/multistatemodel.jl | 200+ | Main model constructor |

### 1.1 Hazard Specification (`Hazard()`)

User-facing constructor for specifying transition hazards.

**Supported Families:**

| Family | Keyword | Parameters | File |
|--------|---------|------------|------|
| Exponential | `:exp` | baseline rate | hazard/evaluation.jl |
| Weibull | `:wei` | scale, shape | hazard/evaluation.jl |
| Gompertz | `:gom` | scale, shape | hazard/evaluation.jl |
| Spline | `:sp` | knots, degree | hazard/spline.jl |
| Phase-Type | `:pt` | n_phases, constraints | phasetype/expansion.jl |

**Call Chain:**
```
Hazard(family, transition, formula; kwargs...)
  ├─ _validate_hazard_family()           # Check family symbol
  ├─ _parse_transition()                 # Extract from/to states
  ├─ _parse_formula()                    # Parse covariate formula
  └─ return HazardFunction struct
```

### 1.2 Model Construction (`multistatemodel()`)

Main entry point for building multistate models.

**Signature:**
```julia
multistatemodel(
    data::DataFrame, 
    hazards::Vector{<:HazardFunction};
    obstype::Symbol = :panel,         # :exact, :panel, :censored
    method::Symbol = :markov,         # :markov, :semi_markov
    SamplingWeights = nothing,
    CensoringPatterns = nothing,
    constraints = nothing,
    ad_backend = :ForwardDiff,
    n_threads = Threads.nthreads()
)
```

**Dispatch Triggers:**

| Condition | Trigger | Result |
|-----------|---------|--------|
| Any hazard has `family = :pt` | `_has_phasetype_hazards()` | Route to `_build_phasetype_model_from_hazards()` |
| All hazards are Markov-compatible | `all(h -> h.family in [:exp, :wei, :gom])` | Standard Markov model |
| `obstype = :exact` | Data has exact transition times | Exact likelihood path |
| `obstype = :panel` | Data has interval-censored observations | Panel likelihood path |
| `method = :semi_markov` | Semi-Markov process | MCEM fitting required |

**Standard (Non-Phase-Type) Call Chain:**
```
multistatemodel(data, hazards; kwargs...)
  │
  ├─ 1. _validate_inputs!()              # Data validation
  │      ├─ check_data!()                # Required columns, types
  │      ├─ _validate_transitions()      # Valid transitions in data
  │      └─ _validate_absorbing()        # Absorbing state consistency
  │
  ├─ 2. enumerate_hazards()              # List all transitions
  │
  ├─ 3. create_tmat()                    # Build transition matrix
  │
  ├─ 4. build_hazards()                  # Convert HazardFunction → _Hazard
  │      └─ _build_hazard_from_spec()    # Per-hazard construction
  │           ├─ _Exponential()          # For :exp
  │           ├─ _Weibull()              # For :wei
  │           ├─ _Gompertz()             # For :gom
  │           └─ _SplineHazard()         # For :sp
  │
  ├─ 5. build_emat()                     # Emission matrix (for censoring)
  │
  ├─ 6. _build_subject_indices()         # Per-subject row indices
  │
  ├─ 7. _build_parameters_namedtuple()   # Initialize parameter storage
  │
  └─ 8. MultistateModel(...)             # Return model struct
```

**Phase-Type Model Call Chain:**
```
multistatemodel(data, hazards; kwargs...)  [when has_phasetype_hazards()]
  │
  └─ _build_phasetype_model_from_hazards()    [phasetype/expansion.jl:941]
       │
       ├─ 1. build_phasetype_mappings()       # State space mappings
       │      ├─ _build_expanded_tmat()       # Expanded transition matrix
       │      └─ _build_expanded_hazard_indices()
       │
       ├─ 2. expand_hazards_for_phasetype()   # Convert PT → Markov hazards
       │      ├─ _build_progression_hazard()  # λ (phase progression)
       │      ├─ _build_exit_hazard()         # μ (state exit)
       │      └─ _build_expanded_hazard()     # Non-PT hazards
       │
       ├─ 3. expand_data_for_phasetype_fitting()  # Expand data
       │      ├─ expand_data_for_phasetype()
       │      └─ _build_phase_censoring_patterns()
       │
       ├─ 4. _generate_sctp_constraints()     # SCTP ratio constraints
       │
       ├─ 5. _build_expanded_parameters()     # Expanded space params
       │
       ├─ 6. _build_original_parameters()     # User-facing params
       │
       └─ 7. MultistateModel(...)             # With phasetype_expansion metadata
```

### 1.3 Output: `MultistateModel` Struct

```julia
mutable struct MultistateModel <: MultistateProcess
    data::DataFrame                        # Input data
    parameters::NamedTuple                 # Parameter storage
    hazards::Vector{<:_Hazard}             # Internal hazard representations
    totalhazards::Vector{_TotalHazard}     # Sum of hazards from each state
    tmat::Matrix{Int64}                    # Transition matrix
    emat::Matrix{Float64}                  # Emission matrix
    hazkeys::Dict{Symbol,Int64}            # Hazard name → index
    subjectindices::Vector{Vector{Int64}}  # Per-subject row indices
    SubjectWeights::Vector{Float64}        # Sampling weights
    ObservationWeights::Union{Nothing,Vector{Float64}}
    CensoringPatterns::Matrix{Float64}
    markovsurrogate::Union{Nothing,MarkovSurrogate}
    modelcall::NamedTuple                  # Construction metadata
    phasetype_expansion::Union{Nothing,PhaseTypeExpansion}  # PT metadata
end
```

**Test Coverage:**
- ✅ `test_modelgeneration.jl` - Standard model construction
- ✅ `test_phasetype.jl` - Phase-type model construction
- ✅ `test_hazards.jl` - Hazard specification

---

## 2. Simulation

### Entry Points

| Function | File | Line | Purpose |
|----------|------|------|---------|
| `simulate()` | simulation/simulate.jl | 50 | Sample a single path |
| `simulate_paths()` | simulation/simulate.jl | 150 | Sample multiple paths |
| `simulate_data()` | simulation/simulate.jl | 250 | Generate synthetic dataset |
| `simulate_path()` | simulation/simulate.jl | 80 | Internal path sampling |

### 2.1 `simulate()` - Single Path

**Signature:**
```julia
simulate(
    model::MultistateModel, 
    subj::Int;
    strategy = DirectTransformStrategy(),
    solver = ExponentialJumpSolver(),
    expanded::Bool = false,          # Return expanded or collapsed path
    rng = Random.default_rng()
)
```

**Strategy Options:**

| Strategy | Purpose | When to Use |
|----------|---------|-------------|
| `DirectTransformStrategy()` | Direct inverse CDF | Small state spaces |
| `CachedTransformStrategy()` | Cached cumulative hazards | Many simulations |

**Solver Options:**

| Solver | Process | Method |
|--------|---------|--------|
| `ExponentialJumpSolver()` | Markov | Closed-form exponential waiting times |
| `OptimJumpSolver()` | Semi-Markov | ITP root-finding for jump times |
| `HybridJumpSolver()` | Mixed | State-dependent solver selection |

**Call Chain:**
```
simulate(model, subj; kwargs...)
  │
  ├─ _get_subject_data()              # Extract subject's data
  │
  ├─ simulate_path()                  # Core simulation
  │    │
  │    ├─ [Markov case: ExponentialJumpSolver]
  │    │    └─ _draw_exponential_jump()   # Closed-form waiting time
  │    │         ├─ compute_total_hazard()
  │    │         └─ _sample_transition()   # Which state to visit
  │    │
  │    └─ [Semi-Markov case: OptimJumpSolver]
  │         └─ _draw_jump_itp()           # ITP root-finding
  │              ├─ cumulative_hazard()
  │              └─ ITP.solve()
  │
  └─ [if has_phasetype_expansion && !expanded]
       └─ _collapse_path()            # Collapse to observed states
            └─ map_phases_to_states() # Phase → original state
```

### 2.2 `simulate_paths()` - Multiple Paths

**Signature:**
```julia
simulate_paths(
    model::MultistateModel,
    n_paths::Int;
    expanded::Bool = false,
    parallel::Bool = false,
    kwargs...
)
```

**Call Chain:**
```
simulate_paths(model, n_paths; kwargs...)
  │
  ├─ [parallel = false]
  │    └─ map(subj -> simulate(model, subj; kwargs...), 1:n_paths)
  │
  └─ [parallel = true]
       └─ ThreadsX.map(subj -> simulate(model, subj; kwargs...), 1:n_paths)
```

### 2.3 `simulate_data()` - Synthetic Dataset

**Signature:**
```julia
simulate_data(
    model::MultistateModel,
    n_subjects::Int;
    observation_times = nothing,     # Panel observation times
    covariates = nothing,            # Covariate distributions
    expanded::Bool = false,
    kwargs...
)
```

**Call Chain:**
```
simulate_data(model, n_subjects; kwargs...)
  │
  ├─ _generate_covariates()           # If covariate distributions provided
  │
  ├─ _generate_observation_times()    # If panel observation scheme
  │
  └─ for subj in 1:n_subjects
       ├─ simulate(model, subj; kwargs...)
       └─ _path_to_dataframe()        # Convert SamplePath → DataFrame rows
```

### 2.4 Output: `SamplePath` Struct

```julia
struct SamplePath
    times::Vector{Float64}      # Jump times
    states::Vector{Int64}       # States visited
    subject::Int                # Subject ID
    metadata::Dict{Symbol,Any}  # Additional info
end
```

**Test Coverage:**
- ✅ `test_simulation.jl` - Standard simulation
- ✅ `longtest_simulation_distribution.jl` - Distribution validation
- ✅ `longtest_simulation_tvc.jl` - Time-varying covariates

---

## 3. Inference (Fitting)

### Entry Point

| Function | File | Line | Purpose |
|----------|------|------|---------|
| `fit()` | inference/fit.jl | 1 | Main fitting entry point |

### 3.1 `fit()` - Main Entry

**Signature:**
```julia
fit(
    model::MultistateModel;
    method::Symbol = :auto,           # :exact, :markov, :mcem
    maxiter::Int = 1000,
    tol::Float64 = 1e-6,
    verbose::Bool = true,
    proposal::Union{Symbol,ProposalConfig} = :auto,
    n_mcem_samples::Int = 100,
    sir_threshold::Float64 = 0.5,
    variance_method::Symbol = :model_based,
    constraints = nothing,
    kwargs...
)
```

### 3.2 Dispatch Triggers

| Condition | Method Selection | Function Called |
|-----------|------------------|-----------------|
| `obstype = :exact` | Direct MLE | `_fit_exact()` |
| `method = :markov` + panel data | Matrix exponential | `_fit_markov_panel()` |
| `method = :semi_markov` OR phase-type | MCEM | `_fit_mcem()` |
| Semi-Markov + phase-type surrogate | MCEM with PT proposal | `_fit_mcem()` + `PhaseTypeSurrogate` |

**Dispatch Logic:**
```julia
function fit(model; method = :auto, kwargs...)
    if method == :auto
        method = _select_fitting_method(model)
    end
    
    if is_exact_data(model)
        return _fit_exact(model; kwargs...)
    elseif is_markov(model) && is_panel_data(model)
        return _fit_markov_panel(model; kwargs...)
    else
        return _fit_mcem(model; kwargs...)
    end
end
```

### 3.3 Exact Observation Fitting (`_fit_exact()`)

For data with exact transition times.

```
_fit_exact(model; kwargs...)
  │
  ├─ _build_objective()               # Negative log-likelihood
  │    └─ loglik_exact()              # Exact observation likelihood
  │
  ├─ _setup_optimizer()               # Optim.jl setup
  │    ├─ _build_constraints()        # Parameter constraints
  │    └─ _select_algorithm()         # LBFGS, Newton, etc.
  │
  ├─ Optim.optimize()                 # Optimization
  │
  └─ _compute_variance()              # Fisher information inversion
       └─ ForwardDiff.hessian()
```

### 3.4 Markov Panel Fitting (`_fit_markov_panel()`)

For panel data with Markov assumption.

```
_fit_markov_panel(model; kwargs...)
  │
  ├─ _build_objective()               # Negative log-likelihood
  │    └─ loglik_markov()             # Matrix exponential likelihood
  │         ├─ build_Q()              # Intensity matrix
  │         └─ exp(Q * dt)            # Transition probabilities
  │
  ├─ _setup_optimizer()
  │
  ├─ Optim.optimize()
  │
  └─ _compute_variance()
       └─ ForwardDiff.hessian()
```

### 3.5 MCEM Fitting (`_fit_mcem()`)

For semi-Markov models or when path sampling is required.

**MCEM Algorithm:**
```
_fit_mcem(model; n_samples, maxiter, kwargs...)
  │
  ├─ 1. INITIALIZATION
  │    ├─ initialize_parameters!()    # Crude initialization
  │    └─ _build_surrogate()          # Importance sampling surrogate
  │         ├─ MarkovSurrogate()      # For Markov proposal
  │         └─ PhaseTypeSurrogate()   # For PT proposal
  │
  ├─ 2. E-STEP (repeated)
  │    ├─ _draw_paths()               # Sample latent paths
  │    │    └─ ffbs()                 # Forward-filtering backward-sampling
  │    │         ├─ forward_filter()
  │    │         └─ backward_sample()
  │    │
  │    └─ _compute_weights()          # Importance weights
  │         ├─ loglik_path()          # Target path likelihood
  │         └─ loglik_surrogate()     # Proposal path likelihood
  │
  ├─ 3. SIR RESAMPLING (if ESS low)
  │    └─ sir_resample()              # Sampling importance resampling
  │         ├─ _compute_ess()         # Effective sample size
  │         └─ _multinomial_resample()
  │
  ├─ 4. M-STEP
  │    ├─ _build_weighted_objective() # Weighted negative log-likelihood
  │    └─ Optim.optimize()            # Parameter update
  │
  ├─ 5. CONVERGENCE CHECK
  │    └─ _caffo_rule()               # Ascent-based stopping rule
  │
  └─ 6. FINAL VARIANCE
       └─ _compute_mcem_variance()    # Louis' method or IJ
```

**Surrogate Selection:**

| Proposal Type | Condition | Surrogate |
|---------------|-----------|-----------|
| `:markov` | Default for Markov models | `MarkovSurrogate` |
| `:phasetype` | Phase-type model or explicit request | `PhaseTypeSurrogate` |
| `:auto` | Automatic selection | Based on `needs_phasetype_proposal()` |

### 3.6 Output: `MultistateModelFitted` Struct

```julia
mutable struct MultistateModelFitted <: MultistateProcess
    # Inherited from MultistateModel
    data::DataFrame
    parameters::NamedTuple
    hazards::Vector{<:_Hazard}
    # ... other MultistateModel fields
    
    # Fitting results
    loglik::Float64                    # Maximum log-likelihood
    vcov::Matrix{Float64}              # Variance-covariance matrix
    convergence::ConvergenceRecords    # Convergence history
    proposed_paths::Union{Nothing,ProposedPaths}  # MCEM paths
    fit_metadata::NamedTuple           # Fitting configuration
end
```

**Test Coverage:**
- ✅ `test_mcem.jl` - MCEM algorithm
- ✅ `longtest_exact_markov.jl` - Exact fitting
- ✅ `longtest_mcem.jl` - Full MCEM fitting
- ✅ `longtest_phasetype_panel.jl` - Phase-type fitting

---

## 4. Likelihood Computation

### Entry Points

| Function | File | Line | Purpose |
|----------|------|------|---------|
| `loglik()` | likelihood/loglik.jl | 50 | Main likelihood dispatcher |
| `loglik_exact()` | likelihood/loglik.jl | 100 | Exact observation likelihood |
| `loglik_markov()` | likelihood/loglik.jl | 200 | Markov panel likelihood |
| `loglik_path()` | likelihood/loglik.jl | 400 | Path likelihood |

### 4.1 Dispatch Logic

```julia
function loglik(model, params)
    if is_exact_data(model)
        return loglik_exact(model, params)
    elseif is_markov(model)
        return loglik_markov(model, params)
    else
        return loglik_semi_markov(model, params)
    end
end
```

### 4.2 Exact Observation Likelihood

```
loglik_exact(model, params)
  │
  ├─ for subj in subjects
  │    ├─ for transition in subject_transitions
  │    │    ├─ log_hazard()           # Log hazard at transition time
  │    │    └─ cumulative_hazard()    # Integrated hazard
  │    │
  │    └─ ll_subj = Σ(log_haz) - Σ(cum_haz)
  │
  └─ return Σ(ll_subj * weight)
```

### 4.3 Markov Panel Likelihood

```
loglik_markov(model, params)
  │
  ├─ build_Q(params)                  # Intensity matrix Q
  │
  ├─ for subj in subjects
  │    ├─ for interval in subject_intervals
  │    │    ├─ P = exp(Q * dt)        # Transition probability matrix
  │    │    └─ ll += log(P[s_from, s_to])
  │    │
  │    └─ [if censored]
  │         └─ ll += log(P * emat)    # Emission matrix convolution
  │
  └─ return Σ(ll_subj * weight)
```

### 4.4 Phase-Type Marginal Likelihood

For MCEM with phase-type models:

```
compute_phasetype_marginal_loglik(model, params)
  │
  ├─ build_expanded_Q(params)         # Expanded intensity matrix
  │
  ├─ for subj in subjects
  │    ├─ forward_pass()              # Forward algorithm
  │    │    ├─ α[0] = π               # Initial distribution
  │    │    └─ α[t] = α[t-1] * P * E  # Forward recursion
  │    │
  │    └─ ll_subj = log(sum(α[T]))    # Terminal probability
  │
  └─ return Σ(ll_subj * weight)
```

**Test Coverage:**
- ✅ `test_mll_consistency.jl` - Likelihood consistency
- ✅ `test_reversible_tvc_loglik.jl` - Time-varying covariates

---

## 5. Variance Estimation

### Methods Available

| Method | Keyword | Description |
|--------|---------|-------------|
| Model-based | `:model_based` | Inverse Fisher information |
| IJ/Sandwich | `:ij` | Information-sandwich estimator |
| Jackknife | `:jackknife` | Leave-one-out variance |
| Robust | `:robust` | Weighted IJ estimator |

### 5.1 Model-Based Variance

```
_variance_model_based(model, params)
  │
  ├─ H = ForwardDiff.hessian(loglik, params)
  │
  └─ return -inv(H)
```

### 5.2 IJ/Sandwich Variance

```
_variance_ij(model, params)
  │
  ├─ H = ForwardDiff.hessian(loglik, params)    # Information matrix
  │
  ├─ J = sum over subjects:                      # Score variance
  │       grad_i = ForwardDiff.gradient(loglik_i, params)
  │       grad_i * grad_i'
  │
  └─ return inv(H) * J * inv(H)                  # Sandwich
```

**Test Coverage:**
- ✅ `test_variance.jl` - Variance estimation
- ✅ `longtest_variance_validation.jl` - Simulation-based validation

---

## 6. Test Coverage Summary

### Unit Tests

| Test File | Coverage Area | Key Functions |
|-----------|---------------|---------------|
| `test_modelgeneration.jl` | Model construction | `multistatemodel`, `Hazard` |
| `test_hazards.jl` | Hazard evaluation | `eval_hazard`, `cumulative_hazard` |
| `test_simulation.jl` | Path simulation | `simulate`, `simulate_paths` |
| `test_mcem.jl` | MCEM algorithm | `_fit_mcem`, `ffbs` |
| `test_phasetype.jl` | Phase-type models | `expand_hazards_for_phasetype` |
| `test_variance.jl` | Variance estimation | `_variance_ij`, `_variance_jackknife` |
| `test_mll_consistency.jl` | Likelihood | `loglik_exact`, `loglik_markov` |

### Long Tests

| Test File | Coverage Area | Validation |
|-----------|---------------|------------|
| `longtest_exact_markov.jl` | Exact fitting | Parameter recovery |
| `longtest_mcem.jl` | MCEM fitting | Convergence, bias |
| `longtest_phasetype_panel.jl` | PT + panel | Full workflow |
| `longtest_phasetype_exact.jl` | PT + exact | Full workflow |
| `longtest_simulation_distribution.jl` | Simulation | Distribution matching |
| `longtest_variance_validation.jl` | Variance | Coverage probabilities |

### Integration Tests

| Test File | Coverage Area |
|-----------|---------------|
| `test_parallel_likelihood.jl` | Multi-threaded likelihood |
| `test_parameter_ordering.jl` | Parameter consistency |

---

## Appendix: Key Design Decisions

### A.1 Phase-Type as Preprocessing

Phase-type models are implemented as **preprocessing**, not a separate model type:
1. PT hazards → expanded Markov hazards (at construction)
2. Standard Markov routines for likelihood/fitting
3. Path collapse for user-facing results

**Rationale:** Eliminates ~3,000 lines of redundant PT-specific code.

### A.2 Trait-Based Dispatch

Model behavior determined by content, not type:
```julia
is_markov(m) = all(h -> h isa _MarkovHazard, m.hazards)
is_panel_data(m) = m.modelcall.obstype == :panel
has_phasetype_expansion(m) = !isnothing(m.phasetype_expansion)
```

**Rationale:** Single `MultistateModel` struct with flexible dispatch.

### A.3 Surrogate-Based MCEM

Importance sampling with learned surrogates:
- `MarkovSurrogate`: Fits Markov model to path statistics
- `PhaseTypeSurrogate`: Uses Coxian phase-type approximation

**Rationale:** Efficient proposal distributions for path sampling.

### A.4 SIR Resampling

Sampling Importance Resampling with O(m log m) pool sizing:
- ESS monitoring for weight degeneracy
- Adaptive pool expansion
- Multinomial resampling

**Rationale:** Handles importance weight variance in MCEM.
