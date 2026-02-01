---
name: codebase-knowledge
description: Deep knowledge of MultistateModels.jl codebase structure, conventions, and current state. READ THIS FIRST when starting any session involving code changes.
---

# MultistateModels.jl Codebase Knowledge

**Read this skill file FIRST at the start of every session.**

**Last Updated**: 2026-01-30  
**Branch**: `penalized_splines` (active development)  
**Active Work**: Performance iteration for smoothing parameter selection (see Part 7)

---

## Part 1: Package Overview

**MultistateModels.jl** implements continuous-time multistate models for survival analysis:
- **Parametric hazards**: Exponential (`:exp`), Weibull (`:wei`), Gompertz (`:gom`)
- **Spline hazards**: B-splines (`:sp`) with optional monotonicity and automatic smoothing
- **Phase-type hazards**: Coxian (`:pt`) for flexible sojourn time modeling
- **Observation types**: Exact (obstype=1), panel (obstype=2), state censoring for panel data with possible states given by censoring patterns (obstype>2)
- **Covariate effects**: Proportional hazards (`:ph`), accelerated failure time (`:aft`)
- **Inference**: Direct MLE, matrix exponential MLE, Monte Carlo EM (MCEM)

### Key Dependencies

| Package | Purpose |
|---------|---------|
| ForwardDiff | AD for gradients/Hessians (NEVER use finite differences) |
| Optimization.jl + OptimizationIpopt | Constrained optimization |
| ExponentialUtilities | Matrix exponentials for TPMs |
| BSplineKit | B-spline basis functions |
| ImplicitDifferentiation | AD through implicit functions (PIJCV) |
| ParetoSmooth | PSIS for importance sampling diagnostics |
| PrecompileTools | TTFX reduction via @compile_workload |

---

## Part 2: Source Architecture

```
src/
├── MultistateModels.jl      # Module definition, exports
├── precompile.jl            # PrecompileTools workload
├── types/                   # Type definitions (loaded first)
├── construction/            # Model building: multistatemodel(), Hazard()
├── hazard/                  # Hazard evaluation, splines, TPM
├── likelihood/              # loglik_exact, loglik_markov, loglik_semi_markov
├── inference/               # fit_exact, fit_markov, fit_mcem, smoothing_selection/
├── phasetype/               # Phase-type expansion, surrogate, FFBS
├── simulation/              # simulate(), simulate_paths()
├── output/                  # accessors, variance/
└── utilities/               # flatten, reconstructor, spline_utils, bounds
```

### Key Directory: inference/smoothing_selection/

Performance iteration implementation (IN DEVELOPMENT):
- `performance_iteration.jl` - Wood (2024) algorithm
- `pijcv.jl` - PIJCV criterion with derivatives
- `implicit_diff.jl` - d(beta_hat)/d_lambda computation
- `dispatch_*.jl` - Data-type specific dispatchers

---

## Part 3: Type System

### Abstract Hierarchy

```
HazardFunction (user-facing)
├── ParametricHazard     # :exp, :wei, :gom
├── SplineHazard         # :sp
└── PhaseTypeHazard      # :pt

_Hazard (internal)
├── _MarkovHazard: MarkovHazard, PhaseTypeCoxianHazard
└── _SemiMarkovHazard: SemiMarkovHazard, RuntimeSplineHazard

MultistateProcess -> MultistateModel -> MultistateModelFitted
```

### Key Struct: MultistateModel

```julia
struct MultistateModel <: MultistateProcess
    data::DataFrame
    parameters::NamedTuple       # (flat, nested, reconstructor)
    hazards::Vector{<:_Hazard}
    tmat::Matrix{Int64}          # Transition matrix
    hazkeys::Dict{Symbol,Int64}  # :h12 -> index
    markovsurrogate, phasetype_surrogate, phasetype_expansion
end
```

### MultistateModelFitted adds:
- `loglik::NamedTuple`, `vcov`, `vcov_type` (:ij, :model, :jk, :none)
- `smoothing_parameters` (lambda per hazard), `edf` (effective degrees of freedom)

---

## Part 4: Key Code Patterns

### Fitting Dispatch

```
fit(model)
  ├─ !is_panel_data -> _fit_exact()
  └─ is_panel_data
       ├─ is_markov -> _fit_markov_panel()
       └─ otherwise -> _fit_mcem()
```

### Parameter Flatten/Unflatten

```julia
rc = model.parameters.reconstructor
flat = flatten(rc, nested_params)       # Standard
nested = unflatten(rc, flat_vector)
flat_dual = flattenAD(rc, params_dual)  # AD-compatible
```

### Hazard Builder Registry

```julia
register_hazard_family!(:wei, _build_weibull_hazard)
builder = _HAZARD_BUILDERS[:wei]
hazard = builder(ctx)
```

### Variance API

```julia
fitted = fit(model; vcov_type=:ij)    # DEFAULT: IJ/sandwich
fitted = fit(model; vcov_type=:model) # Model-based
fitted = fit(model; vcov_type=:none)  # Skip variance
vcov = get_vcov(fitted)
```

---

## Part 5: Critical Gotchas

### Natural Scale Parameters
All parameters are on **natural scale** (not log-transformed). Box constraints (lb >= 0) enforce positivity.

### Monotone Splines Use I-Spline Transform
Optimization params are non-negative increments (ests). Spline coefficients are cumsum: coefs = L * ests. Penalty must transform: S_monotone = L' * S * L. Handled by `transform_penalty_for_monotone()` in spline_utils.jl.

### Phase-Type TPMs Use Schur (Not Eigen)
Eigendecomposition fails for defective matrices. Use `CachedSchurDecomposition` and `compute_tpm_from_schur()`.

### Splines Are Penalized by Default
`fit()` uses `penalty=:auto` which enables PIJCV-selected lambda. Use `penalty=:none` to disable.

---

## Part 6: Test Infrastructure

```
MultistateModelsTests/
├── unit/       # Fast tests (~2 min)
├── integration/# End-to-end tests
├── longtests/  # Statistical validation (~30+ min)
├── fixtures/   # TestFixtures.jl, reference data
└── src/        # Test utilities
```

### Running Tests

```bash
julia --project -e 'using Pkg; Pkg.test()'
julia --project=MultistateModelsTests -e 'include("MultistateModelsTests/unit/test_splines.jl")'
```

---

## Part 7: Active Development - Performance Iteration

**See**: `PERFORMANCE_ITERATION_IMPLEMENTATION.md` (root directory)

### Current State (Nested Optimization)
- Outer loop: search over lambda (50-100 evaluations)
- Inner loop: full optimization of beta per lambda trial (50+ Ipopt iterations)
- Total: ~2500-5000 optimization steps per fit

### Target: Performance Iteration (Wood 2024)
Single loop alternating ONE Newton step for beta and lambda:
```
for iter = 1:maxiter
    # Step 1: One Newton step for beta
    H_lambda = H + lambda*S
    beta_new = beta - H_lambda \ (grad_ell(beta) + lambda*S*beta)
    
    # Step 2: One Newton step for lambda
    lambda_new = lambda - Hess_V(lambda) \ grad_V(lambda)
    
    if converged: break
end
```
Expected: 10-20 total iterations -> 10-50x speedup

### Parallel Workstreams
| Workstream | Scope | Status |
|------------|-------|--------|
| A: PIJCV PI | Core performance iteration | In progress |
| B: EFS/PERF PI | Alternative criteria | Blocked on A |
| C: Critical Issues | Edge cases | Some independent |
| D: Test Infrastructure | Fixtures, benchmarks | Can proceed |

---

## Part 8: File Locations by Task

| Task | Primary File(s) |
|------|-----------------|
| Add hazard family | construction/hazard_builders.jl |
| Modify fitting | inference/fit_exact.jl, fit_markov.jl, fit_mcem.jl |
| Change likelihood | likelihood/loglik_*.jl |
| Modify splines | hazard/spline.jl, utilities/spline_utils.jl |
| Add variance method | output/variance/ |
| Modify phase-type | phasetype/*.jl |
| Smoothing selection | inference/smoothing_selection/ |

---

## Part 9: Validation Commands

```bash
julia --project -e 'using MultistateModels'
julia --project -e 'using Pkg; Pkg.test()'
```

---

## Part 10: Related Skills

| Skill | When to Use |
|-------|-------------|
| stochastic-processes | CTMC, TPM, phase-type, FFBS |
| multistate-survival | Hazards, likelihoods, observation types |
| penalized-likelihood | P-splines, penalties, EDF, performance iteration theory |
| statistical-optimization | Newton, Ipopt, variance estimation, Sherman-Morrison |
| monte-carlo-methods | MCEM, importance sampling, PSIS, Louis identity |
| julia-sciml | Optimization.jl, ForwardDiff, PrecompileTools |
| julia-testing | Test patterns, fixtures |
| error-diagnosis | Debugging common errors |
