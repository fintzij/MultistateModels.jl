# MultistateModels.jl Codebase Optimization Plan

**Date**: December 10, 2024  
**Status**: Draft for Review  
**Branch**: `infrastructure_changes`

---

## Decisions & Constraints

| Decision | Resolution | Impact |
|----------|------------|--------|
| Cross-validation | **Keep in package** (not extension) | No architectural change needed |
| AD Backends | **Support ForwardDiff + Mooncake** (+ future Enzyme) | Must maintain mutating & non-mutating paths |
| Breaking Changes | **Acceptable** | Can remove wrappers, rename functions |
| Phase-type | **Optimize** | Full optimization scope |

---

## Executive Summary

This document outlines a systematic plan to simplify, deduplicate, and optimize the MultistateModels.jl codebase. The package supports Markov, semi-Markov, and phase-type multistate models with MCEM estimation, importance sampling, and multiple variance estimation methods. This complexity has introduced redundancy and performance bottlenecks that can be addressed.

**Primary Goals:**
1. **Profile** simulation and inference to identify actual bottlenecks
2. **Consolidate** redundant parameter handling systems
3. **Reduce allocations** in hot paths (likelihood, simulation, sampling)
4. **Unify** AD code paths where possible without sacrificing compatibility
5. **Simplify** API by removing unnecessary wrappers

### Codebase Overview (by line count)

| File | Lines | Purpose | Priority |
|------|-------|---------|----------|
| `phasetype.jl` | 4,598 | Phase-type distributions, surrogates, expanded state models | Medium |
| `crossvalidation.jl` | 2,432 | Cross-validation, leave-one-out methods | Low |
| `likelihoods.jl` | 1,931 | Log-likelihood computation, batched data structures | **High** |
| `sampling.jl` | 1,822 | Importance sampling, FFBS, path drawing | **High** |
| `helpers.jl` | 1,799 | ReConstructor, flatten/unflatten infrastructure | Medium |
| `modelfitting.jl` | 1,693 | fit() methods for all model types | Medium |
| `hazards.jl` | 1,439 | Hazard evaluation, cumulative hazard | **High** |
| `common.jl` | 1,319 | Type definitions, caching infrastructure | Medium |
| `modelgeneration.jl` | 1,198 | Model construction from hazard specs | Low |
| `modeloutput.jl` | 1,176 | Results extraction, summary methods | Low |
| `simulation.jl` | 1,133 | Sample path simulation | **High** |
| `smooths.jl` | 811 | Spline hazard infrastructure | Low |
| `surrogates.jl` | 721 | Markov surrogate fitting | Medium |
| `initialization.jl` | 540 | Parameter initialization | Low |
| `pathfunctions.jl` | 403 | SamplePath operations | Medium |
| `mcem.jl` | 273 | MCEM helper functions, SQUAREM | Low |
| **Total** | **~23,700** | | |

---

## Part 1: Profiling Strategy (DO FIRST)

### 1.1 Parameter Handling Consolidation

**Current State**: Multiple overlapping parameter transformation systems:

1. **`safe_unflatten`** (helpers.jl) - AD-compatible unflatten with family-aware transforms
2. **`prepare_parameters`** (likelihoods.jl) - Dispatch-based normalization
3. **`get_hazard_params`** (hazards.jl) - Extract params for hazard evaluation
4. **`ReConstructor`** (helpers.jl) - Flatten/unflatten closures with buffers
5. **`unflatten`** (ReConstructor method) - Used in modelfitting.jl post-optimization

**Issues**:
- Confusing which function to use in what context
- Parameter scale conventions (log vs natural) documented but scattered
- Redundant transformations in hot paths

**Recommendation**: 
- [ ] **Create a single `ParameterOps` module** with clear hierarchy:
  - `unflatten_nested(flat, model) → NamedTuple` (raw unflatten, no transforms)
  - `unflatten_natural(flat, model) → NamedTuple` (with exp transforms for baseline)
  - Deprecate `prepare_parameters`, `safe_unflatten`, `get_hazard_params` in favor of unified API
- [ ] Document parameter scale conventions in ONE place (e.g., `parameters.md`)

**Priority**: Medium (clarity improvement, moderate refactoring effort)

---

### 1.2 Caching Infrastructure Review

**Current State**: Multiple caching systems for expensive computations:

1. **`TimeTransformCache`/`TimeTransformContext`** (common.jl) - Tang-style memoization
2. **`CachedPathData`** (likelihoods.jl) - Pre-computed DataFrames for paths
3. **`StackedHazardData`/`BatchedODEData`** (likelihoods.jl) - Batched likelihood data
4. **`SharedBaselineTable`** (common.jl) - Shared baseline hazard caching

**Issues**:
- `TimeTransformContext` allocates hash tables per subject; unclear if cache hit rate justifies overhead
- `CachedPathData` and `StackedHazardData` may duplicate some storage
- No profiling data on actual cache hit rates

**Recommendation**:
- [ ] **Profile cache hit rates** during MCEM iterations
- [ ] Consider removing `TimeTransformCache` if hit rate is low (most evaluations at different times)
- [ ] Consolidate `CachedPathData` - may be able to avoid double storage

**Priority**: High (potential performance win if caching overhead exceeds benefit)

---

### 1.3 Simulation Strategy Simplification

**Current State**: Multiple entry points and strategies:

```julia
simulate(model; ...)           # Main entry
simulate_data(model; ...)      # Wrapper → simulate(data=true, paths=false)
simulate_paths(model; ...)     # Wrapper → simulate(data=false, paths=true)
simulate_path(model, subj; ...)  # Single subject
```

Plus strategy types:
- `CachedTransformStrategy` 
- `DirectTransformStrategy`

**Issues**:
- `simulate_data` and `simulate_paths` are thin wrappers that add no value
- Strategy types may be premature abstraction if only `CachedTransformStrategy` is used

**Recommendation**:
- [ ] **Remove `simulate_data` and `simulate_paths`** - direct users to `simulate(data=true, paths=false)`
- [ ] Evaluate if strategy abstraction is needed; if not, inline `CachedTransformStrategy` behavior

**Priority**: Low (cleanup, minimal impact)

---

### 1.4 Hazard Type Hierarchy Simplification

**Current State**: Rich hierarchy for dispatch:

```
_Hazard (abstract)
├── _MarkovHazard (abstract)
│   ├── MarkovHazard
│   └── PhaseTypeCoxianHazard
├── _SemiMarkovHazard (abstract)
│   └── SemiMarkovHazard
└── _SplineHazard (abstract)
    └── RuntimeSplineHazard
```

Plus user-facing spec types:
- `ParametricHazard`, `SplineHazard`, `PhaseTypeHazardSpec`

**Issues**:
- `_SemiMarkovHazard` and `SemiMarkovHazard` may be unnecessarily separate
- Many `@inline` functions dispatch on hazard type; could use traits instead

**Recommendation**:
- [ ] **Consider trait-based dispatch** for `is_markov(h)`, `is_semimarkov(h)`, `has_covariates(h)`
- [ ] Keep current structure if trait conversion is high effort; current design works

**Priority**: Low (style preference, working code)

---

## Part 2: Redundant Code Elimination

### 2.1 Duplicate Likelihood Functions

**Current State**:
- `loglik_markov` - Standard mutating version
- `loglik_markov_functional` - Non-mutating for reverse-mode AD

**Issues**:
- Two implementations to maintain for same functionality
- Only `loglik_markov_functional` is needed if reverse-mode AD is primary path

**Recommendation**:
- [ ] **Benchmark both versions** with ForwardDiff (current default)
- [ ] If performance difference is <10%, remove `loglik_markov` and use functional version everywhere
- [ ] If keeping both, add compile-time dispatch based on AD backend

**Priority**: Medium (maintenance burden reduction)

---

### 2.2 FFBS Implementation Duplication

**Current State**: FFBS functions appear in multiple contexts:

1. `ForwardFiltering!` / `BackwardSampling!` (sampling.jl) - Standard FFBS
2. Phase-type specific FFBS (phasetype.jl, sampling.jl) - Expanded state space

**Issues**:
- Core algorithm is similar, parameterized by state space dimension
- Could unify with `AbstractStateSpace` or dimension parameter

**Recommendation**:
- [ ] **Review if unification is feasible** without performance loss
- [ ] May need to keep separate for type stability with different matrix sizes

**Priority**: Low (moderate refactoring, uncertain benefit)

---

### 2.3 `phasetype.jl` Size Review

At 4,598 lines, this is the largest file by far.

**Current Contents** (estimated):
- `ProposalConfig`, `PhaseTypeProposal` - ~200 lines
- `PhaseTypeDistribution` fitting - ~500 lines
- `PhaseTypeSurrogate` construction - ~800 lines
- Expanded model construction - ~1,000 lines
- FFBS on expanded space - ~500 lines
- State mapping utilities - ~300 lines
- Parameter transformation utilities - ~500 lines
- Fitting integration - ~700 lines

**Recommendation**:
- [ ] **Split into subfiles**:
  - `phasetype/types.jl` - Type definitions
  - `phasetype/fitting.jl` - Distribution fitting
  - `phasetype/surrogate.jl` - Surrogate construction
  - `phasetype/expanded.jl` - Expanded model operations
- [ ] Create `phasetype.jl` as index file that includes subfiles

**Priority**: Medium (maintainability)

---

### 2.4 Cross-Validation Module Review

At 2,432 lines, `crossvalidation.jl` is substantial.

**Questions**:
- Is this functionality actively used?
- Could this be a separate package or extension?

**Recommendation**:
- [ ] **Determine usage frequency** 
- [ ] If rarely used, consider making it a package extension loaded on demand
- [ ] If frequently used, review for optimization opportunities

**Priority**: Low (conditional on usage assessment)

---

## Part 3: Performance Optimization

### 3.1 Profiling Plan

**Goal**: Identify hot paths in simulation and inference.

**Simulation Profiling**:
```julia
using Profile, ProfileView

# Profile simulation
model = ... # setup model
@profile simulate(model; nsim=100)
ProfileView.view()
```

**Key functions to profile**:
1. `simulate_path` - Main simulation loop
2. `_find_jump_time` - Nonlinear solve for jump times
3. `survprob` - Survival probability (cumulative hazard)
4. `eval_hazard` / `eval_cumhaz` - Hazard evaluation

**Inference Profiling**:
```julia
# Profile MCEM iteration
@profile fit(model; ...)
```

**Key functions to profile**:
1. `DrawSamplePaths!` - Importance sampling E-step
2. `loglik_semi_markov` / `loglik_markov` - Likelihood evaluation
3. `compute_hazmat!` / `compute_tmat!` - TPM computation
4. Optimization solve (M-step)

**Deliverable**: Profile flamegraphs with hotspot annotations

---

### 3.2 Suspected Optimization Opportunities

Based on code review:

#### 3.2.1 Jump Time Solver

**Current**: Uses `NonlinearSolve.jl` ITP algorithm per jump.

```julia
function _find_jump_time(solver::OptimJumpSolver, gap_fn, lower, upper)
    prob = IntervalNonlinearProblem(gap_fn, (lower, upper))
    sol = solve(prob, ITP())
    return sol.u
end
```

**Potential Issue**: Problem construction overhead per jump.

**Recommendations**:
- [ ] **Profile problem construction vs solve time**
- [ ] Consider pre-allocating problem and reusing with `remake`
- [ ] For simple exponential hazards, use closed-form inverse CDF

---

#### 3.2.2 DataFrame Operations in Hot Loops

**Current**: `make_subjdat` creates DataFrames in likelihood computations.

**Potential Issue**: DataFrame allocation is expensive.

**Recommendations**:
- [ ] **Use pre-allocated TypedTables or StructArrays** instead of DataFrame
- [ ] Cache subject DataFrames upfront in `CachedPathData`

---

#### 3.2.3 Covariate Extraction

**Current**: `extract_covariates_fast` creates NamedTuples.

```julia
@inline function extract_covariates_fast(subjdat::DataFrameRow, covar_names::Vector{Symbol})
    isempty(covar_names) && return NamedTuple()
    values = Tuple(_lookup_covariate_value(subjdat, cname) for cname in covar_names)
    return NamedTuple{Tuple(covar_names)}(values)
end
```

**Potential Issue**: Tuple/NamedTuple construction in inner loops.

**Recommendations**:
- [ ] **Pre-compute covariate vectors per interval** in `StackedHazardData`
- [ ] Use `SVector` from StaticArrays for fixed-size covariate vectors

---

#### 3.2.4 Matrix Exponential Computation

**Current**: Uses `ExponentialUtilities.jl` with allocated cache.

**Questions**:
- Is cache being reused across all TPM computations?
- Could use scaling-and-squaring for small matrices?

**Recommendations**:
- [ ] **Verify cache reuse** in `loglik_markov`
- [ ] For 2-3 state models, consider closed-form matrix exponential

---

#### 3.2.5 Importance Weight Computation

**Current**: Uses ParetoSmooth.jl with reshaping.

```julia
logweights = reshape(copy(loglik_target[i] - loglik_surrog[i]), 1, length(loglik_target[i]), 1)
psiw = psis(logweights; source = "other")
```

**Potential Issue**: Reshape allocations, PSIS computation overhead.

**Recommendations**:
- [ ] **Profile PSIS vs simple normalization**
- [ ] Consider skipping PSIS for small weight variance

---

### 3.3 Memory Allocation Audit

**Goal**: Identify unexpected allocations in hot paths.

```julia
using BenchmarkTools

# Audit single path simulation
model = ...
@btime simulate_path($model, 1)

# Audit single likelihood evaluation
path = simulate_path(model, 1)
params = get_parameters_flat(model)
@btime loglik_path($params, $path, $model.hazards, $model.totalhazards, $model.tmat)
```

**Red flags to look for**:
- Allocations proportional to number of intervals (should be O(1) for streaming computation)
- Dynamic dispatch in tight loops
- Type instabilities causing box allocations

---

## Part 4: Profiling Implementation

### 4.1 Create Profiling Script

**Location**: `scratch/profile_performance.jl`

```julia
# Profile simulation and inference performance
# Run with: julia --project=. scratch/profile_performance.jl

using MultistateModels
using Profile
using ProfileView  # or PProf for flamegraph export

# Setup test model (3-state semi-Markov with covariates)
# ... model setup code ...

# --- Simulation Profiling ---
println("Profiling simulation...")
Profile.clear()
@profile for _ in 1:10
    simulate(model; nsim=100)
end
ProfileView.view()
# or: ProfileCanvas.@profview simulate(model; nsim=100)

# --- Inference Profiling ---  
println("Profiling inference...")
Profile.clear()
@profile fit(model; niter=10, verbose=false)
ProfileView.view()

# --- Allocation Analysis ---
println("\nAllocation analysis:")
@time simulate(model; nsim=100)
@time fit(model; niter=5, verbose=false)
```

### 4.2 Benchmark Suite

**Location**: `scratch/benchmark_core.jl`

```julia
using BenchmarkTools
using MultistateModels

# Core operation benchmarks
suite = BenchmarkGroup()

suite["simulation"] = BenchmarkGroup()
suite["simulation"]["single_path"] = @benchmarkable simulate_path($model, 1)
suite["simulation"]["100_paths"] = @benchmarkable simulate($model; nsim=100)

suite["likelihood"] = BenchmarkGroup()
suite["likelihood"]["exact_path"] = @benchmarkable loglik_path(...)
suite["likelihood"]["markov_panel"] = @benchmarkable loglik_markov(...)

suite["hazard"] = BenchmarkGroup()
suite["hazard"]["eval_hazard_exp"] = @benchmarkable eval_hazard(...)
suite["hazard"]["eval_cumhaz_wei"] = @benchmarkable eval_cumhaz(...)

# Run and save
results = run(suite; verbose=true)
BenchmarkTools.save("benchmarks.json", results)
```

---

## Part 5: Implementation Priorities

### High Priority (Do First)
1. **Profile caching hit rates** - Determine if `TimeTransformCache` is beneficial
2. **Profile simulation hot path** - Identify top 3 optimization targets
3. **Profile inference hot path** - Identify MCEM bottlenecks
4. **Review DataFrame allocations** - Convert to pre-allocated structures if needed

### Medium Priority (Do After Profiling)
5. **Parameter handling consolidation** - Unify `safe_unflatten`/`prepare_parameters`
6. **Split `phasetype.jl`** - Improve maintainability
7. **Benchmark `loglik_markov` vs `loglik_markov_functional`** - Eliminate one

### Low Priority (Cleanup)
8. **Remove `simulate_data`/`simulate_paths` wrappers** - Simplify API
9. **Review cross-validation usage** - Consider package extension
10. **Trait-based hazard dispatch** - Only if motivated by perf data

---

## Part 6: Testing Strategy

After each optimization:

1. **Run full test suite**: `julia --project=. -e 'using Pkg; Pkg.test()'`
2. **Run long tests**: Selected tests from `test/longtest_*.jl`
3. **Benchmark comparison**: Compare before/after with benchmark suite
4. **Statistical validation**: Verify parameter recovery on simulated data

---

## Appendix: Files Needing Review

| File | Review Focus |
|------|--------------|
| `likelihoods.jl` | Hot path optimization, allocation audit |
| `simulation.jl` | Jump solver efficiency, covariate caching |
| `sampling.jl` | FFBS optimization, importance weight computation |
| `hazards.jl` | Eval function inlining, dispatch overhead |
| `helpers.jl` | ReConstructor usage, potential simplification |
| `phasetype.jl` | File splitting, deduplication with base code |

---

## Decision Points Requiring Input

Before proceeding with implementation, please confirm:

1. **Cross-validation usage**: Is this feature actively used? Consider package extension?
2. **AD backend priority**: Is ForwardDiff the primary path, or should we optimize for reverse-mode?
3. **Breaking changes**: Is API breakage acceptable (e.g., removing `simulate_data`)?
4. **Phase-type priority**: Should phase-type optimization be prioritized or deferred?

---

*End of Plan*
