# MultistateModels.jl Comprehensive Optimization Plan

**Date**: December 2024  
**Branch**: `infrastructure_changes`  
**Status**: Pre-Implementation Planning

---

## Design Decisions (Confirmed)

1. **Cross-validation**: Keep as core functionality, do NOT make a package extension
2. **AD Compatibility**: Must support both ForwardDiff AND reverse-mode (Mooncake now, Enzyme future)
3. **Breaking Changes**: Acceptable
4. **Scope**: All optimizations on the table

---

## Document Structure

| File | Contents |
|------|----------|
| `00_overview.md` | This file - summary and navigation |
| `01_architecture.md` | Parameter handling, type hierarchy, module organization |
| `02_hot_paths.md` | Simulation, likelihood, MCEM critical paths |
| `03_caching.md` | TimeTransformCache, CachedPathData, batched data |
| `04_ad_compatibility.md` | ForwardDiff vs reverse-mode, mutation patterns |
| `05_deduplication.md` | Redundant code, file splitting, API cleanup |
| `06_profiling_plan.md` | Concrete profiling scripts and benchmarks |
| `07_implementation_order.md` | Prioritized task list with dependencies |

---

## Codebase Statistics

### Source Files by Size

| File | Lines | Category |
|------|-------|----------|
| `phasetype.jl` | 4,598 | Phase-type infrastructure |
| `crossvalidation.jl` | 2,432 | Cross-validation methods |
| `likelihoods.jl` | 1,931 | Log-likelihood computation |
| `sampling.jl` | 1,822 | Importance sampling, FFBS |
| `helpers.jl` | 1,799 | ReConstructor, flatten/unflatten |
| `modelfitting.jl` | 1,693 | fit() for all model types |
| `hazards.jl` | 1,439 | Hazard evaluation |
| `common.jl` | 1,319 | Type definitions, caching |
| `modelgeneration.jl` | 1,198 | Model construction |
| `modeloutput.jl` | 1,176 | Results extraction |
| `simulation.jl` | 1,133 | Sample path simulation |
| `smooths.jl` | 811 | Spline hazard infrastructure |
| `surrogates.jl` | 721 | Markov surrogate fitting |
| `initialization.jl` | 540 | Parameter initialization |
| `pathfunctions.jl` | 403 | SamplePath operations |
| `mcem.jl` | 273 | MCEM helpers, SQUAREM |
| `statsutils.jl` | 44 | Truncated distributions |
| `miscellaneous.jl` | 65 | Misc utilities |
| `macros.jl` | 110 | @hazard macro |
| `MultistateModels.jl` | 216 | Main module |
| **Total** | **~23,700** | |

### Key Workflows

1. **Exact Data Fitting** (`fit` on `MultistateMarkovModel`/`MultistateMarkovModelCensored`)
   - Direct likelihood optimization via Ipopt
   - Hot path: `loglik_exact` → `loglik_path` → `survprob` → `eval_hazard`/`eval_cumhaz`

2. **Panel Data (Markov)** (`fit` on `MultistateMarkovModel` with obstype=2)
   - Matrix exponential TPM computation
   - Hot path: `loglik_markov` → `compute_hazmat!` → `compute_tmat!` → TPM lookups

3. **Semi-Markov MCEM** (`fit` on `MultistateSemiMarkovModel`)
   - Importance sampling E-step + optimization M-step
   - Hot paths:
     - E-step: `DrawSamplePaths!` → `draw_samplepath` → FFBS → `sample_ecctmc!`
     - Likelihood: `loglik_semi_markov` → `loglik_path` (per path)
     - M-step: Optimization with `loglik` as objective

4. **Simulation** (`simulate`, `simulate_path`)
   - Jump time root-finding via NonlinearSolve
   - Hot path: `simulate_path` → `survprob` → `_find_jump_time` → `eval_cumhaz`

---

## High-Level Optimization Categories

### A. Architecture Simplification
- Consolidate parameter handling (`safe_unflatten`, `prepare_parameters`, `get_hazard_params`)
- Review caching infrastructure cost/benefit
- Consider file splitting for `phasetype.jl`

### B. Hot Path Optimization
- Profile and optimize simulation jump-time solver
- Reduce DataFrame allocations in likelihood computation
- Optimize covariate extraction in inner loops
- Review matrix exponential caching

### C. AD Compatibility
- Ensure all hot paths work with both ForwardDiff and Mooncake
- Eliminate mutations that break reverse-mode AD (or provide alternatives)
- Profile AD overhead

### D. Deduplication
- Unify `loglik_markov` / `loglik_markov_functional`
- Remove thin wrapper functions (`simulate_data`, `simulate_paths`)
- Consolidate FFBS implementations where possible

### E. Testing & Validation
- Benchmark suite for regression testing
- Statistical validation on simulated data
- Memory allocation audit

---

## Next Steps

1. Read `01_architecture.md` for parameter handling details
2. Read `02_hot_paths.md` for critical path analysis
3. Read `06_profiling_plan.md` for concrete profiling scripts
4. Execute profiling to get baseline measurements
5. Prioritize based on profiling results
