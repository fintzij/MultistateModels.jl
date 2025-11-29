# MultistateModels.jl Infrastructure Status

**Branch:** `infrastructure_changes`  
**Last updated:** 2025-11-27  
**Test status:** ✅ All 330 tests passing

---

## Executive Summary

The `infrastructure_changes` branch has made significant progress on testing infrastructure, simulation validation, and code organization. The core simulation machinery for analytic hazard families (Exponential, Weibull, Gompertz) is now thoroughly validated with both unit tests and visual diagnostics. Key accomplishments include a comprehensive diagnostic suite, modernized test fixtures, and time-varying covariate support verification.

---

## 1. Completed Work

### 1.1 Test Infrastructure Overhaul

**Test Fixtures Module** (`test/fixtures/TestFixtures.jl`)
- Created centralized fixture module with reusable model factories
- Factories: `toy_two_state_exp_model`, `toy_absorbing_start_model`, `toy_expwei_model`, `toy_weiph_model`, `toy_gompertz_model`
- Helper data generators: `duplicate_transition_data`, `noncontiguous_state_data`, `negative_duration_data`, etc.
- Eliminates setup code duplication across test files

**Test Suite Organization** (`test/runtests.jl`)
- Consolidated entry point runs all suites via `Pkg.test()`
- Test files: `test_modelgeneration.jl`, `test_hazards.jl`, `test_helpers.jl`, `test_make_subjdat.jl`, `test_simulation.jl`
- Deterministic RNG seeding for reproducible CI failures
- Legacy longtests archived to `test/archive/` for reference

**Current Test Coverage:**
| Suite | Tests | Status |
|-------|-------|--------|
| Model Generation | 67 | ✅ |
| Hazards | 45 | ✅ |
| Helpers | 38 | ✅ |
| Subject Data | 52 | ✅ |
| Simulation | 128 | ✅ |
| **Total** | **330** | ✅ |

### 1.2 Simulation Diagnostics Suite

**Generator Script** (`test/diagnostics/generate_model_diagnostics.jl`)
- Automated diagnostic plot generation for 14 scenarios
- Compares analytic vs computed hazard/cumulative hazard/survival functions
- Validates simulation distributions via ECDF comparison
- Time-transform parity verification (`ΔF(t) = 0` confirms solver correctness)
- KS statistic convergence plots at sample sizes 100, 200, 500, 1k, 2k, 5k, 10k, 20k, 40k

**Scenarios Validated:**
| Family | Effect | Covariate Mode | Function Panel | Simulation Panel |
|--------|--------|----------------|----------------|------------------|
| Exp | PH | baseline | ✅ | ✅ |
| Exp | PH | covariate | ✅ | ✅ |
| Exp | PH | TVC | ✅ | ✅ |
| Exp | AFT | baseline | ✅ | ✅ |
| Exp | AFT | covariate | ✅ | ✅ |
| Wei | PH | baseline | ✅ | ✅ |
| Wei | PH | covariate | ✅ | ✅ |
| Wei | PH | TVC | ✅ | ✅ |
| Wei | AFT | baseline | ✅ | ✅ |
| Wei | AFT | covariate | ✅ | ✅ |
| Gom | PH | baseline | ✅ | ✅ |
| Gom | PH | covariate | ✅ | ✅ |
| Gom | AFT | baseline | ✅ | ✅ |
| Gom | AFT | covariate | ✅ | ✅ |

**Key Results:**
- All `call_haz`, `call_cumulhaz`, `survprob` outputs lie exactly on analytic curves
- All simulation ECDFs match theoretical CDFs (KS statistics decrease as expected)
- Time-transform parity: `max |ΔF| = 0.0` for all scenarios (cached vs direct strategies produce identical results)
- Piecewise integration for TVC scenarios validated

### 1.3 Documentation

**Reports** (`test/reports/`)
- `distribution_longtests.md`: Comprehensive visual guide with all 28 diagnostic plots inline
- `testcoverage.md`: Test organization and coverage documentation
- Quarto-rendered HTML for easy browsing

### 1.4 Simulation API Improvements

**Solver Strategy Pattern:**
- `BisectionJumpSolver`: Fast, robust bisection with configurable tolerance (default `1e-10`, max 80 iterations)
- `OptimJumpSolver`: Brent's method fallback via Optim.jl for edge cases
- `CachedTransformStrategy` / `DirectTransformStrategy`: Toggle time-transform caching

**Time-Varying Covariate Support:**
- Multi-row per-subject data correctly handled in simulation
- Piecewise hazard integration across covariate change boundaries
- Validated for Exp PH and Wei PH; extensible to other families

---

## 2. Current Architecture Notes

### 2.1 Model Generation (`src/modelgeneration.jl`)

**Current State:**
- `build_hazards`: Single ~300 line function handles formula parsing, intercept injection, parameter naming, runtime function generation, and storage allocation
- `multistatemodel`: ~200 line constructor validates data, enumerates hazards, builds transition/emission matrices, and selects concrete model types

**Identified Concerns:**
- Seven+ responsibilities in `build_hazards` blur intent
- Deep branching in `multistatemodel` complicates error reporting
- String-based family selection (`"exp"`, `"wei"`, `"gom"`, `"sp"`) prevents extension via dispatch

**Potential Refactoring (Not Yet Implemented):**
- Extract per-family builders (`_build_exponential_hazard`, etc.)
- Introduce family-specific spec types for dispatch-based extension
- Split constructor into `_validate_inputs!`, `_build_transition_structures`, `_make_emissions`

### 2.2 Simulation (`src/simulation.jl`)

**Current State:**
- `simulate_path`: Interleaves survival inversion, cache management, RNG draws, solver dispatch, transition sampling, and censoring bookkeeping in one loop
- Boolean flags (`data`, `paths`, `time_transform`) control behavior

**Strengths:**
- Robust jump-time solving with fallback
- Time-transform parity verified to floating-point precision
- TVC handling correct

**Potential Refactoring (Not Yet Implemented):**
- Extract helpers: `_locate_jump_time`, `_advance_interval!`, `_update_covariates!`
- Replace flag-based branching with explicit entry points (`simulate_data`, `simulate_paths`)

---

## 3. Known Limitations

### 3.1 Spline Hazards Not Yet Integrated
- `setup_splines.jl` is commented out in `runtests.jl`
- Spline (`"sp"`) family exists in codebase but not validated on this branch
- **Blocked on:** Integration testing and diagnostic validation

### 3.2 MCEM / Model Fitting Not Tested
- `src/mcem.jl`, `src/modelfitting.jl` exist but are not exercised by current test suite
- Likelihood computation (`src/likelihoods.jl`) has basic coverage but not comprehensive
- **Blocked on:** Simulation validation (now complete) → can proceed to likelihood tests

### 3.3 Multi-State Paths (>2 States) Limited Testing
- 3-state fixtures exist (`toy_expwei_model`) but complex path scenarios not deeply tested
- Illness-death, competing risks patterns need dedicated test coverage

### 3.4 Edge Cases
- Very short intervals (near machine epsilon) handled but not stress-tested
- Extreme parameter values (very high/low rates) not systematically validated

---

## 4. Next Steps

### 4.1 Immediate Priority: Merge Readiness

1. **Review and finalize diagnostic plots** — Ensure KS convergence plots look correct
2. **Update `distribution_longtests.md`** — Verify all images render in Quarto preview
3. **Run full test suite on CI** — Confirm cross-platform compatibility
4. **Clean up scratch files** — Archive or remove development artifacts

### 4.2 Short-Term: Likelihood Validation

1. **Add `test_likelihoods.jl`** — Unit tests for log-likelihood computation
2. **Create likelihood diagnostic script** — Compare computed vs analytic log-likelihoods for known parameter values
3. **Test observation types** — Exact (1), right-censored (2), interval-censored (3), exactly observed alive (4)

### 4.3 Medium-Term: Spline Integration

1. **Re-enable `setup_splines.jl`** — Fix any breaking changes from infrastructure updates
2. **Add spline scenarios to diagnostics** — Validate monotone/non-monotone spline hazards
3. **Test spline + covariate interactions**

### 4.4 Medium-Term: Model Fitting

1. **Add `test_modelfitting.jl`** — Smoke tests for `fit!`, `mcem`
2. **Create fitting validation script** — Recover known parameters from simulated data
3. **Test convergence diagnostics**

### 4.5 Long-Term: Refactoring

1. **Extract per-family hazard builders** — Improve testability and extensibility
2. **Introduce HazardSpec types** — Enable dispatch-based family registration
3. **Modularize `simulate_path`** — Separate concerns for easier debugging
4. **Add benchmark suite** — Track performance across commits

---

## 5. File Inventory

### Source Files (`src/`)
| File | Purpose | Test Coverage |
|------|---------|---------------|
| `MultistateModels.jl` | Module entry point | — |
| `common.jl` | Shared types/constants | Indirect |
| `hazards.jl` | Hazard struct, `call_haz`, `call_cumulhaz` | ✅ `test_hazards.jl` |
| `helpers.jl` | Utility functions | ✅ `test_helpers.jl` |
| `initialization.jl` | Parameter initialization | Partial |
| `likelihoods.jl` | Log-likelihood computation | Partial |
| `mcem.jl` | MCEM algorithm | ❌ Not tested |
| `miscellaneous.jl` | Misc utilities | ✅ `test_miscellaneous.jl` |
| `modelfitting.jl` | `fit!` entry point | ❌ Not tested |
| `modelgeneration.jl` | `multistatemodel`, `build_hazards` | ✅ `test_modelgeneration.jl` |
| `modeloutput.jl` | Output formatting | Partial |
| `pathfunctions.jl` | Path utilities | Indirect |
| `sampling.jl` | MCMC sampling | ❌ Not tested |
| `simulation.jl` | `simulate`, `simulate_path` | ✅ `test_simulation.jl` |
| `smooths.jl` | Spline basis functions | ❌ Not tested |
| `surrogates.jl` | Surrogate hazards | Indirect |

### Test Files (`test/`)
| File | Tests | Purpose |
|------|-------|---------|
| `runtests.jl` | — | Entry point |
| `test_modelgeneration.jl` | 67 | Model construction, validation |
| `test_hazards.jl` | 45 | Hazard evaluation |
| `test_helpers.jl` | 38 | Utility functions |
| `test_make_subjdat.jl` | 52 | Subject data processing |
| `test_simulation.jl` | 128 | Path simulation |
| `fixtures/TestFixtures.jl` | — | Shared fixtures |
| `diagnostics/generate_model_diagnostics.jl` | — | Visual diagnostics |

### Diagnostic Assets (`test/diagnostics/assets/`)
- 14 function panel PNGs
- 14 simulation panel PNGs
- Symlinked to `test/reports/assets/` for Quarto

---

## 6. Commands Reference

```bash
# Run all tests
julia --project -e 'using Pkg; Pkg.test()'

# Run specific test file
julia --project -e 'using Pkg; Pkg.test(test_args=["test_simulation"])'

# Generate diagnostic plots
julia --project test/diagnostics/generate_model_diagnostics.jl

# Render documentation
cd test/reports && quarto render distribution_longtests.md
```

---

## 7. Observation-Level Weights and Emission Probabilities

### 7.1 Motivation
Extend the model to support:
1. **Observation-level weights** in addition to subject-level weights
2. **Emission probability matrices** with float values (not just 0/1 censoring patterns)

### 7.2 Weight Types

**SubjectWeights** (formerly SamplingWeights):
- Length: `nsubj` (number of subjects)
- Applied at subject level: `ll += subj_ll * SubjectWeights[subj]`

**ObservationWeights**:
- Length: `nrow(data)` (number of observations)
- Applied at observation level within likelihood computation
- **Mutually exclusive with SubjectWeights** - only one can be non-nothing

### 7.3 IJ Variance with Observation-Level Weights

**References**: 
- Parner et al. (2023) "Regression models for clustered interval-censored event-time data"
- R survival package implementation (agreg.fit.c)
- Wood, S.N. (2024) "On Neighbourhood Cross Validation", Section 6

**Key insight from R survival package** (agreg.fit.c, lines 393-396):
```c
for (person=0; person < nused; person++) {
    for (i=0; i<nvar; i++)
        dfbeta[person*nvar +i] *= wt[person];  // weight the residuals
}
```
Then at lines 409-442, weighted residuals are collapsed to cluster (subject) level.

**Concordance with Wood (2024) Section 6:**

Wood's equation (7) for variance estimation with neighbourhoods is:
$$\hat{\mathbf{V}} = \sum_i \boldsymbol{\Delta}_{-i} \sum_{j \in \alpha(i)} \boldsymbol{\Delta}_{-j}^T$$

For subject-level neighbourhoods where $\alpha(i) = \{i\}$ (our case):
- The inner sum collapses: $\sum_{j \in \alpha(i)} \boldsymbol{\Delta}_{-j}^T = \boldsymbol{\Delta}_{-i}^T$
- This reduces to: $\hat{\mathbf{V}} = \sum_i \boldsymbol{\Delta}_{-i} \boldsymbol{\Delta}_{-i}^T$
- Which equals the standard IJ: $\hat{\mathbf{V}} = H^{-1} K H^{-1}$ where $K = \sum_i g_i g_i^T$

**Conclusion**: Wood's Section 6 confirms that when neighbourhoods equal subjects (our case), the standard IJ formula applies. The key is to properly aggregate weighted observation scores to the subject level first.

**Mathematical Formulation**:

Standard IJ variance: $\text{Var}(\hat{\theta}) = H^{-1} K H^{-1}$

where:
- $H$ = Hessian (second derivative of weighted log-likelihood)
- $K = \sum_i g_i g_i^T$ = outer product of subject-level scores

With **observation-level weights**:
1. Compute per-observation score: $U_j = \nabla_\theta \ell_j(\theta)$
2. Weight the score: $U_j^w = w_j \times U_j$
3. Aggregate to subject level: $g_i = \sum_{j \in \text{subject } i} U_j^w$
4. Compute dfbeta: $\Delta_i = g_i \times H^{-1}$
5. Robust variance: $\text{Var} = \sum_i \Delta_i \Delta_i^T = H^{-1} K H^{-1}$

**Critical finding**: The IJ formula itself does NOT change. Observation weights affect:
- How the Hessian $H$ is computed (weighted second derivatives)
- How subject-level scores $g_i$ are computed (aggregation of weighted observation scores)

The computation structure remains: aggregate to subject level, then apply standard IJ formula.

### 7.4 Emission Probability Matrix

**EmissionMatrix**: Matrix of size `nrow(data) × nstates`
- Values are $P(\text{observation} | \text{state})$, i.e., emission probabilities
- Do NOT need to sum to 1 across states
- Zero means "impossible to observe this if in that state"
- Float values allow soft censoring (e.g., "90% confident observed state 2")

**CensoringPatterns** extended to accept Float64:
- Previously: `Matrix{Int64}` with 0/1 values
- Now: `Matrix{Float64}` allowing values in [0, 1]

**emat type change**:
- Previously: `Matrix{Int64}`
- Now: `Matrix{Float64}`

### 7.5 Implementation Changes Required

1. **common.jl**: ✅ COMPLETED
   - Rename `SamplingWeights` → `SubjectWeights` in all 6 model types
   - Add `ObservationWeights::Union{Nothing,Vector{Float64}}` field
   - Change `emat::Matrix{Int64}` → `emat::Matrix{Float64}`
   - Change `CensoringPatterns::Matrix{Int64}` → `CensoringPatterns::Matrix{Float64}`

2. **helpers.jl**: ✅ COMPLETED
   - Rename `check_SamplingWeights` → `check_SubjectWeights`
   - Add `check_ObservationWeights` function
   - Add mutual exclusivity validation
   - Update `check_data!` to accept `Matrix{<:Real}` for emat

3. **modelgeneration.jl**: ✅ COMPLETED
   - Update `_normalize_sampling_weights` → `_normalize_subject_weights`
   - Add `_normalize_observation_weights` function
   - Update `build_emat` to return Float64, accept EmissionMatrix argument
   - Update `_prepare_censoring_patterns` for Float64

4. **likelihoods.jl**: ✅ COMPLETED (partial - field renames only)
   - Updated `SamplingWeights` → `SubjectWeights` references
   - TODO: Handle ObservationWeights in likelihood computation
   - TODO: Change `emat[i,:] .== 1` to `emat[i,:] .> 0` checks
   - TODO: Multiply by emission probabilities in likelihood computation

5. **crossvalidation.jl**: ✅ COMPLETED (partial - field renames only)
   - Updated `SamplingWeights` → `SubjectWeights` references
   - TODO: Update IJ computation to handle ObservationWeights (aggregate weighted scores to subject)

6. **modelfitting.jl**: ✅ COMPLETED
   - Updated `SamplingWeights` → `SubjectWeights` references

7. **modeloutput.jl**: ✅ COMPLETED
   - Updated `SamplingWeights` → `SubjectWeights` references

8. **initialization.jl**: ✅ COMPLETED
   - Updated `SamplingWeights` → `SubjectWeights` references

9. **mcem.jl**: ✅ COMPLETED
   - Updated `SamplingWeights` → `SubjectWeights` in function signatures

10. **surrogates.jl**: ✅ COMPLETED
    - Updated `SamplingWeights` → `SubjectWeights` references

11. **test/fixtures/TestFixtures.jl**: ✅ COMPLETED
    - Updated field names and added ObservationWeights field

### 7.6 Test Status

All 396 tests pass after renaming `SamplingWeights` → `SubjectWeights` and adding `ObservationWeights` field.

---

## 8. Original Code Review Notes (Preserved)

The following observations from the initial code review remain relevant for future refactoring:

### 8.1 `build_hazards` Responsibilities
- Current state: single function parses formulas, injects intercepts, names parameters, generates runtime functions, allocates storage, and assembles ParameterHandling views
- Risk: Seven+ responsibilities blur intent and make incremental changes hard to reason about
- Suggestion: extract helpers per family plus a shared "schema → design matrix" helper

### 8.2 Boolean-Driven Composition in Simulation APIs
- Current state: `simulate` relies on `data`, `paths`, and `time_transform` booleans
- Risk: Flag explosion creates dead branches and unclear invariants
- Suggestion: expose explicit entry points and thread behavior through strategy objects

### 8.3 DIP/OCP Pressure
- Current state: hazard family selection via string comparisons with copy-pasted blocks
- Risk: New baseline types require editing central factory
- Suggestion: introduce family-specific builder types and dispatching helpers
