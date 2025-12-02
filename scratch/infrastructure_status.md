# MultistateModels.jl Infrastructure Status

**Branch:** `infrastructure_changes`  
**Last updated:** 2025-06-03  
**Test status:** ✅ 763 tests passing, 0 broken

---

## Executive Summary

The `infrastructure_changes` branch has made significant progress on testing infrastructure, simulation validation, code organization, and antipattern fixes. Key accomplishments include comprehensive exact data testing, spline hazard integration, ParameterHandling.jl migration, fail-fast error handling improvements, and robust variance estimation with LOO method options.

---

## Recent Session (2025-06-03)

### Documentation Updates
1. **LOO Method Documentation** - Added `loo_method` parameter to all `fit()` docstrings
   - `:direct` (default): Δᵢ = H⁻¹gᵢ, O(p²n) complexity
   - `:cholesky`: Exact H₋ᵢ⁻¹ via rank-k downdates, O(np³) complexity

2. **Optimization Guide** - Created `docs/src/optimization.md`
   - Comprehensive variance estimation documentation
   - LOO method selection criteria
   - Eigenvalue thresholding for numerical stability

3. **Test Documentation Overhaul**
   - Updated `test/reports/testcoverage.md` (763 tests, all passing)
   - Renamed `distribution_longtests.md` → `simulation_longtests.md`
   - Created `test/reports/inference_longtests.md` for MCEM/MLE validation

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
- Test files: `test_modelgeneration.jl`, `test_hazards.jl`, `test_helpers.jl`, `test_make_subjdat.jl`, `test_simulation.jl`, `test_ncv.jl`, `test_exact_data_fitting.jl`, `test_phasetype_is.jl`, `test_splines.jl`
- Deterministic RNG seeding for reproducible CI failures
- Environment variable `MSM_TEST_LEVEL` controls quick vs full test suite

**Current Test Coverage:**
| Suite | Tests | Status |
|-------|-------|--------|
| Model Generation | 41 | ✅ |
| Hazards | 159 | ✅ |
| Helpers | 119 | ✅ |
| Subject Data (make_subjdat) | 34 | ✅ |
| Simulation | 84 | ✅ |
| NCV | 66 | ✅ |
| Exact Data Fitting | 74 | ✅ |
| Phase-Type IS | 114 | ✅ |
| Splines | 72 | ✅ |
| **Total** | **763** | ✅ |

### 1.2 Antipattern Fixes (2025-12-01)

1. **`@error` → `error()` replacement** (13 instances across 5 files)
   - `@error` only logs to stderr; `error()` throws and stops execution
   - Critical for fatal conditions that should halt processing
   
2. **Input validation in `Hazard()` constructor** (`src/modelgeneration.jl`)
   - Validates `statefrom > 0`, `stateto > 0`, `statefrom != stateto`
   - Validates `degree >= 0`, `family ∈ ["exp", "wei", "gom", "sp"]`
   - Fail-fast: catch errors at construction, not fitting
   
3. **DRY: `_update_spline_hazards!()` helper** (`src/smooths.jl`)
   - Extracts common spline remake pattern used in sampling and likelihoods
   - Reduces code duplication
   
4. **Boolean naming conventions**
   - `convergence` → `is_converged`
   - `semimarkov` → `is_semimarkov`
   
5. **`@view` for hot loops** (`src/likelihoods.jl`)
   - Prevents DataFrame row allocation in tight loops
   
6. **`ObservationWeights` bug fix** (`src/helpers.jl`)
   - `check_weight_exclusivity()` now always sets `SubjectWeights = ones(nsubj)` when `nothing`
   - Previously failed when only `ObservationWeights` was provided

### 1.3 Simulation Diagnostics Suite

**Generator Script** (`test/diagnostics/generate_model_diagnostics.jl`)
- Automated diagnostic plot generation for 14 scenarios
- Compares analytic vs computed hazard/cumulative hazard/survival functions
- Validates simulation distributions via ECDF comparison

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

### 1.4 Spline Hazards Integration

- ✅ `setup_splines.jl` enabled in test suite
- ✅ Spline hazard construction and evaluation
- ✅ Monotone splines with I-spline basis
- ✅ Natural boundary conditions
- ✅ `is_separable` trait returns `true` for splines
- ✅ Knot placement (auto and manual)

### 1.5 ParameterHandling.jl Migration

- All model types use `parameters::NamedTuple` with structure:
  ```julia
  parameters = (
      flat = Vector{Float64},        # log-scale flat vector for optimizer
      transformed = NamedTuple,      # positive() wrapped parameters
      natural = NamedTuple,          # exp(flat) - natural scale values
      unflatten = Function           # flat → transformed
  )
  ```
- Consistent `get_parameters_flat()`, `get_parameters()`, `set_parameters!()` API
- Hazard functions receive nested parameters via `nest_params()`

---

## 2. Current Architecture Notes

### 2.1 Model Generation (`src/modelgeneration.jl`)

**Current State:**
- `multistatemodel()`: Main constructor, ~100 lines
- `build_hazards()`: Single ~300 line function handles formula parsing, parameter naming, runtime function generation
- Always creates a `MarkovSurrogate` (line 970)

**Known Issue:** Surrogate is created in model generation but re-created in `fit()` for semi-Markov models. See Section 4.1.

### 2.2 Surrogate Infrastructure (`src/surrogates.jl`)

**Current State:**
- `MarkovSurrogate`: Holds exponential hazards and parameters for importance sampling
- `PhaseTypeSurrogate`: Expanded state space for better sojourn approximation
- `fit_surrogate()`: Fits Markov surrogate via MLE
- `fit_phasetype_surrogate()`: Builds phase-type from Markov rates (HEURISTIC, not MLE)

**Known Issue:** Phase-type parameters are not estimated via MLE. See Section 4.2.

### 2.3 MCEM Algorithm (`src/modelfitting.jl`)

**Current State:**
- `fit(::MultistateSemiMarkovModel)`: Entry point, ~800 lines
- Supports SQUAREM acceleration
- Louis's identity for observed Fisher information
- Block-diagonal Hessian optimization for speed
- Robust variance via IJ/sandwich estimator

---

## 3. Known Limitations

### 3.1 Phase-Type Surrogate Fitting

**Problem:** `fit_phasetype_surrogate()` uses heuristics, not MLE:
```julia
# Build Coxian PH with appropriate rates
# For now, use simple equal-rate phases that match total sojourn rate
phasetype_dists[s] = _build_coxian_from_rate(n_ph, total_rate)
```

**Impact:** Suboptimal importance sampling efficiency.

**Status:** TODO - implement MLE fitting for phase-type distributions

### 3.2 Surrogate Control in Wrong Location

**Problem:** Surrogate creation logic is in `fit()` instead of `multistatemodel()`.

**Current Flow:**
1. `multistatemodel()` creates `MarkovSurrogate` (model.markovsurrogate)
2. `fit()` ignores it and re-creates via `fit_surrogate()`
3. Phase-type surrogate created in `fit()` if requested

**Desired Flow:**
1. `multistatemodel(..., surrogate=:markov)` creates and optionally fits surrogate
2. `fit()` uses pre-built `model.markovsurrogate`

**Status:** TODO - refactor to move surrogate control to model generation

### 3.3 MCEM Not Integrated in Test Suite

- Long tests exist in `test/longtest_mcem.jl`, `test/longtest_mcem_splines.jl`
- Only run with `MSM_TEST_LEVEL=full`
- No dedicated unit tests for MCEM helpers

---

## 4. Proposed Refactoring

### 4.1 Move Surrogate Control to Model Generation

Add `surrogate` argument to `multistatemodel()`:
```julia
function multistatemodel(hazards...; 
    data,
    surrogate::Symbol = :none,        # :none, :markov, :phasetype  
    optimize_surrogate::Bool = true,  # fit surrogate MLE
    surrogate_constraints = nothing,
    phasetype_config = nothing,       # ProposalConfig for :phasetype
    ...)
```

Then `fit()` uses the pre-built surrogate:
```julia
function fit(model::MultistateSemiMarkovModel; ...)
    surrogate = model.markovsurrogate  # already built and optionally fitted
    # ... use surrogate for FFBS sampling
end
```

### 4.2 Implement Phase-Type MLE Fitting

Options:
1. **Moment matching**: Fit Coxian PH to observed sojourn empirical moments
2. **EM algorithm**: Standard EM for phase-type mixture fitting
3. **Spectral methods**: Eigenvalue-based fitting for acyclic PH

### 4.3 Consolidate Surrogate Types

Current:
- `MarkovSurrogate` stores hazards and parameters
- `PhaseTypeSurrogate` stores distributions, state mappings, expanded Q

Proposed:
- Unified `Surrogate` abstract type
- `MarkovSurrogate <: Surrogate`
- `PhaseTypeSurrogate <: Surrogate`
- Common interface: `sample_path(surrogate, ...)`, `loglik(surrogate, path)`

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
| `likelihoods.jl` | Log-likelihood computation | ✅ `test_exact_data_fitting.jl` |
| `mcem.jl` | MCEM helper functions | Long tests only |
| `miscellaneous.jl` | Misc utilities | Indirect |
| `modelfitting.jl` | `fit()` entry points | Long tests only |
| `modelgeneration.jl` | `multistatemodel`, `build_hazards` | ✅ `test_modelgeneration.jl` |
| `modeloutput.jl` | Output formatting | Partial |
| `pathfunctions.jl` | Path utilities | Indirect |
| `sampling.jl` | Path sampling, FFBS | ✅ `test_phasetype_is.jl` |
| `simulation.jl` | `simulate`, `simulate_path` | ✅ `test_simulation.jl` |
| `smooths.jl` | Spline basis functions | ✅ `test_splines.jl` |
| `surrogates.jl` | Surrogate model construction | Indirect |
| `crossvalidation.jl` | IJ/JK variance | Partial |

### Test Files (`test/`)
| File | Tests | Purpose |
|------|-------|---------|
| `runtests.jl` | — | Entry point |
| `test_modelgeneration.jl` | 41 | Model construction, validation |
| `test_hazards.jl` | 159 | Hazard evaluation |
| `test_helpers.jl` | 119 | Utility functions |
| `test_make_subjdat.jl` | 34 | Subject data processing |
| `test_simulation.jl` | 84 | Path simulation |
| `test_ncv.jl` | 66 | Cross-validation, variance estimation |
| `test_exact_data_fitting.jl` | 74 | Exact data MLE |
| `test_phasetype_is.jl` | 114 | Phase-type importance sampling |
| `test_splines.jl` | 72 | Spline hazards |
| `longtest_mcem.jl` | — | MCEM (full suite only) |
| `longtest_mcem_splines.jl` | — | MCEM + splines (full suite only) |
| `longtest_exact_markov.jl` | — | Exact data MLE (full suite only) |
| `longtest_simulation_distribution.jl` | — | Simulation validation (full suite only) |
| `longtest_simulation_tvc.jl` | — | TVC simulation (full suite only) |
| `fixtures/TestFixtures.jl` | — | Shared fixtures |

---

## 6. Commands Reference

```bash
# Run quick tests (default)
julia --project -e 'using Pkg; Pkg.test()'

# Run full tests including MCEM
MSM_TEST_LEVEL=full julia --project -e 'using Pkg; Pkg.test()'

# Generate diagnostic plots
julia --project test/diagnostics/generate_model_diagnostics.jl

# Build documentation
julia --project=docs docs/make.jl
```

---

## 7. Recent Changes (2025-06-03)

1. **LOO method documentation**: Added `loo_method` parameter to all `fit()` docstrings
2. **Optimization guide**: Created `docs/src/optimization.md` with comprehensive variance estimation docs
3. **Test documentation overhaul**:
   - Updated `testcoverage.md` (763 tests)
   - Renamed `distribution_longtests.md` → `simulation_longtests.md`
   - Created `inference_longtests.md` for MCEM/MLE tests
4. **Updated `future_features_todo.txt`**: Added recently completed items

### Previous Changes (2025-12-01)
1. **Antipattern fixes**: `@error`→`error()`, input validation, DRY helpers, boolean naming
2. **ObservationWeights bug**: Fixed mutual exclusivity handling
3. **Spline `is_separable` test**: Replaced `@test_skip` with actual test
4. **Documentation**: Updated `mcem_notes.md` and `infrastructure_status.md`

---

## 8. Next Steps

### Immediate
1. ✅ Update documentation files (completed 2025-06-03)
2. ✅ Document LOO method options for variance estimation
3. Run full test suite to verify no regressions

### Short-Term  
1. Refactor surrogate control to `multistatemodel()`
2. Add unit tests for MCEM helper functions
3. Document SQUAREM acceleration usage

### Medium-Term
1. Implement MLE fitting for phase-type surrogates
2. Add benchmark suite for performance tracking
3. Consolidate surrogate types under abstract interface

---

## 9. Known TODOs in Source Code

Only one TODO remains in the source code:

**`src/modelfitting.jl:974`**
```julia
# TODO: alternatively, use Caffo's power calculation for ESS target
```

This is a potential future enhancement to use Caffo et al. (2005) ascent-based power calculation for determining MCEM sample size, rather than the current adaptive ESS targeting approach.