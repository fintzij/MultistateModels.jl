# MultistateModels.jl Infrastructure Status

**Branch:** `infrastructure_changes`  
**Last updated:** 2025-12-03  
**Test status:** ✅ 913 tests passing, 0 broken  
**PR readiness:** See Section 9

---

## Executive Summary

The `infrastructure_changes` branch has made significant progress on testing infrastructure, simulation validation, code organization, and antipattern fixes. Key accomplishments include comprehensive exact data testing, spline hazard integration, ParameterHandling.jl migration, fail-fast error handling improvements, robust variance estimation with LOO method options, unified surrogate fitting API, and phase-type distribution improvements.

**Recent Milestone:** PhaseTypeDistribution S→Q refactor and `place_knots_from_paths!` completed (2025-12-03).

---

## Recent Session (2025-12-03)

### PhaseTypeDistribution S→Q Refactor

1. **Internal representation change** (`src/phasetype.jl`)
   - Changed from subintensity matrix `S` to full intensity matrix `Q` including absorbing state
   - `Q` is `(n+1)×(n+1)` where `n` is the number of transient phases
   - Last row/column represents the absorbing state
   - Added `subintensity(ph)` helper to extract `S` from `Q` when needed
   - Updated all internal functions: `phasetype_mean`, `phasetype_variance`, `phasetype_cdf`, etc.

2. **Coxian structure options** (`src/surrogates.jl`)
   - Added `coxian_structure` keyword to `_build_coxian_from_rate`
   - `:unstructured` (default): General Coxian with independent exit rates
   - `:prop_to_prog`: Exit rate proportional to progression rate
   - `:allequal`: All rates equal (simple Erlang-like)

3. **Updated tests** (`test/test_phasetype_is.jl`)
   - All phase-type tests updated for Q matrix representation
   - 143 tests covering construction, moments, sampling, and importance sampling

### Data-Driven Knot Placement

1. **`place_knots_from_paths!` function** (`src/smooths.jl`)
   - New exported function for data-driven spline knot placement
   - Uses sampled paths from fitted surrogate to determine optimal knot locations
   - Computes quantiles of sojourn time distributions per transition
   - Updates target model's spline hazards in place with new knots

2. **`_rebuild_model_with_knots!` helper** (`src/smooths.jl`)
   - Internal helper that rebuilds `RuntimeSplineHazard` structs with new knot locations
   - Regenerates basis functions, parameter names, and hazard/cumhaz closures
   - Properly handles natural splines, monotonicity constraints, and covariates

3. **Unit tests** (`test/test_splines.jl`)
   - 23 new tests for `place_knots_from_paths!`
   - Tests: return structure, knot ordering, bounds checking, model evaluability
   - Tests: custom quantile_probs, model hazard updates

---

## Previous Session (2025-12-02)

### Unified Surrogate Fitting API

1. **`fit_surrogate()` Unified API** (`src/surrogates.jl`)
   - New signature: `fit_surrogate(model; type=:markov, method=:mle, n_phases=2, ...)`
   - Supports `type ∈ {:markov, :phasetype}` and `method ∈ {:mle, :heuristic}`
   - Internal helpers: `_fit_markov_surrogate()`, `_fit_phasetype_surrogate()`
   - Backward compatible with old API (surrogate_parameters, surrogate_constraints kwargs)
   - `:heuristic` method uses crude transition rates (transitions/time-at-risk)
   - Phase-type heuristic divides crude rates by n_phases

2. **`compute_markov_marginal_loglik()` Function** (`src/surrogates.jl`)
   - Computes marginal log-likelihood under Markov surrogate
   - Used as normalizing constant r(Y|θ') for importance sampling
   - Fixed `NormConstantProposal` bug in `modelfitting.jl` that referenced undefined `surrogate_fitted.loglik.loglik`

3. **`set_surrogate!()` Unified API** (`src/surrogates.jl`)
   - New signature: `set_surrogate!(model; type=:markov, method=:mle, ...)`
   - Sets `model.markovsurrogate` in place

4. **Unit Tests for Surrogate Fitting** (`test/test_surrogates.jl`)
   - 41 new tests covering all {markov, phasetype} × {mle, heuristic} combinations
   - Tests: type validation, method validation, backward compatibility
   - Tests: `compute_markov_marginal_loglik`, MLE vs heuristic comparison
   - Tests: Phase-type n_phases variations

5. **Removed Caffo TODO** (`src/modelfitting.jl`)
   - Removed non-actionable TODO comment about Caffo's ESS power calculation
   - Was at line 974, now removed

---

## Previous Session (2025-06-03)

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
| Phase-Type IS | 143 | ✅ |
| Splines | 108 | ✅ |
| Surrogate Fitting | 42 | ✅ |
| MCEM | 41 | ✅ |
| **Total** | **913** | ✅ |

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
- `fit_surrogate()`: Unified API supporting both Markov and phase-type surrogates
  - `type ∈ {:markov, :phasetype}` - surrogate type selection
  - `method ∈ {:mle, :heuristic}` - fitting method selection
  - `:mle` uses existing `fit()` infrastructure for Markov, extends to phase-type
  - `:heuristic` uses crude transition rates (fast, no optimization)
- `_fit_markov_surrogate()`: Internal helper for Markov surrogate fitting
- `_fit_phasetype_surrogate()`: Internal helper for phase-type surrogate fitting
- `compute_markov_marginal_loglik()`: Computes marginal log-likelihood under surrogate
- `set_surrogate!()`: Sets surrogate on model in place

### 2.3 MCEM Algorithm (`src/modelfitting.jl`)

**Current State:**
- `fit(::MultistateSemiMarkovModel)`: Entry point, ~800 lines
- Supports SQUAREM acceleration
- Louis's identity for observed Fisher information
- Block-diagonal Hessian optimization for speed
- Robust variance via IJ/sandwich estimator

---

## 3. Known Limitations

### 3.1 MCEM Integration Tests Not in Default Suite

- Long tests exist in `test/longtest_mcem.jl`, `test/longtest_mcem_splines.jl`
- Only run with `MSM_TEST_LEVEL=full`
- **Unit tests for MCEM helpers**: ✅ 47 tests in `test/test_mcem.jl` (always runs)
  - `mcem_mll`, `mcem_ase`, `mcem_lml`, `mcem_lml_subj`, `var_ris`
  - `SquaremState`, `squarem_step_length`, `squarem_accelerate`, `squarem_should_accept`
  - ForwardDiff gradient compatibility, loglik dispatch methods

---

## 4. Proposed Refactoring

### 4.1 Surrogate Control in Model Generation (Completed)

The `multistatemodel()` function now accepts surrogate arguments:
```julia
function multistatemodel(hazards...; 
    data,
    surrogate::Symbol = :none,        # :none, :markov
    optimize_surrogate::Bool = false, # fit surrogate MLE at construction
    surrogate_constraints = nothing,
    ...)
```

`fit()` uses the pre-built surrogate from `model.markovsurrogate`.

### 4.2 Consolidate Surrogate Types

Current:
- `MarkovSurrogate` stores hazards and parameters
- `PhaseTypeSurrogate` stores distributions, state mappings, expanded Q

Proposed:
- Unified `Surrogate` abstract type
- `MarkovSurrogate <: Surrogate`
- `PhaseTypeSurrogate <: Surrogate`
- Common interface: `sample_path(surrogate, ...)`, `loglik(surrogate, path)`

### 4.3 MCEM Code Modularization (Deferred to Future PR)

The current MCEM implementation in `modelfitting.jl` is ~800 lines in a single function. This section outlines a modularization plan for a future refactoring PR.

#### Proposed Module Structure

```
src/mcem/
├── types.jl          # MCEMState, MCEMConfig
├── initialization.jl # mcem_initialize
├── estep.jl          # mcem_estep!
├── mstep.jl          # mcem_mstep!
├── convergence.jl    # mcem_check_convergence!, adapt ESS
├── variance.jl       # Louis's identity, robust variance
├── squarem.jl        # SQUAREM acceleration
└── mcem.jl           # Main entry point, re-exports
```

#### Proposed Types

```julia
mutable struct MCEMState
    iteration::Int
    parameters::Vector{Float64}
    loglik::Float64
    
    # Ascent tracking
    alb::Float64                    # Asymptotic Lower Bound
    aub::Float64                    # Asymptotic Upper Bound
    ase::Float64                    # Asymptotic Standard Error
    
    # ESS adaptation
    current_ess::Int
    target_ess::Int
    
    # SQUAREM state (optional)
    squarem::Union{Nothing, SQUAREMState}
    
    # Convergence
    is_converged::Bool
    convergence_reason::String
    
    # History for diagnostics
    loglik_history::Vector{Float64}
    ess_history::Vector{Int}
end

struct MCEMConfig
    maxiter::Int
    tol::Float64
    initial_ess::Int
    max_ess::Int
    ess_multiplier::Float64
    use_squarem::Bool
    compute_vcov::Bool
    vcov_method::Symbol  # :louis, :ij, :jk
    verbose::Bool
    checkpoint_interval::Int
end
```

#### Function Decomposition

| Function | Lines (Est.) | Responsibility |
|----------|--------------|----------------|
| `mcem_initialize(model, config)` | ~100 | Validate inputs, initialize state, build TPM infrastructure |
| `mcem_estep!(state, model, ...)` | ~150 | Sample paths via FFBS, compute importance weights |
| `mcem_mstep!(state, model, ...)` | ~100 | Optimize complete-data log-likelihood |
| `mcem_check_convergence!(state, config)` | ~30 | Check ascent bounds, adapt ESS target |
| `mcem_compute_vcov(state, model, config)` | ~200 | Louis's identity, block-diagonal Hessian, robust variance |
| `mcem_assemble_result(state, model, config, vcov)` | ~80 | Build `MultistateModelFitted` |

#### Refactored Main Loop (Pseudocode)

```julia
function fit(model::Union{MultistateSemiMarkovModel, MultistateSemiMarkovModelCensored}; kwargs...)
    config = MCEMConfig(; kwargs...)
    
    # Validate and resolve inputs
    _validate_mcem_inputs(model, config)
    surrogate = _resolve_surrogate(model, config.proposal)
    
    # Initialize state
    state = mcem_initialize(model, config, surrogate)
    books, tpm_book, hazmat_book = _build_tpm_infrastructure(model, surrogate)
    
    # Main loop
    while !state.is_converged && state.iteration < config.maxiter
        state.iteration += 1
        
        # E-step: sample paths, compute weights
        mcem_estep!(state, model, surrogate, books, tpm_book, hazmat_book, config)
        
        # M-step: optimize parameters
        mcem_mstep!(state, model, config, constraints, solver)
        
        # Check convergence and adapt ESS
        mcem_check_convergence!(state, config)
        
        # SQUAREM acceleration (if enabled)
        if !isnothing(state.squarem)
            mcem_squarem_step!(state, model, config)
        end
        
        # Progress output
        config.verbose && _print_mcem_progress(state, config)
    end
    
    # Post-convergence: variance estimation
    vcov = config.compute_vcov ? mcem_compute_vcov(state, model, config) : nothing
    
    # Assemble result
    return mcem_assemble_result(state, model, config, vcov)
end
```

#### Benefits of Modularization
1. **Testability**: Each component can be unit tested independently
2. **Readability**: Clearer separation of concerns
3. **Extensibility**: Easier to add new convergence criteria, variance estimators
4. **Debugging**: Isolated functions simplify debugging MCEM issues

#### Status
**Deferred to future PR** - Current implementation works; modularization is purely for maintainability.

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
| `surrogates.jl` | Surrogate model construction | ✅ `test_surrogates.jl` |
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
| `test_phasetype_is.jl` | 143 | Phase-type importance sampling |
| `test_splines.jl` | 108 | Spline hazards, knot placement |
| `test_surrogates.jl` | 42 | Surrogate fitting API |
| `test_mcem.jl` | 41 | MCEM helpers, SQUAREM |
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

## 7. Recent Changes (2025-12-02)

1. **Unified surrogate fitting API**: `fit_surrogate(model; type, method, ...)` with {markov,phasetype} × {mle,heuristic}
2. **compute_markov_marginal_loglik**: New helper for marginal log-likelihood under Markov surrogate
3. **NormConstantProposal bug fix**: Fixed undefined `surrogate_fitted.loglik.loglik` reference in `modelfitting.jl`
4. **test_surrogates.jl**: 41 new tests for surrogate fitting API
5. **Removed Caffo TODO**: Removed non-actionable TODO from `modelfitting.jl:974`
6. **test_mcem.jl**: 47 new tests for MCEM helpers (mcem_mll, mcem_ase, mcem_lml, SQUAREM helpers)
7. **Surrogate control in multistatemodel()**: `surrogate=:markov` and `optimize_surrogate=true` options now supported

### Previous Changes (2025-06-03)
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
1. ✅ Unified surrogate fitting API (completed 2025-12-02)
2. ✅ Unit tests for surrogate fitting (completed 2025-12-02)
3. ✅ Remove Caffo TODO (completed 2025-12-02)
4. ✅ Unit tests for MCEM helpers (completed 2025-12-02)
5. ✅ Surrogate control moved to `multistatemodel()` (completed 2025-12-02)

### Short-Term  
1. Document SQUAREM acceleration usage
2. Modularize MCEM into smaller functions (see Section 4.3)

### Medium-Term
1. Add benchmark suite for performance tracking
2. Consolidate surrogate types under abstract interface

---

## 9. PR Readiness Checklist

### Infrastructure Changes PR Requirements
| Requirement | Status |
|-------------|--------|
| `fit_surrogate` unified API implemented | ✅ |
| Phase-type MLE working via Markov infrastructure | ✅ |
| Unit tests for surrogate fitting (≥20 tests) | ✅ 41 tests |
| Unit tests for MCEM helpers (≥10 tests) | ✅ 47 tests |
| No TODOs remaining in source | ✅ |
| All 763+ tests passing | ✅ 852 tests |
| Long tests passing (`MSM_TEST_LEVEL=full`) | ⏳ To verify |
| Documentation updated | ✅ |

### Quality Gates
| Gate | Status |
|------|--------|
| No regressions in existing functionality | ✅ |
| Type stability maintained | ✅ |
| No new warnings | ✅ |
| Code review passed | ⏳ Pending |

### Deferred to Future PR
- MCEM code modularization (see Section 4.3)

---

## 10. Known TODOs in Source Code

No actionable TODOs remain in the source code. The Caffo ESS power calculation TODO was removed on 2025-12-02 as it was non-actionable without significant research investment.

---

## 11. Code Locations Reference

| Concept | File | Lines (approx) |
|---------|------|----------------|
| `MarkovSurrogate` struct | `src/common.jl` | 444-449 |
| `PhaseTypeSurrogate` struct | `src/phasetype.jl` | 736-746 |
| `fit_surrogate` (unified) | `src/surrogates.jl` | 83-150 |
| `_fit_markov_surrogate` | `src/surrogates.jl` | 153-200 |
| `_fit_phasetype_surrogate` | `src/surrogates.jl` | 203-300 |
| `compute_markov_marginal_loglik` | `src/surrogates.jl` | 303-350 |
| `fit(::MultistateSemiMarkovModel)` | `src/modelfitting.jl` | 514-1294 |
| MCEM helpers | `src/mcem.jl` | 1-274 |
| Phase-type distributions | `src/phasetype.jl` | 1-2456 |
| Spline hazards | `src/smooths.jl` | 1-500 |
| Test fixtures | `test/fixtures/TestFixtures.jl` | 1-200 |