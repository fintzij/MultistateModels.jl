# Changelog

## [0.2.0] - 2025-12-12

### Summary

Version 0.2.0 represents a major overhaul of MultistateModels.jl with significant improvements to:
- **Performance**: 2.3× speedup and 60% allocation reduction in path sampling
- **Phase-Type Surrogates**: New importance sampling proposals for semi-Markov MCEM
- **Variance Estimation**: Robust IJ/JK sandwich estimators
- **Testing Infrastructure**: 11,124+ tests in standalone repository
- **Documentation**: Comprehensive user guide with parameter conventions

This release includes breaking changes to the internal API but maintains backward compatibility for core user-facing functions.

---

## Breaking Changes

### Dependency Changes
- **Removed**: `ArraysOfArrays.jl` 
- **Added**: 
  - `ParameterHandling.jl` - Unified parameter transformation system
  - `ComponentArrays.jl` - Structured parameter containers
  - `ADTypes.jl` - AD backend abstraction
  - `Enzyme.jl`, `Mooncake.jl` - Alternative AD backends
  - `NonlinearSolve.jl` - Root-finding for simulation
  - `HypothesisTests.jl` - Statistical testing utilities
  - `Symbolics.jl` - Symbolic computation support
  - `CairoMakie.jl` - Visualization (diagnostics)
  - `BenchmarkTools.jl` - Performance benchmarking

### Parameter Handling
- Internal parameter storage now uses `ParameterHandling.jl` conventions
- Parameters stored on **estimation scale** (log-transformed for positive params)
- New `get_parameters(model; scale=:natural/:estimation)` API
- Gompertz `shape` parameter is now correctly treated as **unconstrained** (not log-transformed)

### Test Infrastructure
- All tests moved to standalone package: [MultistateModelsTests.jl](https://github.com/fintzij/MultistateModelsTests.jl)
- `test/runtests.jl` is now a thin wrapper that loads the test package
- Environment variable control for long-running tests: `MSM_TEST_LEVEL=full`

---

## New Features

### Phase-Type Importance Sampling (Major Feature)

Phase-type distributions provide improved importance sampling proposals for semi-Markov MCEM, better approximating non-exponential sojourn times (Weibull, Gompertz) compared to simple Markov surrogates.

**New Files**: `src/phasetype.jl` (4,623 lines)

**New Types**:
- `PhaseTypeDistribution` - Coxian phase-type distribution representation
- `PhaseTypeSurrogate` - Expanded state space surrogate for MCEM
- `PhaseTypeModel` - Wrapper model for phase-type expanded state space
- `ProposalConfig` - Unified MCEM proposal configuration
- `PhaseTypeHazardSpec` - User specification for phase-type hazards

**New Functions**:
- `MarkovProposal()`, `PhaseTypeProposal()` - Convenience constructors
- `phasetype_mean()`, `phasetype_variance()`, `phasetype_cv()` - Moments
- `phasetype_cdf()`, `phasetype_pdf()`, `phasetype_hazard()` - Distribution functions
- `phasetype_sample()` - Random sampling
- `build_phasetype_surrogate()` - Construct surrogate from model
- `fit_phasetype_surrogate()` - Fit phase-type to Markov rates

**Usage**:
```julia
# Phase-type proposal with automatic phase selection
fitted = fit(model; proposal=:phasetype)

# Manual phase specification
fitted = fit(model; proposal=PhaseTypeProposal(n_phases=3))

# BIC-based auto-selection
fitted = fit(model; proposal=ProposalConfig(type=:phasetype, n_phases=:auto))
```

**Reference**: Titman & Sharples (2010) Biometrics 66(3):742-752

### Robust Variance Estimation

**New File**: `src/crossvalidation.jl` (2,432 lines)

Three variance estimators now available:

1. **Model-based** (`compute_vcov=true`): Standard inverse Hessian
2. **Infinitesimal Jackknife / Sandwich** (`compute_ij_vcov=true`): `H⁻¹KH⁻¹`
3. **Jackknife** (`compute_jk_vcov=true`): Leave-one-out perturbations

**New Functions**:
- `compute_subject_gradients()` - Per-subject score vectors
- `compute_robust_vcov()` - IJ/JK variance computation
- `get_subject_gradients()`, `get_loo_perturbations()` - Accessor functions
- `get_influence_functions()`, `compare_variance_estimates()` - Diagnostics

**New Options**:
- `loo_method=:direct` (default) or `:cholesky` for LOO computation method
- `vcov_threshold=true` - Eigenvalue thresholding for ill-conditioned Hessians

### SQUAREM Acceleration for MCEM

SQUAREM (Squared Iterative Methods) accelerates MCEM convergence via quasi-Newton updates.

**New Functions in `src/mcem.jl`**:
- `squarem_step_length()` - Compute acceleration step
- `squarem_accelerate()` - Apply accelerated update
- `squarem_should_accept()` - Acceptance criterion

**Usage**:
```julia
fitted = fit(model; acceleration=:squarem)
```

**Reference**: Varadhan & Roland (2008) Scand J Stat 35(2):335-353

### Spline Hazards (Enhanced)

M-spline hazards with automatic knot placement, monotonicity constraints, and time transformation caching.

**New Types**:
- `RuntimeSplineHazard` - Immutable spline hazard with closures
- `SplineHazard` (user-facing specification)

**New Functions**:
- `place_interior_knots()` - Quantile-based knot placement
- `place_knots_from_paths!()` - Data-driven knot calibration
- `calibrate_splines()`, `calibrate_splines!()` - User-facing knot calibration
- `default_nknots()` - Tang et al. formula: `floor(n^(1/5))`

**Usage**:
```julia
# Cubic M-spline with monotone increasing constraint
h12 = Hazard(@formula(0 ~ 1), :sp, 1, 2;
             degree=3,
             knots=[0.5, 1.0, 1.5],
             monotone=1,
             natural_spline=true)

# Automatic knot placement
h21 = Hazard(@formula(0 ~ 1), :sp, 2, 1; knots=nothing)
```

### AD Backend Selection

Support for multiple automatic differentiation backends:

**New Types**:
- `ADBackend` (abstract)
- `ForwardDiffBackend` (default)
- `EnzymeBackend`
- `MooncakeBackend`

**Usage**:
```julia
fitted = fit(model; adbackend=ForwardDiffBackend())  # Default
fitted = fit(model; adbackend=EnzymeBackend())       # Alternative
```

### Simulation Strategies

New pluggable strategies for event time simulation:

**New Types**:
- `CachedTransformStrategy` - Precomputes hazard values (default)
- `DirectTransformStrategy` - On-demand computation
- `OptimJumpSolver`, `ExponentialJumpSolver`, `HybridJumpSolver` - Jump time solvers

**Usage**:
```julia
path = simulate_path(model, subj; strategy=CachedTransformStrategy())
path = simulate_path(model, subj; strategy=DirectTransformStrategy())
```

### Thread-Local Workspaces (Performance)

Pre-allocated workspaces for hot path operations:

**New Types**:
- `PathWorkspace` - Thread-local storage for path sampling
- `TVCIntervalWorkspace` - Thread-local storage for TVC interval computation

**New Functions**:
- `get_path_workspace()` - Get/create thread-local workspace
- `get_tvc_workspace()` - Get/create TVC workspace

**Performance Impact**:
| Metric | v0.1.0 | v0.2.0 | Improvement |
|--------|--------|--------|-------------|
| Time (100×100 paths) | 1.04s | 455ms | **2.3× faster** |
| Memory | 413 MiB | 169 MiB | **59% reduction** |
| Allocations | 10.2M | 4.1M | **60% reduction** |

---

## API Improvements

### New Exports

```julia
# Model accessors
get_parameters, get_vcov, get_loglik, get_parnames
get_pseudovalues, get_convergence_records, get_expanded_parameters

# Variance diagnostics
get_subject_gradients, get_loo_perturbations
get_influence_functions, compare_variance_estimates

# Phase-type accessors
is_phasetype_fitted, get_phasetype_parameters
get_mappings, get_original_data, get_original_tmat, get_convergence

# Simulation strategies
OptimJumpSolver, ExponentialJumpSolver, HybridJumpSolver
CachedTransformStrategy, DirectTransformStrategy

# AD backends
ADBackend, ForwardDiffBackend, EnzymeBackend, MooncakeBackend

# MCEM proposals
ProposalConfig, MarkovProposal, PhaseTypeProposal

# Phase-type types
PhaseTypeDistribution, PhaseTypeModel

# Utilities
get_physical_cores, recommended_nthreads
calibrate_splines, calibrate_splines!
path_to_dataframe, paths_to_dataset
```

### Enhanced `fit()` Signature

```julia
# Markov models
fit(model::MultistateMarkovModel;
    constraints = nothing,
    verbose = true,
    solver = nothing,
    adbackend = ForwardDiffBackend(),
    compute_vcov = true,
    vcov_threshold = true,
    compute_ij_vcov = true,
    compute_jk_vcov = false,
    loo_method = :direct)

# Semi-Markov models (MCEM)
fit(model::MultistateSemiMarkovModel;
    proposal = :auto,                    # NEW: :auto, :markov, :phasetype, or ProposalConfig
    constraints = nothing,
    solver = nothing,
    maxiter = 100,
    tol = 1e-2,
    ascent_threshold = 0.1,
    stopping_threshold = 0.1,
    ess_increase = 2.0,
    ess_target_initial = 50,
    max_ess = 10000,
    max_sampling_effort = 20,
    npaths_additional = 10,
    block_hessian_speedup = 2.0,
    acceleration = :none,               # NEW: :none or :squarem
    verbose = true,
    return_convergence_records = true,
    return_proposed_paths = false,
    compute_vcov = true,
    vcov_threshold = true,
    compute_ij_vcov = true,             # NEW
    compute_jk_vcov = false,            # NEW
    loo_method = :direct)               # NEW
```

---

## Code Organization

### New Files (11 files, ~10,000+ lines)

| File | Lines | Purpose |
|------|-------|---------|
| `src/phasetype.jl` | 4,623 | Phase-type distributions and surrogates |
| `src/crossvalidation.jl` | 2,432 | Robust variance estimation |
| `src/types/hazards.jl` | 666 | Consolidated hazard type definitions |
| `src/types/data.jl` | 286 | Data structure types |
| `src/types/configuration.jl` | 234 | Configuration types |
| `src/types/models.jl` | 227 | Model type definitions |
| `src/pathfunctions.jl` | 403 | SamplePath operations |
| `src/macros.jl` | 110 | `@hazard` macro |
| `src/types/utilities.jl` | 101 | Utility types |
| `src/types/surrogates.jl` | 58 | Surrogate types |
| `src/statsutils.jl` | 44 | Statistical utilities |

### Significantly Expanded Files

| File | v0.1.0 | v0.2.0 | Change |
|------|--------|--------|--------|
| `src/common.jl` | 424 | 1,600 | +1,176 lines |
| `src/hazards.jl` | 751 | 1,762 | +1,011 lines |
| `src/sampling.jl` | 826 | 2,200 | +1,374 lines |
| `src/likelihoods.jl` | 391 | 2,156 | +1,765 lines |
| `src/simulation.jl` | 202 | 1,289 | +1,087 lines |
| `src/helpers.jl` | ~400 | 1,965 | +1,565 lines |
| `src/modelfitting.jl` | ~500 | 1,683 | +1,183 lines |
| `src/smooths.jl` | ~200 | 928 | +728 lines |

### Total Source Code

| Category | v0.1.0 | v0.2.0 |
|----------|--------|--------|
| Source files | 16 | 27 |
| Total lines | ~5,500 | ~27,000 |

---

## Documentation

### New Documentation Pages

- `docs/src/index.md` - Comprehensive user guide (expanded from 1 line to 238 lines)
  - Hazard family specifications with formulas
  - Parameter scale conventions
  - Spline hazard options
  - Simulation strategies
  - Covariate modeling

- `docs/src/optimization.md` (133 lines)
  - Solver selection guidelines
  - Variance estimation methods
  - LOO perturbation methods
  - Diagnostic recommendations

- `docs/src/phasetype_ffbs.md` (204 lines)
  - Phase-type FFBS algorithm details
  - Importance weight computation

### Parameter Scale Conventions (Documented)

| Family | Parameter | Estimation Scale | Natural Scale |
|--------|-----------|-----------------|---------------|
| Exponential | rate | log(rate) | rate |
| Weibull | shape | log(shape) | shape |
| Weibull | scale | log(scale) | scale |
| Gompertz | shape | shape | shape (unconstrained) |
| Gompertz | rate | log(rate) | rate |
| All | covariate β | β | β |

---

## Testing

### Test Infrastructure Reorganization

Tests moved to standalone package: [MultistateModelsTests.jl](https://github.com/fintzij/MultistateModelsTests.jl)

### Test Coverage Summary

| Category | Tests | Description |
|----------|-------|-------------|
| Unit Tests | 1,149 | Fast tests (~2 min) |
| Exact Data Fitting | 45 | Exact Markov inference |
| MCEM Parametric | 45 | Exp/Weibull/Gompertz |
| MCEM Splines | 45 | M-spline hazards |
| MCEM TVC | 38 | Time-varying covariates |
| Simulation Distribution | 65 | Event time correctness |
| Simulation TVC | 9,702 | TVC piecewise validation |
| Phase-Type | 35 | Importance sampling |
| **Total** | **11,124+** | All passing |

### Test Categories

- **Unit tests**: Hazard functions, model generation, simulation mechanics
- **Integration tests**: Parallel likelihood, parameter ordering
- **Long tests**: MCEM convergence, simulation distributions, variance estimation

### Running Tests

```bash
# Quick unit tests
julia --project=. -e 'using Pkg; Pkg.test()'

# Full test suite
MSM_TEST_LEVEL=full julia --project=. -e 'using Pkg; Pkg.test()'

# Specific long test category
MSM_LONGTEST_PHASETYPE=1 MSM_TEST_LEVEL=full julia --project=. -e 'using Pkg; Pkg.test()'
```

---

## Bug Fixes

### Gompertz Parameter Handling
- **Fixed**: Gompertz `shape` parameter was incorrectly being log-transformed in some test fixtures
- **Correct**: Gompertz shape is **unconstrained** (can be negative, zero, or positive)
- **Impact**: Simulation TVC tests now correctly validate against analytic formulas

### Cumulative Hazard Formulas
- **Fixed**: Gompertz cumulative hazard formula corrected to `H(t) = (rate/shape) × (exp(shape×t) - 1)`
- **Fixed**: Piecewise cumhaz TVC formulas were missing `1/shape` factor

### Spline Covariate Extraction
- **Fixed**: `extract_covar_names()` now correctly filters spline basis parameters (sp1, sp2, ...)

---

## Migration Guide

### From v0.1.0 to v0.2.0

1. **No breaking changes** to core user API (`fit()`, `simulate()`, `multistatemodel()`)

2. **New recommended variance estimation**:
   ```julia
   # v0.1.0
   fitted = fit(model)
   vcov = get_vcov(fitted)  # Model-based only
   
   # v0.2.0 (recommended)
   fitted = fit(model; compute_ij_vcov=true)
   vcov_robust = get_vcov(fitted; type=:ij)  # Sandwich variance
   ```

3. **Phase-type proposals for semi-Markov** (optional but recommended):
   ```julia
   # v0.1.0 behavior (still works)
   fitted = fit(model)  # Uses Markov surrogate
   
   # v0.2.0 with phase-type (better ESS)
   fitted = fit(model; proposal=:phasetype)
   ```

4. **Parameter access**:
   ```julia
   # Both scales now available
   get_parameters(fitted; scale=:natural)     # Human-readable
   get_parameters(fitted; scale=:estimation)  # Internal (log-scale)
   ```

5. **Tests**: If you have custom test code, update paths:
   ```julia
   # Old: include("test/test_hazards.jl")
   # New: Tests are in MultistateModelsTests.jl package
   ```

---

## Contributors

- Jon Fintzi (@fintzij) - Lead developer
- GitHub Copilot - Assisted with infrastructure optimization and documentation

## References

- Titman, A. C., & Sharples, L. D. (2010). Semi-Markov Models with Phase-Type Sojourn Distributions. Biometrics, 66(3), 742-752.
- Varadhan, R., & Roland, C. (2008). Simple and Globally Convergent Methods for Accelerating the Convergence of Any EM Algorithm. Scandinavian Journal of Statistics, 35(2), 335-353.
- flexsurv R package for hazard parameterization conventions
