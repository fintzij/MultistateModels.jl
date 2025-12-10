# MultistateModelsTests

Internal test package for MultistateModels.jl.

## Structure

```
MultistateModelsTests/
├── Project.toml                    # Package manifest
├── src/
│   └── MultistateModelsTests.jl    # Main module with runtests()
├── fixtures/
│   └── TestFixtures.jl             # Shared test data generators
├── unit/                           # Quick tests (~2 min)
│   ├── test_hazards.jl
│   ├── test_helpers.jl
│   ├── test_initialization.jl
│   ├── test_mcem.jl
│   ├── test_modelgeneration.jl
│   ├── test_ncv.jl
│   ├── test_phasetype.jl
│   ├── test_reconstructor.jl
│   ├── test_reversible_tvc_loglik.jl
│   ├── test_simulation.jl
│   ├── test_splines.jl
│   ├── test_surrogates.jl
│   └── test_variance.jl            # Variance estimation unit tests
├── integration/                    # Integration tests
│   ├── test_parallel_likelihood.jl
│   └── test_parameter_ordering.jl
└── longtests/                      # Statistical validation (~30+ min)
    ├── longtest_config.jl
    ├── longtest_helpers.jl
    ├── longtest_exact_markov.jl
    ├── longtest_mcem.jl
    ├── longtest_mcem_splines.jl
    ├── longtest_mcem_tvc.jl
    ├── longtest_phasetype.jl
    ├── longtest_phasetype_exact.jl
    ├── longtest_phasetype_panel.jl
    ├── longtest_robust_markov_phasetype.jl
    ├── longtest_robust_parametric.jl
    ├── longtest_simulation_distribution.jl
    ├── longtest_simulation_tvc.jl
    └── longtest_variance_validation.jl  # IJ/JK variance validation
```

## Running Tests

### Standard (via Pkg.test)

```julia
using Pkg
Pkg.test("MultistateModels")
```

By default, this runs quick unit tests only.

### Full Test Suite

Set `MSM_TEST_LEVEL=full` to include long-running statistical validation tests:

```bash
MSM_TEST_LEVEL=full julia --project=. -e 'using Pkg; Pkg.test()'
```

### Selective Long Tests

Run specific long tests by setting environment variables:

```bash
# Run only phase-type long tests
MSM_LONGTEST_PHASETYPE=1 MSM_TEST_LEVEL=full julia --project=. -e 'using Pkg; Pkg.test()'

# Run only MCEM long tests  
MSM_LONGTEST_MCEM=1 MSM_TEST_LEVEL=full julia --project=. -e 'using Pkg; Pkg.test()'
```

Available toggles:
- `MSM_LONGTEST_PHASETYPE` - Phase-type importance sampling validation
- `MSM_LONGTEST_MCEM` - MCEM fitting validation
- `MSM_LONGTEST_SIMULATION` - Simulation distribution tests
- `MSM_LONGTEST_VARIANCE_VALIDATION` - Variance estimation validation

### Long Tests Only

```bash
MSM_LONGTEST_ONLY=1 MSM_TEST_LEVEL=full julia --project=. -e 'using Pkg; Pkg.test()'
```

## Test Categories

### Unit Tests
Fast tests (~2 min total) covering:
- Hazard function correctness (analytic validation)
- Model generation and parsing
- Simulation mechanics
- Phase-type approximations
- Spline construction
- Helper utilities

### Integration Tests
Medium-duration tests covering:
- Parallel likelihood computation
- Parameter ordering consistency

### Long Tests
Statistical validation tests (~30+ min) covering:
- MCEM convergence to true parameters
- Simulation distribution correctness
- Phase-type importance sampling accuracy
- Robust variance estimation
- **Variance-covariance validation** (IJ/JK estimators)

## Variance Validation Tests

The `longtest_variance_validation.jl` test validates the variance-covariance estimation infrastructure:

### What is Tested

1. **IJ vs Model-based Variance**: Under correct model specification, the infinitesimal jackknife (sandwich) variance should approximately equal the model-based (inverse Hessian) variance.

2. **JK = ((n-1)/n) × IJ Relationship**: The jackknife variance is algebraically related to IJ variance by this factor. This is an exact identity, not a statistical property.

3. **Estimated vs Empirical Variance**: Variance estimates are compared against empirical variance computed from 1000 simulation replicates.

4. **95% CI Coverage**: Wald confidence intervals should achieve approximately 95% coverage.

5. **Positive Definiteness**: All variance matrices must have non-negative eigenvalues.

### Variance Estimator Formulas

- **Model-based**: `Var(θ̂) = H⁻¹` (inverse observed Fisher information)
- **IJ (sandwich)**: `Var_{IJ}(θ̂) = H⁻¹ K H⁻¹` where `K = Σᵢ gᵢgᵢᵀ`
- **JK**: `Var_{JK}(θ̂) = ((n-1)/n) × Var_{IJ}(θ̂)`

### Validation Results (1000 replicates)

| Test | Result | Details |
|------|--------|---------|
| IJ vs Model-based (Exponential) | ✅ | Ratio: 0.978 |
| IJ vs Model-based (Weibull) | ✅ | Ratios: [0.982, 0.954] |
| IJ vs Model-based (Markov Panel) | ✅ | Ratios: [1.04, 1.001, 1.007] |
| Model vs Empirical (Exponential) | ✅ | Ratio: 0.965 (SE: 0.0234 vs 0.0238) |
| Model vs Empirical (Weibull) | ✅ | Ratios: [0.991, 0.973] |
| IJ vs Empirical | ✅ | Ratio: 0.922 |
| 95% CI Coverage | ✅ | 94.0% |
| Positive Definiteness | ✅ | All eigenvalues ≥ 0 |

### Running Variance Validation

```bash
# Run only variance validation long test
MSM_LONGTEST_ONLY=variance_validation MSM_TEST_LEVEL=full julia --project=. -e 'using Pkg; Pkg.test()'

# Or directly:
julia --project=. -e 'include("MultistateModelsTests/longtests/longtest_variance_validation.jl")'
```

## Diagnostic Reports

The `diagnostics/` directory contains report generation tools for visual validation of simulation and inference. These generate PNG plots comparing:

- Analytic vs computed hazard/cumulative hazard/survival functions
- Simulated distributions vs theoretical distributions
- Phase-type approximation accuracy

### Regenerating Reports

From Julia:

```julia
include("MultistateModelsTests/src/MultistateModelsTests.jl")
using .MultistateModelsTests
MultistateModelsTests.generate_simulation_diagnostics()
```

Or from command line:

```bash
julia --project=MultistateModelsTests MultistateModelsTests/diagnostics/generate_model_diagnostics.jl
```

Output is saved to `MultistateModelsTests/diagnostics/assets/`. See `diagnostics/README.md` for details.