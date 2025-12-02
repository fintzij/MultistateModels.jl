---
title: "Test Coverage"
format:
  gfm:
    theme: darkly
    highlight-style: breezedark
    code-block-bg: "#161b22"
    code-block-border-left: "#30363d"
---

# Test Coverage

_Last updated: 2025-06-03 UTC_

## Overview
- `Pkg.test()` loads fixtures via `test/runtests.jl`, so every `include("test_*.jl")` automatically participates in CI.
- The suites below focus on infrastructure-critical surfaces (hazard math, subject data construction, helper utilities, likelihood computation, and model builders) used by higher-level fitting code.
- Long-run simulation or integration tests live under `test/longtest_*.jl`; they provide probabilistic sanity checks but are not the focus of this documentation. See `test/reports/simulation_longtests.md` and `test/reports/inference_longtests.md` for details.
- Update this document whenever a new suite is introduced or existing coverage materially changes.
- Section-level dates (e.g., "Latest run 2025-11-26") indicate when that specific test was last validated; the header date reflects the document's last edit.
- Latest status: `julia --project test/runtests.jl` (2025-06-03 UTC, Julia 1.12.1) reported **763 passed, 0 broken** across the unit test suites. Full test suite (`MSM_TEST_LEVEL=full`) includes 5 long tests.

## Suite Summary

| Suite | Tests | Purpose |
|-------|-------|---------|
| `test_modelgeneration.jl` | 41 | Model construction, validation, macro parity |
| `test_hazards.jl` | 159 | Hazard evaluation, PH/AFT, time transforms |
| `test_helpers.jl` | 119 | Parameter utilities, data accessors |
| `test_make_subjdat.jl` | 34 | Subject data processing, weights |
| `test_simulation.jl` | 84 | Path simulation, state trajectories |
| `test_ncv.jl` | 66 | IJ/Jackknife variance, LOO methods |
| `test_exact_data_fitting.jl` | 74 | Exact data MLE fitting |
| `test_phasetype_is.jl` | 114 | Phase-type importance sampling |
| `test_splines.jl` | 72 | Spline hazard construction/evaluation |
| **Total** | **763** | |

## Suite Details

### `test_modelgeneration.jl`
- **Scope:** calls `build_multistatemodel` and helpers on the 2-, 3-, and 4-state fixture families.
- **Key assertions:** transition-matrix ordering, duplicate transition detection, contiguous state numbering, censoring rules around state `0`, and macro vs constructor parity for hazard parameter vectors.
- **Diagnostics:** uses `@test_logs` to ensure misconfigured hazards emit actionable warnings instead of silent failures.
- **Latest run:** part of `Pkg.test()` noted above; runtime contribution ≈0.6 s on Apple M2 Max.

### `test_hazards.jl`
- **Scope:** direct evaluation of `hazard_fn`/`cumhaz_fn`, spline endpoints, and cache plumbing used by the likelihood stack.
- **Key assertions:** analytic parity for exponential/Weibull/Gompertz `survprob`, PH vs AFT effects, time transformation toggles, and guardrails preventing spline transforms until support lands.
- **Coverage:** 18 hazard fixtures (3 families × {PH, AFT} × baseline/covariate) plus five spline configs ensure both analytic and shared-cache code paths execute each run.
- **Numerics:** compares log-space results to within `PARAM_RTOL = 1e-6` relative tolerance (accounts for ParameterHandling.jl soft-transform precision) and exercises both per-hazard and shared cache paths.
- **Parameter handling (2025-12-01):** Tests updated to use `get_log_scale_params()` for expected value computation, ensuring compatibility with ParameterHandling.jl's soft-transform bijections.
- **Latest run:** green in the 2025-06-03 `Pkg.test()` sweep.

### `test_helpers.jl`
- **Scope:** helper utilities that mutate models (`set_parameters!`, `set_covariates!`) plus data iterators, and batched likelihood computation.
- **Key assertions:** vector/tuple/named tuple parameter updates propagate without reallocations; `get_subjinds` and related accessors return contiguous spans even with ragged subject panels.
- **Batched likelihood tests (`loglik_exact_batched`):** validates hazard-centric batched likelihood computation matches sequential path-by-path computation for:
  - Two-state exp/wei model with covariates
  - Three-state illness-death model (exp/wei/gom hazards)
  - Weighted subjects (non-uniform SubjectWeights)
  - Negation parameter handling
- **Latest run:** exercised via `Pkg.test()` with no failures.

### `test_make_subjdat.jl`
- **Scope:** ETL layer building subject-level interval datasets from raw observation tables.
- **Key assertions:** covariate-change boundaries get their own rows, degenerate panels remain valid, sojourn accumulation stays monotone, and optional columns (covariate vs baseline-only) are honored.
- **Edge cases:** includes survival data with tied transitions, instant censoring, and covariate-free panels to ensure no double counting.
- **Latest run:** green in the 2025-06-03 `Pkg.test()` run.

### `test_simulation.jl`
- **Scope:** draws single- and multi-subject trajectories through the MCEM simulator, including optimizer hooks and importance weighting.
- **Key assertions:**
  - Error injection verifies optimizer failures propagate descriptive exceptions.
  - Absorbing-start subjects terminate immediately; censoring obeys thresholds.
  - `observe_path`/`extract_paths` round-trip observed data to the latent simulator.
- **Monte Carlo checks:** forty simulated subjects per fixture confirm jump-time ECDFs track analytic references (via KS tests ≥0.9) while per-subject seeds keep determinism.
- **Latest run:** part of the passing `Pkg.test()` sweep.

### `test_ncv.jl`
- **Scope:** Variance estimation infrastructure for fitted multistate models including infinitesimal jackknife (IJ) and jackknife variance estimators.
- **Test categories:**
  1. **IJ variance (sandwich estimator):** Tests H⁻¹KH⁻¹ computation with K = Σᵢgᵢgᵢᵀ where gᵢ are per-subject score vectors.
  2. **Jackknife variance:** Tests LOO perturbation computation with both `:direct` and `:cholesky` methods.
  3. **LOO method parity:** Verifies `:direct` (Δᵢ = H⁻¹gᵢ) and `:cholesky` (exact rank-k downdates) produce equivalent results.
  4. **Eigenvalue thresholding:** Tests numerical stability for ill-conditioned Hessians.
- **Key infrastructure tested:** `compute_ij_variance`, `compute_jk_variance`, `loo_perturbation_direct`, `loo_perturbation_cholesky`.
- **Latest run (2025-06-03, Julia 1.12.1):** 66 / 66 passing.

### `test_exact_data_fitting.jl`
- **Scope:** MLE fitting for exact (observed transition time) data, testing the complete fitting pipeline without MCEM.
- **Test categories:**
  1. **Gradient computation:** Verifies log-likelihood gradients match finite-difference approximations.
  2. **Parameter recovery:** Fits models to simulated data with known parameters.
  3. **Variance estimation:** Tests both model-based (inverse Hessian) and IJ variance computation.
  4. **Observation weights:** Validates `SubjectWeights` and `ObservationWeights` correctly scale likelihood contributions.
- **Key assertions:** optimizer convergence, gradient accuracy to 1e-6, parameter recovery within 3 SEs.
- **Latest run (2025-06-03, Julia 1.12.1):** 74 / 74 passing.

### `test_phasetype_is.jl`
- **Scope:** Phase-type distribution infrastructure for improved importance sampling in MCEM.
- **Test categories:**
  1. **Phase-type construction:** Tests `PhaseTypeDistribution` struct, Coxian subintensity matrix construction, Titman-Sharples constraints.
  2. **Distribution functions:** Validates `phasetype_mean`, `phasetype_variance`, `phasetype_cdf`, `phasetype_pdf`, `phasetype_hazard`.
  3. **Surrogate building:** Tests `build_phasetype_surrogate`, `build_expanded_tmat`, `build_phasetype_emat`.
  4. **FFBS integration:** Validates forward-filtering-backward-sampling on expanded (n×phases) state space.
  5. **Proposal config:** Tests `ProposalConfig`, `resolve_proposal_config` for `:auto`, `:markov`, `:phasetype` symbols.
- **Key infrastructure tested:** Phase-type approximation of non-exponential sojourn times for improved ESS.
- **Reference:** Titman & Sharples (2010) Biometrics 66(3):742-752.
- **Latest run (2025-06-03, Julia 1.12.1):** 114 / 114 passing.

### `test_splines.jl`
- **Scope:** Spline hazard construction and evaluation via BSplineKit.jl.
- **Test categories:**
  1. **Knot placement:** Tests automatic quantile-based interior knot selection, boundary knot handling.
  2. **Basis construction:** Validates `RecombinedBSplineBasis` with natural boundary conditions.
  3. **Hazard evaluation:** Tests `RuntimeSplineHazard` closures for hazard and cumulative hazard.
  4. **Coefficient transformations:** Validates `_spline_ests2coefs` (log-scale → spline coefficients) and inverse.
  5. **Monotone constraints:** Tests I-spline cumsum transformation for monotone hazards.
  6. **AD compatibility:** Verifies ForwardDiff works through spline evaluation.
- **Key assertions:** Spline evaluations match expected values, cumulative hazard equals analytic integral, extrapolation handles boundary correctly.
- **Latest run (2025-06-03, Julia 1.12.1):** 72 / 72 passing.

### Long Tests (MSM_TEST_LEVEL=full)

Long tests provide statistical validation through simulation studies. See:
- `test/reports/simulation_longtests.md` - simulation distribution validation
- `test/reports/inference_longtests.md` - MCEM and MLE fitting validation

### Setup / Fixture Files
- `test/setup_*.jl` scripts assemble toy models used across suites; each script is scoped to the corresponding suite (e.g., `setup_3state_expwei.jl`).
- `test/fixtures/TestFixtures.jl` exports reusable builders like `toy_expwei_model()`; updates here should be reflected wherever fixtures power assertions.
- **TVC fixtures (added 2025-11-28):** `toy_tvc_exp_model`, `toy_tvc_wei_model`, `toy_tvc_gom_model`, `toy_illness_death_tvc_model`, `toy_semi_markov_tvc_model` provide models with time-varying covariates for testing piecewise hazard computation.
- **Parameter handling (2025-12-01):** All fixtures updated to use ParameterHandling.jl infrastructure. Use `get_log_scale_params(model.parameters)` to extract log-scale parameters for hazard evaluation; use `get_parameters_flat(model)` for optimizer-compatible flat vectors.

### Retired Suites
- The legacy `test/deprecated` folder (manual long tests predating the time transformation solver plus obsolete likelihood/pathfunction suites) has been removed. Recreate those scripts locally if historical behavior must be reproduced.

## Maintenance Checklist
- When adding a suite, describe its intent, primary assertions, and any fixtures it introduces.
- After major refactors, run targeted tests (e.g., `Pkg.test(test_args=["test_hazards"])`) and note new expectations here if they materially expand coverage.
- Keep section ordering mirrored with `runtests.jl` so new contributors can cross-reference quickly.