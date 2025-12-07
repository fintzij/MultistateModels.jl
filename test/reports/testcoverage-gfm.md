# Test Coverage


# Test Coverage

*Last updated: 2025-12-05 UTC*

## Overview

- `Pkg.test()` loads fixtures via `test/runtests.jl`, so every
  `include("test_*.jl")` automatically participates in CI.
- The suites below focus on infrastructure-critical surfaces (hazard
  math, subject data construction, helper utilities, likelihood
  computation, and model builders) used by higher-level fitting code.
- Long-run simulation or integration tests live under
  `test/longtest_*.jl`; they provide probabilistic sanity checks but are
  not the focus of this documentation. See
  `test/reports/simulation_longtests.md` and
  `test/reports/inference_longtests.md` for details.
- Update this document whenever a new suite is introduced or existing
  coverage materially changes.
- Section-level dates (e.g., “Latest run 2025-11-26”) indicate when that
  specific test was last validated; the header date reflects the
  document’s last edit.
- Latest status: `julia --project test/runtests.jl` (2025-12-05 UTC,
  Julia 1.12.2) reported **796 passed, 0 broken** across the unit test
  suites. Full test suite (`MSM_TEST_LEVEL=full`) includes 10 long
  tests.

## Suite Summary

| Suite | Tests | Purpose |
|----|----|----|
| `test_modelgeneration.jl` | 41 | Model construction, validation, macro parity |
| `test_hazards.jl` | 159 | Hazard evaluation, PH/AFT, time transforms |
| `test_helpers.jl` | 119 | Parameter utilities, data accessors |
| `test_make_subjdat.jl` | 34 | Subject data processing, weights |
| `test_simulation.jl` | 84 | Path simulation, state trajectories |
| `test_ncv.jl` | 66 | IJ/Jackknife variance, LOO methods |
| `test_phasetype_is.jl` | 76 | Phase-type importance sampling |
| `test_phasetype_correctness.jl` | 48 | Phase-type hazard correctness |
| `test_splines.jl` | 34 | Spline hazard construction/evaluation |
| `test_mcem.jl` | 18 | MCEM estimation infrastructure |
| `test_surrogates.jl` | 22 | Surrogate model construction |
| `test_reversible_tvc_loglik.jl` | 6 | Reversible TVC likelihood (AD compatibility) |
| `test_parallel_likelihood.jl` | 81 | Parallel/batched likelihood computation |
| `test_parameter_ordering.jl` | 32 | Parameter ordering and model consistency |
| **Total** | **796** |  |

## Suite Details

### `test_modelgeneration.jl`

- **Scope:** calls `build_multistatemodel` and helpers on the 2-, 3-,
  and 4-state fixture families.
- **Key assertions:** transition-matrix ordering, duplicate transition
  detection, contiguous state numbering, censoring rules around state
  `0`, and macro vs constructor parity for hazard parameter vectors.
- **Diagnostics:** uses `@test_logs` to ensure misconfigured hazards
  emit actionable warnings instead of silent failures.
- **Latest run:** part of `Pkg.test()` noted above; runtime contribution
  ≈0.6 s on Apple M2 Max.

### `test_hazards.jl`

- **Scope:** direct evaluation of `hazard_fn`/`cumhaz_fn`, spline
  endpoints, and cache plumbing used by the likelihood stack.
- **Key assertions:** analytic parity for exponential/Weibull/Gompertz
  `survprob`, PH vs AFT effects, time transformation toggles, and
  guardrails preventing spline transforms until support lands.
- **Coverage:** 18 hazard fixtures (3 families × {PH, AFT} ×
  baseline/covariate) plus five spline configs ensure both analytic and
  shared-cache code paths execute each run.
- **Numerics:** compares log-space results to within `PARAM_RTOL = 1e-6`
  relative tolerance (accounts for ParameterHandling.jl soft-transform
  precision) and exercises both per-hazard and shared cache paths.
- **Parameter handling (2025-12-01):** Tests updated to use
  `get_log_scale_params()` for expected value computation, ensuring
  compatibility with ParameterHandling.jl’s soft-transform bijections.
- **Latest run:** green in the 2025-06-03 `Pkg.test()` sweep.

### `test_helpers.jl`

- **Scope:** helper utilities that mutate models (`set_parameters!`,
  `set_covariates!`) plus data iterators, and batched likelihood
  computation.
- **Key assertions:** vector/tuple/named tuple parameter updates
  propagate without reallocations; `get_subjinds` and related accessors
  return contiguous spans even with ragged subject panels.
- **Batched likelihood tests (`loglik_exact_batched`):** validates
  hazard-centric batched likelihood computation matches sequential
  path-by-path computation for:
  - Two-state exp/wei model with covariates
  - Three-state illness-death model (exp/wei/gom hazards)
  - Weighted subjects (non-uniform SubjectWeights)
  - Negation parameter handling
- **Latest run:** exercised via `Pkg.test()` with no failures.

### `test_make_subjdat.jl`

- **Scope:** ETL layer building subject-level interval datasets from raw
  observation tables.
- **Key assertions:** covariate-change boundaries get their own rows,
  degenerate panels remain valid, sojourn accumulation stays monotone,
  and optional columns (covariate vs baseline-only) are honored.
- **Edge cases:** includes survival data with tied transitions, instant
  censoring, and covariate-free panels to ensure no double counting.
- **Latest run:** green in the 2025-06-03 `Pkg.test()` run.

### `test_simulation.jl`

- **Scope:** draws single- and multi-subject trajectories through the
  MCEM simulator, including optimizer hooks and importance weighting.
- **Key assertions:**
  - Error injection verifies optimizer failures propagate descriptive
    exceptions.
  - Absorbing-start subjects terminate immediately; censoring obeys
    thresholds.
  - `observe_path`/`extract_paths` round-trip observed data to the
    latent simulator.
- **Monte Carlo checks:** forty simulated subjects per fixture confirm
  jump-time ECDFs track analytic references (via KS tests ≥0.9) while
  per-subject seeds keep determinism.
- **Latest run:** part of the passing `Pkg.test()` sweep.

### `test_ncv.jl`

- **Scope:** Variance estimation infrastructure for fitted multistate
  models including infinitesimal jackknife (IJ) and jackknife variance
  estimators.
- **Test categories:**
  1.  **IJ variance (sandwich estimator):** Tests H⁻¹KH⁻¹ computation
      with K = Σᵢgᵢgᵢᵀ where gᵢ are per-subject score vectors.
  2.  **Jackknife variance:** Tests LOO perturbation computation with
      both `:direct` and `:cholesky` methods.
  3.  **LOO method parity:** Verifies `:direct` (Δᵢ = H⁻¹gᵢ) and
      `:cholesky` (exact rank-k downdates) produce equivalent results.
  4.  **Eigenvalue thresholding:** Tests numerical stability for
      ill-conditioned Hessians.
- **Key infrastructure tested:** `compute_ij_variance`,
  `compute_jk_variance`, `loo_perturbation_direct`,
  `loo_perturbation_cholesky`.
- **Latest run (2025-06-03, Julia 1.12.1):** 66 / 66 passing.

### `test_surrogates.jl`

- **Scope:** Unified surrogate fitting API for Markov and phase-type
  surrogates.
- **Test categories:**
  1.  **Markov surrogate MLE:** Tests
      `fit_surrogate(model; type=:markov, method=:mle)`.
  2.  **Markov surrogate heuristic:** Tests crude rate estimation
      without optimization.
  3.  **Phase-type surrogate:** Tests
      `fit_surrogate(model; type=:phasetype, n_phases=2)`.
  4.  **Marginal likelihood:** Tests `compute_markov_marginal_loglik()`
      for IS normalizing constant.
- **Key infrastructure tested:** `fit_surrogate()`, `set_surrogate!()`,
  `_fit_markov_surrogate()`, `_fit_phasetype_surrogate()`.
- **Latest run (2025-12-04, Julia 1.12.1):** passing.

### `test_mcem.jl`

- **Scope:** MCEM algorithm helpers and SQUAREM acceleration.
- **Test categories:**
  1.  **SQUAREM fixed-point iteration:** Tests acceleration convergence.
  2.  **ESS computation:** Tests effective sample size from importance
      weights.
  3.  **Convergence criteria:** Tests stopping rule based on parameter
      change.
  4.  **Path sampling infrastructure:** Tests FFBS path extraction.
- **Key infrastructure tested:** `squarem_update!`, `compute_ess`,
  `check_convergence`.
- **Latest run (2025-12-04, Julia 1.12.1):** passing.

### `test_reversible_tvc_loglik.jl`

- **Scope:** Likelihood computation for reversible semi-Markov models
  with TVC.
- **Test categories:**
  1.  **Sojourn time resets:** Verifies sojourn resets correctly when
      re-entering states.
  2.  **Manual vs package likelihood:** Compares hand-computed
      likelihood to package output.
  3.  **TVC with multiple sojourns:** Tests covariate handling across
      multiple state visits.
- **Key assertions:** Sojourn times reset on state entry, likelihood
  matches manual calculation.
- **Latest run (2025-12-04, Julia 1.12.1):** 6 / 6 passing.

### `test_parameter_ordering.jl`

- **Scope:** Parameter ordering, storage, and retrieval consistency
  across the package.
- **Test categories:**
  1.  **Transition matrix ordering:** Verifies `get_parameters_flat`
      returns parameters in transition matrix order (row-major).
  2.  **Hazard family parameter order:** Tests exp (1 param), wei/gom (2
      params: shape, scale), spline (n coefficients).
  3.  **set_parameters! consistency:** Named tuple assignment propagates
      correctly to hazard structs.
  4.  **Simulation parameter usage:** Verifies simulated paths use
      correct hazard parameters.
  5.  **Fitting recovery:** Parameters recovered via MLE match true
      values (round-trip).
  6.  **Fitted object storage:** Fitted object stores parameters
      identically to input model.
- **Key assertions:** Parameter ordering is consistent between
  `get_parameters_flat`, `set_parameters!`, simulation, and fitting.
- **Latest run (2025-12-05, Julia 1.12.2):** 32 / 32 passing.

### `test_phasetype_is.jl`

- **Scope:** Phase-type distribution infrastructure for improved
  importance sampling in MCEM.
- **Test categories:**
  1.  **Phase-type construction:** Tests `PhaseTypeDistribution` struct,
      Coxian subintensity matrix construction, Titman-Sharples
      constraints.
  2.  **Distribution functions:** Validates `phasetype_mean`,
      `phasetype_variance`, `phasetype_cdf`, `phasetype_pdf`,
      `phasetype_hazard`.
  3.  **Surrogate building:** Tests `build_phasetype_surrogate`,
      `build_expanded_tmat`, `build_phasetype_emat`.
  4.  **FFBS integration:** Validates
      forward-filtering-backward-sampling on expanded (n×phases) state
      space.
  5.  **Proposal config:** Tests `ProposalConfig`,
      `resolve_proposal_config` for `:auto`, `:markov`, `:phasetype`
      symbols.
- **Key infrastructure tested:** Phase-type approximation of
  non-exponential sojourn times for improved ESS.
- **Reference:** Titman & Sharples (2010) Biometrics 66(3):742-752.
- **Latest run (2025-06-03, Julia 1.12.1):** 114 / 114 passing.

### `test_splines.jl`

- **Scope:** Spline hazard construction and evaluation via
  BSplineKit.jl.
- **Test categories:**
  1.  **Knot placement:** Tests automatic quantile-based interior knot
      selection, boundary knot handling.
  2.  **Basis construction:** Validates `RecombinedBSplineBasis` with
      natural boundary conditions.
  3.  **Hazard evaluation:** Tests `RuntimeSplineHazard` closures for
      hazard and cumulative hazard.
  4.  **Coefficient transformations:** Validates `_spline_ests2coefs`
      (log-scale → spline coefficients) and inverse.
  5.  **Monotone constraints:** Tests I-spline cumsum transformation for
      monotone hazards.
  6.  **AD compatibility:** Verifies ForwardDiff works through spline
      evaluation.
- **Key assertions:** Spline evaluations match expected values,
  cumulative hazard equals analytic integral, extrapolation handles
  boundary correctly.
- **Latest run (2025-06-03, Julia 1.12.1):** 72 / 72 passing.

### Long Tests (MSM_TEST_LEVEL=full)

Long tests provide statistical validation through simulation studies:

| Test File | Purpose | Runtime |
|----|----|----|
| `longtest_exact_markov.jl` | Exact Markov MLE validation | ~2 min |
| `longtest_mcem.jl` | MCEM convergence for semi-Markov | ~1 min |
| `longtest_mcem_splines.jl` | MCEM with spline hazards | ~3 min |
| `longtest_mcem_tvc.jl` | MCEM with time-varying covariates | ~2 min |
| `longtest_phasetype_hazards.jl` | Phase-type hazard model inference (exact) | ~20 sec |
| `longtest_phasetype_panel.jl` | Phase-type hazard model inference (panel/mixed) | ~45 sec |
| `longtest_simulation_distribution.jl` | Simulation distributional parity | ~10 min |
| `longtest_simulation_tvc.jl` | Simulation with TVC | ~5 min |
| `longtest_robust_parametric.jl` | Parametric families (large n) | ~5 min |
| `longtest_robust_markov_phasetype.jl` | Markov/phase-type IS correctness | ~5 min |

See detailed documentation: - `test/reports/simulation_longtests.md` -
simulation distribution validation -
`test/reports/inference_longtests.md` - MCEM and MLE fitting validation

#### MCEM ESS and Pareto-k Behavior

When running MCEM long tests, certain ESS and Pareto-k patterns are
expected and do not indicate bugs:

| Subject Pattern | Pareto-k | ESS | Explanation |
|----|----|----|----|
| Early transition (first interval) | 1.0 | 1.0 | Path structure is deterministic; only transition time varies |
| No transition | 0.0 | target | All sampled paths identical; uniform weights |
| Late transition | ~1.0 | ~target | PSIS fitting may fail with few samples, but ESS is adequate |

**Pareto-k diagnostic interpretation** (from PSIS literature): -
`k = 0.0`: Uniform weights (no variability) - `k < 0.5`: Good importance
sampling quality - `0.5 < k < 0.7`: Acceptable quality - `k > 0.7`: Poor
quality - weights have heavy tails - `k = 1.0`: PSIS fitting failed
(fallback to standard IS)

**Note:** High Pareto-k values indicate the Markov surrogate may not be
an ideal proposal for the semi-Markov target, but the MCEM algorithm
still converges correctly. Subjects with ESS = 1.0 contribute less
information but do not prevent convergence.

**Test assertions:** The `longtest_mcem_tvc.jl` suite verifies: 1. MCEM
is correctly invoked (ConvergenceRecords has `ess_trace`) 2. All ESS
values are ≥ 1.0 3. Average ESS is reasonable (\> 5.0) 4. Fitted
parameters are finite and in reasonable ranges

### Setup / Fixture Files

- `test/setup_*.jl` scripts assemble toy models used across suites; each
  script is scoped to the corresponding suite (e.g.,
  `setup_3state_expwei.jl`).
- `test/fixtures/TestFixtures.jl` exports reusable builders like
  `toy_expwei_model()`; updates here should be reflected wherever
  fixtures power assertions.
- **TVC fixtures (added 2025-11-28):** `toy_tvc_exp_model`,
  `toy_tvc_wei_model`, `toy_tvc_gom_model`,
  `toy_illness_death_tvc_model`, `toy_semi_markov_tvc_model` provide
  models with time-varying covariates for testing piecewise hazard
  computation.
- **Parameter handling (2025-12-01):** All fixtures updated to use
  ParameterHandling.jl infrastructure. Use
  `get_log_scale_params(model.parameters)` to extract log-scale
  parameters for hazard evaluation; use `get_parameters_flat(model)` for
  optimizer-compatible flat vectors.

### Retired Suites

- The legacy `test/deprecated` folder (manual long tests predating the
  time transformation solver plus obsolete likelihood/pathfunction
  suites) has been removed. Recreate those scripts locally if historical
  behavior must be reproduced.

## Maintenance Checklist

- When adding a suite, describe its intent, primary assertions, and any
  fixtures it introduces.
- After major refactors, run targeted tests (e.g.,
  `Pkg.test(test_args=["test_hazards"])`) and note new expectations here
  if they materially expand coverage.
- Keep section ordering mirrored with `runtests.jl` so new contributors
  can cross-reference quickly.
