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

_Last updated: 2025-11-28 16:40 UTC_

## Overview
- `Pkg.test()` loads fixtures via `test/runtests.jl`, so every `include("test_*.jl")` automatically participates in CI.
- The suites below focus on infrastructure-critical surfaces (hazard math, subject data construction, helper utilities, likelihood computation, and model builders) used by higher-level fitting code.
- Long-run simulation or integration tests live under `test/longtest_*.jl`; they provide probabilistic sanity checks but are not the focus of this documentation.
- Update this document whenever a new suite is introduced or existing coverage materially changes.
- Latest status: `julia --project test/runtests.jl` (2025-11-28 16:30 UTC, Julia 1.12.1) reported **400/400 tests passing** across the suites listed here.

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
- **Numerics:** compares log-space results to within `1e-9` relative tolerance and exercises both per-hazard and shared cache paths.
- **Latest run:** green in the 2025-11-26 `Pkg.test()` sweep.

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
- **Latest run:** green in the 2025-11-26 `Pkg.test()` run.

### `test_simulation.jl`
- **Scope:** draws single- and multi-subject trajectories through the MCEM simulator, including optimizer hooks and importance weighting.
- **Key assertions:**
  - Error injection verifies optimizer failures propagate descriptive exceptions.
  - Absorbing-start subjects terminate immediately; censoring obeys thresholds.
  - `observe_path`/`extract_paths` round-trip observed data to the latent simulator.
- **Monte Carlo checks:** forty simulated subjects per fixture confirm jump-time ECDFs track analytic references (via KS tests ≥0.9) while per-subject seeds keep determinism.
- **Latest run:** part of the passing `Pkg.test()` sweep.

### `longtest_simulation_distribution.jl`
- Draws 1,000,000 time-transformation-disabled sample paths per scenario (≈10 minutes total runtime) to enforce parity across exponential, Weibull, and Gompertz families under PH and AFT effects.
- Quantile checks operate in probability space with a 1e-3 tolerance, roughly twice the ECDF standard error at one million samples to keep seeded runs stable.
- Also enforces relative-mean tolerances and KS thresholds for both the analytic reference and time transformation vs fallback solver parity.
- Scenario workloads are evaluated in parallel across available threads with deterministic MersenneTwister seeds so CI runs stay reproducible while runtimes scale down with hardware.
- Scenario grid: {exp, wei, gom} × {PH, AFT} × {baseline, covariate} with per-case seeds to guarantee reproducibility when tuning tolerances.
- **Latest run (2025-11-26 16:00 UTC, Julia 1.12.1, 24 min total)**

  | Check | Result |
  | --- | --- |
  | Family scenarios | 60 / 60 passing (12 scenario configs × PH/AFT × covariate flag) |
  | Empirical mean | 5.56 vs truncated mean target 5.56 (rel err 2.3e-4) |
  | KS p-value | 0.516 against truncated exponential reference |
  | Parity baseline | Max diff 7.3e-9 (tolerance 1e-8), p-value 0.99 |
  | Hardware | Apple M2 Max, 12 threads enabled |
- **Notes:** scenario seeds pin each million-draw workload; parity checks reuse identical RNG streams to ensure reproducibility when new transforms land.

### `longtest_simulation_tvc.jl`
- Validates simulation correctness for time-varying covariates (TVC) with multiple covariate change points within observation intervals.
- Draws 10,000 sample paths per scenario and compares simulated event time ECDFs against piecewise analytic CDFs using Kolmogorov-Smirnov tests.
- Tests use conditional CDFs (conditioned on event occurring before horizon) since only uncensored observations are compared.
- Scenario grid: {exp, wei, gom} × {PH, AFT} with TVC configuration `t_changes = [1.5, 3.0]`, `x_values = [0.5, 1.5, 2.5]`, `horizon = 5.0`.
- **Additional tests:**
  - Semi-Markov sojourn reset: verifies clock resets to 0 after state transitions in semi-Markov models with TVC.
  - Multi-state illness-death: confirms competing risks work correctly with TVC (1→2→3 and 1→3 pathways).
  - Reproducibility: same RNG seed produces identical paths.
  - Cache/Direct parity: `CachedTransformStrategy` and `DirectTransformStrategy` yield identical results.
- **Fixtures used:** `toy_tvc_exp_model`, `toy_tvc_wei_model`, `toy_tvc_gom_model`, `toy_illness_death_tvc_model`, `toy_semi_markov_tvc_model` from `TestFixtures.jl`.
- **Latest run (2025-11-28, Julia 1.12.1, ~10 s total)**

  | Check | Result |
  | --- | --- |
  | KS tests (6 scenarios) | 6 / 6 passing |
  | Semi-Markov sojourn reset | 9,688 / 10,000 paths with 1→2→1 pattern verified |
  | Illness-death pathways | direct=4,008, via-illness=5,435, censored=557 |
  | Reproducibility | 2 / 2 passing |
  | Cache/Direct parity | 2 / 2 passing |
  | **Total assertions** | **9,702 / 9,702 passing** |

### Setup / Fixture Files
- `test/setup_*.jl` scripts assemble toy models used across suites; each script is scoped to the corresponding suite (e.g., `setup_3state_expwei.jl`).
- `test/fixtures/TestFixtures.jl` exports reusable builders like `toy_expwei_model()`; updates here should be reflected wherever fixtures power assertions.
- **TVC fixtures (added 2025-11-28):** `toy_tvc_exp_model`, `toy_tvc_wei_model`, `toy_tvc_gom_model`, `toy_illness_death_tvc_model`, `toy_semi_markov_tvc_model` provide models with time-varying covariates for testing piecewise hazard computation.

### Retired Suites
- The legacy `test/deprecated` folder (manual long tests predating the time transformation solver plus obsolete likelihood/pathfunction suites) has been removed. Recreate those scripts locally if historical behavior must be reproduced.

## Maintenance Checklist
- When adding a suite, describe its intent, primary assertions, and any fixtures it introduces.
- After major refactors, run targeted tests (e.g., `Pkg.test(test_args=["test_hazards"])`) and note new expectations here if they materially expand coverage.
- Keep section ordering mirrored with `runtests.jl` so new contributors can cross-reference quickly.
