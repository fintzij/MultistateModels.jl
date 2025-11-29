# Simulation Diagnostics

The figure suite now lives under the test tree so that the PNGs, statistical checks, and plotting environment travel with the long tests. The expanded grid (all analytic families × PH/AFT × covariate/no covariate) is described in detail in [Model Generation Testing Guide](model_generation_testing.md#expanded-diagnostics-matrix); this note summarizes how to regenerate the assets and what guarantees they provide.

## Where everything resides

- PNGs: `test/diagnostics/assets/*.png`
- Plotting environment: `test/diagnostics/Project.toml`
- Generator: `test/diagnostics/generate_model_diagnostics.jl`

Each scenario emits two panels:

1. **Function panel** – analytic hazard/cumulative hazard/survival vs. `call_haz`, `call_cumulhaz`, and `survprob` (with/without the time transformation).
2. **Simulation panel** – ECDF vs. analytic CDF, ECDF residual, ECDF difference `ΔF(t) = F_tt(t) − F_fb(t)`, and a histogram/PDF overlay for the simulated durations.

## Guarantees captured

- **Call-stack accuracy:** Blue/orange solver traces lie on top of the black analytic curves for every hazard family and linpred mode, proving that the PH/AFT plumbing and Tang caches behave.
- **Distributional fidelity:** ECDF residuals stay inside ~3×10⁻³ even for the heaviest Gompertz tails, matching the tolerances enforced in `test/longtest_simulation_distribution.jl`.
- **Time-transform parity:** Every simulation panel prints `max |ΔF|` to stdout (currently 0.0 for all scenarios) and plots the same quantity so drift shows up immediately.
- **Family coverage:** The grid spans `{exp, wei, gom} × {ph, aft} × {baseline, covariate}`, mirroring the long-test scenarios.

## Regenerating the figures

```bash
julia --project=test/diagnostics test/diagnostics/generate_model_diagnostics.jl
```

The script reuses the fixtures under `test/fixtures/TestFixtures.jl`, rewrites every PNG in `test/diagnostics/assets`, and logs the maximum ECDF difference for each scenario. Commit the refreshed figures whenever simulator or hazard changes land so reviewers can diff the visuals alongside the code.
