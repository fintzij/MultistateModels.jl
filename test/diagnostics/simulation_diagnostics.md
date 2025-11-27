# Simulation Diagnostics (test-owned)

This note now lives with the plotting scripts so the figures, assets, and statistical guarantees stay versioned alongside the tests. It documents how to regenerate the diagnostic suite and provides an inline gallery of every panel produced by `generate_model_diagnostics.jl`.

- **Generator:** `test/diagnostics/generate_model_diagnostics.jl`
- **Environment:** `test/diagnostics/Project.toml`
- **Assets:** `test/diagnostics/assets/*.png`

## Reproducing the figures

```bash
julia --project=test/diagnostics test/diagnostics/generate_model_diagnostics.jl
```

The script iterates over every combination of family (`exp`, `wei`, `gom`), linpred effect (`ph`, `aft`), and covariate mode (baseline-only vs. single covariate). Each scenario emits:

1. **Function panel** – analytic hazard/cumulative hazard/survival overlaid with `call_haz`, `call_cumulhaz`, and `survprob` (with/without Tang time transforms).
2. **Simulation panel** – ECDF vs. analytic CDF, ECDF residual, Tang parity curve `ΔF(t) = F_tt(t) - F_fb(t)`, plus histogram/PDF overlay of simulated durations.

The stdout log also reports `max |ΔF|` per scenario (all at `0.0` after the latest regeneration) so regressions are obvious even without opening the PNGs.

## Figure gallery

Each table row links the two PNGs emitted for that scenario. Images render directly if you view this document in a Markdown-aware environment (VS Code, GitHub, etc.).

### Exponential family

| Scenario | Function panel | Simulation panel |
| --- | --- | --- |
| PH, baseline-only | ![Exp PH baseline function](assets/function_panel_exp_ph_baseline.png) | ![Exp PH baseline simulation](assets/simulation_panel_exp_ph_baseline.png) |
| PH, covariate | ![Exp PH covariate function](assets/function_panel_exp_ph_covariate.png) | ![Exp PH covariate simulation](assets/simulation_panel_exp_ph_covariate.png) |
| AFT, baseline-only | ![Exp AFT baseline function](assets/function_panel_exp_aft_baseline.png) | ![Exp AFT baseline simulation](assets/simulation_panel_exp_aft_baseline.png) |
| AFT, covariate | ![Exp AFT covariate function](assets/function_panel_exp_aft_covariate.png) | ![Exp AFT covariate simulation](assets/simulation_panel_exp_aft_covariate.png) |

### Weibull family

| Scenario | Function panel | Simulation panel |
| --- | --- | --- |
| PH, baseline-only | ![Wei PH baseline function](assets/function_panel_wei_ph_baseline.png) | ![Wei PH baseline simulation](assets/simulation_panel_wei_ph_baseline.png) |
| PH, covariate | ![Wei PH covariate function](assets/function_panel_wei_ph_covariate.png) | ![Wei PH covariate simulation](assets/simulation_panel_wei_ph_covariate.png) |
| AFT, baseline-only | ![Wei AFT baseline function](assets/function_panel_wei_aft_baseline.png) | ![Wei AFT baseline simulation](assets/simulation_panel_wei_aft_baseline.png) |
| AFT, covariate | ![Wei AFT covariate function](assets/function_panel_wei_aft_covariate.png) | ![Wei AFT covariate simulation](assets/simulation_panel_wei_aft_covariate.png) |

### Gompertz family

| Scenario | Function panel | Simulation panel |
| --- | --- | --- |
| PH, baseline-only | ![Gom PH baseline function](assets/function_panel_gom_ph_baseline.png) | ![Gom PH baseline simulation](assets/simulation_panel_gom_ph_baseline.png) |
| PH, covariate | ![Gom PH covariate function](assets/function_panel_gom_ph_covariate.png) | ![Gom PH covariate simulation](assets/simulation_panel_gom_ph_covariate.png) |
| AFT, baseline-only | ![Gom AFT baseline function](assets/function_panel_gom_aft_baseline.png) | ![Gom AFT baseline simulation](assets/simulation_panel_gom_aft_baseline.png) |
| AFT, covariate | ![Gom AFT covariate function](assets/function_panel_gom_aft_covariate.png) | ![Gom AFT covariate simulation](assets/simulation_panel_gom_aft_covariate.png) |

## Guarantees checked by the gallery

- **Call-stack accuracy:** Blue/orange solver traces are on top of the black analytic curves in every function panel, proving the PH/AFT plumbing (and Tang caches) agree with closed-form hazards.
- **Distributional fidelity:** ECDF residuals stay within ~3×10⁻³, matching `test/longtest_simulation_distribution.jl` tolerances.
- **Time-transform parity:** Tang-enabled simulations (green curve) match the fallback sampler, and the logged `max |ΔF|` values highlight any drift immediately.
- **Family coverage:** `{exp, wei, gom} × {ph, aft} × {baseline, covariate}` mirrors the long-test grid, so any future change that affects a subset will light up the corresponding panels.

Keep this document in sync with the assets whenever simulator or hazard changes land so reviewers can diff both code and visuals in one place.
