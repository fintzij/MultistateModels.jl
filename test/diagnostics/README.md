# Diagnostics Suite (tests/diagnostics)

This directory hosts the plotting environment used to regenerate the model-generation diagnostics that now live alongside the test fixtures. The entry point is `generate_model_diagnostics.jl`, which iterates across every analytic family (exponential, Weibull, Gompertz), both `linpred_effect` settings (PH/AFT), and both covariate configurations (baseline-only vs. a single covariate). For each of the 12 scenarios it emits two PNGs under `tests/diagnostics/assets/`:

1. `function_panel_*`: overlays analytic hazard/cumulative/survival curves with `call_haz`, `call_cumulhaz`, and `survprob` outputs (with and without the time transformation).
2. `simulation_panel_*`: compares `simulate_path` draws against the closed-form duration distribution, plots the ECDF residual, and tracks the ECDF difference between the time-transformed simulator and the fallback path to confirm parity.

For a browsable gallery of the generated PNGs (rendered inline), see [`simulation_diagnostics.md`](simulation_diagnostics.md).

Run everything with:

```bash
julia --project=test/diagnostics test/diagnostics/generate_model_diagnostics.jl
```

The Project/Manifest pair is copied from the original `docs/plot_env` environment so Makie and the downstream plotting stack resolve cleanly during CI. Because this directory lives under `test/`, the generated artifacts can be versioned and linted alongside the long tests they document.
