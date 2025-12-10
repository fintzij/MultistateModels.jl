# Diagnostics Suite

This directory contains diagnostic report generation for validating MultistateModels simulation and inference. The entry point is `generate_model_diagnostics.jl`, which iterates across every analytic family (exponential, Weibull, Gompertz), both `linpred_effect` settings (PH/AFT), and both covariate configurations (baseline-only vs. a single covariate). For each of the 14 scenarios it emits two PNGs under `assets/`:

1. `function_panel_*`: overlays analytic hazard/cumulative/survival curves with `call_haz`, `call_cumulhaz`, and `survprob` outputs (with and without the time transformation).
2. `simulation_panel_*`: compares `simulate_path` draws against the closed-form duration distribution, plots the ECDF residual, and tracks the ECDF difference between the time-transformed simulator and the fallback path to confirm parity.

For a browsable gallery of the generated PNGs (rendered inline), see [`simulation_diagnostics.md`](simulation_diagnostics.md).

## Generating Reports

### Via Julia REPL

```julia
include("MultistateModelsTests/src/MultistateModelsTests.jl")
using .MultistateModelsTests
MultistateModelsTests.generate_simulation_diagnostics()
```

### Via Command Line

```bash
julia --project=MultistateModelsTests MultistateModelsTests/diagnostics/generate_model_diagnostics.jl
```

### Standalone (with main project environment)

```bash
julia --project=. -e 'include("MultistateModelsTests/diagnostics/generate_model_diagnostics.jl"); generate_all()'
```

## Output

Generated artifacts are saved to `MultistateModelsTests/diagnostics/assets/`:

- `function_panel_*.png` - Hazard/cumulative hazard/survival function comparisons
- `simulation_panel_*.png` - Simulation distribution validation
- `phasetype_simulation/*.png` - Phase-type approximation validation
