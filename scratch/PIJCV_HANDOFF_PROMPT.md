# Illness-Death Model Example: Implementation Prompt

**Date:** January 5, 2026  
**Branch:** `penalized_splines`

---

## Context

The file `MultistateModelsTests/reports/spline_comparison_benchmark.qmd` contains a comprehensive comparison of spline smoothing methods across Julia (MultistateModels.jl), mgcv, and flexsurv. **Part 1** implements a simple two-state survival model and is complete. Your task is to implement **Part 2: Illness-Death Model** with the same rigor and structure.

---

## ⚠️ CRITICAL USER REQUIREMENTS

1. **ALL CODE MUST BE EMBEDDED IN THE DOCUMENT** - No external scripts, no pre-generated JSON fixtures
2. **Use native Quarto code chunks**:
   - `{bash}` chunks with `julia --project -e '...'` for Julia code
   - `{r}` chunks for R code
3. **Communication via CSV** - Julia writes CSV files, R reads them
4. The document must be **fully self-contained and reproducible** by running `quarto render`

---

## Current State (Part 1 Complete)

Part 1 (Simple Survival, ~lines 35-210 in Julia chunk) demonstrates:
- Weibull data simulation with known truth
- `calibrate_splines!` for automatic knot placement (5 interior knots at event time quantiles)
- Nine Julia smoothing methods:
  - **Newton-approximated CV (fast)**: PIJCV, PIJCV5, PIJCV10, PIJCV20
  - **Exact CV (slow but gold standard)**: LOOCV, CV5, CV10, CV20
  - **Other**: EFS
- CSV export of curves and summary statistics (including unpenalized log-likelihoods)
- R chunks for mgcv (P-splines via pammtools) and flexsurv comparison
- Plots: hazard, cumulative hazard, survival/CIF with rug plots for event times
- Tables: EDF comparison, AIC/BIC, RMSE accuracy metrics

---

## Task: Implement Part 2 - Illness-Death Model

Create a parallel analysis for a three-state illness-death model:
```
State 1 (Healthy) → State 2 (Ill) → State 3 (Dead)
                  ↘ State 3 (Dead) ↗
```

### 1. Data Simulation (Julia)

- **Transitions**: h12 (healthy→ill), h13 (healthy→dead), h23 (ill→dead)
- **True hazards**: Use distinct Weibull parameterizations for each transition, e.g.:
  - h12: shape=1.3, rate=0.25 (moderately increasing)
  - h13: shape=0.8, rate=0.15 (decreasing - competing risk)
  - h23: shape=1.5, rate=0.35 (strongly increasing after illness)
- **Sample size**: n=200 (larger due to competing risks)
- **Horizon**: max_time=6.0
- **Covariates**: None initially (baseline only)

### 2. Model Specification

```julia
h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; degree=3, knots=Float64[], 
             boundaryknots=[0.0, max_time], natural_spline=true)
h13 = Hazard(@formula(0 ~ 1), "sp", 1, 3; degree=3, knots=Float64[], 
             boundaryknots=[0.0, max_time], natural_spline=true)
h23 = Hazard(@formula(0 ~ 1), "sp", 2, 3; degree=3, knots=Float64[], 
             boundaryknots=[0.0, max_time], natural_spline=true)
model = multistatemodel(h12, h13, h23; data=surv_data)
```

Use `calibrate_splines!(model; nknots=5)` to place knots based on pooled event times.

### 3. Smoothing Methods

Fit all nine Julia methods for each transition:
- **Newton-approximated CV (fast)**: PIJCV, PIJCV5, PIJCV10, PIJCV20
- **Exact CV (slow but gold standard)**: LOOCV, CV5, CV10, CV20
- **Other**: EFS

Note: `select_smoothing_parameters` handles multi-hazard models and returns lambdas for each hazard.

### 4. CSV Exports

Create separate CSV files for illness-death results:
- `_id_surv_data.csv`: Raw survival data in standard format
- `_id_knots.csv`: Interior and boundary knots for each transition
- `_id_curves.csv`: Hazard/cumhaz/survival for each transition and method
- `_id_summary.csv`: Lambda, EDF, log-likelihood per method

### 5. R Analysis

**mgcv/pammtools**: 
- Illness-death requires separate PAM models for each transition
- Use `as_ped()` with appropriate `transition` argument
- Fit P-splines with matching knots for each transition

**flexsurv**:
- Use `flexsurv::flexsurvspline()` with `trans` argument for multi-state
- Or fit separate models per transition

**mstate** (optional):
- Consider `mstate` package for comparison

### 6. Plots Required

For each transition (h12, h13, h23):
- Hazard function comparison (with rug for transition-specific event times)
- Cumulative hazard comparison

Summary plots:
- State occupation probabilities P(X(t) = s | X(0) = 1)
- Cumulative incidence functions (accounting for competing risks)

### 7. Tables Required

- EDF comparison across methods and transitions
- AIC/BIC comparison (using unpenalized log-likelihoods)
- RMSE metrics vs true hazards per transition

---

## Key Implementation Notes

1. **Use the same seed** (e.g., 12345) for reproducibility
2. **Event times for rugs**: Filter by transition type for per-hazard rugs
3. **Log-likelihood**: Use `MultistateModels.loglik_exact()` for unpenalized values
4. **mgcv log-lik correction**: The Poisson-to-survival correction formula is:
   ```r
   loglik_survival = logLik(fit) - sum(ped$ped_status * ped$offset)
   ```
5. **Knot sharing**: Pass the same interior knots to R packages via CSV
6. **Color palette**: Extend the existing palette consistently

---

## Technical Notes

### Julia Hazard API
When evaluating hazard curves from a fitted model:
```julia
haz = model.hazards[1]
haz.hazard_fn(t, beta, ())      # ✓ Correct - use hazard_fn field
haz.cumhaz_fn(0.0, t, beta, ()) # ✓ Correct - use cumhaz_fn field
# NOT haz.hazard() or haz.cumhaz() - those don't exist
```

The `RuntimeSplineHazard` struct has `hazard_fn` and `cumhaz_fn` as callable function fields.

### mgcv/pammtools API
```r
# Create piecewise-exponential data
ped <- as_ped(data = list(surv_obj ~ 1), data = surv_data, id = "id")

# Fit PAM
fit <- pamm(ped_status ~ s(tend, bs = "ps", k = 10), data = ped, method = "REML")

# Get predictions via pammtools helpers
add_hazard(newdata, fit)
add_cumu_hazard(newdata, fit)  
add_surv_prob(newdata, fit)
```

### mgcv P-spline Knot Construction
```r
# Julia uses 5 interior knots + 2 boundary = 7 middle knots
# So k - m = 7, and with m = 2, we have k = 9
k_mgcv <- length(interior_knots) + 4  # 5 + 4 = 9
m_order <- 2

# Build the full knot sequence for P-splines
middle_knots <- c(boundary_lower, interior_knots, boundary_upper)

# Need (k + m + 2) - (k - m) = 2m + 2 = 6 padding knots (3 on each side)
delta <- mean(diff(middle_knots))
padding_lower <- boundary_lower - (3:1) * delta
padding_upper <- boundary_upper + (1:3) * delta

# Full knot sequence (13 knots for k=9, m=2)
mgcv_knots <- c(padding_lower, middle_knots, padding_upper)
```

---

## Structure

Add the illness-death content after the simple survival section, maintaining the same chunk naming convention with `id-` prefix (e.g., `id-julia-all`, `id-hazard-comparison`, etc.).

---

## Validation

After implementation:
1. Render the full document: `quarto render spline_comparison_benchmark.qmd`
2. Verify all plots render without error
3. Confirm EDF values are reasonable (typically 2-5 per hazard)
4. Check that Julia and mgcv results are qualitatively similar

---

## Key Files

| File | Purpose |
|------|---------|
| `src/inference/smoothing_selection.jl` | Contains `select_smoothing_parameters()` |
| `MultistateModelsTests/reports/spline_comparison_benchmark.qmd` | **THE REPORT** - Part 1 complete, Part 2 needs implementation |

---

## References

- Wood, S.N. (2024). "On Neighbourhood Cross Validation." arXiv:2404.16490v4
- Wood, S.N. & Fasiolo, M. (2017). "A generalized Fellner-Schall method." *Statistics and Computing*
- Marra, G. & Radice, R. (2020). "Copula link-based additive models." *JASA*
- Eletti, A., Marra, G. & Radice, R. (2024). "Spline-Based Multi-State Models." arXiv:2312.05345v4
- Li, Z. & Cao, J. (2022). "General P-Splines for Non-Uniform B-Splines." arXiv:2201.06808
