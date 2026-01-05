# Spline Comparison Benchmark Handoff Prompt

**Date:** January 5, 2026  
**Branch:** `penalized_splines`

---

## Context

You are continuing work on a **spline comparison benchmark report** for the MultistateModels.jl package. The report compares penalized spline smoothing parameter selection methods between Julia (MultistateModels.jl) and R packages (mgcv, flexsurv).

**For comprehensive implementation details, see:** `scratch/PENALIZED_SPLINES_REFERENCE.md`

---

## Current State

**Report location**: `MultistateModelsTests/reports/spline_comparison_benchmark.qmd`

### ✅ What's Complete

1. **Simple 1-transition survival example** (n=100, Weibull shape=1.5, rate=0.3)
2. **Julia results for 4 smoothing methods**: PIJCV, GCV, PERF, EFS
3. **mgcv PAM comparison** with GCV.Cp, REML, and NCV/LOOCV
4. **flexsurv spline comparison**
5. **EDF comparison table** (all methods agree ~1.5-2.5 EDF)
6. **Penalty matrix comparison** using gratia package
7. **Hazard/survival/CIF function plots**

### Key Findings Documented

- λ scaling differs ~1000x between Julia and mgcv (due to sample size, penalty normalization, likelihood formulation)
- **EDF is the proper comparison metric** (scale-invariant)
- All methods agree on model complexity
- **NCV in mgcv with `nei=NULL`** = exact LOOCV
- **PIJCV** = approximate LOOCV (Newton step approximation)

---

## Outstanding Task: Add Exact LOOCV to Julia

The user wants **exact LOOCV** added to MultistateModels.jl for the comparison. Currently:

- **PIJCV** = approximate LOOCV (Newton step approximation for LOO parameters, then evaluates actual likelihood)
- **NCV in mgcv** with `nei=NULL` = exact LOOCV

### Implementation Plan

1. **Add `:loocv` method** to `select_smoothing_parameters()` in `src/inference/smoothing_selection.jl`

2. **Algorithm**:
   - For each λ in grid: refit model n times, each time leaving out observation i
   - Compute CV score = Σᵢ ℓᵢ(β̂₋ᵢ) where β̂₋ᵢ is MLE without observation i
   - This is O(n × fitting cost) - expensive but exact

3. **Update fixture generator** (`MultistateModelsTests/fixtures/generate_simple_benchmark.jl`) to include LOOCV results

4. **Update report** to show LOOCV alongside PIJCV

### Implementation Sketch

```julia
function compute_loocv_criterion(log_lambda, beta_init, model, penalty_config, samplepaths)
    lambda = exp.(log_lambda)
    n = length(samplepaths)
    cv_score = 0.0
    
    for i in 1:n
        # Create model/data without observation i
        # Refit to get β̂₋ᵢ
        # Evaluate likelihood of observation i at β̂₋ᵢ
        cv_score += ℓᵢ(β̂₋ᵢ)
    end
    
    return -cv_score  # Return negative for minimization
end
```

**Challenge**: Efficiently refitting without observation i. Options:
1. Create subset data and new model (cleanest but slowest)
2. Use weighted likelihood with weight=0 for observation i
3. Warm-start from full MLE

---

## Key Files

| File | Purpose |
|------|---------|
| `scratch/PENALIZED_SPLINES_REFERENCE.md` | **Consolidated reference** for all penalized spline implementation details |
| `src/inference/smoothing_selection.jl` | Contains `select_smoothing_parameters()` and criterion functions |
| `MultistateModelsTests/fixtures/generate_simple_benchmark.jl` | Generates Julia results for report |
| `MultistateModelsTests/fixtures/simple_benchmark_all_methods.json` | Current Julia results |
| `MultistateModelsTests/reports/spline_comparison_benchmark.qmd` | Quarto report |

---

## Validation

After implementing LOOCV, verify:
1. LOOCV and PIJCV should give **similar λ values** (PIJCV approximates LOOCV)
2. Both should give **similar EDF**
3. Run existing spline tests: `julia --project -e 'using Pkg; Pkg.test()' -- splines`

---

## Report Rendering

```bash
cd MultistateModelsTests/reports
quarto render spline_comparison_benchmark.qmd
```

Output goes to `_site/spline_comparison_benchmark.html`

---

## R Package Notes

### mgcv NCV/LOOCV
- `method="NCV"` with `nei=NULL` (default) is exactly LOOCV
- From documentation: "If `nei==NULL` then leave-one-out cross validation is obtained"

### flexmsm
- Supports exact observation times via `living.exact` argument
- But requires 3+ state illness-death models (not simple 2-state survival)
- Uses PERF as default smoothing method

---

## References

- Wood, S.N. (2024). "On Neighbourhood Cross Validation." arXiv:2404.16490v4
- Wood, S.N. & Fasiolo, M. (2017). "A generalized Fellner-Schall method." *Statistics and Computing*
- Marra, G. & Radice, R. (2020). "Copula link-based additive models." *JASA*
- Eletti, A., Marra, G. & Radice, R. (2024). "Spline-Based Multi-State Models." arXiv:2312.05345v4
