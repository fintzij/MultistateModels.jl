# PIJCV Implementation Handoff Prompt

**Date:** January 4, 2026  
**Branch:** `penalized_splines`

---

## Agent Role

You are a **senior Julia package developer and PhD mathematical statistician** with deep expertise in:
- Penalized likelihood estimation and smoothing parameter selection
- Spline-based hazard models and P-splines
- Multistate models and survival analysis
- Numerical optimization (but using standard packages, not bespoke routines)
- The SciML/Julia ecosystem

---

## Session Summary (January 4, 2026)

### ✅ Completed This Session

1. **PIJCV Implementation Validated**: The `compute_pijcv_criterion` function now correctly evaluates actual likelihoods at LOO parameters (not Taylor approximations), matching Wood (2024).

2. **Long Test Created**: `MultistateModelsTests/longtests/longtest_pijcv_loocv.jl` validates PIJCV against exact LOOCV across n=30, 50, 100 subjects. Results:
   - n=30: 0.27% max error, same optimal λ
   - n=50: 6.4% max error (at extreme λ), same optimal λ  
   - n=100: 0.11% max error, same optimal λ

3. **Cholesky Downdate Bug Fixed**: The algorithm now handles numerical issues gracefully.

4. **All 75 PIJCV Unit Tests Pass**: In `MultistateModelsTests/unit/test_pijcv.jl`

5. **Fixed Pkg.test() Dependency Issue**: Added Dates, Distributions, JSON3 to main package's test dependencies in `Project.toml`.

### ⬜ Remaining Tasks (Priority Order)

| Priority | Task | Description | Estimated Effort |
|----------|------|-------------|------------------|
| 1 | Fix test_penalty_infrastructure.jl | 2 tests fail due to non-existent 5-arg SmoothingSelectionState constructor | 1 hour |
| 2 | Verify GCV formula | Validate against Wood (2017) and mgcv | 2-3 hours |
| 3 | Implement PERF method | Marra & Radice (2020) AIC-equivalent criterion | 4-6 hours |
| 4 | Implement EFS method | Wood & Fasiolo (2017) REML-based criterion | 6-8 hours |
| 5 | Benchmark against R packages | Compare to mgcv, flexsurv, flexmsm | 8-12 hours |

### ⬜ Deferred: Advanced Spline Feature Validation

The following features have infrastructure but **have not been validated** with the corrected PIJCV:

| Feature | Status | Priority | Effort |
|---------|--------|----------|--------|
| Multiple λ (multiple penalties) | Infrastructure exists | High | 4-6 hours |
| `s(x)` smooth covariates | Implemented | High | 4-6 hours |
| `te(x,y)` tensor products | Implemented | Medium | 8-10 hours |
| `ti(x,y)` baseline×covariate interactions | Implemented | Medium | 6-8 hours |
| Shared λ (competing risks) | Implemented | Low | 4-6 hours |

See **Section 12.4** of `PIJCV_FIX_REPORT.md` for detailed plans.

---

## Your Task

Continue implementation of smoothing parameter selection in `MultistateModels.jl`. The PIJCV core is now working; focus on:

1. **Immediate**: Fix the failing tests in `test_penalty_infrastructure.jl`
2. **Short-term**: Verify GCV implementation against mgcv
3. **Medium-term**: Implement PERF and EFS methods from flexmsm (Eletti et al. 2024)
4. **Long-term**: Comprehensive benchmarks against R packages

---

## Required Reading

Before proceeding, read these documents:

1. **`scratch/PIJCV_FIX_REPORT.md`** — Comprehensive implementation report including:
   - Mathematical derivations and proofs
   - Implementation progress summary
   - **Section 12: Future Action Items** — detailed plans for PERF, EFS, benchmarks
   - Acceptance criteria

2. **Eletti, Marra & Radice (2024)** — arXiv:2312.05345v4, Appendix C
   - PERF method (Eq. 13)
   - EFS method (Eq. 14)
   - flexmsm R package interface

---

## Key Constraints

### 1. Faithfulness to References
Every formula must be traceable to the source paper. Wood (2024) for PIJCV, Wood & Fasiolo (2017) for EFS, Marra & Radice (2020) for PERF.

### 2. No Bespoke Optimization Routines
Use standard Julia packages (Optimization.jl, Optim.jl, Ipopt).

### 3. Validation at Every Step
Run code, run tests, report actual results. Never claim completion without verification.

### 4. Test Failures Are Real Errors
Never assume test failures are stochastic. Always investigate and fix.

---

## Key Files

| File | Purpose |
|------|---------|
| `scratch/PIJCV_FIX_REPORT.md` | Comprehensive report with action items |
| `src/inference/smoothing_selection.jl` | PIJCV/GCV implementation |
| `MultistateModelsTests/unit/test_pijcv.jl` | PIJCV unit tests (all pass) |
| `MultistateModelsTests/unit/test_penalty_infrastructure.jl` | Has 2 failing tests to fix |
| `MultistateModelsTests/longtests/longtest_pijcv_loocv.jl` | PIJCV vs LOOCV validation |

---

## Alternative Smoothing Methods to Implement

### PERF (Performance Iteration) — Marra & Radice (2020)

Criterion approximately equivalent to AIC:
$$\lambda^{[a+1]} = \arg\min_\lambda \|M - OM\|^2 - \check{n} + 2\text{tr}(O)$$

### EFS (Extended Fellner-Schall) — Wood & Fasiolo (2017)

REML-based update:
$$\lambda_k^{[a+1]} = \lambda_k^{[a]} \times \frac{\text{tr}(S_\lambda^{-1} \partial S_\lambda/\partial\lambda_k) - \text{tr}([{-H + S_\lambda}]^{-1} \partial S_\lambda/\partial\lambda_k)}{\hat\theta^\top (\partial S_\lambda/\partial\lambda_k) \hat\theta}$$

Both methods are implemented in flexmsm R package (`sp.method = 'perf'` or `'efs'`).

---

## Benchmark Plan

After implementing all methods, benchmark against:
1. **mgcv** — GAM gold standard (GCV, REML)
2. **flexsurv** — Parametric survival (for baseline comparison)
3. **flexmsm** — Direct competitor (PERF, EFS for multistate splines)

---

## References

- Wood, S.N. (2024). "On Neighbourhood Cross Validation." arXiv:2404.16490v4
- Eletti, A., Marra, G. & Radice, R. (2024). "Spline-Based Multi-State Models for Analyzing Disease Progression." arXiv:2312.05345v4
- Wood, S.N. & Fasiolo, M. (2017). "A generalized Fellner-Schall method for smoothing parameter estimation." *Statistics and Computing* 27(3):759-773.
- Marra, G. & Radice, R. (2020). "Copula link-based additive models for right-censored event time data." *JASA* 115(530):886-895.
