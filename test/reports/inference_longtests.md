---
title: "Inference Long Tests"
format:
    html:
        theme:
            light: darkly
            dark: darkly
        prefer-dark: true
        highlight-style: atom-one-dark
---

# Inference Long Tests

_Last updated: 2025-06-03 UTC_

This document describes the long-running statistical validation tests for model fitting and inference in MultistateModels.jl. These tests verify MLE recovery, variance estimation, and MCEM convergence for semi-Markov multistate models.

## Overview

Inference validation tests live in:
- `test/longtest_exact_markov.jl` - Exact data MLE fitting validation
- `test/longtest_mcem.jl` - MCEM algorithm for semi-Markov models
- `test/longtest_mcem_splines.jl` - MCEM with spline hazards

Run with: `MSM_TEST_LEVEL=full julia --project -e 'using Pkg; Pkg.test()'`

---

## `longtest_exact_markov.jl`

**Scope:** Validates exact Markov likelihood computation and MLE fitting for panel-observed data.

### Test Categories

1. **MLE Recovery for Exponential Models**
   - Simulates data from 3-state exponential Markov model (1↔2→3)
   - True parameters: λ₁₂ = 0.223, λ₂₁ = 0.135, λ₂₃ = 0.082
   - Fits model and verifies parameter recovery within 3 SEs
   - N = 500 subjects per simulation

2. **Exact vs Markov Consistency**
   - For obstype=1 (exact observations), both exact and Markov likelihoods should match
   - Validates transition probability matrix computation via `ExpMethodGeneric`
   - Confirms log-likelihood is finite and negative

3. **Subject Weights**
   - Tests that `SubjectWeights` correctly scale likelihood contributions
   - Weighted likelihood = unweighted with duplicated subjects

4. **Observation Weights**
   - Tests `ObservationWeights` for interval-specific weighting
   - Validates mutual exclusivity with `SubjectWeights`

5. **Emission Probabilities**
   - Tests censored observations (statefrom=0) with state uncertainty
   - Validates emission matrix computation for hidden state models

6. **Variance Estimation**
   - Model-based variance (inverse Hessian)
   - IJ variance (sandwich estimator): H⁻¹KH⁻¹
   - Bootstrap comparison (optional, N_BOOTSTRAP=200)

### Key Assertions

| Check | Criterion |
|-------|-----------|
| Parameter recovery | \|θ̂ - θ₀\| < 3 × SE |
| Log-likelihood | Finite and negative |
| Gradient accuracy | \|∇ℓ_FD - ∇ℓ_AD\| < 1e-6 |
| Variance PSD | All eigenvalues > 0 |

### Latest Run

**Date:** 2025-06-03, Julia 1.12.1, ~2 min

| Test | Result |
|------|--------|
| MLE recovery (exponential) | ✅ Pass |
| Exact vs Markov consistency | ✅ Pass |
| Subject weights | ✅ Pass |
| Observation weights | ✅ Pass |
| Emission probabilities | ✅ Pass |
| Variance estimation | ✅ Pass |

---

## `longtest_mcem.jl`

**Scope:** Statistical validation of MCEM algorithm for semi-Markov multistate models.

**References:**
- Morsomme et al. (2025) Biostatistics kxaf038 - multistate semi-Markov MCEM
- Titman & Sharples (2010) Biometrics 66(3):742-752 - phase-type approximations
- Caffo et al. (2005) JRSS-B - ascent-based MCEM stopping rules

### Test Categories

1. **Parameter Recovery (Weibull Illness-Death)**
   - Simulates panel data from illness-death model (1→2→3)
   - Weibull hazards: h₁₂ with shape=1.5, scale=0.3; h₂₃ with shape=1.2, scale=0.2
   - Observation times: [0, 2, 4, 6, 8, 10, 15, 20]
   - N = 100 subjects, MAX_ITER = 30, tolerance = 0.05
   - Validates parameter estimates within 50% of true values (accounts for MC variance)

2. **Phase-Type vs Markov Proposal Selection**
   - Tests `needs_phasetype_proposal()` for hazard type detection
   - Weibull (shape ≠ 1) → needs phase-type proposal
   - Exponential → Markov proposal sufficient
   - Validates `resolve_proposal_config()` for `:auto`, `:markov`, `:phasetype`

3. **Convergence Diagnostics**
   - Verifies convergence records contain valid traces:
     - Monte Carlo log-likelihood trace
     - ESS trace
     - Parameter trace
     - Pareto-k diagnostics for importance weights
   - Tests SQUAREM acceleration integration

4. **Surrogate Fitting**
   - `fit_surrogate()` produces valid Markov surrogate
   - Surrogate log-likelihood is finite
   - TPM book construction succeeds

5. **Viterbi MAP Warm Start**
   - Tests Viterbi path initialization for MCEM
   - Validates path structure (states, times)

### Key Infrastructure Tested

- `DrawSamplePaths!` - FFBS path sampling
- `ComputeImportanceWeightsESS!` - IS weight computation
- `fit_surrogate()` - Markov surrogate MLE
- `resolve_proposal_config()` - Proposal type resolution
- Louis's identity for observed Fisher information
- PSIS weight stabilization

### Latest Run

**Date:** 2025-06-03, Julia 1.12.1, ~40 s

| Test Category | Result |
|---------------|--------|
| Parameter recovery (Weibull) | 8 / 8 ✅ |
| Phase-type vs Markov proposal | 7 / 7 ✅ |
| Convergence diagnostics | 7 / 7 ✅ |
| Surrogate fitting | 4 / 4 ✅ |
| Viterbi MAP warm start | 5 / 5 ✅ |
| **Total** | **31 / 31** |

---

## `longtest_mcem_splines.jl`

**Scope:** Validates MCEM with spline baseline hazards, demonstrating that flexible splines can recover known parametric shapes.

**References:**
- Ramsay (1988) Statistical Science - spline smoothing
- Jackson (2023) survextrap: BMC Med Res Meth 23:282

### Test Categories

1. **Spline Approximation to Exponential**
   - Fits linear spline (degree=1, no interior knots) to exponential data
   - Should recover approximately constant hazard
   - True rate = 0.3
   - Validates hazard evaluation at multiple time points

2. **Piecewise Exponential**
   - Fits linear spline with interior knots to piecewise-constant hazard
   - Tests ability to capture rate changes at knot boundaries
   - Validates cumulative hazard integration

3. **Gompertz Approximation**
   - Fits cubic spline (degree=3) with monotone constraint
   - Should recover exponentially increasing hazard shape
   - Tests I-spline coefficient transformation

4. **Covariates with Splines**
   - Fits spline hazard with PH covariate effect
   - Validates that covariate coefficient is recovered

5. **Monotone Constraints**
   - Tests `monotone=1` constraint for increasing hazards
   - Tests `monotone=-1` constraint for decreasing hazards
   - Validates coefficient cumsum transformation

### Key Infrastructure Tested

- `RuntimeSplineHazard` construction and evaluation
- `_spline_ests2coefs()` / `_spline_coefs2ests()` transformations
- BSplineKit.jl `RecombinedBSplineBasis` with natural boundaries
- Automatic knot placement via `place_interior_knots()`

### Latest Run

**Date:** 2025-06-03, Julia 1.12.1, ~3 min

| Test Category | Result |
|---------------|--------|
| Spline vs exponential | ✅ Pass |
| Piecewise exponential | ✅ Pass |
| Gompertz approximation | ✅ Pass |
| Covariates test | ✅ Pass |
| Monotone constraints | ✅ Pass |

---

## Variance Estimation Details

The package implements two robust variance estimators following Wood (2020) Biometrics:

### Infinitesimal Jackknife (IJ) Variance

$$\hat{V}_{IJ} = H^{-1} K H^{-1}$$

where:
- $H = -\nabla^2 \ell(\hat{\theta})$ is the observed Fisher information
- $K = \sum_{i=1}^n g_i g_i^\top$ where $g_i = \nabla \ell_i(\hat{\theta})$

This is the "sandwich" or "robust" variance that is consistent even under model misspecification.

### Jackknife Variance

$$\hat{V}_{JK} = \frac{n-1}{n} \sum_{i=1}^n \Delta_i \Delta_i^\top$$

where $\Delta_i = \hat{\theta}_{-i} - \hat{\theta}$ are leave-one-out perturbations.

Two methods are available via `loo_method`:
- `:direct` (default): $\Delta_i = H^{-1} g_i$, O(p²n) complexity
- `:cholesky`: Exact $H_{-i}^{-1}$ via rank-k downdates, O(np³) complexity

The `:cholesky` method is more accurate when subjects contribute multiple observations, but `:direct` is faster for large n with few observations per subject.

### Eigenvalue Thresholding

Both estimators apply eigenvalue thresholding to handle ill-conditioned Hessians:
- Eigenvalues below `sqrt(eps())` are replaced
- Warning issued when thresholding is applied
- Ensures variance matrix is positive definite

---

## Maintenance Notes

- Update test counts after adding new test cases
- Re-run long tests after major refactors (MCEM, likelihood, variance)
- Document any changes to tolerance thresholds
- Keep parameter recovery tests realistic (expect MC noise in small samples)
