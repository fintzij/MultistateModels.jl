# MCEM Module Notes

**Date:** 2025-12-01  
**Branch:** `infrastructure_changes`

---

## Current State Summary

### Completed Work
1. **AD Bug Fix**: `loglik_markov` now uses `eltype(parameters)` for `ll` and `ll_subj` arrays
2. **Test Coverage**: 763 tests passing (0 broken)
3. **Antipattern Fixes**: Replaced `@error` with `error()`, added input validation to `Hazard()`, created `_update_spline_hazards!()` helper, renamed booleans to `is_*` form
4. **ObservationWeights Bug Fix**: Fixed `check_weight_exclusivity()` to always set `SubjectWeights` when `nothing`
5. **Spline Hazards**: Now fully implemented and tested

### Test Coverage Achieved
- ✅ Basic exact data fitting (all observation types)
- ✅ State censoring (CensoringPatterns, EmissionMatrix)  
- ✅ All hazard families (exp, wei, gom, sp)
- ✅ Time-varying covariates
- ✅ Time transforms (Tang-style caching)
- ✅ ForwardDiff gradient/Hessian verification
- ✅ Subject/observation weights
- ✅ Multi-state progressive models (4-state, 5-state)
- ✅ Semi-Markov clock reset verification
- ✅ Spline hazard `is_separable` trait
- ✅ Mixed hazard families
- ✅ MLE parameter recovery
- ✅ Exact vs Markov likelihood consistency

---

## MCEM Implementation Overview

### Current Architecture

**Core Files:**
- `src/mcem.jl` (~160 lines): Helper functions for MCEM
  - `mcem_mll`: Marginal log-likelihood (Q function)
  - `mcem_lml`: Log marginal likelihood  
  - `mcem_lml_subj`: Per-subject log marginal likelihood
  - `var_ris`: Variance of ratio of importance-sampled means
  - `mcem_ase`: Asymptotic standard error of ΔQ
  - `SquaremState`, `squarem_step_length`, `squarem_accelerate`, `squarem_should_accept`: SQUAREM acceleration

- `src/modelfitting.jl` (lines 400-1300): Main MCEM algorithm
  - `fit(::MultistateSemiMarkovModel)`: Entry point
  - Optimization via L-BFGS (unconstrained) or Ipopt (constrained)
  - Louis's identity for observed Fisher information
  - Block-diagonal Hessian optimization
  - SQUAREM acceleration support

- `src/sampling.jl`: Path sampling
  - `DrawSamplePaths!`: Main sampling function
  - `draw_samplepath`: Single path drawing
  - `sample_ecctmc`: Endpoint-conditioned CTMC sampling
  - Forward filtering / backward sampling (FFBS)
  - Phase-type proposal infrastructure

- `src/surrogates.jl`: Surrogate model construction
  - `make_surrogate_model`: Create Markov surrogate from semi-Markov model
  - `fit_surrogate`: Fit Markov surrogate via MLE
  - `fit_phasetype_surrogate`: Build phase-type surrogate (heuristic, not MLE-fitted)

- `src/crossvalidation.jl`: Variance estimation
  - `compute_robust_vcov`: IJ (sandwich) variance
  - `compute_subject_fisher_louis_batched`: Batched Fisher info via Louis's identity

### MCEM Algorithm Flow

```
1. Initialize
   - Fit Markov surrogate for path proposals (MLE)
   - Optionally build phase-type surrogate (heuristic expansion)
   - Set initial ESS target (default 50 per subject)
   - Optional Viterbi MAP warm start
   - Draw initial sample paths until ESS target met

2. MCEM Loop (until convergence or max iterations)
   a. M-step: Optimize parameters given current paths
      - Use L-BFGS (unconstrained) or Ipopt (constrained)
      - Objective: E[complete-data log-lik | observed data]
      - Optional SQUAREM acceleration every 2 iterations
   
   b. Evaluate change in Q function (marginal log-likelihood)
      - ΔQ = Q(θ_new) - Q(θ_old)
      - ASE = asymptotic standard error of ΔQ
   
   c. Check convergence criteria
      - Ascent lower bound: ALB = quantile(N(ΔQ, ASE), α)
      - Ascent upper bound: AUB = quantile(N(ΔQ, ASE), 1-α)
      - Converged if AUB < tolerance
   
   d. If ALB < 0, increase ESS target and draw more paths
   
   e. Update importance weights (PSIS smoothing)

3. Post-convergence
   - Compute variance via Louis's identity (model-based)
   - Compute IJ/sandwich variance (robust)
   - Optional: Jackknife variance
```

---

## Known Issues & TODO Items

### 1. Phase-Type Surrogate Not MLE-Fitted (CRITICAL)

**Location:** `src/surrogates.jl`, function `fit_phasetype_surrogate`

**Current State:** Phase-type surrogates are built using heuristics from the Markov surrogate rates:
```julia
# For now, use simple equal-rate phases that match total sojourn rate
phasetype_dists[s] = _build_coxian_from_rate(n_ph, total_rate)
```

**Problem:** The phase-type distribution parameters are NOT estimated via MLE. The current approach:
1. Takes Markov surrogate exit rates
2. Builds Coxian PH with `n_phases` phases using equal rates scaled to match mean sojourn time
3. Uses `p_absorb = 1/n_phases` for early absorption probability

**Desired Behavior:** Fit phase-type distributions to observed sojourn time data via MLE, e.g.:
- Moment matching to empirical sojourn times
- EM algorithm for Coxian PH fitting
- Method of moments or spectral fitting

**Impact:** Suboptimal importance sampling efficiency when sojourn time distributions deviate significantly from Coxian with equal rates.

### 2. Surrogate Control Logic in fit() Instead of multistatemodel()

**Location:** `src/modelfitting.jl`, lines 620-655

**Current State:** Surrogate creation happens in `fit()`:
```julia
if optimize_surrogate
    surrogate_fitted = fit_surrogate(model; ...)
    markov_surrogate = MarkovSurrogate(surrogate_fitted.hazards, surrogate_fitted.parameters)
else
    markov_surrogate = MarkovSurrogate(model.markovsurrogate.hazards, ...)
end

if use_phasetype
    phasetype_surrogate = fit_phasetype_surrogate(model, markov_surrogate; ...)
end
```

**Problem:** 
- Surrogate creation is a model construction concern, not a fitting concern
- `multistatemodel()` already creates a `MarkovSurrogate` (line 970) but it's not used
- User cannot inspect/configure surrogate before fitting
- Violates separation of concerns

**Proposed Refactoring:**

Add `surrogate` option to `multistatemodel()`:
```julia
function multistatemodel(hazards...; 
    data, 
    surrogate::Symbol = :none,  # :none, :markov, :phasetype
    surrogate_config = nothing,  # ProposalConfig for :phasetype
    optimize_surrogate::Bool = true,  # fit surrogate parameters
    surrogate_constraints = nothing,
    ...)
```

Then in `fit()`, use the pre-built surrogate from `model.markovsurrogate`.

### 3. Performance Opportunities

- [ ] Batched likelihood computation available (`loglik_semi_markov_batched!`) but may not be fully utilized
- [ ] Block-diagonal Hessian speedup threshold (default 2.0×) may be suboptimal
- [ ] PSIS Pareto-k diagnostic stored but not acted upon (consider re-sampling when k > 0.7)

### 4. Documentation

- [ ] Limited docstrings for internal MCEM functions
- [ ] Need tutorial/example for SQUAREM acceleration
- [ ] Phase-type proposal documentation incomplete

---

## Reference: Morsomme et al. (2025)

Key equations from the paper:

**Louis's Identity (equation S8):**
```
Iᵢ = Mᵢ Σⱼ νᵢⱼ [-Hᵢⱼ - gᵢⱼgᵢⱼᵀ] + Mᵢ² (Σⱼ νᵢⱼ gᵢⱼ)(Σₖ νᵢₖ gᵢₖ)ᵀ
```

Where:
- `Mᵢ` = subject weight
- `νᵢⱼ` = normalized importance weight for path j of subject i
- `Hᵢⱼ` = Hessian of complete-data log-likelihood for path j
- `gᵢⱼ` = gradient of complete-data log-likelihood for path j

---

## Recent Changes (2025-12-01)

1. Fixed `check_weight_exclusivity()` to set `SubjectWeights = ones(nsubj)` when `ObservationWeights` provided
2. Added spline hazard test for `is_separable` trait (was `@test_skip`, now passes)
3. All 763 tests pass, 0 broken

---

## Next Steps

1. **Refactor surrogate control to model generation** — Move from `fit()` to `multistatemodel()`
2. **Implement MLE fitting for phase-type surrogates** — Replace heuristic with proper estimation
3. **Add MCEM integration tests** — Currently only unit tests, need end-to-end validation
4. **Profile and optimize hot paths** — Identify bottlenecks in MCEM loop
```
Iᵢ = Mᵢ Σⱼ νᵢⱼ [-Hᵢⱼ - gᵢⱼgᵢⱼᵀ] + Mᵢ² (Σⱼ νᵢⱼ gᵢⱼ)(Σₖ νᵢₖ gᵢₖ)ᵀ
```

Where:
- `Mᵢ` = subject weight
- `νᵢⱼ` = normalized importance weight for path j of subject i
- `Hᵢⱼ` = Hessian of complete-data log-likelihood for path j
- `gᵢⱼ` = gradient of complete-data log-likelihood for path j

---

## Next Steps for MCEM Work

1. **Read Morsomme PDF** for algorithm details (user to provide summary or key sections)

2. **Review MCEM Tests**
   - Check `scratch/dev_setup_files/` for MCEM-specific setups
   - Identify gaps in test coverage

3. **Performance Audit**
   - Profile MCEM fit on representative model
   - Identify bottlenecks

4. **Documentation**
   - Add docstrings to helper functions
   - Document algorithm parameters

5. **Robustness**
   - Handle Pareto-k > 0.7 warnings
   - Consider adaptive ESS targeting
