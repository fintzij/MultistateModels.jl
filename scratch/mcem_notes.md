# MCEM Module Notes

**Date:** 2025-11-29  
**Branch:** `infrastructure_changes`

---

## Current State Summary

### Completed Work (Prior Sessions)
1. **AD Bug Fix**: `loglik_markov` now uses `eltype(parameters)` for `ll` and `ll_subj` arrays
2. **Test Coverage**: Comprehensive exact data testing (591 tests passing, 2 broken for ObservationWeights)
3. **Import Resolution**: Fixed `ExactData` import conflicts between test files

### Test Coverage Achieved (Exact Data)
- ✅ Basic exact data fitting (all observation types)
- ✅ State censoring (CensoringPatterns, EmissionMatrix)
- ✅ All hazard families (exp, wei, gom)
- ✅ Time-varying covariates
- ✅ Time transforms (Tang-style caching)
- ✅ ForwardDiff gradient/Hessian verification
- ✅ Subject/observation weights
- ✅ Multi-state progressive models (4-state, 5-state)
- ✅ Semi-Markov clock reset verification
- ✅ Edge cases
- ✅ Mixed hazard families
- ✅ MLE parameter recovery
- ✅ Exact vs Markov likelihood consistency

---

## MCEM Implementation Overview

### Current Architecture

**Core Files:**
- `src/mcem.jl` (~81 lines): Helper functions for MCEM
  - `mcem_mll`: Marginal log-likelihood (Q function)
  - `mcem_lml`: Log marginal likelihood
  - `mcem_lml_subj`: Per-subject log marginal likelihood
  - `var_ris`: Variance of ratio of importance-sampled means
  - `mcem_ase`: Asymptotic standard error of ΔQ

- `src/modelfitting.jl` (lines 400-1000+): Main MCEM algorithm
  - `fit(::MultistateSemiMarkovModel)`: Entry point
  - Optimization via L-BFGS (unconstrained) or Ipopt (constrained)
  - Louis's identity for observed Fisher information
  - Block-diagonal Hessian optimization

- `src/sampling.jl`: Path sampling
  - `DrawSamplePaths!`: Main sampling function
  - `draw_samplepath`: Single path drawing
  - `sample_ecctmc`: Endpoint-conditioned CTMC sampling
  - Forward filtering / backward sampling (FFBS)

- `src/crossvalidation.jl`: Variance estimation
  - `compute_robust_vcov`: IJ (sandwich) variance
  - `compute_subject_fisher_louis_batched`: Batched Fisher info via Louis's identity

### MCEM Algorithm Flow

```
1. Initialize
   - Fit Markov surrogate for path proposals
   - Set initial ESS target (default 50 per subject)
   - Draw initial sample paths until ESS target met

2. MCEM Loop (until convergence or max iterations)
   a. M-step: Optimize parameters given current paths
      - Use L-BFGS (unconstrained) or Ipopt (constrained)
      - Objective: E[complete-data log-lik | observed data]
   
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

### Key Data Structures

```julia
# Path storage (per subject)
samplepaths::Vector{Vector{SamplePath}}     # sampled paths
loglik_surrog::Vector{Vector{Float64}}       # surrogate log-lik
loglik_target_cur::Vector{Vector{Float64}}   # target log-lik (current θ)
loglik_target_prop::Vector{Vector{Float64}}  # target log-lik (proposed θ)
ImportanceWeights::Vector{Vector{Float64}}   # normalized weights

# SMPanelData wrapper for optimization
SMPanelData(model, samplepaths, ImportanceWeights)
```

### Importance Weighting

```julia
# Unnormalized log importance weight
log_w[i][j] = loglik_target[i][j] - loglik_surrog[i][j]

# PSIS smoothing (ParetoSmooth.jl)
psiw = psis(log_w)
ImportanceWeights = psiw.weights  # normalized
ESS = psiw.ess                    # effective sample size
pareto_k = psiw.pareto_k          # diagnostic
```

---

## Known Issues & Improvement Areas

### 1. Performance
- [ ] Batched likelihood computation available (`loglik_semi_markov_batched!`) but may not be fully utilized
- [ ] Block-diagonal Hessian optimization exists but threshold (2.5x speedup) may be suboptimal

### 2. Variance Estimation
- [ ] ObservationWeights not supported for censored observations (forward algorithm)
- [ ] Model-based vcov via Louis's identity can be ill-conditioned

### 3. Algorithm Robustness
- [ ] ESS increase factor (default 2.0) may be too aggressive
- [ ] Max iterations default (100) may be insufficient for complex models
- [ ] PSIS Pareto-k diagnostic not acted upon (just stored)

### 4. Documentation
- [ ] Limited docstrings for internal functions
- [ ] No reference to Morsomme et al. paper in code

### 5. Testing
- [ ] No dedicated MCEM test file in runtests.jl
- [ ] Long tests exist in scratch/ but not integrated

---

## Reference: Morsomme et al. (2025)

Key equations from the paper (as referenced in code):

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
