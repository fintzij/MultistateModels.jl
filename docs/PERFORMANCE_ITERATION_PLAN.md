# Performance Iteration Implementation Plan

**Date**: January 30, 2026  
**Status**: Planning  
**Related**: Wood (2024) NCV, Implicit Differentiation, Smoothing Parameter Selection

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Background: Wood (2024) Performance Iteration](#3-background-wood-2024-performance-iteration)
4. [Current Implementation Issues](#4-current-implementation-issues)
5. [Correct Algorithm Specification](#5-correct-algorithm-specification)
6. [Multi-Lambda and Alpha Considerations](#6-multi-lambda-and-alpha-considerations)
7. [Leveraging Implicit Differentiation](#7-leveraging-implicit-differentiation)
8. [Implementation Plan](#8-implementation-plan)
9. [Testing Strategy](#9-testing-strategy)
10. [Migration Path](#10-migration-path)

---

## 1. Executive Summary

The current smoothing parameter selection in MultistateModels.jl uses **nested optimization**, where the outer loop optimizes λ and the inner loop fully converges β for each λ trial. This is computationally expensive (O(outer_iters × inner_iters × n)) and conceptually different from Wood's **performance iteration** algorithm, which alternates single Newton steps for β and λ until joint convergence.

### Key Issues

1. **Nested optimization is slow**: Each λ evaluation requires ~50 Ipopt iterations for β
2. **EFS warmstart defeats its purpose**: "Fast" EFS estimate uses same slow nested optimization
3. **No true alternating iteration**: Current code doesn't implement Wood's performance iteration
4. **Implicit differentiation underutilized**: We invested in implicit diff but use nested AD

### Proposed Solution

Implement true performance iteration that:
- Alternates **single** Newton steps for β and λ
- Leverages implicit differentiation for ∂β̂/∂λ
- Handles multiple λ (baseline + covariate smooths)
- Supports joint (λ, α) optimization for weighted penalties

---

## 2. Problem Statement

### 2.1 What Performance Iteration Should Be

From Wood (2024) "On Neighbourhood Cross Validation" and Wood & Fasiolo (2017):

```
PERFORMANCE ITERATION ALGORITHM:
─────────────────────────────────
Initialize: β⁽⁰⁾, λ⁽⁰⁾
For k = 0, 1, 2, ... until convergence:
    
    Step 1: β-UPDATE (single Newton step)
    ──────────────────────────────────────
    β⁽ᵏ⁺¹⁾ = β⁽ᵏ⁾ - [H(β⁽ᵏ⁾) + λ⁽ᵏ⁾S]⁻¹ ∇ℓ(β⁽ᵏ⁾)
    
    Step 2: λ-UPDATE (single Newton step on criterion)
    ───────────────────────────────────────────────────
    For PIJCV:  λ⁽ᵏ⁺¹⁾ = argmin V(λ | β⁽ᵏ⁺¹⁾)  [one Newton step]
    For EFS:    λ⁽ᵏ⁺¹⁾ satisfies tr(A⁽ᵏ⁺¹⁾) = γ⁽ᵏ⁺¹⁾  [closed form]
    
    Check: convergence of both β and λ
```

### 2.2 What Current Code Does

```
NESTED OPTIMIZATION (CURRENT):
──────────────────────────────
OUTER LOOP: Optimize λ via L-BFGS/Ipopt
    For each λ trial:
        INNER LOOP: Fully solve for β̂(λ)
            Run Ipopt for ~50 iterations until convergence
        Compute V(λ) at β̂(λ)
        Return V to outer optimizer
    
    Outer optimizer uses V(λ) + gradient to update λ
```

### 2.3 Complexity Comparison

| Approach | β solves | Total Newton steps | Typical runtime |
|----------|----------|-------------------|-----------------|
| Performance Iteration | ~10-20 single steps | 10-20 | Fast |
| Nested Optimization | ~30-60 full solves | 30×50 = 1500 | **75x slower** |

---

## 3. Background: Wood (2024) Performance Iteration

### 3.1 The PIJCV Criterion

The Predictive Information-based Jackknife CV criterion is:

$$V(\lambda) = \sum_{i=1}^n D_i(\hat{\beta}^{-i}(\lambda))$$

where $\hat{\beta}^{-i}$ is the leave-one-out estimate, approximated via Newton:

$$\hat{\beta}^{-i} \approx \hat{\beta} + [H_\lambda - H_i]^{-1} g_i$$

### 3.2 Key Insight: Performance Iteration

Wood's key insight is that we don't need to fully solve for β at each λ. Instead:

1. **β and λ are jointly estimated** via alternating single-step updates
2. **Convergence is to the joint optimum** (β̂, λ̂)
3. **Each iteration is cheap**: One Newton step for β, one for λ

### 3.3 The EFS/REML Update

For Extended Fellner-Schall (EFS), the λ update has a closed form:

$$\lambda_j^{new} = \frac{\gamma_j}{\hat{\beta}^\top S_j \hat{\beta}}$$

where $\gamma_j$ is the effective degrees of freedom for term j.

This makes EFS extremely fast per iteration - no optimization needed for λ.

---

## 4. Current Implementation Issues

### 4.1 Location of Problems

| File | Function | Issue |
|------|----------|-------|
| `dispatch_general.jl` | `_nested_optimization_pijcv` | Full inner β solve per λ eval |
| `dispatch_general.jl` | `_nested_optimization_reml` | Same nested structure |
| `dispatch_exact.jl` | `_fit_inner_coefficients` | Returns fully converged β |
| `dispatch_markov.jl` | `_nested_optimization_pijcv_markov` | Same issues + wrong optimizer |
| `dispatch_mcem.jl` | `_nested_optimization_pijcv_mcem` | Same issues |
| `implicit_diff.jl` | `_nested_optimization_pijcv_implicit` | Uses implicit diff but still nested |

### 4.2 Specific Code Problems

#### Problem 1: Inner loop runs to convergence

```julia
# Current: _fit_inner_coefficients in dispatch_exact.jl
sol = solve(prob, IpoptOptimizer(...);
            maxiters=maxiter,      # 50 iterations!
            tol=LAMBDA_SELECTION_INNER_TOL,
            ...)
```

**Should be**: Single Newton step using penalized Hessian.

#### Problem 2: EFS warmstart uses full nested optimization

```julia
# Current: in _nested_optimization_pijcv (dispatch_general.jl:193-206)
efs_result = _nested_optimization_reml(model, data, penalty;
                                       beta_init=beta_init,
                                       inner_maxiter=inner_maxiter,  # Still 50!
                                       outer_maxiter=30,
                                       ...)
```

**Should be**: EFS has closed-form λ update, no optimization needed.

#### Problem 3: Outer loop is optimization, not iteration

```julia
# Current: in _nested_optimization_pijcv
sol = solve(prob, OptimizationOptimJL.Fminbox(OptimizationOptimJL.LBFGS());
            maxiters=outer_maxiter,
            ...)
```

**Should be**: Single Newton step on V(λ) using implicit derivatives.

### 4.3 Performance Impact

For n=1000 subjects with 10 spline parameters:

| Component | Current | With Perf. Iter. |
|-----------|---------|------------------|
| Inner β solves | 30-60 full | 15-25 single steps |
| Criterion evals | 60 | 15-25 |
| Subject Hessians | 60×1000 | 25×1000 |
| Total time (est.) | 30-60 sec | 2-5 sec |

---

## 5. Correct Algorithm Specification

### 5.1 Single Newton Step for β

```julia
function newton_step_beta(β, λ, S_matrices, model, data)
    # Compute gradient and Hessian of negative log-likelihood
    g = compute_gradient(β, model, data)  # ∇(-ℓ)
    H = compute_hessian(β, model, data)   # ∇²(-ℓ)
    
    # Build penalized Hessian
    H_λ = H + sum(λ[j] * S[j] for (j, S) in enumerate(S_matrices))
    
    # Single Newton step
    Δβ = H_λ \ g  # Solve (H + λS)Δβ = g
    β_new = β - Δβ
    
    return β_new, H_λ  # Return H_λ for reuse in λ update
end
```

### 5.2 Newton Step for λ (PIJCV)

Using implicit differentiation:

```julia
function newton_step_lambda_pijcv(λ, β, H_λ, S_matrices, subject_grads, subject_hessians, model, data)
    # Compute V(λ) and its gradient/Hessian
    V, ∂V_∂λ = compute_pijcv_with_gradient(λ, β, H_λ, ...)
    
    # For Newton step, need ∂²V/∂λ² (or use quasi-Newton/L-BFGS approx)
    # With implicit diff: ∂V/∂λ = ∂V/∂λ|_β + (∂V/∂β)(∂β̂/∂λ)
    #                            = ∂V/∂λ|_β  (at β=β̂, ∂V/∂β=0 by optimality)
    
    # Single step (could be Newton, gradient descent, or quasi-Newton)
    λ_new = λ - step_size * ∂V_∂λ  # Simplified; actual uses Hessian or L-BFGS
    
    return λ_new
end
```

### 5.3 Closed-Form λ Update (EFS)

```julia
function efs_lambda_update(β, H_λ, S_matrices, penalty_config)
    # Compute influence matrix A = H_λ⁻¹ H
    H_λ_inv = inv(H_λ)  # Or Cholesky solve
    A = H_λ_inv * H_unpenalized
    
    # For each smoothing term j
    λ_new = similar(λ)
    for j in 1:n_lambda
        # Effective degrees of freedom for term j
        γ_j = tr(A * S_j * H_λ_inv)  # Partial trace
        
        # Extract β_j (coefficients for term j)
        β_j = β[penalty_config.terms[j].indices]
        
        # EFS update (closed form!)
        λ_new[j] = γ_j / (β_j' * S_j * β_j)
    end
    
    return λ_new
end
```

### 5.4 Complete Performance Iteration

```julia
function performance_iteration_pijcv(model, data, penalty_config;
                                      β_init, λ_init,
                                      max_iter=50, tol=1e-4)
    β = copy(β_init)
    λ = copy(λ_init)
    
    for iter in 1:max_iter
        # Step 1: Single Newton step for β
        β_new, H_λ = newton_step_beta(β, λ, penalty_config.S_matrices, model, data)
        
        # Compute subject-level quantities (needed for PIJCV)
        subject_grads = compute_subject_gradients(β_new, model, data)
        subject_hessians = compute_subject_hessians(β_new, model, data)
        
        # Step 2: Update λ via PIJCV Newton step
        λ_new = newton_step_lambda_pijcv(λ, β_new, H_λ, penalty_config, 
                                          subject_grads, subject_hessians)
        
        # Check convergence
        β_change = norm(β_new - β) / (1 + norm(β))
        λ_change = norm(log.(λ_new) - log.(λ))
        
        if β_change < tol && λ_change < tol
            return (β=β_new, λ=λ_new, converged=true, iterations=iter)
        end
        
        β, λ = β_new, λ_new
    end
    
    return (β=β, λ=λ, converged=false, iterations=max_iter)
end
```

---

## 6. Multi-Lambda and Alpha Considerations

### 6.1 Multiple Smoothing Parameters

In MultistateModels.jl, we may have multiple λ values:

1. **Baseline hazard splines**: One λ per transition (or shared across transitions)
2. **Smooth covariate effects**: One λ per smooth term s(x)
3. **Total hazard penalties**: λ for summed coefficients across hazards

The penalty structure is:

$$P(\beta, \lambda) = \sum_{j=1}^{n_\lambda} \frac{\lambda_j}{2} \beta_j^\top S_j \beta_j$$

### 6.2 Performance Iteration with Multiple λ

For multiple λ, the performance iteration extends naturally:

```julia
# Step 2 becomes a vector update:
for j in 1:n_lambda
    λ_new[j] = update_lambda_j(λ, β, j, criterion, ...)
end
```

For **EFS**, each λ_j has an independent closed-form update.

For **PIJCV**, the criterion V(λ) is scalar, so we need:
- Gradient: ∂V/∂λ ∈ ℝⁿᵏ
- Either: Full Newton with ∂²V/∂λ² (expensive for large n_λ)
- Or: Quasi-Newton (L-BFGS) using gradient only
- Or: Coordinate descent on λ (update one at a time)

### 6.3 At-Risk Weighting and α Learning

The weighted penalty is:

$$P(\beta, \lambda, \alpha) = \sum_{j=1}^{n_\lambda} \frac{\lambda_j}{2} \beta_j^\top W(\alpha) S_j W(\alpha) \beta_j$$

where $W(\alpha)$ is diagonal with $W_{kk} = Y(t_k)^{-\alpha/2}$ (at-risk weighting).

#### Joint (λ, α) Optimization

When `learn_alpha=true`, we optimize over (β, λ, α) jointly:

```
Performance Iteration with α:
────────────────────────────
For k = 0, 1, 2, ...
    Step 1: β-UPDATE (single Newton step with current λ, α)
    Step 2: λ-UPDATE (single Newton step on V(λ) with current α)
    Step 3: α-UPDATE (single step on V(α) or grid search)
```

The α update is typically:
- **Grid search**: Evaluate V at α ∈ {0, 0.25, 0.5, 0.75, 1.0}
- **Gradient step**: ∂V/∂α via implicit differentiation
- **Profile**: For each α, compute optimal (β, λ), then select best α

### 6.4 Shared vs Independent Parameters

| Parameter | Sharing | Update Strategy |
|-----------|---------|-----------------|
| λ_baseline | Per-hazard or shared | EFS: independent; PIJCV: joint gradient |
| λ_covariate | Per-smooth-term | Same as baseline |
| α | Per-transition or shared | Usually shared (1-3 values) |

---

## 7. Leveraging Implicit Differentiation

### 7.1 What We've Implemented

The `implicit_diff.jl` file contains:

1. **`compute_dbeta_dlambda`**: Computes $\frac{\partial \hat{\beta}}{\partial \lambda}$ via implicit differentiation
2. **`_nested_optimization_pijcv_implicit`**: Uses implicit diff for outer λ gradient
3. **Analytical gradient computation**: $\frac{\partial V}{\partial \lambda}$ without nested AD

### 7.2 How to Use in Performance Iteration

Implicit differentiation is key for efficient λ updates:

```julia
# At optimality: ∇_β L(β̂, λ) = 0
# By implicit function theorem:
# ∂β̂/∂λⱼ = -[∇²_ββ L]⁻¹ [∇²_βλⱼ L]
#         = -H_λ⁻¹ (Sⱼ β̂)

function compute_dbeta_dlambda(β, H_λ, S_matrices)
    H_λ_inv = cholesky(Symmetric(H_λ)) \ I
    
    dbeta_dlambda = Matrix{Float64}(undef, length(β), length(S_matrices))
    for j in 1:length(S_matrices)
        dbeta_dlambda[:, j] = -H_λ_inv * (S_matrices[j] * β)
    end
    
    return dbeta_dlambda
end
```

### 7.3 PIJCV Gradient via Implicit Diff

```julia
function pijcv_gradient_implicit(λ, β, H_λ, subject_grads, subject_hessians, 
                                  S_matrices, model, data)
    n_lambda = length(λ)
    
    # Compute ∂β̂/∂λ via implicit differentiation
    dbeta_dlambda = compute_dbeta_dlambda(β, H_λ, S_matrices)
    
    # Compute ∂V/∂β (should be ~0 at optimum, but include for robustness)
    # and ∂V/∂λ|_β (direct effect)
    
    # Chain rule: dV/dλ = ∂V/∂λ|_β + (∂V/∂β)(∂β̂/∂λ)
    # At β = β̂(λ): ∂V/∂β ≈ 0, so dV/dλ ≈ ∂V/∂λ|_β
    
    # Direct computation of ∂V/∂λ|_β for PIJCV
    ∂V_∂λ = zeros(n_lambda)
    for j in 1:n_lambda
        ∂V_∂λ[j] = compute_dV_dlambda_j_direct(j, λ, β, H_λ, S_matrices,
                                                subject_grads, subject_hessians)
    end
    
    return ∂V_∂λ
end
```

### 7.4 Integration Point

The existing `_nested_optimization_pijcv_implicit` already computes analytical gradients. The key change is:

**Before (nested)**: Use gradients within an outer optimizer that calls inner solver
**After (perf. iter.)**: Use gradients for single Newton/quasi-Newton step in alternating loop

---

## 8. Implementation Plan

### Phase 1: Core Performance Iteration Infrastructure

#### Task 1.1: Create `performance_iteration.jl`

**File**: `src/inference/smoothing_selection/performance_iteration.jl`

**Contents**:
- `single_newton_step_beta`: One Newton step for β
- `efs_lambda_update`: Closed-form EFS λ update
- `pijcv_lambda_step`: Single step on V(λ) using implicit gradients
- `performance_iteration_efs`: Main EFS performance iteration
- `performance_iteration_pijcv`: Main PIJCV performance iteration

**Action Items**:
- [ ] Create new file with module structure
- [ ] Implement `single_newton_step_beta` using existing Hessian code
- [ ] Implement `efs_lambda_update` with closed-form formula
- [ ] Implement `pijcv_lambda_step` using implicit diff gradients
- [ ] Implement main `performance_iteration_efs` loop
- [ ] Implement main `performance_iteration_pijcv` loop
- [ ] Add convergence checking and diagnostics

#### Task 1.2: Refactor Criterion Computation

**Current**: Criteria functions assume nested structure
**Needed**: Criteria that work with single-step updates

**Action Items**:
- [ ] Extract `compute_pijcv_criterion_at_state` (no inner solve)
- [ ] Add `compute_pijcv_gradient_at_state` using implicit diff
- [ ] Extract `compute_efs_criterion_at_state`
- [ ] Add `compute_efs_lambda_update` (closed form)

#### Task 1.3: Update EDF Computation

EDF must be computed at each iteration for EFS:

$$\text{EDF}_j = \text{tr}(A S_j H_\lambda^{-1})$$

**Action Items**:
- [ ] Create `compute_partial_edf` for per-term EDF
- [ ] Optimize trace computation using Cholesky factors
- [ ] Cache and reuse matrix factorizations

### Phase 2: Multi-Lambda Support

#### Task 2.1: Vectorized λ Updates

**Action Items**:
- [ ] Generalize `efs_lambda_update` for vector λ
- [ ] Implement coordinate-wise PIJCV updates
- [ ] Add option for joint vs coordinate-wise λ updates
- [ ] Handle shared λ groups (e.g., same λ for all baseline hazards)

#### Task 2.2: Covariate Smooth Integration

**Action Items**:
- [ ] Ensure smooth covariate terms have proper S matrices
- [ ] Add λ indices for covariate smooths to PenaltyConfig
- [ ] Test with models having both baseline and covariate smooths

### Phase 3: Alpha Learning Integration

#### Task 3.1: Joint (λ, α) Performance Iteration

**Action Items**:
- [ ] Add α update step after λ update
- [ ] Implement `alpha_grid_search` (discrete search over α ∈ {0, 0.25, 0.5, 0.75, 1.0})
- [ ] Implement `alpha_gradient_step` using implicit diff
- [ ] Add convergence criteria for α

#### Task 3.2: Weighted Penalty Updates

The penalty matrix changes with α:

$$S_\alpha = W(\alpha)^{1/2} S W(\alpha)^{1/2}$$

**Action Items**:
- [ ] Add `update_penalty_weights!(penalty_config, α)` function
- [ ] Ensure H_λ is rebuilt when α changes
- [ ] Cache at-risk counts for efficient W(α) updates

### Phase 4: Dispatch Integration

#### Task 4.1: Update ExactData Dispatch

**File**: `src/inference/smoothing_selection/dispatch_exact.jl`

**Action Items**:
- [ ] Add `PIJCVSelector.use_perf_iter` flag (default=true)
- [ ] Route to `performance_iteration_pijcv` when flag is true
- [ ] Keep nested optimization as fallback option
- [ ] Update `_select_hyperparameters` dispatch

#### Task 4.2: Update Markov/MCEM Dispatch

**Files**: `dispatch_markov.jl`, `dispatch_mcem.jl`

**Action Items**:
- [ ] Implement `performance_iteration_pijcv_markov`
- [ ] Implement `performance_iteration_pijcv_mcem`
- [ ] Handle MCEM-specific considerations (MC noise, path sampling)

#### Task 4.3: Deprecate Nested Functions

**Action Items**:
- [ ] Mark `_nested_optimization_pijcv` as deprecated
- [ ] Mark `_nested_optimization_reml` as deprecated
- [ ] Add deprecation warnings
- [ ] Keep for backward compatibility (1 release cycle)

### Phase 5: Testing and Validation

#### Task 5.1: Unit Tests

**Action Items**:
- [ ] Test `single_newton_step_beta` reduces objective
- [ ] Test `efs_lambda_update` matches expected formula
- [ ] Test `pijcv_lambda_step` gradient is correct (finite diff check)
- [ ] Test convergence of performance iteration

#### Task 5.2: Integration Tests

**Action Items**:
- [ ] Compare perf. iter. vs nested optimization results
- [ ] Verify same λ is selected (within tolerance)
- [ ] Benchmark speed improvement
- [ ] Test with multiple λ values
- [ ] Test with α learning

#### Task 5.3: Regression Tests

**Action Items**:
- [ ] Run spline longtest suite with new algorithm
- [ ] Ensure coverage and MSE metrics unchanged
- [ ] Document any expected differences

---

## 9. Testing Strategy

### 9.1 Correctness Tests

```julia
@testitem "Performance iteration matches nested optimization" begin
    # Setup simple model
    model, data = create_test_spline_model(n=500)
    
    # Fit with nested optimization (old)
    result_nested = _nested_optimization_pijcv(model, data, penalty, selector;
                                               beta_init=β0, ...)
    
    # Fit with performance iteration (new)
    result_perf = performance_iteration_pijcv(model, data, penalty;
                                               β_init=β0, λ_init=λ0, ...)
    
    # Results should match (within tolerance)
    @test isapprox(result_perf.λ, result_nested.lambda, rtol=0.1)
    @test isapprox(result_perf.β, result_nested.warmstart_beta, rtol=0.01)
end
```

### 9.2 Speed Benchmarks

```julia
@testitem "Performance iteration is faster" begin
    model, data = create_test_spline_model(n=1000)
    
    t_nested = @elapsed _nested_optimization_pijcv(...)
    t_perf = @elapsed performance_iteration_pijcv(...)
    
    # Should be at least 5x faster
    @test t_perf < t_nested / 5
end
```

### 9.3 Multi-Lambda Tests

```julia
@testitem "Multi-lambda performance iteration" begin
    # Model with baseline + covariate smooth
    h12 = Hazard(@formula(0 ~ 1 + s(age)), "sp", 1, 2; ...)
    model = multistatemodel(h12; data=...)
    
    result = fit(model; penalty=:auto, select_lambda=:pijcv)
    
    # Should have 2 smoothing parameters
    @test length(result.smoothing_parameters) == 2
    @test all(result.smoothing_parameters .> 0)
end
```

---

## 10. Migration Path

### 10.1 Version Plan

| Version | Changes |
|---------|---------|
| v0.X.Y (current) | Add performance iteration alongside nested |
| v0.X.Y+1 | Make performance iteration default |
| v0.X.Y+2 | Deprecate nested optimization |
| v0.X+1.0 | Remove nested optimization |

### 10.2 User-Facing Changes

**New keyword argument**:
```julia
fit(model; 
    select_lambda=:pijcv,
    perf_iter=true,  # NEW: use performance iteration (default=true)
    ...)
```

**Backward compatibility**:
```julia
fit(model; select_lambda=:pijcv, perf_iter=false)  # Use old nested optimization
```

### 10.3 Documentation Updates

**Action Items**:
- [ ] Update `optimization.md` with performance iteration explanation
- [ ] Add benchmark comparisons to docs
- [ ] Update docstrings for all affected functions
- [ ] Add migration guide for users of internal APIs

---

## Appendix A: Mathematical Details

### A.1 Newton Step Derivation

At iteration k, the penalized Newton step for β is:

$$\beta^{(k+1)} = \beta^{(k)} - [H(\beta^{(k)}) + \sum_j \lambda_j S_j]^{-1} \nabla \ell(\beta^{(k)})$$

where:
- $H(\beta) = -\nabla^2 \ell(\beta)$ is the observed Fisher information
- $S_j$ are the penalty matrices
- $\nabla \ell(\beta)$ is the score function

### A.2 EFS Update Derivation

The EFS criterion is:

$$V_{EFS}(\lambda) = -\ell(\hat\beta(\lambda)) + \text{tr}(H_\lambda^{-1} H)$$

Setting $\partial V_{EFS}/\partial \lambda_j = 0$ and using the implicit derivative $\partial \hat\beta/\partial \lambda_j$:

$$\lambda_j^{opt} = \frac{\text{tr}(A S_j H_\lambda^{-1})}{\hat\beta^\top S_j \hat\beta} = \frac{\gamma_j}{\hat\beta^\top S_j \hat\beta}$$

where $A = H_\lambda^{-1} H$ is the influence matrix.

### A.3 PIJCV Gradient via Implicit Differentiation

The PIJCV criterion is:

$$V(\lambda) = \sum_{i=1}^n D_i(\hat\beta^{-i}(\lambda))$$

The gradient is:

$$\frac{\partial V}{\partial \lambda_j} = \sum_i \frac{\partial D_i}{\partial \hat\beta^{-i}} \cdot \frac{\partial \hat\beta^{-i}}{\partial \lambda_j}$$

The second term uses implicit differentiation through both the LOO Newton step and the full Newton step.

---

## Appendix B: File Change Summary

| File | Status | Changes |
|------|--------|---------|
| `performance_iteration.jl` | **NEW** | Core performance iteration algorithms |
| `dispatch_exact.jl` | Modify | Add dispatch to performance iteration |
| `dispatch_general.jl` | Modify | Deprecate nested functions |
| `dispatch_markov.jl` | Modify | Add perf. iter. for Markov data |
| `dispatch_mcem.jl` | Modify | Add perf. iter. for MCEM |
| `implicit_diff.jl` | Reuse | Leverage existing gradient computation |
| `common.jl` | Modify | Extract criterion functions |
| `fit_penalized.jl` | Modify | Update to use new selection |

---

## Appendix C: Key Functions to Implement

```julia
# performance_iteration.jl

# Core Newton steps
single_newton_step_beta(β, λ, H, g, S_matrices) -> (β_new, H_λ)
efs_lambda_update(β, H_λ, H, S_matrices) -> λ_new
pijcv_lambda_step(λ, β, state, step_size_method) -> λ_new

# Main iteration functions  
performance_iteration_efs(model, data, penalty; kwargs...) -> PerformanceIterationResult
performance_iteration_pijcv(model, data, penalty; kwargs...) -> PerformanceIterationResult
performance_iteration_pijcv_with_alpha(model, data, penalty; kwargs...) -> PerformanceIterationResult

# Result type
struct PerformanceIterationResult
    β::Vector{Float64}
    λ::Vector{Float64}
    α::Union{Nothing, Vector{Float64}}
    edf::NamedTuple
    criterion_value::Float64
    converged::Bool
    iterations::Int
    diagnostics::NamedTuple
end
```
