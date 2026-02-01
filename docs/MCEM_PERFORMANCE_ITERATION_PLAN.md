# MCEM Performance Iteration Implementation Plan

**Date**: January 31, 2026  
**Status**: 🔴 **NOT IMPLEMENTED**  
**Priority**: HIGH - Blocks penalized spline MCEM tests

---

## Executive Summary

The Performance Iteration algorithm (Wood 2024) for smoothing parameter selection was planned for MCEM but **never actually implemented**. The current code:

1. `performance_iteration.jl` - Has syntax error, disabled via comment in `smoothing_selection.jl`
2. `dispatch_mcem.jl` - Dispatches to `_nested_optimization_*` functions that cause numerical instability
3. Documentation claims completion but functions were never removed

This plan provides concrete steps to implement MCEM Performance Iteration.

---

## Current State Audit

### Files and Issues

| File | Status | Issues |
|------|--------|--------|
| `performance_iteration.jl` | 🔴 Disabled | Syntax error at line 978; calls undefined `_nll_for_pi` |
| `dispatch_mcem.jl` | 🔴 Broken | Uses nested optimization causing `acosh(Inf)` errors |
| `smoothing_selection.jl` | 🟡 Workaround | Line 31 comments out performance_iteration.jl |

### Nested Optimization Functions (Still Exist - Should Be Removed)

```
dispatch_general.jl:38:  _nested_optimization_pijcv
dispatch_general.jl:374: _nested_optimization_reml
dispatch_general.jl:400: _nested_optimization_perf
dispatch_general.jl:425: _nested_optimization_criterion
dispatch_markov.jl:214:  _nested_optimization_pijcv_markov
dispatch_markov.jl:554:  _nested_optimization_criterion_markov
dispatch_mcem.jl:249:    _nested_optimization_pijcv_mcem
dispatch_mcem.jl:561:    _nested_optimization_criterion_mcem
implicit_diff.jl:1650:   _nested_optimization_pijcv_implicit
implicit_diff.jl:1919:   _nested_optimization_pijcv_markov_implicit
implicit_diff.jl:2169:   _nested_optimization_pijcv_mcem_implicit
```

---

## Architecture Overview

### Performance Iteration Algorithm (Wood 2024)

The key insight is that nested optimization (outer loop for λ, inner loop for β) is inefficient. 
Performance iteration alternates **single Newton steps** for β and λ:

```
ALGORITHM: Performance Iteration
INPUT: model, data, penalty, selector, β₀
OUTPUT: (λ_opt, β_opt, V_opt)

Initialize:
  β ← β₀
  λ ← λ_default (e.g., 1.0)
  
FOR iter = 1 to maxiter:
    # STEP 1: One Newton step for β (penalized likelihood)
    # Minimize: f(β) = -ℓ(β) + (1/2) Σⱼ λⱼ β'Sⱼβ
    g ← ∇f(β) = -∇ℓ(β) + Σⱼ λⱼ Sⱼ β
    H ← ∇²f(β) = -∇²ℓ(β) + Σⱼ λⱼ Sⱼ
    Δβ ← solve(H, -g)
    β_new ← project_to_bounds(β + Δβ, lb, ub)
    
    # STEP 2: One Newton step for λ (criterion minimization)
    # Minimize: V(λ) where V is PIJCV, EFS, or PERF criterion
    V, ∇V, H_V ← compute_criterion_with_derivatives(β_new, λ, ...)
    Δλ ← solve(H_V, -∇V)
    λ_new ← clamp(λ + Δλ, λ_min, λ_max)
    
    # Check convergence
    IF ||Δβ||/(1+||β||) < tol_β AND ||Δλ||/(1+||λ||) < tol_λ:
        BREAK
    
    β, λ ← β_new, λ_new

RETURN (λ, β, V)
```

### MCEM-Specific Considerations

For MCEM data, the likelihood is Monte Carlo approximated:

```
ℓ_MC(β) = Σᵢ Σⱼ wᵢⱼ log f(Zᵢⱼ; β)
```

Where:
- `i` indexes subjects
- `j` indexes sample paths for subject `i`  
- `wᵢⱼ` is the importance weight (from SIR)
- `Zᵢⱼ` is the j-th sample path for subject i

The gradient and Hessian have the same form but use importance-weighted sums:

```
∇ℓ_MC(β) = Σᵢ Σⱼ wᵢⱼ ∇log f(Zᵢⱼ; β)
∇²ℓ_MC(β) = Σᵢ Σⱼ wᵢⱼ ∇²log f(Zᵢⱼ; β)
```

---

## Implementation Plan

### Phase 1: Fix performance_iteration.jl (Day 1)

#### Task 1.1: Add Missing `_nll_for_pi` Dispatch Functions

The function `_nll_for_pi(β, data)` is called but never defined. Add dispatches:

**File**: `src/inference/smoothing_selection/performance_iteration.jl`

```julia
# =============================================================================
# _nll_for_pi: Negative Log-Likelihood for Performance Iteration
# =============================================================================
# These functions compute the negative log-likelihood suitable for ForwardDiff.
# They dispatch on data type to handle ExactData, MPanelData, and MCEMSelectionData.
# =============================================================================

"""
    _nll_for_pi(β, data::ExactData) -> Float64

Compute negative log-likelihood for exact data.
"""
function _nll_for_pi(β::AbstractVector, data::ExactData)
    return -loglik_exact(β, data)
end

"""
    _nll_for_pi(β, data::MPanelData) -> Float64

Compute negative log-likelihood for Markov panel data.
"""
function _nll_for_pi(β::AbstractVector, data::MPanelData)
    return _loglik_markov_mutating(β, data; neg=true, return_ll_subj=false)
end

"""
    _nll_for_pi(β, data::SMPanelData) -> Float64

Compute importance-weighted negative log-likelihood for MCEM data.

Uses the complete-data log-likelihood weighted by SIR importance weights:
    -ℓ_MC(β) = -Σᵢ Σⱼ wᵢⱼ log f(Zᵢⱼ; β)
"""
function _nll_for_pi(β::AbstractVector, data::SMPanelData)
    # SMPanelData contains:
    #   - paths: Vector of sample paths per subject
    #   - weights: Importance weights from SIR
    #   - model: The MultistateProcess for evaluation
    
    total_nll = zero(eltype(β))
    
    for i in 1:length(data.paths)
        paths_i = data.paths[i]
        weights_i = data.weights[i]
        
        for (j, (path, w)) in enumerate(zip(paths_i, weights_i))
            if w > 0
                ll_path = _loglik_path(β, path, data.model)
                total_nll -= w * ll_path
            end
        end
    end
    
    return total_nll
end
```

#### Task 1.2: Add Missing `_compute_subject_grads_hessians` Dispatch

The function is called with signature `(β, model, data)` but only defined with `(β, cache)`.

**File**: `src/inference/smoothing_selection/performance_iteration.jl`

```julia
"""
    _compute_subject_grads_hessians(β, model, data::ExactData)

Compute per-subject gradients and Hessians for exact data.
"""
function _compute_subject_grads_hessians(β::Vector{Float64}, model::MultistateProcess, data::ExactData)
    n_subjects = length(unique(data.data.id))
    n_params = length(β)
    
    grads = Vector{Vector{Float64}}(undef, n_subjects)
    hessians = Vector{Matrix{Float64}}(undef, n_subjects)
    
    for i in 1:n_subjects
        # Extract subject i data
        ll_i = β -> loglik_exact_subject_i(β, data, i)
        grads[i] = ForwardDiff.gradient(ll_i, β)
        hessians[i] = ForwardDiff.hessian(ll_i, β)
    end
    
    return grads, hessians
end

"""
    _compute_subject_grads_hessians(β, model, data::MPanelData)

Compute per-subject gradients and Hessians for Markov panel data.
"""
function _compute_subject_grads_hessians(β::Vector{Float64}, model::MultistateProcess, data::MPanelData)
    n_subjects = length(data.books)
    n_params = length(β)
    
    grads = Vector{Vector{Float64}}(undef, n_subjects)
    hessians = Vector{Matrix{Float64}}(undef, n_subjects)
    
    for i in 1:n_subjects
        ll_i = β -> loglik_markov_subject_i(β, data, i)
        grads[i] = ForwardDiff.gradient(ll_i, β)
        hessians[i] = ForwardDiff.hessian(ll_i, β)
    end
    
    return grads, hessians
end

"""
    _compute_subject_grads_hessians(β, model, data::SMPanelData)

Compute per-subject importance-weighted gradients and Hessians for MCEM data.

For MCEM, each subject's contribution is:
    ℓᵢ(β) = Σⱼ wᵢⱼ log f(Zᵢⱼ; β)
"""
function _compute_subject_grads_hessians(β::Vector{Float64}, model::MultistateProcess, data::SMPanelData)
    n_subjects = length(data.paths)
    n_params = length(β)
    
    grads = Vector{Vector{Float64}}(undef, n_subjects)
    hessians = Vector{Matrix{Float64}}(undef, n_subjects)
    
    for i in 1:n_subjects
        # Importance-weighted log-likelihood for subject i
        function ll_i(β_)
            total = zero(eltype(β_))
            paths_i = data.paths[i]
            weights_i = data.weights[i]
            for (path, w) in zip(paths_i, weights_i)
                if w > 0
                    total += w * _loglik_path(β_, path, data.model)
                end
            end
            return total
        end
        
        grads[i] = ForwardDiff.gradient(ll_i, β)
        hessians[i] = ForwardDiff.hessian(ll_i, β)
    end
    
    return grads, hessians
end
```

#### Task 1.3: Add Missing `_get_n_subjects` Dispatch

```julia
_get_n_subjects(data::ExactData) = length(unique(data.data.id))
_get_n_subjects(data::MPanelData) = length(data.books)
_get_n_subjects(data::SMPanelData) = length(data.paths)
```

#### Task 1.4: Fix Syntax Error

**FINDING**: The error "break or continue outside loop" at line 978 is a Julia parser quirk. The `break` statements at lines 1279 and 1290 ARE inside a `for` loop (starting at line 1090). However, because `_nll_for_pi` is called but undefined, Julia's parser may not properly track the loop structure.

**Root cause investigation**:
- Line 1102 calls `_nll_for_pi(b, data)` - NOT DEFINED
- The parser hits an error before completing the function, leaving loop tracking corrupted
- When error reporting, Julia points to line 978 (docstring start) as the "start" of the problematic code

**Action**: 
1. Add the missing `_nll_for_pi` dispatches (Task 1.1)
2. Re-enable include in smoothing_selection.jl
3. If syntax error persists, examine the specific parsing flow

### Phase 2: Enable and Test (Day 1-2)

#### Task 2.1: Enable performance_iteration.jl

**File**: `src/inference/smoothing_selection.jl`

Change line 31 from:
```julia
# include("smoothing_selection/performance_iteration.jl")  # DISABLED: syntax error
```
to:
```julia
include("smoothing_selection/performance_iteration.jl")
```

#### Task 2.2: Update dispatch_mcem.jl

Replace nested optimization calls with Performance Iteration:

**File**: `src/inference/smoothing_selection/dispatch_mcem.jl`

```julia
# Replace lines 83-130 with:

# PIJCVSelector, REMLSelector, PERFSelector: Use Performance Iteration
if selector isa Union{PIJCVSelector, REMLSelector, PERFSelector}
    return _performance_iteration(
        model, data, penalty, selector;
        beta_init=beta_init,
        lambda_init=lambda_init,
        alpha_info=alpha_info,
        alpha_groups=alpha_groups,
        maxiter=outer_maxiter,
        lambda_tol=lambda_tol,
        verbose=verbose
    )
end
```

#### Task 2.3: Unit Tests

**File**: `MultistateModelsTests/unit/test_performance_iteration_mcem.jl`

```julia
@testset "Performance Iteration MCEM" begin
    
    @testset "Basic MCEM PI convergence" begin
        # Create simple exponential model
        h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                     degree=1, knots=Float64[], boundaryknots=[0.0, 5.0])
        
        # Generate panel data via simulation
        # ... setup code ...
        
        model = multistatemodel(h12; data=panel_data, surrogate=:markov)
        
        # Fit with penalty=:auto (should use PI)
        result = fit(model; 
            penalty=:auto, 
            select_lambda=:efs,
            maxiter=5,
            verbose=true)
        
        # Check convergence
        @test isfinite(result.loglik.loglik)
        @test result.smoothing_parameters[1] > 0
    end
    
    @testset "_nll_for_pi dispatch" begin
        # Test that _nll_for_pi works with SMPanelData
        # ... test code ...
    end
    
    @testset "_compute_subject_grads_hessians for MCEM" begin
        # Test importance-weighted gradients/Hessians
        # ... test code ...
    end
end
```

### Phase 3: Remove Nested Optimization (Day 2)

After Performance Iteration is working, remove the legacy nested optimization functions.

#### Task 3.1: Remove from dispatch_mcem.jl

Delete functions:
- `_nested_optimization_pijcv_mcem` (lines 249-559)
- `_nested_optimization_criterion_mcem` (lines 561-750)

#### Task 3.2: Remove from implicit_diff.jl

Delete function:
- `_nested_optimization_pijcv_mcem_implicit` (lines 2169+)

#### Task 3.3: Update dispatch_markov.jl (if needed)

Ensure Markov panel data also uses Performance Iteration.

### Phase 4: Long Tests (Day 3)

Run the full `longtest_mcem_splines_penalized.jl` test suite to validate.

---

## Test Specifications

### Unit Tests (Must Pass Before Merge)

| Test | Description | Success Criteria |
|------|-------------|------------------|
| `_nll_for_pi(β, SMPanelData)` | NLL computation | Finite, matches manual calculation |
| `_compute_subject_grads_hessians(β, model, SMPanelData)` | Per-subject derivatives | Shapes correct, finite values |
| `_performance_iteration` basic | PI converges for simple model | λ > 0, finite loglik |
| `_performance_iteration` with EFS | EFS criterion works | λ > 0, converged |
| `_performance_iteration` with PIJCV | PIJCV criterion works | λ > 0, converged |

### Long Tests (12 Tests from PERFORMANCE_ITERATION_IMPLEMENTATION.md)

| Test ID | DGP | Covariates | Criterion | Tolerance |
|---------|-----|------------|-----------|-----------|
| mcem_pen_exp | Exponential | None | EFS | 50% h(t) |
| mcem_pen_pwe | Piecewise Exp | None | EFS | 50% h(t) |
| mcem_pen_gom | Gompertz | None | EFS | 50% h(t) |
| mcem_pen_cov | Exponential | TFC | EFS | 50% β |
| mcem_pen_pijcv | Weibull | None | PIJCV | 50% h(t) |
| mcem_pen_reml | Weibull | None | REML | 50% h(t) |
| mcem_pen_perf | Weibull | None | PERF | 50% h(t) |
| mcem_pen_atrisk | Weibull | None | EFS | 50% h(t) |
| mcem_pen_learned | Weibull | None | EFS | 50% h(t) |
| mcem_pen_aft_nocov | Weibull | AFT | EFS | 50% h(t) |
| mcem_pen_aft_tfc | Weibull | AFT+TFC | EFS | 50% h(t) |
| mcem_pen_aft_tvc | Weibull | AFT+TVC | EFS | 50% h(t) |

---

## Dependencies

### Required Functions (must exist)

- `loglik_exact(β, data)` - Exact data log-likelihood
- `loglik_exact_subject_i(β, data, i)` - Per-subject exact likelihood
- `_loglik_markov_mutating(β, data; neg, return_ll_subj)` - Markov likelihood
- `loglik_markov_subject_i(β, data, i)` - Per-subject Markov likelihood
- `_loglik_path(β, path, model)` - Single path likelihood for MCEM
- `SmoothingSelectionState{D}` - State type for criterion computation
- `compute_penalty_from_lambda(β, λ, config)` - Penalty value
- `project_to_bounds(x, lb, ub)` - Bound projection

### New Functions to Implement

- `_nll_for_pi(β, data::SMPanelData)` - MCEM negative log-likelihood
- `_compute_subject_grads_hessians(β, model, data::SMPanelData)` - MCEM derivatives
- `_get_n_subjects(data::SMPanelData)` - Subject count

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Performance iteration doesn't converge for MCEM | Medium | High | Add dampening, trust regions, oscillation detection |
| MC noise causes λ oscillation | High | Medium | EMA smoothing for λ warmstart (already in fit_mcem.jl) |
| Syntax error harder to fix than expected | Low | Low | Rewrite affected function from scratch |
| Breaking changes to existing tests | Medium | Low | Run full test suite before merge |

---

## Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 1: Fix PI | 4 hours | performance_iteration.jl compiles and loads |
| Phase 2: Enable | 2 hours | MCEM dispatch uses PI, unit tests pass |
| Phase 3: Cleanup | 2 hours | Nested optimization removed |
| Phase 4: Validate | 4 hours | Long tests pass |
| **Total** | **~12 hours** | Full MCEM PI implementation |

---

## Success Criteria

1. ✅ `using MultistateModels` loads without errors
2. ✅ `fit(model; penalty=:auto)` works for MCEM spline models
3. ✅ No `acosh(Inf)` or similar domain errors
4. ✅ λ is automatically selected (λ > 0)
5. ✅ All 12 long tests pass with 50% tolerance
6. ✅ No nested optimization functions remain in codebase

---

## Appendix: SMPanelData Structure

```julia
# SMPanelData (alias: MCEMSelectionData) contains:
struct SMPanelData
    paths::Vector{Vector{SamplePath}}  # paths[i][j] = j-th path for subject i
    weights::Vector{Vector{Float64}}   # weights[i][j] = importance weight
    model::MultistateProcess           # For likelihood evaluation
    # ... other fields for bookkeeping
end
```

The importance weights come from SIR resampling and satisfy:
- `sum(weights[i]) ≈ 1` for each subject i (normalized)
- `weights[i][j] ≥ 0` for all i, j
