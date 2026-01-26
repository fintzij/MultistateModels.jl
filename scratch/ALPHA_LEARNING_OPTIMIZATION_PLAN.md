# Alpha Learning Performance Optimization Plan

**Created**: January 26, 2026  
**Branch**: `penalized_splines`  
**Problem**: The `learn_alpha=true` option for adaptive penalty weighting is ~7x slower than standard PIJCV

## Current Performance Baseline

| Method | Runtime | Relative |
|--------|---------|----------|
| PIJCV (standard) | 3.7s | 1.0x |
| PIJCV (α=1 fixed) | 6.1s | 1.6x |
| PIJCV (learn α) | 27.5s | 7.4x |

## Root Cause Analysis

### 1. Nested PIJCV Re-selection (Primary Bottleneck)
**Location**: `src/inference/fit_penalized.jl` lines 247-255

The α-learning loop re-runs full PIJCV selection at each iteration:
```
for alpha_iter in 1:alpha_maxiter
    # ... update α via Brent's method ...
    
    # THIS IS THE BOTTLENECK:
    if !(selector isa NoSelection) && alpha_iter < alpha_maxiter
        smoothing_result = _select_hyperparameters(...)  # Full PIJCV again!
    end
end
```

Each PIJCV call takes ~3.7s, and with 5 α iterations, that's ~18.5s just for λ selection.

### 2. Hessian Recomputation in Brent's Method
**Location**: `src/utilities/penalty_weighting.jl` lines 773-774

The Hessian is computed inside `learn_alpha()` before the Brent objective closure:
```julia
H = _compute_hessian_block(model, data, beta, term.hazard_indices)
```

This is computed once per `learn_alpha()` call (good), but each call to `learn_alpha()` recomputes it even though β hasn't changed much.

### 3. Eigendecomposition per Brent Evaluation
**Location**: `src/utilities/penalty_weighting.jl` lines 795-799

Each objective evaluation in Brent's method computes eigenvalues:
```julia
eigvals = eigen(Symmetric(H_pen)).values
logdet_term = sum(log.(eigvals_safe))
```

With ~15 Brent evaluations per α update × 5 α iterations = ~75 eigendecompositions.

---

## Optimization Strategy

### Optimization 1: Warm-Start λ Selection (HIGH IMPACT)
**Estimated Speedup**: 3-4x  
**Risk**: Low  
**Files to Modify**: `src/inference/fit_penalized.jl`

**Current Behavior**: Each α iteration starts PIJCV from scratch with default initial λ.

**Proposed Change**: Pass the previous λ as a warm-start to the next iteration.

**Action Items**:
1. Modify `_select_hyperparameters()` to accept optional `lambda_init` parameter
2. In the α-learning loop, pass `final_penalty.terms[i].lambda` as initial value
3. Reduce PIJCV `maxiter` in α iterations since λ changes minimally between iterations

**Implementation Details**:
```
# In fit_penalized.jl α-learning loop:
# Instead of:
smoothing_result = _select_hyperparameters(model, data, final_penalty, selector; ...)

# Do:
smoothing_result = _select_hyperparameters(
    model, data, final_penalty, selector;
    lambda_init=[term.lambda for term in final_penalty.terms],  # NEW
    maxiter=20,  # Reduced from default since λ is close
    ...
)
```

**Validation**: Run timing test, expect ~2x speedup from this alone.

---

### Optimization 2: Skip Intermediate λ Re-selection (MEDIUM IMPACT)
**Estimated Speedup**: 2x additional  
**Risk**: Medium (may affect convergence quality)  
**Files to Modify**: `src/inference/fit_penalized.jl`

**Current Behavior**: Re-select λ after every α update.

**Proposed Change**: Only re-select λ every 2-3 α iterations, or only on final iteration.

**Action Items**:
1. Add `lambda_reselect_interval::Int=2` parameter to α-learning options
2. Modify loop to only call `_select_hyperparameters` when `alpha_iter % lambda_reselect_interval == 0`
3. Always re-select on final iteration before convergence check

**Implementation Details**:
```
# In α-learning loop:
should_reselect = (alpha_iter % lambda_reselect_interval == 0) || 
                  (alpha_iter == alpha_maxiter) ||
                  (max_alpha_change < 2 * alpha_tol)  # Near convergence
                  
if !(selector isa NoSelection) && should_reselect && alpha_iter < alpha_maxiter
    smoothing_result = _select_hyperparameters(...)
end
```

**Validation**: Compare final α and λ values with/without this optimization.

---

### Optimization 3: Cache Hessian Across α Iterations (MEDIUM IMPACT)
**Estimated Speedup**: 1.3x additional  
**Risk**: Low  
**Files to Modify**: `src/utilities/penalty_weighting.jl`, `src/inference/fit_penalized.jl`

**Current Behavior**: `learn_alpha()` computes Hessian each call.

**Proposed Change**: Compute Hessian once per α iteration (or reuse if β changed < threshold).

**Action Items**:
1. Create `CachedHessianState` struct to hold H and the β at which it was computed
2. Add optional `hessian_cache` parameter to `learn_alpha()`
3. Recompute Hessian only if `norm(β - β_cached) / norm(β_cached) > 0.1`
4. Manage cache lifecycle in α-learning loop

**Implementation Details**:
```julia
mutable struct HessianCache
    H::Matrix{Float64}
    beta_at_computation::Vector{Float64}
    term_idx::Int
end

function learn_alpha(...; hessian_cache::Union{Nothing, HessianCache}=nothing)
    # Reuse or recompute
    if hessian_cache !== nothing && 
       hessian_cache.term_idx == term_idx &&
       norm(beta - hessian_cache.beta_at_computation) / norm(beta) < 0.1
        H = hessian_cache.H
    else
        H = _compute_hessian_block(...)
        # Update cache if provided
    end
end
```

---

### Optimization 4: Use Cholesky Instead of Eigendecomposition (LOW IMPACT)
**Estimated Speedup**: 1.2x additional  
**Risk**: Low  
**Files to Modify**: `src/utilities/penalty_weighting.jl`

**Current Behavior**: Compute eigenvalues for log-determinant.

**Proposed Change**: Use Cholesky factorization (2x faster, O(n³/3) vs O(n³)).

**Action Items**:
1. Replace eigendecomposition with Cholesky in the Brent objective
2. Handle non-positive-definite case gracefully (fall back to eigenvalues)

**Implementation Details**:
```julia
function objective(α)
    # ... build H_pen ...
    
    # Try Cholesky first (faster)
    try
        C = cholesky(Symmetric(H_pen))
        logdet_term = 2 * sum(log.(diag(C.L)))  # log|A| = 2*sum(log(diag(L)))
    catch e
        if e isa PosDefException
            # Fall back to eigenvalues for non-PD case
            eigvals = eigen(Symmetric(H_pen)).values
            eigvals_safe = max.(eigvals, 1e-10)
            logdet_term = sum(log.(eigvals_safe))
        else
            rethrow(e)
        end
    end
    # ... rest ...
end
```

---

### Optimization 5: Coarser Convergence for α (LOW IMPACT)
**Estimated Speedup**: 1.2x  
**Risk**: Very Low  
**Files to Modify**: `src/inference/fit_penalized.jl`

**Current Behavior**: `alpha_tol=1e-2` and up to 5 iterations.

**Proposed Change**: Use coarser tolerance or adaptive stopping.

**Action Items**:
1. Increase default `alpha_tol` to 0.05 (α rarely needs precision beyond first decimal)
2. Reduce `alpha_maxiter` to 3
3. Add early stopping if α is moving toward boundary (0 or 2)

---

## Implementation Order (Recommended)

1. **Optimization 1** (warm-start λ) - Highest impact, lowest risk
2. **Optimization 5** (coarser tolerance) - Quick win, very low risk  
3. **Optimization 4** (Cholesky) - Straightforward, low risk
4. **Optimization 2** (skip λ re-selection) - Test carefully for accuracy
5. **Optimization 3** (cache Hessian) - Most complex, save for last

## Expected Final Performance

| Optimization | Incremental Speedup | Cumulative |
|--------------|---------------------|------------|
| Baseline (learn α) | 1.0x | 27.5s |
| + Warm-start λ | 3-4x | 7-9s |
| + Coarser tolerance | 1.2x | 6-7.5s |
| + Cholesky | 1.2x | 5-6s |
| + Skip λ re-selection | 1.5x | 3.5-4s |
| + Cache Hessian | 1.3x | 2.7-3s |

**Target**: Get `learn_alpha` runtime to within 2x of standard PIJCV (~7.5s → ~5s)

---

## Testing Requirements

### Unit Tests
- Verify `learn_alpha()` returns same value with/without Hessian caching
- Verify Cholesky gives same log-determinant as eigendecomposition (within 1e-10)
- Verify warm-started PIJCV converges to same λ as cold-start

### Integration Tests  
- Run `test_penalty_weighting.jl` after each optimization
- Compare fitted α, λ, and β values before/after (should be identical within tolerance)

### Timing Tests
Add benchmark to `MultistateModelsTests/benchmarks/`:
```julia
# benchmark_alpha_learning.jl
# Compare timing of standard vs learn_alpha with different optimizations
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/inference/fit_penalized.jl` | Warm-start λ, skip λ re-selection, cache management |
| `src/utilities/penalty_weighting.jl` | Cholesky log-det, Hessian caching, tolerance adjustments |
| `src/inference/hyperparameter_selection.jl` | Add `lambda_init` parameter to selectors |
| `MultistateModelsTests/unit/test_penalty_weighting.jl` | Add optimization validation tests |

---

## Success Criteria

1. `learn_alpha=true` runtime ≤ 2x standard PIJCV runtime
2. No change in fitted α, λ, β values (within numerical tolerance)
3. All existing tests pass
4. Spline benchmark document shows comparable accuracy for learn-α method
