# PIJCV Efficiency Optimizations Handoff

**Date**: January 25, 2026  
**Branch**: `penalized_splines`  
**Focus**: Cholesky Downdate + BFGS Outer Optimization

---

## Executive Summary

Two efficiency optimizations for PIJCV lambda selection, based on Wood (2024) "On Neighbourhood Cross Validation" (arXiv:2404.16490v4):

1. **True Cholesky Downdate** (O(p²) vs O(p³) per subject): Currently we use eigendecomposition + rank-1 updates. mgcv uses direct Cholesky downdate algorithms.

2. **BFGS Outer Optimization** (Wood's recommendation): Replace IPNewton with BFGS for the outer λ optimization, with gradient clamping for indefinite directions.

---

## Current State

### Current PIJCV Architecture

```
_nested_optimization_pijcv()
├── EFS warmstart (get initial λ guess)
├── Outer λ optimization (IPNewton)
│   └── For each trial λ:
│       ├── Inner β optimization (Ipopt)
│       ├── Compute per-subject grads & Hessians
│       └── Evaluate NCV criterion V(λ)
│           └── For each subject i:
│               ├── Solve LOO Newton step: Δ⁻ⁱ = H_{λ,-i}⁻¹ gᵢ
│               └── Evaluate ℓᵢ(β̂ + Δ⁻ⁱ)
└── Return optimal λ
```

### Current LOO Solve Implementation

Location: `src/inference/smoothing_selection.jl` lines 3041-3090

```julia
function _solve_loo_newton_step(chol_H::Cholesky, H_i::Matrix{Float64}, g_i::AbstractVector)
    # 1. Copy Cholesky factor L
    L = Matrix(chol_H.L)
    
    # 2. Eigendecompose subject Hessian: H_i = V D V'
    eigen_H = eigen(Symmetric(H_i))
    
    # 3. For each positive eigenvalue, perform rank-1 downdate
    for (idx, d) in enumerate(eigen_H.values)
        if d > tol
            v = eigen_H.vectors[:, idx]
            success = _cholesky_downdate!(L, sqrt(d) * v)
            if !success
                return nothing  # Indefinite
            end
        end
    end
    
    # 4. Solve with downdated factor
    return L \ g_i
end
```

**Problem**: Eigendecomposition is O(p³), defeating the purpose of the O(p²) Cholesky downdate.

---

## Part 1: True Cholesky Downdate

### Mathematical Foundation

**Goal**: Solve (H_λ - H_i)⁻¹ g_i efficiently for each subject i.

**Key insight from Wood (2024)**: For many models, H_i has low rank k << p (typically k = 1 or 2 for multistate models where each subject contributes to few transitions).

**Woodbury Identity** (when Cholesky downdate fails):

$$H_{λ,-i}^{-1} = (R_0^T R_0 - UU^T)^{-1} = R_0^{-1} \left[ I_p - R_0^{-T} U (U^T R_0^{-1} R_0^{-T} U - I_k)^{-1} U^T R_0^{-1} \right] R_0^{-T}$$

where $R_0^T R_0 = H_λ$ is the Cholesky of the full Hessian.

### Algorithm: Direct Rank-1 Downdate

For multistate models, H_i typically has rank 1-2 (one per transition type in subject's path). 

**Rank-1 Downdate** (Golub & van Loan §6.5.4):

Given L such that LL^T = A, update to L̃ such that L̃L̃^T = A - vv^T:

```
Algorithm: rank1_downdate(L, v)
Input: L (lower triangular), v (vector)
Output: L̃ or FAIL if indefinite

for j = 1 to n:
    r² = L[j,j]² - v[j]²
    if r² < ε:
        return FAIL  # Matrix became indefinite
    r = √r²
    c = r / L[j,j]
    s = -v[j] / L[j,j]
    L[j,j] = r
    for i = j+1 to n:
        temp = c * L[i,j] + s * v[i]
        v[i] = s * L[i,j] + c * v[i]
        L[i,j] = temp
return L̃
```

**Complexity**: O(p²) per rank-1 update, total O(kp²) where k = rank(H_i).

### Implementation Plan: Option A (Preferred)

**Leverage sparsity structure of H_i for multistate models**:

In multistate models with exact data, each subject contributes to specific transition hazards. The subject Hessian H_i has structure:

```julia
# For a 3-state illness-death model:
# Subject with path 1→2 only contributes to h₁₂ parameters
# H_i has non-zero blocks only in h₁₂ × h₁₂ submatrix
```

**Algorithm**:
```julia
function _solve_loo_newton_step_fast(chol_H::Cholesky{Float64}, H_i::Matrix{Float64}, g_i::AbstractVector)
    # 1. Identify non-zero structure of H_i
    # For multistate models, this is block-diagonal with small blocks
    nonzero_mask = abs.(H_i) .> tol
    active_indices = findall(any(nonzero_mask, dims=2)[:, 1])
    
    if length(active_indices) <= DIRECT_SOLVE_THRESHOLD  # e.g., 10
        # Direct solve for small active set
        # H_{λ,-i}⁻¹ g via Woodbury on small subset
        return _woodbury_solve_sparse(chol_H, H_i, g_i, active_indices)
    end
    
    # 2. Low-rank factorization of H_i
    # Use truncated eigendecomposition for rank-deficient H_i
    k = rank(H_i, rtol=1e-10)
    if k <= MAX_RANK_DOWNDATE  # e.g., 5
        # Use rank-k decomposition: H_i ≈ V_k D_k V_k^T
        F = eigen(Symmetric(H_i))
        large_eig_idx = findall(F.values .> tol)
        k_actual = length(large_eig_idx)
        
        # Perform k rank-1 downdates
        L = copy(chol_H.L)
        for idx in large_eig_idx
            d = F.values[idx]
            v = F.vectors[:, idx]
            if !_cholesky_downdate!(L, sqrt(d) * v)
                # Fallback to Woodbury
                return _woodbury_solve(chol_H, H_i, g_i)
            end
        end
        return Cholesky(L, 'L', 0) \ g_i
    end
    
    # 3. Fallback to Woodbury identity (never fails, O(kp²))
    return _woodbury_solve(chol_H, H_i, g_i)
end
```

### Implementation Plan: Option B (Simpler)

**Skip eigendecomposition entirely, use Woodbury directly**:

```julia
function _solve_loo_newton_step_woodbury(chol_H::Cholesky{Float64}, H_i::Matrix{Float64}, g_i::AbstractVector)
    # Woodbury: (A - UVᵀ)⁻¹ = A⁻¹ + A⁻¹U(I - VᵀA⁻¹U)⁻¹VᵀA⁻¹
    # Here A = H_λ (factored as R^TR), and UVᵀ = H_i
    
    # For symmetric rank-k update: H_i = UUᵀ where U is p×k
    # Use eigendecomposition once at setup, cache the factors
    
    # Solve with full matrix:
    R = chol_H.U  # Upper Cholesky factor
    
    # Step 1: Solve R^T R x = g (baseline)
    x_base = chol_H \ g_i
    
    # Step 2: Compute correction via Woodbury
    # Need: (I - UᵀR⁻¹R⁻ᵀU)⁻¹
    # For H_i = UUᵀ with small k:
    
    # Use SVD of H_i to get low-rank factors
    F = svd(H_i)
    k = count(F.S .> tol * maximum(F.S))
    if k == 0
        return x_base
    end
    
    U_k = F.U[:, 1:k] .* sqrt.(F.S[1:k])'  # p × k
    
    # Woodbury correction
    RinvU = R' \ U_k                        # p × k, O(kp²)
    M = I(k) - RinvU' * RinvU               # k × k, O(k²p)
    
    if det(M) < tol
        # M is singular, use direct solve as fallback
        return Symmetric(H_λ - H_i) \ g_i
    end
    
    correction = RinvU * (M \ (RinvU' * x_base))  # O(kp)
    return x_base + correction
end
```

**Complexity**: O(kp²) where k = effective rank of H_i.

---

## Part 2: BFGS Outer Optimization

### Why BFGS for Outer Loop?

From Wood (2024) Section 3:

> "Full outer Newton optimization would require the exact Hessian of V, which is both implementationally tedious and computationally expensive to obtain. Making quasi-Newton optimization, based only on first derivatives, a more appealing choice."

> "A stabilized nested optimization approach is: BFGS (outer) – Newton (inner)."

### Key Challenge: Indefiniteness at Large λ

The NCV criterion V(λ) becomes **indefinite** (flat) when λ is very large because:
- Large λ → heavily penalized → β̂ in null space of penalty → β̂ doesn't depend on λ

**Wood's solution**: Detect indefiniteness via:

$$\frac{d\hat\beta^T}{d\rho_j} H_\lambda \frac{d\hat\beta}{d\rho_j} \approx 0$$

where $\rho = \log \lambda$. When this condition is met AND $\partial V / \partial \rho_j \approx 0$, set the j-th component of the BFGS step to zero.

### Implementation: BFGS with Gradient Clamping

```julia
"""
    _nested_optimization_pijcv_bfgs(model, data, penalty, selector; kwargs...)

PIJCV optimization using BFGS outer loop (Wood 2024 recommended approach).

Advantages over IPNewton:
1. Only requires gradient of V(λ), not Hessian
2. Natural handling of indefiniteness via gradient clamping
3. Quasi-Newton Hessian approximation stays positive definite
"""
function _nested_optimization_pijcv_bfgs(
    model::MultistateProcess,
    data::ExactData,
    penalty::AbstractPenalty,
    selector::PIJCVSelector;
    beta_init::Vector{Float64},
    inner_maxiter::Int = 50,
    outer_maxiter::Int = 100,
    lambda_tol::Float64 = 1e-3,
    verbose::Bool = false
)
    # ... setup code same as current _nested_optimization_pijcv ...
    
    # Track dβ̂/dρ for indefiniteness detection
    dbeta_drho_ref = Ref{Union{Nothing, Matrix{Float64}}}(nothing)
    
    function ncv_criterion_with_gradient(log_lambda_vec, _)
        # ... same inner optimization as current ...
        
        # Additionally compute dβ̂/dρⱼ = -λⱼ H_λ⁻¹ Sⱼ β̂ (Wood 2024, implicit diff)
        H_lambda = # ... build penalized Hessian ...
        dbeta_drho = zeros(n_params, n_lambda)
        for j in 1:n_lambda
            S_j = # ... get j-th penalty matrix ...
            dbeta_drho[:, j] = -lambda[j] * (H_lambda \ (S_j * beta))
        end
        dbeta_drho_ref[] = dbeta_drho
        
        # Evaluate criterion
        V = compute_pijcv_criterion(log_lambda_vec, state)
        return V
    end
    
    function gradient_with_clamping!(G, log_lambda_vec, _)
        # Get gradient from ForwardDiff
        G_raw = ForwardDiff.gradient(x -> ncv_criterion_with_gradient(x, nothing), log_lambda_vec)
        
        # Apply gradient clamping for indefinite directions (Wood 2024, Section 3)
        H_lambda = # ... current penalized Hessian ...
        dbeta_drho = dbeta_drho_ref[]
        
        INDEFINITE_TOL = 1e-6
        for j in 1:n_lambda
            # Test: dβ̂ᵀ/dρⱼ H_λ dβ̂/dρⱼ ≈ 0
            dbeta_j = @view dbeta_drho[:, j]
            curvature = dot(dbeta_j, H_lambda * dbeta_j)
            
            if curvature < INDEFINITE_TOL && abs(G_raw[j]) < INDEFINITE_TOL
                G[j] = 0.0  # Clamp gradient in indefinite direction
            else
                G[j] = G_raw[j]
            end
        end
    end
    
    # Use Optimization.jl with BFGS
    optf = OptimizationFunction(ncv_criterion_with_gradient, grad=gradient_with_clamping!)
    prob = OptimizationProblem(optf, current_log_lambda, nothing; lb=log_lb, ub=log_ub)
    
    # Solve with BFGS (bounded L-BFGS for constraints)
    sol = solve(prob, Optim.LBFGS();
                maxiters=outer_maxiter,
                abstol=lambda_tol,
                reltol=lambda_tol)
    
    # ... same result processing as current ...
end
```

### Gradient of NCV Criterion

From Wood (2024) Section 2.2, the gradient of V w.r.t. ρⱼ = log λⱼ:

$$\frac{\partial V}{\partial \rho_j} = \sum_i \frac{\partial \mathcal{D}}{\partial \eta_i} \bigg|_{\eta_i^{-\alpha(i)}} \frac{\partial \eta_i^{-\alpha(i)}}{\partial \rho_j}$$

where $\eta$ is the linear predictor and $\mathcal{D}$ is the loss.

For our implementation, ForwardDiff handles this automatically. The key optimization is:

```julia
# Implicit differentiation gives dβ̂/dρⱼ without refitting:
# dβ̂/dρⱼ = -λⱼ H_λ⁻¹ Sⱼ β̂
```

This is O(p²) per smoothing parameter, using the already-factored H_λ.

---

## Implementation Checklist

### Phase 1: Cholesky Downdate Optimization

- [ ] **1.1** Profile current `_solve_loo_newton_step` to confirm eigendecomposition is the bottleneck
- [ ] **1.2** Implement `_woodbury_solve` function for indefinite cases
- [ ] **1.3** Implement sparse structure detection for multistate H_i
- [ ] **1.4** Add unit test comparing old vs new LOO solve accuracy
- [ ] **1.5** Benchmark: measure speedup on n=1000, p=20 model

### Phase 2: BFGS Outer Optimization

- [ ] **2.1** Create `_nested_optimization_pijcv_bfgs` as new function (don't modify existing)
- [ ] **2.2** Implement `gradient_with_clamping!` for indefiniteness handling
- [ ] **2.3** Compute `dβ̂/dρⱼ` via implicit differentiation
- [ ] **2.4** Add `outer_optimizer=:bfgs` option to `fit()` kwargs
- [ ] **2.5** Add unit tests comparing BFGS vs IPNewton results
- [ ] **2.6** Benchmark: measure speedup on various problem sizes

### Phase 3: Integration & Testing

- [ ] **3.1** Update docstrings with new options
- [ ] **3.2** Run full PIJCV test suite
- [ ] **3.3** Run longtest_splines to verify accuracy preserved
- [ ] **3.4** Performance comparison report

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/inference/smoothing_selection.jl` | New Cholesky downdate; new BFGS outer loop |
| `src/inference/fit_penalized.jl` | Add `outer_optimizer` kwarg routing |
| `MultistateModelsTests/unit/test_pijcv.jl` | Tests for new options |

---

## Key References

1. **Wood, S.N. (2024)**. "On Neighbourhood Cross Validation." arXiv:2404.16490v4
   - Section 2.1: Hessian factor downdate algorithms
   - Section 2.2: Derivatives of NCV criterion
   - Section 3: NCV optimization (BFGS outer, Newton inner)
   - Appendix C: Low-level computational considerations

2. **Golub & van Loan (2013)**. Matrix Computations, 4th ed.
   - Section 6.5.4: Cholesky updating and downdating

3. **Seeger, M. (2004)**. "Low Rank Updates for the Cholesky Decomposition."
   - Efficient rank-1 update algorithms

---

## Expected Performance Gains

| Component | Current | After Optimization | Speedup |
|-----------|---------|-------------------|---------|
| LOO solve per subject | O(p³) eigendecomp | O(kp²) downdate | ~p/k ≈ 10-20x |
| Outer λ iteration | IPNewton (Hessian needed) | BFGS (gradient only) | ~2-3x |
| **Combined** | Baseline | Both optimizations | **3-5x expected** |

---

## Constants and Tolerances

```julia
# Cholesky downdate
const CHOLESKY_DOWNDATE_TOL = 1e-12  # Indefiniteness detection
const MAX_RANK_DOWNDATE = 10         # Max rank for eigendecomp approach
const DIRECT_SOLVE_THRESHOLD = 10    # Use direct solve if active indices < this

# BFGS outer optimization
const INDEFINITE_GRADIENT_TOL = 1e-6 # Clamp gradient if curvature < this
const BFGS_LINE_SEARCH_TOL = 1e-4    # Wolfe conditions
```

---

## Testing Strategy

### Unit Tests

```julia
@testset "Cholesky Downdate" begin
    # Test exact recovery for well-conditioned H_i
    # Test graceful fallback for indefinite cases
    # Test sparse structure detection
end

@testset "BFGS Outer Optimization" begin
    # Test convergence to same λ as IPNewton
    # Test gradient clamping activates for large λ
    # Test bounded optimization respects constraints
end
```

### Accuracy Tests

Compare λ values from:
- Current implementation (IPNewton outer)
- New BFGS outer
- Gold-standard CV10 (for reference)

Tolerance: |λ_BFGS - λ_IPNewton| / λ_IPNewton < 0.1 (10% relative difference)

### Performance Tests

```julia
using BenchmarkTools

# Benchmark LOO solve
@benchmark _solve_loo_newton_step($chol_H, $H_i, $g_i)           # Current
@benchmark _solve_loo_newton_step_fast($chol_H, $H_i, $g_i)     # New

# Benchmark full PIJCV
@benchmark fit($model; select_lambda=:pijcv, outer_optimizer=:ipnewton)  # Current
@benchmark fit($model; select_lambda=:pijcv, outer_optimizer=:bfgs)      # New
```

---

## Agent Handoff Plan

### Session 1: Cholesky Downdate (Est. 2-3 hours)

**Prompt for Agent 1:**
```
Read the handoff document at scratch/PIJCV_EFFICIENCY_HANDOFF_2026-01-25.md.
Implement Phase 1: Cholesky Downdate Optimization.

Tasks:
1. Profile _solve_loo_newton_step (line 3041 of smoothing_selection.jl) to confirm eigendecomposition bottleneck
2. Implement _woodbury_solve function using the Woodbury identity formula in the handoff doc
3. Create _solve_loo_newton_step_fast that leverages low-rank structure of H_i
4. Add unit test in MultistateModelsTests/unit/test_pijcv.jl comparing accuracy
5. Benchmark on n=1000, p=20 model

Do NOT modify existing _solve_loo_newton_step - add new functions alongside it.
Run tests after each change.
```

**Acceptance Criteria:**
- [ ] New `_woodbury_solve` function passes accuracy tests
- [ ] `_solve_loo_newton_step_fast` matches old results within 1e-10
- [ ] Benchmark shows >5x speedup on LOO solve

---

### Session 2: BFGS Outer Optimization (Est. 2-3 hours)

**Prompt for Agent 2:**
```
Read the handoff document at scratch/PIJCV_EFFICIENCY_HANDOFF_2026-01-25.md.
Implement Phase 2: BFGS Outer Optimization.

Tasks:
1. Create _nested_optimization_pijcv_bfgs in smoothing_selection.jl (parallel to existing _nested_optimization_pijcv)
2. Implement gradient_with_clamping! for indefiniteness detection
3. Compute dβ̂/dρⱼ via implicit differentiation: dβ̂/dρⱼ = -λⱼ H_λ⁻¹ Sⱼ β̂
4. Add outer_optimizer kwarg to fit() in fit_penalized.jl, default to :ipnewton (current behavior)
5. Add unit tests comparing BFGS vs IPNewton λ values

Use Optim.LBFGS() from OptimizationOptimJL for bounded optimization.
Do NOT remove existing IPNewton code - this is an additional option.
```

**Acceptance Criteria:**
- [ ] `outer_optimizer=:bfgs` option works end-to-end
- [ ] BFGS λ values within 10% of IPNewton values
- [ ] All existing PIJCV tests still pass

---

### Session 3: Integration & Benchmarking (Est. 1-2 hours)

**Prompt for Agent 3:**
```
Read scratch/PIJCV_EFFICIENCY_HANDOFF_2026-01-25.md.
Implement Phase 3: Integration & Testing.

Tasks:
1. Update docstrings for new options (outer_optimizer kwarg)
2. Run full PIJCV test suite: julia --project MultistateModelsTests -e 'using Pkg; Pkg.test()'
3. Run longtest_splines to verify accuracy preserved
4. Create performance comparison: PIJCV (old) vs PIJCV (new with both optimizations)
5. Document results in a brief report

Target: 3-5x overall speedup for n=1000, p~20.
```

**Acceptance Criteria:**
- [ ] All tests pass
- [ ] Documented speedup of at least 3x
- [ ] No regression in λ selection accuracy

---

## Specific Action Items (Prioritized)

### High Priority (Do First)

| # | Task | File | Function/Location | Est. Time |
|---|------|------|-------------------|-----------|
| 1 | Profile eigendecomposition cost | smoothing_selection.jl | `_solve_loo_newton_step` L3041 | 15 min |
| 2 | Implement Woodbury solve | smoothing_selection.jl | New `_woodbury_solve` | 45 min |
| 3 | Create fast LOO solve | smoothing_selection.jl | New `_solve_loo_newton_step_fast` | 30 min |
| 4 | Create BFGS outer function | smoothing_selection.jl | New `_nested_optimization_pijcv_bfgs` | 1 hr |
| 5 | Implement gradient clamping | smoothing_selection.jl | New `gradient_with_clamping!` | 30 min |

### Medium Priority (After Core Implementation)

| # | Task | File | Function/Location | Est. Time |
|---|------|------|-------------------|-----------|
| 6 | Add `outer_optimizer` kwarg | fit_penalized.jl | `fit()` dispatch | 20 min |
| 7 | Add implicit diff for dβ̂/dρ | smoothing_selection.jl | Inside BFGS function | 30 min |
| 8 | Unit tests for Cholesky | test_pijcv.jl | New `@testset` | 30 min |
| 9 | Unit tests for BFGS | test_pijcv.jl | New `@testset` | 30 min |

### Low Priority (Polish)

| # | Task | File | Est. Time |
|---|------|------|-----------|
| 10 | Update docstrings | Multiple | 20 min |
| 11 | Performance benchmark report | New file | 30 min |
| 12 | Run longtest_splines | CI/local | 2 hr (runtime) |

---

## Quick Start for New Agent

```bash
# 1. Read the skill file first
cat .github/skills/codebase-knowledge/SKILL.md

# 2. Clear cache and load package
cd "/Users/fintzij/Library/CloudStorage/OneDrive-BristolMyersSquibb/Documents/Julia packages/MultistateModels.jl"
rm -rf ~/.julia/compiled/v1.*/MultistateModels*

# 3. Key file to modify
# src/inference/smoothing_selection.jl - lines 3041-3150 for Cholesky, 1700-1870 for nested opt

# 4. Run tests
cd MultistateModelsTests && julia --project -e 'using Pkg; Pkg.test()'
```

---

## Handoff Checklist

- [x] Current implementation understood
- [x] Mathematical foundation documented
- [x] Pseudocode for new algorithms
- [x] File locations identified
- [x] Testing strategy defined
- [x] Expected performance gains estimated
- [x] Agent handoff prompts written
- [x] Specific action items prioritized

**Ready for implementation.**
