# PIJCV Gradient via Full AD (Option A Only)

**Date**: 2026-01-27 (updated 2026-01-28)  
**Branch**: `penalized_splines`  
**Status**: ✅ INTEGRATED INTO CODEBASE - Production `fit()` uses correct gradient

---

## ✅ COMPLETED: Correct PIJCV AD Gradient (2026-01-28)

**The analytical gradient for the CORRECT PIJCV formula is now working.**

### Validation Results

| ρ | dV/dρ (analytical) | dV/dρ (FD) | Ratio |
|---|---|---|---|
| 0.0 | 0.38710 | 0.38710 | 1.0000 |
| 1.0 | 0.39951 | 0.39951 | 1.0000 |
| 2.0 | 0.57071 | 0.57071 | 1.0000 |
| 3.0 | 1.02390 | 1.02390 | 1.0000 |
| 4.0 | 1.59791 | 1.59791 | 1.0000 |

**Working implementation**: `scratch/test_correct_pijcv_ad_v5.jl`

### Key Breakthrough: Sign Error Fix

The critical bug was in `dgᵢ/dρ`. Since gᵢ = -∇ℓᵢ(β̂) and Hᵢ = -∇²ℓᵢ(β̂):

**WRONG**: dgᵢ/dρ = -Hᵢ·dβ̂/dρ  
**CORRECT**: dgᵢ/dρ = +Hᵢ·dβ̂/dρ

### Third Derivatives ARE Required

Contrary to initial hopes, third derivatives (∂Hᵢ/∂β) are necessary for correct gradients. The "simplified" chain rule (ignoring third derivatives) achieved only 87-105% accuracy. With full third derivatives, we get machine precision.

---

## ⚠️ CRITICAL: Correct PIJCV Formulation

**Previous implementation was WRONG.** We were using a quadratic approximation of V, but the correct NCV criterion (Wood 2024, Section 2, Equation 2) evaluates the loss at **pseudo-estimates**:

### Correct Formula

For leave-one-out cross-validation:

**V(ρ) = Σᵢ -ℓᵢ(β̃₋ᵢ)**

where:
- **β̃₋ᵢ = β̂(ρ) - Δ⁻ⁱ** is the pseudo-estimate (one Newton step from β̂)
- **Δ⁻ⁱ = (H_λ - Hᵢ)⁻¹ gᵢ** is the LOO step
- **gᵢ = -∇ℓᵢ(β̂)** is the per-subject score at the full MLE
- **Hᵢ = -∇²ℓᵢ(β̂)** is the per-subject Hessian at the full MLE
- **H_λ = Σᵢ Hᵢ + Σⱼ λⱼ Sⱼ** is the penalized Hessian

### Why This Matters for AD

The correct formulation requires **third derivatives** (∂Hᵢ/∂β) for correct gradients. Initial hopes that third derivatives could be ignored were wrong—the "simplified" chain rule only achieved 87-105% accuracy.

### The Wrong Formula (what we had)

V_wrong = Σᵢ [-ℓᵢ(β̂) + gᵢᵀΔ⁻ⁱ + ½(Δ⁻ⁱ)ᵀHᵢΔ⁻ⁱ]

This is a quadratic approximation of ℓᵢ(β̃₋ᵢ), not the actual value!

---

## ✅ Working Algorithm (Validated 2026-01-27)

### Step 1: dβ̂/dρ via ImplicitDifferentiation.jl

```julia
implicit_beta = ImplicitFunction(forward_solve, optimality_conditions;
    representation=MatrixRepresentation(),
    linear_solver=DirectLinearSolver())

dbeta_drho = ForwardDiff.jacobian(ρ_vec -> implicit_beta(ρ_vec)[1], [ρ])[:, 1]
```

### Step 2: Third Derivative Tensors ∂Hᵢ/∂β

```julia
for i in 1:n_subj
    H_flat_jac = ForwardDiff.jacobian(
        β -> vec(-ForwardDiff.hessian(b -> loglik_subject(b, data, i), β)),
        β_opt
    )
    dH_dbeta_i = reshape(H_flat_jac, n_beta, n_beta, n_beta)
end
```

### Step 3: dH_λ/dρ with Third Derivatives

```julia
dH_λ_drho = λ * S_full
for i in 1:n_subj
    for l in 1:n_beta
        dH_λ_drho .+= dH_dbeta_i[:,:,l] * dbeta_drho[l]
    end
end
```

### Step 4: Per-Subject Gradient (CORRECT SIGNS)

```julia
for i in 1:n_subj
    # gᵢ = -∇ℓᵢ(β̂), Hᵢ = -∇²ℓᵢ(β̂)
    H_loo = H_λ - Hᵢ
    Δᵢ = H_loo \ gᵢ
    β_tilde_i = β_opt - Δᵢ
    
    grad_ll_at_pseudo = ForwardDiff.gradient(b -> loglik_subject(b, data, i), β_tilde_i)
    
    # CRITICAL: dgᵢ/dρ = +Hᵢ·dβ̂/dρ (not -Hᵢ!)
    dgᵢ_drho = Hᵢ * dbeta_drho
    
    # dHᵢ/dρ = (∂Hᵢ/∂β)·dβ̂/dρ
    dHᵢ_drho = zeros(n_beta, n_beta)
    for l in 1:n_beta
        dHᵢ_drho .+= dH_dbeta_i[:,:,l] * dbeta_drho[l]
    end
    
    # dH_loo/dρ = dH_λ/dρ - dHᵢ/dρ
    dH_loo_drho = dH_λ_drho - dHᵢ_drho
    
    # dΔᵢ/dρ
    dDelta_drho = H_loo \ (dgᵢ_drho - dH_loo_drho * Δᵢ)
    dbeta_tilde_drho = dbeta_drho - dDelta_drho
    
    # dVᵢ/dρ = -∇ℓᵢ(β̃₋ᵢ)ᵀ · dβ̃₋ᵢ/dρ
    dV_i_drho = -dot(grad_ll_at_pseudo, dbeta_tilde_drho)
    dV_drho_total += dV_i_drho
end
```

---

## 📋 Current Focus: Implement Correct PIJCV with AD

**Decision**: Option B (analytical gradient) is **REJECTED** due to using wrong formula AND neglecting third derivative terms. We must implement **Option A: Full AD** with the **correct PIJCV formula**.

### Key Insight

The correct V = Σᵢ -ℓᵢ(β̃₋ᵢ) requires differentiating through:
1. **β̂(ρ)** - handled by ImplicitDifferentiation.jl via IFT
2. **Δ⁻ⁱ(β̂, ρ)** - depends on gᵢ(β̂), Hᵢ(β̂), H_λ(β̂, ρ)
3. **ℓᵢ(β̃₋ᵢ)** - standard likelihood evaluation

Since gᵢ, Hᵢ, and ℓᵢ all depend on β̂, and β̂ depends on ρ, the chain rule gives:

dV/dρ = Σᵢ [-∇ℓᵢ(β̃₋ᵢ)ᵀ · dβ̃₋ᵢ/dρ]

where dβ̃₋ᵢ/dρ = dβ̂/dρ - d(Δ⁻ⁱ)/dρ

### Mooncake Testing Results (2026-01-27)

**Mooncake + ImplicitDifferentiation.jl: ❌ DOES NOT WORK**

When using `DifferentiationInterface.gradient(objective, AutoMooncake(), x)`, Mooncake attempts to trace through the *entire* objective function, including the inner optimizer (Ipopt). This fails with `MooncakeRuleCompilationError`.

**ForwardDiff + ImplicitDifferentiation.jl: ✅ WORKS CORRECTLY**

```julia
# This works perfectly:
implicit_beta = ImplicitFunction(forward_solve, optimality_conditions;
    backends=(x=AutoForwardDiff(), y=AutoForwardDiff()))

# dβ̂/dρ matches finite differences to 8 decimal places:
grad_implicit = ForwardDiff.gradient(ρ -> sum(implicit_beta(ρ)[1]), [2.0])
# Result: [-0.0636219...]
# FD:     -0.0636219...
# Ratio:  0.9999999870711714
```

### Updated Architecture (ForwardDiff + ID.jl)

The correct approach uses ForwardDiff as the outer AD backend with ImplicitDifferentiation.jl:

```julia
# 1. Define forward solve and conditions for ImplicitDifferentiation.jl
function forward_solve(log_lambda)
    λ = exp.([ForwardDiff.value(x) for x in log_lambda])
    β_opt = inner_optimizer(model, λ)  # Opaque to AD
    return β_opt, nothing
end

function conditions(log_lambda, β, z)
    λ = exp.(log_lambda)
    return ∇β_loglik(β) - Σⱼ λⱼ Sⱼ β  # AD-compatible
end

# 2. Create implicit function with ForwardDiff backends
implicit_beta = ImplicitFunction(forward_solve, conditions;
    backends=(x=AutoForwardDiff(), y=AutoForwardDiff()))

# 3. Full PIJCV gradient via chain rule
# dV/dρ = (∂V/∂β)·(dβ̂/dρ) + (∂V/∂λ)·(dλ/dρ)
function pijcv_gradient(ρ)
    # dβ̂/dρ from ImplicitDifferentiation.jl (IFT)
    dbeta_drho = ForwardDiff.jacobian(ρ -> implicit_beta(ρ)[1], ρ)
    
    β_opt = implicit_beta(ρ)[1]
    λ = exp.(ρ)
    
    # ∂V/∂β and ∂V/∂λ via ForwardDiff
    dV_dbeta = ForwardDiff.gradient(β -> V(β, λ), β_opt)
    dV_dlambda = ForwardDiff.gradient(λ -> V(β_opt, λ), λ)
    
    # Chain rule
    return dV_dbeta' * dbeta_drho + dV_dlambda .* λ
end
```

### Why Option B Failed (2026-01-28)

| Metric | Analytical (Option B) | True (FD) | Problem |
|--------|----------------------|-----------|---------|
| log(λ) at optimum | 2.77 | 3.12 | **0.35 shift** |
| λ at optimum | ~16 | ~23 | **30% bias** |
| V at optimum | 160.69 | 160.66 | Negligible |

The analytical formula neglects ∂Hᵢ/∂β (third derivatives of log-likelihood). Near the optimum where gradients are small, this omission dominates and causes systematic bias. **This is not a bug—it's a fundamental limitation of the Newton-step approximation.**

### Option A Path Forward

**Key Insight** (verified 2026-01-27): Zygote CAN differentiate through ForwardDiff.hessian. We can bypass ImplicitDifferentiation.jl's buggy Zygote extension by implementing IFT manually inside a Zygote-differentiable function.

**Architecture**:
```julia
function pijcv_objective_zygote(log_λ)
    λ = exp.(log_λ)
---

## 🚫 Why Option B (Analytical Gradient) Was Rejected

**Date**: 2026-01-28

### The Fundamental Problem

The analytical gradient formula:
```
∂V/∂ρₖ = gᵀφₖ + Σᵢ[(Δ⁻ⁱ)ᵀHᵢφₖ + rᵢᵀψᵢₖ]
```

**assumes ∂Hᵢ/∂β = 0** (i.e., per-subject Hessians are constant w.r.t. β). This is false—Hᵢ involves second derivatives of log-likelihood, so ∂Hᵢ/∂β involves third derivatives.

### Empirical Evidence

Testing at log(λ) values from -1 to 5:

| log(λ) | Analytical | FD | Ratio |
|--------|------------|-----|-------|
| -1.00 | +0.20 | +0.18 | 1.11 |
| 0.00 | -0.07 | -0.08 | 0.83 |
| 2.00 | -0.19 | -0.26 | 0.71 |
| **3.00 (opt)** | **+0.10** | **-0.05** | **-1.89** |
| 4.00 | +0.89 | +0.64 | 1.39 |

**At the optimum, the signs disagree.** The analytical gradient crosses zero ~0.35 earlier (on log scale) than the true gradient, causing 30% underestimation of optimal λ.

### Why This Cannot Be Fixed Without Full AD

Adding ∂Hᵢ/∂β requires:
1. Computing third derivatives of per-subject log-likelihoods
2. Implementing complex tensor contractions
3. Significant code complexity with high bug potential

**Full AD (Option A) computes the correct gradient automatically** without manually deriving/implementing third derivatives.

---

## ✅ Option A: Full AD with Zygote + Manual IFT

### Why This Works

**Verified (2026-01-27)**: Zygote CAN differentiate through ForwardDiff.hessian:
```julia
function inner_with_hessian(x)
    f(y) = sum(y.^3) + 0.5 * dot(y, y)
    H = ForwardDiff.hessian(f, x)
    return tr(H)
end

grad_zygote = Zygote.gradient(inner_with_hessian, x_test)[1]  # ✓ Works!
```

This means we can use:
- **ForwardDiff** (forward-mode) for outer differentiation w.r.t. ρ = log(λ)
- **ImplicitDifferentiation.jl** with ForwardDiff backends for ∂β̂/∂ρ
- **ForwardDiff** for inner Hessian computation (gᵢ, Hᵢ)

### AD Backend Summary (2026-01-27)

| Backend | Status | Notes |
|---------|--------|-------|
| **ForwardDiff + ID.jl** | ✅ WORKS | dβ̂/dρ matches FD to 8 decimals |
| **Mooncake + ID.jl** | ❌ FAILS | `MooncakeRuleCompilationError` on Ipopt calls |
| **Zygote + ID.jl** | ❌ FAILS | `DimensionMismatch` with vector outputs |

**Recommended approach**: ForwardDiff for all differentiation, using ImplicitDifferentiation.jl with `backends=(x=AutoForwardDiff(), y=AutoForwardDiff())`.

---

## ⚠️ CRITICAL CONSTRAINT

**NEVER USE FINITE DIFFERENCES.** This project requires analytical gradients computed via automatic differentiation or closed-form derivations. Finite differences violate core project principles:
1. They are numerically unstable
2. They scale poorly with dimension  
3. They defeat the purpose of implicit differentiation infrastructure
4. They are explicitly forbidden in `.github/copilot-instructions.md`

All gradient computations must use ForwardDiff, ReverseDiff, Zygote, Enzyme, or analytically-derived formulas.

---

## Implementation Plan (Option A Only)

### Phase 0: Implement ForwardDiff + ID.jl Full AD ✅ COMPLETED (2026-01-28)

**Goal**: Compute exact PIJCV gradients using ForwardDiff + ImplicitDifferentiation.jl.

| Task | Status | Description |
|------|--------|-------------|
| 0.1 | [x] | Test Mooncake + ID.jl (FAILED - cannot trace through Ipopt) |
| 0.2 | [x] | Test ForwardDiff + ID.jl for dβ̂/dρ (WORKS - matches FD to 8 decimals) |
| 0.3 | [x] | Implement chain rule: dV/dρ = -∇ℓᵢ(β̃₋ᵢ)ᵀ·dβ̃₋ᵢ/dρ with third derivatives |
| 0.4 | [x] | Verify gradient matches FD within 5% at log(λ) ∈ {0, 1, 2, 3, 4} (ratio ≈ 1.0000) |
| 0.5 | [x] | Verify analytical and FD zero-crossings agree (validated in test_correct_pijcv_ad_v5.jl) |
| 0.6 | [x] | Integrate into `_nested_optimization_pijcv_implicit` |
| 0.7 | [x] | Run unit test suite (2079/2079 passed) |
| 0.8 | [ ] | Run full test suite with MSM_TEST_LEVEL=full |

### Codebase Integration (2026-01-28)

**Production `fit()` now uses the correct gradient by default.**

Files added/modified:
- `src/inference/smoothing_selection/implicit_diff.jl` (NEW - 1792 lines):
  - `compute_pijcv_with_gradient` with CORRECT formula + third derivatives
  - `_compute_subject_third_derivatives` helper function
  - `ncv_criterion_and_gradient` computes `dbeta_drho` via ImplicitDifferentiation.jl
  - `_nested_optimization_pijcv_implicit` main entry point for ExactData
  - Support for MPanelData and MCEMSelectionData

- `src/types/penalties.jl`:
  - `PIJCVSelector.use_implicit_diff` defaults to `true`

- `src/inference/smoothing_selection/dispatch_exact.jl`:
  - Routes to `_nested_optimization_pijcv_implicit` when `use_implicit_diff=true` (default)

**Validation**:
- Unit tests: 2079/2079 passed
- Gradient vs FD ratio ≈ 1.0000 across log(λ) ∈ {0, 1, 2, 3, 4}

**Note**: Files are currently UNTRACKED in git. Commit pending.

### Implementation Architecture (ForwardDiff + ImplicitDifferentiation.jl)

```julia
using ImplicitDifferentiation, ADTypes

# Step 1: Define forward and conditions functions for ID.jl
function forward_solve(λ::AbstractVector, model, data)
    # Solve penalized MLE for β̂ at given λ
    return actual_inner_optimization(model, data, λ)
end

function conditions(λ::AbstractVector, β::AbstractVector, model, data)
    # Score equation: ∇ℓ(β) - Sλβ = 0 at optimum
    return compute_penalized_score(β, λ, model, data)
end

# Step 2: Create implicit function with ForwardDiff backends
implicit_beta = ImplicitFunction(
    (λ, args...) -> forward_solve(λ, args...),
    (λ, β, args...) -> conditions(λ, β, args...);
    backends=(x=AutoForwardDiff(), y=AutoForwardDiff())
)

# Step 3: V computation that takes (β, λ) as inputs
function compute_V(β::AbstractVector, λ::AbstractVector, model, data)
    V = zero(promote_type(eltype(β), eltype(λ)))
    H_λ = compute_penalized_hessian(β, λ, model, data)
    
    for i in 1:n_subjects
        ℓᵢ, gᵢ, Hᵢ = compute_subject_derivatives(β, model, data, i)
        H_loo = H_λ - Hᵢ
        Δᵢ = H_loo \ gᵢ
        V += -ℓᵢ + dot(gᵢ, Δᵢ) + 0.5 * dot(Δᵢ, Hᵢ * Δᵢ)
    end
    return V
end

# Step 4: Full PIJCV objective with chain rule gradient
function pijcv_objective(ρ, model, data)
    λ = exp.(ρ)
    β̂ = implicit_beta(λ, model, data)  # ID.jl handles dβ̂/dλ via IFT
    return compute_V(β̂, λ, model, data)
end

# Step 5: Get gradient via ForwardDiff + chain rule
# dV/dρ = (∂V/∂β)·(dβ̂/dρ) + (∂V/∂λ)·(dλ/dρ)
# where dλ/dρ = diag(λ) and dβ̂/dρ = (dβ̂/dλ)·diag(λ) via IFT
grad_V = ForwardDiff.gradient(ρ -> pijcv_objective(ρ, model, data), log_λ)
```

### Key Insight: Why This Works

1. **ImplicitDifferentiation.jl** uses ForwardDiff to compute both:
   - Forward pass: Jacobian of conditions w.r.t. β (for implicit function theorem)
   - Backward pass: Jacobian of conditions w.r.t. λ (for the derivative dβ̂/dλ)

2. **ForwardDiff outer differentiation** computes the total derivative dV/dρ via chain rule, with ID.jl providing the correct dβ̂/dρ term.

3. **No tracing through optimizer**: ID.jl uses the implicit function theorem to avoid differentiating through the Ipopt solver calls.

### Key Components to Implement

1. **`compute_subject_derivatives(β, model, data, i)`**: Returns (ℓᵢ, gᵢ, Hᵢ) using ForwardDiff. Must be Dual-number compatible.

2. **Forward solve wrapper**: Calls existing `fit_inner_coefficients` without modification.

3. **Conditions function**: Returns the score equation residual (should be ≈0 at optimum).

---

## Implementation Plan (Updated 2026-01-27)

### Phase 0: Test ForwardDiff + ID.jl Full Gradient (CURRENT PRIORITY)

**Goal**: Verify that ForwardDiff + ImplicitDifferentiation.jl computes correct PIJCV gradients via chain rule.

**Status**: dβ̂/dρ verified ✅, full gradient via chain rule pending

| Task | Status | Description |
|------|--------|-------------|
| 0.1 | [x] | Test Mooncake + ID.jl (FAILED - cannot trace through Ipopt) |
| 0.2 | [x] | Test ForwardDiff + ID.jl for dβ̂/dρ (WORKS - matches FD to 8 decimals) |
| 0.3 | [ ] | Test ForwardDiff for ∂V/∂β and ∂V/∂λ |
| 0.4 | [ ] | Combine via chain rule: dV/dρ = (∂V/∂β)·(dβ̂/dρ) + (∂V/∂λ)·(dλ/dρ) |
| 0.5 | [ ] | Verify gradient matches FD within 5% at log(λ) ∈ {-1, 0, 1, 2, 3, 4, 5} |
| 0.6 | [ ] | Verify analytical and FD zero-crossings agree within 0.1 on log scale |

**Test Script**: `scratch/test_forwarddiff_implicit.jl`

**Key Results (2026-01-27)**:
- dβ̂/dρ via ImplicitDifferentiation.jl: `[-0.0636219...]`
- dβ̂/dρ via finite difference: `[-0.0636219...]`  
- Ratio: `0.9999999870711714` ✅

### Phase 0b: Debug Option B Analytical Gradient (SECONDARY PRIORITY)

**Goal**: Find and fix the bug causing the analytical gradient to be ~83% of the FD value.

The gradient formula is:
```
∂V/∂ρₖ = gᵀφₖ + Σᵢ[(Δ⁻ⁱ)ᵀHᵢφₖ + rᵢᵀψᵢₖ]
```

where `rᵢ = gᵢ + HᵢΔ⁻ⁱ` and `ψᵢₖ = H_{λ,-i}⁻¹(Hᵢφₖ - λₖSₖΔ⁻ⁱ)`.

To debug, separate into:
- **Term 1**: `gᵀφₖ` (contribution from -ℓᵢ(β̂) through β dependence)
- **Term 2**: `Σᵢ(Δ⁻ⁱ)ᵀHᵢφₖ` (contribution from gᵢ changing in gᵢᵀΔ⁻ⁱ)
- **Term 3**: `Σᵢrᵢᵀψᵢₖ` (contribution from Δ⁻ⁱ changing + quadratic term)

Compare each analytically-computed term against its finite difference.

**Files**:
- Debug script: `scratch/test_gradient.jl`
- Main implementation: `src/inference/smoothing_selection/implicit_diff.jl` lines 875-1065
- Unit tests: `MultistateModelsTests/unit/test_implicit_diff.jl` section 4b

### Phase 1: Enable Integration Tests (After Phase 0)

| Task | Status | Description |
|------|--------|-------------|
| 1.1 | [ ] | Uncomment section 5 in test_implicit_diff.jl |
| 1.2 | [ ] | Verify λ matches legacy within 10% (tighter than before) |
| 1.3 | [ ] | Add performance benchmark test |

### Phase 2: Remove Legacy Code and Option B (After Phase 1)

| Task | Status | Description |
|------|--------|-------------|
| 2.1 | [ ] | Delete `_nested_optimization_pijcv` from dispatch_general.jl |
| 2.2 | [ ] | Delete `_nested_optimization_pijcv_markov` from dispatch_markov.jl |
| 2.3 | [ ] | Delete `_nested_optimization_pijcv_mcem` from dispatch_mcem.jl |
| 2.4 | [ ] | Remove `use_implicit_diff` field from PIJCVSelector |
| 2.5 | [ ] | **Delete `compute_pijcv_with_gradient` (Option B code)** |
| 2.6 | [ ] | Simplify dispatch files |
| 2.7 | [ ] | Update all docstrings |

---

## Code Complexity Estimate

| Component | Lines | Description |
|-----------|-------|-------------|
| `compute_V_zygote()` | ~80 | Zygote-compatible V computation |
| `compute_subject_derivatives()` | ~60 | Per-subject ℓᵢ, gᵢ, Hᵢ via ForwardDiff |
| Custom Zygote adjoint for IFT | ~40 | Manual IFT pullback |
| Integration wrapper | ~50 | Connect to existing optimizer |
| Tests | ~100 | Comprehensive gradient verification |
| **Total** | **~330** |

### Performance Considerations

For p=50 params, n=500 subjects, n_λ=4:
- **Zygote outer + ForwardDiff inner**: Reverse-mode outer is efficient; ForwardDiff Hessians are fast
- **Memory**: Zygote tape + per-subject Hessians ~200MB for this size
- **Expected speedup vs FD outer**: ~4-10x (FD requires 2×n_λ function evaluations)

---

## Appendix A: Mathematical Foundation

### PIJCV Criterion

V(ρ) = Σᵢ [ -ℓᵢ(β̂) + gᵢᵀ Δ⁻ⁱ + ½(Δ⁻ⁱ)ᵀ Hᵢ Δ⁻ⁱ ]

where:
- ρₖ = log(λₖ) (log-smoothing parameters)
- β̂ = β̂(ρ) is the penalized MLE
- gᵢ = -∇ℓᵢ(β̂) is subject i's loss gradient
- Hᵢ = -∇²ℓᵢ(β̂) is subject i's loss Hessian  
- Δ⁻ⁱ = H_{λ,-i}⁻¹ gᵢ is the Newton step for LOO estimate
- H_{λ,-i} = H_λ - Hᵢ (LOO penalized Hessian)
- H_λ = Σᵢ Hᵢ + Σⱼ λⱼ Sⱼ (full penalized Hessian)

### Complete Gradient (via Full AD)

The full gradient ∂V/∂ρ requires differentiating through:
1. β̂(ρ) — handled by IFT: ∂β̂/∂ρₖ = -H_λ⁻¹(λₖSₖβ̂)
2. gᵢ(β̂) — requires ∂gᵢ/∂β = Hᵢ (second derivatives)
3. Hᵢ(β̂) — requires ∂Hᵢ/∂β (THIRD derivatives)
4. Δ⁻ⁱ(β̂, ρ) — depends on both β̂ and λ

**Option B (analytical) ignores term 3**, causing ~30% bias in λ.

**Option A (full AD)** computes all terms automatically via Zygote + ForwardDiff.

### IFT Formula for Custom Adjoint

The implicit function theorem gives:
```
∂β̂/∂λₖ = -H_λ⁻¹(Sₖβ̂)
```

In log-scale (ρₖ = log λₖ):
```
∂β̂/∂ρₖ = λₖ · ∂β̂/∂λₖ = -λₖ · H_λ⁻¹(Sₖβ̂)
```

This is implemented in the custom Zygote adjoint for `fit_inner_coefficients_for_zygote`.

### Optimizations (Performance tuning after correctness verified)

1. **Pre-factor H_λ**: Compute Cholesky factorization once, reuse for all solves
2. **Parallelize over subjects**: Independent LOO computations
3. **Cache subject derivatives**: Reuse gᵢ, Hᵢ across optimizer iterations if β changes slowly

---

## Appendix B: Testing Strategy

### Unit Tests (Phase 0)

1. **Finite difference verification** (test only, NOT in production):
   ```julia
   @testset "PIJCV gradient accuracy (Full AD)" begin
       for log_λ in [[0.0], [2.0], [3.0], [4.0]]
           V, grad_zygote = pijcv_with_gradient_zygote(log_λ, model, data)
           
           grad_fd = FiniteDiff.finite_difference_gradient(
               ρ -> pijcv_objective(ρ, model, data), log_λ
           )
           
           @test isapprox(grad_zygote, grad_fd, rtol=0.05)  # 5% tolerance
       end
   end
   ```

2. **Zero-crossing agreement**:
   ```julia
   @testset "Optimum location agreement" begin
       # Find where Zygote gradient = 0
       opt_zygote = optimize(ρ -> pijcv_with_gradient_zygote(ρ, model, data)...)
       
       # Find where FD gradient = 0
       opt_fd = optimize(ρ -> pijcv_objective(ρ, model, data), ...)
       
       # Must agree within 0.1 on log scale
       @test isapprox(opt_zygote, opt_fd, atol=0.1)
   end
   ```

3. **IFT pullback verification**: Verify custom adjoint matches ForwardDiff through solve

### Integration Tests (Phase 1)

1. **Optimization convergence**: PIJCV optimization converges to reasonable λ
2. **Recovery of true λ**: On simulated data with known truth, λ̂ close to λ*
3. **Comparison with legacy**: Results match legacy within 10%

---

## Current State (Updated 2026-01-28)

### Completed
- [x] AutoFiniteDiff removed from all src/ files  
- [x] 2079 package tests pass
- [x] Core implicit diff infrastructure works (tests 1-4)
- [x] ImplicitDifferentiation.jl correctly computes dβ/dρ
- [x] `ImplicitFunction` created via `make_implicit_beta_function()`
- [x] **Verified Zygote CAN diff through ForwardDiff.hessian** (key finding)
- [x] **Option B REJECTED**: ~30% λ bias due to missing third derivatives
- [x] **Decision**: Proceed with Option A (Zygote + Manual IFT)

### In Progress (Phase 0)
- [ ] Implement `compute_V_zygote()` - Zygote-compatible V computation
- [ ] Implement `compute_subject_derivatives()` - per-subject ℓᵢ, gᵢ, Hᵢ  
- [ ] Implement custom Zygote adjoint for inner optimization (IFT pullback)
- [ ] Verify gradient matches FD within 5%
- [ ] Verify zero-crossings agree within 0.1 on log scale

### Blocked
- Integration tests (section 5) remain commented out pending Full AD implementation

### To Delete (Phase 2)
- `compute_pijcv_with_gradient()` - Option B code with ~30% bias
- Legacy nested optimization paths
- `use_implicit_diff` field

---

## Files to Modify

### Phase 0 (Current - Implement Full AD)

| File | Changes |
|------|---------|
| `src/inference/smoothing_selection/implicit_diff.jl` | Add Zygote-based PIJCV gradient computation |
| `src/inference/smoothing_selection/zygote_pijcv.jl` | **NEW**: Zygote + manual IFT implementation |
| `MultistateModelsTests/unit/test_implicit_diff.jl` | Add Full AD gradient tests |

### Phase 2 (Cleanup - After Validation)

| File | Changes |
|------|---------|
| `src/inference/smoothing_selection/implicit_diff.jl` | **Delete `compute_pijcv_with_gradient()` (Option B)** |
| `src/inference/smoothing_selection/dispatch_general.jl` | Delete legacy `_nested_optimization_pijcv` |
| `src/inference/smoothing_selection/dispatch_exact.jl` | Simplify to always use Full AD |
| `src/inference/smoothing_selection/dispatch_markov.jl` | Delete legacy, simplify |
| `src/inference/smoothing_selection/dispatch_mcem.jl` | Delete legacy, simplify |
| `src/types/penalties.jl` | Remove `use_implicit_diff` field |
| `MultistateModelsTests/unit/test_implicit_diff.jl` | Uncomment section 5, add tests |

---

## Verification Checklist

- [ ] Pkg.test() passes all tests
- [ ] **Zygote gradient matches FD within 5%** at log_λ ∈ {-1, 0, 1, 2, 3, 4, 5}
- [ ] **Zero-crossings agree within 0.1** on log scale (CRITICAL)
- [ ] Full AD PIJCV λ within 10% of legacy λ  
- [ ] Performance acceptable (<5x legacy runtime)
- [ ] No AutoFiniteDiff in src/
- [ ] No use_implicit_diff parameter (after Phase 2)
- [ ] Option B code deleted (after Phase 2)
- [ ] Legacy functions deleted (after Phase 2)

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Zygote + ForwardDiff integration issues | **Medium** | Verified working in isolation; test incrementally |
| Custom IFT pullback bugs | **Medium** | Verify against ID.jl's ForwardDiff output |
| Memory for Zygote tape | **Medium** | For p=50, n=500: ~200MB. Chunk if needed. |
| Threading incompatible with Zygote | **Medium** | Use sequential path for AD; parallel for evaluation only |
| Performance worse than Option B | **Low** | Option B has ~30% bias—correctness > speed |

### Verified Non-Issues
- ✅ Zygote CAN diff through ForwardDiff.hessian (tested 2026-01-27)
- ✅ Linear algebra `\` with Dual matrices (tested, works)
- ✅ IFT formula verified correct (matches ID.jl ForwardDiff output)

---

## References

1. **Zygote.jl**: https://fluxml.ai/Zygote.jl/
   - Reverse-mode AD for Julia
   - Can differentiate through ForwardDiff calls

2. **ForwardDiff.jl**: https://juliadiff.org/ForwardDiff.jl/
   - Forward-mode AD via dual numbers
   - Used for per-subject Hessian computation

3. **ImplicitDifferentiation.jl**: https://gdalle.github.io/ImplicitDifferentiation.jl/
   - Reference for IFT implementation
   - We bypass its Zygote extension due to bugs

4. **Wood (2011)**: "Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models" JRSSB

---

## Change Log

- **2026-01-28**: **MAJOR REVISION**: Option B REJECTED due to ~30% λ bias. Plan now Option A ONLY.
- **2026-01-28**: Discovered Option B gradient has systematic shift causing wrong optimum location
- **2026-01-27**: Verified Zygote can diff through ForwardDiff.hessian
- **2026-01-27**: Adversarial review completed
- **2026-01-27**: Initial plan based on mgcv analysis
