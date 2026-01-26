# PIJCV Smoothing Parameter Selection - Investigation Handoff

**Date**: January 24, 2026  
**Status**: BROKEN - needs systematic investigation  
**Priority**: High - this is the default λ selection method

---

## Problem Statement

PIJCV (Predictive Infinitesimal Jackknife Cross-Validation) for smoothing parameter selection is not working correctly. The algorithm consistently selects extreme λ values (converging to upper bound ~2980) instead of finding the optimal smoothing level.

## Reference Material

**Primary Source**: Wood (2024) "On Neighbourhood Cross Validation" arXiv:2404.16490v4
- Full text available at: `scratch/refs/wood_ncv.txt`
- Key sections: Section 2 (NCV definition), Section 3 (Optimization)

## What We Know

### 1. The Bug Symptom
When fitting a spline model with `λ=:pijcv`:
- λ converges to upper bound (exp(8) ≈ 2980)
- This causes extreme over-smoothing
- PERF and EFS methods work correctly (select reasonable λ values)

### 2. Root Cause Analysis (Partially Complete)
Previous investigation found:
- At **fixed β** (unpenalized MLE), V(λ) is monotonically decreasing with λ
- At **matched β̂(λ)** for each λ, V(λ) shows proper U-shape with minimum around log(λ)≈0
- This suggests the issue is in how β and λ are jointly optimized

### 3. Wood's Algorithm (From Paper)

From `scratch/refs/wood_ncv.txt` lines 494-500:
> "In practice this will involve nested optimization. An outer optimizer seeks the best ρ according to (2), with each trial ρ in turn requiring an inner optimization to obtain the corresponding β̂."

Key insight: "Such nested strategies are not as computationally costly as they naively appear, because the previous iterate's β̂ value serves as an ever better starting value for the inner optimization as the outer optimization converges."

### 4. Current Implementation State

File: `src/inference/smoothing_selection.jl`, function `select_smoothing_parameters` (lines ~1440-1600)

The implementation has been modified multiple times and is currently in an unstable state. Recent changes attempted:
1. Nested optimization where each V(λ) evaluation fits β̂(λ) first
2. Using Ipopt with AutoForwardDiff for outer optimization
3. Extracting Float64 values for inner optimization

**Current issues**:
- ForwardDiff Dual number handling is broken
- Unclear if the nested optimization structure is correct
- No systematic verification against Wood's algorithm

---

## Tasks for Investigation

### Task 1: Understand Wood's NCV Algorithm
Read `scratch/refs/wood_ncv.txt` carefully and document:

1. **The NCV Criterion (Equation 2)**:
   - What exactly is V(λ)?
   - How are the LOO perturbations Δ^{-i} computed?
   - What is the Newton step formula?

2. **The Optimization Strategy (Section 3)**:
   - How does nested optimization work?
   - What is optimized in outer loop vs inner loop?
   - How should gradients be computed?

3. **The Key Approximations**:
   - Is β̂(λ) treated as a function of λ for differentiation?
   - Or is the gradient ∂V/∂λ computed at fixed β̂?

### Task 2: Audit Current Implementation

Compare implementation against spec:

1. **Criterion computation** (`compute_pijcv_criterion`):
   - Does it match Equation 2?
   - Are LOO perturbations correct?

2. **Optimization structure** (`select_smoothing_parameters`):
   - Does it match Wood's nested optimization?
   - Where do gradients come from?

3. **AD compatibility**:
   - What needs to be differentiable?
   - What can be non-differentiable?

### Task 3: Design Correct Implementation

Based on understanding, either:
- Fix the current implementation
- Or rewrite from scratch with clear structure

### Task 4: Verify with Tests

Create diagnostic test that:
1. Computes V(λ) at a grid of λ values using the CORRECT algorithm
2. Verifies U-shape exists
3. Verifies optimizer finds the minimum

---

## Key Files

| File | Purpose |
|------|---------|
| `scratch/refs/wood_ncv.txt` | Wood (2024) NCV paper |
| `src/inference/smoothing_selection.jl` | Main implementation |
| `MultistateModelsTests/unit/test_pijcv.jl` | Unit tests |

## Key Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `select_smoothing_parameters` | smoothing_selection.jl:1440 | Main entry point |
| `compute_pijcv_criterion` | smoothing_selection.jl:~1100 | Computes V(λ) |
| `fit_penalized_beta` | smoothing_selection.jl:~180 | Fits β̂(λ) |
| `compute_subject_gradients` | smoothing_selection.jl:~700 | Per-subject ∇ℓᵢ |
| `compute_subject_hessians_fast` | smoothing_selection.jl:~800 | Per-subject ∇²ℓᵢ |

---

## Critical Questions to Answer

1. **Does V(λ) in Wood's formulation implicitly depend on β̂(λ)?**
   - If yes: gradient ∂V/∂λ must account for ∂β̂/∂λ (implicit differentiation)
   - If no: gradient is simpler but requires nested optimization

2. **What is the correct gradient formula?**
   - Wood mentions BFGS for outer optimization - does this use numerical gradients?
   - Or is there an analytic gradient formula?

3. **Is "performance iteration" a valid alternative to nested optimization?**
   - The old code alternated between optimizing λ and β
   - Wood's paper suggests nested optimization
   - Are these equivalent? Under what conditions?

---

## Testing Commands

```bash
# Run PIJCV unit tests
cd MultistateModelsTests && julia --project -e 'using Test; include("unit/test_pijcv.jl")'

# Quick diagnostic (after fixing)
julia --project -e '
using MultistateModels, DataFrames, Random
Random.seed!(123)
# ... create test data ...
result = fit(model; verbose=true, λ=:pijcv)
println("Selected λ: ", result.SmoothingParameters)
'
```

---

## What NOT to Do

1. **Don't guess at fixes** - understand the algorithm first
2. **Don't use finite differences** - the project requires AD
3. **Don't change tolerances** to make tests pass
4. **Don't skip reading the paper** - it has the answer

---

## Expected Outcome

A working PIJCV implementation that:
1. Correctly implements Wood's NCV algorithm
2. Selects reasonable λ values (not at bounds)
3. Uses AD for efficiency
4. Passes all existing tests
5. Is documented with clear connection to the paper
