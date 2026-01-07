# Constrained Optimization Implementation: Handoff Document

**Date:** January 6, 2026 (Updated)  
**Branch:** `penalized_splines`  
**Session:** Box constraints infrastructure + Critical Review  

---

## Executive Summary

This session completed **Phase 1** (bounds infrastructure) and conducted a **critical mathematical review** of the Strategy B implementation plan. All 11 identified issues have been addressed—8 resolved, 2 marked TODO for implementation, 1 deferred.

### Session Accomplishments

| Task | Status | Notes |
|------|--------|-------|
| Bounds infrastructure (`bounds.jl`) | ✅ COMPLETE | ~340 lines, 55 tests |
| Integration with `fit_exact.jl` | ✅ COMPLETE | `lb`/`ub` args added |
| Critical mathematical review | ✅ COMPLETE | 11 issues identified and addressed |
| Full test suite | ✅ PASSING | 1483 tests |
| Natural-scale penalty work | ⬜ PENDING | Ready for Phase 2 |

### Key Design Decisions from Review

1. **Lower bound:** ε = 0 (non-negativity, not positivity)
2. **Ipopt setting:** `honor_original_bounds = "yes"` ensures exact constraint satisfaction
3. **Monotone splines:** Transform penalty matrix S̃ = LᵀSL so penalty on increments equals penalty on coefficients
4. **PIJCV LOO projection:** Project Newton approximation to feasible region via `max.(beta_loo, 0.0)`
5. **Null space:** Standard P-spline behavior—likelihood constrains level/slope
6. **Parameter storage:** Remove redundancy between `flat` and `natural` (simplify codebase)

---

## 1. What Was Implemented

### 1.1 New File: `src/utilities/bounds.jl`

**Purpose:** Automatic generation of parameter bounds for box-constrained optimization.

**Key Functions:**

| Function | Purpose | Phase 1/2 Action |
|----------|---------|------------------|
| `generate_parameter_bounds(model; user_bounds)` | Main API - generates combined bounds | Remove `scale` arg, Dict-only API |
| `_generate_package_bounds(model)` | Package-level bounds by hazard family | Use `NONNEG_LB = 0.0` |
| `_get_baseline_lb(family, n_baseline)` | Per-family lower bounds on natural scale | Update lb values to 0 |
| `_transform_bounds_to_estimation(...)` | Natural → log scale conversion | **REMOVE** |
| `_get_positive_mask(family, n_baseline)` | Identifies which params are log-transformed | **REMOVE** |
| `_resolve_user_bounds(user_bounds, model)` | Convert Dict to flat vectors | Keep, simplify |
| `_get_flat_parnames(model)` | Parameter name extraction | Keep |
| `validate_initial_values(init, lb, ub; parnames)` | Pre-optimization validation | Keep |

**Design Decisions:**

1. **Natural-scale only (after Phase 2):**
   ```julia
   # Only option: natural scale bounds
   lb, ub = generate_parameter_bounds(model)  # Natural scale, lb=0 for non-negative params
   
   # User bounds via Dict (safer than Vector)
   lb, ub = generate_parameter_bounds(model; user_bounds=Dict(:h12_rate => (lb=0.01, ub=100.0)))
   ```

2. **User bounds on natural scale:** Users specify bounds like "rate ∈ [0.01, 100]" which is intuitive.

3. **Intersection rule:** `final_lb = max(pkg_lb, user_lb)` ensures users cannot accidentally loosen safety constraints.

4. **Per-family handling:**

   | Family | Baseline Params | Natural LB (current) | Natural LB (target) | Log Transform |
   |--------|-----------------|---------------------|---------------------|---------------|
   | `:exp` | rate | 1e-10 | 0 | Yes → No |
   | `:wei` | shape, scale | 1e-10 | 0 | Yes → No |
   | `:gom` | shape, rate | -Inf, 1e-10 | -Inf, 0 | No, Yes → No |
   | `:sp` | coefficients | 1e-10 | 0 | Yes → No |
   | `:pt` | rates | 1e-10 | 0 | Yes → No |
   | Covariates | coefficients | -Inf | -Inf | No |

**Phase 1/2 TODO:** 
- Change `POSITIVE_LB = 1e-10` to `NONNEG_LB = 0.0`
- Remove `scale=:estimation` option entirely
- Remove `_transform_bounds_to_estimation` and `_get_positive_mask` functions
- Change API from `user_lb/user_ub` Vectors to `user_bounds` Dict only

### 1.2 Modified: `src/inference/fit_exact.jl`

**Changes:**

1. Added `lb` and `ub` keyword arguments to `_fit_exact()`
2. Added bounds generation call:
   ```julia
   param_lb, param_ub = generate_parameter_bounds(model; user_lb=lb, user_ub=ub)
   ```
3. Added initial value validation:
   ```julia
   validate_initial_values(parameters, param_lb, param_ub; parnames=_get_flat_parnames(model))
   ```
4. Updated all `OptimizationProblem` calls to include `lb=param_lb, ub=param_ub`

**Locations updated:**
- Line ~135: Main unconstrained optimization
- Line ~189: Re-optimization after λ selection
- Line ~256: Constrained optimization (with `lcons`/`ucons`)

### 1.3 Modified: `src/MultistateModels.jl`

- Added `include("utilities/bounds.jl")` after `misc.jl`
- Added `generate_parameter_bounds` to exports

### 1.4 New Test File: `MultistateModelsTests/unit/test_bounds.jl`

**Coverage:** 55 tests across:
- Package bounds for each hazard family (natural scale)
- Package bounds on estimation scale (log transform)
- User bounds via Vector API
- User bounds via Dict API  
- Constraint combination (intersection rule)
- Multi-hazard models
- Initial value validation
- Edge cases (invalid scale, string keys in Dict)

---

## 2. Critical Review: Issue Summary

The implementation plan underwent rigorous mathematical review. All 11 issues are now addressed:

### Resolved Issues (No Action Needed)

| # | Issue | Resolution |
|---|-------|------------|
| 1 | Barrier function in PIJCV Hessian | N/A — PIJCV computes own Hessian via AD, independent of Ipopt |
| 2 | Active constraints in Newton LOO | Projected Newton + `honor_original_bounds="yes"` |
| 3 | Penalty null space | Standard P-spline behavior; likelihood constrains level/slope |
| 4 | Lower bound choice | Use ε = 0 (non-negativity) |
| 5 | Gradient scaling | Non-issue; Ipopt handles internally |
| 6 | Monotone splines | Transform penalty matrix S̃ = LᵀSL (detailed below) |
| 7 | Weibull shape near zero | Use κ ≥ 0; data prevents degenerate fits |
| 8 | Gompertz shape | Correctly unconstrained by design |

### TODO During Implementation

| # | Issue | Action |
|---|-------|--------|
| 9 | Parameter storage redundancy | Remove `flat`/`natural` duplication—simplify codebase |
| 11 | Missing edge case tests | Add tests for active constraints, sparse data, etc. |

### Deferred

| # | Issue | Reason |
|---|-------|--------|
| 10 | PIJCV validation criteria | Focus on parameter handling first |

### Monotone Splines: Penalty Matrix Transformation

For monotone splines, optimization is over increments γ with β = Lγ. To ensure the penalty on increments equals the penalty on coefficients:

**Monotone increasing (`monotone = 1`):**
$$L_+ = \begin{pmatrix} 
1 & 0 & 0 & \cdots \\
1 & w_2 & 0 & \cdots \\
1 & w_2 & w_3 & \cdots \\
\vdots & & & \ddots
\end{pmatrix}$$
where $w_j = (t_{j+k} - t_j)/k$ are I-spline weights.

**Monotone decreasing (`monotone = -1`):**
$$L_- = R \cdot L_+$$ where R is the reversal matrix.

**Transformed penalty:** $\tilde{S} = L^\top S L$

**Implementation:** See `build_monotone_cumsum_matrix()` and `transform_penalty_matrix_for_monotone()` in the plan.

---

## 3. What Remains (Phases 2-6)

The original plan in `scratch/CONSTRAINED_OPTIMIZATION_IMPLEMENTATION_PLAN.md` has 6 phases. Only Phase 1 is complete.

### Phase 2: Natural-Scale Parameterization (CRITICAL)

**Goal:** Make spline hazards work on natural scale internally so penalty is quadratic.

**Current State:**
- Spline coefficients stored on log scale: `β_stored = log(β_natural)`
- Hazard applies `exp()` internally: `h(t) = exp(B(t) · β_stored)`
- Penalty is non-quadratic: `P = λ/2 · exp(β)ᵀ S exp(β)`

**Target State:**
- Store coefficients on natural scale: `β_stored = β_natural`
- Hazard computes directly: `h(t) = B(t) · β_stored`
- Penalty is quadratic: `P = λ/2 · βᵀ S β`
- Box constraints enforce non-negativity: `lb = 0`

**Key Changes:**
- For `monotone=0`: Optimize β directly with β ≥ 0
- For `monotone≠0`: Optimize increments γ directly with γ ≥ 0, use transformed penalty S̃ = LᵀSL

**Files to modify:**
- `src/hazard/spline.jl` - Remove internal `exp()` transformation
- `src/construction/multistatemodel.jl` - Update `_spline_ests2coefs` and `_spline_coefs2ests`
- `src/utilities/transforms.jl` - Update `estimation_to_natural` for `:sp`
- `src/types/infrastructure.jl` - Set `exp_transform=false` for baseline splines
- `src/utilities/penalty_config.jl` - Apply monotone penalty transform

### Phase 3: Penalty System Updates

**Goal:** Simplify penalty computation now that it's quadratic.

**Changes needed:**
- `compute_penalty()` - Remove exp transformation branch (use `exp_transform=false`)
- `_build_penalized_hessian()` - Use simple `H + λS` formula
- `compute_penalty_from_lambda()` - Simplify
- Add monotone penalty matrix transformation

### Phase 4: PIJCV Criterion Updates

**Goal:** PIJCV should work correctly with quadratic penalty.

**Changes needed:**
- `compute_pijcv_criterion()` - Add LOO projection: `max.(beta_loo, 0.0)`
- Verify Newton approximation is valid with box constraints
- Use `honor_original_bounds="yes"` in Ipopt configuration

### Phase 5: Initialization and Storage Cleanup

**Goal:** Simplify parameter storage and initialization.

**Changes needed:**
- Remove redundancy between `parameters.flat` and `parameters.natural` for splines
- `initialize_parameters!()` - Set initial β to positive values directly
- `calibrate_splines!()` - Ensure initial coefficients are positive
- Audit all parameter access patterns

### Phase 6: Testing and Validation

**Tests needed:**
- Spline hazard evaluation with natural-scale coefficients
- Penalty matrix multiplication (verify quadratic form)
- Monotone penalty transformation correctness
- PIJCV λ selection (should select reasonable λ now)
- Recovery tests (spline recovers Weibull shape)
- mgcv comparison benchmark
- Edge cases: active constraints, sparse data, competing risks

---

## 4. Key Files Reference

| File | Purpose | Session Changes |
|------|---------|-----------------|
| `src/utilities/bounds.jl` | NEW: Bounds infrastructure | Created |
| `src/inference/fit_exact.jl` | Exact data fitting | Added `lb`/`ub` support |
| `src/MultistateModels.jl` | Module definition | Added include/export |
| `MultistateModelsTests/unit/test_bounds.jl` | NEW: Bounds tests | Created |
| `scratch/CONSTRAINED_OPTIMIZATION_IMPLEMENTATION_PLAN.md` | Full implementation plan | Reference |

**Files to modify in future phases:**
| File | Phase | Purpose |
|------|-------|---------|
| `src/hazard/spline.jl` | 2 | Natural-scale spline evaluation |
| `src/utilities/transforms.jl` | 2 | Parameter transforms |
| `src/types/infrastructure.jl` | 3 | PenaltyTerm simplification |
| `src/inference/smoothing_selection.jl` | 3-4 | Penalty and PIJCV |
| `src/utilities/initialization.jl` | 5 | Natural-scale initialization |

---

## 5. Test Results

```
Test Summary: | Pass  Total     Time
Unit Tests    | 1483   1483  6m41.6s
```

All tests pass including:
- 55 new bounds tests
- 67 existing penalty infrastructure tests
- All hazard, likelihood, fitting, and simulation tests

---

## 6. Important Context for Next Session

### Why This Work Matters

The PIJCV criterion selects extreme λ values (exp(8) ≈ 2981) because:
1. The penalty `P(θ) = λ/2 · exp(θ)ᵀ S exp(θ)` is non-quadratic in θ
2. The Newton approximation `β̂⁻ⁱ ≈ β̂ + H⁻¹gᵢ` assumes quadratic penalty
3. With exp-transformed penalty, the Newton step quality degrades

**Solution:** Work on natural scale with box constraints:
1. Store coefficients as `β` (natural scale), not `log(β)`
2. Penalty becomes `P(β) = λ/2 · βᵀ S β` (quadratic)
3. Box constraints `β ≥ 0` enforce non-negativity
4. Newton approximation is exact for quadratic penalty

### Current PIJCV Results (for reference)

| Method | Selected λ | EDF | Weibull RMSE |
|--------|-----------|-----|--------------|
| PIJCV | 2980.958 | 1.03 | 26.7% |
| PERF | 0.2435 | 4.37 | 9.7% |
| EFS | 0.0249 | 4.89 | 12.3% |

After implementing natural-scale optimization, PIJCV should select λ ≈ 0.1-1.0 with EDF ≈ 4-5.

### Implementation Order Recommendation

1. **Phase 2 first** - Natural-scale splines (core change)
2. **Phase 3 second** - Simplify penalty system + monotone transform
3. **Phase 4 third** - Verify PIJCV works with LOO projection
4. **Phase 5 fourth** - Remove parameter storage redundancy
5. **Phase 6 last** - Comprehensive testing including edge cases

### Key Implementation Details

**Ipopt Configuration:**
```julia
IpoptOptimizer(
    honor_original_bounds = "yes",  # Ensures exact constraint satisfaction
    # ... other settings
)
```

**PIJCV LOO Projection:**
```julia
# In Newton LOO approximation
beta_loo = beta_hat + H_lambda_inv * g_j
beta_loo_proj = max.(beta_loo, 0.0)  # Project to feasible region
```

**Monotone Penalty Transform:**
```julia
# In build_penalty_config or build_spline_hazard_info
if hazard.monotone != 0
    L = build_monotone_cumsum_matrix(basis, hazard.monotone)
    S_transformed = L' * S * L
end
```

### Potential Gotchas

1. **Double transformation risk:** If spline code applies `exp()` and penalty code also applies `exp()`, results will be wrong. Carefully audit all code paths.

2. **Initialization values:** Current code initializes log-scale parameters. Natural-scale initialization needs positive values.

3. **Hazard evaluation:** The `hazard_fn` closure captures transformation logic. May need to regenerate these functions.

4. **Backward compatibility:** Existing fitted models store log-scale parameters. Consider version flag or migration path.

---

## 7. Prompt for Next Agent

### Context

Implementing box-constrained optimization for MultistateModels.jl to fix PIJCV λ selection. Phase 1 (bounds infrastructure) is complete and tested. All 1483 tests pass.

### Primary Task

Continue with **Phase 2**: Make spline hazards work on natural scale internally.

### Key Changes Needed

1. **`src/construction/multistatemodel.jl`**: Update `_spline_ests2coefs` to use identity for `monotone=0`
2. **`src/utilities/transforms.jl`**: Update transforms for `:sp` family to use identity
3. **`src/types/infrastructure.jl`**: Set `exp_transform=false` for spline penalty terms
4. **`src/utilities/penalty_config.jl`**: Add monotone penalty matrix transformation
5. **`src/inference/smoothing_selection.jl`**: Add LOO projection in PIJCV

### Mathematical Goal

**Non-monotone splines (`monotone=0`):**
```
Before: h(t) = exp(B(t) · θ), P(θ) = λ/2 · exp(θ)ᵀ S exp(θ)  # NON-QUADRATIC
After:  h(t) = B(t) · β,      P(β) = λ/2 · βᵀ S β            # QUADRATIC
```

**Monotone splines (`monotone≠0`):**
```
Before: β = L·exp(θ), P(θ) = λ/2 · exp(θ)ᵀ S exp(θ)  # Wrong penalty target
After:  β = L·γ,      P(γ) = λ/2 · γᵀ (LᵀSL) γ = λ/2 · βᵀSβ  # Correct
```

### Key Files

| File | Purpose |
|------|---------|
| `scratch/CONSTRAINED_OPTIMIZATION_IMPLEMENTATION_PLAN.md` | Full plan with critical review |
| `src/utilities/bounds.jl` | Bounds infrastructure (Phase 1 complete) |
| `src/construction/multistatemodel.jl` | Spline coefficient transforms (Phase 2 target) |
| `src/utilities/penalty_config.jl` | Penalty configuration (Phase 3 target) |
| `src/inference/smoothing_selection.jl` | PIJCV implementation (Phase 4 target) |

### User Preferences

1. **No finite differences** - Always use AD
2. **Validate everything** - Run tests after each change
3. **Mathematical correctness first** - Ensure penalty is truly quadratic
4. **Simplify codebase** - Remove parameter storage redundancy
5. **Non-negativity** - Use ε = 0 for lower bounds, not small positive values
6. **Don't over-engineer** - Keep solutions lean and maintainable

---

## 8. References

1. **Implementation Plan:** `scratch/CONSTRAINED_OPTIMIZATION_IMPLEMENTATION_PLAN.md`
2. **PIJCV Analysis:** `MultistateModelsTests/PENALIZED_SPLINES_HANDOFF_20260106.md` (Section 2)
3. **Penalty Mathematics:** See Section 1.5 of handoff document for correct Hessian formulas
4. **Li & Cao (2022):** GPS penalty matrix (already implemented correctly)
5. **Wood (2024):** NCV/PIJCV algorithm (works with quadratic penalty)
