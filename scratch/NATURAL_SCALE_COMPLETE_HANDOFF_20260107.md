# MultistateModels.jl Natural Scale Transition - Complete Handoff & Action Plan
## Date: January 7, 2026
## Status: CRITICAL BUGS IDENTIFIED - DO NOT MERGE

---

## EXECUTIVE SUMMARY

The natural scale parameterization transition is **architecturally sound but implementation is incomplete with critical bugs**. Core infrastructure (bounds, penalty, NaNMath) is verified correct. However, show-stopper bugs exist in spline hazards, phase-type fitting, and PIJCV that **block production use**.

**Current Test Status**: 1407 pass, 10 fail, 8 error (after 2 immediate fixes)
**Production Ready**: ❌ NO
**Estimated Remaining Work**: 12-24 hours focused debugging

---

## PART 1: VERIFIED CORRECT (Core Infrastructure)

### 1.1 Natural Scale Parameter Storage ✅
- All parameters stored on natural scale with box constraints
- Zero vestigial transform functions remain (grep verified)
- `NONNEG_LB = 0.0` correctly defined in [src/utilities/bounds.jl](src/utilities/bounds.jl#L19)
- Gompertz shape parameter correctly allows negative values ([-Inf, NONNEG_LB])
- No `exp_transform`, `to_natural_scale`, or `to_estimation_scale` functions exist

### 1.2 Penalty Quadratic Form ✅
- Penalty correctly implemented as P(β) = (λ/2) β^T S β
- No exp() transformations in penalty computation (verified in smoothing_selection.jl)
- Parameters used directly: `β_j = @view beta[term.hazard_indices]`
- Quadratic form returned: `penalty / 2`

### 1.3 NaNMath Integration ✅
- NaNMath.log consistently used in all 13 likelihood functions
- Prevents DomainError during optimization when hazards approach zero
- Correctly implemented in exact, Markov, and semi-Markov likelihoods

### 1.4 Box Constraints via Ipopt ✅
- `get_baseline_lower_bounds()` generates correct bounds by hazard family
- Bounds passed to Ipopt optimizer as vectors
- No transformation functions used anywhere

---

## PART 2: CRITICAL BUGS (BLOCKING PRODUCTION)

### BUG 1: Spline Hazards Produce S(t) > 1 ⚠️ SHOW-STOPPER

**Severity**: CRITICAL - Mathematical impossibility
**Location**: [MultistateModelsTests/unit/test_splines.jl:305](MultistateModelsTests/unit/test_splines.jl#L305)
**Symptom**: Survival probabilities exceed 1.0 (e.g., 1.0399, 1.0699, 1.1204)

**Why This Is Critical**:
- S(t) ∈ [0, 1] by mathematical definition
- Values > 1 indicate fundamental error in hazard/cumulative hazard computation
- Could be sign error, incorrect antiderivative, or hidden exp() transform

**Root Cause Hypotheses**:
1. Sign error in cumulative hazard computation
2. Incorrect antiderivative in spline basis integration
3. Coefficients being exp() transformed somewhere in evaluation path
4. Integration bounds or limits incorrect

**Investigation Plan**:
1. Create minimal test: constant hazard h(t) = 0.5, verify S(t) = exp(-0.5t)
2. Trace spline hazard evaluation path for any exp() calls
3. Verify cumulative hazard integration is computing ∫₀ᵗ h(u)du with correct sign
4. Check BSplineKit antiderivative usage
5. Validate that coefficients are NOT on log-hazard scale internally

**Acceptance Criteria**:
- S(0) = 1 (everyone alive at start)
- S(t) ∈ [0, 1] for all t ≥ 0
- S(t₁) ≥ S(t₂) for t₁ < t₂ (monotonically decreasing)
- lim_{t→∞} S(t) = 0

**Files to Audit**:
- [src/hazard/spline.jl](src/hazard/spline.jl): Hazard evaluation
- [src/hazard/evaluation.jl](src/hazard/evaluation.jl): Cumulative hazard computation
- [src/utilities/spline_utils.jl](src/utilities/spline_utils.jl): Integration utilities

---

### BUG 2: Phase-Type Fitting Produces Positive Log-Likelihood ⚠️ SHOW-STOPPER

**Severity**: CRITICAL - Mathematical impossibility
**Location**: [MultistateModelsTests/unit/test_phasetype.jl:1349](MultistateModelsTests/unit/test_phasetype.jl#L1349)
**Symptom**: Log-likelihood = +488.5 (must be ≤ 0)

**Why This Is Critical**:
- Log of probabilities (which are ≤ 1) must be ≤ 0
- Positive LL indicates sign flip or inverse probability computation
- Optimizer also fails to converge and produces invalid parameters

**Additional Failures in Same Test**:
- `get_convergence(fitted) == false` (optimizer didn't converge)
- Parameters include negative rates (should be > 0)
- Parameters include absurdly large rates (> 100)

**Root Cause Hypotheses**:
1. Sign convention error: returning -ℓ when should return ℓ
2. Double negation somewhere in likelihood chain
3. Box constraints not properly applied to phase-type rates
4. Expanded parameter structure issues

**Investigation Plan**:
1. Trace phase-type likelihood computation in loglik_exact.jl
2. Verify sign convention: `neg=false` returns ℓ, `neg=true` returns -ℓ
3. Check parameter extraction from expanded phase-type structure
4. Verify progression rates and exit rates both have lb = NONNEG_LB
5. Test with simple 2-phase model and known parameters

**Acceptance Criteria**:
- Log-likelihood < 0 for all fitted models
- Optimizer converges (get_convergence = true)
- All rate parameters > 0
- Rate parameters in plausible range (< 100 for typical data)
- Variance-covariance computation succeeds

**Files to Audit**:
- [src/likelihood/loglik_exact.jl](src/likelihood/loglik_exact.jl): Phase-type likelihood
- [src/phasetype/](src/phasetype/): Phase-type model construction
- [src/utilities/bounds.jl](src/utilities/bounds.jl): Phase-type bounds generation

---

### BUG 3: PIJCV Hessian Contains NaN/Inf ⚠️ HIGH PRIORITY

**Severity**: HIGH - Blocks smoothing parameter selection
**Location**: [src/inference/smoothing_selection.jl:799](src/inference/smoothing_selection.jl#L799)
**Symptom**: Subject-level Hessian H_i contains NaN/Inf, causing eigen() to fail

**Partial Fix Applied**:
✅ Added validation to gracefully fall back to direct solve instead of crashing
⚠️ But underlying cause of NaN/Inf is NOT fixed

**Root Cause Hypotheses**:
1. Hazard evaluation producing Inf or NaN values
2. Numerical overflow in second derivatives
3. Spline coefficients hitting constraint boundaries creating singularities
4. Related to BUG 1 (spline hazard issues)

**Investigation Plan**:
1. Log which subjects produce NaN/Inf Hessians
2. Inspect parameter values when this occurs
3. Check if related to boundary of constraint region
4. Verify spline hazard second derivatives are finite
5. May be resolved automatically when BUG 1 is fixed

**Acceptance Criteria**:
- All subject Hessians are finite
- PIJCV criterion computes successfully without fallback
- Smoothing parameter selection converges

---

### BUG 4: Phase-Type Variance-Covariance Returns `nothing` ⚠️ MEDIUM PRIORITY

**Severity**: MEDIUM - Blocks inference for phase-type models
**Location**: [MultistateModelsTests/unit/test_phasetype.jl:1390](MultistateModelsTests/unit/test_phasetype.jl#L1390)
**Symptom**: vcov returns `nothing` instead of Matrix{Float64}

**Root Cause Hypotheses**:
1. Hessian is singular/indefinite (cannot invert)
2. Related to BUG 2 (optimizer doesn't converge)
3. Transformation Jacobian missing for user-facing parameters
4. Likely will be resolved when BUG 2 is fixed

**Investigation Plan**:
1. Trace vcov computation pathway for phase-type models
2. Check if Hessian inversion succeeds
3. Verify proper error handling and informative warnings
4. May require separate Jacobian for expanded→user parameter mapping

**Acceptance Criteria**:
- vcov returns Matrix{Float64} for successful fits
- Informative error/warning when vcov cannot be computed
- Confidence intervals can be constructed

---

## PART 3: IMMEDIATE FIXES ALREADY APPLIED

### Fix 1: Test Function Name Typo ✅
**File**: [MultistateModelsTests/unit/test_splines.jl:489](MultistateModelsTests/unit/test_splines.jl#L489)
**Change**: `unflatten_natural` → `unflatten_parameters`
**Result**: Errors reduced 9 → 8

### Fix 2: PIJCV NaN/Inf Graceful Handling ✅
**File**: [src/inference/smoothing_selection.jl:799](src/inference/smoothing_selection.jl#L799)
**Change**: Added `if !all(isfinite.(H_i))` validation with warning and fallback
**Result**: No more crashes, but underlying issue remains

---

## PART 4: PRIORITIZED ACTION PLAN

### Phase 1: Critical Bug Fixes (URGENT - Required for Merge)

**Task 1.1: Fix Spline S(t) > 1 Bug**
- Priority: ⚠️ CRITICAL
- Estimated Time: 4-8 hours
- Files: [src/hazard/spline.jl], [src/hazard/evaluation.jl]
- Acceptance: All survival probability tests pass, S(t) ∈ [0,1] always

**Task 1.2: Fix Phase-Type Positive Log-Likelihood**
- Priority: ⚠️ CRITICAL
- Estimated Time: 4-8 hours
- Files: [src/likelihood/loglik_exact.jl], [src/phasetype/]
- Acceptance: LL < 0, optimizer converges, parameters valid

**Task 1.3: Investigate PIJCV NaN/Inf Root Cause**
- Priority: HIGH
- Estimated Time: 2-4 hours
- Files: [src/inference/smoothing_selection.jl], hazard evaluation path
- Acceptance: No NaN/Inf in Hessians, PIJCV succeeds
- Note: May be resolved when Task 1.1 is fixed

### Phase 2: Validation & Polish (Required for Release)

**Task 2.1: Fix Phase-Type Variance-Covariance**
- Priority: MEDIUM
- Estimated Time: 2-4 hours
- Files: Variance computation pathway
- Acceptance: vcov works or provides informative error

**Task 2.2: Documentation Sweep**
- Priority: MEDIUM
- Estimated Time: 2-4 hours
- Action: Grep for "estimation scale", "log scale" and update docstrings
- Acceptance: All docs reflect natural scale only

**Task 2.3: Comprehensive Testing**
- Priority: HIGH
- Estimated Time: 4-8 hours
- Action: Run full test suite, longtests, benchmarks against R
- Acceptance: 1425/1425 tests pass, benchmarks validate

---

## PART 5: DEBUGGING WORKFLOW (CRITICAL)

For each bug, follow this rigorous protocol:

1. **Isolate**: Create minimal reproducible example
2. **Trace**: Step through code path with debugger or print statements
3. **Verify**: Check mathematical correctness at each step
4. **Fix**: Implement fix with clear comments
5. **Validate**: Run tests, verify acceptance criteria
6. **Regression**: Ensure fix doesn't break other tests

**Never claim something works without running the tests.**

---

## PART 6: TEST SUITE STATUS

| Category | Before Fixes | After Fixes | Target |
|----------|-------------|-------------|--------|
| Passing | 1406 | 1407 | 1425 |
| Failing | 10 | 10 | 0 |
| Erroring | 9 | 8 | 0 |

**Remaining Work**: Fix 18 failing/erroring tests (mostly from 4 critical bugs above)

---

## PART 7: LEGACY CONTEXT (From Original Handoff)

### Original Objectives
- Store all parameters on natural scale (not log scale) ✅ DONE
- Enforce non-negativity via box constraints (not transforms) ✅ DONE
- Make penalty truly quadratic ✅ DONE
- Dict-only bounds API ✅ DONE
- GPS penalty matrix for splines ✅ DONE
- AD-based smoothing selection ✅ DONE
- Remove all transform functions ✅ DONE
- NaNMath integration ✅ DONE

### What Remains
- Fix bugs in implementation (CRITICAL)
- Validate tests pass 100%
- Documentation sweep
- Benchmarking validation

---

## PART 8: VALIDATION CHECKLIST

### Mathematical Correctness
- [x] All parameters stored on natural scale (no transforms)
- [x] Penalty is quadratic P(β) = (λ/2) β^T S β
- [x] Box constraints enforce positivity
- [x] NaNMath.log prevents DomainError
- [ ] Survival probabilities S(t) ∈ [0, 1] for all t ← **BUG 1**
- [ ] Log-likelihoods ℓ(θ) ≤ 0 for all θ ← **BUG 2**
- [ ] All fitted parameters in plausible ranges ← **BUG 2**

### Numerical Stability
- [x] PIJCV handles NaN/Inf Hessians gracefully (with fallback)
- [ ] PIJCV Hessians are finite (root cause fixed) ← **BUG 3**
- [ ] Optimizer converges for reasonable data ← **BUG 2**
- [ ] Variance-covariance computation succeeds ← **BUG 4**

### Test Suite
- [x] All tests run to completion (no UndefVarError)
- [ ] 1425/1425 tests pass
- [ ] 0 failures
- [ ] 0 errors

---

## PART 9: HANDOFF REQUIREMENTS FOR NEXT AGENT

The next agent MUST:

1. **Operate as skeptical senior Julia developer & PhD mathematical statistician**
2. **Prioritize mathematical correctness above all else**
3. **Never claim implementation without running tests**
4. **Proactively monitor context usage and prepare handoffs**
5. **Assume bugs exist until proven otherwise**
6. **Follow the debugging workflow rigorously**

### Context Management Rules
- Alert user if conversation exceeds 15 exchanges
- Prepare detailed handoff if 5+ files modified
- Stop and checkpoint if context confusion risk
- Never continue in degraded context state

### Validation Rules
- Run code after every change
- Run tests after every fix
- Report actual results, including failures
- Iterate until tests pass before claiming completion
- Be explicit about what was NOT validated

---

## PART 10: RISK ASSESSMENT

### Show-Stopper Bugs (BLOCKING MERGE)
1. ✅ Spline S(t) > 1 — **FUNDAMENTAL MATHEMATICAL ERROR**
2. ✅ Phase-type ℓ > 0 — **FUNDAMENTAL MATHEMATICAL ERROR**
3. ✅ Test suite had non-existent function reference — **FIXED**

### High-Risk Bugs (BLOCKING PRODUCTION)
4. ✅ PIJCV NaN/Inf crash — **GRACEFULLY HANDLED, ROOT CAUSE REMAINS**
5. ✅ Phase-type vcov failure — **BLOCKS INFERENCE**

**DO NOT MERGE** until items 1 and 2 are fixed and test suite passes 100%.

---

## CONCLUSION

The natural scale parameterization transition has **sound architecture** but **critical implementation bugs**. The core infrastructure is verified correct (bounds, penalty, NaNMath, constraints all work as designed). However, bugs in spline hazards and phase-type fitting produce mathematically impossible results that block production use.

**Estimated Time to Production Ready**: 12-24 hours of focused debugging

**Next Immediate Action**: Fix BUG 1 (Spline S(t) > 1) as highest priority

---

**Document Version**: 1.0
**Date**: January 7, 2026
**Status**: Ready for focused debugging effort
**Next Review**: After Phase 1 bugs are fixed

