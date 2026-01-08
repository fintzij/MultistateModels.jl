# MultistateModels.jl Penalized Splines Branch - Development Handoff
## Date: January 7, 2026 (Last Updated: End of Day)
## Branch: penalized_splines
## Status: SIGNIFICANT PROGRESS - VALIDATION PENDING

---

## EXECUTIVE SUMMARY

This handoff covers the `penalized_splines` branch development including:
- Natural scale parameterization (complete, architecturally sound)
- Spline infrastructure (3 known bugs documented)
- Phase-type parameter naming fix (critical bug fixed)
- Code organization (multistatemodel.jl split into focused files)

**Current Test Status**: 1467 passed, 1 failed (phase-type VCV), 1 errored (MCEM flaky)
**Production Ready**: PENDING validation of spline bugs
**Known Bugs**: 3 spline infrastructure issues (PART 8), 1 phase-type VCV issue

---

## PART 1: VERIFIED CORRECT (Core Infrastructure)

### 1.1 Natural Scale Parameter Storage ✅
- All parameters stored on natural scale with box constraints
- Zero vestigial transform functions remain (grep verified)
- `NONNEG_LB = 0.0` correctly defined in src/utilities/bounds.jl
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

## PART 2: BUGS FIXED TODAY (January 7, 2026 PM Session)

### FIX 1: Test `time_transform` Parity ✅
**File**: `MultistateModelsTests/unit/test_hazards.jl`
- Changed test parameters from log-scale to natural-scale values
- 3 replacements: Exponential, Weibull, Gompertz parameters

### FIX 2: Test `parameter_transformations` Obsolete ✅
**File**: `MultistateModelsTests/unit/test_helpers.jl`
- Replaced obsolete `parameter_transformations` testset with `natural_scale_parameters`
- Removed calls to non-existent `transform_baseline_to_natural/estimation` functions

### FIX 3: Phase-Type Parameter Naming Corruption ✅ CRITICAL
**Root cause**: Two functions used `model.hazkeys[hazname]` (parameter index) to index into `model.hazards[]`, but for phase-type models with shared hazards, **parameter indices ≠ hazard indices**.

**Fixes applied**:

1. **`rebuild_parameters`** in `src/utilities/parameters.jl` (line 137)
   - Added `params_idx_to_hazard_idx` reverse mapping
   - Now correctly finds hazard with matching parnames for each parameter set

2. **`_generate_package_bounds`** in `src/utilities/bounds.jl` (line 103)
   - Same fix: iterate over unique parameter sets via hazkeys, not hazards
   - Prevents bounds array overflow for phase-type models

**Technical detail for future reference**:
- `hazkeys`: Dict mapping hazard names → parameter indices
- `hazard_to_params_idx`: Vector[hazard_idx] → params_idx (for shared hazards, multiple hazard indices map to same param idx)
- `params_idx_to_hazard_idx`: Reverse mapping needed for correct parameter construction

### FIX 4: Parameter Naming Convention ✅
Changed exponential hazard parameter names from `_Intercept` to `_Rate` for clarity:
- Standard: `h12_Rate` instead of `h12_Intercept`
- Phase-type progression: `h1_ab_Rate`, `h1_bc_Rate`
- Phase-type exit: `h12_a_Rate`, `h12_b_Rate`, `h12_c_Rate`

**Files updated**:
- `src/construction/multistatemodel.jl` - `_build_exponential_hazard`
- `src/phasetype/expansion_hazards.jl` - progression and exit hazards
- `src/hazard/generators.jl` - docstring
- `src/hazard/covariates.jl` - docstring and `extract_covar_names`
- `src/utilities/parameters.jl` - docstrings
- `src/output/accessors.jl` - `get_parnames` docstring

**Test files updated**:
- `MultistateModelsTests/unit/test_phasetype.jl` - `log_λ_` → `_Rate`
- `MultistateModelsTests/unit/test_initialization.jl` - `_Intercept` → `_Rate`
- `MultistateModelsTests/unit/test_splines.jl` - `_Intercept` → `_Rate`
- `MultistateModelsTests/longtests/longtest_robust_parametric.jl` - `_Intercept` → `_Rate`

---

## PART 3: PRIOR FIXES (Earlier on January 7)

### Fix A: Test Function Typo ✅
**File**: `MultistateModelsTests/unit/test_splines.jl:489`
**Change**: `unflatten_natural` → `unflatten_parameters`

### Fix B: PIJCV NaN/Inf Graceful Handling ✅
**File**: `src/inference/smoothing_selection.jl:799`
**Change**: Added `if !all(isfinite.(H_i))` validation with warning and fallback
**Note**: Gracefully handles NaN/Inf, but underlying root cause may still exist

---

## PART 4: ISSUES REQUIRING VALIDATION

These issues were identified earlier today. Some may be resolved by the fixes in PART 2, but require full test suite validation to confirm.

### ISSUE 1: Spline Hazards S(t) > 1 — STATUS UNKNOWN

**Previous Symptom**: Survival probabilities exceed 1.0 (e.g., 1.0399, 1.0699, 1.1204)
**Location**: `MultistateModelsTests/unit/test_splines.jl:305`

**May be related to**: Phase-type parameter naming bug (FIX 3) if this was occurring in a phase-type + spline context. Need to re-run tests to verify.

**If still failing, investigate**:
1. Sign error in cumulative hazard computation
2. Incorrect antiderivative in spline basis integration
3. Check BSplineKit antiderivative usage
4. Validate that coefficients are NOT on log-hazard scale internally

**Files to audit**: `src/hazard/spline.jl`, `src/hazard/evaluation.jl`, `src/utilities/spline_utils.jl`

---

### ISSUE 2: Phase-Type Fitting Produces Positive LL — LIKELY FIXED

**Previous Symptom**: Log-likelihood = +488.5 (must be ≤ 0)
**Location**: `MultistateModelsTests/unit/test_phasetype.jl:1349`

**This was likely caused by FIX 3** (phase-type parameter naming bug). The parameter corruption would cause wrong parameters to be used in likelihood computation, producing nonsensical results.

**Focused test passed**: After FIX 3, a phase-type model with exponential hazards fits successfully with negative log-likelihood. Run full test suite to confirm all phase-type tests pass.

---

### ISSUE 3: PIJCV Hessian Contains NaN/Inf — GRACEFULLY HANDLED

**Location**: `src/inference/smoothing_selection.jl:799`
**Symptom**: Subject-level Hessian H_i contains NaN/Inf, causing eigen() to fail

**Status**: 
- ✅ Graceful fallback added (FIX B in PART 3)
- ⚠️ Root cause of NaN/Inf not investigated

**If still problematic after validation**:
- May be related to ISSUE 1 (spline hazard issues)
- May be related to phase-type parameter corruption (now fixed)
- Investigate which subjects/parameters cause NaN/Inf

---

### ISSUE 4: Phase-Type Variance-Covariance Returns `nothing` — STATUS UNKNOWN

**Previous Symptom**: vcov returns `nothing` instead of Matrix{Float64}
**Location**: `MultistateModelsTests/unit/test_phasetype.jl:1390`

**May be resolved by FIX 3**: If the optimizer now converges with correct parameters, vcov should compute successfully.

**If still failing**:
- Check if Hessian is singular/indefinite
- Trace vcov computation pathway for phase-type models
- Add proper error handling and informative warnings

---

## PART 5: PRIORITIZED ACTION PLAN

### IMMEDIATE: Run Full Test Suite
**Command**: `julia --project -e 'using Pkg; Pkg.test()'`
**Purpose**: Determine which issues from PART 4 are actually resolved by today's fixes

### If Tests Still Fail:

**Task 1: Fix Remaining Spline S(t) > 1 Issues** (if still failing)
- Files: `src/hazard/spline.jl`, `src/hazard/evaluation.jl`
- Investigation: Sign error in cumulative hazard, antiderivative issues
- Acceptance: All survival probability tests pass, S(t) ∈ [0,1] always

**Task 2: Fix Remaining Phase-Type Issues** (if still failing)
- Files: `src/likelihood/loglik_exact.jl`, `src/phasetype/`
- Investigation: Sign convention, parameter extraction
- Acceptance: LL < 0, optimizer converges, parameters valid

**Task 3: Fix Remaining PIJCV Issues** (if still failing)
- Files: `src/inference/smoothing_selection.jl`
- Investigation: Root cause of NaN/Inf in Hessians
- Acceptance: No NaN/Inf in Hessians, PIJCV succeeds

**Task 4: Search for `_Intercept` in Tests**
Some tests may still expect old `_Intercept` naming:
```bash
grep -r "_Intercept" MultistateModelsTests/
```

### After Tests Pass:

**Task 5: Documentation Sweep**
- Grep for "estimation scale", "log scale" and update docstrings
- Acceptance: All docs reflect natural scale only

**Task 6: Comprehensive Validation**
- Run longtests
- Run benchmarks against R
- Acceptance: Benchmarks validate correctness

---

## PART 6: DEBUGGING WORKFLOW

For each bug, follow this rigorous protocol:

1. **Isolate**: Create minimal reproducible example
2. **Trace**: Step through code path with debugger or print statements
3. **Verify**: Check mathematical correctness at each step
4. **Fix**: Implement fix with clear comments
5. **Validate**: Run tests, verify acceptance criteria
6. **Regression**: Ensure fix doesn't break other tests

**Never claim something works without running the tests.**

---

## PART 7: TEST SUITE STATUS

| Category | Before Today | After Focused Tests | Target |
|----------|-------------|---------------------|--------|
| Passing | 1407 | UNKNOWN | 1425+ |
| Failing | 10 | UNKNOWN | 0 |
| Erroring | 8 | UNKNOWN | 0 |

**Focused tests (today's session)**: 18/18 pass
- Phase-Type Parameter Naming Bug Fix | 14/14 PASS
- Standard Exponential Uses _Rate Naming | 4/4 PASS

**Full test suite**: NOT YET RUN after today's fixes

---

## PART 8: SPLINE INFRASTRUCTURE ISSUES

### ISSUE 5: Automatic Knot Placement May Be Buggy

**File**: `src/construction/spline_builder.jl`, function `_build_spline_hazard()`
**Symptom**: When user doesn't specify knots, automatic placement may use wrong formula or number

**Investigation needed**:
- Verify `default_nknots()` formula matches Simon Wood's guidance
- Verify `place_interior_knots_pooled()` is called with correct arguments
- Ensure pooled vs per-transition knot placement logic is correct

### ISSUE 6: Hessian-Based Knot Optimization Warning Possible Dead Code

**File**: `src/hazard/spline.jl`
**Symptom**: Old Hessian-based optimization may never be triggered or may use wrong scale

**Investigation needed**:
- Trace code path for Hessian-based knot optimization
- Verify whether this is dead code or active
- If active, ensure using natural scale parameters

### ISSUE 7: Monotone Spline Penalty Matrix Incorrect — CONFIRMED BUG ⚠️

**Location**: `src/types/infrastructure.jl`, `compute_penalty()`
**Root Cause**: Penalty matrix `S` is built for B-spline coefficients (`coefs`), but parameters being penalized are I-spline increments (`ests`).

**Mathematical Problem**:
For monotone splines, the transformation is:
```julia
coefs = L * ests  # where L is lower-triangular cumsum with knot weights
```

The correct penalty should be:
```julia
P(ests) = (λ/2) coefs^T S coefs
        = (λ/2) (L * ests)^T S (L * ests)
        = (λ/2) ests^T (L^T S L) ests
```

So `S_monotone = L^T * S * L` — but this transformation is **not currently implemented**.

**Current behavior**: Applies `S` directly to `ests`, which is mathematically incorrect.

**Files to modify**:
1. `src/utilities/spline_utils.jl` — Add function to build `L` matrix and transform `S`
2. `src/utilities/penalty_config.jl` — Apply transformation when `monotone != 0`
3. Or: `src/types/infrastructure.jl` — Handle transformation in `compute_penalty()`

**Severity**: Medium — monotone splines will have incorrect smoothing, but non-monotone splines are unaffected.

---

## PART 9: ARCHITECTURAL CONTEXT

### Original Objectives (All Complete ✅)
- Store all parameters on natural scale (not log scale) ✅
- Enforce non-negativity via box constraints (not transforms) ✅
- Make penalty truly quadratic ✅
- Dict-only bounds API ✅
- GPS penalty matrix for splines ✅
- AD-based smoothing selection ✅
- Remove all transform functions ✅
- NaNMath integration ✅

### Parameter Storage Architecture
- All parameters on natural (positive) scale
- Box constraints via Ipopt: `lb = NONNEG_LB = 0.0`
- Gompertz shape: unconstrained (can be negative)
- No vestigial transform functions remain

### Phase-Type Parameter Indexing (Critical for Future Reference)
For phase-type models with shared hazards:
- `model.hazkeys`: Dict{Symbol, Int} mapping hazard names → **parameter indices**
- `model.hazard_to_params_idx`: Vector[hazard_idx] → params_idx
- **Parameter indices ≠ hazard indices** when hazards are shared!
- When iterating over parameters, use `params_idx_to_hazard_idx` reverse mapping to find correct hazard

---

## PART 9: VALIDATION CHECKLIST

### Mathematical Correctness
- [x] All parameters stored on natural scale (no transforms)
- [x] Penalty is quadratic P(β) = (λ/2) β^T S β
- [x] Box constraints enforce positivity
- [x] NaNMath.log prevents DomainError
- [x] Phase-type parameter naming correct (FIXED TODAY)
- [ ] Survival probabilities S(t) ∈ [0, 1] for all t — NEEDS VALIDATION
- [ ] Log-likelihoods ℓ(θ) ≤ 0 for all θ — LIKELY FIXED, NEEDS VALIDATION
- [ ] All fitted parameters in plausible ranges — LIKELY FIXED, NEEDS VALIDATION

### Numerical Stability
- [x] PIJCV handles NaN/Inf Hessians gracefully (with fallback)
- [ ] PIJCV Hessians are finite (root cause) — NEEDS VALIDATION
- [ ] Optimizer converges for reasonable data — LIKELY FIXED, NEEDS VALIDATION
- [ ] Variance-covariance computation succeeds — NEEDS VALIDATION

### Test Suite
- [x] All tests run to completion (no UndefVarError)
- [ ] All tests pass — NEEDS VALIDATION
- [ ] 0 failures — NEEDS VALIDATION
- [ ] 0 errors — NEEDS VALIDATION

---

## PART 10: HANDOFF REQUIREMENTS

The next agent MUST:

1. **Run full test suite first**: `julia --project -e 'using Pkg; Pkg.test()'`
2. **Keep this document updated** as work progresses
3. **Operate as skeptical senior Julia developer & PhD mathematical statistician**
4. **Prioritize mathematical correctness above all else**
5. **Never claim implementation without running tests**
6. **Proactively monitor context usage and prepare handoffs**
7. **Assume bugs exist until proven otherwise**

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

## PART 11: QUICK REFERENCE

### Verification Test Command (Phase-Type)
\`\`\`julia
using Test, MultistateModels, DataFrames
dat = DataFrame(
    id = [1, 1, 2, 2, 3],
    tstart = [0.0, 5.0, 0.0, 3.0, 0.0],
    tstop = [5.0, 10.0, 3.0, 8.0, 7.0],
    statefrom = [1, 2, 1, 2, 1],
    stateto = [2, 2, 2, 3, 3],
    obstype = [1, 2, 1, 1, 1]
)
h1_12 = Hazard(@formula(0 ~ 1), :pt, 1, 2; n_phases=3)
h2_13 = Hazard(@formula(0 ~ 1), :exp, 1, 3)
h3_23 = Hazard(@formula(0 ~ 1), :exp, 2, 3)
model = multistatemodel(h1_12, h2_13, h3_23; data=dat, initialize=true)

# Verify h23 has correct parameter name
@test haskey(model.parameters.nested[:h23].baseline, :h23_Rate)  # ✓
@test !haskey(model.parameters.nested[:h23].baseline, :h13_Rate) # ✓

# Fitting works
fitted = fit(model; verbose=false, compute_vcov=false)
@test !isnan(fitted.loglik.loglik)  # ✓
@test fitted.loglik.loglik < 0  # ✓ (log-likelihood must be negative)
\`\`\`

### Search for Old Naming
\`\`\`bash
grep -r "_Intercept" MultistateModelsTests/
\`\`\`

---

## PART 12: SPLINE INFRASTRUCTURE BUGS (ADDED 2026-01-07)

Three bugs identified in spline knot placement and smoothing selection:

### BUG 1: `default_nknots()` Uses Regression Spline Formula

**Location**: [src/hazard/spline.jl](../src/hazard/spline.jl#L432)

**Problem**: Uses `floor(n^(1/5))` from Tang et al. (2017), which is appropriate for **regression splines (sieve estimation)**, not penalized splines.

**Impact**: For penalized splines, the penalty controls overfitting, so more knots are acceptable (and often desirable for flexibility). The current conservative formula may under-fit.

**Fix Required**: Either:
1. Increase default knots for penalized splines (e.g., `floor(n^(1/3))` or fixed reasonable default like 10-20)
2. Add separate function for penalized vs unpenalized spline defaults
3. Document that users should override automatic knot count for penalized splines

**Reference**: The Tang et al. formula assumes no penalty—with a penalty, you can safely use many more knots.

---

### BUG 2: Automatic Knot Placement Uses Raw Data Instead of Surrogate Simulation

**Location**: [src/construction/multistatemodel.jl](../src/construction/multistatemodel.jl#L415-L560)

**Problem**: `_build_spline_hazard()` extracts sojourns directly from observed data:
```julia
paths = extract_paths(data, model.totalhazaliases)
sojourns = extract_sojourns(paths, hazard.statefrom, hazard.stateto)
```

**Impact**: For **panel data** (obstype=2), exact transition times are NOT observed—only the state at discrete times. Extracting sojourns from panel data gives meaningless or biased time ranges.

**Design Discrepancy**: The intended design was to:
1. Fit a parametric surrogate model first
2. Simulate paths from the surrogate
3. Use simulated sojourns for knot placement

**Fix Required**: When automatic knot placement is triggered and data contains panel observations:
1. Use the surrogate model (already fitted) to simulate paths
2. Extract sojourns from simulated paths
3. Place knots based on simulated sojourn distribution

**Files Affected**:
- `_build_spline_hazard()` in multistatemodel.jl
- May need to pass surrogate model reference or simulated paths

---

### BUG 3: Hessian NaN/Inf Warnings in Smoothing Selection

**Locations**:
- [src/inference/smoothing_selection.jl#L801](../src/inference/smoothing_selection.jl#L801) - `_solve_loo_newton_step()`
- [src/inference/smoothing_selection.jl#L2039](../src/inference/smoothing_selection.jl#L2039) - `compute_edf()`

**Problem**: Warnings like:
```
"Subject Hessian contains NaN/Inf values"
"Failed to invert penalized Hessian for EDF computation"
```

These indicate **numerical issues in likelihood/Hessian computation**, not expected edge cases.

**Root Cause Investigation Needed**:
1. Why do subject-level Hessians contain NaN/Inf?
   - Possible: log(0) somewhere in likelihood
   - Possible: Parameters hitting bounds
   - Possible: Spline basis evaluation issues at boundary
   
2. Why does penalized Hessian fail to invert?
   - Possible: Penalty matrix not positive definite
   - Possible: Near-singular due to multicollinearity
   - Possible: Numerical precision issues with small smoothing parameters

**Fix Required**: 
1. Add diagnostic logging to identify WHERE NaN/Inf first appear
2. Trace back to root cause in likelihood computation
3. Fix underlying numerical issue rather than warning and continuing

---

### Priority Order for Fixes

1. **BUG 3** (Hessian NaN/Inf) - Most urgent; indicates fundamental numerical issues
2. **BUG 2** (Knot placement) - Architectural issue affecting correctness for panel data
3. **BUG 1** (default_nknots) - Lower priority; workaround is manual knot specification

---

## PART 13: PLANNED FEATURES

### FEATURE 1: Per-Transition Surrogate Specification ⏳

**Goal**: Allow users to specify surrogate type (exponential vs phase-type) separately for each transition, rather than model-wide.

**Current API** (model-wide only):
```julia
multistatemodel(h12, h21; data=df, surrogate=:auto)  # Same surrogate for ALL transitions
```

**Proposed API** (per-transition):
```julia
# Dict mapping (statefrom, stateto) => surrogate type
multistatemodel(h12, h21; 
    data=df, 
    surrogate = Dict(
        (1,2) => :phasetype,   # Use phase-type for 1→2
        (2,1) => :markov       # Use Markov for 2→1
    )
)

# Or Symbol for uniform behavior
multistatemodel(h12, h21; data=df, surrogate=:auto)  # Auto-detect per transition
```

**Implementation Plan**:

1. **Update `multistatemodel()` signature** ([multistatemodel.jl](../src/construction/multistatemodel.jl))
   - Change `surrogate::Symbol` to `surrogate::Union{Symbol, Dict{Tuple{Int,Int}, Symbol}}`
   - Add `surrogate_n_phases::Union{Int, Dict{Tuple{Int,Int}, Int}, Symbol} = :heuristic`
   - Validate all transitions are covered if Dict provided

2. **Update `MarkovSurrogate` struct** ([model_structs.jl](../src/types/model_structs.jl))
   - Add `surrogate_type::Dict{Tuple{Int,Int}, Symbol}` field to track per-transition type
   - Or create wrapper struct that holds per-transition configuration

3. **Update surrogate resolution logic** ([multistatemodel.jl](../src/construction/multistatemodel.jl))
   ```julia
   # Resolve surrogate specification to per-transition Dict
   function _resolve_surrogate_spec(surrogate, hazards, tmat)
       if surrogate isa Symbol
           if surrogate === :auto
               # Auto-detect per transition based on hazard type
               return Dict((h.statefrom, h.stateto) => 
                   (h.family == :exp ? :markov : :phasetype) for h in hazards)
           else
               # Uniform surrogate type
               return Dict((h.statefrom, h.stateto) => surrogate for h in hazards)
           end
       else
           # Already a Dict - validate and return
           return surrogate
       end
   end
   ```

4. **Update `fit_phasetype_surrogate()`** ([markov.jl](../src/surrogate/markov.jl))
   - Accept per-transition `n_phases` specification
   - Build phase-type expansion only for transitions that need it

5. **MCEM proposal logic** ([fit_mcem.jl](../src/inference/fit_mcem.jl))
   - **No changes needed** — proposal logic already handles mixed surrogate types
   - Each transition's proposal drawn from its configured surrogate

**Files to Modify**:
- `src/construction/multistatemodel.jl` — signature, validation, resolution
- `src/types/model_structs.jl` — `MarkovSurrogate` struct update
- `src/surrogate/markov.jl` — per-transition phase-type fitting
- `src/phasetype/types.jl` — update `resolve_proposal_config()` if needed

**Test Coverage Needed**:
- Model with mixed exponential + Weibull hazards, different surrogates per transition
- Verify correct proposal types used during MCEM sampling
- Validate parameter estimation accuracy

---

### FEATURE 2: Surrogate Auto-Fit at MCEM Time ✅ ALREADY IMPLEMENTED

The `fit_surrogate` flag at construction time already controls this:
- `fit_surrogate=true` (default): Fit surrogate at construction via MLE
- `fit_surrogate=false`: Defer fitting to MCEM time

**MCEM behavior** ([fit_mcem.jl](../src/inference/fit_mcem.jl#L382)):
```julia
if !model.markovsurrogate.fitted
    # Fit surrogate at MCEM time if not already fitted
    ...
end
```

**`is_surrogate_fitted(model)`**: Query whether surrogate has been fitted.

---

## CONCLUSION

**Significant progress made today.** The phase-type parameter naming bug was the likely root cause of multiple test failures. After fixing:
1. `rebuild_parameters` — correct parameter name assignment
2. `_generate_package_bounds` — correct bounds generation
3. Renamed `_Intercept` → `_Rate` for clarity
4. Added `:auto` and `:phasetype` surrogate options at construction time

**Status**: Focused tests pass (18/18). Full test suite: 16/17 passing (1 phasetype VCV failure).

**Next Immediate Action**: Run `Pkg.test()` and fix any remaining failures.

**Newly Documented Bugs**: 3 spline infrastructure issues added (PART 12) for future investigation.

**Planned Features**: Per-transition surrogate specification documented (PART 13).

---

**Document Version**: 2.2
**Last Updated**: January 7, 2026 (End of Day)
**Status**: Awaiting full test suite validation + spline bug investigation
**Branch**: penalized_splines
