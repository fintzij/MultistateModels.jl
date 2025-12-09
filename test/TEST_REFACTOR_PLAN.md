# Test Suite Refactoring Plan
**Date:** 2025-12-07  
**Status:** Implementation Ready

## Executive Summary
Test suite has grown to 49 files with 6113 lines. Need to:
1. Fix 27 failing tests (parameter type mismatches)
2. Remove weak/redundant tests  
3. Consolidate and strengthen remaining tests
4. Establish clear testing principles

## Current State Analysis

### Test Failures (27 total)
**Root Cause:** Parameter handling refactor changed from Tuple-of-Vectors to NamedTuple-of-NamedTuples
- Tests mixing vector arithmetic with NamedTuple parameter passing
- `get_log_scale_params_tuple()` helper returns NamedTuples but tests expect vectors
- Solution: Use `extract_params_vector()` for arithmetic, NamedTuples for hazard evaluation

**Affected Tests:**
- test_hazards_weibull (3 errors) - lines 185-186, 197-203
- test_cumulativehazards_weibull (3 errors) - lines 301-303, 309-322  
- test_hazards_gompertz (3 errors) - lines 452-454, 442-443
- test_cumulativehazards_gompertz (4 errors) - lines 506-523
- linpred_effect modes (9 errors) - lines 592-602
- time_transform mode (3 errors) - lines 720-727
- time_transform parity (1 error)
- batched_vs_sequential_parity (1 error)

### File Inventory

**High Quality - Keep As-Is:**
- `test_phasetype_correctness.jl` (459 lines) ✅
  - Excellent analytic validation
  - Compares against known formulas
  - Clear structure and documentation
- `test_reconstructor.jl` ✅
  - Core infrastructure testing
  - Property-based tests
- `test_parameter_ordering.jl` ✅
  - Critical correctness verification
- `test_ncv.jl` ✅
  - Statistical method validation

**Medium Quality - Needs Cleanup:**
- `test_hazards.jl` (1025 lines) ⚠️
  - Good analytic tests but too verbose
  - Many redundant "toy" model tests
  - **Action:** Streamline, keep analytic validation, remove redundancy
- `test_helpers.jl` (288 lines) ⚠️
  - Mixes parameter handling, batching, extraction tests
  - **Action:** Consolidate with parameter tests, remove weak tests
- `test_simulation.jl` ⚠️
  - Some weak "edge case" tests without clear purpose
  - **Action:** Keep core functionality, remove weak tests
- `test_modelgeneration.jl` ⚠️
  - Many overlapping tests
  - **Action:** Streamline to essentials

**Low Quality - Remove or Consolidate:**
- `setup_*.jl` files (8 files) ❌
  - Obscure what's being tested
  - **Action:** Replace with inline fixtures
- `test_phasetype_is.jl`, `test_phasetype_fitting.jl`, `test_phasetype_simulation.jl` ❌
  - Redundant with `test_phasetype_correctness.jl`
  - **Action:** Consolidate best tests into correctness file
- `test_make_subjdat.jl` ❌
  - Tests internal data transformation
  - **Action:** Remove or minimal coverage only
- `test_reversible_tvc_loglik.jl` ❌
  - Weak statistical test without analytic validation
  - **Action:** Remove or strengthen significantly

**Long Tests - Review Separately:**
- `longtest_*.jl` files (13 files)
  - Statistical validation tests
  - **Action:** Review in separate pass, not part of this refactor

## Implementation Plan

### Phase 1: Fix Immediate Failures (Priority 1)

**Step 1.1: Fix test_hazards.jl parameter handling**
Pattern to apply:
```julia
# BEFORE (broken)
pars = get_log_scale_params_tuple(model.parameters, model)[N]
shape = exp(pars[1])  # ERROR: can't index NamedTuple

# AFTER (fixed)
pars_nt = get_log_scale_params_tuple(model.parameters, model)[N]
pars = MultistateModels.extract_params_vector(pars_nt)
shape = exp(pars[1])  # OK: pars is now Vector
# Pass pars_nt (NamedTuple) to eval_hazard, pars (Vector) for arithmetic
```

**Files to fix:**
1. test_hazards.jl - lines 197, 309, 442, 612-613, 648-649, 762, 782, 808, 845-846
2. test_helpers.jl - batched_vs_sequential_parity test
3. test_simulation.jl - fixed-count test
4. test_phasetype_is.jl - 2 tests

**Step 1.2: Fix inline function calls**
Some tests call `exp()`, `expm1()` on NamedTuples. Need to extract scalars first.

### Phase 2: Streamline test_hazards.jl (Priority 2)

**Current:** 1025 lines with extensive redundancy
**Target:** ~400 lines focused on correctness

**Keep:**
- Analytic validation tests (compare against known formulas)
- One test per hazard type: exponential, Weibull, Gompertz
- PH vs AFT mode tests
- Time transform tests

**Remove:**
- Redundant "toy model" tests
- Tests that just verify "code runs" without checking correctness
- Overly detailed tests of implementation details

**Restructure:**
```julia
@testset "Exponential Hazards - Analytic Validation" begin
    # Test: h(t) = λ (constant)
    # Test: H(t) = λt (linear)
    # Test: With covariates h(t|x) = λ exp(β'x)
end

@testset "Weibull Hazards - Analytic Validation" begin
    # Test: h(t) = λκt^(κ-1)
    # Test: H(t) = λt^κ
    # Test: With covariates (PH)
end

@testset "Gompertz Hazards - Analytic Validation" begin
    # Test: h(t) = λγ exp(γt)
    # Test: H(t) = λ(exp(γt) - 1)
end

@testset "PH vs AFT Modes" begin
    # One clear test showing the difference
end
```

### Phase 3: Consolidate Phase-Type Tests (Priority 3)

**Current:** 4 separate files (correctness, fitting, IS, simulation)
**Target:** 1-2 files maximum

**Keep:** `test_phasetype_correctness.jl` as primary file (it's excellent)

**Consolidate into it:**
- Best fitting tests from `test_phasetype_fitting.jl`
- Essential IS tests from `test_phasetype_is.jl`  
- Core simulation tests from `test_phasetype_simulation.jl`

**Remove:**
- Weak tests that don't validate against analytics
- Redundant "sanity" tests

### Phase 4: Remove Setup Files (Priority 4)

**Replace with inline fixtures:**
```julia
# BEFORE (obscure)
fixture = toy_expwei_model()
model = fixture.model

# AFTER (clear)
data = DataFrame(id=[1,1], tstart=[0.0,1.0], tstop=[1.0,2.0], ...)
h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
h13 = Hazard(@formula(0 ~ age), "wei", 1, 3)
model = multistatemodel(h12, h13; data=data)
```

**Files to remove:**
- setup_2state_trans.jl
- setup_3state_expwei.jl
- setup_3state_weiph.jl
- setup_gompertz.jl
- setup_splines.jl (or consolidate into test_splines.jl)
- setup_ffbs.jl
- setup_prog_expwei.jl

### Phase 5: Remove/Consolidate Weak Tests (Priority 5)

**test_make_subjdat.jl** - Remove or minimal
- Tests internal data transformation
- Not user-facing functionality
- Keep only if critical for debugging

**test_reversible_tvc_loglik.jl** - Remove or strengthen
- Weak statistical test
- No analytic validation
- Decision: Remove unless can add analytic validation

**test_helpers.jl** - Consolidate
- Extract parameter handling tests → new `test_parameters.jl`
- Extract batching tests → consolidate into `test_parallel_likelihood.jl`
- Remove weak utility tests

## Testing Principles Going Forward

1. **Test Correctness, Not Implementation**
   - Verify mathematical properties
   - Compare against analytic solutions
   - Don't test internal data structures

2. **Use Property-Based Testing**
   - Test invariants that must hold
   - Example: Row sums of intensity matrices = 0
   - Example: Survival probability ∈ [0,1]

3. **Clear, Self-Contained Tests**
   - Each test should be understandable in isolation
   - Inline fixtures preferred over setup files
   - Clear comments explaining what's being validated

4. **Minimize Test Code**
   - One test per mathematical property
   - Remove redundant tests
   - Target: <3000 lines total test code (down from 6113)

5. **Follow test_phasetype_correctness.jl as Model**
   - Excellent documentation
   - Analytic validation
   - Clear structure
   - Helper functions for complex math

## Success Metrics

- ✅ All tests passing
- ✅ <3000 lines of test code (50% reduction)
- ✅ <20 test files (60% reduction)
- ✅ Every test validates correctness against analytics or properties
- ✅ No "sanity" or "smoke" tests without clear purpose
- ✅ Test execution time <2 minutes for unit tests

## Implementation Order

1. **Day 1:** Fix failing tests (Phase 1)
2. **Day 2:** Streamline test_hazards.jl (Phase 2)
3. **Day 3:** Consolidate phase-type tests (Phase 3)
4. **Day 4:** Remove setup files (Phase 4)  
5. **Day 5:** Clean up remaining weak tests (Phase 5)

## Risk Mitigation

- Keep backup of original test files
- Run full test suite after each phase
- Document any removed tests in CHANGELOG
- Long tests (`longtest_*.jl`) reviewed separately, not touched in this refactor
