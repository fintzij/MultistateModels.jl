# Test Suite Status Report

**Date:** November 7, 2025  
**Branch:** infrastructure_changes  
**Context:** Complete review of all tests in test/ folder

## Summary

**Critical Findings:**
- ❌ Main test suite (`test/runtests.jl`) FAILS
- ❌ Cannot run most tests due to blocking bugs
- ✅ Infrastructure walkthrough scripts (9/9) PASS
- ⚠️ Multiple critical bugs prevent testing

## Detailed Test Status

### Core Test Files in `test/runtests.jl`

```julia
@testset "runtests" begin
    include("test_modelgeneration.jl")      # ❓ Status unknown - blocked by setup
    include("test_hazards.jl")              # ❓ Status unknown - blocked by setup  
    include("test_helpers.jl")              # ❓ Status unknown - blocked by setup
    include("test_make_subjdat.jl")         # ❓ Status unknown - blocked by setup
    include("test_loglik.jl")               # ❓ Status unknown - blocked by setup
end
```

**Blocker:** Setup files fail before tests can run
**Error:** `setup_3state_expwei.jl` fails with interaction term error (`trt*age`)

### Setup Files

| File | Status | Issue |
|------|--------|-------|
| `setup_2state_trans.jl` | ❓ | Not tested - blocked by other setups |
| `setup_3state_expwei.jl` | ❌ | FAILS - interaction terms not supported |
| `setup_3state_weiph.jl` | ❓ | Not tested |
| `setup_ffbs.jl` | ❓ | Not tested |
| `setup_gompertz.jl` | ❓ | Not tested |
| `setup_prog_expwei.jl` | ❓ | Not tested |
| `setup_splines.jl` | ❓ | Not tested |

**Key Error in `setup_3state_expwei.jl`:**
```julia
h13 = Hazard(@formula(0 ~ 1 + trt*age), "exp", 1, 3)
# Later: ArgumentError: column name :trt & age not found in the data frame
```

### Long Tests (Not in Main Suite)

| File | Purpose | Status |
|------|---------|--------|
| `longtest_exp_vs_wei_loglik.jl` | Compare exponential vs Weibull | ❓ |
| `longtest_exp_vs_wei_sanity.jl` | Sanity checks | ❓ |
| `longtest_exponential_mle.jl` | MLE testing | ❓ |
| `longtest_ffbs.jl` | Forward-filtering backward-sampling | ❓ |
| `longtest_markov_cens.jl` | Markov with censoring | ❓ |
| `longtest_sample_ecctmc.jl` | ECCTMC sampling | ❓ |
| `longtest_simulate_paths.jl` | Path simulation | ❌ Likely fails (Bug #4) |

### New Infrastructure Tests (Phase 3)

| File | Purpose | Status |
|------|---------|--------|
| `test_name_based_covariates.jl` | Name-based covariate matching | ❓ |
| `test_parameterhandling.jl` | Parameter management | ❓ |
| `test_pathfunctions.jl` | Path functions | ❓ |
| `test_phase25_log_scale.jl` | Log-scale parameters | ❓ |
| `test_phase3_task1.jl` | Phase 3 task 1 | ❓ |
| `test_phase3_task2.jl` | Phase 3 task 2 | ❓ |
| `test_phase3_task3.jl` | Phase 3 task 3 | ❓ |
| `test_phase3_task4.jl` | Phase 3 task 4 | ❓ |
| `test_phase3_task5.jl` | Phase 3 task 5 | ❓ |

### New Tests Created (Not Yet in Suite)

| File | Purpose | Status |
|------|---------|--------|
| `test_simulation_validation.jl` | Simulation distribution validation | ❌ FAILS (Bug #4) |

## Blocking Bugs

### Bug #1: Parameter Names Include Data Values (CRITICAL)
- **Location:** `src/modelgeneration.jl`
- **Impact:** Breaks ALL covariate models
- **Example:** Parameter names become `[:h12_age: 40.651..., :h12_age: 40.916..., ...]`
- **Priority:** **IMMEDIATE FIX REQUIRED**

### Bug #4: Simulation Requires Multi-Row Data
- **Location:** `src/simulation.jl`
- **Impact:** Cannot simulate from simple initial states
- **Error:** `BoundsError: attempt to access 1×6 SubDataFrame at index [2, :]`
- **Priority:** **HIGH - Blocks simulation testing**

### Bug #5: Interaction Terms Not Supported
- **Location:** `src/hazards.jl`, `src/modelgeneration.jl`
- **Impact:** Blocks setup files and tests using `trt*age` syntax
- **Error:** `ArgumentError: column name :trt & age not found`
- **Priority:** **HIGH - Blocks existing test suite**

## What CAN Be Tested

✅ **Infrastructure Walkthroughs (All Passing):**
1. `walkthrough_00_reference.jl` - ✅ PASS
2. `walkthrough_01_basic_model.jl` - ✅ PASS
3. `walkthrough_02_model_fitting.jl` - ✅ PASS
4. `walkthrough_03_with_covariates.jl` - ✅ PASS
5. `walkthrough_04_multistate.jl` - ✅ PASS
6. `walkthrough_05_hazard_families.jl` - ✅ PASS
7. `walkthrough_06_different_covariates.jl` - ✅ PASS
8. `walkthrough_07_simulation.jl` - ✅ PASS
9. `walkthrough_08_summary.jl` - ✅ PASS

**Note:** These test model *creation* but not *fitting* or *simulation*

## Recommendations

### Immediate Actions Required

1. **Fix Bug #1 (Parameter Naming)** - CRITICAL
   - Prevents all covariate model fitting
   - Required before ANY covariate tests can pass
   
2. **Fix Bug #5 (Interaction Terms)** - HIGH
   - OR update setup files to remove interactions
   - Required to run existing test suite
   
3. **Fix Bug #4 (Simulation Data)** - HIGH
   - Required for simulation validation tests
   - Required for `longtest_simulate_paths.jl`

### Test Suite Cleanup

1. **Update `setup_3state_expwei.jl`:**
   - Remove `trt*age` interaction
   - Use separate terms or pre-computed interaction column
   
2. **Run Each Test File Individually:**
   - Once bugs are fixed, test each file separately
   - Identify any additional issues
   
3. **Add New Tests to Main Suite:**
   - `test_simulation_validation.jl` should be in `runtests.jl`
   - Phase 3 tests should be included

### Documentation Needs

1. Document which tests require which bugs to be fixed
2. Create test dependencies matrix
3. Add continuous integration setup
4. Document expected test coverage

## Test Execution Log

### Attempt 1: Full Test Suite
```bash
julia --project=. test/runtests.jl
```
**Result:** ❌ FAIL  
**Error:** Interaction term in `setup_3state_expwei.jl`  
**Location:** Line 7: `h13 = Hazard(@formula(0 ~ 1 + trt*age), "exp", 1, 3)`

### Attempt 2: Simulation Validation
```bash
julia --project=. test/test_simulation_validation.jl
```
**Result:** ❌ FAIL (4/4 tests error)  
**Error:** `BoundsError` in simulation with single-row data  
**Bug:** #4 - Simulation requires panel data structure

### Attempt 3: Infrastructure Walkthroughs
```bash
for f in scratch/walkthrough_*.jl; do julia --project=. "$f"; done
```
**Result:** ✅ PASS (9/9)  
**Note:** These don't test fitting or simulation, only model creation

## Next Steps

1. ✅ Document all bugs (COMPLETE - see `BUG_REPORT.md`)
2. ✅ Create test status report (COMPLETE - this file)
3. ⏳ Fix Bug #1 (parameter naming) - **BLOCKS EVERYTHING**
4. ⏳ Fix Bug #4 (simulation data) or Bug #5 (interactions)
5. ⏳ Re-run full test suite
6. ⏳ Add simulation validation to main test suite
7. ⏳ Set up continuous integration

## Conclusion

**The infrastructure changes are structurally sound** (9/9 walkthrough scripts pass), but **critical bugs prevent testing of core functionality** (fitting, simulation). 

**Immediate priority:** Fix Bug #1 (parameter naming) as it blocks all covariate model use.

**Secondary priority:** Fix Bug #4 or #5 to enable running existing test suite.

Once these bugs are fixed, the full test suite can be executed and validated.
