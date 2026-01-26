# Reports, Benchmarks, and Diagnostics Overhaul Plan

**Date**: 2026-01-24  
**Branch**: `penalized_splines`  
**Author**: julia-statistician

---

## Executive Summary

This document inventories all report, benchmark, and diagnostic files that need updating to align with the current MultistateModels.jl API on the `penalized_splines` branch.

### Key API Changes

| Old API | New API |
|---------|---------|
| `select_smoothing_parameters(model, SplinePenalty(); method=:pijcv)` | `fit(model; penalty=:auto, select_lambda=:pijcv)` |
| `SplinePenalty()` exported type | `penalty=:auto` or `penalty=:spline` |
| `result.beta`, `result.lambda`, `result.edf.total` | `fitted.smoothing_parameters`, `fitted.edf`, `fitted.loglik` |
| `get_parameters(model)` returns flat vector per hazard | `get_parameters(fitted)` returns nested NamedTuple |
| `get_parameters_flat(fitted)` | Returns flat Vector{Float64} |
| `compute_vcov=true`, `compute_ij_vcov=true`, `compute_jk_vcov=true` | `vcov_type=:model`, `vcov_type=:ij` (default), `vcov_type=:jk` |
| Log-scale parameters: `[log(shape), log(rate)]` | Natural-scale: `[shape, rate]` with box constraints |

---

## File Inventory

### Priority Legend
- **CRITICAL**: Blocks other work or has extensive deprecated API usage
- **HIGH**: Contains deprecated API that will cause errors
- **MEDIUM**: May have stale patterns or documentation
- **LOW**: Cosmetic or optional updates

---

## STREAM A: Quarto Reports (.qmd files)

### 1. spline_comparison_benchmark.qmd (CRITICAL)

**Location**: `MultistateModelsTests/reports/spline_comparison_benchmark.qmd`  
**Lines**: 1939  
**Purpose**: Comprehensive benchmark comparing MultistateModels.jl spline methods vs R (mgcv, flexsurv)

**Issues Found**:
- Line 171: `select_smoothing_parameters(model, SplinePenalty(); method = :pijcv, ...)` × 8+ occurrences
- Lines 172-199: Uses deprecated API extensively in julia-fitting bash block
- Lines 1356-1368: More `select_smoothing_parameters` calls in illness-death section
- Lines 175-199: Accesses `result_pijcv.lambda[1]`, `result_pijcv.edf.total` (old structure)
- Lines 222-225: Uses `MultistateModels.loglik_exact(result_pijcv.beta, ...)` directly

**Required Changes**:
1. Replace all `select_smoothing_parameters(model, SplinePenalty(); method=X)` with `fit(model; penalty=:auto, select_lambda=X)`
2. Update result access: `fitted.smoothing_parameters.h12` instead of `result.lambda[1]`
3. Update EDF access: `fitted.edf.h12` or `fitted.edf.total`
4. Update beta access: `get_parameters_flat(fitted)` instead of `result.beta`
5. Recompute log-likelihood: use `get_loglik(fitted)` or internal function with fitted params

**Estimated Effort**: 3-4 hours  
**Dependencies**: None

---

### 2. 04_simulation_diagnostics.qmd (MEDIUM)

**Location**: `MultistateModelsTests/reports/04_simulation_diagnostics.qmd`  
**Lines**: 592  
**Purpose**: Visual validation of simulation engine distributions

**Issues Found**:
- Line 24-25: Imports `get_hazard_params` which may be renamed/moved
- Uses `set_parameters!` which is still valid but check scale

**Required Changes**:
1. Verify `get_hazard_params` import still works or update to new accessor
2. Verify parameter scale (natural vs log)

**Estimated Effort**: 30 minutes  
**Dependencies**: None

---

### 3. 03_long_tests.qmd (LOW)

**Location**: `MultistateModelsTests/reports/03_long_tests.qmd`  
**Lines**: 1231  
**Purpose**: Long test results display

**Issues Found**:
- No direct API calls, reads from JSON result files
- May need updates if longtest result structure changed

**Required Changes**:
1. Verify result file format compatibility
2. No code changes expected unless longtests changed

**Estimated Effort**: 15 minutes review  
**Dependencies**: Long test infrastructure

---

### 4. 02_unit_tests.qmd (LOW)

**Location**: `MultistateModelsTests/reports/02_unit_tests.qmd`  
**Lines**: 882  
**Purpose**: Unit test documentation

**Issues Found**:
- Documentation only, no executable Julia code
- References may be stale

**Required Changes**:
1. Update any references to old API in documentation

**Estimated Effort**: 15 minutes review  
**Dependencies**: None

---

### 5. 01_architecture.qmd (LOW)

**Location**: `MultistateModelsTests/reports/01_architecture.qmd`  
**Lines**: 143  
**Purpose**: Architecture overview (placeholder)

**Issues Found**:
- Line 37: Shows old `fit!(model)` pattern

**Required Changes**:
1. Update to `fit(model)` (returns fitted object, doesn't mutate)

**Estimated Effort**: 5 minutes  
**Dependencies**: None

---

### 6. 05_benchmarks.qmd (LOW)

**Location**: `MultistateModelsTests/reports/05_benchmarks.qmd`  
**Lines**: ~80  
**Purpose**: Placeholder report

**Issues Found**:
- Placeholder with `eval: false` code blocks
- No functional code to update

**Required Changes**:
1. None required (placeholder)

**Estimated Effort**: 0  
**Dependencies**: None

---

### 7. index.qmd (LOW)

**Location**: `MultistateModelsTests/reports/index.qmd`  
**Lines**: 117  
**Purpose**: Report landing page

**Issues Found**:
- No API calls, just navigation

**Required Changes**:
1. None required

**Estimated Effort**: 0  
**Dependencies**: None

---

## STREAM B: Benchmark Scripts

### 8. run_benchmarks.jl (MEDIUM)

**Location**: `MultistateModelsTests/benchmarks/run_benchmarks.jl`  
**Lines**: ~130  
**Purpose**: Main benchmark runner

**Issues Found**:
- Line 30: Uses log-scale parameters `[log(1.1), log(0.5)]` - NEEDS UPDATE to natural scale
- Line 57: Uses `vcov_type=:none` (correct new API ✓)

**Required Changes**:
1. Update parameter initialization to natural scale: `[1.1, 0.5]` instead of `[log(1.1), log(0.5)]`

**Estimated Effort**: 20 minutes  
**Dependencies**: None

---

### 9. generate_benchmark_data.jl (HIGH)

**Location**: `MultistateModelsTests/benchmarks/spline_comparison/generate_benchmark_data.jl`  
**Lines**: 338  
**Purpose**: Generate illness-death data for spline comparison

**Issues Found**:
- Line 17: Comment says "Updated 2026-01-24: Uses new fit() API" but need to verify
- Check if `get_parameters` usage is current

**Required Changes**:
1. Verify new API is actually implemented
2. Test that script runs without errors

**Estimated Effort**: 30 minutes  
**Dependencies**: None

---

### 10. visualize_comparison.jl (MEDIUM)

**Location**: `MultistateModelsTests/benchmarks/spline_comparison/visualize_comparison.jl`  
**Purpose**: Visualization of benchmark results

**Required Changes**:
1. Review for API compatibility
2. Test execution

**Estimated Effort**: 15 minutes  
**Dependencies**: generate_benchmark_data.jl

---

## STREAM C: Diagnostic Scripts

### 11. generate_model_diagnostics.jl (MEDIUM)

**Location**: `MultistateModelsTests/diagnostics/generate_model_diagnostics.jl`  
**Lines**: 1048  
**Purpose**: Generate simulation diagnostics

**Issues Found**:
- Line 18-21: Conditionally loads MultistateModels (OK)
- Line 24-27: Imports `get_hazard_params` - may need update
- Uses `model.parameters.nested` directly which is still valid

**Required Changes**:
1. Verify `get_hazard_params` import path
2. Test script execution

**Estimated Effort**: 30 minutes  
**Dependencies**: None

---

### 12. mcem_tvc_diagnostic.jl (HIGH)

**Location**: `MultistateModelsTests/diagnostics/mcem_tvc_diagnostic.jl`  
**Lines**: 178  
**Purpose**: MCEM + TVC debugging

**Issues Found**:
- Line 2: Imports `get_parameters_flat` (correct ✓)
- Line 12-17: Uses `log(5.0)` for scale parameter - NEEDS UPDATE to natural scale
- Line 86: Uses `vcov_type` - need to verify kwarg name

**Required Changes**:
1. Update true_params to natural scale (remove log transforms)
2. Verify `fit()` kwargs

**Estimated Effort**: 30 minutes  
**Dependencies**: None

---

### 13. spline_mcem_diagnostic.jl (MEDIUM)

**Location**: `MultistateModelsTests/diagnostics/spline_mcem_diagnostic.jl`  
**Lines**: 470  
**Purpose**: Diagnose spline MCEM boundary coefficient issues

**Issues Found**:
- Line 28-29: Imports many internal functions (need to verify still exported)
- Uses internal APIs which may have changed

**Required Changes**:
1. Verify all imports still exist
2. Test script execution

**Estimated Effort**: 45 minutes  
**Dependencies**: None

---

## STREAM D: Scripts and Documentation

### 14. benchmark_illness_death.jl (HIGH)

**Location**: `MultistateModelsTests/scripts/benchmark_illness_death.jl`  
**Lines**: 493  
**Purpose**: Benchmark vs mgcv for illness-death

**Issues Found**:
- Line 75-78: Uses `log(true_shape_12)`, `log(true_rate_12)` - NEEDS natural scale
- Line 85: Uses `get_parameters(model; scale=:natural)` - verify this accessor exists

**Required Changes**:
1. Update to natural-scale parameters
2. Verify `get_parameters` accessor API

**Estimated Effort**: 30 minutes  
**Dependencies**: None

---

### 15. compare_mgcv.jl (PARTIALLY UPDATED)

**Location**: `MultistateModelsTests/scripts/compare_mgcv.jl`  
**Lines**: 401  
**Purpose**: Compare vs R mgcv PAM

**Issues Found**:
- Line 12: Comment says "Updated 2026-01-24: Uses new fit() API"
- Need to verify the update is complete

**Required Changes**:
1. Verify new API implementation
2. Test execution

**Estimated Effort**: 20 minutes  
**Dependencies**: None

---

### 16. docs/src/optimization.md (HIGH)

**Location**: `docs/src/optimization.md`  
**Lines**: 303  
**Purpose**: Optimization and variance documentation

**Issues Found**:
- Line 57-58: Documents `compute_vcov=true` - DEPRECATED
- Line 68: Documents `compute_ij_vcov=true` - DEPRECATED
- Line 78: Documents `compute_jk_vcov=false` - DEPRECATED

**Required Changes**:
1. Replace all `compute_*_vcov` with new `vcov_type=:X` documentation
2. Update code examples
3. Update tables

**Estimated Effort**: 45 minutes  
**Dependencies**: None

---

### 17. docs/src/index.md (LOW)

**Location**: `docs/src/index.md`  
**Lines**: 298  
**Purpose**: Main package documentation

**Issues Found**:
- Quick start code looks current
- May have minor inconsistencies

**Required Changes**:
1. Review for API consistency

**Estimated Effort**: 15 minutes  
**Dependencies**: None

---

### 18. Other scripts in MultistateModelsTests/scripts/

| File | Priority | Notes |
|------|----------|-------|
| compute_julia_predictions.jl | LOW | Review |
| generate_julia_data.jl | LOW | Review |
| refresh_cache.jl | LOW | Infrastructure |
| run_all_tests.jl | LOW | Test runner |
| run_longtests.jl | LOW | Test runner |
| run_tests_parallel.jl | LOW | Test runner |
| test_cache.jl | LOW | Infrastructure |

---

## Execution Plan

### Phase 1: Critical Path (Stream A - Reports)

1. **spline_comparison_benchmark.qmd** - Most complex, highest priority
   - Update all `select_smoothing_parameters` calls
   - Update result access patterns
   - Test with `quarto render`

### Phase 2: High Priority (Parallel)

Run in parallel:
- **Stream B**: run_benchmarks.jl, generate_benchmark_data.jl
- **Stream C**: mcem_tvc_diagnostic.jl
- **Stream D**: benchmark_illness_death.jl, docs/src/optimization.md

### Phase 3: Medium Priority

- generate_model_diagnostics.jl
- spline_mcem_diagnostic.jl
- 04_simulation_diagnostics.qmd
- compare_mgcv.jl

### Phase 4: Low Priority / Review

- All other scripts
- Documentation review
- Report regeneration

### Phase 5: Validation

1. Run `Pkg.test()` to ensure no regressions
2. Render all Quarto reports
3. Execute benchmark scripts
4. Execute diagnostic scripts
5. Final review

---

## Dependencies Graph

```
spline_comparison_benchmark.qmd (standalone)
    ↓
generate_benchmark_data.jl → visualize_comparison.jl
    ↓
benchmark_illness_death.jl (standalone)
    ↓
compare_mgcv.jl (standalone)
```

---

## Summary Table

| File | Priority | Deprecated API | Estimated Time |
|------|----------|----------------|----------------|
| spline_comparison_benchmark.qmd | CRITICAL | 10+ calls | 3-4 hours |
| docs/src/optimization.md | HIGH | 3 sections | 45 min |
| mcem_tvc_diagnostic.jl | HIGH | log params | 30 min |
| benchmark_illness_death.jl | HIGH | log params | 30 min |
| run_benchmarks.jl | MEDIUM | log params | 20 min |
| generate_benchmark_data.jl | HIGH | verify | 30 min |
| generate_model_diagnostics.jl | MEDIUM | imports | 30 min |
| spline_mcem_diagnostic.jl | MEDIUM | imports | 45 min |
| 04_simulation_diagnostics.qmd | MEDIUM | imports | 30 min |
| compare_mgcv.jl | MEDIUM | verify | 20 min |
| 03_long_tests.qmd | LOW | none | 15 min |
| 02_unit_tests.qmd | LOW | docs only | 15 min |
| 01_architecture.qmd | LOW | 1 call | 5 min |
| docs/src/index.md | LOW | review | 15 min |
| Other scripts | LOW | review | 30 min |
| **TOTAL** | | | ~9-10 hours |

---

## Change Log

| Date | Action |
|------|--------|
| 2026-01-24 | Initial inventory created |

