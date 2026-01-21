# Phase-Type Report Fixes - Action Items

**Created**: 2026-01-13
**Status**: ✅ COMPLETE

## Fix Checklist

### Critical (False Claims)
- [x] 1. Fix pt_panel_tvc status: Changed from "✅ Pass" to "⚠️ Known Issue" in status table
- [x] 2. Fix overall status table: Changed from "7 | 7 | 0" to "9 | 7 | 2"

### Plotting Infrastructure
- [x] 3. Make plot_state_prevalence dynamic: Now reads `result.n_states` from result
- [x] 4. Make plot_cumulative_incidence dynamic: Now reads keys from `result.cumulative_incidence_true`
- [x] 5. Handle missing keys gracefully: Added null checks for confidence band keys

### Data Capture
- [x] 6. Add prevalence_observed computation to capture_phasetype_longtest_result!

### Documentation
- [x] 7. Update callout note to honestly state what curves are shown (True/Estimated only)

## Implementation Log

### Fix 1-2, 7: False Claims in 03_long_tests.qmd
- Changed pt_panel_tvc status from "✅ Pass" to "⚠️ Known Issue"
- Updated overall phase-type count from 7/7/0 to 9/7/2
- Updated callout note to honestly state prevalence_observed is not shown

### Fix 3-5: Dynamic Plotting Functions in 03_long_tests.qmd
- `plot_state_prevalence`: Now iterates `1:result.n_states` instead of hardcoded 1:3
- `plot_cumulative_incidence`: Now uses `keys(result.cumulative_incidence_true)` instead of hardcoded ["1→2", "2→3"]
- Both functions now check for key existence before accessing confidence band dicts

### Fix 6: Add prevalence_observed to capture_phasetype_longtest_result!
- File: `MultistateModelsTests/longtests/longtest_helpers.jl`
- Added `compute_observed_prevalence(fitted.data, eval_times, n_observed_states)` call
- For phase-type panel tests, `fitted.data` is already in observed state space (phases collapsed when creating panel observations)
- Stores result in `result.prevalence_observed[string(s)]` for each state

## Files Modified

1. `MultistateModelsTests/reports/03_long_tests.qmd`
   - Line ~1090: plot_state_prevalence made dynamic
   - Line ~1130: plot_cumulative_incidence made dynamic  
   - Line ~1465: Phase-type overview table corrected
   - Line ~1503: pt_panel_tvc status corrected

2. `MultistateModelsTests/longtests/longtest_helpers.jl`
   - Line ~1150: Added observed prevalence computation to capture_phasetype_longtest_result!

## Validation Status

- [ ] Re-run phase-type longtests to regenerate JSON cache with prevalence_observed
- [ ] Re-render 03_long_tests.qmd to verify plots work correctly
- [ ] Verify no Julia errors in report generation
