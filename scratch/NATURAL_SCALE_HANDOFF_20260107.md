# Natural Scale Test Suite Update - Handoff Document

**Date**: 2026-01-07  
**Branch**: `penalized_splines`  
**Status**: Test parameter updates complete; architectural issue discovered

---

## Current State

### What Works
- All `set_parameters!` calls in test files now use natural-scale parameters
- Test fixtures updated to use natural-scale parameters
- Package compiles and loads successfully
- Reduced failures from 112 → 81

### What's Incomplete
- **81 test failures remain** (21 failed + 60 errored) due to an architectural issue
- The failures are NOT in test parameter settings - they're in the likelihood computation

### What's Broken/Blocked
During optimization (especially during automatic parameter initialization), Ipopt evaluates the likelihood at trial parameter values. Even with box constraints `lb ≥ 0`, Ipopt's internal line search may probe points where parameters are slightly negative. The likelihood functions call `log(haz_value)` without protection, causing `DomainError` when hazard values become negative.

---

## Key Decisions Made

### Design Choices
1. **Pattern applied**: `[log(value)]` → `[value]` for all baseline parameters
2. **Covariate coefficients**: Unchanged (already on natural scale, can be negative)
3. **Gompertz**: Shape stays unconstrained (can be negative for decreasing hazard), rate on natural scale
4. **Weibull**: Both shape and scale on natural scale (both positive-constrained)

### Rejected Alternatives
- Did not add `initialize=false` to all test fixtures (would mask the real issue)
- Did not wrap parameters in `max(x, eps())` in tests (issue is in core code, not tests)

---

## Files Modified

### Unit Tests (MultistateModelsTests/unit/)
| File | Changes |
|------|---------|
| test_compute_hazard.jl | 16 `set_parameters!` calls updated |
| test_regressions.jl | 1 call updated |
| test_error_messages.jl | 2 calls updated |
| test_helpers.jl | 3 calls updated |
| test_infrastructure.jl | 4 calls updated |
| test_reversible_tvc_loglik.jl | 6 calls updated |
| test_initialization.jl | 8 calls updated (including `exp(flat[i])` → `flat[i]`) |
| test_ad_backends.jl | 4 calls updated |
| test_variance.jl | 3 calls updated |
| test_numerical_stability.jl | 16 calls updated |
| test_model_output.jl | 3 calls updated |
| test_simulation.jl | 1 call updated |
| test_hazards.jl | 14 calls updated |

### Integration Tests (MultistateModelsTests/integration/)
| File | Changes |
|------|---------|
| test_parameter_ordering.jl | 10+ calls updated, `exp(get_parameters_flat(...))` → `get_parameters_flat(...)` |

### Long Tests (MultistateModelsTests/longtests/)
| File | Changes |
|------|---------|
| longtest_mcem_splines.jl | 4 calls updated |
| longtest_mcem_tvc.jl | 2 calls updated |
| longtest_robust_markov_phasetype.jl | 2 calls updated |
| longtest_robust_parametric.jl | 2 calls updated |

### Fixtures (MultistateModelsTests/fixtures/)
| File | Changes |
|------|---------|
| TestFixtures.jl | 7 fixture functions updated |

---

## Next Steps

### Priority 1: Fix likelihood functions (ARCHITECTURAL)
The likelihood functions need to handle negative hazard values gracefully during optimization.

**Files to modify:**
- `src/likelihood/loglik_exact.jl` - Lines 591, 671
- `src/likelihood/loglik_markov.jl` - Lines 338, 364, 473, 504  
- `src/likelihood/loglik_markov_functional.jl` - Lines 107, 128, 187, 225
- `src/likelihood/loglik_utils.jl` - Line 137
- `src/likelihood/loglik_semi_markov.jl` - Line 302

**Recommended fix**: Replace `log(haz_value)` with `NaNMath.log(haz_value)` which returns `NaN` instead of throwing an error. The optimizer will then reject the out-of-bounds step.

### Priority 2: Verify all tests pass after fix
After the likelihood fix, re-run the full test suite.

### Priority 3: Check for any remaining log() calls
Search for patterns like:
```bash
grep -rn 'exp(.*get_parameters' MultistateModelsTests/
grep -rn 'set_parameters!.*log(' MultistateModelsTests/
```

---

## Context for New Session

### Essential Background
- The constrained optimization refactor changed parameter storage from log-scale to natural scale
- Parameters are now constrained via Ipopt box constraints (`lb ≥ 0`) instead of exp() transforms
- All user-facing `set_parameters!` calls must now pass natural-scale values
- The `get_parameters_flat()` function now returns natural-scale values (no exp() needed)

### Stack Trace of Root Issue
```
DomainError with -0.3619763025456705:
log was called with a negative real argument...

Stacktrace:
 [4] _compute_path_loglik_fused(...) at loglik_exact.jl:591
     ll += log(haz_value)  # <-- haz_value is negative
```

This occurs during `multistatemodel()` → `initialize_parameters!()` → `_fit_markov_surrogate()` → `fit()` → Ipopt optimization.

### Test Results Progression
| Stage | Passed | Failed | Errored | Total Failures |
|-------|--------|--------|---------|----------------|
| Before updates | 981 | 37 | 75 | 112 |
| After test updates | 1031 | 21 | 60 | 81 |

---

## Verification Commands

```bash
# Check for remaining log() in set_parameters!
grep -rn 'set_parameters!.*log(' MultistateModelsTests/

# Check for exp() around get_parameters
grep -rn 'exp(.*get_parameters' MultistateModelsTests/

# Run tests
cd MultistateModels.jl
julia --project -e 'using Pkg; Pkg.test()'
```
