# Natural-Scale Parameter Architecture Handoff Document

**Date**: 2025-01-07
**Branch**: penalized_splines
**Last Commit**: WIP: Natural-scale parameter architecture - transforms now identity

---

## Current State

### What Works
1. **Package loads successfully** - All modules compile
2. **Transform functions are now identity** - No exp()/log() in parameter transforms
3. **bounds.jl updated** - NONNEG_LB=0.0, Dict-only API, no estimation scale
4. **_spline_ests2coefs** - Returns natural scale (no exp() for monotone==0)
5. **PenaltyTerm** - No exp_transform field, compute_penalty uses β directly
6. **Ipopt** - honor_original_bounds="yes" added
7. **PIJCV** - max.(beta_loo, 0.0) projection added

### What Fails
- **~100+ test failures** - Tests still use log-scale values in `set_parameters!` calls
- Example: `set_parameters!(model, (h12 = [log(0.1),],))` now stores -2.3 instead of 0.1

### Root Cause
Tests were written to pass log-transformed parameters (estimation scale).
With natural-scale storage, the log values are stored directly, resulting in negative rates.

---

## Files Modified (Phase 6)

| File | Changes |
|------|---------|
| `src/utilities/transforms.jl` | `transform_baseline_to_natural/estimation` now identity, updated docstrings |
| `src/utilities/parameters.jl` | `extract_natural_vector` no longer applies exp() |
| `src/utilities/initialization.jl` | `init_par` returns natural-scale, `set_crude_init!` passes crude rates directly |
| `src/phasetype/expansion_model.jl` | `_extract_original_natural_vector` no longer applies exp() |
| `src/output/accessors.jl` | `get_parameters` no longer applies log() for :flat/:estimation/:log scales |

---

## Next Steps (Priority Order)

### 1. Update Test Suite (Required)
All tests using `set_parameters!` with log-transformed values need updating:

**Pattern to find:**
```julia
set_parameters!(model, (h12 = [log(rate),], ...))
```

**Change to:**
```julia
set_parameters!(model, (h12 = [rate,], ...))  # Natural scale
```

Key test files to update:
- `MultistateModelsTests/unit/test_hazards.jl` - Many log() calls
- `MultistateModelsTests/unit/test_splines.jl`
- `MultistateModelsTests/unit/test_simulation.jl`
- `MultistateModelsTests/unit/test_initialization.jl`
- `MultistateModelsTests/unit/test_phasetype.jl`
- `MultistateModelsTests/unit/test_reversible_tvc_loglik.jl`

### 2. Update Docstrings
The `set_parameters!` docstrings still say "Values should be on log scale".
Update to say "Values should be on natural scale".

### 3. Validate PIJCV
After tests pass, verify PIJCV selects reasonable λ values (the original goal).

---

## Key Architectural Decisions

1. **No backward compatibility** - This is a breaking change
2. **Box constraints via Ipopt** - lb ≥ 0 instead of exp() transforms
3. **Dict-only bounds API** - No vector-based bounds specification
4. **NONNEG_LB = 0.0** - Not 1e-10

---

## Grep Patterns for Test Updates

Find log-scale parameter setting:
```bash
grep -r "log(.*rate\|log(.*scale\|log(.*shape" MultistateModelsTests/
```

Find set_parameters with log():
```bash
grep -r "set_parameters!.*log(" MultistateModelsTests/
```

---

## Context for New Session

1. Read this handoff document
2. The core architecture change is complete (transforms are identity)
3. The main work remaining is updating tests to use natural-scale values
4. Consider using sed/awk for bulk test updates, but verify each change makes sense
5. Some tests may need more careful consideration (e.g., testing log-likelihood values)

---

## Validation Checklist

- [ ] All test files updated to use natural-scale parameters
- [ ] `set_parameters!` docstrings updated  
- [ ] Package tests pass (target: 1400+ tests)
- [ ] PIJCV selects reasonable λ (not extreme values)
- [ ] Penalized spline inference produces reasonable curves
