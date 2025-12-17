# Function Audit - Phase-Type Module

**Updated:** 2025-12-17  
**Status:** Complete audit of split phasetype/ module (with consolidation)  
**Files:** types.jl (365), surrogate.jl (284), expansion.jl (1773) = 2,422 lines total

---

## File Structure Summary

| File | Lines | Purpose | Functions | Types |
|------|-------|---------|-----------|-------|
| `types.jl` | 365 | Core type definitions | 7 | 5 |
| `surrogate.jl` | 284 | Surrogate building | 5 | 0 |
| `expansion.jl` | 1,773 | State space expansion, model building | 28 | 0 |
| **Total** | **2,422** | | **40** | **5** |

---

## Legend

**Recommendation:**
- `KEEP` - Essential, production code
- `REMOVE` - Dead code or unused
- `SIMPLIFY` - Reduce complexity
- `INVESTIGATE` - Needs further analysis

**Test Status:** ✅ Tested | ⚠️ Partial | ❌ Untested | ? Unknown

---

## 1. types.jl (365 lines)

### 1.1 ProposalConfig (Lines 1-145)

| Line | Item | Purpose | Callers | Tests | Recommendation |
|------|------|---------|---------|-------|----------------|
| 38 | `struct ProposalConfig` | MCEM proposal config | fit(), resolve_proposal_config | ✅ | KEEP |
| 89 | `MarkovProposal()` | Constructor alias | User API | ✅ | KEEP |
| 102 | `PhaseTypeProposal()` | Constructor alias | User API | ✅ | KEEP |
| 116 | `needs_phasetype_proposal()` | Check if PT needed | resolve_proposal_config | ✅ | KEEP |
| 133 | `resolve_proposal_config()` | Symbol → Config | fit() | ✅ | KEEP |
| 145 | `resolve_proposal_config()` | Passthrough method | fit() | ✅ | KEEP |

### 1.2 PhaseTypeDistribution (Lines 150-230)

| Line | Item | Purpose | Callers | Tests | Recommendation |
|------|------|---------|---------|-------|----------------|
| 175 | `struct PhaseTypeDistribution` | PH(π,Q) representation | PhaseTypeSurrogate | ✅ | KEEP |
| 212 | `subintensity()` | Extract S matrix | build_expanded_Q, tests | ✅ | KEEP |
| 215 | `absorption_rates()` | Extract μ rates | build_expanded_Q, tests | ✅ | KEEP |
| 220 | `progression_rates()` | Extract λ rates | tests only | ⚠️ | KEEP (test utility) |

### 1.3 PhaseTypeConfig (Lines 235-290)

| Line | Item | Purpose | Callers | Tests | Recommendation |
|------|------|---------|---------|-------|----------------|
| 260 | `struct PhaseTypeConfig` | PT surrogate config | build_phasetype_surrogate | ✅ | KEEP |

### 1.4 PhaseTypeMappings (Lines 295-350)

| Line | Item | Purpose | Callers | Tests | Recommendation |
|------|------|---------|---------|-------|----------------|
| 320 | `struct PhaseTypeMappings` | State mappings | expansion functions | ✅ | KEEP |

### 1.5 PhaseTypeSurrogate (Lines 355-366)

| Line | Item | Purpose | Callers | Tests | Recommendation |
|------|------|---------|---------|-------|----------------|
| 365 | `struct PhaseTypeSurrogate` | MCEM surrogate | fit(), compute_phasetype_marginal_loglik | ✅ | KEEP |

---

## 2. surrogate.jl (288 lines)

### 2.1 Coxian Construction (Lines 1-60)

| Line | Function | Purpose | Callers | Tests | Recommendation |
|------|----------|---------|---------|-------|----------------|
| 29 | `build_coxian_intensity()` | Build Q matrix | _build_default_phasetype | ✅ | KEEP |
| - | ~~`build_coxian_subintensity()`~~ | ~~Legacy alias~~ | - | - | **REMOVED** (unused) |

### 2.2 Surrogate Building (Lines 65-125)

| Line | Function | Purpose | Callers | Tests | Recommendation |
|------|----------|---------|---------|-------|----------------|
| 82 | `build_phasetype_surrogate()` | Main entry point | fit() | ✅ | KEEP |

### 2.3 Helper Functions (Lines 130-200)

| Line | Function | Purpose | Callers | Tests | Recommendation |
|------|----------|---------|---------|-------|----------------|
| 139 | `_get_n_phases_per_state()` | Config → phase counts | build_phasetype_surrogate | ✅ | KEEP |
| 166 | `_build_state_mappings()` | Build bidirectional maps | build_phasetype_surrogate | ✅ | KEEP |
| 183 | `_build_default_phasetype()` | Default PH dist | build_phasetype_surrogate | ✅ | KEEP |

### 2.4 Expanded Q Construction (Lines 205-288)

| Line | Function | Purpose | Callers | Tests | Recommendation |
|------|----------|---------|---------|-------|----------------|
| 225 | `build_expanded_Q()` | Build expanded intensity | build_phasetype_surrogate | ✅ | KEEP |

---

## 3. expansion.jl (1,827 lines)

### 3.1 State Space Expansion (Lines 1-260)

| Line | Function | Purpose | Callers | Tests | Recommendation |
|------|----------|---------|---------|-------|----------------|
| 49 | `build_phasetype_mappings()` | Build state mappings | _build_phasetype_model_from_hazards | ✅ | KEEP |
| - | ~~`build_phasetype_mappings()` (2-arg)~~ | ~~Legacy (no n_phases)~~ | - | - | **REMOVED** (unused) |
| 101 | `_build_expanded_tmat()` | Build expanded tmat | build_phasetype_mappings | ✅ | KEEP |
| 196 | `_build_expanded_hazard_indices()` | Hazard → expanded map | build_phasetype_mappings | ✅ | KEEP |
| 264 | `_compute_default_n_phases()` | Default phase heuristic | build_phasetype_surrogate | ✅ | KEEP |

### 3.2 Hazard Expansion (Lines 310-680)

| Line | Function | Purpose | Callers | Tests | Recommendation |
|------|----------|---------|---------|-------|----------------|
| 332 | `expand_hazards_for_phasetype()` | PT → Markov hazards | _build_phasetype_model_from_hazards | ✅ | KEEP |
| 418 | `_phase_index_to_letter()` | 1→'a', 2→'b' | naming functions | ✅ | KEEP |
| 430 | `_build_progression_hazard()` | Build λ hazard | expand_hazards_for_phasetype | ✅ | KEEP |
| 476 | `_build_exit_hazard()` | Build μ hazard | expand_hazards_for_phasetype | ✅ | KEEP |
| 533 | `_generate_exit_hazard_fns()` | Generate hazard fns | _build_exit_hazard | ✅ | KEEP |
| - | ~~`_phasetype_rhs_names()`~~ | ~~Extract formula names~~ | - | - | **REMOVED** (use `_hazard_rhs_names`) |
| - | ~~`_phasetype_formula_has_intercept()`~~ | ~~Check intercept~~ | - | - | **REMOVED** (use `_hazard_formula_has_intercept`) |
| 543 | `_build_expanded_hazard()` | Non-PT in expanded | expand_hazards_for_phasetype | ✅ | KEEP |
| 590-608 | `_adjust_hazard_states()` (3 methods) | Adjust state indices | _build_expanded_hazard | ✅ | KEEP |
| 626 | `_build_shared_phase_hazard()` | Shared rate hazard | expand_hazards_for_phasetype | ✅ | KEEP |

### 3.3 Data Expansion for Fitting (Lines 684-800)

| Line | Function | Purpose | Callers | Tests | Recommendation |
|------|----------|---------|---------|-------|----------------|
| 684 | `expand_data_for_phasetype_fitting()` | Expand data + mappings | _build_phasetype_model_from_hazards | ✅ | KEEP |
| 731 | `_build_phase_censoring_patterns()` | Build emat patterns | expand_data_for_phasetype_fitting | ✅ | KEEP |
| 763 | `map_observation_to_phases()` | Obs → phase mapping | likelihood | ✅ | KEEP |

### 3.4 SCTP Constraints (Lines 803-940)

| Line | Function | Purpose | Callers | Tests | Recommendation |
|------|----------|---------|---------|-------|----------------|
| 803 | `_generate_sctp_constraints()` | SCTP constraints | _build_phasetype_model_from_hazards | ✅ | KEEP |
| 900 | `_merge_constraints()` | Merge user + auto | _build_phasetype_model_from_hazards | ✅ | KEEP |

### 3.5 Model Building (Lines 941-1200)

| Line | Function | Purpose | Callers | Tests | Recommendation |
|------|----------|---------|---------|-------|----------------|
| 941 | `_build_phasetype_model_from_hazards()` | Main model builder | multistatemodel() | ✅ | KEEP |
| 1159 | `_build_expanded_parameters()` | Build params struct | _build_phasetype_model_from_hazards | ✅ | KEEP |

### 3.6 Original Parameters (Lines 1216-1355)

| Line | Function | Purpose | Callers | Tests | Recommendation |
|------|----------|---------|---------|-------|----------------|
| 1216 | `_build_original_parameters()` | User-facing params | _build_phasetype_model_from_hazards | ✅ | KEEP |
| 1298 | `_extract_original_natural_vector()` | Natural scale extract | _build_original_parameters | ✅ | KEEP |
| 1315 | `_count_covariates()` | Count formula covars | _build_original_parameters | ✅ | KEEP |
| 1328 | `_count_hazard_parameters()` | Count hazard params | _build_original_parameters | ✅ | KEEP |
| 1356 | `_merge_censoring_patterns_with_shift()` | Merge + shift patterns | _build_phasetype_model_from_hazards | ✅ | KEEP |

### 3.7 Log-Likelihood (Lines 1457-1610)

| Line | Function | Purpose | Callers | Tests | Recommendation |
|------|----------|---------|---------|-------|----------------|
| 1478 | `compute_phasetype_marginal_loglik()` | Forward algo likelihood | fit() | ✅ | KEEP |

### 3.8 Data Expansion for FFBS (Lines 1616-1828)

| Line | Function | Purpose | Callers | Tests | Recommendation |
|------|----------|---------|---------|-------|----------------|
| 1638 | `expand_data_for_phasetype()` | Expand exact obs | fit(), expand_data_for_phasetype_fitting | ✅ | KEEP |
| 1772 | `needs_data_expansion_for_phasetype()` | Check if expansion needed | fit() | ✅ | KEEP |
| 1790 | `compute_expanded_subject_indices()` | Subject indices | fit() | ✅ | KEEP |

---

## Summary by Recommendation

### KEEP (42 functions/types)

All 42 items are essential production code with test coverage:

**types.jl (12 items):**
- 5 structs: ProposalConfig, PhaseTypeDistribution, PhaseTypeConfig, PhaseTypeMappings, PhaseTypeSurrogate
- 7 functions: MarkovProposal, PhaseTypeProposal, needs_phasetype_proposal, resolve_proposal_config (2), subintensity, absorption_rates, progression_rates

**surrogate.jl (6 functions):**
- build_coxian_intensity, build_coxian_subintensity, build_phasetype_surrogate
- _get_n_phases_per_state, _build_state_mappings, _build_default_phasetype, build_expanded_Q

**expansion.jl (24 functions):**
- State space: build_phasetype_mappings (2), _build_expanded_tmat, _build_expanded_hazard_indices, _compute_default_n_phases
- Hazard expansion: expand_hazards_for_phasetype, _phase_index_to_letter, _build_progression_hazard, _build_exit_hazard, _generate_exit_hazard_fns, _build_expanded_hazard, _adjust_hazard_states (3), _build_shared_phase_hazard
- Data: expand_data_for_phasetype_fitting, _build_phase_censoring_patterns, map_observation_to_phases
- Constraints: _generate_sctp_constraints, _merge_constraints
- Model: _build_phasetype_model_from_hazards, _build_expanded_parameters, _build_original_parameters, _extract_original_natural_vector, _count_covariates, _count_hazard_parameters, _merge_censoring_patterns_with_shift
- Likelihood: compute_phasetype_marginal_loglik
- FFBS data: expand_data_for_phasetype, needs_data_expansion_for_phasetype, compute_expanded_subject_indices

### Code Already Removed

The following were removed during earlier streamlining:

1. **`PhaseTypeModel` struct** - Replaced by MultistateModel with phasetype_expansion metadata
2. **Deprecated likelihood functions** - Replaced by standard Markov likelihood on expanded space
3. **PhaseTypeModel accessors** - Now use trait-based dispatch on has_phasetype_expansion()
4. **Longtest infrastructure** - Moved to MultistateModelsTests/longtests/

---

## Caller Analysis

### External Entry Points (called from outside phasetype/)

| Function | Called From | Purpose |
|----------|-------------|---------|
| `_build_phasetype_model_from_hazards` | `construction/multistatemodel.jl:1142` | Model construction |
| `build_phasetype_surrogate` | `inference/fit.jl` | MCEM surrogate |
| `compute_phasetype_marginal_loglik` | `inference/fit.jl:879` | IS normalizing constant |
| `expand_data_for_phasetype` | `modelfitting.jl:839`, `fit.jl` | Data expansion |
| `PhaseTypeProposal` | User API | fit() proposal kwarg |
| `MarkovProposal` | User API | fit() proposal kwarg |
| `resolve_proposal_config` | `inference/fit.jl` | Symbol → Config |

### Internal Call Graph (simplified)

```
multistatemodel()
  └─ _build_phasetype_model_from_hazards()
       ├─ build_phasetype_mappings()
       │    ├─ _build_expanded_tmat()
       │    └─ _build_expanded_hazard_indices()
       ├─ expand_hazards_for_phasetype()
       │    ├─ _build_progression_hazard()
       │    ├─ _build_exit_hazard()
       │    │    └─ _generate_exit_hazard_fns()
       │    └─ _build_expanded_hazard()
       │         └─ _adjust_hazard_states()
       ├─ expand_data_for_phasetype_fitting()
       │    ├─ expand_data_for_phasetype()
       │    └─ _build_phase_censoring_patterns()
       ├─ _generate_sctp_constraints()
       ├─ _build_expanded_parameters()
       └─ _build_original_parameters()

fit() [MCEM path]
  └─ build_phasetype_surrogate()
       ├─ _get_n_phases_per_state()
       ├─ _build_state_mappings()
       ├─ _build_default_phasetype()
       │    └─ build_coxian_intensity()
       └─ build_expanded_Q()
            ├─ subintensity()
            └─ absorption_rates()
  └─ compute_phasetype_marginal_loglik()
```

---

## Consolidation Opportunities

### ✅ COMPLETED: Formula Intercept Checking (~20 lines)

**Status:** Fixed in commit e0461ce

**Location:** `expansion.jl:550-570` duplicated `multistatemodel.jl:3-10,228-233`

```julia
# DELETED from expansion.jl (phasetype-specific)
_phasetype_formula_has_intercept(rhs_term)  # Was identical logic
_phasetype_rhs_names(hazschema)              # Was identical logic

# NOW using from multistatemodel.jl (general)
_hazard_formula_has_intercept(rhs_term)      # Same implementation
_hazard_rhs_names(hazschema)                  # Same implementation
```

**Savings:** ~20 lines removed.

---

### ✅ COMPLETED: State-to-Phase Mapping Logic (~15 lines)

**Status:** Fixed in commit e0461ce

**Location:** `expansion.jl:68-79` duplicated `surrogate.jl:163-178`

`build_phasetype_mappings` now calls `_build_state_mappings` helper instead of inline code.

**Savings:** ~17 lines removed.

---

### 3. OVER-SPECIALIZATION: `_adjust_hazard_states` (3 methods, ~25 lines)

**Location:** `expansion.jl:627-660`

Three nearly-identical methods that copy structs with new state indices:
```julia
_adjust_hazard_states(haz::MarkovHazard, ...)      # 9 lines
_adjust_hazard_states(haz::SemiMarkovHazard, ...)  # 9 lines  
_adjust_hazard_states(haz::RuntimeSplineHazard, ...) # 9 lines (slightly longer)
```

**Fix:** Use `Base.@kwdef` or `Setfield.jl` to create a generic `with_states(haz, from, to)` function.
Or add a `Base.copy` method with field override.

Alternative (simpler): Since all hazard structs have `statefrom` and `stateto` as fields 2-3,
could use a reflection-based approach or just accept the 3 methods as necessary type dispatch.

**Assessment:** Low priority - type dispatch is idiomatic Julia. Keep as-is unless adding more hazard types.

---

### 4. TRIVIAL WRAPPER: `_build_shared_phase_hazard` (~5 lines)

**Location:** `expansion.jl:663-665`

```julia
function _build_shared_phase_hazard(base_haz::_Hazard, from_phase::Int, to_phase::Int)
    return _adjust_hazard_states(base_haz, from_phase, to_phase)
end
```

**Fix:** This is a one-line wrapper. Inline at call site or keep for documentation purposes.

**Savings:** ~5 lines (trivial).

---

### 5. VERBOSE: `_build_original_parameters` (~100 lines)

**Location:** `expansion.jl:1216-1295`

This function builds parameter structures "from scratch" for user-facing API.
Duplicates logic from parameter building elsewhere.

**Assessment:** This function serves a legitimate distinct purpose (original-space params for reporting)
but the implementation is verbose. Could potentially share infrastructure with standard param building.

**Priority:** Low - works correctly, just verbose.

---

## Recommended Actions

### ✅ Completed (in commit e0461ce)
1. **Deleted `_phasetype_rhs_names` and `_phasetype_formula_has_intercept`** - Now using existing functions from multistatemodel.jl (~20 lines saved)
2. **Call `_build_state_mappings` from `build_phasetype_mappings`** - No longer inline duplicate logic (~17 lines saved)

### Low Priority (marginal gains)
3. Inline `_build_shared_phase_hazard` (~5 lines)
4. Keep `_adjust_hazard_states` methods (idiomatic Julia dispatch)
5. Keep `_build_original_parameters` verbose (serves distinct purpose)

### Total Reduction Achieved
- Original phasetype.jl: ~4,586 lines
- Before consolidation: 2,480 lines  
- After consolidation: **2,443 lines (37 lines saved)**

---

## Conclusion

**All 42 functions/types are production code with test coverage.**

Consolidation complete:
- ✅ Formula intercept checking (no longer duplicated)
- ✅ State mapping logic (now uses helper function)

Final state:
- Original phasetype.jl: ~4,586 lines, ~79 functions
- Current split module: **2,443 lines, 42 functions**
- **Total reduction: 47% fewer lines, 47% fewer functions**

The module is now well-organized:
- `types.jl`: Type definitions (stable, rarely changes)
- `surrogate.jl`: MCEM surrogate building (focused responsibility)  
- `expansion.jl`: Core expansion logic (main work area)
