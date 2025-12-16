# Function Audit - MultistateModels.jl

**Created:** 2025-12-16  
**Purpose:** Detailed function-by-function audit for package streamlining

---

## Audit Legend

**Recommendation codes:**
- `KEEP` - Essential, cannot be removed
- `REMOVE` - Redundant or unused, safe to delete
- `MIGRATE` - Move to different location
- `CONSOLIDATE` - Merge with another function
- `REFACTOR` - Keep but simplify
- `INVESTIGATE` - Need more analysis

**Test status:**
- ✅ Has tests
- ⚠️ Partial tests
- ❌ No tests
- ? Unknown

---

## 1. phasetype.jl (4,586 lines, 79 functions)

### 1.1 Configuration & Proposal Types (Lines 1-250)

| Line | Function/Type | Purpose | Callers | Callees | Tests | Recommendation |
|------|--------------|---------|---------|---------|-------|----------------|
| 94 | `struct ProposalConfig` | Config for MCEM proposals | fit() | - | ? | KEEP (general) |
| 175 | `MarkovProposal()` | Constructor alias | User API | ProposalConfig | ? | KEEP |
| 195 | `PhaseTypeProposal()` | Constructor alias | User API | ProposalConfig | ? | KEEP |
| 217 | `needs_phasetype_proposal()` | Check if PT proposal needed | resolve_proposal_config | - | ? | KEEP |
| 242 | `resolve_proposal_config()` | Symbol → ProposalConfig | fit() | needs_phasetype_proposal | ? | KEEP |

### 1.2 PhaseTypeDistribution (Lines 250-420)

**FINDING:** `PhaseTypeDistribution` IS used by `PhaseTypeSurrogate` - stores n_phases, Q, initial 
per observed state. The dict of `PhaseTypeDistribution` objects in the surrogate provides the
per-state PT parameters used by `build_expanded_Q()`.

**Decision needed:** Keep minimal version or replace with simpler Dict storage?

| Line | Function/Type | Purpose | Callers | Callees | Tests | Recommendation |
|------|--------------|---------|---------|---------|-------|----------------|
| 316 | `struct PhaseTypeDistribution` | Stores PT params per state | PhaseTypeSurrogate, build_expanded_Q | - | ? | **SIMPLIFY** |
| 359 | `subintensity()` | Get S matrix | Tests only | - | ✅ | REMOVE |
| 374 | `absorption_rates()` | Get absorption rates | Tests only | - | ✅ | REMOVE |
| 389 | `progression_rates()` | Get progression rates | Tests only | - | ✅ | REMOVE |

**Note:** Could replace `Dict{Int, PhaseTypeDistribution}` with simpler storage since we only
need n_phases and the rate parameters to build expanded_Q.

### 1.3 PhaseTypeConfig (Lines 420-530)

| Line | Function/Type | Purpose | Callers | Callees | Tests | Recommendation |
|------|--------------|---------|---------|---------|-------|----------------|
| 444 | `struct PhaseTypeConfig` | Config for PT expansion | build_phasetype_surrogate | - | ? | KEEP |
| 524 | `struct PhaseTypeMappings` | State space mappings | Multiple | - | ? | KEEP |

### 1.4 Mapping Functions (Lines 530-900)

| Line | Function/Type | Purpose | Callers | Callees | Tests | Recommendation |
|------|--------------|---------|---------|---------|-------|----------------|
| 611 | `build_phasetype_mappings()` | Build state mappings | expand_hazards_for_phasetype | _build_expanded_tmat | ? | KEEP |
| 658 | `build_phasetype_mappings()` | Overload for n_phases dict | multistatemodel | _build_expanded_tmat | ? | KEEP |
| 701 | `_build_expanded_tmat()` | Build expanded tmat | build_phasetype_mappings | - | ? | KEEP |
| 774 | `_build_expanded_hazard_indices()` | Map hazards to expanded | build_phasetype_mappings | - | ? | KEEP |
| 830 | `has_phasetype_hazards()` | Check for PT hazards | multistatemodel | - | ? | KEEP |
| 840 | `get_phasetype_n_phases()` | Get n_phases for state | multistatemodel | - | ? | KEEP |
| 867 | `_compute_default_n_phases()` | Default phase count | build_phasetype_surrogate | - | ? | MIGRATE (surrogate/) |

### 1.5 Hazard Expansion (Lines 900-1400)

| Line | Function/Type | Purpose | Callers | Callees | Tests | Recommendation |
|------|--------------|---------|---------|---------|-------|----------------|
| 958 | `expand_hazards_for_phasetype()` | PT → expanded Markov hazards | _build_phasetype_model_from_hazards | Multiple | ? | KEEP |
| 1056 | `_build_progression_hazard()` | Build λ hazard | expand_hazards_for_phasetype | - | ? | KEEP |
| 1102 | `_build_exit_hazard()` | Build μ hazard | expand_hazards_for_phasetype | - | ? | KEEP |
| 1159 | `_generate_exit_hazard_fns()` | Generate hazard functions | _build_exit_hazard | - | ? | KEEP |
| 1176 | `_phasetype_rhs_names()` | Get RHS names | _build_exit_hazard | - | ? | KEEP |
| 1206 | `_build_expanded_hazard()` | Non-PT hazard in expanded | expand_hazards_for_phasetype | _adjust_hazard_states | ? | KEEP |
| 1253 | `_adjust_hazard_states()` | Adjust state indices | _build_expanded_hazard | - | ? | KEEP (3 methods) |
| 1289 | `_build_shared_phase_hazard()` | Shared baseline hazard | expand_hazards_for_phasetype | - | ? | INVESTIGATE |
| 1335 | `expand_data_for_phasetype_fitting()` | Expand data for fitting | _build_phasetype_model_from_hazards | - | ? | KEEP |
| 1382 | `_build_phase_censoring_patterns()` | Expand censoring patterns | expand_data_for_phasetype_fitting | - | ? | KEEP |

### 1.6 Observation Mapping (Lines 1400-1650)

| Line | Function/Type | Purpose | Callers | Callees | Tests | Recommendation |
|------|--------------|---------|---------|---------|-------|----------------|
| 1414 | `map_observation_to_phases()` | Map obs to phase space | build_phasetype_emat | - | ? | KEEP |
| 1482 | `_generate_sctp_constraints()` | SCTP constraints | _build_phasetype_model_from_hazards | - | ? | MIGRATE (inference/) |
| 1579 | `_merge_constraints()` | Merge user + auto constraints | _build_phasetype_model_from_hazards | - | ? | MIGRATE (inference/) |
| 1634 | `_build_phasetype_model_from_hazards()` | Build PhaseTypeModel | multistatemodel | Many | ? | REFACTOR |

### 1.7 Parameter Building (Lines 1850-2050)

| Line | Function/Type | Purpose | Callers | Callees | Tests | Recommendation |
|------|--------------|---------|---------|---------|-------|----------------|
| 1851 | `_build_expanded_parameters()` | Build expanded params | _build_phasetype_model_from_hazards | - | ? | KEEP |
| 1902 | `_build_original_parameters()` | Build original params | _build_phasetype_model_from_hazards | - | ? | REMOVE (no PhaseTypeModel) |
| 1947 | `_count_covariates()` | Count formula covars | _build_expanded_parameters | - | ? | MIGRATE (utilities/) |
| 1960 | `_count_hazard_parameters()` | Count hazard params | _build_expanded_parameters | - | ? | MIGRATE (utilities/) |
| 1993 | `_merge_censoring_patterns()` | Merge censoring | _build_phasetype_model_from_hazards | - | ? | KEEP |
| 2032 | `_merge_censoring_patterns_with_shift()` | Merge with shift | _merge_censoring_patterns | - | ? | KEEP |

### 1.8 Coxian Intensity (Lines 2100-2200)

| Line | Function/Type | Purpose | Callers | Callees | Tests | Recommendation |
|------|--------------|---------|---------|---------|-------|----------------|
| 2153 | `build_coxian_intensity()` | Build Coxian Q | build_phasetype_surrogate | - | ? | MIGRATE (surrogate/) |

### 1.9 PhaseTypeSurrogate (Lines 2200-2700)

| Line | Function/Type | Purpose | Callers | Callees | Tests | Recommendation |
|------|--------------|---------|---------|---------|-------|----------------|
| 2208 | `struct PhaseTypeSurrogate` | PT surrogate for MCEM | fit() | - | ? | INVESTIGATE |
| 2232 | `collapse_phases()` | Collapse path to obs | draw_samplepath_phasetype | - | ? | KEEP |
| 2265 | `expand_initial_state()` | Map initial state | draw_samplepath_phasetype | - | ? | KEEP |
| 2321 | `build_phasetype_surrogate()` | Build surrogate | fit() | Many | ? | INVESTIGATE |
| 2388 | `_get_n_phases_per_state()` | Phase count per state | build_phasetype_surrogate | - | ? | KEEP |
| 2428 | `_build_state_mappings()` | Build phase mappings | build_phasetype_surrogate | - | ? | KEEP |
| 2461 | `_build_default_phasetype()` | Default PT params | build_phasetype_surrogate | - | ? | KEEP |
| 2528 | `build_expanded_Q()` | Build expanded Q matrix | build_phasetype_surrogate | - | ? | INVESTIGATE |
| 2637 | `update_expanded_Q()` | Update Q with params | fit() | - | ? | INVESTIGATE |

### 1.10 Emission Matrix (Lines 2700-2900)

| Line | Function/Type | Purpose | Callers | Callees | Tests | Recommendation |
|------|--------------|---------|---------|---------|-------|----------------|
| 2736 | `build_phasetype_emission_matrix()` | Build emat | fit() | - | ? | INVESTIGATE |
| 2777 | `build_phasetype_emission_matrix()` | Overload | fit() | - | ? | INVESTIGATE |
| 2809 | `build_phasetype_emission_matrix_censored()` | Censored emat | build_phasetype_emission_matrix | - | ? | INVESTIGATE |
| 2865 | `collapse_emission_matrix()` | Collapse emat | ? | - | ? | INVESTIGATE |

### 1.11 Deprecated Functions (Lines 2880-3020)

| Line | Function/Type | Purpose | Callers | Callees | Tests | Recommendation |
|------|--------------|---------|---------|---------|-------|----------------|
| 2894 | `compute_phasetype_marginal_loglik_deprecated()` | Marginal loglik | fit() | - | ✅ | REPLACE |

### 1.12 Model Building (Lines 3020-3650)

| Line | Function/Type | Purpose | Callers | Callees | Tests | Recommendation |
|------|--------------|---------|---------|---------|-------|----------------|
| 3022 | `build_phasetype_hazards()` | Build hazards for surrogate | build_phasetype_surrogate | - | ? | MIGRATE (surrogate/) |
| 3097 | `build_expanded_tmat()` | Build expanded tmat | build_phasetype_surrogate | - | ? | MIGRATE (surrogate/) |
| 3154 | `build_phasetype_emat()` | Build emat from data | fit() | map_observation_to_phases | ? | INVESTIGATE |
| 3215 | `expand_data_states!()` | Expand data states | fit() | - | ? | INVESTIGATE |
| 3301 | `expand_data_for_phasetype()` | Full data expansion | fit() | - | ? | KEEP |
| 3435 | `needs_data_expansion_for_phasetype()` | Check if expansion needed | fit() | - | ? | KEEP |
| 3453 | `compute_expanded_subject_indices()` | Compute subj indices | fit() | - | ? | KEEP |
| 3546 | `build_phasetype_model()` | Build PhaseTypeModel | multistatemodel | Many | ? | REMOVE (no PhaseTypeModel) |
| 3642 | `phasetype_parameters_to_Q()` | Params → Q matrix | ? | - | ? | INVESTIGATE |

### 1.13 PhaseTypeModel Methods (Lines 3650-4600)

| Line | Function/Type | Purpose | Callers | Callees | Tests | Recommendation |
|------|--------------|---------|---------|---------|-------|----------------|
| 3694 | `set_crude_init!(::PhaseTypeModel)` | Init PT params | fit() | - | ? | REMOVE |
| 3737 | `_build_phasetype_structure_lookup()` | Structure lookup | set_crude_init! | - | ? | REMOVE |
| 3758 | `_get_crude_rate_for_expanded_hazard_structured()` | Get crude rate | set_crude_init! | - | ? | REMOVE |
| 3799 | `_identify_original_transition()` | Identify transition | set_crude_init! | - | ? | REMOVE |
| 3875 | `_sync_phasetype_parameters_to_original!()` | Sync params | fit() | - | ? | REMOVE |
| 3953 | `_collect_phasetype_params()` | Collect params | _sync_phasetype_parameters_to_original! | - | ? | REMOVE |
| 4012 | `_calculate_crude_phasetype()` | Calc crude PT | set_crude_init! | - | ? | REMOVE |
| 4045 | `_make_collapsed_markov_model()` | Make collapsed model | ? | - | ? | REMOVE |
| 4073 | `_transfer_phasetype_parameters!()` | Transfer params | fit() | - | ? | REMOVE |
| 4099 | `_init_phasetype_from_surrogate_paths!()` | Init from surrog | fit() | - | ? | REMOVE |
| 4173 | `initialize_parameters!(::PhaseTypeModel)` | Init params | User API | - | ? | REMOVE |
| 4205 | `initialize_parameters(::PhaseTypeModel)` | Init params | User API | - | ? | REMOVE |
| 4257 | `get_parameters(::PhaseTypeModel)` | Get params | User API | - | ? | REMOVE |
| 4289 | `get_expanded_parameters(::PhaseTypeModel)` | Get expanded params | User API | - | ? | REMOVE |
| 4321 | `get_parameters_flat(::PhaseTypeModel)` | Get flat params | Internal | - | ? | REMOVE |
| 4332 | `get_parameters_nested(::PhaseTypeModel)` | Get nested params | Internal | - | ? | REMOVE |
| 4343 | `get_parameters_natural(::PhaseTypeModel)` | Get natural params | Internal | - | ? | REMOVE |
| 4355 | `get_unflatten_fn(::PhaseTypeModel)` | Get unflatten | Internal | - | ? | REMOVE |
| 4374 | `set_parameters!(::PhaseTypeModel, ::Vector)` | Set params | User API | - | ? | REMOVE |
| 4404 | `set_parameters!(::PhaseTypeModel, ::NamedTuple)` | Set params | User API | - | ? | REMOVE |
| 4433 | `_expand_phasetype_params()` | Expand params | set_parameters! | - | ? | REMOVE |
| 4502 | `_rebuild_original_params_from_values!()` | Rebuild params | set_parameters! | - | ? | REMOVE |

---

## Summary: phasetype.jl

**Decision: Remove `PhaseTypeModel`, internal expansion/collapsing**

| Category | Count | Lines (est) | Action |
|----------|-------|-------------|--------|
| KEEP | ~15 | ~600 | Core expansion/collapse logic |
| REMOVE | ~45 | ~2,800 | PhaseTypeModel methods (type deleted) |
| MIGRATE | ~10 | ~400 | Move to surrogate/, inference/ |
| SIMPLIFY | ~9 | ~400 | PhaseTypeSurrogate → simpler storage |

### Functions to KEEP (core expansion machinery)
- `build_phasetype_mappings()` - Build state space mappings
- `expand_hazards_for_phasetype()` - PT specs → expanded Markov hazards
- `_build_progression_hazard()` - Build λ hazard
- `_build_exit_hazard()` - Build μ hazard
- `_build_expanded_hazard()` - Non-PT hazard in expanded space
- `_adjust_hazard_states()` - Adjust state indices
- `expand_data_for_phasetype_fitting()` - Expand data
- `map_observation_to_phases()` - Map obs to phases
- `collapse_phases()` - Collapse path to observed (in PhaseTypeSurrogate section)
- `expand_initial_state()` - Map initial state to phases

### Functions to REMOVE (PhaseTypeModel deleted)
All ~45 functions with `PhaseTypeModel` in signature:
- `fit(::PhaseTypeModel)` - Use `fit(::MultistateModel)` 
- `simulate(::PhaseTypeModel)` - Use standard simulate
- `get_parameters(::PhaseTypeModel)` - Standard accessor with collapse
- `set_parameters!(::PhaseTypeModel)` - Standard setter
- `set_crude_init!(::PhaseTypeModel)` - Use standard init
- `initialize_parameters!(::PhaseTypeModel)` - Use standard init
- `_sync_phasetype_parameters_to_original!()` - No dual storage
- `_collect_phasetype_params()` - No dual storage
- `_calculate_crude_phasetype()` - Use standard crude init
- `_make_collapsed_markov_model()` - No separate collapsed model
- `_transfer_phasetype_parameters!()` - No dual storage
- `_init_phasetype_from_surrogate_paths!()` - Use standard init
- `_build_original_parameters()` - No dual storage
- `_build_phasetype_model_from_hazards()` - Returns MultistateModel instead
- `build_phasetype_model()` - Returns MultistateModel instead
- Plus all parameter getters/setters specific to PhaseTypeModel

### Functions to MIGRATE (to surrogate/)
- `build_phasetype_surrogate()` 
- `update_expanded_Q()`
- `build_expanded_Q()`
- `build_coxian_intensity()`
- `_build_default_phasetype()`
- `_compute_default_n_phases()`

### Structs Decision
| Struct | Action | Rationale |
|--------|--------|-----------|
| `ProposalConfig` | KEEP | General (used for Markov too) |
| `PhaseTypeDistribution` | SIMPLIFY | Reduce to just n_phases + rates |
| `PhaseTypeConfig` | KEEP | Configuration for expansion |
| `PhaseTypeMappings` | MERGE | Into `PhaseTypeExpansion` |
| `PhaseTypeSurrogate` | KEEP | For MCEM importance sampling |

---

## 2. common.jl (1,602 lines)

### 2.1 Abstract Types Hierarchy

```
MultistateProcess
├── MultistateMarkovProcess
│   ├── MultistateMarkovModel
│   ├── MultistateMarkovModelCensored
│   └── PhaseTypeModel  ← TO BE REMOVED
└── MultistateSemiMarkovProcess
    ├── MultistateSemiMarkovModel
    └── MultistateSemiMarkovModelCensored

HazardFunction
├── ParametricHazard
├── SplineHazard
└── PhaseTypeHazardSpec  ← KEEP (user specification)

_Hazard
├── _MarkovHazard
│   ├── MarkovHazard
│   └── PhaseTypeCoxianHazard  ← KEEP (expanded state space)
└── _SemiMarkovHazard
    ├── SemiMarkovHazard
    └── _SplineHazard
        └── RuntimeSplineHazard
```

### 2.2 Model Structs to Consolidate

| Struct | Lines | Parent | Action |
|--------|-------|--------|--------|
| `MultistateModel` | ~25 | `MultistateProcess` | **KEEP as base** |
| `MultistateMarkovModel` | ~25 | `MultistateMarkovProcess` | REMOVE |
| `MultistateMarkovModelCensored` | ~25 | `MultistateMarkovProcess` | REMOVE |
| `MultistateSemiMarkovModel` | ~25 | `MultistateSemiMarkovProcess` | REMOVE |
| `MultistateSemiMarkovModelCensored` | ~25 | `MultistateSemiMarkovProcess` | REMOVE |
| `PhaseTypeModel` | ~50 | `MultistateMarkovProcess` | **REMOVE** (internal expansion) |
| `MultistateModelFitted` | ~30 | `MultistateProcess` | KEEP (holds fit results) |

**After consolidation:** 
- 1 unfitted struct: `MultistateModel` with optional `phasetype_expansion` field
- 1 fitted struct: `MultistateModelFitted`
- Traits for behavior: `is_markov()`, `has_censoring()`, `has_phasetype_expansion()`

### 2.3 Hazard Types

| Type | Purpose | Action |
|------|---------|--------|
| `MarkovHazard` | Exponential hazards | KEEP |
| `SemiMarkovHazard` | Weibull/Gompertz | KEEP |
| `RuntimeSplineHazard` | Spline hazards | KEEP |
| `PhaseTypeCoxianHazard` | Expanded PT hazard | KEEP (used in expansion) |
| `PhaseTypeHazardSpec` | User PT specification | KEEP |

### 2.4 Other Structs

| Struct | Purpose | Action |
|--------|---------|--------|
| `HazardMetadata` | Tang/linpred config | KEEP |
| `TimeTransformCache` | Tang caching | KEEP |
| `SharedBaselineTable` | Tang shared baselines | KEEP |
| `_TotalHazardAbsorbing` | Zero hazard | KEEP |
| `_TotalHazardTransient` | Sum of cause-specific | KEEP |
| `MarkovSurrogate` | MCEM surrogate | KEEP |
| `SurrogateControl` | Surrogate fitting | KEEP |
| `SamplePath` | Path storage | KEEP |
| `ExactData` | Exact obs data | KEEP |

### 2.5 Proposed New Struct: PhaseTypeExpansion

```julia
"""
    PhaseTypeExpansion

Metadata for models with phase-type hazards, enabling automatic
expansion/collapsing. Stored in MultistateModel.phasetype_expansion
when model has any :pt hazards.
"""
struct PhaseTypeExpansion
    n_phases_per_state::Vector{Int}      # [1, 3, 1] = state 2 has 3 phases
    state_to_phases::Vector{UnitRange{Int}}  # observed → expanded indices
    phase_to_state::Vector{Int}          # expanded → observed state
    original_tmat::Matrix{Int}           # Original transition matrix
    original_hazard_specs::Vector{HazardFunction}  # Original :pt specs
    original_n_states::Int               # For boundary checking
end
```

---

## 3. likelihoods.jl (2,190 lines)

**No PhaseTypeModel-specific code found.** ✅

All likelihood computation is generic - `loglik_markov`, `loglik_path_exact`, etc.
PhaseTypeModel uses these via dispatch (model <: MultistateMarkovProcess).

---

## 4. sampling.jl (2,234 lines)

### PhaseType-Related Code

| Location | Function | Purpose | Action |
|----------|----------|---------|--------|
| L1430-1475 | `build_phasetype_tpm_book()` | Build TPM for expanded space | **KEEP** (surrogate) |
| L1477-1588 | `build_phasetype_emat_expanded()` | Build emat for FFBS | **KEEP** (surrogate) |
| L1590-1640 | `build_fbmats_phasetype()` | Build FB matrices | **KEEP** (surrogate) |
| L1645-1678 | `expand_emat()` | Expand emission matrix | **KEEP** (surrogate) |
| L1680-1736 | `BackwardSampling_expanded()` | BS for expanded space | **KEEP** (surrogate) |
| L1738-1920 | `draw_samplepath_phasetype()` | Draw path via FFBS | **KEEP** (surrogate) |
| L2023-2100 | `collapse_phasetype_path()` | Collapse path to observed | **KEEP** (utility) |
| L2070-2180 | `loglik_phasetype_expanded_deprecated()` | PT loglik | **REPLACE** |
| L2123-2180 | `loglik_phasetype_path_deprecated()` | Path loglik | **REPLACE** |

**Summary:** ~800 lines for PhaseTypeSurrogate (MCEM). Keep all. ~100 lines deprecated → replace.

---

## 5. modelfitting.jl (1,932 lines)

### PhaseTypeModel-Specific Code

| Location | Function | Lines | Action |
|----------|----------|-------|--------|
| L300-550 | `fit(::PhaseTypeModel)` | ~250 | **REMOVE** |

**Summary:** With internal expansion, `fit()` dispatches on `MultistateModel` with trait check.

---

## 6. simulation.jl (1,290 lines)

### PhaseTypeModel-Specific Code

| Location | Function | Lines | Action |
|----------|----------|-------|--------|
| L903-967 | `simulate(::PhaseTypeModel)` | ~65 | **REMOVE** |
| L971-996 | `simulate_data(::PhaseTypeModel)` | ~25 | **REMOVE** |
| L997-1022 | `simulate_paths(::PhaseTypeModel)` | ~25 | **REMOVE** |
| L1023-1062 | `simulate_path(::PhaseTypeModel)` | ~40 | **REMOVE** |
| L1065-1130 | `_simulate_phasetype_internal()` | ~65 | **REMOVE** |
| L1132-1200 | `_collapse_simulation_result()` | ~70 | **MOVE** to utils |
| L1202-1240 | `_collapse_path()` | ~40 | **MOVE** to utils |

**Summary:** ~330 lines to remove/refactor. Collapsing utilities kept.

---

## 7. modeloutput.jl (1,202 lines)

### PhaseTypeModel-Specific Code

| Location | Function | Lines | Action |
|----------|----------|-------|--------|
| L1125-1200 | `Base.show(::PhaseTypeModel)` | ~75 | **REMOVE** |

---

## 8. crossvalidation.jl (2,432 lines)

### PhaseTypeModel-Specific Code

| Location | Function | Lines | Action |
|----------|----------|-------|--------|
| L132 | `compute_subject_gradients(...PhaseTypeModel...)` | Union dispatch | **UPDATE** |
| L289 | `compute_subject_hessians(...PhaseTypeModel...)` | Union dispatch | **UPDATE** |

**Summary:** Just update Union types to include `MultistateModel`.

---

## TOTAL REMOVAL ESTIMATE

| Source | Lines | Category |
|--------|-------|----------|
| phasetype.jl PhaseTypeModel methods | ~1,400 | Remove |
| common.jl model structs | ~150 | Remove (5 structs) |
| modelfitting.jl fit(::PhaseTypeModel) | ~250 | Remove |
| simulation.jl PT methods | ~330 | Remove/refactor |
| modeloutput.jl show methods | ~75 | Remove |
| **Total removable** | **~2,200** | - |

---

## REFACTORING SUMMARY

### Phase 1: Remove PhaseTypeModel
1. Delete `PhaseTypeModel` struct from common.jl
2. Delete `fit(::PhaseTypeModel)` from modelfitting.jl
3. Delete `simulate(::PhaseTypeModel)` family from simulation.jl
4. Delete all phasetype.jl functions with `::PhaseTypeModel` signature
5. Update exports in MultistateModels.jl

### Phase 2: Add Internal Expansion
1. Add `PhaseTypeExpansion` struct to common.jl
2. Add `phasetype_expansion::Union{Nothing, PhaseTypeExpansion}` to `MultistateModel`
3. Add trait: `has_phasetype_expansion(m) = !isnothing(m.phasetype_expansion)`
4. Update `multistatemodel()` to build expansion metadata for :pt hazards
5. Add automatic collapsing in `get_parameters`, `simulate`, etc.

### Phase 3: Consolidate Model Structs  
1. Keep `MultistateModel` as single unfitted struct
2. Remove 5 redundant model structs
3. Add traits: `is_markov()`, `has_censoring()`
4. Update all dispatch sites

### Phase 4: Organize into Subfolders
(Per subfolder plan in streamlining_plan.md)
