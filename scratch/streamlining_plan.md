# Package Streamlining Plan

**Branch:** `package_streamlining`  
**Started:** 2025-12-16  
**Last Updated:** 2025-12-16

## Guiding Principles

1. **Phase-type models ARE Markov models** - eliminate unnecessary specialization
2. **Breaking changes acceptable** - not a public package yet
3. **Audit before action** - no removals without understanding dependencies
4. **Regular communication** - seek clarification when uncertain

---

## COMPLETED ACTIONS

| Date | Action | Details |
|------|--------|---------|
| 2025-12-16 | Deleted `src/types/` | Orphaned draft code (~1,600 lines never integrated) |

---

## KEY FINDINGS (Updated 2025-12-16)

### Finding 1: ~~`types/` folder is completely orphaned~~ ✅ DELETED

### Finding 2: Phase-type design confirms your insight
The codebase already recognizes that PT models are Markov:
- `PhaseTypeModel <: MultistateMarkovProcess` 
- `PhaseTypeCoxianHazard <: _MarkovHazard`
- Docstring: "The expanded state space is Markovian"
- Deprecated PT likelihood says: "Use loglik_markov with expanded data instead"

**Decision:** Aggressive simplification - PT becomes preprocessing only.

### Finding 3: Multiple nearly-identical model structs (DETAILED ANALYSIS)

**Current 6 unfitted model structs:**

| Struct | Parent | hazards type | Unique fields | Usage |
|--------|--------|--------------|---------------|-------|
| `MultistateModel` | `MultistateProcess` | `Vector{_Hazard}` | none | Exact obs times, mixed hazards |
| `MultistateMarkovModel` | `MultistateMarkovProcess` | `Vector{_MarkovHazard}` | none | Panel data, Markov only |
| `MultistateMarkovModelCensored` | `MultistateMarkovProcess` | `Vector{_MarkovHazard}` | none | Panel + censoring |
| `MultistateSemiMarkovModel` | `MultistateSemiMarkovProcess` | `Vector{_Hazard}` | none | Panel, semi-Markov |
| `MultistateSemiMarkovModelCensored` | `MultistateSemiMarkovProcess` | `Vector{_Hazard}` | none | Panel + censoring |
| `PhaseTypeModel` | `MultistateMarkovProcess` | `Vector{_MarkovHazard}` | 5 extra fields | PT wrapper |

**Plus 1 fitted model struct:**
- `MultistateModelFitted` - has `loglik`, `vcov`, `ConvergenceRecords`, `ProposedPaths`

**Observations:**
1. The `*Censored` variants have IDENTICAL fields to their non-censored counterparts
2. All dispatch uses `Union{X, XCensored}` - proving they shouldn't be separate
3. The distinction is already encoded elsewhere (emission matrix `emat`)
4. `PhaseTypeModel` has 5 extra fields for the "original" space - these become unnecessary with aggressive PT simplification

---

## Q2 DETAILED: Model Struct Consolidation Options

### Option A: Single Parametric Struct

```julia
abstract type ModelClass end
struct MarkovClass <: ModelClass end
struct SemiMarkovClass <: ModelClass end

mutable struct MultistateModel{C<:ModelClass, H<:_Hazard} <: MultistateProcess
    data::DataFrame
    parameters::NamedTuple
    hazards::Vector{H}
    totalhazards::Vector{_TotalHazard}
    tmat::Matrix{Int64}
    emat::Matrix{Float64}
    # ... common fields
end

# Type aliases for clarity
const MarkovModel = MultistateModel{MarkovClass, _MarkovHazard}
const SemiMarkovModel = MultistateModel{SemiMarkovClass, _Hazard}
```

**Pros:**
- Single struct definition
- Eliminates 5 redundant struct definitions
- Type parameter preserves dispatch capability
- Censoring handled by `emat` content, not type

**Cons:**
- Requires updating all dispatch signatures
- Type parameters can complicate inference in some cases

### Option B: Two Structs (Markov + SemiMarkov)

```julia
mutable struct MultistateMarkovModel <: MultistateMarkovProcess
    # All common fields
    # Censoring determined by emat, not type
end

mutable struct MultistateSemiMarkovModel <: MultistateSemiMarkovProcess
    # All common fields
end
```

**Pros:**
- Cleaner than current (eliminates 4 structs)
- No type parameters
- Preserves existing abstract type hierarchy

**Cons:**
- Still some duplication (same fields in both)
- Keeps the Markov/SemiMarkov dichotomy in types

### Option C: Single Struct + Traits

```julia
mutable struct MultistateModel <: MultistateProcess
    # All common fields
end

# Traits for behavior
is_markov(m::MultistateModel) = all(h -> h isa _MarkovHazard, m.hazards)
is_panel_data(m::MultistateModel) = m.modelcall.obstype == :panel
has_censoring(m::MultistateModel) = !all(m.emat .∈ (0.0, 1.0))
```

**Pros:**
- Maximum simplicity - ONE struct
- Behavior determined by content, not type
- Most flexible for future extensions

**Cons:**
- Loses compile-time dispatch on model type
- Some methods may need runtime checks
- Breaks more existing code

---

## RECOMMENDED CONSOLIDATION PATH

Given your preference for aggressive simplification:

**Phase 1:** Eliminate `*Censored` variants immediately (Option B partial)
- Replace `MultistateMarkovModelCensored` → `MultistateMarkovModel`
- Replace `MultistateSemiMarkovModelCensored` → `MultistateSemiMarkovModel`
- Update all `Union{X, XCensored}` patterns to just `X`
- Result: 4 structs instead of 6

**Phase 2:** With aggressive PT simplification, `PhaseTypeModel` becomes unnecessary
- PT hazards expand to regular Markov hazards during construction
- Returns `MultistateMarkovModel` with expanded state space
- Result: 3 structs

**Phase 3:** Consider full consolidation to Option A or C
- This is a larger refactor, may want to stabilize first
- Could be a separate future branch

**Awaiting your input:** Which consolidation approach do you prefer?

---

## Phase 1: Audit & Documentation

### Status: IN PROGRESS

### 1.1 Source File Inventory (UPDATED)

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `MultistateModels.jl` | 222 | Main module, exports | To audit |
| `common.jl` | 1,601 | Type definitions | To audit |
| `hazards.jl` | 1,762 | Hazard functions | To audit |
| `helpers.jl` | 1,965 | Utility functions | To audit |
| `initialization.jl` | 549 | Parameter init | To audit |
| `likelihoods.jl` | 2,190 | Likelihood computation | To audit |
| `mcem.jl` | 273 | MCEM utilities | To audit |
| `miscellaneous.jl` | 65 | Misc utilities | To audit |
| `modelfitting.jl` | 1,931 | fit() and related | To audit |
| `modelgeneration.jl` | 1,230 | multistatemodel() | To audit |
| `modeloutput.jl` | 1,202 | Output/accessors | To audit |
| `pathfunctions.jl` | 403 | Path manipulation | To audit |
| `phasetype.jl` | 4,586 | Phase-type (PRIORITY) | To audit |
| `sampling.jl` | 2,233 | MCMC sampling | To audit |
| `simulation.jl` | 1,289 | simulate() | To audit |
| `smooths.jl` | 928 | Spline hazards | To audit |
| `crossvalidation.jl` | 2,432 | CV/robust variance | To audit |
| `macros.jl` | 110 | @hazard macro | To audit |
| `sir.jl` | 257 | SIR resampling | To audit |
| `statsutils.jl` | 44 | Stats utilities | To audit |
| `surrogates.jl` | 687 | Surrogate models | To audit |

~~**types/ subfolder:**~~ ✅ DELETED

**Total:** ~25,958 lines (after types/ deletion)

### 1.2 Empty Subfolders
See Subfolder Reorganization Plan below.

### 1.3 Export Analysis

Exported symbols (from MultistateModels.jl):
- [ ] Document each exported symbol
- [ ] Verify each has tests
- [ ] Verify each has docstrings

### 1.4 Known Deprecated Code

| Location | Description | Action |
|----------|-------------|--------|
| `phasetype.jl:2883` | Deprecated PT loglik functions | Remove |
| `sampling.jl:2070` | Deprecated loglik_expanded_phasetype | Remove |
| `sampling.jl:2123` | Deprecated loglik_collapsed_phasetype | Remove |

### 1.5 Validation Gaps

From `future_features_todo.txt`:
- ⚠️ SubjectWeights propagation through MCEM - NOT VALIDATED
- ⚠️ Emission matrices for censoring patterns - NOT VALIDATED

---

## Phase 2: Dead Code Removal

### Status: NOT STARTED

### 2.1 Deprecated Functions to Remove
(Populated after Phase 1)

### 2.2 Orphaned Functions
(Populated after Phase 1)

### 2.3 Commented-Out Code Blocks
(Populated after Phase 1)

---

## Phase 3: Consolidation & Reorganization

### Status: NOT STARTED

### 3.1 Phase-Type Simplification (PRIORITY)

**Key insight:** Phase-type models ARE Markov models. The phase-type expansion creates a larger Markov model. Unnecessary specialization should be eliminated.

Questions to resolve:
- What methods are truly PT-specific vs just Markov methods called on expanded data?
- Can PT-specific likelihood functions be replaced by standard Markov loglik on expanded data?
- What about the FFBS sampler - is it truly specialized or just forward-backward on the expanded state space?

### 3.2 File Reorganization Plan
See **Subfolder Reorganization Plan** section below.

---

## SUBFOLDER REORGANIZATION PLAN

### Design Philosophy

1. **Logical grouping by responsibility** - not by implementation detail
2. **Clear dependency hierarchy** - lower-level modules don't depend on higher-level
3. **User-facing vs internal** - separate public API from implementation
4. **Manageable file sizes** - aim for ~200-600 lines per file

### Proposed Structure

```
src/
├── MultistateModels.jl          # Main module: exports, includes
│
├── types/                        # Type definitions (rebuild this properly)
│   ├── abstract.jl              # Abstract types hierarchy
│   ├── hazards.jl               # Hazard types (HazardFunction, _Hazard, etc.)
│   ├── models.jl                # Model types (MultistateModel, etc.)
│   ├── data.jl                  # Data wrapper types (ExactData, MPanelData, etc.)
│   └── configuration.jl         # Config types (ADBackend, ThreadingConfig, etc.)
│
├── hazards/                      # Hazard specification and evaluation
│   ├── specification.jl         # Hazard() constructor, @hazard macro
│   ├── exponential.jl           # Exp/Weibull/Gompertz hazard evaluation
│   ├── spline.jl                # RuntimeSplineHazard evaluation
│   └── total_hazard.jl          # Total hazard computation
│
├── construction/                 # Model building
│   ├── multistatemodel.jl       # multistatemodel() main entry point
│   ├── phasetype_expansion.jl   # PT hazard → expanded Markov conversion
│   └── data_processing.jl       # Data validation, index building
│
├── likelihood/                   # Likelihood computation
│   ├── exact.jl                 # Exact observation likelihood
│   ├── panel_markov.jl          # Panel Markov likelihood (matrix exp)
│   ├── panel_semimarkov.jl      # Panel semi-Markov likelihood (MCEM)
│   └── helpers.jl               # Shared likelihood utilities
│
├── inference/                    # Model fitting
│   ├── fit.jl                   # fit() entry points
│   ├── optimization.jl          # Optimizer setup, constraints
│   ├── mcem.jl                  # MCEM algorithm
│   ├── sampling.jl              # Path sampling (Markov surrogate)
│   └── sir.jl                   # SIR resampling
│
├── output/                       # Post-fit operations
│   ├── accessors.jl             # get_parameters, get_vcov, etc.
│   ├── variance.jl              # Variance estimation (IJ, JK, robust)
│   ├── diagnostics.jl           # Model diagnostics, summaries
│   └── predictions.jl           # cumulative_incidence, etc.
│
├── simulation/                   # Simulation
│   ├── simulate.jl              # simulate(), simulate_paths()
│   ├── path_generation.jl       # draw_paths, path sampling logic
│   └── path_utilities.jl        # SamplePath operations
│
└── utilities/                    # Shared utilities
    ├── parameters.jl            # Parameter manipulation (set/get/flatten)
    ├── initialization.jl        # Crude parameter initialization
    └── stats.jl                 # Shared statistical utilities
```

### File Migration Map

| Current File | Lines | → New Location(s) |
|--------------|-------|-------------------|
| `common.jl` | 1,601 | Split: `types/*.jl` |
| `hazards.jl` | 1,762 | Split: `hazards/*.jl`, `output/predictions.jl` |
| `helpers.jl` | 1,965 | Split: `utilities/*.jl`, others as needed |
| `initialization.jl` | 549 | `utilities/initialization.jl` |
| `likelihoods.jl` | 2,190 | Split: `likelihood/*.jl` |
| `mcem.jl` | 273 | `inference/mcem.jl` |
| `miscellaneous.jl` | 65 | Merge into `utilities/` or delete |
| `modelfitting.jl` | 1,931 | Split: `inference/fit.jl`, `inference/optimization.jl` |
| `modelgeneration.jl` | 1,230 | Split: `construction/*.jl` |
| `modeloutput.jl` | 1,202 | Split: `output/*.jl` |
| `pathfunctions.jl` | 403 | `simulation/path_utilities.jl` |
| `phasetype.jl` | 4,586 | **AGGRESSIVE REDUCTION** → `construction/phasetype_expansion.jl` (~500 lines) |
| `sampling.jl` | 2,233 | Split: `inference/sampling.jl`, `simulation/path_generation.jl` |
| `simulation.jl` | 1,289 | Split: `simulation/*.jl` |
| `smooths.jl` | 928 | `hazards/spline.jl` |
| `crossvalidation.jl` | 2,432 | `output/variance.jl` |
| `macros.jl` | 110 | `hazards/specification.jl` |
| `sir.jl` | 257 | `inference/sir.jl` |
| `statsutils.jl` | 44 | `utilities/stats.jl` |
| `surrogates.jl` | 687 | Split: `inference/sampling.jl` (Markov), delete PT-specific |

### Empty Subfolder Disposition

| Existing Folder | Action |
|-----------------|--------|
| `src/accessors/` | DELETE - use `output/` instead |
| `src/construction/` | KEEP - use as planned |
| `src/evaluation/` | DELETE - use `likelihood/` instead |
| `src/inference/` | KEEP - use as planned |
| `src/initialization/` | DELETE - merge into `utilities/` |
| `src/phasetype/` | DELETE - PT becomes part of `construction/` |
| `src/simulation/` | KEEP - use as planned |
| `src/utilities/` | KEEP - use as planned |

### Implementation Order

1. **Delete deprecated code** (Phase 2) - reduces noise before reorganization
2. **Create new folder structure** - empty but ready
3. **Move types first** (`types/`) - foundational, few dependencies
4. **Move utilities** (`utilities/`) - low-level, used everywhere
5. **Move hazards** (`hazards/`) - depends on types
6. **Move construction** (`construction/`) - depends on hazards, types
7. **Move likelihood** (`likelihood/`) - depends on types, hazards
8. **Move inference** (`inference/`) - depends on likelihood
9. **Move output** (`output/`) - depends on inference
10. **Move simulation** (`simulation/`) - depends on types, hazards

### Questions Before Proceeding

1. **Folder naming:** Does `likelihood/` vs `likelihoods/` matter? Same for `simulation/` vs `simulations/`? I've used singular.

2. **Phase-type code:** With aggressive simplification, I'm proposing PT code shrinks from 4,586 lines to ~500 lines in `construction/phasetype_expansion.jl`. The rest should become unnecessary. Is this the right target?

3. **Surrogates:** The surrogate concept (Markov surrogate for importance sampling) - should this stay in `inference/` or get its own subfolder?

---

## Phase 4: Simplification

### Status: NOT STARTED

### 4.1 Unnecessary Method Specializations
(Populated after Phase 3)

### 4.2 Over-Complex Abstractions
(Populated after Phase 3)

---

## Phase 5: Validation Debt

### Status: NOT STARTED

### 5.1 Features to Validate
- SubjectWeights in MCEM
- Emission matrices

### 5.2 Features to Mark Experimental
(Populated after audit)

---

## Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-12-16 | Phase-type = Markov | PT models are Markov on expanded state space |
| 2025-12-16 | Breaking changes OK | Not a public package |
| 2025-12-16 | Delete `types/` folder | Orphaned draft code, user didn't know it existed |
| 2025-12-16 | Aggressive PT simplification | User preference - PT becomes preprocessing only |
| 2025-12-16 | Organize into subfolders | User wants organized structure, not flat |

---

## PENDING DECISIONS (Need User Input)

1. **Model struct consolidation:** Options A/B/C presented above - which approach?
2. **Subfolder naming:** Singular (`likelihood/`) vs plural (`likelihoods/`)?
3. **PT code target:** 4,586 → ~500 lines acceptable?
4. **Surrogates location:** `inference/` or separate folder?

---

## Session Notes

### 2025-12-16 Session 1

**Accomplished:**
- Created branch `package_streamlining`
- Initial exploration of codebase structure
- Identified 7 empty subfolders
- Found 3 deprecated code locations
- Created this tracking document
- Deleted orphaned `types/` folder (~1,600 lines)
- Detailed analysis of model struct consolidation options
- Developed comprehensive subfolder reorganization plan

**User decisions received:**
- Q1: Delete `types/` folder → YES (done)
- Q2: Model struct consolidation → NEEDS MORE DETAIL (provided)
- Q3: Phase-type approach → AGGRESSIVE (preprocessing only)
- Q4: Subfolders → ORGANIZE (plan developed)

**Next steps:**
1. Get user input on remaining decisions (model structs, naming, PT target)
2. Begin Phase 2: Dead code removal (deprecated functions)
3. Then proceed with reorganization per plan
