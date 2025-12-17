# Package Streamlining Plan

**Branch:** `package_streamlining`  
**Started:** 2025-12-16  
**Last Updated:** 2025-12-18

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
| 2025-12-16 | Struct consolidation | Eliminated 5 model structs → unified `MultistateModel` |
| 2025-12-16 | Trait-based dispatch | Added `is_markov()`, `is_panel_data()`, `has_phasetype_expansion()` |
| 2025-12-16 | PhaseTypeModel removal | Replaced with `phasetype_expansion` metadata field |
| 2025-12-16 | Remove *Censored variants | `MultistateMarkovModelCensored`, `MultistateSemiMarkovModelCensored` eliminated |
| 2025-12-16 | Phase-type accessor fixes | Updated accessors to use `phasetype_expansion` metadata |
| 2025-12-16 | `expanded` kwarg | Added to `simulate`, `simulate_paths`, `simulate_data` |
| 2025-12-16 | Interaction term fix | `extract_covariates_lightweight` now handles interaction terms |
| 2025-12-16 | Phase-type method validation | `initialize_parameters!` rejects `:markov` for PT models |
| 2025-12-17 | File reorganization | Moved 14 files into logical subfolders (commit 3a25ba2) |
| 2025-12-17 | types/ split | Split common.jl into 7 files in types/ subfolder |
| 2025-12-17 | hazard/ split | Split hazards.jl into 7 new files in hazard/ subfolder |
| 2025-12-18 | utilities/ split | Split helpers.jl (1,966 lines) into 6 files in utilities/ |
| 2025-12-18 | Deleted old files | Removed common.jl, hazards.jl, helpers.jl (5,292 lines total) |
| 2025-12-18 | phasetype/ docstring trim | Removed ~1,175 lines of obsolete docstrings (commit 7538ab5) |
| 2025-12-18 | phasetype/ file split | Split phasetype.jl into types.jl, surrogate.jl, expansion.jl (commit 4ac0fbd) |
| 2025-12-18 | phasetype/ consolidation | Eliminated duplicate functions using existing helpers (commit e0461ce) |

---

## CURRENT SOURCE STRUCTURE (after reorganization)

```
src/
├── MultistateModels.jl          # Main module (275 lines)
├── types/                        # Type definitions (7 files)
│   ├── abstract.jl              # Abstract type hierarchy
│   ├── data_containers.jl       # SamplePath, ExactData, MPanelData
│   ├── hazard_metadata.jl       # HazardMetadata, HazardCache
│   ├── hazard_specs.jl          # HazardFunction, Hazard user API
│   ├── hazard_structs.jl        # Internal _Hazard types
│   ├── infrastructure.jl        # ADBackend, ThreadingConfig
│   └── model_structs.jl         # MultistateModel, MultistateModelFitted
├── hazard/                       # Hazard functions (9 files)
│   ├── api.jl                   # User-facing: compute_hazard, cumulative_incidence
│   ├── covariates.jl            # Covariate extraction, linear predictor
│   ├── evaluation.jl            # eval_hazard, eval_cumhaz
│   ├── generators.jl            # Runtime code generation
│   ├── macros.jl                # @hazard macro
│   ├── spline.jl                # Spline hazard functions
│   ├── time_transform.jl        # Time transform optimizations
│   ├── total_hazard.jl          # Total hazard computation
│   └── tpm.jl                   # Transition probability matrices
├── utilities/                    # Utility functions (9 files)
│   ├── books.jl                 # TPM bookkeeping, data containers
│   ├── flatten.jl               # FlattenTypes, construct_flatten
│   ├── initialization.jl        # Crude parameter initialization
│   ├── misc.jl                  # Miscellaneous utilities
│   ├── parameters.jl            # unflatten, set_parameters!, get_hazard_params
│   ├── reconstructor.jl         # ReConstructor struct and API
│   ├── stats.jl                 # Statistical utilities
│   ├── transforms.jl            # Parameter scale transformations
│   └── validation.jl            # Data validation (check_data!, etc.)
├── construction/                 # Model building
│   └── multistatemodel.jl       # multistatemodel() entry point
├── inference/                    # Model fitting
│   ├── fit.jl                   # fit() entry points
│   ├── mcem.jl                  # MCEM algorithm
│   ├── sampling.jl              # Path sampling
│   └── sir.jl                   # SIR resampling
├── likelihood/                   # Likelihood computation
│   └── loglik.jl                # All likelihood functions
├── output/                       # Post-fit operations
│   ├── accessors.jl             # get_parameters, get_vcov, etc.
│   └── variance.jl              # Variance estimation
├── phasetype/                    # Phase-type distributions (3 files, 2,443 lines)
│   ├── types.jl                 # ProposalConfig, PhaseTypeDistribution, etc. (365)
│   ├── surrogate.jl             # Coxian construction, surrogate building (288)
│   └── expansion.jl             # State space expansion, model building (1,790)
├── simulation/                   # Simulation
│   ├── path_utilities.jl        # SamplePath operations
│   └── simulate.jl              # simulate(), simulate_paths()
└── surrogate/                    # Importance sampling surrogates
    └── markov.jl                # Markov surrogate
```

---

## REMAINING WORK

### Phase 3.1: Phase-Type Code Reduction (DEFERRED)

**Goal:** Reduce `phasetype/expansion.jl` from 3,622 lines to ~500 lines.

**Completed:**
- ✅ PhaseTypeModel struct eliminated
- ✅ Phase-type expansion metadata integrated into MultistateModel
- ✅ Accessors work with unified model struct
- ✅ Simulation functions support `expanded` kwarg

**Deferred:**
- ⏸️ Remove ~50 redundant PT-specific functions - **requires migration, not deletion**
  - `compute_phasetype_marginal_loglik_deprecated` still called in `inference/fit.jl:879`
  - `loglik_phasetype_path_deprecated` still called in `inference/sampling.jl:289`
  - These need to be replaced with calls to standard `loglik_markov` on expanded data

### Phase 3.2: File Reorganization ✅ COMPLETE (commit 3a25ba2)

The subfolder reorganization plan is documented below but has not been implemented.
Current files remain in flat `src/` structure.

**Deferred until:** Core functionality validated on main branch

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

**Decision: Full consolidation to single struct (Option C)**

```julia
# Single model struct with behavior determined by content
mutable struct MultistateModel <: MultistateProcess
    data::DataFrame
    parameters::NamedTuple
    hazards::Vector{<:_Hazard}
    totalhazards::Vector{_TotalHazard}
    tmat::Matrix{Int64}
    emat::Matrix{Float64}
    hazkeys::Dict{Symbol, Int64}
    subjectindices::Vector{Vector{Int64}}
    SubjectWeights::Vector{Float64}
    ObservationWeights::Union{Nothing, Vector{Float64}}
    CensoringPatterns::Matrix{Float64}
    markovsurrogate::Union{Nothing, MarkovSurrogate}
    modelcall::NamedTuple
end

# Traits for dispatch (computed from content)
is_markov(m::MultistateModel) = all(h -> h isa _MarkovHazard, m.hazards)
is_panel_data(m::MultistateModel) = m.modelcall.obstype == :panel
has_censoring(m::MultistateModel) = any(0 .< m.emat .< 1)  # Partial observability

# Fitted model retains separate struct (has additional fields)
mutable struct MultistateModelFitted <: MultistateProcess
    # ... existing fields plus loglik, vcov, etc.
end
```

**Implementation plan:**
1. Remove `MultistateMarkovModel`, `MultistateMarkovModelCensored`
2. Remove `MultistateSemiMarkovModel`, `MultistateSemiMarkovModelCensored`  
3. Remove `PhaseTypeModel` (PT becomes preprocessing)
4. Update all dispatch to use traits or `MultistateModel`
5. Keep `MultistateModelFitted` (different purpose - holds results)

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

| Location | Description | Status |
|----------|-------------|--------|
| `phasetype.jl:2883` | `compute_phasetype_marginal_loglik_deprecated` | **STILL IN USE** in modelfitting.jl:1177 |
| `sampling.jl:2070` | `loglik_phasetype_expanded_deprecated` | **STILL IN USE** in tests |
| `sampling.jl:2123` | `loglik_phasetype_path_deprecated` | **STILL IN USE** in sampling.jl:289 |

**Key Insight:** These functions are marked "deprecated" aspirationally - the intent was to migrate to standard Markov routines, but the migration never happened. They are production code.

**Action:** As part of aggressive PT simplification, these should be **replaced** not removed. The replacement is exactly what the deprecation notice says: use standard `loglik_markov` on expanded data. This is the core of the PT simplification work.

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

**Goal:** Reduce `phasetype.jl` from 4,586 lines to ~500 lines.

**Conceptual change:** Phase-type models ARE just Markov models on an expanded state space. The PT "specialization" should be reduced to:
1. **Expansion** - Convert PT hazard specs → expanded Markov model
2. **Collapse** - Map expanded results back to observed states
3. **Nothing else** - All likelihood, fitting, simulation uses standard Markov code

#### Current PT Structs (5 total)

| Struct | Lines | Action | Rationale |
|--------|-------|--------|-----------|
| `ProposalConfig` | ~100 | KEEP | General (used for Markov too) |
| `PhaseTypeDistribution` | ~50 | REMOVE | Math formulas not needed with expanded Markov |
| `PhaseTypeConfig` | ~80 | KEEP | Configuration for expansion |
| `PhaseTypeMappings` | ~100 | KEEP | Essential for expand/collapse |
| `PhaseTypeSurrogate` | ~100 | REPLACE | Use `MarkovSurrogate` on expanded space |

#### Current PT Functions (79 total) - Categorization

**KEEP (~10 functions, ~400 lines) - Core expansion/collapse:**
- `build_phasetype_mappings` - Build state space mappings
- `expand_hazards_for_phasetype` - Convert PT hazards to expanded Markov hazards
- `expand_data_for_phasetype_fitting` - Expand data to phase-level
- `collapse_phases` - Collapse expanded path to observed states
- `_build_expanded_tmat` - Build expanded transition matrix
- `_build_phase_censoring_patterns` - Handle censoring in expansion
- `has_phasetype_hazards` - Detection helper
- `needs_phasetype_proposal` - Proposal selection
- `resolve_proposal_config` - Proposal configuration

**REMOVE (~50 functions, ~3000 lines) - Redundant with standard Markov:**
- `compute_phasetype_marginal_loglik_deprecated` - Use `loglik_markov` instead
- `loglik_phasetype_expanded_deprecated` - Use `loglik` instead
- `loglik_phasetype_path_deprecated` - Use `loglik` instead
- `build_phasetype_surrogate` - Use `MarkovSurrogate` on expanded model
- `build_expanded_Q` - Q is just the intensity matrix of expanded Markov
- `update_expanded_Q` - Standard parameter update
- `build_phasetype_emission_matrix*` - Use standard emission matrix on expanded states
- `build_phasetype_model` - No separate model type
- All `PhaseTypeModel` accessor methods - No separate type
- All `PhaseTypeModel` parameter methods - No separate type

**MOVE (~19 functions, ~800 lines) - To other modules:**
- `ProposalConfig`, `MarkovProposal`, `PhaseTypeProposal` → `surrogate/`
- `PhaseTypeDistribution` methods → DELETE (not needed)
- Constraint generation → `inference/optimization.jl`

#### Migration Strategy

1. **First:** Make PT models return `MultistateMarkovModel` with expanded state space
2. **Then:** Update all callers to use standard Markov routines
3. **Then:** Remove redundant PT-specific functions
4. **Finally:** Consolidate remaining expansion code to `construction/phasetype_expansion.jl`

**Question for user:** Before I start implementing, do you want me to:
- A) Do a detailed function-by-function audit first, OR
- B) Start with the structural changes (folder creation, model consolidation) and audit as we go?

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
│   ├── models.jl                # Model types (MultistateModel - SINGLE struct)
│   ├── data.jl                  # Data wrapper types (ExactData, MPanelData, etc.)
│   └── configuration.jl         # Config types (ADBackend, ThreadingConfig, etc.)
│
├── hazard/                       # Hazard specification and evaluation
│   ├── specification.jl         # Hazard() constructor, @hazard macro
│   ├── exponential.jl           # Exp/Weibull/Gompertz hazard evaluation
│   ├── spline.jl                # RuntimeSplineHazard evaluation
│   └── total_hazard.jl          # Total hazard computation
│
├── construction/                 # Model building
│   ├── multistatemodel.jl       # multistatemodel() main entry point
│   ├── phasetype_expansion.jl   # PT hazard → expanded Markov conversion (~500 lines)
│   └── data_processing.jl       # Data validation, index building
│
├── likelihood/                   # Likelihood computation
│   ├── exact.jl                 # Exact observation likelihood
│   ├── panel_markov.jl          # Panel Markov likelihood (matrix exp)
│   ├── panel_semimarkov.jl      # Panel semi-Markov likelihood (MCEM)
│   └── helpers.jl               # Shared likelihood utilities
│
├── surrogate/                    # Importance sampling surrogates
│   ├── markov.jl                # Markov surrogate for path sampling
│   └── fitting.jl               # Surrogate parameter fitting
│
├── inference/                    # Model fitting
│   ├── fit.jl                   # fit() entry points
│   ├── optimization.jl          # Optimizer setup, constraints
│   ├── mcem.jl                  # MCEM algorithm
│   ├── sampling.jl              # Path sampling
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
| `hazards.jl` | 1,762 | Split: `hazard/*.jl`, `output/predictions.jl` |
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
| `smooths.jl` | 928 | `hazard/spline.jl` |
| `crossvalidation.jl` | 2,432 | `output/variance.jl` |
| `macros.jl` | 110 | `hazard/specification.jl` |
| `sir.jl` | 257 | `inference/sir.jl` |
| `statsutils.jl` | 44 | `utilities/stats.jl` |
| `surrogates.jl` | 687 | `surrogate/markov.jl`, `surrogate/fitting.jl` |

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
5. **Move hazards** (`hazard/`) - depends on types
6. **Move surrogates** (`surrogate/`) - depends on types
7. **Move construction** (`construction/`) - depends on hazards, types
8. **Move likelihood** (`likelihood/`) - depends on types, hazards
9. **Move inference** (`inference/`) - depends on likelihood, surrogate
10. **Move output** (`output/`) - depends on inference
11. **Move simulation** (`simulation/`) - depends on types, hazards

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
| 2025-12-16 | Full struct consolidation | Single struct + traits (Option C) |
| 2025-12-16 | Singular folder names | `likelihood/`, `simulation/`, etc. |
| 2025-12-16 | PT target ~500 lines | Down from 4,586 - acceptable |
| 2025-12-16 | Surrogates separate folder | `surrogate/` subfolder |
| 2025-12-16 | **Remove PhaseTypeModel** | Internal expansion/collapsing - user never sees expanded states |

---

## PhaseTypeModel Removal Design (NEW)

### User Experience (Unchanged)
```julia
# User specifies PT hazard - nothing special required
h12 = Hazard(:pt, 1, 2)
model = multistatemodel(h12, h13; data=df, n_phases=Dict(1=>3))

# Returns MultistateMarkovModel (internally expanded)
# All operations transparent:
fitted = fit(model)                    # Fits on expanded, reports original params
sim = simulate(fitted; paths=true)     # Paths in observed states
get_parameters(fitted)                 # Original PT parameters (λ, μ)
```

### Internal Implementation
```julia
mutable struct MultistateModel <: MultistateProcess
    # ... standard fields ...
    
    # NEW: Optional expansion metadata (nothing for non-PT models)
    phasetype_expansion::Union{Nothing, PhaseTypeExpansion}
end

struct PhaseTypeExpansion
    n_phases_per_state::Vector{Int}      # [1, 3, 1] = state 2 has 3 phases
    state_to_phases::Vector{UnitRange{Int}}  # Mapping: observed → expanded
    phase_to_state::Vector{Int}              # Mapping: expanded → observed
    original_tmat::Matrix{Int}               # Original transition matrix
    original_hazard_specs::Vector{HazardFunction}  # Original PT specs
end

# Traits
has_phasetype_expansion(m::MultistateModel) = !isnothing(m.phasetype_expansion)
```

### What This Enables
1. `fit()` - Works on expanded space, stores original PT params in results
2. `simulate()` - Simulates expanded, collapses paths automatically
3. `get_parameters()` - Returns PT parameters (λ progression, μ exit), not expanded
4. `loglik()` - Just `loglik_markov` on expanded space
5. Accessors - All automatic collapsing

### Lines to Remove
- `PhaseTypeModel` struct definition (~30 lines)
- `fit(::PhaseTypeModel)` dispatch (~100 lines)
- `simulate(::PhaseTypeModel)` and 4 related methods (~200 lines)
- `get_parameters(::PhaseTypeModel)` family (~150 lines)
- `set_parameters!(::PhaseTypeModel)` family (~100 lines)
- `set_crude_init!(::PhaseTypeModel)` (~50 lines)
- `initialize_parameters!(::PhaseTypeModel)` (~100 lines)
- Pretty printing for PhaseTypeModel (~100 lines)
- Plus all internal helpers for dual parameter storage (~600 lines)

**Estimated savings: ~1,400 lines** (more conservative than earlier estimate)

---

## PENDING DECISIONS (Need User Input)

*None currently - proceeding with implementation*

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
- Categorized 79 phasetype.jl functions into KEEP/REMOVE/MOVE

**User decisions received:**
- Q1: Delete `types/` folder → YES (done)
- Q2: Model struct consolidation → FULL CONSOLIDATION (single struct + traits)
- Q3: Phase-type approach → AGGRESSIVE (preprocessing only)
- Q4: Subfolders → ORGANIZE with singular names, surrogates in separate folder

**Key Insights:**
- "Deprecated" PT functions are actually still in production use
- PT simplification requires migration, not just deletion
- 6 model structs → 1 struct + traits
- ~50 of 79 PT functions can be eliminated

**Next steps:**
1. Get user input: start with structure or detailed audit first?
2. Then proceed with implementation
