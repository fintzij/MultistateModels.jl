---
name: codebase-knowledge
description: Deep knowledge of MultistateModels.jl and MultistateModelsTests.jl codebase structure, conventions, and current state. READ THIS FIRST when starting any session involving code changes.
applyTo: '**'
---

# MultistateModels.jl Codebase Knowledge

**Read this skill file FIRST at the start of every session.** It provides the essential context needed to work effectively with this codebase.

**Last Updated**: 2026-01-08  
**Branch**: `penalized_splines` (active development)

---

## 1. Package Overview

**MultistateModels.jl** implements continuous-time multistate models for survival analysis:
- **Parametric hazards**: Exponential (`:exp`), Weibull (`:wei`), Gompertz (`:gom`)
- **Semi-parametric hazards**: B-splines (`:sp`) with optional monotonicity constraints
- **Phase-type hazards** (`:pt`): Coxian distributions for flexible sojourn time modeling
- **Observation types**: Exact (obstype=1), panel (obstype=2), censored (obstype≥3)
- **Covariate effects**: Proportional hazards (`:ph`), accelerated failure time (`:aft`)
- **Inference**: Direct MLE, matrix exponential MLE, Monte Carlo EM (MCEM)
- **Variance estimation**: Model-based, sandwich (IJ), jackknife

**MultistateModelsTests.jl** is a companion test package in `MultistateModelsTests/` containing:
- Unit tests (`unit/`)
- Integration tests (`integration/`)  
- Long-running statistical validation tests (`longtests/`)
- Test fixtures and infrastructure (`fixtures/`, `src/`)

---

## 2. Source Code Architecture

### Directory Structure (src/)

```
src/
├── MultistateModels.jl      # Module definition, exports, include order
├── types/                   # Type definitions (load FIRST)
│   ├── abstract.jl          # Abstract type hierarchy
│   ├── hazard_metadata.jl   # HazardMetadata, Tang caching types
│   ├── hazard_structs.jl    # Internal: MarkovHazard, SemiMarkovHazard, RuntimeSplineHazard
│   ├── hazard_specs.jl      # User-facing: ParametricHazard, SplineHazard, PhaseTypeHazard
│   ├── model_structs.jl     # MultistateModel, MultistateModelFitted, surrogates
│   ├── data_containers.jl   # SamplePath, ExactData, MPanelData
│   └── infrastructure.jl    # ADBackend, PenaltyConfig, compute_penalty()
├── construction/            # Model building pipeline
│   ├── multistatemodel.jl   # Entry point: multistatemodel()
│   ├── hazard_constructors.jl  # User-facing Hazard() constructor
│   ├── hazard_builders.jl   # Registry pattern, parametric builders
│   ├── spline_builder.jl    # Spline hazard construction
│   └── model_assembly.jl    # build_hazards(), build_parameters()
├── hazard/                  # Hazard evaluation
│   ├── api.jl               # compute_hazard(), cumulative_incidence()
│   ├── covariates.jl        # Covariate extraction, linear predictors
│   ├── evaluation.jl        # eval_hazard(), eval_cumhaz()
│   ├── generators.jl        # Runtime code generation for hazards
│   ├── smooth_terms.jl      # s(), te() smooth term parsing
│   ├── spline.jl            # B-spline basis, calibrate_splines()
│   ├── time_transform.jl    # Tang optimization (shared baseline caching)
│   ├── total_hazard.jl      # Total hazard per state
│   └── tpm.jl               # Transition probability matrices
├── likelihood/              # Log-likelihood computation
│   ├── loglik_utils.jl      # ForwardDiff helpers, parameter prep
│   ├── loglik_batched.jl    # Batched hazard-centric infrastructure
│   ├── loglik_exact.jl      # Exact data: loglik_exact(), loglik_exact_penalized()
│   ├── loglik_markov.jl     # Panel + Markov: matrix exponential
│   ├── loglik_markov_functional.jl  # Reverse-mode AD compatible
│   └── loglik_semi_markov.jl  # Semi-Markov MCEM path-based
├── inference/               # Model fitting
│   ├── fit_common.jl        # fit() entry point, dispatch logic
│   ├── fit_exact.jl         # _fit_exact() for exact data
│   ├── fit_markov.jl        # _fit_markov_panel() for panel + Markov
│   ├── fit_mcem.jl          # _fit_mcem() for panel + semi-Markov
│   ├── mcem.jl              # MCEM algorithm core
│   ├── sampling.jl          # Path sampling for MCEM
│   ├── sir.jl               # Sampling importance resampling
│   └── smoothing_selection.jl  # PIJCV, EFS, PERF λ selection
├── phasetype/               # Phase-type expansion
│   ├── types.jl             # PhaseTypeMappings, PhaseTypeDistribution
│   ├── surrogate.jl         # PhaseTypeSurrogate
│   ├── expansion_*.jl       # State space expansion machinery
│   └── expansion_ffbs_data.jl  # FFBS data preparation
├── simulation/              # Path simulation
│   ├── simulate.jl          # simulate(), simulate_paths()
│   └── path_utilities.jl    # path_to_dataframe(), draw_paths()
├── surrogate/               # Importance sampling surrogates
│   └── markov.jl            # MarkovSurrogate for MCEM proposals
├── output/                  # Results extraction
│   ├── accessors.jl         # get_parameters(), get_vcov(), get_loglik()
│   └── variance.jl          # IJ, JK variance, cross-validation
└── utilities/               # Support functions
    ├── constants.jl         # Numerical tolerances (LOADED EARLY)
    ├── flatten.jl           # Parameter flattening type system
    ├── reconstructor.jl     # ReConstructor for flatten/unflatten
    ├── parameters.jl        # set_parameters!, get_parameters
    ├── transforms.jl        # Estimation ↔ natural scale (DEPRECATED)
    ├── bounds.jl            # Parameter bounds for optimization
    ├── spline_utils.jl      # Knot placement, penalty matrices
    ├── penalty_config.jl    # build_penalty_config()
    └── validation.jl        # Input validation
```

### Key Entry Points

| User Action | Entry Point | Dispatches To |
|-------------|-------------|---------------|
| Create model | `multistatemodel(h1, h2; data=df)` | `construction/multistatemodel.jl` |
| Define hazard | `Hazard(@formula(0~x), :wei, 1, 2)` | `construction/hazard_constructors.jl` |
| Fit model | `fit(model)` | `_fit_exact`, `_fit_markov_panel`, or `_fit_mcem` |
| Simulate | `simulate(model; tmax=10)` | `simulation/simulate.jl` |
| Get results | `get_parameters(fitted)` | `output/accessors.jl` |

---

## 3. Type System Map

### Abstract Type Hierarchy

```
HazardFunction (user-facing, abstract)
├── ParametricHazard     # :exp, :wei, :gom
├── SplineHazard         # :sp
└── PhaseTypeHazard      # :pt

_Hazard (internal, abstract)
├── _MarkovHazard
│   ├── MarkovHazard           # Runtime :exp
│   └── PhaseTypeCoxianHazard  # Expanded phase-type
└── _SemiMarkovHazard
    ├── SemiMarkovHazard       # Runtime :wei, :gom
    └── RuntimeSplineHazard    # Runtime :sp (can be Markov or semi-Markov)

MultistateProcess (abstract)
└── MultistateModel (concrete, mutable)
    └── MultistateModelFitted (concrete, mutable)

AbstractSurrogate (abstract)
├── MarkovSurrogate      # Exponential surrogate for MCEM
└── PhaseTypeSurrogate   # Phase-type FFBS surrogate
```

### Key Struct Fields

**MultistateModel** (unfitted):
```julia
struct MultistateModel <: MultistateProcess
    data::DataFrame
    parameters::NamedTuple       # (flat, nested, natural, reconstructor)
    hazards::Vector{<:_Hazard}
    totalhazards::Vector{_TotalHazard}
    tmat::Matrix{Int64}          # Transition matrix
    emat::Matrix{Float64}        # Emission matrix (censoring)
    hazkeys::Dict{Symbol,Int64}  # :h12 → index
    subjectindices::Vector{Vector{Int64}}
    markovsurrogate::Union{Nothing,MarkovSurrogate}
    phasetype_expansion::Union{Nothing,PhaseTypeExpansion}
    # ... weights, censoring patterns, modelcall
end
```

**MultistateModelFitted** adds:
```julia
    loglik::NamedTuple
    vcov::Union{Nothing,Matrix{Float64}}
    ij_vcov::Union{Nothing,Matrix{Float64}}
    jk_vcov::Union{Nothing,Matrix{Float64}}
    subject_gradients::Union{Nothing,Matrix{Float64}}
    ConvergenceRecords::...
    ProposedPaths::...
```

### Parameter Representations

```julia
model.parameters.flat         # Vector{Float64} - optimizer's view
model.parameters.nested       # NamedTuple by hazard: (h12=(baseline=..., covariates=...), ...)
model.parameters.reconstructor  # ReConstructor for flatten/unflatten
```

To get parameters as a flat vector per hazard (legacy "natural" representation), use:
```julia
get_parameters_natural(model)  # Returns NamedTuple: (h12=[...], h23=[...])
```

**Parameter scale**: All parameters are on **NATURAL scale** (v0.3.0+). Box constraints (`lb ≥ 0`) enforce positivity. There is NO log-transform during fitting.

---

## 4. Key Code Patterns

### Hazard Builder Registry
```julia
# Register a hazard family builder (in hazard_builders.jl):
register_hazard_family!(:wei, _build_weibull_hazard)

# Dispatch by family symbol during construction:
builder = _HAZARD_BUILDERS[family]  # e.g., _HAZARD_BUILDERS[:wei]
hazard = builder(ctx)               # ctx::HazardBuildContext
```

### Parameter Flatten/Unflatten (AD-Compatible)
```julia
rc = model.parameters.reconstructor

# Standard usage (fast, type-stable)
flat = flatten(rc, nested_params)
nested = unflatten(rc, flat_vector)

# AD usage (preserves Dual types)
flat_dual = flattenAD(rc, params_dual)
nested_dual = unflattenAD(rc, flat_dual)
```

### Model Classification Traits
```julia
is_markov(model)              # All hazards time-homogeneous? → Matrix exp MLE
is_panel_data(model)          # Any obstype ≥ 2? → Forward algorithm or MCEM
has_phasetype_expansion(model) # Has :pt hazards? → Expanded state space
```

### Fitting Dispatch Logic
```
fit(model; kwargs...)
  │
  ├─ !is_panel_data(model) ────────────────→ _fit_exact(...)
  │
  └─ is_panel_data(model)
       ├─ is_markov(model) ────────────────→ _fit_markov_panel(...)
       └─ !is_markov(model) ───────────────→ _fit_mcem(...)
```

---

## 5. Data Flow Diagrams

### Model Construction Pipeline
```
User Code                          Internal Pipeline
──────────────────────────────────────────────────────────────────────────
Hazard(:wei, 1, 2)                 → ParametricHazard (spec)
    │
multistatemodel(h1, h2; data=df)   → build_hazards() 
    │                                    │
    │                                    ├─ _HAZARD_BUILDERS[:wei](ctx)
    │                                    │       ↓
    │                                    │   SemiMarkovHazard (runtime)
    │                                    │   with hazard_fn, cumhaz_fn closures
    │                                    │
    │                                build_parameters()
    │                                    │
    │                                    ↓
    └───────────────────────────────→ MultistateModel
                                      with (flat, nested, natural, reconstructor)
```

### Fitting Pipeline (Exact Data)
```
fit(model; penalty=..., ...)
  │
  ↓
_fit_exact(model, ...)
  │
  ├─ Build ExactData container
  ├─ Build samplepaths from data
  │
  ├─ [if penalty] build_penalty_config(model, penalty)
  │
  ├─ Define objective: θ → -loglik_exact(θ, ...) [+ penalty]
  │
  ├─ Optimization.solve(problem, solver)
  │       ↓
  │   Ipopt or OptimizationOptimJL
  │
  ├─ Compute vcov (Hessian inverse)
  ├─ [if requested] Compute IJ/JK variance
  │
  └─ MultistateModelFitted(...)
```

---

## 6. Test Infrastructure

### Directory Layout
```
MultistateModelsTests/
├── src/
│   ├── MultistateModelsTests.jl  # Test runner, test filtering
│   ├── LongTestResults.jl        # Result tracking for longtests
│   └── ReportHelpers.jl          # Report generation
├── fixtures/
│   ├── TestFixtures.jl           # Reusable model fixtures
│   └── *.csv, *.json             # Reference data
├── unit/                         # Fast tests (~2 min total)
│   ├── test_hazards.jl
│   ├── test_splines.jl
│   ├── test_phasetype.jl
│   └── ... (33 files)
├── integration/                  # End-to-end tests
└── longtests/                    # Statistical validation (slow)
    ├── longtest_config.jl
    └── longtest_*.jl
```

### Running Tests
```bash
# All unit tests via Pkg.test()
cd MultistateModels.jl
julia --project -e 'using Pkg; Pkg.test()'

# Specific test file
julia --project=MultistateModelsTests -e 'include("MultistateModelsTests/unit/test_splines.jl")'

# Long tests (slow, ~30+ min)
export MSM_TEST_LEVEL=full
julia --project -e 'using Pkg; Pkg.test()'

# Specific longtest only
export MSM_LONGTEST_ONLY=splines
julia --project=MultistateModelsTests -e 'include("MultistateModelsTests/longtests/longtest_splines.jl")'
```

### Key Test Files by Feature

| Feature | Unit Tests | Long Tests |
|---------|------------|------------|
| Hazard evaluation | `test_hazards.jl`, `test_compute_hazard.jl` | - |
| Splines | `test_splines.jl` | `longtest_splines.jl` |
| Phase-type | `test_phasetype.jl`, `test_phasetype_*.jl` | `longtest_robust_markov_phasetype.jl` |
| Penalty/PIJCV | `test_penalty_infrastructure.jl`, `test_pijcv.jl` | - |
| Variance | `test_variance.jl`, `test_efs.jl` | - |
| MCEM | `test_mcem.jl`, `test_mll_consistency.jl` | `longtest_mcem_*.jl` |
| Fitting | `test_initialization.jl` | Various |

---

## 7. Current Development State

### Branch: `penalized_splines`

**Active Work**: Integrating automatic smoothing parameter selection into fitting (Wave 3: Mathematical Correctness Bugs).

**What Works**:
- Core fitting (exact, Markov panel, MCEM)
- Spline hazards with fixed λ
- PIJCV λ selection via `select_smoothing_parameters()` (standalone)
- Phase-type expansion and FFBS
- Monotone spline penalty transformation (Item #15 complete)
- P-spline knot formula via `default_nknots_penalized()` (Item #16 complete)

**Known Issues** (from CODEBASE_REFACTORING_GUIDE.md):

| Issue | Severity | Description | Status |
|-------|----------|-------------|--------|
| Item #15 | ✅ DONE | Monotone spline penalty matrix transformed correctly | Fixed 2026-01-08 |
| Item #16 | ✅ DONE | `default_nknots_penalized()` uses n^(1/3) formula | Fixed 2026-01-08 |
| Item #5 | 🟡 MED | `rectify_coefs!` review for natural scale params | TODO |
| Item #17 | 🟡 MED | Knot placement uses raw data instead of surrogate | TODO |
| Item #18 | 🟡 MED | PIJCV Hessian occasionally NaN/Inf | TODO |
| Item #19 | 🔴 HIGH | `fit()` doesn't call `select_smoothing_parameters()` automatically | TODO |
| Item #24 | 🟡 MED | Make splines penalized by default (API change) | TODO |

**See**: [scratch/CODEBASE_REFACTORING_GUIDE.md](scratch/CODEBASE_REFACTORING_GUIDE.md) for full details and implementation plan.

---

## 8. Critical Gotchas

### ⚠️ Phase-Type Parameter Indexing
```julia
# WRONG: Assuming hazard index == parameter index
params_idx = hazard_idx  # ❌ Breaks with shared hazards

# CORRECT: Use hazkeys mapping
params_idx = model.hazkeys[hazard.hazname]
```

### ⚠️ Monotone Splines Use I-Spline Transform
```julia
# Optimization parameters (ests) are non-negative increments
# Spline coefficients (coefs) are cumulative sums: coefs = L * ests
# where L is the I-spline transformation matrix

# Penalty must be transformed for monotone splines:
# P(ests) = ests' * S_monotone * ests, where S_monotone = L' * S * L
# This is handled by transform_penalty_for_monotone() in spline_utils.jl (Item #15 fixed)
```

### ⚠️ Test Files That Break on Specific Changes

| If You Change... | These Tests Break |
|------------------|-------------------|
| `parameters.flat` structure | 31+ locations across many files |
| Hazard struct fields | `test_compute_hazard.jl` (18 loc), `test_splines.jl` (12 loc) |
| `fit()` return type | `test_penalty_infrastructure.jl`, `test_model_output.jl` |

### ⚠️ No Unit Tests for `_fit_markov_panel`
The `_fit_markov_panel` function is not directly unit tested. Longtests exist in `longtest_robust_markov_phasetype.jl`.

---

## 9. Quick Reference

### File Locations for Common Tasks

| Task | Primary File(s) |
|------|-----------------|
| Add new hazard family | `construction/hazard_builders.jl` (register), `hazard/generators.jl` |
| Modify fitting | `inference/fit_exact.jl`, `fit_markov.jl`, `fit_mcem.jl` |
| Change likelihood | `likelihood/loglik_*.jl` |
| Modify spline behavior | `hazard/spline.jl`, `utilities/spline_utils.jl` |
| Change parameter handling | `utilities/parameters.jl`, `utilities/reconstructor.jl` |
| Add variance method | `output/variance.jl`, `inference/smoothing_selection.jl` |
| Modify phase-type | `phasetype/*.jl` |

### Commands to Validate Changes

```bash
# Check for errors in workspace
julia --project -e 'using MultistateModels'

# Run specific test file
julia --project=MultistateModelsTests -e 'include("MultistateModelsTests/unit/test_splines.jl")'

# Run full test suite
julia --project -e 'using Pkg; Pkg.test()'

# Check type stability (in REPL)
using MultistateModels
@code_warntype some_function(args...)
```

---

## 10. Cross-References to Other Skills

- **Statistical concepts**: See [multistate-domain/SKILL.md](multistate-domain/SKILL.md)
- **Testing patterns**: See [julia-testing/SKILL.md](julia-testing/SKILL.md)
- **Optimization**: See [numerical-optimization/SKILL.md](numerical-optimization/SKILL.md)
- **Spline math**: See [smoothing-splines/SKILL.md](smoothing-splines/SKILL.md)
- **Stochastic processes**: See [stochastic-processes/SKILL.md](stochastic-processes/SKILL.md)

---

## ⚠️ Keeping This Skill Current

**This skill file MUST be updated whenever you make changes to the codebase.**

### When to Update This Skill

| Change Type | Required Update |
|-------------|-----------------|
| Add/remove/rename files | Update Section 2 (Source Architecture) |
| Add/modify types | Update Section 3 (Type System Map) |
| Change API signatures | Update Section 4 (Code Patterns), Section 5 (Data Flow) |
| Fix technical debt items | Update Section 7 (Current State) |
| Discover new gotchas | Add to Section 8 (Gotchas) |
| Modify test infrastructure | Update Section 6 (Test Infrastructure) |

### Update Checklist

Before ending any session where code was modified:
- [ ] Does Section 2 still accurately describe file organization?
- [ ] Are type hierarchies in Section 3 still correct?
- [ ] Do data flow diagrams reflect current code paths?
- [ ] Is the "Current Development State" still accurate?
- [ ] Should any new gotchas be documented?

### How to Update

1. Make changes directly to `.github/skills/codebase-knowledge/SKILL.md`
2. Add a dated entry to the "Change Log" section below
3. If the skill file exceeds ~800 lines, create companion files and reference them

---

## Change Log

| Date | Author | Changes |
|------|--------|---------|
| 2026-01-08 | julia-statistician | Initial creation from CODE_READING_GUIDE.md and CODEBASE_REFACTORING_GUIDE.md |
| 2026-01-08 | julia-statistician | Completed Wave 1 refactoring: deleted BatchedODEData, is_separable(), legacy type aliases, deprecated draw_paths/get_loglik/fit_phasetype_surrogate overloads |
| 2026-01-08 | julia-statistician | Wave 2 partial: Items #8 (get_ij_vcov/get_jk_vcov deleted), #9 (FlattenAll removed), #6 (AD backends unexported). Item #10 skipped (needs benchmarks). Item #21 remaining. |
| 2026-01-10 | julia-statistician | Wave 2 complete: Item #21 - Removed `parameters.natural` field; now computed on-demand via `get_parameters_natural()`. Updated 8 source files, 2 test files, and 1 doc file. Tests: 1458 passed, 1 errored (pre-existing). |
| 2026-01-08 | julia-statistician | Wave 2 finalized: Item #10 resolved - kept both CachedTransformStrategy and DirectTransformStrategy (former for production, latter for debugging). Added unit tests in test_simulation.jl. Wave 2 complete. |
| 2026-01-08 | julia-statistician | Wave 3 partial: Item #16 - Created `default_nknots_penalized(n)` in src/hazard/spline.jl using n^(1/3) P-spline formula. Item #15 - Created `build_ispline_transform_matrix()` and `transform_penalty_for_monotone()` in src/utilities/spline_utils.jl; modified `build_spline_hazard_info()` to apply S_monotone = L'SL transformation for monotone splines. Tests: 1484 passed, 0 failed, 1 errored (pre-existing). |
| 2026-01-08 | julia-statistician | Wave 3 complete: Item #24 - Made splines penalized by default. Added `has_spline_hazards()` helper, `_resolve_penalty()` function, changed `_fit_exact` and `_fit_mcem` defaults to `penalty=:auto`. Added deprecation warning for `penalty=nothing`. New API: `:auto` (default), `:none` (explicit opt-out), `SplinePenalty()` (explicit penalty). Tests: 1486 passed, 0 failed, 0 errored. Waves 1-3 complete. Fixed SplineHazardInfo symmetry check for zero penalty matrices. Removed tests for non-existent mcem_lml functions. |
| 2026-01-12 | julia-statistician | **Item #25 RESOLVED**: Fixed natural-scale parameter migration. Root cause was documentation inconsistency - code already expected natural scale but docstrings said log scale. Updated docstrings in parameters.jl, fit_*.jl, loglik_*.jl. Updated all longtests to pass natural-scale parameters. `simulate()` and `get_parameters(;scale=:natural)` now work correctly. Tests: 1486 passed. |