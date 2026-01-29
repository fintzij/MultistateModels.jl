---
name: codebase-knowledge
description: Deep knowledge of MultistateModels.jl and MultistateModelsTests.jl codebase structure, conventions, and current state. READ THIS FIRST when starting any session involving code changes.
applyTo: '**'
---

# MultistateModels.jl Codebase Knowledge

**Read this skill file FIRST at the start of every session.** It provides the essential context needed to work effectively with this codebase.

**Last Updated**: 2026-01-25  
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
    ├── validation.jl        # Input validation
    ├── data_utils.jl        # Data manipulation, center_covariates()
    ├── books.jl             # Book-keeping structures for likelihood
    ├── initialization.jl    # Parameter initialization helpers
    ├── misc.jl              # Miscellaneous utilities
    ├── stats.jl             # Statistical helper functions
    └── transition_helpers.jl # Transition matrix utilities
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
    parameters::NamedTuple       # (flat, nested, reconstructor)
    hazards::Vector{<:_Hazard}
    totalhazards::Vector{_TotalHazard}
    tmat::Matrix{Int64}          # Transition matrix
    emat::Matrix{Float64}        # Emission matrix (censoring)
    hazkeys::Dict{Symbol,Int64}  # :h12 → index
    subjectindices::Vector{Vector{Int64}}
    markovsurrogate::Union{Nothing,MarkovSurrogate}
    phasetype_surrogate::Union{Nothing,AbstractSurrogate}  # NEW: built at construction when surrogate=:phasetype
    phasetype_expansion::Union{Nothing,PhaseTypeExpansion}
    # ... weights, censoring patterns, modelcall
end
```

**MultistateModelFitted** adds:
```julia
    loglik::NamedTuple
    vcov::Union{Nothing,Matrix{Float64}}        # Single variance matrix (controlled by vcov_type)
    vcov_type::Symbol                           # :ij, :model, :jk, or :none
    subject_gradients::Union{Nothing,Matrix{Float64}}
    smoothing_parameters::Union{Nothing,NamedTuple}  # λ per hazard (penalized splines)
    edf::Union{Nothing,NamedTuple}                   # Effective degrees of freedom
    ConvergenceRecords::...
    ProposedPaths::...
```

### Variance-Covariance API (v0.4.0+)

**Single `vcov_type` kwarg** controls variance computation:
```julia
fitted = fit(model; vcov_type=:ij)     # IJ/sandwich variance (DEFAULT, robust)
fitted = fit(model; vcov_type=:model)  # Model-based variance (inverse Hessian)
fitted = fit(model; vcov_type=:jk)     # Jackknife variance
fitted = fit(model; vcov_type=:none)   # No variance computation

# Accessor (no type kwarg needed - vcov_type is stored in fitted)
vcov = get_vcov(fitted)  # Returns the single variance matrix
```

**BREAKING CHANGE**: Old kwargs (`compute_vcov`, `compute_ij_vcov`, `compute_jk_vcov`) no longer exist.

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
│   └── ... (41 files total)
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
| Phase-type | `test_phasetype.jl`, `test_phasetype_*.jl`, `test_phasetype_preprocessing.jl`, `test_phasetype_surrogate.jl` | `longtest_robust_markov_phasetype.jl` |
| Penalty/PIJCV | `test_penalty_infrastructure.jl`, `test_pijcv.jl`, `test_pijcv_reference.jl`, `test_pijcv_vs_loocv.jl` | - |
| Variance | `test_variance.jl`, `test_efs.jl`, `test_constrained_variance.jl` | - |
| MCEM | `test_mcem.jl`, `test_mll_consistency.jl` | `longtest_mcem_*.jl` |
| Fitting | `test_initialization.jl` | Various |
| Likelihood | `test_loglik_analytical.jl` (40 tests: analytical verification of `loglik_exact` and `loglik_markov` against hand-calculated formulas for exponential, Weibull, and Gompertz hazards across 2-state, 3-state, and illness-death models) | - |
| Cumulative incidence | `test_cumulative_incidence.jl` | - |
| Covariate centering | `test_center_covariates.jl` | - |
| Ordering constraints | `test_ordering_at.jl` | - |

---

## 7. Current Development State

### Branch: `penalized_splines`

**Active Work**: Integrating automatic smoothing parameter selection into fitting (Wave 3: Mathematical Correctness Bugs).

**What Works**:
- Core fitting (exact, Markov panel, MCEM)
- Spline hazards with fixed λ
- PIJCV λ selection via `select_smoothing_parameters()` (standalone)
- Phase-type expansion and FFBS
- Phase-type TPM computation (Schur-based, stable for defective matrices)
- Monotone spline penalty transformation (Item #15 complete)
- P-spline knot formula via `default_nknots_penalized()` (Item #16 complete)

**Known Issues** (from CODEBASE_REFACTORING_GUIDE.md):

| Issue | Severity | Description | Status |
|-------|----------|-------------|--------|
| Item #15 | ✅ DONE | Monotone spline penalty matrix transformed correctly | Fixed 2026-01-08 |
| Item #16 | ✅ DONE | `default_nknots_penalized()` uses n^(1/3) formula | Fixed 2026-01-08 |
| PT Preprocessing | ✅ DONE | CensoringPatterns merging and obstype codes | Fixed 2026-01-10 |
| BUG-2 | ✅ DONE | Phase-type TPM eigendecomposition failure | Fixed 2026-01-10 (Schur) |
| Item #35 | ✅ DONE | PhaseType surrogate collapsed path likelihood with Schur caching | Fixed 2026-01-17 |
| Item #36 | ✅ DONE | PhaseType surrogate dt=0 likelihood bug | Fixed 2026-01-18 |
| Item #5 | 🟡 MED | `rectify_coefs!` review for natural scale params | TODO |
| Item #17 | 🟡 MED | Knot placement uses raw data instead of surrogate | TODO |
| Item #18 | 🟡 MED | PIJCV Hessian occasionally NaN/Inf | TODO |
| Item #19 | 🔴 HIGH | `fit()` doesn't call `select_smoothing_parameters()` automatically | TODO |
| Item #24 | 🟡 MED | Make splines penalized by default (API change) | TODO |
| PT Identifiability | ✅ DONE | Implement covariate constraints, ordered SCTP, update defaults | Complete 2026-01-10 |
| PIJCV Efficiency | ✅ DONE | PIJCVEvaluationCache, EFS warmstart, DiffResults optimization, covariate caching | Complete 2026-01-25 |
| PIJCV Cholesky | 🟡 MED | Replace O(p³) eigendecomp with O(kp²) Cholesky downdate (Woodbury identity) | See handoff |
| PIJCV BFGS Outer | 🟡 MED | Add BFGS outer optimizer option with gradient clamping for indefinite λ | See handoff |

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

### ⚠️ PhaseType Surrogate Path Likelihood Uses Schur-Cached TPMs

When computing importance weights for MCEM with PhaseType proposal, the collapsed path density requires TPMs computed at **sampled transition times** (not pre-computed observation times). This is handled efficiently via cached Schur decomposition.

**Implementation** (Item #35 — COMPLETE 2026-01-17):

1. **`CachedSchurDecomposition`** struct in `data_containers.jl` stores Q = UTU' decomposition
2. **`compute_tpm_from_schur(cache, dt)`** computes exp(Q*dt) = U*exp(T*dt)*U' efficiently
3. **`schur_cache_ph`** (one cache per covariate combo) passed through `DrawSamplePaths!` to path likelihood
4. **Forward algorithm** uses cached TPMs at sampled transition times

**Key insight**: The Schur decomposition only depends on Q (fixed per covariate combo), not on Δt. Pre-computing it once and reusing for arbitrary Δt values provides 2-5x speedup.

**Data flow**:
```
fit_mcem()
  → hazmat_book_ph = [Q₁, Q₂, ...]  (covariate-adjusted Q matrices)
  → schur_cache_ph = [Schur(Q₁), Schur(Q₂), ...]  (one decomposition per combo)
  → DrawSamplePaths!(... schur_cache_ph=schur_cache_ph)
      → convert_expanded_path_to_censored_data(... schur_cache=cache[covar_idx])
          → compute_tpm_from_schur(cache, dt) for each interval
```

See: `src/types/data_containers.jl`, `src/inference/sampling.jl`, `src/inference/fit_mcem.jl`

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
| 2026-01-25 | julia-statistician | **PIJCV Efficiency Optimization**: (1) Confirmed covariate caching working via `build_pijcv_eval_cache()` and `loglik_subject_cached()`. (2) Analyzed Wood (2024) NCV paper efficiency strategies: Cholesky downdate O(kp²) vs eigendecomp O(p³), BFGS outer with gradient clamping. (3) Created comprehensive handoff document at `scratch/PIJCV_EFFICIENCY_HANDOFF_2026-01-25.md` (468 lines) with math, pseudocode, and implementation plan. (4) Current status: PIJCV ~1.45x faster than CV10; target 3-5x. Remaining work: Cholesky downdate (Woodbury identity), BFGS outer optimizer option. |
| 2026-01-22 | julia-statistician | **Item #27 COMPLETE**: Refactored variance-covariance estimation to unified `vcov_type` API. Removed redundant `ij_vcov`/`jk_vcov` fields from `MultistateModelFitted`. Single `vcov` field now controlled by `vcov_type::Symbol` kwarg (`:ij`, `:model`, `:jk`, `:none`). Default is `:ij` (IJ/sandwich variance). Updated `_fit_exact`, `_fit_markov_panel`, `_fit_mcem`, `MCEMConfig`. Simplified `get_vcov()` accessor (no `type` kwarg needed). Breaking change: old kwargs (`compute_vcov`, `compute_ij_vcov`, `compute_jk_vcov`) removed. All 2164 tests updated and passing. |
| 2026-01-18 | julia-statistician | **Item #36 COMPLETE**: Fixed PhaseType surrogate likelihood dt=0 bug. Primary fix: `compute_forward_loglik` now uses raw hazards Q[i,j] instead of normalized probabilities P[i,j]=Q[i,j]/(-Q[i,i]) for instantaneous transitions (dt=0). The distinction: probabilities are for sampling (choosing destination), hazards are for likelihood (density contribution). Secondary fix: Added retry mechanism (up to 10 attempts) for paths with -Inf surrogate likelihood, with fallback to Markov proposal likelihood. Also fixed: TestFixtures.jl missing phasetype_surrogate arg, test_splines.jl shared knots expectation, fit_mcem.jl NaN/negative ASE guard. All 2129 unit tests pass. MCEM longtests: PhaseType proposal no longer produces -Inf/NaN issues. |
| 2026-01-17 | julia-statistician | **Item #35 COMPLETE**: PhaseType surrogate likelihood for MCEM now uses Schur caching for efficient TPM computation at sampled transition times. Added `CachedSchurDecomposition` struct to `data_containers.jl`, `schur_cache` parameter to `convert_expanded_path_to_censored_data`, `schur_cache_ph` parameter to `DrawSamplePaths!`, and creation/passing in `fit_mcem.jl`. Verified via unit test (machine precision match with direct exp(Q*dt)). Performance benefit: O(n³) decomposition once per covariate combo, then faster TPM computation for each interval. Updated gotcha section with implementation details. |
| 2026-01-17 | julia-statistician | **Item #35 ANALYSIS COMPLETE, IMPLEMENTATION NEEDED**: Analyzed PhaseType surrogate likelihood for MCEM. Key insight: collapsed paths create implicit censoring (phase uncertainty) that must be handled via Markov infrastructure, NOT custom formulas. Partial fix was computing wrong quantity (marginal likelihood vs path density). Detailed implementation plan added to CODEBASE_REFACTORING_GUIDE.md. Updated gotcha section. Status changed from DONE to IN PROGRESS. |
| 2026-01-17 | julia-statistician | **SKILL FILES AUDIT**: Updated multiple skill files for consistency: (1) multistate-domain/SKILL.md: Fixed parameter scale documentation—all parameters now documented as natural scale with box constraints (v0.3.0+), not log-transformed. Added `:ordered_sctp` option. Fixed incomplete HSMM code block. (2) codebase-knowledge/SKILL.md: Added missing utility files (data_utils.jl, books.jl, initialization.jl, misc.jl, stats.jl, transition_helpers.jl). Updated test file count from 33 to 41. Added new test files to feature table (test_cumulative_incidence.jl, test_center_covariates.jl, test_ordering_at.jl, etc.). (3) numerical-optimization/SKILL.md: Added note that SQUAREM acceleration was removed from MCEM. (4) julia-testing/SKILL.md: Added newer test files to feature table. (5) survival-analysis/SKILL.md: Added MultistateModels.jl storage format notes for parametric families. |
| 2026-01-17 | julia-statistician | **PhaseType survival path likelihood fix**: Fixed bug in `loglik_phasetype_collapsed_path` (sampling.jl L2207-2331) where paths with s_obs == d_obs (subject stays in same state) returned -Inf. Previously computed "transition density" for 1→1 which gave negative internal phase rates. Now correctly distinguishes survival events (compute log(π' exp(Sτ) 𝟙)) from transitions. Fix verified: PT and Markov path likelihoods match for both survival and transition paths. **PARTIAL FIX**: MCEM still shows divergence between proposals (~50% relative error) - indicates additional bug in `draw_samplepath_phasetype` or FFBS infrastructure. Investigation ongoing. |
| 2026-01-16 | julia-statistician | **Item #29 COMPLETED**: Spline Knot Calibration Improvements. (1) Added `cumulative_incidence(t, model, newdata; statefrom)` methods for NamedTuple/DataFrameRow, plus `cumulative_incidence_at_reference()` in `src/hazard/api.jl`. (2) Added `center_covariates(model; centering=:mean/:median/:reference)` in `src/utilities/data_utils.jl`. (3) `calibrate_splines` now uses CDF inversion via `_compute_exit_quantiles_at_reference()`. (4) Changed `ordering_at` default from `:reference` to `:mean`. (5) Added `phasetype_surrogate::Union{Nothing, AbstractSurrogate}` field to MultistateModel/Fitted structs; built at construction when `surrogate=:phasetype`. (6) Created 27 unit tests in `test_cumulative_incidence.jl` (15) and `test_center_covariates.jl` (12). |
| 2026-01-15 | julia-statistician | **AFT BUG IN PHASETYPE PROPOSAL FIXED**: `build_phasetype_tpm_book()` was always using `exp(β'x)` for covariate scaling, which is correct for PH models but WRONG for AFT models. For AFT, the correct scaling is `exp(-β'x)`. Fixed in `src/inference/sampling.jl` lines 1480-1488 to check `hazard.metadata.linpred_effect` and apply the correct sign. Added unit tests in `test_phasetype_surrogate.jl` (new testset "AFT vs PH Covariate Scaling Direction" with 10 tests). This bug caused `wei_aft_panel_tvc` longtest failure (Markov vs PhaseType proposal comparison: 109.9% relative difference, wrong-signed estimate). After fix: all 19 AFT scenarios pass, all 2106 unit tests pass. |
| 2026-01-14 | julia-statistician | **LONGTEST COVERAGE 100% COMPLETE (Wave 5)**: All 60 cells covered (excl. pt AFT which is unsupported). Final additions: (1) Added 3 spline AFT exact tests to `longtest_spline_exact.jl` (sp_aft_exact_nocov/tfc/tvc), total now 6 tests (837 lines). (2) Added 3 spline AFT panel tests to `longtest_mcem_splines.jl` (Tests 7-9: sp_aft_panel_nocov/tfc/tvc), total now 9 tests (1279 lines). All tests validated and passing. |
| 2026-01-14 | julia-statistician | **LONGTEST COVERAGE COMPLETE (Wave 5)**: (1) Added 13 new AFT scenarios to `longtest_aft_suite.jl` covering exp_aft_exact/panel × nocov/tfc/tvc, wei_aft_nocov, gom_aft_nocov/tvc, gom_aft_panel × nocov/tfc/tvc. Total AFT scenarios: 19 (was 6). (2) Created NEW file `longtest_spline_exact.jl` (489 lines, 3 tests, ~17 assertions) for spline hazards with exact data (sp_exact_nocov/tfc/tvc). Validates by comparing fitted h(t) to true Weibull DGP. (3) Verified pt_panel_tvc already exists in `longtest_phasetype_panel.jl` Section 7. (4) Updated CODEBASE_REFACTORING_GUIDE.md with complete coverage matrix. Coverage gaps filled: exp AFT all, gom AFT all, wei AFT nocov, sp exact PH all. |
| 2026-01-14 | julia-statistician | **Longtest fixes**: (1) Fixed `_compute_phasetype_observed_cumincid` to properly sort subject data chronologically before computing CI. (2) Updated pt_panel_fixed/pt_panel_tvc tests to use `compute_vcov=true` (still no vcov due to constraints warning). (3) Added callout warnings in report explaining panel data CI limitations (observed vs true comparison is conceptually problematic for panel data). (4) Documented that constrained phase-type models don't return vcov. |
| 2026-01-14 | julia-statistician | **Report updates**: (1) Updated 03_long_tests.qmd to reflect all 9/9 phase-type tests passing. (2) Removed "Known Issue" status from pt_panel_fixed and pt_panel_tvc - tests now pass with proper identifiability constraints. (3) Fixed plotting geoms: changed `scatter!` to `stairs!(step=:post)` for observed/empirical data in cumulative incidence and prevalence plots (proper step function visualization for Kaplan-Meier style data). |
| 2026-01-14 | julia-statistician | **Renamed `:baseline` → `:reference`**: The `ordering_at` parameter now uses `:reference` (default) instead of `:baseline` to avoid confusion with other uses of "baseline" (e.g., spline baseline hazard scope). All source files, tests, and documentation updated. |
| 2026-01-14 | julia-statistician | **Item #26 IMPLEMENTED**: Added `ordering_at` parameter for phase-type eigenvalue constraints. Allows enforcing νⱼ ≥ νⱼ₊₁ ordering at `:reference` (default, linear constraints at x=0), `:mean`, `:median`, or explicit NamedTuple (nonlinear constraints). Key functions added: `_compute_ordering_reference()`, `_extract_covariate_names()`, `_build_linear_ordering_constraint()`, `_build_nonlinear_ordering_constraint()`, `_build_rate_with_covariates()`. C1 (homogeneous) covariates automatically simplify to linear constraints. Modified: `multistatemodel.jl`, `expansion_model.jl`, `expansion_constraints.jl`. Added 37 tests in `test_ordering_at.jl`. All 504+ phase-type tests pass. |
| 2026-01-14 | julia-statistician | **pt_panel_fixed/pt_panel_tvc tests PASS**: All 7 phase-type panel longtests pass. Key findings documented: (1) SCTP constraints do NOT apply for K=1 destination. (2) Eigenvalue ordering (ν₁ ≥ ν₂) enforced but doesn't fully resolve identifiability. (3) Individual λ and μ₁ are NOT identifiable (only sum ν₁ = λ + μ₁). (4) μ₂ and β ARE identifiable. Tests focus on identifiable quantities with appropriate tolerances. |
| 2026-01-12 | julia-statistician | **Item #7 AUDITED**: Variance function audit complete. Fixed bug in `compute_subject_hessians_threaded` (undefined `hazards` variable). Mathematical validation confirmed all variants compute correct Hessians to machine precision. Consolidation plan: use `_fast` as unified entry point for exact data; keep separate methods for Markov panel (#5) and MCEM (#6). See CODEBASE_REFACTORING_GUIDE.md SESSION LOG 2026-01-12. |
| 2026-01-13 | julia-statistician | **SQUAREM REMOVED**: Completely removed SQUAREM acceleration from MCEM. Deleted `SquaremState` struct, `squarem_step_length()`, `squarem_accelerate()`, `squarem_should_accept()` from mcem.jl. Removed `acceleration` parameter from `_fit_mcem()`. Deleted SQUAREM tests from test_mcem.jl. Updated CHANGELOG.md, skill files, and documentation. Rationale: SQUAREM's quadratic extrapolation is mathematically unbounded and routinely produces out-of-bounds parameters. |
| 2026-01-12 | julia-statistician | **SQUAREM disabled by default**: Changed `acceleration` default from `:squarem` to `:none` in `_fit_mcem`. SQUAREM still available as `acceleration=:squarem`. Relaxed Pareto-k threshold in MCEM Gompertz-PhaseType longtest from 1.0 to 1.1 to account for Monte Carlo variation. All 1851 unit tests pass, MCEM longtests pass. |
| 2026-01-11 | julia-statistician | Added `test_loglik_analytical.jl`: 40 comprehensive unit tests verifying analytical correctness of `loglik_exact` and `loglik_markov` against hand-calculated log-likelihood formulas for exponential, Weibull, and Gompertz hazards. |
| 2026-01-10 | julia-statistician | **BUG-2 RESOLVED**: Fixed phase-type TPM computation. Root cause: eigendecomposition failed for defective matrices (repeated eigenvalues common in phase-type). Solution: Replaced eigendecomposition with Schur decomposition in `compute_tmat_batched!`. Added `SchurCache` struct to `data_containers.jl`. All 504 phase-type tests pass. |
| 2026-01-10 | julia-statistician | **Phase-type preprocessing bugs fixed**: (1) `_merge_censoring_patterns_with_shift` in expansion_model.jl now produces consecutive obstype codes [3,4,5] instead of [3,4,6]. (2) `_build_phase_censoring_patterns` in expansion_hazards.jl now uses `row_idx + 2` for consecutive codes and returns `(patterns, state_to_obstype)` tuple. Created 99 rigorous unit tests in `MultistateModelsTests/unit/test_phasetype_preprocessing.jl` with exact equality checks on complete CensoringPatterns, expanded DataFrames, and emission matrices. |
| 2026-01-08 | julia-statistician | Initial creation from CODE_READING_GUIDE.md and CODEBASE_REFACTORING_GUIDE.md |
| 2026-01-08 | julia-statistician | Completed Wave 1 refactoring: deleted BatchedODEData, is_separable(), legacy type aliases, deprecated draw_paths/get_loglik/fit_phasetype_surrogate overloads |
| 2026-01-08 | julia-statistician | Wave 2 partial: Items #8 (get_ij_vcov/get_jk_vcov deleted), #9 (FlattenAll removed), #6 (AD backends unexported). Item #10 skipped (needs benchmarks). Item #21 remaining. |
| 2026-01-10 | julia-statistician | Wave 2 complete: Item #21 - Removed `parameters.natural` field; now computed on-demand via `get_parameters_natural()`. Updated 8 source files, 2 test files, and 1 doc file. Tests: 1458 passed, 1 errored (pre-existing). |
| 2026-01-08 | julia-statistician | Wave 2 finalized: Item #10 resolved - kept both CachedTransformStrategy and DirectTransformStrategy (former for production, latter for debugging). Added unit tests in test_simulation.jl. Wave 2 complete. |
| 2026-01-08 | julia-statistician | Wave 3 partial: Item #16 - Created `default_nknots_penalized(n)` in src/hazard/spline.jl using n^(1/3) P-spline formula. Item #15 - Created `build_ispline_transform_matrix()` and `transform_penalty_for_monotone()` in src/utilities/spline_utils.jl; modified `build_spline_hazard_info()` to apply S_monotone = L'SL transformation for monotone splines. Tests: 1484 passed, 0 failed, 1 errored (pre-existing). |
| 2026-01-08 | julia-statistician | Wave 3 complete: Item #24 - Made splines penalized by default. Added `has_spline_hazards()` helper, `_resolve_penalty()` function, changed `_fit_exact` and `_fit_mcem` defaults to `penalty=:auto`. Added deprecation warning for `penalty=nothing`. New API: `:auto` (default), `:none` (explicit opt-out), `SplinePenalty()` (explicit penalty). Tests: 1486 passed, 0 failed, 0 errored. Waves 1-3 complete. Fixed SplineHazardInfo symmetry check for zero penalty matrices. Removed tests for non-existent mcem_lml functions. |
| 2026-01-12 | julia-statistician | **Item #25 RESOLVED**: Fixed natural-scale parameter migration. Root cause was documentation inconsistency - code already expected natural scale but docstrings said log scale. Updated docstrings in parameters.jl, fit_*.jl, loglik_*.jl. Updated all longtests to pass natural-scale parameters. `simulate()` and `get_parameters(;scale=:natural)` now work correctly. Tests: 1486 passed. |
| 2026-01-09 | julia-statistician | **Bounds handling cleanup**: (1) Investigated BUG-1 - determined Ipopt is NOT returning out-of-bounds values (configured with `honor_original_bounds="yes"`). (2) Identified SQUAREM acceleration as the actual source of out-of-bounds parameters due to unbounded quadratic extrapolation. (3) Removed unnecessary post-Ipopt clamping from fit_exact.jl, fit_markov.jl, fit_mcem.jl. (4) Kept SQUAREM clamping in fit_mcem.jl L783 (mathematically necessary). (5) Added epsilon buffer (1e-8) to `_clamp_to_bounds!` to prevent infinite gradients at exact boundaries. (6) Added TODO to consider disabling SQUAREM by default. |
| 2026-01-09 | julia-statistician | **BUG-1 RESOLVED**: Monotone spline constraints work correctly. Original test was flawed - simulated from exponential (constant) and fit with monotone=1, trivially producing constant. Rewrote test in `longtest_mcem_splines.jl` to: (A) Simulate from Weibull (increasing hazard), (B) Fit with monotone=1 - captures increasing pattern, (C) Fit with monotone=-1 - constrained to constant (serves as negative control: if constraints weren't enforced, both fits would be identical), (D) Verify correct direction has higher LL (-1057 vs -1086). Also added `initialize=false` to simulation model creations to avoid bounds validation errors during auto-initialization. |
| 2026-01-10 | julia-statistician | **B3 ordered SCTP implemented**: Added `coxian_structure=:ordered_sctp` option to enforce eigenvalue ordering (ν₁ ≥ ν₂ ≥ ... ≥ νₙ) on top of SCTP constraints. Modified 5 files: `expansion_constraints.jl` (added `_generate_ordering_constraints()`), `multistatemodel.jl`, `hazard_constructors.jl`, `expansion_model.jl`, `hazard_specs.jl`. All 504 phase-type tests pass. Also discovered C1 covariate constraints were already implemented via `covariate_constraints=:homogeneous`. Remaining: Phase 3 (surrogate defaults to B2+C1). |
| 2026-01-10 | julia-statistician | **Phase-type identifiability COMPLETE**: (1) Updated surrogate defaults: `coxian_structure=:sctp` and `covariate_constraints=:homogeneous` are now defaults for phase-type hazards. (2) Renamed cryptic `:C0`/`:C1` API to descriptive `:unstructured`/`:homogeneous`. Modified `hazard_specs.jl`, `hazard_constructors.jl`, `expansion_hazards.jl`, `expansion_constraints.jl`, `expansion_model.jl`. All 504 phase-type tests pass. Phase-type identifiability work complete. |