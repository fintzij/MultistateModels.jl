---
title: "Model Generation Testing Guide"
format:
    html:
        theme:
            light: darkly
            dark: darkly
        prefer-dark: true
        highlight-style: atom-one-dark
---

# Model Generation Testing Guide

This document explains the model generation process in MultistateModels.jl and how each step is unit tested.

## Overview

Model generation in MultistateModels.jl follows a multi-step validation and construction process. Each step has corresponding unit tests in `test/test_modelgeneration.jl` (run via `Pkg.test(test_args=["test_modelgeneration"])`). All of the model-generation-focused tests currently pass ✅. Status last verified on **2025-11-25** by running `julia --project -e 'using Pkg; Pkg.test(test_args=["test_modelgeneration"])'` on branch `infrastructure_changes`.

## Model Generation Process

### STEP 1: User Creates Hazard Objects

**Recommended DSL (`@hazard` macro):**
```julia
h12 = @hazard begin
    family = :exp          # strings also work: "exp"
    statefrom = 1
    stateto = 2
end

h13 = @hazard begin
    family = :wei          # accepts :wei/:weibull
    formula = @formula(0 ~ age)
    transition = 1 => 3    # or from = 1; to = 3; states = (1, 3)
end

h14 = @hazard begin
    family = :gom
    transition = 1 => 4    # formula omitted ⇒ intercept-only hazard
end
```

- `family` accepts case-insensitive strings or symbols for `exp`, `wei`, `gom`, and `sp`. Tags `:ode`/`:neural_ode` are reserved; the macro emits a descriptive error until those hazard types ship.
- Aliases `from`, `to`, `states`, or `transition = 1 => 2` map to `statefrom`/`stateto` so you can pick whatever reads best.
- The macro forwards every extra keyword (e.g., `linpred_effect = :aft`, `time_transform = true`) straight to the underlying `Hazard` constructor.
- If you omit `formula` (or `@formula(0 ~ ...)`) altogether, the macro injects the intercept-only design `@formula(0 ~ 1)` automatically. Use this for baseline-only hazards.

**`@hazard` macro cheat sheet**

| Field in macro block | Accepted aliases | Effect |
|----------------------|------------------|--------|
| `family`             | Case-insensitive strings or symbols (`"exp"`, `:wei`, `:gom`, `:sp`) | Passed directly to `Hazard` and validated the same way. |
| `statefrom`, `stateto` | `from`, `to`, `states = (f, t)`, `transition = f => t` | Normalized to integer origin/destination states. |
| `formula`             | `hazard = @formula(...)` | Injected as the first positional argument; defaults to `@formula(0 ~ 1)` when omitted. |
| Additional keywords  | `linpred_effect`, `time_transform`, spline controls, solver hints, etc. | Forwarded verbatim to the constructor so Tang/ODE settings stay consistent. |

- The macro raises an `ArgumentError` if you forget to supply either `(statefrom, stateto)` or `transition`, or if you pick an unsupported `family` (e.g., future `:ode` tags that are not yet implemented).
- Prefer `@hazard` in tests and docs because the structured block stays readable as we add Tang/ODE knobs; fall back to the raw constructor when you need to metaprogram or generate hazards dynamically.

**Constructor form (still supported):**
```julia
h12 = Hazard(@formula(0 ~ trt), "exp", 1, 2)
h13 = Hazard(@formula(0 ~ age), "wei", 1, 3)
h14 = Hazard("gom", 1, 4)   # formula omitted ⇒ intercept-only
```

Both entry points automatically supply an intercept column even if you omit `+ 1` in the formula. Adding an explicit intercept term is harmless but unnecessary.

**What it does:** 
Creates hazard specifications for each transition:
- `@formula(0 ~ trt)` = log-hazard formula; baseline hazard supplies the intercept so you list only covariates.
- `"exp"` / `:wei` = hazard family (exponential, Weibull, Gompertz, spline).
- `statefrom = 1`, `stateto = 2` (or aliases) = origin/destination states.

**Unit Test:** `test_hazard_construction`

```julia
h_exp = Hazard(@formula(0 ~ trt), "exp", 1, 2)
model_exp = multistatemodel(h_exp; data = dat)
@test length(model_exp.parameters[1]) == 2  # Intercept + trt from the auto-added baseline term
@test model_exp.hazards[1].parnames == [:h12_Intercept, :h12_trt]
```

**Verifies:** Hazard objects create correct parameter structures and names for different hazard families while respecting the unified DSL.

---

### STEP 2: Enumerate and Validate Hazards

**Code:**
```julia
hazinfo = enumerate_hazards(hazards...)
```

**What it does:**
- Extracts `(statefrom, stateto)` pairs from all hazards
- **Checks for duplicate transitions** - each transition should only be specified once
- Sorts hazards by origin state, then destination state
- Assigns transition numbers

**Unit Test:** `test_duplicate_transitions`

```julia
h1 = Hazard("exp", 1, 2)
h2 = Hazard("exp", 1, 2)  # DUPLICATE!

@test_throws ErrorException multistatemodel(h1, h2; data = dat)

# Verify error message mentions duplicate transitions
try
    multistatemodel(h1, h2; data = dat)
    @test false  # Should not reach here
catch e
    @test occursin("Duplicate transitions", string(e))
    @test occursin("(1, 2)", string(e))
end
```

**Verifies:** 
- System catches duplicate transition definitions
- Provides clear, informative error message with the duplicate transition pair

---

### STEP 3: Create Transition Matrix

**Code:**
```julia
tmat = create_tmat(hazinfo)
```

**What it does:**
Creates a matrix where `tmat[i,j]` = transition number from state i to state j. Zero entries mean no direct transition is possible.

**Example:**
```
     State: 1  2  3
State 1:    0  1  2    # Can go from 1→2 (trans 1) or 1→3 (trans 2)
State 2:    3  0  4    # Can go from 2→1 (trans 3) or 2→3 (trans 4)
State 3:    0  0  0    # Absorbing state (no exits)
```

**Unit Test:** `test_tmat`

```julia
@test sort(msm_expwei.tmat[[2,4,7,8]]) == collect(1:4)  # Check transition numbers
@test all(msm_expwei.tmat[Not([2,4,7,8])] .== 0)       # Check zeros elsewhere
```

**Verifies:** 
- Transition matrix has correct structure
- Transitions are numbered correctly and placed in the right positions
- Non-transitions are properly marked as zero

---

### STEP 4: Data Validation

**Code:**
```julia
check_data!(data, tmat, CensoringPatterns; verbose = verbose)
```

**What it does:**
- Ensures required columns exist: `id`, `tstart`, `tstop`, `statefrom`, `stateto`, `obstype`
- **Checks for state 0 misuse** - state 0 should not be in `statefrom`
- Warns about non-contiguous states (e.g., 1, 2, 4 skipping 3)
- Validates observation types and censoring patterns

**Unit Tests:**

#### `test_state_zero_in_data` - State 0 can appear for censoring

```julia
dat = DataFrame(
    id = [1, 1],
    tstart = [0.0, 1.0],
    tstop = [1.0, 2.0],
    statefrom = [1, 1],
    stateto = [2, 0],   # second row ends in state 0 to indicate censoring
    obstype = [1, 3]    # obstype 3 is described by the censoring matrix below
)

CensoringPatterns = Int64[3 0 1 1]  # pattern 3 allows states 2 or 3 at interval end

h1 = Hazard("exp", 1, 2)
h2 = Hazard("exp", 1, 3)

model = multistatemodel(h1, h2; data = dat, CensoringPatterns = CensoringPatterns)
@test isa(model, MultistateModels.MultistateProcess)
```

**Verifies:** State 0 is allowed in the **data** (not hazards) when paired with a `CensoringPatterns` row that enumerates the states compatible with each censoring code.

#### `test_non_contiguous_states` - Non-contiguous states cause issues

```julia
dat = DataFrame(
    statefrom = [1, 2, 4],  # Missing state 3
    stateto = [2, 4, 4],
    ...
)

h1 = Hazard("exp", 1, 2)
h2 = Hazard("exp", 2, 4)

@test_throws BoundsError multistatemodel(h1, h2; data = dat)
```

**Verifies:** System catches and warns about non-contiguous state spaces (gaps in state numbering)

#### `test_hazard_state_zero` - State 0 cannot be in hazard definitions

```julia
dat = DataFrame(
    id = [1],
    tstart = [0.0],
    tstop = [1.0],
    statefrom = [1],
    stateto = [1],
    obstype = [2]
)

h_bad_from = Hazard("exp", 0, 1)  # State 0 in statefrom
h_bad_to = Hazard("exp", 1, 0)    # State 0 in stateto
h_good = Hazard("exp", 1, 2)

@test_throws Exception multistatemodel(h_bad_from, h_good; data = dat)
@test_throws Exception multistatemodel(h_good, h_bad_to; data = dat)
```

**Verifies:** 
- Hazards cannot use state 0 in `statefrom` or `stateto`
- State 0 is reserved exclusively for censoring indicators in data

---

### STEP 5: Build Hazard Structs

**Code:**
```julia
_hazards, parameters, parameters_ph, hazkeys = build_hazards(hazards...; data = data)
```

**What it does:**
- Parses formulas and creates design matrices
- **Generates parameter names** from formula terms (not from data columns)
- Creates runtime-generated functions for hazard/cumulative hazard calculations
- Builds consolidated structs: `MarkovHazard` (exponential) or `SemiMarkovHazard` (Weibull/Gompertz)

**Key Logic for Parameter Naming:**
```julia
# StatsModels applies the schema, so coefnames reflects the actual design columns
coef_names = StatsModels.coefnames(applied_rhs)   # ["(Intercept)", "trt", "age", "trt & age"]
rhs_names = has_explicit_intercept ? coef_names : vcat("(Intercept)", coef_names)
parnames = Symbol.(replace.(hazname * "_" .* rhs_names, "(Intercept)" => "Intercept"))
```

- Formulas without explicit `+ 1` still receive an intercept column because the constructor prepends one when needed.
- Interaction terms retain the StatsModels naming convention (`"trt & age"`), so parameter vectors include symbols like `Symbol("h13_trt & age")`.
- Baseline-only hazards created via `Hazard("exp", 1, 2)` or `@hazard` without a formula only emit the intercept parameter (`:h12_Intercept`).

**Unit Test:** `test_parameter_naming`

```julia
# Exponential without covariates
dat = DataFrame(
    id = [1, 1],
    tstart = [0.0, 1.0],
    tstop = [1.0, 2.0],
    statefrom = [1, 1],
    stateto = [1, 1],
    obstype = [2, 2]
)

h1 = Hazard("exp", 1, 2)
model = multistatemodel(h1; data = dat)

@test :h12_Intercept in model.hazards[1].parnames
@test !(:h12_rate in model.hazards[1].parnames)  # NOT "rate"!

# Exponential with covariates
dat_cov = DataFrame(
    id = [1, 1],
    tstart = [0.0, 1.0],
    tstop = [1.0, 2.0],
    statefrom = [1, 1],
    stateto = [1, 1],
    obstype = [2, 2],
    age = [50, 51]
)

h2 = Hazard(@formula(0 ~ age), "exp", 1, 2)
model2 = multistatemodel(h2; data = dat_cov)

@test :h12_Intercept in model2.hazards[1].parnames
@test :h12_age in model2.hazards[1].parnames

# Interaction terms keep StatsModels' literal naming (including spaces and "&")
h3 = Hazard(@formula(0 ~ trt & age), "exp", 1, 2)
model3 = multistatemodel(h3; data = dat_cov)
@test Symbol("h12_trt & age") in model3.hazards[1].parnames
```

**Verifies:** 
- Parameters are named "Intercept" (not "rate") for consistency with GLM conventions
- Covariate names from the formula appear in parameter names
- Parameter naming works correctly for models with and without covariates, including interaction terms whose symbols match the StatsModels schema exactly

**Unit Test:** `linpred_effect modes`

```julia
@testset "linpred_effect modes" begin
    h_exp_aft = Hazard(@formula(0 ~ x), "exp", 1, 2; linpred_effect = :aft)
    model_exp_aft = multistatemodel(h_exp_aft; data = dat)
    set_parameters!(model_exp_aft, (h12 = [log(λ0), β],))
    @test call_haz(t, model_exp_aft.parameters[1], row, model_exp_aft.hazards[1]; give_log=false) ≈ λ0 * exp(-β * x)
    # similar assertions for Weibull/Gompertz
end
```

**Verifies:**
- The new `HazardMetadata.linpred_effect` flag is threaded through runtime hazard functions.
- Analytic families honor per-transition PH vs AFT semantics (exponential, Weibull, and Gompertz all have closed-form AFT adjustments for both hazard and cumulative hazard evaluations).
- The legacy interface (`call_haz`/`call_cumulhaz`) observes these effects without any downstream changes.

---

### STEP 6: Build Total Hazards

**Code:**
```julia
_totalhazards = build_totalhazards(_hazards, tmat)
```

**What it does:**
- For each origin state, creates a `_TotalHazard` object
- Absorbing states (no exits): `_TotalHazardAbsorbing()`
- Transient states (have exits): `_TotalHazardTransient(transition_indices)`
- Used for computing survival probabilities in likelihood calculations

**Testing:**
No dedicated unit test yet, but `_TotalHazard*` structs are instantiated in every `multistatemodel` call that the suite exercises. The shared-baseline cache regression tests in `test/test_hazards.jl` ("Shared baseline caches are reused across hazards from the same origin") plus the builder coverage in `test/test_modelgeneration.jl` exercise both absorbing and transient branches. TODO: add a lightweight direct test once we expand fixture coverage in FC2.

---

### STEP 7: Build Emission Matrix

**Code:**
```julia
emat = build_emat(data, CensoringPatterns, tmat)
```

**What it does:**
- For each observation, indicates which states are possible
- Exactly observed: `emat[i,state] = 1`, all others = 0
- Censored: `emat[i,:] = [1,1,1,...]` (all states possible)
- Custom censoring: Uses `CensoringPatterns` matrix to specify which states are possible

**Testing:**
`build_emat` is exercised by the panel-data and censoring suites (`test_state_zero_in_data`, `test_non_contiguous_states`, and the Tang cache regression tests). A stand-alone test would still be useful once we promote the shared fixture module to FC2 coverage.

---

### STEP 8: Determine Model Type

**Code:**
```julia
# Check observation types and hazard types to pick correct struct
if all(data.obstype .== 1)
    model = MultistateModel(...)  # Exact observations only
elseif all(isa.(_hazards, _MarkovHazard))
    model = MultistateMarkovModel(...)  # Markov process
elseif any(isa.(_hazards, _SemiMarkovHazard))
    model = MultistateSemiMarkovModel(...)  # Semi-Markov process
```

**What it does:**
- Examines data: exact vs panel vs censored observations
- Examines hazards: all Markov (exponential) vs any Semi-Markov (Weibull/Gompertz)
- Selects appropriate model struct from 6 total types:
  - `MultistateModel` (exact observations)
  - `MultistateMarkovModel` (Markov with panel data)
  - `MultistateSemiMarkovModel` (Semi-Markov with panel data)
  - `MultistateMarkovModelCensored` (Markov with censoring)
  - `MultistateSemiMarkovModelCensored` (Semi-Markov with censoring)

**Testing:**
Implicitly tested by all model construction tests, which create models with different data types and hazard families.

---

## Summary of Test Coverage

| Model Generation Step | Unit Test | What It Verifies | Status |
|----------------------|-----------|------------------|--------|
| **Hazard creation** | `test_hazard_construction` | Correct parameter counts, structures, and naming (including optional formulas) | ✅ |
| **Duplicate checking** | `test_duplicate_transitions` | Catches duplicate transitions with clear error | ✅ |
| **Transition matrix** | `test_tmat` | Correct structure and numbering | ✅ |
| **State 0 in data** | `test_state_zero_in_data` | State 0 allowed for censoring | ✅ |
| **State 0 in hazards** | `test_hazard_state_zero` | State 0 NOT allowed in hazard definitions | ✅ |
| **Non-contiguous states** | `test_non_contiguous_states` | Warning/error for gaps in state space | ✅ |
| **Parameter naming** | `test_parameter_naming` | Uses `Intercept`, includes covariate/interaction names from formulas | ✅ |

**Total: 19 tests passing ✅**

## Key Design Decisions

### Parameter Naming Convention

Parameters are named using `"Intercept"` rather than `"rate"` to maintain consistency with GLM conventions in Julia. This makes the package more intuitive for users familiar with regression modeling.

**Example:**
- ❌ Old: `h12_rate`
- ✅ New: `h12_Intercept`

### State 0 Reserved for Censoring

State 0 is reserved exclusively for censoring indicators in the data and cannot be used in hazard definitions:
- ✅ Allowed: `stateto = 0` with `obstype` indicating censoring
- ❌ Not allowed: `Hazard("exp", 0, 1)`
- ❌ Not allowed: `Hazard("exp", 1, 0)`

This design ensures a clear separation between actual state transitions and censoring mechanisms.

### Non-Contiguous States

While the system will attempt to construct models with non-contiguous states (e.g., states 1, 2, 4 skipping 3), this can lead to errors in indexing. The data validation step warns users about this potential issue.

### Duplicate Transition Detection

The system explicitly checks for and rejects duplicate transition specifications. If a user accidentally defines the same transition twice (e.g., two hazards for 1→2), the system throws a clear error message indicating which transition was duplicated.

## Test File Location

All model generation tests are located in:
```
test/test_modelgeneration.jl
```

Run tests with:
```julia
using Pkg
Pkg.test("MultistateModels")
```

Or run just model generation tests:
```julia
include("test/test_modelgeneration.jl")
```

---

## Roadmap: Unified DSL + ODE / Neural Hazards

This section captures the forward-looking design so Copilot (and contributors) understand how Tang-style ODE hazards, SODEN-style neural RHS, and existing analytic hazards will coexist behind one DSL. These notes guide upcoming refactors before implementation begins.

#### Current Status (Nov 25, 2025)

- ✅ Shared baseline metadata + cache plumbing landed on `infrastructure_changes` and is exercised via `test/test_hazards.jl`.
- ✅ Tang time-transform gating + `TimeTransformContext` helpers are live in exact-path likelihoods; Markov panels still bypass transforms.
- 🚧 Dedicated ODE solves + SciML adjoints: design settled, implementation deferred to FC3 after the model-generation test suite is broadened.
- 💤 Neural RHS integration remains on the roadmap only; no code yet.

### Goals

- **Single DSL surface**: extended `Hazard` constructor (and wrappers) must express classic analytic families, Tang-type ODE models, and future neural RHS without parallel APIs.
- **Full compatibility**: MCEM, path simulation, surrogate building, and likelihood code must call hazards through the same interfaces (`call_haz`, `call_cumulhaz`, `_TotalHazard*`). No algorithm should branch on hazard type.

### Tang-style time transforms (current scope)

- The `time_transform=true` flag is interpreted strictly per hazard. Runtime helpers now gate transforms behind an `apply_transform` keyword so that only code paths that explicitly opt in will ever see the modified trajectories.
- Exact-path likelihoods (`loglik_path`) are the only callers that currently pass `apply_transform=true`, and they do so only when the originating state's outgoing hazards contain at least one Tang-enabled transition. This keeps shared-trajectory mechanics local to the contexts that motivated them.
- Markov likelihood machinery (including FFBS and other panel-data workflows) always leaves `apply_transform=false`. Those routines call hazards through the same APIs but will never touch the transformed branch, so their cached Kolmogorov / matrix-exponential logic stays valid.
- Exact-path likelihoods now build a short-lived `TimeTransformContext` that houses per-hazard caches keyed by the linear predictor and time bounds. The caches exist only for the duration of a `loglik_path` evaluation, so there is no mutation stored on the hazard structs themselves. Markov/FFBS paths continue to rely on their matrix-based caches and never touch the transform branch. Shared trajectory caches that span multiple hazards are still on the roadmap and will build atop this layer.
- `enable_time_transform_cache!(false)` disables caching altogether for workflows that rely on mutation-intolerant AD backends (Zygote, Enzyme). When disabled, the exact log-likelihood still honors Tang transforms; it just recomputes the separable pieces each time.
- `maybe_time_transform_context(parameters, subjectdata, hazards)` is exposed so callers outside the default likelihood stack can create the exact same `TimeTransformContext` without reimplementing the detection logic. Pass a DataFrame (or anything with a `:sojourn` column) plus the current parameter vectors and hazard list; receive `nothing` when no Tang-enabled hazards exist.
- **Shared-baseline metadata:** Each hazard now tracks whether it can participate in a shared Tang trajectory via the tuple `(statefrom, baseline_signature)`. The signature hashes every ingredient that shapes α(t): family, spline definition, knots, and spline hyperparameters. If multiple hazards emit the same signature they can reuse a baseline trajectory cache keyed solely by the origin state. Hazards with `time_transform=false` or unique signatures continue to use the per-hazard caches only.
- **SciML-ready**: integrate DifferentialEquations.jl + SciMLSensitivity incrementally, allowing numerical cumulative hazards / adjoints when analytic forms are unavailable.
- **Maintainable**: keep intermediate objects (time transforms, solver caches) structured and memoized so the code base remains readable and easy to extend.

### Unified DSL Specification

1. **Formula-based declaration (default)**
    ```julia
    Hazard(@formula(0 ~ age + trt), :ode, 1, 2;
             baseline = :alpha_spline,
             q = :exp_link,
             time_transform = true,
             solver = :auto)
    ```
    - `family` keyword accepts existing strings (`"exp"`, `"wei"`, `"gom"`, `"sp"`) plus symbolic tags (`:ode`, `:ode_neural`).
    - Additional keywords choose Tang-style components (baseline α(t), q(Λ), separable time transform) and solver preferences.
    - Legacy hazards become thin wrappers that auto-populate these keywords (e.g., exponential = `:ode` with constant α and q=1).

2. **Programmatic declaration (for advanced users / neural RHS)**
    ```julia
    HazardODE(
        rhs = (Λ, t, covs, pars) -> pars.α(t) * exp(dot(pars.β, covs)) *
                               Lux.apply(pars.net, vcat(Λ, t, covs), pars.net_ps, pars.net_st)[1],
        from = 1,
        to = 2,
        covars = (:age, :trt),
        solver = (:Tsit5, abstol = 1e-8, reltol = 1e-8),
        adjoint = :backsolve
    )
    ```
    - Users provide a callable (pure Julia function or Lux/DiffEqFlux neural layer) obeying the Tang/SODEN separable form when `time_transform=true`.
    - Parameters/states stay explicit (Lux requirement) and flow through ParameterHandling/ComponentArrays.

3. **`linpred_effect` (per-transition PH vs AFT)**
    ```julia
    # Proportional hazards (default)
    Hazard(@formula(0 ~ age + trt), :ode, 1, 2;
           linpred_effect = :ph)

    # Accelerated failure time (per-transition override)
    Hazard(@formula(0 ~ age + trt), :ode, 1, 2;
           linpred_effect = :aft)
    ```
    - `:ph` multiplies the baseline hazard by `exp(Xβ)`; `:aft` rescales the time axis using `exp(-Xβ)` inside the time transform cache.
    - The keyword is optional (default `:ph`), scoped to a single transition, and propagated through the legacy wrappers so existing exponential/Weibull helpers stay zero-config.
    - MCEM, path simulation, and surrogate builders ignore this flag because they only observe the resulting `hazard_fn/cumhaz_fn` callables.

    | Setting | How covariates enter | Time transform behavior | When to use |
    |---------|----------------------|-------------------------|-------------|
    | `:ph` (default) | `λ(t | X) = λ₀(t) · exp(Xβ)` | Standard Tang cache; no rescaling beyond shared baseline | Classic proportional hazards work, aligns with existing parametric families |
    | `:aft` | `λ(t | X) = λ₀(t·exp(-Xβ)) · exp(-Xβ)` | Time grid scaled per subject using `exp(-Xβ)` before evaluating α(t) | Per-transition accelerated failure time effects without leaving Tang solver stack |

    > The DSL keeps mention of the third option (purely neural covariate handling) as a TODO for when SODEN-style RHS are the default; see the Neural RHS section below.

### Hazard Runtime Expectations

- `_Hazard` structs gain new fields: `solver_kind`, `ode_problem_builder`, `time_transformable::Bool`, and room for future shared-trajectory metadata once that roadmap item begins.
- `build_hazards` detects analytic families and injects closed-form `hazard_fn/cumhaz_fn` as today. For ODE hazards it constructs `ODEProblem`s (per transition) and registers SciML solvers **unless** the parametric Tang-style specification signals that a closed-form cumulative hazard exists (in which case we continue to emit analytic callables).
- Shared DSL ensures `haznames`, parameter names, and covariate metadata remain consistent, simplifying downstream code.

#### Shared baseline cache design (Step 1 checkpoint)

- **SharedBaselineKey**: `(statefrom::Int, baseline_signature::UInt64)` uniquely names the baseline trajectory that multiple hazards may reuse. The signature is deterministic for a given hazard family/degree/knots so that structurally identical hazards share a slot even across different models.
- **Shared cache selection**: when a hazard advertises a `SharedBaselineKey`, the runtime routes its Tang cache lookups through the `SharedBaselineTable`, so every matching transition shares the same `TimeTransformCache`. Hazards without a key (or states with mixed baselines) continue to use their dedicated cache slots.
- **Hazard metadata attachment**: `_Hazard` structs keep both the `SharedBaselineKey` (for grouping) and a boolean `time_transformable`. Builders populate these fields early so runtime code can route evaluations without inspecting hazard internals.
- **Context layout**: `TimeTransformContext` now holds both the existing per-hazard caches and a `SharedBaselineTable`. Exact-path likelihoods pull from the shared table automatically whenever all prerequisites are met; the table stays short-lived (per likelihood call) to avoid global mutation.
- **Fallback guarantee**: if any prerequisite is missing—mixed signatures, disabled cache flag, or hazards that skip Tang transforms—we automatically fall back to the per-hazard cache to keep behavior identical to today.

### Likelihood Evaluation Modes

1. **Per-subject solves (default)**
    - `total_cumulhaz` composes cumulative hazards by calling either analytic integrals or solving the ODE for each subject/time interval. Solver workspaces/cache objects live inside hazard structs to avoid reallocations.
    - MCEM/path sampling reuse the same call sites, ensuring identical numerics.

2. **Shared trajectory (Tang time transform)**
        - Applicable only when hazard RHS is separable (`f(t, Λ, x)=f₁(t, x)·f₂(Λ, x)`) and when all outgoing transitions from a state share the same baseline α(t).
        - Workflow:
            1. Transitions with identical baselines reuse a single `TimeTransformCache` via their shared baseline key.
            2. Each cache stores Tang hazard/cumulative-hazard evaluations keyed by `(linpred, time window)`, so repeated calls (across hazards or jumps) simply reuse the stored result.
        - Relies on the per-evaluation `TimeTransformContext` plus the new `SharedBaselineTable`. Keep these objects lightweight and rebuildable so maintenance stays simple.
        - Likelihood API exposes `likelihood_mode = :per_subject | :shared_traj`; automatically falls back to per-subject if constraints fail.

### SciML / Adjoint Integration

- Hazards gain solver configs referencing DifferentialEquations.jl solvers (Tsit5, Vern9, Rosenbrock23, etc.).
- Gradients route through SciMLSensitivity.jl adjoints (BacksolveAdjoint, QuadratureAdjoint) or ForwardDiff if faster. Each hazard declares its supported adjoint to avoid misuse.
- Dependencies are added progressively (initially optional). If SciML features are disabled, the DSL restricts to analytic families only.

### Neural RHS (SODEN readiness)

- Neural `f(t, Λ, x)` implemented via Lux/DiffEqFlux `NeuralODE` layers. Parameters/states reside in ComponentArrays so they interoperate with Optimization.jl and existing parameter utilities.
- Device selection: default CPU, but hazard definitions include `device = :cpu | :gpu | :auto`; future work can enable LuxCUDA/Reactant.
- Time-rescaling/batching: borrow SODEN’s strategy to align observation grids, reducing repeated solves.
- `linpred_effect` flags are ignored once a hazard opts into `:ode_neural`; covariate effects live entirely inside the neural RHS, so the DSL simply plumbs the network outputs into the solver without assuming PH or AFT structure.
- TODO: document the neural-only covariate mode (the planned "third option") with a worked example once SODEN support ships.

### MCEM & Path Simulation Compatibility

- `DrawSamplePaths!`, `sample_ecctmc`, and surrogate builders consume hazards only through exported callables. As long as hazards expose `hazard_fn`, `cumhaz_fn`, and optional `shared_traj` caches, existing sampling code works unchanged.
- Surrogate models: when ODE hazards are present, default surrogate may still use exponential approximations unless the user requests ODE surrogates. Document trade-offs explicitly.
- Importance weights: computed using the same hazard interface as likelihood evaluation to avoid bias.

### Implementation Phases (high-level)

1. **Infrastructure prep**: refactor hazard structs/builders to accept solver metadata; no behavior change yet.
2. **ODE cumulative hazards**: add per-subject SciML solves + caching; prove parity with analytic hazards.
3. **Shared trajectory mode**: implement Tang time-transform caches; add validation and configuration hooks.
4. **Neural RHS support**: integrate Lux/DiffEqFlux-based RHS, including parameter handling and adjoints.
5. **Documentation & tests**: extend unit/integration tests to cover ODE hazards, shared trajectories, and neural examples; update user guides/tutorials.

## Tracking Checklist

| Plan Item | Issue / PR | Status |
|-----------|------------|--------|
| FC1 – Relocate legacy longtests | Draft PR `infrastructure_changes -> main` (includes deprecated test move) | ✅ Completed locally; ship with the next PR |
| FC2 – Shared fixtures + expanded model-generation coverage | GitHub issue **TBD-fc2-shared-fixtures** (open before merging) | 🚧 In progress (fixtures landed; more suites to port) |
| FC3 – Simulation transforms + ODE/neural hazards | GitHub issue **TBD-fc3-simulation-transforms** (blocked on FC2) | ⏳ Not started; depends on FC2 sign-off |

> Replace the `TBD-*` placeholders with real issue numbers/links once you create the tickets so this table stays authoritative.

### Notes / Open Items

- Dependencies (DifferentialEquations.jl, SciMLSensitivity.jl, Lux.jl, DiffEqFlux.jl) are acceptable but should be introduced in the phase they are needed, possibly via package extensions to keep the base install light.
- Tang/SODEN separability is the most complex modeling scenario we plan to support; more exotic models (e.g., non-separable neural CDFs per Rindt/Danks) remain future work and should be revisited once the unified DSL is stable.
- Code simplicity remains paramount: intermediate caches (time transforms, shared baselines) must have clear ownership, minimal mutation, and strong tests to make maintenance easy.
