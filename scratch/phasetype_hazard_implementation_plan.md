# Implementation Plan: `:pt` Phase-Type Hazard Family

**Status**: Planning  
**Created**: 2024-12-06  
**Branch**: `infrastructure_changes`

---

## Table of Contents

1. [Overview](#overview)
2. [Key Design Decisions](#key-design-decisions)
3. [Architecture](#architecture)
4. [New Types](#new-types)
5. [New Functions](#new-functions)
6. [Modified Functions](#modified-functions)
7. [Simulation Updates](#simulation-updates)
8. [`draw_paths` Consolidation](#draw_paths-consolidation)
9. [Initialization Strategy](#initialization-strategy)
10. [Implementation Order](#implementation-order)
11. [Testing Plan](#testing-plan)
12. [Progress Tracking](#progress-tracking)

---

## Overview

The `:pt` hazard family allows users to specify **Coxian phase-type sojourn time distributions** for transitions in multistate models. Unlike MCEM phase-type *proposals* (which approximate non-Markovian models during importance sampling), the `:pt` family defines the **true model** — transitions proceed through latent phases with Markovian dynamics.

### User-Facing API

```julia
# Basic phase-type hazard with 3 phases
h12 = Hazard(:pt, 1, 2; n_phases=3)

# With covariates
h12 = Hazard(@formula(0 ~ age + treatment), :pt, 1, 2; n_phases=2)

# Mixed model (Markov - uses panel likelihood)
model = multistatemodel(
    Hazard(:exp, 1, 2),
    Hazard(:pt, 2, 3; n_phases=2);
    data = df
)

# Semi-Markov with phase-type (requires MCEM)
model = multistatemodel(
    Hazard(:wei, 1, 2),
    Hazard(:pt, 2, 3; n_phases=2);
    data = df
)
```

---

## Key Design Decisions

### 1. Type Hierarchy

- `PhaseTypeCoxianHazard <: _MarkovHazard`: Phase-type hazards are Markovian on the expanded state space
- Models with only `:exp` and/or `:pt` hazards are classified as **Markov** (no MCEM needed)

### 2. Model Classification Rules

| Hazard Types Present | Model Class | Fitting Method |
|---------------------|-------------|----------------|
| `:exp` only | Markov | Panel MLE |
| `:pt` only | Markov | Panel MLE |
| `:exp` + `:pt` | Markov | Panel MLE |
| `:sp` with `degree=0` | Markov | Panel MLE |
| `:wei`, `:gom`, `:sp` with `degree>0` | Semi-Markov | MCEM |
| Any above + `:pt` | Semi-Markov | MCEM |

### 3. Surrogate Defaults for Semi-Markov Models with `:pt`

- Default surrogate for `:pt` hazards: **phase-type surrogate with same n_phases**
- User can override via `surrogate=:markov` to use exponential surrogate instead

### 4. Initialization Strategy

| Model Type | Strategy |
|------------|----------|
| Markov (`:exp`/`:pt` only) | Crude rates |
| Semi-Markov | Fit surrogate first, use surrogate parameters |

### 5. Simulation Output Options

- Default: Return data/paths on **collapsed** (observed) state space
- Optional: Return on **expanded** (phase-level) state space via `expanded=true`

---

## Architecture

### Coxian Phase-Type Structure

For transition `s → d` with `n` phases:

```
                    ┌──────── μ₁ ────────┐
                    │                    │
                    ▼                    │
Obs State s:   [Phase 1] ──λ₁──> [Phase 2] ──λ₂──> ... ──> [Phase n]
                                     │                         │
                                    μ₂                        μₙ
                                     │                         │
                                     ▼                         ▼
Obs State d:                   [Phase 1] <─────────────────────┘
```

From any phase i in state s:
- **Progress** to phase i+1 within state s (rate λᵢ, for i < n)
- **Exit** to first phase of destination state d (rate μᵢ)

Last phase (n) has only exit via μₙ (no progression).

### Parameterization

For transition `s → d` with `n` phases:

| Parameter Type | Count | Names |
|---------------|-------|-------|
| Progression rates | n-1 | `h{s}{d}_λ1`, ..., `h{s}{d}_λ{n-1}` |
| Exit rates | n | `h{s}{d}_μ1`, ..., `h{s}{d}_μn` |
| **Baseline total** | **2n-1** | |
| Covariates | varies | `h{s}{d}_{covar_name}` |

All parameters stored on **log scale** internally.

---

## New Types

### 1. `PhaseTypeHazardSpec <: HazardFunction`

**File**: `src/common.jl`

User-facing specification created by `Hazard(:pt, ...)`:

```julia
"""
Specification for a phase-type (Coxian) hazard.
"""
struct PhaseTypeHazardSpec <: HazardFunction
    hazard::StatsModels.FormulaTerm   # formula for covariates
    family::String                     # "pt"
    statefrom::Int64
    stateto::Int64
    n_phases::Int                      # number of Coxian phases (≥1)
    metadata::HazardMetadata
end
```

### 2. `PhaseTypeCoxianHazard <: _MarkovHazard`

**File**: `src/common.jl`

Runtime hazard — subtype of `_MarkovHazard`:

```julia
"""
Runtime phase-type hazard for the expanded Markov model.

Subtypes `_MarkovHazard` because the expanded model is Markovian.
"""
struct PhaseTypeCoxianHazard <: _MarkovHazard
    hazname::Symbol
    statefrom::Int64                 # observed state from
    stateto::Int64                   # observed state to
    family::String                   # "pt"
    n_phases::Int
    parnames::Vector{Symbol}         # [λ₁, ..., λₙ₋₁, μ₁, ..., μₙ, covariates...]
    npar_baseline::Int               # 2n - 1
    npar::Int                        # baseline + covariates
    hazfun::Function                 # total hazard out of current phase
    cumhazfun::Function              # cumulative hazard
    has_covariates::Bool
    parnames_aliases::Vector{Symbol}
    metadata::HazardMetadata
    shared_baseline_key::Union{Nothing, SharedBaselineKey}
    
    # Phase-type specific
    expanded_phase_from::Vector{Int}  # expanded state indices for phases
    expanded_phase_to::Int            # first phase of destination
    progression_param_indices::UnitRange{Int}  # indices of λ params
    exit_param_indices::UnitRange{Int}         # indices of μ params
end
```

### 3. `PhaseTypeMappings`

**File**: `src/phasetype.jl`

Bidirectional state space mappings:

```julia
"""
Mappings between observed and expanded state spaces.
"""
struct PhaseTypeMappings
    n_observed::Int
    n_expanded::Int
    n_phases_per_state::Vector{Int}
    state_to_phases::Vector{UnitRange{Int}}
    phase_to_state::Vector{Int}
    expanded_tmat::Matrix{Int}
    original_hazards::Vector{<:HazardFunction}
    original_tmat::Matrix{Int}
    expanded_hazard_indices::Dict{Symbol, Vector{Int}}
    pt_hazard_indices::Vector{Int}
end
```

### 4. `PhaseTypeModel <: MultistateProcess`

**File**: `src/common.jl`

Wrapper containing expanded model plus mappings:

```julia
"""
A multistate model with phase-type hazards.
"""
mutable struct PhaseTypeModel <: MultistateProcess
    expanded_model::MultistateProcess
    mappings::PhaseTypeMappings
    modelcall::NamedTuple
    
    # Forward common fields from expanded_model
    data::DataFrame                   # original (collapsed) data
    expanded_data::DataFrame          # expanded data for fitting
    parameters::NamedTuple            # in phase-type parameterization
    expanded_parameters::NamedTuple   # in expanded Markov parameterization
end
```

### 5. `PhaseTypeFittedModel`

**File**: `src/modeloutput.jl`

Fitted result with phase-type interpretation:

```julia
"""
Fitted phase-type model with results in both parameterizations.
"""
struct PhaseTypeFittedModel
    fitted_expanded::MultistateModelFitted
    mappings::PhaseTypeMappings
    parameters::NamedTuple            # phase-type params (λ, μ)
    vcov::Union{Nothing, Matrix}      # variance-covariance in PT params
    loglik::Float64
end
```

---

## New Functions

### State Space Expansion

#### `build_phasetype_mappings()`
**File**: `src/phasetype.jl`

```julia
"""
    build_phasetype_mappings(hazards, tmat) -> PhaseTypeMappings

Build state space mappings from hazard specifications.
Determines n_phases per state from outgoing :pt hazards.
"""
function build_phasetype_mappings(hazards::Vector{<:HazardFunction}, 
                                  tmat::Matrix{Int}) -> PhaseTypeMappings
```

#### `build_expanded_tmat()`
**File**: `src/phasetype.jl`

```julia
"""
    build_expanded_tmat(mappings) -> Matrix{Int}

Build transition matrix for expanded state space.
"""
function build_expanded_tmat(original_tmat::Matrix{Int},
                             mappings::PhaseTypeMappings) -> Matrix{Int}
```

### Hazard Expansion

#### `expand_hazards_for_phasetype()`
**File**: `src/phasetype.jl`

```julia
"""
    expand_hazards_for_phasetype(hazards, mappings, data) -> Vector{_Hazard}

Convert user hazard specs to runtime hazards on expanded space.

For :pt hazards: Creates PhaseTypeCoxianHazard
For :exp hazards: Maps to expanded space as MarkovHazard
For :wei/:gom/:sp: Creates on expanded space
"""
function expand_hazards_for_phasetype(hazards::Vector{<:HazardFunction},
                                      mappings::PhaseTypeMappings,
                                      data::DataFrame) -> Vector{_Hazard}
```

#### `_build_phasetype_hazard()`
**File**: `src/modelgeneration.jl`

```julia
"""
Builder function for phase-type hazards.
Registered with _HAZARD_BUILDERS["pt"].
"""
function _build_phasetype_hazard(ctx::HazardBuildContext)
    # Creates PhaseTypeCoxianHazard with proper structure
end
```

### Data Expansion

#### `expand_data_for_phasetype_fitting()`
**File**: `src/phasetype.jl`

```julia
"""
    expand_data_for_phasetype_fitting(data, mappings) -> (expanded_data, censoring_patterns)

Expand observation data to handle phase uncertainty during fitting.
"""
function expand_data_for_phasetype_fitting(data::DataFrame, 
                                           mappings::PhaseTypeMappings)
```

### Model Building

#### `build_phasetype_model()`
**File**: `src/phasetype.jl`

```julia
"""
    build_phasetype_model(hazards, data; kwargs...) -> PhaseTypeModel

Main entry point for constructing phase-type models.

# Algorithm
1. Build state mappings from :pt hazards
2. Expand data for phase uncertainty
3. Create expanded hazards
4. Build underlying model (Markov or SemiMarkov)
5. Wrap with mappings
"""
function build_phasetype_model(hazards::Vector{<:HazardFunction}, 
                               data::DataFrame; 
                               constraints = nothing,
                               surrogate = :auto,
                               kwargs...) -> PhaseTypeModel
```

### Parameter Management

#### `collapse_phasetype_parameters()`
**File**: `src/phasetype.jl`

```julia
"""
    collapse_phasetype_parameters(expanded_params, mappings) -> NamedTuple

Map expanded Markov parameters to phase-type (λ, μ) parameterization.
"""
function collapse_phasetype_parameters(expanded_params, 
                                       mappings::PhaseTypeMappings)
```

#### `expand_phasetype_parameters()`
**File**: `src/phasetype.jl`

```julia
"""
    expand_phasetype_parameters(pt_params, mappings) -> Vector{Vector{Float64}}

Map phase-type parameters to expanded Markov parameterization.
"""
function expand_phasetype_parameters(pt_params::NamedTuple,
                                     mappings::PhaseTypeMappings)
```

### Initialization

#### `init_par(::PhaseTypeCoxianHazard, ...)`
**File**: `src/initialization.jl`

```julia
"""
Initialize parameters for phase-type hazards.

Strategy: Set all rates to crude_rate so mean sojourn ≈ 1/crude_rate.
"""
function init_par(hazard::PhaseTypeCoxianHazard, crude_log_rate=0.0)
    n = hazard.n_phases
    ncovar = hazard.npar - hazard.npar_baseline
    
    # All progression and exit rates initialized equally
    progression = fill(crude_log_rate, n - 1)
    exit_rates = fill(crude_log_rate, n)
    baseline = vcat(progression, exit_rates)
    
    return hazard.has_covariates ? vcat(baseline, zeros(ncovar)) : baseline
end
```

#### `set_crude_init!(::PhaseTypeModel, ...)`
**File**: `src/initialization.jl`

```julia
"""
Initialize phase-type model using crude transition rates.
"""
function set_crude_init!(model::PhaseTypeModel; constraints = nothing)
```

#### `initialize_parameters!(::PhaseTypeModel, ...)`
**File**: `src/initialization.jl`

```julia
"""
Initialize phase-type model parameters.

Markov models: Use crude rates.
Semi-Markov models: Fit surrogate first, use surrogate parameters.
"""
function initialize_parameters!(model::PhaseTypeModel; 
                                constraints = nothing, 
                                surrogate_constraints = nothing,
                                surrogate_parameters = nothing,
                                crude = false)
```

### Utilities

#### `phasetype_constraints()`
**File**: `src/phasetype.jl`

```julia
"""
    phasetype_constraints(hazname, n_phases, structure) -> NamedTuple

Generate constraint expressions for common Coxian structures.

# Structures
- `:allequal`: All λᵢ equal, all μᵢ equal
- `:prop_to_prog`: μᵢ ∝ λᵢ (Titman-Sharples)
- `:unstructured`: No constraints (default)

# Returns
NamedTuple with `cons`, `lcons`, `ucons` for make_constraints()
"""
function phasetype_constraints(hazname::Symbol, n_phases::Int, 
                               structure::Symbol) -> NamedTuple
```

#### `_is_markov_hazard()`
**File**: `src/modelgeneration.jl`

```julia
"""
Check if a hazard is Markovian.

Returns true for:
- MarkovHazard (family="exp")
- PhaseTypeCoxianHazard (family="pt")
- RuntimeSplineHazard with degree=0
"""
function _is_markov_hazard(hazard::_Hazard) -> Bool
```

---

## Modified Functions

### `Hazard()` Constructor
**File**: `src/modelgeneration.jl`

Add `:pt` to valid families, handle `n_phases` keyword:

```julia
function Hazard(hazard, family, statefrom, stateto;
                n_phases::Int = 2,  # NEW
                ...)
    
    valid_families = ("exp", "wei", "gom", "sp", "pt")  # ADD "pt"
    
    if family_key == "pt"
        @assert n_phases >= 1 "n_phases must be >= 1"
        return PhaseTypeHazardSpec(hazard, family_key, statefrom, stateto, 
                                   n_phases, metadata)
    end
    # ... rest unchanged
end
```

### `_process_class()`
**File**: `src/modelgeneration.jl`

Update to use `_is_markov_hazard()`:

```julia
@inline function _process_class(hazards::Vector{<:_Hazard})
    return all(_is_markov_hazard.(hazards)) ? :markov : :semi_markov
end
```

### `multistatemodel()`
**File**: `src/modelgeneration.jl`

Detect `:pt` hazards and route:

```julia
function multistatemodel(hazards...; data, constraints=nothing,
                         surrogate::Symbol = :auto, ...)
    
    has_pt = any(h -> h isa PhaseTypeHazardSpec, hazards)
    
    if has_pt
        return build_phasetype_model(collect(hazards), data; 
                                     constraints = constraints,
                                     surrogate = surrogate,
                                     kwargs...)
    end
    # ... existing logic
end
```

### `fit()` Dispatch
**File**: `src/modelfitting.jl`

Add dispatch for `PhaseTypeModel`:

```julia
function fit(model::PhaseTypeModel; constraints=nothing, verbose=true, ...)
    # Fit expanded model
    fitted_expanded = fit(model.expanded_model; constraints=constraints, ...)
    
    # Collapse to phase-type parameterization
    return PhaseTypeFittedModel(fitted_expanded, model.mappings)
end
```

### Parameter Accessors
**File**: `src/helpers.jl`

Add methods for `PhaseTypeModel`:

```julia
get_parameters(model::PhaseTypeModel) = model.parameters
get_parameters_flat(model::PhaseTypeModel) = get_parameters_flat(model.expanded_model)

function set_parameters!(model::PhaseTypeModel, newvalues)
    expanded = expand_phasetype_parameters(newvalues, model.mappings)
    set_parameters!(model.expanded_model, expanded)
    # Update model.parameters
end
```

---

## Simulation Updates

### New Keyword: `expanded`

Add `expanded::Bool = false` to simulation functions:

```julia
function simulate(model::PhaseTypeModel; 
                  nsim = 1, 
                  data = true, 
                  paths = false,
                  expanded::Bool = false,  # NEW
                  ...)
```

### Behavior

| `expanded` | Output |
|------------|--------|
| `false` (default) | Data/paths on observed state space |
| `true` | Data/paths on expanded (phase-level) state space |

### Implementation

#### `simulate(::PhaseTypeModel, ...)`
**File**: `src/simulation.jl`

```julia
function simulate(model::PhaseTypeModel; 
                  nsim = 1, data = true, paths = false,
                  expanded::Bool = false, ...)
    
    if expanded
        # Simulate on expanded model directly
        return simulate(model.expanded_model; nsim=nsim, data=data, paths=paths, ...)
    else
        # Simulate on expanded, then collapse
        result = simulate(model.expanded_model; nsim=nsim, data=data, paths=paths, ...)
        return _collapse_simulation_result(result, model.mappings, data, paths)
    end
end
```

#### `_collapse_simulation_result()`
**File**: `src/simulation.jl`

```julia
"""
Collapse simulation results from expanded to observed state space.
"""
function _collapse_simulation_result(result, mappings::PhaseTypeMappings,
                                     return_data::Bool, return_paths::Bool)
    # Map phase states back to observed states
    # Merge consecutive intervals in same observed state
end
```

#### `_collapse_path()`
**File**: `src/simulation.jl`

```julia
"""
Collapse a single SamplePath from expanded to observed states.
"""
function _collapse_path(path::SamplePath, mappings::PhaseTypeMappings) -> SamplePath
    # Map states: phase_to_state
    # Merge consecutive same-state intervals
end
```

#### `_collapse_data()`
**File**: `src/simulation.jl`

```julia
"""
Collapse a simulated DataFrame from expanded to observed states.
"""
function _collapse_data(df::DataFrame, mappings::PhaseTypeMappings) -> DataFrame
    # Map statefrom/stateto using phase_to_state
    # Merge consecutive rows in same observed state
end
```

---

## `draw_paths` Consolidation

### Current State

There are currently **two** `draw_paths` functions with overlapping functionality:

1. `draw_paths(model; min_ess, ...)` — adaptive sampling until ESS target reached
2. `draw_paths(model, npaths; ...)` — fixed number of paths

Both have significant code duplication and neither supports phase-type models properly.

### Issues to Address

1. **Code duplication**: ~150 lines of nearly identical surrogate setup, FFBS, importance weighting
2. **Surrogate redundancy**: When given a `MultistateModelFitted`, the function fits a new surrogate even though the fitted model already stores `markovsurrogate`
3. **No expanded output option**: No way to get paths on the phase-type expanded state space
4. **No phase-type proposal support**: Need to integrate phase-type FFBS for `:pt` hazards

### Design

#### Consolidated Signature

```julia
"""
    draw_paths(model::MultistateProcess; 
               min_ess=100, 
               npaths=nothing,
               paretosmooth=true, 
               return_logliks=false,
               expanded::Bool=false)

Draw sample paths from a fitted or unfitted multistate model conditional on observed data.

# Sampling Mode
- If `npaths` is `nothing` (default): Adaptive sampling until `min_ess` achieved
- If `npaths` is an integer: Draw exactly `npaths` paths per subject

# Arguments
- `model`: MultistateProcess (fitted or unfitted)
- `min_ess`: Target effective sample size for adaptive mode (default: 100)
- `npaths`: Fixed number of paths (overrides adaptive sampling)
- `paretosmooth`: Apply Pareto smoothing to importance weights (default: true)
- `return_logliks`: Include log-likelihoods in output (default: false)
- `expanded`: Return paths on expanded state space for phase-type models (default: false)
- `newdata`: Optional new data template for path sampling (same columns as model data)

# Returns
NamedTuple with:
- `samplepaths`: Vector of SamplePath vectors, one per subject
- `ImportanceWeightsNormalized`: Normalized importance weights
- `loglik_target`, `loglik_surrog`, `subj_ess` (if `return_logliks=true`)
"""
function draw_paths(model::MultistateProcess; 
                    min_ess::Int = 100,
                    npaths::Union{Nothing, Int} = nothing,
                    paretosmooth::Bool = true,
                    return_logliks::Bool = false,
                    expanded::Bool = false,
                    newdata::Union{Nothing, DataFrame} = nothing)
```

#### Key Implementation Details

##### Surrogate Reuse for Fitted Models

```julia
function _get_or_fit_surrogate(model::MultistateProcess, is_semimarkov::Bool)
    if !is_semimarkov
        return nothing  # Markov models don't need surrogate
    end
    
    # Fitted models should already have surrogate stored
    if model isa MultistateModelFitted && !isnothing(model.markovsurrogate)
        return model.markovsurrogate
    end
    
    # Unfitted models or fitted without stored surrogate: fit now
    if !isnothing(model.markovsurrogate)
        return model.markovsurrogate
    end
    
    # Last resort: fit a new surrogate
    surrogate_fitted = fit_surrogate(model; verbose=false)
    return MarkovSurrogate(surrogate_fitted.hazards, surrogate_fitted.parameters)
end
```

##### Core Sampling Loop (Shared)

```julia
function _draw_paths_core!(
    samplepaths, loglik_target, loglik_surrog, ImportanceWeights, subj_ess,
    model, tpm_book, hazmat_book, books, fbmats, absorbingstates,
    params_target, hazards_target, params_surrog, hazards_surrog,
    is_semimarkov, paretosmooth, min_ess, npaths;
    # Phase-type infrastructure (optional)
    phasetype_surrogate=nothing, tpm_book_ph=nothing, hazmat_book_ph=nothing, 
    fbmats_ph=nothing, emat_ph=nothing, expanded_data=nothing,
    expanded_subjectindices=nothing, expanded_tpm_map=nothing)
    
    # Unified sampling loop for both adaptive and fixed modes
    for i in eachindex(model.subjectindices)
        _draw_paths_subject!(i, ...)
    end
end
```

#### Phase-Type Model Support

For `PhaseTypeModel`, `draw_paths` should:

1. Use phase-type FFBS for path sampling (existing infrastructure)
2. Compute surrogate likelihood on expanded paths (`loglik_phasetype_expanded`)
3. Compute target likelihood on collapsed paths
4. Return collapsed or expanded paths based on `expanded` keyword

```julia
function draw_paths(model::PhaseTypeModel; 
                    min_ess=100, npaths=nothing, paretosmooth=true,
                    return_logliks=false, expanded::Bool=false)
    
    # Sample on expanded space using phase-type FFBS
    result = _draw_paths_phasetype(model, ...)
    
    if expanded
        return result  # Paths on expanded state space
    else
        # Collapse paths to observed state space
        return _collapse_draw_paths_result(result, model.mappings)
    end
end
```

### New Helper Functions

#### `_get_or_fit_surrogate()`
**File**: `src/sampling.jl`

```julia
"""
Get existing surrogate from fitted model or fit a new one.
"""
function _get_or_fit_surrogate(model::MultistateProcess, is_semimarkov::Bool)
```

#### `_setup_sampling_infrastructure()`
**File**: `src/sampling.jl`

```julia
"""
Set up TPM books, hazmat books, forward-backward matrices.
Shared by all draw_paths variants.
"""
function _setup_sampling_infrastructure(model, surrogate, is_semimarkov)
```

#### `_draw_paths_subject!()`
**File**: `src/sampling.jl`

```julia
"""
Draw paths for a single subject (shared implementation).
"""
function _draw_paths_subject!(i, model, npaths_to_draw, adaptive, ...)
```

### Modified Functions

#### `MultistateModelFitted` Surrogate Storage

The struct already has a `markovsurrogate` field. Ensure it's populated during fitting:

**File**: `src/modelfitting.jl`

```julia
# After fitting, store the surrogate used
function fit(model::MultistateModel; ...)
    # ... existing fitting logic ...
    
    fitted = MultistateModelFitted(
        # ... other fields ...
        markovsurrogate = surrogate,  # Store for draw_paths reuse
        # ...
    )
end
```

### Implementation Notes

1. **Backward compatibility**: The consolidated function should accept all arguments from both original functions
2. **Default behavior**: `npaths=nothing` triggers adaptive mode (same as old `draw_paths(model; min_ess=...)`)
3. **Deprecation**: Consider deprecating the two-argument form `draw_paths(model, npaths)` with a warning

---

## Initialization Strategy

### For Markov Phase-Type Models

Use crude rates directly:

```julia
# For transition s → d with crude rate r and n phases:
λ₁ = λ₂ = ... = λₙ₋₁ = log(r)
μ₁ = μ₂ = ... = μₙ = log(r)
```

This gives approximately correct mean sojourn time.

### For Semi-Markov Models with `:pt`

1. Fit phase-type surrogate (default: same n_phases as model)
2. Extract fitted λ, μ parameters from surrogate
3. Use as initialization for model's `:pt` parameters
4. For non-`:pt` hazards, use surrogate's rates

### `_init_phasetype_from_surrogate!()`
**File**: `src/initialization.jl`

```julia
"""
Initialize phase-type hazard from fitted surrogate.
"""
function _init_phasetype_from_surrogate!(model::PhaseTypeModel, 
                                         hazard::PhaseTypeCoxianHazard,
                                         surrogate::PhaseTypeSurrogate)
    # Extract corresponding phase-type parameters from surrogate
    # Map to model's parameter vector
end
```

---

## Implementation Order

### Phase 1: Type Definitions ✅
- [x] `PhaseTypeHazardSpec` in `common.jl`
- [x] `PhaseTypeCoxianHazard <: _MarkovHazard` in `common.jl`
- [x] `PhaseTypeMappings` in `phasetype.jl`
- [x] `PhaseTypeModel` in `common.jl`
- [x] `PhaseTypeFittedModel` in `modeloutput.jl`
- [x] `_is_markov_hazard()` helper
- [x] Update `_process_class()` to use helper
- [x] Export new types in `MultistateModels.jl`

### Phase 2: Hazard Constructor ✅
- [x] Add `"pt"` to `valid_families`
- [x] Handle `n_phases` keyword in `Hazard()`
- [x] Return `PhaseTypeHazardSpec` for `:pt`
- [x] Update docstring with `:pt` documentation

### Phase 3: Mapping & Expansion ✅
- [x] `build_phasetype_mappings()` - builds PhaseTypeMappings from hazards/tmat
- [x] `_build_expanded_tmat()` - internal function for expanded tmat
- [x] `_build_expanded_hazard_indices()` - maps hazard names to expanded indices
- [x] `expand_hazards_for_phasetype()` - converts user specs to runtime hazards
- [x] `expand_data_for_phasetype_fitting()` - expands data with phase uncertainty
- [x] `_build_progression_hazard()` - creates λ hazards
- [x] `_build_exit_hazard()` - creates μ hazards
- [x] `_build_expanded_hazard()` - handles non-PT hazards
- [x] `_build_phase_censoring_patterns()` - emission patterns for uncertainty
- [x] Helper functions: `has_phasetype_hazards()`, `get_phasetype_n_phases()`, etc.

### Phase 4: Model Building ✅
- [x] `_build_phasetype_model_from_hazards()` - main model builder in phasetype.jl
- [x] Modify `multistatemodel()` to detect `:pt` hazards and route
- [x] `_build_expanded_parameters()` - builds ParameterHandling structure
- [x] `_build_original_parameters()` - builds user-facing parameters
- [x] `_count_covariates()` - counts covariate parameters
- [x] `_count_hazard_parameters()` - counts total parameters
- [x] `_merge_censoring_patterns()` - merges user patterns with phase uncertainty

### Phase 5: Initialization ✅
- [x] `set_crude_init!(::PhaseTypeModel, ...)` - in phasetype.jl
- [x] `initialize_parameters!(::PhaseTypeModel, ...)` - in phasetype.jl
- [x] `_calculate_crude_phasetype()` - calculates crude rates on observed space
- [x] `_get_crude_rate_for_expanded_hazard()` - maps crude rates to expanded hazards
- [x] `_sync_phasetype_parameters_to_original!()` - syncs expanded to user params
- [x] `_collect_phasetype_params()` - collects λ/μ params from expanded hazards
- Note: `init_par()` already handles MarkovHazard (all expanded hazards are Markov)
- Note: Surrogate init not needed (PhaseTypeModel is Markov on expanded space)

### Phase 6: Parameter Management ✅
- [x] `get_parameters(::PhaseTypeModel)` - returns user-facing params
- [x] `get_expanded_parameters(::PhaseTypeModel)` - returns expanded params
- [x] `set_parameters!(::PhaseTypeModel, Vector{Vector})` - sets from nested vector
- [x] `set_parameters!(::PhaseTypeModel, NamedTuple)` - sets from NamedTuple
- [x] `_expand_phasetype_params()` - expands user params to internal params
- [x] `_rebuild_original_params_from_values!()` - rebuilds user param structure
- [x] Exported `get_expanded_parameters` in MultistateModels.jl

### Phase 7: Fitting ✅
- [x] `fit(::PhaseTypeModel, ...)` dispatch
- [x] Handle surrogate selection for `:pt` (not needed - Markov on expanded space)
- [x] Result interpretation (PhaseTypeFittedModel wraps fitted expanded model)
- [x] **Parameter handling tests** (44 tests):
  - [x] Test `get_parameters()` returns correct user-facing structure
  - [x] Test `get_expanded_parameters()` returns internal structure
  - [x] Test `set_parameters!()` correctly updates both representations
  - [x] Test parameter round-trip (set → get returns same values)
  - [x] Test initialization respects structure (`:unstructured`, `:allequal`, `:prop_to_prog`)

### Phase 8: Simulation ✅
- [x] Add `expanded` keyword to `simulate()`
- [x] `simulate(::PhaseTypeModel, ...)`
- [x] `_collapse_simulation_result()`
- [x] `_collapse_path()`
- [x] `_collapse_data()`
- [x] `simulate_data(::PhaseTypeModel, ...)`
- [x] `simulate_paths(::PhaseTypeModel, ...)`
- [x] `simulate_path(::PhaseTypeModel, ...)`
- [x] Exported `simulate_path` in MultistateModels.jl
- [x] 126 simulation tests pass

### Phase 8.5: `draw_paths` Consolidation ⬜
- [ ] Consolidate two `draw_paths` functions into one
- [ ] Add `npaths` keyword (adaptive when `nothing`, fixed when integer)
- [ ] Add `expanded` keyword for phase-type state space output
- [ ] Add `newdata` keyword for alternative data template (reuse `_prepare_simulation_data`)
- [ ] Implement `_get_or_fit_surrogate()` to reuse fitted model's surrogate
- [ ] Implement `_setup_sampling_infrastructure()` shared helper
- [ ] Implement `_draw_paths_subject!()` shared core loop
- [ ] Add `draw_paths(::PhaseTypeModel, ...)` dispatch
- [ ] Verify `MultistateModelFitted.markovsurrogate` is populated during fitting
- [ ] Add deprecation warning for `draw_paths(model, npaths)` form
- [ ] Update tests for consolidated API

### Phase 9: Utilities ⬜
- [ ] `phasetype_constraints()` helper
- [ ] Summary/print methods for `PhaseTypeModel`
- [ ] Summary/print methods for `PhaseTypeFittedModel`

### Phase 10: Tests ⬜
- [ ] Unit tests for state expansion (`build_phasetype_mappings`)
- [ ] Unit tests for data expansion
- [ ] Unit tests for parameter mapping (collapse/expand)
- [ ] Markov pt model fitting test (panel data)
- [ ] Semi-Markov with pt test (requires MCEM)
- [ ] `n_phases=1` equivalence to exponential test
- [ ] Constraint tests (`:allequal`, `:prop_to_prog`)
- [ ] Covariate tests
- [ ] Simulation collapse tests
- [ ] Simulation expanded output tests

---

## Testing Plan

### Unit Tests

1. **State expansion correctness**
   - 3-state model with 2-phase pt on transition 1→2
   - Verify mappings are bidirectional
   - Verify expanded tmat structure

2. **Parameter mapping**
   - Round-trip: collapse → expand → collapse
   - Verify parameter count matches

3. **Data expansion**
   - Exact observations split correctly
   - Censoring patterns generated correctly

### Integration Tests

1. **Markov pt fitting (no MCEM)**
   - 2-state model with 2-phase pt
   - Panel data
   - Compare to known solution

2. **Semi-Markov with pt (MCEM)**
   - Weibull + pt hazards
   - Verify surrogate is pt by default
   - Check convergence

3. **n_phases=1 equivalence**
   - Compare pt(n=1) to exp
   - Should give identical results

4. **Simulation**
   - Simulate on collapsed vs expanded
   - Verify state mapping consistency

5. **`draw_paths` consolidation**
   - Adaptive mode (`npaths=nothing`) matches original `draw_paths(model; min_ess=...)`
   - Fixed mode (`npaths=100`) matches original `draw_paths(model, 100)`
   - Fitted model reuses stored surrogate (no refit)
   - Unfitted model fits surrogate as needed
   - Phase-type model returns collapsed paths by default
   - `expanded=true` returns paths on expanded state space

### Regression Tests

1. **Backward compatibility**
   - Existing exp/wei/gom/sp models unchanged
   - Existing tests pass

---

## Progress Tracking

| Phase | Status | Notes |
|-------|--------|-------|
| 1. Types | ✅ Complete | All types defined, package compiles |
| 2. Constructor | ✅ Complete | Hazard(:pt, ...) works with n_phases |
| 3. Expansion | ✅ Complete | State space expansion, data expansion |
| 4. Model building | ✅ Complete | multistatemodel() creates PhaseTypeModel |
| 5. Initialization | ✅ Complete | set_crude_init!, initialize_parameters! |
| 6. Parameters | ✅ Complete | get/set_parameters, sync between spaces |
| 7. Fitting | ✅ Complete | fit(PhaseTypeModel) returns MultistateModelFitted (44 tests) |
| 8. Simulation | ✅ Complete | simulate() with expanded kwarg (126 tests) |
| 8.5. `draw_paths` consolidation | ✅ Complete | Unified API, eliminated PhaseTypeFittedModel wrapper |
| 9. Utilities | ✅ Complete | show() for PhaseTypeModel and fitted phase-type models |
| 10. Tests | ✅ Complete | 4 test files: correctness, fitting, simulation, IS |

---

## Notes & Decisions Log

*Record implementation decisions and issues here as work progresses.*

### 2024-12-06
- Initial plan created
- Key decision: `PhaseTypeCoxianHazard <: _MarkovHazard` for correct model classification
- Key decision: Default pt surrogate for semi-Markov models with pt hazards
- Key decision: `expanded` keyword for simulation output control
- Key decision: Consolidate `draw_paths(model; min_ess=...)` and `draw_paths(model, npaths)` into single function with `npaths` keyword
- Key decision: `draw_paths` should reuse `markovsurrogate` from `MultistateModelFitted` instead of refitting
- Key decision: Add `expanded` keyword to `draw_paths` for phase-type state space output (default: collapsed)
- Key decision: Add `newdata` keyword to `draw_paths` (consistent with `simulate()`)
- Key decision: Remove `delta_u` and `delta_t` from simulation API - hardcode `_DELTA_U = sqrt(eps())` internally. The `delta_t` parameter was unused.
- **Phase 1 complete**: Added `PhaseTypeHazardSpec`, `PhaseTypeCoxianHazard`, `PhaseTypeMappings`, `PhaseTypeModel`, `PhaseTypeFittedModel`, `_is_markov_hazard()`, updated `_process_class()`, and exports. Package compiles successfully.

### 2024-12-06 (continued)
- **Phases 2-7 complete**: Full phase-type model building, parameter handling, and fitting
- **Phase 8 complete**: Simulation support with `expanded` keyword
- **Phase 8.5 complete**: 
  - Consolidated `draw_paths` API with `npaths` keyword
  - **Eliminated `PhaseTypeFittedModel` wrapper** - fit(PhaseTypeModel) now returns MultistateModelFitted directly
  - Phase-type specific info stored in modelcall: `is_phasetype`, `mappings`, `original_parameters`, etc.
  - Added accessor functions: `is_phasetype_fitted()`, `get_phasetype_parameters()`, `get_mappings()`, `get_convergence()`, etc.
  - `get_parameters(fitted)` returns user-facing params by default, `expanded=true` for internal
- **Phase 9 complete**: Added `show()` methods for `PhaseTypeModel` and updated fitted model display
- **All 985 tests pass**
