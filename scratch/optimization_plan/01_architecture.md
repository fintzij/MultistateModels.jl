# Part 1: Architecture Deep Dive

## 1.1 Parameter Handling System

### Current Implementation

The package has **5 overlapping parameter handling mechanisms**:

#### 1. `ReConstructor` (helpers.jl:60-340)
```julia
struct ReConstructor{F,S,T,U,V}
    default::FlattenDefault
    flatten_strict::F       # Type-stable flatten → Vector{Float64}
    flatten_flexible::S     # AD-compatible flatten → Vector{T}
    unflatten_strict::T     # Type-stable unflatten (converts to original types)
    unflatten_flexible::U   # AD-compatible unflatten (preserves Dual types)
    _buffer::V              # Pre-allocated buffer
end
```

**Usage pattern**:
- Created during model generation
- Stored in `model.parameters.reconstructor`
- `unflatten(rc, flat)` for standard ops
- `unflattenAD(rc, flat)` for AD contexts

#### 2. `safe_unflatten` / `unflatten_parameters` (helpers.jl:400-450)
```julia
function unflatten_parameters(flat_params::AbstractVector{T}, model::MultistateProcess) where {T<:Real}
    # Unflatten to estimation-scale nested structure
    if T === Float64
        params_estimation = unflatten(model.parameters.reconstructor, flat_params)
    else
        params_estimation = unflattenAD(model.parameters.reconstructor, flat_params)
    end
    # Transform baseline parameters to natural scale
    return to_natural_scale(params_estimation, model.hazards, T)
end
const safe_unflatten = unflatten_parameters
```

**Usage**: Primary entry point in likelihood functions

#### 3. `prepare_parameters` (likelihoods.jl:24-45)
```julia
prepare_parameters(p::Tuple, ::MultistateProcess) = p
prepare_parameters(p::NamedTuple, ::MultistateProcess) = p
prepare_parameters(p::AbstractVector{<:AbstractVector}, ::MultistateProcess) = p
function prepare_parameters(p::AbstractVector{<:Real}, model::MultistateProcess)
    return safe_unflatten(p, model)
end
```

**Usage**: Dispatch-based normalization, just wraps `safe_unflatten`

#### 4. `get_hazard_params` (modeloutput.jl)
```julia
function get_hazard_params(parameters, hazards)
    # Extract per-hazard parameter vectors on natural scale
    ...
end
```

**Usage**: Convert nested params to vector-of-vectors for hazard evaluation

#### 5. `to_natural_scale` (helpers.jl)
```julia
function to_natural_scale(params_nested, hazards, T)
    # Apply exp() to baseline parameters based on hazard family
    ...
end
```

**Usage**: Called by `safe_unflatten` to transform log→natural

### The Problem

```
Optimization calls loglik(flat_params, data)
    ↓
loglik calls safe_unflatten(flat_params, model)
    ↓
safe_unflatten calls unflattenAD(reconstructor, flat_params)
    ↓
safe_unflatten calls to_natural_scale(nested, hazards, T)
    ↓
Returns NamedTuple on natural scale
    ↓
Hazard evaluation accesses params[hazard.hazname]
```

**Issues**:
1. **Redundant dispatch**: `prepare_parameters` just calls `safe_unflatten`
2. **Naming confusion**: `safe_unflatten` vs `unflatten_parameters` vs `unflatten`
3. **Scale transformation scattered**: `to_natural_scale` called in `safe_unflatten`, but hazard fns also document they expect natural scale
4. **No single source of truth**: Comments about parameter scales appear in multiple files

### Proposed Consolidation

```julia
# New unified module in helpers.jl or new file parameters.jl

module ParameterOps

export unflatten_estimation, unflatten_natural, flatten

"""
    unflatten_estimation(flat, model) → NamedTuple

Unflatten to nested structure on ESTIMATION scale (log for baseline params).
Use for spline remake, constraint checking, etc.
"""
unflatten_estimation(flat::AbstractVector{Float64}, model) = 
    unflatten(model.parameters.reconstructor, flat)
unflatten_estimation(flat::AbstractVector, model) = 
    unflattenAD(model.parameters.reconstructor, flat)

"""
    unflatten_natural(flat, model) → NamedTuple

Unflatten to nested structure on NATURAL scale (exp applied to baseline).
Use for hazard evaluation, likelihood computation.
"""
function unflatten_natural(flat::AbstractVector{T}, model) where T
    nested = unflatten_estimation(flat, model)
    return to_natural_scale(nested, model.hazards, T)
end

"""
    flatten(nested, model) → Vector{Float64}

Flatten nested parameters to vector on estimation scale.
"""
flatten(nested, model) = flatten(model.parameters.reconstructor, nested)

end # module
```

**Migration path**:
1. Add new functions alongside old ones
2. Deprecate `safe_unflatten` → `unflatten_natural`
3. Deprecate `prepare_parameters` → remove (just use `unflatten_natural`)
4. Update all call sites
5. Remove deprecated functions after tests pass

---

## 1.2 Type Hierarchy

### Hazard Types

```
HazardFunction (abstract, user-facing specs)
├── ParametricHazard      # User spec for exp/wei/gom
├── SplineHazard          # User spec for spline hazards
└── PhaseTypeHazardSpec   # User spec for phase-type

_Hazard (abstract, runtime evaluation)
├── _MarkovHazard (abstract)
│   ├── MarkovHazard              # Exponential hazard (runtime)
│   └── PhaseTypeCoxianHazard     # Phase-type expanded hazard
├── _SemiMarkovHazard (abstract)
│   └── SemiMarkovHazard          # Weibull/Gompertz hazard (runtime)
└── _SplineHazard (abstract)
    └── RuntimeSplineHazard       # Spline hazard (runtime)
```

**Assessment**: This hierarchy is reasonable. The separation of user-facing specs from runtime types is a good pattern. No changes recommended.

### Model Types

```
MultistateProcess (abstract)
├── MultistateMarkovProcess (abstract)
│   ├── MultistateMarkovModel           # Markov, exact data
│   ├── MultistateMarkovModelCensored   # Markov, censored data
│   └── PhaseTypeModel                  # Phase-type (Markov on expanded space)
└── MultistateSemiMarkovProcess (abstract)
    ├── MultistateSemiMarkovModel         # Semi-Markov, exact data
    └── MultistateSemiMarkovModelCensored # Semi-Markov, censored data
```

**Assessment**: Hierarchy is appropriate for dispatch. The Markov/SemiMarkov split enables correct algorithm selection.

### Data Container Types

```
# Likelihood data containers
ExactData           # Exact observations (obstype 1)
ExactDataAD         # Single path for AD (Fisher info computation)
MPanelData          # Markov panel data
SMPanelData         # Semi-Markov panel data (paths + weights)

# Batched computation containers
CachedPathData      # Pre-computed DataFrame per path
StackedHazardData   # Intervals stacked for batched hazard evaluation
BatchedODEData      # Matrix format for neural network hazards
```

**Issue**: `CachedPathData` and `StackedHazardData` may have redundant storage. Profile before optimizing.

---

## 1.3 Module Organization

### Current Structure (single module)

```julia
# MultistateModels.jl
module MultistateModels

# All code in flat includes
include("common.jl")
include("helpers.jl")
include("hazards.jl")
# ... 16 more includes

export multistatemodel, fit, simulate, ...

end
```

### Proposed Structure (submodules for large files)

```julia
module MultistateModels

# Core types (no submodule, needed everywhere)
include("common.jl")
include("helpers.jl")

# Hazard evaluation
include("hazards.jl")

# Splines (could be submodule)
include("smooths.jl")

# Phase-type (split into submodule)
include("phasetype/PhaseType.jl")
using .PhaseType

# Main functionality
include("modelgeneration.jl")
include("simulation.jl")
include("likelihoods.jl")
include("sampling.jl")
include("modelfitting.jl")

# Output and utilities
include("modeloutput.jl")
include("crossvalidation.jl")
include("initialization.jl")

end
```

**Benefits**:
- `phasetype.jl` (4,598 lines) becomes manageable
- Clear dependency structure
- Easier to test components in isolation

---

## 1.4 Caching Infrastructure

### TimeTransformCache (common.jl:50-120)

```julia
struct TimeTransformHazardKey{L,T}
    linpred::L
    t::T
end

struct TimeTransformCumulKey{L,T}
    linpred::L
    lb::T
    ub::T
end

mutable struct TimeTransformCache{L,T}
    hazard_values::Dict{TimeTransformHazardKey{L,T}, Float64}
    cumulhaz_values::Dict{TimeTransformCumulKey{L,T}, Float64}
end

struct TimeTransformContext{L,T}
    caches::Vector{Union{Nothing, TimeTransformCache{L,T}}}
    shared_baselines::SharedBaselineTable
end
```

**Purpose**: Memoize hazard/cumhaz evaluations when linear predictor is constant across intervals.

**When useful**: 
- Time-varying covariates where linpred changes rarely
- Shared baseline hazards across multiple transitions

**When NOT useful**:
- Most evaluations at unique (linpred, t) or (linpred, lb, ub) combinations
- Dict lookup overhead exceeds computation cost

**Profiling needed**: Measure cache hit rate during typical MCEM run.

### CachedPathData (likelihoods.jl:300-350)

```julia
struct CachedPathData
    subj::Int
    df::DataFrame           # Pre-computed DataFrame from make_subjdat
    linpreds::Dict{Int, Vector{Float64}}  # Cached linear predictors per hazard
end
```

**Purpose**: Avoid repeated `make_subjdat` calls for same path.

**Assessment**: Likely useful - DataFrame construction is expensive. Keep.

### StackedHazardData (likelihoods.jl:350-400)

```julia
struct StackedHazardData
    lb::Vector{Float64}
    ub::Vector{Float64}
    covars::Vector{NamedTuple}
    linpreds::Vector{Float64}
    path_idx::Vector{Int}
    is_transition::Vector{Bool}
    transition_times::Vector{Float64}
end
```

**Purpose**: Stack all intervals for a hazard across all paths for batched evaluation.

**Assessment**: Useful for future GPU/batched neural network hazards. May have overhead for simple parametric hazards.

---

## 1.5 AD Compatibility Architecture

### Current Approach

**ForwardDiff (forward-mode)**:
- Works with mutation (`!` functions)
- Uses `unflattenAD` to preserve Dual types
- Primary backend for optimization

**Mooncake (reverse-mode)**:
- Requires non-mutating code paths
- `loglik_markov_functional` exists for this
- Not fully tested across all code paths

### Dual Path Pattern

```julia
# Mutating version (ForwardDiff compatible)
function loglik_markov(parameters, data; neg=true)
    pars = safe_unflatten(parameters, data.model)
    # ... mutating operations with hazmat_book, tpm_book ...
end

# Non-mutating version (Mooncake/Enzyme compatible)
function loglik_markov_functional(parameters, data; neg=true)
    pars = safe_unflatten(parameters, data.model)
    # ... functional operations, no mutation ...
end
```

### Recommendation

1. **Profile both versions** with ForwardDiff
2. If <10% difference, **use functional version everywhere**
3. If significant difference, **dispatch based on AD backend**:

```julia
function loglik_markov(parameters, data; neg=true, backend=ForwardDiffBackend())
    if backend isa ReverseADBackend
        return loglik_markov_functional(parameters, data; neg=neg)
    else
        return loglik_markov_mutating(parameters, data; neg=neg)
    end
end
```

---

## Action Items from Architecture Review

| Item | Priority | Effort | Impact |
|------|----------|--------|--------|
| Consolidate parameter handling | Medium | Medium | Clarity |
| Profile TimeTransformCache | High | Low | Potential perf win |
| Split phasetype.jl | Medium | Medium | Maintainability |
| Benchmark loglik_markov vs functional | High | Low | Simplification or optimization |
| Document parameter scales | Medium | Low | Developer experience |
