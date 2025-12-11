# Part 6: Implementation Phases

## Phase Overview

| Phase | Focus | Duration | Risk | Breaking Changes | Status |
|-------|-------|----------|------|------------------|--------|
| **Phase 0** | Profiling & Baseline | 1 day | None | No | ✅ Complete |
| **Phase 1** | Quick Wins | 2-3 days | Low | Minimal | ✅ Complete (~24% speedup) |
| **Phase 2** | Parameter System | 3-5 days | Medium | Moderate | ✅ Complete (API consolidated) |
| **Phase 3** | Likelihood Unification | 2-3 days | Medium | Yes | ✅ Complete |
| **Phase 4** | Caching Optimization | 2-3 days | Low | No | Not started |
| **Phase 5** | File Reorganization | 1-2 days | Low | No | Not started |
| **Phase 6** | AD Modernization | 3-5 days | High | Yes | Not started |

## Completed Phase Summary

### Phase 1 Results (Quick Wins)
- **Implemented**: Indexed parameter access via `get_hazard_params_indexed()` - O(1) tuple access instead of O(n) NamedTuple lookup
- **Kept**: `simulate_data()` and `simulate_paths()` wrappers (user requested)
- **Performance**: ~24% speedup on Markov likelihood
- **Files modified**: `src/helpers.jl` (added `get_hazard_params_indexed`), `src/likelihoods.jl` (updated call sites)

### Phase 2 Results (Parameter System Consolidation)
- **New canonical functions**:
  - `unflatten_natural(flat_params, model)` - Returns natural-scale NamedTuple
  - `unflatten_estimation(flat_params, model)` - Returns estimation-scale parameters
- **Backward compatibility aliases**:
  - `safe_unflatten = unflatten_natural`
  - `unflatten_parameters = unflatten_natural`
  - `unflatten_to_estimation_scale = unflatten_estimation`
- **Files modified**: `src/helpers.jl`, `src/likelihoods.jl`, `src/sampling.jl`, `src/crossvalidation.jl`

### Phase 3 Results (Likelihood Unification)
- **Unified API**: Single `loglik_markov(params, data; backend=...)` entry point
- **Backend dispatch**: `backend` kwarg selects implementation
  - `ForwardDiffBackend()` (default) → mutating implementation (faster)
  - `EnzymeBackend()` / `MooncakeBackend()` → functional implementation (AD-compatible)
- **Internal implementations**:
  - `_loglik_markov_mutating` - Uses `compute_hazmat!`, `compute_tmat!`
  - `_loglik_markov_functional` - Non-mutating for reverse-mode AD
- **Backward compatibility**: `loglik_markov_functional` alias preserved
- **Files modified**: `src/likelihoods.jl`, `src/modelfitting.jl`
- **Tests**: All Phase 3 validation tests pass

---

## Phase 0: Profiling & Baseline

**Goal**: Establish quantitative baseline before any changes.

### Tasks

1. [ ] Create profiling fixtures (from 05_profiling_scripts.md)
2. [ ] Run simulation profiling, record results
3. [ ] Run likelihood profiling, record results
4. [ ] Run MCEM profiling, record results
5. [ ] Identify top 5 allocation hotspots
6. [ ] Identify top 5 time hotspots
7. [ ] Save baseline metrics to `scratch/profiling/baseline.json`

### Baseline Metrics to Record

```julia
baseline = (
    # Simulation
    simulate_path_2state_markov_us = 0.0,
    simulate_path_2state_semimarkov_us = 0.0,
    simulate_100paths_allocs = 0,
    simulate_100paths_memory_kb = 0.0,
    
    # Likelihood
    loglik_markov_2state_us = 0.0,
    loglik_markov_3state_us = 0.0,
    loglik_markov_functional_overhead = 0.0,  # ratio
    loglik_path_us = 0.0,
    make_subjdat_us = 0.0,
    
    # MCEM
    mcem_iteration_ms = 0.0,
    estep_path_sampling_us = 0.0,
    mstep_likelihood_ms = 0.0,
    
    # Hazards
    eval_hazard_exp_ns = 0.0,
    eval_hazard_wei_ns = 0.0,
    eval_cumhaz_exp_ns = 0.0,
    eval_cumhaz_wei_ns = 0.0,
)
```

### Success Criteria
- [ ] All profiling scripts run without error
- [ ] Baseline metrics recorded and committed
- [ ] Top hotspots documented

---

## Phase 1: Quick Wins (Low Risk)

**Goal**: Remove obvious inefficiencies without architectural changes.

### 1.1 Remove Thin Wrappers

**Files**: `src/simulation.jl`

```julia
# BEFORE
simulate_data(model; nsim, kwargs...) = simulate(model; nsim, data=true, paths=false, kwargs...)
simulate_paths(model; nsim, kwargs...) = simulate(model; nsim, data=false, paths=true, kwargs...)

# AFTER
# Delete simulate_data and simulate_paths
# Document that users should use simulate(model; nsim=N, data=true/false, paths=true/false)
```

**Migration**:
- Add deprecation warning in v0.X
- Remove in next major version

### 1.2 Eliminate Unnecessary Dict Lookups

**Files**: `src/hazards.jl`, `src/likelihoods.jl`

```julia
# BEFORE (in hot path)
pars = params[hazard.hazname]

# AFTER (hoist outside loop)
# When iterating over hazards, pre-extract all params once
hazard_params = ntuple(i -> params[model.hazards[i].hazname], length(model.hazards))
```

### 1.3 Use @views Consistently

**Files**: `src/likelihoods.jl`, `src/sampling.jl`

```julia
# Audit all DataFrame row access
# Replace: row = df[i, :]
# With: row = @view df[i, :]
```

### 1.4 Replace vcat in Loops

**Files**: `src/simulation.jl`

```julia
# BEFORE
for ...
    all_paths = vcat(all_paths, new_paths)
end

# AFTER
all_paths = Vector{SamplePath}(undef, nsubj * nsim)
idx = 1
for ...
    all_paths[idx:idx+length(new_paths)-1] .= new_paths
    idx += length(new_paths)
end
```

### Testing Checklist
- [ ] Run `test/runtests.jl` after each change
- [ ] Compare benchmark results to baseline
- [ ] Document any regressions

### Expected Improvement
- 10-20% reduction in allocations
- 5-10% speedup in simulation

---

## Phase 2: Parameter System Consolidation

**Goal**: Unify the 5 parameter handling mechanisms.

### 2.1 Design New Parameter System

```julia
"""
Unified parameter system for AD-compatible parameter handling.
"""
struct ParameterSystem{T,N,S}
    flat::Vector{T}              # Flat vector for optimization
    nested::S                    # NamedTuple view for access
    indices::NTuple{N,UnitRange{Int}}  # Slices for each hazard
end

function ParameterSystem(model::MultistateModel)
    # Build from model.parameters
end

# AD-compatible access
@inline function get_hazard_params(ps::ParameterSystem, idx::Int)
    @view ps.flat[ps.indices[idx]]
end

# Zero-copy update
function update_flat!(ps::ParameterSystem, new_flat::AbstractVector)
    copyto!(ps.flat, new_flat)
end
```

### 2.2 Migration Path

1. Create `ParameterSystem` type in new file `src/parametersystem.jl`
2. Add `ParameterSystem` field to `MultistateModel` (alongside existing)
3. Update hot paths to use `ParameterSystem`
4. Deprecate old accessor functions
5. Remove old fields after validation

### 2.3 Affected Functions

- `safe_unflatten` → internal to `ParameterSystem`
- `unflatten_parameters` → deprecated
- `prepare_parameters` → replaced by `ParameterSystem`
- `get_hazard_params` → method on `ParameterSystem`
- `model.parameters` → replaced by `model.params.nested`

### Testing Checklist
- [ ] Unit tests for `ParameterSystem`
- [ ] Integration tests with AD (ForwardDiff, Mooncake)
- [ ] Verify no performance regression vs baseline

---

## Phase 3: Likelihood Unification

**Goal**: Single likelihood function that works for both mutation and AD.

### 3.1 Unified Design

```julia
"""
Unified Markov panel likelihood with AD backend dispatch.
"""
function loglik_markov(
    params::AbstractVector{T}, 
    data::MPanelData;
    backend::ADBackend = CurrentADBackend()
) where T
    # Dispatch based on backend
    _loglik_markov_impl(params, data, backend)
end

# Default (mutating) implementation
function _loglik_markov_impl(params, data, ::MutatingBackend)
    # Current loglik_markov code
end

# ForwardDiff-compatible implementation
function _loglik_markov_impl(params, data, ::ForwardDiffBackend)
    # Current loglik_markov_functional code
end

# Future: Mooncake/Enzyme implementation
function _loglik_markov_impl(params, data, ::ReverseADBackend)
    # Optimized for reverse-mode AD
end
```

### 3.2 Remove Duplicates

After unification:
- [ ] Remove `loglik_markov_functional`
- [ ] Remove `_semi_markov_loglik_functional`
- [ ] Update all call sites

### 3.3 Test AD Compatibility

```julia
# Test script
using ForwardDiff, Mooncake

function test_ad_compatibility(model)
    data = MPanelData(model, ...)
    params = get_parameters_flat(model)
    
    # ForwardDiff
    grad_fd = ForwardDiff.gradient(p -> loglik_markov(p, data), params)
    
    # Mooncake
    grad_mc = Mooncake.gradient(p -> loglik_markov(p, data), params)
    
    @assert isapprox(grad_fd, grad_mc, rtol=1e-6)
end
```

---

## Phase 4: Caching Optimization

**Goal**: Validate and optimize caching strategy.

### 4.1 Profile TimeTransformCache

```julia
# Instrument cache hit/miss
mutable struct InstrumentedCache{C}
    cache::C
    hits::Int
    misses::Int
end

function get_cached!(ic::InstrumentedCache, key)
    if haskey(ic.cache, key)
        ic.hits += 1
        return ic.cache[key]
    else
        ic.misses += 1
        # compute and store
    end
end

# Run MCEM and report
# Expected: hit_rate > 0.8 for benefit
```

### 4.2 Conditional Caching

If cache hit rate < 0.5:
- Remove caching overhead
- Recompute directly

If cache hit rate > 0.8:
- Pre-allocate cache entries
- Consider LRU with fixed size

### 4.3 TPM Caching

```julia
# Current: Recompute TPMs each iteration
# Proposed: Cache TPMs keyed by (unique_intervals, params)

struct TPMCache
    interval_keys::Vector{UInt64}  # Hash of interval
    tpms::Vector{Matrix{Float64}}
    params_hash::UInt64
end

# Invalidate when params change
function update_tpm_cache!(cache, new_params)
    if hash(new_params) != cache.params_hash
        recompute_all!(cache, new_params)
    end
end
```

---

## Phase 5: File Reorganization

**Goal**: Improve maintainability through better file organization.

### 5.1 Split phasetype.jl

```
src/phasetype/
├── core.jl          # Phase-type types, basic operations
├── fitting.jl       # EMFit, parameter estimation
├── sampling.jl      # Path sampling from phase-type
├── phasetype.jl     # Re-exports, main include file
└── utils.jl         # Helper functions
```

### 5.2 Create parametersystem.jl

New file for unified parameter handling.

### 5.3 Update MultistateModels.jl

```julia
# Organized includes
include("common.jl")              # Core types
include("parametersystem.jl")     # Parameter handling
include("hazards.jl")             # Hazard functions
include("likelihoods.jl")         # Likelihood computation
include("sampling.jl")            # Path sampling
include("simulation.jl")          # Simulation
include("modelfitting.jl")        # Model fitting
include("mcem.jl")                # MCEM algorithm
include("phasetype/phasetype.jl") # Phase-type infrastructure
include("crossvalidation.jl")     # CV utilities
include("surrogates.jl")          # Surrogate fitting
```

---

## Phase 6: AD Modernization

**Goal**: Native Mooncake support with ForwardDiff fallback.

### 6.1 Backend Abstraction

```julia
# src/ad_backend.jl

abstract type ADBackend end
struct ForwardDiffBackend <: ADBackend end
struct MooncakeBackend <: ADBackend end
struct EnzymeBackend <: ADBackend end  # Future

const DEFAULT_AD_BACKEND = Ref{ADBackend}(ForwardDiffBackend())

function set_ad_backend!(backend::ADBackend)
    DEFAULT_AD_BACKEND[] = backend
end

function gradient(f, x; backend=DEFAULT_AD_BACKEND[])
    _gradient(f, x, backend)
end

_gradient(f, x, ::ForwardDiffBackend) = ForwardDiff.gradient(f, x)
_gradient(f, x, ::MooncakeBackend) = Mooncake.gradient(f, x)[2]
```

### 6.2 Audit Mutation

Functions that need mutation-free versions for reverse AD:
- [ ] `loglik_markov` → done in Phase 3
- [ ] `loglik_semi_markov`
- [ ] `compute_hazmat!`
- [ ] `compute_tmat!`

### 6.3 Testing Matrix

| Function | ForwardDiff | Mooncake | Notes |
|----------|-------------|----------|-------|
| `loglik_markov` | ✅ | ⬜ | Primary target |
| `loglik_semi_markov` | ✅ | ⬜ | Uses path sampling |
| `hazard evaluation` | ✅ | ⬜ | Simple functions |
| `cumhaz evaluation` | ✅ | ⬜ | May need quadgk handling |

---

## Validation Strategy

### After Each Phase

1. **Run full test suite**: `julia --project=. -e 'using Pkg; Pkg.test()'`
2. **Run benchmarks**: Compare to baseline
3. **Run type stability check**:
```julia
using JET
@report_opt fit(model; maxiter=1)
```

### Regression Criteria

Reject change if:
- Any test fails
- Performance degrades > 10% vs baseline
- Memory usage increases > 20%

### Documentation

Update after each phase:
- [ ] CHANGELOG.md
- [ ] API docstrings
- [ ] Migration guide for breaking changes
