# Part 4: Deduplication & API Cleanup

## 4.1 Redundant Function Pairs

### Likelihood Functions

| Mutating | Functional | Decision |
|----------|------------|----------|
| `loglik_markov` | `loglik_markov_functional` | Benchmark, keep faster or unify |

**Analysis**: Both compute same thing. `_functional` exists for reverse-mode AD.

**Recommendation**:
1. Profile both with ForwardDiff
2. If functional is within 15%, use it everywhere
3. If not, dispatch based on AD backend

### Parameter Access Functions

| Function | Purpose | Keep? |
|----------|---------|-------|
| `get_parameters(model; scale=:natural)` | User API | ✅ Keep |
| `get_parameters_flat(model)` | Internal, estimation scale | ✅ Keep (different purpose) |
| `get_hazard_params(parameters, hazards)` | Extract per-hazard vectors | ⚠️ Review |
| `safe_unflatten(flat, model)` | AD-compatible unflatten | 🔄 Rename to `unflatten_natural` |
| `prepare_parameters(p, model)` | Dispatch wrapper | ❌ Remove (just calls `safe_unflatten`) |

**Migration**:
```julia
# Before
pars = prepare_parameters(parameters, model)

# After
pars = unflatten_natural(parameters, model)
```

### Simulation Functions

| Function | Purpose | Keep? |
|----------|---------|-------|
| `simulate(model; data=true, paths=false)` | Main entry | ✅ Keep |
| `simulate_data(model; ...)` | Wrapper for `simulate(data=true, paths=false)` | ❌ Remove |
| `simulate_paths(model; ...)` | Wrapper for `simulate(data=false, paths=true)` | ❌ Remove |
| `simulate_path(model, subj; ...)` | Single subject simulation | ✅ Keep |

**Migration**:
```julia
# Before
datasets = simulate_data(model; nsim=100)
trajectories = simulate_paths(model; nsim=100)

# After  
datasets = simulate(model; nsim=100, data=true, paths=false)
trajectories = simulate(model; nsim=100, data=false, paths=true)
```

---

## 4.2 File Splitting: `phasetype.jl`

### Current Structure (4,598 lines)

```
phasetype.jl
├── ProposalConfig types (~100 lines)
├── PhaseTypeDistribution type & methods (~300 lines)
├── Distribution fitting functions (~400 lines)
│   ├── fit_coxian_to_hazard
│   ├── fit_coxian_mle
│   └── moment_matching helpers
├── PhaseTypeSurrogate type & construction (~500 lines)
│   ├── fit_phasetype_surrogate
│   ├── build_phasetype_Q
│   └── parameter extraction
├── State expansion functions (~400 lines)
│   ├── expand_state_space
│   ├── build_expanded_tmat
│   └── state mapping utilities
├── PhaseTypeModel type & construction (~600 lines)
│   ├── multistatemodel for :pt hazards
│   ├── build_phasetype_model
│   └── parameter synchronization
├── FFBS on expanded space (~400 lines)
│   ├── build_phasetype_tpm_book
│   ├── build_phasetype_emat_expanded
│   ├── build_fbmats_phasetype
│   └── forward filtering helpers
├── Sampling functions (~300 lines)
│   ├── draw_samplepath_phasetype
│   ├── collapse_path
│   └── expand/collapse utilities
├── Simulation functions (~400 lines)
│   ├── simulate(::PhaseTypeModel)
│   ├── _simulate_phasetype_internal
│   └── collapse helpers
├── Fitting integration (~500 lines)
│   ├── fit(::PhaseTypeModel)
│   ├── parameter sync
│   └── variance estimation
└── Utility functions (~600 lines)
    ├── state mapping
    ├── parameter transformation
    └── debug/display
```

### Proposed Split

```
src/
├── phasetype/
│   ├── PhaseType.jl          # Module definition, exports
│   ├── types.jl              # ProposalConfig, PhaseTypeDistribution, PhaseTypeSurrogate
│   ├── distribution.jl       # Distribution fitting (fit_coxian_*, moment matching)
│   ├── surrogate.jl          # Surrogate construction (fit_phasetype_surrogate)
│   ├── expansion.jl          # State space expansion utilities
│   ├── model.jl              # PhaseTypeModel type and construction
│   ├── sampling.jl           # FFBS, sampling on expanded space
│   ├── simulation.jl         # simulate(::PhaseTypeModel)
│   └── fitting.jl            # fit(::PhaseTypeModel)
└── phasetype.jl              # Include wrapper (for backward compat)
```

**Module definition** (`src/phasetype/PhaseType.jl`):
```julia
module PhaseType

using ..MultistateModels: _Hazard, MultistateProcess, ...

include("types.jl")
include("distribution.jl")
include("surrogate.jl")
include("expansion.jl")
include("model.jl")
include("sampling.jl")
include("simulation.jl")
include("fitting.jl")

export ProposalConfig, PhaseTypeProposal, MarkovProposal
export PhaseTypeDistribution, PhaseTypeSurrogate
export PhaseTypeModel, PhaseTypeFittedModel

end # module
```

**Main module include** (`src/phasetype.jl`):
```julia
include("phasetype/PhaseType.jl")
using .PhaseType
```

---

## 4.3 Code Patterns to Eliminate

### Pattern 1: Redundant Type Checks

**Before**:
```julia
if isa(hazard, MarkovHazard)
    # exponential logic
elseif isa(hazard, SemiMarkovHazard)
    # weibull/gompertz logic
elseif isa(hazard, RuntimeSplineHazard)
    # spline logic
end
```

**After** (dispatch):
```julia
# Define methods for each type
eval_hazard(h::MarkovHazard, t, pars, covars) = ...
eval_hazard(h::SemiMarkovHazard, t, pars, covars) = ...
eval_hazard(h::RuntimeSplineHazard, t, pars, covars) = ...
```

**Status**: Already mostly using dispatch. Clean up remaining `isa` checks.

### Pattern 2: Dict Lookup in Inner Loops

**Before**:
```julia
for i in 1:nrow(data)
    hazard_pars = pars[hazard.hazname]  # Dict lookup
    ...
end
```

**After** (pre-extract):
```julia
hazard_pars = pars[hazard.hazname]  # Once outside loop
for i in 1:nrow(data)
    # Use hazard_pars directly
    ...
end
```

**Or** (vector indexing):
```julia
pars_vec = [pars[h.hazname] for h in hazards]  # Pre-extract
for i in 1:nrow(data)
    hazard_pars = pars_vec[hazard_idx]  # Vector index
    ...
end
```

### Pattern 3: DataFrame Row Access

**Before**:
```julia
for i in 1:nrow(df)
    row = df[i, :]  # Creates DataFrameRow
    value = row.column
end
```

**After** (column access):
```julia
col = df.column  # Get column vector once
for i in 1:nrow(df)
    value = col[i]  # Direct vector index
end
```

**Or** (view):
```julia
for i in 1:nrow(df)
    row = @view df[i, :]  # View, not copy
    value = row.column
end
```

### Pattern 4: Allocation in Hot Loops

**Before**:
```julia
for i in 1:n
    result = zeros(3)  # Allocates every iteration
    result[1] = compute_a(i)
    result[2] = compute_b(i)
    result[3] = compute_c(i)
    process(result)
end
```

**After** (pre-allocate):
```julia
result = zeros(3)  # Allocate once
for i in 1:n
    result[1] = compute_a(i)
    result[2] = compute_b(i)
    result[3] = compute_c(i)
    process(result)
end
```

---

## 4.4 API Cleanup Checklist

### Functions to Remove

| Function | Replacement | Breaking? |
|----------|-------------|-----------|
| `simulate_data` | `simulate(; data=true, paths=false)` | Yes |
| `simulate_paths` | `simulate(; data=false, paths=true)` | Yes |
| `prepare_parameters` | `unflatten_natural` | No (internal) |

### Functions to Rename

| Old Name | New Name | Reason |
|----------|----------|--------|
| `safe_unflatten` | `unflatten_natural` | Clarity: returns natural scale |
| `unflatten_to_estimation_scale` | `unflatten_estimation` | Consistency |

### Functions to Consolidate

| Functions | Into | Reason |
|-----------|------|--------|
| `loglik_markov` + `loglik_markov_functional` | Single with dispatch | Reduce duplication |
| Multiple `get_*_params` | Unified `ParameterOps` module | Single source of truth |

---

## 4.5 Deprecated Function Handling

### Strategy

```julia
# 1. Add deprecation warning
function simulate_data(model; kwargs...)
    Base.depwarn(
        "`simulate_data(model; ...)` is deprecated, use `simulate(model; data=true, paths=false, ...)`",
        :simulate_data
    )
    return simulate(model; data=true, paths=false, kwargs...)
end

# 2. After one release cycle, remove entirely
```

### Timeline

| Version | Action |
|---------|--------|
| Current | Add deprecation warnings |
| Next minor | Remove deprecated functions |

---

## 4.6 Documentation Updates Needed

After API cleanup:

1. **Update docstrings** for renamed functions
2. **Update examples** in docstrings
3. **Update README** if it mentions removed functions
4. **Update docs/src/*.md** pages
5. **Add migration guide** for breaking changes

---

## Action Items from Deduplication Analysis

| Item | Priority | Effort | Breaking? |
|------|----------|--------|-----------|
| Remove `simulate_data`/`simulate_paths` | Low | Low | Yes |
| Rename `safe_unflatten` → `unflatten_natural` | Medium | Low | No |
| Remove `prepare_parameters` | Medium | Low | No |
| Split `phasetype.jl` | Medium | Medium | No |
| Consolidate likelihood functions | Medium | Medium | No |
| Clean up Dict lookups in inner loops | High | Medium | No |
| Pre-extract DataFrame columns | High | Medium | No |
