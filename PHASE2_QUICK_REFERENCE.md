# Phase 2 Architecture - Quick Reference Guide

## Overview

**Phase 2 Architecture**: Runtime-generated functions instead of data-storing structs

---

## Hazard Type Hierarchy

```
AbstractHazard (abstract)
├── MarkovHazard (exponential hazards)
├── SemiMarkovHazard (Weibull, Gompertz)
└── SplineHazard (spline-based, TODO)
```

**Old types (deprecated but still present)**:
- `_Exponential`, `_ExponentialPH`, `_Weibull`, `_WeibullPH`, `_Gompertz`, `_GompertzPH`, `_Spline`, `_SplinePH`

---

## Creating Models

### Method 1: Old API (Still Supported)
```julia
using MultistateModels, DataFrames

# Define hazards
h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
h13 = Hazard(@formula(0 ~ 1 + age), "wei", 1, 3)

# Create model
model = multistatemodel(h12, h13; data = df)

# This creates NEW consolidated hazard types behind the scenes!
typeof(model.hazards[1])  # MarkovHazard
typeof(model.hazards[2])  # SemiMarkovHazard
```

### Method 2: New API
```julia
model = multistatemodel(
    subjectdata = df,
    transitionmatrix = tmat,
    hazardfamily = ["exp", "wei"],
    formulae = [@formula(0 ~ 1), @formula(0 ~ 1 + age)],
    nknots = [0, 0]
)
```

---

## Using Hazards

### Direct Callable Interface (NEW)
```julia
# Get hazard and parameters
hazard = model.hazards[1]
pars = model.parameters[1]
covars = [1.0, 25.0]  # Intercept + age

# Evaluate hazard at time t
t = 2.5
haz = hazard(t, pars, covars)

# Evaluate cumulative hazard over interval [lb, ub]
cumhaz = MultistateModels.cumulative_hazard(hazard, 0.0, t, pars, covars)
```

### Backward Compatible Interface (OLD - Still Works)
```julia
# Log scale
log_haz = MultistateModels.call_haz(t, pars, rowind, hazard; give_log=true)
log_cumhaz = MultistateModels.call_cumulhaz(lb, ub, pars, rowind, hazard; give_log=true)

# Natural scale
haz = MultistateModels.call_haz(t, pars, rowind, hazard; give_log=false)
cumhaz = MultistateModels.call_cumulhaz(lb, ub, pars, rowind, hazard; give_log=false)
```

---

## Hazard Function Signatures

### Exponential (MarkovHazard)
```julia
# Baseline (no covariates)
hazard_fn(t, pars, covars) = exp(pars[1])

# With covariates
hazard_fn(t, pars, covars) = exp(pars[1] + dot(covars, pars[2:end]))

# Cumulative hazard
cumhaz_fn(lb, ub, pars, covars) = hazard_fn(0, pars, covars) * (ub - lb)
```

### Weibull (SemiMarkovHazard)
```julia
# Parameters: pars[1]=log(shape), pars[2]=log(scale), pars[3:end]=covariate effects
# h(t) = shape * scale * t^(shape-1) * exp(covariates' * β)

# Baseline
hazard_fn(t, pars, covars) = exp(pars[1] + pars[2] + expm1(pars[1]) * log(t))

# With covariates
hazard_fn(t, pars, covars) = exp(pars[1] + pars[2] + expm1(pars[1]) * log(t) + dot(covars, pars[3:end]))

# Cumulative hazard: H(lb, ub) = scale * (ub^shape - lb^shape)
cumhaz_fn(lb, ub, pars, covars) = exp(pars[2] + dot(covars, pars[3:end])) * (ub^exp(pars[1]) - lb^exp(pars[1]))
```

### Gompertz (SemiMarkovHazard)
```julia
# Parameters: pars[1]=log(shape), pars[2]=log(scale), pars[3:end]=covariate effects
# h(t) = scale * exp(shape * t + covariates' * β)

# Baseline
hazard_fn(t, pars, covars) = exp(pars[2] + exp(pars[1]) * t)

# With covariates
hazard_fn(t, pars, covars) = exp(pars[2] + exp(pars[1]) * t + dot(covars, pars[3:end]))

# Cumulative hazard: H(lb, ub) = (scale / shape) * (exp(shape * ub) - exp(shape * lb))
cumhaz_fn(lb, ub, pars, covars) = (exp(pars[2]) / exp(pars[1])) * (exp(exp(pars[1]) * ub) - exp(exp(pars[1]) * lb)) * exp(dot(covars, pars[3:end]))
```

---

## Parameter Management

### Setting Parameters
```julia
# Set parameters for hazard index h
new_pars = [log(0.5)]  # On log scale
set_parameters!(model, h, new_pars)

# Verify update
model.parameters[h]  # Should equal new_pars
```

### ParameterHandling.jl Integration (Phase 3 - TODO)
```julia
# Current issue: model.parameters_ph cannot be updated in-place (NamedTuple)
# Workaround: Use helper function
new_params_ph = MultistateModels.update_parameters_ph!(model)

# Then manually reassign (or future: make mutable)
# model.parameters_ph = new_params_ph  # Not currently implemented
```

---

## Accessing Hazard Components

### Metadata
```julia
hazard.hazname       # Symbol: :h12
hazard.statefrom     # Int: 1
hazard.stateto       # Int: 2
hazard.family        # String: "exp", "wei", "gom", "sp"
hazard.parnames      # Vector{Symbol}: [:h12_Intercept]
hazard.npar_baseline # Int: number of baseline parameters
hazard.npar_total    # Int: total parameters (baseline + covariates)
```

### Functions
```julia
# Direct access to runtime-generated functions
hazard.hazard_fn(t, pars, covars)      # Hazard rate
hazard.cumhaz_fn(lb, ub, pars, covars) # Cumulative hazard

# Or use callable interface
hazard(t, pars, covars)  # Calls hazard_fn internally
```

### Covariates
```julia
hazard.has_covariates  # Bool: true if model includes covariates
```

---

## Implementation Details

### Runtime Function Generation
Uses `RuntimeGeneratedFunctions.jl` for performance:

```julia
using RuntimeGeneratedFunctions

function generate_exponential_hazard(has_covariates::Bool)
    if !has_covariates
        hazard_expr = :(exp(pars[1]))
    else
        hazard_expr = :(exp(pars[1] + dot(covars, pars[2:end])))
    end
    
    hazard_fn = @RuntimeGeneratedFunction(
        :(function (t, pars, covars)
            $hazard_expr
        end)
    )
    
    # ... cumhaz_fn similarly ...
    
    return hazard_fn, cumhaz_fn
end
```

### Why Runtime Generation?
- **Performance**: Eliminates closure overhead
- **Flexibility**: Different code paths for covariates vs no covariates
- **Type Stability**: Generated functions are type-stable

---

## Migration Guide

### If You're Using OLD Hazard Types Directly
```julia
# OLD CODE (using _Exponential directly)
hazard = model.hazards[1]  # Returns _Exponential
haz = MultistateModels.call_haz(t, pars, rowind, hazard)

# NEW CODE (still works! Backward compatible)
hazard = model.hazards[1]  # Returns MarkovHazard
haz = MultistateModels.call_haz(t, pars, rowind, hazard)

# OR use new interface
haz_raw = hazard(t, pars, covars)
haz = give_log ? log(haz_raw) : haz_raw
```

### If You're Creating Custom Hazards
```julia
# OLD: Would create new _Hazard subtype and add data

# NEW: Use runtime function generation
function generate_my_hazard(has_covariates)
    hazard_fn = @RuntimeGeneratedFunction(:(function (t, pars, covars)
        # Your hazard formula here
        exp(pars[1] * t)
    end))
    
    cumhaz_fn = @RuntimeGeneratedFunction(:(function (lb, ub, pars, covars)
        # Your cumulative hazard formula here
        (exp(pars[1] * ub) - exp(pars[1] * lb)) / pars[1]
    end))
    
    return hazard_fn, cumhaz_fn
end

# Then create SemiMarkovHazard or MarkovHazard with these functions
```

---

## Common Patterns

### Evaluating Hazards Across Time Points
```julia
times = 0.0:0.1:10.0
hazard = model.hazards[1]
pars = model.parameters[1]
covars = Float64[]

haz_values = [hazard(t, pars, covars) for t in times]
```

### Computing Survival Probability
```julia
# S(t) = exp(-H(0,t))
function survprob(t, hazard, pars, covars)
    cumhaz = MultistateModels.cumulative_hazard(hazard, 0.0, t, pars, covars)
    exp(-cumhaz)
end
```

### Total Hazard from Multiple Transitions
```julia
# For competing risks from state s
function total_hazard(t, model, state, pars_vec, covars_vec)
    total = 0.0
    for (i, hazard) in enumerate(model.hazards)
        if hazard.statefrom == state
            total += hazard(t, pars_vec[i], covars_vec[i])
        end
    end
    return total
end
```

---

## Testing

### Unit Tests
```julia
using Test

# Test exponential property
hazard = model.hazards[1]  # Exponential
pars = model.parameters[1]
t = 2.0

haz = hazard(t, pars, Float64[])
cumhaz = MultistateModels.cumulative_hazard(hazard, 0.0, t, pars, Float64[])

@test cumhaz ≈ haz * t  # For exponential: H(0,t) = λt
```

### Integration Tests
```julia
# Run minimal test suite
include("scratch/test_phase2_minimal.jl")

# Run full test suite
include("test/runtests.jl")
```

---

## Troubleshooting

### "Method call_haz not found"
**Cause**: Missing backward compatibility layer  
**Fix**: Ensure you're on the `infrastructure_changes` branch with Phase 2.6 complete

### "Hazard type not recognized"
**Cause**: Trying to use splines (not yet implemented)  
**Fix**: Use "exp", "wei", or "gom" families only

### "Cannot modify parameters_ph"
**Cause**: NamedTuple immutability  
**Fix**: Use `update_parameters_ph!()` helper and handle externally (Phase 3 issue)

### "No method matching hazard()"
**Cause**: Using old _Hazard type that doesn't have callable interface  
**Fix**: Recreate model with Phase 2 code (should auto-create new types)

---

## Performance Tips

1. **Preallocate**: Reuse covariate arrays instead of allocating new ones
2. **Type Stability**: Ensure `pars` and `covars` have concrete types (e.g., `Vector{Float64}`)
3. **Avoid Closures**: Use runtime-generated functions (already done in Phase 2)
4. **Batch Evaluation**: Evaluate hazards for multiple times/subjects in vectorized manner

---

## Future Enhancements (Phase 3+)

### Planned Features
- [ ] Spline hazard generators
- [ ] ParameterHandling.jl full integration
- [ ] Mutable parameter structure
- [ ] Covariate handling in compatibility layer
- [ ] Direct likelihood evaluation (bypass compatibility layer)
- [ ] Custom hazard family registration system

### Under Consideration
- [ ] Automatic differentiation support
- [ ] GPU acceleration for batch evaluation
- [ ] Parallel hazard evaluation
- [ ] Caching of frequently-evaluated quantities

---

## API Stability

### Stable (Won't Change)
- Direct callable interface: `hazard(t, pars, covars)`
- Cumulative hazard helper: `cumulative_hazard(hazard, lb, ub, pars, covars)`
- Hazard struct fields (hazname, statefrom, stateto, etc.)

### Transitional (May Change in Phase 4)
- Backward compatibility: `call_haz`, `call_cumulhaz` (will be deprecated)
- Parameter structure (will become mutable in Phase 3)

### Internal (Subject to Change)
- Runtime function generation details
- Internal hazard evaluation helpers

---

## References

- **Phase 2 Complete Summary**: `PHASE2_COMPLETE.md`
- **Test Scripts**: 
  - `scratch/test_phase2_minimal.jl` (quick tests)
  - `scratch/test_phase2.jl` (comprehensive)
- **Source Files**:
  - `src/common.jl` (type definitions)
  - `src/hazards.jl` (function generators, callable interface)
  - `src/modelgeneration.jl` (build_hazards)
  - `src/helpers.jl` (parameter management)

---

**Last Updated**: Phase 2.7 completion  
**Branch**: infrastructure_changes  
**Status**: Ready for testing
