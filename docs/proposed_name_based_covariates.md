# Proposed Enhancement: Name-Based Covariate Matching

**Issue:** Runtime-generated hazard functions currently match parameters and covariates by **index**, which is fragile when different hazards depend on different covariates.

**Proposed Solution:** Match by **name** using NamedTuples.

---

## Current Problem

### Current Implementation (Index-Based)
```julia
# In generate_exponential_hazard(has_covariates=true):
hazard_fn = @RuntimeGeneratedFunction(:(
    function(t, pars, covars)
        log_baseline = pars[1]
        linear_pred = zero(eltype(pars))
        for i in 2:length(pars)
            linear_pred += pars[i] * covars[i-1]  # ← INDEX-BASED!
        end
        return exp(log_baseline + linear_pred)
    end
))
```

### Why This Breaks

Consider two hazards with different covariates:
```julia
h12 = Hazard(@formula(0 ~ 1 + age), "exp", 1, 2)       # Only age
h23 = Hazard(@formula(0 ~ 1 + trt + sex), "exp", 2, 3) # trt and sex
```

**Problem:** When evaluating `h23`, we need to extract `[trt_value, sex_value]` from the data row, but the code currently does `covars[1]`, `covars[2]` without knowing **which** columns those should be!

**Current workaround:** Assumes all hazards have the same covariates in the same order (NOT true in general!)

---

## Proposed Solution: Name-Based Matching

### Step 1: Store covariate names in hazard structs

Already done! The hazard structs have `parnames`:
```julia
struct MarkovHazard
    name::Symbol
    statefrom::Int64
    stateto::Int64
    family::String
    parnames::Vector{Symbol}  # ← e.g., [:h12_Intercept, :h12_age]
    # ...
end
```

### Step 2: Generate functions with named access

**New approach:** Generate functions that accept **NamedTuples** and access by name:

```julia
function generate_exponential_hazard(parnames::Vector{Symbol}, has_covariates::Bool)
    if !has_covariates
        # No change for baseline hazards
        hazard_fn = @RuntimeGeneratedFunction(:(
            function(t, pars, covars)
                return exp(pars[1])
            end
        ))
    else
        # NEW: Generate code that accesses covariates by name
        # parnames = [:h12_Intercept, :h12_age, :h12_trt]
        # Extract covariate names (skip intercept)
        covar_names = [name for name in parnames if !endswith(String(name), "Intercept")]
        
        # Build expression that sums over named covariates
        # linear_pred = pars[2]*covars.age + pars[3]*covars.trt
        terms = [:(pars[$i] * covars.$(Symbol(replace(String(covar_names[i-1]), r"^h\d+_" => ""))))
                 for i in 2:length(parnames)]
        
        linear_pred_expr = length(terms) == 0 ? :(zero(eltype(pars))) :
                          length(terms) == 1 ? terms[1] :
                          Expr(:call, :+, terms...)
        
        hazard_fn = @RuntimeGeneratedFunction(:(
            function(t, pars, covars::NamedTuple)
                log_baseline = pars[1]
                linear_pred = $linear_pred_expr
                return exp(log_baseline + linear_pred)
            end
        ))
    end
    
    return hazard_fn, cumhaz_fn
end
```

### Step 3: Pass NamedTuples at evaluation time

When evaluating likelihood:
```julia
# OLD (index-based):
covars = [subjdat[rowind, 7], subjdat[rowind, 8]]  # Which columns?!
haz_val = hazard(t, pars, covars)

# NEW (name-based):
covars = (age = subjdat[rowind, :age], 
          trt = subjdat[rowind, :trt])
haz_val = hazard(t, pars, covars)
```

---

## Benefits

### 1. **Robustness**
Different hazards can have different covariates without coordination:
```julia
h12 = Hazard(@formula(0 ~ 1 + age), "exp", 1, 2)
h23 = Hazard(@formula(0 ~ 1 + trt + sex + age), "exp", 2, 3)
h34 = Hazard(@formula(0 ~ 1), "exp", 3, 4)  # No covariates

# Each hazard's function only looks up the covariates it needs
```

### 2. **Clarity**
Code is self-documenting:
```julia
# Clear what's being used
linear_pred = pars[2] * covars.age + pars[3] * covars.trt

# vs opaque
linear_pred = pars[2] * covars[1] + pars[3] * covars[2]  # What are these?
```

### 3. **Safety**
Compiler catches typos and missing covariates:
```julia
# If we try to access covars.agee (typo), we get compile error
# If we try to access covars.missing_var, we get runtime error with clear message
```

### 4. **Flexibility**
Easy to add/remove covariates without worrying about indices:
```julia
# Add a new covariate to one hazard - no changes needed elsewhere
h12 = Hazard(@formula(0 ~ 1 + age + bmi), "exp", 1, 2)
```

---

## Implementation Plan

### Phase A: Update function generators (src/hazards.jl)

Update all 3 hazard generators:
1. `generate_exponential_hazard(parnames, has_covariates)`
2. `generate_weibull_hazard(parnames, has_covariates)`
3. `generate_gompertz_hazard(parnames, has_covariates)`

Each needs to:
- Accept `parnames` as argument
- Extract covariate names from `parnames` (strip "h##_" prefix, skip "Intercept")
- Generate expressions that access `covars.name` instead of `covars[i]`

### Phase B: Update call sites (src/modelgeneration.jl)

In `build_hazards()`:
```julia
# Before generating functions, prepare covariate names
covar_names = [Symbol(replace(String(pname), r"^h\d+_" => "")) 
               for pname in parnames if !endswith(String(pname), "Intercept")]

# Pass to generator
hazard_fn, cumhaz_fn = generate_exponential_hazard(parnames, has_covariates)

# Store covariate names in hazard struct (if not already there)
haz_struct = MarkovHazard(
    # ...
    parnames,
    covar_names,  # NEW field
    # ...
)
```

### Phase C: Update likelihood evaluation (src/likelihoods.jl, src/pathfunctions.jl)

Wherever hazards are called:
```julia
# OLD:
covars = rowind > size(data, 1) ? Float64[] : 
         [data[rowind, i] for i in 7:ncol(data)]

# NEW:
covars = if rowind > size(data, 1) || isempty(hazard.covar_names)
    NamedTuple()  # Empty NamedTuple for no covariates
else
    NamedTuple{Tuple(hazard.covar_names)}(
        tuple([data[rowind, cname] for cname in hazard.covar_names]...)
    )
end

haz_val = hazard(t, pars, covars)
```

### Phase D: Update tests

Add tests for:
- Hazards with different covariates in same model
- Covariate name mismatches (should error)
- Missing covariates in data (should error with clear message)

---

## Backward Compatibility

**Breaking change:** This changes the signature of runtime-generated functions from:
```julia
hazard_fn(t, pars::Vector, covars::Vector)
```
to:
```julia
hazard_fn(t, pars::Vector, covars::NamedTuple)
```

**Mitigation:** 
- This is internal API (hazard functions are generated, not user-facing)
- Phase 2 is still in development
- No existing user code depends on the signature

**If needed:** Could support both via multiple dispatch:
```julia
# Wrapper that accepts both
function (hazard::MarkovHazard)(t, pars, covars::Union{Vector,NamedTuple})
    if covars isa Vector
        # Convert to NamedTuple using stored covar_names
        nt_covars = NamedTuple{Tuple(hazard.covar_names)}(tuple(covars...))
        return hazard.hazard_fn(t, pars, nt_covars)
    else
        return hazard.hazard_fn(t, pars, covars)
    end
end
```

---

## Example: Complete Workflow

```julia
# Define hazards with different covariates
h12 = Hazard(@formula(0 ~ 1 + age), "exp", 1, 2)
h23 = Hazard(@formula(0 ~ 1 + trt + sex), "exp", 2, 3)

# Build model
model = multistatemodel(h12, h23; data = dat)

# Generated function for h12 looks like:
function hazard_h12(t, pars, covars::NamedTuple)
    log_baseline = pars[1]
    linear_pred = pars[2] * covars.age
    return exp(log_baseline + linear_pred)
end

# Generated function for h23 looks like:
function hazard_h23(t, pars, covars::NamedTuple)
    log_baseline = pars[1]
    linear_pred = pars[2] * covars.trt + pars[3] * covars.sex
    return exp(log_baseline + linear_pred)
end

# In likelihood evaluation for transition 1→2:
covars_12 = (age = dat[i, :age],)
haz_12 = model.hazards[1](t, pars_12, covars_12)  # Only needs age

# In likelihood evaluation for transition 2→3:
covars_23 = (trt = dat[i, :trt], sex = dat[i, :sex])
haz_23 = model.hazards[2](t, pars_23, covars_23)  # Needs trt and sex
```

---

## Estimated Effort

- **Phase A** (Update generators): 2-3 hours
- **Phase B** (Update build_hazards): 1-2 hours  
- **Phase C** (Update likelihood code): 2-3 hours
- **Phase D** (Tests): 1-2 hours
- **Total**: 6-10 hours

---

## Recommendation

**Implement this enhancement before Phase 3 completion** because:
1. It's a fundamental architectural improvement
2. Easier to change now than after release
3. Makes the codebase much more robust and maintainable
4. Aligns with Phase 3 goals (clean, type-safe parameter handling)

**Priority:** HIGH - This is a correctness issue, not just code quality.

**Current status:** The code works IF all hazards have the same covariates, but silently breaks otherwise. This is a latent bug waiting to happen.
