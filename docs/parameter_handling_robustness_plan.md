# Parameter Handling Robustness Plan
**Date**: December 7, 2024  
**Status**: PROPOSAL - Awaiting approval before implementation

---

## Executive Summary

**Current State**: MultistateModels.jl uses **positional parameter vectors** with minimal name-based validation, leading to fragility when users set parameters incorrectly (e.g., [scale, shape] instead of [shape, scale]).

**Problem**: Parameter names (`parnames`) exist during code generation but are **not stored** in runtime hazard objects, preventing validation at the point of parameter assignment.

**Root Cause of Recent Bug**: All Weibull/Gompertz tests used `[scale, shape]` order when hazard functions expect `[shape, scale]`, causing MCEM failures while exact inference succeeded (MLE finds correct values regardless of initialization).

**Proposal**: Implement comprehensive name-based parameter handling using ParameterHandling.jl's NamedTuple capabilities to provide runtime validation and eliminate positional dependencies.

---

## Current Architecture Audit

### 1. Parameter Storage Locations

#### A. Model-Level Storage (`model.parameters`)
```julia
# Structure: (flat, nested, natural, unflatten)
model.parameters = (
    flat = Vector{Float64},              # [θ₁, θ₂, ..., θₙ]
    nested = NamedTuple,                 # (h12=(baseline=[...], covariates=[...]), h23=...)
    natural = NamedTuple,                # (h12=[...], h23=[...]) - exp(baseline), covariates as-is
    unflatten = Function                 # flat → nested
)
```

**Files**:
- `src/common.jl` lines 650, 672, 694, 716, 738, 859: Model struct definitions
- `src/helpers.jl` lines 14-36: `rebuild_parameters()` function
- `src/helpers.jl` lines 266-283: `build_hazard_params()` - splits by position only

**Issue**: The `nested` structure has names at hazard level (`h12`, `h23`) and category level (`baseline`, `covariates`), but **not** at individual parameter level within those categories.

#### B. Hazard-Level Metadata
```julia
# Stored during build but NOT in runtime hazard objects
parnames::Vector{Symbol}  # e.g., [:h12_shape, :h12_scale, :h12_age]
```

**Files**:
- `src/modelgeneration.jl` lines 316, 321: Generate `_shape`, `_scale` names for Wei/Gom
- `src/common.jl` lines 187, 202, 224, 239, 265, 284: Hazard struct definitions
- `src/hazards.jl` lines 17-51: `extract_covar_names()` parses parnames to extract covariates

**Issue**: `parnames` used only during **code generation** (lines 332-354), then **discarded**. Runtime hazard objects have no `parnames` field.

### 2. Parameter Access Patterns

#### A. Positional Access (FRAGILE - 40+ locations)
```julia
# Weibull/Gompertz hazard functions
log_shape, log_scale = pars[1], pars[2]  # ❌ Assumes order!
```

**Files**:
- `src/hazards.jl` lines 208, 220, 240, 253: Time-transform methods
- `src/hazards.jl` lines 401, 416, 426, 441: Weibull PH/AFT functions
- `src/hazards.jl` lines 465, 474, 489, 499: Gompertz PH/AFT functions
- `src/phasetype.jl` lines 1028-1029, 1122: Phase-type single-phase case

**Issue**: No runtime check that `pars[1]` is actually the shape parameter.

#### B. Named Access via `nest_params` (BETTER)
```julia
# Converts flat → nested using model.parameters.unflatten
pars = nest_params(flat_params, model.parameters)
# Returns: (h12=(baseline=[...], covariates=[...]), h23=...)
```

**Files**:
- `src/helpers.jl` lines 342-376: `nest_params()` function - wraps `unflatten`
- `src/likelihoods.jl` lines 38, 702, 737: Used in likelihood functions
- `src/sampling.jl` lines 20, 140: Used in MCEM sampling
- `src/crossvalidation.jl` lines 96, 269, 343, 392, 444: Used in CV

**Limitation**: Still accesses baseline params positionally: `pars.h12.baseline[1]`

#### C. Vector-Based Access (LEGACY)
```julia
# Direct vector assignment - no validation
set_parameters!(model, (h12 = [log(scale), log(shape)], ...))  # ❌ Wrong order!
```

**Files**:
- `src/helpers.jl` lines 51-76: `set_parameters!(::Vector{Vector{Float64}})`
- `src/helpers.jl` lines 126-159: `set_parameters!(::NamedTuple)` - uses names for hazards, positions for params
- `src/helpers.jl` lines 184-210: `set_parameters!(::Int, ::Vector{Float64})`

**Issue**: Accepts vectors with no validation of parameter order or count beyond length check.

### 3. Parameter Transformation Points

#### A. Flatten/Unflatten Operations (8 locations)
```julia
params_flat, unflatten_fn = ParameterHandling.flatten(params_nested)
params_nested = unflatten_fn(params_flat)
```

**Files**:
- `src/helpers.jl` line 25: In `rebuild_parameters()` - creates unflatten function
- `src/modelfitting.jl` lines 238, 448: Unflatten after optimization
- `src/phasetype.jl` lines 1547, 1601, 3946, 4394: Phase-type model building

**Current**: Works correctly for structure, but underlying vectors are still positional.

#### B. Parameter Extraction (12 locations)
```julia
log_pars = get_log_scale_params(model.parameters)  # Returns Vector{Vector{Float64}}
```

**Files**:
- `src/helpers.jl` lines 336-340: Implementation - extracts `baseline` from nested
- `src/simulation.jl` line 593: Used for simulation
- `src/sampling.jl` lines 134, 273-274: Used in MCEM sampling
- `src/initialization.jl` lines 22, 68, 114: Used during initialization

**Issue**: Returns positional vectors with no parameter names attached.

### 4. Hazard Function Invocation (100+ locations)

#### Current Calling Convention:
```julia
hazard_fn(t, pars, covars)  # pars is Vector{Float64}, positional
cumhaz_fn(lb, ub, pars, covars)
```

**Files**:
- `src/hazards.jl` lines 359-380: Exponential hazard runtime functions
- `src/hazards.jl` lines 401-450: Weibull hazard runtime functions  
- `src/hazards.jl` lines 465-515: Gompertz hazard runtime functions
- `src/likelihoods.jl` lines 1803-1870: Path likelihood computation
- `src/simulation.jl` lines 593-650: Simulation path generation

**Verified**: Hazard callers do NOT use parameter names internally - they receive positional vectors.

---

## Identified Vulnerabilities

### Critical (High Impact, Frequent)

1. **`set_parameters!` accepts wrong parameter order** (Lines: helpers.jl 126-159)
   - Users must memorize that Wei/Gom use `[shape, scale]` not `[scale, shape]`
   - No validation at assignment time
   - Recent bug: ALL test files had this backwards

2. **Hazard functions use `pars[1], pars[2]` without validation** (40+ locations)
   - If parameters swapped, calculations silently wrong
   - No runtime check possible with current architecture

3. **`get_log_scale_params` returns anonymous vectors** (Lines: helpers.jl 336-340)
   - No way to verify which parameter is which
   - Downstream code assumes order by convention

### Moderate (Medium Impact, Occasional)

4. **`build_hazard_params` splits by position only** (Lines: helpers.jl 266-283)
   - Assumes first `npar_baseline` params are baseline
   - No name checking

5. **Parameter initialization assumes order** (Lines: initialization.jl 17, 65, 111)
   - `init_par()` returns positional vector
   - No guarantee order matches hazard expectations

6. **Spline parameter handling uses positional access** (Multiple locations in smooths.jl)
   - Spline coefficients accessed by index
   - More complex because of monotone transformations

### Low (Low Impact, Rare)

7. **Phase-type parameter indexing** (phasetype.jl, numerous locations)
   - Complex parameter structure with progression/exit rates
   - Already has some name-based access via `progression_param_indices`
   - Could be improved for consistency

---

## Proposed Solution: Leverage ParameterHandling.jl's NamedTuple Structure

**Key Insight**: ParameterHandling.jl already supports nested NamedTuples and preserves them through flatten/unflatten. We just need to use NamedTuples for baseline parameters instead of anonymous vectors.

### Phase 1: Convert Baseline Parameters to NamedTuples (Foundation)

#### 1.1 Store `parnames` in Runtime Hazard Structs

**Goal**: Make parameter names available at runtime for constructing named parameters.

**Implementation**:
```julia
# Modified hazard struct (example for SemiMarkovHazard)
struct SemiMarkovHazard <: AbstractHazard
    # ... existing fields ...
    parnames::Vector{Symbol}  # NEW: [:shape, :scale, :age, ...] (short names)
    npar_baseline::Int
    npar_total::Int
    # ... rest unchanged ...
end
```

**Note**: Store **short names** (`:shape`, `:scale`) not prefixed names (`:h12_shape`). The hazard name is already known from context.

**Changes Required**:
1. `src/common.jl` lines 224, 239, 265, 284: Add `parnames::Vector{Symbol}` field to all hazard types
2. `src/modelgeneration.jl` lines 276-298: Extract short parameter names and pass to constructors
3. `src/modelgeneration.jl` lines 316, 321: Generate short names for baseline params

**Example for Weibull**:
```julia
# Current (line 316):
parnames = [Symbol(string(ctx.hazname), "_shape"), Symbol(string(ctx.hazname), "_scale"), covariate_names...]

# New approach:
baseline_parnames = [:shape, :scale]  # Short names for baseline
full_parnames = [baseline_parnames..., covariate_names...]  # For documentation
hazard.parnames = full_parnames  # Store in hazard object
```

**Files Affected**: 
- `src/common.jl` (4 struct definitions)
- `src/modelgeneration.jl` (hazard builders: `_build_weibull_hazard`, `_build_gompertz_hazard`, etc.)

**Estimated Effort**: 3-4 hours

#### 1.2 Modify `build_hazard_params` to Return NamedTuple for Baseline

**Goal**: Convert baseline parameter vectors to NamedTuples with proper field names.

**Current Implementation**:
```julia
function build_hazard_params(log_scale_params::Vector{Float64}, npar_baseline::Int)
    baseline = log_scale_params[1:npar_baseline]  # ❌ Anonymous vector
    if npar_total > npar_baseline
        covariates = log_scale_params[(npar_baseline+1):end]
        return (baseline = baseline, covariates = covariates)
    else
        return (baseline = baseline,)
    end
end
```

**New Implementation**:
```julia
function build_hazard_params(
    log_scale_params::Vector{Float64}, 
    parnames::Vector{Symbol},  # NEW: parameter names
    npar_baseline::Int
)
    # Extract baseline parameter names (short names)
    baseline_names = parnames[1:npar_baseline]
    baseline_values = log_scale_params[1:npar_baseline]
    
    # Create NamedTuple for baseline parameters
    baseline = NamedTuple{Tuple(baseline_names)}(baseline_values)
    
    if length(log_scale_params) > npar_baseline
        # Extract covariate names and values
        covar_names = parnames[(npar_baseline+1):end]
        covar_values = log_scale_params[(npar_baseline+1):end]
        covariates = NamedTuple{Tuple(covar_names)}(covar_values)
        return (baseline = baseline, covariates = covariates)
    else
        return (baseline = baseline,)
    end
end
```

**Result Structure**:
```julia
# Instead of: (baseline = [log(1.5), log(0.2)], covariates = [0.3, 0.1])
# We get:     (baseline = (shape = log(1.5), scale = log(0.2)), covariates = (age = 0.3, sex = 0.1))
```

**Changes Required**:
1. `src/helpers.jl` lines 266-283: Rewrite `build_hazard_params` signature and body
2. Update all callers to pass `parnames`:
   - `src/helpers.jl` line 18: In `rebuild_parameters`, extract parnames from hazard objects
   - `src/modelgeneration.jl` line 761: In `build_parameters`, pass parnames

**Files Affected**:
- `src/helpers.jl` (`build_hazard_params`, `rebuild_parameters`)
- `src/modelgeneration.jl` (`build_parameters`)

**Estimated Effort**: 4-5 hours

#### 1.3 Update Helper Functions for NamedTuple Extraction

**Goal**: Convert NamedTuple parameters back to vectors when needed for legacy code.

**New Functions**:
```julia
# Extract baseline values as vector (for legacy code that needs vectors)
function extract_baseline_values(hazard_params::NamedTuple)
    return collect(values(hazard_params.baseline))
end

# Extract covariate values as vector
function extract_covariate_values(hazard_params::NamedTuple)
    return haskey(hazard_params, :covariates) ? 
           collect(values(hazard_params.covariates)) : Float64[]
end

# Extract full parameter vector (baseline + covariates) - replaces extract_params_vector
function extract_params_vector(hazard_params::NamedTuple)
    baseline_vals = collect(values(hazard_params.baseline))
    if haskey(hazard_params, :covariates)
        covar_vals = collect(values(hazard_params.covariates))
        return vcat(baseline_vals, covar_vals)
    else
        return baseline_vals
    end
end

# Extract natural scale vector (for model.parameters.natural)
function extract_natural_vector(hazard_params::NamedTuple)
    baseline_natural = exp.(collect(values(hazard_params.baseline)))
    if haskey(hazard_params, :covariates)
        covar_vals = collect(values(hazard_params.covariates))
        return vcat(baseline_natural, covar_vals)
    else
        return baseline_natural
    end
end
```

**Changes Required**:
- `src/helpers.jl` lines 285-310: Update existing functions to work with NamedTuple baseline

**Files Affected**:
- `src/helpers.jl` (parameter extraction functions)

**Estimated Effort**: 2-3 hours

#### 1.4 Verify ParameterHandling.jl Compatibility

**Goal**: Ensure ParameterHandling.jl's flatten/unflatten works with nested NamedTuples.

**Test Case**:
```julia
# Structure with NamedTuple baseline
params_nested = (
    h12 = (
        baseline = (shape = log(1.5), scale = log(0.2)),
        covariates = (age = 0.3, sex = 0.1)
    ),
    h23 = (
        baseline = (shape = log(2.0)),
    )
)

# Test flatten/unflatten
params_flat, unflatten = ParameterHandling.flatten(params_nested)
params_reconstructed = unflatten(params_flat)

@test params_reconstructed.h12.baseline.shape ≈ log(1.5)
@test params_reconstructed.h12.baseline.scale ≈ log(0.2)
@test params_reconstructed.h12.covariates.age ≈ 0.3
```

**Expected Result**: ✅ Should work out of the box - ParameterHandling.jl supports this pattern.

**Changes Required**:
- Add test in `test/test_helpers.jl`

**Estimated Effort**: 1 hour

### Phase 2: Update Hazard Functions for Named Access

#### 2.1 Modify Hazard Function Signatures to Accept NamedTuples

**Goal**: Hazard functions should access baseline parameters by name, not position.

**Current (Weibull PH)**:
```julia
function generate_weibull_hazard(parnames::Vector{Symbol}, linpred_effect::Symbol)
    linear_pred_expr = _build_linear_pred_expr(parnames, 3)
    
    hazard_fn = @RuntimeGeneratedFunction(:(
        function(t, pars, covars)
            log_shape, log_scale = pars[1], pars[2]  # ❌ Positional
            shape = exp(log_shape)
            linear_pred = $linear_pred_expr
            # ...
        end
    ))
end
```

**New (Weibull PH)**:
```julia
function generate_weibull_hazard(parnames::Vector{Symbol}, linpred_effect::Symbol)
    # Extract baseline parameter names (first npar_baseline elements)
    baseline_names = parnames[1:2]  # [:shape, :scale] for Weibull
    
    # Build linear predictor for covariates (if any)
    linear_pred_expr = if length(parnames) > 2
        covar_names = parnames[3:end]
        _build_linear_pred_expr_named(covar_names)
    else
        :(0.0)
    end
    
    hazard_fn = @RuntimeGeneratedFunction(:(
        function(t, pars, covars)
            # Access baseline by name
            log_shape = pars.baseline.shape  # ✅ Named!
            log_scale = pars.baseline.scale  # ✅ Named!
            shape = exp(log_shape)
            
            # Linear predictor accesses covariates by name
            linear_pred = $linear_pred_expr
            
            # Rest of hazard computation unchanged
            log_haz = log_scale + log_shape + linear_pred
            if shape != 1.0
                log_haz += (shape - 1) * log(t)
            end
            return exp(log_haz)
        end
    ))
end
```

**Key Changes**:
1. Access baseline: `pars.baseline.shape` instead of `pars[1]`
2. Access covariates: `pars.covariates.age` instead of `pars[3]` (via updated linear predictor)
3. Pass full hazard params `(baseline=..., covariates=...)` not just flat vector

**Changes Required**:
- `src/hazards.jl` lines 348-380: Update `generate_exponential_hazard`
- `src/hazards.jl` lines 395-450: Update `generate_weibull_hazard` (PH and AFT)
- `src/hazards.jl` lines 455-515: Update `generate_gompertz_hazard` (PH and AFT)
- `src/hazards.jl` lines 520+: Update spline hazard generators

**Files Affected**:
- `src/hazards.jl` (all hazard generators)

**Estimated Effort**: 8-10 hours (careful work needed for RuntimeGeneratedFunctions)

#### 2.2 Update Linear Predictor Builder for Named Covariate Access

**Goal**: Build linear predictor that accesses covariates by name.

**Current**:
```julia
function _build_linear_pred_expr(parnames::Vector{Symbol}, start_idx::Int)
    # Generates: covars[1] * pars[3] + covars[2] * pars[4] + ...
    # Uses positional indexing
end
```

**New**:
```julia
function _build_linear_pred_expr_named(covar_names::Vector{Symbol})
    """
    Build expression that accesses covariates by name from pars.covariates NamedTuple.
    
    For covar_names = [:age, :sex], generates:
    covars[1] * pars.covariates.age + covars[2] * pars.covariates.sex
    
    Note: covar VALUES still come from covars vector (positional) because that's
    how data is organized. Only COEFFICIENTS are accessed by name.
    """
    if isempty(covar_names)
        return :(0.0)
    end
    
    terms = [:(covars[$i] * pars.covariates.$(covar_names[i])) 
             for i in eachindex(covar_names)]
    
    return Expr(:call, :+, terms...)
end
```

**Rationale**: 
- Covariate **coefficients** accessed by name: `pars.covariates.age`
- Covariate **values** still positional: `covars[1]` (comes from data matrix)
- This is fine because covariate order is fixed by formula at model build time

**Changes Required**:
- `src/hazards.jl` lines 65-85: Replace or supplement `_build_linear_pred_expr`

**Files Affected**:
- `src/hazards.jl` (linear predictor builder)

**Estimated Effort**: 2-3 hours

#### 2.3 Update Hazard Callers to Pass NamedTuple Parameters

**Goal**: Pass structured parameters (with baseline/covariates) to hazard functions.

**Current (in `_compute_path_loglik_fused`)**:
```julia
hazard_pars = pars[h]  # Gets flat vector for hazard h
haz_value = eval_hazard(hazard, ub, hazard_pars, covars)
```

**New**:
```julia
hazard_pars = pars[hazname]  # Gets (baseline=..., covariates=...) for hazard
haz_value = eval_hazard(hazard, ub, hazard_pars, covars)
```

**Note**: `pars` is now the result of `unflatten`, which returns NamedTuple by hazard name, not hazard number.

**Challenge**: Likelihood code currently uses hazard **indices** (1, 2, 3...), not names (h12, h23...).

**Solution Options**:

**Option A - Convert index to name** (minimal changes):
```julia
# In _compute_path_loglik_fused
hazname = collect(keys(pars))[h]  # Convert hazard index to name
hazard_pars = pars[hazname]
```

**Option B - Pass hazard names in** (cleaner):
```julia
# Modify function signature to include hazard names
function _compute_path_loglik_fused(
    path::SamplePath, 
    pars,  # NamedTuple by hazard name
    hazards::Vector{<:_Hazard},
    totalhazards::Vector{<:_TotalHazard},
    hazard_names::Vector{Symbol},  # NEW: map index → name
    tmat::Matrix{Int64},
    # ...
)
    # Access by name
    for h in tothaz.components
        hazname = hazard_names[h]
        hazard_pars = pars[hazname]
        # ...
    end
end
```

**Recommendation**: Option B - pass hazard names explicitly for clarity.

**Changes Required**:
- `src/likelihoods.jl` lines 1803-1920: Update `_compute_path_loglik_fused` signature and body
- `src/likelihoods.jl` lines 700-750: Update callers to pass hazard names
- `src/simulation.jl` lines 593-650: Update simulation code similarly

**Files Affected**:
- `src/likelihoods.jl` (path likelihood)
- `src/simulation.jl` (simulation)
- `src/sampling.jl` (MCEM sampling)

**Estimated Effort**: 6-8 hours

### Phase 3: Validated Parameter Setting

#### 3.1 Enhanced `set_parameters!` with Validation

**Goal**: Validate parameter names and order when users set parameters.

**Implementation**:
```julia
# New method accepting NamedTuple with explicit parameter names
function set_parameters!(model::MultistateProcess, newvalues::NamedTuple; validate::Bool=true)
    for (hazname, params) in pairs(newvalues)
        # Get expected parameter names for this hazard
        hazind = model.hazkeys[hazname]
        expected_names = model.parameters.parnames[hazname]
        
        if validate && params isa NamedTuple
            # User provided named parameters - validate against expected
            provided_names = keys(params)
            
            # Check all expected parameters provided
            missing_params = setdiff(expected_names, provided_names)
            !isempty(missing_params) && 
                error("Missing parameters for $hazname: $missing_params")
            
            # Check no unexpected parameters
            extra_params = setdiff(provided_names, expected_names)
            !isempty(extra_params) && 
                error("Unexpected parameters for $hazname: $extra_params")
            
            # Reorder to match expected order
            param_vec = [params[name] for name in expected_names]
        elseif params isa AbstractVector
            # Positional vector - validate length and warn
            length(params) == length(expected_names) ||
                error("Parameter vector length $(length(params)) doesn't match expected $(length(expected_names)) for $hazname")
            
            if validate
                @warn """Setting parameters for $hazname using positional vector. 
                        Expected order: $expected_names. 
                        Use named tuple for safety: ($(expected_names[1])=..., $(expected_names[2])=..., ...)"""
            end
            param_vec = params
        else
            error("Parameters for $hazname must be NamedTuple or Vector, got $(typeof(params))")
        end
        
        # ... rest of update logic ...
    end
end

# Convenience: accept keyword arguments
function set_parameters!(model::MultistateProcess; kwargs...)
    set_parameters!(model, NamedTuple(kwargs))
end
```

**Usage Examples**:
```julia
# Safe: Named parameters (order-independent)
set_parameters!(model, h12=(shape=1.5, scale=0.2, age=0.3))

# Safe: Keyword arguments
set_parameters!(model; h12=(shape=1.5, scale=0.2))

# Legacy: Positional (with warning if validate=true)
set_parameters!(model, (h12=[1.5, 0.2, 0.3],))

# Disable validation for performance-critical code
set_parameters!(model, (h12=[1.5, 0.2, 0.3],); validate=false)
```

**Changes Required**:
- `src/helpers.jl` lines 126-159: Rewrite with validation logic
- Add tests in `test/test_helpers.jl`

**Estimated Effort**: 6-8 hours

#### 2.3 Named Parameter Extraction Functions

**Goal**: Return parameters with names attached for downstream use.

**Implementation**:
```julia
# Enhanced version returning NamedParameterVector
function get_log_scale_params_named(model::MultistateProcess)
    return NamedTuple(
        hazname => NamedParameterVector(
            collect(model.parameters.nested[hazname].baseline),
            model.parameters.parnames[hazname][1:model.hazards[idx].npar_baseline]
        )
        for (hazname, idx) in model.hazkeys
    )
end

# Enhanced nest_params returning named vectors
function nest_params_named(flat_params::AbstractVector, parameters)
    nested = parameters.unflatten(flat_params)
    return NamedTuple(
        hazname => NamedParameterVector(
            vcat(nested[hazname].baseline, 
                 get(nested[hazname], :covariates, Float64[])),
            parameters.parnames[hazname]
        )
        for hazname in keys(nested)
    )
end
```

**Benefits**:
- Downstream code can use `pars[:h12_shape]` instead of `pars[1]`
- Backward compatible: positional access still works

**Changes Required**:
- `src/helpers.jl`: Add new functions alongside existing ones
- Update callers gradually (Tier 3)

**Estimated Effort**: 3-4 hours

### Tier 3: Named Access in Hazard Functions (Long-term Refactor)

#### 3.1 Modify Hazard Function Signatures (BREAKING CHANGE)

**Goal**: Pass named parameters to hazard functions for internal validation.

**Current**:
```julia
hazard_fn(t, pars::Vector{Float64}, covars)
```

**Proposed**:
```julia
hazard_fn(t, pars::NamedParameterVector, covars)
# Or maintain compatibility:
hazard_fn(t, pars::Union{Vector{Float64}, NamedParameterVector}, covars)
```

**Changes Required**:
- `src/hazards.jl` lines 196-260: Update time-transform methods
- `src/hazards.jl` lines 359-515: Update runtime hazard functions
- `src/likelihoods.jl` lines 1803-1870: Update callers
- `src/simulation.jl` lines 593-650: Update simulation code

**Estimated Effort**: 20-30 hours (high impact, many call sites)

#### 3.2 Use Named Access in Hazard Function Bodies

**Goal**: Replace `pars[1], pars[2]` with `pars[:shape], pars[:scale]`.

**Before**:
```julia
function generate_weibull_hazard(parnames::Vector{Symbol}, linpred_effect::Symbol)
    # ...
    hazard_fn = @RuntimeGeneratedFunction(:(
        function(t, pars, covars)
            log_shape, log_scale = pars[1], pars[2]  # ❌ Positional
            # ...
        end
    ))
end
```

**After**:
```julia
function generate_weibull_hazard(parnames::Vector{Symbol}, linpred_effect::Symbol)
    # Extract baseline parameter names
    shape_name = parnames[1]  # e.g., :h12_shape
    scale_name = parnames[2]  # e.g., :h12_scale
    
    hazard_fn = @RuntimeGeneratedFunction(:(
        function(t, pars, covars)
            log_shape = pars[$(QuoteNode(shape_name))]  # ✅ Named!
            log_scale = pars[$(QuoteNode(scale_name))]
            # ... rest unchanged ...
        end
    ))
end
```

**Challenges**:
- RuntimeGeneratedFunctions.jl requires careful quoting for interpolated symbols
- Need to pass `parnames` to hazard generators (currently not passed)
- Performance impact needs benchmarking

**Changes Required**:
- `src/hazards.jl` lines 348-515: Update all hazard generators to use parnames
- `src/modelgeneration.jl` lines 276-298: Pass parnames to generators

**Estimated Effort**: 15-20 hours

**Alternative (Less Invasive)**: Keep positional access in generated functions but add runtime assertions:
```julia
@assert haskey(pars, :h12_shape) "Expected parameter :h12_shape not found"
log_shape = pars[1]  # Still positional for performance
```

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1) - 10-15 hours
**Goal**: Add parameter names to runtime objects, no breaking changes.

1. Add `parnames` field to hazard structs (**2-3 hours**)
2. Store `parnames` in `model.parameters` (**3-4 hours**)
3. Create `NamedParameterVector` type (**4-5 hours**)
4. Write comprehensive tests (**3-4 hours**)

**Deliverables**:
- Parameter names available at runtime
- New data structure for named access
- All existing code still works (backward compatible)

**Risk**: Low - purely additive changes

### Phase 2: Validation Layer (Week 2) - 15-20 hours
**Goal**: Add opt-in validation to parameter setting.

1. Implement validated `set_parameters!` (**6-8 hours**)
2. Add `get_log_scale_params_named()` and `nest_params_named()` (**3-4 hours**)
3. Update documentation with naming conventions (**2-3 hours**)
4. Add validation tests for wrong parameter order (**4-5 hours**)

**Deliverables**:
- Users can opt into validation with named parameters
- Warnings for positional usage
- Clear documentation of expected parameter order

**Risk**: Low-Medium - new methods alongside existing ones

### Phase 3: Migration (Weeks 3-4) - 20-30 hours
**Goal**: Gradually migrate internal code to use named access.

1. Update `set_parameters!` calls in codebase (**4-6 hours**)
2. Update parameter extraction in tests (**4-6 hours**)
3. Add named access to hazard function internals (optional) (**12-18 hours**)
4. Performance benchmarking (**3-4 hours**)

**Deliverables**:
- Internal code uses named parameters
- Tests verify correct parameter order
- Performance impact documented

**Risk**: Medium - requires careful testing to avoid regressions

### Phase 4: Enforcement (Week 5) - 5-10 hours
**Goal**: Make validation mandatory, deprecate positional-only usage.

1. Enable validation by default in `set_parameters!` (**1-2 hours**)
2. Add deprecation warnings for positional usage (**2-3 hours**)
3. Update all user-facing documentation (**2-3 hours**)
4. Create migration guide for external users (**2-3 hours**)

**Deliverables**:
- Parameter order errors caught at assignment time
- Clear migration path for users
- Breaking change well-documented

**Risk**: Medium - may break user code relying on positional access

---

## Alternative Approaches Considered

### Alternative 1: Dictionaries Instead of NamedParameterVector

**Pros**: More flexible, natural named access
**Cons**: Loss of ordering, no Vector{Float64} compatibility, performance overhead

**Verdict**: Rejected - too disruptive, ParameterHandling.jl expects vectors

### Alternative 2: Macro-Based Parameter Validation

**Approach**: Use `@set_parameters!` macro to validate at parse time
**Pros**: Zero runtime cost after validation
**Cons**: Can't validate dynamically constructed parameters, macro complexity

**Verdict**: Consider for future optimization after runtime validation proven

### Alternative 3: Static Parameter Types per Hazard Family

**Approach**: Define `WeibullParams`, `GompertzParams` structs with named fields
**Pros**: Compile-time safety, self-documenting
**Cons**: High complexity, breaks ParameterHandling.jl integration, massive refactor

**Verdict**: Too disruptive for current architecture

---

## Testing Strategy

### Unit Tests (New File: `test/test_named_parameters.jl`)

1. **NamedParameterVector Tests**:
   - Construction with matching/mismatching lengths
   - Positional access `pars[1]`
   - Named access `pars[:h12_shape]`
   - Invalid name access error handling
   - Vector conversion
   - Iteration over pairs

2. **Validation Tests**:
   - Correct named parameter assignment succeeds
   - Wrong parameter order detected
   - Missing parameters detected
   - Extra parameters detected
   - Positional with warning works
   - Validation disable works

3. **Integration Tests** (Update existing test files):
   - Set Weibull parameters with `(shape=..., scale=...)`
   - Set Gompertz parameters with `(shape=..., scale=...)`
   - Verify simulation/inference with named params
   - Cross-check positional and named produce same results

### Regression Tests

- Run all existing tests with validation enabled
- Verify no performance degradation (benchmark critical paths)
- Check memory usage (NamedParameterVector overhead)

---

## Documentation Updates Required

### User-Facing Documentation

1. **New Section**: "Parameter Naming Conventions" (docs/src/parameters.md)
   - Table of parameter order for each hazard family
   - Examples of safe parameter setting with names
   - Migration guide from positional to named

2. **Update**: API Reference for `set_parameters!`
   - Document new named parameter signature
   - Show keyword argument usage
   - Explain validation behavior

3. **Update**: Hazard family documentation
   - Add "Parameter Order" subsection to each family
   - Warn about positional fragility

### Developer Documentation

1. **New Guide**: "Contributing - Parameter Handling" (docs/src/dev/parameters.md)
   - Architecture overview
   - When to use NamedParameterVector
   - How to add new hazard families
   - Testing guidelines for parameter code

2. **Update**: Architecture documentation
   - Explain three-tier parameter system
   - Document parnames flow from generation to runtime

---

## Performance Considerations

### Potential Overhead Sources

1. **NamedParameterVector**: Wrapper around Vector adds indirection
   - **Mitigation**: Inline getindex, use @inbounds in tight loops
   - **Benchmark**: Measure overhead in likelihood evaluation

2. **Named Access in Hazard Functions**: Symbol lookup vs integer indexing
   - **Mitigation**: Keep positional access in generated code, validate at boundary
   - **Alternative**: Pre-compute indices at build time

3. **Validation Logic**: Additional checks in `set_parameters!`
   - **Mitigation**: Make validation opt-out for performance-critical code
   - **Impact**: Minimal - parameters set infrequently relative to evaluation

### Benchmarking Plan

```julia
# Before and after Phase 2
@benchmark loglik_exact($model, $data)
@benchmark simulate($model)
@benchmark fit($model)  # For MCEM

# Acceptable thresholds:
# - Likelihood evaluation: <5% overhead
# - Simulation: <5% overhead
# - MCEM: <2% overhead (parameter setting is small fraction of time)
```

---

## Migration Path for Users

### Immediate (Phase 1-2): Opt-In Validation

Users can continue using positional parameters, no breaking changes:
```julia
# Still works (with warning if validate=true)
set_parameters!(model, (h12=[log(1.5), log(0.2)],))
```

### Recommended (Phase 2-3): Named Parameters

Encourage users to adopt named parameters:
```julia
# Safe, order-independent
set_parameters!(model, h12=(shape=1.5, scale=0.2))
```

### Required (Phase 4): Validation Enforced

After sufficient adoption period (e.g., 6 months), make validation default:
```julia
# Version 2.0: This will error if order wrong
set_parameters!(model, (h12=[log(0.2), log(1.5)],))  # ❌ Detected!

# Users must use named parameters or explicitly disable validation
set_parameters!(model, h12=(scale=0.2, shape=1.5))  # ✅ Order doesn't matter
# OR
set_parameters!(model, (h12=[log(1.5), log(0.2)],); validate=false)  # ✅ Opt-out
```

---

## Open Questions for Discussion

1. **Breaking Change Timeline**: When to make validation mandatory? Immediate with opt-out, or after deprecation period?

2. **Covariate Parameters**: Should covariate coefficients also use named access, or keep positional (order defined by formula)?

3. **Performance Threshold**: What's acceptable overhead for safety? 5%? 10%?

4. **Phase-Type Parameters**: These have complex structure (progression/exit rates). Extend naming system or handle separately?

5. **Spline Parameters**: Coefficients are basis-dependent. Use generic names (`sp1`, `sp2`, ...) or more semantic names?

6. **Backward Compatibility**: Support positional access indefinitely with deprecation warnings, or eventually remove?

7. **External Optimization**: How do external optimizers (Ipopt, etc.) interact with named parameters? Flatten to Vector still needed?

---

## Success Criteria

### Must Have (Phase 1-2)
- ✅ Parameter names stored at runtime in hazard objects and model.parameters
- ✅ `set_parameters!` accepts NamedTuple with parameter names
- ✅ Validation detects wrong parameter order when enabled
- ✅ All existing tests pass with no changes
- ✅ Performance overhead <5% on critical paths

### Should Have (Phase 3)
- ✅ Internal codebase uses named parameters
- ✅ Tests verify parameter naming explicitly
- ✅ Documentation clearly explains parameter order for all families
- ✅ Migration guide for users

### Nice to Have (Phase 4)
- ✅ Hazard functions use named access internally
- ✅ Validation enabled by default
- ✅ Comprehensive error messages with correction suggestions
- ✅ Auto-reordering for common mistakes (e.g., detect [scale, shape] and fix)

---

## Recommendation

**Proceed with Phased Implementation**:

1. **Start with Phase 1** immediately - adds foundation with zero breaking changes
2. **Deploy Phase 2** after testing - opt-in validation provides immediate value
3. **Evaluate adoption** - if users adopt named parameters, proceed to Phase 3
4. **Phase 4 only after 6-12 month migration period** - give users time to update code

**Key Principle**: Safety without breaking existing functionality. Build the infrastructure first, then encourage adoption through warnings and documentation, finally enforce when ecosystem ready.

**Estimated Total Effort**: 50-75 hours over 5 weeks, but can be staged incrementally.

---

## Next Steps

1. **Review this plan** - discuss timeline, priorities, open questions
2. **Approve Phase 1** - if acceptable, begin implementation of foundation
3. **Prototype NamedParameterVector** - validate performance characteristics
4. **Update project roadmap** - integrate into release schedule

**Awaiting approval to proceed.**
