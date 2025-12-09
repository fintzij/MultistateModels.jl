# Parameter Handling Implementation Plan
**Date**: December 7, 2024  
**Status**: DETAILED IMPLEMENTATION PLAN - Ready for review

---

## Executive Summary

**Approach**: Leverage ParameterHandling.jl's existing NamedTuple support to eliminate positional parameter fragility.

**Key Change**: Store baseline parameters as `(shape=..., scale=...)` instead of `[..., ...]` within the nested structure.

**Benefits**:
- ✅ Order-independent parameter setting
- ✅ Validation at assignment time
- ✅ Self-documenting code (`pars.baseline.shape` vs `pars[1]`)
- ✅ ParameterHandling.jl's flatten/unflatten handles everything
- ✅ Full AD compatibility maintained

---

## Phase 1: Foundation - NamedTuple Baseline Parameters (Week 1)

### Estimated Effort: 12-15 hours

### 1.1 Add `parnames` to Hazard Structs (3-4 hours)

**Goal**: Store parameter names in runtime objects for validation and NamedTuple construction.

**Changes**:

**File**: `src/common.jl`

Add `parnames::Vector{Symbol}` field to all hazard types:
- Line 187: `MarkovHazard`
- Line 202: `MarkovHazardCensored`
- Line 224: `SemiMarkovHazard`
- Line 239: `SemiMarkovHazardCensored`  
- Line 265: `_SplineHazard`
- Line 284: `_PhaseTypeHazard`

**Example**:
```julia
struct SemiMarkovHazard <: AbstractHazard
    hazard::Function
    cumhaz::Function
    parnames::Vector{Symbol}  # NEW: [:shape, :scale, :age, :sex, ...]
    npar_baseline::Int
    npar_total::Int
    metadata::HazardMetadata
end
```

**Note**: Store SHORT names (`:shape`, `:scale`, `:age`) not prefixed names (`:h12_shape`).

**File**: `src/modelgeneration.jl`

Update hazard builders to pass `parnames` to constructors:
- Lines 276-298: `_build_exponential_hazard`, `_build_weibull_hazard`, `_build_gompertz_hazard`
- Extract short names by stripping hazard prefix

**Example for Weibull** (line 316):
```julia
# Current:
parnames = [Symbol(string(ctx.hazname), "_shape"), 
            Symbol(string(ctx.hazname), "_scale"),
            covariate_names...]

# New approach - store short names:
baseline_names = [:shape, :scale]
full_parnames = [baseline_names..., covariate_names...]

hazard_struct = SemiMarkovHazard(
    hazard_fn,
    cumhaz_fn,
    full_parnames,  # NEW parameter
    2,  # npar_baseline
    length(full_parnames),
    metadata
)
```

### 1.2 Convert `build_hazard_params` to Return NamedTuple Baseline (4-5 hours)

**Goal**: Transform baseline from anonymous vector to named NamedTuple.

**File**: `src/helpers.jl` lines 266-283

**Current**:
```julia
function build_hazard_params(log_scale_params::Vector{Float64}, npar_baseline::Int)
    baseline = log_scale_params[1:npar_baseline]  # ❌ Vector
    if length(log_scale_params) > npar_baseline
        covariates = log_scale_params[(npar_baseline+1):end]
        return (baseline = baseline, covariates = covariates)
    else
        return (baseline = baseline,)
    end
end
```

**New**:
```julia
function build_hazard_params(
    log_scale_params::Vector{Float64}, 
    parnames::Vector{Symbol},
    npar_baseline::Int
)
    # Extract names for baseline and covariates
    baseline_names = parnames[1:npar_baseline]
    baseline_values = log_scale_params[1:npar_baseline]
    
    # Create NamedTuple for baseline
    baseline = NamedTuple{Tuple(baseline_names)}(Tuple(baseline_values))
    
    if length(log_scale_params) > npar_baseline
        covar_names = parnames[(npar_baseline+1):end]
        covar_values = log_scale_params[(npar_baseline+1):end]
        covariates = NamedTuple{Tuple(covar_names)}(Tuple(covar_values))
        return (baseline = baseline, covariates = covariates)
    else
        return (baseline = baseline,)
    end
end
```

**Result**:
```julia
# Instead of: (baseline = [log(1.5), log(0.2)], covariates = [0.3, 0.1])
# Now:        (baseline = (shape=log(1.5), scale=log(0.2)), covariates = (age=0.3, sex=0.1))
```

**Update Callers**:

1. **`rebuild_parameters`** (src/helpers.jl line 18):
```julia
function rebuild_parameters(new_param_vectors::Vector{Vector{Float64}}, model::MultistateProcess)
    params_nested_pairs = [
        hazname => build_hazard_params(
            new_param_vectors[idx],
            model.hazards[idx].parnames,  # NEW: pass parnames
            model.hazards[idx].npar_baseline
        )
        for (hazname, idx) in sort(collect(model.hazkeys), by = x -> x[2])
    ]
    # ... rest unchanged ...
end
```

2. **`build_parameters`** (src/modelgeneration.jl line 761):
```julia
function build_parameters(parameters::Vector{Vector{Float64}}, 
                         hazkeys::Dict{Symbol, Int64}, 
                         hazards::Vector{<:_Hazard})
    params_nested_pairs = [
        hazname => build_hazard_params(
            parameters[idx],
            hazards[idx].parnames,  # NEW: pass parnames
            hazards[idx].npar_baseline
        )
        for (hazname, idx) in sort(collect(hazkeys), by = x -> x[2])
    ]
    # ... rest unchanged ...
end
```

### 1.3 Update Helper Functions for NamedTuple Extraction (2-3 hours)

**Goal**: Functions that extract parameter values need to handle NamedTuple baseline.

**File**: `src/helpers.jl` lines 285-310

**Update `extract_params_vector`**:
```julia
function extract_params_vector(hazard_params::NamedTuple)
    """Extract full parameter vector from nested NamedTuple structure."""
    baseline_vals = collect(values(hazard_params.baseline))
    if haskey(hazard_params, :covariates)
        covar_vals = collect(values(hazard_params.covariates))
        return vcat(baseline_vals, covar_vals)
    else
        return baseline_vals
    end
end
```

**Update `extract_natural_vector`**:
```julia
function extract_natural_vector(hazard_params::NamedTuple)
    """Extract natural-scale parameters (exp for baseline, as-is for covariates)."""
    baseline_natural = exp.(collect(values(hazard_params.baseline)))
    if haskey(hazard_params, :covariates)
        covar_vals = collect(values(hazard_params.covariates))
        return vcat(baseline_natural, covar_vals)
    else
        return baseline_natural
    end
end
```

**Add utility functions**:
```julia
function extract_baseline_values(hazard_params::NamedTuple)
    """Extract baseline parameter values as vector."""
    return collect(values(hazard_params.baseline))
end

function extract_covariate_values(hazard_params::NamedTuple)
    """Extract covariate coefficient values as vector."""
    return haskey(hazard_params, :covariates) ? 
           collect(values(hazard_params.covariates)) : Float64[]
end
```

### 1.4 Verification Testing (2-3 hours)

**Goal**: Ensure ParameterHandling.jl works with new structure.

**File**: `test/test_helpers.jl`

**Test ParameterHandling.jl flatten/unflatten**:
```julia
@testset "ParameterHandling.jl with NamedTuple Baseline" begin
    # Create structure with named baseline
    params_nested = (
        h12 = (
            baseline = (shape = log(1.5), scale = log(0.2)),
            covariates = (age = 0.3, sex = 0.1)
        ),
        h23 = (
            baseline = (shape = log(2.0)),
        )
    )
    
    # Test flatten
    flat, unflatten = ParameterHandling.flatten(params_nested)
    @test length(flat) == 5  # 2 + 2 + 1
    
    # Test unflatten
    reconstructed = unflatten(flat)
    @test reconstructed.h12.baseline.shape ≈ log(1.5)
    @test reconstructed.h12.baseline.scale ≈ log(0.2)
    @test reconstructed.h12.covariates.age ≈ 0.3
    @test reconstructed.h12.covariates.sex ≈ 0.1)
    @test reconstructed.h23.baseline.shape ≈ log(2.0)
    
    # Test with ForwardDiff (AD compatibility)
    using ForwardDiff
    function test_obj(flat_params)
        p = unflatten(flat_params)
        return sum(exp.(values(p.h12.baseline))) + sum(values(p.h12.covariates))
    end
    
    grad = ForwardDiff.gradient(test_obj, flat)
    @test length(grad) == 5
    @test all(isfinite.(grad))
end
```

**Test helper functions**:
```julia
@testset "Parameter Extraction with NamedTuple" begin
    hazard_params = (
        baseline = (shape = log(1.5), scale = log(0.2)),
        covariates = (age = 0.3, sex = 0.1)
    )
    
    # Test vector extraction
    vec = extract_params_vector(hazard_params)
    @test length(vec) == 4
    @test vec[1] ≈ log(1.5)
    @test vec[2] ≈ log(0.2)
    @test vec[3] ≈ 0.3
    @test vec[4] ≈ 0.1
    
    # Test natural scale
    nat = extract_natural_vector(hazard_params)
    @test nat[1] ≈ 1.5
    @test nat[2] ≈ 0.2
    @test nat[3] ≈ 0.3
end
```

---

## Phase 2: Update Hazard Functions (Week 2)

### Estimated Effort: 15-20 hours

### 2.1 Update Linear Predictor Builder (2-3 hours)

**Goal**: Generate code that accesses covariate coefficients by name.

**File**: `src/hazards.jl` lines 65-85

**New function**:
```julia
function _build_linear_pred_expr_named(covar_names::Vector{Symbol})
    """
    Build expression for linear predictor with named covariate access.
    
    For covar_names = [:age, :sex], generates:
        covars[1] * pars.covariates.age + covars[2] * pars.covariates.sex
    
    Note: Covariate VALUES (covars[i]) are positional from data matrix.
          Covariate COEFFICIENTS are accessed by name from pars.covariates.
    """
    if isempty(covar_names)
        return :(0.0)
    end
    
    terms = [:(covars[$i] * pars.covariates.$(covar_names[i])) 
             for i in eachindex(covar_names)]
    
    if length(terms) == 1
        return terms[1]
    else
        return Expr(:call, :+, terms...)
    end
end
```

**Keep old function for backward compatibility during transition**:
```julia
# Rename current function
const _build_linear_pred_expr_positional = _build_linear_pred_expr
```

### 2.2 Update Weibull Hazard Generator (3-4 hours)

**File**: `src/hazards.jl` lines 395-450

**Current**:
```julia
function generate_weibull_hazard(parnames::Vector{Symbol}, linpred_effect::Symbol)
    linear_pred_expr = _build_linear_pred_expr(parnames, 3)
    
    if linpred_effect == :ph
        hazard_fn = @RuntimeGeneratedFunction(:(
            function(t, pars, covars)
                log_shape, log_scale = pars[1], pars[2]  # ❌ Positional
                shape = exp(log_shape)
                linear_pred = $linear_pred_expr
                
                log_haz = log_scale + log_shape + linear_pred
                if shape != 1.0
                    log_haz += (shape - 1) * log(t)
                end
                return exp(log_haz)
            end
        ))
        # ... cumhaz similar ...
    end
end
```

**New**:
```julia
function generate_weibull_hazard(parnames::Vector{Symbol}, linpred_effect::Symbol)
    # Weibull has 2 baseline parameters
    npar_baseline = 2
    
    # Build linear predictor for covariates (if any)
    linear_pred_expr = if length(parnames) > npar_baseline
        covar_names = parnames[(npar_baseline+1):end]
        _build_linear_pred_expr_named(covar_names)
    else
        :(0.0)
    end
    
    if linpred_effect == :ph
        hazard_fn = @RuntimeGeneratedFunction(:(
            function(t, pars, covars)
                # Access baseline parameters by name
                log_shape = pars.baseline.shape  # ✅ Named!
                log_scale = pars.baseline.scale  # ✅ Named!
                shape = exp(log_shape)
                
                # Linear predictor accesses covariates by name
                linear_pred = $linear_pred_expr
                
                log_haz = log_scale + log_shape + linear_pred
                if shape != 1.0
                    log_haz += (shape - 1) * log(t)
                end
                return exp(log_haz)
            end
        ))
        
        cumhaz_fn = @RuntimeGeneratedFunction(:(
            function(lb, ub, pars, covars)
                log_shape = pars.baseline.shape
                log_scale = pars.baseline.scale
                shape = exp(log_shape)
                scale = exp(log_scale)
                linear_pred = $linear_pred_expr
                return scale * exp(linear_pred) * (ub^shape - lb^shape)
            end
        ))
    elseif linpred_effect == :aft
        # Similar updates for AFT ...
    end
    
    return hazard_fn, cumhaz_fn
end
```

**Key points**:
- Access: `pars.baseline.shape` instead of `pars[1]`
- Covariate linear predictor: `pars.covariates.age` instead of `pars[3]`
- Data values still positional: `covars[1]` (fixed by formula)

### 2.3 Update Gompertz Hazard Generator (3-4 hours)

**File**: `src/hazards.jl` lines 455-515

**Same pattern as Weibull**:
- 2 baseline parameters: `:shape`, `:scale`
- Access by name: `pars.baseline.shape`, `pars.baseline.scale`
- Covariate linear predictor uses `_build_linear_pred_expr_named`

### 2.4 Update Exponential Hazard Generator (2-3 hours)

**File**: `src/hazards.jl` lines 348-380

**Changes**:
- 1 baseline parameter: `:Intercept` or `:rate`
- Access: `pars.baseline.Intercept`
- Simpler than Weibull/Gompertz

**Example**:
```julia
function generate_exponential_hazard(parnames::Vector{Symbol}, linpred_effect::Symbol)
    npar_baseline = 1
    
    linear_pred_expr = if length(parnames) > npar_baseline
        covar_names = parnames[(npar_baseline+1):end]
        _build_linear_pred_expr_named(covar_names)
    else
        :(0.0)
    end
    
    if linpred_effect == :ph
        hazard_fn = @RuntimeGeneratedFunction(:(
            function(t, pars, covars)
                log_rate = pars.baseline.Intercept  # ✅ Named
                linear_pred = $linear_pred_expr
                return exp(log_rate + linear_pred)
            end
        ))
        # ... cumhaz ...
    end
end
```

### 2.5 Update Spline Hazard Generators (4-5 hours)

**File**: `src/smooths.jl`

**Challenge**: Spline coefficients are numerous and basis-dependent.

**Approach**: Use generic names `sp1`, `sp2`, ..., `spn` for spline coefficients:
```julia
# For spline with 5 coefficients:
baseline_names = [:sp1, :sp2, :sp3, :sp4, :sp5]
```

**Access in hazard function**:
```julia
# Extract all spline coefficients as vector (since they're numerous)
spline_coefs = collect(values(pars.baseline))
# Then use as before: spline_coefs[1], spline_coefs[2], ...
```

**Alternative**: Keep spline coefficients as vector in a special field:
```julia
baseline = (spline_coefs = [...],)  # Single vector field
```

Then access: `pars.baseline.spline_coefs[1]`, `pars.baseline.spline_coefs[2]`, ...

**Decision needed**: Discuss with user which approach is clearer.

---

## Phase 3: Update Hazard Callers (Week 3)

### Estimated Effort: 10-12 hours

### 3.1 Update Likelihood Functions (5-6 hours)

**Goal**: Pass NamedTuple parameters (by hazard name) to hazard functions.

**File**: `src/likelihoods.jl`

**Key Change**: Likelihood code uses hazard **indices**, but `unflatten` returns **names**.

**Solution**: Map index → name

**Add to model build**:
```julia
# In multistatemodel function (src/modelgeneration.jl)
hazard_names = [keys(hazkeys)[findfirst(==(i), values(hazkeys))] for i in 1:length(hazards)]
# Store in model for easy lookup
```

**Update `_compute_path_loglik_fused`** (lines 1803-1920):
```julia
function _compute_path_loglik_fused(
    path::SamplePath, 
    pars,  # Now: NamedTuple by hazard name from unflatten
    hazards::Vector{<:_Hazard},
    totalhazards::Vector{<:_TotalHazard},
    hazard_names::Vector{Symbol},  # NEW: index → name mapping
    tmat::Matrix{Int64},
    subj_cache::SubjectCovarCache, 
    covar_names_per_hazard::Vector{Vector{Symbol}},
    tt_context,
    ::Type{T}
) where T
    # ...
    for h in tothaz.components
        hazard = hazards[h]
        hazname = hazard_names[h]  # NEW: get hazard name
        hazard_pars = pars[hazname]  # Access by name
        
        # Extract covariates
        covars = extract_covariates_lightweight(subj_cache, 1, covar_names_per_hazard[h])
        
        # Call hazard function
        cumhaz = eval_cumhaz(hazard, lb, ub, hazard_pars, covars; ...)
        ll -= cumhaz
    end
    # ...
end
```

**Update all callers to pass `hazard_names`**:
- `loglik_exact` and other top-level functions need to extract hazard names from model

### 3.2 Update Simulation Functions (3-4 hours)

**File**: `src/simulation.jl` lines 593-650

**Similar changes**:
- Pass `hazard_names` vector
- Access parameters by name: `pars[hazname]`
- Hazard functions receive `(baseline=..., covariates=...)` structure

### 3.3 Update MCEM Sampling (2-3 hours)

**File**: `src/sampling.jl`

**Remove `nest_params` calls**:
```julia
# OLD:
pars = nest_params(flat_params, model.parameters)  # Returns tuple of vectors

# NEW:
pars = model.parameters.unflatten(flat_params)  # Returns NamedTuple by name
```

**Update hazard evaluations** to access by name instead of index.

---

## Phase 4: Validated Parameter Setting (Week 4)

### Estimated Effort: 10-12 hours

### 4.1 Enhanced `set_parameters!` with Validation (6-8 hours)

**File**: `src/helpers.jl` lines 126-159

**Implementation**: See detailed code in main plan document section "Phase 3.1".

**Key features**:
- Accept nested NamedTuples with full validation
- Support mixed format (named baseline, vector covariates)
- Legacy vector support with warnings
- Reorder parameters if user provides different order
- Comprehensive error messages

**Example usage**:
```julia
# Best: Fully named
set_parameters!(model, (
    h12 = (
        baseline = (shape = log(1.5), scale = log(0.2)),
        covariates = (age = 0.3, sex = 0.1)
    ),
    h23 = (baseline = (shape = log(2.0)),)
))

# Also good: Named baseline, vector covariates (if formula order clear)
set_parameters!(model, (
    h12 = (baseline = (shape = log(1.5), scale = log(0.2)), covariates = [0.3, 0.1]),
))

# Legacy: Vectors (with warnings)
set_parameters!(model, (
    h12 = (baseline = [log(1.5), log(0.2)], covariates = [0.3, 0.1]),
))
```

### 4.2 Update Parameter Initialization (3-4 hours)

**File**: `src/initialization.jl`

**Update `init_par` methods** to return NamedTuple structure:
```julia
function init_par(hazard::_WeibullHazard, ...)
    baseline = (shape = log(init_shape), scale = log(init_scale))
    
    if !isempty(hazard.parnames[(hazard.npar_baseline+1):end])
        covar_names = hazard.parnames[(hazard.npar_baseline+1):end]
        covar_coefs = # ... compute initial coefficients ...
        covariates = NamedTuple{Tuple(covar_names)}(covar_coefs)
        return (baseline = baseline, covariates = covariates)
    else
        return (baseline = baseline,)
    end
end
```

---

## Phase 5: Comprehensive Testing (Week 5)

### Estimated Effort: 12-15 hours

### 5.1 Unit Tests for New Functionality (4-5 hours)

**File**: `test/test_helpers.jl`

**Test suites**:

1. **NamedTuple Parameter Construction**:
   - `build_hazard_params` with various combinations
   - Baseline only
   - Baseline + covariates
   - Different hazard families

2. **Parameter Extraction**:
   - `extract_params_vector` from NamedTuple structure
   - `extract_natural_vector` with exp() transform
   - Edge cases (no covariates, many covariates)

3. **ParameterHandling.jl Integration**:
   - Flatten/unflatten with nested NamedTuples
   - AD compatibility with ForwardDiff
   - Reconstruction accuracy

### 5.2 Validation Tests (3-4 hours)

**File**: `test/test_parameter_validation.jl` (new)

**Test suites**:

1. **Correct Named Parameter Assignment**:
   - Fully named structure
   - Mixed named/vector
   - Multiple hazards
   - With and without covariates

2. **Error Detection**:
   - Wrong parameter names
   - Missing parameters
   - Extra parameters
   - Wrong hazard names

3. **Parameter Reordering**:
   - User provides `(scale=..., shape=...)`  → correctly reordered to `(shape=..., scale=...)`
   - Covariates in different order

4. **Legacy Compatibility**:
   - Vector parameters still work
   - Warnings issued when validation=true
   - No warnings when validation=false

### 5.3 Integration Tests (5-6 hours)

**Update existing test files**:

1. **`test/test_modelgeneration.jl`**:
   - Verify hazard objects have `parnames` field
   - Check parameter structure is correct

2. **`test/test_exact_data_fitting.jl`**:
   - Rewrite parameter setting to use named format
   - Verify fitting still works
   - Check parameter extraction

3. **`test/longtest_mcem.jl`**:
   - Update Weibull/Gompertz tests to use named parameters
   - Verify no [scale, shape] vs [shape, scale] errors possible
   - Test with covariates

4. **`test/test_simulation.jl`**:
   - Update parameter setting
   - Verify simulation works with new structure

**Regression testing**:
- Run ALL existing tests
- Verify no performance degradation (<5% overhead acceptable)
- Check memory usage

---

## Phase 6: Documentation (Week 6)

### Estimated Effort: 8-10 hours

### 6.1 Update API Documentation (3-4 hours)

**File**: `docs/src/api.md`

**Document**:
1. **`set_parameters!`** with all usage patterns
2. **Parameter structure** showing nested NamedTuple format
3. **Migration guide** from old positional style

**Example documentation**:
````julia
"""
    set_parameters!(model, params; validate=true)

Set model parameters using named or positional format.

# Named Format (Recommended)
```julia
set_parameters!(model, (
    h12 = (baseline = (shape = 1.5, scale = 0.2), covariates = (age = 0.3,)),
    h23 = (baseline = (shape = 2.0),)
))
```

# Parameter Order by Hazard Family
- **Weibull/Gompertz**: `(shape, scale, covariates...)`
- **Exponential**: `(Intercept, covariates...)`
- **Splines**: `(sp1, sp2, ..., spn, covariates...)`

# Validation
When `validate=true` (default):
- Checks all required parameters present
- Warns about positional vectors
- Reorders if necessary

Set `validate=false` for performance-critical code.
"""
````

### 6.2 Add User Guide Section (2-3 hours)

**File**: `docs/src/parameters.md` (new)

**Content**:
1. **Why Named Parameters**: Explain fragility of positional
2. **How to Use**: Step-by-step examples
3. **Common Patterns**: Different hazard families
4. **Troubleshooting**: Common errors and fixes

### 6.3 Update Examples (2-3 hours)

**Update all example scripts**:
- Use named parameter format
- Show validation benefits
- Demonstrate error messages

---

## Success Criteria

### Must Have (Phases 1-3)
- ✅ Parameters stored as NamedTuple internally
- ✅ Hazard functions use named access (`pars.baseline.shape`)
- ✅ All existing tests pass
- ✅ No performance degradation (benchmark shows <5% overhead)
- ✅ AD compatibility maintained (ForwardDiff works)

### Should Have (Phase 4)
- ✅ `set_parameters!` validates named input
- ✅ Helpful error messages for common mistakes
- ✅ Backward compatibility with vector input (with warnings)
- ✅ Documentation updated

### Nice to Have (Phase 5-6)
- ✅ All tests use named parameter style
- ✅ User guide with examples
- ✅ Migration guide for existing code
- ✅ Comprehensive test coverage (>95%)

---

## Risk Assessment

### Low Risk
- **Phase 1**: Additive changes, backward compatible
- **ParameterHandling.jl compatibility**: Already supports nested NamedTuples
- **Testing infrastructure**: Can verify correctness at each step

### Medium Risk
- **Hazard function updates**: RuntimeGeneratedFunctions need careful handling
- **Performance**: Named tuple field access may be slower than array indexing (needs benchmarking)
- **Covariate handling**: Ensuring formula order preserved correctly

### Mitigation Strategies
1. **Incremental implementation**: Each phase independently testable
2. **Benchmark early**: Phase 1 includes performance verification
3. **Keep legacy paths**: Vector input supported during transition
4. **Comprehensive testing**: 95%+ coverage before merging

---

## Timeline Summary

| Phase | Description | Effort | Cumulative |
|-------|-------------|--------|------------|
| 1 | Foundation - NamedTuple baseline | 12-15 hrs | 12-15 hrs |
| 2 | Update hazard functions | 15-20 hrs | 27-35 hrs |
| 3 | Update hazard callers | 10-12 hrs | 37-47 hrs |
| 4 | Validated parameter setting | 10-12 hrs | 47-59 hrs |
| 5 | Comprehensive testing | 12-15 hrs | 59-74 hrs |
| 6 | Documentation | 8-10 hrs | 67-84 hrs |

**Total Estimated Effort**: 67-84 hours over 6 weeks

**Can be staged incrementally** - each phase delivers value independently.

---

## Next Steps

1. ✅ **Review this plan** - verify approach and timeline
2. ⏳ **Approve Phase 1** - begin with foundation
3. ⏳ **Create feature branch** - `feature/named-parameters`
4. ⏳ **Implement Phase 1** - NamedTuple baseline parameters
5. ⏳ **Benchmark** - verify performance acceptable
6. ⏳ **Proceed to Phase 2** - update hazard functions

**Ready to begin implementation upon approval.**
