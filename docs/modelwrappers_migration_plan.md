# Migration Plan: ParameterHandling.jl → ModelWrappers.jl

**Date**: December 7, 2024  
**Status**: IMPLEMENTATION READY  
**Estimated Time**: 4-6 hours

---

## Executive Summary

Replace ParameterHandling.jl with ModelWrappers.jl to gain:
1. **Automatic AD type handling** - eliminates `safe_unflatten()` complexity
2. **Zero-allocation unflatten** - better performance
3. **Built-in constraint support** - for future enhancements
4. **Cleaner architecture** - `flattenAD`/`unflattenAD` express intent clearly

---

## Migration Strategy

### Phase 1: Setup (30 minutes)
1. Add ModelWrappers.jl dependency
2. Keep ParameterHandling.jl temporarily (parallel operation)
3. Create wrapper functions for gradual migration

### Phase 2: Core Infrastructure (2-3 hours)
1. Replace parameter storage structure
2. Update `rebuild_parameters()` to use `ReConstructor`
3. Replace `safe_unflatten()` with `unflattenAD()`
4. Update likelihood functions

### Phase 3: Testing & Validation (1-2 hours)
1. Run full test suite
2. Performance benchmarks
3. Verify AD compatibility

### Phase 4: Cleanup (30 minutes)
1. Remove ParameterHandling.jl dependency
2. Remove obsolete `safe_unflatten()` function
3. Update documentation

---

## Detailed Implementation

### Step 1: Add Dependency

**File: `Project.toml`**

Add to `[deps]` section:
```toml
ModelWrappers = "e30172f5-a6a5-5a46-863b-614d45cd2bd4"
```

**Action**: Run `julia --project=. -e 'using Pkg; Pkg.add("ModelWrappers")'`

---

### Step 2: Update Parameter Storage Structure

**Current Structure (ParameterHandling.jl):**
```julia
model.parameters = (
    flat = Vector{Float64},           # Flat parameter vector
    nested = NamedTuple,              # Nested by hazard name
    natural = NamedTuple,             # Natural scale per hazard
    unflatten = Function              # Reconstruction function (MONOMORPHIC!)
)
```

**New Structure (ModelWrappers.jl):**
```julia
model.parameters = (
    flat = Vector{Float64},           # Flat parameter vector
    nested = NamedTuple,              # Nested by hazard name
    natural = NamedTuple,             # Natural scale per hazard
    reconstructor = ReConstructor     # Handles both Float64 and Dual (POLYMORPHIC!)
)
```

**Key Change**: Replace monomorphic `unflatten` function with polymorphic `ReConstructor` object.

---

### Step 3: Update `rebuild_parameters()` Function

**File: `src/helpers.jl`** (Lines ~47-80)

**Current Implementation:**
```julia
function rebuild_parameters(new_param_vectors::Vector{Vector{Float64}}, model::MultistateProcess)
    params_nested_pairs = [
        hazname => build_hazard_params(new_param_vectors[idx], model.hazards[idx].parnames, 
                                      model.hazards[idx].npar_baseline, model.hazards[idx].npar_total)
        for (hazname, idx) in sort(collect(model.hazkeys), by = x -> x[2])
    ]
    
    params_nested = NamedTuple(params_nested_pairs)
    params_flat, unflatten_fn = ParameterHandling.flatten(params_nested)
    
    # Get natural scale parameters
    params_natural_pairs = [
        hazname => extract_natural_vector(params_nested[hazname])
        for (hazname, idx) in sort(collect(model.hazkeys), by = x -> x[2])
    ]
    params_natural = NamedTuple(params_natural_pairs)
    
    return (
        flat = params_flat,
        nested = params_nested,
        natural = params_natural,
        unflatten = unflatten_fn
    )
end
```

**New Implementation:**
```julia
function rebuild_parameters(new_param_vectors::Vector{Vector{Float64}}, model::MultistateProcess)
    params_nested_pairs = [
        hazname => build_hazard_params(new_param_vectors[idx], model.hazards[idx].parnames, 
                                      model.hazards[idx].npar_baseline, model.hazards[idx].npar_total)
        for (hazname, idx) in sort(collect(model.hazkeys), by = x -> x[2])
    ]
    
    params_nested = NamedTuple(params_nested_pairs)
    
    # Create ReConstructor - handles both Float64 and Dual types automatically
    reconstructor = ModelWrappers.ReConstructor(params_nested)
    params_flat = ModelWrappers.flatten(reconstructor, params_nested)
    
    # Get natural scale parameters
    params_natural_pairs = [
        hazname => extract_natural_vector(params_nested[hazname])
        for (hazname, idx) in sort(collect(model.hazkeys), by = x -> x[2])
    ]
    params_natural = NamedTuple(params_natural_pairs)
    
    return (
        flat = params_flat,
        nested = params_nested,
        natural = params_natural,
        reconstructor = reconstructor  # NEW: Store reconstructor instead of unflatten
    )
end
```

---

### Step 4: Replace `safe_unflatten()` with `unflattenAD()`

**File: `src/helpers.jl`** (Lines ~1-45)

**DELETE the entire `safe_unflatten()` function** (45 lines of complex type-dispatch logic)

**REPLACE with simple wrapper:**
```julia
"""
    unflatten_parameters(flat_params, model::MultistateProcess)

Unflatten parameter vector with automatic AD type handling via ModelWrappers.jl.

Automatically detects whether `flat_params` contains Float64 or ForwardDiff.Dual
and returns appropriately typed NamedTuple.

# Arguments
- `flat_params`: Flat parameter vector (Float64 or Dual)
- `model`: MultistateProcess model containing parameter structure

# Returns
NamedTuple of nested parameters matching model structure
"""
function unflatten_parameters(flat_params::AbstractVector, model::MultistateProcess)
    return ModelWrappers.unflattenAD(model.parameters.reconstructor, flat_params)
end
```

**Benefits:**
- 45 lines → 3 lines of actual logic
- No manual type dispatch needed
- Automatic Dual type handling
- Zero allocations for unflatten

---

### Step 5: Update Likelihood Functions

**File: `src/likelihoods.jl`**

**Function: `prepare_parameters()` (Line ~42)**

Current:
```julia
function prepare_parameters(p::AbstractVector{<:Real}, model::MultistateProcess)
    return safe_unflatten(p, model)
end
```

New:
```julia
function prepare_parameters(p::AbstractVector{<:Real}, model::MultistateProcess)
    return unflatten_parameters(p, model)
end
```

**All other likelihood functions**: No changes needed! They already receive NamedTuple from `prepare_parameters()`.

---

### Step 6: Update Module Imports

**File: `src/MultistateModels.jl`**

Current imports:
```julia
using ParameterHandling
```

New imports:
```julia
using ModelWrappers
```

---

### Step 7: Update Tests

**File: `test/test_helpers.jl`**

**Test: "ParameterHandling.jl with nested NamedTuples"** (Lines ~228-260)

Rename and update:
```julia
@testset "ModelWrappers.jl with nested NamedTuples" begin
    using MultistateModels: build_hazard_params
    
    @testset "Flatten and unflatten with named fields" begin
        # Build parameter structure with named NamedTuples
        params = (
            h12 = build_hazard_params(
                [log(1.5), log(0.2), 0.3, 0.1],
                [:h12_shape, :h12_scale, :h12_age, :h12_sex],
                2, 4
            ),
            h23 = build_hazard_params(
                [log(0.8)],
                [:h23_intercept],
                1, 1
            )
        )
        
        # Create reconstructor and flatten
        reconstructor = ModelWrappers.ReConstructor(params)
        flat = ModelWrappers.flatten(reconstructor, params)
        
        # Test unflatten
        reconstructed = ModelWrappers.unflatten(reconstructor, flat)
        
        # Verify structure preserved
        @test reconstructed.h12.baseline.h12_shape ≈ log(1.5)
        @test reconstructed.h12.baseline.h12_scale ≈ log(0.2)
        @test reconstructed.h12.covariates.h12_age ≈ 0.3
        @test reconstructed.h23.baseline.h23_intercept ≈ log(0.8)
        
        # Test AD compatibility - unflattenAD preserves Dual types
        using ForwardDiff
        flat_dual = ForwardDiff.Dual.(flat, 1.0)
        reconstructed_dual = ModelWrappers.unflattenAD(reconstructor, flat_dual)
        @test reconstructed_dual.h12.baseline.h12_shape isa ForwardDiff.Dual
        
        # Test modification
        modified_flat = flat .+ 0.1
        modified = ModelWrappers.unflatten(reconstructor, modified_flat)
        @test modified.h12.baseline.h12_shape ≈ log(1.5) + 0.1
    end
    
    @testset "Zero allocation unflatten" begin
        using BenchmarkTools
        params = (h12 = build_hazard_params([1.0, 2.0], [:h12_a, :h12_b], 2, 2),)
        reconstructor = ModelWrappers.ReConstructor(params)
        flat = ModelWrappers.flatten(reconstructor, params)
        
        # Unflatten should have zero allocations
        allocs = @allocated ModelWrappers.unflatten(reconstructor, flat)
        @test allocs == 0
    end
end
```

---

## Testing Strategy

### Test 1: Basic Functionality
```julia
# Build simple model
model = multistatemodel(...)

# Verify parameters structure
@test hasfield(typeof(model.parameters), :reconstructor)
@test model.parameters.reconstructor isa ModelWrappers.ReConstructor

# Test unflatten
flat = model.parameters.flat
nested = unflatten_parameters(flat, model)
@test nested == model.parameters.nested
```

### Test 2: AD Compatibility
```julia
# Test ForwardDiff gradient computation
function test_loglik(params, model)
    return loglik_exact(params, model.data, model)
end

flat = model.parameters.flat
grad = ForwardDiff.gradient(p -> test_loglik(p, model), flat)
@test all(isfinite.(grad))
```

### Test 3: Performance Benchmarks
```julia
using BenchmarkTools

# Unflatten performance
flat = model.parameters.flat
@btime unflatten_parameters($flat, $model)  # Target: <500ns, 0 allocations

# Full likelihood evaluation
@btime loglik_exact($flat, $model.data, $model)  # Should match current performance
```

---

## Migration Checklist

### Pre-Migration
- [ ] Commit current working state
- [ ] Create migration branch: `git checkout -b modelwrappers-migration`
- [ ] Run full test suite to establish baseline

### Implementation
- [ ] Step 1: Add ModelWrappers.jl dependency
- [ ] Step 2: Update parameter storage structure in model building
- [ ] Step 3: Update `rebuild_parameters()` function
- [ ] Step 4: Replace `safe_unflatten()` with `unflatten_parameters()`
- [ ] Step 5: Update `prepare_parameters()` in likelihoods
- [ ] Step 6: Update module imports
- [ ] Step 7: Update tests

### Validation
- [ ] Run full test suite
- [ ] All previous tests pass
- [ ] New AD tests pass
- [ ] Performance benchmarks meet targets
- [ ] Zero allocation for unflatten confirmed

### Cleanup
- [ ] Remove ParameterHandling.jl from Project.toml
- [ ] Remove `safe_unflatten()` function
- [ ] Update documentation
- [ ] Update CHANGELOG

### Finalization
- [ ] Merge to infrastructure_changes branch
- [ ] Create PR with detailed description

---

## Rollback Plan

If issues arise:
1. `git checkout infrastructure_changes` (before migration)
2. Cherry-pick any other work done in parallel
3. Document issues encountered
4. Reassess migration strategy

---

## Expected Outcomes

### Lines of Code
- **Removed**: ~45 lines (`safe_unflatten` complexity)
- **Added**: ~15 lines (simple wrappers + tests)
- **Net Change**: -30 lines

### Performance
- **Unflatten**: 0 allocations (current: some allocations for Dual path)
- **Likelihood**: No change (already optimized)

### Maintainability
- **Complexity**: Much lower (no manual type dispatch)
- **Intent**: Clearer (`unflattenAD` vs `safe_unflatten`)
- **Future**: Easy to add parameter constraints

---

## Timeline

| Phase | Duration | Tasks |
|-------|----------|-------|
| Setup | 30 min | Add dependency, create branch |
| Core Changes | 2-3 hours | Update functions, replace safe_unflatten |
| Testing | 1-2 hours | Run tests, validate AD, benchmarks |
| Cleanup | 30 min | Remove old code, update docs |
| **Total** | **4-6 hours** | |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Breaking existing tests | Low | Medium | Comprehensive test suite runs before merge |
| Performance regression | Very Low | High | Benchmarks before/after |
| AD compatibility issues | Very Low | High | Specific AD tests added |
| Unforeseen edge cases | Low | Medium | Gradual rollout, easy rollback |

---

## Next Steps

1. Review this plan
2. Create migration branch
3. Begin Step 1 (add dependency)
4. Implement changes in order
5. Validate thoroughly
6. Merge when all tests pass

---

## Notes

- ModelWrappers.jl is actively maintained (42 releases)
- Designed specifically for AD-compatible parameter handling
- Used in other Julia packages successfully
- Better aligned with Julia ecosystem patterns
- Provides foundation for future enhancements (constraints)
