# Phase 3 Implementation Plan - ParameterHandling.jl Integration

**Status**: Planning  
**Prerequisites**: Phase 2 Complete ✅  
**Deferred**: Spline hazards (will implement after Phase 3)

---

## Goals

**Primary Objective**: Fully integrate ParameterHandling.jl for parameter transformations and optimization

**Key Benefits**:
1. **Automatic transformations**: Parameters on natural scale (positive constraints via log transform)
2. **Optimizer-friendly**: Flat vector representation for optimization algorithms
3. **Type-safe**: Nested NamedTuple structure for clarity
4. **Immutable→Mutable**: Fix the NamedTuple limitation from Phase 2

---

## Current State Assessment

### ✅ What's Already Working
- `parameters_ph` field exists in model structs (NamedTuple)
- `update_parameters_ph!()` helper function creates new structures
- Structure format: `(flat, transformed, natural, unflatten)`

### 🚨 **CRITICAL ISSUE DISCOVERED**
**Parameterization Mismatch Between Phase 2 and Legacy Code**:
- **Old system**: `model.parameters` stores **LOG SCALE** (e.g., `log(rate)`)
- **New Phase 2 hazard functions**: Expect **NATURAL SCALE** (e.g., `rate`)
- **Impact**: Phase 2 backward compatibility layer doesn't transform correctly
- **Status**: Must fix before Phase 3!

### ⚠️ Current Limitations
1. **NamedTuple Immutability**: `model.parameters_ph` cannot be updated in-place
2. **Manual Synchronization**: Users must manually handle `update_parameters_ph!()` returns
3. **Incomplete Integration**: Not used in optimization/fitting functions yet
4. **VectorOfVectors Legacy**: Still primary parameter storage (for compatibility)
5. **Parameter Scale Mismatch**: Phase 2 functions incompatible with legacy storage (see above)

---

## URGENT: Phase 2.5 - Fix Parameter Scale Mismatch ✅ COMPLETE

**Status**: ✅ **COMPLETE** - All hazard generators updated and tested

### The Problem

Currently we have incompatible parameter scales:
- `model.parameters` (VectorOfVectors): **LOG SCALE**
- New hazard functions: Expect **NATURAL SCALE** (BEFORE fix)
- Backward compatibility layer: Doesn't transform correctly

### The Solution Implemented

**Option B: Change New Hazard Functions to Expect Log Scale** ✅
- ✅ Rewrote `generate_*_hazard()` functions to expect log scale
- ✅ Added `exp()` calls in hazard calculations
- ✅ Maintains backward compatibility with `model.parameters`
- ✅ All tests pass (see test/test_phase25_log_scale.jl)

**Why Option B**:
- Preserves backward compatibility with existing `model.parameters` storage
- Minimal changes to existing codebase
- ParameterHandling.jl will handle transformations in Phase 3
- All existing tests work without modification

### Implementation Details (Phase 2.5)

**Updated Functions**:

1. **`generate_exponential_hazard()`**:
   - Parameters: `[log_baseline]` or `[log_baseline, β1, β2, ...]`
   - Formula: `h(t) = exp(log_baseline + β'X)`
   - File: src/hazards.jl lines 13-49

2. **`generate_weibull_hazard()`**:
   - Parameters: `[log_shape, log_scale]` or `[log_shape, log_scale, β1, β2, ...]`
   - Formula: `h(t) = exp(log_shape + expm1(log_shape)*log(t) + log_scale + β'X)`
   - Cumulative: `H(lb,ub) = scale * exp(β'X) * (ub^shape - lb^shape)`
   - File: src/hazards.jl lines 51-108

3. **`generate_gompertz_hazard()`**:
   - Parameters: `[log_shape, log_scale]` or `[log_shape, log_scale, β1, β2, ...]`
   - Formula: `h(t) = exp(log_scale + log_shape + shape*t + β'X)`
   - Cumulative: `H(lb,ub) = scale * exp(β'X) * (exp(shape*ub) - exp(shape*lb))`
   - File: src/hazards.jl lines 110-167

**Testing**:
- ✅ Created comprehensive test suite: test/test_phase25_log_scale.jl
- ✅ All hazard types tested with and without covariates
- ✅ Verified correct log scale parameter handling
- ✅ Backward compatibility layer works correctly

**Note**: Full integration with test/test_hazards.jl pending Symbolics.jl dependency fix (unrelated issue).

---

## Phase 3 Architecture

### 3.1: Make Model Structs Mutable ✨

**Current**:
```julia
struct MultistateModel <: MultistateProcess
    data::DataFrame
    parameters::VectorOfVectors  # Legacy
    parameters_ph::NamedTuple    # NEW but immutable!
    hazards::Vector{_Hazard}
    # ... other fields
end
```

**Proposed**:
```julia
mutable struct MultistateModel <: MultistateProcess
    data::DataFrame
    parameters::VectorOfVectors  # Keep for backward compatibility
    parameters_ph::NamedTuple    # Can now be reassigned!
    hazards::Vector{_Hazard}
    # ... other fields (unchanged)
end
```

**Why Mutable**:
- Allows `model.parameters_ph = new_value` assignments
- Enables proper parameter updates in optimization
- Minimal breaking changes (just adds `mutable` keyword)
- Slightly higher memory usage (acceptable tradeoff)

**Alternative Considered**: `Ref{NamedTuple}` wrapper
- Rejected: More complex API, requires `model.parameters_ph[]` syntax
- Mutable struct is cleaner and more idiomatic

---

### 3.2: Update set_parameters!() Functions

**Current Behavior**:
```julia
set_parameters!(model, h, new_pars)
# Updates model.parameters[h]
# Does NOT update model.parameters_ph (limitation noted in docs)
```

**New Behavior**:
```julia
set_parameters!(model, h, new_pars)
# 1. Updates model.parameters[h]
# 2. Rebuilds and updates model.parameters_ph automatically
# 3. Maintains synchronization between both representations
```

**Implementation**:
```julia
function set_parameters!(model::MultistateProcess, h::Int64, newvalues::Vector{Float64})
    # Existing: Update VectorOfVectors
    copyto!(model.parameters[h], newvalues)
    
    # NEW: Rebuild and update ParameterHandling structure
    model.parameters_ph = update_parameters_ph!(model)
    
    # Existing: Handle spline special case (if needed)
    if isa(model.hazards[h], _SplineHazard)
        remake_splines!(model.hazards[h], newvalues)
        set_riskperiod!(model.hazards[h])
    end
    
    return nothing
end
```

---

### 3.3: Implement get_parameters() Functions

**Purpose**: Extract parameters in different representations

```julia
# Get flat vector for optimization
function get_parameters_flat(model::MultistateProcess)
    return model.parameters_ph.flat
end

# Get transformed parameters (log scale)
function get_parameters_transformed(model::MultistateProcess)
    return model.parameters_ph.transformed
end

# Get natural parameters (original scale)
function get_parameters_natural(model::MultistateProcess)
    return model.parameters_ph.natural
end

# Get unflatten function
function get_unflatten_fn(model::MultistateProcess)
    return model.parameters_ph.unflatten
end

# Get specific hazard parameters
function get_parameters(model::MultistateProcess, h::Int64; scale=:natural)
    if scale == :natural
        hazname = findfirst(==(h), model.hazkeys)
        return model.parameters_ph.natural[hazname]
    elseif scale == :transformed
        hazname = findfirst(==(h), model.hazkeys)
        return model.parameters_ph.transformed[hazname]
    elseif scale == :flat
        # Extract from flat vector
        # Need to track indices...
        error("Not yet implemented - use get_parameters_flat() instead")
    else
        error("scale must be :natural, :transformed, or :flat")
    end
end
```

---

### 3.4: Update Model Fitting Functions

**Files to Update**:
- `src/modelfitting.jl` - Main fitting routines
- `src/mcem.jl` - MCEM algorithm
- Any optimization wrappers

**Changes Needed**:

1. **Extract parameters for optimization**:
```julia
# OLD
initial_params = vcat(model.parameters...)  # Manual flattening

# NEW
initial_params = get_parameters_flat(model)
```

2. **Update parameters from optimizer**:
```julia
# OLD
function update_from_optimizer!(model, flat_params)
    # Manual splitting and assignment
    idx = 1
    for h in eachindex(model.parameters)
        n = length(model.parameters[h])
        model.parameters[h] .= flat_params[idx:idx+n-1]
        idx += n
    end
end

# NEW
function update_from_optimizer!(model, flat_params)
    # Use unflatten to get structured params
    params_transformed = model.parameters_ph.unflatten(flat_params)
    
    # Convert to natural scale and update
    params_natural = ParameterHandling.value(params_transformed)
    
    # Update each hazard
    for (hazname, idx) in model.hazkeys
        set_parameters!(model, idx, params_natural[hazname])
    end
end
```

3. **Apply constraints automatically**:
```julia
# Optimization wrapper
function objective(flat_params, model)
    # Unflatten
    params_transformed = model.parameters_ph.unflatten(flat_params)
    
    # Transform to natural scale (positive constraint applied automatically!)
    params_natural = ParameterHandling.value(params_transformed)
    
    # Update model
    for (hazname, idx) in model.hazkeys
        set_parameters!(model, idx, params_natural[hazname])
    end
    
    # Compute likelihood
    return -loglikelihood(model)
end
```

---

### 3.5: Update Model Construction

**In build_hazards()**:

Ensure `parameters_ph` is properly initialized when model is created:

```julia
function build_hazards(...)
    # ... existing code creates parameters VectorOfVectors ...
    
    # NEW: Build ParameterHandling structure
    params_transformed_pairs = [
        hazname => ParameterHandling.positive(Vector{Float64}(parameters[idx]))
        for (hazname, idx) in sort(collect(hazkeys), by = x -> x[2])
    ]
    
    params_transformed = NamedTuple(params_transformed_pairs)
    params_flat, unflatten_fn = ParameterHandling.flatten(params_transformed)
    params_natural = ParameterHandling.value(params_transformed)
    
    parameters_ph = (
        flat = params_flat,
        transformed = params_transformed,
        natural = params_natural,
        unflatten = unflatten_fn
    )
    
    # Return both parameters and parameters_ph
    return _hazards, parameters, parameters_ph, hazkeys
end
```

**In multistatemodel()**:

```julia
function multistatemodel(...)
    # ... existing code ...
    
    _hazards, parameters, parameters_ph, hazkeys = build_hazards(...)
    
    # Create model with both representations
    model = MultistateModel(
        data = data,
        parameters = parameters,
        parameters_ph = parameters_ph,  # Now properly initialized
        hazards = _hazards,
        # ... other fields ...
    )
    
    return model
end
```

---

## Implementation Tasks

### Task 3.1: Make Structs Mutable
**File**: `src/common.jl`  
**Effort**: Low (5-10 min)  
**Risk**: Low

- [ ] Add `mutable` keyword to `MultistateModel`
- [ ] Add `mutable` keyword to `MultistateModelFitted`
- [ ] Add `mutable` keyword to `MultistateMarkovModel`
- [ ] Add `mutable` keyword to `MultistateMarkovModelCensored`
- [ ] Add `mutable` keyword to `MultistateSemiMarkovModel`
- [ ] Add `mutable` keyword to `MultistateSemiMarkovModelCensored`
- [ ] Test: Verify assignment works: `model.parameters_ph = new_value`

---

### Task 3.2: Update set_parameters!()
**File**: `src/helpers.jl`  
**Effort**: Medium (30-60 min)  
**Risk**: Medium (affects parameter updates)

- [ ] Modify `set_parameters!(model, h::Int64, newvalues)` to update `parameters_ph`
- [ ] Modify `set_parameters!(model, newvalues::VectorOfVectors)` to update `parameters_ph`
- [ ] Modify `set_parameters!(model, newvalues::Tuple)` to update `parameters_ph`
- [ ] Modify `set_parameters!(model, newvalues::NamedTuple)` to update both
- [ ] Update docstrings to reflect new behavior
- [ ] Remove "limitation" notes from docs
- [ ] Test: Verify both `parameters` and `parameters_ph` stay synchronized

---

### Task 3.3: Implement get_parameters() Functions
**File**: `src/helpers.jl`  
**Effort**: Low-Medium (20-30 min)  
**Risk**: Low

- [ ] Implement `get_parameters_flat(model)`
- [ ] Implement `get_parameters_transformed(model)`
- [ ] Implement `get_parameters_natural(model)`
- [ ] Implement `get_unflatten_fn(model)`
- [ ] Implement `get_parameters(model, h; scale=:natural)`
- [ ] Write docstrings with examples
- [ ] Test: Verify all functions return correct values

---

### Task 3.4: Update build_hazards()
**File**: `src/modelgeneration.jl`  
**Effort**: Medium (45-60 min)  
**Risk**: Medium (core construction)

- [ ] Modify `build_hazards(hazards...; data, surrogate)` to build `parameters_ph`
- [ ] Return `parameters_ph` as third output
- [ ] Update all callsites to handle new return value
- [ ] Verify initialization for all hazard families (exp, wei, gom)
- [ ] Test: Create models and verify `parameters_ph` is properly populated

---

### Task 3.5: Update Model Construction
**File**: `src/modelgeneration.jl`  
**Effort**: Medium (30-45 min)  
**Risk**: Medium

- [ ] Update `multistatemodel(hazards...; data=...)` to use `parameters_ph`
- [ ] Update `multistatemodel(; subjectdata=, transitionmatrix=, ...)` to use `parameters_ph`
- [ ] Update all model type constructors
- [ ] Test: Create various models and verify structure

---

### Task 3.6: Update Optimization Functions
**File**: `src/modelfitting.jl`, `src/mcem.jl`  
**Effort**: High (2-3 hours)  
**Risk**: High (core functionality)

- [ ] Identify all parameter extraction points
- [ ] Replace manual flattening with `get_parameters_flat()`
- [ ] Update parameter update logic to use `set_parameters!()` properly
- [ ] Leverage automatic transformations (positive constraints)
- [ ] Update objective function wrappers
- [ ] Test: Run optimization and verify convergence

---

### Task 3.7: Testing
**File**: `test/` directory  
**Effort**: Medium-High (1-2 hours)  
**Risk**: Low

- [ ] Create `test/test_parameterhandling.jl`
- [ ] Test parameter extraction (flat, transformed, natural)
- [ ] Test parameter setting and synchronization
- [ ] Test optimization with constraints
- [ ] Test model construction with ParameterHandling
- [ ] Run existing test suite to ensure no regressions
- [ ] Test edge cases (zero parameters, single parameter, etc.)

---

### Task 3.8: Documentation
**File**: Documentation, docstrings  
**Effort**: Medium (1 hour)  
**Risk**: Low

- [ ] Update function docstrings with ParameterHandling examples
- [ ] Add examples of constrained optimization
- [ ] Document the nested NamedTuple structure
- [ ] Create migration guide from Phase 2 to Phase 3
- [ ] Update PHASE3_PLAN.md to PHASE3_COMPLETE.md when done

---

## Testing Strategy

### Unit Tests
```julia
@testset "ParameterHandling Integration" begin
    # Test parameter extraction
    @testset "get_parameters functions" begin
        model = create_test_model()
        
        flat = get_parameters_flat(model)
        @test flat isa Vector{Float64}
        
        transformed = get_parameters_transformed(model)
        @test transformed isa NamedTuple
        
        natural = get_parameters_natural(model)
        @test natural isa NamedTuple
    end
    
    # Test parameter setting
    @testset "set_parameters! synchronization" begin
        model = create_test_model()
        
        # Set parameters
        new_pars = [0.5, -0.3]
        set_parameters!(model, 1, new_pars)
        
        # Check synchronization
        @test model.parameters[1] == new_pars
        @test all(model.parameters_ph.natural[:h12] .≈ exp.(new_pars))
    end
    
    # Test optimization
    @testset "Optimization with constraints" begin
        model = create_test_model()
        
        # Optimize (all parameters should stay positive)
        fit!(model)
        
        # Check constraints
        for (hazname, idx) in model.hazkeys
            natural_pars = model.parameters_ph.natural[hazname]
            @test all(natural_pars .> 0)  # Positive constraint
        end
    end
end
```

### Integration Tests
- Run existing model fitting tests
- Verify likelihood calculations unchanged
- Check parameter estimates are consistent
- Test with different hazard families

---

## Breaking Changes

### None Expected! 🎉

**Backward Compatibility Maintained**:
- `model.parameters` (VectorOfVectors) still works
- Existing `set_parameters!()` calls still work (just do more now)
- Old test suite should pass without changes
- New functionality is additive

**Migration Path**:
- No changes required for existing code
- Can gradually adopt new `get_parameters_*()` functions
- Optimization code will benefit from automatic constraints

---

## Performance Considerations

### Memory
- **Mutable structs**: ~16 bytes overhead per struct (negligible)
- **Dual representation**: Store both `parameters` and `parameters_ph`
  - Acceptable: Clarity and type-safety worth it
  - Future: Could deprecate `parameters` after Phase 4

### Speed
- **Parameter transformations**: Log/exp operations
  - Minimal overhead (< 1% of total runtime)
- **Synchronization**: Extra update step in `set_parameters!()`
  - Negligible compared to likelihood evaluation

---

## Dependencies

**Required**:
- ✅ ParameterHandling.jl (already in Project.toml)

**No New Dependencies Needed!**

---

## Risks & Mitigations

### Risk 1: Breaking Existing Code
- **Mitigation**: Extensive testing, backward compatibility
- **Severity**: Low (additive changes mostly)

### Risk 2: Optimization Convergence Changes
- **Mitigation**: Compare results before/after, adjust tolerances
- **Severity**: Medium (need validation)

### Risk 3: Symbolics Dependency Still Broken
- **Mitigation**: Phase 3 work doesn't require full package load for development
- **Severity**: Low (can work around)

---

## Success Criteria

Phase 3 is complete when:

- ✅ All model structs are mutable
- ✅ `set_parameters!()` synchronizes both representations
- ✅ `get_parameters_*()` functions implemented and tested
- ✅ `parameters_ph` properly initialized in model construction
- ✅ Optimization uses ParameterHandling transformations
- ✅ All existing tests pass
- ✅ New tests for ParameterHandling integration pass
- ✅ Documentation updated

---

## Timeline Estimate

**Total Effort**: 8-12 hours

**Breakdown**:
- Tasks 3.1-3.3: ~2 hours (make mutable, update setters, add getters)
- Tasks 3.4-3.5: ~2 hours (update construction)
- Task 3.6: ~3 hours (update optimization)
- Task 3.7: ~2 hours (testing)
- Task 3.8: ~1 hour (documentation)
- Buffer: ~2 hours (debugging, edge cases)

**Suggested Approach**:
- Session 1 (2-3 hrs): Tasks 3.1-3.3
- Session 2 (2-3 hrs): Tasks 3.4-3.5
- Session 3 (3-4 hrs): Task 3.6
- Session 4 (2 hrs): Tasks 3.7-3.8

---

## Phase 4 Preview (Future)

After Phase 3, we can:

1. **Full Likelihood Rewrite**
   - Remove backward compatibility layer
   - Use only callable interface
   - Cleaner, faster likelihood code

2. **Deprecate VectorOfVectors**
   - Use only `parameters_ph`
   - Single source of truth

3. **Spline Implementation**
   - Add spline hazard generators
   - Integrate with ParameterHandling

4. **Cleanup**
   - Remove old deprecated hazard types
   - Remove commented-out code
   - Final optimization pass

---

## Questions for Discussion

1. **Mutable vs Immutable**: Confirm mutable structs are acceptable?
2. **Dual Storage**: Keep both `parameters` and `parameters_ph` or deprecate VectorOfVectors eventually?
3. **Testing Priority**: Focus on unit tests or integration tests first?
4. **Optimization Integration**: Update all optimization functions or start with subset?
5. **Timeline**: Prefer multiple short sessions or fewer longer sessions?

---

**Ready to Proceed?** Let me know which task to start with! 🚀

**Recommendation**: Start with Task 3.1 (make structs mutable) - quick win that unblocks everything else.
