# Parameter Handling: Phase 3 - Validated Parameter Setting
**Date**: December 7, 2024  
**Status**: DETAILED SPECIFICATION - Ready for review

---

## Phase 3 Overview

**Goal**: Add validated parameter setting with optional order-independent named parameter input.

**Estimated Effort**: 8-12 hours

**Deliverables**:
- Enhanced `set_parameters!` with validation
- Automatic parameter reordering for named input
- Warning system for positional usage
- Comprehensive error messages

**Risk**: Low-Medium - New functionality alongside existing methods, opt-in by default

---

## Task 3.1: Enhanced `set_parameters!` with Validation

**Duration**: 6-8 hours

**Objective**: Implement validated parameter setting that accepts both named and positional input, with automatic reordering for named parameters.

### Changes Required

**File: `src/helpers.jl`**

**Location: Lines 126-159 - Rewrite `set_parameters!` function**

Current implementation:
```julia
function set_parameters!(
    model::MultistateProcess,
    newvalues::NamedTuple;
    scale::Symbol = :log
)
    # Simple positional update, no validation
    for (hazname, params) in pairs(newvalues)
        hazind = model.hazkeys[hazname]
        # Direct assignment without checking parameter names or order
        # ...
    end
end
```

New implementation with validation:

```julia
"""
    set_parameters!(model::MultistateProcess, newvalues::NamedTuple; 
                    scale::Symbol=:log, validate::Bool=true)

Set model parameters with optional validation and automatic reordering.

# Arguments
- `model`: MultistateProcess model
- `newvalues`: NamedTuple of parameters by hazard name
- `scale`: Parameter scale - `:log` (default) or `:natural`
- `validate`: Enable parameter validation (default: true)

# Parameter Input Formats

## Named Parameters (Recommended - Order Independent)
```julia
# Weibull hazard with covariates
set_parameters!(model, h12=(shape=1.5, scale=0.2, age=0.3))

# Order doesn't matter!
set_parameters!(model, h12=(age=0.3, scale=0.2, shape=1.5))  # ✅ Automatically reordered
```

## Positional Parameters (Legacy - Order Matters)
```julia
# MUST be in correct order: [shape, scale, age]
set_parameters!(model, (h12=[1.5, 0.2, 0.3],))  # ⚠️ Warning issued if validate=true
```

## Keyword Arguments (Convenient)
```julia
set_parameters!(model; h12=(shape=1.5, scale=0.2))
```

# Validation Behavior

When `validate=true` (default):
- Named parameters: Validates all expected parameters present, no extras, automatically reorders
- Positional parameters: Warns about fragility, checks length only
- Errors on: missing parameters, unexpected parameters, length mismatch

When `validate=false`:
- No validation or warnings
- Direct positional assignment
- Use for performance-critical code where parameters known correct

# Examples

```julia
# Safe: Named parameters detected and validated
set_parameters!(model, h12=(shape=1.5, scale=0.2, age=0.3))

# Safe: Order independent
set_parameters!(model, h12=(scale=0.2, age=0.3, shape=1.5))

# Warning: Positional usage
set_parameters!(model, (h12=[1.5, 0.2, 0.3],))
# Warning: Setting parameters for h12 using positional vector.
#          Expected order: [:shape, :scale, :age]
#          Use named tuple for safety: h12=(shape=..., scale=..., age=...)

# Error: Missing parameter
set_parameters!(model, h12=(shape=1.5, age=0.3))
# ERROR: Missing parameters for h12: [:scale]

# Error: Extra parameter
set_parameters!(model, h12=(shape=1.5, scale=0.2, weight=0.5))
# ERROR: Unexpected parameters for h12: [:weight]

# Error: Wrong length positional
set_parameters!(model, (h12=[1.5, 0.2],))
# ERROR: Parameter vector length 2 doesn't match expected 3 for h12

# Performance mode: Skip validation
set_parameters!(model, (h12=[1.5, 0.2, 0.3],); validate=false)
```
"""
function set_parameters!(
    model::MultistateProcess,
    newvalues::NamedTuple;
    scale::Symbol = :log,
    validate::Bool = true
)
    # Validate scale argument
    scale ∈ [:log, :natural] || 
        error("scale must be :log or :natural, got :$scale")
    
    # Process each hazard
    for (hazname, params) in pairs(newvalues)
        # Get hazard information
        haskey(model.hazkeys, hazname) || 
            error("Unknown hazard name: $hazname. Available hazards: $(keys(model.hazkeys))")
        
        hazind = model.hazkeys[hazname]
        hazard = model.hazards[hazind]
        expected_names = hazard.parnames
        npar_expected = hazard.npar_total
        
        # Convert input to parameter vector
        if params isa NamedTuple && validate
            # Named parameters - validate and reorder
            provided_names = collect(keys(params))
            
            # Check all expected parameters provided
            missing_params = setdiff(expected_names, provided_names)
            if !isempty(missing_params)
                error("""
                    Missing parameters for $hazname: $missing_params
                    Expected: $expected_names
                    Provided: $provided_names
                    """)
            end
            
            # Check no unexpected parameters
            extra_params = setdiff(provided_names, expected_names)
            if !isempty(extra_params)
                error("""
                    Unexpected parameters for $hazname: $extra_params
                    Expected: $expected_names
                    Provided: $provided_names
                    
                    Did you mean one of: $(join(expected_names, ", "))?
                    """)
            end
            
            # Reorder to match expected order (key robustness improvement!)
            param_vec = [params[name] for name in expected_names]
            
        elseif params isa NamedTuple && !validate
            # Named without validation - just extract in order
            param_vec = [params[name] for name in expected_names]
            
        elseif params isa AbstractVector && validate
            # Positional with validation - warn and check length
            if length(params) != npar_expected
                error("""
                    Parameter vector length $(length(params)) doesn't match expected $npar_expected for $hazname
                    Expected parameters: $expected_names
                    
                    Use named tuple for safety: $hazname=($(expected_names[1])=..., $(expected_names[2])=..., ...)
                    """)
            end
            
            # Issue warning about positional usage
            @warn """
                Setting parameters for $hazname using positional vector.
                Expected order: $expected_names
                
                For robust code, use named parameters:
                  $hazname=($(join(["$n=..." for n in expected_names], ", ")))
                
                To suppress this warning, use validate=false
                """ maxlog=1
            
            param_vec = params
            
        elseif params isa AbstractVector && !validate
            # Positional without validation - direct use
            param_vec = params
            
        else
            error("""
                Parameters for $hazname must be NamedTuple or Vector, got $(typeof(params))
                
                Named tuple:  $hazname=(shape=1.5, scale=0.2, ...)
                Vector:       $hazname=[1.5, 0.2, ...]
                """)
        end
        
        # Apply scale transformation if needed
        if scale == :natural
            # User provided natural scale, convert baseline to log scale
            npar_baseline = hazard.npar_baseline
            param_vec[1:npar_baseline] = log.(param_vec[1:npar_baseline])
            # Covariates stay as-is (already on natural scale)
        end
        
        # Update model parameters using existing rebuild logic
        # Build nested structure
        hazard_params = build_hazard_params(
            param_vec,
            hazard.parnames,
            hazard.npar_baseline,
            hazard.npar_total
        )
        
        # Update in model.parameters.nested
        old_nested = model.parameters.nested
        new_nested = merge(old_nested, NamedTuple{(hazname,)}((hazard_params,)))
        
        # Rebuild full parameter structure
        model.parameters = rebuild_parameters(model, collect(values(new_nested)))
    end
    
    return nothing
end

"""
    set_parameters!(model::MultistateProcess; kwargs...)

Convenience method accepting keyword arguments.

# Example
```julia
set_parameters!(model; h12=(shape=1.5, scale=0.2), h23=(intercept=0.8))
```
"""
function set_parameters!(model::MultistateProcess; kwargs...)
    set_parameters!(model, NamedTuple(kwargs))
end
```

---

## Task 3.2: Testing Validation Logic

**Duration**: 2-4 hours

**Objective**: Comprehensive tests covering all validation scenarios.

### Test File: `test/test_parameter_validation.jl` (NEW)

```julia
using Test
using MultistateModels
using DataFrames
using StatsModels

@testset "Parameter Validation" begin
    # Setup test model
    tmat = [0 1 0; 0 0 1; 0 0 0]
    dat = DataFrame(
        id = repeat(1:20, inner=2),
        tstart = repeat([0.0, 1.0], 20),
        tstop = repeat([1.0, 2.0], 20),
        from = repeat([1, 2], 20),
        to = repeat([2, 3], 20),
        age = rand(40),
        sex = rand(0:1, 40)
    )
    
    model = multistatemodel(
        @formula(time ~ age + sex),
        dat,
        tmat,
        hazards = [Weibull(:PH), Weibull(:PH)]
    )
    
    @testset "Named parameters - correct order" begin
        # Should succeed without warnings
        @test_nowarn set_parameters!(
            model, 
            h12=(shape=1.5, scale=0.2, age=0.3, sex=0.1),
            h23=(shape=2.0, scale=0.3, age=0.2, sex=-0.1)
        )
        
        # Verify values set correctly
        params = model.parameters.nested
        @test params.h12.baseline.shape == log(1.5)
        @test params.h12.baseline.scale == log(0.2)
        @test params.h12.covariates.age == 0.3
    end
    
    @testset "Named parameters - wrong order (auto-reorder)" begin
        # Order scrambled - should automatically reorder
        @test_nowarn set_parameters!(
            model,
            h12=(sex=0.1, scale=0.2, age=0.3, shape=1.5)  # Wrong order!
        )
        
        # Verify correct values in correct positions
        params = model.parameters.nested
        @test params.h12.baseline.shape == log(1.5)  # First baseline param
        @test params.h12.baseline.scale == log(0.2)  # Second baseline param
        @test params.h12.covariates.age == 0.3       # First covariate
        @test params.h12.covariates.sex == 0.1       # Second covariate
    end
    
    @testset "Named parameters - missing parameter" begin
        # Missing 'sex' parameter
        @test_throws ErrorException set_parameters!(
            model,
            h12=(shape=1.5, scale=0.2, age=0.3)
        )
        
        # Check error message mentions missing parameter
        try
            set_parameters!(model, h12=(shape=1.5, scale=0.2, age=0.3))
        catch e
            @test occursin("Missing parameters", e.msg)
            @test occursin("sex", e.msg)
        end
    end
    
    @testset "Named parameters - extra parameter" begin
        # Extra 'weight' parameter
        @test_throws ErrorException set_parameters!(
            model,
            h12=(shape=1.5, scale=0.2, age=0.3, sex=0.1, weight=0.5)
        )
        
        # Check error message
        try
            set_parameters!(model, h12=(shape=1.5, scale=0.2, age=0.3, sex=0.1, weight=0.5))
        catch e
            @test occursin("Unexpected parameters", e.msg)
            @test occursin("weight", e.msg)
        end
    end
    
    @testset "Positional parameters - correct length with warning" begin
        # Should work but issue warning
        @test_logs (:warn, r"positional vector") set_parameters!(
            model,
            (h12=[log(1.5), log(0.2), 0.3, 0.1],)
        )
        
        # Verify values set
        params = model.parameters.nested
        @test params.h12.baseline.shape == log(1.5)
    end
    
    @testset "Positional parameters - wrong length" begin
        # Wrong length - should error
        @test_throws ErrorException set_parameters!(
            model,
            (h12=[log(1.5), log(0.2)],)  # Only 2 params, need 4
        )
        
        try
            set_parameters!(model, (h12=[log(1.5), log(0.2)],))
        catch e
            @test occursin("length", lowercase(e.msg))
            @test occursin("2", e.msg)
            @test occursin("4", e.msg)
        end
    end
    
    @testset "Validation disabled" begin
        # Positional without warning when validate=false
        @test_nowarn set_parameters!(
            model,
            (h12=[log(1.5), log(0.2), 0.3, 0.1],);
            validate=false
        )
    end
    
    @testset "Natural scale conversion" begin
        # Set parameters on natural scale
        set_parameters!(
            model,
            h12=(shape=1.5, scale=0.2, age=0.3, sex=0.1);
            scale=:natural
        )
        
        # Should be converted to log scale for baseline
        params = model.parameters.nested
        @test params.h12.baseline.shape ≈ log(1.5)
        @test params.h12.baseline.scale ≈ log(0.2)
        # Covariates stay as-is
        @test params.h12.covariates.age == 0.3
    end
    
    @testset "Unknown hazard name" begin
        @test_throws ErrorException set_parameters!(
            model,
            h99=(shape=1.5, scale=0.2)  # h99 doesn't exist
        )
    end
    
    @testset "Keyword argument convenience" begin
        # Using kwargs instead of NamedTuple
        @test_nowarn set_parameters!(
            model;
            h12=(shape=1.5, scale=0.2, age=0.3, sex=0.1),
            h23=(shape=2.0, scale=0.3, age=0.2, sex=-0.1)
        )
    end
end

@testset "Parameter Validation - Edge Cases" begin
    # Test with exponential (1 baseline parameter)
    tmat = [0 1; 0 0]
    dat = DataFrame(id=1:10, tstart=0.0, tstop=rand(10).+1, age=rand(10))
    
    model_exp = multistatemodel(
        @formula(time ~ age),
        dat,
        tmat,
        hazards = [Exponential(:PH)]
    )
    
    @testset "Exponential - named" begin
        @test_nowarn set_parameters!(
            model_exp,
            h12=(intercept=0.5, age=0.3)
        )
        
        # Wrong order
        @test_nowarn set_parameters!(
            model_exp,
            h12=(age=0.3, intercept=0.5)  # Auto-reordered
        )
        
        params = model_exp.parameters.nested
        @test params.h12.baseline.intercept == log(0.5)
        @test params.h12.covariates.age == 0.3
    end
    
    @testset "No covariates" begin
        dat_simple = DataFrame(id=1:10, tstart=0.0, tstop=rand(10).+1)
        model_simple = multistatemodel(
            @formula(time ~ 1),
            dat_simple,
            tmat,
            hazards = [Weibull(:PH)]
        )
        
        # Only baseline parameters
        @test_nowarn set_parameters!(
            model_simple,
            h12=(shape=1.5, scale=0.2)
        )
        
        params = model_simple.parameters.nested
        @test !haskey(params.h12, :covariates)
    end
end
```

---

## Task 3.3: Documentation Updates

**Duration**: 1-2 hours

**Objective**: Update user-facing documentation to explain new validation features.

### File: `docs/src/api_reference.md`

Add section:

```markdown
## Parameter Setting with Validation

### Named Parameters (Recommended)

Named parameters provide order-independent, validated parameter setting:

\`\`\`julia
# Weibull model with covariates
model = multistatemodel(@formula(time ~ age + sex), data, tmat, hazards=[Weibull(:PH)])

# Set parameters by name - order doesn't matter!
set_parameters!(model, h12=(scale=0.2, age=0.3, shape=1.5, sex=0.1))
\`\`\`

**Benefits**:
- ✅ Order independent - parameters automatically reordered
- ✅ Validation - detects missing or extra parameters
- ✅ Self-documenting - clear what each value represents
- ✅ Robust - prevents parameter order bugs

### Parameter Names by Hazard Family

| Hazard Family | Baseline Parameters | Description |
|--------------|-------------------|-------------|
| Exponential  | `intercept` | Log-scale intercept |
| Weibull      | `shape`, `scale` | Shape (α) and scale (λ) parameters |
| Gompertz     | `shape`, `scale` | Shape (γ) and scale (λ) parameters |
| Splines      | `sp1`, `sp2`, ... | Spline basis coefficients |

Covariate parameters use the variable names from the formula.

### Positional Parameters (Legacy)

For backward compatibility, positional parameter vectors are still supported:

\`\`\`julia
# Parameters MUST be in correct order
set_parameters!(model, (h12=[log(1.5), log(0.2), 0.3, 0.1],))
# Warning: Setting parameters for h12 using positional vector...
\`\`\`

**Order requirements**:
1. Baseline parameters first (in hazard family order)
2. Covariate coefficients second (in formula order)

### Validation Options

\`\`\`julia
# Default: validation enabled
set_parameters!(model, h12=(shape=1.5, scale=0.2, age=0.3))

# Disable for performance-critical code
set_parameters!(model, (h12=[log(1.5), log(0.2), 0.3],); validate=false)
\`\`\`

### Scale Conversion

Parameters can be provided on natural scale:

\`\`\`julia
# Natural scale for baseline, direct values for covariates
set_parameters!(model, h12=(shape=1.5, scale=0.2, age=0.3); scale=:natural)

# Internally converted to log scale for baseline:
# log(1.5), log(0.2), 0.3
\`\`\`
```

---

## Phase 3 Summary

**Total Duration**: 8-12 hours

**Completed Tasks**:
1. ✅ Enhanced `set_parameters!` with validation (6-8 hrs)
2. ✅ Comprehensive validation tests (2-4 hrs)
3. ✅ Documentation updates (1-2 hrs)

**Key Features Delivered**:
- Order-independent named parameter setting
- Automatic parameter reordering
- Validation with clear error messages
- Warning system for positional usage
- Backward compatibility maintained
- Performance opt-out via `validate=false`

**Testing Coverage**:
- Named parameters (correct order, wrong order, missing, extra)
- Positional parameters (correct length, wrong length, warnings)
- Validation on/off
- Natural scale conversion
- Edge cases (exponential, no covariates, unknown hazards)

**Risk Assessment**: ✅ LOW-MEDIUM
- New methods alongside existing code
- Opt-in validation by default
- Clear migration path for users
- Comprehensive test coverage

**Next Steps**:
1. Review complete implementation plan (Phases 1-3)
2. Approve for implementation
3. Create feature branch
4. Begin Phase 1 Task 1.1

---

## Complete Implementation Timeline

### Phase 1: Foundation (12-15 hours)
- Task 1.1: Add parnames to structs (3-4 hrs)
- Task 1.2: NamedTuple baseline in build_hazard_params (4-5 hrs)
- Task 1.3: Helper functions (2-3 hrs)
- Task 1.4: ParameterHandling.jl tests (1-2 hrs)

### Phase 2: Hazard Functions (15-20 hours)
- Task 2.1: Linear predictor builder (2-3 hrs)
- Task 2.2: Exponential generators (2-3 hrs)
- Task 2.3: Weibull generators (3-4 hrs)
- Task 2.4: Gompertz generators (3-4 hrs)
- Task 2.5: Likelihood callers (4-6 hrs)
- Task 2.6: Simulation code (2-3 hrs)

### Phase 3: Validation (8-12 hours)
- Task 3.1: Enhanced set_parameters! (6-8 hrs)
- Task 3.2: Validation tests (2-4 hrs)
- Task 3.3: Documentation (1-2 hrs)

**Total: 35-47 hours over 4-5 weeks**

---

## Success Criteria

### Must Have ✅
- [ ] Parameter names stored in all hazard objects
- [ ] Baseline/covariate parameters as NamedTuples
- [ ] Hazard functions use named access
- [ ] Validation detects wrong order/missing/extra parameters
- [ ] All existing tests pass
- [ ] Performance overhead <5%

### Should Have ✅
- [ ] Clear error messages with suggestions
- [ ] Warning system for positional usage
- [ ] Documentation with examples
- [ ] Comprehensive test coverage

### Nice to Have
- [ ] Automatic suggestions for typos in parameter names
- [ ] Performance mode benchmarks in docs
- [ ] Migration guide for external users

---

## Ready for Review

This completes the detailed implementation specification for all three phases of the parameter handling robustness improvements.

**Documents Created**:
1. `parameter_handling_detailed_implementation.md` - Phases 1 & 2
2. `parameter_handling_phase3_validation.md` - Phase 3 (this document)

**Total Specification**: ~35-47 hours of work across 3 phases, with complete file locations, line numbers, code examples, and test cases.

**Awaiting approval to proceed with implementation.**
