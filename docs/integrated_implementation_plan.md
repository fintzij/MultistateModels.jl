# Integrated Implementation Plan: Named Parameters with Manual ReConstructor

**Date**: December 7, 2025  
**Status**: COMPREHENSIVE PLAN - Ready for implementation  
**Strategy**: Implement manual ReConstructor first, then proceed with Phase 1 parameter naming

---

## Executive Summary

This plan integrates two related improvements:

1. **Manual ReConstructor Implementation** (6-8 hours)
   - Extract core ModelWrappers.jl pattern manually
   - Avoid dependency conflicts with SciML ecosystem
   - Enable polymorphic unflatten for AD compatibility

2. **Phase 1: Named Parameters** (12-15 hours from original plan)
   - Store parameter names in hazard structs
   - Convert to NamedTuple structure
   - Update helper functions

**Total Estimated Time**: 18-23 hours over 3-5 sessions

**Why This Order**: ReConstructor provides better foundation than ParameterHandling.jl for the named parameter work in Phase 1.

---

## Strategic Decision: Manual Implementation vs Dependency

### The Dependency Conflict

Attempting to add ModelWrappers.jl causes unsatisfiable requirements:
- ModelWrappers requires Distributions ≥ 0.24, Soss, etc.
- Current package has tight SciML constraints (NonlinearSolve = 4.12.0, etc.)
- Resolver cannot find compatible versions

### The Solution

**Extract and implement** the core ReConstructor pattern manually (~300 lines):
- ✅ No dependency conflicts
- ✅ Full control over implementation
- ✅ Only implement what we need (skip constraints/bijectors)
- ✅ Proven pattern (used in ModelWrappers.jl with 42 releases)
- ✅ Better than 45-line `safe_unflatten` workaround

---

## Implementation Strategy Overview

```
┌─────────────────────────────────────────────────────────────┐
│ PART A: Manual ReConstructor (6-8 hours)                    │
│   - Replaces ParameterHandling.jl's monomorphic unflatten   │
│   - Provides polymorphic AD-compatible flatten/unflatten    │
│   - Foundation for Phase 1 work                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ PART B: Phase 1 Named Parameters (12-15 hours)              │
│   - Add parnames to hazard structs                          │
│   - Convert build_hazard_params to NamedTuple               │
│   - Update helper functions                                 │
│   - Integration testing                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## PART A: Manual ReConstructor Implementation

### Overview

Implement the core pattern from ModelWrappers.jl that solves the AD type problem.

**The Core Problem**:
```julia
# ParameterHandling.jl - BREAKS AD
flat, unflatten = ParameterHandling.flatten(params)
reconstructed = unflatten(flat_dual)  # Returns Float64, not Dual!

# Our Solution - PRESERVES AD TYPES
reconstructor = ReConstructor(params)
reconstructed = unflattenAD(reconstructor, flat_dual)  # Returns Dual!
```

**Key Insight**: Build TWO unflatten functions at construction time:
1. **Strict**: Type-stable (Float64 → Float64) for regular use
2. **Flexible**: Type-polymorphic (Dual → Dual OR Float64 → Float64) for AD

---

### A1: Core Type Definitions (1 hour)

**File**: `src/helpers.jl` - Add at top of file (after module docstring)

**Deliverable**: Type hierarchy and configuration struct

```julia
############################################################################################
# ReConstructor - Manual implementation of core ModelWrappers.jl pattern
# Provides polymorphic flatten/unflatten for AD compatibility
# 
# Key difference from ParameterHandling.jl:
#   - unflatten: Float64 → Float64 (type-stable)
#   - unflattenAD: Dual → Dual OR Float64 → Float64 (type-flexible)
############################################################################################

"""
Abstract type for flatten configuration.
"""
abstract type FlattenTypes end

"""
Only flatten continuous (Real) values. Integers are skipped.
"""
struct FlattenContinuous <: FlattenTypes end

"""
Flatten all numeric values including integers.
"""
struct FlattenAll <: FlattenTypes end

"""
Abstract type for unflatten behavior.
"""
abstract type UnflattenTypes end

"""
Unflatten with strict type conversion - always returns original types.
Used for regular operations where type stability is critical.
"""
struct UnflattenStrict <: UnflattenTypes end

"""
Unflatten preserving input type - adapts to Dual types during AD.
Used for automatic differentiation where type must flow through.
"""
struct UnflattenFlexible <: UnflattenTypes end

"""
    FlattenDefault{T,F}

Configuration for flatten/unflatten behavior.

# Fields
- `output::Type{T}`: Output type for flattened vector (default: Float64)
- `flattentype::F`: Which values to flatten (default: FlattenContinuous)
"""
struct FlattenDefault{T<:AbstractFloat, F<:FlattenTypes}
    output::Type{T}
    flattentype::F
end

function FlattenDefault(;
    output::Type{T}=Float64,
    flattentype::F=FlattenContinuous()
) where {T, F<:FlattenTypes}
    return FlattenDefault(output, flattentype)
end

"""
    ReConstructor{F,S,T,U,V}

Pre-computed flatten/unflatten functions for a parameter structure.

Stores closures that know how to:
- Flatten parameters to vector (type-stable)
- Flatten parameters to vector (type-flexible for AD)
- Unflatten vector to parameters (strict: type-stable)
- Unflatten vector to parameters (flexible: AD-compatible)

# Usage
```julia
params = (h12 = (baseline = (shape = 1.5, scale = 0.2),))
reconstructor = ReConstructor(params)

# Regular use (type-stable)
flat = flatten(reconstructor, params)
recon = unflatten(reconstructor, flat)

# AD use (type-flexible)
using ForwardDiff
grad = ForwardDiff.gradient(θ -> f(unflattenAD(reconstructor, θ)), flat)
```
"""
struct ReConstructor{F<:FlattenDefault, S<:Function, T<:Function, U<:Function, V<:Function}
    default::F
    flatten_strict::S
    flatten_flexible::T
    unflatten_strict::U
    unflatten_flexible::V
    
    function ReConstructor(
        default::F,
        flatten_strict::S,
        flatten_flexible::T,
        unflatten_strict::U,
        unflatten_flexible::V
    ) where {F<:FlattenDefault, S<:Function, T<:Function, U<:Function, V<:Function}
        return new{F,S,T,U,V}(default, flatten_strict, flatten_flexible, 
                              unflatten_strict, unflatten_flexible)
    end
end
```

**Testing**:
```julia
@testset "ReConstructor types" begin
    # Test type hierarchy
    @test FlattenContinuous <: FlattenTypes
    @test FlattenAll <: FlattenTypes
    @test UnflattenStrict <: UnflattenTypes
    @test UnflattenFlexible <: UnflattenTypes
    
    # Test FlattenDefault construction
    config = FlattenDefault()
    @test config.output == Float64
    @test config.flattentype isa FlattenContinuous
    
    config2 = FlattenDefault(output=Float32, flattentype=FlattenAll())
    @test config2.output == Float32
    @test config2.flattentype isa FlattenAll
end
```

---

### A2: Construction Functions for Base Types (2 hours)

**File**: `src/helpers.jl` - Continue after type definitions

**Deliverable**: construct_flatten for Real, Vector

#### A2.1: Real (Scalar) Constructor

```julia
"""
    construct_flatten(output, flattentype, unflattentype, x::Real)

Build flatten/unflatten functions for a single Real value.

# Key Difference
- Strict: `convert(R, only(v))` - forces type conversion
- Flexible: `only(v)` - preserves input type (critical for AD!)

# Returns
Tuple of (flatten_func, unflatten_func)
"""
function construct_flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::UnflattenStrict,
    x::R
) where {T<:AbstractFloat, F<:FlattenTypes, R<:Real}
    
    # Strict: always convert to original type R
    flatten_to_Real(val::Real) = T[val]
    unflatten_to_Real(v::Union{<:Real, AbstractVector{<:Real}}) = convert(R, only(v))
    
    return flatten_to_Real, unflatten_to_Real
end

function construct_flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::UnflattenFlexible,
    x::R
) where {T<:AbstractFloat, F<:FlattenTypes, R<:Real}
    
    # Flexible: preserve input type (NO convert!)
    flatten_to_Real_AD(val::Real) = [val]
    unflatten_to_Real_AD(v::Union{<:Real, AbstractVector{<:Real}}) = only(v)
    
    return flatten_to_Real_AD, unflatten_to_Real_AD
end
```

#### A2.2: Vector Constructor

```julia
"""
    construct_flatten(output, flattentype, unflattentype, x::AbstractVector)

Build flatten/unflatten functions for a Vector.

Skips integer vectors if flattentype is FlattenContinuous.
"""
function construct_flatten(
    output::Type{T},
    flattentype::FlattenContinuous,
    unflattentype::UnflattenStrict,
    x::AbstractVector{R}
) where {T<:AbstractFloat, R<:Real}
    
    # Strict: preserve original vector type
    flatten_to_Vector(val::AbstractVector) = convert(Vector{T}, val)
    unflatten_to_Vector(v::Union{<:Real, AbstractVector{<:Real}}) = convert(Vector{R}, v)
    
    return flatten_to_Vector, unflatten_to_Vector
end

function construct_flatten(
    output::Type{T},
    flattentype::FlattenContinuous,
    unflattentype::UnflattenFlexible,
    x::AbstractVector{R}
) where {T<:AbstractFloat, R<:Real}
    
    # Flexible: create vector of input type
    flatten_to_Vector_AD(val::AbstractVector) = collect(val)
    function unflatten_to_Vector_AD(v::Union{<:Real, AbstractVector{S}}) where S<:Real
        return Vector{S}(v)  # S could be Dual!
    end
    
    return flatten_to_Vector_AD, unflatten_to_Vector_AD
end

# Skip integer vectors if FlattenContinuous
function construct_flatten(
    output::Type{T},
    flattentype::FlattenContinuous,
    unflattentype::U,
    x::AbstractVector{<:Integer}
) where {T<:AbstractFloat, U<:UnflattenTypes}
    
    # Return empty flatten, passthrough unflatten
    flatten_skip(val::AbstractVector) = T[]
    unflatten_skip(v::Union{<:Real, AbstractVector{<:Real}}) = x
    
    return flatten_skip, unflatten_skip
end
```

**Testing**:
```julia
@testset "construct_flatten - base types" begin
    # Test Real strict
    flatten, unflatten = construct_flatten(Float64, FlattenContinuous(), UnflattenStrict(), 1.5)
    flat = flatten(1.5)
    @test flat == [1.5]
    @test unflatten(flat) === 1.5
    
    # Test Real flexible
    flatten_ad, unflatten_ad = construct_flatten(Float64, FlattenContinuous(), UnflattenFlexible(), 1.5)
    flat_ad = flatten_ad(1.5)
    @test unflatten_ad(flat_ad) === 1.5
    
    # Test Vector strict
    vec = [1.0, 2.0, 3.0]
    flatten, unflatten = construct_flatten(Float64, FlattenContinuous(), UnflattenStrict(), vec)
    flat = flatten(vec)
    @test flat == vec
    @test unflatten(flat) == vec
    
    # Test integer skip
    int_vec = [1, 2, 3]
    flatten, unflatten = construct_flatten(Float64, FlattenContinuous(), UnflattenStrict(), int_vec)
    @test flatten(int_vec) == Float64[]
    @test unflatten(Float64[]) == int_vec
end
```

---

### A3: Recursive Construction for Nested Types (2 hours)

**File**: `src/helpers.jl` - Continue after base type constructors

**Deliverable**: construct_flatten for Tuple and NamedTuple (the magic!)

#### A3.1: Tuple Constructor (Recursive)

```julia
"""
    construct_flatten(output, flattentype, unflattentype, x::Tuple)

Build flatten/unflatten functions for a Tuple by recursively processing elements.

This is THE KEY to handling nested structures!

# Algorithm
1. Recursively build flatten/unflatten for each element
2. Flatten once to determine element sizes
3. Compute cumulative sizes for unflatten indexing
4. Return composed flatten/unflatten functions

# Example
For `(1.5, [2.0, 3.0])`:
- Element 1: Real → length 1, cumsize 1
- Element 2: Vector → length 2, cumsize 3
- Flatten: [1.5, 2.0, 3.0]
- Unflatten: elem1 from v[1:1], elem2 from v[2:3]
"""
function construct_flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::U,
    x::Tuple
) where {T<:AbstractFloat, F<:FlattenTypes, U<:UnflattenTypes}
    
    # Step 1: Recursively build flatten/unflatten for each element
    x_constructors = map(xᵢ -> construct_flatten(T, flattentype, unflattentype, xᵢ), x)
    _flatten = first.(x_constructors)
    _unflatten = last.(x_constructors)
    
    # Step 2: Flatten once to determine sizes
    x_vecs = map((flat, xᵢ) -> flat(xᵢ), _flatten, x)
    lengths = map(length, x_vecs)
    cumulative_sizes = cumsum(lengths)
    
    # Step 3: Build composed flatten function
    function flatten_Tuple(val::Tuple)
        return reduce(vcat, map((flat, xᵢ) -> flat(xᵢ), _flatten, val))
    end
    
    # Step 4: Build composed unflatten function
    function unflatten_Tuple(v::Union{R, AbstractVector{R}}) where R<:Real
        return map(_unflatten, lengths, cumulative_sizes) do unflat, len, cumsize
            start_idx = cumsize - len + 1
            return unflat(view(v, start_idx:cumsize))
        end
    end
    
    return flatten_Tuple, unflatten_Tuple
end
```

#### A3.2: NamedTuple Constructor

```julia
"""
    construct_flatten(output, flattentype, unflattentype, x::NamedTuple)

Build flatten/unflatten functions for a NamedTuple by delegating to Tuple constructor.

Preserves field names through type system.

# Critical Difference for AD
- Strict: `typeof(x)(tuple)` - uses concrete types
- Flexible: `NamedTuple{names}(tuple)` - accepts any element types (Dual!)
"""
function construct_flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::UnflattenStrict,
    x::NamedTuple{names}
) where {T<:AbstractFloat, F<:FlattenTypes, names}
    
    # Delegate to Tuple constructor
    _flatten, _unflatten = construct_flatten(T, flattentype, unflattentype, values(x))
    
    function flatten_NamedTuple(val::NamedTuple{names}) where names
        return _flatten(values(val))
    end
    
    function unflatten_NamedTuple(v::Union{<:Real, AbstractVector{<:Real}})
        v_tuple = _unflatten(v)
        return typeof(x)(v_tuple)  # Exact original type
    end
    
    return flatten_NamedTuple, unflatten_NamedTuple
end

function construct_flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::UnflattenFlexible,
    x::NamedTuple{names}
) where {T<:AbstractFloat, F<:FlattenTypes, names}
    
    # Delegate to Tuple constructor
    _flatten, _unflatten = construct_flatten(T, flattentype, unflattentype, values(x))
    
    function flatten_NamedTuple_AD(val::NamedTuple{names}) where names
        return _flatten(values(val))
    end
    
    function unflatten_NamedTuple_AD(v::Union{<:Real, AbstractVector{<:Real}})
        v_tuple = _unflatten(v)
        # CRITICAL: Use NamedTuple{names}(tuple) not typeof(x)(tuple)
        # This allows Dual types to flow through!
        return NamedTuple{names}(v_tuple)
    end
    
    return flatten_NamedTuple_AD, unflatten_NamedTuple_AD
end
```

**Testing**:
```julia
@testset "construct_flatten - nested types" begin
    # Test Tuple
    tup = (1.5, [2.0, 3.0])
    flatten, unflatten = construct_flatten(Float64, FlattenContinuous(), UnflattenStrict(), tup)
    flat = flatten(tup)
    @test flat == [1.5, 2.0, 3.0]
    recon = unflatten(flat)
    @test recon[1] ≈ 1.5
    @test recon[2] ≈ [2.0, 3.0]
    
    # Test NamedTuple strict
    nt = (a = 1.5, b = 0.2)
    flatten, unflatten = construct_flatten(Float64, FlattenContinuous(), UnflattenStrict(), nt)
    flat = flatten(nt)
    @test flat == [1.5, 0.2]
    recon = unflatten(flat)
    @test recon.a ≈ 1.5
    @test recon.b ≈ 0.2
    
    # Test nested NamedTuple
    nested = (h12 = (baseline = (shape = 1.5, scale = 0.2),), h23 = (baseline = (intercept = 0.8,),))
    flatten, unflatten = construct_flatten(Float64, FlattenContinuous(), UnflattenStrict(), nested)
    flat = flatten(nested)
    @test length(flat) == 3
    recon = unflatten(flat)
    @test recon.h12.baseline.shape ≈ 1.5
    @test recon.h23.baseline.intercept ≈ 0.8
end
```

---

### A4: ReConstructor Builder and User API (1 hour)

**File**: `src/helpers.jl` - Continue after construction functions

#### A4.1: ReConstructor Constructor

```julia
"""
    ReConstructor(x)
    ReConstructor(flattendefault, x)

Construct a ReConstructor for parameter structure `x`.

Pre-computes all flatten/unflatten closures for efficient runtime operations.

# Examples
```julia
# Simple
params = (shape = 1.5, scale = 0.2)
reconstructor = ReConstructor(params)

# Nested
params = (
    h12 = (baseline = (shape = 1.5, scale = 0.2), covariates = (age = 0.3,)),
    h23 = (baseline = (intercept = 0.8,),)
)
reconstructor = ReConstructor(params)
```
"""
function ReConstructor(flattendefault::FlattenDefault, x)
    T = flattendefault.output
    F = flattendefault.flattentype
    
    # Build both strict and flexible constructors
    flatten_strict, unflatten_strict = construct_flatten(
        T, F, UnflattenStrict(), x
    )
    flatten_flexible, unflatten_flexible = construct_flatten(
        T, F, UnflattenFlexible(), x
    )
    
    return ReConstructor(
        flattendefault,
        flatten_strict,
        flatten_flexible,
        unflatten_strict,
        unflatten_flexible
    )
end

# Convenience constructor
function ReConstructor(x)
    return ReConstructor(FlattenDefault(), x)
end
```

#### A4.2: User-Facing API

```julia
"""
    flatten(reconstructor::ReConstructor, x)

Flatten parameter structure to vector (type-stable).
"""
function flatten(reconstructor::ReConstructor, x)
    return reconstructor.flatten_strict(x)
end

"""
    flattenAD(reconstructor::ReConstructor, x)

Flatten parameter structure to vector (type-flexible for AD).
"""
function flattenAD(reconstructor::ReConstructor, x)
    return reconstructor.flatten_flexible(x)
end

"""
    unflatten(reconstructor::ReConstructor, v::AbstractVector)

Unflatten vector to parameter structure (type-stable).
"""
function unflatten(reconstructor::ReConstructor, v::AbstractVector)
    return reconstructor.unflatten_strict(v)
end

"""
    unflattenAD(reconstructor::ReConstructor, v::AbstractVector)

Unflatten vector to parameter structure (type-flexible for AD).

CRITICAL: This preserves Dual types during automatic differentiation!
"""
function unflattenAD(reconstructor::ReConstructor, v::AbstractVector)
    return reconstructor.unflatten_flexible(v)
end
```

**Testing**:
```julia
@testset "ReConstructor - complete workflow" begin
    params = (
        h12 = (baseline = (shape = 1.5, scale = 0.2), covariates = (age = 0.3,)),
        h23 = (baseline = (intercept = 0.8,),)
    )
    
    # Build reconstructor
    recon = ReConstructor(params)
    
    # Test flatten
    flat = flatten(recon, params)
    @test length(flat) == 4
    @test flat ≈ [1.5, 0.2, 0.3, 0.8]
    
    # Test unflatten
    reconstructed = unflatten(recon, flat)
    @test reconstructed.h12.baseline.shape ≈ 1.5
    @test reconstructed.h12.baseline.scale ≈ 0.2
    @test reconstructed.h12.covariates.age ≈ 0.3
    @test reconstructed.h23.baseline.intercept ≈ 0.8
    
    # Test AD compatibility
    using ForwardDiff
    dual_vec = ForwardDiff.Dual{Nothing,Float64,4}.(flat, Ref((1.0, 0.0, 0.0, 0.0)))
    recon_dual = unflattenAD(recon, dual_vec)
    @test recon_dual.h12.baseline.shape isa ForwardDiff.Dual
    @test ForwardDiff.value(recon_dual.h12.baseline.shape) ≈ 1.5
end
```

---

### A5: Integration with Existing Code (1-2 hours)

**Deliverable**: Replace ParameterHandling.jl usage with ReConstructor

#### A5.1: Delete safe_unflatten

**File**: `src/helpers.jl` lines 1-45

**Action**: DELETE entire function (45 lines)

```julia
# DELETE:
"""
    safe_unflatten(unflatten::Function, θ::AbstractVector{T}) where T
    ...
"""
function safe_unflatten(unflatten::Function, θ::AbstractVector{T}) where T
    # ... 45 lines ...
end
```

#### A5.2: Update rebuild_parameters

**File**: `src/helpers.jl` line ~50

**Change**:
```julia
# OLD:
function rebuild_parameters(model::AbstractMultistateModel, newvalues::Vector{Vector{T}}) where {T<:Real}
    flat_vec = vcat(newvalues...)
    updated_nested = model.parameters.unflatten(flat_vec)
    # ...
    return (
        flat = flat_vec,
        nested = updated_nested,
        natural = NamedTuple(updated_natural),
        unflatten = model.parameters.unflatten
    )
end

# NEW:
function rebuild_parameters(model::AbstractMultistateModel, newvalues::Vector{Vector{T}}) where {T<:Real}
    flat_vec = vcat(newvalues...)
    updated_nested = unflatten(model.parameters.reconstructor, flat_vec)
    # ...
    return (
        flat = flat_vec,
        nested = updated_nested,
        natural = NamedTuple(updated_natural),
        reconstructor = model.parameters.reconstructor  # Store ReConstructor
    )
end
```

#### A5.3: Update prepare_parameters

**File**: `src/likelihoods.jl` line ~35-45

**Change**:
```julia
# OLD:
function prepare_parameters(θ::AbstractVector{T}, model::AbstractMultistateModel) where {T<:Real}
    parameters = safe_unflatten(model.parameters.unflatten, θ)
    return parameters
end

# NEW:
function prepare_parameters(θ::AbstractVector{T}, model::AbstractMultistateModel) where {T<:Real}
    # Use unflattenAD for AD compatibility
    parameters = unflattenAD(model.parameters.reconstructor, θ)
    return parameters
end
```

#### A5.4: Update Model Generation

**File**: `src/modelgeneration.jl` line ~750-780

**Change**:
```julia
# OLD:
initial_nested = NamedTuple{Tuple(keys(model.hazards))}(...)
flat, unflatten = ParameterHandling.flatten(initial_nested)

model_parameters = (
    flat = flat,
    nested = initial_nested,
    natural = ...,
    unflatten = unflatten
)

# NEW:
initial_nested = NamedTuple{Tuple(keys(model.hazards))}(...)
reconstructor = ReConstructor(initial_nested)
flat = flatten(reconstructor, initial_nested)

model_parameters = (
    flat = flat,
    nested = initial_nested,
    natural = ...,
    reconstructor = reconstructor
)
```

---

### A6: Comprehensive Testing (1 hour)

**File**: `test/test_helpers.jl`

#### Test 1: AD Compatibility (MOST CRITICAL)

```julia
@testset "ReConstructor - ForwardDiff AD compatibility" begin
    using ForwardDiff
    
    params = (shape = 1.5, scale = 0.2)
    recon = ReConstructor(params)
    
    # Test gradient computation
    function test_func(θ)
        p = unflattenAD(recon, θ)
        return p.shape^2 + p.scale^2
    end
    
    flat = flatten(recon, params)
    grad = ForwardDiff.gradient(test_func, flat)
    
    @test grad[1] ≈ 2 * 1.5
    @test grad[2] ≈ 2 * 0.2
    @test !any(isnan, grad)
    
    # Test with nested structure
    nested = (
        h12 = (baseline = (shape = 1.5, scale = 0.2), covariates = (age = 0.3,)),
        h23 = (baseline = (intercept = 0.8,),)
    )
    recon_nested = ReConstructor(nested)
    
    function test_func_nested(θ)
        p = unflattenAD(recon_nested, θ)
        return p.h12.baseline.shape + p.h23.baseline.intercept
    end
    
    flat_nested = flatten(recon_nested, nested)
    grad_nested = ForwardDiff.gradient(test_func_nested, flat_nested)
    
    @test grad_nested[1] ≈ 1.0  # d/d(shape)
    @test grad_nested[4] ≈ 1.0  # d/d(intercept)
end
```

#### Test 2: Performance

```julia
@testset "ReConstructor - Performance" begin
    using BenchmarkTools
    
    large_params = (
        h12 = (baseline = (shape = 1.5, scale = 0.2), covariates = (age = 0.3, sex = 0.1)),
        h13 = (baseline = (shape = 1.8, scale = 0.15), covariates = (age = 0.25,)),
        h23 = (baseline = (intercept = 0.8,),),
        h24 = (baseline = (shape = 1.2, scale = 0.3), covariates = (age = 0.2, sex = 0.15, bmi = 0.05))
    )
    
    recon = ReConstructor(large_params)
    flat = flatten(recon, large_params)
    
    # Benchmark unflatten
    result = @benchmark unflatten($recon, $flat)
    @test result.allocs < 50
    @test median(result).time < 1000  # Under 1 microsecond
    
    # Benchmark unflattenAD
    result_ad = @benchmark unflattenAD($recon, $flat)
    @test result_ad.allocs < 100
    @test median(result_ad).time < 2000  # Under 2 microseconds
    
    println("  unflatten:   $(median(result).time) ns, $(result.allocs) allocations")
    println("  unflattenAD: $(median(result_ad).time) ns, $(result_ad.allocs) allocations")
end
```

---

## PART B: Phase 1 Named Parameters

### Overview

Now that we have ReConstructor providing robust AD-compatible flatten/unflatten, we can proceed with adding parameter names to hazards.

**Prerequisites**: Part A completed and tested

**Duration**: 12-15 hours (from original plan)

---

### B1: Add parnames to Hazard Structs (3-4 hours)

**File**: `src/common.jl`

**Deliverable**: All hazard structs have `parnames::Vector{Symbol}` field

#### Changes

Add `parnames` field to 4 hazard structs (after line 224, 239, 265, 284):

```julia
# Example for SemiMarkovHazard (line 224)
struct SemiMarkovHazard <: AbstractHazard
    hazmat_indices::Vector{Int64}
    nstates::Int64
    hazfunctions::Vector{_Hazard}
    cumhazfunctions::Vector{_CumulativeHazard}
    dhazfunctions::Vector{_HazardGradient}
    dcumhazfunctions::Vector{_CumulativeHazardGradient}
    parnames::Vector{Symbol}        # NEW: [:shape, :scale, :age, ...]
    npar_baseline::Int64
    npar_tv::Int64
    npar_total::Int64
    emat::Union{Nothing, AbstractMatrix{Float64}}
end
```

Repeat for: `MarkovHazard`, `PhaseTypeHazard`, `SplineHazard`

---

### B2: Update Hazard Constructors (2-3 hours)

**File**: `src/modelgeneration.jl`

**Deliverable**: Generate SHORT parameter names (no hazard prefix) and pass to constructors

#### Changes in hazard builders

**Location**: Lines 276-335 (Weibull/Gompertz/Exponential)

```julia
# Generate SHORT parameter names
if hazfamily == :weibull || hazfamily == :gompertz
    baseline_parnames = [:shape, :scale]
else
    baseline_parnames = [:intercept]
end

# Extract variable names from covariate_names (strip h12_ prefix)
covar_short_names = [Symbol(replace(string(name), r"^h\d+_" => "")) 
                     for name in covariate_names]

parnames = vcat(baseline_parnames, covar_short_names)

# Pass to constructor
hazard = SemiMarkovHazard(
    collect(1:length(ctx.hazmat_indices)),
    ctx.nstates,
    hazfuns,
    cumhazfuns,
    dhazfuns,
    dcumhazfuns,
    parnames,           # NEW
    npar_baseline,
    npar_tv,
    npar_total,
    ctx.emat
)
```

Repeat for all hazard types (4 locations)

---

### B3: Update build_hazard_params to Return NamedTuples (4-5 hours)

**File**: `src/helpers.jl`

**Deliverable**: Convert from vector-based to NamedTuple-based parameter structure

#### New Implementation

```julia
"""
    build_hazard_params(log_scale_params, parnames, npar_baseline, npar_total)

Build nested NamedTuple structure for hazard parameters.

# Returns
NamedTuple with structure:
- `baseline`: NamedTuple of baseline parameters (shape=..., scale=...)
- `covariates`: NamedTuple of covariate coefficients (age=..., sex=...) [if present]
"""
function build_hazard_params(
    log_scale_params::AbstractVector{T},
    parnames::Vector{Symbol},
    npar_baseline::Int,
    npar_total::Int
) where {T<:Real}
    
    @assert length(log_scale_params) == npar_total "Parameter vector length mismatch"
    @assert length(parnames) == npar_total "Parameter names length mismatch"
    @assert npar_baseline <= npar_total "Baseline parameters exceed total"
    
    # Extract baseline
    baseline_names = parnames[1:npar_baseline]
    baseline_values = log_scale_params[1:npar_baseline]
    baseline = NamedTuple{Tuple(baseline_names)}(baseline_values)
    
    # Extract covariates if present
    if npar_total > npar_baseline
        covar_names = parnames[(npar_baseline+1):npar_total]
        covar_values = log_scale_params[(npar_baseline+1):npar_total]
        covariates = NamedTuple{Tuple(covar_names)}(covar_values)
        return (baseline = baseline, covariates = covariates)
    else
        return (baseline = baseline,)
    end
end
```

#### Update Callers

1. **rebuild_parameters** (already updated in Part A)
2. **Model generation** (`src/modelgeneration.jl` line ~761)

```julia
# In model generation loop
for (hazname, hazind) in model.hazkeys
    hazard = model.hazards[hazind]
    initial_pars[hazname] = build_hazard_params(
        params_vec[hazind],
        hazard.parnames,        # Use stored names
        hazard.npar_baseline,
        hazard.npar_total
    )
end
```

---

### B4: Add Helper Functions (2-3 hours)

**File**: `src/helpers.jl`

**Deliverable**: Extraction functions for backwards compatibility

```julia
"""Extract baseline values as vector"""
function extract_baseline_values(hazard_params::NamedTuple)
    return collect(values(hazard_params.baseline))
end

"""Extract covariate values as vector"""
function extract_covariate_values(hazard_params::NamedTuple)
    return haskey(hazard_params, :covariates) ? 
           collect(values(hazard_params.covariates)) : Float64[]
end

"""Extract full parameter vector"""
function extract_params_vector(hazard_params::NamedTuple)
    baseline_vals = collect(values(hazard_params.baseline))
    if haskey(hazard_params, :covariates)
        covar_vals = collect(values(hazard_params.covariates))
        return vcat(baseline_vals, covar_vals)
    else
        return baseline_vals
    end
end

"""Extract natural scale parameters"""
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

---

### B5: Comprehensive Testing (1-2 hours)

**File**: `test/test_helpers.jl`

#### Test parnames storage

```julia
@testset "Hazard parnames storage" begin
    tmat = [0 1; 0 0]
    dat = DataFrame(id=1:10, tstart=0.0, tstop=1.0, age=rand(10))
    
    model = multistatemodel(
        @formula(time ~ age),
        dat,
        tmat,
        hazards = [Weibull(:PH)]
    )
    
    hazard = model.hazards[1]
    @test hasfield(typeof(hazard), :parnames)
    @test hazard.parnames == [:shape, :scale, :age]
    @test !any(startswith.(string.(hazard.parnames), "h"))
end
```

#### Test build_hazard_params

```julia
@testset "build_hazard_params - NamedTuple structure" begin
    # Test Weibull baseline only
    params = build_hazard_params([log(1.5), log(0.2)], [:shape, :scale], 2, 2)
    @test params.baseline.shape ≈ log(1.5)
    @test params.baseline.scale ≈ log(0.2)
    @test !haskey(params, :covariates)
    
    # Test with covariates
    params = build_hazard_params(
        [log(1.5), log(0.2), 0.3, 0.1],
        [:shape, :scale, :age, :sex],
        2, 4
    )
    @test haskey(params, :baseline)
    @test haskey(params, :covariates)
    @test params.covariates.age ≈ 0.3
end
```

#### Test extraction helpers

```julia
@testset "Parameter extraction helpers" begin
    params = (
        baseline = (shape = log(1.5), scale = log(0.2)),
        covariates = (age = 0.3, sex = 0.1)
    )
    
    @test extract_baseline_values(params) ≈ [log(1.5), log(0.2)]
    @test extract_covariate_values(params) ≈ [0.3, 0.1]
    @test extract_params_vector(params) ≈ [log(1.5), log(0.2), 0.3, 0.1]
    @test extract_natural_vector(params) ≈ [1.5, 0.2, 0.3, 0.1]
end
```

#### Test ReConstructor integration

```julia
@testset "ReConstructor with NamedTuple parameters" begin
    # Test nested structure matches Phase 1 format
    params = (
        h12 = (
            baseline = (shape = log(1.5), scale = log(0.2)),
            covariates = (age = 0.3, sex = 0.1)
        ),
        h23 = (
            baseline = (intercept = log(0.8),)
        )
    )
    
    recon = ReConstructor(params)
    flat = flatten(recon, params)
    reconstructed = unflatten(recon, flat)
    
    # Verify structure matches
    @test reconstructed.h12.baseline.shape ≈ log(1.5)
    @test reconstructed.h12.covariates.age ≈ 0.3
    
    # Test AD with Phase 1 structure
    using ForwardDiff
    function test_func(θ)
        p = unflattenAD(recon, θ)
        return p.h12.baseline.shape + p.h12.baseline.scale
    end
    
    grad = ForwardDiff.gradient(test_func, flat)
    @test !any(isnan, grad)
end
```

---

## Master Implementation Checklist

### Part A: Manual ReConstructor (6-8 hours)

#### Setup & Types (1 hour)
- [ ] Read ModelWrappers.jl source for understanding
- [ ] Add type definitions to `src/helpers.jl`
- [ ] Add FlattenDefault and ReConstructor structs
- [ ] Run basic type tests

#### Base Constructors (2 hours)
- [ ] Implement construct_flatten for Real (strict & flexible)
- [ ] Implement construct_flatten for Vector (strict & flexible)
- [ ] Handle integer vector skipping
- [ ] Test base constructors independently

#### Recursive Constructors (2 hours)
- [ ] Implement construct_flatten for Tuple (recursive)
- [ ] Implement construct_flatten for NamedTuple (strict & flexible)
- [ ] Test nested structure reconstruction

#### API & Integration (1-2 hours)
- [ ] Implement ReConstructor constructor
- [ ] Implement flatten, unflatten, flattenAD, unflattenAD
- [ ] DELETE safe_unflatten (45 lines)
- [ ] Update rebuild_parameters
- [ ] Update prepare_parameters
- [ ] Update model generation

#### Testing (1 hour)
- [ ] Test basic flatten/unflatten
- [ ] Test AD compatibility with ForwardDiff (CRITICAL)
- [ ] Test performance benchmarks
- [ ] Run full test suite

### Part B: Phase 1 Named Parameters (12-15 hours)

#### Hazard Structs (3-4 hours)
- [ ] Add parnames field to SemiMarkovHazard
- [ ] Add parnames field to MarkovHazard
- [ ] Add parnames field to PhaseTypeHazard
- [ ] Add parnames field to SplineHazard
- [ ] Update all hazard constructors in modelgeneration.jl
- [ ] Generate SHORT parameter names (no prefix)
- [ ] Test parnames storage

#### build_hazard_params (4-5 hours)
- [ ] Rewrite build_hazard_params for NamedTuple output
- [ ] Update rebuild_parameters caller
- [ ] Update model generation caller
- [ ] Add comprehensive parameter structure tests

#### Helper Functions (2-3 hours)
- [ ] Implement extract_baseline_values
- [ ] Implement extract_covariate_values
- [ ] Implement extract_params_vector
- [ ] Implement extract_natural_vector
- [ ] Test all extraction functions

#### Integration Testing (1-2 hours)
- [ ] Test model building with new structure
- [ ] Test parameter reconstruction
- [ ] Test ReConstructor + NamedTuple integration
- [ ] Verify no regressions in existing tests

### Validation

- [ ] All existing tests pass
- [ ] New tests pass (AD, performance, integration)
- [ ] Performance acceptable (< 5% overhead)
- [ ] Code is cleaner (net result documented)
- [ ] Documentation updated

---

## Timeline & Sessions

### Session 1: ReConstructor Foundation (2-3 hours)
- Part A: Tasks A1, A2
- Deliverable: Base type constructors working

### Session 2: ReConstructor Completion (2-3 hours)
- Part A: Tasks A3, A4
- Deliverable: Full ReConstructor working with AD tests passing

### Session 3: ReConstructor Integration (2 hours)
- Part A: Tasks A5, A6
- Deliverable: Integrated into codebase, all tests passing

### Session 4: Named Parameters Start (3-4 hours)
- Part B: Tasks B1, B2
- Deliverable: parnames stored in hazards

### Session 5: Named Parameters Functions (4-5 hours)
- Part B: Tasks B3, B4
- Deliverable: NamedTuple structure implemented

### Session 6: Final Testing (2-3 hours)
- Part B: Task B5
- Deliverable: Complete integration tested

---

## Success Criteria

### Part A Success
- ✅ ReConstructor correctly flattens/unflattens nested NamedTuples
- ✅ unflattenAD preserves Dual types (ForwardDiff tests pass)
- ✅ Performance: < 2µs unflatten, < 100 allocations
- ✅ All existing tests pass
- ✅ safe_unflatten deleted (net -45 lines)

### Part B Success
- ✅ All hazards have parnames field
- ✅ build_hazard_params returns NamedTuple structure
- ✅ Helper functions work correctly
- ✅ Model building produces correct structure
- ✅ No regressions in existing functionality

### Overall Success
- ✅ Named parameters with AD compatibility
- ✅ Clean, maintainable code
- ✅ Comprehensive test coverage
- ✅ Documentation updated
- ✅ Ready for Phase 2 (hazard function updates)

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| ReConstructor bugs | Extensive testing at each step, compare with ParameterHandling.jl |
| AD edge cases | Comprehensive ForwardDiff tests with nested structures |
| Performance regression | Benchmark at each step, optimize if needed |
| Integration issues | Update one function at a time, test incrementally |
| Type stability | Use @code_warntype to verify |

---

## Rollback Plan

If major issues arise:

1. **Git revert** to pre-implementation commit
2. **Partial rollback**: Keep Part A, revert Part B (they're independent)
3. **Document findings**: What worked, what didn't
4. **Reassess**: Adjust plan based on learnings

---

## Next Steps

1. **Review this plan** - Confirm approach
2. **Begin Part A, Session 1** - ReConstructor foundation
3. **Test incrementally** - Don't proceed if tests fail
4. **Complete Part A** before starting Part B
5. **Document progress** - Update this plan as you go

**Estimated Total**: 18-23 hours over 6 sessions
