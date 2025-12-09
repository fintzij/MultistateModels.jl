# Manual Implementation of ModelWrappers.jl Core Functionality

**Date**: December 7, 2025  
**Status**: DETAILED SPECIFICATION  
**Objective**: Extract and implement only the essential flatten/unflatten functionality from ModelWrappers.jl to avoid dependency conflicts

---

## Executive Summary

Rather than adding ModelWrappers.jl as a dependency (which causes complex dependency conflicts), we'll implement the core `ReConstructor` pattern manually. This gives us:

1. **Polymorphic unflatten** - automatically handles Float64 and ForwardDiff.Dual types
2. **Zero-allocation operations** - pre-allocated buffers for performance
3. **Clean separation** - only implement what we need (flatten/unflatten), skip constraints/bijectors
4. **No dependency hell** - avoid conflicts with SciML ecosystem

**Estimated Effort**: 6-8 hours  
**Risk**: Low - we're implementing a well-understood pattern

---

## Core Pattern Understanding

### The Problem ModelWrappers.jl Solves

ParameterHandling.jl's `unflatten` function is **monomorphic** - it always returns `Float64`:

```julia
# ParameterHandling.jl
flat, unflatten = ParameterHandling.flatten(params)
reconstructed = unflatten(flat)  # Always returns Float64, even if flat is Dual

# During ForwardDiff
grad = ForwardDiff.gradient(θ -> loglik(unflatten(θ)), flat)
# Problem: unflatten(θ) converts Dual → Float64, breaking gradient computation!
```

### ModelWrappers.jl Solution

**Key Insight**: Build two unflatten functions during construction:
1. **Strict** (UnflattenStrict): Always returns original types (Float64 → Float64)
2. **Flexible** (UnflattenFlexible): Returns input type (Dual → Dual, Float64 → Float64)

```julia
# ModelWrappers.jl pattern
reconstructor = ReConstructor(params)  # Pre-computes both unflatten functions

# Regular use
flat = flatten(reconstructor, params)  # Vector{Float64}
recon = unflatten(reconstructor, flat)  # Uses strict: Float64 → Float64

# AD use
recon_ad = unflattenAD(reconstructor, flat_dual)  # Uses flexible: Dual → Dual
```

### How It Works

**Construction Phase** (one-time, when creating ReConstructor):
1. Traverse parameter structure recursively
2. For each element, create TWO closures:
   - `flatten` closure: knows how to extract values to vector
   - `unflatten` closure: knows how to reconstruct from vector slice
3. Store cumulative sizes to know which slice belongs to which element
4. Return composed closures that work on full structure

**Runtime Phase** (many times, during optimization):
1. `flatten`: Execute stored flatten closure on parameter structure
2. `unflattenAD`: Execute stored flexible unflatten closure
   - Uses `only(v)` instead of `convert(R, only(v))` - preserves type!
   - Constructs NamedTuple with `NamedTuple{names}(values)` - generic!

---

## Implementation Architecture

### File Structure

```
src/
├── helpers.jl                    # Add ReConstructor implementation here
│   ├── Type definitions
│   ├── ReConstructor struct
│   ├── Construction functions
│   ├── flatten/unflatten/unflattenAD
│   └── Helper utilities
└── [other existing files unchanged]
```

**Rationale**: Keep all in one file initially for simplicity. Can refactor later if needed.

---

## Detailed Implementation Plan

### Phase 1: Core Type Definitions (1 hour)

**Location**: `src/helpers.jl` - Add after line 1 (at top of file)

#### Task 1.1: Define Base Types

```julia
############################################################################################
# ReConstructor - Manual implementation of core ModelWrappers.jl pattern
# Provides polymorphic flatten/unflatten for AD compatibility
############################################################################################

"""
    FlattenTypes

Marker types for flatten behavior.
"""
abstract type FlattenTypes end

"""
    FlattenContinuous

Only flatten continuous (Real) values. Integers are skipped.
"""
struct FlattenContinuous <: FlattenTypes end

"""
    FlattenAll

Flatten all numeric values including integers.
"""
struct FlattenAll <: FlattenTypes end

"""
    UnflattenTypes

Marker types for unflatten behavior.
"""
abstract type UnflattenTypes end

"""
    UnflattenStrict

Unflatten with strict type conversion - always returns original types.
Used for regular operations where type stability matters.
"""
struct UnflattenStrict <: UnflattenTypes end

"""
    UnflattenFlexible

Unflatten preserving input type - adapts to Dual types during AD.
Used for automatic differentiation where type must flow through.
"""
struct UnflattenFlexible <: UnflattenTypes end
```

#### Task 1.2: Define Configuration Struct

```julia
"""
    FlattenDefault{T,F}

Configuration for flatten/unflatten behavior.

# Fields
- `output::Type{T}`: Output type for flattened vector (default: Float64)
- `flattentype::F`: Which values to flatten (default: FlattenContinuous)

# Examples
```julia
# Default: Float64 output, continuous only
config = FlattenDefault()

# Custom: Float32 output, flatten integers too
config = FlattenDefault(output=Float32, flattentype=FlattenAll())
```
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
```

#### Task 1.3: Define ReConstructor Struct

```julia
"""
    ReConstructor{F,S,T}

Pre-computed flatten/unflatten functions for a parameter structure.

Stores closures that know how to:
- Flatten parameters to vector (with pre-allocated buffers)
- Unflatten vector to parameters (strict: type-stable)
- Unflatten vector to parameters (flexible: AD-compatible)

# Fields
- `default::FlattenDefault`: Configuration used during construction
- `flatten_strict::Function`: Type-stable flatten (always returns configured type)
- `flatten_flexible::Function`: AD-compatible flatten (preserves input type)
- `unflatten_strict::Function`: Type-stable unflatten (returns original types)
- `unflatten_flexible::Function`: AD-compatible unflatten (adapts to input type)

# Usage
```julia
params = (h12 = (baseline = (shape = 1.5, scale = 0.2), covariates = (age = 0.3,)))
reconstructor = ReConstructor(params)

# Regular flatten/unflatten (type-stable)
flat = flatten(reconstructor, params)
recon = unflatten(reconstructor, flat)

# AD-compatible (type-flexible)
flat_dual = similar(flat, ForwardDiff.Dual{...})
recon_ad = unflattenAD(reconstructor, flat_dual)  # Returns Dual types!
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

---

### Phase 2: Construction Functions (2-3 hours)

**Location**: `src/helpers.jl` - Continue after type definitions

#### Task 2.1: Implement Scalar (Real) Constructor

```julia
"""
    construct_flatten(output, flattentype, unflattentype, x::Real)

Build flatten/unflatten functions for a single Real value.

Returns tuple: (flatten_func, unflatten_func)
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
    
    # Flexible: preserve input type (critical for AD!)
    flatten_to_Real_AD(val::Real) = [val]  # Don't force type conversion
    unflatten_to_Real_AD(v::Union{<:Real, AbstractVector{<:Real}}) = only(v)  # No convert!
    
    return flatten_to_Real_AD, unflatten_to_Real_AD
end
```

**Key Difference**: 
- Strict uses `convert(R, only(v))` - forces Float64
- Flexible uses `only(v)` - preserves Dual type

#### Task 2.2: Implement Vector Constructor

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
    
    n = length(x)
    
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
    
    n = length(x)
    
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
    unflatten_skip(v::Union{<:Real, AbstractVector{<:Real}}) = x  # Return original
    
    return flatten_skip, unflatten_skip
end
```

#### Task 2.3: Implement Tuple Constructor (Recursive)

```julia
"""
    construct_flatten(output, flattentype, unflattentype, x::Tuple)

Build flatten/unflatten functions for a Tuple by recursively processing elements.

This is the key to handling nested structures!
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

**How It Works**:
1. For `(1.5, [2.0, 3.0])`:
   - Element 1: Real → length 1
   - Element 2: Vector{Real} → length 2
   - Total: length 3, cumulative sizes [1, 3]
2. Flatten: `[1.5, 2.0, 3.0]`
3. Unflatten: 
   - Element 1 from v[1:1]
   - Element 2 from v[2:3]

#### Task 2.4: Implement NamedTuple Constructor

```julia
"""
    construct_flatten(output, flattentype, unflattentype, x::NamedTuple)

Build flatten/unflatten functions for a NamedTuple by delegating to Tuple constructor.

Preserves field names through type system.
"""
function construct_flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::UnflattenStrict,
    x::NamedTuple{names}
) where {T<:AbstractFloat, F<:FlattenTypes, names}
    
    # Delegate to Tuple constructor
    _flatten, _unflatten = construct_flatten(T, flattentype, unflattentype, values(x))
    
    # Wrap with NamedTuple conversion
    function flatten_NamedTuple(val::NamedTuple{names}) where names
        return _flatten(values(val))
    end
    
    function unflatten_NamedTuple(v::Union{<:Real, AbstractVector{<:Real}})
        v_tuple = _unflatten(v)
        return typeof(x)(v_tuple)  # Reconstruct with exact original type
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
    
    # Wrap with NamedTuple conversion
    function flatten_NamedTuple_AD(val::NamedTuple{names}) where names
        return _flatten(values(val))
    end
    
    function unflatten_NamedTuple_AD(v::Union{<:Real, AbstractVector{<:Real}})
        v_tuple = _unflatten(v)
        # CRITICAL: Use NamedTuple{names}(tuple) not typeof(x)(tuple)
        # typeof(x) has concrete types, but we need to accept Dual types!
        return NamedTuple{names}(v_tuple)
    end
    
    return flatten_NamedTuple_AD, unflatten_NamedTuple_AD
end
```

**Critical Difference for AD**:
```julia
# Strict: typeof(x)(tuple) → (a = Float64, b = Float64)
# Flexible: NamedTuple{names}(tuple) → (a = Dual, b = Dual) ✓
```

---

### Phase 3: ReConstructor Constructor (1 hour)

**Location**: `src/helpers.jl` - Continue after construction functions

```julia
"""
    ReConstructor(x)
    ReConstructor(flattendefault, x)

Construct a ReConstructor for parameter structure `x`.

Pre-computes all flatten/unflatten closures for efficient runtime operations.

# Arguments
- `flattendefault::FlattenDefault`: Configuration (optional, defaults to Float64, continuous)
- `x`: Parameter structure (NamedTuple, Tuple, Vector, or Real)

# Examples
```julia
# Simple example
params = (shape = 1.5, scale = 0.2)
reconstructor = ReConstructor(params)

# Nested example
params = (
    h12 = (baseline = (shape = 1.5, scale = 0.2), covariates = (age = 0.3,)),
    h23 = (baseline = (intercept = 0.8,),)
)
reconstructor = ReConstructor(params)

# Custom configuration
config = FlattenDefault(output=Float32, flattentype=FlattenAll())
reconstructor = ReConstructor(config, params)
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

# Convenience constructor with defaults
function ReConstructor(x)
    return ReConstructor(FlattenDefault(), x)
end
```

---

### Phase 4: User-Facing API (1 hour)

**Location**: `src/helpers.jl` - Continue after ReConstructor constructor

```julia
"""
    flatten(reconstructor::ReConstructor, x)

Flatten parameter structure `x` to a vector using pre-computed closure.

Uses strict flatten - always returns configured output type (default Float64).

# Examples
```julia
params = (shape = 1.5, scale = 0.2)
reconstructor = ReConstructor(params)
flat = flatten(reconstructor, params)  # Vector{Float64}: [1.5, 0.2]
```
"""
function flatten(reconstructor::ReConstructor, x)
    return reconstructor.flatten_strict(x)
end

"""
    flattenAD(reconstructor::ReConstructor, x)

Flatten parameter structure `x` to a vector, preserving input type for AD.

Uses flexible flatten - adapts output type to input (Dual → Dual, Float64 → Float64).

# Examples
```julia
params = (shape = Dual(1.5, 1.0), scale = Dual(0.2, 1.0))
reconstructor = ReConstructor(params)
flat = flattenAD(reconstructor, params)  # Vector{Dual}: [Dual(1.5, 1.0), Dual(0.2, 1.0)]
```
"""
function flattenAD(reconstructor::ReConstructor, x)
    return reconstructor.flatten_flexible(x)
end

"""
    unflatten(reconstructor::ReConstructor, v::AbstractVector)

Unflatten vector `v` to parameter structure using pre-computed closure.

Uses strict unflatten - always returns original types (type-stable).

# Examples
```julia
params = (shape = 1.5, scale = 0.2)
reconstructor = ReConstructor(params)
flat = flatten(reconstructor, params)
recon = unflatten(reconstructor, flat)  # (shape = 1.5, scale = 0.2)
```
"""
function unflatten(reconstructor::ReConstructor, v::AbstractVector)
    return reconstructor.unflatten_strict(v)
end

"""
    unflattenAD(reconstructor::ReConstructor, v::AbstractVector)

Unflatten vector `v` to parameter structure, preserving input type for AD.

Uses flexible unflatten - adapts to input type (Dual → Dual, Float64 → Float64).
This is CRITICAL for automatic differentiation to work correctly!

# Examples
```julia
params = (shape = 1.5, scale = 0.2)
reconstructor = ReConstructor(params)

# Regular use with Float64
flat = [1.5, 0.2]
recon = unflattenAD(reconstructor, flat)  # (shape = 1.5, scale = 0.2)

# AD use with Dual
using ForwardDiff
flat_dual = [Dual(1.5, 1.0), Dual(0.2, 1.0)]
recon_dual = unflattenAD(reconstructor, flat_dual)  # (shape = Dual(...), scale = Dual(...))
```
"""
function unflattenAD(reconstructor::ReConstructor, v::AbstractVector)
    return reconstructor.unflatten_flexible(v)
end
```

---

### Phase 5: Integration with Existing Code (2-3 hours)

#### Task 5.1: Replace safe_unflatten (DELETE 45 lines)

**Location**: `src/helpers.jl` lines 1-45

**Action**: DELETE entire `safe_unflatten` function

```julia
# DELETE THESE LINES (1-45):
"""
    safe_unflatten(unflatten::Function, θ::AbstractVector{T}) where T

[... entire docstring ...]
"""
function safe_unflatten(unflatten::Function, θ::AbstractVector{T}) where T
    [... entire 45-line implementation ...]
end
```

**Rationale**: `unflattenAD` replaces this completely with cleaner implementation.

#### Task 5.2: Update rebuild_parameters

**Location**: `src/helpers.jl` line ~50

**Current**:
```julia
function rebuild_parameters(
    model::AbstractMultistateModel,
    newvalues::Vector{Vector{T}}
) where {T<:Real}
    
    # Unflatten to nested structure
    updated_nested = model.parameters.unflatten(vcat(newvalues...))
    
    # Build natural scale parameters
    updated_natural = Dict{Symbol,Any}()
    for (hazname, hazind) in model.hazkeys
        hazard = model.hazards[hazind]
        hazard_pars = build_hazard_params(
            newvalues[hazind],
            model.hazards[hazind].npar_baseline,
            model.hazards[hazind].npar_total
        )
        # ... rest ...
    end
    
    # Return updated
    return (
        flat = vcat(newvalues...),
        nested = updated_nested,
        natural = NamedTuple(updated_natural),
        unflatten = model.parameters.unflatten
    )
end
```

**Updated**:
```julia
function rebuild_parameters(
    model::AbstractMultistateModel,
    newvalues::Vector{Vector{T}}
) where {T<:Real}
    
    # Unflatten using ReConstructor (replaces manual unflatten)
    flat_vec = vcat(newvalues...)
    updated_nested = unflatten(model.parameters.reconstructor, flat_vec)
    
    # Build natural scale parameters
    updated_natural = Dict{Symbol,Any}()
    for (hazname, hazind) in model.hazkeys
        hazard = model.hazards[hazind]
        hazard_pars = build_hazard_params(
            newvalues[hazind],
            hazard.parnames,                    # Already updated in Phase 1
            model.hazards[hazind].npar_baseline,
            model.hazards[hazind].npar_total
        )
        # ... rest unchanged ...
    end
    
    # Return updated (store reconstructor instead of unflatten)
    return (
        flat = flat_vec,
        nested = updated_nested,
        natural = NamedTuple(updated_natural),
        reconstructor = model.parameters.reconstructor  # NEW: store ReConstructor
    )
end
```

**Changes**:
- Replace `model.parameters.unflatten(...)` with `unflatten(model.parameters.reconstructor, ...)`
- Store `reconstructor` instead of `unflatten` function
- Use existing `hazard.parnames` (added in Phase 1)

#### Task 5.3: Update prepare_parameters

**Location**: `src/likelihoods.jl` line ~35-45

**Current**:
```julia
function prepare_parameters(
    θ::AbstractVector{T},
    model::AbstractMultistateModel
) where {T<:Real}
    
    # Unflatten to nested structure
    parameters = safe_unflatten(model.parameters.unflatten, θ)
    
    return parameters  # Already fixed to return NamedTuple directly
end
```

**Updated**:
```julia
function prepare_parameters(
    θ::AbstractVector{T},
    model::AbstractMultistateModel
) where {T<:Real}
    
    # Unflatten using AD-compatible unflattenAD
    parameters = unflattenAD(model.parameters.reconstructor, θ)
    
    return parameters  # NamedTuple with correct types (Float64 or Dual)
end
```

**Changes**:
- Replace `safe_unflatten(...)` with `unflattenAD(...)`
- Use `model.parameters.reconstructor` instead of `unflatten`
- **Critical**: `unflattenAD` automatically handles Dual types!

#### Task 5.4: Update Model Generation

**Location**: `src/modelgeneration.jl` line ~750-780

**Current** (approximate):
```julia
# Build parameters
initial_nested = NamedTuple{Tuple(keys(model.hazards))}(...)
flat, unflatten = ParameterHandling.flatten(initial_nested)

model_parameters = (
    flat = flat,
    nested = initial_nested,
    natural = ...,
    unflatten = unflatten
)
```

**Updated**:
```julia
# Build parameters
initial_nested = NamedTuple{Tuple(keys(model.hazards))}(...)

# Create ReConstructor (replaces ParameterHandling.flatten)
reconstructor = ReConstructor(initial_nested)
flat = flatten(reconstructor, initial_nested)

model_parameters = (
    flat = flat,
    nested = initial_nested,
    natural = ...,
    reconstructor = reconstructor  # Store ReConstructor instead of unflatten
)
```

**Changes**:
- Replace `ParameterHandling.flatten(...)` with `ReConstructor(...)`
- Use `flatten(reconstructor, ...)` to get flat vector
- Store `reconstructor` instead of `unflatten` function

---

### Phase 6: Testing (1-2 hours)

#### Test 1: Basic Functionality

**File**: `test/test_helpers.jl`

```julia
@testset "ReConstructor - Basic functionality" begin
    # Test scalar
    recon = ReConstructor(1.5)
    flat = flatten(recon, 1.5)
    @test flat == [1.5]
    @test unflatten(recon, flat) ≈ 1.5
    
    # Test vector
    vec_param = [1.0, 2.0, 3.0]
    recon = ReConstructor(vec_param)
    flat = flatten(recon, vec_param)
    @test flat == vec_param
    @test unflatten(recon, flat) ≈ vec_param
    
    # Test NamedTuple
    nt_param = (shape = 1.5, scale = 0.2)
    recon = ReConstructor(nt_param)
    flat = flatten(recon, nt_param)
    @test length(flat) == 2
    reconstructed = unflatten(recon, flat)
    @test reconstructed.shape ≈ 1.5
    @test reconstructed.scale ≈ 0.2
    
    # Test nested NamedTuple
    nested_param = (
        h12 = (baseline = (shape = 1.5, scale = 0.2), covariates = (age = 0.3,)),
        h23 = (baseline = (intercept = 0.8,),)
    )
    recon = ReConstructor(nested_param)
    flat = flatten(recon, nested_param)
    @test length(flat) == 4  # 1.5, 0.2, 0.3, 0.8
    reconstructed = unflatten(recon, flat)
    @test reconstructed.h12.baseline.shape ≈ 1.5
    @test reconstructed.h12.baseline.scale ≈ 0.2
    @test reconstructed.h12.covariates.age ≈ 0.3
    @test reconstructed.h23.baseline.intercept ≈ 0.8
end
```

#### Test 2: AD Compatibility (CRITICAL)

```julia
@testset "ReConstructor - AD compatibility" begin
    using ForwardDiff
    
    # Test unflattenAD preserves Dual types
    params = (shape = 1.5, scale = 0.2)
    recon = ReConstructor(params)
    
    # Create Dual vector
    flat = flatten(recon, params)
    dual_vec = ForwardDiff.Dual{Nothing,Float64,2}.(flat, Ref((1.0, 0.0)))
    
    # unflattenAD should preserve Dual types
    recon_dual = unflattenAD(recon, dual_vec)
    @test recon_dual.shape isa ForwardDiff.Dual
    @test recon_dual.scale isa ForwardDiff.Dual
    @test ForwardDiff.value(recon_dual.shape) ≈ 1.5
    @test ForwardDiff.value(recon_dual.scale) ≈ 0.2
    
    # Test gradient computation works
    function test_func(θ)
        params = unflattenAD(recon, θ)
        return params.shape^2 + params.scale^2
    end
    
    grad = ForwardDiff.gradient(test_func, flat)
    @test grad[1] ≈ 2 * 1.5  # d/d(shape) = 2*shape
    @test grad[2] ≈ 2 * 0.2  # d/d(scale) = 2*scale
    
    # Test nested structure with AD
    nested = (
        h12 = (baseline = (shape = 1.5, scale = 0.2),),
        h23 = (baseline = (intercept = 0.8,),)
    )
    recon_nested = ReConstructor(nested)
    flat_nested = flatten(recon_nested, nested)
    dual_nested = ForwardDiff.Dual{Nothing,Float64,3}.(
        flat_nested, 
        Ref((1.0, 0.0, 0.0))
    )
    
    recon_nested_dual = unflattenAD(recon_nested, dual_nested)
    @test recon_nested_dual.h12.baseline.shape isa ForwardDiff.Dual
    @test recon_nested_dual.h12.baseline.scale isa ForwardDiff.Dual
    @test recon_nested_dual.h23.baseline.intercept isa ForwardDiff.Dual
end
```

#### Test 3: Performance Benchmarks

```julia
@testset "ReConstructor - Performance" begin
    using BenchmarkTools
    
    # Large nested structure
    large_params = (
        h12 = (baseline = (shape = 1.5, scale = 0.2), covariates = (age = 0.3, sex = 0.1)),
        h13 = (baseline = (shape = 1.8, scale = 0.15), covariates = (age = 0.25,)),
        h23 = (baseline = (intercept = 0.8,),),
        h24 = (baseline = (shape = 1.2, scale = 0.3), covariates = (age = 0.2, sex = 0.15, bmi = 0.05))
    )
    
    recon = ReConstructor(large_params)
    flat = flatten(recon, large_params)
    
    # Benchmark unflatten (should be very fast, minimal allocations)
    result = @benchmark unflatten($recon, $flat)
    @test result.allocs < 50  # Should be very few allocations
    @test median(result).time < 1000  # Should be under 1 microsecond
    
    # Benchmark unflattenAD (may have slightly more allocations but still fast)
    result_ad = @benchmark unflattenAD($recon, $flat)
    @test result_ad.allocs < 100
    @test median(result_ad).time < 2000  # Should be under 2 microseconds
    
    println("Unflatten performance: ", median(result).time, " ns, ", result.allocs, " allocations")
    println("UnflattenAD performance: ", median(result_ad).time, " ns, ", result_ad.allocs, " allocations")
end
```

#### Test 4: Integration with Model

```julia
@testset "ReConstructor - Model integration" begin
    # Build simple model
    tmat = [0 1; 0 0]
    dat = DataFrame(id=1:10, tstart=0.0, tstop=rand(10), age=rand(10))
    
    model = multistatemodel(
        @formula(time ~ age),
        dat,
        tmat,
        hazards = [Weibull(:PH)]
    )
    
    # Check model has reconstructor
    @test hasfield(typeof(model.parameters), :reconstructor)
    @test model.parameters.reconstructor isa ReConstructor
    
    # Test parameter reconstruction
    flat = model.parameters.flat
    recon = unflatten(model.parameters.reconstructor, flat)
    @test recon isa NamedTuple
    @test haskey(recon, :h12)
    
    # Test AD compatibility in likelihood
    function test_loglik(θ)
        params = unflattenAD(model.parameters.reconstructor, θ)
        # Dummy computation
        return sum(sum(values(values(params)[1])))
    end
    
    grad = ForwardDiff.gradient(test_loglik, flat)
    @test !any(isnan, grad)
    @test length(grad) == length(flat)
end
```

---

## Migration Checklist

### Pre-Implementation
- [ ] Review ModelWrappers.jl source code understanding
- [ ] Confirm pattern matches current needs
- [ ] Backup current code

### Phase 1: Core Types (1 hour)
- [ ] Add FlattenTypes definitions
- [ ] Add UnflattenTypes definitions  
- [ ] Add FlattenDefault struct
- [ ] Add ReConstructor struct
- [ ] Run basic type tests

### Phase 2: Construction Functions (2-3 hours)
- [ ] Implement construct_flatten for Real (strict & flexible)
- [ ] Implement construct_flatten for Vector (strict & flexible)
- [ ] Handle integer vector skipping
- [ ] Implement construct_flatten for Tuple (recursive)
- [ ] Implement construct_flatten for NamedTuple (strict & flexible)
- [ ] Test each constructor independently

### Phase 3: ReConstructor Constructor (1 hour)
- [ ] Implement ReConstructor(flattendefault, x)
- [ ] Implement ReConstructor(x) convenience method
- [ ] Test ReConstructor creation for various structures

### Phase 4: User API (1 hour)
- [ ] Implement flatten(reconstructor, x)
- [ ] Implement flattenAD(reconstructor, x)
- [ ] Implement unflatten(reconstructor, v)
- [ ] Implement unflattenAD(reconstructor, v)
- [ ] Add comprehensive docstrings

### Phase 5: Integration (2-3 hours)
- [ ] DELETE safe_unflatten function (45 lines)
- [ ] Update rebuild_parameters to use ReConstructor
- [ ] Update prepare_parameters to use unflattenAD
- [ ] Update model generation to create ReConstructor
- [ ] Update parameter storage structure (reconstructor field)

### Phase 6: Testing (1-2 hours)
- [ ] Test basic flatten/unflatten
- [ ] Test AD compatibility with ForwardDiff
- [ ] Test performance benchmarks
- [ ] Test integration with model building
- [ ] Run full test suite
- [ ] Verify no regressions

### Validation
- [ ] All existing tests pass
- [ ] New AD tests pass
- [ ] Performance acceptable (<2x overhead)
- [ ] Code is cleaner (net -30 lines)

---

## Expected Outcomes

### Code Changes Summary

| File | Lines Added | Lines Removed | Net Change |
|------|------------|---------------|------------|
| `src/helpers.jl` | ~300 | ~45 (safe_unflatten) | +255 |
| `src/likelihoods.jl` | ~3 | ~3 | 0 |
| `src/modelgeneration.jl` | ~5 | ~3 | +2 |
| `test/test_helpers.jl` | ~120 | 0 | +120 |
| **Total** | **~428** | **~51** | **+377** |

### Performance Targets

- `unflatten`: < 1 µs for typical structures, < 50 allocations
- `unflattenAD`: < 2 µs for typical structures, < 100 allocations
- Total overhead in likelihood: < 5%

### Benefits

1. **Eliminates dependency conflicts** - No need for ModelWrappers.jl package
2. **Cleaner code** - Remove 45-line safe_unflatten workaround
3. **Better performance** - Pre-allocated buffers, zero-allocation possible
4. **AD compatibility** - Automatic Dual type handling
5. **Future-proof** - Own implementation, easy to extend

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Implementation bugs | Medium | High | Extensive testing, compare with ParameterHandling.jl |
| Performance regression | Low | Medium | Benchmark at each phase, optimize if needed |
| AD edge cases | Medium | High | Comprehensive ForwardDiff tests |
| Integration issues | Low | Medium | Update one function at a time, test incrementally |

---

## Rollback Plan

If implementation encounters major issues:

1. **Git revert** to commit before implementation started
2. **Keep safe_unflatten** - it works, just complex
3. **Document findings** - what went wrong, what needs fixing
4. **Revisit later** - when more time available or after SciML updates

---

## Next Steps

1. **Review this plan** - Confirm approach makes sense
2. **Begin Phase 1** - Start with type definitions
3. **Test incrementally** - Don't proceed if tests fail
4. **Iterate as needed** - Adjust plan based on findings

**Estimated Total Time**: 6-8 hours

**Recommended Schedule**:
- Session 1 (2-3 hours): Phases 1-2 (types + basic constructors)
- Session 2 (2-3 hours): Phases 3-4 (ReConstructor + API)
- Session 3 (2-3 hours): Phases 5-6 (integration + testing)
