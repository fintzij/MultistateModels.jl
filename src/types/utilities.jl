# =============================================================================
# Utility Type Definitions
# =============================================================================
# Types for parameter flattening/unflattening, reconstructors, and other utilities.
# =============================================================================

# =============================================================================
# Flatten/Unflatten Types
# =============================================================================

"""
Abstract type for flatten mode selection.
"""
abstract type FlattenTypes end

"""
    FlattenContinuous <: FlattenTypes

Flatten only continuous parameters (default behavior).
"""
struct FlattenContinuous <: FlattenTypes end

"""
    FlattenAll <: FlattenTypes

Flatten all parameters including integers.
"""
struct FlattenAll <: FlattenTypes end

"""
Abstract type for unflatten mode selection.
"""
abstract type UnflattenTypes end

"""
    UnflattenStrict <: UnflattenTypes

Type-stable unflatten that converts to original types.
Use for standard evaluation (non-AD contexts).
"""
struct UnflattenStrict <: UnflattenTypes end

"""
    UnflattenFlexible <: UnflattenTypes

Type-polymorphic unflatten that preserves input types.
Use for automatic differentiation to preserve Dual types.
"""
struct UnflattenFlexible <: UnflattenTypes end

"""
    FlattenDefault{F<:FlattenTypes, U<:UnflattenTypes}

Default settings for flatten/unflatten operations.

# Fields
- `flattentype::F`: Controls which parameters to flatten
- `unflattentype::U`: Controls type handling during unflatten
"""
struct FlattenDefault{F<:FlattenTypes, U<:UnflattenTypes}
    flattentype::F
    unflattentype::U
end

# Convenience constructors
FlattenDefault() = FlattenDefault(FlattenContinuous(), UnflattenStrict())
FlattenDefault(unflattentype::UnflattenTypes) = FlattenDefault(FlattenContinuous(), unflattentype)

"""
    ReConstructor{F,S,T,U,V}

Pre-computed flatten/unflatten closures with buffers for efficient parameter operations.

Stores four function variants:
- `flatten_strict`: Type-stable flatten (returns Vector{Float64})
- `flatten_flexible`: AD-compatible flatten (returns Vector{T})
- `unflatten_strict`: Type-stable unflatten (converts to original types)
- `unflatten_flexible`: Type-polymorphic unflatten (preserves Dual types for AD)

# Usage
```julia
# Create reconstructor
rc = ReConstructor(params)

# Standard usage (fast)
flat = flatten(rc, params)
reconstructed = unflatten(rc, flat)

# AD usage (preserves Dual types)
flat_dual = flattenAD(rc, params_dual)
reconstructed_dual = unflattenAD(rc, flat_dual)
```
"""
struct ReConstructor{F,S,T,U,V}
    default::FlattenDefault
    flatten_strict::F
    flatten_flexible::S
    unflatten_strict::T
    unflatten_flexible::U
    _buffer::V  # Pre-allocated buffer for intermediate operations
end
