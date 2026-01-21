# ============================================================================
# ReConstructor: AD-compatible parameter flattening/unflattening
# ============================================================================
# Based on ModelWrappers.jl pattern - manual implementation to avoid dependencies
#
# Provides:
# - ReConstructor struct with pre-computed flatten/unflatten closures
# - flatten/unflatten for standard usage (type-stable)
# - flattenAD/unflattenAD for AD usage (type-polymorphic, preserves Dual)
# ============================================================================

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
struct ReConstructor{F,S,T,U}
    default::FlattenDefault
    flatten_strict::F
    flatten_flexible::S
    unflatten_strict::T
    unflatten_flexible::U
end

"""
    ReConstructor(x; flattentype=FlattenContinuous(), unflattentype=UnflattenStrict())

Build a ReConstructor for the given parameter structure.

# Arguments
- `x`: Parameter structure (NamedTuple, Tuple, Vector, or Real)
- `flattentype`: FlattenContinuous() (default)
- `unflattentype`: UnflattenStrict() or UnflattenFlexible()

# Returns
ReConstructor with pre-computed flatten/unflatten closures

# Example
```julia
params = (baseline = (shape = 1.5, scale = 0.2), covariates = (age = 0.3,))
rc = ReConstructor(params, unflattentype=UnflattenFlexible())

# Standard usage
flat = flatten(rc, params)

# AD usage (preserves Dual types)
using ForwardDiff
f(p) = sum(unflattenAD(rc, p).baseline)
grad = ForwardDiff.gradient(f, flat)
```
"""
function ReConstructor(
    x;
    flattentype::FlattenTypes = FlattenContinuous(),
    unflattentype::UnflattenTypes = UnflattenStrict()
)
    default = FlattenDefault(flattentype, unflattentype)
    
    # Build strict constructors (for standard usage)
    flatten_strict_fn, unflatten_strict_fn = construct_flatten(Float64, flattentype, UnflattenStrict(), x)
    
    # Build flexible constructors (for AD usage)
    flatten_flexible_fn, unflatten_flexible_fn = construct_flatten(Float64, flattentype, UnflattenFlexible(), x)
    
    return ReConstructor(
        default,
        flatten_strict_fn,
        flatten_flexible_fn,
        unflatten_strict_fn,
        unflatten_flexible_fn
    )
end

"""
    flatten(rc::ReConstructor, x)

Flatten parameter structure to vector (type-stable, returns Vector{Float64}).

Use for standard parameter operations.
"""
flatten(rc::ReConstructor, x) = rc.flatten_strict(x)

"""
    unflatten(rc::ReConstructor, v::AbstractVector)

Unflatten vector to parameter structure (type-stable).

Use for standard parameter operations.
"""
unflatten(rc::ReConstructor, v::AbstractVector) = rc.unflatten_strict(v)

"""
    flattenAD(rc::ReConstructor, x)

Flatten parameter structure to vector (type-polymorphic).

Use when working with AD types (preserves Dual numbers).
"""
flattenAD(rc::ReConstructor, x) = rc.flatten_flexible(x)

"""
    unflattenAD(rc::ReConstructor, v::AbstractVector)

Unflatten vector to parameter structure (type-polymorphic).

Use for automatic differentiation - preserves Dual types.
"""
unflattenAD(rc::ReConstructor, v::AbstractVector) = rc.unflatten_flexible(v)
