# ============================================================================
# Parameter Flattening Type System and Construction Functions
# ============================================================================
# Provides type aliases and factory functions for building flatten/unflatten
# closures. Used by ReConstructor for AD-compatible parameter handling.
# ============================================================================

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

# ============================================================================
# Construction functions for different types
# ============================================================================

"""
    construct_flatten(output::Type{T}, flattentype::FlattenTypes, 
                     unflattentype::UnflattenTypes, x::Real)

Build flatten/unflatten closures for Real numbers.

Returns tuple of (flatten_function, unflatten_function).
"""
function construct_flatten(
    output::Type{T},
    flattentype::FlattenTypes,
    unflattentype::UnflattenTypes,
    x::Real
) where {T<:Real}
    
    # Flatten: Real → Vector{T}
    function flatten_to_Real(val::S) where {S<:Real}
        return T[val]
    end
    
    # Unflatten variants - use different names to avoid method overwriting
    if unflattentype isa UnflattenStrict
        # Strict: convert to original type (type-stable, breaks AD)
        unflatten_to_Real_strict(v::AbstractVector{S}) where {S<:Real} = convert(typeof(x), only(v))
        return flatten_to_Real, unflatten_to_Real_strict
    else  # UnflattenFlexible
        # Flexible: preserve input type (allows Dual types for AD)
        unflatten_to_Real_flexible(v::AbstractVector{S}) where {S<:Real} = only(v)
        return flatten_to_Real, unflatten_to_Real_flexible
    end
end

"""
    construct_flatten(output::Type{T}, flattentype::FlattenTypes,
                     unflattentype::UnflattenTypes, x::AbstractVector{<:Real})

Build flatten/unflatten closures for Vector of Real numbers.

Handles both FlattenContinuous and FlattenAll modes.
"""
function construct_flatten(
    output::Type{T},
    flattentype::FlattenTypes,
    unflattentype::UnflattenTypes,
    x::AbstractVector{<:Real}
) where {T<:Real}
    
    n = length(x)
    
    # Check if all elements are continuous (Float) or if we have integers
    is_continuous = all(xi -> xi isa AbstractFloat, x)
    
    # For FlattenContinuous mode with integer vectors, skip flattening
    if flattentype isa FlattenContinuous && !is_continuous
        # Return identity functions - don't flatten integer vectors
        flatten_Vector_skip(val) = T[]
        unflatten_Vector_skip(v) = x  # Return original vector
        return flatten_Vector_skip, unflatten_Vector_skip
    end
    
    # Flatten: Vector → concatenated Vector{T}
    function flatten_Vector(val::AbstractVector{S}) where {S<:Real}
        return convert(Vector{T}, val)
    end
    
    # Unflatten variants
    if unflattentype isa UnflattenStrict
        # Strict: convert to original element type (type-stable)
        function unflatten_Vector_strict(v::AbstractVector{S}) where {S<:Real}
            return convert(typeof(x), v[1:n])
        end
        return flatten_Vector, unflatten_Vector_strict
    else  # UnflattenFlexible
        # Flexible: preserve input element types (allows Dual for AD)
        function unflatten_Vector_flexible(v::AbstractVector{S}) where {S<:Real}
            # Use view to avoid allocation, return as vector of input type
            return collect(v[1:n])
        end
        return flatten_Vector, unflatten_Vector_flexible
    end
end

"""
    construct_flatten(output::Type{T}, flattentype::FlattenTypes,
                     unflattentype::UnflattenTypes, x::Tuple)

Build flatten/unflatten closures for Tuples (recursive construction).

This is the core of the recursive algorithm:
1. Build constructors for each element recursively
2. Flatten once to determine sizes
3. Compute cumulative sizes for indexing
4. Return composed closures that handle the full structure
"""
function construct_flatten(
    output::Type{T},
    flattentype::FlattenTypes,
    unflattentype::UnflattenTypes,
    x::Tuple
) where {T<:Real}
    
    # Step 1: Recursively build constructors for each element
    x_constructors = map(xᵢ -> construct_flatten(T, flattentype, unflattentype, xᵢ), x)
    _flatten = first.(x_constructors)
    _unflatten = last.(x_constructors)
    
    # Step 2: Flatten once to determine sizes
    x_vecs = map((flat, xᵢ) -> flat(xᵢ), _flatten, x)
    lengths = map(length, x_vecs)
    cumulative_sizes = cumsum(lengths)
    
    # Step 3: Build composed flatten function
    function flatten_Tuple(val::Tuple)
        mapped = map((flat, xᵢ) -> flat(xᵢ), _flatten, val)
        return isempty(mapped) ? T[] : reduce(vcat, mapped)
    end
    
    # Step 4: Build composed unflatten with proper indexing
    function unflatten_Tuple(v::AbstractVector{S}) where {S<:Real}
        return map(_unflatten, lengths, cumulative_sizes) do unflat, len, cumsize
            start_idx = cumsize - len + 1
            if len == 0
                # Empty vector case (e.g., integer vectors in FlattenContinuous mode)
                return unflat(S[])
            else
                return unflat(view(v, start_idx:cumsize))
            end
        end
    end
    
    return flatten_Tuple, unflatten_Tuple
end

"""
    construct_flatten(output::Type{T}, flattentype::FlattenTypes,
                     unflattentype::UnflattenTypes, x::NamedTuple)

Build flatten/unflatten closures for NamedTuples (recursive construction).

Key difference for AD compatibility:
- Strict mode: `typeof(x)(tuple)` - requires concrete types, breaks with Dual
- Flexible mode: `NamedTuple{names}(tuple)` - accepts any types, preserves Dual
"""
function construct_flatten(
    output::Type{T},
    flattentype::FlattenTypes,
    unflattentype::UnflattenTypes,
    x::NamedTuple
) where {T<:Real}
    
    names = keys(x)
    values_tuple = values(x)
    
    # Build constructor for the underlying tuple (recursive)
    flatten_tuple, unflatten_tuple = construct_flatten(T, flattentype, unflattentype, values_tuple)
    
    # Flatten: just use tuple flatten
    flatten_NamedTuple(val::NamedTuple) = flatten_tuple(values(val))
    
    # Unflatten: reconstruct NamedTuple with appropriate type handling
    if unflattentype isa UnflattenStrict
        # Strict: Use typed constructor (type-stable, requires concrete types)
        function unflatten_NamedTuple_strict(v::AbstractVector{S}) where {S<:Real}
            v_tuple = unflatten_tuple(v)
            return typeof(x)(v_tuple)  # Requires concrete types - breaks with Dual
        end
        return flatten_NamedTuple, unflatten_NamedTuple_strict
    else  # UnflattenFlexible
        # Flexible: Use generic constructor (preserves any types including Dual)
        function unflatten_NamedTuple_flexible(v::AbstractVector{S}) where {S<:Real}
            v_tuple = unflatten_tuple(v)
            return NamedTuple{names}(v_tuple)  # Generic - works with Dual!
        end
        return flatten_NamedTuple, unflatten_NamedTuple_flexible
    end
end
