# =============================================================================
# Transition Enumeration and Validation Utilities
# =============================================================================
#
# Functions for enumerating transitions from a transition matrix and validating
# per-transition observation type specifications.
#
# =============================================================================

# =============================================================================
# Transition Enumeration
# =============================================================================

"""
    enumerate_transitions(tmat::AbstractMatrix) -> Vector{Tuple{Int,Int}}

Enumerate all transitions from the transition matrix in row-major order.

Returns a vector of `(statefrom, stateto)` tuples for each non-zero entry
in `tmat`. The order is row-major: transitions from state 1 first, then
state 2, etc.

# Arguments
- `tmat::AbstractMatrix`: Transition matrix where non-zero entries indicate allowed transitions

# Returns
- `Vector{Tuple{Int,Int}}`: Vector of `(statefrom, stateto)` pairs

# Example
```julia
tmat = [0 1 2;
        0 0 3;
        0 0 0]

transitions = enumerate_transitions(tmat)
# Returns: [(1, 2), (1, 3), (2, 3)]
```

See also: [`transition_index_map`](@ref), [`print_transition_map`](@ref)
"""
function enumerate_transitions(tmat::AbstractMatrix)
    transitions = Tuple{Int,Int}[]
    n_states = size(tmat, 1)
    for i in 1:n_states
        for j in 1:n_states
            if tmat[i, j] != 0
                push!(transitions, (i, j))
            end
        end
    end
    return transitions
end

"""
    transition_index_map(tmat::AbstractMatrix) -> Dict{Tuple{Int,Int},Int}

Create a mapping from `(statefrom, stateto)` to transition index (1-based).

The transition index corresponds to the position in [`enumerate_transitions`](@ref).

# Arguments
- `tmat::AbstractMatrix`: Transition matrix where non-zero entries indicate allowed transitions

# Returns
- `Dict{Tuple{Int,Int},Int}`: Mapping from state pairs to transition indices

# Example
```julia
tmat = [0 1 2;
        0 0 3;
        0 0 0]

trans_map = transition_index_map(tmat)
# Returns: Dict((1,2) => 1, (1,3) => 2, (2,3) => 3)
```

See also: [`enumerate_transitions`](@ref), [`print_transition_map`](@ref)
"""
function transition_index_map(tmat::AbstractMatrix)
    transitions = enumerate_transitions(tmat)
    return Dict(trans => idx for (idx, trans) in enumerate(transitions))
end

"""
    transition_index_map(model::MultistateProcess) -> Dict{Tuple{Int,Int},Int}

Create transition index map from model's transition matrix.

Convenience method that extracts `tmat` from the model.

See also: [`enumerate_transitions`](@ref), [`print_transition_map`](@ref)
"""
transition_index_map(model::MultistateProcess) = transition_index_map(model.tmat)

"""
    print_transition_map(model::MultistateProcess)
    print_transition_map(tmat::AbstractMatrix)
    print_transition_map(io::IO, model::MultistateProcess)
    print_transition_map(io::IO, tmat::AbstractMatrix)

Print a human-readable table of transition indices for user reference.

Useful when specifying `obstype_by_transition` dictionaries for simulation.

# Example
```julia
model = multistatemodel(...)
print_transition_map(model)

# Output:
# Transition Index Map:
# Index | From → To
# ------|----------
#     1 | 1 → 2
#     2 | 1 → 3
#     3 | 2 → 3
```

See also: [`enumerate_transitions`](@ref), [`transition_index_map`](@ref)
"""
function print_transition_map(io::IO, tmat::AbstractMatrix)
    transitions = enumerate_transitions(tmat)
    println(io, "Transition Index Map:")
    println(io, "Index | From → To")
    println(io, "------|----------")
    for (idx, (from, to)) in enumerate(transitions)
        println(io, lpad(idx, 5), " | ", from, " → ", to)
    end
end

print_transition_map(tmat::AbstractMatrix) = print_transition_map(stdout, tmat)
print_transition_map(io::IO, model::MultistateProcess) = print_transition_map(io, model.tmat)
print_transition_map(model::MultistateProcess) = print_transition_map(stdout, model.tmat)

# =============================================================================
# Validation Functions
# =============================================================================

"""
    validate_obstype_by_transition(obstype_dict::Dict{Int,Int}, n_transitions::Int)

Validate an `obstype_by_transition` dictionary.

Throws `ArgumentError` if:
- Any key is outside the valid range `[1, n_transitions]`
- Any value (obstype code) is less than 1

# Arguments
- `obstype_dict::Dict{Int,Int}`: Dictionary mapping transition index to obstype code
- `n_transitions::Int`: Total number of transitions in the model

# Observation Type Codes
- `1`: Exact observation (transition time and states fully observed)
- `2`: Panel observation (only endpoint state at interval boundary observed)
- `3+`: Censored observation (endpoint state unknown/missing)
"""
function validate_obstype_by_transition(obstype_dict::Dict{Int,Int}, n_transitions::Int)
    for (trans_idx, obscode) in obstype_dict
        if trans_idx < 1 || trans_idx > n_transitions
            throw(ArgumentError(
                "Transition index $trans_idx out of range [1, $n_transitions]. " *
                "Use print_transition_map(model) to see valid indices."
            ))
        end
        if obscode < 1
            throw(ArgumentError(
                "Observation type code must be >= 1, got $obscode for transition $trans_idx. " *
                "Valid codes: 1=exact, 2=panel, 3+=censored."
            ))
        end
    end
    return nothing
end

# Convenience method accepting tmat
function validate_obstype_by_transition(obstype_dict::Dict{Int,Int}, tmat::AbstractMatrix)
    n_transitions = length(enumerate_transitions(tmat))
    return validate_obstype_by_transition(obstype_dict, n_transitions)
end

"""
    validate_censoring_matrix(censoring_matrix::AbstractMatrix{Int}, n_transitions::Int, pattern::Union{Nothing,Int})

Validate a censoring matrix and pattern index.

Throws `ArgumentError` if:
- Number of rows doesn't equal `n_transitions`
- Pattern index is outside valid column range
- Any value in the matrix is less than 1

# Arguments
- `censoring_matrix::AbstractMatrix{Int}`: Matrix of size `(n_transitions, n_patterns)`
- `n_transitions::Int`: Expected number of rows (transitions)
- `pattern::Union{Nothing,Int}`: Column index to use (1-based), or nothing
"""
function validate_censoring_matrix(
    censoring_matrix::AbstractMatrix{Int},
    n_transitions::Int,
    pattern::Union{Nothing,Int}
)
    nrows, ncols = size(censoring_matrix)
    
    if nrows != n_transitions
        throw(ArgumentError(
            "Censoring matrix must have $n_transitions rows (one per transition), " *
            "got $nrows rows. Use print_transition_map(model) to see transitions."
        ))
    end
    
    if !isnothing(pattern)
        if pattern < 1 || pattern > ncols
            throw(ArgumentError(
                "Censoring pattern index $pattern out of range [1, $ncols]."
            ))
        end
    end
    
    # Validate all values are valid obstype codes
    if any(censoring_matrix .< 1)
        throw(ArgumentError(
            "All censoring matrix values must be >= 1 (valid obstype codes). " *
            "Valid codes: 1=exact, 2=panel, 3+=censored."
        ))
    end
    
    return nothing
end

# Convenience method accepting tmat
function validate_censoring_matrix(
    censoring_matrix::AbstractMatrix{Int},
    tmat::AbstractMatrix,
    pattern::Union{Nothing,Int} = nothing
)
    n_transitions = length(enumerate_transitions(tmat))
    return validate_censoring_matrix(censoring_matrix, n_transitions, pattern)
end

# =============================================================================
# Internal Helpers for observe_path
# =============================================================================

"""
    _get_transition_obstype(statefrom::Int, stateto::Int, trans_map::Dict{Tuple{Int,Int},Int},
                            obstype_by_transition::Union{Nothing,Dict{Int,Int}},
                            censoring_matrix::Union{Nothing,AbstractMatrix{Int}},
                            censoring_pattern::Union{Nothing,Int}) -> Int

Get the observation type for a specific transition.

Priority: obstype_by_transition > censoring_matrix > default (1)

Returns the obstype code (1=exact, 2=panel, 3+=censored).
"""
function _get_transition_obstype(
    statefrom::Int, 
    stateto::Int, 
    trans_map::Dict{Tuple{Int,Int},Int},
    obstype_by_transition::Union{Nothing,Dict{Int,Int}},
    censoring_matrix::Union{Nothing,AbstractMatrix{Int}},
    censoring_pattern::Union{Nothing,Int}
)
    trans_idx = get(trans_map, (statefrom, stateto), nothing)
    
    # If transition not in map (shouldn't happen), default to exact
    if isnothing(trans_idx)
        return 1
    end
    
    # Priority: obstype_by_transition > censoring_matrix > default (1)
    if !isnothing(obstype_by_transition) && haskey(obstype_by_transition, trans_idx)
        return obstype_by_transition[trans_idx]
    elseif !isnothing(censoring_matrix) && !isnothing(censoring_pattern)
        return censoring_matrix[trans_idx, censoring_pattern]
    else
        return 1  # Default to exact
    end
end
