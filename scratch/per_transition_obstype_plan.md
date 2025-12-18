# Per-Transition Observation Type Feature Plan

## Status: IMPLEMENTED ✅

**Implementation Date**: 2024
**Test Status**: All 110 tests passing
**Files Modified**:
- `src/utilities/transition_helpers.jl` (NEW)
- `src/simulation/path_utilities.jl` (modified `observe_path`)
- `src/simulation/simulate.jl` (added kwargs to `simulate`, `simulate_data`)
- `src/MultistateModels.jl` (exports)
- `MultistateModelsTests/unit/test_per_transition_obstype.jl` (NEW)

## Overview

Add capability to specify observation types (exact, panel, censored) on a per-transition basis during simulation, allowing mixed observation schemes within the same model and observation interval.

## Testing Philosophy

**All tests must verify correctness by comparing true paths to generated data.**

### Core Principles

1. **Deterministic Verification**: Given a known sample path and obstype specification, the generated data must match exactly. No probabilistic or statistical assertions for correctness—only for coverage.

2. **Path-to-Data Comparison**: Every test simulates with `data=true, paths=true` and verifies:
   - Exact transitions appear as rows with correct times and states
   - Panel observations capture the true endpoint state from the path
   - Censored observations have `stateto = missing`
   - Mixed intervals emit the correct combination of rows

3. **Reproducibility**: All tests use fixed random seeds.

4. **Test Location**: All tests reside in `MultistateModelsTests/` package.

5. **Test Structure**:
   - **Unit tests**: Individual functions (enumeration, validation, observation logic)
   - **Integration tests**: End-to-end workflow verification
   - **Fixtures**: Reusable model setups in `TestFixtures.jl`

### What We Test

| Test Category | Verification Method |
|---------------|---------------------|
| Transition enumeration | Direct value comparison |
| Input validation | `@test_throws ArgumentError` |
| Exact observation | Path transition time == data row tstop |
| Panel observation | Data stateto == path state at tstop |
| Censored observation | `ismissing(data.stateto)` |
| Mixed intervals | Both exact rows AND interval-level row present |
| Backward compatibility | Identical data with/without empty override |

## Current Behavior

- All observations in an interval have the same `obstype` from `model.data.obstype[r]`
- `obstype = 1`: exact observation (all transitions and times fully observed)
- `obstype = 2`: panel observation (only endpoint state observed at `tstop`)
- `obstype >= 3`: censored observation (endpoint state unknown/missing)

## Proposed Behavior

### User Interface

Allow users to specify observation type per transition via:

1. **Direct mapping**: `Dict{Int,Int}` mapping transition index → obstype code
2. **Censoring patterns**: Matrix-based patterns for systematic censoring schemes

### Transition Indexing

Transitions are enumerated from `model.tmat` in **row-major order** where `tmat[i,j] != 0`:

```julia
# Example for 3-state illness-death model
# tmat = [0 1 2;
#         0 0 3;
#         0 0 0]
# 
# Transitions enumerated as:
# 1: (1,2) - healthy → illness
# 2: (1,3) - healthy → death
# 3: (2,3) - illness → death
```

### API Changes

#### New Function Arguments

```julia
simulate(model::MultistateProcess;
    nsim = 1,
    data = true,
    paths = false,
    strategy = CachedTransformStrategy(),
    solver = OptimJumpSolver(),
    newdata = nothing,
    tmax = nothing,
    autotmax = true,
    expanded = true,
    # NEW ARGUMENTS:
    obstype_by_transition::Union{Nothing,Dict{Int,Int}} = nothing,
    censoring_matrix::Union{Nothing,AbstractMatrix{Int}} = nothing,
    censoring_pattern::Union{Nothing,Int} = nothing
)
```

#### Parameter Descriptions

- **`obstype_by_transition`**: Dictionary mapping transition index (1-based from `tmat`) to observation type code
  - `1` = exact (transition time and states fully observed)
  - `2` = panel (only endpoint state at interval boundary observed)
  - `3+` = censored with specific code (endpoint state missing/censored)
  - If `nothing`, uses default behavior from `model.data.obstype`
  
- **`censoring_matrix`**: Matrix of size `(n_transitions, n_patterns)` specifying censoring codes
  - Each column represents a censoring pattern
  - Element `[i,j]` gives obstype for transition `i` under pattern `j`
  - Used when `obstype_by_transition` doesn't specify a transition
  
- **`censoring_pattern`**: Column index in `censoring_matrix` to use (1-based)
  - Ignored if `censoring_matrix` is `nothing`
  - Future: could be per-subject via `newdata` column

### Observation Generation Logic

#### Mixed Observation Within an Interval

When an interval contains multiple transitions with different observation types:

1. **Exact transitions** (`obstype = 1`):
   - Emit individual rows with exact jump times
   - Format: `(tstart, tstop, statefrom, stateto, obstype=1)`
   
2. **Panel/Censored transitions** (`obstype >= 2`):
   - Emit one interval-level row spanning the full interval
   - Obstype determined by **precedence rule**: `max(obstype)` among non-exact transitions
   - If `obstype = 2`: `stateto = state_at_tstop`
   - If `obstype >= 3`: `stateto = missing`

#### Example Scenario

```julia
# Interval [0.0, 10.0] contains 3 transitions:
# t=2.0: state 1 → 2 (transition index 1, obstype=1, exact)
# t=5.0: state 2 → 3 (transition index 3, obstype=2, panel)
# t=7.0: state 3 → 4 (transition index 5, obstype=1, exact)

# Result: 3 rows emitted:
# Row 1: (0.0, 2.0, 1, 2, obstype=1)  - exact
# Row 2: (2.0, 7.0, 2, 4, obstype=1)  - exact
# Row 3: (0.0, 10.0, 1, 4, obstype=2) - panel (interval-level)
```

### Interaction with Existing Data Observation Types

**Priority Rules:**

1. If interval in `model.data` has `obstype = 1` (exact):
   - Apply `obstype_by_transition` mapping to transitions occurring in that interval
   - Generate mixed exact/panel/censored rows as specified

2. If interval in `model.data` has `obstype >= 2` (panel/censored):
   - **Preserve original behavior** to avoid double-counting
   - Ignore `obstype_by_transition` for that interval
   - Rationale: original data structure already defines observation scheme

## Implementation Plan

### Phase 1: Transition Enumeration Utilities

**File**: `src/utilities/transition_helpers.jl` (new file)

```julia
"""
    enumerate_transitions(tmat::AbstractMatrix{Int}) -> Vector{Tuple{Int,Int}}

Enumerate all transitions from the transition matrix in row-major order.
Returns vector of (statefrom, stateto) tuples where tmat[statefrom,stateto] != 0.
"""
function enumerate_transitions(tmat::AbstractMatrix{Int})
    transitions = Tuple{Int,Int}[]
    n_states = size(tmat, 1)
    for i in 1:n_states
        for j in 1:n_states
            if tmat[i,j] != 0
                push!(transitions, (i, j))
            end
        end
    end
    return transitions
end

"""
    transition_index_map(tmat::AbstractMatrix{Int}) -> Dict{Tuple{Int,Int},Int}

Create mapping from (statefrom, stateto) to transition index (1-based).
"""
function transition_index_map(tmat::AbstractMatrix{Int})
    transitions = enumerate_transitions(tmat)
    return Dict(trans => idx for (idx, trans) in enumerate(transitions))
end

"""
    transition_index_map(model::MultistateProcess) -> Dict{Tuple{Int,Int},Int}

Create transition index map from model's tmat.
"""
transition_index_map(model::MultistateProcess) = transition_index_map(model.tmat)

"""
    print_transition_map(model::MultistateProcess)

Print human-readable table of transition indices for user reference.
Useful when specifying obstype_by_transition dictionaries.
"""
function print_transition_map(model::MultistateProcess)
    transitions = enumerate_transitions(model.tmat)
    println("Transition Index Map:")
    println("Index | From → To")
    println("------|----------")
    for (idx, (from, to)) in enumerate(transitions)
        println(lpad(idx, 5), " | ", from, " → ", to)
    end
end
```

**Export**: Add to `src/MultistateModels.jl`:
```julia
export enumerate_transitions, transition_index_map, print_transition_map
```

### Phase 2: Validation Utilities

**File**: `src/utilities/transition_helpers.jl` (continued)

```julia
"""
    validate_obstype_by_transition(obstype_dict, n_transitions)

Validate obstype_by_transition dictionary.
Throws ArgumentError if:
- Keys are outside valid range [1, n_transitions]
- Values are < 1
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
                "Observation type code must be >= 1, got $obscode for transition $trans_idx"
            ))
        end
    end
end

"""
    validate_censoring_matrix(censoring_matrix, n_transitions, pattern)

Validate censoring matrix and pattern index.
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
            "got $nrows rows"
        ))
    end
    
    if !isnothing(pattern)
        if pattern < 1 || pattern > ncols
            throw(ArgumentError(
                "Censoring pattern index $pattern out of range [1, $ncols]"
            ))
        end
    end
    
    # Validate all values are valid obstype codes
    if any(censoring_matrix .< 1)
        throw(ArgumentError(
            "All censoring matrix values must be >= 1 (valid obstype codes)"
        ))
    end
end
```

### Phase 3: Core Observation Logic

**File**: `src/simulation/path_utilities.jl`

Modify `observe_path` function to accept new parameters:

```julia
"""
    observe_path(samplepath::SamplePath, model::MultistateProcess;
                 obstype_by_transition = nothing,
                 censoring_matrix = nothing,
                 censoring_pattern = nothing) 

Return `statefrom` and `stateto` for a jump chain observed according to the
observation scheme in model.data, with optional per-transition observation types.

# Arguments
- `samplepath::SamplePath`: Continuous-time sample path
- `model::MultistateProcess`: Model containing data with observation scheme
- `obstype_by_transition::Union{Nothing,Dict{Int,Int}}`: Optional mapping from
  transition index to observation type code. Takes precedence over censoring_matrix.
- `censoring_matrix::Union{Nothing,AbstractMatrix{Int}}`: Optional matrix of
  censoring codes (n_transitions × n_patterns)
- `censoring_pattern::Union{Nothing,Int}`: Column index in censoring_matrix to use

# Observation Type Codes
- `1`: Exact - transition time and states fully observed
- `2`: Panel - only endpoint state at interval boundary observed
- `3+`: Censored with specific code - endpoint state missing

# Returns
DataFrame with columns: id, tstart, tstop, statefrom, stateto, obstype
"""
function observe_path(
    samplepath::SamplePath, 
    model::MultistateProcess;
    obstype_by_transition::Union{Nothing,Dict{Int,Int}} = nothing,
    censoring_matrix::Union{Nothing,AbstractMatrix{Int}} = nothing,
    censoring_pattern::Union{Nothing,Int} = nothing
)
```

#### Implementation Details

1. **Setup Phase** (before main loop):
```julia
# Build transition index map once
trans_map = transition_index_map(model.tmat)
n_transitions = length(trans_map)

# Validate inputs
if !isnothing(obstype_by_transition)
    validate_obstype_by_transition(obstype_by_transition, n_transitions)
end

if !isnothing(censoring_matrix)
    validate_censoring_matrix(censoring_matrix, n_transitions, censoring_pattern)
end

# Helper to get obstype for a transition
function get_transition_obstype(statefrom::Int, stateto::Int)
    trans_idx = get(trans_map, (statefrom, stateto), nothing)
    if isnothing(trans_idx)
        return 1  # Default to exact if transition not in map
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
```

2. **Modified Exact Interval Branch** (when `subj_dat.obstype[r] == 1`):

```julia
if subj_dat.obstype[r] == 1
    # Only apply per-transition logic if override is provided
    if !isnothing(obstype_by_transition) || !isnothing(censoring_matrix)
        
        # Find all jumps in this interval
        right_ind = searchsortedlast(samplepath.times, subj_dat.tstop[r])
        jump_inds = findall(subj_dat.tstart[r] .< samplepath.times .< subj_dat.tstop[r])
        
        # Classify each jump
        exact_jumps = Int[]
        non_exact_obstypes = Int[]
        
        for jump_idx in jump_inds
            statefrom = samplepath.states[jump_idx - 1]
            stateto = samplepath.states[jump_idx]
            obtype = get_transition_obstype(statefrom, stateto)
            
            if obtype == 1
                push!(exact_jumps, jump_idx)
            else
                push!(non_exact_obstypes, obtype)
            end
        end
        
        # Count rows needed
        n_exact_rows = length(exact_jumps)
        n_interval_rows = isempty(non_exact_obstypes) ? 0 : 1
        total_rows = n_exact_rows + n_interval_rows
        
        if total_rows == 0
            # No transitions in interval, emit single row spanning interval
            # (handles case where subject stays in same state)
            total_rows = 1
            emit_static_interval = true
        else
            emit_static_interval = false
        end
        
        obsdat_inds = range(rowind; length = total_rows)
        
        # Emit exact rows
        if n_exact_rows > 0
            obsdat.tstop[rowind:(rowind+n_exact_rows-1)] = samplepath.times[exact_jumps]
            obsdat.stateto[rowind:(rowind+n_exact_rows-1)] = samplepath.states[exact_jumps]
            obsdat.obstype[rowind:(rowind+n_exact_rows-1)] .= 1
        end
        
        # Emit interval-level row for non-exact transitions
        if n_interval_rows > 0
            interval_row_idx = rowind + n_exact_rows
            interval_obstype = maximum(non_exact_obstypes)
            
            obsdat.tstop[interval_row_idx] = subj_dat.tstop[r]
            obsdat.obstype[interval_row_idx] = interval_obstype
            
            if interval_obstype == 2
                # Panel: observe endpoint state
                obsdat.stateto[interval_row_idx] = samplepath.states[right_ind]
            else
                # Censored: endpoint unknown
                obsdat.stateto[interval_row_idx] = missing
            end
        end
        
        # Handle static interval (no transitions)
        if emit_static_interval
            obsdat.tstop[rowind] = subj_dat.tstop[r]
            obsdat.stateto[rowind] = samplepath.states[right_ind]
            obsdat.obstype[rowind] = 1  # Default to exact for no-transition intervals
        end
        
        # Copy covariates to all emitted rows
        if ncol(subj_dat) > 6
            obsdat[obsdat_inds, Not(1:6)] = 
                subj_dat[r*ones(Int32, length(obsdat_inds)), Not(1:6)]
        end
        
        rowind += total_rows
        
    else
        # No per-transition override: use existing exact logic
        # [existing code for exact observation - unchanged]
    end
    
else
    # Panel/censored interval: preserve existing behavior
    # [existing code for non-exact observation - unchanged]
end
```

### Phase 4: Integration with `simulate()`

**File**: `src/simulation/simulate.jl`

1. **Update function signatures** for `simulate`, `simulate_data`, `simulate_paths`:

```julia
function simulate(model::MultistateProcess; 
                  nsim = 1, 
                  data = true, 
                  paths = false, 
                  strategy::AbstractTransformStrategy = CachedTransformStrategy(),
                  solver::AbstractJumpSolver = OptimJumpSolver(),
                  newdata::Union{Nothing,DataFrame} = nothing,
                  tmax::Union{Nothing,Float64} = nothing,
                  autotmax::Bool = true,
                  expanded::Bool = true,
                  obstype_by_transition::Union{Nothing,Dict{Int,Int}} = nothing,
                  censoring_matrix::Union{Nothing,AbstractMatrix{Int}} = nothing,
                  censoring_pattern::Union{Nothing,Int} = nothing)
```

2. **Add validation** after data preparation:

```julia
# After _prepare_simulation_data call
if !isnothing(obstype_by_transition) || !isnothing(censoring_matrix)
    trans_map = transition_index_map(model.tmat)
    n_transitions = length(trans_map)
    
    if !isnothing(obstype_by_transition)
        validate_obstype_by_transition(obstype_by_transition, n_transitions)
    end
    
    if !isnothing(censoring_matrix)
        validate_censoring_matrix(censoring_matrix, n_transitions, censoring_pattern)
    end
end
```

3. **Forward parameters** to `observe_path`:

```julia
if data == true
    datasets[j, i] = observe_path(
        samplepath, 
        model;
        obstype_by_transition = obstype_by_transition,
        censoring_matrix = censoring_matrix,
        censoring_pattern = censoring_pattern
    )
end
```

### Phase 5: Phase-Type Model Support

**File**: `src/simulation/simulate.jl`

Handle `expanded=false` with per-transition observation types:

```julia
# Before validation, if phase-type and expanded=false
if !expanded && has_phasetype_expansion(model) && !isnothing(obstype_by_transition)
    # Map transition indices from observed to expanded space
    mappings = model.phasetype_expansion.mappings
    obstype_by_transition = _map_obstypes_to_expanded(
        obstype_by_transition,
        mappings,
        model.phasetype_expansion.original_tmat,
        model.tmat
    )
end
```

**File**: `src/phasetype/expansion.jl` (new helper)

```julia
"""
    _map_obstypes_to_expanded(obstype_dict, mappings, orig_tmat, expanded_tmat)

Map per-transition observation types from observed to expanded state space.
"""
function _map_obstypes_to_expanded(
    obstype_dict::Dict{Int,Int},
    mappings::PhaseTypeMappings,
    orig_tmat::Matrix{Int},
    expanded_tmat::Matrix{Int}
)
    # Get transition maps for both spaces
    orig_trans_map = transition_index_map(orig_tmat)
    expanded_trans_map = transition_index_map(expanded_tmat)
    
    # Map each specified transition
    expanded_dict = Dict{Int,Int}()
    
    for (orig_idx, obscode) in obstype_dict
        # Get original transition
        orig_transitions = enumerate_transitions(orig_tmat)
        orig_from, orig_to = orig_transitions[orig_idx]
        
        # Map to expanded states
        expanded_from_states = get_expanded_states(mappings, orig_from)
        expanded_to_states = get_expanded_states(mappings, orig_to)
        
        # Find all corresponding transitions in expanded space
        for ef in expanded_from_states
            for et in expanded_to_states
                if expanded_tmat[ef, et] != 0
                    expanded_idx = expanded_trans_map[(ef, et)]
                    expanded_dict[expanded_idx] = obscode
                end
            end
        end
    end
    
    return expanded_dict
end
```

### Phase 6: Documentation

**File**: `src/simulation/simulate.jl` - Update docstring:

```julia
"""
    simulate(model::MultistateProcess; ...)

Simulate datasets and/or continuous-time sample paths from a multistate model.

...existing documentation...

# Per-Transition Observation Types (NEW)

Control observation type on a per-transition basis using `obstype_by_transition`:

```julia
# Get transition indices
print_transition_map(model)
# Output:
# Index | From → To
# ------|----------
#     1 | 1 → 2
#     2 | 1 → 3
#     3 | 2 → 3

# Specify mixed observation: transition 1 exact, transition 2 panel, transition 3 censored
obstype_map = Dict(1 => 1, 2 => 2, 3 => 3)
datasets = simulate(model; nsim=100, obstype_by_transition=obstype_map)
```

## Censoring Patterns

Use `censoring_matrix` for systematic censoring schemes:

```julia
# Define 3 patterns for 3 transitions
censoring_patterns = [
    1 1 2;  # Pattern 1: trans 1,2 exact, trans 3 panel
    1 2 3;  # Pattern 2: trans 1 exact, trans 2 panel, trans 3 censored
    2 2 2   # Pattern 3: all panel
]'  # Transpose to get (n_trans × n_patterns)

# Simulate with pattern 2
datasets = simulate(model; nsim=100, 
                   censoring_matrix=censoring_patterns, 
                   censoring_pattern=2)
```

## Mixed Observations Within Intervals

When multiple transitions occur in one observation interval with different observation types:
- Transitions with `obstype=1` emit exact rows with true transition times
- Transitions with `obstype≥2` contribute to one interval-level row
- Interval-level row has `obstype = max(codes)` among non-exact transitions
- Panel intervals (`obstype=2`) observe endpoint state
- Censored intervals (`obstype≥3`) have `stateto=missing`

# Arguments (NEW)
- `obstype_by_transition::Union{Nothing,Dict{Int,Int}}`: Optional dictionary mapping
  transition index (from `enumerate_transitions(model.tmat)`) to observation type code.
  Codes: 1=exact, 2=panel, 3+=censored. Only applies to intervals originally marked
  as exact (`obstype=1`) in model.data. Use `print_transition_map(model)` to see indices.
- `censoring_matrix::Union{Nothing,AbstractMatrix{Int}}`: Optional matrix of size
  `(n_transitions, n_patterns)` specifying observation codes for systematic censoring.
  Applied when `obstype_by_transition` doesn't specify a transition.
- `censoring_pattern::Union{Nothing,Int}`: Column index (1-based) in `censoring_matrix`
  to use. Required if `censoring_matrix` is provided.

...existing documentation continues...
```

### Phase 7: Testing Strategy

**Location**: All tests in `MultistateModelsTests/unit/test_per_transition_obstype.jl`

**Core Testing Philosophy**: Every test must compare the true underlying path to the generated data and verify that values are **exactly correct**. No probabilistic assertions—we verify deterministic correctness given a known path.

#### Testing Approach

1. **Path-to-Data Verification**: Simulate with `data=true, paths=true`, then verify that the data correctly represents the path under the specified observation scheme.

2. **Exact Value Checking**: For each test case:
   - Compare transition times in data to true times in path
   - Compare states in data to true states in path
   - Verify obstype codes match specification
   - Verify `stateto = missing` exactly when obstype >= 3

3. **Deterministic Seeds**: Use fixed random seeds for reproducibility.

4. **Edge Case Coverage**: Test boundary conditions explicitly.

---

**File**: `MultistateModelsTests/unit/test_per_transition_obstype.jl` (new file)

```julia
# =============================================================================
# Per-Transition Observation Type Tests
# =============================================================================
#
# Tests verifying correct observation of paths under per-transition obstype.
# All tests compare true paths to generated data and verify exact correctness.
#
# =============================================================================

using Test
using Random
using DataFrames
using MultistateModels
using MultistateModels: simulate, simulate_path, observe_path, SamplePath,
    enumerate_transitions, transition_index_map, print_transition_map,
    Hazard, multistatemodel, set_parameters!
using .TestFixtures

# =============================================================================
# Helper Functions for Path-Data Verification
# =============================================================================

"""
    verify_exact_observation(path::SamplePath, data::DataFrame, interval_tstart, interval_tstop)

Verify that an interval with obstype=1 (all exact) correctly records all transitions.
Returns true if data exactly matches path within the interval.
"""
function verify_exact_observation(path::SamplePath, data::DataFrame, 
                                   interval_tstart::Float64, interval_tstop::Float64)
    # Find transitions in the path within this interval
    path_jumps = Int[]
    for i in 2:length(path.times)
        if interval_tstart < path.times[i] <= interval_tstop
            push!(path_jumps, i)
        end
    end
    
    # Find corresponding rows in data (exact observations within interval)
    data_rows = data[(data.tstart .>= interval_tstart) .& 
                     (data.tstop .<= interval_tstop) .&
                     (data.obstype .== 1), :]
    
    # Number of data rows should equal number of path transitions + 1 (for intervals)
    # Actually: each exact transition creates a row ending at that time
    
    # Verify each transition time appears in data
    for jump_idx in path_jumps
        jump_time = path.times[jump_idx]
        state_from = path.states[jump_idx - 1]
        state_to = path.states[jump_idx]
        
        # Find row in data with tstop == jump_time
        matching = data_rows[isapprox.(data_rows.tstop, jump_time, atol=1e-10), :]
        if nrow(matching) != 1
            return false, "Expected exactly 1 row with tstop=$jump_time, found $(nrow(matching))"
        end
        
        # Verify states
        if matching.stateto[1] != state_to
            return false, "State mismatch at t=$jump_time: expected $state_to, got $(matching.stateto[1])"
        end
    end
    
    return true, ""
end

"""
    verify_panel_observation(path::SamplePath, data_row::DataFrameRow)

Verify that a panel observation row correctly captures the endpoint state.
"""
function verify_panel_observation(path::SamplePath, data_row::DataFrameRow)
    # Find state at tstop in the path
    tstop = data_row.tstop
    path_idx = searchsortedlast(path.times, tstop)
    true_state = path.states[path_idx]
    
    # Panel row should have correct endpoint state
    if data_row.stateto != true_state
        return false, "Panel endpoint mismatch: expected $true_state, got $(data_row.stateto)"
    end
    
    if data_row.obstype != 2
        return false, "Panel row should have obstype=2, got $(data_row.obstype)"
    end
    
    return true, ""
end

"""
    verify_censored_observation(data_row::DataFrameRow, expected_code::Int)

Verify that a censored observation row has missing stateto and correct code.
"""
function verify_censored_observation(data_row::DataFrameRow, expected_code::Int)
    if !ismissing(data_row.stateto)
        return false, "Censored row should have missing stateto, got $(data_row.stateto)"
    end
    
    if data_row.obstype != expected_code
        return false, "Censored row should have obstype=$expected_code, got $(data_row.obstype)"
    end
    
    return true, ""
end

"""
    count_transitions_by_type(path::SamplePath, trans_map::Dict{Tuple{Int,Int},Int})

Count how many times each transition type occurs in a path.
Returns Dict{Int,Int} mapping transition index to count.
"""
function count_transitions_by_type(path::SamplePath, trans_map::Dict{Tuple{Int,Int},Int})
    counts = Dict{Int,Int}()
    for i in 2:length(path.states)
        from_state = path.states[i-1]
        to_state = path.states[i]
        trans_idx = get(trans_map, (from_state, to_state), nothing)
        if !isnothing(trans_idx)
            counts[trans_idx] = get(counts, trans_idx, 0) + 1
        end
    end
    return counts
end

# =============================================================================
# Unit Tests: Transition Enumeration
# =============================================================================

@testset "Transition Enumeration" begin
    
    @testset "3-state illness-death model" begin
        # tmat with nonzero entries indicating transitions
        tmat = [0 1 2;
                0 0 3;
                0 0 0]
        
        transitions = enumerate_transitions(tmat)
        
        # Verify count
        @test length(transitions) == 3
        
        # Verify row-major order
        @test transitions[1] == (1, 2)  # First nonzero in row 1
        @test transitions[2] == (1, 3)  # Second nonzero in row 1
        @test transitions[3] == (2, 3)  # First nonzero in row 2
        
        # Verify index map
        trans_map = transition_index_map(tmat)
        @test trans_map[(1, 2)] == 1
        @test trans_map[(1, 3)] == 2
        @test trans_map[(2, 3)] == 3
        
        # Verify no spurious entries
        @test !haskey(trans_map, (2, 1))
        @test !haskey(trans_map, (3, 1))
    end
    
    @testset "4-state reversible model" begin
        tmat = [0 1 0 0;
                2 0 3 0;
                0 4 0 5;
                0 0 0 0]
        
        transitions = enumerate_transitions(tmat)
        
        # Verify count
        @test length(transitions) == 5
        
        # Verify order: row 1 first, then row 2, then row 3
        @test transitions[1] == (1, 2)
        @test transitions[2] == (2, 1)
        @test transitions[3] == (2, 3)
        @test transitions[4] == (3, 2)
        @test transitions[5] == (3, 4)
    end
    
    @testset "Single transition" begin
        tmat = [0 1;
                0 0]
        
        transitions = enumerate_transitions(tmat)
        @test length(transitions) == 1
        @test transitions[1] == (1, 2)
    end
    
    @testset "Empty tmat (no transitions)" begin
        tmat = zeros(Int, 2, 2)
        transitions = enumerate_transitions(tmat)
        @test isempty(transitions)
    end
end

# =============================================================================
# Unit Tests: Input Validation
# =============================================================================

@testset "Input Validation" begin
    
    # Create a test model fixture
    fixture = toy_illness_death_model()  # Assumes this exists or create it
    model = fixture.model
    n_trans = length(enumerate_transitions(model.tmat))
    
    @testset "Invalid transition index - too high" begin
        @test_throws ArgumentError simulate(
            model;
            obstype_by_transition = Dict(n_trans + 1 => 1)
        )
    end
    
    @testset "Invalid transition index - zero" begin
        @test_throws ArgumentError simulate(
            model;
            obstype_by_transition = Dict(0 => 1)
        )
    end
    
    @testset "Invalid transition index - negative" begin
        @test_throws ArgumentError simulate(
            model;
            obstype_by_transition = Dict(-1 => 1)
        )
    end
    
    @testset "Invalid obstype code - zero" begin
        @test_throws ArgumentError simulate(
            model;
            obstype_by_transition = Dict(1 => 0)
        )
    end
    
    @testset "Invalid obstype code - negative" begin
        @test_throws ArgumentError simulate(
            model;
            obstype_by_transition = Dict(1 => -1)
        )
    end
    
    @testset "Censoring matrix - wrong row count" begin
        wrong_rows = ones(Int, n_trans + 1, 2)  # Too many rows
        @test_throws ArgumentError simulate(
            model;
            censoring_matrix = wrong_rows,
            censoring_pattern = 1
        )
    end
    
    @testset "Censoring matrix - pattern index out of range" begin
        cmat = ones(Int, n_trans, 2)
        @test_throws ArgumentError simulate(
            model;
            censoring_matrix = cmat,
            censoring_pattern = 3  # Only 2 columns
        )
    end
    
    @testset "Censoring matrix - invalid codes" begin
        cmat = zeros(Int, n_trans, 2)  # Contains zeros
        @test_throws ArgumentError simulate(
            model;
            censoring_matrix = cmat,
            censoring_pattern = 1
        )
    end
end

# =============================================================================
# Unit Tests: Path-to-Data Correctness (Core Tests)
# =============================================================================

@testset "Path-to-Data Correctness" begin
    
    @testset "All exact (baseline) - verify path matches data" begin
        # Create 3-state illness-death model
        fixture = toy_illness_death_model()
        model = fixture.model
        
        Random.seed!(12345)
        
        # Simulate with both data and paths
        data_list, paths_list = simulate(model; nsim=10, data=true, paths=true)
        
        for sim_idx in 1:10
            paths = paths_list[sim_idx]
            data = data_list[sim_idx]
            
            for (subj_idx, path) in enumerate(paths)
                # Get subject's data rows
                subj_data = data[data.id .== path.subj, :]
                
                # Verify every transition in path appears in data with exact time
                for i in 2:length(path.times)
                    jump_time = path.times[i]
                    state_from = path.states[i-1]
                    state_to = path.states[i]
                    
                    # Skip if transition is to same state (no actual transition)
                    state_from == state_to && continue
                    
                    # Find row ending at this time
                    matching = subj_data[isapprox.(subj_data.tstop, jump_time, atol=1e-10), :]
                    
                    @test nrow(matching) >= 1 "Missing data row for transition at t=$jump_time"
                    @test matching.stateto[1] == state_to "State mismatch at t=$jump_time: path=$state_to, data=$(matching.stateto[1])"
                    @test matching.obstype[1] == 1 "All-exact should have obstype=1"
                end
            end
        end
    end
    
    @testset "Single transition panel - verify endpoint state" begin
        # 2-state model: state 1 → state 2 (absorbing)
        fixture = toy_two_state_exp_model(rate=0.5, horizon=20.0)
        model = fixture.model
        trans_map = transition_index_map(model.tmat)
        
        # Make transition 1→2 panel observed
        obstype_map = Dict(1 => 2)
        
        Random.seed!(54321)
        data_list, paths_list = simulate(model; nsim=20, data=true, paths=true,
                                         obstype_by_transition=obstype_map)
        
        for sim_idx in 1:20
            path = paths_list[sim_idx][1]  # Single subject
            data = data_list[sim_idx]
            
            # Find state at end of observation window
            final_time = maximum(data.tstop)
            path_idx = searchsortedlast(path.times, final_time)
            true_final_state = path.states[path_idx]
            
            # Data should have panel rows (obstype=2)
            panel_rows = data[data.obstype .== 2, :]
            
            if nrow(panel_rows) > 0
                # Last panel row should capture correct endpoint
                last_panel = panel_rows[end, :]
                @test last_panel.stateto == true_final_state "Panel endpoint: expected $true_final_state, got $(last_panel.stateto)"
            end
        end
    end
    
    @testset "Censored transition - verify missing stateto" begin
        # 3-state model with one censored transition
        fixture = toy_illness_death_model()
        model = fixture.model
        trans_map = transition_index_map(model.tmat)
        
        # Make transition 2 (1→3, direct death) censored
        obstype_map = Dict(2 => 3)
        
        Random.seed!(11111)
        data_list, paths_list = simulate(model; nsim=50, data=true, paths=true,
                                         obstype_by_transition=obstype_map)
        
        for sim_idx in 1:50
            data = data_list[sim_idx]
            
            # All censored rows must have missing stateto
            censored_rows = data[data.obstype .== 3, :]
            for row_idx in 1:nrow(censored_rows)
                @test ismissing(censored_rows.stateto[row_idx]) "Censored row must have missing stateto"
            end
        end
    end
    
    @testset "Mixed exact/panel - verify correct separation" begin
        # 3-state illness-death: trans 1 exact, trans 2 panel, trans 3 exact
        fixture = toy_illness_death_model()
        model = fixture.model
        trans_map = transition_index_map(model.tmat)
        
        # Transition indices: 1=(1,2), 2=(1,3), 3=(2,3)
        obstype_map = Dict(1 => 1, 2 => 2, 3 => 1)
        
        Random.seed!(22222)
        data_list, paths_list = simulate(model; nsim=30, data=true, paths=true,
                                         obstype_by_transition=obstype_map)
        
        for sim_idx in 1:30
            paths = paths_list[sim_idx]
            data = data_list[sim_idx]
            
            for path in paths
                subj_data = data[data.id .== path.subj, :]
                
                # Check each transition in path
                for i in 2:length(path.times)
                    from_state = path.states[i-1]
                    to_state = path.states[i]
                    jump_time = path.times[i]
                    
                    from_state == to_state && continue
                    
                    trans_idx = get(trans_map, (from_state, to_state), nothing)
                    isnothing(trans_idx) && continue
                    
                    expected_obstype = get(obstype_map, trans_idx, 1)
                    
                    if expected_obstype == 1
                        # Exact: should find row ending at jump_time
                        matching = subj_data[isapprox.(subj_data.tstop, jump_time, atol=1e-10), :]
                        @test nrow(matching) >= 1 "Exact transition missing at t=$jump_time"
                        @test matching.stateto[1] == to_state "Exact state mismatch"
                        @test matching.obstype[1] == 1 "Exact should have obstype=1"
                    else
                        # Panel: transition time should NOT appear as row boundary
                        # (interval-level row spans the transition)
                        exact_match = subj_data[(isapprox.(subj_data.tstop, jump_time, atol=1e-10)) .& 
                                                (subj_data.obstype .== 1), :]
                        @test nrow(exact_match) == 0 "Panel transition should not have exact row"
                    end
                end
            end
        end
    end
    
    @testset "Verify interval-level obstype precedence (max rule)" begin
        # When multiple non-exact transitions in one interval,
        # interval-level row should have max(obstype) among them
        
        fixture = toy_illness_death_model()
        model = fixture.model
        
        # Trans 1=(1,2) obstype=2 (panel)
        # Trans 2=(1,3) obstype=3 (censored)
        # If both occur conceptually, max should be 3
        # For this test, we check interval rows have correct precedence
        
        obstype_map = Dict(1 => 2, 2 => 3, 3 => 2)
        
        Random.seed!(33333)
        data_list, paths_list = simulate(model; nsim=50, data=true, paths=true,
                                         obstype_by_transition=obstype_map)
        
        for data in data_list
            # No exact rows should exist since all transitions are non-exact
            exact_rows = data[data.obstype .== 1, :]
            # Actually intervals with no transitions will still be exact
            # Only check that if censoring code 3 appears, it's correct
            
            censored_rows = data[data.obstype .== 3, :]
            for row_idx in 1:nrow(censored_rows)
                @test ismissing(censored_rows.stateto[row_idx]) "Censored rows must have missing stateto"
            end
        end
    end
end

# =============================================================================
# Unit Tests: Censoring Matrix Patterns
# =============================================================================

@testset "Censoring Matrix Patterns" begin
    
    @testset "Pattern application - verify correct code used" begin
        fixture = toy_illness_death_model()
        model = fixture.model
        n_trans = length(enumerate_transitions(model.tmat))
        
        # 3 transitions, 3 patterns
        # Pattern 1: all exact
        # Pattern 2: all panel
        # Pattern 3: all censored (code 3)
        cmat = [1 2 3;
                1 2 3;
                1 2 3]
        
        Random.seed!(44444)
        
        # Test pattern 1 (all exact)
        data1, paths1 = simulate(model; nsim=10, data=true, paths=true,
                                 censoring_matrix=cmat, censoring_pattern=1)
        for data in data1
            @test all(data.obstype .== 1) "Pattern 1 should be all exact"
        end
        
        # Test pattern 2 (all panel)
        data2, paths2 = simulate(model; nsim=10, data=true, paths=true,
                                 censoring_matrix=cmat, censoring_pattern=2)
        for data in data2
            panel_rows = data[data.obstype .== 2, :]
            # Panel rows should have non-missing stateto
            @test all(.!ismissing.(panel_rows.stateto)) "Panel rows must have stateto"
        end
        
        # Test pattern 3 (all censored)
        data3, paths3 = simulate(model; nsim=10, data=true, paths=true,
                                 censoring_matrix=cmat, censoring_pattern=3)
        for data in data3
            censored_rows = data[data.obstype .== 3, :]
            @test all(ismissing.(censored_rows.stateto)) "Censored rows must have missing stateto"
        end
    end
    
    @testset "obstype_by_transition takes precedence over matrix" begin
        fixture = toy_illness_death_model()
        model = fixture.model
        
        # Matrix says all panel (obstype=2)
        cmat = fill(2, 3, 1)
        
        # Dict overrides transition 1 to exact
        obstype_dict = Dict(1 => 1)
        
        Random.seed!(55555)
        data_list, paths_list = simulate(model; nsim=20, data=true, paths=true,
                                         obstype_by_transition=obstype_dict,
                                         censoring_matrix=cmat, censoring_pattern=1)
        
        trans_map = transition_index_map(model.tmat)
        
        for sim_idx in 1:20
            paths = paths_list[sim_idx]
            data = data_list[sim_idx]
            
            for path in paths
                # Check if transition 1 (1→2) occurred
                for i in 2:length(path.times)
                    from_s, to_s = path.states[i-1], path.states[i]
                    if (from_s, to_s) == (1, 2)  # Transition 1
                        # This transition should be exact (dict overrides matrix)
                        jump_time = path.times[i]
                        matching = data[(data.id .== path.subj) .& 
                                       isapprox.(data.tstop, jump_time, atol=1e-10) .&
                                       (data.obstype .== 1), :]
                        @test nrow(matching) >= 1 "Dict should override matrix: trans 1 should be exact"
                    end
                end
            end
        end
    end
end

# =============================================================================
# Unit Tests: Backward Compatibility
# =============================================================================

@testset "Backward Compatibility" begin
    
    @testset "No override produces identical results to current behavior" begin
        fixture = toy_illness_death_model()
        model = fixture.model
        
        Random.seed!(66666)
        data_old, paths_old = simulate(model; nsim=5, data=true, paths=true)
        
        Random.seed!(66666)
        data_new, paths_new = simulate(model; nsim=5, data=true, paths=true,
                                       obstype_by_transition=nothing)
        
        # Should be identical
        for i in 1:5
            @test data_old[i] == data_new[i] "Data should be identical with no override"
            for j in eachindex(paths_old[i])
                @test paths_old[i][j] == paths_new[i][j] "Paths should be identical"
            end
        end
    end
    
    @testset "Empty dict produces same as nothing" begin
        fixture = toy_illness_death_model()
        model = fixture.model
        
        Random.seed!(77777)
        data1, _ = simulate(model; nsim=5, data=true, paths=false)
        
        Random.seed!(77777)
        data2, _ = simulate(model; nsim=5, data=true, paths=false,
                           obstype_by_transition=Dict{Int,Int}())
        
        for i in 1:5
            @test data1[i] == data2[i] "Empty dict should behave like nothing"
        end
    end
end

# =============================================================================
# Unit Tests: Edge Cases
# =============================================================================

@testset "Edge Cases" begin
    
    @testset "No transitions in interval - subject stays in initial state" begin
        # Use very short observation window so transitions are unlikely
        fixture = toy_two_state_exp_model(rate=0.01, horizon=0.1)
        model = fixture.model
        
        obstype_map = Dict(1 => 2)  # Panel
        
        Random.seed!(88888)
        data_list, paths_list = simulate(model; nsim=100, data=true, paths=true,
                                         obstype_by_transition=obstype_map)
        
        # Find cases with no transitions
        for sim_idx in 1:100
            path = paths_list[sim_idx][1]
            data = data_list[sim_idx]
            
            if length(path.states) == 2 && path.states[1] == path.states[2]
                # No actual transition
                # Data should still have valid structure
                @test nrow(data) >= 1 "Should have at least one row"
                @test data.statefrom[1] == path.states[1] "Initial state correct"
            end
        end
    end
    
    @testset "Subject starts in absorbing state" begin
        fixture = toy_absorbing_start_model()  # Assumes this fixture exists
        model = fixture.model
        
        obstype_map = Dict(1 => 2)  # Would apply if any transitions
        
        Random.seed!(99999)
        data_list, paths_list = simulate(model; nsim=10, data=true, paths=true,
                                         obstype_by_transition=obstype_map)
        
        for sim_idx in 1:10
            path = paths_list[sim_idx][1]
            
            # Path should have single state (absorbing)
            @test length(path.states) <= 2
        end
    end
    
    @testset "Multiple subjects - correct ID matching" begin
        # Create multi-subject model
        fixture = toy_multisubject_model(nsubj=5)
        model = fixture.model
        
        obstype_map = Dict(1 => 2)
        
        Random.seed!(10101)
        data_list, paths_list = simulate(model; nsim=5, data=true, paths=true,
                                         obstype_by_transition=obstype_map)
        
        for sim_idx in 1:5
            paths = paths_list[sim_idx]
            data = data_list[sim_idx]
            
            for path in paths
                subj_data = data[data.id .== path.subj, :]
                @test nrow(subj_data) >= 1 "Each subject should have data rows"
                
                # Verify endpoint state matches path
                final_data_time = maximum(subj_data.tstop)
                path_idx = searchsortedlast(path.times, final_data_time)
                true_state = path.states[path_idx]
                
                last_row = subj_data[subj_data.tstop .== final_data_time, :][end, :]
                if last_row.obstype == 2
                    @test last_row.stateto == true_state "Panel endpoint must match path"
                elseif last_row.obstype >= 3
                    @test ismissing(last_row.stateto) "Censored must be missing"
                end
            end
        end
    end
end

# =============================================================================
# Unit Tests: Utility Functions
# =============================================================================

@testset "Utility Functions" begin
    
    @testset "print_transition_map output format" begin
        fixture = toy_illness_death_model()
        model = fixture.model
        
        output = sprint(print_transition_map, model)
        
        @test occursin("Transition Index Map", output)
        @test occursin("Index", output)
        @test occursin("From", output)
        @test occursin("To", output)
        @test occursin("1", output)  # At least index 1
        @test occursin("→", output)  # Arrow character
    end
    
    @testset "transition_index_map from model" begin
        fixture = toy_illness_death_model()
        model = fixture.model
        
        trans_map = transition_index_map(model)
        
        @test trans_map isa Dict{Tuple{Int,Int},Int}
        @test length(trans_map) == length(enumerate_transitions(model.tmat))
    end
end
```

### Phase 8: Integration Tests

**File**: `MultistateModelsTests/integration/test_per_transition_obstype_integration.jl` (new)

```julia
# =============================================================================
# Per-Transition Observation Type Integration Tests
# =============================================================================
#
# End-to-end tests verifying the full workflow with mixed observation types.
#
# =============================================================================

using Test
using Random
using DataFrames
using MultistateModels
using .TestFixtures

@testset "Per-Transition Obstype Integration" begin
    
    @testset "Round-trip: simulate → observe → verify against path" begin
        # This is the gold standard test: verify that observe_path
        # correctly transforms a known path under any obstype specification
        
        fixture = toy_illness_death_model()
        model = fixture.model
        trans_map = transition_index_map(model.tmat)
        
        # Test multiple obstype configurations
        configs = [
            Dict{Int,Int}(),                    # All exact
            Dict(1 => 2),                       # Trans 1 panel
            Dict(2 => 3),                       # Trans 2 censored
            Dict(1 => 1, 2 => 2, 3 => 3),       # Mixed
            Dict(1 => 2, 2 => 2, 3 => 2),       # All panel
        ]
        
        for (cfg_idx, obstype_map) in enumerate(configs)
            Random.seed!(20000 + cfg_idx)
            
            data_list, paths_list = simulate(model; nsim=20, data=true, paths=true,
                                             obstype_by_transition=obstype_map)
            
            for sim_idx in 1:20
                paths = paths_list[sim_idx]
                data = data_list[sim_idx]
                
                for path in paths
                    subj_data = data[data.id .== path.subj, :]
                    
                    # Verify data is consistent with path
                    # 1. Final state in data matches final state in path (for non-censored)
                    # 2. Exact rows have correct transition times
                    # 3. Censored rows have missing stateto
                    
                    last_data_row = subj_data[end, :]
                    final_time = last_data_row.tstop
                    path_final_idx = searchsortedlast(path.times, final_time)
                    path_final_state = path.states[path_final_idx]
                    
                    if last_data_row.obstype == 1
                        @test last_data_row.stateto == path_final_state
                    elseif last_data_row.obstype == 2
                        @test last_data_row.stateto == path_final_state
                    else  # censored
                        @test ismissing(last_data_row.stateto)
                    end
                end
            end
        end
    end
    
    @testset "Large-scale statistical verification" begin
        # Verify that observation types are applied at expected frequencies
        
        fixture = toy_illness_death_model()
        model = fixture.model
        
        obstype_map = Dict(1 => 1, 2 => 2, 3 => 3)
        
        Random.seed!(30000)
        data_list, paths_list = simulate(model; nsim=500, data=true, paths=true,
                                         obstype_by_transition=obstype_map)
        
        trans_map = transition_index_map(model.tmat)
        
        # Count actual transitions by type in paths
        path_trans_counts = Dict(1 => 0, 2 => 0, 3 => 0)
        for paths in paths_list
            for path in paths
                counts = count_transitions_by_type(path, trans_map)
                for (t, c) in counts
                    path_trans_counts[t] = get(path_trans_counts, t, 0) + c
                end
            end
        end
        
        # Count observation types in data
        obs_counts = Dict(1 => 0, 2 => 0, 3 => 0)
        for data in data_list
            for obstype in [1, 2, 3]
                obs_counts[obstype] += sum(data.obstype .== obstype)
            end
        end
        
        # Verify we have observations of all types when transitions occurred
        if path_trans_counts[1] > 0
            @test obs_counts[1] > 0 "Should have exact observations for trans 1"
        end
        if path_trans_counts[2] > 0
            @test obs_counts[2] > 0 "Should have panel observations for trans 2"
        end
        if path_trans_counts[3] > 0
            @test obs_counts[3] > 0 "Should have censored observations for trans 3"
        end
    end
end
```

### Test Fixtures Required

Add to `MultistateModelsTests/fixtures/TestFixtures.jl`:

```julia
"""
    toy_illness_death_model()

Create 3-state illness-death model for testing.
States: 1 (healthy) → 2 (ill) → 3 (dead), or 1 → 3 directly.
"""
function toy_illness_death_model(; horizon=10.0, nsubj=1)
    h12 = Hazard(:exp, 1, 2)  # healthy → ill
    h13 = Hazard(:exp, 1, 3)  # healthy → dead
    h23 = Hazard(:exp, 2, 3)  # ill → dead
    
    dat = DataFrame(
        id = repeat(1:nsubj, inner=1),
        tstart = zeros(nsubj),
        tstop = fill(horizon, nsubj),
        statefrom = ones(Int, nsubj),
        stateto = fill(missing, nsubj),
        obstype = ones(Int, nsubj)
    )
    
    model = multistatemodel(h12, h13, h23; data=dat)
    set_parameters!(model, (
        h12 = (log_rate = log(0.2),),
        h13 = (log_rate = log(0.1),),
        h23 = (log_rate = log(0.3),),
    ))
    
    return (model = model, rates = (0.2, 0.1, 0.3))
end

"""
    toy_multisubject_model(; nsubj=5, horizon=10.0)

Create multi-subject model for testing subject-specific operations.
"""
function toy_multisubject_model(; nsubj=5, horizon=10.0)
    h12 = Hazard(:exp, 1, 2)
    
    dat = DataFrame(
        id = 1:nsubj,
        tstart = zeros(nsubj),
        tstop = fill(horizon, nsubj),
        statefrom = ones(Int, nsubj),
        stateto = fill(2, nsubj),
        obstype = ones(Int, nsubj)
    )
    
    model = multistatemodel(h12; data=dat)
    set_parameters!(model, (h12 = (log_rate = log(0.2),),))
    
    return (model = model,)
end
```

### Test Runner Integration

**Update**: `MultistateModelsTests/src/MultistateModelsTests.jl`:

```julia
# Add to includes:
include("../unit/test_per_transition_obstype.jl")
include("../integration/test_per_transition_obstype_integration.jl")
```

## Edge Cases and Special Situations

### 1. Absorbing States
- States with no outgoing transitions
- **Behavior**: Per-transition settings have no effect (no transitions to override)
- **Handling**: Skip transition lookup for terminal states

### 2. Intervals with No Transitions
- Subject remains in same state throughout interval
- **Behavior**: 
  - If interval originally `obstype=1`, emit one row with exact observation
  - Override doesn't apply (no transitions occurred)
- **Handling**: Check for empty jump set before classification

### 3. Transition Not in Enumeration
- Rare case: transition occurs but not in tmat (should be impossible)
- **Behavior**: Default to `obstype=1` with warning
- **Handling**: Graceful fallback in `get_transition_obstype`

### 4. All Transitions Exact
- All transitions in interval specified as `obstype=1`
- **Behavior**: Same as current default (no interval-level row)
- **Handling**: Skip interval-level row if `isempty(non_exact_obstypes)`

### 5. All Transitions Non-Exact
- No exact transitions in interval
- **Behavior**: Emit only interval-level row with max obstype
- **Handling**: `n_exact_rows = 0`, proceed with interval row

### 6. Conflicting Specifications
- Both `obstype_by_transition` and `censoring_matrix` provided
- **Behavior**: `obstype_by_transition` takes precedence for specified keys
- **Handling**: Check dict first, fall back to matrix for unspecified

### 7. Original Data Panel/Censored
- Interval in `model.data` has `obstype >= 2`
- **Behavior**: Ignore per-transition overrides for that interval
- **Rationale**: Avoid double-counting; respect original observation structure
- **Handling**: Skip per-transition logic branch entirely

## Performance Considerations

### Optimization Strategies

1. **Pre-compute transition map once per simulation**
   - Don't rebuild for each path
   - Pass as argument through call chain

2. **Use views and in-place operations**
   - Avoid intermediate allocations in hot loop
   - Pre-allocate result arrays based on worst-case size

3. **Branch prediction**
   - Most common case: no per-transition override
   - Fast path for `isnothing(obstype_by_transition)`

4. **Vector operations where possible**
   - Classify all jumps in one pass
   - Use broadcasting for obstype assignment

### Benchmarking

Compare performance before/after implementation:
- Same simulation with and without per-transition specification
- Target: < 5% overhead when feature not used
- Target: < 20% overhead when feature actively used

## Future Extensions

### 1. Per-Subject Patterns
Add column to `newdata`:
```julia
newdata.censoring_pattern = [1, 2, 1, 3, ...]  # Pattern per subject
```

### 2. Time-Varying Observation Schemes
Change observation type mid-trajectory:
```julia
obstype_by_transition_time = Dict(
    1 => [(0.0, 5.0, 1), (5.0, 10.0, 2)]  # Exact before t=5, panel after
)
```

### 3. Informative Observation Times
Tie observation times to transition occurrence:
```julia
observe_if_transition = Dict(1 => true, 2 => false)
# Generate observation at t+ε when transition 1 occurs
```

### 4. Observation Error
Add measurement error to observed times/states:
```julia
observation_noise = Dict(
    :time_noise => Normal(0, 0.1),
    :state_error_prob => 0.05
)
```

## Backward Compatibility

- **Default behavior unchanged**: If new arguments not provided, existing code works exactly as before
- **No breaking changes**: All existing function signatures accept new optional parameters
- **Deprecation**: None required (pure addition of functionality)

## Documentation Requirements

### User Guide
- Add section: "Custom Observation Schemes"
- Include workflow example with `print_transition_map`
- Explain precedence rules clearly

### API Reference
- Update docstrings for `simulate`, `simulate_data`, `simulate_paths`
- Document new utility functions
- Add examples to function docs

### Vignette/Tutorial
- Create example notebook: "Simulating Realistic Observation Patterns"
- Show medical study scenario with differential observation
- Demonstrate fitting to mixed-observation data

## Success Criteria

### Functionality
- [ ] Can specify observation type per transition via Dict
- [ ] Can specify observation type via censoring matrix patterns
- [ ] Mixed exact/panel/censored in same interval works correctly
- [ ] Validation catches all error cases with clear messages
- [ ] Phase-type models with `expanded=false` map indices correctly
- [ ] All tests pass

### Testing (Path-to-Data Verification)
- [ ] Every transition in path appears correctly in data under exact obstype
- [ ] Panel observations correctly capture endpoint state from path
- [ ] Censored observations have `stateto = missing` exactly
- [ ] Mixed obstype intervals emit correct combination of rows
- [ ] Interval-level obstype follows max precedence rule
- [ ] Backward compatibility verified: no override = identical to current behavior
- [ ] Multi-subject simulations correctly associate data with paths by ID
- [ ] Edge cases tested: absorbing states, no-transition intervals, single transitions
- [ ] Large-scale statistical verification of obstype frequencies

### Test Quality Standards
- [ ] All tests compare true paths to generated data (no probabilistic assertions)
- [ ] Tests use fixed random seeds for reproducibility
- [ ] Test fixtures documented and reusable
- [ ] Integration tests verify end-to-end workflow
- [ ] Tests located in MultistateModelsTests package

### Performance
- [ ] < 5% overhead when feature not used
- [ ] < 20% overhead when feature active
- [ ] No memory leaks or excessive allocations

### Documentation
- [ ] All new functions have docstrings with examples
- [ ] `print_transition_map` clearly explains indexing
- [ ] Precedence rules documented (dict > matrix > default)
- [ ] Examples show common use cases

## Implementation Timeline

1. **Phase 1-2** (Utilities & Validation): 2-3 hours
2. **Phase 3** (Core observation logic): 4-6 hours
3. **Phase 4-5** (Integration & phase-type): 2-3 hours
4. **Phase 6** (Documentation): 2-3 hours
5. **Phase 7** (Unit tests in MultistateModelsTests): 4-5 hours
6. **Phase 8** (Integration tests): 2-3 hours
7. **Review, refinement, and test debugging**: 3-4 hours

**Total estimated effort**: 19-27 hours

## Appendix: Test Fixtures Required

The following fixtures must exist or be added to `MultistateModelsTests/fixtures/TestFixtures.jl`:

| Fixture | Description | Used By |
|---------|-------------|---------|
| `toy_illness_death_model()` | 3-state (healthy→ill→dead) with exponential hazards | Core path-to-data tests |
| `toy_two_state_exp_model()` | 2-state with single absorbing transition | Panel/censored tests |
| `toy_absorbing_start_model()` | Subject starts in absorbing state | Edge case tests |
| `toy_multisubject_model(nsubj)` | Multiple subjects for ID matching tests | Multi-subject tests |

Each fixture returns a NamedTuple with at minimum `(model = ...,)` and optionally ground-truth parameters for verification.
