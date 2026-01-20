# ============================================================================
# Path Sampling Infrastructure - Core Utilities
# ============================================================================
#
# This file contains the foundational data structures and utilities for path
# sampling in MCEM inference:
#
# - PathWorkspace: Thread-local workspace for efficient path sampling
# - Thread-local storage management for workspace reuse
#
# These utilities are used by both Markov (sampling_markov.jl) and phase-type
# (sampling_phasetype.jl) sampling algorithms.
# ============================================================================

# =============================================================================
# PathWorkspace: Pre-allocated workspace for efficient path sampling
# =============================================================================
#
# The PathWorkspace struct reduces allocations in MCEM's hot inner loop
# (sampling thousands of paths per subject per iteration). Key optimizations:
#
# 1. Pre-allocated vectors for times/states that grow but don't shrink
# 2. Cached matrices for ECCTMC R-matrix powers (R^k = I + Q/m)
# 3. Thread-local storage to avoid contention in parallel MCEM
#
# Usage pattern:
#   ws = get_path_workspace()  # Gets thread-local workspace
#   reset!(ws)                 # Clear for new path
#   push_time_state!(ws, t, s) # Add points to path
#   ...
#   path = reduce_jumpchain_ws(ws, subj)  # Create final SamplePath
# =============================================================================

"""
    PathWorkspace

Pre-allocated workspace for path sampling to reduce allocations in MCEM hot loop.

# Fields
- `times::Vector{Float64}`: Pre-allocated buffer for path times
- `states::Vector{Int}`: Pre-allocated buffer for path states
- `times_len::Int`: Current length of times (valid data)
- `states_len::Int`: Current length of states (valid data)
- `max_states::Int`: Number of states in state space (for R matrix sizing)
- `R_base::Matrix{Float64}`: Pre-allocated R = I + Q/m matrix
- `R_slices::Array{Float64,3}`: Pre-allocated R^k matrices (R[:,:,k] = R^k)
- `R_power::Matrix{Float64}`: Temporary matrix for R^k computation
- `times_temp::Vector{Float64}`: Temporary vector for sorting jump times
- `states_temp::Vector{Int}`: Temporary vector for jump state sampling
"""
mutable struct PathWorkspace
    # Path storage
    times::Vector{Float64}
    states::Vector{Int}
    times_len::Int
    states_len::Int
    
    # R matrix workspace (for ECCTMC)
    max_states::Int
    R_base::Matrix{Float64}
    R_slices::Array{Float64, 3}
    R_power::Matrix{Float64}
    
    # Temporary vectors for ECCTMC
    times_temp::Vector{Float64}
    states_temp::Vector{Int}
end

"""
    PathWorkspace(initial_capacity::Int=100, max_states::Int=10, max_jumps::Int=100)

Create a new PathWorkspace with specified capacities.

# Arguments
- `initial_capacity`: Initial capacity for path times/states vectors
- `max_states`: Maximum number of states for R matrix allocation
- `max_jumps`: Maximum number of jumps for temporary vectors
"""
function PathWorkspace(initial_capacity::Int=100, max_states::Int=10, max_jumps::Int=100)
    return PathWorkspace(
        sizehint!(Vector{Float64}(), initial_capacity),
        sizehint!(Vector{Int}(), initial_capacity),
        0,
        0,
        max_states,
        zeros(Float64, max_states, max_states),
        zeros(Float64, max_states, max_states, max_jumps),
        zeros(Float64, max_states, max_states),
        zeros(Float64, max_jumps),
        zeros(Int, max_jumps)
    )
end

"""
    reset!(ws::PathWorkspace)

Reset workspace for a new path (clear lengths but keep allocations).
"""
function reset!(ws::PathWorkspace)
    ws.times_len = 0
    ws.states_len = 0
    return ws
end

"""
    push_time_state!(ws::PathWorkspace, t::Float64, s::Int)

Add a (time, state) pair to the workspace.
"""
function push_time_state!(ws::PathWorkspace, t::Float64, s::Int)
    ws.times_len += 1
    ws.states_len += 1
    
    if ws.times_len > length(ws.times)
        push!(ws.times, t)
        push!(ws.states, s)
    else
        @inbounds ws.times[ws.times_len] = t
        @inbounds ws.states[ws.states_len] = s
    end
    
    return ws
end

"""
    ensure_R_capacity!(ws::PathWorkspace, n_states::Int, n_slices::Int)

Ensure workspace has capacity for R matrices of size n_states × n_states × n_slices.
"""
function ensure_R_capacity!(ws::PathWorkspace, n_states::Int, n_slices::Int)
    if n_states > ws.max_states || n_slices > size(ws.R_slices, 3)
        new_max_states = max(n_states, ws.max_states)
        new_max_slices = max(n_slices, size(ws.R_slices, 3))
        ws.R_base = zeros(Float64, new_max_states, new_max_states)
        ws.R_slices = zeros(Float64, new_max_states, new_max_states, new_max_slices)
        ws.R_power = zeros(Float64, new_max_states, new_max_states)
        ws.max_states = new_max_states
    end
end

"""
    ensure_temp_capacity!(ws::PathWorkspace, n::Int)

Ensure temporary vectors have capacity for n elements.
"""
function ensure_temp_capacity!(ws::PathWorkspace, n::Int)
    if n > length(ws.times_temp)
        resize!(ws.times_temp, n)
        resize!(ws.states_temp, n)
    end
end

# =============================================================================
# Thread-local workspace storage
# =============================================================================
#
# Each thread gets its own PathWorkspace to avoid contention during parallel
# MCEM path sampling. The workspaces are lazily initialized on first use.
# =============================================================================

const THREAD_WORKSPACES = Vector{PathWorkspace}()
const WORKSPACE_LOCK = ReentrantLock()

"""
    get_path_workspace() -> PathWorkspace

Get the thread-local PathWorkspace, creating if necessary.
Thread-safe initialization with lazy allocation.
"""
function get_path_workspace()
    tid = Threads.threadid()
    
    # Fast path: workspace already exists
    if tid <= length(THREAD_WORKSPACES) && isassigned(THREAD_WORKSPACES, tid)
        return THREAD_WORKSPACES[tid]
    end
    
    # Slow path: need to initialize
    lock(WORKSPACE_LOCK) do
        # Resize if needed
        if tid > length(THREAD_WORKSPACES)
            resize!(THREAD_WORKSPACES, tid)
        end
        
        # Create workspace if not exists
        if !isassigned(THREAD_WORKSPACES, tid)
            THREAD_WORKSPACES[tid] = PathWorkspace()
        end
    end
    
    return THREAD_WORKSPACES[tid]
end
