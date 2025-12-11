# Part 7: Allocation Reduction Strategies

## 7.1 Pre-allocation Patterns

### Path Storage Pre-allocation

Current issue: Growing vectors during simulation.

```julia
# CURRENT: Allocates incrementally
function simulate(model; nsim)
    all_paths = SamplePath[]
    for subj in 1:nsubj
        for sim in 1:nsim
            path = simulate_path(...)
            push!(all_paths, path)  # Allocation + possible resize
        end
    end
    return all_paths
end

# PROPOSED: Pre-allocate
function simulate(model; nsim)
    nsubj = length(model.subjectindices)
    
    # Pre-allocate exact size
    all_paths = Vector{SamplePath}(undef, nsubj * nsim)
    
    idx = Threads.Atomic{Int}(1)
    @threads for subj in 1:nsubj
        for sim in 1:nsim
            path = simulate_path(...)
            local_idx = Threads.atomic_add!(idx, 1)
            all_paths[local_idx] = path
        end
    end
    
    return all_paths
end
```

### Work Array Pre-allocation

Create reusable work arrays for inner loops:

```julia
struct SimulationWorkspace
    # Reusable vectors for path construction
    times::Vector{Float64}
    states::Vector{Int}
    
    # TPM workspace
    hazmat::Matrix{Float64}
    tpm::Matrix{Float64}
    
    # Uniformization workspace
    poisson_samples::Vector{Float64}
    
    function SimulationWorkspace(nstates::Int, max_jumps::Int=1000)
        new(
            Vector{Float64}(undef, max_jumps),
            Vector{Int}(undef, max_jumps),
            Matrix{Float64}(undef, nstates, nstates),
            Matrix{Float64}(undef, nstates, nstates),
            Vector{Float64}(undef, max_jumps)
        )
    end
end

# Thread-local workspaces
const SIMULATION_WORKSPACES = Dict{Int, SimulationWorkspace}()

function get_workspace(nstates::Int)
    tid = Threads.threadid()
    get!(() -> SimulationWorkspace(nstates), SIMULATION_WORKSPACES, tid)
end
```

---

## 7.2 View-Based Data Access

### DataFrame Row Access

```julia
# CURRENT: Creates temporary DataFrameRow
row = df[i, :]  # Allocates

# BETTER: View
row = @view df[i, :]  # No allocation, but still dispatches

# BEST: Direct column access
# Pre-extract columns once:
struct ColumnAccessor
    tstart::Vector{Float64}
    tstop::Vector{Float64}
    statefrom::Vector{Int}
    stateto::Vector{Int}
    obstype::Vector{Int}
end

# Then access:
tstart = cols.tstart[i]  # Direct, no dispatch
```

### Subject Data Views

```julia
# CURRENT
function process_subject(model, subj_idx)
    subj_data = model.data[model.subjectindices[subj_idx], :]  # Allocates new DataFrame
    ...
end

# PROPOSED
function process_subject(model, subj_idx)
    inds = model.subjectindices[subj_idx]
    # Use column accessors
    for i in inds
        tstart = model.data.tstart[i]
        ...
    end
end
```

---

## 7.3 Type-Stable Containers

### Replace Dict with NamedTuple

```julia
# CURRENT: Dict lookup in hot path
function eval_hazard(hazard, t, params_dict)
    pars = params_dict[hazard.hazname]  # Type unstable
    ...
end

# PROPOSED: NamedTuple
function eval_hazard(hazard, t, params::NamedTuple, idx::Int)
    pars = params[idx]  # Type stable when idx is known
    ...
end

# Or: Pre-extract all parameters
struct HazardParameters{N, T}
    params::NTuple{N, T}  # Fully typed
end
```

### Avoid Union Types in Hot Paths

```julia
# PROBLEMATIC
mutable struct PathState
    current_state::Int
    current_time::Float64
    absorbing::Union{Nothing, Bool}  # Union forces boxing
end

# BETTER
mutable struct PathState
    current_state::Int
    current_time::Float64
    absorbing::Bool  # Always defined
    is_set::Bool     # Separate flag
end

# OR: Use sentinel values
const NOT_SET = -999
```

---

## 7.4 Avoid Closure Captures

### Problem: Closures Capture by Boxing

```julia
# PROBLEMATIC: params is captured, may box
function make_objective(model, params)
    function objective(x)
        # params captured here - may allocate
        return loglik(x, params, model)
    end
    return objective
end

# BETTER: Use functor
struct ObjectiveFunctor{M, P}
    model::M
    params::P
end

function (obj::ObjectiveFunctor)(x)
    loglik(x, obj.params, obj.model)
end
```

### Apply to Root-Finding

```julia
# CURRENT in _find_jump_time
prob = IntervalNonlinearProblem{false}(rootfun, ..., params)
# rootfun may be closure

# PROPOSED
struct RootFunctor{H, P, R}
    hazard::H
    pars::P
    row::R
    target::Float64
    origin::Float64
end

function (rf::RootFunctor)(t, _)
    chaz = eval_cumhaz(rf.hazard, rf.origin, t, rf.pars, rf.row)
    return chaz - rf.target
end

prob = IntervalNonlinearProblem{false}(RootFunctor(...), ...)
```

---

## 7.5 Eliminate String Allocations

### Symbol-Based Keys

```julia
# CURRENT: String keys
params = Dict{String, Vector{Float64}}("h12" => [...], "h23" => [...])

# BETTER: Symbol keys (interned, no allocation)
params = Dict{Symbol, Vector{Float64}}(:h12 => [...], :h23 => [...])

# BEST: NamedTuple (compile-time known)
params = (h12 = [...], h23 = [...])
```

### Avoid String Interpolation in Hot Paths

```julia
# PROBLEMATIC
function log_progress(iter, ll)
    msg = "Iteration $iter: ll = $ll"  # Allocates string
    @info msg
end

# BETTER: Use LazyString or avoid in tight loops
function log_progress(iter, ll)
    @info "Iteration" iter ll  # Structured logging, no interpolation
end
```

---

## 7.6 Batch Operations

### Vectorized Hazard Evaluation

```julia
# CURRENT: Loop over transitions
function compute_hazmat!(H, params, hazards, times, data)
    fill!(H, 0.0)
    for (i, haz) in enumerate(hazards)
        pars = params[haz.hazname]
        for row_idx in relevant_rows
            H[haz.from, haz.to] += eval_hazard(haz, times[row_idx], pars, data[row_idx, :])
        end
    end
end

# PROPOSED: SIMD-friendly batching
function compute_hazmat_batched!(H, params, hazards, times, data)
    fill!(H, 0.0)
    
    for (i, haz) in enumerate(hazards)
        pars = params[i]  # Pre-indexed
        from, to = haz.from, haz.to
        
        # Extract column once
        t_col = times
        
        @inbounds @simd for j in 1:length(t_col)
            H[from, to] += _eval_hazard_inner(haz, t_col[j], pars)
        end
    end
end
```

### Batched Matrix Exponentials

```julia
# CURRENT: One at a time
for interval in intervals
    tpm = exp(Q * dt)
end

# PROPOSED: Batch when many identical dt
dt_groups = group_by_dt(intervals)
for (dt, indices) in dt_groups
    tpm = exp(Q * dt)  # Compute once
    for idx in indices
        tpm_book[idx] = tpm  # Reuse
    end
end
```

---

## 7.7 Reduce Problem Construction Overhead

### Root-Finding Problem Reuse

```julia
# CURRENT: New problem each call
function _find_jump_time(...)
    prob = IntervalNonlinearProblem{false}(rootfun, ...)
    sol = solve(prob, ...)
    return sol.u
end

# PROPOSED: Mutable problem with update
mutable struct ReusableRootProblem{F}
    func::F
    interval::Tuple{Float64, Float64}
    # ... other fields
end

function update_and_solve!(prob::ReusableRootProblem, new_interval, new_func_params)
    # Update in place
    prob.interval = new_interval
    update_params!(prob.func, new_func_params)
    
    # Solve
    return solve_inplace!(prob)
end
```

### Pre-allocated LinearAlgebra Workspace

```julia
# For matrix exponential
using ExponentialUtilities

struct MatExpWorkspace{T}
    cache::ExpMethodGeneric
    mem::AbstractMatrix{T}
end

function MatExpWorkspace(n::Int, T::Type=Float64)
    A = zeros(T, n, n)
    mem = ExponentialUtilities.alloc_mem(A, ExpMethodGeneric())
    MatExpWorkspace(ExpMethodGeneric(), mem)
end

# Reuse across calls
function exp_with_workspace!(result, A, ws::MatExpWorkspace)
    exponential!(result, A, ws.cache, ws.mem)
end
```

---

## 7.8 Memory Layout Optimization

### Struct of Arrays vs Array of Structs

```julia
# CURRENT: Array of Structs (AoS)
struct SamplePath
    subj::Int
    times::Vector{Float64}
    states::Vector{Int}
end
paths = Vector{SamplePath}(...)

# For certain operations, Struct of Arrays (SoA) is better:
struct PathCollection
    subjs::Vector{Int}
    times::Vector{Vector{Float64}}
    states::Vector{Vector{Int}}
end
# Better cache locality for operations on single field
```

### Contiguous Time Arrays

```julia
# If paths have similar length, use Matrix instead of Vector{Vector}
struct FixedLengthPaths
    subjs::Vector{Int}
    times::Matrix{Float64}  # max_jumps × n_paths
    states::Matrix{Int}
    lengths::Vector{Int}    # Actual length of each path
end
```

---

## 7.9 Profiling-Guided Decisions

Before implementing any allocation reduction:

1. **Profile first**: Identify actual hotspots
2. **Measure allocation source**: `--track-allocation=user`
3. **Estimate benefit**: Only optimize if >10% of total allocations
4. **Test impact**: Verify speedup before committing

```bash
# Run with allocation tracking
julia --project=. --track-allocation=user scratch/profiling/profile_simulation.jl

# Check .mem files for allocation sources
```

### Decision Matrix

| Optimization | Effort | Impact | When to Apply |
|-------------|--------|--------|---------------|
| Pre-allocation | Low | Medium | Always for known-size vectors |
| Views | Low | Low-Medium | When passing slices |
| Type stability | Medium | High | Hot paths only |
| Closure removal | Medium | Medium | Root-finding, optimization |
| Batching | High | High | >100 iterations |
| Memory layout | High | Variable | Profile-guided only |
