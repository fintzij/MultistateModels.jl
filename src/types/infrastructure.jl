# =============================================================================
# Infrastructure Types
# =============================================================================
#
# AD backend selection, threading configuration, and other infrastructure
# types that support model fitting and optimization.
#
# =============================================================================

# =============================================================================
# AD Backend Concrete Types
# =============================================================================
# Note: ADBackend abstract type is defined in types/abstract.jl

"""
    ForwardDiffBackend <: ADBackend

Use ForwardDiff.jl for automatic differentiation.

**Characteristics:**
- Forward-mode AD: O(n) cost where n = number of parameters
- Efficient for small to medium parameter counts (< ~100 params)
- Tolerates in-place mutation in the objective function
- Default choice for most multistate models

**When to use:**
- Models with few parameters (exponential, Weibull hazards)
- When the mutating likelihood implementation is preferred for speed
"""
struct ForwardDiffBackend <: ADBackend end

"""
    EnzymeBackend <: ADBackend

Use Enzyme.jl for automatic differentiation.

**Characteristics:**
- Reverse-mode AD: O(1) cost in parameters (scales with output size)  
- Efficient for large parameter counts (> ~100 params)
- Requires mutation-free objective function
- Uses `loglik_*_functional` variants internally

**When to use:**
- Models with many parameters (complex spline hazards, many covariates)
- Neural ODE hazards (future extension)
- When Hessian computation is also needed efficiently

**Requirements:**
Enzyme requires the likelihood function to be mutation-free. The package
automatically selects functional (non-mutating) likelihood implementations
when this backend is specified.

**Note:** Enzyme.jl Julia 1.12 support is experimental (as of Dec 2024).
For Julia 1.12, use ForwardDiffBackend or MooncakeBackend.
"""
struct EnzymeBackend <: ADBackend end

"""
    MooncakeBackend <: ADBackend

Use Mooncake.jl for automatic differentiation (reverse-mode).

**Characteristics:**
- Reverse-mode AD: O(1) cost in number of parameters
- Efficient for large parameter counts (> ~100 params)  
- Pure Julia, good version compatibility
- Supports mutation (unlike Zygote)

**Works well for:**
- Semi-Markov models (no matrix exponential in likelihood)
- Models with many parameters where reverse-mode efficiency matters

**Known limitation (as of Dec 2024):**
Does NOT work for Markov panel models. The matrix exponential computation
uses LAPACK.gebal! internally, which Mooncake cannot differentiate through.
ChainRules.jl has an rrule for exp(::Matrix), but that rule itself calls
LAPACK, so Mooncake still fails. Use `ForwardDiffBackend()` for Markov models.
"""
struct MooncakeBackend <: ADBackend end

"""
    default_ad_backend(n_params::Int; is_markov::Bool=false) -> ADBackend

Select default AD backend based on parameter count and model type.

# Arguments
- `n_params::Int`: Number of parameters in the model
- `is_markov::Bool=false`: Whether the model uses Markov panel likelihoods

# Returns
- `ADBackend`: ForwardDiff for Markov models, Mooncake for large non-Markov models

# Notes
ForwardDiff is used for Markov models because matrix exponential differentiation
requires forward-mode AD (Mooncake/Enzyme cannot differentiate LAPACK calls).
For non-Markov models with many parameters, Mooncake's reverse-mode is more efficient.
"""
function default_ad_backend(n_params::Int; is_markov::Bool=false)
    if is_markov
        # Markov models require ForwardDiff due to matrix exponential
        return ForwardDiffBackend()
    else
        # For non-Markov, use reverse-mode for large parameter counts
        return n_params < 100 ? ForwardDiffBackend() : MooncakeBackend()
    end
end

"""
    get_optimization_ad(backend::ADBackend)

Convert ADBackend to Optimization.jl AD specification.
"""
get_optimization_ad(::ForwardDiffBackend) = Optimization.AutoForwardDiff()
get_optimization_ad(::EnzymeBackend) = Optimization.AutoEnzyme()
get_optimization_ad(::MooncakeBackend) = Optimization.AutoMooncake()

# =============================================================================
# Threading Configuration
# =============================================================================
#
# Parallelization support for likelihood evaluation. Uses Julia's built-in 
# Threads.@threads with physical core detection to avoid hyperthreading overhead.
#
# Thread safety considerations:
# - Each subject/path computes an independent likelihood contribution
# - Thread-local accumulators are used for the final sum
# - Shared read-only data (TPM books, hazards) is accessed without locks
# - No mutation of shared state during parallel execution
#
# Usage:
#   fit(model; parallel=true)  # Enable parallel likelihood evaluation
#   fit(model; parallel=false) # Sequential evaluation (default for AD)
#   fit(model; nthreads=4)     # Use exactly 4 threads
#
# =============================================================================

"""
    get_physical_cores() -> Int

Detect the number of physical CPU cores (excluding hyperthreads).

Uses Sys.CPU_THREADS as total threads, then estimates physical cores by
dividing by 2 on systems that typically have 2 threads per core (Intel/AMD x86).
On ARM (Apple Silicon), threads typically equal physical cores.

Returns at least 1 to ensure valid thread count.
"""
function get_physical_cores()
    total_threads = Sys.CPU_THREADS
    # Heuristic: ARM typically doesn't hyperthread, x86 does
    # Check architecture via pointer size and platform hints
    if Sys.ARCH == :aarch64 || Sys.ARCH == :arm64
        # Apple Silicon and ARM: threads ≈ physical cores
        return max(1, total_threads)
    else
        # x86: assume 2 threads per core (hyperthreading)
        return max(1, total_threads ÷ 2)
    end
end

"""
    recommended_nthreads(; task_count::Int=0) -> Int

Recommend number of threads for parallel likelihood evaluation.

# Arguments
- `task_count::Int=0`: Number of parallel tasks (subjects/paths). If 0, ignored.

# Returns
Number of threads to use, considering:
1. Available Julia threads (Threads.nthreads())
2. Physical cores (to avoid hyperthreading overhead)
3. Task count (no benefit from more threads than tasks)

# Notes
- Returns min(available_threads, physical_cores, task_count)
- Leaves 1 core free for main thread if > 4 physical cores available
- Returns 1 if threading provides no benefit
"""
function recommended_nthreads(; task_count::Int=0)
    available = Threads.nthreads()
    physical = get_physical_cores()
    
    # Don't use more threads than physical cores
    n = min(available, physical)
    
    # Leave 1 core for main thread on larger systems
    if n > 4
        n = n - 1
    end
    
    # Don't use more threads than tasks
    if task_count > 0
        n = min(n, task_count)
    end
    
    # At least 1 thread
    return max(1, n)
end

"""
    ThreadingConfig

Configuration for parallel likelihood evaluation.

# Fields
- `enabled::Bool`: Whether parallelization is active
- `nthreads::Int`: Number of threads to use
- `min_batch_size::Int`: Minimum tasks per thread to justify overhead
"""
struct ThreadingConfig
    enabled::Bool
    nthreads::Int
    min_batch_size::Int
end

"""
    ThreadingConfig(; parallel=false, nthreads=nothing, min_batch_size=10)

Create threading configuration for likelihood evaluation.

# Arguments
- `parallel::Bool=false`: Enable parallel execution
- `nthreads::Union{Nothing,Int}=nothing`: Number of threads. If nothing, auto-detect.
- `min_batch_size::Int=10`: Minimum tasks per thread to justify threading overhead

# Notes
When `parallel=true` and `nthreads=nothing`, uses `recommended_nthreads()` for auto-detection.
"""
function ThreadingConfig(; parallel::Bool=false, nthreads::Union{Nothing,Int}=nothing, 
                          min_batch_size::Int=10)
    if !parallel
        return ThreadingConfig(false, 1, min_batch_size)
    end
    
    n = isnothing(nthreads) ? recommended_nthreads() : nthreads
    
    # Disable if only 1 thread available
    if n <= 1
        return ThreadingConfig(false, 1, min_batch_size)
    end
    
    return ThreadingConfig(true, n, min_batch_size)
end

"""
    should_parallelize(config::ThreadingConfig, task_count::Int) -> Bool

Determine whether to use parallel execution for given task count.

Returns true if:
1. Threading is enabled in config
2. Task count exceeds min_batch_size × nthreads threshold
"""
function should_parallelize(config::ThreadingConfig, task_count::Int)
    config.enabled || return false
    return task_count >= config.nthreads * config.min_batch_size
end

# Global threading configuration (can be overridden per-call)
# Protected by lock for thread-safe access (M18_P2 fix)
const _GLOBAL_THREADING_CONFIG = Ref(ThreadingConfig(parallel=false))
const _GLOBAL_THREADING_CONFIG_LOCK = ReentrantLock()

"""
    set_threading_config!(; parallel=false, nthreads=nothing, min_batch_size=10)

Set global threading configuration for likelihood evaluation.

Thread-safe: Uses a lock to prevent race conditions when called from multiple threads.

# Example
```julia
# Enable parallel likelihood evaluation globally
set_threading_config!(parallel=true)

# Use exactly 4 threads
set_threading_config!(parallel=true, nthreads=4)

# Disable parallelization
set_threading_config!(parallel=false)
```
"""
function set_threading_config!(; parallel::Bool=false, nthreads::Union{Nothing,Int}=nothing,
                                 min_batch_size::Int=10)
    config = ThreadingConfig(parallel=parallel, nthreads=nthreads, min_batch_size=min_batch_size)
    lock(_GLOBAL_THREADING_CONFIG_LOCK) do
        _GLOBAL_THREADING_CONFIG[] = config
    end
    return config
end

"""
    get_threading_config() -> ThreadingConfig

Get current global threading configuration.

Thread-safe: Uses a lock to prevent race conditions when called from multiple threads.
"""
function get_threading_config()
    lock(_GLOBAL_THREADING_CONFIG_LOCK) do
        return _GLOBAL_THREADING_CONFIG[]
    end
end