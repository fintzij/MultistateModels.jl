# =============================================================================
# Configuration Type Definitions
# =============================================================================
# Types for AD backends, threading configuration, proposal configs, and other settings.
# =============================================================================

using Optimization

# =============================================================================
# AD Backend Types
# =============================================================================

"""
    ADBackend

Abstract type for automatic differentiation backend selection.
Enables switching between ForwardDiff (forward-mode) and Enzyme/Mooncake (reverse-mode).
"""
abstract type ADBackend end

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

**When to use:**
- Models with many parameters (complex spline hazards, many covariates)
- Neural ODE hazards (future extension)

**Note:** Enzyme.jl Julia 1.12 support is experimental (as of Dec 2024).
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

**Known limitation (as of Dec 2024):**
Does NOT work for Markov panel models (matrix exponential uses LAPACK.gebal!).
Use `ForwardDiffBackend()` for Markov models.
"""
struct MooncakeBackend <: ADBackend end

"""
    default_ad_backend(n_params::Int; is_markov::Bool=false) -> ADBackend

Select default AD backend based on parameter count and model type.
ForwardDiff for Markov models, Mooncake for large non-Markov models.
"""
function default_ad_backend(n_params::Int; is_markov::Bool=false)
    if is_markov
        return ForwardDiffBackend()
    else
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

"""
    get_physical_cores() -> Int

Detect the number of physical CPU cores (excluding hyperthreads).
"""
function get_physical_cores()
    total_threads = Sys.CPU_THREADS
    if Sys.ARCH == :aarch64 || Sys.ARCH == :arm64
        return max(1, total_threads)
    else
        return max(1, total_threads ÷ 2)
    end
end

"""
    recommended_nthreads(; task_count::Int=0) -> Int

Recommend number of threads for parallel likelihood evaluation.
"""
function recommended_nthreads(; task_count::Int=0)
    available = Threads.nthreads()
    physical = get_physical_cores()
    n = min(available, physical)
    
    if n > 4
        n = n - 1
    end
    
    if task_count > 0
        n = min(n, task_count)
    end
    
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
"""
function ThreadingConfig(; parallel::Bool=false, nthreads::Union{Nothing,Int}=nothing, 
                          min_batch_size::Int=10)
    if !parallel
        return ThreadingConfig(false, 1, min_batch_size)
    end
    
    n = isnothing(nthreads) ? recommended_nthreads() : nthreads
    
    if n <= 1
        return ThreadingConfig(false, 1, min_batch_size)
    end
    
    return ThreadingConfig(true, n, min_batch_size)
end

"""
    should_parallelize(config::ThreadingConfig, task_count::Int) -> Bool

Determine whether to use parallel execution for given task count.
"""
function should_parallelize(config::ThreadingConfig, task_count::Int)
    config.enabled || return false
    return task_count >= config.nthreads * config.min_batch_size
end

# Global threading configuration
const _GLOBAL_THREADING_CONFIG = Ref(ThreadingConfig(parallel=false))

"""
    set_threading_config!(; parallel=false, nthreads=nothing, min_batch_size=10)

Set global threading configuration for likelihood evaluation.
"""
function set_threading_config!(; parallel::Bool=false, nthreads::Union{Nothing,Int}=nothing,
                                 min_batch_size::Int=10)
    _GLOBAL_THREADING_CONFIG[] = ThreadingConfig(parallel=parallel, nthreads=nthreads,
                                                  min_batch_size=min_batch_size)
    return _GLOBAL_THREADING_CONFIG[]
end

"""
    get_threading_config() -> ThreadingConfig

Get current global threading configuration.
"""
get_threading_config() = _GLOBAL_THREADING_CONFIG[]

# =============================================================================
# Proposal Configuration (for MCEM)
# =============================================================================

"""
    ProposalConfig

Configuration for MCEM proposal/importance sampling settings.

# Fields
- `n_paths::Int`: Number of paths to sample per subject
- `ess_threshold::Float64`: Effective sample size threshold for resampling
- `max_retries::Int`: Maximum retry attempts for path sampling
- `verbose::Bool`: Print diagnostic messages
"""
struct ProposalConfig
    n_paths::Int
    ess_threshold::Float64
    max_retries::Int
    verbose::Bool
end

function ProposalConfig(; n_paths::Int=100, ess_threshold::Float64=0.5, 
                         max_retries::Int=3, verbose::Bool=false)
    ProposalConfig(n_paths, ess_threshold, max_retries, verbose)
end

# =============================================================================
# CovariateData Type Alias
# =============================================================================

const CovariateData = Union{NamedTuple, DataFrameRow}
