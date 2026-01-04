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
const _GLOBAL_THREADING_CONFIG = Ref(ThreadingConfig(parallel=false))

"""
    set_threading_config!(; parallel=false, nthreads=nothing, min_batch_size=10)

Set global threading configuration for likelihood evaluation.

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
# Spline Penalty Configuration Types
# =============================================================================
#
# User-facing and internal types for penalized spline hazards.
# See scratch/penalized_splines_plan.md for design documentation.
#
# =============================================================================

"""
    SplinePenalty

User-facing configuration for spline penalties in baseline hazards and smooth covariates.

Penalties are configured at the model level using a rule-based API.
Rules are applied from general to specific: transition > origin > global.

# Arguments
- `selector`: Which hazards this rule applies to.
  - `:all` — All spline hazards (default for global settings)
  - `r::Int` — All hazards from origin state `r`
  - `(r, s)::Tuple{Int,Int}` — Specific transition `r → s`
- `order::Int=2`: Derivative to penalize (1=slope, 2=curvature, 3=change in curvature)
- `total_hazard::Bool=false`: Penalize smoothness of total hazard out of this origin
- `share_lambda::Bool=false`: Share λ across competing hazards from same origin
- `share_covariate_lambda::Union{Bool, Symbol}=false`: Sharing mode for smooth covariate λ
  - `false` — Separate λ per smooth term (default)
  - `:hazard` — Share λ across all s() terms within each hazard
  - `:global` — Share λ across all s() terms in the model

# Examples
```julia
# Global curvature penalty (default)
penalty = SplinePenalty()

# Stiffer penalty (3rd derivative) for all hazards
penalty = SplinePenalty(order=3)

# Origin 1: shared λ across competing risks
penalty = SplinePenalty(1, share_lambda=true)

# Transition 1→2: first derivative (flatness) penalty
penalty = SplinePenalty((1, 2), order=1)

# Share smooth covariate λ within each hazard
penalty = SplinePenalty(share_covariate_lambda=:hazard)

# Share smooth covariate λ globally across all hazards
penalty = SplinePenalty(share_covariate_lambda=:global)
```

# Notes
- When `share_lambda=true` or `total_hazard=true`, all spline hazards from that 
  origin MUST have identical knot locations. The constructor validates this.
- For competing risks, the likelihood-motivated decomposition suggests separate
  control of total hazard and deviation penalties. Use `total_hazard=true` to
  enable the additional total hazard penalty term.
- Smooth covariate sharing (`share_covariate_lambda`) is independent of baseline
  hazard sharing (`share_lambda`).

See also: [`PenaltyConfig`](@ref), [`SplineHazardInfo`](@ref)
"""
struct SplinePenalty
    selector::Union{Symbol, Int, Tuple{Int,Int}}
    order::Int
    total_hazard::Bool
    share_lambda::Bool
    share_covariate_lambda::Union{Bool, Symbol}
    
    function SplinePenalty(selector::Union{Symbol, Int, Tuple{Int,Int}}=:all;
                            order::Int=2, total_hazard::Bool=false, 
                            share_lambda::Bool=false,
                            share_covariate_lambda::Union{Bool, Symbol}=false)
        order >= 1 || throw(ArgumentError("order must be ≥ 1, got $order"))
        
        # Validate selector type
        if selector isa Symbol && selector != :all
            throw(ArgumentError("Symbol selector must be :all, got :$selector"))
        end
        if selector isa Int && selector < 1
            throw(ArgumentError("Origin state selector must be ≥ 1, got $selector"))
        end
        if selector isa Tuple{Int,Int}
            selector[1] >= 1 && selector[2] >= 1 || 
                throw(ArgumentError("Transition selector states must be ≥ 1, got $selector"))
        end
        
        # Validate share_covariate_lambda
        if share_covariate_lambda isa Symbol && share_covariate_lambda ∉ (:hazard, :global)
            throw(ArgumentError("share_covariate_lambda must be false, :hazard, or :global, got :$share_covariate_lambda"))
        end
        
        new(selector, order, total_hazard, share_lambda, share_covariate_lambda)
    end
end

"""
    PenaltyTerm

Internal representation of a single penalty term in the penalized likelihood.

# Fields
- `hazard_indices::UnitRange{Int}`: Indices into flat parameter vector
- `S::Matrix{Float64}`: Penalty matrix (K × K)
- `lambda::Float64`: Current smoothing parameter (log-scale internally)
- `order::Int`: Derivative order
- `hazard_names::Vector{Symbol}`: Names of hazards covered by this term
"""
struct PenaltyTerm
    hazard_indices::UnitRange{Int}
    S::Matrix{Float64}
    lambda::Float64
    order::Int
    hazard_names::Vector{Symbol}
end

"""
    TotalHazardPenaltyTerm

Internal representation of a total hazard penalty term (for competing risks).

The total hazard penalty penalizes the curvature of H(t) = Σₖ hₖ(t) where
the sum is over all competing hazards from the same origin state.

# Fields
- `origin::Int`: Origin state for this penalty
- `hazard_indices::Vector{UnitRange{Int}}`: Indices for each competing hazard
- `S::Matrix{Float64}`: Base penalty matrix (shared across hazards)
- `lambda_H::Float64`: Smoothing parameter for total hazard
- `order::Int`: Derivative order
"""
struct TotalHazardPenaltyTerm
    origin::Int
    hazard_indices::Vector{UnitRange{Int}}
    S::Matrix{Float64}
    lambda_H::Float64
    order::Int
end

"""
    SmoothCovariatePenaltyTerm

Internal representation of a smooth covariate penalty term (for s(x) terms).

# Fields
- `param_indices::Vector{Int}`: Indices into flat parameter vector (may not be contiguous)
- `S::Matrix{Float64}`: Penalty matrix (K × K)
- `lambda::Float64`: Smoothing parameter
- `order::Int`: Derivative order (from smooth term)
- `label::String`: Label for the smooth term (e.g., "s(age)")
- `hazard_name::Symbol`: Name of hazard containing this smooth term
"""
struct SmoothCovariatePenaltyTerm
    param_indices::Vector{Int}
    S::Matrix{Float64}
    lambda::Float64
    order::Int
    label::String
    hazard_name::Symbol
end

"""
    PenaltyConfig

Internal representation of resolved penalty configuration for a model.

Created by resolving `SplinePenalty` rules against model structure.
Used during likelihood evaluation to compute penalty contribution.

# Fields
- `terms::Vector{PenaltyTerm}`: Individual baseline hazard penalty terms
- `total_hazard_terms::Vector{TotalHazardPenaltyTerm}`: Total hazard penalties
- `smooth_covariate_terms::Vector{SmoothCovariatePenaltyTerm}`: Smooth covariate penalties
- `shared_lambda_groups::Dict{Int, Vector{Int}}`: Origin → indices of shared-λ terms (baseline)
- `shared_smooth_groups::Vector{Vector{Int}}`: Groups of smooth covariate term indices sharing λ
- `n_lambda::Int`: Total number of smoothing parameters

# Notes
The penalty contribution to negative log-likelihood is:
    P(β; λ) = (1/2) Σⱼ λⱼ βⱼᵀ Sⱼ βⱼ + (1/2) Σᵣ λ_H,r βᵣᵀ (𝟙𝟙ᵀ ⊗ S) βᵣ + (1/2) Σₖ λₖ βₖᵀ Sₖ βₖ
"""
struct PenaltyConfig
    terms::Vector{PenaltyTerm}
    total_hazard_terms::Vector{TotalHazardPenaltyTerm}
    smooth_covariate_terms::Vector{SmoothCovariatePenaltyTerm}
    shared_lambda_groups::Dict{Int, Vector{Int}}
    shared_smooth_groups::Vector{Vector{Int}}
    n_lambda::Int
    
    function PenaltyConfig(terms::Vector{PenaltyTerm},
                            total_hazard_terms::Vector{TotalHazardPenaltyTerm},
                            smooth_covariate_terms::Vector{SmoothCovariatePenaltyTerm},
                            shared_lambda_groups::Dict{Int, Vector{Int}},
                            shared_smooth_groups::Vector{Vector{Int}},
                            n_lambda::Int)
        n_lambda >= 0 || throw(ArgumentError("n_lambda must be ≥ 0"))
        new(terms, total_hazard_terms, smooth_covariate_terms, shared_lambda_groups, shared_smooth_groups, n_lambda)
    end
    
    # 5-argument constructor (no shared_smooth_groups) for backward compatibility
    function PenaltyConfig(terms::Vector{PenaltyTerm},
                            total_hazard_terms::Vector{TotalHazardPenaltyTerm},
                            smooth_covariate_terms::Vector{SmoothCovariatePenaltyTerm},
                            shared_lambda_groups::Dict{Int, Vector{Int}},
                            n_lambda::Int)
        n_lambda >= 0 || throw(ArgumentError("n_lambda must be ≥ 0"))
        new(terms, total_hazard_terms, smooth_covariate_terms, shared_lambda_groups, Vector{Int}[], n_lambda)
    end
    
    # Legacy 4-argument constructor for backward compatibility
    function PenaltyConfig(terms::Vector{PenaltyTerm},
                            total_hazard_terms::Vector{TotalHazardPenaltyTerm},
                            shared_lambda_groups::Dict{Int, Vector{Int}},
                            n_lambda::Int)
        n_lambda >= 0 || throw(ArgumentError("n_lambda must be ≥ 0"))
        new(terms, total_hazard_terms, SmoothCovariatePenaltyTerm[], shared_lambda_groups, Vector{Int}[], n_lambda)
    end
end

"""
    PenaltyConfig()

Create an empty penalty configuration (no penalties).
"""
PenaltyConfig() = PenaltyConfig(PenaltyTerm[], TotalHazardPenaltyTerm[], SmoothCovariatePenaltyTerm[], Dict{Int,Vector{Int}}(), Vector{Int}[], 0)

"""
    has_penalties(config::PenaltyConfig) -> Bool

Check if the penalty configuration contains any active penalty terms.
"""
has_penalties(config::PenaltyConfig) = !isempty(config.terms) || !isempty(config.total_hazard_terms) || !isempty(config.smooth_covariate_terms)

"""
    compute_penalty(beta::AbstractVector, config::PenaltyConfig) -> Float64

Compute the total penalty contribution for given coefficients.

# Arguments
- `beta`: Coefficient vector on natural scale (not log-transformed)
- `config`: Resolved penalty configuration

# Returns
Penalty value: (1/2) Σⱼ λⱼ βⱼᵀ Sⱼ βⱼ + total hazard penalties + smooth covariate penalties

# Notes
- Coefficients must be on natural scale (positive for hazard splines)
- Returns 0.0 if config has no penalties
"""
function compute_penalty(beta::AbstractVector{T}, config::PenaltyConfig) where T
    has_penalties(config) || return zero(T)
    
    penalty = zero(T)
    
    # Individual baseline hazard penalties
    for term in config.terms
        β_j = @view beta[term.hazard_indices]
        penalty += term.lambda * dot(β_j, term.S * β_j)
    end
    
    # Total hazard penalties (if any)
    for term in config.total_hazard_terms
        # Sum coefficients across competing hazards for total hazard
        K = size(term.S, 1)
        β_total = zeros(T, K)
        for idx_range in term.hazard_indices
            β_total .+= @view beta[idx_range]
        end
        penalty += term.lambda_H * dot(β_total, term.S * β_total)
    end
    
    # Smooth covariate penalties
    for term in config.smooth_covariate_terms
        β_k = beta[term.param_indices]  # Use Vector indexing (may not be contiguous)
        penalty += term.lambda * dot(β_k, term.S * β_k)
    end
    
    return penalty / 2
end