# Sampling Importance Resampling (SIR) utilities for MCEM
# 
# Implements variance-reduced resampling methods based on:
# - Li (2006): Pool size O(m log m) for well-behaved importance weights
# - LHS-SIR: Latin Hypercube Sampling for variance reduction

"""
    sir_pool_size(ess_target, c, max_pool)

Compute the pool size for SIR as `min(ceil(c * m * log(m)), max_pool)` where `m = ess_target`.

This follows Li (2006) which shows that pool size should be O(m log m) when importance 
weights have a moment generating function.

# Arguments
- `ess_target::Int`: Target effective sample size (the subsample size after resampling)
- `c::Float64`: Pool size constant (default 2.0 in fit())
- `max_pool::Int`: Maximum pool size cap

# Returns
- `Int`: The target pool size
"""
function sir_pool_size(ess_target::Int, c::Float64, max_pool::Int)
    ess_target <= 1 && return max(1, ess_target)
    return min(ceil(Int, c * ess_target * log(ess_target)), max_pool)
end

"""
    resample_multinomial(weights, n_resample)

Resample `n_resample` indices with replacement, proportional to `weights`.

This is standard SIR resampling using multinomial sampling.

# Arguments
- `weights::AbstractVector{<:Real}`: Normalized importance weights (should sum to 1)
- `n_resample::Int`: Number of indices to resample

# Returns
- `Vector{Int}`: Indices into the original array (may contain duplicates)
"""
function resample_multinomial(weights::AbstractVector{<:Real}, n_resample::Int)
    return StatsBase.sample(1:length(weights), StatsBase.Weights(weights), n_resample; replace=true)
end

"""
    resample_lhs(weights, n_resample)

Resample `n_resample` indices using Latin Hypercube Sampling on the CDF.

This variance-reduced resampling method:
1. Divides [0,1] into `n_resample` equal strata
2. Samples one uniform random value per stratum
3. Maps each to an index via the inverse CDF of cumulative weights

This ensures better coverage of the weight distribution compared to multinomial resampling,
reducing the variance of subsequent estimators.

# Arguments
- `weights::AbstractVector{<:Real}`: Normalized importance weights (should sum to ~1)
- `n_resample::Int`: Number of indices to resample

# Returns
- `Vector{Int}`: Indices into the original array (may contain duplicates)
"""
function resample_lhs(weights::AbstractVector{<:Real}, n_resample::Int)
    n_resample <= 0 && return Int[]
    
    # Compute normalized cumulative weights
    cumweights = cumsum(weights)
    total = cumweights[end]
    if total > 0
        cumweights ./= total
    else
        # Fallback to uniform if weights are all zero
        cumweights .= (1:length(weights)) ./ length(weights)
    end
    
    indices = Vector{Int}(undef, n_resample)
    for i in 1:n_resample
        # Sample uniformly in stratum [(i-1)/n, i/n]
        u = (i - 1 + rand()) / n_resample
        # Find index via inverse CDF (binary search)
        indices[i] = searchsortedfirst(cumweights, u)
        # Clamp to valid range (edge case protection)
        indices[i] = clamp(indices[i], 1, length(weights))
    end
    return indices
end

"""
    get_sir_subsample_indices(weights, n_resample, method)

Dispatcher for SIR resampling methods.

# Arguments
- `weights::AbstractVector{<:Real}`: Normalized importance weights
- `n_resample::Int`: Number of indices to resample
- `method::Symbol`: Either `:sir` (multinomial) or `:lhs` (Latin Hypercube)

# Returns
- `Vector{Int}`: Indices into the original array

# Throws
- `ErrorException` if method is not `:sir` or `:lhs`
"""
function get_sir_subsample_indices(weights::AbstractVector{<:Real}, n_resample::Int, method::Symbol)
    if method == :sir
        return resample_multinomial(weights, n_resample)
    elseif method == :lhs
        return resample_lhs(weights, n_resample)
    else
        error("Unknown SIR method: $method. Must be :sir or :lhs")
    end
end

"""
    should_resample(sir_resample, psis_pareto_k, threshold)

Determine whether to resample based on the resampling mode and PSIS diagnostics.

# Arguments
- `sir_resample::Symbol`: Resampling mode (`:always` or `:degeneracy`)
- `psis_pareto_k::Float64`: Pareto-k diagnostic from PSIS (higher = more weight degeneracy)
- `threshold::Float64`: Pareto-k threshold for `:degeneracy` mode (typically 0.7)

# Returns
- `Bool`: `true` if resampling should occur

# Modes
- `:always`: Always resample (returns `true`)
- `:degeneracy`: Only resample if `psis_pareto_k > threshold`
"""
function should_resample(sir_resample::Symbol, psis_pareto_k::Float64, threshold::Float64)
    if sir_resample == :always
        return true
    elseif sir_resample == :degeneracy
        return psis_pareto_k > threshold
    else
        return false
    end
end

# =============================================================================
# MCEM Functions for SIR
# =============================================================================

"""
    mcem_mll_sir(logliks, sir_indices, SubjectWeights)

Compute the marginal log likelihood Q(θ|θ') for MCEM with SIR.

With SIR, we use simple (unweighted) averages on the subsample:
```math
Q(θ|θ') = Σᵢ SubjectWeights[i] × (1/mᵢ) × Σⱼ∈sir_indices[i] logliks[i][j]
```
where mᵢ = length(sir_indices[i]) is the subsample size for subject i.

# Arguments
- `logliks`: Vector of vectors of log-likelihoods per subject
- `sir_indices`: Vector of vectors of indices into logliks for each subject
- `SubjectWeights`: Subject weights

# Returns
- `Float64`: The weighted Q function value
"""
function mcem_mll_sir(logliks, sir_indices, SubjectWeights)
    obj = 0.0
    for i in eachindex(logliks)
        indices = sir_indices[i]
        if !isempty(indices)
            # Simple average over subsampled paths
            obj += mean(@view logliks[i][indices]) * SubjectWeights[i]
        end
    end
    return obj
end

"""
    mcem_ase_sir(loglik_target_prop, loglik_target_cur, sir_indices, SubjectWeights)

Asymptotic standard error of the change in the MCEM objective function with SIR.

With SIR, the ASE is based on the sample variance of the subsample:
```math
Var(ΔQ) ≈ Σᵢ (SubjectWeights[i]² / mᵢ) × Var(Δlᵢⱼ)
```
where Δlᵢⱼ = loglik_target_prop[i][j] - loglik_target_cur[i][j].

# Arguments
- `loglik_target_prop`: Log-likelihoods at proposed parameters
- `loglik_target_cur`: Log-likelihoods at current parameters
- `sir_indices`: SIR subsample indices per subject
- `SubjectWeights`: Subject weights

# Returns
- `Float64`: The asymptotic standard error
"""
function mcem_ase_sir(loglik_target_prop, loglik_target_cur, sir_indices, SubjectWeights)
    VarSum = 0.0
    for i in eachindex(SubjectWeights)
        indices = sir_indices[i]
        m = length(indices)
        if m > 1
            # Compute sample variance of the difference
            diffs = @view(loglik_target_prop[i][indices]) .- @view(loglik_target_cur[i][indices])
            sample_var = var(diffs)
            # Variance of the mean
            VarSum += (SubjectWeights[i]^2 / m) * sample_var
        end
    end
    return sqrt(VarSum)
end

"""
    create_sir_subsampled_data(model, samplepaths, sir_indices)

Create subsampled path and weight arrays for SIR-based M-step.

This creates **views** into the original paths for memory efficiency.
The returned importance weights are uniform (1/m for m subsampled paths).

# Arguments
- `model`: The multistate model
- `samplepaths`: Full pool of sample paths per subject
- `sir_indices`: Indices into samplepaths for each subject

# Returns
- Named tuple `(paths, weights)` with subsampled paths and uniform weights
"""
function create_sir_subsampled_data(samplepaths, sir_indices)
    nsubj = length(samplepaths)
    
    # Create vectors of subsampled paths (using views for efficiency)
    subsampled_paths = Vector{Vector{SamplePath}}(undef, nsubj)
    uniform_weights = Vector{Vector{Float64}}(undef, nsubj)
    
    for i in 1:nsubj
        indices = sir_indices[i]
        m = length(indices)
        
        if m == 0
            # No paths (shouldn't happen, but handle gracefully)
            subsampled_paths[i] = SamplePath[]
            uniform_weights[i] = Float64[]
        else
            # Subsample paths (copy - views don't work well with nested vectors)
            subsampled_paths[i] = samplepaths[i][indices]
            # Uniform weights
            uniform_weights[i] = fill(1.0 / m, m)
        end
    end
    
    return (paths = subsampled_paths, weights = uniform_weights)
end
