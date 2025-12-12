# =============================================================================
# MCEM Helper Functions
# =============================================================================
#
# This module provides helper functions for Monte Carlo EM (MCEM) estimation
# of semi-Markov multistate models. The MCEM algorithm iterates between:
#
#   E-step: Estimate E[Q(θ|θ') | Y] via importance sampling from Markov surrogate
#   M-step: Maximize Q(θ|θ') = Σᵢ Σⱼ wᵢⱼ ℓᵢⱼ(θ) w.r.t. θ
#
# where wᵢⱼ are normalized importance weights and ℓᵢⱼ is the complete-data
# log-likelihood for path j of subject i.
#
# SQUAREM acceleration (Varadhan & Roland, 2008) can be optionally enabled to
# speed up convergence by treating the EM mapping as a fixed-point iteration
# and applying quasi-Newton acceleration.
#
# # References
#
# - Morsomme, R., Liang, C. J., Mateja, A., Follmann, D. A., O'Brien, M. P., Wang, C.,
#   & Fintzi, J. (2025). Assessing treatment efficacy for interval-censored endpoints
#   using multistate semi-Markov models fit to multiple data streams. Biostatistics,
#   26(1), kxaf038. https://doi.org/10.1093/biostatistics/kxaf038
# - Wei, G. C., & Tanner, M. A. (1990). A Monte Carlo implementation of the 
#   EM algorithm and the poor man's data augmentation algorithms. JASA, 85(411), 699-704.
# - Caffo, B. S., Jank, W., & Jones, G. L. (2005). Ascent-based Monte Carlo 
#   expectation-maximization. JRSS-B, 67(2), 235-251.
# - Varadhan, R., & Roland, C. (2008). Simple and globally convergent methods for 
#   accelerating the convergence of any EM algorithm. Scandinavian Journal of 
#   Statistics, 35(2), 335-353.
# - Zhou, H., Alexander, D., & Lange, K. (2011). A quasi-Newton acceleration for
#   high-dimensional optimization algorithms. Statistics and Computing, 21(2), 261-273.
#
# =============================================================================

"""
    mcem_mll(logliks, ImportanceWeights, SubjectWeights)

Compute the marginal log likelihood Q(θ|θ') for MCEM.

This is the importance-weighted expected complete-data log-likelihood:
```math
Q(θ|θ') = Σᵢ SubjectWeights[i] × Σⱼ ImportanceWeights[i][j] × logliks[i][j]
```

See also: [`mcem_ase`](@ref), [`mcem_lml`](@ref)
"""
function mcem_mll(logliks, ImportanceWeights, SubjectWeights)

    obj = 0.0
    
    for i in eachindex(logliks)
        obj += dot(logliks[i], ImportanceWeights[i]) * SubjectWeights[i]
    end

    return obj
end

"""
    mcem_lml(logliks, ImportanceWeights, SubjectWeights)

Compute the log marginal likelihood for MCEM.
"""
function mcem_lml(logliks, ImportanceWeights, SubjectWeights)

    obj = 0.0
    
    for i in eachindex(logliks)
        obj += log(dot(exp.(logliks[i]), ImportanceWeights[i])) * SubjectWeights[i]
    end

    return obj
end

"""
    mcem_lml_subj(logliks, ImportanceWeights)

Compute the log marginal likelihood of each subject for MCEM.
"""
function mcem_lml_subj(logliks, ImportanceWeights)

    subj_lml = zeros(length(logliks))
    
    for i in eachindex(logliks)
        subj_lml[i] = log(sum(exp.(logliks[i]) .* ImportanceWeights[i])) 
    end

    return subj_lml
end

"""
    var_ris(l, w)

Helper to compute the variance of the ratio of two means estimated via importance sampling. 

# Arguments:

- l: For MCEM, the difference in log-likelihoods
- w: Normalized importance weights
"""
function var_ris(l, w)

    all(isapprox.(l, 0.0)) ? 0.0 : (sum(l .* w)) ^ 2 * ((sum((w .* l).^2)) / sum(w .* l) ^2 - 2 * (sum(w.^2 .* l) / (sum(w .* l))) + sum(w .^ 2)) 
    
end

"""
    mcem_ase(loglik_target_prop, loglik_target_cur, ImportanceWeights, SubjectWeights)

Asymptotic standard error of the change in the MCEM objective function.
"""
function mcem_ase(loglik_target_prop, loglik_target_cur, ImportanceWeights, SubjectWeights)

    VarRis = 0.0
    for i in eachindex(SubjectWeights)
        if length(ImportanceWeights[i]) != 1
            VarRis += var_ris(loglik_target_prop[i] - loglik_target_cur[i], ImportanceWeights[i]) * SubjectWeights[i]^2
        end
    end

    # return the asymptotic standard error
    sqrt(VarRis)
end

# =============================================================================
# SQUAREM Acceleration for MCEM
# =============================================================================
#
# SQUAREM (Squared Iterative Methods) accelerates fixed-point iterations by 
# treating θ_{k+1} = M(θ_k) as a fixed-point mapping and applying a quasi-Newton
# step. For MCEM, M(θ) is one complete EM iteration (E-step + M-step).
#
# The algorithm:
# 1. Compute θ₁ = M(θ₀)           (first EM step)
# 2. Compute θ₂ = M(θ₁)           (second EM step)
# 3. r = θ₁ - θ₀                  (first increment)
# 4. v = (θ₂ - θ₁) - r            (second increment minus first)
# 5. α = -‖r‖/‖v‖                 (step length)
# 6. θ_acc = θ₀ - 2αr + α²v       (accelerated update)
#
# If the accelerated step decreases the objective, we may need to backtrack
# or fall back to the standard EM update.
#
# For MCEM specifically, the Monte Carlo noise means we use a stabilized version
# that clamps the step length and uses monotone backtracking when needed.
#
# Reference: Varadhan & Roland (2008), JSS 92(7)
# =============================================================================

"""
    SquaremState

State container for SQUAREM acceleration in MCEM.

# Fields
- `θ0::Vector{Float64}`: Starting parameters for SQUAREM cycle
- `θ1::Vector{Float64}`: Parameters after first EM step
- `θ2::Vector{Float64}`: Parameters after second EM step
- `step::Int`: Current step in SQUAREM cycle (0, 1, or 2)
- `α::Float64`: Computed step length
- `n_accelerations::Int`: Number of successful accelerations
- `n_fallbacks::Int`: Number of fallbacks to standard EM
"""
mutable struct SquaremState
    θ0::Vector{Float64}
    θ1::Vector{Float64}
    θ2::Vector{Float64}
    step::Int
    α::Float64
    n_accelerations::Int
    n_fallbacks::Int
end

"""
    SquaremState(nparams::Int)

Initialize SQUAREM state for a parameter vector of length `nparams`.
"""
function SquaremState(nparams::Int)
    SquaremState(
        zeros(nparams),
        zeros(nparams),
        zeros(nparams),
        0,
        0.0,
        0,
        0
    )
end

"""
    squarem_step_length(θ0, θ1, θ2; α_min=-1.0, α_max=-1e-4)

Compute the SQUAREM step length α from the parameter sequence.

The step length is computed as:
    α = -‖r‖ / ‖v‖
where r = θ₁ - θ₀ and v = (θ₂ - θ₁) - r.

# Arguments
- `θ0`, `θ1`, `θ2`: Parameter vectors from two EM iterations
- `α_min`, `α_max`: Bounds on the step length (negative values)

# Returns
- `α::Float64`: The clamped step length
- `r::Vector{Float64}`: First increment θ₁ - θ₀
- `v::Vector{Float64}`: Second-order term (θ₂ - θ₁) - r
"""
function squarem_step_length(θ0::Vector{Float64}, θ1::Vector{Float64}, θ2::Vector{Float64};
                             α_min::Float64=-1.0, α_max::Float64=-1e-4)
    r = θ1 .- θ0           # First increment
    v = (θ2 .- θ1) .- r    # Second-order term
    
    r_norm_sq = sum(abs2, r)
    v_norm_sq = sum(abs2, v)
    
    # Avoid division by zero
    if v_norm_sq < eps(Float64) * r_norm_sq || r_norm_sq < eps(Float64)
        # No acceleration possible; return fallback indicator
        return -1.0, r, v
    end
    
    # Compute step length
    α = -sqrt(r_norm_sq / v_norm_sq)
    
    # Clamp to valid range
    α = clamp(α, α_min, α_max)
    
    return α, r, v
end

"""
    squarem_accelerate(θ0, r, v, α)

Compute the accelerated parameter update.

# Arguments
- `θ0`: Starting parameters
- `r`: First increment (θ₁ - θ₀)
- `v`: Second-order term ((θ₂ - θ₁) - r)
- `α`: Step length (typically negative)

# Returns
- `θ_acc::Vector{Float64}`: Accelerated parameters θ₀ - 2αr + α²v
"""
function squarem_accelerate(θ0::Vector{Float64}, r::Vector{Float64}, 
                            v::Vector{Float64}, α::Float64)
    # θ_acc = θ₀ - 2αr + α²v
    return θ0 .- 2 * α .* r .+ α^2 .* v
end

"""
    squarem_should_accept(mll_acc, mll_θ2, mll_θ0; tol=0.0)

Determine whether to accept the accelerated update.

For MCEM, we accept the accelerated step if it improves the marginal 
log-likelihood compared to the starting point. Due to Monte Carlo noise,
we may also accept if it's close to θ₂'s objective.

# Arguments
- `mll_acc`: Marginal log-likelihood at accelerated point
- `mll_θ2`: Marginal log-likelihood at θ₂ (standard EM)
- `mll_θ0`: Marginal log-likelihood at starting point

# Returns
- `accept::Bool`: Whether to accept the accelerated update
"""
function squarem_should_accept(mll_acc::Float64, mll_θ2::Float64, mll_θ0::Float64;
                               tol::Float64=0.0)
    # Accept if accelerated point is better than starting point
    # (allowing small tolerance for Monte Carlo noise)
    return mll_acc >= mll_θ0 - tol
end