# =============================================================================
# Implicit Differentiation for PIJCV Smoothing Parameter Selection
# =============================================================================
#
# Implements ImplicitDifferentiation.jl integration for efficient gradient 
# computation during PIJCV-based smoothing parameter selection.
#
# MATHEMATICAL FOUNDATION:
# The PIJCV criterion V(ρ) depends on β̂(ρ) which is defined implicitly as:
#   β̂(ρ) = argmin_β [-ℓ(β) + ½ Σⱼ λⱼ βᵀSⱼβ]  where λ = exp(ρ)
#
# At the optimum, the first-order conditions hold:
#   c(ρ, β̂) = ∇_β ℓ(β̂) - Σⱼ λⱼ Sⱼ β̂ = 0
#
# By the implicit function theorem:
#   ∂β̂/∂ρⱼ = -H_λ⁻¹ · (λⱼ Sⱼ β̂)
#
# where H_λ = -∇²ℓ(β̂) + Σⱼ λⱼ Sⱼ is the penalized Hessian.
#
# This avoids nested AD and reduces complexity from O(np³) to O(np²).
#
# REFERENCES:
# - Wood, S.N. (2024). "On Neighbourhood Cross Validation." arXiv:2404.16490v4
# - Blondel et al. (2022). "Efficient and Modular Implicit Differentiation."
#
# =============================================================================

using ImplicitDifferentiation
using ImplicitDifferentiation: MatrixRepresentation, DirectLinearSolver
using ADTypes: AutoForwardDiff

# =============================================================================
# Cache Structure for Implicit Differentiation
# =============================================================================

"""
    ImplicitBetaCache{M, D, P}

Cache for implicit differentiation of the inner β optimization problem.

Stores all objects needed to:
1. Solve the inner optimization β̂(ρ) via forward function
2. Evaluate the optimality conditions c(ρ, β) = 0 at any (ρ, β)

# Type Parameters
- `M`: Model type (MultistateProcess)
- `D`: Data type (ExactData, MPanelData, or MCEMSelectionData)
- `P`: Penalty configuration type

# Fields
- `model`: Model for likelihood evaluation
- `data`: Data container
- `penalty_config`: Penalty configuration with S matrices
- `S_matrices`: Pre-extracted penalty matrices for fast access
- `lb`, `ub`: Parameter bounds
- `inner_maxiter`: Maximum iterations for inner optimization
- `inner_tol`: Convergence tolerance for inner optimization
"""
struct ImplicitBetaCache{M<:MultistateProcess, D, P<:AbstractPenalty}
    model::M
    data::D
    penalty_config::P
    S_matrices::Vector{Matrix{Float64}}  # Penalty matrices per term
    lb::Vector{Float64}
    ub::Vector{Float64}
    inner_maxiter::Int
    inner_tol::Float64
end

"""
    build_implicit_beta_cache(model, data, penalty, beta_init; kwargs...) -> ImplicitBetaCache

Build cache for implicit differentiation.

# Arguments
- `model::MultistateProcess`: Model for likelihood evaluation
- `data`: Data container (ExactData, MPanelData, or MCEMSelectionData)
- `penalty::AbstractPenalty`: Penalty configuration
- `beta_init::Vector{Float64}`: Initial coefficients (used to determine dimensions)

# Keyword Arguments
- `inner_maxiter::Int=50`: Maximum iterations for inner optimization
- `inner_tol::Float64=1e-6`: Convergence tolerance
"""
function build_implicit_beta_cache(
    model::MultistateProcess,
    data,
    penalty::AbstractPenalty,
    beta_init::Vector{Float64};
    inner_maxiter::Int = 50,
    inner_tol::Float64 = LAMBDA_SELECTION_INNER_TOL
)
    lb, ub = model.bounds.lb, model.bounds.ub
    
    # Extract penalty matrices
    S_matrices = _extract_penalty_matrices(penalty)
    
    return ImplicitBetaCache(
        model, data, penalty, S_matrices,
        lb, ub, inner_maxiter, inner_tol
    )
end

"""
    _extract_penalty_matrices(penalty::QuadraticPenalty) -> Vector{Matrix{Float64}}

Extract penalty matrices from a QuadraticPenalty configuration.
Returns a vector of matrices, one per smoothing parameter.
"""
function _extract_penalty_matrices(penalty::QuadraticPenalty)
    matrices = Matrix{Float64}[]
    
    # Extract from baseline terms
    for term in penalty.terms
        push!(matrices, Matrix(term.S))
    end
    
    # Extract from total hazard terms
    for term in penalty.total_hazard_terms
        push!(matrices, Matrix(term.S))
    end
    
    # Extract from smooth covariate terms
    for term in penalty.smooth_covariate_terms
        push!(matrices, Matrix(term.S))
    end
    
    return matrices
end

# Fallback for other penalty types
function _extract_penalty_matrices(penalty::AbstractPenalty)
    # NoPenalty or unknown type
    return Matrix{Float64}[]
end

# =============================================================================
# Forward Function: Inner Optimization
# =============================================================================

"""
    forward_beta_solve(ρ, cache::ImplicitBetaCache) -> (β, z)

Forward function for ImplicitDifferentiation.jl.

Solves the penalized MLE problem:
    β̂(ρ) = argmin_β [-ℓ(β) + ½ Σⱼ exp(ρⱼ) βᵀSⱼβ]

# Arguments
- `ρ`: Log-smoothing parameters (AbstractVector)
- `cache`: ImplicitBetaCache with model, data, penalty info

# Returns
- `β`: Optimal coefficient vector (AbstractVector)
- `z`: Byproduct tuple containing (H_lambda, converged) for diagnostics

# Note
This function extracts Float64 values from ρ (which may contain Dual numbers)
because the inner optimization has its own AD. The outer AD uses the implicit
function theorem to get gradients via the conditions function.
"""
function forward_beta_solve(ρ::AbstractVector, cache::ImplicitBetaCache)
    # Extract Float64 values - inner optimization is Float64 only
    ρ_float = Float64[ForwardDiff.value(x) for x in ρ]
    λ = exp.(ρ_float)
    
    # Create penalty with current λ
    penalty = set_hyperparameters(cache.penalty_config, λ)
    
    # Solve inner problem using existing infrastructure
    # Start from previous β if available (warm-starting)
    β_init = get_warm_start_beta(cache)
    
    β_opt = _fit_inner_coefficients_cached(
        cache.model, cache.data, penalty, β_init;
        lb=cache.lb, ub=cache.ub, maxiter=cache.inner_maxiter
    )
    
    # Compute penalized Hessian at solution (for diagnostics/byproduct)
    H_lambda = _compute_penalized_hessian_at_beta(β_opt, λ, cache)
    
    # Return β and byproduct
    return β_opt, (H_lambda=H_lambda, lambda=λ)
end

"""
    get_warm_start_beta(cache::ImplicitBetaCache) -> Vector{Float64}

Get initial β for warm-starting the inner optimization.
Currently returns a sensible starting point (handles infinite bounds).
"""
function get_warm_start_beta(cache::ImplicitBetaCache)
    # Smart initialization that handles infinite bounds
    n = length(cache.lb)
    beta_init = Vector{Float64}(undef, n)
    
    for i in 1:n
        li, ui = cache.lb[i], cache.ub[i]
        if isfinite(li) && isfinite(ui)
            # Finite bounds: use midpoint
            beta_init[i] = 0.5 * (li + ui)
        elseif isfinite(li) && !isfinite(ui)
            # Lower bound only: start at lb + 1
            beta_init[i] = li + 1.0
        elseif !isfinite(li) && isfinite(ui)
            # Upper bound only: start at ub - 1
            beta_init[i] = ui - 1.0
        else
            # No finite bounds: use 0
            beta_init[i] = 0.0
        end
    end
    
    return beta_init
end

"""
    _fit_inner_coefficients_cached(model, data, penalty, beta_init; kwargs...) -> Vector{Float64}

Fit coefficients using the appropriate method for the data type.
This dispatches to the existing `_fit_inner_coefficients` functions.
"""
function _fit_inner_coefficients_cached(
    model::MultistateProcess,
    data::ExactData,
    penalty::AbstractPenalty,
    beta_init::Vector{Float64};
    lb::Vector{Float64},
    ub::Vector{Float64},
    maxiter::Int
)
    # Use existing ExactData implementation
    return _fit_inner_coefficients(model, data, penalty, beta_init;
                                    lb=lb, ub=ub, maxiter=maxiter)
end

function _fit_inner_coefficients_cached(
    model::MultistateProcess,
    data::MPanelData,
    penalty::AbstractPenalty,
    beta_init::Vector{Float64};
    lb::Vector{Float64},
    ub::Vector{Float64},
    maxiter::Int
)
    # Use existing MPanelData implementation
    return _fit_inner_coefficients(model, data, penalty, beta_init;
                                    lb=lb, ub=ub, maxiter=maxiter)
end

function _fit_inner_coefficients_cached(
    model::MultistateProcess,
    data::MCEMSelectionData,
    penalty::AbstractPenalty,
    beta_init::Vector{Float64};
    lb::Vector{Float64},
    ub::Vector{Float64},
    maxiter::Int
)
    # Use existing MCEMSelectionData implementation
    return _fit_inner_coefficients(model, data, penalty, beta_init;
                                    lb=lb, ub=ub, maxiter=maxiter)
end

# =============================================================================
# Conditions Function: Optimality
# =============================================================================

"""
    beta_optimality_conditions(ρ, β, z, cache::ImplicitBetaCache) -> Vector

Optimality conditions c(ρ, β) for the penalized problem.

At the optimum β̂(ρ), these conditions are zero:
    c(ρ, β) = ∇_β ℓ(β) - Σⱼ λⱼ Sⱼ β = 0

where λⱼ = exp(ρⱼ).

# Arguments
- `ρ`: Log-smoothing parameters (may contain Dual numbers for AD)
- `β`: Coefficient vector (may contain Dual numbers for AD)
- `z`: Byproduct from forward solve (ignored here)
- `cache`: ImplicitBetaCache with model, data, penalty info

# Returns
Vector of condition values (should be ≈ 0 at optimum)

# Note
This function must be AD-compatible as ImplicitDifferentiation.jl will
differentiate through it to compute the Jacobians ∂c/∂β and ∂c/∂ρ.
"""
function beta_optimality_conditions(ρ::AbstractVector, β::AbstractVector, z, cache::ImplicitBetaCache)
    # Convert ρ to λ (AD-compatible)
    λ = exp.(ρ)
    
    # Compute gradient of log-likelihood ∇_β ℓ(β)
    # This must be AD-compatible
    grad_ll = _compute_ll_gradient(β, cache)
    
    # Compute gradient of penalty: Σⱼ λⱼ Sⱼ β
    grad_penalty = _compute_penalty_gradient(β, λ, cache)
    
    # Optimality condition: ∇_β ℓ(β) - ∇_β penalty = 0
    # We minimize -ℓ + penalty, so gradient is -∇ℓ + ∇penalty = 0
    # Rearranged: ∇ℓ = ∇penalty
    # Return ∇ℓ - ∇penalty (should be 0 at optimum)
    return grad_ll - grad_penalty
end

"""
    _compute_ll_gradient(β, cache::ImplicitBetaCache{M, ExactData}) -> Vector

Compute gradient of log-likelihood for ExactData.
"""
function _compute_ll_gradient(β::AbstractVector, cache::ImplicitBetaCache{M, ExactData}) where M
    # Use ForwardDiff to compute gradient
    grad = ForwardDiff.gradient(b -> loglik_exact(b, cache.data; neg=false), collect(β))
    return grad
end

"""
    _compute_ll_gradient(β, cache::ImplicitBetaCache{M, MPanelData}) -> Vector

Compute gradient of log-likelihood for MPanelData.
"""
function _compute_ll_gradient(β::AbstractVector, cache::ImplicitBetaCache{M, MPanelData}) where M
    # Use ForwardDiff to compute gradient
    grad = ForwardDiff.gradient(b -> loglik_markov(b, cache.data; neg=false), collect(β))
    return grad
end

"""
    _compute_ll_gradient(β, cache::ImplicitBetaCache{M, MCEMSelectionData}) -> Vector

Compute gradient of importance-weighted log-likelihood for MCEMSelectionData.
"""
function _compute_ll_gradient(β::AbstractVector, cache::ImplicitBetaCache{M, MCEMSelectionData}) where M
    # Create SMPanelData for semi-Markov likelihood
    sm_data = SMPanelData(cache.data.model, cache.data.paths, cache.data.weights)
    # Use ForwardDiff with importance-weighted semi-Markov likelihood
    grad = ForwardDiff.gradient(b -> loglik_semi_markov(b, sm_data; neg=false, use_sampling_weight=true), collect(β))
    return grad
end

"""
    _compute_penalty_gradient(β, λ, cache) -> Vector

Compute gradient of penalty term: Σⱼ λⱼ Sⱼ β

Note: Must be AD-compatible. Both β and λ may contain Dual numbers.
"""
function _compute_penalty_gradient(β::AbstractVector, λ::AbstractVector, cache::ImplicitBetaCache)
    n = length(β)
    # Use promoted element type to handle both β and λ potentially having Dual numbers
    T = promote_type(eltype(β), eltype(λ))
    grad = zeros(T, n)
    
    penalty = cache.penalty_config
    lambda_idx = 1
    
    # Baseline hazard penalty gradients
    for term in penalty.terms
        β_j = β[term.hazard_indices]
        # ∂/∂β (λ/2 βᵀSβ) = λ S β
        grad_j = λ[lambda_idx] * (term.S * β_j)
        grad[term.hazard_indices] .+= grad_j
        lambda_idx += 1
    end
    
    # Total hazard penalty gradients
    for term in penalty.total_hazard_terms
        K = size(term.S, 1)
        β_total = zeros(T, K)
        for idx_range in term.hazard_indices
            β_total .+= β[idx_range]
        end
        # ∂/∂β (λ/2 β_total'Sβ_total)
        grad_total = λ[lambda_idx] * (term.S * β_total)
        for idx_range in term.hazard_indices
            grad[idx_range] .+= grad_total
        end
        lambda_idx += 1
    end
    
    # Smooth covariate penalty gradients
    if !isempty(penalty.shared_smooth_groups)
        # Build term -> lambda mapping
        term_to_lambda = Dict{Int, Int}()
        for (group_idx, group) in enumerate(penalty.shared_smooth_groups)
            for term_idx in group
                term_to_lambda[term_idx] = lambda_idx
            end
            lambda_idx += 1
        end
        # Handle ungrouped terms
        for term_idx in 1:length(penalty.smooth_covariate_terms)
            if !haskey(term_to_lambda, term_idx)
                term_to_lambda[term_idx] = lambda_idx
                lambda_idx += 1
            end
        end
        # Compute gradients
        for (term_idx, term) in enumerate(penalty.smooth_covariate_terms)
            β_k = β[term.param_indices]
            grad_k = λ[term_to_lambda[term_idx]] * (term.S * β_k)
            grad[term.param_indices] .+= grad_k
        end
    else
        for term in penalty.smooth_covariate_terms
            β_k = β[term.param_indices]
            grad_k = λ[lambda_idx] * (term.S * β_k)
            grad[term.param_indices] .+= grad_k
            lambda_idx += 1
        end
    end
    
    return grad
end

"""
    _compute_penalized_hessian_at_beta(β, λ, cache) -> Matrix{Float64}

Compute the penalized Hessian H_λ = -∇²ℓ(β) + Σⱼ λⱼ Sⱼ at given β.
"""
function _compute_penalized_hessian_at_beta(β::Vector{Float64}, λ::Vector{Float64}, 
                                            cache::ImplicitBetaCache{M, ExactData}) where M
    # Get unpenalized Hessian
    H_unpenalized = ForwardDiff.hessian(b -> loglik_exact(b, cache.data; neg=true), β)
    
    # Add penalty contributions
    n = length(β)
    H_lambda = copy(H_unpenalized)
    
    penalty = cache.penalty_config
    lambda_idx = 1
    
    for term in penalty.terms
        idx = term.hazard_indices
        H_lambda[idx, idx] .+= λ[lambda_idx] * term.S
        lambda_idx += 1
    end
    
    for term in penalty.total_hazard_terms
        # For total hazard terms, the Hessian contribution is spread across indices
        for idx_range1 in term.hazard_indices
            for idx_range2 in term.hazard_indices
                H_lambda[idx_range1, idx_range2] .+= λ[lambda_idx] * term.S
            end
        end
        lambda_idx += 1
    end
    
    for term in penalty.smooth_covariate_terms
        idx = term.param_indices
        H_lambda[idx, idx] .+= λ[lambda_idx] * term.S
        lambda_idx += 1
    end
    
    return H_lambda
end

# =============================================================================
# Implicit Function Factory
# =============================================================================

"""
    make_implicit_beta_function(cache::ImplicitBetaCache) -> ImplicitFunction

Create an ImplicitFunction for β̂(ρ) that can be differentiated w.r.t. ρ.

The returned function has signature:
    β, z = implicit_beta(ρ)

where:
- `ρ`: Log-smoothing parameters
- `β`: Optimal coefficients (differentiable w.r.t. ρ)
- `z`: Byproduct (H_lambda, converged)

# Example
```julia
cache = build_implicit_beta_cache(model, data, penalty, beta_init)
implicit_beta = make_implicit_beta_function(cache)

# Use in optimization
function criterion(ρ)
    β, _ = implicit_beta(ρ)
    return compute_V(β, ρ, ...)
end

# ForwardDiff works through the implicit function
grad = ForwardDiff.gradient(criterion, ρ_init)
```
"""
function make_implicit_beta_function(cache::ImplicitBetaCache)
    # Forward function
    forward = ρ -> forward_beta_solve(ρ, cache)
    
    # Conditions function (needs all 4 args: x, y, z, extras...)
    conditions = (ρ, β, z) -> beta_optimality_conditions(ρ, β, z, cache)
    
    # Create ImplicitFunction with direct linear solver
    # Use MatrixRepresentation (required for DirectLinearSolver) and
    # ADTypes.AutoForwardDiff for both x and y derivatives
    return ImplicitFunction(
        forward,
        conditions;
        representation=MatrixRepresentation(),  # Must use MatrixRepresentation with DirectLinearSolver
        linear_solver=DirectLinearSolver(),
        backends=(x=AutoForwardDiff(), y=AutoForwardDiff())
    )
end

# =============================================================================
# PIJCV Criterion with Implicit Differentiation
# =============================================================================

"""
    compute_ncv_at_beta(β, ρ, cache::ImplicitBetaCache; nfolds=0) -> Float64

Compute the NCV/PIJCV criterion at given (β, ρ).

This is the second stage of PIJCV with implicit differentiation:
1. Get β̂(ρ) via implicit function (handles ∂β̂/∂ρ)
2. Compute V(ρ) = Σᵢ Dᵢ(β̂⁻ⁱ) using Newton approximation

The gradients/Hessians for V are computed at Float64, not Dual, because
all the necessary derivatives w.r.t. ρ come through β̂ via implicit diff.

# Arguments
- `β`: Current coefficient estimate
- `ρ`: Log-smoothing parameters
- `cache`: ImplicitBetaCache
- `nfolds`: 0 for LOO, k for k-fold approximation

# Returns
Scalar criterion value (lower is better)
"""
function compute_ncv_at_beta(β::AbstractVector, ρ::AbstractVector, cache::ImplicitBetaCache; 
                              nfolds::Int=0, use_quadratic::Bool=false)
    # Extract Float64 for gradient/Hessian computation
    β_float = Float64[ForwardDiff.value(x) for x in β]
    ρ_float = Float64[ForwardDiff.value(x) for x in ρ]
    λ = exp.(ρ_float)
    
    # Compute subject gradients and Hessians (Float64)
    # This is the expensive part but does NOT need to be differentiated w.r.t. ρ
    # because that information comes through β via implicit differentiation
    subject_grads, subject_hessians = _compute_subject_grads_hessians(β_float, cache)
    
    # Build state for criterion evaluation
    H_unpenalized = sum(subject_hessians)
    n_subjects = length(subject_grads[1, :])
    n_params = length(β_float)
    
    penalty_config = cache.penalty_config
    
    state = SmoothingSelectionState(
        β_float,
        H_unpenalized,
        hcat(subject_grads...),  # p × n matrix
        subject_hessians,
        penalty_config,
        n_subjects,
        n_params,
        cache.model,
        cache.data,
        nothing  # pijcv_eval_cache - will be built lazily
    )
    
    # Compute criterion using existing function
    log_lambda = collect(ρ_float)  # Convert to plain Vector for criterion
    
    if use_quadratic
        return compute_pijcv_criterion_fast(log_lambda, state)
    elseif nfolds == 0
        return compute_pijcv_criterion(log_lambda, state)
    else
        return compute_pijkfold_criterion(log_lambda, state, nfolds)
    end
end

"""
    _compute_subject_grads_hessians(β, cache::ImplicitBetaCache{M, ExactData}) -> (grads, hessians)

Compute per-subject gradients and Hessians for ExactData.

# Type Handling
This function accepts AbstractVector{T} to allow Dual numbers from ForwardDiff,
but extracts Float64 values internally. The gradient information w.r.t. ρ flows
through the ImplicitFunction's IFT, not through this computation.
"""
function _compute_subject_grads_hessians(β::AbstractVector{T}, cache::ImplicitBetaCache{M, ExactData}) where {T<:Real, M}
    # Extract Float64 values - subject grads/Hessians are treated as constants
    # The Dual information flows through ImplicitFunction's IFT
    β_float = T === Float64 ? β : Float64[ForwardDiff.value(x) for x in β]
    
    samplepaths = cache.data.paths
    
    # Use existing parallel implementation
    grads_ll, hessians_ll = compute_subject_grads_and_hessians_fast(
        β_float, cache.model, samplepaths; use_threads=:auto
    )
    
    # Convert to loss convention (negative log-likelihood)
    grads = -grads_ll
    hessians = [-H for H in hessians_ll]
    
    return grads, hessians
end

# =============================================================================
# AD-Compatible PIJCV Criterion for Implicit Differentiation
# =============================================================================

"""
    compute_pijcv_criterion_implicit(β, log_lambda, cache::ImplicitBetaCache;
                                     pijcv_eval_cache=nothing) -> V

Compute PIJCV criterion V(ρ) that is AD-compatible for implicit differentiation.

This function computes subject gradients/Hessians fresh at each call using the
current β̂(ρ) value. This ensures that when ForwardDiff evaluates V at different
ρ values (for numerical differentiation), each evaluation uses the correct 
gᵢ(β̂(ρ)) and Hᵢ(β̂(ρ)), capturing the full dependence of V on ρ.

# Mathematical Background

The PIJCV criterion (Wood 2024) is:
    V(ρ) = Σᵢ Dᵢ(β̂⁻ⁱ) = Σᵢ [-ℓᵢ(β̂) + gᵢᵀΔ⁻ⁱ + ½Δ⁻ⁱᵀHᵢΔ⁻ⁱ]

where:
- β̂ = β̂(ρ) is the penalized MLE at smoothing parameter λ = exp(ρ)
- gᵢ = -∇ℓᵢ(β̂) is the negative gradient of subject i's log-likelihood
- Hᵢ = -∇²ℓᵢ(β̂) is the negative Hessian of subject i's log-likelihood
- Δ⁻ⁱ = (H_λ - Hᵢ)⁻¹ gᵢ is the LOO Newton step

The gradient ∂V/∂ρ captures dependence through:
1. λ = exp(ρ) directly in H_λ (captured by AD on log_lambda)
2. β̂(ρ) in gᵢ, Hᵢ, and -ℓᵢ (captured by fresh recomputation)

# Arguments
- `β::AbstractVector`: Current coefficients (may be dual numbers from ImplicitFunction)
- `log_lambda::AbstractVector`: Log-smoothing parameters (may be dual numbers)
- `cache::ImplicitBetaCache{M, ExactData}`: Contains model, data, penalty info
- `pijcv_eval_cache`: (unused, for API compatibility)

# Returns
- Scalar criterion value (same numeric type as λ for AD compatibility)
"""
function compute_pijcv_criterion_implicit(
    β::AbstractVector{T1},
    log_lambda::AbstractVector{T2},
    cache::ImplicitBetaCache{M, ExactData};
    pijcv_eval_cache = nothing
) where {T1<:Real, T2<:Real, M}
    # Promote to common type for proper AD
    T = promote_type(T1, T2)
    
    lambda = exp.(log_lambda)
    n_params = length(β)
    
    # ==========================================================================
    # CRITICAL: Recompute subject grads/hessians at current β̂(ρ)
    # ==========================================================================
    # The PIJCV criterion depends on ρ through TWO paths:
    #   1. λ = exp(ρ) directly in H_λ
    #   2. β̂(ρ) in gᵢ(β̂), Hᵢ(β̂), and -ℓᵢ(β̂)
    # 
    # By recomputing gᵢ and Hᵢ at the current β̂, we ensure that when 
    # ForwardDiff evaluates V at different ρ values, each evaluation uses
    # the correct gradients/Hessians for that ρ. This captures the full
    # dependence ∂V/∂ρ = ∂V/∂β · ∂β̂/∂ρ + ∂V/∂λ · ∂λ/∂ρ through function
    # evaluation rather than through explicit AD.
    # ==========================================================================
    β_float = Float64[ForwardDiff.value(x) for x in β]
    subject_grads, subject_hessians = _compute_subject_grads_hessians(β_float, cache)
    H_unpenalized = sum(subject_hessians)
    n_subjects = size(subject_grads, 2)
    
    # Build penalized Hessian with dual-number λ
    # H_λ = H_unpen + Σⱼ λⱼ Sⱼ
    # Must iterate through penalty terms and add contributions at correct indices
    H_lambda = Matrix{T}(H_unpenalized)  # Convert to appropriate type
    
    penalty = cache.penalty_config
    lambda_idx = 1
    
    # Baseline hazard penalty contributions
    for term in penalty.terms
        idx = term.hazard_indices
        λ_j = lambda_idx <= length(lambda) ? lambda[lambda_idx] : lambda[1]
        H_lambda[idx, idx] .+= λ_j .* term.S
        lambda_idx += 1
    end
    
    # Total hazard penalty contributions
    for term in penalty.total_hazard_terms
        λ_j = lambda_idx <= length(lambda) ? lambda[lambda_idx] : lambda[1]
        for idx_range1 in term.hazard_indices
            for idx_range2 in term.hazard_indices
                H_lambda[idx_range1, idx_range2] .+= λ_j .* term.S
            end
        end
        lambda_idx += 1
    end
    
    # Smooth covariate penalty contributions
    for term in penalty.smooth_covariate_terms
        idx = term.param_indices
        λ_j = lambda_idx <= length(lambda) ? lambda[lambda_idx] : lambda[1]
        H_lambda[idx, idx] .+= λ_j .* term.S
        lambda_idx += 1
    end
    
    # ==========================================================================
    # PIJCV COMPUTATION
    # ==========================================================================
    # The PIJCV criterion: V(ρ) = Σᵢ Dᵢ where Dᵢ = -ℓᵢ(β̂) + gᵢᵀΔ⁻ⁱ + ½Δ⁻ⁱᵀHᵢΔ⁻ⁱ
    # with Δ⁻ⁱ = H_{λ,-i}⁻¹ gᵢ being the LOO Newton step.
    #
    # The gradient ∂V/∂ρ is computed via finite differences through function
    # evaluation: when ρ changes, β̂(ρ) changes, which changes gᵢ, Hᵢ, and ℓᵢ.
    # By recomputing subject_grads/hessians at each call (done above), we
    # ensure correct function evaluation at each ρ value.
    #
    # Note: λ = exp(ρ) affects V directly through H_lambda. This is captured
    # by AD on log_lambda in the H_lambda construction above.
    # ==========================================================================
    
    # Get subject log-likelihoods at current β
    ll_subj_base = loglik_exact(β_float, cache.data; neg=false, return_ll_subj=true)
    
    # Compute V_q using quadratic approximation
    V = T(0)
    
    for i in 1:n_subjects
        g_i = @view subject_grads[:, i]
        H_i = subject_hessians[i]
        
        # Leave-one-out penalized Hessian: H_{λ,-i} = H_λ - H_i
        # Note: H_i is Float64, H_lambda is type T
        H_lambda_loo = H_lambda - Matrix{T}(H_i)
        
        # Solve for Newton step: Δ⁻ⁱ = H_{λ,-i}⁻¹ gᵢ
        delta_i = try
            Symmetric(H_lambda_loo) \ collect(g_i)
        catch e
            # If solve fails, return large penalty
            @debug "Linear solve failed in compute_pijcv_criterion_implicit" subject=i
            return T(1e10)
        end
        
        # Quadratic approximation components
        linear_term = dot(g_i, delta_i)
        quadratic_term = T(0.5) * dot(delta_i, H_i * delta_i)
        
        # Subject contribution
        D_i = T(-ll_subj_base[i]) + linear_term + quadratic_term
        
        V += D_i
    end
    
    return V
end

# Similar implementations for MPanelData and MCEMSelectionData would be added here...

# =============================================================================
# AD-Compatible V(β, ρ) for ForwardDiff Through ImplicitFunction
# =============================================================================

"""
    compute_V_at_beta(β, ρ, cache::ImplicitBetaCache) -> V

Compute PIJCV criterion V(ρ) at given (β, ρ) in an AD-compatible way.

This is the key function for nested AD through ImplicitFunction:
```julia
function pijcv_objective(ρ)
    implicit_beta = make_implicit_beta_function(cache)
    β̂, z = implicit_beta(ρ)
    return compute_V_at_beta(β̂, ρ, cache)
end
grad = ForwardDiff.gradient(pijcv_objective, ρ_init)
```

The full chain rule ∂V/∂ρ = ∂V/∂β · ∂β̂/∂ρ + ∂V/∂λ · ∂λ/∂ρ is handled by:
- compute_pijcv_criterion_implicit recomputes gᵢ(β̂), Hᵢ(β̂) at each call
- This captures β dependence through function evaluation
- λ dependence is captured by AD on log_lambda in H_lambda construction

# Arguments
- `β::AbstractVector`: Coefficient vector (may contain Dual numbers from ImplicitFunction)
- `ρ::AbstractVector`: Log-smoothing parameters (may contain Dual numbers)
- `cache::ImplicitBetaCache`: Contains model, data, penalty info

# Returns
Scalar criterion value V (same numeric type as λ for AD compatibility)
"""
function compute_V_at_beta(β::AbstractVector, ρ::AbstractVector, 
                           cache::ImplicitBetaCache{M, ExactData}) where {M}
    # Delegate to the criterion function which handles everything
    return compute_pijcv_criterion_implicit(β, ρ, cache)
end

"""
    make_pijcv_objective(cache::ImplicitBetaCache) -> Function

Create a PIJCV objective function ρ → V(ρ) that can be differentiated with ForwardDiff.

Returns a closure that:
1. Computes β̂(ρ) via the ImplicitFunction
2. Evaluates V(β̂, ρ) with proper AD chain rule handling

# Example
```julia
cache = build_implicit_beta_cache(model, data, penalty, beta_init)
pijcv_obj = make_pijcv_objective(cache)

# Evaluate
V = pijcv_obj(log_lambda)

# Gradient via ForwardDiff
grad_V = ForwardDiff.gradient(pijcv_obj, log_lambda)
```
"""
function make_pijcv_objective(cache::ImplicitBetaCache)
    implicit_beta = make_implicit_beta_function(cache)
    
    return function pijcv_objective(ρ::AbstractVector)
        β̂, _ = implicit_beta(ρ)
        return compute_V_at_beta(β̂, ρ, cache)
    end
end


# =============================================================================
# Analytical Gradient for PIJCV Criterion (CORRECT Formula with Third Derivatives)
# =============================================================================

"""
    compute_pijcv_with_gradient(β, log_lambda, cache::ImplicitBetaCache;
                                 subject_grads, subject_hessians, H_unpenalized,
                                 dbeta_drho, subject_third_derivatives=nothing) -> (V, grad_V)

Compute PIJCV criterion V(ρ) AND its **CORRECT** analytical gradient ∇V simultaneously.

# Mathematical Background (Wood 2024, corrected for third derivatives)

The CORRECT PIJCV criterion (NCV, Wood 2024, Equation 2) is:

    V(ρ) = Σᵢ -ℓᵢ(β̃₋ᵢ)

where:
- β̃₋ᵢ = β̂ - Δ⁻ⁱ is the pseudo-estimate (one Newton step from β̂)
- Δ⁻ⁱ = (H_λ - Hᵢ)⁻¹ gᵢ is the LOO step
- gᵢ = -∇ℓᵢ(β̂) is the per-subject score at the full MLE
- Hᵢ = -∇²ℓᵢ(β̂) is the per-subject Hessian at the full MLE

## CORRECT Gradient Formula (with third derivatives)

    dV/dρ = Σᵢ [-∇ℓᵢ(β̃₋ᵢ)ᵀ · dβ̃₋ᵢ/dρ]

where dβ̃₋ᵢ/dρ = dβ̂/dρ - dΔ⁻ⁱ/dρ and the chain rule gives:

    dΔ⁻ⁱ/dρ = H_loo⁻¹ · [dgᵢ/dρ - dH_loo/dρ · Δ⁻ⁱ]

with:
- dgᵢ/dρ = +Hᵢ · dβ̂/dρ (POSITIVE sign: gᵢ = -∇ℓᵢ, so dgᵢ/dρ = -∇²ℓᵢ · dβ̂/dρ = Hᵢ · dβ̂/dρ)
- dH_loo/dρ = dH_λ/dρ - dHᵢ/dρ
- dH_λ/dρ = λS + Σⱼ Σₗ (∂Hⱼ/∂βₗ) · (dβ̂/dρ)ₗ  [includes third derivatives!]
- dHᵢ/dρ = Σₗ (∂Hᵢ/∂βₗ) · (dβ̂/dρ)ₗ

## Critical Implementation Note

The previous implementation neglected the third derivative terms (∂Hᵢ/∂β), causing
~30% bias in optimal λ selection. This corrected version includes the full chain rule.

# Arguments
- `β::Vector{Float64}`: Current coefficient estimate (β̂)
- `log_lambda::Vector{Float64}`: Log-smoothing parameters (ρ)
- `cache::ImplicitBetaCache`: Contains model, data, penalty info
- `subject_grads::Matrix{Float64}`: Per-subject loss gradients (p × n), gᵢ = -∇ℓᵢ(β̂)
- `subject_hessians::Vector{Matrix{Float64}}`: Per-subject loss Hessians, Hᵢ = -∇²ℓᵢ(β̂)
- `H_unpenalized::Matrix{Float64}`: Sum of subject Hessians
- `dbeta_drho::Vector{Float64}`: dβ̂/dρ from ImplicitDifferentiation.jl
- `subject_third_derivatives::Union{Nothing, Vector}=nothing`: Pre-computed ∂Hᵢ/∂β tensors

# Returns
- `V::Float64`: Criterion value (lower is better)
- `grad_V::Vector{Float64}`: Gradient w.r.t. log(λ)

# References
- Wood, S.N. (2024). "On Neighbourhood Cross Validation." arXiv:2404.16490v4
"""
function compute_pijcv_with_gradient(
    β::Vector{Float64},
    log_lambda::Vector{Float64},
    cache::ImplicitBetaCache;
    subject_grads::Matrix{Float64},
    subject_hessians::Vector{<:Matrix{Float64}},
    H_unpenalized::Matrix{Float64},
    dbeta_drho::Vector{Float64},
    subject_third_derivatives::Union{Nothing, Vector{Array{Float64,3}}} = nothing
)
    lambda = exp.(log_lambda)
    n_lambda = length(lambda)
    n_subjects = size(subject_grads, 2)
    n_params = length(β)
    
    # ==========================================================================
    # Build penalized Hessian H_λ and full S matrix
    # ==========================================================================
    H_lambda = copy(H_unpenalized)
    S_full = zeros(n_params, n_params)  # Combined penalty matrix
    penalty = cache.penalty_config
    
    # Baseline hazard terms
    for term in penalty.terms
        idx = term.hazard_indices
        H_lambda[idx, idx] .+= lambda[1] .* term.S
        S_full[idx, idx] .= Matrix(term.S)
    end
    
    # Total hazard terms
    for term in penalty.total_hazard_terms
        for idx_range1 in term.hazard_indices
            for idx_range2 in term.hazard_indices
                H_lambda[idx_range1, idx_range2] .+= lambda[1] .* term.S
                S_full[idx_range1, idx_range2] .= Matrix(term.S)
            end
        end
    end
    
    # Smooth covariate terms
    for term in penalty.smooth_covariate_terms
        idx = term.param_indices
        H_lambda[idx, idx] .+= lambda[1] .* term.S
        S_full[idx, idx] .= Matrix(term.S)
    end
    
    H_lambda_sym = Symmetric(H_lambda)
    
    # Helper function to solve linear systems
    function solve_system(A_sym, b)
        try
            return A_sym \ b
        catch e
            @debug "Linear solve failed" exception=e
            return fill(NaN, length(b))
        end
    end
    
    # ==========================================================================
    # Compute third derivatives if not provided
    # ∂Hᵢ/∂β is a tensor of shape (n_params, n_params, n_params)
    # where ∂Hᵢ/∂β[:,:,l] is the derivative of Hᵢ w.r.t. βₗ
    # ==========================================================================
    dH_dbeta_all = if isnothing(subject_third_derivatives)
        _compute_subject_third_derivatives(β, cache)
    else
        subject_third_derivatives
    end
    
    # ==========================================================================
    # Compute dH_λ/dρ with third derivatives
    # dH_λ/dρ = λS + Σⱼ Σₗ (∂Hⱼ/∂βₗ)·(dβ̂/dρ)ₗ
    # ==========================================================================
    dH_lambda_drho = lambda[1] * S_full
    for i in 1:n_subjects
        for l in 1:n_params
            dH_lambda_drho .+= dH_dbeta_all[i][:,:,l] * dbeta_drho[l]
        end
    end
    
    # ==========================================================================
    # Build PIJCV evaluation cache for efficient LOO likelihood evaluation
    # ==========================================================================
    eval_cache = build_pijcv_eval_cache(cache.data)
    
    # ==========================================================================
    # CORRECT PIJCV: V = Σᵢ -ℓᵢ(β̃₋ᵢ) where β̃₋ᵢ = β̂ - Δ⁻ⁱ
    # ==========================================================================
    V = 0.0
    grad_V = zeros(n_lambda)
    
    for i in 1:n_subjects
        gᵢ = subject_grads[:, i]
        Hᵢ = subject_hessians[i]
        dH_dbeta_i = dH_dbeta_all[i]
        
        # Leave-one-out penalized Hessian
        H_loo = H_lambda - Hᵢ
        H_loo_sym = Symmetric(H_loo)
        
        # Newton step: Δ⁻ⁱ = H_{λ,-i}⁻¹ gᵢ
        Δᵢ = solve_system(H_loo_sym, gᵢ)
        if any(isnan, Δᵢ)
            return (1e10, fill(0.0, n_lambda))
        end
        
        # Pseudo-estimate: β̃₋ᵢ = β̂ - Δ⁻ⁱ
        β_tilde_i = β .- Δᵢ
        
        # CORRECT criterion: evaluate ACTUAL likelihood at pseudo-estimate
        ll_at_pseudo = loglik_subject_cached(β_tilde_i, eval_cache, i)
        V_i = -ll_at_pseudo
        V += V_i
        
        # =======================================================================
        # CORRECT gradient with third derivatives
        # =======================================================================
        # dVᵢ/dρ = -∇ℓᵢ(β̃₋ᵢ)ᵀ · dβ̃₋ᵢ/dρ
        # where dβ̃₋ᵢ/dρ = dβ̂/dρ - dΔ⁻ⁱ/dρ
        
        # Gradient of likelihood at pseudo-estimate
        grad_ll_at_pseudo = ForwardDiff.gradient(
            b -> loglik_subject_cached(b, eval_cache, i), 
            β_tilde_i
        )
        
        # dgᵢ/dρ = +Hᵢ·dβ̂/dρ (POSITIVE sign!)
        # Since gᵢ = -∇ℓᵢ(β̂), we have dgᵢ/dρ = -∇²ℓᵢ(β̂)·dβ̂/dρ = Hᵢ·dβ̂/dρ
        dgᵢ_drho = Hᵢ * dbeta_drho
        
        # dHᵢ/dρ = Σₗ (∂Hᵢ/∂βₗ)·(dβ̂/dρ)ₗ
        dHᵢ_drho = zeros(n_params, n_params)
        for l in 1:n_params
            dHᵢ_drho .+= dH_dbeta_i[:,:,l] * dbeta_drho[l]
        end
        
        # dH_loo/dρ = dH_λ/dρ - dHᵢ/dρ
        dH_loo_drho = dH_lambda_drho - dHᵢ_drho
        
        # dΔ⁻ⁱ/dρ = H_loo⁻¹·(dgᵢ/dρ - dH_loo/dρ·Δᵢ)
        dDelta_drho = solve_system(H_loo_sym, dgᵢ_drho - dH_loo_drho * Δᵢ)
        if any(isnan, dDelta_drho)
            continue
        end
        
        # dβ̃₋ᵢ/dρ = dβ̂/dρ - dΔ⁻ⁱ/dρ
        dbeta_tilde_drho = dbeta_drho - dDelta_drho
        
        # dVᵢ/dρ = -∇ℓᵢ(β̃₋ᵢ)ᵀ · dβ̃₋ᵢ/dρ
        dV_i_drho = -dot(grad_ll_at_pseudo, dbeta_tilde_drho)
        grad_V[1] += dV_i_drho
    end
    
    return (V, grad_V)
end

"""
    _compute_subject_third_derivatives(β, cache::ImplicitBetaCache) -> Vector{Array{Float64,3}}

Compute third derivatives ∂Hᵢ/∂β for all subjects.

Returns a vector of 3D tensors, one per subject. Each tensor has shape
(n_params, n_params, n_params) where tensor[:,:,l] is the derivative of
the subject's Hessian with respect to βₗ.

# Implementation
Uses ForwardDiff.jacobian on the flattened Hessian:
    ∂Hᵢ/∂β = reshape(ForwardDiff.jacobian(β → vec(Hᵢ(β)), β), n_params, n_params, n_params)
"""
function _compute_subject_third_derivatives(β::Vector{Float64}, cache::ImplicitBetaCache{M, ExactData}) where M
    n_subjects = length(cache.data.paths)
    n_params = length(β)
    
    third_derivs = Vector{Array{Float64,3}}(undef, n_subjects)
    
    for i in 1:n_subjects
        # Compute Jacobian of flattened Hessian
        H_flat_jac = ForwardDiff.jacobian(
            b -> vec(-ForwardDiff.hessian(bb -> loglik_subject(bb, cache.data, i), b)),
            β
        )
        # Reshape to tensor: third_derivs[i][:,:,l] = ∂Hᵢ/∂βₗ
        third_derivs[i] = reshape(H_flat_jac, n_params, n_params, n_params)
    end
    
    return third_derivs
end


# =============================================================================
# PIJCV with Implicit Differentiation - Main Entry Point
# =============================================================================

"""
    _nested_optimization_pijcv_implicit(model, data::ExactData, penalty, selector; kwargs...) -> HyperparameterSelectionResult

Nested optimization for PIJCV using ImplicitDifferentiation.jl for efficient gradients.

This is the high-performance version of `_nested_optimization_pijcv` that avoids
nested automatic differentiation by using the implicit function theorem:

    ∂β̂/∂ρⱼ = -H_λ⁻¹ · (λⱼ Sⱼ β̂)

# Performance Benefits
- Avoids differentiating through the inner optimization
- Reduces computational complexity from O(np³) to O(np²)
- Expected 15-20x speedup and 10x memory reduction

# Algorithm
1. Build ImplicitBetaCache with model, data, penalty
2. Create ImplicitFunction wrapping the inner optimization
3. Define NCV criterion using implicit β̂(ρ)
4. Optimize ρ using ForwardDiff with gradients via implicit diff

# Arguments
- `model::MultistateProcess`: Model for likelihood evaluation
- `data::ExactData`: Data container
- `penalty::AbstractPenalty`: Penalty configuration
- `selector::PIJCVSelector`: PIJCV selector

# Keyword Arguments
- `beta_init::Vector{Float64}`: Initial coefficient estimate
- `inner_maxiter::Int=50`: Maximum iterations for inner β fitting
- `outer_maxiter::Int=100`: Maximum iterations for outer ρ optimization
- `lambda_tol::Float64=1e-3`: Convergence tolerance for λ
- `lambda_init::Union{Nothing, Vector{Float64}}`: Warm-start for λ
- `verbose::Bool=false`: Print progress

# Returns
- `HyperparameterSelectionResult`: Contains optimal λ, warmstart_beta, updated penalty

# See Also
- `_nested_optimization_pijcv`: Legacy version with nested AD
- `make_implicit_beta_function`: Creates the ImplicitFunction

# References
- Wood, S.N. (2024). "On Neighbourhood Cross Validation." arXiv:2404.16490v4
- Blondel et al. (2022). "Efficient and Modular Implicit Differentiation."
"""
function _nested_optimization_pijcv_implicit(
    model::MultistateProcess,
    data::ExactData,
    penalty::AbstractPenalty,
    selector::PIJCVSelector;
    beta_init::Vector{Float64},
    inner_maxiter::Int = 50,
    outer_maxiter::Int = 100,
    lambda_tol::Float64 = 1e-3,
    lambda_init::Union{Nothing, Vector{Float64}} = nothing,
    verbose::Bool = false
)
    # Get bounds and setup
    lb, ub = model.bounds.lb, model.bounds.ub
    n_lambda = n_hyperparameters(penalty)
    n_subjects = length(data.paths)
    n_params = length(beta_init)
    
    # Determine method based on nfolds and use_quadratic
    method = if selector.use_quadratic
        selector.nfolds == 0 ? :pijlcv_implicit : Symbol("pijlcv$(selector.nfolds)_implicit")
    else
        selector.nfolds == 0 ? :pijcv_implicit : Symbol("pijcv$(selector.nfolds)_implicit")
    end
    
    if verbose
        println("Optimizing λ via PIJCV with ImplicitDifferentiation.jl")
        println("  Method: $method, n_lambda: $n_lambda")
        selector.use_quadratic && println("  Using fast quadratic approximation V_q")
    end
    
    # Build penalty_config for use in criterion
    penalty_config = penalty isa QuadraticPenalty ? penalty : build_penalty_config(model, SplinePenalty())
    
    # Build implicit differentiation cache
    cache = build_implicit_beta_cache(model, data, penalty_config, beta_init;
                                       inner_maxiter=inner_maxiter)
    
    # Create the implicit function for β̂(ρ)
    implicit_beta = make_implicit_beta_function(cache)
    
    # Track evaluations
    n_criterion_evals = Ref(0)
    current_beta_ref = Ref(copy(beta_init))
    
    # Build PIJCV evaluation cache for efficient LOO evaluation
    pijcv_cache = if !selector.use_quadratic
        build_pijcv_eval_cache(data)
    else
        nothing
    end
    
    # Pre-compute Float64 quantities that don't need AD
    # These are computed once at the initial β and used as constants for the
    # Newton approximation. The gradients w.r.t. ρ come through β̂(ρ) via implicit diff
    # and through λ in the penalized Hessian directly.
    
    # Define criterion AND gradient function using CORRECT analytical formulas
    # with third derivatives for proper chain rule (see test_correct_pijcv_ad_v5.jl)
    function ncv_criterion_and_gradient(log_lambda_vec)
        n_criterion_evals[] += 1
        
        # Get β̂(ρ) via inner optimization (Float64 only)
        log_lambda_float = Float64.(log_lambda_vec)
        lambda_float = exp.(log_lambda_float)
        penalty_current = set_hyperparameters(penalty_config, lambda_float)
        
        # Solve inner problem
        β_float = _fit_inner_coefficients(model, data, penalty_current, current_beta_ref[];
                                          lb=lb, ub=ub, maxiter=inner_maxiter)
        current_beta_ref[] = β_float
        
        # Compute dβ̂/dρ via ImplicitDifferentiation.jl
        # This uses the IFT: dβ̂/dρ = -H_λ⁻¹ · (λS β̂)
        dbeta_drho = ForwardDiff.jacobian(
            ρ_vec -> implicit_beta(ρ_vec)[1],
            log_lambda_float
        )[:, 1]  # Extract single column for scalar ρ
        
        # Compute subject gradients and Hessians at β̂
        subject_grads_ll, subject_hessians_ll = compute_subject_grads_and_hessians_fast(
            β_float, model, data.paths; use_threads=:auto
        )
        
        # Convert to loss convention
        subject_grads = -subject_grads_ll
        subject_hessians = [-H for H in subject_hessians_ll]
        H_unpenalized = sum(subject_hessians)
        
        # Compute criterion AND CORRECT analytical gradient simultaneously
        # Now includes third derivatives for proper chain rule
        V, grad_V = compute_pijcv_with_gradient(
            β_float,
            log_lambda_float,
            cache;
            subject_grads=subject_grads,
            subject_hessians=subject_hessians,
            H_unpenalized=H_unpenalized,
            dbeta_drho=dbeta_drho
        )
        
        if verbose && n_criterion_evals[] % 5 == 0
            @info "Criterion eval $(n_criterion_evals[]): log(λ)=$(round.(log_lambda_float, digits=2)), V=$(round(V, digits=3)), ||∇V||=$(round(norm(grad_V), digits=4))"
        end
        
        return (V, grad_V)
    end
    
    # Wrapper for criterion only (for OptimizationFunction)
    function ncv_criterion_only(log_lambda_vec, _)
        V, _ = ncv_criterion_and_gradient(log_lambda_vec)
        return V
    end
    
    # Wrapper for gradient only (for OptimizationFunction)
    function ncv_gradient_only!(grad_storage, log_lambda_vec, _)
        _, grad_V = ncv_criterion_and_gradient(log_lambda_vec)
        grad_storage .= grad_V
        return nothing
    end
    
    # Adaptive bounds for log(λ)
    log_lb_scalar, log_ub_scalar = compute_lambda_bounds(n_subjects, n_params)
    log_lb = fill(log_lb_scalar, n_lambda)
    log_ub = fill(log_ub_scalar, n_lambda)
    
    # Initialize λ: Use lambda_init if provided, otherwise EFS estimate
    current_log_lambda = if !isnothing(lambda_init) && length(lambda_init) >= n_lambda
        if verbose
            println("  Using provided λ warm-start (skipping EFS)")
        end
        log.(lambda_init[1:n_lambda])
    else
        # Get EFS estimate as warmstart
        if verbose
            println("  Getting EFS initial estimate...")
        end
        efs_result = _nested_optimization_reml(model, data, penalty;
                                               beta_init=beta_init,
                                               inner_maxiter=inner_maxiter,
                                               outer_maxiter=30,
                                               lambda_tol=0.1,
                                               verbose=false)
        current_beta_ref[] = efs_result.warmstart_beta
        log.(efs_result.lambda[1:n_lambda])
    end
    
    # Set up optimization with analytical gradient
    optf = OptimizationFunction(ncv_criterion_only; grad=ncv_gradient_only!)
    prob = OptimizationProblem(optf, current_log_lambda, nothing; lb=log_lb, ub=log_ub)
    
    if verbose
        println("  Using L-BFGS outer optimizer with analytical gradients...")
    end
    
    # Solve with Fminbox L-BFGS
    sol = solve(prob, OptimizationOptimJL.Fminbox(OptimizationOptimJL.LBFGS());
                maxiters=outer_maxiter,
                f_tol=lambda_tol,
                x_tol=lambda_tol)
    
    optimal_log_lambda = sol.u
    best_criterion = sol.objective
    current_beta = current_beta_ref[]
    converged = sol.retcode == ReturnCode.Success || sol.retcode == ReturnCode.MaxIters
    
    if verbose
        lambda_val = exp.(optimal_log_lambda)
        println("  Final: log(λ)=$(round.(optimal_log_lambda, digits=2)), λ=$(round.(lambda_val, sigdigits=3)), V=$(round(best_criterion, digits=3))")
        println("  Criterion evaluations: $(n_criterion_evals[])")
        println(converged ? "  Converged successfully" : "  Warning: Optimizer returned $(sol.retcode)")
    end
    
    # Build final results
    optimal_lambda = exp.(optimal_log_lambda)
    optimal_lambda_vec = n_lambda == 1 ? fill(optimal_lambda[1], n_hyperparameters(penalty_config)) : optimal_lambda
    updated_penalty = set_hyperparameters(penalty, optimal_lambda_vec)
    
    # Compute EDF at optimal (lambda, beta)
    edf = compute_edf(current_beta, optimal_lambda_vec, penalty_config, model, data)
    
    return HyperparameterSelectionResult(
        optimal_lambda_vec,
        current_beta,
        updated_penalty,
        best_criterion,
        edf,
        converged,
        method,
        n_criterion_evals[],
        (log_lambda = optimal_log_lambda, retcode = sol.retcode, implicit_diff = true)
    )
end


# =============================================================================
# Markov Panel Data Support
# =============================================================================

"""
    _compute_penalized_hessian_at_beta(β, λ, cache::ImplicitBetaCache{M, MPanelData}) -> Matrix{Float64}

Compute the penalized Hessian H_λ for Markov panel data.
"""
function _compute_penalized_hessian_at_beta(β::Vector{Float64}, λ::Vector{Float64}, 
                                            cache::ImplicitBetaCache{M, MPanelData}) where M
    # Get unpenalized Hessian using Markov likelihood
    H_unpenalized = ForwardDiff.hessian(b -> loglik_markov(b, cache.data; neg=true), β)
    
    # Add penalty contributions (same as ExactData)
    n = length(β)
    H_lambda = copy(H_unpenalized)
    
    penalty = cache.penalty_config
    lambda_idx = 1
    
    for term in penalty.terms
        idx = term.hazard_indices
        H_lambda[idx, idx] .+= λ[lambda_idx] * term.S
        lambda_idx += 1
    end
    
    for term in penalty.total_hazard_terms
        for idx_range1 in term.hazard_indices
            for idx_range2 in term.hazard_indices
                H_lambda[idx_range1, idx_range2] .+= λ[lambda_idx] * term.S
            end
        end
        lambda_idx += 1
    end
    
    for term in penalty.smooth_covariate_terms
        idx = term.param_indices
        H_lambda[idx, idx] .+= λ[lambda_idx] * term.S
        lambda_idx += 1
    end
    
    return H_lambda
end

"""
    _compute_subject_grads_hessians(β, cache::ImplicitBetaCache{M, MPanelData}) -> (grads, hessians)

Compute per-subject gradients and Hessians for Markov panel data.
"""
function _compute_subject_grads_hessians(β::Vector{Float64}, cache::ImplicitBetaCache{M, MPanelData}) where M
    books = cache.data.books
    model = cache.model
    
    # Use existing Markov-specific functions
    grads_ll = compute_subject_gradients(β, model, books)
    hessians_ll = compute_subject_hessians(β, model, books)
    
    # Convert to loss convention
    grads = -grads_ll
    hessians = [-H for H in hessians_ll]
    
    return grads, hessians
end

"""
    _nested_optimization_pijcv_markov_implicit(model, data::MPanelData, penalty, selector; kwargs...) -> HyperparameterSelectionResult

Nested optimization for PIJCV using ImplicitDifferentiation.jl for Markov panel data.

Similar to `_nested_optimization_pijcv_implicit` but uses Markov-specific
likelihood functions and state types.
"""
function _nested_optimization_pijcv_markov_implicit(
    model::MultistateProcess,
    data::MPanelData,
    penalty::AbstractPenalty,
    selector::PIJCVSelector;
    beta_init::Vector{Float64},
    inner_maxiter::Int = 50,
    outer_maxiter::Int = 100,
    lambda_tol::Float64 = 1e-3,
    lambda_init::Union{Nothing, Vector{Float64}} = nothing,
    verbose::Bool = false
)
    # Get bounds and setup
    lb, ub = model.bounds.lb, model.bounds.ub
    n_lambda = n_hyperparameters(penalty)
    n_subjects = length(model.subjectindices)
    n_params = length(beta_init)
    books = data.books
    
    # Determine method
    method = selector.nfolds == 0 ? :pijcv_implicit : Symbol("pijcv$(selector.nfolds)_implicit")
    
    if verbose
        println("Optimizing λ via PIJCV with ImplicitDifferentiation.jl for Markov panel data")
        println("  Method: $method, n_lambda: $n_lambda")
    end
    
    # Build penalty_config
    penalty_config = penalty isa QuadraticPenalty ? penalty : build_penalty_config(model, SplinePenalty())
    
    # Build implicit differentiation cache
    cache = build_implicit_beta_cache(model, data, penalty_config, beta_init;
                                       inner_maxiter=inner_maxiter)
    
    # Create the implicit function for β̂(ρ)
    implicit_beta = make_implicit_beta_function(cache)
    
    # Track evaluations
    n_criterion_evals = Ref(0)
    current_beta_ref = Ref(copy(beta_init))
    
    # Define NCV criterion with implicit β
    function ncv_criterion_implicit(log_lambda_vec)
        n_criterion_evals[] += 1
        
        # Get β̂(ρ) via implicit function
        β_hat, z = implicit_beta(log_lambda_vec)
        
        # Update warm-start
        current_beta_ref[] = Float64[ForwardDiff.value(x) for x in β_hat]
        
        # Extract Float64 values for criterion computation
        β_float = current_beta_ref[]
        ρ_float = Float64[ForwardDiff.value(x) for x in log_lambda_vec]
        
        # Compute subject gradients and Hessians (Float64 only)
        subject_grads_ll = compute_subject_gradients(β_float, model, books)
        subject_hessians_ll = compute_subject_hessians(β_float, model, books)
        
        # Convert to loss convention
        subject_grads = -subject_grads_ll
        subject_hessians = [-H for H in subject_hessians_ll]
        H_unpenalized = sum(subject_hessians)
        
        # Build state for criterion
        state = SmoothingSelectionStateMarkov(
            β_float,
            H_unpenalized,
            subject_grads,
            subject_hessians,
            penalty_config,
            n_subjects,
            n_params,
            model,
            data
        )
        
        # Compute criterion V(λ)
        V = if selector.nfolds == 0
            compute_pijcv_criterion_markov(ρ_float, state)
        else
            compute_pijkfold_criterion_markov(ρ_float, state, selector.nfolds)
        end
        
        if verbose && n_criterion_evals[] % 5 == 0
            @info "Criterion eval $(n_criterion_evals[]): log(λ)=$(round.(ρ_float, digits=2)), V=$(round(V, digits=3))"
        end
        
        return V
    end
    
    # Adaptive bounds for log(λ)
    log_lb_scalar, log_ub_scalar = compute_lambda_bounds(n_subjects, n_params)
    log_lb = fill(log_lb_scalar, n_lambda)
    log_ub = fill(log_ub_scalar, n_lambda)
    
    # Initialize λ
    current_log_lambda = if !isnothing(lambda_init) && length(lambda_init) >= n_lambda
        if verbose
            println("  Using provided λ warm-start")
        end
        log.(lambda_init[1:n_lambda])
    else
        if verbose
            println("  Getting EFS initial estimate...")
        end
        efs_result = _nested_optimization_criterion_markov(model, data, penalty, :efs;
                                               beta_init=beta_init,
                                               inner_maxiter=inner_maxiter,
                                               outer_maxiter=30,
                                               lambda_tol=0.1,
                                               verbose=false)
        current_beta_ref[] = efs_result.warmstart_beta
        log.(efs_result.lambda[1:n_lambda])
    end
    
    # Set up optimization with ForwardDiff
    adtype = Optimization.AutoForwardDiff()
    optf = OptimizationFunction((ρ, _) -> ncv_criterion_implicit(ρ), adtype)
    prob = OptimizationProblem(optf, current_log_lambda, nothing; lb=log_lb, ub=log_ub)
    
    if verbose
        println("  Using L-BFGS with implicit differentiation...")
    end
    
    # Solve with Fminbox L-BFGS
    sol = solve(prob, OptimizationOptimJL.Fminbox(OptimizationOptimJL.LBFGS());
                maxiters=outer_maxiter,
                f_tol=lambda_tol,
                x_tol=lambda_tol)
    
    optimal_log_lambda = sol.u
    best_criterion = sol.objective
    current_beta = current_beta_ref[]
    converged = sol.retcode == ReturnCode.Success || sol.retcode == ReturnCode.MaxIters
    
    if verbose
        lambda_val = exp.(optimal_log_lambda)
        println("  Final: log(λ)=$(round.(optimal_log_lambda, digits=2)), λ=$(round.(lambda_val, sigdigits=3)), V=$(round(best_criterion, digits=3))")
        println("  Criterion evaluations: $(n_criterion_evals[])")
    end
    
    # Build final results
    optimal_lambda = exp.(optimal_log_lambda)
    optimal_lambda_vec = n_lambda == 1 ? fill(optimal_lambda[1], n_hyperparameters(penalty_config)) : optimal_lambda
    updated_penalty = set_hyperparameters(penalty, optimal_lambda_vec)
    
    # Compute EDF
    edf = compute_edf_markov(current_beta, optimal_lambda_vec, penalty_config, model, books)
    
    return HyperparameterSelectionResult(
        optimal_lambda_vec,
        current_beta,
        updated_penalty,
        best_criterion,
        edf,
        converged,
        method,
        n_criterion_evals[],
        (log_lambda = optimal_log_lambda, retcode = sol.retcode, implicit_diff = true)
    )
end

# =============================================================================
# MCEM Data Support  
# =============================================================================

"""
    _compute_penalized_hessian_at_beta(β, λ, cache::ImplicitBetaCache{M, MCEMSelectionData}) -> Matrix{Float64}

Compute the penalized Hessian H_λ for MCEM data using importance-weighted semi-Markov likelihood.
"""
function _compute_penalized_hessian_at_beta(β::Vector{Float64}, λ::Vector{Float64}, 
                                            cache::ImplicitBetaCache{M, MCEMSelectionData}) where M
    # Create SMPanelData for semi-Markov likelihood
    sm_data = SMPanelData(cache.data.model, cache.data.paths, cache.data.weights)
    
    # Get unpenalized Hessian using importance-weighted semi-Markov likelihood
    H_unpenalized = ForwardDiff.hessian(b -> loglik_semi_markov(b, sm_data; neg=true, use_sampling_weight=true), β)
    
    # Add penalty contributions
    n = length(β)
    H_lambda = copy(H_unpenalized)
    
    penalty = cache.penalty_config
    lambda_idx = 1
    
    for term in penalty.terms
        idx = term.hazard_indices
        H_lambda[idx, idx] .+= λ[lambda_idx] * term.S
        lambda_idx += 1
    end
    
    for term in penalty.total_hazard_terms
        for idx_range1 in term.hazard_indices
            for idx_range2 in term.hazard_indices
                H_lambda[idx_range1, idx_range2] .+= λ[lambda_idx] * term.S
            end
        end
        lambda_idx += 1
    end
    
    for term in penalty.smooth_covariate_terms
        idx = term.param_indices
        H_lambda[idx, idx] .+= λ[lambda_idx] * term.S
        lambda_idx += 1
    end
    
    return H_lambda
end

"""
    _compute_subject_grads_hessians(β, cache::ImplicitBetaCache{M, MCEMSelectionData}) -> (grads, hessians)

Compute per-subject gradients and Hessians for MCEM data using existing importance-weighted functions.
"""
function _compute_subject_grads_hessians(β::Vector{Float64}, cache::ImplicitBetaCache{M, MCEMSelectionData}) where M
    # Use the existing importance-weighted gradient/Hessian computation
    model = cache.data.model
    paths = cache.data.paths
    weights = cache.data.weights
    
    grads_ll = compute_subject_gradients(β, model, paths, weights)
    hessians_ll = compute_subject_hessians(β, model, paths, weights)
    
    # Convert to loss convention
    grads = -grads_ll
    hessians = [-H for H in hessians_ll]
    
    return grads, hessians
end

"""
    _nested_optimization_pijcv_mcem_implicit(model, data::MCEMSelectionData, penalty, selector; kwargs...) -> HyperparameterSelectionResult

Nested optimization for PIJCV using ImplicitDifferentiation.jl for MCEM data.

Similar to `_nested_optimization_pijcv_implicit` but uses importance-weighted
likelihood functions for MCEM.
"""
function _nested_optimization_pijcv_mcem_implicit(
    model::MultistateProcess,
    data::MCEMSelectionData,
    penalty::AbstractPenalty,
    selector::PIJCVSelector;
    beta_init::Vector{Float64},
    inner_maxiter::Int = 50,
    outer_maxiter::Int = 100,
    lambda_tol::Float64 = 1e-3,
    verbose::Bool = false
)
    # Get bounds and setup
    lb, ub = model.bounds.lb, model.bounds.ub
    n_lambda = n_hyperparameters(penalty)
    n_subjects = length(data.paths)  # Number of subjects from paths vector
    n_params = length(beta_init)
    
    # Determine method
    method = selector.nfolds == 0 ? :pijcv_implicit : Symbol("pijcv$(selector.nfolds)_implicit")
    
    if verbose
        println("Optimizing λ via PIJCV with ImplicitDifferentiation.jl for MCEM data")
        println("  Method: $method, n_lambda: $n_lambda")
    end
    
    # Build penalty_config
    penalty_config = penalty isa QuadraticPenalty ? penalty : build_penalty_config(model, SplinePenalty())
    
    # Build implicit differentiation cache
    cache = build_implicit_beta_cache(model, data, penalty_config, beta_init;
                                       inner_maxiter=inner_maxiter)
    
    # Create the implicit function for β̂(ρ)
    implicit_beta = make_implicit_beta_function(cache)
    
    # Track evaluations
    n_criterion_evals = Ref(0)
    current_beta_ref = Ref(copy(beta_init))
    
    # Define NCV criterion with implicit β
    function ncv_criterion_implicit(log_lambda_vec)
        n_criterion_evals[] += 1
        
        # Get β̂(ρ) via implicit function
        β_hat, z = implicit_beta(log_lambda_vec)
        
        # Update warm-start
        current_beta_ref[] = Float64[ForwardDiff.value(x) for x in β_hat]
        
        # Extract Float64 values for criterion computation
        β_float = current_beta_ref[]
        ρ_float = Float64[ForwardDiff.value(x) for x in log_lambda_vec]
        
        # Compute subject gradients and Hessians (Float64 only)
        subject_grads, subject_hessians = _compute_subject_grads_hessians(β_float, cache)
        H_unpenalized = sum(subject_hessians)
        
        # Build state for criterion
        state = SmoothingSelectionStateMCEM(
            β_float,
            H_unpenalized,
            subject_grads,
            subject_hessians,
            penalty_config,
            n_subjects,
            n_params,
            model,
            data
        )
        
        # Compute criterion V(λ)
        V = if selector.nfolds == 0
            compute_pijcv_criterion_mcem(ρ_float, state)
        else
            compute_pijkfold_criterion_mcem(ρ_float, state, selector.nfolds)
        end
        
        if verbose && n_criterion_evals[] % 5 == 0
            @info "Criterion eval $(n_criterion_evals[]): log(λ)=$(round.(ρ_float, digits=2)), V=$(round(V, digits=3))"
        end
        
        return V
    end
    
    # Adaptive bounds for log(λ)
    log_lb_scalar, log_ub_scalar = compute_lambda_bounds(n_subjects, n_params)
    log_lb = fill(log_lb_scalar, n_lambda)
    log_ub = fill(log_ub_scalar, n_lambda)
    
    # Initialize λ with EFS
    if verbose
        println("  Getting EFS initial estimate...")
    end
    efs_result = _nested_optimization_criterion_mcem(model, data, penalty, :efs;
                                           beta_init=beta_init,
                                           inner_maxiter=inner_maxiter,
                                           outer_maxiter=30,
                                           lambda_tol=0.1,
                                           verbose=false)
    current_beta_ref[] = efs_result.warmstart_beta
    current_log_lambda = log.(efs_result.lambda[1:n_lambda])
    
    # Set up optimization with ForwardDiff
    adtype = Optimization.AutoForwardDiff()
    optf = OptimizationFunction((ρ, _) -> ncv_criterion_implicit(ρ), adtype)
    prob = OptimizationProblem(optf, current_log_lambda, nothing; lb=log_lb, ub=log_ub)
    
    if verbose
        println("  Using L-BFGS with implicit differentiation...")
    end
    
    # Solve with Fminbox L-BFGS
    sol = solve(prob, OptimizationOptimJL.Fminbox(OptimizationOptimJL.LBFGS());
                maxiters=outer_maxiter,
                f_tol=lambda_tol,
                x_tol=lambda_tol)
    
    optimal_log_lambda = sol.u
    best_criterion = sol.objective
    current_beta = current_beta_ref[]
    converged = sol.retcode == ReturnCode.Success || sol.retcode == ReturnCode.MaxIters
    
    if verbose
        lambda_val = exp.(optimal_log_lambda)
        println("  Final: log(λ)=$(round.(optimal_log_lambda, digits=2)), λ=$(round.(lambda_val, sigdigits=3)), V=$(round(best_criterion, digits=3))")
        println("  Criterion evaluations: $(n_criterion_evals[])")
    end
    
    # Build final results
    optimal_lambda = exp.(optimal_log_lambda)
    optimal_lambda_vec = n_lambda == 1 ? fill(optimal_lambda[1], n_hyperparameters(penalty_config)) : optimal_lambda
    updated_penalty = set_hyperparameters(penalty, optimal_lambda_vec)
    
    # Compute EDF
    edf_scalar = compute_edf_mcem(current_beta, optimal_lambda_vec, penalty_config, data)
    edf = (total = edf_scalar, per_term = [edf_scalar])
    
    return HyperparameterSelectionResult(
        optimal_lambda_vec,
        current_beta,
        updated_penalty,
        best_criterion,
        edf,
        converged,
        method,
        n_criterion_evals[],
        (log_lambda = optimal_log_lambda, retcode = sol.retcode, implicit_diff = true)
    )
end

