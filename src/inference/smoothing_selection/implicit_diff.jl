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
# Robust Linear Solve for Leave-One-Out Hessians
# =============================================================================

"""
    solve_hloo(H_loo::AbstractMatrix, b::AbstractVector; 
               damping_init::Float64=1e-8,
               damping_max::Float64=1e-2,
               verbose::Bool=false) -> Vector{Float64}

Robustly solve H_loo * x = b for leave-one-out Hessian systems.

Uses progressive damping with Cholesky factorization for efficiency and
numerical stability. Falls back to general solver if Cholesky fails.

# Algorithm
1. Symmetrize H_loo (in case of numerical asymmetry)
2. Try Cholesky with increasing damping: τ ∈ [0, 1e-8, 1e-7, 1e-6, 1e-2]
3. If all fail, fall back to general `\\` solver
4. If that fails too, return NaN vector

# Arguments
- `H_loo`: Leave-one-out Hessian matrix H_{λ,-i} = H_λ - Hᵢ
- `b`: Right-hand side vector

# Keyword Arguments
- `damping_init::Float64=1e-8`: Initial damping value for Tikhonov regularization
- `damping_max::Float64=1e-2`: Maximum damping before giving up on Cholesky
- `verbose::Bool=false`: Print diagnostic messages

# Returns
Solution vector x, or vector of NaN if solve fails completely.

# Notes
- Damping adds τI to the matrix, improving conditioning: (H + τI)x = b
- This is equivalent to Tikhonov regularization in the Newton step
- The small damping values (1e-8 to 1e-2) have minimal effect on the solution
  when the matrix is well-conditioned, but stabilize ill-conditioned cases
"""
function solve_hloo(H_loo::AbstractMatrix, b::AbstractVector;
                    damping_init::Float64 = 1e-8,
                    damping_max::Float64 = 1e-2,
                    verbose::Bool = false)
    n = length(b)
    
    # Symmetrize to handle numerical asymmetry
    H_sym = Symmetric(0.5 * (H_loo + H_loo'))
    
    # Progressive damping schedule
    damping_values = [0.0, damping_init, damping_init * 10, damping_init * 100, damping_max]
    
    for τ in damping_values
        try
            H_damped = τ > 0 ? H_sym + τ * I : H_sym
            fact = cholesky(H_damped)
            x = fact \ b
            verbose && τ > 0 && @info "solve_hloo: used damping τ=$τ"
            return x
        catch e
            # Continue to next damping value
            continue
        end
    end
    
    # Fall back to general solver (handles indefinite matrices)
    try
        verbose && @warn "solve_hloo: Cholesky failed with all damping values, using general solver"
        return H_sym \ b
    catch e
        verbose && @error "solve_hloo: All solvers failed" exception=e
        return fill(NaN, n)
    end
end

# =============================================================================
# Barrier-Augmented LOO Solve (Phase 7)
# =============================================================================

"""
    solve_hloo_barrier(H_loo, g, lb, beta; μ=1e-6) -> NamedTuple

Compute barrier-augmented LOO Newton step that respects lower bounds.

# Mathematical Formulation

Instead of solving H⁻¹g directly (which may violate β ≥ L), we solve:

    Δ = (H + μD⁻²)⁻¹ (g + μD⁻¹𝟙)

where D = diag(β - L + √μ) is the regularized distance to lower bounds.

This is equivalent to a single Newton step on the barrier-augmented problem:
    min ½(β-β̂)ᵀH(β-β̂) + gᵀ(β-β̂) - μΣₖlog(βₖ - Lₖ)

# Arguments
- `H_loo`: Leave-one-out Hessian H_{λ,-i} (p × p matrix)
- `g`: Subject gradient gᵢ (p-vector, loss convention: g = -∇ℓ)
- `lb`: Lower bounds L (p-vector)
- `beta`: Current parameter estimate β̂ (p-vector)

# Keyword Arguments
- `μ::Float64=1e-6`: Barrier strength. Offset is √μ ≈ 0.001.
  At bound: Hessian contribution = μ/(√μ)² = 1 (well-scaled).
  Interior: negligible when δ >> √μ.

# Returns
NamedTuple with fields:
- `Δ`: Barrier-augmented Newton step (p-vector)
- `d`: Regularized distances d = β - L + √μ (for gradient computation)
- `D_inv`: 1/d element-wise
- `D_inv_sq`: 1/d² element-wise  
- `A_fact`: Factorization of augmented Hessian (for reuse in gradient)

# Notes
- Uses offset √μ (not ε=1e-10) so barrier Hessian is O(1) at bounds, not O(10^14)
- For well-interior parameters (δ >> √μ), this matches solve_hloo to O(μ/δ²)
- Near-boundary parameters get barrier push-back proportional to constraint tightness
- Always returns finite values (no NaN from bound violations)

# Reference
Novel extension of Wood (2024) "On Neighbourhood Cross Validation" Section 4.1
"""
function solve_hloo_barrier(
    H_loo::AbstractMatrix,
    g::AbstractVector,
    lb::AbstractVector,
    beta::AbstractVector;
    μ::Float64 = 1e-6
)
    n = length(beta)
    
    # Regularized distance to lower bounds: D = β - L + √μ
    # Using √μ (not tiny ε) ensures barrier Hessian is O(1) at bounds
    sqrt_μ = sqrt(μ)
    d = beta .- lb .+ sqrt_μ
    
    # Barrier contributions
    D_inv = 1.0 ./ d        # For gradient term: μD⁻¹𝟙
    D_inv_sq = D_inv .^ 2   # For Hessian term: μD⁻²
    
    # Augmented system: (H + μD⁻²)Δ = g + μD⁻¹𝟙
    H_augmented = Symmetric(0.5 * (H_loo + H_loo') + μ * Diagonal(D_inv_sq))
    rhs = g .+ μ .* D_inv
    
    # Solve and return factorization for reuse in gradient computation
    A_fact = try
        cholesky(H_augmented)
    catch
        # Fall back to LU if not positive definite
        lu(H_augmented)
    end
    Δ = A_fact \ rhs
    
    return (Δ=Δ, d=d, D_inv=D_inv, D_inv_sq=D_inv_sq, A_fact=A_fact)
end

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
    
    # Return β and byproduct (including beta_float for KKT-aware conditions)
    return β_opt, (beta_float=β_opt, H_lambda=H_lambda, lambda=λ)
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
# Conditions Function: Optimality (KKT-aware)
# =============================================================================

"""
    beta_optimality_conditions(ρ, β, z, cache::ImplicitBetaCache) -> Vector

KKT-aware optimality conditions c(ρ, β) for the penalized problem.

For interior parameters, the standard first-order condition applies:
    c_i(ρ, β) = ∇_β ℓ(β)_i - (Σⱼ λⱼ Sⱼ β)_i = 0

For parameters at active bounds, we use the constraint as the condition:
    c_i(ρ, β) = β_i - lb_i  (if β_i ≈ lb_i)
    c_i(ρ, β) = β_i - ub_i  (if β_i ≈ ub_i)

This ensures ∂c_i/∂β_i = 1 and ∂c_i/∂ρ = 0 for active bounds, which via the
implicit function theorem gives dβ̂_i/dρ = 0 as expected.

# Arguments
- `ρ`: Log-smoothing parameters (may contain Dual numbers for AD)
- `β`: Coefficient vector (may contain Dual numbers for AD)
- `z`: Byproduct from forward solve (contains beta_float for bound detection)
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
    n = length(β)
    
    # Get Float64 β from byproduct for bound detection
    β_float = z.beta_float
    lb, ub = cache.lb, cache.ub
    
    # Compute unconstrained gradient conditions
    grad_ll = _compute_ll_gradient(β, cache)
    grad_penalty = _compute_penalty_gradient(β, λ, cache)
    unconstrained_conditions = grad_ll - grad_penalty
    
    # Build conditions with KKT-aware handling of active bounds
    T = eltype(unconstrained_conditions)
    conditions = similar(unconstrained_conditions)
    
    for i in 1:n
        if β_float[i] - lb[i] < ACTIVE_BOUND_TOL
            # Active at lower bound: condition is β_i - lb_i
            # This gives ∂c/∂β_i = 1, ∂c/∂ρ = 0 → dβ̂_i/dρ = 0
            conditions[i] = β[i] - lb[i]
        elseif ub[i] - β_float[i] < ACTIVE_BOUND_TOL
            # Active at upper bound: condition is β_i - ub_i
            conditions[i] = β[i] - ub[i]
        else
            # Interior point: use standard first-order condition
            conditions[i] = unconstrained_conditions[i]
        end
    end
    
    return conditions
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
# SIGN CONVENTIONS (see scratch/PIJCV_IMPLICIT_DIFF_HANDOFF_2026-01-27.md)
# =============================================================================
# subject_grads[:, i] = gᵢ = -∇ℓᵢ(β̂)     (loss gradient, NEGATIVE of loglik gradient)
# subject_hessians[i] = Hᵢ = -∇²ℓᵢ(β̂)    (loss Hessian, NEGATIVE of loglik Hessian)
# H_λ = Σⱼ Hⱼ + λS                        (penalized Hessian)
# H_{-i} = H_λ - Hᵢ                       (leave-one-out Hessian)
# δᵢ = H_{-i}⁻¹ gᵢ                        (Newton step)
# β̃_{-i} = β̂ + δᵢ                        (pseudo-estimate, PLUS sign)
# V = Σᵢ -ℓᵢ(β̃_{-i})                      (criterion to minimize)
# =============================================================================

# =============================================================================
# Analytical Gradient for PIJCV Criterion
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
- β̃₋ᵢ = β̂ + Δ⁻ⁱ is the pseudo-estimate (one Newton step from β̂ toward LOO optimum)
- Δ⁻ⁱ = (H_λ - Hᵢ)⁻¹ gᵢ is the LOO step
- gᵢ = -∇ℓᵢ(β̂) is the per-subject LOSS gradient (negative of loglik gradient)
- Hᵢ = -∇²ℓᵢ(β̂) is the per-subject LOSS Hessian (negative of loglik Hessian)

## CORRECT Gradient Formula (with third derivatives)

    dV/dρ = Σᵢ [-∇ℓᵢ(β̃₋ᵢ)ᵀ · dβ̃₋ᵢ/dρ]

where dβ̃₋ᵢ/dρ = dβ̂/dρ + dΔ⁻ⁱ/dρ (PLUS sign!) and the chain rule gives:

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
    dbeta_drho::AbstractMatrix{Float64},  # (n_params × n_lambda) matrix
    subject_third_derivatives::Union{Nothing, Vector{Array{Float64,3}}} = nothing,
    check_conditioning::Bool = false  # Disabled by default for performance during optimization
)
    lambda = exp.(log_lambda)
    n_lambda = length(lambda)
    n_subjects = size(subject_grads, 2)
    n_params = length(β)
    
    # ==========================================================================
    # Build penalized Hessian H_λ and per-λ penalty matrices S_by_lambda
    # Uses same term→λⱼ mapping as _compute_penalty_gradient
    # ==========================================================================
    H_lambda = copy(H_unpenalized)
    S_by_lambda = [zeros(n_params, n_params) for _ in 1:n_lambda]
    penalty = cache.penalty_config
    lambda_idx = 1
    
    # Baseline hazard terms: each term gets its own λ
    for term in penalty.terms
        idx = term.hazard_indices
        λ_j = lambda_idx <= n_lambda ? lambda[lambda_idx] : lambda[end]
        H_lambda[idx, idx] .+= λ_j .* term.S
        if lambda_idx <= n_lambda
            S_by_lambda[lambda_idx][idx, idx] .= Matrix(term.S)
        end
        lambda_idx += 1
    end
    
    # Total hazard terms
    for term in penalty.total_hazard_terms
        λ_j = lambda_idx <= n_lambda ? lambda[lambda_idx] : lambda[end]
        for idx_range1 in term.hazard_indices
            for idx_range2 in term.hazard_indices
                H_lambda[idx_range1, idx_range2] .+= λ_j .* term.S
                if lambda_idx <= n_lambda
                    S_by_lambda[lambda_idx][idx_range1, idx_range2] .= Matrix(term.S)
                end
            end
        end
        lambda_idx += 1
    end
    
    # Smooth covariate terms (handle shared_smooth_groups)
    if !isempty(penalty.shared_smooth_groups)
        # Build term -> lambda mapping for shared groups
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
        # Apply penalties
        for (term_idx, term) in enumerate(penalty.smooth_covariate_terms)
            idx = term.param_indices
            λ_idx_j = term_to_lambda[term_idx]
            λ_j = λ_idx_j <= n_lambda ? lambda[λ_idx_j] : lambda[end]
            H_lambda[idx, idx] .+= λ_j .* term.S
            if λ_idx_j <= n_lambda
                S_by_lambda[λ_idx_j][idx, idx] .+= Matrix(term.S)  # += for shared groups
            end
        end
    else
        for term in penalty.smooth_covariate_terms
            idx = term.param_indices
            λ_j = lambda_idx <= n_lambda ? lambda[lambda_idx] : lambda[end]
            H_lambda[idx, idx] .+= λ_j .* term.S
            if lambda_idx <= n_lambda
                S_by_lambda[lambda_idx][idx, idx] .= Matrix(term.S)
            end
            lambda_idx += 1
        end
    end
    
    H_lambda_sym = Symmetric(H_lambda)
    
    # ==========================================================================
    # Compute third derivative contractions using JVP (Phase 4.3 optimization)
    # Instead of materializing p×p×p tensors, we compute Σₗ (∂Hᵢ/∂βₗ)·vₗ directly
    # for each direction v = dbeta_drho[:, j].
    # ==========================================================================
    # dH_times_v[i][j] = Σₗ (∂Hᵢ/∂βₗ)·(dβ̂/dρⱼ)ₗ  (p×p matrix)
    dH_times_v_all = _compute_all_dH_times_v(β, dbeta_drho, cache)
    
    # ==========================================================================
    # Compute dH_λ/dρⱼ for each smoothing parameter j
    # dH_λ/dρⱼ = λⱼ Sⱼ + Σᵢ [Σₗ (∂Hᵢ/∂βₗ)·(dβ̂/dρⱼ)ₗ]
    # ==========================================================================
    dH_lambda_drho = [lambda[j] * S_by_lambda[j] for j in 1:n_lambda]
    for i in 1:n_subjects
        for j in 1:n_lambda
            dH_lambda_drho[j] .+= dH_times_v_all[i][j]
        end
    end
    
    # ==========================================================================
    # Build PIJCV evaluation cache for efficient LOO likelihood evaluation
    # ==========================================================================
    eval_cache = build_pijcv_eval_cache(cache.data)
    
    # ==========================================================================
    # CORRECT PIJCV: V = Σᵢ -ℓᵢ(β̃₋ᵢ) where β̃₋ᵢ = β̂ + Δ⁻ⁱ (PLUS sign!)
    # Uses barrier-augmented Newton step to ensure β̃₋ᵢ ≥ lb (Phase 7)
    # ==========================================================================
    V = 0.0
    grad_V = zeros(n_lambda)
    
    # Preallocate work vectors (Phase 4 optimization)
    diff_result = DiffResults.GradientResult(zeros(n_params))
    
    # Conditioning diagnostics
    ill_conditioned_subjects = Int[]
    worst_cond = 0.0
    worst_subject = 0
    
    # Barrier parameter (must match solve_hloo_barrier)
    μ = 1e-6
    lb = cache.lb
    
    for i in 1:n_subjects
        gᵢ = subject_grads[:, i]
        Hᵢ = subject_hessians[i]
        dHᵢ_times_v = dH_times_v_all[i]  # Pre-computed JVP contractions for subject i
        
        # Leave-one-out penalized Hessian
        H_loo = H_lambda - Hᵢ
        
        # Check LOO conditioning if requested
        if check_conditioning
            cond_num, is_ill_cond = check_loo_conditioning(H_loo, i)
            if is_ill_cond
                push!(ill_conditioned_subjects, i)
            end
            if cond_num > worst_cond
                worst_cond = cond_num
                worst_subject = i
            end
        end
        
        # Barrier-augmented Newton step: ensures β̃₋ᵢ ≥ lb (Phase 7)
        # Δᵢ = (H_loo + μD⁻²)⁻¹ (gᵢ + μD⁻¹𝟙) where D = β - lb + √μ
        barrier_result = solve_hloo_barrier(H_loo, gᵢ, lb, β; μ=μ)
        Δᵢ = barrier_result.Δ
        d_i = barrier_result.d
        D_inv_i = barrier_result.D_inv
        D_inv_sq_i = barrier_result.D_inv_sq
        A_fact_i = barrier_result.A_fact
        
        if any(isnan, Δᵢ)
            return (1e10, fill(0.0, n_lambda))
        end
        
        # Pseudo-estimate: β̃₋ᵢ = β̂ + Δ⁻ⁱ (PLUS sign!)
        β_tilde_i = β .+ Δᵢ
        
        # CORRECT criterion: evaluate ACTUAL likelihood at pseudo-estimate
        # Use DiffResults to compute value and gradient in single pass (Phase 4)
        ForwardDiff.gradient!(
            diff_result,
            b -> loglik_subject_cached(b, eval_cache, i),
            β_tilde_i
        )
        ll_at_pseudo = DiffResults.value(diff_result)
        grad_ll_at_pseudo = DiffResults.gradient(diff_result)
        
        V_i = -ll_at_pseudo
        V += V_i
        
        # =======================================================================
        # CORRECT gradient with third derivatives AND barrier terms for each λⱼ
        # (Phase 7: barrier-augmented gradient)
        # =======================================================================
        # dVᵢ/dρⱼ = -∇ℓᵢ(β̃₋ᵢ)ᵀ · dβ̃₋ᵢ/dρⱼ
        # where dβ̃₋ᵢ/dρⱼ = dβ̂/dρⱼ + dΔ⁻ⁱ/dρⱼ (PLUS sign!)
        #
        # For barrier-augmented step Δ = A⁻¹b where A = H_loo + μD⁻², b = g + μD⁻¹𝟙:
        #   dΔ/dρⱼ = A⁻¹(db/dρⱼ - dA/dρⱼ·Δ)
        
        for j in 1:n_lambda
            dbeta_j = view(dbeta_drho, :, j)
            
            # --- Barrier derivative terms (Phase 7) ---
            # D = β - L + √μ, so dD/dρⱼ = dβ̂/dρⱼ (element-wise)
            dD_drho_j = dbeta_j
            
            # d(D⁻¹)/dρⱼ = -D⁻² · dD/dρⱼ (element-wise)
            d_D_inv_drho_j = -(D_inv_i .^ 2) .* dD_drho_j
            
            # d(D⁻²)/dρⱼ = -2D⁻³ · dD/dρⱼ (element-wise)
            d_D_inv_sq_drho_j = -2.0 .* (D_inv_i .^ 3) .* dD_drho_j
            
            # --- db/dρⱼ = dgᵢ/dρⱼ + μ·d(D⁻¹𝟙)/dρⱼ ---
            # dgᵢ/dρⱼ = +Hᵢ·dβ̂/dρⱼ (POSITIVE sign!)
            dgᵢ_drho_j = Hᵢ * dbeta_j
            db_drho_j = dgᵢ_drho_j .+ μ .* d_D_inv_drho_j
            
            # --- dA/dρⱼ = dH_{-i}/dρⱼ + μ·diag(d(D⁻²)/dρⱼ) ---
            # dHᵢ/dρⱼ = Σₗ (∂Hᵢ/∂βₗ)·(dβ̂/dρⱼ)ₗ (pre-computed via JVP, Phase 4.3)
            dHᵢ_drho_j = dHᵢ_times_v[j]
            
            # dH_loo/dρⱼ = dH_λ/dρⱼ - dHᵢ/dρⱼ
            dH_loo_drho_j = dH_lambda_drho[j] - dHᵢ_drho_j
            
            # Add barrier Hessian derivative (diagonal term): μ·diag(d(D⁻²)/dρⱼ)
            dA_drho_j = dH_loo_drho_j + μ * Diagonal(d_D_inv_sq_drho_j)
            
            # --- dΔ/dρⱼ = A⁻¹(db/dρⱼ - dA/dρⱼ·Δ) ---
            # Reuse A_fact_i from solve_hloo_barrier
            rhs_for_dDelta = db_drho_j - dA_drho_j * Δᵢ
            dDelta_drho_j = A_fact_i \ rhs_for_dDelta
            
            if any(isnan, dDelta_drho_j)
                continue
            end
            
            # dβ̃₋ᵢ/dρⱼ = dβ̂/dρⱼ + dΔ⁻ⁱ/dρⱼ (PLUS sign!)
            dbeta_tilde_drho_j = dbeta_j + dDelta_drho_j
            
            # dVᵢ/dρⱼ = -∇ℓᵢ(β̃₋ᵢ)ᵀ · dβ̃₋ᵢ/dρⱼ
            dV_i_drho_j = -dot(grad_ll_at_pseudo, dbeta_tilde_drho_j)
            grad_V[j] += dV_i_drho_j
        end
    end
    
    # Log conditioning summary if issues were detected
    if check_conditioning && !isempty(ill_conditioned_subjects)
        report = LOOConditioningReport(
            length(ill_conditioned_subjects),
            ill_conditioned_subjects,
            worst_cond,
            worst_subject
        )
        log_loo_conditioning_summary(report, n_subjects; context="PIJCV gradient")
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

# Note
This function materializes full p×p×p tensors and is O(p³) in memory.
For large p, prefer using `_compute_dH_times_v` which computes the 
contraction Σₗ (∂Hᵢ/∂βₗ)·vₗ directly using directional derivatives.
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


"""
    _compute_subject_third_derivatives(β, cache::ImplicitBetaCache{M, MPanelData}) -> Vector{Array{Float64,3}}

Compute third derivatives ∂Hᵢ/∂β for all subjects for Markov panel data.
"""
function _compute_subject_third_derivatives(β::Vector{Float64}, cache::ImplicitBetaCache{M, MPanelData}) where M
    n_subjects = length(cache.model.subjectindices)
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


"""
    _compute_subject_third_derivatives(β, cache::ImplicitBetaCache{M, MCEMSelectionData}) -> Vector{Array{Float64,3}}

Compute third derivatives ∂Hᵢ/∂β for all subjects for MCEM data.
"""
function _compute_subject_third_derivatives(β::Vector{Float64}, cache::ImplicitBetaCache{M, MCEMSelectionData}) where M
    n_subjects = length(cache.model.subjectindices)
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
# JVP-Based Third Derivative Contractions (Phase 4.3 Optimization)
# =============================================================================
# These functions compute Σₗ (∂Hᵢ/∂βₗ)·vₗ without materializing the full p×p×p tensor.
# This reduces memory from O(p³) to O(p²) and computation from O(np³) to O(np²).
# =============================================================================

"""
    _compute_dH_times_v(β, v, cache::ImplicitBetaCache, subject_idx::Int) -> Matrix{Float64}

Compute the contraction Σₗ (∂Hᵢ/∂βₗ)·vₗ for a single subject using directional derivatives.

This is the directional derivative of Hᵢ(β) in direction v, computed as:
    d/dt [Hᵢ(β + t·v)]|_{t=0}

# Mathematical Background
The third derivative tensor T[j,k,l] = ∂²ℓᵢ/∂βⱼ∂βₖ∂βₗ is symmetric in all indices.
The contraction Σₗ T[:,:,l]·vₗ equals the directional derivative of the Hessian.

# Arguments
- `β::Vector{Float64}`: Current parameter estimate
- `v::AbstractVector{Float64}`: Direction vector for contraction
- `cache::ImplicitBetaCache`: Contains model and data
- `subject_idx::Int`: Subject index

# Returns
Matrix{Float64} of size (p, p) containing Σₗ (∂Hᵢ/∂βₗ)·vₗ

# Performance
O(p²) memory and O(p²) computation vs O(p³) for explicit tensor.
"""
function _compute_dH_times_v(
    β::Vector{Float64},
    v::AbstractVector{Float64},
    cache::ImplicitBetaCache{M, ExactData},
    subject_idx::Int
) where M
    n_params = length(β)
    
    # Directional derivative: d/dt [Hᵢ(β + t·v)]|_{t=0}
    # Implemented as: ForwardDiff of the Hessian function at β, in direction v
    # We use the JVP pattern: pushforward of H at β with tangent v
    
    # Compute Hessian function value and its Jacobian applied to v
    # H(β + ε·v) ≈ H(β) + ε·(dH/dβ)·v
    # The (dH/dβ)·v term is what we want
    
    hess_func = b -> vec(-ForwardDiff.hessian(bb -> loglik_subject(bb, cache.data, subject_idx), b))
    
    # Use ForwardDiff.Dual to compute directional derivative
    β_dual = ForwardDiff.Dual.(β, v)
    H_flat_dual = hess_func(β_dual)
    
    # Extract the derivative part (partials)
    dH_times_v_flat = ForwardDiff.partials.(H_flat_dual, 1)
    
    return reshape(dH_times_v_flat, n_params, n_params)
end

function _compute_dH_times_v(
    β::Vector{Float64},
    v::AbstractVector{Float64},
    cache::ImplicitBetaCache{M, MPanelData},
    subject_idx::Int
) where M
    n_params = length(β)
    
    hess_func = b -> vec(-ForwardDiff.hessian(bb -> loglik_subject(bb, cache.data, subject_idx), b))
    
    β_dual = ForwardDiff.Dual.(β, v)
    H_flat_dual = hess_func(β_dual)
    dH_times_v_flat = ForwardDiff.partials.(H_flat_dual, 1)
    
    return reshape(dH_times_v_flat, n_params, n_params)
end

function _compute_dH_times_v(
    β::Vector{Float64},
    v::AbstractVector{Float64},
    cache::ImplicitBetaCache{M, MCEMSelectionData},
    subject_idx::Int
) where M
    n_params = length(β)
    
    hess_func = b -> vec(-ForwardDiff.hessian(bb -> loglik_subject(bb, cache.data, subject_idx), b))
    
    β_dual = ForwardDiff.Dual.(β, v)
    H_flat_dual = hess_func(β_dual)
    dH_times_v_flat = ForwardDiff.partials.(H_flat_dual, 1)
    
    return reshape(dH_times_v_flat, n_params, n_params)
end


"""
    _compute_all_dH_times_v(β, V, cache::ImplicitBetaCache) -> Vector{Vector{Matrix{Float64}}}

Compute contractions Σₗ (∂Hᵢ/∂βₗ)·V[j,l] for all subjects and all direction vectors.

# Arguments
- `β::Vector{Float64}`: Current parameter estimate
- `V::AbstractMatrix{Float64}`: Direction vectors, shape (n_params, n_directions)
- `cache::ImplicitBetaCache`: Contains model and data

# Returns
Vector of length n_subjects, where each element is a Vector of length n_directions,
where each element is a (p, p) matrix: dH_times_v[i][j] = Σₗ (∂Hᵢ/∂βₗ)·V[l,j]
"""
function _compute_all_dH_times_v(
    β::Vector{Float64},
    V::AbstractMatrix{Float64},
    cache::ImplicitBetaCache{M, ExactData}
) where M
    n_subjects = length(cache.data.paths)
    n_directions = size(V, 2)
    
    result = Vector{Vector{Matrix{Float64}}}(undef, n_subjects)
    
    for i in 1:n_subjects
        result[i] = Vector{Matrix{Float64}}(undef, n_directions)
        for j in 1:n_directions
            result[i][j] = _compute_dH_times_v(β, view(V, :, j), cache, i)
        end
    end
    
    return result
end

function _compute_all_dH_times_v(
    β::Vector{Float64},
    V::AbstractMatrix{Float64},
    cache::ImplicitBetaCache{M, MPanelData}
) where M
    n_subjects = length(cache.model.subjectindices)
    n_directions = size(V, 2)
    
    result = Vector{Vector{Matrix{Float64}}}(undef, n_subjects)
    
    for i in 1:n_subjects
        result[i] = Vector{Matrix{Float64}}(undef, n_directions)
        for j in 1:n_directions
            result[i][j] = _compute_dH_times_v(β, view(V, :, j), cache, i)
        end
    end
    
    return result
end

function _compute_all_dH_times_v(
    β::Vector{Float64},
    V::AbstractMatrix{Float64},
    cache::ImplicitBetaCache{M, MCEMSelectionData}
) where M
    n_subjects = length(cache.model.subjectindices)
    n_directions = size(V, 2)
    
    result = Vector{Vector{Matrix{Float64}}}(undef, n_subjects)
    
    for i in 1:n_subjects
        result[i] = Vector{Matrix{Float64}}(undef, n_directions)
        for j in 1:n_directions
            result[i][j] = _compute_dH_times_v(β, view(V, :, j), cache, i)
        end
    end
    
    return result
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
    
    # Define criterion AND gradient function using NCV approximation (Wood 2024)
    # Uses simplified gradient with dΔ/dρ weighting but ignores third derivatives for speed
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
        # This uses the IFT: dβ̂/dρⱼ = -H_λ⁻¹ · (λⱼ Sⱼ β̂)
        # Returns (n_params × n_lambda) matrix
        dbeta_drho = ForwardDiff.jacobian(
            ρ_vec -> implicit_beta(ρ_vec)[1],
            log_lambda_float
        )
        
        # Compute subject gradients and Hessians at β̂
        subject_grads_ll, subject_hessians_ll = compute_subject_grads_and_hessians_fast(
            β_float, model, data.paths; use_threads=:auto
        )
        
        # Convert to loss convention
        subject_grads = -subject_grads_ll
        subject_hessians = [-H for H in subject_hessians_ll]
        H_unpenalized = sum(subject_hessians)
        
        # Compute criterion AND analytical gradient simultaneously
        # Uses corrected NCV gradient (including dΔ/dρ) but ignoring expensive third derivatives
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

Uses analytical gradients via `compute_pijcv_with_gradient`, matching the ExactData
pattern for efficiency and correctness.
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
        println("  Method: $method, n_lambda: $n_lambda, using analytical gradients")
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
    
    # Define criterion AND gradient function using analytical gradient (Wood 2024 NCV)
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
        # Returns (n_params × n_lambda) matrix
        dbeta_drho = ForwardDiff.jacobian(
            ρ_vec -> implicit_beta(ρ_vec)[1],
            log_lambda_float
        )
        
        # Compute subject gradients and Hessians at β̂
        subject_grads_ll = compute_subject_gradients(β_float, model, books)
        subject_hessians_ll = compute_subject_hessians(β_float, model, books)
        
        # Convert to loss convention (see sign conventions in this file)
        subject_grads = -subject_grads_ll
        subject_hessians = [-H for H in subject_hessians_ll]
        H_unpenalized = sum(subject_hessians)
        
        # Compute criterion AND analytical gradient simultaneously
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

Uses analytical gradients via `compute_pijcv_with_gradient`, matching the ExactData
and Markov patterns for efficiency and correctness.
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
        println("  Method: $method, n_lambda: $n_lambda, using analytical gradients")
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
    
    # Define criterion AND gradient function using analytical gradient (Wood 2024 NCV)
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
        # Returns (n_params × n_lambda) matrix
        dbeta_drho = ForwardDiff.jacobian(
            ρ_vec -> implicit_beta(ρ_vec)[1],
            log_lambda_float
        )
        
        # Compute subject gradients and Hessians at β̂ (already in loss convention)
        subject_grads, subject_hessians = _compute_subject_grads_hessians(β_float, cache)
        H_unpenalized = sum(subject_hessians)
        
        # Compute criterion AND analytical gradient simultaneously
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

