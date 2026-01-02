# =============================================================================
# Smoothing Parameter Selection for Penalized Splines
# =============================================================================
#
# Implements PIJCV (Predictive Infinitesimal Jackknife Cross-Validation) for 
# automatic selection of smoothing parameters λ in penalized spline models.
#
# Based on Wood (2024): "Neighbourhood Cross-Validation" arXiv:2404.16490
#
# Key insight: Inner optimization uses PENALIZED loss, outer optimization
# minimizes UNPENALIZED prediction error via leave-one-out approximation.
#
# =============================================================================

using LinearAlgebra
using Optim

"""
    PIJCVState

Internal state for PIJCV computation, storing cached matrices and intermediate results.

# Fields
- `beta_hat::Vector{Float64}`: Current coefficient estimate
- `H_unpenalized::Matrix{Float64}`: Unpenalized Hessian (sum of subject Hessians)
- `subject_grads::Matrix{Float64}`: Subject gradients (p × n)
- `subject_hessians::Vector{Matrix{Float64}}`: Subject Hessians
- `penalty_config::PenaltyConfig`: Penalty configuration
- `n_subjects::Int`: Number of subjects
- `n_params::Int`: Number of parameters
- `model::MultistateProcess`: Model for likelihood evaluation
- `data::ExactData`: Data container
"""
mutable struct PIJCVState
    beta_hat::Vector{Float64}
    H_unpenalized::Matrix{Float64}
    subject_grads::Matrix{Float64}
    subject_hessians::Vector{Matrix{Float64}}
    penalty_config::PenaltyConfig
    n_subjects::Int
    n_params::Int
    model::MultistateProcess
    data::ExactData
end

"""
    compute_pijcv_criterion(log_lambda::Vector{Float64}, state::PIJCVState) -> Float64

Compute the PIJCV criterion V(λ) for given log-smoothing parameters.

The criterion is the sum of leave-one-out prediction errors:
    V(λ) = Σᵢ D(yᵢ, θᵢ⁽⁻ⁱ⁾)

where D is the unpenalized negative log-likelihood contribution and θᵢ⁽⁻ⁱ⁾
is the prediction using parameters estimated with subject i omitted.

# Arguments
- `log_lambda`: Vector of log-smoothing parameters (one per penalty term)
- `state`: PIJCVState with cached gradients/Hessians

# Returns
- `Float64`: PIJCV criterion value (lower is better)
"""
function compute_pijcv_criterion(log_lambda::Vector{Float64}, state::PIJCVState)
    lambda = exp.(log_lambda)
    
    # Build penalized Hessian: H_λ = H + Σⱼ λⱼ Sⱼ
    H_lambda = copy(state.H_unpenalized)
    _add_penalty_to_hessian!(H_lambda, lambda, state.penalty_config)
    
    # Try Cholesky factorization
    H_lambda_sym = Symmetric(H_lambda)
    chol_fact = try
        cholesky(H_lambda_sym)
    catch e
        # If Cholesky fails, H_λ is not positive definite
        # Return a large value to steer optimization away
        return 1e10
    end
    
    # Get subject-level likelihoods at current estimate
    ll_subj_base = loglik_exact(state.beta_hat, state.data; neg=false, return_ll_subj=true)
    
    # Compute PIJCV criterion: sum over subjects
    V = 0.0
    for i in 1:state.n_subjects
        # Get subject i's gradient and Hessian
        g_i = @view state.subject_grads[:, i]
        H_i = state.subject_hessians[i]
        
        # Leave-one-out Hessian: H_{λ,-i} = H_λ - H_i
        H_lambda_loo = H_lambda - H_i
        
        # Solve for coefficient perturbation: Δ⁽⁻ⁱ⁾ = H_{λ,-i}⁻¹ g_i
        H_loo_sym = Symmetric(H_lambda_loo)
        
        delta_i = try
            H_loo_sym \ g_i
        catch e
            # If solve fails, return large value
            return 1e10
        end
        
        # Leave-one-out prediction: β̂⁽⁻ⁱ⁾ = β̂ - Δ⁽⁻ⁱ⁾
        beta_loo = state.beta_hat - delta_i
        
        # Evaluate UNPENALIZED subject loss at leave-one-out parameters
        # Use linear approximation: ℓᵢ(β̂ - Δ) ≈ ℓᵢ(β̂) - gᵢᵀΔ
        # The loss (negative log-lik) is: D_i ≈ -ℓᵢ(β̂) + gᵢᵀΔ
        D_i = -ll_subj_base[i] + dot(g_i, delta_i)
        V += D_i
    end
    
    return V
end

"""
    _add_penalty_to_hessian!(H::Matrix{Float64}, lambda::Vector{Float64}, 
                              config::PenaltyConfig)

Add penalty matrices to Hessian in-place: H += Σⱼ λⱼ Sⱼ
"""
function _add_penalty_to_hessian!(H::Matrix{Float64}, lambda::Vector{Float64}, 
                                   config::PenaltyConfig)
    lambda_idx = 1
    
    for term in config.terms
        idx = term.hazard_indices
        H[idx, idx] .+= lambda[lambda_idx] .* term.S
        lambda_idx += 1
    end
    
    # Handle total hazard terms
    for term in config.total_hazard_terms
        # Total hazard penalty uses Kronecker structure
        # For now, use the simplified addition per hazard
        for idx_range in term.hazard_indices
            H[idx_range, idx_range] .+= lambda[lambda_idx] .* term.S
        end
        lambda_idx += 1
    end
end

"""
    select_smoothing_parameters(model::MultistateProcess, data::ExactData,
                                 penalty_config::PenaltyConfig,
                                 beta_init::Vector{Float64};
                                 method::Symbol=:pijcv,
                                 maxiter::Int=100,
                                 verbose::Bool=false) -> NamedTuple

Select smoothing parameters using PIJCV or GCV.

# Arguments
- `model`: MultistateProcess model
- `data`: ExactData container
- `penalty_config`: Penalty configuration with initial λ values
- `beta_init`: Initial coefficient estimate (warm start)
- `method`: Selection method (:pijcv or :gcv)
- `maxiter`: Maximum outer iterations for λ optimization
- `verbose`: Print progress

# Returns
NamedTuple with:
- `lambda`: Optimal smoothing parameters
- `beta`: Final coefficient estimate
- `criterion`: Final criterion value
- `converged`: Whether optimization converged
- `method_used`: Actual method used (may fall back from PIJCV to GCV)
- `penalty_config`: Updated penalty configuration with optimal λ
"""
function select_smoothing_parameters(model::MultistateProcess, data::ExactData,
                                      penalty_config::PenaltyConfig,
                                      beta_init::Vector{Float64};
                                      method::Symbol=:pijcv,
                                      maxiter::Int=100,
                                      verbose::Bool=false)
    n_lambda = penalty_config.n_lambda
    n_lambda > 0 || return (lambda=Float64[], beta=beta_init, criterion=NaN, 
                            converged=true, method_used=:none, penalty_config=penalty_config)
    
    # Initialize log-lambda from current penalty config
    log_lambda_init = zeros(n_lambda)
    lambda_idx = 1
    for term in penalty_config.terms
        log_lambda_init[lambda_idx] = log(term.lambda)
        lambda_idx += 1
    end
    for term in penalty_config.total_hazard_terms
        log_lambda_init[lambda_idx] = log(term.lambda_H)
        lambda_idx += 1
    end
    
    # Compute subject-level gradients and Hessians at initial estimate
    if verbose
        println("Computing subject-level gradients and Hessians...")
    end
    samplepaths = data.paths
    subject_grads = compute_subject_gradients(beta_init, model, samplepaths)
    subject_hessians = compute_subject_hessians(beta_init, model, samplepaths)
    
    # Aggregate unpenalized Hessian
    n_params = length(beta_init)
    H_unpenalized = zeros(n_params, n_params)
    for H_i in subject_hessians
        H_unpenalized .+= H_i
    end
    
    # Create PIJCV state
    state = PIJCVState(
        copy(beta_init),
        H_unpenalized,
        subject_grads,
        subject_hessians,
        penalty_config,
        length(samplepaths),
        n_params,
        model,
        data
    )
    
    # Try PIJCV first
    method_used = method
    converged = false
    final_lambda = exp.(log_lambda_init)
    final_criterion = Inf
    
    if method == :pijcv
        if verbose
            println("Running PIJCV optimization...")
        end
        
        # Optimize log-lambda using BFGS
        objective = log_lam -> compute_pijcv_criterion(log_lam, state)
        
        result = try
            optimize(objective, log_lambda_init, BFGS(), 
                     Optim.Options(iterations=maxiter, show_trace=verbose))
        catch e
            if verbose
                println("PIJCV failed: $e")
                println("Falling back to GCV...")
            end
            nothing
        end
        
        if !isnothing(result) && Optim.converged(result)
            final_lambda = exp.(Optim.minimizer(result))
            final_criterion = Optim.minimum(result)
            converged = true
            
            # Check for degenerate solution
            if any(final_lambda .< 1e-10) || any(final_lambda .> 1e10)
                if verbose
                    println("PIJCV produced degenerate λ, falling back to GCV...")
                end
                converged = false
            end
        end
        
        # Fall back to GCV if PIJCV failed
        if !converged
            method_used = :gcv
        end
    end
    
    if method_used == :gcv
        if verbose
            println("Running GCV optimization...")
        end
        
        result = try
            gcv_objective = log_lam -> compute_gcv_criterion(log_lam, state)
            optimize(gcv_objective, log_lambda_init, BFGS(),
                     Optim.Options(iterations=maxiter, show_trace=verbose))
        catch e
            if verbose
                println("GCV also failed: $e")
                println("Using default high penalty...")
            end
            nothing
        end
        
        if !isnothing(result) && Optim.converged(result)
            final_lambda = exp.(Optim.minimizer(result))
            final_criterion = Optim.minimum(result)
            converged = true
        else
            # Final fallback: use high penalty
            final_lambda = fill(100.0, n_lambda)
            final_criterion = NaN
            converged = false
            method_used = :fallback
            
            if verbose
                @warn "Smoothing parameter selection failed. Using default high penalty (λ=100)."
            end
        end
    end
    
    # Update penalty config with optimal lambda
    updated_config = _update_penalty_lambda(penalty_config, final_lambda)
    
    return (
        lambda = final_lambda,
        beta = state.beta_hat,
        criterion = final_criterion,
        converged = converged,
        method_used = method_used,
        penalty_config = updated_config
    )
end

"""
    compute_gcv_criterion(log_lambda::Vector{Float64}, state::PIJCVState) -> Float64

Compute Generalized Cross-Validation criterion.

GCV approximates leave-one-out CV using the effective degrees of freedom:
    V_GCV(λ) = n * deviance / (n - edf)²

where edf = tr(A) is the trace of the influence matrix.
"""
function compute_gcv_criterion(log_lambda::Vector{Float64}, state::PIJCVState)
    lambda = exp.(log_lambda)
    
    # Build penalized Hessian
    H_lambda = copy(state.H_unpenalized)
    _add_penalty_to_hessian!(H_lambda, lambda, state.penalty_config)
    
    # Compute influence matrix trace: edf = tr(H_unpen * H_lambda^{-1})
    H_lambda_sym = Symmetric(H_lambda)
    
    H_lambda_inv = try
        inv(H_lambda_sym)
    catch e
        return 1e10
    end
    
    # edf = trace of influence matrix
    edf = tr(state.H_unpenalized * H_lambda_inv)
    
    # Compute deviance (sum of unpenalized losses)
    ll_subj = loglik_exact(state.beta_hat, state.data; neg=false, return_ll_subj=true)
    deviance = -sum(ll_subj)
    
    n = state.n_subjects
    
    # GCV criterion
    if n <= edf
        return 1e10  # Invalid: more parameters than data
    end
    
    gcv = n * deviance / (n - edf)^2
    return gcv
end

"""
    _update_penalty_lambda(config::PenaltyConfig, lambda::Vector{Float64}) -> PenaltyConfig

Create a new PenaltyConfig with updated lambda values.
"""
function _update_penalty_lambda(config::PenaltyConfig, lambda::Vector{Float64})
    lambda_idx = 1
    
    new_terms = map(config.terms) do term
        new_term = PenaltyTerm(
            term.hazard_indices,
            term.S,
            lambda[lambda_idx],
            term.order,
            term.hazard_names
        )
        lambda_idx += 1
        new_term
    end
    
    new_total_terms = map(config.total_hazard_terms) do term
        new_term = TotalHazardPenaltyTerm(
            term.origin,
            term.hazard_indices,
            term.S,
            lambda[lambda_idx],
            term.order
        )
        lambda_idx += 1
        new_term
    end
    
    return PenaltyConfig(
        collect(new_terms),
        collect(new_total_terms),
        config.shared_lambda_groups,
        config.n_lambda
    )
end
