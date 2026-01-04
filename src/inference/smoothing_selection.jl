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
    SmoothingSelectionState

Internal state for smoothing parameter selection via PIJCV/GCV, storing cached matrices and intermediate results.

Note: This is separate from `PIJCVState` in variance.jl which is the lower-level
state for computing PIJCV criterion from matrices.

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
mutable struct SmoothingSelectionState
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
    compute_pijcv_criterion(log_lambda::Vector{Float64}, state::SmoothingSelectionState) -> Float64

Compute the PIJCV criterion V(λ) for given log-smoothing parameters.

The criterion is the sum of leave-one-out prediction errors:
    V(λ) = Σᵢ D(yᵢ, θᵢ⁽⁻ⁱ⁾)

where D is the unpenalized negative log-likelihood contribution and θᵢ⁽⁻ⁱ⁾
is the prediction using parameters estimated with subject i omitted.

# Arguments
- `log_lambda`: Vector of log-smoothing parameters (one per penalty term)
- `state`: SmoothingSelectionState with cached gradients/Hessians

# Returns
- `Float64`: PIJCV criterion value (lower is better)
"""
function compute_pijcv_criterion(log_lambda::Vector{Float64}, state::SmoothingSelectionState)
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

Handles baseline hazard terms, total hazard terms, and smooth covariate terms.
"""
function _add_penalty_to_hessian!(H::Matrix{Float64}, lambda::Vector{Float64}, 
                                   config::PenaltyConfig)
    lambda_idx = 1
    
    # Baseline hazard penalty terms
    for term in config.terms
        idx = term.hazard_indices
        H[idx, idx] .+= lambda[lambda_idx] .* term.S
        lambda_idx += 1
    end
    
    # Total hazard terms (for competing risks)
    for term in config.total_hazard_terms
        # Total hazard penalty uses Kronecker structure
        # For now, use the simplified addition per hazard
        for idx_range in term.hazard_indices
            H[idx_range, idx_range] .+= lambda[lambda_idx] .* term.S
        end
        lambda_idx += 1
    end
    
    # Smooth covariate penalty terms
    # Handle shared lambda groups
    if !isempty(config.shared_smooth_groups)
        # When lambda is shared, use the same lambda for all terms in group
        # First assign lambdas to groups
        term_to_lambda = Dict{Int, Int}()
        
        for (group_idx, group) in enumerate(config.shared_smooth_groups)
            for term_idx in group
                term_to_lambda[term_idx] = lambda_idx
            end
            lambda_idx += 1
        end
        
        # Handle terms not in any group (if n_lambda accounts for them separately)
        for (term_idx, term) in enumerate(config.smooth_covariate_terms)
            if !haskey(term_to_lambda, term_idx)
                term_to_lambda[term_idx] = lambda_idx
                lambda_idx += 1
            end
        end
        
        # Apply penalties
        for (term_idx, term) in enumerate(config.smooth_covariate_terms)
            lam = lambda[term_to_lambda[term_idx]]
            indices = term.param_indices
            for (i, pi) in enumerate(indices)
                for (j, pj) in enumerate(indices)
                    H[pi, pj] += lam * term.S[i, j]
                end
            end
        end
    else
        # No sharing - each term gets its own lambda
        for term in config.smooth_covariate_terms
            indices = term.param_indices
            for (i, pi) in enumerate(indices)
                for (j, pj) in enumerate(indices)
                    H[pi, pj] += lambda[lambda_idx] * term.S[i, j]
                end
            end
            lambda_idx += 1
        end
    end
end

"""
    _create_scoped_penalty_config(config::PenaltyConfig, scope::Symbol) -> Tuple{PenaltyConfig, Vector{Float64}, Vector{Float64}}

Create a penalty config containing only terms within the specified scope.

# Arguments
- `config`: Original penalty configuration
- `scope`: Which terms to include (:all, :baseline, :covariates)

# Returns
- Scoped PenaltyConfig
- Vector of fixed baseline lambdas (empty if baseline is in scope)
- Vector of fixed covariate lambdas (empty if covariates are in scope)
"""
function _create_scoped_penalty_config(config::PenaltyConfig, scope::Symbol)
    if scope == :all
        # No filtering needed
        return (config, Float64[], Float64[])
    end
    
    if scope == :baseline
        # Include only baseline terms, exclude covariates
        fixed_covariate_lambdas = [term.lambda for term in config.smooth_covariate_terms]
        
        # Create config with no covariate terms
        n_lambda_baseline = length(config.terms) + length(config.total_hazard_terms)
        scoped_config = PenaltyConfig(
            config.terms,
            config.total_hazard_terms,
            SmoothCovariatePenaltyTerm[],  # Empty covariate terms
            config.shared_lambda_groups,
            Vector{Int}[],  # Empty shared smooth groups
            n_lambda_baseline
        )
        return (scoped_config, Float64[], fixed_covariate_lambdas)
    end
    
    if scope == :covariates
        # Include only covariate terms, exclude baseline
        fixed_baseline_lambdas = vcat(
            [term.lambda for term in config.terms],
            [term.lambda_H for term in config.total_hazard_terms]
        )
        
        # Calculate n_lambda for covariate terms
        n_lambda_cov = if !isempty(config.shared_smooth_groups)
            length(config.shared_smooth_groups)
        else
            length(config.smooth_covariate_terms)
        end
        
        # Create config with no baseline terms
        scoped_config = PenaltyConfig(
            PenaltyTerm[],  # Empty baseline terms
            TotalHazardPenaltyTerm[],  # Empty total hazard terms
            config.smooth_covariate_terms,
            Dict{Int,Vector{Int}}(),  # Empty shared baseline groups
            config.shared_smooth_groups,
            n_lambda_cov
        )
        return (scoped_config, fixed_baseline_lambdas, Float64[])
    end
    
    error("Unknown scope: $scope")
end

"""
    _merge_scoped_lambdas(original_config, scoped_config, optimized_lambda, 
                          fixed_baseline, fixed_covariate, scope) -> PenaltyConfig

Merge optimized lambdas back into the original config structure.
"""
function _merge_scoped_lambdas(original_config::PenaltyConfig, 
                                scoped_config::PenaltyConfig,
                                optimized_lambda::Vector{Float64},
                                fixed_baseline_lambdas::Vector{Float64},
                                fixed_covariate_lambdas::Vector{Float64},
                                scope::Symbol)
    if scope == :all
        # All terms were optimized, just update the config
        return _update_penalty_lambda(original_config, optimized_lambda)
    end
    
    if scope == :baseline
        # Baseline was optimized, covariates are fixed
        # First update the scoped config with optimized lambdas
        updated_baseline = _update_penalty_lambda(scoped_config, optimized_lambda)
        
        # Reconstruct with original covariate terms (keep their original lambdas)
        return PenaltyConfig(
            updated_baseline.terms,
            updated_baseline.total_hazard_terms,
            original_config.smooth_covariate_terms,
            original_config.shared_lambda_groups,
            original_config.shared_smooth_groups,
            original_config.n_lambda
        )
    end
    
    if scope == :covariates
        # Covariates were optimized, baseline is fixed
        # First update the scoped config with optimized lambdas
        updated_covariate = _update_penalty_lambda(scoped_config, optimized_lambda)
        
        # Reconstruct with original baseline terms (keep their original lambdas)
        return PenaltyConfig(
            original_config.terms,
            original_config.total_hazard_terms,
            updated_covariate.smooth_covariate_terms,
            original_config.shared_lambda_groups,
            original_config.shared_smooth_groups,
            original_config.n_lambda
        )
    end
    
    error("Unknown scope: $scope")
end

"""
    select_smoothing_parameters(model::MultistateProcess, data::ExactData,
                                 penalty_config::PenaltyConfig,
                                 beta_init::Vector{Float64};
                                 method::Symbol=:pijcv,
                                 scope::Symbol=:all,
                                 maxiter::Int=100,
                                 verbose::Bool=false) -> NamedTuple

Select smoothing parameters using PIJCV or GCV.

# Arguments
- `model`: MultistateProcess model
- `data`: ExactData container
- `penalty_config`: Penalty configuration with initial λ values
- `beta_init`: Initial coefficient estimate (warm start)
- `method`: Selection method (:pijcv or :gcv)
- `scope`: Which splines to calibrate:
  - `:all` (default): Calibrate all spline penalties (baseline + covariates)
  - `:baseline`: Calibrate only baseline hazard splines
  - `:covariates`: Calibrate only smooth covariate splines
- `maxiter`: Maximum outer iterations for λ optimization
- `verbose`: Print progress

# Returns
NamedTuple with:
- `lambda`: Optimal smoothing parameters (for calibrated terms only)
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
                                      scope::Symbol=:all,
                                      maxiter::Int=100,
                                      verbose::Bool=false)
    # Validate scope
    scope ∈ (:all, :baseline, :covariates) || 
        throw(ArgumentError("scope must be :all, :baseline, or :covariates, got :$scope"))
    
    # Create scoped penalty config based on scope parameter
    scoped_config, fixed_baseline_lambdas, fixed_covariate_lambdas = _create_scoped_penalty_config(penalty_config, scope)
    
    n_lambda = scoped_config.n_lambda
    n_lambda > 0 || return (lambda=Float64[], beta=beta_init, criterion=NaN, 
                            converged=true, method_used=:none, penalty_config=penalty_config)
    
    # Initialize log-lambda from scoped penalty config
    log_lambda_init = zeros(n_lambda)
    lambda_idx = 1
    
    # Baseline hazard terms (only if in scope)
    for term in scoped_config.terms
        log_lambda_init[lambda_idx] = log(term.lambda)
        lambda_idx += 1
    end
    
    # Total hazard terms (only if in scope)
    for term in scoped_config.total_hazard_terms
        log_lambda_init[lambda_idx] = log(term.lambda_H)
        lambda_idx += 1
    end
    
    # Smooth covariate terms (handle shared groups)
    if !isempty(scoped_config.shared_smooth_groups)
        # One lambda per group
        for group in scoped_config.shared_smooth_groups
            # Use first term's lambda as representative
            term = scoped_config.smooth_covariate_terms[group[1]]
            log_lambda_init[lambda_idx] = log(term.lambda)
            lambda_idx += 1
        end
        # Any terms not in groups
        grouped_indices = Set(vcat(scoped_config.shared_smooth_groups...))
        for (idx, term) in enumerate(scoped_config.smooth_covariate_terms)
            if idx ∉ grouped_indices
                log_lambda_init[lambda_idx] = log(term.lambda)
                lambda_idx += 1
            end
        end
    else
        for term in scoped_config.smooth_covariate_terms
            log_lambda_init[lambda_idx] = log(term.lambda)
            lambda_idx += 1
        end
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
    
    # Create smoothing selection state with scoped config
    state = SmoothingSelectionState(
        copy(beta_init),
        H_unpenalized,
        subject_grads,
        subject_hessians,
        scoped_config,
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
    # Merge optimized lambdas back with fixed lambdas from excluded terms
    updated_config = _merge_scoped_lambdas(penalty_config, scoped_config, final_lambda, 
                                           fixed_baseline_lambdas, fixed_covariate_lambdas, scope)
    
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
    compute_gcv_criterion(log_lambda::Vector{Float64}, state::SmoothingSelectionState) -> Float64

Compute Generalized Cross-Validation criterion.

GCV approximates leave-one-out CV using the effective degrees of freedom:
    V_GCV(λ) = n * deviance / (n - edf)²

where edf = tr(A) is the trace of the influence matrix.
"""
function compute_gcv_criterion(log_lambda::Vector{Float64}, state::SmoothingSelectionState)
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
    
    # Update baseline hazard terms
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
    
    # Update total hazard terms
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
    
    # Update smooth covariate terms
    # Handle shared lambda groups
    if !isempty(config.shared_smooth_groups)
        # Build term -> lambda mapping
        term_to_lambda = Dict{Int, Float64}()
        
        for (group_idx, group) in enumerate(config.shared_smooth_groups)
            lam = lambda[lambda_idx]
            for term_idx in group
                term_to_lambda[term_idx] = lam
            end
            lambda_idx += 1
        end
        
        # Handle terms not in any group
        for term_idx in 1:length(config.smooth_covariate_terms)
            if !haskey(term_to_lambda, term_idx)
                term_to_lambda[term_idx] = lambda[lambda_idx]
                lambda_idx += 1
            end
        end
        
        # Update terms
        new_smooth_terms = map(enumerate(config.smooth_covariate_terms)) do (idx, term)
            SmoothCovariatePenaltyTerm(
                term.param_indices,
                term.S,
                term_to_lambda[idx],
                term.order,
                term.label,
                term.hazard_name
            )
        end
    else
        # No sharing - update each term independently
        new_smooth_terms = map(config.smooth_covariate_terms) do term
            new_term = SmoothCovariatePenaltyTerm(
                term.param_indices,
                term.S,
                lambda[lambda_idx],
                term.order,
                term.label,
                term.hazard_name
            )
            lambda_idx += 1
            new_term
        end
    end
    
    return PenaltyConfig(
        collect(new_terms),
        collect(new_total_terms),
        collect(new_smooth_terms),
        config.shared_lambda_groups,
        config.shared_smooth_groups,
        config.n_lambda
    )
end
