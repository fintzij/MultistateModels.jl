# =============================================================================
# Smoothing Parameter Selection for Penalized Splines
# =============================================================================
#
# Implements PIJCV (Predictive Infinitesimal Jackknife Cross-Validation) for 
# automatic selection of smoothing parameters λ in penalized spline models.
#
# Based on Wood (2024): "Neighbourhood Cross-Validation" arXiv:2404.16490
#
# Algorithm: Profile likelihood optimization of V(λ, β(λ)) where β(λ) is the
# penalized MLE at each λ. Uses golden section search for 1D optimization 
# (single λ) or Nelder-Mead for multi-dimensional λ optimization.
#
# Key insight: Inner optimization uses PENALIZED loss, outer optimization
# minimizes UNPENALIZED prediction error via leave-one-out approximation.
#
# =============================================================================

using LinearAlgebra

"""
    SmoothingSelectionState

Internal state for smoothing parameter selection via PIJCV/CV, storing cached matrices and intermediate results.

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

# =============================================================================
# Helper Functions for Performance Iteration
# =============================================================================

"""
    compute_penalty_from_lambda(beta::AbstractVector{T}, lambda::AbstractVector, 
                                 config::PenaltyConfig) where T

Compute penalty term Σⱼ λⱼ βⱼᵀ Sⱼ βⱼ / 2 using explicit lambda values.

Unlike `compute_penalty` which uses lambdas stored in the config, this function
takes explicit lambda values as an argument. Used during optimization when
lambda is being varied.

# Arguments
- `beta`: Coefficient vector (natural scale)
- `lambda`: Vector of smoothing parameters
- `config`: Penalty configuration containing S matrices and index mappings

# Returns
Scalar penalty value (half the quadratic form)

# Notes
- Parameters are on natural scale with box constraints (β ≥ 0)
- Penalty is quadratic: P(β) = (λ/2) βᵀSβ
- This must match the behavior of `compute_penalty` in infrastructure.jl
"""
function compute_penalty_from_lambda(beta::AbstractVector{T}, lambda::AbstractVector, 
                                      config::PenaltyConfig) where T
    penalty = zero(T)
    lambda_idx = 1
    
    # Baseline hazard penalties - parameters on natural scale
    for term in config.terms
        β_j = @view beta[term.hazard_indices]
        penalty += lambda[lambda_idx] * dot(β_j, term.S * β_j)
        lambda_idx += 1
    end
    
    # Total hazard penalties - sum natural-scale coefficients
    for term in config.total_hazard_terms
        K = size(term.S, 1)
        β_total = zeros(T, K)
        for idx_range in term.hazard_indices
            β_k = @view beta[idx_range]
            β_total .+= β_k  # Parameters already on natural scale
        end
        penalty += lambda[lambda_idx] * dot(β_total, term.S * β_total)
        lambda_idx += 1
    end
    
    # Smooth covariate penalties - no transformation (linear predictor scale)
    if !isempty(config.shared_smooth_groups)
        # Build term -> lambda mapping
        term_to_lambda = Dict{Int, Int}()
        for (group_idx, group) in enumerate(config.shared_smooth_groups)
            for term_idx in group
                term_to_lambda[term_idx] = lambda_idx
            end
            lambda_idx += 1
        end
        # Handle ungrouped terms
        for term_idx in 1:length(config.smooth_covariate_terms)
            if !haskey(term_to_lambda, term_idx)
                term_to_lambda[term_idx] = lambda_idx
                lambda_idx += 1
            end
        end
        # Apply penalties
        for (term_idx, term) in enumerate(config.smooth_covariate_terms)
            β_k = beta[term.param_indices]
            penalty += lambda[term_to_lambda[term_idx]] * dot(β_k, term.S * β_k)
        end
    else
        # No sharing - each term gets its own lambda
        for term in config.smooth_covariate_terms
            β_k = beta[term.param_indices]
            penalty += lambda[lambda_idx] * dot(β_k, term.S * β_k)
            lambda_idx += 1
        end
    end
    
    return penalty / 2
end

"""
    fit_penalized_beta(model::MultistateProcess, data::ExactData, 
                       lambda::Vector{Float64}, penalty_config::PenaltyConfig,
                       beta_init::Vector{Float64};
                       maxiters::Int=100, use_polyalgorithm::Bool=false,
                       verbose::Bool=false) -> Vector{Float64}

Fit coefficients β given fixed smoothing parameters λ.

Minimizes the penalized negative log-likelihood:
    f(β) = -ℓ(β) + (1/2) Σⱼ λⱼ βⱼᵀ Sⱼ βⱼ

# Arguments
- `model`: MultistateProcess model
- `data`: ExactData container
- `lambda`: Vector of smoothing parameters (natural scale)
- `penalty_config`: Penalty configuration with S matrices
- `beta_init`: Warm start for optimization
- `maxiters`: Maximum optimizer iterations
- `use_polyalgorithm`: If true, use LBFGS→Ipopt; if false, pure Ipopt
- `verbose`: Print optimization progress

# Returns
Fitted coefficient vector β
"""
function fit_penalized_beta(model::MultistateProcess, data::ExactData,
                            lambda::Vector{Float64}, penalty_config::PenaltyConfig,
                            beta_init::Vector{Float64};
                            maxiters::Int=100, use_polyalgorithm::Bool=false,
                            verbose::Bool=false, ipopt_options...)
    
    # Define penalized negative log-likelihood objective
    function penalized_nll(β, p)
        # Unpenalized negative log-likelihood
        nll = loglik_exact(β, data; neg=true)
        # Add penalty
        pen = compute_penalty_from_lambda(β, lambda, penalty_config)
        return nll + pen
    end
    
    # Set up optimization with automatic differentiation
    adtype = DifferentiationInterface.SecondOrder(
        Optimization.AutoForwardDiff(), 
        Optimization.AutoForwardDiff()
    )
    optf = OptimizationFunction(penalized_nll, adtype)
    prob = OptimizationProblem(optf, beta_init, nothing)
    
    # Merge default options with user overrides
    # Convert kwargs to NamedTuple to enable merging
    ipopt_options_nt = (;ipopt_options...)
    merged_options = merge(DEFAULT_IPOPT_OPTIONS, (maxiters=maxiters, tol=1e-6), ipopt_options_nt)
    
    if use_polyalgorithm
        # Phase 1: LBFGS warm-start with loose tolerance
        if verbose
            println("    β fit: LBFGS warm-start...")
        end
        sol_warmstart = solve(prob, LBFGS();
                              maxiters=min(maxiters, 50),
                              abstol=1e-2,
                              reltol=1e-2,
                              show_trace=false)
        
        # Phase 2: Ipopt refinement
        if verbose
            println("    β fit: Ipopt refinement...")
        end
        prob_refined = remake(prob, u0=sol_warmstart.u)
        sol = solve(prob_refined, IpoptOptimizer(); merged_options...)
    else
        # Pure Ipopt from warm start
        if verbose
            println("    β fit: Ipopt from warm start...")
        end
        sol = solve(prob, IpoptOptimizer(); merged_options...)
    end
    
    return sol.u
end

"""
    extract_lambda_vector(config::PenaltyConfig) -> Vector{Float64}

Extract the current λ values from a penalty configuration as a flat vector.
"""
function extract_lambda_vector(config::PenaltyConfig)
    lambdas = Float64[]
    
    # Baseline hazard terms
    for term in config.terms
        push!(lambdas, term.lambda)
    end
    
    # Total hazard terms
    for term in config.total_hazard_terms
        push!(lambdas, term.lambda_H)
    end
    
    # Smooth covariate terms (handle shared groups)
    if !isempty(config.shared_smooth_groups)
        for group in config.shared_smooth_groups
            # Use first term's lambda as representative
            term = config.smooth_covariate_terms[group[1]]
            push!(lambdas, term.lambda)
        end
        # Handle ungrouped terms
        grouped_indices = Set(vcat(config.shared_smooth_groups...))
        for (idx, term) in enumerate(config.smooth_covariate_terms)
            if idx ∉ grouped_indices
                push!(lambdas, term.lambda)
            end
        end
    else
        for term in config.smooth_covariate_terms
            push!(lambdas, term.lambda)
        end
    end
    
    return lambdas
end

# =============================================================================
# PIJCV and Cross-Validation Criterion Functions
# =============================================================================

"""
    compute_pijcv_criterion(log_lambda::Vector{Float64}, state::SmoothingSelectionState) -> Float64

Compute the PIJCV/NCV criterion V(λ) for given log-smoothing parameters.

Implements the Neighbourhood Cross-Validation criterion from Wood (2024) 
"On Neighbourhood Cross Validation" arXiv:2404.16490v4.

# NCV Criterion (Wood 2024, Equation 2)

The criterion is the sum of leave-one-out prediction errors:

    V(λ) = Σᵢ Dᵢ(β̂⁻ⁱ)

where:
- Dᵢ(β) = -ℓᵢ(β) is subject i's negative log-likelihood contribution
- β̂⁻ⁱ is the penalized MLE with subject i omitted

# Efficient Computation

Rather than refit n times, we use Newton's approximation (Wood 2024, Equation 3):

    β̂⁻ⁱ ≈ β̂ + Δ⁻ⁱ,  where  Δ⁻ⁱ = H_{λ,-i}⁻¹ gᵢ

where H_{λ,-i} = H_λ - Hᵢ is the leave-one-out penalized Hessian.

# Key Implementation Detail

**This function evaluates the ACTUAL likelihood** Dᵢ(β̂⁻ⁱ) at the LOO parameters,
NOT a Taylor approximation. The quadratic approximation V_q is used as a fallback
only when the actual likelihood is non-finite (Wood 2024, Section 4.1).

# Arguments
- `log_lambda`: Vector of log-smoothing parameters (one per penalty term)
- `state`: SmoothingSelectionState with cached gradients/Hessians

# Returns
- `Float64`: PIJCV criterion value (lower is better)

# References
- Wood, S.N. (2024). "On Neighbourhood Cross Validation." arXiv:2404.16490v4
"""
function compute_pijcv_criterion(log_lambda::AbstractVector{T}, state::SmoothingSelectionState) where T<:Real
    lambda = exp.(log_lambda)
    
    # Build penalized Hessian (penalty is quadratic: λ βᵀSβ)
    # Create matrix with appropriate eltype for AD
    H_lambda = _build_penalized_hessian(state.H_unpenalized, lambda, state.penalty_config; 
                                         beta=state.beta_hat)
    
    # Check if we're in AD mode (Dual numbers)
    # In AD mode, skip Cholesky downdate optimization and use direct solves
    use_cholesky_downdate = (T === Float64)
    
    # Try Cholesky factorization of full penalized Hessian
    H_lambda_sym = Symmetric(H_lambda)
    chol_fact = if use_cholesky_downdate
        try
            cholesky(H_lambda_sym)
        catch e
            nothing
        end
    else
        nothing  # Skip factorization for AD mode
    end
    
    # If Cholesky failed and not in AD mode, return large value
    if chol_fact === nothing && use_cholesky_downdate
        return T(1e10)
    end
    
    # Get subject-level likelihoods at current estimate (for V_q fallback)
    ll_subj_base = loglik_exact(state.beta_hat, state.data; neg=false, return_ll_subj=true)
    
    # Compute NCV criterion: V = Σᵢ Dᵢ(β̂⁻ⁱ)
    # Following Wood (2024): evaluate ACTUAL likelihood at LOO parameters
    V = zero(T)
    n_fallback = 0  # Track how many times we fall back to V_q
    
    for i in 1:state.n_subjects
        # Get subject i's gradient and Hessian (of negative log-likelihood)
        # Convention: gᵢ = ∇Dᵢ = -∇ℓᵢ, Hᵢ = ∇²Dᵢ = -∇²ℓᵢ
        g_i = @view state.subject_grads[:, i]
        H_i = state.subject_hessians[i]
        
        # Solve for coefficient perturbation: Δ⁻ⁱ = H_{λ,-i}⁻¹ gᵢ
        delta_i = if use_cholesky_downdate && chol_fact !== nothing
            # Use Cholesky downdate for efficiency (Wood 2024, Section 2.1)
            _solve_loo_newton_step(chol_fact, H_i, g_i)
        else
            nothing  # Will use direct solve below
        end
        
        if delta_i === nothing
            # Either Cholesky downdate failed or we're in AD mode
            # Use direct solve: (H_λ - H_i)⁻¹ g_i
            H_lambda_loo = H_lambda - H_i
            H_loo_sym = Symmetric(H_lambda_loo)
            
            delta_i = try
                H_loo_sym \ collect(g_i)
            catch e
                # If solve fails (e.g., indefinite Hessian), return large value
                return T(1e10)
            end
        end
        
        # Compute LOO parameters: β̂⁻ⁱ = β̂ + Δ⁻ⁱ
        beta_loo = state.beta_hat .+ delta_i
        
        # Project to feasible region: β ≥ 0 for non-negative constrained params
        # This handles cases where Newton step overshoots into infeasible region
        beta_loo = max.(beta_loo, zero(T))
        
        # CORE OF NCV: Evaluate ACTUAL likelihood at LOO parameters
        ll_loo = loglik_subject(beta_loo, state.data, i)
        
        # Check if likelihood is finite
        if isfinite(ll_loo)
            # Normal case: use actual likelihood
            D_i = -ll_loo
        else
            # Fallback to quadratic approximation V_q (Wood 2024, Section 4.1)
            linear_term = dot(g_i, delta_i)
            quadratic_term = T(0.5) * dot(delta_i, H_i * delta_i)
            D_i = -ll_subj_base[i] + linear_term + quadratic_term
            n_fallback += 1
        end
        
        V += D_i
    end
    
    return V
end

"""
    compute_pijkfold_criterion(log_lambda::Vector{Float64}, state::SmoothingSelectionState, 
                               nfolds::Int) -> Float64

Compute the k-fold PIJCV criterion V(λ) using Newton-approximated fold-out estimates.

This is a generalization of the LOO-based PIJCV (Wood 2024) to k-fold cross-validation.
Instead of approximating each subject's leave-one-out estimate via a Newton step,
we approximate each fold's leave-fold-out estimate.

# k-fold NCV Criterion

The criterion is the sum of fold-out prediction errors:

    V(λ) = Σₖ Σᵢ∈foldₖ Dᵢ(β̂⁻ᵏ)

where:
- Dᵢ(β) = -ℓᵢ(β) is subject i's negative log-likelihood contribution
- β̂⁻ᵏ is the penalized MLE with fold k omitted

# Efficient Computation via Newton Approximation

Rather than refit k times, we use Newton's approximation (generalizing Wood 2024):

    β̂⁻ᵏ ≈ β̂ + Δ⁻ᵏ,  where  Δ⁻ᵏ = H_{λ,-k}⁻¹ gₖ

where:
- gₖ = Σᵢ∈foldₖ gᵢ is the sum of gradients for subjects in fold k
- H_{λ,-k} = H_λ - Hₖ is the leave-fold-out penalized Hessian
- Hₖ = Σᵢ∈foldₖ Hᵢ is the sum of Hessians for subjects in fold k

# Complexity
- Exact k-fold: O(k × fitting_cost) — requires k refits
- PIJKFOLD: O(k × p³) — only k linear solves, no refitting

# Arguments
- `log_lambda`: Vector of log-smoothing parameters (one per penalty term)
- `state`: SmoothingSelectionState with cached gradients/Hessians
- `nfolds`: Number of folds (e.g., 5, 10, 20)

# Returns
- `Float64`: k-fold PIJCV criterion value (lower is better)

# Notes
- For nfolds = n_subjects, this is equivalent to `compute_pijcv_criterion` (LOO)
- Uses deterministic fold assignment: subject i goes to fold (i-1) % nfolds + 1
- Falls back to quadratic approximation V_q when actual likelihood is non-finite

# References
- Wood, S.N. (2024). "On Neighbourhood Cross Validation." arXiv:2404.16490v4
  (This extends Wood's LOO approximation to k-fold)
"""
function compute_pijkfold_criterion(log_lambda::AbstractVector{T}, state::SmoothingSelectionState, 
                                    nfolds::Int) where T<:Real
    lambda = exp.(log_lambda)
    n_subjects = state.n_subjects
    
    # Validate nfolds
    nfolds >= 2 || throw(ArgumentError("nfolds must be at least 2, got $nfolds"))
    nfolds <= n_subjects || throw(ArgumentError("nfolds ($nfolds) cannot exceed n_subjects ($n_subjects)"))
    
    # Build penalized Hessian (penalty is quadratic: λ βᵀSβ)
    H_lambda = _build_penalized_hessian(state.H_unpenalized, lambda, state.penalty_config;
                                         beta=state.beta_hat)
    
    # Try Cholesky factorization of full penalized Hessian
    H_lambda_sym = Symmetric(H_lambda)
    chol_fact = try
        cholesky(H_lambda_sym)
    catch e
        # If Cholesky fails, H_λ is not positive definite
        return T(1e10)
    end
    
    # Get subject-level likelihoods at current estimate (for V_q fallback)
    ll_subj_base = loglik_exact(state.beta_hat, state.data; neg=false, return_ll_subj=true)
    
    # Create fold assignments (deterministic, in order)
    # Subject i goes to fold (i-1) % nfolds + 1
    fold_assignments = [(i - 1) % nfolds + 1 for i in 1:n_subjects]
    
    # Compute k-fold NCV criterion: V = Σₖ Σᵢ∈foldₖ Dᵢ(β̂⁻ᵏ)
    V = zero(T)
    n_fallback = 0
    
    for k in 1:nfolds
        # Get indices for this fold
        fold_indices = findall(==(k), fold_assignments)
        
        # Compute fold-level gradient and Hessian
        # gₖ = Σᵢ∈foldₖ gᵢ,  Hₖ = Σᵢ∈foldₖ Hᵢ
        p = state.n_params
        g_k = zeros(T, p)
        H_k = zeros(T, p, p)
        
        for i in fold_indices
            g_k .+= @view state.subject_grads[:, i]
            H_k .+= state.subject_hessians[i]
        end
        
        # Solve for fold-out coefficient perturbation: Δ⁻ᵏ = H_{λ,-k}⁻¹ gₖ
        # H_{λ,-k} = H_λ - Hₖ
        H_lambda_fold = H_lambda - H_k
        H_fold_sym = Symmetric(H_lambda_fold)
        
        delta_k = try
            H_fold_sym \ g_k
        catch e
            # If solve fails (e.g., indefinite Hessian), return large value
            return T(1e10)
        end
        
        # Compute fold-out parameters: β̂⁻ᵏ = β̂ + Δ⁻ᵏ
        beta_fold_out = state.beta_hat .+ delta_k
        
        # Evaluate ACTUAL likelihood for each subject in this fold at fold-out parameters
        for i in fold_indices
            ll_fold = loglik_subject(beta_fold_out, state.data, i)
            
            if isfinite(ll_fold)
                # Normal case: use actual likelihood
                D_i = -ll_fold
            else
                # Fallback to quadratic approximation V_q
                g_i = @view state.subject_grads[:, i]
                H_i = state.subject_hessians[i]
                linear_term = dot(g_i, delta_k)
                quadratic_term = T(0.5) * dot(delta_k, H_i * delta_k)
                D_i = -ll_subj_base[i] + linear_term + quadratic_term
                n_fallback += 1
            end
            
            V += D_i
        end
    end
    
    return V
end

"""
    compute_loocv_criterion(lambda::Vector{Float64}, beta_init::Vector{Float64},
                            model::MultistateProcess, data::ExactData,
                            penalty_config::PenaltyConfig;
                            maxiters::Int=50, verbose::Bool=false) -> Float64

Compute exact Leave-One-Out Cross-Validation criterion by refitting n times.

This is the gold-standard LOOCV criterion:
```math
V_{LOOCV}(\\lambda) = \\sum_{i=1}^{n} D_i(\\hat\\beta^{-i})
```

where ``\\hat\\beta^{-i}`` is the penalized MLE with subject ``i`` excluded from the data.
Unlike PIJCV, which approximates ``\\hat\\beta^{-i}`` via a Newton step, exact LOOCV
refits the model n times, each time leaving out one observation.

# Computational Cost
- O(n × fitting_cost) — expensive but exact
- For n=100 subjects and 50 iterations per fit: ~5000 optimization iterations total

# Implementation
Uses subject weights to exclude observations: sets weight=0 for subject i,
refits the model, then evaluates subject i's likelihood at the LOO estimate.

# Arguments
- `lambda`: Smoothing parameter vector (natural scale, not log)
- `beta_init`: Initial coefficient estimate (warm start for each LOO fit)
- `model`: MultistateProcess model (subject weights will be temporarily modified)
- `data`: ExactData container
- `penalty_config`: Penalty configuration
- `maxiters`: Maximum iterations for each LOO fit (default 50)
- `verbose`: Print progress for each subject (default false)

# Returns
- `Float64`: LOOCV criterion value (sum of LOO deviances, lower is better)

# Notes
- This function temporarily modifies `model.SubjectWeights` (restored after each subject)
- Use `:loocv` in `select_smoothing_parameters()` to select λ via exact LOOCV
- For approximate LOOCV that's O(n) instead of O(n²), use `:pijcv`

# References
- Stone, M. (1974). "Cross-Validatory Choice and Assessment of Statistical Predictions."
  Journal of the Royal Statistical Society B 36(2):111-147.
- Wood, S.N. (2024). "On Neighbourhood Cross Validation." arXiv:2404.16490v4
  (PIJCV approximates this criterion via Newton steps)

# Example
```julia
# Compute LOOCV at a specific λ
loocv_value = compute_loocv_criterion(
    [10.0],           # lambda
    beta_mle,         # starting coefficients
    model, data, penalty_config
)

# Compare with PIJCV at the same λ
pijcv_value = compute_pijcv_criterion(log.([10.0]), state)
```

See also: [`compute_pijcv_criterion`](@ref), [`select_smoothing_parameters`](@ref)
"""
function compute_loocv_criterion(lambda::Vector{Float64}, beta_init::Vector{Float64},
                                  model::MultistateProcess, data::ExactData,
                                  penalty_config::PenaltyConfig;
                                  maxiters::Int=50, verbose::Bool=false)
    n_subjects = length(data.paths)
    
    # Store original weights
    original_weights = copy(model.SubjectWeights)
    
    total_criterion = 0.0
    
    for i in 1:n_subjects
        # Exclude subject i by setting weight to 0
        model.SubjectWeights[i] = 0.0
        
        # Fit model on data excluding subject i
        # Warm-start from the provided initial estimate
        beta_loo = fit_penalized_beta(model, data, lambda, penalty_config, beta_init;
                                      maxiters=maxiters, verbose=false)
        
        # Restore original weight for this subject
        model.SubjectWeights[i] = original_weights[i]
        
        # Compute deviance contribution for subject i at LOO estimate
        # D_i = -ℓ_i(β̂⁻ⁱ) (loss convention: positive deviance)
        ll_i = loglik_subject(beta_loo, data, i)
        D_i = -ll_i
        
        total_criterion += D_i
        
        if verbose
            println("  Subject $i/$n_subjects: D_i = $(round(D_i, digits=4))")
        end
    end
    
    # Restore all original weights (safety)
    model.SubjectWeights .= original_weights
    
    return total_criterion
end

"""
    compute_kfold_cv_criterion(lambda::Vector{Float64}, beta_init::Vector{Float64},
                               model::MultistateProcess, data::ExactData,
                               penalty_config::PenaltyConfig, nfolds::Int;
                               maxiters::Int=50, verbose::Bool=false) -> Float64

Compute k-fold cross-validation criterion by refitting k times.

This is a standard k-fold CV criterion:
```math
V_{k-fold}(\\lambda) = \\sum_{k=1}^{K} \\sum_{i \\in \\text{fold}_k} D_i(\\hat\\beta^{-\\text{fold}_k})
```

where ``\\hat\\beta^{-\\text{fold}_k}`` is the penalized MLE with fold k excluded from the data.
This is a computationally cheaper alternative to exact LOOCV.

# Computational Cost
- O(k × fitting_cost) — k times cheaper than LOOCV for same effective CV
- For k=5 with 100 subjects: 5 refits (vs 100 for LOOCV)

# Implementation
Uses subject weights to exclude observations: sets weight=0 for all subjects in fold k,
refits the model, then evaluates those subjects' likelihood at the fold-out estimate.
Subjects are assigned to folds in order (subject 1..n/k → fold 1, etc.).

# Arguments
- `lambda`: Smoothing parameter vector (natural scale, not log)
- `beta_init`: Initial coefficient estimate (warm start for each fold fit)
- `model`: MultistateProcess model (subject weights will be temporarily modified)
- `data`: ExactData container
- `penalty_config`: Penalty configuration
- `nfolds`: Number of folds (e.g., 5, 10, 20)
- `maxiters`: Maximum iterations for each fold fit (default 50)
- `verbose`: Print progress for each fold (default false)

# Returns
- `Float64`: k-fold CV criterion value (sum of fold-out deviances, lower is better)

# Notes
- This function temporarily modifies `model.SubjectWeights` (restored after each fold)
- Folds are created deterministically: subjects are assigned to folds by index order
- For reproducibility with different orderings, shuffle your data before model creation

# References
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). "The Elements of Statistical Learning."
  2nd edition, Springer. Chapter 7.

# Example
```julia
# Compute 5-fold CV at a specific λ
cv5_value = compute_kfold_cv_criterion(
    [10.0],           # lambda
    beta_mle,         # starting coefficients
    model, data, penalty_config,
    5                 # number of folds
)
```

See also: [`compute_loocv_criterion`](@ref), [`select_smoothing_parameters`](@ref)
"""
function compute_kfold_cv_criterion(lambda::Vector{Float64}, beta_init::Vector{Float64},
                                    model::MultistateProcess, data::ExactData,
                                    penalty_config::PenaltyConfig, nfolds::Int;
                                    maxiters::Int=50, verbose::Bool=false)
    n_subjects = length(data.paths)
    
    # Validate nfolds
    nfolds >= 2 || throw(ArgumentError("nfolds must be at least 2, got $nfolds"))
    nfolds <= n_subjects || throw(ArgumentError("nfolds ($nfolds) cannot exceed n_subjects ($n_subjects)"))
    
    # Store original weights
    original_weights = copy(model.SubjectWeights)
    
    # Create fold assignments (deterministic, in order)
    # Subject i goes to fold (i-1) % nfolds + 1
    fold_assignments = [(i - 1) % nfolds + 1 for i in 1:n_subjects]
    
    total_criterion = 0.0
    
    for k in 1:nfolds
        # Get indices for this fold
        fold_indices = findall(==(k), fold_assignments)
        
        # Exclude fold k by setting weights to 0
        for i in fold_indices
            model.SubjectWeights[i] = 0.0
        end
        
        # Fit model on data excluding fold k
        # Warm-start from the provided initial estimate
        beta_fold = fit_penalized_beta(model, data, lambda, penalty_config, beta_init;
                                       maxiters=maxiters, verbose=false)
        
        # Restore original weights for this fold
        for i in fold_indices
            model.SubjectWeights[i] = original_weights[i]
        end
        
        # Compute deviance contribution for all subjects in fold k
        fold_criterion = 0.0
        for i in fold_indices
            ll_i = loglik_subject(beta_fold, data, i)
            D_i = -ll_i
            fold_criterion += D_i
        end
        
        total_criterion += fold_criterion
        
        if verbose
            println("  Fold $k/$nfolds (n=$(length(fold_indices))): D = $(round(fold_criterion, digits=4))")
        end
    end
    
    # Restore all original weights (safety)
    model.SubjectWeights .= original_weights
    
    return total_criterion
end

"""
    _solve_loo_newton_step(chol_H::Cholesky, H_i::Matrix, g_i::AbstractVector) -> Union{Vector, Nothing}

Solve the leave-one-out Newton step using Cholesky rank-k downdate.

Computes δᵢ = (H - Hᵢ)⁻¹ gᵢ efficiently via Cholesky downdate.

# Algorithm (Wood 2024, Section 2.1)

Given the Cholesky factor L where H = LLᵀ, we want to solve (H - Hᵢ)x = g.

1. Eigendecompose Hᵢ = VDVᵀ where D = diag(d₁, ..., dₖ) with dₖ > 0
2. For each positive eigenvalue dⱼ, perform rank-1 downdate:
   LLᵀ - dⱼvⱼvⱼᵀ → L̃L̃ᵀ
3. Solve L̃L̃ᵀx = g

# Complexity
- Naive: O(p³) per subject (form matrix + factorize)  
- Downdate: O(kp²) per subject, where k = rank(Hᵢ) ≤ p

For multistate models with few transitions per subject, k is typically small.

# Returns
- `Vector`: Solution δᵢ if successful
- `nothing`: If downdate fails (H_{λ,-i} is indefinite)

# Reference
- Wood, S.N. (2024). On Neighbourhood Cross Validation, arXiv:2404.16490v4, Section 2.1
- Seeger, M. (2004). Low Rank Updates for the Cholesky Decomposition.
"""
function _solve_loo_newton_step(chol_H::Cholesky, H_i::Matrix{Float64}, g_i::AbstractVector)
    # Copy Cholesky factor as a regular Matrix (we'll modify it)
    L = Matrix(chol_H.L)
    n = size(L, 1)
    
    # Validate Hessian before eigendecomposition
    if !all(isfinite.(H_i))
        # Provide diagnostic information about the non-finite values
        nan_count = count(isnan, H_i)
        inf_count = count(isinf, H_i)
        nan_rows = unique(findall(isnan, H_i) .|> x -> x[1])
        @warn "Subject Hessian contains non-finite values ($(nan_count) NaN, $(inf_count) Inf). " *
              "This typically indicates:\n" *
              "  1. Zero/negative hazard values (check parameter bounds)\n" *
              "  2. Spline evaluation outside knot range\n" *
              "  3. Extreme parameter values during optimization\n" *
              "Affected parameter indices: $(nan_rows)" maxlog=3
        return nothing
    end
    
    # Eigendecompose subject Hessian to get rank-k representation
    # H_i = V * D * V' where D = diag(eigenvalues)
    eigen_H = eigen(Symmetric(H_i))
    
    # Perform rank-1 downdates for each positive eigenvalue
    # This computes L̃ such that L̃L̃ᵀ = LLᵀ - Hᵢ = H - Hᵢ
    tol = sqrt(eps(Float64))
    
    for (idx, d) in enumerate(eigen_H.values)
        if d > tol
            # Downdate: LLᵀ - d*v*vᵀ
            v = eigen_H.vectors[:, idx]
            success = _cholesky_downdate!(L, sqrt(d) * v)
            if !success
                # Matrix became indefinite
                return nothing
            end
        end
    end
    
    # Solve using downdated Cholesky: L̃L̃ᵀ x = g
    # Forward solve: L̃ y = g
    # Backward solve: L̃ᵀ x = y
    try
        L_chol = Cholesky(L, 'L', 0)
        return L_chol \ collect(g_i)
    catch e
        # Cholesky solve failed
        return nothing
    end
end

"""
    _cholesky_downdate!(L::Matrix, v::Vector; tol=1e-10) -> Bool

Perform in-place rank-1 downdate of Cholesky factor: LLᵀ → L̃L̃ᵀ where L̃L̃ᵀ = LLᵀ - vvᵀ.

Uses the standard sequential downdate algorithm with Givens rotations.

# Arguments
- `L`: Lower triangular Cholesky factor (modified in place)
- `v`: Vector for rank-1 downdate (will be modified)
- `tol`: Tolerance for indefiniteness detection

# Returns
- `true` if downdate succeeded
- `false` if matrix became indefinite (diagonal element would be imaginary)

# Algorithm
For j = 1, ..., n:
1. r² = L[j,j]² - v[j]²
2. If r² < tol, return false (indefinite)
3. r = √r², c = r/L[j,j], s = v[j]/L[j,j]
4. L[j,j] = r
5. For i > j: Update L[i,j] and v[i] using Givens rotation

# Reference
Seeger, M. (2004). Low Rank Updates for the Cholesky Decomposition.
"""
function _cholesky_downdate!(L::Matrix{Float64}, v::Vector{Float64}; tol::Float64=1e-10)
    n = size(L, 1)
    w = copy(v)
    
    for j in 1:n
        # Hyperbolic Givens rotation to eliminate w[j]
        # We want r² = L[j,j]² - w[j]² where r is the new diagonal
        a = L[j, j]
        b = w[j]
        
        r² = a^2 - b^2
        
        # Check for indefiniteness
        if r² < tol
            return false
        end
        
        r = sqrt(r²)
        
        # Hyperbolic rotation: [c s; s c] with c = a/r, s = -b/r, c² - s² = 1
        c = a / r
        s = -b / r
        
        L[j, j] = r
        
        # Apply rotation to remaining elements: [L[i,j]; w[i]] → [c*L + s*w; s*L + c*w]
        @inbounds for i in (j+1):n
            temp = c * L[i, j] + s * w[i]
            w[i] = s * L[i, j] + c * w[i]
            L[i, j] = temp
        end
    end
    
    return true
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
    _build_penalized_hessian(H_unpen::Matrix{Float64}, lambda::AbstractVector{T}, 
                              config::PenaltyConfig;
                              beta::Union{Nothing, AbstractVector{<:Real}}=nothing) where T<:Real

Build penalized Hessian with proper element type for AD compatibility.

Parameters are on natural scale with box constraints. The penalty is quadratic:
P(β) = (λ/2) βᵀSβ, so the penalty Hessian is simply λS.

Returns: H_unpen + penalty Hessian

This is a non-mutating version that creates a new matrix with the appropriate type.
"""
function _build_penalized_hessian(H_unpen::Matrix{Float64}, lambda::AbstractVector{T}, 
                                   config::PenaltyConfig;
                                   beta::Union{Nothing, AbstractVector{<:Real}}=nothing) where T<:Real
    # Convert to eltype compatible with lambda
    H = convert(Matrix{T}, copy(H_unpen))
    
    lambda_idx = 1
    
    # Baseline hazard penalty terms - quadratic penalty with Hessian = λS
    for term in config.terms
        idx = term.hazard_indices
        lam = lambda[lambda_idx]
        
        # Penalty Hessian is λS (penalty is quadratic in natural-scale β)
        @inbounds for i in idx, j in idx
            H[i, j] += lam * term.S[i - first(idx) + 1, j - first(idx) + 1]
        end
        lambda_idx += 1
    end
    
    # Total hazard terms (for competing risks) - sum of natural-scale coefficients
    for term in config.total_hazard_terms
        # Total hazard: H(t) = Σ h_k(t), penalty on sum of coefficients
        # For natural-scale params, Hessian is simply λS per competing hazard
        lam = lambda[lambda_idx]
        for idx_range in term.hazard_indices
            @inbounds for i in idx_range, j in idx_range
                H[i, j] += lam * term.S[i - first(idx_range) + 1, j - first(idx_range) + 1]
            end
        end
        lambda_idx += 1
    end
    
    # Smooth covariate penalty terms - no transformation needed
    if !isempty(config.shared_smooth_groups)
        term_to_lambda = Dict{Int, Int}()
        
        for (group_idx, group) in enumerate(config.shared_smooth_groups)
            for term_idx in group
                term_to_lambda[term_idx] = lambda_idx
            end
            lambda_idx += 1
        end
        
        for (term_idx, term) in enumerate(config.smooth_covariate_terms)
            if !haskey(term_to_lambda, term_idx)
                term_to_lambda[term_idx] = lambda_idx
                lambda_idx += 1
            end
        end
        
        for (term_idx, term) in enumerate(config.smooth_covariate_terms)
            lam = lambda[term_to_lambda[term_idx]]
            indices = term.param_indices
            @inbounds for (i, pi) in enumerate(indices)
                for (j, pj) in enumerate(indices)
                    H[pi, pj] += lam * term.S[i, j]
                end
            end
        end
    else
        for term in config.smooth_covariate_terms
            indices = term.param_indices
            @inbounds for (i, pi) in enumerate(indices)
                for (j, pj) in enumerate(indices)
                    H[pi, pj] += lambda[lambda_idx] * term.S[i, j]
                end
            end
            lambda_idx += 1
        end
    end
    
    return H
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
    select_smoothing_parameters(model::MultistateModel, penalty::SplinePenalty;
                                 method::Symbol=:pijcv,
                                 scope::Symbol=:all,
                                 max_outer_iter::Int=20,
                                 inner_maxiter::Int=50,
                                 lambda_maxiter::Int=100,
                                 beta_tol::Float64=1e-4,
                                 lambda_tol::Float64=1e-3,
                                 lambda_init::Float64=1.0,
                                 verbose::Bool=false) -> NamedTuple

User-friendly interface for smoothing parameter selection using performance iteration.

This function implements mgcv-style "performance iteration" that alternates:
1. Given β, compute Hessians and optimize λ via PIJCV/PERF/EFS
2. Given λ, update β via penalized maximum likelihood

This is more accurate than one-shot approximation when the penalized solution β(λ*)
differs significantly from the unpenalized solution β(0).

# Arguments
- `model`: MultistateModel
- `penalty`: SplinePenalty specification
- `method`: Selection method. Options:
  - Newton-approximated CV (fast): `:pijcv`/`:pijlcv` (LOO), `:pijcv5`, `:pijcv10`, `:pijcv20` (k-fold)
  - Exact CV (slow but gold standard): `:loocv`, `:cv5`, `:cv10`, `:cv20`
  - Other: `:perf`, `:efs`
- `scope`: Which splines to calibrate (:all, :baseline, or :covariates)
- `max_outer_iter`: Maximum outer iterations (β-λ alternation)
- `inner_maxiter`: Maximum iterations for β update per outer iteration
- `lambda_maxiter`: Maximum iterations for λ optimization
- `beta_tol`: Convergence tolerance for β (relative change)
- `lambda_tol`: Convergence tolerance for λ (relative change in log scale)
- `lambda_init`: Initial value for smoothing parameters
- `verbose`: Print progress

# Returns
NamedTuple with:
- `lambda`: Optimal smoothing parameters
- `beta`: Final coefficient estimate
- `edf`: NamedTuple with `total` (total model EDF) and `per_term` (Vector of per-term EDFs)
- `criterion`: Final criterion value
- `converged`: Whether optimization converged
- `method_used`: Actual method used (may fall back from PIJCV to EFS)
- `penalty_config`: Updated penalty configuration with optimal λ
- `n_outer_iter`: Number of outer iterations performed

# Notes on Cross-Validation Methods

**Exact CV methods** (require refitting for each fold/subject):
- `:loocv`: Exact leave-one-out CV. O(n × grid_points × fitting_cost)
- `:cv5`, `:cv10`, `:cv20`: Exact K-fold CV with 5, 10, or 20 folds. O(k × grid_points × fitting_cost)

**Newton-approximated CV methods** (fast, no refitting):
- `:pijcv` / `:pijlcv`: Newton-approximated leave-one-out CV (NCV from Wood 2024). O(grid_points × n × p²)
- `:pijcv5`, `:pijcv10`, `:pijcv20`: Newton-approximated K-fold CV. O(grid_points × k × p³)
  These generalize Wood's NCV to k-fold by approximating β̂⁻ᵏ via a Newton step.
"""
function select_smoothing_parameters(model::MultistateModel, penalty::SplinePenalty;
                                      method::Symbol=:pijcv,
                                      scope::Symbol=:all,
                                      max_outer_iter::Int=20,
                                      inner_maxiter::Int=50,
                                      lambda_maxiter::Int=100,
                                      beta_tol::Float64=1e-4,
                                      lambda_tol::Float64=1e-3,
                                      lambda_init::Float64=1.0,
                                      verbose::Bool=false)
    # Extract sample paths
    samplepaths = extract_paths(model)
    
    # Build penalty configuration
    penalty_config = build_penalty_config(model, penalty; lambda_init=lambda_init)
    
    # Create ExactData container
    data = ExactData(model, samplepaths)
    
    # Fit unpenalized model to get initial coefficients
    if verbose
        println("Fitting unpenalized model for initialization...")
    end
    
    # Get initial parameters from model
    parameters = get_parameters_flat(model)
    
    # Define unpenalized likelihood function
    loglik_fn = loglik_exact
    adtype = DifferentiationInterface.SecondOrder(
        Optimization.AutoForwardDiff(), 
        Optimization.AutoForwardDiff()
    )
    optf = OptimizationFunction(loglik_fn, adtype)
    prob = OptimizationProblem(optf, parameters, data)
    
    sol = _solve_optimization(prob, IpoptOptimizer())
    beta_unpen = sol.u
    
    if verbose
        println("Unpenalized fit complete.")
    end
    
    # Fit PENALIZED model with initial λ to get proper starting point for performance iteration
    # This is critical: starting from unpenalized MLE causes PIJCV to push λ → ∞
    # because at the unpenalized MLE, the score is zero and more smoothing only reduces variance
    if verbose
        println("Fitting penalized model with initial λ=$(lambda_init)...")
    end
    lambda_init_vec = fill(lambda_init, penalty_config.n_lambda)
    beta_init = fit_penalized_beta(model, data, lambda_init_vec, penalty_config, beta_unpen;
                                   maxiters=inner_maxiter, use_polyalgorithm=true, verbose=false)
    if verbose
        println("Penalized initialization complete.")
    end
    
    # Call the internal function
    return select_smoothing_parameters(model, data, penalty_config, beta_init;
                                        method=method, scope=scope, 
                                        max_outer_iter=max_outer_iter,
                                        inner_maxiter=inner_maxiter,
                                        lambda_maxiter=lambda_maxiter,
                                        beta_tol=beta_tol, lambda_tol=lambda_tol,
                                        verbose=verbose)
end

"""
    select_smoothing_parameters(model::MultistateProcess, data::ExactData,
                                 penalty_config::PenaltyConfig,
                                 beta_init::Vector{Float64};
                                 method::Symbol=:pijcv,
                                 scope::Symbol=:all,
                                 max_outer_iter::Int=20,
                                 inner_maxiter::Int=50,
                                 lambda_maxiter::Int=100,
                                 beta_tol::Float64=1e-4,
                                 lambda_tol::Float64=1e-3,
                                 verbose::Bool=false) -> NamedTuple

Select smoothing parameters using mgcv-style performance iteration.

Algorithm:
1. Initialize β from unpenalized fit, λ from penalty config
2. Outer loop:
   a. Given β, compute Hessians and optimize λ via PIJCV/PERF/EFS
   b. Given λ, update β via penalized maximum likelihood (warm-started)
   c. Check convergence
3. Return final (β, λ)

# Arguments
- `model`: MultistateProcess model
- `data`: ExactData container
- `penalty_config`: Penalty configuration with initial λ values
- `beta_init`: Initial coefficient estimate (warm start)
- `method`: Selection method. Options:
  - Newton-approximated CV (fast): `:pijcv`/`:pijlcv` (LOO), `:pijcv5`, `:pijcv10`, `:pijcv20` (k-fold)
  - Exact CV (slow but gold standard): `:loocv`, `:cv5`, `:cv10`, `:cv20`
  - Other: `:perf`, `:efs`
- `scope`: Which splines to calibrate:
  - `:all` (default): Calibrate all spline penalties (baseline + covariates)
  - `:baseline`: Calibrate only baseline hazard splines
  - `:covariates`: Calibrate only smooth covariate splines
- `max_outer_iter`: Maximum outer iterations (β-λ alternation)
- `inner_maxiter`: Maximum iterations for β update
- `lambda_maxiter`: Maximum iterations for λ optimization
- `beta_tol`: Convergence tolerance for β (relative change)
- `lambda_tol`: Convergence tolerance for λ (relative change in log scale)
- `verbose`: Print progress

# Returns
NamedTuple with:
- `lambda`: Optimal smoothing parameters (for calibrated terms only)
- `beta`: Final coefficient estimate
- `edf`: NamedTuple with `total` (total model EDF) and `per_term` (Vector of per-term EDFs)
- `criterion`: Final criterion value
- `converged`: Whether optimization converged
- `method_used`: Actual method used (may fall back from PIJCV to EFS)
- `penalty_config`: Updated penalty configuration with optimal λ
- `n_outer_iter`: Number of outer iterations performed

# Notes on EDF Computation
EDF is only computed for exact data models. For MCEM-fitted models, the EDF
fields will contain NaN values. See `compute_edf` for details.

# Notes on Cross-Validation Methods

**Exact CV methods** (require refitting for each fold/subject):
- `:loocv`: Exact leave-one-out CV. O(n × grid_points × fitting_cost)
- `:cv5`, `:cv10`, `:cv20`: Exact K-fold CV with 5, 10, or 20 folds. O(k × grid_points × fitting_cost)

**Newton-approximated CV methods** (fast, no refitting):
- `:pijcv` / `:pijlcv`: Newton-approximated leave-one-out CV (NCV from Wood 2024). O(grid_points × n × p²)
- `:pijcv5`, `:pijcv10`, `:pijcv20`: Newton-approximated K-fold CV. O(grid_points × k × p³)
  These generalize Wood's NCV to k-fold by approximating β̂⁻ᵏ via a Newton step.
"""
function select_smoothing_parameters(model::MultistateProcess, data::ExactData,
                                      penalty_config::PenaltyConfig,
                                      beta_init::Vector{Float64};
                                      method::Symbol=:pijcv,
                                      scope::Symbol=:all,
                                      max_outer_iter::Int=20,
                                      inner_maxiter::Int=50,
                                      lambda_maxiter::Int=100,
                                      beta_tol::Float64=1e-4,
                                      lambda_tol::Float64=1e-3,
                                      verbose::Bool=false)
    # Validate method
    valid_methods = (:pijcv, :pijlcv, :pijcv5, :pijcv10, :pijcv20, :perf, :efs, :loocv, :cv5, :cv10, :cv20)
    method ∈ valid_methods || 
        throw(ArgumentError("method must be one of $valid_methods, got :$method"))
    
    # Validate scope
    scope ∈ (:all, :baseline, :covariates) || 
        throw(ArgumentError("scope must be :all, :baseline, or :covariates, got :$scope"))
    
    # Create scoped penalty config based on scope parameter
    scoped_config, fixed_baseline_lambdas, fixed_covariate_lambdas = _create_scoped_penalty_config(penalty_config, scope)
    
    n_lambda = scoped_config.n_lambda
    n_lambda > 0 || return (lambda=Float64[], beta=beta_init, edf=(total=0.0, per_term=Float64[]), 
                            criterion=NaN, converged=true, method_used=:none, 
                            penalty_config=penalty_config, n_outer_iter=0)
    
    samplepaths = data.paths
    n_params = length(beta_init)
    n_subjects = length(samplepaths)
    
    # Check if method supports direct λ optimization (Newton-approximated methods)
    # Exact CV methods require refitting at each λ - use simpler approach
    use_direct_optimization = method ∈ (:pijcv, :pijlcv, :pijcv5, :pijcv10, :pijcv20, :perf, :efs)
    
    if !use_direct_optimization
        # For exact CV methods, fall back to coarse grid + refinement
        return _select_lambda_grid_search(model, data, scoped_config, beta_init, method, 
                                          inner_maxiter, verbose, samplepaths, n_subjects, n_params,
                                          penalty_config, fixed_baseline_lambdas, fixed_covariate_lambdas, scope)
    end
    
    if verbose
        println("Optimizing λ via performance iteration (Wood 2024)")
        println("  Method: $method")
        println("  n_lambda: $n_lambda")
    end
    
    # Warn user about multi-λ shared optimization
    if n_lambda > 1
        @warn """Multiple smoothing parameters detected (n_lambda=$n_lambda).
        Current implementation uses shared λ for all smooth terms.
        Consider calibrating each term separately with `scope=:baseline` or `scope=:covariates`."""
    end
    
    # Performance iteration: alternate between optimizing λ and updating β
    # At each iteration:
    #   1. Given β, optimize λ via AD on V(λ) (criterion is differentiable w.r.t. λ)
    #   2. Given λ, update β via penalized MLE
    
    current_beta = copy(beta_init)
    current_log_lambda = zeros(n_lambda)  # Start at λ = 1
    best_criterion = Inf
    n_total_evals = 0
    converged = false
    
    for outer_iter in 1:max_outer_iter
        # Step 1: Compute subject gradients and Hessians at current β
        subject_grads_ll = compute_subject_gradients(current_beta, model, samplepaths)
        subject_hessians_ll = compute_subject_hessians_fast(current_beta, model, samplepaths)
        
        # Convert to loss convention (negative log-likelihood)
        subject_grads = -subject_grads_ll
        subject_hessians = [-H for H in subject_hessians_ll]
        H_unpenalized = sum(subject_hessians)
        
        # Create state for criterion evaluation
        state = SmoothingSelectionState(
            copy(current_beta),
            H_unpenalized,
            subject_grads,
            subject_hessians,
            scoped_config,
            n_subjects,
            n_params,
            model,
            data
        )
        
        # Step 2: Optimize λ at fixed β via AD
        # The criterion V(λ) is directly differentiable w.r.t. log(λ)
        function criterion_at_fixed_beta(log_lambda_vec, _)
            if method ∈ (:pijcv, :pijlcv)
                compute_pijcv_criterion(log_lambda_vec, state)
            elseif method == :pijcv5
                compute_pijkfold_criterion(log_lambda_vec, state, 5)
            elseif method == :pijcv10
                compute_pijkfold_criterion(log_lambda_vec, state, 10)
            elseif method == :pijcv20
                compute_pijkfold_criterion(log_lambda_vec, state, 20)
            elseif method == :perf
                compute_perf_criterion(log_lambda_vec, state)
            else  # :efs
                compute_efs_criterion(log_lambda_vec, state)
            end
        end
        
        # Set up bounded optimization on log(λ) ∈ [-8, 8]
        lb = fill(-8.0, n_lambda)
        ub = fill(8.0, n_lambda)
        
        # Use ForwardDiff for gradients - criterion is AD-compatible
        adtype = Optimization.AutoForwardDiff()
        optf = OptimizationFunction(criterion_at_fixed_beta, adtype)
        prob = OptimizationProblem(optf, current_log_lambda, nothing; lb=lb, ub=ub)
        
        # Solve with LBFGS (gradient-based, bounded)
        sol = solve(prob, LBFGS();
                    maxiters=lambda_maxiter,
                    abstol=lambda_tol,
                    reltol=lambda_tol)
        
        new_log_lambda = sol.u
        new_criterion = sol.objective
        n_total_evals += 1  # Count outer iterations
        
        if verbose
            lambda_val = exp(new_log_lambda[1])
            println("  Iter $outer_iter: log(λ)=$(round(new_log_lambda[1], digits=2)), λ=$(round(lambda_val, sigdigits=3)), V=$(round(new_criterion, digits=3))")
        end
        
        # Check convergence in λ
        lambda_change = maximum(abs.(new_log_lambda .- current_log_lambda))
        
        # Step 3: Update β at new λ
        new_lambda = exp.(new_log_lambda)
        lambda_vec = n_lambda == 1 ? fill(new_lambda[1], scoped_config.n_lambda) : new_lambda
        
        new_beta = fit_penalized_beta(model, data, lambda_vec, scoped_config, current_beta;
                                      maxiters=inner_maxiter,
                                      use_polyalgorithm=false,
                                      verbose=false)
        
        # Check convergence in β
        beta_change = maximum(abs.(new_beta .- current_beta))
        
        # Update state
        current_log_lambda = new_log_lambda
        current_beta = new_beta
        best_criterion = new_criterion
        
        # Check convergence
        if lambda_change < lambda_tol && beta_change < beta_tol
            converged = true
            if verbose
                println("  Converged after $outer_iter iterations")
            end
            break
        end
    end
    
    # Final results
    best_lambda = exp.(current_log_lambda)
    best_lambda_vec = n_lambda == 1 ? fill(best_lambda[1], scoped_config.n_lambda) : best_lambda
    
    if verbose && !converged
        println("  Warning: Did not converge after $max_outer_iter iterations")
    end
    
    if verbose
        println("\n  Optimal log(λ): $(round.(current_log_lambda, digits=3))")
        println("  Optimal λ: $(round.(best_lambda, sigdigits=4))")
        println("  Final criterion: $(round(best_criterion, digits=4))")
    end
    
    # Update penalty config with final λ
    updated_config = _merge_scoped_lambdas(penalty_config, scoped_config, best_lambda_vec, 
                                           fixed_baseline_lambdas, fixed_covariate_lambdas, scope)
    
    # Compute EDF at optimal (lambda, beta)
    edf_vec = compute_edf(current_beta, best_lambda_vec, updated_config, model, data)
    
    return (
        lambda = best_lambda_vec,
        beta = current_beta,
        edf = edf_vec,
        criterion = best_criterion,
        converged = converged,
        method_used = method,
        penalty_config = updated_config,
        n_outer_iter = n_total_evals
    )
end

"""
    _select_lambda_grid_search(...)

Fallback grid search for exact CV methods that require refitting at each λ.
Used for :loocv, :cv5, :cv10, :cv20 methods.
"""
function _select_lambda_grid_search(model, data, scoped_config, beta_init, method,
                                    inner_maxiter, verbose, samplepaths, n_subjects, n_params,
                                    penalty_config, fixed_baseline_lambdas, fixed_covariate_lambdas, scope)
    n_lambda = scoped_config.n_lambda
    
    if verbose
        println("Using grid search for exact CV method: $method")
        println("  WARNING: This requires n refits per λ value")
    end
    
    # Coarse grid + refinement
    log_lambda_grid = collect(-4.0:1.0:4.0)  # 9 points
    
    best_lambda = Float64[]
    best_beta = copy(beta_init)
    best_criterion = Inf
    
    for log_lam in log_lambda_grid
        lam = exp(log_lam)
        lambda_vec = fill(lam, n_lambda)
        
        beta_lam = fit_penalized_beta(model, data, lambda_vec, scoped_config, beta_init;
                                       maxiters=inner_maxiter, verbose=false)
        
        criterion = if method == :loocv
            compute_loocv_criterion(lambda_vec, beta_lam, model, data, scoped_config;
                                    maxiters=inner_maxiter, verbose=false)
        elseif method == :cv5
            compute_kfold_cv_criterion(lambda_vec, beta_lam, model, data, scoped_config, 5;
                                       maxiters=inner_maxiter, verbose=false)
        elseif method == :cv10
            compute_kfold_cv_criterion(lambda_vec, beta_lam, model, data, scoped_config, 10;
                                       maxiters=inner_maxiter, verbose=false)
        else  # :cv20
            compute_kfold_cv_criterion(lambda_vec, beta_lam, model, data, scoped_config, 20;
                                       maxiters=inner_maxiter, verbose=false)
        end
        
        if verbose
            println("  log(λ)=$(round(log_lam, digits=1)), V=$(round(criterion, digits=3))")
        end
        
        if criterion < best_criterion
            best_criterion = criterion
            best_lambda = lambda_vec
            best_beta = copy(beta_lam)
        end
    end
    
    # Refinement around best
    best_log_lam = log(best_lambda[1])
    fine_grid = range(best_log_lam - 0.5, best_log_lam + 0.5, length=5)
    
    for log_lam in fine_grid
        lam = exp(log_lam)
        lambda_vec = fill(lam, n_lambda)
        
        beta_lam = fit_penalized_beta(model, data, lambda_vec, scoped_config, best_beta;
                                       maxiters=inner_maxiter, verbose=false)
        
        criterion = if method == :loocv
            compute_loocv_criterion(lambda_vec, beta_lam, model, data, scoped_config;
                                    maxiters=inner_maxiter, verbose=false)
        elseif method == :cv5
            compute_kfold_cv_criterion(lambda_vec, beta_lam, model, data, scoped_config, 5;
                                       maxiters=inner_maxiter, verbose=false)
        elseif method == :cv10
            compute_kfold_cv_criterion(lambda_vec, beta_lam, model, data, scoped_config, 10;
                                       maxiters=inner_maxiter, verbose=false)
        else
            compute_kfold_cv_criterion(lambda_vec, beta_lam, model, data, scoped_config, 20;
                                       maxiters=inner_maxiter, verbose=false)
        end
        
        if criterion < best_criterion
            best_criterion = criterion
            best_lambda = lambda_vec
            best_beta = copy(beta_lam)
        end
    end
    
    # Update penalty config
    updated_config = _merge_scoped_lambdas(penalty_config, scoped_config, best_lambda, 
                                           fixed_baseline_lambdas, fixed_covariate_lambdas, scope)
    
    edf_vec = compute_edf(best_beta, best_lambda, updated_config, model, data)
    
    return (
        lambda = best_lambda,
        beta = best_beta,
        edf = edf_vec,
        criterion = best_criterion,
        converged = true,
        method_used = method,
        penalty_config = updated_config,
        n_outer_iter = length(log_lambda_grid) + 5
    )
end

"""
    compute_perf_criterion(log_lambda::AbstractVector{T}, state::SmoothingSelectionState) where T<:Real

Compute PERF (Performance Iteration) criterion from Marra & Radice (2020).

PERF is an AIC-type criterion based on prediction error:
    V_PERF(λ) = ‖M - OM‖² - ň + 2·edf

where:
- M = H^{1/2}β + H^{-1/2}(−g) is the working response
- O = H^{1/2}(H + S_λ)^{-1}H^{1/2} is the influence matrix
- edf = tr(O) is the effective degrees of freedom
- ň is a constant (typically n) that doesn't affect optimization

This criterion estimates the complexity of smooth terms not supported by data
and suppresses them, providing stable smoothing parameter selection.

# Arguments
- `log_lambda`: Vector of log-smoothing parameters (one per penalty term)
- `state`: SmoothingSelectionState with cached gradients/Hessians

# Returns
- `Float64`: PERF criterion value (lower is better)

# References
- Marra, G. & Radice, R. (2020). "Copula link-based additive models for 
  right-censored event time data." JASA 115(530):886-895.
- Eletti, A., Marra, G. & Radice, R. (2024). "Spline-Based Multi-State Models 
  for Analyzing Disease Progression." arXiv:2312.05345v4, Appendix C
"""
function compute_perf_criterion(log_lambda::AbstractVector{T}, state::SmoothingSelectionState) where T<:Real
    lambda = exp.(log_lambda)
    
    # Build penalized Hessian (penalty is quadratic: λ βᵀSβ)
    H = state.H_unpenalized
    H_lambda = _build_penalized_hessian(H, lambda, state.penalty_config; beta=state.beta_hat)
    
    n = state.n_subjects
    p = state.n_params
    
    # Compute H^{1/2} via eigendecomposition for numerical stability
    # H should be positive definite (we're at a maximum)
    H_sym = Symmetric(H)
    eig = try
        eigen(H_sym)
    catch e
        return T(1e10)  # Return large value if eigen fails
    end
    
    # Check for positive definiteness
    if any(eig.values .< 1e-10)
        # H is not positive definite - regularize
        min_eval = minimum(eig.values)
        regularization = abs(min_eval) + 1e-6
        H_reg = H + regularization * I(p)
        eig = eigen(Symmetric(H_reg))
    end
    
    # H^{1/2} = V * D^{1/2} * V'
    sqrt_evals = sqrt.(max.(eig.values, 1e-10))
    H_sqrt = eig.vectors * Diagonal(sqrt_evals) * eig.vectors'
    
    # H^{-1/2} = V * D^{-1/2} * V'
    inv_sqrt_evals = 1.0 ./ sqrt_evals
    H_inv_sqrt = eig.vectors * Diagonal(inv_sqrt_evals) * eig.vectors'
    
    # Compute gradient of loss at current estimate
    # At penalized MLE: ∇ℓ + S_λβ = 0, so ∇ℓ = -S_λβ
    # For PERF, we need -∇ℓ = S_λβ (loss gradient, not log-likelihood)
    # But for robustness, compute directly from subject gradients
    # subject_grads are already in loss convention (negated log-likelihood)
    g = vec(sum(state.subject_grads, dims=2))  # Total loss gradient
    
    # Working response: M = H^{1/2}β + H^{-1/2}(−g)
    # Note: The −g term represents the residual; at MLE, g ≈ 0 for unpenalized,
    # but at penalized MLE, g = -S_λβ ≠ 0
    beta_T = convert(Vector{T}, state.beta_hat)
    g_T = convert(Vector{T}, g)
    
    M = H_sqrt * beta_T - H_inv_sqrt * g_T
    
    # Influence matrix: O = H^{1/2} (H + S_λ)^{-1} H^{1/2}
    H_lambda_sym = Symmetric(H_lambda)
    H_lambda_inv = try
        inv(H_lambda_sym)
    catch e
        return T(1e10)
    end
    
    O = H_sqrt * H_lambda_inv * H_sqrt
    
    # Effective degrees of freedom
    edf = tr(O)
    
    # Fitted values in working space: OM
    OM = O * M
    
    # Residual: M - OM
    residual = M - OM
    
    # Residual sum of squares: ‖M - OM‖²
    RSS = dot(residual, residual)
    
    # PERF criterion: RSS - n + 2*edf
    # The -n term is a constant that doesn't affect optimization,
    # but we include it for completeness
    perf = RSS - n + 2 * edf
    
    return perf
end

"""
    compute_efs_criterion(log_lambda::AbstractVector{T}, state::SmoothingSelectionState) where T<:Real

Compute the negative REML (Restricted Maximum Likelihood) criterion for EFS method.

EFS (Extended Fellner-Schall) from Wood & Fasiolo (2017) maximizes the Laplace-approximated
restricted marginal likelihood:

    ℓ_LA(λ) = ℓ(θ̂) - ½θ̂ᵀS_λθ̂ + ½log|S_λ|₊ - ½log|H + S_λ|

where |S_λ|₊ is the pseudo-determinant (product of non-zero eigenvalues).

This function returns the NEGATIVE of ℓ_LA so that minimization corresponds to REML maximization.

# Arguments
- `log_lambda`: Vector of log-smoothing parameters (one per penalty term)
- `state`: SmoothingSelectionState with cached gradients/Hessians

# Returns
- `Float64`: Negative REML criterion value (lower is better, i.e., higher REML)

# References
- Wood, S.N. & Fasiolo, M. (2017). "A generalized Fellner-Schall method for smoothing 
  parameter estimation." Statistics and Computing 27(3):759-773.
- Eletti, A., Marra, G. & Radice, R. (2024). arXiv:2312.05345v4, Appendix C
"""
function compute_efs_criterion(log_lambda::AbstractVector{T}, state::SmoothingSelectionState) where T<:Real
    lambda = exp.(log_lambda)
    
    # Build penalized Hessian (penalty is quadratic: λ βᵀSβ)
    H = state.H_unpenalized
    H_lambda = _build_penalized_hessian(H, lambda, state.penalty_config; beta=state.beta_hat)
    
    p = state.n_params
    beta = state.beta_hat
    
    # Compute log-likelihood at current estimate
    ll_subj = loglik_exact(beta, state.data; neg=false, return_ll_subj=true)
    ll_theta = sum(ll_subj)
    
    # Build total penalty matrix S_λ = Σⱼ λⱼ Sⱼ
    S_lambda = zeros(T, p, p)
    lambda_idx = 1
    for term in state.penalty_config.terms
        idx = term.hazard_indices
        for i in idx, j in idx
            S_lambda[i, j] += lambda[lambda_idx] * term.S[i - first(idx) + 1, j - first(idx) + 1]
        end
        lambda_idx += 1
    end
    
    # Penalty term: -½ θ̂ᵀ S_λ θ̂
    beta_T = convert(Vector{T}, beta)
    penalty_term = -0.5 * dot(beta_T, S_lambda * beta_T)
    
    # Log pseudo-determinant of S_λ: ½ log|S_λ|₊
    # Use eigenvalues, keeping only positive ones (for rank-deficient S)
    eig_S = eigen(Symmetric(S_lambda))
    pos_eigs_S = filter(e -> e > 1e-10, eig_S.values)
    log_det_S = if isempty(pos_eigs_S)
        T(0.0)  # S_λ is zero (no penalty)
    else
        0.5 * sum(log.(pos_eigs_S))
    end
    
    # Log determinant of penalized Hessian: -½ log|H + S_λ|
    H_lambda_sym = Symmetric(H_lambda)
    eig_H = try
        eigen(H_lambda_sym)
    catch e
        return T(1e10)
    end
    
    # Check for positive definiteness
    if any(eig_H.values .< 1e-10)
        # Not positive definite - return large value
        return T(1e10)
    end
    
    log_det_H_lambda = -0.5 * sum(log.(eig_H.values))
    
    # REML criterion: ℓ(θ̂) - ½θ̂ᵀS_λθ̂ + ½log|S_λ|₊ - ½log|H + S_λ|
    reml = ll_theta + penalty_term + log_det_S + log_det_H_lambda
    
    # Return negative for minimization
    return -reml
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

"""
    compute_edf(beta::Vector{Float64}, lambda::Vector{Float64}, 
                penalty_config::PenaltyConfig, model::MultistateProcess,
                data::ExactData) -> Vector{Float64}

Compute effective degrees of freedom (EDF) for each spline penalty term.

EDF measures the effective number of parameters used by each smooth term,
accounting for the shrinkage imposed by the penalty. For an unpenalized
term, EDF equals the number of basis functions; as λ → ∞, EDF → 0.

The EDF for term j is computed as:
    edf_j = tr(H_j * H_λ^{-1})

where:
- A = H_unpen × H_λ⁻¹ is the influence (hat) matrix
- H_λ is the full penalized Hessian

The per-term EDF is the sum of diagonal elements of A for that term's parameter indices.
The total model EDF is the sum of all per-term EDFs (= tr(A)).

# Arguments
- `beta`: Current coefficient estimate
- `lambda`: Current smoothing parameters
- `penalty_config`: Penalty configuration
- `model`: MultistateProcess model  
- `data`: ExactData container

# Returns
NamedTuple with:
- `total`: Total model EDF (scalar)
- `per_term`: Vector of EDF values, one per penalty term

# Limitations
- **Exact data only**: This function requires exact observation times. For MCEM-fitted
  models with panel data, the observed information matrix has a different structure
  that accounts for the Monte Carlo variance. EDF computation for MCEM is not yet
  implemented.
- **Subject weights**: Subject weights from `model.SubjectWeights` are included in the
  Hessian computation via `compute_subject_hessians_fast`.

# References
- Wood, S.N. (2017). Generalized Additive Models: An Introduction with R. 2nd ed.
  Section 6.1.2 discusses EDF computation for penalized models.
"""
function compute_edf(beta::Vector{Float64}, lambda::Vector{Float64},
                     penalty_config::PenaltyConfig, model::MultistateProcess,
                     data::ExactData)
    # Compute subject-level Hessians (includes subject weights)
    samplepaths = extract_paths(model)
    subject_hessians_ll = compute_subject_hessians_fast(beta, model, samplepaths)
    
    # Validate subject Hessians for NaN/Inf
    nan_subjects = findall(H -> any(!isfinite, H), subject_hessians_ll)
    if !isempty(nan_subjects)
        @warn "$(length(nan_subjects)) subject Hessians contain NaN/Inf values. " *
              "Check for extreme parameter values or zero hazards. " *
              "Affected subjects: $(first(nan_subjects, 5))..." maxlog=3
    end
    
    # Aggregate to full Hessian (negative because we want Fisher information)
    H_unpenalized = -sum(subject_hessians_ll)
    
    # Build penalized Hessian (penalty is quadratic: λ βᵀSβ)
    H_lambda = _build_penalized_hessian(H_unpenalized, lambda, penalty_config; beta=beta)
    
    # Validate penalized Hessian
    if !all(isfinite.(H_lambda))
        nan_count = count(isnan, H_lambda)
        inf_count = count(isinf, H_lambda)
        @warn "Penalized Hessian contains non-finite values ($(nan_count) NaN, $(inf_count) Inf). " *
              "Returning NaN EDFs."
        n_terms = length(penalty_config.terms) + length(penalty_config.smooth_covariate_terms)
        return (total = NaN, per_term = fill(NaN, n_terms))
    end
    
    # Invert penalized Hessian
    H_lambda_inv = try
        inv(Symmetric(H_lambda))
    catch e
        @warn "Failed to invert penalized Hessian for EDF computation: $e. " *
              "Matrix may be singular or ill-conditioned. cond(H) = $(cond(H_lambda))"
        n_terms = length(penalty_config.terms) + length(penalty_config.smooth_covariate_terms)
        return (total = NaN, per_term = fill(NaN, n_terms))
    end
    
    # Compute influence matrix A = H_unpen * H_lambda_inv
    # EDF for term j = sum of diagonal elements of A for indices in term j
    # Total EDF = tr(A) = sum of all diagonal elements
    A = H_unpenalized * H_lambda_inv
    
    # Compute per-term EDF
    edf_vec = Float64[]
    
    # Process baseline hazard terms
    for term in penalty_config.terms
        idx = term.hazard_indices
        # Per-term EDF = sum of A[i,i] for i in this term's indices
        edf_j = sum(A[i, i] for i in idx)
        push!(edf_vec, edf_j)
    end
    
    # Process smooth covariate terms
    for term in penalty_config.smooth_covariate_terms
        idx = term.param_indices
        # Per-term EDF = sum of A[i,i] for i in this term's indices
        edf_j = sum(A[i, i] for i in idx)
        push!(edf_vec, edf_j)
    end
    
    # Total EDF = tr(A) = sum of all per-term EDFs
    # (Note: this equals sum(edf_vec) only if all parameters belong to some term,
    #  which should be true for properly constructed penalty configs)
    total_edf = tr(A)
    
    return (total = total_edf, per_term = edf_vec)
end
