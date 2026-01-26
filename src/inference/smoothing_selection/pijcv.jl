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
            @debug "Cholesky factorization failed in _ncv_criterion (LOO): " exception=(e, catch_backtrace()) lambda=lambda
            nothing
        end
    else
        nothing  # Skip factorization for AD mode
    end
    
    # If Cholesky failed and not in AD mode, return large value indicating poor λ
    if isnothing(chol_fact) && use_cholesky_downdate
        @debug "Returning large criterion value due to Cholesky failure"
        return T(1e10)
    end
    
    # Get subject-level likelihoods at current estimate (for V_q fallback)
    ll_subj_base = loglik_exact(state.beta_hat, state.data; neg=false, return_ll_subj=true)
    
    # PIJCV optimization: Build or reuse evaluation cache for efficient LOO evaluation
    # This avoids rebuilding SubjectCovarCache for each of n subjects
    eval_cache = if isnothing(state.pijcv_eval_cache)
        # First call: build and store the cache
        cache = build_pijcv_eval_cache(state.data)
        state.pijcv_eval_cache = cache
        cache
    else
        state.pijcv_eval_cache
    end
    
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
        
        if isnothing(delta_i)
            # Either Cholesky downdate failed or we're in AD mode
            # Use direct solve: (H_λ - H_i)⁻¹ g_i
            H_lambda_loo = H_lambda - H_i
            H_loo_sym = Symmetric(H_lambda_loo)
            
            delta_i = try
                H_loo_sym \ collect(g_i)
            catch e
                # If solve fails (e.g., indefinite Hessian), return large value
                @debug "Linear solve failed for LOO Hessian in _ncv_criterion: " exception=(e, catch_backtrace()) subject=i
                return T(1e10)
            end
        end
        
        # Compute LOO parameters: β̂⁻ⁱ = β̂ + Δ⁻ⁱ
        beta_loo = state.beta_hat .+ delta_i
        
        # Note: As of v0.3.0, baseline parameters are on NATURAL scale (positive values).
        # Box constraints ensure positivity during optimization.
        # Some spline coefficients may be negative when they control log-hazard scale.
        
        # CORE OF NCV: Evaluate ACTUAL likelihood at LOO parameters using cached structures
        ll_loo = loglik_subject_cached(beta_loo, eval_cache, i)
        
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
    compute_pijcv_criterion_fast(log_lambda::Vector{Float64}, state::SmoothingSelectionState) -> Float64

Compute the PIJCV/NCV criterion V_q(λ) using the quadratic approximation.

This is a **FAST** version that uses the quadratic (Taylor) approximation V_q from
Wood (2024), Section 4.1, instead of evaluating the actual likelihood at LOO parameters.

# Quadratic Approximation V_q (Wood 2024, Equation 5)

    V_q(λ) = Σᵢ [ -ℓᵢ(β̂) + gᵢᵀ Δ⁻ⁱ + ½ (Δ⁻ⁱ)ᵀ Hᵢ Δ⁻ⁱ ]

where:
- ℓᵢ(β̂) is subject i's log-likelihood at the full MLE
- gᵢ = -∇ℓᵢ(β̂) is the negative gradient
- Hᵢ = -∇²ℓᵢ(β̂) is the negative Hessian
- Δ⁻ⁱ = H_{λ,-i}⁻¹ gᵢ is the Newton step

# Why This Is Faster

Unlike `compute_pijcv_criterion`, this function:
1. Does NOT call `loglik_subject_cached` for each of n subjects
2. Only uses the pre-computed gradients and Hessians
3. Cost is O(n × p²) matrix-vector operations vs O(n × likelihood_eval)

# When To Use

- Use this for faster λ selection during initial search
- The full NCV criterion (actual likelihood) can be used for final validation
- Wood (2024) shows V_q has similar asymptotic properties to V

# Arguments
- `log_lambda`: Vector of log-smoothing parameters (one per penalty term)
- `state`: SmoothingSelectionState with cached gradients/Hessians

# Returns
- `Float64`: Quadratic PIJCV criterion value (lower is better)

# References
- Wood, S.N. (2024). "On Neighbourhood Cross Validation." arXiv:2404.16490v4, Section 4.1
"""
function compute_pijcv_criterion_fast(log_lambda::AbstractVector{T}, state::SmoothingSelectionState) where T<:Real
    lambda = exp.(log_lambda)
    
    # Build penalized Hessian
    H_lambda = _build_penalized_hessian(state.H_unpenalized, lambda, state.penalty_config; 
                                         beta=state.beta_hat)
    
    # Check if we're in AD mode
    use_cholesky_downdate = (T === Float64)
    
    # Try Cholesky factorization
    H_lambda_sym = Symmetric(H_lambda)
    chol_fact = if use_cholesky_downdate
        try
            cholesky(H_lambda_sym)
        catch e
            @debug "Cholesky failed in compute_pijcv_criterion_fast" lambda=lambda
            nothing
        end
    else
        nothing
    end
    
    if isnothing(chol_fact) && use_cholesky_downdate
        return T(1e10)
    end
    
    # Get subject-level likelihoods at current estimate (base term for V_q)
    ll_subj_base = loglik_exact(state.beta_hat, state.data; neg=false, return_ll_subj=true)
    
    # Compute V_q = Σᵢ [ -ℓᵢ(β̂) + gᵢᵀ Δ⁻ⁱ + ½ (Δ⁻ⁱ)ᵀ Hᵢ Δ⁻ⁱ ]
    V = zero(T)
    
    for i in 1:state.n_subjects
        g_i = @view state.subject_grads[:, i]
        H_i = state.subject_hessians[i]
        
        # Solve for Newton step: Δ⁻ⁱ = H_{λ,-i}⁻¹ gᵢ
        delta_i = if use_cholesky_downdate && chol_fact !== nothing
            _solve_loo_newton_step(chol_fact, H_i, g_i)
        else
            nothing
        end
        
        if isnothing(delta_i)
            H_lambda_loo = H_lambda - H_i
            H_loo_sym = Symmetric(H_lambda_loo)
            delta_i = try
                H_loo_sym \ collect(g_i)
            catch e
                @debug "Linear solve failed in compute_pijcv_criterion_fast" subject=i
                return T(1e10)
            end
        end
        
        # Quadratic approximation: D_i^q = -ℓᵢ + gᵢᵀ Δ + ½ Δᵀ Hᵢ Δ
        linear_term = dot(g_i, delta_i)
        quadratic_term = T(0.5) * dot(delta_i, H_i * delta_i)
        D_i = -ll_subj_base[i] + linear_term + quadratic_term
        
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
        @debug "Cholesky factorization failed in _ncv_criterion (k-fold): " exception=(e, catch_backtrace()) lambda=lambda nfolds=nfolds
        return T(1e10)
    end
    
    # Get subject-level likelihoods at current estimate (for V_q fallback)
    ll_subj_base = loglik_exact(state.beta_hat, state.data; neg=false, return_ll_subj=true)
    
    # PIJCV optimization: Build or reuse evaluation cache for efficient LOO evaluation
    eval_cache = if isnothing(state.pijcv_eval_cache)
        cache = build_pijcv_eval_cache(state.data)
        state.pijcv_eval_cache = cache
        cache
    else
        state.pijcv_eval_cache
    end
    
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
            @debug "Linear solve failed for fold-out Hessian in _ncv_criterion (k-fold): " exception=(e, catch_backtrace()) fold=k
            return T(1e10)
        end
        
        # Compute fold-out parameters: β̂⁻ᵏ = β̂ + Δ⁻ᵏ
        beta_fold_out = state.beta_hat .+ delta_k
        
        # Evaluate ACTUAL likelihood for each subject in this fold at fold-out parameters
        # Using cached structures for efficiency
        for i in fold_indices
            ll_fold = loglik_subject_cached(beta_fold_out, eval_cache, i)
            
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
- Use `select_lambda=:loocv` in `fit()` to select λ via exact LOOCV
- For approximate LOOCV that's O(n) instead of O(n²), use `select_lambda=:pijcv`

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

See also: [`compute_pijcv_criterion`](@ref), [`_select_hyperparameters`](@ref)
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

See also: [`compute_loocv_criterion`](@ref), [`_select_hyperparameters`](@ref)
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
        # Cholesky solve failed - expected for ill-conditioned problems
        @debug "Cholesky solve failed in _cholesky_downdate_solve: " exception=(e, catch_backtrace())
        return nothing
    end
end

"""
    _cholesky_downdate!(L::Matrix, v::Vector; tol=CHOLESKY_DOWNDATE_TOL) -> Bool

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
function _cholesky_downdate!(L::Matrix{Float64}, v::Vector{Float64}; tol::Float64=CHOLESKY_DOWNDATE_TOL)
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

