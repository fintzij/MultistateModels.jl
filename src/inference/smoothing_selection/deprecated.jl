# =============================================================================
# DEPRECATED: Old select_smoothing_parameters functions
# =============================================================================
# These functions are no longer used by the main fitting path (Phase 4 refactoring).
# Smoothing parameter selection now happens through fit() -> _fit_exact ->
# _fit_exact_penalized -> _select_hyperparameters.
#
# Kept for backward compatibility but will be removed in a future version.
# =============================================================================

"""
    select_smoothing_parameters(model::MultistateModel, penalty::SplinePenalty; kwargs...)

!!! warning "Deprecated"
    This function is deprecated and will be removed in a future version.
    Smoothing parameter selection is now integrated into `fit()`:
    ```julia
    # Old way (deprecated):
    result = select_smoothing_parameters(model, SplinePenalty(); method=:pijcv)
    
    # New way:
    fitted = fit(model; penalty=SplinePenalty(), select_lambda=:pijcv)
    ```

User-friendly interface for smoothing parameter selection using performance iteration.
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
    Base.depwarn(
        "select_smoothing_parameters is deprecated. " *
        "Use fit(model; penalty=SplinePenalty(), select_lambda=:pijcv) instead.",
        :select_smoothing_parameters
    )
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
    
    # Extract bounds from model
    lb, ub = model.bounds.lb, model.bounds.ub
    
    # Define unpenalized likelihood function
    loglik_fn = loglik_exact
    adtype = DifferentiationInterface.SecondOrder(
        Optimization.AutoForwardDiff(), 
        Optimization.AutoForwardDiff()
    )
    optf = OptimizationFunction(loglik_fn, adtype)
    prob = OptimizationProblem(optf, parameters, data; lb=lb, ub=ub)
    
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
    select_smoothing_parameters(model::MultistateProcess, data::ExactData, ...; kwargs...)

!!! warning "Deprecated"
    This function is deprecated and will be removed in a future version.
    Smoothing parameter selection is now integrated into `fit()`:
    ```julia
    fitted = fit(model; penalty=SplinePenalty(), select_lambda=:pijcv)
    ```

Internal version of smoothing parameter selection.
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
    # Note: No deprecation warning here since this is called by the other overload
    # which already emits the warning
    
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
        println("Optimizing λ via nested optimization (Wood 2024 NCV)")
        println("  Method: $method")
        println("  n_lambda: $n_lambda")
    end
    
    # Warn user about multi-λ shared optimization
    if n_lambda > 1
        @warn """Multiple smoothing parameters detected (n_lambda=$n_lambda).
        Current implementation uses shared λ for all smooth terms.
        Consider calibrating each term separately with `scope=:baseline` or `scope=:covariates`."""
    end
    
    # =========================================================================
    # NESTED OPTIMIZATION (Wood 2024, Section 3)
    # =========================================================================
    # "In practice this will involve nested optimization. An outer optimizer 
    # seeks the best ρ according to (2), with each trial ρ in turn requiring 
    # an inner optimization to obtain the corresponding β̂."
    #
    # OUTER LOOP: Ipopt with ForwardDiff minimizes V(log_λ)
    #   - V is PIJCV, PERF, or EFS criterion
    #   - ForwardDiff computes ∂V/∂λ at FIXED β̂ (approximate gradient)
    #   - This ignores the implicit derivative dβ̂/dλ from Wood (2024, Section 2.2)
    #   - Acceptable because function values V(λ) are exact (β̂(λ) is properly fitted)
    #
    # INNER LOOP: For each trial λ proposed by outer optimizer:
    #   1. Fit β̂(λ) by maximizing penalized log-likelihood: ℓ(β) - (λ/2)β'Sβ
    #   2. Compute V(λ) at the fitted β̂(λ)
    #
    # Key insight: "Such nested strategies are not as computationally costly
    # as they naively appear, because the previous iterate's β̂ value serves
    # as an ever better starting value for the inner optimization as the 
    # outer optimization converges."
    # =========================================================================
    
    # Track β̂ across evaluations for warm-starting
    # Use Ref to allow mutation inside the criterion closure
    current_beta_ref = Ref(copy(beta_init))
    n_criterion_evals = Ref(0)
    
    # Cache for state at current β̂ - updated when β changes
    # This allows AD to differentiate V(λ) at fixed β̂
    current_state_ref = Ref{Union{Nothing, SmoothingSelectionState}}(nothing)
    
    # Helper to extract base Float64 value from potentially nested Duals
    extract_value(x::Float64) = x
    extract_value(x::ForwardDiff.Dual) = extract_value(ForwardDiff.value(x))
    
    # Define the NCV criterion function with nested β optimization
    # Structure: For each trial λ, fit β̂(λ), then compute V(λ) at that β̂
    # AD differentiates through V(λ) at fixed β̂ (ignoring ∂β̂/∂λ)
    function ncv_criterion_with_nested_beta(log_lambda_vec, _)
        # Extract Float64 values for the inner optimization
        # Inner optimization doesn't need AD - we're fitting β̂(λ) numerically
        # Handle potentially nested Duals from Hessian computation
        log_lambda_float = Float64[extract_value(x) for x in log_lambda_vec]
        lambda_vec_float = exp.(log_lambda_float)
        lambda_expanded = n_lambda == 1 ? fill(lambda_vec_float[1], scoped_config.n_lambda) : lambda_vec_float
        
        # Inner optimization: fit β̂(λ) at this trial λ, warm-started from previous β̂
        beta_at_lambda = fit_penalized_beta(model, data, lambda_expanded, scoped_config, 
                                            current_beta_ref[];
                                            maxiters=inner_maxiter,
                                            use_polyalgorithm=false,
                                            verbose=false)
        
        # Update warm-start for next evaluation
        current_beta_ref[] = beta_at_lambda
        n_criterion_evals[] += 1
        
        # Compute subject gradients and Hessians at β̂(λ) in parallel (Float64 computation)
        subject_grads_ll, subject_hessians_ll = compute_subject_grads_and_hessians_fast(
            beta_at_lambda, model, samplepaths; use_threads=:auto)
        
        # Convert to loss convention (negative log-likelihood)
        subject_grads = -subject_grads_ll
        subject_hessians = [-H for H in subject_hessians_ll]
        H_unpenalized = sum(subject_hessians)
        
        # Create state for criterion evaluation at β̂(λ)
        state = SmoothingSelectionState(
            copy(beta_at_lambda),
            H_unpenalized,
            subject_grads,
            subject_hessians,
            scoped_config,
            n_subjects,
            n_params,
            model,
            data
        )
        current_state_ref[] = state
        
        # Compute criterion V(λ) at β̂(λ)
        # Pass the original log_lambda_vec (may have Dual type) for AD
        V = if method == :pijcv
            compute_pijcv_criterion(log_lambda_vec, state)
        elseif method == :pijlcv
            # Fast quadratic approximation (V_q)
            compute_pijcv_criterion_fast(log_lambda_vec, state)
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
        
        if verbose && n_criterion_evals[] % 5 == 0
            V_float = extract_value(V)
            @info "Criterion eval $(n_criterion_evals[]): log(λ)=$(round.(log_lambda_float, digits=2)), V=$(round(V_float, digits=3))"
        end
        
        return V
    end
    
    # Bounds for log(λ) ∈ [-8, 8] corresponds to λ ∈ [0.00034, 2981]
    lb = fill(-8.0, n_lambda)
    ub = fill(8.0, n_lambda)
    current_log_lambda = zeros(n_lambda)  # Start at λ = 1
    
    # Use Ipopt with ForwardDiff for robust bounded optimization
    adtype = Optimization.AutoForwardDiff()
    optf = OptimizationFunction(ncv_criterion_with_nested_beta, adtype)
    prob = OptimizationProblem(optf, current_log_lambda, nothing; lb=lb, ub=ub)
    
    # Solve with Ipopt - interior point method with AD gradients
    sol = solve(prob, OptimizationOptimJL.IPNewton();
                maxiters=lambda_maxiter * max_outer_iter,
                abstol=lambda_tol,
                reltol=lambda_tol)
    
    new_log_lambda = sol.u
    best_criterion = sol.objective
    current_beta = current_beta_ref[]
    converged = sol.retcode == ReturnCode.Success || sol.retcode == ReturnCode.MaxIters
    
    if verbose
        lambda_val = exp(new_log_lambda[1])
        println("  Final: log(λ)=$(round(new_log_lambda[1], digits=2)), λ=$(round(lambda_val, sigdigits=3)), V=$(round(best_criterion, digits=3))")
        println("  Criterion evaluations: $(n_criterion_evals[])")
        if converged
            println("  Converged successfully")
        else
            println("  Warning: Optimizer returned $(sol.retcode)")
        end
    end
    
    # Final results
    best_lambda = exp.(new_log_lambda)
    best_lambda_vec = n_lambda == 1 ? fill(best_lambda[1], scoped_config.n_lambda) : best_lambda
    
    if verbose && !converged
        println("  Warning: Did not converge after $max_outer_iter iterations")
    end
    
    if verbose
        println("\n  Optimal log(λ): $(round.(new_log_lambda, digits=3))")
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
        n_outer_iter = n_criterion_evals[]
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
        @debug "Eigendecomposition failed in _perf_criterion: " exception=(e, catch_backtrace()) lambda=lambda
        return T(1e10)  # Return large value if eigen fails
    end
    
    # Check for positive definiteness
    if any(eig.values .< EIGENVALUE_ZERO_TOL)
        # H is not positive definite - regularize
        min_eval = minimum(eig.values)
        regularization = abs(min_eval) + MATRIX_REGULARIZATION_EPS
        H_reg = H + regularization * I(p)
        eig = eigen(Symmetric(H_reg))
    end
    
    # H^{1/2} = V * D^{1/2} * V'
    sqrt_evals = sqrt.(max.(eig.values, EIGENVALUE_ZERO_TOL))
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
        @debug "Matrix inversion failed in _perf_criterion: " exception=(e, catch_backtrace()) lambda=lambda
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
    # Use regularized logdet to avoid eigen() which is not AD-compatible with ForwardDiff.
    # For d-th order difference penalty, null space has dimension d.
    # Pseudo-logdet: log|S_λ|₊ ≈ log|S_λ + εI| - null_dim * log(ε)
    null_dim = sum(term.order for term in state.penalty_config.terms; init=0)
    eps_reg = T(EIGENVALUE_ZERO_TOL)
    
    log_det_S = try
        S_lambda_reg = Symmetric(S_lambda + eps_reg * I)
        T(0.5) * (logdet(S_lambda_reg) - null_dim * log(eps_reg))
    catch e
        @debug "logdet(S_lambda) failed in compute_efs_criterion" exception=(e, catch_backtrace())
        T(0.0)  # S_λ is effectively zero
    end
    
    # Log determinant of penalized Hessian: -½ log|H + S_λ|
    # Use logdet() which is AD-compatible via Cholesky factorization
    H_lambda_sym = Symmetric(H_lambda)
    log_det_H_lambda = try
        T(-0.5) * logdet(H_lambda_sym)
    catch e
        # logdet fails if matrix is not positive definite (Cholesky fails)
        @debug "logdet(H_lambda) failed in compute_efs_criterion (matrix may not be PD)" exception=(e, catch_backtrace())
        return T(1e10)
    end
    
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
