# =============================================================================
# End MPanelData Selection Functions
# =============================================================================

# =============================================================================
# MCEMSelectionData Dispatch: Semi-Markov MCEM Hyperparameter Selection (Phase M3)
# =============================================================================

"""
    _select_hyperparameters(model, data::MCEMSelectionData, penalty, selector; kwargs...) -> HyperparameterSelectionResult

Dispatch hyperparameter selection for MCEM within each iteration.

Within each MCEM iteration, paths/weights are FIXED. Q(β; paths, weights) is valid for 
ANY β via importance weighting. This allows jointly optimizing (λ, β) within each 
iteration using the same Monte Carlo approximation.

# Algorithm
1. For each trial λ: fit β̂(λ) via penalized M-step using SAME paths/weights
2. Compute importance-weighted gradients/Hessians at β̂(λ)
3. Evaluate PIJCV criterion using Newton-approximated LOO
4. Return optimal λ and warmstart β̂(λ_opt) for M-step

# Arguments
- `model::MultistateProcess`: Model for likelihood evaluation
- `data::MCEMSelectionData`: MCEM data with paths and importance weights
- `penalty::AbstractPenalty`: Penalty configuration
- `selector::AbstractHyperparameterSelector`: Selection strategy

# Keyword Arguments
- `beta_init::Vector{Float64}`: Initial coefficient estimate  
- `inner_maxiter::Int=50`: Maximum iterations for inner coefficient fitting
- `outer_maxiter::Int=100`: Maximum iterations for outer λ optimization
- `lambda_tol::Float64=1e-3`: Convergence tolerance for λ
- `verbose::Bool=false`: Print progress

# Returns
- `HyperparameterSelectionResult`: Contains lambda, warmstart_beta, penalty, etc.

# Notes
- MC variance in gradients/Hessians may affect selection stability
- Uses importance weights from E-step for all gradient/Hessian computations

See also: [`MCEMSelectionData`](@ref), [`_fit_inner_coefficients`](@ref)
"""
function _select_hyperparameters(
    model::MultistateProcess,
    data::MCEMSelectionData,
    penalty::AbstractPenalty,
    selector::AbstractHyperparameterSelector;
    beta_init::Vector{Float64},
    inner_maxiter::Int = 50,
    outer_maxiter::Int = 100,
    lambda_tol::Float64 = 1e-3,
    alpha_info::Union{Nothing, Dict{Int, AlphaLearningInfo}} = nothing,  # For joint α optimization
    alpha_groups::Union{Nothing, Vector{Vector{Int}}} = nothing,  # Groups of terms sharing α
    verbose::Bool = false
)
    # Joint alpha optimization for MCEM data - TODO: implement JointAlphaLambdaStateMCEM
    if !isnothing(alpha_info) && !isempty(alpha_info)
        @warn "Joint (α, λ) optimization not yet implemented for MCEM data. Using λ-only selection."
    end
    
    # NoSelection: return default λ with no optimization
    if selector isa NoSelection
        lambda = get_hyperparameters(penalty)
        edf_scalar = compute_edf_mcem(beta_init, lambda, penalty, data)
        edf = (total = edf_scalar, per_term = [edf_scalar])
        return HyperparameterSelectionResult(
            lambda,
            beta_init,
            penalty,
            NaN,  # No criterion value for NoSelection
            edf,
            true,  # converged
            :none,
            0,
            (;)  # empty diagnostics
        )
    end
    
    # PIJCVSelector: Newton-approximated CV (Wood 2024 NCV)
    if selector isa PIJCVSelector
        method = selector.nfolds == 0 ? :pijcv : Symbol("pijcv$(selector.nfolds)")
        
        # Dispatch to implicit differentiation version if requested
        if selector.use_implicit_diff
            return _nested_optimization_pijcv_mcem_implicit(
                model, data, penalty, selector;
                beta_init=beta_init,
                inner_maxiter=inner_maxiter,
                outer_maxiter=outer_maxiter,
                lambda_tol=lambda_tol,
                verbose=verbose
            )
        end
        
        # Default: legacy nested AD
        return _nested_optimization_pijcv_mcem(
            model, data, penalty, selector;
            beta_init=beta_init,
            inner_maxiter=inner_maxiter,
            outer_maxiter=outer_maxiter,
            lambda_tol=lambda_tol,
            verbose=verbose
        )
    end
    
    # REMLSelector: EFS/REML criterion
    if selector isa REMLSelector
        return _nested_optimization_criterion_mcem(
            model, data, penalty, :efs;
            beta_init=beta_init,
            inner_maxiter=inner_maxiter,
            outer_maxiter=outer_maxiter,
            lambda_tol=lambda_tol,
            verbose=verbose
        )
    end
    
    # PERFSelector: PERF criterion
    if selector isa PERFSelector
        return _nested_optimization_criterion_mcem(
            model, data, penalty, :perf;
            beta_init=beta_init,
            inner_maxiter=inner_maxiter,
            outer_maxiter=outer_maxiter,
            lambda_tol=lambda_tol,
            verbose=verbose
        )
    end
    
    # ExactCVSelector not supported for MCEM - too expensive with MC variance
    if selector isa ExactCVSelector
        throw(ArgumentError(
            "ExactCVSelector is not supported for MCEM. " *
            "Use PIJCVSelector for Newton-approximated CV or REMLSelector/PERFSelector."
        ))
    end
    
    # Unknown selector type
    throw(ArgumentError("Unknown selector type: $(typeof(selector))"))
end

"""
    _fit_inner_coefficients(model, data::MCEMSelectionData, penalty, beta_init; kwargs...) -> Vector{Float64}

Inner loop coefficient fitting at fixed hyperparameters for MCEM.

Fits β at fixed λ using the importance-weighted complete-data log-likelihood:
```math
Q(β; paths, weights) = Σᵢ Σⱼ wᵢⱼ log f(Zᵢⱼ; β)
```

# Arguments
- `model::MultistateProcess`: Model for likelihood evaluation
- `data::MCEMSelectionData`: MCEM data with paths and importance weights
- `penalty::AbstractPenalty`: Penalty configuration with current λ values
- `beta_init::Vector{Float64}`: Initial/warm-start coefficients

# Keyword Arguments
- `lb::Vector{Float64}`: Lower bounds on parameters
- `ub::Vector{Float64}`: Upper bounds on parameters
- `maxiter::Int=50`: Maximum iterations (fewer than final fit)

# Returns
- `Vector{Float64}`: Fitted coefficient vector

# Notes
- Uses Ipopt with ForwardDiff (HARD REQUIREMENT)
- Uses `loglik_semi_markov` for importance-weighted likelihood evaluation
"""
function _fit_inner_coefficients(
    model::MultistateProcess,
    data::MCEMSelectionData,
    penalty::AbstractPenalty,
    beta_init::Vector{Float64};
    lb::Vector{Float64},
    ub::Vector{Float64},
    maxiter::Int = 50
)
    # Create SMPanelData for likelihood computation
    sm_data = SMPanelData(data.model, data.paths, data.weights)
    
    # Define penalized negative log-likelihood objective using semi-Markov likelihood
    function penalized_nll(β, p)
        nll = loglik_semi_markov(β, sm_data; neg=true, use_sampling_weight=true)
        pen = compute_penalty(β, penalty)
        return nll + pen
    end
    
    # Set up optimization with second-order AD (required for Ipopt)
    adtype = DifferentiationInterface.SecondOrder(
        Optimization.AutoForwardDiff(), 
        Optimization.AutoForwardDiff()
    )
    optf = OptimizationFunction(penalized_nll, adtype)
    prob = OptimizationProblem(optf, beta_init, nothing; lb=lb, ub=ub)
    
    # Solve with Ipopt using relaxed tolerance for inner loop
    sol = solve(prob, IpoptOptimizer(additional_options=Dict("sb" => "yes"));
                maxiters=maxiter,
                tol=LAMBDA_SELECTION_INNER_TOL,
                print_level=0,
                honor_original_bounds="yes",
                bound_relax_factor=IPOPT_BOUND_RELAX_FACTOR,
                mu_strategy="adaptive")
    
    return sol.u
end

"""
    SmoothingSelectionStateMCEM

Internal state for smoothing parameter selection via PIJCV/CV for MCEM.

Similar to `SmoothingSelectionState` but holds MCEMSelectionData and
uses importance-weighted gradients/Hessians.
"""
mutable struct SmoothingSelectionStateMCEM
    beta_hat::Vector{Float64}
    H_unpenalized::Matrix{Float64}
    subject_grads::Matrix{Float64}
    subject_hessians::Vector{Matrix{Float64}}
    penalty_config::PenaltyConfig
    n_subjects::Int
    n_params::Int
    model::MultistateProcess
    data::MCEMSelectionData
end

"""
    _nested_optimization_pijcv_mcem(model, data::MCEMSelectionData, penalty, selector; kwargs...)

Nested optimization for PIJCV-based hyperparameter selection within MCEM.

Implements Wood (2024) NCV algorithm adapted for importance-weighted likelihoods:
- OUTER LOOP: Optimize V(log_λ) using gradient-free search (Brent's method)
- INNER LOOP: For each trial λ, fit β̂(λ) via `_fit_inner_coefficients(model, data::MCEMSelectionData, ...)`

# Key Difference from Exact/Markov
- Gradients and Hessians are importance-weighted averages over sampled paths
- Monte Carlo variance in these quantities may affect λ selection stability
- Uses Brent's method (gradient-free) for outer loop due to MC noise

See also: [`_nested_optimization_pijcv`](@ref), [`_nested_optimization_pijcv_markov`](@ref)
"""
function _nested_optimization_pijcv_mcem(
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
    n_subjects = length(data.paths)
    n_params = length(beta_init)
    
    # Determine method based on nfolds
    method = selector.nfolds == 0 ? :pijcv : Symbol("pijcv$(selector.nfolds)")
    
    if verbose
        println("Optimizing λ via nested optimization (Wood 2024 NCV) for MCEM")
        println("  Method: $method, n_lambda: $n_lambda")
    end
    
    # Track β̂ across evaluations for warm-starting
    current_beta_ref = Ref(copy(beta_init))
    n_criterion_evals = Ref(0)
    
    # Build penalty_config
    penalty_config = penalty isa QuadraticPenalty ? penalty : build_penalty_config(model, SplinePenalty())
    
    # Helper to extract Float64 from potentially nested Duals
    extract_value(x::Float64) = x
    extract_value(x::ForwardDiff.Dual) = extract_value(ForwardDiff.value(x))
    
    # Define the NCV criterion function with nested β optimization
    function ncv_criterion_with_nested_beta(log_lambda_vec, _)
        # Extract Float64 values for the inner optimization
        log_lambda_float = Float64[extract_value(x) for x in log_lambda_vec]
        lambda_vec_float = exp.(log_lambda_float)
        lambda_expanded = n_lambda == 1 ? fill(lambda_vec_float[1], n_hyperparameters(penalty_config)) : lambda_vec_float
        
        # Update penalty with current lambda for inner fit
        inner_penalty = set_hyperparameters(penalty, lambda_expanded)
        
        # Inner optimization: fit β̂(λ) at this trial λ, warm-started from previous β̂
        beta_at_lambda = _fit_inner_coefficients(model, data, inner_penalty, current_beta_ref[];
                                                  lb=lb, ub=ub, maxiter=inner_maxiter)
        
        # Update warm-start for next evaluation
        current_beta_ref[] = beta_at_lambda
        n_criterion_evals[] += 1
        
        # Compute importance-weighted subject gradients and Hessians at β̂(λ)
        subject_grads_ll = compute_subject_gradients(beta_at_lambda, data.model, data.paths, data.weights)
        subject_hessians_ll = compute_subject_hessians(beta_at_lambda, data.model, data.paths, data.weights)
        
        # Convert to loss convention (negative log-likelihood)
        subject_grads = -subject_grads_ll
        subject_hessians = [-H for H in subject_hessians_ll]
        H_unpenalized = sum(subject_hessians)
        
        # Create state for criterion evaluation at β̂(λ)
        state = SmoothingSelectionStateMCEM(
            copy(beta_at_lambda),
            H_unpenalized,
            subject_grads,
            subject_hessians,
            penalty_config,
            n_subjects,
            n_params,
            data.model,
            data
        )
        
        # Compute criterion V(λ) at β̂(λ)
        V = if selector.nfolds == 0
            compute_pijcv_criterion_mcem(log_lambda_vec, state)
        else
            compute_pijkfold_criterion_mcem(log_lambda_vec, state, selector.nfolds)
        end
        
        if verbose && n_criterion_evals[] % 5 == 0
            V_float = extract_value(V)
            @info "Criterion eval $(n_criterion_evals[]): log(λ)=$(round.(log_lambda_float, digits=2)), V=$(round(V_float, digits=3))"
        end
        
        return V
    end
    
    # Adaptive bounds for log(λ) based on sample size
    log_lb_scalar, log_ub_scalar = compute_lambda_bounds(n_subjects, n_params)
    log_lb = fill(log_lb_scalar, n_lambda)
    log_ub = fill(log_ub_scalar, n_lambda)
    current_log_lambda = zeros(n_lambda)  # Start at λ = 1
    
    # Use Brent's method for 1D or NelderMead for multi-D (gradient-free due to MC noise)
    # But for consistency with Markov, use IPNewton with SecondOrder ForwardDiff
    adtype = DifferentiationInterface.SecondOrder(
        Optimization.AutoForwardDiff(), 
        Optimization.AutoForwardDiff()
    )
    optf = OptimizationFunction(ncv_criterion_with_nested_beta, adtype)
    prob = OptimizationProblem(optf, current_log_lambda, nothing; lb=log_lb, ub=log_ub)
    
    # Solve with IPNewton from OptimizationOptimJL
    sol = solve(prob, OptimizationOptimJL.IPNewton();
                maxiters=outer_maxiter,
                abstol=lambda_tol,
                reltol=lambda_tol)
    
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
    edf_scalar = compute_edf_mcem(current_beta, optimal_lambda_vec, penalty_config, data)
    # Wrap in NamedTuple to match expected format for HyperparameterSelectionResult
    edf = (total = edf_scalar, per_term = [edf_scalar])
    
    return HyperparameterSelectionResult(
        optimal_lambda_vec,
        current_beta,       # warmstart_beta for final fit
        updated_penalty,
        best_criterion,
        edf,
        converged,
        method,
        n_criterion_evals[],
        (log_lambda = optimal_log_lambda, retcode = sol.retcode)
    )
end

"""
    compute_pijcv_criterion_mcem(log_lambda, state::SmoothingSelectionStateMCEM) -> Float64

Compute the PIJCV/NCV criterion V(λ) for MCEM.

Uses importance-weighted gradients and Hessians. LOO likelihood is evaluated
using the importance-weighted likelihood at perturbed parameters.
"""
function compute_pijcv_criterion_mcem(log_lambda::AbstractVector{T}, state::SmoothingSelectionStateMCEM) where T<:Real
    lambda = exp.(log_lambda)
    
    # Build penalized Hessian
    H_lambda = _build_penalized_hessian(state.H_unpenalized, lambda, state.penalty_config; 
                                         beta=state.beta_hat)
    
    # Check if we're in AD mode (Dual numbers)
    use_cholesky_downdate = (T === Float64)
    
    # Try Cholesky factorization of full penalized Hessian
    H_lambda_sym = Symmetric(H_lambda)
    chol_fact = if use_cholesky_downdate
        try
            cholesky(H_lambda_sym)
        catch e
            @debug "Cholesky factorization failed in pijcv_criterion_mcem: " exception=(e, catch_backtrace()) lambda=lambda
            nothing
        end
    else
        nothing
    end
    
    # If Cholesky failed and not in AD mode, return large value
    if isnothing(chol_fact) && use_cholesky_downdate
        @debug "Returning large criterion value due to Cholesky failure"
        return T(1e10)
    end
    
    # Create SMPanelData for likelihood evaluation
    sm_data = SMPanelData(state.data.model, state.data.paths, state.data.weights)
    
    # Get importance-weighted subject likelihoods at current estimate
    ll_subj_base = loglik_semi_markov_subject(state.beta_hat, sm_data)
    
    # Compute NCV criterion: V = Σᵢ Dᵢ(β̂⁻ⁱ)
    V = zero(T)
    
    for i in 1:state.n_subjects
        # Get subject i's gradient and Hessian (of negative log-likelihood)
        g_i = @view state.subject_grads[:, i]
        H_i = state.subject_hessians[i]
        
        # Solve for coefficient perturbation: Δ⁻ⁱ = H_{λ,-i}⁻¹ gᵢ
        delta_i = if use_cholesky_downdate && chol_fact !== nothing
            _solve_loo_newton_step(chol_fact, H_i, g_i)
        else
            nothing
        end
        
        if isnothing(delta_i)
            # Direct solve: (H_λ - H_i)⁻¹ g_i
            H_lambda_loo = H_lambda - H_i
            H_loo_sym = Symmetric(H_lambda_loo)
            
            delta_i = try
                H_loo_sym \ collect(g_i)
            catch e
                @debug "Linear solve failed for LOO Hessian in pijcv_criterion_mcem: " exception=(e, catch_backtrace()) subject=i
                return T(1e10)
            end
        end
        
        # Compute LOO parameters: β̂⁻ⁱ = β̂ + Δ⁻ⁱ
        beta_loo = state.beta_hat .+ delta_i
        
        # Evaluate ACTUAL importance-weighted likelihood for subject i at LOO parameters
        ll_loo = loglik_semi_markov_subject_i(beta_loo, sm_data, i)
        
        # Check if likelihood is finite
        if isfinite(ll_loo)
            D_i = -ll_loo
        else
            # Fallback to quadratic approximation
            linear_term = dot(g_i, delta_i)
            quadratic_term = T(0.5) * dot(delta_i, H_i * delta_i)
            D_i = -ll_subj_base[i] + linear_term + quadratic_term
        end
        
        V += D_i
    end
    
    return V
end

"""
    compute_pijkfold_criterion_mcem(log_lambda, state::SmoothingSelectionStateMCEM, nfolds::Int)

Compute k-fold PIJCV criterion for MCEM.
"""
function compute_pijkfold_criterion_mcem(log_lambda::AbstractVector{T}, state::SmoothingSelectionStateMCEM, nfolds::Int) where T<:Real
    lambda = exp.(log_lambda)
    n_subjects = state.n_subjects
    
    # Build penalized Hessian
    H_lambda = _build_penalized_hessian(state.H_unpenalized, lambda, state.penalty_config;
                                         beta=state.beta_hat)
    
    # Create SMPanelData for likelihood evaluation
    sm_data = SMPanelData(state.data.model, state.data.paths, state.data.weights)
    
    # Get importance-weighted subject likelihoods at current estimate
    ll_subj_base = loglik_semi_markov_subject(state.beta_hat, sm_data)
    
    # Create fold assignments (balanced)
    fold_assignments = mod1.(1:n_subjects, nfolds)
    
    V = zero(T)
    
    for k in 1:nfolds
        # Get indices for subjects in fold k
        fold_mask = fold_assignments .== k
        fold_indices = findall(fold_mask)
        
        # Sum gradients and Hessians for fold k
        g_fold = sum(state.subject_grads[:, i] for i in fold_indices)
        H_fold = sum(state.subject_hessians[i] for i in fold_indices)
        
        # Compute fold-out Hessian and solve
        H_lambda_fold = H_lambda - H_fold
        H_fold_sym = Symmetric(H_lambda_fold)
        
        delta_fold = try
            H_fold_sym \ g_fold
        catch e
            @debug "Linear solve failed for fold-out Hessian" fold=k
            return T(1e10)
        end
        
        # LOO parameters for fold k
        beta_fold = state.beta_hat .+ delta_fold
        
        # Evaluate fold subjects at fold-out parameters
        for i in fold_indices
            ll_loo = loglik_semi_markov_subject_i(beta_fold, sm_data, i)
            
            if isfinite(ll_loo)
                V += -ll_loo
            else
                # Fallback to quadratic approximation
                g_i = @view state.subject_grads[:, i]
                H_i = state.subject_hessians[i]
                linear_term = dot(g_i, delta_fold)
                quadratic_term = T(0.5) * dot(delta_fold, H_i * delta_fold)
                V += -ll_subj_base[i] + linear_term + quadratic_term
            end
        end
    end
    
    return V
end

"""
    _nested_optimization_criterion_mcem(model, data::MCEMSelectionData, penalty, criterion_method; kwargs...)

Generic nested optimization for EFS/PERF criteria with MCEM data.
"""
function _nested_optimization_criterion_mcem(
    model::MultistateProcess,
    data::MCEMSelectionData,
    penalty::AbstractPenalty,
    criterion_method::Symbol;
    beta_init::Vector{Float64},
    inner_maxiter::Int = 50,
    outer_maxiter::Int = 100,
    lambda_tol::Float64 = 1e-3,
    verbose::Bool = false
)
    lb, ub = model.bounds.lb, model.bounds.ub
    n_lambda = n_hyperparameters(penalty)
    n_subjects = length(data.paths)
    n_params = length(beta_init)
    
    penalty_config = penalty isa QuadraticPenalty ? penalty : build_penalty_config(model, SplinePenalty())
    
    current_beta_ref = Ref(copy(beta_init))
    n_criterion_evals = Ref(0)
    
    extract_value(x::Float64) = x
    extract_value(x::ForwardDiff.Dual) = extract_value(ForwardDiff.value(x))
    
    function criterion_with_nested_beta(log_lambda_vec, _)
        log_lambda_float = Float64[extract_value(x) for x in log_lambda_vec]
        lambda_vec_float = exp.(log_lambda_float)
        lambda_expanded = n_lambda == 1 ? fill(lambda_vec_float[1], n_hyperparameters(penalty_config)) : lambda_vec_float
        
        inner_penalty = set_hyperparameters(penalty, lambda_expanded)
        beta_at_lambda = _fit_inner_coefficients(model, data, inner_penalty, current_beta_ref[];
                                                  lb=lb, ub=ub, maxiter=inner_maxiter)
        
        current_beta_ref[] = beta_at_lambda
        n_criterion_evals[] += 1
        
        # Compute importance-weighted gradients and Hessians
        subject_grads_ll = compute_subject_gradients(beta_at_lambda, data.model, data.paths, data.weights)
        subject_hessians_ll = compute_subject_hessians(beta_at_lambda, data.model, data.paths, data.weights)
        subject_grads = -subject_grads_ll
        subject_hessians = [-H for H in subject_hessians_ll]
        H_unpenalized = sum(subject_hessians)
        
        state = SmoothingSelectionStateMCEM(
            copy(beta_at_lambda), H_unpenalized, subject_grads, subject_hessians,
            penalty_config, n_subjects, n_params, data.model, data
        )
        
        V = criterion_method == :efs ? 
            compute_efs_criterion_mcem(log_lambda_vec, state) :
            compute_perf_criterion_mcem(log_lambda_vec, state)
        
        return V
    end
    
    # Adaptive bounds for log(λ) based on sample size
    log_lb_scalar, log_ub_scalar = compute_lambda_bounds(n_subjects, n_params)
    log_lb = fill(log_lb_scalar, n_lambda)
    log_ub = fill(log_ub_scalar, n_lambda)
    current_log_lambda = zeros(n_lambda)
    
    adtype = DifferentiationInterface.SecondOrder(
        Optimization.AutoForwardDiff(), 
        Optimization.AutoForwardDiff()
    )
    optf = OptimizationFunction(criterion_with_nested_beta, adtype)
    prob = OptimizationProblem(optf, current_log_lambda, nothing; lb=log_lb, ub=log_ub)
    
    sol = solve(prob, OptimizationOptimJL.IPNewton();
                maxiters=outer_maxiter, abstol=lambda_tol, reltol=lambda_tol)
    
    optimal_log_lambda = sol.u
    optimal_lambda = exp.(optimal_log_lambda)
    optimal_lambda_vec = n_lambda == 1 ? fill(optimal_lambda[1], n_hyperparameters(penalty_config)) : optimal_lambda
    updated_penalty = set_hyperparameters(penalty, optimal_lambda_vec)
    converged = sol.retcode == ReturnCode.Success || sol.retcode == ReturnCode.MaxIters
    
    edf_scalar = compute_edf_mcem(current_beta_ref[], optimal_lambda_vec, penalty_config, data)
    edf = (total = edf_scalar, per_term = [edf_scalar])
    
    return HyperparameterSelectionResult(
        optimal_lambda_vec,
        current_beta_ref[],
        updated_penalty,
        sol.objective,
        edf,
        converged,
        criterion_method,
        n_criterion_evals[],
        (log_lambda = optimal_log_lambda, retcode = sol.retcode)
    )
end

"""
    compute_efs_criterion_mcem(log_lambda, state::SmoothingSelectionStateMCEM)

Compute EFS/REML criterion for MCEM data.
"""
function compute_efs_criterion_mcem(log_lambda::AbstractVector{T}, state::SmoothingSelectionStateMCEM) where T<:Real
    lambda = exp.(log_lambda)
    
    H_lambda = _build_penalized_hessian(state.H_unpenalized, lambda, state.penalty_config;
                                         beta=state.beta_hat)
    
    sm_data = SMPanelData(state.data.model, state.data.paths, state.data.weights)
    ll = loglik_semi_markov(state.beta_hat, sm_data; neg=false, use_sampling_weight=true)
    pen = compute_penalty_from_lambda(state.beta_hat, lambda, state.penalty_config)
    nll_penalized = -ll + pen
    
    log_det_H = try
        logdet(Symmetric(H_lambda))
    catch e
        @debug "logdet(H_lambda) failed in compute_efs_criterion_mcem (matrix may not be PD)" exception=(e, catch_backtrace()) lambda=lambda
        return T(CRITERION_FAILURE_VALUE)
    end
    
    return nll_penalized + T(0.5) * log_det_H
end

"""
    compute_perf_criterion_mcem(log_lambda, state::SmoothingSelectionStateMCEM)

Compute PERF criterion for MCEM data.
"""
function compute_perf_criterion_mcem(log_lambda::AbstractVector{T}, state::SmoothingSelectionStateMCEM) where T<:Real
    lambda = exp.(log_lambda)
    
    H_lambda = _build_penalized_hessian(state.H_unpenalized, lambda, state.penalty_config;
                                         beta=state.beta_hat)
    
    sm_data = SMPanelData(state.data.model, state.data.paths, state.data.weights)
    ll = loglik_semi_markov(state.beta_hat, sm_data; neg=false, use_sampling_weight=true)
    pen = compute_penalty_from_lambda(state.beta_hat, lambda, state.penalty_config)
    nll_penalized = -ll + pen
    
    H_inv = try
        inv(Symmetric(H_lambda))
    catch e
        @debug "Matrix inversion failed in compute_perf_criterion_mcem" exception=(e, catch_backtrace()) lambda=lambda
        return T(CRITERION_FAILURE_VALUE)
    end
    
    edf = tr(state.H_unpenalized * H_inv)
    
    return nll_penalized + edf
end

"""
    compute_edf_mcem(beta, lambda, penalty_config, data::MCEMSelectionData)

Compute effective degrees of freedom for MCEM at given (β, λ).

For MCEM, EDF is computed from the importance-weighted Hessian.
"""
function compute_edf_mcem(beta::Vector{Float64}, lambda::Vector{Float64}, 
                          penalty_config::AbstractPenalty, data::MCEMSelectionData)
    # Compute importance-weighted Hessian
    subject_hessians_ll = compute_subject_hessians(beta, data.model, data.paths, data.weights)
    subject_hessians = [-H for H in subject_hessians_ll]
    H_unpenalized = sum(subject_hessians)
    
    # Build penalized Hessian
    H_lambda = _build_penalized_hessian(H_unpenalized, lambda, penalty_config; beta=beta)
    
    # EDF = trace(H_unpenalized * H_lambda^{-1})
    H_inv = try
        inv(Symmetric(H_lambda))
    catch
        return NaN
    end
    
    return tr(H_unpenalized * H_inv)
end

"""
    loglik_semi_markov_subject(params, sm_data::SMPanelData) -> Vector{Float64}

Compute importance-weighted log-likelihood for each subject.

Returns a vector of length n_subjects, where element i is:
    Σⱼ wᵢⱼ log f(Zᵢⱼ; params)
"""
function loglik_semi_markov_subject(params::AbstractVector, sm_data::SMPanelData)
    n_subjects = length(sm_data.paths)
    ll_subjects = Vector{Float64}(undef, n_subjects)
    
    for i in 1:n_subjects
        ll_subjects[i] = loglik_semi_markov_subject_i(params, sm_data, i)
    end
    
    return ll_subjects
end

"""
    loglik_semi_markov_subject_i(params, sm_data::SMPanelData, i::Int) -> Float64

Compute importance-weighted log-likelihood for subject i.

Returns: Σⱼ wᵢⱼ log f(Zᵢⱼ; params) for subject i
"""
function loglik_semi_markov_subject_i(params::AbstractVector{T}, sm_data::SMPanelData, i::Int) where T
    model = sm_data.model
    paths = sm_data.paths[i]
    weights = sm_data.ImportanceWeights[i]
    npaths = length(paths)
    
    # Unflatten parameters
    pars_nested = unflatten_parameters(params, model)
    hazards = model.hazards
    totalhazards = model.totalhazards
    tmat = model.tmat
    samplingweight = model.SubjectWeights[i]
    
    ll_weighted = zero(T)
    
    for j in 1:npaths
        path = paths[j]
        subj_inds = model.subjectindices[path.subj]
        subj_dat = view(model.data, subj_inds, :)
        subjdat_df = make_subjdat(path, subj_dat)
        ll_j = loglik_path(pars_nested, subjdat_df, hazards, totalhazards, tmat) * samplingweight
        ll_weighted += weights[j] * ll_j
    end
    
    return ll_weighted
end

