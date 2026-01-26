# =============================================================================
# Smoothing Parameter Selection for Penalized Splines
# =============================================================================
#
# Implements PIJCV (Predictive Infinitesimal Jackknife Cross-Validation) for 
# automatic selection of smoothing parameters λ in penalized spline models.
#
# Based on Wood (2024): "Neighbourhood Cross-Validation" arXiv:2404.16490
#
# Algorithm: Nested optimization of V(λ) where:
#   - Outer loop: optimize λ using Ipopt with ForwardDiff gradients
#   - Inner loop: for each trial λ, fit β̂(λ) via penalized MLE
#   - V(λ) is computed at the matched β̂(λ)
#
# GRADIENT APPROXIMATION NOTE:
# The gradient ∂V/∂λ computed by ForwardDiff is at FIXED β̂, ignoring the 
# implicit dependence ∂β̂/∂λ. Wood (2024) Section 2.2 shows the exact gradient
# requires: dβ̂/dρⱼ = -λⱼ H_λ⁻¹ Sⱼ β̂. This approximation works because:
# 1. At the optimum, ∂V/∂λ = 0, so the implicit term contribution is small
# 2. Ipopt is robust to approximate gradients
# 3. Function values V(λ) are exact (β̂(λ) is correctly matched)
#
# Key insight: Inner optimization uses PENALIZED loss, outer optimization
# minimizes UNPENALIZED prediction error via leave-one-out approximation.
#
# =============================================================================
# AD-SAFETY NOTES
# =============================================================================
#
# The functions in this module are designed for AD-compatibility where needed:
#
# AD-SAFE (can be differentiated through):
# - compute_penalty_from_lambda: Uses eltype(beta) for penalty accumulation
# - _fit_inner_coefficients: Uses ForwardDiff for optimization
# - pijcv_criterion: Core criterion is AD-safe (uses T = eltype)
#
# AD-UNSAFE (contain control flow/exceptions that break AD):
# - _select_hyperparameters: Outer loop dispatcher with convergence checks
# - _golden_section_search: Contains conditional logic
# - Catch blocks throughout: Return fallback values that preserve type T
#
# For catch blocks: When optimization fails during λ search, fallbacks use
# T(1e10) where T = eltype(parameters). This preserves AD type information
# even though gradients will be zero through the fallback path.
#
# =============================================================================

using LinearAlgebra

# =============================================================================
# Phase 3: New Dispatch-Based Selection Architecture
# =============================================================================
# 
# The new architecture separates concerns:
# - _select_hyperparameters: Dispatcher that returns HyperparameterSelectionResult
# - _nested_optimization_pijcv: Nested optimization (returns HyperparameterSelectionResult)
# - _fit_inner_coefficients: Inner loop coefficient fitting (returns beta vector)
#
# Selection functions return HyperparameterSelectionResult with warmstart_beta,
# NOT fitted models. The final fit ALWAYS happens in _fit_coefficients_at_fixed_hyperparameters.
# =============================================================================

"""
    _select_hyperparameters(model, data, penalty, selector; kwargs...) -> HyperparameterSelectionResult

Dispatch to the appropriate hyperparameter selection strategy.

This is the main entry point for smoothing parameter selection. It dispatches
based on the selector type and returns a `HyperparameterSelectionResult` containing
the optimal λ values and a warm-start point for the final fit.

# Key Architectural Property
This function returns `HyperparameterSelectionResult`, NOT a fitted model.
The `warmstart_beta` field in the result is used to warm-start the final fit,
which is always performed by `_fit_coefficients_at_fixed_hyperparameters`.

# Arguments
- `model::MultistateProcess`: Model for likelihood evaluation
- `data::ExactData`: Data container
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

# Dispatch Rules
- `NoSelection`: Returns immediately with default λ (no optimization)
- `PIJCVSelector(0)`: LOO Newton-approximated CV (Wood 2024 NCV)
- `PIJCVSelector(k)`: k-fold Newton-approximated CV
- `ExactCVSelector`: Exact k-fold or LOO CV (grid search)
- `REMLSelector`: REML/EFS criterion
- `PERFSelector`: PERF criterion

See also: [`HyperparameterSelectionResult`](@ref), [`_nested_optimization_pijcv`](@ref)
"""
function _select_hyperparameters(
    model::MultistateProcess,
    data::ExactData,
    penalty::AbstractPenalty,
    selector::AbstractHyperparameterSelector;
    beta_init::Vector{Float64},
    inner_maxiter::Int = 50,
    outer_maxiter::Int = 100,
    lambda_tol::Float64 = 1e-3,
    verbose::Bool = false
)
    # NoSelection: return default λ with no optimization
    if selector isa NoSelection
        lambda = get_hyperparameters(penalty)
        edf = compute_edf(beta_init, lambda, penalty, model, data)
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
        return _nested_optimization_pijcv(
            model, data, penalty, selector;
            beta_init=beta_init,
            inner_maxiter=inner_maxiter,
            outer_maxiter=outer_maxiter,
            lambda_tol=lambda_tol,
            verbose=verbose
        )
    end
    
    # ExactCVSelector: Grid search with exact refitting
    if selector isa ExactCVSelector
        method = selector.nfolds == 0 ? :loocv : Symbol("cv$(selector.nfolds)")
        return _grid_search_exact_cv(
            model, data, penalty, selector;
            beta_init=beta_init,
            inner_maxiter=inner_maxiter,
            verbose=verbose
        )
    end
    
    # REMLSelector: EFS/REML criterion
    if selector isa REMLSelector
        return _nested_optimization_reml(
            model, data, penalty;
            beta_init=beta_init,
            inner_maxiter=inner_maxiter,
            outer_maxiter=outer_maxiter,
            lambda_tol=lambda_tol,
            verbose=verbose
        )
    end
    
    # PERFSelector: PERF criterion
    if selector isa PERFSelector
        return _nested_optimization_perf(
            model, data, penalty;
            beta_init=beta_init,
            inner_maxiter=inner_maxiter,
            outer_maxiter=outer_maxiter,
            lambda_tol=lambda_tol,
            verbose=verbose
        )
    end
    
    # Unknown selector type
    throw(ArgumentError("Unknown selector type: $(typeof(selector))"))
end

"""
    _fit_inner_coefficients(model, data, penalty, beta_init; kwargs...) -> Vector{Float64}

Inner loop coefficient fitting at fixed hyperparameters.

This is used during nested optimization for hyperparameter selection. It fits
coefficients β at a fixed λ value, with relaxed convergence criteria compared
to the final fit (which uses `_fit_coefficients_at_fixed_hyperparameters`).

# Arguments
- `model::MultistateProcess`: Model for likelihood evaluation
- `data::ExactData`: Data container
- `penalty::AbstractPenalty`: Penalty configuration with current λ values
- `beta_init::Vector{Float64}`: Initial/warm-start coefficients

# Keyword Arguments
- `lb::Vector{Float64}`: Lower bounds on parameters
- `ub::Vector{Float64}`: Upper bounds on parameters
- `maxiter::Int=50`: Maximum iterations (fewer than final fit)

# Returns
- `Vector{Float64}`: Fitted coefficient vector

# Notes
- Uses Ipopt with ForwardDiff (HARD REQUIREMENT - no LBFGS, no finite differences)
- Relaxed convergence tolerance compared to final fit
- Returns just the coefficient vector, not an OptimizationSolution
"""
function _fit_inner_coefficients(
    model::MultistateProcess,
    data::ExactData,
    penalty::AbstractPenalty,
    beta_init::Vector{Float64};
    lb::Vector{Float64},
    ub::Vector{Float64},
    maxiter::Int = 50
)
    # Define penalized negative log-likelihood objective
    function penalized_nll(β, p)
        nll = loglik_exact(β, data; neg=true)
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

# =============================================================================
# MPanelData Dispatch: Markov Panel Data Hyperparameter Selection (Phase M2)
# =============================================================================

"""
    _select_hyperparameters(model, data::MPanelData, penalty, selector; kwargs...) -> HyperparameterSelectionResult

Dispatch hyperparameter selection for Markov panel data.

This method dispatches to the appropriate selection strategy based on the selector type,
using Markov-specific likelihood evaluation.

# Arguments
- `model::MultistateProcess`: Model for likelihood evaluation
- `data::MPanelData`: Markov panel data container
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

# Supported Selectors
- `NoSelection`: Returns immediately with default λ
- `PIJCVSelector(0)`: LOO Newton-approximated CV
- `PIJCVSelector(k)`: k-fold Newton-approximated CV  
- `REMLSelector`: REML/EFS criterion
- `PERFSelector`: PERF criterion

See also: [`_fit_inner_coefficients`](@ref), [`HyperparameterSelectionResult`](@ref)
"""
function _select_hyperparameters(
    model::MultistateProcess,
    data::MPanelData,
    penalty::AbstractPenalty,
    selector::AbstractHyperparameterSelector;
    beta_init::Vector{Float64},
    inner_maxiter::Int = 50,
    outer_maxiter::Int = 100,
    lambda_tol::Float64 = 1e-3,
    verbose::Bool = false
)
    # NoSelection: return default λ with no optimization
    if selector isa NoSelection
        lambda = get_hyperparameters(penalty)
        # Use books tuple for EDF computation
        edf = compute_edf_markov(beta_init, lambda, penalty, model, data.books)
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
        return _nested_optimization_pijcv_markov(
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
        return _nested_optimization_criterion_markov(
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
        return _nested_optimization_criterion_markov(
            model, data, penalty, :perf;
            beta_init=beta_init,
            inner_maxiter=inner_maxiter,
            outer_maxiter=outer_maxiter,
            lambda_tol=lambda_tol,
            verbose=verbose
        )
    end
    
    # ExactCVSelector not supported for Markov panel - requires refitting
    if selector isa ExactCVSelector
        throw(ArgumentError(
            "ExactCVSelector is not supported for Markov panel data. " *
            "Use PIJCVSelector for Newton-approximated CV or REMLSelector/PERFSelector for criteria-based selection."
        ))
    end
    
    # Unknown selector type
    throw(ArgumentError("Unknown selector type: $(typeof(selector))"))
end

"""
    _fit_inner_coefficients(model, data::MPanelData, penalty, beta_init; kwargs...) -> Vector{Float64}

Inner loop coefficient fitting at fixed hyperparameters for Markov panel data.

This is used during nested optimization for hyperparameter selection. It fits
coefficients β at a fixed λ value using Markov panel likelihood.

# Arguments
- `model::MultistateProcess`: Model for likelihood evaluation
- `data::MPanelData`: Markov panel data container
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
- Relaxed convergence tolerance compared to final fit
- Uses `loglik_markov` for likelihood evaluation
"""
function _fit_inner_coefficients(
    model::MultistateProcess,
    data::MPanelData,
    penalty::AbstractPenalty,
    beta_init::Vector{Float64};
    lb::Vector{Float64},
    ub::Vector{Float64},
    maxiter::Int = 50
)
    # Define penalized negative log-likelihood objective using Markov likelihood
    function penalized_nll(β, p)
        nll = _loglik_markov_mutating(β, data; neg=true, return_ll_subj=false)
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
    _nested_optimization_pijcv_markov(model, data::MPanelData, penalty, selector; kwargs...)

Nested optimization for PIJCV-based hyperparameter selection with Markov panel data.

Implements Wood (2024) NCV algorithm adapted for Markov models:
- OUTER LOOP: Ipopt minimizes V(log_λ)  
- INNER LOOP: For each trial λ, fit β̂(λ) via `_fit_inner_coefficients(model, data::MPanelData, ...)`

See [`_nested_optimization_pijcv`](@ref) for algorithm details.
"""
function _nested_optimization_pijcv_markov(
    model::MultistateProcess,
    data::MPanelData,
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
    n_subjects = length(model.subjectindices)
    n_params = length(beta_init)
    books = data.books
    
    # Determine method based on nfolds
    method = selector.nfolds == 0 ? :pijcv : Symbol("pijcv$(selector.nfolds)")
    
    if verbose
        println("Optimizing λ via nested optimization (Wood 2024 NCV) for Markov panel data")
        println("  Method: $method, n_lambda: $n_lambda")
    end
    
    # Track β̂ across evaluations for warm-starting
    current_beta_ref = Ref(copy(beta_init))
    n_criterion_evals = Ref(0)
    
    # Cache for state at current β̂ - updated when β changes
    current_state_ref = Ref{Union{Nothing, SmoothingSelectionStateMarkov}}(nothing)
    
    # Helper to extract Float64 from potentially nested Duals
    extract_value(x::Float64) = x
    extract_value(x::ForwardDiff.Dual) = extract_value(ForwardDiff.value(x))
    
    # Build penalty_config for use in inner loop
    penalty_config = penalty isa QuadraticPenalty ? penalty : build_penalty_config(model, SplinePenalty())
    
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
        
        # Compute subject gradients and Hessians at β̂(λ) for Markov panel data
        subject_grads_ll = compute_subject_gradients(beta_at_lambda, model, books)
        subject_hessians_ll = compute_subject_hessians(beta_at_lambda, model, books)
        
        # Convert to loss convention (negative log-likelihood)
        subject_grads = -subject_grads_ll
        subject_hessians = [-H for H in subject_hessians_ll]
        H_unpenalized = sum(subject_hessians)
        
        # Create state for criterion evaluation at β̂(λ)
        state = SmoothingSelectionStateMarkov(
            copy(beta_at_lambda),
            H_unpenalized,
            subject_grads,
            subject_hessians,
            penalty_config,
            n_subjects,
            n_params,
            model,
            data
        )
        current_state_ref[] = state
        
        # Compute criterion V(λ) at β̂(λ)
        V = if selector.nfolds == 0
            compute_pijcv_criterion_markov(log_lambda_vec, state)
        else
            compute_pijkfold_criterion_markov(log_lambda_vec, state, selector.nfolds)
        end
        
        if verbose && n_criterion_evals[] % 5 == 0
            V_float = extract_value(V)
            @info "Criterion eval $(n_criterion_evals[]): log(λ)=$(round.(log_lambda_float, digits=2)), V=$(round(V_float, digits=3))"
        end
        
        return V
    end
    
    # Bounds for log(λ) ∈ [-8, 8] corresponds to λ ∈ [0.00034, 2981]
    log_lb = fill(-8.0, n_lambda)
    log_ub = fill(8.0, n_lambda)
    current_log_lambda = zeros(n_lambda)  # Start at λ = 1
    
    # Use Ipopt with ForwardDiff for robust bounded optimization
    adtype = Optimization.AutoForwardDiff()
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
    edf = compute_edf_markov(current_beta, optimal_lambda_vec, penalty_config, model, books)
    
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
    SmoothingSelectionStateMarkov

Internal state for smoothing parameter selection via PIJCV/CV for Markov panel data.

Similar to `SmoothingSelectionState` but holds MPanelData instead of ExactData.
"""
mutable struct SmoothingSelectionStateMarkov
    beta_hat::Vector{Float64}
    H_unpenalized::Matrix{Float64}
    subject_grads::Matrix{Float64}
    subject_hessians::Vector{Matrix{Float64}}
    penalty_config::PenaltyConfig
    n_subjects::Int
    n_params::Int
    model::MultistateProcess
    data::MPanelData
end

"""
    compute_pijcv_criterion_markov(log_lambda, state::SmoothingSelectionStateMarkov) -> Float64

Compute the PIJCV/NCV criterion V(λ) for Markov panel data.

Same algorithm as `compute_pijcv_criterion` but uses `loglik_subject` dispatch
for MPanelData to evaluate subject-level likelihoods.
"""
function compute_pijcv_criterion_markov(log_lambda::AbstractVector{T}, state::SmoothingSelectionStateMarkov) where T<:Real
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
            @debug "Cholesky factorization failed in pijcv_criterion_markov: " exception=(e, catch_backtrace()) lambda=lambda
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
    
    # Get subject-level likelihoods at current estimate (for V_q fallback)
    ll_subj_base = loglik_markov(state.beta_hat, state.data; neg=false, return_ll_subj=true)
    
    # Compute NCV criterion: V = Σᵢ Dᵢ(β̂⁻ⁱ)
    V = zero(T)
    n_fallback = 0
    
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
                @debug "Linear solve failed for LOO Hessian in pijcv_criterion_markov: " exception=(e, catch_backtrace()) subject=i
                return T(1e10)
            end
        end
        
        # Compute LOO parameters: β̂⁻ⁱ = β̂ + Δ⁻ⁱ
        beta_loo = state.beta_hat .+ delta_i
        
        # Evaluate ACTUAL likelihood at LOO parameters
        ll_loo = loglik_subject(beta_loo, state.data, i)
        
        # Check if likelihood is finite
        if isfinite(ll_loo)
            D_i = -ll_loo
        else
            # Fallback to quadratic approximation
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
    compute_pijkfold_criterion_markov(log_lambda, state::SmoothingSelectionStateMarkov, nfolds::Int)

Compute k-fold PIJCV criterion for Markov panel data.
"""
function compute_pijkfold_criterion_markov(log_lambda::AbstractVector{T}, state::SmoothingSelectionStateMarkov, nfolds::Int) where T<:Real
    lambda = exp.(log_lambda)
    n_subjects = state.n_subjects
    
    # Build penalized Hessian
    H_lambda = _build_penalized_hessian(state.H_unpenalized, lambda, state.penalty_config;
                                         beta=state.beta_hat)
    
    # Get subject-level likelihoods at current estimate
    ll_subj_base = loglik_markov(state.beta_hat, state.data; neg=false, return_ll_subj=true)
    
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
            ll_loo = loglik_subject(beta_fold, state.data, i)
            
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
    _nested_optimization_criterion_markov(model, data::MPanelData, penalty, criterion_method; kwargs...)

Generic nested optimization for EFS/PERF criteria with Markov panel data.
"""
function _nested_optimization_criterion_markov(
    model::MultistateProcess,
    data::MPanelData,
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
    n_subjects = length(model.subjectindices)
    n_params = length(beta_init)
    books = data.books
    
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
        
        # Compute gradients and Hessians for Markov panel data
        subject_grads_ll = compute_subject_gradients(beta_at_lambda, model, books)
        subject_hessians_ll = compute_subject_hessians(beta_at_lambda, model, books)
        subject_grads = -subject_grads_ll
        subject_hessians = [-H for H in subject_hessians_ll]
        H_unpenalized = sum(subject_hessians)
        
        state = SmoothingSelectionStateMarkov(
            copy(beta_at_lambda), H_unpenalized, subject_grads, subject_hessians,
            penalty_config, n_subjects, n_params, model, data
        )
        
        # Compute criterion - reuse existing compute_efs_criterion and compute_perf_criterion
        # by creating a compatible ExactData-like wrapper
        V = criterion_method == :efs ? 
            compute_efs_criterion_markov(log_lambda_vec, state) :
            compute_perf_criterion_markov(log_lambda_vec, state)
        
        return V
    end
    
    log_lb = fill(-8.0, n_lambda)
    log_ub = fill(8.0, n_lambda)
    current_log_lambda = zeros(n_lambda)
    
    adtype = Optimization.AutoForwardDiff()
    optf = OptimizationFunction(criterion_with_nested_beta, adtype)
    prob = OptimizationProblem(optf, current_log_lambda, nothing; lb=log_lb, ub=log_ub)
    
    sol = solve(prob, OptimizationOptimJL.IPNewton();
                maxiters=outer_maxiter, abstol=lambda_tol, reltol=lambda_tol)
    
    optimal_log_lambda = sol.u
    optimal_lambda = exp.(optimal_log_lambda)
    optimal_lambda_vec = n_lambda == 1 ? fill(optimal_lambda[1], n_hyperparameters(penalty_config)) : optimal_lambda
    updated_penalty = set_hyperparameters(penalty, optimal_lambda_vec)
    converged = sol.retcode == ReturnCode.Success || sol.retcode == ReturnCode.MaxIters
    
    edf = compute_edf_markov(current_beta_ref[], optimal_lambda_vec, penalty_config, model, books)
    
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
    compute_efs_criterion_markov(log_lambda, state::SmoothingSelectionStateMarkov)

Compute EFS/REML criterion for Markov panel data.
Uses the same mathematical formulation as ExactData version.
"""
function compute_efs_criterion_markov(log_lambda::AbstractVector{T}, state::SmoothingSelectionStateMarkov) where T<:Real
    lambda = exp.(log_lambda)
    
    # Build penalized Hessian
    H_lambda = _build_penalized_hessian(state.H_unpenalized, lambda, state.penalty_config;
                                         beta=state.beta_hat)
    
    # Penalized negative log-likelihood at β̂
    ll = loglik_markov(state.beta_hat, state.data; neg=false)
    pen = compute_penalty_from_lambda(state.beta_hat, lambda, state.penalty_config)
    nll_penalized = -ll + pen
    
    # EFS criterion: -2*ll + log|H_λ|
    # This is an approximation to REML
    log_det_H = try
        logdet(Symmetric(H_lambda))
    catch
        return T(1e10)
    end
    
    return nll_penalized + T(0.5) * log_det_H
end

"""
    compute_perf_criterion_markov(log_lambda, state::SmoothingSelectionStateMarkov)

Compute PERF criterion for Markov panel data.
"""
function compute_perf_criterion_markov(log_lambda::AbstractVector{T}, state::SmoothingSelectionStateMarkov) where T<:Real
    lambda = exp.(log_lambda)
    
    # Build penalized Hessian
    H_lambda = _build_penalized_hessian(state.H_unpenalized, lambda, state.penalty_config;
                                         beta=state.beta_hat)
    
    # Penalized negative log-likelihood at β̂
    ll = loglik_markov(state.beta_hat, state.data; neg=false)
    pen = compute_penalty_from_lambda(state.beta_hat, lambda, state.penalty_config)
    nll_penalized = -ll + pen
    
    # PERF criterion: NLL + trace(H_unpen * H_lambda^{-1})
    H_inv = try
        inv(Symmetric(H_lambda))
    catch
        return T(1e10)
    end
    
    edf = tr(state.H_unpenalized * H_inv)
    
    return nll_penalized + edf
end

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
    verbose::Bool = false
)
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
    
    # Bounds for log(λ) ∈ [-8, 8] corresponds to λ ∈ [0.00034, 2981]
    log_lb = fill(-8.0, n_lambda)
    log_ub = fill(8.0, n_lambda)
    current_log_lambda = zeros(n_lambda)  # Start at λ = 1
    
    # Use Brent's method for 1D or NelderMead for multi-D (gradient-free due to MC noise)
    # But for consistency with Markov, use IPNewton with AutoForwardDiff
    adtype = Optimization.AutoForwardDiff()
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
    
    log_lb = fill(-8.0, n_lambda)
    log_ub = fill(8.0, n_lambda)
    current_log_lambda = zeros(n_lambda)
    
    adtype = Optimization.AutoForwardDiff()
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
    catch
        return T(1e10)
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
    catch
        return T(1e10)
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

# =============================================================================
# End MCEMSelectionData Selection Functions
# =============================================================================

"""
    _nested_optimization_pijcv(model, data, penalty, selector; kwargs...) -> HyperparameterSelectionResult

Nested optimization for PIJCV-based hyperparameter selection.

Implements the Wood (2024) NCV algorithm:
- OUTER LOOP: Ipopt minimizes V(log_λ)
- INNER LOOP: For each trial λ, fit β̂(λ) via _fit_inner_coefficients

# Key Properties
- Returns `HyperparameterSelectionResult` with warmstart_beta (NOT final fit)
- Uses Ipopt with ForwardDiff for both inner and outer optimization
- Gradient approximation: ∂V/∂λ computed at fixed β̂ (ignores implicit ∂β̂/∂λ)

# Arguments
- `model::MultistateProcess`: Model for likelihood evaluation
- `data::ExactData`: Data container
- `penalty::AbstractPenalty`: Penalty configuration
- `selector::PIJCVSelector`: PIJCV selector with nfolds specification

# Keyword Arguments
- `beta_init::Vector{Float64}`: Initial coefficient estimate
- `inner_maxiter::Int=50`: Maximum iterations for inner β fitting
- `outer_maxiter::Int=100`: Maximum iterations for outer λ optimization
- `lambda_tol::Float64=1e-3`: Convergence tolerance for λ
- `verbose::Bool=false`: Print progress

# Returns
- `HyperparameterSelectionResult`: Contains optimal λ, warmstart_beta, updated penalty

# References
- Wood, S.N. (2024). "On Neighbourhood Cross Validation." arXiv:2404.16490v4
"""
function _nested_optimization_pijcv(
    model::MultistateProcess,
    data::ExactData,
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
    samplepaths = data.paths
    
    # Determine method based on nfolds and use_quadratic
    method = if selector.use_quadratic
        selector.nfolds == 0 ? :pijlcv : Symbol("pijlcv$(selector.nfolds)")
    else
        selector.nfolds == 0 ? :pijcv : Symbol("pijcv$(selector.nfolds)")
    end
    
    if verbose
        println("Optimizing λ via nested optimization (Wood 2024 NCV)")
        println("  Method: $method, n_lambda: $n_lambda")
        selector.use_quadratic && println("  Using fast quadratic approximation V_q")
    end
    
    # Track β̂ across evaluations for warm-starting
    current_beta_ref = Ref(copy(beta_init))
    n_criterion_evals = Ref(0)
    
    # Cache for state at current β̂ - updated when β changes
    current_state_ref = Ref{Union{Nothing, SmoothingSelectionState}}(nothing)
    
    # Helper to extract Float64 from potentially nested Duals
    extract_value(x::Float64) = x
    extract_value(x::ForwardDiff.Dual) = extract_value(ForwardDiff.value(x))
    
    # Build penalty_config for use in inner loop
    # This is a PenaltyConfig (alias for QuadraticPenalty) which is compatible with existing code
    penalty_config = penalty isa QuadraticPenalty ? penalty : build_penalty_config(model, SplinePenalty())
    
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
            penalty_config,
            n_subjects,
            n_params,
            model,
            data
        )
        current_state_ref[] = state
        
        # Compute criterion V(λ) at β̂(λ)
        # If use_quadratic is true, use the fast V_q approximation
        V = if selector.use_quadratic
            compute_pijcv_criterion_fast(log_lambda_vec, state)
        elseif selector.nfolds == 0
            compute_pijcv_criterion(log_lambda_vec, state)
        else
            compute_pijkfold_criterion(log_lambda_vec, state, selector.nfolds)
        end
        
        if verbose && n_criterion_evals[] % 5 == 0
            V_float = extract_value(V)
            @info "Criterion eval $(n_criterion_evals[]): log(λ)=$(round.(log_lambda_float, digits=2)), V=$(round(V_float, digits=3))"
        end
        
        return V
    end
    
    # Bounds for log(λ) ∈ [-8, 8] corresponds to λ ∈ [0.00034, 2981]
    log_lb = fill(-8.0, n_lambda)
    log_ub = fill(8.0, n_lambda)
    
    # OPTIMIZATION: Get EFS estimate as warmstart for faster PIJCV convergence
    # EFS is ~6x faster and provides a good initial λ guess, reducing PIJCV iterations from ~70 to ~15
    if verbose
        println("  Getting EFS initial estimate for fast convergence...")
    end
    efs_result = _nested_optimization_reml(model, data, penalty;
                                           beta_init=beta_init,
                                           inner_maxiter=inner_maxiter,
                                           outer_maxiter=30,
                                           lambda_tol=0.1,
                                           verbose=false)
    efs_log_lambda = log.(efs_result.lambda[1:n_lambda])
    current_log_lambda = efs_log_lambda
    current_beta_ref[] = efs_result.warmstart_beta
    
    # Use ForwardDiff for robust bounded optimization
    adtype = Optimization.AutoForwardDiff()
    optf = OptimizationFunction(ncv_criterion_with_nested_beta, adtype)
    prob = OptimizationProblem(optf, current_log_lambda, nothing; lb=log_lb, ub=log_ub)
    
    # Solve with IPNewton (interior-point Newton with automatic Hessian via ForwardDiff)
    if verbose
        println("  Using IPNewton outer optimizer...")
    end
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
    edf = compute_edf(current_beta, optimal_lambda_vec, penalty_config, model, data)
    
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
    _grid_search_exact_cv(model, data, penalty, selector; kwargs...) -> HyperparameterSelectionResult

Grid search for exact CV methods that require refitting at each λ.
Used for :loocv, :cv5, :cv10, :cv20 methods.
"""
function _grid_search_exact_cv(
    model::MultistateProcess,
    data::ExactData,
    penalty::AbstractPenalty,
    selector::ExactCVSelector;
    beta_init::Vector{Float64},
    inner_maxiter::Int = 50,
    verbose::Bool = false
)
    lb, ub = model.bounds.lb, model.bounds.ub
    n_lambda = n_hyperparameters(penalty)
    nfolds = selector.nfolds == 0 ? length(data.paths) : selector.nfolds
    method = selector.nfolds == 0 ? :loocv : Symbol("cv$(selector.nfolds)")
    
    # Build penalty_config for compatibility with existing functions
    penalty_config = penalty isa QuadraticPenalty ? penalty : build_penalty_config(model, SplinePenalty())
    
    if verbose
        println("Using grid search for exact CV method: $method")
        println("  WARNING: This requires $(nfolds) refits per λ value")
    end
    
    # Coarse grid + refinement
    log_lambda_grid = collect(-4.0:1.0:4.0)  # 9 points
    
    best_lambda = Float64[]
    best_beta = copy(beta_init)
    best_criterion = Inf
    
    for log_lam in log_lambda_grid
        lam = exp(log_lam)
        lambda_vec = fill(lam, n_lambda)
        inner_penalty = set_hyperparameters(penalty, lambda_vec)
        
        beta_lam = _fit_inner_coefficients(model, data, inner_penalty, beta_init;
                                           lb=lb, ub=ub, maxiter=inner_maxiter)
        
        criterion = if nfolds == length(data.paths)
            compute_loocv_criterion(lambda_vec, beta_lam, model, data, penalty_config;
                                    maxiters=inner_maxiter, verbose=false)
        else
            compute_kfold_cv_criterion(lambda_vec, beta_lam, model, data, penalty_config, nfolds;
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
        inner_penalty = set_hyperparameters(penalty, lambda_vec)
        
        beta_lam = _fit_inner_coefficients(model, data, inner_penalty, best_beta;
                                           lb=lb, ub=ub, maxiter=inner_maxiter)
        
        criterion = if nfolds == length(data.paths)
            compute_loocv_criterion(lambda_vec, beta_lam, model, data, penalty_config;
                                    maxiters=inner_maxiter, verbose=false)
        else
            compute_kfold_cv_criterion(lambda_vec, beta_lam, model, data, penalty_config, nfolds;
                                       maxiters=inner_maxiter, verbose=false)
        end
        
        if criterion < best_criterion
            best_criterion = criterion
            best_lambda = lambda_vec
            best_beta = copy(beta_lam)
        end
    end
    
    # Update penalty with final lambda
    updated_penalty = set_hyperparameters(penalty, best_lambda)
    edf = compute_edf(best_beta, best_lambda, penalty_config, model, data)
    
    return HyperparameterSelectionResult(
        best_lambda,
        best_beta,
        updated_penalty,
        best_criterion,
        edf,
        true,  # Grid search always "converges"
        method,
        length(log_lambda_grid) + 5,  # n_iterations
        (;)
    )
end

"""
    _nested_optimization_reml(model, data, penalty; kwargs...) -> HyperparameterSelectionResult

Nested optimization using REML/EFS criterion.
"""
function _nested_optimization_reml(
    model::MultistateProcess,
    data::ExactData,
    penalty::AbstractPenalty;
    beta_init::Vector{Float64},
    inner_maxiter::Int = 50,
    outer_maxiter::Int = 100,
    lambda_tol::Float64 = 1e-3,
    verbose::Bool = false
)
    # Delegate to common nested optimization infrastructure with EFS criterion
    return _nested_optimization_criterion(
        model, data, penalty, :efs;
        beta_init=beta_init,
        inner_maxiter=inner_maxiter,
        outer_maxiter=outer_maxiter,
        lambda_tol=lambda_tol,
        verbose=verbose
    )
end

"""
    _nested_optimization_perf(model, data, penalty; kwargs...) -> HyperparameterSelectionResult

Nested optimization using PERF criterion.
"""
function _nested_optimization_perf(
    model::MultistateProcess,
    data::ExactData,
    penalty::AbstractPenalty;
    beta_init::Vector{Float64},
    inner_maxiter::Int = 50,
    outer_maxiter::Int = 100,
    lambda_tol::Float64 = 1e-3,
    verbose::Bool = false
)
    return _nested_optimization_criterion(
        model, data, penalty, :perf;
        beta_init=beta_init,
        inner_maxiter=inner_maxiter,
        outer_maxiter=outer_maxiter,
        lambda_tol=lambda_tol,
        verbose=verbose
    )
end

"""
    _nested_optimization_criterion(model, data, penalty, criterion_method; kwargs...)

Generic nested optimization for EFS/PERF criteria.
"""
function _nested_optimization_criterion(
    model::MultistateProcess,
    data::ExactData,
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
    samplepaths = data.paths
    
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
        
        # Compute gradients and Hessians in parallel
        subject_grads_ll, subject_hessians_ll = compute_subject_grads_and_hessians_fast(
            beta_at_lambda, model, samplepaths; use_threads=:auto)
        subject_grads = -subject_grads_ll
        subject_hessians = [-H for H in subject_hessians_ll]
        H_unpenalized = sum(subject_hessians)
        
        state = SmoothingSelectionState(
            copy(beta_at_lambda), H_unpenalized, subject_grads, subject_hessians,
            penalty_config, n_subjects, n_params, model, data
        )
        
        V = criterion_method == :efs ? 
            compute_efs_criterion(log_lambda_vec, state) :
            compute_perf_criterion(log_lambda_vec, state)
        
        return V
    end
    
    log_lb = fill(-8.0, n_lambda)
    log_ub = fill(8.0, n_lambda)
    current_log_lambda = zeros(n_lambda)
    
    adtype = Optimization.AutoForwardDiff()
    optf = OptimizationFunction(criterion_with_nested_beta, adtype)
    prob = OptimizationProblem(optf, current_log_lambda, nothing; lb=log_lb, ub=log_ub)
    
    sol = solve(prob, OptimizationOptimJL.IPNewton();
                maxiters=outer_maxiter, abstol=lambda_tol, reltol=lambda_tol)
    
    optimal_log_lambda = sol.u
    optimal_lambda = exp.(optimal_log_lambda)
    optimal_lambda_vec = n_lambda == 1 ? fill(optimal_lambda[1], n_hyperparameters(penalty_config)) : optimal_lambda
    updated_penalty = set_hyperparameters(penalty, optimal_lambda_vec)
    converged = sol.retcode == ReturnCode.Success || sol.retcode == ReturnCode.MaxIters
    
    edf = compute_edf(current_beta_ref[], optimal_lambda_vec, penalty_config, model, data)
    
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

# =============================================================================
# End of Phase 3 New Functions
# =============================================================================

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
- `pijcv_eval_cache::Union{Nothing, PIJCVEvaluationCache}`: Pre-built cache for efficient LOO evaluation
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
    pijcv_eval_cache::Union{Nothing, PIJCVEvaluationCache}  # PIJCV optimization: lazily built
end

# Constructor for backward compatibility (without cache)
function SmoothingSelectionState(beta_hat, H_unpenalized, subject_grads, subject_hessians,
                                  penalty_config, n_subjects, n_params, model, data)
    SmoothingSelectionState(beta_hat, H_unpenalized, subject_grads, subject_hessians,
                            penalty_config, n_subjects, n_params, model, data, nothing)
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
- This must match the behavior of `compute_penalty` in penalties.jl
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
                       lb=nothing, ub=nothing,
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
- `lb`, `ub`: Parameter bounds (optional; extracted from model if not provided)
- `maxiters`: Maximum optimizer iterations
- `use_polyalgorithm`: If true, use LBFGS→Ipopt; if false, pure Ipopt
- `verbose`: Print optimization progress

# Returns
Fitted coefficient vector β
"""
function fit_penalized_beta(model::MultistateProcess, data::ExactData,
                            lambda::Vector{Float64}, penalty_config::PenaltyConfig,
                            beta_init::Vector{Float64};
                            lb::Union{Nothing, Vector{Float64}}=nothing,
                            ub::Union{Nothing, Vector{Float64}}=nothing,
                            maxiters::Int=100, use_polyalgorithm::Bool=false,
                            verbose::Bool=false, ipopt_options...)
    
    # Extract bounds from model if not provided
    param_lb = isnothing(lb) ? model.bounds.lb : lb
    param_ub = isnothing(ub) ? model.bounds.ub : ub
    
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
    prob = OptimizationProblem(optf, beta_init, nothing; lb=param_lb, ub=param_ub)
    
    # Merge default options with user overrides
    # Convert kwargs to NamedTuple to enable merging
    ipopt_options_nt = (;ipopt_options...)
    merged_options = merge(DEFAULT_IPOPT_OPTIONS, (maxiters=maxiters, tol=LAMBDA_SELECTION_INNER_TOL), ipopt_options_nt)
    
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
