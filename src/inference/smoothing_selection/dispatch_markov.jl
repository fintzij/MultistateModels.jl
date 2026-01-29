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
    lambda_init::Union{Nothing, Vector{Float64}} = nothing,  # Warm-start for λ
    alpha_info::Union{Nothing, Dict{Int, AlphaLearningInfo}} = nothing,  # For joint α optimization
    alpha_groups::Union{Nothing, Vector{Vector{Int}}} = nothing,  # Groups of terms sharing α
    verbose::Bool = false
)
    # Joint alpha optimization for Markov data - TODO: implement JointAlphaLambdaStateMarkov
    if !isnothing(alpha_info) && !isempty(alpha_info)
        @warn "Joint (α, λ) optimization not yet implemented for Markov panel data. Using λ-only selection."
    end
    
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
        
        # Dispatch to implicit differentiation version if requested
        if selector.use_implicit_diff
            return _nested_optimization_pijcv_markov_implicit(
                model, data, penalty, selector;
                beta_init=beta_init,
                inner_maxiter=inner_maxiter,
                outer_maxiter=outer_maxiter,
                lambda_tol=lambda_tol,
                lambda_init=lambda_init,
                verbose=verbose
            )
        end
        
        # Default: legacy nested AD
        return _nested_optimization_pijcv_markov(
            model, data, penalty, selector;
            beta_init=beta_init,
            inner_maxiter=inner_maxiter,
            outer_maxiter=outer_maxiter,
            lambda_tol=lambda_tol,
            lambda_init=lambda_init,
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
    lambda_init::Union{Nothing, Vector{Float64}} = nothing,  # Warm-start for λ
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
    
    # Adaptive bounds for log(λ) based on sample size
    log_lb_scalar, log_ub_scalar = compute_lambda_bounds(n_subjects, n_params)
    log_lb = fill(log_lb_scalar, n_lambda)
    log_ub = fill(log_ub_scalar, n_lambda)
    
    # Initialize λ: Use lambda_init if provided (warm-start), otherwise start at λ = 1
    # Warm-starting with previous λ is crucial for α learning efficiency
    current_log_lambda = if !isnothing(lambda_init) && length(lambda_init) >= n_lambda
        if verbose
            println("  Using provided λ warm-start")
        end
        log.(lambda_init[1:n_lambda])
    else
        zeros(n_lambda)  # Start at λ = 1
    end
    
    # Use Ipopt with SecondOrder ForwardDiff for robust bounded optimization
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
    catch e
        @debug "logdet(H_lambda) failed in compute_efs_criterion_markov (matrix may not be PD)" exception=(e, catch_backtrace()) lambda=lambda
        return T(CRITERION_FAILURE_VALUE)
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
    catch e
        @debug "Matrix inversion failed in compute_perf_criterion_markov" exception=(e, catch_backtrace()) lambda=lambda
        return T(CRITERION_FAILURE_VALUE)
    end
    
    edf = tr(state.H_unpenalized * H_inv)
    
    return nll_penalized + edf
end

