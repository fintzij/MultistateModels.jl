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
    lambda_init::Union{Nothing, Vector{Float64}} = nothing,  # Warm-start for λ (skips EFS)
    alpha_info::Union{Nothing, Dict{Int, AlphaLearningInfo}} = nothing,  # For joint α optimization
    alpha_groups::Union{Nothing, Vector{Vector{Int}}} = nothing,  # Groups of terms sharing α
    verbose::Bool = false
)
    # Check if joint (α, λ) optimization is needed
    do_joint_alpha = !isnothing(alpha_info) && !isempty(alpha_info) && 
                     !isnothing(alpha_groups) && !isempty(alpha_groups)
    
    if do_joint_alpha
        # Use joint optimization
        return _nested_optimization_pijcv_joint_alpha(
            model, data, penalty, selector;
            beta_init=beta_init,
            inner_maxiter=inner_maxiter,
            outer_maxiter=outer_maxiter,
            lambda_tol=lambda_tol,
            lambda_init=lambda_init,
            alpha_info=alpha_info,
            alpha_groups=alpha_groups,
            verbose=verbose
        )
    end
    
    # Standard λ-only optimization (no alpha learning)
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
    
    # PIJCV OPTIMIZATION: Build evaluation cache ONCE before optimization loop
    # This cache holds pre-computed SubjectCovarCache, covariate names, etc.
    # Without this, build_pijcv_eval_cache was called ~59 times (once per criterion eval)
    # causing massive memory allocations and 50-100x slowdown.
    pijcv_cache = if !selector.use_quadratic
        build_pijcv_eval_cache(data)
    else
        nothing  # Quadratic approximation V_q doesn't need the cache
    end
    
    # INNER FIT OPTIMIZATION: Build reusable optimization cache ONCE
    # This avoids ~1.5 GB allocation per inner fit call (59 calls = 90 GB!)
    # by reusing OptimizationFunction, OptimizationProblem, and AD infrastructure
    inner_opt_cache = build_inner_optimization_cache(model, data, penalty, beta_init; lb=lb, ub=ub)
    
    # Define the NCV criterion function with nested β optimization
    function ncv_criterion_with_nested_beta(log_lambda_vec, _)
        # Extract Float64 values for the inner optimization
        log_lambda_float = Float64[extract_value(x) for x in log_lambda_vec]
        lambda_vec_float = exp.(log_lambda_float)
        lambda_expanded = n_lambda == 1 ? fill(lambda_vec_float[1], n_hyperparameters(penalty_config)) : lambda_vec_float
        
        # Update penalty with current lambda for inner fit
        inner_penalty = set_hyperparameters(penalty, lambda_expanded)
        
        # Inner optimization: fit β̂(λ) using REUSABLE cache (avoids 1.5 GB alloc per call)
        beta_at_lambda = fit_inner_with_cache!(inner_opt_cache, inner_penalty, current_beta_ref[];
                                                maxiter=inner_maxiter)
        
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
        # Pass pre-built pijcv_cache to avoid rebuilding it 59x
        state = SmoothingSelectionState(
            copy(beta_at_lambda),
            H_unpenalized,
            subject_grads,
            subject_hessians,
            penalty_config,
            n_subjects,
            n_params,
            model,
            data,
            pijcv_cache  # Pre-built cache, or nothing for V_q mode
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
    
    # Adaptive bounds for log(λ) based on sample size
    log_lb_scalar, log_ub_scalar = compute_lambda_bounds(n_subjects, n_params)
    log_lb = fill(log_lb_scalar, n_lambda)
    log_ub = fill(log_ub_scalar, n_lambda)
    
    # Initialize λ: Use lambda_init if provided (warm-start), otherwise get EFS estimate
    # Warm-starting with previous λ skips the expensive EFS computation (~3x speedup in α learning)
    current_log_lambda = if !isnothing(lambda_init) && length(lambda_init) >= n_lambda
        # Use provided warm-start (e.g., from previous alpha iteration)
        if verbose
            println("  Using provided λ warm-start (skipping EFS)")
        end
        log.(lambda_init[1:n_lambda])
    else
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
        current_beta_ref[] = efs_result.warmstart_beta
        log.(efs_result.lambda[1:n_lambda])
    end
    
    # Use ForwardDiff for the outer λ optimization
    # This is the LEGACY NON-IMPLICIT version: each V(λ) evaluation requires solving the inner
    # β optimization, requiring nested AD. For better performance, use the implicit version
    # (_nested_optimization_pijcv_implicit) which uses ImplicitDifferentiation.jl.
    # NOTE: With use_implicit_diff=true (default), this code path is rarely taken.
    adtype = Optimization.AutoForwardDiff()
    optf = OptimizationFunction(ncv_criterion_with_nested_beta, adtype)
    prob = OptimizationProblem(optf, current_log_lambda, nothing; lb=log_lb, ub=log_ub)
    
    # Solve with Fminbox L-BFGS (quasi-Newton with ForwardDiff gradients)
    if verbose
        println("  Using L-BFGS outer optimizer (ForwardDiff, nested AD)...")
    end
    sol = solve(prob, OptimizationOptimJL.Fminbox(OptimizationOptimJL.LBFGS());
                maxiters=outer_maxiter,
                f_reltol=lambda_tol,
                x_abstol=lambda_tol)
    
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
    
    # Adaptive bounds for log(λ) based on sample size
    log_lb_scalar, log_ub_scalar = compute_lambda_bounds(n_subjects, n_params)
    log_lb = fill(log_lb_scalar, n_lambda)
    log_ub = fill(log_ub_scalar, n_lambda)
    current_log_lambda = zeros(n_lambda)
    
    # Use ForwardDiff for the outer λ optimization (nested AD through inner β fit)
    # EFS/PERF are primarily used for warm-starts, so some overhead is acceptable.
    # λ is low-dimensional (1-5 params), so ForwardDiff is efficient here.
    adtype = Optimization.AutoForwardDiff()
    optf = OptimizationFunction(criterion_with_nested_beta, adtype)
    prob = OptimizationProblem(optf, current_log_lambda, nothing; lb=log_lb, ub=log_ub)
    
    # Solve with Fminbox L-BFGS (quasi-Newton with ForwardDiff gradients)
    sol = solve(prob, OptimizationOptimJL.Fminbox(OptimizationOptimJL.LBFGS());
                maxiters=outer_maxiter,
                f_reltol=lambda_tol,
                x_abstol=lambda_tol)
    
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

