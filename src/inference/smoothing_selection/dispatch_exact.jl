
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
    lambda_init::Union{Nothing, Vector{Float64}} = nothing,  # Warm-start for λ (skips EFS)
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
            lambda_init=lambda_init,
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

