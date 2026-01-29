
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
    alpha_info::Union{Nothing, Dict{Int, AlphaLearningInfo}} = nothing,  # For joint α optimization
    alpha_groups::Union{Nothing, Vector{Vector{Int}}} = nothing,  # Groups of terms sharing α
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
        
        # Dispatch to implicit differentiation version if requested
        if selector.use_implicit_diff
            return _nested_optimization_pijcv_implicit(
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
        return _nested_optimization_pijcv(
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
    InnerOptimizationCache

Reusable optimization cache for inner coefficient fitting.

This avoids the 1.5 GB allocation overhead of recreating OptimizationFunction,
OptimizationProblem, and AD infrastructure on each of ~59 inner fits during
PIJCV λ selection.

Also stores pre-built SubjectCovarCache and covariate names to avoid
rebuilding them on every loglik_exact call (~2 MB per call at n=1000).

# Usage
```julia
# Build once before the loop
cache = build_inner_optimization_cache(model, data, penalty_config, beta_init; lb, ub)

# Reuse in the loop  
for λ in lambda_values
    penalty = set_hyperparameters(penalty_config, λ)
    beta = fit_inner_with_cache!(cache, penalty, current_beta; maxiter=50)
end
```
"""
struct InnerOptimizationCache
    opt_cache::Any  # Optimization.jl cache (type too complex to specify)
    penalty_ref::Ref{AbstractPenalty}  # Mutable reference to current penalty
    subject_covars::Vector{SubjectCovarCache}  # Pre-built subject covariate caches
    covar_names_per_hazard::Vector{Vector{Symbol}}  # Pre-computed covariate names
end

"""
    build_inner_optimization_cache(model, data, penalty, beta_init; lb, ub) -> InnerOptimizationCache

Build a reusable optimization cache for inner coefficient fitting.

This should be called ONCE before the nested optimization loop. The cache
is then reused via `fit_inner_with_cache!` for each λ trial.
"""
function build_inner_optimization_cache(
    model::MultistateProcess,
    data::ExactData,
    penalty::AbstractPenalty,
    beta_init::Vector{Float64};
    lb::Vector{Float64},
    ub::Vector{Float64}
)
    # Mutable reference to penalty (will be updated for each λ)
    penalty_ref = Ref{AbstractPenalty}(penalty)
    
    # Pre-build subject covariate caches (avoids ~2 MB allocation per loglik_exact call)
    subject_covars = build_subject_covar_cache(model)
    
    # Pre-compute covariate names per hazard
    hazards = model.hazards
    covar_names_per_hazard = [
        hasfield(typeof(hazards[h]), :covar_names) ? 
            hazards[h].covar_names : 
            extract_covar_names(hazards[h].parnames)
        for h in 1:length(hazards)
    ]
    
    # Define penalized negative log-likelihood objective
    # Captures penalty_ref and pre-built caches so we can update λ without rebuilding
    function penalized_nll(β, p)
        nll = loglik_exact(β, data; neg=true, 
                          subject_covars=subject_covars,
                          covar_names_per_hazard=covar_names_per_hazard)
        pen = compute_penalty(β, penalty_ref[])
        return nll + pen
    end
    
    # Set up optimization with second-order AD
    adtype = DifferentiationInterface.SecondOrder(
        Optimization.AutoForwardDiff(), 
        Optimization.AutoForwardDiff()
    )
    optf = OptimizationFunction(penalized_nll, adtype)
    prob = OptimizationProblem(optf, beta_init, nothing; lb=lb, ub=ub)
    
    # Initialize cache (this is the expensive part - only done once!)
    opt_cache = Optimization.init(prob, IpoptOptimizer(additional_options=Dict("sb" => "yes"));
                                   maxiters=50,
                                   tol=LAMBDA_SELECTION_INNER_TOL,
                                   print_level=0,
                                   honor_original_bounds="yes",
                                   bound_relax_factor=IPOPT_BOUND_RELAX_FACTOR,
                                   mu_strategy="adaptive")
    
    return InnerOptimizationCache(opt_cache, penalty_ref, subject_covars, covar_names_per_hazard)
end

"""
    fit_inner_with_cache!(cache::InnerOptimizationCache, penalty, beta_init; maxiter) -> Vector{Float64}

Fit coefficients using a pre-built optimization cache.

This avoids the ~1.5 GB allocation overhead of creating new OptimizationFunction
and OptimizationProblem on each call, plus ~2 MB per loglik_exact call from
SubjectCovarCache rebuilding.
"""
function fit_inner_with_cache!(
    cache::InnerOptimizationCache,
    penalty::AbstractPenalty,
    beta_init::Vector{Float64};
    maxiter::Int = 50
)
    # Update penalty (λ changes, but objective function structure is the same)
    cache.penalty_ref[] = penalty
    
    # Reinitialize with new starting point (avoids full cache rebuild)
    Optimization.reinit!(cache.opt_cache; u0=beta_init)
    
    # Solve using existing cache
    sol = Optimization.solve!(cache.opt_cache)
    
    return sol.u
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
- For repeated calls (e.g., PIJCV), use `build_inner_optimization_cache` + 
  `fit_inner_with_cache!` to avoid 1.5 GB allocation per call.
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
    
    # Set up optimization with second-order AD
    adtype = DifferentiationInterface.SecondOrder(
        Optimization.AutoForwardDiff(), 
        Optimization.AutoForwardDiff()
    )
    optf = OptimizationFunction(penalized_nll, adtype)
    prob = OptimizationProblem(optf, beta_init, nothing; lb=lb, ub=ub)
    
    # Solve with Ipopt (standard call for non-cached usage)
    # NOTE: For repeated calls (PIJCV), use build_inner_optimization_cache + 
    #       fit_inner_with_cache! which avoids 1.5 GB allocation per call
    sol = solve(prob, IpoptOptimizer(additional_options=Dict("sb" => "yes"));
                maxiters=maxiter,
                tol=LAMBDA_SELECTION_INNER_TOL,
                print_level=0,
                honor_original_bounds="yes",
                bound_relax_factor=IPOPT_BOUND_RELAX_FACTOR,
                mu_strategy="adaptive")
    
    return sol.u
end

