# =============================================================================
# Penalized Likelihood Fitting Functions
# =============================================================================
#
# This file contains functions for penalized maximum likelihood estimation
# with optional hyperparameter (smoothing parameter) selection.
#
# Key architectural principle: Selection functions return HyperparameterSelectionResult
# with warmstart_beta, NOT fitted models. The final fit ALWAYS happens in
# _fit_coefficients_at_fixed_hyperparameters.
#
# Contents:
# - _fit_exact_penalized: Main penalized fitting function with optional selection
# - _fit_coefficients_at_fixed_hyperparameters: Final optimization at fixed λ
# - Alpha learning iteration for adaptive penalty weighting
#
# All optimizations use Ipopt with ForwardDiff (no LBFGS, no finite differences).
#
# =============================================================================

"""
    _fit_exact_penalized(model, data, samplepaths, penalty, selector; kwargs...)

Penalized maximum likelihood fitting with optional hyperparameter selection.

This function handles all penalized fitting paths:
1. Fixed λ (selector = NoSelection): Direct optimization at provided λ
2. Selection-based λ (selector = PIJCVSelector, etc.): 
   - First select optimal λ via nested optimization
   - Then perform final fit at selected λ
3. Alpha learning (when `penalty_specs` contains `AtRiskWeighting(learn=true)`):
   - Alternates between fitting β/λ and updating α until convergence

# Key Architectural Decision
Selection functions return `HyperparameterSelectionResult` with a `warmstart_beta`
field that is used to warm-start the final fit. The final fit ALWAYS happens
via `_fit_coefficients_at_fixed_hyperparameters` - selection functions do NOT
return the final fitted model.

# Arguments
- `model::MultistateModel`: Model to fit
- `data::ExactData`: Data container
- `samplepaths::Vector{SamplePath}`: Sample paths from data
- `penalty::AbstractPenalty`: Penalty configuration (must NOT be NoPenalty)
- `selector::AbstractHyperparameterSelector`: Selection method

# Keyword Arguments
- `constraints`: Parameter constraints (not supported with selection)
- `verbose::Bool=true`: Print progress messages
- `solver`: Optimization solver (default: Ipopt)
- `adtype=:auto`: AD backend for gradients/Hessians
- `parallel::Bool=false`: Enable parallel likelihood evaluation
- `nthreads::Union{Nothing,Int}=nothing`: Number of threads
- `vcov_type::Symbol=:ij`: Variance-covariance type
- `vcov_threshold::Bool=true`: Use adaptive tolerance for pseudo-inverse
- `loo_method::Symbol=:direct`: LOO perturbation method
- `inner_maxiter::Int=100`: Maximum iterations for inner coefficient fitting
- `lambda_init::Float64=1.0`: Initial λ for selection (if selector != NoSelection)
- `penalty_specs::Union{Nothing, SplinePenalty, Vector{SplinePenalty}}=nothing`: 
  Original penalty specification (for alpha learning)
- `alpha_maxiter::Int=3`: Maximum iterations for alpha learning (coarse tolerance is usually sufficient)
- `alpha_tol::Float64=0.05`: Convergence tolerance for alpha changes (α rarely needs > 1 decimal precision)

# Returns
- `MultistateModelFitted`: Fitted model with coefficients, variance, and λ selection info
"""
function _fit_exact_penalized(
    model::MultistateModel,
    data::ExactData,
    samplepaths::Vector{SamplePath},
    penalty::AbstractPenalty,
    selector::AbstractHyperparameterSelector;
    constraints = nothing,
    verbose::Bool = true,
    solver = nothing,
    adtype = :auto,
    parallel::Bool = false,
    nthreads = nothing,
    vcov_type::Symbol = :ij,
    vcov_threshold::Bool = true,
    loo_method::Symbol = :direct,
    inner_maxiter::Int = 100,
    lambda_init::Float64 = 1.0,
    penalty_specs::Union{Nothing, SplinePenalty, Vector{SplinePenalty}} = nothing,
    alpha_maxiter::Int = 3,  # Coarse tolerance is usually sufficient
    alpha_tol::Float64 = 0.05,  # α rarely needs > 1 decimal precision
    kwargs...
)
    # =========================================================================
    # Validate inputs
    # =========================================================================
    @assert !(penalty isa NoPenalty) "NoPenalty should not reach _fit_exact_penalized"
    
    # Constraints with selection is not supported
    if !isnothing(constraints) && !(selector isa NoSelection)
        throw(ArgumentError(
            "Constrained optimization with smoothing parameter selection is not supported. " *
            "Either use constraints with select_lambda=:none (fixed λ), " *
            "or remove constraints for automatic λ selection."
        ))
    end
    
    # =========================================================================
    # Get initial parameters and bounds
    # =========================================================================
    parameters = get_parameters_flat(model)
    lb, ub = model.bounds.lb, model.bounds.ub
    
    # Configure threading
    n_paths = length(samplepaths)
    threading_config = ThreadingConfig(parallel=parallel, nthreads=nthreads)
    use_parallel = should_parallelize(threading_config, n_paths)
    
    if use_parallel && verbose
        println("Using $(threading_config.nthreads) threads for likelihood evaluation ($(n_paths) paths)")
    end
    
    # =========================================================================
    # Check for alpha learning
    # =========================================================================
    do_alpha_learning = !isnothing(penalty_specs) && needs_alpha_learning(penalty_specs)
    alpha_info = if do_alpha_learning
        collect_alpha_learning_info(model, penalty, penalty_specs)
    else
        Dict{Int, AlphaLearningInfo}()
    end
    alpha_groups = do_alpha_learning ? get_shared_alpha_groups(penalty, alpha_info) : Vector{Vector{Int}}()
    
    # Initialize current alphas
    current_alphas = Dict{Int, Float64}()
    if do_alpha_learning
        for (term_idx, info) in alpha_info
            # Get initial alpha from specs
            specs = penalty_specs isa SplinePenalty ? [penalty_specs] : penalty_specs
            for spec in specs
                if spec.weighting isa AtRiskWeighting
                    current_alphas[term_idx] = spec.weighting.alpha
                    break
                end
            end
        end
    end
    
    # =========================================================================
    # Hyperparameter selection (if selector is not NoSelection)
    # =========================================================================
    smoothing_result = nothing
    final_penalty = penalty
    warmstart_beta = parameters  # Default: start from model parameters
    
    if !(selector isa NoSelection)
        if verbose
            println("Selecting smoothing parameter(s) via $(typeof(selector))...")
        end
        
        # Call the dispatcher - returns HyperparameterSelectionResult, NOT a fitted model
        smoothing_result = _select_hyperparameters(
            model, data, final_penalty, selector;
            beta_init=parameters,
            inner_maxiter=inner_maxiter,
            verbose=verbose
        )
        
        if verbose
            println("Selected λ: $(round.(smoothing_result.lambda, sigdigits=4))")
            println("Effective degrees of freedom: $(round(smoothing_result.edf.total, digits=2))")
        end
        
        # Update penalty with selected λ and get warmstart
        final_penalty = smoothing_result.penalty
        warmstart_beta = smoothing_result.warmstart_beta
    end
    
    # =========================================================================
    # Alpha learning iteration (if needed)
    # =========================================================================
    # When learn_alpha=true in SplinePenalty, we alternate between:
    # 1. Fitting β and selecting λ at current α
    # 2. Updating α via marginal likelihood maximization
    # This typically converges in 3-5 iterations.
    
    if do_alpha_learning && !isempty(alpha_info)
        if verbose
            println("Learning adaptive penalty weighting parameters (α)...")
        end
        
        for alpha_iter in 1:alpha_maxiter
            alpha_changed = false
            max_alpha_change = 0.0
            
            # Fit beta at current penalty (with current alpha values)
            if alpha_iter > 1
                # Re-fit beta at updated penalty
                warmstart_beta = _fit_inner_coefficients(
                    model, data, final_penalty, warmstart_beta;
                    lb=lb, ub=ub, maxiter=inner_maxiter
                )
            end
            
            # Update alpha for each group
            for group in alpha_groups
                # Use first term's info for shared groups
                term_idx = group[1]
                info = alpha_info[term_idx]
                old_alpha = current_alphas[term_idx]
                
                # Learn optimal alpha via marginal likelihood
                new_alpha = learn_alpha(
                    model, data, final_penalty, warmstart_beta,
                    term_idx, info.hazard, info.atrisk;
                    alpha_bounds=(0.0, 2.0), tol=alpha_tol
                )
                
                # Update penalty matrix for all terms in group
                for idx in group
                    term_info = alpha_info[idx]
                    final_penalty = update_penalty_with_alpha(
                        final_penalty, model, idx, new_alpha,
                        term_info.hazard, term_info.atrisk
                    )
                    current_alphas[idx] = new_alpha
                end
                
                # Track convergence
                alpha_change = abs(new_alpha - old_alpha)
                max_alpha_change = max(max_alpha_change, alpha_change)
                
                if alpha_change > alpha_tol
                    alpha_changed = true
                end
                
                if verbose
                    terms_str = length(group) > 1 ? "terms $(group)" : "term $(term_idx)"
                    println("  Iter $alpha_iter: α=$( round(new_alpha, digits=4)) for $terms_str (Δα=$(round(alpha_change, digits=4)))")
                end
            end
            
            # Check convergence
            if !alpha_changed
                if verbose
                    println("Alpha learning converged after $alpha_iter iteration(s)")
                end
                break
            end
            
            # Re-select lambda with updated penalty if not NoSelection
            # OPTIMIZATION: Warm-start λ from previous iteration (skips EFS, ~3x faster)
            # Also use reduced outer_maxiter since λ typically changes little between α iterations
            if !(selector isa NoSelection) && alpha_iter < alpha_maxiter
                previous_lambda = get_hyperparameters(final_penalty)
                smoothing_result = _select_hyperparameters(
                    model, data, final_penalty, selector;
                    beta_init=warmstart_beta,
                    inner_maxiter=inner_maxiter,
                    outer_maxiter=30,  # Reduced from default (λ is already close)
                    lambda_init=previous_lambda,  # Warm-start from previous iteration
                    verbose=false
                )
                final_penalty = smoothing_result.penalty
                warmstart_beta = smoothing_result.warmstart_beta
            end
        end
        
        # CRITICAL: Final λ re-selection after α converges
        # The within-loop λ re-selection only happens when alpha_iter < alpha_maxiter,
        # so we need a final re-selection after α learning completes to ensure λ is
        # optimized at the converged α values.
        if !(selector isa NoSelection)
            if verbose
                println("Re-selecting λ at final α values...")
            end
            previous_lambda = get_hyperparameters(final_penalty)
            smoothing_result = _select_hyperparameters(
                model, data, final_penalty, selector;
                beta_init=warmstart_beta,
                inner_maxiter=inner_maxiter,
                outer_maxiter=30,
                lambda_init=previous_lambda,
                verbose=false
            )
            final_penalty = smoothing_result.penalty
            warmstart_beta = smoothing_result.warmstart_beta
            
            if verbose
                final_lambda = get_hyperparameters(final_penalty)
                println("Final λ at converged α: $(round.(final_lambda, digits=4))")
            end
        end
        
        if verbose
            println("Final α values: $(collect(values(current_alphas)))")
        end
    end
    
    # =========================================================================
    # FINAL FIT: Always happens here via _fit_coefficients_at_fixed_hyperparameters
    # =========================================================================
    # This is the key architectural change: selection functions return
    # HyperparameterSelectionResult with warmstart_beta, NOT the final fit.
    # The final fit is always performed here with full convergence criteria.
    
    if verbose
        println("Performing final coefficient fitting at fixed λ...")
    end
    
    sol = _fit_coefficients_at_fixed_hyperparameters(
        model, data, final_penalty, warmstart_beta;
        lb=lb, ub=ub,
        solver=solver, adtype=adtype,
        parallel=use_parallel, nthreads=nthreads,
        maxiter=500,  # Full convergence for final fit
        verbose=verbose
    )
    
    # =========================================================================
    # Post-processing
    # =========================================================================
    
    # Rectify spline coefficients to clean up numerical errors
    if any(isa.(model.hazards, _SplineHazard))
        rectify_coefs!(sol.u, model)
    end
    
    # Check convergence and monotone spline constraints
    sol_converged = hasfield(typeof(sol), :retcode) ? 
        (sol.retcode == ReturnCode.Success || sol.retcode == :Success) : 
        (sol.retcode == ReturnCode.Success)
    has_monotone = any(map(x -> (isa(x, _SplineHazard) && x.monotone != 0), model.hazards))
    
    # Handle constraints case (only with NoSelection)
    at_bounds = nothing
    if !isnothing(constraints)
        at_bounds = identify_bound_parameters(sol.u, lb, ub)
    end
    
    # Warn about model-based variance with penalized models
    if vcov_type == :model && has_penalties(final_penalty)
        @warn """Model-based variance (`vcov_type=:model`) may be inappropriate for penalized models.
        The inverse Hessian of the PENALIZED likelihood does not account for smoothing bias.
        Consider using `vcov_type=:ij` (infinitesimal jackknife, default) for robust variance estimates,
        or `vcov_type=:jk` for jackknife variance."""
    end
    
    # Compute variance-covariance matrix
    vcov, vcov_type_used, subject_grads, vcov_model_base = _compute_vcov_exact(
        sol.u, model, samplepaths, vcov_type;
        vcov_threshold=vcov_threshold, loo_method=loo_method,
        converged=sol_converged, has_monotone=has_monotone, verbose=verbose,
        at_bounds=at_bounds
    )
    
    # =========================================================================
    # Assemble fitted model
    # =========================================================================
    
    # Compute subject-level likelihood at the estimate (always UNPENALIZED for model comparison)
    ll_subj = loglik_exact(sol.u, data; return_ll_subj=true)
    
    # Build parameters structure for fitted parameters
    params_nested = unflatten(model.parameters.reconstructor, sol.u)
    parameters_fitted = (
        flat = Vector{Float64}(sol.u),
        nested = params_nested,
        reconstructor = model.parameters.reconstructor
    )
    
    # Split sol.u into per-hazard vectors
    block_sizes = [model.hazards[i].npar_total for i in 1:length(model.hazards)]
    fitted_params = Vector{Vector{Float64}}(undef, length(block_sizes))
    offset = 0
    for i in eachindex(block_sizes)
        fitted_params[i] = sol.u[(offset+1):(offset+block_sizes[i])]
        offset += block_sizes[i]
    end
    
    # Extract smoothing parameters and EDF from selection result
    selected_lambda = isnothing(smoothing_result) ? get_hyperparameters(final_penalty) : smoothing_result.lambda
    selected_edf = if isnothing(smoothing_result)
        # Compute EDF for fixed λ case
        compute_edf(sol.u, selected_lambda, final_penalty, model, data)
    else
        smoothing_result.edf
    end
    
    # Create fitted model
    model_fitted = MultistateModelFitted(
        model.data,
        parameters_fitted,
        model.bounds,
        (loglik = -sol.objective, subj_lml = ll_subj),
        vcov,
        vcov_type_used,
        vcov_model_base,  # Model-based variance (H⁻¹) when IJ/JK requested
        subject_grads,
        model.hazards,
        model.totalhazards,
        model.tmat,
        model.emat,
        model.hazkeys,
        model.subjectindices,
        model.SubjectWeights,
        model.ObservationWeights,
        model.CensoringPatterns,
        model.surrogate,
        (solution = sol,),  # ConvergenceRecords - now a REAL OptimizationSolution
        nothing,  # ProposedPaths
        model.modelcall,
        model.phasetype_expansion,
        selected_lambda,  # smoothing_parameters
        selected_edf)     # edf
    
    return model_fitted
end

"""
    _fit_coefficients_at_fixed_hyperparameters(model, data, penalty, beta_init; kwargs...)

Final coefficient optimization at fixed (selected or user-specified) hyperparameters.

This function performs the FINAL optimization for penalized fitting. It is always
called after hyperparameter selection (or directly if using fixed λ).

# Key Property
Returns a REAL `OptimizationSolution` from actual Ipopt optimization - never
a fake tuple or mock object.

# Arguments
- `model::MultistateProcess`: Model for likelihood evaluation
- `data::ExactData`: Data container
- `penalty::AbstractPenalty`: Penalty configuration with final λ values
- `beta_init::Vector{Float64}`: Initial/warm-start coefficients

# Keyword Arguments
- `lb::Vector{Float64}`: Lower bounds on parameters
- `ub::Vector{Float64}`: Upper bounds on parameters
- `solver`: Optimization solver (default: Ipopt)
- `adtype`: AD backend (:auto selects based on solver)
- `parallel::Bool=false`: Enable parallel likelihood evaluation
- `nthreads::Union{Nothing,Int}=nothing`: Number of threads
- `maxiter::Int=500`: Maximum iterations
- `verbose::Bool=false`: Print optimization progress

# Returns
- `OptimizationSolution`: REAL solution object from Ipopt optimization

# Notes
- Uses Ipopt with ForwardDiff for robust optimization
- Box constraints ensure parameter positivity where required
- Convergence tolerance is tighter than inner loop fitting
"""
function _fit_coefficients_at_fixed_hyperparameters(
    model::MultistateProcess,
    data::ExactData,
    penalty::AbstractPenalty,
    beta_init::Vector{Float64};
    lb::Vector{Float64},
    ub::Vector{Float64},
    solver = nothing,
    adtype = :auto,
    parallel::Bool = false,
    nthreads = nothing,
    maxiter::Int = 500,
    verbose::Bool = false
)
    # Build penalized likelihood function
    function penalized_nll(params, data)
        # Unpenalized negative log-likelihood
        nll = loglik_exact(params, data; neg=true, parallel=parallel)
        # Add penalty
        pen = compute_penalty(params, penalty)
        return nll + pen
    end
    
    # Resolve AD backend
    resolved_adtype = _resolve_adtype(adtype, solver)
    
    # Set up optimization problem
    optf = OptimizationFunction(penalized_nll, resolved_adtype)
    prob = OptimizationProblem(optf, beta_init, data; lb=lb, ub=ub)
    
    # Solve with Ipopt (robust interior point method)
    sol = _solve_optimization(prob, solver; maxiters=maxiter)
    
    if verbose
        if sol.retcode == ReturnCode.Success
            println("Final fit converged successfully")
        else
            println("Final fit returned: $(sol.retcode)")
        end
    end
    
    return sol
end
