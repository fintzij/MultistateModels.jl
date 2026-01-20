# =============================================================================
# Exact Data Fitting (_fit_exact)
# =============================================================================
#
# Fitting for continuously observed (exact) multistate data.
# Called by fit() when is_panel_data(model) == false.
#
# Features:
# - Direct MLE via Ipopt optimization
# - Optional parallel likelihood evaluation
# - Model-based and robust (IJ/jackknife) variance estimation
# - Support for constrained optimization
# - Penalized likelihood for spline hazards
#
# =============================================================================

"""
    _fit_exact(model::MultistateModel; kwargs...)

Internal implementation: Fit a multistate model to continuously observed (exact) data.

This is called by `fit()` when `is_panel_data(model) == false`.

# Keyword Arguments
- `penalty`: Penalty specification for spline hazards. Can be:
  - `:auto` (default): Apply `SplinePenalty()` if model has spline hazards, `nothing` otherwise
  - `:none`: Explicit opt-out for unpenalized fitting
  - `SplinePenalty()`: Curvature penalty on all spline hazards
  - `Vector{SplinePenalty}`: Multiple rules resolved by specificity
  - `nothing`: DEPRECATED - use `:none` instead
- `lambda_init::Float64=1.0`: Initial smoothing parameter value
- `select_lambda::Symbol=:pijcv`: Method for selecting smoothing parameter λ when penalty is active
  - `:pijcv` (default): Proximal iteration jackknife CV (fast, AD-optimized)
  - `:pijcv5`, `:pijcv10`, `:pijcv20`: k-fold Newton-approximated CV
  - `:efs`: Expected Fisher scoring criterion
  - `:perf`: Performance iteration criterion
  - `:none`: Use fixed λ from `lambda_init` (no automatic selection)

See also: [`SplinePenalty`](@ref), [`build_penalty_config`](@ref), [`select_smoothing_parameters`](@ref)
"""
function _fit_exact(model::MultistateModel; constraints = nothing, verbose = true, solver = nothing,
             adtype = :auto,
             parallel = false, nthreads = nothing,
             compute_vcov = true, vcov_threshold = true, compute_ij_vcov = true, 
             compute_jk_vcov = false, loo_method = :direct,
             penalty = :auto, lambda_init::Float64 = 1.0, 
             select_lambda::Symbol = :pijcv, kwargs...)

    # Resolve penalty specification (handles :auto, :none, deprecation warning)
    resolved_penalty = _resolve_penalty(penalty, model)

    # initialize array of sample paths
    samplepaths = extract_paths(model)

    # extract model parameters
    # Phase 3: Use ParameterHandling.jl flat parameters (natural scale since v0.3.0)
    parameters = get_parameters_flat(model)

    # Use model constraints if none provided and model has them
    if isnothing(constraints) && haskey(model.modelcall, :constraints) && !isnothing(model.modelcall.constraints)
        constraints = model.modelcall.constraints
    end

    # Configure threading
    n_paths = length(samplepaths)
    threading_config = ThreadingConfig(parallel=parallel, nthreads=nthreads)
    use_parallel = should_parallelize(threading_config, n_paths)
    
    if use_parallel && verbose
        println("Using $(threading_config.nthreads) threads for likelihood evaluation ($(n_paths) paths)")
    end

    # Build penalty configuration from resolved penalty
    penalty_config = build_penalty_config(model, resolved_penalty; lambda_init=lambda_init)
    use_penalty = has_penalties(penalty_config)
    
    if use_penalty && verbose
        println("Using penalized likelihood with $(penalty_config.n_lambda) smoothing parameter(s)")
    end

    # Use parameter bounds from model (generated at construction time)
    lb, ub = model.bounds.lb, model.bounds.ub

    # =========================================================================
    # SMOOTHING PARAMETER SELECTION (when penalty is active)
    # =========================================================================
    # If penalty is active and select_lambda != :none, use performance iteration
    # to jointly optimize (β, λ). This replaces the fixed-λ optimization below.
    # Note: Only perform selection when resolved_penalty is not nothing (i.e., when
    # there are baseline spline terms, not just smooth covariate terms).
    
    smoothing_result = nothing  # Will hold λ, edf, etc. if selection is performed
    
    if use_penalty && select_lambda != :none && isnothing(constraints) && !isnothing(resolved_penalty)
        if verbose
            println("Selecting smoothing parameter(s) via :$select_lambda...")
        end
        
        # Call select_smoothing_parameters with performance iteration
        smoothing_result = select_smoothing_parameters(
            model, resolved_penalty;
            method=select_lambda,
            lambda_init=lambda_init,
            verbose=verbose
        )
        
        if verbose
            println("Selected λ: $(round.(smoothing_result.lambda, sigdigits=4))")
            println("Effective degrees of freedom: $(round(smoothing_result.edf.total, digits=2))")
        end
        
        # Update penalty_config with selected λ values
        penalty_config = smoothing_result.penalty_config
    end

    # parse constraints, or not, and solve
    if isnothing(constraints) 
        # If smoothing parameter selection was performed, use those results
        # Otherwise, run standard optimization
        if !isnothing(smoothing_result)
            # Use results from performance iteration (already computed optimal β at optimal λ)
            sol_u = smoothing_result.beta
            sol_objective = loglik_exact_penalized(sol_u, ExactData(model, samplepaths), penalty_config; neg=true)
            sol_retcode = smoothing_result.converged ? ReturnCode.Success : ReturnCode.MaxIters
            
            # Create a mock solution struct compatible with downstream code
            sol = (u = sol_u, objective = sol_objective, retcode = sol_retcode)
        else
            # Standard optimization path (unpenalized or fixed λ)
            # Create likelihood function - parallel or sequential
            # Note: ForwardDiff uses the function passed to OptimizationFunction for both
            # objective and gradient. We pass the sequential version for AD correctness,
            # but create a parallel version for objective-only evaluation.
            if use_penalty
                # Penalized likelihood
                if use_parallel
                    loglik_fn = (params, data) -> loglik_exact_penalized(params, data, penalty_config; neg=true, parallel=true)
                else
                    loglik_fn = (params, data) -> loglik_exact_penalized(params, data, penalty_config; neg=true, parallel=false)
                end
            else
                # Unpenalized likelihood
                if use_parallel
                    loglik_fn = (params, data) -> loglik_exact(params, data; neg=true, parallel=true)
                else
                    loglik_fn = loglik_exact
                end
            end
            
            # get estimates - use Ipopt for box-constrained optimization
            # Resolve AD backend: :auto selects based on solver requirements
            resolved_adtype = _resolve_adtype(adtype, solver)

            optf = OptimizationFunction(loglik_fn, resolved_adtype)
            prob = OptimizationProblem(optf, parameters, ExactData(model, samplepaths); lb=lb, ub=ub)

            # solve with user-specified solver or default Ipopt
            sol = _solve_optimization(prob, solver)
        end

        # rectify spline coefs - CHECK THIS!!!!
        if any(isa.(model.hazards, _SplineHazard))
            rectify_coefs!(sol.u, model)
        end
        
        # get vcov (use UNPENALIZED likelihood for Fisher information)
        # Note: For penalized models, the vcov is an approximation; consider bootstrap for more accurate inference
        sol_converged = hasfield(typeof(sol), :retcode) ? 
            (sol.retcode == ReturnCode.Success || sol.retcode == :Success) : 
            (sol.retcode == ReturnCode.Success)
        if compute_vcov && sol_converged && !any(map(x -> (isa(x, _SplineHazard) && x.monotone != 0), model.hazards))
            # Compute subject-level gradients and Hessians to cache them for robust variance
            # This avoids redundant computation when computing robust variance estimates
            subject_grads_cache = compute_subject_gradients(sol.u, model, samplepaths)
            subject_hessians_cache = compute_subject_hessians_fast(sol.u, model, samplepaths)
            
            # Aggregate Fisher information from subject Hessians
            nparams = length(sol.u)
            fishinf = zeros(Float64, nparams, nparams)
            for H_i in subject_hessians_cache
                fishinf .-= H_i
            end
            
            vcov = pinv(Symmetric(fishinf), atol = vcov_threshold ? (log(length(samplepaths)) * length(sol.u))^-2 : sqrt(eps(real(float(oneunit(eltype(fishinf)))))))
            vcov[isapprox.(vcov, 0.0; atol = sqrt(eps(Float64)), rtol = sqrt(eps(Float64)))] .= 0.0
            vcov = Symmetric(vcov)
        else
            subject_grads_cache = nothing
            subject_hessians_cache = nothing
            vcov = nothing
        end
    else
        # create constraint function and check that constraints are satisfied at the initial values
        _constraints = deepcopy(constraints)
        consfun_multistate = parse_constraints(_constraints.cons, model.hazards; consfun_name = :consfun_multistate)

        initcons = consfun_multistate(zeros(length(constraints.cons)), parameters, nothing)
        badcons = findall(initcons .< constraints.lcons .|| initcons .> constraints.ucons)
        if length(badcons) > 0
            throw(ArgumentError("Constraints $badcons are violated at the initial parameter values."))
        end

        # Create likelihood function - parallel or sequential, with or without penalty
        if use_penalty
            if use_parallel
                loglik_fn = (params, data) -> loglik_exact_penalized(params, data, penalty_config; neg=true, parallel=true)
            else
                loglik_fn = (params, data) -> loglik_exact_penalized(params, data, penalty_config; neg=true, parallel=false)
            end
        else
            if use_parallel
                loglik_fn = (params, data) -> loglik_exact(params, data; neg=true, parallel=true)
            else
                loglik_fn = loglik_exact
            end
        end

        # Resolve AD backend
        resolved_adtype = _resolve_adtype(adtype, solver)

        optf = OptimizationFunction(loglik_fn, resolved_adtype, cons = consfun_multistate)
        prob = OptimizationProblem(optf, parameters, ExactData(model, samplepaths); lb=lb, ub=ub, lcons = constraints.lcons, ucons = constraints.ucons)
        
        # solve with user-specified solver or default Ipopt
        sol = _solve_optimization(prob, solver)

        # rectify spline coefs
        if any(isa.(model.hazards, _SplineHazard))
            rectify_coefs!(sol.u, model)
        end

        # Check for parameters at box bounds (may affect variance estimates)
        warn_bound_parameters(sol.u, lb, ub)
        
        # Compute subject gradients and Hessians for IJ/JK variance (can be done with constraints)
        # IJ/JK variance only requires subject-level score vectors and Hessians, which are
        # computed at the MLE regardless of whether constraints were used to find it.
        if compute_vcov || compute_ij_vcov || compute_jk_vcov
            if verbose && (compute_ij_vcov || compute_jk_vcov)
                println("Computing subject gradients and Hessians for variance estimation...")
            end
            # Compute subject gradients and Hessians separately (matching unconstrained branch)
            subject_grads_cache = compute_subject_gradients(sol.u, model, samplepaths)
            subject_hessians_cache = compute_subject_hessians_fast(sol.u, model, samplepaths)
        else
            subject_grads_cache = nothing
            subject_hessians_cache = nothing
        end
        
        # Model-based vcov with constraints: use reduced Hessian approach
        if compute_vcov && !isnothing(subject_hessians_cache)
            # Build AD-compatible constraint function for Jacobian computation
            # The function must return an array of the same type as input parameters
            # to support ForwardDiff.jacobian
            n_cons = length(constraints.cons)
            cons_fn = function(x)
                # Create result array with correct element type for AD compatibility
                res = similar(x, n_cons)
                consfun_multistate(res, x, nothing)
                return res
            end
            constraints_for_jacobian = (cons_fn = cons_fn, lcons = constraints.lcons, ucons = constraints.ucons)
            
            # Identify which constraints are active at the solution
            active = identify_active_constraints(sol.u, constraints_for_jacobian)
            n_active = sum(active)
            
            if n_active > 0
                # Compute Jacobian of active constraints
                J_full = compute_constraint_jacobian(sol.u, constraints_for_jacobian)
                J_active = J_full[active, :]
                
                if verbose
                    println("Computing model-based variance with $(n_active) active constraint(s) using reduced Hessian...")
                end
                
                # Use reduced Hessian approach
                vcov = compute_constrained_vcov_from_components(subject_hessians_cache, J_active;
                                                                vcov_threshold=vcov_threshold)
            else
                # No active constraints at solution - use standard inverse Hessian
                if verbose
                    println("No constraints active at solution; using standard model-based variance...")
                end
                nparams = length(sol.u)
                fishinf = zeros(Float64, nparams, nparams)
                for H_i in subject_hessians_cache
                    fishinf .-= H_i
                end
                vcov = pinv(Symmetric(fishinf), atol = vcov_threshold ? (log(length(samplepaths)) * length(sol.u))^-2 : sqrt(eps(Float64)))
                vcov[isapprox.(vcov, 0.0; atol = sqrt(eps(Float64)), rtol = sqrt(eps(Float64)))] .= 0.0
                vcov = Symmetric(vcov)
            end
        else
            vcov = nothing
        end
    end

    # compute subject-level likelihood at the estimate (always UNPENALIZED for model comparison)
    ll_subj = loglik_exact(sol.u, ExactData(model, samplepaths); return_ll_subj = true)

    # Compute robust variance estimates if requested
    # IJ/JK variance can be computed with or without constraints - they use subject-level
    # gradients which are computed at the MLE regardless of how it was found.
    ij_variance = nothing
    jk_variance = nothing
    subject_grads = nothing
    
    if compute_ij_vcov || compute_jk_vcov
        if verbose
            println("Computing robust variance estimates...")
        end
        robust_result = compute_robust_vcov(sol.u, model, samplepaths;
                                           compute_ij = compute_ij_vcov,
                                           compute_jk = compute_jk_vcov,
                                           loo_method = loo_method,
                                           vcov_threshold = vcov_threshold,
                                           subject_grads = subject_grads_cache,
                                           subject_hessians = subject_hessians_cache)
        ij_variance = robust_result.ij_vcov
        jk_variance = robust_result.jk_vcov
        subject_grads = robust_result.subject_gradients
    end

    # Build parameters structure for fitted parameters
    # Use the unflatten function from the model to convert flat params back to nested
    params_nested = unflatten(model.parameters.reconstructor, sol.u)
    
    # v0.3.0+: No separate .natural field - compute on-demand via accessors
    parameters_fitted = (
        flat = Vector{Float64}(sol.u),
        nested = params_nested,
        reconstructor = model.parameters.reconstructor  # Keep same reconstructor
    )

    # Split sol.u into per-hazard vectors for spline remaking
    block_sizes = [model.hazards[i].npar_total for i in 1:length(model.hazards)]
    fitted_params = Vector{Vector{Float64}}(undef, length(block_sizes))
    offset = 0
    for i in eachindex(block_sizes)
        fitted_params[i] = sol.u[(offset+1):(offset+block_sizes[i])]
        offset += block_sizes[i]
    end

    # Extract smoothing parameters and EDF from smoothing result (if available)
    selected_lambda = isnothing(smoothing_result) ? nothing : smoothing_result.lambda
    selected_edf = isnothing(smoothing_result) ? nothing : smoothing_result.edf

    # wrap results
    model_fitted = MultistateModelFitted(
        model.data,
        parameters_fitted,
        model.bounds,  # Pass bounds from unfitted model
        (loglik = -sol.objective, subj_lml = ll_subj),
        vcov,
        isnothing(ij_variance) ? nothing : Matrix(ij_variance),
        isnothing(jk_variance) ? nothing : Matrix(jk_variance),
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
        model.markovsurrogate,
        model.phasetype_surrogate,  # phasetype_surrogate
        (solution = sol,), # ConvergenceRecords::Union{Nothing, NamedTuple}
        nothing, # ProposedPaths::Union{Nothing, NamedTuple}
        model.modelcall,
        model.phasetype_expansion,
        selected_lambda,  # smoothing_parameters
        selected_edf)     # edf

    # return fitted object
    return model_fitted;
end

# NOTE: fit(::PhaseTypeModel) has been removed as part of package streamlining.
# Phase-type hazards are now handled internally via MultistateModel with
# phasetype_expansion metadata. The standard fit() method handles expansion.

"""
    _fit_markov_panel(model::MultistateModel; kwargs...)

Internal implementation: Fit a Markov model to interval-censored/panel data.

This is called by `fit()` when `is_panel_data(model) && is_markov(model)`.

# Arguments
- `model::MultistateModel`: Markov model with panel observations
- `constraints`: parameter constraints (see Constraints documentation)
- `verbose::Bool=true`: print optimization messages
- `solver`: optimization solver (default: Ipopt)
- `adbackend::ADBackend=ForwardDiffBackend()`: automatic differentiation backend

# Notes
- For Markov models, the likelihood involves matrix exponentials of the intensity matrix
- Censored state observations are handled via marginalization over possible states
"""
