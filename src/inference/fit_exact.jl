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
  - `nothing` (default): No penalty
  - `SplinePenalty()`: Curvature penalty on all spline hazards
  - `Vector{SplinePenalty}`: Multiple rules resolved by specificity
- `lambda_init::Float64=1.0`: Initial smoothing parameter value

See also: [`SplinePenalty`](@ref), [`build_penalty_config`](@ref)
"""
function _fit_exact(model::MultistateModel; constraints = nothing, verbose = true, solver = nothing, 
             parallel = false, nthreads = nothing,
             compute_vcov = true, vcov_threshold = true, compute_ij_vcov = true, 
             compute_jk_vcov = false, loo_method = :direct,
             penalty = nothing, lambda_init::Float64 = 1.0, kwargs...)

    # initialize array of sample paths
    samplepaths = extract_paths(model)

    # extract model parameters
    # Phase 3: Use ParameterHandling.jl flat parameters (log scale)
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

    # Build penalty configuration if specified
    penalty_config = build_penalty_config(model, penalty; lambda_init=lambda_init)
    use_penalty = has_penalties(penalty_config)
    
    if use_penalty && verbose
        println("Using penalized likelihood with $(penalty_config.n_lambda) smoothing parameter(s)")
    end

    # Generate parameter bounds for box-constrained optimization
    lb, ub = generate_parameter_bounds(model)

    # parse constraints, or not, and solve
    if isnothing(constraints) 
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
        # Use SecondOrder AD if solver requires it (Newton, Ipopt) to avoid warnings
        adtype = Optimization.AutoForwardDiff()
        if isnothing(solver) || (solver isa Optim.Newton) || (solver isa Optim.NewtonTrustRegion) || (solver isa IpoptOptimizer)
            adtype = DifferentiationInterface.SecondOrder(Optimization.AutoForwardDiff(), Optimization.AutoForwardDiff())
        end

        optf = OptimizationFunction(loglik_fn, adtype)
        prob = OptimizationProblem(optf, parameters, ExactData(model, samplepaths); lb=lb, ub=ub)

        # solve with user-specified solver or default Ipopt
        sol = _solve_optimization(prob, solver)

        # rectify spline coefs
        if any(isa.(model.hazards, _SplineHazard))
            rectify_coefs!(sol.u, model)
        end
        
        # get vcov (use UNPENALIZED likelihood for Fisher information)
        # Note: For penalized models, the vcov is an approximation; consider bootstrap for more accurate inference
        if compute_vcov && (sol.retcode == ReturnCode.Success) && !any(map(x -> (isa(x, _SplineHazard) && x.monotone != 0), model.hazards))
            # Compute subject-level gradients and Hessians to cache them for robust variance
            # This avoids redundant computation when computing robust variance estimates
            subject_grads_cache = compute_subject_gradients(sol.u, model, samplepaths)
            subject_hessians_cache = compute_subject_hessians(sol.u, model, samplepaths)
            
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

        # Use SecondOrder AD if solver requires it
        adtype = Optimization.AutoForwardDiff()
        if isnothing(solver) || (solver isa Optim.Newton) || (solver isa Optim.NewtonTrustRegion) || (solver isa IpoptOptimizer)
            adtype = DifferentiationInterface.SecondOrder(Optimization.AutoForwardDiff(), Optimization.AutoForwardDiff())
        end

        optf = OptimizationFunction(loglik_fn, adtype, cons = consfun_multistate)
        prob = OptimizationProblem(optf, parameters, ExactData(model, samplepaths); lb=lb, ub=ub, lcons = constraints.lcons, ucons = constraints.ucons)
        
        # solve with user-specified solver or default Ipopt
        sol = _solve_optimization(prob, solver)

        # rectify spline coefs
        if any(isa.(model.hazards, _SplineHazard))
            rectify_coefs!(sol.u, model)
        end

        # no hessian when there are constraints
        if compute_vcov == true
            @warn "No covariance matrix is returned when constraints are provided."
        end
        subject_grads_cache = nothing
        subject_hessians_cache = nothing
        vcov = nothing
    end

    # compute subject-level likelihood at the estimate (always UNPENALIZED for model comparison)
    ll_subj = loglik_exact(sol.u, ExactData(model, samplepaths); return_ll_subj = true)

    # compute robust variance estimates if requested
    ij_variance = nothing
    jk_variance = nothing
    subject_grads = nothing
    
    if (compute_ij_vcov || compute_jk_vcov) && !isnothing(vcov) && isnothing(constraints)
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
    
    # Compute natural scale parameters (family-aware transformation)
    params_natural_pairs = [
        hazname => extract_natural_vector(params_nested[hazname], model.hazards[idx].family)
        for (hazname, idx) in sort(collect(model.hazkeys), by = x -> x[2])
    ]
    params_natural = NamedTuple(params_natural_pairs)
    
    parameters_fitted = (
        flat = Vector{Float64}(sol.u),
        nested = params_nested,
        natural = params_natural,
        reconstructor = model.parameters.reconstructor  # Keep same reconstructor
    )

    # Split sol.u into per-hazard log-scale vectors for spline remaking
    natural_vals = values(model.parameters.natural)
    block_sizes = [length(v) for v in natural_vals]
    log_scale_params = Vector{Vector{Float64}}(undef, length(block_sizes))
    offset = 0
    for i in eachindex(block_sizes)
        log_scale_params[i] = sol.u[(offset+1):(offset+block_sizes[i])]
        offset += block_sizes[i]
    end

    # wrap results
    model_fitted = MultistateModelFitted(
        model.data,
        parameters_fitted,
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
        (solution = sol,), # ConvergenceRecords::Union{Nothing, NamedTuple}
        nothing, # ProposedPaths::Union{Nothing, NamedTuple}
        model.modelcall,
        model.phasetype_expansion)

    # remake splines and calculate risk periods using log-scale parameters
    for i in eachindex(model_fitted.hazards)
        if isa(model_fitted.hazards[i], _SplineHazard)
            remake_splines!(model_fitted.hazards[i], log_scale_params[i])
            set_riskperiod!(model_fitted.hazards[i])
        end
    end

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
