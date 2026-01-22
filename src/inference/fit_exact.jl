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
# - Unified variance-covariance estimation via vcov_type
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
- `vcov_type::Symbol=:ij`: Type of variance-covariance matrix to compute:
  - `:ij` (default): Infinitesimal jackknife / sandwich variance (robust)
  - `:model`: Model-based variance (inverse Fisher information)
  - `:jk`: Jackknife variance (leave-one-out)
  - `:none`: Skip variance computation
- `vcov_threshold::Bool=true`: use adaptive threshold for pseudo-inverse

See also: [`SplinePenalty`](@ref), [`build_penalty_config`](@ref), [`select_smoothing_parameters`](@ref)
"""
function _fit_exact(model::MultistateModel; constraints = nothing, verbose = true, solver = nothing,
             adtype = :auto,
             parallel = false, nthreads = nothing,
             vcov_type::Symbol = :ij, vcov_threshold = true,
             loo_method = :direct,
             penalty = :auto, lambda_init::Float64 = 1.0, 
             select_lambda::Symbol = :pijcv, kwargs...)

    # Validate vcov_type
    _validate_vcov_type(vcov_type)

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

        # Rectify spline coefficients to clean up numerical errors from I-spline transformation.
        # See rectify_coefs! docstring in hazard/spline.jl for mathematical justification.
        if any(isa.(model.hazards, _SplineHazard))
            rectify_coefs!(sol.u, model)
        end
        
        # Check convergence and monotone spline constraints
        sol_converged = hasfield(typeof(sol), :retcode) ? 
            (sol.retcode == ReturnCode.Success || sol.retcode == :Success) : 
            (sol.retcode == ReturnCode.Success)
        has_monotone = any(map(x -> (isa(x, _SplineHazard) && x.monotone != 0), model.hazards))
        
        # Compute variance-covariance matrix based on vcov_type
        vcov, vcov_type_used, subject_grads = _compute_vcov_exact(
            sol.u, model, samplepaths, vcov_type;
            vcov_threshold=vcov_threshold, loo_method=loo_method,
            converged=sol_converged, has_monotone=has_monotone, verbose=verbose
        )
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
        at_bounds = identify_bound_parameters(sol.u, lb, ub)
        
        # Check convergence and monotone spline constraints  
        sol_converged = hasfield(typeof(sol), :retcode) ? 
            (sol.retcode == ReturnCode.Success || sol.retcode == :Success) : 
            (sol.retcode == ReturnCode.Success)
        has_monotone = any(map(x -> (isa(x, _SplineHazard) && x.monotone != 0), model.hazards))
        
        # Compute variance-covariance matrix based on vcov_type
        vcov, vcov_type_used, subject_grads = _compute_vcov_exact(
            sol.u, model, samplepaths, vcov_type;
            vcov_threshold=vcov_threshold, loo_method=loo_method,
            converged=sol_converged, has_monotone=has_monotone, verbose=verbose,
            at_bounds=at_bounds
        )
    end

    # compute subject-level likelihood at the estimate (always UNPENALIZED for model comparison)
    ll_subj = loglik_exact(sol.u, ExactData(model, samplepaths); return_ll_subj = true)

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
        vcov_type_used,  # Type of vcov that was computed
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
        model.surrogate,  # Unified surrogate field
        (solution = sol,), # ConvergenceRecords::Union{Nothing, NamedTuple}
        nothing, # ProposedPaths::Union{Nothing, NamedTuple}
        model.modelcall,
        model.phasetype_expansion,
        selected_lambda,  # smoothing_parameters
        selected_edf)     # edf

    # return fitted object
    return model_fitted;
end

"""
    _compute_vcov_exact(params, model, samplepaths, vcov_type; kwargs...)

Compute variance-covariance matrix for exact data fitting.

Unified variance computation that handles all vcov_type options and constraint scenarios.

# Arguments
- `params::AbstractVector`: Fitted parameter vector
- `model::MultistateModel`: Model
- `samplepaths::Vector{SamplePath}`: Sample paths
- `vcov_type::Symbol`: Type of variance (:ij, :model, :jk, :none)

# Keyword Arguments
- `vcov_threshold::Bool=true`: Use adaptive pseudo-inverse tolerance
- `loo_method::Symbol=:direct`: LOO perturbation method
- `converged::Bool=true`: Whether optimization converged
- `has_monotone::Bool=false`: Whether model has monotone spline constraints
- `verbose::Bool=true`: Print progress messages
- `at_bounds::Union{Nothing,BitVector}=nothing`: Indicators for params at box bounds

# Returns
- `(vcov, vcov_type_used, subject_grads)`: Tuple with vcov matrix, actual type used, and gradients
"""
function _compute_vcov_exact(params::AbstractVector, model::MultistateModel, 
                             samplepaths::Vector{SamplePath}, vcov_type::Symbol;
                             vcov_threshold::Bool=true, loo_method::Symbol=:direct,
                             converged::Bool=true, has_monotone::Bool=false,
                             verbose::Bool=true, at_bounds::Union{Nothing,BitVector}=nothing)
    
    # Early return if no vcov requested
    if vcov_type == :none
        return nothing, :none, nothing
    end
    
    # Skip vcov if not converged or has monotone constraints
    if !converged
        verbose && @warn "Optimization did not converge; skipping variance computation."
        return nothing, :none, nothing
    end
    if has_monotone
        verbose && @warn "Model has monotone spline constraints; variance computation not supported."
        return nothing, :none, nothing
    end
    
    nparams = length(params)
    nsubj = length(samplepaths)
    
    # Compute subject-level gradients and Hessians (needed for all variance types)
    if verbose
        println("Computing subject gradients and Hessians for variance estimation...")
    end
    subject_grads = compute_subject_gradients(params, model, samplepaths)
    subject_hessians = compute_subject_hessians_fast(params, model, samplepaths)
    
    # Aggregate Fisher information
    fishinf = zeros(Float64, nparams, nparams)
    for H_i in subject_hessians
        fishinf .-= H_i
    end
    
    # Compute base inverse Hessian (needed for most variance types)
    pinv_tol = _compute_vcov_tolerance(nsubj, nparams, vcov_threshold)
    H_inv = pinv(Symmetric(fishinf), atol=pinv_tol)
    _clean_vcov_matrix!(H_inv)
    
    # Compute requested variance type
    if vcov_type == :model
        vcov = Symmetric(H_inv)
        vcov_type_used = :model
        
    elseif vcov_type == :ij
        # IJ / Sandwich variance: H⁻¹ J H⁻¹ where J = Σᵢ gᵢgᵢᵀ
        J = subject_grads * subject_grads'  # p×n × n×p = p×p
        vcov_mat = H_inv * J * H_inv
        _clean_vcov_matrix!(vcov_mat)
        vcov = Matrix(Symmetric(vcov_mat))
        vcov_type_used = :ij
        
    elseif vcov_type == :jk
        # Jackknife variance: ((n-1)/n) Σᵢ ΔᵢΔᵢᵀ where Δᵢ = H⁻¹gᵢ
        deltas = H_inv * subject_grads  # p×n
        vcov_mat = ((nsubj - 1) / nsubj) * (deltas * deltas')
        _clean_vcov_matrix!(vcov_mat)
        vcov = Matrix(Symmetric(vcov_mat))
        vcov_type_used = :jk
        
    else
        throw(ArgumentError("Unknown vcov_type: $vcov_type"))
    end
    
    # Handle parameters at bounds: set variance to NaN with warning
    if !isnothing(at_bounds) && any(at_bounds)
        bound_indices = findall(at_bounds)
        @warn "Parameters at bounds: $bound_indices. " *
              "Setting variance to NaN for these parameters (standard asymptotics don't hold at boundaries)."
        for i in bound_indices
            vcov[i, :] .= NaN
            vcov[:, i] .= NaN
        end
    end
    
    return vcov, vcov_type_used, subject_grads
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
