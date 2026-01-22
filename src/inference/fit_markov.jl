# =============================================================================
# Markov Panel Data Fitting (_fit_markov_panel)
# =============================================================================
#
# Fitting for Markov models with interval-censored (panel) observations.
# Called by fit() when is_panel_data(model) && is_markov(model).
#
# Features:
# - Matrix exponential likelihood via forward algorithm
# - Support for censored state observations
# - ForwardDiff or Mooncake AD backends
# - Unified variance-covariance estimation via vcov_type
#
# =============================================================================

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
- `vcov_type::Symbol=:ij`: Type of variance-covariance matrix (:ij, :model, :jk, :none)

# Notes
- For Markov models, the likelihood involves matrix exponentials of the intensity matrix
- Censored state observations are handled via marginalization over possible states
"""
function _fit_markov_panel(model::MultistateModel; constraints = nothing, verbose = true, 
                          solver = nothing, adbackend::ADBackend = ForwardDiffBackend(), 
                          vcov_type::Symbol = :ij, vcov_threshold = true,
                          loo_method = :direct, kwargs...)

    # Validate vcov_type
    _validate_vcov_type(vcov_type)

    # containers for bookkeeping TPMs
    books = build_tpm_mapping(model.data)

    # extract model parameters
    # Phase 3: Use ParameterHandling.jl flat parameters (natural scale since v0.3.0)
    parameters = get_parameters_flat(model)

    # number of subjects
    nsubj = length(model.subjectindices)

    # Use model constraints if none provided and model has them
    if isnothing(constraints) && haskey(model.modelcall, :constraints) && !isnothing(model.modelcall.constraints)
        constraints = model.modelcall.constraints
    end

    # Warn if using reverse-mode AD for Markov models (matrix exponential issue)
    if adbackend isa MooncakeBackend
        @warn "MooncakeBackend may fail for Markov models due to LAPACK calls in matrix exponential. " *
              "ChainRules.jl's exp rule also uses LAPACK internally. Use ForwardDiffBackend() if you encounter errors."
    end

    # Create closure that dispatches to correct implementation based on backend
    loglik_fn = (p, d) -> loglik_markov(p, d; backend=adbackend)

    # Use parameter bounds from model (generated at construction time)
    lb, ub = model.bounds.lb, model.bounds.ub

    # parse constraints, or not, and solve
    if isnothing(constraints)
        # get estimates - use Ipopt for box-constrained optimization
        optf = OptimizationFunction(loglik_fn, get_optimization_ad(adbackend))
        prob = OptimizationProblem(optf, parameters, MPanelData(model, books); lb=lb, ub=ub)
        
        # solve with user-specified solver or default Ipopt
        sol = _solve_optimization(prob, solver)

        # Compute variance-covariance based on vcov_type
        converged = (sol.retcode == ReturnCode.Success)
        vcov, vcov_type_used, subject_grads = _compute_vcov_markov(
            sol.u, model, books, vcov_type;
            vcov_threshold=vcov_threshold, loo_method=loo_method,
            converged=converged, verbose=verbose
        )
    else
        # create constraint function and check that constraints are satisfied at the initial values
        _constraints = deepcopy(constraints)
        consfun_markov = parse_constraints(_constraints.cons, model.hazards; consfun_name = :consfun_markov)

        initcons = consfun_markov(zeros(length(constraints.cons)), parameters, nothing)
        badcons = findall(initcons .< constraints.lcons .|| initcons .> constraints.ucons)
        if length(badcons) > 0
            throw(ArgumentError("Constraints $badcons are violated at the initial parameter values."))
        end

        optf = OptimizationFunction(loglik_fn, get_optimization_ad(adbackend), cons = consfun_markov)
        prob = OptimizationProblem(optf, parameters, MPanelData(model, books); lb=lb, ub=ub, lcons = constraints.lcons, ucons = constraints.ucons)
        
        # solve with user-specified solver or default Ipopt
        sol = _solve_optimization(prob, solver)

        # Check for parameters at box bounds
        at_bounds = identify_bound_parameters(sol.u, lb, ub)
        
        # Compute variance-covariance based on vcov_type
        converged = (sol.retcode == ReturnCode.Success)
        vcov, vcov_type_used, subject_grads = _compute_vcov_markov(
            sol.u, model, books, vcov_type;
            vcov_threshold=vcov_threshold, loo_method=loo_method,
            converged=converged, verbose=verbose, at_bounds=at_bounds
        )
    end

    # compute loglikelihood at the estimate
    logliks = (loglik = -sol.objective, subj_lml = loglik_markov(sol.u, MPanelData(model, books); return_ll_subj = true))

    # Build parameters structure for fitted parameters
    # Use the unflatten function from the model to convert flat params back to nested
    params_nested = unflatten(model.parameters.reconstructor, sol.u)
    
    parameters_fitted = (
        flat = Vector{Float64}(sol.u),
        nested = params_nested,
        reconstructor = model.parameters.reconstructor  # Keep same reconstructor
    )

    # wrap results
    return MultistateModelFitted(
        model.data,
        parameters_fitted,
        model.bounds,  # Pass bounds from unfitted model
        logliks,
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
        nothing,  # smoothing_parameters (not yet implemented for Markov panel)
        nothing); # edf
end

"""
    _compute_vcov_markov(params, model, books, vcov_type; kwargs...)

Compute variance-covariance matrix for Markov panel data fitting.

Unified variance computation that handles all vcov_type options.

# Arguments
- `params::AbstractVector`: Fitted parameter vector
- `model::MultistateModel`: Model
- `books::Tuple`: TPM bookkeeping structures
- `vcov_type::Symbol`: Type of variance (:ij, :model, :jk, :none)

# Keyword Arguments
- `vcov_threshold::Bool=true`: Use adaptive pseudo-inverse tolerance
- `loo_method::Symbol=:direct`: LOO perturbation method
- `converged::Bool=true`: Whether optimization converged
- `verbose::Bool=true`: Print progress messages
- `at_bounds::Union{Nothing,BitVector}=nothing`: Indicators for params at box bounds

# Returns
- `(vcov, vcov_type_used, subject_grads)`: Tuple with vcov matrix, actual type used, and gradients
"""
function _compute_vcov_markov(params::AbstractVector, model::MultistateModel, 
                              books::Tuple, vcov_type::Symbol;
                              vcov_threshold::Bool=true, loo_method::Symbol=:direct,
                              converged::Bool=true, verbose::Bool=true,
                              at_bounds::Union{Nothing,BitVector}=nothing)
    
    # Early return if no vcov requested
    if vcov_type == :none
        return nothing, :none, nothing
    end
    
    # Skip vcov if not converged
    if !converged
        verbose && @warn "Optimization did not converge; skipping variance computation."
        return nothing, :none, nothing
    end
    
    nparams = length(params)
    nsubj = length(model.subjectindices)
    
    # Compute subject-level gradients and Hessians (needed for all variance types)
    if verbose
        println("Computing subject gradients and Hessians for variance estimation...")
    end
    subject_grads = compute_subject_gradients(params, model, books)
    subject_hessians = compute_subject_hessians(params, model, books)
    
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
