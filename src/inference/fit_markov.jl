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
# - Penalized likelihood for spline hazards (Phase M1)
#
# =============================================================================

"""
    _fit_markov_panel(model::MultistateModel; kwargs...)

Dispatcher for Markov panel data fitting: routes to penalized or unpenalized paths.

This is called by `fit()` when `is_panel_data(model) && is_markov(model)`.

# Dispatch Logic
- **Penalized path** (`_fit_markov_panel_penalized`): When `penalty` is active and model has spline hazards
- **Unpenalized path**: Standard MLE with Ipopt when no penalties apply

# Arguments
- `model::MultistateModel`: Markov model with panel observations
- `constraints`: parameter constraints (see Constraints documentation)
- `verbose::Bool=true`: print optimization messages
- `solver`: optimization solver (default: Ipopt)
- `adbackend::ADBackend=ForwardDiffBackend()`: automatic differentiation backend
- `vcov_type::Symbol=:ij`: Type of variance-covariance matrix (:ij, :model, :jk, :none)
- `penalty`: Penalty specification for spline hazards. Can be:
  - `:auto` (default): Apply `SplinePenalty()` if model has spline hazards, `nothing` otherwise
  - `:none`: Explicit opt-out for unpenalized fitting
  - `SplinePenalty()`: Curvature penalty on all spline hazards
- `lambda_init::Float64=1.0`: Initial smoothing parameter value for penalized fitting
- `select_lambda::Symbol=:none`: Method for selecting smoothing parameter λ
  - `:none` (default): Use fixed λ from `lambda_init` (Phase M1)
  - Other methods (`:pijcv`, etc.) to be added in Phase M2

# Notes
- For Markov models, the likelihood involves matrix exponentials of the intensity matrix
- Censored state observations are handled via marginalization over possible states
- Penalized fitting uses Ipopt with ForwardDiff (Phase M1: fixed λ only)

See also: [`_fit_markov_panel_penalized`](@ref), [`SplinePenalty`](@ref)
"""
function _fit_markov_panel(model::MultistateModel; constraints = nothing, verbose = true, 
                          solver = nothing, adbackend::ADBackend = ForwardDiffBackend(), 
                          vcov_type::Symbol = :ij, vcov_threshold = true,
                          loo_method = :direct,
                          # Penalty kwargs (Phase M1)
                          penalty = :auto, lambda_init::Float64 = 1.0,
                          select_lambda::Symbol = :none,
                          kwargs...)

    # =========================================================================
    # Validate inputs
    # =========================================================================
    _validate_vcov_type(vcov_type)

    # Resolve penalty specification (handles :auto, :none, deprecation warning)
    resolved_penalty = _resolve_penalty(penalty, model)
    
    # Build penalty configuration from resolved penalty
    penalty_config = build_penalty_config(model, resolved_penalty; lambda_init=lambda_init)
    use_penalty = has_penalties(penalty_config)

    # containers for bookkeeping TPMs
    books = build_tpm_mapping(model.data)

    # Use model constraints if none provided and model has them
    if isnothing(constraints) && haskey(model.modelcall, :constraints) && !isnothing(model.modelcall.constraints)
        constraints = model.modelcall.constraints
    end

    # =========================================================================
    # DISPATCH: Penalized vs Unpenalized path
    # =========================================================================
    if use_penalty && !isnothing(resolved_penalty)
        # Resolve selector from symbol to type
        selector = _resolve_selector(select_lambda, penalty_config)
        
        if verbose
            println("Using penalized likelihood with $(penalty_config.n_lambda) smoothing parameter(s)")
        end
        
        # Dispatch to penalized fitting function
        return _fit_markov_panel_penalized(
            model, books, penalty_config, selector;
            constraints=constraints, verbose=verbose, solver=solver,
            vcov_type=vcov_type, vcov_threshold=vcov_threshold,
            loo_method=loo_method, lambda_init=lambda_init, kwargs...
        )
    end

    # =========================================================================
    # UNPENALIZED PATH: Standard MLE fitting (existing code)
    # =========================================================================
    
    # extract model parameters
    # Phase 3: Use ParameterHandling.jl flat parameters (natural scale since v0.3.0)
    parameters = get_parameters_flat(model)

    # number of subjects
    nsubj = length(model.subjectindices)

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
# =============================================================================
# Penalized Markov Panel Fitting (Phase M1)
# =============================================================================

"""
    _fit_markov_panel_penalized(model, books, penalty, selector; kwargs...)

Penalized maximum likelihood fitting for Markov panel data.

This function handles penalized fitting paths for Markov panel models:
1. Fixed λ (selector = NoSelection): Direct optimization at provided λ

# Note
Phase M1 only supports fixed λ (NoSelection). Automatic λ selection
(PIJCVSelector, etc.) will be added in Phase M2.

# Arguments
- `model::MultistateModel`: Markov model with panel observations
- `books::Tuple`: TPM bookkeeping from `build_tpm_mapping`
- `penalty::AbstractPenalty`: Penalty configuration (must NOT be NoPenalty)
- `selector::AbstractHyperparameterSelector`: Selection method (NoSelection for Phase M1)

# Keyword Arguments
- `constraints`: Parameter constraints (not supported with selection)
- `verbose::Bool=true`: Print progress messages
- `solver`: Optimization solver (default: Ipopt)
- `vcov_type::Symbol=:ij`: Variance-covariance type
- `vcov_threshold::Bool=true`: Use adaptive tolerance for pseudo-inverse
- `loo_method::Symbol=:direct`: LOO perturbation method
- `lambda_init::Float64=1.0`: Initial λ for selection (used if selector != NoSelection)

# Returns
- `MultistateModelFitted`: Fitted model with coefficients, variance, smoothing_parameters, and edf

# Notes
- Uses Ipopt with ForwardDiff for all optimizations
- Supports automatic λ selection via PIJCV, EFS, PERF criteria

See also: [`_fit_markov_panel`](@ref), [`loglik_markov_penalized`](@ref)
"""
function _fit_markov_panel_penalized(
    model::MultistateModel,
    books::Tuple,
    penalty::AbstractPenalty,
    selector::AbstractHyperparameterSelector;
    constraints = nothing,
    verbose::Bool = true,
    solver = nothing,
    vcov_type::Symbol = :ij,
    vcov_threshold::Bool = true,
    loo_method::Symbol = :direct,
    lambda_init::Float64 = 1.0,
    inner_maxiter::Int = 50,
    outer_maxiter::Int = 100,
    lambda_tol::Float64 = 1e-3,
    kwargs...
)
    # =========================================================================
    # Validate inputs
    # =========================================================================
    @assert !(penalty isa NoPenalty) "NoPenalty should not reach _fit_markov_panel_penalized"
    
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
    
    # Create MPanelData container
    data = MPanelData(model, books)
    
    # =========================================================================
    # Hyperparameter Selection (if selector != NoSelection)
    # =========================================================================
    smoothing_result = nothing
    final_penalty = penalty
    warmstart_beta = parameters
    
    if !(selector isa NoSelection)
        # Automatic λ selection via _select_hyperparameters dispatch
        if verbose
            println("Selecting smoothing parameters via $(typeof(selector).name)...")
        end
        
        smoothing_result = _select_hyperparameters(
            model, data, penalty, selector;
            beta_init=parameters,
            inner_maxiter=inner_maxiter,
            outer_maxiter=outer_maxiter,
            lambda_tol=lambda_tol,
            verbose=verbose
        )
        
        final_penalty = smoothing_result.penalty
        warmstart_beta = smoothing_result.warmstart_beta
        
        if verbose
            lambda_vals = get_hyperparameters(final_penalty)
            println("Selected λ = $(round.(lambda_vals, sigdigits=4))")
        end
    else
        # NoSelection: use fixed λ
        if verbose
            lambda_vals = get_hyperparameters(final_penalty)
            println("Fitting with fixed λ = $(round.(lambda_vals, sigdigits=4))")
        end
    end
    
    # =========================================================================
    # Final fit at selected/fixed λ
    # =========================================================================
    
    # Define penalized negative log-likelihood
    function penalized_nll(params, d)
        # Base Markov likelihood (uses ForwardDiff-compatible mutating implementation)
        nll = _loglik_markov_mutating(params, d; neg=true, return_ll_subj=false)
        # Add penalty
        pen = compute_penalty(params, final_penalty)
        return nll + pen
    end
    
    # Set up optimization with Ipopt + ForwardDiff (HARD REQUIREMENT)
    adtype = DifferentiationInterface.SecondOrder(
        Optimization.AutoForwardDiff(), 
        Optimization.AutoForwardDiff()
    )
    
    optf = OptimizationFunction(penalized_nll, adtype)
    prob = OptimizationProblem(optf, warmstart_beta, data; lb=lb, ub=ub)
    
    # Solve with Ipopt
    sol = _solve_optimization(prob, solver)
    
    if verbose
        if sol.retcode == ReturnCode.Success
            println("Optimization converged successfully")
        else
            println("Optimization returned: $(sol.retcode)")
        end
    end
    
    # =========================================================================
    # Post-processing
    # =========================================================================
    
    # Check convergence
    sol_converged = hasfield(typeof(sol), :retcode) ? 
        (sol.retcode == ReturnCode.Success || sol.retcode == :Success) : 
        (sol.retcode == ReturnCode.Success)
    
    # Warn about model-based variance with penalized models
    if vcov_type == :model && has_penalties(final_penalty)
        @warn """Model-based variance (`vcov_type=:model`) may be inappropriate for penalized models.
        The inverse Hessian of the PENALIZED likelihood does not account for smoothing bias.
        Consider using `vcov_type=:ij` (infinitesimal jackknife, default) for robust variance estimates,
        or `vcov_type=:jk` for jackknife variance."""
    end
    
    # Compute variance-covariance matrix
    vcov, vcov_type_used, subject_grads = _compute_vcov_markov(
        sol.u, model, books, vcov_type;
        vcov_threshold=vcov_threshold, loo_method=loo_method,
        converged=sol_converged, verbose=verbose
    )
    
    # =========================================================================
    # Compute smoothing parameters and EDF
    # =========================================================================
    selected_lambda = get_hyperparameters(final_penalty)
    
    # Compute EDF for the fitted model
    selected_edf = compute_edf_markov(sol.u, selected_lambda, final_penalty, model, books)
    
    if verbose
        println("Effective degrees of freedom: $(round(selected_edf.total, digits=2))")
    end
    
    # =========================================================================
    # Compute UNPENALIZED log-likelihood for model comparison
    # =========================================================================
    # Always report the UNPENALIZED likelihood - this is comparable across models
    ll_unpenalized = -loglik_markov(sol.u, data; neg=true, return_ll_subj=false)
    ll_subj = loglik_markov(sol.u, data; return_ll_subj=true)
    logliks = (loglik = ll_unpenalized, subj_lml = ll_subj)
    
    # =========================================================================
    # Assemble fitted model
    # =========================================================================
    
    # Build parameters structure for fitted parameters
    params_nested = unflatten(model.parameters.reconstructor, sol.u)
    parameters_fitted = (
        flat = Vector{Float64}(sol.u),
        nested = params_nested,
        reconstructor = model.parameters.reconstructor
    )
    
    return MultistateModelFitted(
        model.data,
        parameters_fitted,
        model.bounds,
        logliks,
        vcov,
        vcov_type_used,
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
        (solution = sol,),  # ConvergenceRecords
        nothing,  # ProposedPaths
        model.modelcall,
        model.phasetype_expansion,
        selected_lambda,  # smoothing_parameters
        selected_edf)     # edf
end

"""
    compute_edf_markov(beta, lambda, penalty_config, model, books)

Compute effective degrees of freedom (EDF) for penalized Markov panel model.

The EDF measures the effective model complexity accounting for penalization.
It is computed as tr(A) where A = H_unpen * H_pen⁻¹, and H_pen = H_unpen + λS.

# Arguments
- `beta::Vector{Float64}`: Fitted parameter vector
- `lambda::Vector{Float64}`: Current smoothing parameter values
- `penalty_config::PenaltyConfig`: Penalty configuration
- `model::MultistateProcess`: The model
- `books::Tuple`: TPM bookkeeping structures

# Returns
NamedTuple with:
- `total::Float64`: Total effective degrees of freedom
- `per_term::Vector{Float64}`: EDF per penalty term

# Notes
Uses subject-level Hessians computed via `compute_subject_hessians`.
"""
function compute_edf_markov(beta::Vector{Float64}, lambda::Vector{Float64},
                            penalty_config::PenaltyConfig, model::MultistateProcess,
                            books::Tuple)
    # Compute subject-level Hessians for Markov panel data
    subject_hessians_ll = compute_subject_hessians(beta, model, books)
    
    # Validate subject Hessians for NaN/Inf
    nan_subjects = findall(H -> any(!isfinite, H), subject_hessians_ll)
    if !isempty(nan_subjects)
        @warn "$(length(nan_subjects)) subject Hessians contain NaN/Inf values. " *
              "Check for extreme parameter values or zero hazards. " *
              "Affected subjects: $(first(nan_subjects, 5))..." maxlog=3
    end
    
    # Aggregate to full Hessian (negative because we want Fisher information)
    # Note: For Markov panel, compute_subject_hessians returns Hessian of negative log-likelihood
    H_unpenalized = sum(subject_hessians_ll)
    
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
    
    # Total EDF = tr(A)
    total_edf = tr(A)
    
    return (total = total_edf, per_term = edf_vec)
end