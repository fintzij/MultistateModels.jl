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
# - Model-based and robust (IJ/jackknife) variance estimation
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

# Notes
- For Markov models, the likelihood involves matrix exponentials of the intensity matrix
- Censored state observations are handled via marginalization over possible states
"""
function _fit_markov_panel(model::MultistateModel; constraints = nothing, verbose = true, solver = nothing, adbackend::ADBackend = ForwardDiffBackend(), compute_vcov = true, vcov_threshold = true, compute_ij_vcov = true, compute_jk_vcov = false, loo_method = :direct, kwargs...)

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

        # get vcov
        if compute_vcov && (sol.retcode == ReturnCode.Success)
            # Compute subject-level gradients and Hessians to cache them for robust variance
            # This avoids redundant computation when computing robust variance estimates
            subject_grads_cache = compute_subject_gradients(sol.u, model, books)
            subject_hessians_cache = compute_subject_hessians(sol.u, model, books)
            
            # Aggregate Fisher information from subject Hessians
            nparams = length(sol.u)
            fishinf = zeros(Float64, nparams, nparams)
            for H_i in subject_hessians_cache
                fishinf .-= H_i
            end
            
            vcov = pinv(Symmetric(fishinf), atol = vcov_threshold ? (log(nsubj) * length(sol.u))^-2 : sqrt(eps(real(float(oneunit(eltype(fishinf)))))))
            vcov[isapprox.(vcov, 0.0; atol = eps(Float64))] .= 0.0
            vcov = Symmetric(vcov)
        else
            subject_grads_cache = nothing
            subject_hessians_cache = nothing
            vcov = nothing
        end
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

        # Check for parameters at box bounds (may affect variance estimates)
        warn_bound_parameters(sol.u, lb, ub)
        
        # Compute subject gradients and Hessians for variance estimation (can be done with constraints)
        if compute_vcov || compute_ij_vcov || compute_jk_vcov
            if verbose && (compute_ij_vcov || compute_jk_vcov)
                println("Computing subject gradients and Hessians for variance estimation...")
            end
            subject_grads_cache = compute_subject_gradients(sol.u, model, books)
            subject_hessians_cache = compute_subject_hessians(sol.u, model, books)
        else
            subject_grads_cache = nothing
            subject_hessians_cache = nothing
        end
        
        # Model-based vcov with constraints: use reduced Hessian approach
        if compute_vcov && !isnothing(subject_hessians_cache)
            # Build constraint function for Jacobian computation
            # Note: result array must be compatible with ForwardDiff Dual types
            n_cons = length(constraints.cons)
            cons_fn = x -> consfun_markov(zeros(eltype(x), n_cons), x, nothing)
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
                vcov = pinv(Symmetric(fishinf), atol = vcov_threshold ? (log(nsubj) * length(sol.u))^-2 : sqrt(eps(Float64)))
                vcov[isapprox.(vcov, 0.0; atol = eps(Float64))] .= 0.0
                vcov = Symmetric(vcov)
            end
        else
            vcov = nothing
        end
    end

    # compute loglikelihood at the estimate
    logliks = (loglik = -sol.objective, subj_lml = loglik_markov(sol.u, MPanelData(model, books); return_ll_subj = true))

    # compute robust variance estimates if requested
    # IJ variance uses outer product of gradients - works with or without constraints
    # JK variance uses leave-one-out - works with or without constraints
    ij_variance = nothing
    jk_variance = nothing
    subject_grads = nothing
    
    if compute_ij_vcov || compute_jk_vcov
        if verbose
            println("Computing robust variance estimates...")
        end
        robust_result = compute_robust_vcov(sol.u, model, books;
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

    # wrap results
    return MultistateModelFitted(
        model.data,
        parameters_fitted,
        model.bounds,  # Pass bounds from unfitted model
        logliks,
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
        model.surrogate,  # Unified surrogate field
        (solution = sol,), # ConvergenceRecords::Union{Nothing, NamedTuple}
        nothing, # ProposedPaths::Union{Nothing, NamedTuple}
        model.modelcall,
        model.phasetype_expansion,
        nothing,  # smoothing_parameters (not yet implemented for Markov panel)
        nothing); # edf
end
