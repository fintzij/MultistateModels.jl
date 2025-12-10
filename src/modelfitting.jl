"""
    fit(model::MultistateModel; constraints = nothing, verbose = true, compute_vcov = true, kwargs...)

Fit a multistate model to continuously observed (exact) data.

Uses Ipopt optimization for both constrained and unconstrained problems by default.

# Arguments
- `model::MultistateModel`: multistate model object with exact observation times
- `constraints`: parameter constraints (see Constraints documentation)
- `verbose::Bool=true`: print optimization messages
- `solver`: optimization solver (default: Ipopt for both constrained and unconstrained).
  See [Optimization Solvers](@ref) for available options.
- `parallel::Bool=false`: enable parallel likelihood evaluation using Julia threads.
  Recommended for large datasets (n > 500) with multiple cores available.
  Uses physical cores (not hyperthreads) for optimal performance.
- `nthreads::Union{Nothing,Int}=nothing`: number of threads for parallel execution.
  If nothing, auto-detects based on physical cores (not hyperthreads).
  Ignored when `parallel=false`.
- `compute_vcov::Bool=true`: compute model-based variance-covariance matrix (H⁻¹).
  Useful for diagnostics by comparing to robust variance.
- `vcov_threshold::Bool=true`: if true, uses adaptive threshold `1/√(log(n)·p)` for pseudo-inverse 
  of Fisher information; otherwise uses `√eps()`. Helps with near-singular Hessians.
- `compute_ij_vcov::Bool=true`: compute infinitesimal jackknife (sandwich/robust) variance H⁻¹KH⁻¹.
  **This is the recommended variance for inference** as it remains valid under model misspecification.
- `compute_jk_vcov::Bool=false`: compute jackknife variance ((n-1)/n)·Σᵢ ΔᵢΔᵢᵀ
- `loo_method::Symbol=:direct`: method for computing leave-one-out (LOO) perturbations Δᵢ.
  **Only affects jackknife variance** (not IJ), since IJ uses Var_IJ = H⁻¹KH⁻¹ directly
  without computing individual LOO estimates. Options:
  - `:direct`: approximate H₋ᵢ⁻¹ ≈ H⁻¹, so Δᵢ = H⁻¹gᵢ. O(p²n), faster when n >> p.
  - `:cholesky`: compute H₋ᵢ⁻¹ exactly via Cholesky rank-k downdates of H = LLᵀ.
    O(np³), more numerically stable for ill-conditioned problems.

# Returns
- `MultistateModelFitted`: fitted model object with estimates, standard errors, and diagnostics

# Variance Estimation (Default Behavior)

By default, both model-based and IJ (sandwich) variance are computed:
- **IJ/Sandwich variance** (`compute_ij_vcov=true`): **Primary for inference**. Robust variance
  H⁻¹KH⁻¹ that remains valid even under model misspecification.
- **Model-based variance** (`compute_vcov=true`): **For diagnostics**. Standard MLE variance H⁻¹,
  valid only under correct model specification. Compare to IJ variance to assess model adequacy.

Use `compare_variance_estimates(fitted)` to diagnose potential model misspecification.

# Parallelization

When `parallel=true`, likelihood evaluations during optimization use multiple threads.
This is beneficial for:
- Large datasets (n > 500 subjects/paths)
- Models with expensive per-path computations (splines, many covariates)
- Systems with multiple physical cores

Thread count is auto-detected based on physical cores (not hyperthreads) to avoid
overhead from hardware threads. Use `nthreads` to override.

Note: Gradient computation always uses sequential ForwardDiff for AD correctness.
Parallelization applies to objective function evaluation only.

# Example
```julia
# Default fit: computes both model-based and robust (IJ) variance
fitted = fit(model)

# Get robust standard errors for inference
robust_se = sqrt.(diag(get_ij_vcov(fitted)))

# Compare model-based and robust SEs to check model specification
result = compare_variance_estimates(fitted)
# If SE_robust >> SE_model, model may be misspecified

# Fit with model-based variance only (faster, but less robust)
fitted = fit(model; compute_ij_vcov=false)

# Enable parallel likelihood evaluation (useful for large datasets)
fitted = fit(model; parallel=true)

# Use exactly 4 threads for parallel evaluation
fitted = fit(model; parallel=true, nthreads=4)

# Use a custom solver (e.g., BFGS instead of L-BFGS)
using Optim
fitted = fit(model; solver=Optim.BFGS())
```

See also: [`get_vcov`](@ref), [`get_ij_vcov`](@ref), [`compare_variance_estimates`](@ref),
          [`recommended_nthreads`](@ref), [`get_physical_cores`](@ref)
"""
function fit(model::MultistateModel; constraints = nothing, verbose = true, solver = nothing, 
             parallel = false, nthreads = nothing,
             compute_vcov = true, vcov_threshold = true, compute_ij_vcov = true, 
             compute_jk_vcov = false, loo_method = :direct, kwargs...)

    # initialize array of sample paths

    samplepaths = extract_paths(model)

    # Initialize parameters to crude estimates for better optimizer starting point
    # This is especially important for Gompertz/Weibull hazards where poor initialization
    # can lead to degenerate local optima
    if isnothing(constraints)
        set_crude_init!(model)
    end

    # extract and initialize model parameters
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

    # parse constraints, or not, and solve
    if isnothing(constraints) 
        # Create likelihood function - parallel or sequential
        # Note: ForwardDiff uses the function passed to OptimizationFunction for both
        # objective and gradient. We pass the sequential version for AD correctness,
        # but create a parallel version for objective-only evaluation.
        if use_parallel
            # Create wrapper that uses parallel evaluation
            # ForwardDiff will still use sequential for gradient computation
            loglik_fn = (params, data) -> loglik_exact(params, data; neg=true, parallel=true)
        else
            loglik_fn = loglik_exact
        end
        
        # get estimates - use Ipopt for unconstrained
        optf = OptimizationFunction(loglik_fn, Optimization.AutoForwardDiff())
        prob = OptimizationProblem(optf, parameters, ExactData(model, samplepaths))

        # solve with user-specified solver or default Ipopt
        _solver = isnothing(solver) ? Ipopt.Optimizer() : solver
        sol  = solve(prob, _solver; print_level = 0)

        # rectify spline coefs
        if any(isa.(model.hazards, _SplineHazard))
            rectify_coefs!(sol.u, model)
        end
        
        # get vcov
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
            error("Constraints $badcons are violated at the initial parameter values.")
        end

        # Create likelihood function - parallel or sequential
        if use_parallel
            loglik_fn = (params, data) -> loglik_exact(params, data; neg=true, parallel=true)
        else
            loglik_fn = loglik
        end

        optf = OptimizationFunction(loglik_fn, Optimization.AutoForwardDiff(), cons = consfun_multistate)
        prob = OptimizationProblem(optf, parameters, ExactData(model, samplepaths), lcons = constraints.lcons, ucons = constraints.ucons)
        
        # solve with user-specified solver or default Ipopt
        _solver = isnothing(solver) ? Ipopt.Optimizer() : solver
        sol  = solve(prob, _solver; print_level = 0)

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

    # compute subject-level likelihood at the estimate
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
        model.modelcall)

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

# =============================================================================
# Phase 7: Fitting for PhaseTypeModel
# =============================================================================

"""
    fit(model::PhaseTypeModel; kwargs...)

Fit a phase-type model to panel data via maximum likelihood.

Phase-type models are Markov on the expanded (internal) state space, so fitting
uses the panel data likelihood with the expanded hazards. The fitted parameters
are then collapsed back to the user-facing phase-type parameterization.

# Arguments
- `model::PhaseTypeModel`: The phase-type model to fit
- `constraints`: Optional parameter constraints (on expanded parameters)
- `verbose::Bool = true`: Print progress messages
- `solver`: Optimization solver (default: Ipopt)
- `adbackend::ADBackend`: Automatic differentiation backend (default: ForwardDiff)
- `compute_vcov::Bool = true`: Compute variance-covariance matrix
- `vcov_threshold::Bool = true`: Apply threshold in vcov computation
- `compute_ij_vcov::Bool = true`: Compute infinitesimal jackknife variance
- `compute_jk_vcov::Bool = false`: Compute jackknife variance

# Returns
- `PhaseTypeFittedModel`: Fitted model with results in phase-type parameterization

# Algorithm
1. Initialize parameters using crude rates (respecting structure constraints)
2. Fit the model on the expanded state space using Markov panel likelihood
3. Collapse fitted parameters to phase-type (λ, μ) representation
4. Compute variance-covariance matrix

# Example
```julia
h12 = Hazard(@formula(0 ~ 1), "pt", 1, 2; n_phases=3, coxian_structure=:allequal)
h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
model = multistatemodel(h12, h23; data=data)
fitted = fit(model)

# Get parameters on natural scale
params = get_parameters(fitted)

# Access the fitted expanded model for diagnostics
fitted.fitted_expanded
```

See also: [`PhaseTypeModel`](@ref), [`PhaseTypeFittedModel`](@ref), [`get_parameters`](@ref)
"""
function fit(model::PhaseTypeModel; 
             constraints = nothing, 
             verbose = true, 
             solver = nothing, 
             adbackend::ADBackend = ForwardDiffBackend(), 
             compute_vcov = true, 
             vcov_threshold = true, 
             compute_ij_vcov = true, 
             compute_jk_vcov = false, 
             loo_method = :direct, 
             kwargs...)

    # Build TPM mapping - model.data is expanded data for PhaseTypeModel
    books = build_tpm_mapping(model.data)

    # Initialize parameters if not constrained
    if isnothing(constraints)
        set_crude_init!(model)
    end

    # Get flat parameters on expanded space
    parameters = get_parameters_flat(model)

    # Number of subjects
    nsubj = length(model.subjectindices)

    # Use model constraints if none provided and model has them
    if isnothing(constraints) && haskey(model.modelcall, :constraints) && !isnothing(model.modelcall.constraints)
        constraints = model.modelcall.constraints
    end

    # Warn if using reverse-mode AD for Markov models
    if adbackend isa MooncakeBackend
        @warn "MooncakeBackend may fail for Markov models due to LAPACK calls in matrix exponential. " *
              "Use ForwardDiffBackend() if you encounter errors."
    end

    # Select likelihood function based on AD backend
    loglik_fn = if adbackend isa EnzymeBackend || adbackend isa MooncakeBackend
        loglik_markov_functional
    else
        loglik_markov
    end

    # Create panel data object using expanded data and hazards
    panel_data = MPanelData(model, books)

    # Parse constraints and solve
    if isnothing(constraints)
        optf = OptimizationFunction(loglik_fn, get_optimization_ad(adbackend))
        prob = OptimizationProblem(optf, parameters, panel_data)
        
        _solver = isnothing(solver) ? Ipopt.Optimizer() : solver
        sol = solve(prob, _solver; print_level = 0)

        # Compute variance-covariance
        if compute_vcov && (sol.retcode == ReturnCode.Success)
            subject_grads_cache = compute_subject_gradients(sol.u, model, books)
            subject_hessians_cache = compute_subject_hessians(sol.u, model, books)
            
            nparams = length(sol.u)
            fishinf = zeros(Float64, nparams, nparams)
            for H_i in subject_hessians_cache
                fishinf .-= H_i
            end
            
            vcov_expanded = pinv(Symmetric(fishinf), 
                atol = vcov_threshold ? (log(nsubj) * length(sol.u))^-2 : sqrt(eps(real(float(oneunit(eltype(fishinf)))))))
            vcov_expanded[isapprox.(vcov_expanded, 0.0; atol = eps(Float64))] .= 0.0
            vcov_expanded = Symmetric(vcov_expanded)
        else
            subject_grads_cache = nothing
            subject_hessians_cache = nothing
            vcov_expanded = nothing
        end
    else
        _constraints = deepcopy(constraints)
        consfun_markov = parse_constraints(_constraints.cons, model.hazards; consfun_name = :consfun_markov)

        initcons = consfun_markov(zeros(length(constraints.cons)), parameters, nothing)
        badcons = findall(initcons .< constraints.lcons .|| initcons .> constraints.ucons)
        if length(badcons) > 0
            error("Constraints $badcons are violated at the initial parameter values.")
        end

        optf = OptimizationFunction(loglik_fn, get_optimization_ad(adbackend), cons = consfun_markov)
        prob = OptimizationProblem(optf, parameters, panel_data, lcons = constraints.lcons, ucons = constraints.ucons)
        
        _solver = isnothing(solver) ? Ipopt.Optimizer() : solver
        sol = solve(prob, _solver; print_level = 0)

        if compute_vcov == true
            @warn "No covariance matrix is returned when constraints are provided."
        end
        subject_grads_cache = nothing
        subject_hessians_cache = nothing
        vcov_expanded = nothing
    end

    # Update model with fitted parameters
    params_nested = unflatten(model.parameters.reconstructor, sol.u)
    
    # Compute natural scale for expanded parameters (family-aware transformation)
    params_natural_pairs = [
        hazname => extract_natural_vector(params_nested[hazname], model.hazards[idx].family)
        for (hazname, idx) in sort(collect(model.hazkeys), by = x -> x[2])
    ]
    params_natural = NamedTuple(params_natural_pairs)
    
    # Update expanded parameters in model (model.parameters is the expanded params)
    model.parameters = (
        flat = Vector{Float64}(sol.u),
        nested = params_nested,
        natural = params_natural,
        reconstructor = model.parameters.reconstructor  # Keep same reconstructor
    )

    # Sync back to user-facing parameters
    _sync_phasetype_parameters_to_original!(model)

    # Compute loglikelihood
    loglik_val = -sol.objective

    # Build logliks tuple for expanded model (for compatibility)
    logliks_expanded = (
        loglik = loglik_val, 
        subj_lml = loglik_markov(sol.u, panel_data; return_ll_subj = true)
    )

    # Build parameters for expanded model
    expanded_parameters_fitted = (
        flat = Vector{Float64}(sol.u),
        nested = params_nested,
        natural = params_natural,
        reconstructor = model.parameters.reconstructor  # Keep same reconstructor
    )

    # Compute robust variance if requested
    ij_variance = nothing
    jk_variance = nothing
    subject_grads = nothing
    
    if (compute_ij_vcov || compute_jk_vcov) && !isnothing(vcov_expanded) && isnothing(constraints)
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

    # Build modelcall with phase-type specific info for later access
    # This allows get_parameters to return user-facing phase-type parameters
    phasetype_modelcall = (
        # Standard modelcall fields
        hazards = model.modelcall.hazards,
        data = model.modelcall.data,
        constraints = model.modelcall.constraints,
        SubjectWeights = model.modelcall.SubjectWeights,
        ObservationWeights = model.modelcall.ObservationWeights,
        CensoringPatterns = model.modelcall.CensoringPatterns,
        EmissionMatrix = model.modelcall.EmissionMatrix,
        # Phase-type specific fields
        is_phasetype = true,
        mappings = model.mappings,
        original_parameters = model.original_parameters,
        original_tmat = model.original_tmat,
        original_data = model.original_data,
        convergence = sol.retcode == ReturnCode.Success,
        solution = sol
    )

    # Return MultistateModelFitted directly (no wrapper)
    # Phase-type info stored in modelcall for accessor functions
    return MultistateModelFitted(
        model.data,  # expanded data
        expanded_parameters_fitted,
        logliks_expanded,
        isnothing(vcov_expanded) ? nothing : Matrix(vcov_expanded),
        isnothing(ij_variance) ? nothing : Matrix(ij_variance),
        isnothing(jk_variance) ? nothing : Matrix(jk_variance),
        subject_grads,
        model.hazards,
        model.totalhazards,
        model.mappings.expanded_tmat,
        model.emat,
        model.hazkeys,
        model.subjectindices,
        model.SubjectWeights,
        model.ObservationWeights,
        model.CensoringPatterns,
        model.markovsurrogate,
        (solution = sol,),
        nothing,
        phasetype_modelcall
    )
end

"""
    fit(model::MultistateMarkovModel; constraints = nothing, verbose = true, compute_vcov = true, kwargs...)

Fit a Markov multistate model to interval-censored or panel data.

Uses Ipopt optimization for both constrained and unconstrained problems by default.

# Arguments
- `model::Union{MultistateMarkovModel, MultistateMarkovModelCensored}`: Markov model with panel observations
- `constraints`: parameter constraints (see Constraints documentation)
- `verbose::Bool=true`: print optimization messages
- `solver`: optimization solver (default: Ipopt for both constrained and unconstrained).
  See [Optimization Solvers](@ref) for available options.
- `adbackend::ADBackend=ForwardDiffBackend()`: automatic differentiation backend.
  - `ForwardDiffBackend()`: forward-mode AD (default, required for Markov models)
  - `EnzymeBackend()`: reverse-mode AD (Julia 1.12 not supported)
  - `MooncakeBackend()`: reverse-mode AD (fails on Markov models - see below)
  
  **Important:** Markov panel likelihoods require `ForwardDiffBackend()` because matrix
  exponential differentiation uses LAPACK calls that reverse-mode AD cannot handle.
  ChainRules.jl has an rrule for exp(::Matrix), but it also uses LAPACK internally.
  
- `compute_vcov::Bool=true`: compute model-based variance-covariance matrix (for diagnostics)
- `vcov_threshold::Bool=true`: use adaptive threshold for pseudo-inverse of Fisher information
- `compute_ij_vcov::Bool=true`: compute infinitesimal jackknife (sandwich/robust) variance.
  **This is the recommended variance for inference.**
- `compute_jk_vcov::Bool=false`: compute jackknife variance
- `loo_method::Symbol=:direct`: method for LOO perturbations Δᵢ = θ̂₋ᵢ - θ̂ (jackknife only, not IJ):
  - `:direct`: Δᵢ = H⁻¹gᵢ (faster, O(p²n))
  - `:cholesky`: exact H₋ᵢ⁻¹ via Cholesky downdates (stable, O(np³))

# Returns
- `MultistateModelFitted`: fitted model with estimates and variance matrices

# Variance Estimation (Default Behavior)

By default, both model-based and IJ (sandwich) variance are computed:
- **IJ variance** (`compute_ij_vcov=true`): Primary for inference (robust to misspecification)
- **Model-based** (`compute_vcov=true`): For diagnostics (compare SE ratios)

# Notes
- For Markov models, the likelihood involves matrix exponentials of the intensity matrix
- Censored state observations are handled via marginalization over possible states
- IJ/JK variance estimation works for both censored and uncensored Markov models

# Example
```julia
# Default fit: robust and model-based variance computed
fitted = fit(markov_model)

# Use robust SEs for inference
robust_se = sqrt.(diag(get_ij_vcov(fitted)))

# Diagnose model specification
result = compare_variance_estimates(fitted)

# Use a custom solver
using Optim
fitted = fit(markov_model; solver=Optim.BFGS())

# Use Enzyme for reverse-mode AD (useful for large models)
fitted = fit(markov_model; adbackend=EnzymeBackend())

# Use Mooncake for reverse-mode AD (works on all Julia versions)
fitted = fit(markov_model; adbackend=MooncakeBackend())
```

See also: [`fit(::MultistateModel)`](@ref), [`compare_variance_estimates`](@ref)
"""
function fit(model::Union{MultistateMarkovModel,MultistateMarkovModelCensored}; constraints = nothing, verbose = true, solver = nothing, adbackend::ADBackend = ForwardDiffBackend(), compute_vcov = true, vcov_threshold = true, compute_ij_vcov = true, compute_jk_vcov = false, loo_method = :direct, kwargs...)

    # containers for bookkeeping TPMs
    books = build_tpm_mapping(model.data)

    # Initialize parameters to crude estimates for better optimizer starting point
    # This is especially important for Gompertz/Weibull hazards where poor initialization
    # can lead to degenerate local optima
    if isnothing(constraints)
        set_crude_init!(model)
    end

    # extract and initialize model parameters
    # Phase 3: Use ParameterHandling.jl flat parameters (log scale)
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

    # Select likelihood function based on AD backend
    # Enzyme and Mooncake (reverse-mode) require non-mutating code
    loglik_fn = if adbackend isa EnzymeBackend || adbackend isa MooncakeBackend
        loglik_markov_functional
    else
        loglik_markov
    end

    # parse constraints, or not, and solve
    if isnothing(constraints)
        # get estimates - use Ipopt for unconstrained
        optf = OptimizationFunction(loglik_fn, get_optimization_ad(adbackend))
        prob = OptimizationProblem(optf, parameters, MPanelData(model, books))
        
        # solve with user-specified solver or default Ipopt
        _solver = isnothing(solver) ? Ipopt.Optimizer() : solver
        sol  = solve(prob, _solver; print_level = 0)

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
            error("Constraints $badcons are violated at the initial parameter values.")
        end

        optf = OptimizationFunction(loglik_fn, get_optimization_ad(adbackend), cons = consfun_markov)
        prob = OptimizationProblem(optf, parameters, MPanelData(model, books), lcons = constraints.lcons, ucons = constraints.ucons)
        
        # solve with user-specified solver or default Ipopt
        _solver = isnothing(solver) ? Ipopt.Optimizer() : solver
        sol  = solve(prob, _solver; print_level = 0)

        # no hessian when there are constraints
        if compute_vcov == true
            @warn "No covariance matrix is returned when constraints are provided."
        end
        subject_grads_cache = nothing
        subject_hessians_cache = nothing
        vcov = nothing
    end

    # compute loglikelihood at the estimate
    logliks = (loglik = -sol.objective, subj_lml = loglik_markov(sol.u, MPanelData(model, books); return_ll_subj = true))

    # compute robust variance estimates if requested
    ij_variance = nothing
    jk_variance = nothing
    subject_grads = nothing
    
    if (compute_ij_vcov || compute_jk_vcov) && !isnothing(vcov) && isnothing(constraints)
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
        model.markovsurrogate,
        (solution = sol,), # ConvergenceRecords::Union{Nothing, NamedTuple}
        nothing, # ProposedPaths::Union{Nothing, NamedTuple}
        model.modelcall);
end


"""
    fit(model::Union{MultistateSemiMarkovModel, MultistateSemiMarkovModelCensored}; kwargs...)

Fit a semi-Markov model to panel data via Monte Carlo EM (MCEM).

Uses Ipopt optimization for both constrained and unconstrained M-steps by default.

!!! note "Surrogate Required"
    MCEM requires a Markov surrogate for importance sampling proposals. 
    You must call `set_surrogate!(model)` or use `surrogate=:markov` in 
    `multistatemodel()` before fitting.

# Algorithm

The MCEM algorithm iterates between:
1. **E-step**: Sample latent paths via importance sampling from a Markov surrogate
2. **M-step**: Maximize the expected complete-data log-likelihood

Convergence is assessed using the ascent-based stopping rule from Caffo et al. (2005),
with Pareto-smoothed importance sampling (PSIS) for stable weight estimation.

# References

- Morsomme, R., Liang, C. J., Mateja, A., Follmann, D. A., O'Brien, M. P., Wang, C.,
  & Fintzi, J. (2025). Assessing treatment efficacy for interval-censored endpoints
  using multistate semi-Markov models fit to multiple data streams. Biostatistics,
  26(1), kxaf038. https://doi.org/10.1093/biostatistics/kxaf038
- Wei, G. C., & Tanner, M. A. (1990). A Monte Carlo implementation of the 
  EM algorithm and the poor man's data augmentation algorithms. JASA, 85(411), 699-704.
- Caffo, B. S., Jank, W., & Jones, G. L. (2005). Ascent-based Monte Carlo 
  expectation-maximization. JRSS-B, 67(2), 235-251.
- Vehtari, A., Simpson, D., Gelman, A., Yao, Y., & Gabry, J. (2024). 
  Pareto Smoothed Importance Sampling. JMLR, 25(72), 1-58.
- Varadhan, R., & Roland, C. (2008). Simple and globally convergent methods 
  for accelerating the convergence of any EM algorithm. Scand. J. Stat., 35(2), 335-353.

# Arguments

**Model and constraints:**
- `model::Union{MultistateSemiMarkovModel, MultistateSemiMarkovModelCensored}`: semi-Markov model
- `constraints`: parameter constraints tuple

**Optimization:**
- `solver`: optimization solver for M-step (default: Ipopt for both constrained and unconstrained).
  See [Optimization Solvers](@ref) for available options.

**MCEM algorithm control:**
- `maxiter::Int=100`: maximum MCEM iterations
- `tol::Float64=1e-2`: tolerance for MLL change in stopping rule
- `ascent_threshold::Float64=0.1`: standard normal quantile for ascent lower bound
- `stopping_threshold::Float64=0.1`: standard normal quantile for stopping criterion
- `ess_increase::Float64=2.0`: ESS inflation factor when more paths needed
- `ess_target_initial::Int=50`: initial effective sample size target per subject
- `max_ess::Int=10000`: maximum ESS before stopping for non-convergence
- `max_sampling_effort::Int=20`: maximum factor of ESS for additional path sampling
- `npaths_additional::Int=10`: increment for additional paths when augmenting
- `block_hessian_speedup::Float64=2.0`: minimum speedup factor to use block-diagonal Hessian
- `acceleration::Symbol=:none`: acceleration method for MCEM. Options:
  - `:none` (default): standard MCEM without acceleration
  - `:squarem`: SQUAREM acceleration (Varadhan & Roland, 2008), applies quasi-Newton 
    extrapolation every 2 iterations to speed up convergence

**Output control:**
- `verbose::Bool=true`: print progress messages
- `return_convergence_records::Bool=true`: save iteration history
- `return_proposed_paths::Bool=false`: save latent paths and importance weights

**Variance estimation:**
- `compute_vcov::Bool=true`: compute model-based variance via Louis's identity (for diagnostics)
- `vcov_threshold::Bool=true`: use adaptive threshold for pseudo-inverse
- `compute_ij_vcov::Bool=true`: compute IJ/sandwich variance (robust, **recommended for inference**)
- `compute_jk_vcov::Bool=false`: compute jackknife variance
- `loo_method::Symbol=:direct`: method for LOO perturbations Δᵢ (jackknife only, not IJ):
  - `:direct`: Δᵢ = H⁻¹gᵢ (faster)
  - `:cholesky`: exact H₋ᵢ⁻¹ via Cholesky downdates (stable)

# Returns
- `MultistateModelFitted`: fitted model with MCEM solution and variance estimates

# Variance Estimation (Default Behavior)

By default, both model-based and IJ (sandwich) variance are computed:
- **IJ variance** (`compute_ij_vcov=true`): Primary for inference (robust to misspecification)
- **Model-based** (`compute_vcov=true`): For diagnostics via Louis's identity

For semi-Markov models, the observed Fisher information is computed using Louis's identity,
which accounts for the missing data (unobserved paths):
```
I_obs = E[I_comp | Y] - Var[S_comp | Y]
```

# Example
```julia
# Default fit: robust and model-based variance computed
fitted = fit(semimarkov_model; ess_target_initial=100, verbose=true)

# Use robust SEs for inference
robust_se = sqrt.(diag(get_ij_vcov(fitted)))

# Diagnose model specification
result = compare_variance_estimates(fitted)

# Check convergence
records = get_convergence_records(fitted)

# Use a custom solver for the M-step
using Optim
fitted = fit(semimarkov_model; solver=Optim.BFGS())

# Use SQUAREM acceleration for faster convergence
fitted = fit(semimarkov_model; acceleration=:squarem)
```

See also: [`fit(::MultistateModel)`](@ref), [`compare_variance_estimates`](@ref)
"""
function fit(model::Union{MultistateSemiMarkovModel, MultistateSemiMarkovModelCensored}; proposal::Union{Symbol, ProposalConfig} = :auto, constraints = nothing, solver = nothing, maxiter = 100, tol = 1e-2, ascent_threshold = 0.1, stopping_threshold = 0.1, ess_increase = 2.0, ess_target_initial = 50, max_ess = 10000, max_sampling_effort = 20, npaths_additional = 10, block_hessian_speedup = 2.0, acceleration::Symbol = :none, verbose = true, return_convergence_records = true, return_proposed_paths = false, compute_vcov = true, vcov_threshold = true, compute_ij_vcov = true, compute_jk_vcov = false, loo_method = :direct, kwargs...)

    # Validate acceleration parameter
    if acceleration ∉ (:none, :squarem)
        error("acceleration must be :none or :squarem, got :$acceleration")
    end
    use_squarem = acceleration === :squarem
    
    if verbose && use_squarem
        println("Using SQUAREM acceleration for MCEM.\n")
    end

    # Resolve proposal configuration
    proposal_config = resolve_proposal_config(proposal, model)
    use_phasetype = proposal_config.type === :phasetype
    
    if verbose && use_phasetype
        println("Using phase-type proposal for MCEM importance sampling.\n")
    end

    # copy of data
    data_original = deepcopy(model.data)

    # Use model constraints if none provided and model has them
    if isnothing(constraints) && haskey(model.modelcall, :constraints) && !isnothing(model.modelcall.constraints)
        constraints = model.modelcall.constraints
    end

    # check that constraints for the initial values are satisfied
    if !isnothing(constraints)
        # create constraint function and check that constraints are satisfied at the initial values
        _constraints = deepcopy(constraints)
        consfun_semimarkov = parse_constraints(_constraints.cons, model.hazards; consfun_name = :consfun_semimarkov)

        # Phase 3: Use ParameterHandling.jl flat parameters for constraint check
        initcons = consfun_semimarkov(zeros(length(constraints.cons)), get_parameters_flat(model), nothing)
        badcons = findall(initcons .< constraints.lcons .|| initcons .> constraints.ucons)
        if length(badcons) > 0
            error("Constraints $badcons are violated at the initial parameter values.")
        end
    end

    # check that max_sampling_effort is greater than 1
    if max_sampling_effort <= 1
        error("max_sampling_effort must be greater than 1.")
    end

    # check that ess_increase is greater than 1
    if ess_increase <= 1
        error("ess_increase must be greater than 1.")
    end

    # throw a warning if trying to fit a spline model where the degree is 0 for all splines
    if all(map(x -> (isa(x, _MarkovHazard) | (isa(x, _SplineHazard) && (x.degree == 0) && (length(x.knots) == 2))), model.hazards))
        error("Attempting to fit a time-homogeneous Markov model via MCEM. Recode as exponential hazards and refit.")
    end

    # MCEM initialization
    keep_going = true
    iter = 0
    is_converged = false

    # number of subjects
    nsubj = length(model.subjectindices)

    # identify absorbing states
    absorbingstates = findall(map(x -> all(x .== 0), eachrow(model.tmat)))

    # extract and initialize model parameters
    # Phase 3: Use ParameterHandling.jl flat parameters (log scale)
    params_cur = get_parameters_flat(model)

    # initialize ess target
    ess_target = ess_target_initial

    # containers for latent sample paths, proposal and target log likelihoods, importance sampling weights
    ess_cur = zeros(nsubj)
    psis_pareto_k = zeros(nsubj)

    # initialize containers
    samplepaths     = [sizehint!(Vector{SamplePath}(), ess_target_initial * max_sampling_effort * 20) for i in 1:nsubj]

    # surrogate log likelihood
    loglik_surrog   = [sizehint!(Vector{Float64}(undef, 0), 
        ess_target_initial * max_sampling_effort * 2) for i in 1:nsubj]

    # target log likelihood - current parameters
    loglik_target_cur  = [sizehint!(Vector{Float64}(undef, 0), 
        ess_target_initial * max_sampling_effort * 2) for i in 1:nsubj]

    # target log likelihood - proposed parameters
    loglik_target_prop = [sizehint!(Vector{Float64}(undef, 0), 
        ess_target_initial * max_sampling_effort * 2) for i in 1:nsubj]

    # Log (unnormalize) importance weights
    _logImportanceWeights  = [sizehint!(Vector{Float64}(undef, 0), 
        ess_target_initial * max_sampling_effort * 2) for i in 1:nsubj]

    # exponentiated and normalized importance weights
    ImportanceWeights  = [sizehint!(Vector{Float64}(undef, 0), 
        ess_target_initial * max_sampling_effort * 2) for i in 1:nsubj]

    # make fbmats if necessary
    if any(model.data.obstype .> 2)
        fbmats = build_fbmats(model)
    else
        fbmats = nothing
    end

    # containers for traces
    mll_trace = Vector{Float64}() # marginal loglikelihood
    ess_trace = ElasticArray{Float64, 2}(undef, nsubj, 0) # effective sample size (one per subject)
    path_count_trace = ElasticArray{Int, 2}(undef, nsubj, 0) # actual path counts (one per subject)
    # Phase 3: Use ParameterHandling.jl flat parameter length
    parameters_trace = ElasticArray{Float64, 2}(undef, length(get_parameters_flat(model)), 0) # parameter estimates

    # Require a pre-built Markov surrogate for MCEM
    # Users should call set_surrogate!(model) or use surrogate=:markov in multistatemodel() beforehand
    if isnothing(model.markovsurrogate)
        error("MCEM requires a Markov surrogate. Call `set_surrogate!(model)` or use `surrogate=:markov` in `multistatemodel()` before fitting.")
    end
    
    # Check if surrogate needs to be fitted (not yet fitted)
    # This happens when surrogate=:markov is used with fit_surrogate=false
    if !model.markovsurrogate.fitted
        if verbose
            println("Markov surrogate not yet fitted. Fitting via MLE...")
        end
        # Fit the surrogate via set_surrogate! with MLE method
        set_surrogate!(model; type=:markov, method=:mle, verbose=verbose)
    end
    
    markov_surrogate = model.markovsurrogate
    if verbose
        println("Using model's Markov surrogate for MCEM.\n")
    end

    # Build phase-type surrogate if requested
    phasetype_surrogate = nothing
    tpm_book_ph = nothing
    hazmat_book_ph = nothing
    fbmats_ph = nothing
    emat_ph = nothing
    
    if use_phasetype
        phasetype_surrogate = fit_phasetype_surrogate(model, markov_surrogate; 
                                                       config=proposal_config, verbose=verbose)
    end
    
    # Use Markov surrogate for compatibility (phase-type uses it for sampling infrastructure)
    surrogate = markov_surrogate

     # containers for bookkeeping TPMs
     books = build_tpm_mapping(model.data)    

     # transition probability objects for Markov surrogate
     hazmat_book_surrogate = build_hazmat_book(Float64, model.tmat, books[1])
     tpm_book_surrogate = build_tpm_book(Float64, model.tmat, books[1])
 
     # allocate memory for matrix exponential
     cache = ExponentialUtilities.alloc_mem(similar(hazmat_book_surrogate[1]), ExpMethodGeneric())

    # Solve Kolmogorov equations for TPMs
    # Get natural-scale surrogate parameters for hazard evaluation (family-aware)
    surrogate_pars = get_hazard_params(surrogate.parameters, surrogate.hazards)
    for t in eachindex(books[1])
        # compute the transition intensity matrix
        compute_hazmat!(hazmat_book_surrogate[t], surrogate_pars, surrogate.hazards, books[1][t], model.data)
        # compute transition probability matrices
        compute_tmat!(tpm_book_surrogate[t], hazmat_book_surrogate[t], books[1][t], cache)
    end

    # Build phase-type infrastructure if using phase-type proposals
    # For exact observations, we need to expand the data to properly express
    # phase uncertainty during sojourn times
    expanded_ph_data = nothing
    ph_censoring_patterns = nothing
    ph_original_row_map = nothing
    ph_subjectindices = nothing
    expanded_ph_tpm_map = nothing  # tpm_map for expanded data
    
    if use_phasetype
        # Check if data needs expansion (has exact observations)
        if needs_data_expansion_for_phasetype(model.data)
            n_states = size(model.tmat, 1)
            expansion_result = expand_data_for_phasetype(model.data, n_states)
            expanded_ph_data = expansion_result.expanded_data
            ph_censoring_patterns = expansion_result.censoring_patterns
            ph_original_row_map = expansion_result.original_row_map
            ph_subjectindices = compute_expanded_subject_indices(expanded_ph_data)
            
            if verbose
                n_orig = nrow(model.data)
                n_exp = nrow(expanded_ph_data)
                println("  Expanded data for phase-type: $n_orig → $n_exp rows")
            end
        end
        
        # Build TPM book using original or expanded data
        data_for_ph = isnothing(expanded_ph_data) ? model.data : expanded_ph_data
        
        # Rebuild books for expanded data if needed
        if !isnothing(expanded_ph_data)
            books_ph = build_tpm_mapping(expanded_ph_data)
            expanded_ph_tpm_map = books_ph[2]  # Save tpm_map for sampling
            tpm_book_ph, hazmat_book_ph = build_phasetype_tpm_book(phasetype_surrogate, books_ph, expanded_ph_data)
        else
            tpm_book_ph, hazmat_book_ph = build_phasetype_tpm_book(phasetype_surrogate, books, model.data)
        end
        
        # Build fbmats with correct sizes (using expanded subject indices if available)
        subj_inds_for_ph = isnothing(ph_subjectindices) ? model.subjectindices : ph_subjectindices
        fbmats_ph = build_fbmats_phasetype_with_indices(subj_inds_for_ph, phasetype_surrogate)
        
        emat_ph = build_phasetype_emat_expanded(model, phasetype_surrogate;
                                                 expanded_data = expanded_ph_data,
                                                 censoring_patterns = ph_censoring_patterns)
    end

    # compute normalizing constant of proposal distribution
    # For Markov proposal: this is the log-likelihood under the Markov surrogate
    # For phase-type proposal: compute marginal likelihood via forward algorithm
    #   This is r(Y|θ') in the importance sampling formula:
    #   log f̂(Y|θ) = log r(Y|θ') + Σᵢ log(mean(νᵢ))
    if use_phasetype
        NormConstantProposal = compute_phasetype_marginal_loglik(
            model, phasetype_surrogate, emat_ph;
            expanded_data = expanded_ph_data,
            expanded_subjectindices = ph_subjectindices)
    else
        NormConstantProposal = compute_markov_marginal_loglik(model, markov_surrogate)
    end

    # draw sample paths until the target ess is reached 
    if verbose  println("Initializing sample paths ...\n") end
    DrawSamplePaths!(model; 
        ess_target = ess_target, 
        ess_cur = ess_cur, 
        max_sampling_effort = max_sampling_effort,
        samplepaths = samplepaths, 
        loglik_surrog = loglik_surrog, 
        loglik_target_prop = loglik_target_prop, 
        loglik_target_cur = loglik_target_cur, 
        _logImportanceWeights = _logImportanceWeights, 
        ImportanceWeights = ImportanceWeights, 
        tpm_book_surrogate = tpm_book_surrogate, 
        hazmat_book_surrogate = hazmat_book_surrogate, 
        books = books, 
        npaths_additional = npaths_additional, 
        params_cur = params_cur, 
        surrogate = surrogate, 
        psis_pareto_k = psis_pareto_k,
        fbmats = fbmats,
        absorbingstates = absorbingstates,
        # Phase-type infrastructure (nothing if not using)
        phasetype_surrogate = phasetype_surrogate,
        tpm_book_ph = tpm_book_ph,
        hazmat_book_ph = hazmat_book_ph,
        fbmats_ph = fbmats_ph,
        emat_ph = emat_ph,
        # Expanded data infrastructure (nothing if not using/not needed)
        expanded_ph_data = expanded_ph_data,
        expanded_ph_subjectindices = ph_subjectindices,
        expanded_ph_tpm_map = expanded_ph_tpm_map,
        ph_original_row_map = ph_original_row_map)
    
    # get current estimate of marginal log likelihood
    mll_cur = mcem_mll(loglik_target_cur, ImportanceWeights, model.SubjectWeights)

    # generate optimization problem
    if isnothing(constraints)
        optf = OptimizationFunction(loglik, Optimization.AutoForwardDiff())
        prob = OptimizationProblem(optf, params_cur, SMPanelData(model, samplepaths, ImportanceWeights))
    else
        optf = OptimizationFunction(loglik, Optimization.AutoForwardDiff(), cons = consfun_semimarkov)
        prob = OptimizationProblem(optf, params_cur, SMPanelData(model, samplepaths, ImportanceWeights), lcons = constraints.lcons, ucons = constraints.ucons)
    end

    # print output
    if verbose
        println("Initial target ESS: $(round(ess_target;digits=2)) per-subject")
        println("Initial range of the number of sample paths per-subject: [$(ceil(ess_target)), $(maximum(length.(samplepaths)))]")
        println("Initial estimate of the marginal log-likelihood, Q: $(round(mll_cur;digits=3))")
        println("Initial estimate of the log marginal likelihood: $(round(compute_loglik(model, loglik_surrog, loglik_target_cur, NormConstantProposal).loglik;digits=3))\n")
        println("Starting Monte Carlo EM...\n")
    end
    
    # initialize MCEM
    mll_prop = mll_cur
    mll_change = 0.0
    ase = 0.0
    ascent_lb = 0.0
    ascent_ub = 0.0

    # Initialize SQUAREM state if using acceleration
    squarem_state = use_squarem ? SquaremState(length(params_cur)) : nothing

    # start algorithm
    while keep_going

        # increment the iteration
        iter += 1
        
        # =====================================================================
        # SQUAREM: Save θ₀ at start of cycle (every 2 iterations)
        # =====================================================================
        if use_squarem && squarem_state.step == 0
            squarem_state.θ0 .= params_cur
            squarem_state.step = 1
        end
        
        # solve M-step: use user-specified solver or default Ipopt
        if isnothing(constraints)
            _solver = isnothing(solver) ? Ipopt.Optimizer() : solver
            params_prop_optim = solve(remake(prob, u0 = Vector(params_cur), p = SMPanelData(model, samplepaths, ImportanceWeights)), _solver; print_level = 0)
        else
            _solver = isnothing(solver) ? Ipopt.Optimizer() : solver
            params_prop_optim = solve(remake(prob, u0 = Vector(params_cur), p = SMPanelData(model, samplepaths, ImportanceWeights)), _solver; print_level = 0)
        end
        params_prop = params_prop_optim.u

        # calculate the log likelihoods for the proposed parameters
        loglik!(params_prop, loglik_target_prop, SMPanelData(model, samplepaths, ImportanceWeights))
        
        # calculate the marginal log likelihood 
        mll_cur  = mcem_mll(loglik_target_cur , ImportanceWeights, model.SubjectWeights)
        mll_prop = mcem_mll(loglik_target_prop, ImportanceWeights, model.SubjectWeights)

        # compute the ALB and AUB
        if params_prop != params_cur

            # change in mll
            mll_change = mll_prop - mll_cur
    
            # calculate the ASE for ΔQ
            ase = mcem_ase(loglik_target_prop, loglik_target_cur, ImportanceWeights, model.SubjectWeights)
    
            # calculate the lower bound for ΔQ
            ascent_lb = quantile(Normal(mll_change, ase), ascent_threshold)
            ascent_ub = quantile(Normal(mll_change, ase), 1-stopping_threshold)
        else
            loglik_target_prop = loglik_target_cur
            mll_prop = mll_cur
            mll_change = 0
            ase = 0
            ascent_lb = 0
            ascent_ub = 0
            is_converged = true
        end

        # =====================================================================
        # SQUAREM: Apply acceleration every 2 iterations
        # =====================================================================
        if use_squarem && !is_converged
            if squarem_state.step == 1
                # After first EM step: save θ₁, continue to second step
                squarem_state.θ1 .= params_prop
                squarem_state.step = 2
                
                # Standard update for this iteration
                params_cur        = deepcopy(params_prop)
                mll_cur           = deepcopy(mll_prop)
                loglik_target_cur = deepcopy(loglik_target_prop)
                
            elseif squarem_state.step == 2
                # After second EM step: θ₂ = params_prop, now compute acceleration
                squarem_state.θ2 .= params_prop
                
                # Compute step length and acceleration vectors
                α, r, v = squarem_step_length(squarem_state.θ0, squarem_state.θ1, squarem_state.θ2)
                squarem_state.α = α
                
                if α == -1.0
                    # No acceleration possible (v too small), use standard EM update
                    params_cur        = deepcopy(params_prop)
                    mll_cur           = deepcopy(mll_prop)
                    loglik_target_cur = deepcopy(loglik_target_prop)
                    squarem_state.n_fallbacks += 1
                    if verbose
                        println("  [SQUAREM: no acceleration (Δ too small), using standard EM]\n")
                    end
                else
                    # Compute accelerated parameters
                    params_acc = squarem_accelerate(squarem_state.θ0, r, v, α)
                    
                    # Evaluate likelihood at accelerated point
                    loglik!(params_acc, loglik_target_prop, SMPanelData(model, samplepaths, ImportanceWeights))
                    mll_acc = mcem_mll(loglik_target_prop, ImportanceWeights, model.SubjectWeights)
                    
                    # Decide whether to accept acceleration
                    # Use mll at θ₀ as reference (start of SQUAREM cycle)
                    mll_θ0 = mcem_mll(loglik_target_cur, ImportanceWeights, model.SubjectWeights)
                    
                    if squarem_should_accept(mll_acc, mll_prop, mll_θ0)
                        # Accept accelerated step
                        params_cur = params_acc
                        mll_cur = mll_acc
                        # loglik_target_cur already holds logliks at params_acc from loglik! call above
                        loglik_target_cur = deepcopy(loglik_target_prop)
                        squarem_state.n_accelerations += 1
                        
                        # Update mll_change for reporting
                        mll_change = mll_acc - mll_θ0
                        
                        if verbose
                            println("  [SQUAREM: acceleration accepted, α=$(round(α;sigdigits=3))]\n")
                        end
                    else
                        # Fallback to standard EM update (θ₂)
                        loglik!(squarem_state.θ2, loglik_target_prop, SMPanelData(model, samplepaths, ImportanceWeights))
                        params_cur        = deepcopy(squarem_state.θ2)
                        mll_cur           = deepcopy(mll_prop)
                        loglik_target_cur = deepcopy(loglik_target_prop)
                        squarem_state.n_fallbacks += 1
                        if verbose
                            println("  [SQUAREM: acceleration rejected (mll decreased), using θ₂]\n")
                        end
                    end
                end
                
                # Reset SQUAREM cycle
                squarem_state.step = 0
            end
        else
            # Standard MCEM (no SQUAREM) or converged
            # increment the current values of the parameter, marginal log likelihood, target loglik
            params_cur        = deepcopy(params_prop)
            mll_cur           = deepcopy(mll_prop)
            loglik_target_cur = deepcopy(loglik_target_prop)
        end

        # increment log importance weights, importance weights, effective sample size and pareto shape parameter
        ComputeImportanceWeightsESS!(loglik_target_cur, loglik_surrog, _logImportanceWeights, ImportanceWeights, ess_cur, ess_target, psis_pareto_k)

        # Note: psis_pareto_k > 0.7 indicates unreliable importance weights (Vehtari et al., 2024)
        # The values are stored in the returned convergence_records for diagnostic purposes

        # print update
        if verbose
            println("Iteration: $iter")
            println("Target ESS: $(round(ess_target;digits=2)) per-subject")
            println("Range of the number of sampled paths per-subject: [$(ceil(ess_target)), $(max(length.(samplepaths)...))]")
            println("Estimate of the marginal log-likelihood, Q: $(round(mll_cur;digits=3))")
            println("Gain in marginal log-likelihood, ΔQ: $(round(mll_change;sigdigits=3))")
            println("MCEM asymptotic standard error: $(round(ase;sigdigits=3))")
            println("Ascent lower and upper bound: [$(round(ascent_lb; sigdigits=3)), $(round(ascent_ub; sigdigits=3))]")
            println("Estimate of the log marginal likelihood, l(θ): $(round(compute_loglik(model, loglik_surrog, loglik_target_cur, NormConstantProposal).loglik;digits=3))")
            if use_squarem
                println("SQUAREM: $(squarem_state.n_accelerations) accelerations, $(squarem_state.n_fallbacks) fallbacks\n")
            else
                println()
            end
        end

        # save marginal log likelihood, parameters and effective sample size
        append!(parameters_trace, params_cur)
        push!(mll_trace, mll_cur)
        append!(ess_trace, ess_cur)
        append!(path_count_trace, length.(samplepaths))

        # check for convergence
        if ascent_ub < tol
            is_converged = true
            if verbose  println("The MCEM algorithm has converged.\n") end
            break
        end

        # check if limits are reached
        if ess_target > max_ess  
            @warn "The maximum target ESS ($ess_target) has been reached.\n"; break 
        end
        
        if iter >= maxiter  
            @warn "The maximum number of iterations ($maxiter) has been reached.\n"; break 
        end

        # increase ess is necessary
        if ascent_lb < 0
            # increase the target ess for the factor ess_increase
            ess_target = ceil(ess_increase * ess_target)
            if verbose  println("Target ESS is increased to $ess_target, because ascent lower bound < 0.\n") end

            # Note: No need to clear arrays - DrawSamplePaths! will append only if ess_cur[i] < ess_target
            # The bug was in ComputeImportanceWeightsESS! incorrectly setting ess_cur[i] = ess_target
            # instead of the actual path count for uniform weights. This has been fixed.
        end

        # ensure that ess per person is sufficient
        DrawSamplePaths!(model; 
            ess_target = ess_target, 
            ess_cur = ess_cur, 
            max_sampling_effort = max_sampling_effort,
            samplepaths = samplepaths, 
            loglik_surrog = loglik_surrog, 
            loglik_target_prop = loglik_target_prop, 
            loglik_target_cur = loglik_target_cur, 
            _logImportanceWeights = _logImportanceWeights, 
            ImportanceWeights = ImportanceWeights,   
            tpm_book_surrogate = tpm_book_surrogate,   
            hazmat_book_surrogate = hazmat_book_surrogate, 
            books = books, 
            npaths_additional = npaths_additional, 
            params_cur = params_cur, 
            surrogate = surrogate,
            psis_pareto_k = psis_pareto_k,
            fbmats = fbmats,
            absorbingstates = absorbingstates,
            # Phase-type infrastructure (nothing if not using)
            phasetype_surrogate = phasetype_surrogate,
            tpm_book_ph = tpm_book_ph,
            hazmat_book_ph = hazmat_book_ph,
            fbmats_ph = fbmats_ph,
            emat_ph = emat_ph,
            # Expanded data infrastructure (nothing if not using/not needed)
            expanded_ph_data = expanded_ph_data,
            expanded_ph_subjectindices = ph_subjectindices,
            expanded_ph_tpm_map = expanded_ph_tpm_map,
            ph_original_row_map = ph_original_row_map)
    end # end-while

    if !is_converged
        @warn "MCEM did not converge."
    end

    # rectify spline coefs
    if any(isa.(model.hazards, _SplineHazard))
        rectify_coefs!(params_cur, model)
    end

    # hessian
    if !isnothing(constraints) || any(map(x -> (isa(x, _SplineHazard) && x.monotone != 0), model.hazards))

        # no hessian when there are constraints
        if compute_vcov == true
            @warn "No covariance matrix is returned when constraints are provided or when using monotone splines."
        end
        vcov = nothing
    
    elseif is_converged && compute_vcov
        if verbose
            println("Computing variance-covariance matrix at final estimates.")
        end

        # set up containers for path and sampling weight
        path = Array{SamplePath}(undef, 1)
        samplingweight = Vector{Float64}(undef, 1)
        
        # Get hazard parameter blocks from parameters.natural
        # Each hazard has a vector of parameters; compute block indices
        natural_pars = model.parameters.natural
        nhaz = length(natural_pars)
        block_sizes = [length(natural_pars[k]) for k in 1:nhaz]
        
        # Build index ranges for each hazard block
        elem_ptr = cumsum([1; block_sizes])
        blocks = [elem_ptr[k]:(elem_ptr[k+1]-1) for k in 1:nhaz]
        
        # initialize Fisher information matrix containers
        nparams = length(params_cur)
        fishinf = zeros(Float64, nparams, nparams)
        
        # The Hessian of the log-likelihood is BLOCK-DIAGONAL with blocks corresponding to 
        # each hazard's parameters. This is because:
        #   log L = Σᵢ [-Λ₁(t) - Λ₂(t) - ... + log hₖ(t)]  (for path with transition via hazard k)
        # Each hazard's parameters only appear in its own cumulative hazard Λₖ and log-hazard,
        # so ∂²L/∂θₐ∂θᵦ = 0 when θₐ and θᵦ belong to different hazards.
        #
        # Note: This block-diagonal structure holds because we skip vcov computation when 
        # there are constraints (constraints could couple parameters across hazards).
        # For constrained problems, one would need to compute the bordered Hessian or use
        # SparseDiffTools.jl with automatic sparsity detection.
        #
        # We exploit this structure by computing each block's Hessian separately, which is
        # O(Σᵢ bᵢ²) instead of O(n²) for a dense Hessian (where bᵢ = size of block i, n = total params).
        
        # Decide whether to use block-diagonal or dense Hessian computation
        # Due to function call overhead, we only use block-diagonal when theoretical speedup
        # exceeds the threshold (default 2.0× speedup required)
        sum_block_sq = sum(bs^2 for bs in block_sizes)
        theoretical_speedup = nparams^2 / sum_block_sq
        use_block_diagonal = theoretical_speedup > block_hessian_speedup && nhaz > 1
        
        # Full likelihood function
        ll_full = pars -> (loglik_AD(pars, ExactDataAD(path, samplingweight, model.hazards, model); neg=false))

        if use_block_diagonal
            # Block-diagonal Hessian computation (faster for models with many hazards)
            
            # Pre-allocate containers for each block
            diffres_blocks = [DiffResults.HessianResult(zeros(bs)) for bs in block_sizes]
            fish_i1_blocks = [zeros(Float64, bs, bs) for bs in block_sizes]
            
            # Create block likelihood functions (one per hazard)
            # These compute likelihood as a function of only that block's parameters
            params_base = copy(params_cur)
            ll_blocks = Vector{Function}(undef, nhaz)
            for k in 1:nhaz
                block = blocks[k]
                ll_blocks[k] = function(θ_block)
                    T = eltype(θ_block)
                    θ_full = T.(params_base)
                    θ_full[block] = θ_block
                    return ll_full(θ_full)
                end
            end
            
            # accumulate Fisher information
            for i in 1:nsubj

                # set importance weight
                samplingweight[1] = model.SubjectWeights[i]

                # number of paths
                npaths = length(samplepaths[i])

                # for accumulating gradients (full parameter vector)
                grads = zeros(Float64, nparams, npaths)
                
                # Process each block independently (exploiting block-diagonal Hessian structure)
                for (k, block) in enumerate(blocks)
                    bs = block_sizes[k]
                    fill!(fish_i1_blocks[k], 0.0)
                    grads_block = zeros(Float64, bs, npaths)
                    
                    # calculate gradient and hessian for paths (block k only)
                    for j in 1:npaths
                        path[1] = samplepaths[i][j]
                        diffres_blocks[k] = ForwardDiff.hessian!(diffres_blocks[k], ll_blocks[k], params_cur[block])

                        # grab block hessian and gradient
                        g_block = DiffResults.gradient(diffres_blocks[k])
                        H_block = DiffResults.hessian(diffres_blocks[k])
                        
                        grads_block[:, j] = g_block
                        grads[block, j] = g_block

                        # just to be safe wrt nans or infs
                        if !all(isfinite, H_block)
                            fill!(H_block, 0.0)
                        end

                        if !all(isfinite, g_block)
                            fill!(g_block, 0.0)
                            grads_block[:, j] .= 0.0
                            grads[block, j] .= 0.0
                        end

                        fish_i1_blocks[k] .+= ImportanceWeights[i][j] * (-H_block - g_block * transpose(g_block))
                    end
                    
                    # Block contribution to fish_i2: (Σⱼ wⱼgⱼ)(Σⱼ wⱼgⱼ)ᵀ for this block
                    g_weighted_block = grads_block * ImportanceWeights[i]
                    fish_i2_block = g_weighted_block * transpose(g_weighted_block)
                    
                    fishinf[block, block] .+= fish_i1_blocks[k] .+ fish_i2_block
                end
            end
        else
            # Dense Hessian computation (faster for models with few hazards/parameters)
            fish_i1 = zeros(Float64, nparams, nparams)
            diffres = DiffResults.HessianResult(params_cur)
            
            # accumulate Fisher information
            for i in 1:nsubj

                # set importance weight
                samplingweight[1] = model.SubjectWeights[i]

                # number of paths
                npaths = length(samplepaths[i])

                # for accumulating gradients and hessians
                grads = Array{Float64}(undef, nparams, npaths)

                # reset matrices for accumulating Fisher info contributions
                fill!(fish_i1, 0.0)

                # calculate gradient and hessian for paths
                for j in 1:npaths
                    path[1] = samplepaths[i][j]
                    diffres = ForwardDiff.hessian!(diffres, ll_full, params_cur)

                    # grab hessian and gradient
                    grads[:,j] = DiffResults.gradient(diffres)

                    # just to be safe wrt nans or infs
                    if !all(isfinite, DiffResults.hessian(diffres))
                        fill!(DiffResults.hessian(diffres), 0.0)
                    end

                    if !all(isfinite, DiffResults.gradient(diffres))
                        fill!(DiffResults.gradient(diffres), 0.0)
                    end

                    fish_i1 .+= ImportanceWeights[i][j] * (-DiffResults.hessian(diffres) - DiffResults.gradient(diffres) * transpose(DiffResults.gradient(diffres)))
                end

                # Optimized: Σⱼ Σₖ wⱼwₖ gⱼgₖᵀ = (Σⱼ wⱼgⱼ)(Σₖ wₖgₖ)ᵀ = g_weighted * g_weighted'
                g_weighted = grads * ImportanceWeights[i]
                fish_i2 = g_weighted * transpose(g_weighted)

                fishinf .+= fish_i1 .+ fish_i2
            end
        end

        # get the variance-covariance matrix
        vcov = pinv(Symmetric(fishinf), atol = vcov_threshold ? (log(nsubj) * length(params_cur))^-2 : sqrt(eps(real(float(oneunit(eltype(fishinf)))))))
        vcov[findall(isapprox.(vcov, 0.0; atol = sqrt(eps(Float64))))] .= 0.0
        vcov = Symmetric(vcov)
    else
        vcov = nothing
    end

    # subject marginal likelihood
    logliks = compute_loglik(model, loglik_surrog, loglik_target_cur, NormConstantProposal)

    # compute robust variance estimates if requested
    ij_variance = nothing
    jk_variance = nothing
    subject_grads = nothing
    
    if (compute_ij_vcov || compute_jk_vcov) && !isnothing(vcov) && isnothing(constraints) && is_converged
        if verbose
            println("Computing robust variance estimates...")
        end
        robust_result = compute_robust_vcov(params_cur, model, samplepaths, ImportanceWeights;
                                           compute_ij = compute_ij_vcov,
                                           compute_jk = compute_jk_vcov,
                                           loo_method = loo_method,
                                           vcov_threshold = vcov_threshold)
        ij_variance = robust_result.ij_vcov
        jk_variance = robust_result.jk_vcov
        subject_grads = robust_result.subject_gradients
    end

    # return convergence records
    ConvergenceRecords = return_convergence_records ? (mll_trace=mll_trace, ess_trace=ess_trace, path_count_trace=path_count_trace, parameters_trace=parameters_trace, psis_pareto_k = psis_pareto_k) : nothing

    # return sampled paths and importance weights
    ProposedPaths = return_proposed_paths ? (paths=samplepaths, weights=ImportanceWeights) : nothing

    # Build ParameterHandling structure for fitted parameters
    # params_cur contains log-scale parameters; we need to split into per-hazard vectors
    # Compute block sizes from model.parameters.natural structure
    natural_vals = values(model.parameters.natural)
    block_sizes = [length(v) for v in natural_vals]
    
    # Split params_cur into per-hazard log-scale vectors
    log_scale_params = Vector{Vector{Float64}}(undef, length(block_sizes))
    offset = 0
    for i in eachindex(block_sizes)
        log_scale_params[i] = params_cur[(offset+1):(offset+block_sizes[i])]
        offset += block_sizes[i]
    end
    
    # Rebuild parameters with proper constraints using model's hazard info
    parameters_fitted = rebuild_parameters(log_scale_params, model)

    # wrap results
    model_fitted = MultistateModelFitted(
        data_original,
        parameters_fitted,
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
        surrogate,
        ConvergenceRecords,
        ProposedPaths,
        model.modelcall)

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
