"""
    fit(model::MultistateModel; constraints = nothing, verbose = true, compute_vcov = true, kwargs...)

Fit a multistate model to continuously observed (exact) data.

Uses L-BFGS optimization for unconstrained problems (5-6× faster than interior-point methods)
and Ipopt for constrained problems by default.

# Arguments
- `model::MultistateModel`: multistate model object with exact observation times
- `constraints`: parameter constraints (see Constraints documentation)
- `verbose::Bool=true`: print optimization messages
- `solver`: optimization solver to use. Default is `nothing`, which uses:
  - `Optim.LBFGS()` for unconstrained problems (fast, gradient-based)
  - `Ipopt.Optimizer()` for constrained problems (interior-point method)
  Users can specify any solver compatible with Optimization.jl, e.g.:
  - `Optim.BFGS()`, `Optim.NelderMead()` for unconstrained
  - `Ipopt.Optimizer()` for constrained
- `compute_vcov::Bool=true`: compute model-based variance-covariance matrix (H⁻¹).
  Useful for diagnostics by comparing to robust variance.
- `vcov_threshold::Bool=true`: if true, uses adaptive threshold `1/√(log(n)·p)` for pseudo-inverse 
  of Fisher information; otherwise uses `√eps()`. Helps with near-singular Hessians.
- `compute_ij_vcov::Bool=true`: compute infinitesimal jackknife (sandwich/robust) variance H⁻¹KH⁻¹.
  **This is the recommended variance for inference** as it remains valid under model misspecification.
- `compute_jk_vcov::Bool=false`: compute jackknife variance ((n-1)/n)·Σᵢ ΔᵢΔᵢᵀ
- `loo_method::Symbol=:direct`: method for leave-one-out perturbations:
  - `:direct`: compute H⁻¹ once, multiply by each gᵢ (faster for n >> p)
  - `:cholesky`: use Cholesky rank-k downdates (more stable, slower)

# Returns
- `MultistateModelFitted`: fitted model object with estimates, standard errors, and diagnostics

# Variance Estimation (Default Behavior)

By default, both model-based and IJ (sandwich) variance are computed:
- **IJ/Sandwich variance** (`compute_ij_vcov=true`): **Primary for inference**. Robust variance
  H⁻¹KH⁻¹ that remains valid even under model misspecification.
- **Model-based variance** (`compute_vcov=true`): **For diagnostics**. Standard MLE variance H⁻¹,
  valid only under correct model specification. Compare to IJ variance to assess model adequacy.

Use `compare_variance_estimates(fitted)` to diagnose potential model misspecification.

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

# Use a custom solver (e.g., BFGS instead of L-BFGS)
using Optim
fitted = fit(model; solver=Optim.BFGS())
```

See also: [`get_vcov`](@ref), [`get_ij_vcov`](@ref), [`compare_variance_estimates`](@ref)
"""
function fit(model::MultistateModel; constraints = nothing, verbose = true, solver = nothing, compute_vcov = true, vcov_threshold = true, compute_ij_vcov = true, compute_jk_vcov = false, loo_method = :direct, kwargs...)

    # initialize array of sample paths
    samplepaths = extract_paths(model)

    # extract and initialize model parameters
    # Phase 3: Use ParameterHandling.jl flat parameters (log scale)
    parameters = get_parameters_flat(model)

    # parse constraints, or not, and solve
    if isnothing(constraints) 
        # get estimates - use L-BFGS for unconstrained (5-6x faster than Ipopt)
        optf = OptimizationFunction(loglik_exact, Optimization.AutoForwardDiff())
        prob = OptimizationProblem(optf, parameters, ExactData(model, samplepaths))

        # solve with user-specified solver or default L-BFGS
        _solver = isnothing(solver) ? Optim.LBFGS() : solver
        sol  = solve(prob, _solver)

        # rectify spline coefs
        if any(isa.(model.hazards, _SplineHazard))
            rectify_coefs!(sol.u, model)
        end
        
        # get vcov
        if compute_vcov && (sol.retcode == ReturnCode.Success) && !any(map(x -> (isa(x, _SplineHazard) && x.monotone != 0), model.hazards))
            # preallocate the hessian matrix
            diffres = DiffResults.HessianResult(sol.u)

            # single argument function for log-likelihood            
            ll = pars -> loglik_exact(pars, ExactData(model, samplepaths); neg=false)

            # compute gradient and hessian
            diffres = ForwardDiff.hessian!(diffres, ll, sol.u)

            # grab results
            gradient = DiffResults.gradient(diffres)
            fishinf = -DiffResults.hessian(diffres)
            vcov = pinv(Symmetric(fishinf), atol = vcov_threshold ? (log(length(samplepaths)) * length(sol.u))^-2 : sqrt(eps(real(float(oneunit(eltype(fishinf)))))))
            vcov[isapprox.(vcov, 0.0; atol = sqrt(eps(Float64)), rtol = sqrt(eps(Float64)))] .= 0.0
            vcov = Symmetric(vcov)
        else
            vcov = nothing
        end
    else
        # create constraint function and check that constraints are satisfied at the initial values
        _constraints = deepcopy(constraints)
        consfun_multistate = parse_constraints(_constraints.cons, model.hazards; consfun_name = :consfun_multistate)

        initcons = consfun_multistate(zeros(length(constraints.cons)), parameters, nothing)
        badcons = findall(initcons .< constraints.lcons .|| initcons .> constraints.ucons)
        if length(badcons) > 0
            @error "Constraints $badcons are violated at the initial parameter values."
        end

        optf = OptimizationFunction(loglik, Optimization.AutoForwardDiff(), cons = consfun_multistate)
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
                                           vcov_threshold = vcov_threshold)
        ij_variance = robust_result.ij_vcov
        jk_variance = robust_result.jk_vcov
        subject_grads = robust_result.subject_gradients
    end

    # create parameters VectorOfVectors from solution
    parameters_fitted = VectorOfVectors(sol.u, model.parameters.elem_ptr)
    
    # build ParameterHandling structure for fitted parameters
    # Use the unflatten function from the model to convert flat params back to transformed
    params_transformed = model.parameters_ph.unflatten(sol.u)
    params_natural = ParameterHandling.value(params_transformed)
    parameters_ph_fitted = (
        flat = sol.u,
        transformed = params_transformed,
        natural = params_natural,
        unflatten = model.parameters_ph.unflatten
    )

    # wrap results
    model_fitted = MultistateModelFitted(
        model.data,
        parameters_fitted,
        parameters_ph_fitted,
        (loglik = -sol.minimum, subj_lml = ll_subj),
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

    # remake splines and calculate risk periods
    for i in eachindex(model_fitted.hazards)
        if isa(model_fitted.hazards[i], _SplineHazard)
            remake_splines!(model_fitted.hazards[i], model_fitted.parameters[i])
            set_riskperiod!(model_fitted.hazards[i])
        end
    end

    # return fitted object
    return model_fitted;
end


"""
    fit(model::MultistateMarkovModel; constraints = nothing, verbose = true, compute_vcov = true, kwargs...)

Fit a Markov multistate model to interval-censored or panel data.

Uses L-BFGS optimization for unconstrained problems (5-6× faster than interior-point methods)
and Ipopt for constrained problems by default.

# Arguments
- `model::Union{MultistateMarkovModel, MultistateMarkovModelCensored}`: Markov model with panel observations
- `constraints`: parameter constraints (see Constraints documentation)
- `verbose::Bool=true`: print optimization messages
- `solver`: optimization solver to use. Default is `nothing`, which uses:
  - `Optim.LBFGS()` for unconstrained problems
  - `Ipopt.Optimizer()` for constrained problems
  Users can specify any solver compatible with Optimization.jl.
- `compute_vcov::Bool=true`: compute model-based variance-covariance matrix (for diagnostics)
- `vcov_threshold::Bool=true`: use adaptive threshold for pseudo-inverse of Fisher information
- `compute_ij_vcov::Bool=true`: compute infinitesimal jackknife (sandwich/robust) variance.
  **This is the recommended variance for inference.**
- `compute_jk_vcov::Bool=false`: compute jackknife variance
- `loo_method::Symbol=:direct`: method for LOO perturbations (`:direct` or `:cholesky`)

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
```

See also: [`fit(::MultistateModel)`](@ref), [`compare_variance_estimates`](@ref)
"""
function fit(model::Union{MultistateMarkovModel,MultistateMarkovModelCensored}; constraints = nothing, verbose = true, solver = nothing, compute_vcov = true, vcov_threshold = true, compute_ij_vcov = true, compute_jk_vcov = false, loo_method = :direct, kwargs...)

    # containers for bookkeeping TPMs
    books = build_tpm_mapping(model.data)

    # extract and initialize model parameters
    # Phase 3: Use ParameterHandling.jl flat parameters (log scale)
    parameters = get_parameters_flat(model)

    # number of subjects
    nsubj = length(model.subjectindices)

    # parse constraints, or not, and solve
    if isnothing(constraints)
        # get estimates - use L-BFGS for unconstrained (5-6x faster than Ipopt)
        optf = OptimizationFunction(loglik_markov, Optimization.AutoForwardDiff())
        prob = OptimizationProblem(optf, parameters, MPanelData(model, books))
        
        # solve with user-specified solver or default L-BFGS
        _solver = isnothing(solver) ? Optim.LBFGS() : solver
        sol  = solve(prob, _solver)

        # get vcov
        if compute_vcov && (sol.retcode == ReturnCode.Success)
            # preallocate the hessian matrix
            diffres = DiffResults.HessianResult(sol.u)

            # single argument function for log-likelihood            
            ll = pars -> loglik_markov(pars, MPanelData(model, books); neg=false)

            # compute gradient and hessian
            diffres = ForwardDiff.hessian!(diffres, ll, sol.u)

            # grab results
            gradient = DiffResults.gradient(diffres)
            fishinf = Symmetric(.-DiffResults.hessian(diffres))
            vcov = pinv(Symmetric(fishinf), atol = vcov_threshold ? (log(nsubj) * length(sol.u))^-2 : sqrt(eps(real(float(oneunit(eltype(fishinf)))))))
            vcov[isapprox.(vcov, 0.0; atol = eps(Float64))] .= 0.0
            vcov = Symmetric(vcov)
        else
            vcov = nothing
        end
    else
        # create constraint function and check that constraints are satisfied at the initial values
        _constraints = deepcopy(constraints)
        consfun_markov = parse_constraints(_constraints.cons, model.hazards; consfun_name = :consfun_markov)

        initcons = consfun_markov(zeros(length(constraints.cons)), parameters, nothing)
        badcons = findall(initcons .< constraints.lcons .|| initcons .> constraints.ucons)
        if length(badcons) > 0
            @error "Constraints $badcons are violated at the initial parameter values."
        end

        optf = OptimizationFunction(loglik_markov, Optimization.AutoForwardDiff(), cons = consfun_markov)
        prob = OptimizationProblem(optf, parameters, MPanelData(model, books), lcons = constraints.lcons, ucons = constraints.ucons)
        
        # solve with user-specified solver or default Ipopt
        _solver = isnothing(solver) ? Ipopt.Optimizer() : solver
        sol  = solve(prob, _solver; print_level = 0)

        # no hessian when there are constraints
        if compute_vcov == true
            @warn "No covariance matrix is returned when constraints are provided."
        end
        vcov = nothing
    end

    # compute loglikelihood at the estimate
    logliks = (loglik = -sol.minimum, subj_lml = loglik_markov(sol.u, MPanelData(model, books); return_ll_subj = true))

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
                                           vcov_threshold = vcov_threshold)
        ij_variance = robust_result.ij_vcov
        jk_variance = robust_result.jk_vcov
        subject_grads = robust_result.subject_gradients
    end

    # create parameters VectorOfVectors from solution
    parameters_fitted = VectorOfVectors(sol.u, model.parameters.elem_ptr)
    
    # build ParameterHandling structure for fitted parameters
    params_transformed_pairs = [
        hazname => ParameterHandling.positive(Vector{Float64}(parameters_fitted[idx]))
        for (hazname, idx) in sort(collect(model.hazkeys), by = x -> x[2])
    ]
    params_transformed = NamedTuple(params_transformed_pairs)
    params_flat, unflatten_fn = ParameterHandling.flatten(params_transformed)
    params_natural = ParameterHandling.value(params_transformed)
    parameters_ph_fitted = (
        flat = params_flat,
        transformed = params_transformed,
        natural = params_natural,
        unflatten = unflatten_fn
    )

    # wrap results
    return MultistateModelFitted(
        model.data,
        parameters_fitted,
        parameters_ph_fitted,
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

Uses L-BFGS optimization for unconstrained M-steps (5-6× faster than interior-point methods)
and Ipopt for constrained M-steps by default.

# Arguments

**Model and constraints:**
- `model::Union{MultistateSemiMarkovModel, MultistateSemiMarkovModelCensored}`: semi-Markov model
- `optimize_surrogate::Bool=true`: optimize Markov surrogate parameters for path proposal
- `constraints`: parameter constraints tuple
- `surrogate_constraints`: constraints for Markov surrogate optimization
- `surrogate_parameters`: initial surrogate parameters (if not optimizing)

**Optimization:**
- `solver`: optimization solver for M-step. Default is `nothing`, which uses:
  - `Optim.LBFGS()` for unconstrained problems
  - `Ipopt.Optimizer()` for constrained problems
  Users can specify any solver compatible with Optimization.jl.

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

**Output control:**
- `verbose::Bool=true`: print progress messages
- `return_convergence_records::Bool=true`: save iteration history
- `return_proposed_paths::Bool=false`: save latent paths and importance weights

**Variance estimation:**
- `compute_vcov::Bool=true`: compute model-based variance via Louis's identity (for diagnostics)
- `vcov_threshold::Bool=true`: use adaptive threshold for pseudo-inverse
- `compute_ij_vcov::Bool=true`: compute IJ/sandwich variance (robust, **recommended for inference**)
- `compute_jk_vcov::Bool=false`: compute jackknife variance
- `loo_method::Symbol=:direct`: method for LOO perturbations

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
```

See also: [`fit(::MultistateModel)`](@ref), [`compare_variance_estimates`](@ref)
"""
function fit(model::Union{MultistateSemiMarkovModel, MultistateSemiMarkovModelCensored}; optimize_surrogate = true, constraints = nothing, surrogate_constraints = nothing, surrogate_parameters = nothing, solver = nothing, maxiter = 100, tol = 1e-2, ascent_threshold = 0.1, stopping_threshold = 0.1, ess_increase = 2.0, ess_target_initial = 50, max_ess = 10000, max_sampling_effort = 20, npaths_additional = 10, verbose = true, return_convergence_records = true, return_proposed_paths = false, compute_vcov = true, vcov_threshold = true, compute_ij_vcov = true, compute_jk_vcov = false, loo_method = :direct, kwargs...)

    # copy of data
    data_original = deepcopy(model.data)

    # check that constraints for the initial values are satisfied
    if !isnothing(constraints)
        # create constraint function and check that constraints are satisfied at the initial values
        _constraints = deepcopy(constraints)
        consfun_semimarkov = parse_constraints(_constraints.cons, model.hazards; consfun_name = :consfun_semimarkov)

        # Phase 3: Use ParameterHandling.jl flat parameters for constraint check
        initcons = consfun_semimarkov(zeros(length(constraints.cons)), get_parameters_flat(model), nothing)
        badcons = findall(initcons .< constraints.lcons .|| initcons .> constraints.ucons)
        if length(badcons) > 0
            @error "Constraints $badcons are violated at the initial parameter values."
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
        @error "Attempting to fit a Markov model via MCEM. Recode degree 0 splines as exponential hazards and refit."
    end

    # MCEM initialization
    keep_going = true
    iter = 0
    convergence = false

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
    # Phase 3: Use ParameterHandling.jl flat parameter length
    parameters_trace = ElasticArray{Float64, 2}(undef, length(get_parameters_flat(model)), 0) # parameter estimates

    # build surrogate
    if optimize_surrogate

        surrogate_fitted = fit_surrogate(model; surrogate_parameters=surrogate_parameters, surrogate_constraints=surrogate_constraints, verbose=verbose)

        # create the surrogate object
        surrogate = MarkovSurrogate(surrogate_fitted.hazards, surrogate_fitted.parameters)
        
    else
        # set to supplied initial values
        if isnothing(surrogate_parameters)
            @error "Parameters for the Markov surrogate must be supplied if optimize_surrogate=false."
        end 

        # if parameters for the surrogate are provided, simply use these values
        surrogate = MarkovSurrogate(model.markovsurrogate.hazards, surrogate_parameters)
    end

     # containers for bookkeeping TPMs
     books = build_tpm_mapping(model.data)    

     # transition probability objects for Markov surrogate
     hazmat_book_surrogate = build_hazmat_book(Float64, model.tmat, books[1])
     tpm_book_surrogate = build_tpm_book(Float64, model.tmat, books[1])
 
     # allocate memory for matrix exponential
     cache = ExponentialUtilities.alloc_mem(similar(hazmat_book_surrogate[1]), ExpMethodGeneric())

    # Solve Kolmogorov equations for TPMs
    for t in eachindex(books[1])
        # compute the transition intensity matrix
        compute_hazmat!(hazmat_book_surrogate[t], surrogate.parameters, surrogate.hazards, books[1][t], model.data)
        # compute transition probability matrices
        compute_tmat!(tpm_book_surrogate[t], hazmat_book_surrogate[t], books[1][t], cache)
    end

    # compute normalizing constant of Markov proposal
    NormConstantProposal = surrogate_fitted.loglik.loglik

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
        absorbingstates = absorbingstates)
    
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

    # start algorithm
    while keep_going

        # increment the iteration
        iter += 1
        
        # solve M-step: use user-specified solver or default (L-BFGS for unconstrained, Ipopt for constrained)
        if isnothing(constraints)
            _solver = isnothing(solver) ? Optim.LBFGS() : solver
            params_prop_optim = solve(remake(prob, u0 = Vector(params_cur), p = SMPanelData(model, samplepaths, ImportanceWeights)), _solver)
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
            convergence = true
        end

        # increment the current values of the parameter, marginal log likelihood, target loglik
        params_cur        = deepcopy(params_prop)
        mll_cur           = deepcopy(mll_prop)
        loglik_target_cur = deepcopy(loglik_target_prop)

        # increment log importance weights, importance weights, effective sample size and pareto shape parameter
        ComputeImportanceWeightsESS!(loglik_target_cur, loglik_surrog, _logImportanceWeights, ImportanceWeights, ess_cur, ess_target, psis_pareto_k)

        # print update
        if verbose
            println("Iteration: $iter")
            println("Target ESS: $(round(ess_target;digits=2)) per-subject")
            println("Range of the number of sampled paths per-subject: [$(ceil(ess_target)), $(max(length.(samplepaths)...))]")
            println("Estimate of the marginal log-likelihood, Q: $(round(mll_cur;digits=3))")
            println("Gain in marginal log-likelihood, ΔQ: $(round(mll_change;sigdigits=3))")
            println("MCEM asymptotic standard error: $(round(ase;sigdigits=3))")
            println("Ascent lower and upper bound: [$(round(ascent_lb; sigdigits=3)), $(round(ascent_ub; sigdigits=3))]")
            println("Estimate of the log marginal likelihood, l(θ): $(round(compute_loglik(model, loglik_surrog, loglik_target_cur, NormConstantProposal).loglik;digits=3))\n")
        end

        # save marginal log likelihood, parameters and effective sample size
        append!(parameters_trace, params_cur)
        push!(mll_trace, mll_cur)
        append!(ess_trace, ess_cur)

        # check for convergence
        if ascent_ub < tol
            convergence = true
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
            ess_target = ceil(ess_increase*ess_target) # TODO: alternatively, use Caffo's power calculation to determine the new ESS target
            if verbose  println("Target ESS is increased to $ess_target, because ascent lower bound < 0.\n") end

            # no need to sample paths for subjects with a single possible path
            ess_cur[findall(length.(ImportanceWeights) .== 1)] .= ess_target
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
            absorbingstates = absorbingstates)
    end # end-while

    if !convergence
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
    
    elseif convergence && compute_vcov
        if verbose
            println("Computing variance-covariance matrix at final estimates.")
        end

        # set up containers for path and sampling weight
        path = Array{SamplePath}(undef, 1)
        samplingweight = Vector{Float64}(undef, 1)
        
        # Get hazard parameter blocks from elem_ptr
        # elem_ptr[i] gives the start index for hazard i (1-indexed into flat parameter vector)
        elem_ptr = model.parameters.elem_ptr
        nhaz = length(elem_ptr) - 1
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
        # Due to function call overhead, we only use block-diagonal when theoretical speedup > 2.5
        block_sizes = [length(b) for b in blocks]
        sum_block_sq = sum(bs^2 for bs in block_sizes)
        theoretical_speedup = nparams^2 / sum_block_sq
        use_block_diagonal = theoretical_speedup > 2.5 && nhaz > 1
        
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
    
    if (compute_ij_vcov || compute_jk_vcov) && !isnothing(vcov) && isnothing(constraints) && convergence
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
    ConvergenceRecords = return_convergence_records ? (mll_trace=mll_trace, ess_trace=ess_trace, parameters_trace=parameters_trace, psis_pareto_k = psis_pareto_k) : nothing

    # return sampled paths and importance weights
    ProposedPaths = return_proposed_paths ? (paths=samplepaths, weights=ImportanceWeights) : nothing

    # create parameters VectorOfVectors from current parameters
    parameters_fitted = VectorOfVectors(params_cur, model.parameters.elem_ptr)
    
    # build ParameterHandling structure for fitted parameters
    params_transformed_pairs = [
        hazname => ParameterHandling.positive(Vector{Float64}(parameters_fitted[idx]))
        for (hazname, idx) in sort(collect(model.hazkeys), by = x -> x[2])
    ]
    params_transformed = NamedTuple(params_transformed_pairs)
    params_flat, unflatten_fn = ParameterHandling.flatten(params_transformed)
    params_natural = ParameterHandling.value(params_transformed)
    parameters_ph_fitted = (
        flat = params_flat,
        transformed = params_transformed,
        natural = params_natural,
        unflatten = unflatten_fn
    )

    # wrap results
    model_fitted = MultistateModelFitted(
        data_original,
        parameters_fitted,
        parameters_ph_fitted,
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

    # remake splines and calculate risk periods
    for i in eachindex(model_fitted.hazards)
        if isa(model_fitted.hazards[i], _SplineHazard)
            remake_splines!(model_fitted.hazards[i], model_fitted.parameters[i])
            set_riskperiod!(model_fitted.hazards[i])
        end
    end

    # return fitted object
    return model_fitted;
end
