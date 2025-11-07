"""
    fit(model::MultistateModel; constraints = nothing, verbose = true, compute_vcov = true, kwargs...)

Fit a multistate model to continuously observed data.

# Arguments
- model: multistate model object
- constraints: constraints on model parameters
- verbose: print messages, defaults to true
- compute_vcov: defaults to true
- vcov_threshold: if true, the variance covariance matrix calculation only inverts singular values of the fisher information matrix that are greater than 1 / sqrt(log(n) * k) where k is the number of parameters and n is the number of subjects in the dataset. otherwise, the absolute tolerance is set to the square root of eps(). 
"""
function fit(model::MultistateModel; constraints = nothing, verbose = true, compute_vcov = true, vcov_threshold = true, kwargs...)

    # initialize array of sample paths
    samplepaths = extract_paths(model)

    # extract and initialize model parameters
    # Phase 3: Use ParameterHandling.jl flat parameters (log scale)
    parameters = get_parameters_flat(model)

    # parse constraints, or not, and solve
    if isnothing(constraints) 
        # get estimates
        optf = OptimizationFunction(loglik_exact, Optimization.AutoForwardDiff())
        prob = OptimizationProblem(optf, parameters, ExactData(model, samplepaths))

        # solve
        sol  = solve(prob, Ipopt.Optimizer(); print_level = 0)

        # rectify spline coefs
        if any(isa.(model.hazards, _SplineHazard))
            rectify_coefs!(sol.u, model)
        end
        
        # get vcov
        if compute_vcov && (sol.retcode == ReturnCode.Success) && !any(map(x -> (isa(x, _SplineHazard) && x.monotone != 0), model.hazards))
            # preallocate the hessian matrix
            diffres = DiffResults.HessianResult(sol.u)

            # single argument function for log-likelihood            
            ll = pars -> loglik(pars, ExactData(model, samplepaths); neg=false)

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
        sol  = solve(prob, Ipopt.Optimizer(); print_level = 0)

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
    model_fitted = MultistateModelFitted(
        model.data,
        parameters_fitted,
        parameters_ph_fitted,
        (loglik = -sol.minimum, subj_lml = ll_subj),
        vcov,
        model.hazards,
        model.totalhazards,
        model.tmat,
        model.emat,
        model.hazkeys,
        model.subjectindices,
        model.SamplingWeights,
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

Fit a multistate markov model to interval censored data or a mix of panel data and exact jump times.

- model: multistate Markov model, possibly with censoring.
- constraints: constraints on model parameters.
- verbose: print messages, defaults to true.
- compute_vcov: compute variance-covariance matrix, defaults to true if no constraints or false otherwise.
- vcov_threshold: if true, the variance covariance matrix calculation only inverts singular values of the fisher information matrix that are greater than 1 / sqrt(log(n) * k) where k is the number of parameters and n is the number of subjects in the dataset. otherwise, the absolute tolerance is set to the square root of eps(). 
"""
function fit(model::Union{MultistateMarkovModel,MultistateMarkovModelCensored}; constraints = nothing, verbose = true, compute_vcov = true, vcov_threshold = true, kwargs...)

    # containers for bookkeeping TPMs
    books = build_tpm_mapping(model.data)

    # extract and initialize model parameters
    # Phase 3: Use ParameterHandling.jl flat parameters (log scale)
    parameters = get_parameters_flat(model)

    # number of subjects
    nsubj = length(model.subjectindices)

    # parse constraints, or not, and solve
    if isnothing(constraints)
        # get estimates
        optf = OptimizationFunction(loglik_markov, Optimization.AutoForwardDiff())
        prob = OptimizationProblem(optf, parameters, MPanelData(model, books))
        sol  = solve(prob, Ipopt.Optimizer(); print_level = 0)

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
        sol  = solve(prob, Ipopt.Optimizer(); print_level = 0)

        # no hessian when there are constraints
        if compute_vcov == true
            @warn "No covariance matrix is returned when constraints are provided."
        end
        vcov = nothing
    end

    # compute loglikelihood at the estimate
    logliks = (loglik = -sol.minimum, subj_lml = loglik_markov(sol.u, MPanelData(model, books); return_ll_subj = true))

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
        model.hazards,
        model.totalhazards,
        model.tmat,
        model.emat,
        model.hazkeys,
        model.subjectindices,
        model.SamplingWeights,
        model.CensoringPatterns,
        model.markovsurrogate,
        (solution = sol,), # ConvergenceRecords::Union{Nothing, NamedTuple}
        nothing, # ProposedPaths::Union{Nothing, NamedTuple}
        model.modelcall);
end


"""
fit(model::Union{MultistateSemiMarkovModel, MultistateSemiMarkovModelCensored}; optimize_surrogate = true, constraints = nothing, surrogate_constraints = nothing, surrogate_parameters = nothing,  maxiter = 100, tol = 1e-2, ascent_threshold = 0.1, stopping_threshold = 0.1, ess_increase = 1.5, ess_target_initial = 100, max_sampling_effort = 20, npaths_additional = 10, verbose = true, return_convergence_records = true, return_proposed_paths = false, compute_vcov = true, kwargs...)

Fit a semi-Markov model to panel data via Monte Carlo EM.

# Arguments

- model: multistate model object
- optimize_surrogate: should the parameters Markov surrogate for proposing paths be set to the MLE? defaults to true
- constraints: tuple for specifying parameter constraints
- surrogate_constraints: tuple for specifying parameter constraints for the Markov surrogate
- maxiter: maximum number of MCEM iterations
- tol: tolerance for the change in the MLL, i.e., upper bound of the stopping rule to be ruled out. Defaults to 0.01.
- ascent_threshold: standard normal quantile for asymptotic lower bound for ascent
- stopping_threshold: standard normal quantile for stopping the MCEM algorithm
- ess_increase: Inflation factor for target ESS per person, ESS_new = ESS_cur * ess_increase
- ess_target_initial: initial number of particles per participant for MCEM
- max_ess: maximum ess after which the mcem is stopped for nonconvergence
- max_sampling_effort: factor of the ESS at which to break the loop for sampling additional paths
- npaths_additional: increment for number of additional paths when augmenting the pool of paths
- verbose: print status
- return_convergence_records: save history throughout the run
- return_proposed_paths save latent paths and importance weights
- compute_vcov: should the variance-covariance matrix be computed at the final estimates? defaults to true.
- vcov_threshold: if true, the variance covariance matrix calculation only inverts singular values of the fisher information matrix that are greater than 1 / sqrt(log(n) * k) where k is the number of parameters and n is the number of subjects in the dataset. otherwise, the absolute tolerance is set to the square root of eps(). 
"""
function fit(model::Union{MultistateSemiMarkovModel, MultistateSemiMarkovModelCensored}; optimize_surrogate = true, constraints = nothing, surrogate_constraints = nothing, surrogate_parameters = nothing, maxiter = 100, tol = 1e-2, ascent_threshold = 0.1, stopping_threshold = 0.1, ess_increase = 2.0, ess_target_initial = 50, max_ess = 10000, max_sampling_effort = 20, npaths_additional = 10, verbose = true, return_convergence_records = true, return_proposed_paths = false, compute_vcov = true, vcov_threshold = true, kwargs...)

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
    mll_cur = mcem_mll(loglik_target_cur, ImportanceWeights, model.SamplingWeights)

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
        
        params_prop_optim = solve(remake(prob, u0 = Vector(params_cur), p = SMPanelData(model, samplepaths, ImportanceWeights)), Ipopt.Optimizer(); print_level = 0)
        params_prop = params_prop_optim.u

        # calculate the log likelihoods for the proposed parameters
        loglik!(params_prop, loglik_target_prop, SMPanelData(model, samplepaths, ImportanceWeights))
        
        # calculate the marginal log likelihood 
        mll_cur  = mcem_mll(loglik_target_cur , ImportanceWeights, model.SamplingWeights)
        mll_prop = mcem_mll(loglik_target_prop, ImportanceWeights, model.SamplingWeights)

        # compute the ALB and AUB
        if params_prop != params_cur

            # change in mll
            mll_change = mll_prop - mll_cur
    
            # calculate the ASE for ΔQ
            ase = mcem_ase(loglik_target_prop, loglik_target_cur, ImportanceWeights, model.SamplingWeights)
    
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
        
        # initialize Fisher information matrix containers
        fishinf = zeros(Float64, length(params_cur), length(params_cur))
        fish_i1 = zeros(Float64, length(params_cur), length(params_cur))
        fish_i2 = similar(fish_i1)
        
        # container for gradient and hessian
        diffres = DiffResults.HessianResult(params_cur)
    
        ll = pars -> (loglik_AD(pars, ExactDataAD(path, samplingweight, model.hazards, model); neg=false))

        # accumulate Fisher information
        for i in 1:nsubj

            # set importance weight
            samplingweight[1] = model.SamplingWeights[i]

            # number of paths
            npaths = length(samplepaths[i])

            # for accumulating gradients and hessians
            grads = Array{Float64}(undef, length(params_cur), length(samplepaths[i]))

            # reset matrices for accumulating Fisher info contributions
            fill!(fish_i1, 0.0)
            fill!(fish_i2, 0.0)

            # calculate gradient and hessian for paths
            for j in 1:npaths
                path[1] = samplepaths[i][j]
                diffres = ForwardDiff.hessian!(diffres, ll, params_cur)

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

            # sum of outer products of gradients
            for j in 1:npaths
                for k in 1:npaths
                    fish_i2 .+= ImportanceWeights[i][j] * ImportanceWeights[i][k] * grads[:,j] * transpose(grads[:,k])
                end
            end

            fishinf += fish_i1 + fish_i2
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
        model.hazards,
        model.totalhazards,
        model.tmat,
        model.emat,
        model.hazkeys,
        model.subjectindices,
        model.SamplingWeights,
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
