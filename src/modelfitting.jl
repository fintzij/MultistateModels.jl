"""
    fit(model::MultistateModel; constraints = nothing, compute_vcov = true)

Fit a multistate model given exactly observed sample paths.
"""
function fit(model::MultistateModel; constraints = nothing, verbose = true, compute_vcov = true)

    # initialize array of sample paths
    samplepaths = extract_paths(model; self_transitions = false)

    # extract and initialize model parameters
    parameters = flatview(model.parameters)

    # parse constraints, or not, and solve
    if isnothing(constraints) 
        # get estimates
        optf = OptimizationFunction(loglik, Optimization.AutoForwardDiff())
        prob = OptimizationProblem(optf, parameters, ExactData(model, samplepaths))
        sol  = solve(prob, Newton())
        
        # get vcov
        if compute_vcov
            ll = pars -> loglik(pars, ExactData(model, samplepaths); neg=false)
            gradient = ForwardDiff.gradient(ll, sol.u)
            vcov = pinv(.-ForwardDiff.hessian(ll, sol.u))
            vcov[isapprox.(vcov, 0.0; atol = eps(Float64))] .= 0.0
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
        sol  = solve(prob, IPNewton())

        # no hessian when there are constraints
        @warn "No covariance matrix is returned when constraints are provided."
        vcov = nothing
    end

    # wrap results
    return MultistateModelFitted(
        model.data,
        VectorOfVectors(sol.u, model.parameters.elem_ptr),
        -sol.minimum,
        vcov,
        model.hazards,
        model.totalhazards,
        model.tmat,
        model.hazkeys,
        model.subjectindices,
        model.SamplingWeights,
        model.CensoringPatterns,
        model.markovsurrogate,
        nothing, # ConvergenceRecords::Union{Nothing, NamedTuple}
        nothing, # ProposedPaths::Union{Nothing, NamedTuple}
        model.modelcall)
end


"""
    fit(model::MultistateMarkovModel; constraints = nothing, verbose = true, compute_vcov = true)

Fit a multistate markov model to interval censored data (i.e. model.data.obstype .== 2 and all hazards are exponential with possibly piecewise homogeneous transition intensities), or a mix of panel data and exact jump times.
"""
function fit(model::Union{MultistateMarkovModel,MultistateMarkovModelCensored}; constraints = nothing, verbose = true, compute_vcov = true)

    # containers for bookkeeping TPMs
    books = build_tpm_mapping(model.data)

    # extract and initialize model parameters
    parameters = flatview(model.parameters)

    # parse constraints, or not, and solve
    if isnothing(constraints)
        # get estimates
        optf = OptimizationFunction(loglik, Optimization.AutoForwardDiff())
        prob = OptimizationProblem(optf, parameters, MPanelData(model, books))
        sol  = solve(prob, Newton())

        # get vcov
        if compute_vcov
            # get the variance-covariance matrix
            ll = pars -> loglik(pars, MPanelData(model, books); neg=false)
            gradient = ForwardDiff.gradient(ll, sol.u)
            vcov = pinv(.-ForwardDiff.hessian(ll, sol.u))
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

        optf = OptimizationFunction(loglik, Optimization.AutoForwardDiff(), cons = consfun_markov)
        prob = OptimizationProblem(optf, parameters, MPanelData(model, books), lcons = constraints.lcons, ucons = constraints.ucons)
        sol  = solve(prob, IPNewton())

        # no hessian when there are constraints
        @warn "No covariance matrix is returned when constraints are provided."
        vcov = nothing
    end

    # wrap results
    return MultistateModelFitted(
        model.data,
        VectorOfVectors(sol.u, model.parameters.elem_ptr),
        -sol.minimum,
        vcov,
        model.hazards,
        model.totalhazards,
        model.tmat,
        model.hazkeys,
        model.subjectindices,
        model.SamplingWeights,
        model.CensoringPatterns,
        model.markovsurrogate,
        nothing, # ConvergenceRecords::Union{Nothing, NamedTuple}
        nothing, # ProposedPaths::Union{Nothing, NamedTuple}
        model.modelcall)
end


"""
fit(model::Union{MultistateSemiMarkovModel, MultistateSemiMarkovModelCensored}; optimize_surrogate = true, constraints = nothing, surrogate_constraints = nothing, surrogate_parameters = nothing,  maxiter = 100, tol = 1e-3, α = 0.01, γ = 0.05, κ = 4/3, ess_target_initial = 100, MaxSamplingEffort = 20, npaths_additional = 10, verbose = true, return_ConvergenceRecords = true, return_ProposedPaths = false, compute_vcov = true)

Fit a semi-Markov model to panel data via Monte Carlo EM.

# Arguments

- model: multistate model object
- optimize_surrogate: should the parameters Markov surrogate for proposing paths be set to the MLE? defaults to true
- constraints: tuple for specifying parameter constraints
- surrogate_constraints: tuple for specifying parameter constraints for the Markov surrogate
- nparticles: initial number of particles per participant for MCEM
- npaths_initial: initial number of sample paths per participant for MCEM
- maxiter: maximum number of MCEM iterations
- tol: tolerance for the change in the MLL, i.e., upper bound of the stopping rule to be ruled out
- α: standard normal quantile for asymptotic lower bound for ascent
- γ: standard normal quantile for stopping the MCEM algorithm
- κ: Inflation factor for target ESS per person, ESS_new = ESS_cur * κ
- MaxSamplingEffort: factor of the ESS at which to break the loop for sampling additional paths
- npaths_additional: increment for number of additional paths when augmenting the pool of paths
- verbose: print status
- return_ConvergenceRecords: save history throughout the run
- return_ProposedPaths: save latent paths and importance weights
- compute_vcov: should the variance-covariance matrix be computed at the final estimates? defaults to true.
"""
function fit(model::Union{MultistateSemiMarkovModel, MultistateSemiMarkovModelCensored}; optimize_surrogate = true, constraints = nothing, surrogate_constraints = nothing, surrogate_parameters = nothing,  maxiter = 100, tol = 1e-3, α = 0.01, γ = 0.05, κ = 4/3, ess_target_initial = 100, MaxSamplingEffort = 20, npaths_additional = 10, verbose = true, return_ConvergenceRecords = true, return_ProposedPaths = false, compute_vcov = true)

    # check that constraints for the initial values are satisfied
    if !isnothing(constraints)
        # create constraint function and check that constraints are satisfied at the initial values
        _constraints = deepcopy(constraints)
        consfun_semimarkov = parse_constraints(_constraints.cons, model.hazards; consfun_name = :consfun_semimarkov)

        initcons = consfun_semimarkov(zeros(length(constraints.cons)), flatview(model.parameters), nothing)
        badcons = findall(initcons .< constraints.lcons .|| initcons .> constraints.ucons)
        if length(badcons) > 0
            @error "Constraints $badcons are violated at the initial parameter values."
        end
    end

    # check that MaxSamplingEffort is greater than 1
    if MaxSamplingEffort <= 1
        error("MaxSamplingEffort must be greater than 1.")
    end

    # check that κ is greater than 1
    if κ <= 1
        error("κ must be greater than 1.")
    end

    # MCEM initialization
    keep_going = true
    iter = 0
    convergence = false

    # number of subjects
    nsubj = length(model.subjectindices)

    # extract and initialize model parameters
    params_cur = flatview(model.parameters)

    # initialize ess target
    ess_target = ess_target_initial

    # containers for latent sample paths, proposal and target log likelihoods, importance sampling weights
    ess_cur = zeros(nsubj)
    psis_pareto_k = zeros(nsubj)

    # initialize containers
    samplepaths     = [sizehint!(Vector{SamplePath}(), ess_target_initial * MaxSamplingEffort * 20) for i in 1:nsubj]
    loglik_surrog   = [sizehint!(Vector{Float64}(undef, 0), 
        ess_target_initial * MaxSamplingEffort * 2) for i in 1:nsubj]
    loglik_target_cur  = [sizehint!(Vector{Float64}(undef, 0), 
        ess_target_initial * MaxSamplingEffort * 2) for i in 1:nsubj]
    loglik_target_prop = [sizehint!(Vector{Float64}(undef, 0), 
        ess_target_initial * MaxSamplingEffort * 2) for i in 1:nsubj]
    ImportanceWeights  = [sizehint!(Vector{Float64}(undef, 0), 
        ess_target_initial * MaxSamplingEffort * 2) for i in 1:nsubj]

    # containers for traces
    mll_trace = Vector{Float64}() # marginal loglikelihood
    ess_trace = ElasticArray{Float64, 2}(undef, nsubj, 0) # effective sample size (one per subject)
    parameters_trace = ElasticArray{Float64, 2}(undef, length(flatview(model.parameters)), 0) # parameter estimates

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
        compute_hazmat!(hazmat_book_surrogate[t], surrogate.parameters, surrogate.hazards, books[1][t])
        # compute transition probability matrices
        compute_tmat!(tpm_book_surrogate[t], hazmat_book_surrogate[t], books[1][t], cache)
    end

    # target ess
    ess_target = ess_target_initial

    # draw sample paths until the target ess is reached 
    if verbose
        println("Initializing sample paths ...\n")
    end

    DrawSamplePaths!(model; 
        ess_target = ess_target, 
        ess_cur = ess_cur, 
        MaxSamplingEffort = MaxSamplingEffort,
        samplepaths = samplepaths, 
        loglik_surrog = loglik_surrog, 
        loglik_target_prop = loglik_target_prop, 
        loglik_target_cur = loglik_target_cur, 
        ImportanceWeights = ImportanceWeights, 
        tpm_book_surrogate = tpm_book_surrogate, 
        hazmat_book_surrogate = hazmat_book_surrogate, 
        books = books, 
        npaths_additional = npaths_additional, 
        params_cur = params_cur, 
        surrogate = surrogate, 
        psis_pareto_k = psis_pareto_k)
    
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
        println("Range of the number of sample paths per-subject: [$(ceil(ess_target)), $(max(length.(samplepaths)...))]")
        println("Estimate of the marginal log-likelihood: $(round(mll_cur;digits=3))\n")

        println("Starting Monte Carlo EM...\n")
    end

    # counter for whether successive iterations of ascent UB below tol
    convergence_counter = 0

    # start algorithm
    while keep_going

        # recalculate the marginal log likelihood 
        mll_cur = mcem_mll(loglik_target_cur, ImportanceWeights, model.SamplingWeights)

        # optimize the monte carlo marginal likelihood
        if verbose 
            println("Optimizing...")
        end

        if isnothing(constraints)
            params_prop_optim = solve(remake(prob, u0 = Vector(params_cur), p = SMPanelData(model, samplepaths, ImportanceWeights)), Newton()) # hessian-based
        else
            params_prop_optim = solve(remake(prob, u0 = Vector(params_cur), p = SMPanelData(model, samplepaths, ImportanceWeights)), IPNewton())
        end

        params_prop = params_prop_optim.u

        # just make sure they're not equal
        if params_prop != params_cur 
            # recalculate the log likelihoods
            loglik!(params_prop, loglik_target_prop, SMPanelData(model, samplepaths, ImportanceWeights); use_sampling_weight = false)
    
            # recalculate the marginal log likelihood
            mll_prop = mcem_mll(loglik_target_prop, ImportanceWeights, model.SamplingWeights)
    
            # change in mll
            mll_change = mll_prop - mll_cur
    
            # calculate the ASE for ΔQ
            ase = mcem_ase(loglik_target_prop, loglik_target_cur, ImportanceWeights, model.SamplingWeights)
    
             # calculate the lower bound for ΔQ
            ascent_lb = quantile(Normal(mll_change, ase), α)
            ascent_ub = quantile(Normal(mll_change, ase), 1-γ)
        else
            loglik_target_prop = loglik_target_cur
            mll_prop = mll_cur
            mll_change = 0
            ase = 0
            ascent_lb = 0
            ascent_ub = 0
            convergence_counter = 2
        end

        if ascent_lb < 0
            # increase the target ess for the factor κ
            ess_target = ceil(κ*ess_target)

            # ensure that ess per person is sufficient
            DrawSamplePaths!(model; 
                ess_target = ess_target, 
                ess_cur = ess_cur, 
                MaxSamplingEffort = MaxSamplingEffort,
                samplepaths = samplepaths, 
                loglik_surrog = loglik_surrog, 
                loglik_target_prop = loglik_target_prop, 
                loglik_target_cur = loglik_target_cur, 
                ImportanceWeights = ImportanceWeights,   tpm_book_surrogate = tpm_book_surrogate,   hazmat_book_surrogate = hazmat_book_surrogate, 
                books = books, 
                npaths_additional = npaths_additional, 
                params_cur = params_cur, 
                surrogate = surrogate,
                psis_pareto_k = psis_pareto_k)
        else
            # increment the iteration
            iter += 1

            # set proposed parameter and marginal likelihood values to current values
            params_cur, params_prop = params_prop, params_cur
            mll_cur, mll_prop = mll_prop, mll_cur

            # update the current log-likelihoods
            loglik_target_cur, loglik_target_prop = loglik_target_prop, loglik_target_cur
            
            # recalculate the importance ImportanceWeights and ess
            for i in 1:nsubj
                if length(ImportanceWeights[i]) != 1
                    logweights = reshape(loglik_target_cur[i] - loglik_surrog[i], 1, length(loglik_target_cur[i]), 1) 
                    psiw = psis(logweights; source = "other");
                # save importance weights and ess
                    copyto!(ImportanceWeights[i], psiw.weights)
                    ess_cur[i] = psiw.ess[1]
                    psis_pareto_k[i] = psiw.pareto_k[1]
                end
            end

            # save marginal log likelihood, parameters and effective sample size
            append!(parameters_trace, params_cur)
            push!(mll_trace, mll_cur)
            append!(ess_trace, ess_cur)

            if verbose
                println("Iteration: $iter")
                println("Current target ESS: $(round(ess_target;digits=2)) per-subject")
                println("Range of the number of sample paths per-subject: [$(ceil(ess_target)), $(max(length.(samplepaths)...))]")
                println("Current estimate of the marginal log-likelihood: $(round(mll_cur;digits=3))")
                println("Reweighted prior estimate of the marginal log-likelihood: $(round(mll_prop;digits=3))")
                println("Change in marginal log-likelihood: $(round(mll_change;sigdigits=3))")
                println("MCEM Asymptotic SE: $(round(ase;sigdigits=3))")
                println("Ascent lower bound: $(round(ascent_lb; sigdigits=3))")
                println("Ascent upper bound: $(round(ascent_ub; sigdigits=3))\n")
                #println("Time: $(Dates.format(now(), "HH:MM"))\n")
            end

            # check convergence
            if ascent_ub < tol
                convergence_counter += 1
            else
                convergence_counter = 0
            end

            # check if convergence in successive iterations
            convergence = convergence_counter > 1

            # check whether to stop
            if convergence
                keep_going = false
                if verbose
                    println("The MCEM algorithm has converged.\n")
                end
            end
            if iter >= maxiter
                keep_going = false
                @warn "The maximum number of iterations ($maxiter) has been reached.\n"
            end
        end
    end

    if !convergence
        @warn "MCEM did not converge."
    end

    # hessian
    if !isnothing(constraints)

        # no hessian when there are constraints
        @warn "No covariance matrix is returned when constraints are provided."
        vcov = nothing
    
    elseif convergence && compute_vcov
        if verbose
            println("Computing variance-covariance matrix at final estimates.")
        end

        # initialize Fisher information matrix
        fisher = zeros(Float64, length(params_cur), length(params_cur), nsubj)
        
        # set up containers for path and sampling weight
        path = Array{SamplePath}(undef, 1)
        samplingweight = Vector{Float64}(undef, 1)
    
        # container for gradient and hessian
        diffres = DiffResults.HessianResult(params_cur)
        fisher_i1 = zeros(Float64, length(params_cur), length(params_cur))
        fisher_i2 = similar(fisher_i1)
    
        # define objective
        ll = pars -> (loglik(pars, ExactDataAD(path, samplingweight, model.hazards, model); neg=false))

        # accumulate Fisher information
        for i in 1:nsubj

            # set importance weight
            samplingweight[1] = model.SamplingWeights[i]

            # number of paths
            npaths = length(samplepaths[i])

            # for accumulating gradients and hessians
            grads = Array{Float64}(undef, length(params_cur), length(samplepaths[i]))
            hesns = Array{Float64}(undef, length(params_cur), length(params_cur), npaths)

            # reset matrices for accumulating Fisher info contributions
            fill!(fisher_i1, 0.0)
            fill!(fisher_i2, 0.0)

            # calculate gradient and hessian for paths
            for j in 1:npaths
                path[1] = samplepaths[i][j]
                diffres = ForwardDiff.hessian!(diffres, ll, params_cur)

                # grab hessian and gradient
                hesns[:,:,j] = DiffResults.hessian(diffres)
                grads[:,j] = DiffResults.gradient(diffres)
            end

            # accumulate
            for j in 1:npaths
                fisher_i1 .+= ImportanceWeights[i][j] * (-hesns[:,:,j] - grads[:,j] * transpose(grads[:,j]))
            end

            for j in 1:npaths
                for k in 1:npaths
                    fisher_i2 .+= ImportanceWeights[i][j] * ImportanceWeights[i][k] * grads[:,j] * transpose(grads[:,k])
                end
            end

            fisher[:,:,i] = fisher_i1 + fisher_i2
        end

        # get the variance-covariance matrix
        fisherinfo = reduce(+, fisher, dims = 3)[:,:,1]
        fisherinfo[findall(isapprox.(fisherinfo, 0.0; atol = eps(Float64)))] .= 0.0
        vcov = pinv(fisherinfo)
        vcov[findall(isapprox.(vcov, 0.0; atol = eps(Float64)))] .= 0.0
        vcov = Symmetric(vcov)
    else
        vcov = nothing
    end

    # return convergence records
    ConvergenceRecords = return_ConvergenceRecords ? (mll_trace=mll_trace, ess_trace=ess_trace, parameters_trace=parameters_trace, psis_pareto_k = psis_pareto_k) : nothing

    # return sampled paths and importance weights
    ProposedPaths = return_ProposedPaths ? (paths=samplepaths, weights=ImportanceWeights) : nothing

    # wrap results
    return MultistateModelFitted(
        model.data,
        VectorOfVectors(params_cur, model.parameters.elem_ptr),
        mll_cur,
        vcov,
        model.hazards,
        model.totalhazards,
        model.tmat,
        model.hazkeys,
        model.subjectindices,
        model.SamplingWeights,
        model.CensoringPatterns,
        surrogate,
        ConvergenceRecords,
        ProposedPaths,
        model.modelcall)
end
