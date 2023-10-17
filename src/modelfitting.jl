# """
#     fit(model::MultistateModel; alg = "ml")

# Fit a model. 
# """ 
# function fit(model::MultistateModel; alg = "ml", nparticles = 100)
    
#     # if sample paths are fully observed, maximize the likelihood directly
#     if all(model.data.obstype .== 1)
        
#         fitted = fit_exact(model)

#     elseif all(model.data.obstype .== 2) & # Multistate Markov model, panel data, no censored state
#         all(isa.(model.hazards, MultistateModels._Exponential) .|| 
#             isa.(model.hazards, MultistateModels._ExponentialPH))
        
#         fitted = fit_markov_interval(model)

#     elseif all(model.data.obstype .== 2) & 
#         !all(isa.(model.hazards, MultistateModels._Exponential) .|| 
#              isa.(model.hazards, MultistateModels._ExponentialPH))
        
#         fitted = fit_semimarkov_interval(model; nparticles = nparticles)
#     end

#     ### mixed likelihoods to add
#     # 1. Mixed panel + fully observed data, no censored states, Markov process
#         # Easy.
#     # 2. Mixed panel + fully observed data, no censored states, semi-Markov process
#         # Easy, just need to append fully observed parts of the path in proposal.
#     # 3. Mixed panel + fully observed data, with censored states, semi-Markov process
#         # Easy, just sample the censored state. 
#     # 4. Mixed panel + fully observed data, with censored states, Markov process
#         # Medium complicated - marginalize over the possible states.

#     # return fitted object
#     return fitted
# end 

"""
    fit(model::MultistateModel)

Fit a multistate model given exactly observed sample paths.
"""
function fit(model::MultistateModel)

    # initialize array of sample paths
    samplepaths = extract_paths(model; self_transitions = false)

    # extract and initialize model parameters
    parameters = flatview(model.parameters)

    # optimize
    optf = OptimizationFunction(loglik, Optimization.AutoForwardDiff())
    prob = OptimizationProblem(optf, parameters, ExactData(model, samplepaths))    
    sol  = solve(prob, Newton())

    # oh eff yes.
    ll = pars -> loglik(pars, ExactData(model, samplepaths); neg=false)
    gradient = ForwardDiff.gradient(ll, sol.u)
    vcov = inv(.-ForwardDiff.hessian(ll, sol.u))

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
    fit(model::MultistateMarkovModel)

Fit a multistate markov model to 
interval censored data (i.e. model.data.obstype .== 2 and all hazards are exponential with possibly piecewise homogeneous transition intensities),
or a mix of panel data and exact jump times.
"""
function fit(model::Union{MultistateMarkovModel,MultistateMarkovModelCensored})

    # containers for bookkeeping TPMs
    books = build_tpm_mapping(model.data)

    # extract and initialize model parameters
    parameters = flatview(model.parameters)

    # optimize the likelihood
    optf = OptimizationFunction(loglik, Optimization.AutoForwardDiff())
    prob = OptimizationProblem(optf, parameters, MPanelData(model, books))
    sol  = solve(prob, Newton())

    # get the variance-covariance matrix
    ll = pars -> loglik(pars, MPanelData(model, books); neg=false)
    gradient = ForwardDiff.gradient(ll, sol.u)
    vcov = inv(.-ForwardDiff.hessian(ll, sol.u))
    
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
fit(model::Union{MultistateSemiMarkovModel,MultistateSemiMarkovModelCensored}; 
npaths_initial = 10, npaths_max = 500, maxiter = 100, tol = 1e-4, α = 0.1, γ = 0.05, κ = 3,
    surrogate_parameter = nothing, ess_target_initial = 50,
    MaxSamplingEffort = 10,
    verbose = true, return_ConvergenceRecords = true, return_ProposedPaths = true)

Fit a semi-Markov model to panel data via Monte Carlo EM. 

# Arguments

- model: multistate model object
- npaths_initial: initial number of sample paths per participant for MCEM
- maxiter: maximum number of MCEM iterations
- tol: tolerance for the change in the MLL, i.e., upper bound of the stopping rule to be ruled out
- α: Standard normal quantile for asymptotic lower bound for ascent
- γ: Standard normal quantile for stopping
- κ: Inflation factor for target ESS per person, ESS_new = ESS_cur * κ
- MaxSamplingEffort: factor of the ESS needed to trigger subsampling
- verbose: print status
- return_ConvergenceRecords: save history throughout the run
- return_ProposedPaths: save latent paths and importance weights
"""
function fit(
    model::Union{MultistateSemiMarkovModel, MultistateSemiMarkovModelCensored}; 
    maxiter = 10, tol = 1e-4, α = 0.1, γ = 0.05, κ = 1.5,
    surrogate_parameter = nothing, ess_target_initial = 100, MaxSamplingEffort = 20,
    verbose = true, return_ConvergenceRecords = true, return_ProposedPaths = true)

    #
    # checks
    if MaxSamplingEffort <= 1
        error("MaxSamplingEffort must be greater than 1.")
    end
    if κ <= 1
        error("κ must be greater than 1.")
    end


    #
    # initialization

    # number of subjects
    nsubj = length(model.subjectindices)

    # containers for bookkeeping TPMs
    books = build_tpm_mapping(model.data)
    
    # extract and initialize model parameters
    params_cur = flatview(model.parameters)

    # number of additional paths
    npaths_additional = 10 # we could let the user chose this value

    # initialize ess target
    ess_target = ess_target_initial
        
    # containers for latent sample paths, proposal and target log likelihoods, importance sampling weights
    TotImportanceWeights = zeros(nsubj)
    ess_cur = zeros(nsubj)
    samplepaths        = Vector{ElasticArray{SamplePath}}(undef, nsubj)
    loglik_surrog      = Vector{ElasticArray{Float64}}(undef, nsubj)
    loglik_target_cur  = Vector{ElasticArray{Float64}}(undef, nsubj)
    loglik_target_prop = Vector{ElasticArray{Float64}}(undef, nsubj)
    ImportanceWeights  = Vector{ElasticArray{Float64}}(undef, nsubj)
    for i in 1:nsubj
        samplepaths[i]        = ElasticArray{SamplePath}(undef, 0)
        loglik_surrog[i]      = ElasticArray{Float64}(undef, 0)
        loglik_target_cur[i]  = ElasticArray{Float64}(undef, 0)
        loglik_target_prop[i] = ElasticArray{Float64}(undef, 0)
        ImportanceWeights[i]  = ElasticArray{Float64}(undef, 0)
    end
    
    # containers for traces
    mll_trace = Vector{Float64}() # marginal loglikelihood
    ess_trace = ElasticArray{Float64}(undef, nsubj, 0) # effective sample size (one per subject)
    parameters_trace = ElasticArray{Float64}(undef, length(flatview(model.parameters)), 0) # parameter estimates


    #
    # Markov surrogate model
    if isnothing(surrogate_parameter)
        # compute the mle of the surrogate model if no parameters are provided
        if verbose
            println("Obtaining the MLE for the Markov surrogate model ...\n")
        end    
        surrogate = fit_surrogate(model)
    else
        surrogate = MarkovSurrogate(model.markovsurrogate.hazards, surrogate_parameter)
    end    

    # transition probability objects for Markov surrogate
    hazmat_book_surrogate = build_hazmat_book(Float64, model.tmat, books[1])
    tpm_book_surrogate = build_tpm_book(Float64, model.tmat, books[1])

    # allocate memory for matrix exponential
    cache = ExponentialUtilities.alloc_mem(similar(hazmat_book_surrogate[1]), ExpMethodGeneric())

    # Solve Kolmogorov equations for TPMs
    for t in eachindex(books[1])

        # compute the transition intensity matrix
        compute_hazmat!(
            hazmat_book_surrogate[t],
            surrogate.parameters,
            surrogate.hazards,
            books[1][t])

        # compute transition probability matrices
        compute_tmat!(
            tpm_book_surrogate[t],
            hazmat_book_surrogate[t],
            books[1][t],
            cache)
    end

    # draw additional sample paths to reach the target ess
    if verbose
        println("Sampling the initial sample paths ...\n")
    end

    for i in 1:nsubj
        DrawAdditionalSamplePaths!(
            i, model, ess_target, ess_cur, MaxSamplingEffort, 
            samplepaths, loglik_surrog, loglik_target_prop, loglik_target_cur, ImportanceWeights, TotImportanceWeights,
            tpm_book_surrogate, hazmat_book_surrogate, books,
            npaths_additional, params_cur, surrogate)
    end

    # get current estimate of marginal log likelihood
    mll_cur = mcem_mll(loglik_target_cur, ImportanceWeights, TotImportanceWeights)

    # optimization function + problem
    optf = OptimizationFunction(loglik, Optimization.AutoForwardDiff())
    prob = OptimizationProblem(optf, params_cur, SMPanelData(model, samplepaths, ImportanceWeights, TotImportanceWeights))
  
    # go on then
    keep_going = true
    iter = 0
    convergence = false

    if verbose
        println("Initial target ESS: $(round(ess_target; digits=2)) per-subject")
        println("Range of the number of sample paths per-subject: ($(min(length.(samplepaths)...)), $(max(length.(samplepaths)...)))")
        println("Loglikelihood: $mll_cur\n")

        println("Starting the MCEM iterations ...\n")
    end

    while keep_going

        # ensure that ess per person is sufficient
        for i in 1:nsubj
            DrawAdditionalSamplePaths!(
                i, model, ess_target, ess_cur, MaxSamplingEffort, 
                samplepaths, loglik_surrog, loglik_target_prop, loglik_target_cur, ImportanceWeights, TotImportanceWeights,
                tpm_book_surrogate, hazmat_book_surrogate, books,
                npaths_additional, params_cur, surrogate)
        end
        
        # recalculate the marginal log likelihood
        mll_cur = mcem_mll(loglik_target_cur, ImportanceWeights, TotImportanceWeights)

        # optimize the monte carlo marginal likelihood
        println("Starting optimization ...")
        #params_prop_optim = solve(remake(prob, u0 = Vector(params_cur), p = SMPanelData(model, samplepaths, ImportanceWeights, TotImportanceWeights)), Newton()) # hessian-based
        params_prop_optim = solve(remake(prob, u0 = Vector(params_cur), p = SMPanelData(model, samplepaths, ImportanceWeights, TotImportanceWeights)), BFGS()) # gradient-based
        params_prop = params_prop_optim.u
        println("Done with optimization.\n")

        # recalculate the log likelihoods
        loglik!(params_prop, loglik_target_prop, SMPanelData(model, samplepaths, ImportanceWeights, TotImportanceWeights))

        # recalculate the marginal log likelihood
        mll_prop = mcem_mll(loglik_target_prop, ImportanceWeights, TotImportanceWeights)

        # change in mll
        mll_change = mll_prop - mll_cur

        # calculate the ASE for ΔQ
        ase = mcem_ase(loglik_target_prop .- loglik_target_cur, ImportanceWeights, TotImportanceWeights)

         # calculate the lower bound for ΔQ
        ascent_lb = quantile(Normal(mll_change, ase), α)
        ascent_ub = quantile(Normal(mll_change, ase), 1-γ)

        if verbose
            println("Iteration: $(iter+1)")
            println("Current target ESS: $(round(ess_target; digits=2)) per-subject")
            println("Range of the number of sample paths per-subject: ($(min(length.(samplepaths)...)), $(max(length.(samplepaths)...)))")
            println("Estimate of the marginal loglikelihood: $mll_cur")
            println("MCEM Asymptotic SE: $ase")
            println("Ascent lower bound: $ascent_lb")
            println("Ascent upper bound: $ascent_ub\n")
            #println("Time: $(Dates.format(now(), "HH:MM"))\n")
        end

        if ascent_lb < 0
            # increase the target ess for the factor κ
            ess_target = κ*ess_target            
        else 
            # cache the results

            # increment the iteration
            iter += 1

            # set proposed parameter and marginal likelihood values to current values
            params_cur = params_prop
            mll_cur = mll_prop
            
            # swap current and proposed log likelihoods
            loglik_target_cur, loglik_target_prop = loglik_target_prop, loglik_target_cur
            
            # recalculate the importance ImportanceWeights and ess
            for i in 1:nsubj
                ImportanceWeights[i] = exp.(loglik_target_cur[i] .- loglik_surrog[i])
                TotImportanceWeights[i] = sum(ImportanceWeights[i])
                NormalizedImportanceWeights = ImportanceWeights[i] ./ TotImportanceWeights[i]
                ess_cur[i] = 1 / sum(NormalizedImportanceWeights .^ 2)
            end

            # save marginal log likelihood, parameters and effective sample size
            append!(parameters_trace, params_cur)
            push!(mll_trace, mll_cur)
            append!(ess_trace, ess_cur)

            # check convergence
            convergence = ascent_ub < tol

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

    # initialize Fisher information matrix
    fisher = zeros(Float64, length(params_cur), length(params_cur), nsubj)
    
    # compute complete data gradients and hessians
    Threads.@threads for i in 1:nsubj

        npaths = length(samplepaths[i])

        path = Array{SamplePath}(undef, 1)
        diffres = DiffResults.HessianResult(params_cur)
        ll = pars -> (loglik(pars, ExactData(model, path); neg=false) * model.SamplingWeights[i])

        grads = Array{Float64}(undef, length(params_cur), length(samplepaths[i]))
        hesns = Array{Float64}(undef, length(params_cur), length(params_cur), npaths)
        fisher_i1 = zeros(Float64, length(params_cur), length(params_cur))
        fisher_i2 = similar(fisher_i1)

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
        fisher_i1 ./= TotImportanceWeights[i]

        for j in 1:npaths
            for k in 1:npaths
                fisher_i2 .+= ImportanceWeights[i][j] * ImportanceWeights[i][k] * grads[:,j] * transpose(grads[:,k])
            end
        end
        fisher_i2 ./= TotImportanceWeights[i]^2

        fisher[:,:,i] = fisher_i1 + fisher_i2
    end

    # get the variance-covariance matrix
    vcov = inv(reduce(+, fisher, dims = 3)[:,:,1])

    # return convergence records
    ConvergenceRecords = return_ConvergenceRecords ? (mll_trace=mll_trace, ess_trace=ess_trace, parameters_trace=parameters_trace) : nothing

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


"""
fit_surrogate(model::MultistateSemiMarkovModel)

Fit a Markov surrogate model. 

# Arguments

- model: multistate model object
"""
function fit_surrogate(model::MultistateSemiMarkovModel)
    surrogate = MultistateModels.MultistateMarkovModel(
        model.data,
        model.markovsurrogate.parameters,
        model.markovsurrogate.hazards,
        model.totalhazards,
        model.tmat,
        model.emat,
        model.hazkeys,
        model.subjectindices,
        model.SamplingWeights,
        model.CensoringPatterns,
        model.markovsurrogate,
        model.modelcall);
    set_crude_init!(surrogate)
    return fit(surrogate)
end


"""
fit_surrogate(model::MultistateSemiMarkovModelCensored)

Fit a Markov surrogate model with censored states. 

# Arguments

- model: multistate model object
"""
function fit_surrogate(model::MultistateSemiMarkovModelCensored)
    surrogate = MultistateModels.MultistateMarkovModelCensored(
        model.data,
        model.markovsurrogate.parameters,
        model.markovsurrogate.hazards,
        model.totalhazards,
        model.tmat,
        model.emat,
        model.hazkeys,
        model.subjectindices,
        model.SamplingWeights,
        model.CensoringPatterns,
        model.markovsurrogate,
        model.modelcall);
    set_crude_init!(surrogate)
    return fit(surrogate)
end