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
    nparticles = 10, npaths_max = 500, maxiter = 100, tol = 1e-4, α = 0.1, γ = 0.05, κ = 3,
    surrogate_parameter = nothing, ess_target_initial = 50,
    MaxSamplingEffort = 10,
    verbose = true, return_ConvergenceRecords = true, return_ProposedPaths = true)

Fit a semi-Markov model to panel data via Monte Carlo EM. 

# Arguments

- model: multistate model object
- nparticles: initial number of particles per participant for MCEM
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
    nparticles = 10, maxiter = 100, tol = 1e-4, α = 0.1, γ = 0.05, κ = 1.5,
    surrogate_parameter = nothing, ess_target_initial = 100, MaxSamplingEffort = 10,
    verbose = true, return_ConvergenceRecords = true, return_ProposedPaths = true)

    # number of subjects
    nsubj = length(model.subjectindices)

    # containers for bookkeeping TPMs
    books = build_tpm_mapping(model.data)

    # create Markov surrogate model
    if isnothing(surrogate_parameter)      

        if verbose
            println("Obtaining the MLE for the Markov surrogate model ...\n")
        end

        # get mle of surrogate model
        model_markov = MultistateMarkovModelCensored(
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
            model.modelcall)
        model_markov_fitted = fit(model_markov)
        surrogate_parameter=model_markov_fitted.parameters
    end

    surrogate = MarkovSurrogate(model.markovsurrogate.hazards, surrogate_parameter)

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

    # containers for latent sample paths, proposal and target log likelihoods, importance sampling weights
    samplepaths        = Vector{ElasticArray{SamplePath}}(undef, nsubj)
    loglik_surrog      = Vector{ElasticArray{Float64}}(undef, nsubj)
    loglik_target_cur  = Vector{ElasticArray{Float64}}(undef, nsubj)
    loglik_target_prop = Vector{ElasticArray{Float64}}(undef, nsubj)
    ImportanceWeights  = Vector{ElasticArray{Float64}}(undef, nsubj)
    for i in 1:nsubj
        samplepaths[i]        = ElasticArray{SamplePath}(undef, nparticles)
        loglik_surrog[i]      = ElasticArray{Float64}(undef, nparticles)
        loglik_target_cur[i]  = ElasticArray{Float64}(undef, nparticles)
        loglik_target_prop[i] = ElasticArray{Float64}(undef, nparticles)
        ImportanceWeights[i]  = ElasticArray{Float64}(undef, nparticles)
    end
    
    # containers for traces
    mll_trace = Vector{Float64}() # marginal loglikelihood
    ess_trace = ElasticArray{Float64}(undef, nsubj, 0) # effective sample size (one per subject)
    parameters_trace = ElasticArray{Float64}(undef, length(flatview(model.parameters)), 0) # parameter estimates

    # draw sample paths
    for i in 1:nsubj
        for j in 1:nparticles
            samplepaths[i][j]       = draw_samplepath(i, model, tpm_book_surrogate, hazmat_book_surrogate, books[2])
            loglik_surrog[i][j]     = loglik(surrogate.parameters, samplepaths[i][j], surrogate.hazards, model) * model.SamplingWeights[i]
            loglik_target_cur[i][j] = loglik(model.parameters, samplepaths[i][j], model.hazards, model) * model.SamplingWeights[i]
            ImportanceWeights[i][j] = exp(loglik_target_cur[i][j] - loglik_surrog[i][j])
        end
    end

    # compute ess
    TotImportanceWeights = zeros(nsubj)
    ess_cur = zeros(nsubj)
    for i in 1:nsubj
        TotImportanceWeights[i] = sum(ImportanceWeights[i])
        NormalizedImportanceWeights = ImportanceWeights[i] ./ TotImportanceWeights[i]
        ess_cur[i] = 1 / sum(NormalizedImportanceWeights .^ 2)
    end

    # while ess too low, sample additional paths
    ess_target = ess_target_initial
    npaths_additional = nparticles    

    if verbose
        println("Sampling the initial sample paths ...\n")
    end

    for i in 1:nsubj
        # increase the number of sample paths as long as necessary
        while ess_cur[i] < ess_target
            # if too many sample paths, resample
            n_path_max = MaxSamplingEffort*ess_target
            if length(samplepaths[i]) > n_path_max
                @warn "More than $n_path_max sample paths are required to obtain ess>$ess_target for individual $i."
                break
            end
            npaths = length(samplepaths[i])
            append!(samplepaths[i], Vector{SamplePath}(undef, npaths_additional))
            append!(loglik_surrog[i], zeros(npaths_additional))
            append!(loglik_target_prop[i], zeros(npaths_additional))
            append!(loglik_target_cur[i], zeros(npaths_additional))
            append!(ImportanceWeights[i], zeros(npaths_additional))
            for j in npaths.+(1:npaths_additional)
                samplepaths[i][j]       = draw_samplepath(i, model, tpm_book_surrogate, hazmat_book_surrogate, books[2])
                loglik_surrog[i][j]     = loglik(surrogate.parameters, samplepaths[i][j], surrogate.hazards, model) * model.SamplingWeights[i]
                loglik_target_cur[i][j] = loglik(model.parameters, samplepaths[i][j], model.hazards, model) * model.SamplingWeights[i]
                ImportanceWeights[i][j] = exp(loglik_target_cur[i][j] - loglik_surrog[i][j])
            end
            # update ess
            TotImportanceWeights[i] = sum(ImportanceWeights[i])
            NormalizedImportanceWeights = ImportanceWeights[i] ./ TotImportanceWeights[i]
            ess_cur[i] = 1 / sum(NormalizedImportanceWeights .^ 2)
        end
    end

    # get current estimate of marginal log likelihood
    mll_cur = mcem_mll(loglik_target_cur, ImportanceWeights, TotImportanceWeights)

    # extract and initialize model parameters
    params_cur = flatview(model.parameters)

    # optimization function + problem
    optf = OptimizationFunction(loglik, Optimization.AutoForwardDiff())
    prob = OptimizationProblem(optf, params_cur, SMPanelData(model, samplepaths, ImportanceWeights, TotImportanceWeights))
  
    # go on then
    keep_going = true
    iter = 0
    convergence = false

    if verbose
        println("Starting the MCEM iterations ...\n")
    end

    while keep_going
        #println(iter)        

        # ensure that ess per person is sufficient
        for i in 1:nsubj
            # increase the number of sample paths as long as necessary
            while ess_cur[i] < ess_target
                # if too many sample paths, resample
                n_path_max = MaxSamplingEffort*ess_target
                if length(samplepaths[i]) > n_path_max
                    @warn "More than $n_path_max sample paths are required to obtain ess>$ess_target for individual $i."
                    break
                    # npaths = Integer(round(n_path_max/2))
                    # path_indices = wsample(1:length(samplepaths[i]), NormalizedImportanceWeights, npaths) # sample with replacements
                    # samplepaths[i] = samplepaths[i][path_indices]
                    # loglik_surrog[i] = ones(npaths)
                    # loglik_target_cur[i] = ones(npaths)
                    # ImportanceWeights[i] = ones(npaths) # ./ npaths
                end
                npaths = length(samplepaths[i])
                append!(samplepaths[i], Vector{SamplePath}(undef, npaths_additional))
                append!(loglik_surrog[i], zeros(npaths_additional))
                append!(loglik_target_prop[i], zeros(npaths_additional))
                append!(loglik_target_cur[i], zeros(npaths_additional))
                append!(ImportanceWeights[i], zeros(npaths_additional))
                for j in npaths.+(1:npaths_additional)
                    samplepaths[i][j]       = draw_samplepath(i, model, tpm_book_surrogate, hazmat_book_surrogate, books[2])
                    loglik_surrog[i][j]     = loglik(surrogate.parameters, samplepaths[i][j], surrogate.hazards, model) * model.SamplingWeights[i]
                    loglik_target_cur[i][j] = loglik(VectorOfVectors(params_cur, model.parameters.elem_ptr), samplepaths[i][j], model.hazards, model) * model.SamplingWeights[i]
                    ImportanceWeights[i][j] = exp(loglik_target_cur[i][j] - loglik_surrog[i][j])
                end
                # update ess
                TotImportanceWeights[i] = sum(ImportanceWeights[i])
                NormalizedImportanceWeights = ImportanceWeights[i] ./ TotImportanceWeights[i]
                ess_cur[i] = 1 / sum(NormalizedImportanceWeights .^ 2)
            end
        end

        # recalculate the marginal log likelihood
        mll_cur = mcem_mll(loglik_target_cur, ImportanceWeights, TotImportanceWeights)

        # optimize the monte carlo marginal likelihood
        params_prop_optim = solve(remake(prob, u0 = Vector(params_cur), p = SMPanelData(model, samplepaths, ImportanceWeights, TotImportanceWeights)), Newton())
        params_prop = params_prop_optim.u

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
            #println("Monte Carlo sample size: $nparticles")
            println("Loglikelihood: $mll_cur")
            println("MCEM Asymptotic SE: $ase")
            println("Smallest ESS per-subject: $(round(min(ess_cur...), digits = 2))")
            println("Largest number of sample paths per-subject: $(max(length.(samplepaths)...))")
            println("Ascent lower bound: $ascent_lb")
            println("Ascent upper bound: $ascent_ub\n")
            #println("Time: $(Dates.format(now(), "HH:MM"))\n")
        end

        if ascent_lb < 0
            # increase the target ess per individual
            ess_target = κ*ess_target
            
        else 
            # cache results

            # increment the iteration
            iter += 1

            # set proposed values to current values
            params_cur = params_prop
            
            # swap current and proposed log likelihoods
            loglik_target_cur, loglik_target_prop = loglik_target_prop, loglik_target_cur

            # recalculate the importance ImportanceWeights and ess
            for i in 1:nsubj
                ImportanceWeights[i] = exp.(loglik_target_cur[i] .- loglik_surrog[i])
                TotImportanceWeights[i] = sum(ImportanceWeights[i])
                NormalizedImportanceWeights = ImportanceWeights[i] ./ TotImportanceWeights[i]
                ess_cur[i] = 1 / sum(NormalizedImportanceWeights .^ 2)
            end

            # swap current value of the marginal log likelihood
            mll_cur = mll_prop

            # save marginal log likelihood, effective sample size and parameters
            push!(mll_trace, mll_cur)
            append!(ess_trace, ess_cur)
            append!(parameters_trace, params_cur)

            # check convergence
            convergence = ascent_ub < tol

            # check whether to stop 
            if convergence
                keep_going = false
                if verbose
                    println("The MCEM algorithm has converged.\n")
                end
            end
            if iter > maxiter
                keep_going = false
                @warn "The maximum number of iterations ($maxiter) has been reached."
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