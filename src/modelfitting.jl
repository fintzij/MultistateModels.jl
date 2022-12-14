"""
    fit(model::MultistateModel; alg = "ml")

Fit a model. 
""" 
function fit(model::MultistateModel; alg = "ml", nparticles = 100)
    
    # if sample paths are fully observed, maximize the likelihood directly
    if all(model.data.obstype .== 1)
        
        fitted = fit_exact(model)

    elseif all(model.data.obstype .== 2) & 
        # multistate Markov model or competing risks model
        all(isa.(model.hazards, MultistateModels._Exponential) .|| 
            isa.(model.hazards, MultistateModels._ExponentialPH))
        fitted = fit_markov_interval(model)
    elseif all(model.data.obstype .== 2) & 
        !all(isa.(model.hazards, MultistateModels._Exponential) .|| 
             isa.(model.hazards, MultistateModels._ExponentialPH))
        fitted = fit_semimarkov_interval(model; nparticles = nparticles)
    end

    # return fitted object
    return fitted
end 

"""
    fit_exact(model::MultistateModel)

Fit a multistate model given exactly observed sample paths.
"""
function fit_exact(model::MultistateModel)

    # initialize array of sample paths
    samplepaths = extract_paths(model)

    # extract and initialize model parameters
    parameters = flatview(model.parameters)

    # optimize
    optf = OptimizationFunction(loglik, Optimization.AutoForwardDiff())
    prob = OptimizationProblem(optf, parameters, ExactData(model, samplepaths))    
    sol  = solve(prob, Newton())

    # oh eff yes.
    ll = pars -> loglik(pars, ExactData(model, samplepaths))
    vcov = inv(ForwardDiff.hessian(ll, sol.u))

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
        model.markovsurrogate,
        model.modelcall)
end


"""
    fit_markov_interval(model::MultistateModel)

Fit a multistate markov model to interval censored data (i.e. model.data.obstype .== 2 and all hazards are exponential with possibly piecewise homogeneous transition intensities).
"""
function fit_markov_interval(model::MultistateModel)
    
    # containers for bookkeeping TPMs
    books = build_tpm_mapping(model.data)

    # extract and initialize model parameters
    parameters = flatview(model.parameters)

    # optimize the likelihood
    optf = OptimizationFunction(loglik, Optimization.AutoForwardDiff())
    prob = OptimizationProblem(optf, parameters, MPanelData(model, books))
    sol  = solve(prob, Newton())

    # get the variance-covariance matrix
    ll = pars -> loglik(pars, MPanelData(model, books))
    vcov = inv(ForwardDiff.hessian(ll, sol.u))

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
        model.markovsurrogate,
        model.modelcall)
end

"""
    fit_semimarkov_interval(model::MultistateModel; nparticles)

Fit a semi-Markov model to panel data via Monte Carlo EM. 

Latent paths are sampled via MCMC and are subsampled at points t_k = x_1 + ... + x_k, where x_i - 1 ~ Pois(subrate * k ^ subscale). The arguments subrate and subscale default to 1 and 0.5, respectively.

# Arguments

- model: multistate model object
- nparticles: initial number of particles per participant for MCEM
- subrate: Poisson rate for subsampling points
- subscale: scaling of the thinning rate for subsampling points
- maxiter: maximum number of MCEM iterations
- trace: return traces of the log-likelihood and parameters for diagnostics
"""
function fit_semimarkov_interval(model::MultistateModel; nparticles = 10, subrate = 1, subscale = 0.5, maxiter = 50, trace = true)

    # number of subjects
    nsubj = length(model.subjectindices)

    # containers for bookkeeping TPMs
    books = build_tpm_mapping(model.data)

    # transition probability objects for markov surrogate
    hazmat_book = build_hazmat_book(Float64, model.tmat, books[1])
    tpm_book = build_tpm_book(Float64, model.tmat, books[1])

    # allocate memory for matrix exponential
    cache = ExponentialUtilities.alloc_mem(similar(hazmat_book[1]), ExpMethodGeneric())

    # Solve Kolmogorov equations for TPMs
    for t in eachindex(books[1])

        # compute the transition intensity matrix
        compute_hazmat!(
            hazmat_book[t],
            model.markovsurrogate.parameters,
            model.markovsurrogate.hazards,
            books[1][t])

        # compute transition probability matrices
        compute_tmat!(
            tpm_book[t],
            hazmat_book[t],
            books[1][t],
            cache)
    end

    # initialize latent sample paths
    samplepaths = ElasticArray{SamplePath}(undef, nsubj, nparticles)

    # initialize proposal log likelihoods
    loglik_prop = ElasticArray{Float64}(undef, nsubj, nparticles)

    # draw sample paths
    Threads.@threads for i in 1:nsubj
        for j in 1:nparticles
            samplepaths[i,j] = draw_samplepath(i, model, tpm_book, hazmat_book, books[2])
        end
    end

    # calculate likelihoods for importance sampling


    # extract and initialize model parameters
    parameters = flatview(model.parameters)

end