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

Fit a semi-Markov model to panel data via Monte Carlo maximum marginal likelihood.
"""
function fit_semimarkov_interval(model::MultistateModel; nparticles)

    # containers for bookkeeping TPMs
    books = build_tpm_mapping(model.data)

    # extract and initialize model parameters
    parameters = flatview(model.parameters)

end