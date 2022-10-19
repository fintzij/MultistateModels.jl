"""
    fit(model::MultistateModel; alg = "ml")

Fit a model. 
""" 
function fit(model::MultistateModel; alg = "ml")
    
    # if sample paths are fully observed, maximize the likelihood directly
    if all(model.data.obstype .== 1)
        
        fitted = fit_exact(model)

    elseif all(model.data.obstype .== 2) & 
        # multistate Markov model or competing risks model
        all(isa.(model.hazards, MultistateModels._Exponential) .|| 
            isa.(model.hazards, MultistateModels._ExponentialPH) || 
            sum(mapslices(sum, model.tmat, dims = [2,]) .!= 0) == 1) 
        fitted = fit_homog_markov_interval(model)
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
    prob = OptimizationProblem(optf, parameters, ExactData(samplepaths, model))    
    sol  = solve(prob, Newton())

    # oh eff yes.
    ll = pars -> MultistateModels.loglik(pars, MultistateModels.ExactData(samplepaths, model))
    vcov = inv(ForwardDiff.hessian(ll, sol.u))

    # wrap results
    # return (pars, vcov, -sol.minimum): 
end


"""
    fit_markov_interval(model::MultistateModel)

Fit a multistate markov model to interval censored data (i.e. model.data.obstype .== 2 and all hazards are exponential with possibly piecewise homogeneous transition intensities).
"""
function fit_markov_interval(model::MultistateModel)
    
    # containers for bookkeeping TPMs
    books = build_tpm_mapping(model.data, model.tmat)

    # extract and initialize model parameters
    parameters = flatview(model.parameters)

    # optimize the likelihood
    optf = OptimizationFunction(loglik, Optimization.AutoForwardDiff())
    prob = OptimizationProblem(optf, parameters, PanelData(model, books))
    sol  = solve(prob, Newton())

    # get the variance-covariance matrix
    ll = pars -> MultistateModels.loglik(pars, MultistateModels.PanelData(model, books))
    vcov = inv(ForwardDiff.hessian(ll, sol.u))

    # wrap results
    # return (pars, vcov, -sol.minimum): 
end
