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
    # I think everything works if we use this view of the parameters object
    parameters = flatview(model.parameters)

    # optimize
    optf = OptimizationFunction(MultistateModels.loglik, Optimization.AutoForwardDiff())
    prob = OptimizationProblem(optf, parameters, MultistateModels.ExactData(samplepaths, model))
    sol = solve(prob, Newton())

    # oh eff yes.
    ll = pars -> MultistateModels.loglik(pars, MultistateModels.ExactData(samplepaths, model))
    hess = inv(ForwardDiff.hessian(ll, sol.u))

    # wrap results
    
end



"""
    fit_markov_interval(model::MultistateModel)

Fit a multistate markov model to interval censored data (i.e. model.data.obstype .== 2 and all hazards are exponential but possibly time-inhomogeneous).
"""
function fit_markov_interval(model::MultistateModel)

    # the process is time-homogeneous 

    # identify unique covariates/gaps/intervals (index)
    # mapping identifies which TPM applies to which row in the data
    index,mapping = 
        build_tpm_containers(model.data; timehomogeneous=true)
end
