"""
    fit(model::MultistateModel; alg = "ml")

Fit a model. 
""" 
function fit(model::MultistateModel; alg = "ml")
    
    # if sample paths are fully observed, maximize the likelihood directly
    if all(model.data.obstype .== 1)
        fitted = fit_exact(model)
    elseif all(model.data.obstype .== 2) & all(isa.(model.hazards, MultistateModels._Exponential))
        fitted = fit_interval(model)
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

    # identify unique covariate combinations and intervals
    ucovars,ugaps,mapping = collapse_interval_dat(model.data)
end
