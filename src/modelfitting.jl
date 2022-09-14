"""
    fit(model::MultistateModel; alg = "ml")

Fit a model. 
""" 
function fit(model::MultistateModel; alg = "ml")
    
    # if sample paths are fully observed, maximize the likelihood directly
    # Note: not doing obstype 0 for a while
    if all(model.data.obstype .== 1)
        fitted = fit_exact(model)
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
