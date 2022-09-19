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
    fit_interval(model::MultistateModel)

Fit a multistate model given all observed paths are interval censored (i.e. model.data.obstype .== 2) AND all hazards are exponential
"""
function fit_interval(model::MultistateModel)

    # parse the data and hazards to generate the container of transition probability matrices
    # for now, we don't have time-inhomogeneous smoothly varying baseline cause-specific hazards
    # 

end
