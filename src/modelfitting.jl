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
    parameters = copy(model.parameters)

    # optimize
    prob = OptimizationProblem(MultistateModels.loglik, parameters, MultistateModels.ExactData(samplepaths, model))
    solve(prob, NelderMead())

    
    optimize(function(x) 
                set_parameters!(model, x)
                -loglik(samplepaths, model)
            end, model.parameters)

end
