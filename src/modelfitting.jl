"""
    fit(model::MultistateModel; alg = "ml")

Fit a model. 
""" 
function fit(model::MultistateModel; alg = "ml")
    
    # if sample paths are fully observed, maximize the likelihood directly
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

    # the anonymous functino in optimize() will set parameter values, then evaluate log likelihood of each path and sum up results using loglik(samplepaths::Array{SamplePath}, model::MultistateModel) 
    function sum_ll(param, samplepaths, model)
        set_parameters!(model, param)
        loglik(samplepaths, model)
    end

    optimize(x -> sum_ll(x, samplepaths, model), model.parameters)

    optimize(
        function(x) 
            set_parameters!(model, x)
            loglik(samplepaths, model)
        end, model.parameters)
end
