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
    function sum_ll(param; samplepaths = samplepaths, model = model)
        set_parameters!(model, param[eachindex(param)])
        -loglik(samplepaths, model)
    end

    sum_ll_ad = TwiceDifferentiable(sum_ll, model.parameters; autodiff = :forward)

    optimize(sum_ll_ad, model.parameters, Newton())

    sum_ll_ad = OnceDifferentiable(sum_ll, model.parameters; autodiff = :forward)

    optimize(sum_ll_ad, model.parameters, BFGS())

    
    optimize(function(x) 
                set_parameters!(model, x)
                -loglik(samplepaths, model)
            end, model.parameters)

end

function testfun(p)
    ((p[1] - 1.0))^2
end
td = TwiceDifferentiable(testfun, [1.0]; autodiff = :forward)
Optim.minimizer(optimize(td, [1.0], BFGS()))


function f(x)
    return (1.0 - x[1])^2
end
initial_x = zeros(1)
td = TwiceDifferentiable(f, initial_x; autodiff = :forward)
Optim.minimizer(optimize(td, initial_x, Newton()))
