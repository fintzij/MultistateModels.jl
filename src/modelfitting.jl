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
- maxiter: maximum number of MCEM iterations
- α: Standard normal quantile for asymptotic lower bound for ascent
- β: Standard normal quantile for inflation in # particles
- γ: Standard normal quantile for stopping
- κ: Inflation factor for MCEM sample size, m_new = m_cur + m_cur/κ
"""
function fit_semimarkov_interval(model::MultistateModel; nparticles = 10, maxiter = 100, tol = 1e-4, α = 0.1, β = 0.3, γ = 0.05, κ)

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
    weights            = ElasticArray{Float64}(undef, nsubj, nparticles)
    loglik_surrog      = ElasticArray{Float64}(undef, nsubj, nparticles)
    loglik_target_cur  = ElasticArray{Float64}(undef, nsubj, nparticles)
    loglik_target_prop = ElasticArray{Float64}(undef, nsubj, nparticles)

    # draw sample paths
    for i in 1:nsubj
        for j in 1:nparticles
            samplepaths[i,j] = draw_samplepath(i, model, tpm_book, hazmat_book, books[2])

            loglik_surrog[i,j] = loglik(model.markovsurrogate.parameters, samplepaths[i,j], model.markovsurrogate.hazards, model)

            loglik_target_cur[i,j] = loglik(model.parameters, samplepaths[i,j], model.hazards, model)

            weights[i,j] = exp(loglik_target_cur[i,j] - loglik_surrog[i,j])
        end
    end

    # normalizing constants for weights
    totweights = sum(weights, dims = 2)

    # get current estimate of the MCEM objective function
    obj_cur = mcem_objective(loglik_target_cur, weights, totweights)

    # extract and initialize model parameters
    params_cur = flatview(model.parameters)

    # initialize inference
    mll = [obj_cur, ]
    ests = ElasticArray{Float64}(undef, size(params_cur,1), 0)
    append!(ests, params_cur)

    # optimize the monte carlo marginal likelihood
    optf = OptimizationFunction(loglik, Optimization.AutoForwardDiff())
    prob = OptimizationProblem(optf, params_cur, SMPanelData(model, samplepaths, weights, totweights))
    params_prop = solve(prob, Newton())

    # recalculate the log likelihoods
    loglik!(params_prop, loglik_target_prop, SMPanelData(model, samplepaths, weights, totweights))

    # recalculate the objective function
    obj_prop = mcem_objective(loglik_target_prop, weights, totweights)
    
    # calculate the ASE for ΔQ
    ase = mcem_ase(loglik_target_prop .- loglik_target_cur, weights, totweights)

    # calculate the lower bound for ΔQ
    ascent_lb = obj_prop - obj_cur - quantile(Normal(), α)[1] * ase
    
    # cache results or increase MCEM effort
    if ascent_lb > 0

        # save traces

    else 
    end

    # pseudo-code
    # input: z_α, z_β, z_γ, nparticles, K (MC sample size inflation)
    # while keep_going
    #     1. maximize Qtil from eq 5
    #     2. Compute change in Qtil from eq 7
    #     3. compute ASE from eq 14
    #     4. 2 + 3 -> asymptotic lower bound, ALB, eq 12. 
    #         4a. If ALB > 0, accept proposed maximizer 
    #         4b. If ALB < 0, add Monte Carlo samples and return to step 1. 
    #     5. Check stopping rule, eq 13. (depends on z_γ)
    #     6. Check if additional samples are required in the next iteration, eq. 15 (depends on z_α and z_β)
    # end

end