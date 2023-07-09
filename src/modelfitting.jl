# """
#     fit(model::MultistateModel; alg = "ml")

# Fit a model. 
# """ 
# function fit(model::MultistateModel; alg = "ml", nparticles = 100)
    
#     # if sample paths are fully observed, maximize the likelihood directly
#     if all(model.data.obstype .== 1)
        
#         fitted = fit_exact(model)

#     elseif all(model.data.obstype .== 2) & # Multistate Markov model, panel data, no censored state
#         all(isa.(model.hazards, MultistateModels._Exponential) .|| 
#             isa.(model.hazards, MultistateModels._ExponentialPH))
        
#         fitted = fit_markov_interval(model)

#     elseif all(model.data.obstype .== 2) & 
#         !all(isa.(model.hazards, MultistateModels._Exponential) .|| 
#              isa.(model.hazards, MultistateModels._ExponentialPH))
        
#         fitted = fit_semimarkov_interval(model; nparticles = nparticles)
#     end

#     ### mixed likelihoods to add
#     # 1. Mixed panel + fully observed data, no censored states, Markov process
#         # Easy.
#     # 2. Mixed panel + fully observed data, no censored states, semi-Markov process
#         # Easy, just need to append fully observed parts of the path in proposal.
#     # 3. Mixed panel + fully observed data, with censored states, semi-Markov process
#         # Easy, just sample the censored state. 
#     # 4. Mixed panel + fully observed data, with censored states, Markov process
#         # Medium complicated - marginalize over the possible states.

#     # return fitted object
#     return fitted
# end 

"""
    fit(model::MultistateModel)

Fit a multistate model given exactly observed sample paths.
"""
function fit(model::MultistateModel)

    # initialize array of sample paths
    samplepaths = extract_paths(model; self_transitions = false)

    # extract and initialize model parameters
    parameters = flatview(model.parameters)

    # optimize
    optf = OptimizationFunction(loglik, Optimization.AutoForwardDiff())
    prob = OptimizationProblem(optf, parameters, ExactData(model, samplepaths))    
    sol  = solve(prob, Newton())

    # oh eff yes.
    ll = pars -> loglik(pars, ExactData(model, samplepaths); neg=false)
    gradient = ForwardDiff.gradient(ll, sol.u)
    vcov = inv(.-ForwardDiff.hessian(ll, sol.u))

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
        model.weights,
        model.markovsurrogate,
        model.modelcall)
end


"""
    fit(model::MultistateMarkovModel)

Fit a multistate markov model to 
interval censored data (i.e. model.data.obstype .== 2 and all hazards are exponential with possibly piecewise homogeneous transition intensities),
or a mix of panel data and exact jump times.
"""
function fit(model::MultistateMarkovModel)
    
    # containers for bookkeeping TPMs
    books = build_tpm_mapping(model.data)

    # extract and initialize model parameters
    parameters = flatview(model.parameters)

    # optimize the likelihood
    optf = OptimizationFunction(loglik, Optimization.AutoForwardDiff())
    prob = OptimizationProblem(optf, parameters, MPanelData(model, books))
    sol  = solve(prob, Newton())

    # get the variance-covariance matrix
    ll = pars -> loglik(pars, MPanelData(model, books); neg=false)
    gradient = ForwardDiff.gradient(ll, sol.u)
    vcov = inv(.-ForwardDiff.hessian(ll, sol.u))

    # wrap results
    return  MultistateModelFitted(
        model.data,
        VectorOfVectors(sol.u, model.parameters.elem_ptr),
        -sol.minimum,
        vcov,
        model.hazards,
        model.totalhazards,
        model.tmat,
        model.hazkeys,
        model.subjectindices,
        model.weights,
        model.markovsurrogate,
        model.modelcall)
end

"""
    fit(model::MultistateMarkovModelCensored)

Fit a multistate markov model to 
interval censored data, some of which are censored,
or a mix of panel data, some of which are censored, and exact jump times.
"""
function fit(model::MultistateMarkovModelCensored)
    
    if all(model.data.obstype .!= 1) # only panel data
    # TODO
    # Equation 13 in msm package
    # https://cran.r-project.org/web/packages/msm/vignettes/msm-manual.pdf
    elseif any(model.data.obstype .== 1) # mix of panel data and exact jump times.
    # TODO
    # introduce a censored state before each observed jump time.
    # use Equation 13 from the msm package on each interval between the observed jump times
    end

end

# check with J&J that the previous two `fit` functions are correct before doing the same gymnastic with semi-Markov models
# do the same gymnastic for semi-Markov model

#  MCEM pseudo-code
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

"""
    fit_semimarkov_interval(model::MultistateSemiMarkovModel; nparticles)

Fit a semi-Markov model to panel data via Monte Carlo EM. 

Latent paths are sampled via MCMC and are subsampled at points t_k = x_1 + ... + x_k, where x_i - 1 ~ Pois(subrate * k ^ subscale). The arguments subrate and subscale default to 1 and 0.5, respectively.

# Arguments

- model: multistate model object
- nparticles: initial number of particles per participant for MCEM
- poolsize: multiple of nparticles for number of Markov surrogate paths to initialize
- maxiter: maximum number of MCEM iterations
- α: Standard normal quantile for asymptotic lower bound for ascent
- β: Standard normal quantile for inflation in # particles
- γ: Standard normal quantile for stopping
- κ: Inflation factor for MCEM sample size, m_new = m_cur + m_cur/κ
"""
function fit_semimarkov_interval(model::MultistateSemiMarkovModel; nparticles = 10, poolsize = 20, maxiter = 100, tol = 1e-4, α = 0.1, β = 0.3, γ = 0.05, κ = 3, verbose = false)

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
    for j in 1:nparticles
        for i in 1:nsubj
            samplepaths[i,j] = draw_samplepath(i, model, tpm_book, hazmat_book, books[2])

            loglik_surrog[i,j] = loglik(model.markovsurrogate.parameters, samplepaths[i,j], model.markovsurrogate.hazards, model)

            loglik_target_cur[i,j] = loglik(model.parameters, samplepaths[i,j], model.hazards, model)

            weights[i,j] = exp(loglik_target_cur[i,j] - loglik_surrog[i,j])
        end
    end

    # normalizing constants for weights
    totweights = sum(weights, dims = 2)

    # get current estimate of marginal log likelihood
    mll_cur = mcem_mll(loglik_target_cur, weights, totweights)

    # extract and initialize model parameters
    params_cur = flatview(model.parameters)

    # initialize inference
    mll = Vector{Float64}()
    ests = ElasticArray{Float64}(undef, size(params_cur,1), 0)

    # optimization function + problem
    optf = OptimizationFunction(loglik, Optimization.AutoForwardDiff())
    prob = OptimizationProblem(optf, params_cur, SMPanelData(model, samplepaths, weights, totweights))
  
    # go on then
    keep_going = true; iter = 0
    convergence = false
    while keep_going

        # println(iter)

        # optimize the monte carlo marginal likelihood
        params_prop = solve(remake(prob, u0 = Vector(params_cur), p = SMPanelData(model, samplepaths, weights, totweights)), Newton())

        # recalculate the log likelihoods
        loglik!(params_prop, loglik_target_prop, SMPanelData(model, samplepaths, weights, totweights))

        # recalculate the marginal log likelihood
        mll_prop = mcem_mll(loglik_target_prop, weights, totweights)

        # change in mll
        mll_change = mll_prop - mll_cur
        
        # calculate the ASE for ΔQ
        ase = mcem_ase(loglik_target_prop .- loglik_target_cur, weights, totweights)

         # calculate the lower bound for ΔQ
        ascent_lb = quantile(Normal(mll_change, ase), α)

        if verbose
            println("Iteration: $(iter+1)")
            println("Monte Carlo sample size: $nparticles")
            println("MCEM Asymptotic SE: $ase")
            println("Ascent lower bound: $ascent_lb")
        end

         # cache results or increase MCEM effort
        if ascent_lb > 0

            # increment the iteration
            iter += 1

            # check convergence
            convergence = quantile(Normal(mll_change, ase), 1-γ) < tol

            # set proposed values to current values
            params_cur = params_prop
            
            # swap current and proposed log likelihoods
            loglik_target_cur, loglik_target_prop = loglik_target_prop, loglik_target_cur

            # recalculate the importance weights
            weights = exp.(loglik_target_cur .- loglik_surrog)
            totweights = sum(weights; dims = 2)

            # swap current value of the marginal log likelihood
            mll_cur = mll_prop

            # save marginal log likelihood and parameters
            push!(mll, mll_cur)
            append!(ests, params_cur)

            # check whether to stop 
            if convergence || (iter > maxiter)

                keep_going = false
                
            else 
                # check whether to sample more
                nparticles = ceil(Int64, max(nparticles, ase^2 * sum(quantile(Normal(), [α, β]))^2 / mll_change^2))

                # draw from surrogate if required
                if nparticles > size(samplepaths, 2)

                    # draw sample paths
                    for j in (1 + size(samplepaths, 2)):nparticles
                        
                        # expand containers
                        append!(samplepaths, ElasticVector{SamplePath}(undef, nsubj))
                        append!(loglik_surrog, zeros(nsubj))
                        append!(loglik_target_prop, zeros(nsubj))
                        append!(loglik_target_cur, zeros(nsubj))
                        append!(weights, zeros(nsubj))

                        for i in 1:nsubj
                            samplepaths[i,j] = draw_samplepath(i, model, tpm_book, hazmat_book, books[2])

                            loglik_surrog[i,j] = loglik(model.markovsurrogate.parameters, samplepaths[i,j], model.markovsurrogate.hazards, model)

                            loglik_target_cur[i,j] = loglik(VectorOfVectors(params_cur, model.parameters.elem_ptr), samplepaths[i,j], model.hazards, model)

                            weights[i,j] = exp(loglik_target_cur[i,j] - loglik_surrog[i,j])
                        end
                    end

                    # normalizing constants for weights
                    totweights = sum(weights; dims = 2)

                    # recalculate the marginal log likelihood
                    mll_cur = mcem_mll(loglik_target_cur, weights, totweights)
                end
            end
        else 
            # increase the Monte Carlo sample size
            nparticles = ceil(Int64, nparticles + nparticles / κ)

            # draw sample paths
            for j in (1 + size(samplepaths, 2)):nparticles
                        
                # expand containers
                append!(samplepaths, ElasticVector{SamplePath}(undef, nsubj))
                append!(loglik_surrog, zeros(nsubj))
                append!(loglik_target_prop, zeros(nsubj))
                append!(loglik_target_cur, zeros(nsubj))
                append!(weights, zeros(nsubj))

                for i in 1:nsubj
                    samplepaths[i,j] = draw_samplepath(i, model, tpm_book, hazmat_book, books[2])

                    loglik_surrog[i,j] = loglik(model.markovsurrogate.parameters, samplepaths[i,j], model.markovsurrogate.hazards, model)

                    loglik_target_cur[i,j] = loglik(VectorOfVectors(params_cur, model.parameters.elem_ptr), samplepaths[i,j], model.hazards, model)

                    weights[i,j] = exp(loglik_target_cur[i,j] - loglik_surrog[i,j])
                end
            end

            # normalizing constants for weights
            totweights = sum(weights; dims = 2)

            # recalculate the marginal log likelihood
            mll_cur = mcem_mll(loglik_target_cur, weights, totweights)
        end
    end

    # initialize Fisher information matrix
    fisher = zeros(Float64, length(params_cur), length(params_cur), nsubj)
    
    # compute complete data gradients and hessians
    Threads.@threads for i in 1:nsubj

        path = Array{SamplePath}(undef, 1)
        diffres = DiffResults.HessianResult(params_cur)
        ll = pars -> loglik(pars, ExactData(model, path); neg=false)

        grads = Array{Float64}(undef, length(params_cur), nparticles)
        hesns = Array{Float64}(undef, length(params_cur), length(params_cur), nparticles)
        fisher_i1 = zeros(Float64, length(params_cur), length(params_cur))
        fisher_i2 = similar(fisher_i1)

        fill!(fisher_i1, 0.0)
        fill!(fisher_i2, 0.0)

        # calculate gradient and hessian for paths
        for j in 1:nparticles
            path[1] = samplepaths[i,j]
            diffres = ForwardDiff.hessian!(diffres, ll, params_cur)

            # grab hessian and gradient
            hesns[:,:,j] = DiffResults.hessian(diffres)
            grads[:,j] = DiffResults.gradient(diffres)
        end

        # accumulate
        for j in 1:nparticles
            fisher_i1 .+= weights[i,j] * (-hesns[:,:,j] - grads[:,j] * transpose(grads[:,j]))
        end
        fisher_i1 ./= totweights[i]

        for j in 1:nparticles
            for k in 1:nparticles
                fisher_i2 .+= weights[i,j] * weights[i,k] * grads[:,j] * transpose(grads[:,k])
            end
        end
        fisher_i2 ./= totweights[i]^2

        fisher[:,:,i] = fisher_i1 + fisher_i2
    end

    # get the variance-covariance matrix
    vcov = inv(reduce(+, fisher, dims = 3)[:,:,1])

    # return results
    
    
end