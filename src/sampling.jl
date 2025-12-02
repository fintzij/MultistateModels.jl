"""
    DrawSamplePaths(model; ...)

Draw additional sample paths until sufficient ESS or until the maximum number of paths is reached.

Supports both Markov and phase-type surrogate proposals. When phase-type infrastructure
is provided (phasetype_surrogate, tpm_book_ph, etc.), uses phase-type FFBS for sampling.
"""
function DrawSamplePaths!(model::MultistateProcess; ess_target, ess_cur, max_sampling_effort, samplepaths, loglik_surrog, loglik_target_prop, loglik_target_cur, _logImportanceWeights, ImportanceWeights,tpm_book_surrogate, hazmat_book_surrogate, books, npaths_additional, params_cur, surrogate, psis_pareto_k, fbmats, absorbingstates,
    # Phase-type proposal infrastructure (optional)
    phasetype_surrogate=nothing, tpm_book_ph=nothing, hazmat_book_ph=nothing, fbmats_ph=nothing, emat_ph=nothing)

    # Determine if using phase-type proposals
    use_phasetype = !isnothing(phasetype_surrogate)

    # make sure spline parameters are assigned correctly
    # nest the model parameters using VectorOfVectors (AD-compatible)
    pars = nest_params(params_cur, model.parameters)

    # update spline hazards with current parameters (no-op for functional splines)
    _update_spline_hazards!(model.hazards, pars)

    for i in eachindex(model.subjectindices)
        DrawSamplePaths!(i, model; 
            ess_target = ess_target,
            ess_cur = ess_cur, 
            max_sampling_effort = max_sampling_effort,
            samplepaths = samplepaths, 
            loglik_surrog = loglik_surrog, 
            loglik_target_prop = loglik_target_prop, 
            loglik_target_cur = loglik_target_cur, 
            _logImportanceWeights = _logImportanceWeights, 
            ImportanceWeights = ImportanceWeights, 
            tpm_book_surrogate = tpm_book_surrogate, 
            hazmat_book_surrogate = hazmat_book_surrogate, 
            books = books, 
            npaths_additional = npaths_additional, 
            params_cur = params_cur, 
            surrogate = surrogate, 
            psis_pareto_k = psis_pareto_k,
            fbmats = fbmats,
            absorbingstates = absorbingstates,
            # Phase-type infrastructure
            phasetype_surrogate = phasetype_surrogate,
            tpm_book_ph = tpm_book_ph,
            hazmat_book_ph = hazmat_book_ph,
            fbmats_ph = fbmats_ph,
            emat_ph = emat_ph)
    end
end

"""
    DrawSamplePaths(i, model; ...)

Draw additional sample paths for subject i until sufficient ESS or max paths reached.

Dispatches to either Markov or phase-type sampling based on whether phase-type
infrastructure is provided.
"""
function DrawSamplePaths!(i, model::MultistateProcess; ess_target, ess_cur, max_sampling_effort, samplepaths, loglik_surrog, loglik_target_prop, loglik_target_cur, _logImportanceWeights, ImportanceWeights, tpm_book_surrogate, hazmat_book_surrogate, books, npaths_additional, params_cur, surrogate, psis_pareto_k, fbmats, absorbingstates,
    # Phase-type proposal infrastructure (optional)
    phasetype_surrogate=nothing, tpm_book_ph=nothing, hazmat_book_ph=nothing, fbmats_ph=nothing, emat_ph=nothing)

    # Determine if using phase-type proposals
    use_phasetype = !isnothing(phasetype_surrogate)

    n_path_max = max_sampling_effort*ess_target

    # sample new paths if the current ess is less than the target
    keep_sampling = ess_cur[i] < ess_target

    # subject data
    subj_inds = model.subjectindices[i]
    subj_dat  = view(model.data, subj_inds, :)

    # compute fbmats here (for Markov FFBS, not phase-type)
    if !use_phasetype && any(subj_dat.obstype .∉ Ref([1,2]))
        # subject data
        subj_tpm_map = view(books[2], subj_inds, :)
        subj_emat    = view(model.emat, subj_inds, :)
        ForwardFiltering!(fbmats[i], subj_dat, tpm_book_surrogate, subj_tpm_map, subj_emat)
    end

    # sample
    while keep_sampling
        # make sure there are at least 50 paths in order to fit pareto
        npaths = length(samplepaths[i])
        n_add  = npaths == 0 ? maximum([50, Int(ceil(ess_target))]) : npaths_additional

        # augment the number of paths
        append!(samplepaths[i], Vector{SamplePath}(undef, n_add))
        append!(loglik_surrog[i], zeros(n_add))
        append!(loglik_target_prop[i], zeros(n_add))
        append!(loglik_target_cur[i], zeros(n_add))
        append!(_logImportanceWeights[i], zeros(n_add))
        append!(ImportanceWeights[i], zeros(n_add))

        # sample new paths and compute log likelihoods
        for j in npaths.+(1:n_add)
            if use_phasetype
                # Phase-type proposal: sample in expanded space, collapse to observed
                path_result = draw_samplepath_phasetype(i, model, tpm_book_ph, hazmat_book_ph, 
                                                         books[2], fbmats_ph, emat_ph, 
                                                         phasetype_surrogate, absorbingstates)
                
                # Store collapsed path for target likelihood evaluation
                samplepaths[i][j] = path_result.collapsed
                
                # Surrogate log-likelihood: unconditional density of expanded path under phase-type CTMC
                # This is h(Z|θ') in the importance weight formula: ν = f(Z|θ) / h(Z|θ')
                loglik_surrog[i][j] = loglik_phasetype_expanded(path_result.expanded, phasetype_surrogate)
            else
                # Markov proposal: standard sampling
                samplepaths[i][j] = draw_samplepath(i, model, tpm_book_surrogate, hazmat_book_surrogate, 
                                                    books[2], fbmats, absorbingstates)
                
                # Surrogate log-likelihood under Markov proposal
                # Use log-scale parameters for hazard evaluation
                surrogate_pars = get_log_scale_params(surrogate.parameters)
                loglik_surrog[i][j] = loglik(surrogate_pars, samplepaths[i][j], surrogate.hazards, model)
            end
            
            # target log-likelihood (same for both proposal types)
            # Use nest_params for AD-compatible parameter access (returns log-scale params)
            target_pars = nest_params(params_cur, model.parameters)
            loglik_target_cur[i][j] = loglik(target_pars, samplepaths[i][j], model.hazards, model) 
            
            # unnormalized log importance weight
            _logImportanceWeights[i][j] = loglik_target_cur[i][j] - loglik_surrog[i][j]
        end

        # no need to keep all paths
        if allequal(loglik_surrog[i])
            samplepaths[i]        = [first(samplepaths[i]),]
            loglik_target_cur[i]  = [first(loglik_target_cur[i]),]
            loglik_target_prop[i] = [first(loglik_target_prop[i]),]
            loglik_surrog[i]      = [first(loglik_surrog[i]),]
            ess_cur[i]            = ess_target
            ImportanceWeights[i]  = [1.0,]
            _logImportanceWeights[i] = [first(_logImportanceWeights[i]),]

        else
            # the case when the target and the surrogate are the same
            if all(iszero.(_logImportanceWeights[i]))
                fill!(ImportanceWeights[i], 1/length(ImportanceWeights[i]))
                ess_cur[i] = length(ImportanceWeights[i])
                psis_pareto_k[i] = 0.0
            else
                # might fail if not enough samples to fit pareto
                try
                    # pareto smoothed importance weights
                    psiw = ParetoSmooth.psis(reshape(copy(_logImportanceWeights[i]), 1, length(_logImportanceWeights[i]), 1); source = "other");
    
                    # save normalized importance weights and ess
                    copyto!(ImportanceWeights[i], psiw.weights)
                    ess_cur[i] = psiw.ess[1]
                    psis_pareto_k[i] = psiw.pareto_k[1]
    
                catch err
                    # exponentiate and normalize the unnormalized log weights
                    copyto!(ImportanceWeights[i], normalize(exp.(_logImportanceWeights[i]), 1))

                    # calculate the ess
                    ess_cur[i] = 1 / sum(ImportanceWeights[i] .^ 2)
                    psis_pareto_k[i] = 1.0
                end
            end
        end
        
        # check whether to stop
        if ess_cur[i] >= ess_target
            keep_sampling = false
        end
        
        if length(samplepaths[i]) > n_path_max
            keep_sampling = false
            @warn "More than $n_path_max sample paths are required to obtain ess>$ess_target for individual $i."
        end
    end
end

"""
    draw_paths(model::MultistateProcess; min_ess = 100, paretosmooth = true)

Draw sample paths conditional on the data. Require that the minimum effective sample size is greater than min_ess.

Arguments
- model: multistate model
- min_ess: minimum effective sample size, defaults to 100.
- paretosmooth: pareto smooth importance weights, defaults to true unless min_ess < 25. 
"""
function draw_paths(model::MultistateProcess; min_ess = 100, paretosmooth = true, return_logliks = false)

    # if exact data just return the loglik and subj_lml from the model fit
    if model isa MultistateModelFitted && all(model.data.obstype .== 1)
        return (loglik = model.loglik.loglik,
                subj_lml = model.loglik.subj_lml)
    end

    # number of subjects
    nsubj = length(model.subjectindices)

    # is the model markov?
    is_semimarkov = !all(isa.(model.hazards, _MarkovHazard))

    # get log-scale parameters as tuples for hazard evaluation
    params_target = get_log_scale_params(model.parameters)
    params_surrog = is_semimarkov ? get_log_scale_params(model.markovsurrogate.parameters) : params_target

    # get hazards
    hazards_target = model.hazards
    hazards_surrog = is_semimarkov ? model.markovsurrogate.hazards : model.hazards

    # containers for bookkeeping TPMs
    books = build_tpm_mapping(model.data)

    # build containers for transition intensity and prob mtcs for Markov surrogate
    hazmat_book_surrogate = build_hazmat_book(Float64, model.tmat, books[1])
    tpm_book_surrogate = build_tpm_book(Float64, model.tmat, books[1])

    # allocate memory for matrix exponential
    cache = ExponentialUtilities.alloc_mem(similar(hazmat_book_surrogate[1]), ExpMethodGeneric())

    # Solve Kolmogorov equations for TPMs
    for t in eachindex(books[1])
        # compute the transition intensity matrix
        compute_hazmat!(hazmat_book_surrogate[t], params_surrog, hazards_surrog, books[1][t], model.data)
        # compute transition probability matrices
        compute_tmat!(tpm_book_surrogate[t], hazmat_book_surrogate[t], books[1][t], cache)
    end

    # set up objects for simulation
    samplepaths     = [sizehint!(Vector{SamplePath}(), ceil(Int64, 4 * min_ess)) for i in 1:nsubj]
    loglik_target   = [sizehint!(Vector{Float64}(), ceil(Int64, 4 * min_ess)) for i in 1:nsubj]
    
    loglik_surrog = [sizehint!(Vector{Float64}(), ceil(Int64, 4 * min_ess)) for i in 1:nsubj]
    ImportanceWeights = [sizehint!(Vector{Float64}(), ceil(Int64, 4 * min_ess)) for i in 1:nsubj]

    # continers
    subj_ll                   = Vector{Float64}(undef, nsubj)
    subj_ess                  = Vector{Float64}(undef, nsubj)
    subj_pareto_k             = zeros(nsubj)
    
    # make fbmats if necessary
    fbmats = build_fbmats(model)
    
    # identify absorbing states
    absorbingstates = findall(map(x -> all(x .== 0), eachrow(model.tmat)))

    # compute the normalizing constant of the proposal density
    # subj_normalizing_constant = loglik(parameters, data::MPanelData; neg = true, return_ll_subj = true)

    for i in eachindex(model.subjectindices) 

        keep_sampling = true

        # subject data
        subj_inds = model.subjectindices[i]
        subj_dat  = view(model.data, subj_inds, :)

        # compute fbmats here
        if any(subj_dat.obstype .∉ Ref([1,2]))
            # subject data
            subj_tpm_map = view(books[2], subj_inds, :)
            subj_emat    = view(model.emat, subj_inds, :)
            ForwardFiltering!(fbmats[i], subj_dat, tpm_book_surrogate, subj_tpm_map, subj_emat)
        end

        # sampling
        while keep_sampling

            # make sure there are at least 25 paths in order to fit pareto
            npaths = length(samplepaths[i])
            n_add  = npaths == 0 ? min_ess : ceil(Int64, npaths * 1.4)
    
            # augment the number of paths
            append!(samplepaths[i], Vector{SamplePath}(undef, n_add))
            append!(loglik_target[i], zeros(n_add))
            append!(loglik_surrog[i], zeros(n_add))
            append!(ImportanceWeights[i], zeros(n_add))
    
            # sample new paths and compute log likelihoods
            for j in npaths.+(1:n_add)
                # draw path
                samplepaths[i][j] = draw_samplepath(i, model, tpm_book_surrogate, hazmat_book_surrogate, books[2], fbmats, absorbingstates)

                # compute log likelihood
                loglik_target[i][j] = loglik(params_target, samplepaths[i][j], hazards_target, model)

                # log likelihood of the surrogate
                if is_semimarkov
                    loglik_surrog[i][j] = loglik(params_surrog, samplepaths[i][j], hazards_surrog, model) 
                else
                    loglik_surrog[i][j] = loglik_target[i][j]
                end

                # compute the unsmoothed importance weight
                ImportanceWeights[i][j] = exp(loglik_target[i][j] - loglik_surrog[i][j])
            end
    
            # no need to keep all paths if redundant
            if allequal(loglik_surrog[i])
                samplepaths[i]       = [first(samplepaths[i]),]
                loglik_target[i]     = [first(loglik_target[i]),]
                loglik_surrog[i]     = [first(loglik_surrog[i]),]
                ImportanceWeights[i] = [1.0,]
                subj_ess[i]          = min_ess

            else
                if !is_semimarkov
                    subj_ess[i] = length(samplepaths[i])
                else
                    # raw log importance weights
                    logweights = reshape(copy(loglik_target[i] - loglik_surrog[i]), 1, length(loglik_target[i]), 1) 

                     # might fail if not enough samples to fit pareto, e.g. a single sample if only one path is possible.
                    if any(logweights .!= 0.0)
                        if paretosmooth 
                            try
                                # pareto smoothed importance weights
                                psiw = psis(logweights; source = "other");
                
                                # save importance weights and ess
                                copyto!(ImportanceWeights[i], psiw.weights)
                                subj_ess[i] = psiw.ess[1]
                                subj_pareto_k[i] = psiw.pareto_k[1]

                            catch err
                                copyto!(ImportanceWeights[i], normalize(exp.(logweights), 1))
                                subj_ess[i] = ParetoSmooth.relative_eff(logweights; source = "other")[1] * length(loglik_target[i])
                            end
                        else
                            subj_ess[i] = ParetoSmooth.relative_eff(logweights; source = "other")[1] * length(loglik_target[i])
                        end
                    else
                        subj_ess[i] = length(samplepaths[i])
                    end
                end
            end
            
            # check whether to stop
            if subj_ess[i] >= min_ess
                keep_sampling = false
            end
        end
    end

    # normalize importance weights
    # normalize!.(ImportanceWeights, 1)
    ImportanceWeightsNormalized = normalize.(ImportanceWeights, 1)

    if return_logliks
        return (; samplepaths, loglik_target, subj_ess, loglik_surrog, ImportanceWeightsNormalized, ImportanceWeights)
    else
        return (; samplepaths, ImportanceWeightsNormalized)
    end
end

"""
    draw_paths(model::MultistateProcess, npaths)

Draw sample paths conditional on the data. Require that the minimum effective sample size is greater than min_ess.

Arguments
- model: multistate model.
- npaths: number of paths to sample.
- paretosmooth: pareto smooth importance weights, defaults to true. 
"""
function draw_paths(model::MultistateProcess, npaths; paretosmooth = true, return_logliks = false)

    # if exact data just return the loglik and subj_lml from the model fit
    if model isa MultistateModelFitted && all(model.data.obstype .== 1)
        return (loglik = model.loglik.loglik,
                subj_lml = model.loglik.subj_lml)
    end

    # number of subjects
    nsubj = length(model.subjectindices)

    # is the model markov?
    is_semimarkov = !all(isa.(model.hazards, _MarkovHazard))

    # Build surrogate if needed for semi-Markov models
    markovsurrogate = model.markovsurrogate
    if is_semimarkov && isnothing(markovsurrogate)
        # Fit a Markov surrogate model
        surrogate_fitted = fit_surrogate(model; verbose = false)
        markovsurrogate = MarkovSurrogate(surrogate_fitted.hazards, surrogate_fitted.parameters)
    end

    # get log-scale parameters as tuples for hazard evaluation
    params_target = get_log_scale_params(model.parameters)
    params_surrog = is_semimarkov ? get_log_scale_params(markovsurrogate.parameters) : params_target

    # get hazards
    hazards_target = model.hazards
    hazards_surrog = is_semimarkov ? markovsurrogate.hazards : model.hazards

    # containers for bookkeeping TPMs
    books = build_tpm_mapping(model.data)

    # build containers for transition intensity and prob mtcs
    hazmat_book = build_hazmat_book(Float64, model.tmat, books[1])
    tpm_book = build_tpm_book(Float64, model.tmat, books[1])

    # allocate memory for matrix exponential
    cache = ExponentialUtilities.alloc_mem(similar(hazmat_book[1]), ExpMethodGeneric())

    # Solve Kolmogorov equations for TPMs
    for t in eachindex(books[1])

        # compute the transition intensity matrix
        compute_hazmat!(
            hazmat_book[t],
            params_surrog,
            hazards_surrog,
            books[1][t],
            model.data)

        # compute transition probability matrices
        compute_tmat!(
            tpm_book[t],
            hazmat_book[t],
            books[1][t],
            cache)
    end

    # set up objects for simulation
    samplepaths     = [Vector{SamplePath}(undef, npaths) for i in 1:nsubj]
    loglik_target   = [Vector{Float64}(undef, npaths) for i in 1:nsubj]
    
    loglik_surrog = [Vector{Float64}(undef, npaths) for i in 1:nsubj]
    ImportanceWeights = [Vector{Float64}(undef, npaths) for i in 1:nsubj]

    # for ess 
    subj_ll   = Vector{Float64}(undef, nsubj)
    subj_ess  = Vector{Float64}(undef, nsubj)
    
    # make fbmats if necessary
    fbmats = build_fbmats(model)
    
    # identify absorbing states
    absorbingstates = findall(map(x -> all(x .== 0), eachrow(model.tmat)))

    for i in eachindex(model.subjectindices) 

        keep_sampling = true

        # subject data
        subj_inds = model.subjectindices[i]
        subj_dat  = view(model.data, subj_inds, :)

        # compute fbmats here
        if any(subj_dat.obstype .∉ Ref([1,2]))
            # subject data
            subj_tpm_map = view(books[2], subj_inds, :)
            subj_emat    = view(model.emat, subj_inds, :)
            ForwardFiltering!(fbmats[i], subj_dat, tpm_book, subj_tpm_map, subj_emat)
        end

        # sample new paths and compute log likelihoods
        for j in 1:npaths
            # draw path
            samplepaths[i][j] = draw_samplepath(i, model, tpm_book, hazmat_book, books[2], fbmats, absorbingstates)

            # log likelihood of the target
            loglik_target[i][j] = loglik(params_target, samplepaths[i][j], hazards_target, model)

            # log likelihood of the surrogate
            if is_semimarkov
                loglik_surrog[i][j] = loglik(params_surrog, samplepaths[i][j], hazards_surrog, model) 
            else
                loglik_surrog[i][j] = loglik_target[i][j]
            end

            # compute the unsmoothed importance weight
            ImportanceWeights[i][j] = exp(loglik_target[i][j] - loglik_surrog[i][j])
        end

        # no need to keep all paths if redundant
        if allequal(loglik_surrog[i])
            samplepaths[i]       = [first(samplepaths[i]),]
            loglik_target[i]     = [first(loglik_target[i]),]
            loglik_surrog[i]     = [first(loglik_surrog[i]),]
            ImportanceWeights[i] = [1.0,]
            subj_ess[i]          = npaths

        else
            if !is_semimarkov
                subj_ess[i] = length(samplepaths[i])
            else
                # raw log importance weights
                logweights = reshape(copy(log.(ImportanceWeights[i])), 1, length(loglik_target[i]), 1) 

                # might fail if not enough samples to fit pareto
                if any(logweights .!= 0.0)
                    if paretosmooth 
                        try
                            # pareto smoothed importance weights
                            psiw = psis(logweights; source = "other");
            
                            # save importance weights and ess
                            copyto!(ImportanceWeights[i], psiw.weights)
                            subj_ess[i] = psiw.ess[1]            

                        catch err
                            subj_ess[i] = ParetoSmooth.relative_eff(logweights; source = "other")[1] * length(loglik_target[i])
                        end
                    else
                        subj_ess[i] = ParetoSmooth.relative_eff(logweights; source = "other")[1] * length(loglik_target[i])
                    end
                else
                    subj_ess[i] = length(samplepaths[i])
                end
            end
        end
    end

    # normalize importance weights
    ImportanceWeightsNormalized = normalize.(ImportanceWeights, 1)

    if return_logliks
        return (; samplepaths, loglik_target, subj_ess, loglik_surrog, ImportanceWeightsNormalized, ImportanceWeights)
    else
        return (; samplepaths, ImportanceWeightsNormalized)
    end
end

"""
   sample_ecctmc(P, Q, a, b, t0, t1)

Sample path for an endpoint conditioned CTMC whose states at times `t0` and `t1` are `a` and `b`. `P` is the transition probability matrix over the interval, `Q` is the transition intensity matrix. 
"""
function sample_ecctmc(P, Q, a, b, t0, t1)

    # number of states
    nstates = size(Q, 1)

    # length of time interval
    T = t1 - t0

    # maximum total hazard
    m = maximum(abs.(diag(Q)))

    # extract the a,b element of P
    p_ab = P[a,b]

    # generate the auxilliary tpm - optimize this later
    R = ElasticArray{Float64}(undef, nstates, nstates, 1)
    R[:,:,1] = diagm(ones(Float64, nstates)) + Q / m

    # sample threshold for determining number of states
    nthresh = rand(1)[1]
    
    # initialize number of jumps and contitional prob of jumps
    njumps = 0
    cprob = exp(-m*T) * (a==b) / p_ab # cprob of one jump

    if cprob > nthresh 
        return # no jumps, the full path is `a` in [t0,t1]
    else
        # increment the number of jumps and compute cprob
        njumps += 1
        cprob  += exp(-m*T) * (m*T) * R[a,b,1] / p_ab

        # if there is exactly one jump
        if cprob > nthresh
            if a == b
                return # jump is virtual, path is `a` in [t0,t1]
            else
                # jump is real
                times  = rand(Uniform(t0, t1), 1)
                states = [b,]

                # return times and states
                return times, states
            end
        else
            # calculate the number of jumps
            while cprob < nthresh
                # increment the number of jumps
                njumps += 1

                # append the new power of R to the array
                append!(R, R[:,:,1]^njumps)

                # calculate cprob
                cprob += exp(-m*T) * (m*T)^njumps / factorial(big(njumps)) * R[a,b,njumps] / p_ab
            end

            # transition times are uniformly distributed in [t0,t1]
            times = sort!(rand(Uniform(t0, t1), njumps))

            # sample the states at the transition times
            scur   = a
            states = zeros(Int64, njumps)

            for s in 1:(njumps-1)
                snext = sample(1:nstates, Weights(R[scur, :, 1] .* R[:, b, njumps-s] ./ R[scur, b, njumps-s+1]))
                if snext != scur
                    scur = snext
                    states[s] = scur
                end
            end

            states[end] = scur != b ? b : 0

            # determine which transitions are virtual transitions
            jumpinds = findall(states .!= 0)
            
            # return state sequence and times
            return times[jumpinds], states[jumpinds]
        end
    end
end

"""
    sample_ecctmc!(jumptimes, stateseq, P, Q, a, b, t0, t1)

Sample path for an endpoint conditioned CTMC whose states at times `t0` and `t1` are `a` and `b`. `P` is the transition probability matrix over the interval, `Q` is the transition intensity matrix. Jump times and state sequence get appended to `jumptimes` and `stateseq`.
"""
function sample_ecctmc!(jumptimes, stateseq, P, Q, a, b, t0, t1)

    # number of states
    nstates = size(Q, 1)

    # length of time interval
    T = t1 - t0

    # maximum total hazard
    m = maximum(abs.(diag(Q)))

    # extract the a,b element of P
    p_ab = P[a,b]

    # generate the auxilliary tpm - optimize this later
    R = ElasticArray{Float64}(undef, nstates, nstates, 1)
    R[:,:,1] = diagm(ones(Float64, nstates)) + Q / m

    # sample threshold for determining number of states
    nthresh = rand(1)[1]
    
    # initialize number of jumps and contitional prob of jumps
    njumps = 0
    cprob = exp(-m*T) * (a==b) / p_ab # cprob of one jump

    if cprob > nthresh 
        return # no jumps, the full path is `a` in [t0,t1]
    else
        # increment the number of jumps and compute cprob
        njumps += 1
        cprob  += exp(-m*T) * (m*T) * R[a,b,1] / p_ab

        # if there is exactly one jump
        if cprob > nthresh
            if a == b
                return # jump is virtual, path is `a` in [t0,t1]
            else
                # jump is real
                times  = rand(Uniform(t0, t1), 1)
                states = [b,]

                # append times and states
                append!(jumptimes, times)
                append!(stateseq, states)

                return 
            end
        else
            # calculate the number of jumps
            while cprob < nthresh
                # increment the number of jumps
                njumps += 1

                # append the new power of R to the array
                append!(R, R[:,:,1]^njumps)

                # calculate cprob
                cprob += exp(-m*T) * (m*T)^njumps / factorial(big(njumps)) * R[a,b,njumps] / p_ab
            end

            # transition times are uniformly distributed in [t0,t1]
            times = sort!(rand(Uniform(t0, t1), njumps))

            # sample the states at the transition times
            scur   = a
            states = zeros(Int64, njumps)

            for s in 1:(njumps-1)
                snext = sample(1:nstates, Weights(R[scur, :, 1] .* R[:, b, njumps-s] ./ R[scur, b, njumps-s+1]))
                if snext != scur
                    scur = snext
                    states[s] = scur
                end
            end

            states[end] = scur != b ? b : 0

            # determine which transitions are virtual transitions
            jumpinds = findall(states .!= 0)
            
            # return state sequence and times
            append!(jumptimes, times[jumpinds])
            append!(stateseq, states[jumpinds])

            return
        end
    end
end

"""
    draw_samplepath(subj::Int64, model::MultistateProcess, tpm_book, hazmat_book, tpm_map)

Draw sample paths from a Markov surrogate process conditional on panel data.
"""
function draw_samplepath(subj::Int64, model::MultistateProcess, tpm_book, hazmat_book, tpm_map, fbmats, absorbingstates)

    # subject data
    subj_inds = model.subjectindices[subj] # rows in the dataset corresponding to the subject
    subj_dat     = view(model.data, subj_inds, :) # subject's data - no shallow copy, just pointer
    subj_tpm_map = view(tpm_map, subj_inds, :)

    # sample any censored observation
    if any(subj_dat.obstype .∉ Ref([1,2]))
        BackwardSampling!(subj_dat, fbmats[subj])
    end

    # initialize sample path
    times  = [subj_dat.tstart[1]]; sizehint!(times, size(model.tmat, 2) * 2)
    states = [subj_dat.statefrom[1]]; sizehint!(states, size(model.tmat, 2) * 2)

    # loop through data and sample endpoint conditioned paths - need to give control flow some thought
    for i in eachindex(subj_inds) # loop over each interval for the subject
        if subj_dat.obstype[i] == 1 
            push!(times, subj_dat.tstop[i])
            push!(states, subj_dat.stateto[i])
        else
            sample_ecctmc!(times, states, tpm_book[subj_tpm_map[i,1]][subj_tpm_map[i,2]], hazmat_book[subj_tpm_map[i,1]], subj_dat.statefrom[i], subj_dat.stateto[i], subj_dat.tstart[i], subj_dat.tstop[i])
        end
    end

    # append last state and time
    if subj_dat.obstype[end] != 1
        push!(times, subj_dat.tstop[end])
        push!(states, subj_dat.stateto[end])
    end

    # truncate at entry to absorbing states
    truncind = findfirst(states .∈ Ref(absorbingstates))
    if !isnothing(truncind)
        times = first(times, truncind)
        states = first(states, truncind)
    end

    # return path
    return reduce_jumpchain(SamplePath(subj, times, states))
end

"""
    ForwardFiltering!(subj_fbmats, subj_dat, tpm_book, subj_tpm_map, subj_emat; init_state=nothing)

Computes the forward recursion matrices for the FFBS algorithm. Writes into subj_fbmats.

# Arguments
- `subj_fbmats`: Pre-allocated forward-backward matrices
- `subj_dat`: Subject's data view
- `tpm_book`: TPM book
- `subj_tpm_map`: Subject's TPM mapping  
- `subj_emat`: Subject's emission matrix
- `init_state`: Optional initial state index. If nothing, uses subj_dat.statefrom[1].
                For phase-type, pass the first phase of the initial observed state.
"""
function ForwardFiltering!(subj_fbmats, subj_dat, tpm_book, subj_tpm_map, subj_emat; init_state=nothing)

    n_times  = size(subj_fbmats, 1)
    n_states = size(subj_fbmats, 2)

    # initialize
    p0 = zeros(Float64, n_states)
    initial = isnothing(init_state) ? subj_dat.statefrom[1] : init_state
    p0[initial] = 1.0
    subj_fbmats[1, :, :] = p0 * subj_emat[1,:]'

    # recurse
    if n_times > 1
        for s in 2:n_times
            subj_fbmats[s, 1:n_states, 1:n_states] = (sum(subj_fbmats[s-1,:,:], dims = 1)' * subj_emat[s,:]') .* tpm_book[subj_tpm_map[s,1]][subj_tpm_map[s,2]]
            normalize!(subj_fbmats[s,:,:], 1)
        end
    end
end


"""
    BackwardSampling!(subj_dat, subj_fbmats)

Samples a path and writes it in to subj_dat.
"""
function BackwardSampling!(subj_dat, subj_fbmats)

    # initialize
    n_times  = size(subj_fbmats, 1)
    n_states = size(subj_fbmats, 2)

    p = normalize(sum(subj_fbmats[n_times,:,:], dims=1), 1) #  dims=1 or  dims=2 ?

    subj_dat.stateto[end] = rand(Categorical(vec(p)))

    # recurse
    if n_times > 1
        for t in (n_times - 1):-1:1
            subj_dat.stateto[t] = rand(Categorical(normalize(subj_fbmats[t+1, :, subj_dat.stateto[t + 1]], 1)))
        end
        subj_dat.statefrom[Not(1)] .= subj_dat.stateto[Not(end)]
    end
end


function BackwardSampling(m, p) 
    
    n_obs = size(p, 1) # number of observations
    h = Array{Int64}(undef, n_obs)

    # 1. draw draw h_n ~ pi_n
    h[n_obs] = rand(Categorical(m[n_obs+1,:]))

    # 2. draw h_t|h_{t+1}=s ~ p_{t,.,s}
    for t in (n_obs-1):-1:1
        w = p[t+1,:,h[t+1]] / sum(p[t+1,:,h[t+1]])
       h[t] = rand(Categorical(w)) # [Eq. 10]
    end

    return h

end


# =============================================================================
# Viterbi MAP functions for MCEM warm start initialization
# =============================================================================
#
# These functions compute the marginal posterior mode (Viterbi MAP) at each
# observation time, selecting the most probable state given all observed data.
# This is used to initialize MCEM with a single high-quality path per subject
# that is close to the mode of the posterior distribution.
#
# Key distinction from Viterbi sequence:
# - Viterbi sequence: joint mode argmax P(h_1:T | y_1:T)
# - Viterbi MAP: marginal mode [argmax P(h_1 | y_1:T), ..., argmax P(h_T | y_1:T)]
#
# The marginal mode is computed using the forward-backward algorithm:
# P(h_t | y_1:T) ∝ α_t(h_t) * β_t(h_t)
# where α_t are forward probabilities and β_t are backward probabilities.
#
# The marginal mode provides a good initialization point for MCEM.
# =============================================================================

"""
    BackwardMAP!(subj_dat, subj_fbmats)

Compute the Viterbi MAP (marginal posterior mode) path for a subject.

Instead of sampling from the posterior distribution of hidden states, this function
takes the argmax of the marginal posterior at each observation time. This provides
a deterministic path that lies near the mode of the posterior distribution.

The marginal posterior is computed from the forward-backward matrices:
```math
P(h_t = s | y_{1:T}) \\propto \\sum_{s'} \\text{fbmats}[t, s, s']
```

where fbmats contains the filtered probabilities from forward-backward recursion.

# Arguments
- `subj_dat`: Subject data (modified in-place)
- `subj_fbmats`: Forward-backward matrices from `ForwardFiltering!`

# Notes
This function modifies `subj_dat.statefrom` and `subj_dat.stateto` in-place,
setting them to the argmax of the marginal posterior at each time point.

See also: [`BackwardSampling!`](@ref), [`viterbi_map_path`](@ref)
"""
function BackwardMAP!(subj_dat, subj_fbmats)

    # initialize
    n_times  = size(subj_fbmats, 1)
    n_states = size(subj_fbmats, 2)

    # marginal probability at final time: sum over the "from" state
    p_final = vec(sum(subj_fbmats[n_times,:,:], dims=1))
    
    # take argmax instead of sampling
    subj_dat.stateto[end] = argmax(p_final)

    # recurse backward, taking argmax at each step
    if n_times > 1
        for t in (n_times - 1):-1:1
            # marginal posterior for state at time t given state at t+1
            p_t = subj_fbmats[t+1, :, subj_dat.stateto[t + 1]]
            subj_dat.stateto[t] = argmax(p_t)
        end
        subj_dat.statefrom[Not(1)] .= subj_dat.stateto[Not(end)]
    end
end

"""
    viterbi_map_path(subj::Int64, model::MultistateProcess, tpm_book, hazmat_book, tpm_map, fbmats, absorbingstates)

Compute the Viterbi MAP (marginal posterior mode) path for a subject.

This function is analogous to `draw_samplepath` but instead of sampling from the
posterior distribution, it returns the deterministic path obtained by taking
the argmax of the marginal posterior at each observation time.

# Arguments
- `subj::Int64`: Subject index
- `model::MultistateProcess`: Multistate model
- `tpm_book`: Pre-computed transition probability matrices
- `hazmat_book`: Pre-computed hazard matrices
- `tpm_map`: Mapping to TPM book indices
- `fbmats`: Forward-backward matrices container
- `absorbingstates`: Indices of absorbing states

# Returns
- `SamplePath`: The MAP path for this subject

# Usage in MCEM

This function is used for MCEM warm start initialization. By starting with
a single high-quality path per subject (the marginal MAP), the first M-step
can take a large step toward the mode, accelerating convergence.

See also: [`draw_samplepath`](@ref), [`BackwardMAP!`](@ref)
"""
function viterbi_map_path(subj::Int64, model::MultistateProcess, tpm_book, hazmat_book, tpm_map, fbmats, absorbingstates)

    # subject data
    subj_inds = model.subjectindices[subj]
    subj_dat = view(model.data, subj_inds, :)
    subj_tpm_map = view(tpm_map, subj_inds, :)

    # for observations with censoring, compute forward probabilities and backward MAP
    if any(subj_dat.obstype .∉ Ref([1,2]))
        # compute forward matrices
        subj_emat = view(model.emat, subj_inds, :)
        ForwardFiltering!(fbmats[subj], subj_dat, tpm_book, subj_tpm_map, subj_emat)
        
        # take MAP instead of sampling
        BackwardMAP!(subj_dat, fbmats[subj])
    end

    # initialize sample path
    times  = [subj_dat.tstart[1]]; sizehint!(times, size(model.tmat, 2) * 2)
    states = [subj_dat.statefrom[1]]; sizehint!(states, size(model.tmat, 2) * 2)

    # loop through data and construct path
    # For exact observations, use the observed states
    # For panel/censored observations, use the MAP states from backward pass
    for i in eachindex(subj_inds)
        if subj_dat.obstype[i] == 1 
            # exact observation - use observed states
            push!(times, subj_dat.tstop[i])
            push!(states, subj_dat.stateto[i])
        else
            # panel observation - for MAP path, we don't sample intermediate states
            # just use the MAP endpoints. Use draw_map_ecctmc for deterministic interior.
            map_path = draw_map_ecctmc(
                tpm_book[subj_tpm_map[i,1]][subj_tpm_map[i,2]], 
                hazmat_book[subj_tpm_map[i,1]], 
                subj_dat.statefrom[i], 
                subj_dat.stateto[i], 
                subj_dat.tstart[i], 
                subj_dat.tstop[i]
            )
            # append intermediate times and states (excluding start which is already in path)
            if length(map_path.times) > 1
                append!(times, map_path.times[2:end])
                append!(states, map_path.states[2:end])
            end
        end
    end

    # append last state and time if not exact observation
    if subj_dat.obstype[end] != 1
        push!(times, subj_dat.tstop[end])
        push!(states, subj_dat.stateto[end])
    end

    # truncate at entry to absorbing states
    truncind = findfirst(states .∈ Ref(absorbingstates))
    if !isnothing(truncind)
        times = first(times, truncind)
        states = first(states, truncind)
    end

    # return path
    return reduce_jumpchain(SamplePath(subj, times, states))
end

"""
    draw_map_ecctmc(P, Q, a, b, t0, t1)

Compute the MAP (most probable) path for an endpoint-conditioned CTMC.

This is a deterministic analog of `sample_ecctmc` that returns the most probable
interior path rather than sampling from the distribution of paths.

For a single direct transition from state `a` to state `b`, the conditional density 
of the transition time `t` given the endpoints is:

```math
f(t | X(t_0)=a, X(t_1)=b) \\propto q_{ab} \\cdot S_a(t-t_0) \\cdot P_{bb}(t_1-t)
```

where `S_a(τ) = exp(-q_a τ)` is survival in state `a`, `q_a = -Q[a,a]` is the total 
exit rate from state `a`, and `P_{bb}(τ)` is the probability of being in state `b` 
after time `τ` starting from `b`.

The MAP transition time is found by maximizing this density using Brent's method.

# Arguments
- `P`: Transition probability matrix over the interval [t0, t1]
- `Q`: Transition intensity matrix
- `a`: Initial state
- `b`: Final state
- `t0`: Start time
- `t1`: End time

# Returns
- `NamedTuple{(:times, :states)}`: Times and states of the MAP path

# References
- Morsomme et al. (2025) Biostatistics, kxaf038

See also: [`sample_ecctmc`](@ref), [`viterbi_map_path`](@ref)
"""
function draw_map_ecctmc(P, Q, a, b, t0, t1)
    if a == b
        # No transition needed - stay in state a for entire interval
        return (times = [t0], states = [a])
    end
    
    # For a single direct transition a → b, find the MAP transition time
    # The conditional density is: f(t) ∝ q_ab * exp(-q_a * (t-t0)) * P_bb(t1-t)
    # where q_a = -Q[a,a], q_ab = Q[a,b]
    
    T = t1 - t0
    q_a = -Q[a, a]      # Total exit rate from state a
    q_ab = Q[a, b]      # Rate of transition a → b
    
    # Handle edge cases
    if T ≤ 0 || q_ab ≤ 0
        # Degenerate case: return midpoint
        return (times = [t0, (t0 + t1) / 2], states = [a, b])
    end
    
    # Compute P_bb(τ) = exp(Q * τ)[b,b] for remaining time τ after transition
    # For efficiency, we use the matrix exponential cache structure
    # The objective to maximize is: log(q_ab) - q_a*(t-t0) + log(P_bb(t1-t))
    # We minimize the negative of this
    
    # For small state spaces, compute matrix exponential at each evaluation
    nstates = size(Q, 1)
    
    # Objective function (negative log density, for minimization)
    function neg_log_density(t_rel)
        # t_rel is time since t0
        if t_rel ≤ 0 || t_rel ≥ T
            return Inf
        end
        
        τ_remaining = T - t_rel
        
        # Survival in state a: exp(-q_a * t_rel)
        log_surv_a = -q_a * t_rel
        
        # P_bb(τ_remaining) via matrix exponential
        if τ_remaining > 0
            P_remaining = exp(Q * τ_remaining)
            p_bb = P_remaining[b, b]
            if p_bb ≤ 0
                return Inf
            end
            log_p_bb = log(p_bb)
        else
            log_p_bb = 0.0  # P_bb(0) = 1
        end
        
        # Negative log density (for minimization)
        return -(log(q_ab) + log_surv_a + log_p_bb)
    end
    
    # Find MAP time using Brent's method on the interior of [0, T]
    # Use small epsilon to stay in interior
    ε = min(T * 1e-6, 1e-10)
    
    try
        result = Optim.optimize(neg_log_density, ε, T - ε, Optim.Brent())
        t_map_rel = Optim.minimizer(result)
        t_map = t0 + t_map_rel
        
        # Ensure t_map is strictly between t0 and t1
        t_map = clamp(t_map, t0 + ε, t1 - ε)
        
        return (times = [t0, t_map], states = [a, b])
    catch
        # Fallback to midpoint if optimization fails
        return (times = [t0, (t0 + t1) / 2], states = [a, b])
    end
end


"""
    ComputeImportanceWeights!(loglik_target, loglik_surrog, _logImportanceWeights, ImportanceWeights, ess_target)

Compute the importance weights and ess.
"""
function ComputeImportanceWeightsESS!(loglik_target, loglik_surrog, _logImportanceWeights, ImportanceWeights, ess_cur, ess_target, psis_pareto_k)

    for i in eachindex(loglik_surrog)
        # recompute the log unnormalized importance weight
        _logImportanceWeights[i] = loglik_target[i] .- loglik_surrog[i]

        if length(_logImportanceWeights[i]) == 1
            # make sure the ESS is equal to the target
            ImportanceWeights[i] = [1.0,]
            ess_cur[i] = ess_target

        elseif length(_logImportanceWeights[i]) != 1
            if all(isapprox.(_logImportanceWeights[i], 0.0; atol = sqrt(eps())))
                fill!(ImportanceWeights[i], 1/length(ImportanceWeights[i]))
                ess_cur[i] = ess_target
                psis_pareto_k[i] = 0.0
            else
                # might fail if not enough samples to fit pareto
                try
                    # pareto smoothed importance weights
                    psiw = ParetoSmooth.psis(reshape(copy(_logImportanceWeights[i]), 1, length(_logImportanceWeights[i]), 1); source = "other");
    
                    # save normalized importance weights and ess
                    copyto!(ImportanceWeights[i], psiw.weights)
                    ess_cur[i] = psiw.ess[1]
                    psis_pareto_k[i] = psiw.pareto_k[1]
    
                catch err
                    # exponentiate and normalize the unnormalized log weights
                    copyto!(ImportanceWeights[i], normalize(exp.(_logImportanceWeights[i]), 1))

                    # calculate the ess
                    ess_cur[i] = 1 / sum(ImportanceWeights[i] .^ 2)
                    psis_pareto_k[i] = 1.0
                end
            end
        end
    end
end


# =============================================================================
# Phase-Type Surrogate Sampling Functions
# =============================================================================
#
# These functions implement forward-filtering backward-sampling (FFBS) on an
# expanded phase-type state space for improved MCEM importance sampling.
#
# Key idea: Each observed state is expanded into multiple phases. The expanded
# Markov chain better approximates non-exponential sojourn times. After sampling
# in the expanded space, paths are collapsed back to observed states.
# =============================================================================

"""
    build_phasetype_tpm_book(surrogate::PhaseTypeSurrogate, books, data)

Build transition probability matrix book for phase-type expanded state space.

# Arguments
- `surrogate`: PhaseTypeSurrogate with expanded Q matrix
- `books`: Time interval book from build_tpm_mapping
- `data`: Model data

# Returns
- `tpm_book_ph`: Nested vector of TPMs [covar_combo][time_interval] in expanded space
- `hazmat_book_ph`: Vector of intensity matrices for each covariate combination
"""
function build_phasetype_tpm_book(surrogate::PhaseTypeSurrogate, books, data)
    n_expanded = surrogate.n_expanded_states
    Q_expanded = surrogate.expanded_Q
    
    # books[1] is a vector of DataFrames, one per covariate combination
    # Each DataFrame has columns (tstart, tstop, datind) with rows for unique time intervals
    n_covar_combos = length(books[1])
    
    # Allocate TPM book: [covar_combo][time_interval]
    # For phase-type with homogeneous Q, we still need this structure to match
    # how the Markov surrogate book is organized
    tpm_book_ph = [[zeros(Float64, n_expanded, n_expanded) for _ in 1:nrow(books[1][c])] for c in 1:n_covar_combos]
    
    # Allocate hazmat book: one Q per covariate combination (all same for homogeneous PH)
    hazmat_book_ph = [copy(Q_expanded) for _ in 1:n_covar_combos]
    
    # Allocate cache for matrix exponential
    cache = ExponentialUtilities.alloc_mem(Q_expanded, ExpMethodGeneric())
    
    # Compute TPMs for each covariate combination and time interval
    for c in 1:n_covar_combos
        tpm_index = books[1][c]  # DataFrame with (tstart, tstop, datind)
        for t in 1:nrow(tpm_index)
            dt = tpm_index.tstop[t]  # Time interval length (tstart is always 0)
            # TPM = exp(Q * dt)
            tpm_book_ph[c][t] .= exponential!(copy(Q_expanded) .* dt, ExpMethodGeneric(), cache)
        end
    end
    
    return tpm_book_ph, hazmat_book_ph
end


"""
    build_phasetype_emat_expanded(model, surrogate::PhaseTypeSurrogate)

Build emission matrix mapping expanded phases to observed states for FFBS.

For each observation, the emission matrix E[i,j] = 1 if phase j corresponds
to the observed state at that observation, 0 otherwise.

# Returns
- Matrix of size (n_observations, n_expanded_states)
"""
function build_phasetype_emat_expanded(model, surrogate::PhaseTypeSurrogate)
    n_obs = nrow(model.data)
    n_expanded = surrogate.n_expanded_states
    
    emat = zeros(Float64, n_obs, n_expanded)
    
    for i in 1:n_obs
        obstype = model.data.obstype[i]
        
        if obstype == 1 || obstype == 2
            # Exact or right-censored observation
            # Only the phases corresponding to the observed state are possible
            observed_state = model.data.stateto[i]
            phases = surrogate.state_to_phases[observed_state]
            emat[i, phases] .= 1.0
        elseif obstype == 3
            # Interval censored - any state in the censored set is possible
            # For now, allow all non-absorbing phases
            for s in 1:surrogate.n_observed_states
                phases = surrogate.state_to_phases[s]
                emat[i, phases] .= 1.0
            end
        else
            # Unknown observation type, allow all phases
            emat[i, :] .= 1.0
        end
        
        # Normalize if any positive entries
        row_sum = sum(emat[i, :])
        if row_sum > 0
            emat[i, :] ./= row_sum
        end
    end
    
    return emat
end


"""
    build_fbmats_phasetype(model, surrogate::PhaseTypeSurrogate)

Allocate forward-backward matrices for FFBS on expanded phase-type state space.

# Returns
- Vector of 3D arrays, one per subject, of size (n_obs, n_expanded, n_expanded)
"""
function build_fbmats_phasetype(model, surrogate::PhaseTypeSurrogate)
    n_expanded = surrogate.n_expanded_states
    
    fbmats = Vector{Array{Float64, 3}}(undef, length(model.subjectindices))
    
    for i in eachindex(model.subjectindices)
        subj_inds = model.subjectindices[i]
        n_obs = length(subj_inds)
        fbmats[i] = zeros(Float64, n_obs, n_expanded, n_expanded)
    end
    
    return fbmats
end


# =============================================================================
# Phase-Type Sampling Functions
# =============================================================================
#
# These functions support MCEM with phase-type proposals. The key insight is that
# a phase-type expanded model is still Markov, so we can reuse the existing FFBS
# machinery. The only differences are:
#
# 1. The state space is expanded (each observed state split into phases)
# 2. The emission matrix duplicates indicators across phases of the same state
# 3. After sampling, we collapse the expanded path back to observed states
#
# =============================================================================

"""
    expand_emat(emat, surrogate::PhaseTypeSurrogate)

Expand emission matrix from observed states to phase-type expanded states.

For each row (observation), the emission probability for an observed state is
duplicated across all phases of that state. This is correct because:
  P(obstype | phase_k of state s) = P(obstype | state s)

# Arguments
- `emat`: Original emission matrix (n_obs × n_observed_states)
- `surrogate`: PhaseTypeSurrogate with state-to-phase mappings

# Returns
- Expanded emission matrix (n_obs × n_expanded_states)
"""
function expand_emat(emat::AbstractMatrix, surrogate::PhaseTypeSurrogate)
    n_obs = size(emat, 1)
    n_expanded = surrogate.n_expanded_states
    
    emat_expanded = zeros(Float64, n_obs, n_expanded)
    
    for obs in 1:n_obs
        for (state, phases) in enumerate(surrogate.state_to_phases)
            emission_prob = emat[obs, state]
            for phase in phases
                emat_expanded[obs, phase] = emission_prob
            end
        end
    end
    
    return emat_expanded
end


"""
    BackwardSampling_expanded(subj_fbmats, surrogate::PhaseTypeSurrogate)

Backward sampling that returns expanded state indices.

Unlike `BackwardSampling!` which writes observed states to `subj_dat`, this
function returns the sampled expanded state sequence. The caller is responsible
for mapping back to observed states.

# Arguments
- `subj_fbmats`: Forward-backward matrices from ForwardFiltering!
- `surrogate`: PhaseTypeSurrogate (used only for n_expanded_states)

# Returns
- `Vector{Int}`: Sampled expanded state at each observation time
"""
function BackwardSampling_expanded(subj_fbmats, n_expanded::Int)
    n_times = size(subj_fbmats, 1)
    
    expanded_states = Vector{Int}(undef, n_times)
    
    # Sample final state
    p = normalize(vec(sum(subj_fbmats[n_times, :, :], dims=1)), 1)
    expanded_states[n_times] = rand(Categorical(p))
    
    # Backward recursion
    if n_times > 1
        for t in (n_times - 1):-1:1
            cond_probs = normalize(subj_fbmats[t+1, :, expanded_states[t+1]], 1)
            expanded_states[t] = rand(Categorical(cond_probs))
        end
    end
    
    return expanded_states
end


"""
    draw_samplepath_phasetype(subj, model, tpm_book_ph, hazmat_book_ph, 
                               tpm_map, fbmats_ph, emat_ph, surrogate, absorbingstates)

Draw a sample path from the phase-type surrogate, collapsed to observed states.

Uses the existing FFBS machinery on the expanded state space, then collapses
the sampled path back to observed states.

# Algorithm
1. Run ForwardFiltering! on expanded state space (existing function)
2. Run BackwardSampling_expanded to get expanded state endpoints
3. Run sample_ecctmc! on expanded Q matrix between expanded endpoints
4. Collapse the full path from expanded to observed states

# Returns
- `SamplePath`: Path in observed (not expanded) state space
"""
function draw_samplepath_phasetype(subj::Int64, model::MultistateProcess, 
                                    tpm_book_ph, hazmat_book_ph, tpm_map, 
                                    fbmats_ph, emat_ph, surrogate::PhaseTypeSurrogate, 
                                    absorbingstates)
    
    # Subject data
    subj_inds = model.subjectindices[subj]
    subj_dat = view(model.data, subj_inds, :)
    subj_tpm_map = view(tpm_map, subj_inds, :)
    subj_emat_ph = view(emat_ph, subj_inds, :)
    
    n_obs = length(subj_inds)
    n_expanded = surrogate.n_expanded_states
    
    # Initial expanded state: first phase of initial observed state
    init_expanded = first(surrogate.state_to_phases[subj_dat.statefrom[1]])
    
    # Get expanded state endpoints via FFBS if there are censored observations
    if any(subj_dat.obstype .∉ Ref([1, 2]))
        # Forward filtering on expanded space (reuse existing function with init_state)
        ForwardFiltering!(fbmats_ph[subj], subj_dat, tpm_book_ph, subj_tpm_map, subj_emat_ph;
                          init_state=init_expanded)
        
        # Backward sample to get expanded state sequence
        expanded_states = BackwardSampling_expanded(fbmats_ph[subj], n_expanded)
    else
        # No censored observations - map observed states to first phase
        expanded_states = [first(surrogate.state_to_phases[subj_dat.stateto[i]]) for i in 1:n_obs]
    end
    
    # Initialize path in expanded space
    times_expanded = [subj_dat.tstart[1]]
    states_expanded = [init_expanded]
    sizehint!(times_expanded, n_expanded * 2)
    sizehint!(states_expanded, n_expanded * 2)
    
    # Loop through intervals and sample endpoint-conditioned paths in expanded space
    for i in 1:n_obs
        # Get transition probability matrix and rate matrix for this interval
        covar_idx = subj_tpm_map[i, 1]
        time_idx = subj_tpm_map[i, 2]
        P_expanded = tpm_book_ph[covar_idx][time_idx]
        Q_expanded = hazmat_book_ph[covar_idx]
        
        # Source state in expanded space
        a_expanded = states_expanded[end]
        
        if subj_dat.obstype[i] == 1
            # Exact/continuous observation - path is OBSERVED, not sampled
            # Just record the observed transition at tstop
            dest_obs_state = subj_dat.stateto[i]
            
            if subj_dat.statefrom[i] != dest_obs_state
                # Observed transition: record it at the observed time
                # Map to first phase of destination state
                b_expanded = first(surrogate.state_to_phases[dest_obs_state])
                push!(times_expanded, subj_dat.tstop[i])
                push!(states_expanded, b_expanded)
            end
            # If no transition (statefrom == stateto), nothing to record
        else
            # Censored/panel observation - use FFBS-sampled destination
            b_expanded = expanded_states[i]
            
            # Sample path between expanded endpoints
            sample_ecctmc!(times_expanded, states_expanded, P_expanded, Q_expanded, 
                          a_expanded, b_expanded, subj_dat.tstart[i], subj_dat.tstop[i])
            
            # Ensure we end at the sampled destination
            if states_expanded[end] != b_expanded
                push!(times_expanded, subj_dat.tstop[i])
                push!(states_expanded, b_expanded)
            end
        end
    end
    
    # Create expanded SamplePath for surrogate likelihood
    expanded_path = SamplePath(subj, copy(times_expanded), copy(states_expanded))
    
    # Collapse expanded path to observed states
    collapsed_path = collapse_phasetype_path(expanded_path, surrogate, absorbingstates)
    
    # Return both collapsed and expanded paths
    # Collapsed path is used for target likelihood
    # Expanded path is used for surrogate likelihood
    return (collapsed=collapsed_path, expanded=expanded_path)
end


"""
    collapse_phasetype_path(expanded_path::SamplePath, surrogate::PhaseTypeSurrogate, absorbingstates)

Collapse a path from the expanded phase-type state space to the observed state space.

Maps each phase back to its corresponding observed state and removes consecutive 
duplicates (transitions between phases of the same state).

# Arguments
- `expanded_path`: SamplePath in the expanded phase state space
- `surrogate`: PhaseTypeSurrogate with phase_to_state mapping
- `absorbingstates`: Vector of absorbing state indices

# Returns
- `SamplePath`: Path in the observed (collapsed) state space
"""
function collapse_phasetype_path(expanded_path::SamplePath, surrogate::PhaseTypeSurrogate, absorbingstates)
    times_expanded = expanded_path.times
    states_expanded = expanded_path.states
    subj = expanded_path.subj
    
    # Map to observed states, keeping only transitions that change observed state
    times_obs = [times_expanded[1]]
    states_obs = [surrogate.phase_to_state[states_expanded[1]]]
    
    for i in 2:length(times_expanded)
        obs_state = surrogate.phase_to_state[states_expanded[i]]
        # Only record if observed state changes
        if obs_state != states_obs[end]
            push!(times_obs, times_expanded[i])
            push!(states_obs, obs_state)
        end
    end
    
    # Truncate at absorbing states
    truncind = findfirst(states_obs .∈ Ref(absorbingstates))
    if !isnothing(truncind)
        times_obs = first(times_obs, truncind)
        states_obs = first(states_obs, truncind)
    end
    
    return reduce_jumpchain(SamplePath(subj, times_obs, states_obs))
end


"""
    loglik_phasetype_expanded(expanded_path::SamplePath, surrogate::PhaseTypeSurrogate)

Compute log-likelihood of an expanded sample path under the phase-type surrogate.

This computes the CTMC path density in the expanded state space:
  log f(path) = sum of log(survival) + sum of log(hazard at transitions)

For importance sampling:
  log_weight = loglik_target(collapsed_path) - loglik_phasetype_expanded(expanded_path)

# Arguments
- `expanded_path`: Sample path in the expanded phase state space
- `surrogate`: PhaseTypeSurrogate with the expanded Q matrix

# Returns
- `Float64`: Log-likelihood (density) of the expanded path
"""
function loglik_phasetype_expanded(expanded_path::SamplePath, surrogate::PhaseTypeSurrogate)
    loglik = 0.0
    Q = surrogate.expanded_Q
    
    n_transitions = length(expanded_path.times) - 1
    if n_transitions == 0
        return 0.0
    end
    
    for i in 1:n_transitions
        t0 = expanded_path.times[i]
        t1 = expanded_path.times[i + 1]
        dt = t1 - t0
        
        s = expanded_path.states[i]
        d = expanded_path.states[i + 1]
        
        # Survival term: exp(-q_s * dt) where q_s is total exit rate from s
        q_s = -Q[s, s]  # Diagonal is negative total exit rate
        loglik += -q_s * dt
        
        # Transition term: log(q_{s,d}) if s != d
        if s != d
            q_sd = Q[s, d]
            if q_sd > 0
                loglik += log(q_sd)
            else
                return -Inf  # Impossible transition
            end
        end
    end
    
    return loglik
end


"""
    loglik_phasetype_path(path::SamplePath, surrogate::PhaseTypeSurrogate, 
                          model::MultistateProcess)

Compute log-likelihood of a sample path under the phase-type surrogate.

This computes the **marginal** probability of the collapsed path under the
phase-type distribution, marginalizing over all possible phase sequences.

For importance sampling:
  log_weight = loglik_target - loglik_surrogate
  
where loglik_surrogate is the marginal probability of the observed transitions.

# Algorithm

For each transition from observed state `s` to state `d` over time `dt`:
1. Start from the stationary phase distribution within state `s`
2. Compute probability of ending in any phase of state `d`
3. This is: π_s' * P_dt * 1_d where π_s is the phase distribution and 1_d sums phases of d

For the first transition, we start in phase 1 of the initial state.
For subsequent transitions, we use the conditional phase distribution given
we're in state `s`.

# Arguments
- `path`: Sample path in observed state space
- `surrogate`: PhaseTypeSurrogate
- `model`: Original model (for data access)

# Returns
- `Float64`: Log-likelihood (density, marginal over phases) under phase-type surrogate
"""
function loglik_phasetype_path(path::SamplePath, surrogate::PhaseTypeSurrogate,
                                model::MultistateProcess)
    
    loglik = 0.0
    Q = surrogate.expanded_Q
    n_expanded = surrogate.n_expanded_states
    
    n_transitions = length(path.times) - 1
    if n_transitions == 0
        return 0.0  # No transitions
    end
    
    # The density of a collapsed path under phase-type is computed as follows:
    # For each observed state sojourn, we compute the marginal density of:
    # 1. Surviving in that state for duration dt
    # 2. Transitioning to the next observed state at time t
    #
    # This requires integrating over all possible phase sequences within states.
    # We use the matrix exponential formula for phase-type distributions.
    
    # Current distribution over phases - start in first phase of initial state
    phase_dist = zeros(Float64, n_expanded)
    init_phases = surrogate.state_to_phases[path.states[1]]
    phase_dist[first(init_phases)] = 1.0
    
    for i in 1:n_transitions
        t0 = path.times[i]
        t1 = path.times[i + 1]
        dt = t1 - t0
        
        s_obs = path.states[i]
        d_obs = path.states[i + 1]
        
        # Get phases for source and destination states
        s_phases = surrogate.state_to_phases[s_obs]
        d_phases = surrogate.state_to_phases[d_obs]
        
        # Compute the phase-type density contribution for this transition
        # The density is: π' * exp(S*dt) * s_d
        # where π is current phase distribution within source state
        # S is the sub-generator for phases within source state
        # s_d is the exit rate vector to destination state
        
        # Extract sub-generator S for source state
        # S[i,j] = Q[s_phases[i], s_phases[j]] for within-state transitions
        n_phases_s = length(s_phases)
        
        if n_phases_s == 1
            # Single phase case (exponential distribution)
            phase_idx = first(s_phases)
            
            # Total exit rate from this phase to destination state
            exit_rate_to_dest = sum(Q[phase_idx, d_phases])
            
            # Total exit rate (for survival)
            total_exit_rate = -Q[phase_idx, phase_idx]
            
            # Log-density: survival * exit_rate_to_dest
            # log f(dt) = -total_exit_rate * dt + log(exit_rate_to_dest)
            if exit_rate_to_dest > 0
                loglik += -total_exit_rate * dt + log(exit_rate_to_dest)
            else
                return -Inf
            end
            
            # Update phase distribution (now in destination state, first phase)
            fill!(phase_dist, 0.0)
            phase_dist[first(d_phases)] = 1.0
        else
            # Multi-phase case: need to use matrix exponential
            # Construct within-state sub-generator and exit rates to destination
            S_within = zeros(Float64, n_phases_s, n_phases_s)
            exit_to_dest = zeros(Float64, n_phases_s)
            
            for (ii, pi) in enumerate(s_phases)
                for (jj, pj) in enumerate(s_phases)
                    S_within[ii, jj] = Q[pi, pj]
                end
                # Exit rate to destination state
                exit_to_dest[ii] = sum(Q[pi, d_phases])
            end
            
            # Current distribution over phases within source state
            pi_s = phase_dist[s_phases]
            pi_s = pi_s / sum(pi_s)  # Normalize to within-state distribution
            
            # Compute exp(S*dt) * exit_to_dest
            expSdt = exp(S_within .* dt)
            
            # Density: π' * exp(S*dt) * exit_to_dest
            density = transpose(pi_s) * expSdt * exit_to_dest
            
            if density > 0
                loglik += log(density)
            else
                return -Inf
            end
            
            # Update phase distribution: conditional on transitioning to destination
            # The probability of arriving in each destination phase given we left source
            # is proportional to the rate of exiting to that phase
            fill!(phase_dist, 0.0)
            arrived_dist = expSdt' * pi_s
            for (ii, pi) in enumerate(s_phases)
                for dj in d_phases
                    rate_to_dj = Q[pi, dj]
                    if rate_to_dj > 0
                        phase_dist[dj] += arrived_dist[ii] * rate_to_dj
                    end
                end
            end
            # Normalize
            phase_dist ./= sum(phase_dist)
        end
    end
    
    return loglik
end