"""
    DrawSamplePaths(i, model, ess_target, ess_cur, MaxSamplingEffort, samplepaths, loglik_surrog, loglik_target_prop, loglik_target_cur, ImportanceWeights, tpm_book_surrogate, hazmat_book_surrogate, books, npaths_additional, params_cur, surrogate, psis_pareto_k)

Draw additional sample paths until sufficient ess or until the maximum number of paths is reached
"""
function DrawSamplePaths!(model::MultistateProcess; ess_target, ess_cur, MaxSamplingEffort, samplepaths, loglik_surrog, loglik_target_prop, loglik_target_cur, ImportanceWeights,tpm_book_surrogate, hazmat_book_surrogate, books, npaths_additional, params_cur, surrogate, psis_pareto_k, fbmats)

    for i in eachindex(model.subjectindices)
        DrawSamplePaths!(i, model; 
            ess_target = ess_target,
            ess_cur = ess_cur, 
            MaxSamplingEffort = MaxSamplingEffort,
            samplepaths = samplepaths, 
            loglik_surrog = loglik_surrog, 
            loglik_target_prop = loglik_target_prop, 
            loglik_target_cur = loglik_target_cur, 
            ImportanceWeights = ImportanceWeights, 
            tpm_book_surrogate = tpm_book_surrogate, 
            hazmat_book_surrogate = hazmat_book_surrogate, 
            books = books, 
            npaths_additional = npaths_additional, 
            params_cur = params_cur, 
            surrogate = surrogate, 
            psis_pareto_k = psis_pareto_k,
            fbmats = fbmats)
    end
end

"""
    DrawSamplePaths(i, model, ess_target, ess_cur, MaxSamplingEffort, samplepaths, loglik_surrog, loglik_target_prop, loglik_target_cur, ImportanceWeights, tpm_book_surrogate, hazmat_book_surrogate, books, npaths_additional, params_cur, surrogate, psis_pareto_k)

Draw additional sample paths until sufficient ess or until the maximum number of paths is reached
"""
function DrawSamplePaths!(i, model::MultistateProcess; ess_target, ess_cur, MaxSamplingEffort, samplepaths, loglik_surrog, loglik_target_prop, loglik_target_cur, ImportanceWeights, tpm_book_surrogate, hazmat_book_surrogate, books, npaths_additional, params_cur, surrogate, psis_pareto_k, fbmats)

    n_path_max = MaxSamplingEffort*ess_target
    keep_sampling = ess_cur[i] < ess_target

    # subject data
    subj_inds = model.subjectindices[i]
    subj_dat  = view(model.data, subj_inds, :)

    # compute fbmats here
    if any(subj_dat.obstype .∉ Ref([1,2]))
        # subject data
        subj_tpm_map = view(books[2], subj_inds, :)
        subj_emat    = view(model.emat, subj_inds, :)
        #subj_fbmats  = view(fbmats, i)
        ForwardFiltering!(fbmats[i], subj_dat, tpm_book_surrogate, subj_tpm_map, subj_emat)
    end

    while keep_sampling

        # make sure there are at least 25 paths in order to fit pareto
        npaths = length(samplepaths[i])
        n_add  = npaths == 0 ? maximum([50, ess_target]) : npaths_additional

        # augment the number of paths
        append!(samplepaths[i], Vector{SamplePath}(undef, n_add))
        append!(loglik_surrog[i], zeros(n_add))
        append!(loglik_target_prop[i], zeros(n_add))
        append!(loglik_target_cur[i], zeros(n_add))
        append!(ImportanceWeights[i], zeros(n_add))

        # sample new paths and compute log likelihoods
        for j in npaths.+(1:n_add)
            samplepaths[i][j]       = draw_samplepath(i, model, tpm_book_surrogate, hazmat_book_surrogate, books[2], fbmats)
            loglik_surrog[i][j]     = loglik(surrogate.parameters, samplepaths[i][j], surrogate.hazards, model)
            loglik_target_cur[i][j] = loglik(VectorOfVectors(params_cur, model.parameters.elem_ptr), samplepaths[i][j], model.hazards, model) 
        end

        # no need to keep all paths
        if allequal(loglik_target_cur[i])
            samplepaths[i]        = [first(samplepaths[i]),]
            loglik_target_cur[i]  = [first(loglik_target_cur[i]),]
            loglik_target_prop[i] = [first(loglik_target_prop[i]),]
            loglik_surrog[i]      = [first(loglik_surrog[i]),]
            ImportanceWeights[i]  = [1.0,]
            ess_cur[i]            = ess_target
        else
            # raw log importance weights
            logweights = reshape(loglik_target_cur[i] - loglik_surrog[i], 1, length(loglik_target_cur[i]), 1) 

            # might fail if not enough samples to fit pareto
            try
                # pareto smoothed importance weights
                psiw = psis(logweights; source = "other");

                # save importance weights and ess
                copyto!(ImportanceWeights[i], psiw.weights)
                ess_cur[i] = psiw.ess[1]
                psis_pareto_k[i] = psiw.pareto_k[1]

            catch err
                ess_cur[i] = 0.0
                psis_pareto_k[i] = 0.0
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
function draw_samplepath(subj::Int64, model::MultistateProcess, tpm_book, hazmat_book, tpm_map, fbmats)

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

    return SamplePath(subj, times, states)
end

"""
    ForwardFiltering!(subj_fbmats, subj_dat, tpm_book, subj_tpm_map, subj_emat)

Computes the forward recursion matrices for the FFBS algorithm. Writes into subj_fbmats.
"""
function ForwardFiltering!(subj_fbmats, subj_dat, tpm_book, subj_tpm_map, subj_emat)

    n_times  = size(subj_fbmats, 1)
    n_states = size(subj_fbmats, 2)

    # initialize
    p0 = zeros(Float64, n_states)
    p0[subj_dat.statefrom[1]] = 1.0
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


# function ForwardFiltering(subj_dat, tpm_book, subj_tpm_map, subj_emat) 

#     n_obs    = size(subj_emat, 1) # number of states visited
#     n_states = size(subj_emat, 2) # number of states
#     m = zeros(n_obs+1, n_states)  # matrix of marginal probabilities
#     p = zeros(n_obs, n_states, n_states)  # joint distribution Pr(h_{t-1}=r,h_t=s|d_{1:t})
    
#     # # forward filtering
#     m[1, subj_dat.statefrom[1]] = 1 # first state is assumed to be known
    
#     for t in 1:n_obs # loop over each interval for the subject
#         q_t = tpm_book[subj_tpm_map[t,1]][subj_tpm_map[t,2]]
#         #q_t = view(tpm_book[subj_tpm_map[t,1], subj_tpm_map[t,2]], :, :)
#         # joint p_t
#         p_trs = Array{Float64}(undef, n_states, n_states) # unnormalized p_t
#         for r in 1:n_states, s in 1:n_states
#             p_trs[r,s] = m[t,r] * q_t[r,s] * subj_emat[t,s] # marginal * transition * emission [Eq. 6]
#         end
#         normalizing_constant = sum(p_trs)
#         if normalizing_constant == 0
#             id = subj_dat.id[1]
#             error("No trajectory satisfies the censoring patterns for subject $id.")
#             #error("test error.")
#         end
#         p[t,:,:] = p_trs ./ normalizing_constant # normalize p_t [Eq. 6]
#         # posterior
#         for s in 1:n_states
#             m[t+1,s] = sum(p[t,:,s]) # marginalize the joint distribution p_t
#         end
#     end

#     return m, p

# end