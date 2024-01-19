########################################################
### Likelihod functions for sample paths, tpms, etc. ###
########################################################

### Exactly observed sample paths ----------------------
"""
    loglik(parameters, path::SamplePath, hazards::Vector{<:_Hazard}, model::MultistateProcess) 

Log-likelihood for a single sample path. The sample path object is `path::SamplePath` and contains the subject index and the jump chain.
"""
function loglik(parameters, path::SamplePath, hazards::Vector{<:_Hazard}, model::MultistateProcess) 

    # initialize log likelihood
    ll = 0.0

    # subject data
    subj_inds = model.subjectindices[path.subj]
    subj_dat  = view(model.data, subj_inds, :)

    # number of jumps/left endpoints
    n_intervals = length(path.times) - 1

    # current index
    subj_dat_ind = 1 # row in subject data
    comp_dat_ind = subj_inds[subj_dat_ind] # index in complete data

    # data time interval
    time_R = subj_dat.tstop[subj_dat_ind]

    # recurse through the sample path
    for i in Base.OneTo(n_intervals)

        # current and next state
        scur  = path.states[i]
        snext = path.states[i+1]

        # keep accruing log-likelihood contributions for sojourn
        keep_going = isa(model.totalhazards[scur], _TotalHazardTransient)

        # times in the jump chain (clock forward)
        # gets reset each time as i gets incremented
        tcur  = path.times[i]   # current time
        tstop = path.times[i+1] # time of next jump

        # time in state (clock reset)
        timespent   = 0.0   # accumulates occupancy time
        timeinstate = tstop - tcur # sojourn time

        # initialize survival probability
        log_surv_prob = 0.0

        # accumulate log likelihood
        while keep_going

            if tstop <= time_R
                # event happens in (time_L, time_R]
                # accumulate log(Pr(T ≥ timeinstate | T ≥ timespent))
                log_surv_prob += survprob(timespent, timeinstate, parameters, comp_dat_ind, model.totalhazards[scur], hazards; give_log = true)

                # increment log likelihood
                ll += log_surv_prob

                # if event happened, accrue hazard
                if snext != scur
                    ll += call_haz(timeinstate, parameters[model.tmat[scur, snext]], comp_dat_ind, hazards[model.tmat[scur, snext]]; give_log = true)
                end

                # increment row index in subj_dat
                if (tstop == time_R) & (subj_dat_ind != size(subj_dat, 1))
                    subj_dat_ind += 1
                    comp_dat_ind += 1

                    # increment time_R
                    time_R = subj_dat.tstop[subj_dat_ind]
                end

                # break out of the while loop
                keep_going = false

            else
                # event doesn't hapen in (time_L, time_R]
                # accumulate log-survival
                # accumulate log(Pr(T ≥ time_R | T ≥ timespent))
                log_surv_prob += survprob(timespent, timespent + time_R - tcur, parameters, comp_dat_ind, model.totalhazards[scur], hazards; give_log = true)
                
                # increment timespent
                timespent += time_R - tcur

                # increment current time
                tcur = time_R

                # increment row index in subj_dat
                if subj_dat_ind != size(subj_dat, 1)
                    subj_dat_ind += 1
                    comp_dat_ind += 1

                    # increment time_R
                    time_R = subj_dat.tstop[subj_dat_ind]
                end
            end
        end
    end

    # unweighted loglikelihood
    return ll
end

########################################################
##################### Wrappers #########################
########################################################

"""
    loglik(parameters, data::ExactData; neg = true) 

Return sum of (negative) log likelihoods for all sample paths. Use mapreduce() to call loglik() and sum the results. Each sample path object is `path::SamplePath` and contains the subject index and the jump chain. 
"""
function loglik(parameters, data::ExactData; neg=true, return_ll_subj=false)

    pars = VectorOfVectors(parameters, data.model.parameters.elem_ptr)

    if return_ll_subj
        # send each element of samplepaths to loglik
        map((x, w) -> loglik(pars, x, data.model.hazards, data.model) * w, data.paths, data.model.SamplingWeights) # weighted
    else
        ll = mapreduce((x, w) -> loglik(pars, x, data.model.hazards, data.model) * w, +, data.paths, data.model.SamplingWeights)    
        neg ? -ll : ll
    end
end

"""
    loglik(parameters, data::ExactDataAD; neg = true) 

Return sum of (negative) log likelihoods for all sample paths. Use mapreduce() to call loglik() and sum the results. Each sample path object is `path::SamplePath` and contains the subject index and the jump chain. 
"""
function loglik(parameters, data::ExactDataAD; neg = true)

    pars = VectorOfVectors(parameters, data.model.parameters.elem_ptr)

    # send each element of samplepaths to loglik
    ll = loglik(pars, data.path[1], data.model.hazards, data.model) * data.samplingweight[1]

    neg ? -ll : ll
end

"""
    loglik(parameters, data::MPanelData; neg = true)

Return sum of (negative) log likelihood for a Markov model fit to panel and/or exact and/or censored data. 
"""
function loglik(parameters, data::MPanelData; neg = true, return_ll_subj = false)

    # nest the model parameters
    pars = VectorOfVectors(parameters, data.model.parameters.elem_ptr)

    # build containers for transition intensity and prob mtcs
    hazmat_book = build_hazmat_book(eltype(parameters), data.model.tmat, data.books[1])
    tpm_book = build_tpm_book(eltype(parameters), data.model.tmat, data.books[1])

    # allocate memory for matrix exponential
    cache = ExponentialUtilities.alloc_mem(similar(hazmat_book[1]), ExpMethodGeneric())

    # Solve Kolmogorov equations for TPMs
    for t in eachindex(data.books[1])

        # compute the transition intensity matrix
        compute_hazmat!(
            hazmat_book[t],
            pars,
            data.model.hazards,
            data.books[1][t])

        # compute transition probability matrices
        compute_tmat!(
            tpm_book[t],
            hazmat_book[t],
            data.books[1][t],
            cache)
    end

    # number of subjects
    nsubj = length(data.model.subjectindices)

    # accumulate the log likelihood
    ll = 0.0

    # container for subject-level loglikelihood
    if return_ll_subj
        ll_subj = zeros(Float64, nsubj)
    end

    # number of states
    S = size(data.model.tmat, 1)

    # initialize Q 
    q = zeros(eltype(parameters), S, S)

    # for each subject, compute the likelihood contribution
    for subj in Base.OneTo(nsubj)

        # subject data
        subj_inds = data.model.subjectindices[subj]
        # subj_dat  = view(data.model.data, subj_inds, :)

        # no state is censored
        if all(data.model.data.obstype[subj_inds] .∈ Ref([1,2]))
            
            # subject contribution to the loglikelihood
            subj_ll = 0.0

            # add the contribution of each observation
            for i in subj_inds
                if data.model.data.obstype[i] == 1 # exact data
                    
                    subj_ll += survprob(0, data.model.data.tstop[i] - data.model.data.tstart[i], pars, i, data.model.totalhazards[data.model.data.statefrom[i]], data.model.hazards; give_log = true)
                                        
                    if data.model.data.statefrom[i] != data.model.data.stateto[i] # if there is a transition, add log hazard
                        subj_ll += call_haz(data.model.data.tstop[i] - data.model.data.tstart[i], pars[data.model.tmat[data.model.data.statefrom[i], data.model.data.stateto[i]]], i, data.model.hazards[data.model.tmat[data.model.data.statefrom[i], data.model.data.stateto[i]]]; give_log = true)
                    end

                else # panel data
                    subj_ll += log(tpm_book[data.books[2][i, 1]][data.books[2][i, 2]][data.model.data.statefrom[i], data.model.data.stateto[i]])
                end
            end

        else
            # initialize likelihood matrix
            lmat = zeros(eltype(parameters), S, length(subj_inds) + 1)
            lmat[data.model.data.statefrom[subj_inds[1]], 1] = 1

            # initialize counter for likelihood matrix
            ind = 1

            # update the vector l
            for i in subj_inds

                # increment counter for likelihood matrix
                ind += 1

                # compute q, the transition probability matrix
                if data.model.data.obstype[i] != 1
                    # if panel data, simply grab q from tpm_book
                    copyto!(q, tpm_book[data.books[2][i, 1]][data.books[2][i, 2]])
                    
                else
                    # if exact data (obstype = 1), compute q by hand
                    # reset Q
                    fill!(q, -Inf)
                    
                    # compute q(r,s)
                    for r in 1:S
                        if isa(data.model.totalhazards[r], _TotalHazardAbsorbing)
                            q[r,r] = 0.0
                        else
                            # survival probability
                            q[r, findall(data.model.tmat[r,:] .!= 0)] .= survprob(0, data.model.data.tstop[i] - data.model.data.tstart[i], pars, i, data.model.totalhazards[r], data.model.hazards; give_log = true) 
                            
                            # hazard
                            for s in 1:S
                                if (s != r) & (data.model.tmat[r,s] != 0)
                                    q[r, s] += call_haz(data.model.data.tstop[i] - data.model.data.tstart[i], pars[data.model.tmat[r, s]], i, data.model.hazards[data.model.tmat[r, s]]; give_log = true)
                                end
                            end
                        end

                        # pedantic b/c of numerical error
                        q[r,r] = maximum([1 - exp(logsumexp(q[r, Not(r)])), eps()])
                        q[r,Not(r)] = exp.(q[r, Not(r)])               
                    end
                end # end-compute q

                # compute the set of possible "states to"
                StatesTo = data.model.data.stateto[i] > 0 ? [data.model.data.stateto[i]] : findall(data.model.emat[i,:] .== 1)

                for s in 1:S
                    if s ∈ StatesTo
                        for r in 1:S
                            lmat[s, ind] += q[r,s] * lmat[r, ind - 1]
                        end
                    end
                end
            end

            # log likelihood
            subj_ll=log(sum(lmat[:,size(lmat, 2)]))
        end

        if return_ll_subj
            # weighted subject loglikelihood
            ll_subj[subj] = subj_ll * data.model.SamplingWeights[subj]
        else
            # weighted loglikelihood
            ll += subj_ll * data.model.SamplingWeights[subj]
        end        
    end

    if return_ll_subj
        ll_subj
    else
        neg ? -ll : ll
    end
end

"""
    loglik(parameters, data::SMPanelData; neg = true)

Return sum of (negative) complete data log-likelihood terms in the Monte Carlo maximum likelihood algorithm for fitting a semi-Markov model to panel data. 
"""
function loglik(parameters, data::SMPanelData; neg = true, use_sampling_weight = true)

    # nest the model parameters
    pars = VectorOfVectors(parameters, data.model.parameters.elem_ptr)

    # compute the semi-markov log-likelihoods
    ll = 0.0
    for i in eachindex(data.paths)
        lls = 0.0
        for j in eachindex(data.paths[i])
            lls += loglik(pars, data.paths[i][j], data.model.hazards, data.model) * data.ImportanceWeights[i][j] 
        end
        if use_sampling_weight
            lls *= data.model.SamplingWeights[i]
        end
        ll += lls
    end

    # return the log-likelihood
    neg ? -ll : ll
end

"""
    loglik(parameters, data::SMPanelData)

Return sum of (negative) complete data log-likelihood terms in the Monte Carlo maximum likelihood algorithm for fitting a semi-Markov model to panel data. 
"""
function loglik!(parameters, logliks::Vector{}, data::SMPanelData; use_sampling_weight = false)

    # nest the model parameters
    pars = VectorOfVectors(parameters, data.model.parameters.elem_ptr)

    for i in eachindex(data.paths)
        for j in eachindex(data.paths[i])
            logliks[i][j] = use_sampling_weight ? loglik(pars, data.paths[i][j], data.model.hazards, data.model) * data.model.SamplingWeights[i] : loglik(pars, data.paths[i][j], data.model.hazards, data.model)
        end
    end
end


# """
#     loglik(parameters, data::MPanelData; neg = true)

# Return sum of (negative) log likelihood for a Markov model fit to panel and/or exact and/or censored data. 
# """
# function loglik(parameters, data::MPanelData; neg = true)

#     # nest the model parameters
#     pars = VectorOfVectors(parameters, data.model.parameters.elem_ptr)

#     # build containers for transition intensity and prob mtcs
#     hazmat_book = build_hazmat_book(eltype(parameters), data.model.tmat, data.books[1])
#     tpm_book = build_tpm_book(eltype(parameters), data.model.tmat, data.books[1])

#     # allocate memory for matrix exponential
#     cache = ExponentialUtilities.alloc_mem(similar(hazmat_book[1]), ExpMethodGeneric())

#     # Solve Kolmogorov equations for TPMs
#     for t in eachindex(data.books[1])

#         # compute the transition intensity matrix
#         compute_hazmat!(
#             hazmat_book[t],
#             pars,
#             data.model.hazards,
#             data.books[1][t])

#         # compute transition probability matrices
#         compute_tmat!(
#             tpm_book[t],
#             hazmat_book[t],
#             data.books[1][t],
#             cache)
#     end

#     # accumulate the log likelihood
#     ll = 0.0

#     # number of states
#     S = size(data.model.tmat, 1)

#     # initialize Q
#     q = zeros(eltype(parameters), S, S)

#     # initialize l_t0 and l_t1
#     P_0 = zeros(eltype(parameters), S, S)
#     P_1 = zeros(eltype(parameters), S, S)
    
#     # for each subject, compute the likelihood contribution
#     for subj in Base.OneTo(length(data.model.subjectindices))

#         # subject data
#         subj_inds = data.model.subjectindices[subj]

#         # subject contribution to the loglikelihood
#         subj_ll = 0.0

#         # no state is censored
#         if all(data.model.data.obstype[subj_inds] .∈ Ref([1,2]))

#             # add the contribution of each observation
#             for i in subj_inds
#                 if data.model.data.obstype[i] == 1 # exact data
                    
#                     subj_ll += survprob(0, data.model.data.tstop[i] - data.model.data.tstart[i], pars, i, data.model.totalhazards[data.model.data.statefrom[i]], data.model.hazards; give_log = true)
                                        
#                     if data.model.data.statefrom[i] != data.model.data.stateto[i] # if there is a transition, add log hazard
#                         subj_ll += call_haz(data.model.data.tstop[i] - data.model.data.tstart[i], pars[data.model.tmat[data.model.data.statefrom[i], data.model.data.stateto[i]]], i, data.model.hazards[data.model.tmat[data.model.data.statefrom[i], data.model.data.stateto[i]]]; give_log = true)
#                     end

#                 else # panel data
#                     subj_ll += log(tpm_book[data.books[2][i, 1]][data.books[2][i, 2]][data.model.data.statefrom[i], data.model.data.stateto[i]])
#                 end
#             end

#         else
#             # reset P_0, and P_1
#             fill!(P_0, 0.0)
#             fill!(P_1, 0.0)

#             # counter
#             n_inds = length(subj_inds)
#             ind = 0

#             # update the vector l
#             for i in subj_inds

#                 # increment counter
#                 ind += 1

#                 # marginal probabilities
#                 if ind == 1
#                     marg_probs = zeros(eltype(parameters), S); 
#                     marg_probs[data.model.data.statefrom[subj_inds[1]]] = 1.0
#                 else
#                     marg_probs = sum(P_0, dims = 1)
#                 end

#                 # compute q, the transition probability matrix
#                 if data.model.data.obstype[i] != 1
#                     # if panel data, simply grab q from tpm_book
#                     copyto!(q, tpm_book[data.books[2][i, 1]][data.books[2][i, 2]])
                    
#                 else
#                     # if exact data (obstype = 1), compute q by hand
#                     # reset Q
#                     fill!(q, -Inf)
                    
#                     # compute q(r,s)
#                     for r in 1:S
#                         if isa(data.model.totalhazards[r], _TotalHazardAbsorbing)
#                             q[r,r] = 0.0
#                         else
#                             # survival probability
#                             q[r, findall(data.model.tmat[r,:] .!= 0)] .= survprob(0, data.model.data.tstop[i] - data.model.data.tstart[i], pars, i, data.model.totalhazards[r], data.model.hazards; give_log = true) 
                            
#                             # hazard
#                             for s in 1:S
#                                 if (s != r) & (data.model.tmat[r,s] != 0)
#                                     q[r, s] += call_haz(data.model.data.tstop[i] - data.model.data.tstart[i], pars[data.model.tmat[r, s]], i, data.model.hazards[data.model.tmat[r, s]]; give_log = true)
#                                 end
#                             end
#                         end
#                         q[r,:] = softmax(q[r,:])
#                     end
#                 end # end-compute q

#                 # recursion
#                 P_1 = normalize(transpose(marg_probs) * transpose(data.model.emat[i,:]) .* q, 1)

#                 # accumulate likelihood


#                 # swap P_0 and P_1
#                 if ind != n_inds
#                     P_0, P_1 = P_1, P_0
#                 end
#             end

#             # log likelihood
#             # subj_ll=log(subj_lik)
#         end
        
#         # weighted loglikelihood
#         ll += subj_ll * data.model.SamplingWeights[subj]
#     end

#     neg ? -ll : ll
# end