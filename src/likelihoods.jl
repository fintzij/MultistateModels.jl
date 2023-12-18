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

    # weighted loglikelihood
    return ll
end

########################################################
##################### Wrappers #########################
########################################################

"""
    loglik(parameters, data::ExactData; neg = true) 

Return sum of (negative) log likelihoods for all sample paths. Use mapreduce() to call loglik() and sum the results. Each sample path object is `path::SamplePath` and contains the subject index and the jump chain. 
"""
function loglik(parameters, data::ExactData; neg = true)

    pars = VectorOfVectors(parameters, data.model.parameters.elem_ptr)

    # send each element of samplepaths to loglik
    ll = mapreduce((x, w) -> loglik(pars, x, data.model.hazards, data.model) * w, +, data.paths, data.model.SamplingWeights)
    
    neg ? -ll : ll
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
function loglik(parameters, data::MPanelData; neg = true)

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

    # accumulate the log likelihood
    ll = 0.0

    # number of states
    S = size(data.model.tmat, 1)

    # initialize Q
    q = zeros(S,S)

    # initialize l_t0 and l_t1
    l_t0 = zeros(eltype(parameters), S)
    l_t1 = zeros(eltype(parameters), S)
    
    # for each subject, compute the likelihood contribution
    for subj in Base.OneTo(length(data.model.subjectindices))

        # reset q
        fill!(q, 0.0)

        # subject data
        subj_inds = data.model.subjectindices[subj]

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
            # at least one state is censored
            StatesFrom = data.model.data.statefrom[subj_inds[1]] > 0 ? [data.model.data.statefrom[subj_inds[1]]] : findall(data.model.emat[subj_inds[1],:] .== 1)
            
            # initialize l_0
            for s in 1:S
                l_t0[s] = s ∈ StatesFrom ? 1.0 : 0.0
            end

            # initialize Q
            if any(data.model.data.obstype[subj_inds] .!= 1)
                fill!(q, 0.0)
            end

            n_inds = length(subj_inds)
            ind = 0

            # update the vector l
            for i in subj_inds

                # increment counter
                ind += 1

                # compute q, the transition probability matrix
                if data.model.data.obstype[i] != 1
                    # if panel data, simply grab q from tpm_book
                    q = copy(tpm_book[data.books[2][i, 1]][data.books[2][i, 2]])
                    
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
                        q[r,:] = exp.(q[r,:])
                    end
                end # end-compute q

                # compute the set of possible "states to"
                StatesTo = data.model.data.stateto[i] > 0 ? [data.model.data.stateto[i]] : findall(data.model.emat[i,:] .== 1)

                # reset l_t1
                fill!(l_t1, 0.0)

                for s in 1:S
                    if s ∈ StatesTo
                        for r in 1:S
                            l_t1[s] += q[r,s] * l_t0[r]
                        end
                    end
                end

                # swap l_0t and l_t1
                if ind != n_inds
                    l_t0, l_t1 = l_t1, l_t0
                end
            end

            # log likelihood
            subj_ll=log(sum(l_t1))
        end
        
        # weighted loglikelihood
        ll += subj_ll * data.model.SamplingWeights[subj]
    end

    neg ? -ll : ll
end


                
                # # get the state(s) of origin - Raphael to confirm this is correct
                # StatesFrom = data.model.data.statefrom[i] > 0 ? [data.model.data.statefrom[i]] : findall(data.model.emat[i-1,:] .== 1)

                # # get the state(s) of destination
                # StatesTo = data.model.data.stateto[i] > 0 ? [data.model.data.stateto[i]] : findall(data.model.emat[i,:] .== 1)

                # # forward filtering
                # if data.model.data.obstype[i] == 1 # exact data 

                #     # verify that, when we have an exact observation, there is a single state to which the subject transitions.
                #     if size(StatesTo,1)>1
                #         error("Observation $subj_inds is exact")
                #     end

                #     # contribution of observation i
                #     subj_lik_i = 0.0 # should this be a 0.0?
                    
                #     # add the contribution of each possible statefrom
                #     for sf in StatesFrom
                        
                #         subj_lik_i_sf = 1.0 # should this be 1.0?

                #         if (data.model.data.tstop[i] - data.model.data.tstart[i]) < eps()
                #             subj_lik_i_sf *= survprob(0, data.model.data.tstop[i] - data.model.data.tstart[i], pars, i, data.model.totalhazards[sf], data.model.hazards; give_log = false)                        
                #         end

                #         if sf != data.model.data.stateto[i] # if there is a transition, add hazard
                #             subj_lik_i_sf *= call_haz(data.model.data.tstop[i] - data.model.data.tstart[i], pars[data.model.tmat[sf, data.model.data.stateto[i]]], i, data.model.hazards[data.model.tmat[sf, data.model.data.stateto[i]]]; give_log = false)
                #         end

                #         subj_lik_i += subj_lik_i_sf # is this right?
                #     end

                #     subj_lik *= subj_lik_i

                # else # panel data
                #     subj_lik *= tpm_book[data.books[2][i, 1]][data.books[2][i, 2]][StatesFrom, StatesTo]
                # end
            # end
            
            # subj_ll = log(subj_lik)
        # end
# end


                
                # # get the state(s) of origin - Raphael to confirm this is correct
                # StatesFrom = data.model.data.statefrom[i] > 0 ? [data.model.data.statefrom[i]] : findall(data.model.emat[i-1,:] .== 1)

                # # get the state(s) of destination
                # StatesTo = data.model.data.stateto[i] > 0 ? [data.model.data.stateto[i]] : findall(data.model.emat[i,:] .== 1)

                # # forward filtering
                # if data.model.data.obstype[i] == 1 # exact data 

                #     # verify that, when we have an exact observation, there is a single state to which the subject transitions.
                #     if size(StatesTo,1)>1
                #         error("Observation $subj_inds is exact")
                #     end

                #     # contribution of observation i
                #     subj_lik_i = 0.0 # should this be a 0.0?
                    
                #     # add the contribution of each possible statefrom
                #     for sf in StatesFrom
                        
                #         subj_lik_i_sf = 1.0 # should this be 1.0?

                #         if (data.model.data.tstop[i] - data.model.data.tstart[i]) < eps()
                #             subj_lik_i_sf *= survprob(0, data.model.data.tstop[i] - data.model.data.tstart[i], pars, i, data.model.totalhazards[sf], data.model.hazards; give_log = false)                        
                #         end

                #         if sf != data.model.data.stateto[i] # if there is a transition, add hazard
                #             subj_lik_i_sf *= call_haz(data.model.data.tstop[i] - data.model.data.tstart[i], pars[data.model.tmat[sf, data.model.data.stateto[i]]], i, data.model.hazards[data.model.tmat[sf, data.model.data.stateto[i]]]; give_log = false)
                #         end

                #         subj_lik_i += subj_lik_i_sf # is this right?
                #     end

                #     subj_lik *= subj_lik_i

                # else # panel data
                #     subj_lik *= tpm_book[data.books[2][i, 1]][data.books[2][i, 2]][StatesFrom, StatesTo]
                # end
            # end
            
            # subj_ll = log(subj_lik)
        # end
# end

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

