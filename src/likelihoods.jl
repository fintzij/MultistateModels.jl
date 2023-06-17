########################################################
### Likelihod functions for sample paths, tpms, etc. ###
########################################################

### Exactly observed sample paths ----------------------
"""
    loglik(parameters, path::SamplePath, hazards::Vector{T}, model::MultistateProcess) where T <: _Hazard

Log-likelihood for a single sample path. The sample path object is `path::SamplePath` and contains the subject index and the jump chain.
"""
function loglik(parameters, path::SamplePath, hazards::Vector{T}, model::MultistateProcess) where T <: _Hazard

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
                log_surv_prob += survprob(timespent, timeinstate, parameters, comp_dat_ind, model.totalhazards[scur], hazards; give_log = true, newtime = false)

                # increment log likelihood
                ll += log_surv_prob

                # if event happened, accrue hazard
                if snext != scur
                    ll += call_haz(timeinstate, parameters[model.tmat[scur, snext]], comp_dat_ind, hazards[model.tmat[scur, snext]]; give_log = true, newtime = false)
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
                log_surv_prob += survprob(timespent, timespent + time_R - tcur, parameters, comp_dat_ind, model.totalhazards[scur], hazards; give_log = true, newtime = false)
                
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
    ll = mapreduce(x -> loglik(pars, x, data.model.hazards, data.model), +, data.paths)

    neg ? -ll : ll
end

"""
    loglik(parameters, data::MPanelData; neg = true)

Return sum of (negative) log likelihood for a Markov model fit to panel data. 
"""
function loglik(parameters, data::MPanelData; neg = true) # Raph: work on this

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
    for i in Base.OneTo(nrow(data.model.data))

        if data.model.data.obstype[i] == 1 # panel data

            ll += log(tpm_book[data.books[2][i, 1]][data.books[2][i, 2]][data.model.data.statefrom[i],
             data.model.data.stateto[i]])

        elseif data.model.data.obstype[i] == 2 # exact data

            ll += survprob(0, data.model.data.tstop[i] - data.model.data.tstart[i], parameters, i, data.model.totalhazards[data.model.data.statefrom[i]], hazards; give_log = true, newtime = false)

            if data.model.data.statefrom[i] != data.model.data.stateto[i] # if there is a transition, add log hazard

                ll += call_haz(data.model.data.tstop[i] - data.model.data.tstart[i], parameters[data.model.tmat[data.model.data.statefrom[i], data.model.data.stateto[i]]], i, hazards[data.model.tmat[data.model.data.statefrom[i], data.model.data.stateto[i]]]; give_log = true, newtime = false)
        end
        
    end

    neg ? -ll : ll
end

"""
    loglik(parameters, data::SMPanelData; neg = true)

Return sum of (negative) complete data log-likelihood terms in the Monte Carlo maximum likelihood algorithm for fitting a semi-Markov model to panel data. 
"""
function loglik(parameters, data::SMPanelData; neg = true)

    # nest the model parameters
    pars = VectorOfVectors(parameters, data.model.parameters.elem_ptr)

    # compute the semi-markov log-likelihoods
    ll = 0.0
    for j in Base.OneTo(size(data.paths, 2))
        for i in Base.OneTo(size(data.paths, 1))
            ll += loglik(pars, data.paths[i, j], data.model.hazards, data.model) * data.weights[i,j] / data.totweights[i]
        end
    end

    # return the log-likelihood
    neg ? -ll : ll
end

"""
    loglik(parameters, data::SMPanelData)

Return sum of (negative) complete data log-likelihood terms in the Monte Carlo maximum likelihood algorithm for fitting a semi-Markov model to panel data. 
"""
function loglik!(parameters, logliks::ElasticArray{Float64}, data::SMPanelData)

    # nest the model parameters
    pars = VectorOfVectors(parameters, data.model.parameters.elem_ptr)

    # compute the semi-markov log-likelihoods
    for j in Base.OneTo(size(data.paths, 2))
        for i in Base.OneTo(size(data.paths, 1))
            logliks[i,j] = loglik(pars, data.paths[i, j], data.model.hazards, data.model)
        end
    end
end

