########################################################
### Likelihod functions for sample paths, tpms, etc. ###
########################################################

### Exactly observed sample paths ----------------------
"""
    loglik(path::SamplePath, model::MultistateModel) 

Log-likelihood for a single sample path. The sample path object is `path::SamplePath` and contains the subject index and the jump chain.
"""
function loglik(parameters, path::SamplePath, model::MultistateModel)

    # initialize log likelihood
    ll = 0.0

    # number of jumps/left endpoints
    n_intervals = length(path.times) - 1

    # subject data
    subj_inds = model.subjectindices[path.subj]
    subj_dat  = view(model.data, subj_inds, :)

    # current index
    subj_dat_ind = 1 # row in subject data
    comp_dat_ind = subj_inds[subj_dat_ind] # index in complete data

    # data time interval
    time_L = subj_dat.tstart[1]
    time_R = subj_dat.tstop[1]

    # recurse through the sample path
    for i in Base.OneTo(n_intervals)

        # current and next need this?
        scur  = path.states[i]
        snext = path.states[i+1]

        # keep accruing log-likelihood contributions for sojourn
        keep_going = isa(model.totalhazards[scur], _TotalHazardTransient)

        # times in the jump chain (clock forward)
        # gets reset each time below i gets incremented
        tcur  = path.times[i]
        tstop = path.times[i+1]

        # time in state (clock reset)
        timespent   = 0.0
        timeinstate = tstop - tcur 

        # initialize survival probability
        log_surv_prob = 0.0

        # accumulate log likelihood
        while keep_going
            
            if tstop <= time_R
                # event happens in (time_L, time_R]
                # accumulate log(Pr(T ≥ timeinstate | T ≥ timespent))
                log_surv_prob += survprob(timespent, timeinstate, parameters, comp_dat_ind, model.totalhazards[scur], model.hazards; give_log = true)

                # increment log likelihood
                ll += log_surv_prob

                # if event happened, accrue hazard
                if snext != scur
                    ll += call_haz(timeinstate, parameters[model.tmat[scur, snext]], comp_dat_ind, model.hazards[model.tmat[scur, snext]]; give_log = true)
                end

                # break out of the while loop
                keep_going = false

            else
                # event doesn't hapen in (time_L, time_R]
                # accumulate log-survival
                # accumulate log(Pr(T ≥ time_R | T ≥ timespent))
                log_surv_prob += survprob(timespent, timespent + time_R - tcur, parameters, comp_dat_ind, model.totalhazards[scur], model.hazards; give_log = true)
                
                # increment timespent
                timespent += time_R - tcur

                # increment current time
                tcur = time_R

                # increment row index in subj_dat
                subj_dat_ind += 1
                comp_dat_ind += 1

                # increment time_L, time_R
                time_L = subj_dat.tstart[subj_dat_ind]
                time_R = subj_dat.tstop[subj_dat_ind]
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
    ll = mapreduce(x -> loglik(pars, x, data.model), +, data.paths)

    neg ? -ll : ll
end

"""
    loglik(parameters, data::MPanelData; neg = true)

Return sum of (negative) log likelihood for a Markov model fit to panel data. 
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
    for i in Base.OneTo(nrow(data.model.data))
        ll += 
            log(tpm_book[data.books[2][i, 1]][data.books[2][i, 2]][data.model.data.statefrom[i],
             data.model.data.stateto[i]])
    end

    neg ? -ll : ll
end

"""
    loglik(parameters, data::SMPanelData; neg = true)

Return sum of (negative) log-likelihood for a semi-Markov model fit to panel data. 
"""
function loglik(parameters, data::SMPanelData; neg = true)

    # nest the model parameters
    pars = VectorOfVectors(parameters, data.model.parameters.elem_ptr)

    # find the surrogate parameters for path proposals that minimize the discrepancy between state occupancy probs
    surrogate_pars = optimize_surrogate(pars, data.model)

end