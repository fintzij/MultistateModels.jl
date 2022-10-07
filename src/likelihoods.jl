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

"""
    transitionprobs(parameters, tpm_index::DataFrame, hazards::Vector{_Hazard}, tmat::Matrix{Int64})

Calculate transition probability matrices for a multistate Markov process. 
"""
function transitionprobs(parameters, tpm_index::DataFrame, hazards::Vector{_Hazard}, tmat::Matrix{Int64})

    # initialize transition probability matrix
    Q = zeros(Float64, size(tmat))
    compute_hazards!(Q, parameters, tpm_index, hazards, tmat)

end

########################################################
##################### Wrappers #########################
########################################################

"""
    loglik(parameters, exactdata::ExactData; neg = true) 

Return sum of (negative) log likelihoods for all sample paths. Use mapreduce() to call loglik() and sum the results. Each sample path object is `path::SamplePath` and contains the subject index and the jump chain. 
"""

function loglik(parameters, exactdata::ExactData; neg = true)

    # send each element of samplepaths to loglik
    ll = mapreduce(x -> loglik(parameters, x, exactdata.model), +, exactdata.samplepaths)

    neg ? -ll : ll
end

"""
    loglik(parameters, paneldata::PanelData; neg = true)

Return sum of (negative) log likelihood for panel data. 
"""
function loglik(parameters, paneldata::PanelData; neg = true) 

    # Solve Kolmogorov equations for TPMs
    

    # ll = loglik(parameters, paneldata.data, paneldata.model, paneldata.books)
    # ll = mapreduce(x -> loglik(tpms, inds), +, paneldata.data; dims = )


    neg ? -ll : ll
end





# """
#     loglik(parameters, samplepaths::Array{SamplePath}, model::MultistateModel) 

# Return sum of log likelihoods for all sample paths. Use mapreduce() to call loglik() and sum the results. Each sample path object is `path::SamplePath` and contains the subject index and the jump chain.
# """

# function loglik(parameters, paths::Array{SamplePath}, model::MultistateModel)

#     # send each element of samplepaths to loglik(path::SamplePath, model::MultistateModel) and sum up results
#     return mapreduce(x -> loglik(parameters, x, model), +, paths)
# end