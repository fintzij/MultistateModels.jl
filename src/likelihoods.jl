"""
    loglik(path::SamplePath, model::MultistateModel) 

Log-likelihood for a single sample path. The sample path object is `path::SamplePath` and contains the subject index and the jump chain.
"""
function loglik(path::SamplePath, model::MultistateModel)

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
                log_surv_prob -= cumulhaz(model.totalhazards[scur], model.hazards, timespent, timeinstate, comp_dat_ind)

                # increment log likelihood
                ll += log_surv_prob

                # if event happened, accrue hazard
                if snext != scur
                    ll += call_haz(timeinstate, comp_dat_ind, model.hazards[model.tmat[scur, snext]])
                end

                # break out of the while loop
                keep_going = false

            else
                # event doesn't hapen in (time_L, time_R]
                # accumulate log-survival
                # accumulate log(Pr(T ≥ time_R | T ≥ timespent))
                log_surv_prob -= cumulhaz(model.totalhazards[scur], model.hazards, timespent, timespent + time_R - tcur, comp_dat_ind)

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


