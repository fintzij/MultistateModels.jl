"""
    loglik_path_OLD(parameters, path::SamplePath, hazards::Vector{<:_Hazard}, model::MultistateProcess) 

Log-likelihood for a single sample path. The sample path object is `path::SamplePath` and contains the subject index and the jump chain.
"""
function loglik_path_OLD(parameters, path, hazards, model) 

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
        keep_going = isa(model.totalhazards[scur], MultistateModels._TotalHazardTransient)

        # times in the jump chain (clock forward)
        # gets reset each time as i gets incremented
        tcur  = path.times[i]   # current time
        tstop = path.times[i+1] # time of next jump

        # time in state (clock reset)
        timespent   = 0.0   # accumulates occupancy time
        timeinstate = tstop - tcur # sojourn time in the jump chain

        # initialize survival probability
        log_surv_prob = 0.0

        # accumulate log likelihood
        while keep_going

            if tstop <= time_R
                # event happens in (time_L, time_R]
                # accumulate log(Pr(T ≥ timeinstate | T ≥ timespent))
                log_surv_prob += MultistateModels.survprob(timespent, timeinstate, parameters, comp_dat_ind, model.totalhazards[scur], hazards; give_log = true)

                # increment log likelihood
                ll += log_surv_prob

                # if event happened, accrue hazard
                if snext != scur
                    ll += MultistateModels.call_haz(timeinstate, parameters[model.tmat[scur, snext]], comp_dat_ind, hazards[model.tmat[scur, snext]]; give_log = true)
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
                log_surv_prob += MultistateModels.survprob(timespent, timespent + time_R - tcur, parameters, comp_dat_ind, model.totalhazards[scur], hazards; give_log = true)
                
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