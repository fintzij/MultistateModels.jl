"""
    survprob(lb::Float64, ub::Float64, parameters::Vector{Vector{Float64}}, rowind::Int64, _totalhazard::_TotalHazardTransient, _hazards::Vector{_Hazard})

Return the survival probability over the interval [lb, ub]. 

# Arguments 
- `lb`: start time
- `ub`: end time
- `parameters`: model parameters, a vector of vectors
- `rowind`: row index in data
- `_totalhazard::_TotalHazardTransient`: total hazard from transient state, contains indices for hazards that contribute to the total hazard
- `_hazards::_Hazard`: vector of cause-specific hazards
- `give_log::Bool`: should the log total hazard be returned (default)
"""
function survprob(lb, ub, parameters, rowind, _totalhazard::_TotalHazardTransient, _hazards::Vector{_Hazard}; give_log = true)

    # log total cumulative hazard
    log_survprob = -total_cumulhaz(lb, ub, parameters, rowind, _totalhazard, _hazards; give_log = false)

    # return survival probability or not
    give_log ? log_survprob : exp(log_survprob)
end

"""
    total_cumulhaz(lb, ub, parameters, rowind, _totalhazard::_TotalHazardTransient, _hazards::Vector{_Hazard}; give_log = true)

Return the log-total cumulative hazard out of a transient state over the interval [lb, ub]. 

# Arguments 
- `lb`: start time
- `ub`: end time
- `parameters`: model parameters, a vector of vectors
- `rowind`: row index in data
- `_totalhazard::_TotalHazardTransient`: total hazard from transient state, contains indices for hazards that contribute to the total hazard
- `_hazards::_Hazard`: vector of cause-specific hazards
- `give_log::Bool`: should the log total hazard be returned (default)
"""
function total_cumulhaz(lb, ub, parameters, rowind, _totalhazard::_TotalHazardTransient, _hazards::Vector{_Hazard}; give_log = true)

    # log total cumulative hazard
    log_tot_haz = 
        logsumexp(
            map(x -> 
                call_cumulhaz(
                    lb,
                    ub, 
                    parameters[x],
                    rowind, 
                    _hazards[x];
                    give_log = true
                ), _totalhazard.components
            )
        )
    
    # return the log, or not
    give_log ? log_tot_haz : exp(log_tot_haz)
end

"""
    total_cumulhaz(lb, ub, parameters, rowind::Int64, _totalhazard::_TotalHazardAbsorbing, _hazards::Vector{_Hazard}; give_log = true)

Return zero log-total cumulative hazard over the interval [lb, ub] as the current state is absorbing. 

# Arguments 
- `lb`: start time
- `ub`: end time
- `parameters`: model parameters, a vector of vectors
- `rowind`: row index in data
- `_totalhazard::_TotalHazardAbsorbing`: absorbing state.
- `_hazards::_Hazard`: vector of cause-specific hazards
- `give_log::Bool`: should the log total hazard be returned (default)
"""
function total_cumulhaz(lb, ub, parameters, rowind, _totalhazard::_TotalHazardAbsorbing, _hazards::Vector{_Hazard}; give_log = true)

    # return 0 cumulative hazard
    give_log ? -Inf : 0

end

"""
    call_haz(t, parameters, rowind, _hazard::_Exponential; give_log = true)

Return the exponential cause-specific hazards.
"""
function call_haz(t, parameters, rowind, _hazard::_Exponential; give_log = true)
    give_log ? parameters[1] : exp(parameters[1])
end

"""
    call_cumulhaz(lb, ub, parameters, rowind, _hazard::_Exponential; give_log = true)

Cumulative hazard for the exponential hazards over the interval [lb, ub]. 
"""
function call_cumulhaz(lb, ub, parameters, rowind, _hazard::_Exponential; give_log = true)
    give_log ? parameters[1] + log(ub - lb) : exp(parameters[1] + log(ub - lb))
end

"""
    call_haz(t, parameters, rowind, _hazard::_ExponentialPH; give_log = true)

Return the exponential cause-specific hazards with covariate adjustment.
"""
function call_haz(t, parameters, rowind, _hazard::_ExponentialPH; give_log = true)
    log_haz = dot(parameters, _hazard.data[rowind,:])
    give_log ? log_haz : exp(log_haz)
end

"""
    call_cumulhaz(lb, ub, parameters, rowind, _hazard::_ExponentialPH; give_log = true)

Cumulative hazard for exponential proportional hazards over the interval [lb, ub].
"""
function call_cumulhaz(lb, ub, parameters, rowind, _hazard::_ExponentialPH; give_log = true)
    log_haz = dot(parameters, _hazard.data[rowind,:])
    give_log ? log_haz + log(ub - lb) : exp(log_haz + log(ub - lb))
end

"""
    call_haz(t, parameters, rowind, _hazard::_Weibull; give_log = true)

Return the Weibull cause-specific hazards. No covariate adjustement, parameterized as in Section 2.2.2 of Kalbfleisch and Prentice.
"""
function call_haz(t, parameters, rowind, _hazard::_Weibull; give_log = true)

    # scale and shape
    log_shape = parameters[1]
    log_scale = parameters[2]

    # compute hazard 
    log_haz = 
        log_shape + expm1(log_shape) * log(t) + exp(log_shape) * log_scale 

    give_log ? log_haz : exp(log_haz)
end

"""
    call_cumulhaz(lb, ub, parameters, rowind, _hazard::_Weibull; give_log = true)

Cumulative hazard for Weibulll hazards over the interval [lb, ub].
"""
function call_cumulhaz(lb, ub, parameters, rowind, _hazard::_Weibull; give_log = true)

    # scale and shape
    shape = exp(parameters[1])
    log_scale = parameters[2]

    # cumulative hazard
    log_cumul_haz = 
        log(ub ^ shape - lb ^ shape) + shape * log_scale

    give_log ? log_cumul_haz : exp(log_cumul_haz)
end

"""
    call_haz(t, parameters, rowind, _hazard::_WeibullPH; give_log = true)

Return the Weibull cause-specific proportional hazards. Weibull proportional hazards model parameterized like in the `rstanarm` package in R.
"""
function call_haz(t, parameters, rowind, _hazard::_WeibullPH; give_log = true)

    # scale and shape
    log_shape = parameters[1]

    # compute hazard
    log_haz = 
        log_shape + expm1(log_shape) * log(t) + dot(parameters[2:end], _hazard.data[rowind,:])

    give_log ? log_haz : exp(log_haz)
end

"""
    call_cumulhaz(lb, ub, parameters, rowind, _hazard::_WeibullPH; give_log = true)

Cumulative hazard for Weibulll proportional hazards over the interval [lb, ub].
"""
function call_cumulhaz(lb, ub, parameters, rowind, _hazard::_WeibullPH; give_log = true)

    # scale and shape
    shape = exp(parameters[1])

    # cumulative hazard
    log_cumul_haz = 
        log(ub ^ shape - lb ^ shape) + dot(parameters[2:end], _hazard.data[rowind,:])

    give_log ? log_cumul_haz : exp(log_cumul_haz)
end

"""
    next_state_probs!(ns_probs, scur, ind, model)

Update ns_probs with probabilities of transitioning to each state based on hazards from current state. 

# Arguments 
- ns_probs: vector of probabilities corresponding to each state, modified in place
- t: time at which hazards should be calculated
- scur: current state
- ind: index at complete dataset
- parameters: vector of vectors of model parameters
- hazards: vector of cause-specific hazards
- totalhazards: vector of total hazards
- tmat: transition matrix
"""
function next_state_probs!(ns_probs, t, scur, ind, parameters, hazards, totalhazards, tmat)

    # set ns_probs to zero for impossible transitions
    ns_probs[findall(tmat[scur,:] .== 0.0)] .= 0.0

    # indices for possible destination states
    trans_inds = findall(tmat[scur,:] .!= 0.0)
        
    # calculate log hazards for possible transitions
    ns_probs[trans_inds] = 
        map(x -> call_haz(t, parameters[x], ind, hazards[x]), totalhazards[scur].components)

    # normalize ns_probs
    ns_probs[trans_inds] = 
        softmax(ns_probs[totalhazards[scur].components])
end

"""
    next_state_probs!(ns_probs, scur, ind, model)

Return a vector ns_probs with probabilities of transitioning to each state based on hazards from current state. 

# Arguments 
- t: time at which hazards should be calculated
- scur: current state
- ind: index at complete dataset
- parameters: vector of vectors of model parameters
- hazards: vector of cause-specific hazards
- totalhazards: vector of total hazards
- tmat: transition matrix
"""
function next_state_probs(t, scur, ind, parameters, hazards, totalhazards, tmat)

    # initialize vector of next state transition probabilities
    ns_probs = zeros(size(model.tmat, 2))

    # indices for possible destination states
    trans_inds = findall(tmat[scur,:] .!= 0.0)
        
    # calculate log hazards for possible transitions
    ns_probs[trans_inds] = 
        map(x -> call_haz(t, parameters[x], ind, hazards[x]), totalhazards[scur].components)

    # normalize ns_probs
    ns_probs[trans_inds] = 
        softmax(ns_probs[totalhazards[scur].components])

    # return the next state probabilities
    return ns_probs
end

########################################################
############# multistate markov process ################
###### transition intensities and probabilities ########
########################################################

"""
    compute_hazmat!(Q, parameters, hazards::Vector{_Hazard}, tpm_index::DataFrame, ind::Int64)

Fill in a matrix of transition intensities for a multistate Markov model.
"""
function compute_hazmat!(Q, parameters, hazards::Vector{_Hazard}, tpm_index::DataFrame)

    # compute transition intensities
    for h in eachindex(hazards) 
        Q[hazards[h].statefrom, hazards[h].stateto] = 
            call_haz(
                tpm_index.tstart[1], 
                parameters[h],
                tpm_index.datind[1],
                hazards[h]; 
                give_log = false)
    end

    # set diagonal elements equal to the sum of off-diags
    Q[diagind(Q)] = -sum(Q, dims = 2)
end

"""
    compute_tmat!(P, Q, tpm_index::DataFrame, hazards::Vector{_Hazard}, tmat::Matrix{Int64})

Calculate transition probability matrices for a multistate Markov process. 
"""
function compute_tmat!(P, Q, tpm_index::DataFrame, cache)

    for t in eachindex(P)
        copyto!(P[t], exponential!(Q * tpm_index.tstop[t], ExpMethodGeneric(), cache))
    end  
end


"""
    cumulative_incidence(model::MultistateModel, subj, time_since_entry)

Compute the cumulative incidence for each possible transition as a function of time since state entry. Assumes the starts their observation period at risk and saves cumulative incidence at the supplied vector of times since state entry.
"""
function cumulative_incidence(model::MultistateModel, subj, time_since_entry)

    # grab parameters, hazards and total hazards
    parameters   = model.parameters
    hazards      = model.hazards
    totalhazards = model.totalhazards

    # subject data
    subj_inds = model.subjectindices[subj]
    subj_dat  = view(model.data, subj_inds, :)

    # merge times with left endpoints of subject observation intervals
    subj_times = sort(unique([0.0; time_since_entry]))

    # identify transient states
    transients = findall(isa.(totalhazards, _TotalHazardTransient))

    # identify which transient state to grab for each hazard (as transients[transinds[h]])
    transinds  = reduce(vcat, [i * ones(Int64, length(totalhazards[transients[i]].components)) for i in eachindex(transients)])

    # initialize cumulative incidence
    n_intervals = length(subj_times) - 1
    incidences  = zeros(Float64, n_intervals, length(hazards))
    survprobs   = ones(Float64, n_intervals, length(transients))

    # indices for starting cumulative incidence increments
    interval_inds = map(x -> searchsortedlast(subj_dat.tstart .- minimum(subj_dat.tstart), x), subj_times[Not(end)])

    # compute the survival probabilities to start each interval
    if n_intervals > 1
        for s in eachindex(transients)
            # initialize sprob and identify origin state
            sprob = 1.0
            statefrom = transients[s]

            # compute survival probabilities
            for i in 2:n_intervals
                survprobs[i,s] = sprob * survprob(subj_times[i], subj_times[i+1], parameters, subj_inds[interval_inds[i]], totalhazards[statefrom], hazards; give_log = false)
                sprob = survprobs[i,s]
            end
        end
    end
    
    # compute the cumulative incidence for each transition type
    for h in eachindex(hazards)
        # identify origin state
        statefrom = transients[transinds[h]]

        # compute incidences
        for r in 1:n_intervals
            incidences[r,h] = 
                survprobs[r,transinds[h]] * 
                quadgk(t -> (
                        call_haz(t, parameters[h], subj_inds[interval_inds[r]], hazards[h]; give_log = false) * 
                        survprob(subj_times[r], t, parameters, subj_inds[interval_inds[r]], totalhazards[statefrom], hazards; give_log = false)), 
                        subj_times[r], subj_times[r + 1])[1]
        end        
    end

    # return cumulative incidences
    return cumsum(incidences; dims = 1)
end

"""
    cuminc_discrepancy(model::MultistateModel, subj)

Compute the cumulative incidence for each possible transition as a function of time since state entry. Assumes the starts their observation period at risk and saves cumulative incidence at the supplied vector of times since state entry.
"""
function cuminc_discrepancy(model::MultistateModel, subj)

    # grab parameters, hazards and total hazards
    parameters   = model.parameters
    hazards      = model.hazards
    totalhazards = model.totalhazards

    # grab surrogate parameters, hazards and total hazards
    surr_haz  = model.markovsurrogate[1]
    surr_pars = model.markovsurrogate[2]

    # subject data
    subj_inds = model.subjectindices[subj]
    subj_dat  = view(model.data, subj_inds, :)

    # merge times with left endpoints of subject observation intervals
    subj_times = 
        sort([subj_dat.tstart; subj_dat.tstop[end];
              2/3 .* subj_dat.tstart + 1/3 .* subj_dat.tstop;
              1/3 .* subj_dat.tstart + 2/3 .* subj_dat.tstop]) .- subj_dat.tstart[1]
    
    # identify transient states
    transients = findall(isa.(totalhazards, _TotalHazardTransient))

    # identify the transient states for each hazard
    transinds  = reduce(vcat, [i * ones(Int64, length(totalhazards[transients[i]].components)) for i in eachindex(transients)])

    # initialize cumulative incidence
    n_intervals      = length(subj_times) - 1
    survprobs_target = ones(Float64, n_intervals, length(transients))
    survprobs_surr   = ones(Float64, n_intervals, length(transients))
    discrepancies    = zeros(Float64, length(hazards))

    # indices for starting cumulative incidence increments
    interval_inds = map(x -> searchsortedlast(subj_dat.tstart .- minimum(subj_dat.tstart), x), subj_times[Not(end)])

    # compute the survival probabilities to start each interval
    for s in eachindex(transients)
        # initialize survival probabilities
        sprob_target = 1.0
        sprob_surr   = 1.0

        # get the origin state
        statefrom = transients[s]

        # loop through intervals
        for i in 2:n_intervals
            # compute survival probability
            survprobs_target[i,s] = 
                sprob_target * survprob(subj_times[i], subj_times[i+1], parameters, subj_inds[interval_inds[i]], totalhazards[statefrom], hazards; give_log = false)

            survprobs_surr[i,s] = 
                sprob_surr * survprob(subj_times[i], subj_times[i+1], surr_pars, subj_inds[interval_inds[i]], totalhazards[statefrom], surr_haz; give_log = false)

            # increment survival probability
            sprob_target = survprobs_target[i,s]
            sprob_surr   = survprobs_surr[i,s]
        end
    end
    
    # compute the discrepancy in cumulative incidence for each transition type
    for h in eachindex(hazards)

        # get the origin state
        statefrom = transients[transinds[h]]

        for r in 1:n_intervals
            discrepancies[h] += 
                (quadgk(t -> (
                        survprobs_target[r, transinds[h]] *
                        call_haz(t, parameters[h], subj_inds[interval_inds[r]], hazards[h]; give_log = false) * 
                        survprob(subj_times[r], t, parameters, subj_inds[interval_inds[r]], totalhazards[statefrom], hazards; give_log = false)), 
                    subj_times[r], subj_times[r + 1])[1] - 
                    quadgk(t -> (
                        survprobs_surr[r, transinds[h]] *
                        call_haz(t, surr_pars[h], subj_inds[interval_inds[r]], surr_haz[h]; give_log = false) * 
                        survprob(subj_times[r], t, surr_pars, subj_inds[interval_inds[r]], totalhazards[statefrom], surr_haz; give_log = false)), 
                    subj_times[r], subj_times[r + 1])[1])^2                        
        end        
    end

    # return cumulative incidences
    return discrepancies
end

"""
    cuminc_discrepancy(model::MultistateModel, subj, statefrom)

Compute the cumulative incidence for each possible transition as a function of time since state entry. Assumes the starts their observation period at risk and saves cumulative incidence at the supplied vector of times since state entry.
"""
function cuminc_discrepancy(model::MultistateModel, subj, statefrom)

    # grab parameters, hazards and total hazards
    parameters   = model.parameters
    hazards      = model.hazards
    totalhazards = model.totalhazards

    # grab surrogate parameters, hazards and total hazards
    surr_haz  = model.markovsurrogate[1]
    surr_pars = model.markovsurrogate[2]

    # subject data
    subj_inds = model.subjectindices[subj]
    subj_dat  = view(model.data, subj_inds, :)

    # merge times with left endpoints of subject observation intervals
    subj_times = 
        sort([subj_dat.tstart; subj_dat.tstop[end];
              2/3 .* subj_dat.tstart + 1/3 .* subj_dat.tstop;
              1/3 .* subj_dat.tstart + 2/3 .* subj_dat.tstop]) .- subj_dat.tstart[1]

    # initialize cumulative incidence
    n_intervals      = length(subj_times) - 1
    survprobs_target = ones(Float64, n_intervals)
    survprobs_surr   = ones(Float64, n_intervals)
    hazinds          = totalhazards[statefrom].components
    discrepancies    = zeros(Float64, length(hazinds))

    # indices for starting cumulative incidence increments
    interval_inds = map(x -> searchsortedlast(subj_dat.tstart .- minimum(subj_dat.tstart), x), subj_times[Not(end)])

    # compute the survival probabilities to start each interval
    # initialize survival probabilities
    sprob_target = 1.0
    sprob_surr   = 1.0

    # loop through intervals
    for i in 2:n_intervals
        # compute survival probability
        survprobs_target[i] = 
            sprob_target * survprob(subj_times[i], subj_times[i+1], parameters, subj_inds[interval_inds[i]], totalhazards[statefrom], hazards; give_log = false)

        survprobs_surr[i] = 
            sprob_surr * survprob(subj_times[i], subj_times[i+1], surr_pars, subj_inds[interval_inds[i]], totalhazards[statefrom], surr_haz; give_log = false)

        # increment survival probability
        sprob_target = survprobs_target[i]
        sprob_surr   = survprobs_surr[i]
    end
    
    # compute the discrepancy in cumulative incidence for each transition type
    for h in eachindex(hazinds)
        for r in 1:n_intervals
            discrepancies[h] += 
                (quadgk(t -> (
                        survprobs_target[r] *
                        call_haz(t, parameters[hazinds[h]], subj_inds[interval_inds[r]], hazards[hazinds[h]]; give_log = false) * 
                        survprob(subj_times[r], t, parameters, subj_inds[interval_inds[r]], totalhazards[statefrom], hazards; give_log = false)), 
                    subj_times[r], subj_times[r + 1])[1] - 
                    quadgk(t -> (
                        survprobs_surr[r] *
                        call_haz(t, surr_pars[hazinds[h]], subj_inds[interval_inds[r]], surr_haz[hazinds[h]]; give_log = false) * 
                        survprob(subj_times[r], t, surr_pars, subj_inds[interval_inds[r]], totalhazards[statefrom], surr_haz; give_log = false)), 
                    subj_times[r], subj_times[r + 1])[1])^2                        
        end        
    end

    # return cumulative incidences
    return discrepancies
end
