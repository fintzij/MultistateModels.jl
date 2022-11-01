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
    cumulative_incidence(model::MultistateModel, parameters, hazards, subj, times)

Compute the cumulative incidence for each possible transition as a function of time since state entry. Assumes the starts their observation period at risk and saves cumulative incidence at the supplied vector of times.
"""
# cumulative_incidence(model::MultistateModel, parameters, hazards, totalhazards, subj, times)

#     # subject data
#     subj_inds = model.subjectindices[subj]
#     subj_dat  = view(model.data, subj_inds, :)

#     # initialize cumulative incidence
#     incidences = zeros(Float64, length(times), length(hazards))

#     # indices for starting cumulative incidence increments
#     cuminc_inds = map(x -> searchsortedlast(subj_dat.tstart, x), times)

#     # cumulative incidence to start each interval
#     cuminc_inits = zeros(Float64, nrow(subj_dat), length(hazards))
    
#     # compute the cumulative incidence for each transition type
#     for h in eachindex(transients)

#         # compute the cumulative incidence prior to the start of each interval
#         if nrow(subj_data) > 1
#             for r in 2:nrow(subj_dat)
#                 cuminc_inits[r,h] = 
#                     quadgk(t -> call_haz(t, parameters[h], subj_inds[r-1], totalhazards[h], hazards; give_log = false) * survprob(subj_dat.tstart[r-1], subj_dat.tstart[r-1] + t, parameters[h], subj_inds[r - 1], totalhazards[h], hazards; give_log = false), subj_dat.tstart[r-1], subj_dat.tstop[r-1])
#             end
#         end
        
#         # initialize row index
#         row = 1
#         ind = subj_inds[row]

#         # set current state
#         scur = hazards[h].statefrom

#         # tcur and tstop
#         tcur  = subj_dat.tstart[1]
#         tstop = subj_dat.tstop[1]

#         # initialize time in state and cumulative incidence
#         timeinstate = 0.0
#         cuminc = 0.0

        
#     end
# end