"""
    survprob(lb, ub, parameters, rowind, _totalhazard::_TotalHazardTransient, _hazards::Vector{<:_Hazard}; give_log = true) 

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
function survprob(lb, ub, parameters, rowind, _totalhazard::_TotalHazardTransient, _hazards; give_log = true) 

    # log total cumulative hazard
    log_survprob = -total_cumulhaz(lb, ub, parameters, rowind, _totalhazard, _hazards; give_log = false)

    # return survival probability or not
    give_log ? log_survprob : exp(log_survprob)
end

"""
    total_cumulhaz(lb, ub, parameters, rowind, _totalhazard::_TotalHazardTransient, _hazards::Vector{<:_Hazard}; give_log = true) 

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
function total_cumulhaz(lb, ub, parameters, rowind, _totalhazard::_TotalHazardTransient, _hazards; give_log = true) 

    # log total cumulative hazard
    tot_haz = 0.0

    for x in _totalhazard.components
        tot_haz += call_cumulhaz(
                    lb,
                    ub, 
                    parameters[x],
                    rowind, 
                    _hazards[x];
                    give_log = false)
    end
    
    # return the log, or not
    give_log ? log(tot_haz) : tot_haz
end

"""
    total_cumulhaz(lb, ub, parameters, rowind, _totalhazard::_TotalHazardAbsorbing, _hazards::Vector{<:_Hazard}; give_log = true) 

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
function total_cumulhaz(lb, ub, parameters, rowind, _totalhazard::_TotalHazardAbsorbing, _hazards; give_log = true) 

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

Return the Weibull cause-specific hazards. No covariate adjustement. See rstanarm R package for parameterization.
"""
function call_haz(t, parameters, rowind, _hazard::_Weibull; give_log = true)

    # scale and shape
    log_shape = parameters[1]
    log_scale = parameters[2]

    # compute hazard 
    log_haz = log_scale + log_shape + expm1(log_shape) * log(t) 

    give_log ? log_haz : exp(log_haz)
end

"""
    call_cumulhaz(lb, ub, parameters, rowind, _hazard::_Weibull; give_log = true)

Cumulative hazard for Weibull hazards over the interval [lb, ub]. See rstanarm R package for parameterization.
"""
function call_cumulhaz(lb, ub, parameters, rowind, _hazard::_Weibull; give_log = true)

    # scale and shape
    shape = exp(parameters[1])
    log_scale = parameters[2]

    # cumulative hazard
    log_cumul_haz = log(ub ^ shape - lb ^ shape) + log_scale

    give_log ? log_cumul_haz : exp(log_cumul_haz)
end

"""
    call_haz(t, parameters, rowind, _hazard::_WeibullPH; give_log = true)

Return the Weibull cause-specific proportional hazards.
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

Cumulative hazard for Weibull proportional hazards over the interval [lb, ub].
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
    call_haz(t, parameters, rowind, _hazard::_Gompertz; give_log = true)

Return the Gompertz cause-specific hazards. No covariate adjustement.
"""
function call_haz(t, parameters, rowind, _hazard::_Gompertz; give_log = true)

    # compute hazard 
    log_haz = parameters[2] + parameters[1] + exp(parameters[1]) * t 

    give_log ? log_haz : exp(log_haz)
end

"""
    call_cumulhaz(lb, ub, parameters, rowind, _hazard::_Gompertz; give_log = true)

Cumulative hazard for Gompertz hazards over the interval [lb, ub].
"""
function call_cumulhaz(lb, ub, parameters, rowind, _hazard::_Gompertz; give_log = true)

    shape = exp(parameters[1])
    
    # cumulative hazard
    log_cumul_haz = parameters[2] + log(exp(ub * shape) - exp(lb * shape))
    
    give_log ? log_cumul_haz : exp(log_cumul_haz)
end

"""
    call_haz(t, parameters, rowind, _hazard::_GompertzPH; give_log = true)

Return the Gompertz cause-specific proportional hazards.
"""
function call_haz(t, parameters, rowind, _hazard::_GompertzPH; give_log = true)

    # compute hazard
    log_haz = dot(parameters[2:end], _hazard.data[rowind,:]) + parameters[1] + exp(parameters[1]) * t 

    give_log ? log_haz : exp(log_haz)
end

"""
    call_cumulhaz(lb, ub, parameters, rowind, _hazard::_GompertzPH; give_log = true)

Cumulative hazard for Gompertz proportional hazards over the interval [lb, ub].
"""
function call_cumulhaz(lb, ub, parameters, rowind, _hazard::_GompertzPH; give_log = true)

    shape = exp(parameters[1])
    
    # cumulative hazard
    log_cumul_haz = dot(parameters[2:end], _hazard.data[rowind,:]) + log(exp(ub * shape) - exp(lb * shape))

    give_log ? log_cumul_haz : exp(log_cumul_haz)
end

"""
    call_haz(t, parameters, rowind, _hazard::_Spline; give_log = true)

Return the spline cause-specific hazards.
"""
function call_haz(t, parameters, rowind, _hazard::_Spline; give_log = true)

    # compute the hazard
    haz = (_hazard.riskperiod[1] < t < _hazard.riskperiod[2]) ? _hazard.hazsp(t) : 0.0
    # haz = _hazard.hazsp(clamp(t, _hazard.riskperiod[1], _hazard.riskperiod[2]))

    # return the log hazard
    give_log ? log(haz) : haz
end

"""
    call_cumulhaz(lb, ub, parameters, rowind, _hazard::_Spline, give_log = true)

Return the spline cause-specific cumulative hazards over the interval [lb,ub].
"""
function call_cumulhaz(lb, ub, parameters, rowind, _hazard::_Spline; give_log = true)

    # clamp lb and ub to the risk period
    riskstart = _hazard.riskperiod[1]
    riskend   = _hazard.riskperiod[2]

    l = clamp(lb, riskstart, riskend)
    u = clamp(ub, riskstart, riskend)

    # get bounds
    sp_bounds = BSplineKit.boundaries(_hazard.hazsp.spline.basis)

    if l == u
        # no cumulative hazard accrued
        chaz = 0.0

    elseif (u <= sp_bounds[1]) || (l >= sp_bounds[2])
        # only extrapolation
        if _hazard.hazsp.method == BSplineKit.Linear()
            # trapezoid
            chaz = (u - l) * mean([_hazard.hazsp(l), _hazard.hazsp(u)]) 

        elseif _hazard.hazsp.method == BSplineKit.Flat()
            # rectangle
            chaz = (u - l) * ((u <= sp_bounds[1]) ? _hazard.hazsp(l) : _hazard.hazsp(u))
        end

    elseif ((sp_bounds[1] < l < sp_bounds[2]) && (sp_bounds[1] < u < sp_bounds[2]))
        # contributions in the initial extrapolation range cancel out
        chaz = _hazard.chazsp(u) - _hazard.chazsp(l)

    else
        # cumulative hazard at l
        l_chaz = 0.0

        if _hazard.hazsp.method == BSplineKit.Linear()
            # trapezoids
            l_chaz += l < sp_bounds[1] ? 
                        (l - riskstart) * mean(_hazard.hazsp.([l, riskstart])) : 
                        (sp_bounds[1] - riskstart) * mean(_hazard.hazsp.([sp_bounds[1], riskstart]))

        elseif _hazard.hazsp.method == BSplineKit.Flat()
            # rectangles
            l_chaz += l < sp_bounds[1] ? 
                        (l - riskstart) * _hazard.hazsp(l) : 
                        (sp_bounds[1] - riskstart) * _hazard.hazsp(sp_bounds[1]) 
        end

        if l >= sp_bounds[1]
            l_chaz += _hazard.chazsp(l)
        end

        # cumulative hazard at u
        u_chaz = 0.0

        if _hazard.hazsp.method == BSplineKit.Linear()
            u_chaz += (sp_bounds[1] - riskstart) * mean(_hazard.hazsp.([sp_bounds[1], riskstart]))

        elseif _hazard.hazsp.method == BSplineKit.Flat()
            u_chaz += (sp_bounds[1] - riskstart) * _hazard.hazsp(sp_bounds[1]) 
        end

        if u <= sp_bounds[2]
            u_chaz += _hazard.chazsp(u)

        elseif u > sp_bounds[2]
            u_chaz += _hazard.chazsp(sp_bounds[2]) 

            if _hazard.hazsp.method == BSplineKit.Linear()
                u_chaz += (u - sp_bounds[2]) * mean(_hazard.hazsp.([u, sp_bounds[2]])) 
                
            elseif _hazard.hazsp.method == BSplineKit.Flat()
                u_chaz += (u - sp_bounds[2]) * _hazard.hazsp(sp_bounds[2])
            end
        end

        chaz = u_chaz - l_chaz
    end 

    # return the log hazard
    give_log ? log(chaz) : chaz
end

"""
    call_haz(t, parameters, rowind, _hazard::_SplinePH; give_log = true)

Return the spline cause-specific hazards.
"""
function call_haz(t, parameters, rowind, _hazard::_SplinePH; give_log = true)

    # compute the hazard
    haz = (_hazard.riskperiod[1] < t < _hazard.riskperiod[2]) ? _hazard.hazsp(t) : 0.0
    # haz = _hazard.hazsp(clamp(t, _hazard.riskperiod[1], _hazard.riskperiod[2]))

    # compute the log hazard
    loghaz = log(haz) + dot(_hazard.data[rowind, :], parameters[Not(1:_hazard.nbasis)])

    # return the log hazard
    give_log ? loghaz : exp(loghaz)
end

"""
    call_cumulhaz(lb, ub, parameters, rowind, _hazard::_SplinePH; give_log = true)

Return the spline cause-specific cumulative hazards over the interval [lb,ub].
"""
function call_cumulhaz(lb, ub, parameters, rowind, _hazard::_SplinePH; give_log = true)

    # clamp lb and ub to the risk period
    riskstart = _hazard.riskperiod[1]
    riskend   = _hazard.riskperiod[2]

    l = clamp(lb, riskstart, riskend)
    u = clamp(ub, riskstart, riskend)

    # get bounds
    sp_bounds = BSplineKit.boundaries(_hazard.hazsp.spline.basis)

    if l == u
        # no cumulative hazard accrued
        chaz = 0.0

    elseif (u <= sp_bounds[1]) || (l >= sp_bounds[2])
        # only extrapolation
        if _hazard.hazsp.method == BSplineKit.Linear()
            # trapezoid
            chaz = (u - l) * mean([_hazard.hazsp(l), _hazard.hazsp(u)]) 

        elseif _hazard.hazsp.method == BSplineKit.Flat()
            # rectangle
            chaz = (u - l) * ((u <= sp_bounds[1]) ? _hazard.hazsp(l) : _hazard.hazsp(u))
        end

    elseif ((sp_bounds[1] < l < sp_bounds[2]) && (sp_bounds[1] < u < sp_bounds[2]))
        # contributions in the initial extrapolation range cancel out
        chaz = _hazard.chazsp(u) - _hazard.chazsp(l)

    else
        # cumulative hazard at l
        l_chaz = 0.0

        if _hazard.hazsp.method == BSplineKit.Linear()
            # trapezoids
            l_chaz += l < sp_bounds[1] ? 
                        (l - riskstart) * mean(_hazard.hazsp.([l, riskstart])) : 
                        (sp_bounds[1] - riskstart) * mean(_hazard.hazsp.([sp_bounds[1], riskstart]))

        elseif _hazard.hazsp.method == BSplineKit.Flat()
            # rectangles
            l_chaz += l < sp_bounds[1] ? 
                        (l - riskstart) * _hazard.hazsp(l) : 
                        (sp_bounds[1] - riskstart) * _hazard.hazsp(sp_bounds[1]) 
        end

        if l >= sp_bounds[1]
            l_chaz += _hazard.chazsp(l)
        end

        # cumulative hazard at u
        u_chaz = 0.0

        if _hazard.hazsp.method == BSplineKit.Linear()
            u_chaz += (sp_bounds[1] - riskstart) * mean(_hazard.hazsp.([sp_bounds[1], riskstart]))

        elseif _hazard.hazsp.method == BSplineKit.Flat()
            u_chaz += (sp_bounds[1] - riskstart) * _hazard.hazsp(sp_bounds[1]) 
        end

        if u <= sp_bounds[2]
            u_chaz += _hazard.chazsp(u)

        elseif u > sp_bounds[2]
            u_chaz += _hazard.chazsp(sp_bounds[2]) 

            if _hazard.hazsp.method == BSplineKit.Linear()
                u_chaz += (u - sp_bounds[2]) * mean(_hazard.hazsp.([u, sp_bounds[2]])) 
                
            elseif _hazard.hazsp.method == BSplineKit.Flat()
                u_chaz += (u - sp_bounds[2]) * _hazard.hazsp(sp_bounds[2])
            end
        end

        chaz = u_chaz - l_chaz
    end 

    # log cumulative hazard
    logchaz = log(chaz) + dot(_hazard.data[rowind, :], parameters[Not(1:_hazard.nbasis)])

    # return the log hazard
    give_log ? logchaz : exp(logchaz)
end

"""
    next_state_probs(t, scur, ind, parameters, hazards, totalhazards, tmat)

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
    ns_probs = zeros(size(tmat, 2))

    # indices for possible destination states
    trans_inds = findall(tmat[scur,:] .!= 0.0)

    if length(trans_inds) == 1
        ns_probs[trans_inds] .= 1.0
    else
        ns_probs[trans_inds] = softmax(map(x -> call_haz(t, parameters[x], ind, hazards[x]), totalhazards[scur].components))
    end

    # catch for numerical instabilities (weird edge case)
    if all(isnan.(ns_probs[trans_inds]))
        ns_probs[trans_inds] .= 1 / length(trans_inds)
    elseif any(isnan.(ns_probs[trans_inds]))
        pisnan = findall(isnan.(ns_probs[trans_inds]))
        if length(pisnan) == 1
            ns_probs[trans_inds][pisnan] = 1 - sum(ns_probs[trans_inds][Not(pisnan)])
        else
            ns_probs[trans_inds][pisnan] .= 1 - sum(ns_probs[trans_inds][Not(pisnan)])/length(pisnan)
        end        
    end

    # return the next state probabilities
    return ns_probs
end

########################################################
############# multistate markov process ################
###### transition intensities and probabilities ########
########################################################

"""
    compute_hazmat!(Q, parameters, hazards::Vector{T}, tpm_index::DataFrame) where T <: _Hazard

Fill in a matrix of transition intensities for a multistate Markov model.
"""
function compute_hazmat!(Q, parameters, hazards::Vector{T}, tpm_index::DataFrame) where T <: _Hazard

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
    compute_tmat!(P, Q, tpm_index::DataFrame, cache)

Calculate transition probability matrices for a multistate Markov process. 
"""
function compute_tmat!(P, Q, tpm_index::DataFrame, cache)

    for t in eachindex(P)
        copyto!(P[t], exponential!(Q * tpm_index.tstop[t], ExpMethodGeneric(), cache))
    end  
end


"""
    cumulative_incidence(t, model::MultistateProcess, subj::Int64=1)

Compute the cumulative incidence for each possible transition as a function of time since state entry. Assumes the subject starts their observation period at risk and saves cumulative incidence at the supplied vector of times, t.
"""
function cumulative_incidence(t, model::MultistateProcess, subj::Int64=1)

    # grab parameters, hazards and total hazards
    parameters   = model.parameters
    hazards      = model.hazards
    totalhazards = model.totalhazards

    # subject data
    subj_inds = model.subjectindices[subj]
    subj_dat  = view(model.data, subj_inds, :)

    # merge times with left endpoints of subject observation intervals
    subj_times = sort(unique([0.0; t]))

    # identify transient states
    transients = findall(isa.(totalhazards, _TotalHazardTransient))

    # identify which transient state to grab for each hazard (as transients[trans_inds[h]])
    trans_inds  = reduce(vcat, [i * ones(Int64, length(totalhazards[transients[i]].components)) for i in eachindex(transients)])

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
                survprobs[i,s] = sprob * survprob(subj_times[i-1], subj_times[i], parameters, subj_inds[interval_inds[i-1]], totalhazards[statefrom], hazards; give_log = false)
                sprob = survprobs[i,s]
            end
        end
    end
    
    # compute the cumulative incidence for each transition type
    for h in eachindex(hazards)
        # identify origin state
        statefrom = transients[trans_inds[h]]

        # compute incidences
        for r in 1:n_intervals
            incidences[r,h] = 
                survprobs[r,trans_inds[h]] * 
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
    cumulative_incidence(t, model::MultistateProcess, statefrom, subj::Int64=1)

Compute the cumulative incidence for each possible transition originating in `statefrom` as a function of time since state entry. Assumes the subject starts their observation period at risk and saves cumulative incidence at the supplied vector of times since state entry. This function is used internally.
"""
function cumulative_incidence(t, model::MultistateProcess, parameters, statefrom, subj::Int64=1)

    # get hazards
    hazards = model.hazards

    # get total hazards
    totalhazards = model.totalhazards

    # return zero if starting from absorbing state
    if isa(totalhazards[statefrom], _TotalHazardAbsorbing)
        return zeros(length(t))
    end

    # subject data
    subj_inds = model.subjectindices[subj]
    subj_dat  = view(model.data, subj_inds, :)

    # merge times with left endpoints of subject observation intervals
    subj_times = sort(unique([0.0; t]))

    # initialize cumulative incidence
    n_intervals = length(subj_times) - 1
    hazinds     = totalhazards[statefrom].components
    incidences  = zeros(Float64, n_intervals, length(hazinds))
    survprobs   = ones(Float64, n_intervals)

    # indices for starting cumulative incidence increments
    interval_inds = map(x -> searchsortedlast(subj_dat.tstart .- minimum(subj_dat.tstart), x), subj_times[Not(end)])

    # compute the survival probabilities to start each interval
    if n_intervals > 1

        # initialize sprob
        sprob = 1.0

        # compute survival probabilities
        for i in 2:n_intervals
            survprobs[i] = sprob * survprob(subj_times[i-1], subj_times[i], parameters, subj_inds[interval_inds[i-1]], totalhazards[statefrom], hazards; give_log = false)
            sprob = survprobs[i]
        end
    end
    
    # compute the cumulative incidence for each transition type
    for h in eachindex(hazinds)
        for r in 1:n_intervals
            incidences[r,h] = 
                survprobs[r] * 
                quadgk(t -> (
                        call_haz(t, parameters[hazinds[h]], subj_inds[interval_inds[r]], hazards[hazinds[h]]; give_log = false) * 
                        survprob(subj_times[r], t, parameters, subj_inds[interval_inds[r]], totalhazards[statefrom], hazards; give_log = false)), 
                        subj_times[r], subj_times[r + 1])[1]
        end        
    end

    # return cumulative incidences
    return cumsum(incidences; dims = 1)
end

"""
    compute_hazard(t, model::MultistateProcess, hazard::Symbol)

Compute the hazard at times t. 

# Arguments
- t: time or vector of times. 
- model: MultistateProcess object. 
- hazard: Symbol specifying the hazard, e.g., :h12 for the hazard for transitioning from state 1 to state 2. 
- subj: subject id. 
"""
function compute_hazard(t, model::MultistateProcess, hazard::Symbol, subj::Int64 = 1)

    # get hazard index
    hazind = model.hazkeys[hazard]

    # compute hazards
    hazards = zeros(Float64, length(t))
    for s in eachindex(t)
        # get row index
        rowind = findlast((model.data.id .== subj) .& (model.data.tstart .<= t[s]))

        # compute hazard
        hazards[s] = call_haz(t[s], model.parameters[hazind], rowind, model.hazards[hazind]; give_log = false)
    end

    # return hazards
    return hazards
end

"""
    compute_cumulative_hazard(tstart, tstop, model::MultistateProcess, hazard::Symbol, subj::Int64=1)

Compute the cumulative hazard over [tstart,tstop]. 

# Arguments
- tstart: starting times
- tstop: stopping times
- model: MultistateProcess object. 
- hazard: Symbol specifying the hazard, e.g., :h12 for the hazard for transitioning from state 1 to state 2. 
- subj: subject id. 
"""
function compute_cumulative_hazard(tstart, tstop, model::MultistateProcess, hazard::Symbol, subj::Int64 = 1)

    # check bounds
    if (length(tstart) == length(tstop))
        # nothing to do
    elseif (length(tstart) == 1) & (length(tstop) != 1)
        tstart = rep(tstart, length(tstart))
    elseif (length(tstart) != 1) & (length(tstop) == 1)
        tstop = rep(tstop, length(tstart))
    else
        error("Lengths of tstart and tstop are not compatible.")
    end

    # get hazard index
    hazind = model.hazkeys[hazard]

    # compute hazards
    cumulative_hazards = zeros(Float64, length(tstart))
    for s in eachindex(tstart)

        # find times between tstart and tstop
        times = [tstart[s]; model.data.tstart[findall((model.data.id .== subj) .& (model.data.tstart .> tstart[s]) .& (model.data.tstart .< tstop[s]))]; tstop[s]]

        # initialize cumulative hazard
        chaz = 0.0

        # accumulate
        for i in 1:(length(times) - 1)
            # get row index
            rowind = findlast((model.data.id .== subj) .& (model.data.tstart .<= times[i]))

            # compute hazard
            chaz += call_cumulhaz(times[i], times[i+1], model.parameters[hazind], rowind, model.hazards[hazind]; give_log = false)
        end

        # save
        cumulative_hazards[s] = chaz
    end

    # return cumulative hazards
    return cumulative_hazards
end
