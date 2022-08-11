"""
    survprob(lb::Float64, ub::Float64, parameters::Vector{Vector{Float64}}, rowind::Int64, _totalhazard::_TotalHazardTransient, _hazards::Vector{_Hazard}; give_log = true)

Return the survival probability over the interval [lb, ub]. 

# Arguments 
- `lb::Float64`: start time
- `ub::Float64`: end time
- `parameters::Vector{Vector{Float64}}`: model parameters, a vector of vectors
- `rowind::Int64`: row index in data
- `_totalhazard::_TotalHazardTransient`: total hazard from transient state, contains indices for hazards that contribute to the total hazard
- `_hazards::_Hazard`: vector of cause-specific hazards
- `give_log::Bool`: should the log total hazard be returned (default)
"""
function survprob(lb::Float64, ub::Float64, parameters::Vector{Vector{Float64}}, rowind::Int64, _totalhazard::_TotalHazardTransient, _hazards::Vector{_Hazard})

    # log total cumulative hazard
    exp(-total_cumulhaz(lb, ub, parameters, rowind, _totalhazard, _hazards; give_log = false))
end

"""
    total_cumulhaz(lb::Float64, ub::Float64, parameters::Vector{Vector{Float64}}, rowind::Int64, _totalhazard::_TotalHazardTransient, _hazards::Vector{_Hazard}; give_log = true)

Return the log-total cumulative hazard out of a transient state over the interval [lb, ub]. 

# Arguments 
- `lb::Float64`: start time
- `ub::Float64`: end time
- `parameters::Vector{Vector{Float64}}`: model parameters, a vector of vectors
- `rowind::Int64`: row index in data
- `_totalhazard::_TotalHazardTransient`: total hazard from transient state, contains indices for hazards that contribute to the total hazard
- `_hazards::_Hazard`: vector of cause-specific hazards
- `give_log::Bool`: should the log total hazard be returned (default)
"""
function total_cumulhaz(lb::Float64, ub::Float64, parameters::Vector{Vector{Float64}}, rowind::Int64, _totalhazard::_TotalHazardTransient, _hazards::Vector{_Hazard}; give_log = true)

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
    total_cumulhaz(lb::Float64, ub::Float64, parameters::Vector{Vector{Float64}}, rowind::Int64, _totalhazard::_TotalHazardAbsorbing, _hazards::Vector{_Hazard}; give_log = true)

Return zero log-total cumulative hazard over the interval [lb, ub] as the current state is absorbing. 

# Arguments 
- `lb::Float64`: start time
- `ub::Float64`: end time
- `parameters::Vector{Vector{Float64}}`: model parameters, a vector of vectors
- `rowind::Int64`: row index in data
- `_totalhazard::_TotalHazardAbsorbing`: absorbing state.
- `_hazards::_Hazard`: vector of cause-specific hazards
- `give_log::Bool`: should the log total hazard be returned (default)
"""
function total_cumulhaz(lb::Float64, ub::Float64, parameters::Vector{Vector{Float64}}, rowind::Int64, _totalhazard::_TotalHazardAbsorbing, _hazards::Vector{_Hazard}; give_log = true)

    # return 0 cumulative hazard
    give_log ? -Inf : 0

end

"""
    call_haz(t::Float64, rowind::Int64, _hazard::_Exponential; give_log = true)

Return the exponential cause-specific hazards.
"""
function call_haz(t::Float64, parameters::Vector{Float64}, rowind::Int64, _hazard::_Exponential; give_log = true)
    give_log ? parameters[1] : exp(parameters[1])
end

"""
    call_cumulhaz(lb::Float64, ub::Float64, parameters::Vector{Float64}, rowind::Int64, _hazard::_Exponential; give_log = true)

Cumulative hazard for the exponential hazards over the interval [lb, ub]. 
"""
function call_cumulhaz(lb::Float64, ub::Float64, parameters::Vector{Float64}, rowind::Int64, _hazard::_Exponential; give_log = true)
    give_log ? parameters[1] + log(ub - lb) : exp(parameters[1] + log(ub - lb))
end

"""
    call_haz(t::Float64, parameters::Vector{Float64}, rowind::Int64, _hazard::_ExponentialPH; give_log = true)

Return the exponential cause-specific hazards with covariate adjustment.
"""
function call_haz(t::Float64, parameters::Vector{Float64}, rowind::Int64, _hazard::_ExponentialPH; give_log = true)
    log_haz = dot(parameters, _hazard.data[rowind,:])
    give_log ? log_haz : exp(log_haz)
end

"""
    call_cumulhaz(lb::Float64, ub::Float64, parameters::Vector{Float64}, rowind::Int64, _hazard::_ExponentialPH; give_log = true)

Cumulative hazard for exponential proportional hazards over the interval [lb, ub].
"""
function call_cumulhaz(lb::Float64, ub::Float64, parameters::Vector{Float64}, rowind::Int64, _hazard::_ExponentialPH; give_log = true)
    log_haz = dot(parameters, _hazard.data[rowind,:])
    give_log ? log_haz + log(ub - lb) : exp(log_haz + log(ub - lb))
end

"""
    call_haz(t::Float64, parameters::Vector{Float64}, rowind::Int64, _hazard::_Weibull; give_log = true)

Return the Weibull cause-specific hazards. No covariate adjustement, parameterized as in Section 2.2.2 of Kalbfleisch and Prentice.
"""
function call_haz(t::Float64, parameters::Vector{Float64}, rowind::Int64, _hazard::_Weibull; give_log = true)

    # scale and shape
    log_shape = parameters[1]
    log_scale = parameters[2]

    # compute hazard - do we need a special case for t=0?
    log_haz = 
        log_shape + expm1(log_shape) * log(t) + exp(log_shape) * log_scale 

    give_log ? log_haz : exp(log_haz)
end

"""
    call_cumulhaz(lb::Float64, ub::Float64, parameters::Vector{Float64}, rowind::Int64, _hazard::_Weibull; give_log = true)

Cumulative hazard for Weibulll hazards over the interval [lb, ub].
"""
function call_cumulhaz(lb::Float64, ub::Float64, parameters::Vector{Float64}, rowind::Int64, _hazard::_Weibull; give_log = true)

    # scale and shape
    shape = exp(parameters[1])
    log_scale = parameters[2]

    # cumulative hazard
    log_cumul_haz = 
        log(ub ^ shape - lb ^ shape) + shape * log_scale

    give_log ? log_cumul_haz : exp(log_cumul_haz)
end

"""
    call_haz(t::Float64, parameters::Vector{Float64}, rowind::Int64, _hazard::_WeibullPH; give_log = true)

Return the Weibull cause-specific proportional hazards. Weibull proportional hazards model parameterized like in Section 2.3.1 of Kalbfleisch and Prentice.
"""
function call_haz(t::Float64, parameters::Vector{Float64}, rowind::Int64, _hazard::_WeibullPH; give_log = true)

    # scale and shape
    log_shape = parameters[1]

    # compute hazard
    log_haz = 
        log_shape + expm1(log_shape) * log(t) + dot(parameters[2:end], _hazard.data[rowind,:])

    give_log ? log_haz : exp(log_haz)
end

"""
    call_cumulhaz(lb::Float64, ub::Float64, parameters::Vector{Float64}, rowind::Int64, _hazard::_WeibullPH; give_log = true)

Cumulative hazard for Weibulll proportional hazards over the interval [lb, ub].
"""
function call_cumulhaz(lb::Float64, ub::Float64, parameters::Vector{Float64}, rowind::Int64, _hazard::_WeibullPH; give_log = true)

    # scale and shape
    shape = exp(parameters[1])

    # cumulative hazard
    log_cumul_haz = 
        log(ub ^ shape - lb ^ shape) + dot(parameters[2:end], _hazard.data[rowind,:])

    give_log ? log_cumul_haz : exp(log_cumul_haz)
end