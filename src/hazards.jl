"""
Function call for total hazard for an absorbing state, always returns zero.
"""
function call_tothaz(t::Float64, _totalhazard::_TotalHazardAbsorbing, _hazards::Vector{_Hazard})
    0.0
end

"""
Function call to return the log-total hazard out of an origin state. 
"""
function call_tothaz(t::Float64, rowind::Int64, _totalhazard::_TotalHazardTransient, _hazards::Vector{_Hazard}; give_log = true)
    # log total hazard
    log_tot_haz = 
        StatsFuns.logsumexp(
            map(x -> call_haz(t, rowind, x), _hazards[_totalhazard.components]))
    
    # return the log, or not
    give_log ? log_tot_haz : exp(log_tot_haz)
end

"""
Caller for exponential cause-specific hazards.
"""
function call_haz(t::Float64, rowind::Int64, _hazard::_Exponential; give_log = true)
    log_haz = _hazard.parameters[1]
    give_log ? log_haz : exp(log_haz)
end

"""
Caller for exponential cause-specific hazards with covariate adjustment.
"""
function call_haz(t::Float64, rowind::Int64, _hazard::_ExponentialReg; give_log = true)
    log_haz = dot(_hazard.parameters, _hazard.data[rowind,:])
    give_log ? log_haz : exp(log_haz)
end

"""
Caller for Weibull cause-specific hazards.
"""
function call_haz(t::Float64, rowind::Int64, _hazard::_Weibull; give_log = true)

    # scale and shape
    log_scale = _hazard.parameters[1]
    log_shape = _hazard.parameters[2]

    # compute hazard
    log_haz = log_shape + exp(log_shape) * log_scale + expm1(log_shape) * log(t)

    give_log ? log_haz : exp(log_haz)
end

"""
Caller for Weibull cause-specific hazards with covariate adjustment.
"""
function call_haz(t::Float64, rowind::Int64, _hazard::_WeibullReg; give_log = true)

    # scale and shape
    log_scale = dot(_hazard.parameters[_hazard.scaleinds], _hazard.data[rowind,:])
    log_shape = dot(_hazard.parameters[_hazard.shapeinds], _hazard.data[rowind,:])

    # compute hazard
    log_haz = log_shape + exp(log_shape) * log_scale + expm1(log_shape) * log(t)

    give_log ? log_haz : exp(log_haz)
end
