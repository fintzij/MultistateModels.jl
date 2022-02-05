"""
    call_cumulhaz()

Return something.
"""
function call_cumulhaz(_totalhazard::_TotalHazardTransient, _hazards::Vector{_Hazard}, lb::Float64, ub::Float64, rowind::Int64)
    
    # solve the quadrature problem
    quadgk(
        t -> MultistateModels.call_tothaz(
            t, 
            rowind, 
            _totalhazard,
            _hazards;
            give_log = false),
        lb,
        ub)[1]
end

"""
    # Arguments 
- `t::Float64`: current time
- `rowind::Int64`: row index in data
- `_totalhazard::_TotalHazardAbsorbing`: total hazard from an absorbing state, empty but indicates the type for dispatch
- `_hazards::_Hazard`: vector of cause-specific hazards
- `give_log::Bool`: should the log total hazard be returned (default)

Return the total hazard for an absorbing state, which is always zero.
"""
function call_tothaz(t::Float64, rowind::Int64, _totalhazard::_TotalHazardAbsorbing, _hazards::Vector{_Hazard}; give_log = false)
    give_log ? -Inf : 0
end

"""
    call_tothaz(t::Float64, rowind::Int64, _totalhazard::_TotalHazardTransient, _hazards::Vector{_Hazard}; give_log = true)

Return the log-total hazard out of a transient state. 

# Arguments 
- `t::Float64`: current time
- `rowind::Int64`: row index in data
- `_totalhazard::_TotalHazardTransient`: total hazard from transient state, contains indices for hazards that contribute to the total hazard
- `_hazards::_Hazard`: vector of cause-specific hazards
- `give_log::Bool`: should the log total hazard be returned (default)
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
    call_haz(t::Float64, rowind::Int64, _hazard::_Exponential; give_log = true)

Return the exponential cause-specific hazards.
"""
function call_haz(t::Float64, rowind::Int64, _hazard::_Exponential; give_log = true)
    log_haz = _hazard.parameters[1]
    give_log ? log_haz : exp(log_haz)
end

"""
    call_haz(t::Float64, rowind::Int64, _hazard::_ExponentialReg; give_log = true)

Return the exponential cause-specific hazards with covariate adjustment.
"""
function call_haz(t::Float64, rowind::Int64, _hazard::_ExponentialReg; give_log = true)
    log_haz = dot(_hazard.parameters, _hazard.data[rowind,:])
    give_log ? log_haz : exp(log_haz)
end

"""
    call_haz(t::Float64, rowind::Int64, _hazard::_Weibull; give_log = true)

Return the Weibull cause-specific hazards.
"""
function call_haz(t::Float64, rowind::Int64, _hazard::_Weibull; give_log = true)

    # scale and shape
    log_scale = _hazard.parameters[1]
    log_shape = _hazard.parameters[2]

    # compute hazard - do we need a special case for t=0?
    log_haz = 
        log_shape + exp(log_shape) * log_scale + expm1(log_shape) * log(t) 

    give_log ? log_haz : exp(log_haz)
end

"""
    call_haz(t::Float64, rowind::Int64, _hazard::_WeibullReg; give_log = true)

Return the Weibull cause-specific hazards with covariate adjustment.
"""
function call_haz(t::Float64, rowind::Int64, _hazard::_WeibullReg; give_log = true)

    # scale and shape
    log_scale = dot(_hazard.parameters[_hazard.scaleinds], _hazard.data[rowind,:])
    log_shape = dot(_hazard.parameters[_hazard.shapeinds], _hazard.data[rowind,:])

    # compute hazard - do we need a special case for t = 0?
    log_haz = 
        log_shape + exp(log_shape) * log_scale + expm1(log_shape) * log(t)

    give_log ? log_haz : exp(log_haz)
end

"""
    call_haz(t::Float64, rowind::Int64, _hazard::_WeibullPH; give_log = true)

Return the Weibull cause-specific proportional hazards.
"""
function call_haz(t::Float64, rowind::Int64, _hazard::_WeibullPH; give_log = true)

    # scale and shape
    log_scale = _hazard.parameters[1]
    log_shape = _hazard.parameters[2]
    log_PH = dot(_hazard.parameters[3:end], _hazard.data[rowind,:])

    # compute hazard - do we need a special case for t = 0?
    log_haz = 
        log_shape + exp(log_shape) * log_scale + expm1(log_shape) * log(t) + log_PH

    give_log ? log_haz : exp(log_haz)
end