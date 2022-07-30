"""
    
    prob(_totalhazard::_TotalHazardTransient, _hazards::Vector{_Hazard}, lb::Float64, ub::Float64, rowind::Int64)

Compute the survival probability over the interval from `lb` to `ub` by integrating the total hazard via quadrature.
"""

function survprob(_totalhazard::_TotalHazardTransient, _hazards::Vector{_Hazard}, lb::Float64, ub::Float64, rowind::Int64)

    # solve the quadrature problem
    exp(-quadgk(
        t -> MultistateModels.tothaz(
            t,
            rowind, 
            _totalhazard,
            _hazards;
            give_log = false),
        lb,
        ub)[1])

end

"""
    cumulhaz(_totalhazard::_TotalHazardTransient, _hazards::Vector{_Hazard}, lb::Float64, ub::Float64, rowind::Int64)

Compute the cumulative hazard over the interval from `lb` to `ub` by integrating the total hazard via quadrature.
"""
function cumulhaz(_totalhazard::_TotalHazardTransient, _hazards::Vector{_Hazard}, lb::Float64, ub::Float64, rowind::Int64)
    
    # solve the quadrature problem
    quadgk(
        t -> MultistateModels.tothaz(
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
function tothaz(t::Float64, rowind::Int64, _totalhazard::_TotalHazardAbsorbing, _hazards::Vector{_Hazard}; give_log = false)
    give_log ? -Inf : 0
end

"""
    tothaz(t::Float64, rowind::Int64, _totalhazard::_TotalHazardTransient, _hazards::Vector{_Hazard}; give_log = true)

Return the log-total hazard out of a transient state. 

# Arguments 
- `t::Float64`: current time
- `rowind::Int64`: row index in data
- `_totalhazard::_TotalHazardTransient`: total hazard from transient state, contains indices for hazards that contribute to the total hazard
- `_hazards::_Hazard`: vector of cause-specific hazards
- `give_log::Bool`: should the log total hazard be returned (default)
"""
function tothaz(t::Float64, rowind::Int64, _totalhazard::_TotalHazardTransient, _hazards::Vector{_Hazard}; give_log = true)

    # log total hazard
    log_tot_haz = 
        logsumexp(
            map(x -> call_haz(t, rowind, x), _hazards[_totalhazard.components]))
    
    # return the log, or not
    give_log ? log_tot_haz : exp(log_tot_haz)
end

"""
    call_haz(t::Float64, rowind::Int64, _hazard::_Exponential; give_log = true)

Return the exponential cause-specific hazards.
"""
# RESUME HERE - This is the signature we want
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
        exp(log_shape) * log_scale + log_shape + expm1(log_shape) * log(t)

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
        shape * log_scale + log(ub ^ shape - lb ^ shape)

    give_log ? log_cumul_haz : exp(log_cumul_haz)
end

"""
    call_haz(t::Float64, parameters::Vector{Float64}, rowind::Int64, _hazard::_WeibullPH; give_log = true)

Return the Weibull cause-specific proportional hazards. Weibull proportional hazards model parameterized like in Section 2.3.1 of Kalbfleisch and Prentice.
"""
function call_haz(t::Float64, parameters::Vector{Float64}, rowind::Int64, _hazard::_WeibullPH; give_log = true)

    # scale and shape
    log_shape = parameters[1]

    # compute hazard - do we need a special case for t = 0?
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
        dot(parameters[2:end], _hazard.data[rowind,:]) + log(ub ^ shape - lb ^ shape)

    give_log ? log_cumul_haz : exp(log_cumul_haz)
end