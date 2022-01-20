    """
    Hazard(haz::StatsModels.FormulaTerm, family::string, statefrom::Int64, stateto::Int64)

Composite type for a cause-specific hazard function. Documentation to follow. 
"""
struct Hazard
    hazard::StatsModels.FormulaTerm   # StatsModels.jl formula
    family::String     # one of "exp", "wei", "gg", or "sp"
    statefrom::Int64   # starting state number
    stateto::Int64     # destination state number
end

"""
Abstract struct for internal _Hazard types
"""
abstract type _Hazard end

"""
Exponential cause-specific hazard.
"""
Base.@kwdef mutable struct _Exponential <: _Hazard
    hazname::Symbol
    hazpars::Vector{Symbol}
    data::Array{Float64}
    parameters::Vector{Float64} 
end

"""
Exponential cause-specific hazard with covariate adjustment. Rate is a log-linear function of covariates.
"""
Base.@kwdef mutable struct _ExponentialReg <: _Hazard
    hazname::Symbol
    hazpars::Vector{Symbol}
    data::Array{Float64}
    parameters::Vector{Float64} 
end

"""
Weibull cause-specific hazard.
"""
Base.@kwdef mutable struct _Weibull <: _Hazard
    hazname::Symbol
    hazpars::Vector{Symbol}
    scaleinds::UnitRange{Int64} # always 1
    shapeinds::UnitRange{Int64} # always 2
    data::Array{Float64} # just an intercept
    parameters::Vector{Float64}
end

"""
Weibull cause-specific hazard with covariate adjustment. Scale and shape are log-linear functions of covariates.
"""
Base.@kwdef mutable struct _WeibullReg <: _Hazard
    hazname::Symbol
    hazpars::Vector{Symbol}
    scaleinds::UnitRange{Int64}
    shapeinds::UnitRange{Int64}
    data::Array{Float64}
    parameters::Vector{Float64}
end

"""
Abstract type for total hazards.
"""
abstract type _TotalHazard end

"""
Total hazard for absorbing states, contains nothing as the total hazard is always zero.
"""
struct _TotalHazardAbsorbing <: _TotalHazard 
end

"""
Function call for total hazard for an absorbing state, always returns zero.
"""
function call_tothaz(t::Float64, _totalhazard::_TotalHazardAbsorbing, _hazards::Vector{_Hazard})
    0.0
end

"""
Total hazard struct for transient states, contains the indices of cause-specific hazards that contribute to the total hazard.
"""
struct _TotalHazardTransient <: _TotalHazard
    components::Vector{Int64}
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
