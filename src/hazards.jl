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

# abstract type for internal hazard structs
abstract type _Hazard end

# specialized subtypes for parametric hazards
Base.@kwdef mutable struct _Exponential <: _Hazard
    hazname::Symbol
    hazpars::Vector{Symbol}
    parameters::Vector{Float64} 
end

Base.@kwdef mutable struct _ExponentialReg <: _Hazard
    hazname::Symbol
    hazpars::Vector{Symbol}
    data::Array{Float64}
    parameters::Vector{Float64} 
end

# redefine haz_exp + haz_wei to dispatch on the internal struct, a la distributions.jl
Base.@kwdef mutable struct _Weibull <: _Hazard
    hazname::Symbol
    hazpars::Vector{Symbol}
    parameters::Vector{Float64}
end

Base.@kwdef mutable struct _WeibullReg <: _Hazard
    hazname::Symbol
    hazpars::Vector{Symbol}
    scaleinds::UnitRange{Int64}
    shapeinds::UnitRange{Int64}
    data::Array{Float64}
    parameters::Vector{Float64}
end

### callers for hazard functions
# exponential hazard, no covariate adjustment
function call_haz(t::Float64, _hazard::_Exponential; give_log = true)
    log_haz = _hazard.parameters[1]
    give_log ? log_haz : exp(log_haz)
end

# exponential hazard with covariate adjustment
function call_haz(t::Float64, _hazard::_ExponentialReg, rowind::Int64; give_log = true)
    log_haz = dot(_hazard.parameters, _hazard.data[rowind,:])
    give_log ? log_haz : exp(log_haz)
end

# weibull case no covariate adjustment
function call_haz(t::Float64, _hazard::_Weibull; give_log = true)

    # scale and shape
    log_scale = _hazard.parameters[1]
    log_shape = _hazard.parameters[2]

    # compute hazard
    log_haz = log_shape + exp(log_shape) * log_scale + expm1(log_shape) * log(t)

    give_log ? log_haz : exp(log_haz)
end

# weibull with covariate adjustment
function call_haz(t::Float64, _hazard::_WeibullReg; give_log = true)

    # scale and shape
    log_scale = _hazard.parameters[_hazard.scaleinds]
    log_shape = _hazard.parameters[_hazard.shapeinds]

    # compute hazard
    log_haz = log_shape + exp(log_shape) * log_scale + expm1(log_shape) * log(t)

    give_log ? log_haz : exp(log_haz)
end


# gamma case
# function haz_gamma(t::Float64, parameters::Vector{Float64}, data; loghaz = true, scale_inds, shape_inds)

# end

# generalized gamma

# log-normal