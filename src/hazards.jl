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
    statefrom::Int64
    stateto::Int64 
    parameters::Vector{Float64}
end

Base.@kwdef mutable struct _ExponentialReg <: _Hazard
    hazname::Symbol
    hazpars::Vector{Symbol}
    statefrom::Int64
    stateto::Int64 
    data::Array{Float64}
    parameters::Vector{Float64}
end

# redefine haz_exp + haz_wei to dispatch on the internal struct, a la distributions.jl
Base.@kwdef mutable struct _Weibull <: _Hazard
    hazname::Symbol
    hazpars::Vector{Symbol}
    statefrom::Int64
    stateto::Int64 
    parameters::Vector{Float64}
end

Base.@kwdef mutable struct _WeibullReg <: _Hazard
    hazname::Symbol
    hazpars::Vector{Symbol}
    statefrom::Int64
    stateto::Int64 
    data::Array{Float64}
    parameters::Vector{Float64}
end

### callers for hazard functions
# exponential hazard, no covariate adjustment
function call_haz(t::Float64, _hazard::_Exponential; give_log = true)
    give_log ? _hazard.parameters : exp.(hazard.parameters)
end

# exponential hazard with covariate adjustment
function call_haz(t::Float64, _hazard::_ExponentialReg; give_log = true)
    give_log ? _hazard.parameters * _hazard.data : exp.(_hazard.parameters * _hazard.data)
end

# weibull case
function haz_wei(t::Float64, parameters::Vector{Float64}, data::Array{Float64,2}; give_log = true, args...)

    # compute parameters
    log_shape = data * parameters[args[:shape_inds]] # log(p)
    log_scale = data * parameters[args[:scale_inds]] # log(lambda)

    # calculate log-hazard
    log_haz = log_shape + exp(log_shape) * log_scale + expm1(log_shape) * log(t)

    # calculate log-hazard
    if give_log == true 
        return log_haz
    else 
        return exp(log_haz)
    end
end

# gamma case
# function haz_gamma(t::Float64, parameters::Vector{Float64}, data; loghaz = true, scale_inds, shape_inds)

# end

# generalized gamma

# log-normal