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

Base.@kwdef mutable struct _Hazard
    hazname::Symbol
    hazpars::Vector{Symbol}
    statefrom::Int64
    stateto::Int64
    hazfam::String 
    data::Array{Float64}
    parameters::Vector{Float64}
    hazfun::Function
end

function haz_exp(t::Float64, parameters::Vector{Float64}, data::Array{Float64,2}; give_log = true)

    log_haz = data * parameters

    if give_log == true 
        return log_haz
    else 
        return exp(log_haz)
    end
end

# weibull case
function haz_wei(t::Float64, parameters::Vector{Float64}, data::Array{Float64,2}; give_log = true)

    # compute parameters
    log_shape = data * parameters[args.shape_inds] # log(p)
    log_scale = data * parameters[args.scale_inds] # log(lambda)

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