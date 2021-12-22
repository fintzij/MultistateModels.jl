"""
    haz(haz::StatsModels.FormulaTerm, family::string, statefrom::Int64, stateto::Int64)

Composite type for a cause-specific hazard function. Documentation to follow. 
"""
struct Hazard
    hazard::StatsModels.FormulaTerm   # StatsModels.jl formula
    family::String     # one of "exp", "wei", "gg", or "sp"
    statefrom::Int64   # starting state number
    stateto::Int64     # destination state number
end

# this needs more thought - internal hazard type
mutable struct _Hazard_exp
    hazfun::Function 
    parameters::Vector{Float64}
    data::Array{Float64}
    hazfam::String 
    statefrom::Int64
    stateto::Int64
    inds::Vector{Int64}
end

function haz_exp(t::Float64, parameters::Vector{Float64}, data::Array{Float64,2}; loghaz = true, rate_inds::Vector{Int64})

    log_haz = data * parameters[rate_inds]

    # calculate log-hazard
    if loghaz == true 
        return log_haz
    else 
        return exp(log_haz)
    end
end

# weibull case
function haz_wei(t::Float64, parameters::Vector{Float64}, data::Array{Float64,2}; 
    loghaz = true, scale_inds::Vector{Float64}, shape_inds::Vector{Float64})

    # compute parameters
    log_shape = data * parameters[shape_inds] # log(p)
    log_scale = data * parameters[scale_inds] # log(lambda)

    # calculate log-hazard
    log_haz = log_shape + exp(log_shape) * log_scale + expm1(log_shape) * log(t)

    # calculate log-hazard
    if loghaz == true 
        return log_haz
    else 
        return exp(log_haz)
    end
end



# gamma case
function haz_gamma(t::Float64, parameters::Vector{Float64}, data; loghaz = true, scale_inds, shape_inds)

end

# generalized gamma

# log-normal