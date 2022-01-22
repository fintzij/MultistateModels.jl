"""
    MultistateProcess(data::DataFrame, hazards::Vector{_Hazard}, totalhazards::Vector{_TotalHazard}, tmat::Matrix{Int64})

Mutable struct that fully specifies a multistate process for simulation or inference. 
"""
Base.@kwdef mutable struct MultistateModel 
    data::DataFrame
    hazards::Vector{_Hazard}
    totalhazards::Vector{_TotalHazard}
    tmat::Matrix{Int64}
end

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
Total hazard struct for transient states, contains the indices of cause-specific hazards that contribute to the total hazard.
"""
struct _TotalHazardTransient <: _TotalHazard
    components::Vector{Int64}
end
