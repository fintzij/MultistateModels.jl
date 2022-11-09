"""
    Hazard(haz::StatsModels.FormulaTerm, family::string, statefrom::Int64, stateto::Int64)

Specify a cause-specific hazard function. 

# Arguments
- 'hazard': regression formula for the (log) hazard, parsed using StatsModels.jl.
- 'family': parameterization for the baseline hazard, one of "exp" for exponential, "wei" for Weibull, "gom" for Gompertz (not yet implemented),  "ms" for M-spline (on the baseline hazard, not yet implemented), or "bs" for B-spline (on the log baseline hazard, not yet implemented). 
- 'statefrom': state number for the origin state.
- 'stateto': state number for the destination state.
"""
struct Hazard
    hazard::StatsModels.FormulaTerm   # StatsModels.jl formula
    family::String     # one of "exp", "wei", "gom", "ms", or "bs"
    statefrom::Int64   # starting state number
    stateto::Int64     # destination state number
end

"""
Abstract struct for internal _Hazard types.
"""
abstract type _Hazard end

"""
Exponential cause-specific hazard.
"""
Base.@kwdef struct _Exponential <: _Hazard
    hazname::Symbol
    data::Array{Float64}
    parnames::Vector{Symbol}
    statefrom::Int64   # starting state number
    stateto::Int64     # destination state number
end

"""
Exponential cause-specific hazard with covariate adjustment. Rate is a log-linear function of covariates.
"""
Base.@kwdef struct _ExponentialPH <: _Hazard
    hazname::Symbol
    data::Array{Float64}
    parnames::Vector{Symbol}
    statefrom::Int64   # starting state number
    stateto::Int64     # destination state number
end

"""
Weibull cause-specific hazard.
"""
Base.@kwdef struct _Weibull <: _Hazard
    hazname::Symbol
    data::Array{Float64} # just an intercept
    parnames::Vector{Symbol}
    statefrom::Int64   # starting state number
    stateto::Int64     # destination state number
end


"""
Weibull cause-specific proportional hazard. The baseline hazard is Weibull and covariates have a multiplicative effect vis-a-vis the baseline hazard.
"""
Base.@kwdef struct _WeibullPH <: _Hazard
    hazname::Symbol
    data::Array{Float64}
    parnames::Vector{Symbol}
    statefrom::Int64   # starting state number
    stateto::Int64     # destination state number
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
Total hazard struct for transient states, contains the indices of cause-specific hazards that contribute to the total hazard. The components::Vector{Int64} are indices of Vector{_Hazard} when call_tothaz needs to extract the correct cause-specific hazards.
"""
struct _TotalHazardTransient <: _TotalHazard
    components::Vector{Int64}
end

"""
    Abstract type for MultistateProcess.
"""
abstract type MultistateProcess end

"""
    MarkovSurrogate(hazards::Vector{_Hazard}, parameters::VectorOfVectors)
"""
Base.@kwdef struct MarkovSurrogate
    hazards::Vector{_Hazard}
    parameters::VectorOfVectors
end

"""
    SurrogateControl(model::MultistateModel, statefrom, targets, uinds, ginds)

Struct containing objects for computing the discrepancy of a Markov surrogate.
"""
Base.@kwdef struct SurrogateControl
    model::MultistateModel
    statefrom::Int64
    targets::Vector{Vector{Matrix{Float64}}}
    uinds::Vector{Union{Nothing, Int64}}
    ginds::Vector{Union{Nothing, Int64}}
end


"""
    MultistateModel(data::DataFrame, parameters::VectorOfVectors,hazards::Vector{_Hazard}, totalhazards::Vector{_TotalHazard},tmat::Matrix{Int64}, hazkeys::Dict{Symbol, Int64}, subjectindices::Vector{Vector{Int64}})

Mutable struct that fully specifies a multistate process for simulation or inference. 
"""
Base.@kwdef struct MultistateModel <: MultistateProcess
    data::DataFrame
    parameters::VectorOfVectors 
    hazards::Vector{_Hazard}
    totalhazards::Vector{_TotalHazard}
    tmat::Matrix{Int64}
    hazkeys::Dict{Symbol, Int64}
    subjectindices::Vector{Vector{Int64}}
    markovsurrogate::MarkovSurrogate
end

"""
    MultistateModelFitted(data::DataFrame, parameters::VectorOfVectors,hazards::Vector{_Hazard}, totalhazards::Vector{_TotalHazard},tmat::Matrix{Int64}, hazkeys::Dict{Symbol, Int64}, subjectindices::Vector{Vector{Int64}})

Mutable struct that fully specifies a fitted multistate model. 
"""
Base.@kwdef struct MultistateModelFitted <: MultistateProcess
    data::DataFrame
    parameters::VectorOfVectors 
    hazards::Vector{_Hazard}
    totalhazards::Vector{_TotalHazard}
    tmat::Matrix{Int64}
    hazkeys::Dict{Symbol, Int64}
    subjectindices::Vector{Vector{Int64}}
    markovsurrogate::MarkovSurrogate
    loglik::Float64
    vcov::Matrix{Float64}
end

"""
    SamplePath(subjID::Int64, times::Vector{Float64}, states::Vector{Int64})

Struct for storing a sample path, consists of subject identifier, jump times, state sequence.
"""
struct SamplePath
    subj::Int64
    times::Vector{Float64}
    states::Vector{Int64}
end

"""
    ExactData(samplepaths::Array{SamplePath}, model::MultistateModel)

Struct containing exactly observed sample paths and a model object. Used in fitting a multistate model to completely observed data.
"""
struct ExactData
    model::MultistateModel
    paths::Array{SamplePath}
end

"""
    MPanelData(model::MultistateModel, books::Tuple)

Struct containing panel data, a model object, and bookkeeping objects. Used in fitting a multistate Markov model to panel data.
"""
struct MPanelData
    model::MultistateModel
    books::Tuple # tpm_index and tpm_map, from build_tpm_containers
end

"""
    SMPanelData(model::MultistateModel, books::Tuple)

Struct containing panel data, a model object, and bookkeeping objects. Used in fitting a multistate semi-Markov model to panel data.
"""
struct SMPanelData
    model::MultistateModel
    books::Tuple # tpm_index and tpm_map, from build_tpm_containers
end

