"""
    Abstract type for hazard functions. Subtypes are ParametricHazard or SplineHazard.
"""
abstract type HazardFunction end

"""
    ParametricHazard(haz::StatsModels.FormulaTerm, family::string, statefrom::Int64, stateto::Int64)

Specify a cause-specific baseline hazard. 

# Arguments
- `hazard`: regression formula for the (log) hazard, parsed using StatsModels.jl.
- `family`: parameterization for the baseline hazard, one of "exp" for exponential, "wei" for Weibull, "gom" for Gompert. 
- `statefrom`: state number for the origin state.
- `stateto`: state number for the destination state.
"""
struct ParametricHazard <: HazardFunction
    hazard::StatsModels.FormulaTerm   # StatsModels.jl formula
    family::String     # one of "exp", "wei", "gom"
    statefrom::Int64   # starting state number
    stateto::Int64     # destination state number
end

"""
    SplineHazard(haz::StatsModels.FormulaTerm, family::string, statefrom::Int64, stateto::Int64; df::Union{Int64,Nothing}, degree::Int64, knots::Union{Vector{Float64},Nothing}, boundaryknots::Union{Vector{Float64},Nothing}, intercept::Bool, periodic::Bool)

Specify a cause-specific baseline hazard. 

# Arguments
- `hazard`: regression formula for the (log) hazard, parsed using StatsModels.jl.
- `family`: "ms" for M-spline for the baseline hazard.
- `statefrom`: state number for the origin state.
- `stateto`: state number for the destination state.
- `df`: Degrees of freedom.
- `degree`: Degree of the spline polynomial basis.
- `knots`: Vector of knots.
- `boundaryknots`: Length 2 vector of boundary knots.
- `intercept`: Defaults to true for whether the spline should include an intercept.
- `periodic`: Periodic spline basis, defaults to false.
- `monotonic`: Assume that baseline hazard is monotonic, defaults to false. If true, use an I-spline basis for the hazard and a C-spline for the cumulative hazard.
"""
struct SplineHazard <: HazardFunction
    hazard::StatsModels.FormulaTerm   # StatsModels.jl formula
    family::String     # "ms" for M-Splines
    statefrom::Int64   # starting state number
    stateto::Int64     # destination state number
    df::Union{Nothing,Int64}
    degree::Int64
    knots::Union{Nothing,Vector{Float64}}
    boundaryknots::Union{Nothing,Vector{Float64}}
    intercept::Bool
    periodic::Bool
    monotonic::Bool
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
Weibull cause-specific proportional hazard. The log baseline hazard is a linear function of log time and covariates have a multiplicative effect vis-a-vis the baseline hazard.
"""
Base.@kwdef struct _WeibullPH <: _Hazard
    hazname::Symbol
    data::Array{Float64}
    parnames::Vector{Symbol}
    statefrom::Int64   # starting state number
    stateto::Int64     # destination state number
end

"""
Gompertz cause-specific hazard.
"""
Base.@kwdef struct _Gompertz <: _Hazard
    hazname::Symbol
    data::Array{Float64} # just an intercept
    parnames::Vector{Symbol}
    statefrom::Int64   # starting state number
    stateto::Int64     # destination state number
end


"""
Gompertz cause-specific proportional hazard. The log baseline hazard is a linear function of time and covariates have a multiplicative effect vis-a-vis the baseline hazard.
"""
Base.@kwdef struct _GompertzPH <: _Hazard
    hazname::Symbol
    data::Array{Float64}
    parnames::Vector{Symbol}
    statefrom::Int64   # starting state number
    stateto::Int64     # destination state number
end

"""
Spline for cause-specific hazard. The baseline hazard evaluted at a time, t, is a linear combination of M-spline basis functions, or an I-spline if the hazard is monotonic. Hence, the cumulative hazard is an I-spline or C-spline, respectively. Covariates have a multiplicative effect vis-a-vis the baseline hazard.
"""
Base.@kwdef struct _Spline <: _Hazard
    hazname::Symbol
    data::Array{Float64}
    parnames::Vector{Symbol}
    statefrom::Int64
    stateto::Int64
    times::ElasticVector{Float64}
    hazbasis::ElasticArray{Float64}
    chazbasis::ElasticArray{Float64}
    hazobj::RObject{RealSxp}
    chazobj::RObject{RealSxp}
    attr::OrderedDict{Symbol, Any}
end

# """
# Spline for cause-specific proportional hazard. The baseline hazard evaluted at a time, t, is a linear combination of M-spline basis functions or an I-spline if the hazard is monotonic. Hence, the cumulative hazard is an I-spline or C-spline, respectively. Covariates have a multiplicative effect vis-a-vis the baseline hazard.
# """
# Base.@kwdef struct _SplinePH <: _Hazard
#     hazname::Symbol
#     data::Array{Float64}
#     parnames::Vector{Symbol}
#     statefrom::Int64
#     stateto::Int64
#     times::Vector{Float64}
#     hazbasis::ElasticArray{Float64}
#     chazbasis::ElasticArray{Float64}
#     hazobj::RObject{RealSxp}
#     chazobj::RObject{RealSxp}
#     attr::OrderedDict{Symbol, Any}
# end

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
    model::MultistateProcess
    statefrom::Int64
    targets::Matrix{Float64}
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
    modelcall::NamedTuple
end

"""
    MultistateModelFitted(data::DataFrame, parameters::VectorOfVectors,hazards::Vector{_Hazard}, totalhazards::Vector{_TotalHazard},tmat::Matrix{Int64}, hazkeys::Dict{Symbol, Int64}, subjectindices::Vector{Vector{Int64}})

Mutable struct that fully specifies a fitted multistate model. 
"""
Base.@kwdef struct MultistateModelFitted <: MultistateProcess
    data::DataFrame
    parameters::VectorOfVectors 
    loglik::Float64
    vcov::Matrix{Float64}
    hazards::Vector{_Hazard}
    totalhazards::Vector{_TotalHazard}
    tmat::Matrix{Int64}
    hazkeys::Dict{Symbol, Int64}
    subjectindices::Vector{Vector{Int64}}
    markovsurrogate::MarkovSurrogate
    modelcall::NamedTuple
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
    ExactData(model::MultistateModel, samplepaths::Array{SamplePath})

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
    SMPanelData(model::MultistateModel, paths::Array{SamplePath}, weights::ElasticArray{Float64})

Struct containing panel data, a model object, and bookkeeping objects. Used in fitting a multistate semi-Markov model to panel data via MCEM.
"""
struct SMPanelData
    model::MultistateModel
    paths::Array{SamplePath}
    weights::ElasticArray{Float64}
    totweights::ElasticArray{Float64}
end

