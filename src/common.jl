"""
    Abstract type for hazard functions. Subtypes are ParametricHazard or SplineHazard.
"""
abstract type HazardFunction end

"""
Abstract struct for internal _Hazard types.
"""
abstract type _Hazard end

"""
Abstract struct for internal Markov _Hazard types.
"""
abstract type _MarkovHazard <: _Hazard end

"""
Abstract struct for internal semi-Markov _Hazard types.
"""
abstract type _SemiMarkovHazard <: _Hazard end

"""
Abstract type for total hazards.
"""
abstract type _TotalHazard end

"""
    Abstract type for multistate process.
"""
abstract type MultistateProcess end


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
Exponential cause-specific hazard.
"""
struct _Exponential <: _MarkovHazard
    hazname::Symbol
    data::Array{Float64}
    parnames::Vector{Symbol}
    statefrom::Int64   # starting state number
    stateto::Int64     # destination state number
end

"""
Exponential cause-specific hazard with covariate adjustment. Rate is a log-linear function of covariates.
"""
struct _ExponentialPH <: _MarkovHazard
    hazname::Symbol
    data::Array{Float64}
    parnames::Vector{Symbol}
    statefrom::Int64   # starting state number
    stateto::Int64     # destination state number
end

"""
Weibull cause-specific hazard.
"""
struct _Weibull <: _SemiMarkovHazard
    hazname::Symbol
    data::Array{Float64} # just an intercept
    parnames::Vector{Symbol}
    statefrom::Int64   # starting state number
    stateto::Int64     # destination state number
end


"""
Weibull cause-specific proportional hazard. The log baseline hazard is a linear function of log time and covariates have a multiplicative effect vis-a-vis the baseline hazard.
"""
struct _WeibullPH <: _SemiMarkovHazard
    hazname::Symbol
    data::Array{Float64}
    parnames::Vector{Symbol}
    statefrom::Int64   # starting state number
    stateto::Int64     # destination state number
end

"""
Gompertz cause-specific hazard.
"""
struct _Gompertz <: _SemiMarkovHazard
    hazname::Symbol
    data::Array{Float64} # just an intercept
    parnames::Vector{Symbol}
    statefrom::Int64   # starting state number
    stateto::Int64     # destination state number
end


"""
Gompertz cause-specific proportional hazard. The log baseline hazard is a linear function of time and covariates have a multiplicative effect vis-a-vis the baseline hazard.
"""
struct _GompertzPH <: _SemiMarkovHazard
    hazname::Symbol
    data::Array{Float64}
    parnames::Vector{Symbol}
    statefrom::Int64   # starting state number
    stateto::Int64     # destination state number
end

"""
Spline for cause-specific hazard. The baseline hazard evaluted at a time, t, is a linear combination of M-spline basis functions, or an I-spline if the hazard is monotonic. Hence, the cumulative hazard is an I-spline or C-spline, respectively. Covariates have a multiplicative effect vis-a-vis the baseline hazard.
"""
struct _Spline <: _SemiMarkovHazard
    hazname::Symbol
    data::Array{Float64}
    parnames::Vector{Symbol}
    statefrom::Int64
    stateto::Int64
    times::Vector{Float64}
    hazbasis::ElasticArray{Float64}
    chazbasis::ElasticArray{Float64}
    hazobj::RObject{RealSxp}
    chazobj::RObject{RealSxp}
    attr::OrderedDict{Symbol, Any}
end

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
    MarkovSurrogate(hazards::Vector{_Hazard}, parameters::VectorOfVectors)
"""
struct MarkovSurrogate
    hazards::Vector{_Hazard}
    parameters::VectorOfVectors
end

"""
    SurrogateControl(model::MultistateModel, statefrom, targets, uinds, ginds)

Struct containing objects for computing the discrepancy of a Markov surrogate.
"""
struct SurrogateControl
    model::MultistateProcess
    statefrom::Int64
    targets::Matrix{Float64}
    uinds::Vector{Union{Nothing, Int64}}
    ginds::Vector{Union{Nothing, Int64}}
end

"""
    MultistateModel(data::DataFrame, parameters::VectorOfVectors,hazards::Vector{Union{_Exponential, _ExponentialPH}}, totalhazards::Vector{_TotalHazard},tmat::Matrix{Int64}, hazkeys::Dict{Symbol, Int64}, subjectindices::Vector{Vector{Int64}})

Struct that fully specifies a multistate process for simulation or inference, used in the case when sample paths are fully observed. 
"""
struct MultistateModel <: MultistateProcess
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
    MultistateMarkovModel(data::DataFrame, parameters::VectorOfVectors,hazards::Vector{Union{_Exponential, _ExponentialPH}}, totalhazards::Vector{_TotalHazard},tmat::Matrix{Int64}, hazkeys::Dict{Symbol, Int64}, subjectindices::Vector{Vector{Int64}})

Struct that fully specifies a multistate Markov process with no censored state, used with panel data.
"""
struct MultistateMarkovModel <: MultistateProcess
    data::DataFrame
    parameters::VectorOfVectors 
    hazards::Vector{Union{_Exponential,_ExponentialPH}}
    totalhazards::Vector{_TotalHazard}
    tmat::Matrix{Int64}
    hazkeys::Dict{Symbol, Int64}
    subjectindices::Vector{Vector{Int64}}
    markovsurrogate::MarkovSurrogate
    modelcall::NamedTuple
end

"""
MultistateMarkovModelCensored(data::DataFrame, parameters::VectorOfVectors,hazards::Vector{_Hazard}, totalhazards::Vector{_TotalHazard},tmat::Matrix{Int64}, hazkeys::Dict{Symbol, Int64}, subjectindices::Vector{Vector{Int64}})

Struct that fully specifies a multistate Markov process with some censored states, used with panel data.
"""
struct MultistateMarkovModelCensored <: MultistateProcess
    data::DataFrame
    parameters::VectorOfVectors 
    hazards::Vector{Union{_Exponential,_ExponentialPH}}
    totalhazards::Vector{_TotalHazard}
    tmat::Matrix{Int64}
    hazkeys::Dict{Symbol, Int64}
    subjectindices::Vector{Vector{Int64}}
    markovsurrogate::MarkovSurrogate
    modelcall::NamedTuple
end

"""
MultistateSemiMarkovModel(data::DataFrame, parameters::VectorOfVectors,hazards::Vector{_Hazard}, totalhazards::Vector{_TotalHazard},tmat::Matrix{Int64}, hazkeys::Dict{Symbol, Int64}, subjectindices::Vector{Vector{Int64}})

Struct that fully specifies a multistate semi-Markov process with no censored state for simulation or inference. 
"""
struct MultistateSemiMarkovModel <: MultistateProcess
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
MultistateSemiMarkovModelCensored(data::DataFrame, parameters::VectorOfVectors,hazards::Vector{_Hazard}, totalhazards::Vector{_TotalHazard},tmat::Matrix{Int64}, hazkeys::Dict{Symbol, Int64}, subjectindices::Vector{Vector{Int64}})

Struct that fully specifies a multistate semi-Markov process with some censored states for simulation or inference. 
"""
struct MultistateSemiMarkovModelCensored <: MultistateProcess
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

Struct that fully specifies a fitted multistate model. 
"""
struct MultistateModelFitted <: MultistateProcess
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


# """
#     MultistateMarkovModelFitted(data::DataFrame, parameters::VectorOfVectors,hazards::Vector{_Hazard}, totalhazards::Vector{_TotalHazard},tmat::Matrix{Int64}, hazkeys::Dict{Symbol, Int64}, subjectindices::Vector{Vector{Int64}})

# Struct that fully specifies a fitted multistate Markov model. 
# """
# struct MultistateMarkovModelFitted <: MultistateProcess
#     data::DataFrame
#     parameters::VectorOfVectors 
#     loglik::Float64
#     vcov::Matrix{Float64}
#     hazards::Vector{Union{_Exponential,_ExponentialPH}}
#     totalhazards::Vector{_TotalHazard}
#     tmat::Matrix{Int64}
#     hazkeys::Dict{Symbol, Int64}
#     subjectindices::Vector{Vector{Int64}}
#     markovsurrogate::MarkovSurrogate
#     modelcall::NamedTuple
# end

# """
#     MultistateMarkovModelFitted(data::DataFrame, parameters::VectorOfVectors,hazards::Vector{_Hazard}, totalhazards::Vector{_TotalHazard},tmat::Matrix{Int64}, hazkeys::Dict{Symbol, Int64}, subjectindices::Vector{Vector{Int64}})

# Struct that fully specifies a fitted multistate Markov model with some censored states. 
# """
# struct MultistateMarkovModelCensoredFitted <: MultistateProcess
#     data::DataFrame
#     parameters::VectorOfVectors 
#     loglik::Float64
#     vcov::Matrix{Float64}
#     hazards::Vector{Union{_Exponential,_ExponentialPH}}
#     totalhazards::Vector{_TotalHazard}
#     tmat::Matrix{Int64}
#     hazkeys::Dict{Symbol, Int64}
#     subjectindices::Vector{Vector{Int64}}
#     markovsurrogate::MarkovSurrogate
#     modelcall::NamedTuple
# end

# """
#     MultistateSemiMarkovModelFitted(data::DataFrame, parameters::VectorOfVectors,hazards::Vector{_Hazard}, totalhazards::Vector{_TotalHazard},tmat::Matrix{Int64}, hazkeys::Dict{Symbol, Int64}, subjectindices::Vector{Vector{Int64}})

# Struct that fully specifies a fitted multistate semi-Markov model. 
# """
# struct MultistateSemiMarkovModelFitted <: MultistateProcess
#     data::DataFrame
#     parameters::VectorOfVectors 
#     loglik::Float64
#     vcov::Matrix{Float64}
#     hazards::Vector{_Hazard}
#     totalhazards::Vector{_TotalHazard}
#     tmat::Matrix{Int64}
#     hazkeys::Dict{Symbol, Int64}
#     subjectindices::Vector{Vector{Int64}}
#     markovsurrogate::MarkovSurrogate
#     modelcall::NamedTuple
# end

# """
#     MultistateSemiMarkovModelFitted(data::DataFrame, parameters::VectorOfVectors,hazards::Vector{_Hazard}, totalhazards::Vector{_TotalHazard},tmat::Matrix{Int64}, hazkeys::Dict{Symbol, Int64}, subjectindices::Vector{Vector{Int64}})

# Struct that fully specifies a fitted multistate semi-Markov model with some censored states. 
# """
# struct MultistateSemiMarkovModelCensoredFitted <: MultistateProcess
#     data::DataFrame
#     parameters::VectorOfVectors 
#     loglik::Float64
#     vcov::Matrix{Float64}
#     hazards::Vector{_Hazard}
#     totalhazards::Vector{_TotalHazard}
#     tmat::Matrix{Int64}
#     hazkeys::Dict{Symbol, Int64}
#     subjectindices::Vector{Vector{Int64}}
#     markovsurrogate::MarkovSurrogate
#     modelcall::NamedTuple
# end

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
    model::MultistateProcess
    paths::Array{SamplePath}
end

"""
    MPanelData(model::MultistateModel, books::Tuple)

Struct containing panel data, a model object, and bookkeeping objects. Used in fitting a multistate Markov model to panel data.
"""
struct MPanelData
    model::MultistateProcess
    books::Tuple # tpm_index and tpm_map, from build_tpm_containers
end

"""
    SMPanelData(model::MultistateModel, paths::Array{SamplePath}, weights::ElasticArray{Float64})

Struct containing panel data, a model object, and bookkeeping objects. Used in fitting a multistate semi-Markov model to panel data via MCEM.
"""
struct SMPanelData
    model::MultistateProcess
    paths::Array{SamplePath}
    weights::ElasticArray{Float64}
    totweights::ElasticArray{Float64}
end

