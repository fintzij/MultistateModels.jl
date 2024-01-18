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
Abstract struct for internal spline _Hazard types.
"""
abstract type _SplineHazard <: _SemiMarkovHazard end

"""
Abstract type for total hazards.
"""
abstract type _TotalHazard end

"""
    Abstract type for multistate process.
"""
abstract type MultistateProcess end

"""
    Abstract type for multistate Markov process.
"""
abstract type MultistateMarkovProcess <: MultistateProcess end

"""
    Abstract type for multistate semi-Markov process.
"""
abstract type MultistateSemiMarkovProcess <: MultistateProcess end

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
- `family`: "sp" for splines for the baseline hazard.
- `statefrom`: state number for the origin state.
- `stateto`: state number for the destination state.
- `df`: Degrees of freedom.
- `degree`: Degree of the spline polynomial basis.
- `knots`: Vector of knots.
- `boundaryknots`: Length 2 vector of boundary knots.
- `monotonic`: Assume that baseline hazard is monotonic, defaults to "nonmonotonic". If "increasing" or "decreasing", use an I-spline basis for the hazard and a C-spline for the cumulative hazard.
- `meshsize`: number of intervals into which to discretize the spline basis, defaults to 10000. 
"""
struct SplineHazard <: HazardFunction
    hazard::StatsModels.FormulaTerm   # StatsModels.jl formula
    family::String     # "sp" for splines
    statefrom::Int64   # starting state number
    stateto::Int64     # destination state number
    df::Union{Nothing,Int64}
    degree::Int64
    knots::Union{Nothing,Vector{Float64}}
    boundaryknots::Union{Nothing,Vector{Float64}}
    monotonic::String
    meshsize::Int64
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
Weibull cause-specific proportional hazard. The log baseline hazard is a linear function of log time and covariates have a multiplicative effect on the baseline hazard.
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
Gompertz cause-specific proportional hazard. The log baseline hazard is a linear function of time and covariates have a multiplicative effect on the baseline hazard.
"""
struct _GompertzPH <: _SemiMarkovHazard
    hazname::Symbol
    data::Array{Float64}
    parnames::Vector{Symbol}
    statefrom::Int64   # starting state number
    stateto::Int64     # destination state number
end

"""
M-spline for cause-specific hazard. The baseline hazard evaluted at a time, t, is a linear combination of M-spline basis functions. The cumulative hazard is an I-spline. 
"""
struct _MSpline <: _SplineHazard
    hazname::Symbol
    data::Array{Float64}
    parnames::Vector{Symbol}
    statefrom::Int64
    stateto::Int64
    meshsize::Int64
    meshrange::Vector{Float64}
    hazbasis::Array{Float64}
    chazbasis::Array{Float64}
end

"""
M-spline for cause-specific hazard. The baseline hazard evaluted at a time, t, is a linear combination of M-spline basis functions. The cumulative hazard is an I-spline. Covariates have a multiplicative effect on the baseline hazard.
"""
struct _MSplinePH <: _SplineHazard
    hazname::Symbol
    data::Array{Float64}
    parnames::Vector{Symbol}
    statefrom::Int64
    stateto::Int64
    meshsize::Int64
    meshrange::Vector{Float64}
    hazbasis::Array{Float64}
    chazbasis::Array{Float64}
end

"""
Monotone increasing I-spline for cause-specific hazard. The baseline hazard evaluted at a time, t, is a linear combination of I-spline basis functions. The cumulative hazard is an C-spline. 
"""
struct _ISplineIncreasing <: _SplineHazard
    hazname::Symbol
    data::Array{Float64}
    parnames::Vector{Symbol}
    statefrom::Int64
    stateto::Int64
    meshsize::Int64
    meshrange::Vector{Float64}
    hazbasis::Array{Float64}
    chazbasis::Array{Float64}
end

"""
Monotone increasing I-spline for cause-specific hazard. The baseline hazard evaluted at a time, t, is a linear combination of I-spline basis functions. The cumulative hazard is an C-spline. Covariates have a multiplicative effect on the baseline hazard.
"""
struct _ISplineIncreasingPH <: _SplineHazard
    hazname::Symbol
    data::Array{Float64}
    parnames::Vector{Symbol}
    statefrom::Int64
    stateto::Int64
    meshsize::Int64
    meshrange::Vector{Float64}
    hazbasis::Array{Float64}
    chazbasis::Array{Float64}
end

"""
Monotone decreasing I-spline for cause-specific hazard. The baseline hazard evaluted at a time, t, is a linear combination of I-spline basis functions. The cumulative hazard is an C-spline. 
"""
struct _ISplineDecreasing <: _SplineHazard
    hazname::Symbol
    data::Array{Float64}
    parnames::Vector{Symbol}
    statefrom::Int64
    stateto::Int64
    meshsize::Int64
    meshrange::Vector{Float64}
    hazbasis::Array{Float64}
    chazbasis::Array{Float64}
end

"""
Monotone decreasing I-spline for cause-specific hazard. The baseline hazard evaluted at a time, t, is a linear combination of I-spline basis functions. The cumulative hazard is an C-spline. Covariates have a multiplicative effect on the baseline hazard.
"""
struct _ISplineDecreasingPH <: _SplineHazard
    hazname::Symbol
    data::Array{Float64}
    parnames::Vector{Symbol}
    statefrom::Int64
    stateto::Int64
    meshsize::Int64
    meshrange::Vector{Float64}
    hazbasis::Array{Float64}
    chazbasis::Array{Float64}
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
    MarkovSurrogate(hazards::Vector{_MarkovHazard}, parameters::VectorOfVectors)
"""
struct MarkovSurrogate
    hazards::Vector{_MarkovHazard}
    parameters::VectorOfVectors
end

"""
    SurrogateControl(model::MultistateProcess, statefrom, targets, uinds, ginds)

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
    SamplingWeights::Vector{Float64}
    CensoringPatterns::Matrix{Int64}
    markovsurrogate::MarkovSurrogate
    modelcall::NamedTuple
end

"""
    MultistateMarkovModel(data::DataFrame, parameters::VectorOfVectors,hazards::Vector{Union{_Exponential, _ExponentialPH}}, totalhazards::Vector{_TotalHazard},tmat::Matrix{Int64}, hazkeys::Dict{Symbol, Int64}, subjectindices::Vector{Vector{Int64}})

Struct that fully specifies a multistate Markov process with no censored state, used with panel data.
"""
struct MultistateMarkovModel <: MultistateMarkovProcess
    data::DataFrame
    parameters::VectorOfVectors 
    hazards::Vector{_MarkovHazard}
    totalhazards::Vector{_TotalHazard}
    tmat::Matrix{Int64}
    hazkeys::Dict{Symbol, Int64}
    subjectindices::Vector{Vector{Int64}}
    SamplingWeights::Vector{Float64}
    CensoringPatterns::Matrix{Int64}
    markovsurrogate::MarkovSurrogate
    modelcall::NamedTuple
end

"""
MultistateMarkovModelCensored(data::DataFrame, parameters::VectorOfVectors,hazards::Vector{_Hazard}, totalhazards::Vector{_TotalHazard},tmat::Matrix{Int64}, hazkeys::Dict{Symbol, Int64}, subjectindices::Vector{Vector{Int64}})

Struct that fully specifies a multistate Markov process with some censored states, used with panel data.
"""
struct MultistateMarkovModelCensored <: MultistateMarkovProcess
    data::DataFrame
    parameters::VectorOfVectors 
    hazards::Vector{_MarkovHazard}
    totalhazards::Vector{_TotalHazard}
    tmat::Matrix{Int64}
    emat::Matrix{Int64}
    hazkeys::Dict{Symbol, Int64}
    subjectindices::Vector{Vector{Int64}}
    SamplingWeights::Vector{Float64}
    CensoringPatterns::Matrix{Int64}
    markovsurrogate::MarkovSurrogate
    modelcall::NamedTuple
end

"""
MultistateSemiMarkovModel(data::DataFrame, parameters::VectorOfVectors,hazards::Vector{_Hazard}, totalhazards::Vector{_TotalHazard},tmat::Matrix{Int64}, hazkeys::Dict{Symbol, Int64}, subjectindices::Vector{Vector{Int64}})

Struct that fully specifies a multistate semi-Markov process with no censored state for simulation or inference. 
"""
struct MultistateSemiMarkovModel <: MultistateSemiMarkovProcess
    data::DataFrame
    parameters::VectorOfVectors 
    hazards::Vector{_Hazard}
    totalhazards::Vector{_TotalHazard}
    tmat::Matrix{Int64}
    hazkeys::Dict{Symbol, Int64}
    subjectindices::Vector{Vector{Int64}}
    SamplingWeights::Vector{Float64}
    CensoringPatterns::Matrix{Int64}
    markovsurrogate::MarkovSurrogate
    modelcall::NamedTuple
end

"""
MultistateSemiMarkovModelCensored(data::DataFrame, parameters::VectorOfVectors,hazards::Vector{_Hazard}, totalhazards::Vector{_TotalHazard},tmat::Matrix{Int64}, hazkeys::Dict{Symbol, Int64}, subjectindices::Vector{Vector{Int64}})

Struct that fully specifies a multistate semi-Markov process with some censored states for simulation or inference. 
"""
struct MultistateSemiMarkovModelCensored <: MultistateSemiMarkovProcess
    data::DataFrame
    parameters::VectorOfVectors 
    hazards::Vector{_Hazard}
    totalhazards::Vector{_TotalHazard}
    tmat::Matrix{Int64}
    emat::Matrix{Int64}
    hazkeys::Dict{Symbol, Int64}
    subjectindices::Vector{Vector{Int64}}
    SamplingWeights::Vector{Float64}
    CensoringPatterns::Matrix{Int64}
    markovsurrogate::MarkovSurrogate
    modelcall::NamedTuple
end

"""
    MultistateModelFitted(data::DataFrame, parameters::VectorOfVectors, gradient::Vector{Float64}, hazards::Vector{_Hazard}, totalhazards::Vector{_TotalHazard},tmat::Matrix{Int64}, hazkeys::Dict{Symbol, Int64}, subjectindices::Vector{Vector{Int64}})

Struct that fully specifies a fitted multistate model. 
"""
struct MultistateModelFitted <: MultistateProcess
    data::DataFrame
    parameters::VectorOfVectors 
    loglik::Float64
    vcov::Union{Nothing,Matrix{Float64}}
    hazards::Vector{_Hazard}
    totalhazards::Vector{_TotalHazard}
    tmat::Matrix{Int64}
    hazkeys::Dict{Symbol, Int64}
    subjectindices::Vector{Vector{Int64}}
    SamplingWeights::Vector{Float64}
    CensoringPatterns::Matrix{Int64}
    markovsurrogate::MarkovSurrogate
    ConvergenceRecords::Union{Nothing, NamedTuple, Optim.OptimizationResults}
    ProposedPaths::Union{Nothing, NamedTuple}
    subj_ll::Union{Nothing, Vector{Float64}}
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
    ExactData(model::MultistateProcess, samplepaths::Array{SamplePath})

Struct containing exactly observed sample paths and a model object. Used in fitting a multistate model to completely observed data.
"""
struct ExactData
    model::MultistateProcess
    paths::Array{SamplePath}
end

"""
    ExactDataAD(model::MultistateProcess, samplepaths::Array{SamplePath})

Struct containing exactly observed sample paths and a model object. Used in fitting a multistate model to completely observed data. Used for computing the variance-covariance matrix via autodiff.
"""
struct ExactDataAD
    path::Vector{SamplePath}
    samplingweight::Vector{Float64}
    hazards::Vector{<:_Hazard}
    model::MultistateProcess
end

"""
    MPanelData(model::MultistateProcess, books::Tuple)

Struct containing panel data, a model object, and bookkeeping objects. Used in fitting a multistate Markov model to panel data.
"""
struct MPanelData
    model::MultistateProcess
    books::Tuple # tpm_index and tpm_map, from build_tpm_containers
end

"""
    SMPanelData(model::MultistateProcess
    paths::Vector{Vector{SamplePath}}
    ImportanceWeights::Vector{Vector{Float64}}

Struct containing panel data, a model object, and bookkeeping objects. Used in fitting a multistate semi-Markov model to panel data via MCEM.
"""
struct SMPanelData
    model::MultistateProcess
    paths::Vector{Vector{SamplePath}}
    ImportanceWeights::Vector{Vector{Float64}}
end

