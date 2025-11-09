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

#=============================================================================
PHASE 2: Consolidated Hazard Types (3 types instead of 8)
=============================================================================# 

"""
    MarkovHazard

Consolidated hazard type for Markov processes (time-homogeneous).
Supports exponential family hazards with optional covariates.

# Fields
- `hazname::Symbol`: Name identifier (e.g., :h12)
- `statefrom::Int64`: Origin state
- `stateto::Int64`: Destination state  
- `family::String`: Distribution family ("exp")
- `parnames::Vector{Symbol}`: Parameter names
- `npar_baseline::Int64`: Number of baseline parameters (without covariates)
- `npar_total::Int64`: Total number of parameters (baseline + covariates)
- `hazard_fn`: Runtime-generated hazard function (t, pars, covars) -> Float64
- `cumhaz_fn`: Runtime-generated cumulative hazard function
- `has_covariates::Bool`: Whether covariates are present
"""
struct MarkovHazard <: _MarkovHazard
    hazname::Symbol
    statefrom::Int64
    stateto::Int64
    family::String
    parnames::Vector{Symbol}
    npar_baseline::Int64
    npar_total::Int64
    hazard_fn::Function
    cumhaz_fn::Function
    has_covariates::Bool
end

"""
    SemiMarkovHazard

Consolidated hazard type for semi-Markov processes (time-dependent).
Supports Weibull and Gompertz families with optional covariates.

# Fields
- `hazname::Symbol`: Name identifier (e.g., :h12)
- `statefrom::Int64`: Origin state
- `stateto::Int64`: Destination state
- `family::String`: Distribution family ("wei", "gom")
- `parnames::Vector{Symbol}`: Parameter names
- `npar_baseline::Int64`: Number of baseline parameters (shape + scale)
- `npar_total::Int64`: Total number of parameters (baseline + covariates)
- `hazard_fn`: Runtime-generated hazard function (t, pars, covars) -> Float64
- `cumhaz_fn`: Runtime-generated cumulative hazard function
- `has_covariates::Bool`: Whether covariates are present
"""
struct SemiMarkovHazard <: _SemiMarkovHazard
    hazname::Symbol
    statefrom::Int64
    stateto::Int64
    family::String
    parnames::Vector{Symbol}
    npar_baseline::Int64
    npar_total::Int64
    hazard_fn::Function
    cumhaz_fn::Function
    has_covariates::Bool
end

"""
    SplineHazard

Consolidated hazard type for spline-based hazards.
Uses B-spline basis functions for flexible baseline hazard.

# Fields
- `hazname::Symbol`: Name identifier (e.g., :h12)
- `statefrom::Int64`: Origin state
- `stateto::Int64`: Destination state
- `family::String`: Always "sp" for splines
- `parnames::Vector{Symbol}`: Parameter names
- `npar_baseline::Int64`: Number of spline coefficients
- `npar_total::Int64`: Total number of parameters (spline + covariates)
- `hazard_fn`: Runtime-generated hazard function (t, pars, covars) -> Float64
- `cumhaz_fn`: Runtime-generated cumulative hazard function
- `has_covariates::Bool`: Whether covariates are present
- `degree::Int64`: Spline degree
- `knots::Vector{Float64}`: Knot locations
- `natural_spline::Bool`: Natural spline constraint
- `monotone::Int64`: Monotonicity constraint (0, -1, 1)
end
"""
struct SplineHazard <: _SplineHazard
    hazname::Symbol
    statefrom::Int64
    stateto::Int64
    family::String
    parnames::Vector{Symbol}
    npar_baseline::Int64
    npar_total::Int64
    hazard_fn::Function
    cumhaz_fn::Function
    has_covariates::Bool
    degree::Int64
    knots::Vector{Float64}
    natural_spline::Bool
    monotone::Int64
end

#=============================================================================
OLD HAZARD TYPES (Will be deprecated after Phase 2)
=============================================================================#

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
    SplineHazard(haz::StatsModels.FormulaTerm, family::string, statefrom::Int64, stateto::Int64; df::Union{Int64,Nothing}, degree::Int64, knots::Union{Vector{Float64},Float64,Nothing}, boundaryknots::Union{Vector{Float64},Nothing}, extrapolation::String, natural_spline::Bool)

Specify a cause-specific baseline hazard. 

# Arguments
- `hazard`: regression formula for the (log) hazard, parsed using StatsModels.jl.
- `family`: "sp" for splines for the baseline hazard.
- `statefrom`: state number for the origin state.
- `stateto`: state number for the destination state.
- `df`: Degrees of freedom.
- `degree`: Degree of the spline polynomial basis.
- `knots`: Vector of knot locations.
- `boundaryknots`: Vector of boundary knot locations
- `extrapolation`: Either "linear" or "flat"
- `natural_spline`: Restrict the second derivative to zero at the boundaries (natural spline).
- `monotone`: 
"""
struct SplineHazard <: HazardFunction
    hazard::StatsModels.FormulaTerm   # StatsModels.jl formula
    family::String     # "sp" for splines
    statefrom::Int64   # starting state number
    stateto::Int64     # destination state number
    degree::Int64
    knots::Union{Nothing,Float64,Vector{Float64}}
    boundaryknots::Union{Nothing,Vector{Float64}}
    extrapolation::String
    natural_spline::Bool
    monotone::Int64
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

# New ParameterHandling fields (Phase 1):
- `parameters_ph`: NamedTuple containing:
  - `flat::Vector{Float64}` - flat parameter vector for optimizer
  - `transformed` - ParameterHandling transformed parameters (with positive() etc.)
  - `natural::NamedTuple` - natural scale parameters by hazard name
  - `unflatten::Function` - function to unflatten flat vector

# Phase 3 Change: Now mutable to allow `parameters_ph` updates
"""
mutable struct MultistateModel <: MultistateProcess
    data::DataFrame
    parameters::VectorOfVectors  # Legacy - keep for Phase 1 compatibility
    parameters_ph::NamedTuple  # NEW: Nested ParameterHandling structure
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
    MultistateMarkovModel(data::DataFrame, parameters::VectorOfVectors,hazards::Vector{Union{_Exponential, _ExponentialPH}}, totalhazards::Vector{_TotalHazard},tmat::Matrix{Int64}, hazkeys::Dict{Symbol, Int64}, subjectindices::Vector{Vector{Int64}})

Struct that fully specifies a multistate Markov process with no censored state, used with panel data.

# Phase 3 Change: Now mutable to allow `parameters_ph` updates
"""
mutable struct MultistateMarkovModel <: MultistateMarkovProcess
    data::DataFrame
    parameters::VectorOfVectors  # Legacy - keep for Phase 1 compatibility
    parameters_ph::NamedTuple  # Nested: (flat, transformed, natural, unflatten)
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
MultistateMarkovModelCensored(data::DataFrame, parameters::VectorOfVectors,hazards::Vector{_Hazard}, totalhazards::Vector{_TotalHazard},tmat::Matrix{Int64}, hazkeys::Dict{Symbol, Int64}, subjectindices::Vector{Vector{Int64}})

Struct that fully specifies a multistate Markov process with some censored states, used with panel data.

# Phase 3 Change: Now mutable to allow `parameters_ph` updates
"""
mutable struct MultistateMarkovModelCensored <: MultistateMarkovProcess
    data::DataFrame
    parameters::VectorOfVectors  # Legacy - keep for Phase 1 compatibility
    parameters_ph::NamedTuple  # Nested: (flat, transformed, natural, unflatten)
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

Struct that fully specifies a multistate semi-Markov process, used with exact death times.

# Phase 3 Change: Now mutable to allow `parameters_ph` updates
"""
mutable struct MultistateSemiMarkovModel <: MultistateSemiMarkovProcess
    data::DataFrame
    parameters::VectorOfVectors  # Legacy - keep for Phase 1 compatibility
    parameters_ph::NamedTuple  # Nested: (flat, transformed, natural, unflatten)
    hazards::Vector{_Hazard}  # Can contain both MarkovHazard and SemiMarkovHazard
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
MultistateSemiMarkovModelCensored(data::DataFrame, parameters::VectorOfVectors,hazards::Vector{_Hazard}, totalhazards::Vector{_TotalHazard},tmat::Matrix{Int64}, hazkeys::Dict{Symbol, Int64}, subjectindices::Vector{Vector{Int64}})

Struct that fully specifies a multistate semi-Markov process with some censored states, used with panel data.

# Phase 3 Change: Now mutable to allow `parameters_ph` updates
"""
mutable struct MultistateSemiMarkovModelCensored <: MultistateSemiMarkovProcess
    data::DataFrame
    parameters::VectorOfVectors  # Legacy - keep for Phase 1 compatibility
    parameters_ph::NamedTuple  # Nested: (flat, transformed, natural, unflatten)
    hazards::Vector{_Hazard}  # Can contain both MarkovHazard and SemiMarkovHazard
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

# Phase 3 Change: Now mutable (though less critical for fitted models)
"""
mutable struct MultistateModelFitted <: MultistateProcess
    data::DataFrame
    parameters::VectorOfVectors  # Legacy - keep for Phase 1 compatibility
    parameters_ph::NamedTuple  # NEW Phase 3: (flat, transformed, natural, unflatten)
    loglik::NamedTuple
    vcov::Union{Nothing,Matrix{Float64}}
    hazards::Vector{_Hazard}
    totalhazards::Vector{_TotalHazard}
    tmat::Matrix{Int64}
    emat::Matrix{Int64}
    hazkeys::Dict{Symbol, Int64}
    subjectindices::Vector{Vector{Int64}}
    SamplingWeights::Vector{Float64}
    CensoringPatterns::Matrix{Int64}
    markovsurrogate::MarkovSurrogate
    ConvergenceRecords::Union{Nothing, NamedTuple, Optim.OptimizationResults, Optim.MultivariateOptimizationResults}
    ProposedPaths::Union{Nothing, NamedTuple}
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
    SamplePath(subj, times, states) = length(times) != length(states) ? error("Number of times in a jump chain must equal the number of states.") : new(subj, times, states)
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
