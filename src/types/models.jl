# =============================================================================
# Model Type Definitions
# =============================================================================
# Concrete multistate model structs.
# Abstract types (MultistateProcess, etc.) are defined in hazards.jl.
# =============================================================================

using DataFrames
using Optim

# =============================================================================
# Concrete Model Types
# =============================================================================

"""
    MultistateModel

Struct that fully specifies a multistate process for simulation or inference, 
used in the case when sample paths are fully observed. 

# Fields
- `data::DataFrame`: Long-format dataset with observations
- `parameters::NamedTuple`: Parameter structure containing:
  - `flat::Vector{Float64}` - flat parameter vector for optimizer (log scale for baseline)
  - `nested::NamedTuple` - nested parameters by hazard name with baseline/covariates fields
  - `natural::NamedTuple` - natural scale parameters by hazard name
  - `unflatten::Function` - function to unflatten flat vector to nested structure
- `hazards::Vector{_Hazard}`: Cause-specific hazard functions
- Plus other model specification fields...
"""
mutable struct MultistateModel <: MultistateProcess
    data::DataFrame
    parameters::NamedTuple  # Sole parameter storage: (flat, nested, natural, unflatten)
    hazards::Vector{_Hazard}
    totalhazards::Vector{_TotalHazard}
    tmat::Matrix{Int64}
    emat::Matrix{Float64}
    hazkeys::Dict{Symbol, Int64}
    subjectindices::Vector{Vector{Int64}}
    SubjectWeights::Vector{Float64}
    ObservationWeights::Union{Nothing, Vector{Float64}}
    CensoringPatterns::Matrix{Float64}
    markovsurrogate::Union{Nothing, MarkovSurrogate}
    modelcall::NamedTuple
end

"""
    MultistateMarkovModel

Struct that fully specifies a multistate Markov process with no censored state, used with panel data.
Parameters are stored in `parameters` as (flat, nested, natural, unflatten).
"""
mutable struct MultistateMarkovModel <: MultistateMarkovProcess
    data::DataFrame
    parameters::NamedTuple  # Sole parameter storage: (flat, nested, natural, unflatten)
    hazards::Vector{_MarkovHazard}
    totalhazards::Vector{_TotalHazard}
    tmat::Matrix{Int64}
    emat::Matrix{Float64}
    hazkeys::Dict{Symbol, Int64}
    subjectindices::Vector{Vector{Int64}}
    SubjectWeights::Vector{Float64}
    ObservationWeights::Union{Nothing, Vector{Float64}}
    CensoringPatterns::Matrix{Float64}
    markovsurrogate::Union{Nothing, MarkovSurrogate}
    modelcall::NamedTuple
end

"""
    MultistateMarkovModelCensored

Struct that fully specifies a multistate Markov process with some censored states, used with panel data.
Parameters are stored in `parameters` as (flat, nested, natural, unflatten).
"""
mutable struct MultistateMarkovModelCensored <: MultistateMarkovProcess
    data::DataFrame
    parameters::NamedTuple  # Sole parameter storage: (flat, nested, natural, unflatten)
    hazards::Vector{_MarkovHazard}
    totalhazards::Vector{_TotalHazard}
    tmat::Matrix{Int64}
    emat::Matrix{Float64}
    hazkeys::Dict{Symbol, Int64}
    subjectindices::Vector{Vector{Int64}}
    SubjectWeights::Vector{Float64}
    ObservationWeights::Union{Nothing, Vector{Float64}}
    CensoringPatterns::Matrix{Float64}
    markovsurrogate::Union{Nothing, MarkovSurrogate}
    modelcall::NamedTuple
end

"""
    MultistateSemiMarkovModel

Struct that fully specifies a multistate semi-Markov process, used with exact death times.
Parameters are stored in `parameters` as (flat, nested, natural, unflatten).
"""
mutable struct MultistateSemiMarkovModel <: MultistateSemiMarkovProcess
    data::DataFrame
    parameters::NamedTuple  # Sole parameter storage: (flat, nested, natural, unflatten)
    hazards::Vector{_Hazard}  # Can contain both MarkovHazard and SemiMarkovHazard
    totalhazards::Vector{_TotalHazard}
    tmat::Matrix{Int64}
    emat::Matrix{Float64}
    hazkeys::Dict{Symbol, Int64}
    subjectindices::Vector{Vector{Int64}}
    SubjectWeights::Vector{Float64}
    ObservationWeights::Union{Nothing, Vector{Float64}}
    CensoringPatterns::Matrix{Float64}
    markovsurrogate::Union{Nothing, MarkovSurrogate}
    modelcall::NamedTuple
end

"""
    MultistateSemiMarkovModelCensored

Struct that fully specifies a multistate semi-Markov process with some censored states, used with panel data.
Parameters are stored in `parameters` as (flat, nested, natural, unflatten).
"""
mutable struct MultistateSemiMarkovModelCensored <: MultistateSemiMarkovProcess
    data::DataFrame
    parameters::NamedTuple  # Sole parameter storage: (flat, nested, natural, unflatten)
    hazards::Vector{_Hazard}  # Can contain both MarkovHazard and SemiMarkovHazard
    totalhazards::Vector{_TotalHazard}
    tmat::Matrix{Int64}
    emat::Matrix{Float64}
    hazkeys::Dict{Symbol, Int64}
    subjectindices::Vector{Vector{Int64}}
    SubjectWeights::Vector{Float64}
    ObservationWeights::Union{Nothing, Vector{Float64}}
    CensoringPatterns::Matrix{Float64}
    markovsurrogate::Union{Nothing, MarkovSurrogate}
    modelcall::NamedTuple
end

# =============================================================================
# Phase-Type Model
# =============================================================================

"""
    PhaseTypeModel <: MultistateMarkovProcess

A multistate model with phase-type (Coxian) hazards.

This is a wrapper around an expanded Markov model where states with `:pt` hazards
are split into multiple latent phases. The user interacts with the model in terms
of the original observed states and phase-type parameters (λ progression rates, 
μ exit rates), while internally the model operates on the expanded state space.

Phase-type models are classified as Markov processes (`<: MultistateMarkovProcess`)
because the expanded state space is Markovian - each phase transition follows an
exponential distribution.

# Fields

**Original (observed) state space:**
- `data::DataFrame`: Original data on observed states
- `parameters::NamedTuple`: Parameters in phase-type parameterization (λ, μ)
- `tmat::Matrix{Int64}`: Original transition matrix
- `hazards_spec::Vector{<:HazardFunction}`: Original user hazard specifications

**Expanded (internal) state space:**
- `expanded_data::DataFrame`: Data expanded to phase-level observations
- `expanded_parameters::NamedTuple`: Parameters for expanded Markov model
- `expanded_model::MultistateMarkovProcess`: The internal expanded Markov model
- `mappings::PhaseTypeMappings`: Bidirectional state space mappings

See also: [`PhaseTypeHazardSpec`](@ref), [`PhaseTypeMappings`](@ref)
"""
mutable struct PhaseTypeModel <: MultistateMarkovProcess
    # Expanded (internal) state space - these are standard fields for loglik compatibility
    data::DataFrame                  # Expanded data (phase-type internal states)
    tmat::Matrix{Int64}              # Expanded transition matrix
    parameters::NamedTuple           # Expanded parameters (for loglik_markov)
    expanded_model::Any              # MultistateMarkovProcess (expanded), may be nothing
    mappings::Any                    # PhaseTypeMappings (defined in phasetype.jl)
    
    # Original (observed) state space - user-facing
    original_data::DataFrame         # Original user data (observed states)
    original_tmat::Matrix{Int64}     # Original transition matrix
    original_parameters::NamedTuple  # Phase-type parameterization (λ, μ) for users
    hazards_spec::Vector{<:HazardFunction}  # Original user hazard specs
    
    # Standard model fields (on expanded space for internal operations)
    hazards::Vector{_MarkovHazard}   # Expanded hazards (all Markov)
    totalhazards::Vector{_TotalHazard}
    emat::Matrix{Float64}
    hazkeys::Dict{Symbol, Int64}
    subjectindices::Vector{Vector{Int64}}
    SubjectWeights::Vector{Float64}
    ObservationWeights::Union{Nothing, Vector{Float64}}
    CensoringPatterns::Matrix{Float64}
    markovsurrogate::Union{Nothing, MarkovSurrogate}
    modelcall::NamedTuple
end

# =============================================================================
# Fitted Model Type
# =============================================================================

"""
    MultistateModelFitted

Struct that fully specifies a fitted multistate model.
Parameters are stored in `parameters` as (flat, nested, natural, unflatten).
"""
mutable struct MultistateModelFitted <: MultistateProcess
    data::DataFrame
    parameters::NamedTuple  # Sole parameter storage: (flat, nested, natural, unflatten)
    loglik::NamedTuple
    vcov::Union{Nothing,Matrix{Float64}}
    ij_vcov::Union{Nothing,Matrix{Float64}}  # Infinitesimal jackknife variance-covariance
    jk_vcov::Union{Nothing,Matrix{Float64}}  # Jackknife variance-covariance
    subject_gradients::Union{Nothing,Matrix{Float64}}  # Subject-level score vectors (p × n)
    hazards::Vector{_Hazard}
    totalhazards::Vector{_TotalHazard}
    tmat::Matrix{Int64}
    emat::Matrix{Float64}
    hazkeys::Dict{Symbol, Int64}
    subjectindices::Vector{Vector{Int64}}
    SubjectWeights::Vector{Float64}
    ObservationWeights::Union{Nothing, Vector{Float64}}
    CensoringPatterns::Matrix{Float64}
    markovsurrogate::Union{Nothing, MarkovSurrogate}
    ConvergenceRecords::Union{Nothing, NamedTuple, Optim.OptimizationResults, Optim.MultivariateOptimizationResults}
    ProposedPaths::Union{Nothing, NamedTuple}
    modelcall::NamedTuple
end
