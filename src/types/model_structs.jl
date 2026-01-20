# =============================================================================
# Model and Surrogate Struct Definitions
# =============================================================================
#
# Core model structs: MultistateModel, MultistateModelFitted, MarkovSurrogate,
# and related types for model construction and fitting.
#
# =============================================================================

# =============================================================================
# Model Classification Traits
# =============================================================================
# These traits replace the previous MultistateProcess/MultistateProcess
# abstract types. Behavior is now determined by model content, not struct type.
# =============================================================================

"""
    is_markov(model::MultistateProcess) -> Bool

Check if a model has all Markov (time-homogeneous) hazards.

Returns true if all hazards are `_MarkovHazard` subtypes or degree-0 splines.
This determines whether the likelihood uses matrix exponentials (Markov panel)
or path sampling (semi-Markov MCEM).

See also: [`is_panel_data`](@ref), [`has_phasetype_expansion`](@ref)
"""
function is_markov(model::MultistateProcess)
    return all(_is_markov_hazard.(model.hazards))
end

"""
    is_panel_data(model::MultistateProcess) -> Bool

Check if a model was constructed with panel/interval-censored observation mode.

Returns true if the model uses panel observations (obstype >= 2 in data):
- obstype 1 = exact (continuously observed transitions)
- obstype 2 = panel (endpoint state observed)
- obstype > 2 = state censoring (endpoint state partially/not observed)

See also: [`is_markov`](@ref)
"""
function is_panel_data(model::MultistateProcess)
    # Check data directly: obstype 1 = exact, obstype >= 2 = panel/censored
    return any(model.data.obstype .>= 2)
end

# =============================================================================
# Surrogate Types
# =============================================================================

"""
    AbstractSurrogate

Abstract supertype for importance sampling surrogates.

All concrete subtypes must provide:
- `fitted::Bool`: Whether parameters have been fitted or are placeholders

Conventional fields (not enforced, but expected):
- `parameters`: Parameter storage (structure varies by surrogate type)

# Subtypes
- [`MarkovSurrogate`](@ref): Exponential hazard surrogate for importance sampling
- [`PhaseTypeSurrogate`](@ref): Phase-type expanded Q-matrix surrogate for FFBS

# Interface Functions
- `is_fitted(s::AbstractSurrogate)`: Check if surrogate parameters are fitted

See also: [`MarkovSurrogate`](@ref), [`PhaseTypeSurrogate`](@ref)
"""
abstract type AbstractSurrogate end

"""
    is_fitted(s::AbstractSurrogate) -> Bool

Check whether a surrogate has been fitted (parameters optimized or set).

# Examples
```julia
surrogate = fit_surrogate(model; method=:mle)
is_fitted(surrogate)  # true

surrogate_unfitted = MarkovSurrogate(hazards, params)  # fitted=false by default
is_fitted(surrogate_unfitted)  # false
```
"""
is_fitted(s::AbstractSurrogate) = s.fitted

"""
    MarkovSurrogate(hazards::Vector{_MarkovHazard}, parameters::NamedTuple; fitted::Bool=false)

Markov surrogate for importance sampling proposals in MCEM.
Uses ParameterHandling.jl for parameter management.

# Fields
- `hazards::Vector{_MarkovHazard}`: Exponential hazard functions for each transition
- `parameters::NamedTuple`: Parameter structure (flat, nested, natural, unflatten)
- `fitted::Bool`: Whether the surrogate parameters have been fitted via MLE.
  If `false`, parameters are default/placeholder values and the surrogate should be
  fitted before use in MCEM or importance sampling.

# Construction
```julia
# Unfitted surrogate (needs fitting before use)
surrogate = MarkovSurrogate(hazards, params)  # fitted=false by default

# Fitted surrogate
surrogate = MarkovSurrogate(hazards, params; fitted=true)
```

See also: [`set_surrogate!`](@ref), [`AbstractSurrogate`](@ref), [`is_fitted`](@ref)
"""
struct MarkovSurrogate <: AbstractSurrogate
    hazards::Vector{_MarkovHazard}
    parameters::NamedTuple
    fitted::Bool
    
    # Inner constructor that accepts any vector of hazards (converts to _MarkovHazard)
    function MarkovSurrogate(hazards::Vector{<:_Hazard}, parameters::NamedTuple; fitted::Bool=false)
        # Verify all hazards are Markov-compatible
        for h in hazards
            h isa _MarkovHazard || throw(ArgumentError(
                "MarkovSurrogate requires all hazards to be Markov (exponential). Got $(typeof(h))."))
        end
        new(convert(Vector{_MarkovHazard}, hazards), parameters, fitted)
    end
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

# =============================================================================
# Phase-Type Expansion (internal representation for :pt hazards)
# =============================================================================

"""
    PhaseTypeExpansion

Internal metadata for phase-type hazard expansion within a MultistateModel.

When a model contains phase-type hazards, the model's main fields (`data`, `tmat`,
`hazards`, `parameters`) operate on the **expanded** state space, while this struct
stores information needed to map back to the **observed** state space.

This design keeps the user-facing API simple: they work with a MultistateModel
and the phase-type expansion is handled internally. The `has_phasetype_expansion(m)`
trait indicates whether a model has this expansion.

# Fields
- `mappings::PhaseTypeMappings`: Bidirectional state/hazard mappings (observed ↔ expanded)
- `original_data::DataFrame`: User's original data on observed state space
- `original_tmat::Matrix{Int}`: Transition matrix on observed state space
- `original_hazards::Vector{<:HazardFunction}`: User-specified hazard specs
- `original_parameters::NamedTuple`: Parameters on observed space (for user-facing API)

# Trait-based dispatch
```julia
has_phasetype_expansion(m::MultistateProcess) = !isnothing(m.phasetype_expansion)
```

# Example
```julia
# User specifies phase-type hazard
h12 = Hazard(:pt, 1, 2)  # 3-phase Coxian for transition 1→2
model = multistatemodel(h12; data=data, n_phases=Dict(1=>3))

# Internally, model is MultistateModel on expanded space
# model.phasetype_expansion contains mappings back to observed space
has_phasetype_expansion(model)  # true
model.phasetype_expansion.mappings.n_observed  # 2 (original states)
model.phasetype_expansion.mappings.n_expanded  # 4 (with 3 phases for state 1)
```

See also: [`PhaseTypeMappings`](@ref), [`has_phasetype_expansion`](@ref)
"""
struct PhaseTypeExpansion
    mappings::AbstractPhaseTypeMappings  # PhaseTypeMappings <: AbstractPhaseTypeMappings
    original_data::DataFrame
    original_tmat::Matrix{Int}
    original_hazards::Vector{<:HazardFunction}
    original_parameters::NamedTuple
end

"""
    has_phasetype_expansion(m::MultistateProcess) -> Bool

Check if a multistate model has phase-type expansion metadata.

Returns `true` if the model was built from hazard specifications that included
phase-type (`:pt`) hazards. Such models operate internally on an expanded state
space but expose the original observed state space to users.

# Example
```julia
h12 = Hazard(:pt, 1, 2)
model = multistatemodel(h12; data=data, n_phases=Dict(1=>3))
has_phasetype_expansion(model)  # true

h12 = Hazard(:exp, 1, 2)
model = multistatemodel(h12; data=data)
has_phasetype_expansion(model)  # false
```
"""
has_phasetype_expansion(m::MultistateProcess) = hasproperty(m, :phasetype_expansion) && !isnothing(m.phasetype_expansion)

# =============================================================================
# Model Structs
# =============================================================================

"""
    MultistateModel

Unified struct for all unfitted multistate models.

Model behavior is determined by content (hazards, observation type) rather than
struct type, using traits like `is_markov()` and `is_panel_data()` for dispatch.

# Fields
- `data::DataFrame`: Long-format dataset with observations
- `parameters::NamedTuple`: Parameter structure containing:
  - `flat::Vector{Float64}` - flat parameter vector for optimizer (natural scale since v0.3.0)
  - `nested::NamedTuple` - nested parameters by hazard name with baseline/covariates fields
  - `reconstructor` - reconstructor for unflatten operations
- `hazards::Vector{_Hazard}`: Cause-specific hazard functions
- `totalhazards::Vector{_TotalHazard}`: Total hazard functions per state
- `tmat::Matrix{Int64}`: Transition matrix (allowed transitions)
- `emat::Matrix{Float64}`: Emission matrix (for censored observations)
- `hazkeys::Dict{Symbol, Int64}`: Hazard name to index mapping
- `subjectindices::Vector{Vector{Int64}}`: Data row indices per subject
- `SubjectWeights::Vector{Float64}`: Per-subject weights
- `ObservationWeights::Union{Nothing, Vector{Float64}}`: Per-observation weights
- `CensoringPatterns::Matrix{Float64}`: Censoring pattern matrix
- `markovsurrogate::Union{Nothing, MarkovSurrogate}`: MCEM importance sampling surrogate
- `modelcall::NamedTuple`: Original model construction arguments
- `phasetype_expansion::Union{Nothing, PhaseTypeExpansion}`: Phase-type expansion metadata

# Traits
Use these traits for dispatch instead of type checking:
- `is_markov(model)` - true if all hazards are Markov (for panel likelihood)
- `is_panel_data(model)` - true if panel observation mode
- `has_phasetype_expansion(model)` - true if model has phase-type hazards

# Example
```julia
# Markov model (exponential hazards, panel data)
h12 = Hazard(:exp, 1, 2)
model = multistatemodel(h12; data=data)
is_markov(model)  # true
is_panel_data(model)  # true (default for panel data)

# Semi-Markov model (Weibull hazards)
h12 = Hazard(:wei, 1, 2)
model = multistatemodel(h12; data=data)
is_markov(model)  # false

# Phase-type model
h12 = Hazard(:pt, 1, 2)
model = multistatemodel(h12; data=data, n_phases=Dict(1=>3))
is_markov(model)  # true (expanded space is Markov)
has_phasetype_expansion(model)  # true
```

See also: [`is_markov`](@ref), [`is_panel_data`](@ref), [`has_phasetype_expansion`](@ref)
"""
mutable struct MultistateModel <: MultistateProcess
    data::DataFrame
    parameters::NamedTuple  # Sole parameter storage: (flat, nested, natural, reconstructor)
    bounds::NamedTuple{(:lb, :ub), Tuple{Vector{Float64}, Vector{Float64}}}  # Parameter bounds for box-constrained optimization
    hazards::Vector{<:_Hazard}
    totalhazards::Vector{_TotalHazard}
    tmat::Matrix{Int64}
    emat::Matrix{Float64}
    hazkeys::Dict{Symbol, Int64}
    subjectindices::Vector{Vector{Int64}}
    SubjectWeights::Vector{Float64}
    ObservationWeights::Union{Nothing, Vector{Float64}}
    CensoringPatterns::Matrix{Float64}
    markovsurrogate::Union{Nothing, MarkovSurrogate}
    phasetype_surrogate::Union{Nothing, AbstractSurrogate}  # Phase-type FFBS surrogate (PhaseTypeSurrogate, built when surrogate=:phasetype)
    modelcall::NamedTuple
    phasetype_expansion::Union{Nothing, PhaseTypeExpansion}  # Phase-type expansion metadata
end

"""
    MultistateModelFitted

Struct that fully specifies a fitted multistate model.
Parameters are stored in `parameters` as (flat, nested, reconstructor).

# Penalty/Smoothing Fields (for penalized spline models)
- `smoothing_parameters`: Selected λ values from cross-validation (nothing if unpenalized)
- `edf`: Effective degrees of freedom NamedTuple with `total` and `per_term` (nothing if unpenalized)
"""
mutable struct MultistateModelFitted <: MultistateProcess
    data::DataFrame
    parameters::NamedTuple  # Sole parameter storage: (flat, nested, reconstructor)
    bounds::NamedTuple{(:lb, :ub), Tuple{Vector{Float64}, Vector{Float64}}}  # Parameter bounds for box-constrained optimization
    loglik::NamedTuple
    vcov::Union{Nothing,Matrix{Float64}}
    ij_vcov::Union{Nothing,Matrix{Float64}}  # Infinitesimal jackknife variance-covariance
    jk_vcov::Union{Nothing,Matrix{Float64}}  # Jackknife variance-covariance
    subject_gradients::Union{Nothing,Matrix{Float64}}  # Subject-level score vectors (p × n)
    hazards::Vector{<:_Hazard}
    totalhazards::Vector{_TotalHazard}
    tmat::Matrix{Int64}
    emat::Matrix{Float64}
    hazkeys::Dict{Symbol, Int64}
    subjectindices::Vector{Vector{Int64}}
    SubjectWeights::Vector{Float64}
    ObservationWeights::Union{Nothing, Vector{Float64}}
    CensoringPatterns::Matrix{Float64}
    markovsurrogate::Union{Nothing, MarkovSurrogate}
    phasetype_surrogate::Union{Nothing, AbstractSurrogate}  # Phase-type FFBS surrogate (PhaseTypeSurrogate)
    ConvergenceRecords::Union{Nothing, NamedTuple, Optim.OptimizationResults, Optim.MultivariateOptimizationResults}
    ProposedPaths::Union{Nothing, NamedTuple}
    modelcall::NamedTuple
    phasetype_expansion::Union{Nothing, PhaseTypeExpansion}  # Phase-type expansion metadata
    smoothing_parameters::Union{Nothing, Vector{Float64}}    # Selected λ from penalized fitting
    edf::Union{Nothing, NamedTuple}                          # Effective degrees of freedom
end
