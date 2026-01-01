# =============================================================================
# Concrete Hazard Struct Definitions
# =============================================================================
#
# Internal hazard types for runtime evaluation. Users specify hazards via
# ParametricHazard, SplineHazard, or PhaseTypeHazard (defined in 
# hazard_specs.jl), which are then converted to these internal types during
# model construction.
#
# =============================================================================

# =============================================================================
# Internal Runtime Hazard Types
# =============================================================================

"""
    MarkovHazard

Consolidated hazard type for Markov processes (time-homogeneous).
Supports exponential family hazards with optional covariates.

# Fields
- `hazname::Symbol`: Name identifier (e.g., :h12)
- `statefrom::Int64`: Origin state
- `stateto::Int64`: Destination state  
- `family::Symbol`: Distribution family (`:exp`)
- `parnames::Vector{Symbol}`: Parameter names
- `npar_baseline::Int64`: Number of baseline parameters (without covariates)
- `npar_total::Int64`: Total number of parameters (baseline + covariates)
- `hazard_fn`: Runtime-generated hazard function (t, pars, covars) -> Float64
- `cumhaz_fn`: Runtime-generated cumulative hazard function
- `has_covariates::Bool`: Whether covariates are present
- `covar_names::Vector{Symbol}`: Pre-extracted covariate names for fast lookup
- `metadata::HazardMetadata`: Tang/linpred metadata
- `shared_baseline_key::Union{Nothing,SharedBaselineKey}`: identifies Tang-sharable baselines
"""
struct MarkovHazard <: _MarkovHazard
    hazname::Symbol
    statefrom::Int64
    stateto::Int64
    family::Symbol
    parnames::Vector{Symbol}
    npar_baseline::Int64
    npar_total::Int64
    hazard_fn::Function
    cumhaz_fn::Function
    has_covariates::Bool
    covar_names::Vector{Symbol}
    metadata::HazardMetadata
    shared_baseline_key::Union{Nothing,SharedBaselineKey}
end

"""
    SemiMarkovHazard

Consolidated hazard type for semi-Markov processes (time-dependent).
Supports Weibull and Gompertz families with optional covariates.

# Fields
- `hazname::Symbol`: Name identifier (e.g., :h12)
- `statefrom::Int64`: Origin state
- `stateto::Int64`: Destination state
- `family::Symbol`: Distribution family (`:wei`, `:gom`)
- `parnames::Vector{Symbol}`: Parameter names
- `npar_baseline::Int64`: Number of baseline parameters (shape + scale)
- `npar_total::Int64`: Total number of parameters (baseline + covariates)
- `hazard_fn`: Runtime-generated hazard function (t, pars, covars) -> Float64
- `cumhaz_fn`: Runtime-generated cumulative hazard function
- `has_covariates::Bool`: Whether covariates are present
- `covar_names::Vector{Symbol}`: Pre-extracted covariate names for fast lookup
- `metadata::HazardMetadata`: Tang/linpred metadata
- `shared_baseline_key::Union{Nothing,SharedBaselineKey}`
"""
struct SemiMarkovHazard <: _SemiMarkovHazard
    hazname::Symbol
    statefrom::Int64
    stateto::Int64
    family::Symbol
    parnames::Vector{Symbol}
    npar_baseline::Int64
    npar_total::Int64
    hazard_fn::Function
    cumhaz_fn::Function
    has_covariates::Bool
    covar_names::Vector{Symbol}
    metadata::HazardMetadata
    shared_baseline_key::Union{Nothing,SharedBaselineKey}
end

"""
    RuntimeSplineHazard

Internal hazard type for spline-based hazards (runtime evaluation).
Uses B-spline basis functions for flexible baseline hazard.

This is the internal/runtime version - users specify hazards using
`SplineHazard <: HazardFunction` which is converted to this type
during model construction.

# Fields
- `hazname::Symbol`: Name identifier (e.g., :h12)
- `statefrom::Int64`: Origin state
- `stateto::Int64`: Destination state
- `family::Symbol`: Always `:sp` for splines
- `parnames::Vector{Symbol}`: Parameter names
- `npar_baseline::Int64`: Number of spline coefficients
- `npar_total::Int64`: Total number of parameters (spline + covariates)
- `hazard_fn`: Runtime-generated hazard function (t, pars, covars) -> Float64
- `cumhaz_fn`: Runtime-generated cumulative hazard function
- `has_covariates::Bool`: Whether covariates are present
- `covar_names::Vector{Symbol}`: Pre-extracted covariate names for fast lookup
- `degree::Int64`: Spline degree
- `knots::Vector{Float64}`: Knot locations
- `natural_spline::Bool`: Natural spline constraint
- `monotone::Int64`: Monotonicity constraint (0, -1, 1)
- `extrapolation::String`: Extrapolation method ("flat", "linear", or "survextrap")
- `metadata::HazardMetadata`: Tang/linpred metadata
- `shared_baseline_key::Union{Nothing,SharedBaselineKey}`
"""
struct RuntimeSplineHazard <: _SplineHazard
    hazname::Symbol
    statefrom::Int64
    stateto::Int64
    family::Symbol
    parnames::Vector{Symbol}
    npar_baseline::Int64
    npar_total::Int64
    hazard_fn::Function
    cumhaz_fn::Function
    has_covariates::Bool
    covar_names::Vector{Symbol}
    degree::Int64
    knots::Vector{Float64}
    natural_spline::Bool
    monotone::Int64
    extrapolation::String
    metadata::HazardMetadata
    shared_baseline_key::Union{Nothing,SharedBaselineKey}
end

"""
    PhaseTypeCoxianHazard <: _MarkovHazard

Runtime hazard type for phase-type (Coxian) transitions on the expanded state space.

This type represents the internal structure of a phase-type hazard after model
construction. It inherits from `_MarkovHazard` because the expanded state space
is Markovian (each phase transition is exponential).

# Coxian Structure

For a transition s → d with n phases, this hazard manages:
- Progression rates λ₁...λₙ₋₁ (between phases within origin state)
- Exit rates μ₁...μₙ (from each phase to destination state)

The expanded hazard from phase i has rate:
- λᵢ + μᵢ (for i < n)
- μₙ (for i = n, final phase)

# Fields

**Standard hazard fields:**
- `hazname::Symbol`: Name identifier (e.g., :h12)
- `statefrom::Int64`: Observed origin state
- `stateto::Int64`: Observed destination state
- `family::Symbol`: Always `:pt`
- `parnames::Vector{Symbol}`: Parameter names [λ₁...λₙ₋₁, μ₁...μₙ, covariates...]
- `npar_baseline::Int`: Baseline parameters (2n - 1)
- `npar_total::Int`: Total parameters (baseline + covariates)
- `hazard_fn::Function`: Total hazard out of current phase
- `cumhaz_fn::Function`: Cumulative hazard
- `has_covariates::Bool`: Whether covariates are present
- `covar_names::Vector{Symbol}`: Pre-extracted covariate names
- `metadata::HazardMetadata`: Tang/linpred metadata
- `shared_baseline_key`: Tang baseline sharing key

**Phase-type specific fields:**
- `n_phases::Int`: Number of Coxian phases
- `phase_index::Int`: Which phase this hazard represents (1 to n_phases)
- `is_progression::Bool`: True if this is a progression hazard (λ), false if exit (μ)
- `progression_param_indices::UnitRange{Int}`: Indices of λ parameters in parnames
- `exit_param_indices::UnitRange{Int}`: Indices of μ parameters in parnames

See also: [`PhaseTypeHazard`](@ref), [`PhaseTypeModel`](@ref)
"""
struct PhaseTypeCoxianHazard <: _MarkovHazard
    hazname::Symbol
    statefrom::Int64                 # observed state from
    stateto::Int64                   # observed state to
    family::Symbol                   # :pt
    parnames::Vector{Symbol}         # [λ₁, ..., λₙ₋₁, μ₁, ..., μₙ, covariates...]
    npar_baseline::Int64             # 2n - 1
    npar_total::Int64                # baseline + covariates
    hazard_fn::Function              # hazard function (t, pars, covars) -> rate
    cumhaz_fn::Function              # cumulative hazard function
    has_covariates::Bool
    covar_names::Vector{Symbol}
    metadata::HazardMetadata
    shared_baseline_key::Union{Nothing, SharedBaselineKey}
    
    # Phase-type specific fields
    n_phases::Int                    # number of Coxian phases
    phase_index::Int                 # which phase this hazard represents (1..n_phases)
    is_progression::Bool             # true = progression (λ), false = exit (μ)
    progression_param_indices::UnitRange{Int}  # indices of λ params (1:n-1)
    exit_param_indices::UnitRange{Int}         # indices of μ params (n:2n-1)
end

# =============================================================================
# Hazard Classification Helpers
# =============================================================================

"""
    _is_markov_hazard(hazard::_Hazard) -> Bool

Check if a hazard is Markovian (time-homogeneous).
Returns true for `_MarkovHazard` subtypes and degree-0 splines.

This is the authoritative check for model classification - models with all
Markov hazards are classified as `:markov`, otherwise `:semi_markov`.

Note: `PhaseTypeCoxianHazard <: _MarkovHazard`, so phase-type hazards are
considered Markovian (the expanded state space is Markovian).
"""
@inline function _is_markov_hazard(hazard::_Hazard)
    # MarkovHazard and PhaseTypeCoxianHazard are Markovian
    hazard isa _MarkovHazard && return true
    
    # Degree-0 splines are piecewise constant (step hazards) - Markovian
    if hazard isa RuntimeSplineHazard
        return hazard.degree == 0
    end
    
    return false
end

# =============================================================================
# Total Hazard Types
# =============================================================================

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
