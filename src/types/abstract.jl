# =============================================================================
# Abstract Type Hierarchy for Multistate Models
# =============================================================================
#
# This file defines the abstract type hierarchy used throughout MultistateModels.
# The hierarchy enables multiple dispatch for hazard evaluation, likelihood
# computation, and model fitting.
#
# =============================================================================

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
Abstract type for phase-type state space mappings.

Used as a forward reference type since PhaseTypeMappings is defined
in phasetype/types.jl which is loaded after model_structs.jl.
"""
abstract type AbstractPhaseTypeMappings end

"""
    ADBackend

Abstract type for automatic differentiation backend selection.
Enables switching between ForwardDiff (forward-mode, mutation-tolerant) and 
Enzyme (reverse-mode, mutation-free) based on problem characteristics.
"""
abstract type ADBackend end

# =============================================================================
# Penalty and Hyperparameter Selection Type Hierarchy
# =============================================================================

"""
    AbstractPenalty

Abstract type for penalty configurations in penalized likelihood fitting.

**Required Interface Methods:**
- `compute_penalty(params::AbstractVector, penalty) -> Real`: Compute penalty contribution
- `n_hyperparameters(penalty) -> Int`: Number of tuning parameters (λ values)
- `get_hyperparameters(penalty) -> Vector{Float64}`: Extract current λ values
- `set_hyperparameters(penalty, lambda::Vector{Float64}) -> AbstractPenalty`: Return new penalty with updated λ
- `hyperparameter_bounds(penalty) -> Tuple{Vector{Float64}, Vector{Float64}}`: (lb, ub) for log(λ)
- `has_penalties(penalty) -> Bool`: Whether penalty is active (n_hyperparameters > 0)

**Concrete Subtypes:**
- `NoPenalty`: Unpenalized MLE
- `QuadraticPenalty`: P(β; λ) = (1/2) Σⱼ λⱼ βⱼᵀ Sⱼ βⱼ

See also: [`NoPenalty`](@ref), [`QuadraticPenalty`](@ref)
"""
abstract type AbstractPenalty end

"""
    AbstractHyperparameterSelector

Abstract type for hyperparameter (smoothing parameter) selection strategies.

Selection functions dispatch on this type to determine which algorithm to use
for choosing optimal λ values. Selection functions return `HyperparameterSelectionResult`
containing the optimal λ and a warm-start point, NOT a fitted model.

**Concrete Subtypes:**
- `NoSelection`: Use fixed λ (no selection)
- `PIJCVSelector`: Newton-approximated LOO-CV (Wood 2024 NCV algorithm)
- `ExactCVSelector`: Exact cross-validation (requires refitting)
- `REMLSelector`: REML/EFS criterion
- `PERFSelector`: PERF criterion (Marra & Radice 2020)

See also: [`HyperparameterSelectionResult`](@ref)
"""
abstract type AbstractHyperparameterSelector end
