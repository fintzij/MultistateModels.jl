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
