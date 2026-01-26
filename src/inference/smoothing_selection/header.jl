# =============================================================================
# Smoothing Parameter Selection for Penalized Splines
# =============================================================================
#
# Implements PIJCV (Predictive Infinitesimal Jackknife Cross-Validation) for 
# automatic selection of smoothing parameters λ in penalized spline models.
#
# Based on Wood (2024): "Neighbourhood Cross-Validation" arXiv:2404.16490
#
# Algorithm: Nested optimization of V(λ) where:
#   - Outer loop: optimize λ using Ipopt with ForwardDiff gradients
#   - Inner loop: for each trial λ, fit β̂(λ) via penalized MLE
#   - V(λ) is computed at the matched β̂(λ)
#
# GRADIENT APPROXIMATION NOTE:
# The gradient ∂V/∂λ computed by ForwardDiff is at FIXED β̂, ignoring the 
# implicit dependence ∂β̂/∂λ. Wood (2024) Section 2.2 shows the exact gradient
# requires: dβ̂/dρⱼ = -λⱼ H_λ⁻¹ Sⱼ β̂. This approximation works because:
# 1. At the optimum, ∂V/∂λ = 0, so the implicit term contribution is small
# 2. Ipopt is robust to approximate gradients
# 3. Function values V(λ) are exact (β̂(λ) is correctly matched)
#
# Key insight: Inner optimization uses PENALIZED loss, outer optimization
# minimizes UNPENALIZED prediction error via leave-one-out approximation.
#
# =============================================================================
# AD-SAFETY NOTES
# =============================================================================
#
# The functions in this module are designed for AD-compatibility where needed:
#
# AD-SAFE (can be differentiated through):
# - compute_penalty_from_lambda: Uses eltype(beta) for penalty accumulation
# - _fit_inner_coefficients: Uses ForwardDiff for optimization
# - pijcv_criterion: Core criterion is AD-safe (uses T = eltype)
#
# AD-UNSAFE (contain control flow/exceptions that break AD):
# - _select_hyperparameters: Outer loop dispatcher with convergence checks
# - _golden_section_search: Contains conditional logic
# - Catch blocks throughout: Return fallback values that preserve type T
#
# For catch blocks: When optimization fails during λ search, fallbacks use
# T(1e10) where T = eltype(parameters). This preserves AD type information
# even though gradients will be zero through the fallback path.
#
# =============================================================================

using LinearAlgebra
