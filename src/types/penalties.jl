# =============================================================================
# Penalty Types for Penalized Likelihood Fitting
# =============================================================================
#
# This file defines the type hierarchy and interface implementations for 
# penalized likelihood fitting in multistate models.
#
# Type Hierarchy:
#   AbstractPenalty (from abstract.jl)
#   ├── NoPenalty          # Unpenalized MLE
#   └── QuadraticPenalty   # P(β; λ) = (1/2) Σⱼ λⱼ βⱼᵀ Sⱼ βⱼ
#
#   AbstractHyperparameterSelector (from abstract.jl)
#   ├── NoSelection        # Fixed λ (no selection)
#   ├── PIJCVSelector      # Newton-approximated LOO-CV (Wood 2024)
#   ├── ExactCVSelector    # Exact cross-validation
#   ├── REMLSelector       # REML/EFS criterion
#   └── PERFSelector       # PERF criterion (Marra & Radice 2020)
#
#   PenaltyWeighting (for adaptive penalty weighting)
#   ├── UniformWeighting   # Standard P-spline (uniform weight)
#   └── AtRiskWeighting    # Weight by inverse at-risk count: w(t) = Y(t)^(-α)
#
# =============================================================================

using LinearAlgebra: dot

# =============================================================================
# Penalty Weighting Types
# =============================================================================

"""
    PenaltyWeighting

Abstract type for penalty weight specifications.

Penalty weighting allows the penalty strength to vary across time, adapting to
local data density. Standard P-splines use uniform weighting; adaptive methods
can weight by the inverse at-risk count to penalize more heavily where fewer
subjects contribute information.

Subtypes:
- [`UniformWeighting`](@ref): Standard P-spline with uniform penalty weight
- [`AtRiskWeighting`](@ref): Adaptive weighting by inverse at-risk count

See also: [`SplinePenalty`](@ref), [`UniformWeighting`](@ref), [`AtRiskWeighting`](@ref)
"""
abstract type PenaltyWeighting end

"""
    UniformWeighting <: PenaltyWeighting

No time-varying weights (standard P-spline).

This is the default weighting for spline penalties, where the penalty is:
P(β) = λ ∫₀ᵀ [f''(t)]² dt = λ βᵀSβ

with the same weight applied uniformly across all times.

See also: [`AtRiskWeighting`](@ref), [`PenaltyWeighting`](@ref)
"""
struct UniformWeighting <: PenaltyWeighting end

"""
    AtRiskWeighting <: PenaltyWeighting

Weight penalty by inverse at-risk count: w(t) = Y(t)^(-α)

This produces an adaptive penalty where times with fewer subjects at risk
receive higher penalty (more shrinkage toward smoothness), while times with
more subjects at risk allow more flexibility.

# Fields
- `alpha::Float64`: Power on at-risk count (default 1.0). Must be ≥ 0.
  - α = 0: Equivalent to uniform weighting
  - α = 1: Weight proportional to 1/Y(t) (default)
  - α > 1: Stronger adaptation (rarely needed)
- `learn::Bool`: Whether to estimate α from data via marginal likelihood (default false)

# Penalty Formula
P(β; α) = λ ∫₀ᵀ w(t; α) [f''(t)]² dt = λ βᵀS_w(α)β

where S_w is the weighted penalty matrix computed using w(t) = Y(t)^(-α).

# Examples
```julia
AtRiskWeighting()                    # Default: α=1.0, learn=false
AtRiskWeighting(alpha=0.5)           # Weaker adaptation
AtRiskWeighting(alpha=1.0, learn=true)  # Estimate α from data
```

See also: [`UniformWeighting`](@ref), [`PenaltyWeighting`](@ref), [`SplinePenalty`](@ref)
"""
struct AtRiskWeighting <: PenaltyWeighting
    alpha::Float64
    learn::Bool
    
    function AtRiskWeighting(; alpha::Float64=1.0, learn::Bool=false)
        alpha >= 0 || throw(ArgumentError("alpha must be non-negative, got $alpha"))
        new(alpha, learn)
    end
end

# =============================================================================
# User-facing Penalty Specification
# =============================================================================

"""
    SplinePenalty

User-facing configuration for spline penalties in baseline hazards and smooth covariates.

Penalties are configured at the model level using a rule-based API.
Rules are applied from general to specific: transition > origin > global.

# Arguments
- `selector`: Which hazards this rule applies to.
  - `:all` — All spline hazards (default for global settings)
  - `r::Int` — All hazards from origin state `r`
  - `(r, s)::Tuple{Int,Int}` — Specific transition `r → s`
- `order::Int=2`: Derivative to penalize (1=slope, 2=curvature, 3=change in curvature)
- `total_hazard::Bool=false`: Penalize smoothness of total hazard out of this origin
- `share_lambda::Bool=false`: Share λ across competing hazards from same origin
- `share_covariate_lambda::Union{Bool, Symbol}=false`: Sharing mode for smooth covariate λ
  - `false` — Separate λ per smooth term (default)
  - `:hazard` — Share λ across all s() terms within each hazard
  - `:global` — Share λ across all s() terms in the model
- `adaptive_weight::Symbol=:none`: Penalty weighting strategy
  - `:none` — Uniform weighting (standard P-spline, default)
  - `:atrisk` — Weight by inverse at-risk count w(t) = Y(t)^(-α)
- `alpha::Float64=1.0`: Power on at-risk count when `adaptive_weight=:atrisk`
- `learn_alpha::Bool=false`: Estimate α from data via marginal likelihood

# Examples
```julia
# Global curvature penalty (default)
penalty = SplinePenalty()

# Stiffer penalty (3rd derivative) for all hazards
penalty = SplinePenalty(order=3)

# Origin 1: shared λ across competing risks
penalty = SplinePenalty(1, share_lambda=true)

# Transition 1→2: first derivative (flatness) penalty
penalty = SplinePenalty((1, 2), order=1)

# Share smooth covariate λ within each hazard
penalty = SplinePenalty(share_covariate_lambda=:hazard)

# Share smooth covariate λ globally across all hazards
penalty = SplinePenalty(share_covariate_lambda=:global)

# At-risk adaptive weighting (penalize more where fewer subjects at risk)
penalty = SplinePenalty(adaptive_weight=:atrisk)

# At-risk weighting with custom power
penalty = SplinePenalty(adaptive_weight=:atrisk, alpha=0.5)

# Estimate optimal α from data
penalty = SplinePenalty(adaptive_weight=:atrisk, learn_alpha=true)
```

# Notes
- When `share_lambda=true` or `total_hazard=true`, all spline hazards from that 
  origin MUST have identical knot locations. The constructor validates this.
- For competing risks, the likelihood-motivated decomposition suggests separate
  control of total hazard and deviation penalties. Use `total_hazard=true` to
  enable the additional total hazard penalty term.
- Smooth covariate sharing (`share_covariate_lambda`) is independent of baseline
  hazard sharing (`share_lambda`).
- When `share_lambda=true`, penalties that share λ also share α.
- The `alpha` and `learn_alpha` arguments are only used when `adaptive_weight=:atrisk`.

See also: [`PenaltyConfig`](@ref), [`QuadraticPenalty`](@ref), [`PenaltyWeighting`](@ref)
"""
struct SplinePenalty
    selector::Union{Symbol, Int, Tuple{Int,Int}}
    order::Int
    total_hazard::Bool
    share_lambda::Bool
    share_covariate_lambda::Union{Bool, Symbol}
    weighting::PenaltyWeighting
    
    function SplinePenalty(selector::Union{Symbol, Int, Tuple{Int,Int}}=:all;
                            order::Int=2, total_hazard::Bool=false, 
                            share_lambda::Bool=false,
                            share_covariate_lambda::Union{Bool, Symbol}=false,
                            adaptive_weight::Symbol=:none,
                            alpha::Float64=1.0,
                            learn_alpha::Bool=false)
        order >= 1 || throw(ArgumentError("order must be ≥ 1, got $order"))
        
        # Validate selector type
        if selector isa Symbol && selector != :all
            throw(ArgumentError("Symbol selector must be :all, got :$selector"))
        end
        if selector isa Int && selector < 1
            throw(ArgumentError("Origin state selector must be ≥ 1, got $selector"))
        end
        if selector isa Tuple{Int,Int}
            selector[1] >= 1 && selector[2] >= 1 || 
                throw(ArgumentError("Transition selector states must be ≥ 1, got $selector"))
        end
        
        # Validate share_covariate_lambda
        if share_covariate_lambda isa Symbol && share_covariate_lambda ∉ (:hazard, :global)
            throw(ArgumentError("share_covariate_lambda must be false, :hazard, or :global, got :$share_covariate_lambda"))
        end
        
        # Validate adaptive_weight and construct weighting
        weighting = if adaptive_weight == :none
            UniformWeighting()
        elseif adaptive_weight == :atrisk
            AtRiskWeighting(alpha=alpha, learn=learn_alpha)
        else
            throw(ArgumentError("adaptive_weight must be :none or :atrisk, got :$adaptive_weight"))
        end
        
        new(selector, order, total_hazard, share_lambda, share_covariate_lambda, weighting)
    end
end

# =============================================================================
# Internal Penalty Term Types
# =============================================================================

"""
    PenaltyTerm

Internal representation of a single penalty term in the penalized likelihood.

# Fields
- `hazard_indices::UnitRange{Int}`: Indices into flat parameter vector
- `S::Matrix{Float64}`: Penalty matrix (K × K)
- `lambda::Float64`: Current smoothing parameter
- `order::Int`: Derivative order
- `hazard_names::Vector{Symbol}`: Names of hazards covered by this term

# Notes
Parameters are on natural scale with box constraints. The penalty is quadratic:
P(β) = (λ/2) βᵀSβ
"""
struct PenaltyTerm
    hazard_indices::UnitRange{Int}
    S::Matrix{Float64}
    lambda::Float64
    order::Int
    hazard_names::Vector{Symbol}
end

"""
    TotalHazardPenaltyTerm

Internal representation of a total hazard penalty term (for competing risks).

The total hazard penalty penalizes the curvature of H(t) = Σₖ hₖ(t) where
the sum is over all competing hazards from the same origin state.

# Fields
- `origin::Int`: Origin state for this penalty
- `hazard_indices::Vector{UnitRange{Int}}`: Indices for each competing hazard
- `S::Matrix{Float64}`: Base penalty matrix (shared across hazards)
- `lambda_H::Float64`: Smoothing parameter for total hazard
- `order::Int`: Derivative order
"""
struct TotalHazardPenaltyTerm
    origin::Int
    hazard_indices::Vector{UnitRange{Int}}
    S::Matrix{Float64}
    lambda_H::Float64
    order::Int
end

"""
    SmoothCovariatePenaltyTerm

Internal representation of a smooth covariate penalty term (for s(x) terms).

# Fields
- `param_indices::Vector{Int}`: Indices into flat parameter vector (may not be contiguous)
- `S::Matrix{Float64}`: Penalty matrix (K × K)
- `lambda::Float64`: Smoothing parameter
- `order::Int`: Derivative order (from smooth term)
- `label::String`: Label for the smooth term (e.g., "s(age)")
- `hazard_name::Symbol`: Name of hazard containing this smooth term
"""
struct SmoothCovariatePenaltyTerm
    param_indices::Vector{Int}
    S::Matrix{Float64}
    lambda::Float64
    order::Int
    label::String
    hazard_name::Symbol
end

# =============================================================================
# Concrete Penalty Types
# =============================================================================

"""
    NoPenalty <: AbstractPenalty

No penalty (unpenalized maximum likelihood estimation).

Use this for standard MLE without any regularization.
"""
struct NoPenalty <: AbstractPenalty end

"""
    QuadraticPenalty <: AbstractPenalty

Quadratic penalty for penalized likelihood fitting.

The penalty contribution to negative log-likelihood is:
    P(β; λ) = (1/2) Σⱼ λⱼ βⱼᵀ Sⱼ βⱼ + (1/2) Σᵣ λ_H,r β_rᵀ (𝟙𝟙ᵀ ⊗ S) β_r + (1/2) Σₖ λₖ βₖᵀ Sₖ βₖ

where the three terms correspond to baseline hazard penalties, total hazard penalties,
and smooth covariate penalties respectively.

# Fields
- `terms::Vector{PenaltyTerm}`: Individual baseline hazard penalty terms
- `total_hazard_terms::Vector{TotalHazardPenaltyTerm}`: Total hazard penalties
- `smooth_covariate_terms::Vector{SmoothCovariatePenaltyTerm}`: Smooth covariate penalties
- `shared_lambda_groups::Dict{Int, Vector{Int}}`: Origin → indices of shared-λ terms (baseline)
- `shared_smooth_groups::Vector{Vector{Int}}`: Groups of smooth covariate term indices sharing λ
- `n_lambda::Int`: Total number of smoothing parameters

# Notes
This type is created by `build_penalty_config()` from user-facing `SplinePenalty` rules.
It should not be constructed directly by users.

See also: [`SplinePenalty`](@ref), [`build_penalty_config`](@ref)
"""
struct QuadraticPenalty <: AbstractPenalty
    terms::Vector{PenaltyTerm}
    total_hazard_terms::Vector{TotalHazardPenaltyTerm}
    smooth_covariate_terms::Vector{SmoothCovariatePenaltyTerm}
    shared_lambda_groups::Dict{Int, Vector{Int}}
    shared_smooth_groups::Vector{Vector{Int}}
    n_lambda::Int
    
    function QuadraticPenalty(terms::Vector{PenaltyTerm},
                              total_hazard_terms::Vector{TotalHazardPenaltyTerm},
                              smooth_covariate_terms::Vector{SmoothCovariatePenaltyTerm},
                              shared_lambda_groups::Dict{Int, Vector{Int}},
                              shared_smooth_groups::Vector{Vector{Int}},
                              n_lambda::Int)
        n_lambda >= 0 || throw(ArgumentError("n_lambda must be ≥ 0"))
        new(terms, total_hazard_terms, smooth_covariate_terms, shared_lambda_groups, shared_smooth_groups, n_lambda)
    end
    
    # 5-argument constructor (no shared_smooth_groups) for backward compatibility
    function QuadraticPenalty(terms::Vector{PenaltyTerm},
                              total_hazard_terms::Vector{TotalHazardPenaltyTerm},
                              smooth_covariate_terms::Vector{SmoothCovariatePenaltyTerm},
                              shared_lambda_groups::Dict{Int, Vector{Int}},
                              n_lambda::Int)
        n_lambda >= 0 || throw(ArgumentError("n_lambda must be ≥ 0"))
        new(terms, total_hazard_terms, smooth_covariate_terms, shared_lambda_groups, Vector{Vector{Int}}(), n_lambda)
    end
    
    # Legacy 4-argument constructor for backward compatibility
    function QuadraticPenalty(terms::Vector{PenaltyTerm},
                              total_hazard_terms::Vector{TotalHazardPenaltyTerm},
                              shared_lambda_groups::Dict{Int, Vector{Int}},
                              n_lambda::Int)
        n_lambda >= 0 || throw(ArgumentError("n_lambda must be ≥ 0"))
        new(terms, total_hazard_terms, SmoothCovariatePenaltyTerm[], shared_lambda_groups, Vector{Vector{Int}}(), n_lambda)
    end
end

"""
    QuadraticPenalty()

Create an empty quadratic penalty configuration (no penalties).
"""
QuadraticPenalty() = QuadraticPenalty(PenaltyTerm[], TotalHazardPenaltyTerm[], SmoothCovariatePenaltyTerm[], Dict{Int,Vector{Int}}(), Vector{Vector{Int}}(), 0)

# =============================================================================
# Legacy Alias: PenaltyConfig
# =============================================================================

"""
    PenaltyConfig

Alias for `QuadraticPenalty` for backward compatibility.

See [`QuadraticPenalty`](@ref) for documentation.
"""
const PenaltyConfig = QuadraticPenalty

# =============================================================================
# Hyperparameter Selector Types
# =============================================================================

"""
    NoSelection <: AbstractHyperparameterSelector

Use fixed hyperparameters (no selection).

Use this when:
- Fitting unpenalized models (with `NoPenalty`)
- Using pre-specified λ values without optimization
"""
struct NoSelection <: AbstractHyperparameterSelector end

"""
    PIJCVSelector <: AbstractHyperparameterSelector

Newton-approximated leave-one-out cross-validation (Wood 2024 NCV algorithm).

This is the default and recommended method for smoothing parameter selection.
It uses a nested optimization approach where the outer loop minimizes the
approximate CV criterion using Newton steps, and the inner loop fits
coefficients at each trial λ value.

# Fields
- `nfolds::Int`: Number of folds (0 = leave-one-out, k = k-fold approximation)
- `use_quadratic::Bool`: If true, use fast V_q approximation instead of actual likelihood
- `use_implicit_diff::Bool`: If true (DEFAULT), use ImplicitDifferentiation.jl for efficient gradients
  (avoids nested AD, ~15-20x speedup). Set to false only for debugging.
- `gamma::Float64`: Robustness parameter (default=1.0). Wood Section 4.1: γ > 1 extrapolates the 
  LOO perturbation, adding a stability penalty. Common choices: γ=1.0 (standard), γ=1.4 (robust).

# Examples
```julia
PIJCVSelector()                   # Leave-one-out with implicit differentiation (DEFAULT)
PIJCVSelector(5)                  # 5-fold approximation
PIJCVSelector(0, true)            # LOO with fast quadratic approximation V_q
PIJCVSelector(0, true, true, 1.4) # Robust V_q with γ=1.4 (Wood Section 4.1)
PIJCVSelector(0, false, false)    # LOO without implicit diff (legacy, not recommended)
```

# Performance Notes
When `use_implicit_diff=true` (default):
- Uses implicit function theorem: ∂β̂/∂ρ = -H_λ⁻¹ · (λⱼ Sⱼ β̂)
- Avoids differentiating through the inner optimization
- Reduces complexity from O(np³) to O(np²)
- Expected 15-20x speedup and 10x memory reduction

# References
- Wood, S.N. (2024). "On Neighbourhood Cross Validation." arXiv:2404.16490
- Blondel et al. (2022). "Efficient and Modular Implicit Differentiation."
"""
struct PIJCVSelector <: AbstractHyperparameterSelector
    nfolds::Int
    use_quadratic::Bool  # If true, use fast V_q approximation instead of actual likelihood
    use_implicit_diff::Bool  # If true, use ImplicitDifferentiation.jl for efficient gradients
    
    # Default use_implicit_diff=true for efficient gradient computation via IFT
    function PIJCVSelector(nfolds::Int=0, use_quadratic::Bool=false, use_implicit_diff::Bool=true)
        nfolds >= 0 || throw(ArgumentError("nfolds must be ≥ 0"))
        new(nfolds, use_quadratic, use_implicit_diff)
    end
end

"""
    ExactCVSelector <: AbstractHyperparameterSelector

Exact cross-validation for smoothing parameter selection.

This method refits the model on each fold, providing exact (rather than
approximated) CV scores. More computationally expensive than `PIJCVSelector`.

# Fields
- `nfolds::Int`: Number of folds (0 = leave-one-out/n-fold, k = k-fold)

# Examples
```julia
ExactCVSelector()     # Leave-one-out CV
ExactCVSelector(5)    # 5-fold CV
ExactCVSelector(10)   # 10-fold CV
```
"""
struct ExactCVSelector <: AbstractHyperparameterSelector
    nfolds::Int
    
    function ExactCVSelector(nfolds::Int=0)
        nfolds >= 0 || throw(ArgumentError("nfolds must be ≥ 0"))
        new(nfolds)
    end
end

"""
    REMLSelector <: AbstractHyperparameterSelector

REML-based (Restricted Maximum Likelihood) smoothing parameter selection.

Also known as EFS (Extended Fellner-Schall) method. Provides a computationally
efficient alternative to cross-validation that tends to work well when the
assumed model structure is correct.

# References
- Fellner, W.H. (1986). Robust estimation of variance components.
- Schall, R. (1991). Estimation in generalized linear models with random effects.
"""
struct REMLSelector <: AbstractHyperparameterSelector end

"""
    PERFSelector <: AbstractHyperparameterSelector

PERF criterion for smoothing parameter selection (Marra & Radice 2020).

An alternative model selection criterion that balances fit and complexity.

# References
- Marra, G. & Radice, R. (2020). "PERF: Prediction error using random forests."
"""
struct PERFSelector <: AbstractHyperparameterSelector end

# =============================================================================
# Result Types
# =============================================================================

"""
    HyperparameterSelectionResult

Result from hyperparameter (smoothing parameter) selection.

This type is returned by selection functions like `_select_hyperparameters`.
It contains the optimal λ values and a warm-start point for the final fit,
but NOT the final fitted coefficients.

# Fields
- `lambda::Vector{Float64}`: Optimal smoothing parameters
- `warmstart_beta::Vector{Float64}`: Warm-start coefficients for final fit
- `penalty::AbstractPenalty`: Penalty configuration with optimal λ set
- `criterion_value::Float64`: Value of selection criterion at optimum
- `edf::NamedTuple`: Effective degrees of freedom (total=, per_term=)
- `converged::Bool`: Whether selection converged
- `method::Symbol`: Selection method used (:pijcv, :exactcv, :reml, :perf)
- `n_iterations::Int`: Number of iterations
- `diagnostics::NamedTuple`: Additional diagnostic information

# Notes
The `warmstart_beta` field contains coefficients from the last inner optimization
during selection. These are used to warm-start the final fit but are NOT the
final fitted coefficients. The final fit is always performed by
`_fit_coefficients_at_fixed_hyperparameters`.

See also: [`_select_hyperparameters`](@ref), [`AbstractHyperparameterSelector`](@ref)
"""
struct HyperparameterSelectionResult
    lambda::Vector{Float64}
    warmstart_beta::Vector{Float64}
    penalty::AbstractPenalty
    criterion_value::Float64
    edf::NamedTuple
    converged::Bool
    method::Symbol
    n_iterations::Int
    diagnostics::NamedTuple
end

# =============================================================================
# Interface Implementations: NoPenalty
# =============================================================================

"""
    compute_penalty(params::AbstractVector, ::NoPenalty) -> Float64

Return 0.0 (no penalty contribution).
"""
compute_penalty(::AbstractVector, ::NoPenalty) = 0.0

"""
    n_hyperparameters(::NoPenalty) -> Int

Return 0 (no hyperparameters).
"""
n_hyperparameters(::NoPenalty) = 0

"""
    get_hyperparameters(::NoPenalty) -> Vector{Float64}

Return empty vector (no hyperparameters).
"""
get_hyperparameters(::NoPenalty) = Float64[]

"""
    set_hyperparameters(p::NoPenalty, ::Vector{Float64}) -> NoPenalty

Return the penalty unchanged (no hyperparameters to set).
"""
set_hyperparameters(p::NoPenalty, ::Vector{Float64}) = p

"""
    hyperparameter_bounds(::NoPenalty) -> Tuple{Vector{Float64}, Vector{Float64}}

Return empty bounds (no hyperparameters).
"""
hyperparameter_bounds(::NoPenalty) = (Float64[], Float64[])

"""
    has_penalties(::NoPenalty) -> Bool

Return false (no penalty contribution).
"""
has_penalties(::NoPenalty) = false

# =============================================================================
# Interface Implementations: QuadraticPenalty
# =============================================================================

"""
    compute_penalty(params::AbstractVector, p::QuadraticPenalty) -> Real

Compute the total penalty contribution for given parameters.

# Arguments
- `params`: Parameter vector (natural scale)
- `p`: Quadratic penalty configuration

# Returns
Penalty value: (1/2) Σⱼ λⱼ βⱼᵀ Sⱼ βⱼ + total hazard penalties + smooth covariate penalties

# Notes
- Parameters are on natural scale with box constraints (β ≥ 0)
- Penalty is quadratic: P(β) = (λ/2) βᵀSβ
- Returns 0.0 if penalty has no active terms
"""
function compute_penalty(params::AbstractVector{T}, p::QuadraticPenalty) where T
    has_penalties(p) || return zero(T)
    
    penalty = zero(T)
    
    # Baseline hazard penalties
    for term in p.terms
        β_j = @view params[term.hazard_indices]
        penalty += term.lambda * dot(β_j, term.S * β_j)
    end
    
    # Total hazard penalties (if any)
    for term in p.total_hazard_terms
        K = size(term.S, 1)
        β_total = zeros(T, K)
        for idx_range in term.hazard_indices
            β_k = @view params[idx_range]
            β_total .+= β_k
        end
        penalty += term.lambda_H * dot(β_total, term.S * β_total)
    end
    
    # Smooth covariate penalties
    for term in p.smooth_covariate_terms
        β_k = params[term.param_indices]
        penalty += term.lambda * dot(β_k, term.S * β_k)
    end
    
    return penalty / 2
end

"""
    n_hyperparameters(p::QuadraticPenalty) -> Int

Return the number of smoothing parameters (λ values).
"""
n_hyperparameters(p::QuadraticPenalty) = p.n_lambda

"""
    get_hyperparameters(p::QuadraticPenalty) -> Vector{Float64}

Extract current λ values from all penalty terms.

Returns λ values in order: baseline terms, total hazard terms, smooth covariate terms.
"""
function get_hyperparameters(p::QuadraticPenalty)
    lambdas = Float64[]
    for term in p.terms
        push!(lambdas, term.lambda)
    end
    for term in p.total_hazard_terms
        push!(lambdas, term.lambda_H)
    end
    for term in p.smooth_covariate_terms
        push!(lambdas, term.lambda)
    end
    return lambdas
end

"""
    set_hyperparameters(p::QuadraticPenalty, lambda::Vector{Float64}) -> QuadraticPenalty

Return a new QuadraticPenalty with updated λ values.

The λ values should be in the same order as returned by `get_hyperparameters`:
baseline terms, total hazard terms, smooth covariate terms.
"""
function set_hyperparameters(p::QuadraticPenalty, lambda::Vector{Float64})
    length(lambda) == p.n_lambda || throw(ArgumentError(
        "Expected $(p.n_lambda) hyperparameters, got $(length(lambda))"
    ))
    
    idx = 1
    
    # Update baseline terms
    new_terms = map(p.terms) do term
        new_lambda = lambda[idx]
        idx += 1
        PenaltyTerm(term.hazard_indices, term.S, new_lambda, term.order, term.hazard_names)
    end
    
    # Update total hazard terms
    new_total_terms = map(p.total_hazard_terms) do term
        new_lambda = lambda[idx]
        idx += 1
        TotalHazardPenaltyTerm(term.origin, term.hazard_indices, term.S, new_lambda, term.order)
    end
    
    # Update smooth covariate terms
    new_smooth_terms = map(p.smooth_covariate_terms) do term
        new_lambda = lambda[idx]
        idx += 1
        SmoothCovariatePenaltyTerm(term.param_indices, term.S, new_lambda, term.order, term.label, term.hazard_name)
    end
    
    return QuadraticPenalty(
        new_terms,
        new_total_terms,
        new_smooth_terms,
        p.shared_lambda_groups,
        p.shared_smooth_groups,
        p.n_lambda
    )
end

"""
    hyperparameter_bounds(p::QuadraticPenalty) -> Tuple{Vector{Float64}, Vector{Float64}}

Return (lower, upper) bounds for log(λ) optimization.

Default bounds are (-8, 8) in log space, corresponding to λ ∈ [exp(-8), exp(8)] ≈ [3.4e-4, 2981].
"""
function hyperparameter_bounds(p::QuadraticPenalty)
    lb = fill(-8.0, p.n_lambda)
    ub = fill(8.0, p.n_lambda)
    return (lb, ub)
end

"""
    has_penalties(p::QuadraticPenalty) -> Bool

Check if the penalty configuration contains any active penalty terms.
"""
has_penalties(p::QuadraticPenalty) = !isempty(p.terms) || !isempty(p.total_hazard_terms) || !isempty(p.smooth_covariate_terms)
