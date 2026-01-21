# =============================================================================
# User-Facing Hazard Specification Types
# =============================================================================
#
# These types are used by users to specify hazards via the Hazard() and 
# @hazard macros. They are then processed by build_hazards() to create
# the internal _Hazard types used for computation.
#
# =============================================================================

"""
    ParametricHazard(haz::StatsModels.FormulaTerm, family::Symbol, statefrom::Int64, stateto::Int64)

Specify a cause-specific baseline hazard. 

# Arguments
- `hazard`: regression formula for the (log) hazard, parsed using StatsModels.jl.
- `family`: parameterization for the baseline hazard, one of `:exp` for exponential, `:wei` for Weibull, `:gom` for Gompertz. 
- `statefrom`: state number for the origin state.
- `stateto`: state number for the destination state.
"""
struct ParametricHazard <: HazardFunction
    hazard::StatsModels.FormulaTerm   # StatsModels.jl formula
    family::Symbol     # one of :exp, :wei, :gom
    statefrom::Int64   # starting state number
    stateto::Int64     # destination state number
    metadata::HazardMetadata
end

"""
    SplineHazard(haz::StatsModels.FormulaTerm, family::Symbol, statefrom::Int64, stateto::Int64; df::Union{Int64,Nothing}, degree::Int64, knots::Union{Vector{Float64},Float64,Nothing}, boundaryknots::Union{Vector{Float64},Nothing}, extrapolation::String, natural_spline::Bool)

Specify a cause-specific baseline hazard. 

# Arguments
- `hazard`: regression formula for the (log) hazard, parsed using StatsModels.jl.
- `family`: `:sp` for splines for the baseline hazard.
- `statefrom`: state number for the origin state.
- `stateto`: state number for the destination state.
- `df`: Degrees of freedom.
- `degree`: Degree of the spline polynomial basis.
- `knots`: Vector of knot locations.
- `boundaryknots`: Vector of boundary knot locations
- `extrapolation`: Extrapolation method beyond boundary knots:
    - "constant" (default): Constant hazard beyond boundaries with C¹ continuity (smooth transition).
      Uses basis recombination to enforce h'=0 at boundaries (Neumann BC). Requires degree >= 2.
    - "linear": Linear extrapolation using derivative at boundary.
- `natural_spline`: Restrict the second derivative to zero at the boundaries (natural spline).
- `monotone`: 
"""
struct SplineHazard <: HazardFunction
    hazard::StatsModels.FormulaTerm   # StatsModels.jl formula
    family::Symbol     # :sp for splines
    statefrom::Int64   # starting state number
    stateto::Int64     # destination state number
    degree::Int64
    knots::Union{Nothing,Float64,Vector{Float64}}
    boundaryknots::Union{Nothing,Vector{Float64}}
    extrapolation::String
    natural_spline::Bool
    monotone::Int64
    metadata::HazardMetadata
end

"""
    PhaseTypeHazard(hazard, family, statefrom, stateto, n_phases, structure, covariate_constraints, metadata)

User-facing specification for a phase-type (Coxian) hazard.
Created by `Hazard(:pt, ...)` and converted to internal types during model construction.

A phase-type hazard models the sojourn time as absorption in a Coxian Markov chain
with `n_phases` latent phases. This provides a flexible family of distributions
(including exponential as n_phases=1) while maintaining Markovian structure.

# Parameterization (Coxian)

For a transition s → d with n phases, the parameter vector contains:
- λ₁, ..., λₙ₋₁: progression rates between phases (n-1 parameters)
- μ₁, ..., μₙ: exit rates to destination state (n parameters)

Total baseline parameters: 2n - 1

# Fields
- `hazard`: StatsModels.jl formula for covariates
- `family`: Always `:pt`
- `statefrom`: Origin state number
- `stateto`: Destination state number  
- `n_phases`: Number of Coxian phases (≥ 1)
- `structure`: Coxian structure constraint (`:eigorder_sctp`, `:sctp`, `:ordered_sctp`, or `:unstructured`)
- `covariate_constraints`: Covariate sharing constraint (`:unstructured` or `:homogeneous`)
- `metadata`: HazardMetadata for time_transform and linpred_effect

# Structures
- `:sctp` (default): SCTP (Stationary Conditional Transition Probability) constraints only, ensuring
  P(dest | leaving state) is constant across phases
- `:eigorder_sctp`: SCTP plus eigenvalue ordering constraints (ν₁ ≥ ν₂ ≥ ... ≥ νₙ)
  where νⱼ = λⱼ + Σ_d μ_{j,d} is the total rate out of phase j. May cause constraint binding.
- `:ordered_sctp`: Alias for `:eigorder_sctp` (deprecated)
- `:unstructured`: All λᵢ and μᵢ are free parameters (no constraints)

# Covariate Constraints
- `:homogeneous` (default): Destination-specific shared covariate effects. All phases share the same
  covariate effect per destination (e.g., `h12_x`). Parameter count: K × p.
  **Recommended for better identifiability and interpretability.**
- `:unstructured`: Phase-specific covariate effects. Each phase has independent
  covariate parameters (e.g., `h12_a_x`, `h12_b_x`). Parameter count: n × K × p.

# Example
```julia
# 3-phase Coxian hazard for transition 1 → 2 with default constraints
# (eigorder_sctp structure + homogeneous covariates)
h12 = Hazard(@formula(0 ~ age), :pt, 1, 2; n_phases=3)
# Parameter names: h12_age (shared across all phases)

# With unstructured (phase-specific) covariates:
h12 = Hazard(@formula(0 ~ age), :pt, 1, 2; n_phases=3, covariate_constraints=:unstructured)
# Parameter names: h12_a_age, h12_b_age, h12_c_age (one per phase)

# With SCTP constraint only (no eigenvalue ordering)
model = multistatemodel(h12; data=df, n_phases=Dict(1 => 3), coxian_structure=:sctp)

# Fully unconstrained (not recommended)
model = multistatemodel(h12; data=df, n_phases=Dict(1 => 3), coxian_structure=:unstructured)
```

See also: [`PhaseTypeCoxianHazard`](@ref), [`PhaseTypeModel`](@ref)
"""
struct PhaseTypeHazard <: HazardFunction
    hazard::StatsModels.FormulaTerm   # StatsModels.jl formula
    family::Symbol                     # :pt
    statefrom::Int64
    stateto::Int64
    n_phases::Int                      # number of Coxian phases (≥1)
    structure::Symbol                  # :unstructured or :sctp (eigenvalue ordering, increasing by default)
    covariate_constraints::Symbol      # :unstructured (phase-specific) or :homogeneous (destination-shared)
    metadata::HazardMetadata
    
    function PhaseTypeHazard(hazard::StatsModels.FormulaTerm, family::Symbol,
                              statefrom::Int64, stateto::Int64, n_phases::Int,
                              structure::Symbol, covariate_constraints::Symbol,
                              metadata::HazardMetadata)
        family == :pt || throw(ArgumentError("PhaseTypeHazard family must be :pt"))
        n_phases >= 1 || throw(ArgumentError("n_phases must be ≥ 1, got $n_phases"))
        # Note: :sctp_increasing and :sctp_decreasing were removed - :sctp now implies increasing eigenvalues
        structure in (:unstructured, :sctp) ||
            throw(ArgumentError("structure must be :unstructured or :sctp, got :$structure"))
        covariate_constraints in (:unstructured, :homogeneous) ||
            throw(ArgumentError("covariate_constraints must be :unstructured or :homogeneous, got :$covariate_constraints"))
        new(hazard, family, statefrom, stateto, n_phases, structure, covariate_constraints, metadata)
    end
end

# =============================================================================
# Baseline Signature Helpers (Tang shared trajectories)
# =============================================================================

@inline _hashable_tuple(x::Nothing) = nothing
@inline _hashable_tuple(x::AbstractVector) = Tuple(x)
@inline _hashable_tuple(x) = x

baseline_signature(::HazardFunction, ::Symbol) = nothing

function baseline_signature(h::ParametricHazard, runtime_family::Symbol)
    parts = (:parametric, runtime_family)
    return UInt64(hash(parts))
end

function baseline_signature(h::SplineHazard, runtime_family::Symbol)
    if runtime_family != :sp
        return UInt64(hash((:parametric, runtime_family)))
    end
    knots_repr = _hashable_tuple(h.knots)
    boundary_repr = _hashable_tuple(h.boundaryknots)
    parts = (
        :spline,
        runtime_family,
        h.degree,
        knots_repr,
        boundary_repr,
        Symbol(h.extrapolation),
        h.natural_spline,
        h.monotone,
    )
    return UInt64(hash(parts))
end

shared_baseline_key(::HazardFunction, ::Symbol) = nothing

function shared_baseline_key(h::ParametricHazard, runtime_family::Symbol)
    h.metadata.time_transform || return nothing
    sig = baseline_signature(h, runtime_family)
    isnothing(sig) && return nothing
    return SharedBaselineKey(h.statefrom, sig)
end

function shared_baseline_key(h::SplineHazard, runtime_family::Symbol)
    h.metadata.time_transform || return nothing
    sig = baseline_signature(h, runtime_family)
    isnothing(sig) && return nothing
    return SharedBaselineKey(h.statefrom, sig)
end

function baseline_signature(h::PhaseTypeHazard, runtime_family::Symbol)
    parts = (:phasetype, runtime_family, h.n_phases)
    return UInt64(hash(parts))
end

function shared_baseline_key(h::PhaseTypeHazard, runtime_family::Symbol)
    h.metadata.time_transform || return nothing
    sig = baseline_signature(h, runtime_family)
    isnothing(sig) && return nothing
    return SharedBaselineKey(h.statefrom, sig)
end
