# =============================================================================
# Phase-Type Distribution Fitting for Semi-Markov Surrogates
# =============================================================================
#
# This file provides phase-type (PH) distribution fitting for use as 
# improved importance sampling proposals in MCEM. Phase-type distributions
# can better approximate non-exponential sojourns (Weibull, Gompertz) compared
# to simple Markov (exponential) surrogates.
#
# A phase-type distribution represents the time until absorption in a finite
# Markov chain. For a Coxian PH with p phases:
#
#   State 1 → State 2 → ... → State p → Absorption
#            ↘        ↘            ↘
#           Absorption  Absorption   Absorption
#
# The distribution is characterized by:
#   - S: p×p sub-intensity matrix (rates between transient phases)
#   - π: length-p initial distribution (probability of starting in each phase)
#   - s: length-p absorption rates (rate of absorption from each phase)
#
# Key property: s = -S * 1 (absorption rates = negative row sums of S)
#
# References:
# - Titman & Sharples (2010) Biometrics 66(3):742-752
# - Asmussen et al. (1996) Scand J Stat 23(4):419-441
#
# =============================================================================

# =============================================================================
# ProposalConfig: Unified Configuration for MCEM Proposals
# =============================================================================

"""
    ProposalConfig

Configuration for MCEM path proposals.

# Fields
- `type::Symbol`: Proposal type - `:markov` (default) or `:phasetype`
- `n_phases::Union{Symbol, Int, Vector{Int}}`: Number of phases (for `:phasetype` only)
  - `:auto`: BIC-based selection (fits 1 to max_phases, selects best by BIC)
  - `:heuristic`: 2 phases for states with non-Markovian transitions, 1 for exponential
  - `Int`: Same number of phases for all transient states
  - `Vector{Int}`: Per-state specification (length = number of transient states)
- `structure::Symbol`: Coxian structure for phase-type initialization (default: `:unstructured`)
  - `:unstructured`: All progression rates (r₁, r₂, ...) and absorption rates (a₁, a₂, ..., aₙ) 
    are free parameters (most flexible, recommended default)
  - `:prop_to_prog`: Absorption rates proportional to progression rates: aᵢ = c × rᵢ for i < n,
    where c is a common proportionality constant (Titman & Sharples 2010, Section 2)
  - `:allequal`: All progression rates equal (r₁ = r₂ = ... = r), all absorption rates equal 
    (a₁ = a₂ = ... = a), but r and a may differ
- `max_phases::Int`: Maximum phases for `:auto` BIC selection (default: 5)
- `optimize::Bool`: Optimize surrogate parameters (default: true)
- `parameters`: Manual parameter override (default: nothing)
- `constraints`: Constraints for surrogate optimization (default: nothing)

# n_phases Options

| Option | Description | Use case |
|--------|-------------|----------|
| `:auto` | Fits models with 1–max_phases and selects by BIC | Best fit, higher computational cost |
| `:heuristic` | 2 phases for non-Markov states, 1 for exponential | Fast sensible default |
| `Int` | Same for all transient states | Simple manual control |
| `Vector{Int}` | Per-state specification | Fine-grained control |

# Usage

```julia
# Default: Markov surrogate (exponential sojourn proposals)
fit(model)

# Markov surrogate with manual parameters
fit(model; proposal=ProposalConfig(parameters=my_params, optimize=false))

# Phase-type surrogate with BIC-based auto-selection
fit(model; proposal=ProposalConfig(type=:phasetype, n_phases=:auto))

# Phase-type with heuristic phases (2 for non-Markov, 1 for exponential)
fit(model; proposal=ProposalConfig(type=:phasetype, n_phases=:heuristic))

# Phase-type with manual override
fit(model; proposal=ProposalConfig(type=:phasetype, n_phases=3))
fit(model; proposal=ProposalConfig(type=:phasetype, n_phases=[2, 3, 1]))

# Phase-type with Titman-Sharples proportionality constraint
fit(model; proposal=ProposalConfig(type=:phasetype, n_phases=3, structure=:prop_to_prog))
```

For backward compatibility, the existing kwargs `optimize_surrogate`,
`surrogate_parameters`, and `surrogate_constraints` still work when no
`proposal` is specified.

See also: [`PhaseTypeConfig`](@ref), [`PhaseTypeSurrogate`](@ref), [`PhaseTypeProposal`](@ref)
"""
struct ProposalConfig
    type::Symbol
    n_phases::Union{Symbol, Int, Vector{Int}}
    structure::Symbol
    max_phases::Int
    optimize::Bool
    parameters::Union{Nothing, Any}
    constraints::Union{Nothing, Any}
    
    function ProposalConfig(;
            type::Symbol = :markov,
            n_phases::Union{Symbol, Int, Vector{Int}} = :auto,
            structure::Symbol = :unstructured,
            max_phases::Int = 5,
            optimize::Bool = true,
            parameters = nothing,
            constraints = nothing)
        
        # Validate type
        type in (:markov, :phasetype) || 
            throw(ArgumentError("type must be :markov or :phasetype, got :$type"))
        
        # Validate n_phases
        if n_phases isa Symbol
            n_phases in (:auto, :heuristic) || 
                throw(ArgumentError("n_phases symbol must be :auto or :heuristic, got :$n_phases"))
        elseif n_phases isa Int
            n_phases >= 1 || throw(ArgumentError("n_phases must be >= 1"))
        else
            all(n >= 1 for n in n_phases) || throw(ArgumentError("all n_phases must be >= 1"))
        end
        
        # Validate structure
        structure in (:unstructured, :prop_to_prog, :allequal) ||
            throw(ArgumentError("structure must be :unstructured, :prop_to_prog, or :allequal, got :$structure"))
        
        # Validate max_phases
        max_phases >= 1 || throw(ArgumentError("max_phases must be >= 1"))
        
        # Warn if phase-type options specified for Markov type
        if type == :markov && n_phases !== :auto
            @warn "n_phases is ignored for type=:markov proposals"
        end
        
        new(type, n_phases, structure, max_phases, optimize, parameters, constraints)
    end
end

"""
    MarkovProposal(; optimize=true, parameters=nothing, constraints=nothing)

Convenience constructor for Markov (exponential) proposal configuration.

Markov proposals use exponential sojourn times for all states. This is
the default and simplest option, but may be less efficient for models
with strongly non-exponential sojourn distributions.

# Arguments
- `optimize`: Optimize surrogate parameters (default: true)
- `parameters`: Manual parameter override (default: nothing)
- `constraints`: Constraints for optimization (default: nothing)

# Examples
```julia
# Default Markov proposal
fit(model; proposal=MarkovProposal())

# Markov proposal with fixed parameters
fit(model; proposal=MarkovProposal(optimize=false, parameters=my_params))
```

See also: [`ProposalConfig`](@ref), [`PhaseTypeProposal`](@ref)
"""
MarkovProposal(; kwargs...) = ProposalConfig(type=:markov; kwargs...)

"""
    PhaseTypeProposal(; n_phases=:auto, max_phases=5, kwargs...)

Convenience constructor for phase-type proposal configuration.

Phase-type proposals use expanded Markov chains to better approximate
non-exponential sojourn time distributions in importance sampling.

# Arguments
- `n_phases`: Number of latent phases per transient state
  - `:auto`: BIC-based selection (fits 1–max_phases, selects best)
  - `:heuristic`: 2 phases for non-Markov states, 1 for exponential
  - `Int`: Same for all transient states
  - `Vector{Int}`: Per-state specification
- `max_phases`: Maximum phases for `:auto` (default: 5)
- `optimize`: Optimize parameters (default: true)
- `parameters`: Manual parameter override (default: nothing)
- `constraints`: Constraints for optimization (default: nothing)

# Examples
```julia
# BIC-based selection (default)
fit(model; proposal=PhaseTypeProposal())

# Heuristic: 2 phases for non-Markov, 1 for exponential
fit(model; proposal=PhaseTypeProposal(n_phases=:heuristic))

# Manual specification
fit(model; proposal=PhaseTypeProposal(n_phases=3))
fit(model; proposal=PhaseTypeProposal(n_phases=[2, 3, 1]))
```

See also: [`ProposalConfig`](@ref), [`MarkovProposal`](@ref)
"""
PhaseTypeProposal(; n_phases::Union{Symbol, Int, Vector{Int}} = :auto, kwargs...) = 
    ProposalConfig(type=:phasetype, n_phases=n_phases; kwargs...)

"""
    needs_phasetype_proposal(hazards::Vector) -> Bool

Check if the model has any non-exponential hazards that would benefit from
phase-type proposals. Returns true if any hazard is not exponential.

Used by proposal=:auto to decide between Markov and phase-type proposals.

Note: This checks the `family` field for ParametricHazard types, not the
internal _MarkovHazard type used for surrogates.
"""
function needs_phasetype_proposal(hazards::Vector)
    for h in hazards
        # Check family field for exponential
        if hasproperty(h, :family)
            family = h.family
            # Family is now always a Symbol
            if family != :exp
                return true
            end
        elseif !isa(h, _MarkovHazard)
            # Fallback for other hazard types
            return true
        end
    end
    return false
end

"""
    resolve_proposal_config(proposal::Symbol, model) -> ProposalConfig

Resolve a Symbol proposal specification to a ProposalConfig.
- `:auto` → phase-type if any non-exponential hazards, else Markov
- `:markov` → MarkovProposal()
- `:phasetype` → PhaseTypeProposal(n_phases=:heuristic)
"""
function resolve_proposal_config(proposal::Symbol, model)
    if proposal === :auto
        if needs_phasetype_proposal(model.hazards)
            return PhaseTypeProposal(n_phases=:heuristic)
        else
            return MarkovProposal()
        end
    elseif proposal === :markov
        return MarkovProposal()
    elseif proposal === :phasetype
        return PhaseTypeProposal(n_phases=:heuristic)
    else
        throw(ArgumentError("Unknown proposal type: $proposal. Use :auto, :markov, :phasetype, or a ProposalConfig."))
    end
end

resolve_proposal_config(proposal::ProposalConfig, model) = proposal

# Uses: LinearAlgebra (dot, diag, I, exp), QuadGK (quadgk), SpecialFunctions (gamma)
# These are imported in the main module

"""
    PhaseTypeDistribution

Representation of a phase-type distribution for sojourn time approximation.

# Fields
- `n_phases::Int`: Number of latent phases (transient states)
- `Q::Matrix{Float64}`: Full intensity matrix ((p+1) × (p+1)), including absorbing state
- `initial::Vector{Float64}`: Initial distribution over phases (length p, sums to 1)

# Mathematical Background

A phase-type distribution PH(π, Q) represents the time until absorption in a 
finite-state continuous-time Markov chain. The intensity matrix Q has the form:

```
Q = [ S   s ]
    [ 0   0 ]
```

where S is the p×p sub-intensity matrix (transient states), s is the p×1 
absorption rate vector, and the last row is zeros (absorbing state).

The CDF is:
```
F(t) = 1 - π' exp(St) 𝟙
```

where π is the initial distribution, S is extracted from Q[1:p, 1:p],
and 𝟙 is a vector of ones.

For a Coxian distribution with p phases:
```
Q = [-(r₁+a₁)    r₁        0     ...    a₁  ]
    [    0    -(r₂+a₂)    r₂     ...    a₂  ]
    [    ⋮        ⋮        ⋱      ⋱      ⋮   ]
    [    0        0       ...   -aₚ     aₚ  ]
    [    0        0       ...     0      0  ]
```

where rᵢ is the progression rate to phase i+1 and aᵢ is the absorption rate.

# Example
```julia
# 2-phase Coxian distribution with Q matrix
Q = [-2.0  1.5  0.5;    # Phase 1: prog=1.5, abs=0.5
      0.0 -1.0  1.0;    # Phase 2: abs=1.0
      0.0  0.0  0.0]    # Absorbing state
ph = PhaseTypeDistribution(2, Q, [1.0, 0.0])  # Start in phase 1
```

See also: [`PhaseTypeConfig`](@ref), [`absorption_rates`](@ref), [`subintensity`](@ref)
"""
struct PhaseTypeDistribution
    n_phases::Int
    Q::Matrix{Float64}           # Full intensity matrix ((p+1) × (p+1))
    initial::Vector{Float64}     # Initial distribution (length p)
    
    function PhaseTypeDistribution(n_phases::Int, Q::Matrix{Float64}, 
                                   initial::Vector{Float64})
        # Validate dimensions - Q is (n_phases+1) × (n_phases+1)
        size(Q, 1) == size(Q, 2) == n_phases + 1 || 
            throw(DimensionMismatch("Q must be $(n_phases+1) × $(n_phases+1)"))
        length(initial) == n_phases || 
            throw(DimensionMismatch("initial must have length $n_phases"))
        
        # Validate initial distribution
        abs(sum(initial) - 1.0) < 1e-10 || 
            throw(ArgumentError("initial distribution must sum to 1"))
        all(initial .>= 0) || 
            throw(ArgumentError("initial distribution must be non-negative"))
        
        # Validate Q matrix structure
        # Diagonal of transient states should be negative
        all(diag(Q)[1:n_phases] .<= 0) || 
            throw(ArgumentError("diagonal of Q (transient states) must be non-positive"))
        
        # Last row should be zeros (absorbing state)
        all(Q[end, :] .== 0) || 
            throw(ArgumentError("last row of Q (absorbing state) must be zeros"))
        
        new(n_phases, Q, initial)
    end
end

"""
    subintensity(ph::PhaseTypeDistribution)

Extract the sub-intensity matrix S from the full intensity matrix Q.

The sub-intensity matrix S is the p×p upper-left block of Q, containing
only the transitions between transient phases.

# Returns
- `Matrix{Float64}`: p × p sub-intensity matrix
"""
function subintensity(ph::PhaseTypeDistribution)
    return ph.Q[1:ph.n_phases, 1:ph.n_phases]
end

"""
    absorption_rates(ph::PhaseTypeDistribution)

Extract absorption rates from the full intensity matrix Q.

The absorption rate from each phase is the rate of transitioning to the
absorbing state (last column of Q, excluding the absorbing state row).

# Returns
- `Vector{Float64}`: Absorption rate from each phase
"""
function absorption_rates(ph::PhaseTypeDistribution)
    return ph.Q[1:ph.n_phases, end]
end

"""
    progression_rates(ph::PhaseTypeDistribution)

Extract progression rates from the full intensity matrix Q (for Coxian distributions).

For a Coxian distribution, the progression rate from phase i to phase i+1
is Q[i, i+1].

# Returns
- `Vector{Float64}`: Progression rates r₁, r₂, ..., rₚ₋₁ (length n_phases - 1)
"""
function progression_rates(ph::PhaseTypeDistribution)
    n = ph.n_phases
    if n == 1
        return Float64[]
    end
    return [ph.Q[i, i+1] for i in 1:n-1]
end

"""
    PhaseTypeConfig

Configuration options for phase-type surrogate model.

# Fields
- `n_phases::Union{Symbol, Int, Vector{Int}}`: Number of phases per state
  - `:auto`: Select via BIC when building surrogate (requires data)
  - `:heuristic`: 2 phases for states with non-Markovian transitions, 1 for exponential
  - `Int`: Same number of phases for all transient states
  - `Vector{Int}`: Per-state phase counts (length = n_transient_states)
- `structure::Symbol`: Coxian structure for initialization (default: `:unstructured`)
  - `:unstructured`: All progression rates (r₁, r₂, ...) and absorption rates (a₁, a₂, ..., aₙ) 
    are free parameters (most flexible, recommended default)
  - `:prop_to_prog`: Absorption rates proportional to progression rates: aᵢ = c × rᵢ for i < n,
    where c is a common constant of proportionality (Titman & Sharples 2010, Section 2)
  - `:allequal`: All progression rates equal (r₁ = r₂ = ... = r), all absorption rates equal 
    (a₁ = a₂ = ... = a), but r and a may differ
  
  Note: This controls initial structure. For MLE fitting with custom constraints, pass
  `surrogate_constraints` to `fit()` or `set_surrogate!()` which will supersede these defaults.
- `constraints::Bool`: Apply Titman-Sharples constraints (default: true)
- `max_phases::Int`: Maximum phases to consider when n_phases=:auto (default: 5)

For **inference** (fitting phase-type hazard models), you must explicitly specify
`n_phases` as an `Int` or `Vector{Int}`.

For **MCEM proposals** (building surrogates):
- Use `:auto` to select the number of phases via BIC comparison
- Use `:heuristic` for a fast sensible default without fitting multiple models

# Example
```julia
# For inference - explicit specification required
config = PhaseTypeConfig(n_phases=3)

# For MCEM surrogates - auto-selection via BIC
config = PhaseTypeConfig(n_phases=:auto)
surrogate = build_phasetype_surrogate(tmat, config; data=my_data)

# For MCEM surrogates - heuristic defaults without BIC fitting
config = PhaseTypeConfig(n_phases=:heuristic)
surrogate = build_phasetype_surrogate(tmat, config; hazards=model.hazards)

# Different Coxian structures
config = PhaseTypeConfig(n_phases=3, structure=:prop_to_prog)  # Titman-Sharples constraint
config = PhaseTypeConfig(n_phases=3, structure=:unstructured)  # Free parameters
```
"""
struct PhaseTypeConfig
    n_phases::Union{Symbol, Int, Vector{Int}}
    structure::Symbol
    constraints::Bool
    max_phases::Int
    
    function PhaseTypeConfig(; 
            n_phases::Union{Symbol, Int, Vector{Int}} = 2,
            structure::Symbol = :unstructured,
            constraints::Bool = true,
            max_phases::Int = 5)
        
        if n_phases isa Symbol
            n_phases in (:auto, :heuristic) || 
                throw(ArgumentError("n_phases symbol must be :auto or :heuristic, got :$n_phases"))
        elseif n_phases isa Int
            n_phases >= 1 || throw(ArgumentError("n_phases must be >= 1"))
        else
            all(n_phases .>= 1) || throw(ArgumentError("all n_phases must be >= 1"))
        end
        structure in (:unstructured, :prop_to_prog, :allequal) ||
            throw(ArgumentError("structure must be :unstructured, :prop_to_prog, or :allequal, got :$structure"))
        max_phases >= 1 || throw(ArgumentError("max_phases must be >= 1"))
        
        new(n_phases, structure, constraints, max_phases)
    end
end

# =============================================================================
# Phase-Type Hazard Model Mappings
# =============================================================================

"""
    PhaseTypeMappings

Bidirectional mappings between observed and expanded state spaces for phase-type hazard models.

This struct is used when building a model with `:pt` (phase-type) hazards. It provides
the infrastructure to map between the original observed state space and the expanded
Markov state space where each state with phase-type hazards is split into multiple phases.

# Fields

**State space dimensions:**
- `n_observed::Int`: Number of observed (original) states
- `n_expanded::Int`: Number of expanded states (sum of phases across all states)

**Per-state phase information:**
- `n_phases_per_state::Vector{Int}`: Number of phases for each observed state
- `state_to_phases::Vector{UnitRange{Int}}`: Observed state → expanded phase indices
- `phase_to_state::Vector{Int}`: Expanded phase index → observed state

**Transition structure:**
- `expanded_tmat::Matrix{Int}`: Transition matrix on expanded state space
- `original_tmat::Matrix{Int}`: Original transition matrix (for reference)

**Hazard tracking:**
- `original_hazards::Vector{<:HazardFunction}`: Original user-specified hazards
- `pt_hazard_indices::Vector{Int}`: Indices of hazards that are phase-type
- `expanded_hazard_indices::Dict{Symbol, Vector{Int}}`: Maps original hazard name to
  expanded hazard indices (e.g., :h12 → [1, 2, 3] for λ₁, λ₂, μ₁ hazards)

# Example

For a 3-state model (states 1, 2, 3) with 2-phase Coxian on transition 1→2:

```
Original: State 1 ──h12──> State 2 ──h23──> State 3

Expanded: Phase 1.1 ──λ₁──> Phase 1.2 ──μ₂──> Phase 2.1 ──h23──> State 3
              │                                    │
              └────────────μ₁─────────────────────>│
```

- `n_phases_per_state = [2, 1, 1]` (state 1 has 2 phases)
- `state_to_phases = [1:2, 3:3, 4:4]`
- `phase_to_state = [1, 1, 2, 3]`

See also: [`PhaseTypeHazardSpec`](@ref), [`PhaseTypeModel`](@ref)
"""
struct PhaseTypeMappings
    # State space dimensions
    n_observed::Int
    n_expanded::Int
    
    # Per-state phase information
    n_phases_per_state::Vector{Int}
    state_to_phases::Vector{UnitRange{Int}}
    phase_to_state::Vector{Int}
    
    # Transition structure
    expanded_tmat::Matrix{Int}
    original_tmat::Matrix{Int}
    
    # Hazard tracking
    original_hazards::Vector{<:HazardFunction}
    pt_hazard_indices::Vector{Int}
    expanded_hazard_indices::Dict{Symbol, Vector{Int}}
    
    function PhaseTypeMappings(n_observed::Int, n_expanded::Int,
                                n_phases_per_state::Vector{Int},
                                state_to_phases::Vector{UnitRange{Int}},
                                phase_to_state::Vector{Int},
                                expanded_tmat::Matrix{Int},
                                original_tmat::Matrix{Int},
                                original_hazards::Vector{<:HazardFunction},
                                pt_hazard_indices::Vector{Int},
                                expanded_hazard_indices::Dict{Symbol, Vector{Int}})
        # Validation
        length(n_phases_per_state) == n_observed || 
            throw(DimensionMismatch("n_phases_per_state length must equal n_observed"))
        length(state_to_phases) == n_observed || 
            throw(DimensionMismatch("state_to_phases length must equal n_observed"))
        length(phase_to_state) == n_expanded || 
            throw(DimensionMismatch("phase_to_state length must equal n_expanded"))
        size(expanded_tmat) == (n_expanded, n_expanded) || 
            throw(DimensionMismatch("expanded_tmat must be n_expanded × n_expanded"))
        size(original_tmat) == (n_observed, n_observed) || 
            throw(DimensionMismatch("original_tmat must be n_observed × n_observed"))
        sum(n_phases_per_state) == n_expanded || 
            throw(ArgumentError("sum of n_phases_per_state must equal n_expanded"))
        
        new(n_observed, n_expanded, n_phases_per_state, state_to_phases,
            phase_to_state, expanded_tmat, original_tmat, original_hazards,
            pt_hazard_indices, expanded_hazard_indices)
    end
end

# =============================================================================
# Phase 3: State Space Expansion Functions
# =============================================================================

"""
    build_phasetype_mappings(hazards, tmat) -> PhaseTypeMappings

Build state space mappings from hazard specifications.

Analyzes hazard specifications to determine which transitions use phase-type hazards,
computes the number of phases per state, and builds bidirectional mappings between
the original observed state space and the expanded phase-type state space.

# Algorithm

1. Identify :pt hazards and extract n_phases for each origin state
2. States without :pt outgoing hazards get 1 phase (no expansion)
3. Build state_to_phases and phase_to_state mappings
4. Construct expanded transition matrix with internal phase transitions
5. Track which original hazards map to which expanded hazard indices

# Arguments
- `hazards::Vector{<:HazardFunction}`: User-specified hazard specifications
- `tmat::Matrix{Int}`: Original transition matrix

# Returns
- `PhaseTypeMappings`: Complete bidirectional mappings

# Example
```julia
h12 = Hazard(:pt, 1, 2; n_phases=3)
h23 = Hazard(:exp, 2, 3)
tmat = [0 1 0; 0 0 1; 0 0 0]
mappings = build_phasetype_mappings([h12, h23], tmat)
# n_phases_per_state = [3, 1, 1]  # State 1 has 3 phases from :pt hazard
# n_expanded = 5  # Total phases
```

See also: [`PhaseTypeMappings`](@ref), [`build_expanded_tmat`](@ref)
"""
function build_phasetype_mappings(hazards::Vector{<:HazardFunction}, 
                                   tmat::Matrix{Int})
    n_observed = size(tmat, 1)
    
    # Step 1: Determine n_phases per state from :pt hazards
    # A state's n_phases is determined by the maximum n_phases of any outgoing :pt hazard
    n_phases_per_state = ones(Int, n_observed)
    pt_hazard_indices = Int[]
    
    for (i, h) in enumerate(hazards)
        if h isa PhaseTypeHazardSpec
            push!(pt_hazard_indices, i)
            s = h.statefrom
            # Take maximum n_phases if multiple :pt hazards from same state
            n_phases_per_state[s] = max(n_phases_per_state[s], h.n_phases)
        end
    end
    
    # Step 2: Build state_to_phases mapping
    n_expanded = sum(n_phases_per_state)
    state_to_phases = Vector{UnitRange{Int}}(undef, n_observed)
    phase_to_state = Vector{Int}(undef, n_expanded)
    
    phase_idx = 1
    for s in 1:n_observed
        n_phases = n_phases_per_state[s]
        state_to_phases[s] = phase_idx:(phase_idx + n_phases - 1)
        for p in 1:n_phases
            phase_to_state[phase_idx + p - 1] = s
        end
        phase_idx += n_phases
    end
    
    # Step 3: Build expanded transition matrix
    expanded_tmat = _build_expanded_tmat(tmat, n_phases_per_state, state_to_phases, hazards)
    
    # Step 4: Build expanded hazard indices mapping
    expanded_hazard_indices = _build_expanded_hazard_indices(hazards, n_phases_per_state, state_to_phases)
    
    return PhaseTypeMappings(
        n_observed, n_expanded,
        n_phases_per_state, state_to_phases, phase_to_state,
        expanded_tmat, tmat,
        hazards, pt_hazard_indices, expanded_hazard_indices
    )
end

"""
    _build_expanded_tmat(original_tmat, n_phases_per_state, state_to_phases, hazards)

Build the expanded transition matrix for the phase-type state space.

# Structure

For a state s with n phases and outgoing transition to state d:

```
Within state s (progression):
  Phase 1 → Phase 2 → ... → Phase n  (rates λ₁, ..., λₙ₋₁)

Exit transitions (to first phase of destination d):
  Phase 1 → d.Phase 1  (rate μ₁)
  Phase 2 → d.Phase 1  (rate μ₂)
  ...
  Phase n → d.Phase 1  (rate μₙ)
```

The expanded tmat uses positive integers to index hazards:
- Internal progressions: indexed sequentially
- Exit transitions: indexed after progressions

# Returns
- `Matrix{Int}`: Expanded transition matrix with hazard indices
"""
function _build_expanded_tmat(original_tmat::Matrix{Int}, 
                               n_phases_per_state::Vector{Int},
                               state_to_phases::Vector{UnitRange{Int}},
                               hazards::Vector{<:HazardFunction})
    n_observed = size(original_tmat, 1)
    n_expanded = sum(n_phases_per_state)
    expanded_tmat = zeros(Int, n_expanded, n_expanded)
    
    hazard_counter = 1
    
    # For each observed state
    for s in 1:n_observed
        phases_s = state_to_phases[s]
        n_phases = n_phases_per_state[s]
        
        # Check if this state has :pt outgoing hazards
        has_pt = any(h -> h isa PhaseTypeHazardSpec && h.statefrom == s, hazards)
        
        if has_pt && n_phases > 1
            # Add internal progression transitions (λ rates)
            for p in 1:(n_phases - 1)
                from_phase = phases_s[p]
                to_phase = phases_s[p + 1]
                expanded_tmat[from_phase, to_phase] = hazard_counter
                hazard_counter += 1
            end
        end
        
        # Add exit transitions to each destination
        for d in 1:n_observed
            if original_tmat[s, d] > 0
                # Find the hazard for this transition
                haz_idx = findfirst(h -> h.statefrom == s && h.stateto == d, hazards)
                @assert !isnothing(haz_idx) "No hazard found for transition $s → $d"
                
                h = hazards[haz_idx]
                first_phase_d = first(state_to_phases[d])
                
                if h isa PhaseTypeHazardSpec
                    # Phase-type: exit from each phase (μ rates)
                    for p in 1:n_phases
                        from_phase = phases_s[p]
                        expanded_tmat[from_phase, first_phase_d] = hazard_counter
                        hazard_counter += 1
                    end
                else
                    # Non-PT hazard: single transition from each phase
                    # All phases use the same hazard (will share parameters)
                    for p in 1:n_phases
                        from_phase = phases_s[p]
                        expanded_tmat[from_phase, first_phase_d] = hazard_counter
                    end
                    hazard_counter += 1
                end
            end
        end
    end
    
    return expanded_tmat
end

"""
    _build_expanded_hazard_indices(hazards, n_phases_per_state, state_to_phases)

Build mapping from original hazard names to expanded hazard indices.

For each original hazard, tracks which indices in the expanded hazard vector
correspond to it. For :pt hazards, this includes both progression (λ) and
exit (μ) transitions.

# Returns
- `Dict{Symbol, Vector{Int}}`: Maps hazard name (e.g., :h12) to expanded indices
"""
function _build_expanded_hazard_indices(hazards::Vector{<:HazardFunction},
                                         n_phases_per_state::Vector{Int},
                                         state_to_phases::Vector{UnitRange{Int}})
    expanded_indices = Dict{Symbol, Vector{Int}}()
    hazard_counter = 1
    n_observed = length(n_phases_per_state)
    
    # Track which hazards we've processed (by origin state)
    processed_states = Set{Int}()
    
    for s in 1:n_observed
        n_phases = n_phases_per_state[s]
        
        # Check if this state has :pt outgoing hazards
        has_pt = any(h -> h isa PhaseTypeHazardSpec && h.statefrom == s, hazards)
        
        if has_pt && n_phases > 1
            # Count progression hazards (internal λ transitions)
            # These are shared across all :pt hazards from this state
            progression_indices = collect(hazard_counter:(hazard_counter + n_phases - 2))
            hazard_counter += n_phases - 1
        end
        
        # Process each outgoing hazard from state s
        for h in hazards
            h.statefrom == s || continue
            
            hazname = Symbol("h$(s)$(h.stateto)")
            
            if h isa PhaseTypeHazardSpec
                # PT hazard: n exit transitions (μ rates)
                exit_indices = collect(hazard_counter:(hazard_counter + n_phases - 1))
                hazard_counter += n_phases
                
                # Include both progression and exit indices
                if has_pt && n_phases > 1
                    expanded_indices[hazname] = vcat(progression_indices, exit_indices)
                else
                    expanded_indices[hazname] = exit_indices
                end
            else
                # Non-PT hazard: single index
                expanded_indices[hazname] = [hazard_counter]
                hazard_counter += 1
            end
        end
    end
    
    return expanded_indices
end

"""
    has_phasetype_hazards(hazards::Vector{<:HazardFunction}) -> Bool

Check if any hazard in the vector is a phase-type specification.
"""
function has_phasetype_hazards(hazards::Vector{<:HazardFunction})
    return any(h -> h isa PhaseTypeHazardSpec, hazards)
end

"""
    get_phasetype_n_phases(hazards::Vector{<:HazardFunction}, statefrom::Int)

Get the number of phases for outgoing transitions from a state.
Returns the maximum n_phases across all :pt hazards from that state, or 1 if none.
"""
function get_phasetype_n_phases(hazards::Vector{<:HazardFunction}, statefrom::Int)
    max_phases = 1
    for h in hazards
        if h isa PhaseTypeHazardSpec && h.statefrom == statefrom
            max_phases = max(max_phases, h.n_phases)
        end
    end
    return max_phases
end

"""
    _compute_default_n_phases(tmat, hazards)

Compute default number of phases per state based on hazard types.

Returns a Vector{Int} with:
- 2 phases for states with at least one non-Markovian (non-exponential) outgoing transition
- 1 phase for states where all outgoing transitions are exponential (MarkovHazard)
- 1 phase for absorbing states

# Arguments
- `tmat::Matrix{Int64}`: Transition matrix
- `hazards::Vector`: Vector of hazard objects

# Returns
- `Vector{Int}`: Number of phases per state
"""
function _compute_default_n_phases(tmat::Matrix{Int64}, hazards::Vector)
    n_states = size(tmat, 1)
    is_absorbing = [all(tmat[s, :] .== 0) for s in 1:n_states]
    
    # Build index for hazards by (from_state, to_state)
    hazard_is_markov = Dict{Tuple{Int,Int}, Bool}()
    for h in hazards
        # Check if hazard is a MarkovHazard (exponential) - uses abstract type _MarkovHazard
        is_markov = isa(h, _MarkovHazard)
        hazard_is_markov[(h.statefrom, h.stateto)] = is_markov
    end
    
    n_phases_per_state = ones(Int, n_states)
    
    for s in 1:n_states
        if is_absorbing[s]
            n_phases_per_state[s] = 1
            continue
        end
        
        # Check all outgoing transitions from state s
        has_non_markov = false
        for t in 1:n_states
            if tmat[s, t] > 0  # Transition s→t is allowed
                # Get the hazard for this transition
                if haskey(hazard_is_markov, (s, t))
                    if !hazard_is_markov[(s, t)]
                        has_non_markov = true
                        break
                    end
                else
                    # If no hazard found, assume non-Markov (conservative)
                    has_non_markov = true
                    break
                end
            end
        end
        
        # 2 phases for non-Markov states, 1 for all-exponential states
        n_phases_per_state[s] = has_non_markov ? 2 : 1
    end
    
    return n_phases_per_state
end

# =============================================================================
# Phase 3: Hazard and Data Expansion for Phase-Type Fitting
# =============================================================================

"""
    expand_hazards_for_phasetype(hazards, mappings, data) -> Vector{_Hazard}

Convert user hazard specifications to runtime hazards on the expanded phase-type state space.

This function transforms the user's hazard specifications into the internal hazard
representations needed for the expanded Markov model. Each original hazard maps to
one or more hazards in the expanded space.

# Expansion Logic

For each original hazard `s → d`:

**Phase-type hazards (`:pt`)**:
Creates multiple `MarkovHazard` instances:
- `n - 1` progression hazards (λ): phase i → phase i+1 within state s
- `n` exit hazards (μ): phase i → first phase of destination d

**Exponential hazards (`:exp`)**:
- Creates one `MarkovHazard` per source phase, all sharing the same rate

**Semi-Markov hazards (`:wei`, `:gom`, `:sp`)**:
- Creates the appropriate hazard type on the expanded space
- All source phases share the same hazard (approximation for phase-type fitting)

# Arguments
- `hazards::Vector{<:HazardFunction}`: User-specified hazard specifications
- `mappings::PhaseTypeMappings`: State space mappings from `build_phasetype_mappings`
- `data::DataFrame`: Model data for parameter schema resolution

# Returns
- `Vector{_Hazard}`: Runtime hazard objects for the expanded model
- `Vector{Vector{Float64}}`: Initial parameters for each hazard

# Example
```julia
mappings = build_phasetype_mappings(hazards, tmat)
expanded_hazards, expanded_params = expand_hazards_for_phasetype(hazards, mappings, data)
```

See also: [`build_phasetype_mappings`](@ref), [`PhaseTypeCoxianHazard`](@ref)
"""
function expand_hazards_for_phasetype(hazards::Vector{<:HazardFunction}, 
                                       mappings::PhaseTypeMappings,
                                       data::DataFrame)
    expanded_hazards = _Hazard[]
    expanded_params = Vector{Float64}[]
    
    n_observed = mappings.n_observed
    
    # Process each observed state's outgoing transitions
    for s in 1:n_observed
        n_phases = mappings.n_phases_per_state[s]
        phases_s = mappings.state_to_phases[s]
        
        # Check if this state has :pt hazards (determines if we need progression hazards)
        has_pt = any(h -> h isa PhaseTypeHazardSpec && h.statefrom == s, hazards)
        
        # Add progression hazards (λ: phase i → phase i+1) if this state has :pt
        if has_pt && n_phases > 1
            for p in 1:(n_phases - 1)
                from_phase = phases_s[p]
                to_phase = phases_s[p + 1]
                
                # Create progression hazard (exponential rate)
                prog_haz, prog_pars = _build_progression_hazard(s, p, n_phases, from_phase, to_phase)
                push!(expanded_hazards, prog_haz)
                push!(expanded_params, prog_pars)
            end
        end
        
        # Add exit hazards for each destination
        for h in hazards
            h.statefrom == s || continue
            d = h.stateto
            first_phase_d = first(mappings.state_to_phases[d])
            
            if h isa PhaseTypeHazardSpec
                # PT hazard: create exit hazard from each phase
                for p in 1:n_phases
                    from_phase = phases_s[p]
                    exit_haz, exit_pars = _build_exit_hazard(h, s, d, p, n_phases, 
                                                              from_phase, first_phase_d, data)
                    push!(expanded_hazards, exit_haz)
                    push!(expanded_params, exit_pars)
                end
            else
                # Non-PT hazard: single hazard from each phase (shared rate)
                # Build the hazard once, then replicate for each phase
                base_haz, base_pars = _build_expanded_hazard(h, s, d, phases_s[1], first_phase_d, data)
                push!(expanded_hazards, base_haz)
                push!(expanded_params, base_pars)
                
                # For multi-phase states, create additional hazards that share parameters
                # These are distinct hazard objects but will use the same parameter indices
                for p in 2:n_phases
                    from_phase = phases_s[p]
                    shared_haz = _build_shared_phase_hazard(base_haz, from_phase, first_phase_d)
                    push!(expanded_hazards, shared_haz)
                    # No additional parameters - shares with base_haz
                end
            end
        end
    end
    
    return expanded_hazards, expanded_params
end

"""
    _build_progression_hazard(observed_state, phase_index, n_phases, from_phase, to_phase)

Build a MarkovHazard for internal phase progression (λ rate).

These hazards represent transitions between phases within the same observed state.
They are exponential hazards with no covariates.
"""
function _build_progression_hazard(observed_state::Int, phase_index::Int, n_phases::Int,
                                    from_phase::Int, to_phase::Int)
    # Parameter name: λᵢ for phase i progression
    hazname = Symbol("h$(observed_state)_prog$(phase_index)")
    parname = Symbol("$(hazname)_lambda")
    
    # Simple exponential hazard function (no covariates)
    # pars is a NamedTuple: (baseline = (lambda = ...,),)
    hazard_fn = (t, pars, covars) -> exp(only(values(pars.baseline)))
    cumhaz_fn = (lb, ub, pars, covars) -> exp(only(values(pars.baseline))) * (ub - lb)
    
    metadata = HazardMetadata(
        linpred_effect = :ph,
        time_transform = false
    )
    
    haz = MarkovHazard(
        hazname,
        from_phase,        # expanded state from
        to_phase,          # expanded state to
        :exp,
        [parname],
        1,                 # npar_baseline
        1,                 # npar_total
        hazard_fn,
        cumhaz_fn,
        false,             # no covariates
        Symbol[],          # covar_names
        metadata,
        nothing            # shared_baseline_key
    )
    
    return haz, [0.0]  # Initial rate = exp(0) = 1
end

"""
    _build_exit_hazard(pt_spec, observed_from, observed_to, phase_index, n_phases, 
                       from_phase, to_phase, data)

Build a MarkovHazard for phase-type exit transition (μ rate).

These hazards represent transitions from a specific phase to the destination state.
Covariates from the original :pt specification are included.
"""
function _build_exit_hazard(pt_spec::PhaseTypeHazardSpec, 
                             observed_from::Int, observed_to::Int,
                             phase_index::Int, n_phases::Int,
                             from_phase::Int, to_phase::Int,
                             data::DataFrame)
    # Parameter name: μᵢ for phase i exit
    hazname = Symbol("h$(observed_from)$(observed_to)_exit$(phase_index)")
    baseline_parname = Symbol("$(hazname)_mu")
    
    # Get covariate info from original specification
    schema = StatsModels.schema(pt_spec.hazard, data)
    hazschema = apply_schema(pt_spec.hazard, schema)
    rhs_names = _phasetype_rhs_names(hazschema)
    
    # Build parameter names
    covar_labels = length(rhs_names) > 1 ? rhs_names[2:end] : String[]
    covar_parnames = [Symbol("$(hazname)_$(c)") for c in covar_labels]
    parnames = vcat([baseline_parname], covar_parnames)
    
    has_covars = !isempty(covar_parnames)
    covar_names = Symbol.(covar_labels)
    npar_total = 1 + length(covar_parnames)
    
    # Build hazard functions with covariates if present
    linpred_effect = pt_spec.metadata.linpred_effect
    hazard_fn, cumhaz_fn = _generate_exit_hazard_fns(parnames, linpred_effect, has_covars)
    
    metadata = HazardMetadata(
        linpred_effect = linpred_effect,
        time_transform = false  # Exit hazards are exponential
    )
    
    haz = MarkovHazard(
        hazname,
        from_phase,
        to_phase,
        :exp,
        parnames,
        1,                 # npar_baseline (just μ)
        npar_total,
        hazard_fn,
        cumhaz_fn,
        has_covars,
        covar_names,
        metadata,
        nothing            # Each exit hazard is independent
    )
    
    return haz, zeros(Float64, npar_total)
end

"""
    _generate_exit_hazard_fns(parnames, linpred_effect, has_covars)

Generate hazard and cumulative hazard functions for exit transitions.
"""
function _generate_exit_hazard_fns(parnames::Vector{Symbol}, linpred_effect::Symbol, has_covars::Bool)
    if !has_covars
        # pars is a NamedTuple: (baseline = (param_name = ...,),)
        hazard_fn = (t, pars, covars) -> exp(only(values(pars.baseline)))
        cumhaz_fn = (lb, ub, pars, covars) -> exp(only(values(pars.baseline))) * (ub - lb)
    else
        # Use generate_exponential_hazard from hazards.jl
        hazard_fn, cumhaz_fn = generate_exponential_hazard(parnames, linpred_effect)
    end
    return hazard_fn, cumhaz_fn
end

"""
    _phasetype_rhs_names(hazschema)

Extract RHS names from a hazard formula schema (helper for phase-type expansion).
"""
function _phasetype_rhs_names(hazschema)
    has_intercept = _phasetype_formula_has_intercept(hazschema.rhs)
    coef_names = StatsModels.coefnames(hazschema.rhs)
    coef_vec = coef_names isa AbstractVector ? collect(coef_names) : [coef_names]
    return has_intercept ? coef_vec : vcat("(Intercept)", coef_vec)
end

"""
    _phasetype_formula_has_intercept(rhs_term)

Check if formula RHS includes an intercept term.
Uses the same logic as _hazard_formula_has_intercept in modelgeneration.jl.
"""
@inline function _phasetype_formula_has_intercept(rhs_term)
    rhs_term isa StatsModels.ConstantTerm && return true
    rhs_term isa StatsModels.InterceptTerm && return true
    if rhs_term isa StatsModels.MatrixTerm
        return any(_phasetype_formula_has_intercept, rhs_term.terms)
    end
    return false
end

"""
    _build_expanded_hazard(h, observed_from, observed_to, from_phase, to_phase, data)

Build a runtime hazard for non-PT hazard types on the expanded space.

This handles :exp, :wei, :gom, and :sp hazards by creating the appropriate
hazard type with adjusted state indices for the expanded space.
"""
function _build_expanded_hazard(h::HazardFunction, 
                                 observed_from::Int, observed_to::Int,
                                 from_phase::Int, to_phase::Int,
                                 data::DataFrame)
    # Create a modified hazard spec with expanded state indices
    # The family determines which builder to use
    family = h.family isa Symbol ? h.family : Symbol(lowercase(String(h.family)))
    
    hazname = Symbol("h$(observed_from)$(observed_to)")
    
    # Use the existing hazard building infrastructure
    # Create context for the hazard
    schema = StatsModels.schema(h.hazard, data)
    hazschema = apply_schema(h.hazard, schema)
    modelcols(hazschema, data)  # validate
    rhs_names = _phasetype_rhs_names(hazschema)
    shared_key = shared_baseline_key(h, family)
    
    ctx = HazardBuildContext(
        h,
        hazname,
        family,
        h.metadata,
        rhs_names,
        shared_key,
        data
    )
    
    # Get the builder and create the hazard
    builder = get(_HAZARD_BUILDERS, family, nothing)
    if builder === nothing
        error("Unknown hazard family for phase-type expansion: $(family)")
    end
    
    haz_struct, hazpars = builder(ctx)
    
    # Adjust the state indices to expanded space
    adjusted_haz = _adjust_hazard_states(haz_struct, from_phase, to_phase)
    
    return adjusted_haz, hazpars
end

"""
    _adjust_hazard_states(haz, new_statefrom, new_stateto)

Create a copy of a hazard with adjusted state indices for the expanded space.
"""
function _adjust_hazard_states(haz::MarkovHazard, new_statefrom::Int, new_stateto::Int)
    return MarkovHazard(
        haz.hazname, new_statefrom, new_stateto, haz.family,
        haz.parnames, haz.npar_baseline, haz.npar_total,
        haz.hazard_fn, haz.cumhaz_fn, haz.has_covariates,
        haz.covar_names, haz.metadata, haz.shared_baseline_key
    )
end

function _adjust_hazard_states(haz::SemiMarkovHazard, new_statefrom::Int, new_stateto::Int)
    return SemiMarkovHazard(
        haz.hazname, new_statefrom, new_stateto, haz.family,
        haz.parnames, haz.npar_baseline, haz.npar_total,
        haz.hazard_fn, haz.cumhaz_fn, haz.has_covariates,
        haz.covar_names, haz.metadata, haz.shared_baseline_key
    )
end

function _adjust_hazard_states(haz::RuntimeSplineHazard, new_statefrom::Int, new_stateto::Int)
    return RuntimeSplineHazard(
        haz.hazname, new_statefrom, new_stateto, haz.family,
        haz.parnames, haz.npar_baseline, haz.npar_total,
        haz.hazard_fn, haz.cumhaz_fn, haz.has_covariates,
        haz.covar_names, haz.degree, haz.knots, haz.natural_spline,
        haz.monotone, haz.extrapolation, haz.metadata, haz.shared_baseline_key
    )
end

"""
    _build_shared_phase_hazard(base_haz, from_phase, to_phase)

Create a hazard that shares parameters with a base hazard but has different state indices.

Used for non-PT hazards when the source state has multiple phases.
All phases share the same transition rate to the destination.
"""
function _build_shared_phase_hazard(base_haz::_Hazard, from_phase::Int, to_phase::Int)
    return _adjust_hazard_states(base_haz, from_phase, to_phase)
end

"""
    expand_data_for_phasetype_fitting(data, mappings) -> (expanded_data, censoring_patterns)

Expand observation data to handle phase uncertainty during phase-type model fitting.

When fitting a phase-type model, the latent phase of each observation is unknown.
This function prepares the data for the forward-backward algorithm by mapping
observed states to their corresponding phase ranges in the expanded space.

# Phase Uncertainty

For an observation in observed state `s` with `n` phases:
- The subject could be in any of phases `1, 2, ..., n` of state `s`
- The emission matrix will encode this uncertainty
- Forward-backward will marginalize over the unknown phase

# Data Transformation

The function:
1. Maps `statefrom` and `stateto` to their first phases (conservative)
2. Builds censoring patterns that allow any phase of the observed state
3. Preserves all covariates and timing information

# Arguments
- `data::DataFrame`: Original observation data with observed state indices
- `mappings::PhaseTypeMappings`: State space mappings

# Returns
- `expanded_data::DataFrame`: Data with phase-space state indices
- `censoring_patterns::Matrix{Float64}`: Emission patterns for phase uncertainty

# Example
```julia
mappings = build_phasetype_mappings(hazards, tmat)
exp_data, cens_pats = expand_data_for_phasetype_fitting(data, mappings)
```

See also: [`build_phasetype_mappings`](@ref)
"""
function expand_data_for_phasetype_fitting(data::DataFrame, mappings::PhaseTypeMappings)
    # Copy the data to avoid mutation
    expanded_data = copy(data)
    
    n_expanded = mappings.n_expanded
    n_observed = mappings.n_observed
    
    # Map observed states to first phase of each state
    # This is the "reference" state for each observation
    expanded_data.statefrom = [first(mappings.state_to_phases[s]) for s in data.statefrom]
    expanded_data.stateto = [first(mappings.state_to_phases[s]) for s in data.stateto]
    
    # Build censoring patterns for phase uncertainty
    # Each observed state maps to a set of possible phases
    censoring_patterns = _build_phase_censoring_patterns(mappings)
    
    return expanded_data, censoring_patterns
end

"""
    _build_phase_censoring_patterns(mappings) -> Matrix{Float64}

Build censoring patterns that encode phase uncertainty.

For each observed state, creates a pattern where all phases of that state
have probability 1 (possible) and all other states have probability 0 (impossible).

The patterns are indexed by observed state (rows) and expanded state (columns).
"""
function _build_phase_censoring_patterns(mappings::PhaseTypeMappings)
    n_observed = mappings.n_observed
    n_expanded = mappings.n_expanded
    
    # Pattern matrix: row = observed state code, column = expanded state
    # Pattern 0 = absorbing (no uncertainty)
    # Patterns 1..n_observed = each observed state's phase set
    patterns = zeros(Float64, n_observed + 1, n_expanded)
    
    # Pattern 0: all zeros (for absorbing states, not used typically)
    # Patterns 1..n_observed: each observed state allows its phases
    for s in 1:n_observed
        phases = mappings.state_to_phases[s]
        for p in phases
            patterns[s + 1, p] = 1.0
        end
    end
    
    return patterns
end

"""
    map_observation_to_phases(obstype, statefrom, stateto, mappings) -> (emission_pattern_index, allowed_phases)

Map an observation to its compatible phases in the expanded space.

# Returns
- `emission_pattern_index::Int`: Index into censoring patterns (0 = exact, 1+ = uncertain)
- `allowed_phases::Vector{Int}`: Which expanded states are compatible

This is used during likelihood computation to determine the emission matrix entries.
"""
function map_observation_to_phases(obstype::Int, statefrom::Int, stateto::Int, 
                                    mappings::PhaseTypeMappings)
    if obstype == 1
        # Exact observation: still uncertain about phase
        return statefrom + 1, collect(mappings.state_to_phases[statefrom])
    elseif obstype == 2
        # Right-censored: could be in any phase of current state
        return statefrom + 1, collect(mappings.state_to_phases[statefrom])
    elseif obstype == 3
        # Interval-censored: could be in source or dest phases
        src_phases = collect(mappings.state_to_phases[statefrom])
        dst_phases = collect(mappings.state_to_phases[stateto])
        return 0, vcat(src_phases, dst_phases)  # Custom pattern needed
    else
        # Default: all phases of statefrom
        return statefrom + 1, collect(mappings.state_to_phases[statefrom])
    end
end

# =============================================================================
# Phase 4: Model Building for Phase-Type Hazards
# =============================================================================

"""
    _build_phasetype_model_from_hazards(hazards, data; kwargs...) -> PhaseTypeModel

Internal function to build a PhaseTypeModel from user hazard specifications.

This is called by `multistatemodel()` when any hazard is a `PhaseTypeHazardSpec`.
It builds the expanded state space model and wraps it with the necessary
mappings for parameter interpretation.

# Algorithm

1. Build state mappings from :pt hazards (`build_phasetype_mappings`)
2. Expand hazards to internal representation (`expand_hazards_for_phasetype`)
3. Expand data for phase uncertainty (`expand_data_for_phasetype_fitting`)
4. Build the internal Markov model on expanded space
5. Wrap with `PhaseTypeModel` containing mappings and original specifications

# Arguments
- `hazards::Tuple{Vararg{HazardFunction}}`: User hazard specifications (some :pt)
- `data::DataFrame`: Observation data
- `constraints`: Optional parameter constraints
- `SubjectWeights`, `ObservationWeights`: Optional weights
- `CensoringPatterns`: Optional censoring patterns
- `EmissionMatrix`: Optional emission matrix
- `verbose`: Print progress information

# Returns
- `PhaseTypeModel`: Complete phase-type model with expanded internal representation

See also: [`PhaseTypeModel`](@ref), [`build_phasetype_mappings`](@ref)
"""
function _build_phasetype_model_from_hazards(hazards::Tuple{Vararg{HazardFunction}};
                                              data::DataFrame,
                                              constraints = nothing,
                                              SubjectWeights::Union{Nothing,Vector{Float64}} = nothing,
                                              ObservationWeights::Union{Nothing,Vector{Float64}} = nothing,
                                              CensoringPatterns::Union{Nothing,Matrix{<:Real}} = nothing,
                                              EmissionMatrix::Union{Nothing,Matrix{Float64}} = nothing,
                                              verbose::Bool = false)
    
    hazards_vec = collect(hazards)
    
    # Step 1: Enumerate hazards and build original tmat
    hazinfo = enumerate_hazards(hazards...)
    hazards_ordered = hazards_vec[hazinfo.order]
    select!(hazinfo, Not(:order))
    tmat_original = create_tmat(hazinfo)
    
    if verbose
        println("Building phase-type model...")
        println("  Original states: $(size(tmat_original, 1))")
    end
    
    # Step 2: Build mappings from hazard specifications
    mappings = build_phasetype_mappings(hazards_ordered, tmat_original)
    
    if verbose
        println("  Expanded states: $(mappings.n_expanded)")
        println("  Phases per state: $(mappings.n_phases_per_state)")
    end
    
    # Step 3: Expand hazards to runtime representation
    expanded_hazards, expanded_params_list = expand_hazards_for_phasetype(
        hazards_ordered, mappings, data
    )
    
    if verbose
        println("  Expanded hazards: $(length(expanded_hazards))")
    end
    
    # Step 4: Expand data for phase uncertainty
    expanded_data, phase_censoring_patterns = expand_data_for_phasetype_fitting(data, mappings)
    
    # Step 5: Build expanded hazkeys
    expanded_hazkeys = Dict{Symbol, Int64}()
    for (idx, h) in enumerate(expanded_hazards)
        expanded_hazkeys[h.hazname] = idx
    end
    
    # Step 6: Build parameters structure
    expanded_parameters = _build_expanded_parameters(expanded_params_list, expanded_hazkeys, expanded_hazards)
    
    # Step 7: Get subject indices on expanded data
    subjinds, nsubj = get_subjinds(expanded_data)
    
    # Step 8: Handle weights
    SubjectWeights, ObservationWeights = check_weight_exclusivity(SubjectWeights, ObservationWeights, nsubj)
    
    # Step 9: Prepare censoring patterns for expanded space
    if CensoringPatterns === nothing
        CensoringPatterns_expanded = phase_censoring_patterns
    else
        # Merge user patterns with phase uncertainty patterns
        CensoringPatterns_expanded = _merge_censoring_patterns(CensoringPatterns, phase_censoring_patterns, mappings)
    end
    
    # Validate inputs
    CensoringPatterns_final = _prepare_censoring_patterns(CensoringPatterns_expanded, mappings.n_expanded)
    _validate_inputs!(expanded_data, mappings.expanded_tmat, CensoringPatterns_final, SubjectWeights, ObservationWeights; verbose = verbose)
    
    # Step 10: Build emission matrix
    emat = build_emat(expanded_data, CensoringPatterns_final, EmissionMatrix, mappings.expanded_tmat)
    
    # Step 11: Build total hazards on expanded space
    expanded_totalhazards = build_totalhazards(expanded_hazards, mappings.expanded_tmat)
    
    # Step 12: Store modelcall
    modelcall = (
        hazards = hazards, 
        data = data, 
        constraints = constraints, 
        SubjectWeights = SubjectWeights, 
        ObservationWeights = ObservationWeights, 
        CensoringPatterns = CensoringPatterns, 
        EmissionMatrix = EmissionMatrix
    )
    
    # Step 13: Build original parameters structure (for user-facing API)
    original_parameters = _build_original_parameters(hazards_ordered, data)
    
    # Step 14: Assemble the PhaseTypeModel
    # Note: `data` and `tmat` fields contain expanded versions for loglik_markov compatibility
    # Original user data/tmat stored in `original_data` and `original_tmat` fields
    model = PhaseTypeModel(
        # Expanded state space (standard fields for loglik compatibility)
        expanded_data,
        mappings.expanded_tmat,
        expanded_parameters,
        nothing,  # expanded_model - set later if needed
        mappings,
        # Original state space (user-facing)
        data,
        tmat_original,
        original_parameters,
        hazards_ordered,
        # Standard model fields (on expanded space)
        expanded_hazards,
        expanded_totalhazards,
        emat,
        expanded_hazkeys,
        subjinds,
        SubjectWeights,
        ObservationWeights,
        CensoringPatterns_final,
        nothing,  # markovsurrogate - set during fitting
        modelcall
    )
    
    if verbose
        println("  PhaseTypeModel created successfully")
    end
    
    return model
end

"""
    _build_expanded_parameters(params_list, hazkeys, hazards)

Build ParameterHandling-compatible parameter structure for expanded hazards.
"""
function _build_expanded_parameters(params_list::Vector{Vector{Float64}}, 
                                     hazkeys::Dict{Symbol, Int64}, 
                                     hazards::Vector{<:_Hazard})
    # Build nested parameters structure per hazard
    params_nested_pairs = [
        hazname => build_hazard_params(params_list[idx], hazards[idx].parnames, hazards[idx].npar_baseline, length(params_list[idx]))
        for (hazname, idx) in sort(collect(hazkeys), by = x -> x[2])
    ]
    params_nested = NamedTuple(params_nested_pairs)
    
    # Create ReConstructor for AD-compatible flatten/unflatten
    reconstructor = ReConstructor(params_nested, unflattentype=UnflattenFlexible())
    params_flat = flatten(reconstructor, params_nested)
    
    # Get natural scale parameters (family-aware transformation)
    params_natural_pairs = [
        hazname => extract_natural_vector(params_nested[hazname], hazards[idx].family)
        for (hazname, idx) in sort(collect(hazkeys), by = x -> x[2])
    ]
    params_natural = NamedTuple(params_natural_pairs)
    
    return (
        flat = params_flat,
        nested = params_nested,
        natural = params_natural,
        reconstructor = reconstructor
    )
end

"""
    _build_original_parameters(hazards, data)

Build parameter structure for the original (user-facing) hazard specifications.

This is used for storing and reporting parameters in terms of the user's
specification, not the expanded internal representation.
"""
function _build_original_parameters(hazards::Vector{<:HazardFunction}, data::DataFrame)
    # For now, build a placeholder structure
    # Will be properly populated during initialization
    params_list = Vector{Vector{Float64}}()
    hazkeys = Dict{Symbol, Int64}()
    
    for (idx, h) in enumerate(hazards)
        hazname = Symbol("h$(h.statefrom)$(h.stateto)")
        hazkeys[hazname] = idx
        
        if h isa PhaseTypeHazardSpec
            # PT hazard: 2n-1 baseline + covariates
            n = h.n_phases
            npar_baseline = 2 * n - 1
            npar_covar = _count_covariates(h.hazard, data)
            push!(params_list, zeros(Float64, npar_baseline + npar_covar))
        else
            # Other hazards: use their standard parameter count
            npar = _count_hazard_parameters(h, data)
            push!(params_list, zeros(Float64, npar))
        end
    end
    
    # Build simple parameter structure (will be updated during init)
    params_nested_pairs = [
        hazname => (baseline = zeros(1), covariates = Float64[])
        for (hazname, idx) in sort(collect(hazkeys), by = x -> x[2])
    ]
    params_nested = NamedTuple(params_nested_pairs)
    reconstructor = ReConstructor(params_nested, unflattentype=UnflattenFlexible())
    params_flat = flatten(reconstructor, params_nested)
    
    return (
        flat = params_flat,
        nested = params_nested,
        natural = params_nested,  # Placeholder
        reconstructor = reconstructor
    )
end

"""
    _count_covariates(formula, data)

Count the number of covariate parameters in a hazard formula.
"""
function _count_covariates(formula::FormulaTerm, data::DataFrame)
    schema = StatsModels.schema(formula, data)
    hazschema = apply_schema(formula, schema)
    rhs_names = _phasetype_rhs_names(hazschema)
    # Subtract 1 for intercept
    return max(0, length(rhs_names) - 1)
end

"""
    _count_hazard_parameters(h, data)

Count the total number of parameters for a hazard specification.
"""
function _count_hazard_parameters(h::HazardFunction, data::DataFrame)
    ncovar = _count_covariates(h.hazard, data)
    family = h.family isa Symbol ? h.family : Symbol(lowercase(String(h.family)))
    
    if family == :exp
        return 1 + ncovar
    elseif family in (:wei, :gom)
        return 2 + ncovar
    elseif family == :sp
        # Splines have variable number of basis functions
        # This is an approximation; actual count determined during build
        return 5 + ncovar
    else
        return 1 + ncovar  # Default
    end
end

"""
    _merge_censoring_patterns(user_patterns, phase_patterns, mappings)

Merge user-provided censoring patterns with phase uncertainty patterns.

User patterns are on the observed state space; this expands them to the
phase space and combines with phase uncertainty.
"""
function _merge_censoring_patterns(user_patterns::Matrix{<:Real}, 
                                    phase_patterns::Matrix{Float64},
                                    mappings::PhaseTypeMappings)
    n_user = size(user_patterns, 1)
    n_expanded = mappings.n_expanded
    
    # Create expanded patterns
    expanded = zeros(Float64, n_user, n_expanded)
    
    for p in 1:n_user
        for s_obs in 1:mappings.n_observed
            obs_prob = user_patterns[p, s_obs]
            phases = mappings.state_to_phases[s_obs]
            n_phases = length(phases)
            # Divide probability mass equally among phases of this observed state
            # This preserves the total probability mass for each observed state
            for phase in phases
                expanded[p, phase] = obs_prob / n_phases
            end
        end
    end
    
    return expanded
end

# Internal: Select optimal number of phases for a single transient state via BIC
# Fits 1, 2, ..., max_phases phase models and selects by BIC
function _select_n_phases_for_state(tmat::Matrix{Int64}, data::DataFrame, 
                                    target_state::Int, max_phases::Int;
                                    verbose::Bool=true)
    n_states = size(tmat, 1)
    is_absorbing = [all(tmat[s, :] .== 0) for s in 1:n_states]
    n_obs = nrow(data)
    
    best_bic = Inf
    best_n_phases = 1
    results = Vector{NamedTuple{(:n_phases, :loglik, :n_params, :bic), Tuple{Int, Float64, Int, Float64}}}()
    
    for n_phases in 1:max_phases
        # Build n_phases_per_state with target_state having n_phases, others having 1
        n_phases_per_state = ones(Int, n_states)
        for s in 1:n_states
            if is_absorbing[s]
                n_phases_per_state[s] = 1
            elseif s == target_state
                n_phases_per_state[s] = n_phases
            else
                n_phases_per_state[s] = 1  # Will be optimized separately
            end
        end
        
        # Build surrogate with this configuration
        config = PhaseTypeConfig(n_phases=n_phases_per_state, constraints=true, max_phases=max_phases)
        surrogate = build_phasetype_surrogate(tmat, config)
        
        # Compute log-likelihood
        ll = loglik_phasetype_panel(surrogate, data; neg=false)
        
        # Count parameters: (2p - 1) for this state's Coxian
        n_params = 2 * n_phases - 1
        
        # BIC = -2 * loglik + k * log(n)
        bic = -2 * ll + n_params * log(n_obs)
        
        push!(results, (n_phases=n_phases, loglik=ll, n_params=n_params, bic=bic))
        
        if bic < best_bic
            best_bic = bic
            best_n_phases = n_phases
        end
    end
    
    if verbose
        println("  State $target_state: selected $best_n_phases phases (BIC)")
    end
    
    return best_n_phases
end

# Internal: Select optimal number of phases for all transient states via BIC
# Optimizes each state sequentially
function _select_n_phases_bic(tmat::Matrix{Int64}, data::DataFrame;
                              max_phases::Int=5, verbose::Bool=true)
    n_states = size(tmat, 1)
    is_absorbing = [all(tmat[s, :] .== 0) for s in 1:n_states]
    transient_states = findall(.!is_absorbing)
    
    if verbose
        println("\nPhase-type model selection (BIC):")
        println("─" ^ 40)
    end
    
    # Optimize each transient state sequentially
    n_phases_per_state = ones(Int, n_states)
    for s in transient_states
        n_phases_per_state[s] = _select_n_phases_for_state(
            tmat, data, s, max_phases; verbose=verbose)
    end
    
    if verbose
        println("─" ^ 40)
        transient_phases = n_phases_per_state[transient_states]
        println("Selected phases: ", transient_phases)
    end
    
    # Return as Vector for transient states only (matches PhaseTypeConfig expectation)
    return n_phases_per_state[transient_states]
end

# =============================================================================
# Coxian Phase-Type Construction
# =============================================================================

"""
    build_coxian_intensity(λ::Vector{Float64}, μ::Vector{Float64})

Build the full intensity matrix Q for a Coxian phase-type distribution.

# Arguments
- `λ::Vector{Float64}`: Progression rates between consecutive phases (length p-1)
- `μ::Vector{Float64}`: Absorption rates from each phase (length p)

# Returns
- `Q::Matrix{Float64}`: (p+1) × (p+1) intensity matrix (including absorbing state)

# Structure
For a p-phase Coxian, the (p+1)×(p+1) intensity matrix Q is:
```
Q = [-(λ₁+μ₁)    λ₁        0    ...    0      μ₁  ]
    [    0    -(λ₂+μ₂)    λ₂   ...    0      μ₂  ]
    [    ⋮        ⋮        ⋱    ⋱      ⋮       ⋮   ]
    [    0        0       ...  -(λₚ₋₁+μₚ₋₁) λₚ₋₁  μₚ₋₁]
    [    0        0       ...     0     -μₚ    μₚ  ]
    [    0        0       ...     0      0      0  ]
```

The last row is the absorbing state (all zeros).
"""
function build_coxian_intensity(λ::Vector{Float64}, μ::Vector{Float64})
    p = length(μ)
    length(λ) == p - 1 || throw(DimensionMismatch("λ must have length $(p-1)"))
    
    Q = zeros(p + 1, p + 1)
    
    for i in 1:p
        # Diagonal: -(progression rate + absorption rate)
        if i < p
            Q[i, i] = -(λ[i] + μ[i])
            Q[i, i+1] = λ[i]          # Progression to next phase
        else
            Q[i, i] = -μ[i]           # Last phase: only absorption
        end
        Q[i, p+1] = μ[i]              # Absorption (transition to absorbing state)
    end
    # Last row (absorbing state) is already zeros
    
    return Q
end

# Legacy alias for backward compatibility
build_coxian_subintensity(λ::Vector{Float64}, μ::Vector{Float64}) = 
    build_coxian_intensity(λ, μ)[1:end-1, 1:end-1]

# =============================================================================
# Phase-Type Surrogate Struct (model integration in surrogates.jl)
# =============================================================================

"""
    PhaseTypeSurrogate

A phase-type augmented surrogate model for improved importance sampling.

This extends the Markov surrogate by replacing each transient state with
multiple latent phases. The expanded state space allows better approximation
of non-exponential sojourn distributions.

# Fields
- `phasetype_dists::Dict{Int, PhaseTypeDistribution}`: PH distribution per observed state
- `n_observed_states::Int`: Number of observed states
- `n_expanded_states::Int`: Total expanded states (sum of phases)
- `state_to_phases::Vector{UnitRange{Int}}`: Mapping observed state → phase indices
- `phase_to_state::Vector{Int}`: Mapping phase index → observed state
- `expanded_Q::Matrix{Float64}`: Expanded intensity matrix
- `config::PhaseTypeConfig`: Configuration used to build this surrogate

# Example
```julia
config = PhaseTypeConfig(n_phases=3)
surrogate = build_phasetype_surrogate(model, config)
```

See also: [`PhaseTypeConfig`](@ref), [`build_phasetype_surrogate`](@ref)
"""
struct PhaseTypeSurrogate
    phasetype_dists::Dict{Int, PhaseTypeDistribution}
    n_observed_states::Int
    n_expanded_states::Int
    state_to_phases::Vector{UnitRange{Int}}
    phase_to_state::Vector{Int}
    expanded_Q::Matrix{Float64}
    config::PhaseTypeConfig
end

"""
    collapse_phases(times::Vector{Float64}, states::Vector{Int}, 
                    surrogate::PhaseTypeSurrogate)

Collapse an expanded (state, phase) path to observed states only.

# Arguments
- `times::Vector{Float64}`: Jump times in expanded space
- `states::Vector{Int}`: State sequence in expanded space (phase indices)
- `surrogate::PhaseTypeSurrogate`: The phase-type surrogate

# Returns
- `(collapsed_times, collapsed_states)`: Path with only observed state transitions
"""
function collapse_phases(times::Vector{Float64}, states::Vector{Int}, 
                        surrogate::PhaseTypeSurrogate)
    collapsed_times = Float64[]
    collapsed_states = Int[]
    
    push!(collapsed_times, times[1])
    push!(collapsed_states, surrogate.phase_to_state[states[1]])
    
    for i in 2:length(times)
        new_state = surrogate.phase_to_state[states[i]]
        
        # Only record if observed state changed
        if new_state != collapsed_states[end]
            push!(collapsed_times, times[i])
            push!(collapsed_states, new_state)
        end
    end
    
    return collapsed_times, collapsed_states
end

"""
    expand_initial_state(observed_state::Int, surrogate::PhaseTypeSurrogate)

Map an observed initial state to its first phase in expanded space.

# Arguments
- `observed_state::Int`: State in original state space
- `surrogate::PhaseTypeSurrogate`: The phase-type surrogate

# Returns
- `Int`: First phase index for this state
"""
function expand_initial_state(observed_state::Int, surrogate::PhaseTypeSurrogate)
    return first(surrogate.state_to_phases[observed_state])
end

# =============================================================================
# Building Phase-Type Surrogates
# =============================================================================

"""
    build_phasetype_surrogate(tmat::Matrix{Int64}, config::PhaseTypeConfig;
                              data::Union{Nothing, DataFrame}=nothing,
                              hazards::Union{Nothing, Vector}=nothing,
                              verbose::Bool=true)

Construct a PhaseTypeSurrogate from a transition matrix and configuration.

# Arguments
- `tmat::Matrix{Int64}`: Transition matrix where tmat[i,j] > 0 indicates transition i→j is allowed
- `config::PhaseTypeConfig`: Configuration specifying number of phases per state
- `data::Union{Nothing, DataFrame}=nothing`: Required when config.n_phases=:auto for BIC selection
- `hazards::Union{Nothing, Vector}=nothing`: Required when config.n_phases=:heuristic
- `verbose::Bool=true`: Print selection results when using :auto or :heuristic

# Returns
- `PhaseTypeSurrogate`: Surrogate with expanded state space and mappings

# Structure
For each transient observed state, we expand into multiple latent phases.
Absorbing states remain as single states. The expanded Q matrix has structure:

```
Observed states:  1 (transient)  →  2 (transient)  →  3 (absorbing)
                    ↓                  ↓
Expanded:        [1a→1b→1c]     →  [2a→2b]        →  [3]
                 (3 phases)        (2 phases)        (1 phase)
```

# Example
```julia
tmat = [0 1 1; 0 0 1; 0 0 0]  # 1→2, 1→3, 2→3

# With explicit n_phases (for inference)
config = PhaseTypeConfig(n_phases=3)
surrogate = build_phasetype_surrogate(tmat, config)

# With BIC-based auto-selection (for MCEM proposals)
config = PhaseTypeConfig(n_phases=:auto)
surrogate = build_phasetype_surrogate(tmat, config; data=my_data)

# With heuristic defaults based on hazard types (faster, no model fitting)
config = PhaseTypeConfig(n_phases=:heuristic)
surrogate = build_phasetype_surrogate(tmat, config; hazards=model.hazards)
```

See also: [`PhaseTypeConfig`](@ref), [`PhaseTypeSurrogate`](@ref)
"""
function build_phasetype_surrogate(tmat::Matrix{Int64}, config::PhaseTypeConfig;
                                   data::Union{Nothing, DataFrame}=nothing,
                                   hazards::Union{Nothing, Vector}=nothing,
                                   verbose::Bool=true)
    n_states = size(tmat, 1)
    
    # Handle :auto - select via BIC
    if config.n_phases === :auto
        if data === nothing
            throw(ArgumentError("data is required when n_phases=:auto for BIC-based selection"))
        end
        n_phases = _select_n_phases_bic(tmat, data; max_phases=config.max_phases, verbose=verbose)
        config = PhaseTypeConfig(n_phases=n_phases, constraints=config.constraints, 
                                 max_phases=config.max_phases)
    # Handle :heuristic - use smart defaults based on hazard types
    elseif config.n_phases === :heuristic
        if hazards === nothing
            throw(ArgumentError("hazards is required when n_phases=:heuristic"))
        end
        n_phases = _compute_default_n_phases(tmat, hazards)
        config = PhaseTypeConfig(n_phases=n_phases, constraints=config.constraints, 
                                 max_phases=config.max_phases)
        if verbose
            println("Heuristic n_phases per state: $n_phases")
        end
    end
    
    # Identify transient vs absorbing states
    # A state is absorbing if it has no outgoing transitions
    is_absorbing = [all(tmat[s, :] .== 0) for s in 1:n_states]
    transient_states = findall(.!is_absorbing)
    absorbing_states = findall(is_absorbing)
    
    # Determine number of phases per state
    n_phases_per_state = _get_n_phases_per_state(n_states, is_absorbing, config)
    
    # Build state mappings
    state_to_phases, phase_to_state, n_expanded = _build_state_mappings(
        n_states, n_phases_per_state)
    
    # Initialize PH distributions for transient states
    # Rates will be estimated during inference
    phasetype_dists = Dict{Int, PhaseTypeDistribution}()
    for s in transient_states
        phasetype_dists[s] = _build_default_phasetype(n_phases_per_state[s])
    end
    
    # Build the expanded Q matrix
    expanded_Q = build_expanded_Q(tmat, n_phases_per_state, state_to_phases, 
                                  phase_to_state, phasetype_dists, n_expanded)
    
    return PhaseTypeSurrogate(
        phasetype_dists,
        n_states,
        n_expanded,
        state_to_phases,
        phase_to_state,
        expanded_Q,
        config
    )
end

"""
    _get_n_phases_per_state(n_states, is_absorbing, config)

Determine number of phases for each observed state based on configuration.
"""
function _get_n_phases_per_state(n_states::Int, is_absorbing::Vector{Bool}, 
                                 config::PhaseTypeConfig)
    n_phases_per_state = zeros(Int, n_states)
    
    if config.n_phases isa Symbol
        # :auto - default to 2 phases for transient states
        for s in 1:n_states
            n_phases_per_state[s] = is_absorbing[s] ? 1 : 2
        end
    elseif config.n_phases isa Int
        # Same number of phases for all transient states
        for s in 1:n_states
            n_phases_per_state[s] = is_absorbing[s] ? 1 : config.n_phases
        end
    else
        # Per-state specification (Vector{Int})
        transient_idx = 1
        for s in 1:n_states
            if is_absorbing[s]
                n_phases_per_state[s] = 1
            else
                n_phases_per_state[s] = config.n_phases[transient_idx]
                transient_idx += 1
            end
        end
    end
    
    return n_phases_per_state
end

"""
    _build_state_mappings(n_states, n_phases_per_state)

Build bidirectional mappings between observed states and expanded phases.

# Returns
- `state_to_phases::Vector{UnitRange{Int}}`: observed state → range of phase indices
- `phase_to_state::Vector{Int}`: phase index → observed state
- `n_expanded::Int`: total number of expanded states
"""
function _build_state_mappings(n_states::Int, n_phases_per_state::Vector{Int})
    state_to_phases = Vector{UnitRange{Int}}(undef, n_states)
    phase_to_state = Int[]
    
    current_phase = 1
    for s in 1:n_states
        n_phases = n_phases_per_state[s]
        state_to_phases[s] = current_phase:(current_phase + n_phases - 1)
        
        # Add mappings from phases back to observed state
        for _ in 1:n_phases
            push!(phase_to_state, s)
        end
        
        current_phase += n_phases
    end
    
    n_expanded = current_phase - 1
    return state_to_phases, phase_to_state, n_expanded
end

"""
    _build_default_phasetype(n_phases)

Build a default Coxian phase-type distribution with unit mean.

For Coxian structure, from each phase i you can:
1. Progress to phase i+1 with rate λᵢ  
2. Absorb (exit) with rate μᵢ

This allows a mixture of sojourn times (some short via early absorption,
some long via progression through all phases).
"""
function _build_default_phasetype(n_phases::Int)
    if n_phases == 1
        # Single phase = exponential with rate 1
        # Q is 2×2: phase + absorbing state
        Q = [-1.0  1.0;
              0.0  0.0]
        return PhaseTypeDistribution(1, Q, [1.0])
    else
        # Coxian-n: progression rates and absorption from each phase
        # Choose rates so mean ≈ 1 with balanced absorption
        
        # Base rate scaled by number of phases
        base_rate = Float64(n_phases)
        
        # Progression rates (λ): rate of moving to next phase
        # Absorption rates (μ): rate of exiting from each phase
        λ = Vector{Float64}(undef, n_phases - 1)
        μ = Vector{Float64}(undef, n_phases)
        
        # Coxian structure: higher absorption probability from later phases
        # This creates a mixture that can approximate various distributions
        for i in 1:(n_phases - 1)
            # Progression rate decreases with phase (spend more time in later phases)
            λ[i] = base_rate * (n_phases - i) / n_phases
            # Absorption rate increases with phase
            μ[i] = base_rate * i / (2 * n_phases)
        end
        # Last phase: must absorb (no further progression)
        μ[n_phases] = base_rate
        
        Q = build_coxian_intensity(λ, μ)
        initial = zeros(n_phases)
        initial[1] = 1.0  # Always start in phase 1
        
        return PhaseTypeDistribution(n_phases, Q, initial)
    end
end

"""
    build_expanded_Q(tmat, n_phases_per_state, state_to_phases, phase_to_state,
                     phasetype_dists, n_expanded)

Construct the expanded intensity matrix Q for the phase-type augmented model.

The expanded Q matrix combines:
1. Within-state phase transitions (from PH sub-intensity matrices)
2. Between-state transitions (at absorption from last phase or any phase with absorption)

# Structure for Coxian PH
For a state with p phases and transitions to states j₁, j₂, ...:
- Phases 1 to p-1 can transition to next phase OR absorb (→ transition to another state)
- Phase p must absorb (→ transition to another state)
- Absorption rates are distributed among destination states proportionally

# Arguments
- `tmat`: Original transition matrix
- `n_phases_per_state`: Number of phases per observed state
- `state_to_phases`: Mapping from observed state to phase indices
- `phase_to_state`: Mapping from phase index to observed state
- `phasetype_dists`: Dictionary of PhaseTypeDistribution per transient state
- `n_expanded`: Total number of expanded states
- `transition_rates`: Optional dictionary mapping (statefrom, stateto) → rate for 
   weighted distribution of absorption rates among destinations

# Returns
- `Matrix{Float64}`: Expanded intensity matrix (n_expanded × n_expanded)
"""
function build_expanded_Q(tmat::Matrix{Int64}, 
                          n_phases_per_state::Vector{Int},
                          state_to_phases::Vector{UnitRange{Int}},
                          phase_to_state::Vector{Int},
                          phasetype_dists::Dict{Int, PhaseTypeDistribution},
                          n_expanded::Int;
                          transition_rates::Union{Nothing, Dict{Tuple{Int,Int}, Float64}}=nothing)
    
    n_states = size(tmat, 1)
    Q = zeros(Float64, n_expanded, n_expanded)
    
    for s in 1:n_states
        phases = state_to_phases[s]
        n_phases = n_phases_per_state[s]
        
        # Find destination states from this observed state
        dest_states = findall(tmat[s, :] .> 0)
        
        if isempty(dest_states)
            # Absorbing state: no transitions
            continue
        end
        
        if !haskey(phasetype_dists, s)
            # No PH distribution for this state (shouldn't happen for transient)
            continue
        end
        
        ph = phasetype_dists[s]
        abs_rates = absorption_rates(ph)
        S = subintensity(ph)  # Extract subintensity for within-state transitions
        
        # Fill in within-state phase transitions from S matrix
        for i in 1:n_phases
            phase_i = phases[i]
            
            # Diagonal (will be set at the end based on row sums)
            # For now, copy off-diagonal transitions within state
            for j in 1:n_phases
                if i != j
                    phase_j = phases[j]
                    # S[i,j] is the rate from phase i to phase j within the state
                    if S[i, j] > 0
                        Q[phase_i, phase_j] = S[i, j]
                    end
                end
            end
            
            # Between-state transitions (absorption from this phase)
            # Distribute absorption rate among destination states
            abs_rate = abs_rates[i]
            if abs_rate > 0 && !isempty(dest_states)
                if isnothing(transition_rates)
                    # Distribute equally among destinations
                    rate_per_dest = abs_rate / length(dest_states)
                    for d in dest_states
                        dest_phase = first(state_to_phases[d])
                        Q[phase_i, dest_phase] = rate_per_dest
                    end
                else
                    # Distribute according to transition_rates from Markov surrogate
                    total_rate_out = sum(get(transition_rates, (s, d), 0.0) for d in dest_states)
                    if total_rate_out > 0
                        for d in dest_states
                            rate_sd = get(transition_rates, (s, d), 0.0)
                            if rate_sd > 0
                                dest_phase = first(state_to_phases[d])
                                # Proportion of absorption going to state d
                                prop = rate_sd / total_rate_out
                                Q[phase_i, dest_phase] = abs_rate * prop
                            end
                        end
                    else
                        # Fallback to equal distribution
                        rate_per_dest = abs_rate / length(dest_states)
                        for d in dest_states
                            dest_phase = first(state_to_phases[d])
                            Q[phase_i, dest_phase] = rate_per_dest
                        end
                    end
                end
            end
        end
    end
    
    # Set diagonal elements: negative sum of off-diagonal row entries
    for i in 1:n_expanded
        Q[i, i] = -sum(Q[i, j] for j in 1:n_expanded if j != i)
    end
    
    return Q
end

"""
    update_expanded_Q!(surrogate::PhaseTypeSurrogate, 
                       phasetype_dists::Dict{Int, PhaseTypeDistribution},
                       transition_rates::Dict{Tuple{Int,Int}, Float64})

Update the expanded Q matrix with new phase-type distributions and transition rates.

# Arguments
- `surrogate`: PhaseTypeSurrogate to update (modifies expanded_Q in place conceptually,
               but since the struct is immutable, returns a new surrogate)
- `phasetype_dists`: Updated PH distributions per state
- `transition_rates`: Dictionary mapping (statefrom, stateto) → rate

# Returns  
- `PhaseTypeSurrogate`: New surrogate with updated Q matrix
"""
function update_expanded_Q(surrogate::PhaseTypeSurrogate,
                           phasetype_dists::Dict{Int, PhaseTypeDistribution},
                           transition_rates::Dict{Tuple{Int,Int}, Float64})
    
    n_expanded = surrogate.n_expanded
    n_states = surrogate.n_observed_states
    Q = zeros(Float64, n_expanded, n_expanded)
    
    for s in 1:n_states
        phases = surrogate.state_to_phases[s]
        n_phases = length(phases)
        
        if !haskey(phasetype_dists, s)
            continue  # Absorbing state
        end
        
        ph = phasetype_dists[s]
        abs_rates = absorption_rates(ph)
        S = subintensity(ph)  # Extract subintensity for within-state transitions
        
        # Within-state phase transitions
        for i in 1:n_phases
            phase_i = phases[i]
            for j in 1:n_phases
                if i != j && S[i, j] > 0
                    Q[phase_i, phases[j]] = S[i, j]
                end
            end
            
            # Between-state transitions using provided rates
            abs_rate = abs_rates[i]
            if abs_rate > 0
                # Distribute according to transition_rates
                total_rate_out = sum(get(transition_rates, (s, d), 0.0) 
                                     for d in 1:n_states if d != s)
                
                if total_rate_out > 0
                    for d in 1:n_states
                        rate_sd = get(transition_rates, (s, d), 0.0)
                        if rate_sd > 0
                            dest_phase = first(surrogate.state_to_phases[d])
                            # Proportion of absorption going to state d
                            prop = rate_sd / total_rate_out
                            Q[phase_i, dest_phase] = abs_rate * prop
                        end
                    end
                end
            end
        end
    end
    
    # Set diagonals
    for i in 1:n_expanded
        Q[i, i] = -sum(Q[i, j] for j in 1:n_expanded if j != i)
    end
    
    return PhaseTypeSurrogate(
        phasetype_dists,
        surrogate.n_observed_states,
        surrogate.n_expanded_states,
        surrogate.state_to_phases,
        surrogate.phase_to_state,
        Q,
        surrogate.config
    )
end

# =============================================================================
# Phase-Type Emission Matrix Builder
# =============================================================================

"""
    build_phasetype_emission_matrix(surrogate::PhaseTypeSurrogate, n_obs::Int)

Build an emission matrix for phase-type augmented models.

The emission matrix E has dimensions (n_obs × n_expanded_states) where:
- E[i, p] = P(observation i | latent state p)

For phase-type models, the emission probability is:
- 1.0 if the phase p corresponds to the observed state for observation i
- 0.0 otherwise

This implements a deterministic mapping from latent phases to observed states.

# Arguments
- `surrogate::PhaseTypeSurrogate`: The phase-type surrogate model
- `n_obs::Int`: Number of observations
- `observed_states::Vector{Int}`: Observed state for each observation (optional)

# Returns
- `Matrix{Float64}`: Emission matrix (n_obs × n_expanded_states)

# Example
```julia
surrogate = build_phasetype_surrogate(tmat, config)
emat = build_phasetype_emission_matrix(surrogate, 100)
```
"""
function build_phasetype_emission_matrix(surrogate::PhaseTypeSurrogate, 
                                         n_obs::Int;
                                         observed_states::Union{Nothing, Vector{Int}} = nothing)
    n_expanded = surrogate.n_expanded_states
    emat = zeros(Float64, n_obs, n_expanded)
    
    if observed_states === nothing
        # No specific observations - allow all phases for all observations
        # This is used when we don't have observation-specific constraints
        for i in 1:n_obs
            emat[i, :] .= 1.0
        end
    else
        # Set emission probabilities based on observed states
        for i in 1:n_obs
            obs_state = observed_states[i]
            # All phases of the observed state have emission probability 1
            for p in surrogate.state_to_phases[obs_state]
                emat[i, p] = 1.0
            end
        end
    end
    
    return emat
end

"""
    build_phasetype_emission_matrix(surrogate::PhaseTypeSurrogate, 
                                    data::DataFrame;
                                    state_column::Symbol = :stateto)

Build emission matrix from a DataFrame with observed states.

# Arguments
- `surrogate`: PhaseTypeSurrogate
- `data`: DataFrame containing observations
- `state_column`: Column name containing observed states (default: :stateto)

# Returns
- `Matrix{Float64}`: Emission matrix (nrow(data) × n_expanded_states)
"""
function build_phasetype_emission_matrix(surrogate::PhaseTypeSurrogate,
                                         data::DataFrame;
                                         state_column::Symbol = :stateto)
    observed_states = Vector{Int}(data[!, state_column])
    return build_phasetype_emission_matrix(surrogate, nrow(data); 
                                           observed_states = observed_states)
end

"""
    build_phasetype_emission_matrix_censored(surrogate::PhaseTypeSurrogate,
                                             data::DataFrame,
                                             CensoringPatterns::Matrix{Float64};
                                             state_column::Symbol = :stateto,
                                             obstype_column::Symbol = :obstype)

Build emission matrix for phase-type models with censored observations.

Handles different observation types:
- obstype 1, 2: Exact observation → phases of observed state have prob 1
- obstype 0: Fully censored → all phases have prob 1  
- obstype > 2: Partial censoring → use CensoringPatterns to determine allowed states

# Arguments
- `surrogate`: PhaseTypeSurrogate
- `data`: DataFrame with observations
- `CensoringPatterns`: Matrix defining which states are possible for each censoring pattern
- `state_column`: Column with observed state (default: :stateto)
- `obstype_column`: Column with observation type (default: :obstype)

# Returns
- `Matrix{Float64}`: Emission matrix accounting for censoring
"""
function build_phasetype_emission_matrix_censored(
        surrogate::PhaseTypeSurrogate,
        data::DataFrame,
        CensoringPatterns::Matrix{Float64};
        state_column::Symbol = :stateto,
        obstype_column::Symbol = :obstype)
    
    n_obs = nrow(data)
    n_expanded = surrogate.n_expanded_states
    n_states = surrogate.n_observed_states
    emat = zeros(Float64, n_obs, n_expanded)
    
    for i in 1:n_obs
        obstype = data[i, obstype_column]
        
        if obstype ∈ [1, 2]
            # Exact observation
            obs_state = data[i, state_column]
            for p in surrogate.state_to_phases[obs_state]
                emat[i, p] = 1.0
            end
        elseif obstype == 0
            # Fully censored - all states possible
            emat[i, :] .= 1.0
        else
            # Partial censoring - use CensoringPatterns
            pattern_idx = obstype - 2
            for s in 1:n_states
                if CensoringPatterns[pattern_idx, s + 1] > 0
                    for p in surrogate.state_to_phases[s]
                        emat[i, p] = CensoringPatterns[pattern_idx, s + 1]
                    end
                end
            end
        end
    end
    
    return emat
end

"""
    collapse_emission_matrix(emat_expanded::Matrix{Float64}, 
                            surrogate::PhaseTypeSurrogate)

Collapse an expanded emission matrix back to observed state space.

For each observation and observed state, takes the maximum emission probability
across all phases of that state.

# Arguments
- `emat_expanded`: Emission matrix on expanded state space
- `surrogate`: PhaseTypeSurrogate with state mappings

# Returns
- `Matrix{Float64}`: Emission matrix (n_obs × n_observed_states)
"""
function collapse_emission_matrix(emat_expanded::Matrix{Float64},
                                  surrogate::PhaseTypeSurrogate)
    n_obs = size(emat_expanded, 1)
    n_states = surrogate.n_observed_states
    emat_collapsed = zeros(Float64, n_obs, n_states)
    
    for i in 1:n_obs
        for s in 1:n_states
            # Max probability across all phases of this state
            emat_collapsed[i, s] = maximum(emat_expanded[i, p] 
                                           for p in surrogate.state_to_phases[s])
        end
    end
    
    return emat_collapsed
end

# =============================================================================
# Phase-Type Log-Likelihood (Forward Algorithm)
# =============================================================================

"""
    loglik_phasetype(Q::Matrix{Float64}, emat::Matrix{Float64},
                     times::Vector{Float64}, statefrom::Vector{Int},
                     stateto::Vector{Int}, obstype::Vector{Int},
                     surrogate::PhaseTypeSurrogate;
                     neg::Bool = true)

Compute the log-likelihood for a single subject using the forward algorithm
on the phase-type expanded state space.

This implements the forward algorithm from Jackson (2025) and Lange et al.,
treating the phase-type model as a hidden Markov model where:
- Latent states are the expanded phases
- Observations are the observed states
- Emissions map phases to observed states

# Algorithm
For each time interval [t_{k-1}, t_k]:
1. Compute transition probability matrix P(t) = exp(Q * Δt)
2. Forward recursion: α_k = P(t)' * diag(e_k) * α_{k-1}
   where e_k is the emission vector for observation k
3. Log-likelihood = log(sum(α_K)) at final observation

# Arguments
- `Q::Matrix{Float64}`: Expanded intensity matrix (n_expanded × n_expanded)
- `emat::Matrix{Float64}`: Emission matrix (n_obs × n_expanded)
- `times::Vector{Float64}`: Observation times (length n_obs + 1, includes start time)
- `statefrom::Vector{Int}`: Starting state for each interval (observed states)
- `stateto::Vector{Int}`: Ending state for each interval (observed states, -1 for censored)
- `obstype::Vector{Int}`: Observation type (1=exact, 2=panel, 0=censored, >2=partial censor)
- `surrogate::PhaseTypeSurrogate`: Surrogate with state mappings

# Keyword Arguments
- `neg::Bool`: Return negative log-likelihood (default: true for optimization)

# Returns
- `Float64`: Log-likelihood (or negative log-likelihood if neg=true)

# Mathematical Background

For phase-type models interpreted as hidden Markov models (Jackson 2025):
- The hazard at time t is: h(t) = π' exp(St) s / S(t)
- The survival function is: S(t) = π' exp(St) 𝟙
- The forward algorithm computes: P(observations) = ∑_z P(observations, z)
  by marginalizing over latent phase sequences z

The forward variable α_k(i) = P(y_1,...,y_k, z_k = i) satisfies:
  α_k(i) = e_k(i) * ∑_j P_{ji}(Δt_k) * α_{k-1}(j)

where e_k(i) is the emission probability of observation k given phase i.

See also: [`build_phasetype_surrogate`](@ref), [`loglik_phasetype_panel`](@ref)
"""
function loglik_phasetype(Q::Matrix{Float64}, 
                          emat::Matrix{Float64},
                          times::Vector{Float64}, 
                          statefrom::Vector{Int},
                          stateto::Vector{Int}, 
                          obstype::Vector{Int},
                          surrogate::PhaseTypeSurrogate;
                          neg::Bool = true)
    
    n_obs = length(statefrom)
    n_expanded = surrogate.n_expanded_states
    
    # Allocate memory for matrix exponential
    cache = ExponentialUtilities.alloc_mem(similar(Q), ExpMethodGeneric())
    
    # Initialize forward variable (likelihood over phases)
    # Start in phase 1 of the initial observed state
    α = zeros(Float64, n_expanded)
    initial_state = statefrom[1]
    initial_phase = first(surrogate.state_to_phases[initial_state])
    α[initial_phase] = 1.0
    
    # Working arrays
    α_new = zeros(Float64, n_expanded)
    P = similar(Q)       # Transition probability matrix
    Q_scaled = similar(Q)  # For holding Q * Δt
    
    # Forward recursion
    for k in 1:n_obs
        # Time interval
        Δt = times[k + 1] - times[k]
        
        if Δt > 0
            # Compute transition probability matrix P = exp(Q * Δt)
            # Note: exponential! returns result but does NOT modify input
            copyto!(Q_scaled, Q)
            Q_scaled .*= Δt
            copyto!(P, exponential!(Q_scaled, ExpMethodGeneric(), cache))
            
            # Forward step: α_new = P' * diag(e) * α
            # Equivalently: α_new[j] = e[j] * ∑_i P[i,j] * α[i]
            fill!(α_new, 0.0)
            
            if obstype[k] ∈ [1, 2] && stateto[k] > 0
                # Exact or panel observation - state is known
                # Only phases of observed state have non-zero emission
                obs_state = stateto[k]
                allowed_phases = surrogate.state_to_phases[obs_state]
                
                for j in allowed_phases
                    for i in 1:n_expanded
                        α_new[j] += P[i, j] * α[i]
                    end
                    # Emission probability is 1 for allowed phases
                end
            else
                # Censored observation - use emission matrix
                for j in 1:n_expanded
                    e_j = emat[k, j]
                    if e_j > 0
                        for i in 1:n_expanded
                            α_new[j] += P[i, j] * α[i] * e_j
                        end
                    end
                end
            end
            
            # Update forward variable
            copyto!(α, α_new)
        else
            # Δt = 0: just apply emission probabilities
            if obstype[k] ∈ [1, 2] && stateto[k] > 0
                # Exact observation - restrict to phases of observed state
                obs_state = stateto[k]
                for j in 1:n_expanded
                    if surrogate.phase_to_state[j] != obs_state
                        α[j] = 0.0
                    end
                end
            else
                # Apply emission probabilities
                for j in 1:n_expanded
                    α[j] *= emat[k, j]
                end
            end
        end
        
        # Rescale to prevent underflow (optional, can use log-sum-exp instead)
        scale = sum(α)
        if scale > 0
            α ./= scale
        end
    end
    
    # Log-likelihood = log(sum(final forward variable))
    # Note: Due to rescaling, we need to track the accumulated log-scale
    # For simplicity, recompute without rescaling if needed
    ll = log(sum(α))
    
    return neg ? -ll : ll
end

"""
    loglik_phasetype_stable(Q::Matrix{Float64}, 
                            emat::Matrix{Float64},
                            times::Vector{Float64}, 
                            statefrom::Vector{Int},
                            stateto::Vector{Int}, 
                            obstype::Vector{Int},
                            surrogate::PhaseTypeSurrogate;
                            neg::Bool = true)

Numerically stable version of loglik_phasetype using log-scaling.

Uses the log-sum-exp trick to prevent numerical underflow in long sequences.
The log-likelihood is accumulated incrementally by tracking the log of the
scaling factor at each step.

# Arguments
Same as `loglik_phasetype`

# Returns
- `Float64`: Log-likelihood (properly accumulated, no numerical issues)
"""
function loglik_phasetype_stable(Q::Matrix{Float64}, 
                                 emat::Matrix{Float64},
                                 times::Vector{Float64}, 
                                 statefrom::Vector{Int},
                                 stateto::Vector{Int}, 
                                 obstype::Vector{Int},
                                 surrogate::PhaseTypeSurrogate;
                                 neg::Bool = true)
    
    n_obs = length(statefrom)
    n_expanded = surrogate.n_expanded_states
    
    # Allocate memory for matrix exponential
    cache = ExponentialUtilities.alloc_mem(similar(Q), ExpMethodGeneric())
    
    # Initialize forward variable
    α = zeros(Float64, n_expanded)
    initial_state = statefrom[1]
    initial_phase = first(surrogate.state_to_phases[initial_state])
    α[initial_phase] = 1.0
    
    # Accumulated log-likelihood from scaling
    log_ll = 0.0
    
    # Working arrays
    α_new = zeros(Float64, n_expanded)
    P = similar(Q)
    Q_scaled = similar(Q)  # For holding Q * Δt
    
    # Forward recursion with log-scaling
    for k in 1:n_obs
        Δt = times[k + 1] - times[k]
        
        if Δt > 0
            # Compute P = exp(Q * Δt)
            # Note: exponential! returns result but does NOT modify input
            copyto!(Q_scaled, Q)
            Q_scaled .*= Δt
            copyto!(P, exponential!(Q_scaled, ExpMethodGeneric(), cache))
            
            # Forward step
            fill!(α_new, 0.0)
            
            if obstype[k] ∈ [1, 2] && stateto[k] > 0
                obs_state = stateto[k]
                allowed_phases = surrogate.state_to_phases[obs_state]
                
                for j in allowed_phases
                    for i in 1:n_expanded
                        α_new[j] += P[i, j] * α[i]
                    end
                end
            else
                for j in 1:n_expanded
                    e_j = emat[k, j]
                    if e_j > 0
                        for i in 1:n_expanded
                            α_new[j] += P[i, j] * α[i] * e_j
                        end
                    end
                end
            end
            
            copyto!(α, α_new)
        else
            if obstype[k] ∈ [1, 2] && stateto[k] > 0
                obs_state = stateto[k]
                for j in 1:n_expanded
                    if surrogate.phase_to_state[j] != obs_state
                        α[j] = 0.0
                    end
                end
            else
                for j in 1:n_expanded
                    α[j] *= emat[k, j]
                end
            end
        end
        
        # Log-scale normalization
        scale = sum(α)
        if scale > 0
            log_ll += log(scale)
            α ./= scale
        else
            # All paths have zero probability - return -Inf
            return neg ? Inf : -Inf
        end
    end
    
    # Final forward sum (should be ~1 after rescaling, but include for completeness)
    log_ll += log(sum(α))
    
    return neg ? -log_ll : log_ll
end

"""
    loglik_phasetype_panel(surrogate::PhaseTypeSurrogate,
                           data::DataFrame;
                           SubjectWeights::Union{Nothing, Vector{Float64}} = nothing,
                           CensoringPatterns::Union{Nothing, Matrix{Float64}} = nothing,
                           neg::Bool = true,
                           return_ll_subj::Bool = false)

Compute the phase-type log-likelihood for panel data with multiple subjects.

This is the main entry point for computing phase-type likelihoods on 
MultistateModels-style data.

# Arguments
- `surrogate::PhaseTypeSurrogate`: Phase-type surrogate with expanded Q matrix
- `data::DataFrame`: Panel data with columns id, tstart, tstop, statefrom, stateto, obstype

# Keyword Arguments
- `SubjectWeights`: Optional subject-level weights (default: 1.0 for all)
- `CensoringPatterns`: Optional censoring patterns matrix for obstype > 2
- `neg::Bool`: Return negative log-likelihood (default: true)
- `return_ll_subj::Bool`: Return vector of subject likelihoods (default: false)

# Returns
- `Float64` or `Vector{Float64}`: Total (negative) log-likelihood or per-subject values

# Example
```julia
surrogate = build_phasetype_surrogate(tmat, config)
ll = loglik_phasetype_panel(surrogate, data; neg=false)
```

See also: [`loglik_phasetype`](@ref), [`build_phasetype_surrogate`](@ref)
"""
function loglik_phasetype_panel(surrogate::PhaseTypeSurrogate,
                                data::DataFrame;
                                SubjectWeights::Union{Nothing, Vector{Float64}} = nothing,
                                CensoringPatterns::Union{Nothing, Matrix{Float64}} = nothing,
                                neg::Bool = true,
                                return_ll_subj::Bool = false)
    
    # Build emission matrix
    if CensoringPatterns === nothing
        emat = build_phasetype_emission_matrix(surrogate, nrow(data);
                                               observed_states = Vector{Int}(data.stateto))
    else
        emat = build_phasetype_emission_matrix_censored(surrogate, data, CensoringPatterns)
    end
    
    # Get unique subjects
    subjects = unique(data.id)
    n_subj = length(subjects)
    
    # Initialize weights
    weights = SubjectWeights === nothing ? ones(Float64, n_subj) : SubjectWeights
    
    # Get expanded Q
    Q = surrogate.expanded_Q
    
    # Accumulate log-likelihood
    if return_ll_subj
        ll_subj = zeros(Float64, n_subj)
    end
    ll_total = 0.0
    
    for (s, subj) in enumerate(subjects)
        # Get subject data
        subj_mask = data.id .== subj
        subj_data = data[subj_mask, :]
        n_obs_subj = nrow(subj_data)
        
        # Extract times (include initial time)
        times = vcat(subj_data.tstart[1], subj_data.tstop)
        
        # Extract state info
        statefrom = Vector{Int}(subj_data.statefrom)
        stateto = Vector{Int}(subj_data.stateto)
        obstype = Vector{Int}(subj_data.obstype)
        
        # Get subject-specific emission matrix rows
        subj_emat = emat[subj_mask, :]
        
        # Compute subject log-likelihood (not negated)
        subj_ll = loglik_phasetype_stable(Q, subj_emat, times, statefrom, stateto, 
                                          obstype, surrogate; neg=false)
        
        # Weight and accumulate
        weighted_ll = subj_ll * weights[s]
        
        if return_ll_subj
            ll_subj[s] = weighted_ll
        else
            ll_total += weighted_ll
        end
    end
    
    if return_ll_subj
        return ll_subj
    else
        return neg ? -ll_total : ll_total
    end
end


"""
    compute_phasetype_marginal_loglik(model::MultistateProcess, 
                                      surrogate::PhaseTypeSurrogate,
                                      emat_ph::Matrix{Float64};
                                      expanded_data::Union{Nothing, DataFrame} = nothing,
                                      expanded_subjectindices::Union{Nothing, Vector{UnitRange{Int64}}} = nothing)

Compute the marginal log-likelihood of the observed data under the phase-type surrogate.

This is the normalization constant r(Y|θ') needed for importance sampling:
  log f̂(Y|θ) = log r(Y|θ') + Σᵢ log(mean(νᵢ))

where νᵢ = f(Zᵢ|θ) / h(Zᵢ|θ') are the importance weights.

# Arguments
- `model::MultistateProcess`: The multistate model with observed data
- `surrogate::PhaseTypeSurrogate`: The phase-type surrogate
- `emat_ph::Matrix{Float64}`: Emission matrix mapping expanded states to observations
- `expanded_data`: Optional expanded data (for exact observations)
- `expanded_subjectindices`: Optional subject indices for expanded data

# Returns
- `Float64`: Log marginal likelihood Σᵢ wᵢ * log P(Yᵢ|θ') under phase-type surrogate

# Mathematical Details
For each subject i, the marginal likelihood P(Yᵢ|θ') is computed via the
forward algorithm on the expanded state space, marginalizing over all possible
phase sequences consistent with the observations.
"""
function compute_phasetype_marginal_loglik(model::MultistateProcess, 
                                           surrogate::PhaseTypeSurrogate,
                                           emat_ph::Matrix{Float64};
                                           expanded_data::Union{Nothing, DataFrame} = nothing,
                                           expanded_subjectindices::Union{Nothing, Vector{UnitRange{Int64}}} = nothing)
    
    Q = surrogate.expanded_Q
    n_expanded = surrogate.n_expanded_states
    
    # Use expanded data and indices if provided
    data = isnothing(expanded_data) ? model.data : expanded_data
    subjectindices = isnothing(expanded_subjectindices) ? model.subjectindices : expanded_subjectindices
    
    # Allocate memory for matrix exponential
    cache = ExponentialUtilities.alloc_mem(similar(Q), ExpMethodGeneric())
    
    ll_total = 0.0
    
    for subj_idx in eachindex(subjectindices)
        # Get subject data indices
        subj_inds = subjectindices[subj_idx]
        n_obs = length(subj_inds)
        
        # Extract times (include initial time)
        times = vcat(data.tstart[subj_inds[1]], data.tstop[subj_inds])
        
        # Extract state info
        statefrom_subj = data.statefrom[subj_inds]
        stateto_subj = data.stateto[subj_inds]
        obstype_subj = data.obstype[subj_inds]
        
        # Get subject-specific emission matrix rows
        subj_emat = emat_ph[subj_inds, :]
        
        # Compute subject log-likelihood via forward algorithm
        subj_ll = loglik_phasetype_stable_internal(Q, subj_emat, times, statefrom_subj, 
                                                    stateto_subj, obstype_subj, surrogate, cache)
        
        # Weight and accumulate
        ll_total += subj_ll * model.SubjectWeights[subj_idx]
    end
    
    return ll_total
end


"""
    loglik_phasetype_stable_internal(...)

Internal version of loglik_phasetype_stable that reuses an allocated cache.
"""
function loglik_phasetype_stable_internal(Q::Matrix{Float64}, 
                                          emat::Matrix{Float64},
                                          times::AbstractVector, 
                                          statefrom::AbstractVector,
                                          stateto::AbstractVector, 
                                          obstype::AbstractVector,
                                          surrogate::PhaseTypeSurrogate,
                                          cache)
    
    n_obs = length(statefrom)
    n_expanded = surrogate.n_expanded_states
    
    # Initialize forward variable
    α = zeros(Float64, n_expanded)
    initial_state = statefrom[1]
    initial_phase = first(surrogate.state_to_phases[initial_state])
    α[initial_phase] = 1.0
    
    # Accumulated log-likelihood from scaling
    log_ll = 0.0
    
    # Working arrays
    α_new = zeros(Float64, n_expanded)
    P = similar(Q)
    Q_scaled = similar(Q)
    
    # Forward recursion with log-scaling
    for k in 1:n_obs
        Δt = times[k + 1] - times[k]
        
        if Δt > 0
            # Compute P = exp(Q * Δt)
            copyto!(Q_scaled, Q)
            Q_scaled .*= Δt
            copyto!(P, exponential!(Q_scaled, ExpMethodGeneric(), cache))
            
            # Forward step
            fill!(α_new, 0.0)
            
            if obstype[k] ∈ [1, 2] && stateto[k] > 0
                obs_state = stateto[k]
                allowed_phases = surrogate.state_to_phases[obs_state]
                
                for j in allowed_phases
                    for i in 1:n_expanded
                        α_new[j] += P[i, j] * α[i]
                    end
                end
            else
                for j in 1:n_expanded
                    e_j = emat[k, j]
                    if e_j > 0
                        for i in 1:n_expanded
                            α_new[j] += P[i, j] * α[i] * e_j
                        end
                    end
                end
            end
            
            copyto!(α, α_new)
        else
            if obstype[k] ∈ [1, 2] && stateto[k] > 0
                obs_state = stateto[k]
                for j in 1:n_expanded
                    if surrogate.phase_to_state[j] != obs_state
                        α[j] = 0.0
                    end
                end
            else
                for j in 1:n_expanded
                    α[j] *= emat[k, j]
                end
            end
        end
        
        # Log-scale normalization
        scale = sum(α)
        if scale > 0
            log_ll += log(scale)
            α ./= scale
        else
            return -Inf  # All paths have zero probability
        end
    end
    
    # Final forward sum
    log_ll += log(sum(α))
    
    return log_ll
end

# =============================================================================
# Phase-Type Model Building for Inference
# =============================================================================

"""
    build_phasetype_hazards(tmat::Matrix{Int64}, config::PhaseTypeConfig, 
                            surrogate::PhaseTypeSurrogate;
                            covariate_formula::Union{Nothing, FormulaTerm} = nothing)

Generate Hazard specifications for the expanded phase-type model.

For each transition in the expanded state space, creates an exponential hazard.
The expanded model has two types of transitions:
1. **Progression transitions** (λᵢ): Rate of moving from phase i to phase i+1 within a state
2. **Absorption/exit transitions** (μᵢ): Rate of exiting from phase i to another observed state

# Arguments
- `tmat::Matrix{Int64}`: Original transition matrix
- `config::PhaseTypeConfig`: Configuration with number of phases
- `surrogate::PhaseTypeSurrogate`: Pre-built surrogate with state mappings
- `covariate_formula`: Optional formula for covariates (applied to ALL rates)

# Returns
- `Vector{HazardFunction}`: Vector of Hazard specifications for the expanded model

# Hazard Naming Convention
- Progression: `h{phase_i}{phase_{i+1}}` (e.g., h12 for phase 1 → phase 2 within a state)
- Exit: `h{phase_i}{dest_first_phase}` (e.g., h34 for phase 3 → first phase of next state)
"""
function build_phasetype_hazards(tmat::Matrix{Int64}, config::PhaseTypeConfig,
                                 surrogate::PhaseTypeSurrogate;
                                 covariate_formula::Union{Nothing, FormulaTerm} = nothing)
    
    n_observed = size(tmat, 1)
    n_expanded = surrogate.n_expanded_states
    
    hazards = HazardFunction[]
    
    # Identify transient states (those with outgoing transitions)
    is_absorbing = [all(tmat[s, :] .== 0) for s in 1:n_observed]
    
    for obs_state in 1:n_observed
        if is_absorbing[obs_state]
            continue  # No transitions from absorbing states
        end
        
        phases = surrogate.state_to_phases[obs_state]
        n_phases = length(phases)
        
        # Find destination states from this observed state
        dest_states = findall(tmat[obs_state, :] .> 0)
        n_dests = length(dest_states)
        
        # 1. Progression transitions (λ): phase i → phase i+1 within this state
        for local_i in 1:(n_phases - 1)
            phase_from = phases[local_i]
            phase_to = phases[local_i + 1]
            
            if covariate_formula === nothing
                h = Hazard(@formula(0 ~ 1), "exp", phase_from, phase_to)
            else
                h = Hazard(covariate_formula, "exp", phase_from, phase_to)
            end
            push!(hazards, h)
        end
        
        # 2. Absorption/exit transitions (μ): from each phase to each destination state
        # From each phase i, we can exit to any of the destination states
        for local_i in 1:n_phases
            phase_from = phases[local_i]
            
            for dest_state in dest_states
                # Transition goes to first phase of destination state
                dest_phase = first(surrogate.state_to_phases[dest_state])
                
                if covariate_formula === nothing
                    h = Hazard(@formula(0 ~ 1), "exp", phase_from, dest_phase)
                else
                    h = Hazard(covariate_formula, "exp", phase_from, dest_phase)
                end
                push!(hazards, h)
            end
        end
    end
    
    return hazards
end

"""
    build_expanded_tmat(tmat::Matrix{Int64}, surrogate::PhaseTypeSurrogate)

Build the expanded transition matrix for the phase-type model.

The expanded tmat enumerates transitions in the phase space:
- `tmat_exp[i, j] > 0` if there's a direct transition from phase i to phase j
- The value is the transition number (for indexing hazards)

# Arguments
- `tmat::Matrix{Int64}`: Original transition matrix (observed states)
- `surrogate::PhaseTypeSurrogate`: Surrogate with state/phase mappings

# Returns
- `Matrix{Int64}`: Expanded transition matrix (n_expanded × n_expanded)
"""
function build_expanded_tmat(tmat::Matrix{Int64}, surrogate::PhaseTypeSurrogate)
    n_observed = size(tmat, 1)
    n_expanded = surrogate.n_expanded_states
    
    tmat_exp = zeros(Int64, n_expanded, n_expanded)
    trans_num = 1
    
    # Identify transient states
    is_absorbing = [all(tmat[s, :] .== 0) for s in 1:n_observed]
    
    for obs_state in 1:n_observed
        if is_absorbing[obs_state]
            continue
        end
        
        phases = surrogate.state_to_phases[obs_state]
        n_phases = length(phases)
        dest_states = findall(tmat[obs_state, :] .> 0)
        
        # Progression transitions
        for local_i in 1:(n_phases - 1)
            tmat_exp[phases[local_i], phases[local_i + 1]] = trans_num
            trans_num += 1
        end
        
        # Exit transitions  
        for local_i in 1:n_phases
            for dest_state in dest_states
                dest_phase = first(surrogate.state_to_phases[dest_state])
                tmat_exp[phases[local_i], dest_phase] = trans_num
                trans_num += 1
            end
        end
    end
    
    return tmat_exp
end

"""
    build_phasetype_emat(data::DataFrame, surrogate::PhaseTypeSurrogate,
                         CensoringPatterns::Matrix{Float64})

Build the emission matrix for the phase-type model.

Maps observations (in observed state space) to the expanded phase space.
For exact/panel observations (obstype 1,2), all phases of the observed state
have emission probability 1.

# Arguments
- `data::DataFrame`: Data with statefrom, stateto, obstype columns
- `surrogate::PhaseTypeSurrogate`: Surrogate with state/phase mappings
- `CensoringPatterns::Matrix{Float64}`: Censoring patterns for obstype > 2

# Returns
- `Matrix{Float64}`: Emission matrix (nrow(data) × n_expanded)
"""
function build_phasetype_emat(data::DataFrame, surrogate::PhaseTypeSurrogate,
                              CensoringPatterns::Matrix{Float64})
    
    n_obs = nrow(data)
    n_expanded = surrogate.n_expanded_states
    n_observed = surrogate.n_observed_states
    
    emat = zeros(Float64, n_obs, n_expanded)
    
    for i in 1:n_obs
        obstype = data.obstype[i]
        
        if obstype ∈ [1, 2]
            # Exact/panel observation - phases of observed state have prob 1
            obs_state = data.stateto[i]
            for p in surrogate.state_to_phases[obs_state]
                emat[i, p] = 1.0
            end
        elseif obstype == 0
            # Fully censored - all phases possible
            emat[i, :] .= 1.0
        else
            # Partial censoring - use CensoringPatterns
            # CensoringPatterns rows are indexed by (obstype - 2)
            # CensoringPatterns columns are [code, state1, state2, ...]
            pattern_idx = obstype - 2
            for s in 1:n_observed
                state_prob = size(CensoringPatterns, 2) > s ? CensoringPatterns[pattern_idx, s + 1] : 0.0
                if state_prob > 0
                    for p in surrogate.state_to_phases[s]
                        emat[i, p] = state_prob
                    end
                end
            end
        end
    end
    
    return emat
end

"""
    expand_data_states!(data::DataFrame, surrogate::PhaseTypeSurrogate)

Expand statefrom and stateto columns to use phase indices.

For the expanded Markov model, we need states to refer to phases, not observed states.
This maps each observed state to its first phase.

# Arguments
- `data::DataFrame`: Data to modify (modified in place!)
- `surrogate::PhaseTypeSurrogate`: Surrogate with state/phase mappings

# Returns
- `DataFrame`: Modified data (same reference as input)
"""
function expand_data_states!(data::DataFrame, surrogate::PhaseTypeSurrogate)
    # Map observed states to first phase of each state
    data.statefrom = [first(surrogate.state_to_phases[s]) for s in data.statefrom]
    data.stateto = [first(surrogate.state_to_phases[s]) for s in data.stateto]
    return data
end

# =============================================================================
# Data Expansion for Phase-Type Forward-Backward
# =============================================================================
#
# For phase-type importance sampling, the data must be expanded to properly
# express uncertainty about which phase the subject is in during sojourn times.
#
# **Problem**: With exact observations (obstype=1), the user provides:
#   (tstart=t1, tstop=t2, statefrom=1, stateto=2, obstype=1)
# This says: "Subject was in state 1, then transitioned to state 2 at time t2."
# But for phase-type FFBS, we need to express:
#   - During [t1, t2): subject was in state 1 but UNKNOWN which phase
#   - At t2: subject transitioned to state 2
#
# **Solution**: Expand exact observations into two rows:
#   1. (tstart=t1, tstop=t2-ε, statefrom=1, stateto=0, obstype=censoring_code)
#      → "During this interval, subject was in some state in the censored set"
#   2. (tstart=t2-ε, tstop=t2, statefrom=0, stateto=2, obstype=1)
#      → "At this time, we observe the subject in state 2"
#
# For panel observations (obstype=2), no expansion is needed since we already
# don't know the exact transition time.
# =============================================================================

"""
    expand_data_for_phasetype(data::DataFrame, n_states::Int; 
                              epsilon::Float64 = sqrt(eps()))

Expand data for phase-type forward-backward sampling.

Exact observations (obstype=1) are split into two rows:
1. A sojourn interval where the subject is in `statefrom` but phase is unknown
2. An exact observation of the transition to `stateto`

This ensures the forward-backward algorithm properly accounts for phase
uncertainty during sojourn times.

# Arguments
- `data::DataFrame`: Original data with columns id, tstart, tstop, statefrom, stateto, obstype
- `n_states::Int`: Number of observed states (used to generate censoring patterns)
- `epsilon::Float64`: Time offset for splitting exact observations (default: sqrt(eps()))

# Returns
- `NamedTuple` with fields:
  - `expanded_data::DataFrame`: Data with exact observations expanded
  - `censoring_patterns::Matrix{Float64}`: Censoring patterns for the expanded obstypes
  - `original_row_map::Vector{Int}`: Maps expanded row index to original row index

# Censoring Pattern Convention
For each observed state s, we create a censoring pattern with obstype = 2 + s
that indicates "subject is known to be in state s (but phase is unknown)".

The censoring patterns matrix has structure:
  - Row 1 (obstype=3): state 1 possible (1.0 in column 2, 0.0 elsewhere)
  - Row 2 (obstype=4): state 2 possible (1.0 in column 3, 0.0 elsewhere)
  - ...
  - Row n (obstype=2+n): state n possible

# Example
```julia
# Original exact data
data = DataFrame(
    id = [1, 1],
    tstart = [0.0, 1.0],
    tstop = [1.0, 2.0],
    statefrom = [1, 2],
    stateto = [2, 3],
    obstype = [1, 1]  # Both exact observations
)

result = expand_data_for_phasetype(data, 3)

# Expanded data has 4 rows:
# Row 1: id=1, [0.0, 1.0-ε), statefrom=1, stateto=0, obstype=3 (censored to state 1)
# Row 2: id=1, [1.0-ε, 1.0], statefrom=0, stateto=2, obstype=1 (exact obs of state 2)
# Row 3: id=1, [1.0, 2.0-ε), statefrom=2, stateto=0, obstype=4 (censored to state 2)
# Row 4: id=1, [2.0-ε, 2.0], statefrom=0, stateto=3, obstype=1 (exact obs of state 3)
```

See also: [`build_phasetype_emat`](@ref), [`build_phasetype_emat_expanded`](@ref)
"""
function expand_data_for_phasetype(data::DataFrame, n_states::Int; 
                                   epsilon::Float64 = sqrt(eps()))
    
    # Count how many rows we'll need
    n_original = nrow(data)
    n_exact = count(data.obstype .== 1)
    n_expanded = n_original + n_exact  # Each exact obs becomes 2 rows
    
    # Pre-allocate expanded data columns
    # Get covariate column names (everything except core columns)
    core_cols = [:id, :tstart, :tstop, :statefrom, :stateto, :obstype]
    covar_cols = setdiff(propertynames(data), core_cols)
    
    # Initialize expanded arrays
    exp_id = Vector{eltype(data.id)}(undef, n_expanded)
    exp_tstart = Vector{Float64}(undef, n_expanded)
    exp_tstop = Vector{Float64}(undef, n_expanded)
    exp_statefrom = Vector{Int}(undef, n_expanded)
    exp_stateto = Vector{Int}(undef, n_expanded)
    exp_obstype = Vector{Int}(undef, n_expanded)
    
    # Initialize covariate arrays
    covar_arrays = Dict{Symbol, Vector}()
    for col in covar_cols
        covar_arrays[col] = Vector{eltype(data[!, col])}(undef, n_expanded)
    end
    
    # Map from expanded row to original row
    original_row_map = Vector{Int}(undef, n_expanded)
    
    # Expand the data
    exp_idx = 0
    for orig_idx in 1:n_original
        row = data[orig_idx, :]
        
        if row.obstype == 1
            # Exact observation: split into sojourn + observation
            
            # Row 1: Sojourn interval [tstart, tstop - epsilon)
            # Subject is in statefrom, phase unknown
            # Use censoring code = 2 + statefrom
            exp_idx += 1
            exp_id[exp_idx] = row.id
            exp_tstart[exp_idx] = row.tstart
            exp_tstop[exp_idx] = row.tstop - epsilon
            exp_statefrom[exp_idx] = row.statefrom
            exp_stateto[exp_idx] = 0  # Censored (state unknown at this point)
            exp_obstype[exp_idx] = 2 + row.statefrom  # Censoring pattern for statefrom
            original_row_map[exp_idx] = orig_idx
            
            # Copy covariates
            for col in covar_cols
                covar_arrays[col][exp_idx] = row[col]
            end
            
            # Row 2: Exact observation at [tstop - epsilon, tstop]
            # Transition to stateto observed
            exp_idx += 1
            exp_id[exp_idx] = row.id
            exp_tstart[exp_idx] = row.tstop - epsilon
            exp_tstop[exp_idx] = row.tstop
            exp_statefrom[exp_idx] = 0  # Coming from censored interval
            exp_stateto[exp_idx] = row.stateto
            exp_obstype[exp_idx] = 1  # Exact observation
            original_row_map[exp_idx] = orig_idx
            
            # Copy covariates
            for col in covar_cols
                covar_arrays[col][exp_idx] = row[col]
            end
            
        else
            # Non-exact observation: keep as-is
            exp_idx += 1
            exp_id[exp_idx] = row.id
            exp_tstart[exp_idx] = row.tstart
            exp_tstop[exp_idx] = row.tstop
            exp_statefrom[exp_idx] = row.statefrom
            exp_stateto[exp_idx] = row.stateto
            exp_obstype[exp_idx] = row.obstype
            original_row_map[exp_idx] = orig_idx
            
            # Copy covariates
            for col in covar_cols
                covar_arrays[col][exp_idx] = row[col]
            end
        end
    end
    
    # Build expanded DataFrame
    expanded_data = DataFrame(
        id = exp_id,
        tstart = exp_tstart,
        tstop = exp_tstop,
        statefrom = exp_statefrom,
        stateto = exp_stateto,
        obstype = exp_obstype
    )
    
    # Add covariate columns
    for col in covar_cols
        expanded_data[!, col] = covar_arrays[col]
    end
    
    # Build censoring patterns matrix
    # Each row corresponds to obstype = 3, 4, ..., 2 + n_states
    # Column 1 is the code (not used), columns 2:n_states+1 are state indicators
    censoring_patterns = zeros(Float64, n_states, n_states + 1)
    for s in 1:n_states
        censoring_patterns[s, 1] = s + 2.0  # obstype code (for reference)
        censoring_patterns[s, s + 1] = 1.0  # state s is possible
    end
    
    return (
        expanded_data = expanded_data,
        censoring_patterns = censoring_patterns,
        original_row_map = original_row_map
    )
end

"""
    needs_data_expansion_for_phasetype(data::DataFrame) -> Bool

Check if the data contains exact observations that need expansion for phase-type.

Returns true if any observations have obstype == 1 (exact), which require
splitting for proper phase-type forward-backward sampling.
"""
function needs_data_expansion_for_phasetype(data::DataFrame)
    return any(data.obstype .== 1)
end

"""
    compute_expanded_subject_indices(expanded_data::DataFrame)

Compute subject indices for expanded phase-type data.

Returns a vector of UnitRange{Int64} where each element gives the row indices
for one subject in the expanded data.

# Arguments
- `expanded_data::DataFrame`: Expanded data with id column

# Returns
- `Vector{UnitRange{Int64}}`: Subject indices for the expanded data
"""
function compute_expanded_subject_indices(expanded_data::DataFrame)
    # Group by id and get row ranges
    subject_ids = unique(expanded_data.id)
    n_subjects = length(subject_ids)
    
    subjectindices = Vector{UnitRange{Int64}}(undef, n_subjects)
    
    current_row = 1
    for (i, subj_id) in enumerate(subject_ids)
        # Find rows for this subject
        subj_mask = expanded_data.id .== subj_id
        subj_rows = findall(subj_mask)
        
        # Should be contiguous
        @assert subj_rows == subj_rows[1]:subj_rows[end] "Subject rows must be contiguous"
        
        subjectindices[i] = subj_rows[1]:subj_rows[end]
    end
    
    return subjectindices
end

"""
    build_phasetype_model(tmat::Matrix{Int64}, config::PhaseTypeConfig;
                          data::DataFrame,
                          covariate_formula::Union{Nothing, FormulaTerm} = nothing,
                          SubjectWeights::Union{Nothing, Vector{Float64}} = nothing,
                          CensoringPatterns::Union{Nothing, Matrix{<:Real}} = nothing,
                          verbose::Bool = false)

Build a multistate Markov model on the expanded phase-type state space.

This creates a standard MultistateMarkovModel that can be fitted using the
existing `fit()` infrastructure. The model represents phase-type sojourn time
distributions as a hidden Markov model on an expanded state space.

# Model Structure

For each transient observed state with p phases:
- p-1 **progression rates** (λ₁, ..., λ_{p-1}): Rate of moving through phases
- p × n_dest **exit rates** (μᵢⱼ): Rate of exiting from phase i to destination state j

The sojourn time in each observed state follows a Coxian phase-type distribution.

# Arguments
- `tmat::Matrix{Int64}`: Transition matrix for observed states (n_obs_states × n_obs_states)
- `config::PhaseTypeConfig`: Configuration specifying number of phases

# Keyword Arguments
- `data::DataFrame`: Panel data with columns id, tstart, tstop, statefrom, stateto, obstype
- `covariate_formula`: Optional formula for covariates (applied to ALL rates via proportional hazards)
- `SubjectWeights`: Optional per-subject weights
- `CensoringPatterns`: Optional censoring patterns matrix
- `verbose::Bool`: Print diagnostic information

# Returns
- `NamedTuple` with fields:
  - `model`: MultistateMarkovModel on expanded state space
  - `surrogate`: PhaseTypeSurrogate with state mappings
  - `tmat_expanded`: Expanded transition matrix
  - `tmat_original`: Original transition matrix

# Example
```julia
# Original 3-state illness-death model
tmat = [0 1 1; 0 0 1; 0 0 0]

# Expand with 3 phases per transient state
config = PhaseTypeConfig(n_phases=3)

# Build and fit
result = build_phasetype_model(tmat, config; data=panel_data)
fitted = fit(result.model)
```

# Parameter Interpretation

After fitting, parameters represent:
- Progression rates: How quickly subjects move through phases (controls sojourn time shape)
- Exit rates: Competing hazards for leaving each phase to different destinations

The effective sojourn time distribution in state s is a Coxian phase-type with:
- λᵢ = sum of progression + exit rates from phase i (total exit rate)
- Transition probabilities determined by ratio of rates

# Covariates

When `covariate_formula` is provided, covariates multiply ALL rates via
exp(Xβ). This is a proportional hazards model on the expanded state space,
which translates to proportional hazards on the original state space.

See also: [`PhaseTypeConfig`](@ref), [`fit`](@ref), [`PhaseTypeSurrogate`](@ref)
"""
function build_phasetype_model(tmat::Matrix{Int64}, config::PhaseTypeConfig;
                               data::DataFrame,
                               covariate_formula::Union{Nothing, FormulaTerm} = nothing,
                               SubjectWeights::Union{Nothing, Vector{Float64}} = nothing,
                               CensoringPatterns::Union{Nothing, Matrix{<:Real}} = nothing,
                               verbose::Bool = false)
    
    # Step 1: Build the phase-type surrogate (state mappings, expanded Q structure)
    surrogate = build_phasetype_surrogate(tmat, config)
    
    if verbose
        println("Phase-type model structure:")
        println("  Observed states: $(surrogate.n_observed_states)")
        println("  Expanded states: $(surrogate.n_expanded_states)")
        for s in 1:surrogate.n_observed_states
            phases = surrogate.state_to_phases[s]
            println("  State $s → phases $(first(phases)):$(last(phases)) ($(length(phases)) phases)")
        end
    end
    
    # Step 2: Build expanded transition matrix
    tmat_expanded = build_expanded_tmat(tmat, surrogate)
    
    # Step 3: Generate hazard specifications for expanded model
    hazards = build_phasetype_hazards(tmat, config, surrogate;
                                       covariate_formula = covariate_formula)
    
    if verbose
        println("  Number of hazards: $(length(hazards))")
    end
    
    # Step 4: Prepare data - expand state references to phases
    # Make a copy to avoid modifying the original
    data_expanded = copy(data)
    expand_data_states!(data_expanded, surrogate)
    
    # Step 5: Prepare censoring patterns for expanded state space
    n_expanded = surrogate.n_expanded_states
    if CensoringPatterns !== nothing
        # Expand censoring patterns to phase space
        n_patterns = size(CensoringPatterns, 1)
        CensoringPatterns_expanded = zeros(Float64, n_patterns, n_expanded + 1)
        
        # First column is the censoring code (preserve)
        CensoringPatterns_expanded[:, 1] = CensoringPatterns[:, 1]
        
        # Map observed state probabilities to their phases
        for p in 1:n_patterns
            for s in 1:size(tmat, 1)
                state_prob = CensoringPatterns[p, s + 1]
                for phase in surrogate.state_to_phases[s]
                    CensoringPatterns_expanded[p, phase + 1] = state_prob
                end
            end
        end
    else
        CensoringPatterns_expanded = nothing
    end
    
    # Step 6: Build the emission matrix for expanded model
    emat_expanded = build_phasetype_emat(data, surrogate, 
                                          CensoringPatterns === nothing ? 
                                          Matrix{Float64}(undef, 0, size(tmat, 1)) : 
                                          Float64.(CensoringPatterns))
    
    # Step 7: Build the multistate model using existing infrastructure
    model = multistatemodel(hazards...; 
                           data = data_expanded,
                           SubjectWeights = SubjectWeights,
                           CensoringPatterns = CensoringPatterns_expanded,
                           EmissionMatrix = emat_expanded,
                           verbose = verbose)
    
    return (
        model = model,
        surrogate = surrogate,
        tmat_expanded = tmat_expanded,
        tmat_original = tmat
    )
end

"""
    phasetype_parameters_to_Q(fitted_model, surrogate::PhaseTypeSurrogate)

Extract fitted parameters and construct the intensity matrix Q.

After fitting a phase-type model, this converts the fitted hazard parameters
back to an intensity matrix on the expanded state space.

# Arguments
- `fitted_model`: Result from `fit(model)` where model was built by `build_phasetype_model`
- `surrogate::PhaseTypeSurrogate`: The surrogate used to build the model

# Returns
- `Matrix{Float64}`: Fitted intensity matrix Q on expanded state space
"""
function phasetype_parameters_to_Q(fitted_model, surrogate::PhaseTypeSurrogate)
    n_expanded = surrogate.n_expanded_states
    Q = zeros(Float64, n_expanded, n_expanded)
    
    # Get fitted parameters (on natural/positive scale)
    pars = fitted_model.parameters
    hazards = fitted_model.model.hazards
    
    # Fill in Q matrix from fitted hazard rates
    for (idx, haz) in enumerate(hazards)
        # Get intercept parameter (baseline rate on natural scale)
        # pars[idx] is a NamedTuple: (baseline = (param_name = ...,),)
        rate = exp(only(values(pars[idx].baseline)))  # Parameters stored on log scale
        Q[haz.statefrom, haz.stateto] = rate
    end
    
    # Set diagonal elements
    for i in 1:n_expanded
        Q[i, i] = -sum(Q[i, j] for j in 1:n_expanded if j != i)
    end
    
    return Q
end

# =============================================================================
# Phase 5: Phase-Type Model Initialization
# =============================================================================

"""
    set_crude_init!(model::PhaseTypeModel; constraints = nothing)

Initialize phase-type model parameters using crude transition rates.

For phase-type models, initialization is performed on the expanded (internal)
Markov state space. The crude rates are calculated from the observed data,
then mapped to the expanded space.

# Algorithm

For built-in structures (`:unstructured`, `:allequal`, `:prop_to_prog`):
1. Calculate crude rates on the observed state space  
2. For each phase-type hazard with structure:
   - `:unstructured`: Progression (λ) and exit (μ) rates set independently
   - `:allequal` or `:prop_to_prog`: All λ and μ set to crude_rate/(2n-1)
     where n is the number of phases (assumes equal probability of 
     progression vs absorption at each phase)

Custom constraints are not supported - use `set_parameters!` instead.

The goal is to achieve approximate mean sojourn times that match the data.
"""
function set_crude_init!(model::PhaseTypeModel; constraints = nothing)
    if !isnothing(constraints)
        error("Cannot initialize parameters to crude estimates when there are parameter constraints. " *
              "For custom constraints, set parameters manually using set_parameters!().")
    end
    
    # Calculate crude rates on observed space
    crude_par = _calculate_crude_phasetype(model)
    
    # Build a lookup for structure by original hazard
    # Maps (statefrom, stateto) -> structure for phase-type hazards
    structure_lookup = _build_phasetype_structure_lookup(model.mappings)
    
    # Build new parameter vectors for expanded hazards
    new_expanded_params = Vector{Vector{Float64}}(undef, length(model.hazards))
    
    for (idx, h) in enumerate(model.hazards)
        # Determine crude rate for this expanded hazard, respecting structure
        log_rate = _get_crude_rate_for_expanded_hazard_structured(
            h, crude_par, model.mappings, structure_lookup
        )
        # Get initialized parameters (handles covariates too)
        set_par_to = init_par(h, log_rate)
        new_expanded_params[idx] = set_par_to
    end
    
    # Rebuild expanded_parameters
    model.parameters = _build_expanded_parameters(
        new_expanded_params, model.hazkeys, model.hazards
    )
    
    # Sync to user-facing parameters
    _sync_phasetype_parameters_to_original!(model)
end

"""
    _build_phasetype_structure_lookup(mappings::PhaseTypeMappings)

Build a lookup table mapping (statefrom, stateto) to Coxian structure.

Returns a Dict where keys are (observed_from, observed_to) tuples and values
are the structure symbol (`:unstructured`, `:allequal`, `:prop_to_prog`).
"""
function _build_phasetype_structure_lookup(mappings::PhaseTypeMappings)
    lookup = Dict{Tuple{Int,Int}, Symbol}()
    
    for h in mappings.original_hazards
        if h isa PhaseTypeHazardSpec
            lookup[(h.statefrom, h.stateto)] = h.structure
        end
    end
    
    return lookup
end

"""
    _get_crude_rate_for_expanded_hazard_structured(h, crude_par, mappings, structure_lookup)

Determine the appropriate crude rate for an expanded hazard, respecting structure.

For all built-in structures (`:unstructured`, `:allequal`, `:prop_to_prog`):
- All λ and μ are set to crude_rate / (2n-1) where n = number of phases
- This assumes equal probability of progression to next phase vs absorption
"""
function _get_crude_rate_for_expanded_hazard_structured(
    h::_Hazard, 
    crude_par::Matrix{Float64}, 
    mappings::PhaseTypeMappings,
    structure_lookup::Dict{Tuple{Int,Int}, Symbol}
)
    hazname_str = String(h.hazname)
    
    # Determine which original transition this expanded hazard belongs to
    origin_state, dest_state, structure = _identify_original_transition(
        hazname_str, mappings, structure_lookup
    )
    
    # For all built-in structures, use uniform initialization
    # This gives crude_rate / (2n-1) for all λs and μs
    if structure in (:unstructured, :allequal, :prop_to_prog)
        n_phases = mappings.n_phases_per_state[origin_state]
        n_params = 2 * n_phases - 1  # λ₁...λₙ₋₁, μ₁...μₙ
        
        # Total outgoing rate from observed state
        total_rate = -crude_par[origin_state, origin_state]
        
        # All rates equal to crude_rate / (2n-1)
        uniform_rate = total_rate / n_params
        return log(max(uniform_rate, 0.01))
    end
    
    # Unknown structure - error
    error("Unknown Coxian structure: $structure. " *
          "Supported structures are :unstructured, :allequal, :prop_to_prog")
end

"""
    _identify_original_transition(hazname_str, mappings, structure_lookup)

Identify which original (observed) transition an expanded hazard belongs to.

Returns (origin_state, dest_state, structure) tuple.
"""
function _identify_original_transition(
    hazname_str::String,
    mappings::PhaseTypeMappings,
    structure_lookup::Dict{Tuple{Int,Int}, Symbol}
)
    # Pattern for exit hazards: hXY_exitZ (X=from, Y=to, Z=phase)
    m_exit = match(r"h(\d+)(\d+)_exit", hazname_str)
    if !isnothing(m_exit)
        origin = parse(Int, m_exit.captures[1])
        dest = parse(Int, m_exit.captures[2])
        structure = get(structure_lookup, (origin, dest), :unstructured)
        return (origin, dest, structure)
    end
    
    # Pattern for progression hazards: hX_progY (X=state, Y=phase)
    m_prog = match(r"h(\d+)_prog", hazname_str)
    if !isnothing(m_prog)
        origin = parse(Int, m_prog.captures[1])
        # Progression hazards belong to all destinations from this origin
        # Find the structure - should be same for all destinations with PT
        for (key, struc) in structure_lookup
            from, to = key
            if from == origin
                return (origin, to, struc)
            end
        end
        # If no PT hazard found (shouldn't happen), use unstructured
        return (origin, 0, :unstructured)
    end
    
    # Regular hazard (non-PT transition)
    m_regular = match(r"h(\d+)(\d+)", hazname_str)
    if !isnothing(m_regular)
        origin = parse(Int, m_regular.captures[1])
        dest = parse(Int, m_regular.captures[2])
        return (origin, dest, :unstructured)
    end
    
    error("Could not parse hazard name: $hazname_str")
end

"""
    _sync_phasetype_parameters_to_original!(model::PhaseTypeModel)

Synchronize expanded parameters back to the user-facing original parameter structure.

This collapses the expanded hazard parameters (λ₁...λₙ₋₁, μ₁...μₙ) into the 
phase-type parameterization expected by the user.
"""
function _sync_phasetype_parameters_to_original!(model::PhaseTypeModel)
    mappings = model.mappings
    original_hazards = mappings.original_hazards
    
    # Build new parameter vectors for original hazards
    params_list = Vector{Vector{Float64}}()
    hazkeys = Dict{Symbol, Int64}()
    npar_baseline_list = Int[]
    
    for (orig_idx, h) in enumerate(original_hazards)
        hazname = Symbol("h$(h.statefrom)$(h.stateto)")
        hazkeys[hazname] = orig_idx
        
        if h isa PhaseTypeHazardSpec
            # PT hazard: collect λ and μ parameters
            params = _collect_phasetype_params(model, h, mappings)
            push!(params_list, params)
            push!(npar_baseline_list, 2 * h.n_phases - 1)
        else
            # Regular hazard: find in expanded hazards
            exp_idx = model.hazkeys[hazname]
            params = extract_params_vector(model.parameters.nested[hazname])
            push!(params_list, params)
            push!(npar_baseline_list, model.hazards[exp_idx].npar_baseline)
        end
    end
    
    # Rebuild original parameters structure
    # For each hazard, construct parnames vector by extracting from existing parameters
    params_nested_pairs = [
        let 
            hazname_sym = first(pair)
            idx = last(pair)
            # Get parameter names from existing nested structure if available, 
            # otherwise construct from hazname and npar counts
            if haskey(model.parameters.nested, hazname_sym)
                existing_params = model.parameters.nested[hazname_sym]
                parnames = vcat(
                    collect(keys(existing_params.baseline)),
                    haskey(existing_params, :covariates) ? collect(keys(existing_params.covariates)) : Symbol[]
                )
            else
                # Fallback: construct generic names (shouldn't happen in practice)
                npar_baseline = npar_baseline_list[idx]
                npar_total = length(params_list[idx])
                parnames = [Symbol(hazname_sym, "_p", i) for i in 1:npar_total]
            end
            hazname_sym => build_hazard_params(params_list[idx], parnames, npar_baseline_list[idx], length(params_list[idx]))
        end
        for pair in sort(collect(hazkeys), by = x -> x[2])
    ]
    params_nested = NamedTuple(params_nested_pairs)
    reconstructor = ReConstructor(params_nested, unflattentype=UnflattenFlexible())
    params_flat = flatten(reconstructor, params_nested)
    
    # Get natural scale parameters (family-aware transformation)
    params_natural_pairs = [
        hazname => extract_natural_vector(params_nested[hazname], original_hazards[idx].family)
        for (hazname, idx) in sort(collect(hazkeys), by = x -> x[2])
    ]
    params_natural = NamedTuple(params_natural_pairs)
    
    # Update original_parameters (user-facing), not parameters (internal expanded)
    model.original_parameters = (
        flat = params_flat,
        nested = params_nested,
        natural = params_natural,
        reconstructor = reconstructor
    )
end

"""
    _collect_phasetype_params(model, h, mappings)

Collect λ and μ parameters from expanded hazards for a :pt hazard specification.

Returns parameters ordered as: [λ₁, λ₂, ..., λₙ₋₁, μ₁, μ₂, ..., μₙ, covariates...]
"""
function _collect_phasetype_params(model::PhaseTypeModel, 
                                    h::PhaseTypeHazardSpec,
                                    mappings::PhaseTypeMappings)
    n = h.n_phases
    params = Float64[]
    covariate_params = Float64[]
    
    # Find expanded hazard indices for this transition
    orig_from = h.statefrom
    orig_to = h.stateto
    
    # Get phase range for origin state
    phase_range = mappings.state_to_phases[orig_from]
    
    # Collect progression rates λ₁...λₙ₋₁ (internal phase transitions)
    for phase_idx in 1:(n-1)
        prog_name = Symbol("h$(orig_from)_prog$(phase_idx)")
        if haskey(model.hazkeys, prog_name)
            exp_idx = model.hazkeys[prog_name]
            exp_params = model.parameters.nested[prog_name]
            # First baseline param is the rate
            push!(params, exp_params.baseline[1])
        else
            # Should not happen if model built correctly
            push!(params, 0.0)
        end
    end
    
    # Collect exit rates μ₁...μₙ 
    for phase_idx in 1:n
        exit_name = Symbol("h$(orig_from)$(orig_to)_exit$(phase_idx)")
        if haskey(model.hazkeys, exit_name)
            exp_idx = model.hazkeys[exit_name]
            exp_params = model.parameters.nested[exit_name]
            push!(params, exp_params.baseline[1])
            
            # Collect covariates from first exit hazard (all should share same structure)
            if phase_idx == 1 && haskey(exp_params, :covariates)
                covariate_params = collect(exp_params.covariates)
            end
        else
            push!(params, 0.0)
        end
    end
    
    return vcat(params, covariate_params)
end

"""
    _calculate_crude_phasetype(model::PhaseTypeModel)

Calculate crude transition rates for a phase-type model on the observed state space.

Uses the original (observed) data and transition matrix to compute rates,
which are then used to initialize the expanded hazards.
"""
function _calculate_crude_phasetype(model::PhaseTypeModel)
    # Use original_data (observed data) and original_tmat
    n_rs, T_r = compute_suff_stats(model.original_data, model.original_tmat, model.SubjectWeights)
    
    # Avoid log of zero
    n_rs = max.(n_rs, 0.5)
    T_r[T_r .== 0] .= mean(model.original_data.tstop .- model.original_data.tstart)
    
    crude_mat = n_rs ./ T_r
    crude_mat[findall(model.original_tmat .== 0)] .= 0
    
    for i in 1:length(T_r)
        crude_mat[i, i] = -sum(crude_mat[i, Not(i)])
    end
    
    return crude_mat
end

# =============================================================================
# Phase-Type Initialization
# =============================================================================

"""
    _make_collapsed_markov_model(model::PhaseTypeModel)

Create a simple Markov model on the original (collapsed) state space.

This is used to generate paths for initializing phase-type models. The collapsed
model uses exponential hazards for each transition in the original state space.

# Returns
- `MultistateMarkovModel` or `MultistateMarkovModelCensored` on original states
"""
function _make_collapsed_markov_model(model::PhaseTypeModel)
    # Build exponential hazards for each transition in original_tmat
    collapsed_hazards = HazardFunction[]
    
    for hazspec in model.hazards_spec
        # Create exponential version of this hazard (same formula for covariates)
        exp_haz = Hazard(hazspec.hazard, :exp, hazspec.statefrom, hazspec.stateto)
        push!(collapsed_hazards, exp_haz)
    end
    
    # Create Markov model on original data
    collapsed_model = multistatemodel(collapsed_hazards...; 
                                       data = model.original_data,
                                       SubjectWeights = model.SubjectWeights,
                                       CensoringPatterns = nothing)
    
    return collapsed_model
end

"""
    _transfer_phasetype_parameters!(target_model::PhaseTypeModel, source_fitted)

Transfer parameters from a fitted phase-type model to another phase-type model.

# Arguments
- `target_model::PhaseTypeModel`: Model to receive parameters
- `source_fitted`: Fitted PhaseTypeModel to copy from
"""
function _transfer_phasetype_parameters!(target_model::PhaseTypeModel, source_fitted)
    # Get flat parameters from source (on expanded space)
    source_params = get_parameters_flat(source_fitted)
    
    # Copy to target
    copyto!(target_model.parameters.flat, source_params)
end

"""
    _init_phasetype_from_surrogate_paths!(model::PhaseTypeModel, npaths; constraints)

Initialize phase-type model by fitting collapsed Markov, simulating paths, and fitting to exact data.

# Algorithm
1. Create and fit a collapsed Markov model on original state space
2. Simulate paths from the fitted Markov model
3. Convert paths to exact-observation data with interpolated covariates
4. Create phase-type model with exact data
5. Initialize with crude rates and fit
6. Transfer parameters to original model

# Arguments
- `model::PhaseTypeModel`: Model to initialize
- `npaths::Int`: Number of paths per subject (used to determine simulation count)
- `constraints`: Parameter constraints for exact-data fitting
"""
function _init_phasetype_from_surrogate_paths!(model::PhaseTypeModel,
                                                npaths::Int;
                                                constraints = nothing)
    # Step 1: Create collapsed Markov model on original state space
    collapsed_model = _make_collapsed_markov_model(model)
    
    # Step 2: Initialize and fit collapsed model
    set_crude_init!(collapsed_model)
    collapsed_fitted = fit(collapsed_model; compute_vcov = false, 
                           compute_ij_vcov = false, verbose = false)
    
    # Step 3: Simulate paths from fitted collapsed model
    # simulate returns Vector{Vector{SamplePath}} when paths=true
    sim_result = simulate(collapsed_fitted; nsim = 1, paths = true, data = false)
    
    # sim_result is Vector{Vector{SamplePath}}, one per simulation
    simulated_paths = sim_result[1]  # First (only) simulation
    
    # Step 4: Convert to exact data
    exact_data = paths_to_dataset(simulated_paths)
    
    # Check if we have any transitions
    if nrow(exact_data) == 0
        @warn "No transitions in simulated paths; using crude initialization"
        set_crude_init!(model; constraints = constraints)
        return
    end
    
    # Step 5: Interpolate covariates from original data
    _interpolate_covariates!(exact_data, model.original_data)
    
    # Step 6: Create phase-type model with exact data
    exact_pt_model = multistatemodel(model.hazards_spec...; 
                                      data = exact_data,
                                      SubjectWeights = nothing)
    
    # Step 7: Initialize with crude rates (phase-type is Markov on expanded space)
    set_crude_init!(exact_pt_model)
    
    # Step 8: Fit phase-type model to exact data
    exact_fitted = fit(exact_pt_model; constraints = constraints,
                       compute_vcov = false, compute_ij_vcov = false,
                       verbose = false)
    
    # Step 9: Transfer parameters to original model
    _transfer_phasetype_parameters!(model, exact_fitted)
end

"""
    initialize_parameters!(model::PhaseTypeModel; method=:auto, npaths=10, ...)

Initialize phase-type model parameters using the specified method.

# Arguments
- `model::PhaseTypeModel`: The model to initialize (modified in place)
- `constraints`: Parameter constraints for fitting
- `surrogate_constraints`: Ignored for phase-type models
- `surrogate_parameters`: Ignored for phase-type models
- `method::Symbol = :auto`: Initialization method
  - `:auto` or `:surrogate` - Fit collapsed Markov, simulate paths, fit PT to exact data
  - `:crude` - Use crude rates on expanded space
- `npaths::Int = 10`: Number of paths per subject for :surrogate method

# Examples
```julia
# Auto-select method (uses :surrogate for PhaseType)
initialize_parameters!(model)

# Force crude initialization
initialize_parameters!(model; method = :crude)
```

See also: [`initialize_parameters`](@ref), [`set_crude_init!`](@ref)
"""
function initialize_parameters!(model::PhaseTypeModel; 
                                constraints = nothing, 
                                surrogate_constraints = nothing, 
                                surrogate_parameters = nothing, 
                                method::Symbol = :auto,
                                npaths::Int = 10)
    # Validate npaths
    npaths > 0 || throw(ArgumentError("npaths must be positive, got $npaths"))
    
    # Resolve :auto → :surrogate for PhaseType
    actual_method = method == :auto ? :surrogate : method
    
    # Validate method
    actual_method in (:crude, :surrogate) ||
        throw(ArgumentError("PhaseTypeModel supports :crude or :surrogate, got :$method"))
    
    if actual_method == :crude
        set_crude_init!(model; constraints = constraints)
    else  # :surrogate
        _init_phasetype_from_surrogate_paths!(model, npaths; constraints = constraints)
    end
    
    return nothing
end

"""
    initialize_parameters(model::PhaseTypeModel; method=:auto, npaths=10, ...) -> PhaseTypeModel

Return a new phase-type model with initialized parameters (non-mutating version).

See [`initialize_parameters!`](@ref) for argument descriptions.
"""
function initialize_parameters(model::PhaseTypeModel; 
                               constraints = nothing, 
                               surrogate_constraints = nothing, 
                               surrogate_parameters = nothing, 
                               method::Symbol = :auto,
                               npaths::Int = 10)
    model_copy = deepcopy(model)
    initialize_parameters!(model_copy; 
                           constraints = constraints,
                           surrogate_constraints = surrogate_constraints,
                           surrogate_parameters = surrogate_parameters,
                           method = method,
                           npaths = npaths)
    return model_copy
end

# =============================================================================
# Phase 6: Parameter Management for PhaseTypeModel
# =============================================================================

"""
    get_parameters(model::PhaseTypeModel; scale::Symbol = :natural)

Get parameters from a phase-type model.

# Arguments
- `model::PhaseTypeModel`: The phase-type model
- `scale::Symbol = :natural`: Parameter scale
  - `:natural` - Human-readable scale (rates as positive values)
  - `:estimation` or `:log` or `:flat` - Flat vector on log scale (for optimization)
  - `:nested` - Nested NamedTuple with ParameterHandling structure

# Returns
For a phase-type hazard h12 with n phases, the parameters are organized as:
- λ₁, λ₂, ..., λₙ₋₁ (progression rates)
- μ₁, μ₂, ..., μₙ (exit rates)
- covariates (if any)

# Example
```julia
h12 = Hazard(:pt, 1, 2; n_phases=3)
model = multistatemodel(h12; data=data)
initialize_parameters!(model)

# Get natural-scale parameters
params = get_parameters(model)
# (h12 = [λ₁, λ₂, μ₁, μ₂, μ₃], ...)

# Get flat vector for optimization
params_flat = get_parameters(model; scale=:estimation)
```
"""
function get_parameters(model::PhaseTypeModel; scale::Symbol = :natural)
    # Return user-facing parameters (original parameterization)
    if scale == :natural
        return model.original_parameters.natural
    elseif scale == :estimation || scale == :log || scale == :flat
        return model.original_parameters.flat
    elseif scale == :nested
        return model.original_parameters.nested
    else
        throw(ArgumentError("scale must be :natural, :estimation, :log, :flat, or :nested (got :$scale)"))
    end
end

"""
    get_expanded_parameters(model::PhaseTypeModel; scale::Symbol = :natural)

Get parameters on the expanded (internal) state space.

This returns parameters for the expanded hazards used internally for likelihood
computation and simulation. Most users should use `get_parameters` instead.

# Arguments
- `model::PhaseTypeModel`: The phase-type model
- `scale::Symbol = :natural`: Parameter scale (same options as `get_parameters`)

# Example
```julia
# Get expanded hazard parameters
exp_params = get_expanded_parameters(model)
# (h1_prog1 = [λ₁], h1_prog2 = [λ₂], h12_exit1 = [μ₁], ...)
```
"""
function get_expanded_parameters(model::PhaseTypeModel; scale::Symbol = :natural)
    if scale == :natural
        return model.parameters.natural
    elseif scale == :estimation || scale == :log || scale == :flat
        return model.parameters.flat
    elseif scale == :nested
        return model.parameters.nested
    else
        throw(ArgumentError("scale must be :natural, :estimation, :log, :flat, or :nested (got :$scale)"))
    end
end

# -----------------------------------------------------------------------------
# Standard accessor functions for compatibility with fitting infrastructure
# These operate on the EXPANDED parameters (internal representation)
# -----------------------------------------------------------------------------

"""
    get_parameters_flat(model::PhaseTypeModel)

Get expanded model parameters as a flat vector for optimization.

For phase-type models, fitting operates on the expanded (internal) state space,
so this returns the flattened expanded parameters.

# Returns
- `Vector{Float64}`: Flat parameter vector on log scale

# Note
This is used internally by the fitting machinery. For user-facing parameter
access, use `get_parameters(model)` or `get_expanded_parameters(model)`.
"""
function get_parameters_flat(model::PhaseTypeModel)
    return model.parameters.flat
end

"""
    get_parameters_nested(model::PhaseTypeModel)

Get expanded model parameters as a nested NamedTuple.

Returns the expanded hazard parameters organized by expanded hazard name.
"""
function get_parameters_nested(model::PhaseTypeModel)
    return model.parameters.nested
end

"""
    get_parameters_natural(model::PhaseTypeModel)

Get expanded model parameters on the natural (human-readable) scale.

Returns rates as positive values rather than log-transformed.
"""
function get_parameters_natural(model::PhaseTypeModel)
    return model.parameters.natural
end

"""
    get_unflatten_fn(model::PhaseTypeModel)

Get the function that converts flat parameters to nested NamedTuple.

This is the inverse of flatten() and is used by the fitting machinery
to convert optimizer output back to structured parameters.
"""
function get_unflatten_fn(model::PhaseTypeModel)
    return p -> unflatten(model.parameters.reconstructor, p)
end

"""
    set_parameters!(model::PhaseTypeModel, newvalues::Vector{Vector{Float64}})

Set phase-type model parameters from nested vectors.

Parameters should be organized as one vector per original hazard, with
phase-type hazards having [λ₁...λₙ₋₁, μ₁...μₙ, covariates...].

# Arguments
- `model::PhaseTypeModel`: The model to update
- `newvalues`: Nested vector with parameters for each original hazard

# Note
This updates both the user-facing `parameters` and the internal `expanded_parameters`.
"""
function set_parameters!(model::PhaseTypeModel, newvalues::Vector{Vector{Float64}})
    mappings = model.mappings
    original_hazards = mappings.original_hazards
    n_hazards = length(original_hazards)
    
    if length(newvalues) != n_hazards
        error("Expected $n_hazards parameter vectors (one per original hazard), got $(length(newvalues))")
    end
    
    # Build expanded parameter vectors from original values
    new_expanded_params = _expand_phasetype_params(newvalues, model)
    
    # Rebuild expanded_parameters
    model.parameters = _build_expanded_parameters(
        new_expanded_params, model.hazkeys, model.hazards
    )
    
    # Rebuild user-facing parameters from the new values
    _rebuild_original_params_from_values!(model, newvalues)
    
    return nothing
end

"""
    set_parameters!(model::PhaseTypeModel, newvalues::NamedTuple)

Set phase-type model parameters from a NamedTuple.

Keys should match original hazard names (e.g., :h12, :h23).
"""
function set_parameters!(model::PhaseTypeModel, newvalues::NamedTuple)
    mappings = model.mappings
    original_hazards = mappings.original_hazards
    
    # Build vector of vectors from NamedTuple
    param_vectors = Vector{Vector{Float64}}(undef, length(original_hazards))
    
    for (idx, h) in enumerate(original_hazards)
        hazname = Symbol("h$(h.statefrom)$(h.stateto)")
        if haskey(newvalues, hazname)
            param_vectors[idx] = collect(newvalues[hazname])
        else
            # Keep existing values
            param_vectors[idx] = collect(model.parameters.natural[hazname])
        end
    end
    
    set_parameters!(model, param_vectors)
    return nothing
end

"""
    _expand_phasetype_params(original_params, model)

Expand original hazard parameters to expanded hazard parameters.

For a :pt hazard with parameters [λ₁...λₙ₋₁, μ₁...μₙ, covariates...],
creates separate parameter vectors for each progression and exit hazard.
"""
function _expand_phasetype_params(original_params::Vector{Vector{Float64}}, 
                                   model::PhaseTypeModel)
    mappings = model.mappings
    n_expanded = length(model.hazards)
    expanded_params = Vector{Vector{Float64}}(undef, n_expanded)
    
    # Track which expanded hazards have been filled
    filled = falses(n_expanded)
    
    for (orig_idx, h) in enumerate(mappings.original_hazards)
        params = original_params[orig_idx]
        orig_name = Symbol("h$(h.statefrom)$(h.stateto)")
        
        if h isa PhaseTypeHazardSpec
            n = h.n_phases
            npar_baseline = 2 * n - 1
            
            # Extract λ and μ values
            λ_vals = params[1:(n-1)]
            μ_vals = params[n:(2n-1)]
            covar_vals = length(params) > npar_baseline ? params[(npar_baseline+1):end] : Float64[]
            
            # Assign to progression hazards
            for phase_idx in 1:(n-1)
                prog_name = Symbol("h$(h.statefrom)_prog$(phase_idx)")
                if haskey(model.hazkeys, prog_name)
                    exp_idx = model.hazkeys[prog_name]
                    expanded_params[exp_idx] = [λ_vals[phase_idx]]
                    filled[exp_idx] = true
                end
            end
            
            # Assign to exit hazards
            for phase_idx in 1:n
                exit_name = Symbol("h$(h.statefrom)$(h.stateto)_exit$(phase_idx)")
                if haskey(model.hazkeys, exit_name)
                    exp_idx = model.hazkeys[exit_name]
                    expanded_params[exp_idx] = vcat([μ_vals[phase_idx]], covar_vals)
                    filled[exp_idx] = true
                end
            end
        else
            # Regular hazard: direct assignment
            if haskey(model.hazkeys, orig_name)
                exp_idx = model.hazkeys[orig_name]
                expanded_params[exp_idx] = params
                filled[exp_idx] = true
            end
        end
    end
    
    # Check all hazards were filled
    if !all(filled)
        unfilled = findall(.!filled)
        unfilled_names = [model.hazards[i].hazname for i in unfilled]
        error("Failed to assign parameters to expanded hazards: $unfilled_names")
    end
    
    return expanded_params
end

"""
    _rebuild_original_params_from_values!(model, newvalues)

Rebuild the user-facing parameters structure from new values.
"""
function _rebuild_original_params_from_values!(model::PhaseTypeModel, 
                                                newvalues::Vector{Vector{Float64}})
    mappings = model.mappings
    original_hazards = mappings.original_hazards
    
    hazkeys = Dict{Symbol, Int64}()
    npar_baseline_list = Int[]
    
    for (idx, h) in enumerate(original_hazards)
        hazname = Symbol("h$(h.statefrom)$(h.stateto)")
        hazkeys[hazname] = idx
        
        if h isa PhaseTypeHazardSpec
            push!(npar_baseline_list, 2 * h.n_phases - 1)
        else
            push!(npar_baseline_list, model.hazards[model.hazkeys[hazname]].npar_baseline)
        end
    end
    
    # Rebuild parameters structure
    # For each hazard, construct parnames vector
    params_nested_pairs = [
        let
            hazname_sym = first(pair)
            idx = last(pair)
            # Get parameter names from existing structure
            # Try original_parameters first, then fall back to expanded model parameters
            parnames = Symbol[]
            found_parnames = false
            
            if haskey(model.original_parameters.nested, hazname_sym)
                existing_params = model.original_parameters.nested[hazname_sym]
                # Check if baseline is a NamedTuple (has Symbol keys) or Vector (has Int keys)
                if existing_params.baseline isa NamedTuple
                    parnames = vcat(
                        collect(keys(existing_params.baseline)),
                        haskey(existing_params, :covariates) && existing_params.covariates isa NamedTuple ? 
                            collect(keys(existing_params.covariates)) : Symbol[]
                    )
                    found_parnames = true
                end
            end
            
            # If original_parameters had placeholder structure, try expanded model parameters
            if !found_parnames && haskey(model.parameters.nested, hazname_sym)
                existing_params = model.parameters.nested[hazname_sym]
                if existing_params.baseline isa NamedTuple
                    parnames = vcat(
                        collect(keys(existing_params.baseline)),
                        haskey(existing_params, :covariates) && existing_params.covariates isa NamedTuple ? 
                            collect(keys(existing_params.covariates)) : Symbol[]
                    )
                    found_parnames = true
                end
            end
            
            # Fallback: construct from hazname
            if !found_parnames
                npar_total = length(newvalues[idx])
                parnames = [Symbol(hazname_sym, "_p", i) for i in 1:npar_total]
            end
            
            hazname_sym => build_hazard_params(newvalues[idx], parnames, npar_baseline_list[idx], length(newvalues[idx]))
        end
        for pair in sort(collect(hazkeys), by = x -> x[2])
    ]
    params_nested = NamedTuple(params_nested_pairs)
    reconstructor = ReConstructor(params_nested, unflattentype=UnflattenFlexible())
    params_flat = flatten(reconstructor, params_nested)
    
    # Get natural scale parameters (family-aware transformation)
    params_natural_pairs = [
        hazname => extract_natural_vector(params_nested[hazname], original_hazards[idx].family)
        for (hazname, idx) in sort(collect(hazkeys), by = x -> x[2])
    ]
    params_natural = NamedTuple(params_natural_pairs)
    
    # Update original_parameters (user-facing), not parameters (internal expanded)
    model.original_parameters = (
        flat = params_flat,
        nested = params_nested,
        natural = params_natural,
        reconstructor = reconstructor
    )
end
