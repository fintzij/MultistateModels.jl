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
            # Handle both String and Symbol representations
            if family != "exp" && family != :exp
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
                                      emat_ph::Matrix{Float64})

Compute the marginal log-likelihood of the observed data under the phase-type surrogate.

This is the normalization constant r(Y|θ') needed for importance sampling:
  log f̂(Y|θ) = log r(Y|θ') + Σᵢ log(mean(νᵢ))

where νᵢ = f(Zᵢ|θ) / h(Zᵢ|θ') are the importance weights.

# Arguments
- `model::MultistateProcess`: The multistate model with observed data
- `surrogate::PhaseTypeSurrogate`: The phase-type surrogate
- `emat_ph::Matrix{Float64}`: Emission matrix mapping expanded states to observations

# Returns
- `Float64`: Log marginal likelihood Σᵢ wᵢ * log P(Yᵢ|θ') under phase-type surrogate

# Mathematical Details
For each subject i, the marginal likelihood P(Yᵢ|θ') is computed via the
forward algorithm on the expanded state space, marginalizing over all possible
phase sequences consistent with the observations.
"""
function compute_phasetype_marginal_loglik(model::MultistateProcess, 
                                           surrogate::PhaseTypeSurrogate,
                                           emat_ph::Matrix{Float64})
    
    Q = surrogate.expanded_Q
    n_expanded = surrogate.n_expanded_states
    
    # Allocate memory for matrix exponential
    cache = ExponentialUtilities.alloc_mem(similar(Q), ExpMethodGeneric())
    
    ll_total = 0.0
    
    for subj_idx in eachindex(model.subjectindices)
        # Get subject data indices
        subj_inds = model.subjectindices[subj_idx]
        n_obs = length(subj_inds)
        
        # Extract times (include initial time)
        times = vcat(model.data.tstart[subj_inds[1]], model.data.tstop[subj_inds])
        
        # Extract state info
        statefrom = model.data.statefrom[subj_inds]
        stateto = model.data.stateto[subj_inds]
        obstype = model.data.obstype[subj_inds]
        
        # Get subject-specific emission matrix rows
        subj_emat = emat_ph[subj_inds, :]
        
        # Compute subject log-likelihood via forward algorithm
        subj_ll = loglik_phasetype_stable_internal(Q, subj_emat, times, statefrom, 
                                                    stateto, obstype, surrogate, cache)
        
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
        rate = exp(pars[idx][1])  # Parameters stored on log scale
        Q[haz.statefrom, haz.stateto] = rate
    end
    
    # Set diagonal elements
    for i in 1:n_expanded
        Q[i, i] = -sum(Q[i, j] for j in 1:n_expanded if j != i)
    end
    
    return Q
end
