# =============================================================================
# Phase-Type Core Types and Configuration
# =============================================================================
#
# This file defines the core types for phase-type distribution modeling:
# - ProposalConfig: MCEM proposal configuration  
# - PhaseTypeDistribution: PH distribution representation
# - PhaseTypeConfig: Configuration for phase-type surrogates
# - PhaseTypeMappings: State space mappings for phase-type hazard models
# - PhaseTypeSurrogate: Augmented surrogate for importance sampling
#
# =============================================================================

# =============================================================================
# ProposalConfig: Unified Configuration for MCEM Proposals
# =============================================================================

"""
    ProposalConfig

Configuration for MCEM path proposals.

# Fields
- `type::Symbol`: `:markov` (default) or `:phasetype`
- `n_phases::Union{Symbol,Int,Dict{Int,Int}}`: `:auto` (BIC), `:heuristic`, `Int`, or `Dict`
- `structure::Symbol`: `:unstructured` (default) or `:sctp`
- `max_phases::Int`: Max for BIC selection (default: 5)
- `optimize::Bool`: Optimize surrogate params (default: true)
- `parameters`: Manual override (default: nothing)
- `constraints`: Optimization constraints (default: nothing)

# Example
```julia
fit(model; proposal=ProposalConfig(type=:phasetype, n_phases=:auto))
fit(model; proposal=PhaseTypeProposal(n_phases=Dict(1 => 3), structure=:sctp))
```

See also: [`PhaseTypeConfig`](@ref), [`PhaseTypeSurrogate`](@ref)
"""
struct ProposalConfig
    type::Symbol
    n_phases::Union{Symbol, Int, Dict{Int,Int}}
    structure::Symbol
    max_phases::Int
    optimize::Bool
    parameters::Union{Nothing, Any}
    constraints::Union{Nothing, Any}
    
    function ProposalConfig(;
            type::Symbol = :markov,
            n_phases::Union{Symbol, Int, Dict{Int,Int}} = :auto,
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
        elseif n_phases isa Dict{Int,Int}
            all(v >= 1 for v in values(n_phases)) || 
                throw(ArgumentError("all n_phases values must be >= 1"))
            all(k >= 1 for k in keys(n_phases)) || 
                throw(ArgumentError("all n_phases keys (state indices) must be >= 1"))
        end
        
        # Validate structure
        structure in (:unstructured, :sctp) ||
            throw(ArgumentError("structure must be :unstructured or :sctp, got :$structure"))
        
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

# Example
```julia
fit(model; proposal=MarkovProposal(optimize=false, parameters=my_params))
```
"""
MarkovProposal(; kwargs...) = ProposalConfig(type=:markov; kwargs...)

"""
    PhaseTypeProposal(; n_phases=:auto, max_phases=5, kwargs...)

Convenience constructor for phase-type proposal configuration.

# Arguments
- `n_phases`: `:auto` (BIC), `:heuristic`, `Int`, or `Dict{Int,Int}`
- `max_phases`: Max for BIC selection (default: 5)
- `optimize`, `parameters`, `constraints`: See `ProposalConfig`

# Example
```julia
fit(model; proposal=PhaseTypeProposal(n_phases=Dict(1 => 3, 2 => 2)))
```
"""
PhaseTypeProposal(; n_phases::Union{Symbol, Int, Dict{Int,Int}} = :auto, kwargs...) = 
    ProposalConfig(type=:phasetype, n_phases=n_phases; kwargs...)

"""
    needs_phasetype_proposal(hazards::Vector) -> Bool

Check if any hazards are non-exponential (would benefit from phase-type proposals).
"""
function needs_phasetype_proposal(hazards::Vector)
    for h in hazards
        if hasproperty(h, :family)
            family = h.family
            if family != :exp
                return true
            end
        elseif !isa(h, _MarkovHazard)
            return true
        end
    end
    return false
end

"""
    resolve_proposal_config(proposal::Symbol, model) -> ProposalConfig

Resolve a Symbol proposal specification to a ProposalConfig.
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

# =============================================================================
# PhaseTypeDistribution
# =============================================================================

"""
    PhaseTypeDistribution

Phase-type distribution PH(π, Q) for sojourn time approximation.

# Fields
- `n_phases::Int`: Number of latent phases
- `Q::Matrix{Float64}`: (p+1)×(p+1) intensity matrix with absorbing state
- `initial::Vector{Float64}`: Initial distribution over phases (sums to 1)

Q has form [S s; 0 0] where S is p×p sub-intensity, s is absorption rates.
CDF: F(t) = 1 - π' exp(St) 𝟙

# Example
```julia
Q = [-2.0 1.5 0.5; 0.0 -1.0 1.0; 0.0 0.0 0.0]  # 2-phase Coxian
ph = PhaseTypeDistribution(2, Q, [1.0, 0.0])
```

See also: [`PhaseTypeConfig`](@ref), [`subintensity`](@ref), [`absorption_rates`](@ref)
"""
struct PhaseTypeDistribution
    n_phases::Int
    Q::Matrix{Float64}
    initial::Vector{Float64}
    
    function PhaseTypeDistribution(n_phases::Int, Q::Matrix{Float64}, 
                                   initial::Vector{Float64})
        size(Q, 1) == size(Q, 2) == n_phases + 1 || 
            throw(DimensionMismatch("Q must be $(n_phases+1) × $(n_phases+1)"))
        length(initial) == n_phases || 
            throw(DimensionMismatch("initial must have length $n_phases"))
        abs(sum(initial) - 1.0) < 1e-10 || 
            throw(ArgumentError("initial distribution must sum to 1"))
        all(initial .>= 0) || 
            throw(ArgumentError("initial distribution must be non-negative"))
        all(diag(Q)[1:n_phases] .<= 0) || 
            throw(ArgumentError("diagonal of Q (transient states) must be non-positive"))
        all(Q[end, :] .== 0) || 
            throw(ArgumentError("last row of Q (absorbing state) must be zeros"))
        
        new(n_phases, Q, initial)
    end
end

"""Extract the p×p sub-intensity matrix S from Q."""
subintensity(ph::PhaseTypeDistribution) = ph.Q[1:ph.n_phases, 1:ph.n_phases]

"""Extract absorption rates from Q (last column, excluding absorbing row)."""
absorption_rates(ph::PhaseTypeDistribution) = ph.Q[1:ph.n_phases, end]

"""Extract Coxian progression rates (off-diagonal: Q[i, i+1])."""
function progression_rates(ph::PhaseTypeDistribution)
    n = ph.n_phases
    n == 1 ? Float64[] : [ph.Q[i, i+1] for i in 1:n-1]
end

# =============================================================================
# PhaseTypeConfig
# =============================================================================

"""
    PhaseTypeConfig

Configuration for phase-type surrogate model.

# Fields
- `n_phases::Union{Symbol,Int,Dict{Int,Int}}`: `:auto`, `:heuristic`, `Int`, or `Dict{Int,Int}`
- `structure::Symbol`: `:unstructured` (default) or `:sctp`
- `constraints::Bool`: Apply Titman-Sharples constraints (default: true)
- `max_phases::Int`: Max for `:auto` BIC selection (default: 5)

For inference: specify `n_phases` explicitly. For MCEM surrogates: use `:auto` or `:heuristic`.

# Example
```julia
config = PhaseTypeConfig(n_phases=Dict(1 => 3, 2 => 2), structure=:sctp)
```
"""
struct PhaseTypeConfig
    n_phases::Union{Symbol, Int, Dict{Int,Int}}
    structure::Symbol
    constraints::Bool
    max_phases::Int
    
    function PhaseTypeConfig(; 
            n_phases::Union{Symbol, Int, Dict{Int,Int}} = 2,
            structure::Symbol = :unstructured,
            constraints::Bool = true,
            max_phases::Int = 5)
        
        if n_phases isa Symbol
            n_phases in (:auto, :heuristic) || 
                throw(ArgumentError("n_phases symbol must be :auto or :heuristic, got :$n_phases"))
        elseif n_phases isa Int
            n_phases >= 1 || throw(ArgumentError("n_phases must be >= 1"))
        elseif n_phases isa Dict{Int,Int}
            all(v >= 1 for v in values(n_phases)) || 
                throw(ArgumentError("all n_phases values must be >= 1"))
            all(k >= 1 for k in keys(n_phases)) || 
                throw(ArgumentError("all n_phases keys (state indices) must be >= 1"))
        end
        structure in (:unstructured, :sctp) ||
            throw(ArgumentError("structure must be :unstructured or :sctp, got :$structure"))
        max_phases >= 1 || throw(ArgumentError("max_phases must be >= 1"))
        
        new(n_phases, structure, constraints, max_phases)
    end
end

# =============================================================================
# PhaseTypeMappings
# =============================================================================

"""
    PhaseTypeMappings

Bidirectional mappings between observed and expanded state spaces for phase-type models.

# Fields
- `n_observed::Int`, `n_expanded::Int`: State space dimensions
- `n_phases_per_state::Vector{Int}`: Phases per observed state
- `state_to_phases::Vector{UnitRange{Int}}`: Observed state → phase indices
- `phase_to_state::Vector{Int}`: Phase index → observed state
- `expanded_tmat::Matrix{Int}`: Transition matrix on expanded space
- `original_tmat::Matrix{Int}`: Original transition matrix
- `original_hazards::Vector`: Original hazard specifications
- `pt_hazard_indices::Vector{Int}`: Indices of phase-type hazards
- `expanded_hazard_indices::Dict{Symbol,Vector{Int}}`: Original hazard name → expanded indices

See also: [`PhaseTypeHazardSpec`](@ref), [`PhaseTypeExpansion`](@ref)
"""
struct PhaseTypeMappings
    n_observed::Int
    n_expanded::Int
    n_phases_per_state::Vector{Int}
    state_to_phases::Vector{UnitRange{Int}}
    phase_to_state::Vector{Int}
    expanded_tmat::Matrix{Int}
    original_tmat::Matrix{Int}
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
# PhaseTypeSurrogate
# =============================================================================

"""
    PhaseTypeSurrogate

Phase-type augmented surrogate for importance sampling.

# Fields
- `phasetype_dists::Dict{Int,PhaseTypeDistribution}`: PH per observed state
- `n_observed_states::Int`, `n_expanded_states::Int`: State counts
- `state_to_phases::Vector{UnitRange{Int}}`, `phase_to_state::Vector{Int}`: Mappings
- `expanded_Q::Matrix{Float64}`: Expanded intensity matrix
- `config::PhaseTypeConfig`: Configuration

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
