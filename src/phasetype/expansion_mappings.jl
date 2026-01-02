# =============================================================================
# Phase-Type State Space Expansion Mappings
# =============================================================================
#
# State space expansion for phase-type distribution fitting.
# Maps between observed states and expanded phase states.
#
# Contents:
# - build_phasetype_mappings: Create bidirectional state mappings
# - _build_expanded_tmat: Expand transition matrix for phase states
# - _build_expanded_hazard_indices: Track hazard indices in expanded space
# - _compute_default_n_phases: Determine phases per state from hazard types
#
# References:
# - Titman & Sharples (2010) Biometrics 66(3):742-752
# - Asmussen et al. (1996) Scand J Stat 23(4):419-441
#
# Related files:
# - expansion_hazards.jl: Hazard expansion for phase-type fitting
# - expansion_constraints.jl: SCTP constraint generation
# - expansion_model.jl: Model building for phase-type hazards
# - expansion_loglik.jl: Phase-type log-likelihood computation
# - expansion_ffbs_data.jl: Data expansion for forward-backward sampling
#
# =============================================================================

# =============================================================================
# Phase 3: State Space Expansion Functions
# =============================================================================

"""
    build_phasetype_mappings(hazards, tmat, n_phases_per_state) -> PhaseTypeMappings

Build bidirectional mappings between observed and expanded phase-type state spaces.

# Arguments
- `hazards`: User-specified hazard specifications
- `tmat::Matrix{Int}`: Original transition matrix
- `n_phases_per_state::Vector{Int}`: Phases per state

# Returns
`PhaseTypeMappings` with state mappings, expanded tmat, and hazard index tracking.

# Example
```julia
mappings = build_phasetype_mappings([Hazard(:pt, 1, 2)], tmat, [3, 1, 1])
```

See also: [`PhaseTypeMappings`](@ref)
"""
function build_phasetype_mappings(hazards::Vector{<:HazardFunction}, 
                                   tmat::Matrix{Int},
                                   n_phases_per_state::Vector{Int})
    n_observed = size(tmat, 1)
    
    # Validate n_phases_per_state
    @assert length(n_phases_per_state) == n_observed "n_phases_per_state length must match number of states"
    @assert all(n >= 1 for n in n_phases_per_state) "all n_phases must be >= 1"
    
    # Identify pt hazard indices
    pt_hazard_indices = Int[]
    for (i, h) in enumerate(hazards)
        if h isa PhaseTypeHazard
            push!(pt_hazard_indices, i)
        end
    end
    
    # Build state_to_phases mapping (reuse helper from surrogate.jl)
    state_to_phases, phase_to_state, n_expanded = _build_state_mappings(n_observed, n_phases_per_state)
    
    # Build expanded transition matrix
    expanded_tmat = _build_expanded_tmat(tmat, n_phases_per_state, state_to_phases, hazards)
    
    # Build expanded hazard indices mapping
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

Build expanded transition matrix for phase-type state space.

For state s with n phases: Phase 1 → Phase 2 → ... → Phase n (progression),
plus exit transitions from each phase to destination states.

# Returns
`Matrix{Int}` with hazard indices (progressions indexed first, then exits).
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
        has_pt = any(h -> h isa PhaseTypeHazard && h.statefrom == s, hazards)
        
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
                
                if h isa PhaseTypeHazard
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
        has_pt = any(h -> h isa PhaseTypeHazard && h.statefrom == s, hazards)
        
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
            
            if h isa PhaseTypeHazard
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

