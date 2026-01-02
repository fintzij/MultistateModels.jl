# =============================================================================
# Phase-Type SCTP Constraint Generation
# =============================================================================
#
# Generate SCTP (Same Conditional Transition Probabilities) constraints
# for phase-type models. These ensure that the probability of transitioning
# to each destination state is constant across phases.
#
# Implements: (h_{r_j,d1} - h_{r_1,d1}) - (h_{r_j,d2} - h_{r_1,d2}) = 0
#
# Contents:
# - _generate_sctp_constraints: Generate SCTP constraints for phase-type models
# - _merge_constraints: Merge user and auto-generated constraints
#
# =============================================================================

"""
    _generate_sctp_constraints(expanded_hazards, mappings, pt_states) -> NamedTuple

Generate SCTP constraints ensuring P(dest | leaving state) is constant across phases.

Implements: (h_{r_j,d1} - h_{r_1,d1}) - (h_{r_j,d2} - h_{r_1,d2}) = 0

# Arguments
- `expanded_hazards`: Expanded hazards from phase-type model
- `mappings::PhaseTypeMappings`: State-to-phase mappings
- `pt_states::Set{Int}`: States with phase-type hazards

# Returns
NamedTuple with `cons`, `lcons`, `ucons` fields, or `nothing` if no constraints needed.

See also: [`_build_phasetype_model_from_hazards`](@ref)
"""
function _generate_sctp_constraints(expanded_hazards::Vector{<:_Hazard},
                                    mappings::PhaseTypeMappings,
                                    pt_states::Set{Int})
    cons = Expr[]
    lcons = Float64[]
    ucons = Float64[]
    
    # For each phase-type state with multiple phases and multiple destinations
    for s in pt_states
        n_phases = mappings.n_phases_per_state[s]
        n_phases <= 1 && continue  # No constraints for single-phase states
        
        # Find all exit hazards from state s (h{s}{d}_{letter} patterns)
        # Group by destination
        exit_hazards_by_dest = Dict{Int, Vector{_Hazard}}()
        
        for h in expanded_hazards
            # Check if this is an exit hazard from state s
            # Exit hazards are named h{s}{d}_{letter} (e.g., h12_a, h12_b)
            hazname_str = string(h.hazname)
            # Match pattern: h{s}{d}_{letter}
            m = match(r"^h(\d+)(\d+)_([a-z])$", hazname_str)
            isnothing(m) && continue
            
            from_state = parse(Int, m.captures[1])
            to_state = parse(Int, m.captures[2])
            phase_letter = m.captures[3][1]  # Get the character
            
            from_state == s || continue
            
            if !haskey(exit_hazards_by_dest, to_state)
                exit_hazards_by_dest[to_state] = _Hazard[]
            end
            push!(exit_hazards_by_dest[to_state], h)
        end
        
        destinations = sort(collect(keys(exit_hazards_by_dest)))
        length(destinations) <= 1 && continue  # Need at least 2 destinations for constraints
        
        # Sort hazards by phase letter (a, b, c, ...)
        for d in destinations
            sort!(exit_hazards_by_dest[d], by = h -> begin
                m = match(r"_([a-z])$", string(h.hazname))
                m.captures[1][1]  # Return the letter character
            end)
        end
        
        # Use first destination as reference
        ref_dest = destinations[1]
        ref_hazards = exit_hazards_by_dest[ref_dest]
        
        # For each non-reference destination and each non-reference phase,
        # generate constraint: (h_{other}_p - h_{other}_1) - (h_{ref}_p - h_{ref}_1) = 0
        for other_dest in destinations[2:end]
            other_hazards = exit_hazards_by_dest[other_dest]
            
            # Skip if different number of phases (shouldn't happen, but be safe)
            length(other_hazards) == length(ref_hazards) || continue
            
            for p in 2:n_phases
                # Get baseline parameter names (first parameter = log rate)
                ref_phase1_par = ref_hazards[1].parnames[1]
                ref_phaseP_par = ref_hazards[p].parnames[1]
                other_phase1_par = other_hazards[1].parnames[1]
                other_phaseP_par = other_hazards[p].parnames[1]
                
                # Constraint: (other_p - other_1) - (ref_p - ref_1) = 0
                # Which is: other_p - other_1 - ref_p + ref_1 = 0
                constraint_expr = :( $other_phaseP_par - $other_phase1_par - $ref_phaseP_par + $ref_phase1_par )
                
                push!(cons, constraint_expr)
                push!(lcons, 0.0)
                push!(ucons, 0.0)
            end
        end
    end
    
    isempty(cons) && return nothing
    
    return (cons = cons, lcons = lcons, ucons = ucons)
end

"""
    _merge_constraints(user_constraints, auto_constraints) -> NamedTuple

Merge user-provided constraints with auto-generated constraints (e.g., SCTP).

User constraints are appended after auto-generated constraints.

# Arguments
- `user_constraints`: User-provided constraints (NamedTuple or nothing)
- `auto_constraints`: Auto-generated constraints (NamedTuple or nothing)

# Returns
- Combined constraints as NamedTuple{(:cons, :lcons, :ucons)}
- Returns `nothing` if both inputs are nothing
"""
function _merge_constraints(user_constraints, auto_constraints)
    if isnothing(user_constraints) && isnothing(auto_constraints)
        return nothing
    elseif isnothing(auto_constraints)
        return user_constraints
    elseif isnothing(user_constraints)
        return auto_constraints
    else
        # Merge both
        return (
            cons = vcat(auto_constraints.cons, user_constraints.cons),
            lcons = vcat(auto_constraints.lcons, user_constraints.lcons),
            ucons = vcat(auto_constraints.ucons, user_constraints.ucons)
        )
    end
end

