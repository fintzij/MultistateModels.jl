# =============================================================================
# Phase-Type Constraint Generation
# =============================================================================
#
# Generate identifiability constraints for phase-type models:
#
# 1. SCTP (Stationary Conditional Transition Probabilities) - B2
#    Ensures P(dest | leaving state) is constant across phases.
#    On NATURAL scale: μ_{j,d1}/μ_{1,d1} = μ_{j,d2}/μ_{1,d2}
#    Implemented as: μ_{j,d1} * μ_{1,d2} - μ_{j,d2} * μ_{1,d1} = 0
#
# 2. Eigenvalue Ordering - B3 (ordered_sctp)
#    Enforces total rate ordering: ν₁ ≥ ν₂ ≥ ... ≥ νₙ
#    where νⱼ = λⱼ + Σ_d μ_{j,d} is the total rate out of phase j.
#
# 3. C1 Covariate Constraints
#    Enforces shared covariate effects across phases per destination.
#    Implements: β_{j,d} = β_{1,d} for all j, per destination d.
#
# Contents:
# - _generate_sctp_constraints: SCTP constraints (B2)
# - _generate_ordering_constraints: Eigenvalue ordering (B1/B3)
# - _generate_c1_constraints: Covariate equality constraints (C1)
# - _merge_constraints: Combine constraint sets
#
# Reference: docs/src/phasetype_identifiability.md
#
# =============================================================================

"""
    _generate_sctp_constraints(expanded_hazards, mappings, pt_states) -> NamedTuple

Generate SCTP constraints ensuring P(dest | leaving state) is constant across phases.

On the NATURAL scale, the constraint is:
    μ_{j,d1}/μ_{1,d1} = μ_{j,d2}/μ_{1,d2}
    
Rearranged as: μ_{j,d1} * μ_{1,d2} - μ_{j,d2} * μ_{1,d1} = 0

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
                # Get baseline parameter names (first parameter = rate on NATURAL scale)
                ref_phase1_par = ref_hazards[1].parnames[1]
                ref_phaseP_par = ref_hazards[p].parnames[1]
                other_phase1_par = other_hazards[1].parnames[1]
                other_phaseP_par = other_hazards[p].parnames[1]
                
                # SCTP constraint on NATURAL scale:
                # τ_j must be the same for all destinations, where τ_j = μ_{j,d}/μ_{1,d}
                # So: μ_{j,d1}/μ_{1,d1} = μ_{j,d2}/μ_{1,d2}
                # Rearranged: μ_{j,d1} * μ_{1,d2} - μ_{j,d2} * μ_{1,d1} = 0
                constraint_expr = :( $other_phaseP_par * $ref_phase1_par - $ref_phaseP_par * $other_phase1_par )
                
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
    _generate_ordering_constraints(expanded_hazards, mappings, pt_states, user_hazards, ordering_reference) -> NamedTuple

Generate eigenvalue (total rate) ordering constraints for phase-type models.

Enforces the SCTP canonical representation constraint:
    ν₁ ≤ ν₂ ≤ ... ≤ νₙ (increasing eigenvalues)

where νⱼ = λⱼ + Σ_d μ_{j,d}·exp(β_{j,d}'x̄) is the total rate out of phase j
at reference covariate values x̄.

For the final phase (j = n), there is no progression rate (λₙ = 0), so νₙ = Σ_d μ_{n,d}·exp(β_{n,d}'x̄).

The constraints are implemented as inequality constraints:
    ν_{j-1}(x̄) - νⱼ(x̄) ≤ 0  for j = 2, ..., n

# Arguments
- `expanded_hazards`: Expanded hazards from phase-type model
- `mappings::PhaseTypeMappings`: State-to-phase mappings  
- `pt_states::Set{Int}`: States with phase-type hazards
- `user_hazards`: Original user hazard specifications (to detect C1 constraints)
- `ordering_reference::Dict{Symbol, Float64}`: Reference covariate values.
  Empty dict signals baseline (x=0) → linear constraints.

# Returns
NamedTuple with `cons`, `lcons`, `ucons` fields, or `nothing` if no constraints needed.

# Notes
- SCTP identifiability requires increasing eigenvalue ordering (ν₁ ≤ ν₂ ≤ ... ≤ νₙ).
- At baseline (empty `ordering_reference`) or with C1 constraints (homogeneous covariates),
  the constraint simplifies to linear: λⱼ + Σ_d μ_{j,d} - λ_{j-1} - Σ_d μ_{j-1,d} ≤ 0
- With heterogeneous covariate effects at non-baseline reference, the constraint is nonlinear
  and includes exp(β'x̄) terms.
- This function generates `n-1` inequality constraints per PT state.

See also: [`_generate_sctp_constraints`](@ref), [`_build_phasetype_model_from_hazards`](@ref)
"""
function _generate_ordering_constraints(expanded_hazards::Vector{<:_Hazard},
                                        mappings::PhaseTypeMappings,
                                        pt_states::Set{Int},
                                        user_hazards::Vector{<:HazardFunction},
                                        ordering_reference::Dict{Symbol, Float64})
    cons = Expr[]
    lcons = Float64[]
    ucons = Float64[]
    
    # Check if all PT hazards have homogeneous (C1) covariate constraints
    # In this case, exp(β'x̄) factors cancel and we can use linear constraints
    all_homogeneous = all(h -> begin
        !(h isa PhaseTypeHazard) || h.covariate_constraints === :homogeneous
    end, user_hazards)
    
    # Use baseline (linear) constraint logic if:
    # 1. ordering_reference is empty (baseline ordering requested), OR
    # 2. all PT hazards have C1 (homogeneous) covariate constraints
    use_linear_constraints = isempty(ordering_reference) || all_homogeneous
    
    # For each phase-type state with multiple phases
    for s in pt_states
        n_phases = mappings.n_phases_per_state[s]
        n_phases <= 1 && continue  # No ordering constraints for single-phase states
        
        # Find all progression hazards from state s (h{s}_{ab}, h{s}_{bc}, etc.)
        # Pattern: h{state}_{from_letter}{to_letter}
        progression_hazards = Dict{Int, _Hazard}()  # phase_idx => hazard
        
        for h in expanded_hazards
            hazname_str = string(h.hazname)
            m = match(r"^h(\d+)_([a-z])([a-z])$", hazname_str)
            isnothing(m) && continue
            
            from_state = parse(Int, m.captures[1])
            from_state == s || continue
            
            from_letter = m.captures[2][1]
            from_phase = Int(from_letter - 'a' + 1)
            progression_hazards[from_phase] = h
        end
        
        # Find all exit hazards from state s (h{s}{d}_{letter})
        # Group by phase
        exit_hazards_by_phase = Dict{Int, Vector{_Hazard}}()
        
        for h in expanded_hazards
            hazname_str = string(h.hazname)
            m = match(r"^h(\d+)(\d+)_([a-z])$", hazname_str)
            isnothing(m) && continue
            
            from_state = parse(Int, m.captures[1])
            from_state == s || continue
            
            phase_letter = m.captures[3][1]
            phase_idx = Int(phase_letter - 'a' + 1)
            
            if !haskey(exit_hazards_by_phase, phase_idx)
                exit_hazards_by_phase[phase_idx] = _Hazard[]
            end
            push!(exit_hazards_by_phase[phase_idx], h)
        end
        
        # Build total rate expression for each phase: νⱼ = λⱼ + Σ_d μ_{j,d}·exp(β_{j,d}'x̄)
        # SCTP constraint: ν_{j-1} - νⱼ ≤ 0 (i.e., νⱼ ≥ ν_{j-1}, increasing eigenvalues)
        for j in 2:n_phases
            constraint_expr = if use_linear_constraints
                _build_linear_ordering_constraint(j, progression_hazards, exit_hazards_by_phase)
            else
                _build_nonlinear_ordering_constraint(j, progression_hazards, exit_hazards_by_phase, ordering_reference)
            end
            
            if !isnothing(constraint_expr)
                push!(cons, constraint_expr)
                push!(lcons, -Inf)  # No lower bound
                push!(ucons, 0.0)   # constraint ≤ 0
            end
        end
    end
    
    isempty(cons) && return nothing
    
    return (cons = cons, lcons = lcons, ucons = ucons)
end

# Backward-compatible overload for existing code that doesn't pass ordering_reference
function _generate_ordering_constraints(expanded_hazards::Vector{<:_Hazard},
                                        mappings::PhaseTypeMappings,
                                        pt_states::Set{Int})
    # Create empty user_hazards and ordering_reference for linear constraints
    return _generate_ordering_constraints(
        expanded_hazards, mappings, pt_states, 
        HazardFunction[], Dict{Symbol, Float64}()
    )
end

"""
    _build_linear_ordering_constraint(j, progression_hazards, exit_hazards_by_phase) -> Expr

Build linear SCTP ordering constraint: ν_{j-1} - νⱼ ≤ 0 (i.e., νⱼ ≥ ν_{j-1})

where νⱼ = λⱼ + Σ_d μ_{j,d} (baseline rates only).

Used when ordering_at=:baseline or when C1 constraints make covariate factors cancel.
"""
function _build_linear_ordering_constraint(j::Int, 
                                            progression_hazards::Dict,
                                            exit_hazards_by_phase::Dict)
    terms = Expr[]
    
    # SCTP constraint: ν_{j-1} - νⱼ ≤ 0 (increasing eigenvalues)
    # So: -νⱼ and +ν_{j-1}
    sign_j = :-
    sign_jm1 = :+
    
    # Add ±λⱼ if phase j has progression (phases 1 to n-1 have progression)
    if haskey(progression_hazards, j)
        prog_par = progression_hazards[j].parnames[1]
        push!(terms, Expr(:call, sign_j, prog_par))
    end
    
    # Add ±λ_{j-1}
    if haskey(progression_hazards, j-1)
        prog_par_prev = progression_hazards[j-1].parnames[1]
        push!(terms, Expr(:call, sign_jm1, prog_par_prev))
    end
    
    # Add ±μ_{j,d} for all destinations
    if haskey(exit_hazards_by_phase, j)
        for exit_haz in exit_hazards_by_phase[j]
            exit_par = exit_haz.parnames[1]  # baseline rate
            push!(terms, Expr(:call, sign_j, exit_par))
        end
    end
    
    # Add ±μ_{j-1,d} for all destinations
    if haskey(exit_hazards_by_phase, j-1)
        for exit_haz in exit_hazards_by_phase[j-1]
            exit_par_prev = exit_haz.parnames[1]
            push!(terms, Expr(:call, sign_jm1, exit_par_prev))
        end
    end
    
    isempty(terms) && return nothing
    return Expr(:call, :+, terms...)
end

"""
    _build_nonlinear_ordering_constraint(j, progression_hazards, exit_hazards_by_phase, ordering_reference) -> Expr

Build nonlinear SCTP ordering constraint: ν_{j-1}(x̄) - νⱼ(x̄) ≤ 0 (i.e., νⱼ ≥ ν_{j-1})

where νⱼ(x̄) = λⱼ + Σ_d μ_{j,d}·exp(β_{j,d}'x̄).

For proportional hazards, the exit rate at reference values is:
    μ_{j,d}(x̄) = μ_{j,d} · exp(Σ_k β_{j,d,k} · x̄_k)

Used when ordering_at is not :baseline and covariates are not homogeneous.
"""
function _build_nonlinear_ordering_constraint(j::Int, 
                                               progression_hazards::Dict,
                                               exit_hazards_by_phase::Dict,
                                               ordering_reference::Dict{Symbol, Float64})
    terms = Expr[]
    
    # SCTP constraint: ν_{j-1} - νⱼ ≤ 0 (increasing eigenvalues)
    # So: -νⱼ and +ν_{j-1}
    sign_j = :negative
    sign_jm1 = :positive
    
    # Progression hazards don't have covariates in phase-type models (they're internal transitions)
    # So λⱼ and λ_{j-1} are simple baseline parameters
    
    # Add ±λⱼ if phase j has progression
    if haskey(progression_hazards, j)
        prog_par = progression_hazards[j].parnames[1]
        push!(terms, sign_j === :positive ? :( +$prog_par ) : :( -$prog_par ))
    end
    
    # Add ±λ_{j-1}
    if haskey(progression_hazards, j-1)
        prog_par_prev = progression_hazards[j-1].parnames[1]
        push!(terms, sign_jm1 === :positive ? :( +$prog_par_prev ) : :( -$prog_par_prev ))
    end
    
    # Add ±μ_{j,d}·exp(β_{j,d}'x̄) for all destinations in phase j
    if haskey(exit_hazards_by_phase, j)
        for exit_haz in exit_hazards_by_phase[j]
            rate_expr = _build_rate_with_covariates(exit_haz, ordering_reference, sign_j)
            push!(terms, rate_expr)
        end
    end
    
    # Add ±μ_{j-1,d}·exp(β_{j-1,d}'x̄) for all destinations in phase j-1
    if haskey(exit_hazards_by_phase, j-1)
        for exit_haz in exit_hazards_by_phase[j-1]
            rate_expr = _build_rate_with_covariates(exit_haz, ordering_reference, sign_jm1)
            push!(terms, rate_expr)
        end
    end
    
    isempty(terms) && return nothing
    return Expr(:call, :+, terms...)
end

"""
    _build_rate_with_covariates(hazard, ordering_reference, sign) -> Expr

Build expression for μ·exp(β'x̄) for a single exit hazard.

# Arguments
- `hazard`: The exit hazard (_Hazard)
- `ordering_reference`: Reference covariate values
- `sign`: :positive for +μ·exp(...) or :negative for -μ·exp(...)

# Returns
Expression like: +h12_a_rate * exp(h12_a_age * 50.0 + h12_a_trt * 0.5)
"""
function _build_rate_with_covariates(hazard::_Hazard,
                                      ordering_reference::Dict{Symbol, Float64},
                                      sign::Symbol)
    baseline_par = hazard.parnames[1]  # First parameter is baseline rate
    
    # Check if hazard has covariates (more than just baseline parameters)
    if hazard.npar_baseline >= hazard.npar_total || isempty(ordering_reference)
        # No covariates or no reference values → just the baseline rate
        return sign === :positive ? :( +$baseline_par ) : :( -$baseline_par )
    end
    
    # Build linear predictor: β₁·x̄₁ + β₂·x̄₂ + ...
    covar_parnames = hazard.parnames[(hazard.npar_baseline + 1):end]
    covar_names = hazard.covar_names  # Covariate names (symbols like :age, :trt)
    
    linpred_terms = Expr[]
    for (i, covar_par) in enumerate(covar_parnames)
        if i <= length(covar_names)
            covar_name = covar_names[i]
            if haskey(ordering_reference, covar_name)
                xbar = ordering_reference[covar_name]
                # Term: β_k * x̄_k
                push!(linpred_terms, :( $covar_par * $xbar ))
            end
        end
    end
    
    if isempty(linpred_terms)
        # No matching covariates in reference → just baseline rate
        return sign === :positive ? :( +$baseline_par ) : :( -$baseline_par )
    end
    
    # Build exp(β'x̄) expression
    linpred_expr = if length(linpred_terms) == 1
        linpred_terms[1]
    else
        Expr(:call, :+, linpred_terms...)
    end
    
    # Full expression: ±μ * exp(β'x̄)
    if sign === :positive
        return :( +$baseline_par * exp($linpred_expr) )
    else
        return :( -$baseline_par * exp($linpred_expr) )
    end
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

"""
    _generate_c1_constraints(expanded_hazards, user_hazards) -> NamedTuple

Generate equality constraints for C1 covariate parameter sharing.

With C1 constraints, phases share covariate effects per destination. Parameter names
remain phase-specific (e.g., `h12_a_age`, `h12_b_age`) for unique identification,
but equality constraints tie them together:

    h12_a_age - h12_b_age = 0

This effectively reduces the degrees of freedom from n×p to p for n phases and p
covariates, while maintaining unique parameter names in the flat vector.

# Arguments
- `expanded_hazards`: Expanded hazards from phase-type model
- `user_hazards`: Original user hazard specifications (to check covariate_constraints)

# Returns
NamedTuple with `cons`, `lcons`, `ucons` fields, or `nothing` if no C1 hazards.

# Example
For a 2-phase PT hazard from state 1→2 with covariate `age`:
- Parameters: `h12_a_age`, `h12_b_age`
- Constraint: `h12_a_age - h12_b_age = 0`

See also: [`_build_phasetype_model_from_hazards`](@ref), [`_generate_sctp_constraints`](@ref)
"""
function _generate_c1_constraints(expanded_hazards::Vector{<:_Hazard},
                                   user_hazards::Vector{<:HazardFunction})
    cons = Expr[]
    lcons = Float64[]
    ucons = Float64[]
    
    # Find all PhaseTypeHazards with C1 constraints
    for h in user_hazards
        !(h isa PhaseTypeHazard) && continue
        h.covariate_constraints !== :homogeneous && continue
        h.n_phases <= 1 && continue  # No sharing needed for single phase
        
        s = h.statefrom
        d = h.stateto
        
        # Find all expanded exit hazards for this transition (h{s}{d}_{letter})
        exit_hazards = filter(expanded_hazards) do eh
            hazname_str = string(eh.hazname)
            m = match(r"^h(\d+)(\d+)_([a-z])$", hazname_str)
            !isnothing(m) && parse(Int, m.captures[1]) == s && parse(Int, m.captures[2]) == d
        end
        
        length(exit_hazards) <= 1 && continue  # Need at least 2 phases
        
        # Sort by phase letter (a, b, c, ...)
        sort!(exit_hazards, by = eh -> begin
            m = match(r"_([a-z])$", string(eh.hazname))
            m.captures[1][1]
        end)
        
        # Use first phase as reference
        ref_haz = exit_hazards[1]
        
        # Get covariate parameter names (skip baseline parameter)
        if ref_haz.npar_baseline < length(ref_haz.parnames)
            # Covariate parameters start after baseline
            ref_covar_parnames = ref_haz.parnames[(ref_haz.npar_baseline + 1):end]
            
            # For each covariate parameter, constrain all other phases to equal phase 1
            for (covar_idx, ref_parname) in enumerate(ref_covar_parnames)
                for other_haz in exit_hazards[2:end]
                    # Get corresponding covariate parameter from other hazard
                    other_parname = other_haz.parnames[other_haz.npar_baseline + covar_idx]
                    
                    # Generate constraint: ref_par - other_par = 0
                    # Uses the unique phase-specific parameter names
                    constraint_expr = :( $ref_parname - $other_parname )
                    
                    push!(cons, constraint_expr)
                    push!(lcons, 0.0)
                    push!(ucons, 0.0)
                end
            end
        end
    end
    
    isempty(cons) && return nothing
    
    return (cons = cons, lcons = lcons, ucons = ucons)
end
