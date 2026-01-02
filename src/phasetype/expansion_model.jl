# =============================================================================
# Phase-Type Model Building
# =============================================================================
#
# Build MultistateMarkovModel with phase-type expansion from user hazard specs.
# Called by multistatemodel() when any hazard is :pt.
#
# Contents:
# - _build_phasetype_model_from_hazards: Main model building function
# - _build_expanded_parameters: Build parameters for expanded hazards
# - _build_original_parameters: Extract initial parameters from user hazards
# - _extract_original_natural_vector: Extract parameter vector from hazard
# - _count_covariates: Count covariates in formula
# - _count_hazard_parameters: Count parameters in hazard specification
# - _merge_censoring_patterns_with_shift: Merge user and phase censoring patterns
#
# =============================================================================

"""
    _build_phasetype_model_from_hazards(hazards, data; kwargs...) -> MultistateMarkovModel

Build a MultistateMarkovModel with phase-type expansion from user hazard specs.

Called by `multistatemodel()` when any hazard is `:pt`. Builds expanded state space,
expands hazards and data, and stores metadata in `phasetype_expansion` field.

# Arguments
- `hazards::Tuple`: User hazard specifications (some :pt)
- `data::DataFrame`: Observation data
- `n_phases::Union{Nothing,Dict{Int,Int}}`: Phases per state (default: 2)
- `coxian_structure::Symbol`: `:unstructured` or `:sctp`
- Other kwargs: `constraints`, `SubjectWeights`, `CensoringPatterns`, etc.

# Returns
`MultistateMarkovModel` on expanded state space with `phasetype_expansion` metadata.

See also: [`PhaseTypeExpansion`](@ref), [`build_phasetype_mappings`](@ref)
"""
function _build_phasetype_model_from_hazards(hazards::Tuple{Vararg{HazardFunction}};
                                              data::DataFrame,
                                              constraints = nothing,
                                              initialize::Bool = true,
                                              n_phases::Union{Nothing, Dict{Int,Int}} = nothing,
                                              coxian_structure::Symbol = :unstructured,
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
    n_states = size(tmat_original, 1)
    
    if verbose
        println("Building phase-type model...")
        println("  Original states: $n_states")
    end
    
    # Step 1b: Validate and process n_phases
    # Identify states with :pt hazards
    pt_states = Set{Int}()
    for h in hazards_ordered
        if h isa PhaseTypeHazard
            push!(pt_states, h.statefrom)
        end
    end
    
    # Build n_phases_per_state from model-level n_phases dict
    n_phases_per_state = ones(Int, n_states)  # Default: 1 phase for non-pt states
    
    if isnothing(n_phases) || isempty(n_phases)
        # Legacy path: use n_phases from PhaseTypeHazard
        for h in hazards_ordered
            if h isa PhaseTypeHazard
                s = h.statefrom
                n_phases_per_state[s] = max(n_phases_per_state[s], h.n_phases)
            end
        end
    else
        # New path: use model-level n_phases dict
        # Validate: all pt states must be in dict
        for s in pt_states
            if !haskey(n_phases, s)
                throw(ArgumentError("State $s has :pt hazards but is not in n_phases dict. " *
                      "Specify n_phases = Dict($s => k) where k is the number of phases."))
            end
        end
        
        # Validate: dict shouldn't have states without :pt hazards
        for (s, k) in n_phases
            if s ∉ pt_states
                throw(ArgumentError("n_phases specifies $k phases for state $s, but state $s has no :pt hazards."))
            end
            if k < 1
                throw(ArgumentError("n_phases[$s] must be ≥ 1, got $k"))
            end
            # Coerce n_phases = 1 to exponential (no warning, just do it)
            n_phases_per_state[s] = k
        end
    end
    
    # Step 1c: Check for mixed hazard types from states with :pt hazards
    for s in pt_states
        outgoing_hazards = filter(h -> h.statefrom == s, hazards_ordered)
        non_pt_hazards = filter(h -> !(h isa PhaseTypeHazard), outgoing_hazards)
        if !isempty(non_pt_hazards)
            non_pt_names = [Symbol("h$(h.statefrom)$(h.stateto)") for h in non_pt_hazards]
            @warn "State $s has :pt hazards but also has non-:pt hazards: $non_pt_names. " *
                  "All outgoing hazards from state $s will be treated as phase-type."
        end
    end
    
    if verbose
        println("  Phases per state: $n_phases_per_state")
        println("  Coxian structure: $coxian_structure")
    end
    
    # Step 2: Build mappings from n_phases_per_state
    mappings = build_phasetype_mappings(hazards_ordered, tmat_original, n_phases_per_state)
    
    if verbose
        println("  Expanded states: $(mappings.n_expanded)")
        println("  Phases per state: $(mappings.n_phases_per_state)")
    end
    
    # Step 3: Expand hazards to runtime representation
    # hazard_to_params_idx maps each hazard index to its parameter index
    # (shared hazards point to the same parameter index as their base)
    expanded_hazards, expanded_params_list, hazard_to_params_idx = expand_hazards_for_phasetype(
        hazards_ordered, mappings, data
    )
    
    if verbose
        println("  Expanded hazards: $(length(expanded_hazards))")
    end
    
    # Step 4: Expand data for phase uncertainty
    expanded_data, phase_censoring_patterns, original_row_map = expand_data_for_phasetype_fitting(data, mappings)
    
    # Step 5: Build expanded hazkeys
    # Use hazard_to_params_idx to map to parameter indices (handles shared hazards)
    expanded_hazkeys = Dict{Symbol, Int64}()
    for (idx, h) in enumerate(expanded_hazards)
        # Use the parameter index, not the hazard index
        # For shared hazards, this will be the base hazard's parameter index
        expanded_hazkeys[h.hazname] = hazard_to_params_idx[idx]
    end
    
    # Step 6: Build parameters structure
    expanded_parameters = _build_expanded_parameters(expanded_params_list, expanded_hazkeys, expanded_hazards, hazard_to_params_idx)
    
    # Step 7: Get subject indices on expanded data
    subjinds, nsubj = get_subjinds(expanded_data)
    
    # Step 8: Handle weights
    SubjectWeights, ObservationWeights = check_weight_exclusivity(SubjectWeights, ObservationWeights, nsubj)
    
    # Step 9: Prepare censoring patterns for expanded space
    if CensoringPatterns === nothing
        CensoringPatterns_expanded = phase_censoring_patterns
    else
        # Merge user patterns with phase uncertainty patterns
        # User patterns are shifted to avoid conflict with phase patterns
        CensoringPatterns_expanded, expanded_data = _merge_censoring_patterns_with_shift(
            CensoringPatterns, phase_censoring_patterns, mappings, expanded_data
        )
    end
    
    # Validate inputs
    CensoringPatterns_final = _prepare_censoring_patterns(CensoringPatterns_expanded, mappings.n_expanded)
    _validate_inputs!(expanded_data, mappings.expanded_tmat, CensoringPatterns_final, SubjectWeights, ObservationWeights; verbose = verbose)
    
    # Step 10: Build emission matrix
    if !isnothing(EmissionMatrix)
        # Expand user-supplied emission matrix to expanded state space
        expanded_emat = expand_emission_matrix(EmissionMatrix, original_row_map, mappings)
        emat = build_emat(expanded_data, CensoringPatterns_final, expanded_emat, mappings.expanded_tmat)
    else
        emat = build_emat(expanded_data, CensoringPatterns_final, nothing, mappings.expanded_tmat)
    end
    
    # Step 11: Build total hazards on expanded space
    expanded_totalhazards = build_totalhazards(expanded_hazards, mappings.expanded_tmat)
    
    # Step 11b: Generate SCTP constraints if requested
    final_constraints = constraints
    if coxian_structure === :sctp
        sctp_constraints = _generate_sctp_constraints(expanded_hazards, mappings, pt_states)
        if !isnothing(sctp_constraints)
            if verbose
                println("  Generated $(length(sctp_constraints.cons)) SCTP constraints")
            end
            final_constraints = _merge_constraints(constraints, sctp_constraints)
        end
    end
    
    # Step 12: Store modelcall
    modelcall = (
        hazards = hazards, 
        data = data, 
        constraints = final_constraints, 
        SubjectWeights = SubjectWeights, 
        ObservationWeights = ObservationWeights, 
        CensoringPatterns = CensoringPatterns, 
        EmissionMatrix = EmissionMatrix,
        is_phasetype = true  # Mark this as a phase-type model for fitted model accessors
    )
    
    # Step 13: Build original parameters structure (for user-facing API)
    original_parameters = _build_original_parameters(hazards_ordered, data)
    
    # Step 14: Create PhaseTypeExpansion metadata
    phasetype_expansion = PhaseTypeExpansion(
        mappings,
        data,              # original data
        tmat_original,     # original tmat
        hazards_ordered,   # original hazard specs
        original_parameters
    )
    
    # Step 15: Assemble the MultistateMarkovModel on expanded space
    # The model operates on expanded state space; phasetype_expansion holds original mappings
    model = MultistateMarkovModel(
        expanded_data,
        expanded_parameters,
        expanded_hazards,
        expanded_totalhazards,
        mappings.expanded_tmat,
        emat,
        expanded_hazkeys,
        subjinds,
        SubjectWeights,
        ObservationWeights,
        CensoringPatterns_final,
        nothing,  # markovsurrogate - set during fitting if needed
        modelcall,
        phasetype_expansion
    )
    
    # Step 16: Initialize parameters (phase-type models use :crude like other Markov models)
    if initialize
        initialize_parameters!(model; constraints = final_constraints)
    end
    
    if verbose
        println("  MultistateMarkovModel (phase-type) created successfully")
    end
    
    return model
end

"""
    _build_expanded_parameters(params_list, hazkeys, hazards, hazard_to_params_idx)

Build ParameterHandling-compatible parameter structure for expanded hazards.

# Arguments
- `params_list`: Vector of parameter vectors, one per unique parameter set
- `hazkeys`: Dict mapping hazard name to parameter index
- `hazards`: Vector of all expanded hazards (may have more entries than params_list for shared hazards)
- `hazard_to_params_idx`: Maps each hazard index to its parameter index (default: identity mapping)
"""
function _build_expanded_parameters(params_list::Vector{Vector{Float64}}, 
                                     hazkeys::Dict{Symbol, Int64}, 
                                     hazards::Vector{<:_Hazard},
                                     hazard_to_params_idx::Vector{Int} = collect(1:length(hazards)))
    # Build a reverse mapping: for each parameter index, find a hazard that uses it
    # (for shared hazards, any of them will have the same parnames/npar_baseline)
    params_idx_to_hazard_idx = Dict{Int, Int}()
    for (haz_idx, params_idx) in enumerate(hazard_to_params_idx)
        if !haskey(params_idx_to_hazard_idx, params_idx)
            params_idx_to_hazard_idx[params_idx] = haz_idx
        end
    end
    
    # Build nested parameters structure per unique parameter set
    params_nested_pairs = [
        let haz_idx = params_idx_to_hazard_idx[params_idx]
            hazname => build_hazard_params(params_list[params_idx], hazards[haz_idx].parnames, hazards[haz_idx].npar_baseline, length(params_list[params_idx]))
        end
        for (hazname, params_idx) in sort(collect(hazkeys), by = x -> x[2])
    ]
    params_nested = NamedTuple(params_nested_pairs)
    
    # Create ReConstructor for AD-compatible flatten/unflatten
    reconstructor = ReConstructor(params_nested, unflattentype=UnflattenFlexible())
    params_flat = flatten(reconstructor, params_nested)
    
    # Get natural scale parameters (family-aware transformation)
    params_natural_pairs = [
        let haz_idx = params_idx_to_hazard_idx[params_idx]
            hazname => extract_natural_vector(params_nested[hazname], hazards[haz_idx].family)
        end
        for (hazname, params_idx) in sort(collect(hazkeys), by = x -> x[2])
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

Returns a structure matching the standard model parameters format:
- `flat`: Vector of log-scale baseline params + covariate coefficients
- `nested`: NamedTuple per hazard with `baseline` and `covariates` NamedTuples
- `natural`: NamedTuple per hazard with simple Vector{Float64} of natural-scale values
- `reconstructor`: ReConstructor for flatten/unflatten operations
"""
function _build_original_parameters(hazards::Vector{<:HazardFunction}, data::DataFrame)
    hazkeys = Dict{Symbol, Int64}()
    params_nested_pairs = Vector{Pair{Symbol, NamedTuple}}()
    
    for (idx, h) in enumerate(hazards)
        hazname = Symbol("h$(h.statefrom)$(h.stateto)")
        hazkeys[hazname] = idx
        
        if h isa PhaseTypeHazard
            # PT hazard: 2n-1 baseline params (λ rates + μ exit rates) + covariates
            n = h.n_phases
            npar_baseline = 2 * n - 1
            npar_covar = _count_covariates(h.hazard, data)
            
            # Build baseline parameter names: λ₁, λ₂, ..., μ₁, μ₂, ..., μₙ
            baseline_names = Symbol[]
            for i in 1:(n-1)
                push!(baseline_names, Symbol("$(hazname)_λ$i"))
            end
            for i in 1:n
                push!(baseline_names, Symbol("$(hazname)_μ$i"))
            end
            baseline_nt = NamedTuple{Tuple(baseline_names)}(zeros(npar_baseline))
            
            # Build covariate parameter names if any
            if npar_covar > 0
                covar_names = [Symbol("$(hazname)_covar$i") for i in 1:npar_covar]
                covar_nt = NamedTuple{Tuple(covar_names)}(zeros(npar_covar))
                push!(params_nested_pairs, hazname => (baseline = baseline_nt, covariates = covar_nt))
            else
                push!(params_nested_pairs, hazname => (baseline = baseline_nt,))
            end
        else
            # Non-PT hazard: use standard parameter structure
            npar = _count_hazard_parameters(h, data)
            npar_covar = _count_covariates(h.hazard, data)
            npar_baseline = npar - npar_covar
            
            baseline_names = [Symbol("$(hazname)_baseline$i") for i in 1:npar_baseline]
            baseline_nt = NamedTuple{Tuple(baseline_names)}(zeros(npar_baseline))
            
            if npar_covar > 0
                covar_names = [Symbol("$(hazname)_covar$i") for i in 1:npar_covar]
                covar_nt = NamedTuple{Tuple(covar_names)}(zeros(npar_covar))
                push!(params_nested_pairs, hazname => (baseline = baseline_nt, covariates = covar_nt))
            else
                push!(params_nested_pairs, hazname => (baseline = baseline_nt,))
            end
        end
    end
    
    # Sort by hazard index and build nested structure
    sorted_pairs = sort(params_nested_pairs, by = p -> hazkeys[p.first])
    params_nested = NamedTuple(sorted_pairs)
    
    # Build reconstructor and flat vector
    reconstructor = ReConstructor(params_nested, unflattentype=UnflattenFlexible())
    params_flat = flatten(reconstructor, params_nested)
    
    # Build natural-scale parameters (simple vectors per hazard)
    # For phase-type, all baseline params are rates (positive), so exp transform
    params_natural_pairs = [
        hazname => _extract_original_natural_vector(params_nested[hazname])
        for hazname in keys(params_nested)
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
    _extract_original_natural_vector(hazard_params::NamedTuple)

Extract natural-scale parameter vector from original phase-type hazard params.
All baseline parameters are rates (positive), so apply exp transform.
Covariate coefficients are unconstrained (kept as-is).
"""
function _extract_original_natural_vector(hazard_params::NamedTuple)
    baseline_vals = collect(values(hazard_params.baseline))
    baseline_natural = exp.(baseline_vals)
    
    if haskey(hazard_params, :covariates)
        covar_vals = collect(values(hazard_params.covariates))
        return vcat(baseline_natural, covar_vals)
    else
        return baseline_natural
    end
end

"""
    _count_covariates(formula, data)

Count the number of covariate parameters in a hazard formula.
"""
function _count_covariates(formula::FormulaTerm, data::DataFrame)
    schema = StatsModels.schema(formula, data)
    hazschema = apply_schema(formula, schema)
    rhs_names = _hazard_rhs_names(hazschema)
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
    _merge_censoring_patterns_with_shift(user_patterns, phase_patterns, mappings, data)

Merge user-provided censoring patterns with phase uncertainty patterns,
shifting user codes to avoid conflicts with phase codes.

Phase patterns use codes 3 to 2+n_observed (for sojourn in each observed state).
User patterns are shifted by n_observed, so user code 3 becomes 3+n_observed, etc.

Returns the combined censoring patterns AND the updated data with shifted obstypes.
"""
function _merge_censoring_patterns_with_shift(user_patterns::Matrix{<:Real}, 
                                               phase_patterns::Matrix{Float64},
                                               mappings::PhaseTypeMappings,
                                               data::DataFrame)
    n_observed = mappings.n_observed
    n_expanded = mappings.n_expanded
    n_user = size(user_patterns, 1)
    
    # Phase patterns use codes 3 to 2+n_observed
    # Shift user patterns by n_observed: user code 3 → 3+n_observed
    shift = n_observed
    
    # Create expanded user patterns with shifted codes
    expanded_user = zeros(Float64, n_user, n_expanded + 1)
    
    for p in 1:n_user
        original_code = Int(user_patterns[p, 1])
        shifted_code = original_code + shift
        expanded_user[p, 1] = shifted_code
        
        for s_obs in 1:n_observed
            obs_prob = user_patterns[p, s_obs + 1]
            phases = mappings.state_to_phases[s_obs]
            n_phases = length(phases)
            for phase in phases
                expanded_user[p, phase + 1] = obs_prob / n_phases
            end
        end
    end
    
    # Combine phase patterns and expanded user patterns
    combined = vcat(phase_patterns, expanded_user)
    
    # Update data.obstype to use shifted codes for user patterns
    # Original user codes are >= 3 (standard) or >= 3+n_observed already (from earlier expansion)
    # We need to shift any code > 2+n_observed by shift amount
    # Actually, in the expanded data, phase-generated codes are already 3 to 2+n_observed
    # We need to shift codes that came from user's original data
    
    # The user's original obstype values > 2 that aren't phase-generated should be shifted
    # But how do we know which are user's vs phase-generated?
    # In expand_data_for_phasetype, sojourn rows get obstype = 2 + statefrom (3 to 2+n_states)
    # Panel observations (obstype=2) stay as 2
    # User's custom obstypes (>=3) in panel data would also be present
    
    # The issue: user's original obstype=3 could mean their first custom pattern
    # After expansion, obstype=3 means "sojourn in observed state 1" (phase-generated)
    
    # We need to remap: if original data had obstype > 2, shift by n_observed in expanded data
    # This requires tracking which rows came from original custom obstypes
    
    # For now, let's shift ALL obstypes > 2+n_observed in the expanded data
    # This handles the case where user's codes start at 3+n_observed or higher
    
    # Actually, the data expansion preserves original obstypes for panel observations
    # Let's check what codes are actually in the data and shift appropriately
    
    # Find the maximum phase code (2 + n_observed)
    max_phase_code = 2 + n_observed
    
    # Create a copy of data to modify
    updated_data = copy(data)
    
    # For any obstype > max_phase_code, it's already shifted or doesn't need shifting
    # For any obstype in [3, 2+n_observed], it could be either:
    #   - Phase-generated (sojourn) - don't shift
    #   - User's original pattern - need to shift
    #
    # The only way to distinguish is by checking if it's a sojourn row (stateto=0 and dt>0)
    # or a non-sojourn row (came from panel data with user's obstype)
    
    # Non-sojourn rows with obstype > 2 need shifting
    for i in 1:nrow(updated_data)
        ot = updated_data.obstype[i]
        if ot > 2 && updated_data.stateto[i] != 0
            # This is a non-sojourn row with a custom obstype
            # Could be from user's original panel data with obstype >= 3
            # BUT: in expand_data_for_phasetype, exact obs (obstype=1) become sojourn + instantaneous
            #      panel obs (obstype=2) stay as is with obstype=2
            #      custom obstype (>=3) should stay as is...
            # So actually, the phase expansion only creates sojourn rows with new codes
            # User's original codes >= 3 in non-exact observations should be shifted
            if ot >= 3 && ot <= max_phase_code
                # Potential conflict with phase codes - shift it
                updated_data.obstype[i] = ot + shift
            end
        end
    end
    
    return combined, updated_data
end

