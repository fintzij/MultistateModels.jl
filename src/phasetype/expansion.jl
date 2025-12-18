# =============================================================================
# Phase-Type Distribution Fitting for Semi-Markov Surrogates
# =============================================================================
#
# This file provides phase-type (PH) distribution fitting for use as 
# improved importance sampling proposals in MCEM. Phase-type distributions
# can better approximate non-exponential sojourns (Weibull, Gompertz) compared
# to simple Markov (exponential) surrogates.
#
# Core types and surrogate building are in types.jl and surrogate.jl.
# This file contains:
#   - State space expansion functions
#   - Hazard and data expansion for phase-type fitting
#   - SCTP constraint generation
#   - Model building for phase-type hazards
#   - Log-likelihood computation
#   - Data expansion for forward-backward sampling
#
# References:
# - Titman & Sharples (2010) Biometrics 66(3):742-752
# - Asmussen et al. (1996) Scand J Stat 23(4):419-441
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
        if h isa PhaseTypeHazardSpec
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
    expand_hazards_for_phasetype(hazards, mappings, data) -> (Vector{_Hazard}, Vector{Vector{Float64}})

Convert user hazard specs to runtime hazards on expanded phase-type state space.

- **:pt hazards**: (n-1) progression hazards (λ) + n exit hazards (μ)
- **:exp hazards**: One MarkovHazard per source phase, shared rate
- **Semi-Markov**: Appropriate hazard type on expanded space

# Arguments
- `hazards`: User-specified hazard specifications
- `mappings::PhaseTypeMappings`: State space mappings
- `data::DataFrame`: Model data for schema resolution

# Returns
Tuple of expanded hazards and initial parameters.

See also: [`build_phasetype_mappings`](@ref)
"""
function expand_hazards_for_phasetype(hazards::Vector{<:HazardFunction}, 
                                       mappings::PhaseTypeMappings,
                                       data::DataFrame)
    expanded_hazards = _Hazard[]
    expanded_params = Vector{Float64}[]
    # Maps each hazard index to its parameter index (for shared hazards, points to base)
    hazard_to_params_idx = Int[]
    
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
                push!(hazard_to_params_idx, length(expanded_params))
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
                    push!(hazard_to_params_idx, length(expanded_params))
                end
            else
                # Non-PT hazard: single hazard from each phase (shared rate)
                # Build the hazard once, then replicate for each phase
                base_haz, base_pars = _build_expanded_hazard(h, s, d, phases_s[1], first_phase_d, data)
                push!(expanded_hazards, base_haz)
                push!(expanded_params, base_pars)
                base_params_idx = length(expanded_params)
                push!(hazard_to_params_idx, base_params_idx)
                
                # For multi-phase states, create additional hazards that share parameters
                # These are distinct hazard objects but will use the same parameter indices
                for p in 2:n_phases
                    from_phase = phases_s[p]
                    shared_haz = _build_shared_phase_hazard(base_haz, from_phase, first_phase_d)
                    push!(expanded_hazards, shared_haz)
                    # Shared hazards point to the base hazard's parameter index
                    push!(hazard_to_params_idx, base_params_idx)
                end
            end
        end
    end
    
    return expanded_hazards, expanded_params, hazard_to_params_idx
end

"""
    _phase_index_to_letter(idx)

Convert a 1-based phase index to a letter (1='a', 2='b', ..., 26='z').

Used for naming phase-type hazards with letter-based phase labels.
"""
@inline function _phase_index_to_letter(idx::Int)
    @assert 1 <= idx <= 26 "Phase index must be between 1 and 26, got $idx"
    return Char('a' + idx - 1)
end

"""
    _build_progression_hazard(observed_state, phase_index, n_phases, from_phase, to_phase)

Build a MarkovHazard for internal phase progression (λ rate).

These hazards represent transitions between phases within the same observed state.
They are exponential hazards with no covariates.

# Naming Convention
Progression hazards are named `h{state}_{from_letter}{to_letter}`, e.g.:
- `h1_ab`: state 1, phase a → phase b
- `h1_bc`: state 1, phase b → phase c
"""
function _build_progression_hazard(observed_state::Int, phase_index::Int, n_phases::Int,
                                    from_phase::Int, to_phase::Int)
    # Parameter name using letter convention: h1_ab, h1_bc, etc.
    from_letter = _phase_index_to_letter(phase_index)
    to_letter = _phase_index_to_letter(phase_index + 1)
    hazname = Symbol("h$(observed_state)_$(from_letter)$(to_letter)")
    parname = Symbol("log_λ_$(hazname)")
    
    # Simple exponential hazard function (no covariates)
    # pars is a NamedTuple with baseline on NATURAL scale (no exp needed)
    hazard_fn = (t, pars, covars) -> only(values(pars.baseline))
    cumhaz_fn = (lb, ub, pars, covars) -> only(values(pars.baseline)) * (ub - lb)
    
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
    # Parameter name using letter convention: h12_a, h12_b, etc.
    phase_letter = _phase_index_to_letter(phase_index)
    hazname = Symbol("h$(observed_from)$(observed_to)_$(phase_letter)")
    baseline_parname = Symbol("log_λ_$(hazname)")
    
    # Get covariate info from original specification
    schema = StatsModels.schema(pt_spec.hazard, data)
    hazschema = apply_schema(pt_spec.hazard, schema)
    rhs_names = _hazard_rhs_names(hazschema)
    
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
        # pars is a NamedTuple with baseline on NATURAL scale (no exp needed)
        hazard_fn = (t, pars, covars) -> only(values(pars.baseline))
        cumhaz_fn = (lb, ub, pars, covars) -> only(values(pars.baseline)) * (ub - lb)
    else
        # Use generate_exponential_hazard from hazards.jl
        hazard_fn, cumhaz_fn = generate_exponential_hazard(parnames, linpred_effect)
    end
    return hazard_fn, cumhaz_fn
end

# NOTE: Use _hazard_rhs_names from construction/multistatemodel.jl instead of duplicating here

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
    rhs_names = _hazard_rhs_names(hazschema)
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
    expand_emission_matrix(EmissionMatrix, original_row_map, mappings)

Expand a user-supplied emission matrix (n_obs x n_observed) to the expanded state space
(n_expanded_obs x n_expanded_states).

# Arguments
- `EmissionMatrix`: User-supplied emission matrix
- `original_row_map`: Mapping from expanded rows to original rows
- `mappings`: Phase-type mappings

# Returns
Expanded emission matrix.
"""
function expand_emission_matrix(EmissionMatrix::Matrix{Float64}, 
                                original_row_map::Vector{Int}, 
                                mappings::PhaseTypeMappings)
    n_expanded_obs = length(original_row_map)
    n_expanded_states = mappings.n_expanded
    n_observed = mappings.n_observed
    
    # Validate input dimensions
    if size(EmissionMatrix, 2) != n_observed
        error("EmissionMatrix must have \$(n_observed) columns (one per observed state).")
    end
    
    expanded_emat = zeros(Float64, n_expanded_obs, n_expanded_states)
    
    for i in 1:n_expanded_obs
        orig_row = original_row_map[i]
        
        # For each observed state k, map its probability to all its phases
        for k in 1:n_observed
            prob = EmissionMatrix[orig_row, k]
            if prob > 0
                for p in mappings.state_to_phases[k]
                    expanded_emat[i, p] = prob
                end
            end
        end
    end
    
    return expanded_emat
end

"""
    expand_data_for_phasetype_fitting(data, mappings) -> (DataFrame, Matrix{Float64})

Expand observation data to handle phase uncertainty during phase-type model fitting.

Maps observed states to phase ranges, builds censoring patterns for phase uncertainty,
and splits exact observations for forward-backward marginalization.

# Arguments
- `data::DataFrame`: Original observation data
- `mappings::PhaseTypeMappings`: State space mappings

# Returns
Tuple of expanded data and censoring patterns matrix.

See also: [`build_phasetype_mappings`](@ref), [`expand_data_for_phasetype`](@ref)
"""
function expand_data_for_phasetype_fitting(data::DataFrame, mappings::PhaseTypeMappings)
    n_expanded = mappings.n_expanded
    n_observed = mappings.n_observed
    
    # Use expand_data_for_phasetype to split exact observations
    # This handles the sojourn + transition split correctly
    expansion_result = expand_data_for_phasetype(data, n_observed)
    expanded_data = expansion_result.expanded_data
    original_row_map = expansion_result.original_row_map
    
    # Build censoring patterns for phase uncertainty FIRST (needed for obstype mapping)
    # Each observed state maps to a set of possible phases
    # Row s corresponds to obstype = s + 2
    censoring_patterns = _build_phase_censoring_patterns(mappings)
    
    # Now map state indices to phase indices in expanded space
    # For statefrom: map to first phase of the state (always unambiguous - we know 
    # which phase we're starting from based on previous observation)
    # Handle special case: statefrom=0 (from censored interval) stays 0
    expanded_data.statefrom = [
        s == 0 ? 0 : first(mappings.state_to_phases[s]) 
        for s in expanded_data.statefrom
    ]
    
    # For stateto: handle based on observation type and phase uncertainty
    # - obstype=1 (exact): instantaneous transition, map to first phase (will be refined by hazard ratios)
    # - obstype=2 (panel): destination state has phase uncertainty, use censoring pattern
    # - obstype=4 (sojourn): stateto=0, stays 0
    # - other obstypes: already censored, stays 0
    n_rows = nrow(expanded_data)
    new_stateto = Vector{Int}(undef, n_rows)
    new_obstype = copy(expanded_data.obstype)
    
    for i in 1:n_rows
        s = expanded_data.stateto[i]
        obstype_i = expanded_data.obstype[i]
        
        if s == 0
            # Already censored (sojourn intervals, etc.) - keep as is
            new_stateto[i] = 0
        elseif obstype_i == 2
            # Panel observation: destination state has phase uncertainty
            # The observation says "in state s at time t" but we don't know which phase
            # Use censoring pattern to allow all phases of state s
            if s > length(mappings.state_to_phases)
                error("State $s found in data but not in model (max state: $(length(mappings.state_to_phases))).")
            end
            n_phases_in_state = length(mappings.state_to_phases[s])
            if n_phases_in_state > 1
                # Multiple phases: use censoring pattern (obstype = s + 2)
                # This triggers build_emat to use censoring_patterns[s, :]
                new_obstype[i] = s + 2
                new_stateto[i] = 0  # Triggers forward algorithm with emission matrix
            else
                # Single phase: no uncertainty, map directly
                new_stateto[i] = first(mappings.state_to_phases[s])
            end
        elseif obstype_i == 1
            # Exact observation (instantaneous transition): map to first phase
            # The exact destination phase will be determined by hazard ratios in loglik_markov
            # Check if s is a valid state index (could be > n_observed if using custom states, but here s is from data)
            if s <= length(mappings.state_to_phases)
                new_stateto[i] = first(mappings.state_to_phases[s])
            else
                # This might happen if data contains states not in the model definition
                # But validation should have caught this.
                # For now, assume it's a valid state or handle gracefully
                new_stateto[i] = s
            end
        else
            # Other observation types: keep stateto as is (already handled)
            # Check if s is a valid state index
            if s > 0 && s <= length(mappings.state_to_phases)
                new_stateto[i] = first(mappings.state_to_phases[s])
            else
                new_stateto[i] = s
            end
        end
    end
    
    expanded_data.stateto = new_stateto
    expanded_data.obstype = new_obstype
    
    return expanded_data, censoring_patterns, original_row_map
end

"""
    _build_phase_censoring_patterns(mappings) -> Matrix{Float64}

Build censoring patterns that encode phase uncertainty.

For each observed state, creates a pattern where all phases of that state
have probability 1 (possible) and all other states have probability 0 (impossible).

The patterns are indexed by observed state (rows) and expanded state (columns).

Note: The format must match what `build_emat` expects:
- Row 1 corresponds to obstype=3 (sojourn in state 1)
- Row 2 corresponds to obstype=4 (sojourn in state 2)
- etc.
- Column 1 is the pattern code (not used by build_emat for indexing)
- Columns 2:n_expanded+1 are the state indicators
"""
function _build_phase_censoring_patterns(mappings::PhaseTypeMappings)
    n_observed = mappings.n_observed
    n_expanded = mappings.n_expanded
    
    # Pattern matrix format for build_emat compatibility:
    # - Row s corresponds to obstype = s + 2 (sojourn in observed state s)
    # - Column 1 is the pattern code (s + 2.0)
    # - Columns 2:n_expanded+1 are indicators for which phases are allowed
    patterns = zeros(Float64, n_observed, n_expanded + 1)
    
    for s in 1:n_observed
        patterns[s, 1] = s + 2.0  # Pattern code (for reference/debugging)
        phases = mappings.state_to_phases[s]
        for p in phases
            patterns[s, p + 1] = 1.0  # Phase p is allowed (column index = p + 1)
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
# Phase 3.5: SCTP Constraint Generation
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

# =============================================================================
# Phase 4: Model Building for Phase-Type Hazards
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
        if h isa PhaseTypeHazardSpec
            push!(pt_states, h.statefrom)
        end
    end
    
    # Build n_phases_per_state from model-level n_phases dict
    n_phases_per_state = ones(Int, n_states)  # Default: 1 phase for non-pt states
    
    if isnothing(n_phases) || isempty(n_phases)
        # Legacy path: use n_phases from PhaseTypeHazardSpec
        for h in hazards_ordered
            if h isa PhaseTypeHazardSpec
                s = h.statefrom
                n_phases_per_state[s] = max(n_phases_per_state[s], h.n_phases)
            end
        end
    else
        # New path: use model-level n_phases dict
        # Validate: all pt states must be in dict
        for s in pt_states
            if !haskey(n_phases, s)
                error("State $s has :pt hazards but is not in n_phases dict. " *
                      "Specify n_phases = Dict($s => k) where k is the number of phases.")
            end
        end
        
        # Validate: dict shouldn't have states without :pt hazards
        for (s, k) in n_phases
            if s ∉ pt_states
                error("n_phases specifies $k phases for state $s, but state $s has no :pt hazards.")
            end
            if k < 1
                error("n_phases[$s] must be ≥ 1, got $k")
            end
            # Coerce n_phases = 1 to exponential (no warning, just do it)
            n_phases_per_state[s] = k
        end
    end
    
    # Step 1c: Check for mixed hazard types from states with :pt hazards
    for s in pt_states
        outgoing_hazards = filter(h -> h.statefrom == s, hazards_ordered)
        non_pt_hazards = filter(h -> !(h isa PhaseTypeHazardSpec), outgoing_hazards)
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
        
        if h isa PhaseTypeHazardSpec
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

# =============================================================================
# Phase-Type Log-Likelihood Functions
# =============================================================================
# These functions compute likelihoods for phase-type surrogates used in MCEM.
# They use the forward algorithm on the expanded state space with proper
# emission matrix handling for censoring.
# =============================================================================

"""
    compute_phasetype_marginal_loglik(model, surrogate, emat_ph; kwargs...)

Compute the marginal log-likelihood of observed data under the phase-type surrogate.

This is used as the normalizing constant r(Y|θ') in importance sampling:
    log f̂(Y|θ) = log r(Y|θ') + Σᵢ log(mean(νᵢ))

Uses the forward algorithm on the expanded phase state space.

# Arguments
- `model::MultistateProcess`: The multistate model containing the data
- `surrogate::PhaseTypeSurrogate`: The fitted phase-type surrogate
- `emat_ph::Matrix{Float64}`: Expanded emission matrix for phase states

# Keyword Arguments
- `expanded_data`: Optional expanded data for exact observations
- `expanded_subjectindices`: Subject indices for expanded data

# Returns
- `Float64`: The marginal log-likelihood under the phase-type surrogate
"""
function compute_phasetype_marginal_loglik(model::MultistateProcess, 
                                           surrogate::PhaseTypeSurrogate,
                                           emat_ph::Matrix{Float64};
                                           expanded_data::Union{Nothing, DataFrame} = nothing,
                                           expanded_subjectindices::Union{Nothing, Vector{UnitRange{Int64}}} = nothing)
    
    Q = surrogate.expanded_Q
    n_expanded = surrogate.n_expanded_states
    
    data = isnothing(expanded_data) ? model.data : expanded_data
    subjectindices = isnothing(expanded_subjectindices) ? model.subjectindices : expanded_subjectindices
    
    cache = ExponentialUtilities.alloc_mem(similar(Q), ExpMethodGeneric())
    
    ll_total = 0.0
    
    for subj_idx in eachindex(subjectindices)
        subj_inds = subjectindices[subj_idx]
        n_obs = length(subj_inds)
        
        times = vcat(data.tstart[subj_inds[1]], data.tstop[subj_inds])
        statefrom_subj = data.statefrom[subj_inds]
        stateto_subj = data.stateto[subj_inds]
        obstype_subj = data.obstype[subj_inds]
        subj_emat = emat_ph[subj_inds, :]
        
        # Initialize forward variable
        α = zeros(Float64, n_expanded)
        initial_state = statefrom_subj[1]
        initial_phase = first(surrogate.state_to_phases[initial_state])
        α[initial_phase] = 1.0
        
        log_ll = 0.0
        α_new = zeros(Float64, n_expanded)
        P = similar(Q)
        Q_scaled = similar(Q)
        
        for k in 1:n_obs
            Δt = times[k + 1] - times[k]
            
            if Δt > 0
                copyto!(Q_scaled, Q)
                Q_scaled .*= Δt
                copyto!(P, exponential!(Q_scaled, ExpMethodGeneric(), cache))
                
                fill!(α_new, 0.0)
                
                if obstype_subj[k] ∈ [1, 2] && stateto_subj[k] > 0
                    obs_state = stateto_subj[k]
                    allowed_phases = surrogate.state_to_phases[obs_state]
                    for j in allowed_phases
                        for i in 1:n_expanded
                            α_new[j] += P[i, j] * α[i]
                        end
                    end
                else
                    for j in 1:n_expanded
                        e_j = subj_emat[k, j]
                        if e_j > 0
                            for i in 1:n_expanded
                                α_new[j] += P[i, j] * α[i] * e_j
                            end
                        end
                    end
                end
                
                copyto!(α, α_new)
            else
                if obstype_subj[k] ∈ [1, 2] && stateto_subj[k] > 0
                    obs_state = stateto_subj[k]
                    for j in 1:n_expanded
                        if surrogate.phase_to_state[j] != obs_state
                            α[j] = 0.0
                        end
                    end
                else
                    for j in 1:n_expanded
                        α[j] *= subj_emat[k, j]
                    end
                end
            end
            
            scale = sum(α)
            if scale > 0
                log_ll += log(scale)
                α ./= scale
            else
                log_ll = -Inf
                break
            end
        end
        
        log_ll += log(sum(α))
        ll_total += log_ll * model.SubjectWeights[subj_idx]
    end
    
    return ll_total
end

# =============================================================================
# NOTE: Longtest Infrastructure Moved (Package Streamlining)
# =============================================================================
# The following functions were moved to MultistateModelsTests/longtests/phasetype_longtest_helpers.jl:
#   - build_phasetype_hazards(tmat, config, surrogate; ...)
#   - build_expanded_tmat(tmat, surrogate)
#   - build_phasetype_emat(data, surrogate, CensoringPatterns)
#   - expand_data_states!(data, surrogate)
#   - build_phasetype_model(tmat, config; data, ...)
#
# These provide an alternative API for building phase-type models directly
# from tmat + PhaseTypeConfig. The production API uses Hazard(:pt, ...) + multistatemodel().
# =============================================================================

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
    expand_data_for_phasetype(data::DataFrame, n_states::Int)

Expand data for phase-type forward-backward sampling.

Splits exact observations (obstype=1) into:
1. Sojourn interval [tstart, tstop) with censored state
2. Instantaneous exact observation at tstop

# Arguments
- `data::DataFrame`: Original data with id, tstart, tstop, statefrom, stateto, obstype
- `n_states::Int`: Number of observed states

# Returns
NamedTuple with:
- `expanded_data::DataFrame`: Data with exact obs expanded
- `censoring_patterns::Matrix{Float64}`: Patterns for phase uncertainty
- `original_row_map::Vector{Int}`: Maps expanded → original row indices

Censoring patterns: obstype = 2 + s indicates "subject in state s, phase unknown".

See also: [`build_phasetype_emat`](@ref)
"""
function expand_data_for_phasetype(data::DataFrame, n_states::Int)
    
    # Count how many rows we'll need
    # Only split exact observations where statefrom > 0 and tstart < tstop
    n_original = nrow(data)
    n_to_split = count(i -> data.obstype[i] == 1 && data.statefrom[i] > 0 && data.tstart[i] < data.tstop[i], 
                       1:n_original)
    n_expanded = n_original + n_to_split  # Each split obs becomes 2 rows
    
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
        
        # Should this exact observation be split?
        # Split if: obstype=1, statefrom > 0 (known source state), and tstart < tstop (duration > 0)
        # Don't split if: statefrom=0 (unknown source), or tstart==tstop (already instantaneous)
        should_split = row.obstype == 1 && row.statefrom > 0 && row.tstart < row.tstop
        
        if should_split
            # Exact observation: split into sojourn + instantaneous observation
            
            # Row 1: Sojourn interval [tstart, tstop)
            # Subject is in statefrom, phase unknown
            # Use censoring code = 2 + statefrom
            exp_idx += 1
            exp_id[exp_idx] = row.id
            exp_tstart[exp_idx] = row.tstart
            exp_tstop[exp_idx] = row.tstop
            exp_statefrom[exp_idx] = row.statefrom
            exp_stateto[exp_idx] = 0  # Censored (state unknown at this point)
            exp_obstype[exp_idx] = 2 + row.statefrom  # Censoring pattern for statefrom
            original_row_map[exp_idx] = orig_idx
            
            # Copy covariates
            for col in covar_cols
                covar_arrays[col][exp_idx] = row[col]
            end
            
            # Row 2: Instantaneous exact observation at tstop
            # Transition to stateto observed (dt = 0)
            exp_idx += 1
            exp_id[exp_idx] = row.id
            exp_tstart[exp_idx] = row.tstop
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

# =============================================================================
# NOTE: Phase-Type Model Methods Removed (Package Streamlining)
# =============================================================================
# The following PhaseTypeModel-specific methods have been removed:
#   - set_crude_init!(::PhaseTypeModel, ...)
#   - initialize_parameters[!](::PhaseTypeModel, ...)
#   - get_parameters[_flat/_nested/_natural](::PhaseTypeModel, ...)
#   - get_expanded_parameters(::PhaseTypeModel, ...)
#   - set_parameters!(::PhaseTypeModel, ...)
#   - get_unflatten_fn(::PhaseTypeModel)
#   - And related internal helpers
#
# Phase-type hazards are now handled internally via MultistateModel with
# phasetype_expansion metadata. Standard parameter methods dispatch on
# MultistateModel with trait `has_phasetype_expansion(m)`.
# =============================================================================
