# =============================================================================
# Phase-Type Hazard Expansion
# =============================================================================
#
# Convert user hazard specifications to runtime hazards on the expanded
# phase-type state space.
#
# Contents:
# - expand_hazards_for_phasetype: Main hazard expansion function
# - _build_progression_hazard: Build λ (phase progression) hazards
# - _build_exit_hazard: Build μ (exit) hazards
# - _generate_exit_hazard_fns: Generate hazard function closures
# - _build_expanded_hazard: Expand standard hazards to phase space
# - _adjust_hazard_states: Adjust hazard state indices
# - _build_shared_phase_hazard: Build shared-rate phase hazards
# - expand_emission_matrix: Expand emission matrix for phase states
# - expand_data_for_phasetype_fitting: Expand data for phase-type fitting
# - _build_phase_censoring_patterns: Build censoring patterns for phases
# - map_observation_to_phases: Map observation types to phase states
#
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
        has_pt = any(h -> h isa PhaseTypeHazard && h.statefrom == s, hazards)
        
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
            
            if h isa PhaseTypeHazard
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
        nothing,           # shared_baseline_key
        SmoothTermInfo[]   # smooth_info (progression hazards have no covariates)
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
function _build_exit_hazard(pt_spec::PhaseTypeHazard, 
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
    
    # TODO: Extract smooth_info from pt_spec if exit hazards support s(x) in future
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
        nothing,           # Each exit hazard is independent
        SmoothTermInfo[]   # smooth_info (exit hazards don't yet support smooth terms)
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
        data,
        hazschema
    )
    
    # Get the builder and create the hazard
    builder = get(_HAZARD_BUILDERS, family, nothing)
    if builder === nothing
        throw(ArgumentError("Unknown hazard family for phase-type expansion: $(family). Supported: $(keys(_HAZARD_BUILDERS))"))
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
        haz.covar_names, haz.metadata, haz.shared_baseline_key,
        haz.smooth_info
    )
end

function _adjust_hazard_states(haz::SemiMarkovHazard, new_statefrom::Int, new_stateto::Int)
    return SemiMarkovHazard(
        haz.hazname, new_statefrom, new_stateto, haz.family,
        haz.parnames, haz.npar_baseline, haz.npar_total,
        haz.hazard_fn, haz.cumhaz_fn, haz.has_covariates,
        haz.covar_names, haz.metadata, haz.shared_baseline_key,
        haz.smooth_info
    )
end

function _adjust_hazard_states(haz::RuntimeSplineHazard, new_statefrom::Int, new_stateto::Int)
    return RuntimeSplineHazard(
        haz.hazname, new_statefrom, new_stateto, haz.family,
        haz.parnames, haz.npar_baseline, haz.npar_total,
        haz.hazard_fn, haz.cumhaz_fn, haz.has_covariates,
        haz.covar_names, haz.degree, haz.knots, haz.natural_spline,
        haz.monotone, haz.extrapolation, haz.metadata, haz.shared_baseline_key,
        haz.smooth_info
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
        throw(ArgumentError("EmissionMatrix must have $n_observed columns (one per observed state), got $(size(EmissionMatrix, 2))."))
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
                throw(ArgumentError("State $s found in data but not in model (max state: $(length(mappings.state_to_phases)))."))
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
    # - Columns 2:n_expanded+1 are indicator values (1.0 = possible, 0.0 = impossible)
    patterns = zeros(Float64, n_observed, n_expanded + 1)
    
    for s in 1:n_observed
        patterns[s, 1] = s + 2.0  # Pattern code (for reference/debugging)
        phases = mappings.state_to_phases[s]
        # Use indicator values (1.0 for all possible phases)
        # The likelihood computation sums transition probabilities for all allowed phases:
        #   L = Σ_s P(transition to s) * indicator(s is allowed)
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

