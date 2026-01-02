# =============================================================================
# Phase-Type Data Expansion for Forward-Backward Sampling
# =============================================================================
#
# Data expansion utilities for phase-type forward-backward sampling (FFBS).
#
# Contents:
# - expand_data_for_phasetype: Expand data for FFBS (split exact observations)
# - needs_data_expansion_for_phasetype: Check if data needs expansion
# - compute_expanded_subject_indices: Compute subject indices in expanded data
#
# NOTE: Longtest Infrastructure Moved (Package Streamlining)
# The following functions were moved to MultistateModelsTests/longtests/phasetype_longtest_helpers.jl:
#   - build_phasetype_hazards(tmat, config, surrogate; ...)
#   - build_expanded_tmat(tmat, surrogate)
#   - build_phasetype_emat(data, surrogate, CensoringPatterns)
#   - expand_data_states!(data, surrogate)
#   - build_phasetype_model(tmat, config; data, ...)
#
# These provide an alternative API for building phase-type models directly
# from tmat + PhaseTypeConfig. The production API uses Hazard(:pt, ...) + multistatemodel().
#
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
