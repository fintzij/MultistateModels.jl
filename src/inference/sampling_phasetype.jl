# ============================================================================
# Path Sampling Infrastructure - Phase-Type FFBS and Likelihood
# ============================================================================
#
# This file contains phase-type extensions to the Markov FFBS machinery for
# improved MCEM importance sampling with non-exponential sojourn times:
#
# - convert_*_to_censored_data: Convert sampled paths to forward algorithm format
# - compute_forward_loglik: Marginal likelihood via forward algorithm
# - loglik_phasetype_forward: Evaluate surrogate likelihood for importance weights
# - build_phasetype_tpm_book: Build TPM books for expanded state space
# - build_phasetype_emat_expanded: Emission matrix for phase-type FFBS
# - draw_samplepath_phasetype: Sample paths from phase-type surrogate
# - collapse_phasetype_path: Map expanded paths back to observed states
# - loglik_phasetype_expanded_path: CTMC path density on expanded space
#
# Key insight: Each observed state is expanded into multiple phases. The expanded
# Markov chain better approximates non-exponential sojourn times. After sampling
# in the expanded space, paths are collapsed back to observed states.
#
# Dependencies:
# - sampling_core.jl: PathWorkspace and thread-local storage
# - sampling_markov.jl: ForwardFiltering!, BackwardSampling!, sample_ecctmc!
# ============================================================================

# ============================================================================
# Phase-Type Path Likelihood via Forward Algorithm
# ============================================================================
# These functions compute the marginal likelihood of a sampled path under the
# phase-type surrogate using the forward algorithm. This is the CORRECT way to 
# compute surrogate log-likelihood for importance sampling when the original 
# observations are panel (interval-censored) data.
#
# The key insight: A sampled path Z = (τ₁→s₁, τ₂→s₂, ...) "fills in" the missing
# transition times between panel observations, but the surrogate likelihood should
# STILL respect the original observation structure. We need:
#
#   q(Z | Y, θ') = p(Z | Y, θ')   [proposal density]
#
# NOT the unconditional path density p(Z | θ'), which ignores observation constraints.
#
# The marginal likelihood over phase sequences is:
#   q(Z_collapsed | θ') = Σ_{Z_expanded: collapse(Z_expanded) = Z_collapsed} q(Z_expanded | θ')
#
# This is computed via forward algorithm on the expanded state space, constrained
# ONLY at the original observation times (not at path transition times).
# ============================================================================

"""
    convert_collapsed_path_to_censored_data(collapsed_path, original_subj_data, model)

Convert a sampled collapsed path to censored data format for forward algorithm evaluation.

CRITICAL: We only create intervals at ORIGINAL observation times, not at path transition
times. The forward algorithm should only be constrained at times where we actually
observed the subject. Path transition times are internal to the sampled path and 
should not create additional observation constraints.

# Arguments
- `collapsed_path::SamplePath`: Sampled path in collapsed (observed) state space
- `original_subj_data::SubDataFrame`: Original observation data for this subject
- `model::MultistateProcess`: The multistate model (for covariate interpolation)

# Returns
- `DataFrame`: Censored data with columns (tstart, tstop, statefrom, stateto, obstype, ...)
  All observations are panel (obstype=2) to allow the forward algorithm to marginalize
  over phase sequences.

# Algorithm
1. Use ONLY original observation times (tstart, tstop from original data)
2. Look up what state the sampled path was in at each observation time
3. Create panel observations (obstype=2) - this allows marginalization over phases

# Note
This computes the surrogate log-likelihood q(Z|Y,θ') correctly by:
- Respecting the original observation structure
- Marginalizing over all phase sequences compatible with the collapsed path
"""
function convert_collapsed_path_to_censored_data(
    collapsed_path::SamplePath,
    original_subj_data::SubDataFrame,
    model::MultistateProcess
)
    # Get path times and states for lookup
    path_times = collapsed_path.times
    path_states = collapsed_path.states
    
    n_obs = nrow(original_subj_data)
    
    # Pre-allocate output vectors - same size as original data
    tstart_out = Vector{Float64}(undef, n_obs)
    tstop_out = Vector{Float64}(undef, n_obs)
    statefrom_out = Vector{Int}(undef, n_obs)
    stateto_out = Vector{Int}(undef, n_obs)
    obstype_out = Vector{Int}(undef, n_obs)
    
    for i in 1:n_obs
        t0 = original_subj_data.tstart[i]
        t1 = original_subj_data.tstop[i]
        
        # Find state at t0 from collapsed path
        path_idx_t0 = searchsortedlast(path_times, t0)
        state_at_t0 = path_states[max(1, path_idx_t0)]
        
        # Find state at t1 from collapsed path  
        path_idx_t1 = searchsortedlast(path_times, t1)
        state_at_t1 = path_states[max(1, path_idx_t1)]
        
        tstart_out[i] = t0
        tstop_out[i] = t1
        statefrom_out[i] = state_at_t0
        stateto_out[i] = state_at_t1
        
        # ALL observations are panel (obstype=2) to allow forward algorithm to
        # marginalize over phase sequences. Even if the original observation was
        # exact (obstype=1), we treat it as panel here because we're computing
        # the marginal likelihood over phases, not conditioning on a specific phase.
        obstype_out[i] = 2
    end
    
    # Create DataFrame
    censored_data = DataFrame(
        tstart = tstart_out,
        tstop = tstop_out,
        statefrom = statefrom_out,
        stateto = stateto_out,
        obstype = obstype_out
    )
    
    # Handle covariates if present - just copy from original data since rows match 1:1
    if ncol(original_subj_data) > 6
        covar_cols = names(original_subj_data)[7:end]
        for col in covar_cols
            censored_data[!, col] = copy(original_subj_data[!, col])
        end
    end
    
    return censored_data
end


"""
    convert_expanded_path_to_censored_data(expanded_path, surrogate; ...)

Convert an expanded phase-space path to censored data for surrogate likelihood computation.

This function creates the censored data structure that allows the forward algorithm to
marginalize over phase paths while respecting the exact transition times from the sampled path.

# Why Compute TPMs On-The-Fly?

Unlike the Markov surrogate (which uses exact hazard evaluations for path likelihood),
the phase-type surrogate requires marginalizing over phase uncertainty within macro-states.
This requires TPMs at the exact sampled transition times.

The pre-computed `tpm_book_ph` contains TPMs for the *observation* time intervals, but
sampled paths have *different* time intervals (the exact simulated jump times from ECCTMC).
Therefore, we compute path-specific TPMs using the covariate-adjusted Q matrix.

This is consistent with the Markov case where path likelihood is computed from exact
hazard evaluations, not from pre-computed TPMs.

# Algorithm

For each macro-state sojourn in the expanded path:
1. Create a SURVIVAL row from sojourn start to exit time
   - statefrom = first phase of macro-state (Coxian entry)
   - stateto = 0 (unknown - could be any phase of that macro-state)
   - obstype = censoring pattern index (allows any phase of the macro-state)

2. Create an EXACT TRANSITION row at the exit time
   - statefrom = 0 (unknown which phase we exited from)
   - stateto = destination phase (exactly known from sampled path)
   - obstype = 1 (exact observation)

# Example

For expanded path with phases {1,2} ∈ macro-state 1, {3} ∈ macro-state 2:
```
times_expanded = [0.0, 0.5, 1.2]
states_expanded = [1, 2, 3]
```

Creates:
```
tstart = [0.0, 1.2]
tstop = [1.2, 1.2]
statefrom = [1, 0]      # Start in phase 1, exit from unknown phase
stateto = [0, 3]        # End in unknown phase, transition to phase 3
obstype = [3, 1]        # Censoring pattern (obstype-CENSORING_OBSTYPE_OFFSET=1), exact
```

Where censoring pattern 1 = [1, 1, 0] allows phases 1 or 2 at time 1.2.

# Arguments
- `expanded_path::SamplePath`: Path in expanded phase space (states are phase indices)
- `surrogate::PhaseTypeSurrogate`: The phase-type surrogate (for phase→macro-state mapping)
- `original_subj_data::SubDataFrame`: Original subject data with covariates (for TVC interpolation)
- `hazmat_book::Vector{Matrix{Float64}}`: Covariate-indexed Q matrices from `hazmat_book_ph`
- `schur_cache_book::Vector{CachedSchurDecomposition}`: Covariate-indexed Schur caches
- `subj_tpm_map::SubArray`: Subject's tpm_map from books[2] (for covariate index lookup)

For subjects with time-varying covariates (TVC), each interval may have a different 
covariate combination. The function interpolates covariates at transition times using
the original subject data and builds TPMs using the appropriate Q matrix for each interval.

# Legacy signature (single covariate - backward compatible)
For subjects without TVC or when testing, you can pass:
- `hazmat::Matrix{Float64}`: Single covariate-adjusted Q matrix
- `schur_cache::CachedSchurDecomposition`: Single Schur cache

# Returns
- `censored_data::DataFrame`: Censored data with (tstart, tstop, statefrom, stateto, obstype)
- `emat::Matrix{Float64}`: Emission matrix for the forward algorithm
- `tpm_map::Matrix{Int}`: Mapping from rows to (covariate_idx, time_idx)
- `tpm_book::Vector{Vector{Matrix{Float64}}}`: TPMs indexed by [covar_idx][time_idx]
- `hazmat_book_out::Vector{Matrix{Float64}}`: Q matrices for instantaneous observations
"""
function convert_expanded_path_to_censored_data(
    expanded_path::SamplePath,
    surrogate::PhaseTypeSurrogate;
    # New TVC-aware signature
    original_subj_data::Union{Nothing, SubDataFrame} = nothing,
    hazmat_book::Union{Nothing, Vector{Matrix{Float64}}} = nothing,
    schur_cache_book::Union{Nothing, Vector{CachedSchurDecomposition}} = nothing,
    subj_tpm_map::Union{Nothing, SubArray} = nothing,
    # Legacy single-covariate signature (backward compatible)
    hazmat::Union{Nothing, Matrix{Float64}} = nothing,
    schur_cache::Union{Nothing, CachedSchurDecomposition} = nothing
)
    times = expanded_path.times
    phases = expanded_path.states
    n_expanded = surrogate.n_expanded_states
    n_observed = surrogate.n_observed_states
    
    # Determine if using TVC-aware mode or legacy single-covariate mode
    use_tvc = !isnothing(hazmat_book) && !isnothing(original_subj_data) && !isnothing(subj_tpm_map)
    
    # For legacy mode, use single Q matrix
    if !use_tvc
        Q = isnothing(hazmat) ? surrogate.expanded_Q : hazmat
    end
    
    # Map each phase to its macro-state
    macro_states = [surrogate.phase_to_state[p] for p in phases]
    
    # Find indices where macro-state changes (these are the transition points)
    transition_indices = Int[]
    for i in 1:(length(macro_states) - 1)
        if macro_states[i] != macro_states[i + 1]
            push!(transition_indices, i)
        end
    end
    
    n_transitions = length(transition_indices)
    
    # Build censoring patterns: one pattern per macro-state
    # Pattern k allows all phases of macro-state k
    censoring_patterns = zeros(Int, n_observed, n_expanded)
    for s in 1:n_observed
        for p in surrogate.state_to_phases[s]
            censoring_patterns[s, p] = 1
        end
    end
    
    # Helper function to get covariate index at a given time
    # Uses piecewise constant interpolation: find the original data row where tstart ≤ t < tstop
    function _get_covar_idx_at_time(t::Float64)
        if !use_tvc
            return 1  # Single covariate level
        end
        # Find row in original subject data where tstart ≤ t < tstop
        for row_idx in 1:nrow(original_subj_data)
            tstart = original_subj_data.tstart[row_idx]
            tstop = original_subj_data.tstop[row_idx]
            if tstart <= t < tstop
                return subj_tpm_map[row_idx, 1]  # Covariate combo index from books[2]
            end
        end
        # Edge case: t equals final tstop, use last interval
        return subj_tpm_map[nrow(original_subj_data), 1]
    end
    
    # If no macro-state transitions, just survival in initial state
    if n_transitions == 0
        final_time = times[end]
        initial_phase = phases[1]
        initial_macro = macro_states[1]
        Δt = final_time - times[1]
        
        censored_data = DataFrame(
            tstart = [times[1]],
            tstop = [final_time],
            statefrom = [initial_phase],
            stateto = [0],
            obstype = [initial_macro + 2]  # Censoring pattern for initial_macro
        )
        
        # Build emission matrix
        emat = zeros(Float64, 1, n_expanded)
        for p in surrogate.state_to_phases[initial_macro]
            emat[1, p] = 1.0
        end
        
        # Get covariate index at initial time
        covar_idx = _get_covar_idx_at_time(times[1])
        
        # Compute TPM for this interval
        if use_tvc
            Q_interval = hazmat_book[covar_idx]
            if !isnothing(schur_cache_book)
                P = compute_tpm_from_schur(schur_cache_book[covar_idx], Δt)
            else
                P = exp(Q_interval * Δt)
            end
            # Return all Q matrices used (just this one)
            hazmat_book_out = [Q_interval]
            tpm_book = [[P]]
            tpm_map = reshape([1, 1], 1, 2)
        else
            if !isnothing(schur_cache)
                P = compute_tpm_from_schur(schur_cache, Δt)
            else
                P = exp(Q * Δt)
            end
            hazmat_book_out = [Q]
            tpm_book = [[P]]
            tpm_map = reshape([1, 1], 1, 2)
        end
        
        return censored_data, emat, tpm_map, tpm_book, hazmat_book_out
    end
    
    # Build rows dynamically - we may skip survival rows when Δt = 0
    # Preallocate for worst case (2 rows per transition)
    tstart_out = Float64[]
    tstop_out = Float64[]
    statefrom_out = Int[]
    stateto_out = Int[]
    obstype_out = Int[]
    emat_rows = Vector{Vector{Float64}}()
    
    # Track (covar_idx, Δt) pairs for TPM computation
    # Key: (covar_idx, Δt), Value: time_idx within that covariate level
    covar_dt_to_idx = Dict{Tuple{Int, Float64}, Tuple{Int, Int}}()
    tpm_map_list = Vector{Tuple{Int, Int}}()
    covar_indices_used = Set{Int}()
    
    for (k, trans_idx) in enumerate(transition_indices)
        # Sojourn start time
        if k == 1
            t_start = times[1]
            entry_phase = phases[1]
            is_first_sojourn = true
        else
            prev_trans_idx = transition_indices[k - 1]
            t_start = times[prev_trans_idx + 1]
            entry_phase = phases[prev_trans_idx + 1]
            is_first_sojourn = false
        end
        
        # Transition time and destination
        t_trans = times[trans_idx + 1]
        s_macro = macro_states[trans_idx]
        d_phase = phases[trans_idx + 1]
        
        Δt = t_trans - t_start
        
        # Get covariate index at interval start time
        covar_idx = _get_covar_idx_at_time(t_start)
        push!(covar_indices_used, covar_idx)
        
        # Only create survival row if Δt > 0
        # When Δt = 0 (immediate transition), skip the survival row
        if Δt > 0
            # Row: Survival in source macro-state
            push!(tstart_out, t_start)
            push!(tstop_out, t_trans)
            #
            # CRITICAL FOR MARGINALIZATION:
            # - First sojourn: initialize α to the actual starting phase
            # - Subsequent sojourns: do NOT re-initialize α; use distribution from 
            #   previous transition row (which marginalizes over entry phases)
            push!(statefrom_out, is_first_sojourn ? entry_phase : 0)
            push!(stateto_out, 0)
            push!(obstype_out, s_macro + 2)  # Censoring pattern
            
            erow = zeros(Float64, n_expanded)
            for p in surrogate.state_to_phases[s_macro]
                erow[p] = 1.0
            end
            push!(emat_rows, erow)
            
            # Track (covar_idx, Δt) for TPM computation
            key = (covar_idx, Δt)
            if !haskey(covar_dt_to_idx, key)
                # Assign new index: (covar_idx, time_idx_within_covar)
                # For simplicity, use covar_idx directly as first index
                n_times_this_covar = count(k -> k[1] == covar_idx, keys(covar_dt_to_idx))
                covar_dt_to_idx[key] = (covar_idx, n_times_this_covar + 1)
            end
            push!(tpm_map_list, covar_dt_to_idx[key])
        end
        
        # Row: Exact transition (Δt = 0)
        # This row captures the instantaneous transition to the destination macro-state.
        #
        # CRITICAL FOR IMPORTANCE SAMPLING:
        # The collapsed path only observes macro-state transitions, not specific phases.
        # For proper marginalization, the emission matrix must allow ANY phase of the
        # destination macro-state, not just the specific phase that was sampled.
        # This ensures q(Z_collapsed) = P(macro-state sequence), marginalizing over phases.
        push!(tstart_out, t_trans)
        push!(tstop_out, t_trans)
        # When we skip survival row, we need statefrom to be entry_phase for initialization
        push!(statefrom_out, Δt > 0 ? 0 : entry_phase)
        push!(stateto_out, d_phase)  # Store actual destination phase for tracking
        push!(obstype_out, 1)
        
        # Get destination macro-state from the destination phase
        d_macro = surrogate.phase_to_state[d_phase]
        
        # Emission allows ANY phase of the destination macro-state
        erow = zeros(Float64, n_expanded)
        for p in surrogate.state_to_phases[d_macro]
            erow[p] = 1.0
        end
        push!(emat_rows, erow)
        
        # For Δt=0, we use instantaneous Q matrix
        key_zero = (covar_idx, 0.0)
        if !haskey(covar_dt_to_idx, key_zero)
            n_times_this_covar = count(k -> k[1] == covar_idx, keys(covar_dt_to_idx))
            covar_dt_to_idx[key_zero] = (covar_idx, n_times_this_covar + 1)
        end
        push!(tpm_map_list, covar_dt_to_idx[key_zero])
    end
    
    # =========================================================================
    # CRITICAL: Handle final sojourn after last macro-state transition
    # =========================================================================
    # The loop above only handles intervals UP TO and INCLUDING each transition.
    # If the path continues in the final state after the last transition, we need
    # to add a survival row for that final sojourn.
    #
    # Example: times = [0, 2, 5], phases = [1a, 2a, 2a]
    #          macro_states = [1, 2, 2]
    #          transition_indices = [1] (1→2 at index 1)
    #
    # The loop creates:
    #   - Survival row: 0 to 2, in state 1
    #   - Transition row: at t=2, to phase 2a
    #
    # But we're missing: survival in state 2 from t=2 to t=5!
    # =========================================================================
    final_time = times[end]
    if n_transitions > 0
        # Entry into final state is right after the last transition
        last_trans_idx = transition_indices[end]
        final_entry_time = times[last_trans_idx + 1]
        final_entry_phase = phases[last_trans_idx + 1]
        final_macro = macro_states[last_trans_idx + 1]
        
        Δt_final = final_time - final_entry_time
        
        if Δt_final > 0
            # Row: Final survival (after last transition)
            covar_idx_final = _get_covar_idx_at_time(final_entry_time)
            push!(covar_indices_used, covar_idx_final)
            
            push!(tstart_out, final_entry_time)
            push!(tstop_out, final_time)
            # Do NOT re-initialize α; use distribution from the last transition row
            # This ensures proper marginalization over entry phases
            push!(statefrom_out, 0)
            push!(stateto_out, 0)  # Censored/survival, not transitioning
            push!(obstype_out, final_macro + 2)  # Censoring pattern for final_macro
            
            erow = zeros(Float64, n_expanded)
            for p in surrogate.state_to_phases[final_macro]
                erow[p] = 1.0
            end
            push!(emat_rows, erow)
            
            # Track (covar_idx, Δt) for TPM computation
            key_final = (covar_idx_final, Δt_final)
            if !haskey(covar_dt_to_idx, key_final)
                n_times_this_covar = count(k -> k[1] == covar_idx_final, keys(covar_dt_to_idx))
                covar_dt_to_idx[key_final] = (covar_idx_final, n_times_this_covar + 1)
            end
            push!(tpm_map_list, covar_dt_to_idx[key_final])
        end
    end
    
    # Convert to matrices
    n_rows = length(tstart_out)
    emat = zeros(Float64, n_rows, n_expanded)
    for (i, erow) in enumerate(emat_rows)
        emat[i, :] .= erow
    end
    
    tpm_map = zeros(Int, n_rows, 2)
    for (i, (c, t)) in enumerate(tpm_map_list)
        tpm_map[i, 1] = c
        tpm_map[i, 2] = t
    end
    
    # Build tpm_book and hazmat_book_out indexed by covariate
    # Find max covariate index used
    max_covar = maximum(covar_indices_used)
    
    # Build tpm_book: [covar_idx][time_idx] = P matrix
    tpm_book = Vector{Vector{Matrix{Float64}}}(undef, max_covar)
    hazmat_book_out = Vector{Matrix{Float64}}(undef, max_covar)
    
    for c in 1:max_covar
        if c in covar_indices_used
            # Get Q matrix for this covariate level
            if use_tvc
                Q_c = hazmat_book[c]
                schur_c = isnothing(schur_cache_book) ? nothing : schur_cache_book[c]
            else
                Q_c = Q
                schur_c = schur_cache
            end
            hazmat_book_out[c] = Q_c
            
            # Find all unique Δt values for this covariate level
            dts_for_c = [key[2] for key in keys(covar_dt_to_idx) if key[1] == c]
            sort!(dts_for_c)
            
            # Compute TPMs for each unique Δt
            tpms_c = Vector{Matrix{Float64}}(undef, length(dts_for_c))
            for (idx, dt) in enumerate(dts_for_c)
                if dt > 0
                    if !isnothing(schur_c)
                        tpms_c[idx] = compute_tpm_from_schur(schur_c, dt)
                    else
                        tpms_c[idx] = exp(Q_c * dt)
                    end
                else
                    # Identity for dt=0 (compute_forward_loglik will use hazmat_book instead)
                    tpms_c[idx] = Matrix{Float64}(I, n_expanded, n_expanded)
                end
            end
            tpm_book[c] = tpms_c
            
            # Re-map time indices to be consecutive within each covariate level
            dt_to_time_idx = Dict(dt => idx for (idx, dt) in enumerate(dts_for_c))
            for i in 1:n_rows
                if tpm_map[i, 1] == c
                    old_key = (c, 0.0)  # Need to find the original Δt
                    # Find Δt from the data
                    interval_dt = tstop_out[i] - tstart_out[i]
                    tpm_map[i, 2] = dt_to_time_idx[interval_dt]
                end
            end
        else
            # Placeholder for unused covariate levels
            tpm_book[c] = Vector{Matrix{Float64}}()
            hazmat_book_out[c] = use_tvc ? hazmat_book[1] : Q
        end
    end
    
    censored_data = DataFrame(
        tstart = tstart_out,
        tstop = tstop_out,
        statefrom = statefrom_out,
        stateto = stateto_out,
        obstype = obstype_out
    )
    
    return censored_data, emat, tpm_map, tpm_book, hazmat_book_out
end


"""
    compute_forward_loglik(censored_data, emat, tpm_map, tpm_book, hazmat_book, n_states)

Compute log-likelihood using forward algorithm with pre-computed TPMs.

This is a lightweight wrapper that runs the forward algorithm on censored data
produced by `convert_expanded_path_to_censored_data`. It uses the existing
Markov forward algorithm infrastructure.

# Arguments
- `censored_data::DataFrame`: Censored data with (tstart, tstop, statefrom, stateto, obstype)
- `emat::Matrix{Float64}`: Emission matrix (n_obs × n_states)
- `tpm_map::Matrix{Int}`: Mapping from rows to (covariate_idx, time_idx)
- `tpm_book::Vector{Vector{Matrix{Float64}}}`: Pre-computed TPMs indexed by [covar_idx][time_idx]
- `hazmat_book::Vector{Matrix{Float64}}`: Q matrices for instantaneous transitions (one per covariate level)
- `n_states::Int`: Number of states in the expanded phase space

For backward compatibility, `hazmat_book` can also be a single AbstractMatrix (legacy single-covariate mode).

# Returns
- `Float64`: Log-likelihood of the censored data
"""
function compute_forward_loglik(
    censored_data::DataFrame,
    emat::Matrix{Float64},
    tpm_map::Matrix{Int},
    tpm_book::Vector{Vector{Matrix{Float64}}},
    hazmat_book::Union{AbstractMatrix, Vector{Matrix{Float64}}},
    n_states::Int
)
    n_obs = nrow(censored_data)
    if n_obs == 0
        return 0.0
    end
    
    # Handle legacy single-matrix mode
    use_single_Q = hazmat_book isa AbstractMatrix
    
    # Initialize forward variable
    initial_phase = censored_data.statefrom[1]
    if initial_phase > 0 && initial_phase <= n_states
        α = zeros(Float64, n_states)
        α[initial_phase] = 1.0
    else
        # Fallback: use emission constraint from first row
        α = copy(emat[1, :])
        s = sum(α)
        if s > 0
            α ./= s
        end
    end
    
    log_ll = 0.0
    α_new = zeros(Float64, n_states)
    
    for k in 1:n_obs
        # Check if we need to re-initialize α (happens when survival row was skipped)
        # This occurs when statefrom > 0 for a non-first row
        if k > 1 && censored_data.statefrom[k] > 0
            fill!(α, 0.0)
            α[censored_data.statefrom[k]] = 1.0
        end
        
        Δt = censored_data.tstop[k] - censored_data.tstart[k]
        
        # Get covariate index for this interval
        covar_idx = tpm_map[k, 1]
        
        # Forward propagation with emission constraint
        fill!(α_new, 0.0)
        
        if Δt > 0
            # dt > 0: Use transition probability matrix P = exp(Q·dt)
            P = tpm_book[covar_idx][tpm_map[k, 2]]
            for j in 1:n_states
                if emat[k, j] > 0
                    for i in 1:n_states
                        α_new[j] += P[i, j] * α[i]
                    end
                end
            end
        else
            # dt = 0: Instantaneous transition, use hazards directly
            # Get Q matrix for this covariate level
            Q = use_single_Q ? hazmat_book : hazmat_book[covar_idx]
            
            # Contribution: α_new[j] = Σᵢ Q[i,j] × α[i] × emission[j]
            # 
            # KEY: Q[i,j] is the UNNORMALIZED hazard rate, not probability.
            # We need the density contribution, not conditional probability.
            # The survival probability was already accounted for in the preceding row.
            # The emission mask (emat) already excludes self-transitions (j=i).
            for j in 1:n_states
                if emat[k, j] > 0
                    for i in 1:n_states
                        if i != j  # Only off-diagonal (actual transitions)
                            α_new[j] += Q[i, j] * α[i]
                        end
                    end
                end
            end
        end
        
        # Normalize and accumulate log-likelihood
        scale = sum(α_new)
        if scale > 0
            log_ll += log(scale)
            α_new ./= scale
            copyto!(α, α_new)
        else
            # This can happen if path starts in an impossible phase state
            # After fixing the Δt=0 survival row issue, this should be rare
            return -Inf
        end
    end
    
    return log_ll
end


"""
    loglik_phasetype_forward(censored_data, surrogate)

Compute log-likelihood of censored data under the phase-type surrogate using forward algorithm.

This function evaluates the marginal likelihood of a subject's censored data under
the phase-type surrogate. It correctly handles panel (interval-censored) data by
using the forward algorithm on the expanded phase state space.

# Arguments
- `censored_data::DataFrame`: Censored data with (tstart, tstop, statefrom, stateto, obstype)
- `surrogate::PhaseTypeSurrogate`: The fitted phase-type surrogate

# Returns
- `Float64`: Log-likelihood of the censored data under the phase-type surrogate

# Algorithm
Uses the forward algorithm:
1. Initialize α = (1, 0, ..., 0) at first phase of initial state
2. For each interval [t_k, t_{k+1}]:
   - Propagate: α' = exp(Q * Δt)' * α
   - Constrain: Zero out phases inconsistent with observation
   - Normalize and accumulate: loglik += log(sum(α')), α = α' / sum(α')
3. Return accumulated log-likelihood

# See also
- `convert_collapsed_path_to_censored_data`: Prepares data for this function
- `compute_phasetype_marginal_loglik`: Similar algorithm for full dataset
"""
function loglik_phasetype_forward(
    censored_data::DataFrame,
    surrogate::PhaseTypeSurrogate
)
    Q = surrogate.expanded_Q
    n_expanded = surrogate.n_expanded_states
    
    n_obs = nrow(censored_data)
    if n_obs == 0
        return 0.0
    end
    
    # Allocate workspace
    cache = ExponentialUtilities.alloc_mem(similar(Q), ExpMethodGeneric())
    α = zeros(Float64, n_expanded)
    α_new = zeros(Float64, n_expanded)
    P = similar(Q)
    Q_scaled = similar(Q)
    
    # Initialize forward variable at first phase of initial state
    initial_state = censored_data.statefrom[1]
    if initial_state > 0 && initial_state <= surrogate.n_observed_states
        initial_phase = first(surrogate.state_to_phases[initial_state])
        α[initial_phase] = 1.0
    else
        # Fallback: uniform over all phases
        α .= 1.0 / n_expanded
    end
    
    log_ll = 0.0
    
    for k in 1:n_obs
        Δt = censored_data.tstop[k] - censored_data.tstart[k]
        
        if Δt > 0
            # Compute transition probability matrix P = exp(Q * Δt)
            copyto!(Q_scaled, Q)
            Q_scaled .*= Δt
            copyto!(P, exponential!(Q_scaled, ExpMethodGeneric(), cache))
            
            # Forward propagation: α_new[j] = Σᵢ P[i,j] * α[i]
            fill!(α_new, 0.0)
            
            # Get observation constraints
            s_from = censored_data.statefrom[k]
            s_to = censored_data.stateto[k]
            obstype = censored_data.obstype[k]
            
            if obstype == 1 && s_to > 0 && s_to <= surrogate.n_observed_states
                # Exact observation: constrain to phases of destination state
                # For Coxian, transitions enter at phase 1, but here we're computing
                # the marginal probability of being in ANY phase of the observed state
                allowed_phases = surrogate.state_to_phases[s_to]
                for j in allowed_phases
                    for i in 1:n_expanded
                        α_new[j] += P[i, j] * α[i]
                    end
                end
            elseif obstype == 2 && s_to > 0 && s_to <= surrogate.n_observed_states
                # Panel observation: constrain to phases of observed state
                allowed_phases = surrogate.state_to_phases[s_to]
                for j in allowed_phases
                    for i in 1:n_expanded
                        α_new[j] += P[i, j] * α[i]
                    end
                end
            else
                # No constraint or unknown obstype: allow all phases
                for j in 1:n_expanded
                    for i in 1:n_expanded
                        α_new[j] += P[i, j] * α[i]
                    end
                end
            end
            
            copyto!(α, α_new)
            
        elseif Δt == 0
            # Instantaneous observation (Δt = 0): constrain to observed state/phase
            s_to = censored_data.stateto[k]
            if s_to > 0
                if s_to <= surrogate.n_observed_states
                    # s_to is a macro-state: zero out phases not in that state
                    for j in 1:n_expanded
                        if surrogate.phase_to_state[j] != s_to
                            α[j] = 0.0
                        end
                    end
                elseif s_to <= n_expanded
                    # s_to is a phase index: zero out all other phases
                    for j in 1:n_expanded
                        if j != s_to
                            α[j] = 0.0
                        end
                    end
                end
            end
        end
        
        # Normalize and accumulate log-likelihood
        scale = sum(α)
        if scale > 0
            log_ll += log(scale)
            α ./= scale
        else
            # Zero probability - impossible path under surrogate
            return -Inf
        end
    end
    
    return log_ll
end


# =============================================================================
# Phase-Type Surrogate Sampling Functions
# =============================================================================
#
# These functions implement forward-filtering backward-sampling (FFBS) on an
# expanded phase-type state space for improved MCEM importance sampling.
#
# Key idea: Each observed state is expanded into multiple phases. The expanded
# Markov chain better approximates non-exponential sojourn times. After sampling
# in the expanded space, paths are collapsed back to observed states.
# =============================================================================

"""
    build_phasetype_tpm_book(surrogate::PhaseTypeSurrogate, books, data)

Build transition probability matrix book for phase-type expanded state space.

This function correctly incorporates covariate effects into the phase-type expanded 
Q matrix. For each covariate combination, inter-state transition rates are scaled 
by exp(β'x) where β are the covariate coefficients stored in the surrogate.

**Key insight**: Internal phase progression rates (within-state transitions) are NOT
scaled by covariates. Only inter-state transition rates (absorption from phases)
are scaled, because covariate effects apply to the transition rate between observed
states, not to the phase progression dynamics within a state.

# Arguments
- `surrogate`: PhaseTypeSurrogate with expanded Q matrix, hazards, and fitted parameters
- `books`: Time interval book from build_tpm_mapping (books[1] = tpm_index per covariate combo)
- `data`: Model data (for extracting covariate values)

# Returns
- `tpm_book_ph`: Nested vector of TPMs [covar_combo][time_interval] in expanded space
- `hazmat_book_ph`: Vector of intensity matrices for each covariate combination
"""
function build_phasetype_tpm_book(surrogate::PhaseTypeSurrogate, books, data)
    n_expanded = surrogate.n_expanded_states
    Q_baseline = surrogate.expanded_Q
    phase_to_state = surrogate.phase_to_state
    
    # books[1] is a vector of DataFrames, one per covariate combination
    # Each DataFrame has columns (tstart, tstop, datind) with rows for unique time intervals
    n_covar_combos = length(books[1])
    
    # Allocate TPM book: [covar_combo][time_interval]
    tpm_book_ph = [[zeros(Float64, n_expanded, n_expanded) for _ in 1:nrow(books[1][c])] for c in 1:n_covar_combos]
    
    # Allocate hazmat book: one Q per covariate combination
    hazmat_book_ph = [copy(Q_baseline) for _ in 1:n_covar_combos]
    
    # Pre-compute covariate scaling factors for each transition and covariate combo
    # This is the exp(β'x) factor that scales inter-state transition rates
    for c in 1:n_covar_combos
        tpm_index = books[1][c]  # DataFrame with (tstart, tstop, datind)
        
        # Get a representative data row for this covariate combination
        # All rows in this combo have the same covariate values
        data_row_idx = tpm_index.datind[1]
        data_row = data[data_row_idx, :]
        
        # For each hazard in the surrogate, compute the scaling factor and apply it
        # to the corresponding inter-state transitions in Q_expanded
        for hazard in surrogate.hazards
            s_from = hazard.statefrom  # Observed state (from)
            s_to = hazard.stateto      # Observed state (to)
            
            # Compute linear predictor β'x for this hazard and covariate combination
            hazard_pars = surrogate.parameters.nested[hazard.hazname]
            
            # Extract covariates for this hazard
            covars = extract_covariates_fast(data_row, hazard.covar_names)
            
            # Compute scaling factor based on linpred_effect (PH vs AFT)
            # For proportional hazards (PH): h(t|x) = h₀(t) * exp(β'x)
            # For accelerated failure time (AFT): h(t|x) = h₀(t * exp(-β'x)) * exp(-β'x)
            linpred = _linear_predictor(hazard_pars, covars, hazard)
            if hazard.metadata.linpred_effect == :aft
                scaling_factor = exp(-linpred)  # AFT: rate scales by exp(-β'x)
            else
                scaling_factor = exp(linpred)   # PH: rate scales by exp(β'x)
            end
            
            # Apply scaling to all inter-state transitions in expanded Q
            # Inter-state transitions go from any phase in s_from to first phase in s_to
            # (in Coxian models, transitions always enter at phase 1 of destination)
            # We scale Q[phase_i, first_phase_of_s_to] for all phases i in s_from
            # Note: The baseline Q already has correct relative rates; we just scale them
            for i in 1:n_expanded
                for j in 1:n_expanded
                    # Check if this is an inter-state transition from s_from to s_to
                    if phase_to_state[i] == s_from && phase_to_state[j] == s_to
                        # Scale the inter-state transition rate
                        hazmat_book_ph[c][i, j] = Q_baseline[i, j] * scaling_factor
                    end
                end
            end
        end
        
        # Recompute diagonal elements (row sums must be zero for generator matrix)
        for i in 1:n_expanded
            row_sum = sum(hazmat_book_ph[c][i, j] for j in 1:n_expanded if j != i)
            hazmat_book_ph[c][i, i] = -row_sum
        end
    end
    
    # Allocate cache for matrix exponential
    cache = ExponentialUtilities.alloc_mem(Q_baseline, ExpMethodGeneric())
    
    # Compute TPMs for each covariate combination and time interval
    for c in 1:n_covar_combos
        Q_c = hazmat_book_ph[c]  # Covariate-specific Q matrix
        tpm_index = books[1][c]  # DataFrame with (tstart, tstop, datind)
        for t in 1:nrow(tpm_index)
            dt = tpm_index.tstop[t]  # Time interval length (tstart is always 0)
            # TPM = exp(Q_c * dt)
            tpm_book_ph[c][t] .= exponential!(copy(Q_c) .* dt, ExpMethodGeneric(), cache)
        end
    end
    
    return tpm_book_ph, hazmat_book_ph
end


"""
    build_phasetype_emat_expanded(model, surrogate::PhaseTypeSurrogate;
                                  expanded_data::Union{Nothing, DataFrame} = nothing,
                                  censoring_patterns::Union{Nothing, Matrix{Float64}} = nothing)

Build emission matrix mapping expanded phases to observed states for FFBS.

For each observation, the emission matrix E[i,j] gives the probability that
the subject is in phase j given the observation.

# Observation Types
- obstype 1: Exact transition → only FIRST phase of stateto has probability 1
  (In Coxian models, transitions always enter at phase 1 of the destination state)
- obstype 2: Panel/right-censored → all phases of stateto have equal probability
- obstype 0: Fully censored → all phases equally likely
- obstype > 2: Partial censoring → use censoring_patterns matrix
  - For phase-type data expansion, obstype = 2 + s means "in state s, phase unknown"

# Arguments
- `model`: MultistateProcess with data
- `surrogate`: PhaseTypeSurrogate with state/phase mappings
- `expanded_data`: Optional expanded data (if None, uses model.data)
- `censoring_patterns`: Optional censoring patterns for obstype > 2

# Returns
- Matrix of size (n_observations, n_expanded_states)
"""
function build_phasetype_emat_expanded(model, surrogate::PhaseTypeSurrogate;
                                       expanded_data::Union{Nothing, DataFrame} = nothing,
                                       censoring_patterns::Union{Nothing, Matrix{Float64}} = nothing)
    
    data = isnothing(expanded_data) ? model.data : expanded_data
    n_obs = nrow(data)
    n_expanded = surrogate.n_expanded_states
    n_observed = surrogate.n_observed_states
    
    emat = zeros(Float64, n_obs, n_expanded)
    
    for i in 1:n_obs
        obstype = data.obstype[i]
        
        if obstype == 1
            # Exact observation - transition always goes to FIRST phase of destination
            # (In Coxian models, you always enter a state at phase 1)
            observed_state = data.stateto[i]
            if observed_state > 0 && observed_state <= n_observed
                first_phase = first(surrogate.state_to_phases[observed_state])
                emat[i, first_phase] = 1.0
            else
                # Invalid state, allow all phases
                emat[i, :] .= 1.0
            end
        elseif obstype == 2
            # Panel observation - any phase of observed state is possible
            observed_state = data.stateto[i]
            if observed_state > 0 && observed_state <= n_observed
                phases = surrogate.state_to_phases[observed_state]
                emat[i, phases] .= 1.0
            else
                # Invalid state, allow all phases
                emat[i, :] .= 1.0
            end
        elseif obstype == 0
            # Fully censored - all phases equally likely
            emat[i, :] .= 1.0
        elseif obstype > OBSTYPE_PANEL
            # Partial censoring
            if !isnothing(censoring_patterns)
                # Use provided censoring patterns
                # For phase-type expansion: obstype = CENSORING_OBSTYPE_OFFSET + s means state s is known
                pattern_idx = obstype - CENSORING_OBSTYPE_OFFSET
                if pattern_idx <= size(censoring_patterns, 1)
                    for s in 1:min(n_observed, size(censoring_patterns, 2) - 1)
                        state_prob = censoring_patterns[pattern_idx, s + 1]
                        if state_prob > 0
                            phases = surrogate.state_to_phases[s]
                            n_phases = length(phases)
                            # Divide probability mass equally among phases
                            emat[i, phases] .= state_prob / n_phases
                        end
                    end
                else
                    # Pattern index out of range, allow all phases
                    emat[i, :] .= 1.0
                end
            else
                # No censoring patterns provided
                # For phase-type expansion convention: obstype = CENSORING_OBSTYPE_OFFSET + s means in state s
                censored_state = obstype - CENSORING_OBSTYPE_OFFSET
                if censored_state >= 1 && censored_state <= n_observed
                    phases = surrogate.state_to_phases[censored_state]
                    emat[i, phases] .= 1.0
                else
                    # Invalid censoring code, allow all phases
                    emat[i, :] .= 1.0
                end
            end
        else
            # Unknown observation type, allow all phases
            emat[i, :] .= 1.0
        end
        
        # Normalize if any positive entries
        row_sum = sum(emat[i, :])
        if row_sum > 0
            emat[i, :] ./= row_sum
        end
    end
    
    return emat
end


"""
    build_fbmats_phasetype(model, surrogate::PhaseTypeSurrogate)

Allocate forward-backward matrices for FFBS on expanded phase-type state space.

# Returns
- Vector of 3D arrays, one per subject, of size (n_obs, n_expanded, n_expanded)
"""
function build_fbmats_phasetype(model, surrogate::PhaseTypeSurrogate)
    return build_fbmats_phasetype_with_indices(model.subjectindices, surrogate)
end

"""
    build_fbmats_phasetype_with_indices(subjectindices, surrogate::PhaseTypeSurrogate)

Allocate forward-backward matrices for FFBS on expanded phase-type state space.

This version takes explicit subject indices, useful when working with expanded data.

# Arguments
- `subjectindices`: Vector of indices per subject (UnitRange or Vector{Int})
- `surrogate`: PhaseTypeSurrogate

# Returns
- Vector of 3D arrays, one per subject, of size (n_obs, n_expanded, n_expanded)
"""
function build_fbmats_phasetype_with_indices(subjectindices, surrogate::PhaseTypeSurrogate)
    n_expanded = surrogate.n_expanded_states
    
    fbmats = Vector{Array{Float64, 3}}(undef, length(subjectindices))
    
    for i in eachindex(subjectindices)
        subj_inds = subjectindices[i]
        n_obs = length(subj_inds)
        fbmats[i] = zeros(Float64, n_obs, n_expanded, n_expanded)
    end
    
    return fbmats
end


# =============================================================================
# Phase-Type Sampling Functions
# =============================================================================
#
# These functions support MCEM with phase-type proposals. The key insight is that
# a phase-type expanded model is still Markov, so we can reuse the existing FFBS
# machinery. The only differences are:
#
# 1. The state space is expanded (each observed state split into phases)
# 2. The emission matrix duplicates indicators across phases of the same state
# 3. After sampling, we collapse the expanded path back to observed states
#
# =============================================================================

"""
    expand_emat(emat, surrogate::PhaseTypeSurrogate)

Expand emission matrix from observed states to phase-type expanded states.

For each row (observation), the emission probability for an observed state is
duplicated across all phases of that state. This is correct because:
  P(obstype | phase_k of state s) = P(obstype | state s)

# Arguments
- `emat`: Original emission matrix (n_obs × n_observed_states)
- `surrogate`: PhaseTypeSurrogate with state-to-phase mappings

# Returns
- Expanded emission matrix (n_obs × n_expanded_states)
"""
function expand_emat(emat::AbstractMatrix, surrogate::PhaseTypeSurrogate)
    n_obs = size(emat, 1)
    n_expanded = surrogate.n_expanded_states
    
    emat_expanded = zeros(Float64, n_obs, n_expanded)
    
    for obs in 1:n_obs
        for (state, phases) in enumerate(surrogate.state_to_phases)
            emission_prob = emat[obs, state]
            for phase in phases
                emat_expanded[obs, phase] = emission_prob
            end
        end
    end
    
    return emat_expanded
end


"""
    BackwardSampling_expanded(subj_fbmats, surrogate::PhaseTypeSurrogate)

Backward sampling that returns expanded state indices.

Unlike `BackwardSampling!` which writes observed states to `subj_dat`, this
function returns the sampled expanded state sequence. The caller is responsible
for mapping back to observed states.

# Arguments
- `subj_fbmats`: Forward-backward matrices from ForwardFiltering!
- `surrogate`: PhaseTypeSurrogate (used only for n_expanded_states)

# Returns
- `Vector{Int}`: Sampled expanded state at each observation time
"""
function BackwardSampling_expanded(subj_fbmats, n_expanded::Int)
    n_times = size(subj_fbmats, 1)
    
    expanded_states = Vector{Int}(undef, n_times)
    
    # Sample final state
    p = normalize(vec(sum(subj_fbmats[n_times, :, :], dims=1)), 1)
    expanded_states[n_times] = rand(Categorical(p))
    
    # Backward recursion
    if n_times > 1
        for t in (n_times - 1):-1:1
            cond_probs = normalize(subj_fbmats[t+1, :, expanded_states[t+1]], 1)
            expanded_states[t] = rand(Categorical(cond_probs))
        end
    end
    
    return expanded_states
end


"""
    draw_samplepath_phasetype(subj, model, tpm_book_ph, hazmat_book_ph, 
                               tpm_map, fbmats_ph, emat_ph, surrogate, absorbingstates)

Draw a sample path from the phase-type surrogate, collapsed to observed states.

Uses the existing FFBS machinery on the expanded state space, then collapses
the sampled path back to observed states.

# Algorithm
1. Run ForwardFiltering! on expanded state space (existing function)
2. Run BackwardSampling_expanded to get expanded state endpoints
3. Run sample_ecctmc! on expanded Q matrix between expanded endpoints
4. Collapse the full path from expanded to observed states

# Returns
- `NamedTuple` with fields:
  - `collapsed`: SamplePath in observed state space (for target likelihood)
  - `expanded`: SamplePath in expanded phase state space (for surrogate likelihood)
"""
function draw_samplepath_phasetype(subj::Int64, model::MultistateProcess, 
                                    tpm_book_ph, hazmat_book_ph, tpm_map, 
                                    fbmats_ph, emat_ph, surrogate::PhaseTypeSurrogate, 
                                    absorbingstates;
                                    # Optional expanded data infrastructure for exact observations
                                    expanded_data::Union{Nothing, DataFrame} = nothing,
                                    expanded_subjectindices::Union{Nothing, Vector{UnitRange{Int64}}} = nothing,
                                    original_row_map::Union{Nothing, Vector{Int}} = nothing)
    
    # Determine if we should use expanded data for FFBS
    # This is needed when data has exact observations (obstype=1) to properly
    # account for phase uncertainty during sojourn times
    use_expanded = !isnothing(expanded_data) && !isnothing(expanded_subjectindices)
    
    if use_expanded
        return _draw_samplepath_phasetype_expanded(subj, model, tpm_book_ph, hazmat_book_ph,
                                                    tpm_map, fbmats_ph, emat_ph, surrogate,
                                                    absorbingstates, expanded_data,
                                                    expanded_subjectindices, original_row_map)
    else
        return _draw_samplepath_phasetype_original(subj, model, tpm_book_ph, hazmat_book_ph,
                                                    tpm_map, fbmats_ph, emat_ph, surrogate,
                                                    absorbingstates)
    end
end

"""
    _draw_samplepath_phasetype_original(...)

Internal implementation for panel/censored data without expansion.
Uses the original data structure for FFBS.
"""
function _draw_samplepath_phasetype_original(subj::Int64, model::MultistateProcess, 
                                              tpm_book_ph, hazmat_book_ph, tpm_map, 
                                              fbmats_ph, emat_ph, surrogate::PhaseTypeSurrogate, 
                                              absorbingstates)
    
    # Subject data from original model
    subj_inds = model.subjectindices[subj]
    subj_dat = view(model.data, subj_inds, :)
    subj_tpm_map = view(tpm_map, subj_inds, :)
    subj_emat_ph = view(emat_ph, subj_inds, :)
    
    n_obs = length(subj_inds)
    n_expanded = surrogate.n_expanded_states
    
    # Determine initial phase distribution based on first observation type
    initial_obs_state = subj_dat.statefrom[1]
    initial_phases = surrogate.state_to_phases[initial_obs_state]
    
    if subj_dat.obstype[1] == 1
        # Exact observation: entry is always into first phase (Coxian assumption)
        init_expanded = first(initial_phases)
    else
        # Panel/censored: uniform over phases of initial state
        # Create distribution vector for ForwardFiltering!
        init_expanded = zeros(Float64, n_expanded)
        init_expanded[initial_phases] .= 1.0 / length(initial_phases)
    end
    
    # Run FFBS to sample phase endpoints
    ForwardFiltering!(fbmats_ph[subj], subj_dat, tpm_book_ph, subj_tpm_map, subj_emat_ph;
                      init_state=init_expanded, hazmat_book=hazmat_book_ph)
    
    # Backward sample to get expanded state sequence
    expanded_states = BackwardSampling_expanded(fbmats_ph[subj], n_expanded)
    
    # For path construction, sample initial phase from backward distribution
    # (first element of expanded_states is already the sampled initial phase)
    init_phase_for_path = if subj_dat.obstype[1] == 1
        first(initial_phases)  # Exact: always first phase
    else
        expanded_states[1]  # Panel: use sampled phase (but we need to sample it)
    end
    
    # For panel data, we need to sample the initial phase (at tstart[1])
    # conditioned on the endpoint phase at tstop[1] (which is expanded_states[1]).
    # This ensures we only sample start phases that can reach the endpoint.
    if subj_dat.obstype[1] != 1
        # Sample initial phase conditioned on the endpoint phase at tstop[1]
        p0_given_endpoint = normalize(fbmats_ph[subj][1, :, expanded_states[1]], 1)
        init_phase_for_path = rand(Categorical(p0_given_endpoint))
    end
    
    # Initialize path in expanded space
    times_expanded = [subj_dat.tstart[1]]
    states_expanded = [init_phase_for_path]
    sizehint!(times_expanded, n_expanded * 2)
    sizehint!(states_expanded, n_expanded * 2)
    
    # Loop through intervals and sample endpoint-conditioned paths in expanded space
    for i in 1:n_obs
        # Get transition probability matrix and rate matrix for this interval
        covar_idx = subj_tpm_map[i, 1]
        time_idx = subj_tpm_map[i, 2]
        P_expanded = tpm_book_ph[covar_idx][time_idx]
        Q_expanded = hazmat_book_ph[covar_idx]
        
        # Source state in expanded space
        a_expanded = states_expanded[end]
        
        # Destination phase from FFBS
        b_expanded = expanded_states[i]
        
        if subj_dat.obstype[i] == 1
            # Exact observation: transition time is known, phase is sampled
            # ALWAYS record the endpoint time, even for survival (same state)
            # This ensures the path has the correct duration for likelihood computation
            push!(times_expanded, subj_dat.tstop[i])
            push!(states_expanded, b_expanded)
        else
            # Censored/panel observation - sample path between endpoints
            sample_ecctmc!(times_expanded, states_expanded, P_expanded, Q_expanded, 
                          a_expanded, b_expanded, subj_dat.tstart[i], subj_dat.tstop[i])
            
            # ALWAYS ensure we have the endpoint time in the path.
            # sample_ecctmc! may not add the endpoint if:
            # - No jumps occurred and a==b (returns early without adding anything)
            # - All jumps were virtual (state didn't change)
            # We need the endpoint time for likelihood computation (survival/censoring).
            if times_expanded[end] != subj_dat.tstop[i]
                push!(times_expanded, subj_dat.tstop[i])
                push!(states_expanded, b_expanded)
            end
        end
    end
    
    # Create expanded SamplePath for surrogate likelihood
    expanded_path = SamplePath(subj, copy(times_expanded), copy(states_expanded))
    
    # Collapse expanded path to observed states
    collapsed_path = collapse_phasetype_path(expanded_path, surrogate, absorbingstates)
    
    # Return both collapsed and expanded paths
    return (collapsed=collapsed_path, expanded=expanded_path)
end


"""
    _draw_samplepath_phasetype_expanded(...)

Internal implementation for exact data using expanded data structure.
Runs FFBS on the expanded data to properly account for phase uncertainty
during sojourn times.
"""
function _draw_samplepath_phasetype_expanded(subj::Int64, model::MultistateProcess, 
                                              tpm_book_ph, hazmat_book_ph, tpm_map,
                                              fbmats_ph, emat_ph, surrogate::PhaseTypeSurrogate,
                                              absorbingstates, expanded_data::DataFrame,
                                              expanded_subjectindices::Vector{UnitRange{Int64}},
                                              original_row_map::Union{Nothing, Vector{Int}})
    
    # Get subject data from both original and expanded datasets
    orig_subj_inds = model.subjectindices[subj]
    orig_subj_dat = view(model.data, orig_subj_inds, :)
    
    exp_subj_inds = expanded_subjectindices[subj]
    exp_subj_dat = view(expanded_data, exp_subj_inds, :)
    exp_subj_tpm_map = view(tpm_map, exp_subj_inds, :)
    exp_subj_emat_ph = view(emat_ph, exp_subj_inds, :)
    
    n_orig_obs = length(orig_subj_inds)
    n_exp_obs = length(exp_subj_inds)
    n_expanded = surrogate.n_expanded_states
    
    # Determine initial phase based on first observation type in ORIGINAL data
    initial_obs_state = orig_subj_dat.statefrom[1]
    initial_phases = surrogate.state_to_phases[initial_obs_state]
    
    if orig_subj_dat.obstype[1] == 1
        # Exact observation: entry is always into first phase (Coxian assumption)
        init_expanded = first(initial_phases)
        init_phase_for_path = init_expanded
    else
        # Panel/censored: uniform over phases of initial state
        init_expanded = zeros(Float64, n_expanded)
        init_expanded[initial_phases] .= 1.0 / length(initial_phases)
        init_phase_for_path = nothing  # Will be sampled after forward pass
    end
    
    # Run FFBS on expanded data to sample phase endpoints
    ForwardFiltering!(fbmats_ph[subj], exp_subj_dat, tpm_book_ph, exp_subj_tpm_map, exp_subj_emat_ph;
                      init_state=init_expanded, hazmat_book=hazmat_book_ph)
    
    # Backward sample to get expanded state sequence for all expanded rows
    exp_expanded_states = BackwardSampling_expanded(fbmats_ph[subj], n_expanded)
    
    # Sample initial phase if panel data
    if isnothing(init_phase_for_path)
        p0_given_obs = normalize(vec(sum(fbmats_ph[subj][1, :, :], dims=2)), 1)
        init_phase_for_path = rand(Categorical(p0_given_obs))
    end
    
    # Build mapping from original rows to their corresponding transition rows in expanded data
    # For each original exact observation, find the expanded row that has the transition (obstype=1)
    # Non-exact observations map 1:1 (they weren't expanded)
    orig_to_exp_phase = Vector{Int}(undef, n_orig_obs)
    
    exp_offset = 0
    for i in 1:n_orig_obs
        if orig_subj_dat.obstype[i] == 1
            # Exact observation: maps to the second expanded row (the transition row)
            # The first expanded row is the sojourn, second is the transition
            orig_to_exp_phase[i] = exp_expanded_states[exp_offset + 2]
            exp_offset += 2
        else
            # Non-exact: maps directly (wasn't expanded)
            exp_offset += 1
            orig_to_exp_phase[i] = exp_expanded_states[exp_offset]
        end
    end
    
    # Now construct the path using original observation structure but sampled phases
    times_expanded = [orig_subj_dat.tstart[1]]
    states_expanded = [init_phase_for_path]
    sizehint!(times_expanded, n_expanded * 2)
    sizehint!(states_expanded, n_expanded * 2)
    
    # For exact observations, we also need the phase at the end of sojourn
    # to sample the path during the sojourn interval
    exp_idx = 0
    for i in 1:n_orig_obs
        # Source state in expanded space
        a_expanded = states_expanded[end]
        
        if orig_subj_dat.obstype[i] == 1
            # Exact observation: we need to sample path during sojourn then record transition
            
            # Phase at end of sojourn (from first expanded row of this pair)
            sojourn_end_phase = exp_expanded_states[exp_idx + 1]
            # Phase after transition (from second expanded row)
            transition_phase = exp_expanded_states[exp_idx + 2]
            
            # Get TPM for sojourn interval [tstart, tstop - ε]
            sojourn_covar_idx = exp_subj_tpm_map[exp_idx + 1, 1]
            sojourn_time_idx = exp_subj_tpm_map[exp_idx + 1, 2]
            P_sojourn = tpm_book_ph[sojourn_covar_idx][sojourn_time_idx]
            Q_expanded = hazmat_book_ph[sojourn_covar_idx]
            
            # Sample path during sojourn (from current phase to sojourn_end_phase)
            sojourn_tstart = orig_subj_dat.tstart[i]
            sojourn_tstop = exp_subj_dat.tstop[exp_idx + 1]  # = orig tstop - epsilon
            
            sample_ecctmc!(times_expanded, states_expanded, P_sojourn, Q_expanded,
                          a_expanded, sojourn_end_phase, sojourn_tstart, sojourn_tstop)
            
            # Ensure we end at sojourn_end_phase
            if states_expanded[end] != sojourn_end_phase
                push!(times_expanded, sojourn_tstop)
                push!(states_expanded, sojourn_end_phase)
            end
            
            # Record the transition at the original observation time
            if surrogate.phase_to_state[sojourn_end_phase] != surrogate.phase_to_state[transition_phase]
                push!(times_expanded, orig_subj_dat.tstop[i])
                push!(states_expanded, transition_phase)
            end
            
            exp_idx += 2
        else
            # Censored/panel observation: sample path as before
            exp_idx += 1
            b_expanded = exp_expanded_states[exp_idx]
            
            covar_idx = exp_subj_tpm_map[exp_idx, 1]
            time_idx = exp_subj_tpm_map[exp_idx, 2]
            P_expanded = tpm_book_ph[covar_idx][time_idx]
            Q_expanded = hazmat_book_ph[covar_idx]
            
            sample_ecctmc!(times_expanded, states_expanded, P_expanded, Q_expanded,
                          a_expanded, b_expanded, orig_subj_dat.tstart[i], orig_subj_dat.tstop[i])
            
            if states_expanded[end] != b_expanded
                push!(times_expanded, orig_subj_dat.tstop[i])
                push!(states_expanded, b_expanded)
            end
        end
    end
    
    # Create expanded SamplePath for surrogate likelihood
    expanded_path = SamplePath(subj, copy(times_expanded), copy(states_expanded))
    
    # Collapse expanded path to observed states
    collapsed_path = collapse_phasetype_path(expanded_path, surrogate, absorbingstates)
    
    return (collapsed=collapsed_path, expanded=expanded_path)
end


"""
    collapse_phasetype_path(expanded_path::SamplePath, surrogate::PhaseTypeSurrogate, absorbingstates)

Collapse a path from the expanded phase-type state space to the observed state space.

Maps each phase back to its corresponding observed state and removes consecutive 
duplicates (transitions between phases of the same state).

# Arguments
- `expanded_path`: SamplePath in the expanded phase state space
- `surrogate`: PhaseTypeSurrogate with phase_to_state mapping
- `absorbingstates`: Vector of absorbing state indices

# Returns
- `SamplePath`: Path in the observed (collapsed) state space
"""
function collapse_phasetype_path(expanded_path::SamplePath, surrogate::PhaseTypeSurrogate, absorbingstates)
    times_expanded = expanded_path.times
    states_expanded = expanded_path.states
    subj = expanded_path.subj
    
    # Map to observed states, keeping only transitions that change observed state
    times_obs = [times_expanded[1]]
    states_obs = [surrogate.phase_to_state[states_expanded[1]]]
    
    for i in 2:length(times_expanded)
        obs_state = surrogate.phase_to_state[states_expanded[i]]
        # Only record if observed state changes
        if obs_state != states_obs[end]
            push!(times_obs, times_expanded[i])
            push!(states_obs, obs_state)
        end
    end
    
    # CRITICAL: Always include the final time/state, even if observed state didn't change.
    # This is needed for computing the survival likelihood (time spent in final state).
    # Without this, a path that survives in state 1 from t=0 to t=4 would be collapsed to
    # just [0], [1] with no duration information, yielding loglik = 0 (incorrect).
    final_obs_state = surrogate.phase_to_state[states_expanded[end]]
    final_time = times_expanded[end]
    if length(times_obs) == 1 || (times_obs[end] != final_time)
        # Add final point with same observed state (representing survival/censoring)
        push!(times_obs, final_time)
        push!(states_obs, final_obs_state)
    end
    
    # Truncate at absorbing states
    truncind = findfirst(states_obs .∈ Ref(absorbingstates))
    if !isnothing(truncind)
        times_obs = first(times_obs, truncind)
        states_obs = first(states_obs, truncind)
    end
    
    return reduce_jumpchain(SamplePath(subj, times_obs, states_obs))
end


"""
    loglik_phasetype_expanded_path(expanded_path::SamplePath, Q::Matrix{Float64})

Compute log-density of a sample path under a CTMC with intensity matrix Q.

This computes the CTMC path density directly:
  log f(path) = Σᵢ [-qₛᵢ * Δtᵢ + log(qₛᵢ,dᵢ)]

where qₛ = -Q[s,s] is the total exit rate from state s, and q_{s,d} = Q[s,d] is the 
transition rate from s to d.

# Arguments
- `expanded_path::SamplePath`: Sample path in expanded phase space (states index into Q)
- `Q::Matrix{Float64}`: CTMC intensity matrix in expanded phase space

# Returns
- `Float64`: Log-likelihood (density) of the path

# See also
- [`loglik_phasetype_forward`](@ref): For marginal likelihood with panel data
"""
function loglik_phasetype_expanded_path(expanded_path::SamplePath, Q::Matrix{Float64})
    loglik = 0.0
    
    n_transitions = length(expanded_path.times) - 1
    if n_transitions == 0
        return 0.0
    end
    
    for i in 1:n_transitions
        t0 = expanded_path.times[i]
        t1 = expanded_path.times[i + 1]
        dt = t1 - t0
        
        s = expanded_path.states[i]
        d = expanded_path.states[i + 1]
        
        # Survival term: exp(-q_s * dt) where q_s is total exit rate from s
        q_s = -Q[s, s]  # Diagonal is negative total exit rate
        loglik += -q_s * dt
        
        # Transition term: log(q_{s,d}) if s != d
        if s != d
            q_sd = Q[s, d]
            if q_sd > 0
                loglik += log(q_sd)
            else
                return -Inf  # Impossible transition
            end
        end
    end
    
    return loglik
end

# Convenience method using PhaseTypeSurrogate
loglik_phasetype_expanded_path(expanded_path::SamplePath, surrogate::PhaseTypeSurrogate) = 
    loglik_phasetype_expanded_path(expanded_path, surrogate.expanded_Q)


"""
    loglik_phasetype_expanded_path_covar(expanded_path, hazmat_book_ph, subj_tpm_map, original_subj_data)

Compute log-density of an expanded phase-space path with covariate support.

This computes the CTMC path density using the appropriate Q matrix for each
interval based on the covariate combination at that time.

# Arguments
- `expanded_path::SamplePath`: Sample path in expanded phase space
- `hazmat_book_ph::Vector{Matrix{Float64}}`: Q matrices for each covariate combination
- `subj_tpm_map::SubArray`: Mapping from intervals to [covar_idx, time_idx]
- `original_subj_data::SubDataFrame`: Original observation data for this subject

# Returns
- `Float64`: Log-likelihood (density) of the path under the PhaseType surrogate

# Algorithm
For each transition in the expanded path:
1. Find the original observation interval that contains this transition time
2. Look up the covariate combination for that interval
3. Use the corresponding Q matrix from hazmat_book_ph
4. Compute: -q_s * dt + log(q_{s,d}) for each transition
"""
function loglik_phasetype_expanded_path_covar(
    expanded_path::SamplePath,
    hazmat_book_ph::Vector{Matrix{Float64}},
    subj_tpm_map::SubArray,
    original_subj_data::SubDataFrame
)
    loglik = 0.0
    
    n_transitions = length(expanded_path.times) - 1
    if n_transitions == 0
        return 0.0
    end
    
    # Pre-extract observation interval endpoints for efficient lookup
    obs_tstarts = original_subj_data.tstart
    obs_tstops = original_subj_data.tstop
    n_obs = nrow(original_subj_data)
    
    for i in 1:n_transitions
        t0 = expanded_path.times[i]
        t1 = expanded_path.times[i + 1]
        dt = t1 - t0
        
        s = expanded_path.states[i]
        d = expanded_path.states[i + 1]
        
        # Find the original observation interval containing this transition
        # The interval [t0, t1] should fall within some observation interval
        # Use t0 to determine which interval (in case of boundary)
        obs_idx = 1
        for k in 1:n_obs
            if t0 >= obs_tstarts[k] - eps() && t1 <= obs_tstops[k] + eps()
                obs_idx = k
                break
            end
        end
        
        # Get the covariate combination index for this interval
        covar_idx = subj_tpm_map[obs_idx, 1]
        Q = hazmat_book_ph[covar_idx]
        
        # Survival term: exp(-q_s * dt) where q_s is total exit rate from s
        q_s = -Q[s, s]  # Diagonal is negative total exit rate
        loglik += -q_s * dt
        
        # Transition term: log(q_{s,d}) if s != d
        if s != d
            q_sd = Q[s, d]
            if q_sd > 0
                loglik += log(q_sd)
            else
                return -Inf  # Impossible transition
            end
        end
    end
    
    return loglik
end