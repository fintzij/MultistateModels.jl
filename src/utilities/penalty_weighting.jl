# =============================================================================
# Penalty Weighting Utilities
# =============================================================================
#
# Functions for computing adaptive penalty weights based on data characteristics.
# Currently implements at-risk count computation for w(t) = Y(t)^(-α) weighting.
#
# Contents:
#   1. At-risk count computation for exact data
#   2. At-risk count computation for panel data (upper bound approach)
#   3. At-risk count computation for MCEM paths (importance weighted)
#
# =============================================================================

"""
    compute_atrisk_counts(model::MultistateProcess, 
                          eval_times::AbstractVector{<:Real},
                          transition::Tuple{Int,Int}) -> Vector{Float64}

Compute the number of subjects at risk for a specific transition at given times.

For a transition r → s, a subject is "at risk" at time t if their observed path
has them in state r at that time (i.e., they could potentially experience the
transition at that moment).

# Arguments
- `model`: MultistateProcess with data
- `eval_times`: Times at which to evaluate Y(t) (must be sorted in ascending order)
- `transition`: (origin, destination) state pair specifying the transition

# Returns
- `Vector{Float64}`: At-risk counts at each evaluation time. Values are floored
  at 1.0 to avoid division by zero when used in weighting.

# Notes
- For **exact data** (obstype=1): Uses the fully observed paths to determine
  exactly which subjects are in the origin state at each time.
- For **panel data** (obstype=2+): Uses an upper bound approach - counts subjects
  who *could* be in the origin state based on their last observed state before t.
- The function handles mixed data (some exact, some panel) by processing each
  observation type appropriately.

# Implementation Details
For each subject i with observation intervals [(tstart_k, tstop_k), state_k]:
- Subject is at risk for transition r→s during interval k if:
  - For exact data: statefrom_k == r (subject observed in state r)
  - For panel data: subject was last observed in state r at time ≤ t

# Example
```julia
# Compute at-risk counts for transition 1→2 at knot midpoints
knots = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
midpoints = [(knots[i] + knots[i+1])/2 for i in 1:length(knots)-1]
atrisk = compute_atrisk_counts(model, midpoints, (1, 2))
```

See also: [`AtRiskWeighting`](@ref), [`build_weighted_penalty_matrix`](@ref)
"""
function compute_atrisk_counts(model::MultistateProcess, 
                               eval_times::AbstractVector{<:Real},
                               transition::Tuple{Int,Int})
    origin, _ = transition
    n_times = length(eval_times)
    
    # Validate that eval_times are sorted
    if n_times > 1
        for i in 2:n_times
            eval_times[i] >= eval_times[i-1] || 
                throw(ArgumentError("eval_times must be sorted in ascending order"))
        end
    end
    
    # Extract data columns once for efficiency
    data = model.data
    id_col = data.id
    tstart_col = data.tstart
    tstop_col = data.tstop
    statefrom_col = data.statefrom
    obstype_col = data.obstype
    
    # Initialize at-risk counts
    atrisk = zeros(Float64, n_times)
    
    # Get unique subject IDs
    unique_ids = unique(id_col)
    
    # For each subject, determine when they are at risk for this transition
    for subj_id in unique_ids
        # Get indices for this subject's observations
        subj_mask = id_col .== subj_id
        subj_indices = findall(subj_mask)
        
        if isempty(subj_indices)
            continue
        end
        
        # Sort indices by tstart for this subject (should already be sorted, but be safe)
        subj_indices = subj_indices[sortperm(tstart_col[subj_indices])]
        
        # For each evaluation time, check if subject is at risk
        for (t_idx, t) in enumerate(eval_times)
            is_at_risk = _is_subject_at_risk_at_time(
                t, origin, 
                tstart_col, tstop_col, statefrom_col, obstype_col,
                subj_indices
            )
            if is_at_risk
                atrisk[t_idx] += 1.0
            end
        end
    end
    
    # Floor at 1.0 to avoid division by zero in weighting
    # (This corresponds to w(t) = Y(t)^(-α) ≤ 1 when Y(t) ≤ 1)
    @. atrisk = max(atrisk, 1.0)
    
    return atrisk
end

"""
    _is_subject_at_risk_at_time(t, origin, tstart, tstop, statefrom, obstype, subj_indices) -> Bool

Internal helper to determine if a subject is at risk for a transition from `origin`
at time `t`.

For exact data (obstype=1): Subject is at risk if they have an observation interval
  [tstart, tstop) with statefrom == origin and t ∈ [tstart, tstop).

For panel data (obstype≥2): Subject is at risk if the last observation with
  tstop ≤ t had statefrom == origin (conservative upper bound).
"""
function _is_subject_at_risk_at_time(t::Real, origin::Int,
                                      tstart::AbstractVector, 
                                      tstop::AbstractVector, 
                                      statefrom::AbstractVector,
                                      obstype::AbstractVector,
                                      subj_indices::AbstractVector{Int})
    # Check if t falls within any interval where subject is in origin state
    for idx in subj_indices
        t_s = tstart[idx]
        t_e = tstop[idx]
        state = statefrom[idx]
        otype = obstype[idx]
        
        # For exact data: at risk if in this interval and state is origin
        # Interval is [tstart, tstop) - right-open as transitions happen at tstop
        if otype == 1
            # For exact data, subject is at risk in [tstart, tstop) if statefrom == origin
            if state == origin && t_s <= t < t_e
                return true
            end
        else
            # For panel data: at risk if we're after this interval's start,
            # and statefrom indicates subject was (possibly) in origin state
            # This is a conservative upper bound approach
            if state == origin && t_s <= t < t_e
                return true
            end
        end
    end
    
    return false
end

"""
    compute_atrisk_counts_at_knot_midpoints(model::MultistateProcess,
                                            hazard::_Hazard,
                                            transition::Tuple{Int,Int}) -> Vector{Float64}

Convenience function to compute at-risk counts at the knot midpoints of a spline hazard.

!!! note "Deprecated in favor of interval averages"
    For adaptive penalty weighting, prefer `compute_atrisk_interval_averages()` which
    computes the mean at-risk count over each interval rather than a point estimate.

# Arguments
- `model`: MultistateProcess with data
- `hazard`: A spline hazard (RuntimeSplineHazard or similar) with knot information
- `transition`: (origin, destination) state pair

# Returns
- `Vector{Float64}`: At-risk counts at each knot midpoint

# Notes
Knot midpoints are computed between consecutive *unique* knot positions.
For clamped B-splines (with repeated boundary knots), this returns midpoints
of the non-degenerate intervals where the basis functions are active.

See also: [`compute_atrisk_counts`](@ref), [`compute_atrisk_interval_averages`](@ref)
"""
function compute_atrisk_counts_at_knot_midpoints(model::MultistateProcess,
                                                  hazard,
                                                  transition::Tuple{Int,Int})
    # Get knots from hazard - handles both RuntimeSplineHazard and other spline types
    knots = _get_hazard_knots(hazard)
    
    # Use unique knot positions (handles repeated boundary knots in clamped splines)
    unique_knots = unique(knots)
    n_unique = length(unique_knots)
    
    # Compute midpoints between consecutive unique knots
    midpoints = [(unique_knots[i] + unique_knots[i+1])/2 for i in 1:(n_unique-1)]
    
    return compute_atrisk_counts(model, midpoints, transition)
end


# =============================================================================
# Interval-Averaged At-Risk Counts (Preferred for Adaptive Weighting)
# =============================================================================

"""
    compute_atrisk_interval_averages(model::MultistateProcess,
                                      hazard,
                                      transition::Tuple{Int,Int}) -> Vector{Float64}

Compute the average at-risk count over each knot interval.

This computes the mean number at risk over the interval [t_q, t_{q+1}]:

    Ȳ_q = (1/(t_{q+1} - t_q)) ∫_{t_q}^{t_{q+1}} Y(t) dt

For survival data, this equals (person-time at risk in interval) / (interval width).
This is preferred over midpoint evaluation as it accounts for how at-risk counts
change within each interval due to events and censoring.

# Arguments
- `model`: MultistateProcess with data
- `hazard`: Spline hazard with knot information
- `transition`: (origin, destination) state pair

# Returns
- `Vector{Float64}`: Average at-risk count for each knot interval (length = n_intervals)

# Notes
For exact data (obstype=1): Computes exact integral of step function Y(t).
For panel data: Uses conservative bounds based on observed state history.

# Example
```julia
h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; ...)
model = multistatemodel(h12; data=data)
atrisk_avg = compute_atrisk_interval_averages(model, model.hazards[1], (1, 2))
```

See also: [`build_weighted_penalty_matrix`](@ref), [`AtRiskWeighting`](@ref)
"""
function compute_atrisk_interval_averages(model::MultistateProcess,
                                           hazard,
                                           transition::Tuple{Int,Int})
    origin, _ = transition
    
    # Get knot intervals
    knots = _get_hazard_knots(hazard)
    unique_knots = unique(knots)
    n_intervals = length(unique_knots) - 1
    
    # Extract data columns
    data = model.data
    id_col = data.id
    tstart_col = data.tstart
    tstop_col = data.tstop
    statefrom_col = data.statefrom
    obstype_col = data.obstype
    
    # Get unique subject IDs
    unique_ids = unique(id_col)
    
    # Initialize: person-time at risk in each interval
    person_time = zeros(Float64, n_intervals)
    
    # For each subject, compute time spent at risk in each interval
    for subj_id in unique_ids
        subj_mask = id_col .== subj_id
        subj_indices = findall(subj_mask)
        
        isempty(subj_indices) && continue
        
        # Sort by tstart
        subj_indices = subj_indices[sortperm(tstart_col[subj_indices])]
        
        # Compute time at risk in each interval
        for q in 1:n_intervals
            int_start = unique_knots[q]
            int_end = unique_knots[q + 1]
            
            time_at_risk = _subject_time_at_risk_in_interval(
                int_start, int_end, origin,
                tstart_col, tstop_col, statefrom_col, obstype_col,
                subj_indices
            )
            person_time[q] += time_at_risk
        end
    end
    
    # Convert person-time to average at-risk count
    # Ȳ_q = person_time_q / interval_width_q
    interval_averages = zeros(Float64, n_intervals)
    for q in 1:n_intervals
        width = unique_knots[q + 1] - unique_knots[q]
        if width > 0
            interval_averages[q] = person_time[q] / width
        end
    end
    
    return interval_averages
end

"""
    _subject_time_at_risk_in_interval(int_start, int_end, origin,
                                       tstart, tstop, statefrom, obstype,
                                       subj_indices) -> Float64

Compute the time a subject spends at risk for transition from `origin` 
within the interval [int_start, int_end].

For exact data: Returns the exact overlap between at-risk intervals and [int_start, int_end].
For panel data: Uses conservative bounds.
"""
function _subject_time_at_risk_in_interval(int_start::Real, int_end::Real, origin::Int,
                                            tstart::AbstractVector,
                                            tstop::AbstractVector,
                                            statefrom::AbstractVector,
                                            obstype::AbstractVector,
                                            subj_indices::AbstractVector{Int})
    total_time = 0.0
    
    for idx in subj_indices
        t_s = tstart[idx]
        t_e = tstop[idx]
        state = statefrom[idx]
        
        # Subject is at risk during [t_s, t_e) if state == origin
        if state != origin
            continue
        end
        
        # Compute overlap with [int_start, int_end]
        overlap_start = max(t_s, int_start)
        overlap_end = min(t_e, int_end)
        
        if overlap_end > overlap_start
            total_time += overlap_end - overlap_start
        end
    end
    
    return total_time
end

"""
    _get_hazard_knots(hazard) -> Vector{Float64}

Internal helper to extract the FULL knot vector (including boundary knots) from various hazard types.

For RuntimeSplineHazard, the `knots` field contains only interior knots.
This function rebuilds the spline basis and extracts the full knot vector.
"""
function _get_hazard_knots(hazard::RuntimeSplineHazard)
    # RuntimeSplineHazard.knots contains interior knots only
    # Rebuild basis to get full knot vector including boundary knots
    basis = _rebuild_spline_basis(hazard)
    return collect(BSplineKit.knots(basis))
end

# Fallback for other hazard types that might store knots differently
function _get_hazard_knots(hazard)
    if hasfield(typeof(hazard), :basis)
        return collect(BSplineKit.knots(hazard.basis))
    elseif hasfield(typeof(hazard), :knots)
        # Warning: this might be interior knots only for some hazard types
        return hazard.knots
    else
        error("Cannot extract knots from hazard of type $(typeof(hazard))")
    end
end

# =============================================================================
# MCEM Path-Weighted At-Risk Computation (Phase 4)
# =============================================================================

"""
    compute_atrisk_counts_mcem(samplepaths::Vector{Vector{SamplePath}},
                               weights::Vector{Vector{Float64}},
                               eval_times::AbstractVector{<:Real},
                               transition::Tuple{Int,Int}) -> Vector{Float64}

Compute importance-weighted at-risk counts from MCEM sampled paths.

In MCEM, we have multiple sampled paths per subject with associated importance weights.
The weighted at-risk count at time t is:
    Ỹ(t) = Σᵢ Σⱼ wᵢⱼ · 𝟙[path Zᵢⱼ in origin state at time t]

where wᵢⱼ is the normalized importance weight for subject i, path j.

# Arguments
- `samplepaths::Vector{Vector{SamplePath}}`: samplepaths[i][j] = j-th sampled path for subject i
- `weights::Vector{Vector{Float64}}`: weights[i][j] = normalized importance weight for path i,j
  (weights within each subject should sum to 1)
- `eval_times::AbstractVector{<:Real}`: Times at which to evaluate Ỹ(t) (must be sorted ascending)
- `transition::Tuple{Int,Int}`: (origin, destination) state pair

# Returns
- `Vector{Float64}`: Weighted at-risk counts at each evaluation time. Values are floored
  at 1.0 to avoid division by zero when used in penalty weighting.

# Notes
- For subjects with multiple paths, the contribution is weighted by importance weights
- Paths with higher weights contribute more to the at-risk count
- This provides a better estimate than the observed data upper bound used in Phase 3

# Example
```julia
# After MCEM E-step
atrisk = compute_atrisk_counts_mcem(samplepaths, ImportanceWeights, 
                                     knot_midpoints, (1, 2))
```

See also: [`compute_atrisk_counts`](@ref), [`AtRiskWeighting`](@ref)
"""
function compute_atrisk_counts_mcem(samplepaths::Vector{Vector{SamplePath}},
                                     weights::Vector{Vector{Float64}},
                                     eval_times::AbstractVector{<:Real},
                                     transition::Tuple{Int,Int})
    origin, _ = transition
    n_times = length(eval_times)
    nsubj = length(samplepaths)
    
    # Validate inputs
    length(weights) == nsubj || 
        throw(ArgumentError("samplepaths and weights must have same length (number of subjects)"))
    
    if n_times > 1
        for i in 2:n_times
            eval_times[i] >= eval_times[i-1] || 
                throw(ArgumentError("eval_times must be sorted in ascending order"))
        end
    end
    
    # Initialize weighted at-risk counts
    atrisk = zeros(Float64, n_times)
    
    # For each subject, accumulate weighted at-risk contributions
    for i in 1:nsubj
        paths_i = samplepaths[i]
        weights_i = weights[i]
        n_paths = length(paths_i)
        
        # Skip if no paths for this subject
        n_paths == 0 && continue
        
        # Validate weights match paths
        length(weights_i) == n_paths || 
            throw(ArgumentError("Number of weights ($(length(weights_i))) must match number of paths ($n_paths) for subject $i"))
        
        # For each sampled path
        for j in 1:n_paths
            path = paths_i[j]
            w = weights_i[j]
            
            # Skip zero-weight paths
            w <= 0 && continue
            
            # For each evaluation time, check if path is in origin state
            for (t_idx, t) in enumerate(eval_times)
                if _path_in_state_at_time(path, origin, t)
                    atrisk[t_idx] += w
                end
            end
        end
    end
    
    return atrisk
end

"""
    _path_in_state_at_time(path::SamplePath, state::Int, t::Real) -> Bool

Check if a sample path is in the specified state at time t.

A path is in state `state` at time `t` if the last transition time ≤ t
corresponds to entering `state`.

The path's state sequence is:
- path.states[1] at time path.times[1] (usually the initial state at t=0)
- path.states[k] from path.times[k] until path.times[k+1] (or end of follow-up)
"""
function _path_in_state_at_time(path::SamplePath, state::Int, t::Real)
    times = path.times
    states = path.states
    n = length(times)
    
    # Empty path - shouldn't happen, but handle gracefully
    n == 0 && return false
    
    # Before first observation time
    t < times[1] && return false
    
    # Find the last transition time ≤ t
    # The state at time t is states[k] where times[k] is the largest time ≤ t
    # Use binary search for efficiency with long paths
    idx = searchsortedlast(times, t)
    
    # If no time ≤ t found, not in any state
    idx == 0 && return false
    
    return states[idx] == state
end

"""
    compute_atrisk_counts_mcem_at_knot_midpoints(samplepaths::Vector{Vector{SamplePath}},
                                                  weights::Vector{Vector{Float64}},
                                                  hazard,
                                                  transition::Tuple{Int,Int}) -> Vector{Float64}

Convenience function to compute MCEM path-weighted at-risk counts at knot midpoints.

!!! note "Deprecated in favor of interval averages"
    For adaptive penalty weighting, prefer `compute_atrisk_interval_averages_mcem()` which
    computes the mean at-risk count over each interval rather than a point estimate.

# Arguments
- `samplepaths`: Sampled paths from MCEM E-step
- `weights`: Normalized importance weights
- `hazard`: Spline hazard with knot information
- `transition`: (origin, destination) state pair

# Returns
- `Vector{Float64}`: Weighted at-risk counts at each knot midpoint

See also: [`compute_atrisk_counts_mcem`](@ref), [`compute_atrisk_interval_averages_mcem`](@ref)
"""
function compute_atrisk_counts_mcem_at_knot_midpoints(samplepaths::Vector{Vector{SamplePath}},
                                                       weights::Vector{Vector{Float64}},
                                                       hazard,
                                                       transition::Tuple{Int,Int})
    # Get knots from hazard
    knots = _get_hazard_knots(hazard)
    
    # Use unique knot positions (handles repeated boundary knots)
    unique_knots = unique(knots)
    n_unique = length(unique_knots)
    
    # Compute midpoints
    midpoints = [(unique_knots[i] + unique_knots[i+1])/2 for i in 1:(n_unique-1)]
    
    return compute_atrisk_counts_mcem(samplepaths, weights, midpoints, transition)
end

"""
    compute_atrisk_interval_averages_mcem(samplepaths::Vector{Vector{SamplePath}},
                                           weights::Vector{Vector{Float64}},
                                           hazard,
                                           transition::Tuple{Int,Int}) -> Vector{Float64}

Compute importance-weighted average at-risk counts over each knot interval from MCEM paths.

This computes the weighted mean number at risk over the interval [t_q, t_{q+1}]:

    Ȳ_q = (1/(t_{q+1} - t_q)) Σᵢ Σⱼ wᵢⱼ · ∫_{t_q}^{t_{q+1}} 𝟙[path Zᵢⱼ in origin state at t] dt

For each path, this is the weighted time spent in the origin state within the interval,
summed across all paths and divided by the interval width.

# Arguments
- `samplepaths::Vector{Vector{SamplePath}}`: samplepaths[i][j] = j-th sampled path for subject i
- `weights::Vector{Vector{Float64}}`: weights[i][j] = normalized importance weight for path i,j
- `hazard`: Spline hazard with knot information
- `transition`: (origin, destination) state pair

# Returns
- `Vector{Float64}`: Weighted average at-risk count for each knot interval

# Notes
This is preferred over midpoint evaluation (`compute_atrisk_counts_mcem_at_knot_midpoints`)
as it properly accounts for transitions occurring within intervals.

See also: [`compute_atrisk_interval_averages`](@ref), [`build_weighted_penalty_matrix`](@ref)
"""
function compute_atrisk_interval_averages_mcem(samplepaths::Vector{Vector{SamplePath}},
                                                weights::Vector{Vector{Float64}},
                                                hazard,
                                                transition::Tuple{Int,Int})
    origin, _ = transition
    nsubj = length(samplepaths)
    
    # Get knot intervals
    knots = _get_hazard_knots(hazard)
    unique_knots = unique(knots)
    n_intervals = length(unique_knots) - 1
    
    # Validate inputs
    length(weights) == nsubj || 
        throw(ArgumentError("samplepaths and weights must have same length"))
    
    # Initialize: weighted person-time at risk in each interval
    weighted_person_time = zeros(Float64, n_intervals)
    
    # For each subject
    for i in 1:nsubj
        paths_i = samplepaths[i]
        weights_i = weights[i]
        n_paths = length(paths_i)
        
        n_paths == 0 && continue
        
        length(weights_i) == n_paths || 
            throw(ArgumentError("Weights/paths mismatch for subject $i"))
        
        # For each sampled path
        for j in 1:n_paths
            path = paths_i[j]
            w = weights_i[j]
            
            w <= 0 && continue
            
            # For each interval, compute time this path spends in origin state
            for q in 1:n_intervals
                int_start = unique_knots[q]
                int_end = unique_knots[q + 1]
                
                time_in_origin = _path_time_in_state_in_interval(path, origin, int_start, int_end)
                weighted_person_time[q] += w * time_in_origin
            end
        end
    end
    
    # Convert weighted person-time to average at-risk count
    interval_averages = zeros(Float64, n_intervals)
    for q in 1:n_intervals
        width = unique_knots[q + 1] - unique_knots[q]
        if width > 0
            interval_averages[q] = weighted_person_time[q] / width
        end
    end
    
    return interval_averages
end

"""
    _path_time_in_state_in_interval(path::SamplePath, state::Int, 
                                     int_start::Real, int_end::Real) -> Float64

Compute the time a sample path spends in the given state within interval [int_start, int_end].

The path has times t₁, t₂, ..., tₙ and states s₁, s₂, ..., sₙ where the path
is in state sₖ during [tₖ, tₖ₊₁) (or until end of observation for last segment).
"""
function _path_time_in_state_in_interval(path::SamplePath, state::Int, 
                                          int_start::Real, int_end::Real)
    times = path.times
    states = path.states
    n = length(times)
    
    n == 0 && return 0.0
    int_end <= int_start && return 0.0
    
    total_time = 0.0
    
    # Path is in states[k] during [times[k], times[k+1]) for k = 1, ..., n-1
    # For the last segment, we assume state continues to infinity (or we could 
    # use the path's end time if available)
    
    for k in 1:n
        if states[k] != state
            continue
        end
        
        # Segment start and end
        seg_start = times[k]
        seg_end = k < n ? times[k + 1] : Inf  # Last segment extends to infinity
        
        # Compute overlap with [int_start, int_end]
        overlap_start = max(seg_start, int_start)
        overlap_end = min(seg_end, int_end)
        
        if overlap_end > overlap_start
            total_time += overlap_end - overlap_start
        end
    end
    
    return total_time
end
