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

# =============================================================================
# Alpha Learning for Adaptive Penalty Weighting (Phase 5)
# =============================================================================

"""
    AlphaLearningInfo

Struct to hold precomputed data needed for alpha learning.

# Fields
- `hazard_idx::Int`: Index of hazard in model.hazards
- `hazard::_Hazard`: The hazard object
- `transition::Tuple{Int,Int}`: (origin, dest) state pair
- `order::Int`: Penalty derivative order
- `atrisk::Vector{Float64}`: Pre-computed interval-averaged at-risk counts
- `param_indices::UnitRange{Int}`: Indices into flat parameter vector
"""
struct AlphaLearningInfo
    hazard_idx::Int
    hazard::_Hazard  # Runtime hazard object
    transition::Tuple{Int,Int}
    order::Int
    atrisk::Vector{Float64}  # At-risk interval averages
    param_indices::UnitRange{Int}  # Indices into flat beta vector
end

"""
    learn_alpha(model::MultistateProcess,
                data,
                penalty::QuadraticPenalty,
                beta::Vector{Float64},
                term_idx::Int,
                hazard,
                atrisk::Vector{Float64};
                alpha_bounds::Tuple{Float64,Float64}=(0.0, 2.0),
                tol::Float64=1e-2) -> Float64

Estimate optimal α for a single penalty term by maximizing marginal likelihood.

For fixed λ and β, the marginal likelihood criterion for α is:
    p(y|λ,α) ∝ |H + λS_w(α)|^(-1/2) exp(-ℓ(β) - λ/2 β'S_w(α)β)

Taking log and ignoring constants:
    log p(y|λ,α) = -1/2 log|H + λS_w(α)| - ℓ(β) - λ/2 β'S_w(α)β

Since ℓ(β) is fixed, we minimize:
    -1/2 log|H + λS_w(α)| + λ/2 β'S_w(α)β

Uses 1D optimization (Brent's method) since α is scalar.

# Arguments
- `model::MultistateProcess`: Model with hazard definitions
- `data`: Data container (ExactData or MPanelData)
- `penalty::QuadraticPenalty`: Current penalty configuration
- `beta::Vector{Float64}`: Current coefficient estimates
- `term_idx::Int`: Index of penalty term to update
- `hazard`: RuntimeSplineHazard for this term
- `atrisk::Vector{Float64}`: At-risk interval averages for this hazard

# Keyword Arguments  
- `alpha_bounds::Tuple{Float64,Float64}=(0.0, 2.0)`: Bounds on α search
- `tol::Float64=1e-2`: Convergence tolerance for α

# Returns
- `Float64`: Optimal α value

# Notes
- Uses Brent's method for 1D optimization (bracketed search)
- α=0 recovers uniform weighting (standard P-spline)
- α>2 rarely needed in practice
"""
function learn_alpha(model::MultistateProcess,
                     data,
                     penalty::QuadraticPenalty,
                     beta::Vector{Float64},
                     term_idx::Int,
                     hazard,
                     atrisk::Vector{Float64};
                     alpha_bounds::Tuple{Float64,Float64}=(0.0, 2.0),
                     tol::Float64=1e-2)
    
    # Get term info
    term = penalty.terms[term_idx]
    λ = term.lambda
    β_j = @view beta[term.hazard_indices]
    
    # Rebuild basis for this hazard
    basis = _rebuild_spline_basis(hazard)
    
    # Compute Hessian of log-likelihood at current beta (negative Fisher info)
    H = _compute_hessian_block(model, data, beta, term.hazard_indices)
    
    # Define objective: -1/2 log|H + λS_w(α)| + λ/2 β'S_w(α)β
    # We MINIMIZE this (corresponds to maximizing marginal likelihood)
    function objective(α)
        # Build weighted penalty matrix at this α
        weighting = AtRiskWeighting(alpha=α, learn=false)
        S_bspline = build_weighted_penalty_matrix(basis, term.order, weighting, atrisk)
        
        # Transform for monotone splines if needed
        S_w = if hazard.monotone != 0
            transform_penalty_for_monotone(S_bspline, basis; direction=hazard.monotone)
        else
            S_bspline
        end
        
        # Penalized Hessian: H + λS_w(α)
        H_pen = H + λ * S_w
        
        # Log determinant term (use eigenvalues for numerical stability)
        # Add small ridge for numerical stability
        eigvals = eigen(Symmetric(H_pen)).values
        # Replace near-zero eigenvalues with small positive value
        eigvals_safe = max.(eigvals, 1e-10)
        logdet_term = sum(log.(eigvals_safe))
        
        # Quadratic penalty term
        quad_term = λ * dot(β_j, S_w * β_j)
        
        return -0.5 * logdet_term + 0.5 * quad_term
    end
    
    # Brent's method for 1D minimization on [α_min, α_max]
    α_min, α_max = alpha_bounds
    
    # Use golden section search with Brent's method
    result = _brent_minimize(objective, α_min, α_max; tol=tol)
    
    return result
end

"""
    _brent_minimize(f, a, b; tol=1e-4, maxiter=100) -> Float64

Brent's method for 1D minimization on interval [a, b].

Combines golden section and parabolic interpolation for efficient convergence.

# Arguments
- `f`: Function to minimize
- `a, b`: Bracket endpoints (must contain minimum)
- `tol`: Convergence tolerance
- `maxiter`: Maximum iterations

# Returns
- Approximate minimizer x* ∈ [a, b]
"""
function _brent_minimize(f, a::Float64, b::Float64; tol::Float64=1e-4, maxiter::Int=100)
    # Golden ratio constant
    ϕ = 0.5 * (3.0 - sqrt(5.0))  # ≈ 0.382
    
    # Initialize
    x = w = v = a + ϕ * (b - a)
    fx = fw = fv = f(x)
    
    e = 0.0  # Distance moved on step before last
    d = 0.0  # Most recent step size
    
    for iter in 1:maxiter
        midpoint = 0.5 * (a + b)
        tol1 = tol * abs(x) + 1e-10
        tol2 = 2.0 * tol1
        
        # Check convergence
        if abs(x - midpoint) <= tol2 - 0.5 * (b - a)
            return x
        end
        
        # Try parabolic interpolation
        if abs(e) > tol1
            # Fit parabola
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            
            if q > 0
                p = -p
            else
                q = -q
            end
            
            r = e
            e = d
            
            # Check if parabolic step is acceptable
            if abs(p) < abs(0.5 * q * r) && p > q * (a - x) && p < q * (b - x)
                # Take parabolic step
                d = p / q
                u = x + d
                
                # f must not be evaluated too close to a or b
                if (u - a) < tol2 || (b - u) < tol2
                    d = x < midpoint ? tol1 : -tol1
                end
            else
                # Take golden section step
                e = (x < midpoint ? b : a) - x
                d = ϕ * e
            end
        else
            # Take golden section step
            e = (x < midpoint ? b : a) - x
            d = ϕ * e
        end
        
        # Compute new trial point
        if abs(d) >= tol1
            u = x + d
        else
            u = x + (d > 0 ? tol1 : -tol1)
        end
        
        fu = f(u)
        
        # Update state
        if fu <= fx
            if u < x
                b = x
            else
                a = x
            end
            v, fv = w, fw
            w, fw = x, fx
            x, fx = u, fu
        else
            if u < x
                a = u
            else
                b = u
            end
            
            if fu <= fw || w == x
                v, fv = w, fw
                w, fw = u, fu
            elseif fu <= fv || v == x || v == w
                v, fv = u, fu
            end
        end
    end
    
    # Return best found even if maxiter reached
    return x
end

"""
    _compute_hessian_block(model, data::ExactData, beta, indices) -> Matrix{Float64}

Compute the Hessian block for a subset of parameters (for alpha learning).

Returns the negative Hessian of the UNPENALIZED log-likelihood with respect
to parameters at `indices`.
"""
function _compute_hessian_block(model::MultistateProcess, 
                                 data::ExactData, 
                                 beta::Vector{Float64},
                                 indices::UnitRange{Int})
    K = length(indices)
    
    # Extract sub-parameters
    β_sub = beta[indices]
    
    # Wrapper that maps sub-parameters to full likelihood
    # Must create β_full with element type matching β_k for AD compatibility
    function loglik_wrapper(β_k)
        T = eltype(β_k)
        β_full = Vector{T}(undef, length(beta))
        copyto!(β_full, beta)  # Copy Float64 values, promoted to T
        β_full[indices] = β_k
        return -loglik_exact(β_full, data; neg=false)  # Return negative log-lik
    end
    
    # Compute Hessian using ForwardDiff
    H = ForwardDiff.hessian(loglik_wrapper, β_sub)
    
    return H
end

"""
    _compute_hessian_block(model, data::MPanelData, beta, indices) -> Matrix{Float64}

Compute the Hessian block for Markov panel data.
"""
function _compute_hessian_block(model::MultistateProcess, 
                                 data::MPanelData, 
                                 beta::Vector{Float64},
                                 indices::UnitRange{Int})
    K = length(indices)
    
    β_sub = beta[indices]
    
    # Must create β_full with element type matching β_k for AD compatibility
    function loglik_wrapper(β_k)
        T = eltype(β_k)
        β_full = Vector{T}(undef, length(beta))
        copyto!(β_full, beta)  # Copy Float64 values, promoted to T
        β_full[indices] = β_k
        return _loglik_markov_mutating(β_full, data; neg=true, return_ll_subj=false)
    end
    
    H = ForwardDiff.hessian(loglik_wrapper, β_sub)
    
    return H
end

"""
    needs_alpha_learning(penalty_specs::Union{Nothing, SplinePenalty, Vector{SplinePenalty}}) -> Bool

Check if any penalty specification requires alpha learning.

Returns true if at least one SplinePenalty has `AtRiskWeighting` with `learn=true`.
"""
function needs_alpha_learning(penalty_specs::Union{Nothing, SplinePenalty, Vector{SplinePenalty}})
    isnothing(penalty_specs) && return false
    
    specs = penalty_specs isa SplinePenalty ? [penalty_specs] : penalty_specs
    
    for spec in specs
        if spec.weighting isa AtRiskWeighting && spec.weighting.learn
            return true
        end
    end
    
    return false
end

"""
    update_penalty_with_alpha(penalty::QuadraticPenalty,
                               model::MultistateProcess,
                               term_idx::Int,
                               alpha::Float64,
                               hazard,
                               atrisk::Vector{Float64}) -> QuadraticPenalty

Create new penalty config with updated penalty matrix for new alpha value.

# Arguments
- `penalty`: Current penalty configuration
- `model`: Model with hazard definitions
- `term_idx`: Index of term to update
- `alpha`: New alpha value
- `hazard`: Hazard for this term
- `atrisk`: At-risk interval averages

# Returns
- New `QuadraticPenalty` with updated S matrix for specified term
"""
function update_penalty_with_alpha(penalty::QuadraticPenalty,
                                    model::MultistateProcess,
                                    term_idx::Int,
                                    alpha::Float64,
                                    hazard,
                                    atrisk::Vector{Float64})
    term = penalty.terms[term_idx]
    
    # Rebuild basis
    basis = _rebuild_spline_basis(hazard)
    
    # Build new weighted penalty matrix
    weighting = AtRiskWeighting(alpha=alpha, learn=false)
    S_bspline = build_weighted_penalty_matrix(basis, term.order, weighting, atrisk)
    
    # Transform for monotone splines if needed
    S_new = if hazard.monotone != 0
        transform_penalty_for_monotone(S_bspline, basis; direction=hazard.monotone)
    else
        S_bspline
    end
    
    # Create new terms vector
    new_terms = copy(penalty.terms)
    new_terms[term_idx] = PenaltyTerm(
        term.hazard_indices,
        S_new,
        term.lambda,
        term.order,
        term.hazard_names
    )
    
    return QuadraticPenalty(
        new_terms,
        penalty.total_hazard_terms,
        penalty.smooth_covariate_terms,
        penalty.shared_lambda_groups,
        penalty.shared_smooth_groups,
        penalty.n_lambda
    )
end

"""
    collect_alpha_learning_info(model::MultistateProcess,
                                 penalty::QuadraticPenalty,
                                 penalty_specs::Union{SplinePenalty, Vector{SplinePenalty}}) -> Dict{Int, AlphaLearningInfo}

Collect precomputed information for all terms that need alpha learning.

Returns a Dict mapping term_idx to AlphaLearningInfo struct.

# Arguments
- `model`: Model with hazard definitions
- `penalty`: Penalty configuration
- `penalty_specs`: Original SplinePenalty specifications

# Returns  
- Dict{Int, AlphaLearningInfo}: Maps term index to learning info

# Notes
- Only includes terms where the SplinePenalty has AtRiskWeighting with learn=true
- Precomputes at-risk interval averages for efficiency
- Handles share_lambda grouping (grouped terms share the same alpha)
"""
function collect_alpha_learning_info(model::MultistateProcess,
                                      penalty::QuadraticPenalty,
                                      penalty_specs::Union{SplinePenalty, Vector{SplinePenalty}})
    specs = penalty_specs isa SplinePenalty ? [penalty_specs] : penalty_specs
    result = Dict{Int, AlphaLearningInfo}()
    
    # Build map from hazard name to (spec, hazard_idx)
    # We need to match penalty terms to their original specs
    for (term_idx, term) in enumerate(penalty.terms)
        # Find the hazard for this term
        for hazname in term.hazard_names
            # Find spec that matches this hazard
            haz_idx = findfirst(h -> h isa _SplineHazard && h.hazname == hazname, model.hazards)
            isnothing(haz_idx) && continue
            
            hazard = model.hazards[haz_idx]
            origin = hazard.statefrom
            dest = hazard.stateto
            
            # Find matching spec
            for spec in specs
                if _rule_matches(spec.selector, origin, dest)
                    if spec.weighting isa AtRiskWeighting && spec.weighting.learn
                        # This term needs alpha learning
                        transition = (origin, dest)
                        
                        # Compute at-risk interval averages
                        atrisk = compute_atrisk_interval_averages(model, hazard, transition)
                        
                        result[term_idx] = AlphaLearningInfo(
                            haz_idx,
                            hazard,
                            transition,
                            term.order,
                            atrisk,
                            term.hazard_indices
                        )
                        break
                    end
                end
            end
            break  # Only process first hazard name (they share same settings if grouped)
        end
    end
    
    return result
end

"""
    get_shared_alpha_groups(penalty::QuadraticPenalty,
                            alpha_info::Dict{Int, AlphaLearningInfo}) -> Vector{Vector{Int}}

Get groups of term indices that should share the same alpha.

Terms share alpha when:
1. They are in the same share_lambda group, AND
2. They all need alpha learning

# Arguments
- `penalty`: Penalty configuration (contains shared_lambda_groups)
- `alpha_info`: Dict mapping term_idx to AlphaLearningInfo

# Returns
- Vector of groups, each group is a Vector{Int} of term indices sharing alpha
- Terms not in any group are returned as singleton groups
"""
function get_shared_alpha_groups(penalty::QuadraticPenalty,
                                  alpha_info::Dict{Int, AlphaLearningInfo})
    # Start with all alpha-learning terms as singletons
    term_indices = Set(keys(alpha_info))
    grouped = Set{Int}()
    groups = Vector{Vector{Int}}()
    
    # Check shared_lambda_groups
    for (origin, group_term_indices) in penalty.shared_lambda_groups
        # Find which of these terms need alpha learning
        alpha_terms = [idx for idx in group_term_indices if idx in term_indices]
        
        if length(alpha_terms) > 1
            # These terms share alpha
            push!(groups, alpha_terms)
            for idx in alpha_terms
                push!(grouped, idx)
            end
        end
    end
    
    # Add remaining terms as singletons
    for idx in term_indices
        if !(idx in grouped)
            push!(groups, [idx])
        end
    end
    
    return groups
end
