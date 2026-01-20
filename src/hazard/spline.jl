# =============================================================================
# Spline Penalty Infrastructure (Phase 1: Baseline Hazard Splines)
# =============================================================================
#
# This section implements penalized spline support following the design in
# scratch/penalized_splines_plan.md. Key components:
#
# 1. SplineHazardInfo: Internal struct holding penalty information per hazard
# 2. build_penalty_matrix: Wood's (2016) derivative-based penalty construction
# 3. place_interior_knots_pooled: Shared knot placement for competing risks
#
# The penalty structure follows the likelihood-motivated decomposition from
# scratch/penalized_splines_literature_review.md.
# =============================================================================

# =============================================================================
# Spline Hazard Detection
# =============================================================================

"""
    has_spline_hazards(model::MultistateProcess) -> Bool

Check if the model contains any spline hazards.

This function is used internally to determine whether automatic penalization
should be applied when `penalty=:auto` is specified in `fit()`.

# Arguments
- `model::MultistateProcess`: The multistate model to check

# Returns
- `true` if any hazard in the model is a spline hazard (`RuntimeSplineHazard`)
- `false` otherwise

# Examples
```julia
# Model with spline hazard
h12 = Hazard(@formula(0 ~ 1), :sp, 1, 2)
model_sp = multistatemodel(h12; data=data)
has_spline_hazards(model_sp)  # true

# Model with parametric hazard
h12 = Hazard(@formula(0 ~ 1), :wei, 1, 2)
model_wei = multistatemodel(h12; data=data)
has_spline_hazards(model_wei)  # false
```

See also: [`SplinePenalty`](@ref), [`fit`](@ref)
"""
function has_spline_hazards(model::MultistateProcess)
    return any(h -> h isa _SplineHazard, model.hazards)
end

# =============================================================================
# Spline Hazard Info
# =============================================================================

"""
    SplineHazardInfo

Internal struct holding penalty-related information for a spline hazard.
Created during model construction and stored for use in penalized likelihood
evaluation and smoothing parameter selection.

# Fields
- `origin::Int`: Origin state number
- `dest::Int`: Destination state number  
- `nbasis::Int`: Number of basis functions (K)
- `breakpoints::Vector{Float64}`: Knot locations (boundary + interior)
- `basis`: BSplineKit basis object (BSplineBasis or RecombinedBSplineBasis)
- `S::Matrix{Float64}`: Penalty matrix ∫B^(m)(t)B^(m)(t)ᵀdt
- `penalty_order::Int`: Derivative order for penalty (default 2 = curvature)

# Notes
- The penalty matrix `S` is positive semi-definite with null space consisting of
  polynomials of degree < `penalty_order`
- For shared penalty (competing risks), all hazards from the same origin share
  identical `breakpoints` and compatible `S` matrices
"""
struct SplineHazardInfo{B<:Union{BSplineBasis, RecombinedBSplineBasis}}
    origin::Int
    dest::Int
    nbasis::Int
    breakpoints::Vector{Float64}
    basis::B  # Parametric for type stability
    S::Matrix{Float64}
    penalty_order::Int
    
    function SplineHazardInfo(origin::Int, dest::Int, nbasis::Int,
                               breakpoints::Vector{Float64}, basis::B,
                               S::Matrix{Float64}, penalty_order::Int) where {B<:Union{BSplineBasis, RecombinedBSplineBasis}}
        # Validate dimensions
        nbasis > 0 || throw(ArgumentError("nbasis must be positive, got $nbasis"))
        penalty_order >= 1 || throw(ArgumentError("penalty_order must be ≥ 1, got $penalty_order"))
        size(S, 1) == nbasis && size(S, 2) == nbasis || 
            throw(ArgumentError("S must be $nbasis × $nbasis, got $(size(S))"))
        length(breakpoints) >= 2 || 
            throw(ArgumentError("breakpoints must have at least 2 elements"))
        
        # Validate S is symmetric (handle zero matrix case where norm(S) == 0)
        norm_S = norm(S)
        tol = norm_S > 0 ? 1e-10 * norm_S : 1e-15
        norm(S - S') < tol || 
            throw(ArgumentError("Penalty matrix S must be symmetric"))
        
        new{B}(origin, dest, nbasis, breakpoints, basis, S, penalty_order)
    end
end

# build_penalty_matrix and helpers moved to utilities/spline_utils.jl

"""
    place_interior_knots_pooled(model::MultistateProcess, origin::Int, nknots::Int;
                                 lower_bound::Real=0.0) -> Vector{Float64}

For a B-spline basis of order `m₁` and penalty order `m₂`, computes:
    S_{ij} = ∫ B_i^(m₂)(t) B_j^(m₂)(t) dt

The algorithm uses Gauss-Legendre quadrature on each knot interval with enough
points to integrate the product exactly.

# Arguments
- `basis`: A `BSplineBasis` or `RecombinedBSplineBasis` from BSplineKit
- `order::Int`: Derivative order (m₂). Default is 2 (curvature penalty)
- `knots::Vector{Float64}`: Explicit knot vector (used if basis is recombined)

# Returns
- `S::Matrix{Float64}`: Symmetric positive semi-definite penalty matrix

# Properties of S
- Symmetric: S = Sᵀ
- Positive semi-definite: xᵀSx ≥ 0 for all x
- Null space: polynomials of degree < order (dim = order)
- Banded: bandwidth = 2(m₁ - 1) + 1 for B-splines

# Example
```julia
knots = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
basis = BSplineBasis(BSplineOrder(4), knots)  # Cubic
S = build_penalty_matrix(basis, 2)  # Curvature penalty

# Verify null space contains constants and linears
v_const = ones(length(basis))
v_linear = collect(1.0:length(basis))
@assert v_const' * S * v_const < 1e-10
@assert v_linear' * S * v_linear < 1e-10
```

# References
- Wood, S.N. (2016). "P-splines with derivative based penalties and tensor 
  product smoothing of unevenly distributed data." Statistics and Computing.
"""
function place_interior_knots_pooled(model::MultistateProcess, origin::Int, nknots::Int;
                                      lower_bound::Real=0.0)
    nknots <= 0 && return Float64[]
    
    # Validate origin state
    tmat = model.tmat
    n_states = size(tmat, 1)
    1 <= origin <= n_states || 
        throw(ArgumentError("Origin state $origin is out of range [1, $n_states]"))
    
    # Find all destination states from this origin
    destinations = findall(j -> tmat[origin, j] != 0, 1:n_states)
    isempty(destinations) && 
        throw(ArgumentError("Origin state $origin has no outgoing transitions"))
    
    # Pool sojourn times from all transitions from this origin
    pooled_sojourns = Float64[]
    
    # Check if we have exact data
    has_exact = _has_exact_transitions_from_origin(model, origin)
    
    if has_exact
        # Extract sojourns from exact data
        data = model.data
        for row in eachrow(data)
            obstype = row.obstype
            # Only exact observations have known transition times
            if obstype != OBSTYPE_EXACT
                continue
            end
            
            statefrom = row.statefrom
            stateto = row.stateto
            
            # Only transitions from this origin
            if statefrom != origin || statefrom == stateto
                continue
            end
            
            sojourn = row.tstop - row.tstart
            push!(pooled_sojourns, sojourn)
        end
    else
        # Use surrogate simulation (delegate to existing infrastructure)
        sojourns_by_transition = _extract_sojourns_from_surrogate(model, 1000, 100)
        for dest in destinations
            key = (origin, dest)
            if haskey(sojourns_by_transition, key)
                append!(pooled_sojourns, sojourns_by_transition[key])
            end
        end
    end
    
    # Fall back to uniform if no data
    if isempty(pooled_sojourns)
        @warn "No transitions from origin state $origin; using uniform knots on [0, 10]"
        return collect(range(1.0, 9.0, length=nknots))
    end
    
    # Use standard knot placement on pooled sojourns
    upper_bound = maximum(pooled_sojourns)
    return place_interior_knots(pooled_sojourns, nknots; 
                                 lower_bound=lower_bound, upper_bound=upper_bound)
end

"""
    _has_exact_transitions_from_origin(model::MultistateProcess, origin::Int) -> Bool

Check if model data contains any exact transitions from the given origin state.
"""
function _has_exact_transitions_from_origin(model::MultistateProcess, origin::Int)
    data = model.data
    for row in eachrow(data)
        obstype = row.obstype
        # Only exact observations (obstype=1) have known transition times
        if obstype == OBSTYPE_EXACT && 
           row.statefrom == origin && 
           row.statefrom != row.stateto
            return true
        end
    end
    return false
end

"""
    validate_shared_knots(model::MultistateProcess, origin::Int) -> Bool

Validate that all spline hazards from the given origin state have identical knots.

This is required for the Kronecker product penalty structure when `shared_origin_tensor`
or `share_lambda=true` is specified. Throws `ArgumentError` if knots differ.

# Arguments
- `model::MultistateProcess`: The model to validate
- `origin::Int`: The origin state to check

# Returns
- `true` if all spline hazards from origin have identical knots (or only one exists)

# Throws
- `ArgumentError` if spline hazards from this origin have different knot locations
"""
function validate_shared_knots(model::MultistateProcess, origin::Int)
    # Find all spline hazards from this origin
    spline_hazards = filter(h -> h isa RuntimeSplineHazard && h.statefrom == origin, 
                            model.hazards)
    
    length(spline_hazards) <= 1 && return true
    
    # Compare knots
    reference_knots = spline_hazards[1].knots
    
    for (i, haz) in enumerate(spline_hazards[2:end])
        if haz.knots != reference_knots
            ref_name = spline_hazards[1].hazname
            this_name = haz.hazname
            throw(ArgumentError(
                "Spline hazards from origin state $origin have mismatched knots. " *
                "Hazard $ref_name has knots $(round.(reference_knots, digits=3)), " *
                "but hazard $this_name has knots $(round.(haz.knots, digits=3)). " *
                "When using shared penalty (share_lambda=true or shared_origin_tensor), " *
                "all competing hazards must have identical knot locations."
            ))
        end
    end
    
    return true
end

"""
    build_spline_hazard_info(hazard::RuntimeSplineHazard; penalty_order::Int=2) -> SplineHazardInfo

Construct SplineHazardInfo for a RuntimeSplineHazard.

This creates the penalty matrix and stores all information needed for
penalized likelihood evaluation.

# Arguments
- `hazard::RuntimeSplineHazard`: The hazard to create info for
- `penalty_order::Int=2`: Derivative order for penalty (2 = curvature)

# Returns
- `SplineHazardInfo`: Struct containing penalty matrix and basis information

# Notes
For monotone splines (hazard.monotone ≠ 0), the penalty matrix is automatically
transformed from B-spline coefficient space to I-spline parameter space using
`transform_penalty_for_monotone`. This ensures the penalty correctly penalizes
the curvature of the underlying function, not the raw optimization parameters.
"""
function build_spline_hazard_info(hazard::RuntimeSplineHazard; penalty_order::Int=2)
    # Rebuild basis from hazard (same logic as _rebuild_spline_basis)
    basis = _rebuild_spline_basis(hazard)
    
    # Build penalty matrix for B-spline coefficients
    # NOTE: Don't pass hazard.knots - it contains simple breakpoints, not the full
    # clamped knot vector. Let build_penalty_matrix extract knots from the basis.
    S_bspline = build_penalty_matrix(basis, penalty_order)
    
    # For monotone splines, transform penalty to I-spline parameter space
    # P(ests) = (λ/2) ests' (L' S L) ests where coefs = L * ests
    if hazard.monotone != 0
        S = transform_penalty_for_monotone(S_bspline, basis; direction=hazard.monotone)
    else
        S = S_bspline
    end
    
    return SplineHazardInfo(
        hazard.statefrom,
        hazard.stateto,
        hazard.npar_baseline,
        hazard.knots,
        basis,
        S,
        penalty_order
    )
end

# =============================================================================
# End Penalty Infrastructure
# =============================================================================

"""
    place_interior_knots(sojourns::AbstractVector{<:Real}, nknots::Integer; 
                         lower_bound::Real=0.0, upper_bound::Real=Inf)

Place interior knots at quantiles of sojourn times, following survextrap convention.

This function computes knot locations at evenly-spaced quantiles of the observed 
sojourn times. On the clock-reset (semi-Markov) scale, there may be ties in sojourn 
times. When quantile locations are tied, knots are equally spaced between unique 
quantile values to avoid degenerate spline bases.

# Arguments
- `sojourns`: Vector of sojourn times (reset scale)
- `nknots`: Number of interior knots to place (default: 10 following survextrap)
- `lower_bound`: Lower boundary knot (default: 0.0)
- `upper_bound`: Upper boundary knot (default: maximum sojourn)

# Returns
- Vector of interior knot locations (excluding boundary knots)

# Notes
- Quantile levels are spaced evenly on (0, 1): k/(nknots+1) for k = 1, ..., nknots
- When ties occur, knots are equally spaced between unique quantile values
- Returns empty vector if nknots ≤ 0 or insufficient unique sojourns
"""
function place_interior_knots(sojourns::AbstractVector{<:Real}, nknots::Integer;
                              lower_bound::Real=0.0, upper_bound::Real=Inf)
    
    # Handle edge cases
    nknots <= 0 && return Float64[]
    isempty(sojourns) && return Float64[]
    
    # Use finite upper bound
    ub = isfinite(upper_bound) ? upper_bound : maximum(sojourns)
    lb = lower_bound
    
    # Compute quantile levels: evenly spaced on (0, 1)
    quantile_levels = [(k / (nknots + 1)) for k in 1:nknots]
    
    # Get raw quantile locations
    raw_knots = quantile(sojourns, quantile_levels)
    
    # Handle ties: unique quantile values, then interpolate if needed
    unique_raw = unique(raw_knots)
    
    if length(unique_raw) == nknots
        # No ties - use raw quantiles directly
        knots = raw_knots
    elseif length(unique_raw) == 1
        # All quantiles are tied to single value - spread evenly in interval
        # This can happen with very sparse data
        mid = unique_raw[1]
        if mid > lb && mid < ub
            # Spread nknots evenly between lb and ub (interior only)
            knots = collect(range(lb + (ub - lb)/(nknots + 1), 
                                  stop=ub - (ub - lb)/(nknots + 1), 
                                  length=nknots))
        else
            knots = collect(range(lb + (ub - lb)/(nknots + 1),
                                  stop=ub - (ub - lb)/(nknots + 1),
                                  length=nknots))
        end
    else
        # Some ties - interpolate between unique values
        # Strategy: preserve unique quantile locations, fill gaps evenly
        knots = Float64[]
        sizehint!(knots, nknots)
        
        # Group consecutive raw_knots by value
        i = 1
        while i <= nknots
            current_val = raw_knots[i]
            
            # Count how many consecutive knots have same value
            j = i
            while j <= nknots && raw_knots[j] ≈ current_val
                j += 1
            end
            n_tied = j - i
            
            if n_tied == 1
                # No tie - use as-is
                push!(knots, current_val)
            else
                # Tied knots: determine interval for spacing
                left_bound = i == 1 ? lb : raw_knots[i - 1]
                right_bound = j > nknots ? ub : raw_knots[j]
                
                # Equally space n_tied knots between left and right bounds
                if left_bound < current_val && current_val < right_bound
                    # Current value is in interior - spread around it
                    spacing = (right_bound - left_bound) / (n_tied + 1)
                    for k in 1:n_tied
                        push!(knots, left_bound + k * spacing)
                    end
                else
                    # Degenerate case - just use current value
                    for _ in 1:n_tied
                        push!(knots, current_val)
                    end
                end
            end
            
            i = j
        end
        
        # Ensure uniqueness after interpolation
        unique!(sort!(knots))
        
        # If we lost some knots due to ties, fill with evenly spaced
        if length(knots) < nknots
            # Add more evenly spaced knots
            remaining = nknots - length(knots)
            gaps = Float64[]
            
            # Find largest gaps
            all_points = sort(vcat([lb], knots, [ub]))
            for i in 1:(length(all_points) - 1)
                push!(gaps, all_points[i + 1] - all_points[i])
            end
            
            # Add knots in largest gaps
            gap_order = sortperm(gaps, rev=true)
            for g in gap_order[1:min(remaining, length(gap_order))]
                new_knot = (all_points[g] + all_points[g + 1]) / 2
                push!(knots, new_knot)
            end
            
            sort!(knots)
        end
    end
    
    # Clamp to (lower_bound, upper_bound) - knots must be strictly interior
    filter!(k -> lb < k < ub, knots)
    
    return knots
end

"""
    default_nknots(n_observations::Integer)

Uses floor(n^(1/5)) for sieve estimation (unpenalized splines).

For typical survival data sizes:
- n = 100: 2-3 knots
- n = 500: 3-4 knots  
- n = 1000: 4 knots
- n = 10000: 6 knots

For small samples, returns at least 2 knots for flexibility.

**Note**: This formula is appropriate for **unpenalized** (regression) splines
where the number of knots controls model complexity. For penalized splines,
use `default_nknots_penalized()` instead.

survextrap uses a fixed 10 knots by default.

See also: [`default_nknots_penalized`](@ref)
"""
function default_nknots(n_observations::Integer)
    n_observations <= 0 && return 0
    return max(2, floor(Int, n_observations^(1/5)))
end

"""
    default_nknots_penalized(n_observations::Integer)

Default number of interior knots for penalized (P-spline) estimation.

Uses floor(n^(1/3)) following recommendations in the smoothing literature.
For penalized splines, the penalty controls overfitting, so more knots are 
acceptable (and often desirable) for flexibility. The key is having enough 
knots to capture the underlying shape—the penalty prevents overfitting.

For typical survival data sizes:
- n = 100: 4 knots
- n = 500: 7 knots
- n = 1000: 10 knots
- n = 10000: 21 knots

Bounded to [4, 40] knots: at least 4 for flexibility, at most 40 to avoid
computational overhead.

# References
- Ruppert, D. (2002). "Selecting the number of knots for penalized splines."
  JCGS 11(4), 735-757.
- Wood, S.N. (2017). "Generalized Additive Models: An Introduction with R."
  2nd ed. Chapman & Hall/CRC, Chapter 5.

See also: [`default_nknots`](@ref)
"""
function default_nknots_penalized(n_observations::Integer)
    n_observations <= 0 && return 0
    return clamp(floor(Int, n_observations^(1/3)), 4, 40)
end

# =============================================================================
# Spline Knot Calibration Functions
# =============================================================================

"""
    calibrate_splines(model::MultistateProcess; 
                      quantiles=nothing, 
                      nknots=nothing,
                      n_paths::Int=1000,
                      min_ess::Int=100,
                      n_grid::Int=1000,
                      use_cdf_inversion::Bool=true,
                      verbose::Bool=true) -> NamedTuple

Compute recommended knot locations for spline hazards based on exit time quantiles.

For exact data (obstype 1 or 3), uses observed sojourn times pooled across all
destinations from each origin state (shared knots for competing hazards).

For panel data (obstype 2), computes cumulative incidence at reference covariate
level (x=0) and numerically inverts the exit CDF to get quantiles. This approach
is deterministic and avoids Monte Carlo noise from path simulation.

# Arguments
- `model::MultistateProcess`: An unfitted model with one or more spline hazards.
- `quantiles`: Interior knot quantile levels. Can be:
  - `Vector{Float64}`: Quantile levels (e.g., `[0.25, 0.5, 0.75]`) applied to all hazards
  - `NamedTuple`: Per-hazard quantiles, e.g., `(h12 = [0.25, 0.5], h23 = [0.33, 0.67])`
  - `nothing` (default): Use `nknots` to generate evenly-spaced quantiles
- `nknots`: Number of interior knots. Can be:
  - `Int`: Number of interior knots for all hazards
  - `NamedTuple`: Per-hazard counts, e.g., `(h12 = 3, h23 = 2)`
  - `nothing` (default): Use `floor(n^(1/5))`
- `n_paths::Int=1000`: Number of paths to sample per subject (simulation fallback only)
- `min_ess::Int=100`: Minimum effective sample size (simulation fallback only)
- `n_grid::Int=1000`: Number of points in time grid for CDF evaluation (panel data)
- `use_cdf_inversion::Bool=true`: Use deterministic CDF inversion instead of simulation
  for panel data. When `true` (default), knots are placed at quantiles of the exit time
  distribution computed at reference covariate level. When `false`, uses legacy
  simulation-based approach.
- `verbose::Bool=true`: Print info about knot placement

# Returns
A `NamedTuple` with one entry per spline hazard, each containing:
- `boundary_knots::Vector{Float64}`: `[0.0, max_time]` for the transition
- `interior_knots::Vector{Float64}`: Recommended interior knot locations

# Errors
- `ArgumentError` if model is a `MultistateModelFitted`
- `ArgumentError` if model has no spline hazards
- `ArgumentError` if both `quantiles` and `nknots` are specified

# Notes
- All hazards from the same origin state share the same knots (exit time distribution is shared)
- For exact data: uses observed sojourn times directly (pooled across destinations)
- For panel data with `use_cdf_inversion=true`: computes cumulative incidence at reference
  level and inverts CDF (deterministic, no Monte Carlo noise)
- For panel data with `use_cdf_inversion=false`: fits surrogate, simulates paths (legacy)
- Lower boundary is always 0.0 (sojourns are non-negative)
- The surrogate must be fitted before calling for panel data with CDF inversion

# Example
```julia
h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; degree=3)
h23 = Hazard(@formula(0 ~ 1), "sp", 2, 3; degree=3)
model = multistatemodel(h12, h23; data=data)

# Auto-select nknots based on sample size (uses CDF inversion for panel data)
knots = calibrate_splines(model)

# Specify number of knots
knots = calibrate_splines(model; nknots=3)

# Different knots per hazard (Note: hazards from same origin will still share knots)
knots = calibrate_splines(model; nknots=(h12=4, h23=2))

# Specify explicit quantiles
knots = calibrate_splines(model; quantiles=[0.2, 0.4, 0.6, 0.8])

# Use legacy simulation approach instead of CDF inversion
knots = calibrate_splines(model; use_cdf_inversion=false)
```

See also: [`calibrate_splines!`](@ref), [`place_interior_knots`](@ref), 
          [`cumulative_incidence_at_reference`](@ref)
"""
function calibrate_splines(model::MultistateProcess;
                           quantiles::Union{Vector{Float64}, NamedTuple, Nothing}=nothing,
                           nknots::Union{Int, NamedTuple, Nothing}=nothing,
                           n_paths::Int=1000,
                           min_ess::Int=100,
                           n_grid::Int=1000,
                           use_cdf_inversion::Bool=true,
                           verbose::Bool=true)
    
    # Validation
    if model isa MultistateModelFitted
        throw(ArgumentError("Cannot calibrate splines on a fitted model. Use an unfitted MultistateModel."))
    end
    
    if !isnothing(quantiles) && !isnothing(nknots)
        throw(ArgumentError("Specify either `quantiles` or `nknots`, not both."))
    end
    
    # Find spline hazards
    spline_indices = findall(h -> h isa RuntimeSplineHazard, model.hazards)
    if isempty(spline_indices)
        throw(ArgumentError("Model has no spline hazards to calibrate."))
    end
    
    # Determine if data is exact or panel
    has_exact_data = _has_exact_transitions(model)
    
    # Group spline hazards by origin state for shared knot placement
    # Key insight: all hazards from same origin share same sojourn time distribution
    origin_to_hazards = Dict{Int, Vector{Int}}()
    for idx in spline_indices
        haz = model.hazards[idx]
        origin = haz.statefrom
        if !haskey(origin_to_hazards, origin)
            origin_to_hazards[origin] = Int[]
        end
        push!(origin_to_hazards[origin], idx)
    end
    
    # Compute knots per origin state
    results = Dict{Symbol, NamedTuple}()
    
    if has_exact_data
        # Use observed data directly
        verbose && @info "Using observed transition times for knot calibration"
        sojourns_by_transition = _extract_sojourns_from_data(model)
        
        # Pool sojourns by origin (for shared knots across competing hazards)
        for (origin, haz_indices) in origin_to_hazards
            # Pool all sojourns from this origin state
            pooled_sojourns = Float64[]
            for idx in haz_indices
                haz = model.hazards[idx]
                key = (haz.statefrom, haz.stateto)
                sojourns = get(sojourns_by_transition, key, Float64[])
                append!(pooled_sojourns, sojourns)
            end
            
            # Compute shared knots for all hazards from this origin
            _compute_shared_knots_for_origin!(
                results, model, origin, haz_indices, pooled_sojourns,
                nknots, quantiles, verbose
            )
        end
    else
        # Panel data: use CDF inversion (deterministic) or fallback to simulation
        if use_cdf_inversion
            verbose && @info "Using CDF inversion at reference level for knot calibration"
            
            # Ensure surrogate is fitted for cumulative incidence calculation
            if model.markovsurrogate === nothing || !is_fitted(model.markovsurrogate)
                verbose && @info "  Fitting surrogate for cumulative incidence computation..."
                set_surrogate!(model; type=:auto, verbose=false)
            end
            
            # Compute exit quantiles by origin via CDF inversion
            for (origin, haz_indices) in origin_to_hazards
                # Determine nknots for this origin (use first hazard's settings as representative)
                first_haz = model.hazards[haz_indices[1]]
                nk = _get_nknots_for_hazard(first_haz.hazname, nknots, nrow(model.data))
                
                # Get quantile levels (if user specified) or generate from nknots
                qlevels = _get_quantiles_for_hazard(first_haz.hazname, quantiles, nk)
                if qlevels === nothing
                    # Generate evenly spaced quantiles from nknots
                    qlevels = collect(range(1/(nk+1), 1 - 1/(nk+1), length=nk))
                end
                
                # Compute exit time quantiles via CDF inversion
                exit_quantiles = try
                    _compute_exit_quantiles_at_reference(
                        model, origin;
                        quantiles=qlevels,
                        n_grid=n_grid
                    )
                catch e
                    @warn "CDF inversion failed for origin $origin: $e. Falling back to simulation."
                    # Fallback: use old simulation-based approach
                    sojourns_by_transition = _extract_sojourns_from_surrogate(model, n_paths, min_ess)
                    pooled = Float64[]
                    for idx in haz_indices
                        haz = model.hazards[idx]
                        key = (haz.statefrom, haz.stateto)
                        append!(pooled, get(sojourns_by_transition, key, Float64[]))
                    end
                    isempty(pooled) ? qlevels .* 10.0 : quantile(pooled, qlevels)
                end
                
                # Compute boundary knots
                lb = 0.0
                ub = maximum(exit_quantiles) * 1.5  # Extend beyond last quantile
                ub = max(ub, maximum(model.data.tstop))  # At least cover observed data
                
                # Assign shared knots to all hazards from this origin
                for idx in haz_indices
                    haz = model.hazards[idx]
                    hazname = haz.hazname
                    verbose && @info "Calibrated $(length(exit_quantiles)) interior knots for $hazname from CDF: $(round.(exit_quantiles, digits=3))"
                    results[hazname] = (boundary_knots=[lb, ub], interior_knots=exit_quantiles)
                end
            end
        else
            # Legacy: simulation-based approach
            verbose && @info "Fitting Markov surrogate and simulating paths for knot calibration"
            sojourns_by_transition = _extract_sojourns_from_surrogate(model, n_paths, min_ess)
            
            for (origin, haz_indices) in origin_to_hazards
                # Pool all sojourns from this origin state
                pooled_sojourns = Float64[]
                for idx in haz_indices
                    haz = model.hazards[idx]
                    key = (haz.statefrom, haz.stateto)
                    sojourns = get(sojourns_by_transition, key, Float64[])
                    append!(pooled_sojourns, sojourns)
                end
                
                # Compute shared knots for all hazards from this origin
                _compute_shared_knots_for_origin!(
                    results, model, origin, haz_indices, pooled_sojourns,
                    nknots, quantiles, verbose
                )
            end
        end
    end
    
    return NamedTuple(results)
end

"""
    _compute_shared_knots_for_origin!(results, model, origin, haz_indices, pooled_sojourns, 
                                       nknots, quantiles, verbose)

Helper to compute shared knots for all hazards from a given origin state.
Mutates `results` dict in-place.
"""
function _compute_shared_knots_for_origin!(results::Dict{Symbol, NamedTuple},
                                            model::MultistateProcess,
                                            origin::Int,
                                            haz_indices::Vector{Int},
                                            pooled_sojourns::Vector{Float64},
                                            nknots::Union{Int, NamedTuple, Nothing},
                                            quantiles::Union{Vector{Float64}, NamedTuple, Nothing},
                                            verbose::Bool)
    # Use first hazard as representative for settings
    first_haz = model.hazards[haz_indices[1]]
    first_hazname = first_haz.hazname
    
    if isempty(pooled_sojourns)
        @warn "No transitions from state $origin; using model boundaries"
        bknots = [first_haz.knots[1], first_haz.knots[end]]
        for idx in haz_indices
            haz = model.hazards[idx]
            results[haz.hazname] = (boundary_knots=bknots, interior_knots=Float64[])
        end
        return
    end
    
    # Determine number of knots
    nk = _get_nknots_for_hazard(first_hazname, nknots, length(pooled_sojourns))
    
    # Determine quantile levels or compute from nknots
    qlevels = _get_quantiles_for_hazard(first_hazname, quantiles, nk)
    
    # Compute boundary knots
    lb = 0.0  # Lower boundary is always 0 for sojourns
    ub = maximum(pooled_sojourns)
    
    # Compute shared interior knots
    if qlevels !== nothing
        # User-specified quantiles
        interior = quantile(pooled_sojourns, qlevels)
        interior = unique(interior)
    else
        # Use place_interior_knots for automatic placement with tie handling
        interior = place_interior_knots(pooled_sojourns, nk; lower_bound=lb, upper_bound=ub)
    end
    
    # Assign same knots to all hazards from this origin
    for idx in haz_indices
        haz = model.hazards[idx]
        hazname = haz.hazname
        verbose && @info "Calibrated $(length(interior)) shared interior knots for $hazname at: $(round.(interior, digits=3))"
        results[hazname] = (boundary_knots=[lb, ub], interior_knots=interior)
    end
end

"""
    calibrate_splines!(model::MultistateProcess; quantiles=nothing, nknots=nothing, 
                       n_paths=1000, min_ess=100, n_grid=1000, use_cdf_inversion=true,
                       verbose=true)

Compute and apply knot locations for spline hazards in-place.

Modifies the spline hazards in `model` to use the computed knot locations.
See [`calibrate_splines`](@ref) for argument details.

# Arguments
Same as [`calibrate_splines`](@ref), plus:
- `n_grid::Int=1000`: Grid resolution for CDF inversion (panel data only)
- `use_cdf_inversion::Bool=true`: Use deterministic CDF inversion instead of simulation (panel data only)

# Returns
`NamedTuple` with computed knot locations (same as [`calibrate_splines`](@ref)).

# Notes
- Modifies `model.hazards` in place for all `RuntimeSplineHazard` hazards
- Also rebuilds the parameter structure to match the new number of basis functions
- After calling this, the model's spline hazards will have updated knots
- For panel data, CDF inversion provides deterministic knot placement without Monte Carlo noise

# Example
```julia
model = multistatemodel(h12, h23; data=data)
knots = calibrate_splines!(model; nknots=5)  # Updates knots in-place
fitted = fit(model)  # Fit with calibrated knots
```

See also: [`calibrate_splines`](@ref)
"""
function calibrate_splines!(model::MultistateProcess;
                            quantiles::Union{Vector{Float64}, NamedTuple, Nothing}=nothing,
                            nknots::Union{Int, NamedTuple, Nothing}=nothing,
                            n_paths::Int=1000,
                            min_ess::Int=100,
                            n_grid::Int=1000,
                            use_cdf_inversion::Bool=true,
                            verbose::Bool=true)
    
    # Get recommended knots (this also does validation)
    knots = calibrate_splines(model; quantiles=quantiles, nknots=nknots, 
                              n_paths=n_paths, min_ess=min_ess, n_grid=n_grid,
                              use_cdf_inversion=use_cdf_inversion, verbose=verbose)
    
    # Build knot_locations dict for _rebuild_model_with_knots!
    knot_locations = Dict{Tuple{Int,Int}, Vector{Float64}}()
    for idx in findall(h -> h isa RuntimeSplineHazard, model.hazards)
        haz = model.hazards[idx]
        result = knots[haz.hazname]
        knot_locations[(haz.statefrom, haz.stateto)] = result.interior_knots
    end
    
    # Rebuild model with new knots
    _rebuild_model_with_knots!(model, knot_locations)
    
    # Update model parameters to match new hazard dimensions
    _rebuild_model_parameters!(model)
    
    return knots
end

"""
    _rebuild_model_with_knots!(model::MultistateProcess, knot_locations::Dict{Tuple{Int,Int}, Vector{Float64}})

Rebuild spline hazards in-place with new interior knot locations.

This function creates new `RuntimeSplineHazard` objects with updated knots
and replaces them in the model's hazard vector. The boundary knots are preserved
from the original hazards.

# Arguments
- `model::MultistateProcess`: The model to modify
- `knot_locations::Dict{Tuple{Int,Int}, Vector{Float64}}`: Map from (statefrom, stateto) 
  to new interior knot locations

# Notes
- Only modifies `RuntimeSplineHazard` entries
- Preserves all other hazard properties (degree, natural_spline, monotone, etc.)
- After calling this, `_rebuild_model_parameters!` should be called to update parameters
"""
function _rebuild_model_with_knots!(model::MultistateProcess, 
                                     knot_locations::Dict{Tuple{Int,Int}, Vector{Float64}})
    
    for (idx, haz) in enumerate(model.hazards)
        # Only process spline hazards
        haz isa RuntimeSplineHazard || continue
        
        key = (haz.statefrom, haz.stateto)
        haskey(knot_locations, key) || continue
        
        new_interior_knots = knot_locations[key]
        
        # Preserve boundary knots from existing hazard
        old_knots = haz.knots
        bknots = [old_knots[1], old_knots[end]]
        
        # Combine and sort knots
        allknots = unique(sort([bknots[1]; new_interior_knots; bknots[2]]))
        
        # Build new B-spline basis with same settings as original
        B = BSplineBasis(BSplineOrder(haz.degree + 1), copy(allknots))
        
        # Apply boundary conditions via basis recombination (same as original)
        # Only enforce conditions at RIGHT boundary - no constraint at left (t=0)
        use_constant = haz.extrapolation == "constant"
        if use_constant && (haz.degree >= 2)
            B = RecombinedBSplineBasis(B, (), Derivative(1))  # free left, Neumann right
        elseif (haz.degree > 1) && haz.natural_spline
            B = RecombinedBSplineBasis(B, (), Derivative(2))  # free left, natural right
        end
        
        # Determine extrapolation method
        extrap_method = if haz.extrapolation == "linear"
            BSplineKit.SplineExtrapolations.Linear()
        else
            BSplineKit.SplineExtrapolations.Flat()
        end
        
        # Number of basis functions = number of spline coefficients
        nbasis = length(B)
        
        # Build new parameter names
        n_covar = length(haz.covar_names)
        baseline_names = [Symbol(string(haz.hazname), "_sp", i) for i in 1:nbasis]
        covar_parnames = [Symbol(string(haz.hazname), "_", cn) for cn in haz.covar_names]
        parnames = vcat(baseline_names, covar_parnames)
        npar_total = nbasis + n_covar
        
        # Generate new hazard and cumhaz functions
        hazard_fn, cumhaz_fn = _generate_spline_hazard_fns(
            B, extrap_method, haz.monotone, nbasis, parnames, haz.metadata.linpred_effect
        )
        
        # Create new RuntimeSplineHazard with updated knots
        new_haz = RuntimeSplineHazard(
            haz.hazname,
            haz.statefrom,
            haz.stateto,
            haz.family,
            parnames,
            nbasis,
            npar_total,
            hazard_fn,
            cumhaz_fn,
            haz.has_covariates,
            haz.covar_names,
            haz.degree,
            allknots,
            haz.natural_spline,
            haz.monotone,
            haz.extrapolation,
            haz.metadata,
            haz.shared_baseline_key,
            haz.smooth_info  # Preserve smooth term info from original hazard
        )
        
        # Replace in model
        model.hazards[idx] = new_haz
    end
    
    # Also update totalhazards if they reference spline hazards
    _rebuild_totalhazards!(model)
    
    return nothing
end

"""
    _rebuild_totalhazards!(model::MultistateProcess)

Rebuild the totalhazards structure after hazard modifications.
"""
function _rebuild_totalhazards!(model::MultistateProcess)
    # Rebuild totalhazards from scratch using current hazards
    model.totalhazards = build_totalhazards(model.hazards, model.tmat)
    return nothing
end

# Import _generate_spline_hazard_fns from modelgeneration - it's already available
# since smooths.jl is included after modelgeneration.jl

"""
    _extract_sojourns_from_data(model::MultistateProcess) -> Dict{Tuple{Int,Int}, Vector{Float64}}

Extract observed sojourn times for each transition from the model's data.

Returns dictionary mapping (statefrom, stateto) to vector of sojourn times.
Only includes exact transitions (obstype=1) where the transition time is known.
"""
function _extract_sojourns_from_data(model::MultistateProcess)
    data = model.data
    result = Dict{Tuple{Int,Int}, Vector{Float64}}()
    
    for row in eachrow(data)
        # Only exact transitions have known sojourn times
        obstype = row.obstype
        if obstype != OBSTYPE_EXACT
            continue
        end
        
        statefrom = row.statefrom
        stateto = row.stateto
        
        # Skip if no actual transition
        if statefrom == stateto
            continue
        end
        
        # Sojourn time
        sojourn = row.tstop - row.tstart
        
        key = (statefrom, stateto)
        if !haskey(result, key)
            result[key] = Float64[]
        end
        push!(result[key], sojourn)
    end
    
    return result
end

"""
    _has_exact_transitions(model::MultistateProcess) -> Bool

Check if the model data contains any exact transition observations (obstype=1).
"""
function _has_exact_transitions(model::MultistateProcess)
    data = model.data
    for row in eachrow(data)
        obstype = row.obstype
        # Only obstype=1 means exact observation with known transition time
        if obstype == OBSTYPE_EXACT && row.statefrom != row.stateto
            return true
        end
    end
    return false
end

"""
    _extract_sojourns_from_surrogate(model::MultistateProcess, n_paths::Int, min_ess::Int) 
        -> Dict{Tuple{Int,Int}, Vector{Float64}}

Fit a surrogate to the model, simulate sample paths, and extract sojourn times.

Uses the appropriate surrogate type based on the model's hazards:
- Markov surrogate for exponential hazards
- Phase-type surrogate for non-exponential hazards (Weibull, Gompertz, spline, etc.)

Used for panel data where exact transition times are not observed.

!!! warning "Deprecated"
    This function uses Monte Carlo simulation which introduces noise into knot placement.
    Prefer `_compute_exit_quantiles_at_reference` for deterministic knot placement via
    CDF inversion.
"""
function _extract_sojourns_from_surrogate(model::MultistateProcess, n_paths::Int, min_ess::Int)
    # Determine appropriate surrogate type based on hazards
    # Non-exponential hazards benefit from phase-type proposals
    surrogate_type = needs_phasetype_proposal(model.hazards) ? :phasetype : :markov
    
    # Fit and store surrogate in model
    set_surrogate!(model; type=surrogate_type, verbose=false)
    
    # Draw sample paths from the model (uses stored surrogate)
    path_result = draw_paths(model; min_ess=min_ess)
    
    # Extract paths - handle different return types from draw_paths
    # For panel data: returns NamedTuple with samplepaths::Vector{Vector{SamplePath}}
    # For exact data: returns NamedTuple with loglik (paths come from data)
    all_paths = if path_result isa NamedTuple && haskey(path_result, :samplepaths)
        # Panel data: NamedTuple with samplepaths field - flatten nested vectors
        SamplePath[p for subj_paths in path_result.samplepaths for p in subj_paths]
    elseif path_result isa NamedTuple && haskey(path_result, :loglik)
        # Exact data case (shouldn't happen here, but handle gracefully)
        extract_paths(model.data)
    elseif path_result isa Vector{<:Vector}
        # Direct Vector{Vector{SamplePath}} - flatten
        SamplePath[p for subj_paths in path_result for p in subj_paths]
    else
        # Already a flat vector
        path_result
    end
    
    # Extract sojourns for each transition
    result = Dict{Tuple{Int,Int}, Vector{Float64}}()
    
    # Get all possible transitions from tmat
    tmat = model.tmat
    n_states = size(tmat, 1)
    for i in 1:n_states
        for j in 1:n_states
            if tmat[i, j] != 0
                sojourns = extract_sojourns(i, j, all_paths)
                if !isempty(sojourns)
                    result[(i, j)] = sojourns
                end
            end
        end
    end
    
    return result
end

"""
    _compute_exit_quantiles_at_reference(model::MultistateProcess, statefrom::Int;
                                         quantiles::Vector{Float64}=[0.25, 0.5, 0.75],
                                         t_max::Union{Nothing, Float64}=nothing,
                                         n_grid::Int=1000) -> Vector{Float64}

Compute exit time quantiles from a given state at reference covariate level (x=0).

This function uses numerical CDF inversion rather than Monte Carlo simulation,
providing deterministic and noise-free knot placement. The exit time distribution
is computed from the cumulative incidence at reference covariate values.

# Arguments
- `model::MultistateProcess`: The multistate model (with fitted surrogate if panel data)
- `statefrom::Int`: Origin state for which to compute exit time quantiles
- `quantiles::Vector{Float64}=[0.25, 0.5, 0.75]`: Quantile levels to compute
- `t_max::Union{Nothing, Float64}=nothing`: Maximum time for grid. If `nothing`,
  uses `2 * max(tstop)` from data (with fallback to 100.0)
- `n_grid::Int=1000`: Number of points in time grid for CDF evaluation

# Returns
Vector of exit times corresponding to each requested quantile.

# Notes
- Exit time = time until leaving `statefrom` via any transition (competing risks)
- Quantiles are computed from the total exit CDF: `F(t) = 1 - S(t)` where
  `S(t) = exp(-∫₀ᵗ Λ(u) du)` is survival and `Λ(u)` is total hazard from state
- For panel data, the surrogate's fitted parameters are used (via `cumulative_incidence_at_reference`)
- Uses linear interpolation to invert the CDF

# Algorithm
1. Evaluate cumulative incidence of exit on fine time grid at reference (x=0)
2. Total exit CDF = sum of cause-specific cumulative incidences across destinations
3. For each quantile level q, find t such that F(t) = q via linear interpolation

# Example
```julia
# Get median and quartiles of exit time from state 1
q = _compute_exit_quantiles_at_reference(model, 1; quantiles=[0.25, 0.5, 0.75])
# q[1] = 25th percentile, q[2] = median, q[3] = 75th percentile
```

See also: [`cumulative_incidence_at_reference`](@ref), [`calibrate_splines`](@ref)
"""
function _compute_exit_quantiles_at_reference(model::MultistateProcess, statefrom::Int;
                                               quantiles::Vector{Float64}=[0.25, 0.5, 0.75],
                                               t_max::Union{Nothing, Float64}=nothing,
                                               n_grid::Int=1000)
    # Validate quantiles
    all(0 < q < 1 for q in quantiles) || 
        throw(ArgumentError("All quantiles must be in (0, 1), got $quantiles"))
    
    # Determine t_max if not specified
    if t_max === nothing
        max_tstop = maximum(model.data.tstop)
        t_max = 2.0 * max_tstop  # Extend beyond observed data
        t_max = max(t_max, 10.0)  # Ensure minimum range
    end
    
    # Create time grid (start slightly above 0 to avoid issues at boundary)
    t_grid = collect(range(1e-6, t_max, length=n_grid))
    
    # Compute cumulative incidence at reference level
    # This returns matrix: (n_times × n_destinations)
    ci_matrix = cumulative_incidence_at_reference(t_grid, model; statefrom=statefrom)
    
    # Total exit probability = sum across all destinations (competing risks)
    # ci_matrix is cumulative incidence for each cause, sum gives total exit CDF
    total_exit_cdf = vec(sum(ci_matrix, dims=2))
    
    # Invert CDF to get quantiles via linear interpolation
    result = Float64[]
    sorted_quantiles = sort(quantiles)
    
    for q in sorted_quantiles
        # Find interval where CDF crosses q
        idx = searchsortedfirst(total_exit_cdf, q)
        
        if idx == 1
            # Quantile is below first grid point - use first point
            push!(result, t_grid[1])
        elseif idx > length(t_grid)
            # Quantile is above maximum CDF value - cap at t_max
            # This can happen if exit probability doesn't reach q by t_max
            @warn "Exit probability at t_max=$(t_max) is $(total_exit_cdf[end]) < $q; extending grid may help"
            push!(result, t_grid[end])
        else
            # Linear interpolation between idx-1 and idx
            t_lo, t_hi = t_grid[idx-1], t_grid[idx]
            cdf_lo, cdf_hi = total_exit_cdf[idx-1], total_exit_cdf[idx]
            
            # Handle edge case of flat CDF
            if abs(cdf_hi - cdf_lo) < 1e-12
                push!(result, t_lo)
            else
                t_q = t_lo + (t_hi - t_lo) * (q - cdf_lo) / (cdf_hi - cdf_lo)
                push!(result, t_q)
            end
        end
    end
    
    # Return in original quantile order
    return result[sortperm(sortperm(quantiles))]
end

"""
    _get_exit_quantiles_by_origin(model::MultistateProcess;
                                   quantiles::Vector{Float64}=[0.25, 0.5, 0.75],
                                   t_max::Union{Nothing, Float64}=nothing,
                                   n_grid::Int=1000,
                                   verbose::Bool=false) -> Dict{Int, Vector{Float64}}

Compute exit time quantiles for all transient states at reference covariate level.

Returns a dictionary mapping origin state → exit time quantiles.
This is the main entry point for knot calibration using CDF inversion.

# Arguments
- `model::MultistateProcess`: The multistate model
- `quantiles::Vector{Float64}`: Quantile levels to compute (default: quartiles)
- `t_max`: Maximum time for CDF evaluation grid (default: 2× max observed time)
- `n_grid::Int=1000`: Grid resolution for CDF evaluation
- `verbose::Bool=false`: Print progress information

# Returns
`Dict{Int, Vector{Float64}}` mapping each transient origin state to its exit quantiles.

# Notes
- Skips absorbing states (no outgoing transitions)
- For panel data, the surrogate should be fitted before calling this function
- All competing hazards from the same origin share the same exit time distribution,
  so knots are naturally shared across them

See also: [`_compute_exit_quantiles_at_reference`](@ref), [`calibrate_splines`](@ref)
"""
function _get_exit_quantiles_by_origin(model::MultistateProcess;
                                        quantiles::Vector{Float64}=[0.25, 0.5, 0.75],
                                        t_max::Union{Nothing, Float64}=nothing,
                                        n_grid::Int=1000,
                                        verbose::Bool=false)
    result = Dict{Int, Vector{Float64}}()
    
    # Find transient states (states with outgoing transitions)
    tmat = model.tmat
    n_states = size(tmat, 1)
    
    for s in 1:n_states
        # Check if state has any outgoing transitions
        has_outgoing = any(tmat[s, :] .!= 0)
        if !has_outgoing
            continue  # Absorbing state, skip
        end
        
        verbose && @info "Computing exit quantiles from state $s..."
        
        try
            q_values = _compute_exit_quantiles_at_reference(
                model, s;
                quantiles=quantiles,
                t_max=t_max,
                n_grid=n_grid
            )
            result[s] = q_values
            verbose && @info "  State $s: quantiles at $(round.(q_values, digits=4))"
        catch e
            @warn "Failed to compute exit quantiles from state $s: $e"
            # Fall back to uniform placement
            result[s] = collect(range(0.1, t_max === nothing ? 10.0 : t_max*0.9, length=length(quantiles)))
        end
    end
    
    return result
end

"""
    _get_nknots_for_hazard(hazname::Symbol, nknots, n_obs::Int) -> Int

Determine number of interior knots for a specific hazard.
"""
function _get_nknots_for_hazard(hazname::Symbol, 
                                 nknots::Union{Int, NamedTuple, Nothing},
                                 n_obs::Int)
    if nknots === nothing
        return default_nknots(n_obs)
    elseif nknots isa Int
        return nknots
    else
        # NamedTuple - look up by hazard name
        return haskey(nknots, hazname) ? nknots[hazname] : default_nknots(n_obs)
    end
end

"""
    _get_quantiles_for_hazard(hazname::Symbol, quantiles, nknots::Int) -> Union{Vector{Float64}, Nothing}

Determine quantile levels for a specific hazard, or nothing if using automatic placement.
"""
function _get_quantiles_for_hazard(hazname::Symbol,
                                    quantiles::Union{Vector{Float64}, NamedTuple, Nothing},
                                    nknots::Int)
    if quantiles === nothing
        return nothing  # Use place_interior_knots for automatic placement
    elseif quantiles isa Vector{Float64}
        return quantiles
    else
        # NamedTuple - look up by hazard name
        return haskey(quantiles, hazname) ? quantiles[hazname] : nothing
    end
end

"""
    _rebuild_model_parameters!(model::MultistateProcess)

Rebuild model parameter structure after spline hazard modification.

When spline hazards are rebuilt with different knot counts, the number of
basis functions changes. This function rebuilds the parameter structure
to match the new hazard dimensions, using the same `rebuild_parameters` function
that `set_parameters!` uses to ensure consistency.

# Notes
- Parameters are initialized to sensible defaults:
  - Spline coefficients: 1.0 (natural scale, gives constant hazard)
  - Covariate coefficients: 0.0 (no effect)
- The resulting parameter structure includes a proper `reconstructor` field
  for AD-compatible flatten/unflatten operations
- Bounds are regenerated to match the new parameter dimensions
"""
function _rebuild_model_parameters!(model::MultistateProcess)
    # Build new parameter vectors for each hazard
    # Spline coefficients: initialize to 1.0 (constant hazard)
    # Covariate coefficients: initialize to 0.0 (no effect)
    new_param_vectors = Vector{Vector{Float64}}(undef, length(model.hazards))
    for i in eachindex(model.hazards)
        haz = model.hazards[i]
        params = zeros(Float64, haz.npar_total)
        
        # Initialize baseline spline coefficients to 1.0 for non-zero hazard
        if haz isa _SplineHazard
            params[1:haz.npar_baseline] .= 1.0
        end
        
        new_param_vectors[i] = params
    end
    
    # First, regenerate bounds to match new parameter dimensions
    # This must happen BEFORE rebuild_parameters because it validates bounds
    flat_params = reduce(vcat, new_param_vectors)
    model.bounds = _generate_package_bounds_from_components(flat_params, model.hazards, model.hazkeys)
    
    # Use rebuild_parameters to create proper parameter structure with reconstructor
    model.parameters = rebuild_parameters(new_param_vectors, model)
    
    return nothing
end

"""
    _build_unflatten_function(hazards::Vector{<:_Hazard})

Build a function that converts flat parameter vector to nested structure.
"""
function _build_unflatten_function(hazards::Vector{<:_Hazard})
    # Precompute offsets and sizes
    offsets = Int[]
    sizes = Int[]
    baselines = Int[]
    haznames = Symbol[]
    
    offset = 0
    for haz in hazards
        push!(offsets, offset)
        push!(sizes, haz.npar_total)
        push!(baselines, haz.npar_baseline)
        push!(haznames, haz.hazname)
        offset += haz.npar_total
    end
    
    function unflatten(flat::AbstractVector)
        result = Dict{Symbol, NamedTuple}()
        for i in eachindex(hazards)
            start_idx = offsets[i] + 1
            end_idx = offsets[i] + sizes[i]
            haz_params = flat[start_idx:end_idx]
            baseline = haz_params[1:baselines[i]]
            covariates = sizes[i] > baselines[i] ? haz_params[(baselines[i]+1):end] : eltype(flat)[]
            result[haznames[i]] = (baseline=baseline, covariates=covariates)
        end
        return NamedTuple(result)
    end
    
    return unflatten
end

"""
    rectify_coefs!(ests, model)

Pass model estimates through spline coefficient transformations to remove numerical zeros.

For spline hazards, the I-spline transformation can accumulate numerical errors
during optimization. This function applies a round-trip transformation
(ests → coefs → ests) with zero-clamping to clean up near-zero values.

# Note
All parameters are stored on **natural scale** (v0.3.0+). Box constraints enforce
positivity where needed (e.g., spline coefficients ≥ 0). No log/exp transforms are
applied during the round-trip; the transformation is purely for numerical cleanup
of accumulated floating-point errors.

# AD Compatibility
This function is called **post-optimization only** and is never in the AD-traced path.
The likelihood functions (`loglik_exact`, `loglik_path`, etc.) that ARE differentiated
during optimization are fully AD-compatible with both forward and reverse modes.

# Arguments
- `ests::AbstractVector`: Flat parameter vector (natural scale) to be modified in-place
- `model::MultistateProcess`: The multistate model containing hazard definitions

# See Also
- [`_spline_ests2coefs`](@ref): Transform estimates to B-spline coefficients
- [`_spline_coefs2ests`](@ref): Transform B-spline coefficients back to estimates
"""
function rectify_coefs!(ests, model)
    offset = 0
    
    for haz in model.hazards
        if haz isa RuntimeSplineHazard
            nbasis = haz.npar_baseline
            basis = _rebuild_spline_basis(haz)
            
            # Extract baseline parameters directly from flat vector
            baseline_view = @view ests[offset+1:offset+nbasis]
            
            # Round-trip: ests → coefs → ests with zero-clamping
            coefs = _spline_ests2coefs(baseline_view, basis, haz.monotone)
            rectified = _spline_coefs2ests(coefs, basis, haz.monotone; clamp_zeros=true)
            
            # Update in-place
            baseline_view .= rectified
        end
        offset += haz.npar_total
    end
    
    return nothing
end

"""
    _rebuild_spline_basis(hazard::RuntimeSplineHazard)

Rebuild the BSpline basis from a RuntimeSplineHazard for coefficient transformations.

Handles natural splines and constant extrapolation boundary conditions:
- constant: enforces D¹=0 (Neumann BC) at RIGHT boundary for smooth constant extrapolation
- natural_spline: enforces D²=0 (natural spline) at RIGHT boundary
No boundary condition at left (t=0) for full flexibility near the origin.
"""
function _rebuild_spline_basis(hazard::RuntimeSplineHazard)
    B = BSplineBasis(BSplineOrder(hazard.degree + 1), copy(hazard.knots))
    
    # Apply boundary conditions at RIGHT boundary only
    # No constraint at left (t=0) for full flexibility near the origin
    if hazard.extrapolation == "constant" && hazard.degree >= 2
        # constant: D¹=0 at right boundary for C¹ continuity
        B = RecombinedBSplineBasis(B, (), Derivative(1))  # free left, Neumann right
    elseif (hazard.degree > 1) && hazard.natural_spline
        # Natural spline: D²=0 at right boundary
        B = RecombinedBSplineBasis(B, (), Derivative(2))  # free left, natural right
    end
    return B
end
