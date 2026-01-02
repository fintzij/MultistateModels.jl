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
- `basis::BSplineBasis`: BSplineKit basis object (may be recombined)
- `S::Matrix{Float64}`: Penalty matrix ∫B^(m)(t)B^(m)(t)ᵀdt
- `penalty_order::Int`: Derivative order for penalty (default 2 = curvature)

# Notes
- The penalty matrix `S` is positive semi-definite with null space consisting of
  polynomials of degree < `penalty_order`
- For shared penalty (competing risks), all hazards from the same origin share
  identical `breakpoints` and compatible `S` matrices
"""
struct SplineHazardInfo
    origin::Int
    dest::Int
    nbasis::Int
    breakpoints::Vector{Float64}
    basis::Any  # BSplineBasis or RecombinedBSplineBasis - use Any for type stability
    S::Matrix{Float64}
    penalty_order::Int
    
    function SplineHazardInfo(origin::Int, dest::Int, nbasis::Int,
                               breakpoints::Vector{Float64}, basis,
                               S::Matrix{Float64}, penalty_order::Int)
        # Validate dimensions
        nbasis > 0 || throw(ArgumentError("nbasis must be positive, got $nbasis"))
        penalty_order >= 1 || throw(ArgumentError("penalty_order must be ≥ 1, got $penalty_order"))
        size(S, 1) == nbasis && size(S, 2) == nbasis || 
            throw(ArgumentError("S must be $nbasis × $nbasis, got $(size(S))"))
        length(breakpoints) >= 2 || 
            throw(ArgumentError("breakpoints must have at least 2 elements"))
        
        # Validate S is symmetric
        norm(S - S') < 1e-10 * norm(S) || 
            throw(ArgumentError("Penalty matrix S must be symmetric"))
        
        new(origin, dest, nbasis, breakpoints, basis, S, penalty_order)
    end
end

"""
    build_penalty_matrix(basis, order::Int; knots::Vector{Float64}=Float64[])

Construct the derivative-based penalty matrix using Wood's (2016) algorithm.

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
function build_penalty_matrix(basis, order::Int; knots::Vector{Float64}=Float64[])
    order >= 1 || throw(ArgumentError("Penalty order must be ≥ 1, got $order"))
    
    # Get knot vector - handle both BSplineBasis and RecombinedBSplineBasis
    if isempty(knots)
        knots = collect(BSplineKit.knots(basis))
    end
    
    # Get number of basis functions
    K = length(basis)
    
    # Get spline order (degree + 1)
    m1 = BSplineKit.order(basis)
    
    # Derivative reduces order
    m_deriv = m1 - order
    if m_deriv < 1
        # Derivative order exceeds spline degree, penalty is identically zero
        @warn "Penalty order $order exceeds spline degree $(m1 - 1); returning zero matrix"
        return zeros(K, K)
    end
    
    # Number of Gauss-Legendre points per interval
    # For exact integration of product of two derivatives (each degree m_deriv - 1):
    # Product has degree 2*(m_deriv - 1), need (2*(m_deriv-1)+1)/2 ≈ m_deriv points
    n_gauss = max(m_deriv, 2)
    
    # Get Gauss-Legendre nodes and weights on [-1, 1]
    gl_nodes, gl_weights = _gauss_legendre(n_gauss)
    
    # Unique interior breakpoints (excluding repeated boundary knots)
    unique_knots = unique(knots)
    n_intervals = length(unique_knots) - 1
    
    # Initialize penalty matrix
    S = zeros(K, K)
    
    # Derivative operator
    deriv_op = Derivative(order)
    
    # Integrate over each knot interval
    for q in 1:n_intervals
        a, b = unique_knots[q], unique_knots[q + 1]
        h = b - a
        
        # Skip degenerate intervals
        h < 1e-14 && continue
        
        # Transform Gauss points to [a, b]
        t_points = @. a + (gl_nodes + 1) * h / 2
        w_scaled = @. gl_weights * h / 2
        
        # Evaluate basis derivatives at quadrature points
        # BSplineKit returns a matrix: rows = basis functions, cols = evaluation points
        for (i_pt, t) in enumerate(t_points)
            # Get derivative values at this point
            deriv_vals = _evaluate_basis_derivatives(basis, t, order)
            
            # Outer product contribution to penalty matrix
            w = w_scaled[i_pt]
            for i in 1:K
                for j in i:K  # Only upper triangle due to symmetry
                    S[i, j] += w * deriv_vals[i] * deriv_vals[j]
                end
            end
        end
    end
    
    # Symmetrize (copy upper to lower)
    for i in 1:K
        for j in (i+1):K
            S[j, i] = S[i, j]
        end
    end
    
    return S
end

"""
    _evaluate_basis_derivatives(basis, t::Real, order::Int) -> Vector{Float64}

Evaluate all basis function derivatives of given order at point t.

Returns a vector of length K where K is the number of basis functions.
Uses BSplineKit's `diff` function to construct the derivative spline.
"""
function _evaluate_basis_derivatives(basis, t::Real, order::Int)
    K = length(basis)
    deriv_vals = zeros(K)
    
    # For each basis function, create a spline with unit coefficient
    # and take the derivative using diff()
    for i in 1:K
        coeffs = zeros(K)
        coeffs[i] = 1.0
        spl = Spline(basis, coeffs)
        
        # Apply diff() `order` times to get the requested derivative
        dspl = spl
        for _ in 1:order
            dspl = diff(dspl)
        end
        
        deriv_vals[i] = dspl(t)
    end
    
    return deriv_vals
end

"""
    _gauss_legendre(n::Int) -> (nodes::Vector{Float64}, weights::Vector{Float64})

Compute n-point Gauss-Legendre quadrature nodes and weights on [-1, 1].

Uses the Golub-Welsch algorithm via eigenvalue decomposition of the 
Jacobi matrix.
"""
function _gauss_legendre(n::Int)
    n >= 1 || throw(ArgumentError("n must be ≥ 1"))
    
    if n == 1
        return [0.0], [2.0]
    end
    
    # Jacobi matrix for Legendre polynomials
    # β_k = k / sqrt(4k² - 1)
    beta = [k / sqrt(4.0 * k^2 - 1.0) for k in 1:(n-1)]
    
    # Tridiagonal Jacobi matrix (symmetric, so just need sub/super diagonal)
    J = SymTridiagonal(zeros(n), beta)
    
    # Eigenvalues are nodes, eigenvectors give weights
    eig = eigen(J)
    nodes = eig.values
    
    # Weights: w_k = 2 * (first component of k-th eigenvector)²
    weights = [2.0 * eig.vectors[1, k]^2 for k in 1:n]
    
    return nodes, weights
end

"""
    place_interior_knots_pooled(model::MultistateProcess, origin::Int, nknots::Int;
                                 lower_bound::Real=0.0) -> Vector{Float64}

Place interior knots at quantiles of pooled event times from all transitions 
originating from the given state.

This function is used for competing risks when hazards share a penalty structure
(via `shared_origin_tensor`). All hazards from the same origin must use the same
knot locations to enable the Kronecker product penalty formulation.

# Arguments
- `model::MultistateProcess`: The multistate model (must have data attached)
- `origin::Int`: The origin state to pool events from
- `nknots::Int`: Number of interior knots to place
- `lower_bound::Real=0.0`: Lower boundary (default 0 for sojourn times)

# Returns
- `Vector{Float64}`: Interior knot locations (excluding boundaries)

# Notes
- Pools all exact transitions (obstype 1 or 3) from the origin state
- For panel data, uses surrogate simulation to estimate transition times
- Returns evenly-spaced knots if no transitions are observed

# Example
```julia
# Model with transitions 1→2 and 1→3
model = multistatemodel(h12, h13; data=data)
knots = place_interior_knots_pooled(model, 1, 5)  # Pool 1→2 and 1→3 events
```
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
            if !(obstype == 1 || obstype == 3)
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
        if (obstype == 1 || obstype == 3) && 
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
"""
function build_spline_hazard_info(hazard::RuntimeSplineHazard; penalty_order::Int=2)
    # Rebuild basis from hazard (same logic as _rebuild_spline_basis)
    basis = _rebuild_spline_basis(hazard)
    
    # Build penalty matrix
    S = build_penalty_matrix(basis, penalty_order; knots=hazard.knots)
    
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

Uses floor(n^(1/5)) for sieve estimation. For typical survival data sizes:
- n = 100: 2-3 knots
- n = 500: 3-4 knots  
- n = 1000: 4 knots
- n = 10000: 6 knots

For small samples, returns at least 2 knots for flexibility.
survextrap uses a fixed 10 knots by default.
"""
function default_nknots(n_observations::Integer)
    n_observations <= 0 && return 0
    return max(2, floor(Int, n_observations^(1/5)))
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
                      verbose::Bool=true) -> NamedTuple

Compute recommended knot locations for spline hazards based on transition times.

For exact data (obstype 1 or 3), uses observed sojourn times directly.
For panel data (obstype 2), fits a Markov surrogate, simulates sample paths,
and uses the simulated transition times to determine knot locations.

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
- `n_paths::Int=1000`: Number of paths to sample per subject (panel data only)
- `min_ess::Int=100`: Minimum effective sample size for importance sampling (panel data only)
- `verbose::Bool=true`: Print info about knot placement

# Returns
A `NamedTuple` with one entry per spline hazard, each containing:
- `boundary_knots::Vector{Float64}`: `[0.0, max_sojourn]` for the transition
- `interior_knots::Vector{Float64}`: Recommended interior knot locations

# Errors
- `ArgumentError` if model is a `MultistateModelFitted`
- `ArgumentError` if model has no spline hazards
- `ArgumentError` if both `quantiles` and `nknots` are specified

# Notes
- For exact data: uses observed sojourn times directly (no surrogate fitting)
- For panel data: fits Markov surrogate, simulates paths, extracts transition times
- Lower boundary is always 0.0 (sojourns are non-negative)
- Upper boundary is the maximum observed/simulated sojourn time
- For ties, knots are spread evenly using [`place_interior_knots`](@ref)

# Example
```julia
h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; degree=3)
h23 = Hazard(@formula(0 ~ 1), "sp", 2, 3; degree=3)
model = multistatemodel(h12, h23; data=data)

# Auto-select nknots based on sample size
knots = calibrate_splines(model)

# Specify number of knots
knots = calibrate_splines(model; nknots=3)

# Different knots per hazard
knots = calibrate_splines(model; nknots=(h12=4, h23=2))

# Specify explicit quantiles
knots = calibrate_splines(model; quantiles=[0.2, 0.4, 0.6, 0.8])
```

See also: [`calibrate_splines!`](@ref), [`place_interior_knots`](@ref)
"""
function calibrate_splines(model::MultistateProcess;
                           quantiles::Union{Vector{Float64}, NamedTuple, Nothing}=nothing,
                           nknots::Union{Int, NamedTuple, Nothing}=nothing,
                           n_paths::Int=1000,
                           min_ess::Int=100,
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
    
    # Get sojourns by transition
    if has_exact_data
        # Use observed data directly
        verbose && @info "Using observed transition times for knot calibration"
        sojourns_by_transition = _extract_sojourns_from_data(model)
    else
        # Fit surrogate and simulate paths
        verbose && @info "Fitting Markov surrogate and simulating paths for knot calibration"
        sojourns_by_transition = _extract_sojourns_from_surrogate(model, n_paths, min_ess)
    end
    
    # Compute knots for each spline hazard
    results = Dict{Symbol, NamedTuple}()
    
    for idx in spline_indices
        haz = model.hazards[idx]
        hazname = haz.hazname
        key = (haz.statefrom, haz.stateto)
        
        # Get sojourns for this transition
        sojourns = get(sojourns_by_transition, key, Float64[])
        
        if isempty(sojourns)
            @warn "No transitions $(key[1])→$(key[2]) for hazard $hazname; using model boundaries"
            bknots = [haz.knots[1], haz.knots[end]]
            results[hazname] = (boundary_knots=bknots, interior_knots=Float64[])
            continue
        end
        
        # Determine number of knots
        nk = _get_nknots_for_hazard(hazname, nknots, length(sojourns))
        
        # Determine quantile levels or compute from nknots
        qlevels = _get_quantiles_for_hazard(hazname, quantiles, nk)
        
        # Compute boundary knots
        lb = 0.0  # Lower boundary is always 0 for sojourns
        ub = maximum(sojourns)
        
        # Compute interior knots
        if qlevels !== nothing
            # User-specified quantiles
            interior = quantile(sojourns, qlevels)
            interior = unique(interior)
        else
            # Use place_interior_knots for automatic placement with tie handling
            interior = place_interior_knots(sojourns, nk; lower_bound=lb, upper_bound=ub)
        end
        
        verbose && @info "Calibrated $(length(interior)) interior knots for $hazname at: $(round.(interior, digits=3))"
        results[hazname] = (boundary_knots=[lb, ub], interior_knots=interior)
    end
    
    return NamedTuple(results)
end

"""
    calibrate_splines!(model::MultistateProcess; quantiles=nothing, nknots=nothing, 
                       n_paths=1000, min_ess=100, verbose=true)

Compute and apply knot locations for spline hazards in-place.

Modifies the spline hazards in `model` to use the computed knot locations.
See [`calibrate_splines`](@ref) for argument details.

# Arguments
Same as [`calibrate_splines`](@ref).

# Returns
`NamedTuple` with computed knot locations (same as [`calibrate_splines`](@ref)).

# Notes
- Modifies `model.hazards` in place for all `RuntimeSplineHazard` hazards
- Also rebuilds the parameter structure to match the new number of basis functions
- After calling this, the model's spline hazards will have updated knots

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
                            verbose::Bool=true)
    
    # Get recommended knots (this also does validation)
    knots = calibrate_splines(model; quantiles=quantiles, nknots=nknots, 
                              n_paths=n_paths, min_ess=min_ess, verbose=verbose)
    
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
        use_constant = haz.extrapolation == "constant"
        if use_constant && (haz.degree >= 2)
            B = RecombinedBSplineBasis(B, Derivative(1))
        elseif (haz.degree > 1) && haz.natural_spline
            B = RecombinedBSplineBasis(B, Natural())
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
Only includes exact transitions (obstype 1 or 3) where the transition actually occurred.
"""
function _extract_sojourns_from_data(model::MultistateProcess)
    data = model.data
    result = Dict{Tuple{Int,Int}, Vector{Float64}}()
    
    for row in eachrow(data)
        # Only exact transitions
        obstype = row.obstype
        if !(obstype == 1 || obstype == 3)
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

Check if the model data contains any exact transition observations (obstype 1 or 3).
"""
function _has_exact_transitions(model::MultistateProcess)
    data = model.data
    for row in eachrow(data)
        obstype = row.obstype
        if (obstype == 1 || obstype == 3) && row.statefrom != row.stateto
            return true
        end
    end
    return false
end

"""
    _extract_sojourns_from_surrogate(model::MultistateProcess, n_paths::Int, min_ess::Int) 
        -> Dict{Tuple{Int,Int}, Vector{Float64}}

Fit a Markov surrogate to the model, simulate sample paths, and extract sojourn times.

Used for panel data where exact transition times are not observed.
"""
function _extract_sojourns_from_surrogate(model::MultistateProcess, n_paths::Int, min_ess::Int)
    # Fit and store surrogate in model (set_surrogate! fits and stores in model.markovsurrogate)
    set_surrogate!(model; verbose=false)
    
    # Draw sample paths from the model (uses stored markovsurrogate)
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
- Parameters are initialized to zero on log scale (rates = 1.0)
- The resulting parameter structure includes a proper `reconstructor` field
  for AD-compatible flatten/unflatten operations
"""
function _rebuild_model_parameters!(model::MultistateProcess)
    # Build new parameter vectors (zeros on log scale) for each hazard
    new_param_vectors = Vector{Vector{Float64}}(undef, length(model.hazards))
    for i in eachindex(model.hazards)
        haz = model.hazards[i]
        new_param_vectors[i] = zeros(Float64, haz.npar_total)
    end
    
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
"""
function rectify_coefs!(ests, model)
    nested = unflatten(model.parameters.reconstructor, ests)

    # Get hazard names in sorted order
    haznames_sorted = [hazname for (hazname, idx) in sort(collect(model.hazkeys), by = x -> x[2])]
    
    # Process each hazard
    for (i, hazname) in enumerate(haznames_sorted)
        haz = model.hazards[i]
        if isa(haz, RuntimeSplineHazard)
            # New functional design - use basis stored in the hazard functions
            # We need to rebuild the basis from the knots
            nbasis = haz.npar_baseline
            basis = _rebuild_spline_basis(haz)
            
            # Extract baseline parameters from nested structure
            hazard_params = nested[hazname]
            baseline_values = collect(values(hazard_params.baseline))
            
            # Round-trip transformation to clean up numerical zeros
            coefs = _spline_ests2coefs(baseline_values, basis, haz.monotone)
            
            # Clamp numerical zeros in coefficients
            coefs[findall(isapprox.(coefs, 0.0; atol = sqrt(eps())))] .= zero(eltype(coefs))
            
            # Transform back
            rectified_baseline = _spline_coefs2ests(coefs, basis, haz.monotone; clamp_zeros=true)
            
            # Update ests vector directly (need to find position in flat vector)
            # Compute offset for this hazard's parameters in flat vector
            offset = sum(model.hazards[j].npar_total for j in 1:(i-1); init=0)
            ests[offset+1:offset+nbasis] .= rectified_baseline
        end
    end
end

"""
    _rebuild_spline_basis(hazard::RuntimeSplineHazard)

Rebuild the BSpline basis from a RuntimeSplineHazard for coefficient transformations.

Handles natural splines and constant extrapolation boundary conditions:
- constant: enforces D¹=0 (Neumann BC) at boundaries for smooth constant extrapolation
- natural_spline: enforces D²=0 (natural spline) at boundaries
"""
function _rebuild_spline_basis(hazard::RuntimeSplineHazard)
    B = BSplineBasis(BSplineOrder(hazard.degree + 1), copy(hazard.knots))
    
    # Apply boundary conditions based on extrapolation method
    if hazard.extrapolation == "constant" && hazard.degree >= 2
        # constant: D¹=0 at both boundaries for C¹ continuity
        B = RecombinedBSplineBasis(B, Derivative(1))
    elseif (hazard.degree > 1) && hazard.natural_spline
        # Natural spline: D²=0 at boundaries
        B = RecombinedBSplineBasis(B, Natural())
    end
    return B
end

"""
    remake_splines!(hazard::RuntimeSplineHazard, parameters)

No-op for RuntimeSplineHazard since splines are constructed on-the-fly during evaluation.
The functional design means no internal state needs updating.
"""
function remake_splines!(hazard::RuntimeSplineHazard, parameters)
    return nothing
end

"""
    set_riskperiod!(hazard::RuntimeSplineHazard)

No-op for RuntimeSplineHazard since risk period is handled dynamically during 
extrapolation evaluation in the hazard/cumhaz closures.
"""
function set_riskperiod!(hazard::RuntimeSplineHazard)
    return nothing
end

"""
    _update_spline_hazards!(hazards, pars)

Update all spline hazards with new parameters. This is a no-op for the current
functional spline design (splines are constructed on-the-fly), but provides a
single call site for spline parameter updates.

# Arguments
- `hazards`: Vector of hazard objects
- `pars`: Nested parameter vectors (log-scale for splines)

This helper reduces code duplication across likelihood, sampling, and fitting code.
"""
function _update_spline_hazards!(hazards::Vector{<:_Hazard}, pars)
    for i in eachindex(hazards)
        if isa(hazards[i], _SplineHazard)
            # Note: For RuntimeSplineHazard, remake_splines! is a no-op
            remake_splines!(hazards[i], nothing)
            set_riskperiod!(hazards[i])
        end
    end
    return nothing
end
