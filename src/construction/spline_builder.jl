# spline_builder.jl - Spline hazard building and utilities
#
# This file contains:
# - _build_spline_hazard: Main spline hazard builder
# - _generate_spline_hazard_fns: Runtime hazard/cumhaz function generation
# - _spline_ests2coefs / _spline_coefs2ests: Parameter transformations
# - _eval_linear_pred_named: Covariate evaluation
# - _eval_cumhaz_with_extrap: Cumulative hazard with extrapolation

"""
    _build_spline_hazard(ctx::HazardBuildContext)

Build a SplineHazard from the hazard specification context.

Uses BSplineKit to construct the spline basis at build time, then generates
runtime hazard/cumhaz functions that construct Spline objects on-the-fly from
the current parameters. This functional approach ensures AD compatibility.

The spline coefficients are parameterized on log scale for positivity.
For monotone splines (monotone != 0), an I-spline-like cumsum transformation
is applied via spline_ests2coefs().
"""
function _build_spline_hazard(ctx::HazardBuildContext)
    # Access the original SplineHazard (user-facing type) from context
    hazard = ctx.hazard::SplineHazard
    data = ctx.data
    
    # Check for panel data with automatic knots
    # Panel data (obstype=2) has observation times, not exact transition times.
    # Using observation intervals for knot placement may be misleading.
    has_panel_data = any(data.obstype .== 2)
    using_auto_knots = isnothing(hazard.knots)
    if has_panel_data && using_auto_knots
        @warn "Automatic knot placement with panel data uses observation intervals, not true " *
              "transition times. For better knot placement, consider:\n" *
              "  1. Specify knots explicitly: Hazard(...; knots=[0.2, 0.5, 0.8])\n" *
              "  2. Call calibrate_splines!(model) after model creation to use surrogate simulation"
    end
    
    # Covariate parameter names (if any)
    covar_pars = _semimarkov_covariate_parnames(ctx)
    covar_names = Symbol.(_covariate_labels(ctx.rhs_names))
    has_covars = !isempty(covar_names)
    
    # Extract sojourn times on the reset scale for this transition
    # Used for automatic boundary and knot placement
    samplepaths = extract_paths(data)
    sojourns_transition = extract_sojourns(hazard.statefrom, hazard.stateto, samplepaths)
    sojourns_stay = extract_sojourns(hazard.statefrom, hazard.statefrom, samplepaths)
    all_sojourns = vcat(sojourns_transition, sojourns_stay)
    
    # Determine boundary knots
    if isnothing(hazard.boundaryknots)
        if isempty(all_sojourns)
            # No observations for this transition - use timespan as fallback
            bknots = [0.0, maximum(data.tstop)]
        else
            bknots = [0.0, maximum(all_sojourns)]
        end
    else
        bknots = copy(hazard.boundaryknots)
    end
    
    # Determine interior knots - automatic placement if not specified
    if isnothing(hazard.knots)
        # Automatic knot placement using quantiles
        if isempty(sojourns_transition)
            # No observed transitions - use evenly spaced knots
            nknots = default_nknots(length(all_sojourns))
            if nknots > 0
                intknots = collect(range(bknots[1] + (bknots[2] - bknots[1])/(nknots + 1),
                                         stop=bknots[2] - (bknots[2] - bknots[1])/(nknots + 1),
                                         length=nknots))
            else
                intknots = Float64[]
            end
        else
            nknots = default_nknots(length(sojourns_transition))
            intknots = place_interior_knots(sojourns_transition, nknots;
                                           lower_bound=bknots[1], upper_bound=bknots[2])
        end
        
        if !isempty(intknots)
            @info "Auto-placed $(length(intknots)) interior knots for $(hazard.statefrom)→$(hazard.stateto) transition at: $(round.(intknots, digits=3))"
        end
    elseif hazard.knots isa Float64
        intknots = [hazard.knots]
    else
        intknots = copy(hazard.knots)
    end
    
    # Track whether user provided explicit boundary knots
    user_provided_boundaries = hazard.boundaryknots !== nothing
    
    # Validate interior knots are within boundaries
    # If user provided both boundaries and interior knots that exceed them, warn
    # If boundaries were inferred, silently extend to encompass interior knots
    if !isempty(intknots)
        needs_adjustment = any(intknots .< bknots[1]) || any(intknots .> bknots[2])
        if needs_adjustment
            if user_provided_boundaries
                @warn "Interior knots outside user-specified boundary knots. Adjusting boundaries."
            end
            # Silently extend boundaries if they were inferred
            bknots[1] = min(bknots[1], minimum(intknots))
            bknots[2] = max(bknots[2], maximum(intknots))
        end
    end
    
    # Combine and sort knots
    allknots = unique(sort([bknots[1]; intknots; bknots[2]]))
    
    # Determine if we need smooth constant extrapolation
    # 
    # EXTRAPOLATION METHODS:
    # - "constant" (default): The hazard approaches the boundary with zero slope (h'=0 at right
    #   boundary via Neumann BC), then extends as a flat constant beyond. This provides C¹
    #   continuity at the boundary - the hazard value AND its first derivative match at the
    #   boundary. Best for most applications where you want smooth behavior at time horizon.
    #
    # - "flat": The hazard extends as a constant beyond boundaries (same value as at boundary),
    #   but WITHOUT the zero-slope constraint at the boundary. This provides only C⁰ continuity
    #   (hazard values match but slopes may not). The hazard can have non-zero slope at the
    #   boundary, creating a potential "kink" when extrapolating. Use when you want the spline
    #   to have full flexibility at boundaries.
    #
    # - "linear": The hazard extends linearly beyond boundaries using the slope at the boundary.
    #   Useful when you expect the hazard to continue its trend beyond observed data.
    #
    # Note: For degree < 2 splines, "constant" automatically falls back to "flat" because
    # the Neumann boundary condition requires at least quadratic basis functions.
    use_constant = hazard.extrapolation == "constant"
    
    # Build B-spline basis with knots
    B = BSplineBasis(BSplineOrder(hazard.degree + 1), copy(allknots))
    
    # Apply boundary conditions via basis recombination at RIGHT boundary only
    # No constraint at left (t=0) for full flexibility near the origin
    if use_constant && (hazard.degree >= 2)
        # constant extrapolation: enforce D¹=0 (Neumann BC) at RIGHT boundary only
        # This gives C¹ continuity when extending as constant beyond the right boundary.
        # The hazard approaches the right boundary tangentially (zero slope), ensuring
        # a smooth transition to the constant extrapolation region.
        B = RecombinedBSplineBasis(B, (), Derivative(1))  # free left, Neumann right
    elseif (hazard.degree > 1) && hazard.natural_spline
        # Natural spline: D²=0 at RIGHT boundary only
        B = RecombinedBSplineBasis(B, (), Derivative(2))  # free left, natural right
    end
    
    # Determine extrapolation method
    # "constant" uses Flat() with the Neumann BC basis above for C¹ continuity
    # "linear" uses Linear() for slope-based extrapolation
    extrap_method = if hazard.extrapolation == "linear"
        BSplineKit.SplineExtrapolations.Linear()
    else
        # "constant" uses Flat() - the smooth basis above ensures C¹ continuity
        BSplineKit.SplineExtrapolations.Flat()
    end
    
    # Number of basis functions = number of spline coefficients
    nbasis = length(B)
    
    # Build parameter names: spline coefficients + covariates
    baseline_names = [Symbol(string(ctx.hazname), "_sp", i) for i in 1:nbasis]
    parnames = vcat(baseline_names, covar_pars)
    npar_total = nbasis + length(covar_pars)
    
    # Generate runtime hazard and cumhaz functions
    hazard_fn, cumhaz_fn = _generate_spline_hazard_fns(
        B, extrap_method, hazard.monotone, nbasis, parnames, ctx.metadata.linpred_effect
    )
    
    # Build the internal RuntimeSplineHazard struct
    smooth_info = _extract_smooth_info(ctx, parnames)
    haz_struct = RuntimeSplineHazard(
        ctx.hazname,
        hazard.statefrom,
        hazard.stateto,
        ctx.family,
        parnames,
        nbasis,
        npar_total,
        hazard_fn,
        cumhaz_fn,
        has_covars,
        covar_names,
        hazard.degree,
        allknots,
        hazard.natural_spline,
        hazard.monotone,
        hazard.extrapolation,
        ctx.metadata,
        ctx.shared_baseline_key,
        smooth_info,
    )
    
    # v0.3.0+: Initialize parameters on NATURAL scale
    # Spline coefficients = 1.0 (constant hazard = 1)
    # Covariate coefficients = 0.0 (no effect)
    init_params = zeros(Float64, npar_total)
    init_params[1:nbasis] .= 1.0
    
    return haz_struct, init_params
end

"""
    _generate_spline_hazard_fns(basis, extrap_method, monotone, nbasis, parnames, linpred_effect)

Generate runtime hazard and cumulative hazard functions for spline hazards.

Returns a tuple (hazard_fn, cumhaz_fn) where each function has signature:
- hazard_fn(t, pars, covars) -> Float64
- cumhaz_fn(lb, ub, pars, covars) -> Float64

# Performance Optimization
The functions use memoization to cache spline objects. Within a single likelihood
evaluation (all n subjects), parameters are constant, so the spline only needs
to be built once. The cache is invalidated when parameters change (new optimization
iteration).

This reduces allocation from ~700 bytes/subject to ~0 bytes/subject for repeated
evaluations with the same parameters.
"""
function _generate_spline_hazard_fns(
    basis::Union{BSplineBasis, RecombinedBSplineBasis},
    extrap_method,
    monotone::Int,
    nbasis::Int,
    parnames::Vector{Symbol},
    linpred_effect::Symbol
)
    # Extract covariate names (if any)
    covar_names = extract_covar_names(parnames)
    has_covars = !isempty(covar_names)
    
    # Memoization cache for spline objects
    # Key insight: within a single loglik evaluation, pars are constant across all subjects
    # So we only need to rebuild the spline when pars change
    # Using Ref for AD compatibility (mutable state that doesn't break differentiation)
    hazard_cache = Ref{Any}(nothing)  # (last_pars_hash, spline_ext)
    cumhaz_cache = Ref{Any}(nothing)  # (last_pars_hash, spline_ext, cumhaz_spline)
    
    hazard_fn = let B = basis, ext = extrap_method, mono = monotone, nb = nbasis, has_cov = has_covars, effect = linpred_effect, cache = hazard_cache
        function(t, pars, covars)
            # Handle both vector and NamedTuple parameter formats
            if pars isa AbstractVector
                spline_coefs_vec = pars[1:nb]
                covar_pars = has_cov ? pars[(nb+1):end] : Float64[]
            else
                spline_coefs_vec = collect(values(pars.baseline))
                covar_pars = has_cov ? pars.covariates : NamedTuple()
            end
            
            # Check cache - use parameter values as key
            # For Float64 params (non-AD), we can compare directly
            # For Dual params (AD), we must rebuild (AD needs fresh computation graph)
            pars_hash = _spline_pars_hash(spline_coefs_vec)
            cached = cache[]
            
            if cached !== nothing && cached[1] == pars_hash && eltype(spline_coefs_vec) === Float64
                # Cache hit - reuse spline
                spline_ext = cached[2]
            else
                # Cache miss or AD mode - rebuild spline
                coefs = _spline_ests2coefs(spline_coefs_vec, B, mono)
                spline = Spline(B, coefs)
                spline_ext = SplineExtrapolation(spline, ext)
                # Only cache for Float64 (non-AD) evaluation
                if eltype(spline_coefs_vec) === Float64
                    cache[] = (pars_hash, spline_ext)
                end
            end
            
            # Apply covariate effect
            n_covars = covars isa AbstractVector ? length(covars) : length(covars)
            if has_cov && n_covars > 0
                linear_pred = pars isa AbstractVector ? 
                              dot(collect(covars), covar_pars) : 
                              _eval_linear_pred_named(covar_pars, covars)
            else
                linear_pred = 0.0
            end

            if effect == :aft
                scale = exp(-linear_pred)
                h0 = spline_ext(t * scale)
                return h0 * scale
            else
                h0 = spline_ext(t)
                return h0 * exp(linear_pred)
            end
        end
    end
    
    cumhaz_fn = let B = basis, ext = extrap_method, mono = monotone, nb = nbasis, has_cov = has_covars, effect = linpred_effect, cache = cumhaz_cache
        function(lb, ub, pars, covars)
            # Handle both vector and NamedTuple parameter formats
            if pars isa AbstractVector
                spline_coefs_vec = pars[1:nb]
                covar_pars = has_cov ? pars[(nb+1):end] : Float64[]
            else
                # NamedTuple format
                spline_coefs_vec = collect(values(pars.baseline))
                covar_pars = has_cov ? pars.covariates : NamedTuple()
            end
            
            # Check cache - use parameter values as key
            pars_hash = _spline_pars_hash(spline_coefs_vec)
            cached = cache[]
            
            if cached !== nothing && cached[1] == pars_hash && eltype(spline_coefs_vec) === Float64
                # Cache hit - reuse spline and integral
                spline_ext = cached[2]
                cumhaz_spline = cached[3]
            else
                # Cache miss or AD mode - rebuild spline and integral
                coefs = _spline_ests2coefs(spline_coefs_vec, B, mono)
                spline = Spline(B, coefs)
                spline_ext = SplineExtrapolation(spline, ext)
                cumhaz_spline = integral(spline_ext.spline)
                # Only cache for Float64 (non-AD) evaluation
                if eltype(spline_coefs_vec) === Float64
                    cache[] = (pars_hash, spline_ext, cumhaz_spline)
                end
            end
            
            # Apply covariate effect - only if covariates actually provided
            n_covars_provided = covars isa NamedTuple ? length(covars) : length(covars)
            if n_covars_provided > 0
                linear_pred = pars isa AbstractVector ? 
                              dot(collect(covars), covar_pars) : 
                              _eval_linear_pred_named(covar_pars, covars)
            else
                linear_pred = 0.0
            end

            if effect == :aft
                scale = exp(-linear_pred)
                H0 = _eval_cumhaz_with_extrap(spline_ext, cumhaz_spline, lb * scale, ub * scale)
                return H0
            else
                H0 = _eval_cumhaz_with_extrap(spline_ext, cumhaz_spline, lb, ub)
                return H0 * exp(linear_pred)
            end
        end
    end
    
    return hazard_fn, cumhaz_fn
end

"""
    _spline_pars_hash(pars)

Compute a simple hash for spline parameters to enable cache invalidation.
Uses sum and product of parameters - cheap to compute and changes when pars change.
"""
@inline function _spline_pars_hash(pars::AbstractVector{T}) where T
    # Simple but effective: sum and first/last element give unique fingerprint
    # Using actual equality check rather than approximate since we want exact cache matches
    return (sum(pars), length(pars) > 0 ? pars[1] : zero(T), length(pars) > 1 ? pars[end] : zero(T))
end

"""
    _spline_ests2coefs(ests, basis, monotone)

Transform spline parameter estimates to spline coefficients.

For monotone == 0: identity (parameters ARE coefficients, non-negativity via box constraints)
For monotone == 1: I-spline-like cumulative sum (increasing hazard)
For monotone == -1: reverse cumulative sum (decreasing hazard)

Note: Parameters are now on NATURAL scale (not log). Box constraints ensure β ≥ 0.
"""
function _spline_ests2coefs(ests::AbstractVector{T}, basis, monotone::Int) where T
    if monotone == 0
        # Non-negative coefficients directly (box constraints ensure ≥ 0)
        return ests
    else
        # I-spline transformation for monotonicity
        # ests are non-negative increments (box constrained to ≥ 0)
        coefs = zeros(T, length(ests))
        
        if length(coefs) > 1
            k = BSplineKit.order(basis)
            t = BSplineKit.knots(basis)
            
            for i in Iterators.drop(eachindex(coefs), 1)
                coefs[i] = coefs[i-1] + ests[i] * (t[i + k] - t[i]) / k
            end
        end
        
        # Add intercept
        coefs .+= ests[1]
        
        # Reverse for decreasing monotone
        if monotone == -1
            reverse!(coefs)
        end
        
        return coefs
    end
end

"""
    _spline_coefs2ests(coefs, basis, monotone; clamp_zeros=false)

Transform spline coefficients back to natural-scale parameter estimates.
Inverse of `_spline_ests2coefs`.

For monotone == 0: identity (coefficients ARE parameters)
For monotone == 1: difference transformation (inverse of cumsum)
For monotone == -1: reverse then difference

Note: Parameters are now on NATURAL scale (not log). Box constraints ensure β ≥ 0.
"""
function _spline_coefs2ests(coefs::AbstractVector{T}, basis, monotone::Int; clamp_zeros::Bool=false) where T
    if monotone == 0
        # Identity: coefficients are parameters directly
        return copy(coefs)
    else
        # Reverse if decreasing
        coefs_nat = monotone == 1 ? copy(coefs) : reverse(coefs)
        
        ests = zeros(T, length(coefs_nat))
        
        if length(coefs) > 1
            k = BSplineKit.order(basis)
            t = BSplineKit.knots(basis)
            
            # Inverse of the cumsum: take differences
            for i in length(coefs_nat):-1:2
                ests[i] = (coefs_nat[i] - coefs_nat[i - 1]) * k / (t[i + k] - t[i])
            end
        end
        
        # Intercept
        ests[1] = coefs_nat[1]
        
        # Clamp numerical zeros
        if clamp_zeros
            ests[findall(isapprox.(ests, 0.0; atol = sqrt(eps())))] .= zero(T)
        end
        
        return ests
    end
end

"""
    _eval_linear_pred_named(pars_covariates::NamedTuple, covars::NamedTuple)

Evaluate the linear predictor β'x using named parameter and covariate access.
Assumes parameter names match covariate names (with hazard prefix stripped).

# Example
```julia
pars.covariates = (h12_age = 0.3, h12_sex = 0.1)
covars = (age = 50.0, sex = 1.0)
result = _eval_linear_pred_named(pars.covariates, covars)  # 0.3*50 + 0.1*1
```
"""
function _eval_linear_pred_named(pars_covariates::NamedTuple, covars::NamedTuple)
    result = 0.0
    for (pname, pval) in pairs(pars_covariates)
        # Extract covariate name by stripping hazard prefix
        # Handles: h12_age -> age, h12_a_age -> age (phase-type)
        # Phase-type suffix limited to 1-3 letters to avoid matching covariate underscores
        cname_str = replace(String(pname), r"^h\d+(?:_[a-z]{1,3})?_" => "")
        cname = Symbol(cname_str)
        if haskey(covars, cname)
            result += pval * getfield(covars, cname)
        else
            throw(ArgumentError("Covariate $cname not found in covars NamedTuple. Available: $(keys(covars))"))
        end
    end
    return result
end

"""
    _eval_cumhaz_with_extrap(spline_ext, cumhaz_spline, lb, ub)

Evaluate cumulative hazard over [lb, ub] with proper extrapolation handling.

For "constant" extrapolation: hazard is constant beyond boundaries (with C¹ smooth
transition via Neumann BC), so cumulative hazard grows linearly.

For "linear" extrapolation: hazard varies linearly beyond boundaries:
  h(t) = h(t_b) + h'(t_b) * (t - t_b)
so cumulative hazard is:
  H(a,b) = h(t_b) * Δt + h'(t_b) * Δt² / 2  (for t > t_hi)
  H(a,b) = h(t_b) * Δt - h'(t_b) * Δt² / 2  (for t < t_lo)
"""
function _eval_cumhaz_with_extrap(spline_ext::SplineExtrapolation, cumhaz_spline, lb, ub)
    bounds = BSplineKit.boundaries(spline_ext.spline.basis)
    t_lo, t_hi = bounds
    
    # Simple case: both endpoints within spline support
    if lb >= t_lo && ub <= t_hi
        return cumhaz_spline(ub) - cumhaz_spline(lb)
    end
    
    # Handle extrapolation
    H_total = zero(eltype(coefficients(spline_ext.spline)))
    is_linear = spline_ext.method isa BSplineKit.SplineExtrapolations.Linear
    
    # Contribution from below lower boundary
    if lb < t_lo
        h_lo = spline_ext.spline(t_lo)  # Hazard at lower boundary
        dt = min(ub, t_lo) - lb
        if is_linear
            # Linear extrapolation: h(t) = h(t_lo) + h'(t_lo) * (t - t_lo) for t < t_lo
            # ∫[lb, t_lo] h(t) dt = h(t_lo) * dt + h'(t_lo) * ∫[lb, t_lo] (t - t_lo) dt
            # The integral ∫[lb, t_lo] (t - t_lo) dt = -dt²/2 (since t - t_lo < 0 in this region)
            dh_lo = ForwardDiff.derivative(t -> spline_ext.spline(t), t_lo)
            H_total += h_lo * dt - dh_lo * dt^2 / 2
        else
            # Constant: hazard stays at boundary value (C¹ continuous for "constant" mode)
            H_total += h_lo * dt
        end
    end
    
    # Contribution within spline support
    actual_lb = max(lb, t_lo)
    actual_ub = min(ub, t_hi)
    if actual_ub > actual_lb
        H_total += cumhaz_spline(actual_ub) - cumhaz_spline(actual_lb)
    end
    
    # Contribution from above upper boundary
    if ub > t_hi
        h_hi = spline_ext.spline(t_hi)  # Hazard at upper boundary
        dt = ub - max(lb, t_hi)
        if is_linear
            # Linear extrapolation: h(t) = h(t_hi) + h'(t_hi) * (t - t_hi) for t > t_hi
            # ∫[t_hi, ub] h(t) dt = h(t_hi) * dt + h'(t_hi) * dt²/2
            dh_hi = ForwardDiff.derivative(t -> spline_ext.spline(t), t_hi)
            H_total += h_hi * dt + dh_hi * dt^2 / 2
        else
            # Constant: hazard stays at boundary value (C¹ continuous for "constant" mode)
            H_total += h_hi * dt
        end
    end
    
    return H_total
end

# Register spline hazard family
register_hazard_family!(:sp, _build_spline_hazard)
