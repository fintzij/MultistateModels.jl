# =============================================================================
# Penalty Configuration Builder
# =============================================================================
#
# This file contains functions for building PenaltyConfig from user-facing
# SplinePenalty specifications. It must be loaded AFTER hazard/spline.jl
# since it uses build_spline_hazard_info and validate_shared_knots.
#
# Also AFTER penalty_weighting.jl for compute_atrisk_counts functions.
#
# =============================================================================

"""
    _build_penalty_matrix_with_weighting(model, hazard, order, weighting) -> Matrix{Float64}

Build the penalty matrix for a spline hazard, accounting for weighting specification.

For `UniformWeighting`, uses the standard GPS penalty matrix.
For `AtRiskWeighting`, computes at-risk counts at knot midpoints and builds
a weighted penalty matrix.

# Arguments
- `model`: MultistateProcess with data
- `hazard`: RuntimeSplineHazard 
- `order::Int`: Penalty derivative order
- `weighting::PenaltyWeighting`: Weighting specification

# Returns
- Penalty matrix (K × K), possibly transformed for monotone splines

# Notes
All penalty matrices are normalized so that their maximum eigenvalue ≈ 1.
This ensures that the smoothing parameter λ has a consistent interpretation
regardless of time scale or knot spacing. Without normalization, the GPS 
penalty matrix eigenvalues scale as O(1/h²) where h is the knot spacing,
causing λ selection algorithms to produce extreme values.

For at-risk weighted penalties, the weighting is applied first, then the
matrix is normalized like the uniform case.
"""
function _build_penalty_matrix_with_weighting(model::MultistateProcess, 
                                               hazard, 
                                               order::Int, 
                                               weighting::PenaltyWeighting)
    # Rebuild basis from hazard
    basis = _rebuild_spline_basis(hazard)
    
    # Build penalty matrix based on weighting type
    S_unnormalized = if weighting isa UniformWeighting
        # Standard GPS penalty matrix
        build_penalty_matrix(basis, order)
    elseif weighting isa AtRiskWeighting
        # Compute interval-averaged at-risk counts (preferred over midpoint evaluation)
        transition = (hazard.statefrom, hazard.stateto)
        atrisk = compute_atrisk_interval_averages(model, hazard, transition)
        
        # Build weighted penalty matrix (unnormalized)
        build_weighted_penalty_matrix(basis, order, weighting, atrisk)
    else
        throw(ArgumentError("Unsupported weighting type: $(typeof(weighting))"))
    end
    
    # CRITICAL: Normalize penalty matrix so max eigenvalue ≈ 1
    # The GPS penalty matrix eigenvalues scale as O(1/h²) where h is knot spacing.
    # For data with small time scales (e.g., [0,1] vs [0,100]), this causes 
    # eigenvalues to vary by factors of 10^4 or more. Without normalization,
    # λ selection algorithms converge to extreme values (λ → 0 or λ → ∞).
    # Normalizing ensures λ=1 gives roughly equal weight to data fit vs smoothness.
    λ_max = maximum(eigvals(Symmetric(S_unnormalized)))
    
    S_bspline = if λ_max > NUMERICAL_ZERO_TOL  # Avoid division by near-zero
        S_unnormalized / λ_max
    else
        S_unnormalized  # Degenerate case: penalty is essentially zero
    end
    
    # For monotone splines, transform penalty to I-spline parameter space
    if hazard.monotone != 0
        S = transform_penalty_for_monotone(S_bspline, basis; direction=hazard.monotone)
    else
        S = S_bspline
    end
    
    return S
end

"""
    build_penalty_config(model::MultistateProcess, 
                         penalties::Union{Nothing, SplinePenalty, Vector{SplinePenalty}};
                         lambda_init::Float64=1.0) -> PenaltyConfig

Resolve user-facing `SplinePenalty` specifications into a `PenaltyConfig`.

This function processes the rule-based penalty specification and builds the
internal data structures needed for penalized likelihood evaluation.

# Arguments
- `model::MultistateProcess`: The multistate model
- `penalties`: Penalty specification(s):
  - `nothing`: No penalties (returns empty config)
  - `SplinePenalty`: Single rule applied to all matching hazards
  - `Vector{SplinePenalty}`: Multiple rules, resolved by specificity
- `lambda_init::Float64=1.0`: Initial value for smoothing parameters

# Returns
- `PenaltyConfig`: Resolved penalty configuration ready for optimization

# Rule Resolution
For a transition r → s, settings are resolved by specificity:
1. Transition rule `(r, s)` — highest priority
2. Origin rule `r`
3. Global rule `:all`
4. System defaults (order=2, total_hazard=false, share_lambda=false)

# Validation
- Validates that hazards sharing penalties have identical knots
- Throws `ArgumentError` for invalid configurations

# Example
```julia
model = multistatemodel(h12, h13; data=data)

# Simple: curvature penalty on all spline hazards
config = build_penalty_config(model, SplinePenalty())

# Complex: different settings per origin
config = build_penalty_config(model, [
    SplinePenalty(1, share_lambda=true, total_hazard=true),
    SplinePenalty(2, order=1)
])
```

See also: [`SplinePenalty`](@ref), [`PenaltyConfig`](@ref), [`compute_penalty`](@ref)
"""
function build_penalty_config(model::MultistateProcess, 
                               penalties::Union{Nothing, SplinePenalty, Vector{SplinePenalty}};
                               lambda_init::Float64=1.0,
                               include_smooth_covariates::Bool=true)
    # Compute parameter indices for each hazard (flat parameter vector)
    param_offsets = _compute_hazard_param_offsets(model)
    
    # Determine covariate lambda sharing mode from penalties
    share_covariate_lambda = false
    if !isnothing(penalties)
        penalty_rules = penalties isa SplinePenalty ? [penalties] : penalties
        # Use the most specific (last) non-false setting found
        for rule in penalty_rules
            if rule.share_covariate_lambda != false
                share_covariate_lambda = rule.share_covariate_lambda
            end
        end
    end
    
    # Check for smooth covariate terms in any hazard
    smooth_covariate_terms, shared_smooth_groups, n_smooth_lambda = if include_smooth_covariates
        _build_smooth_covariate_penalty_terms(model, param_offsets, lambda_init, share_covariate_lambda)
    else
        (SmoothCovariatePenaltyTerm[], Vector{Int}[], 0)
    end
    
    # Handle nothing case - still include smooth covariate terms if present
    if isnothing(penalties)
        if n_smooth_lambda > 0
            return PenaltyConfig(PenaltyTerm[], TotalHazardPenaltyTerm[], smooth_covariate_terms,
                                  Dict{Int,Vector{Int}}(), shared_smooth_groups, n_smooth_lambda)
        else
            return PenaltyConfig()
        end
    end
    
    # Normalize to vector
    penalty_rules = penalties isa SplinePenalty ? [penalties] : penalties
    
    # Find all spline hazards
    spline_hazard_indices = findall(h -> h isa _SplineHazard, model.hazards)
    
    # If no spline hazards, return config with only smooth covariate terms
    if isempty(spline_hazard_indices)
        if n_smooth_lambda > 0
            return PenaltyConfig(PenaltyTerm[], TotalHazardPenaltyTerm[], smooth_covariate_terms,
                                  Dict{Int,Vector{Int}}(), shared_smooth_groups, n_smooth_lambda)
        else
            return PenaltyConfig()
        end
    end
    
    # Resolve settings for each spline hazard
    resolved_settings = Dict{Int, NamedTuple}()  # hazard_index => (order, share_lambda, total_hazard, weighting, specificity)
    
    for haz_idx in spline_hazard_indices
        hazard = model.hazards[haz_idx]
        origin = hazard.statefrom
        dest = hazard.stateto
        
        # Default settings
        order = 2
        share_lambda = false
        total_hazard = false
        weighting = UniformWeighting()  # Default: uniform weighting
        
        # Apply rules in order of decreasing specificity
        for rule in penalty_rules
            matches = _rule_matches(rule.selector, origin, dest)
            if matches
                # More specific rules override
                specificity = _rule_specificity(rule.selector)
                current_spec = get(resolved_settings, haz_idx, nothing)
                if isnothing(current_spec) || specificity > current_spec.specificity
                    order = rule.order
                    share_lambda = rule.share_lambda
                    total_hazard = rule.total_hazard
                    weighting = rule.weighting
                    resolved_settings[haz_idx] = (
                        order=order, 
                        share_lambda=share_lambda, 
                        total_hazard=total_hazard,
                        weighting=weighting,
                        specificity=specificity
                    )
                end
            end
        end
        
        # Ensure we have settings even if no rule matched
        if !haskey(resolved_settings, haz_idx)
            resolved_settings[haz_idx] = (order=2, share_lambda=false, total_hazard=false, 
                                          weighting=UniformWeighting(), specificity=0)
        end
    end
    
    # Group hazards by origin for shared lambda handling
    hazards_by_origin = Dict{Int, Vector{Int}}()
    for haz_idx in spline_hazard_indices
        origin = model.hazards[haz_idx].statefrom
        if !haskey(hazards_by_origin, origin)
            hazards_by_origin[origin] = Int[]
        end
        push!(hazards_by_origin[origin], haz_idx)
    end
    
    # Validate shared knots where needed
    for (origin, haz_indices) in hazards_by_origin
        if length(haz_indices) > 1
            # Check if any hazard from this origin uses shared_lambda or total_hazard
            needs_shared_knots = any(haz_indices) do idx
                settings = resolved_settings[idx]
                settings.share_lambda || settings.total_hazard
            end
            
            if needs_shared_knots
                validate_shared_knots(model, origin)
            end
        end
    end
    
    # Build penalty terms
    terms = PenaltyTerm[]
    total_hazard_terms = TotalHazardPenaltyTerm[]
    shared_lambda_groups = Dict{Int, Vector{Int}}()
    lambda_idx = 0
    
    # Track which origins need total hazard penalty
    total_hazard_origins = Set{Int}()
    for haz_idx in spline_hazard_indices
        settings = resolved_settings[haz_idx]
        if settings.total_hazard
            push!(total_hazard_origins, model.hazards[haz_idx].statefrom)
        end
    end
    
    # Process hazards grouped by origin for shared lambda
    for origin in sort(collect(keys(hazards_by_origin)))
        haz_indices = hazards_by_origin[origin]
        
        # Check if this origin has shared lambda
        any_shared = any(idx -> resolved_settings[idx].share_lambda, haz_indices)
        
        if any_shared && length(haz_indices) > 1
            # All hazards from this origin share one lambda
            lambda_idx += 1
            group_term_indices = Int[]
            
            for haz_idx in haz_indices
                hazard = model.hazards[haz_idx]
                settings = resolved_settings[haz_idx]
                
                # Build penalty matrix with weighting
                S = _build_penalty_matrix_with_weighting(model, hazard, settings.order, settings.weighting)
                
                # Parameter indices for this hazard's baseline coefficients
                offset_start, offset_end = param_offsets[haz_idx]
                n_baseline = hazard.npar_baseline
                idx_range = offset_start:(offset_start + n_baseline - 1)
                
                push!(terms, PenaltyTerm(
                    idx_range, S, lambda_init, settings.order, [hazard.hazname]
                ))
                push!(group_term_indices, length(terms))
            end
            
            shared_lambda_groups[origin] = group_term_indices
        else
            # Independent lambda for each hazard
            for haz_idx in haz_indices
                lambda_idx += 1
                hazard = model.hazards[haz_idx]
                settings = resolved_settings[haz_idx]
                
                # Build penalty matrix with weighting
                S = _build_penalty_matrix_with_weighting(model, hazard, settings.order, settings.weighting)
                
                # Parameter indices
                offset_start, offset_end = param_offsets[haz_idx]
                n_baseline = hazard.npar_baseline
                idx_range = offset_start:(offset_start + n_baseline - 1)
                
                push!(terms, PenaltyTerm(
                    idx_range, S, lambda_init, settings.order, [hazard.hazname]
                ))
            end
        end
    end
    
    # Build total hazard penalty terms
    for origin in sort(collect(total_hazard_origins))
        haz_indices = hazards_by_origin[origin]
        length(haz_indices) >= 2 || continue  # Need competing risks
        
        # Get reference hazard for penalty matrix
        ref_haz = model.hazards[haz_indices[1]]
        ref_settings = resolved_settings[haz_indices[1]]
        ref_info = build_spline_hazard_info(ref_haz; penalty_order=ref_settings.order)
        
        # Collect index ranges for all competing hazards
        competing_idx_ranges = UnitRange{Int}[]
        for haz_idx in haz_indices
            hazard = model.hazards[haz_idx]
            offset_start, _ = param_offsets[haz_idx]
            n_baseline = hazard.npar_baseline
            push!(competing_idx_ranges, offset_start:(offset_start + n_baseline - 1))
        end
        
        lambda_idx += 1
        push!(total_hazard_terms, TotalHazardPenaltyTerm(
            origin, competing_idx_ranges, ref_info.S, lambda_init, ref_settings.order
        ))
    end
    
    # Add smooth covariate penalty terms (already computed at start of function)
    lambda_idx += n_smooth_lambda
    
    return PenaltyConfig(terms, total_hazard_terms, smooth_covariate_terms, shared_lambda_groups, shared_smooth_groups, lambda_idx)
end

"""
    _build_smooth_covariate_penalty_terms(model, param_offsets, lambda_init, share_covariate_lambda)
        -> (Vector{SmoothCovariatePenaltyTerm}, Vector{Vector{Int}}, Int)

Discover and build penalty terms for smooth covariate effects (s(x)) in all hazards.

Returns a tuple of:
- `smooth_terms`: Vector of SmoothCovariatePenaltyTerm
- `shared_smooth_groups`: Groups of term indices sharing λ (empty if no sharing)
- `n_lambda`: Number of smoothing parameters for smooth terms

# Sharing Modes
- `false`: Each smooth term gets its own λ
- `:hazard`: All smooth terms within each hazard share one λ
- `:global`: All smooth terms in the model share one λ
"""
function _build_smooth_covariate_penalty_terms(model::MultistateProcess,
                                                param_offsets::Vector{Tuple{Int,Int}},
                                                lambda_init::Float64,
                                                share_covariate_lambda::Union{Bool, Symbol})
    smooth_terms = SmoothCovariatePenaltyTerm[]
    shared_smooth_groups = Vector{Vector{Int}}()
    
    # Collect all smooth terms
    for (haz_idx, hazard) in enumerate(model.hazards)
        # Check if this hazard has smooth_info
        !hasproperty(hazard, :smooth_info) && continue
        isempty(hazard.smooth_info) && continue
        
        offset_start, _ = param_offsets[haz_idx]
        
        for info in hazard.smooth_info
            # Adjust parameter indices from hazard-local to global flat vector
            global_indices = [idx + offset_start - 1 for idx in info.par_indices]
            
            push!(smooth_terms, SmoothCovariatePenaltyTerm(
                global_indices,
                info.S,
                lambda_init,
                2,  # Default penalty order for smooth covariates
                info.label,
                hazard.hazname
            ))
        end
    end
    
    # No smooth terms found
    if isempty(smooth_terms)
        return (smooth_terms, shared_smooth_groups, 0)
    end
    
    # Determine number of lambdas and groupings based on sharing mode
    if share_covariate_lambda == false
        # Each term gets its own lambda
        n_lambda = length(smooth_terms)
        # No groups - each term independent
    elseif share_covariate_lambda == :global
        # All terms share one lambda
        n_lambda = 1
        push!(shared_smooth_groups, collect(1:length(smooth_terms)))
    elseif share_covariate_lambda == :hazard
        # Group by hazard name
        hazard_to_indices = Dict{Symbol, Vector{Int}}()
        for (i, term) in enumerate(smooth_terms)
            if !haskey(hazard_to_indices, term.hazard_name)
                hazard_to_indices[term.hazard_name] = Int[]
            end
            push!(hazard_to_indices[term.hazard_name], i)
        end
        
        n_lambda = 0
        for hazname in sort(collect(keys(hazard_to_indices)))
            indices = hazard_to_indices[hazname]
            if length(indices) > 1
                push!(shared_smooth_groups, indices)
            end
            n_lambda += 1  # One lambda per hazard with smooth terms
        end
    else
        throw(ArgumentError("Unexpected share_covariate_lambda value: $share_covariate_lambda. Expected :none, :per_covariate, or :per_hazard"))
    end
    
    return (smooth_terms, shared_smooth_groups, n_lambda)
end

"""
    _rule_matches(selector, origin::Int, dest::Int) -> Bool

Check if a penalty rule selector matches a transition.
"""
function _rule_matches(selector, origin::Int, dest::Int)
    if selector isa Symbol && selector == :all
        return true
    elseif selector isa Int
        return selector == origin
    elseif selector isa Tuple{Int,Int}
        return selector == (origin, dest)
    else
        return false
    end
end

"""
    _rule_specificity(selector) -> Int

Return specificity level for rule precedence (higher = more specific).
"""
function _rule_specificity(selector)
    if selector isa Tuple{Int,Int}
        return 3  # Transition-specific
    elseif selector isa Int
        return 2  # Origin-specific
    else  # :all
        return 1  # Global
    end
end

"""
    _compute_hazard_param_offsets(model::MultistateProcess) -> Vector{Tuple{Int,Int}}

Compute parameter index ranges for each hazard in the flat parameter vector.

Returns a vector where entry i is (start_index, end_index) for hazard i.
"""
function _compute_hazard_param_offsets(model::MultistateProcess)
    n_hazards = length(model.hazards)
    offsets = Vector{Tuple{Int,Int}}(undef, n_hazards)
    
    current_offset = 1
    for i in 1:n_hazards
        n_params = model.hazards[i].npar_total
        offsets[i] = (current_offset, current_offset + n_params - 1)
        current_offset += n_params
    end
    
    return offsets
end

# =============================================================================
# MCEM Penalty Update (Phase 4)
# =============================================================================

"""
    update_penalty_weights_mcem(penalty_config::PenaltyConfig,
                                 model::MultistateProcess,
                                 samplepaths::Vector{Vector{SamplePath}},
                                 weights::Vector{Vector{Float64}},
                                 penalty_specs::Union{SplinePenalty, Vector{SplinePenalty}}) -> PenaltyConfig

Create updated penalty configuration for adaptive weighting using MCEM sampled paths.

This function is called during MCEM iterations to update penalty matrices based
on the current importance-weighted sample paths. It only updates terms that use
`AtRiskWeighting`; terms with `UniformWeighting` are unchanged.

Since `PenaltyTerm` and `PenaltyConfig` are immutable, this returns a new config
with updated penalty matrices rather than mutating in place.

# Arguments
- `penalty_config::PenaltyConfig`: Current penalty configuration
- `model::MultistateProcess`: Model with hazard definitions
- `samplepaths`: samplepaths[i][j] = j-th sampled path for subject i
- `weights`: weights[i][j] = normalized importance weight for path i,j
- `penalty_specs`: Original SplinePenalty specifications (to get weighting info)

# Returns
- New `PenaltyConfig` with updated penalty matrices for `AtRiskWeighting` terms
- If no updates needed, returns the original `penalty_config`

# Notes
- Only updates terms where the corresponding hazard has `AtRiskWeighting`
- Returns a new PenaltyConfig (immutable structs prevent in-place modification)
- Thread-safe: each term is processed independently

# Example
```julia
# In MCEM loop, after E-step:
if has_adaptive_weighting(penalty_specs)
    penalty_config = update_penalty_weights_mcem(penalty_config, model, samplepaths, 
                                                  ImportanceWeights, penalty_specs)
end
```

See also: [`compute_atrisk_counts_mcem`](@ref), [`build_weighted_penalty_matrix`](@ref)
"""
function update_penalty_weights_mcem(penalty_config::PenaltyConfig,
                                      model::MultistateProcess,
                                      samplepaths::Vector{Vector{SamplePath}},
                                      weights::Vector{Vector{Float64}},
                                      penalty_specs::Union{SplinePenalty, Vector{SplinePenalty}})
    # Normalize to vector
    specs = penalty_specs isa SplinePenalty ? [penalty_specs] : penalty_specs
    
    # Build map from hazard name to weighting specification
    # We need to match penalty terms back to their original weighting specs
    hazard_weightings = Dict{Symbol, Tuple{PenaltyWeighting, Int}}()
    for spec in specs
        for (haz_idx, hazard) in enumerate(model.hazards)
            if hazard isa _SplineHazard
                origin = hazard.statefrom
                dest = hazard.stateto
                if _rule_matches(spec.selector, origin, dest)
                    # Store weighting and order for this hazard
                    hazard_weightings[hazard.hazname] = (spec.weighting, spec.order)
                end
            end
        end
    end
    
    # Build new terms vector, replacing S matrices where needed
    new_terms = Vector{PenaltyTerm}(undef, length(penalty_config.terms))
    any_updated = false
    
    for (term_idx, term) in enumerate(penalty_config.terms)
        # Check if this term needs updating
        needs_update = false
        weighting_to_use = nothing
        order_to_use = nothing
        hazard_to_use = nothing
        
        for hazname in term.hazard_names
            weighting_info = get(hazard_weightings, hazname, nothing)
            if !isnothing(weighting_info)
                weighting, order = weighting_info
                if weighting isa AtRiskWeighting
                    # Find the hazard
                    haz_idx = findfirst(h -> h isa _SplineHazard && h.hazname == hazname, model.hazards)
                    if !isnothing(haz_idx)
                        needs_update = true
                        weighting_to_use = weighting
                        order_to_use = order
                        hazard_to_use = model.hazards[haz_idx]
                        break
                    end
                end
            end
        end
        
        if needs_update && !isnothing(hazard_to_use)
            # Rebuild basis
            basis = _rebuild_spline_basis(hazard_to_use)
            
            # Compute interval-averaged path-weighted at-risk counts
            transition = (hazard_to_use.statefrom, hazard_to_use.stateto)
            atrisk = compute_atrisk_interval_averages_mcem(
                samplepaths, weights, hazard_to_use, transition
            )
            
            # Build new weighted penalty matrix
            S_bspline = build_weighted_penalty_matrix(basis, order_to_use, weighting_to_use, atrisk)
            
            # Transform for monotone splines if needed
            S_new = if hazard_to_use.monotone != 0
                transform_penalty_for_monotone(S_bspline, basis; direction=hazard_to_use.monotone)
            else
                S_bspline
            end
            
            # Create new PenaltyTerm with updated S
            new_terms[term_idx] = PenaltyTerm(
                term.hazard_indices,
                S_new,
                term.lambda,
                term.order,
                term.hazard_names
            )
            any_updated = true
        else
            # Keep original term
            new_terms[term_idx] = term
        end
    end
    
    # If nothing was updated, return original config
    if !any_updated
        return penalty_config
    end
    
    # Return new PenaltyConfig with updated terms
    return PenaltyConfig(
        new_terms,
        penalty_config.total_hazard_terms,
        penalty_config.smooth_covariate_terms,
        penalty_config.shared_lambda_groups,
        penalty_config.shared_smooth_groups,
        penalty_config.n_lambda
    )
end

"""
    has_adaptive_weighting(penalty_specs::Union{Nothing, SplinePenalty, Vector{SplinePenalty}}) -> Bool

Check if any penalty specification uses adaptive (non-uniform) weighting.

Returns true if at least one SplinePenalty specifies `AtRiskWeighting`.
This is used to determine whether `update_penalty_weights_mcem` needs to be
called during MCEM iterations.
"""
function has_adaptive_weighting(penalty_specs::Union{Nothing, SplinePenalty, Vector{SplinePenalty}})
    isnothing(penalty_specs) && return false
    
    specs = penalty_specs isa SplinePenalty ? [penalty_specs] : penalty_specs
    
    for spec in specs
        if spec.weighting isa AtRiskWeighting
            return true
        end
    end
    
    return false
end
