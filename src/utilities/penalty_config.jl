# =============================================================================
# Penalty Configuration Builder
# =============================================================================
#
# This file contains functions for building PenaltyConfig from user-facing
# SplinePenalty specifications. It must be loaded AFTER hazard/spline.jl
# since it uses build_spline_hazard_info and validate_shared_knots.
#
# =============================================================================

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
    resolved_settings = Dict{Int, NamedTuple}()  # hazard_index => (order, share_lambda, total_hazard)
    
    for haz_idx in spline_hazard_indices
        hazard = model.hazards[haz_idx]
        origin = hazard.statefrom
        dest = hazard.stateto
        
        # Default settings
        order = 2
        share_lambda = false
        total_hazard = false
        
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
                    resolved_settings[haz_idx] = (
                        order=order, 
                        share_lambda=share_lambda, 
                        total_hazard=total_hazard,
                        specificity=specificity
                    )
                end
            end
        end
        
        # Ensure we have settings even if no rule matched
        if !haskey(resolved_settings, haz_idx)
            resolved_settings[haz_idx] = (order=2, share_lambda=false, total_hazard=false, specificity=0)
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
                
                # Build SplineHazardInfo if not cached
                info = build_spline_hazard_info(hazard; penalty_order=settings.order)
                
                # Parameter indices for this hazard's baseline coefficients
                offset_start, offset_end = param_offsets[haz_idx]
                n_baseline = hazard.npar_baseline
                idx_range = offset_start:(offset_start + n_baseline - 1)
                
                push!(terms, PenaltyTerm(
                    idx_range, info.S, lambda_init, settings.order, [hazard.hazname]
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
                
                # Build SplineHazardInfo
                info = build_spline_hazard_info(hazard; penalty_order=settings.order)
                
                # Parameter indices
                offset_start, offset_end = param_offsets[haz_idx]
                n_baseline = hazard.npar_baseline
                idx_range = offset_start:(offset_start + n_baseline - 1)
                
                push!(terms, PenaltyTerm(
                    idx_range, info.S, lambda_init, settings.order, [hazard.hazname]
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
        error("Unexpected share_covariate_lambda value: $share_covariate_lambda")
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
