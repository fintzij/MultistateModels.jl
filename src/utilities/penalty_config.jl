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
                               lambda_init::Float64=1.0)
    # Handle nothing case
    isnothing(penalties) && return PenaltyConfig()
    
    # Normalize to vector
    penalty_rules = penalties isa SplinePenalty ? [penalties] : penalties
    
    # Find all spline hazards
    spline_hazard_indices = findall(h -> h isa _SplineHazard, model.hazards)
    isempty(spline_hazard_indices) && return PenaltyConfig()
    
    # Compute parameter indices for each hazard (flat parameter vector)
    param_offsets = _compute_hazard_param_offsets(model)
    
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
    
    return PenaltyConfig(terms, total_hazard_terms, shared_lambda_groups, lambda_idx)
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
