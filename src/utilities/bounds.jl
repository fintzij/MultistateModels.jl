# =============================================================================
# Parameter Bounds for Box-Constrained Optimization
# =============================================================================
#
# This module provides automatic generation of parameter bounds for box-constrained
# optimization.
#
# Parameters are stored on NATURAL scale. Box constraints handle non-negativity.
#
# Design: Package bounds (hard) cannot be relaxed by user bounds (soft).
# Combination rule: final_lb = max(pkg_lb, user_lb), final_ub = min(pkg_ub, user_ub)
#
# =============================================================================

using LinearAlgebra

# Default lower bound for non-negative parameters (rates, shapes, spline coefs)
const NONNEG_LB = 0.0

"""
    generate_parameter_bounds(model::MultistateProcess; user_bounds=nothing) -> (lb, ub)

Generate combined lower and upper bounds for box-constrained optimization.

# Arguments
- `model::MultistateProcess`: Model with hazard definitions
- `user_bounds`: Optional Dict{Symbol, NamedTuple} mapping parameter names to bounds.
  Each entry should be `:param_name => (lb=..., ub=...)` with lb and/or ub specified.
  Bounds are on the **natural scale** (what you'd interpret as "the rate is at least 0.01").
  Example: `Dict(:h12_rate => (lb=0.01, ub=100.0), :h12_x => (ub=5.0,))`

# Returns
- `(lb::Vector{Float64}, ub::Vector{Float64})`: Combined bounds vectors on natural scale

# Bounds by Family
- Exponential: rate ≥ 0  
- Weibull: shape ≥ 0, scale ≥ 0  
- Gompertz: shape ∈ ℝ (can be negative), rate ≥ 0
- Spline: all coefficients ≥ 0 (non-negativity for hazard)
- Phase-type: all rates ≥ 0
- Covariate coefficients: unconstrained (lb = -Inf, ub = Inf)

# Examples
```julia
# Default package bounds (most common)
lb, ub = generate_parameter_bounds(model)

# User tightens rate bounds
lb, ub = generate_parameter_bounds(model; 
    user_bounds = Dict(:h12_rate => (lb=0.01, ub=100.0)))

# User specifies only lower bound
lb, ub = generate_parameter_bounds(model;
    user_bounds = Dict(:h12_rate => (lb=0.01,)))
```

See also: [`_generate_package_bounds`](@ref)
"""
function generate_parameter_bounds(model::MultistateProcess; user_bounds=nothing)
    # 1. Use model's pre-computed bounds (v0.3.0+) or generate them
    if hasproperty(model, :bounds) && !isnothing(model.bounds)
        pkg_lb_nat, pkg_ub_nat = model.bounds.lb, model.bounds.ub
    else
        # Fallback for models created before bounds were stored
        pkg_lb_nat, pkg_ub_nat = _generate_package_bounds(model)
    end
    
    # 2. If no user bounds, return package bounds directly
    if isnothing(user_bounds)
        return pkg_lb_nat, pkg_ub_nat
    end
    
    # 3. Convert user bounds Dict to vectors (natural scale)
    user_lb_vec, user_ub_vec = _resolve_user_bounds(user_bounds, model)
    
    # 4. Combine via intersection on natural scale (most restrictive)
    final_lb_nat = max.(pkg_lb_nat, user_lb_vec)
    final_ub_nat = min.(pkg_ub_nat, user_ub_vec)
    
    # 5. Validate combined bounds (on natural scale)
    conflicts = findall(final_lb_nat .> final_ub_nat)
    if !isempty(conflicts)
        parnames = _get_flat_parnames(model)
        conflict_names = parnames[conflicts]
        throw(ArgumentError(
            "User bounds conflict with package constraints for parameters: $(conflict_names). " *
            "Package requires lb=$(pkg_lb_nat[conflicts]), ub=$(pkg_ub_nat[conflicts]). " *
            "User specified lb=$(user_lb_vec[conflicts]), ub=$(user_ub_vec[conflicts])."
        ))
    end
    
    return final_lb_nat, final_ub_nat
end

"""
    _generate_package_bounds(model::MultistateProcess) -> (lb, ub)

Generate package-level (hard) bounds based on hazard family.
These are the minimum constraints required for model validity.

# Bounds by Family
- `:exp` (Exponential): rate ≥ 0
- `:wei` (Weibull): shape ≥ 0, scale ≥ 0
- `:gom` (Gompertz): shape ∈ ℝ, rate ≥ 0
- `:sp` (Spline): all coefficients ≥ 0 (non-negativity of hazard)
- `:pt` (Phase-type): all rates ≥ 0
- Covariate coefficients: always unconstrained (lb = -Inf, ub = Inf)

# Note
For phase-type models with shared hazards, this iterates over unique parameter sets
(via hazkeys) rather than individual hazards to avoid indexing beyond the parameter vector.

# Returns
- `(lb::Vector{Float64}, ub::Vector{Float64})`: Package bounds
"""
function _generate_package_bounds(model::MultistateProcess)
    n_params = length(model.parameters.flat)
    lb = fill(-Inf, n_params)
    ub = fill(Inf, n_params)
    
    # Build reverse mapping: parameter index -> hazard index (for phase-type shared hazards)
    params_idx_to_hazard_idx = Dict{Int, Int}()
    for (haz_idx, hazard) in enumerate(model.hazards)
        params_idx = model.hazkeys[hazard.hazname]
        if !haskey(params_idx_to_hazard_idx, params_idx)
            params_idx_to_hazard_idx[params_idx] = haz_idx
        end
    end
    
    # Iterate over unique parameter sets (sorted by parameter index)
    param_offset = 0
    for (hazname, params_idx) in sort(collect(model.hazkeys), by = x -> x[2])
        haz_idx = params_idx_to_hazard_idx[params_idx]
        hazard = model.hazards[haz_idx]
        family = hazard.family
        n_baseline = hazard.npar_baseline
        n_total = hazard.npar_total
        
        # Set bounds for baseline parameters based on family (on natural scale)
        baseline_lb = _get_baseline_lb(family, n_baseline)
        lb[param_offset+1:param_offset+n_baseline] .= baseline_lb
        
        # Covariate coefficients are unconstrained (already initialized to -Inf/Inf)
        # No action needed
        
        param_offset += n_total
    end
    
    return lb, ub
end

"""
    _get_baseline_lb(family::Symbol, n_baseline::Int) -> Vector{Float64}

Get lower bounds for baseline parameters by hazard family.

# Arguments
- `family`: Hazard family symbol (`:exp`, `:wei`, `:gom`, `:sp`, `:pt`)
- `n_baseline`: Number of baseline parameters

# Returns
Vector of lower bounds for baseline parameters
"""
function _get_baseline_lb(family::Symbol, n_baseline::Int)
    if family == :exp
        # Exponential: rate ≥ 0
        return fill(NONNEG_LB, n_baseline)
    elseif family == :wei
        # Weibull: shape ≥ 0, scale ≥ 0
        return fill(NONNEG_LB, n_baseline)
    elseif family == :gom
        # Gompertz: shape ∈ ℝ (can be negative for decreasing hazard), rate ≥ 0
        # Parameter order: [shape, rate]
        if n_baseline == 2
            return [-Inf, NONNEG_LB]
        else
            # Fallback for unexpected n_baseline
            @warn "Gompertz hazard with n_baseline=$n_baseline (expected 2)"
            return fill(NONNEG_LB, n_baseline)
        end
    elseif family == :sp
        # Spline: coefficients are on hazard scale and must be non-negative
        return fill(NONNEG_LB, n_baseline)
    elseif family == :pt
        # Phase-type: all rates ≥ 0 (λ progression rates, μ exit rates)
        return fill(NONNEG_LB, n_baseline)
    else
        @warn "Unknown hazard family :$family, using unconstrained bounds"
        return fill(-Inf, n_baseline)
    end
end

# REMOVED: _transform_bounds_to_estimation
# Parameters are now always on natural scale with box constraints.
# The old estimation scale approach (log transforms) is no longer used.

# REMOVED: _get_positive_mask  
# No longer needed since we don't distinguish positive-constrained from unconstrained
# for transformation purposes. All constraints are handled via box bounds.

"""
    _resolve_user_bounds(user_bounds, model) -> (lb_vec, ub_vec)

Convert user bounds Dict to flat vectors.

# Arguments
- `user_bounds`: Dict{Symbol, NamedTuple} mapping parameter names to bounds,
  or `nothing` for no user bounds. Each NamedTuple should have optional :lb and :ub fields.
- `model`: MultistateProcess model

# Returns
Tuple of (lb_vec, ub_vec) vectors with default -Inf/Inf for unspecified bounds
"""
function _resolve_user_bounds(user_bounds, model)
    n_params = length(model.parameters.flat)
    lb_vec = fill(-Inf, n_params)  # Default: no lower constraint from user
    ub_vec = fill(Inf, n_params)   # Default: no upper constraint from user
    
    if isnothing(user_bounds)
        return lb_vec, ub_vec
    end
    
    if !(user_bounds isa AbstractDict)
        throw(ArgumentError(
            "user_bounds must be a Dict mapping parameter names to (lb=..., ub=...) NamedTuples. " *
            "Got $(typeof(user_bounds))"
        ))
    end
    
    # Build parameter name -> index mapping
    parnames = _get_flat_parnames(model)
    name_to_idx = Dict(name => i for (i, name) in enumerate(parnames))
    
    for (name, bounds) in user_bounds
        name_sym = name isa Symbol ? name : Symbol(name)
        if !haskey(name_to_idx, name_sym)
            throw(ArgumentError(
                "Unknown parameter name :$name_sym in user_bounds. " *
                "Available parameters: $(parnames)"
            ))
        end
        idx = name_to_idx[name_sym]
        
        # Extract lb and ub from NamedTuple/Dict if present
        if _has_field(bounds, :lb)
            lb_vec[idx] = Float64(_get_field(bounds, :lb))
        end
        if _has_field(bounds, :ub)
            ub_vec[idx] = Float64(_get_field(bounds, :ub))
        end
    end
    
    return lb_vec, ub_vec
end

# Helper functions for accessing fields from NamedTuple or Dict
_has_field(nt::NamedTuple, field::Symbol) = haskey(nt, field)
_has_field(d::AbstractDict, field::Symbol) = haskey(d, field)
_get_field(nt::NamedTuple, field::Symbol) = nt[field]
_get_field(d::AbstractDict, field::Symbol) = d[field]

"""
    _get_flat_parnames(model::MultistateProcess) -> Vector{Symbol}

Get parameter names in flat vector order.

# Returns
Vector of parameter name symbols in the same order as `model.parameters.flat`
"""
function _get_flat_parnames(model::MultistateProcess)
    return reduce(vcat, [h.parnames for h in model.hazards])
end

"""
    validate_initial_values(init::AbstractVector, lb::AbstractVector, ub::AbstractVector;
                            parnames=nothing) -> Nothing

Check that initial values satisfy bounds, throwing informative error if not.

Both `init` and bounds should be on the SAME scale (typically estimation scale).

# Arguments
- `init`: Initial parameter values
- `lb`: Lower bounds
- `ub`: Upper bounds
- `parnames`: Optional parameter names for error messages

# Throws
- `ArgumentError`: If any initial value violates bounds
"""
function validate_initial_values(init::AbstractVector, lb::AbstractVector, ub::AbstractVector;
                                  parnames=nothing)
    violations_lb = findall(init .< lb)
    violations_ub = findall(init .> ub)
    
    if !isempty(violations_lb) || !isempty(violations_ub)
        msg = "Initial values violate bounds:\n"
        
        if !isempty(violations_lb)
            for i in violations_lb
                name = isnothing(parnames) ? "param[$i]" : string(parnames[i])
                msg *= "  $name: init=$(init[i]) < lb=$(lb[i])\n"
            end
        end
        
        if !isempty(violations_ub)
            for i in violations_ub
                name = isnothing(parnames) ? "param[$i]" : string(parnames[i])
                msg *= "  $name: init=$(init[i]) > ub=$(ub[i])\n"
            end
        end
        
        throw(ArgumentError(msg))
    end
    
    return nothing
end
