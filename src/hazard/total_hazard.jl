# =============================================================================
# Total Hazard and Survival Probability Functions
# =============================================================================
#
# Functions for computing total cumulative hazard, survival probabilities,
# and next state transition probabilities.
#
# =============================================================================

"""
    survprob(lb, ub, parameters, subjdat_row, _totalhazard::_TotalHazardTransient, _hazards; give_log = true) 

Return the survival probability over the interval [lb, ub].
"""
function survprob(lb, ub, parameters, subjdat_row, _totalhazard::_TotalHazardTransient, _hazards;
                   give_log = true,
                   apply_transform::Bool = false,
                   cache_context::Union{Nothing,TimeTransformContext}=nothing) 

    # log total cumulative hazard
    log_survprob = -total_cumulhaz(
        lb,
        ub,
        parameters,
        subjdat_row,
        _totalhazard,
        _hazards;
        give_log = false,
        apply_transform = apply_transform,
        cache_context = cache_context)

    # return survival probability or not
    give_log ? log_survprob : exp(log_survprob)
end

"""
    total_cumulhaz(lb, ub, parameters, subjdat_row, _totalhazard::_TotalHazardTransient, _hazards; give_log = true) 

Return the log-total cumulative hazard out of a transient state over the interval [lb, ub].

PARAMETER CONVENTION: Expects natural-scale parameters (from unflatten_natural or get_hazard_params).
"""
function total_cumulhaz(lb, ub, parameters, subjdat_row, _totalhazard::_TotalHazardTransient, _hazards;
                        give_log = true,
                        apply_transform::Bool = false,
                        cache_context::Union{Nothing,TimeTransformContext}=nothing) 

    # Parameters should already be on natural scale (from unflatten_natural or get_hazard_params)
    # Use directly without additional transformation
    
    # log total cumulative hazard
    tot_haz = 0.0

    for x in _totalhazard.components
        hazard = _hazards[x]
        tot_haz += eval_cumhaz(
            hazard,
            lb,
            ub,
            parameters[hazard.hazname],
            subjdat_row;
            apply_transform = apply_transform,
            cache_context = cache_context,
            hazard_slot = x)
    end
    
    # return the log, or not
    give_log ? log(tot_haz) : tot_haz
end

"""
    total_cumulhaz with cached covariates (AbstractVector{<:NamedTuple})
"""
function total_cumulhaz(lb, ub, parameters, covars_cache::AbstractVector{<:NamedTuple}, _totalhazard::_TotalHazardTransient, _hazards;
                        give_log = true,
                        apply_transform::Bool = false,
                        cache_context::Union{Nothing,TimeTransformContext}=nothing)
    
    # Parameters should already be on natural scale
    tot_haz = 0.0

    for x in _totalhazard.components
        hazard = _hazards[x]
        covars = _covariate_entry(covars_cache, x)
        tot_haz += eval_cumhaz(
            hazard,
            lb,
            ub,
            parameters[hazard.hazname],
            covars;
            apply_transform = apply_transform,
            cache_context = cache_context,
            hazard_slot = x)
    end

    give_log ? log(tot_haz) : tot_haz
end

# =============================================================================
# Indexed Parameter Access (Performance Optimization)
# =============================================================================
#
# The following methods accept parameters as a Tuple (indexed by hazard position)
# instead of NamedTuple (indexed by symbol). This avoids runtime symbol lookup
# overhead in hot loops. Use `values(named_tuple)` to convert NamedTuple to Tuple.
#
# =============================================================================

"""
    total_cumulhaz(lb, ub, parameters::Tuple, ...)

Optimized version using indexed parameter access (Tuple instead of NamedTuple).
Call `values(params_named)` once outside the loop, then pass the tuple to this method.
"""
function total_cumulhaz(lb, ub, parameters::Tuple, subjdat_row, _totalhazard::_TotalHazardTransient, _hazards;
                        give_log = true,
                        apply_transform::Bool = false,
                        cache_context::Union{Nothing,TimeTransformContext}=nothing)
    tot_haz = 0.0
    
    for x in _totalhazard.components
        hazard = _hazards[x]
        tot_haz += eval_cumhaz(
            hazard, lb, ub, parameters[x], subjdat_row;  # Indexed access
            apply_transform = apply_transform,
            cache_context = cache_context,
            hazard_slot = x)
    end
    
    give_log ? log(tot_haz) : tot_haz
end

function total_cumulhaz(lb, ub, parameters::Tuple, covars_cache::AbstractVector{<:NamedTuple}, _totalhazard::_TotalHazardTransient, _hazards;
                        give_log = true,
                        apply_transform::Bool = false,
                        cache_context::Union{Nothing,TimeTransformContext}=nothing)
    tot_haz = 0.0
    
    for x in _totalhazard.components
        hazard = _hazards[x]
        covars = _covariate_entry(covars_cache, x)
        tot_haz += eval_cumhaz(
            hazard, lb, ub, parameters[x], covars;  # Indexed access
            apply_transform = apply_transform,
            cache_context = cache_context,
            hazard_slot = x)
    end
    
    give_log ? log(tot_haz) : tot_haz
end

"""
    survprob(lb, ub, parameters::Tuple, ...)

Optimized version using indexed parameter access.
"""
function survprob(lb, ub, parameters::Tuple, subjdat_row, _totalhazard::_TotalHazardTransient, _hazards;
                  give_log = true,
                  apply_transform::Bool = false,
                  cache_context::Union{Nothing,TimeTransformContext}=nothing)
    log_survprob = -total_cumulhaz(
        lb, ub, parameters, subjdat_row, _totalhazard, _hazards;
        give_log = false,
        apply_transform = apply_transform,
        cache_context = cache_context)
    
    give_log ? log_survprob : exp(log_survprob)
end

function survprob(lb, ub, parameters::Tuple, covars_cache::AbstractVector{<:NamedTuple}, _totalhazard::_TotalHazardTransient, _hazards;
                  give_log = true,
                  apply_transform::Bool = false,
                  cache_context::Union{Nothing,TimeTransformContext}=nothing)
    log_survprob = -total_cumulhaz(
        lb, ub, parameters, covars_cache, _totalhazard, _hazards;
        give_log = false,
        apply_transform = apply_transform,
        cache_context = cache_context)
    
    give_log ? log_survprob : exp(log_survprob)
end

# =============================================================================
# Absorbing State Handlers
# =============================================================================

"""
    total_cumulhaz(lb, ub, parameters, subjdat_row, _totalhazard::_TotalHazardAbsorbing, _hazards; give_log = true) 

Return zero log-total cumulative hazard over the interval [lb, ub] as the current state is absorbing.
"""
function total_cumulhaz(lb, ub, parameters, subjdat_row, _totalhazard::_TotalHazardAbsorbing, _hazards;
                        give_log = true,
                        apply_transform::Bool = false,
                        cache_context::Union{Nothing,TimeTransformContext}=nothing) 
    # return 0 cumulative hazard
    give_log ? -Inf : 0
end

function total_cumulhaz(lb, ub, parameters, covars_cache::AbstractVector{<:NamedTuple}, _totalhazard::_TotalHazardAbsorbing, _hazards;
                        give_log = true,
                        apply_transform::Bool = false,
                        cache_context::Union{Nothing,TimeTransformContext}=nothing)
    give_log ? -Inf : 0
end

"""
    survprob(lb, ub, parameters, subjdat_row, _totalhazard::_TotalHazardAbsorbing, _hazards; give_log = true)

Return survival probability = 1.0 (log = 0.0) for absorbing states, since no transitions can occur.
"""
function survprob(lb, ub, parameters, subjdat_row, _totalhazard::_TotalHazardAbsorbing, _hazards;
                  give_log = true,
                  apply_transform::Bool = false,
                  cache_context::Union{Nothing,TimeTransformContext}=nothing)
    give_log ? 0.0 : 1.0
end

function survprob(lb, ub, parameters, covars_cache::AbstractVector{<:NamedTuple}, _totalhazard::_TotalHazardTransient, _hazards;
                  give_log = true,
                  apply_transform::Bool = false,
                  cache_context::Union{Nothing,TimeTransformContext}=nothing)
    log_survprob = -total_cumulhaz(
        lb,
        ub,
        parameters,
        covars_cache,
        _totalhazard,
        _hazards;
        give_log = false,
        apply_transform = apply_transform,
        cache_context = cache_context)

    give_log ? log_survprob : exp(log_survprob)
end

# =============================================================================
# Next State Probability Functions
# =============================================================================

"""
    next_state_probs(t, scur, subjdat_row, parameters, hazards, totalhazards, tmat;
                     apply_transform = false,
                     cache_context = nothing)

Return a vector `ns_probs` with probabilities of transitioning to each state based on hazards from the current state.

# Arguments 
- `t`: time at which hazards should be calculated
- `scur`: current state
- `subjdat_row`: DataFrame row containing subject covariates
- `parameters`: vector of vectors of model parameters
- `hazards`: vector of cause-specific hazards
- `totalhazards`: vector of total hazards
- `tmat`: transition matrix
- `apply_transform`: pass `true` to allow Tang-enabled hazards to reuse cached trajectories
- `cache_context`: optional `TimeTransformContext` shared across hazards
"""
function _next_state_probs!(ns_probs::AbstractVector{Float64}, trans_inds::AbstractVector{Int}, t, scur, covars_cache, parameters, hazards, totalhazards;
                           apply_transform::Bool,
                           cache_context::Union{Nothing,TimeTransformContext})
    fill!(ns_probs, 0.0)
    isempty(trans_inds) && return ns_probs

    if length(trans_inds) == 1
        ns_probs[trans_inds[1]] = 1.0
        return ns_probs
    end

    # Compute log-hazards for softmax
    # Support both NamedTuple (symbol access) and Tuple (index access)
    vals = map(totalhazards[scur].components) do x
        hazard = hazards[x]
        covars = _covariate_entry(covars_cache, x)
        # Use indexed access for Tuple, symbol access for NamedTuple
        hazard_pars = parameters isa Tuple ? parameters[x] : parameters[hazard.hazname]
        haz = eval_hazard(hazard, t, hazard_pars, covars;
                          apply_transform = apply_transform && hazard.metadata.time_transform,
                          cache_context = cache_context,
                          hazard_slot = x)
        log(haz)  # softmax expects log scale
    end
    ns_probs[trans_inds] = softmax(vals)

    local_probs = view(ns_probs, trans_inds)
    if all(isnan.(local_probs))
        local_probs .= 1 / length(trans_inds)
    elseif any(isnan.(local_probs))
        pisnan = findall(isnan.(local_probs))
        if length(pisnan) == 1
            local_probs[pisnan] = 1 - sum(local_probs[Not(pisnan)])
        else
            local_probs[pisnan] .= (1 - sum(local_probs[Not(pisnan)])) / length(pisnan)
        end
    end

    return ns_probs
end

function next_state_probs!(ns_probs::AbstractVector{Float64}, trans_inds::AbstractVector{Int}, t, scur, subjdat_row, parameters, hazards, totalhazards;
                           apply_transform::Bool = false,
                           cache_context::Union{Nothing,TimeTransformContext}=nothing)
    return _next_state_probs!(ns_probs, trans_inds, t, scur, subjdat_row, parameters, hazards, totalhazards;
        apply_transform = apply_transform,
        cache_context = cache_context)
end

function next_state_probs(t, scur, subjdat_row, parameters, hazards, totalhazards, tmat;
                          apply_transform::Bool = false,
                          cache_context::Union{Nothing,TimeTransformContext}=nothing)
    ns_probs = zeros(size(tmat, 2))
    trans_inds = findall(tmat[scur,:] .!= 0.0)
    return next_state_probs!(ns_probs, trans_inds, t, scur, subjdat_row, parameters, hazards, totalhazards;
        apply_transform = apply_transform,
        cache_context = cache_context)
end
