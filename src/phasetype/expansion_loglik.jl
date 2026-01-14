# =============================================================================
# Phase-Type Log-Likelihood Functions
# =============================================================================
# These functions compute likelihoods for phase-type surrogates used in MCEM.
# They use the forward algorithm on the expanded state space with proper
# emission matrix handling for censoring.
# =============================================================================

"""
    compute_phasetype_marginal_loglik(model, surrogate, emat_ph; kwargs...)

Compute the marginal log-likelihood of observed data under the phase-type surrogate.

This is used as the normalizing constant r(Y|θ') in importance sampling:
    log f̂(Y|θ) = log r(Y|θ') + Σᵢ log(mean(νᵢ))

Uses the forward algorithm on the expanded phase state space.

# Arguments
- `model::MultistateProcess`: The multistate model containing the data
- `surrogate::PhaseTypeSurrogate`: The fitted phase-type surrogate
- `emat_ph::Matrix{Float64}`: Expanded emission matrix for phase states

# Keyword Arguments
- `expanded_data`: Optional expanded data for exact observations
- `expanded_subjectindices`: Subject indices for expanded data

# Returns
- `Float64`: The marginal log-likelihood under the phase-type surrogate
"""
function compute_phasetype_marginal_loglik(model::MultistateProcess, 
                                           surrogate::PhaseTypeSurrogate,
                                           emat_ph::Matrix{Float64};
                                           expanded_data::Union{Nothing, DataFrame} = nothing,
                                           expanded_subjectindices::Union{Nothing, Vector{UnitRange{Int64}}} = nothing)
    
    Q = surrogate.expanded_Q
    n_expanded = surrogate.n_expanded_states
    
    data = isnothing(expanded_data) ? model.data : expanded_data
    subjectindices = isnothing(expanded_subjectindices) ? model.subjectindices : expanded_subjectindices
    
    cache = ExponentialUtilities.alloc_mem(similar(Q), ExpMethodGeneric())
    
    ll_total = 0.0
    
    for subj_idx in eachindex(subjectindices)
        subj_inds = subjectindices[subj_idx]
        n_obs = length(subj_inds)
        
        times = vcat(data.tstart[subj_inds[1]], data.tstop[subj_inds])
        statefrom_subj = data.statefrom[subj_inds]
        stateto_subj = data.stateto[subj_inds]
        obstype_subj = data.obstype[subj_inds]
        subj_emat = emat_ph[subj_inds, :]
        
        # Initialize forward variable
        α = zeros(Float64, n_expanded)
        initial_state = statefrom_subj[1]
        initial_phase = first(surrogate.state_to_phases[initial_state])
        α[initial_phase] = 1.0
        
        log_ll = 0.0
        α_new = zeros(Float64, n_expanded)
        P = similar(Q)
        Q_scaled = similar(Q)
        
        for k in 1:n_obs
            Δt = times[k + 1] - times[k]
            
            if Δt > 0
                copyto!(Q_scaled, Q)
                Q_scaled .*= Δt
                copyto!(P, exponential!(Q_scaled, ExpMethodGeneric(), cache))
                
                fill!(α_new, 0.0)
                
                if obstype_subj[k] ∈ [1, 2] && stateto_subj[k] > 0
                    obs_state = stateto_subj[k]
                    allowed_phases = surrogate.state_to_phases[obs_state]
                    for j in allowed_phases
                        for i in 1:n_expanded
                            α_new[j] += P[i, j] * α[i]
                        end
                    end
                else
                    for j in 1:n_expanded
                        e_j = subj_emat[k, j]
                        if e_j > 0
                            for i in 1:n_expanded
                                α_new[j] += P[i, j] * α[i] * e_j
                            end
                        end
                    end
                end
                
                copyto!(α, α_new)
            else
                # Δt = 0: Instantaneous observation (no time passes)
                if obstype_subj[k] ∈ [1, 2] && stateto_subj[k] > 0
                    obs_state = stateto_subj[k]
                    obs_phases = surrogate.state_to_phases[obs_state]
                    
                    # Check if we're already in the observed state or need to transition
                    already_in_state = false
                    for j in obs_phases
                        if α[j] > 0
                            already_in_state = true
                            break
                        end
                    end
                    
                    if already_in_state
                        # Already in the observed state - just zero out other states
                        for j in 1:n_expanded
                            if surrogate.phase_to_state[j] != obs_state
                                α[j] = 0.0
                            end
                        end
                    else
                        # Need to transition INTO the observed state
                        # Use Q matrix to compute rate-weighted probability transfer:
                        # α_new[j] = Σᵢ Q[i,j] * α[i] for j in destination phases
                        fill!(α_new, 0.0)
                        for j in obs_phases
                            for i in 1:n_expanded
                                if α[i] > 0 && Q[i, j] > 0
                                    α_new[j] += Q[i, j] * α[i]
                                end
                            end
                        end
                        copyto!(α, α_new)
                    end
                else
                    for j in 1:n_expanded
                        α[j] *= subj_emat[k, j]
                    end
                end
            end
            
            scale = sum(α)
            if scale > 0
                log_ll += log(scale)
                α ./= scale
            else
                log_ll = -Inf
                break
            end
        end
        
        log_ll += log(sum(α))
        ll_total += log_ll * model.SubjectWeights[subj_idx]
    end
    
    return ll_total
end

# =============================================================================
