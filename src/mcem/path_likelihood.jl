# =============================================================================
# MCEM Path Likelihood Dispatch
# =============================================================================
#
# This file contains dispatch methods for computing:
# 1. Normalizing constant (marginal likelihood under surrogate)
# 2. Surrogate log-likelihood for individual paths
# 3. Path collapsing (PhaseType expanded → original states)
#
# All functions dispatch on MCEMInfrastructure type parameter.
#
# =============================================================================

# =============================================================================
# Normalizing Constant
# =============================================================================

"""
    compute_normalizing_constant(model::MultistateModel, infra::MCEMInfrastructure)

Compute the log normalizing constant for importance sampling.

This is the marginal log-likelihood of the data under the surrogate model:
- For Markov: direct matrix exponential likelihood
- For PhaseType: forward algorithm on expanded state space

# Returns
`Float64`: Log normalizing constant (sum over subjects)
"""
function compute_normalizing_constant end

function compute_normalizing_constant(model::MultistateModel, infra::MCEMInfrastructure{MarkovSurrogate})
    # Markov marginal likelihood via matrix exponential forward algorithm
    compute_markov_marginal_loglik(model, infra.surrogate)
end

function compute_normalizing_constant(model::MultistateModel, infra::MCEMInfrastructure{PhaseTypeSurrogate})
    # Phase-type marginal likelihood via forward algorithm on expanded space
    # Uses expanded data and emission matrix if data was expanded
    expanded_data = isnothing(infra.original_row_map) ? nothing : infra.data
    expanded_subjectindices = isnothing(infra.original_row_map) ? nothing : infra.subjectindices
    
    compute_phasetype_marginal_loglik(
        model, infra.surrogate, infra.emat;
        expanded_data = expanded_data,
        expanded_subjectindices = expanded_subjectindices
    )
end

# =============================================================================
# Surrogate Log-Likelihood for Paths
# =============================================================================

"""
    compute_surrogate_path_loglik(path::SamplePath, subj_idx::Int, 
                                   model::MultistateModel, infra::MCEMInfrastructure;
                                   expanded_path=nothing, subj_data=nothing, subj_tpm_map=nothing)

Compute log q(path | surrogate) for importance weight denominator.

For Markov surrogate: CTMC path density
For PhaseType surrogate: Marginal over phase sequences via forward algorithm

# Arguments
- `path`: The collapsed sample path (on original state space)
- `subj_idx`: Subject index
- `model`: The target model
- `infra`: MCEM infrastructure

# Keyword arguments (PhaseType only)
- `expanded_path`: The expanded path (on phase state space)
- `subj_data`: Subject's data view
- `subj_tpm_map`: Subject's TPM map view

# Returns
`Float64`: Log surrogate likelihood
"""
function compute_surrogate_path_loglik end

function compute_surrogate_path_loglik(
    path::SamplePath, 
    subj_idx::Int,
    model::MultistateModel, 
    infra::MCEMInfrastructure{MarkovSurrogate};
    expanded_path=nothing, subj_data=nothing, subj_tpm_map=nothing
)
    # Markov surrogate: compute CTMC path density using existing loglik function
    # log q(path | θ_surrogate) = Σ_transitions [log h_ij(t) - ∫₀ᵗ Σ_j h_ij(s) ds]
    
    # Get surrogate parameters (family-aware)
    surrogate_pars = get_hazard_params(infra.surrogate.parameters, infra.surrogate.hazards)
    
    # Use existing path likelihood computation
    loglik(surrogate_pars, path, infra.surrogate.hazards, model)
end

function compute_surrogate_path_loglik(
    path::SamplePath, 
    subj_idx::Int,
    model::MultistateModel, 
    infra::MCEMInfrastructure{PhaseTypeSurrogate};
    expanded_path=nothing, subj_data=nothing, subj_tpm_map=nothing
)
    # Phase-type surrogate: compute marginal likelihood over phase sequences
    # log q(collapsed_path | θ_surrogate) = log Σ_{expanded} q(expanded | θ_surrogate)
    #
    # This is computed via forward algorithm on the censored data representation
    # of the collapsed path, which marginalizes over all phase configurations.
    
    @assert !isnothing(expanded_path) "PhaseType surrogate requires expanded_path"
    @assert !isnothing(subj_data) "PhaseType surrogate requires subj_data"
    
    # Determine if we have time-varying covariates
    has_tvc = length(infra.hazmat_book) > 1
    
    # Convert expanded path to censored data format for forward algorithm
    if has_tvc
        censored_data, emat_path, tpm_map_path, tpm_book_path, hazmat_book_path = 
            convert_expanded_path_to_censored_data(
                expanded_path, infra.surrogate;
                original_subj_data = subj_data,
                hazmat_book = infra.hazmat_book,
                schur_cache_book = infra.schur_cache,
                subj_tpm_map = subj_tpm_map
            )
    else
        censored_data, emat_path, tpm_map_path, tpm_book_path, hazmat_book_path = 
            convert_expanded_path_to_censored_data(
                expanded_path, infra.surrogate;
                hazmat = infra.surrogate.expanded_Q,
                schur_cache = isnothing(infra.schur_cache) ? nothing : infra.schur_cache[1]
            )
    end
    
    # Compute forward log-likelihood (marginalizes over phases)
    compute_forward_loglik(
        censored_data, emat_path, tpm_map_path, tpm_book_path, 
        hazmat_book_path, infra.surrogate.n_expanded_states
    )
end

# =============================================================================
# Path Collapsing (PhaseType only)
# =============================================================================

"""
    collapse_expanded_path(expanded_path::SamplePath, infra::MCEMInfrastructure{PhaseTypeSurrogate})

Collapse an expanded (phase-level) path to original (macro-state) path.

# Arguments
- `expanded_path`: Path on expanded state space (phases)
- `infra`: PhaseType infrastructure containing state mappings

# Returns
`SamplePath` on original state space
"""
function collapse_expanded_path(expanded_path::SamplePath, infra::MCEMInfrastructure{PhaseTypeSurrogate})
    # Map each phase state back to its macro-state
    phase_to_state = infra.surrogate.phase_to_state
    
    # Collapse states
    collapsed_states = [phase_to_state[s] for s in expanded_path.states]
    
    # Remove consecutive duplicates (phases in same macro-state)
    unique_times = Float64[]
    unique_states = Int[]
    
    push!(unique_times, expanded_path.times[1])
    push!(unique_states, collapsed_states[1])
    
    for i in 2:length(collapsed_states)
        if collapsed_states[i] != collapsed_states[i-1]
            push!(unique_times, expanded_path.times[i])
            push!(unique_states, collapsed_states[i])
        end
    end
    
    # Add final time if different from last recorded
    if unique_times[end] != expanded_path.times[end]
        push!(unique_times, expanded_path.times[end])
    end
    
    SamplePath(unique_times, unique_states)
end

# No-op for Markov (paths are already on original state space)
collapse_expanded_path(path::SamplePath, infra::MCEMInfrastructure{MarkovSurrogate}) = path

# =============================================================================
# Convenience: Get surrogate parameters
# =============================================================================

"""
    get_surrogate_hazard_params(infra::MCEMInfrastructure)

Get hazard parameters from the surrogate in infrastructure.

# Returns
Named tuple of hazard parameters suitable for hazard evaluation.
"""
get_surrogate_hazard_params(infra::MCEMInfrastructure) = 
    get_hazard_params(infra.surrogate.parameters, infra.surrogate.hazards)
