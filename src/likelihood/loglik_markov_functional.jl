# =============================================================================
# Non-Mutating Markov Likelihood (Reverse-Mode AD Compatible)
# =============================================================================
#
# Contents:
# - _loglik_markov_functional: Pure functional forward algorithm
# - _forward_algorithm_functional: Accumulator-free forward pass
# - setindex_immutable_vec: Immutable vector indexing for AD
#
# This implementation supports Enzyme and Mooncake reverse-mode AD backends
# by avoiding all in-place mutations.
#
# Split from loglik.jl for maintainability (January 2026)
# =============================================================================

"""
    _loglik_markov_functional(parameters, data::MPanelData; neg=true)

Non-mutating implementation of Markov panel likelihood for reverse-mode AD (Enzyme, Mooncake).

This version avoids all in-place mutations:
- Uses `compute_hazmat` and `compute_tmat` instead of their `!` variants
- Accumulates log-likelihood functionally without pre-allocated containers
- Uses comprehensions and reductions instead of loops with mutation

Performance: ~10-20% slower than mutating version due to allocations,
but enables reverse-mode gradient computation which may be faster overall for large models.

This is an internal function. Use `loglik_markov(params, data; backend=EnzymeBackend())` 
for the public API.

# Arguments
- `parameters`: Flat parameter vector
- `data::MPanelData`: Markov panel data container
- `neg::Bool=true`: Return negative log-likelihood

# Returns
Scalar (negative) log-likelihood value.
"""
function _loglik_markov_functional(parameters, data::MPanelData; neg = true)
    # Unflatten parameters to natural scale (AD-compatible)
    pars = unflatten_parameters(parameters, data.model)
    
    # Model components
    hazards = data.model.hazards
    tmat = data.model.tmat
    n_states = size(tmat, 1)
    T = eltype(parameters)
    
    # Pre-extracted column accessors for O(1) access (avoids DataFrame dispatch overhead)
    cols = data.columns
    
    # Build TPM book functionally (no mutation)
    # For each unique covariate/time pattern, compute Q then P = exp(Q*dt)
    tpm_dict = Dict{Tuple{Int,Int}, Matrix{T}}()
    for (t_idx, tpm_index_df) in enumerate(data.books[1])
        Q = compute_hazmat(T, n_states, pars, hazards, tpm_index_df, data.model.data)
        # Compute P for each time interval in this pattern
        for t in eachindex(tpm_index_df.tstop)
            dt = tpm_index_df.tstop[t]
            P = compute_tmat(Q, dt)
            # Store with composite key (could be optimized)
            tpm_dict[(t_idx, t)] = P
        end
    end
    
    # Accumulate log-likelihood functionally
    nsubj = length(data.model.subjectindices)
    has_obs_weights = !isnothing(data.model.ObservationWeights)
    
    # Compute subject contributions using map (no mutation)
    subj_contributions = map(1:nsubj) do subj
        subj_inds = data.model.subjectindices[subj]
        subj_weight = data.model.SubjectWeights[subj]
        
        # Check if any censored observations using direct column access
        # Manual loop avoids broadcast allocations from all(... .∈ Ref([1,2]))
        all_uncensored = true
        for i in subj_inds
            if cols.obstype[i] > 2
                all_uncensored = false
                break
            end
        end
        
        if all_uncensored
            # Simple case: no forward algorithm needed
            subj_ll = sum(subj_inds) do i
                obs_weight = has_obs_weights ? data.model.ObservationWeights[i] : one(T)
                obstype_i = cols.obstype[i]
                
                if obstype_i == 1  # exact data
                    row_data = @view data.model.data[i, :]
                    statefrom_i = cols.statefrom[i]
                    stateto_i = cols.stateto[i]
                    dt = cols.tstop[i] - cols.tstart[i]
                    
                    obs_ll = dt ≈ 0 ? 0 : survprob(zero(T), dt, pars, row_data, 
                                     data.model.totalhazards[statefrom_i], hazards; 
                                     give_log = true)
                    
                    if statefrom_i != stateto_i
                        trans_idx = tmat[statefrom_i, stateto_i]
                        hazard = hazards[trans_idx]
                        hazard_pars = pars[hazard.hazname]
                        haz_value = eval_hazard(hazard, dt, hazard_pars, row_data)
                        obs_ll += NaNMath.log(haz_value)
                    end
                    
                    obs_ll * obs_weight
                else  # panel data (obstype == 2, since we're in all_exact_or_panel branch)
                    # Forward algorithm for single observation:
                    # L = Σ_s P(X=s | X_0=statefrom) * P(Y | X=s)
                    #   = Σ_s TPM[statefrom, s] * emat[i, s]
                    statefrom_i = cols.statefrom[i]
                    book_idx1 = cols.tpm_map_col1[i]
                    book_idx2 = cols.tpm_map_col2[i]
                    P = tpm_dict[(book_idx1, book_idx2)]
                    
                    # Compute likelihood: Σ_s P(transition to s) * P(observation | s)
                    prob_sum = zero(T)
                    for s in 1:n_states
                        emission_prob = data.model.emat[i, s]
                        if emission_prob > zero(T)
                            prob_sum += P[statefrom_i, s] * emission_prob
                        end
                    end
                    NaNMath.log(prob_sum) * obs_weight
                end
            end
            
            subj_ll * subj_weight
        else
            # Forward algorithm for censored observations
            # Build log-likelihood via matrix-vector products
            subj_ll = _forward_algorithm_functional(
                subj_inds, pars, data, tpm_dict, T, n_states, hazards, tmat
            )
            subj_ll * subj_weight
        end
    end
    
    ll = sum(subj_contributions)
    return neg ? -ll : ll
end

# Backward compatibility alias
# Use loglik_markov(params, data; backend=EnzymeBackend()) instead
const loglik_markov_functional = _loglik_markov_functional

"""
    _forward_algorithm_functional(subj_inds, pars, data, tpm_dict, T, n_states, hazards, tmat)

Non-mutating forward algorithm for censored state observations.
Returns the log-likelihood contribution for one subject.
"""
function _forward_algorithm_functional(subj_inds, pars, data, tpm_dict, ::Type{T}, 
                                       n_states::Int, hazards, tmat) where T
    # Pre-extracted column accessors for O(1) access
    cols = data.columns
    
    # Initialize: probability vector for initial state
    # For phase-type models with panel data, we have phase uncertainty at the initial state:
    # if the subject is observed "in state s" at time 0, they could be in any phase of state s.
    first_obs_idx = subj_inds[1]
    init_state = cols.statefrom[first_obs_idx]
    α = zeros(T, n_states)
    
    phasetype_exp = data.model.phasetype_expansion
    if !isnothing(phasetype_exp)
        # Phase-type model: check if starting state has multiple phases
        mappings = phasetype_exp.mappings
        observed_state = mappings.phase_to_state[init_state]
        phases_in_state = mappings.state_to_phases[observed_state]
        for phase in phases_in_state
            α = setindex_immutable_vec(α, one(T), phase)  # Uniform over phases
        end
    else
        # Standard model: known starting state
        α = setindex_immutable_vec(α, one(T), init_state)
    end
    
    # Forward pass: α[t+1] = α[t] * P[t] (with emission probabilities for censored states)
    for i in subj_inds
        obstype_i = cols.obstype[i]
        dt = cols.tstop[i] - cols.tstart[i]
        
        if obstype_i == 1
            # Exact data: compute transition probabilities directly
            row_data = @view data.model.data[i, :]
            statefrom = cols.statefrom[i]
            stateto = cols.stateto[i]
            
            # Survival probability + hazard
            log_surv = dt ≈ 0 ? 0 : survprob(zero(T), dt, pars, row_data,
                               data.model.totalhazards[statefrom], hazards; give_log = true)
            
            if statefrom != stateto
                trans_idx = tmat[statefrom, stateto]
                hazard = hazards[trans_idx]
                hazard_pars = pars[hazard.hazname]
                haz_value = eval_hazard(hazard, dt, hazard_pars, row_data)
                log_prob = log_surv + NaNMath.log(haz_value)
            else
                log_prob = log_surv
            end
            
            # Update probability vector: concentrate mass on observed state
            new_α = zeros(T, n_states)
            prob_from = α[statefrom]
            new_α = setindex_immutable_vec(new_α, prob_from * exp(log_prob), stateto)
            α = new_α
        else
            # Panel/censored data: multiply by TPM
            book_idx1 = cols.tpm_map_col1[i]
            book_idx2 = cols.tpm_map_col2[i]
            P = tpm_dict[(book_idx1, book_idx2)]
            
            # Matrix-vector product (non-mutating)
            α = P' * α  # transpose because α is a column vector
            
            # Apply emission probabilities: α_i(s) = e_{is} * α_i(s)
            # The emission matrix is constructed by build_emat():
            # - For exact observations: e[i, stateto] = 1, e[i, other] = 0
            # - For censored observations: e[i, :] from CensoringPatterns
            # - For user-supplied EmissionMatrix: uses provided soft evidence
            #
            # This unified approach enables soft evidence for any observation type.
            new_α = zeros(T, n_states)
            for s in 1:n_states
                emission_prob = data.model.emat[i, s]
                if emission_prob > zero(T)
                    new_α = setindex_immutable_vec(new_α, new_α[s] + α[s] * emission_prob, s)
                end
            end
            α = new_α
        end
    end
    
    # Log-likelihood is log of sum of final probabilities
    return NaNMath.log(sum(α))
end

"""
    setindex_immutable_vec(v, val, i)

Return a new vector with v[i] = val without mutating v.
"""
@inline function setindex_immutable_vec(v::AbstractVector{T}, val, i::Int) where T
    w = copy(v)
    w[i] = convert(T, val)
    return w
end

"""
    loglik_semi_markov(parameters, data::SMPanelData; neg=true, use_sampling_weight=true, parallel=false)

Compute importance-weighted log-likelihood for semi-Markov panel data (MCEM).

This function computes:
```math
Q(θ|θ') = Σᵢ SubjectWeights[i] × Σⱼ ImportanceWeights[i][j] × ℓᵢⱼ(θ)
```

where `ℓᵢⱼ` is the complete-data log-likelihood for path j of subject i.

# Arguments
- `parameters`: Flat parameter vector
- `data::SMPanelData`: Semi-Markov panel data with sample paths and importance weights
- `neg::Bool=true`: Return negative log-likelihood
- `use_sampling_weight::Bool=true`: Apply subject sampling weights
- `parallel::Bool=false`: Use multi-threaded parallel computation

# Parallel Execution
When `parallel=true` and `Threads.nthreads() > 1`, uses flat path-level parallelism
with `@threads :static`. This provides good load balance even when subjects have
highly variable numbers of paths (e.g., some subjects have 1 path, others have 500).

The flat-indexing approach maps each (subject, path) pair to a linear index,
ensuring work is distributed evenly across threads regardless of path distribution.

# Returns
Scalar (negative) log-likelihood value.

# See Also
- [`mcem_mll`](@ref): Uses this for MCEM objective computation
- [`loglik_semi_markov!`](@ref): In-place version for path log-likelihoods
"""
