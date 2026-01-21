# =============================================================================
# Markov Panel Data Likelihood (Forward Algorithm Implementation)
# =============================================================================
#
# Contents:
# - Generic dispatch methods (loglik, loglik!)
# - loglik_markov: Main entry point for Markov panel likelihood
# - _loglik_markov_mutating: ForwardDiff-compatible forward algorithm
# - Forward pass with eigendecomposition caching
# - Subject-level parallel evaluation
#
# Split from loglik.jl for maintainability (January 2026)
# =============================================================================

# =============================================================================
# Generic dispatch methods for loglik and loglik!
# =============================================================================
#
# These dispatch methods provide a unified interface for likelihood computation
# across different data types. The MCEM algorithm calls loglik(p, data) and
# loglik!(p, logliks, data) generically, and these methods dispatch to the
# appropriate specialized implementations.
# =============================================================================

"""
    loglik(parameters, data::SMPanelData; neg=true, use_sampling_weight=true)

Dispatch method for semi-Markov panel data. Calls `loglik_semi_markov`.

See [`loglik_semi_markov`](@ref) for details.
"""
loglik(parameters, data::SMPanelData; neg=true, use_sampling_weight=true) = 
    loglik_semi_markov(parameters, data; neg=neg, use_sampling_weight=use_sampling_weight)

"""
    loglik!(parameters, logliks::Vector{}, data::SMPanelData)

In-place dispatch method for semi-Markov panel data. Calls `loglik_semi_markov!`.

See [`loglik_semi_markov!`](@ref) for details.
"""
loglik!(parameters, logliks::Vector{}, data::SMPanelData) = 
    loglik_semi_markov!(parameters, logliks, data)

"""
    loglik(parameters, data::MPanelData; neg=true, return_ll_subj=false)

Dispatch method for Markov panel data. Calls `loglik_markov`.

See [`loglik_markov`](@ref) for details.
"""
loglik(parameters, data::MPanelData; neg=true, return_ll_subj=false) = 
    loglik_markov(parameters, data; neg=neg, return_ll_subj=return_ll_subj)

"""
    loglik_AD(parameters, data::ExactDataAD; neg = true) 

Compute (negative) log-likelihood for a single sample path with AD support.

This function is used for per-path Fisher information computation in variance 
estimation, where we need to differentiate the likelihood of individual paths.
Unlike `loglik_exact` which operates on `ExactData` (multiple paths), this
operates on `ExactDataAD` containing a single path.

# Arguments
- `parameters`: Flat parameter vector
- `data::ExactDataAD`: Single path data container
- `neg::Bool=true`: Return negative log-likelihood

# Returns
Scalar (negative) log-likelihood for the single path, weighted by sampling weight.

See also: [`loglik_exact`](@ref), [`ExactDataAD`](@ref)
"""
function loglik_AD(parameters, data::ExactDataAD; neg = true)

    # unflatten parameters
    pars = unflatten_parameters(parameters, data.model)

    # snag the hazards
    hazards = data.model.hazards

    # send each element of samplepaths to loglik
    # Convert SamplePath to DataFrame using make_subjdat
    path = data.path[1]
    subj_inds = data.model.subjectindices[path.subj]
    subj_dat = view(data.model.data, subj_inds, :)
    subjdat_df = make_subjdat(path, subj_dat)
    ll = loglik_path(pars, subjdat_df, hazards, data.model.totalhazards, data.model.tmat) * data.samplingweight[1]

    neg ? -ll : ll
end

"""
    loglik_markov(parameters, data::MPanelData; neg=true, return_ll_subj=false, backend=nothing)

Unified Markov panel likelihood with automatic AD backend dispatch.

Return sum of (negative) log likelihood for a Markov model fit to panel and/or exact 
and/or censored data.

# Arguments
- `parameters`: Flat parameter vector
- `data::MPanelData`: Markov panel data container
- `neg::Bool=true`: Return negative log-likelihood
- `return_ll_subj::Bool=false`: Return vector of subject-level log-likelihoods
- `backend::Union{Nothing,ADBackend}=nothing`: AD backend for implementation dispatch
  - `nothing` or `ForwardDiffBackend()`: Use mutating implementation (default)
  - `EnzymeBackend()` or `MooncakeBackend()`: Use non-mutating functional implementation

# Implementation Details
- **Mutating implementation** (`_loglik_markov_mutating`): Uses in-place operations
  (`compute_hazmat!`, `compute_tmat!`) for efficiency. Compatible with ForwardDiff.
- **Functional implementation** (`_loglik_markov_functional`): Avoids all mutations
  for reverse-mode AD compatibility (Enzyme, Mooncake). ~10-20% slower due to allocations.

# Returns
- If `return_ll_subj=false`: Scalar (negative) log-likelihood
- If `return_ll_subj=true`: Vector of per-subject weighted log-likelihoods

# Note
For reverse-mode AD backends (Enzyme, Mooncake), `return_ll_subj` is not supported
and will raise an error.

See also: [`loglik_exact`](@ref), [`loglik_semi_markov`](@ref)
"""
function loglik_markov(parameters, data::MPanelData; 
                       neg = true, 
                       return_ll_subj = false,
                       backend::Union{Nothing, ADBackend} = nothing)
    # Dispatch based on backend type
    if isnothing(backend) || backend isa ForwardDiffBackend
        # Use mutating implementation (default, ForwardDiff-compatible)
        return _loglik_markov_mutating(parameters, data; neg=neg, return_ll_subj=return_ll_subj)
    elseif backend isa EnzymeBackend || backend isa MooncakeBackend
        # Use functional implementation (reverse-mode AD compatible)
        if return_ll_subj
            throw(ArgumentError("return_ll_subj=true is not supported with reverse-mode AD backends"))
        end
        return _loglik_markov_functional(parameters, data; neg=neg)
    else
        throw(ArgumentError("Unknown AD backend type: $(typeof(backend))"))
    end
end

"""
    _loglik_markov_mutating(parameters, data::MPanelData; neg=true, return_ll_subj=false)

Mutating implementation of Markov panel likelihood. Uses in-place operations for efficiency. Compatible with ForwardDiff (forward-mode AD).

This is an internal function. Use `loglik_markov` for the public API.
"""
function _loglik_markov_mutating(parameters, data::MPanelData; neg = true, return_ll_subj = false)

    # Element type for AD compatibility
    T = eltype(parameters)

    # Pre-extracted column accessors for O(1) access (avoids DataFrame dispatch overhead)
    cols = data.columns
    
    # Check if we can use pre-computed hazard rates (Markov models only)
    # For Markov hazards, rates are time-invariant and can be computed once per likelihood call
    # But interaction terms (e.g., trt * age) can't be pre-cached, so check validity
    is_markov = data.model isa MultistateProcess
    can_use_rate_cache = is_markov && T === Float64 && 
                         !isempty(data.cache.hazard_rates_cache) &&
                         _covars_cache_valid(data.cache.covars_cache, data.model.hazards)

    # Use cached arrays for Float64, allocate fresh for Dual types (AD)
    if T === Float64
        # Reuse pre-allocated cache (zero-allocation path for non-AD evaluations)
        hazmat_book = data.cache.hazmat_book
        tpm_book = data.cache.tpm_book
        exp_cache = data.cache.exp_cache
        
        # Unflatten to NamedTuple structure (required by hazard functions)
        # NamedTuple allocations are minimal (~13) and mostly stack-allocated
        pars = unflatten_parameters(parameters, data.model)
        
        # Pre-compute hazard rates once for all patterns (Markov models only)
        if can_use_rate_cache
            compute_hazard_rates!(data.cache.hazard_rates_cache, pars, 
                                  data.model.hazards, data.cache.covars_cache)
        end
        
        # Reset hazard matrices and compute TPMs
        # Use batched eigendecomposition approach when multiple Δt values exist
        @inbounds for t in eachindex(data.books[1])
            # Use cached rates path for Markov models
            if can_use_rate_cache
                compute_hazmat_from_rates!(
                    hazmat_book[t],
                    data.cache.hazard_rates_cache[t],
                    data.model.hazards)
            else
                fill!(hazmat_book[t], zero(Float64))
                compute_hazmat!(
                    hazmat_book[t],
                    pars,
                    data.model.hazards,
                    data.books[1][t],
                    data.model.data)
            end
            
            # Use batched Schur approach when >= 2 unique Δt values (O(n²) per Δt pays off)
            n_dt = length(data.cache.dt_values[t])
            if n_dt >= 2
                compute_tmat_batched!(
                    tpm_book[t],
                    hazmat_book[t],
                    data.cache.dt_values[t],
                    data.cache.schur_cache,
                    t)
            else
                # Single Δt: use standard approach
                compute_tmat!(
                    tpm_book[t],
                    hazmat_book[t],
                    data.books[1][t],
                    exp_cache)
            end
        end
    else
        # Allocate fresh arrays for AD (element type must be Dual)
        pars = unflatten_parameters(parameters, data.model)
        hazmat_book = build_hazmat_book(T, data.model.tmat, data.books[1])
        tpm_book = build_tpm_book(T, data.model.tmat, data.books[1])
        exp_cache = ExponentialUtilities.alloc_mem(similar(hazmat_book[1]), ExpMethodGeneric())
        
        # Solve Kolmogorov equations for TPMs
        @inbounds for t in eachindex(data.books[1])
            compute_hazmat!(
                hazmat_book[t],
                pars,
                data.model.hazards,
                data.books[1][t],
                data.model.data)
            compute_tmat!(
                tpm_book[t],
                hazmat_book[t],
                data.books[1][t],
                exp_cache)
        end
    end

    # number of subjects
    nsubj = length(data.model.subjectindices)

    # accumulate the log likelihood
    ll = zero(T)

    # container for subject-level loglikelihood
    if return_ll_subj
        ll_subj = zeros(T, nsubj)
    end

    # number of states
    S = size(data.model.tmat, 1)

    # initialize Q (use cached version for Float64, allocate for AD)
    if T === Float64
        q = data.cache.q_work
        fill!(q, zero(Float64))
    else
        q = zeros(T, S, S)
    end
    
    # check if observation weights are provided (once, outside loop)
    has_obs_weights = !isnothing(data.model.ObservationWeights)
    
    # Cache hazard covar_names to avoid repeated lookups (once, outside loop)
    hazards = data.model.hazards

    # for each subject, compute the likelihood contribution
    for subj in Base.OneTo(nsubj)

        # subject data
        subj_inds = data.model.subjectindices[subj]

        # Check if all observations are exact (1) or panel (2) - no censoring
        # Manual loop avoids broadcast allocations from all(... .∈ Ref([1,2]))
        all_exact_or_panel = true
        @inbounds for i in subj_inds
            if cols.obstype[i] > 2
                all_exact_or_panel = false
                break
            end
        end

        # no state is censored
        if all_exact_or_panel
            
            # subject contribution to the loglikelihood
            subj_ll = zero(T)

            # add the contribution of each observation
            @inbounds for i in subj_inds
                # get observation weight (default to 1.0)
                obs_weight = has_obs_weights ? data.model.ObservationWeights[i] : 1.0
                
                obstype_i = cols.obstype[i]
                
                if obstype_i == 1 # exact data
                    # Use @view to avoid DataFrameRow allocation
                    row_data = @view data.model.data[i, :]
                    
                    statefrom_i = cols.statefrom[i]
                    stateto_i = cols.stateto[i]
                    dt = cols.tstop[i] - cols.tstart[i]
                    
                    obs_ll = dt ≈ 0 ? 0 : survprob(
                        0,
                        dt,
                        pars,
                        row_data,
                        data.model.totalhazards[statefrom_i],
                        hazards;
                        give_log = true)
                                        
                    if statefrom_i != stateto_i # if there is a transition, add log hazard
                        trans_idx = data.model.tmat[statefrom_i, stateto_i]
                        hazard = hazards[trans_idx]
                        hazard_pars = pars[hazard.hazname]
                        haz_value = eval_hazard(hazard, dt, hazard_pars, row_data)
                        obs_ll += NaNMath.log(haz_value)
                    end
                    
                    subj_ll += obs_ll * obs_weight

                else # panel data (obstype == 2, since we're in all_exact_or_panel branch)
                    # Forward algorithm for single observation:
                    # L = Σ_s P(X=s | X_0=statefrom) * P(Y | X=s)
                    #   = Σ_s TPM[statefrom, s] * emat[i, s]
                    #
                    # When emat[i, stateto] = 1 and emat[i, other] = 0 (default for exact observations),
                    # this reduces to TPM[statefrom, stateto].
                    # When EmissionMatrix provides soft evidence, this properly weights all states.
                    statefrom_i = cols.statefrom[i]
                    book_idx1 = cols.tpm_map_col1[i]
                    book_idx2 = cols.tpm_map_col2[i]
                    tpm = tpm_book[book_idx1][book_idx2]
                    
                    # Compute likelihood: Σ_s P(transition to s) * P(observation | s)
                    prob_sum = zero(T)
                    for s in 1:S
                        emission_prob = data.model.emat[i, s]
                        if emission_prob > 0
                            prob_sum += tpm[statefrom_i, s] * emission_prob
                        end
                    end
                    subj_ll += NaNMath.log(prob_sum) * obs_weight
                end
            end

        else
            # Forward algorithm for censored observations
            # NOTE: ObservationWeights are currently not supported for censored observations
            # (forward algorithm). Use SubjectWeights instead for weighted estimation.
            if has_obs_weights && any(data.model.ObservationWeights[subj_inds] .!= 1.0)
                @warn "ObservationWeights are not supported for censored observations (obstype > 2). Using unweighted likelihood for subject $subj."
            end
            
            # Pre-compute transient state transitions (avoid findall allocations in loop)
            tmat_cache = data.model.tmat
            transient_dests = [findall(tmat_cache[r,:] .!= 0) for r in 1:S]
            
            # initialize likelihood matrix (use cache for Float64)
            nobs_subj = length(subj_inds)
            if T === Float64
                # Use cached lmat_work, resize if needed
                if size(data.cache.lmat_work, 2) < nobs_subj + 1
                    data.cache.lmat_work = zeros(Float64, S, nobs_subj + 1)
                end
                lmat = @view data.cache.lmat_work[:, 1:(nobs_subj + 1)]
                fill!(lmat, zero(Float64))
            else
                lmat = zeros(T, S, nobs_subj + 1)
            end
            
            # Initialize forward probabilities at time 0 (before first observation)
            # For phase-type models, the expanded data already has the correct phase index
            # in statefrom. For exact transitions into a state, this is phase 1 (Coxian property).
            # For panel data, the emission matrix handles phase uncertainty.
            first_obs_idx = subj_inds[1]
            first_statefrom = cols.statefrom[first_obs_idx]
            
            @inbounds lmat[first_statefrom, 1] = 1.0

            # initialize counter for likelihood matrix
            ind = 1

            # update the vector l
            @inbounds for i in subj_inds

                # increment counter for likelihood matrix
                ind += 1
                
                obstype_i = cols.obstype[i]
                dt = cols.tstop[i] - cols.tstart[i]

                # compute q, the transition probability matrix
                if obstype_i != 1
                    # if panel data, simply grab q from tpm_book
                    book_idx1 = cols.tpm_map_col1[i]
                    book_idx2 = cols.tpm_map_col2[i]
                    copyto!(q, tpm_book[book_idx1][book_idx2])
                    
                else
                    # if exact data (obstype = 1), compute q by hand
                    # reset Q
                    fill!(q, -Inf)
                    
                    # Use @view for data row
                    row_data = @view data.model.data[i, :]
                    
                    if dt ≈ 0
                        # Instantaneous observation (dt=0): use raw hazard values
                        # This occurs in phase-type expanded data where the exact transition
                        # is recorded at a single point in time (dt=0 between sojourn and transition).
                        #
                        # For a density contribution, q[r,s] = h(r,s), NOT normalized.
                        # The likelihood contribution is: Σ_r P(in phase r) × h(r,s)
                        # which gives the density of transitioning to state s at this instant.
                        #
                        # Note: We do NOT normalize because we need the density (hazard),
                        # not the conditional probability. The forward algorithm accumulates
                        # the unnormalized likelihood which is later summed and log-transformed.
                        fill!(q, zero(eltype(q)))
                        for r in 1:S
                            if isa(data.model.totalhazards[r], _TotalHazardAbsorbing)
                                # Absorbing states don't contribute to transitions
                                # q[r, :] stays zero (no outgoing transitions)
                            else
                                dest_states = transient_dests[r]
                                # Compute raw hazards for each destination (no normalization)
                                for s in dest_states
                                    trans_idx = tmat_cache[r, s]
                                    hazard = hazards[trans_idx]
                                    hazard_pars = pars[hazard.hazname]
                                    haz_value = eval_hazard(hazard, dt, hazard_pars, row_data)
                                    q[r, s] = haz_value
                                end
                                # Diagonal is 0: transition definitely occurred at this instant
                                q[r, r] = 0.0
                            end
                        end
                    else
                        # Regular exact observation (dt > 0)
                        # compute q(r,s) 
                        for r in 1:S
                            if isa(data.model.totalhazards[r], _TotalHazardAbsorbing)
                                q[r,r] = 0.0 
                            else
                                # survival probability - use pre-computed destinations
                                dest_states = transient_dests[r]
                                log_surv = survprob(0, dt, pars, row_data,
                                    data.model.totalhazards[r], hazards; give_log = true)
                                for dest in dest_states
                                    q[r, dest] = log_surv
                                end
                                
                                # hazard
                                for s in dest_states
                                    trans_idx = tmat_cache[r, s]
                                    hazard = hazards[trans_idx]
                                    hazard_pars = pars[hazard.hazname]
                                    haz_value = eval_hazard(hazard, dt, hazard_pars, row_data)
                                    q[r, s] += NaNMath.log(haz_value)
                                end
                            end

                            # Ensure diagonal is a valid probability (H5_P1 fix: use eps(eltype(q)) for AD compatibility)
                            q[r,r] = maximum([1 - exp(logsumexp(q[r, Not(r)])), eps(eltype(q))])
                            q[r,Not(r)] = exp.(q[r, Not(r)])               
                        end
                    end
                end # end-compute q

                # Forward algorithm update: α_i(s) = e_{is} * Σ_r Q_{rs} * α_{i-1}(r)
                # where e_{is} = P(Y_i | X_i = s) is the emission probability.
                #
                # The emission matrix is constructed by build_emat():
                # - For exact observations: e[i, stateto] = 1, e[i, other] = 0
                # - For censored observations: e[i, :] from CensoringPatterns
                # - For user-supplied EmissionMatrix: uses provided soft evidence
                #
                # This unified approach enables soft evidence for any observation type.
                for s in 1:S
                    emission_prob = data.model.emat[i, s]
                    if emission_prob > 0
                        for r in 1:S
                            lmat[s, ind] += q[r, s] * lmat[r, ind - 1] * emission_prob
                        end
                    end
                end
            end

            # log likelihood
            subj_ll=NaNMath.log(sum(lmat[:,size(lmat, 2)]))
        end

        if return_ll_subj
            # weighted subject loglikelihood
            ll_subj[subj] = subj_ll * data.model.SubjectWeights[subj]
        else
            # weighted loglikelihood
            ll += subj_ll * data.model.SubjectWeights[subj]
        end        
    end

    if return_ll_subj
        ll_subj
    else
        neg ? -ll : ll
    end
end

