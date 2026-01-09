# =============================================================================
# Semi-Markov Panel Data Likelihood (Path-Based MCEM)
# =============================================================================
#
# Contents:
# - loglik_semi_markov: Main entry point for semi-Markov panel likelihood
# - loglik_semi_markov!: In-place version for MCEM
# - loglik_semi_markov_batched!: Batched version for efficiency
#
# These functions compute importance-weighted likelihoods over sampled paths
# for the Monte Carlo E-step of MCEM.
#
# Split from loglik.jl for maintainability (January 2026)
# =============================================================================

function loglik_semi_markov(parameters, data::SMPanelData; neg=true, use_sampling_weight=true, parallel=false)

    # Unflatten parameters to natural scale (AD-compatible)
    pars = unflatten_parameters(parameters, data.model)

    # Get hazards and model components
    hazards = data.model.hazards
    totalhazards = data.model.totalhazards
    tmat = data.model.tmat
    n_hazards = length(hazards)
    nsubj = length(data.paths)

    # Build subject covariate cache (reusable across all paths)
    subject_covars = build_subject_covar_cache(data.model)
    
    # Element type for AD compatibility
    T = eltype(parameters)
    
    # Get covariate names for each hazard (precomputed)
    covar_names_per_hazard = [
        hasfield(typeof(hazards[h]), :covar_names) ? 
            hazards[h].covar_names : 
            extract_covar_names(hazards[h].parnames)
        for h in 1:n_hazards
    ]
    
    # Check if any hazard uses time transform
    any_time_transform = any(h -> h.metadata.time_transform, hazards)
    
    # Count total paths for parallel decision
    n_total_paths = sum(length(data.paths[i]) for i in eachindex(data.paths))
    use_parallel = parallel && Threads.nthreads() > 1 && n_total_paths >= 50
    
    if use_parallel
        # Flat path-level parallelism for load balance
        # Build flat index mapping: path k → (subject i, path j)
        path_to_subj = Vector{Int}(undef, n_total_paths)
        path_to_j = Vector{Int}(undef, n_total_paths)
        k = 1
        for i in eachindex(data.paths)
            for j in eachindex(data.paths[i])
                path_to_subj[k] = i
                path_to_j[k] = j
                k += 1
            end
        end
        
        # Pre-allocate flat log-likelihood array
        ll_flat = Vector{T}(undef, n_total_paths)
        
        # Parallel over flat path index
        Threads.@threads :static for k in 1:n_total_paths
            i = path_to_subj[k]
            j = path_to_j[k]
            path = data.paths[i][j]
            subj_cache = subject_covars[path.subj]
            
            # Thread-local TimeTransformContext
            tt_context = if any_time_transform
                sample_df = isempty(subj_cache.covar_data) ? nothing : subj_cache.covar_data[1:1, :]
                maybe_time_transform_context(pars, sample_df, hazards)
            else
                nothing
            end
            
            ll_flat[k] = _compute_path_loglik_fused(
                path, pars, hazards, totalhazards, tmat,
                subj_cache, covar_names_per_hazard, tt_context, T
            )
        end
        
        # Weighted reduction: reassemble by subject and apply importance weights
        ll = zero(T)
        k = 1
        for i in eachindex(data.paths)
            subj_ll = zero(T)
            for j in eachindex(data.paths[i])
                subj_ll += ll_flat[k] * data.ImportanceWeights[i][j]
                k += 1
            end
            if use_sampling_weight
                ll += subj_ll * data.model.SubjectWeights[i]
            else
                ll += subj_ll
            end
        end
    else
        # Sequential path: simpler, AD-compatible
        tt_context = if any_time_transform && !isempty(data.paths) && !isempty(data.paths[1])
            sample_subj = subject_covars[data.paths[1][1].subj]
            sample_df = isempty(sample_subj.covar_data) ? nothing : sample_subj.covar_data[1:1, :]
            maybe_time_transform_context(pars, sample_df, hazards)
        else
            nothing
        end

        ll = zero(T)
        for i in eachindex(data.paths)
            lls = zero(T)
            for j in eachindex(data.paths[i])
                path = data.paths[i][j]
                subj_cache = subject_covars[path.subj]
                
                path_ll = _compute_path_loglik_fused(
                    path, pars, hazards, totalhazards, tmat,
                    subj_cache, covar_names_per_hazard, tt_context, T
                )
                lls += path_ll * data.ImportanceWeights[i][j]
            end
            if use_sampling_weight
                lls *= data.model.SubjectWeights[i]
            end
            ll += lls
        end
    end

    # Return the log-likelihood
    neg ? -ll : ll
end

"""
    loglik(parameters, data::SMPanelData)

Update log-likelihood for each individual and each path of panel data in a semi-Markov model.

This implementation uses the fused path-centric approach from `loglik_exact`, calling
`_compute_path_loglik_fused` directly to avoid DataFrame allocation overhead.
"""
function loglik_semi_markov!(parameters, logliks::Vector{}, data::SMPanelData)

    # Unflatten parameters to natural scale (AD-compatible)
    pars = unflatten_parameters(parameters, data.model)

    # snag the hazards and model components
    hazards = data.model.hazards
    totalhazards = data.model.totalhazards
    tmat = data.model.tmat
    n_hazards = length(hazards)

    # Build subject covariate cache (reusable across all paths)
    subject_covars = build_subject_covar_cache(data.model)
    
    # Element type for computation (Float64 for in-place version)
    T = Float64
    
    # Get covariate names for each hazard (precomputed)
    covar_names_per_hazard = [
        hasfield(typeof(hazards[h]), :covar_names) ? 
            hazards[h].covar_names : 
            extract_covar_names(hazards[h].parnames)
        for h in 1:n_hazards
    ]
    
    # Check if any hazard uses time transform and create context if needed
    any_time_transform = any(h -> h.metadata.time_transform, hazards)
    tt_context = if any_time_transform && !isempty(data.paths) && !isempty(data.paths[1])
        sample_subj = subject_covars[data.paths[1][1].subj]
        sample_df = isempty(sample_subj.covar_data) ? nothing : sample_subj.covar_data[1:1, :]
        maybe_time_transform_context(pars, sample_df, hazards)
    else
        nothing
    end

    # Compute log-likelihoods using fused path-centric approach
    for i in eachindex(data.paths)
        for j in eachindex(data.paths[i])
            path = data.paths[i][j]
            subj_cache = subject_covars[path.subj]
            
            logliks[i][j] = _compute_path_loglik_fused(
                path, pars, hazards, totalhazards, tmat,
                subj_cache, covar_names_per_hazard, tt_context, T
            )
        end
    end
end

"""
    loglik_semi_markov_batched!(parameters, logliks, data::SMPanelData)

Batched version of `loglik_semi_markov!` that computes all path log-likelihoods
using the batched hazard evaluation approach. This reduces redundant computations
when there are many paths per subject.

Arguments:
- `parameters`: Flat parameter vector
- `logliks`: Nested Vector{Vector{Float64}} to store log-likelihoods (modified in-place)
- `data`: SMPanelData containing the model and paths
"""
function loglik_semi_markov_batched!(parameters, logliks::Vector{Vector{Float64}}, data::SMPanelData)
    # Unflatten parameters to natural scale (AD-compatible)
    pars = unflatten_parameters(parameters, data.model)
    
    # Get hazards
    hazards = data.model.hazards
    n_hazards = length(hazards)
    
    # Flatten paths for batched processing
    # Build mapping from flat index to (subject_idx, path_idx)
    n_total_paths = sum(length(ps) for ps in data.paths)
    flat_paths = Vector{SamplePath}(undef, n_total_paths)
    path_mapping = Vector{Tuple{Int,Int}}(undef, n_total_paths)
    
    flat_idx = 0
    for i in eachindex(data.paths)
        for j in eachindex(data.paths[i])
            flat_idx += 1
            flat_paths[flat_idx] = data.paths[i][j]
            path_mapping[flat_idx] = (i, j)
        end
    end
    
    # Pre-cache all path DataFrames
    cached_paths = cache_path_data(flat_paths, data.model)
    
    # Create TimeTransformContext if needed
    sample_df = isempty(cached_paths) ? nothing : first(cached_paths).df
    tt_context = maybe_time_transform_context(pars, sample_df, hazards)
    
    # Initialize flat log-likelihoods
    ll_flat = zeros(n_total_paths)
    
    # Pre-process intervals for each hazard
    stacked_data = Vector{StackedHazardData}(undef, n_hazards)
    for h in 1:n_hazards
        stacked_data[h] = stack_intervals_for_hazard(
            h, cached_paths, hazards, data.model.totalhazards, data.model.tmat;
            pars=pars)
    end
    
    # Compute log-likelihoods in batched manner
    for h in 1:n_hazards
        sd = stacked_data[h]
        n_intervals = length(sd.lb)
        
        if n_intervals == 0
            continue
        end
        
        hazard = hazards[h]
        hazard_pars = pars[hazard.hazname]
        use_transform = hazard.metadata.time_transform
        
        for i in 1:n_intervals
            # Cumulative hazard contribution (survival component)
            cumhaz = eval_cumhaz(
                hazard, sd.lb[i], sd.ub[i], hazard_pars, sd.covars[i];
                apply_transform = use_transform,
                cache_context = tt_context,
                hazard_slot = h)
            
            ll_flat[sd.path_idx[i]] -= cumhaz
            
            # Transition hazard
            if sd.is_transition[i]
                haz_value = eval_hazard(
                    hazard, sd.transition_times[i], hazard_pars, sd.covars[i];
                    apply_transform = use_transform,
                    cache_context = tt_context,
                    hazard_slot = h)
                ll_flat[sd.path_idx[i]] += NaNMath.log(haz_value)
            end
        end
    end
    
    # Map back to nested structure
    for k in 1:n_total_paths
        i, j = path_mapping[k]
        logliks[i][j] = ll_flat[k]
    end
end


# =============================================================================
# Penalized Semi-Markov Likelihood (for MCEM with spline penalties)
# =============================================================================

"""
    loglik_semi_markov_penalized(parameters, data::SMPanelData, penalty_config::PenaltyConfig;
                                  neg::Bool=true, use_sampling_weight::Bool=true)

Compute penalized log-likelihood for semi-Markov panel data.

This wraps `loglik_semi_markov` and adds the spline penalty term:
    ℓₚ(β; λ) = ℓ(β) - (1/2) Σⱼ λⱼ βⱼᵀ Sⱼ βⱼ

# Arguments
- `parameters`: Flat parameter vector (natural scale for baseline as of v0.3.0)
- `data::SMPanelData`: Semi-Markov panel data with paths and importance weights
- `penalty_config::PenaltyConfig`: Penalty specification with matrices and lambdas
- `neg::Bool=true`: Return negative log-likelihood for minimization
- `use_sampling_weight::Bool=true`: Apply subject sampling weights

# Returns
Scalar (penalized) log-likelihood

# Notes
- As of v0.3.0, all parameters are on natural scale. Box constraints enforce positivity.
- Used in MCEM M-step when fitting penalized spline models
- For unpenalized likelihood, use `loglik_semi_markov` directly
"""
function loglik_semi_markov_penalized(parameters, data::SMPanelData, penalty_config::PenaltyConfig;
                                       neg::Bool=true, use_sampling_weight::Bool=true)
    # Compute base log-likelihood (as negative for consistency)
    nll_base = loglik_semi_markov(parameters, data; neg=true, use_sampling_weight=use_sampling_weight)
    
    # If no penalties, return base likelihood
    has_penalties(penalty_config) || return neg ? nll_base : -nll_base
    
    # Extract natural-scale baseline coefficients for penalty
    pars_natural = unflatten_parameters(parameters, data.model)
    
    # Build flat natural-scale vector for penalty computation
    T = eltype(parameters)
    n_params = length(parameters)
    beta_natural = Vector{T}(undef, n_params)
    
    offset = 0
    for (hazname, idx) in sort(collect(data.model.hazkeys), by = x -> x[2])
        hazard = data.model.hazards[idx]
        n_total = hazard.npar_total
        n_baseline = hazard.npar_baseline
        
        # Natural scale baseline (exp-transformed)
        haz_pars = pars_natural[hazname]
        baseline_vals = values(haz_pars.baseline)
        for i in 1:n_baseline
            beta_natural[offset + i] = baseline_vals[i]
        end
        
        # Covariate coefficients unchanged
        if n_total > n_baseline
            covar_vals = values(haz_pars.covariates)
            for i in 1:(n_total - n_baseline)
                beta_natural[offset + n_baseline + i] = covar_vals[i]
            end
        end
        
        offset += n_total
    end
    
    # Compute penalty
    penalty = compute_penalty(beta_natural, penalty_config)
    
    # Return penalized negative log-likelihood
    nll_penalized = nll_base + penalty
    return neg ? nll_penalized : -nll_penalized
end

"""
    loglik(parameters, data::SMPanelData, penalty_config::PenaltyConfig; kwargs...)

Dispatch method for penalized semi-Markov likelihood.

This enables a consistent interface for both penalized and unpenalized fitting:
- `loglik(params, data)` → unpenalized
- `loglik(params, data, penalty_config)` → penalized
"""
function loglik(parameters, data::SMPanelData, penalty_config::PenaltyConfig; 
                neg::Bool=true, use_sampling_weight::Bool=true)
    loglik_semi_markov_penalized(parameters, data, penalty_config;
                                  neg=neg, use_sampling_weight=use_sampling_weight)
end
