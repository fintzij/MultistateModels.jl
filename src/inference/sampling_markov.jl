# ============================================================================
# Path Sampling Infrastructure - Markov FFBS and ECCTMC
# ============================================================================
#
# This file contains the core sampling algorithms for Markov surrogate-based
# importance sampling in MCEM:
#
# - DrawSamplePaths!: Main entry point for MCEM path sampling with ESS targeting
# - ForwardFiltering!/BackwardSampling!: Forward-filtering backward-sampling (FFBS)
# - sample_ecctmc/sample_ecctmc!: Endpoint-conditioned CTMC sampling (Hobolth & Stone)
# - draw_samplepath: Sample paths from Markov surrogate for a single subject
# - draw_paths: High-level API for path sampling with importance weights
# - ComputeImportanceWeightsESS!: Pareto-smoothed importance sampling (PSIS)
#
# Dependencies:
# - sampling_core.jl: PathWorkspace and thread-local storage (must be loaded first)
#
# Used by:
# - sampling_phasetype.jl: Phase-type sampling uses ForwardFiltering! from this file
# - mcem.jl: MCEM algorithm calls DrawSamplePaths! for E-step
# ============================================================================

# ============================================================================
# Main DrawSamplePaths! functions - Infrastructure-based API (Phase 3 refactor)
# ============================================================================

"""
    DrawSamplePaths!(model, infra::MCEMInfrastructure, containers; kwargs...)

Draw sample paths using infrastructure-based dispatch. This is the new unified API
that replaces the legacy kwargs-based signature with 12+ phase-type arguments.

# Arguments
- `model::MultistateProcess`: The target model
- `infra::MCEMInfrastructure`: Pre-built infrastructure (Markov or PhaseType)
- `containers`: NamedTuple with sampling containers:
  - `samplepaths`, `loglik_surrog`, `loglik_target_prop`, `loglik_target_cur`
  - `_logImportanceWeights`, `ImportanceWeights`, `ess_cur`, `psis_pareto_k`

# Keyword Arguments  
- `ess_target::Int`: Target effective sample size
- `max_sampling_effort::Int`: Maximum multiplier for paths vs ESS
- `npaths_additional::Int`: Paths to add per sampling round
- `params_cur::Vector{Float64}`: Current parameter values for target likelihood

See also: [`MCEMInfrastructure`](@ref), [`build_mcem_infrastructure`](@ref)
"""
function DrawSamplePaths!(model::MultistateProcess, infra::MCEMInfrastructure, containers;
    ess_target, max_sampling_effort, npaths_additional, params_cur)
    
    # Unflatten parameters for target likelihood evaluation
    pars = unflatten_parameters(params_cur, model)

    for i in 1:infra.nsubj
        DrawSamplePaths!(i, model, infra, containers;
            ess_target = ess_target,
            max_sampling_effort = max_sampling_effort,
            npaths_additional = npaths_additional,
            params_cur = params_cur)
    end
end

"""
    DrawSamplePaths!(i::Int, model, infra::MCEMInfrastructure, containers; kwargs...)

Draw sample paths for subject i using infrastructure-based dispatch.
Dispatches to surrogate-specific sampling based on infra type parameter.
"""
function DrawSamplePaths!(i::Int, model::MultistateProcess, infra::MCEMInfrastructure, containers;
    ess_target, max_sampling_effort, npaths_additional, params_cur)
    
    # Extract containers
    samplepaths = containers.samplepaths
    loglik_surrog = containers.loglik_surrog
    loglik_target_prop = containers.loglik_target_prop
    loglik_target_cur = containers.loglik_target_cur
    _logImportanceWeights = containers._logImportanceWeights
    ImportanceWeights = containers.ImportanceWeights
    ess_cur = containers.ess_cur
    psis_pareto_k = containers.psis_pareto_k
    
    n_path_max = max_sampling_effort * ess_target
    keep_sampling = ess_cur[i] < ess_target
    
    # Subject data view (from original model, not infra - needed for target loglik)
    subj_inds = model.subjectindices[i]
    subj_dat = view(model.data, subj_inds, :)
    
    # For Markov surrogate: compute FFBS matrices if needed for censored observations
    if infra isa MCEMInfrastructure{MarkovSurrogate}
        _prepare_ffbs_markov!(i, model, infra, subj_dat)
    end
    
    while keep_sampling
        npaths = length(samplepaths[i])
        n_add = npaths == 0 ? maximum([50, Int(ceil(ess_target))]) : npaths_additional
        
        # Augment containers
        append!(samplepaths[i], Vector{SamplePath}(undef, n_add))
        append!(loglik_surrog[i], zeros(n_add))
        append!(loglik_target_prop[i], zeros(n_add))
        append!(loglik_target_cur[i], zeros(n_add))
        append!(_logImportanceWeights[i], zeros(n_add))
        append!(ImportanceWeights[i], zeros(n_add))
        
        # Sample new paths
        for j in npaths .+ (1:n_add)
            _sample_path_with_infra!(j, i, model, infra, containers, params_cur, subj_dat)
        end
        
        # Update ESS and importance weights
        _update_ess_and_weights!(i, samplepaths, loglik_surrog, loglik_target_cur,
                                  loglik_target_prop, _logImportanceWeights, ImportanceWeights, 
                                  ess_cur, psis_pareto_k, ess_target)
        
        # Check stopping criteria
        keep_sampling = (ess_cur[i] < ess_target) && (length(samplepaths[i]) <= n_path_max)
        
        if length(samplepaths[i]) > n_path_max
            @warn "More than $n_path_max sample paths required for individual $i."
        end
    end
end

"""
    _prepare_ffbs_markov!(i, model, infra::MCEMInfrastructure{MarkovSurrogate}, subj_dat)

Prepare forward-backward matrices for Markov surrogate if subject has censored observations.
"""
function _prepare_ffbs_markov!(i::Int, model::MultistateProcess, 
                                infra::MCEMInfrastructure{MarkovSurrogate}, subj_dat)
    if !isnothing(infra.fbmats) && any(subj_dat.obstype .∉ Ref([1,2]))
        subj_inds = model.subjectindices[i]
        subj_tpm_map = view(infra.books[2], subj_inds, :)
        subj_emat = view(model.emat, subj_inds, :)
        ForwardFiltering!(infra.fbmats[i], subj_dat, infra.tpm_book, subj_tpm_map, subj_emat;
                         hazmat_book=infra.hazmat_book)
    end
end

"""
    _sample_path_with_infra!(j, i, model, infra::MCEMInfrastructure{MarkovSurrogate}, ...)

Sample a single path using Markov surrogate infrastructure.
"""
function _sample_path_with_infra!(j::Int, i::Int, model::MultistateProcess,
    infra::MCEMInfrastructure{MarkovSurrogate}, containers, params_cur, subj_dat)
    
    samplepaths = containers.samplepaths
    loglik_surrog = containers.loglik_surrog
    loglik_target_cur = containers.loglik_target_cur
    _logImportanceWeights = containers._logImportanceWeights
    
    # Draw path from Markov surrogate
    samplepaths[i][j] = draw_samplepath(i, model, infra.tpm_book, infra.hazmat_book,
                                         infra.books[2], infra.fbmats, infra.absorbingstates)
    
    # Surrogate log-likelihood (using dispatch)
    loglik_surrog[i][j] = compute_surrogate_path_loglik(samplepaths[i][j], i, model, infra)
    
    # Target log-likelihood
    target_pars = unflatten_parameters(params_cur, model)
    loglik_target_cur[i][j] = loglik(target_pars, samplepaths[i][j], model.hazards, model)
    
    # Unnormalized log importance weight
    _logImportanceWeights[i][j] = loglik_target_cur[i][j] - loglik_surrog[i][j]
end

"""
    _sample_path_with_infra!(j, i, model, infra::MCEMInfrastructure{PhaseTypeSurrogate}, ...)

Sample a single path using PhaseType surrogate infrastructure.
Samples in expanded space, collapses to original states, computes marginal likelihood.
"""
function _sample_path_with_infra!(j::Int, i::Int, model::MultistateProcess,
    infra::MCEMInfrastructure{PhaseTypeSurrogate}, containers, params_cur, subj_dat)
    
    samplepaths = containers.samplepaths
    loglik_surrog = containers.loglik_surrog
    loglik_target_cur = containers.loglik_target_cur
    _logImportanceWeights = containers._logImportanceWeights
    
    # Get tpm_map for this subject (expanded if data was expanded)
    ph_tpm_map = isnothing(infra.expanded_tpm_map) ? infra.books[2] : infra.expanded_tpm_map
    
    # Get subject's tpm_map on original data (for covariate lookup during TVC)
    subj_inds = model.subjectindices[i]
    subj_tpm_map = view(infra.books[2], subj_inds, :)
    
    # Sample with retry for -Inf likelihoods
    max_retries = 10
    retry_count = 0
    valid_path = false
    
    while !valid_path && retry_count < max_retries
        # Sample in expanded phase space
        path_result = draw_samplepath_phasetype(i, model, infra.tpm_book, infra.hazmat_book,
                                                 ph_tpm_map, infra.fbmats, infra.emat,
                                                 infra.surrogate, infra.absorbingstates;
                                                 expanded_data = infra.original_row_map === nothing ? nothing : infra.data,
                                                 expanded_subjectindices = infra.original_row_map === nothing ? nothing : infra.subjectindices,
                                                 original_row_map = infra.original_row_map)
        
        # Store collapsed path for target likelihood evaluation
        samplepaths[i][j] = path_result.collapsed
        
        # Compute surrogate log-likelihood using dispatch (marginal over phases)
        loglik_surrog[i][j] = compute_surrogate_path_loglik(
            samplepaths[i][j], i, model, infra;
            expanded_path = path_result.expanded,
            subj_data = subj_dat,
            subj_tpm_map = subj_tpm_map
        )
        
        if isfinite(loglik_surrog[i][j])
            valid_path = true
        else
            retry_count += 1
            if retry_count == max_retries
                @warn "PhaseType proposal: -Inf surrogate likelihood for subject $i path $j after $max_retries retries; using path anyway" maxlog=5
                # Fall back to Markov proposal likelihood (approximate)
                markov_pars = get_hazard_params(infra.surrogate.parameters, infra.surrogate.hazards)
                loglik_surrog[i][j] = loglik(markov_pars, samplepaths[i][j], infra.surrogate.hazards, model)
                valid_path = true
            end
        end
    end
    
    # Target log-likelihood (on collapsed path)
    target_pars = unflatten_parameters(params_cur, model)
    loglik_target_cur[i][j] = loglik(target_pars, samplepaths[i][j], model.hazards, model)
    
    # Unnormalized log importance weight
    _logImportanceWeights[i][j] = loglik_target_cur[i][j] - loglik_surrog[i][j]
end

"""
    _update_ess_and_weights!(i, ...)

Update ESS and importance weights for subject i after sampling.
Handles degenerate cases and applies PSIS smoothing.
"""
function _update_ess_and_weights!(i::Int, samplepaths, loglik_surrog, loglik_target_cur,
                                   loglik_target_prop, _logImportanceWeights, ImportanceWeights, 
                                   ess_cur, psis_pareto_k, ess_target)
    
    # Handle degenerate case: all paths have same surrogate likelihood
    if allequal(loglik_surrog[i])
        samplepaths[i] = [first(samplepaths[i])]
        loglik_target_cur[i] = [first(loglik_target_cur[i])]
        loglik_target_prop[i] = [first(loglik_target_prop[i])]  # Also truncate proposed likelihoods
        loglik_surrog[i] = [first(loglik_surrog[i])]
        _logImportanceWeights[i] = [first(_logImportanceWeights[i])]
        ImportanceWeights[i] = [1.0]
        ess_cur[i] = ess_target
        return
    end
    
    # Check for degenerate weights
    weights_range = maximum(_logImportanceWeights[i]) - minimum(_logImportanceWeights[i])
    weights_degenerate = allequal(_logImportanceWeights[i]) || weights_range < 1e-10
    
    if weights_degenerate
        fill!(ImportanceWeights[i], 1/length(ImportanceWeights[i]))
        ess_cur[i] = length(ImportanceWeights[i])
        psis_pareto_k[i] = 0.0
    else
        # Apply PSIS
        psiw = try
            ParetoSmooth.psis(reshape(copy(_logImportanceWeights[i]), 1, length(_logImportanceWeights[i]), 1); source = "other")
        catch e
            if e isa ArgumentError && occursin("all tail values are the same", string(e))
                @warn "PSIS failed for subject $i (degenerate tail); using uniform weights" maxlog=5
                nothing
            else
                rethrow(e)
            end
        end
        
        if isnothing(psiw)
            fill!(ImportanceWeights[i], 1/length(ImportanceWeights[i]))
            ess_cur[i] = length(ImportanceWeights[i])
            psis_pareto_k[i] = Inf
        else
            copyto!(ImportanceWeights[i], psiw.weights)
            ess_cur[i] = psiw.ess[1]
            psis_pareto_k[i] = psiw.pareto_k[1]
        end
        
        # Handle NaN/Inf ESS from PSIS
        if isnan(ess_cur[i]) || isinf(ess_cur[i])
            copyto!(ImportanceWeights[i], normalize(exp.(_logImportanceWeights[i] .- maximum(_logImportanceWeights[i])), 1))
            ess_cur[i] = 1 / sum(ImportanceWeights[i] .^ 2)
        end
    end
end

# ============================================================================
# Legacy DrawSamplePaths! functions (kwargs-based API)
# ============================================================================
# These remain for backward compatibility during Phase 4 transition.
# Will be removed once fit_mcem.jl is fully updated to use infrastructure.

"""
    DrawSamplePaths(model; ...)

Draw additional sample paths until sufficient ESS or until the maximum number of paths is reached.

Supports both Markov and phase-type surrogate proposals. When phase-type infrastructure
is provided (phasetype_surrogate, tpm_book_ph, etc.), uses phase-type FFBS for sampling.
"""
function DrawSamplePaths!(model::MultistateProcess; ess_target, ess_cur, max_sampling_effort, samplepaths, loglik_surrog, loglik_target_prop, loglik_target_cur, _logImportanceWeights, ImportanceWeights,tpm_book_surrogate, hazmat_book_surrogate, books, npaths_additional, params_cur, surrogate, psis_pareto_k, fbmats, absorbingstates,
    # Phase-type proposal infrastructure (optional)
    phasetype_surrogate=nothing, tpm_book_ph=nothing, hazmat_book_ph=nothing, fbmats_ph=nothing, emat_ph=nothing,
    # Expanded data infrastructure for exact observations (optional)
    expanded_ph_data=nothing, expanded_ph_subjectindices=nothing, expanded_ph_tpm_map=nothing, ph_original_row_map=nothing,
    # Cached Schur decompositions for efficient TPM computation
    schur_cache_ph=nothing)

    # Determine if using phase-type proposals
    use_phasetype = !isnothing(phasetype_surrogate)

    # make sure spline parameters are assigned correctly
    # unflatten parameters to natural scale (AD-compatible)
    pars = unflatten_parameters(params_cur, model)

    for i in eachindex(model.subjectindices)
        DrawSamplePaths!(i, model; 
            ess_target = ess_target,
            ess_cur = ess_cur, 
            max_sampling_effort = max_sampling_effort,
            samplepaths = samplepaths, 
            loglik_surrog = loglik_surrog, 
            loglik_target_prop = loglik_target_prop, 
            loglik_target_cur = loglik_target_cur, 
            _logImportanceWeights = _logImportanceWeights, 
            ImportanceWeights = ImportanceWeights, 
            tpm_book_surrogate = tpm_book_surrogate, 
            hazmat_book_surrogate = hazmat_book_surrogate, 
            books = books, 
            npaths_additional = npaths_additional, 
            params_cur = params_cur, 
            surrogate = surrogate, 
            psis_pareto_k = psis_pareto_k,
            fbmats = fbmats,
            absorbingstates = absorbingstates,
            # Phase-type infrastructure
            phasetype_surrogate = phasetype_surrogate,
            tpm_book_ph = tpm_book_ph,
            hazmat_book_ph = hazmat_book_ph,
            fbmats_ph = fbmats_ph,
            emat_ph = emat_ph,
            # Expanded data infrastructure
            expanded_ph_data = expanded_ph_data,
            expanded_ph_subjectindices = expanded_ph_subjectindices,
            expanded_ph_tpm_map = expanded_ph_tpm_map,
            ph_original_row_map = ph_original_row_map,
            # Cached Schur decompositions
            schur_cache_ph = schur_cache_ph)
    end
end

"""
    DrawSamplePaths(i, model; ...)

Draw additional sample paths for subject i until sufficient ESS or max paths reached.

Dispatches to either Markov or phase-type sampling based on whether phase-type
infrastructure is provided.
"""
function DrawSamplePaths!(i, model::MultistateProcess; ess_target, ess_cur, max_sampling_effort, samplepaths, loglik_surrog, loglik_target_prop, loglik_target_cur, _logImportanceWeights, ImportanceWeights, tpm_book_surrogate, hazmat_book_surrogate, books, npaths_additional, params_cur, surrogate, psis_pareto_k, fbmats, absorbingstates,
    # Phase-type proposal infrastructure (optional)
    phasetype_surrogate=nothing, tpm_book_ph=nothing, hazmat_book_ph=nothing, fbmats_ph=nothing, emat_ph=nothing,
    # Expanded data infrastructure for exact observations (optional)
    expanded_ph_data=nothing, expanded_ph_subjectindices=nothing, expanded_ph_tpm_map=nothing, ph_original_row_map=nothing,
    # Cached Schur decompositions for efficient TPM computation
    schur_cache_ph=nothing)

    # Determine if using phase-type proposals
    use_phasetype = !isnothing(phasetype_surrogate)

    n_path_max = max_sampling_effort*ess_target

    # sample new paths if the current ess is less than the target
    keep_sampling = ess_cur[i] < ess_target

    # subject data
    subj_inds = model.subjectindices[i]
    subj_dat  = view(model.data, subj_inds, :)

    # compute fbmats here (for Markov FFBS, not phase-type)
    if !use_phasetype && any(subj_dat.obstype .∉ Ref([1,2]))
        # subject data
        subj_tpm_map = view(books[2], subj_inds, :)
        subj_emat    = view(model.emat, subj_inds, :)
        ForwardFiltering!(fbmats[i], subj_dat, tpm_book_surrogate, subj_tpm_map, subj_emat;
                         hazmat_book=hazmat_book_surrogate)
    end

    # sample
    while keep_sampling
        # make sure there are at least 50 paths in order to fit pareto
        npaths = length(samplepaths[i])
        n_add  = npaths == 0 ? maximum([50, Int(ceil(ess_target))]) : npaths_additional

        # augment the number of paths
        append!(samplepaths[i], Vector{SamplePath}(undef, n_add))
        append!(loglik_surrog[i], zeros(n_add))
        append!(loglik_target_prop[i], zeros(n_add))
        append!(loglik_target_cur[i], zeros(n_add))
        append!(_logImportanceWeights[i], zeros(n_add))
        append!(ImportanceWeights[i], zeros(n_add))

        # sample new paths and compute log likelihoods
        for j in npaths.+(1:n_add)
            if use_phasetype
                # Phase-type proposal: sample in expanded space, collapse to observed
                # Use expanded data for tpm_map and FFBS when available (for exact observations)
                ph_tpm_map = isnothing(expanded_ph_tpm_map) ? books[2] : expanded_ph_tpm_map
                
                # Get subject's tpm_map (for covariate index lookup during TVC interpolation)
                subj_tpm_map = view(books[2], subj_inds, :)
                
                # Sample path with retry for -Inf likelihoods
                # This can happen in edge cases where the sampled phase configuration
                # has zero probability under the Q matrix (e.g., immediate transitions
                # from phases with zero absorption rate)
                max_retries = 10
                retry_count = 0
                valid_path = false
                
                while !valid_path && retry_count < max_retries
                    path_result = draw_samplepath_phasetype(i, model, tpm_book_ph, hazmat_book_ph, 
                                                             ph_tpm_map, fbmats_ph, emat_ph, 
                                                             phasetype_surrogate, absorbingstates;
                                                             expanded_data = expanded_ph_data,
                                                             expanded_subjectindices = expanded_ph_subjectindices,
                                                             original_row_map = ph_original_row_map)
                    
                    # Store collapsed path for target likelihood evaluation
                    samplepaths[i][j] = path_result.collapsed
                    
                    # Surrogate log-likelihood: MARGINAL over phase sequences
                    # 
                    # For importance sampling with phase-type expansion:
                    #   - Numerator: p(Z_collapsed | θ_target) — density of collapsed path
                    #   - Denominator: q(Z_collapsed | θ_surrogate) — marginal over phase paths
                    #
                    # The denominator must be the MARGINAL over all expanded paths that
                    # collapse to the same macro-state sequence:
                    #   q(Z_collapsed | θ') = Σ_{Z_e: collapse(Z_e) = Z_collapsed} q(Z_e | θ')
                    #
                    # This is computed via forward algorithm on the properly censored data:
                    # 1. convert_expanded_path_to_censored_data creates intervals at macro-state
                    #    transition times, with emission matrices allowing any phase of each macro-state
                    # 2. compute_forward_loglik runs forward algorithm, marginalizing over phases
                    #
                    # Using the CTMC path density of the expanded path would be WRONG because
                    # it would over-penalize specific phase sequences that happen to have low
                    # probability, even if the collapsed path has high probability.
                    
                    has_tvc = length(hazmat_book_ph) > 1
                    
                    # Convert expanded path to censored data format for forward algorithm
                    if has_tvc
                        censored_data, emat_path, tpm_map_path, tpm_book_path, hazmat_book_path = 
                            convert_expanded_path_to_censored_data(
                                path_result.expanded, phasetype_surrogate;
                                original_subj_data = subj_dat,
                                hazmat_book = hazmat_book_ph,
                                schur_cache_book = schur_cache_ph,
                                subj_tpm_map = subj_tpm_map
                            )
                    else
                        censored_data, emat_path, tpm_map_path, tpm_book_path, hazmat_book_path = 
                            convert_expanded_path_to_censored_data(
                                path_result.expanded, phasetype_surrogate;
                                hazmat = phasetype_surrogate.expanded_Q,
                                schur_cache = isnothing(schur_cache_ph) ? nothing : schur_cache_ph[1]
                            )
                    end
                    
                    # Compute surrogate log-likelihood via forward algorithm
                    loglik_surrog[i][j] = compute_forward_loglik(
                        censored_data, emat_path, tpm_map_path, tpm_book_path, 
                        hazmat_book_path, phasetype_surrogate.n_expanded_states
                    )
                    
                    if isfinite(loglik_surrog[i][j])
                        valid_path = true
                    else
                        retry_count += 1
                        if retry_count == max_retries
                            @warn "PhaseType proposal: -Inf surrogate likelihood for subject $i path $j after $max_retries retries; using path anyway" maxlog=5
                            # Fall back to Markov proposal likelihood to avoid NaN
                            # This is approximate but prevents algorithm failure
                            surrogate_pars = get_hazard_params(surrogate.parameters, surrogate.hazards)
                            loglik_surrog[i][j] = loglik(surrogate_pars, samplepaths[i][j], surrogate.hazards, model)
                            valid_path = true
                        end
                    end
                end
            else
                # Markov proposal: standard sampling
                samplepaths[i][j] = draw_samplepath(i, model, tpm_book_surrogate, hazmat_book_surrogate, 
                                                    books[2], fbmats, absorbingstates)
                
                # Surrogate log-likelihood under Markov proposal (family-aware)
                surrogate_pars = get_hazard_params(surrogate.parameters, surrogate.hazards)
                loglik_surrog[i][j] = loglik(surrogate_pars, samplepaths[i][j], surrogate.hazards, model)
            end
            
            # target log-likelihood (same for both proposal types)
            # unflatten_parameters returns natural-scale params
            target_pars = unflatten_parameters(params_cur, model)
            loglik_target_cur[i][j] = loglik(target_pars, samplepaths[i][j], model.hazards, model) 
            
            # unnormalized log importance weight
            _logImportanceWeights[i][j] = loglik_target_cur[i][j] - loglik_surrog[i][j]
        end

        # no need to keep all paths
        if allequal(loglik_surrog[i])
            samplepaths[i]        = [first(samplepaths[i]),]
            loglik_target_cur[i]  = [first(loglik_target_cur[i]),]
            loglik_target_prop[i] = [first(loglik_target_prop[i]),]
            loglik_surrog[i]      = [first(loglik_surrog[i]),]
            ess_cur[i]            = ess_target
            ImportanceWeights[i]  = [1.0,]
            _logImportanceWeights[i] = [first(_logImportanceWeights[i]),]

        else
            # Handle the case when all importance weights are equal (target ≡ surrogate)
            # This can happen when:
            # 1. All weights are exactly zero (target and surrogate match perfectly)
            # 2. All weights are the same non-zero constant (due to floating point agreement)
            # 3. All weights in the tail (sorted top) are identical (PSIS needs variability)
            # In all cases, PSIS will fail because it needs variability in weights
            
            # Check for degenerate weights - either exactly equal or nearly so
            weights_range = maximum(_logImportanceWeights[i]) - minimum(_logImportanceWeights[i])
            weights_degenerate = allequal(_logImportanceWeights[i]) || weights_range < 1e-10
            
            if weights_degenerate
                fill!(ImportanceWeights[i], 1/length(ImportanceWeights[i]))
                ess_cur[i] = length(ImportanceWeights[i])
                psis_pareto_k[i] = 0.0
            else
                # Try PSIS, but catch errors from degenerate tail distributions
                psiw = try
                    ParetoSmooth.psis(reshape(copy(_logImportanceWeights[i]), 1, length(_logImportanceWeights[i]), 1); source = "other")
                catch e
                    if e isa ArgumentError && occursin("all tail values are the same", string(e))
                        # Fall back to uniform weights when tail is degenerate
                        @warn "PSIS failed for subject $i (degenerate tail); using uniform weights" maxlog=5
                        nothing
                    else
                        rethrow(e)
                    end
                end
                
                if isnothing(psiw)
                    # PSIS failed, use uniform weights
                    fill!(ImportanceWeights[i], 1/length(ImportanceWeights[i]))
                    ess_cur[i] = length(ImportanceWeights[i])  
                    psis_pareto_k[i] = Inf  # Mark as unreliable
                else
                    # save normalized importance weights and ess
                    copyto!(ImportanceWeights[i], psiw.weights)
                    ess_cur[i] = psiw.ess[1]
                    psis_pareto_k[i] = psiw.pareto_k[1]
                end
                    
                # Handle NaN ESS from PSIS (can happen with high Pareto-k)
                # Fall back to simple ESS calculation
                if isnan(ess_cur[i]) || isinf(ess_cur[i])
                    # exponentiate and normalize the unnormalized log weights
                    copyto!(ImportanceWeights[i], normalize(exp.(_logImportanceWeights[i] .- maximum(_logImportanceWeights[i])), 1))
                    ess_cur[i] = 1 / sum(ImportanceWeights[i] .^ 2)
                    # Keep the high pareto_k to indicate unreliable weights
                end
            end
        end
        
        # check whether to stop
        if ess_cur[i] >= ess_target
            keep_sampling = false
        end
        
        if length(samplepaths[i]) > n_path_max
            keep_sampling = false
            @warn "More than $n_path_max sample paths are required to obtain ess>$ess_target for individual $i."
        end
    end
end

"""
    draw_paths(model::MultistateProcess; min_ess=100, npaths=nothing, paretosmooth=true, return_logliks=false)

Draw sample paths conditional on observed data using importance sampling.

This function samples latent paths from a Markov surrogate proposal distribution
and computes importance weights for the target model. Supports both adaptive
sampling (until ESS target is met) and fixed-count sampling.

# Sampling Mode
- If `npaths` is `nothing` (default): Adaptive sampling until `min_ess` is achieved
- If `npaths` is an integer: Draw exactly `npaths` paths per subject

# Arguments
- `model::MultistateProcess`: Fitted or unfitted multistate model
- `min_ess::Int`: Target effective sample size for adaptive mode (default: 100)
- `npaths::Union{Nothing, Int}`: Fixed number of paths per subject (overrides adaptive)
- `paretosmooth::Bool`: Apply Pareto smoothing to importance weights (default: true)
- `return_logliks::Bool`: Include log-likelihoods and ESS in output (default: false)

# Returns
NamedTuple with:
- `samplepaths`: Vector of SamplePath vectors, one per subject
- `ImportanceWeightsNormalized`: Normalized importance weights per subject
- If `return_logliks=true`: Also includes `loglik_target`, `loglik_surrog`, `subj_ess`, `ImportanceWeights`
- If exact data (all obstype==1) on fitted model: Returns `(loglik=..., subj_lml=...)` shortcut

# Example
```julia
# Adaptive sampling until ESS >= 100 (default)
result = draw_paths(fitted_model)

# Fixed number of paths
result = draw_paths(fitted_model; npaths=500)

# Get additional diagnostics
result = draw_paths(fitted_model; npaths=200, return_logliks=true)
paths, weights = result.samplepaths, result.ImportanceWeightsNormalized
```

See also: [`fit`](@ref), [`simulate`](@ref)
"""
function draw_paths(model::MultistateProcess; 
                    min_ess::Int = 100, 
                    npaths::Union{Nothing, Int} = nothing,
                    paretosmooth::Bool = true, 
                    return_logliks::Bool = false)

    # Exact data shortcut for fitted models
    if model isa MultistateModelFitted && all(model.data.obstype .== 1)
        return (loglik = model.loglik.loglik,
                subj_lml = model.loglik.subj_lml)
    end

    # Determine sampling mode
    adaptive_mode = isnothing(npaths)

    # number of subjects
    nsubj = length(model.subjectindices)

    # is the model semi-Markov (needs importance sampling)?
    is_semimarkov = !all(isa.(model.hazards, _MarkovHazard))

    # Get or fit surrogate for semi-Markov models
    surrogate = _get_or_fit_surrogate(model, is_semimarkov)

    # get natural-scale parameters for hazard evaluation (family-aware)
    params_target = get_hazard_params(model.parameters, model.hazards)
    params_surrog = is_semimarkov ? get_hazard_params(surrogate.parameters, surrogate.hazards) : params_target

    # get hazards
    hazards_target = model.hazards
    hazards_surrog = is_semimarkov ? surrogate.hazards : model.hazards

    # Set up sampling infrastructure
    books, tpm_book, hazmat_book, cache = _setup_tpm_infrastructure(model, params_surrog, hazards_surrog)

    # Set up result containers
    initial_capacity = adaptive_mode ? ceil(Int64, 4 * min_ess) : npaths
    samplepaths, loglik_target, loglik_surrog, ImportanceWeights = 
        _allocate_path_containers(nsubj, initial_capacity, adaptive_mode)

    # ESS and diagnostic tracking
    subj_ess = Vector{Float64}(undef, nsubj)
    subj_pareto_k = zeros(nsubj)
    
    # Forward-backward matrices for panel data
    fbmats = build_fbmats(model)
    
    # Absorbing states
    absorbingstates = findall(map(x -> all(x .== 0), eachrow(model.tmat)))

    # Sample paths for each subject
    for i in eachindex(model.subjectindices) 
        _draw_paths_for_subject!(
            i, model, adaptive_mode, min_ess, npaths,
            samplepaths, loglik_target, loglik_surrog, ImportanceWeights, subj_ess, subj_pareto_k,
            params_target, hazards_target, params_surrog, hazards_surrog,
            tpm_book, hazmat_book, books, fbmats, absorbingstates,
            is_semimarkov, paretosmooth
        )
    end

    # Normalize importance weights
    ImportanceWeightsNormalized = normalize.(ImportanceWeights, 1)

    if return_logliks
        return (; samplepaths, loglik_target, subj_ess, loglik_surrog, ImportanceWeightsNormalized, ImportanceWeights)
    else
        return (; samplepaths, ImportanceWeightsNormalized)
    end
end

# ============================================================================
# draw_paths Helper Functions
# ============================================================================

"""
    _get_or_fit_surrogate(model, is_semimarkov)

Get existing surrogate from model or fit a new one if needed.

For fitted models, reuses the stored `markovsurrogate` to avoid refitting.
For unfitted models, fits a new Markov surrogate.
"""
function _get_or_fit_surrogate(model::MultistateProcess, is_semimarkov::Bool)
    if !is_semimarkov
        return nothing  # Markov models don't need surrogate
    end
    
    # Check if model already has a fitted surrogate
    if !isnothing(model.markovsurrogate) && model.markovsurrogate.fitted
        return model.markovsurrogate
    end
    
    # Need to fit a new surrogate (either no surrogate or not yet fitted)
    fitted_surrogate = _fit_markov_surrogate(model; verbose = false)
    return fitted_surrogate
end

"""
    _setup_tpm_infrastructure(model, params_surrog, hazards_surrog)

Set up TPM books, hazmat books, and solve Kolmogorov equations.
Returns (books, tpm_book, hazmat_book, cache).
"""
function _setup_tpm_infrastructure(model::MultistateProcess, params_surrog, hazards_surrog)
    # containers for bookkeeping TPMs
    books = build_tpm_mapping(model.data)

    # build containers for transition intensity and prob matrices
    hazmat_book = build_hazmat_book(Float64, model.tmat, books[1])
    tpm_book = build_tpm_book(Float64, model.tmat, books[1])

    # allocate memory for matrix exponential
    cache = ExponentialUtilities.alloc_mem(similar(hazmat_book[1]), ExpMethodGeneric())

    # Solve Kolmogorov equations for TPMs
    for t in eachindex(books[1])
        compute_hazmat!(hazmat_book[t], params_surrog, hazards_surrog, books[1][t], model.data)
        compute_tmat!(tpm_book[t], hazmat_book[t], books[1][t], cache)
    end

    return books, tpm_book, hazmat_book, cache
end

"""
    _allocate_path_containers(nsubj, capacity, adaptive)

Allocate containers for sample paths and likelihoods.
"""
function _allocate_path_containers(nsubj::Int, capacity::Int, adaptive::Bool)
    if adaptive
        samplepaths = [sizehint!(Vector{SamplePath}(), capacity) for _ in 1:nsubj]
        loglik_target = [sizehint!(Vector{Float64}(), capacity) for _ in 1:nsubj]
        loglik_surrog = [sizehint!(Vector{Float64}(), capacity) for _ in 1:nsubj]
        ImportanceWeights = [sizehint!(Vector{Float64}(), capacity) for _ in 1:nsubj]
    else
        samplepaths = [Vector{SamplePath}(undef, capacity) for _ in 1:nsubj]
        loglik_target = [Vector{Float64}(undef, capacity) for _ in 1:nsubj]
        loglik_surrog = [Vector{Float64}(undef, capacity) for _ in 1:nsubj]
        ImportanceWeights = [Vector{Float64}(undef, capacity) for _ in 1:nsubj]
    end
    return samplepaths, loglik_target, loglik_surrog, ImportanceWeights
end

"""
    _draw_paths_for_subject!(i, model, adaptive_mode, min_ess, npaths, ...)

Draw sample paths for subject i. Handles both adaptive and fixed-count modes.
"""
function _draw_paths_for_subject!(
        i::Int, model::MultistateProcess, adaptive_mode::Bool, min_ess::Int, npaths_fixed::Union{Nothing, Int},
        samplepaths, loglik_target, loglik_surrog, ImportanceWeights, subj_ess, subj_pareto_k,
        params_target, hazards_target, params_surrog, hazards_surrog,
        tpm_book, hazmat_book, books, fbmats, absorbingstates,
        is_semimarkov::Bool, paretosmooth::Bool)

    # Subject data
    subj_inds = model.subjectindices[i]
    subj_dat = view(model.data, subj_inds, :)

    # Compute forward-backward matrices for panel data
    if any(subj_dat.obstype .∉ Ref([1,2]))
        subj_tpm_map = view(books[2], subj_inds, :)
        subj_emat = view(model.emat, subj_inds, :)
        ForwardFiltering!(fbmats[i], subj_dat, tpm_book, subj_tpm_map, subj_emat;
                         hazmat_book=hazmat_book)
    end

    if adaptive_mode
        _draw_paths_adaptive!(
            i, model, min_ess,
            samplepaths, loglik_target, loglik_surrog, ImportanceWeights, subj_ess, subj_pareto_k,
            params_target, hazards_target, params_surrog, hazards_surrog,
            tpm_book, hazmat_book, books, fbmats, absorbingstates,
            is_semimarkov, paretosmooth
        )
    else
        _draw_paths_fixed!(
            i, model, npaths_fixed,
            samplepaths, loglik_target, loglik_surrog, ImportanceWeights, subj_ess, subj_pareto_k,
            params_target, hazards_target, params_surrog, hazards_surrog,
            tpm_book, hazmat_book, books, fbmats, absorbingstates,
            is_semimarkov, paretosmooth
        )
    end
end

"""
    _draw_paths_adaptive!(i, model, min_ess, ...)

Adaptive sampling: keep sampling until ESS >= min_ess.
"""
function _draw_paths_adaptive!(
        i::Int, model::MultistateProcess, min_ess::Int,
        samplepaths, loglik_target, loglik_surrog, ImportanceWeights, subj_ess, subj_pareto_k,
        params_target, hazards_target, params_surrog, hazards_surrog,
        tpm_book, hazmat_book, books, fbmats, absorbingstates,
        is_semimarkov::Bool, paretosmooth::Bool)

    keep_sampling = true

    while keep_sampling
        # Determine how many paths to add
        current_npaths = length(samplepaths[i])
        n_add = current_npaths == 0 ? min_ess : ceil(Int64, current_npaths * 1.4)

        # Augment containers
        append!(samplepaths[i], Vector{SamplePath}(undef, n_add))
        append!(loglik_target[i], zeros(n_add))
        append!(loglik_surrog[i], zeros(n_add))
        append!(ImportanceWeights[i], zeros(n_add))

        # Sample new paths
        for j in current_npaths .+ (1:n_add)
            _sample_one_path!(
                j, i, model,
                samplepaths, loglik_target, loglik_surrog, ImportanceWeights,
                params_target, hazards_target, params_surrog, hazards_surrog,
                tpm_book, hazmat_book, books, fbmats, absorbingstates,
                is_semimarkov
            )
        end

        # Compute ESS and update importance weights
        _compute_ess_and_weights!(
            i, samplepaths, loglik_target, loglik_surrog, ImportanceWeights, 
            subj_ess, subj_pareto_k, is_semimarkov, paretosmooth, min_ess
        )

        # Check stopping criterion
        if subj_ess[i] >= min_ess
            keep_sampling = false
        end
    end
end

"""
    _draw_paths_fixed!(i, model, npaths, ...)

Fixed-count sampling: draw exactly npaths paths.
"""
function _draw_paths_fixed!(
        i::Int, model::MultistateProcess, npaths::Int,
        samplepaths, loglik_target, loglik_surrog, ImportanceWeights, subj_ess, subj_pareto_k,
        params_target, hazards_target, params_surrog, hazards_surrog,
        tpm_book, hazmat_book, books, fbmats, absorbingstates,
        is_semimarkov::Bool, paretosmooth::Bool)

    # Sample all paths
    for j in 1:npaths
        _sample_one_path!(
            j, i, model,
            samplepaths, loglik_target, loglik_surrog, ImportanceWeights,
            params_target, hazards_target, params_surrog, hazards_surrog,
            tpm_book, hazmat_book, books, fbmats, absorbingstates,
            is_semimarkov
        )
    end

    # Compute ESS and update importance weights
    _compute_ess_and_weights!(
        i, samplepaths, loglik_target, loglik_surrog, ImportanceWeights, 
        subj_ess, subj_pareto_k, is_semimarkov, paretosmooth, npaths
    )
end

"""
    _sample_one_path!(j, i, model, ...)

Sample a single path for subject i, store at index j.
"""
function _sample_one_path!(
        j::Int, i::Int, model::MultistateProcess,
        samplepaths, loglik_target, loglik_surrog, ImportanceWeights,
        params_target, hazards_target, params_surrog, hazards_surrog,
        tpm_book, hazmat_book, books, fbmats, absorbingstates,
        is_semimarkov::Bool)

    # Draw path from surrogate
    samplepaths[i][j] = draw_samplepath(i, model, tpm_book, hazmat_book, books[2], fbmats, absorbingstates)

    # Target log-likelihood
    loglik_target[i][j] = loglik(params_target, samplepaths[i][j], hazards_target, model)

    # Surrogate log-likelihood
    if is_semimarkov
        loglik_surrog[i][j] = loglik(params_surrog, samplepaths[i][j], hazards_surrog, model)
    else
        loglik_surrog[i][j] = loglik_target[i][j]
    end

    # Unsmoothed importance weight
    ImportanceWeights[i][j] = exp(loglik_target[i][j] - loglik_surrog[i][j])
end

"""
    _compute_ess_and_weights!(i, samplepaths, loglik_target, loglik_surrog, ImportanceWeights, subj_ess, subj_pareto_k, is_semimarkov, paretosmooth, default_ess)

Compute ESS and optionally apply Pareto smoothing to importance weights.
"""
function _compute_ess_and_weights!(
        i::Int, samplepaths, loglik_target, loglik_surrog, ImportanceWeights,
        subj_ess, subj_pareto_k, is_semimarkov::Bool, paretosmooth::Bool, default_ess::Int)

    # Handle redundant paths (all same likelihood)
    if allequal(loglik_surrog[i])
        samplepaths[i] = [first(samplepaths[i])]
        loglik_target[i] = [first(loglik_target[i])]
        loglik_surrog[i] = [first(loglik_surrog[i])]
        ImportanceWeights[i] = [1.0]
        subj_ess[i] = default_ess
        return
    end

    # Markov models: ESS = number of paths
    if !is_semimarkov
        subj_ess[i] = length(samplepaths[i])
        return
    end

    # Semi-Markov: compute ESS from importance weights
    logweights = reshape(copy(loglik_target[i] - loglik_surrog[i]), 1, length(loglik_target[i]), 1)

    # All weights equal (no importance sampling needed)
    if !any(logweights .!= 0.0)
        subj_ess[i] = length(samplepaths[i])
        return
    end

    if paretosmooth
        psiw = psis(logweights; source = "other")
        copyto!(ImportanceWeights[i], psiw.weights)
        subj_ess[i] = psiw.ess[1]
        subj_pareto_k[i] = psiw.pareto_k[1]
    else
        weights_raw = exp.(vec(logweights))
        copyto!(ImportanceWeights[i], normalize(weights_raw, 1))
        subj_ess[i] = ParetoSmooth.relative_eff(logweights; source = "other")[1] * length(loglik_target[i])
    end
end

"""
   sample_ecctmc(P, Q, a, b, t0, t1)

Sample path for an endpoint conditioned CTMC whose states at times `t0` and `t1` are `a` and `b`. `P` is the transition probability matrix over the interval, `Q` is the transition intensity matrix. 
"""
function sample_ecctmc(P, Q, a, b, t0, t1)

    # number of states
    nstates = size(Q, 1)

    # length of time interval
    T = t1 - t0

    # maximum total hazard
    m = maximum(abs.(diag(Q)))

    # extract the a,b element of P
    p_ab = P[a,b]

    # generate the auxilliary tpm - optimize this later
    R = ElasticArray{Float64}(undef, nstates, nstates, 1)
    R[:,:,1] = diagm(ones(Float64, nstates)) + Q / m

    # sample threshold for determining number of states
    nthresh = rand(1)[1]
    
    # initialize number of jumps and contitional prob of jumps
    njumps = 0
    cprob = exp(-m*T) * (a==b) / p_ab # cprob of one jump

    if cprob > nthresh 
        return # no jumps, the full path is `a` in [t0,t1]
    else
        # increment the number of jumps and compute cprob
        njumps += 1
        cprob  += exp(-m*T) * (m*T) * R[a,b,1] / p_ab

        # if there is exactly one jump
        if cprob > nthresh
            if a == b
                return # jump is virtual, path is `a` in [t0,t1]
            else
                # jump is real
                times  = rand(Uniform(t0, t1), 1)
                states = [b,]

                # return times and states
                return times, states
            end
        else
            # calculate the number of jumps
            while cprob < nthresh
                # increment the number of jumps
                njumps += 1

                # append the new power of R to the array
                append!(R, R[:,:,1]^njumps)

                # calculate cprob
                cprob += exp(-m*T) * (m*T)^njumps / factorial(big(njumps)) * R[a,b,njumps] / p_ab
            end

            # transition times are uniformly distributed in [t0,t1]
            times = sort!(rand(Uniform(t0, t1), njumps))

            # sample the states at the transition times
            scur   = a
            states = zeros(Int64, njumps)

            for s in 1:(njumps-1)
                snext = sample(1:nstates, Weights(R[scur, :, 1] .* R[:, b, njumps-s] ./ R[scur, b, njumps-s+1]))
                if snext != scur
                    scur = snext
                    states[s] = scur
                end
            end

            states[end] = scur != b ? b : 0

            # determine which transitions are virtual transitions
            jumpinds = findall(states .!= 0)
            
            # return state sequence and times
            return times[jumpinds], states[jumpinds]
        end
    end
end

"""
    sample_ecctmc!(jumptimes, stateseq, P, Q, a, b, t0, t1)

Sample path for an endpoint conditioned CTMC whose states at times `t0` and `t1` are `a` and `b`. `P` is the transition probability matrix over the interval, `Q` is the transition intensity matrix. Jump times and state sequence get appended to `jumptimes` and `stateseq`.
"""
function sample_ecctmc!(jumptimes, stateseq, P, Q, a, b, t0, t1)

    # number of states
    nstates = size(Q, 1)

    # length of time interval
    T = t1 - t0

    # maximum total hazard
    m = maximum(abs.(diag(Q)))

    # extract the a,b element of P
    p_ab = P[a,b]

    # generate the auxilliary tpm - optimize this later
    R = ElasticArray{Float64}(undef, nstates, nstates, 1)
    R[:,:,1] = diagm(ones(Float64, nstates)) + Q / m

    # sample threshold for determining number of states
    nthresh = rand(1)[1]
    
    # initialize number of jumps and contitional prob of jumps
    njumps = 0
    cprob = exp(-m*T) * (a==b) / p_ab # cprob of one jump

    if cprob > nthresh 
        return # no jumps, the full path is `a` in [t0,t1]
    else
        # increment the number of jumps and compute cprob
        njumps += 1
        cprob  += exp(-m*T) * (m*T) * R[a,b,1] / p_ab

        # if there is exactly one jump
        if cprob > nthresh
            if a == b
                return # jump is virtual, path is `a` in [t0,t1]
            else
                # jump is real
                times  = rand(Uniform(t0, t1), 1)
                states = [b,]

                # append times and states
                append!(jumptimes, times)
                append!(stateseq, states)

                return 
            end
        else
            # calculate the number of jumps
            while cprob < nthresh
                # increment the number of jumps
                njumps += 1

                # append the new power of R to the array
                append!(R, R[:,:,1]^njumps)

                # calculate cprob
                cprob += exp(-m*T) * (m*T)^njumps / factorial(big(njumps)) * R[a,b,njumps] / p_ab
            end

            # transition times are uniformly distributed in [t0,t1]
            times = sort!(rand(Uniform(t0, t1), njumps))

            # sample the states at the transition times
            scur   = a
            states = zeros(Int64, njumps)

            for s in 1:(njumps-1)
                snext = sample(1:nstates, Weights(R[scur, :, 1] .* R[:, b, njumps-s] ./ R[scur, b, njumps-s+1]))
                if snext != scur
                    scur = snext
                    states[s] = scur
                end
            end

            states[end] = scur != b ? b : 0

            # determine which transitions are virtual transitions
            jumpinds = findall(states .!= 0)
            
            # return state sequence and times
            append!(jumptimes, times[jumpinds])
            append!(stateseq, states[jumpinds])

            return
        end
    end
end

"""
    draw_samplepath(subj::Int64, model::MultistateProcess, tpm_book, hazmat_book, tpm_map)

Draw sample paths from a Markov surrogate process conditional on panel data.
Uses thread-local workspace for reduced allocations in hot paths.
"""
function draw_samplepath(subj::Int64, model::MultistateProcess, tpm_book, hazmat_book, tpm_map, fbmats, absorbingstates)
    # Get thread-local workspace
    ws = get_path_workspace()
    return draw_samplepath!(ws, subj, model, tpm_book, hazmat_book, tpm_map, fbmats, absorbingstates)
end

"""
    draw_samplepath!(ws::PathWorkspace, subj::Int64, model::MultistateProcess, ...)

Workspace-based version of draw_samplepath for reduced allocations.
Uses pre-allocated workspace vectors, only allocates final SamplePath.
"""
function draw_samplepath!(ws::PathWorkspace, subj::Int64, model::MultistateProcess, tpm_book, hazmat_book, tpm_map, fbmats, absorbingstates)
    # Reset workspace
    reset!(ws)

    # subject data
    subj_inds = model.subjectindices[subj] # rows in the dataset corresponding to the subject
    subj_dat     = view(model.data, subj_inds, :) # subject's data - no shallow copy, just pointer
    subj_tpm_map = view(tpm_map, subj_inds, :)

    # sample any censored observation
    if any(subj_dat.obstype .∉ Ref([1,2]))
        BackwardSampling!(subj_dat, fbmats[subj])
    end

    # initialize sample path with first time/state
    push_time_state!(ws, subj_dat.tstart[1], subj_dat.statefrom[1])

    # loop through data and sample endpoint conditioned paths
    for i in eachindex(subj_inds) # loop over each interval for the subject
        if subj_dat.obstype[i] == 1 
            push_time_state!(ws, subj_dat.tstop[i], subj_dat.stateto[i])
        else
            # sample_ecctmc! needs regular vectors to append to
            # Use workspace view as temporary working vectors
            _sample_ecctmc_ws!(ws, tpm_book[subj_tpm_map[i,1]][subj_tpm_map[i,2]], hazmat_book[subj_tpm_map[i,1]], subj_dat.statefrom[i], subj_dat.stateto[i], subj_dat.tstart[i], subj_dat.tstop[i])
        end
    end

    # append last state and time
    if subj_dat.obstype[end] != 1
        push_time_state!(ws, subj_dat.tstop[end], subj_dat.stateto[end])
    end

    # truncate at entry to absorbing states
    truncind = nothing
    @inbounds for k in 1:ws.states_len
        if ws.states[k] in absorbingstates
            truncind = k
            break
        end
    end
    if !isnothing(truncind)
        ws.times_len = truncind
        ws.states_len = truncind
    end

    # Create and return reduced path
    return reduce_jumpchain_ws(ws, subj)
end

"""
    _sample_ecctmc_ws!(ws::PathWorkspace, P, Q, a, b, t0, t1)

Workspace-based endpoint-conditioned CTMC sampling. Appends to workspace.
Uses pre-allocated arrays from workspace to minimize allocations.
"""
function _sample_ecctmc_ws!(ws::PathWorkspace, P, Q, a, b, t0, t1)
    # number of states
    nstates = size(Q, 1)

    # length of time interval
    T = t1 - t0

    # maximum total hazard
    m = maximum(abs.(diag(Q)))

    # extract the a,b element of P
    p_ab = P[a,b]

    # Ensure workspace has capacity for this state space
    ensure_R_capacity!(ws, nstates, 100)
    
    # Build base R matrix = I + Q/m in workspace (avoid diagm allocation)
    R_base = @view ws.R_base[1:nstates, 1:nstates]
    @inbounds for i in 1:nstates
        for j in 1:nstates
            R_base[i,j] = (i == j ? 1.0 : 0.0) + Q[i,j] / m
        end
    end
    
    # Store first slice (R^1 = R_base)
    @inbounds for i in 1:nstates
        for j in 1:nstates
            ws.R_slices[i,j,1] = R_base[i,j]
        end
    end

    # sample threshold for determining number of states
    nthresh = rand()
    
    # initialize number of jumps and conditional prob of jumps
    njumps = 0
    cprob = exp(-m*T) * (a==b) / p_ab

    if cprob > nthresh 
        return # no jumps, the full path is `a` in [t0,t1]
    else
        # increment the number of jumps and compute cprob
        njumps += 1
        cprob += exp(-m*T) * (m*T) * ws.R_slices[a,b,1] / p_ab

        # if there is exactly one jump
        if cprob > nthresh
            if a == b
                return # jump is virtual, path is `a` in [t0,t1]
            else
                # jump is real - append single time and state
                push_time_state!(ws, rand() * T + t0, b)
                return 
            end
        else
            # calculate the number of jumps - compute R^k iteratively
            R_power = @view ws.R_power[1:nstates, 1:nstates]
            
            while cprob < nthresh
                # increment the number of jumps
                njumps += 1
                
                # Ensure capacity
                if njumps > size(ws.R_slices, 3)
                    ensure_R_capacity!(ws, nstates, 2 * njumps)
                end
                
                # Compute R^njumps = R_base^njumps using matrix multiplication
                # R_slices[:,:,njumps] = R_base^njumps
                if njumps == 2
                    # R^2 = R_base * R_base
                    mul!(@view(ws.R_slices[1:nstates, 1:nstates, 2]), R_base, R_base)
                else
                    # R^k = R^(k-1) * R_base
                    mul!(@view(ws.R_slices[1:nstates, 1:nstates, njumps]), 
                         @view(ws.R_slices[1:nstates, 1:nstates, njumps-1]), R_base)
                end

                # calculate cprob
                cprob += exp(-m*T) * (m*T)^njumps / factorial(big(njumps)) * ws.R_slices[a,b,njumps] / p_ab
            end

            # Ensure temp vectors have capacity
            ensure_temp_capacity!(ws, njumps)
            
            # Generate uniform random times and sort
            @inbounds for k in 1:njumps
                ws.times_temp[k] = rand() * T + t0
            end
            sort!(@view(ws.times_temp[1:njumps]))

            # sample the states at the transition times
            scur = a
            @inbounds for k in 1:njumps
                ws.states_temp[k] = 0
            end

            @inbounds for s in 1:(njumps-1)
                # Compute weights for state sampling
                # weights[i] = R[scur,i,1] * R[i,b,njumps-s] / R[scur,b,njumps-s+1]
                denom = ws.R_slices[scur, b, njumps-s+1]
                weight_sum = 0.0
                for i in 1:nstates
                    weight_sum += ws.R_slices[scur, i, 1] * ws.R_slices[i, b, njumps-s] / denom
                end
                
                # Sample from categorical distribution
                u = rand() * weight_sum
                cumsum = 0.0
                snext = nstates
                for i in 1:nstates
                    cumsum += ws.R_slices[scur, i, 1] * ws.R_slices[i, b, njumps-s] / denom
                    if cumsum >= u
                        snext = i
                        break
                    end
                end
                
                if snext != scur
                    scur = snext
                    ws.states_temp[s] = scur
                end
            end

            ws.states_temp[njumps] = scur != b ? b : 0

            # append only real transitions (non-virtual)
            @inbounds for k in 1:njumps
                if ws.states_temp[k] != 0
                    push_time_state!(ws, ws.times_temp[k], ws.states_temp[k])
                end
            end

            return
        end
    end
end

"""
    reduce_jumpchain_ws(ws::PathWorkspace, subj::Int)

Reduce jump chain directly from workspace (avoid intermediate allocation).
Returns SamplePath with only actual state changes.
"""
function reduce_jumpchain_ws(ws::PathWorkspace, subj::Int)
    pathlen = ws.states_len
    
    # No need to reduce short paths
    if pathlen <= 2
        return SamplePath(subj, ws.times[1:ws.times_len], ws.states[1:ws.states_len])
    end
    
    # Find jump indices (where state actually changes)
    # Always include first and last
    jump_count = 1
    @inbounds for i in 2:pathlen
        if ws.states[i] != ws.states[i-1]
            jump_count += 1
        end
    end
    # Always include last point
    if ws.states[pathlen] == ws.states[pathlen-1]
        jump_count += 1  # Last wasn't counted as a change
    end
    
    # Build reduced arrays
    new_times = Vector{Float64}(undef, jump_count)
    new_states = Vector{Int}(undef, jump_count)
    
    @inbounds begin
        new_times[1] = ws.times[1]
        new_states[1] = ws.states[1]
        
        j = 2
        prev_state = ws.states[1]
        for i in 2:pathlen-1
            if ws.states[i] != prev_state
                new_times[j] = ws.times[i]
                new_states[j] = ws.states[i]
                prev_state = ws.states[i]
                j += 1
            end
        end
        
        # Always include last point
        new_times[j] = ws.times[pathlen]
        new_states[j] = ws.states[pathlen]
    end
    
    return SamplePath(subj, new_times, new_states)
end

"""
    ForwardFiltering!(subj_fbmats, subj_dat, tpm_book, subj_tpm_map, subj_emat; 
                      init_state=nothing, hazmat_book=nothing)

Computes the forward recursion matrices for the FFBS algorithm. Writes into subj_fbmats.

# Arguments
- `subj_fbmats`: Pre-allocated forward-backward matrices
- `subj_dat`: Subject's data view
- `tpm_book`: TPM book
- `subj_tpm_map`: Subject's TPM mapping  
- `subj_emat`: Subject's emission matrix
- `init_state`: Optional initial state specification. Can be:
  - `nothing`: uses subj_dat.statefrom[1] (default)
  - `Int`: single state index (point mass)
  - `Vector{Float64}`: distribution over states (for phase-type with uniform phases)
- `hazmat_book`: Optional hazard rate matrices for instantaneous (dt=0) observations.
                 Required when data contains instantaneous observations (phase-type expanded data).
"""
function ForwardFiltering!(subj_fbmats, subj_dat, tpm_book, subj_tpm_map, subj_emat; 
                           init_state=nothing, hazmat_book=nothing)

    n_times  = size(subj_fbmats, 1)
    n_states = size(subj_fbmats, 2)

    # initialize - handle both point mass and distribution
    if isnothing(init_state)
        p0 = zeros(Float64, n_states)
        p0[subj_dat.statefrom[1]] = 1.0
    elseif init_state isa Integer
        p0 = zeros(Float64, n_states)
        p0[init_state] = 1.0
    else
        # init_state is a distribution vector
        p0 = init_state
    end
    
    # Get TPM for first step - handle instantaneous observations
    tpm = _get_tpm_for_step(1, subj_dat, tpm_book, subj_tpm_map, hazmat_book, n_states)
    
    # First step: include TPM to account for transition probabilities over the interval.
    # This is essential when init_state is a distribution (e.g., phase-type with panel data)
    # because we need to know which (start_phase, end_phase) pairs are reachable.
    # For point mass init_state, the TPM multiplication is still correct (just filters column j).
    subj_fbmats[1, :, :] = (p0 * subj_emat[1,:]') .* tpm
    normalize!(subj_fbmats[1,:,:], 1)

    # recurse
    if n_times > 1
        for s in 2:n_times
            tpm = _get_tpm_for_step(s, subj_dat, tpm_book, subj_tpm_map, hazmat_book, n_states)
            subj_fbmats[s, 1:n_states, 1:n_states] = (sum(subj_fbmats[s-1,:,:], dims = 1)' * subj_emat[s,:]') .* tpm
            normalize!(subj_fbmats[s,:,:], 1)
        end
    end
end

"""
    _get_tpm_for_step(s, subj_dat, tpm_book, subj_tpm_map, hazmat_book, n_states)

Get transition probability matrix for step s. For dt≈0 (instantaneous observations),
computes transition probabilities from hazard ratios instead of matrix exponential.
"""
function _get_tpm_for_step(s::Int, subj_dat, tpm_book, subj_tpm_map, hazmat_book, n_states::Int)
    dt = subj_dat.tstop[s] - subj_dat.tstart[s]
    
    if dt ≈ 0 && !isnothing(hazmat_book)
        # Instantaneous observation: compute P[i,j] = h(i,j) / Σ_k h(i,k) from Q matrix
        # Q[i,j] = h(i,j) for i≠j, Q[i,i] = -Σ_k h(i,k)
        covar_idx = subj_tpm_map[s, 1]
        Q = hazmat_book[covar_idx]
        return _instantaneous_tpm_from_Q(Q, n_states)
    else
        # Regular observation: use pre-computed TPM
        return tpm_book[subj_tpm_map[s,1]][subj_tpm_map[s,2]]
    end
end

"""
    _instantaneous_tpm_from_Q(Q, n_states)

Compute instantaneous transition probabilities from Q matrix.
For a transition that definitely occurred at time t:
- P[i,j] = h(i,j) / Σ_k h(i,k) = Q[i,j] / (-Q[i,i]) for i≠j
- P[i,i] = 0 (cannot stay in same state when transition observed)
"""
function _instantaneous_tpm_from_Q(Q::AbstractMatrix, n_states::Int)
    P = zeros(eltype(Q), n_states, n_states)
    @inbounds for i in 1:n_states
        total_haz = -Q[i, i]  # Q[i,i] = -Σ_k h(i,k)
        if total_haz > 0
            for j in 1:n_states
                if i != j
                    P[i, j] = Q[i, j] / total_haz
                end
                # P[i,i] = 0 by default (transition definitely occurred)
            end
        else
            # Absorbing state: stays in place
            P[i, i] = 1.0
        end
    end
    return P
end


"""
    BackwardSampling!(subj_dat, subj_fbmats)

Samples a path and writes it in to subj_dat.
"""
function BackwardSampling!(subj_dat, subj_fbmats)

    # initialize
    n_times  = size(subj_fbmats, 1)
    n_states = size(subj_fbmats, 2)

    p = normalize(sum(subj_fbmats[n_times,:,:], dims=1), 1) #  dims=1 or  dims=2 ?

    subj_dat.stateto[end] = rand(Categorical(vec(p)))

    # recurse
    if n_times > 1
        for t in (n_times - 1):-1:1
            subj_dat.stateto[t] = rand(Categorical(normalize(subj_fbmats[t+1, :, subj_dat.stateto[t + 1]], 1)))
        end
        subj_dat.statefrom[Not(1)] .= subj_dat.stateto[Not(end)]
    end
end


function BackwardSampling(m, p) 
    
    n_obs = size(p, 1) # number of observations
    h = Array{Int64}(undef, n_obs)

    # 1. draw draw h_n ~ pi_n
    h[n_obs] = rand(Categorical(m[n_obs+1,:]))

    # 2. draw h_t|h_{t+1}=s ~ p_{t,.,s}
    for t in (n_obs-1):-1:1
        w = p[t+1,:,h[t+1]] / sum(p[t+1,:,h[t+1]])
       h[t] = rand(Categorical(w)) # [Eq. 10]
    end

    return h

end


"""
    ComputeImportanceWeights!(loglik_target, loglik_surrog, _logImportanceWeights, ImportanceWeights, ess_target)

Compute the importance weights and ess.
"""
function ComputeImportanceWeightsESS!(loglik_target, loglik_surrog, _logImportanceWeights, ImportanceWeights, ess_cur, ess_target, psis_pareto_k)

    for i in eachindex(loglik_surrog)
        # recompute the log unnormalized importance weight
        _logImportanceWeights[i] = loglik_target[i] .- loglik_surrog[i]

        if length(_logImportanceWeights[i]) == 1
            # make sure the ESS is equal to the target
            ImportanceWeights[i] = [1.0,]
            ess_cur[i] = ess_target

        elseif length(_logImportanceWeights[i]) != 1
            # Check for degenerate weights - either all near zero or very small range
            weights_range = maximum(_logImportanceWeights[i]) - minimum(_logImportanceWeights[i])
            weights_degenerate = all(isapprox.(_logImportanceWeights[i], 0.0; atol = sqrt(eps()))) || 
                                 weights_range < 1e-10
            
            if weights_degenerate
                fill!(ImportanceWeights[i], 1/length(ImportanceWeights[i]))
                ess_cur[i] = ess_target
                psis_pareto_k[i] = 0.0
            else
                # Try PSIS, but catch errors from degenerate tail distributions
                psiw = try
                    ParetoSmooth.psis(reshape(copy(_logImportanceWeights[i]), 1, length(_logImportanceWeights[i]), 1); source = "other")
                catch e
                    if e isa ArgumentError && occursin("all tail values are the same", string(e))
                        # Fall back to uniform weights when tail is degenerate
                        @warn "PSIS failed for subject $i in ComputeImportanceWeightsESS! (degenerate tail); using uniform weights" maxlog=5
                        nothing
                    else
                        rethrow(e)
                    end
                end
                
                if isnothing(psiw)
                    # PSIS failed, use uniform weights
                    fill!(ImportanceWeights[i], 1/length(ImportanceWeights[i]))
                    ess_cur[i] = length(ImportanceWeights[i])
                    psis_pareto_k[i] = Inf  # Mark as unreliable
                else
                    # save normalized importance weights and ess
                    copyto!(ImportanceWeights[i], psiw.weights)
                    ess_cur[i] = psiw.ess[1]
                    psis_pareto_k[i] = psiw.pareto_k[1]
                end
            end
        end
    end
end


