# ============================================================================
# Thread-local workspace for path sampling (allocation reduction)
# ============================================================================

"""
    PathWorkspace

Pre-allocated workspace for path sampling to reduce allocations in hot loops.
Each thread gets its own workspace to avoid contention.

Contains:
- `times`, `states`: Main path vectors
- `times_temp`, `states_temp`: Temporary vectors for ECCTMC sampling
- `R_slices`: Pre-allocated 3D array for R matrix powers
- `R_base`, `R_power`: Workspace matrices for matrix operations
"""
mutable struct PathWorkspace
    # Main path storage
    times::Vector{Float64}
    states::Vector{Int}
    times_len::Int
    states_len::Int
    
    # ECCTMC temporary vectors
    times_temp::Vector{Float64}
    states_temp::Vector{Int}
    
    # R matrix storage (3D: nstates x nstates x max_jumps)
    R_slices::Array{Float64, 3}
    R_base::Matrix{Float64}     # Base R = I + Q/m
    R_power::Matrix{Float64}    # Workspace for matrix power
    nstates::Int                # Current state space size
    
    function PathWorkspace(max_jumps::Int=1000, max_states::Int=10)
        new(
            Vector{Float64}(undef, max_jumps),
            Vector{Int}(undef, max_jumps),
            0, 0,
            Vector{Float64}(undef, max_jumps),
            Vector{Int}(undef, max_jumps),
            Array{Float64}(undef, max_states, max_states, max_jumps),
            Matrix{Float64}(undef, max_states, max_states),
            Matrix{Float64}(undef, max_states, max_states),
            max_states
        )
    end
end

"""Reset workspace for new path"""
@inline function reset!(ws::PathWorkspace)
    ws.times_len = 0
    ws.states_len = 0
end

"""Ensure R matrices are sized correctly for nstates"""
@inline function ensure_R_capacity!(ws::PathWorkspace, nstates::Int, njumps::Int)
    if nstates > ws.nstates || njumps > size(ws.R_slices, 3)
        new_nstates = max(nstates, ws.nstates)
        new_njumps = max(njumps, size(ws.R_slices, 3))
        ws.R_slices = Array{Float64}(undef, new_nstates, new_nstates, new_njumps)
        ws.R_base = Matrix{Float64}(undef, new_nstates, new_nstates)
        ws.R_power = Matrix{Float64}(undef, new_nstates, new_nstates)
        ws.nstates = new_nstates
    end
end

"""Ensure temp vectors have capacity"""
@inline function ensure_temp_capacity!(ws::PathWorkspace, n::Int)
    if n > length(ws.times_temp)
        resize!(ws.times_temp, max(n, 2 * length(ws.times_temp)))
        resize!(ws.states_temp, max(n, 2 * length(ws.states_temp)))
    end
end

"""Push time to workspace, growing if needed"""
@inline function push_time!(ws::PathWorkspace, t::Float64)
    ws.times_len += 1
    if ws.times_len > length(ws.times)
        resize!(ws.times, 2 * ws.times_len)
    end
    @inbounds ws.times[ws.times_len] = t
end

"""Push state to workspace, growing if needed"""
@inline function push_state!(ws::PathWorkspace, s::Int)
    ws.states_len += 1
    if ws.states_len > length(ws.states)
        resize!(ws.states, 2 * ws.states_len)
    end
    @inbounds ws.states[ws.states_len] = s
end

"""Push time and state together"""
@inline function push_time_state!(ws::PathWorkspace, t::Float64, s::Int)
    push_time!(ws, t)
    push_state!(ws, s)
end

"""Append multiple times to workspace"""
@inline function append_times!(ws::PathWorkspace, times::AbstractVector{Float64})
    n = length(times)
    new_len = ws.times_len + n
    if new_len > length(ws.times)
        resize!(ws.times, max(2 * new_len, new_len + 100))
    end
    @inbounds for i in 1:n
        ws.times[ws.times_len + i] = times[i]
    end
    ws.times_len = new_len
end

"""Append multiple states to workspace"""
@inline function append_states!(ws::PathWorkspace, states::AbstractVector{Int})
    n = length(states)
    new_len = ws.states_len + n
    if new_len > length(ws.states)
        resize!(ws.states, max(2 * new_len, new_len + 100))
    end
    @inbounds for i in 1:n
        ws.states[ws.states_len + i] = states[i]
    end
    ws.states_len = new_len
end

"""Get current times as view"""
@inline function get_times(ws::PathWorkspace)
    @view ws.times[1:ws.times_len]
end

"""Get current states as view"""  
@inline function get_states(ws::PathWorkspace)
    @view ws.states[1:ws.states_len]
end

"""Create SamplePath from workspace (copies data)"""
function to_samplepath(ws::PathWorkspace, subj::Int)
    SamplePath(subj, ws.times[1:ws.times_len], ws.states[1:ws.states_len])
end

# Thread-local workspace storage
const PATH_WORKSPACES = Dict{Int, PathWorkspace}()
const PATH_WORKSPACE_LOCK = ReentrantLock()

"""
    get_path_workspace() -> PathWorkspace

Get or create a thread-local PathWorkspace for the current thread.

Thread-safe: Uses a lock to ensure only one workspace is created per thread
even under concurrent access from multiple threads.
"""
function get_path_workspace()::PathWorkspace
    tid = Threads.threadid()
    # Always acquire lock to avoid TOCTOU race condition
    # The lock ensures thread-safe initialization
    lock(PATH_WORKSPACE_LOCK) do
        get!(PATH_WORKSPACES, tid) do
            PathWorkspace(1000)
        end
    end
end

# ============================================================================
# Main DrawSamplePaths! functions
# ============================================================================

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
    expanded_ph_data=nothing, expanded_ph_subjectindices=nothing, expanded_ph_tpm_map=nothing, ph_original_row_map=nothing)

    # Determine if using phase-type proposals
    use_phasetype = !isnothing(phasetype_surrogate)

    # make sure spline parameters are assigned correctly
    # unflatten parameters to natural scale (AD-compatible)
    pars = unflatten_natural(params_cur, model)

    # update spline hazards with current parameters (no-op for functional splines)
    _update_spline_hazards!(model.hazards, pars)

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
            ph_original_row_map = ph_original_row_map)
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
    expanded_ph_data=nothing, expanded_ph_subjectindices=nothing, expanded_ph_tpm_map=nothing, ph_original_row_map=nothing)

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
                
                path_result = draw_samplepath_phasetype(i, model, tpm_book_ph, hazmat_book_ph, 
                                                         ph_tpm_map, fbmats_ph, emat_ph, 
                                                         phasetype_surrogate, absorbingstates;
                                                         expanded_data = expanded_ph_data,
                                                         expanded_subjectindices = expanded_ph_subjectindices,
                                                         original_row_map = ph_original_row_map)
                
                # Store collapsed path for target likelihood evaluation
                samplepaths[i][j] = path_result.collapsed
                
                # Surrogate log-likelihood: marginal density of COLLAPSED path under phase-type
                # This ensures importance weight = f(Z|θ) / q(Z|θ') evaluates the SAME path Z
                # in both numerator and denominator (essential for correct IS)
                loglik_surrog[i][j] = loglik_phasetype_collapsed_path(path_result.collapsed, phasetype_surrogate)
            else
                # Markov proposal: standard sampling
                samplepaths[i][j] = draw_samplepath(i, model, tpm_book_surrogate, hazmat_book_surrogate, 
                                                    books[2], fbmats, absorbingstates)
                
                # Surrogate log-likelihood under Markov proposal (family-aware)
                surrogate_pars = get_hazard_params(surrogate.parameters, surrogate.hazards)
                loglik_surrog[i][j] = loglik(surrogate_pars, samplepaths[i][j], surrogate.hazards, model)
            end
            
            # target log-likelihood (same for both proposal types)
            # unflatten_natural returns natural-scale params
            target_pars = unflatten_natural(params_cur, model)
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
            # the case when the target and the surrogate are the same
            if all(iszero.(_logImportanceWeights[i]))
                fill!(ImportanceWeights[i], 1/length(ImportanceWeights[i]))
                ess_cur[i] = length(ImportanceWeights[i])
                psis_pareto_k[i] = 0.0
            else
                # might fail if not enough samples to fit pareto
                try
                    # pareto smoothed importance weights
                    psiw = ParetoSmooth.psis(reshape(copy(_logImportanceWeights[i]), 1, length(_logImportanceWeights[i]), 1); source = "other");
    
                    # save normalized importance weights and ess
                    copyto!(ImportanceWeights[i], psiw.weights)
                    ess_cur[i] = psiw.ess[1]
                    psis_pareto_k[i] = psiw.pareto_k[1]
                    
                    # Handle NaN ESS from PSIS (can happen with high Pareto-k)
                    # Fall back to simple ESS calculation
                    if isnan(ess_cur[i]) || isinf(ess_cur[i])
                        # exponentiate and normalize the unnormalized log weights
                        copyto!(ImportanceWeights[i], normalize(exp.(_logImportanceWeights[i] .- maximum(_logImportanceWeights[i])), 1))
                        ess_cur[i] = 1 / sum(ImportanceWeights[i] .^ 2)
                        # Keep the high pareto_k to indicate unreliable weights
                    end
    
                catch err
                    # exponentiate and normalize the unnormalized log weights
                    copyto!(ImportanceWeights[i], normalize(exp.(_logImportanceWeights[i]), 1))

                    # calculate the ess
                    ess_cur[i] = 1 / sum(ImportanceWeights[i] .^ 2)
                    psis_pareto_k[i] = 1.0
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

# Backward compatibility: positional npaths argument (deprecated)
"""
    draw_paths(model::MultistateProcess, npaths::Int; paretosmooth=true, return_logliks=false)

Draw a fixed number of sample paths. This is a convenience method equivalent to
`draw_paths(model; npaths=npaths, ...)`.

!!! note
    This positional argument form is deprecated. Use `draw_paths(model; npaths=n)` instead.
"""
function draw_paths(model::MultistateProcess, npaths::Int; paretosmooth::Bool = true, return_logliks::Bool = false)
    Base.depwarn(
        "draw_paths(model, npaths) is deprecated. Use draw_paths(model; npaths=npaths) instead.",
        :draw_paths
    )
    return draw_paths(model; npaths=npaths, paretosmooth=paretosmooth, return_logliks=return_logliks)
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
        try
            psiw = psis(logweights; source = "other")
            copyto!(ImportanceWeights[i], psiw.weights)
            subj_ess[i] = psiw.ess[1]
            subj_pareto_k[i] = psiw.pareto_k[1]
        catch err
            # Fall back to simple normalization
            copyto!(ImportanceWeights[i], normalize(exp.(vec(logweights)), 1))
            subj_ess[i] = ParetoSmooth.relative_eff(logweights; source = "other")[1] * length(loglik_target[i])
        end
    else
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
            if all(isapprox.(_logImportanceWeights[i], 0.0; atol = sqrt(eps())))
                fill!(ImportanceWeights[i], 1/length(ImportanceWeights[i]))
                ess_cur[i] = ess_target
                psis_pareto_k[i] = 0.0
            else
                # might fail if not enough samples to fit pareto
                try
                    # pareto smoothed importance weights
                    psiw = ParetoSmooth.psis(reshape(copy(_logImportanceWeights[i]), 1, length(_logImportanceWeights[i]), 1); source = "other");
    
                    # save normalized importance weights and ess
                    copyto!(ImportanceWeights[i], psiw.weights)
                    ess_cur[i] = psiw.ess[1]
                    psis_pareto_k[i] = psiw.pareto_k[1]
    
                catch err
                    @debug "PSIS failed for subject $i; falling back to standard importance weights" exception=(err, catch_backtrace())
                    # exponentiate and normalize the unnormalized log weights using log-sum-exp for numerical stability
                    max_logw = maximum(_logImportanceWeights[i])
                    log_sum_exp = max_logw + log(sum(exp.(_logImportanceWeights[i] .- max_logw)))
                    copyto!(ImportanceWeights[i], exp.(_logImportanceWeights[i] .- log_sum_exp))

                    # calculate the ess
                    ess_cur[i] = 1 / sum(ImportanceWeights[i] .^ 2)
                    psis_pareto_k[i] = 1.0
                end
            end
        end
    end
end


# =============================================================================
# Phase-Type Surrogate Sampling Functions
# =============================================================================
#
# These functions implement forward-filtering backward-sampling (FFBS) on an
# expanded phase-type state space for improved MCEM importance sampling.
#
# Key idea: Each observed state is expanded into multiple phases. The expanded
# Markov chain better approximates non-exponential sojourn times. After sampling
# in the expanded space, paths are collapsed back to observed states.
# =============================================================================

"""
    build_phasetype_tpm_book(surrogate::PhaseTypeSurrogate, books, data)

Build transition probability matrix book for phase-type expanded state space.

# Arguments
- `surrogate`: PhaseTypeSurrogate with expanded Q matrix
- `books`: Time interval book from build_tpm_mapping
- `data`: Model data

# Returns
- `tpm_book_ph`: Nested vector of TPMs [covar_combo][time_interval] in expanded space
- `hazmat_book_ph`: Vector of intensity matrices for each covariate combination
"""
function build_phasetype_tpm_book(surrogate::PhaseTypeSurrogate, books, data)
    n_expanded = surrogate.n_expanded_states
    Q_expanded = surrogate.expanded_Q
    
    # books[1] is a vector of DataFrames, one per covariate combination
    # Each DataFrame has columns (tstart, tstop, datind) with rows for unique time intervals
    n_covar_combos = length(books[1])
    
    # Allocate TPM book: [covar_combo][time_interval]
    # For phase-type with homogeneous Q, we still need this structure to match
    # how the Markov surrogate book is organized
    tpm_book_ph = [[zeros(Float64, n_expanded, n_expanded) for _ in 1:nrow(books[1][c])] for c in 1:n_covar_combos]
    
    # Allocate hazmat book: one Q per covariate combination (all same for homogeneous PH)
    hazmat_book_ph = [copy(Q_expanded) for _ in 1:n_covar_combos]
    
    # Allocate cache for matrix exponential
    cache = ExponentialUtilities.alloc_mem(Q_expanded, ExpMethodGeneric())
    
    # Compute TPMs for each covariate combination and time interval
    for c in 1:n_covar_combos
        tpm_index = books[1][c]  # DataFrame with (tstart, tstop, datind)
        for t in 1:nrow(tpm_index)
            dt = tpm_index.tstop[t]  # Time interval length (tstart is always 0)
            # TPM = exp(Q * dt)
            tpm_book_ph[c][t] .= exponential!(copy(Q_expanded) .* dt, ExpMethodGeneric(), cache)
        end
    end
    
    return tpm_book_ph, hazmat_book_ph
end


"""
    build_phasetype_emat_expanded(model, surrogate::PhaseTypeSurrogate;
                                  expanded_data::Union{Nothing, DataFrame} = nothing,
                                  censoring_patterns::Union{Nothing, Matrix{Float64}} = nothing)

Build emission matrix mapping expanded phases to observed states for FFBS.

For each observation, the emission matrix E[i,j] gives the probability that
the subject is in phase j given the observation.

# Observation Types
- obstype 1: Exact transition → only FIRST phase of stateto has probability 1
  (In Coxian models, transitions always enter at phase 1 of the destination state)
- obstype 2: Panel/right-censored → all phases of stateto have equal probability
- obstype 0: Fully censored → all phases equally likely
- obstype > 2: Partial censoring → use censoring_patterns matrix
  - For phase-type data expansion, obstype = 2 + s means "in state s, phase unknown"

# Arguments
- `model`: MultistateProcess with data
- `surrogate`: PhaseTypeSurrogate with state/phase mappings
- `expanded_data`: Optional expanded data (if None, uses model.data)
- `censoring_patterns`: Optional censoring patterns for obstype > 2

# Returns
- Matrix of size (n_observations, n_expanded_states)
"""
function build_phasetype_emat_expanded(model, surrogate::PhaseTypeSurrogate;
                                       expanded_data::Union{Nothing, DataFrame} = nothing,
                                       censoring_patterns::Union{Nothing, Matrix{Float64}} = nothing)
    
    data = isnothing(expanded_data) ? model.data : expanded_data
    n_obs = nrow(data)
    n_expanded = surrogate.n_expanded_states
    n_observed = surrogate.n_observed_states
    
    emat = zeros(Float64, n_obs, n_expanded)
    
    for i in 1:n_obs
        obstype = data.obstype[i]
        
        if obstype == 1
            # Exact observation - transition always goes to FIRST phase of destination
            # (In Coxian models, you always enter a state at phase 1)
            observed_state = data.stateto[i]
            if observed_state > 0 && observed_state <= n_observed
                first_phase = first(surrogate.state_to_phases[observed_state])
                emat[i, first_phase] = 1.0
            else
                # Invalid state, allow all phases
                emat[i, :] .= 1.0
            end
        elseif obstype == 2
            # Panel observation - any phase of observed state is possible
            observed_state = data.stateto[i]
            if observed_state > 0 && observed_state <= n_observed
                phases = surrogate.state_to_phases[observed_state]
                emat[i, phases] .= 1.0
            else
                # Invalid state, allow all phases
                emat[i, :] .= 1.0
            end
        elseif obstype == 0
            # Fully censored - all phases equally likely
            emat[i, :] .= 1.0
        elseif obstype > 2
            # Partial censoring
            if !isnothing(censoring_patterns)
                # Use provided censoring patterns
                # For phase-type expansion: obstype = 2 + s means state s is known
                pattern_idx = obstype - 2
                if pattern_idx <= size(censoring_patterns, 1)
                    for s in 1:min(n_observed, size(censoring_patterns, 2) - 1)
                        state_prob = censoring_patterns[pattern_idx, s + 1]
                        if state_prob > 0
                            phases = surrogate.state_to_phases[s]
                            n_phases = length(phases)
                            # Divide probability mass equally among phases
                            emat[i, phases] .= state_prob / n_phases
                        end
                    end
                else
                    # Pattern index out of range, allow all phases
                    emat[i, :] .= 1.0
                end
            else
                # No censoring patterns provided
                # For phase-type expansion convention: obstype = 2 + s means in state s
                censored_state = obstype - 2
                if censored_state >= 1 && censored_state <= n_observed
                    phases = surrogate.state_to_phases[censored_state]
                    emat[i, phases] .= 1.0
                else
                    # Invalid censoring code, allow all phases
                    emat[i, :] .= 1.0
                end
            end
        else
            # Unknown observation type, allow all phases
            emat[i, :] .= 1.0
        end
        
        # Normalize if any positive entries
        row_sum = sum(emat[i, :])
        if row_sum > 0
            emat[i, :] ./= row_sum
        end
    end
    
    return emat
end


"""
    build_fbmats_phasetype(model, surrogate::PhaseTypeSurrogate)

Allocate forward-backward matrices for FFBS on expanded phase-type state space.

# Returns
- Vector of 3D arrays, one per subject, of size (n_obs, n_expanded, n_expanded)
"""
function build_fbmats_phasetype(model, surrogate::PhaseTypeSurrogate)
    return build_fbmats_phasetype_with_indices(model.subjectindices, surrogate)
end

"""
    build_fbmats_phasetype_with_indices(subjectindices, surrogate::PhaseTypeSurrogate)

Allocate forward-backward matrices for FFBS on expanded phase-type state space.

This version takes explicit subject indices, useful when working with expanded data.

# Arguments
- `subjectindices`: Vector of indices per subject (UnitRange or Vector{Int})
- `surrogate`: PhaseTypeSurrogate

# Returns
- Vector of 3D arrays, one per subject, of size (n_obs, n_expanded, n_expanded)
"""
function build_fbmats_phasetype_with_indices(subjectindices, surrogate::PhaseTypeSurrogate)
    n_expanded = surrogate.n_expanded_states
    
    fbmats = Vector{Array{Float64, 3}}(undef, length(subjectindices))
    
    for i in eachindex(subjectindices)
        subj_inds = subjectindices[i]
        n_obs = length(subj_inds)
        fbmats[i] = zeros(Float64, n_obs, n_expanded, n_expanded)
    end
    
    return fbmats
end


# =============================================================================
# Phase-Type Sampling Functions
# =============================================================================
#
# These functions support MCEM with phase-type proposals. The key insight is that
# a phase-type expanded model is still Markov, so we can reuse the existing FFBS
# machinery. The only differences are:
#
# 1. The state space is expanded (each observed state split into phases)
# 2. The emission matrix duplicates indicators across phases of the same state
# 3. After sampling, we collapse the expanded path back to observed states
#
# =============================================================================

"""
    expand_emat(emat, surrogate::PhaseTypeSurrogate)

Expand emission matrix from observed states to phase-type expanded states.

For each row (observation), the emission probability for an observed state is
duplicated across all phases of that state. This is correct because:
  P(obstype | phase_k of state s) = P(obstype | state s)

# Arguments
- `emat`: Original emission matrix (n_obs × n_observed_states)
- `surrogate`: PhaseTypeSurrogate with state-to-phase mappings

# Returns
- Expanded emission matrix (n_obs × n_expanded_states)
"""
function expand_emat(emat::AbstractMatrix, surrogate::PhaseTypeSurrogate)
    n_obs = size(emat, 1)
    n_expanded = surrogate.n_expanded_states
    
    emat_expanded = zeros(Float64, n_obs, n_expanded)
    
    for obs in 1:n_obs
        for (state, phases) in enumerate(surrogate.state_to_phases)
            emission_prob = emat[obs, state]
            for phase in phases
                emat_expanded[obs, phase] = emission_prob
            end
        end
    end
    
    return emat_expanded
end


"""
    BackwardSampling_expanded(subj_fbmats, surrogate::PhaseTypeSurrogate)

Backward sampling that returns expanded state indices.

Unlike `BackwardSampling!` which writes observed states to `subj_dat`, this
function returns the sampled expanded state sequence. The caller is responsible
for mapping back to observed states.

# Arguments
- `subj_fbmats`: Forward-backward matrices from ForwardFiltering!
- `surrogate`: PhaseTypeSurrogate (used only for n_expanded_states)

# Returns
- `Vector{Int}`: Sampled expanded state at each observation time
"""
function BackwardSampling_expanded(subj_fbmats, n_expanded::Int)
    n_times = size(subj_fbmats, 1)
    
    expanded_states = Vector{Int}(undef, n_times)
    
    # Sample final state
    p = normalize(vec(sum(subj_fbmats[n_times, :, :], dims=1)), 1)
    expanded_states[n_times] = rand(Categorical(p))
    
    # Backward recursion
    if n_times > 1
        for t in (n_times - 1):-1:1
            cond_probs = normalize(subj_fbmats[t+1, :, expanded_states[t+1]], 1)
            expanded_states[t] = rand(Categorical(cond_probs))
        end
    end
    
    return expanded_states
end


"""
    draw_samplepath_phasetype(subj, model, tpm_book_ph, hazmat_book_ph, 
                               tpm_map, fbmats_ph, emat_ph, surrogate, absorbingstates)

Draw a sample path from the phase-type surrogate, collapsed to observed states.

Uses the existing FFBS machinery on the expanded state space, then collapses
the sampled path back to observed states.

# Algorithm
1. Run ForwardFiltering! on expanded state space (existing function)
2. Run BackwardSampling_expanded to get expanded state endpoints
3. Run sample_ecctmc! on expanded Q matrix between expanded endpoints
4. Collapse the full path from expanded to observed states

# Returns
- `NamedTuple` with fields:
  - `collapsed`: SamplePath in observed state space (for target likelihood)
  - `expanded`: SamplePath in expanded phase state space (for surrogate likelihood)
"""
function draw_samplepath_phasetype(subj::Int64, model::MultistateProcess, 
                                    tpm_book_ph, hazmat_book_ph, tpm_map, 
                                    fbmats_ph, emat_ph, surrogate::PhaseTypeSurrogate, 
                                    absorbingstates;
                                    # Optional expanded data infrastructure for exact observations
                                    expanded_data::Union{Nothing, DataFrame} = nothing,
                                    expanded_subjectindices::Union{Nothing, Vector{UnitRange{Int64}}} = nothing,
                                    original_row_map::Union{Nothing, Vector{Int}} = nothing)
    
    # Determine if we should use expanded data for FFBS
    # This is needed when data has exact observations (obstype=1) to properly
    # account for phase uncertainty during sojourn times
    use_expanded = !isnothing(expanded_data) && !isnothing(expanded_subjectindices)
    
    if use_expanded
        return _draw_samplepath_phasetype_expanded(subj, model, tpm_book_ph, hazmat_book_ph,
                                                    tpm_map, fbmats_ph, emat_ph, surrogate,
                                                    absorbingstates, expanded_data,
                                                    expanded_subjectindices, original_row_map)
    else
        return _draw_samplepath_phasetype_original(subj, model, tpm_book_ph, hazmat_book_ph,
                                                    tpm_map, fbmats_ph, emat_ph, surrogate,
                                                    absorbingstates)
    end
end

"""
    _draw_samplepath_phasetype_original(...)

Internal implementation for panel/censored data without expansion.
Uses the original data structure for FFBS.
"""
function _draw_samplepath_phasetype_original(subj::Int64, model::MultistateProcess, 
                                              tpm_book_ph, hazmat_book_ph, tpm_map, 
                                              fbmats_ph, emat_ph, surrogate::PhaseTypeSurrogate, 
                                              absorbingstates)
    
    # Subject data from original model
    subj_inds = model.subjectindices[subj]
    subj_dat = view(model.data, subj_inds, :)
    subj_tpm_map = view(tpm_map, subj_inds, :)
    subj_emat_ph = view(emat_ph, subj_inds, :)
    
    n_obs = length(subj_inds)
    n_expanded = surrogate.n_expanded_states
    
    # Determine initial phase distribution based on first observation type
    initial_obs_state = subj_dat.statefrom[1]
    initial_phases = surrogate.state_to_phases[initial_obs_state]
    
    if subj_dat.obstype[1] == 1
        # Exact observation: entry is always into first phase (Coxian assumption)
        init_expanded = first(initial_phases)
    else
        # Panel/censored: uniform over phases of initial state
        # Create distribution vector for ForwardFiltering!
        init_expanded = zeros(Float64, n_expanded)
        init_expanded[initial_phases] .= 1.0 / length(initial_phases)
    end
    
    # Run FFBS to sample phase endpoints
    ForwardFiltering!(fbmats_ph[subj], subj_dat, tpm_book_ph, subj_tpm_map, subj_emat_ph;
                      init_state=init_expanded, hazmat_book=hazmat_book_ph)
    
    # Backward sample to get expanded state sequence
    expanded_states = BackwardSampling_expanded(fbmats_ph[subj], n_expanded)
    
    # For path construction, sample initial phase from backward distribution
    # (first element of expanded_states is already the sampled initial phase)
    init_phase_for_path = if subj_dat.obstype[1] == 1
        first(initial_phases)  # Exact: always first phase
    else
        expanded_states[1]  # Panel: use sampled phase (but we need to sample it)
    end
    
    # For panel data, we need to sample the initial phase (at tstart[1])
    # conditioned on the endpoint phase at tstop[1] (which is expanded_states[1]).
    # This ensures we only sample start phases that can reach the endpoint.
    if subj_dat.obstype[1] != 1
        # Sample initial phase conditioned on the endpoint phase at tstop[1]
        p0_given_endpoint = normalize(fbmats_ph[subj][1, :, expanded_states[1]], 1)
        init_phase_for_path = rand(Categorical(p0_given_endpoint))
    end
    
    # Initialize path in expanded space
    times_expanded = [subj_dat.tstart[1]]
    states_expanded = [init_phase_for_path]
    sizehint!(times_expanded, n_expanded * 2)
    sizehint!(states_expanded, n_expanded * 2)
    
    # Loop through intervals and sample endpoint-conditioned paths in expanded space
    for i in 1:n_obs
        # Get transition probability matrix and rate matrix for this interval
        covar_idx = subj_tpm_map[i, 1]
        time_idx = subj_tpm_map[i, 2]
        P_expanded = tpm_book_ph[covar_idx][time_idx]
        Q_expanded = hazmat_book_ph[covar_idx]
        
        # Source state in expanded space
        a_expanded = states_expanded[end]
        
        # Destination phase from FFBS
        b_expanded = expanded_states[i]
        
        if subj_dat.obstype[i] == 1
            # Exact observation: transition time is known, phase is sampled
            # Record the transition at the observed time with sampled phase
            if subj_dat.statefrom[i] != subj_dat.stateto[i]
                push!(times_expanded, subj_dat.tstop[i])
                push!(states_expanded, b_expanded)
            end
        else
            # Censored/panel observation - sample path between endpoints
            sample_ecctmc!(times_expanded, states_expanded, P_expanded, Q_expanded, 
                          a_expanded, b_expanded, subj_dat.tstart[i], subj_dat.tstop[i])
            
            # Ensure we end at the sampled destination
            if states_expanded[end] != b_expanded
                push!(times_expanded, subj_dat.tstop[i])
                push!(states_expanded, b_expanded)
            end
        end
    end
    
    # Create expanded SamplePath for surrogate likelihood
    expanded_path = SamplePath(subj, copy(times_expanded), copy(states_expanded))
    
    # Collapse expanded path to observed states
    collapsed_path = collapse_phasetype_path(expanded_path, surrogate, absorbingstates)
    
    # Return both collapsed and expanded paths
    return (collapsed=collapsed_path, expanded=expanded_path)
end


"""
    _draw_samplepath_phasetype_expanded(...)

Internal implementation for exact data using expanded data structure.
Runs FFBS on the expanded data to properly account for phase uncertainty
during sojourn times.
"""
function _draw_samplepath_phasetype_expanded(subj::Int64, model::MultistateProcess, 
                                              tpm_book_ph, hazmat_book_ph, tpm_map,
                                              fbmats_ph, emat_ph, surrogate::PhaseTypeSurrogate,
                                              absorbingstates, expanded_data::DataFrame,
                                              expanded_subjectindices::Vector{UnitRange{Int64}},
                                              original_row_map::Union{Nothing, Vector{Int}})
    
    # Get subject data from both original and expanded datasets
    orig_subj_inds = model.subjectindices[subj]
    orig_subj_dat = view(model.data, orig_subj_inds, :)
    
    exp_subj_inds = expanded_subjectindices[subj]
    exp_subj_dat = view(expanded_data, exp_subj_inds, :)
    exp_subj_tpm_map = view(tpm_map, exp_subj_inds, :)
    exp_subj_emat_ph = view(emat_ph, exp_subj_inds, :)
    
    n_orig_obs = length(orig_subj_inds)
    n_exp_obs = length(exp_subj_inds)
    n_expanded = surrogate.n_expanded_states
    
    # Determine initial phase based on first observation type in ORIGINAL data
    initial_obs_state = orig_subj_dat.statefrom[1]
    initial_phases = surrogate.state_to_phases[initial_obs_state]
    
    if orig_subj_dat.obstype[1] == 1
        # Exact observation: entry is always into first phase (Coxian assumption)
        init_expanded = first(initial_phases)
        init_phase_for_path = init_expanded
    else
        # Panel/censored: uniform over phases of initial state
        init_expanded = zeros(Float64, n_expanded)
        init_expanded[initial_phases] .= 1.0 / length(initial_phases)
        init_phase_for_path = nothing  # Will be sampled after forward pass
    end
    
    # Run FFBS on expanded data to sample phase endpoints
    ForwardFiltering!(fbmats_ph[subj], exp_subj_dat, tpm_book_ph, exp_subj_tpm_map, exp_subj_emat_ph;
                      init_state=init_expanded, hazmat_book=hazmat_book_ph)
    
    # Backward sample to get expanded state sequence for all expanded rows
    exp_expanded_states = BackwardSampling_expanded(fbmats_ph[subj], n_expanded)
    
    # Sample initial phase if panel data
    if isnothing(init_phase_for_path)
        p0_given_obs = normalize(vec(sum(fbmats_ph[subj][1, :, :], dims=2)), 1)
        init_phase_for_path = rand(Categorical(p0_given_obs))
    end
    
    # Build mapping from original rows to their corresponding transition rows in expanded data
    # For each original exact observation, find the expanded row that has the transition (obstype=1)
    # Non-exact observations map 1:1 (they weren't expanded)
    orig_to_exp_phase = Vector{Int}(undef, n_orig_obs)
    
    exp_offset = 0
    for i in 1:n_orig_obs
        if orig_subj_dat.obstype[i] == 1
            # Exact observation: maps to the second expanded row (the transition row)
            # The first expanded row is the sojourn, second is the transition
            orig_to_exp_phase[i] = exp_expanded_states[exp_offset + 2]
            exp_offset += 2
        else
            # Non-exact: maps directly (wasn't expanded)
            exp_offset += 1
            orig_to_exp_phase[i] = exp_expanded_states[exp_offset]
        end
    end
    
    # Now construct the path using original observation structure but sampled phases
    times_expanded = [orig_subj_dat.tstart[1]]
    states_expanded = [init_phase_for_path]
    sizehint!(times_expanded, n_expanded * 2)
    sizehint!(states_expanded, n_expanded * 2)
    
    # For exact observations, we also need the phase at the end of sojourn
    # to sample the path during the sojourn interval
    exp_idx = 0
    for i in 1:n_orig_obs
        # Source state in expanded space
        a_expanded = states_expanded[end]
        
        if orig_subj_dat.obstype[i] == 1
            # Exact observation: we need to sample path during sojourn then record transition
            
            # Phase at end of sojourn (from first expanded row of this pair)
            sojourn_end_phase = exp_expanded_states[exp_idx + 1]
            # Phase after transition (from second expanded row)
            transition_phase = exp_expanded_states[exp_idx + 2]
            
            # Get TPM for sojourn interval [tstart, tstop - ε]
            sojourn_covar_idx = exp_subj_tpm_map[exp_idx + 1, 1]
            sojourn_time_idx = exp_subj_tpm_map[exp_idx + 1, 2]
            P_sojourn = tpm_book_ph[sojourn_covar_idx][sojourn_time_idx]
            Q_expanded = hazmat_book_ph[sojourn_covar_idx]
            
            # Sample path during sojourn (from current phase to sojourn_end_phase)
            sojourn_tstart = orig_subj_dat.tstart[i]
            sojourn_tstop = exp_subj_dat.tstop[exp_idx + 1]  # = orig tstop - epsilon
            
            sample_ecctmc!(times_expanded, states_expanded, P_sojourn, Q_expanded,
                          a_expanded, sojourn_end_phase, sojourn_tstart, sojourn_tstop)
            
            # Ensure we end at sojourn_end_phase
            if states_expanded[end] != sojourn_end_phase
                push!(times_expanded, sojourn_tstop)
                push!(states_expanded, sojourn_end_phase)
            end
            
            # Record the transition at the original observation time
            if surrogate.phase_to_state[sojourn_end_phase] != surrogate.phase_to_state[transition_phase]
                push!(times_expanded, orig_subj_dat.tstop[i])
                push!(states_expanded, transition_phase)
            end
            
            exp_idx += 2
        else
            # Censored/panel observation: sample path as before
            exp_idx += 1
            b_expanded = exp_expanded_states[exp_idx]
            
            covar_idx = exp_subj_tpm_map[exp_idx, 1]
            time_idx = exp_subj_tpm_map[exp_idx, 2]
            P_expanded = tpm_book_ph[covar_idx][time_idx]
            Q_expanded = hazmat_book_ph[covar_idx]
            
            sample_ecctmc!(times_expanded, states_expanded, P_expanded, Q_expanded,
                          a_expanded, b_expanded, orig_subj_dat.tstart[i], orig_subj_dat.tstop[i])
            
            if states_expanded[end] != b_expanded
                push!(times_expanded, orig_subj_dat.tstop[i])
                push!(states_expanded, b_expanded)
            end
        end
    end
    
    # Create expanded SamplePath for surrogate likelihood
    expanded_path = SamplePath(subj, copy(times_expanded), copy(states_expanded))
    
    # Collapse expanded path to observed states
    collapsed_path = collapse_phasetype_path(expanded_path, surrogate, absorbingstates)
    
    return (collapsed=collapsed_path, expanded=expanded_path)
end


"""
    collapse_phasetype_path(expanded_path::SamplePath, surrogate::PhaseTypeSurrogate, absorbingstates)

Collapse a path from the expanded phase-type state space to the observed state space.

Maps each phase back to its corresponding observed state and removes consecutive 
duplicates (transitions between phases of the same state).

# Arguments
- `expanded_path`: SamplePath in the expanded phase state space
- `surrogate`: PhaseTypeSurrogate with phase_to_state mapping
- `absorbingstates`: Vector of absorbing state indices

# Returns
- `SamplePath`: Path in the observed (collapsed) state space
"""
function collapse_phasetype_path(expanded_path::SamplePath, surrogate::PhaseTypeSurrogate, absorbingstates)
    times_expanded = expanded_path.times
    states_expanded = expanded_path.states
    subj = expanded_path.subj
    
    # Map to observed states, keeping only transitions that change observed state
    times_obs = [times_expanded[1]]
    states_obs = [surrogate.phase_to_state[states_expanded[1]]]
    
    for i in 2:length(times_expanded)
        obs_state = surrogate.phase_to_state[states_expanded[i]]
        # Only record if observed state changes
        if obs_state != states_obs[end]
            push!(times_obs, times_expanded[i])
            push!(states_obs, obs_state)
        end
    end
    
    # Truncate at absorbing states
    truncind = findfirst(states_obs .∈ Ref(absorbingstates))
    if !isnothing(truncind)
        times_obs = first(times_obs, truncind)
        states_obs = first(states_obs, truncind)
    end
    
    return reduce_jumpchain(SamplePath(subj, times_obs, states_obs))
end


"""
    loglik_phasetype_expanded_path(expanded_path::SamplePath, Q::Matrix{Float64})

Compute log-density of a sample path under a CTMC with intensity matrix Q.

This computes the CTMC path density directly:
  log f(path) = Σᵢ [-qₛᵢ * Δtᵢ + log(qₛᵢ,dᵢ)]

where qₛ = -Q[s,s] is the total exit rate from state s, and q_{s,d} = Q[s,d] is the 
transition rate from s to d.

# Arguments
- `expanded_path::SamplePath`: Sample path in expanded phase space (states index into Q)
- `Q::Matrix{Float64}`: CTMC intensity matrix in expanded phase space

# Returns
- `Float64`: Log-likelihood (density) of the path

# See also
- [`loglik_phasetype_collapsed_path`](@ref): For paths in observed (collapsed) state space
"""
function loglik_phasetype_expanded_path(expanded_path::SamplePath, Q::Matrix{Float64})
    loglik = 0.0
    
    n_transitions = length(expanded_path.times) - 1
    if n_transitions == 0
        return 0.0
    end
    
    for i in 1:n_transitions
        t0 = expanded_path.times[i]
        t1 = expanded_path.times[i + 1]
        dt = t1 - t0
        
        s = expanded_path.states[i]
        d = expanded_path.states[i + 1]
        
        # Survival term: exp(-q_s * dt) where q_s is total exit rate from s
        q_s = -Q[s, s]  # Diagonal is negative total exit rate
        loglik += -q_s * dt
        
        # Transition term: log(q_{s,d}) if s != d
        if s != d
            q_sd = Q[s, d]
            if q_sd > 0
                loglik += log(q_sd)
            else
                return -Inf  # Impossible transition
            end
        end
    end
    
    return loglik
end

# Convenience method using PhaseTypeSurrogate
loglik_phasetype_expanded_path(expanded_path::SamplePath, surrogate::PhaseTypeSurrogate) = 
    loglik_phasetype_expanded_path(expanded_path, surrogate.expanded_Q)


"""
    loglik_phasetype_collapsed_path(path::SamplePath, surrogate::PhaseTypeSurrogate)

Compute log-density of a collapsed sample path under the phase-type surrogate.

For Coxian phase-type models, we always enter each state in phase 1.
The density of sojourn time τ in state s followed by exit to state d is:
  f(τ; s→d) = π' * exp(S*τ) * r
where π = (1,0,...,0)', S = sub-intensity, r = exit rates to d

This function analytically integrates over all possible phase-level paths
that could produce the observed collapsed path.

# Arguments
- `path::SamplePath`: Sample path in observed (collapsed) state space
- `surrogate::PhaseTypeSurrogate`: The phase-type surrogate with expanded Q matrix

# Returns
- `Float64`: Log-density of the collapsed path under the phase-type model

# See also
- [`loglik_phasetype_expanded_path`](@ref): For paths in expanded phase space
"""
function loglik_phasetype_collapsed_path(path::SamplePath, surrogate::PhaseTypeSurrogate)
    
    loglik = 0.0
    Q = surrogate.expanded_Q
    
    n_transitions = length(path.times) - 1
    if n_transitions == 0
        return 0.0  # No transitions, density = 1 (just conditioning on initial state)
    end
    
    # For Coxian phase-type, we always enter each state in phase 1.
    # The density of sojourn time τ in state s followed by exit to state d is:
    #   f(τ; s→d) = π' * exp(S*τ) * r
    # where π = (1,0,...,0)', S = sub-intensity, r = exit rates to d
    
    for i in 1:n_transitions
        t0 = path.times[i]
        t1 = path.times[i + 1]
        τ = t1 - t0
        
        # Validate sojourn time
        if τ < 0
            return -Inf
        end
        
        s_obs = path.states[i]      # Source observed state
        d_obs = path.states[i + 1]  # Destination observed state
        
        # Get phase indices for source and destination states
        s_phases = surrogate.state_to_phases[s_obs]
        d_phases = surrogate.state_to_phases[d_obs]
        n_phases_s = length(s_phases)
        
        if n_phases_s == 1
            # Single phase case: exponential distribution
            # f(τ) = λ_{s→d} * exp(-λ_s * τ)
            # where λ_s = total exit rate, λ_{s→d} = rate to destination d
            
            phase_idx = first(s_phases)
            
            # Exit rate to destination state d (sum over all phases of d)
            exit_rate_to_dest = sum(Q[phase_idx, dp] for dp in d_phases)
            
            # Total exit rate from this phase
            total_exit_rate = -Q[phase_idx, phase_idx]
            
            if exit_rate_to_dest <= 0 || total_exit_rate <= 0
                return -Inf  # Impossible transition
            end
            
            # Log-density: log(λ_{s→d}) - λ_s * τ
            loglik += log(exit_rate_to_dest) - total_exit_rate * τ
            
        else
            # Multi-phase Coxian case: use matrix exponential formula
            # f(τ; s→d) = π' * exp(S*τ) * r
            
            # Extract sub-intensity matrix S for phases within state s
            # S[i,j] = Q[s_phases[i], s_phases[j]]
            S_within = zeros(Float64, n_phases_s, n_phases_s)
            for (ii, pi) in enumerate(s_phases)
                for (jj, pj) in enumerate(s_phases)
                    S_within[ii, jj] = Q[pi, pj]
                end
            end
            
            # Exit rate vector to destination state d
            # r[i] = Σⱼ Q[s_phases[i], d_phases[j]]
            exit_to_dest = zeros(Float64, n_phases_s)
            for (ii, pi) in enumerate(s_phases)
                exit_to_dest[ii] = sum(Q[pi, dp] for dp in d_phases)
            end
            
            # Check if transition is possible
            if all(exit_to_dest .<= 0)
                return -Inf  # No exit rate to destination
            end
            
            # Initial distribution: Coxian always starts in phase 1
            # π = (1, 0, ..., 0)'
            π_s = zeros(Float64, n_phases_s)
            π_s[1] = 1.0
            
            # Compute matrix exponential: exp(S * τ)
            # Note: S * τ, NOT S .* τ (element-wise would be wrong!)
            expSτ = exp(S_within * τ)  # LinearAlgebra.exp for matrix exponential
            
            # Density: π' * exp(S*τ) * r
            # = (first row of exp(S*τ)) ⋅ r  (since π has only first element = 1)
            density = dot(expSτ[1, :], exit_to_dest)
            
            if density <= 0
                return -Inf
            end
            
            loglik += log(density)
        end
    end
    
    return loglik
end