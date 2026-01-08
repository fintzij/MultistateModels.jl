# =============================================================================
# Fused Batched Likelihood Computation
# =============================================================================
#
# This section provides an optimized batched likelihood implementation that:
# 1. Avoids DataFrame allocation by computing intervals directly from SamplePath
# 2. Caches covariate lookups per subject (not per path)
# 3. Uses columnar storage for better cache locality
# 4. Pre-allocates all working memory for reuse across iterations
#
# Note: LightweightInterval and SubjectCovarCache structs are defined in common.jl
#
# =============================================================================

"""
    build_subject_covar_cache(model::MultistateProcess)

Build a cache of covariate data per subject.
This is called once per model and reused across all likelihood evaluations.
"""
function build_subject_covar_cache(model::MultistateProcess)
    n_subjects = length(model.subjectindices)
    # Convert names to Symbol for proper comparison (names() returns Vector{String})
    covar_cols = setdiff(Symbol.(names(model.data)), [:id, :tstart, :tstop, :statefrom, :stateto, :obstype])
    has_covars = !isempty(covar_cols)
    
    caches = Vector{SubjectCovarCache}(undef, n_subjects)
    
    for subj in 1:n_subjects
        subj_inds = model.subjectindices[subj]
        subj_data = view(model.data, subj_inds, :)
        
        if has_covars
            covar_data = subj_data[:, covar_cols]
            tstart = collect(subj_data.tstart)
        else
            covar_data = DataFrame()
            tstart = Float64[]
        end
        
        caches[subj] = SubjectCovarCache(tstart, covar_data)
    end
    
    return caches
end

"""
    compute_intervals_from_path(path::SamplePath, subject_covar::SubjectCovarCache)

Compute likelihood intervals directly from a SamplePath without creating a DataFrame.
Returns a vector of LightweightIntervals.

This implements the same logic as `make_subjdat` but avoids DataFrame allocation.

For repeated calls (e.g., MCEM), uses thread-local workspace to reduce allocations.
"""
function compute_intervals_from_path(path::SamplePath, subject_covar::SubjectCovarCache)
    # Get thread-local workspace for TVC case
    ws = get_tvc_workspace()
    return compute_intervals_from_path!(ws, path, subject_covar)
end

"""
    compute_intervals_from_path!(ws::TVCIntervalWorkspace, path::SamplePath, subject_covar::SubjectCovarCache)

Workspace-based version that minimizes allocations for repeated calls.
"""
function compute_intervals_from_path!(ws::TVCIntervalWorkspace, path::SamplePath, subject_covar::SubjectCovarCache)
    n_transitions = length(path.times) - 1
    
    if isempty(subject_covar.covar_data) || nrow(subject_covar.covar_data) <= 1
        # No time-varying covariates - use path times directly
        # Still need to allocate result, but workspace not needed
        intervals = Vector{LightweightInterval}(undef, n_transitions)
        
        sojourn = 0.0
        for i in 1:n_transitions
            increment = path.times[i+1] - path.times[i]
            intervals[i] = LightweightInterval(
                sojourn,
                sojourn + increment,
                path.states[i],
                path.states[i+1],
                1  # Single covariate row
            )
            
            # Reset sojourn if state changes (semi-Markov clock reset)
            if path.states[i] != path.states[i+1]
                sojourn = 0.0
            else
                sojourn += increment
            end
        end
        
        return intervals
    else
        # Time-varying covariates - use workspace to reduce allocations
        tstart = subject_covar.tstart
        covar_data = subject_covar.covar_data
        
        # Identify covariate change times using workspace
        n_change = 1
        @inbounds ws.change_times[1] = tstart[1]
        for i in 2:length(tstart)
            if !isequal(covar_data[i-1, :], covar_data[i, :])
                n_change += 1
                if n_change > length(ws.change_times)
                    resize!(ws.change_times, 2 * n_change)
                end
                ws.change_times[n_change] = tstart[i]
            end
        end
        
        # Merge path times with covariate change times into utimes
        # Collect all unique times in range
        n_utimes = 0
        path_start = path.times[1]
        path_end = path.times[end]
        
        # Add path times
        for t in path.times
            n_utimes += 1
            if n_utimes > length(ws.utimes)
                resize!(ws.utimes, 2 * n_utimes)
            end
            @inbounds ws.utimes[n_utimes] = t
        end
        
        # Add change times within range
        for i in 1:n_change
            t = ws.change_times[i]
            if path_start <= t <= path_end
                n_utimes += 1
                if n_utimes > length(ws.utimes)
                    resize!(ws.utimes, 2 * n_utimes)
                end
                @inbounds ws.utimes[n_utimes] = t
            end
        end
        
        # Sort and remove duplicates (in-place)
        sort!(@view(ws.utimes[1:n_utimes]))
        
        # Unique in-place
        j = 1
        @inbounds for i in 2:n_utimes
            if ws.utimes[i] != ws.utimes[j]
                j += 1
                ws.utimes[j] = ws.utimes[i]
            end
        end
        n_utimes = j
        
        n_intervals = n_utimes - 1
        
        # Ensure workspace capacity
        if n_intervals > length(ws.intervals)
            resize!(ws.intervals, max(n_intervals, 2 * length(ws.intervals)))
            resize!(ws.sojourns, length(ws.intervals))
            resize!(ws.pathinds, length(ws.intervals) + 1)
            resize!(ws.datinds, length(ws.intervals) + 1)
        end
        
        # Compute path and data indices
        @inbounds for i in 1:n_utimes
            ws.pathinds[i] = searchsortedlast(path.times, ws.utimes[i])
            ws.datinds[i] = searchsortedlast(tstart, ws.utimes[i])
        end
        
        # Compute sojourns
        current_sojourn = 0.0
        current_pathind = ws.pathinds[1]
        
        @inbounds for i in 1:n_intervals
            increment = ws.utimes[i+1] - ws.utimes[i]
            
            if ws.pathinds[i] != current_pathind
                current_sojourn = 0.0
                current_pathind = ws.pathinds[i]
            end
            
            ws.sojourns[i] = current_sojourn
            current_sojourn += increment
        end
        
        # Build intervals (need to allocate result vector)
        intervals = Vector{LightweightInterval}(undef, n_intervals)
        @inbounds for i in 1:n_intervals
            increment = ws.utimes[i+1] - ws.utimes[i]
            intervals[i] = LightweightInterval(
                ws.sojourns[i],
                ws.sojourns[i] + increment,
                path.states[ws.pathinds[i]],
                path.states[ws.pathinds[i+1]],
                ws.datinds[i]
            )
        end
        
        return intervals
    end
end

"""
    extract_covariates_lightweight(subject_covar::SubjectCovarCache, row_idx::Int, covar_names::Vector{Symbol})

Extract covariates from the subject cache without DataFrame row access overhead.

Handles interaction terms (e.g., `Symbol("trt & age")`) by computing the product
of their component values.
"""
@inline function extract_covariates_lightweight(subject_covar::SubjectCovarCache, row_idx::Int, covar_names::Vector{Symbol})
    isempty(covar_names) && return NamedTuple()
    
    # Clamp row_idx to valid range
    idx = clamp(row_idx, 1, max(1, nrow(subject_covar.covar_data)))
    
    if isempty(subject_covar.covar_data)
        return NamedTuple()
    end
    
    # Extract values for requested covariates, handling interaction terms
    values = Tuple(_lookup_covar_value_lightweight(subject_covar.covar_data, idx, cname) for cname in covar_names)
    return NamedTuple{Tuple(covar_names)}(values)
end

"""
    _lookup_covar_value_lightweight(covar_data::DataFrame, row_idx::Int, cname::Symbol)

Look up a covariate value, handling interaction terms (e.g., `:"trt & age"`).

For interaction terms, computes the product of the component values.
"""
@inline function _lookup_covar_value_lightweight(covar_data::DataFrame, row_idx::Int, cname::Symbol)
    # Direct column access for simple covariates
    if hasproperty(covar_data, cname)
        return covar_data[row_idx, cname]
    end
    
    # Handle interaction terms: "trt & age" or "trt:age"
    cname_str = String(cname)
    if occursin("&", cname_str)
        parts = split(cname_str, "&")
        return prod(_lookup_covar_value_lightweight(covar_data, row_idx, Symbol(strip(part))) for part in parts)
    elseif occursin(":", cname_str)
        parts = split(cname_str, ":")
        return prod(_lookup_covar_value_lightweight(covar_data, row_idx, Symbol(strip(part))) for part in parts)
    else
        throw(ArgumentError("column name :$cname not found in covariate data"))
    end
end

# =============================================================================
# Penalized Log-Likelihood Wrapper
# =============================================================================

"""
    loglik_exact_penalized(parameters, data::ExactData, penalty_config::PenaltyConfig;
                           neg=true, parallel=false) -> Real

Compute **penalized** negative log-likelihood for exact data.

The penalized objective combines the data log-likelihood with a roughness penalty:
    -ℓ_p(β) = -ℓ(β) + (1/2) Σⱼ λⱼ βⱼᵀ Sⱼ βⱼ

# Arguments
- `parameters`: Flat parameter vector on estimation scale (log-transformed baseline)
- `data::ExactData`: Exact data containing model and sample paths
- `penalty_config::PenaltyConfig`: Resolved penalty configuration
- `neg::Bool=true`: Return negative penalized log-likelihood
- `parallel::Bool=false`: Use multi-threaded parallel computation for base likelihood

# Returns
Scalar (penalized) negative log-likelihood when `neg=true`

# Notes
- Penalty is computed on **natural scale** coefficients (exp-transformed baseline)
- This function is AD-compatible (works with ForwardDiff.Dual)
- For likelihood-only computation (no penalty), use `loglik_exact` directly

# Example
```julia
data = ExactData(model, paths)
config = build_penalty_config(model, SplinePenalty())

# Penalized negative log-likelihood
nll = loglik_exact_penalized(params, data, config)

# Use in optimization
optf = OptimizationFunction((p, d) -> loglik_exact_penalized(p, d, config), AutoForwardDiff())
```

See also: [`loglik_exact`](@ref), [`build_penalty_config`](@ref), [`compute_penalty`](@ref)
"""
function loglik_exact_penalized(parameters, data::ExactData, penalty_config::PenaltyConfig;
                                 neg::Bool=true, parallel::Bool=false)
    # Compute base log-likelihood (always as negative for consistency)
    nll_base = loglik_exact(parameters, data; neg=true, return_ll_subj=false, parallel=parallel)
    
    # If no penalties, return base likelihood
    has_penalties(penalty_config) || return neg ? nll_base : -nll_base
    
    # Compute penalty on parameters.
    # As of v0.3.0, spline parameters are stored on NATURAL scale (β, not log(β)).
    # The penalty is quadratic: P(β) = λ/2 β'Sβ, where S penalizes roughness.
    # This enables correct Newton approximation in PIJCV λ selection.
    # Box constraints (β ≥ POSITIVE_LB) enforce positivity.
    penalty = compute_penalty(parameters, penalty_config)
    
    # Return penalized negative log-likelihood
    nll_penalized = nll_base + penalty
    return neg ? nll_penalized : -nll_penalized
end

"""
    loglik_exact(parameters, data::ExactData; neg=true, return_ll_subj=false, parallel=false)

Compute log-likelihood for exact (fully observed) multistate data.

# Arguments
- `parameters`: Flat parameter vector
- `data::ExactData`: Exact data containing model and sample paths
- `neg::Bool=true`: Return negative log-likelihood
- `return_ll_subj::Bool=false`: Return per-path weighted log-likelihoods instead of scalar
- `parallel::Bool=false`: Use multi-threaded parallel computation

# Parallel Execution
When `parallel=true` and `Threads.nthreads() > 1`, uses `@threads :static` for 
path-level parallelism. This is beneficial when:
- Number of paths > 100
- Per-path computation cost > 10μs

Note: Parallel mode is NOT used during AD gradient computation (ForwardDiff uses
the sequential path). Use parallel for objective evaluation during line search.

# Returns
- If `return_ll_subj=false`: Scalar (negative) log-likelihood
- If `return_ll_subj=true`: Vector of per-path weighted log-likelihoods
"""
function loglik_exact(parameters, data::ExactData; neg=true, return_ll_subj=false, parallel=false)
    # Unflatten parameters to natural scale - preserves dual number types (AD-compatible)
    pars = unflatten_parameters(parameters, data.model)
    
    # Get model components
    hazards = data.model.hazards
    totalhazards = data.model.totalhazards
    tmat = data.model.tmat
    n_hazards = length(hazards)
    n_paths = length(data.paths)
    
    # Build subject covariate cache (this is type-stable, no parameters involved)
    subject_covars = build_subject_covar_cache(data.model)
    
    # Element type for AD compatibility (Float64 or Dual)
    T = eltype(parameters)
    
    # Get covariate names for each hazard (precomputed, doesn't depend on parameters)
    covar_names_per_hazard = [
        hasfield(typeof(hazards[h]), :covar_names) ? 
            hazards[h].covar_names : 
            extract_covar_names(hazards[h].parnames)
        for h in 1:n_hazards
    ]
    
    # Check if any hazard uses time transform
    any_time_transform = any(h -> h.metadata.time_transform, hazards)
    
    # Subject weights (precomputed lookup for efficiency)
    subj_weights = data.model.SubjectWeights
    
    # Parallel vs sequential execution
    use_parallel = parallel && Threads.nthreads() > 1 && n_paths >= 10
    
    if use_parallel
        # Parallel path: pre-allocate and use @threads :static
        ll_array = Vector{T}(undef, n_paths)
        
        Threads.@threads :static for path_idx in 1:n_paths
            path = data.paths[path_idx]
            
            # Thread-local TimeTransformContext (avoid cache sharing)
            tt_context = if any_time_transform
                sample_subj = subject_covars[path.subj]
                sample_df = isempty(sample_subj.covar_data) ? nothing : sample_subj.covar_data[1:1, :]
                maybe_time_transform_context(pars, sample_df, hazards)
            else
                nothing
            end
            
            ll_array[path_idx] = _compute_path_loglik_fused(
                path, pars, hazards, totalhazards, tmat,
                subject_covars[path.subj], covar_names_per_hazard,
                tt_context, T
            )
        end
    else
        # Sequential path: functional style for reverse-mode AD compatibility
        # Create TimeTransformContext once (shared across all paths)
        tt_context = if any_time_transform && !isempty(data.paths)
            sample_subj = subject_covars[data.paths[1].subj]
            sample_df = isempty(sample_subj.covar_data) ? nothing : sample_subj.covar_data[1:1, :]
            maybe_time_transform_context(pars, sample_df, hazards)
        else
            nothing
        end
        
        ll_paths = map(enumerate(data.paths)) do (path_idx, path)
            _compute_path_loglik_fused(
                path, pars, hazards, totalhazards, tmat, 
                subject_covars[path.subj], covar_names_per_hazard,
                tt_context, T
            )
        end
        ll_array = collect(T, ll_paths)
    end
    
    if return_ll_subj
        # Element-wise multiplication preserves AD types
        return ll_array .* [subj_weights[path.subj] for path in data.paths]
    else
        # Weighted sum
        ll = sum(ll_array[i] * subj_weights[data.paths[i].subj] for i in eachindex(data.paths))
        return neg ? -ll : ll
    end
end

"""
    _compute_path_loglik_fused(path, pars, hazards, totalhazards, tmat, 
                                subj_cache, covar_names_per_hazard, tt_context, T)

Compute log-likelihood for a single path. Extracted for functional style (reverse-mode AD).

This function is the core likelihood computation shared by `loglik_exact` and `loglik_semi_markov`.
It uses a path-centric approach that iterates over sojourn intervals in a sample path, 
accumulating log-survival contributions (via `eval_cumhaz`) and transition hazard 
contributions (via `eval_hazard`).

See also: `loglik_exact`, `loglik_semi_markov`, `eval_cumhaz`
"""
function _compute_path_loglik_fused(
    path::SamplePath, 
    pars,  # Parameters as nested structure (from unflatten or Tuple)
    hazards::Vector{<:_Hazard},
    totalhazards::Vector{<:_TotalHazard}, 
    tmat::Matrix{Int64},
    subj_cache::SubjectCovarCache, 
    covar_names_per_hazard::Vector{Vector{Symbol}},
    tt_context,
    ::Type{T}
) where T
    
    n_hazards = length(hazards)
    n_transitions = length(path.times) - 1
    
    # Initialize log-likelihood for this path
    ll = zero(T)
    
    # Track effective times for AFT models
    # We need one accumulator per hazard
    effective_times = zeros(T, n_hazards)
    
    # Pre-extract hazard parameters by index to avoid repeated NamedTuple lookups
    # This converts dynamic symbol lookup to static tuple indexing
    pars_indexed = values(pars)  # Convert NamedTuple to Tuple for indexed access
    
    # Check if we have time-varying covariates
    has_tvc = !isempty(subj_cache.covar_data) && nrow(subj_cache.covar_data) > 1
    
    if !has_tvc
        # Fast path: no time-varying covariates
        sojourn = 0.0
        
        for i in 1:n_transitions
            increment = path.times[i+1] - path.times[i]
            lb = sojourn
            ub = sojourn + increment
            statefrom = path.states[i]
            stateto = path.states[i+1]
            
            # Reset effective times if clock reset (start of sojourn)
            if lb == 0.0
                fill!(effective_times, zero(T))
            end
            
            # Get total hazard for origin state
            tothaz = totalhazards[statefrom]
            
            if tothaz isa _TotalHazardTransient
                # Determine if time transform is enabled for this state
                use_transform = _time_transform_enabled(tothaz, hazards)
                
                # Accumulate cumulative hazards for all exit hazards
                for h in tothaz.components
                    hazard = hazards[h]
                    hazard_pars = pars_indexed[h]  # Fast indexed access
                    
                    # Extract covariates
                    covars = extract_covariates_lightweight(subj_cache, 1, covar_names_per_hazard[h])
                    
                    # Check for AFT
                    is_aft = hazard.metadata.linpred_effect == :aft
                    
                    # Calculate effective time increment
                    # For PH, scale is 1.0, so effective time = clock time
                    # For AFT, scale = exp(-linpred)
                    # We compute linpred here to get the scale
                    # Note: eval_cumhaz will recompute linpred, but that's unavoidable unless we refactor deeper
                    # Optimization: if not AFT, we don't strictly need effective_times logic, 
                    # but keeping it uniform is cleaner. However, for PH, we can just use lb/ub.
                    
                    if is_aft
                        linpred = _linear_predictor(hazard_pars, covars, hazard)
                        scale = exp(-linpred)
                        delta_tau = (ub - lb) * scale
                        
                        tau_start = effective_times[h]
                        tau_end = tau_start + delta_tau
                        effective_times[h] = tau_end
                        
                        # Cumulative hazard using effective time
                        cumhaz = eval_cumhaz(
                            hazard, tau_start, tau_end, hazard_pars, covars;
                            apply_transform = hazard.metadata.time_transform,
                            cache_context = tt_context,
                            hazard_slot = h,
                            use_effective_time = true)
                    else
                        # Standard PH evaluation (effective time = clock time)
                        cumhaz = eval_cumhaz(
                            hazard, lb, ub, hazard_pars, covars;
                            apply_transform = hazard.metadata.time_transform,
                            cache_context = tt_context,
                            hazard_slot = h,
                            use_effective_time = false)
                    end
                    
                    ll -= cumhaz
                end
                
                # Add transition hazard if transition occurred
                if statefrom != stateto
                    trans_h = tmat[statefrom, stateto]
                    hazard = hazards[trans_h]
                    hazard_pars = pars_indexed[trans_h]  # Fast indexed access
                    covars = extract_covariates_lightweight(subj_cache, 1, covar_names_per_hazard[trans_h])
                    
                    is_aft = hazard.metadata.linpred_effect == :aft
                    
                    if is_aft
                        # Use the effective time we just calculated
                        tau_end = effective_times[trans_h]
                        
                        haz_value = eval_hazard(
                            hazard, tau_end, hazard_pars, covars;
                            apply_transform = hazard.metadata.time_transform,
                            cache_context = tt_context,
                            hazard_slot = trans_h,
                            use_effective_time = true)
                    else
                        haz_value = eval_hazard(
                            hazard, ub, hazard_pars, covars;
                            apply_transform = hazard.metadata.time_transform,
                            cache_context = tt_context,
                            hazard_slot = trans_h,
                            use_effective_time = false)
                    end
                    
                    ll += NaNMath.log(haz_value)
                end
            end
            
            # Update sojourn (reset on state change)
            sojourn = (statefrom != stateto) ? 0.0 : ub
        end
    else
        # Slow path: time-varying covariates (use full interval computation)
        intervals = compute_intervals_from_path(path, subj_cache)
        
        for interval in intervals
            # Reset effective times if clock reset
            if interval.lb == 0.0
                fill!(effective_times, zero(T))
            end
            
            tothaz = totalhazards[interval.statefrom]
            
            if tothaz isa _TotalHazardTransient
                for h in tothaz.components
                    hazard = hazards[h]
                    hazard_pars = pars_indexed[h]  # Fast indexed access
                    covars = extract_covariates_lightweight(subj_cache, interval.covar_row_idx, covar_names_per_hazard[h])
                    
                    is_aft = hazard.metadata.linpred_effect == :aft
                    
                    if is_aft
                        linpred = _linear_predictor(hazard_pars, covars, hazard)
                        scale = exp(-linpred)
                        delta_tau = (interval.ub - interval.lb) * scale
                        
                        tau_start = effective_times[h]
                        tau_end = tau_start + delta_tau
                        effective_times[h] = tau_end
                        
                        cumhaz = eval_cumhaz(
                            hazard, tau_start, tau_end, hazard_pars, covars;
                            apply_transform = hazard.metadata.time_transform,
                            cache_context = tt_context,
                            hazard_slot = h,
                            use_effective_time = true)
                    else
                        cumhaz = eval_cumhaz(
                            hazard, interval.lb, interval.ub, hazard_pars, covars;
                            apply_transform = hazard.metadata.time_transform,
                            cache_context = tt_context,
                            hazard_slot = h,
                            use_effective_time = false)
                    end
                    
                    ll -= cumhaz
                end
                
                if interval.statefrom != interval.stateto
                    trans_h = tmat[interval.statefrom, interval.stateto]
                    hazard = hazards[trans_h]
                    hazard_pars = pars_indexed[trans_h]  # Fast indexed access
                    covars = extract_covariates_lightweight(subj_cache, interval.covar_row_idx, covar_names_per_hazard[trans_h])
                    
                    is_aft = hazard.metadata.linpred_effect == :aft
                    
                    if is_aft
                        tau_end = effective_times[trans_h]
                        
                        haz_value = eval_hazard(
                            hazard, tau_end, hazard_pars, covars;
                            apply_transform = hazard.metadata.time_transform,
                            cache_context = tt_context,
                            hazard_slot = trans_h,
                            use_effective_time = true)
                    else
                        haz_value = eval_hazard(
                            hazard, interval.ub, hazard_pars, covars;
                            apply_transform = hazard.metadata.time_transform,
                            cache_context = tt_context,
                            hazard_slot = trans_h,
                            use_effective_time = false)
                    end
                    
                    ll += NaNMath.log(haz_value)
                end
            end
        end
    end
    
    return ll
end
# =============================================================================
# Single-Subject Likelihood Evaluation
# =============================================================================

"""
    loglik_subject(parameters, data::ExactData, subject_idx::Int) -> Real

Compute the log-likelihood contribution for a single subject at given parameters.

This function is used by NCV (Neighbourhood Cross-Validation) to evaluate the 
actual loss at leave-one-out parameters, following Wood (2024) "On Neighbourhood 
Cross Validation" arXiv:2404.16490v4.

# Arguments
- `parameters`: Parameter vector (on estimation scale, e.g., log-transformed baseline)
- `data::ExactData`: Exact data container with model and paths
- `subject_idx::Int`: Index of the subject (1-based)

# Returns
- `Real`: Subject's log-likelihood contribution (NOT negated)

# Mathematical Background

The NCV criterion (Wood 2024, Equation 2) is:
```math
V(\\lambda) = \\sum_{i=1}^{n} D_i(\\hat\\beta^{-i})
```
where ``D_i(\\beta) = -\\ell_i(\\beta)`` is subject ``i``'s negative log-likelihood
contribution and ``\\hat\\beta^{-i}`` is the penalized MLE with subject ``i`` omitted.

This function computes ``\\ell_i(\\beta)`` at arbitrary parameter values, enabling
evaluation of the NCV criterion by calling the actual likelihood function at the
Newton-approximated leave-one-out parameters.

# Notes
- Does NOT include the penalty term (that's only for total likelihood)
- Works at arbitrary parameter values (not just MLE)
- Returns log-likelihood (positive when likelihood is high), not negative log-likelihood
- Subject weights from the model are applied

# Example
```julia
data = ExactData(model, paths)
params = get_parameters(fitted_model)

# Evaluate subject 5's log-likelihood at current parameters
ll_5 = loglik_subject(params, data, 5)

# Evaluate at perturbed parameters (e.g., LOO estimate)
params_loo = params + delta
ll_5_loo = loglik_subject(params_loo, data, 5)
```

See also: [`loglik_exact`](@ref), [`compute_pijcv_criterion`](@ref)
"""
function loglik_subject(parameters, data::ExactData, subject_idx::Int)
    model = data.model
    
    # Validate subject index
    @assert 1 <= subject_idx <= length(model.subjectindices) "Subject index $subject_idx out of range [1, $(length(model.subjectindices))]"
    
    # Get the path for this subject
    # Note: For exact data, there's one path per subject, indexed by subject_idx
    path = data.paths[subject_idx]
    
    # Get subject weight
    w = model.SubjectWeights[path.subj]
    
    # Unflatten parameters to natural scale (preserves AD types)
    pars = unflatten_parameters(parameters, model)
    
    # Get model components
    hazards = model.hazards
    totalhazards = model.totalhazards
    tmat = model.tmat
    n_hazards = length(hazards)
    
    # Build subject covariate cache for this subject only
    # (This is a slight inefficiency - could cache across calls, but keeps function pure)
    subj_inds = model.subjectindices[path.subj]
    subj_data = view(model.data, subj_inds, :)
    
    covar_cols = setdiff(Symbol.(names(model.data)), [:id, :tstart, :tstop, :statefrom, :stateto, :obstype])
    has_covars = !isempty(covar_cols)
    
    if has_covars
        covar_data = subj_data[:, covar_cols]
        tstart = collect(subj_data.tstart)
    else
        covar_data = DataFrame()
        tstart = Float64[]
    end
    
    subject_covar = SubjectCovarCache(tstart, covar_data)
    
    # Element type for AD compatibility
    T = eltype(parameters)
    
    # Get covariate names for each hazard
    covar_names_per_hazard = [
        hasfield(typeof(hazards[h]), :covar_names) ? 
            hazards[h].covar_names : 
            extract_covar_names(hazards[h].parnames)
        for h in 1:n_hazards
    ]
    
    # Check if any hazard uses time transform
    any_time_transform = any(h -> h.metadata.time_transform, hazards)
    
    # Create TimeTransformContext if needed
    tt_context = if any_time_transform
        sample_df = isempty(subject_covar.covar_data) ? nothing : subject_covar.covar_data[1:1, :]
        maybe_time_transform_context(pars, sample_df, hazards)
    else
        nothing
    end
    
    # Compute single-path log-likelihood using the fused implementation
    ll = _compute_path_loglik_fused(
        path, pars, hazards, totalhazards, tmat,
        subject_covar, covar_names_per_hazard,
        tt_context, T
    )
    
    # Apply subject weight and return
    return ll * w
end