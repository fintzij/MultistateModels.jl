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
    covar_cols = setdiff(names(model.data), [:id, :tstart, :tstop, :statefrom, :stateto, :obstype])
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
    
    # Extract natural-scale baseline coefficients for penalty
    # The penalty applies to exp(β) where β are the baseline spline coefficients
    pars_natural = unflatten_natural(parameters, data.model)
    
    # Build flat natural-scale vector for penalty computation
    # Only baseline parameters need transformation; coefficients are unchanged
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
    pars = unflatten_natural(parameters, data.model)
    
    # Get model components
    hazards = data.model.hazards
    totalhazards = data.model.totalhazards
    tmat = data.model.tmat
    n_hazards = length(hazards)
    n_paths = length(data.paths)
    
    # Remake spline parameters if needed
    # Note: For RuntimeSplineHazard (the current implementation), remake_splines! is a no-op
    # since splines are constructed on-the-fly during evaluation. We still call it for 
    # future-proofing if other spline implementations need parameter updates.
    for i in eachindex(hazards)
        if isa(hazards[i], _SplineHazard)
            # RuntimeSplineHazard.remake_splines! is a no-op, but call for extensibility
            remake_splines!(hazards[i], nothing)
            set_riskperiod!(hazards[i])
        end
    end
    
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

# Neural ODE Extension Point
The `eval_cumhaz` invocations are the extension points for neural ODE-based hazards.
When `is_separable(hazard) == false` for an ODE-based hazard, `eval_cumhaz` should be 
extended to invoke a numerical ODE solver (e.g., DifferentialEquations.jl) to compute 
the cumulative hazard as an integral: Λ(t₀, t₁) = ∫_{t₀}^{t₁} λ(s) ds.

For reverse-mode AD compatibility with neural ODEs:
- Use SciMLSensitivity.jl adjoints (BacksolveAdjoint, QuadratureAdjoint)
- Ensure the hazard's metadata declares supported adjoint methods
- The functional accumulation style in this function avoids in-place mutation 
  required for Zygote/Enzyme compatibility

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
                    
                    # Cumulative hazard (use hazard-specific transform flag)
                    cumhaz = eval_cumhaz(
                        hazard, lb, ub, hazard_pars, covars;
                        apply_transform = hazard.metadata.time_transform,
                        cache_context = tt_context,
                        hazard_slot = h)
                    
                    ll -= cumhaz
                end
                
                # Add transition hazard if transition occurred
                if statefrom != stateto
                    trans_h = tmat[statefrom, stateto]
                    hazard = hazards[trans_h]
                    hazard_pars = pars_indexed[trans_h]  # Fast indexed access
                    covars = extract_covariates_lightweight(subj_cache, 1, covar_names_per_hazard[trans_h])
                    
                    haz_value = eval_hazard(
                        hazard, ub, hazard_pars, covars;
                        apply_transform = hazard.metadata.time_transform,
                        cache_context = tt_context,
                        hazard_slot = trans_h)
                    
                    ll += log(haz_value)
                end
            end
            
            # Update sojourn (reset on state change)
            sojourn = (statefrom != stateto) ? 0.0 : ub
        end
    else
        # Slow path: time-varying covariates (use full interval computation)
        intervals = compute_intervals_from_path(path, subj_cache)
        
        for interval in intervals
            tothaz = totalhazards[interval.statefrom]
            
            if tothaz isa _TotalHazardTransient
                for h in tothaz.components
                    hazard = hazards[h]
                    hazard_pars = pars_indexed[h]  # Fast indexed access
                    covars = extract_covariates_lightweight(subj_cache, interval.covar_row_idx, covar_names_per_hazard[h])
                    
                    cumhaz = eval_cumhaz(
                        hazard, interval.lb, interval.ub, hazard_pars, covars;
                        apply_transform = hazard.metadata.time_transform,
                        cache_context = tt_context,
                        hazard_slot = h)
                    
                    ll -= cumhaz
                end
                
                if interval.statefrom != interval.stateto
                    trans_h = tmat[interval.statefrom, interval.stateto]
                    hazard = hazards[trans_h]
                    hazard_pars = pars_indexed[trans_h]  # Fast indexed access
                    covars = extract_covariates_lightweight(subj_cache, interval.covar_row_idx, covar_names_per_hazard[trans_h])
                    
                    haz_value = eval_hazard(
                        hazard, interval.ub, hazard_pars, covars;
                        apply_transform = hazard.metadata.time_transform,
                        cache_context = tt_context,
                        hazard_slot = trans_h)
                    
                    ll += log(haz_value)
                end
            end
        end
    end
    
    return ll
end
