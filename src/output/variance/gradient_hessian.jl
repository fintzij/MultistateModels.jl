"""
    Cross-validation and robust covariance estimation for multistate models.

This module provides tools for computing robust variance estimates and leave-one-out
cross-validation diagnostics for fitted multistate models. The key quantities computed are:

## Variance Estimators

1. **Infinitesimal Jackknife (IJ) / Sandwich Variance**: `Var(θ̂) ≈ H⁻¹ K H⁻¹`
   - H is the Hessian (observed Fisher information)
   - K = Σᵢ gᵢgᵢᵀ is the sum of outer products of subject-level score vectors
   - Robust to model misspecification; standard errors remain valid even if the model is wrong

2. **Jackknife Variance**: `Var(θ̂) ≈ ((n-1)/n) Σᵢ ΔᵢΔᵢᵀ`
   - Δᵢ = H⁻¹gᵢ are the leave-one-out parameter perturbations
   - Closely related to IJ but with different finite-sample properties

## Leave-One-Out Perturbations

The key approximation is: `θ̂₋ᵢ ≈ θ̂ + H⁻¹gᵢ = θ̂ + Δᵢ`

where θ̂₋ᵢ is the estimate obtained by leaving out subject i. This avoids
refitting the model n times.

Two methods are provided:
- **Direct solve**: Compute H⁻¹ once, then multiply by each gᵢ. O(p²n) complexity.
- **Cholesky downdate**: Use rank-1 updates to the Cholesky factor. More expensive
  due to eigendecomposition of subject Hessians, but can be more stable.

## References

- Efron, B. (1982). The Jackknife, the Bootstrap and Other Resampling Plans. SIAM.
- Huber, P. J. (1967). The behavior of maximum likelihood estimates under nonstandard conditions.
- White, H. (1982). Maximum likelihood estimation of misspecified models. Econometrica.
- Wood, S. N. (2017). Generalized Additive Models: An Introduction with R (2nd ed.), Section 6.12.

## Usage

These functions are typically called internally by `fit()` with `vcov_type` specified.
Users can access results via:

```julia
fitted = fit(model, data; vcov_type=:ij)  # IJ/sandwich variance (default)
v = get_vcov(fitted)                       # Get variance-covariance matrix
grads = get_subject_gradients(fitted)      # Subject-level scores
deltas = get_loo_perturbations(fitted)     # LOO perturbations
```
"""

# --- Core gradient/hessian computation ---

"""
    compute_subject_gradients(params, model::MultistateModel, samplepaths::Vector{SamplePath})

Compute subject-level score vectors (gradients of log-likelihood) for exact (continuously observed) data.

The score vector for subject i is:
```math
gᵢ = ∇_θ ℓᵢ(θ) = ∂ log L_i(θ) / ∂θ
```

At the MLE, the sum of subject gradients is approximately zero: `Σᵢ gᵢ ≈ 0`.
However, individual gradients are non-zero and measure each subject's "pull" on the estimate.

# Arguments
- `params::AbstractVector`: parameter vector (flat, on transformed scale)
- `model::MultistateModel`: multistate model with exact observation times
- `samplepaths::Vector{SamplePath}`: observed paths for each subject

# Returns
- `Matrix{Float64}`: p × n matrix where column i contains gᵢ

# Notes
- Gradients are computed via ForwardDiff automatic differentiation
- Subject weights from `model.SubjectWeights` are applied
- Used internally by `compute_robust_vcov()` and `fit()` when IJ/JK variance is requested
"""
function compute_subject_gradients(params::AbstractVector, model::MultistateModel, samplepaths::Vector{SamplePath})
    nsubj = length(model.subjectindices)
    nparams = length(params)
    
    # preallocate output
    grads = Matrix{Float64}(undef, nparams, nsubj)
    
    # set up data structure for single-path likelihood
    data = ExactData(model, samplepaths)
    
    # compute gradient for each subject
    for i in 1:nsubj
        path = samplepaths[i]
        w = model.SubjectWeights[i]
        
        # closure for subject i's log-likelihood
        function ll_subj_i(pars)
            pars_nested = unflatten_parameters(pars, model)
            subj_inds = model.subjectindices[path.subj]
            subj_dat = view(model.data, subj_inds, :)
            subjdat_df = make_subjdat(path, subj_dat)
            return loglik_path(pars_nested, subjdat_df, model.hazards, model.totalhazards, model.tmat) * w
        end
        
        grads[:, i] = ForwardDiff.gradient(ll_subj_i, params)
    end
    
    return grads
end

"""
    compute_subject_gradients_threaded(params, model::MultistateModel, samplepaths)

Compute subject-level score vectors (gradients) using multiple threads.

This is the parallelized version of `compute_subject_gradients` for exact data.

# Arguments
- `params::AbstractVector`: parameter vector (flat, on transformed scale)
- `model::MultistateModel`: multistate model with exact observation times
- `samplepaths::Vector{SamplePath}`: observed paths for each subject

# Returns
- `Matrix{Float64}`: p × n matrix where column i contains the score vector gᵢ
"""
function compute_subject_gradients_threaded(params::AbstractVector, model::MultistateModel, samplepaths::Vector{SamplePath})
    nsubj = length(samplepaths)
    nparams = length(params)
    
    # preallocate output
    grads = Matrix{Float64}(undef, nparams, nsubj)
    
    # Cache model references for thread safety
    hazards = model.hazards
    totalhazards = model.totalhazards
    tmat = model.tmat
    
    Threads.@threads for i in 1:nsubj
        path = samplepaths[i]
        w = model.SubjectWeights[i]
        
        function ll_subj_i(pars)
            pars_nested = unflatten_parameters(pars, model)
            subj_inds = model.subjectindices[path.subj]
            subj_dat = view(model.data, subj_inds, :)
            subjdat_df = make_subjdat(path, subj_dat)
            return loglik_path(pars_nested, subjdat_df, hazards, totalhazards, tmat) * w
        end
        
        grads[:, i] = ForwardDiff.gradient(ll_subj_i, params)
    end
    
    return grads
end

"""
    compute_subject_grads_and_hessians_fast(params, model::MultistateModel, samplepaths; use_threads=:auto)

Compute both subject-level gradients AND Hessians together efficiently.

This combined function avoids redundant computation when both quantities are needed,
and uses threading for parallelization when available.

# Arguments
- `params::AbstractVector`: parameter vector
- `model::MultistateModel`: multistate model with exact observation times
- `samplepaths::Vector{SamplePath}`: observed paths for each subject
- `use_threads::Symbol`: `:auto` (default), `:always`, or `:never`

# Returns
- `(grads::Matrix{Float64}, hessians::Vector{Matrix{Float64}})`: 
  - `grads`: p × n matrix of subject gradients
  - `hessians`: length-n vector of p × p Hessian matrices
"""
function compute_subject_grads_and_hessians_fast(params::AbstractVector, model::MultistateModel, 
                                                  samplepaths::Vector{SamplePath};
                                                  use_threads::Symbol = :auto)
    nsubj = length(samplepaths)
    nparams = length(params)
    
    # Preallocate outputs
    grads = Matrix{Float64}(undef, nparams, nsubj)
    hessians = [Matrix{Float64}(undef, nparams, nparams) for _ in 1:nsubj]
    
    # Cache model references for thread safety
    hazards = model.hazards
    totalhazards = model.totalhazards
    tmat = model.tmat
    
    should_thread = use_threads == :always || (use_threads == :auto && Threads.nthreads() > 1)
    
    # Pre-allocate DiffResults containers for each thread to avoid allocation in hot loop
    if should_thread
        # Create per-thread DiffResults to avoid contention
        # Use max thread ID possible in the system to handle all threading scenarios
        max_tid = Threads.maxthreadid()
        diffresults = [DiffResults.HessianResult(params) for _ in 1:max_tid]
        
        Threads.@threads for i in 1:nsubj
            tid = Threads.threadid()
            path = samplepaths[i]
            w = model.SubjectWeights[i]
            
            function ll_subj_i(pars)
                pars_nested = unflatten_parameters(pars, model)
                subj_inds = model.subjectindices[path.subj]
                subj_dat = view(model.data, subj_inds, :)
                subjdat_df = make_subjdat(path, subj_dat)
                return loglik_path(pars_nested, subjdat_df, hazards, totalhazards, tmat) * w
            end
            
            # Compute gradient and Hessian in ONE forward pass using DiffResults
            ForwardDiff.hessian!(diffresults[tid], ll_subj_i, params)
            grads[:, i] = DiffResults.gradient(diffresults[tid])
            hessians[i] .= DiffResults.hessian(diffresults[tid])
        end
    else
        diffresult = DiffResults.HessianResult(params)
        for i in 1:nsubj
            path = samplepaths[i]
            w = model.SubjectWeights[i]
            
            function ll_subj_i(pars)
                pars_nested = unflatten_parameters(pars, model)
                subj_inds = model.subjectindices[path.subj]
                subj_dat = view(model.data, subj_inds, :)
                subjdat_df = make_subjdat(path, subj_dat)
                return loglik_path(pars_nested, subjdat_df, hazards, totalhazards, tmat) * w
            end
            
            # Compute gradient and Hessian in ONE forward pass
            ForwardDiff.hessian!(diffresult, ll_subj_i, params)
            grads[:, i] = DiffResults.gradient(diffresult)
            hessians[i] .= DiffResults.hessian(diffresult)
        end
    end
    
    return grads, hessians
end

"""
    compute_subject_gradients(params, model::MultistateProcess, books)

Compute subject-level score vectors (gradients of log-likelihood) for Markov panel data.

For Markov models observed at discrete time points (panel data), the likelihood for subject i is:
```math
L_i(θ) = ∏_t P(X_{t+1} | X_t; θ)
```
where P is the transition probability matrix computed via matrix exponentials.

# Arguments
- `params::AbstractVector`: parameter vector (flat, on transformed scale)
- `model::MultistateProcess`: Markov model
- `books::Tuple`: bookkeeping structure for transition probability matrix computation

# Returns
- `Matrix{Float64}`: p × n matrix where column i contains the score vector gᵢ

# Notes
- Handles both fully observed and censored state observations
- Gradients computed via ForwardDiff through the matrix exponential
"""
function compute_subject_gradients(params::AbstractVector, model::MultistateProcess, books::Tuple)
    nsubj = length(model.subjectindices)
    nparams = length(params)
    
    # preallocate output
    grads = Matrix{Float64}(undef, nparams, nsubj)
    
    # get subject-level log-likelihoods via ForwardDiff
    data = MPanelData(model, books)
    
    for i in 1:nsubj
        # closure for subject i's log-likelihood
        function ll_subj_i(pars)
            # This is less efficient but necessary for per-subject gradients
            # We compute the full likelihood and extract subject i's contribution
            ll_subj_vec = loglik_markov(pars, data; neg=false, return_ll_subj=true)
            return ll_subj_vec[i]
        end
        
        grads[:, i] = ForwardDiff.gradient(ll_subj_i, params)
    end
    
    return grads
end

"""
    compute_subject_gradients(params, model::MultistateProcess, 
                             samplepaths, ImportanceWeights)

Compute subject-level score vectors (gradients of expected complete-data log-likelihood) for MCEM.

For semi-Markov models fitted via Monte Carlo EM, the score is computed as an importance-weighted
average over sampled paths:
```math
gᵢ = Σⱼ wᵢⱼ ∇_θ ℓᵢⱼ(θ)
```
where:
- wᵢⱼ are the normalized importance weights for path j of subject i
- ℓᵢⱼ is the complete-data log-likelihood for path j

# Arguments
- `params::AbstractVector`: parameter vector (flat, on transformed scale)
- `model::MultistateProcess`: semi-Markov model
- `samplepaths::Vector{Vector{SamplePath}}`: sampled paths for each subject (outer vector over subjects)
- `ImportanceWeights::Vector{Vector{Float64}}`: normalized importance weights

# Returns
- `Matrix{Float64}`: p × n matrix where column i contains the importance-weighted score gᵢ

# Notes
- Non-finite gradients (from numerical issues with extreme paths) are set to zero
- The sum Σᵢ gᵢ should be approximately zero at the MCEM solution
"""
function compute_subject_gradients(params::AbstractVector, 
                                   model::MultistateProcess,
                                   samplepaths::Vector{Vector{SamplePath}},
                                   ImportanceWeights::Vector{Vector{Float64}})
    nsubj = length(model.subjectindices)
    nparams = length(params)
    
    # preallocate output
    grads = Matrix{Float64}(undef, nparams, nsubj)
    
    # set up containers for path and sampling weight
    path = Array{SamplePath}(undef, 1)
    samplingweight = Vector{Float64}(undef, 1)
    
    # single argument function for log-likelihood
    ll = pars -> loglik_AD(pars, ExactDataAD(path, samplingweight, model.hazards, model); neg=false)
    
    # for each subject
    for i in 1:nsubj
        # set importance weight
        samplingweight[1] = model.SubjectWeights[i]
        
        # number of paths for this subject
        npaths = length(samplepaths[i])
        
        # accumulate weighted gradient
        g_i = zeros(Float64, nparams)
        
        for j in 1:npaths
            path[1] = samplepaths[i][j]
            grad_ij = ForwardDiff.gradient(ll, params)
            
            # handle non-finite gradients or importance weights
            w_j = ImportanceWeights[i][j]
            if !isfinite(w_j) || !all(isfinite, grad_ij)
                fill!(grad_ij, 0.0)
                w_j = 0.0
            end
            
            g_i .+= w_j * grad_ij
        end
        
        grads[:, i] = g_i
    end
    
    return grads
end

"""
    compute_subject_hessians(params, model::MultistateModel, samplepaths::Vector{SamplePath})

Compute subject-level Hessian (second derivative) contributions for exact data.

The Hessian for subject i is:
```math
Hᵢ = ∇²_θ ℓᵢ(θ) = ∂² log L_i(θ) / ∂θ∂θᵀ
```

The total Hessian is the sum: `H = Σᵢ Hᵢ`, and the observed Fisher information is `-H`.

# Arguments
- `params::AbstractVector`: parameter vector (flat, on transformed scale)
- `model::MultistateModel`: multistate model with exact observation times
- `samplepaths::Vector{SamplePath}`: observed paths for each subject

# Returns
- `Vector{Matrix{Float64}}`: length-n vector where element i is the p × p Hessian Hᵢ

# Notes
- Required for Cholesky downdate method of LOO perturbations
- Computed via ForwardDiff.hessian()

# Performance Note
This is the sequential implementation. For better performance, consider using
[`compute_subject_hessians_fast`](@ref) which automatically selects between
batched and threaded implementations based on available resources.
"""
function compute_subject_hessians(params::AbstractVector, model::MultistateModel, samplepaths::Vector{SamplePath})
    nsubj = length(model.subjectindices)
    nparams = length(params)
    
    # preallocate output
    hessians = [Matrix{Float64}(undef, nparams, nparams) for _ in 1:nsubj]
    
    # compute Hessian for each subject
    for i in 1:nsubj
        path = samplepaths[i]
        w = model.SubjectWeights[i]
        
        # closure for subject i's log-likelihood
        function ll_subj_i(pars)
            pars_nested = unflatten_parameters(pars, model)
            subj_inds = model.subjectindices[path.subj]
            subj_dat = view(model.data, subj_inds, :)
            subjdat_df = make_subjdat(path, subj_dat)
            return loglik_path(pars_nested, subjdat_df, model.hazards, model.totalhazards, model.tmat) * w
        end
        
        hessians[i] = ForwardDiff.hessian(ll_subj_i, params)
    end
    
    return hessians
end

"""
    BatchedHessianCache

Preallocated buffers for batched Hessian computation.
Reusing this cache across iterations avoids repeated large allocations.

# Fields
- `H_all::Matrix{Float64}`: (n*p) × p workspace for nested Jacobian
- `hessians::Vector{Matrix{Float64}}`: n preallocated p × p Hessian matrices
"""
mutable struct BatchedHessianCache
    H_all::Matrix{Float64}
    hessians::Vector{Matrix{Float64}}
end

"""
    BatchedHessianCache(n::Int, p::Int)

Create a cache for computing n subject Hessians each of size p × p.
"""
function BatchedHessianCache(n::Int, p::Int)
    H_all = Matrix{Float64}(undef, n * p, p)
    hessians = [Matrix{Float64}(undef, p, p) for _ in 1:n]
    return BatchedHessianCache(H_all, hessians)
end

"""
    resize!(cache::BatchedHessianCache, n::Int, p::Int)

Resize cache buffers for n subjects and p parameters.
"""
function Base.resize!(cache::BatchedHessianCache, n::Int, p::Int)
    cache.H_all = Matrix{Float64}(undef, n * p, p)
    cache.hessians = [Matrix{Float64}(undef, p, p) for _ in 1:n]
    return cache
end

"""
    compute_subject_hessians_batched(params, model::MultistateModel, samplepaths::Vector{SamplePath})

Compute all subject-level Hessians in a single batched operation.

This is much faster than `compute_subject_hessians` when n >> p because it requires
O(p) ForwardDiff passes instead of O(n).

# Algorithm
The key insight is that computing the Jacobian of the gradient function gives us
all Hessians:
1. Define f: ℝᵖ → ℝⁿ where f(β)[i] = ℓᵢ(β) (all subject log-likelihoods)
2. Define G: ℝᵖ → ℝⁿˣᵖ where G(β)[i,:] = ∇ℓᵢ(β) (all subject gradients)
3. Jacobian of vec(G) w.r.t. β gives (n*p) × p matrix
4. Reshape to (n, p, p) to get all subject Hessians

ForwardDiff.jacobian uses forward-mode AD, so cost is O(p) passes regardless of n.

# Arguments
- `params::AbstractVector`: parameter vector (flat, on transformed scale)
- `model::MultistateModel`: multistate model with exact observation times  
- `samplepaths::Vector{SamplePath}`: observed paths for each subject

# Returns
- `Vector{Matrix{Float64}}`: length-n vector where element i is the p × p Hessian Hᵢ

# Performance
For n=1000 subjects, p=15 params:
- Sequential: ~0.4s (1000 ForwardDiff.hessian calls)
- Batched: ~0.03s (1 ForwardDiff.jacobian call with p columns)
"""
function compute_subject_hessians_batched(params::AbstractVector, model::MultistateModel, 
                                          samplepaths::Vector{SamplePath};
                                          cache::Union{Nothing, BatchedHessianCache} = nothing)
    nsubj = length(samplepaths)
    nparams = length(params)
    
    # Use provided cache or create temporary one
    c = isnothing(cache) ? BatchedHessianCache(nsubj, nparams) : cache
    
    # Resize cache if needed (e.g., if n changed)
    if size(c.H_all, 1) != nsubj * nparams || size(c.H_all, 2) != nparams
        resize!(c, nsubj, nparams)
    end
    
    # Define function that returns all subject log-likelihoods as a vector
    function all_subject_logliks(pars)
        pars_nested = unflatten_parameters(pars, model)
        hazards = model.hazards
        totalhazards = model.totalhazards
        tmat = model.tmat
        
        lls = similar(pars, nsubj)
        for i in 1:nsubj
            path = samplepaths[i]
            w = model.SubjectWeights[i]
            subj_inds = model.subjectindices[path.subj]
            subj_dat = view(model.data, subj_inds, :)
            subjdat_df = make_subjdat(path, subj_dat)
            lls[i] = loglik_path(pars_nested, subjdat_df, hazards, totalhazards, tmat) * w
        end
        return lls
    end
    
    # Step 1: Get gradient function (returns n × p matrix flattened to vector)
    function all_gradients_flat(pars)
        J = ForwardDiff.jacobian(all_subject_logliks, pars)
        return vec(J)
    end
    
    # Step 2: Jacobian of gradients with preallocated output
    # J is n × p where J[i,k] = ∂ℓᵢ/∂βₖ
    # vec(J) is column-major: vec(J)[(k-1)*n + i] = J[i,k]
    # So H_all[(k-1)*n + i, j] = ∂J[i,k]/∂βⱼ = ∂²ℓᵢ/(∂βₖ∂βⱼ) = Hᵢ[k,j]
    ForwardDiff.jacobian!(c.H_all, all_gradients_flat, params)
    
    # Step 3: Extract individual Hessians into preallocated matrices
    for i in 1:nsubj
        for k in 1:nparams
            row_idx = (k-1) * nsubj + i
            for j in 1:nparams
                c.hessians[i][k, j] = c.H_all[row_idx, j]
            end
        end
    end
    
    return c.hessians
end

"""
    compute_subject_hessians_threaded(params, model::MultistateModel, samplepaths)

Compute all subject-level Hessians using multithreading.

This parallelizes the per-subject Hessian computation across available threads,
providing 3-6x speedup depending on thread count and problem size.

# Algorithm
Each thread computes ForwardDiff.hessian! for its assigned subjects independently.
The per-subject Hessians are embarrassingly parallel since they don't share state.

# Arguments
- `params::AbstractVector`: parameter vector (flat, on transformed scale)
- `model::MultistateModel`: multistate model with exact observation times  
- `samplepaths::Vector{SamplePath}`: observed paths for each subject

# Returns
- `Vector{Matrix{Float64}}`: length-n vector where element i is the p × p Hessian Hᵢ

# Performance (with 8 threads)
For n=2000 subjects, p=7 params:
- Sequential: ~148ms
- Batched: ~96ms  
- Threaded: ~30ms (5x speedup)

# Thread Safety
This function is thread-safe. Each thread operates on independent subject data
and writes to separate preallocated Hessian matrices.
"""
function compute_subject_hessians_threaded(params::AbstractVector, model::MultistateModel, 
                                           samplepaths::Vector{SamplePath})
    nsubj = length(samplepaths)
    nparams = length(params)
    
    # Preallocate output matrices
    hessians = [Matrix{Float64}(undef, nparams, nparams) for _ in 1:nsubj]
    
    # Cache model references for thread safety
    hazards = model.hazards
    totalhazards = model.totalhazards
    tmat = model.tmat
    
    # Parallel computation across subjects
    Threads.@threads for i in 1:nsubj
        # Define log-likelihood for subject i (closure captures i)
        function single_loglik(pars)
            pars_nested = unflatten_parameters(pars, model)
            path = samplepaths[i]
            w = model.SubjectWeights[i]
            subj_inds = model.subjectindices[path.subj]
            subj_dat = view(model.data, subj_inds, :)
            subjdat_df = make_subjdat(path, subj_dat)
            
            return loglik_path(pars_nested, subjdat_df, hazards, totalhazards, tmat) * w
        end
        
        # Compute Hessian directly into preallocated matrix
        ForwardDiff.hessian!(hessians[i], single_loglik, params)
    end
    
    return hessians
end

"""
    compute_subject_hessians_fast(params, model::MultistateModel, samplepaths; use_threads=:auto)

Compute all subject-level Hessians using the optimal available strategy.

Automatically selects between batched and threaded implementations based on
available threads and problem size.

# Arguments  
- `params::AbstractVector`: parameter vector (flat, on transformed scale)
- `model::MultistateModel`: multistate model with exact observation times
- `samplepaths::Vector{SamplePath}`: observed paths for each subject
- `use_threads::Symbol`: Strategy selection
  - `:auto` (default): Use threaded if nthreads() > 1, else batched
  - `:always`: Always use threaded (even with 1 thread)
  - `:never`: Always use batched

# Returns
- `Vector{Matrix{Float64}}`: length-n vector where element i is the p × p Hessian Hᵢ

# Performance Guidelines
- With 4+ threads: Threaded is 3-5x faster than batched
- With 1 thread: Batched is ~1.5x faster than sequential
- Threaded scales nearly linearly with thread count

See also: [`compute_subject_hessians_batched`](@ref), [`compute_subject_hessians_threaded`](@ref)
"""
function compute_subject_hessians_fast(params::AbstractVector, model::MultistateModel, 
                                       samplepaths::Vector{SamplePath};
                                       use_threads::Symbol = :auto)
    if use_threads == :always || (use_threads == :auto && Threads.nthreads() > 1)
        return compute_subject_hessians_threaded(params, model, samplepaths)
    else
        return compute_subject_hessians_batched(params, model, samplepaths)
    end
end

"""
    compute_subject_hessians(params, model::MultistateProcess, books)

Compute subject-level Hessian contributions for Markov panel data.

Returns Vector{Matrix{Float64}} of length n, each matrix is p × p.
"""
function compute_subject_hessians(params::AbstractVector, model::MultistateProcess, books::Tuple)
    nsubj = length(model.subjectindices)
    nparams = length(params)
    
    # preallocate output
    hessians = [Matrix{Float64}(undef, nparams, nparams) for _ in 1:nsubj]
    
    data = MPanelData(model, books)
    
    for i in 1:nsubj
        # closure for subject i's log-likelihood
        function ll_subj_i(pars)
            ll_subj_vec = loglik_markov(pars, data; neg=false, return_ll_subj=true)
            return ll_subj_vec[i]
        end
        
        hessians[i] = ForwardDiff.hessian(ll_subj_i, params)
    end
    
    return hessians
end

# =============================================================================
# ExactData interface: gradient and Hessian computation for exact (continuously observed) data
# =============================================================================
#
# These methods use loglik_exact with return_ll_subj=true to compute per-subject
# log-likelihoods, following the same pattern as the MPanelData versions.
# This allows the gradient/Hessian computation to use the vectorized loglik_exact
# infrastructure rather than computing path-by-path.
#
# IMPORTANT: These methods are designed to be compatible with the PIJCV analytical
# gradient formula in compute_pijcv_with_gradient(), which requires subject-level
# gradients and Hessians as inputs. Using the same interface as MPanelData ensures
# consistency across data types.
# =============================================================================

"""
    compute_subject_gradients(params, model::MultistateProcess, data::ExactData)

Compute subject-level score vectors (gradients of log-likelihood) for exact data.

This method uses `loglik_exact` with `return_ll_subj=true` to compute per-subject
log-likelihoods, then differentiates each subject's contribution via ForwardDiff.

# Arguments
- `params::AbstractVector`: parameter vector (flat, on transformed scale)
- `model::MultistateProcess`: multistate model with exact observation times
- `data::ExactData`: exact data container with sample paths

# Returns
- `Matrix{Float64}`: p × n matrix where column i contains the score vector gᵢ

# Notes
- Uses the vectorized `loglik_exact` infrastructure for consistent behavior
- Compatible with `compute_pijcv_with_gradient` analytical gradient formula
"""
function compute_subject_gradients(params::AbstractVector, model::MultistateProcess, data::ExactData)
    nsubj = length(data.paths)
    nparams = length(params)
    
    # preallocate output
    grads = Matrix{Float64}(undef, nparams, nsubj)
    
    for i in 1:nsubj
        # closure for subject i's log-likelihood
        function ll_subj_i(pars)
            # Use loglik_exact with return_ll_subj=true to get per-subject contributions
            ll_subj_vec = loglik_exact(pars, data; neg=false, return_ll_subj=true)
            return ll_subj_vec[i]
        end
        
        grads[:, i] = ForwardDiff.gradient(ll_subj_i, params)
    end
    
    return grads
end

"""
    compute_subject_hessians(params, model::MultistateProcess, data::ExactData)

Compute subject-level Hessian contributions for exact data.

This method uses `loglik_exact` with `return_ll_subj=true` to compute per-subject
log-likelihoods, then differentiates twice via ForwardDiff.hessian.

# Arguments
- `params::AbstractVector`: parameter vector (flat, on transformed scale)
- `model::MultistateProcess`: multistate model with exact observation times
- `data::ExactData`: exact data container with sample paths

# Returns
- `Vector{Matrix{Float64}}`: length-n vector where element i is the p × p Hessian Hᵢ

# Notes
- Uses the vectorized `loglik_exact` infrastructure for consistent behavior
- Compatible with `compute_pijcv_with_gradient` analytical gradient formula
"""
function compute_subject_hessians(params::AbstractVector, model::MultistateProcess, data::ExactData)
    nsubj = length(data.paths)
    nparams = length(params)
    
    # preallocate output
    hessians = [Matrix{Float64}(undef, nparams, nparams) for _ in 1:nsubj]
    
    for i in 1:nsubj
        # closure for subject i's log-likelihood
        function ll_subj_i(pars)
            # Use loglik_exact with return_ll_subj=true to get per-subject contributions
            ll_subj_vec = loglik_exact(pars, data; neg=false, return_ll_subj=true)
            return ll_subj_vec[i]
        end
        
        hessians[i] = ForwardDiff.hessian(ll_subj_i, params)
    end
    
    return hessians
end

