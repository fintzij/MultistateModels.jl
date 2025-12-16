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

These functions are typically called internally by `fit()` when `compute_ij_vcov=true` or
`compute_jk_vcov=true` is specified. Users can access results via:

```julia
fitted = fit(model, data; compute_ij_vcov=true, compute_jk_vcov=true)
ij_v = get_ij_vcov(fitted)           # Sandwich variance
jk_v = get_jk_vcov(fitted)           # Jackknife variance  
grads = get_subject_gradients(fitted) # Subject-level scores
deltas = get_loo_perturbations(fitted) # LOO perturbations
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
            pars_nested = unflatten_natural(pars, model)
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
    compute_subject_gradients(params, model::MultistateMarkovProcess, books)

Compute subject-level score vectors (gradients of log-likelihood) for Markov panel data.

For Markov models observed at discrete time points (panel data), the likelihood for subject i is:
```math
L_i(θ) = ∏_t P(X_{t+1} | X_t; θ)
```
where P is the transition probability matrix computed via matrix exponentials.

# Arguments
- `params::AbstractVector`: parameter vector (flat, on transformed scale)
- `model::MultistateMarkovProcess`: Markov model
- `books::Tuple`: bookkeeping structure for transition probability matrix computation

# Returns
- `Matrix{Float64}`: p × n matrix where column i contains the score vector gᵢ

# Notes
- Handles both fully observed and censored state observations
- Gradients computed via ForwardDiff through the matrix exponential
"""
function compute_subject_gradients(params::AbstractVector, model::MultistateMarkovProcess, books::Tuple)
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
    compute_subject_gradients(params, model::MultistateSemiMarkovProcess, 
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
- `model::MultistateSemiMarkovProcess`: semi-Markov model
- `samplepaths::Vector{Vector{SamplePath}}`: sampled paths for each subject (outer vector over subjects)
- `ImportanceWeights::Vector{Vector{Float64}}`: normalized importance weights

# Returns
- `Matrix{Float64}`: p × n matrix where column i contains the importance-weighted score gᵢ

# Notes
- Non-finite gradients (from numerical issues with extreme paths) are set to zero
- The sum Σᵢ gᵢ should be approximately zero at the MCEM solution
"""
function compute_subject_gradients(params::AbstractVector, 
                                   model::MultistateSemiMarkovProcess,
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
            
            # handle non-finite gradients
            if !all(isfinite, grad_ij)
                fill!(grad_ij, 0.0)
            end
            
            g_i .+= ImportanceWeights[i][j] * grad_ij
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
            pars_nested = unflatten_natural(pars, model)
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
    compute_subject_hessians(params, model::MultistateMarkovProcess, books)

Compute subject-level Hessian contributions for Markov panel data.

Returns Vector{Matrix{Float64}} of length n, each matrix is p × p.
"""
function compute_subject_hessians(params::AbstractVector, model::MultistateMarkovProcess, books::Tuple)
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

# ============================================================================
# Batched Fisher Information Computation for MCEM
# ============================================================================

"""
    loglik_weighted_with_params(θw, npaths, nparams, subject_paths, samplingweight, model)

Compute weighted sum of log-likelihoods where θ and w are concatenated into one vector.

This enables computing all path gradients via the mixed Hessian ∂²L/∂θ∂w.
If L(θ,w) = Σⱼ wⱼ·llⱼ(θ), then ∂²L/∂θₖ∂wⱼ = ∂llⱼ/∂θₖ = gⱼₖ.

# Arguments
- `θw::AbstractVector`: concatenated vector [θ; w] of length (nparams + npaths)
- `npaths::Int`: number of paths
- `nparams::Int`: number of parameters
- `subject_paths::Vector{SamplePath}`: all sampled paths for this subject
- `samplingweight::Float64`: subject-level sampling weight  
- `model::MultistateProcess`: model containing data and hazard specifications

# Returns
- Scalar: weighted sum Σⱼ wⱼ·llⱼ(θ)
"""
function loglik_weighted_with_params(θw::AbstractVector{T}, npaths::Int, nparams::Int,
                                     subject_paths::Vector{SamplePath}, 
                                     samplingweight::Float64, 
                                     model::MultistateProcess) where T
    # Split the concatenated vector
    θ = @view θw[1:nparams]
    w = @view θw[nparams+1:end]
    
    # unflatten parameters to natural scale
    pars_nested = unflatten_natural(θ, model)
    hazards = model.hazards
    
    # remake spline parameters if needed (no-op for RuntimeSplineHazard)
    for i in eachindex(hazards)
        if isa(hazards[i], _SplineHazard)
            remake_splines!(hazards[i], nothing)
            set_riskperiod!(hazards[i])
        end
    end
    
    # compute weighted sum of log-likelihoods
    ll_weighted = zero(T)
    for j in 1:npaths
        path = subject_paths[j]
        subj_inds = model.subjectindices[path.subj]
        subj_dat = view(model.data, subj_inds, :)
        subjdat_df = make_subjdat(path, subj_dat)
        ll_j = loglik_path(pars_nested, subjdat_df, hazards, model.totalhazards, model.tmat) * samplingweight
        ll_weighted += w[j] * ll_j
    end
    
    return ll_weighted
end

"""
    loglik_paths_subject(pars, subject_paths, samplingweight, model)

Compute log-likelihoods for all paths of a single subject as a vector.

This function enables efficient Jacobian computation: ForwardDiff.jacobian applied to this
function gives all per-path gradients in a single reverse-mode AD pass, which is much faster
than computing O(n_paths) individual gradients.

# Arguments
- `pars::AbstractVector`: parameter vector (flat, on transformed scale)
- `subject_paths::Vector{SamplePath}`: all sampled paths for this subject
- `samplingweight::Float64`: subject-level sampling weight
- `model::MultistateProcess`: model containing data and hazard specifications

# Returns
- `Vector{T}`: vector of length n_paths containing log-likelihood for each path
"""
function loglik_paths_subject(pars::AbstractVector{T}, subject_paths::Vector{SamplePath}, 
                              samplingweight::Float64, model::MultistateProcess) where T
    npaths = length(subject_paths)
    lls = Vector{T}(undef, npaths)
    
    # unflatten parameters to natural scale
    pars_nested = unflatten_natural(pars, model)
    hazards = model.hazards
    
    # remake spline parameters if needed (no-op for RuntimeSplineHazard)
    for i in eachindex(hazards)
        if isa(hazards[i], _SplineHazard)
            remake_splines!(hazards[i], nothing)
            set_riskperiod!(hazards[i])
        end
    end
    
    # compute log-likelihood for each path
    for j in 1:npaths
        path = subject_paths[j]
        subj_inds = model.subjectindices[path.subj]
        subj_dat = view(model.data, subj_inds, :)
        subjdat_df = make_subjdat(path, subj_dat)
        lls[j] = loglik_path(pars_nested, subjdat_df, hazards, model.totalhazards, model.tmat) * samplingweight
    end
    
    return lls
end

"""
    loglik_weighted_subject(pars, subject_paths, weights, samplingweight, model)

Compute importance-weighted sum of log-likelihoods for all paths of a single subject.

This function enables efficient weighted Hessian computation: ForwardDiff.hessian applied
to this function gives `Σⱼ wⱼ Hⱼ` in a single second-order AD pass, instead of O(n_paths)
individual Hessian computations.

From the linearity of differentiation:
    ∇²[Σⱼ wⱼ log f(Zⱼ;θ)] = Σⱼ wⱼ ∇² log f(Zⱼ;θ)

# Arguments
- `pars::AbstractVector`: parameter vector (flat, on transformed scale)
- `subject_paths::Vector{SamplePath}`: all sampled paths for this subject
- `weights::Vector{Float64}`: importance weights for each path (should sum to 1)
- `samplingweight::Float64`: subject-level sampling weight
- `model::MultistateProcess`: model containing data and hazard specifications

# Returns
- `T`: weighted sum `Σⱼ wⱼ llⱼ` of log-likelihoods
"""
function loglik_weighted_subject(pars::AbstractVector{T}, subject_paths::Vector{SamplePath}, 
                                 weights::Vector{Float64}, samplingweight::Float64, 
                                 model::MultistateProcess) where T
    npaths = length(subject_paths)
    ll_weighted = zero(T)
    
    # unflatten parameters to natural scale
    pars_nested = unflatten_natural(pars, model)
    hazards = model.hazards
    
    # remake spline parameters if needed (no-op for RuntimeSplineHazard)
    for i in eachindex(hazards)
        if isa(hazards[i], _SplineHazard)
            remake_splines!(hazards[i], nothing)
            set_riskperiod!(hazards[i])
        end
    end
    
    # compute weighted sum of log-likelihoods
    for j in 1:npaths
        path = subject_paths[j]
        subj_inds = model.subjectindices[path.subj]
        subj_dat = view(model.data, subj_inds, :)
        subjdat_df = make_subjdat(path, subj_dat)
        ll_j = loglik_path(pars_nested, subjdat_df, hazards, model.totalhazards, model.tmat) * samplingweight
        ll_weighted += weights[j] * ll_j
    end
    
    return ll_weighted
end

"""
    compute_subject_gradients_batched(params, model, samplepaths, ImportanceWeights)

Batched computation of subject-level gradients using Jacobian.

Instead of computing O(n_paths) individual gradients per subject, this uses a single
Jacobian call to get all path gradients simultaneously:
    J = ForwardDiff.jacobian(θ -> [ll₁(θ), ll₂(θ), ..., llₘ(θ)], θ)
gives J[j,k] = ∂llⱼ/∂θₖ, so gradients are rows of J.

The subject-level gradient is then: gᵢ = Σⱼ wᵢⱼ J[j,:]

# Arguments
- `params::AbstractVector`: parameter vector (flat, on transformed scale)
- `model`: semi-Markov model
- `samplepaths::Vector{Vector{SamplePath}}`: sampled paths for each subject
- `ImportanceWeights::Vector{Vector{Float64}}`: normalized importance weights

# Returns
- `subject_grads::Matrix{Float64}`: p × n matrix of subject-level gradients
- `path_grads::Vector{Matrix{Float64}}`: vector of p × n_paths_i matrices for each subject
"""
function compute_subject_gradients_batched(params::AbstractVector,
                                           model::MultistateSemiMarkovProcess,
                                           samplepaths::Vector{Vector{SamplePath}},
                                           ImportanceWeights::Vector{Vector{Float64}})
    nsubj = length(model.subjectindices)
    nparams = length(params)
    
    # preallocate outputs
    subject_grads = Matrix{Float64}(undef, nparams, nsubj)
    path_grads = Vector{Matrix{Float64}}(undef, nsubj)
    
    for i in 1:nsubj
        samplingweight = model.SubjectWeights[i]
        subject_paths = samplepaths[i]
        weights = ImportanceWeights[i]
        npaths = length(subject_paths)
        
        # closure for vectorized log-likelihoods
        ll_vec = pars -> loglik_paths_subject(pars, subject_paths, samplingweight, model)
        
        # Jacobian: J[j,k] = ∂llⱼ/∂θₖ  (n_paths × n_params)
        J = ForwardDiff.jacobian(ll_vec, params)
        
        # Store path gradients (transpose to get p × n_paths)
        path_grads[i] = Matrix(transpose(J))
        
        # Handle non-finite values
        for j in 1:npaths
            if !all(isfinite, @view(path_grads[i][:, j]))
                path_grads[i][:, j] .= 0.0
            end
        end
        
        # Subject gradient: gᵢ = Σⱼ wᵢⱼ gⱼ
        subject_grads[:, i] = path_grads[i] * weights
    end
    
    return subject_grads, path_grads
end

"""
    compute_fisher_via_mixed_hessian(params, model, samplepaths, ImportanceWeights)

Compute Fisher information using mixed Hessian trick for gradients.

This exploits the insight that if L(θ, w) = Σⱼ wⱼ·llⱼ(θ), then the mixed partial
∂²L/∂θₖ∂wⱼ = ∂llⱼ/∂θₖ = gⱼₖ gives us all path gradients from ONE Hessian computation.

The full Hessian of L w.r.t. [θ; w] has structure:
    ⎡ ∂²L/∂θ²    ∂²L/∂θ∂w ⎤   ⎡ Σⱼ wⱼHⱼ    G      ⎤
    ⎣ ∂²L/∂w∂θ   ∂²L/∂w²  ⎦ = ⎣ Gᵀ         0      ⎦

where G[:,j] = gⱼ (the gradient of path j's log-likelihood).

This gives us:
1. Σⱼ wⱼHⱼ from the upper-left block (n_params × n_params)
2. All path gradients gⱼ from the upper-right block (n_params × n_paths)

# Arguments
- `params::AbstractVector`: parameter vector
- `model`: semi-Markov model  
- `samplepaths::Vector{Vector{SamplePath}}`: sampled paths
- `ImportanceWeights::Vector{Vector{Float64}}`: importance weights

# Returns
NamedTuple with fishinf, subject_grads, subject_hessians
"""
function compute_fisher_via_mixed_hessian(params::AbstractVector,
                                          model::MultistateSemiMarkovProcess,
                                          samplepaths::Vector{Vector{SamplePath}},
                                          ImportanceWeights::Vector{Vector{Float64}})
    nsubj = length(model.subjectindices)
    nparams = length(params)
    
    # preallocate outputs
    subject_grads = Matrix{Float64}(undef, nparams, nsubj)
    subject_hessians = [zeros(Float64, nparams, nparams) for _ in 1:nsubj]
    fishinf = zeros(Float64, nparams, nparams)
    
    for i in 1:nsubj
        samplingweight = model.SubjectWeights[i]
        subject_paths = samplepaths[i]
        weights = ImportanceWeights[i]
        npaths = length(subject_paths)
        
        # Concatenate [θ; w] for the mixed Hessian computation
        θw = vcat(params, weights)
        
        # Closure for L(θ, w) = Σⱼ wⱼ·llⱼ(θ)
        L = θw_vec -> loglik_weighted_with_params(θw_vec, npaths, nparams, subject_paths, samplingweight, model)
        
        # Compute full Hessian of L w.r.t. [θ; w]
        H_full = ForwardDiff.hessian(L, θw)
        
        # Extract blocks:
        # H_θθ = ∂²L/∂θ² = Σⱼ wⱼHⱼ (upper-left, n_params × n_params)
        H_weighted = H_full[1:nparams, 1:nparams]
        
        # G = ∂²L/∂θ∂w where G[:,j] = gⱼ (upper-right, n_params × n_paths)
        path_grads_i = H_full[1:nparams, nparams+1:end]
        
        # Handle non-finite values
        if !all(isfinite, H_weighted)
            fill!(H_weighted, 0.0)
        end
        for j in 1:npaths
            if !all(isfinite, @view(path_grads_i[:, j]))
                path_grads_i[:, j] .= 0.0
            end
        end
        
        # Subject gradient: gᵢ = Σⱼ wⱼ gⱼ
        subject_grads[:, i] = path_grads_i * weights
        
        # Compute fish_i1: Σⱼ wⱼ (-Hⱼ - gⱼgⱼᵀ)
        # = -Σⱼ wⱼ Hⱼ - Σⱼ wⱼ gⱼgⱼᵀ
        fish_i1 = -H_weighted
        for j in 1:npaths
            gj = @view path_grads_i[:, j]
            fish_i1 .-= weights[j] * (gj * transpose(gj))
        end
        
        # Compute fish_i2: (Σⱼ wⱼ gⱼ)(Σₖ wₖ gₖ)ᵀ = gᵢ gᵢᵀ
        gi = @view subject_grads[:, i]
        fish_i2 = gi * transpose(gi)
        
        # Subject i's Fisher information contribution
        subject_hessians[i] = fish_i1 + fish_i2
        fishinf .+= subject_hessians[i]
    end
    
    return (fishinf = Symmetric(fishinf),
            subject_grads = subject_grads,
            subject_hessians = subject_hessians)
end

"""
    compute_subject_fisher_louis_batched(params, model, samplepaths, ImportanceWeights)

Batched computation of subject-level Fisher information using Louis's identity.

This optimized implementation uses:
1. **One Jacobian call** per subject to get all path gradients
2. **One Hessian call** per subject to get the weighted sum of Hessians

Louis's identity for subject i (equation S8 from Morsomme et al. 2025):
```math
Iᵢ = Mᵢ Σⱼ νᵢⱼ [-Hᵢⱼ - gᵢⱼgᵢⱼᵀ] + Mᵢ² (Σⱼ νᵢⱼ gᵢⱼ)(Σₖ νᵢₖ gᵢₖ)ᵀ
```

The key optimization is that `Σⱼ wⱼ Hⱼ = ∇²[Σⱼ wⱼ llⱼ]` can be computed with
ONE Hessian call instead of O(n_paths) calls.

# Complexity
- Old: O(n_subjects × n_paths) Hessian calls
- New: O(n_subjects) Jacobian + O(n_subjects) Hessian calls

# Arguments
- `params::AbstractVector`: parameter vector (flat, on transformed scale)
- `model`: semi-Markov model
- `samplepaths::Vector{Vector{SamplePath}}`: sampled paths for each subject
- `ImportanceWeights::Vector{Vector{Float64}}`: normalized importance weights

# Returns
NamedTuple with:
- `fishinf::Symmetric{Float64}`: total observed Fisher information matrix
- `subject_grads::Matrix{Float64}`: p × n matrix of subject-level gradients
- `subject_hessians::Vector{Matrix{Float64}}`: length-n vector of subject Fisher contributions
"""
function compute_subject_fisher_louis_batched(params::AbstractVector,
                                              model::MultistateSemiMarkovProcess,
                                              samplepaths::Vector{Vector{SamplePath}},
                                              ImportanceWeights::Vector{Vector{Float64}})
    nsubj = length(model.subjectindices)
    nparams = length(params)
    
    # Get batched gradients (subject-level and per-path)
    subject_grads, path_grads = compute_subject_gradients_batched(params, model, samplepaths, ImportanceWeights)
    
    # preallocate Fisher information containers
    subject_hessians = [zeros(Float64, nparams, nparams) for _ in 1:nsubj]
    fishinf = zeros(Float64, nparams, nparams)
    
    # Containers for Hessian computation
    diffres = DiffResults.HessianResult(params)
    
    for i in 1:nsubj
        samplingweight = model.SubjectWeights[i]
        subject_paths = samplepaths[i]
        weights = ImportanceWeights[i]
        npaths = length(subject_paths)
        
        # Closure for weighted log-likelihood sum
        ll_weighted = pars -> loglik_weighted_subject(pars, subject_paths, weights, samplingweight, model)
        
        # ONE Hessian call gives Σⱼ wⱼ Hⱼ
        diffres = ForwardDiff.hessian!(diffres, ll_weighted, params)
        H_weighted = DiffResults.hessian(diffres)
        
        # Handle non-finite values
        if !all(isfinite, H_weighted)
            fill!(H_weighted, 0.0)
        end
        
        # Compute fish_i1: Σⱼ wⱼ (-Hⱼ - gⱼgⱼᵀ)
        # = -Σⱼ wⱼ Hⱼ - Σⱼ wⱼ gⱼgⱼᵀ
        fish_i1 = -H_weighted
        for j in 1:npaths
            gj = @view path_grads[i][:, j]
            fish_i1 .-= weights[j] * (gj * transpose(gj))
        end
        
        # Compute fish_i2: (Σⱼ wⱼ gⱼ)(Σₖ wₖ gₖ)ᵀ = gᵢ gᵢᵀ
        # where gᵢ = Σⱼ wⱼ gⱼ is the subject-level gradient
        gi = @view subject_grads[:, i]
        fish_i2 = gi * transpose(gi)
        
        # Subject i's Fisher information contribution
        subject_hessians[i] = fish_i1 + fish_i2
        fishinf .+= subject_hessians[i]
    end
    
    return (fishinf = Symmetric(fishinf),
            subject_grads = subject_grads,
            subject_hessians = subject_hessians)
end

# ============================================================================
# End Batched Fisher Information Computation
# ============================================================================

"""
    compute_subject_hessians(params, model::MultistateSemiMarkovProcess, 
                            samplepaths, ImportanceWeights)

Compute subject-level observed Fisher information contributions for MCEM using Louis's identity.

For models fitted via Monte Carlo EM, the observed Fisher information cannot be computed
directly from the observed-data likelihood. Louis's identity provides a way to compute it
from the complete-data quantities:

```math
I_i^{obs} = E[I_i^{comp}|Y_i] - Var[S_i^{comp}|Y_i]
```

Implemented as:
```math
I_i = Σⱼ wᵢⱼ (-Hᵢⱼ - gᵢⱼgᵢⱼᵀ) + (Σⱼ wᵢⱼ gᵢⱼ)(Σₖ wᵢₖ gᵢₖ)ᵀ
```

where:
- wᵢⱼ are normalized importance weights
- Hᵢⱼ is the complete-data Hessian for path j
- gᵢⱼ is the complete-data score for path j

# Arguments
- `params::AbstractVector`: parameter vector (flat, on transformed scale)
- `model::MultistateSemiMarkovProcess`: semi-Markov model
- `samplepaths::Vector{Vector{SamplePath}}`: sampled paths for each subject
- `ImportanceWeights::Vector{Vector{Float64}}`: normalized importance weights

# Returns
- `Vector{Matrix{Float64}}`: length-n vector where element i is the p × p Fisher information contribution

# References
- Louis, T. A. (1982). Finding the observed information matrix when using the EM algorithm. 
  Journal of the Royal Statistical Society: Series B, 44(2), 226-233.
"""
function compute_subject_hessians(params::AbstractVector, 
                                  model::MultistateSemiMarkovProcess,
                                  samplepaths::Vector{Vector{SamplePath}},
                                  ImportanceWeights::Vector{Vector{Float64}})
    nsubj = length(model.subjectindices)
    nparams = length(params)
    
    # preallocate output
    hessians = [zeros(Float64, nparams, nparams) for _ in 1:nsubj]
    
    # set up containers for path and sampling weight
    path = Array{SamplePath}(undef, 1)
    samplingweight = Vector{Float64}(undef, 1)
    
    # DiffResults for combined gradient/Hessian computation
    diffres = DiffResults.HessianResult(params)
    
    # single argument function for log-likelihood
    ll = pars -> loglik_AD(pars, ExactDataAD(path, samplingweight, model.hazards, model); neg=false)
    
    # temporary storage for path gradients
    grads_j = Vector{Vector{Float64}}()
    
    # for each subject
    for i in 1:nsubj
        # set importance weight
        samplingweight[1] = model.SubjectWeights[i]
        
        # number of paths for this subject
        npaths = length(samplepaths[i])
        
        # resize and reset gradient storage
        resize!(grads_j, npaths)
        
        # First pass: compute gradients and first part of Louis's identity
        fish_i1 = zeros(Float64, nparams, nparams)
        
        for j in 1:npaths
            path[1] = samplepaths[i][j]
            diffres = ForwardDiff.hessian!(diffres, ll, params)
            
            # grab gradient
            grads_j[j] = copy(DiffResults.gradient(diffres))
            hess_j = DiffResults.hessian(diffres)
            
            # handle non-finite values
            if !all(isfinite, hess_j)
                fill!(hess_j, 0.0)
            end
            if !all(isfinite, grads_j[j])
                fill!(grads_j[j], 0.0)
            end
            
            # Accumulate: wᵢⱼ * (-Hᵢⱼ - gᵢⱼgᵢⱼ')
            fish_i1 .+= ImportanceWeights[i][j] * (-hess_j - grads_j[j] * transpose(grads_j[j]))
        end
        
        # Optimized: Σⱼ Σₖ wⱼwₖ gⱼgₖᵀ = (Σⱼ wⱼgⱼ)(Σₖ wₖgₖ)ᵀ
        # Stack gradients into matrix and compute weighted sum
        G = reduce(hcat, grads_j)  # n_params × n_paths
        g_weighted = G * ImportanceWeights[i]  # n_params × 1
        fish_i2 = g_weighted * transpose(g_weighted)
        
        # Subject i's contribution to Fisher information (negative Hessian)
        # Store as -H_i for consistency with variance estimation
        hessians[i] = fish_i1 + fish_i2
    end
    
    return hessians
end

"""
    compute_fisher_components(params, model, data_or_paths; 
                             compute_subject_contributions=false,
                             importance_weights=nothing)

Unified helper to compute Fisher information and optionally subject-level contributions.

# Arguments
- `params`: parameter vector
- `model`: multistate model
- `data_or_paths`: either ExactData, MPanelData, or (samplepaths, ImportanceWeights) tuple for MCEM
- `compute_subject_contributions`: whether to also return per-subject gradients and Hessians

# Returns NamedTuple with:
- `fishinf`: Fisher information matrix (p × p)
- `subject_grads`: Matrix (p × n) if compute_subject_contributions=true, else nothing
- `subject_hessians`: Vector of matrices if compute_subject_contributions=true, else nothing
"""
function compute_fisher_components(params::AbstractVector, 
                                   model::MultistateModel,
                                   samplepaths::Vector{SamplePath};
                                   compute_subject_contributions::Bool = false)
    nparams = length(params)
    
    if compute_subject_contributions
        # Compute subject-level gradients and Hessians
        subject_grads = compute_subject_gradients(params, model, samplepaths)
        subject_hessians = compute_subject_hessians(params, model, samplepaths)
        
        # Aggregate Fisher information
        fishinf = zeros(Float64, nparams, nparams)
        for H_i in subject_hessians
            fishinf .-= H_i  # Fisher info is negative Hessian
        end
        
        return (fishinf = Symmetric(fishinf), 
                subject_grads = subject_grads, 
                subject_hessians = subject_hessians)
    else
        # Just compute total Fisher info via ForwardDiff
        data = ExactData(model, samplepaths)
        diffres = DiffResults.HessianResult(params)
        ll = pars -> loglik_exact(pars, data; neg=false)
        diffres = ForwardDiff.hessian!(diffres, ll, params)
        fishinf = -DiffResults.hessian(diffres)
        
        return (fishinf = Symmetric(fishinf), 
                subject_grads = nothing, 
                subject_hessians = nothing)
    end
end

function compute_fisher_components(params::AbstractVector,
                                   model::MultistateMarkovProcess,
                                   books::Tuple;
                                   compute_subject_contributions::Bool = false)
    nparams = length(params)
    
    if compute_subject_contributions
        # Compute subject-level gradients and Hessians
        subject_grads = compute_subject_gradients(params, model, books)
        subject_hessians = compute_subject_hessians(params, model, books)
        
        # Aggregate Fisher information
        fishinf = zeros(Float64, nparams, nparams)
        for H_i in subject_hessians
            fishinf .-= H_i
        end
        
        return (fishinf = Symmetric(fishinf), 
                subject_grads = subject_grads, 
                subject_hessians = subject_hessians)
    else
        # Just compute total Fisher info
        data = MPanelData(model, books)
        diffres = DiffResults.HessianResult(params)
        ll = pars -> loglik_markov(pars, data; neg=false)
        diffres = ForwardDiff.hessian!(diffres, ll, params)
        fishinf = -DiffResults.hessian(diffres)
        
        return (fishinf = Symmetric(fishinf), 
                subject_grads = nothing, 
                subject_hessians = nothing)
    end
end

function compute_fisher_components(params::AbstractVector,
                                   model::MultistateSemiMarkovProcess,
                                   samplepaths::Vector{Vector{SamplePath}},
                                   ImportanceWeights::Vector{Vector{Float64}};
                                   compute_subject_contributions::Bool = false,
                                   use_batched::Bool = false)
    nsubj = length(model.subjectindices)
    nparams = length(params)
    
    # For MCEM, we always need per-subject contributions to compute Fisher info
    # Note: Batched implementation is available but sequential is often faster for simple models
    # due to ForwardDiff overhead. Batched may be beneficial for complex models with many parameters.
    if use_batched
        result = compute_subject_fisher_louis_batched(params, model, samplepaths, ImportanceWeights)
        fishinf = result.fishinf
        subject_grads = result.subject_grads
        subject_hessians = result.subject_hessians
    else
        # Original sequential implementation
        subject_grads = compute_subject_gradients(params, model, samplepaths, ImportanceWeights)
        subject_hessians = compute_subject_hessians(params, model, samplepaths, ImportanceWeights)
        
        # Aggregate Fisher information (subject_hessians already contains -H contributions)
        fishinf = zeros(Float64, nparams, nparams)
        for H_i in subject_hessians
            fishinf .+= H_i
        end
        fishinf = Symmetric(fishinf)
    end
    
    if compute_subject_contributions
        return (fishinf = Symmetric(fishinf), 
                subject_grads = subject_grads, 
                subject_hessians = subject_hessians)
    else
        return (fishinf = Symmetric(fishinf), 
                subject_grads = nothing, 
                subject_hessians = nothing)
    end
end

# --- LOO perturbations ---

"""
    loo_perturbations_direct(H_inv, subject_grads)

Compute leave-one-out (LOO) parameter perturbations via direct matrix-vector multiplication.

This implements the one-step Newton approximation to the LOO estimates:
```math
θ̂_{-i} ≈ θ̂ + H⁻¹ gᵢ = θ̂ + Δᵢ
```

The perturbation Δᵢ = H⁻¹gᵢ represents how much the parameter estimates would change
if subject i were removed from the dataset.

# Arguments
- `H_inv::AbstractMatrix`: inverse Hessian (or variance-covariance matrix), p × p
- `subject_grads::AbstractMatrix`: subject-level score vectors, p × n

# Returns
- `Matrix{Float64}`: p × n matrix where column i is Δᵢ = H⁻¹gᵢ

# Computational Complexity
- O(p²n) given precomputed H⁻¹
- Preferred when n >> p (typical case)

# Example
```julia
# Get perturbations from fitted model
fitted = fit(model, data; compute_ij_vcov=true)
deltas = get_loo_perturbations(fitted)  # Uses this function internally

# LOO estimate for subject 3:
theta_loo_3 = get_parameters_flat(fitted) .+ deltas[:, 3]
```

See also: [`loo_perturbations_cholesky`](@ref), [`get_loo_perturbations`](@ref)
"""
function loo_perturbations_direct(H_inv::AbstractMatrix, subject_grads::AbstractMatrix)
    return H_inv * subject_grads
end

"""
    loo_perturbations_cholesky(H_chol, subject_hessians, subject_grads)

Compute leave-one-out parameter perturbations via Cholesky rank-k downdates.

This method exploits the structure of the LOO Hessian:
```math
H_{-i} = H - Hᵢ
```

Rather than inverting H₋ᵢ from scratch, we update the Cholesky factor L (where H = LLᵀ)
by performing rank-1 downdates for each eigencomponent of Hᵢ.

# Algorithm
For each subject i:
1. Eigendecompose: `Hᵢ = VDVᵀ = Σₖ dₖvₖvₖᵀ`
2. For each positive eigenvalue dₖ, perform rank-1 downdate: `LLᵀ - dₖvₖvₖᵀ → L₋ᵢL₋ᵢᵀ`
3. Solve: `L₋ᵢL₋ᵢᵀ Δᵢ = gᵢ`

# Arguments
- `H_chol::Cholesky`: Cholesky factorization of the full Fisher information matrix
- `subject_hessians::Vector{<:AbstractMatrix}`: subject-level Hessian contributions, length n
- `subject_grads::AbstractMatrix`: subject-level score vectors, p × n

# Returns
- `Matrix{Float64}`: p × n matrix where column i is Δᵢ

# Computational Complexity
- O(np³) due to eigendecomposition of each p × p subject Hessian
- More expensive than direct solve when n >> p
- Potentially more stable for ill-conditioned problems

# Fallback Behavior
If the Cholesky downdate fails (matrix becomes indefinite), automatically falls back
to direct solve for that subject.

# When to Use
- When p is large relative to n
- When numerical stability is a concern
- When subject Hessians have low rank (few eigenvalues to update)

See also: [`loo_perturbations_direct`](@ref)
"""
function loo_perturbations_cholesky(H_chol::Cholesky, 
                                    subject_hessians::Vector{<:AbstractMatrix}, 
                                    subject_grads::AbstractMatrix)
    nparams, nsubj = size(subject_grads)
    loo_deltas = Matrix{Float64}(undef, nparams, nsubj)
    
    # Precompute H_inv for fallback
    H_inv = inv(H_chol)
    
    for i in 1:nsubj
        # Get subject i's Hessian contribution (stored as -H_i or Fisher contribution)
        H_i = subject_hessians[i]
        
        # Try Cholesky downdate approach
        success = true
        L_copy = copy(H_chol.L)
        
        try
            # Eigendecompose subject Hessian
            eigen_H = eigen(Symmetric(H_i))
            
            # For each positive eigenvalue, downdate
            for (idx, d) in enumerate(eigen_H.values)
                if d > sqrt(eps(Float64))
                    v = eigen_H.vectors[:, idx]
                    # lowrankdowndate! modifies L in place: L'L - vv' -> L_new' L_new
                    lowrankdowndate!(L_copy, sqrt(d) * v)
                end
            end
            
            # Solve using downdated Cholesky
            L_chol_downdate = Cholesky(L_copy, 'L', 0)
            loo_deltas[:, i] = L_chol_downdate \ subject_grads[:, i]
            
        catch e
            # If downdate fails (e.g., matrix becomes indefinite), fall back to direct
            if e isa PosDefException || e isa LAPACKException
                success = false
            else
                rethrow(e)
            end
        end
        
        if !success
            # Fallback to direct solve
            loo_deltas[:, i] = H_inv * subject_grads[:, i]
        end
    end
    
    return loo_deltas
end

# --- Variance estimators ---

"""
    ij_vcov(H_inv, subject_grads)

Compute the infinitesimal jackknife (IJ) variance-covariance matrix.

# Formula
```math
Var_{IJ}(θ̂) = H⁻¹ K H⁻¹
```
where:
- H is the Hessian (observed Fisher information)
- K = Σᵢ gᵢgᵢᵀ is the outer product of score vectors ("meat" of the sandwich)
- H⁻¹ is the "bread" of the sandwich

# Interpretation
This is also known as the **sandwich** or **robust** or **Huber-White** variance estimator.
It provides valid standard errors even when the model is misspecified, because it does not
assume that Var(gᵢ) = -E[Hᵢ] (which holds only under correct specification).

# Arguments
- `H_inv::AbstractMatrix`: inverse Hessian (variance-covariance matrix from model), p × p
- `subject_grads::AbstractMatrix`: subject-level score vectors, p × n

# Returns
- `Symmetric{Float64, Matrix{Float64}}`: p × p IJ variance-covariance matrix

# Relationship to Other Estimators
- Under correct model specification: IJ variance ≈ model-based variance (H⁻¹)
- Under misspecification: IJ variance captures additional variability
- IJ variance ≈ jackknife variance for large n

# Example
```julia
fitted = fit(model, data; compute_ij_vcov=true)
ij_se = sqrt.(diag(get_ij_vcov(fitted)))  # Robust standard errors
```

See also: [`jk_vcov`](@ref), [`get_ij_vcov`](@ref)
"""
function ij_vcov(H_inv::AbstractMatrix, subject_grads::AbstractMatrix)
    K = subject_grads * subject_grads'  # p × p, sum of outer products
    return Symmetric(H_inv * K * H_inv)
end

"""
    jk_vcov(loo_deltas)

Compute the jackknife variance-covariance matrix from leave-one-out perturbations.

# Formula
```math
Var_{JK}(θ̂) = \frac{n-1}{n} Σᵢ ΔᵢΔᵢᵀ
```
where Δᵢ = θ̂₋ᵢ - θ̂ ≈ H⁻¹gᵢ are the LOO perturbations.

# Interpretation
The jackknife variance can also be written in terms of pseudo-values:
```math
Var_{JK}(θ̂) = \frac{1}{n(n-1)} Σᵢ (θ̃ᵢ - θ̂)(θ̃ᵢ - θ̂)ᵀ
```
where θ̃ᵢ = nθ̂ - (n-1)θ̂₋ᵢ are the jackknife pseudo-values.

# Arguments
- `loo_deltas::AbstractMatrix`: LOO parameter perturbations, p × n

# Returns
- `Symmetric{Float64, Matrix{Float64}}`: p × p jackknife variance-covariance matrix

# Relationship to IJ Variance
```math
Var_{JK}(θ̂) = \frac{n-1}{n} H⁻¹ K H⁻¹ = \frac{n-1}{n} Var_{IJ}(θ̂)
```

The factor (n-1)/n is a finite-sample correction. For large n, JK ≈ IJ.

# Example
```julia
fitted = fit(model, data; compute_jk_vcov=true)
jk_se = sqrt.(diag(get_jk_vcov(fitted)))  # Jackknife standard errors
```

See also: [`ij_vcov`](@ref), [`get_jk_vcov`](@ref)
"""
function jk_vcov(loo_deltas::AbstractMatrix)
    n = size(loo_deltas, 2)
    return Symmetric(((n - 1) / n) * (loo_deltas * loo_deltas'))
end

# --- High-level convenience functions ---

"""
    compute_robust_vcov(params, model, data_or_paths; kwargs...)

Compute model-based and robust variance-covariance estimates.

This is the main entry point for variance estimation, called internally by `fit()`
when robust variance options are requested.

# Arguments
- `params::AbstractVector`: fitted parameter vector (flat, transformed scale)
- `model`: multistate model object
- `data_or_paths`: data container appropriate for model type:
  - `Vector{SamplePath}` for exact observation models
  - `Tuple` (books) for Markov panel models  
  - `(Vector{Vector{SamplePath}}, Vector{Vector{Float64}})` for MCEM models

# Keyword Arguments
- `compute_ij::Bool=true`: compute infinitesimal jackknife (sandwich) variance
- `compute_jk::Bool=false`: compute jackknife variance
- `loo_method::Symbol=:direct`: method for LOO perturbations (`:direct` or `:cholesky`)
- `vcov_threshold::Bool=true`: use adaptive threshold for matrix pseudo-inverse

# Returns
`NamedTuple` with fields:
- `vcov`: model-based variance-covariance matrix (H⁻¹)
- `ij_vcov`: IJ/sandwich variance (H⁻¹KH⁻¹) or `nothing`
- `jk_vcov`: jackknife variance or `nothing`
- `subject_gradients`: p × n matrix of subject scores or `nothing`

# Example
```julia
# Direct call (usually called internally by fit())
result = compute_robust_vcov(params, model, samplepaths; 
                             compute_ij=true, compute_jk=true)

# Access results
model_se = sqrt.(diag(result.vcov))     # Model-based SE
robust_se = sqrt.(diag(result.ij_vcov)) # Robust SE
```

# Notes
- For `:cholesky` method, falls back to `:direct` if Cholesky factorization fails
- Subject gradients are only computed when `compute_ij=true` or `compute_jk=true`

See also: [`ij_vcov`](@ref), [`jk_vcov`](@ref), [`loo_perturbations_direct`](@ref)
"""
function compute_robust_vcov end

function compute_robust_vcov(params::AbstractVector,
                            model::MultistateModel,
                            samplepaths::Vector{SamplePath};
                            compute_ij::Bool = true,
                            compute_jk::Bool = false,
                            loo_method::Symbol = :direct,
                            vcov_threshold::Bool = true,
                            subject_grads::Union{Nothing, Matrix{Float64}} = nothing,
                            subject_hessians::Union{Nothing, Vector{Matrix{Float64}}} = nothing)
    nsubj = length(model.subjectindices)
    nparams = length(params)
    
    # We need subject contributions for IJ/JK
    need_subject_contributions = compute_ij || compute_jk
    
    # Use provided subject contributions if available; otherwise compute them
    if !isnothing(subject_grads) && !isnothing(subject_hessians)
        # Reuse cached gradients and Hessians
        fishinf = zeros(Float64, nparams, nparams)
        for H_i in subject_hessians
            fishinf .-= H_i
        end
        fisher_result = (fishinf = Symmetric(fishinf), 
                        subject_grads = subject_grads, 
                        subject_hessians = subject_hessians)
    else
        # Compute Fisher components
        fisher_result = compute_fisher_components(params, model, samplepaths; 
                                                  compute_subject_contributions=need_subject_contributions)
    end
    
    # Compute standard vcov
    atol = vcov_threshold ? (log(nsubj) * nparams)^-2 : sqrt(eps(Float64))
    vcov = pinv(fisher_result.fishinf, atol=atol)
    vcov[isapprox.(vcov, 0.0; atol=sqrt(eps(Float64)), rtol=sqrt(eps(Float64)))] .= 0.0
    vcov = Symmetric(vcov)
    
    # Compute IJ variance
    ij_variance = nothing
    if compute_ij && !isnothing(fisher_result.subject_grads)
        ij_variance = ij_vcov(vcov, fisher_result.subject_grads)
    end
    
    # Compute JK variance
    jk_variance = nothing
    if compute_jk && !isnothing(fisher_result.subject_grads)
        if loo_method == :cholesky && !isnothing(fisher_result.subject_hessians)
            try
                H_chol = cholesky(fisher_result.fishinf)
                loo_deltas = loo_perturbations_cholesky(H_chol, fisher_result.subject_hessians, 
                                                        fisher_result.subject_grads)
            catch e
                @debug "Cholesky factorization failed for JK variance; falling back to direct solve" exception=(e, catch_backtrace())
                # Fall back to direct if Cholesky fails
                loo_deltas = loo_perturbations_direct(vcov, fisher_result.subject_grads)
            end
        else
            loo_deltas = loo_perturbations_direct(vcov, fisher_result.subject_grads)
        end
        jk_variance = jk_vcov(loo_deltas)
    end
    
    return (vcov = vcov,
            ij_vcov = ij_variance,
            jk_vcov = jk_variance,
            subject_gradients = fisher_result.subject_grads)
end

function compute_robust_vcov(params::AbstractVector,
                            model::MultistateMarkovProcess,
                            books::Tuple;
                            compute_ij::Bool = true,
                            compute_jk::Bool = false,
                            loo_method::Symbol = :direct,
                            vcov_threshold::Bool = true,
                            subject_grads::Union{Nothing, Matrix{Float64}} = nothing,
                            subject_hessians::Union{Nothing, Vector{Matrix{Float64}}} = nothing)
    nsubj = length(model.subjectindices)
    nparams = length(params)
    
    # We need subject contributions for IJ/JK
    need_subject_contributions = compute_ij || compute_jk
    
    # Use provided subject contributions if available; otherwise compute them
    if !isnothing(subject_grads) && !isnothing(subject_hessians)
        # Reuse cached gradients and Hessians
        fishinf = zeros(Float64, nparams, nparams)
        for H_i in subject_hessians
            fishinf .-= H_i
        end
        fisher_result = (fishinf = Symmetric(fishinf), 
                        subject_grads = subject_grads, 
                        subject_hessians = subject_hessians)
    else
        # Compute Fisher components
        fisher_result = compute_fisher_components(params, model, books;
                                                  compute_subject_contributions=need_subject_contributions)
    end
    
    # Compute standard vcov
    atol = vcov_threshold ? (log(nsubj) * nparams)^-2 : sqrt(eps(Float64))
    vcov = pinv(fisher_result.fishinf, atol=atol)
    vcov[isapprox.(vcov, 0.0; atol=eps(Float64))] .= 0.0
    vcov = Symmetric(vcov)
    
    # Compute IJ variance
    ij_variance = nothing
    if compute_ij && !isnothing(fisher_result.subject_grads)
        ij_variance = ij_vcov(vcov, fisher_result.subject_grads)
    end
    
    # Compute JK variance
    jk_variance = nothing
    if compute_jk && !isnothing(fisher_result.subject_grads)
        if loo_method == :cholesky && !isnothing(fisher_result.subject_hessians)
            try
                H_chol = cholesky(fisher_result.fishinf)
                loo_deltas = loo_perturbations_cholesky(H_chol, fisher_result.subject_hessians,
                                                        fisher_result.subject_grads)
            catch e
                @debug "Cholesky factorization failed for JK variance; falling back to direct solve" exception=(e, catch_backtrace())
                loo_deltas = loo_perturbations_direct(vcov, fisher_result.subject_grads)
            end
        else
            loo_deltas = loo_perturbations_direct(vcov, fisher_result.subject_grads)
        end
        jk_variance = jk_vcov(loo_deltas)
    end
    
    return (vcov = vcov,
            ij_vcov = ij_variance,
            jk_vcov = jk_variance,
            subject_gradients = fisher_result.subject_grads)
end

"""
    compute_robust_vcov(params, model::MultistateSemiMarkovProcess,
                       samplepaths, ImportanceWeights; kwargs...)

Compute robust variance estimates for semi-Markov models fitted via MCEM.

For models fitted with Monte Carlo EM, the subject-level scores and Fisher information
are computed using Louis's identity, which relates complete-data quantities to 
observed-data quantities:

```math
I^{obs} = E[I^{comp}|Y] - Var[S^{comp}|Y]
```

The subject-level gradient for the IJ estimator is the importance-weighted score:
```math
gᵢ = Σⱼ wᵢⱼ · gᵢⱼ^{comp}
```

where wᵢⱼ are the normalized importance weights and gᵢⱼ^{comp} is the complete-data
score for path j of subject i.

## Monte Carlo Considerations

Unlike exact data models, the gradients and Hessian are estimated from Monte Carlo samples.
This introduces additional variability, but for large numbers of Monte Carlo paths (typical
in MCEM, e.g., 1000+), this variability is negligible relative to sampling variability.

The IJ/sandwich estimator remains valid because:
1. The importance weights are self-normalized (sum to 1 per subject)
2. We estimate E[gᵢ|Yᵢ], not the point estimate itself
3. Louis's identity provides the correct expected Fisher information

# Arguments
- `params::AbstractVector`: parameter vector (flat, on transformed scale)
- `model::MultistateSemiMarkovProcess`: fitted semi-Markov model
- `samplepaths::Vector{Vector{SamplePath}}`: sampled paths for each subject
- `ImportanceWeights::Vector{Vector{Float64}}`: normalized importance weights

# Keyword Arguments
- `compute_ij::Bool = true`: compute infinitesimal jackknife (sandwich) variance
- `compute_jk::Bool = false`: compute jackknife variance
- `loo_method::Symbol = :direct`: method for LOO perturbations (`:direct` or `:cholesky`)
- `vcov_threshold::Bool = true`: use adaptive threshold for pseudo-inverse

# Returns
NamedTuple with fields:
- `vcov`: model-based variance-covariance matrix (inverse Fisher information)
- `ij_vcov`: IJ/sandwich variance matrix (robust to misspecification)
- `jk_vcov`: jackknife variance matrix (if requested)
- `subject_gradients`: p × n matrix of subject-level scores

# References
- Louis, T. A. (1982). Finding the observed information matrix when using the EM algorithm.
  Journal of the Royal Statistical Society: Series B, 44(2), 226-233.
- Morsomme, R. et al. (2025). MultistateModels.jl technical supplement.
"""
function compute_robust_vcov(params::AbstractVector,
                            model::MultistateSemiMarkovProcess,
                            samplepaths::Vector{Vector{SamplePath}},
                            ImportanceWeights::Vector{Vector{Float64}};
                            compute_ij::Bool = true,
                            compute_jk::Bool = false,
                            loo_method::Symbol = :direct,
                            vcov_threshold::Bool = true,
                            subject_grads::Union{Nothing, Matrix{Float64}} = nothing,
                            subject_hessians::Union{Nothing, Vector{Matrix{Float64}}} = nothing)
    nsubj = length(model.subjectindices)
    nparams = length(params)
    
    # We need subject contributions for IJ/JK
    need_subject_contributions = compute_ij || compute_jk
    
    # Use provided subject contributions if available; otherwise compute them
    if !isnothing(subject_grads) && !isnothing(subject_hessians)
        # Reuse cached gradients and Hessians
        fisher_result = compute_fisher_components(params, model, samplepaths, ImportanceWeights;
                                                  compute_subject_contributions=false)
        fisher_result = (fishinf = fisher_result.fishinf,
                        subject_grads = subject_grads,
                        subject_hessians = subject_hessians)
    else
        # Compute Fisher components using Louis's identity
        fisher_result = compute_fisher_components(params, model, samplepaths, ImportanceWeights;
                                                  compute_subject_contributions=need_subject_contributions)
    end
    
    # Compute standard vcov (model-based, from Louis's Fisher information)
    atol = vcov_threshold ? (log(nsubj) * nparams)^-2 : sqrt(eps(Float64))
    vcov = pinv(fisher_result.fishinf, atol=atol)
    vcov[isapprox.(vcov, 0.0; atol=sqrt(eps(Float64)), rtol=sqrt(eps(Float64)))] .= 0.0
    vcov = Symmetric(vcov)
    
    # Compute IJ variance (sandwich estimator)
    ij_variance = nothing
    if compute_ij && !isnothing(fisher_result.subject_grads)
        ij_variance = ij_vcov(vcov, fisher_result.subject_grads)
    end
    
    # Compute JK variance
    jk_variance = nothing
    if compute_jk && !isnothing(fisher_result.subject_grads)
        if loo_method == :cholesky && !isnothing(fisher_result.subject_hessians)
            try
                H_chol = cholesky(fisher_result.fishinf)
                loo_deltas = loo_perturbations_cholesky(H_chol, fisher_result.subject_hessians,
                                                        fisher_result.subject_grads)
            catch e
                @debug "Cholesky factorization failed for JK variance; falling back to direct solve" exception=(e, catch_backtrace())
                # Fall back to direct if Cholesky fails
                loo_deltas = loo_perturbations_direct(vcov, fisher_result.subject_grads)
            end
        else
            loo_deltas = loo_perturbations_direct(vcov, fisher_result.subject_grads)
        end
        jk_variance = jk_vcov(loo_deltas)
    end
    
    return (vcov = vcov,
            ij_vcov = ij_variance,
            jk_vcov = jk_variance,
            subject_gradients = fisher_result.subject_grads)
end


# ============================================================================
# NCV: Neighbourhood Cross-Validation
# ============================================================================
# 
# Implementation of Neighbourhood Cross-Validation (NCV) for smoothing parameter
# selection in penalized likelihood models, following:
# 
# Wood, S.N. (2024). Neighbourhood Cross Validation. arXiv:2404.16490
# 
# Key equations:
#   V = Σₖ Σᵢ∈δ(k) D(yᵢ, θᵢ^{-α(k)})                    (Equation 2)
#   Δ_{-α(i)} = H_{λ,α(i)}^{-1} g_{α(i)}                  (Equation 3)
#   H_{λ,α(i)} = H_λ - H_{α(i),α(i)}
#
# Where:
#   - δ(k) is neighbourhood k (here, subjects/neighbourhoods)
#   - α(k) is the set of indices for neighbourhood k
#   - D(y, θ) is the deviance contribution for observation y at parameter θ
#   - H_λ is the penalized Hessian
#   - g_{α(i)} is the gradient contribution from neighbourhood i
#   - H_{α(i),α(i)} is the Hessian contribution from neighbourhood i
#
# Degeneracy protections (Section 2.1 and 4.1):
#   1. Cholesky downdate with indefiniteness detection
#   2. Woodbury identity fallback for indefinite cases
#   3. Quadratic approximation V_q for finite deviance robustness
#   4. High smoothing parameter detection via near-zero quadratic form
# ============================================================================

"""
    NCVState

Mutable state object for Neighbourhood Cross-Validation computations.

Stores the penalized Hessian factorization and subject/neighbourhood-level 
quantities needed for efficient NCV criterion computation.

# Fields
- `H_lambda::Matrix{Float64}`: Penalized Hessian H_λ = -∂²ℓ/∂β² + Sλ
- `H_chol::Union{Nothing, Cholesky{Float64,Matrix{Float64}}}`: Cholesky of H_λ
- `subject_grads::Matrix{Float64}`: p × n matrix of neighbourhood gradients g_{α(k)}
- `subject_hessians::Union{Nothing, Array{Float64,3}}`: p × p × n neighbourhood Hessians
- `deltas::Matrix{Float64}`: p × n LOO perturbations Δ_{-α(k)}
- `indefinite_flags::BitVector`: Flags for neighbourhoods with indefinite H_{λ,α(k)}
- `penalty_matrix::Union{Nothing, Matrix{Float64}}`: Penalty matrix Sλ (optional)
- `log_smoothing_params::Union{Nothing, Vector{Float64}}`: log(λ) for each penalty

# Reference
Wood, S.N. (2024). Neighbourhood Cross Validation. arXiv:2404.16490
"""
mutable struct NCVState
    H_lambda::Matrix{Float64}
    H_chol::Union{Nothing, Cholesky{Float64,Matrix{Float64}}}
    subject_grads::Matrix{Float64}
    subject_hessians::Union{Nothing, Array{Float64,3}}
    deltas::Matrix{Float64}
    indefinite_flags::BitVector
    penalty_matrix::Union{Nothing, Matrix{Float64}}
    log_smoothing_params::Union{Nothing, Vector{Float64}}
end

"""
    NCVState(H_lambda, subject_grads; subject_hessians=nothing, 
             penalty_matrix=nothing, log_smoothing_params=nothing)

Initialize an NCV state from penalized Hessian and neighbourhood gradients.

# Arguments
- `H_lambda::AbstractMatrix`: Penalized Hessian H_λ = -∂²ℓ/∂β² + Sλ, p × p
- `subject_grads::AbstractMatrix`: Neighbourhood gradient contributions, p × n
- `subject_hessians::Union{Nothing, AbstractArray}=nothing`: p × p × n Hessian contributions
- `penalty_matrix::Union{Nothing, AbstractMatrix}=nothing`: Penalty matrix Sλ
- `log_smoothing_params::Union{Nothing, AbstractVector}=nothing`: log(λ) values

# Returns
- `NCVState`: Initialized state with Cholesky factorization attempted

# Example
```julia
# From fitted model with splines
state = NCVState(penalized_hessian, subject_grads;
                 subject_hessians=subj_hess,
                 penalty_matrix=S_lambda)
```

# Notes
The Cholesky factorization is attempted on initialization. If it fails (indicating
H_λ is not positive definite), `H_chol` will be `nothing` and subsequent computations
will use direct methods.
"""
function NCVState(H_lambda::AbstractMatrix, 
                  subject_grads::AbstractMatrix;
                  subject_hessians::Union{Nothing, AbstractArray}=nothing,
                  penalty_matrix::Union{Nothing, AbstractMatrix}=nothing,
                  log_smoothing_params::Union{Nothing, AbstractVector}=nothing)
    
    nparams, nsubj = size(subject_grads)
    
    # Convert to concrete types
    H_mat = Matrix{Float64}(H_lambda)
    grads_mat = Matrix{Float64}(subject_grads)
    
    hess_arr = if isnothing(subject_hessians)
        nothing
    else
        Array{Float64,3}(subject_hessians)
    end
    
    pen_mat = isnothing(penalty_matrix) ? nothing : Matrix{Float64}(penalty_matrix)
    log_sp = isnothing(log_smoothing_params) ? nothing : Vector{Float64}(log_smoothing_params)
    
    # Attempt Cholesky factorization
    H_chol = try
        cholesky(Symmetric(H_mat))
    catch e
        @debug "Cholesky factorization failed during NCVState initialization; NCV will use direct solves" exception=(e, catch_backtrace())
        nothing
    end
    
    # Initialize deltas and flags
    deltas = zeros(Float64, nparams, nsubj)
    indefinite_flags = falses(nsubj)
    
    return NCVState(H_mat, H_chol, grads_mat, hess_arr, deltas, 
                    indefinite_flags, pen_mat, log_sp)
end

"""
    cholesky_downdate!(L, v; tol=1e-10)

Perform rank-1 downdate of Cholesky factor: L'L → L̃'L̃ where L̃'L̃ = L'L - vv'.

Uses the standard sequential downdate algorithm with Givens rotations.
Returns `true` if successful, `false` if the matrix becomes indefinite.

# Arguments
- `L::AbstractMatrix`: Lower triangular Cholesky factor (modified in place)
- `v::AbstractVector`: Vector for rank-1 downdate
- `tol::Float64=1e-10`: Tolerance for indefiniteness detection

# Returns
- `Bool`: `true` if downdate succeeded, `false` if indefinite

# Algorithm
For each column j, compute:
```
r = √(L[j,j]² - v[j]²)   # May be imaginary if indefinite
c = r / L[j,j]
s = v[j] / L[j,j]
L[j,j] = r
```
Then apply Givens rotation to remaining elements.

# Reference
Seeger, M. (2004). Low Rank Updates for the Cholesky Decomposition.
Wood, S.N. (2024). Neighbourhood Cross Validation, Section 2.1.
"""
function cholesky_downdate!(L::AbstractMatrix, v::AbstractVector; tol::Float64=1e-10)
    n = size(L, 1)
    w = copy(v)
    
    for j in 1:n
        r² = L[j,j]^2 - w[j]^2
        
        # Check for indefiniteness
        if r² < tol
            return false
        end
        
        r = sqrt(r²)
        c = r / L[j,j]
        s = w[j] / L[j,j]
        
        L[j,j] = r
        
        # Update remaining elements in column j and vector w
        for i in (j+1):n
            L[i,j] = (L[i,j] - s * w[i]) / c
            w[i] = c * w[i] - s * L[i,j]
        end
    end
    
    return true
end

"""
    cholesky_downdate_copy(L, v; tol=1e-10)

Non-mutating version of cholesky_downdate! that returns a new matrix.

# Arguments
- `L::AbstractMatrix`: Lower triangular Cholesky factor
- `v::AbstractVector`: Vector for rank-1 downdate
- `tol::Float64=1e-10`: Tolerance for indefiniteness detection

# Returns
- `Tuple{Matrix{Float64}, Bool}`: (L̃, success) where L̃ is the updated factor

# Example
```julia
L = cholesky(Symmetric(H)).L
L_new, success = cholesky_downdate_copy(L, v)
if success
    # Use L_new
else
    # Fall back to Woodbury
end
```
"""
function cholesky_downdate_copy(L::AbstractMatrix, v::AbstractVector; tol::Float64=1e-10)
    L_copy = copy(L)
    success = cholesky_downdate!(L_copy, v; tol=tol)
    return L_copy, success
end

"""
    ncv_loo_perturbation_cholesky(H_chol, H_k, g_k; tol=1e-10)

Compute LOO perturbation Δ_{-α(k)} using Cholesky downdate.

For neighbourhood k with Hessian contribution H_k and gradient g_k:
```math
Δ_{-α(k)} = (H_λ - H_k)^{-1} g_k = H_{λ,α(k)}^{-1} g_k
```

Uses efficient Cholesky downdate when H_k has low rank structure, 
with Woodbury fallback for indefinite cases.

# Arguments
- `H_chol::Cholesky`: Cholesky factorization of H_λ
- `H_k::AbstractMatrix`: Neighbourhood Hessian contribution H_{α(k),α(k)}
- `g_k::AbstractVector`: Neighbourhood gradient g_{α(k)}
- `tol::Float64=1e-10`: Indefiniteness tolerance

# Returns
NamedTuple with:
- `delta::Vector{Float64}`: LOO perturbation Δ_{-α(k)}
- `indefinite::Bool`: Whether H_{λ,α(k)} was indefinite

# Method Selection
1. If H_k is rank-1 or low-rank: Use sequential Cholesky downdate
2. If downdate fails (indefinite): Use Woodbury identity fallback
3. If Woodbury fails: Use direct solve (most expensive)

# Reference
Wood, S.N. (2024). Neighbourhood Cross Validation, Equation 3 and Section 2.1.
"""
function ncv_loo_perturbation_cholesky(H_chol::Cholesky,
                                       H_k::AbstractMatrix,
                                       g_k::AbstractVector;
                                       tol::Float64=1e-10)
    p = length(g_k)
    
    # Try eigendecomposition of H_k for low-rank update
    H_k_sym = Symmetric(H_k)
    eig = eigen(H_k_sym)
    
    # Filter significant eigenvalues
    sig_idx = findall(x -> abs(x) > tol * maximum(abs.(eig.values)), eig.values)
    
    if isempty(sig_idx)
        # H_k ≈ 0, just use H_λ^{-1}
        delta = H_chol \ g_k
        return (delta=delta, indefinite=false)
    end
    
    # Try sequential downdates for each significant eigenvector
    L = copy(H_chol.L)
    success = true
    
    for i in sig_idx
        λi = eig.values[i]
        vi = eig.vectors[:, i]
        
        if λi > 0
            # Downdate: H_{λ,α(k)} = H_λ - H_k, and H_k has positive eigenvalue
            # This is a downdate of H_λ by λi * vi * vi'
            success = cholesky_downdate!(L, sqrt(λi) * vi; tol=tol)
        else
            # Negative eigenvalue means we're adding to the matrix
            # This is a rank-1 update, use standard update
            # For now, recompute (this is the rare case)
            success = false
        end
        
        if !success
            break
        end
    end
    
    if success
        # Solve with downdated Cholesky
        y = L \ g_k
        delta = L' \ y
        return (delta=delta, indefinite=false)
    else
        # Fall back to Woodbury identity
        return ncv_loo_perturbation_woodbury(H_chol, H_k, g_k)
    end
end

"""
    ncv_loo_perturbation_woodbury(H_chol, H_k, g_k)

Compute LOO perturbation using Woodbury matrix identity.

When direct Cholesky downdate fails due to indefiniteness, use:
```math
(H_λ - H_k)^{-1} = H_λ^{-1} + H_λ^{-1} H_k (I - H_λ^{-1} H_k)^{-1} H_λ^{-1}
```

Simplified for the gradient computation:
```math
Δ_{-α(k)} = H_λ^{-1} g_k + H_λ^{-1} H_k (I - H_λ^{-1} H_k)^{-1} H_λ^{-1} g_k
```

# Arguments
- `H_chol::Cholesky`: Cholesky factorization of H_λ
- `H_k::AbstractMatrix`: Neighbourhood Hessian contribution
- `g_k::AbstractVector`: Neighbourhood gradient

# Returns
NamedTuple with:
- `delta::Vector{Float64}`: LOO perturbation
- `indefinite::Bool`: Always `true` (indicates Woodbury was used)

# Reference
Wood, S.N. (2024). Neighbourhood Cross Validation, Equation 4.
"""
function ncv_loo_perturbation_woodbury(H_chol::Cholesky,
                                       H_k::AbstractMatrix,
                                       g_k::AbstractVector)
    # Compute H_λ^{-1} g_k
    H_inv_g = H_chol \ g_k
    
    # Compute H_λ^{-1} H_k
    H_inv_Hk = H_chol \ H_k
    
    # Compute (I - H_λ^{-1} H_k)
    p = length(g_k)
    M = I - H_inv_Hk
    
    # Solve for correction term
    # (I - H_λ^{-1} H_k)^{-1} H_λ^{-1} g_k
    try
        correction = M \ H_inv_g
        delta = H_inv_g + H_inv_Hk * correction
        return (delta=delta, indefinite=true)
    catch e1
        @debug "Woodbury formula failed; trying direct solve" exception=(e1, catch_backtrace())
        # If Woodbury also fails, fall back to direct solve
        H_lambda_k = H_chol.L * H_chol.U - H_k
        try
            delta = Symmetric(H_lambda_k) \ g_k
            return (delta=delta, indefinite=true)
        catch e2
            @debug "Direct solve also failed; returning NaN" exception=(e2, catch_backtrace())
            # Complete failure - return NaN to signal problem
            return (delta=fill(NaN, length(g_k)), indefinite=true)
        end
    end
end

"""
    ncv_loo_perturbation_direct(H_lambda, H_k, g_k)

Compute LOO perturbation by direct solve (no Cholesky).

# Arguments
- `H_lambda::AbstractMatrix`: Full penalized Hessian
- `H_k::AbstractMatrix`: Neighbourhood Hessian contribution
- `g_k::AbstractVector`: Neighbourhood gradient

# Returns
NamedTuple with:
- `delta::Vector{Float64}`: LOO perturbation
- `indefinite::Bool`: Whether solve required regularization
"""
function ncv_loo_perturbation_direct(H_lambda::AbstractMatrix,
                                     H_k::AbstractMatrix,
                                     g_k::AbstractVector)
    H_lambda_k = H_lambda - H_k
    
    try
        delta = Symmetric(H_lambda_k) \ g_k
        return (delta=delta, indefinite=false)
    catch e1
        @debug "Direct solve failed; adding ridge regularization" exception=(e1, catch_backtrace())
        # Matrix singular or indefinite - try with regularization
        try
            # Add small ridge for stability
            p = length(g_k)
            ridge = 1e-8 * I(p)
            delta = (Symmetric(H_lambda_k) + ridge) \ g_k
            return (delta=delta, indefinite=true)
        catch e2
            @debug "Ridge-regularized solve also failed; returning NaN" exception=(e2, catch_backtrace())
            return (delta=fill(NaN, length(g_k)), indefinite=true)
        end
    end
end

"""
    compute_ncv_perturbations!(state::NCVState)

Compute all LOO perturbations Δ_{-α(k)} for each neighbourhood.

Updates `state.deltas` and `state.indefinite_flags` in place.

# Arguments
- `state::NCVState`: NCV state with H_λ and neighbourhood quantities

# Returns
- `state`: The modified NCVState

# Algorithm
For each neighbourhood k = 1, ..., n:
1. Extract H_k (neighbourhood Hessian) and g_k (neighbourhood gradient)
2. Compute Δ_{-α(k)} = H_{λ,α(k)}^{-1} g_k using best available method
3. Mark indefinite flag if Woodbury or direct solve was needed

# Notes
Uses efficient Cholesky downdate when H_chol is available and H_k allows.
Falls back to Woodbury identity or direct solve as needed.
"""
function compute_ncv_perturbations!(state::NCVState)
    nsubj = size(state.subject_grads, 2)
    
    # Check if we have subject Hessians
    have_hessians = !isnothing(state.subject_hessians)
    
    # Check if we have Cholesky factorization
    have_chol = !isnothing(state.H_chol)
    
    for k in 1:nsubj
        g_k = @view state.subject_grads[:, k]
        
        if have_hessians
            H_k = @view state.subject_hessians[:, :, k]
        else
            # Without individual Hessians, approximate with outer product (rank-1)
            H_k = g_k * g_k'
        end
        
        result = if have_chol
            ncv_loo_perturbation_cholesky(state.H_chol, H_k, g_k)
        else
            ncv_loo_perturbation_direct(state.H_lambda, H_k, g_k)
        end
        
        state.deltas[:, k] .= result.delta
        state.indefinite_flags[k] = result.indefinite
    end
    
    return state
end

"""
    ncv_criterion(state::NCVState, params::AbstractVector, 
                  loss_fn::Function, data; use_quadratic=false)

Compute the NCV criterion (approximate leave-neighbourhood-out cross-validation loss).

# Arguments
- `state::NCVState`: NCV state with computed perturbations
- `params::AbstractVector`: Current parameter estimates β̂
- `loss_fn::Function`: Function `loss_fn(params, data, k)` returning loss for neighbourhood k
- `data`: Data object passed to loss function
- `use_quadratic::Bool=false`: Use quadratic approximation V_q for robustness

# Returns
- `Float64`: NCV criterion V = (1/n) Σₖ D(y_{α(k)}, β̂ + Δ_{-α(k)})

# Quadratic Approximation
When `use_quadratic=true`, uses:
```math
V_q = (1/n) Σₖ [D_k(β̂) + g_k'Δ_{-α(k)} + (1/2)Δ_{-α(k)}'H_k Δ_{-α(k)}]
```

This is more robust when some neighbourhoods have extreme perturbations that
would cause numerical issues in the exact loss evaluation.

# Reference
Wood, S.N. (2024). Neighbourhood Cross Validation, Equations 2 and 5.
"""
function ncv_criterion(state::NCVState,
                       params::AbstractVector,
                       loss_fn::Function,
                       data;
                       use_quadratic::Bool=false)
    nsubj = size(state.deltas, 2)
    
    if use_quadratic
        return ncv_criterion_quadratic(state, params, loss_fn, data)
    end
    
    V = 0.0
    for k in 1:nsubj
        # LOO parameter estimate for neighbourhood k
        params_loo = params .+ @view(state.deltas[:, k])
        
        # Evaluate loss at LOO estimate
        D_k = loss_fn(params_loo, data, k)
        
        # Check for numerical issues
        if !isfinite(D_k)
            # Fall back to quadratic approximation for this neighbourhood
            D_k = ncv_loss_quadratic_k(state, params, loss_fn, data, k)
        end
        
        V += D_k
    end
    
    return V / nsubj
end

"""
    ncv_criterion_quadratic(state::NCVState, params::AbstractVector,
                            loss_fn::Function, data)

Compute NCV criterion using quadratic approximation (Equation 5 in Wood 2024).

More robust than exact evaluation when perturbations are large.
```math
V_q = (1/n) Σₖ [D_k(β̂) + g_k'Δ_{-α(k)} + (1/2)Δ_{-α(k)}'H_k Δ_{-α(k)}]
```
"""
function ncv_criterion_quadratic(state::NCVState,
                                 params::AbstractVector,
                                 loss_fn::Function,
                                 data)
    nsubj = size(state.deltas, 2)
    have_hessians = !isnothing(state.subject_hessians)
    
    V_q = 0.0
    for k in 1:nsubj
        V_q += ncv_loss_quadratic_k(state, params, loss_fn, data, k)
    end
    
    return V_q / nsubj
end

"""
    ncv_loss_quadratic_k(state, params, loss_fn, data, k)

Quadratic approximation of LOO loss for neighbourhood k.
"""
function ncv_loss_quadratic_k(state::NCVState,
                              params::AbstractVector,
                              loss_fn::Function,
                              data, k::Int)
    have_hessians = !isnothing(state.subject_hessians)
    
    # Loss at current estimate
    D_k = loss_fn(params, data, k)
    
    # Gradient contribution
    g_k = @view state.subject_grads[:, k]
    delta_k = @view state.deltas[:, k]
    
    grad_term = dot(g_k, delta_k)
    
    # Hessian contribution
    hess_term = if have_hessians
        H_k = @view state.subject_hessians[:, :, k]
        0.5 * dot(delta_k, H_k * delta_k)
    else
        # Approximate with gradient outer product
        0.5 * dot(g_k, delta_k)^2
    end
    
    return D_k + grad_term + hess_term
end

"""
    ncv_criterion_derivatives(state::NCVState, params::AbstractVector,
                              loss_fn::Function, grad_loss_fn::Function, data, 
                              penalty_derivatives)

Compute NCV criterion and its derivatives with respect to log smoothing parameters.

# Arguments
- `state::NCVState`: NCV state with computed perturbations
- `params::AbstractVector`: Current parameter estimates β̂
- `loss_fn::Function`: Function `loss_fn(params, data, k)` returning loss
- `grad_loss_fn::Function`: Function returning gradient of loss w.r.t. params
- `data`: Data object
- `penalty_derivatives`: Vector of penalty matrix derivatives ∂S/∂ρⱼ

# Returns
NamedTuple with:
- `V`: NCV criterion value
- `dV_drho`: Vector of ∂V/∂ρⱼ derivatives

# Derivative Computation
From Wood (2024) Section 3, the derivative of V w.r.t. log(λⱼ) = ρⱼ is:
```math
∂V/∂ρⱼ = Σₖ D_k'(β̂_{-α(k)}) ∂β̂_{-α(k)}/∂ρⱼ
```

where:
```math
∂β̂_{-α(k)}/∂ρⱼ = -H_{λ,α(k)}^{-1} (∂S_λ/∂ρⱼ) β̂_{-α(k)}
```

# Degeneracy Test
Monitors ∂β̂/∂ρⱼ · H_λ · ∂β̂/∂ρⱼ ≈ 0 which indicates smoothing parameter
is too large and has effectively removed those basis functions.

# Reference
Wood, S.N. (2024). Neighbourhood Cross Validation, Section 3.
"""
function ncv_criterion_derivatives(state::NCVState,
                                   params::AbstractVector,
                                   loss_fn::Function,
                                   grad_loss_fn::Function,
                                   data,
                                   penalty_derivatives::AbstractVector)
    nsubj = size(state.deltas, 2)
    n_smoothing = length(penalty_derivatives)
    
    # Compute NCV criterion
    V = ncv_criterion(state, params, loss_fn, data)
    
    # Initialize derivative accumulator
    dV_drho = zeros(Float64, n_smoothing)
    
    have_chol = !isnothing(state.H_chol)
    
    for k in 1:nsubj
        # LOO parameter estimate
        params_loo = params .+ @view(state.deltas[:, k])
        
        # Gradient of loss w.r.t. parameters at LOO estimate
        D_prime_k = grad_loss_fn(params_loo, data, k)
        
        for j in 1:n_smoothing
            dS_drho_j = penalty_derivatives[j]
            
            # Compute ∂β̂_{-α(k)}/∂ρⱼ = -H_{λ,α(k)}^{-1} (∂S/∂ρⱼ) β̂_{-α(k)}
            # First compute (∂S/∂ρⱼ) β̂_{-α(k)}
            Sj_params_loo = dS_drho_j * params_loo
            
            # Then solve H_{λ,α(k)} x = Sj_params_loo
            # We can reuse the LOO perturbation machinery
            H_k = if !isnothing(state.subject_hessians)
                @view state.subject_hessians[:, :, k]
            else
                g_k = @view state.subject_grads[:, k]
                g_k * g_k'
            end
            
            if have_chol
                result = ncv_loo_perturbation_cholesky(state.H_chol, H_k, Sj_params_loo)
            else
                result = ncv_loo_perturbation_direct(state.H_lambda, H_k, Sj_params_loo)
            end
            
            d_params_loo_drho_j = -result.delta
            
            # Accumulate derivative
            dV_drho[j] += dot(D_prime_k, d_params_loo_drho_j)
        end
    end
    
    # Average over neighbourhoods
    dV_drho ./= nsubj
    
    return (V=V, dV_drho=dV_drho)
end

"""
    ncv_degeneracy_test(state::NCVState, params::AbstractVector, 
                        penalty_derivative::AbstractMatrix; tol=1e-6)

Test for degeneracy in smoothing parameter (Section 3 of Wood 2024).

A smoothing parameter λⱼ is degenerate if:
```math
(∂β̂/∂ρⱼ)' H_λ (∂β̂/∂ρⱼ) ≈ 0
```

This occurs when λⱼ is so large that the corresponding basis functions
have been effectively removed from the model.

# Arguments
- `state::NCVState`: NCV state
- `params::AbstractVector`: Current parameters
- `penalty_derivative::AbstractMatrix`: ∂S/∂ρⱼ for the smoothing parameter
- `tol::Float64=1e-6`: Tolerance for degeneracy detection

# Returns
- `Bool`: `true` if the smoothing parameter is degenerate
"""
function ncv_degeneracy_test(state::NCVState,
                             params::AbstractVector,
                             penalty_derivative::AbstractMatrix;
                             tol::Float64=1e-6)
    # Compute ∂β̂/∂ρⱼ = -H_λ^{-1} (∂S/∂ρⱼ) β̂
    Sj_params = penalty_derivative * params
    
    d_params_drho = if !isnothing(state.H_chol)
        -(state.H_chol \ Sj_params)
    else
        -(Symmetric(state.H_lambda) \ Sj_params)
    end
    
    # Compute quadratic form
    quad_form = dot(d_params_drho, state.H_lambda * d_params_drho)
    
    # Compare to expected magnitude
    expected_mag = dot(params, state.H_lambda * params)
    
    return abs(quad_form) < tol * max(expected_mag, 1.0)
end

"""
    ncv_get_loo_estimates(state::NCVState, params::AbstractVector)

Get leave-neighbourhood-out parameter estimates from NCV state.

# Arguments
- `state::NCVState`: NCV state with computed perturbations
- `params::AbstractVector`: Current (full-data) parameter estimates, length p

# Returns
- `Matrix{Float64}`: p × n matrix where column k is β̂_{-α(k)} = β̂ + Δ_{-α(k)}
"""
function ncv_get_loo_estimates(state::NCVState, params::AbstractVector)
    return params .+ state.deltas
end

"""
    ncv_get_perturbations(state::NCVState)

Get the LOO perturbations Δ_{-α(k)} from NCV state.

# Arguments
- `state::NCVState`: NCV state with computed perturbations

# Returns
- `Matrix{Float64}`: p × n matrix where column k is Δ_{-α(k)} = β̂_{-α(k)} - β̂
"""
function ncv_get_perturbations(state::NCVState)
    return state.deltas
end

"""
    ncv_vcov(state::NCVState)

Compute variance estimates from NCV state perturbations.

Returns both IJ and jackknife variance estimates based on the LOO perturbations,
analogous to the standard IJ/JK variance estimators.

# Arguments
- `state::NCVState`: NCV state with computed perturbations

# Returns
NamedTuple with:
- `ij_vcov`: Infinitesimal jackknife variance
- `jk_vcov`: Jackknife variance

# Notes
These variance estimates account for the penalization structure through the
perturbations, which are computed from the penalized Hessian H_λ.
"""
function ncv_vcov(state::NCVState)
    n = size(state.deltas, 2)
    delta_outer = state.deltas * state.deltas'
    
    return (
        ij_vcov = Symmetric(delta_outer / n),
        jk_vcov = Symmetric(((n - 1) / n) * delta_outer)
    )
end

# --- Variance Comparison Diagnostics ---

"""
    compare_variance_estimates(fitted; use_ij = true, threshold = 1.5, verbose = true)

Compare model-based and robust standard errors to diagnose potential model misspecification.

This diagnostic compares the standard errors from two variance estimation methods:
- **Model-based** (Fisher information inverse): Valid under correct model specification
- **Robust** (IJ/sandwich): Valid even under model misspecification

The ratio `SE_robust / SE_model` provides insight into model adequacy:
- **Ratio ≈ 1**: Model appears correctly specified
- **Ratio > 1**: Model may be misspecified (robust SEs are larger, conservative)
- **Ratio < 1**: Unusual, may indicate numerical issues or very efficient estimation

# Arguments
- `fitted::MultistateModelFitted`: fitted model object with variance estimates
- `use_ij::Bool=true`: if true, use IJ (sandwich) variance; if false, use jackknife
- `threshold::Float64=1.5`: ratio above which to flag potential misspecification
- `verbose::Bool=true`: print detailed comparison table

# Returns
NamedTuple with:
- `ratio`: Vector of SE ratios (robust/model) for each parameter
- `model_se`: Vector of model-based standard errors
- `robust_se`: Vector of robust standard errors
- `mean_ratio`: Mean ratio across parameters
- `max_ratio`: Maximum ratio across parameters
- `flagged`: Indices of parameters where ratio exceeds threshold
- `diagnosis`: String summary of model specification assessment

# Example
```julia
# Fit model with both variance estimates
fitted = fit(model; compute_vcov=true, compute_ij_vcov=true)

# Compare variance estimates
result = compare_variance_estimates(fitted)

# Check for potential misspecification
if length(result.flagged) > 0
    println("Parameters with SE ratio > 1.5: ", result.flagged)
    println("Consider using robust (IJ) standard errors for inference")
end
```

# Interpretation Guide

| Mean Ratio | Interpretation |
|------------|----------------|
| 0.9 - 1.1  | Model appears well-specified |
| 1.1 - 1.5  | Minor misspecification possible |
| 1.5 - 2.0  | Moderate misspecification; use robust SEs |
| > 2.0      | Substantial misspecification; reconsider model |

# Notes
- Robust SEs should be used for inference when misspecification is suspected
- Model-based SEs remain useful as the Cramér-Rao lower bound
- This diagnostic is based on the sandwich estimator consistency property
- NaN ratios may occur if a standard error is zero (e.g., boundary parameter)

See also: [`get_vcov`](@ref), [`get_ij_vcov`](@ref), [`get_jk_vcov`](@ref)
"""
function compare_variance_estimates(fitted; use_ij::Bool = true, threshold::Float64 = 1.5, verbose::Bool = true)
    # Get model-based variance
    model_vcov = get_vcov(fitted)
    if isnothing(model_vcov)
        error("Model-based variance not available. Fit model with compute_vcov=true.")
    end
    
    # Get robust variance
    robust_vcov = use_ij ? get_ij_vcov(fitted) : get_jk_vcov(fitted)
    if isnothing(robust_vcov)
        error_msg = use_ij ? "IJ variance not available. Fit model with compute_ij_vcov=true." :
                           "JK variance not available. Fit model with compute_jk_vcov=true."
        error(error_msg)
    end
    
    # Compute standard errors
    model_se = sqrt.(diag(model_vcov))
    robust_se = sqrt.(diag(robust_vcov))
    
    # Compute ratios (handle zeros carefully)
    ratio = similar(model_se)
    for i in eachindex(ratio)
        if model_se[i] > sqrt(eps(Float64))
            ratio[i] = robust_se[i] / model_se[i]
        else
            ratio[i] = isapprox(robust_se[i], 0.0, atol=sqrt(eps(Float64))) ? 1.0 : NaN
        end
    end
    
    # Summary statistics
    valid_ratios = filter(!isnan, ratio)
    mean_ratio = isempty(valid_ratios) ? NaN : mean(valid_ratios)
    max_ratio = isempty(valid_ratios) ? NaN : maximum(valid_ratios)
    
    # Flag parameters exceeding threshold
    flagged = findall(ratio .> threshold)
    
    # Generate diagnosis
    diagnosis = if isnan(mean_ratio)
        "Unable to compute variance comparison (check for boundary parameters)"
    elseif mean_ratio <= 1.1
        "Model appears correctly specified (mean SE ratio = $(round(mean_ratio, digits=3)))"
    elseif mean_ratio <= 1.5
        "Minor model misspecification possible (mean SE ratio = $(round(mean_ratio, digits=3)))"
    elseif mean_ratio <= 2.0
        "Moderate model misspecification detected (mean SE ratio = $(round(mean_ratio, digits=3))). Recommend using robust SEs for inference."
    else
        "Substantial model misspecification detected (mean SE ratio = $(round(mean_ratio, digits=3))). Strongly recommend using robust SEs and reconsidering model specification."
    end
    
    # Get parameter names if available
    parnames = try
        get_parnames(fitted)
    catch
        ["param_$i" for i in 1:length(model_se)]
    end
    
    # Print verbose output
    if verbose
        robust_type = use_ij ? "IJ (Sandwich)" : "Jackknife"
        println("\n╔══════════════════════════════════════════════════════════════════════════╗")
        println("║                    VARIANCE ESTIMATION COMPARISON                        ║")
        println("╠══════════════════════════════════════════════════════════════════════════╣")
        println("║ Robust method: $robust_type")
        println("║ Threshold for flagging: $(threshold)")
        println("╠══════════════════════════════════════════════════════════════════════════╣")
        println("║ Parameter            │  Model SE  │  Robust SE │   Ratio   │ Flag       ║")
        println("╠──────────────────────┼────────────┼────────────┼───────────┼────────────╣")
        
        for i in eachindex(parnames)
            name = length(parnames[i]) > 20 ? parnames[i][1:17] * "..." : rpad(parnames[i], 20)
            flag = ratio[i] > threshold ? "  ***" : ""
            ratio_str = isnan(ratio[i]) ? "      NaN" : lpad(round(ratio[i], digits=3), 9)
            model_str = lpad(round(model_se[i], digits=4), 10)
            robust_str = lpad(round(robust_se[i], digits=4), 10)
            println("║ $name │ $model_str │ $robust_str │$ratio_str │$flag           ║")
        end
        
        println("╠══════════════════════════════════════════════════════════════════════════╣")
        println("║ Mean ratio: $(lpad(round(mean_ratio, digits=3), 8))                                                      ║")
        println("║ Max ratio:  $(lpad(round(max_ratio, digits=3), 8))                                                      ║")
        println("║ Flagged parameters: $(length(flagged))                                                       ║")
        println("╠══════════════════════════════════════════════════════════════════════════╣")
        println("║ DIAGNOSIS: $(rpad(diagnosis[1:min(60, length(diagnosis))], 60))  ║")
        if length(diagnosis) > 60
            println("║            $(rpad(diagnosis[61:min(120, length(diagnosis))], 60))  ║")
        end
        println("╚══════════════════════════════════════════════════════════════════════════╝\n")
    end
    
    return (
        ratio = ratio,
        model_se = model_se,
        robust_se = robust_se,
        mean_ratio = mean_ratio,
        max_ratio = max_ratio,
        flagged = flagged,
        diagnosis = diagnosis
    )
end