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
    pars_nested = unflatten_parameters(θ, model)
    hazards = model.hazards
    
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
    pars_nested = unflatten_parameters(pars, model)
    hazards = model.hazards
    
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
    pars_nested = unflatten_parameters(pars, model)
    hazards = model.hazards
    
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
                                           model::MultistateProcess,
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
                                          model::MultistateProcess,
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
                                              model::MultistateProcess,
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

