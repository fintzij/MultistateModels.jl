# ============================================================================
# End Batched Fisher Information Computation
# ============================================================================

"""
    compute_subject_hessians(params, model::MultistateProcess, 
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
- `model::MultistateProcess`: semi-Markov model
- `samplepaths::Vector{Vector{SamplePath}}`: sampled paths for each subject
- `ImportanceWeights::Vector{Vector{Float64}}`: normalized importance weights

# Returns
- `Vector{Matrix{Float64}}`: length-n vector where element i is the p × p Fisher information contribution

# References
- Louis, T. A. (1982). Finding the observed information matrix when using the EM algorithm. 
  Journal of the Royal Statistical Society: Series B, 44(2), 226-233.
"""
function compute_subject_hessians(params::AbstractVector, 
                                  model::MultistateProcess,
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
            
            # handle non-finite values in Hessian, gradient, or importance weight
            w_j = ImportanceWeights[i][j]
            if !isfinite(w_j) || !all(isfinite, hess_j)
                fill!(hess_j, 0.0)
                w_j = 0.0
            end
            if !all(isfinite, grads_j[j])
                fill!(grads_j[j], 0.0)
            end
            
            # Accumulate: wᵢⱼ * (-Hᵢⱼ - gᵢⱼgᵢⱼ')
            fish_i1 .+= w_j * (-hess_j - grads_j[j] * transpose(grads_j[j]))
        end
        
        # Optimized: Σⱼ Σₖ wⱼwₖ gⱼgₖᵀ = (Σⱼ wⱼgⱼ)(Σₖ wₖgₖ)ᵀ
        # Stack gradients into matrix and compute weighted sum
        # Handle non-finite importance weights
        w_clean = copy(ImportanceWeights[i])
        for j in 1:npaths
            if !isfinite(w_clean[j])
                w_clean[j] = 0.0
            end
        end
        G = reduce(hcat, grads_j)  # n_params × n_paths
        g_weighted = G * w_clean  # n_params × 1
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
                                   model::MultistateProcess,
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
                                   model::MultistateProcess,
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
        # Skip subjects with NaN/Inf in their Hessian contribution
        fishinf = zeros(Float64, nparams, nparams)
        n_skipped = 0
        for H_i in subject_hessians
            if all(isfinite, H_i)
                fishinf .+= H_i
            else
                n_skipped += 1
            end
        end
        if n_skipped > 0
            @warn "$(n_skipped) of $(nsubj) subject Hessians contained NaN/Inf and were skipped " *
                  "in Fisher information aggregation."
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
fitted = fit(model, data; vcov_type=:ij)
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
fitted = fit(model, data; vcov_type=:ij)
ij_se = sqrt.(diag(get_vcov(fitted)))  # Robust standard errors
```

See also: [`jk_vcov`](@ref), [`get_vcov`](@ref)
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
fitted = fit(model, data; vcov_type=:jk)
jk_se = sqrt.(diag(get_vcov(fitted)))  # Jackknife standard errors
```

See also: [`ij_vcov`](@ref), [`get_vcov`](@ref)
"""
function jk_vcov(loo_deltas::AbstractMatrix)
    n = size(loo_deltas, 2)
    
    # Warn if n < 2 - jackknife variance is undefined or numerically unstable
    if n < 2
        @warn "Jackknife variance requested with n=$(n) subjects. " *
              "Jackknife requires at least 2 subjects for valid variance estimation. " *
              "Results may be NaN or numerically unstable. " *
              "Consider using model-based variance (type=:model) instead."
    end
    
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
                            model::MultistateProcess,
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
    compute_robust_vcov(params, model::MultistateProcess,
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
- `model::MultistateProcess`: fitted semi-Markov model
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
                            model::MultistateProcess,
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
    
    # Check for non-finite values in Fisher information matrix
    if !all(isfinite, fisher_result.fishinf)
        nan_count = count(isnan, fisher_result.fishinf)
        inf_count = count(isinf, fisher_result.fishinf)
        @warn "Fisher information matrix contains non-finite values ($(nan_count) NaN, $(inf_count) Inf). " *
              "Variance estimation failed. Check for numerical issues in hazard evaluation " *
              "(extreme parameter values, zero hazards, or problematic importance weights)."
        vcov_nan = fill(NaN, nparams, nparams)
        return (vcov = Symmetric(vcov_nan),
                ij_vcov = Symmetric(vcov_nan),
                jk_vcov = compute_jk ? Symmetric(vcov_nan) : nothing,
                subject_gradients = fisher_result.subject_grads)
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


