# =============================================================================
# Smoothing Parameter Selection for Penalized Splines
# =============================================================================
#
# Implements PIJCV (Predictive Infinitesimal Jackknife Cross-Validation) for 
# automatic selection of smoothing parameters λ in penalized spline models.
#
# Based on Wood (2024): "Neighbourhood Cross-Validation" arXiv:2404.16490
#
# Algorithm: mgcv-style "performance iteration" that alternates:
#   1. Given β, update Hessians and optimize λ
#   2. Given λ, update β with warm-started optimization
# This is more accurate than one-shot approximation when β(λ*) differs from β(0).
#
# Key insight: Inner optimization uses PENALIZED loss, outer optimization
# minimizes UNPENALIZED prediction error via leave-one-out approximation.
#
# =============================================================================

using LinearAlgebra

"""
    SmoothingSelectionState

Internal state for smoothing parameter selection via PIJCV/GCV, storing cached matrices and intermediate results.

Note: This is separate from `PIJCVState` in variance.jl which is the lower-level
state for computing PIJCV criterion from matrices.

# Fields
- `beta_hat::Vector{Float64}`: Current coefficient estimate
- `H_unpenalized::Matrix{Float64}`: Unpenalized Hessian (sum of subject Hessians)
- `subject_grads::Matrix{Float64}`: Subject gradients (p × n)
- `subject_hessians::Vector{Matrix{Float64}}`: Subject Hessians
- `penalty_config::PenaltyConfig`: Penalty configuration
- `n_subjects::Int`: Number of subjects
- `n_params::Int`: Number of parameters
- `model::MultistateProcess`: Model for likelihood evaluation
- `data::ExactData`: Data container
"""
mutable struct SmoothingSelectionState
    beta_hat::Vector{Float64}
    H_unpenalized::Matrix{Float64}
    subject_grads::Matrix{Float64}
    subject_hessians::Vector{Matrix{Float64}}
    penalty_config::PenaltyConfig
    n_subjects::Int
    n_params::Int
    model::MultistateProcess
    data::ExactData
end

# =============================================================================
# Helper Functions for Performance Iteration
# =============================================================================

"""
    compute_penalty_from_lambda(beta::AbstractVector{T}, lambda::AbstractVector, 
                                 config::PenaltyConfig) where T

Compute penalty term Σⱼ λⱼ βⱼᵀ Sⱼ βⱼ / 2 using explicit lambda values.

Unlike `compute_penalty` which uses lambdas stored in the config, this function
takes explicit lambda values as an argument. Used during optimization when
lambda is being varied.

# Arguments
- `beta`: Coefficient vector (flat, on estimation scale)
- `lambda`: Vector of smoothing parameters
- `config`: Penalty configuration containing S matrices and index mappings

# Returns
Scalar penalty value (half the quadratic form)
"""
function compute_penalty_from_lambda(beta::AbstractVector{T}, lambda::AbstractVector, 
                                      config::PenaltyConfig) where T
    penalty = zero(T)
    lambda_idx = 1
    
    # Baseline hazard penalties
    for term in config.terms
        β_j = @view beta[term.hazard_indices]
        penalty += lambda[lambda_idx] * dot(β_j, term.S * β_j)
        lambda_idx += 1
    end
    
    # Total hazard penalties
    for term in config.total_hazard_terms
        K = size(term.S, 1)
        β_total = zeros(T, K)
        for idx_range in term.hazard_indices
            β_total .+= @view beta[idx_range]
        end
        penalty += lambda[lambda_idx] * dot(β_total, term.S * β_total)
        lambda_idx += 1
    end
    
    # Smooth covariate penalties (handle shared groups)
    if !isempty(config.shared_smooth_groups)
        # Build term -> lambda mapping
        term_to_lambda = Dict{Int, Int}()
        for (group_idx, group) in enumerate(config.shared_smooth_groups)
            for term_idx in group
                term_to_lambda[term_idx] = lambda_idx
            end
            lambda_idx += 1
        end
        # Handle ungrouped terms
        for term_idx in 1:length(config.smooth_covariate_terms)
            if !haskey(term_to_lambda, term_idx)
                term_to_lambda[term_idx] = lambda_idx
                lambda_idx += 1
            end
        end
        # Apply penalties
        for (term_idx, term) in enumerate(config.smooth_covariate_terms)
            β_k = beta[term.param_indices]
            penalty += lambda[term_to_lambda[term_idx]] * dot(β_k, term.S * β_k)
        end
    else
        # No sharing - each term gets its own lambda
        for term in config.smooth_covariate_terms
            β_k = beta[term.param_indices]
            penalty += lambda[lambda_idx] * dot(β_k, term.S * β_k)
            lambda_idx += 1
        end
    end
    
    return penalty / 2
end

"""
    fit_penalized_beta(model::MultistateProcess, data::ExactData, 
                       lambda::Vector{Float64}, penalty_config::PenaltyConfig,
                       beta_init::Vector{Float64};
                       maxiters::Int=100, use_polyalgorithm::Bool=false,
                       verbose::Bool=false) -> Vector{Float64}

Fit coefficients β given fixed smoothing parameters λ.

Minimizes the penalized negative log-likelihood:
    f(β) = -ℓ(β) + (1/2) Σⱼ λⱼ βⱼᵀ Sⱼ βⱼ

# Arguments
- `model`: MultistateProcess model
- `data`: ExactData container
- `lambda`: Vector of smoothing parameters (natural scale)
- `penalty_config`: Penalty configuration with S matrices
- `beta_init`: Warm start for optimization
- `maxiters`: Maximum optimizer iterations
- `use_polyalgorithm`: If true, use LBFGS→Ipopt; if false, pure Ipopt
- `verbose`: Print optimization progress

# Returns
Fitted coefficient vector β
"""
function fit_penalized_beta(model::MultistateProcess, data::ExactData,
                            lambda::Vector{Float64}, penalty_config::PenaltyConfig,
                            beta_init::Vector{Float64};
                            maxiters::Int=100, use_polyalgorithm::Bool=false,
                            verbose::Bool=false)
    
    # Define penalized negative log-likelihood objective
    function penalized_nll(β, p)
        # Unpenalized negative log-likelihood
        nll = loglik_exact(β, data; neg=true)
        # Add penalty
        pen = compute_penalty_from_lambda(β, lambda, penalty_config)
        return nll + pen
    end
    
    # Set up optimization with automatic differentiation
    adtype = DifferentiationInterface.SecondOrder(
        Optimization.AutoForwardDiff(), 
        Optimization.AutoForwardDiff()
    )
    optf = OptimizationFunction(penalized_nll, adtype)
    prob = OptimizationProblem(optf, beta_init, nothing)
    
    if use_polyalgorithm
        # Phase 1: LBFGS warm-start with loose tolerance
        if verbose
            println("    β fit: LBFGS warm-start...")
        end
        sol_warmstart = solve(prob, LBFGS();
                              maxiters=min(maxiters, 50),
                              abstol=1e-2,
                              reltol=1e-2,
                              show_trace=false)
        
        # Phase 2: Ipopt refinement
        if verbose
            println("    β fit: Ipopt refinement...")
        end
        prob_refined = remake(prob, u0=sol_warmstart.u)
        sol = solve(prob_refined, IpoptOptimizer();
                    maxiters=maxiters, print_level=0, tol=1e-6)
    else
        # Pure Ipopt from warm start
        if verbose
            println("    β fit: Ipopt from warm start...")
        end
        sol = solve(prob, IpoptOptimizer();
                    maxiters=maxiters, print_level=0, tol=1e-6)
    end
    
    return sol.u
end

"""
    optimize_lambda(state::SmoothingSelectionState, log_lambda_init::Vector{Float64};
                    method::Symbol=:pijcv, maxiter::Int=100, verbose::Bool=false,
                    use_polyalgorithm::Bool=true) -> Tuple{Vector{Float64}, Float64, Bool}

Optimize smoothing parameters given fixed Hessians.

# Arguments
- `state`: SmoothingSelectionState with current β and cached Hessians
- `log_lambda_init`: Initial log-smoothing parameters (warm start)
- `method`: Selection method (:pijcv or :gcv)
- `maxiter`: Maximum iterations for λ optimization
- `verbose`: Print progress
- `use_polyalgorithm`: If true, use LBFGS→Ipopt; if false, pure Ipopt

# Returns
- `lambda`: Optimal smoothing parameters (natural scale)
- `criterion`: Final criterion value
- `converged`: Whether optimization succeeded
"""
function optimize_lambda(state::SmoothingSelectionState, log_lambda_init::Vector{Float64};
                         method::Symbol=:pijcv, maxiter::Int=100, verbose::Bool=false,
                         use_polyalgorithm::Bool=true)
    
    n_lambda = length(log_lambda_init)
    
    # Bounds on log-lambda (λ from 1e-6 to 1e6)
    lower_bounds = fill(-13.8, n_lambda)
    upper_bounds = fill(13.8, n_lambda)
    
    # Select criterion function
    objective = if method == :pijcv
        (log_lam, p) -> compute_pijcv_criterion(log_lam, state)
    else
        (log_lam, p) -> compute_gcv_criterion(log_lam, state)
    end
    
    optf = OptimizationFunction(objective, AutoForwardDiff())
    prob = OptimizationProblem(optf, log_lambda_init, nothing;
                               lb=lower_bounds, ub=upper_bounds)
    
    result = try
        if use_polyalgorithm
            # Phase 1: LBFGS warm-start
            if verbose
                println("    λ opt: LBFGS warm-start...")
            end
            sol_warmstart = solve(prob, LBFGS();
                                  maxiters=min(maxiter, 50),
                                  abstol=1e-2,
                                  reltol=1e-2,
                                  show_trace=false)
            
            # Phase 2: Ipopt refinement
            if verbose
                println("    λ opt: Ipopt refinement...")
            end
            prob_refined = remake(prob, u0=sol_warmstart.u)
            solve(prob_refined, IpoptOptimizer();
                  maxiters=maxiter, print_level=0, tol=1e-6)
        else
            # Pure Ipopt
            if verbose
                println("    λ opt: Ipopt from warm start...")
            end
            solve(prob, IpoptOptimizer();
                  maxiters=maxiter, print_level=0, tol=1e-6)
        end
    catch e
        if verbose
            println("    λ optimization failed: ", e)
        end
        nothing
    end
    
    if !isnothing(result) && (result.retcode == ReturnCode.Success || result.retcode == ReturnCode.MaxIters)
        lambda = exp.(result.u)
        criterion = result.objective
        converged = (result.retcode == ReturnCode.Success)
        
        # Check for degenerate solution
        if any(lambda .< 1e-10) || any(lambda .> 1e10)
            if verbose
                println("    λ optimization produced degenerate values")
            end
            converged = false
        end
        
        return (lambda, criterion, converged)
    else
        # Failed - return initial values
        return (exp.(log_lambda_init), Inf, false)
    end
end

"""
    extract_lambda_vector(config::PenaltyConfig) -> Vector{Float64}

Extract the current λ values from a penalty configuration as a flat vector.
"""
function extract_lambda_vector(config::PenaltyConfig)
    lambdas = Float64[]
    
    # Baseline hazard terms
    for term in config.terms
        push!(lambdas, term.lambda)
    end
    
    # Total hazard terms
    for term in config.total_hazard_terms
        push!(lambdas, term.lambda_H)
    end
    
    # Smooth covariate terms (handle shared groups)
    if !isempty(config.shared_smooth_groups)
        for group in config.shared_smooth_groups
            # Use first term's lambda as representative
            term = config.smooth_covariate_terms[group[1]]
            push!(lambdas, term.lambda)
        end
        # Handle ungrouped terms
        grouped_indices = Set(vcat(config.shared_smooth_groups...))
        for (idx, term) in enumerate(config.smooth_covariate_terms)
            if idx ∉ grouped_indices
                push!(lambdas, term.lambda)
            end
        end
    else
        for term in config.smooth_covariate_terms
            push!(lambdas, term.lambda)
        end
    end
    
    return lambdas
end

# =============================================================================
# PIJCV and GCV Criterion Functions
# =============================================================================

"""
    compute_pijcv_criterion(log_lambda::Vector{Float64}, state::SmoothingSelectionState) -> Float64

Compute the PIJCV/NCV criterion V(λ) for given log-smoothing parameters.

Implements the Neighbourhood Cross-Validation criterion from Wood (2024) 
"On Neighbourhood Cross Validation" arXiv:2404.16490v4.

# NCV Criterion (Wood 2024, Equation 2)

The criterion is the sum of leave-one-out prediction errors:

    V(λ) = Σᵢ Dᵢ(β̂⁻ⁱ)

where:
- Dᵢ(β) = -ℓᵢ(β) is subject i's negative log-likelihood contribution
- β̂⁻ⁱ is the penalized MLE with subject i omitted

# Efficient Computation

Rather than refit n times, we use Newton's approximation (Wood 2024, Equation 3):

    β̂⁻ⁱ ≈ β̂ + Δ⁻ⁱ,  where  Δ⁻ⁱ = H_{λ,-i}⁻¹ gᵢ

where H_{λ,-i} = H_λ - Hᵢ is the leave-one-out penalized Hessian.

# Key Implementation Detail

**This function evaluates the ACTUAL likelihood** Dᵢ(β̂⁻ⁱ) at the LOO parameters,
NOT a Taylor approximation. The quadratic approximation V_q is used as a fallback
only when the actual likelihood is non-finite (Wood 2024, Section 4.1).

# Arguments
- `log_lambda`: Vector of log-smoothing parameters (one per penalty term)
- `state`: SmoothingSelectionState with cached gradients/Hessians

# Returns
- `Float64`: PIJCV criterion value (lower is better)

# References
- Wood, S.N. (2024). "On Neighbourhood Cross Validation." arXiv:2404.16490v4
"""
function compute_pijcv_criterion(log_lambda::AbstractVector{T}, state::SmoothingSelectionState) where T<:Real
    lambda = exp.(log_lambda)
    
    # Build penalized Hessian: H_λ = H + Σⱼ λⱼ Sⱼ
    # Create matrix with appropriate eltype for AD
    H_lambda = _build_penalized_hessian(state.H_unpenalized, lambda, state.penalty_config)
    
    # Try Cholesky factorization of full penalized Hessian
    H_lambda_sym = Symmetric(H_lambda)
    chol_fact = try
        cholesky(H_lambda_sym)
    catch e
        # If Cholesky fails, H_λ is not positive definite
        # Return a large value to steer optimization away
        return T(1e10)
    end
    
    # Get subject-level likelihoods at current estimate (for V_q fallback)
    ll_subj_base = loglik_exact(state.beta_hat, state.data; neg=false, return_ll_subj=true)
    
    # Compute NCV criterion: V = Σᵢ Dᵢ(β̂⁻ⁱ)
    # Following Wood (2024): evaluate ACTUAL likelihood at LOO parameters
    V = zero(T)
    n_fallback = 0  # Track how many times we fall back to V_q
    n_downdate_fallback = 0  # Track Cholesky downdate failures
    
    # Precompute inverse for fallback (only needed if downdates fail)
    H_inv_fallback = nothing
    
    for i in 1:state.n_subjects
        # Get subject i's gradient and Hessian (of negative log-likelihood)
        # Convention: gᵢ = ∇Dᵢ = -∇ℓᵢ, Hᵢ = ∇²Dᵢ = -∇²ℓᵢ
        g_i = @view state.subject_grads[:, i]
        H_i = state.subject_hessians[i]
        
        # Solve for coefficient perturbation: Δ⁻ⁱ = H_{λ,-i}⁻¹ gᵢ
        # Use Cholesky downdate for efficiency (Wood 2024, Section 2.1)
        delta_i = _solve_loo_newton_step(chol_fact, H_i, g_i)
        
        if delta_i === nothing
            # Cholesky downdate failed (indefinite H_{λ,-i})
            # Try direct solve as fallback
            n_downdate_fallback += 1
            
            H_lambda_loo = H_lambda - H_i
            H_loo_sym = Symmetric(H_lambda_loo)
            
            delta_i = try
                H_loo_sym \ g_i
            catch e
                # If solve fails (e.g., indefinite Hessian), return large value
                # This handles Wood's "Degeneracy 1" (Section 3.1)
                return T(1e10)
            end
        end
        
        # Compute LOO parameters: β̂⁻ⁱ = β̂ + Δ⁻ⁱ
        #
        # Derivation (loss convention):
        # At penalized MLE: ∇D(β̂) + λSβ̂ = 0, where D = Σⱼ Dⱼ
        # LOO gradient at β̂: ∇D⁻ⁱ(β̂) = Σⱼ≠ᵢ gⱼ + λSβ̂ = -gᵢ
        # Newton step: β̂⁻ⁱ = β̂ - H_{λ,-i}⁻¹(-gᵢ) = β̂ + H_{λ,-i}⁻¹ gᵢ
        beta_loo = state.beta_hat .+ delta_i
        
        # CORE OF NCV: Evaluate ACTUAL likelihood at LOO parameters
        # This is what makes NCV correct - NOT a Taylor approximation
        ll_loo = loglik_subject(beta_loo, state.data, i)
        
        # Check if likelihood is finite
        if isfinite(ll_loo)
            # Normal case: use actual likelihood
            # D_i = -ℓ_i(β̂⁻ⁱ)
            D_i = -ll_loo
        else
            # Fallback to quadratic approximation V_q (Wood 2024, Section 4.1)
            # This handles "Degeneracy 3" (infinite LOO loss)
            #
            # V_q uses Taylor expansion: Dᵢ(β̂ + δ) ≈ Dᵢ(β̂) + gᵢᵀδ + ½δᵀHᵢδ
            linear_term = dot(g_i, delta_i)
            quadratic_term = T(0.5) * dot(delta_i, H_i * delta_i)
            D_i = -ll_subj_base[i] + linear_term + quadratic_term
            n_fallback += 1
        end
        
        V += D_i
    end
    
    # Note: Could log n_fallback if verbose mode is needed
    # if n_fallback > 0
    #     @info "NCV: Used V_q fallback for $n_fallback of $(state.n_subjects) subjects"
    # end
    # if n_downdate_fallback > 0
    #     @info "NCV: Cholesky downdate failed for $n_downdate_fallback subjects"
    # end
    
    return V
end

"""
    _solve_loo_newton_step(chol_H::Cholesky, H_i::Matrix, g_i::AbstractVector) -> Union{Vector, Nothing}

Solve the leave-one-out Newton step using Cholesky rank-k downdate.

Computes δᵢ = (H - Hᵢ)⁻¹ gᵢ efficiently via Cholesky downdate.

# Algorithm (Wood 2024, Section 2.1)

Given the Cholesky factor L where H = LLᵀ, we want to solve (H - Hᵢ)x = g.

1. Eigendecompose Hᵢ = VDVᵀ where D = diag(d₁, ..., dₖ) with dₖ > 0
2. For each positive eigenvalue dⱼ, perform rank-1 downdate:
   LLᵀ - dⱼvⱼvⱼᵀ → L̃L̃ᵀ
3. Solve L̃L̃ᵀx = g

# Complexity
- Naive: O(p³) per subject (form matrix + factorize)  
- Downdate: O(kp²) per subject, where k = rank(Hᵢ) ≤ p

For multistate models with few transitions per subject, k is typically small.

# Returns
- `Vector`: Solution δᵢ if successful
- `nothing`: If downdate fails (H_{λ,-i} is indefinite)

# Reference
- Wood, S.N. (2024). On Neighbourhood Cross Validation, arXiv:2404.16490v4, Section 2.1
- Seeger, M. (2004). Low Rank Updates for the Cholesky Decomposition.
"""
function _solve_loo_newton_step(chol_H::Cholesky, H_i::Matrix{Float64}, g_i::AbstractVector)
    # Copy Cholesky factor as a regular Matrix (we'll modify it)
    L = Matrix(chol_H.L)
    n = size(L, 1)
    
    # Eigendecompose subject Hessian to get rank-k representation
    # H_i = V * D * V' where D = diag(eigenvalues)
    eigen_H = eigen(Symmetric(H_i))
    
    # Perform rank-1 downdates for each positive eigenvalue
    # This computes L̃ such that L̃L̃ᵀ = LLᵀ - Hᵢ = H - Hᵢ
    tol = sqrt(eps(Float64))
    
    for (idx, d) in enumerate(eigen_H.values)
        if d > tol
            # Downdate: LLᵀ - d*v*vᵀ
            v = eigen_H.vectors[:, idx]
            success = _cholesky_downdate!(L, sqrt(d) * v)
            if !success
                # Matrix became indefinite
                return nothing
            end
        end
    end
    
    # Solve using downdated Cholesky: L̃L̃ᵀ x = g
    # Forward solve: L̃ y = g
    # Backward solve: L̃ᵀ x = y
    try
        L_chol = Cholesky(L, 'L', 0)
        return L_chol \ collect(g_i)
    catch e
        # Cholesky solve failed
        return nothing
    end
end

"""
    _cholesky_downdate!(L::Matrix, v::Vector; tol=1e-10) -> Bool

Perform in-place rank-1 downdate of Cholesky factor: LLᵀ → L̃L̃ᵀ where L̃L̃ᵀ = LLᵀ - vvᵀ.

Uses the standard sequential downdate algorithm with Givens rotations.

# Arguments
- `L`: Lower triangular Cholesky factor (modified in place)
- `v`: Vector for rank-1 downdate (will be modified)
- `tol`: Tolerance for indefiniteness detection

# Returns
- `true` if downdate succeeded
- `false` if matrix became indefinite (diagonal element would be imaginary)

# Algorithm
For j = 1, ..., n:
1. r² = L[j,j]² - v[j]²
2. If r² < tol, return false (indefinite)
3. r = √r², c = r/L[j,j], s = v[j]/L[j,j]
4. L[j,j] = r
5. For i > j: Update L[i,j] and v[i] using Givens rotation

# Reference
Seeger, M. (2004). Low Rank Updates for the Cholesky Decomposition.
"""
function _cholesky_downdate!(L::Matrix{Float64}, v::Vector{Float64}; tol::Float64=1e-10)
    n = size(L, 1)
    w = copy(v)
    
    for j in 1:n
        # Hyperbolic Givens rotation to eliminate w[j]
        # We want r² = L[j,j]² - w[j]² where r is the new diagonal
        a = L[j, j]
        b = w[j]
        
        r² = a^2 - b^2
        
        # Check for indefiniteness
        if r² < tol
            return false
        end
        
        r = sqrt(r²)
        
        # Hyperbolic rotation: [c s; s c] with c = a/r, s = -b/r, c² - s² = 1
        c = a / r
        s = -b / r
        
        L[j, j] = r
        
        # Apply rotation to remaining elements: [L[i,j]; w[i]] → [c*L + s*w; s*L + c*w]
        @inbounds for i in (j+1):n
            temp = c * L[i, j] + s * w[i]
            w[i] = s * L[i, j] + c * w[i]
            L[i, j] = temp
        end
    end
    
    return true
end

"""
    _add_penalty_to_hessian!(H::Matrix{Float64}, lambda::Vector{Float64}, 
                              config::PenaltyConfig)

Add penalty matrices to Hessian in-place: H += Σⱼ λⱼ Sⱼ

Handles baseline hazard terms, total hazard terms, and smooth covariate terms.
"""
function _add_penalty_to_hessian!(H::Matrix{Float64}, lambda::Vector{Float64}, 
                                   config::PenaltyConfig)
    lambda_idx = 1
    
    # Baseline hazard penalty terms
    for term in config.terms
        idx = term.hazard_indices
        H[idx, idx] .+= lambda[lambda_idx] .* term.S
        lambda_idx += 1
    end
    
    # Total hazard terms (for competing risks)
    for term in config.total_hazard_terms
        # Total hazard penalty uses Kronecker structure
        # For now, use the simplified addition per hazard
        for idx_range in term.hazard_indices
            H[idx_range, idx_range] .+= lambda[lambda_idx] .* term.S
        end
        lambda_idx += 1
    end
    
    # Smooth covariate penalty terms
    # Handle shared lambda groups
    if !isempty(config.shared_smooth_groups)
        # When lambda is shared, use the same lambda for all terms in group
        # First assign lambdas to groups
        term_to_lambda = Dict{Int, Int}()
        
        for (group_idx, group) in enumerate(config.shared_smooth_groups)
            for term_idx in group
                term_to_lambda[term_idx] = lambda_idx
            end
            lambda_idx += 1
        end
        
        # Handle terms not in any group (if n_lambda accounts for them separately)
        for (term_idx, term) in enumerate(config.smooth_covariate_terms)
            if !haskey(term_to_lambda, term_idx)
                term_to_lambda[term_idx] = lambda_idx
                lambda_idx += 1
            end
        end
        
        # Apply penalties
        for (term_idx, term) in enumerate(config.smooth_covariate_terms)
            lam = lambda[term_to_lambda[term_idx]]
            indices = term.param_indices
            for (i, pi) in enumerate(indices)
                for (j, pj) in enumerate(indices)
                    H[pi, pj] += lam * term.S[i, j]
                end
            end
        end
    else
        # No sharing - each term gets its own lambda
        for term in config.smooth_covariate_terms
            indices = term.param_indices
            for (i, pi) in enumerate(indices)
                for (j, pj) in enumerate(indices)
                    H[pi, pj] += lambda[lambda_idx] * term.S[i, j]
                end
            end
            lambda_idx += 1
        end
    end
end

"""
    _build_penalized_hessian(H_unpen::Matrix{Float64}, lambda::AbstractVector{T}, 
                              config::PenaltyConfig) where T<:Real

Build penalized Hessian with proper element type for AD compatibility.
Returns: H + Σⱼ λⱼ Sⱼ

This is a non-mutating version of `_add_penalty_to_hessian!` that creates 
a new matrix with the appropriate type (Float64 or Dual).
"""
function _build_penalized_hessian(H_unpen::Matrix{Float64}, lambda::AbstractVector{T}, 
                                   config::PenaltyConfig) where T<:Real
    # Convert to eltype compatible with lambda
    H = convert(Matrix{T}, copy(H_unpen))
    
    lambda_idx = 1
    
    # Baseline hazard penalty terms
    for term in config.terms
        idx = term.hazard_indices
        @inbounds for i in idx, j in idx
            H[i, j] += lambda[lambda_idx] * term.S[i - first(idx) + 1, j - first(idx) + 1]
        end
        lambda_idx += 1
    end
    
    # Total hazard terms (for competing risks)
    for term in config.total_hazard_terms
        for idx_range in term.hazard_indices
            @inbounds for i in idx_range, j in idx_range
                H[i, j] += lambda[lambda_idx] * term.S[i - first(idx_range) + 1, j - first(idx_range) + 1]
            end
        end
        lambda_idx += 1
    end
    
    # Smooth covariate penalty terms
    if !isempty(config.shared_smooth_groups)
        term_to_lambda = Dict{Int, Int}()
        
        for (group_idx, group) in enumerate(config.shared_smooth_groups)
            for term_idx in group
                term_to_lambda[term_idx] = lambda_idx
            end
            lambda_idx += 1
        end
        
        for (term_idx, term) in enumerate(config.smooth_covariate_terms)
            if !haskey(term_to_lambda, term_idx)
                term_to_lambda[term_idx] = lambda_idx
                lambda_idx += 1
            end
        end
        
        for (term_idx, term) in enumerate(config.smooth_covariate_terms)
            lam = lambda[term_to_lambda[term_idx]]
            indices = term.param_indices
            @inbounds for (i, pi) in enumerate(indices)
                for (j, pj) in enumerate(indices)
                    H[pi, pj] += lam * term.S[i, j]
                end
            end
        end
    else
        for term in config.smooth_covariate_terms
            indices = term.param_indices
            @inbounds for (i, pi) in enumerate(indices)
                for (j, pj) in enumerate(indices)
                    H[pi, pj] += lambda[lambda_idx] * term.S[i, j]
                end
            end
            lambda_idx += 1
        end
    end
    
    return H
end

"""
    _create_scoped_penalty_config(config::PenaltyConfig, scope::Symbol) -> Tuple{PenaltyConfig, Vector{Float64}, Vector{Float64}}

Create a penalty config containing only terms within the specified scope.

# Arguments
- `config`: Original penalty configuration
- `scope`: Which terms to include (:all, :baseline, :covariates)

# Returns
- Scoped PenaltyConfig
- Vector of fixed baseline lambdas (empty if baseline is in scope)
- Vector of fixed covariate lambdas (empty if covariates are in scope)
"""
function _create_scoped_penalty_config(config::PenaltyConfig, scope::Symbol)
    if scope == :all
        # No filtering needed
        return (config, Float64[], Float64[])
    end
    
    if scope == :baseline
        # Include only baseline terms, exclude covariates
        fixed_covariate_lambdas = [term.lambda for term in config.smooth_covariate_terms]
        
        # Create config with no covariate terms
        n_lambda_baseline = length(config.terms) + length(config.total_hazard_terms)
        scoped_config = PenaltyConfig(
            config.terms,
            config.total_hazard_terms,
            SmoothCovariatePenaltyTerm[],  # Empty covariate terms
            config.shared_lambda_groups,
            Vector{Int}[],  # Empty shared smooth groups
            n_lambda_baseline
        )
        return (scoped_config, Float64[], fixed_covariate_lambdas)
    end
    
    if scope == :covariates
        # Include only covariate terms, exclude baseline
        fixed_baseline_lambdas = vcat(
            [term.lambda for term in config.terms],
            [term.lambda_H for term in config.total_hazard_terms]
        )
        
        # Calculate n_lambda for covariate terms
        n_lambda_cov = if !isempty(config.shared_smooth_groups)
            length(config.shared_smooth_groups)
        else
            length(config.smooth_covariate_terms)
        end
        
        # Create config with no baseline terms
        scoped_config = PenaltyConfig(
            PenaltyTerm[],  # Empty baseline terms
            TotalHazardPenaltyTerm[],  # Empty total hazard terms
            config.smooth_covariate_terms,
            Dict{Int,Vector{Int}}(),  # Empty shared baseline groups
            config.shared_smooth_groups,
            n_lambda_cov
        )
        return (scoped_config, fixed_baseline_lambdas, Float64[])
    end
    
    error("Unknown scope: $scope")
end

"""
    _merge_scoped_lambdas(original_config, scoped_config, optimized_lambda, 
                          fixed_baseline, fixed_covariate, scope) -> PenaltyConfig

Merge optimized lambdas back into the original config structure.
"""
function _merge_scoped_lambdas(original_config::PenaltyConfig, 
                                scoped_config::PenaltyConfig,
                                optimized_lambda::Vector{Float64},
                                fixed_baseline_lambdas::Vector{Float64},
                                fixed_covariate_lambdas::Vector{Float64},
                                scope::Symbol)
    if scope == :all
        # All terms were optimized, just update the config
        return _update_penalty_lambda(original_config, optimized_lambda)
    end
    
    if scope == :baseline
        # Baseline was optimized, covariates are fixed
        # First update the scoped config with optimized lambdas
        updated_baseline = _update_penalty_lambda(scoped_config, optimized_lambda)
        
        # Reconstruct with original covariate terms (keep their original lambdas)
        return PenaltyConfig(
            updated_baseline.terms,
            updated_baseline.total_hazard_terms,
            original_config.smooth_covariate_terms,
            original_config.shared_lambda_groups,
            original_config.shared_smooth_groups,
            original_config.n_lambda
        )
    end
    
    if scope == :covariates
        # Covariates were optimized, baseline is fixed
        # First update the scoped config with optimized lambdas
        updated_covariate = _update_penalty_lambda(scoped_config, optimized_lambda)
        
        # Reconstruct with original baseline terms (keep their original lambdas)
        return PenaltyConfig(
            original_config.terms,
            original_config.total_hazard_terms,
            updated_covariate.smooth_covariate_terms,
            original_config.shared_lambda_groups,
            original_config.shared_smooth_groups,
            original_config.n_lambda
        )
    end
    
    error("Unknown scope: $scope")
end

"""
    select_smoothing_parameters(model::MultistateModel, penalty::SplinePenalty;
                                 method::Symbol=:pijcv,
                                 scope::Symbol=:all,
                                 max_outer_iter::Int=20,
                                 inner_maxiter::Int=50,
                                 lambda_maxiter::Int=100,
                                 beta_tol::Float64=1e-4,
                                 lambda_tol::Float64=1e-3,
                                 lambda_init::Float64=1.0,
                                 verbose::Bool=false) -> NamedTuple

User-friendly interface for smoothing parameter selection using performance iteration.

This function implements mgcv-style "performance iteration" that alternates:
1. Given β, compute Hessians and optimize λ via PIJCV/GCV
2. Given λ, update β via penalized maximum likelihood

This is more accurate than one-shot approximation when the penalized solution β(λ*)
differs significantly from the unpenalized solution β(0).

# Arguments
- `model`: MultistateModel
- `penalty`: SplinePenalty specification
- `method`: Selection method (:pijcv or :gcv)
- `scope`: Which splines to calibrate (:all, :baseline, or :covariates)
- `max_outer_iter`: Maximum outer iterations (β-λ alternation)
- `inner_maxiter`: Maximum iterations for β update per outer iteration
- `lambda_maxiter`: Maximum iterations for λ optimization
- `beta_tol`: Convergence tolerance for β (relative change)
- `lambda_tol`: Convergence tolerance for λ (relative change in log scale)
- `lambda_init`: Initial value for smoothing parameters
- `verbose`: Print progress

# Returns
NamedTuple with:
- `lambda`: Optimal smoothing parameters
- `beta`: Final coefficient estimate
- `criterion`: Final criterion value
- `converged`: Whether optimization converged
- `method_used`: Actual method used (may fall back from PIJCV to GCV)
- `penalty_config`: Updated penalty configuration with optimal λ
- `n_outer_iter`: Number of outer iterations performed
"""
function select_smoothing_parameters(model::MultistateModel, penalty::SplinePenalty;
                                      method::Symbol=:pijcv,
                                      scope::Symbol=:all,
                                      max_outer_iter::Int=20,
                                      inner_maxiter::Int=50,
                                      lambda_maxiter::Int=100,
                                      beta_tol::Float64=1e-4,
                                      lambda_tol::Float64=1e-3,
                                      lambda_init::Float64=1.0,
                                      verbose::Bool=false)
    # Extract sample paths
    samplepaths = extract_paths(model)
    
    # Build penalty configuration
    penalty_config = build_penalty_config(model, penalty; lambda_init=lambda_init)
    
    # Create ExactData container
    data = ExactData(model, samplepaths)
    
    # Fit unpenalized model to get initial coefficients
    if verbose
        println("Fitting unpenalized model for initialization...")
    end
    
    # Get initial parameters from model
    parameters = get_parameters_flat(model)
    
    # Define unpenalized likelihood function
    loglik_fn = loglik_exact
    adtype = DifferentiationInterface.SecondOrder(
        Optimization.AutoForwardDiff(), 
        Optimization.AutoForwardDiff()
    )
    optf = OptimizationFunction(loglik_fn, adtype)
    prob = OptimizationProblem(optf, parameters, data)
    
    sol = _solve_optimization(prob, IpoptOptimizer())
    beta_unpen = sol.u
    
    if verbose
        println("Unpenalized fit complete.")
    end
    
    # Fit PENALIZED model with initial λ to get proper starting point for performance iteration
    # This is critical: starting from unpenalized MLE causes PIJCV to push λ → ∞
    # because at the unpenalized MLE, the score is zero and more smoothing only reduces variance
    if verbose
        println("Fitting penalized model with initial λ=$(lambda_init)...")
    end
    lambda_init_vec = fill(lambda_init, penalty_config.n_lambda)
    beta_init = fit_penalized_beta(model, data, lambda_init_vec, penalty_config, beta_unpen;
                                   maxiters=inner_maxiter, use_polyalgorithm=true, verbose=false)
    if verbose
        println("Penalized initialization complete.")
    end
    
    # Call the internal function
    return select_smoothing_parameters(model, data, penalty_config, beta_init;
                                        method=method, scope=scope, 
                                        max_outer_iter=max_outer_iter,
                                        inner_maxiter=inner_maxiter,
                                        lambda_maxiter=lambda_maxiter,
                                        beta_tol=beta_tol, lambda_tol=lambda_tol,
                                        verbose=verbose)
end

"""
    select_smoothing_parameters(model::MultistateProcess, data::ExactData,
                                 penalty_config::PenaltyConfig,
                                 beta_init::Vector{Float64};
                                 method::Symbol=:pijcv,
                                 scope::Symbol=:all,
                                 max_outer_iter::Int=20,
                                 inner_maxiter::Int=50,
                                 lambda_maxiter::Int=100,
                                 beta_tol::Float64=1e-4,
                                 lambda_tol::Float64=1e-3,
                                 verbose::Bool=false) -> NamedTuple

Select smoothing parameters using mgcv-style performance iteration.

Algorithm:
1. Initialize β from unpenalized fit, λ from penalty config
2. Outer loop:
   a. Given β, compute Hessians and optimize λ via PIJCV/GCV
   b. Given λ, update β via penalized maximum likelihood (warm-started)
   c. Check convergence
3. Return final (β, λ)

# Arguments
- `model`: MultistateProcess model
- `data`: ExactData container
- `penalty_config`: Penalty configuration with initial λ values
- `beta_init`: Initial coefficient estimate (warm start)
- `method`: Selection method (:pijcv or :gcv)
- `scope`: Which splines to calibrate:
  - `:all` (default): Calibrate all spline penalties (baseline + covariates)
  - `:baseline`: Calibrate only baseline hazard splines
  - `:covariates`: Calibrate only smooth covariate splines
- `max_outer_iter`: Maximum outer iterations (β-λ alternation)
- `inner_maxiter`: Maximum iterations for β update
- `lambda_maxiter`: Maximum iterations for λ optimization
- `beta_tol`: Convergence tolerance for β (relative change)
- `lambda_tol`: Convergence tolerance for λ (relative change in log scale)
- `verbose`: Print progress

# Returns
NamedTuple with:
- `lambda`: Optimal smoothing parameters (for calibrated terms only)
- `beta`: Final coefficient estimate
- `criterion`: Final criterion value
- `converged`: Whether optimization converged
- `method_used`: Actual method used (may fall back from PIJCV to GCV)
- `penalty_config`: Updated penalty configuration with optimal λ
- `n_outer_iter`: Number of outer iterations performed
"""
function select_smoothing_parameters(model::MultistateProcess, data::ExactData,
                                      penalty_config::PenaltyConfig,
                                      beta_init::Vector{Float64};
                                      method::Symbol=:pijcv,
                                      scope::Symbol=:all,
                                      max_outer_iter::Int=20,
                                      inner_maxiter::Int=50,
                                      lambda_maxiter::Int=100,
                                      beta_tol::Float64=1e-4,
                                      lambda_tol::Float64=1e-3,
                                      verbose::Bool=false)
    # Validate scope
    scope ∈ (:all, :baseline, :covariates) || 
        throw(ArgumentError("scope must be :all, :baseline, or :covariates, got :$scope"))
    
    # Create scoped penalty config based on scope parameter
    scoped_config, fixed_baseline_lambdas, fixed_covariate_lambdas = _create_scoped_penalty_config(penalty_config, scope)
    
    n_lambda = scoped_config.n_lambda
    n_lambda > 0 || return (lambda=Float64[], beta=beta_init, criterion=NaN, 
                            converged=true, method_used=:none, penalty_config=penalty_config,
                            n_outer_iter=0)
    
    # Use PROFILE OPTIMIZATION instead of alternating optimization
    # The alternating approach fails because V(λ) at fixed β is monotonic in λ.
    # Profile optimization evaluates V(λ, β(λ)) where β(λ) is fitted at each λ.
    
    if verbose
        println("Using profile optimization (grid search) for λ selection")
        println("  Method: $method")
    end
    
    samplepaths = data.paths
    n_params = length(beta_init)
    n_subjects = length(samplepaths)
    
    # Grid search over log(λ)
    # Wide range to capture both under-penalized and over-penalized regimes
    # log(λ) from -8 to 8 covers λ from ~0.0003 to ~3000
    log_lambda_grid = collect(-8.0:1.0:8.0)
    
    best_lambda = Float64[]
    best_beta = copy(beta_init)
    best_criterion = Inf
    grid_results = Vector{NamedTuple{(:log_lambda, :lambda, :criterion), Tuple{Float64, Float64, Float64}}}()
    
    if verbose
        println("  Grid search over log(λ) ∈ [$(log_lambda_grid[1]), $(log_lambda_grid[end])]")
        println("\n  log(λ)    λ          criterion")
        println("  " * "-"^40)
    end
    
    for log_lam in log_lambda_grid
        lam = exp(log_lam)
        lambda_vec = fill(lam, n_lambda)
        
        # Fit β at this λ
        beta_lam = fit_penalized_beta(model, data, lambda_vec, scoped_config, beta_init;
                                       maxiters=inner_maxiter,
                                       use_polyalgorithm=true,
                                       verbose=false)
        
        # Compute Hessians at β(λ)
        subject_grads_ll = compute_subject_gradients(beta_lam, model, samplepaths)
        subject_hessians_ll = compute_subject_hessians_fast(beta_lam, model, samplepaths)
        
        # Convert to loss convention
        subject_grads = -subject_grads_ll
        subject_hessians = [-H for H in subject_hessians_ll]
        H_unpenalized = sum(subject_hessians)
        
        # Create state for criterion evaluation
        state = SmoothingSelectionState(
            copy(beta_lam),
            H_unpenalized,
            subject_grads,
            subject_hessians,
            scoped_config,
            n_subjects,
            n_params,
            model,
            data
        )
        
        # Compute criterion at (λ, β(λ))
        # Note: log_lambda must be a vector of length n_lambda
        log_lambda_vec = fill(log_lam, n_lambda)
        criterion = if method == :pijcv
            compute_pijcv_criterion(log_lambda_vec, state)
        else
            compute_gcv_criterion(log_lambda_vec, state)
        end
        
        push!(grid_results, (log_lambda=log_lam, lambda=lam, criterion=criterion))
        
        if verbose
            println("  $(round(log_lam, digits=1))\t$(round(lam, sigdigits=3))\t$(round(criterion, digits=3))")
        end
        
        if criterion < best_criterion
            best_criterion = criterion
            best_lambda = lambda_vec
            best_beta = copy(beta_lam)
        end
    end
    
    # Refine around the best point with finer grid
    best_log_lam = log(best_lambda[1])
    fine_grid = range(max(-2.0, best_log_lam - 1.5), min(8.0, best_log_lam + 1.5), length=11)
    
    if verbose
        println("\n  Refining around log(λ) = $(round(best_log_lam, digits=2))")
    end
    
    for log_lam in fine_grid
        # Skip if already evaluated
        any(r -> abs(r.log_lambda - log_lam) < 0.01, grid_results) && continue
        
        lam = exp(log_lam)
        lambda_vec = fill(lam, n_lambda)
        
        beta_lam = fit_penalized_beta(model, data, lambda_vec, scoped_config, best_beta;
                                       maxiters=inner_maxiter,
                                       use_polyalgorithm=false,
                                       verbose=false)
        
        subject_grads_ll = compute_subject_gradients(beta_lam, model, samplepaths)
        subject_hessians_ll = compute_subject_hessians_fast(beta_lam, model, samplepaths)
        subject_grads = -subject_grads_ll
        subject_hessians = [-H for H in subject_hessians_ll]
        H_unpenalized = sum(subject_hessians)
        
        state = SmoothingSelectionState(
            copy(beta_lam),
            H_unpenalized,
            subject_grads,
            subject_hessians,
            scoped_config,
            n_subjects,
            n_params,
            model,
            data
        )
        
        log_lambda_vec = fill(log_lam, n_lambda)
        criterion = if method == :pijcv
            compute_pijcv_criterion(log_lambda_vec, state)
        else
            compute_gcv_criterion(log_lambda_vec, state)
        end
        
        if criterion < best_criterion
            best_criterion = criterion
            best_lambda = lambda_vec
            best_beta = copy(beta_lam)
        end
    end
    
    if verbose
        println("\n  Optimal λ: $(round.(best_lambda, sigdigits=4))")
        println("  Final criterion: $(round(best_criterion, digits=4))")
    end
    
    # Update penalty config with final λ
    updated_config = _merge_scoped_lambdas(penalty_config, scoped_config, best_lambda, 
                                           fixed_baseline_lambdas, fixed_covariate_lambdas, scope)
    
    return (
        lambda = best_lambda,
        beta = best_beta,
        criterion = best_criterion,
        converged = true,  # Grid search always "converges"
        method_used = method,
        penalty_config = updated_config,
        n_outer_iter = length(log_lambda_grid) + length(fine_grid)
    )
end


"""
    _select_smoothing_parameters_legacy(...)

Legacy alternating optimization approach (deprecated).
Kept for reference but not recommended - see PIJCV_ALGORITHM_FIX_NEEDED.md

The issue is that at any fixed β, V(λ) is monotonically decreasing with λ,
causing the optimizer to always select λ → ∞.
"""
function _select_smoothing_parameters_legacy(model::MultistateProcess, data::ExactData,
                                      penalty_config::PenaltyConfig,
                                      beta_init::Vector{Float64};
                                      method::Symbol=:pijcv,
                                      scope::Symbol=:all,
                                      max_outer_iter::Int=20,
                                      inner_maxiter::Int=50,
                                      lambda_maxiter::Int=100,
                                      beta_tol::Float64=1e-4,
                                      lambda_tol::Float64=1e-3,
                                      verbose::Bool=false)
    # ... original alternating implementation preserved for reference ...
    # Validate scope
    scope ∈ (:all, :baseline, :covariates) || 
        throw(ArgumentError("scope must be :all, :baseline, or :covariates, got :$scope"))
    
    # Create scoped penalty config based on scope parameter
    scoped_config, fixed_baseline_lambdas, fixed_covariate_lambdas = _create_scoped_penalty_config(penalty_config, scope)
    
    n_lambda = scoped_config.n_lambda
    n_lambda > 0 || return (lambda=Float64[], beta=beta_init, criterion=NaN, 
                            converged=true, method_used=:none, penalty_config=penalty_config,
                            n_outer_iter=0)
    
    # Initialize λ from scoped penalty config
    lambda = extract_lambda_vector(scoped_config)
    
    # Initialize β
    beta = copy(beta_init)
    
    samplepaths = data.paths
    n_params = length(beta)
    n_subjects = length(samplepaths)
    
    if verbose
        println("Starting performance iteration (max $max_outer_iter outer iterations)")
        println("  β tolerance: $beta_tol, λ tolerance: $lambda_tol")
    end
    
    for iter in 1:max_outer_iter
        outer_iter = iter
        
        # ========== Step 1: Compute Hessians at current β ==========
        if verbose
            println("\nOuter iteration $iter:")
            println("  Computing subject-level gradients and Hessians...")
        end
        
        # Use fast batched/threaded variants - O(p) AD passes instead of O(n)
        # Note: These return gradients/Hessians of LOG-LIKELIHOOD (not loss)
        subject_grads_ll = compute_subject_gradients(beta, model, samplepaths)
        subject_hessians_ll = compute_subject_hessians_fast(beta, model, samplepaths)
        
        # Convert from log-likelihood to negative log-likelihood (loss) convention
        # This makes H_unpenalized positive definite at the MLE, as required for Cholesky
        subject_grads = -subject_grads_ll
        subject_hessians = [-H_i for H_i in subject_hessians_ll]
        
        # Aggregate unpenalized Hessian (of loss, so positive definite at MLE)
        H_unpenalized = zeros(n_params, n_params)
        for H_i in subject_hessians
            H_unpenalized .+= H_i
        end
        
        # Create state for λ optimization
        state = SmoothingSelectionState(
            copy(beta),
            H_unpenalized,
            subject_grads,
            subject_hessians,
            scoped_config,
            n_subjects,
            n_params,
            model,
            data
        )
        
        # ========== Step 2: Optimize λ given current β and Hessians ==========
        log_lambda_init = log.(lambda)
        
        # Try primary method first
        lambda_new, criterion, lambda_ok = optimize_lambda(state, log_lambda_init;
                                                            method=method_used, 
                                                            maxiter=lambda_maxiter,
                                                            verbose=verbose,
                                                            use_polyalgorithm=(iter == 1))
        
        # Fall back to GCV if PIJCV failed
        if !lambda_ok && method_used == :pijcv
            if verbose
                println("  PIJCV failed, trying GCV...")
            end
            method_used = :gcv
            lambda_new, criterion, lambda_ok = optimize_lambda(state, log_lambda_init;
                                                                method=:gcv, 
                                                                maxiter=lambda_maxiter,
                                                                verbose=verbose,
                                                                use_polyalgorithm=false)
        end
        
        # If both methods fail, use high penalty fallback
        if !lambda_ok
            if verbose
                @warn "λ optimization failed. Using default high penalty (λ=100)."
            end
            lambda_new = fill(100.0, n_lambda)
            criterion = NaN
            method_used = :fallback
        end
        
        final_criterion = criterion
        
        # ========== Step 3: Update β given new λ ==========
        # Use polyalgorithm on first iteration (β hasn't been penalized yet),
        # pure Ipopt thereafter (β is warm-started from previous penalized solution)
        use_polyalgorithm = (iter == 1)
        
        if verbose
            println("  Fitting penalized β (λ = $(round.(lambda_new, sigdigits=3)))...")
        end
        
        beta_new = fit_penalized_beta(model, data, lambda_new, scoped_config, beta;
                                      maxiters=inner_maxiter,
                                      use_polyalgorithm=use_polyalgorithm,
                                      verbose=verbose)
        
        # ========== Step 4: Check convergence ==========
        beta_norm = norm(beta)
        beta_change = norm(beta_new - beta) / (beta_norm + 1e-8)
        
        log_lambda_new = log.(lambda_new .+ 1e-10)  # Add small value to avoid log(0)
        log_lambda_old = log.(lambda .+ 1e-10)
        lambda_norm = norm(log_lambda_old)
        lambda_change = norm(log_lambda_new - log_lambda_old) / (lambda_norm + 1e-8)
        
        if verbose
            println("  β_change = $(round(beta_change, sigdigits=4)), λ_change = $(round(lambda_change, sigdigits=4)), criterion = $(round(criterion, sigdigits=6))")
        end
        
        if beta_change < beta_tol && lambda_change < lambda_tol
            converged = true
            beta = beta_new
            lambda = lambda_new
            if verbose
                println("\nConverged after $iter outer iterations")
            end
            break
        end
        
        # Update for next iteration
        beta = beta_new
        lambda = lambda_new
    end
    
    if !converged && verbose
        println("\nReached maximum outer iterations ($max_outer_iter) without convergence")
    end
    
    # Update penalty config with final λ
    updated_config = _merge_scoped_lambdas(penalty_config, scoped_config, lambda, 
                                           fixed_baseline_lambdas, fixed_covariate_lambdas, scope)
    
    return (
        lambda = lambda,
        beta = beta,
        criterion = final_criterion,
        converged = converged,
        method_used = method_used,
        penalty_config = updated_config,
        n_outer_iter = outer_iter
    )
end

"""
    compute_gcv_criterion(log_lambda::AbstractVector{T}, state::SmoothingSelectionState) where T<:Real

Compute Generalized Cross-Validation criterion.

GCV approximates leave-one-out CV using the effective degrees of freedom:
    V_GCV(λ) = n * deviance / (n - edf)²

where edf = tr(A) is the trace of the influence matrix.
"""
function compute_gcv_criterion(log_lambda::AbstractVector{T}, state::SmoothingSelectionState) where T<:Real
    lambda = exp.(log_lambda)
    
    # Build penalized Hessian with proper type
    H_lambda = _build_penalized_hessian(state.H_unpenalized, lambda, state.penalty_config)
    
    # Compute influence matrix trace: edf = tr(H_unpen * H_lambda^{-1})
    H_lambda_sym = Symmetric(H_lambda)
    
    H_lambda_inv = try
        inv(H_lambda_sym)
    catch e
        return T(1e10)
    end
    
    # edf = trace of influence matrix
    # Convert H_unpenalized to same type for matrix multiply
    H_unpen_T = convert(Matrix{T}, state.H_unpenalized)
    edf = tr(H_unpen_T * H_lambda_inv)
    
    # Compute deviance (sum of unpenalized losses)
    ll_subj = loglik_exact(state.beta_hat, state.data; neg=false, return_ll_subj=true)
    deviance = -sum(ll_subj)
    
    n = state.n_subjects
    
    # GCV criterion
    if n <= edf
        return T(1e10)  # Invalid: more parameters than data
    end
    
    gcv = n * deviance / (n - edf)^2
    return gcv
end

"""
    _update_penalty_lambda(config::PenaltyConfig, lambda::Vector{Float64}) -> PenaltyConfig

Create a new PenaltyConfig with updated lambda values.
"""
function _update_penalty_lambda(config::PenaltyConfig, lambda::Vector{Float64})
    lambda_idx = 1
    
    # Update baseline hazard terms
    new_terms = map(config.terms) do term
        new_term = PenaltyTerm(
            term.hazard_indices,
            term.S,
            lambda[lambda_idx],
            term.order,
            term.hazard_names
        )
        lambda_idx += 1
        new_term
    end
    
    # Update total hazard terms
    new_total_terms = map(config.total_hazard_terms) do term
        new_term = TotalHazardPenaltyTerm(
            term.origin,
            term.hazard_indices,
            term.S,
            lambda[lambda_idx],
            term.order
        )
        lambda_idx += 1
        new_term
    end
    
    # Update smooth covariate terms
    # Handle shared lambda groups
    if !isempty(config.shared_smooth_groups)
        # Build term -> lambda mapping
        term_to_lambda = Dict{Int, Float64}()
        
        for (group_idx, group) in enumerate(config.shared_smooth_groups)
            lam = lambda[lambda_idx]
            for term_idx in group
                term_to_lambda[term_idx] = lam
            end
            lambda_idx += 1
        end
        
        # Handle terms not in any group
        for term_idx in 1:length(config.smooth_covariate_terms)
            if !haskey(term_to_lambda, term_idx)
                term_to_lambda[term_idx] = lambda[lambda_idx]
                lambda_idx += 1
            end
        end
        
        # Update terms
        new_smooth_terms = map(enumerate(config.smooth_covariate_terms)) do (idx, term)
            SmoothCovariatePenaltyTerm(
                term.param_indices,
                term.S,
                term_to_lambda[idx],
                term.order,
                term.label,
                term.hazard_name
            )
        end
    else
        # No sharing - update each term independently
        new_smooth_terms = map(config.smooth_covariate_terms) do term
            new_term = SmoothCovariatePenaltyTerm(
                term.param_indices,
                term.S,
                lambda[lambda_idx],
                term.order,
                term.label,
                term.hazard_name
            )
            lambda_idx += 1
            new_term
        end
    end
    
    return PenaltyConfig(
        collect(new_terms),
        collect(new_total_terms),
        collect(new_smooth_terms),
        config.shared_lambda_groups,
        config.shared_smooth_groups,
        config.n_lambda
    )
end
