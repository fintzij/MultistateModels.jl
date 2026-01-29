# =============================================================================
# Joint (α, λ) Optimization for PIJCV with Adaptive Penalty Weighting
# =============================================================================
#
# This file implements joint optimization of smoothing parameters λ and 
# adaptive penalty weighting parameter α within the PIJCV framework.
#
# Key insight: Instead of alternating between α learning and λ selection,
# we optimize both simultaneously by including α as optimization variables.
#
# The main challenge is that the penalty matrix S depends on α:
#   S(α) = B' D' W(α) D B,  where W(α) = diag(Y(t_q)^(-α))
#
# To make this AD-compatible with proper eigenvalue normalization:
# 1. Precompute λ_max(S(α)) on a grid of α values at initialization
# 2. Interpolate during optimization (cubic or linear)
# 3. Build S(α) efficiently using cached ingredients
#
# =============================================================================

using LinearAlgebra
using BSplineKit
using Statistics: mean

# =============================================================================
# Cache Structure for Joint Optimization
# =============================================================================

"""
    AlphaJointOptCache

Cache for efficient joint (α, λ) optimization with eigenvalue normalization.

Stores precomputed ingredients for building S(α) and normalizing it so that
λ has consistent interpretation across α values.

The key design for AD-compatibility: precompute penalty contributions per
interval (which don't depend on α), then weight by (Y_q/Ȳ)^(-α) during optimization.

**Centering**: At-risk counts are centered by their mean, so:
    S(α) = Σ_q (Y_q/Ȳ)^(-α) * S_q
This makes log(λ) interpretable as the smoothing at the "average" at-risk level.

# Fields
- `term_idx::Int`: Index of penalty term this cache corresponds to
- `hazard_idx::Int`: Index of hazard in model.hazards
- `hazard`: The RuntimeSplineHazard object
- `basis`: Cached B-spline basis (avoids rebuilding)
- `order::Int`: Penalty derivative order (usually 2)
- `atrisk_centered::Vector{Float64}`: Centered at-risk (Y_q/Ȳ), length = n_intervals
- `atrisk_mean::Float64`: Mean at-risk count Ȳ (for reference/logging)
- `S_uniform::Matrix{Float64}`: Uniform penalty matrix S(α=0)
- `λ_max_uniform::Float64`: Maximum eigenvalue of S_uniform
- `alpha_grid::Vector{Float64}`: Grid of α values for interpolation
- `λ_max_grid::Vector{Float64}`: λ_max(S(α)) at each grid point
- `monotone::Int`: Monotonicity direction (0, 1, or -1)
- `interval_penalty_matrices::Vector{Matrix{Float64}}`: Penalty contribution from each interval
"""
struct AlphaJointOptCache
    term_idx::Int
    hazard_idx::Int
    hazard::Any  # RuntimeSplineHazard
    basis::Any   # BSplineBasis or RecombinedBSplineBasis
    order::Int
    atrisk_centered::Vector{Float64}  # Y_q / Ȳ (centered)
    atrisk_mean::Float64              # Ȳ (for reference)
    S_uniform::Matrix{Float64}
    λ_max_uniform::Float64
    alpha_grid::Vector{Float64}
    λ_max_grid::Vector{Float64}
    monotone::Int
    interval_penalty_matrices::Vector{Matrix{Float64}}  # S_q for each interval
end

"""
    build_alpha_joint_opt_cache(model, penalty, term_idx, hazard, atrisk;
                                 alpha_grid_points=21) -> AlphaJointOptCache

Build cache for joint (α, λ) optimization for a single penalty term.

Precomputes:
1. B-spline basis (cached for reuse)
2. Uniform penalty matrix and its max eigenvalue
3. λ_max(S(α)) on a fine grid for interpolation during optimization
4. Per-interval penalty matrices for AD-compatible S(α) construction
5. Centered at-risk counts (Y_q/Ȳ) so log(λ) is interpretable at mean at-risk level

# Arguments
- `model`: MultistateProcess
- `penalty`: QuadraticPenalty
- `term_idx`: Index of penalty term
- `hazard`: RuntimeSplineHazard for this term
- `atrisk`: At-risk interval averages
- `alpha_grid_points=21`: Number of points in α grid (default: 0.0:0.1:2.0)

# Returns
- `AlphaJointOptCache`: Cache for efficient S(α) computation
"""
function build_alpha_joint_opt_cache(model::MultistateProcess,
                                      penalty::QuadraticPenalty,
                                      term_idx::Int,
                                      hazard,
                                      atrisk::Vector{Float64};
                                      alpha_grid_points::Int = 21)
    term = penalty.terms[term_idx]
    
    # Rebuild basis (cache it)
    basis = _rebuild_spline_basis(hazard)
    
    # Get order from term
    order = term.order
    
    # Build uniform penalty matrix
    S_uniform = build_penalty_matrix(basis, order)
    
    # Handle monotone splines
    monotone_dir = hazard.monotone
    if monotone_dir != 0
        S_uniform = transform_penalty_for_monotone(S_uniform, basis; direction=monotone_dir)
    end
    
    # Compute max eigenvalue of uniform penalty
    λ_max_uniform = maximum(eigvals(Symmetric(S_uniform)))
    
    # Precompute interval penalty matrices for AD-compatible S(α) construction
    interval_penalty_matrices = _compute_interval_penalty_matrices(basis, order, monotone_dir)
    
    # Center at-risk counts: compute Y_q / Ȳ
    # This makes log(λ) interpretable as smoothing at the "average" at-risk level
    atrisk_clamped = max.(atrisk, 1.0)  # Clamp to avoid issues with zeros
    atrisk_mean = mean(atrisk_clamped)
    atrisk_centered = atrisk_clamped ./ atrisk_mean
    
    # Build α grid and compute λ_max(S(α)) at each point
    # IMPORTANT: Must use centered at-risk for consistency with build_penalty_matrix_at_alpha
    alpha_grid = range(0.0, 2.0, length=alpha_grid_points) |> collect
    λ_max_grid = Vector{Float64}(undef, alpha_grid_points)
    
    for (i, α) in enumerate(alpha_grid)
        # Compute S(α) using CENTERED at-risk counts (same as optimization)
        # Manually compute weighted sum since AtRiskWeighting uses raw counts
        K = size(interval_penalty_matrices[1], 1)
        S_alpha = zeros(K, K)
        for q in 1:length(interval_penalty_matrices)
            w_q = atrisk_centered[q] ^ (-α)
            S_alpha .+= w_q .* interval_penalty_matrices[q]
        end
        
        # Transform for monotone if needed
        if monotone_dir != 0
            S_alpha = transform_penalty_for_monotone(S_alpha, basis; direction=monotone_dir)
        end
        
        λ_max_grid[i] = maximum(eigvals(Symmetric(S_alpha)))
    end
    
    # Get hazard index
    hazard_idx = findfirst(h -> h === hazard, model.hazards)
    if isnothing(hazard_idx)
        hazard_idx = 0  # Fallback if not found
    end
    
    return AlphaJointOptCache(
        term_idx,
        hazard_idx,
        hazard,
        basis,
        order,
        atrisk_centered,  # Centered: Y_q / Ȳ
        atrisk_mean,      # Store mean for reference
        S_uniform,
        λ_max_uniform,
        alpha_grid,
        λ_max_grid,
        monotone_dir,
        interval_penalty_matrices
    )
end

"""
    _compute_interval_penalty_matrices(basis, order, monotone_dir) -> Vector{Matrix{Float64}}

Compute the (unweighted) penalty contribution from each knot interval.

This enables AD-compatible S(α) construction:
    S(α) = Σ_q Y_q^(-α) * S_q
where S_q is the penalty contribution from interval q.

# Returns
- Vector of K×K matrices, one per interval
"""
function _compute_interval_penalty_matrices(basis, order::Int, monotone_dir::Int)
    # Get knot vector
    knots = if basis isa BSplineBasis
        collect(BSplineKit.knots(basis))
    elseif basis isa RecombinedBSplineBasis
        collect(BSplineKit.knots(parent(basis)))
    else
        throw(ArgumentError("Unsupported basis type: $(typeof(basis))"))
    end
    
    K = length(basis)
    spline_order = BSplineKit.order(basis)
    deriv_order = spline_order - order
    
    # Unique interior breakpoints
    unique_knots = unique(knots)
    n_intervals = length(unique_knots) - 1
    
    # Gauss-Legendre quadrature
    n_gauss = max(deriv_order, 3)
    gl_nodes, gl_weights = _gauss_legendre(n_gauss)
    
    # Compute contribution from each interval
    interval_matrices = Vector{Matrix{Float64}}(undef, n_intervals)
    
    for q in 1:n_intervals
        a, b = unique_knots[q], unique_knots[q + 1]
        h = b - a
        
        S_q = zeros(K, K)
        
        if h >= 1e-14
            # Transform Gauss points to [a, b]
            t_points = @. a + (gl_nodes + 1) * h / 2
            w_scaled = @. gl_weights * h / 2  # No interval weight here
            
            for (i_pt, t) in enumerate(t_points)
                deriv_vals = _evaluate_basis_derivatives(basis, t, order)
                w = w_scaled[i_pt]
                for i in 1:K
                    for j in i:K
                        S_q[i, j] += w * deriv_vals[i] * deriv_vals[j]
                    end
                end
            end
            
            # Symmetrize
            for i in 1:K
                for j in (i+1):K
                    S_q[j, i] = S_q[i, j]
                end
            end
        end
        
        # Transform for monotone if needed
        if monotone_dir != 0
            S_q = transform_penalty_for_monotone(S_q, basis; direction=monotone_dir)
        end
        
        interval_matrices[q] = S_q
    end
    
    return interval_matrices
end

"""
    interpolate_lambda_max(cache::AlphaJointOptCache, α) -> Real

Interpolate λ_max(S(α)) using precomputed grid.

Uses linear interpolation for simplicity and robustness.
"""
function interpolate_lambda_max(cache::AlphaJointOptCache, α::T) where T<:Real
    grid = cache.alpha_grid
    vals = cache.λ_max_grid
    n = length(grid)
    
    # Clamp α to grid bounds
    α_clamped = clamp(α, grid[1], grid[end])
    
    # Find interval (linear search is fine for small grids)
    idx = 1
    for i in 1:(n-1)
        if α_clamped >= grid[i] && α_clamped <= grid[i+1]
            idx = i
            break
        end
    end
    
    # Linear interpolation
    t = (α_clamped - grid[idx]) / (grid[idx+1] - grid[idx])
    return vals[idx] + t * (vals[idx+1] - vals[idx])
end

"""
    build_penalty_matrix_at_alpha(cache::AlphaJointOptCache, α::T) where T -> Matrix{T}

Build normalized penalty matrix S(α) using cached ingredients (AD-compatible).

The construction uses precomputed interval penalty matrices with centered at-risk:
    S(α) = Σ_q (Y_q/Ȳ)^(-α) * S_q

This centering makes log(λ) interpretable as smoothing at the mean at-risk level.

The normalization ensures λ has consistent interpretation:
    S_normalized(α) = S(α) * (λ_max(S_uniform) / λ_max(S(α)))

where λ_max(S(α)) is interpolated from a precomputed grid.

# Arguments
- `cache`: AlphaJointOptCache with precomputed data
- `α`: Current alpha value (can be Dual for AD)

# Returns
- `Matrix{T}`: Normalized penalty matrix (same eltype as α for AD compatibility)
"""
function build_penalty_matrix_at_alpha(cache::AlphaJointOptCache, α::T) where T<:Real
    atrisk_centered = cache.atrisk_centered  # Already Y_q / Ȳ
    interval_mats = cache.interval_penalty_matrices
    n_intervals = length(interval_mats)
    K = size(interval_mats[1], 1)
    
    # Build S(α) = Σ_q (Y_q/Ȳ)^(-α) * S_q  (AD-compatible)
    # Initialize with correct element type to support Dual numbers
    S_alpha = zeros(T, K, K)
    
    for q in 1:n_intervals
        # Weight for this interval: (Y_q/Ȳ)^(-α)
        # Note: atrisk_centered is already clamped and divided by mean
        Y_centered = atrisk_centered[q]
        w_q = Y_centered ^ (-α)  # This is AD-compatible
        
        # Add weighted interval contribution
        S_q = interval_mats[q]
        for i in 1:K
            for j in 1:K
                S_alpha[i, j] += w_q * S_q[i, j]
            end
        end
    end
    
    # Normalize using interpolated λ_max
    λ_max_alpha = interpolate_lambda_max(cache, α)
    
    if λ_max_alpha > 1e-14
        scale = cache.λ_max_uniform / λ_max_alpha
        S_alpha = S_alpha * scale
    end
    
    return S_alpha
end

# =============================================================================
# Joint Optimization State
# =============================================================================

"""
    JointAlphaLambdaState

State for joint (α, λ) optimization within PIJCV.

Extends SmoothingSelectionState with alpha learning infrastructure.

# Fields (from SmoothingSelectionState)
- `beta_hat`, `H_unpenalized`, `subject_grads`, `subject_hessians`
- `penalty_config`, `n_subjects`, `n_params`, `model`, `data`

# Additional fields for joint optimization
- `n_lambda::Int`: Number of λ parameters
- `n_alpha::Int`: Number of α parameters  
- `alpha_caches::Vector{AlphaJointOptCache}`: One cache per learnable α
- `alpha_term_map::Dict{Int,Int}`: term_idx → alpha_idx mapping
"""
mutable struct JointAlphaLambdaState
    # From SmoothingSelectionState
    beta_hat::Vector{Float64}
    H_unpenalized::Matrix{Float64}
    subject_grads::Matrix{Float64}
    subject_hessians::Vector{Matrix{Float64}}
    penalty_config::QuadraticPenalty
    n_subjects::Int
    n_params::Int
    model::MultistateProcess
    data::Any  # ExactData, MPanelData, or MCEMData
    pijcv_eval_cache::Union{Nothing, PIJCVEvaluationCache}
    
    # Joint optimization specifics
    n_lambda::Int
    n_alpha::Int
    alpha_caches::Vector{AlphaJointOptCache}
    alpha_term_map::Dict{Int,Int}  # term_idx → alpha_idx
    current_alphas::Vector{Float64}  # Current α values
end

"""
    build_joint_optimization_state(state::SmoothingSelectionState,
                                    model, penalty, alpha_info, alpha_groups) -> JointAlphaLambdaState

Convert a SmoothingSelectionState to JointAlphaLambdaState for joint optimization.
"""
function build_joint_optimization_state(
    state::SmoothingSelectionState,
    model::MultistateProcess,
    penalty::QuadraticPenalty,
    alpha_info::Dict{Int, AlphaLearningInfo},
    alpha_groups::Vector{Vector{Int}}
)
    # Build alpha caches
    alpha_caches = AlphaJointOptCache[]
    alpha_term_map = Dict{Int,Int}()
    
    alpha_idx = 0
    for group in alpha_groups
        alpha_idx += 1
        # Use first term in group as representative
        term_idx = group[1]
        info = alpha_info[term_idx]
        
        cache = build_alpha_joint_opt_cache(
            model, penalty, term_idx, info.hazard, info.atrisk
        )
        push!(alpha_caches, cache)
        
        # Map all terms in group to this alpha
        for idx in group
            alpha_term_map[idx] = alpha_idx
        end
    end
    
    n_lambda = n_hyperparameters(penalty)
    n_alpha = length(alpha_groups)
    
    # Initialize alphas at 1.0 (uniform weighting)
    current_alphas = ones(n_alpha)
    
    return JointAlphaLambdaState(
        state.beta_hat,
        state.H_unpenalized,
        state.subject_grads,
        state.subject_hessians,
        state.penalty_config,
        state.n_subjects,
        state.n_params,
        state.model,
        state.data,
        state.pijcv_eval_cache,
        n_lambda,
        n_alpha,
        alpha_caches,
        alpha_term_map,
        current_alphas
    )
end

# =============================================================================
# Joint PIJCV Criterion
# =============================================================================

"""
    _build_penalized_hessian_joint(H_unpen, lambda, alpha_vec, state) -> Matrix

Build penalized Hessian H + λS(α) for joint optimization.

Rebuilds penalty matrices at current α values with proper normalization.
"""
function _build_penalized_hessian_joint(
    H_unpen::Matrix{Float64},
    lambda::AbstractVector{T},
    alpha_vec::AbstractVector,
    state::JointAlphaLambdaState
) where T<:Real
    # Start with unpenalized Hessian
    H = convert(Matrix{T}, copy(H_unpen))
    
    penalty = state.penalty_config
    lambda_idx = 1
    
    # Process each baseline hazard term
    for (term_idx, term) in enumerate(penalty.terms)
        idx = term.hazard_indices
        lam = lambda[lambda_idx]
        
        # Check if this term has learnable α
        if haskey(state.alpha_term_map, term_idx)
            alpha_idx = state.alpha_term_map[term_idx]
            α = alpha_vec[alpha_idx]
            cache = state.alpha_caches[alpha_idx]
            
            # Build S(α) with normalization
            S_alpha = build_penalty_matrix_at_alpha(cache, α)
            
            # Add λS(α) to Hessian
            @inbounds for i in idx, j in idx
                H[i, j] += lam * S_alpha[i - first(idx) + 1, j - first(idx) + 1]
            end
        else
            # Use stored S matrix (no α learning for this term)
            @inbounds for i in idx, j in idx
                H[i, j] += lam * term.S[i - first(idx) + 1, j - first(idx) + 1]
            end
        end
        lambda_idx += 1
    end
    
    # Total hazard terms (unchanged - no α learning here)
    for term in penalty.total_hazard_terms
        lam = lambda[lambda_idx]
        for idx_range in term.hazard_indices
            @inbounds for i in idx_range, j in idx_range
                H[i, j] += lam * term.S[i - first(idx_range) + 1, j - first(idx_range) + 1]
            end
        end
        lambda_idx += 1
    end
    
    # Smooth covariate terms (unchanged)
    if !isempty(penalty.shared_smooth_groups)
        term_to_lambda = Dict{Int, Int}()
        for (group_idx, group) in enumerate(penalty.shared_smooth_groups)
            for tidx in group
                term_to_lambda[tidx] = lambda_idx
            end
            lambda_idx += 1
        end
        for (tidx, term) in enumerate(penalty.smooth_covariate_terms)
            if !haskey(term_to_lambda, tidx)
                term_to_lambda[tidx] = lambda_idx
                lambda_idx += 1
            end
        end
        for (tidx, term) in enumerate(penalty.smooth_covariate_terms)
            lam = lambda[term_to_lambda[tidx]]
            indices = term.param_indices
            @inbounds for (i, pi) in enumerate(indices)
                for (j, pj) in enumerate(indices)
                    H[pi, pj] += lam * term.S[i, j]
                end
            end
        end
    else
        for term in penalty.smooth_covariate_terms
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
    compute_pijcv_criterion_joint(theta::AbstractVector{T}, state::JointAlphaLambdaState) where T

Joint PIJCV criterion for (log λ, α) optimization.

theta = [log(λ₁), ..., log(λₖ), α₁, ..., αₘ]

# Arguments
- `theta`: Combined optimization variables
- `state`: JointAlphaLambdaState with cached data

# Returns
- `T`: PIJCV criterion value (lower is better)
"""
function compute_pijcv_criterion_joint(theta::AbstractVector{T}, state::JointAlphaLambdaState) where T<:Real
    # Split theta into lambda and alpha components
    n_lambda = state.n_lambda
    n_alpha = state.n_alpha
    
    log_lambda = @view theta[1:n_lambda]
    alpha_vec = @view theta[n_lambda+1:n_lambda+n_alpha]
    
    lambda = exp.(log_lambda)
    
    # Build penalized Hessian with S(α)
    H_lambda = _build_penalized_hessian_joint(
        state.H_unpenalized, lambda, alpha_vec, state
    )
    
    # Check if we're in AD mode
    use_cholesky_downdate = (T === Float64)
    
    # Try Cholesky factorization
    H_lambda_sym = Symmetric(H_lambda)
    chol_fact = if use_cholesky_downdate
        try
            cholesky(H_lambda_sym)
        catch e
            nothing
        end
    else
        nothing
    end
    
    # If Cholesky failed in Float64 mode, return large value
    if isnothing(chol_fact) && use_cholesky_downdate
        return T(1e10)
    end
    
    # Get subject-level likelihoods at current estimate
    ll_subj_base = loglik_exact(state.beta_hat, state.data; neg=false, return_ll_subj=true)
    
    # Build evaluation cache if needed
    eval_cache = if isnothing(state.pijcv_eval_cache)
        cache = build_pijcv_eval_cache(state.data)
        state.pijcv_eval_cache = cache
        cache
    else
        state.pijcv_eval_cache
    end
    
    # Compute NCV criterion
    V = zero(T)
    
    for i in 1:state.n_subjects
        g_i = @view state.subject_grads[:, i]
        H_i = state.subject_hessians[i]
        
        # Solve for LOO perturbation
        delta_i = if use_cholesky_downdate && chol_fact !== nothing
            _solve_loo_newton_step(chol_fact, H_i, g_i)
        else
            nothing
        end
        
        if isnothing(delta_i)
            H_lambda_loo = H_lambda - H_i
            H_loo_sym = Symmetric(H_lambda_loo)
            delta_i = try
                H_loo_sym \ collect(g_i)
            catch e
                return T(1e10)
            end
        end
        
        # LOO parameters
        beta_loo = state.beta_hat .+ delta_i
        
        # Check positivity (baseline parameters must be > 0)
        lb = state.model.bounds.lb
        if any(j -> lb[j] > 0 && beta_loo[j] <= 0, eachindex(beta_loo))
            V += ll_subj_base[i] + dot(g_i, delta_i) + 0.5 * dot(delta_i, H_i * delta_i)
            continue
        end
        
        # Evaluate actual subject likelihood at LOO parameters
        ll_i_loo = try
            _evaluate_subject_likelihood_at_beta(eval_cache, i, beta_loo, state.data)
        catch
            nothing
        end
        
        if isnothing(ll_i_loo) || !isfinite(ll_i_loo)
            V += ll_subj_base[i] + dot(g_i, delta_i) + 0.5 * dot(delta_i, H_i * delta_i)
        else
            V += -ll_i_loo
        end
    end
    
    return V
end

# =============================================================================
# Helper Functions
# =============================================================================

"""
    extract_alpha_from_solution(sol_u, n_lambda, n_alpha) -> Vector{Float64}

Extract α values from joint optimization solution.
"""
function extract_alpha_from_solution(sol_u::Vector{Float64}, n_lambda::Int, n_alpha::Int)
    return sol_u[n_lambda+1:n_lambda+n_alpha]
end

"""
    update_penalty_with_joint_alphas(penalty, alpha_caches, alpha_term_map, alpha_vec) -> QuadraticPenalty

Update penalty matrices at final α values after joint optimization.
"""
function update_penalty_with_joint_alphas(
    penalty::QuadraticPenalty,
    alpha_caches::Vector{AlphaJointOptCache},
    alpha_term_map::Dict{Int,Int},
    alpha_vec::Vector{Float64}
)
    new_terms = copy(penalty.terms)
    
    for (term_idx, alpha_idx) in alpha_term_map
        α = alpha_vec[alpha_idx]
        cache = alpha_caches[alpha_idx]
        
        # Build normalized S(α)
        S_new = build_penalty_matrix_at_alpha(cache, α)
        
        term = penalty.terms[term_idx]
        new_terms[term_idx] = PenaltyTerm(
            term.hazard_indices,
            S_new,
            term.lambda,
            term.order,
            term.hazard_names
        )
    end
    
    return QuadraticPenalty(
        new_terms,
        penalty.total_hazard_terms,
        penalty.smooth_covariate_terms,
        penalty.shared_lambda_groups,
        penalty.shared_smooth_groups,
        penalty.n_lambda
    )
end

"""
    run_joint_alpha_lambda_optimization(state::JointAlphaLambdaState, selector::PIJCVSelector;
                                         lambda_init, outer_maxiter, lambda_tol, verbose) -> (sol, best_criterion)

Run joint (α, λ) optimization using PIJCV criterion.

# Arguments
- `state`: JointAlphaLambdaState with precomputed caches
- `selector`: PIJCVSelector (for configuration options)
- `lambda_init`: Initial λ values (from EFS or previous iteration)
- `alpha_init`: Initial α values (default 1.0)
- `outer_maxiter`: Maximum optimization iterations
- `lambda_tol`: Convergence tolerance
- `verbose`: Print progress

# Returns
- `sol`: Optimization solution object
- `best_criterion`: Final criterion value
"""
function run_joint_alpha_lambda_optimization(
    state::JointAlphaLambdaState,
    selector::PIJCVSelector;
    lambda_init::Vector{Float64},
    alpha_init::Vector{Float64} = ones(state.n_alpha),
    outer_maxiter::Int = 100,
    lambda_tol::Float64 = 1e-3,
    verbose::Bool = false
)
    n_lambda = state.n_lambda
    n_alpha = state.n_alpha
    n_subjects = state.n_subjects
    n_params = state.n_params
    
    # Bounds for log(λ) and α
    log_lb_scalar, log_ub_scalar = compute_lambda_bounds(n_subjects, n_params)
    log_lb = fill(log_lb_scalar, n_lambda)
    log_ub = fill(log_ub_scalar, n_lambda)
    alpha_lb = fill(0.0, n_alpha)
    alpha_ub = fill(2.0, n_alpha)
    
    theta_lb = vcat(log_lb, alpha_lb)
    theta_ub = vcat(log_ub, alpha_ub)
    
    # Initial point
    log_lambda_init = log.(lambda_init[1:n_lambda])
    theta_init = vcat(log_lambda_init, alpha_init)
    
    if verbose
        println("  Joint (λ, α) optimization: $n_lambda λ params, $n_alpha α params")
    end
    
    # Define criterion function
    n_criterion_evals = Ref(0)
    
    function criterion(theta_vec, _)
        n_criterion_evals[] += 1
        return compute_pijcv_criterion_joint(theta_vec, state)
    end
    
    # Setup optimization with SecondOrder AD (IPNewton requires Hessians)
    adtype = DifferentiationInterface.SecondOrder(
        Optimization.AutoForwardDiff(), 
        Optimization.AutoForwardDiff()
    )
    optf = OptimizationFunction(criterion, adtype)
    prob = OptimizationProblem(optf, theta_init, nothing; lb=theta_lb, ub=theta_ub)
    
    # Solve with IPNewton
    sol = solve(prob, OptimizationOptimJL.IPNewton();
                maxiters=outer_maxiter,
                abstol=lambda_tol,
                reltol=lambda_tol)
    
    best_criterion = sol.objective
    converged = sol.retcode == ReturnCode.Success || sol.retcode == ReturnCode.MaxIters
    
    if verbose
        lambda_opt = exp.(sol.u[1:n_lambda])
        alpha_opt = sol.u[n_lambda+1:end]
        println("  Final: log(λ)=$(round.(sol.u[1:n_lambda], digits=2)), α=$(round.(alpha_opt, digits=3))")
        println("  λ=$(round.(lambda_opt, sigdigits=3)), V=$(round(best_criterion, digits=3))")
        println("  Criterion evaluations: $(n_criterion_evals[])")
        println(converged ? "  Converged successfully" : "  Warning: $(sol.retcode)")
    end
    
    return sol, best_criterion
end

"""
    _nested_optimization_pijcv_joint_alpha(model, data, penalty, selector; kwargs...) -> HyperparameterSelectionResult

Joint (α, λ) optimization for PIJCV with adaptive penalty weighting.

This function replaces the alternating approach for α learning with a single
joint optimization over both smoothing parameters λ and weighting parameters α.

# Algorithm
1. Get initial λ estimate via EFS (or use provided warm-start)
2. Fit β at initial λ
3. Compute subject gradients/Hessians
4. Build joint state with precomputed α caches (including eigenvalue grids)
5. Optimize (log λ, α) jointly using PIJCV criterion
6. Return result with optimal λ, α, and warm-start β

# Arguments
- `model::MultistateProcess`: Model for likelihood evaluation
- `data::ExactData`: Data container
- `penalty::AbstractPenalty`: Penalty configuration
- `selector::PIJCVSelector`: PIJCV selector
- `beta_init`: Initial coefficient estimate
- `alpha_info`: Dict mapping term_idx to AlphaLearningInfo
- `alpha_groups`: Groups of terms sharing same α

# Returns
- `HyperparameterSelectionResult` with optimal λ, α applied to penalty, warmstart_beta
"""
function _nested_optimization_pijcv_joint_alpha(
    model::MultistateProcess,
    data::ExactData,
    penalty::AbstractPenalty,
    selector::PIJCVSelector;
    beta_init::Vector{Float64},
    inner_maxiter::Int = 50,
    outer_maxiter::Int = 100,
    lambda_tol::Float64 = 1e-3,
    lambda_init::Union{Nothing, Vector{Float64}} = nothing,
    alpha_info::Dict{Int, AlphaLearningInfo},
    alpha_groups::Vector{Vector{Int}},
    verbose::Bool = false
)
    lb, ub = model.bounds.lb, model.bounds.ub
    n_lambda = n_hyperparameters(penalty)
    n_subjects = length(data.paths)
    n_params = length(beta_init)
    samplepaths = data.paths
    
    penalty_config = penalty isa QuadraticPenalty ? penalty : build_penalty_config(model, SplinePenalty())
    
    if verbose
        n_alpha = length(alpha_groups)
        println("Joint (λ, α) optimization via PIJCV")
        println("  n_lambda: $n_lambda, n_alpha: $n_alpha")
    end
    
    # Step 1: Get initial λ estimate
    initial_lambda = if !isnothing(lambda_init) && length(lambda_init) >= n_lambda
        if verbose
            println("  Using provided λ warm-start")
        end
        lambda_init[1:n_lambda]
    else
        if verbose
            println("  Getting EFS initial estimate...")
        end
        efs_result = _nested_optimization_reml(model, data, penalty;
                                               beta_init=beta_init,
                                               inner_maxiter=inner_maxiter,
                                               outer_maxiter=30,
                                               lambda_tol=0.1,
                                               verbose=false)
        efs_result.lambda[1:n_lambda]
    end
    
    # Step 2: Fit β at initial λ
    lambda_expanded = n_lambda == 1 ? fill(initial_lambda[1], n_hyperparameters(penalty_config)) : initial_lambda
    initial_penalty = set_hyperparameters(penalty, lambda_expanded)
    
    beta_at_lambda = _fit_inner_coefficients(model, data, initial_penalty, beta_init;
                                              lb=lb, ub=ub, maxiter=inner_maxiter)
    
    # Step 3: Compute subject gradients and Hessians
    subject_grads_ll, subject_hessians_ll = compute_subject_grads_and_hessians_fast(
        beta_at_lambda, model, samplepaths; use_threads=:auto)
    
    subject_grads = -subject_grads_ll  # Convert to loss convention
    subject_hessians = [-H for H in subject_hessians_ll]
    H_unpenalized = sum(subject_hessians)
    
    # Step 4: Build smoothing selection state
    base_state = SmoothingSelectionState(
        copy(beta_at_lambda),
        H_unpenalized,
        subject_grads,
        subject_hessians,
        penalty_config,
        n_subjects,
        n_params,
        model,
        data,
        nothing  # pijcv_eval_cache - will be built lazily
    )
    
    # Step 5: Build joint state with alpha caches
    joint_state = build_joint_optimization_state(
        base_state, model, penalty_config, alpha_info, alpha_groups
    )
    
    # Step 6: Run joint optimization
    n_alpha = joint_state.n_alpha
    alpha_init_vec = ones(n_alpha)  # Start from uniform weighting
    
    sol, best_criterion = run_joint_alpha_lambda_optimization(
        joint_state, selector;
        lambda_init=initial_lambda,
        alpha_init=alpha_init_vec,
        outer_maxiter=outer_maxiter,
        lambda_tol=lambda_tol,
        verbose=verbose
    )
    
    # Extract results
    optimal_log_lambda = sol.u[1:n_lambda]
    optimal_alpha = sol.u[n_lambda+1:n_lambda+n_alpha]
    optimal_lambda = exp.(optimal_log_lambda)
    
    converged = sol.retcode == ReturnCode.Success || sol.retcode == ReturnCode.MaxIters
    
    # Update penalty with optimal α values
    updated_penalty = update_penalty_with_joint_alphas(
        penalty_config, joint_state.alpha_caches, joint_state.alpha_term_map, optimal_alpha
    )
    
    # Set optimal λ
    optimal_lambda_vec = n_lambda == 1 ? fill(optimal_lambda[1], n_hyperparameters(updated_penalty)) : optimal_lambda
    updated_penalty = set_hyperparameters(updated_penalty, optimal_lambda_vec)
    
    # Compute EDF at optimal (λ, α, β)
    edf = compute_edf(beta_at_lambda, optimal_lambda_vec, updated_penalty, model, data)
    
    method = selector.nfolds == 0 ? :pijcv_joint : Symbol("pijcv$(selector.nfolds)_joint")
    
    if verbose
        println("  Final λ: $(round.(optimal_lambda, sigdigits=4))")
        println("  Final α: $(round.(optimal_alpha, digits=3))")
        println("  EDF: $(round(edf.total, digits=2))")
    end
    
    return HyperparameterSelectionResult(
        optimal_lambda_vec,
        beta_at_lambda,  # warmstart_beta
        updated_penalty,
        best_criterion,
        edf,
        converged,
        method,
        0,  # n_criterion_evals tracked internally
        (log_lambda = optimal_log_lambda, alpha = optimal_alpha, retcode = sol.retcode)
    )
end
