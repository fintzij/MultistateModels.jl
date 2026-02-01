# =============================================================================
# Performance Iteration for Smoothing Parameter Selection
# =============================================================================
#
# Implements Wood (2024) "On Neighbourhood Cross Validation" performance
# iteration algorithm for efficient PIJCV-based λ selection.
#
# Key insight: Instead of nested optimization (outer loop for λ, inner loop
# for β that runs to convergence), performance iteration alternates SINGLE
# Newton steps for β and λ, achieving O(n) total iterations instead of O(n²).
#
# Algorithm (Phase 1: Scalar λ):
#   for iter = 1:maxiter
#       # Step 1: One Newton step for β
#       H_λ = H_unpenalized + λ * S
#       g = -∇ℓ(β) + λ * S * β
#       Δβ = -(H_λ \ g)
#       β_new = project_to_bounds(β + Δβ, lb, ub)
#
#       # Step 2: One Newton step for λ on PIJCV criterion
#       V, ∇λ_V, Hλ_V = compute_pijcv_criterion_with_derivatives(β_new, λ, ...)
#       Δλ = -(Hλ_V \ ∇λ_V)  # Scalar: Δλ = -∇λ_V / Hλ_V
#       λ_new = max(λ + Δλ, 1e-8)
#
#       # Check convergence
#       if converged: break
#   end
#
# REFERENCES:
# - Wood, S.N. (2024). "On Neighbourhood Cross Validation." arXiv:2404.16490v4
# - Marra & Radice (2020). "Penalized regression splines: theory and application"
#
# =============================================================================

# NOTE: LinearAlgebra and ForwardDiff are imported at package level in MultistateModels.jl

# =============================================================================
# Performance Iteration Constants and Helper Functions
# =============================================================================
# NOTE: Constants (PI_DEFAULT_MAXITER, PI_BETA_TOL, etc.) and helper functions
# (project_to_bounds, armijo_line_search, etc.) are now defined in common.jl 
# to allow sharing with dispatch_markov.jl and other dispatch files.
# See common.jl for these definitions.
# =============================================================================

"""
    compute_criterion_with_derivatives(β, λ, penalty_config, model, data, selector)

Compute selection criterion V(λ) and its first/second derivatives w.r.t. λ.

This is the unified interface for all criteria (PIJCV, EFS, PERF). The algorithm
is criterion-agnostic - only the V(λ) function changes.

# Arguments
- `β`: Current coefficient estimate
- `λ`: Smoothing parameter(s)
- `penalty_config`: Penalty configuration
- `model`: MultistateProcess model
- `data`: Data container (ExactData, MPanelData, or MCEMSelectionData)
- `selector`: Selector type (PIJCVSelector, REMLSelector, or PERFSelector)

# Returns
- `V::Float64`: Criterion value
- `∇λ_V`: Gradient (scalar if n_lambda=1, vector otherwise)
- `Hλ_V`: Hessian (scalar if n_lambda=1, matrix otherwise)

# Notes
This unified function handles all data types via the `_compute_subject_grads_hessians`
dispatch helper defined in pijcv.jl.
"""
function compute_criterion_with_derivatives(
    β::Vector{Float64},
    λ::Vector{Float64},
    penalty_config::PenaltyConfig,
    model::MultistateProcess,
    data::D,  # Generic data type - works with ExactData, MPanelData, MCEMSelectionData
    selector::AbstractHyperparameterSelector
) where D
    # Build SmoothingSelectionState (shared across all criteria)
    # Use dispatch helper to handle all data types
    subject_grads_ll, subject_hessians_ll = _compute_subject_grads_hessians(β, model, data)
    
    # Convert to loss convention (negative log-likelihood)
    subject_grads = -subject_grads_ll
    subject_hessians = [-H for H in subject_hessians_ll]
    H_unpenalized = sum(subject_hessians)
    n_subjects = _get_n_subjects(data)
    n_params = length(β)
    
    state = SmoothingSelectionState{D}(
        β, H_unpenalized, subject_grads, subject_hessians,
        penalty_config, n_subjects, n_params, model, data, nothing
    )
    
    # Dispatch to criterion-specific function
    return _compute_criterion_with_derivatives_impl(state, λ, selector)
end

# Dispatch for PIJCVSelector
function _compute_criterion_with_derivatives_impl(
    state::SmoothingSelectionState,
    λ::Vector{Float64},
    selector::PIJCVSelector
)
    return _compute_pijcv_derivatives(state, λ, selector.nfolds)
end

# Dispatch for REMLSelector (EFS criterion)
function _compute_criterion_with_derivatives_impl(
    state::SmoothingSelectionState,
    λ::Vector{Float64},
    selector::REMLSelector
)
    return _compute_efs_derivatives(state, λ)
end

# Dispatch for PERFSelector
function _compute_criterion_with_derivatives_impl(
    state::SmoothingSelectionState,
    λ::Vector{Float64},
    selector::PERFSelector
)
    return _compute_perf_derivatives(state, λ)
end

"""
    _compute_pijcv_derivatives(state, λ, nfolds) -> (V, ∇V, H_V)

Compute PIJCV criterion and derivatives using ForwardDiff through log(λ).
"""
function _compute_pijcv_derivatives(
    state::SmoothingSelectionState,
    λ::Vector{Float64},
    nfolds::Int
)
    n_lambda = length(λ)
    log_lambda = log.(λ)
    
    if n_lambda == 1
        # Scalar case: efficient second-order AD
        function V_of_rho(rho_scalar)
            rho_vec = [rho_scalar]
            return compute_pijcv_criterion(rho_vec, state; check_conditioning=false)
        end
        
        V_val = V_of_rho(log_lambda[1])
        dV_drho = ForwardDiff.derivative(V_of_rho, log_lambda[1])
        d2V_drho2 = ForwardDiff.derivative(rho -> ForwardDiff.derivative(V_of_rho, rho), log_lambda[1])
        
        grad_lambda, hess_lambda = transform_log_scale_derivatives(λ, dV_drho, d2V_drho2)
        
        # DEBUG: Trace Newton step components
        @debug "PIJCV Newton step trace" begin
            newton_step = hess_lambda > 0 ? -grad_lambda / hess_lambda : NaN
            "λ=$(λ[1]), V=$V_val, dV/dρ=$dV_drho, d²V/dρ²=$d2V_drho2, " *
            "dV/dλ=$grad_lambda, d²V/dλ²=$hess_lambda, " *
            "Newton Δλ=$newton_step, direction=$(newton_step > 0 ? "UP" : "DOWN")"
        end
        
        return (V_val, grad_lambda, hess_lambda)
    else
        # Vector case
        function V_of_rho_vec(rho_vec)
            return compute_pijcv_criterion(rho_vec, state; check_conditioning=false)
        end
        
        V_val = V_of_rho_vec(log_lambda)
        dV_drho = ForwardDiff.gradient(V_of_rho_vec, log_lambda)
        d2V_drho2 = ForwardDiff.hessian(V_of_rho_vec, log_lambda)
        
        grad_lambda, hess_lambda = transform_log_scale_derivatives(λ, dV_drho, d2V_drho2)
        
        # DEBUG: Trace Newton step components for vector case
        @debug "PIJCV derivatives (vector)" begin
            "λ=$(round.(λ, sigdigits=3)), V=$V_val, " *
            "dV/dρ=$(round.(dV_drho, sigdigits=3)), " *
            "d²V/dρ² diag=$(round.(diag(d2V_drho2), sigdigits=3)), " *
            "dV/dλ=$(round.(grad_lambda, sigdigits=3)), " *
            "d²V/dλ² diag=$(round.(diag(hess_lambda), sigdigits=3))"
        end
        
        return (V_val, grad_lambda, hess_lambda)
    end
end

"""
    _compute_efs_derivatives(state, λ) -> (V, ∇V, H_V)

Compute EFS/REML criterion and derivatives using ForwardDiff through log(λ).

The EFS criterion is the negative REML log-likelihood:
    V_EFS(λ) = -ℓ_LA(λ) = -ℓ(β̂) + ½β̂ᵀS_λβ̂ - ½log|S_λ|₊ + ½log|H_λ|
"""
function _compute_efs_derivatives(
    state::SmoothingSelectionState,
    λ::Vector{Float64}
)
    n_lambda = length(λ)
    log_lambda = log.(λ)
    
    if n_lambda == 1
        function V_of_rho(rho_scalar)
            rho_vec = [rho_scalar]
            return compute_efs_criterion(rho_vec, state)
        end
        
        V_val = V_of_rho(log_lambda[1])
        dV_drho = ForwardDiff.derivative(V_of_rho, log_lambda[1])
        d2V_drho2 = ForwardDiff.derivative(rho -> ForwardDiff.derivative(V_of_rho, rho), log_lambda[1])
        
        grad_lambda, hess_lambda = transform_log_scale_derivatives(λ, dV_drho, d2V_drho2)
        return (V_val, grad_lambda, hess_lambda)
    else
        function V_of_rho_vec(rho_vec)
            return compute_efs_criterion(rho_vec, state)
        end
        
        V_val = V_of_rho_vec(log_lambda)
        dV_drho = ForwardDiff.gradient(V_of_rho_vec, log_lambda)
        d2V_drho2 = ForwardDiff.hessian(V_of_rho_vec, log_lambda)
        
        grad_lambda, hess_lambda = transform_log_scale_derivatives(λ, dV_drho, d2V_drho2)
        return (V_val, grad_lambda, hess_lambda)
    end
end

"""
    _compute_perf_derivatives(state, λ) -> (V, ∇V, H_V)

Compute PERF criterion and derivatives using ForwardDiff through log(λ).

The PERF criterion (Marra & Radice 2020):
    V_PERF(λ) = ‖M - OM‖² - n + 2·tr(O)
"""
function _compute_perf_derivatives(
    state::SmoothingSelectionState,
    λ::Vector{Float64}
)
    n_lambda = length(λ)
    log_lambda = log.(λ)
    
    if n_lambda == 1
        function V_of_rho(rho_scalar)
            rho_vec = [rho_scalar]
            return compute_perf_criterion(rho_vec, state)
        end
        
        V_val = V_of_rho(log_lambda[1])
        dV_drho = ForwardDiff.derivative(V_of_rho, log_lambda[1])
        d2V_drho2 = ForwardDiff.derivative(rho -> ForwardDiff.derivative(V_of_rho, rho), log_lambda[1])
        
        grad_lambda, hess_lambda = transform_log_scale_derivatives(λ, dV_drho, d2V_drho2)
        return (V_val, grad_lambda, hess_lambda)
    else
        function V_of_rho_vec(rho_vec)
            return compute_perf_criterion(rho_vec, state)
        end
        
        V_val = V_of_rho_vec(log_lambda)
        dV_drho = ForwardDiff.gradient(V_of_rho_vec, log_lambda)
        d2V_drho2 = ForwardDiff.hessian(V_of_rho_vec, log_lambda)
        
        grad_lambda, hess_lambda = transform_log_scale_derivatives(λ, dV_drho, d2V_drho2)
        return (V_val, grad_lambda, hess_lambda)
    end
end

# Legacy wrapper for backward compatibility
function compute_pijcv_criterion_with_derivatives(
    β::Vector{Float64},
    λ::Vector{Float64},
    penalty_config::PenaltyConfig,
    model::MultistateProcess,
    data::ExactData;
    nfolds::Int = 0
)
    # Use the new generic interface
    selector = PIJCVSelector(nfolds)
    return compute_criterion_with_derivatives(β, λ, penalty_config, model, data, selector)
end

# =============================================================================
# Joint (λ, α) Performance Iteration Helpers
# =============================================================================

"""
    JointPIState

Minimal state for joint (λ, α) performance iteration.
Contains precomputed α caches and term mappings.
"""
struct JointPIState
    n_lambda::Int
    n_alpha::Int
    alpha_caches::Vector{AlphaJointOptCache}
    alpha_term_map::Dict{Int, Int}  # term_idx → alpha_idx
end

"""
    _build_joint_state_for_pi(model, penalty_config, alpha_info, alpha_groups) -> JointPIState

Build joint optimization state for performance iteration.
"""
function _build_joint_state_for_pi(
    model::MultistateProcess,
    penalty_config::QuadraticPenalty,
    alpha_info::Dict{Int, AlphaLearningInfo},
    alpha_groups::Vector{Vector{Int}}
)
    n_lambda = n_hyperparameters(penalty_config)
    n_alpha = length(alpha_groups)
    
    # Build α caches (one per α parameter, which may cover multiple terms)
    alpha_caches = AlphaJointOptCache[]
    alpha_term_map = Dict{Int, Int}()
    
    for (alpha_idx, group) in enumerate(alpha_groups)
        # Use first term in group to build cache (all have same α)
        term_idx = first(group)
        info = alpha_info[term_idx]
        
        # Build cache for this α
        cache = build_alpha_joint_opt_cache(
            model, penalty_config, term_idx, info.hazard, info.atrisk
        )
        push!(alpha_caches, cache)
        
        # Map all terms in group to this α
        for tidx in group
            alpha_term_map[tidx] = alpha_idx
        end
    end
    
    return JointPIState(n_lambda, n_alpha, alpha_caches, alpha_term_map)
end

"""
    _compute_penalized_gradient_joint(β, λ, α, penalty_config, joint_state) -> Vector{Float64}

Compute gradient of penalty with S(α) rebuilt at current α values.
"""
function _compute_penalized_gradient_joint(
    β::AbstractVector{T},
    λ::AbstractVector,
    α::AbstractVector,
    penalty_config::PenaltyConfig,
    joint_state::JointPIState
) where T
    n = length(β)
    grad = zeros(T, n)
    lambda_idx = 1
    
    # Baseline hazard penalty gradients with S(α) for learnable terms
    for (term_idx, term) in enumerate(penalty_config.terms)
        idx = term.hazard_indices
        β_j = @view β[idx]
        
        # Check if this term has learnable α
        if haskey(joint_state.alpha_term_map, term_idx)
            alpha_idx = joint_state.alpha_term_map[term_idx]
            cache = joint_state.alpha_caches[alpha_idx]
            S_alpha = build_penalty_matrix_at_alpha(cache, α[alpha_idx])
            grad[idx] .+= λ[lambda_idx] .* (S_alpha * β_j)
        else
            # Use stored S matrix
            grad[idx] .+= λ[lambda_idx] .* (term.S * β_j)
        end
        lambda_idx += 1
    end
    
    # Total hazard penalty gradients (unchanged - no α learning here)
    for term in penalty_config.total_hazard_terms
        K = size(term.S, 1)
        β_total = zeros(T, K)
        for idx_range in term.hazard_indices
            β_total .+= β[idx_range]
        end
        grad_total = λ[lambda_idx] .* (term.S * β_total)
        for idx_range in term.hazard_indices
            grad[idx_range] .+= grad_total
        end
        lambda_idx += 1
    end
    
    # Smooth covariate penalty gradients (unchanged)
    if !isempty(penalty_config.shared_smooth_groups)
        term_to_lambda = Dict{Int, Int}()
        for (group_idx, group) in enumerate(penalty_config.shared_smooth_groups)
            for term_idx in group
                term_to_lambda[term_idx] = lambda_idx
            end
            lambda_idx += 1
        end
        for term_idx in 1:length(penalty_config.smooth_covariate_terms)
            if !haskey(term_to_lambda, term_idx)
                term_to_lambda[term_idx] = lambda_idx
                lambda_idx += 1
            end
        end
        for (term_idx, term) in enumerate(penalty_config.smooth_covariate_terms)
            β_k = β[term.param_indices]
            grad[term.param_indices] .+= λ[term_to_lambda[term_idx]] .* (term.S * β_k)
        end
    else
        for term in penalty_config.smooth_covariate_terms
            β_k = β[term.param_indices]
            grad[term.param_indices] .+= λ[lambda_idx] .* (term.S * β_k)
            lambda_idx += 1
        end
    end
    
    return grad
end

"""
    _build_penalty_hessian_for_pi(H_nll, λ, α, penalty_config, joint_state) -> Matrix

Build penalty Hessian with S(α) for joint optimization.
Returns H_nll + H_penalty for efficiency.
"""
function _build_penalty_hessian_for_pi(
    H_nll::Matrix{Float64},
    λ::AbstractVector{T},
    α::AbstractVector,
    penalty_config::PenaltyConfig,
    joint_state::JointPIState
) where T<:Real
    n_params = size(H_nll, 1)
    H_pen = zeros(T, n_params, n_params)
    lambda_idx = 1
    
    # Baseline hazard penalty Hessians with S(α) for learnable terms
    for (term_idx, term) in enumerate(penalty_config.terms)
        idx = term.hazard_indices
        
        if haskey(joint_state.alpha_term_map, term_idx)
            alpha_idx = joint_state.alpha_term_map[term_idx]
            cache = joint_state.alpha_caches[alpha_idx]
            S_alpha = build_penalty_matrix_at_alpha(cache, α[alpha_idx])
            H_pen[idx, idx] .+= λ[lambda_idx] .* S_alpha
        else
            H_pen[idx, idx] .+= λ[lambda_idx] .* term.S
        end
        lambda_idx += 1
    end
    
    # Total hazard penalty Hessians (unchanged)
    for term in penalty_config.total_hazard_terms
        for idx_range1 in term.hazard_indices
            for idx_range2 in term.hazard_indices
                H_pen[idx_range1, idx_range2] .+= λ[lambda_idx] .* term.S
            end
        end
        lambda_idx += 1
    end
    
    # Smooth covariate penalty Hessians (unchanged)
    if !isempty(penalty_config.shared_smooth_groups)
        term_to_lambda = Dict{Int, Int}()
        for (group_idx, group) in enumerate(penalty_config.shared_smooth_groups)
            for term_idx in group
                term_to_lambda[term_idx] = lambda_idx
            end
            lambda_idx += 1
        end
        for term_idx in 1:length(penalty_config.smooth_covariate_terms)
            if !haskey(term_to_lambda, term_idx)
                term_to_lambda[term_idx] = lambda_idx
                lambda_idx += 1
            end
        end
        for (term_idx, term) in enumerate(penalty_config.smooth_covariate_terms)
            idx = term.param_indices
            H_pen[idx, idx] .+= λ[term_to_lambda[term_idx]] .* term.S
        end
    else
        for term in penalty_config.smooth_covariate_terms
            idx = term.param_indices
            H_pen[idx, idx] .+= λ[lambda_idx] .* term.S
            lambda_idx += 1
        end
    end
    
    return H_pen
end

"""
    _compute_penalty_at_theta(β, λ, α, penalty_config, joint_state) -> Float64

Compute penalty value with S(α) for line search.
"""
function _compute_penalty_at_theta(
    β::AbstractVector{T},
    λ::AbstractVector,
    α::AbstractVector,
    penalty_config::PenaltyConfig,
    joint_state::JointPIState
) where T
    pen = zero(T)
    lambda_idx = 1
    
    # Baseline hazard penalties with S(α)
    for (term_idx, term) in enumerate(penalty_config.terms)
        idx = term.hazard_indices
        β_j = @view β[idx]
        
        if haskey(joint_state.alpha_term_map, term_idx)
            alpha_idx = joint_state.alpha_term_map[term_idx]
            cache = joint_state.alpha_caches[alpha_idx]
            S_alpha = build_penalty_matrix_at_alpha(cache, α[alpha_idx])
            pen += 0.5 * λ[lambda_idx] * dot(β_j, S_alpha * β_j)
        else
            pen += 0.5 * λ[lambda_idx] * dot(β_j, term.S * β_j)
        end
        lambda_idx += 1
    end
    
    # Total hazard penalties (unchanged)
    for term in penalty_config.total_hazard_terms
        K = size(term.S, 1)
        β_total = zeros(T, K)
        for idx_range in term.hazard_indices
            β_total .+= β[idx_range]
        end
        pen += 0.5 * λ[lambda_idx] * dot(β_total, term.S * β_total)
        lambda_idx += 1
    end
    
    # Smooth covariate penalties (unchanged - using compute_penalty_from_lambda logic)
    if !isempty(penalty_config.shared_smooth_groups)
        term_to_lambda = Dict{Int, Int}()
        for (group_idx, group) in enumerate(penalty_config.shared_smooth_groups)
            for term_idx in group
                term_to_lambda[term_idx] = lambda_idx
            end
            lambda_idx += 1
        end
        for term_idx in 1:length(penalty_config.smooth_covariate_terms)
            if !haskey(term_to_lambda, term_idx)
                term_to_lambda[term_idx] = lambda_idx
                lambda_idx += 1
            end
        end
        for (term_idx, term) in enumerate(penalty_config.smooth_covariate_terms)
            β_k = β[term.param_indices]
            pen += 0.5 * λ[term_to_lambda[term_idx]] * dot(β_k, term.S * β_k)
        end
    else
        for term in penalty_config.smooth_covariate_terms
            β_k = β[term.param_indices]
            pen += 0.5 * λ[lambda_idx] * dot(β_k, term.S * β_k)
            lambda_idx += 1
        end
    end
    
    return pen
end

"""
    compute_criterion_with_derivatives_joint(β, λ, α, penalty_config, model, data, selector, joint_state)

Compute criterion and derivatives for joint (λ, α) optimization.

Returns (V, grad_θ, hess_θ) where θ = [λ; α] on natural scale.
Uses ForwardDiff through log(λ) and α to compute derivatives.
"""
function compute_criterion_with_derivatives_joint(
    β::Vector{Float64},
    λ::Vector{Float64},
    α::Vector{Float64},
    penalty_config::PenaltyConfig,
    model::MultistateProcess,
    data::D,
    selector::AbstractHyperparameterSelector,
    joint_state::JointPIState
) where D
    n_lambda = length(λ)
    n_alpha = length(α)
    n_theta = n_lambda + n_alpha
    
    # Build θ_internal = [log(λ); α] for optimization
    # λ is on log scale, α is on natural scale
    log_lambda = log.(λ)
    
    # Build state for criterion computation  
    subject_grads_ll, subject_hessians_ll = _compute_subject_grads_hessians(β, model, data)
    subject_grads = -subject_grads_ll
    subject_hessians = [-H for H in subject_hessians_ll]
    H_unpenalized = sum(subject_hessians)
    n_subjects = _get_n_subjects(data)
    n_params = length(β)
    
    # Criterion function that takes θ_internal = [log(λ); α]
    function V_of_theta(theta_internal)
        log_lam = theta_internal[1:n_lambda]
        alpha_vec = theta_internal[n_lambda+1:n_lambda+n_alpha]
        lam = exp.(log_lam)
        
        # Build penalized Hessian with S(α)
        H_lambda = _build_penalized_hessian_joint_for_criterion(
            H_unpenalized, lam, alpha_vec, penalty_config, joint_state
        )
        
        # Compute criterion based on selector type
        return _compute_criterion_value(
            β, lam, H_lambda, H_unpenalized, subject_grads, subject_hessians,
            penalty_config, n_subjects, selector
        )
    end
    
    # Compute V and derivatives via ForwardDiff
    theta_internal = vcat(log_lambda, α)
    
    V_val = V_of_theta(theta_internal)
    dV_dtheta = ForwardDiff.gradient(V_of_theta, theta_internal)
    d2V_dtheta2 = ForwardDiff.hessian(V_of_theta, theta_internal)
    
    # Transform λ derivatives from log scale to natural scale
    # α derivatives stay on natural scale
    grad_theta = similar(dV_dtheta)
    grad_theta[1:n_lambda] = dV_dtheta[1:n_lambda] ./ λ  # ∂V/∂λ = ∂V/∂(log λ) / λ
    grad_theta[n_lambda+1:end] = dV_dtheta[n_lambda+1:end]  # ∂V/∂α unchanged
    
    hess_theta = zeros(n_theta, n_theta)
    # λ-λ block
    for i in 1:n_lambda
        for j in 1:n_lambda
            if i == j
                hess_theta[i,j] = (d2V_dtheta2[i,j] - dV_dtheta[i]) / (λ[i]^2)
            else
                hess_theta[i,j] = d2V_dtheta2[i,j] / (λ[i] * λ[j])
            end
        end
    end
    # λ-α block
    for i in 1:n_lambda
        for j in 1:n_alpha
            hess_theta[i, n_lambda+j] = d2V_dtheta2[i, n_lambda+j] / λ[i]
            hess_theta[n_lambda+j, i] = hess_theta[i, n_lambda+j]
        end
    end
    # α-α block (unchanged)
    hess_theta[n_lambda+1:end, n_lambda+1:end] = d2V_dtheta2[n_lambda+1:end, n_lambda+1:end]
    
    return (V_val, grad_theta, hess_theta)
end

"""
    _build_penalized_hessian_joint_for_criterion(H_unpen, λ, α, penalty_config, joint_state)

Build H + λS(α) for criterion evaluation. AD-compatible.
"""
function _build_penalized_hessian_joint_for_criterion(
    H_unpen::Matrix{Float64},
    lambda::AbstractVector{T},
    alpha_vec::AbstractVector{T},
    penalty_config::PenaltyConfig,
    joint_state::JointPIState
) where T<:Real
    H = convert(Matrix{T}, copy(H_unpen))
    lambda_idx = 1
    
    for (term_idx, term) in enumerate(penalty_config.terms)
        idx = term.hazard_indices
        lam = lambda[lambda_idx]
        
        if haskey(joint_state.alpha_term_map, term_idx)
            alpha_idx = joint_state.alpha_term_map[term_idx]
            α = alpha_vec[alpha_idx]
            cache = joint_state.alpha_caches[alpha_idx]
            S_alpha = build_penalty_matrix_at_alpha(cache, α)
            
            for i in idx, j in idx
                H[i, j] += lam * S_alpha[i - first(idx) + 1, j - first(idx) + 1]
            end
        else
            for i in idx, j in idx
                H[i, j] += lam * term.S[i - first(idx) + 1, j - first(idx) + 1]
            end
        end
        lambda_idx += 1
    end
    
    # Total hazard terms (unchanged)
    for term in penalty_config.total_hazard_terms
        for idx_range1 in term.hazard_indices
            for idx_range2 in term.hazard_indices
                for i in idx_range1, j in idx_range2
                    H[i, j] += lambda[lambda_idx] * term.S[i - first(idx_range1) + 1, j - first(idx_range2) + 1]
                end
            end
        end
        lambda_idx += 1
    end
    
    # Smooth covariate terms (unchanged)
    if !isempty(penalty_config.shared_smooth_groups)
        term_to_lambda = Dict{Int, Int}()
        for (group_idx, group) in enumerate(penalty_config.shared_smooth_groups)
            for tidx in group
                term_to_lambda[tidx] = lambda_idx
            end
            lambda_idx += 1
        end
        for tidx in 1:length(penalty_config.smooth_covariate_terms)
            if !haskey(term_to_lambda, tidx)
                term_to_lambda[tidx] = lambda_idx
                lambda_idx += 1
            end
        end
        for (tidx, term) in enumerate(penalty_config.smooth_covariate_terms)
            lam = lambda[term_to_lambda[tidx]]
            indices = term.param_indices
            for (i, pi) in enumerate(indices)
                for (j, pj) in enumerate(indices)
                    H[pi, pj] += lam * term.S[i, j]
                end
            end
        end
    else
        for term in penalty_config.smooth_covariate_terms
            indices = term.param_indices
            for (i, pi) in enumerate(indices)
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
    _compute_criterion_value(β, λ, H_lambda, H_unpenalized, subject_grads, subject_hessians, 
                              penalty_config, n_subjects, selector)

Compute criterion value V(θ) for a given selector type. AD-compatible.
"""
function _compute_criterion_value(
    β::Vector{Float64},
    λ::AbstractVector{T},
    H_lambda::Matrix{T},
    H_unpenalized::Matrix{Float64},
    subject_grads::Matrix{Float64},
    subject_hessians::Vector{Matrix{Float64}},
    penalty_config::PenaltyConfig,
    n_subjects::Int,
    selector::AbstractHyperparameterSelector
) where T<:Real
    if selector isa PIJCVSelector
        # PIJCV criterion: LOO or k-fold
        return _compute_pijcv_criterion_at_theta(
            β, λ, H_lambda, subject_grads, subject_hessians, 
            penalty_config, n_subjects, selector.nfolds
        )
    elseif selector isa REMLSelector
        # EFS/REML criterion
        return _compute_efs_criterion_at_theta(β, λ, H_lambda, H_unpenalized, penalty_config)
    elseif selector isa PERFSelector
        # PERF criterion
        return _compute_perf_criterion_at_theta(β, λ, H_lambda, subject_grads, n_subjects)
    else
        error("Unknown selector type: $(typeof(selector))")
    end
end

"""
    _compute_pijcv_criterion_at_theta(β, λ, H_lambda, subject_grads, subject_hessians, 
                                       penalty_config, n_subjects, nfolds)

Compute PIJCV criterion at current θ. AD-compatible.
"""
function _compute_pijcv_criterion_at_theta(
    β::Vector{Float64},
    λ::AbstractVector{T},
    H_lambda::Matrix{T},
    subject_grads::Matrix{Float64},
    subject_hessians::Vector{Matrix{Float64}},
    penalty_config::PenaltyConfig,
    n_subjects::Int,
    nfolds::Int
) where T<:Real
    # Compute H_lambda inverse
    H_sym = Symmetric(H_lambda)
    
    # Use Cholesky for stability
    H_inv = try
        inv(cholesky(H_sym))
    catch
        # Fall back to pseudoinverse
        pinv(Matrix(H_sym))
    end
    
    V = zero(T)
    if nfolds == 0
        # LOO-PIJCV
        for i in 1:n_subjects
            g_i = subject_grads[:, i]
            H_i = convert(Matrix{T}, subject_hessians[i])
            
            # LOO influence: Δβ_{-i} ≈ H_λ^{-1} g_i / (1 - tr(H_λ^{-1} H_i))
            H_inv_Hi = H_inv * H_i
            denom = max(1 - tr(H_inv_Hi), 0.1)  # Clamp to avoid instability
            delta_beta_i = (H_inv * g_i) / denom
            
            # LOO residual contribution
            V += dot(g_i, delta_beta_i)
        end
    else
        # k-fold PIJCV (simplified - use random fold assignment)
        fold_size = div(n_subjects, nfolds)
        for k in 1:nfolds
            fold_start = (k-1) * fold_size + 1
            fold_end = min(k * fold_size, n_subjects)
            
            g_fold = zeros(T, size(subject_grads, 1))
            H_fold = zeros(T, size(H_lambda))
            for i in fold_start:fold_end
                g_fold .+= subject_grads[:, i]
                H_fold .+= convert(Matrix{T}, subject_hessians[i])
            end
            
            H_inv_Hk = H_inv * H_fold
            denom = max(1 - tr(H_inv_Hk) / nfolds, 0.1)
            delta_beta_k = (H_inv * g_fold) / denom
            
            V += dot(g_fold, delta_beta_k)
        end
    end
    
    return V
end

"""
    _compute_efs_criterion_at_theta(β, λ, H_lambda, H_unpenalized, penalty_config)

Compute EFS/REML criterion at current θ. AD-compatible.
"""
function _compute_efs_criterion_at_theta(
    β::Vector{Float64},
    λ::AbstractVector{T},
    H_lambda::Matrix{T},
    H_unpenalized::Matrix{Float64},
    penalty_config::PenaltyConfig
) where T<:Real
    # V_EFS = -ℓ_LA = -ℓ(β̂) + ½β̂ᵀS_λβ̂ - ½log|S_λ|₊ + ½log|H_λ|
    # Since ℓ(β̂) is fixed, we use:
    # V_EFS ∝ ½log|H_λ| - ½log|S_λ|₊ + ½β̂ᵀS_λβ̂
    
    # Compute log|H_λ| via Cholesky
    H_sym = Symmetric(H_lambda)
    log_det_H = try
        logdet(cholesky(H_sym))
    catch
        # Regularize if not positive definite
        eigvals_H = eigvals(H_sym)
        sum(log.(max.(eigvals_H, 1e-10)))
    end
    
    # Compute penalty quadratic form β'S_λβ
    penalty_quad = zero(T)
    lambda_idx = 1
    for term in penalty_config.terms
        idx = term.hazard_indices
        β_j = β[idx]
        # Note: We're using term.S here which may not account for current α
        # This is a simplification - full implementation would rebuild S(α)
        penalty_quad += λ[lambda_idx] * dot(β_j, term.S * β_j)
        lambda_idx += 1
    end
    
    # Compute log|S_λ|₊ (only counting non-zero eigenvalues)
    S_total = zeros(T, size(H_lambda))
    lambda_idx = 1
    for term in penalty_config.terms
        idx = term.hazard_indices
        S_total[idx, idx] .+= λ[lambda_idx] .* term.S
        lambda_idx += 1
    end
    
    eigvals_S = eigvals(Symmetric(S_total))
    log_det_S_plus = sum(log.(max.(eigvals_S, 1e-10)))
    
    return 0.5 * log_det_H - 0.5 * log_det_S_plus + 0.5 * penalty_quad
end

"""
    _compute_perf_criterion_at_theta(β, λ, H_lambda, subject_grads, n_subjects)

Compute PERF criterion at current θ. AD-compatible.
"""
function _compute_perf_criterion_at_theta(
    β::Vector{Float64},
    λ::AbstractVector{T},
    H_lambda::Matrix{T},
    subject_grads::Matrix{Float64},
    n_subjects::Int
) where T<:Real
    # PERF criterion: V = ‖M - OM‖² - n + 2·tr(O)
    # where M is the score matrix and O = H_λ^{-1} H_unpen
    
    H_sym = Symmetric(H_lambda)
    H_inv = try
        inv(cholesky(H_sym))
    catch
        pinv(Matrix(H_sym))
    end
    
    # Compute O = H_λ^{-1} Σ_i H_i ≈ I - influence adjustment
    # Simplified: use tr(H_λ^{-1} H_unpen) ≈ EDF
    n_params = size(H_lambda, 1)
    H_unpen_approx = sum(subject_grads * subject_grads') / n_subjects
    O = H_inv * H_unpen_approx
    
    # Residual norm: ‖M - OM‖²
    M = subject_grads  # n_params × n_subjects
    OM = O * M
    resid_norm = sum((M .- OM).^2)
    
    # PERF criterion
    return resid_norm - n_subjects + 2 * tr(O)
end

"""
    _joint_newton_step(λ, α, grad_V, hess_V, damp, n_lambda, n_alpha) -> (λ_new, α_new)

Compute Newton step for joint (λ, α) optimization with trust region.
"""
function _joint_newton_step(
    λ::Vector{Float64},
    α::Vector{Float64},
    grad_V::Vector{Float64},
    hess_V::Matrix{Float64},
    damp::Float64,
    n_lambda::Int,
    n_alpha::Int
)
    n_theta = n_lambda + n_alpha
    
    # Newton direction
    hess_V_sym = Symmetric(hess_V)
    Δθ_newton = if isposdef(hess_V_sym)
        -(hess_V_sym \ grad_V)
    else
        # Regularize Hessian
        eigvals_H = eigvals(hess_V)
        reg = max(1e-6, -minimum(eigvals_H)) + 1e-4
        -((hess_V + reg * I) \ grad_V)
    end
    
    # Apply dampening
    Δθ_damped = damp .* Δθ_newton
    
    # Split into λ and α components
    Δλ = Δθ_damped[1:n_lambda]
    Δα = Δθ_damped[n_lambda+1:end]
    
    # Apply trust region to λ (on log scale)
    Δλ_constrained = apply_lambda_trust_region(λ, Δλ)
    
    # Apply trust region to α (on natural scale, bounded [0, 2])
    # Limit step to 0.5 per iteration
    Δα_constrained = clamp.(Δα, -0.5, 0.5)
    
    # Update with bounds
    λ_new = clamp.(λ .+ Δλ_constrained, PI_LAMBDA_MIN, PI_LAMBDA_MAX)
    α_new = clamp.(α .+ Δα_constrained, 0.0, 2.0)
    
    return (λ_new, α_new)
end

# =============================================================================
# Core Performance Iteration Algorithm (Criterion-Agnostic)
# =============================================================================

"""
    _performance_iteration(model, data, penalty, selector; kwargs...) -> HyperparameterSelectionResult

Performance iteration algorithm for smoothing parameter selection.

This is a criterion-agnostic implementation of Wood (2024) "On Neighbourhood 
Cross Validation" performance iteration. It works with any selection criterion
(PIJCV, EFS/REML, PERF) by dispatching to the appropriate derivative function.

**Unified Design**: This function handles both λ-only and joint (λ,α) optimization.
When `alpha_info` and `alpha_groups` are provided, the hyperparameter vector becomes
θ = [log(λ₁), ..., log(λₖ), α₁, ..., αₘ] and the algorithm optimizes both simultaneously.

# Algorithm (pseudocode)
Iterate until convergence:
1. One Newton step for β (penalized likelihood): solve H_θ * Δβ = -gradient
2. One Newton step for θ (criterion V): solve Hθ_V * Δθ = -∇θ_V
3. Check convergence based on step sizes

# Supported Selectors
- `PIJCVSelector`: Newton-approximated cross-validation (Wood 2024 NCV)
- `REMLSelector`: REML/EFS criterion (Wood & Fasiolo 2017)
- `PERFSelector`: PERF criterion (Marra & Radice 2020)

# Arguments
- `model::MultistateProcess`: Model for likelihood evaluation
- `data::ExactData`: Data container
- `penalty::AbstractPenalty`: Penalty configuration
- `selector::AbstractHyperparameterSelector`: Selection criterion

# Keyword Arguments
- `beta_init::Vector{Float64}`: Initial coefficient estimate
- `lambda_init::Union{Nothing, Vector{Float64}}`: Initial λ (uses default if nothing)
- `alpha_info::Union{Nothing, Dict{Int, AlphaLearningInfo}}`: α learning metadata per term
- `alpha_groups::Union{Nothing, Vector{Vector{Int}}}`: Groups of terms sharing same α
- `maxiter::Int`: Maximum performance iterations
- `beta_tol::Float64`: Convergence tolerance for β
- `lambda_tol::Float64`: Convergence tolerance for λ (and α)
- `use_line_search::Bool`: Use Armijo line search for β step
- `verbose::Bool`: Print progress

# Returns
`HyperparameterSelectionResult` with optimal λ, warmstart β, criterion value, etc.
For joint optimization, the returned penalty has updated α values applied.

# Notes
This unified function handles all data types (ExactData, MPanelData, MCEMSelectionData)
via multiple dispatch on the `_nll_for_pi` and `compute_criterion_with_derivatives` helpers.
"""
function _performance_iteration(
    model::MultistateProcess,
    data::D,  # Generic data type - works with ExactData, MPanelData, MCEMSelectionData
    penalty::AbstractPenalty,
    selector::AbstractHyperparameterSelector;
    beta_init::Vector{Float64},
    lambda_init::Union{Nothing, Vector{Float64}} = nothing,
    # Joint (λ,α) support - when provided, optimizes θ = [log(λ); α]
    alpha_info::Union{Nothing, Dict{Int, AlphaLearningInfo}} = nothing,
    alpha_groups::Union{Nothing, Vector{Vector{Int}}} = nothing,
    maxiter::Int = PI_DEFAULT_MAXITER,
    beta_tol::Float64 = PI_BETA_TOL,
    lambda_tol::Float64 = PI_LAMBDA_TOL,
    use_line_search::Bool = true,
    verbose::Bool = false
) where D
    # Input validation
    @assert length(beta_init) == length(model.bounds.lb) "beta_init length mismatch"
    @assert all(isfinite, beta_init) "beta_init contains NaN or Inf"
    
    # Get penalty configuration
    penalty_config = penalty isa QuadraticPenalty ? penalty : build_penalty_config(model)
    n_lambda = n_hyperparameters(penalty)
    
    # Determine if we're doing joint (λ, α) optimization
    do_joint_alpha = !isnothing(alpha_info) && !isempty(alpha_info) &&
                     !isnothing(alpha_groups) && !isempty(alpha_groups)
    
    # Initialize λ
    λ = if !isnothing(lambda_init) && length(lambda_init) == n_lambda
        copy(lambda_init)
    else
        # Use default from penalty
        get_hyperparameters(penalty)
    end
    @assert length(λ) == n_lambda "lambda length mismatch: got $(length(λ)), expected $n_lambda"
    
    # Initialize α if doing joint optimization
    n_alpha = do_joint_alpha ? length(alpha_groups) : 0
    α = do_joint_alpha ? ones(n_alpha) : Float64[]  # Start from uniform weighting (α=1)
    
    # Build joint optimization state if needed
    joint_state = if do_joint_alpha
        _build_joint_state_for_pi(model, penalty_config, alpha_info, alpha_groups)
    else
        nothing
    end
    
    # Initialize β (clamp to bounds)
    β = project_to_bounds(copy(beta_init), model.bounds.lb, model.bounds.ub)
    lb, ub = model.bounds.lb, model.bounds.ub
    n_params = length(β)
    
    # Tracking
    converged = false
    n_iter = 0
    V_history = Float64[]
    λ_history = Vector{Float64}[]  # Track λ for oscillation detection
    α_history = do_joint_alpha ? Vector{Float64}[] : nothing  # Track α if doing joint
    n_oscillations = 0  # Count oscillation events
    
    verbose && @info "Performance iteration starting" n_lambda n_alpha n_params maxiter data_type=D do_joint_alpha
    
    for iter in 1:maxiter
        n_iter = iter
        
        # ====================================================================
        # STEP 1: One Newton step for β (penalized negative log-likelihood)
        # ====================================================================
        # Objective: min_β f(β) = -ℓ(β) + (1/2) Σⱼ λⱼ β'Sⱼ(α)β
        # Gradient: g = -∇ℓ(β) + Σⱼ λⱼ Sⱼ(α) β
        # Hessian: H = -∇²ℓ(β) + Σⱼ λⱼ Sⱼ(α)
        
        # Compute log-likelihood gradient and Hessian using data-type dispatch
        # _nll_for_pi handles ExactData, MPanelData, and MCEMSelectionData
        nll_func = b -> _nll_for_pi(b, data)
        g_nll = ForwardDiff.gradient(nll_func, β)
        H_nll = ForwardDiff.hessian(nll_func, β)
        
        # Add penalty contributions - dispatch based on whether we have α
        if do_joint_alpha
            # Joint case: rebuild S(α) at current α values
            g_penalty = _compute_penalized_gradient_joint(β, λ, α, penalty_config, joint_state)
            H_penalty = _build_penalty_hessian_for_pi(H_nll, λ, α, penalty_config, joint_state)
        else
            # λ-only case: use cached S matrices
            g_penalty = _compute_penalized_gradient(β, λ, penalty_config)
            H_penalty = _build_penalty_hessian(λ, penalty_config, n_params)
        end
        
        # Combined gradient and Hessian for penalized NLL
        g = g_nll + g_penalty
        H = H_nll + H_penalty
        
        # Newton direction
        H_sym = Symmetric(H)
        Δβ = try
            -(H_sym \ g)
        catch e
            @debug "Hessian solve failed in performance iteration" iter exception=e
            # Fall back to gradient descent with small step
            -0.01 * g
        end
        
        # Line search for β step (optional)
        if use_line_search
            f_beta = if do_joint_alpha
                b -> _nll_for_pi(b, data) + _compute_penalty_at_theta(b, λ, α, penalty_config, joint_state)
            else
                b -> _nll_for_pi(b, data) + compute_penalty_from_lambda(b, λ, penalty_config)
            end
            α_β, _ = armijo_line_search(f_beta, β, Δβ, g)
            β_new = project_to_bounds(β + α_β * Δβ, lb, ub)
        else
            β_new = project_to_bounds(β + Δβ, lb, ub)
        end
        
        # ====================================================================
        # STEP 2: One Newton step for θ (criterion V) with safeguards
        # θ = [λ] for λ-only, θ = [λ; α] for joint optimization
        # ====================================================================
        V, grad_V, hess_V = if do_joint_alpha
            compute_criterion_with_derivatives_joint(
                β_new, λ, α, penalty_config, model, data, selector, joint_state
            )
        else
            compute_criterion_with_derivatives(
                β_new, λ, penalty_config, model, data, selector
            )
        end
        push!(V_history, V)
        
        # Check for oscillation in V to adaptively reduce step sizes
        if detect_oscillation(V_history)
            n_oscillations += 1
            @debug "Oscillation detected in V" iter n_oscillations
        end
        
        # Compute adaptive dampening factor
        damp = adaptive_dampening(iter, n_oscillations)
        
        # Newton step for θ with trust region and dampening
        # For joint optimization, θ = [λ; α] with mixed bounds
        if do_joint_alpha
            # Joint (λ, α) Newton step
            λ_new, α_new = _joint_newton_step(
                λ, α, grad_V, hess_V, damp, n_lambda, n_alpha
            )
            push!(α_history, copy(α_new))
        elseif n_lambda == 1
            # Scalar λ case
            # DEBUG: Print what we received
            @debug "Newton step inputs" iter λ=λ[1] V grad_V hess_V
            
            if abs(hess_V) > 1e-10 && hess_V > 0  # Ensure positive definite
                Δλ_newton = -grad_V / hess_V
                @debug "Newton step (Hessian OK)" Δλ_newton direction=(Δλ_newton > 0 ? "UP" : "DOWN")
            else
                # Gradient descent fallback with adaptive step
                grad_step = -sign(grad_V) * min(abs(grad_V) * 0.1, λ[1] * 0.5)
                Δλ_newton = grad_step
                @debug "Newton fallback (Hessian bad)" hess_V Δλ_newton direction=(Δλ_newton > 0 ? "UP" : "DOWN")
            end
            
            # Apply dampening
            Δλ_damped = [damp * Δλ_newton]
            
            # Apply trust region (limit log-scale change)
            Δλ_constrained = apply_lambda_trust_region(λ, Δλ_damped)
            
            # DEBUG: Print the final step
            @debug "Newton step result" damp Δλ_damped[1] Δλ_constrained[1]
            
            # Try full Newton step first, use Armijo line search if needed
            λ_trial = clamp.(λ .+ Δλ_constrained, PI_LAMBDA_MIN, PI_LAMBDA_MAX)
            
            # Evaluate criterion at trial point (using current β_new, which is fixed for this λ step)
            # For efficiency, skip line search if step is small enough
            if abs(Δλ_constrained[1]) / (1 + abs(λ[1])) < 0.1
                λ_new = λ_trial
            else
                # Line search for λ step
                V_func = λ_test -> begin
                    # Re-evaluate criterion at test λ with current β_new
                    V_test, _, _ = compute_criterion_with_derivatives(
                        β_new, λ_test, penalty_config, model, data, selector
                    )
                    V_test
                end
                _, _, λ_new = armijo_line_search_lambda(V_func, λ, Δλ_constrained, V, grad_V;
                                                         α_init=1.0)
            end
            α_new = α  # No α in this case
        else
            # Vector λ case (no joint α)
            hess_V_sym = Symmetric(hess_V)
            
            # DEBUG: Print the Hessian and gradient for vector case
            @debug "Vector Newton step inputs" begin
                eigs = eigvals(hess_V_sym)
                "iter: λ=$(round.(λ, sigdigits=3)), grad_V=$(round.(grad_V, sigdigits=3)), " *
                "hess_diag=$(round.(diag(hess_V), sigdigits=3)), " *
                "eigenvalues=$(round.(eigs, sigdigits=3)), " *
                "isposdef=$(all(eigs .> 0))"
            end
            
            if isposdef(hess_V_sym)
                Δλ_newton = -(hess_V_sym \ grad_V)
                @debug "Vector Newton step (posdef)" Δλ_newton=round.(Δλ_newton, sigdigits=3)
            else
                # Regularize Hessian or use gradient descent
                reg = max(1e-6, -minimum(eigvals(hess_V))) + 1e-4
                Δλ_newton = -((hess_V + reg * I) \ grad_V)
                @debug "Vector Newton step (regularized)" reg Δλ_newton=round.(Δλ_newton, sigdigits=3)
            end
            end
            
            # Apply dampening
            Δλ_damped = damp .* Δλ_newton
            
            # Apply trust region
            Δλ_constrained = apply_lambda_trust_region(λ, Δλ_damped)
            
            # For vector case, use simpler update with trust region
            λ_new = clamp.(λ .+ Δλ_constrained, PI_LAMBDA_MIN, PI_LAMBDA_MAX)
            α_new = α  # No α in this case
        end
        
        # Track λ history
        push!(λ_history, copy(λ_new))
        
        # ====================================================================
        # Check convergence (using log-scale change for λ, and α if joint)
        # ====================================================================
        β_change = norm(β_new - β) / (1 + norm(β))
        λ_log_change = norm(log.(λ_new) - log.(λ)) / (1 + norm(log.(λ)))
        α_change = do_joint_alpha ? norm(α_new - α) / (1 + norm(α)) : 0.0
        θ_change = max(λ_log_change, α_change)  # Combined hyperparameter convergence
        
        if verbose && (iter % 5 == 0 || iter == 1)
            if do_joint_alpha
                @info "PI iter $iter" V=round(V, sigdigits=5) β_change=round(β_change, sigdigits=3) λ_log_change=round(λ_log_change, sigdigits=3) α_change=round(α_change, sigdigits=3) λ=round.(λ_new, sigdigits=4) α=round.(α_new, digits=3) damp=round(damp, sigdigits=2)
            else
                @info "PI iter $iter" V=round(V, sigdigits=5) β_change=round(β_change, sigdigits=3) λ_log_change=round(λ_log_change, sigdigits=3) λ=round.(λ_new, sigdigits=4) damp=round(damp, sigdigits=2)
            end
        end
        
        # Primary convergence: both β and θ (λ and α) meet tolerance
        if β_change < beta_tol && θ_change < lambda_tol
            converged = true
            verbose && @info "Performance iteration converged" iter V β_change θ_change
            β, λ, α = β_new, λ_new, α_new
            break
        end
        
        # Auxiliary convergence: criterion V is stable (useful for vector λ)
        # When V changes by < PI_V_REL_TOL for PI_V_STABLE_WINDOW consecutive iterations,
        # and β is reasonably converged, we declare convergence even if λ is still jittering
        # Use a looser β tolerance (1e-3) for V-stability since the criterion is what matters
        if β_change < 1e-3 && detect_criterion_stability(V_history)
            converged = true
            verbose && @info "Performance iteration converged (V-stable)" iter V β_change θ_change
            β, λ, α = β_new, λ_new, α_new
            break
        end
        
        β, λ, α = β_new, λ_new, α_new
    end
    
    if !converged && verbose
        @warn "Performance iteration did not converge" maxiter n_iter β_change=norm(β - beta_init)/(1+norm(beta_init))
    end
    
    # ========================================================================
    # Compute final EDF and return result
    # ========================================================================
    # Update penalty with optimal α values if doing joint optimization
    final_penalty = if do_joint_alpha
        updated_penalty = update_penalty_with_joint_alphas(
            penalty_config, joint_state.alpha_caches, joint_state.alpha_term_map, α
        )
        set_hyperparameters(updated_penalty, λ)
    else
        set_hyperparameters(penalty, λ)
    end
    
    edf = compute_edf(β, λ, final_penalty, model, data)
    final_V = isempty(V_history) ? NaN : V_history[end]
    
    # Determine method symbol based on selector type and whether we did joint optimization
    method = if selector isa PIJCVSelector
        do_joint_alpha ? :pijcv_joint_pi : :pijcv_pi
    elseif selector isa REMLSelector
        do_joint_alpha ? :efs_joint_pi : :efs_pi
    elseif selector isa PERFSelector
        do_joint_alpha ? :perf_joint_pi : :perf_pi
    else
        :unknown_pi
    end
    
    # Build diagnostics tuple
    diagnostics = if do_joint_alpha
        (V_history = V_history, λ_history = λ_history, α_history = α_history, 
         n_oscillations = n_oscillations, final_alpha = α, final_criterion = final_V)
    else
        (V_history = V_history, λ_history = λ_history, n_oscillations = n_oscillations,
         final_criterion = final_V)
    end
    
    return HyperparameterSelectionResult(
        λ,                          # lambda
        β,                          # warmstart_beta
        final_penalty,              # penalty
        final_V,                    # criterion_value
        edf,                        # edf
        converged,                  # converged
        method,                     # method (performance iteration)
        n_iter,                     # n_iterations
        diagnostics                 # diagnostics
    )
end

# Legacy wrapper for backward compatibility with PIJCV-specific calls
function _performance_iteration_pijcv(
    model::MultistateProcess,
    data::ExactData,
    penalty::AbstractPenalty,
    selector::PIJCVSelector;
    kwargs...
)
    return _performance_iteration(model, data, penalty, selector; kwargs...)
end
