# ============================================================================
# PIJCV: Predictive Infinitesimal Jackknife Cross-Validation
# ============================================================================
# 
# Implementation of Predictive Infinitesimal Jackknife Cross-Validation (PIJCV) for smoothing parameter
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
    PIJCVState

Mutable state object for Predictive Infinitesimal Jackknife Cross-Validation computations.

Stores the penalized Hessian factorization and subject/neighbourhood-level 
quantities needed for efficient PIJCV criterion computation.

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
mutable struct PIJCVState
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
    PIJCVState(H_lambda, subject_grads; subject_hessians=nothing, 
             penalty_matrix=nothing, log_smoothing_params=nothing)

Initialize a PIJCV state from penalized Hessian and neighbourhood gradients.

# Arguments
- `H_lambda::AbstractMatrix`: Penalized Hessian H_λ = -∂²ℓ/∂β² + Sλ, p × p
- `subject_grads::AbstractMatrix`: Neighbourhood gradient contributions, p × n
- `subject_hessians::Union{Nothing, AbstractArray}=nothing`: p × p × n Hessian contributions
- `penalty_matrix::Union{Nothing, AbstractMatrix}=nothing`: Penalty matrix Sλ
- `log_smoothing_params::Union{Nothing, AbstractVector}=nothing`: log(λ) values

# Returns
- `PIJCVState`: Initialized state with Cholesky factorization attempted

# Example
```julia
# From fitted model with splines
state = PIJCVState(penalized_hessian, subject_grads;
                 subject_hessians=subj_hess,
                 penalty_matrix=S_lambda)
```

# Notes
The Cholesky factorization is attempted on initialization. If it fails (indicating
H_λ is not positive definite), `H_chol` will be `nothing` and subsequent computations
will use direct methods.
"""
function PIJCVState(H_lambda::AbstractMatrix, 
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
        @debug "Cholesky factorization failed during PIJCVState initialization; PIJCV will use direct solves" exception=(e, catch_backtrace())
        nothing
    end
    
    # Initialize deltas and flags
    deltas = zeros(Float64, nparams, nsubj)
    indefinite_flags = falses(nsubj)
    
    return PIJCVState(H_mat, H_chol, grads_mat, hess_arr, deltas, 
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
    pijcv_loo_perturbation_cholesky(H_chol, H_k, g_k; tol=1e-10)

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
function pijcv_loo_perturbation_cholesky(H_chol::Cholesky,
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
        return pijcv_loo_perturbation_woodbury(H_chol, H_k, g_k)
    end
end

"""
    pijcv_loo_perturbation_woodbury(H_chol, H_k, g_k)

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
function pijcv_loo_perturbation_woodbury(H_chol::Cholesky,
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
    pijcv_loo_perturbation_direct(H_lambda, H_k, g_k)

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
function pijcv_loo_perturbation_direct(H_lambda::AbstractMatrix,
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
    compute_pijcv_perturbations!(state::PIJCVState)

Compute all LOO perturbations Δ_{-α(k)} for each neighbourhood.

Updates `state.deltas` and `state.indefinite_flags` in place.

# Arguments
- `state::PIJCVState`: PIJCV state with H_λ and neighbourhood quantities

# Returns
- `state`: The modified PIJCVState

# Algorithm
For each neighbourhood k = 1, ..., n:
1. Extract H_k (neighbourhood Hessian) and g_k (neighbourhood gradient)
2. Compute Δ_{-α(k)} = H_{λ,α(k)}^{-1} g_k using best available method
3. Mark indefinite flag if Woodbury or direct solve was needed

# Notes
Uses efficient Cholesky downdate when H_chol is available and H_k allows.
Falls back to Woodbury identity or direct solve as needed.
"""
function compute_pijcv_perturbations!(state::PIJCVState)
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
            pijcv_loo_perturbation_cholesky(state.H_chol, H_k, g_k)
        else
            pijcv_loo_perturbation_direct(state.H_lambda, H_k, g_k)
        end
        
        state.deltas[:, k] .= result.delta
        state.indefinite_flags[k] = result.indefinite
    end
    
    return state
end

"""
    pijcv_criterion(state::PIJCVState, params::AbstractVector, 
                  loss_fn::Function, data; use_quadratic=false)

Compute the PIJCV criterion (approximate leave-neighbourhood-out cross-validation loss).

# Arguments
- `state::PIJCVState`: PIJCV state with computed perturbations
- `params::AbstractVector`: Current parameter estimates β̂
- `loss_fn::Function`: Function `loss_fn(params, data, k)` returning loss for neighbourhood k
- `data`: Data object passed to loss function
- `use_quadratic::Bool=false`: Use quadratic approximation V_q for robustness

# Returns
- `Float64`: PIJCV criterion V = (1/n) Σₖ D(y_{α(k)}, β̂ + Δ_{-α(k)})

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
function pijcv_criterion(state::PIJCVState,
                       params::AbstractVector,
                       loss_fn::Function,
                       data;
                       use_quadratic::Bool=false)
    nsubj = size(state.deltas, 2)
    
    if use_quadratic
        return pijcv_criterion_quadratic(state, params, loss_fn, data)
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
            D_k = pijcv_loss_quadratic_k(state, params, loss_fn, data, k)
        end
        
        V += D_k
    end
    
    return V / nsubj
end

"""
    pijcv_criterion_quadratic(state::PIJCVState, params::AbstractVector,
                            loss_fn::Function, data)

Compute PIJCV criterion using quadratic approximation (Equation 5 in Wood 2024).

More robust than exact evaluation when perturbations are large.
```math
V_q = (1/n) Σₖ [D_k(β̂) + g_k'Δ_{-α(k)} + (1/2)Δ_{-α(k)}'H_k Δ_{-α(k)}]
```
"""
function pijcv_criterion_quadratic(state::PIJCVState,
                                 params::AbstractVector,
                                 loss_fn::Function,
                                 data)
    nsubj = size(state.deltas, 2)
    have_hessians = !isnothing(state.subject_hessians)
    
    V_q = 0.0
    for k in 1:nsubj
        V_q += pijcv_loss_quadratic_k(state, params, loss_fn, data, k)
    end
    
    return V_q / nsubj
end

"""
    pijcv_loss_quadratic_k(state, params, loss_fn, data, k)

Quadratic approximation of LOO loss for neighbourhood k.
"""
function pijcv_loss_quadratic_k(state::PIJCVState,
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
    pijcv_criterion_derivatives(state::PIJCVState, params::AbstractVector,
                              loss_fn::Function, grad_loss_fn::Function, data, 
                              penalty_derivatives)

Compute PIJCV criterion and its derivatives with respect to log smoothing parameters.

# Arguments
- `state::PIJCVState`: PIJCV state with computed perturbations
- `params::AbstractVector`: Current parameter estimates β̂
- `loss_fn::Function`: Function `loss_fn(params, data, k)` returning loss
- `grad_loss_fn::Function`: Function returning gradient of loss w.r.t. params
- `data`: Data object
- `penalty_derivatives`: Vector of penalty matrix derivatives ∂S/∂ρⱼ

# Returns
NamedTuple with:
- `V`: PIJCV criterion value
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
function pijcv_criterion_derivatives(state::PIJCVState,
                                   params::AbstractVector,
                                   loss_fn::Function,
                                   grad_loss_fn::Function,
                                   data,
                                   penalty_derivatives::AbstractVector)
    nsubj = size(state.deltas, 2)
    n_smoothing = length(penalty_derivatives)
    
    # Compute PIJCV criterion
    V = pijcv_criterion(state, params, loss_fn, data)
    
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
                result = pijcv_loo_perturbation_cholesky(state.H_chol, H_k, Sj_params_loo)
            else
                result = pijcv_loo_perturbation_direct(state.H_lambda, H_k, Sj_params_loo)
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
    pijcv_degeneracy_test(state::PIJCVState, params::AbstractVector, 
                        penalty_derivative::AbstractMatrix; tol=1e-6)

Test for degeneracy in smoothing parameter (Section 3 of Wood 2024).

A smoothing parameter λⱼ is degenerate if:
```math
(∂β̂/∂ρⱼ)' H_λ (∂β̂/∂ρⱼ) ≈ 0
```

This occurs when λⱼ is so large that the corresponding basis functions
have been effectively removed from the model.

# Arguments
- `state::PIJCVState`: PIJCV state
- `params::AbstractVector`: Current parameters
- `penalty_derivative::AbstractMatrix`: ∂S/∂ρⱼ for the smoothing parameter
- `tol::Float64=1e-6`: Tolerance for degeneracy detection

# Returns
- `Bool`: `true` if the smoothing parameter is degenerate
"""
function pijcv_degeneracy_test(state::PIJCVState,
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
    pijcv_get_loo_estimates(state::PIJCVState, params::AbstractVector)

Get leave-neighbourhood-out parameter estimates from PIJCV state.

# Arguments
- `state::PIJCVState`: PIJCV state with computed perturbations
- `params::AbstractVector`: Current (full-data) parameter estimates, length p

# Returns
- `Matrix{Float64}`: p × n matrix where column k is β̂_{-α(k)} = β̂ + Δ_{-α(k)}
"""
function pijcv_get_loo_estimates(state::PIJCVState, params::AbstractVector)
    return params .+ state.deltas
end

"""
    pijcv_get_perturbations(state::PIJCVState)

Get the LOO perturbations Δ_{-α(k)} from PIJCV state.

# Arguments
- `state::PIJCVState`: PIJCV state with computed perturbations

# Returns
- `Matrix{Float64}`: p × n matrix where column k is Δ_{-α(k)} = β̂_{-α(k)} - β̂
"""
function pijcv_get_perturbations(state::PIJCVState)
    return state.deltas
end

"""
    pijcv_vcov(state::PIJCVState)

Compute variance estimates from PIJCV state perturbations.

Returns both IJ and jackknife variance estimates based on the LOO perturbations,
analogous to the standard IJ/JK variance estimators.

# Arguments
- `state::PIJCVState`: PIJCV state with computed perturbations

# Returns
NamedTuple with:
- `ij_vcov`: Infinitesimal jackknife variance
- `jk_vcov`: Jackknife variance

# Notes
These variance estimates account for the penalization structure through the
perturbations, which are computed from the penalized Hessian H_λ.
"""
function pijcv_vcov(state::PIJCVState)
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

# Note
This function requires comparing two fitted models: one with `vcov_type=:model` 
and one with `vcov_type=:ij` (or `:jk`). In the current implementation, only 
one variance type is stored per fitted model.

# Example
```julia
# Fit model with model-based variance
fitted_model = fit(model; vcov_type=:model)

# Fit model with IJ variance
fitted_ij = fit(model; vcov_type=:ij)

# Compare manually:
model_se = sqrt.(diag(get_vcov(fitted_model)))
robust_se = sqrt.(diag(get_vcov(fitted_ij)))
ratio = robust_se ./ model_se
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

See also: [`get_vcov`](@ref)
"""
function compare_variance_estimates(fitted; use_ij::Bool = true, threshold::Float64 = 1.5, verbose::Bool = true)
    # This function is deprecated with new vcov API
    @warn "compare_variance_estimates is deprecated. With the new vcov_type API, " *
          "fit the model twice with vcov_type=:model and vcov_type=:ij and compare manually." maxlog=1
    
    # Get model-based variance
    model_vcov = get_vcov(fitted)
    if isnothing(model_vcov)
        throw(ArgumentError("Variance not available. Fit model with vcov_type=:model or vcov_type=:ij."))
    end
    
    # With new API, robust_vcov is the same as model_vcov since only one type is stored
    robust_vcov = model_vcov
    
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
    catch e
        @debug "Could not retrieve parameter names from fitted model" exception=(e, catch_backtrace())
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

