# =============================================================================
# Constrained Variance Estimation (Item #27)
# =============================================================================
#
# For constrained MLE, the standard variance formula Var(θ̂) = H⁻¹ does not apply
# directly. These functions implement the reduced Hessian approach for computing
# variance under active constraints.
#
# Background:
# - For equality constraints c(θ) = 0, variance is computed in the null space
#   of the constraint Jacobian
# - For inequality constraints c(θ) ≤ 0, only active constraints (c(θ) ≈ 0)
#   affect the variance
# - The reduced Hessian approach: Var(θ̂) = Z (Z'HZ)⁻¹ Z'
#   where Z spans the null space of the active constraint Jacobian
#
# =============================================================================

"""
    compute_null_space_basis(J; tol=1e-10)

Compute an orthonormal basis for the null space of matrix `J` using SVD.

The null space of J consists of all vectors x such that Jx = 0. This is spanned
by the right singular vectors corresponding to singular values below `tol`.

# Arguments
- `J::AbstractMatrix`: Matrix whose null space to compute (p_c × p)
- `tol::Float64=1e-10`: Singular values below this are treated as zero

# Returns
- `Matrix{Float64}`: p × r matrix where columns form orthonormal basis for null(J),
  and r = p - rank(J) is the dimension of the null space

# Notes
- Returns an identity matrix if J is empty or all zeros (no constraints)
- Returns empty matrix (p × 0) if J has full column rank (null space is trivial)

# Example
```julia
J = [1.0 0.0 1.0; 0.0 1.0 -1.0]  # 2 constraints on 3 parameters
Z = compute_null_space_basis(J)   # 3 × 1 matrix spanning null(J)
```
"""
function compute_null_space_basis(J::AbstractMatrix; tol::Float64=1e-10)
    p = size(J, 2)  # Number of parameters
    
    # Handle edge cases
    if isempty(J) || size(J, 1) == 0
        # No constraints → all directions are free
        return Matrix{Float64}(I, p, p)
    end
    
    if all(iszero, J)
        # All-zero constraint Jacobian → all directions are free
        return Matrix{Float64}(I, p, p)
    end
    
    # For an m × p matrix J (m constraints, p parameters), the null space has dimension
    # p - rank(J). When m < p, the non-full SVD only returns min(m, p) = m singular values,
    # so we need full=true to get the full V matrix.
    F = svd(J, full=true)
    
    p_c = size(J, 1)  # Number of constraints
    n_sv = length(F.S)  # Number of computed singular values = min(p_c, p)
    
    # Null space dimension = p - rank(J)
    # rank(J) = number of singular values > tol
    rank_J = sum(F.S .> tol)
    null_dim = p - rank_J
    
    if null_dim == 0
        # Full column rank → null space is trivial (zero-dimensional)
        return zeros(Float64, p, 0)
    end
    
    # The null space is spanned by:
    # 1. Columns of V corresponding to singular values < tol (if any)
    # 2. Additional columns of V beyond the first n_sv columns (if p > p_c)
    #
    # With full=true, F.V is p × p. The null space columns are the last null_dim columns.
    return Matrix(F.V[:, (p - null_dim + 1):p])
end

"""
    compute_constrained_vcov(H, J_active; tol=1e-10)

Compute variance-covariance matrix under active constraints using the reduced Hessian.

For constrained MLE with active constraints c(θ̂) = 0, the variance is:
```math
\\text{Var}(\\hat{\\theta}) = Z (Z^\\top H Z)^{-1} Z^\\top
```
where:
- H is the Hessian of the (negative) log-likelihood at the MLE
- Z is an orthonormal basis for the null space of the active constraint Jacobian J

This projects the variance onto the subspace of feasible directions (those that
don't violate the active constraints to first order).

# Arguments
- `H::AbstractMatrix`: Hessian of negative log-likelihood (observed information), p × p
- `J_active::AbstractMatrix`: Jacobian of active constraints evaluated at MLE, p_a × p
- `tol::Float64=1e-10`: Tolerance for null space computation

# Returns
- `Matrix{Float64}`: p × p variance-covariance matrix. Directions constrained to zero
  will have zero variance.

# Notes
- If no constraints are active (J_active is empty), returns standard H⁻¹
- If all directions are constrained, returns zero matrix (degenerate)
- Uses pseudo-inverse for numerical stability

# Example
```julia
# After fitting with constraints
H = -ForwardDiff.hessian(loglik, fitted_params)
J = compute_constraint_jacobian(fitted_params, constraints)
active = identify_active_constraints(fitted_params, constraints)
J_active = J[active, :]
vcov_constrained = compute_constrained_vcov(H, J_active)
```

# References
- Nocedal & Wright (2006), Numerical Optimization, Section 12.4
- Magnus & Neudecker (1999), Matrix Differential Calculus, Chapter 3
"""
function compute_constrained_vcov(H::AbstractMatrix, J_active::AbstractMatrix; tol::Float64=1e-10)
    p = size(H, 1)
    
    # Handle case of no active constraints
    if isempty(J_active) || size(J_active, 1) == 0 || all(iszero, J_active)
        # No active constraints → standard inverse
        return Matrix(pinv(Symmetric(-H)))
    end
    
    # Compute null-space basis of active constraint Jacobian
    Z = compute_null_space_basis(J_active; tol=tol)
    
    if size(Z, 2) == 0
        # All directions constrained → zero variance (degenerate case)
        @warn "All parameter directions are constrained. Variance is degenerate (zero)."
        return zeros(Float64, p, p)
    end
    
    # Reduced Hessian: project H onto feasible subspace
    # H is the Hessian of NEGATIVE log-likelihood, so -H is Fisher information
    H_reduced = Z' * (-H) * Z
    
    # Inverse of reduced Hessian (Fisher information in feasible subspace)
    try
        H_reduced_sym = Symmetric(H_reduced)
        H_reduced_inv = pinv(H_reduced_sym)
    catch e
        @warn "Reduced Hessian inversion failed: $e. Returning zero variance."
        return zeros(Float64, p, p)
    end
    
    # Project back to full parameter space
    # Var(θ̂) = Z * Var(Z'θ̂) * Z' = Z * (Z'HZ)⁻¹ * Z'
    H_reduced_inv = pinv(Symmetric(H_reduced))
    vcov = Z * H_reduced_inv * Z'
    
    # Ensure symmetry and clean up small values
    vcov = Matrix(Symmetric(vcov))
    vcov[abs.(vcov) .< tol] .= 0.0
    
    return vcov
end

"""
    compute_constrained_vcov_from_components(subject_hessians, J_active; tol=1e-10, vcov_threshold=true)

Compute constrained variance from subject-level Hessians and active constraint Jacobian.

This is a convenience function that first computes the total Hessian from subject
contributions, then applies the reduced Hessian approach.

# Arguments
- `subject_hessians::Vector{Matrix{Float64}}`: Subject-level Hessian contributions
- `J_active::AbstractMatrix`: Jacobian of active constraints
- `tol::Float64=1e-10`: Tolerance for null space computation
- `vcov_threshold::Bool=true`: Use adaptive threshold for pseudo-inverse

# Returns
- `Matrix{Float64}`: Constrained variance-covariance matrix
"""
function compute_constrained_vcov_from_components(subject_hessians::Vector{<:AbstractMatrix}, 
                                                  J_active::AbstractMatrix;
                                                  tol::Float64=1e-10,
                                                  vcov_threshold::Bool=true)
    # Sum subject Hessians to get total (negative) Hessian
    nparams = size(subject_hessians[1], 1)
    nsubj = length(subject_hessians)
    
    fishinf = zeros(Float64, nparams, nparams)
    for H_i in subject_hessians
        fishinf .-= H_i  # fishinf = -Σ Hᵢ = observed Fisher information
    end
    
    # Apply pseudo-inverse threshold
    if vcov_threshold
        atol_pinv = (log(nsubj) * nparams)^-2
    else
        atol_pinv = sqrt(eps(Float64))
    end
    
    # Use reduced Hessian approach if there are active constraints
    if isempty(J_active) || size(J_active, 1) == 0
        vcov = pinv(Symmetric(fishinf), atol=atol_pinv)
        vcov[isapprox.(vcov, 0.0; atol=sqrt(eps(Float64)))] .= 0.0
        return Matrix(Symmetric(vcov))
    else
        return compute_constrained_vcov(-fishinf, J_active; tol=tol)
    end
end