# =============================================================================
# Spline Utilities
# =============================================================================
#
# Shared utilities for B-spline basis construction and penalty matrices.
# Used by both baseline splines (SplineHazard) and covariate splines (s()).
#
# =============================================================================

using BSplineKit
using LinearAlgebra
using QuadGK

"""
    build_penalty_matrix(basis, order::Int; knots::Vector{Float64}=Float64[])

Construct the derivative-based penalty matrix using Wood's (2016) algorithm.

For a B-spline basis of order m₁ and penalty order m₂, computes:
    S_{ij} = ∫ B_i^(m₂)(t) B_j^(m₂)(t) dt

The algorithm uses Gauss-Legendre quadrature on each knot interval with enough
points to integrate the product exactly.
"""
function build_penalty_matrix(basis, order::Int; knots::Vector{Float64}=Float64[])
    order >= 1 || throw(ArgumentError("Penalty order must be ≥ 1, got $order"))
    
    # Get knot vector - handle both BSplineBasis and RecombinedBSplineBasis
    if isempty(knots)
        if basis isa BSplineBasis
            knots = collect(BSplineKit.knots(basis))
        elseif basis isa RecombinedBSplineBasis
            knots = collect(BSplineKit.knots(parent(basis)))
        else
            error("Unsupported basis type: $(typeof(basis))")
        end
    end
    
    # Get number of basis functions
    K = length(basis)
    
    # Get spline order (degree + 1)
    m1 = BSplineKit.order(basis)
    
    # Derivative reduces order
    m_deriv = m1 - order
    if m_deriv < 1
        # Derivative order exceeds spline degree, penalty is identically zero
        return zeros(K, K)
    end
    
    # Number of Gauss-Legendre points per interval
    n_gauss = max(m_deriv, 2)
    
    # Get Gauss-Legendre nodes and weights on [-1, 1]
    gl_nodes, gl_weights = _gauss_legendre(n_gauss)
    
    # Unique interior breakpoints (excluding repeated boundary knots)
    unique_knots = unique(knots)
    n_intervals = length(unique_knots) - 1
    
    # Initialize penalty matrix
    S = zeros(K, K)
    
    # Integrate over each knot interval
    for q in 1:n_intervals
        a, b = unique_knots[q], unique_knots[q + 1]
        h = b - a
        
        # Skip degenerate intervals
        h < 1e-14 && continue
        
        # Transform Gauss points to [a, b]
        t_points = @. a + (gl_nodes + 1) * h / 2
        w_scaled = @. gl_weights * h / 2
        
        # Evaluate basis derivatives at quadrature points
        for (i_pt, t) in enumerate(t_points)
            # Get derivative values at this point
            deriv_vals = _evaluate_basis_derivatives(basis, t, order)
            
            # Outer product contribution to penalty matrix
            w = w_scaled[i_pt]
            for i in 1:K
                for j in i:K  # Only upper triangle due to symmetry
                    S[i, j] += w * deriv_vals[i] * deriv_vals[j]
                end
            end
        end
    end
    
    # Symmetrize (copy upper to lower)
    for i in 1:K
        for j in (i+1):K
            S[j, i] = S[i, j]
        end
    end
    
    return S
end

"""
    _evaluate_basis_derivatives(basis, t::Real, order::Int) -> Vector{Float64}

Evaluate all basis function derivatives of given order at point t.
"""
function _evaluate_basis_derivatives(basis, t::Real, order::Int)
    K = length(basis)
    deriv_vals = zeros(K)
    
    # For each basis function, create a spline with unit coefficient
    # and take the derivative using diff()
    for i in 1:K
        coeffs = zeros(K)
        coeffs[i] = 1.0
        spl = Spline(basis, coeffs)
        
        # Apply diff() `order` times to get the requested derivative
        dspl = spl
        for _ in 1:order
            dspl = diff(dspl)
        end
        
        deriv_vals[i] = dspl(t)
    end
    
    return deriv_vals
end

"""
    _gauss_legendre(n::Int) -> (nodes::Vector{Float64}, weights::Vector{Float64})

Compute n-point Gauss-Legendre quadrature nodes and weights on [-1, 1].
"""
function _gauss_legendre(n::Int)
    n >= 1 || throw(ArgumentError("n must be ≥ 1"))
    
    if n == 1
        return [0.0], [2.0]
    end
    
    # Jacobi matrix for Legendre polynomials
    beta = [k / sqrt(4.0 * k^2 - 1.0) for k in 1:(n-1)]
    J = SymTridiagonal(zeros(n), beta)
    eig = eigen(J)
    nodes = eig.values
    weights = [2.0 * eig.vectors[1, k]^2 for k in 1:n]
    
    return nodes, weights
end

"""
    build_tensor_penalty_matrix(Sx::Matrix{Float64}, Sy::Matrix{Float64}) -> Matrix{Float64}

Build the isotropic tensor product penalty matrix.

For marginal penalty matrices Sx (kx × kx) and Sy (ky × ky), the tensor
penalty is:
    S_te = Sx ⊗ Iy + Ix ⊗ Sy

This penalizes roughness equally in both directions, producing smooth surfaces.
The ordering matches the row-wise Kronecker product: kron(Bx, By).

# Arguments
- `Sx`: Penalty matrix for the first dimension (kx × kx)
- `Sy`: Penalty matrix for the second dimension (ky × ky)

# Returns
- `S_te`: Tensor penalty matrix ((kx*ky) × (kx*ky))

# References
- Wood, S.N. (2006). Generalized Additive Models: An Introduction with R, Section 4.1.8
"""
function build_tensor_penalty_matrix(Sx::Matrix{Float64}, Sy::Matrix{Float64})
    kx = size(Sx, 1)
    ky = size(Sy, 1)
    
    @assert size(Sx, 2) == kx "Sx must be square"
    @assert size(Sy, 2) == ky "Sy must be square"
    
    Ix = Matrix{Float64}(I, kx, kx)
    Iy = Matrix{Float64}(I, ky, ky)
    
    # S_te = Sx ⊗ Iy + Ix ⊗ Sy
    S_te = kron(Sx, Iy) + kron(Ix, Sy)
    
    return S_te
end
