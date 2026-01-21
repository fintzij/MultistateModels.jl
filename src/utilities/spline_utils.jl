# =============================================================================
# Spline Utilities
# =============================================================================
#
# Shared utilities for B-spline basis construction and penalty matrices.
# Used by both baseline splines (SplineHazard) and covariate splines (s()).
#
# Penalty matrix construction methods:
# - GPS (default): Li & Cao (2022) general P-spline via weighted differences
# - Integral: O'Sullivan (1986) O-spline via quadrature of ∫B''B''dt
#
# =============================================================================

using BSplineKit
using LinearAlgebra
using QuadGK

# =============================================================================
# General P-Spline (GPS) Implementation - Li & Cao (2022)
# =============================================================================

"""
    build_general_difference_matrix(knots::Vector{Float64}, d::Int, m::Int) -> Matrix{Float64}

Construct the general difference matrix D_m for Li & Cao's general P-spline penalty.

For B-splines of order `d` (degree `d-1`) with penalty order `m`, this computes the
weighted difference matrix that correctly accounts for non-uniform knot spacing.

# Arguments
- `knots`: Full knot sequence (with repeated boundary knots for clamped splines).
  For K basis functions with order d, this should have length K + d.
- `d`: B-spline order (degree + 1). For cubic splines, d = 4.
- `m`: Penalty order. m = 2 for curvature penalty.

# Returns
- `D_m`: General difference matrix of size (K - m) × K, where K = length(knots) - d

# Mathematical Details
The matrix is computed iteratively:

    D_m = W_m⁻¹ Δ W_{m-1}⁻¹ Δ ⋯ W_1⁻¹ Δ

where:
- Δ is the first-difference matrix: Δ_{i,i} = -1, Δ_{i,i+1} = +1
- W_j is diagonal with entries: (W_j)_{ii} = (t_{i+d-j+1} - t_{i+1}) / (d-j)

The indexing uses 1-based Julia conventions. For the j-th iteration processing
row i, the weights use knot indices i+1 and i+d-j+1.

# Notes
For clamped B-splines (like those from BSplineKit), the knot vector has 
repeated boundary knots (e.g., [0,0,0,0, ..., 1,1,1,1] for cubic). This function
handles such knot vectors correctly.

# References
- Li, Z. & Cao, J. (2022). "General P-Splines for Non-Uniform B-Splines." 
  arXiv:2201.06808
"""
function build_general_difference_matrix(knots::Vector{Float64}, d::Int, m::Int)
    n_knots = length(knots)
    K = n_knots - d  # Number of basis functions
    
    @assert m >= 1 "Penalty order must be ≥ 1, got m=$m"
    @assert m < d "Penalty order must be < spline order (m=$m, d=$d)"
    @assert K >= m "Need at least m basis functions for order-m penalty (K=$K, m=$m)"
    
    # Algorithm from gps R package (Li & Cao 2022):
    # Iteratively build D_m = W_m^{-1} Δ W_{m-1}^{-1} Δ ... W_1^{-1} Δ
    # where Δ is first-difference and W_j has entries h_j[i] = (t[i+k] - t[i]) / k
    # with k = d - j and starting index shifting by 1 each iteration.
    
    # Start with identity (K × K)
    D = Matrix{Float64}(I, K, K)
    
    for j in 1:m
        n_current = K - j + 1  # Current number of rows/cols in D
        n_next = K - j         # Number of rows after this iteration
        
        # Compute knot-spacing weights
        # h[i] = (t[i + k] - t[i]) / k where:
        #   k = d - j (the "lag" in knot positions)
        #   i runs from (j + 1) to (j + n_next) in 1-based Julia indexing
        # This matches R's Diff(xt, k=d-j, n=K-2*j, xi=j+1)
        k = d - j
        W_diag = zeros(n_next)
        for i in 1:n_next
            # In R's convention: i goes from (j+1) to (K-j)
            # Julia index: i_knot = j + i
            i_knot = j + i
            t_high = knots[i_knot + k]
            t_low = knots[i_knot]
            W_diag[i] = (t_high - t_low) / k
        end
        
        # Check for zero weights (would indicate coincident knots in interior)
        if any(abs.(W_diag) .< 1e-14)
            @warn "Near-zero weight in general difference matrix at iteration j=$j (coincident knots?)"
            # Replace near-zero with small value to avoid division by zero
            W_diag[abs.(W_diag) .< 1e-14] .= 1e-14
        end
        
        # Build weighted difference matrix: (n_next × n_current)
        # Entry (i, i) = -1/h[i], entry (i, i+1) = 1/h[i]
        WtDelta = zeros(n_next, n_current)
        for i in 1:n_next
            WtDelta[i, i] = -1.0 / W_diag[i]
            WtDelta[i, i + 1] = 1.0 / W_diag[i]
        end
        
        # Update D: D ← W_j^{-1} Δ D
        D = WtDelta * D
    end
    
    return D
end

"""
    build_penalty_matrix_gps(basis, order::Int; knots::Vector{Float64}=Float64[])

Construct the penalty matrix using Li & Cao's General P-Spline (GPS) algorithm.

The penalty matrix is S = D_m' D_m where D_m is the weighted m-th order difference 
matrix from the GPS algorithm. The weighting accounts for non-uniform knot spacing.

# Arguments
- `basis`: BSplineKit basis object (BSplineBasis or RecombinedBSplineBasis)
- `order::Int`: Difference order (m). Default 2 = curvature penalty.
- `knots`: Full knot vector for GPS weighting. If empty, extracted from basis.

# Returns
- `S`: Symmetric positive semi-definite penalty matrix (K × K)

# Notes
The penalty matrix has:
- Symmetry: S = S'
- Positive semi-definiteness: all eigenvalues ≥ 0
- Null space of dimension `order`: polynomials of degree < `order` are unpenalized

The GPS algorithm correctly handles both uniform and non-uniform knot spacing.
For BSplineKit bases (which use clamped B-splines), this function extracts the 
full knot vector including repeated boundary knots.

# Mathematical Details
The GPS algorithm builds the weighted difference matrix iteratively:

    D_m = W_m⁻¹ Δ W_{m-1}⁻¹ Δ ⋯ W_1⁻¹ Δ

where Δ is first-difference and W_j has diagonal entries h_j[i] = (t[i+k] - t[i]) / k
with k = d - j. This weighting ensures the penalty correctly accounts for 
non-uniform knot spacing.

# References
- Li, Z. & Cao, J. (2022). "General P-Splines for Non-Uniform B-Splines." 
  arXiv:2201.06808
- Eilers, P.H.C. & Marx, B.D. (1996). "Flexible Smoothing with B-splines and Penalties."
  Statistical Science 11(2), 89-121.
"""
function build_penalty_matrix_gps(basis, order::Int; knots::Vector{Float64}=Float64[])
    order >= 1 || throw(ArgumentError("Penalty order must be ≥ 1, got $order"))
    
    # Handle RecombinedBSplineBasis by building penalty for parent and projecting
    if basis isa BSplineKit.Recombinations.RecombinedBSplineBasis
        # Get parent basis and build penalty for it
        # Note: we don't pass explicit knots to parent - let it extract from parent basis
        parent_basis = BSplineKit.parent(basis)
        S_parent = build_penalty_matrix_gps(parent_basis, order)
        
        # Get recombination matrix R (transforms from reduced to parent coefficients)
        R = BSplineKit.Recombinations.recombination_matrix(basis)
        R_dense = Matrix{Float64}(R)
        
        # Project penalty into reduced basis: S_reduced = R^T * S_parent * R
        S_reduced = transpose(R_dense) * S_parent * R_dense
        return S_reduced
    end
    
    # Standard BSplineBasis case
    # Get spline order (degree + 1)
    d = BSplineKit.order(basis)
    K = length(basis)
    
    # Check if penalty order is valid
    if order >= d
        # Derivative order exceeds spline degree, penalty is identically zero
        return zeros(K, K)
    end
    
    # Get full knot vector from basis if not provided
    # For BSplineKit, this includes repeated boundary knots (clamped splines)
    if isempty(knots)
        knots = collect(BSplineKit.knots(basis))
    end
    
    # Verify knot vector length: should be K + d for K basis functions
    n_knots = length(knots)
    if n_knots != K + d
        throw(ArgumentError("Knot vector length ($n_knots) != K + d ($K + $d = $(K+d))"))
    end
    
    # Use GPS algorithm with full knot vector
    D_m = build_general_difference_matrix(knots, d, order)
    
    # Penalty matrix is D'D
    S = transpose(D_m) * D_m
    
    return S
end

"""
    _build_difference_matrix(K::Int, m::Int) -> Matrix{Float64}

Build the m-th order difference matrix of size (K-m) × K.

The difference matrix is computed iteratively by applying first differences m times.
"""
function _build_difference_matrix(K::Int, m::Int)
    @assert K > m "Need K > m for difference matrix, got K=$K, m=$m"
    
    # Start with identity (K × K)
    D = Matrix{Float64}(I, K, K)
    
    for j in 1:m
        n_rows = K - j
        n_cols = K - j + 1
        
        # First-difference matrix
        Delta = zeros(n_rows, n_cols)
        for i in 1:n_rows
            Delta[i, i] = -1.0
            Delta[i, i + 1] = 1.0
        end
        
        # Update D
        D = Delta * D
    end
    
    return D
end

# =============================================================================
# Integral-based (O-Spline) Implementation - O'Sullivan (1986)
# =============================================================================

"""
    build_penalty_matrix_integral(basis, order::Int; knots::Vector{Float64}=Float64[])

Construct the penalty matrix using the integral/derivative (O-spline) formulation.

Computes S^(m)_{ij} = ∫ B_i^(m)(t) B_j^(m)(t) dt via Gauss-Legendre quadrature.

# Arguments
- `basis`: BSplineKit basis object (BSplineBasis or RecombinedBSplineBasis)
- `order::Int`: Derivative order (m). Default 2 = curvature penalty.
- `knots`: Optional explicit knot sequence. If empty, extracted from basis.

# Returns
- `S`: Symmetric positive semi-definite penalty matrix (K × K)

# Notes
This is the original O-spline formulation (O'Sullivan, 1986). Uses Gauss-Legendre
quadrature on each knot interval with enough points to integrate exactly.

Preserved for comparison with the GPS formulation. Use `method=:integral` in
`build_penalty_matrix` to select this method.

# References
- O'Sullivan, F. (1986). "A Statistical Perspective on Ill-Posed Inverse Problems."
  Statistical Science 1(4), 502-518.
- Wood, S.N. (2017). Generalized Additive Models: An Introduction with R. 2nd ed.
"""
function build_penalty_matrix_integral(basis, order::Int; knots::Vector{Float64}=Float64[])
    order >= 1 || throw(ArgumentError("Penalty order must be ≥ 1, got $order"))
    
    # Get knot vector - handle both BSplineBasis and RecombinedBSplineBasis
    if isempty(knots)
        if basis isa BSplineBasis
            knots = collect(BSplineKit.knots(basis))
        elseif basis isa RecombinedBSplineBasis
            knots = collect(BSplineKit.knots(parent(basis)))
        else
            throw(ArgumentError("Unsupported basis type: $(typeof(basis)). Supported: BSplineBasis, RecombinedBSplineBasis"))
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

# =============================================================================
# Wrapper Function with Method Selection
# =============================================================================

"""
    build_penalty_matrix(basis, order::Int; knots::Vector{Float64}=Float64[], method::Symbol=:gps)

Construct the penalty matrix for penalized B-splines.

# Arguments
- `basis`: BSplineKit basis object (BSplineBasis or RecombinedBSplineBasis)
- `order::Int`: Derivative/difference order (m). Default 2 = curvature penalty.
- `knots`: Optional explicit knot sequence. If empty, extracted from basis.
- `method`: Penalty construction method:
  - `:gps` (default): General P-Spline (Li & Cao, 2022) — weighted difference D'D
  - `:integral`: O-spline (O'Sullivan, 1986) — integral ∫B''B''dt via quadrature

# Returns
- `S`: Symmetric positive semi-definite penalty matrix (K × K)

# Mathematical Details
Both methods produce penalty matrices that have:
- **Symmetry**: S = S'
- **Positive semi-definiteness**: all eigenvalues ≥ 0
- **Null space of dimension m**: polynomials of degree < m are unpenalized

## GPS Method (default)
Uses the general difference matrix D_m from Li & Cao (2022):
    S = D_m' D_m
This correctly handles non-uniform knot spacing and is O(K²) to compute.

## Integral Method
Computes the integral of products of m-th derivatives:
    S_{ij} = ∫ B_i^(m)(t) B_j^(m)(t) dt
This is computed via Gauss-Legendre quadrature and is O(K² × n_quad).

# Notes
- **Scale difference**: λ values are NOT directly comparable between methods.
  Use effective degrees of freedom (EDF) for comparison.
- **Non-uniform knots**: Both methods correctly handle non-uniform spacing.
- **Performance**: GPS is faster (no quadrature) and matches the P-spline literature.

# Examples
```julia
using BSplineKit

# Create cubic B-spline basis with 10 interior knots
knots = collect(range(0, 1, length=14))  # 10 interior + 4 boundary
basis = BSplineBasis(BSplineOrder(4), knots)

# Default: GPS method (recommended)
S_gps = build_penalty_matrix(basis, 2)

# Alternative: integral method (O'Sullivan formulation)
S_int = build_penalty_matrix(basis, 2; method=:integral)
```

# References
- Li, Z. & Cao, J. (2022). "General P-Splines for Non-Uniform B-Splines."
  arXiv:2201.06808
- O'Sullivan, F. (1986). "A Statistical Perspective on Ill-Posed Inverse Problems."
  Statistical Science 1(4), 502-518.
- Eilers, P.H.C. & Marx, B.D. (1996). "Flexible Smoothing with B-splines and Penalties."
  Statistical Science 11(2), 89-121.
"""
function build_penalty_matrix(basis, order::Int; 
                              knots::Vector{Float64}=Float64[], 
                              method::Symbol=:gps)
    if method == :gps
        return build_penalty_matrix_gps(basis, order; knots=knots)
    elseif method == :integral
        return build_penalty_matrix_integral(basis, order; knots=knots)
    else
        throw(ArgumentError("Unknown penalty method: $method. Use :gps or :integral"))
    end
end

# =============================================================================
# Helper Functions
# =============================================================================

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
# =============================================================================
# Monotone Spline Penalty Transformation
# =============================================================================

"""
    build_ispline_transform_matrix(basis; direction::Int=1, warn_on_ill_conditioned::Bool=true) -> Matrix{Float64}

Build the transformation matrix L for I-spline (monotone) parameterization.

For monotone splines, parameters are non-negative increments (`ests`) that are
transformed to B-spline coefficients (`coefs`) via:
    coefs = L * ests

where L encodes the cumulative sum transformation with knot-spacing weights.

# Arguments
- `basis`: BSplineKit basis object
- `direction::Int=1`: Direction of monotonicity (+1 for increasing, -1 for decreasing)
- `warn_on_ill_conditioned::Bool=true`: If true, warn when cond(L) > ISPLINE_CONDITION_WARNING_THRESHOLD

# Returns
- `L`: Transformation matrix (K × K) where K is the number of basis functions

# Details
The transformation in _spline_ests2coefs is:
1. Start with coefs = 0
2. For i = 2, ..., K: coefs[i] = coefs[i-1] + ests[i] * w[i]
   where w[i] = (t[i+k] - t[i]) / k
3. Add intercept: coefs .+= ests[1]

This means:
- coefs[1] = ests[1] + 0 = ests[1]
- coefs[i] = ests[1] + Σⱼ₌₂ⁱ ests[j] * w[j] for i ≥ 2

So L has the structure:
- L[i,1] = 1 for all i (intercept contributes everywhere)
- L[i,j] = w[j] for 2 ≤ j ≤ i (cumulative weighted sum)

For decreasing monotonicity (direction=-1), the output is reversed.

# Numerical Stability
The condition number of L determines precision loss when inverting (e.g., in rectify_coefs!).
If cond(L) > 1e10 (ISPLINE_CONDITION_WARNING_THRESHOLD), a warning is issued. Common causes:
- Very closely spaced knots
- Too many basis functions
- Poorly chosen knot placement

# Mathematical Background
The I-spline transformation ensures that when the parameters (ests) are constrained
to be non-negative (via box constraints), the resulting B-spline has non-negative
derivatives, producing a monotone function.

The penalty on B-spline coefficients P(coefs) = (λ/2) coefs' S coefs must be
transformed to parameter space:
    P(ests) = (λ/2) (L * ests)' S (L * ests) = (λ/2) ests' (L' S L) ests

Use `transform_penalty_for_monotone` to apply this transformation.

# References
- Ramsay, J.O. (1988). "Monotone Regression Splines in Action."
  Statistical Science 3(4), 425-441.

See also: [`transform_penalty_for_monotone`](@ref), [`_spline_ests2coefs`](@ref), `ISPLINE_CONDITION_WARNING_THRESHOLD`
"""
function build_ispline_transform_matrix(basis; direction::Int=1, warn_on_ill_conditioned::Bool=true)
    direction ∈ [-1, 1] || throw(ArgumentError("direction must be ±1, got $direction"))
    
    k = BSplineKit.order(basis)
    t = collect(BSplineKit.knots(basis))
    K = length(basis)  # Number of basis functions
    
    # Build transformation matrix L matching _spline_ests2coefs
    # coefs[1] = ests[1]
    # coefs[i] = ests[1] + Σⱼ₌₂ⁱ ests[j] * w[j] for i ≥ 2
    L = zeros(K, K)
    
    # First column: intercept contributes to all coefficients
    L[:, 1] .= 1.0
    
    # Compute weights w[j] = (t[j+k] - t[j]) / k for j = 2, ..., K
    # Then L[i,j] = w[j] for j ≤ i
    for j in 2:K
        w_j = (t[j + k] - t[j]) / k
        # This weight contributes to coefs[j], coefs[j+1], ..., coefs[K]
        for i in j:K
            L[i, j] = w_j
        end
    end
    
    # For decreasing monotonicity, we need to account for the reverse operation
    # In _spline_ests2coefs with monotone=-1, we compute increasing then reverse!(coefs)
    # This reverses only the output, not the input
    # So L_dec = P * L_inc where P is the permutation matrix that reverses rows
    if direction == -1
        # Reverse rows only (output reversal)
        L = L[K:-1:1, :]
    end
    
    # H8_P2: Check condition number and warn if ill-conditioned
    # An ill-conditioned L means the inverse (used in rectify_coefs! and coefficient recovery)
    # will lose significant precision, potentially causing numerical issues
    if warn_on_ill_conditioned
        cond_L = cond(L)
        if cond_L > ISPLINE_CONDITION_WARNING_THRESHOLD
            @warn "I-spline transformation matrix is ill-conditioned (cond(L) = $(round(cond_L, sigdigits=3))). " *
                  "This can cause numerical precision loss in coefficient rectification. " *
                  "Consider: (1) reducing the number of knots, (2) ensuring knots are not too closely spaced, " *
                  "or (3) using wider knot spacing. Current basis has $K functions."
        end
    end
    
    return L
end

"""
    transform_penalty_for_monotone(S::Matrix{Float64}, basis; direction::Int=1) -> Matrix{Float64}

Transform a B-spline penalty matrix for use with monotone (I-spline) parameterization.

For monotone splines, the optimization parameters are I-spline increments (`ests`),
not B-spline coefficients (`coefs`). The relationship is:
    coefs = L * ests

where L is the I-spline transformation matrix.

The penalty P(β) = (λ/2) β'Sβ on B-spline coefficients must be transformed:
    P(ests) = (λ/2) (L * ests)' S (L * ests) = (λ/2) ests' (L' S L) ests

This function computes S_monotone = L' * S * L.

# Arguments
- `S`: Penalty matrix for B-spline coefficients (K × K)
- `basis`: BSplineKit basis object used to build L
- `direction::Int=1`: Monotonicity direction (+1 increasing, -1 decreasing)

# Returns
- `S_monotone`: Transformed penalty matrix for monotone parameters (K × K)

# Notes
This transformation is **required** for correct penalized fitting of monotone splines.
Without it, the penalty acts on the wrong parameter space, leading to incorrect
smoothing behavior.

The transformed matrix S_monotone inherits the positive semi-definiteness of S,
but may have additional null space dimensions due to the structure of L (particularly
for boundary basis functions with zero knot-spacing weights).

# Example
```julia
basis = BSplineBasis(4, [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0])
S_bspline = build_penalty_matrix(basis, 2)  # Penalty for B-spline coefs
S_monotone = transform_penalty_for_monotone(S_bspline, basis)  # Penalty for I-spline ests
```

See also: [`build_ispline_transform_matrix`](@ref), [`build_penalty_matrix`](@ref)
"""
function transform_penalty_for_monotone(S::Matrix{Float64}, basis; direction::Int=1)
    L = build_ispline_transform_matrix(basis; direction=direction)
    return L' * S * L
end