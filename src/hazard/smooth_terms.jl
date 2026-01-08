# =============================================================================
# Smooth Terms for Formula DSL
# =============================================================================
#
# Implementation of s(x) and te(x, y) for smooth covariate effects.
# Extends StatsModels.jl formula DSL following mgcv conventions.
#
# =============================================================================

using StatsModels
using BSplineKit
using LinearAlgebra

# -----------------------------------------------------------------------------
# Syntax Functions
# -----------------------------------------------------------------------------

"""
    s(x; k=10, m=2, order=4)
    s(x, k)
    s(x, k, m)
    s(x, k, m, order)

Specify a smooth term in a hazard formula, following mgcv conventions.

# Arguments (positional in formula)
- `x`: Covariate name (required)
- `k`: Number of basis functions (default 10)
- `m`: Penalty order for derivative penalty (default 2 = curvature penalty)
- `order`: B-spline order (default 4 = cubic). Order 3 = quadratic, order 2 = linear.

# Examples
```julia
# Cubic spline with 10 basis functions and curvature penalty (defaults)
Hazard(@formula(0 ~ s(age)), :sp, 1, 2)

# 15 basis functions
Hazard(@formula(0 ~ s(age, 15)), :sp, 1, 2)

# 15 basis functions with 3rd derivative penalty
Hazard(@formula(0 ~ s(age, 15, 3)), :sp, 1, 2)

# Quadratic splines (order 3) with 15 basis functions
Hazard(@formula(0 ~ s(age, 15, 2, 3)), :sp, 1, 2)
```

# Notes
- `order=4` (cubic) is most common and recommended for smooth curves
- `order=3` (quadratic) useful when cubic is overparameterized
- `order=2` (linear) for piecewise linear splines
- The penalty matrix uses m-th derivative differences
- Follows R's mgcv package conventions where possible
"""
s(x; k=10, m=2, order=4) = x # Dummy for syntax

"""
    te(x, y; k=5, m=2, order=4)
    te(x, y, k)
    te(x, y, k, m)
    te(x, y, kx, ky, m)
    te(x, y, kx, ky, m, order)

Specify a tensor product smooth term in a hazard formula.

The basis is the row-wise Kronecker product of two B-spline bases, producing
`kx * ky` basis functions. The penalty uses isotropic sum: S = Sx ⊗ Iy + Ix ⊗ Sy.

# Arguments (positional in formula)
- `x`, `y`: Covariate names (required)
- `k` or `kx`, `ky`: Number of basis functions per dimension (default 5 each)
- `m`: Penalty order (default 2 = curvature penalty)
- `order`: B-spline order (default 4 = cubic)

# Examples
```julia
# Default tensor product smooth
Hazard(@formula(0 ~ te(age, bmi)), :sp, 1, 2)

# 8 basis functions per dimension
Hazard(@formula(0 ~ te(age, bmi, 8)), :sp, 1, 2)

# Different k per dimension: 10 for age, 6 for bmi
Hazard(@formula(0 ~ te(age, bmi, 10, 6, 2)), :sp, 1, 2)

# Quadratic splines with custom k
Hazard(@formula(0 ~ te(age, bmi, 8, 8, 2, 3)), :sp, 1, 2)
```

# Notes
- Total parameters = kx × ky, so be mindful of dimensionality
- For two continuous covariates, consider starting with k=5 per dimension (25 params)
"""
te(x, y; k=5, m=2, order=4) = (x, y) # Dummy for syntax

# -----------------------------------------------------------------------------
# Term Types
# -----------------------------------------------------------------------------

struct SmoothTerm{T, B, S_mat} <: AbstractTerm
    term::T          # Underlying ContinuousTerm
    basis::B         # BSplineBasis
    S::S_mat         # Penalty matrix
    knots::Int       # k
    order::Int       # Spline order (default 4)
    penalty_order::Int # m
    label::String    # For coefnames
end

"""
    TensorProductTerm <: AbstractTerm

Represents a tensor product smooth `te(x, y)` for modeling smooth interactions.

The basis is the row-wise Kronecker product of two B-spline bases, producing
`kx * ky` basis functions. The penalty matrix uses the isotropic sum:
    S_te = S_x ⊗ I_y + I_x ⊗ S_y
"""
struct TensorProductTerm{Tx, Ty, Bx, By, S_mat} <: AbstractTerm
    term_x::Tx       # First ContinuousTerm (x)
    term_y::Ty       # Second ContinuousTerm (y)
    basis_x::Bx      # BSplineBasis for x
    basis_y::By      # BSplineBasis for y
    S::S_mat         # Tensor penalty matrix (kx*ky × kx*ky)
    kx::Int          # Number of basis functions for x
    ky::Int          # Number of basis functions for y
    order::Int       # Spline order (default 4)
    penalty_order::Int # m
    label::String    # For coefnames
end

# -----------------------------------------------------------------------------
# StatsModels Interface
# -----------------------------------------------------------------------------

function StatsModels.apply_schema(t::FunctionTerm{typeof(s)}, sch::StatsModels.Schema, Mod::Type{<:Any})
    # Extract variable term
    var_term = apply_schema(t.args[1], sch, Mod)
    isa(var_term, ContinuousTerm) || throw(ArgumentError("s() only works with continuous terms, got $(typeof(var_term))"))
    
    # Extract options from positional args: s(x, k, m, order)
    # Defaults follow mgcv conventions
    k = 10            # number of basis functions
    m = 2             # penalty order (2 = curvature penalty)
    spline_order = 4  # B-spline order (4 = cubic, 3 = quadratic, 2 = linear)
    
    # Parse k (2nd arg)
    if length(t.args) > 1
        if t.args[2] isa ConstantTerm
            k = Int(t.args[2].n)
        end
    end
    
    # Parse m (3rd arg) - penalty order
    if length(t.args) > 2
        if t.args[3] isa ConstantTerm
            m = Int(t.args[3].n)
        end
    end
    
    # Parse order (4th arg) - B-spline order
    if length(t.args) > 3
        if t.args[4] isa ConstantTerm
            spline_order = Int(t.args[4].n)
        end
    end
    
    # Validate spline order
    if spline_order < 1 || spline_order > 6
        throw(ArgumentError("spline order must be between 1 and 6, got $spline_order"))
    end
    
    # Data range
    min_val = var_term.min
    max_val = var_term.max
    
    # BSplineKit augments boundary knots with (order-1) repetitions
    # To get k basis functions with given order: n_input_knots = k + 2 - order
    n_input_knots = k + 2 - spline_order
    if n_input_knots < 2
        throw(ArgumentError("k=$k is too small for spline order $spline_order. " *
                           "Need k ≥ $(spline_order) (minimum $spline_order basis functions)"))
    end
    
    # Create evenly spaced input knots
    knots = collect(range(min_val, max_val, length=n_input_knots))
    basis = BSplineBasis(BSplineOrder(spline_order), knots)
    
    # Verify we got the expected number of basis functions
    @assert length(basis) == k "Expected $k basis functions, got $(length(basis))"
    
    # Compute penalty matrix
    S = build_penalty_matrix(basis, m)
    
    label = "s($(var_term.sym))"
    
    return SmoothTerm(var_term, basis, S, k, spline_order, m, label)
end

"""
    apply_schema for te(x, y, ...) tensor product smooth

Supports multiple syntaxes:
- `te(x, y)`: defaults kx=ky=5, m=2, order=4
- `te(x, y, k)`: same k for both dimensions
- `te(x, y, k, m)`: same k for both, custom penalty order
- `te(x, y, kx, ky, m)`: different k per dimension
- `te(x, y, kx, ky, m, order)`: full control including spline order
"""
function StatsModels.apply_schema(t::FunctionTerm{typeof(te)}, sch::StatsModels.Schema, Mod::Type{<:Any})
    # Must have at least 2 variable arguments
    length(t.args) >= 2 || throw(ArgumentError("te() requires at least two variable arguments"))
    
    # Extract variable terms
    var_term_x = apply_schema(t.args[1], sch, Mod)
    var_term_y = apply_schema(t.args[2], sch, Mod)
    
    isa(var_term_x, ContinuousTerm) || throw(ArgumentError("te() first argument must be continuous, got $(typeof(var_term_x))"))
    isa(var_term_y, ContinuousTerm) || throw(ArgumentError("te() second argument must be continuous, got $(typeof(var_term_y))"))
    
    # Extract options from positional args
    # Syntax: te(x, y), te(x, y, k), te(x, y, k, m), te(x, y, kx, ky, m), te(x, y, kx, ky, m, order)
    spline_order = 4  # default: cubic splines
    kx = 5            # default number of basis functions
    ky = 5
    m = 2             # default: curvature penalty
    
    n_extra_args = length(t.args) - 2
    
    if n_extra_args == 0
        # te(x, y) - Use defaults: kx=ky=5, m=2, order=4
    elseif n_extra_args == 1
        # te(x, y, k) - same k for both, m=2, order=4
        if t.args[3] isa ConstantTerm
            kx = ky = Int(t.args[3].n)
        end
    elseif n_extra_args == 2
        # te(x, y, k, m) - same k for both, custom m, order=4
        if t.args[3] isa ConstantTerm
            kx = ky = Int(t.args[3].n)
        end
        if t.args[4] isa ConstantTerm
            m = Int(t.args[4].n)
        end
    elseif n_extra_args == 3
        # te(x, y, kx, ky, m) - different k per dimension, order=4
        if t.args[3] isa ConstantTerm
            kx = Int(t.args[3].n)
        end
        if t.args[4] isa ConstantTerm
            ky = Int(t.args[4].n)
        end
        if t.args[5] isa ConstantTerm
            m = Int(t.args[5].n)
        end
    elseif n_extra_args >= 4
        # te(x, y, kx, ky, m, order) - full control
        if t.args[3] isa ConstantTerm
            kx = Int(t.args[3].n)
        end
        if t.args[4] isa ConstantTerm
            ky = Int(t.args[4].n)
        end
        if t.args[5] isa ConstantTerm
            m = Int(t.args[5].n)
        end
        if t.args[6] isa ConstantTerm
            spline_order = Int(t.args[6].n)
        end
    end
    
    # Validate spline order
    if spline_order < 1 || spline_order > 6
        throw(ArgumentError("spline order must be between 1 and 6, got $spline_order"))
    end
    
    # Build basis for x
    min_x, max_x = var_term_x.min, var_term_x.max
    n_input_knots_x = kx + 2 - spline_order
    if n_input_knots_x < 2
        throw(ArgumentError("kx=$kx is too small for spline order $spline_order. " *
                           "Need kx ≥ $spline_order"))
    end
    knots_x = collect(range(min_x, max_x, length=n_input_knots_x))
    basis_x = BSplineBasis(BSplineOrder(spline_order), knots_x)
    @assert length(basis_x) == kx "Expected $kx basis functions for x, got $(length(basis_x))"
    
    # Build basis for y
    min_y, max_y = var_term_y.min, var_term_y.max
    n_input_knots_y = ky + 2 - spline_order
    if n_input_knots_y < 2
        throw(ArgumentError("ky=$ky is too small for spline order $spline_order. " *
                           "Need ky ≥ $spline_order"))
    end
    knots_y = collect(range(min_y, max_y, length=n_input_knots_y))
    basis_y = BSplineBasis(BSplineOrder(spline_order), knots_y)
    @assert length(basis_y) == ky "Expected $ky basis functions for y, got $(length(basis_y))"
    
    # Compute marginal penalty matrices
    Sx = build_penalty_matrix(basis_x, m)
    Sy = build_penalty_matrix(basis_y, m)
    
    # Tensor penalty: S = Sx ⊗ Iy + Ix ⊗ Sy
    S = build_tensor_penalty_matrix(Sx, Sy)
    
    label = "te($(var_term_x.sym),$(var_term_y.sym))"
    
    return TensorProductTerm(var_term_x, var_term_y, basis_x, basis_y, S, kx, ky, spline_order, m, label)
end

StatsModels.terms(p::SmoothTerm) = terms(p.term)
StatsModels.termvars(p::SmoothTerm) = StatsModels.termvars(p.term)
StatsModels.width(p::SmoothTerm) = p.knots

function StatsModels.coefnames(p::SmoothTerm)
    return ["$(p.label)_$i" for i in 1:p.knots]
end

# TensorProductTerm StatsModels interface
StatsModels.terms(p::TensorProductTerm) = (terms(p.term_x)..., terms(p.term_y)...)
StatsModels.termvars(p::TensorProductTerm) = vcat(StatsModels.termvars(p.term_x), StatsModels.termvars(p.term_y))
StatsModels.width(p::TensorProductTerm) = p.kx * p.ky

function StatsModels.coefnames(p::TensorProductTerm)
    return ["$(p.label)_$i" for i in 1:(p.kx * p.ky)]
end

"""
    _eval_basis_full(basis, x) -> Vector{Float64}

Evaluate all basis functions at point x, returning a full-length vector.
BSplineKit's basis(x) returns (first_index, values_tuple), we expand to full vector.
"""
function _eval_basis_full(basis, x::Real)
    K = length(basis)
    order = BSplineKit.order(basis)
    idx, vals = basis(x)
    result = zeros(K)
    for (j, v) in enumerate(vals)
        i = idx - order + j
        if 1 <= i <= K
            result[i] = v
        end
    end
    return result
end

function StatsModels.modelcols(p::SmoothTerm, d::NamedTuple)
    # Validate that required variable exists in data
    var_syms = StatsModels.termvars(p.term)
    for sym in var_syms
        haskey(d, sym) || throw(ArgumentError("SmoothTerm requires variable :$sym but it was not found in data"))
    end
    
    val = modelcols(p.term, d)
    # Evaluate basis at val
    if val isa AbstractVector
        # Matrix of basis functions
        n = length(val)
        B = zeros(n, p.knots)
        for i in 1:n
            B[i, :] = _eval_basis_full(p.basis, val[i])
        end
        return B
    else
        # Single row - return row vector
        return reshape(_eval_basis_full(p.basis, val), 1, :)
    end
end

# Handle DataFrameRow by converting to NamedTuple
function StatsModels.modelcols(p::SmoothTerm, d::DataFrameRow)
    return modelcols(p, NamedTuple(d))
end

"""
    modelcols for TensorProductTerm

Computes row-wise Kronecker product of the two marginal bases.
For each row, B_te[i,:] = kron(B_x[i,:], B_y[i,:])
"""
function StatsModels.modelcols(p::TensorProductTerm, d::NamedTuple)
    # Validate that required variables exist in data
    var_syms = StatsModels.termvars(p)
    for sym in var_syms
        haskey(d, sym) || throw(ArgumentError("TensorProductTerm requires variable :$sym but it was not found in data"))
    end
    
    val_x = modelcols(p.term_x, d)
    val_y = modelcols(p.term_y, d)
    
    # Handle scalar vs vector case
    if val_x isa AbstractVector && val_y isa AbstractVector
        n = length(val_x)
        @assert length(val_y) == n "Tensor product terms must have same number of observations"
        
        k_total = p.kx * p.ky
        B = zeros(n, k_total)
        for i in 1:n
            Bx_i = _eval_basis_full(p.basis_x, val_x[i])
            By_i = _eval_basis_full(p.basis_y, val_y[i])
            # Row-wise Kronecker: kron(Bx, By) produces kx*ky elements
            B[i, :] = kron(Bx_i, By_i)
        end
        return B
    else
        # Single observation
        x_val = val_x isa AbstractVector ? val_x[1] : val_x
        y_val = val_y isa AbstractVector ? val_y[1] : val_y
        Bx = _eval_basis_full(p.basis_x, x_val)
        By = _eval_basis_full(p.basis_y, y_val)
        return reshape(kron(Bx, By), 1, :)
    end
end

function StatsModels.modelcols(p::TensorProductTerm, d::DataFrameRow)
    return modelcols(p, NamedTuple(d))
end

# -----------------------------------------------------------------------------
# Basis Expansion for Data
# -----------------------------------------------------------------------------

"""
    expand_smooth_term_columns!(data::DataFrame, hazard::HazardFunction)

Expand smooth term basis functions into columns in the DataFrame.

For a hazard with formula `0 ~ s(age, 5, 2)`, this adds columns 
`s(age)_1`, `s(age)_2`, ..., `s(age)_5` to the data.

This must be called before likelihood computation since the likelihood
code expects all covariate names to be actual columns in the data.
"""
function expand_smooth_term_columns!(data::DataFrame, hazard::HazardFunction)
    schema = StatsModels.schema(hazard.hazard, data)
    hazschema = apply_schema(hazard.hazard, schema)
    
    # Find SmoothTerms in the schema
    rhs = hazschema.rhs
    _expand_smooth_columns_from_term!(data, rhs)
    return data
end

"""
    _expand_smooth_columns_from_term!(data::DataFrame, term)

Recursively find and expand SmoothTerms and TensorProductTerms in a term tree.
"""
function _expand_smooth_columns_from_term!(data::DataFrame, term)
    # Base case: SmoothTerm - expand it
    if term isa SmoothTerm
        _add_smooth_basis_columns!(data, term)
    # Base case: TensorProductTerm - expand it
    elseif term isa TensorProductTerm
        _add_tensor_basis_columns!(data, term)
    # MatrixTerm - check inner terms
    elseif term isa StatsModels.MatrixTerm
        for t in term.terms
            _expand_smooth_columns_from_term!(data, t)
        end
    # InteractionTerm - check components
    elseif term isa StatsModels.InteractionTerm
        for t in term.terms
            _expand_smooth_columns_from_term!(data, t)
        end
    # Tuple of terms (e.g., from + in formula)
    elseif term isa Tuple
        for t in term
            _expand_smooth_columns_from_term!(data, t)
        end
    end
    # Other term types (ContinuousTerm, CategoricalTerm, etc.) - no expansion needed
    return nothing
end

"""
    _add_smooth_basis_columns!(data::DataFrame, p::SmoothTerm)

Add basis function columns for a SmoothTerm to the DataFrame.

Note: The basis is constructed from the data range when the schema is applied,
so this function cannot detect extrapolation (the basis adapts to the new data).
To detect extrapolation, the original basis boundaries would need to be stored
in SmoothTermInfo during model construction and checked here.
"""
function _add_smooth_basis_columns!(data::DataFrame, p::SmoothTerm)
    coefs = StatsModels.coefnames(p)
    
    # Check if columns already exist
    if all(Symbol(c) in propertynames(data) for c in coefs)
        return data
    end
    
    # Get the underlying variable values
    var_sym = StatsModels.termvars(p.term)[1]
    x = data[!, var_sym]
    
    # Evaluate basis for each row
    n = nrow(data)
    for (j, cname) in enumerate(coefs)
        col_sym = Symbol(cname)
        if col_sym ∉ propertynames(data)
            vals = zeros(Float64, n)
            for i in 1:n
                bvals = _eval_basis_full(p.basis, x[i])
                vals[i] = bvals[j]
            end
            data[!, col_sym] = vals
        end
    end
    
    return data
end

"""
    _add_tensor_basis_columns!(data::DataFrame, p::TensorProductTerm)

Add basis function columns for a TensorProductTerm to the DataFrame.
The tensor product basis is the row-wise Kronecker product of the marginal bases.

Note: The bases are constructed from the data range when the schema is applied,
so this function cannot detect extrapolation (the bases adapt to the new data).
To detect extrapolation, the original basis boundaries would need to be stored
in SmoothTermInfo during model construction and checked here.
"""
function _add_tensor_basis_columns!(data::DataFrame, p::TensorProductTerm)
    coefs = StatsModels.coefnames(p)
    
    # Check if columns already exist
    if all(Symbol(c) in propertynames(data) for c in coefs)
        return data
    end
    
    # Get the underlying variable values
    var_syms = StatsModels.termvars(p)
    x = data[!, var_syms[1]]
    y = data[!, var_syms[2]]
    
    # Evaluate tensor basis for each row
    n = nrow(data)
    k_total = p.kx * p.ky
    
    # Pre-compute all basis values
    for (j, cname) in enumerate(coefs)
        col_sym = Symbol(cname)
        if col_sym ∉ propertynames(data)
            vals = zeros(Float64, n)
            for i in 1:n
                Bx_i = _eval_basis_full(p.basis_x, x[i])
                By_i = _eval_basis_full(p.basis_y, y[i])
                # Row-wise Kronecker: kron(Bx, By)
                tensor_vals = kron(Bx_i, By_i)
                vals[i] = tensor_vals[j]
            end
            data[!, col_sym] = vals
        end
    end
    
    return data
end

"""
    expand_all_smooth_terms!(data::DataFrame, hazards)

Expand smooth term columns for all hazards into the data DataFrame.
Returns the modified data.
"""
function expand_all_smooth_terms!(data::DataFrame, hazards)
    for hazard in hazards
        if hazard isa HazardFunction
            expand_smooth_term_columns!(data, hazard)
        end
    end
    return data
end

# -----------------------------------------------------------------------------
# Display
# -----------------------------------------------------------------------------

function Base.show(io::IO, p::SmoothTerm)
    print(io, "$(p.label) (k=$(p.knots), m=$(p.penalty_order))")
end

function Base.show(io::IO, p::TensorProductTerm)
    print(io, "$(p.label) (kx=$(p.kx), ky=$(p.ky), m=$(p.penalty_order))")
end
