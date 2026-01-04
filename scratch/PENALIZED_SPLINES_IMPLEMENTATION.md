# Penalized Splines Implementation in MultistateModels.jl

## Document Status
**Version**: 1.0  
**Last Updated**: January 2, 2026  
**Branch**: `penalized_splines`  
**Status**: Phase 3 Complete (Feature-Complete)

---

## 1. Executive Summary

### What Was Implemented

The `penalized_splines` branch adds comprehensive support for flexible, semi-parametric hazard modeling through:

1. **Baseline Hazard Splines (`:sp` family)**: B-spline basis representation of baseline hazards with derivative-based penalties
2. **Smooth Covariate Effects**: GAM-style `s(x)` and `te(x,y)` syntax for smooth functions of covariates
3. **Automatic Smoothing Parameter Selection**: PIJCV (Predictive Infinitesimal Jackknife Cross-Validation) for data-driven λ selection

### Key Design Decisions

- **flexsurv/survextrap compatibility**: Follows R flexsurv parameterization conventions for hazard families
- **BSplineKit integration**: Leverages BSplineKit.jl for numerical B-spline evaluation
- **StatsModels extension**: Implements `SmoothTerm <: AbstractTerm` for formula DSL integration
- **Log-scale parameterization**: Spline coefficients stored on log scale (γ) for positivity; penalty applied to natural scale (β = exp(γ))
- **Functional hazard generation**: RuntimeGeneratedFunctions create AD-compatible hazard/cumhaz closures

### Current Status

- ✅ Phase 0: Penalty infrastructure and baseline spline hazards
- ✅ Phase 1: Spline knot calibration and helper functions
- ✅ Phase 2: Smooth covariate terms `s(x)` with penalty matrix construction
- ✅ Phase 3: Tensor product smooths `te(x,y)` and lambda sharing options

---

## 2. Implemented Components

### 2.1 Baseline Hazard Splines (`:sp` family)

#### Purpose
Enable flexible, non-parametric baseline hazard estimation using penalized B-splines. The hazard is represented as:

$$h_0(t) = \sum_{i=1}^{K} \beta_i B_i(t)$$

where $B_i(t)$ are B-spline basis functions and $\beta_i > 0$ are coefficients.

#### Key Types and Functions

| Type/Function | File | Description |
|---------------|------|-------------|
| `SplineHazard` | `src/hazard/hazard_specs.jl` | User-facing hazard specification |
| `RuntimeSplineHazard` | `src/types/hazard_structs.jl` | Internal runtime hazard type |
| `SplineHazardInfo` | `src/hazard/spline.jl` | Penalty metadata per hazard |

**Knot Placement Functions** ([spline.jl](../src/hazard/spline.jl)):
- `default_nknots(n)`: Sieve estimation rule: `floor(n^(1/5))`
- `place_interior_knots(sojourns, nknots; ...)`: Quantile-based knot placement with tie handling
- `place_interior_knots_pooled(model, origin, nknots)`: Pool sojourns across competing hazards

**Calibration Functions**:
- `calibrate_splines(model; ...)`: Compute knot locations from data (returns NamedTuple)
- `calibrate_splines!(model; ...)`: In-place knot calibration and model rebuild

#### Public API
```julia
# Create spline hazard
h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
             degree=3,                    # Cubic B-splines
             knots=[0.3, 0.5, 0.7],       # Interior knots
             boundaryknots=[0.0, 1.0],    # Boundary knots
             natural_spline=true,         # Natural boundary conditions
             monotone=0)                  # 0=none, 1=increasing, -1=decreasing

# Calibrate knots from data
model = multistatemodel(h12; data=data)
knots = calibrate_splines!(model; nknots=5)
```

#### Internal API
- `build_spline_hazard_info(hazard; penalty_order=2)`: Construct penalty metadata
- `_rebuild_spline_basis(hazard)`: Reconstruct BSplineBasis from RuntimeSplineHazard
- `_generate_spline_hazard_fns(...)`: Generate RuntimeGeneratedFunction closures

### 2.2 Penalty Infrastructure

#### Purpose
Provide a rule-based system for specifying penalties on spline hazards and smooth covariates, with support for shared smoothing parameters across competing risks.

#### Key Types

**`SplinePenalty`** ([infrastructure.jl](../src/types/infrastructure.jl#L355) — User-facing):
```julia
struct SplinePenalty
    selector::Union{Symbol, Int, Tuple{Int,Int}}  # :all, origin, or (origin, dest)
    order::Int                                     # Derivative order (default 2)
    total_hazard::Bool                            # Penalize sum of competing hazards
    share_lambda::Bool                            # Share λ across competing hazards
    share_covariate_lambda::Union{Bool, Symbol}   # false, :hazard, or :global
end
```

**`PenaltyConfig`** ([infrastructure.jl](../src/types/infrastructure.jl#L474) — Internal):
```julia
struct PenaltyConfig
    terms::Vector{PenaltyTerm}                    # Baseline hazard penalties
    total_hazard_terms::Vector{TotalHazardPenaltyTerm}
    smooth_covariate_terms::Vector{SmoothCovariatePenaltyTerm}
    shared_lambda_groups::Dict{Int, Vector{Int}} # Origin → term indices
    shared_smooth_groups::Vector{Vector{Int}}    # Groups sharing λ
    n_lambda::Int                                 # Total smoothing parameters
end
```

#### Key Functions

**`build_penalty_config`** ([penalty_config.jl](../src/utilities/penalty_config.jl)):
```julia
build_penalty_config(model, penalties; lambda_init=1.0, include_smooth_covariates=true)
```
Resolves `SplinePenalty` rules into `PenaltyConfig` by:
1. Finding all spline hazards
2. Applying rules by specificity (transition > origin > global)
3. Validating shared knots for competing risks
4. Building penalty terms with parameter index ranges

**`compute_penalty`** ([infrastructure.jl](../src/types/infrastructure.jl#L542)):
```julia
compute_penalty(beta::AbstractVector, config::PenaltyConfig) -> Float64
```
Evaluates: $(1/2) \sum_j \lambda_j \beta_j^\top S_j \beta_j$

#### Usage Example
```julia
# Simple: curvature penalty on all splines
config = build_penalty_config(model, SplinePenalty())

# Complex: different settings per origin
config = build_penalty_config(model, [
    SplinePenalty(1; share_lambda=true, total_hazard=true),  # Origin 1
    SplinePenalty(2; order=1),                                # Origin 2
    SplinePenalty((1,2); share_covariate_lambda=:hazard)     # Transition 1→2
])
```

### 2.3 Smoothing Parameter Selection (PIJCV)

#### Purpose
Automatic selection of smoothing parameters λ using cross-validation. Based on Wood (2024) "Neighbourhood Cross-Validation" (arXiv:2404.16490).

#### Key Function

**`select_smoothing_parameters`** ([smoothing_selection.jl](../src/inference/smoothing_selection.jl#L318)):
```julia
select_smoothing_parameters(model, data, penalty_config, beta_init;
                            method=:pijcv,     # or :gcv
                            scope=:all,        # :all, :baseline, :covariates
                            maxiter=100,
                            verbose=false)
```

**Returns**: NamedTuple with:
- `lambda`: Optimal smoothing parameters
- `beta`: Final coefficient estimate  
- `criterion`: Final PIJCV/GCV value
- `converged`: Convergence status
- `method_used`: Actual method (may fall back from PIJCV to GCV)
- `penalty_config`: Updated config with optimal λ

#### Algorithm Overview

1. Compute subject-level gradients $g_i$ and Hessians $H_i$
2. For each λ candidate:
   - Build penalized Hessian: $H_\lambda = H + \sum_j \lambda_j S_j$
   - For each subject $i$:
     - Compute leave-one-out perturbation: $\Delta^{(-i)} = H_{\lambda,-i}^{-1} g_i$
     - Evaluate prediction error at $\hat\beta - \Delta^{(-i)}$
   - Sum LOO prediction errors: $V(\lambda) = \sum_i D(y_i, \hat\theta^{(-i)})$
3. Minimize $V(\lambda)$ over log-λ using BFGS

#### Scope Filtering
The `scope` kwarg enables selective calibration:
- `:all`: Calibrate all spline penalties
- `:baseline`: Only baseline hazard splines  
- `:covariates`: Only smooth covariate terms

Internal helpers:
- `_create_scoped_penalty_config(config, scope)`: Filter config to scope
- `_merge_scoped_lambdas(...)`: Merge optimized λ back into full config

### 2.4 Smooth Covariate Terms

#### Purpose
Enable GAM-style smooth functions of covariates in hazard models:
- `s(x)`: Univariate smooth
- `te(x, y)`: Tensor product smooth (bivariate)

#### Types

**`SmoothTerm <: AbstractTerm`** ([smooth_terms.jl](../src/hazard/smooth_terms.jl#L46)):
```julia
struct SmoothTerm{T, B, S_mat} <: AbstractTerm
    term::T           # Underlying ContinuousTerm
    basis::B          # BSplineBasis
    S::S_mat          # Penalty matrix
    knots::Int        # Number of basis functions (k)
    order::Int        # Spline order (default 4 = cubic)
    penalty_order::Int # Derivative order for penalty (m)
    label::String     # e.g., "s(age)"
end
```

**`TensorProductTerm <: AbstractTerm`** ([smooth_terms.jl](../src/hazard/smooth_terms.jl#L56)):
```julia
struct TensorProductTerm{Tx, Ty, Bx, By, S_mat} <: AbstractTerm
    term_x::Tx, term_y::Ty   # Marginal ContinuousTerms
    basis_x::Bx, basis_y::By # Marginal B-spline bases
    S::S_mat                  # Tensor penalty matrix (kx*ky × kx*ky)
    kx::Int, ky::Int         # Basis dimensions
    order::Int, penalty_order::Int
    label::String
end
```

**`SmoothTermInfo`** ([hazard_structs.jl](../src/types/hazard_structs.jl#L27)):
```julia
struct SmoothTermInfo
    par_indices::Vector{Int}  # Indices in hazard's parameter vector
    S::Matrix{Float64}        # Penalty matrix
    label::String
end
```

#### StatsModels Integration

The implementation extends StatsModels.jl via:
- `apply_schema(::FunctionTerm{typeof(s)}, sch, Mod)`: Parse `s(x, k, m)` syntax
- `apply_schema(::FunctionTerm{typeof(te)}, sch, Mod)`: Parse `te(x, y, k, m)` syntax  
- `coefnames(::SmoothTerm)`: Generate coefficient names like `s(age)_1`, `s(age)_2`, ...
- `modelcols(::SmoothTerm, d)`: Evaluate B-spline basis at data points
- `width(::SmoothTerm)`: Return number of coefficients

#### Basis Expansion

Smooth terms require pre-expanding basis columns into the data before likelihood computation:

```julia
expand_smooth_term_columns!(data, hazard)  # Single hazard
expand_all_smooth_terms!(data, hazards)    # All hazards
```

This is called automatically during model construction ([multistatemodel.jl](../src/construction/multistatemodel.jl#L295)).

#### Usage Examples
```julia
# Smooth effect of age
h12 = Hazard(@formula(0 ~ s(age, 10, 2)), "wei", 1, 2)

# Tensor product: smooth interaction of age and time
h12 = Hazard(@formula(0 ~ te(age, bmi, 5, 5, 2)), "exp", 1, 2)

# Mixed: smooth + linear terms
h12 = Hazard(@formula(0 ~ s(age) + treatment), "gom", 1, 2)
```

### 2.5 Lambda Sharing Options

#### Purpose
Control how smoothing parameters are shared across smooth terms, enabling efficient estimation with reduced degrees of freedom.

#### Options

| Value | Description |
|-------|-------------|
| `false` (default) | Each smooth term gets its own λ |
| `:hazard` | All smooth terms within each hazard share one λ |
| `:global` | All smooth terms in the model share one λ |

#### Implementation

Set via `SplinePenalty`:
```julia
SplinePenalty(share_covariate_lambda=:global)
```

In `build_penalty_config`, sharing is handled by `_build_smooth_covariate_penalty_terms`:
- Groups term indices based on sharing mode
- Stores groups in `config.shared_smooth_groups`
- Adjusts `n_lambda` accordingly

In `compute_penalty` and `select_smoothing_parameters`, shared groups use the same λ value.

---

## 3. Mathematical Framework

### 3.1 B-Spline Basis Representation

The hazard function is represented as:
$$h(t) = \sum_{i=1}^{K} \beta_i B_i(t)$$

where:
- $B_i(t)$ are B-spline basis functions of order $m_1$ (default 4 = cubic)
- $\beta_i > 0$ are coefficients stored as $\gamma_i = \log(\beta_i)$ for optimization
- $K$ = number of basis functions = number of interior knots + order

### 3.2 Penalty Matrix Construction

The derivative-based penalty matrix ([spline_utils.jl](../src/utilities/spline_utils.jl#L22)):
$$S_{ij} = \int B_i^{(m)}(t) B_j^{(m)}(t) \, dt$$

where $m$ is the penalty order (default 2 = curvature penalty).

**Properties of S**:
- Symmetric: $S = S^\top$
- Positive semi-definite: $x^\top S x \geq 0$
- Null space: polynomials of degree < $m$ (dimension = $m$)
- Banded: bandwidth = $2(m_1 - 1) + 1$ for B-splines

Computed using Gauss-Legendre quadrature on each knot interval.

### 3.3 Penalized Likelihood

The penalized log-likelihood:
$$\ell_p(\beta; \lambda) = \ell(\beta) - \frac{1}{2} \sum_j \lambda_j \beta_j^\top S_j \beta_j$$

For spline hazards, $\beta$ is on natural scale (positive). The penalty contribution is:
$$P(\beta; \lambda) = \frac{1}{2} \sum_j \lambda_j \beta_j^\top S_j \beta_j$$

### 3.4 Total Hazard Penalty (Competing Risks)

For competing hazards from origin $r$, the total hazard is:
$$H_r(t) = \sum_{s \neq r} h_{rs}(t)$$

The total hazard penalty penalizes the curvature of this sum:
$$P_H = \frac{\lambda_H}{2} \left(\sum_s \beta_{rs}\right)^\top S \left(\sum_s \beta_{rs}\right)$$

This requires shared knots across competing hazards (validated by `validate_shared_knots`).

### 3.5 Tensor Product Penalty

For `te(x, y)`, the penalty uses the isotropic sum:
$$S_{te} = S_x \otimes I_y + I_x \otimes S_y$$

where $S_x$, $S_y$ are marginal penalty matrices and $\otimes$ is the Kronecker product.

This penalizes roughness equally in both dimensions. Built by `build_tensor_penalty_matrix` in [spline_utils.jl](../src/utilities/spline_utils.jl#L159).

---

## 4. Integration Points

### 4.1 Model Fitting

**`fit()`**: Penalty passed via `penalty` kwarg:
```julia
fitted = fit(model; 
    penalty=SplinePenalty(),
    penalty_lambda_selection=true,  # Auto-select λ via PIJCV
    verbose=true)
```

Internal flow:
1. `build_penalty_config(model, penalty)` → `PenaltyConfig`
2. If `penalty_lambda_selection`: `select_smoothing_parameters(...)` → optimal λ
3. Pass config to `loglik_exact_penalized` for optimization

### 4.2 Likelihood Evaluation

**`loglik_exact_penalized`** ([loglik_exact.jl](../src/likelihood/loglik_exact.jl#L293)):
```julia
loglik_exact_penalized(parameters, data, penalty_config; neg=true, parallel=false)
```

Flow:
1. Compute base log-likelihood via `loglik_exact`
2. Transform parameters to natural scale
3. Compute `compute_penalty(beta_natural, config)`
4. Return `nll + penalty`

### 4.3 MCEM

For panel data, `fit_mcem!` incorporates penalties in the M-step optimization using the same `loglik_exact_penalized` pathway.

### 4.4 Simulation

Spline hazards are evaluated during simulation via their generated `hazard_fn` and `cumhaz_fn` closures. The RuntimeGeneratedFunctions ensure efficient evaluation without allocations in the hot path.

---

## 5. Test Coverage

### Test File
[MultistateModelsTests/unit/test_splines.jl](../MultistateModelsTests/unit/test_splines.jl) — 1,224 lines

### Test Structure

**Core Verification** (Lines 80-298):
- Cumulative hazard vs QuadGK numerical integration
- Cumulative hazard with covariates
- PH covariate effect verification  
- Survival probability correctness
- Cumulative hazard additivity

**Spline Infrastructure** (Lines 299-610):
- Automatic knot placement
- `default_nknots` function
- Time transform parity verification
- `rectify_coefs!` functionality
- Round-trip coefficient transforms
- Edge cases

**Spline Knot Calibration** (Lines 611-785):
- `calibrate_splines` basic functionality
- Explicit nknots/quantiles
- Error handling
- In-place modification
- Parameter structure integrity

**Smooth Covariate Terms `s(x)`** (Lines 786-944):
- Formula parsing
- Penalty matrix properties (symmetry, PSD, null space)
- Basis evaluation (`modelcols`)
- Column expansion
- Model creation with smooth terms
- `smooth_info` extraction
- Penalty config building
- Mixed `s(x) + linear` terms
- Multiple `s(x)` terms

**Tensor Product Smooths `te(x,y)`** (Lines 946-1086):
- Formula parsing (same k, different k)
- Coefficient names
- Penalty matrix properties
- Kronecker product `modelcols`
- Column expansion
- Model creation
- Penalty config
- `build_tensor_penalty_matrix`

**Lambda Sharing** (Lines 1087-1136):
- Default: separate lambda per term
- `share_covariate_lambda=:global`
- `share_covariate_lambda=:hazard`

**Combined Tests** (Lines 1137-1224):
- `s(x) + te(x,y)` model creation
- Mixed with linear terms
- Combined penalty config
- Spline baseline + smooth covariates

---

## 6. Next Steps

### 6.1 End-to-End Validation

**Priority: High**

Create long-running tests (longtests) that:

1. **Smooth covariate recovery** (`s(x)`):
   - Simulate data with known smooth effect $f(x)$
   - Fit model with `s(x)` term
   - Verify estimated $\hat{f}(x)$ matches true $f(x)$ within tolerance

2. **Tensor product recovery** (`te(x,y)`):
   - Simulate data with 2D smooth surface
   - Fit model with `te(x,y)` term
   - Verify surface recovery

3. **Combined baseline + covariate**:
   - Simulate with spline baseline AND smooth covariate effect
   - Verify both are recovered correctly

**Implementation**: Add to `MultistateModelsTests/longtests/`

### 6.2 Documentation

**Priority: High**

1. **User-facing tutorial**: `docs/src/smooth_covariates.md`
   - Introduction to smooth covariate effects
   - `s(x)` syntax and options
   - `te(x,y)` for interactions
   - Penalty specification
   - Lambda selection

2. **Update optimization docs**: `docs/src/optimization.md`
   - Add `penalty` kwarg documentation
   - Lambda selection options
   - Performance considerations

3. **Docstring review**: Ensure all exported functions have:
   - Description
   - Arguments section
   - Returns section
   - Example (preferably `jldoctest`)

### 6.3 Performance Optimization

**Priority: Medium**

1. **Profile smooth term evaluation**:
   - Identify hot spots in likelihood computation
   - Consider caching basis evaluations for repeated covariate values

2. **Benchmark penalty computation**:
   - Measure overhead of `compute_penalty` call
   - Consider sparse matrix storage for large penalty matrices

3. **Parallel smooth column expansion**:
   - `expand_all_smooth_terms!` could parallelize over hazards

### 6.4 Edge Cases & Robustness

**Priority: Medium**

1. **Data outside knot range**:
   - Add warning when covariate values exceed basis support
   - Consider extrapolation options

2. **Rank-deficient penalties**:
   - Graceful handling when penalty matrix is singular
   - Add regularization fallback

3. **Incompatible settings**:
   - Error for `share_lambda=true` with different knot counts
   - Warning for very high/low λ values

---

## 7. File Inventory

### Source Files (Modified/Created)

| File | Lines | Description |
|------|-------|-------------|
| `src/hazard/spline.jl` | 1,074 | Spline hazard types, knot placement, calibration |
| `src/hazard/smooth_terms.jl` | 464 | `SmoothTerm`, `TensorProductTerm`, StatsModels integration |
| `src/utilities/spline_utils.jl` | 188 | `build_penalty_matrix`, `build_tensor_penalty_matrix` |
| `src/inference/smoothing_selection.jl` | 661 | PIJCV, GCV, `select_smoothing_parameters` |
| `src/utilities/penalty_config.jl` | 411 | `build_penalty_config`, penalty term builders |
| `src/types/infrastructure.jl` | 570 | `SplinePenalty`, `PenaltyConfig`, `compute_penalty` |
| `src/types/hazard_structs.jl` | 276 | `SmoothTermInfo`, hazard struct definitions |
| `src/construction/multistatemodel.jl` | 1,312 | `_extract_smooth_info`, smooth term discovery |

### Test Files

| File | Lines | Description |
|------|-------|-------------|
| `MultistateModelsTests/unit/test_splines.jl` | 1,224 | Comprehensive unit tests |

---

## 8. Known Limitations

1. **Natural boundary conditions**: Incompatible with `monotone != 0` splines. Natural splines enforce $h''(t) = 0$ at boundaries, which conflicts with monotonicity constraints.

2. **Isotropic tensor penalty**: `te(x,y)` uses equal smoothing in both dimensions. Anisotropic smoothing (different λ per dimension) is not yet supported.

3. **PIJCV sample size**: Requires at least ~20 subjects for reliable λ selection. With fewer subjects, consider using fixed λ or GCV fallback.

4. **Extrapolation**: Spline hazards extrapolate according to `extrapolation` option ("constant" or "linear"). Covariate values outside the basis support may produce warnings.

5. **Memory**: Large tensor products (`te(x,y)` with $k_x \cdot k_y > 100$) create dense penalty matrices. Consider smaller bases for high-dimensional problems.

---

## 9. Important Implementation Details

### Parameter Storage Convention

Spline coefficients are stored on **log scale** (γ) for optimization:
- Estimation scale: `γ = [log(β₁), log(β₂), ..., log(βₖ)]`
- Natural scale: `β = exp.(γ)` (positive coefficients)

The **penalty applies to natural scale**: $\lambda \beta^\top S \beta$

This is handled in `loglik_exact_penalized` via `unflatten_natural(parameters, model)`.

### Smooth Term Discovery

The `_extract_smooth_info()` function in [multistatemodel.jl](../src/construction/multistatemodel.jl#L335) walks the formula schema tree to find `SmoothTerm` and `TensorProductTerm` nodes:

```julia
function _extract_smooth_info(ctx::HazardBuildContext, parnames::Vector{Symbol})
    smooth_info = SmoothTermInfo[]
    rhs = ctx.hazschema.rhs
    terms = rhs isa StatsModels.MatrixTerm ? rhs.terms : (rhs,)
    
    for term in terms
        if term isa SmoothTerm
            # Extract coefficient indices and penalty matrix
            ...
            push!(smooth_info, SmoothTermInfo(indices, term.S, term.label))
        elseif term isa TensorProductTerm
            # Same for tensor products
            ...
        end
    end
    return smooth_info
end
```

### Column Expansion

Smooth terms require basis columns pre-expanded into the DataFrame before likelihood computation. This is done by `expand_all_smooth_terms!()` during model construction, which calls `expand_smooth_term_columns!()` for each hazard.

The expanded columns have names like `s(age)_1`, `s(age)_2`, ... which are then treated as regular covariates in the likelihood code.

### PIJCV Scope Filtering

When `scope != :all`, the smoothing selection creates a filtered penalty config:

1. `_create_scoped_penalty_config(config, scope)`:
   - Returns `(scoped_config, fixed_baseline_lambdas, fixed_covariate_lambdas)`
   - Scoped config contains only terms within scope
   - Fixed lambdas preserve values from excluded terms

2. After optimization, `_merge_scoped_lambdas(...)`:
   - Combines optimized lambdas with fixed lambdas
   - Reconstructs full PenaltyConfig with updated values

### Tensor Product Penalty Construction

`build_tensor_penalty_matrix(Sx, Sy)` implements:
$$S_{te} = S_x \otimes I_y + I_x \otimes S_y$$

This isotropic penalty smooths equally in both directions. The Kronecker product ordering matches the row-wise Kronecker product used in `modelcols` for `TensorProductTerm`.

---

## Appendix: References

- Wood, S.N. (2016). "P-splines with derivative based penalties and tensor product smoothing of unevenly distributed data." *Statistics and Computing*.
- Wood, S.N. (2024). "Neighbourhood Cross-Validation." *arXiv:2404.16490*.
- Jackson, C.H. (2016). "flexsurv: A platform for parametric survival modeling in R." *Journal of Statistical Software*.
- Jackson, C.H. et al. (2022). "survextrap: A package for flexible and transparent extrapolation of survival curves." *arXiv*.
