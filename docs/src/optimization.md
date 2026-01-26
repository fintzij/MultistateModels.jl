# Optimization and Variance Estimation

## Optimization Solvers

MultistateModels.jl uses [Optimization.jl](https://github.com/SciML/Optimization.jl) as a 
unified interface to various optimization backends. By default:

- **Unconstrained problems**: L-BFGS via [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl)
  (5-6× faster than interior-point methods)
- **Constrained problems**: Interior-point via [Ipopt.jl](https://github.com/jump-dev/Ipopt.jl)

### Available Solvers

You can pass any Optimization.jl-compatible solver to `fit()` via the `solver` keyword:

```julia
using Optim, OptimizationIpopt

# Unconstrained solvers (Optim.jl)
fit(model; solver=Optim.LBFGS())      # Default, quasi-Newton (recommended)
fit(model; solver=Optim.BFGS())       # Full BFGS, uses more memory
fit(model; solver=Optim.NelderMead()) # Derivative-free, slower but robust
fit(model; solver=Optim.Newton())     # Newton's method, requires Hessian

# Constrained solvers
fit(model; solver=IpoptOptimizer(), constraints=cons)  # Default for constrained
```

### Solver Selection Guidelines

| Scenario | Recommended Solver | Notes |
|----------|-------------------|-------|
| Small model, no constraints | `Optim.LBFGS()` | Fast, low memory |
| Large model, no constraints | `Optim.LBFGS()` | Scales well |
| Any model with constraints | `IpoptOptimizer()` | Interior-point, handles inequality constraints |
| Convergence issues | `Optim.NelderMead()` | Derivative-free, more robust but slower |
| High precision needed | `Optim.Newton()` | Quadratic convergence near optimum |

### MCEM-Specific Considerations

For semi-Markov models fit via MCEM, the solver is used in each M-step. The default L-BFGS 
is typically sufficient since:
1. Each M-step starts near the optimum (warm start from previous iteration)
2. The objective changes slightly between iterations
3. Fast M-steps allow more MCEM iterations for the same compute budget

## Variance Estimation

MultistateModels.jl provides three variance estimators, controlled by the `vcov_type` keyword argument to `fit()`:

```julia
# IJ/Sandwich variance (DEFAULT - recommended for inference)
fitted = fit(model; vcov_type=:ij)

# Model-based variance (inverse Hessian)
fitted = fit(model; vcov_type=:model)

# Jackknife variance
fitted = fit(model; vcov_type=:jk)

# Skip variance computation
fitted = fit(model; vcov_type=:none)
```

### Model-Based Variance (`vcov_type=:model`)

The standard MLE variance estimator:
```math
\text{Var}(\hat{\theta}) = H^{-1} = \left(-\nabla^2 \ell(\hat{\theta})\right)^{-1}
```

- Valid only under correct model specification
- Computed via ForwardDiff automatic differentiation
- For semi-Markov models, uses Louis's identity to account for missing data

### Infinitesimal Jackknife / Sandwich Variance (`vcov_type=:ij`)

**Recommended for inference (DEFAULT).** The robust sandwich estimator:
```math
\text{Var}_{IJ}(\hat{\theta}) = H^{-1} K H^{-1}, \quad K = \sum_i g_i g_i^\top
```

where ``g_i = \nabla \ell_i(\hat{\theta})`` is the score contribution from subject ``i``.

- Remains valid under model misspecification
- Compare to model-based variance to diagnose misspecification:
  if ``\text{SE}_{IJ} \gg \text{SE}_{model}``, the model may be misspecified

### Jackknife Variance (`vcov_type=:jk`)

The finite-sample delete-one jackknife:
```math
\text{Var}_{JK}(\hat{\theta}) = \frac{n-1}{n} \sum_i \Delta_i \Delta_i^\top, \quad \Delta_i = \hat{\theta}_{-i} - \hat{\theta}
```

- More computationally expensive than IJ
- Uses one-step Newton approximation: ``\Delta_i \approx H_{-i}^{-1} g_i``
- Controlled by `loo_method` parameter (see below)

### LOO Perturbation Methods (`loo_method`)

The `loo_method` parameter controls how leave-one-out perturbations ``\Delta_i`` are computed 
for **jackknife variance only** (not IJ, since IJ uses the formula ``H^{-1}KH^{-1}`` directly).

| Method | Formula | Complexity | When to Use |
|--------|---------|------------|-------------|
| `:direct` (default) | ``\Delta_i = H^{-1} g_i`` | ``O(p^2 n)`` | Typical case (``n \gg p``) |
| `:cholesky` | Exact ``H_{-i}^{-1}`` via rank-k downdate | ``O(n p^3)`` | Ill-conditioned problems |

The `:direct` method approximates ``H_{-i}^{-1} \approx H^{-1}``, which is the infinitesimal 
limit. The `:cholesky` method computes the exact leave-one-out Hessian by updating the 
Cholesky factorization, which is more stable but computationally expensive.

### Variance with Constraints

When fitting models with constraints (e.g., phase-type SCTP or inequality constraints), 
the MLE lies on a constraint manifold. Standard variance estimation must account for 
the reduced degrees of freedom due to active constraints.

MultistateModels.jl uses the **reduced Hessian approach** for constrained variance:

```math
\text{Var}(\hat{\theta}) = Z(Z^\top H Z)^{-1} Z^\top
```

where:
- ``H = -\nabla^2 \ell(\hat{\theta})`` is the observed information (negative Hessian)
- ``J = \nabla c(\hat{\theta})`` is the Jacobian of active constraints at the MLE
- ``Z`` is an orthonormal basis for ``\text{null}(J)`` (feasible directions)

**Key properties:**
1. **Correctly reduces degrees of freedom**: If ``k`` constraints are active, variance lives in a ``(p-k)``-dimensional subspace
2. **Zero variance in constrained directions**: Parameters fixed by constraints have zero variance
3. **Reduces to standard ``H^{-1}`` when no constraints are active**

**Active constraint identification:**
A constraint ``l_i \le c_i(\theta) \le u_i`` is **active** at the MLE if:
- It's an equality constraint (``l_i = u_i``), or
- The constraint value is at its bound (``c(\hat{\theta}) \approx l_i`` or ``c(\hat{\theta}) \approx u_i``)

```julia
# Example: Phase-type model with SCTP constraints
h12 = Hazard(@formula(0 ~ 1), "pt", 1, 2)
h13 = Hazard(@formula(0 ~ 1), "pt", 1, 3)

model = multistatemodel(h12, h13; data=df, n_phases=Dict(1 => 3), coxian_structure=:sctp)

# fit() automatically computes constrained variance
fitted = fit(model)

# vcov is now available (previously was `nothing` for constrained models)
@assert !isnothing(fitted.vcov)

# Variance in constrained directions will be near-zero
# Free directions will have positive variance
se = sqrt.(diag(fitted.vcov))
```

**Warning messages:**
- If parameters are at their box bounds (not constraint bounds), you'll see a warning:
  "Parameters at bounds may have unreliable variance estimates"
- Consider using the IJ (sandwich) variance which is more robust to boundary effects

## Penalized Spline Smoothing

When fitting spline hazards (`:sp`), MultistateModels.jl supports **penalized maximum likelihood** estimation with automatic smoothing parameter selection. This prevents overfitting while preserving flexibility.

### Penalty Specification

The `penalty` argument to `fit()` controls spline penalization:

```julia
# Default: automatic penalty for spline hazards
fitted = fit(model)  # penalty=:auto

# Explicit penalty (same as default for splines)
fitted = fit(model; penalty=SplinePenalty())

# Disable penalization
fitted = fit(model; penalty=:none)
```

#### SplinePenalty Options

```julia
SplinePenalty(selector=:all;
              order=2,              # Derivative order (2 = curvature penalty)
              total_hazard=false,   # Penalize total hazard from each state
              share_lambda=false,   # Share λ across hazards
              share_covariate_lambda=false)  # Share λ for smooth covariates
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `selector` | `:all`, origin state `Int`, or transition `(from, to)` | `:all` |
| `order` | Derivative order to penalize (1=roughness, 2=curvature) | `2` |
| `total_hazard` | Also penalize sum of competing hazards | `false` |
| `share_lambda` | Use single λ across all selected hazards | `false` |
| `share_covariate_lambda` | `false`, `:hazard`, or `:global` | `false` |

### Smoothing Parameter Selection Methods

The `select_lambda` argument controls how the smoothing parameter λ is chosen:

```julia
# Default: PIJCV (fast, accurate)
fitted = fit(model; select_lambda=:pijcv)

# Alternative methods
fitted = fit(model; select_lambda=:efs)    # Expected Fisher scoring
fitted = fit(model; select_lambda=:perf)   # Performance iteration
fitted = fit(model; select_lambda=:loocv)  # Exact leave-one-out CV (slow)
fitted = fit(model; select_lambda=:cv5)    # 5-fold cross-validation
```

| Method | Description | Complexity | Recommended For |
|--------|-------------|------------|-----------------|
| `:pijcv` | Predictive infinitesimal jackknife CV (Wood 2024) | O(np²) | Default choice |
| `:pijcv5`, `:pijcv10`, `:pijcv20` | K-fold PIJCV variants | O(np²) | Large n |
| `:efs` | Expected Fisher scoring criterion | O(np²) | Fast alternative |
| `:perf` | Performance iteration (Wood & Fasiolo 2017) | O(np²) | GCV-like behavior |
| `:loocv` | Exact leave-one-out cross-validation | O(n²p²) | Gold standard (slow) |
| `:cv5`, `:cv10`, `:cv20` | K-fold CV with refitting | O(knp²) | Model checking |

**Reference**: Wood, S.N. (2024). "On Neighbourhood Cross Validation." arXiv:2404.16490

### Initial Lambda Value

The `lambda_init` parameter sets the starting value for λ optimization:

```julia
# Default
fitted = fit(model; lambda_init=1.0)

# Start with less smoothing
fitted = fit(model; lambda_init=0.1)

# Start with more smoothing
fitted = fit(model; lambda_init=10.0)
```

### Effective Degrees of Freedom

After fitting, the effective degrees of freedom (EDF) can be retrieved:

```julia
fitted = fit(model)

# EDF per hazard (named tuple)
edf = fitted.edf  # e.g., (h12 = 4.2, h21 = 3.8)

# Total EDF
total_edf = sum(values(fitted.edf))
```

Lower EDF indicates more smoothing (simpler model). When EDF ≈ number of spline coefficients, the penalty has little effect.

### Example: Illness-Death Model with Spline Hazards

```julia
using MultistateModels, DataFrames

# Define spline hazards with automatic knot placement
h12 = Hazard(@formula(0 ~ 1 + age), :sp, 1, 2; degree=3, knots=nothing)
h13 = Hazard(@formula(0 ~ 1 + age), :sp, 1, 3; degree=3, knots=nothing)
h23 = Hazard(@formula(0 ~ 1), :sp, 2, 3; degree=3, knots=nothing, monotone=1)

# Create model
model = multistatemodel(h12, h13, h23; data=df)

# Fit with automatic smoothing selection
fitted = fit(model)  # Uses penalty=:auto, select_lambda=:pijcv

# Check smoothing parameters and EDF
println("Smoothing parameters: ", fitted.smoothing_parameters)
println("Effective degrees of freedom: ", fitted.edf)

# Get parameter estimates with robust standard errors
params = get_parameters(fitted)
se_robust = sqrt.(diag(get_ij_vcov(fitted)))
```

### Variance Estimation Recommendations

```julia
# Default: IJ/sandwich variance (recommended for inference)
fitted = fit(model)  # vcov_type=:ij is the default

# Get robust standard errors
using LinearAlgebra
robust_se = sqrt.(diag(fitted.vcov))

# Compare IJ vs model-based to check specification
fitted_model = fit(model; vcov_type=:model)
# Large ratios (SE_IJ / SE_model >> 1) suggest misspecification

# Speed-focused: skip variance estimation
fitted = fit(model; vcov_type=:none)

# Research/diagnostics: include jackknife
fitted = fit(model; vcov_type=:jk)
```

## Pseudo-Inverse Thresholding (`vcov_threshold`)

The Fisher information matrix may be near-singular, especially with:
- Overparameterized models
- Sparse data for some transitions
- Parameters near boundaries

The `vcov_threshold` parameter controls the pseudo-inverse tolerance:
- `vcov_threshold=true` (default): adaptive threshold ``1/(\log n \cdot p)^2``
- `vcov_threshold=false`: machine epsilon ``\sqrt{\epsilon}``

The adaptive threshold is more aggressive at zeroing small eigenvalues, which can 
stabilize variance estimates when the information matrix is ill-conditioned.
