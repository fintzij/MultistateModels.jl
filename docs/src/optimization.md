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

MultistateModels.jl provides three variance estimators, controlled by keyword arguments to `fit()`:

### Model-Based Variance (`compute_vcov=true`)

The standard MLE variance estimator:
```math
\text{Var}(\hat{\theta}) = H^{-1} = \left(-\nabla^2 \ell(\hat{\theta})\right)^{-1}
```

- Valid only under correct model specification
- Computed via ForwardDiff automatic differentiation
- For semi-Markov models, uses Louis's identity to account for missing data

### Infinitesimal Jackknife / Sandwich Variance (`compute_ij_vcov=true`)

**Recommended for inference.** The robust sandwich estimator:
```math
\text{Var}_{IJ}(\hat{\theta}) = H^{-1} K H^{-1}, \quad K = \sum_i g_i g_i^\top
```

where ``g_i = \nabla \ell_i(\hat{\theta})`` is the score contribution from subject ``i``.

- Remains valid under model misspecification
- Compare to model-based variance to diagnose misspecification:
  if ``\text{SE}_{IJ} \gg \text{SE}_{model}``, the model may be misspecified

### Jackknife Variance (`compute_jk_vcov=false`)

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

### Variance Estimation Recommendations

```julia
# Default: both model-based and IJ (recommended)
fitted = fit(model)

# Get robust standard errors for inference
using LinearAlgebra
robust_se = sqrt.(diag(get_ij_vcov(fitted)))

# Compare variances to check model specification
result = compare_variance_estimates(fitted)
# Large ratios (SE_IJ / SE_model >> 1) suggest misspecification

# Speed-focused: skip robust variance
fitted = fit(model; compute_ij_vcov=false)

# Research/diagnostics: include jackknife
fitted = fit(model; compute_jk_vcov=true)
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
