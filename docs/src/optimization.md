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
