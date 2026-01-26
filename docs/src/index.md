# MultistateModels.jl

`MultistateModels.jl` is a Julia package for simulating and fitting multistate models, especially semi-Markov models, to panel data and interval censored data.

## Features

- **Parametric hazard families**: Exponential, Weibull, Gompertz
- **Spline hazards**: Flexible B-spline baseline hazards with optional monotonicity constraints
- **Covariates**: Time-fixed covariates with proportional hazards (PH) or accelerated failure time (AFT) effects
- **Interval censoring**: Full support for panel data with interval-censored transition times
- **Monte Carlo EM**: Maximum likelihood estimation via MCEM for complex observation patterns

## Installation

```julia
using Pkg
Pkg.add("MultistateModels")
```

## Quick Start

```julia
using MultistateModels
using DataFrames

# Create panel data
data = DataFrame(
    id = [1, 1, 2, 2],
    tstart = [0.0, 1.0, 0.0, 0.5],
    tstop = [1.0, 2.0, 0.5, 1.5],
    statefrom = [1, 2, 1, 2],
    stateto = [2, 1, 2, 1],
    obstype = [1, 1, 1, 1]
)

# Define hazards
h12 = Hazard(@formula(0 ~ 1), :wei, 1, 2)
h21 = Hazard(@formula(0 ~ 1), :wei, 2, 1)

# Create and fit model
model = multistatemodel(h12, h21; data=data)
fitted = fit(model)
```

## Hazard Families

### Parametric Hazards

All parametric hazards follow the **flexsurv** R package parameterizations.

#### Exponential (`:exp`)

Constant hazard rate:
- **Hazard**: h(t) = rate
- **Cumulative hazard**: H(t) = rate × t
- **Parameters**: rate > 0

```julia
h12 = Hazard(@formula(0 ~ 1), :exp, 1, 2)
```

#### Weibull (`:wei`)

Flexible shape for increasing/decreasing hazards:
- **Hazard**: h(t) = shape × scale × t^(shape-1)
- **Cumulative hazard**: H(t) = scale × t^shape
- **Parameters**: shape > 0, scale > 0
  - shape < 1: decreasing hazard (infant mortality)
  - shape = 1: constant hazard (reduces to exponential)
  - shape > 1: increasing hazard (wear-out)

```julia
h12 = Hazard(@formula(0 ~ 1), :wei, 1, 2)
```

#### Gompertz (`:gom`)

Exponentially changing hazard, commonly used for mortality modeling:
- **Hazard**: h(t) = rate × exp(shape × t)
- **Cumulative hazard**: H(t) = (rate/shape) × (exp(shape×t) - 1) for shape ≠ 0
- **Parameters**: 
  - shape ∈ ℝ (unconstrained)
    - shape > 0: exponentially increasing hazard (typical aging)
    - shape = 0: constant hazard (reduces to exponential)
    - shape < 0: exponentially decreasing hazard (cure/defective models)
  - rate > 0: baseline hazard at t=0

```julia
h12 = Hazard(@formula(0 ~ 1), :gom, 1, 2)
```

**Note**: The Gompertz `shape` parameter is unconstrained (can be negative, zero, or positive), while `rate` must be positive. This matches the flexsurv parameterization.

### Spline Hazards

Spline hazards (`:sp`) provide flexible baseline hazard modeling using B-splines:

```julia
# Cubic natural spline with user-specified knots
h12 = Hazard(@formula(0 ~ 1), :sp, 1, 2;
             degree=3,
             knots=[0.5, 1.0, 1.5],
             natural_spline=true)

# Automatic knot placement based on data
h21 = Hazard(@formula(0 ~ 1 + x), :sp, 2, 1;
             degree=3,
             knots=nothing)  # Knots placed at sojourn quantiles

# Monotone increasing hazard (for disease progression)
h13 = Hazard(@formula(0 ~ 1), :sp, 1, 3;
             degree=3,
             knots=[0.5, 1.0],
             monotone=1)

# Monotone decreasing hazard (for recovery)
h31 = Hazard(@formula(0 ~ 1), :sp, 3, 1;
             degree=3,
             knots=[0.5, 1.0],
             monotone=-1)
```

#### Spline Options

| Option | Description | Default |
|--------|-------------|---------|
| `degree` | Polynomial degree (1=linear, 3=cubic) | 3 |
| `knots` | Interior knot locations, or `nothing` for auto | Required |
| `natural_spline` | Natural boundary conditions | `false` |
| `monotone` | 0=none, 1=increasing, -1=decreasing | 0 |
| `extrapolation` | `"constant"` (smooth) or `"linear"` beyond knots | `"constant"` |

#### Automatic Knot Placement

When `knots=nothing`, interior knots are automatically placed at quantiles of sojourn times for the relevant transition. The number of knots follows the Tang et al. formula: `floor(n^(1/5))` where `n` is the number of observed transitions.

## Time Transformation Caching

For improved computational efficiency during likelihood evaluation, spline hazards support time transformation caching:

```julia
h12 = Hazard(@formula(0 ~ 1 + x), :sp, 1, 2;
             degree=3,
             knots=[0.5, 1.0],
             time_transform=true)  # Enable caching
```

This caches the h(t)·exp(-H(t)) terms for faster repeated evaluations.

## Covariates

Covariates are specified using `@formula` syntax:

```julia
# Proportional hazards (default)
h12 = Hazard(@formula(0 ~ 1 + age + sex), :wei, 1, 2)

# Accelerated failure time
h21 = Hazard(@formula(0 ~ 1 + treatment), :wei, 2, 1;
             linpred_effect=:aft)
```

## Model Fitting

```julia
# Basic fit
fitted = fit(model)

# Get parameter estimates
get_parameters(fitted)

# Get variance-covariance matrix
get_vcov(fitted)

# Log-likelihood and information criteria
get_loglik(fitted)
aic(fitted)
bic(fitted)
```

## Simulation

```julia
# Simulate paths from fitted model
paths = simulate(fitted, n=100)

# Simulate panel data
simdata = simulate_data(fitted, n=100, times=[0, 1, 2, 3])
```

### Simulation Strategies

For models with `time_transform=true` hazards, you can choose between two simulation strategies:

```julia
using MultistateModels: CachedTransformStrategy, DirectTransformStrategy

# Cached strategy (default): precomputes and caches hazard values
# - Best for repeated simulations with the same parameters
# - Uses more memory but faster for multiple paths
path = simulate_path(model, subject_id; strategy=CachedTransformStrategy())

# Direct strategy: computes hazard values on-demand
# - Lower memory usage
# - Better for one-off simulations or when parameters change frequently
path = simulate_path(model, subject_id; strategy=DirectTransformStrategy())
```

Both strategies produce identical results given the same random seed - the choice affects only computational performance.

## Parameter Scale Convention

**All parameters are stored on natural scale** (v0.3.0+). Positivity constraints (e.g., for rate parameters) are enforced via box constraints during optimization, not log-transformation.

```julia
# Get parameters (always natural scale)
params = get_parameters(fitted)

# Set parameters (expects natural scale)
set_parameters!(model, (h12 = [shape, scale, beta],))

# For hazard families with positivity constraints:
# - Exponential: rate > 0 (box constraint)
# - Weibull: shape > 0, scale > 0 (box constraints)
# - Gompertz: shape ∈ ℝ (unconstrained), rate > 0 (box constraint)
# - Spline: coefficients ≥ 0 for monotone hazards (box constraints)
```

## Penalized Splines

Spline hazards support automatic smoothing parameter selection via penalized maximum likelihood:

```julia
# Default: automatic penalty with PIJCV smoothing selection
fitted = fit(model)

# Explicit penalty specification
fitted = fit(model; 
             penalty=SplinePenalty(order=2),  # Curvature penalty
             lambda_init=1.0,                  # Initial smoothing parameter
             select_lambda=:pijcv)             # Selection method

# Check effective degrees of freedom
println(fitted.edf)          # Per-hazard EDF
println(fitted.smoothing_parameters)  # Optimal λ values
```

**Smoothing selection methods:**
- `:pijcv` (default): Predictive infinitesimal jackknife CV (fast, accurate)
- `:efs`: Expected Fisher scoring criterion  
- `:perf`: Performance iteration (GCV-like)
- `:loocv`: Exact leave-one-out CV (slow but exact)

### Adaptive Penalty Weighting

By default, spline penalties apply uniform smoothing across time. However, when the number of subjects at risk varies substantially over time (e.g., many subjects at early times, few at late times), it can be beneficial to penalize more heavily where less information is available.

**At-risk weighting** adapts the penalty strength based on $w(t) = Y(t)^{-\alpha}$, where $Y(t)$ is the number at risk at time $t$:
- $\alpha = 0$: Uniform weighting (standard P-spline)
- $\alpha = 1$: Penalize proportionally to $1/Y(t)$ (default when enabled)
- $\alpha > 1$: Stronger adaptation to at-risk counts

```julia
# At-risk adaptive weighting with default α=1.0
fitted = fit(model;
             penalty=SplinePenalty(adaptive_weight=:atrisk),
             select_lambda=:efs)

# Custom α value
fitted = fit(model;
             penalty=SplinePenalty(adaptive_weight=:atrisk, alpha=0.5),
             select_lambda=:efs)

# Learn α from data via marginal likelihood
fitted = fit(model;
             penalty=SplinePenalty(adaptive_weight=:atrisk, learn_alpha=true),
             select_lambda=:efs)
```

**When to use adaptive weighting:**
- Right-censored survival data with few events at late times
- Panel data with varying observation intensity across time
- Competing risks where different transitions have different at-risk patterns

**Per-transition specification:**
```julia
# Different settings for different transitions
fitted = fit(model;
             penalty=[
                 SplinePenalty((1,2), adaptive_weight=:atrisk, alpha=1.0),
                 SplinePenalty((1,3), adaptive_weight=:atrisk, alpha=0.5)
             ],
             select_lambda=:efs)
```

When `share_lambda=true` is used, grouped transitions automatically share the same α value.

See the [Optimization](optimization.md) documentation for details on penalty configuration and smoothing parameter selection.

## Warning Messages

MultistateModels.jl provides informative warnings for common issues. Most warnings are rate-limited (via `maxlog`) to avoid flooding output during long-running fits.

### Suppressing Warnings

To suppress specific warnings:

```julia
using Logging

# Suppress all warnings from MultistateModels during a fit
with_logger(SimpleLogger(stderr, Logging.Error)) do
    fitted = fit(model; verbose=false)
end

# Or filter specific warning messages
filtered_logger = ConsoleLogger(stderr, Logging.Warn; meta_formatter=(level, _module, group, id, file, line) -> "")
with_logger(filtered_logger) do
    fitted = fit(model)
end
```

### Common Warnings and Their Meaning

| Warning | Meaning | Action |
|---------|---------|--------|
| "Parameters at bounds" | MLE is at box constraint boundary | Check if bounds are appropriate; may need wider bounds |
| "No transitions from state X" | Data has no observed transitions for a hazard | Check data or simplify model |
| "Model has no absorbing states" | Potential infinite sojourn times | Verify this is intentional |
| "Variance not available" | VCOV couldn't be computed | May need more data or simpler model |
| "PSIS failed" | Importance sampling diagnostics warn | Increase ESS or simplify model |

### Debug-Level Messages

For detailed diagnostic output, enable debug logging:

```julia
using Logging
global_logger(ConsoleLogger(stderr, Logging.Debug))

fitted = fit(model)
``` 