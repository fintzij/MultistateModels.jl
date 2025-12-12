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

## Parameter Scale Conventions

MultistateModels.jl stores parameters on **estimation scale** internally but provides convenience functions for working with **natural scale** parameters:

| Family | Parameter | Estimation Scale | Natural Scale |
|--------|-----------|-----------------|---------------|
| Exponential | rate | log(rate) | rate |
| Weibull | shape | log(shape) | shape |
| Weibull | scale | log(scale) | scale |
| Gompertz | shape | shape | shape (unconstrained) |
| Gompertz | rate | log(rate) | rate |
| All | covariate β | β | β |

**Note**: Gompertz `shape` is unconstrained (can be negative), so it is stored as-is without log transformation.

```julia
# Get parameters on natural scale
natural_params = get_parameters(fitted; scale=:natural)

# Get parameters on estimation (log) scale  
est_params = get_parameters(fitted; scale=:estimation)

# Set parameters (expects estimation scale)
set_parameters!(model, (h12 = [log(shape), log(scale), beta],))

# For Gompertz specifically:
set_parameters!(model, (h12 = [shape, log(rate), beta],))  # shape NOT logged
``` 