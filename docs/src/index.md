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
h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
h21 = Hazard(@formula(0 ~ 1), "wei", 2, 1)

# Create and fit model
model = multistatemodel(h12, h21; data=data)
fitted = fit(model)
```

## Hazard Families

### Parametric Hazards

- `"exp"` - Exponential (constant hazard)
- `"wei"` - Weibull
- `"gom"` - Gompertz

### Spline Hazards

Spline hazards (`"sp"`) provide flexible baseline hazard modeling using B-splines:

```julia
# Cubic natural spline with user-specified knots
h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2;
             degree=3,
             knots=[0.5, 1.0, 1.5],
             natural_spline=true)

# Automatic knot placement based on data
h21 = Hazard(@formula(0 ~ 1 + x), "sp", 2, 1;
             degree=3,
             knots=nothing)  # Knots placed at sojourn quantiles

# Monotone increasing hazard (for disease progression)
h13 = Hazard(@formula(0 ~ 1), "sp", 1, 3;
             degree=3,
             knots=[0.5, 1.0],
             monotone=1)

# Monotone decreasing hazard (for recovery)
h31 = Hazard(@formula(0 ~ 1), "sp", 3, 1;
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
| `extrapolation` | `"flat"` or `"linear"` beyond knots | `"flat"` |

#### Automatic Knot Placement

When `knots=nothing`, interior knots are automatically placed at quantiles of sojourn times for the relevant transition. The number of knots follows the Tang et al. formula: `floor(n^(1/5))` where `n` is the number of observed transitions.

## Time Transformation Caching

For improved computational efficiency during likelihood evaluation, spline hazards support time transformation caching:

```julia
h12 = Hazard(@formula(0 ~ 1 + x), "sp", 1, 2;
             degree=3,
             knots=[0.5, 1.0],
             time_transform=true)  # Enable caching
```

This caches the h(t)·exp(-H(t)) terms for faster repeated evaluations.

## Covariates

Covariates are specified using `@formula` syntax:

```julia
# Proportional hazards (default)
h12 = Hazard(@formula(0 ~ 1 + age + sex), "wei", 1, 2)

# Accelerated failure time
h21 = Hazard(@formula(0 ~ 1 + treatment), "wei", 2, 1;
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