# Changelog

## [Unreleased] - SIR Branch

### Sampling Importance Resampling (SIR) for MCEM M-step

**File:** `src/sir.jl` (258 lines)

Added Sampling Importance Resampling (SIR) to accelerate the M-step optimization in MCEM. Instead of computing log-likelihoods over the full importance sampling pool (~1000 paths/subject), SIR resamples a smaller subset (~200 paths/subject) with uniform weights, providing ~5x speedup in M-step optimization with minimal loss in statistical efficiency.

**New Functions:**

```julia
# Pool size calculation
sir_pool_size(ess_target, sir_pool_constant, max_pool_size) -> Int

# Resampling methods
resample_multinomial(weights, n) -> Vector{Int}  # Standard SIR
resample_lhs(weights, n) -> Vector{Int}          # Latin Hypercube Sampling (lower variance)
get_sir_subsample_indices(weights, n, method) -> Vector{Int}  # Dispatcher

# Resampling decision
should_resample(weights, ess_target, min_ess_ratio, max_ess_ratio) -> Bool

# Data structure creation
create_sir_subsampled_data(samplepaths, sir_indices) -> (paths, weights)

# MLL/ASE computation on resampled paths
mcem_mll_sir(logliks, sir_indices, SubjectWeights) -> Float64
mcem_ase_sir(loglik_prop, loglik_cur, sir_indices, SubjectWeights) -> Float64
```

**New `fit()` Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sir` | `Symbol` | `:off` | Resampling method: `:off`, `:sir`, or `:lhs` |
| `sir_pool_constant` | `Float64` | `2.0` | Pool size = c × ESS × log(ESS) |
| `max_sir_pool_size` | `Int` | `8192` | Maximum pool size per subject |

**Performance:**

With ESS target = 200 and pool size ~1000:
- **Log-likelihood computation**: ~5x speedup (105ms → 21ms for 50 subjects)
- **Variance ratio (SIR/LHS)**: ~1.24 (LHS has ~20% lower variance)
- **Unbiasedness**: Both SIR and LHS are unbiased estimators

**Usage:**

```julia
# Standard MCEM (no resampling)
fitted = fit(model; ess_target_initial=200)

# MCEM with SIR resampling (faster M-step)
fitted = fit(model; ess_target_initial=200, sir=:sir)

# MCEM with LHS resampling (faster + lower variance)
fitted = fit(model; ess_target_initial=200, sir=:lhs)
```

### Parameter Renaming

Renamed several MCEM parameters for clarity:

| Old Name | New Name | Description |
|----------|----------|-------------|
| `ess_increase` | `ess_growth_factor` | ESS inflation factor when more paths needed |
| `caffo_c` | *(removed)* | Now uses `ess_growth_factor` for mid-iteration increase |
| `caffo_alpha` | `ascent_alpha` | Type I error rate for Caffo power calculation |
| `caffo_beta` | `ascent_beta` | Type II error rate for Caffo power calculation |
| `:caffo` | `:adaptive` | ESS increase method (Caffo et al. 2005 referenced in docstring) |

**Improved Docstrings:**

- `block_hessian_speedup`: Now explains that it's the minimum speedup factor required to use block-diagonal Hessian approximation instead of full Hessian. Higher values = more conservative (prefer full Hessian).

### New Unit Tests

**File:** `MultistateModelsTests/unit/test_mll_consistency.jl`

Comprehensive tests for MLL consistency across estimation methods:
- Sample-level and subject-level MLL agreement
- Variance comparison: Full pool IS (zero variance) vs SIR vs LHS
- Runtime benchmarks showing ~5x speedup in log-likelihood computation
- Unbiasedness verification
- Edge cases (uniform weights, skewed weights)

---

## [0.2.0] - 2025-01-12

Version 0.2.0 is a comprehensive rewrite of MultistateModels.jl with major improvements across performance, features, and code organization. The source code has grown from approximately 5,500 lines to 27,000 lines across 27 files.

---

## Table of Contents

1. [Breaking Changes](#breaking-changes)
2. [New Features](#new-features)
   - [Phase-Type Importance Sampling](#1-phase-type-importance-sampling)
   - [Robust Variance Estimation](#2-robust-variance-estimation)
   - [SQUAREM Acceleration](#3-squarem-acceleration)
   - [Spline Hazards](#4-spline-hazards-enhanced)
   - [AD Backend Selection](#5-ad-backend-selection)
   - [Simulation Infrastructure](#6-simulation-infrastructure)
   - [Performance Optimizations](#7-performance-optimizations)
3. [New Types](#new-types)
4. [New Functions](#new-functions)
5. [API Changes](#api-changes)
6. [Documentation](#documentation)
7. [Testing](#testing)
8. [Bug Fixes](#bug-fixes)

---

## Breaking Changes

### Dependency Changes

**Removed:**
- `ArraysOfArrays.jl`
- `Chain.jl` (replaced by more explicit code)

**Added:**
- `ParameterHandling.jl` - Unified parameter transformation system
- `ComponentArrays.jl` - Structured parameter containers
- `ADTypes.jl` - AD backend type definitions
- `Enzyme.jl` - Reverse-mode AD backend
- `Mooncake.jl` - Alternative reverse-mode AD backend
- `NonlinearSolve.jl` - Root-finding for simulation jump times
- `HypothesisTests.jl` - Statistical testing utilities
- `Symbolics.jl` - Symbolic computation support
- `CairoMakie.jl` - Visualization for diagnostics
- `BenchmarkTools.jl` - Performance benchmarking
- `JSON.jl` - Configuration file support
- `SpecialFunctions.jl` - Mathematical special functions
- `Statistics.jl` - Standard statistical functions
- `Random.jl` - Explicit random number control

### Hazard Family Symbols

Hazard family specifiers changed from strings to symbols:
```julia
# v0.1.0
Hazard(@formula(0 ~ 1), "exp", 1, 2)
Hazard(@formula(0 ~ 1), "wei", 1, 2)
Hazard(@formula(0 ~ 1), "gom", 1, 2)
Hazard(@formula(0 ~ 1), "sp", 1, 2; ...)

# v0.2.0
Hazard(@formula(0 ~ 1), :exp, 1, 2)
Hazard(@formula(0 ~ 1), :wei, 1, 2)
Hazard(@formula(0 ~ 1), :gom, 1, 2)
Hazard(@formula(0 ~ 1), :sp, 1, 2; ...)
```

### Internal Hazard Type Consolidation

The internal hazard types have been consolidated from 8 types to 4:

**v0.1.0 Internal Types (Removed):**
- `_Exponential`, `_ExponentialPH`
- `_Weibull`, `_WeibullPH`
- `_Gompertz`, `_GompertzPH`

**v0.2.0 Internal Types:**
- `MarkovHazard` - All Markov (exponential) hazards
- `SemiMarkovHazard` - All semi-Markov (Weibull, Gompertz) hazards
- `RuntimeSplineHazard` - All spline-based hazards
- `PhaseTypeCoxianHazard` - Phase-type expanded state hazards

### Parameter Scale Conventions

Parameters are now consistently stored on **estimation scale** internally:

| Family | Parameter | Estimation Scale | Natural Scale |
|--------|-----------|------------------|---------------|
| Exponential | rate | log(rate) | rate |
| Weibull | shape | log(shape) | shape |
| Weibull | scale | log(scale) | scale |
| Gompertz | shape | shape (unconstrained) | shape |
| Gompertz | rate | log(rate) | rate |
| Spline | coefficients | log(coef) | coef |
| All | covariate β | β | β |

**Important**: Gompertz `shape` is NOT log-transformed because it can be negative (representing decreasing hazards).

### Test Infrastructure

All tests moved to standalone repository: [MultistateModelsTests.jl](https://github.com/fintzij/MultistateModelsTests.jl)

The `test/runtests.jl` file is now a 3-line wrapper:
```julia
include(joinpath(@__DIR__, "..", "MultistateModelsTests", "src", "MultistateModelsTests.jl"))
using .MultistateModelsTests
MultistateModelsTests.runtests()
```

---

## New Features

### 1. Phase-Type Importance Sampling

**File:** `src/phasetype.jl` (4,623 lines)

Phase-type distributions provide improved importance sampling proposals for semi-Markov MCEM. They better approximate non-exponential sojourn times compared to simple Markov surrogates, yielding higher effective sample sizes.

A phase-type distribution models sojourn time as absorption in a Coxian Markov chain:
```
Phase 1 → Phase 2 → ... → Phase n → Absorption
   ↘         ↘              ↘
  Absorption  Absorption    Absorption
```

**Key Types:**

```julia
# User-facing configuration
struct ProposalConfig
    type::Symbol           # :markov or :phasetype
    n_phases::Union{Symbol, Int, Vector{Int}}  # :auto, :heuristic, or manual
    structure::Symbol      # :unstructured, :prop_to_prog, :allequal
    max_phases::Int        # Maximum for BIC selection
    optimize::Bool
    parameters::Any
    constraints::Any
end

# Internal representations
struct PhaseTypeDistribution
    n_phases::Int
    Q::Matrix{Float64}     # Full (n+1)×(n+1) intensity matrix
    initial::Vector{Float64}
end

struct PhaseTypeSurrogate
    expanded_Q::Matrix{Float64}
    n_observed_states::Int
    n_expanded_states::Int
    state_to_phases::Dict{Int, UnitRange{Int}}
    phase_to_state::Vector{Int}
    phasetypes::Dict{Int, PhaseTypeDistribution}
end

struct PhaseTypeMappings
    original_tmat::Matrix{Int}
    expanded_tmat::Matrix{Int}
    n_phases_per_state::Vector{Int}
    state_to_phases::Dict{Int, UnitRange{Int}}
    phase_to_state::Vector{Int}
    hazard_to_expanded::Dict{Symbol, Vector{Symbol}}
    expanded_to_hazard::Dict{Symbol, Symbol}
end

mutable struct PhaseTypeModel <: MultistateMarkovProcess
    # Full model for fitting on expanded state space
end
```

**Key Functions:**

```julia
# Convenience constructors
MarkovProposal(; optimize=true, parameters=nothing, constraints=nothing)
PhaseTypeProposal(; n_phases=:auto, max_phases=5, structure=:unstructured, ...)

# Distribution functions
phasetype_mean(ph::PhaseTypeDistribution) -> Float64
phasetype_variance(ph::PhaseTypeDistribution) -> Float64
phasetype_cv(ph::PhaseTypeDistribution) -> Float64
phasetype_cdf(ph::PhaseTypeDistribution, t::Real) -> Float64
phasetype_pdf(ph::PhaseTypeDistribution, t::Real) -> Float64
phasetype_hazard(ph::PhaseTypeDistribution, t::Real) -> Float64
phasetype_sample(ph::PhaseTypeDistribution) -> Float64

# Model construction
build_phasetype_surrogate(tmat, config; rates=nothing) -> PhaseTypeSurrogate
build_phasetype_model(tmat, config; data, hazards, ...) -> PhaseTypeModel
expand_data_states!(data, surrogate) -> DataFrame
collapse_phases(times, states, surrogate) -> (times, states)

# Likelihood
loglik_phasetype(Q, dt, statefrom, stateto) -> Float64
loglik_phasetype_stable(Q, dt, statefrom, stateto) -> Float64
loglik_phasetype_panel(surrogate, data, params) -> Float64

# BIC-based selection
_select_n_phases_bic(tmat, data; max_phases=5) -> Vector{Int}
```

**Usage Examples:**

```julia
# Default: auto-selects proposal type based on hazard families
fitted = fit(model)  # Uses :markov for exponential, :phasetype for Weibull/Gompertz

# Force Markov proposal
fitted = fit(model; proposal=:markov)
fitted = fit(model; proposal=MarkovProposal())

# Phase-type with BIC-based auto-selection (default)
fitted = fit(model; proposal=:phasetype)
fitted = fit(model; proposal=PhaseTypeProposal())

# Phase-type with heuristic (2 phases for non-Markov, 1 for exponential)
fitted = fit(model; proposal=PhaseTypeProposal(n_phases=:heuristic))

# Manual phase specification
fitted = fit(model; proposal=PhaseTypeProposal(n_phases=3))
fitted = fit(model; proposal=PhaseTypeProposal(n_phases=[2, 3, 1]))

# With Coxian structure constraint
fitted = fit(model; proposal=ProposalConfig(
    type=:phasetype,
    n_phases=3,
    structure=:allequal  # or :prop_to_prog
))
```

**Reference:** Titman & Sharples (2010) Biometrics 66(3):742-752

---

### 2. Robust Variance Estimation

**File:** `src/crossvalidation.jl` (2,432 lines)

Three variance estimators are now available:

#### Model-Based Variance
```math
\text{Var}(\hat{\theta}) = H^{-1} = \left(-\nabla^2 \ell(\hat{\theta})\right)^{-1}
```
- Valid only under correct model specification
- Default: `compute_vcov=true`

#### Infinitesimal Jackknife (Sandwich) Variance
```math
\text{Var}_{IJ}(\hat{\theta}) = H^{-1} K H^{-1}, \quad K = \sum_i g_i g_i^\top
```
- Robust to model misspecification
- `g_i = \nabla \ell_i(\hat{\theta})` is subject i's score contribution
- Default: `compute_ij_vcov=true`

#### Jackknife Variance
```math
\text{Var}_{JK}(\hat{\theta}) = \frac{n-1}{n} \sum_i \Delta_i \Delta_i^\top
```
- `Δ_i = \hat{\theta}_{-i} - \hat{\theta}` approximated via `H^{-1} g_i`
- Default: `compute_jk_vcov=false`

**Key Functions:**

```julia
# Per-subject score computation
compute_subject_gradients(params, model, samplepaths) -> Matrix{Float64}  # p × n
compute_subject_gradients(params, model, books) -> Matrix{Float64}        # For Markov

# Per-subject Hessian computation
compute_subject_hessians(params, model, samplepaths) -> Vector{Matrix{Float64}}
compute_subject_hessians(params, model, books) -> Vector{Matrix{Float64}}

# Fisher information components
compute_fisher_components(params, model, samplepaths) -> (H, K, grads, hessians)
compute_fisher_components(params, model, books) -> (H, K, grads, hessians)

# LOO perturbations
loo_perturbations_direct(H_inv, subject_grads) -> Matrix{Float64}   # Δᵢ = H⁻¹gᵢ
loo_perturbations_cholesky(H_chol, subject_grads, subject_hessians) -> Matrix{Float64}

# Variance computation
ij_vcov(H_inv, subject_grads) -> Matrix{Float64}     # H⁻¹KH⁻¹
jk_vcov(loo_deltas) -> Matrix{Float64}               # ((n-1)/n) ΣΔᵢΔᵢᵀ

# Combined computation
compute_robust_vcov(params, model, samplepaths; 
    compute_ij=true, compute_jk=false, loo_method=:direct) -> NamedTuple

# Diagnostics
compare_variance_estimates(fitted; use_ij=true, threshold=1.5) -> NamedTuple
get_influence_functions(fitted) -> Matrix{Float64}
```

**LOO Methods:**

| Method | Formula | Complexity | When to Use |
|--------|---------|------------|-------------|
| `:direct` | `Δᵢ = H⁻¹gᵢ` | O(p²n) | Default, n >> p |
| `:cholesky` | Exact `H₋ᵢ⁻¹` via rank-k downdate | O(np³) | Ill-conditioned problems |

**Neighborhood Cross-Validation (NCV):**

Also implemented for penalized spline smoothing parameter selection:

```julia
struct NCVState
    H_lambda::AbstractMatrix
    K::AbstractMatrix
    subject_grads::AbstractMatrix
    subject_hessians::Vector{Matrix{Float64}}
    loo_perturbations::Matrix{Float64}
    # ...
end

ncv_criterion(state, params, lambda) -> Float64
ncv_criterion_derivatives(state, params, lambda) -> NamedTuple
```

**Reference:** Wood (2020) Biometrics - Neighborhood Cross-Validation

---

### 3. SQUAREM Acceleration

**File:** `src/mcem.jl` (273 lines, enhanced)

SQUAREM (Squared Iterative Methods) accelerates MCEM convergence via quasi-Newton updates.

**Algorithm:**
1. Compute θ₁ = M(θ₀) (first EM step)
2. Compute θ₂ = M(θ₁) (second EM step)
3. r = θ₁ - θ₀ (first increment)
4. v = (θ₂ - θ₁) - r (second-order term)
5. α = -‖r‖/‖v‖ (step length)
6. θ_acc = θ₀ - 2αr + α²v (accelerated update)
7. Accept if mll(θ_acc) ≥ mll(θ₀); else fall back to θ₂

**Functions:**

```julia
squarem_step_length(r, v) -> Float64
squarem_accelerate(θ₀, r, v, α) -> Vector{Float64}
squarem_should_accept(mll_acc, mll_base) -> Bool
```

**Usage:**

```julia
fitted = fit(model; acceleration=:squarem)
fitted = fit(model; acceleration=:none)  # Default, standard MCEM
```

**Reference:** Varadhan & Roland (2008) Scand J Stat 35(2):335-353

---

### 4. Spline Hazards (Enhanced)

**File:** `src/smooths.jl` (928 lines, up from ~200)

M-spline hazards with automatic knot placement, monotonicity constraints, and time transformation caching.

**New Type:**

```julia
struct RuntimeSplineHazard <: _SplineHazard
    hazname::Symbol
    statefrom::Int64
    stateto::Int64
    family::Symbol                    # Always :sp
    parnames::Vector{Symbol}
    npar_baseline::Int64              # Number of spline coefficients
    npar_total::Int64
    hazard_fn::Function               # Runtime-generated (t, pars, covars) -> Float64
    cumhaz_fn::Function               # Runtime-generated cumulative hazard
    has_covariates::Bool
    covar_names::Vector{Symbol}
    degree::Int64                     # Spline degree (1=linear, 3=cubic)
    knots::Vector{Float64}            # Interior knot locations
    natural_spline::Bool              # Zero 2nd derivative at boundaries
    monotone::Int64                   # 0=none, 1=increasing, -1=decreasing
    extrapolation::String             # "constant" or "linear"
    metadata::HazardMetadata
    shared_baseline_key::Union{Nothing,SharedBaselineKey}
end
```

**Key Functions:**

```julia
# Automatic knot placement
place_interior_knots(sojourns, nknots; 
    quantiles=nothing, lower=nothing, upper=nothing) -> Vector{Float64}

default_nknots(n_observations) -> Int  # floor(n^(1/5)) per Tang et al.

# User-facing calibration
calibrate_splines(model; n_paths=100, min_ess=50, nknots=:auto) -> MultistateProcess
calibrate_splines!(model; n_paths=100, min_ess=50, nknots=:auto) -> Nothing

# Internal rebuilding
_rebuild_model_with_knots!(model, knot_dict) -> Nothing
_rebuild_totalhazards!(model) -> Nothing
_rebuild_model_parameters!(model) -> Nothing
_rebuild_spline_basis(hazard) -> BSplineBasis
```

**Spline Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `degree` | Polynomial degree (1=linear, 3=cubic) | 3 |
| `knots` | Interior knot locations, or `nothing` for auto | Required |
| `natural_spline` | Natural boundary conditions (0 curvature) | `false` |
| `monotone` | 0=none, 1=increasing, -1=decreasing | 0 |
| `extrapolation` | `"constant"` (C¹) or `"linear"` beyond boundaries | `"constant"` |

**Usage:**

```julia
# Cubic M-spline with user-specified knots
h12 = Hazard(@formula(0 ~ 1), :sp, 1, 2;
             degree=3,
             knots=[0.5, 1.0, 1.5],
             natural_spline=true)

# Automatic knot placement
h21 = Hazard(@formula(0 ~ 1), :sp, 2, 1; knots=nothing)

# Monotone increasing (disease progression)
h13 = Hazard(@formula(0 ~ 1), :sp, 1, 3;
             degree=3, knots=[0.5, 1.0], monotone=1)

# Monotone decreasing (recovery)
h31 = Hazard(@formula(0 ~ 1), :sp, 3, 1;
             degree=3, knots=[0.5, 1.0], monotone=-1)

# Data-driven knot calibration
model = multistatemodel(h12, h21; data=mydata)
calibrate_splines!(model; nknots=3)
```

---

### 5. AD Backend Selection

**File:** `src/common.jl` (lines 1310-1430)

Support for multiple automatic differentiation backends:

```julia
abstract type ADBackend end

struct ForwardDiffBackend <: ADBackend end
# Forward-mode AD: O(n) cost in parameters
# Tolerates in-place mutation
# Default and recommended choice

struct EnzymeBackend <: ADBackend end
# Reverse-mode AD: O(1) cost in parameters
# Requires mutation-free code

struct MooncakeBackend <: ADBackend end
# Alternative reverse-mode AD
# Pure Julia implementation
```

**Backend Compatibility:**

| Data Type | ForwardDiff | Mooncake | Enzyme |
|-----------|-------------|----------|--------|
| Exact data | ✓ | ✓ | Untested |
| Panel data (Markov) | ✓ | ✗ (LAPACK issue) | ✗ (LAPACK issue) |

**Note:** Markov panel models use matrix exponential which calls `LAPACK.gebal!` internally. Reverse-mode AD backends cannot differentiate through LAPACK. However, this is not a practical limitation since Markov models typically have few parameters where forward-mode AD is already efficient.

**Helper Functions:**

```julia
get_physical_cores() -> Int
recommended_nthreads() -> Int
default_ad_backend(n_params; is_markov=false) -> ADBackend
get_optimization_ad(backend::ADBackend) -> Optimization.AbstractADType
```

**Usage:**

```julia
# Default (ForwardDiff) - works for all model types
fitted = fit(model)
fitted = fit(model; adbackend=ForwardDiffBackend())

# Mooncake - works for exact data, may be faster for many parameters
fitted = fit(model; adbackend=MooncakeBackend())
```

---

### 6. Simulation Infrastructure

**File:** `src/simulation.jl` (1,289 lines, up from 202)

Pluggable simulation strategies for event time computation:

**Transform Strategies:**

```julia
abstract type AbstractTransformStrategy end

struct CachedTransformStrategy <: AbstractTransformStrategy end
# Precomputes and caches hazard values
# Best for repeated simulations with same parameters
# Uses more memory but faster for multiple paths

struct DirectTransformStrategy <: AbstractTransformStrategy end
# Computes hazard values on-demand
# Lower memory usage
# Better for one-off simulations
```

**Jump Solvers:**

```julia
abstract type AbstractJumpSolver end

struct OptimJumpSolver <: AbstractJumpSolver
    lower::Float64
    upper::Float64
    abstol::Float64
end
# Optimization-based root finding
# Most robust, handles complex hazard functions

struct ExponentialJumpSolver <: AbstractJumpSolver end
# Analytic solution for exponential hazards
# Fastest for Markov models

struct HybridJumpSolver <: AbstractJumpSolver
    threshold::Float64
end
# Switches between methods based on hazard properties
```

**Key Functions:**

```julia
# Unified simulation interface
simulate(model; nsim=1, data=true, paths=false, 
         strategy=CachedTransformStrategy()) -> NamedTuple

simulate_data(model; nsim=1, times=nothing, ...) -> Vector{DataFrame}
simulate_paths(model; nsim=1, ...) -> Vector{Vector{SamplePath}}
simulate_path(model, subj; strategy=CachedTransformStrategy()) -> SamplePath

# Internal
_find_jump_time(solver::OptimJumpSolver, gap_fn, lo, hi) -> Float64
_prepare_simulation_data(model, ...) -> DataFrame
_collapse_to_single_interval(data) -> DataFrame
```

**SamplePath Utilities:**

```julia
path_to_dataframe(path::SamplePath) -> DataFrame
paths_to_dataset(paths::Vector{SamplePath}; times=nothing) -> DataFrame
```

---

### 7. Performance Optimizations

**Thread-Local Workspaces:**

```julia
# Path sampling workspace (src/sampling.jl)
mutable struct PathWorkspace
    times::Vector{Float64}           # Main path storage
    states::Vector{Int}
    times_len::Int
    states_len::Int
    times_temp::Vector{Float64}      # ECCTMC temporaries
    states_temp::Vector{Int}
    R_slices::Array{Float64, 3}      # R matrix storage
    R_base::Matrix{Float64}          # Base R = I + Q/m
    R_power::Matrix{Float64}         # Matrix power workspace
    nstates::Int
end

# TVC interval workspace (src/common.jl)
mutable struct TVCIntervalWorkspace
    change_times::Vector{Float64}
    utimes::Vector{Float64}
    intervals::Vector{LightweightInterval}
    sojourns::Vector{Float64}
    pathinds::Vector{Int}
    datinds::Vector{Int}
end

# Thread-local access
get_path_workspace() -> PathWorkspace
get_tvc_workspace() -> TVCIntervalWorkspace
```

**Time Transform Caching:**

```julia
struct TimeTransformCache{LinType,TimeType}
    hazard_values::Dict{TimeTransformHazardKey{LinType,TimeType}, LinType}
    cumulhaz_values::Dict{TimeTransformCumulKey{LinType,TimeType}, LinType}
end

struct SharedBaselineTable{LinType,TimeType}
    caches::Dict{SharedBaselineKey, TimeTransformCache{LinType,TimeType}}
end

mutable struct TimeTransformContext{LinType,TimeType}
    caches::Vector{Union{Nothing,TimeTransformCache{LinType,TimeType}}}
    shared_baselines::SharedBaselineTable{LinType,TimeType}
end

# Tang shared trajectories
maybe_time_transform_context(pars, subjectdata, hazards; 
    time_column=:sojourn) -> Union{Nothing, TimeTransformContext}
enable_time_transform_cache!(flag::Bool) -> Bool
```

**Performance Results:**

| Metric | v0.1.0 | v0.2.0 | Improvement |
|--------|--------|--------|-------------|
| Time (100×100 paths) | 1.04s | 455ms | **2.3× faster** |
| Memory | 413 MiB | 169 MiB | **59% reduction** |
| Allocations | 10.2M | 4.1M | **60% reduction** |

---

## New Types

### Core Model Types

```julia
# Enhanced model types with variance storage
mutable struct MultistateModelFitted <: MultistateProcess
    model::MultistateProcess
    loglik::Float64
    parameters::NamedTuple
    vcov::Union{Nothing, Matrix{Float64}}           # Model-based
    ij_vcov::Union{Nothing, Matrix{Float64}}        # Sandwich
    jk_vcov::Union{Nothing, Matrix{Float64}}        # Jackknife
    subject_gradients::Union{Nothing, Matrix{Float64}}
    loo_perturbations::Union{Nothing, Matrix{Float64}}
    convergence_records::Union{Nothing, DataFrame}
    # ... additional fields
end
```

### Data Structures

```julia
# Batched ODE data for vectorized likelihood
struct BatchedODEData
    intervals::Vector{Tuple{Float64,Float64}}
    statefrom::Vector{Int}
    stateto::Vector{Int}
    path_indices::Vector{Int}
    # ...
end

struct StackedHazardData
    hazard_idx::Int
    intervals::Vector{Tuple{Float64,Float64}}
    subject_indices::Vector{Int}
    covar_data::Vector{NamedTuple}
end

struct CachedPathData
    path::SamplePath
    intervals::Vector{LightweightInterval}
    sojourns::Vector{Float64}
end

struct LightweightInterval
    lb::Float64
    ub::Float64
    statefrom::Int
    stateto::Int
end

struct SubjectCovarCache
    subject_id::Int
    covar_data::Vector{NamedTuple}  # Per-interval covariates
end
```

### Hazard Metadata

```julia
struct HazardMetadata
    time_transform::Bool      # Enable Tang time transformation
    linpred_effect::Symbol    # :ph or :aft
end

struct TimeTransformHazardKey{LinType,TimeType}
    linpred::LinType
    t::TimeType
end

struct TimeTransformCumulKey{LinType,TimeType}
    linpred::LinType
    lb::TimeType
    ub::TimeType
end

struct SharedBaselineKey
    statefrom::Int
    baseline_signature::UInt64  # Hash of spline configuration
end
```

---

## New Functions

### Parameter Handling

```julia
# Flatten/unflatten with multiple scales
unflatten_natural(flat_params, model) -> NamedTuple
unflatten_estimation(flat_params, model) -> NamedTuple
rebuild_parameters(new_vectors, model) -> NamedTuple

# Scale conversion
to_natural_scale(params_nested, hazards, ::Type{T}) -> NamedTuple
to_estimation_scale(params_nested, hazards) -> NamedTuple

# Covariate extraction
extract_covar_names(parnames) -> Vector{Symbol}
extract_covariates(subjdat, parnames) -> Vector{Float64}
extract_baseline_values(hazard_params) -> NamedTuple
extract_covariate_values(hazard_params) -> NamedTuple

# Parameter manipulation
set_parameters!(model, newvalues::Vector{Vector{Float64}})
set_parameters!(model, h::Int, newvalues::Vector{Float64})
get_parameters(model; scale=:natural) -> NamedTuple
get_parameters(fitted; scale=:natural, expanded=false) -> NamedTuple
get_expanded_parameters(model; scale=:natural) -> NamedTuple
```

### Hazard Evaluation

```julia
# Callable hazard types
(hazard::MarkovHazard)(t, pars, covars) -> Float64
(hazard::SemiMarkovHazard)(t, pars, covars) -> Float64
(hazard::_SplineHazard)(t, pars, covars) -> Float64

# Cumulative hazard
cumulative_hazard(hazard, lb, ub, pars, covars; ttctx=nothing) -> Float64

# Hazard rate matrix computation
compute_hazmat!(Q, parameters, hazards, tpm_index, model_data)
compute_hazmat_cached!(Q, parameters, hazards, covars_cache)
compute_hazmat_from_rates!(Q, rates_cache, hazards)
compute_hazard_rates!(rates_cache, parameters, hazards, covars_cache)

# Transition probability matrix
compute_tmat(Q, dt) -> Matrix{Float64}
compute_tmat_batched!(P_vec, Q_vec, dts)
```

### Likelihood Functions

```julia
# Unified interface with AD backend dispatch
loglik_markov(parameters, data::MPanelData; neg=true, backend=ForwardDiffBackend())
loglik_semi_markov(parameters, data::SMPanelData; neg=true, parallel=false)
loglik_semi_markov!(parameters, logliks, data::SMPanelData)
loglik_semi_markov_batched!(parameters, logliks, data::SMPanelData)

# Internal implementations
_loglik_markov_mutating(parameters, data; neg=true)      # For ForwardDiff
_loglik_markov_functional(parameters, data; neg=true)    # For Enzyme/Mooncake
_forward_algorithm_functional(subj_inds, pars, data, tpm_dict, ::Type{T}, ...)

# Single path likelihood
loglik(parameters, path::SamplePath, hazards, model)
loglik_path(pars_nested, subjdat_df, hazards, totalhazards, tmat)

# Caching
cache_path_data(paths, model) -> Vector{CachedPathData}
stack_intervals_for_hazard(hazard_idx, cached_paths, ...) -> StackedHazardData
to_batched_ode_data(sd::StackedHazardData; use_views=false) -> BatchedODEData
```

### Model Output

```julia
# Variance accessors
get_vcov(fitted; type=:model) -> Matrix{Float64}  # type ∈ {:model, :ij, :jk}
get_ij_vcov(fitted) -> Matrix{Float64}
get_jk_vcov(fitted) -> Matrix{Float64}
get_subject_gradients(fitted) -> Matrix{Float64}  # p × n
get_loo_perturbations(fitted; method=:direct) -> Matrix{Float64}
get_pseudovalues(fitted; type=:jk) -> Matrix{Float64}
get_influence_functions(fitted) -> Matrix{Float64}
compare_variance_estimates(fitted; use_ij=true, threshold=1.5)

# Phase-type accessors
is_phasetype_fitted(fitted) -> Bool
get_phasetype_parameters(fitted; scale=:natural) -> NamedTuple
get_mappings(fitted) -> PhaseTypeMappings
get_original_data(fitted) -> DataFrame
get_original_tmat(fitted) -> Matrix{Int}
get_convergence(fitted) -> DataFrame
```

### Surrogate Fitting

```julia
fit_surrogate(model; type=:markov, method=:mle, verbose=true) -> Surrogate
is_surrogate_fitted(model) -> Bool
set_surrogate!(model; Q=nothing, parameters=nothing)
fit_phasetype_surrogate(model, markov_surrogate; n_phases=:auto) -> PhaseTypeSurrogate
_build_coxian_from_rate(n_phases, total_rate; structure=:unstructured) -> (λ, μ)
compute_markov_marginal_loglik(model, surrogate) -> Float64
compute_phasetype_marginal_loglik(model, surrogate) -> Float64
```

### Path Sampling

```julia
# High-level
draw_paths(model; npaths=100, paretosmooth=true) -> Vector{Vector{SamplePath}}
DrawSamplePaths!(model; ess_target, ess_cur, max_sampling_effort, ...)

# Low-level with workspace
draw_samplepath!(ws::PathWorkspace, subj, model, tpm_book, ...)
_sample_ecctmc_ws!(ws, P, Q, a, b, t0, t1)
reduce_jumpchain_ws(ws, subj) -> SamplePath
to_samplepath(ws, subj) -> SamplePath

# FFBS
ForwardFiltering!(subj_fbmats, subj_dat, tpm_book, tpm_map, emat; init_state=nothing)
BackwardSampling!(subj_dat, subj_fbmats) -> Vector{Int}
BackwardSampling(m, p) -> Int  # Single draw

# Phase-type FFBS
build_phasetype_tpm_book(surrogate, books, data)
build_phasetype_emat_expanded(model, surrogate; ...)
build_fbmats_phasetype(model, surrogate)
draw_samplepath_phasetype(subj, model, surrogate, ...)
BackwardSampling_expanded(subj_fbmats, n_expanded) -> Vector{Int}
collapse_phasetype_path(expanded_path, surrogate, absorbingstates) -> SamplePath
loglik_phasetype_path(path, surrogate, params) -> Float64
loglik_phasetype_expanded(expanded_path, surrogate) -> Float64
```

### Initialization

```julia
initialize_parameters!(model; method=:crude, verbose=false)
initialize_parameters(model; method=:crude) -> NamedTuple
_init_from_surrogate_rates!(model; verbose=false)
_init_from_surrogate_paths!(model, surrogate; n_paths=100)
compute_suff_stats(data, tmat, weights) -> NamedTuple
init_par(hazard, crude_log_rate) -> Vector{Float64}
```

---

## API Changes

### Enhanced `fit()` Signature

**Markov Models:**
```julia
fit(model::MultistateMarkovModel;
    constraints = nothing,
    verbose = true,
    solver = nothing,
    adbackend::ADBackend = ForwardDiffBackend(),
    compute_vcov = true,
    vcov_threshold = true,
    compute_ij_vcov = true,
    compute_jk_vcov = false,
    loo_method = :direct)
```

**Semi-Markov Models (MCEM):**
```julia
fit(model::MultistateSemiMarkovModel;
    proposal::Union{Symbol, ProposalConfig} = :auto,  # NEW
    constraints = nothing,
    solver = nothing,
    maxiter = 100,
    tol = 1e-2,
    ascent_threshold = 0.1,
    stopping_threshold = 0.1,
    ess_increase = 2.0,
    ess_target_initial = 50,
    max_ess = 10000,
    max_sampling_effort = 20,
    npaths_additional = 10,
    block_hessian_speedup = 2.0,
    acceleration::Symbol = :none,          # NEW: :none or :squarem
    verbose = true,
    return_convergence_records = true,
    return_proposed_paths = false,
    compute_vcov = true,
    vcov_threshold = true,
    compute_ij_vcov = true,                # NEW
    compute_jk_vcov = false,               # NEW
    loo_method = :direct)                  # NEW
```

**Phase-Type Models:**
```julia
fit(model::PhaseTypeModel;
    constraints = nothing,
    verbose = true,
    solver = nothing,
    compute_vcov = true,
    vcov_threshold = true,
    compute_ij_vcov = true,
    compute_jk_vcov = false,
    loo_method = :direct)
```

### New Exports

```julia
# Model accessors
get_parameters, get_vcov, get_loglik, get_parnames
get_pseudovalues, get_convergence_records, get_expanded_parameters

# Variance diagnostics
get_subject_gradients, get_loo_perturbations
get_influence_functions, compare_variance_estimates

# Phase-type
is_phasetype_fitted, get_phasetype_parameters
get_mappings, get_original_data, get_original_tmat, get_convergence
PhaseTypeDistribution, PhaseTypeModel

# Simulation strategies
OptimJumpSolver, ExponentialJumpSolver, HybridJumpSolver
CachedTransformStrategy, DirectTransformStrategy

# AD backends
ADBackend, ForwardDiffBackend, EnzymeBackend, MooncakeBackend

# MCEM proposals
ProposalConfig, MarkovProposal, PhaseTypeProposal

# Utilities
get_physical_cores, recommended_nthreads
calibrate_splines, calibrate_splines!
path_to_dataframe, paths_to_dataset
```

---

## Documentation

### New Documentation Files

- `docs/src/index.md` - Comprehensive user guide (238 lines, up from 1)
  - Hazard family specifications with mathematical formulas
  - Parameter scale conventions
  - Spline hazard options
  - Simulation strategies
  - Covariate modeling (PH vs AFT)

- `docs/src/optimization.md` - Optimization and variance guide (133 lines)
  - Solver selection guidelines
  - Variance estimation methods
  - LOO method comparison
  - Diagnostic recommendations

- `docs/src/phasetype_ffbs.md` - Phase-type FFBS algorithm (204 lines)
  - Mathematical details
  - Importance weight computation

### Key Documentation Topics

**Hazard Parameterizations** (following flexsurv R package):

| Family | Hazard h(t) | Cumulative H(t) | Parameters |
|--------|-------------|-----------------|------------|
| Exponential | rate | rate × t | rate > 0 |
| Weibull | shape × scale × t^(shape-1) | scale × t^shape | shape > 0, scale > 0 |
| Gompertz | rate × exp(shape × t) | (rate/shape) × (exp(shape×t) - 1) | shape ∈ ℝ, rate > 0 |

**Covariate Effects:**
- **Proportional Hazards (PH)**: h(t|x) = h₀(t) × exp(xᵀβ)
- **Accelerated Failure Time (AFT)**: h(t|x) = h₀(t × exp(xᵀβ)) × exp(xᵀβ)

---

## Testing

### Test Infrastructure

All tests moved to: [MultistateModelsTests.jl](https://github.com/fintzij/MultistateModelsTests.jl)

### Test Coverage

| Category | Tests | Time | Description |
|----------|-------|------|-------------|
| Unit Tests | 1,149 | ~2 min | Fast validation |
| Exact Data Fitting | 45 | ~5 min | Exact Markov inference |
| MCEM Parametric | 45 | ~15 min | Exp/Weibull/Gompertz convergence |
| MCEM Splines | 45 | ~15 min | M-spline hazard convergence |
| MCEM TVC | 38 | ~10 min | Time-varying covariates |
| Simulation Distribution | 65 | ~10 min | Event time correctness |
| Simulation TVC | 9,702 | ~20 min | Piecewise hazard validation |
| Phase-Type | 35 | ~10 min | Importance sampling validation |
| **Total** | **11,124+** | **~90 min** | All passing |

### Running Tests

```bash
# Quick unit tests only
julia --project=. -e 'using Pkg; Pkg.test()'

# Full test suite (requires MSM_TEST_LEVEL=full)
MSM_TEST_LEVEL=full julia --project=. -e 'using Pkg; Pkg.test()'

# Specific long test categories
MSM_LONGTEST_PHASETYPE=1 MSM_TEST_LEVEL=full julia --project=. -e 'using Pkg; Pkg.test()'
MSM_LONGTEST_MCEM=1 MSM_TEST_LEVEL=full julia --project=. -e 'using Pkg; Pkg.test()'
MSM_LONGTEST_SIMULATION=1 MSM_TEST_LEVEL=full julia --project=. -e 'using Pkg; Pkg.test()'
MSM_LONGTEST_VARIANCE_VALIDATION=1 MSM_TEST_LEVEL=full julia --project=. -e 'using Pkg; Pkg.test()'

# Long tests only
MSM_LONGTEST_ONLY=1 MSM_TEST_LEVEL=full julia --project=. -e 'using Pkg; Pkg.test()'
```

---

## Bug Fixes

### Gompertz Parameter Handling
- **Fixed**: Gompertz `shape` was incorrectly log-transformed in some code paths
- **Correct behavior**: Shape is unconstrained (can be negative, zero, or positive)
- **Impact**: Fixes simulation and fitting for decreasing hazard models

### Cumulative Hazard Formulas
- **Fixed**: Gompertz cumulative hazard missing `1/shape` factor
- **Fixed**: Piecewise TVC cumulative hazard formula errors
- **Correct formula**: H(t) = (rate/shape) × (exp(shape×t) - 1)

### Spline Covariate Extraction
- **Fixed**: `extract_covar_names()` incorrectly included spline basis parameters
- **Correct behavior**: Filters out `sp1`, `sp2`, ... from covariate list

### ECCTMC Sampling
- **Fixed**: Matrix power computation using pre-allocated workspace
- **Impact**: Eliminates allocation hotspot in path sampling

### Initialization
- **Fixed**: Crude rate initialization for spline hazards
- **Fixed**: Surrogate-based initialization for complex models

---

## Code Statistics

### Source File Changes

| File | v0.1.0 Lines | v0.2.0 Lines | Change |
|------|--------------|--------------|--------|
| `common.jl` | 424 | 1,600 | +1,176 |
| `hazards.jl` | 751 | 1,762 | +1,011 |
| `sampling.jl` | 826 | 2,200 | +1,374 |
| `likelihoods.jl` | 391 | 2,156 | +1,765 |
| `simulation.jl` | 202 | 1,289 | +1,087 |
| `helpers.jl` | ~400 | 1,965 | +1,565 |
| `modelfitting.jl` | ~500 | 1,683 | +1,183 |
| `smooths.jl` | ~200 | 928 | +728 |
| `modeloutput.jl` | ~200 | 1,176 | +976 |
| `initialization.jl` | ~150 | 549 | +399 |
| `surrogates.jl` | ~150 | 721 | +571 |
| `modelgeneration.jl` | ~600 | 1,198 | +598 |
| `mcem.jl` | ~100 | 273 | +173 |

### New Files

| File | Lines | Purpose |
|------|-------|---------|
| `phasetype.jl` | 4,623 | Phase-type distributions and surrogates |
| `crossvalidation.jl` | 2,432 | Robust variance estimation |
| `types/hazards.jl` | 666 | Consolidated hazard type definitions |
| `types/data.jl` | 286 | Data structure types |
| `types/configuration.jl` | 234 | Configuration types |
| `types/models.jl` | 227 | Model type definitions |
| `pathfunctions.jl` | 403 | SamplePath utilities |
| `macros.jl` | 110 | `@hazard` macro |
| `types/utilities.jl` | 101 | Utility types |
| `types/surrogates.jl` | 58 | Surrogate types |
| `statsutils.jl` | 44 | Statistical utilities |
| `types/types.jl` | 27 | Type re-exports |

### Totals

| Metric | v0.1.0 | v0.2.0 |
|--------|--------|--------|
| Source files | 16 | 27 |
| Total lines | ~5,500 | ~27,000 |
| Test files | 15 | 98 (in test package) |
| Tests | ~200 | 11,124+ |

---

## References

- Titman, A. C., & Sharples, L. D. (2010). Semi-Markov Models with Phase-Type Sojourn Distributions. Biometrics, 66(3), 742-752.
- Varadhan, R., & Roland, C. (2008). Simple and Globally Convergent Methods for Accelerating the Convergence of Any EM Algorithm. Scandinavian Journal of Statistics, 35(2), 335-353.
- Wood, S. N. (2020). Inference and Computation with Generalized Additive Models and their Extensions. TEST, 29, 249-283.
- flexsurv R package for hazard parameterization conventions
