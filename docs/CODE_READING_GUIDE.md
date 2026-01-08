# MultistateModels.jl Code Reading Guide

A structured guide for understanding the codebase, organized by workflow.

---

## Overview

MultistateModels.jl implements continuous-time multistate models for survival analysis. The typical workflow is:

1. **Model Generation** → Define hazards & construct model
2. **Hazard Functions** → Compute hazards, cumulative hazards, survival
3. **Simulation** → Generate sample paths from the model
4. **Inference** → Fit model to data via MLE or MCEM

---

## 1. Model Generation

### Entry Point
- `src/construction/multistatemodel.jl` — Orchestrates includes & `multistatemodel()` function

### Construction Submodules (included in dependency order)

| File | Purpose | Key Functions |
|------|---------|---------------|
| `src/construction/hazard_constructors.jl` | User-facing hazard specification | `Hazard()`, `enumerate_hazards()`, `create_tmat()` |
| `src/construction/hazard_builders.jl` | Internal hazard building infrastructure | `HazardBuildContext`, `_HAZARD_BUILDERS` registry, parametric builders |
| `src/construction/spline_builder.jl` | Spline hazard construction | `_build_spline_hazard()`, `_generate_spline_hazard_fns()` |
| `src/construction/model_assembly.jl` | Model component assembly | `build_hazards()`, `build_parameters()`, `_assemble_model()` |

### Type Definitions
| File | Purpose |
|------|---------|
| `src/types/hazard_specs.jl` | User-facing types: `ParametricHazard`, `SplineHazard`, `PhaseTypeHazard` |
| `src/types/hazard_structs.jl` | Internal runtime types: `MarkovHazard`, `SemiMarkovHazard`, `RuntimeSplineHazard` |
| `src/types/model_structs.jl` | Model types: `MultistateMarkovModel`, `MultistateSemiMarkovModel`, fitted variants |
| `src/types/hazard_metadata.jl` | `HazardMetadata`, Tang caching types |

### Reading Order
1. `ParametricHazard`/`SplineHazard` in `types/hazard_specs.jl`
2. `multistatemodel()` in `construction/multistatemodel.jl` (entry point)
3. `Hazard()` in `construction/hazard_constructors.jl` — User-facing constructor
4. `_HAZARD_BUILDERS` registry in `construction/hazard_builders.jl`
5. `build_hazards()` in `construction/model_assembly.jl`
6. `MultistateMarkovModel` etc. in `types/model_structs.jl`

### Hazard Builder Registry Pattern
```julia
# Register a hazard family builder:
register_hazard_family!(:wei, _build_weibull_hazard)

# Builders are dispatched by family symbol:
_HAZARD_BUILDERS[:wei](ctx)  # ctx = HazardBuildContext
```

---

## 2. Hazard Functions

### Internal Hazard Types
The internal runtime hazard structs (`MarkovHazard`, `SemiMarkovHazard`, `RuntimeSplineHazard`) contain:
- **Metadata**: `hazname`, `statefrom`, `stateto`, `family`, `parnames`
- **Generated closures**: `hazard_fn(t, pars, covars)`, `cumhaz_fn(t, pars, covars)`
- **Covariate names**: `covar_names::Vector{Symbol}` for lookup (NOT data copies)
- **Cache keys**: `shared_baseline_key` for Tang optimization
- **Penalty info**: `smooth_info::Vector{SmoothTermInfo}` for smooth covariate terms

Hazards do **not** store copies of the data — only covariate names for lookup.

### Hazard Types by Family
| Family | Internal Type | Builder Location |
|--------|--------------|-----------------|
| `:exp` | `MarkovHazard` | `construction/hazard_builders.jl` |
| `:wei` | `SemiMarkovHazard` | `construction/hazard_builders.jl` |
| `:gom` | `SemiMarkovHazard` | `construction/hazard_builders.jl` |
| `:sp` | `RuntimeSplineHazard` | `construction/spline_builder.jl` |

### Hazard Metadata
From `src/types/hazard_metadata.jl`:
```julia
struct HazardMetadata
    time_transform::Bool     # Tang-style trajectory caching
    linpred_effect::Symbol   # :ph or :aft
end
```
Also defines `TimeTransformCache`, `SharedBaselineKey`, `SharedBaselineTable` for shared trajectory caching.

### Spline Infrastructure
| File | Purpose |
|------|---------|
| `src/construction/spline_builder.jl` | Builds `RuntimeSplineHazard`, generates hazard/cumhaz closures |
| `src/hazard/spline.jl` | B-spline basis evaluation, `calibrate_splines()` utilities |
| `src/utilities/spline_utils.jl` | Knot placement, GPS penalty matrices (`build_penalty_matrix_gps`) |

### Monotone Splines
Monotone splines use an I-spline-like cumulative sum transformation:
```julia
coefs = cumsum(ests .* weights)  # _spline_ests2coefs
```
The optimization parameters `ests` are non-negative increments; the spline coefficients `coefs` are the cumulative sums.

**Known issue**: The penalty matrix is built for `coefs` (B-spline coefficients) but applied to `ests` (increments). The correct penalty should transform via `S_monotone = L' * S * L` where `L` is the cumulative sum matrix. This transformation is **not currently implemented**.

---

## 3. Simulation

### Key Files
| File | Purpose |
|------|---------|
| `src/simulation/simulate.jl` | Main `simulate()` dispatch |
| `src/simulation/path_utilities.jl` | Path extraction, `path_to_dataframe()` |

### Entry Point
`simulate()` in `src/simulation/simulate.jl`

---

## 4. Inference

### Entry Point
`fit()` in `src/inference/fit_common.jl`

### Fitting Files
| File | Purpose |
|------|---------|
| `src/inference/fit_common.jl` | Main interface, dispatch |
| `src/inference/fit_exact.jl` | Exact observation fitting |
| `src/inference/fit_markov.jl` | Markov panel fitting |
| `src/inference/fit_mcem.jl` | Semi-Markov MCEM fitting |
| `src/inference/mcem.jl` | MCEM algorithm core |
| `src/inference/sampling.jl` | Path sampling for MCEM |
| `src/inference/smoothing_selection.jl` | PIJCV smoothing parameter selection |
| `src/inference/sir.jl` | Sampling importance resampling |

### Likelihood Files
| File | Purpose |
|------|---------|
| `src/likelihood/loglik_utils.jl` | ForwardDiff helpers, parameter prep |
| `src/likelihood/loglik_batched.jl` | Batched hazard-centric infrastructure |
| `src/likelihood/loglik_markov.jl` | Markov panel, forward algorithm |
| `src/likelihood/loglik_markov_functional.jl` | Reverse-mode AD compatible |
| `src/likelihood/loglik_semi_markov.jl` | Semi-Markov MCEM path-based |
| `src/likelihood/loglik_exact.jl` | Exact data, fused computation |

### Penalty Computation
`compute_penalty()` in `src/types/infrastructure.jl`:
```julia
penalty = (1/2) Σⱼ λⱼ βⱼᵀ Sⱼ βⱼ + total_hazard_penalties + smooth_covariate_penalties
```
Where `S` is the GPS penalty matrix (second-order differences for curvature).

---

## 5. Phase-Type Models

### Key Files
| File | Purpose |
|------|---------|
| `src/phasetype/types.jl` | Phase-type type definitions |
| `src/phasetype/surrogate.jl` | Phase-type surrogate model |
| `src/phasetype/expansion_mappings.jl` | State space mappings |
| `src/phasetype/expansion_hazards.jl` | Hazard expansion |
| `src/phasetype/expansion_constraints.jl` | SCTP constraint generation |
| `src/phasetype/expansion_model.jl` | Model building |
| `src/phasetype/expansion_loglik.jl` | Log-likelihood computation |
| `src/phasetype/expansion_ffbs_data.jl` | Data expansion for FFBS |

### Concept
Phase-type distributions model hazards as absorption times of underlying CTMCs. Observed states are coarse; internally, the model tracks finer "phases."

---

## 6. Utilities

| File | Purpose |
|------|---------|
| `src/utilities/constants.jl` | Package-wide numerical constants |
| `src/utilities/flatten.jl` | Parameter flattening type system |
| `src/utilities/reconstructor.jl` | ReConstructor struct, unflatten API |
| `src/utilities/parameters.jl` | Parameter set/get, build functions |
| `src/utilities/transforms.jl` | Estimation ↔ natural scale conversions |
| `src/utilities/validation.jl` | Input validation |
| `src/utilities/initialization.jl` | Crude parameter initialization |
| `src/utilities/bounds.jl` | Parameter bounds for optimization |
| `src/utilities/spline_utils.jl` | Knot placement, penalty matrices |
| `src/utilities/penalty_config.jl` | `PenaltyConfig` builder |
| `src/utilities/books.jl` | TPM bookkeeping, data containers |
| `src/utilities/misc.jl` | Miscellaneous helpers |
| `src/utilities/stats.jl` | Shared stats utilities |
| `src/utilities/transition_helpers.jl` | Transition enumeration |

### Surrogate
| File | Purpose |
|------|---------|
| `src/surrogate/markov.jl` | Markov surrogate for importance sampling |

### Output/Accessors
| File | Purpose |
|------|---------|
| `src/output/accessors.jl` | `get_parameters()`, `get_vcov()`, `get_loglik()` |
| `src/output/variance.jl` | Cross-validation, robust covariance |

---

## 7. Type Hierarchy

### Model Types
```
AbstractMultistateModel
  ├── MultistateMarkovModel           # Markov, exact or panel
  ├── MultistateMarkovModelFitted
  ├── MultistateSemiMarkovModel       # Semi-Markov, requires MCEM
  └── MultistateSemiMarkovModelFitted
```

### Hazard Types (User-Facing)
```
HazardFunction (abstract)
  ├── ParametricHazard    # exp, wei, gom
  ├── SplineHazard        # sp
  └── PhaseTypeHazard     # pt
```

### Hazard Types (Internal Runtime)
```
_Hazard (abstract)
  ├── MarkovHazard        # exp
  ├── SemiMarkovHazard    # wei, gom
  └── RuntimeSplineHazard # sp
```

### Surrogate Types
```
AbstractSurrogate
  ├── MarkovSurrogate         # For semi-Markov importance sampling
  └── PhaseTypeSurrogate      # For phase-type importance sampling
```

---

## 8. Key Design Patterns

### Hazard Builder Registry
```julia
register_hazard_family!(:wei, _build_weibull_hazard)
# Dispatched by: _HAZARD_BUILDERS[:wei](ctx)
```

### Parameter Views
`parameters.hazards` provides views into `parameters.flat` (shared memory).

### Observation Types
- `obstype=1`: Exact transition time observed
- `obstype=2`: State observed at discrete times (panel)
- `obstype≥3`: Censoring patterns

---

## 9. Actual File Inventory

### src/types/
- `abstract.jl` — Abstract type hierarchy
- `data_containers.jl` — `SamplePath`, data containers
- `hazard_metadata.jl` — `HazardMetadata`, Tang caching types
- `hazard_specs.jl` — User-facing `ParametricHazard`, `SplineHazard`, `PhaseTypeHazard`
- `hazard_structs.jl` — Internal `MarkovHazard`, `SemiMarkovHazard`, `RuntimeSplineHazard`
- `infrastructure.jl` — `PenaltyConfig`, `ADBackend`, `compute_penalty()`
- `model_structs.jl` — Model types

### src/construction/
- `multistatemodel.jl` — Entry point, includes other files
- `hazard_constructors.jl` — `Hazard()` user constructor
- `hazard_builders.jl` — Registry, parametric builders
- `spline_builder.jl` — Spline hazard builder
- `model_assembly.jl` — Assembly pipeline

### src/hazard/
- `api.jl` — User-facing API (`compute_hazard`, `cumulative_incidence`)
- `covariates.jl` — Covariate extraction
- `evaluation.jl` — Callable hazard interface
- `generators.jl` — Runtime code generation
- `macros.jl` — `@hazard` macro
- `smooth_terms.jl` — `s()`, `te()` smooth term parsing
- `spline.jl` — B-spline evaluation, calibration
- `time_transform.jl` — Tang optimizations
- `total_hazard.jl` — Total hazard computation
- `tpm.jl` — Transition probability matrices

### src/likelihood/
- `loglik_utils.jl`, `loglik_batched.jl`, `loglik_markov.jl`, `loglik_markov_functional.jl`, `loglik_semi_markov.jl`, `loglik_exact.jl`

### src/inference/
- `fit_common.jl`, `fit_exact.jl`, `fit_markov.jl`, `fit_mcem.jl`, `mcem.jl`, `sampling.jl`, `smoothing_selection.jl`, `sir.jl`

### src/phasetype/
- `types.jl`, `surrogate.jl`, `expansion_mappings.jl`, `expansion_hazards.jl`, `expansion_constraints.jl`, `expansion_model.jl`, `expansion_loglik.jl`, `expansion_ffbs_data.jl`

### src/utilities/
- `bounds.jl`, `books.jl`, `constants.jl`, `flatten.jl`, `initialization.jl`, `misc.jl`, `parameters.jl`, `penalty_config.jl`, `reconstructor.jl`, `spline_utils.jl`, `stats.jl`, `transforms.jl`, `transition_helpers.jl`, `validation.jl`

### src/simulation/
- `simulate.jl`, `path_utilities.jl`

### src/surrogate/
- `markov.jl`

### src/output/
- `accessors.jl`, `variance.jl`

---

## 10. Testing

Tests are in `test/` (main suite) and `MultistateModelsTests/` (extended):
- `test/runtests.jl` — Main test runner
- `MultistateModelsTests/unit/` — Unit tests
- `MultistateModelsTests/integration/` — End-to-end tests
- `MultistateModelsTests/longtests/` — Extended validation

---

*Updated: January 7, 2026*
