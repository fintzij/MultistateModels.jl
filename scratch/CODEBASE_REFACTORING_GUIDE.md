# MultistateModels.jl Codebase Refactoring Guide

**Created**: 2026-01-08  
**Last Updated**: 2026-01-22  
**Branch**: penalized_splines  
**Status**: 🔴 **BLOCKING BUG** — Unpenalized spline MCEM has boundary coefficient failure

---

## Quick Status

| Status | Description |
|--------|-------------|
| ✅ Waves 1-3 | Foundation, technical debt, math correctness — COMPLETE |
| ✅ Items #35-37 | PhaseType infrastructure cleanup — COMPLETE |
| ✅ Item #27 (partial) | VCoV API unification — COMPLETE (constraint handling TODO) |
| 🔴 Items #30-34 | Spline MCEM boundary coefficient bug — ACTIVE |
| 🟡 Item #38 | MCEM surrogate-agnostic refactor — IN PROGRESS |
| 🟡 Item #39 | PhaseType MCEM bias investigation — IN PROGRESS |
| ⏳ Item #28 | Simulation diagnostics — PLANNED |

---

## 🔴 CRITICAL BUG: Unpenalized Spline MCEM Boundary Coefficient Failure

### Executive Summary

**BLOCKING BUG**: Unpenalized spline hazards fitted via MCEM on panel data produce catastrophically wrong estimates for the rightmost B-spline coefficient (coef_4). The 4th coefficient is overestimated by 300-1000%.

**Impact**: 6 of 38 long tests fail. All failures are in `longtest_spline_suite.jl` for panel data scenarios.

### Failure Pattern

| Test | coef_4 True | coef_4 Est | Rel Error | Status |
|------|-------------|------------|-----------|--------|
| sp_ph_panel_markov_nocov | 0.18 | 0.87 | 382% | ❌ FAIL |
| sp_aft_panel_markov_nocov | 0.18 | 0.89 | 394% | ❌ FAIL |
| h23_coef_4 | 0.14 | 1.55 | 1005% | ❌ FAIL |

### What Works vs What Fails

| Scenario | Status |
|----------|--------|
| Spline + exact data | ✅ PASS |
| Spline + panel + TVC | ✅ PASS (elevated estimates) |
| Spline + panel + nocov/fixed | ❌ FAIL |
| Parametric MCEM | ✅ PASS |

### Root Cause Hypotheses

1. **Boundary spline extrapolation** (HIGH): `extrapolation="constant"` may weight boundary basis incorrectly for sampled paths extending beyond observation window
2. **MCEM path sampling time distribution** (MEDIUM): Sampled paths may concentrate transitions near right boundary
3. **Unpenalized identifiability** (MEDIUM): coef_4 poorly identified when data sparse near boundary

### Action Items

| Item | Task | Priority | Status |
|------|------|----------|--------|
| #30 | Diagnostic script for spline MCEM debugging | 🔴 CRITICAL | ⏳ TODO |
| #31 | Investigate spline extrapolation in MCEM | 🔴 CRITICAL | ⏳ TODO |
| #32 | Compare basis function weights | 🟡 HIGH | ⏳ TODO |
| #33 | Test with penalized splines | 🟡 MEDIUM | ⏳ TODO |
| #34 | Verify TVC "passes" are not false positives | 🟡 MEDIUM | ⏳ TODO |

### Key Files

- `src/hazard/spline_hazard.jl` — extrapolation logic
- `src/inference/sampling_markov.jl`, `sampling_phasetype.jl` — path sampling
- `src/likelihood/loglik_path.jl` — complete-data likelihood
- `MultistateModelsTests/longtests/longtest_spline_suite.jl` — failing tests

---

## 🟡 Item #38: MCEM Surrogate-Agnostic Refactor

**Priority**: 🟡 HIGH (Reduces complexity, enables future fixes)  
**Status**: 🟡 IN PROGRESS — Phase 1 started  
**Effort**: 16-24 hours  
**Reference**: [MCEM_REFACTOR_PLAN.md](MCEM_REFACTOR_PLAN.md)

### Problem Statement

`fit_mcem.jl` (1247 lines) has 16+ `if use_phasetype` branches interleaved throughout, making the code hard to maintain, debug, and extend. The PhaseType and Markov surrogate paths have different but parallel data structures that should be unified.

### Solution

Create `MCEMInfrastructure{S<:AbstractSurrogate}` parametric type that encapsulates all surrogate-specific state, then dispatch on this type instead of boolean flags.

### Implementation Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Type definitions (`MCEMInfrastructure{S}`) | 🟡 Started |
| 2 | Markov infrastructure builder | ⏳ TODO |
| 3 | PhaseType infrastructure builder | ⏳ TODO |
| 4 | Dispatch methods (`compute_surrogate_path_loglik`, etc.) | ⏳ TODO |
| 5 | Integration & validation | ⏳ TODO |

### New Type Structure

```julia
abstract type AbstractSurrogate end
struct MarkovSurrogate <: AbstractSurrogate end
struct PhaseTypeSurrogate <: AbstractSurrogate end

struct MCEMInfrastructure{S<:AbstractSurrogate}
    surrogate::S
    books::Tuple
    hazmat_book::Vector{Matrix{Float64}}
    tpm_book::Vector{Vector{Matrix{Float64}}}
    fbmats::Union{Nothing, ...}
    # PhaseType-specific fields (nothing for Markov)
    phasetype_surrogate::Union{Nothing, PhaseTypeSurrogate}
    schur_cache::Union{Nothing, Vector{CachedSchurDecomposition}}
end
```

### Key Dispatch Methods

```julia
build_mcem_infrastructure(model, ::MarkovSurrogate; kwargs...)
build_mcem_infrastructure(model, ::PhaseTypeSurrogate; kwargs...)
compute_surrogate_path_loglik(path, infra::MCEMInfrastructure{MarkovSurrogate}, ...)
compute_surrogate_path_loglik(path, infra::MCEMInfrastructure{PhaseTypeSurrogate}, ...)
compute_normalizing_constant(model, infra::MCEMInfrastructure{S}, ...) where S
```

### Test Maintenance

**Summary**: 13 test files categorized — 5 Keep, 3 NEW, 5 Review (~70 line references)

| Test File | Action | Internal Functions Used |
|-----------|--------|------------------------|
| `test_subject_weights.jl` | ✅ Keep | `build_tpm_mapping` (utility, unchanged) |
| `test_observation_weights_emat.jl` | ✅ Keep | `build_tpm_mapping` |
| `test_phasetype.jl` | 🔄 Review | `build_fbmats_phasetype`, `build_phasetype_tpm_book`, `draw_samplepath_phasetype` |
| `test_phasetype_tvc.jl` | 🔄 Review | `_build_phasetype_from_markov`, `compute_forward_loglik` |
| `test_phasetype_surrogate.jl` | 🔄 Review | Multiple internal functions (20+ uses) |
| `test_mcem_infrastructure.jl` | ➕ NEW | `MCEMInfrastructure`, dispatch methods |
| `test_path_likelihood.jl` | ➕ NEW | `compute_surrogate_loglik` |

**Phase-Specific Updates**:
- Phases 1-2: New test file, import updates
- Phases 3-4: Build pattern → infrastructure struct, signature changes
- Phase 5: Full verification

---

## 🟡 Item #39: PhaseType MCEM Proposal Bias Investigation

**Priority**: 🔴 CRITICAL (Correctness Bug)  
**Status**: 🟡 IN PROGRESS — Bugs fixed but problem persists  
**Reference**: [MCEM_PHASETYPE_HANDOFF_20250609.md](../MCEM_PHASETYPE_HANDOFF_20250609.md)  
**Effort**: 4-8 hours remaining

### Problem Statement

MCEM with PhaseType proposal systematically biases h23 parameters in illness-death models:

| Parameter | True | Markov Est | PhaseType Est | PhaseType Error |
|-----------|------|------------|---------------|-----------------|
| shape_23 | 1.1 | 1.26 (14%) | 1.59 | **45%** |
| scale_23 | 0.12 | 0.094 (21%) | 0.071 | **41%** |

### Bugs Fixed (But Problem Persists)

Two bugs in `convert_expanded_path_to_censored_data` were fixed:
1. **Emission matrix**: Transition rows only allowed sampled phase, not all destination phases
2. **statefrom re-initialization**: Incorrectly set for censored rows

**Mystery**: Fixes don't change MCEM results at all.

### Investigation Hypotheses

1. Wrong code path (fix bypassed by MCEM flow)
2. Caching (stale data structures)
3. Sampling bug (paths themselves biased)
4. Parameter transform issue
5. Deterministic bug (same bias regardless of Monte Carlo)

---

## ✅ Item #27: Variance-Covariance API Unification (PARTIAL)

**Priority**: 🟡 HIGH  
**Status**: ✅ PARTIAL COMPLETE (2026-01-22)  
**Effort**: 12-18 hours

### Problem

Currently `fit()` returns `vcov=nothing` when constraints are present (including all phase-type models). The fitted model struct also has three separate vcov fields (`vcov`, `ij_vcov`, `jk_vcov`) which complicates post-processing.

### Completed (2026-01-22)

**Unified `vcov_type` API**:
- Removed `ij_vcov`, `jk_vcov` fields from `MultistateModelFitted`
- Added `vcov_type::Symbol` field to store which variance type was computed
- Added `vcov_type::Symbol=:ij` kwarg to `fit()` functions
- Created `_validate_vcov_type()` helper in `fit_common.jl`
- Simplified `get_vcov()` accessor (no `type` kwarg needed)
- Updated all 36+ test files

**VCoV Types Supported**:
| Type | Description | When to Use |
|------|-------------|-------------|
| `:ij` (default) | Information-Jackknife/Sandwich: $H^{-1} J H^{-1}$ | Robust, always works |
| `:model` | Model-based: $H^{-1}$ (inverse Fisher information) | When model is correctly specified |
| `:jk` | Jackknife (leave-one-out) | Cross-validation of variance |
| `:none` | Skip variance computation | Speed when vcov not needed |

### Remaining Work (Item #27-B)

**Parameters at Active Constraints** (not yet implemented):
- Detect parameters at active box constraints (within `BOUND_TOL` of bounds)
- Set variance to `NaN` for those parameters
- Emit warning listing affected parameters
- Add unit tests for constrained model variance

### API Changes

**Before:**
```julia
fitted = fit(model; compute_vcov=true, compute_ij_vcov=true, compute_jk_vcov=false)
vcov = fitted.vcov        # or fitted.ij_vcov, fitted.jk_vcov
vcov = get_vcov(fitted)   # returns model-based
vcov = get_vcov(fitted; type=:ij)
```

**After:**
```julia
fitted = fit(model; vcov_type=:ij)  # default
vcov = fitted.vcov        # single field
vcov_type = fitted.vcov_type  # :ij, :model, :jk, or :none
vcov = get_vcov(fitted)   # simple accessor
```

### Implementation Tasks (Completed)

| Task | Description | Status |
|------|-------------|--------|
| #27a | Remove `ij_vcov`, `jk_vcov` fields from `MultistateModelFitted` | ✅ Done |
| #27b | Add `vcov_type::Symbol=:ij` kwarg to `fit()` | ✅ Done |
| #27c | Implement `_compute_vcov_*` functions in fit files | ✅ Done |
| #27d | Detect parameters at active constraints, set variance to NaN | ⏳ TODO |
| #27e | Update `get_vcov()` accessor (remove `type` kwarg) | ✅ Done |
| #27f | Update all tests that check vcov fields | ✅ Done (36+ files) |
| #27g | Add tests for constrained model variance | ⏳ TODO |

### Key Files Modified

| File | Changes |
|------|---------|
| `src/types/model_structs.jl` | Removed `ij_vcov`, `jk_vcov` fields; added `vcov_type::Symbol` |
| `src/inference/fit_common.jl` | Added `_validate_vcov_type()` helper |
| `src/inference/fit_exact.jl` | Added `vcov_type` kwarg, `_compute_vcov_exact()` |
| `src/inference/fit_markov.jl` | Added `vcov_type` kwarg, `_compute_vcov_markov()` |
| `src/inference/fit_mcem.jl` | Added `vcov_type` kwarg, `_compute_vcov_mcem()`, updated MCEMConfig |
| `src/output/accessors.jl` | Simplified `get_vcov()` |
| `src/output/variance.jl` | Updated docstrings |

---

## ⏳ Item #28: Simulation Diagnostics Completion

**Priority**: 🟡 HIGH (Verification)  
**Status**: ⏳ PLANNED  
**Effort**: 16-24 hours

### Problem

Need to verify `simulate()` produces correct distributions for ALL 24 combinations:
- Family: exp, wei, gom, sp
- Effect: PH, AFT
- Covariates: none, fixed, TVC

Currently 8 TVC scenarios missing validation.

---

## ✅ Completed Work Summary

### Waves 1-3: Foundation (Complete)

| Wave | Focus | Items | Status |
|------|-------|-------|--------|
| 1 | Zombie code deletion | #1-4, #8, #9, #11, #13, #22, #23 | ✅ ~350 lines removed |
| 2 | Technical debt | #6, #10, #21 | ✅ Parameter structure simplified |
| 3 | Math correctness | #5, #15-18, #24 | ✅ Spline/penalty bugs fixed |

### Phase-Type Infrastructure (Complete)

| Item | Description | Date |
|------|-------------|------|
| #35 | PhaseType surrogate likelihood — forward algorithm with Schur caching | 2026-01-17 |
| #36 | PhaseType dt=0 bug — use hazards not normalized probabilities | 2026-01-18 |
| #37 | Infrastructure cleanup — split sampling.jl, add obstype constants | 2026-01-18 |

### Other Completed Items

| Item | Description | Date |
|------|-------------|------|
| #7 | Variance function audit (bug fixed, plan documented) | 2026-01-12 |
| #14 | make_constraints test coverage | 2026-01-12 |
| #25 | Natural-scale parameter migration (documentation fix) | 2026-01-12 |
| #26 | Eigenvalue ordering at covariate reference | 2026-01-14 |
| BUG-1 | Monotone spline constraint (test was flawed) | 2026-01-09 |
| BUG-2 | Phase-Type TPM eigendecomposition | 2026-01-10 |
| SQUAREM | Removed acceleration (unstable) | 2026-01-13 |

### Key Bug Resolutions

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| BUG-1: Monotone constraints | Test flawed (constant hazard) | Rewrote test with Weibull |
| BUG-2: PhaseType divergence | Covariate effects ignored | Fixed `build_phasetype_tpm_book` |
| Item #25: simulate() broken | Documentation inconsistency | Fixed docstrings |
| Item #36: dt=0 handling | Normalized probs vs hazards | Use Q[i,j] directly |

---

## Test Status

**Unit Tests**: 2127 passing  
**Long Tests**: 32/38 passing (6 spline MCEM failures)

### Failing Tests (All Spline MCEM Panel)

1. `sp_ph_panel_markov_nocov`
2. `sp_ph_panel_phasetype_nocov`
3. `sp_ph_panel_markov_fixed`
4. `sp_ph_panel_phasetype_fixed`
5. `sp_aft_panel_markov_nocov`
6. `sp_aft_panel_phasetype_nocov`

---

## Key Files Reference

| Purpose | Location |
|---------|----------|
| Hazard definitions | `src/construction/hazard_builders.jl` |
| Fitting dispatch | `src/inference/fit_common.jl` |
| MCEM implementation | `src/inference/fit_mcem.jl` |
| Markov sampling | `src/inference/sampling_markov.jl` |
| PhaseType sampling | `src/inference/sampling_phasetype.jl` |
| Spline utilities | `src/utilities/spline_utils.jl` |
| MCEM infrastructure | `src/mcem/infrastructure.jl` |

---

## Validation Checklist

### ✅ Complete
- [x] Parameters stored on natural scale
- [x] Penalty is quadratic P(β) = (λ/2) β^T S β
- [x] Box constraints enforce positivity
- [x] Phase-type parameter naming correct
- [x] `simulate()` produces correct transitions
- [x] `get_parameters(; scale=:natural)` works

### ⏳ Pending
- [ ] Spline MCEM boundary coefficients correct
- [ ] Variance-covariance always returned
- [ ] All 24 simulation scenarios validated

---

## Change Log (Recent)

| Date | Changes |
|------|---------|
| 2026-01-21 | Streamlined document — removed verbose session logs, summarized completed work |
| 2026-01-18 | Item #36 complete (dt=0 bug), Item #37 complete (infrastructure cleanup) |
| 2026-01-17 | Item #35 complete (PhaseType forward algorithm with Schur caching) |
| 2026-01-15 | Wave 6 complete (PhaseType covariate bug fix), spline MCEM bug discovered |
| 2026-01-14 | Wave 5 complete (longtest coverage), Item #26 added |
| 2026-01-13 | SQUAREM removed |
| 2026-01-12 | Item #25 resolved, Item #7 audited, Item #14 complete |
| 2026-01-10 | Item #21 complete, BUG-2 resolved, identifiability constraints added |
| 2026-01-09 | BUG-1 resolved, Waves 1-3 complete |
