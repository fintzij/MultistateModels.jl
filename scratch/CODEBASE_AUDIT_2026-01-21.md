# MultistateModels.jl Codebase Audit Report

**Date**: 2026-01-21  
**Branch**: `penalized_splines`  
**Auditor**: Adversarial Code Review Agent  
**Status**: Phase 1 + Phase 2 Complete

---

## Executive Summary

Comprehensive two-phase audit of the `src/` directory identified **54 issues** across severity levels:

| Severity | Phase 1 | Phase 2 | Total |
|----------|---------|---------|-------|
| Critical (P0) | 5 | 6 | 11 |
| High (P1) | 7 | 5 | 12 |
| Medium (P2) | 10 | 8 | 18 |
| Low (P3) | 8 | 5 | 13 |
| **Total** | **30** | **24** | **54** |

### Phase 1 Findings (Surface Analysis)
- **Error handling**: Silent catch blocks, AD-incompatible exception handling
- **Magic numbers**: Hardcoded tolerances scattered throughout
- **Code duplication**: Variance computation, optimization setup
- **Validation gaps**: Incomplete input validation, inconsistent bounds checking
- **Dead code**: Functions defined but never called

### Phase 2 Findings (Deep Analysis)
- **Object lifecycle bugs**: Partial construction failures, memory leaks
- **Concurrency hazards**: Thread safety violations, race conditions
- **Mathematical edge cases**: Boundary condition failures, numerical instability
- **Implicit invariants**: Undocumented assumptions that can be violated
- **Round-trip issues**: Parameter transformations that lose precision

**Risk Assessment**: **HIGH**. While core likelihood computations are mathematically sound, the system is fragile under:
- Concurrent access
- Edge-case inputs
- Long-running sessions
- Manual parameter manipulation

---

## Issue Tracking

### Legend
- ⬜ Not Started
- 🟡 In Progress  
- ✅ Complete
- 🔴 Blocked

### Issue ID Naming Convention
- `C*` = Critical (P0)
- `H*` = High (P1)
- `M*` = Medium (P2)
- `L*` = Low (P3)
- Suffix `_P1` = Phase 1 finding, `_P2` = Phase 2 finding

---

# PHASE 1 ISSUES (Surface Analysis)

---

## Critical Issues (P0 - Must Fix Before Release)

### C1_P1. `_clamp_to_bounds!` Defined But Never Called
| Field | Value |
|-------|-------|
| **File** | `src/inference/fit_common.jl` L171-200 |
| **Status** | ⬜ Not Started |
| **Owner** | TBD |
| **Impact** | Dead code; potential confusion; maintenance burden |

**Description**: Function `_clamp_to_bounds!` is defined with full documentation but has zero call sites. The docstring says "This should NOT be needed after Ipopt optimization" suggesting it was removed from use but not deleted.

**Action Items**:
- [ ] Search for any dynamic/eval-based calls that might invoke this
- [ ] If truly dead, delete the function
- [ ] If needed, add calls at appropriate locations (post-SQUAREM extrapolation?)
- [ ] Add unit test if keeping

---

### C2_P1. `rectify_coefs!` Called With Uncertainty Marker
| Field | Value |
|-------|-------|
| **File** | `src/inference/fit_exact.jl` L163, L229; `src/inference/fit_mcem.jl` L807 |
| **Status** | ⬜ Not Started |
| **Owner** | TBD |
| **Impact** | Unknown correctness; modifies optimized parameters post-hoc |

**Description**: The comment "# rectify spline coefs - CHECK THIS!!!!" precedes calls to `rectify_coefs!`. This function performs a round-trip transformation on spline parameters after optimization, which may or may not be mathematically necessary.

**Action Items**:
- [ ] Document the mathematical justification for rectification
- [ ] Verify that rectification is idempotent: `rectify(rectify(x)) == rectify(x)`
- [ ] Add unit tests: compare log-likelihood before/after rectification
- [ ] Determine if this is compensating for an optimizer bug or a genuine mathematical requirement
- [ ] Remove "CHECK THIS!!!!" comment once resolved
- [ ] If unnecessary, remove the calls and function

---

### C3_P1. Silent Catch Blocks in smoothing_selection.jl
| Field | Value |
|-------|-------|
| **File** | `src/inference/smoothing_selection.jl` L320-325, L362-368, L465-472, L503-510, L833-840, L1724-1730, L1764-1770, L1859-1865 |
| **Status** | ⬜ Not Started |
| **Owner** | TBD |
| **Impact** | Errors silently swallowed; debugging impossible; incorrect results possible |

**Description**: At least 8 `try-catch` blocks catch exceptions and return fallback values (typically zeros or identity matrices) without logging the original error. Example:
```julia
delta_i = try
    H_lambda_inv * g_i
catch e
    zeros(n_params)  # Silent failure - no logging!
end
```

**Action Items**:
- [ ] Audit all catch blocks in smoothing_selection.jl
- [ ] Add `@warn` or `@debug` logging for each caught exception
- [ ] Consider whether fallback values are mathematically justified
- [ ] Document expected exceptions vs unexpected ones
- [ ] Add monitoring/counters for exception frequency
- [ ] Create unit tests that trigger each catch path

---

### C4_P1. AD-Incompatible try-catch Pattern
| Field | Value |
|-------|-------|
| **File** | `src/inference/smoothing_selection.jl` (multiple), `src/hazard/spline.jl` L706-715, L1283-1295 |
| **Status** | ⬜ Not Started |
| **Owner** | TBD |
| **Impact** | ForwardDiff Dual types lose gradient information; silent gradient failures |

**Description**: When code inside a try block is being differentiated via ForwardDiff and throws an exception, the catch block returns a Float64 value. This breaks the AD type chain:
```julia
# During AD, if inner computation fails:
result = try
    complex_dual_computation(x)  # Returns Dual{...}
catch
    0.0  # Returns Float64 - type mismatch!
end
```

**Action Items**:
- [ ] Identify all try-catch blocks in AD code paths
- [ ] For necessary catches, return `zero(eltype(input))` not `0.0`
- [ ] Consider using `@something` or similar patterns instead of try-catch
- [ ] Add AD-specific tests that verify gradients through error paths
- [ ] Document which functions are AD-safe vs AD-unsafe

---

### C5_P1. Phase-Type Structure Validation Inconsistency
| Field | Value |
|-------|-------|
| **File** | `src/types/hazard_specs.jl` L139-141 vs `src/construction/hazard_constructors.jl` L114 |
| **Status** | ⬜ Not Started |
| **Owner** | TBD |
| **Impact** | User confusion; inconsistent API behavior |

**Description**: 
- `PhaseTypeHazard` inner constructor accepts: `(:unstructured, :sctp, :sctp_increasing, :sctp_decreasing)`
- `Hazard()` constructor validates: `(:unstructured, :sctp)` only
- Documentation mentions all four options

**Action Items**:
- [ ] Decide canonical set of allowed structure values
- [ ] Align validation in both locations
- [ ] Update docstrings to match implementation
- [ ] Add unit tests for each structure option
- [ ] Consider deprecating unused options

---

## High Severity Issues (P1 - Fix This Sprint)

### H1_P1. Hardcoded Magic Numbers Throughout
| Field | Value |
|-------|-------|
| **Files** | Multiple - see list below |
| **Status** | ⬜ Not Started |
| **Owner** | TBD |
| **Impact** | Unmaintainable; users can't adjust; fails on edge cases |

**Locations**:
| File | Line | Value | Purpose |
|------|------|-------|---------|
| `spline.jl` | 102 | `1e-10`, `1e-15` | Symmetry tolerance |
| `spline.jl` | 1188 | `1e-6` | CDF grid start |
| `spline.jl` | 1220 | `1e-12` | CDF interpolation tolerance |
| `time_transform.jl` | 152 | `1e-6` | Gompertz zero-shape threshold |
| `fit_common.jl` | 131-140 | Various | Ipopt tolerances |
| `fit_common.jl` | 190 | `1e-8` | Bounds epsilon buffer |
| `fit_common.jl` | 316 | `1e-6` | Active constraint tolerance |
| `parameters.jl` | 207-210 | `1e-6` | Bounds validation tolerance |
| `loglik_markov.jl` | 473 | `eps()` | Transition probability floor |

**Action Items**:
- [ ] Create/update `src/utilities/constants.jl` with named constants
- [ ] Group constants by domain: `SPLINE_*`, `OPTIMIZATION_*`, `NUMERICAL_*`
- [ ] Add docstrings explaining each constant's purpose and safe ranges
- [ ] Replace all hardcoded values with named constants
- [ ] Consider making critical tolerances user-configurable
- [ ] Add validation tests at boundary tolerance values

---

### H2_P1. Inconsistent Tolerance Handling in vcov Computation
| Field | Value |
|-------|-------|
| **File** | `src/inference/fit_exact.jl` L185, `src/inference/fit_markov.jl` L84 |
| **Status** | ⬜ Not Started |
| **Owner** | TBD |
| **Impact** | Inconsistent numerical behavior; potential incorrect standard errors |

**Description**: The vcov computation uses different formulas:
- fit_exact: `atol = (log(length(samplepaths)) * length(sol.u))^-2`
- fit_markov: `atol = (log(nsubj) * length(sol.u))^-2`

Additionally, the "near-zero zeroing" step uses different tolerances:
- fit_exact: `atol = sqrt(eps(Float64)), rtol = sqrt(eps(Float64))`
- fit_markov: `atol = eps(Float64)` (no rtol)

**Action Items**:
- [ ] Document the mathematical justification for the `(log(n)*p)^-2` formula
- [ ] Decide on a single consistent formula
- [ ] Extract to shared function `_compute_vcov_tolerance(n_obs, n_params)`
- [ ] Standardize near-zero zeroing across all fit functions
- [ ] Add unit tests comparing vcov from different code paths on same data

---

### H3_P1. Weight Validation Gap
| Field | Value |
|-------|-------|
| **File** | `src/construction/model_assembly.jl` L191-197 |
| **Status** | ⬜ Not Started |
| **Owner** | TBD |
| **Impact** | Silent incorrect results with invalid weights |

**Description**: `check_SubjectWeights` and `check_ObservationWeights` are called but:
1. No validation that weights are non-negative
2. No validation that weights are finite (not NaN/Inf)
3. Zero weights could cause division issues

**Action Items**:
- [ ] Add validation: all weights must be non-negative
- [ ] Add validation: all weights must be finite
- [ ] Consider warning if any weights are exactly zero
- [ ] Document weight semantics in docstrings
- [ ] Add unit tests with edge-case weights

---

### H4_P1. Missing Bounds Validation in rebuild_parameters
| Field | Value |
|-------|-------|
| **File** | `src/utilities/parameters.jl` L168 |
| **Status** | ⬜ Not Started |
| **Owner** | TBD |
| **Impact** | Invalid parameters can silently pass validation |

**Description**: When `model.bounds` is `nothing` (possible for models created before bounds were added), bounds validation silently skips:
```julia
if validate_bounds && hasproperty(model, :bounds) && !isnothing(model.bounds)
```

**Action Items**:
- [ ] Decide: should bounds always exist? If yes, make field non-optional
- [ ] If bounds can be missing, generate them on-demand in validation
- [ ] Add warning when skipping validation due to missing bounds
- [ ] Add migration path for old models without bounds

---

### H5_P1. Numerical Instability in loglik_markov
| Field | Value |
|-------|-------|
| **File** | `src/likelihood/loglik_markov.jl` L473 |
| **Status** | ⬜ Not Started |
| **Owner** | TBD |
| **Impact** | Potential numerical errors in transition probabilities |

**Description**: 
```julia
q[r,r] = maximum([1 - exp(logsumexp(q[r, Not(r)])), eps()])
```
Issues:
1. `eps()` is architecture-dependent; should use `eps(eltype(q))`
2. The log-sum-exp → exp → 1-x chain can amplify numerical errors
3. No check that result is actually a valid probability (≤1)

**Action Items**:
- [ ] Replace `eps()` with `eps(eltype(q))` or named constant
- [ ] Add assertion that diagonal elements are in valid range
- [ ] Consider using log-space computation throughout
- [ ] Add numerical stability tests with extreme transition rates

---

### H6_P1. Backward Compatibility Shims Without Deprecation
| Field | Value |
|-------|-------|
| **File** | `src/types/model_structs.jl` L295-320 |
| **Status** | ⬜ Not Started |
| **Owner** | TBD |
| **Impact** | Maintenance burden; confusing API |

**Description**: `markovsurrogate` and `phasetype_surrogate` property accessors are marked "DEPRECATED" in comments but:
1. No `@deprecate` macro used
2. No warning issued when accessed
3. No removal timeline documented

**Action Items**:
- [ ] Add `Base.depwarn()` calls in the getproperty methods
- [ ] Document removal version in CHANGELOG
- [ ] Search codebase for internal uses and migrate
- [ ] Add tests that verify deprecation warnings fire
- [ ] Schedule removal for next major version

---

### H7_P1. Thread-Local Storage Without Cleanup
| Field | Value |
|-------|-------|
| **File** | `src/types/data_containers.jl` L560-570 |
| **Status** | ⬜ Not Started |
| **Owner** | TBD |
| **Impact** | Memory leak in long-running processes |

**Description**: 
```julia
const TVC_INTERVAL_WORKSPACES = Dict{Int, TVCIntervalWorkspace}()
const TVC_WORKSPACE_LOCK = ReentrantLock()
```
Workspaces are created per-thread but never cleaned up.

**Action Items**:
- [ ] Add `clear_tvc_workspaces!()` function
- [ ] Document memory management in user guide
- [ ] Consider using WeakRef or similar for automatic cleanup
- [ ] Add finalizer if workspaces hold significant memory
- [ ] Test memory usage in long-running scenarios

---

## Medium Severity Issues (P2 - Fix This Month)

### M1_P1. Duplicated Logic: vcov Computation
| Field | Value |
|-------|-------|
| **Files** | `fit_exact.jl` L175-188, `fit_markov.jl` L76-87 |
| **Status** | ⬜ Not Started |
| **Action** | Extract shared function `_compute_vcov_from_hessians` |

### M2_P1. Inconsistent isnothing vs === nothing Usage
| Field | Value |
|-------|-------|
| **Files** | Throughout codebase |
| **Status** | ⬜ Not Started |
| **Action** | Standardize on `isnothing()` everywhere |

### M3_P1. Complex Control Flow in _fit_exact
| Field | Value |
|-------|-------|
| **File** | `src/inference/fit_exact.jl` |
| **Status** | ⬜ Not Started |
| **Action** | Decompose into `_setup_optimization`, `_run_optimization`, `_compute_variance`, `_package_results` |

### M4_P1. Unused Function Arguments
| Field | Value |
|-------|-------|
| **File** | `src/construction/spline_builder.jl` L353 |
| **Status** | ⬜ Not Started |
| **Action** | Audit `clamp_zeros` usage; consider removing parameter |

### M5_P1. Incomplete Input Validation in Hazard Constructor
| Field | Value |
|-------|-------|
| **File** | `src/construction/hazard_constructors.jl` L113-116 |
| **Status** | ⬜ Not Started |
| **Action** | Align validation with documented options |

### M6_P1. No Validation of Emission Matrix Row Sums
| Field | Value |
|-------|-------|
| **File** | `src/construction/model_assembly.jl` L135-145 |
| **Status** | ⬜ Not Started |
| **Action** | Add validation that rows sum to ≤ 1.0 |

### M7_P1. Error Messages Without Actionable Guidance
| Field | Value |
|-------|-------|
| **Files** | Multiple |
| **Status** | ⬜ Not Started |
| **Action** | Audit error messages; add "did you mean..." suggestions |

### M8_P1. TimeTransformContext Type Instability
| Field | Value |
|-------|-------|
| **File** | `src/types/hazard_metadata.jl` L168-175 |
| **Status** | ⬜ Not Started |
| **Action** | Consider wrapper type or refactor to avoid Union return |

### M9_P1. ReConstructor Buffer Never Used
| Field | Value |
|-------|-------|
| **File** | `src/utilities/reconstructor.jl` L86 |
| **Status** | ⬜ Not Started |
| **Action** | Remove `_buffer` field or use it |

### M10_P1. Spline Extrapolation Method Inconsistency
| Field | Value |
|-------|-------|
| **Files** | `spline_builder.jl`, `hazard_constructors.jl` |
| **Status** | ⬜ Not Started |
| **Action** | Clarify and document "constant" vs "flat" distinction |

---

## Low Severity Issues (P3 - Backlog)

| ID | Issue | File | Action |
|----|-------|------|--------|
| L1_P1 | Overly long function signatures | `fit_mcem.jl` | Create config structs |
| L2_P1 | Inconsistent boolean parameter naming | Multiple | Standardize naming convention |
| L3_P1 | @inline overuse | Multiple | Profile and remove unnecessary hints |
| L4_P1 | Redundant type assertions | `hazard_structs.jl` | Use `Int` instead of `Int64` |
| L5_P1 | Docstring/implementation drift | Multiple | Audit all docstrings |
| L6_P1 | Global constants not exported | Multiple | Export or document access pattern |
| L7_P1 | Deep nesting in likelihood functions | `loglik_exact.jl` | Refactor to reduce nesting |
| L8_P1 | Inconsistent use of @view | Multiple | Standardize slice handling |

---

# PHASE 2 ISSUES (Deep Analysis)

Phase 2 traced object lifecycles, call trees, data flows, concurrency patterns, and mathematical edge cases.

---

## Critical Issues (P0) - Phase 2

### C6_P2. Partial Construction Failure Leaves Model Inconsistent
| Field | Value |
|-------|-------|
| **File** | `src/construction/multistatemodel.jl` L260-270 |
| **Status** | ⬜ Not Started |
| **Owner** | TBD |
| **Impact** | Data corruption; inconsistent model state |

**Description**: In `multistatemodel()`, if `initialize_parameters!` succeeds but `initialize_surrogate!` fails, the function returns a partially constructed model:
```julia
model = _assemble_model(mode, process, components, model_surrogate, modelcall)

if initialize
    initialize_parameters!(model; constraints = constraints)  # Can succeed
end

if fit_surrogate && resolved_surrogate in (:markov, :phasetype)
    initialize_surrogate!(model; ...)  # If this throws, model is in inconsistent state
end

return model  # Returned with partial initialization!
```

**Action Items**:
- [ ] Wrap initialization steps in transaction-like pattern
- [ ] On failure, rollback model to pre-initialized state
- [ ] Consider returning `Result{Model, Error}` type
- [ ] Add test: verify model state after partial failure
- [ ] Document which initialization steps are atomic vs composite

---

### C7_P2. TOCTOU Race in Thread-Local Workspace Creation
| Field | Value |
|-------|-------|
| **File** | `src/types/data_containers.jl` L555-565 |
| **Status** | ⬜ Not Started |
| **Owner** | TBD |
| **Impact** | Thread safety violation; potential memory corruption |

**Description**: The workspace creation has a time-of-check-to-time-of-use race:
```julia
function get_tvc_workspace()::TVCIntervalWorkspace
    tid = Threads.threadid()
    ws = get(TVC_INTERVAL_WORKSPACES, tid, nothing)  # Check OUTSIDE lock
    if isnothing(ws)
        lock(TVC_WORKSPACE_LOCK) do
            ws = get!(TVC_INTERVAL_WORKSPACES, tid) do  # Use INSIDE lock
                TVCIntervalWorkspace(200)
            end
        end
    end
    return ws
end
```
Thread A and B can both see `nothing`, both enter lock, B overwrites A's workspace.

**Action Items**:
- [ ] Move entire `get` + `get!` inside the lock
- [ ] OR use `@lock_once` pattern for initialization
- [ ] OR use ConcurrentDict from Concurrent.jl
- [ ] Add concurrent stress test with 64+ threads
- [ ] Same fix needed for `_SAMPLING_WORKSPACES` in `sampling_core.jl`

---

### C8_P2. Zero Hazard Rate Causes Infinite Loop in Simulation
| Field | Value |
|-------|-------|
| **File** | `src/simulation/simulate.jl` L195 |
| **Status** | ⬜ Not Started |
| **Owner** | TBD |
| **Impact** | Application hang; denial of service |

**Description**: When all hazard rates from a state are zero:
```julia
@inline function _exponential_jump_time(u::Float64, cuminc::Float64, total_rate::Float64, interval_len::Float64)
    remaining_survival = 1.0 - cuminc
    if remaining_survival <= _DELTA_U || total_rate <= 0.0
        return interval_len  # Returns interval_len
    end
    # ...
end
```
This returns `interval_len`, which in the simulation loop may not cause state change, leading to infinite looping in absorbing-state-like scenarios.

**Action Items**:
- [ ] Validate total hazard rate > 0 before simulation loop
- [ ] Add iteration limit as safety guard
- [ ] Throw informative error when stuck: "No transitions possible from state X"
- [ ] Add timeout parameter to `simulate()`
- [ ] Add test with all-zero hazards

---

### C9_P2. Concurrent `fit()` on Same Model is Unsafe
| Field | Value |
|-------|-------|
| **Files** | `src/inference/fit_exact.jl`, `fit_markov.jl`, `fit_mcem.jl` |
| **Status** | ⬜ Not Started |
| **Owner** | TBD |
| **Impact** | Data corruption; undefined behavior |

**Description**: No locking prevents concurrent mutation. Multiple threads calling `fit(model)` simultaneously will:
1. Race on `initialize_parameters!` if called
2. Race on `model.parameters` updates
3. Race on `model.surrogate` updates

**Action Items**:
- [ ] Document that `fit()` is NOT thread-safe
- [ ] Add `@atomic` markers on mutable fields
- [ ] Consider adding `ReentrantLock` to `MultistateModel`
- [ ] OR create `SafeModel` wrapper with locking
- [ ] Add test: verify warning/error on concurrent fit

---

### C10_P2. `model.bounds` Used Without Null Check
| Field | Value |
|-------|-------|
| **File** | `src/utilities/parameters.jl` L167, `src/inference/fit_exact.jl` L75 |
| **Status** | ⬜ Not Started |
| **Owner** | TBD |
| **Impact** | NullPointerException equivalent; crash |

**Description**: Code assumes `model.bounds` is always present:
```julia
lb, ub = model.bounds.lb, model.bounds.ub  # No null check!
```
While `bounds` should always be set for new models, edge cases exist:
- Models deserialized from old versions
- Manual construction bypassing `multistatemodel()`

**Action Items**:
- [ ] Add `@assert !isnothing(model.bounds)` at fit entry
- [ ] OR generate bounds on-demand if missing
- [ ] Document bounds as required invariant
- [ ] Add migration guide for old serialized models
- [ ] Add test: fitting model with missing bounds

---

### C11_P2. Thread-Local Workspaces Never Freed (Memory Leak)
| Field | Value |
|-------|-------|
| **Files** | `src/types/data_containers.jl` L550, `src/inference/sampling_core.jl` L162 |
| **Status** | ⬜ Not Started |
| **Owner** | TBD |
| **Impact** | Memory leak in long-running processes |

**Description**: Two global workspace dictionaries accumulate entries:
- `TVC_INTERVAL_WORKSPACES` - Thread-local TVC workspaces
- `_SAMPLING_WORKSPACES` - Thread-local sampling workspaces

With 64 threads, each workspace holding ~200 intervals × 8 bytes × multiple vectors, memory accumulates without cleanup mechanism.

**Action Items**:
- [ ] Add `clear_all_workspaces!()` exported function
- [ ] Add finalizers to workspaces if possible
- [ ] Document memory management in user guide
- [ ] Consider using `WeakRef` or `ConcreteRCache`
- [ ] Add memory usage test: 1000 fit() calls, measure memory

---

## High Severity Issues (P1) - Phase 2

### H8_P2. Spline I-Spline Transform Condition Number Not Checked
| Field | Value |
|-------|-------|
| **File** | `src/hazard/spline.jl` L1440-1520 (rectify_coefs!) |
| **Status** | ⬜ Not Started |
| **Owner** | TBD |
| **Impact** | Numerical instability; silent precision loss |

**Description**: For monotone splines, the transformation is:
```
coefs = L * ests   (L = I-spline transform matrix)
ests = L⁻¹ * coefs
```
If L is ill-conditioned (possible with poorly-placed knots), the inverse loses precision. No condition number check exists.

**Action Items**:
- [ ] Add `cond(L)` check in `build_ispline_transform_matrix`
- [ ] Warn if condition number > 1e10
- [ ] Consider using pseudoinverse for ill-conditioned cases
- [ ] Add test with pathological knot placement
- [ ] Document recommended knot spacing for stability

---

### H9_P2. Weibull Hazard Infinite at t=0 for Shape < 1
| Field | Value |
|-------|-------|
| **File** | `src/hazard/generators.jl` L69-80 |
| **Status** | ⬜ Not Started |
| **Owner** | TBD |
| **Impact** | NaN/Inf in likelihood; optimization failure |

**Description**: Generated Weibull hazard computes:
```julia
haz *= t^(shape - 1)  # When shape < 1 and t = 0: 0^negative = Inf
```
While mathematically correct (Weibull with κ<1 has infinite hazard at t=0), this causes numerical issues.

**Action Items**:
- [ ] Add guard: `if t == 0 && shape < 1: return _LARGE_HAZARD`
- [ ] Document mathematical behavior in docstring
- [ ] Consider alternative parameterization avoiding singularity
- [ ] Add test: Weibull with shape=0.5 at t=0
- [ ] Ensure cumulative hazard handles this correctly

---

### H10_P2. Gompertz Overflow for Large t
| Field | Value |
|-------|-------|
| **File** | `src/hazard/generators.jl` L170-200 |
| **Status** | ⬜ Not Started |
| **Owner** | TBD |
| **Impact** | Inf overflow; optimization failure |

**Description**: Gompertz hazard: `h(t) = rate * exp(shape * t)`. For large t and positive shape, this overflows to Inf.

**Action Items**:
- [ ] Add overflow guard using log-space computation
- [ ] Warn when parameters suggest overflow risk
- [ ] Consider clamping shape parameter during optimization
- [ ] Add test with t > 1000 and shape > 0.1
- [ ] Document safe parameter ranges

---

### H11_P2. TPM Negative dt Not Validated
| Field | Value |
|-------|-------|
| **File** | `src/types/data_containers.jl` L222-260 |
| **Status** | ⬜ Not Started |
| **Owner** | TBD |
| **Impact** | Invalid transition probability matrices |

**Description**: `compute_tpm_from_schur!` doesn't validate `dt >= 0`:
```julia
function compute_tpm_from_schur!(P::Matrix{Float64}, cached::CachedSchurDecomposition, dt::Float64)
    if dt == 0.0
        # ... identity
    else
        cached.E_work .= exp(cached.T * dt)  # If dt < 0, invalid TPM!
        # ...
    end
end
```

**Action Items**:
- [ ] Add `@assert dt >= 0.0 "Negative time interval"` 
- [ ] Document in function signature that dt must be non-negative
- [ ] Add test: verify error on negative dt
- [ ] Audit all callers to ensure dt is always non-negative

---

### H12_P2. Phase-Type Round-Trip May Lose Precision
| Field | Value |
|-------|-------|
| **Files** | `src/output/accessors.jl` L380-450, `src/utilities/parameters.jl` L480-520 |
| **Status** | ⬜ Not Started |
| **Owner** | TBD |
| **Impact** | Parameters change after get/set cycle |

**Description**: For phase-type models:
- `get_parameters(model)` computes user-facing (λ, μ) from expanded params
- `set_parameters!(model, params)` expects same format
- Mapping between user-facing and internal uses string manipulation
- If mapping differs between get and set, round-trip fails

**Action Items**:
- [ ] Add round-trip test: `get → set → get` should be identical
- [ ] Use deterministic mapping (not string parsing)
- [ ] Document exact mapping algorithm
- [ ] Consider storing user-facing params as cache
- [ ] Add tolerance for floating-point comparison in tests

---

## Medium Severity Issues (P2) - Phase 2

### M11_P2. TPM Row Sum Invariant Not Verified
| Field | Value |
|-------|-------|
| **File** | `src/likelihood/loglik_markov.jl`, `src/types/data_containers.jl` |
| **Status** | ⬜ Not Started |
| **Action** | Add debug assertion: `@assert all(isapprox.(sum(P, dims=2), 1.0, atol=1e-10))` |

### M12_P2. Hazard Positivity Not Verified  
| Field | Value |
|-------|-------|
| **File** | `src/hazard/evaluation.jl` |
| **Status** | ⬜ Not Started |
| **Action** | Add debug assertion: `@assert haz >= 0.0 "Hazard must be non-negative"` |

### M13_P2. Subject Indices Contiguity Assumed But Not Validated
| Field | Value |
|-------|-------|
| **File** | `src/construction/model_assembly.jl` |
| **Status** | ⬜ Not Started |
| **Action** | Add validation in `get_subjinds()` that indices are contiguous |

### M14_P2. Single Subject JK Variance = NaN (No Warning)
| Field | Value |
|-------|-------|
| **File** | `src/output/variance.jl` |
| **Status** | ⬜ Not Started |
| **Action** | Add warning when n_subjects < 2 and JK variance requested |

### M15_P2. All-Zero Weights Not Validated
| Field | Value |
|-------|-------|
| **File** | `src/construction/model_assembly.jl` |
| **Status** | ⬜ Not Started |
| **Action** | Add validation: `sum(weights) > 0` to prevent division by zero |

### M16_P2. Fitted Model vcov Dependencies Not Documented
| Field | Value |
|-------|-------|
| **File** | `src/inference/fit_exact.jl` |
| **Status** | ⬜ Not Started |
| **Action** | Document that `vcov=nothing` implies `subject_gradients=nothing` |

### M17_P2. TPMCache Invalidation on Hazard Modification
| Field | Value |
|-------|-------|
| **File** | `src/types/data_containers.jl` |
| **Status** | ⬜ Not Started |
| **Action** | Add version counter to invalidate cache when hazards change |

### M18_P2. Global Threading Config Race Condition
| Field | Value |
|-------|-------|
| **File** | `src/types/infrastructure.jl` L261 |
| **Status** | ⬜ Not Started |
| **Action** | Use thread-safe Atomic for `_GLOBAL_THREADING_CONFIG` |

---

## Low Severity Issues (P3) - Phase 2

| ID | Issue | File | Action |
|----|-------|------|--------|
| L9_P2 | No cleanup function exported for workspaces | `data_containers.jl` | Add `MultistateModels.clear_caches!()` |
| L10_P2 | Absorbing state model not validated | `multistatemodel.jl` | Add check for at least one absorbing state |
| L11_P2 | Boundary epsilon constants duplicated | Multiple | Unify into `constants.jl` |
| L12_P2 | Debug assertions disabled by default | N/A | Add `MSM_DEBUG_ASSERTIONS` env var |
| L13_P2 | No structured logging for errors | Multiple | Consider Logging.jl integration |

---

## Recommended New Tests

| Test Category | Description | Priority |
|---------------|-------------|----------|
| Edge case weights | Zero, negative, very large weights | High |
| Numerical stability | Extreme parameter values, near-singular Hessians | High |
| Thread safety | Concurrent fit() calls | Medium |
| Memory leaks | Repeated fit() calls | Medium |
| AD correctness | Gradient verification against finite differences | High |
| Boundary values | Parameters at bounds, t=0 evaluation | High |
| Round-trip tests | rectify_coefs! idempotence | High |
| Error path coverage | Trigger each catch block | Medium |

---

## Fix Order Dependencies

```
Phase 1 (Foundation) - Week 1:
├── H1: Consolidate magic numbers
├── M2: Standardize isnothing
└── L6: Export/document constants

Phase 2 (Error Handling) - Week 2:
├── C3: Add catch block logging
├── C4: Make catches AD-aware
└── M7: Improve error messages

Phase 3 (Refactoring) - Week 3:
├── M1: Extract shared vcov computation
├── M3: Decompose _fit_exact
└── L1: Create config structs

Phase 4 (Validation) - Week 4:
├── C5: Align phase-type validation
├── H3: Add weight validation
├── H4: Robust bounds validation
└── M6: Emission matrix validation

Phase 5 (Cleanup) - Week 5:
├── C1: Remove dead _clamp_to_bounds!
├── C2: Resolve rectify_coefs!
├── M4: Remove unused parameters
├── M9: Remove unused buffer
└── H6: Add deprecation warnings

Phase 6 (Polish) - Week 6:
├── H7: Add workspace cleanup
├── M8: Fix type instability
├── M10: Document extrapolation
└── All L* items
```

---

## Metrics

| Metric | Value |
|--------|-------|
| Critical Issues | 5 |
| High Issues | 7 |
| Medium Issues | 10 |
| Low Issues | 8 |
| **Total** | **30** |
| Estimated Remediation | 8 weeks |

---

# INVARIANTS & CONSTRAINTS

These are system invariants that must hold. Add runtime assertions where marked.

## Model Invariants
| Invariant | Location | Assertion Needed |
|-----------|----------|------------------|
| `model.bounds` must not be nothing after construction | `multistatemodel.jl` | Yes |
| At least one absorbing state exists | `model_assembly.jl` | Yes |
| Subject indices are contiguous | `get_subjinds()` | Yes |
| `length(model.parameters) == model.totpars` | `MultistateModel` | Yes |

## Hazard Invariants
| Invariant | Location | Assertion Needed |
|-----------|----------|------------------|
| `hazard(t) >= 0` for all t >= 0 | `evaluation.jl` | Debug mode |
| `cumulative_hazard(t) >= 0` for all t >= 0 | `evaluation.jl` | Debug mode |
| Spline knots are strictly increasing | `spline_builder.jl` | Yes |

## Transition Probability Invariants
| Invariant | Location | Assertion Needed |
|-----------|----------|------------------|
| `sum(P[i,:]) ≈ 1.0` for each row | TPM computations | Debug mode |
| `P[i,j] >= 0` for all entries | TPM computations | Debug mode |
| `dt >= 0` for all time intervals | `compute_tpm_from_schur!` | Yes |

## Data Invariants
| Invariant | Location | Assertion Needed |
|-----------|----------|------------------|
| `tstart <= tstop` for each row | Data validation | Yes |
| `obstype` in valid range for model | `ExactData`, `MPanelData` | Yes |
| Weights >= 0 and sum(weights) > 0 | Data validation | Yes |

---

# EDGE CASES REQUIRING TESTS

## Numerical Edge Cases
- [ ] Weibull with shape < 1 at t = 0
- [ ] Gompertz with large t (>1000 time units)
- [ ] Splines with closely-spaced knots (< 1e-6 apart)
- [ ] Parameters exactly at bounds during optimization
- [ ] Empty covariate data (no covariates specified)

## Structural Edge Cases
- [ ] Single observation per subject
- [ ] All subjects observed at exact same times
- [ ] Model with only one transient state
- [ ] Model where all paths go to same absorbing state

## Concurrency Edge Cases
- [ ] 64+ threads calling `fit()` on different models
- [ ] Concurrent `simulate()` calls
- [ ] Workspace creation under thread contention

## Memory Edge Cases
- [ ] 1000 sequential `fit()` calls (memory accumulation)
- [ ] Very large model (>100 transitions, >50 states)
- [ ] Very long observation periods (t > 10000)

---

# TESTING RECOMMENDATIONS

## Missing Test Coverage (High Priority)
1. Partial construction failure recovery
2. Thread-local workspace cleanup
3. TOCTOU race conditions
4. Parameter round-trip identity for phase-type
5. TPM row sum validation under numerical stress

## Recommended New Test Files
```
unit/
├── test_invariants.jl           # Runtime invariant assertions
├── test_concurrency.jl          # Multi-threaded stress tests
├── test_numerical_edge_cases.jl # Boundary condition tests
└── test_memory.jl               # Memory leak detection
```

---

# REMEDIATION ROADMAP

## Week 1-2: Critical Safety (P0)
| Issue | Task | Owner |
|-------|------|-------|
| C1_P1 | Validate spline knots in constructor | TBD |
| C6_P2 | Transactional construction pattern | TBD |
| C7_P2 | Fix TOCTOU in workspace creation | TBD |
| C8_P2 | Add iteration limit to simulation | TBD |

## Week 3-4: Data Integrity (P1)
| Issue | Task | Owner |
|-------|------|-------|
| H1_P1 | Add spline monotonicity validation | TBD |
| H8_P2 | Add condition number check | TBD |
| H11_P2 | Validate dt >= 0 | TBD |
| H12_P2 | Round-trip test for phase-type | TBD |

## Week 5-6: Memory & Concurrency
| Issue | Task | Owner |
|-------|------|-------|
| C11_P2 | Add workspace cleanup function | TBD |
| C9_P2 | Document thread-safety constraints | TBD |
| M18_P2 | Make global config thread-safe | TBD |

## Week 7-8: Test Infrastructure & Documentation
| Task | Owner |
|------|-------|
| Add invariant assertion framework | TBD |
| Add concurrency stress tests | TBD |
| Update user documentation | TBD |
| Create migration guide for old serialized models | TBD |

---

## Appendix: Files Reviewed

```
src/
├── types/
│   ├── abstract.jl ✓
│   ├── data_containers.jl ✓
│   ├── hazard_metadata.jl ✓
│   ├── hazard_specs.jl ✓
│   ├── hazard_structs.jl ✓
│   ├── infrastructure.jl ✓
│   └── model_structs.jl ✓
├── construction/
│   ├── hazard_builders.jl ✓
│   ├── hazard_constructors.jl ✓
│   ├── model_assembly.jl ✓
│   ├── multistatemodel.jl ✓
│   └── spline_builder.jl ✓
├── hazard/
│   ├── spline.jl ✓
│   ├── time_transform.jl ✓
│   └── (others - cursory)
├── likelihood/
│   ├── loglik_exact.jl ✓
│   ├── loglik_markov.jl ✓
│   └── (others - cursory)
├── inference/
│   ├── fit_common.jl ✓
│   ├── fit_exact.jl ✓
│   ├── fit_markov.jl ✓
│   ├── fit_mcem.jl ✓ (header)
│   └── smoothing_selection.jl ✓
├── output/
│   ├── variance.jl ✓ (header)
│   └── accessors.jl (cursory)
├── utilities/
│   ├── parameters.jl ✓
│   ├── bounds.jl ✓
│   └── reconstructor.jl ✓
└── simulation/
    └── simulate.jl ✓ (header)
```

---

## Revision History

| Date | Author | Changes |
|------|--------|---------|
| 2026-01-21 | Audit Agent | Initial audit report (Phase 1: Surface analysis, 30 issues) |
| 2026-01-21 | Audit Agent | Phase 2: Deep analysis (24 additional issues, 54 total) |
| 2026-01-21 | Audit Agent | Added invariants, edge cases, testing recommendations, remediation roadmap |

