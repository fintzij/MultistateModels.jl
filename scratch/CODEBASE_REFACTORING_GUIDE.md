# MultistateModels.jl Codebase Refactoring Guide

**Created**: 2026-01-08  
**Last Updated**: 2026-01-08  
**Branch**: penalized_splines  
**Status**: 🔴 Active - Items pending cleanup

---

## � ADVERSARIAL REVIEW FINDINGS (2026-01-08)

**Reviewer**: julia-statistician agent  
**Scope**: Systematic verification of all line numbers, function references, cross-references, and claims

### 🔴 CRITICAL Issues (Must Fix Before Implementation)

| # | Finding | Location | Impact | Action Required |
|---|---------|----------|--------|-----------------|
| C1 | **Missing documentation update for Item #21** | `MultistateModelsTests/reports/architecture.qmd` L415 | Example code shows `.parameters.natural` - will be WRONG after Item #21 | Add to Item #21 test maintenance |
| C2 | **Missing refactoring items for deprecated APIs** | `src/output/accessors.jl` L259-270, `src/surrogate/markov.jl` L439 | `get_loglik(model, "string")` and `fit_phasetype_surrogate()` are deprecated but not in guide | Add Items #22, #23 |
| C3 | **Test uses deprecated API** | `MultistateModelsTests/unit/test_surrogates.jl` L186 | Uses `_fit_phasetype_surrogate` which is marked deprecated | Must update when deprecated fn is removed |
| C4 | **Item #15 (monotone penalty bug) CONFIRMED** | `src/types/infrastructure.jl` L555, `src/construction/spline_builder.jl` L298-340 | Penalty matrix S built for B-spline coefs but applied to ests (increments). Correct: `ests' * (L' * S * L) * ests` | Mathematical fix required |

### 🟡 WARNING Issues (Should Fix)

| # | Finding | Location | Guide Says | Actual | Action |
|---|---------|----------|------------|--------|--------|
| W1 | Line number error | Item #16 | `default_nknots` at line ~432 | Actually at **L425** | Correct line reference |
| W2 | Line number error | Item #11 | Legacy aliases at L306-307 | Actually at **L301-302** (L51-52 also has aliases) | Correct line references |
| W3 | Line range error | Item #8 | `get_ij_vcov` at 627-630 | Actually 627-629 (3 lines, not 4) | Minor correction |
| W4 | Incomplete TODO list | Expansion hazards | N/A | TODO at `src/phasetype/expansion_hazards.jl` L221 (future feature) | Add to future work if relevant |

### ✅ VERIFIED Correct

| Claim | Verification Method | Result |
|-------|---------------------|--------|
| Test file line numbers for fit() calls | `grep -n "fit("` on each test file | All 10 files verified correct |
| default_nknots test lines L391-398, L628-637 | `grep -n "default_nknots"` | ✅ Correct |
| is_separable() safe to delete | `grep -rn "is_separable" src/` | ✅ No conditional usage found |
| BatchedODEData isolated to loglik_batched.jl | `grep -rn "BatchedODEData" src/` | ✅ Only in loglik_batched.jl |
| statetable not used anywhere | `grep -rn "statetable"` | ✅ Only defined, never called |
| select_smoothing_parameters NOT in _fit_exact | `grep -n` in fit_exact.jl | ✅ Confirmed - broken as described |
| _fit_markov_panel does NOT accept penalty/lambda_init | L34 in fit_markov.jl | ✅ Confirmed - kwargs silently ignored |
| FlattenAll at test_reconstructor.jl L9, L16, L82-85 | `grep -n "FlattenAll"` | ✅ Correct |
| AD backends all currently exported | L134-136 in MultistateModels.jl | ✅ All three exported |
| Longtest fit() calls for Markov panel | L105, L142, L338, L398, L443 | ✅ Correct |

### Missing Items to Add

**Item #22: Remove deprecated `get_loglik(model, "string")` argument**
- Location: `src/output/accessors.jl` L259-270
- The `ll::String` parameter is deprecated, use `type::Symbol` instead
- Test impact: None found using string form

**Item #23: Remove deprecated `fit_phasetype_surrogate()` function**
- Location: `src/surrogate/markov.jl` L439-455
- Docstring says: "This function is deprecated. Use `fit_surrogate(model; type=:phasetype, ...)`"
- Test impact: `test_surrogates.jl` L186 uses `_fit_phasetype_surrogate` - must update

---

## �📋 Test Maintenance Audit Summary

**Audit completed**: 2026-01-08 by julia-statistician agent  
**Verification method**: Systematic grep/search of `MultistateModelsTests/` directory

### Corrections Made During Audit

| Original Claim | Verified Finding | Correction |
|----------------|-----------------|------------|
| Item #11: `phasetype_longtest_helpers.jl` L214 needs "comment" update | Correct - but it's a **docstring**, not code | Updated terminology |
| Wave 1: "0 test files need updates" | Actually 1 docstring needs update | Fixed count |
| Item #21: Tests `test_helpers.jl` and `test_phasetype.jl` use `.parameters.natural` | **FALSE** - they use `get_parameters_natural()` function calls | Split into Type A (field access) vs Type B (function calls) |
| Item #21: "~15 locations" | Actually **8 locations** for field access, +5 for function calls | Fixed counts |
| Item #19.3: "No unit tests call fit() with Markov panel data" | TRUE for unit tests; BUT `longtests/longtest_robust_markov_phasetype.jl` DOES test Markov panel fitting (L105, L142, L338, L398, L443) | Added longtest reference |

### Key Findings

1. **31+ locations** access `model.parameters.flat` across unit tests - any change to parameter structure affects these
2. **8 locations** in 2 files (`test_initialization.jl`, `test_surrogates.jl`) directly access `.parameters.natural` field - **MUST** update for Item #21
3. **34+ locations** access `model.hazards[i]` - changes to hazard types affect these
4. **No unit tests** for `_fit_markov_panel`, but longtests DO exist
5. **`_fit_markov_panel` does NOT accept `penalty`/`lambda_init`** - parameters silently ignored via `kwargs...`

---

## ⚡ IMPLEMENTATION ORDER (Optimized for Success)

This guide is organized into **4 waves**. Complete each wave before moving to the next. Within each wave, items can be done in any order unless marked with dependencies.

### Wave 1: Foundation & Quick Wins (Do First)
Low-risk changes that clean the codebase and reduce noise. Build confidence with the codebase.

| Order | Item | Description | Risk | Est. Time |
|-------|------|-------------|------|-----------|
| 1.1 | #3 | Delete commented `statetable()` | 🟢 LOW | 5 min |
| 1.2 | #13 | Delete "function deleted" comment notes | 🟢 LOW | 5 min |
| 1.3 | #1 | Delete BatchedODEData zombie infrastructure | 🟢 LOW | 15 min |
| 1.4 | #2 | Delete is_separable() trait (always true) | 🟢 LOW | 15 min |
| 1.5 | #4 | Delete deprecated draw_paths overload | 🟢 LOW | 10 min |
| 1.6 | #11 | Delete legacy type aliases | 🟢 LOW | 10 min |
| 1.7 | #22 | Delete deprecated `get_loglik(model, "string")` | 🟢 LOW | 10 min |
| 1.8 | #23 | Delete deprecated `fit_phasetype_surrogate()` | 🟢 LOW | 10 min |

**Wave 1 Success Criteria**: All tests pass, ~350+ lines removed, codebase cleaner.

### Wave 2: Technical Debt & Internal Simplification
Structural improvements that make later work easier.

| Order | Item | Description | Risk | Est. Time | Depends On |
|-------|------|-------------|------|-----------|------------|
| 2.1 | #21 | Remove `parameters.natural` redundancy | 🟡 MED | 2-3 hrs | Wave 1 |
| 2.2 | #8 | Delete get_ij_vcov/get_jk_vcov wrappers | 🟢 LOW | 10 min | Wave 1 |
| 2.3 | #9 | Delete FlattenAll unused type | 🟢 LOW | 15 min | Wave 1 |
| 2.4 | #6 | Unexport unsupported AD backends | 🟡 MED | 15 min | Wave 1 |
| 2.5 | #10 | Review transform strategy abstraction | 🟡 MED | 30 min | Wave 1 |

**Wave 2 Success Criteria**: All tests pass, parameter structure simplified, API cleaner.

### Wave 3: Mathematical Correctness Bugs (⚠️ Critical)
These must be understood/fixed BEFORE Item #19. Each affects penalty/spline infrastructure.

| Order | Item | Description | Risk | Est. Time | Blocking |
|-------|------|-------------|------|-----------|----------|
| 3.1 | #16 | Fix default_nknots() for P-splines | 🟡 MED | 30 min | Item #19 |
| 3.2 | #15 | Fix monotone spline penalty matrix | 🔴 HIGH | 2-3 hrs | Item #19 |
| 3.3 | #5 | Verify rectify_coefs! with natural scale | 🟡 MED | 1 hr | Item #19 |
| 3.4 | #17 | Fix knot placement for panel data | 🟡 MED | 2 hrs | Item #19 |
| 3.5 | #18 | Investigate PIJCV Hessian NaN/Inf | 🟡 MED | 2-4 hrs | Item #19 |

**Wave 3 Success Criteria**: Mathematical correctness validated, penalty infrastructure sound.

### Wave 4: Major Features (Architectural)
Large-scope work that depends on everything above being solid.

| Order | Item | Description | Risk | Est. Time | Depends On |
|-------|------|-------------|------|-----------|------------|
| 4.1 | #19.1 | Penalized exact data fitting | 🔴 HIGH | 4-6 hrs | Waves 1-3 |
| 4.2 | #19.3 | Penalized Markov fitting | 🔴 HIGH | 3-4 hrs | #19.1 |
| 4.3 | #19.2 | Penalized MCEM fitting | 🔴 HIGH | 6-8 hrs | #19.1 |
| 4.4 | #7 | Consolidate variance functions | 🟡 MED | 2-3 hrs | #19.1-19.3 |
| 4.5 | #20 | Per-transition surrogate spec | 🟢 LOW | 2-3 hrs | Independent |

**Wave 4 Success Criteria**: Penalized fitting fully integrated, automatic λ selection working.

---

## Test Maintenance Summary by Wave

**Audit Status**: ✅ Verified 2026-01-08 via grep/search commands  
**Methodology**: All claims verified by searching actual test files; line numbers confirmed

This section provides a quick overview of test files that need modification **before** implementing each wave.

### ⚠️ CRITICAL: Comprehensive Test Impact Analysis

The following tests call `fit()` directly and are affected by ANY change to fitting infrastructure:

| Test File | fit() Calls | Data Type | Fitting Method | Potentially Affected By |
|-----------|-------------|-----------|----------------|------------------------|
| `test_penalty_infrastructure.jl` | L250, L254, L264, L297, L378 | Exact (obstype=1) | `_fit_exact` | #19.1 (exact penalized) |
| `test_phasetype.jl` | L1335, L1388, L1407, L1436 | Exact (obstype=1) | `_fit_exact` | #19.1, return type changes |
| `test_efs.jl` | L54, L110, L168, L225, L328 | Exact (obstype=1) | `_fit_exact` | #19.1, variance functions |
| `test_initialization.jl` | L414, L487, L707 | Mixed | `_fit_exact` | #19.1, #21 (parameters.natural) |
| `test_splines.jl` | L752, L812 | Exact (obstype=1) | `_fit_exact` | #19.1, spline changes |
| `test_ad_backends.jl` | L128, L160, L192, L227, L239, L277 | Exact (obstype=1) | `_fit_exact` | #6 (AD backends), #19.1 |
| `test_mll_consistency.jl` | L70 | Panel (obstype=2) | `_fit_mcem` | #19.2 (MCEM penalized) |
| `test_variance.jl` | L29, L60, L94, L127, L163 | Exact (obstype=1) | `_fit_exact` | #7, #19.1 |
| `test_perf.jl` | L53, L109, L167, L231 | Exact (obstype=1) | `_fit_exact` | #19.1 |
| `test_model_output.jl` | L52, L85, L334, L335 | Exact (obstype=1) | `_fit_exact` | #19.1, return type changes |

**Key observation**: There are NO unit tests that call `fit()` with Markov panel data (`is_markov(model) == true` AND `obstype=2`). The `_fit_markov_panel` function is not directly unit tested. This means adding `penalty`/`lambda_init` parameters to `_fit_markov_panel` signature won't break existing unit tests, BUT:
- Integration tests or longtests may exist
- The function is indirectly called via `fit()` but only for Markov models with panel data

### Tests That Access `MultistateModelFitted` Fields Directly

Changes to `MultistateModelFitted` struct (e.g., adding penalty fields for #19) will affect:

| Test File | Lines | Field Accessed | Impact |
|-----------|-------|----------------|--------|
| `test_mll_consistency.jl` | 81-90 | `fitted.ProposedPaths`, `fitted.parameters.flat`, `fitted.hazards` | Safe if fields preserved |
| `test_phasetype.jl` | 821, 830, 1413 | `surrogate_fitted.parameters`, `fitted.hazards` | Safe if fields preserved |
| `test_splines.jl` | 814, 817 | `fitted.loglik.loglik`, `fitted.parameters.flat` | Safe if fields preserved |
| `test_model_output.jl` | 179, 195, 205 | `fitted.SubjectWeights` | Safe if fields preserved |

### ⚠️ Tests Accessing `model.parameters.flat` (CRITICAL for Item #21)

These tests directly access `model.parameters.flat`. Item #21 (remove `parameters.natural` redundancy) should NOT affect these, but any change to the parameters structure will:

| Test File | Lines | Access Pattern | Notes |
|-----------|-------|----------------|-------|
| `test_helpers.jl` | 32, 54, 99 | `model.parameters.flat` | Gradient/Hessian computation |
| `test_reversible_tvc_loglik.jl` | 111, 179, 268, 319, 361, 433, 467, 468 | `model.parameters.flat` | Log-likelihood computation |
| `test_initialization.jl` | 162, 187, 193, 230, 362 | `model.parameters.flat` | Parameter initialization |
| `test_splines.jl` | 772, 773, 796, 817, 831, 836, 979, 1021, 1033, 1149, 1257, 1273 | `model.parameters.flat` | Spline parameterization |
| `test_mll_consistency.jl` | 86 | `fitted.parameters.flat` | MCEM consistency check |
| `test_phasetype_panel_expansion.jl` | 43 | `model.parameters.flat` | Phase-type setup |
| `test_phasetype.jl` | 830 | `surrogate_fitted.parameters.flat[1]` | Surrogate rate extraction |

**Total**: 31+ locations access `.parameters.flat`

### ⚠️ Tests Accessing `model.parameters.natural` (MUST UPDATE for Item #21)

These tests directly access `.parameters.natural` and WILL BREAK when Item #21 removes this field:

| Test File | Lines | Access Pattern | Required Action |
|-----------|-------|----------------|-----------------|
| `test_initialization.jl` | 164, 232, 336, 364 | `model.parameters.natural` | Replace with accessor or use `.nested` |
| `test_surrogates.jl` | 120, 168, 173, 244 | `surrogate.parameters.natural` | Replace with accessor or use `.nested` |

**Total**: 8 locations MUST be updated for Item #21

### Tests Accessing `model.hazards[i]` (CRITICAL for Hazard Type Changes)

These tests directly index into the hazards array. Changes to hazard structure will affect:

| Test File | Access Count | Notes |
|-----------|--------------|-------|
| `test_compute_hazard.jl` | 18 locations | Heavy direct access, tests hazard evaluation |
| `test_penalty_infrastructure.jl` | 1 location (L187) | Gets hazard for penalty testing |
| `test_pijcv.jl` | 2 locations (L601, L662) | Gets `npar_total` |
| `test_splines.jl` | 12 locations | Tests spline hazard properties |
| `test_phasetype.jl` | 1 location (L1413) | Tests `length(fitted.hazards)` |

**Total**: 34+ locations access `.hazards[i]`

### Wave 1: Foundation & Quick Wins

| Item | Test Files Affected | Changes Required |
|------|---------------------|------------------|
| #3 (statetable) | None | ✅ No changes |
| #13 (comments) | None | ✅ No changes |
| #1 (BatchedODEData) | None | ✅ No changes |
| #2 (is_separable) | None | ✅ No changes |
| #4 (draw_paths) | `test_simulation.jl` | ✅ Verify keyword form used (no changes needed) |
| #11 (type aliases) | `longtests/phasetype_longtest_helpers.jl` L214 | 🔄 Update 1 docstring (uses `MultistateMarkovModel`) |
| #22 (get_loglik string) | None | ✅ No tests use deprecated string form |
| #23 (fit_phasetype_surrogate) | `test_surrogates.jl` L186 | 🔄 Update to use `_build_phasetype_from_markov` |

**Wave 1 Total**: 1 docstring + 1 test function call update

### Wave 2: Technical Debt & Simplification

| Item | Test Files Affected | Changes Required |
|------|---------------------|------------------|
| #21 (parameters.natural) | `test_initialization.jl` (4), `test_surrogates.jl` (4), **`reports/architecture.qmd` (1)** | 🔄 Update 8 test + 1 doc location |
| #8 (get_*_vcov) | None | ✅ No changes |
| #9 (FlattenAll) | `test_reconstructor.jl` L9, L16, L82-85 | ❌ Delete tests |
| #6 (AD backends) | `test_ad_backends.jl` L3, L21, L61-77 | 🔄 Update imports/comments, tests access internal types |
| #10 (TransformStrategy) | None | ✅ No changes |

**Wave 2 Total**: 4 test files + 1 doc file need updates; ~19 locations

### Wave 3: Mathematical Correctness Bugs

| Item | Test Files Affected | Changes Required |
|------|---------------------|------------------|
| #16 (default_nknots) | `test_splines.jl` L391-398, L628-637 | 🔄 Update expected values OR ➕ Add new tests |
| #15 (monotone penalty) | `test_penalty_infrastructure.jl` | ➕ Add new testset for L' S L transformation |
| #5 (rectify_coefs) | `test_splines.jl` L459-537 | ✅ Verify existing tests pass |
| #17 (knot placement) | None existing | ➕ Add `test_panel_auto_knots.jl` |
| #18 (Hessian NaN) | `test_pijcv.jl`, `test_efs.jl` | ➕ Add `test_hessian_nan.jl` |

**Wave 3 Total**: 1-3 new test files; potential updates to 1 existing file

### Wave 4: Major Features

| Item | Test Files Affected | Changes Required |
|------|---------------------|------------------|
| #19.1 (exact penalized) | See "Critical Impact Analysis" above | 🔄 All `fit()` tests with exact data need verification |
| #19.2 (MCEM penalized) | `test_mll_consistency.jl` L70 | 🔄 Verify MCEM fit() still works; ➕ Add tests |
| #19.3 (Markov penalized) | None in unit tests | ➕ Add `test_penalized_markov.jl` |
| #7 (variance functions) | `test_penalty_infrastructure.jl` L309-322, `test_pijcv.jl` L30, `test_efs.jl` L65-121 | 🔄 Update if `compute_subject_hessians_fast` renamed |
| #20 (per-trans surrogate) | `test_surrogates.jl` | ➕ Add new testset for Dict API |
| #12 (calibrate_splines) | `test_splines.jl` L687-822 | ✅ Tests exist - verify |
| #14 (make_constraints) | None existing | ➕ Add tests OR remove from exports |

**Wave 4 Total**: 3-4 new test files; many existing tests need verification

---

### Items Requiring User Input (Decision Points)

These items have design decisions that need user confirmation before proceeding:

| Item | Decision Needed | Options |
|------|-----------------|---------|
| #19.1.3.3 | Storage for penalty info | A: Add fields to `MultistateModelFitted` struct; B: Add to `ConvergenceRecords` |
| #19.3.3.1 | Markov λ selection approach | A: Add `likelihood_type` param to `select_smoothing_parameters`; B: Create separate `select_smoothing_parameters_markov` |

**⚠️ Stop and ask before implementing these decisions.**

---

## Current Test Status (as of 2026-01-07)

| Metric | Count | Notes |
|--------|-------|-------|
| **Passing** | 1467 | Core functionality working |
| **Failing** | 1 | Phase-type VCV returns `nothing` |
| **Erroring** | 1 | MCEM flaky test (intermittent) |

**Known Issues Under Investigation**:
- Spline S(t) > 1 for some parameter configurations
- Phase-type positive log-likelihood (LL should be ≤ 0)
- PIJCV Hessian NaN/Inf warnings (gracefully handled but root cause unknown)

---

## Instructions for Agents

### Prerequisites

1. **Read the code reading guide first**: [docs/CODE_READING_GUIDE.md](../docs/CODE_READING_GUIDE.md)
2. You are acting as a **senior Julia developer and expert PhD mathematical statistician**
3. **Correctness is paramount** - statistical validity and mathematical fidelity must be preserved
4. **Backward compatibility is NOT required** - break APIs if it improves the codebase
5. After each set of changes, **run tests** to validate: `julia --project -e 'using Pkg; Pkg.test()'`
6. Document all changes in this file under the relevant item, mark items as ✅ COMPLETED with date when done
7. ALWAYS SOLICIT INPUT from the codebase owner if confused and after completing an item
8. ACTIVELY MONITOR CONTEXT AND PROACTIVELY TAKE ACTION TO MITIGATE CONTEXT CONFUSION - prepare handoff prompts and prepare to handoff to another agent whenever your performance begins to degrade


### Workflow for Each Item

1. Read the item description and evidence carefully
2. Search for all usages of the function/type being modified
3. **Review "Test Maintenance" section** for the item and make required test updates FIRST
4. Make the implementation change
5. Run the test suite
6. Update this document: mark item as ✅ COMPLETED with date
7. If tests fail, investigate and fix or revert

### Test Maintenance Philosophy

Each item includes a **"Test Maintenance" section** that lists tests requiring updates **BEFORE running tests**. This prevents confusion about whether test failures are from the refactoring itself or from tests that reference deleted/modified code.

**Order of operations:**
1. Update tests that reference the code being modified
2. Make the implementation change  
3. Run tests (should pass if both steps done correctly)

### Key Design Decisions (Current as of 2026-01-08)

- **Parameters are on NATURAL scale** (not log scale) - this was a recent refactoring (v0.3.0+)
- Box constraints (Ipopt `lb ≥ 0`) ensure positivity where needed (rates, shape parameters)
- There is NO separate "estimation scale" anymore - no exp/log transforms during fitting
- `parameters.flat` = flat vector for optimizer (natural scale)
- `parameters.nested` = nested NamedTuple by hazard `(h12 = (baseline = (rate=0.5,), covariates = (x=0.3,)), ...)`
- `parameters.natural` = **⚠️ REDUNDANT (see Item 21)** - same values as nested, different structure `(h12 = [0.5, 0.3], ...)`
- Spline coefficients use I-spline transformation for monotonicity constraints
- The `reconstructor` field handles flatten/unflatten operations (AD-compatible)

### Parameter Convention Reference

```julia
# All of these are on NATURAL scale (v0.3.0+):
model.parameters.flat      # Vector{Float64} for optimizer
model.parameters.nested    # NamedTuple{hazname => (baseline=NamedTuple, covariates=NamedTuple)}
model.parameters.natural   # ⚠️ REDUNDANT: NamedTuple{hazname => Vector{Float64}} - Item 21 will remove

# Unflatten preserves natural scale:
nested = unflatten(model.parameters.reconstructor, flat_vector)  # → NamedTuple
```

### Phase-Type Parameter Indexing (CRITICAL)

For phase-type models with shared hazards, parameter indices differ from hazard indices:

```julia
model.hazkeys                  # Dict{Symbol, Int} - hazard names → parameter indices
model.hazard_to_params_idx     # Vector[hazard_idx] → params_idx

# IMPORTANT: Parameter indices ≠ hazard indices when hazards are shared!
# When iterating over parameters, use params_idx_to_hazard_idx reverse mapping
```

This was the root cause of a critical bug fixed on 2026-01-07 (see handoff Part 2, FIX 3).

---

## 🔄 Agent Handoff Strategy

**Purpose**: Plan context-aware handoff points to prevent quality degradation during this multi-wave refactoring project.

### Handoff Triggers

| Trigger | Action | Rationale |
|---------|--------|-----------|
| Wave completed | **MANDATORY handoff** | Fresh context for new wave |
| 10+ exchanges on single item | Checkpoint or handoff | Accumulating details degrade performance |
| 3+ files modified in single session | Prepare handoff notes | Hard to track concurrent changes |
| Error spiral (3+ failed attempts) | Immediate handoff | Fresh perspective needed |
| Decision point reached | Pause, document, handoff if user unavailable | Avoid guessing at design decisions |

### Recommended Handoff Schedule

```
┌─────────────────────────────────────────────────────────────────────┐
│  WAVE 1: Foundation & Quick Wins (~1 hour total)                   │
│  ─────────────────────────────────────────────────────────────────  │
│  Items: #3, #13, #1, #2, #4, #11                                   │
│  ALL 6 ITEMS → Single Session (low complexity, no dependencies)    │
│                                                                     │
│  ✅ HANDOFF 1: After Wave 1 completion                             │
│     Document: Wave 1 complete, ~350 lines removed, tests passing   │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  WAVE 2: Technical Debt (~3-4 hours total)                         │
│  ─────────────────────────────────────────────────────────────────  │
│  Session 2A: Items #8, #9, #6, #10 (~1 hour)                       │
│              Quick deletions and export changes                     │
│                                                                     │
│  ✅ HANDOFF 2A (optional): If agent shows confusion                │
│                                                                     │
│  Session 2B: Item #21 ONLY (~2-3 hours)                            │
│              parameters.natural removal - largest Wave 2 item       │
│              8 test updates + 20 call site updates                  │
│                                                                     │
│  ✅ HANDOFF 2: After Wave 2 completion                             │
│     Document: Parameter structure simplified, 8 tests updated       │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  WAVE 3: Math Correctness Bugs (~8-12 hours total)                 │
│  ─────────────────────────────────────────────────────────────────  │
│  Session 3A: Items #16, #5 (~1.5 hours)                            │
│              default_nknots fix, rectify_coefs verification         │
│                                                                     │
│  ✅ HANDOFF 3A: After #16, #5 complete                             │
│                                                                     │
│  Session 3B: Item #15 ONLY (~2-3 hours) ⚠️ HIGH RISK              │
│              Monotone spline penalty matrix fix                     │
│              Complex math, requires fresh context                   │
│                                                                     │
│  ✅ HANDOFF 3B: After #15 complete                                 │
│     Document: Matrix derivation, test results, validation method    │
│                                                                     │
│  Session 3C: Items #17, #18 (~3-4 hours)                           │
│              Knot placement fix, Hessian NaN investigation          │
│                                                                     │
│  ✅ HANDOFF 3: After Wave 3 completion                             │
│     Document: All math issues resolved, penalty infra validated     │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  WAVE 4: Major Features (~15-20 hours total)                       │
│  ─────────────────────────────────────────────────────────────────  │
│  Session 4A: Item #19.1 Phases 1-2 (~3 hours)                      │
│              Create optimization wrappers, integrate select_λ       │
│                                                                     │
│  ✅ HANDOFF 4A: After Phases 1-2                                   │
│     Document: New function signatures, dispatch logic               │
│                                                                     │
│  Session 4B: Item #19.1 Phases 3-4 (~2 hours)                      │
│              Storage decision (needs USER INPUT), docstrings        │
│                                                                     │
│  ✅ HANDOFF 4B: After #19.1 complete                               │
│     Document: Exact penalized fitting working, λ selection tested   │
│                                                                     │
│  Session 4C: Item #19.3 Markov penalized (~3-4 hours)              │
│              Simpler than MCEM, builds on #19.1 patterns            │
│                                                                     │
│  ✅ HANDOFF 4C: After #19.3 complete                               │
│                                                                     │
│  Session 4D: Item #19.2 MCEM penalized (~6-8 hours) ⚠️ COMPLEX    │
│              May need 2 sessions with handoff midway                │
│                                                                     │
│  ✅ HANDOFF 4D: After #19.2 complete                               │
│     Document: All three fitting methods support penalty             │
│                                                                     │
│  Session 4E: Items #7, #20 (~4-6 hours)                            │
│              Variance consolidation, per-transition surrogate       │
│                                                                     │
│  ✅ HANDOFF 4: After Wave 4 completion                             │
│     Document: Full feature set complete, integration tests passing  │
└─────────────────────────────────────────────────────────────────────┘
```

### Session Duration Guidelines

| Session Type | Max Duration | Max Exchanges | Files Modified |
|-------------|--------------|---------------|----------------|
| Simple deletions (Wave 1) | 1-2 hours | 15-20 | 5-8 |
| Moderate refactoring | 2-3 hours | 20-25 | 3-5 |
| Complex math/algorithms | 2 hours | 15 | 2-3 |
| High-risk items (#15, #19.2) | 2 hours MAX | 12-15 | 1-2 |

### Handoff Document Template

When preparing handoff, create/update this file: `scratch/HANDOFF_NOTES.md`

```markdown
# Handoff: [Wave/Item]
**Date**: YYYY-MM-DD
**From Session**: [Brief description]
**To Session**: [Next focus]

## Completed This Session
- [ ] Item #X: [description]
- [ ] Tests updated: [list]
- [ ] Tests passing: YES/NO (if NO, explain)

## State of Codebase
- Branch: penalized_splines
- Last commit: [hash or message]
- Test status: [passing count] / [total]

## In-Progress Work
[If any incomplete items, describe state]

## Critical Context for Next Agent
1. [Key insight or decision from this session]
2. [Any gotchas discovered]
3. [Files that are partially modified]

## Next Steps
1. [First task for next session]
2. [Second task]

## Decision Points Pending
- [ ] #19.1.3.3: Storage location (struct vs ConvergenceRecords)
- [ ] #19.3.3.1: Markov λ selection approach
```

### Emergency Handoff (Context Degradation Detected)

If you notice:
- Repeating earlier explanations
- Forgetting recent changes
- Inconsistent recommendations
- Difficulty tracking file states

**IMMEDIATELY:**
1. Stop current work
2. Create handoff notes with current state
3. Document exactly where you are in the task
4. Alert user: "⚠️ Context degradation detected. Recommend handoff."

### Agent Mode Recommendations

| Task Type | Recommended Mode |
|-----------|-----------------|
| Code deletions (Wave 1) | `julia-statistician` |
| Parameter refactoring (#21) | `julia-statistician` |
| Math corrections (Wave 3) | `julia-statistician` |
| Complex features (Wave 4) | `julia-statistician` |
| Post-implementation review | Hand off to `julia-reviewer` |
| Architecture concerns | Hand off to `antipattern-scanner` |

---


## Progress Tracking (by Wave)

### Wave 1: Foundation & Quick Wins
| # | Item | Status | Date | Notes |
|---|------|--------|------|-------|
| 3 | Commented statetable() | ⬜ TODO | - | |
| 13 | Deleted function notes | ⬜ TODO | - | |
| 1 | BatchedODEData zombie code | ⬜ TODO | - | |
| 2 | is_separable() trait | ⬜ TODO | - | |
| 4 | Deprecated draw_paths overload | ⬜ TODO | - | |
| 11 | Legacy type aliases | ⬜ TODO | - | |
| 22 | Deprecated get_loglik string arg | ⬜ TODO | - | Found in adversarial review |
| 23 | Deprecated fit_phasetype_surrogate | ⬜ TODO | - | Found in adversarial review |

### Wave 2: Technical Debt & Simplification
| # | Item | Status | Date | Notes |
|---|------|--------|------|-------|
| 21 | Remove `parameters.natural` redundancy | ⬜ TODO | - | ⚠️ Largest Wave 2 item |
| 8 | get_ij_vcov/get_jk_vcov wrappers | ⬜ TODO | - | |
| 9 | FlattenAll unused type | ⬜ TODO | - | |
| 6 | AD Backend exports | ⬜ TODO | - | |
| 10 | Transform strategy abstraction | ⬜ TODO | - | |

### Wave 3: Mathematical Correctness Bugs
| # | Item | Status | Date | Notes |
|---|------|--------|------|-------|
| 16 | default_nknots() formula | ⬜ TODO | - | |
| 15 | Monotone spline penalty matrix | ⬜ TODO | - | ⚠️ Math correctness |
| 5 | rectify_coefs! update | ⬜ TODO | - | |
| 17 | Knot placement uses raw data | ⬜ TODO | - | |
| 18 | PIJCV Hessian NaN/Inf root cause | ⬜ TODO | - | |

### Wave 4: Major Features
| # | Item | Status | Date | Notes |
|---|------|--------|------|-------|
| 19.1 | Penalized exact fitting | ⬜ TODO | - | Depends on Waves 1-3 |
| 19.3 | Penalized Markov fitting | ⬜ TODO | - | Depends on 19.1 |
| 19.2 | Penalized MCEM fitting | ⬜ TODO | - | Depends on 19.1 |
| 7 | Variance function consolidation | ⬜ TODO | - | |
| 20 | Per-transition surrogate spec | ⬜ TODO | - | Independent |

### Remaining Items (No Fixed Order)
| # | Item | Status | Date | Notes |
|---|------|--------|------|-------|
| 12 | calibrate_splines verification | ⬜ TODO | - | |
| 14 | make_constraints export | ⬜ TODO | - | |
| 22 | Remove deprecated `get_loglik(model, "string")` | ⬜ TODO | - | Found in adversarial review |
| 23 | Remove deprecated `fit_phasetype_surrogate()` | ⬜ TODO | - | Found in adversarial review |

---

## 🔴 ARCHITECTURAL REFACTORING: Penalized Likelihood Fitting (Item #19)

This section covers integration of automatic smoothing parameter selection into **all three fitting methods**:
- **19.1** Exact data fitting (`_fit_exact`)
- **19.2** MCEM fitting (`_fit_mcem`) 
- **19.3** Markov panel fitting (`_fit_markov_panel`)

---

### 19.1 Exact Data Fitting (`_fit_exact`)

#### Test Maintenance Summary (Do BEFORE implementation phases)

| Test File | Lines | Action | Details |
|-----------|-------|--------|---------|
| `unit/test_penalty_infrastructure.jl` | 254-275 | 🔄 Update | Tests `fit(model; penalty=SplinePenalty())` - verify API changes don't break |
| `unit/test_pijcv.jl` | 571-670 | ✅ Keep | Tests `select_smoothing_parameters` directly - remain valid |
| `unit/test_efs.jl` | 266-301 | ✅ Keep | Tests `select_smoothing_parameters` with `:efs` - remain valid |
| `unit/test_perf.jl` | 274-309 | ✅ Keep | Tests `select_smoothing_parameters` with `:perf` - remain valid |
| `unit/test_pijcv_vs_loocv.jl` | 163-215 | ✅ Keep | Comparison tests - remain valid |
| **NEW** | - | ➕ Add | `MultistateModelsTests/unit/test_penalized_exact.jl` - integration tests for new `fit()` API |

**Key insight**: Most existing tests call `select_smoothing_parameters()` directly. Those tests remain valid. New tests are needed for the **integrated** `fit(...; penalty=..., select_lambda=...)` API.

#### Problem Statement

**Current State (BROKEN):**
- `_fit_exact` accepts `penalty` and `lambda_init` parameters
- When penalty is provided, it optimizes β at **fixed λ = `lambda_init`** (default 1.0)
- `select_smoothing_parameters` exists and implements proper alternating β-λ optimization
- **BUT `select_smoothing_parameters` is NEVER called from `fit`**

**What Currently Happens:**
```julia
fit(model; penalty=SplinePenalty())
  ↓
_fit_exact(model; penalty=SplinePenalty(), lambda_init=1.0)
  ↓
build_penalty_config(model, penalty; lambda_init=1.0)  # λ = 1.0 FIXED
  ↓
loglik_exact_penalized(params, data, penalty_config)   # Optimize β at λ = 1.0
  ↓
# Returns β̂(λ=1) — NOT the optimal (β̂, λ̂)
```

**Impact:** Users who specify `penalty=SplinePenalty()` get:
- Fixed λ = 1.0 (arbitrary default)
- No cross-validation for λ selection  
- Potentially severe over- or under-smoothing
- **Silently wrong results** - users think they're getting proper penalized splines

---

### Proposed Architecture

#### New Function Hierarchy

```
fit(model; penalty=..., select_lambda=..., ...)
  │
  ├─ is_panel_data? → _fit_markov_panel / _fit_mcem
  │
  └─ exact data:
       │
       ├─ penalty == nothing → _optimize_unpenalized_exact(...)
       │                         Pure MLE, current behavior
       │
       └─ penalty != nothing → _optimize_penalized_exact(...)
                                 │
                                 ├─ select_lambda == :none → Fixed λ optimization
                                 │                           (current behavior, for advanced users)
                                 │
                                 └─ select_lambda != :none → Performance iteration
                                                              Alternates β | λ optimization
                                                              Returns (β̂, λ̂) jointly optimal
```

#### API Changes

**Current API (confusing):**
```julia
fit(model; penalty=SplinePenalty(), lambda_init=1.0)  # Silently uses fixed λ
select_smoothing_parameters(model, SplinePenalty())   # Separate call, returns NamedTuple
```

**Proposed API (integrated):**
```julia
# Default: automatic λ selection via PIJCV (AD-optimized, NOT grid search)
fit(model; penalty=SplinePenalty())  
# → Automatically calls performance iteration with gradient-based λ optimization
# → Returns MultistateModelFitted with optimal (β̂, λ̂)

# Explicit method selection (all use AD-based optimization)
fit(model; penalty=SplinePenalty(), select_lambda=:pijcv)    # Default, fast
fit(model; penalty=SplinePenalty(), select_lambda=:pijcv10)  # 10-fold Newton-approximated CV

# Fixed λ (advanced users, testing)
fit(model; penalty=SplinePenalty(), select_lambda=:none, lambda_init=1.0)
```

#### ⚠️ CRITICAL: AD-Based λ Optimization Required

**All λ selection MUST use gradient-based optimization via automatic differentiation.**

The PIJCV criterion V(λ) is differentiable with respect to log(λ). The existing `select_smoothing_parameters` already implements this correctly:

```julia
# From smoothing_selection.jl - CORRECT approach:
adtype = Optimization.AutoForwardDiff()
optf = OptimizationFunction(criterion_at_fixed_beta, adtype)
prob = OptimizationProblem(optf, current_log_lambda, nothing; lb=lb, ub=ub)
sol = solve(prob, LBFGS(); ...)  # Gradient-based optimization
```

**Why no grid search:**
1. PIJCV is smooth and differentiable — use that information
2. Grid search is O(G × n × p²) vs gradient methods O(log(1/ε) × n × p²)
3. Grid search can miss the optimum between grid points
4. Wood (2024) designed NCV specifically for gradient-based optimization

**Note:** The codebase has a `_select_lambda_grid_search` fallback for exact CV methods (`:loocv`, `:cv5`) that inherently require refitting at each λ. These methods are slow and should be discouraged; `:pijcv` is preferred.

---

### Implementation Plan for Exact Data Fitting (Item 19.1)

#### Phase 1: Create Internal Optimization Wrappers

##### Phase 1 Itemized Tasks

| # | Task | File | Lines | Details | Verification |
|---|------|------|-------|---------|--------------|
| 19.1.1.1 | Read current `_fit_exact` implementation | `src/inference/fit_exact.jl` | 1-288 | Understand existing control flow, identify unpenalized/penalized branches | Document current structure |
| 19.1.1.2 | Identify unpenalized optimization section | `src/inference/fit_exact.jl` | ~45-120 | Lines between data prep and result construction | Mark section boundaries |
| 19.1.1.3 | Identify penalized optimization section | `src/inference/fit_exact.jl` | ~45-120 | Same section with penalty_config != nothing | Mark section boundaries |
| 19.1.1.4 | Create function stub `_optimize_unpenalized_exact` | `src/inference/fit_exact.jl` | NEW after line 40 | Signature: `(model, data, samplepaths; solver, parallel, constraints, verbose) → (sol, vcov, caches)` | Stub compiles |
| 19.1.1.5 | Extract unpenalized logic into `_optimize_unpenalized_exact` | `src/inference/fit_exact.jl` | varies | Move: objective construction, Optimization.jl setup, solve call, vcov computation | Function returns correct type |
| 19.1.1.6 | Create function stub `_optimize_penalized_exact` | `src/inference/fit_exact.jl` | NEW after `_optimize_unpenalized_exact` | Signature: `(model, data, samplepaths, penalty_config; select_lambda, solver, verbose) → (sol, vcov, caches, penalty_result)` | Stub compiles |
| 19.1.1.7 | Extract penalized logic into `_optimize_penalized_exact` | `src/inference/fit_exact.jl` | varies | Move: penalty objective construction, optimization setup, solve call | Function returns correct type |
| 19.1.1.8 | Add dispatch logic to `_fit_exact` | `src/inference/fit_exact.jl` | ~45 | `if isnothing(penalty) ... else ...` with calls to new functions | Dispatch works |
| 19.1.1.9 | Create `_build_fitted_model` helper | `src/inference/fit_exact.jl` | NEW at end | Extract: `MultistateModelFitted` construction, variance matrix setting, convergence records | Helper works |
| 19.1.1.10 | Verify unpenalized path unchanged | - | - | Run existing unpenalized tests | All pass, same results |
| 19.1.1.11 | Verify penalized path unchanged (fixed λ) | - | - | Run existing penalized tests with `select_lambda=:none` | All pass, same results |

##### Phase 1 Function Signatures (Exact Specifications)

**`_optimize_unpenalized_exact`** (NEW):
```julia
function _optimize_unpenalized_exact(
    model::MultistateModel,
    data::ExactData,
    samplepaths::Vector{SamplePath};
    solver = Optimization.LBFGS(),
    parallel::Bool = false,
    constraints = nothing,
    verbose::Bool = true,
    maxiter::Int = 500,
    gtol::Float64 = 1e-6
)::Tuple{OptimizationSolution, Union{Matrix{Float64}, Nothing}, NamedTuple}
    # Returns: (solution, variance_matrix, caches)
end
```

**`_optimize_penalized_exact`** (NEW):
```julia
function _optimize_penalized_exact(
    model::MultistateModel,
    data::ExactData,
    samplepaths::Vector{SamplePath},
    penalty_config::PenaltyConfig;
    select_lambda::Symbol = :pijcv,  # :pijcv, :pijcv10, :cv5, :cv10, :none
    lambda_init::Float64 = 1.0,
    solver = Optimization.LBFGS(),
    verbose::Bool = true,
    maxiter::Int = 500,
    gtol::Float64 = 1e-6
)::Tuple{OptimizationSolution, Union{Matrix{Float64}, Nothing}, NamedTuple, NamedTuple}
    # Returns: (solution, variance_matrix, caches, penalty_result)
    # penalty_result = (lambda=Vector, edf=NamedTuple, criterion=Float64, method=Symbol)
end
```

**`_build_fitted_model`** (NEW):
```julia
function _build_fitted_model(
    model::MultistateModel,
    sol::OptimizationSolution,
    vcov::Union{Matrix{Float64}, Nothing},
    caches::NamedTuple,
    penalty_result::Union{Nothing, NamedTuple};
    constraints = nothing
)::MultistateModelFitted
end
```

---

#### Phase 2: Integrate `select_smoothing_parameters`

##### Phase 2 Itemized Tasks

| # | Task | File | Lines | Details | Verification |
|---|------|------|-------|---------|--------------|
| 19.1.2.1 | Read `select_smoothing_parameters` return type | `src/inference/smoothing_selection.jl` | ~1376-1500 | Document fields in returned NamedTuple | Document structure |
| 19.1.2.2 | Read `select_smoothing_parameters` calling conventions | `src/inference/smoothing_selection.jl` | ~1241-1376 | Identify required vs optional args | Document requirements |
| 19.1.2.3 | Create `_smoothing_result_to_solution` adapter | `src/inference/smoothing_selection.jl` | NEW at line ~1500 | Convert NamedTuple → OptimizationSolution-like | Adapter compiles |
| 19.1.2.4 | Implement adapter logic | `src/inference/smoothing_selection.jl` | ~1500-1550 | Extract beta, compute final objective, create convergence info | Adapter returns valid |
| 19.1.2.5 | Add call to `select_smoothing_parameters` in `_optimize_penalized_exact` | `src/inference/fit_exact.jl` | within new function | When `select_lambda != :none` | Call succeeds |
| 19.1.2.6 | Handle `select_lambda == :none` branch | `src/inference/fit_exact.jl` | within new function | Use fixed λ optimization (current behavior) | Branch works |
| 19.1.2.7 | Convert result using `_smoothing_result_to_solution` | `src/inference/fit_exact.jl` | within new function | Transform result for `_build_fitted_model` | Conversion correct |
| 19.1.2.8 | Test integration with PIJCV | - | - | Call `fit(model; penalty=SplinePenalty())` | λ selected, model fits |

##### `_smoothing_result_to_solution` Specification

```julia
function _smoothing_result_to_solution(
    result::NamedTuple,
    model::MultistateModel,
    data::ExactData,
    penalty_config::PenaltyConfig
)::Tuple{OptimizationSolution, NamedTuple}
    # Input result fields:
    #   result.lambda::Vector{Float64}     - selected λ values
    #   result.beta::Vector{Float64}       - optimal parameters at λ
    #   result.edf::NamedTuple             - effective degrees of freedom
    #   result.criterion::Float64          - final PIJCV/CV value
    #   result.converged::Bool             - convergence status
    #   result.iterations::Int             - number of iterations
    
    # Create solution-like object:
    sol = (
        u = result.beta,
        objective = compute_penalized_loglik(result.beta, data, penalty_config, result.lambda),
        retcode = result.converged ? :Success : :MaxIters,
        stats = (iterations = result.iterations,)
    )
    
    penalty_result = (
        lambda = result.lambda,
        edf = result.edf,
        criterion = result.criterion,
        method = :pijcv  # or whatever was used
    )
    
    return sol, penalty_result
end
```

---

#### Phase 3: Update `MultistateModelFitted` Storage

##### Phase 3 Itemized Tasks

| # | Task | File | Lines | Details | Verification |
|---|------|------|-------|---------|--------------|
| 19.1.3.1 | Read `MultistateModelFitted` struct | `src/types/model_structs.jl` | ~330-380 | Identify existing fields | Document fields |
| 19.1.3.2 | Read `ConvergenceRecords` definition | `src/types/model_structs.jl` | ~290-330 | Identify if penalty info can go here | Document structure |
| 19.1.3.3 | **DECISION**: Store in struct vs ConvergenceRecords | - | - | Struct change is invasive; ConvergenceRecords is NamedTuple | Get user input |
| 19.1.3.4a | **Option A**: Add fields to `MultistateModelFitted` | `src/types/model_structs.jl` | ~350 | Add `penalty_config`, `lambda`, `edf` | Struct updated |
| 19.1.3.4b | **Option B**: Add to `ConvergenceRecords` | `src/inference/fit_exact.jl` | construction site | Include `penalty=(lambda=..., edf=...)` | Records updated |
| 19.1.3.5 | Create accessor `get_smoothing_parameters(model)` | `src/output/accessors.jl` | NEW at end | Return selected λ values | Accessor works |
| 19.1.3.6 | Create accessor `get_edf(model)` | `src/output/accessors.jl` | NEW at end | Return effective degrees of freedom | Accessor works |
| 19.1.3.7 | Export new accessors | `src/MultistateModels.jl` | export list | Add to exports | Exports visible |
| 19.1.3.8 | Add docstrings for new accessors | `src/output/accessors.jl` | NEW | Document parameters, returns, examples | Docstrings complete |

##### New Accessor Signatures

```julia
"""
    get_smoothing_parameters(model::MultistateModelFitted) → Union{Nothing, Vector{Float64}}

Return the selected smoothing parameters (λ) from penalized likelihood fitting.
Returns `nothing` if the model was fit without a penalty.

# Example
```julia
fitted = fit(model; penalty=SplinePenalty())
λ = get_smoothing_parameters(fitted)  # e.g., [0.23, 0.15]
```
"""
function get_smoothing_parameters(model::MultistateModelFitted)::Union{Nothing, Vector{Float64}}
    # Implementation depends on storage decision
end

"""
    get_edf(model::MultistateModelFitted) → Union{Nothing, NamedTuple}

Return the effective degrees of freedom for each hazard from penalized fitting.
Returns `nothing` if the model was fit without a penalty.

# Example
```julia
fitted = fit(model; penalty=SplinePenalty())
edf = get_edf(fitted)  # (h12 = 3.5, h21 = 4.2)
```
"""
function get_edf(model::MultistateModelFitted)::Union{Nothing, NamedTuple}
    # Implementation depends on storage decision
end
```

---

#### Phase 4: Update Docstrings and API

##### Phase 4 Itemized Tasks

| # | Task | File | Lines | Details | Verification |
|---|------|------|-------|---------|--------------|
| 19.1.4.1 | Find `fit` function docstring | `src/inference/fit_common.jl` | ~1-60 | Locate existing docstring | Found |
| 19.1.4.2 | Document `penalty` argument | `src/inference/fit_common.jl` | in docstring | Add description, type, default | Documented |
| 19.1.4.3 | Document `select_lambda` argument | `src/inference/fit_common.jl` | in docstring | Add description, allowed values, default | Documented |
| 19.1.4.4 | Add example for penalized fitting | `src/inference/fit_common.jl` | in docstring | Show `fit(model; penalty=SplinePenalty())` | Example works |
| 19.1.4.5 | Add example for fixed λ | `src/inference/fit_common.jl` | in docstring | Show `fit(model; penalty=..., select_lambda=:none, lambda_init=X)` | Example works |
| 19.1.4.6 | Add note about `:pijcv` being recommended | `src/inference/fit_common.jl` | in docstring | Explain why PIJCV preferred | Noted |
| 19.1.4.7 | Update `select_smoothing_parameters` docstring | `src/inference/smoothing_selection.jl` | ~1241-1280 | Add note that `fit(...; penalty=...)` is preferred | Noted |
| 19.1.4.8 | Verify docstrings render correctly | - | - | Build docs, check output | Docs look good |

##### Docstring Template for `fit`

```julia
"""
    fit(model::MultistateModel; penalty=nothing, select_lambda=:pijcv, ...) → MultistateModelFitted

Fit a multistate model to data.

# Penalized Spline Fitting

When fitting spline hazard models, you can use penalized likelihood with automatic 
smoothing parameter selection:

```julia
# Automatic λ selection (recommended)
fitted = fit(model; penalty=SplinePenalty())

# Specify cross-validation method
fitted = fit(model; penalty=SplinePenalty(), select_lambda=:pijcv)    # Default, fast
fitted = fit(model; penalty=SplinePenalty(), select_lambda=:cv10)     # 10-fold CV

# Fixed λ (for advanced users)
fitted = fit(model; penalty=SplinePenalty(), select_lambda=:none, lambda_init=0.5)
```

# Arguments
- `model::MultistateModel`: Model to fit
- `penalty=nothing`: Penalty specification. Use `SplinePenalty()` for spline hazards.
- `select_lambda::Symbol=:pijcv`: Method for selecting smoothing parameter λ.
  - `:pijcv` (default): Proximal iteration jackknife CV (fast, AD-optimized)
  - `:cv5`, `:cv10`: K-fold cross-validation
  - `:none`: Use fixed λ from `lambda_init`
- `lambda_init::Float64=1.0`: Initial (or fixed) smoothing parameter value
- `solver`: Optimization solver (default: LBFGS)
- `verbose::Bool=true`: Print progress messages
- ...

# Returns
`MultistateModelFitted` with estimated parameters. For penalized fits, use:
- `get_smoothing_parameters(fitted)` to retrieve selected λ
- `get_edf(fitted)` to retrieve effective degrees of freedom

# Notes
⚠️ Smoothing parameter selection uses AD-based optimization (not grid search).
This is mathematically correct and computationally efficient.
"""
```

---

#### Phase 5: Handle Constraints

##### Phase 5 Itemized Tasks

| # | Task | File | Lines | Details | Verification |
|---|------|------|-------|---------|--------------|
| 19.1.5.1 | Identify where constraints are checked | `src/inference/fit_exact.jl` | ~35-45 | Find `constraints !== nothing` checks | Found |
| 19.1.5.2 | Add mutual exclusion check | `src/inference/fit_exact.jl` | ~36 | Error if both `penalty` and `constraints` provided | Error thrown |
| 19.1.5.3 | Write clear error message | `src/inference/fit_exact.jl` | ~36 | `"Penalized likelihood with constraints is not yet supported..."` | Message clear |
| 19.1.5.4 | Add test for mutual exclusion | `MultistateModelsTests/unit/test_fit_errors.jl` | NEW | Test that `fit(model; penalty=..., constraints=...)` throws | Test passes |

##### Error Implementation

```julia
# In _fit_exact, early in function body:
if !isnothing(penalty) && !isnothing(constraints)
    throw(ArgumentError(
        "Penalized likelihood fitting with parameter constraints is not yet supported. " *
        "Please use either `penalty` (for penalized splines) OR `constraints` " *
        "(for constrained parameters), but not both simultaneously. " *
        "For constrained spline fitting, fit without penalty first, then use the " *
        "fitted values as starting points for a penalized fit."
    ))
end
```

---

#### Phase 6: Testing for Exact Data Fitting

##### Phase 6 Itemized Tasks

| # | Task | File | Lines | Details | Verification |
|---|------|------|-------|---------|--------------|
| 19.1.6.1 | Create test file | `MultistateModelsTests/unit/test_penalized_exact.jl` | NEW | Test fixture and tests | File created |
| 19.1.6.2 | Create simple spline model fixture | test file | ~10-40 | 2-state, 100 subjects, spline hazard | Model builds |
| 19.1.6.3 | Test: unpenalized fit unchanged | test file | ~50-70 | `fit(model)` produces same result as before | `@test` passes |
| 19.1.6.4 | Test: penalized fit with auto λ | test file | ~80-100 | `fit(model; penalty=SplinePenalty())` runs | `@test` passes |
| 19.1.6.5 | Test: selected λ is reasonable | test file | ~100-120 | `0.001 ≤ λ ≤ 100` (not at bounds) | `@test` passes |
| 19.1.6.6 | Test: EDF is computed | test file | ~120-140 | `get_edf(fitted)` returns NamedTuple with positive values | `@test` passes |
| 19.1.6.7 | Test: fixed λ option | test file | ~140-160 | `select_lambda=:none, lambda_init=0.5` uses 0.5 | `@test` passes |
| 19.1.6.8 | Test: different CV methods | test file | ~160-200 | `:pijcv`, `:cv5`, `:cv10` all work | `@test` passes |
| 19.1.6.9 | Test: penalty + constraints error | test file | ~200-220 | `@test_throws ArgumentError` | `@test_throws` passes |
| 19.1.6.10 | Run full test suite | - | - | `julia --project -e 'using Pkg; Pkg.test()'` | All pass |

---

### 19.2 MCEM Fitting (`_fit_mcem`)

#### Test Maintenance Summary (Do BEFORE implementation phases)

| Test File | Lines | Action | Details |
|-----------|-------|--------|---------|
| `unit/test_mcem.jl` | all | ✅ Keep | Existing MCEM tests don't use penalty - remain valid |
| `unit/test_mll_consistency.jl` | 280-306 | ✅ Keep | Uses `SMPanelData` and `loglik!` - remain valid |
| **NEW** | - | ➕ Add | `MultistateModelsTests/unit/test_penalized_mcem.jl` - new tests for penalized MCEM |

**Note**: MCEM penalty support is a NEW feature, not modifying existing behavior. Existing MCEM tests should continue to pass unchanged.

#### Current State

**Partial Implementation:**
- `_fit_mcem` accepts `penalty` and `lambda_init` parameters (lines 132-133)
- Builds `penalty_config` at line 299-303
- Uses fixed λ throughout MCEM iterations
- **No automatic λ selection implemented**

**What Currently Happens:**
```julia
fit(semimarkov_model; penalty=SplinePenalty())
  ↓
_fit_mcem(model; penalty=SplinePenalty(), lambda_init=1.0)
  ↓
penalty_config = build_penalty_config(model, penalty; lambda_init=1.0)
  ↓
# MCEM iterations with fixed λ = 1.0
# E-step: sample paths
# M-step: optimize β at fixed λ
  ↓
# Returns β̂(λ=1) — NOT the optimal (β̂, λ̂)
```

#### Challenge: λ Selection in MCEM Context

Smoothing parameter selection in MCEM is more complex than exact data:

1. **Complete-data likelihood is weighted**: Q(β|β') = Σᵢ wᵢ · ℓᵢ(β)
2. **Weights change each iteration**: As β changes, importance weights change
3. **PIJCV requires Hessians**: Need weighted Hessians from complete data
4. **Computational cost**: Each λ evaluation requires E-step + partial M-step

#### ⚠️ CRITICAL: AD-Based λ Optimization Required (No Grid Search)

**Grid search is NOT acceptable for λ selection.** The PIJCV criterion V(λ) is differentiable with respect to log(λ), so we MUST use gradient-based optimization:

```julia
# CORRECT: AD-based optimization
adtype = Optimization.AutoForwardDiff()
optf = OptimizationFunction(weighted_pijcv_criterion, adtype)
prob = OptimizationProblem(optf, log_lambda_init, state; lb=lb, ub=ub)
sol = solve(prob, LBFGS(); ...)  # Gradient-based

# WRONG: Grid search (DO NOT IMPLEMENT)
# for λ in grid  # ❌ Inefficient, ignores gradient information
```

#### Proposed Approach: Outer-Inner Iteration at each MCEM step (Option A - RECOMMENDED)

```
fit(semimarkov; penalty=SplinePenalty(), select_lambda=:pijcv)
  │
  └─ _fit_mcem_penalized(...)
       │
       ├─ Phase 1: Fit with fixed λ = λ_init until approximate convergence
       │            (Get reasonable β̂ for λ selection)
       │
       ├─ Phase 2: Select λ given current paths and weights
       │            - Use weighted PIJCV criterion (AD-optimized, NOT grid search)
       │            - V(λ) = Σᵢ wᵢ · Vᵢ(λ, β̂₋ᵢ(λ))
       │
       └─ Phase 3: Continue MCEM with selected λ until convergence
```

---

### Implementation Plan for MCEM (Item 19.2)

#### Phase 1: Create MCEM Penalized Wrapper

##### Phase 1 Itemized Tasks

| # | Task | File | Lines | Details | Verification |
|---|------|------|-------|---------|--------------|
| 19.2.1.1 | Read current `_fit_mcem` implementation | `src/inference/fit_mcem.jl` | 1-1314 | Identify penalty handling, M-step optimization | Document structure |
| 19.2.1.2 | Find `penalty` parameter handling | `src/inference/fit_mcem.jl` | 132-133 | Note where `penalty` and `lambda_init` are accepted | Found |
| 19.2.1.3 | Find `build_penalty_config` call | `src/inference/fit_mcem.jl` | 299-303 | Note where penalty_config is built | Found |
| 19.2.1.4 | Find M-step optimization | `src/inference/fit_mcem.jl` | varies | Identify `_mcem_mstep` or equivalent | Found |
| 19.2.1.5 | Create `_fit_mcem_penalized` function stub | `src/inference/fit_mcem.jl` | NEW after line ~100 | Signature: `(model, penalty_config; select_lambda, ...) → MultistateModelFitted` | Stub compiles |
| 19.2.1.6 | Create `_fit_mcem_fixed_lambda` function | `src/inference/fit_mcem.jl` | NEW | Extract current logic: fixed λ MCEM | Function works |
| 19.2.1.7 | Add dispatch in `_fit_mcem` | `src/inference/fit_mcem.jl` | ~300 | `if penalty !== nothing && select_lambda != :none` | Dispatch works |
| 19.2.1.8 | Verify fixed λ path unchanged | - | - | Run existing MCEM tests with `select_lambda=:none` | Tests pass |

##### `_fit_mcem_penalized` Specification

```julia
function _fit_mcem_penalized(
    model::MultistateModel,
    penalty_config::PenaltyConfig;
    select_lambda::Symbol = :pijcv,
    lambda_init::Float64 = 1.0,
    verbose::Bool = true,
    maxiter::Int = 200,
    phase1_maxiter::Int = 20,  # Initial MCEM before λ selection
    # ... other MCEM params ...
)::MultistateModelFitted
    
    # Phase 1: Initial MCEM with fixed λ
    # Phase 2: Select λ using AD-based optimization
    # Phase 3: Full MCEM with optimal λ
end
```

---

#### Phase 2: Implement Weighted λ Selection

##### Phase 2 Itemized Tasks

| # | Task | File | Lines | Details | Verification |
|---|------|------|-------|---------|--------------|
| 19.2.2.1 | Read existing PIJCV criterion | `src/inference/smoothing_selection.jl` | ~600-800 | Understand criterion computation | Document |
| 19.2.2.2 | Read subject Hessian computation | `src/output/variance.jl` | varies | Find existing `compute_subject_hessians` | Document |
| 19.2.2.3 | Create `_select_lambda_mcem` function stub | `src/inference/smoothing_selection.jl` | NEW at ~2100 | Signature below | Stub compiles |
| 19.2.2.4 | Create `compute_weighted_pijcv_criterion` | `src/inference/smoothing_selection.jl` | NEW | Apply importance weights to LOO contributions | Function compiles |
| 19.2.2.5 | Implement weighted Hessian aggregation | `src/inference/smoothing_selection.jl` | within 19.2.2.4 | `H_weighted = Σᵢ wᵢ * Hᵢ` | Correct aggregation |
| 19.2.2.6 | Set up AD-based λ optimization | `src/inference/smoothing_selection.jl` | within `_select_lambda_mcem` | `Optimization.AutoForwardDiff()`, LBFGS solver | AD works |
| 19.2.2.7 | Add bounds on log(λ) | `src/inference/smoothing_selection.jl` | within `_select_lambda_mcem` | `lb = [-10.0, ...]`, `ub = [10.0, ...]` | Bounds set |
| 19.2.2.8 | Test λ selection on simple MCEM model | - | - | Run `_select_lambda_mcem` manually | Returns reasonable λ |

##### `_select_lambda_mcem` Specification

```julia
"""
    _select_lambda_mcem(model, mcem_result, penalty_config; method=:pijcv) → Vector{Float64}

Select smoothing parameters λ for MCEM using weighted PIJCV criterion.

Uses AD-based optimization (not grid search) to minimize the weighted PIJCV criterion.

# Arguments
- `model`: The multistate model
- `mcem_result`: Result from Phase 1 MCEM, containing:
  - `paths`: Sampled paths from E-step
  - `weights`: Importance weights for each path
  - `beta`: Current parameter estimates
- `penalty_config`: Penalty configuration
- `method`: CV method (default `:pijcv`)

# Returns
- `Vector{Float64}`: Optimal λ values (one per penalized hazard)
"""
function _select_lambda_mcem(
    model::MultistateModel,
    mcem_result::NamedTuple,
    penalty_config::PenaltyConfig;
    method::Symbol = :pijcv,
    verbose::Bool = true
)::Vector{Float64}
    
    # Extract paths and weights from MCEM result
    paths = mcem_result.paths
    weights = mcem_result.weights
    current_beta = mcem_result.beta
    
    # Define weighted PIJCV criterion (closure over paths, weights)
    function weighted_criterion(log_lambda, _)
        λ = exp.(log_lambda)
        return compute_weighted_pijcv_criterion(model, paths, weights, penalty_config, λ)
    end
    
    # AD-based optimization (NOT grid search)
    adtype = Optimization.AutoForwardDiff()
    n_lambda = length(penalty_config.lambda)
    optf = OptimizationFunction(weighted_criterion, adtype)
    
    log_lambda_init = log.(penalty_config.lambda)
    lb = fill(-10.0, n_lambda)  # λ ∈ [exp(-10), exp(10)] ≈ [4.5e-5, 22026]
    ub = fill(10.0, n_lambda)
    
    prob = OptimizationProblem(optf, log_lambda_init, nothing; lb=lb, ub=ub)
    sol = solve(prob, Optim.LBFGS(); maxiters=100)
    
    return exp.(sol.u)
end
```

---

#### Phase 3: Test MCEM Penalty Integration

##### Phase 3 Itemized Tasks

| # | Task | File | Lines | Details | Verification |
|---|------|------|-------|---------|--------------|
| 19.2.3.1 | Create test file | `MultistateModelsTests/unit/test_penalized_mcem.jl` | NEW | Test fixture and tests | File created |
| 19.2.3.2 | Create 3-state semi-Markov model fixture | test file | ~10-50 | Spline hazards, 200 subjects, panel data | Model builds |
| 19.2.3.3 | Test: fixed λ MCEM unchanged | test file | ~60-80 | `select_lambda=:none` produces same result | `@test` passes |
| 19.2.3.4 | Test: auto λ MCEM runs | test file | ~90-120 | `fit(model; penalty=SplinePenalty())` completes | `@test` passes |
| 19.2.3.5 | Test: λ selection is reasonable | test file | ~130-150 | `0.001 ≤ λ ≤ 100` | `@test` passes |
| 19.2.3.6 | Test: weighted criterion is AD-compatible | test file | ~160-180 | `ForwardDiff.gradient(criterion, log_lambda)` works | `@test` passes |
| 19.2.3.7 | Run full test suite | - | - | `Pkg.test()` | All pass |

---

### 19.3 Markov Panel Fitting (`_fit_markov_panel`)

#### Test Maintenance Summary (Do BEFORE implementation phases)

| Test File | Lines | Action | Details |
|-----------|-------|--------|---------|
| **`longtests/longtest_robust_markov_phasetype.jl`** | L105, L142, L338, L398, L443 | 🔄 Verify unchanged | Calls `fit()` with Markov panel data (exponential hazards) |
| **`longtests/longtest_mcem_tvc.jl`** | L148 (comment) | ✅ Note | Documents that exponential hazards use `_fit_markov_panel` |
| `unit/test_subject_weights.jl` | 100-560 | ✅ Keep | Tests `MPanelData` structure, doesn't call `fit()` |
| `unit/test_observation_weights_emat.jl` | 93-373 | ✅ Keep | Tests `MPanelData` structure, doesn't call `fit()` |
| `unit/test_phasetype_panel_expansion.jl` | 45+ | ✅ Keep | Tests data expansion, doesn't call `fit()` |
| **NEW** | - | ➕ Add | `MultistateModelsTests/unit/test_penalized_markov.jl` |

**⚠️ CRITICAL FINDINGS**:

1. **Longtests DO test Markov panel fitting**: `longtest_robust_markov_phasetype.jl` has 5 `fit()` calls with Markov panel data. These tests MUST continue to pass unchanged.

2. **`_fit_markov_panel` does NOT accept penalty parameters**: Current signature (from `src/inference/fit_markov.jl` L34):
```julia
function _fit_markov_panel(model::MultistateModel; 
    constraints = nothing, verbose = true, solver = nothing, 
    adbackend::ADBackend = ForwardDiffBackend(), 
    compute_vcov = true, vcov_threshold = true, 
    compute_ij_vcov = true, compute_jk_vcov = false, 
    loo_method = :direct, kwargs...)  # <-- penalty passed here, IGNORED
```

3. **Silent failure mode today**: If user passes `penalty=SplinePenalty()` to `fit()` with Markov panel data, it passes through `kwargs...` and is silently ignored. No error, no warning, wrong results.

**Required implementation changes**:
1. Add `penalty = nothing` parameter to signature
2. Add `lambda_init::Float64 = 1.0` parameter
3. Add `select_lambda::Symbol = :pijcv` parameter
4. Build penalty config: `penalty_config = build_penalty_config(model, penalty; lambda_init=lambda_init)`
5. Create penalized Markov objective: `loglik_markov_penalized`
6. Integrate with smoothing parameter selection (new function or generalize existing)

**Risk assessment**:
- LOW risk to existing tests if implementation preserves default behavior (`penalty=nothing` → unpenalized)
- MEDIUM risk if internal restructuring changes return values or convergence behavior

#### Current State

**No Penalty Support:**
- `_fit_markov_panel` has **no penalty or lambda_init parameters**
- No `build_penalty_config` call
- Users **cannot** fit penalized spline models to Markov panel data

**This is a Feature Gap**, not a bug—penalty support was never implemented.

#### Challenge: Likelihood Structure

Markov panel likelihood uses matrix exponentials:
```
P(data | θ) = Π_i Π_j P(Y_{j+1} | Y_j, Q(θ))
            = Π_i Π_j exp(Q(θ) · Δt)
```

Where Q(θ) is the intensity matrix built from hazard parameters.

**Key insight:** The penalty structure is identical to exact data. Only the likelihood computation differs.

---

### Implementation Plan for Markov (Item 19.3)

#### Phase 1: Add Penalty Parameters

##### Phase 1 Itemized Tasks

| # | Task | File | Lines | Details | Verification |
|---|------|------|-------|---------|--------------|
| 19.3.1.1 | Read current `_fit_markov_panel` signature | `src/inference/fit_markov.jl` | 1-186 | Note existing parameters | Document signature |
| 19.3.1.2 | Add `penalty` parameter | `src/inference/fit_markov.jl` | signature | `penalty = nothing` | Compiles |
| 19.3.1.3 | Add `lambda_init` parameter | `src/inference/fit_markov.jl` | signature | `lambda_init = 1.0` | Compiles |
| 19.3.1.4 | Add `select_lambda` parameter | `src/inference/fit_markov.jl` | signature | `select_lambda = :pijcv` | Compiles |
| 19.3.1.5 | Add mutual exclusion check | `src/inference/fit_markov.jl` | ~25 | Error if `penalty` and `constraints` both provided | Error thrown |
| 19.3.1.6 | Verify unpenalized path unchanged | - | - | Run existing Markov tests | Tests pass |

##### Updated Signature

```julia
function _fit_markov_panel(
    model::MultistateModel; 
    constraints = nothing, 
    verbose::Bool = true, 
    solver = nothing,
    maxiter::Int = 500,
    gtol::Float64 = 1e-6,
    # NEW parameters:
    penalty = nothing,
    lambda_init::Float64 = 1.0,
    select_lambda::Symbol = :pijcv
)::MultistateModelFitted
```

---

#### Phase 2: Create Penalized Markov Optimization

##### Phase 2 Itemized Tasks

| # | Task | File | Lines | Details | Verification |
|---|------|------|-------|---------|--------------|
| 19.3.2.1 | Create `_optimize_unpenalized_markov` function | `src/inference/fit_markov.jl` | NEW ~40 | Extract current optimization logic | Function works |
| 19.3.2.2 | Create `_optimize_penalized_markov` function stub | `src/inference/fit_markov.jl` | NEW ~80 | Signature below | Stub compiles |
| 19.3.2.3 | Build `penalty_config` in `_optimize_penalized_markov` | `src/inference/fit_markov.jl` | within function | `build_penalty_config(model, penalty; lambda_init)` | Config built |
| 19.3.2.4 | Create penalized Markov objective | `src/inference/fit_markov.jl` | within function | `loglik_markov - compute_penalty` | Objective compiles |
| 19.3.2.5 | Add dispatch logic to `_fit_markov_panel` | `src/inference/fit_markov.jl` | ~30-50 | `if isnothing(penalty) ... else ...` | Dispatch works |
| 19.3.2.6 | Integrate `select_smoothing_parameters` | `src/inference/fit_markov.jl` | within penalized function | Call when `select_lambda != :none` | Integration works |
| 19.3.2.7 | Verify penalized path works (fixed λ) | - | - | Test with `select_lambda=:none` | Test passes |

##### `_optimize_penalized_markov` Specification

```julia
function _optimize_penalized_markov(
    model::MultistateModel,
    penalty_config::PenaltyConfig;
    select_lambda::Symbol = :pijcv,
    solver = nothing,
    verbose::Bool = true,
    maxiter::Int = 500,
    gtol::Float64 = 1e-6
)::Tuple{OptimizationSolution, Union{Matrix{Float64}, Nothing}, NamedTuple, NamedTuple}
    # Returns: (solution, variance_matrix, caches, penalty_result)
    
    if select_lambda == :none
        # Fixed λ optimization
        sol = optimize_at_fixed_lambda(model, penalty_config; ...)
        penalty_result = (lambda = penalty_config.lambda, edf = nothing, method = :none)
    else
        # AD-based λ selection (NOT grid search)
        result = select_smoothing_parameters_markov(model, penalty_config; method=select_lambda)
        sol, penalty_result = _smoothing_result_to_solution(result, model, penalty_config)
    end
    
    # Compute variance matrix
    vcov = compute_markov_vcov(sol.u, model)
    caches = (books = model.books,)
    
    return sol, vcov, caches, penalty_result
end
```

---

#### Phase 3: Generalize or Create Markov Selection

##### Phase 3 Itemized Tasks

| # | Task | File | Lines | Details | Verification |
|---|------|------|-------|---------|--------------|
| 19.3.3.1 | **DECISION**: Generalize existing vs create new | - | - | Option A: Add `likelihood_type` param to `select_smoothing_parameters`; Option B: Create `select_smoothing_parameters_markov` | Get user input |
| 19.3.3.2a | **Option A**: Add `likelihood_type` parameter | `src/inference/smoothing_selection.jl` | signature | `likelihood_type::Symbol = :exact` | Param added |
| 19.3.3.3a | **Option A**: Add dispatch on likelihood_type | `src/inference/smoothing_selection.jl` | within function | `if likelihood_type == :markov ... elseif ... end` | Dispatch works |
| 19.3.3.2b | **Option B**: Create `select_smoothing_parameters_markov` | `src/inference/smoothing_selection.jl` | NEW ~2200 | Copy structure from exact version | Function created |
| 19.3.3.3b | **Option B**: Replace `loglik_exact` with `loglik_markov` | `src/inference/smoothing_selection.jl` | within function | Update likelihood calls | Correct likelihood |
| 19.3.3.4 | Verify subject Hessians work for Markov | `src/output/variance.jl` | varies | Check `compute_subject_hessians` dispatches correctly | Works |
| 19.3.3.5 | Ensure AD compatibility with `loglik_markov` | - | - | `ForwardDiff.hessian(loglik_markov, params)` | No errors |
| 19.3.3.6 | Test PIJCV with Markov likelihood | - | - | Manual test with simple model | Criterion computes |

---

#### Phase 4: Test Markov Penalty Integration

##### Phase 4 Itemized Tasks

| # | Task | File | Lines | Details | Verification |
|---|------|------|-------|---------|--------------|
| 19.3.4.1 | Create test file | `MultistateModelsTests/unit/test_penalized_markov.jl` | NEW | Test fixture and tests | File created |
| 19.3.4.2 | Create 2-state Markov model fixture | test file | ~10-40 | Spline hazards, panel data, 150 subjects | Model builds |
| 19.3.4.3 | Test: unpenalized Markov unchanged | test file | ~50-70 | `fit(model)` same as before | `@test` passes |
| 19.3.4.4 | Test: penalized Markov with fixed λ | test file | ~80-100 | `fit(model; penalty=..., select_lambda=:none)` | `@test` passes |
| 19.3.4.5 | Test: penalized Markov with auto λ | test file | ~110-140 | `fit(model; penalty=SplinePenalty())` | `@test` passes |
| 19.3.4.6 | Test: λ is reasonable | test file | ~150-170 | `0.001 ≤ λ ≤ 100` | `@test` passes |
| 19.3.4.7 | Test: EDF stored in result | test file | ~180-200 | `get_edf(fitted)` returns values | `@test` passes |
| 19.3.4.8 | Test: penalty + constraints error | test file | ~210-230 | `@test_throws ArgumentError` | `@test_throws` passes |
| 19.3.4.9 | Run full test suite | - | - | `Pkg.test()` | All pass |

---

### 19.4 Unified Architecture Summary

#### Target State

```
fit(model; penalty=..., select_lambda=..., ...)
  │
  ├─ is_panel_data(model)?
  │     │
  │     ├─ is_markov(model)?
  │     │     │
  │     │     ├─ penalty == nothing → _optimize_unpenalized_markov(...)
  │     │     └─ penalty != nothing → _optimize_penalized_markov(...)
  │     │                              └─ select_smoothing_parameters(...; likelihood_type=:markov)
  │     │
  │     └─ semi-Markov (MCEM):
  │           │
  │           ├─ penalty == nothing → _fit_mcem_unpenalized(...)
  │           └─ penalty != nothing → _fit_mcem_penalized(...)
  │                                    └─ _select_lambda_mcem(...)  # weighted PIJCV
  │
  └─ exact data:
        │
        ├─ penalty == nothing → _optimize_unpenalized_exact(...)
        └─ penalty != nothing → _optimize_penalized_exact(...)
                                 └─ select_smoothing_parameters(...; likelihood_type=:exact)
```

#### Shared Components (No Changes Needed)

| Component | Used By | Notes |
|-----------|---------|-------|
| `PenaltyConfig` | All | Existing, no changes |
| `SplinePenalty` | All | Existing, no changes |
| `build_penalty_config` | All | Existing, no changes |
| `compute_penalty` | All | Existing, no changes |

#### Shared Components (Require Modification)

| Component | Modification | Files |
|-----------|-------------|-------|
| `select_smoothing_parameters` | Add `likelihood_type` parameter OR create Markov variant | `src/inference/smoothing_selection.jl` |
| `fit_penalized_beta` | May need Markov variant | `src/inference/smoothing_selection.jl` |
| `MultistateModelFitted` | Add λ/EDF storage | `src/types/model_structs.jl` |
| `fit` docstring | Document new penalty params | `src/inference/fit_common.jl` |

#### New Components Required

| Component | File | Purpose |
|-----------|------|---------|
| `_optimize_unpenalized_exact` | `src/inference/fit_exact.jl` | Pure MLE, extract from `_fit_exact` |
| `_optimize_penalized_exact` | `src/inference/fit_exact.jl` | Penalized with λ selection |
| `_build_fitted_model` | `src/inference/fit_exact.jl` | Common result construction |
| `_smoothing_result_to_solution` | `src/inference/smoothing_selection.jl` | Adapt result format |
| `_optimize_unpenalized_markov` | `src/inference/fit_markov.jl` | Pure MLE for Markov |
| `_optimize_penalized_markov` | `src/inference/fit_markov.jl` | Penalized with λ selection |
| `_fit_mcem_penalized` | `src/inference/fit_mcem.jl` | Penalized MCEM wrapper |
| `_fit_mcem_fixed_lambda` | `src/inference/fit_mcem.jl` | Extract current MCEM logic |
| `_select_lambda_mcem` | `src/inference/smoothing_selection.jl` | Weighted PIJCV for MCEM |
| `compute_weighted_pijcv_criterion` | `src/inference/smoothing_selection.jl` | MCEM weighted criterion |
| `get_smoothing_parameters` | `src/output/accessors.jl` | Accessor for λ |
| `get_edf` | `src/output/accessors.jl` | Accessor for EDF |

#### Implementation Priority

| Priority | Task | Complexity | Prerequisite | Est. LOC |
|----------|------|------------|--------------|----------|
| 1 | Item 19.1: Exact data integration | Medium | None | ~300 |
| 2 | Item 19.3: Markov penalty support | Medium | 19.1 (reuse patterns) | ~250 |
| 3 | Item 19.2: MCEM penalty integration | High | 19.1 (reuse patterns) | ~400 |

#### Success Criteria (All Methods)

1. ✅ `fit(model; penalty=SplinePenalty())` automatically selects λ via AD-based optimization
2. ✅ Grid search is NEVER used for λ selection
3. ✅ Selected λ stored in fitted model (via `get_smoothing_parameters`)
4. ✅ EDF computed and stored (via `get_edf`)
5. ✅ All existing tests pass unchanged
6. ✅ New tests verify λ selection correctness
7. ✅ Documentation updated with examples
8. ✅ Consistent API across exact, Markov, and MCEM fitting

---

---

# DETAILED ITEM DESCRIPTIONS

Items are organized by wave. **Complete all items in a wave before proceeding to the next wave.**

---

## 📦 WAVE 1: Foundation & Quick Wins

These are low-risk deletions that clean the codebase. Do these first to build familiarity.

### Item 3: Commented statetable() Function - DEAD CODE

**Location**: `src/utilities/misc.jl` lines 1-23

**Problem**: Entire function has been commented out. Pure dead code taking up space.

#### Test Maintenance (Do BEFORE implementation)

| Test File | Lines | Action | Details |
|-----------|-------|--------|---------|
| **None** | - | ✅ No changes needed | No tests reference `statetable` |

**Verification**: `grep -r "statetable" MultistateModelsTests/` returns no matches.

#### Itemized Tasks

| # | Task | File | Lines | Verification |
|---|------|------|-------|--------------|
| 3.1 | Verify function is entirely commented | `src/utilities/misc.jl` | 1-23 | Visual inspection |
| 3.2 | Search for any `statetable` references | all `src/` | - | `grep -r "statetable" src/` - should find nothing uncommented |
| 3.3 | Delete lines 1-23 | `src/utilities/misc.jl` | 1-23 | Lines gone |
| 3.4 | Run full test suite | - | - | All tests pass |

**Expected Result**: 23 lines deleted.

**Risk**: LOW - Already non-functional

---

### Item 13: Deleted Function Notes - DEAD COMMENTS

**Location**: `src/inference/smoothing_selection.jl` lines 218, 1676, 1880

**Problem**: Comments like `# NOTE: function was deleted on 2025-01-05...` add noise. Git history preserves this information.

#### Test Maintenance (Do BEFORE implementation)

| Test File | Lines | Action | Details |
|-----------|-------|--------|---------|
| **None** | - | ✅ No changes needed | Comments have no test references |

#### Itemized Tasks

| # | Task | File | Lines | Verification |
|---|------|------|-------|--------------|
| 13.1 | Find all deletion notes | `src/inference/smoothing_selection.jl` | - | `grep -n "deleted on" src/inference/smoothing_selection.jl` |
| 13.2 | Delete comment at line ~218 | `src/inference/smoothing_selection.jl` | 218 | Comment gone |
| 13.3 | Delete comment at line ~1676 | `src/inference/smoothing_selection.jl` | 1676 | Comment gone |
| 13.4 | Delete comment at line ~1880 | `src/inference/smoothing_selection.jl` | 1880 | Comment gone |
| 13.5 | Run full test suite | - | - | All tests pass |

**Expected Result**: 3 comments deleted, cleaner code.

**Risk**: LOW - Just comments

---

### Item 1: BatchedODEData and to_batched_ode_data() - ZOMBIE INFRASTRUCTURE

**Location**: `src/likelihood/loglik_batched.jl` lines 179-340

**Problem**: This entire infrastructure (~160 lines) was built for planned "ODE hazards" and "neural network hazards" that were never implemented. The `to_batched_ode_data()` function has zero call sites.

**Evidence**:
```bash
grep -r "to_batched_ode_data(" src/  # Only finds definition and docstring
grep -r "BatchedODEData" src/        # Only finds definition
```

#### Test Maintenance (Do BEFORE implementation)

| Test File | Lines | Action | Details |
|-----------|-------|--------|---------|
| **None** | - | ✅ No changes needed | No tests reference `BatchedODEData` or `to_batched_ode_data` |

**Verification**: `grep -r "BatchedODEData\|to_batched_ode_data" MultistateModelsTests/` returns no matches.

#### Itemized Tasks

| # | Task | File | Lines | Verification |
|---|------|------|-------|--------------|
| 1.1 | Delete `BatchedODEData` struct definition | `src/likelihood/loglik_batched.jl` | 179-222 | `grep -r "BatchedODEData" src/` returns nothing |
| 1.2 | Delete `to_batched_ode_data()` function | `src/likelihood/loglik_batched.jl` | 258-342 | `grep -r "to_batched_ode_data" src/` returns nothing |
| 1.3 | Delete docstring for `to_batched_ode_data` | `src/likelihood/loglik_batched.jl` | 224-256 | Docstring gone |
| 1.4 | Search CHANGELOG.md for references | `CHANGELOG.md` | - | Remove any mentions |
| 1.5 | Run full test suite | - | - | `julia --project -e 'using Pkg; Pkg.test()'` passes |

**Expected Result**: ~160 lines deleted, no functionality lost, all tests pass.

**Risk**: LOW - No production code uses this

---

### Item 2: is_separable() Trait System - ALWAYS RETURNS TRUE

**Location**: `src/likelihood/loglik_batched.jl` lines 19-75

**Problem**: All 4 dispatch methods return `true`. No code branch ever takes a `false` path. This is premature abstraction for ODE hazards that don't exist.

**Evidence**:
```julia
is_separable(::_Hazard) = true  # Default - line 23
is_separable(::MarkovHazard) = true  # line 27  
is_separable(::SemiMarkovHazard) = true  # line 31
is_separable(::_SplineHazard) = true  # line 35
```

#### Test Maintenance (Do BEFORE implementation)

| Test File | Lines | Action | Details |
|-----------|-------|--------|---------|
| **None** | - | ✅ No changes needed | No tests reference `is_separable` |

**Verification**: `grep -r "is_separable" MultistateModelsTests/` returns no matches.

#### Itemized Tasks

| # | Task | File | Lines | Verification |
|---|------|------|-------|--------------|
| 2.1 | Search for `is_separable` calls | all `src/` | - | `grep -rn "is_separable(" src/` - identify all call sites |
| 2.2 | For each call site, verify it's in dead code or always-true branch | varies | - | Manual inspection |
| 2.3 | Delete `is_separable` docstring | `src/likelihood/loglik_batched.jl` | 1-18 | Gone |
| 2.4 | Delete all 4 `is_separable` method definitions | `src/likelihood/loglik_batched.jl` | 19-35 | Gone |
| 2.5 | Delete any conditional branches that check `is_separable` | varies | - | No `if is_separable` remains |
| 2.6 | Run full test suite | - | - | All tests pass |

**Expected Result**: ~60 lines deleted (docstring + definitions), simplified code paths.

**Risk**: LOW - All code assumes separability

---

### Item 4: Deprecated draw_paths(model, npaths) Overload

**Location**: `src/inference/sampling.jl` lines 486-501

**Problem**: This deprecated positional argument form just wraps the keyword form and emits a deprecation warning. Since backward compatibility is not required, delete it.

#### Test Maintenance (Do BEFORE implementation)

| Test File | Lines | Action | Details |
|-----------|-------|--------|---------|
| `unit/test_simulation.jl` | 12, 107-123 | ✅ Verify uses keyword form | Tests already use `draw_paths(model; npaths=X)` syntax |

**Verification**: Tests already use the keyword form `draw_paths(model; npaths=3, ...)` — no updates needed.

```bash
# Verify: All test uses should be keyword form
grep -n "draw_paths" MultistateModelsTests/unit/test_simulation.jl
# Line 12: import statement
# Lines 107-123: All use `draw_paths(model; npaths=...)` keyword form
```

#### Itemized Tasks

| # | Task | File | Lines | Verification |
|---|------|------|-------|--------------|
| 4.1 | Identify the deprecated function | `src/inference/sampling.jl` | 486-501 | Function with `@warn "draw_paths(model, npaths) is deprecated"` |
| 4.2 | Search for positional-form callers | all `src/` and `MultistateModelsTests/` | - | `grep -rn "draw_paths(.*,.*[0-9]" src/ MultistateModelsTests/` |
| 4.3 | Update any callers to use keyword form | varies | - | Change `draw_paths(model, 100)` → `draw_paths(model; npaths=100)` |
| 4.4 | Delete the deprecated overload | `src/inference/sampling.jl` | 486-501 | Function gone |
| 4.5 | Run full test suite | - | - | All tests pass |

**Expected Result**: ~15 lines deleted, cleaner API.

**Risk**: LOW - Callers already get deprecation warning

---

### Item 11: Legacy Type Aliases

**Location**: `src/types/model_structs.jl` lines 51-52 and 301-302 (verified)

**Problem**: These aliases exist for backward compatibility:
```julia
const MultistateMarkovProcess = MultistateProcess
const MultistateSemiMarkovProcess = MultistateProcess
const MultistateMarkovModel = MultistateModel
const MultistateSemiMarkovModel = MultistateModel
```

#### Test Maintenance (Do BEFORE implementation)

| Test File | Lines | Action | Details |
|-----------|-------|--------|---------|
| `longtests/phasetype_longtest_helpers.jl` | 214 | 🔄 Update docstring | Docstring mentions `MultistateMarkovModel`; change to `MultistateModel` |

**Details** (verified via grep):
```bash
grep -rn "MultistateMarkovModel" MultistateModelsTests/
# → longtests/phasetype_longtest_helpers.jl:214
# Context: In a docstring: "This creates a standard MultistateMarkovModel that can be fitted..."
```

**Action**: Update the docstring at lines 213-215 to use `MultistateModel` instead of `MultistateMarkovModel`.

#### Itemized Tasks

| # | Task | File | Lines | Verification |
|---|------|------|-------|--------------|
| 11.1 | Find exact line numbers of aliases | `src/types/model_structs.jl` | 51-52, 301-302 | `grep -n "const Multistate" src/types/model_structs.jl` |
| 11.2 | Search for `MultistateMarkovProcess` usage | all `src/` | - | `grep -r "MultistateMarkovProcess" src/` |
| 11.3 | Search for `MultistateSemiMarkovProcess` usage | all `src/` | - | `grep -r "MultistateSemiMarkovProcess" src/` |
| 11.4 | Search for `MultistateMarkovModel` usage | all `src/` | - | `grep -r "MultistateMarkovModel" src/` |
| 11.5 | Search for `MultistateSemiMarkovModel` usage | all `src/` | - | `grep -r "MultistateSemiMarkovModel" src/` |
| 11.6 | Replace any internal usages with `MultistateProcess`/`MultistateModel` | varies | - | All call sites updated |
| 11.7 | Delete the 4 `const` alias lines | `src/types/model_structs.jl` | 51-52, 301-302 | Aliases gone |
| 11.8 | Update any docstring/comment references | varies | - | No mentions remain |
| 11.9 | Run full test suite | - | - | All tests pass |

**Expected Result**: 4 lines deleted, simpler API without legacy aliases.

**Risk**: LOW - Just aliases

---

---

## 📦 WAVE 2: Technical Debt & Simplification

Complete Wave 1 first. These items reduce complexity and make later work easier.

### Item 21: Remove `parameters.natural` Redundancy

**Location**: `src/construction/model_assembly.jl`, `src/utilities/parameters.jl`, `src/utilities/transforms.jl`

**Problem**: The `parameters` NamedTuple stores both `nested` and `natural` fields with **identical numerical values** in different structures. Post v0.3.0, `extract_natural_vector()` is an identity transform.

**Solution**: Remove `natural` field, compute on-demand via `nested_to_natural_vectors()` helper.

#### Test Maintenance (Do BEFORE implementation)

**Type A: Direct field access (`.parameters.natural`) - MUST UPDATE**

| Test File | Lines | Action | Details |
|-----------|-------|--------|---------|
| `unit/test_initialization.jl` | 164 | 🔄 Update | Change `model_init.parameters.natural` → `get_parameters(model_init; scale=:natural)` |
| `unit/test_initialization.jl` | 232 | 🔄 Update | Change `model.parameters.natural` → `get_parameters(model; scale=:natural)` |
| `unit/test_initialization.jl` | 336 | 🔄 Update | Change `model1.parameters.natural` → `get_parameters(model1; scale=:natural)` |
| `unit/test_initialization.jl` | 364 | 🔄 Update | Change `model1.parameters.natural` → `get_parameters(model1; scale=:natural)` |
| `unit/test_surrogates.jl` | 120 | 🔄 Update | Change `surrogate.parameters.natural` → use accessor |
| `unit/test_surrogates.jl` | 168 | 🔄 Update | Change `surrogate.parameters.natural` → use accessor |
| `unit/test_surrogates.jl` | 173 | 🔄 Update | Change `surrogate.parameters.natural` → use accessor |
| `unit/test_surrogates.jl` | 244 | 🔄 Update | Change `model.markovsurrogate.parameters.natural` → use accessor |

**Type A Total**: 8 locations in 2 files MUST be updated (verified via grep)

**Type B: Function call (`get_parameters_natural()`) - MAY need update**

If `get_parameters_natural()` is removed or renamed, these tests need updating:

| Test File | Lines | Action | Details |
|-----------|-------|--------|---------|
| `unit/test_helpers.jl` | 321, 335, 350 | 🔄 Verify | Update if `get_parameters_natural` deprecated |
| `unit/test_phasetype.jl` | 44 (import), 1251 (use) | 🔄 Verify | Update import and usage if deprecated |

**Type B Total**: 5 locations in 2 files, contingent on API decision

**Pattern for Type A updates**:
```julia
# BEFORE:
natural = model.parameters.natural
rate = first(values(surrogate.parameters.natural))[1]

# AFTER:
natural = get_parameters(model; scale=:natural)
rate = get_parameters(surrogate; scale=:natural)[:h12][1]
```

**Implementation decision needed**: Keep `get_parameters_natural()` as a helper function (computes from `.nested`) or deprecate it in favor of `get_parameters(m; scale=:natural)`.

**Type C: Documentation file (MUST UPDATE)**

| File | Line | Action | Details |
|------|------|--------|---------|
| `MultistateModelsTests/reports/architecture.qmd` | 415 | 🔄 Update | Example code shows `model.parameters.natural` - change to accessor |

**Risk**: 🟡 MEDIUM - 8-13 test locations + 1 doc location need updating, straightforward

---

### Item 22: Remove Deprecated `get_loglik(model, "string")` Argument

**Location**: `src/output/accessors.jl` lines 259-270

**Problem**: The `ll::String` parameter is deprecated with a `Base.depwarn`. Users should use `type::Symbol` instead.

```julia
# Current code (deprecated):
get_loglik(model, "loglik")     # Uses deprecated string argument

# Correct usage:
get_loglik(model; type=:loglik)  # Uses symbol keyword
```

#### Test Maintenance (Do BEFORE implementation)

| Test File | Lines | Action | Details |
|-----------|-------|--------|---------|
| **None** | - | ✅ No changes needed | No tests use the deprecated string form |

**Verification**: `grep -rn 'get_loglik.*"' MultistateModelsTests/` shows no usages of string form.

#### Itemized Tasks

| # | Task | File | Lines | Verification |
|---|------|------|-------|--------------|
| 22.1 | Search for callers using string argument | all `src/` | - | `grep -rn 'get_loglik.*"' src/` |
| 22.2 | Update any callers to use keyword form | varies | - | No string args remain |
| 22.3 | Remove `ll::Union{Nothing,String}=nothing` parameter | `src/output/accessors.jl` | 267 | Parameter gone |
| 22.4 | Remove deprecation warning block | `src/output/accessors.jl` | 269-272 | Block gone |
| 22.5 | Update docstring to remove deprecated usage | `src/output/accessors.jl` | 259 | Docstring updated |
| 22.6 | Run full test suite | - | - | All tests pass |

**Expected Result**: ~10 lines removed, cleaner API.

**Risk**: 🟢 LOW - Removing already-deprecated code path

---

### Item 23: Remove Deprecated `fit_phasetype_surrogate()` Function

**Location**: `src/surrogate/markov.jl` lines 439-455

**Problem**: The docstring explicitly states: "This function is deprecated. Use `fit_surrogate(model; type=:phasetype, ...)` or `_build_phasetype_from_markov()` instead."

#### Test Maintenance (Do BEFORE implementation)

| Test File | Lines | Action | Details |
|-----------|-------|--------|---------|
| `unit/test_surrogates.jl` | 186 | 🔄 Update | Change `MultistateModels._fit_phasetype_surrogate(...)` → `MultistateModels._build_phasetype_from_markov(...)` |

**Verification**: 
```bash
grep -rn "fit_phasetype_surrogate" MultistateModelsTests/
# → MultistateModelsTests/unit/test_surrogates.jl:186
```

#### Itemized Tasks

| # | Task | File | Lines | Verification |
|---|------|------|-------|--------------|
| 23.1 | Update test to use new API | `MultistateModelsTests/unit/test_surrogates.jl` | 186 | Test uses `_build_phasetype_from_markov` |
| 23.2 | Search for other callers | all `src/` | - | `grep -rn "fit_phasetype_surrogate" src/` |
| 23.3 | Update any internal callers | varies | - | No deprecated calls remain |
| 23.4 | Delete deprecated function | `src/surrogate/markov.jl` | 433-455 | Function gone |
| 23.5 | Run full test suite | - | - | All tests pass |

**Expected Result**: ~22 lines removed, single canonical API path.

**Risk**: 🟢 LOW - Removing already-deprecated function, test update is straightforward

---

### Item 8: get_ij_vcov() and get_jk_vcov() Internal Helpers

**Location**: `src/output/accessors.jl` lines 627-634

**Problem**: These are trivial one-line wrappers that add no value.

#### Test Maintenance (Do BEFORE implementation)

| Test File | Lines | Action | Details |
|-----------|-------|--------|---------|
| **None** | - | ✅ No changes needed | No tests reference `get_ij_vcov` or `get_jk_vcov` |

**Verification**: `grep -r "get_ij_vcov\|get_jk_vcov" MultistateModelsTests/` returns no matches.

#### Itemized Tasks

| # | Task | File | Lines | Verification |
|---|------|------|-------|--------------|
| 8.1 | Find all call sites of `get_ij_vcov` | all `src/` | - | `grep -rn "get_ij_vcov" src/` |
| 8.2 | Find all call sites of `get_jk_vcov` | all `src/` | - | `grep -rn "get_jk_vcov" src/` |
| 8.3 | Replace each `get_ij_vcov(m)` with `get_vcov(m; type=:ij)` | varies | - | All call sites updated |
| 8.4 | Replace each `get_jk_vcov(m)` with `get_vcov(m; type=:jk)` | varies | - | All call sites updated |
| 8.5 | Delete `get_ij_vcov` function | `src/output/accessors.jl` | 627-630 | Function gone |
| 8.6 | Delete `get_jk_vcov` function | `src/output/accessors.jl` | 631-634 | Function gone |
| 8.7 | Run full test suite | - | - | All tests pass |

**Expected Result**: 8 lines deleted, simpler API.

**Risk**: LOW - Internal functions, not exported

---

### Item 9: FlattenAll Unused Type

**Location**: `src/utilities/flatten.jl`

**Problem**: `FlattenAll` type exists but appears unused in production code.

#### Test Maintenance (Do BEFORE implementation)

| Test File | Lines | Action | Details |
|-----------|-------|--------|---------|
| `unit/test_reconstructor.jl` | 9 | 🔄 Update | Remove `FlattenAll` from import statement |
| `unit/test_reconstructor.jl` | 16 | ❌ Delete | Remove test `@test FlattenAll <: MultistateModels.FlattenTypes` |
| `unit/test_reconstructor.jl` | 82-85 | ❌ Delete | Remove testset "Integer vector with FlattenAll - should flatten" |

**Details**:
```bash
grep -rn "FlattenAll" MultistateModelsTests/
# → unit/test_reconstructor.jl:9   (import)
# → unit/test_reconstructor.jl:16  (type check test)
# → unit/test_reconstructor.jl:82  (comment)
# → unit/test_reconstructor.jl:84  (actual usage in test)
```

**Action**: Delete the `FlattenAll` tests before deleting the type from source.

#### Itemized Tasks

| # | Task | File | Lines | Verification |
|---|------|------|-------|--------------|
| 9.1 | Search for `FlattenAll` in source | `src/` | - | `grep -rn "FlattenAll" src/` |
| 9.2 | Search for `FlattenAll` in tests | `MultistateModelsTests/` | - | `grep -rn "FlattenAll" MultistateModelsTests/` |
| 9.3 | If only in definition + tests, note test locations | - | - | List test files |
| 9.4 | Delete `FlattenAll` struct definition | `src/utilities/flatten.jl` | varies | Struct gone |
| 9.5 | Delete any `FlattenAll` branches in `construct_flatten` | `src/utilities/flatten.jl` | varies | No `FlattenAll` references |
| 9.6 | Update/delete affected tests | `MultistateModelsTests/` | varies | Tests updated or removed |
| 9.7 | Run full test suite | - | - | All tests pass |

**Expected Result**: Simpler flatten.jl with only `FlattenContinuous`.

**Risk**: LOW-MEDIUM - May be used in tests

---

### Item 6: EnzymeBackend and MooncakeBackend - EXPORTED BUT UNSUPPORTED

**Location**: `src/types/infrastructure.jl` lines 33-115

**Problem**: These backends are exported in the public API but the code warns against using them. `MooncakeBackend` explicitly warns it "may fail for Markov models due to LAPACK calls". Only `ForwardDiffBackend` is production-ready.

#### Test Maintenance (Do BEFORE implementation)

| Test File | Lines | Action | Details |
|-----------|-------|--------|---------|
| `unit/test_ad_backends.jl` | 3 | 🔄 Update | Update comment to reflect internal status |
| `unit/test_ad_backends.jl` | 21 | 🔄 Update | Change import to `import MultistateModels: EnzymeBackend, MooncakeBackend` (internal access) |
| `unit/test_ad_backends.jl` | 61-67 | 🔄 Keep | Tests still valid (testing internal types work), just not exported |
| `unit/test_ad_backends.jl` | 72-77 | 🔄 Keep | Tests still valid (testing internal types work) |

**Note**: Tests can still test internal (unexported) types using `MultistateModels.EnzymeBackend`. The tests should continue to work but verify the types exist and function correctly even though they're not exported.

**Decision Point**: Consider whether to keep the AD backend tests or mark them as internal-only tests (run only during development).

#### Itemized Tasks

| # | Task | File | Lines | Verification |
|---|------|------|-------|--------------|
| 6.1 | Find export statement | `src/MultistateModels.jl` | ~80-90 | Locate `EnzymeBackend, MooncakeBackend` in exports |
| 6.2 | Remove from exports | `src/MultistateModels.jl` | varies | Delete `EnzymeBackend,` and `MooncakeBackend,` from export list |
| 6.3 | Keep struct definitions internal | `src/types/infrastructure.jl` | 33-115 | Leave code, just unexport |
| 6.4 | Update docstrings to note internal status | `src/types/infrastructure.jl` | varies | Add "Internal: not part of public API" |
| 6.5 | Search for external usage in tests | `MultistateModelsTests/` | - | `grep -r "EnzymeBackend\|MooncakeBackend" MultistateModelsTests/` |
| 6.6 | Run full test suite | - | - | All tests pass |

**Expected Result**: Backends remain available internally but not exported.

**Risk**: MEDIUM - Users may have code referencing these types

---

### Item 10: CachedTransformStrategy vs DirectTransformStrategy

**Location**: `src/simulation/simulate.jl` lines 35-59

**Problem**: Two strategies exist but `DirectTransformStrategy` appears unused except in docstrings.

#### Test Maintenance (Do BEFORE implementation)

| Test File | Lines | Action | Details |
|-----------|-------|--------|---------|
| **None found** | - | ✅ No changes needed | No tests reference `DirectTransformStrategy` or `CachedTransformStrategy` |

**Verification**: `grep -r "DirectTransformStrategy\|CachedTransformStrategy" MultistateModelsTests/` returns no matches.

**Note**: If tests are added for this item (benchmarking the strategies), they should be placed in `MultistateModelsTests/benchmarks/` rather than `unit/`.

#### Itemized Tasks

| # | Task | File | Lines | Verification |
|---|------|------|-------|--------------|
| 10.1 | Search for `DirectTransformStrategy` usage | `src/` | - | `grep -rn "DirectTransformStrategy" src/` |
| 10.2 | Search for `CachedTransformStrategy` usage | `src/` | - | `grep -rn "CachedTransformStrategy" src/` |
| 10.3 | If `DirectTransformStrategy` only in docstrings, note this | - | - | Document finding |
| 10.4 | Create benchmark comparing both strategies | - | - | 1000 path simulations each |
| 10.5 | If no significant difference (<5%), consolidate to single impl | `src/simulation/simulate.jl` | varies | Remove abstraction |
| 10.6 | If significant difference, document when to use each | `src/simulation/simulate.jl` | varies | Add guidance to docstring |
| 10.7 | Run full test suite | - | - | All tests pass |

**Expected Result**: Either single strategy or documented performance guidance.

**Risk**: MEDIUM - Need performance data before deciding

---

---

## 📦 WAVE 3: Mathematical Correctness Bugs

Complete Waves 1-2 first. These affect the penalty/spline infrastructure and must be fixed before Item #19.

### Item 16: default_nknots() Uses Regression Spline Formula

**Location**: `src/hazard/spline.jl` line 425 (verified)

**Problem**: Uses `floor(n^(1/5))` from Tang et al. (2017), appropriate for **regression splines (sieve estimation)**, NOT penalized splines.

**Impact**: For penalized splines, the penalty controls overfitting, so more knots are acceptable (often desirable for flexibility). The current conservative formula may under-fit.

#### Test Maintenance (Do BEFORE implementation)

| Test File | Lines | Action | Details |
|-----------|-------|--------|---------|
| `unit/test_splines.jl` | 391-398 | 🔄 Update | Tests verify specific values from `n^(1/5)` formula; update expected values if formula changes |
| `unit/test_splines.jl` | 628-637 | 🔄 Update | Tests verify monotonicity and bounds; may need updates depending on new formula |

**Details**: Current tests include:
```julia
@test MultistateModels.default_nknots(0) == 0
@test MultistateModels.default_nknots(1) == 2  # min 2
@test MultistateModels.default_nknots(10) == 2
@test MultistateModels.default_nknots(32) == 2  # 32^(1/5) ≈ 2.0
@test MultistateModels.default_nknots(100) == 2  # 100^(1/5) ≈ 2.51
@test MultistateModels.default_nknots(1000) == 3  # 1000^(1/5) ≈ 3.98
@test MultistateModels.default_nknots(10000) == 6  # 10000^(1/5) ≈ 6.31
```

**Strategy**:
1. If creating a NEW function `default_nknots_penalized()` (recommended), existing tests remain valid
2. If modifying `default_nknots()` behavior, update these test expectations
3. Add new tests for `default_nknots_penalized()` with appropriate expected values

#### Itemized Tasks

| # | Task | File | Lines | Verification |
|---|------|------|-------|--------------|
| 16.1 | Find `default_nknots` function | `src/hazard/spline.jl` | ~432 | `grep -n "default_nknots" src/hazard/spline.jl` |
| 16.2 | Read current formula implementation | `src/hazard/spline.jl` | ~432-450 | Note `floor(n^(1/5))` |
| 16.3 | Research P-spline knot recommendations | EXTERNAL | - | Wood (2017), Eilers & Marx (1996), Ruppert (2002) |
| 16.4 | Create `default_nknots_penalized(n)::Int` function | `src/hazard/spline.jl` | NEW | Return `min(max(10, floor(Int, n^(1/3))), 30)` |
| 16.5 | Modify call sites to choose formula based on penalty | varies | - | `penalty === nothing ? default_nknots(n) : default_nknots_penalized(n)` |
| 16.6 | Add docstring explaining when each is appropriate | `src/hazard/spline.jl` | NEW | Document rationale |
| 16.7 | Create test comparing knot counts | `MultistateModelsTests/unit/test_default_knots.jl` | NEW | Test `n=100,500,1000` |
| 16.8 | Run full test suite | - | - | All tests pass |

**Expected Result**: Penalized splines get appropriate default knot counts.

**Risk**: LOW-MEDIUM - May change default behavior

---

### Item 15: Monotone Spline Penalty Matrix Incorrect — CONFIRMED BUG ⚠️

**Location**: `src/types/infrastructure.jl`, `compute_penalty()`

**Problem**: Penalty matrix `S` is built for B-spline coefficients (`coefs`), but parameters being penalized are I-spline increments (`ests`).

**Mathematical Issue**:
For monotone splines, the transformation is:
```julia
coefs = L * ests  # where L is lower-triangular cumsum with knot weights
```

The correct penalty should be:
```julia
P(ests) = (λ/2) coefs^T S coefs
        = (λ/2) (L * ests)^T S (L * ests)
        = (λ/2) ests^T (L^T S L) ests
```

So `S_monotone = L^T * S * L` — but this transformation is **not currently implemented**.

#### Test Maintenance (Do BEFORE implementation)

| Test File | Lines | Action | Details |
|-----------|-------|--------|---------|
| `unit/test_penalty_infrastructure.jl` | 53-96 | ✅ Keep | Tests for `build_penalty_matrix` with non-monotone splines remain valid |
| `unit/test_penalty_infrastructure.jl` | NEW | ➕ Add | Add new testset for monotone spline penalty matrix correctness |

**New tests needed**:
```julia
@testset "Monotone spline penalty matrix transformation" begin
    # Create monotone spline basis
    basis = MultistateModels.BSplineBasis(4, [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0])
    
    # Get L matrix
    L = MultistateModels.build_ispline_transform_matrix(basis)
    
    # Get B-spline penalty S
    S_bspline = MultistateModels.build_penalty_matrix(basis, 2)
    
    # Transformed penalty should be L' * S * L
    S_expected = L' * S_bspline * L
    
    # Test that build_penalty_config applies this transformation for monotone hazards
    # ...
end
```

**Verification**: Existing tests for non-monotone splines (`monotone=0`) should continue to pass unchanged.

#### Itemized Tasks

| # | Task | File | Lines | Verification |
|---|------|------|-------|--------------|
| 15.1 | Read current GPS penalty computation | `src/utilities/spline_utils.jl` | varies | Find `gps_penalty_matrix` function |
| 15.2 | Read current `compute_penalty` function | `src/types/infrastructure.jl` | varies | Find penalty application logic |
| 15.3 | Identify where `monotone` flag is checked | `src/types/infrastructure.jl` or `src/utilities/penalty_config.jl` | varies | Find conditional for monotone |
| 15.4 | Create function `build_ispline_transform_matrix(basis, k)::Matrix` | `src/utilities/spline_utils.jl` | NEW | Returns lower-triangular L matrix |
| 15.5 | Implement L matrix construction per I-spline definition | `src/utilities/spline_utils.jl` | NEW | `L[i,j] = (t[j+k] - t[j]) / k if j ≤ i else 0` |
| 15.6 | Create function `transform_penalty_monotone(S, L)::Matrix` | `src/utilities/spline_utils.jl` | NEW | Returns `L' * S * L` |
| 15.7 | Modify `build_penalty_config` to apply transformation when `monotone != 0` | `src/utilities/penalty_config.jl` | varies | `S_ests = monotone == 0 ? S : transform_penalty_monotone(S, L)` |
| 15.8 | Create test with known correct penalty | `MultistateModelsTests/unit/test_monotone_penalty.jl` | NEW | Hand-computed L, S, expected result |
| 15.9 | Run full test suite | - | - | All tests pass |

**Expected Result**: Monotone spline penalties mathematically correct.

**Severity**: MEDIUM — monotone splines will have incorrect smoothing, but non-monotone splines are unaffected.

**Risk**: MEDIUM - Core numerical correctness issue

---

### Item 5: rectify_coefs! Review - VERIFY CORRECTNESS WITH NATURAL SCALE PARAMS

**Location**: `src/hazard/spline.jl` lines 975-1010

**Problem**: This function performs a round-trip transformation to clean up numerical zeros in spline coefficients. With the recent refactoring to store ALL parameters on natural scale (v0.3.0+), this function's logic needs verification.

#### Test Maintenance (Do BEFORE implementation)

| Test File | Lines | Action | Details |
|-----------|-------|--------|---------|
| `unit/test_splines.jl` | 459-490 | ✅ Verify | Existing tests cover `rectify_coefs!`; verify they pass with current implementation |
| `unit/test_splines.jl` | 517-537 | ✅ Verify | Tests for round-trip consistency; ensure they validate natural scale behavior |

**Details**: Existing tests at lines 462-537 include:
- Basic `rectify_coefs!` functionality test
- Round-trip consistency test (applying twice gives same result)

**Strategy**: These tests should continue to work. If they fail during verification, that indicates a bug in `rectify_coefs!` that needs fixing.

#### Itemized Tasks

| # | Task | File | Lines | Verification |
|---|------|------|-------|--------------|
| 5.1 | Read `_spline_ests2coefs` function | `src/construction/spline_builder.jl` | ~288-330 | Verify NO `exp()` or `log()` transforms |
| 5.2 | Read `_spline_coefs2ests` function | `src/construction/spline_builder.jl` | ~335-370 | Verify NO `exp()` or `log()` transforms |
| 5.3 | Verify `rectify_coefs!` parameter extraction | `src/hazard/spline.jl` | 975-985 | `collect(values(hazard_params.baseline))` correct order? |
| 5.4 | Add comment block explaining parameter scale | `src/hazard/spline.jl` | 975 | Add `# NOTE: All params on natural scale (v0.3.0+)` |
| 5.5 | Search for existing tests | `MultistateModelsTests/` | - | `grep -r "rectify_coefs" MultistateModelsTests/` |
| 5.6 | If no tests exist, create unit test | `MultistateModelsTests/unit/test_splines.jl` | NEW | Test round-trip: `rectify_coefs!(copy(params), model)` preserves values |
| 5.7 | Run full test suite | - | - | All tests pass |

**Expected Result**: Verified correct or fixed if wrong, with test coverage.

**Risk**: LOW-MEDIUM - Function appears correct but needs test verification

---

### Item 17: Automatic Knot Placement Uses Raw Data Instead of Surrogate Simulation

**Location**: `src/construction/multistatemodel.jl` lines ~415-560

**Problem**: `_build_spline_hazard()` extracts sojourns directly from observed data:
```julia
paths = extract_paths(data, model.totalhazaliases)
sojourns = extract_sojourns(paths, hazard.statefrom, hazard.stateto)
```

**Impact**: For **panel data** (obstype=2), exact transition times are NOT observed—only the state at discrete times. Extracting sojourns from panel data gives meaningless or biased time ranges.

#### Test Maintenance (Do BEFORE implementation)

| Test File | Lines | Action | Details |
|-----------|-------|--------|---------|
| **None directly affected** | - | ➕ Add new tests | No existing tests cover panel data with automatic knots |

**New test file needed**: `MultistateModelsTests/integration/test_panel_auto_knots.jl`

**Test scenarios to add**:
```julia
@testset "Automatic knot placement with panel data" begin
    # Create model with panel data (obstype=2)
    # Verify knots are placed using surrogate-simulated sojourns, not raw data
    # Compare knot locations to surrogate simulation
end
```

**Note**: This is a feature enhancement, not a bug fix for existing functionality. New tests verify the new behavior.

#### Itemized Tasks

| # | Task | File | Lines | Verification |
|---|------|------|-------|--------------|
| 17.1 | Find `_build_spline_hazard` function | `src/construction/multistatemodel.jl` | ~415 | `grep -n "_build_spline_hazard" src/construction/multistatemodel.jl` |
| 17.2 | Identify sojourn extraction logic | `src/construction/multistatemodel.jl` | ~450-470 | Find `extract_sojourns` call |
| 17.3 | Identify where obstype is available | `src/construction/multistatemodel.jl` | varies | Check if data.obstype accessible |
| 17.4 | Check if surrogate model is available at this point | `src/construction/multistatemodel.jl` | varies | Surrogate fitted before/after? |
| 17.5 | Create function `simulate_sojourns_from_surrogate(surrogate, transition, n)::Vector{Float64}` | `src/surrogate/markov.jl` | NEW | Simulate paths, extract sojourns |
| 17.6 | Modify `_build_spline_hazard` to check for panel data | `src/construction/multistatemodel.jl` | ~450 | `has_panel = any(data.obstype .== 2)` |
| 17.7 | If panel + automatic knots: use surrogate simulation | `src/construction/multistatemodel.jl` | ~455 | Call `simulate_sojourns_from_surrogate` |
| 17.8 | Add warning if surrogate not yet fitted | `src/construction/multistatemodel.jl` | ~455 | `@warn "Cannot use surrogate..."` |
| 17.9 | Create test with panel data + automatic knots | `MultistateModelsTests/integration/test_panel_auto_knots.jl` | NEW | Verify knots placed reasonably |
| 17.10 | Run full test suite | - | - | All tests pass |

**Expected Result**: Automatic knot placement works correctly for panel data.

**Risk**: MEDIUM - Requires architectural understanding of model construction order

---

### Item 18: PIJCV Hessian NaN/Inf Root Cause

**Locations**:
- `src/inference/smoothing_selection.jl` line ~801 - `_solve_loo_newton_step()`
- `src/inference/smoothing_selection.jl` line ~2039 - `compute_edf()`

**Problem**: Warnings like:
```
"Subject Hessian contains NaN/Inf values"
"Failed to invert penalized Hessian for EDF computation"
```

**Status**: ✅ Graceful fallback added (returns fallback value instead of crashing)
**Remaining**: ⚠️ Root cause not investigated

#### Test Maintenance (Do BEFORE implementation)

| Test File | Lines | Action | Details |
|-----------|-------|--------|---------|
| `unit/test_pijcv.jl` | 26, 30, 413-426, 538-540 | ✅ Keep | Existing PIJCV tests with normal Hessians remain valid |
| `unit/test_efs.jl` | 65-74, 118-121 | ✅ Keep | Existing EFS tests with normal Hessians remain valid |
| **NEW** | - | ➕ Add | Add test for NaN/Inf Hessian handling and root cause |

**New test needed**: `MultistateModelsTests/unit/test_hessian_nan.jl`

**Test scenarios**:
```julia
@testset "Hessian NaN/Inf handling" begin
    @testset "Detection" begin
        # Create minimal model that triggers NaN Hessian
        # Verify warning is emitted
        # Verify fallback value is returned
    end
    
    @testset "Root cause scenarios" begin
        # Test log(0) in hazard evaluation
        # Test spline evaluation at boundary
        # Test extreme parameter values
    end
end
```

**Note**: Investigation may reveal parameter configurations that trigger NaN. Tests should document these edge cases.

#### Itemized Tasks

| # | Task | File | Lines | Verification |
|---|------|------|-------|--------------|
| 18.1 | Find `_solve_loo_newton_step` function | `src/inference/smoothing_selection.jl` | ~801 | `grep -n "_solve_loo_newton_step" src/inference/smoothing_selection.jl` |
| 18.2 | Find `compute_edf` function | `src/inference/smoothing_selection.jl` | ~2039 | `grep -n "compute_edf" src/inference/smoothing_selection.jl` |
| 18.3 | Add diagnostic: log parameters before Hessian computation | `src/inference/smoothing_selection.jl` | before Hessian call | `@debug "Pre-Hessian params: $(params)"` |
| 18.4 | Add diagnostic: log likelihood value before Hessian | `src/inference/smoothing_selection.jl` | before Hessian call | `@debug "Likelihood: $(ll)"` |
| 18.5 | Create minimal reproducer test case | `MultistateModelsTests/unit/test_hessian_nan.jl` | NEW | Test that triggers NaN warning |
| 18.6 | Run reproducer with `JULIA_DEBUG=MultistateModels` | - | - | Capture parameter state |
| 18.7 | Identify parameter values that cause NaN | - | - | Document which params trigger |
| 18.8 | Check for `log(0)` in hazard evaluation | `src/hazard/*.jl` | varies | `log(x)` where `x ≤ 0` possible |
| 18.9 | Check spline basis evaluation at boundaries | `src/hazard/spline.jl` | varies | `x < knot_min` or `x > knot_max` |
| 18.10 | Fix root cause (add guards or clamp parameters) | varies | varies | Root cause fixed |
| 18.11 | Run full test suite | - | - | All tests pass, no NaN warnings |

**Expected Result**: Root cause fixed, no more NaN/Inf warnings in normal operation.

**Risk**: MEDIUM - May require deep debugging

---

---

## 📦 WAVE 4: Major Features

Complete Waves 1-3 first. These are the architectural changes.

### Item 7: Duplicate Variance Computation Functions

**Location**: `src/output/variance.jl` (2672 lines - largest file)

**Problem**: Multiple near-duplicate functions exist for computing subject Hessians.

#### Test Maintenance (Do BEFORE implementation)

| Test File | Lines | Action | Details |
|-----------|-------|--------|---------|
| `unit/test_penalty_infrastructure.jl` | 309-322 | 🔄 Update if needed | Uses `compute_subject_hessians_fast`; update if function name changes |
| `unit/test_pijcv.jl` | 30 | 🔄 Update if needed | Imports `compute_subject_hessians_fast`; update import if name changes |
| `unit/test_efs.jl` | 65-74, 118-121 | 🔄 Update if needed | Uses `compute_subject_hessians_fast`; update if signature changes |

**Strategy**: 
- If consolidating to single function with `parallel::Bool` parameter, update all call sites
- Ensure tests cover both parallel and non-parallel paths after consolidation

**Pattern for updates**:
```julia
# BEFORE (if functions are consolidated):
subject_hessians = compute_subject_hessians_fast(beta, model, paths)
# or
subject_hessians = compute_subject_hessians_threaded(beta, model, paths)

# AFTER:
subject_hessians = compute_subject_hessians(beta, model, paths; parallel=false)
# or
subject_hessians = compute_subject_hessians(beta, model, paths; parallel=true)
```

#### Itemized Tasks

| # | Task | File | Lines | Verification |
|---|------|------|-------|--------------|
| 7.1 | Identify all `compute_subject_hessians*` functions | `src/output/variance.jl` | - | `grep -n "^function compute_subject_hessians" src/output/variance.jl` |
| 7.2 | Document the signature and purpose of each | - | - | Create comparison table |
| 7.3 | Identify which is called by `_fit_exact` | `src/inference/fit_exact.jl` | ~110 | Note the dispatch chain |
| 7.4 | Identify which is called by `_fit_markov_panel` | `src/inference/fit_markov.jl` | ~75 | Note the dispatch chain |
| 7.5 | Diff `_batched` vs `_threaded` implementations | `src/output/variance.jl` | varies | Identify actual differences |
| 7.6 | Benchmark both on 100-subject model | - | - | `@time` or `@benchmark` |
| 7.7 | If identical, consolidate into single function with `parallel::Bool` param | `src/output/variance.jl` | varies | Reduce duplication |
| 7.8 | Update all call sites | `src/inference/fit_*.jl` | varies | Use new unified function |
| 7.9 | Run full test suite | - | - | All tests pass |

**Expected Result**: Single `compute_subject_hessians` with optional parallelization, ~200 lines saved.

**Risk**: MEDIUM - Requires careful testing of numerical equivalence

---

### Item 20: Per-Transition Surrogate Specification

**Goal**: Allow users to specify surrogate type (exponential vs phase-type) separately for each transition, rather than model-wide.

**Current API** (model-wide only):
```julia
multistatemodel(h12, h21; data=df, surrogate=:auto)  # Same for ALL transitions
```

**Proposed API** (per-transition):
```julia
multistatemodel(h12, h21; 
    data=df, 
    surrogate = Dict(
        (1,2) => :phasetype,   # Use phase-type for 1→2
        (2,1) => :markov       # Use Markov for 2→1
    )
)
```

#### Test Maintenance (Do BEFORE implementation)

| Test File | Lines | Action | Details |
|-----------|-------|--------|---------|
| `unit/test_surrogates.jl` | all | ✅ Keep | Existing tests use Symbol API; should remain valid |
| **NEW** | - | ➕ Add | Add tests for Dict-based per-transition specification |

**New tests needed**:
```julia
@testset "Per-transition surrogate specification" begin
    @testset "Dict API" begin
        # Test Dict{Tuple{Int,Int}, Symbol} API
        # Verify different surrogates used for different transitions
    end
    
    @testset "Validation" begin
        # Test error for invalid transition tuples
        # Test error for invalid surrogate symbols
        # Test warning for missing transitions (defaults to :auto)
    end
    
    @testset "Backward compatibility" begin
        # Test Symbol API still works (model-wide)
    end
end
```

**Note**: This is a feature addition. Existing tests using the Symbol API should continue to work unchanged.

#### Implementation Plan

##### Phase 1: Update Type Signatures

| # | Task | File | Lines | Details | Verification |
|---|------|------|-------|---------|--------------|
| 20.1.1 | Find `multistatemodel` signature | `src/construction/multistatemodel.jl` | ~1-50 | Locate `surrogate` parameter | Found |
| 20.1.2 | Change `surrogate` type | `src/construction/multistatemodel.jl` | signature | `surrogate::Union{Symbol, Dict{Tuple{Int,Int}, Symbol}} = :auto` | Type updated |
| 20.1.3 | Add `surrogate_n_phases` parameter | `src/construction/multistatemodel.jl` | signature | `surrogate_n_phases::Union{Int, Dict{Tuple{Int,Int}, Int}, Symbol} = :heuristic` | Param added |
| 20.1.4 | Verify signature compiles | - | - | `include("src/construction/multistatemodel.jl")` | No errors |

##### Phase 2: Update Validation Logic

| # | Task | File | Lines | Details | Verification |
|---|------|------|-------|---------|--------------|
| 20.2.1 | Find surrogate validation | `src/construction/multistatemodel.jl` | varies | Locate `_validate_surrogate` or equivalent | Found |
| 20.2.2 | Add Dict validation | `src/construction/multistatemodel.jl` | within validation | Check keys are valid `(from, to)` tuples | Error on invalid |
| 20.2.3 | Add Dict value validation | `src/construction/multistatemodel.jl` | within validation | Values ∈ `[:markov, :phasetype, :auto]` | Error on invalid |
| 20.2.4 | Add missing transition warning | `src/construction/multistatemodel.jl` | within validation | `@warn "Transition (i,j) not in surrogate Dict, using :auto"` | Warning emitted |

##### Phase 3-6: See detailed implementation in original Item 20 description (if needed)

**Risk**: LOW - Feature addition, not modifying existing behavior

---

### Item 14: make_constraints Export Verification

**Location**: `src/utilities/misc.jl` lines 26-70

**Problem**: Advanced feature with only 5 grep hits, mostly in docstrings. No test coverage visible.

#### Test Maintenance (Do BEFORE implementation)

| Test File | Lines | Action | Details |
|-----------|-------|--------|---------|
| **None exist** | - | ➕ Add or ❌ Remove from exports | No tests for `make_constraints` |

**Decision point**:
1. **Keep exported**: Add test file `MultistateModelsTests/unit/test_constraints.jl`
2. **Remove from exports**: Keep function internal, no tests needed

**If adding tests**:
```julia
@testset "make_constraints" begin
    @testset "Basic functionality" begin
        # Create model
        # Define constraints
        # Verify constraint format is correct for Ipopt
    end
    
    @testset "Error handling" begin
        # Test invalid constraint specification
        # Test constraint on non-existent parameter
    end
end
```

#### Itemized Tasks

| # | Task | File | Lines | Verification |
|---|------|------|-------|--------------|
| 14.1 | Count all usages | all `src/` and tests | - | `grep -r "make_constraints" src/ MultistateModelsTests/` |
| 14.2 | Check if exported | `src/MultistateModels.jl` | - | `grep "make_constraints" src/MultistateModels.jl` |
| 14.3 | Identify production call sites (exclude docstrings) | varies | - | Manual review of grep output |
| 14.4 | If no production use: remove from exports | `src/MultistateModels.jl` | export line | Remove from export list |
| 14.5 | If keeping: add test coverage | `MultistateModelsTests/unit/test_constraints.jl` | NEW | Test `make_constraints` with known inputs |
| 14.6 | Run full test suite | - | - | All tests pass |

**Expected Result**: Either proper test coverage or removed from public API.

**Risk**: LOW - Niche feature

---

### Item 12: calibrate_splines / calibrate_splines! Verification

**Location**: `src/hazard/spline.jl` lines 500-630

**Problem**: These exported functions may not have test coverage. Need to verify they are actually used and tested.

#### Test Maintenance (Do BEFORE implementation)

| Test File | Lines | Action | Details |
|-----------|-------|--------|---------|
| `unit/test_splines.jl` | 687-822 | ✅ Keep | Comprehensive tests exist for `calibrate_splines` and `calibrate_splines!` |
| `unit/test_bounds.jl` | 96 | ✅ Keep | Uses `calibrate_splines!` in bounds testing |

**Status**: Tests already exist! This item is about **verifying** coverage, not adding it.

**Existing test coverage** (from `unit/test_splines.jl`):
- Lines 687-710: `calibrate_splines - basic functionality`
- Lines 711-722: `calibrate_splines - explicit nknots`
- Lines 724-730: `calibrate_splines - explicit quantiles`
- Lines 732-753: `calibrate_splines - error handling`
- Lines 756-782: `calibrate_splines! - in-place modification`
- Lines 784-804: `calibrate_splines! - parameter structure integrity`
- Lines 806-817: `calibrate_splines! - model remains functional`
- Lines 820-824: `calibrate_splines! - set_parameters! works after calibration`

**Conclusion**: Test coverage is adequate. This item can be marked as ✅ VERIFIED (tests exist).

#### Itemized Tasks

| # | Task | File | Lines | Verification |
|---|------|------|-------|--------------|
| 12.1 | Search for test coverage | `MultistateModelsTests/` | - | `grep -r "calibrate_splines" MultistateModelsTests/` |
| 12.2 | Search for export statement | `src/MultistateModels.jl` | - | `grep -n "calibrate_splines" src/MultistateModels.jl` |
| 12.3 | Count all usage in source | all `src/` | - | `grep -r "calibrate_splines" src/ \| wc -l` |
| 12.4 | If no tests, create unit test file | `MultistateModelsTests/unit/test_calibrate_splines.jl` | NEW | Test creates model, calls `calibrate_splines!`, verifies spline bases updated |
| 12.5 | Alternatively: remove from exports if internal only | `src/MultistateModels.jl` | export line | Remove from export list |
| 12.6 | Run full test suite | - | - | All tests pass |

**Expected Result**: Either proper test coverage or removed from public API.

**Risk**: LOW-MEDIUM - May be used by external users

---

---

## Summary Statistics

| Wave | Items | Focus | Risk |
|------|-------|-------|------|
| **Wave 1** | 6 | Foundation cleanup - zombie code, dead comments | 🟢 LOW |
| **Wave 2** | 5 | Technical debt - parameter redundancy, unused types | 🟡 LOW-MED |
| **Wave 3** | 5 | Mathematical correctness - spline/penalty bugs | 🔴 MED-HIGH |
| **Wave 4** | 5 | Major features - penalized fitting, variance | 🔴 HIGH |

| Category | Count | Items |
|----------|-------|-------|
| Pure deletions (low risk) | 8 | #1, #2, #3, #4, #8, #9, #11, #13 |
| Verification/fix needed | 5 | #5, #6, #10, #12, #14 |
| Bug fixes (math correctness) | 4 | #15, #16, #17, #18 |
| Structural refactoring | 2 | #7, #21 |
| New features/architecture | 2 | #19, #20 |
| **Total** | **21** | - |

---

## Validation Checklist (from handoff)

### Mathematical Correctness
- [x] All parameters stored on natural scale (no transforms)
- [x] Penalty is quadratic P(β) = (λ/2) β^T S β
- [x] Box constraints enforce positivity
- [x] NaNMath.log prevents DomainError
- [x] Phase-type parameter naming correct (FIXED 2026-01-07)
- [ ] Survival probabilities S(t) ∈ [0, 1] for all t — NEEDS VALIDATION
- [ ] Log-likelihoods ℓ(θ) ≤ 0 for all θ — LIKELY FIXED, NEEDS VALIDATION
- [ ] All fitted parameters in plausible ranges — LIKELY FIXED, NEEDS VALIDATION

### Numerical Stability
- [x] PIJCV handles NaN/Inf Hessians gracefully (with fallback)
- [ ] PIJCV Hessians are finite (root cause) — NEEDS INVESTIGATION
- [ ] Optimizer converges for reasonable data — LIKELY FIXED, NEEDS VALIDATION
- [ ] Variance-covariance computation succeeds — NEEDS VALIDATION

---

## Completion Checklist

When all items are complete:
- [ ] All tests pass
- [ ] No new warnings introduced
- [ ] This document shows all items as ✅ COMPLETED
- [ ] Consider archiving this document to `scratch/completed/`

---

## Change Log

| Date | Agent | Changes Made |
|------|-------|--------------|
| 2026-01-08 | Initial | Created document from code audit |
| 2026-01-08 | Update | Integrated context from PENALIZED_SPLINES_BRANCH_HANDOFF_20260107.md: added test status, 4 spline infrastructure bugs (#15-18), phase-type parameter indexing context, validation checklist, planned feature |
| 2026-01-08 | Update | Added Item #19: Detailed architectural plan for penalized likelihood fitting integration with automatic smoothing parameter selection (exact, MCEM, and Markov panel methods) |
| 2026-01-08 | Expansion | Converted all items to meticulous itemized task lists with specific file paths, line numbers, function signatures, and verification steps |
| 2026-01-08 | Addition | Added Item #21: Remove `parameters.natural` redundancy — detailed plan with 7 phases covering helper function creation, 20 call site updates, and documentation changes |
| 2026-01-08 | Reorganization | **Major restructuring for implementation success**: (1) Added 4-wave implementation order at top with dependencies; (2) Reorganized all items by wave instead of priority; (3) Added decision points section; (4) Consolidated duplicate sections; (5) Updated summary statistics |
| 2026-01-08 | Test Maintenance | **Added Test Maintenance sections to ALL items**: (1) Added "Test Maintenance Summary by Wave" overview section; (2) Added "Test Maintenance (Do BEFORE implementation)" subsection to each item listing specific test files, line numbers, and required changes; (3) Updated workflow instructions to prioritize test updates before implementation; (4) Total: ~30 test locations identified across 15 test files |