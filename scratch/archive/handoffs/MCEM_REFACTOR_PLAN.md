# MCEM Surrogate-Agnostic Refactor Plan

**Created**: 2026-01-21  
**Branch**: `penalized_splines`  
**Status**: PLANNING COMPLETE, READY FOR IMPLEMENTATION

---

## 1. Goal

Refactor `fit_mcem.jl` so MCEM has **ONE flow regardless of surrogate type**. All `MarkovSurrogate` vs `PhaseTypeSurrogate` differences handled via method dispatch on specialized subroutines.

**Success Criteria:**
- [ ] Zero `if use_phasetype` or `if surrogate isa PhaseTypeSurrogate` in main MCEM loop
- [ ] Single `MCEMInfrastructure` struct used throughout
- [ ] All surrogate-specific behavior via type dispatch
- [ ] Code reduction: ~1247 → ~650-750 lines in fit_mcem.jl
- [ ] All existing tests pass (2100+ tests)

---

## 2. Design Decisions (APPROVED)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Infrastructure struct location | `src/mcem/infrastructure.jl` | MCEM-specific, not general types |
| Dispatch functions location | `src/inference/fit_mcem.jl` (Option A) | Keep main loop with its helpers |
| SIR handling | Surrogate-agnostic, no dispatch | Operates only on weights |
| DrawSamplePaths! | Refactor to use infrastructure | Simplifies kwargs significantly |
| Backward compatibility | Not needed | Internal API only |
| Path sampling | Option A: sample expanded → collapse → marginal | Correct for IS weights |

---

## 3. Key Insight: What Actually Differs

**Sampling algorithm is SAME** (FFBS → ECCTMC) — phase-type is still Markovian.

**Dispatch needed only for:**

| Operation | Markov | PhaseType |
|-----------|--------|-----------|
| `build_mcem_infrastructure` | Original state space | Expanded state space |
| `compute_normalizing_constant` | Markov marginal loglik | PhaseType forward algorithm |
| `compute_surrogate_loglik` | CTMC path density | Marginal via forward algorithm |
| `collapse_path` | Identity (no-op) | Expanded → observed states |

---

## 4. Files to Create

### 4.1 `src/mcem/infrastructure.jl` (NEW)

```julia
# Contains:
# - MCEMInfrastructure{S<:AbstractSurrogate} struct
# - build_mcem_infrastructure(model, ::MarkovSurrogate)
# - build_mcem_infrastructure(model, ::PhaseTypeSurrogate)
```

### 4.2 `src/mcem/path_likelihood.jl` (NEW)

```julia
# Contains:
# - compute_normalizing_constant(model, infra::MCEMInfrastructure{MarkovSurrogate})
# - compute_normalizing_constant(model, infra::MCEMInfrastructure{PhaseTypeSurrogate})
# - compute_surrogate_loglik(path, model, infra::MCEMInfrastructure{MarkovSurrogate})
# - compute_surrogate_loglik(path, model, infra::MCEMInfrastructure{PhaseTypeSurrogate})
# - collapse_path(expanded_path, infra) — PhaseType only
```

---

## 5. Files to Modify

### 5.1 `src/MultistateModels.jl`
- Add `include("mcem/infrastructure.jl")`
- Add `include("mcem/path_likelihood.jl")`

### 5.2 `src/inference/fit_mcem.jl`
- Remove 16+ `if use_phasetype` branches
- Remove duplicate infrastructure variables
- Use `MCEMInfrastructure` throughout
- Single-flow main loop

### 5.3 `src/inference/sampling_markov.jl`
- Refactor `DrawSamplePaths!` to accept `infra::MCEMInfrastructure`
- Remove 12+ phase-type kwargs
- Keep sampling logic unchanged (FFBS → ECCTMC)

---

## 6. Implementation Phases & Handoff Points

### PHASE 1: Infrastructure Foundation (Agent 1)
**Scope**: Create new files, define types, no behavior changes yet

- [x] 1.1 Create `src/mcem/` directory
- [x] 1.2 Create `src/mcem/infrastructure.jl` with `MCEMInfrastructure` struct
- [x] 1.3 Implement `build_mcem_infrastructure(model, ::MarkovSurrogate)`
- [x] 1.4 Implement `build_mcem_infrastructure(model, ::PhaseTypeSurrogate)`
- [x] 1.5 Create `src/mcem/path_likelihood.jl` with dispatch stubs
- [x] 1.6 Update `src/MultistateModels.jl` to include new files
- [x] 1.7 **CHECKPOINT**: Verify package loads without errors

**Handoff criteria**: Package compiles, new types exist, no tests broken (new code unused)
**Status**: ✅ COMPLETE (2026-01-21)
- Package loads successfully
- Both Markov and PhaseType infrastructure builders tested
- Test suite: 2151 passed, 4+1 pre-existing failures (unchanged from before)

---

### PHASE 2: Likelihood Dispatch (Agent 1 or 2)
**Scope**: Implement dispatch methods, extract existing code

- [x] 2.1 Implement `compute_normalizing_constant` for MarkovSurrogate (extract from fit_mcem.jl L458)
- [x] 2.2 Implement `compute_normalizing_constant` for PhaseTypeSurrogate (extract from fit_mcem.jl L453-457)
- [x] 2.3 Implement `compute_surrogate_path_loglik` for MarkovSurrogate (extract from sampling_markov.jl)
- [x] 2.4 Implement `compute_surrogate_path_loglik` for PhaseTypeSurrogate (extract marginal computation)
- [x] 2.5 Implement `collapse_path` for PhaseTypeSurrogate
- [x] 2.6 **CHECKPOINT**: Write unit tests for dispatch methods
- [x] 2.7 Verify dispatch methods produce same values as current inline code

**Handoff criteria**: All dispatch methods tested, produce identical results to current code
**Status**: ✅ COMPLETE (2026-01-21)
- `compute_normalizing_constant` verified for Markov and PhaseType (exact match)
- `compute_surrogate_path_loglik` verified for Markov (exact match)
- PhaseType `compute_surrogate_path_loglik` implemented; requires expanded path from MCEM to test (will validate in Phase 4)

---

### PHASE 3: DrawSamplePaths! Refactor (Agent 2 or 3)
**Scope**: Simplify path sampling interface

- [x] 3.1 Create new signature: `DrawSamplePaths!(model, infra::MCEMInfrastructure, containers; kwargs)`
- [x] 3.2 Replace 12+ phase-type kwargs with `infra` fields
- [x] 3.3 Keep internal sampling logic (FFBS, ECCTMC) unchanged
- [x] 3.4 Use `compute_surrogate_path_loglik(path, model, infra)` for weight denominator
- [x] 3.5 **CHECKPOINT**: Verify identical paths sampled (set RNG seed, compare)

**Handoff criteria**: DrawSamplePaths! uses infrastructure, produces identical samples
**Status**: ✅ COMPLETE (2026-01-21)
- New `DrawSamplePaths!(model, infra, containers; kwargs...)` signature implemented
- Replaced 12+ phase-type kwargs with reads from `infra` fields
- Uses `compute_surrogate_path_loglik` dispatch for weight denominator
- Include order updated: sampling_phasetype.jl → infrastructure.jl → sampling_markov.jl
- Tested with both Markov and PhaseType surrogates (identical results)
- All 2151 tests pass (4+1 pre-existing failures unchanged)

---

### PHASE 4: Main Loop Rewrite (Agent 3 or 4)
**Scope**: Rewrite _fit_mcem to single flow

- [ ] 4.1 Remove all `if use_phasetype` branches
- [ ] 4.2 Remove duplicate infrastructure variables (tpm_book_ph, fbmats_ph, etc.)
- [ ] 4.3 Replace infrastructure building with `build_mcem_infrastructure(model, surrogate)`
- [ ] 4.4 Replace normalizing constant with `compute_normalizing_constant(model, infra)`
- [ ] 4.5 Replace DrawSamplePaths! calls with new signature
- [ ] 4.6 Keep M-step, variance computation, SIR logic unchanged
- [ ] 4.7 **CHECKPOINT**: Run unit tests, verify pass

**Handoff criteria**: All unit tests pass, fit_mcem.jl < 800 lines

---

### PHASE 5: Integration Testing (Agent 4 or 5)
**Scope**: Full validation

- [ ] 5.1 Run full test suite: `julia --project -e 'using Pkg; Pkg.test()'`
- [ ] 5.2 Run MCEM longtests specifically
- [ ] 5.3 Verify Markov surrogate MCEM produces same results (seed RNG)
- [ ] 5.4 Verify PhaseType surrogate MCEM produces same results
- [ ] 5.5 Fix any regressions
- [ ] 5.6 Update CHANGELOG.md

**Handoff criteria**: All 2100+ tests pass, no regressions

---

## 7. Detailed Action Items

### Phase 1 Action Items

#### 1.1 Create directory
```bash
mkdir -p src/mcem
```

#### 1.2 Create infrastructure.jl
```julia
# MCEMInfrastructure struct with fields:
# - tpm_book::Vector{Matrix{Float64}}
# - hazmat_book::Vector{Matrix{Float64}}
# - fbmats::Union{Nothing, Vector{FBMats}}
# - emat::Matrix{Float64}
# - books::Tuple{Vector, Matrix}  # (tpm_times, tpm_map)
# - data::DataFrame  # May be expanded for PhaseType
# - subjectindices::Vector{Vector{Int}}
# - surrogate::S where S<:AbstractSurrogate
# - schur_cache::Union{Nothing, Vector{CachedSchurDecomposition}}
# - original_row_map::Union{Nothing, Vector{Int}}
# - absorbingstates::Vector{Int}
```

#### 1.3 Markov infrastructure builder
Extract and encapsulate lines 378-412 of fit_mcem.jl:
- `build_tpm_mapping`
- `build_hazmat_book`
- `build_tpm_book`
- Kolmogorov equation solving
- `build_fbmats`

#### 1.4 PhaseType infrastructure builder
Extract and encapsulate lines 413-450 of fit_mcem.jl:
- Data expansion check and execution
- `build_phasetype_tpm_book`
- Schur cache creation
- `build_fbmats_phasetype_with_indices`
- `build_phasetype_emat_expanded`

---

## 8. Risk Mitigation

### Agent Context Confusion
- **Handoff after each phase** with clear criteria
- **Checkpoint validation** before proceeding
- **This document** serves as single source of truth

### Regression Risk
- **No behavior changes** until Phase 4
- **Extract, don't rewrite** existing logic
- **Seed RNG** for reproducibility testing

### Performance Risk
- Infrastructure built once (not per-iteration)
- Dispatch is zero-cost in Julia (compile-time)
- No additional allocations vs current code

---

## 9. Testing Strategy (POST-IMPLEMENTATION)

⚠️ **NOTE**: Once implementation is complete, testing infrastructure needs overhaul:

### Current Test Coverage Gaps
- [ ] No direct unit tests for `_fit_mcem` (only integration via `fit()`)
- [ ] No tests verifying Markov vs PhaseType produce equivalent results
- [ ] No tests for `DrawSamplePaths!` in isolation

### Recommended New Tests
1. **Unit tests for MCEMInfrastructure**
   - `test_mcem_infrastructure.jl`: verify builds correctly for each surrogate
   
2. **Unit tests for dispatch methods**
   - `test_path_likelihood.jl`: verify `compute_surrogate_loglik`, `compute_normalizing_constant`
   
3. **Integration tests for surrogate equivalence**
   - Same model, same seed → Markov and PhaseType surrogates should give similar MLE
   
4. **Regression tests**
   - Capture current MCEM outputs, verify refactored code matches

---

## 10. Reference: Current Code Locations

| Functionality | Current Location | Lines |
|---------------|------------------|-------|
| Markov infrastructure build | fit_mcem.jl | 378-412 |
| PhaseType infrastructure build | fit_mcem.jl | 413-450 |
| Normalizing constant (Markov) | fit_mcem.jl | 458 |
| Normalizing constant (PhaseType) | fit_mcem.jl | 453-457 |
| DrawSamplePaths! (outer) | sampling_markov.jl | 35-92 |
| DrawSamplePaths! (inner) | sampling_markov.jl | 94-340 |
| Markov path sampling | sampling_markov.jl | 340-600 |
| PhaseType path sampling | sampling_markov.jl | 145-280 |
| Surrogate loglik (PhaseType marginal) | sampling_markov.jl | 180-220 |

---

## 11. Changelog for This Refactor

When complete, add to CHANGELOG.md:

```markdown
### Changed
- **MCEM Refactor**: Complete rewrite of `_fit_mcem` for surrogate-agnostic design
  - New `MCEMInfrastructure` struct encapsulates all MCEM precomputation
  - New dispatch methods: `build_mcem_infrastructure`, `compute_normalizing_constant`, 
    `compute_surrogate_loglik` for MarkovSurrogate and PhaseTypeSurrogate
  - Removed 16+ `if use_phasetype` branches from main loop
  - Reduced fit_mcem.jl from 1247 to ~700 lines
  - DrawSamplePaths! simplified to use infrastructure struct
  - No user-facing API changes
```

---

## 12. Agent Instructions

### Starting Work
1. Read this entire document
2. Read the codebase-knowledge skill
3. Verify you understand the current code structure
4. Work on ONE phase at a time
5. Run checkpoints before moving to next phase

### Handoff Protocol
1. Update this document with completed items (check boxes)
2. Note any deviations from plan
3. Document any issues encountered
4. Ensure package compiles and tests pass
5. Create handoff summary in this document

### If Stuck
1. Do NOT guess or improvise
2. Ask user for clarification
3. If context is confused, request handoff to new agent
4. Prefer smaller, validated changes over large rewrites

---

## HANDOFF LOG

### Agent 1 (2026-01-21)
- Status: Phases 1-2 COMPLETE
- Completed: 
  - Design proposal, user approval, plan document
  - Created `src/mcem/infrastructure.jl` with `MCEMInfrastructure{S}` struct
  - Created `src/mcem/path_likelihood.jl` with dispatch methods
  - Implemented and verified `build_mcem_infrastructure` for both Markov and PhaseType
  - Implemented and verified `compute_normalizing_constant` for both types
  - Implemented `compute_surrogate_path_loglik` for both types
  - Implemented `collapse_expanded_path` for PhaseType
- Verified:
  - Package loads successfully
  - All 2151 unit tests pass (4+1 pre-existing failures unchanged)
  - `compute_normalizing_constant` produces exact match for both surrogate types
  - `compute_surrogate_path_loglik` produces exact match for Markov (PhaseType needs MCEM context)
- Next: Phase 3 - Refactor DrawSamplePaths! to use MCEMInfrastructure
- Files created:
  - `src/mcem/infrastructure.jl` (~270 lines)
  - `src/mcem/path_likelihood.jl` (~170 lines)
- Files modified:
  - `src/MultistateModels.jl` (added includes)
- Notes: 
  - `tpm_book` is `Vector{Vector{Matrix}}` not `Vector{Matrix}` (outer for covariate combos, inner for time intervals)
  - PhaseType infrastructure builds BOTH expanded and Markov books (Markov needed for sampling)
