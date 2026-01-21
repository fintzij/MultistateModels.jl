# fit_mcem.jl Complete Refactor Handoff

**Date**: 2026-01-21
**Branch**: penalized_splines
**Status**: PREVIOUS AGENT FAILED - needs complete rewrite

---

## The Problem

The current fit_mcem.jl is a disaster. The previous agent attempted a surrogate refactor but made it worse by:

1. Keeping separate codepaths for MarkovSurrogate vs PhaseTypeSurrogate
2. Creating duplicate infrastructure variables (tpm_book vs tpm_book_ph, fbmats vs fbmats_ph, etc.)
3. Scattering `if use_phasetype` conditionals throughout 1200+ lines
4. Not leveraging Julia's type dispatch at all

---

## The Core Insight

**There is ALWAYS a surrogate.** Either MarkovSurrogate or PhaseTypeSurrogate.

**Both surrogate types need the SAME infrastructure:**
- TPM books (transition probability matrices)
- Hazmat books (hazard rate matrices)  
- Forward-backward matrices (fbmats)
- Emission matrices (emat)
- Schur caches (for efficient matrix exponentials)

**The MCEM algorithm is IDENTICAL regardless of surrogate type:**
1. Initialize infrastructure from surrogate
2. Draw sample paths
3. Compute importance weights
4. M-step optimization
5. Check convergence
6. Repeat

The ONLY differences should be in specialized subroutines that dispatch on surrogate type.

---

## Design Principles for Refactor

### 1. Single Flow, Type Dispatch

The main `_fit_mcem` function should have ONE codepath. Surrogate-specific behavior comes from methods that dispatch on `::MarkovSurrogate` vs `::PhaseTypeSurrogate`.

### 2. Infrastructure Builder Functions

Create a unified infrastructure type and builder:

```
struct MCEMInfrastructure
    tpm_book
    hazmat_book
    fbmats
    emat
    schur_cache
    books           # tpm_mapping result
    data            # possibly expanded data for phase-type
    subjectindices  # possibly recomputed for expanded data
end

build_mcem_infrastructure(model, surrogate::MarkovSurrogate) -> MCEMInfrastructure
build_mcem_infrastructure(model, surrogate::PhaseTypeSurrogate) -> MCEMInfrastructure
```

### 3. Dispatch-Based Subroutines

Each major operation should dispatch on surrogate type:

```
draw_sample_paths!(paths, model, surrogate::MarkovSurrogate, infra, ...)
draw_sample_paths!(paths, model, surrogate::PhaseTypeSurrogate, infra, ...)

compute_path_loglik(path, model, surrogate::MarkovSurrogate, infra)
compute_path_loglik(path, model, surrogate::PhaseTypeSurrogate, infra)

compute_normalizing_constant(model, surrogate::MarkovSurrogate, infra)
compute_normalizing_constant(model, surrogate::PhaseTypeSurrogate, infra)
```

### 4. No Conditional Branching in Main Loop

The main MCEM loop should look like:

```
infra = build_mcem_infrastructure(model, surrogate)
norm_const = compute_normalizing_constant(model, surrogate, infra)

while !converged
    draw_sample_paths!(paths, model, surrogate, infra, ...)
    update_importance_weights!(weights, paths, model, surrogate, infra)
    params_new = m_step(model, paths, weights, ...)
    converged = check_convergence(...)
end
```

NO `if surrogate isa PhaseTypeSurrogate` in the main loop.

---

## Specific Problems in Current Code (fit_mcem.jl)

### Lines 405-420: Duplicate Infrastructure
```julia
phasetype_surrogate = use_phasetype ? surrogate : nothing
# ... then later creates SEPARATE tpm_book_ph, hazmat_book_ph, fbmats_ph
```
This should be ONE set of infrastructure, built by dispatch.

### Lines 430-450: Duplicate TPM Construction
Builds `hazmat_book_surrogate` and `tpm_book_surrogate` for Markov, then ALSO builds phase-type versions later. Should be unified.

### Lines 460-490: Data Expansion Scattered
Phase-type data expansion logic is inline instead of encapsulated in a builder function.

### Lines 490-510: Conditional Normalizing Constant
```julia
if use_phasetype
    NormConstantProposal = compute_phasetype_marginal_loglik(...)
else
    NormConstantProposal = compute_markov_marginal_loglik(...)
end
```
This should be ONE call that dispatches: `compute_normalizing_constant(model, surrogate, infra)`

### DrawSamplePaths! (Lines 520+)
Passes BOTH markov and phasetype infrastructure as separate kwargs. Should pass unified `infra`.

---

## Step-by-Step Refactor Plan

### Phase 1: Design Infrastructure Type

Work with user to define MCEMInfrastructure struct. Decide what fields are needed for both surrogate types.

### Phase 2: Write Builder Functions

Implement `build_mcem_infrastructure` methods for each surrogate type. These encapsulate all the TPM book building, data expansion, emission matrix construction, etc.

### Phase 3: Extract Dispatch Subroutines

Identify every place with `if use_phasetype` or surrogate-type branching. Extract into methods that dispatch on surrogate type.

### Phase 4: Rewrite Main Loop

With infrastructure and subroutines in place, rewrite the main MCEM loop to be surrogate-agnostic.

### Phase 5: Validate

Run full test suite. Ensure 2151 tests still pass.

---

## Files to Understand Before Starting

1. **src/inference/fit_mcem.jl** - the target (1245 lines of mess)
2. **src/inference/sampling_markov.jl** - Markov path sampling
3. **src/inference/sampling_phasetype.jl** - Phase-type FFBS sampling
4. **src/surrogate/markov.jl** - MarkovSurrogate type and methods
5. **src/phasetype/types.jl** - PhaseTypeSurrogate type

---

## Questions for User Before Starting

1. Should MCEMInfrastructure be a new type in src/types/ or kept local to fit_mcem.jl?
2. Should the dispatch subroutines live in fit_mcem.jl or in sampling_markov.jl/sampling_phasetype.jl?
3. Are there performance constraints we need to preserve (preallocated buffers, etc.)?
4. Should we also refactor DrawSamplePaths! or leave that for a separate pass?

---

## Success Criteria

1. NO `if use_phasetype` or `if surrogate isa PhaseTypeSurrogate` in main MCEM loop
2. Single MCEMInfrastructure type used throughout
3. All surrogate-specific behavior via type dispatch
4. Code is < 800 lines (down from 1245)
5. All existing tests pass
6. Easier to add new surrogate types in future

---

## Warning to Next Agent

DO NOT just rename variables or add wrapper functions around the existing mess. This needs a ground-up rewrite of the function structure. Start with pseudocode, get user approval, then implement.

Read the codebase-knowledge skill FIRST. Work incrementally with user feedback. Do not write 500 lines of code without validation.
