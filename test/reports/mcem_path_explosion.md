# MCEM Path Explosion Diagnostics

_Last updated: 2025-12-07_

## Overview

Issue: In certain MCEM longtests, the adaptive importance sampler requires more than 600 sample paths per subject to reach a modest ESS target (e.g., ESS>30). This is unexpected for the current panel designs and parameter values and suggests a bug in the MCEM implementation and/or test setup.

## Affected Tests

- `test/longtest_mcem.jl`
  - `MCEM Weibull - No Covariates`: currently passes under a hard ESS-based path cap.
  - `MCEM Weibull - With Covariate`: passes under the current cap, but historically showed path inflation.
  - `MCEM Gompertz - No Covariates`: **fails** with path explosion when run with a diagnostic cap.

## Reproduction

Command:

```julia
cd "MultistateModels.jl"
julia --project=. test/longtest_mcem.jl
```

Diagnostics in `test/longtest_mcem.jl`:

- Added `const MAX_PATHS_PER_SUBJECT = 200`.
- Added wrapper `fit_mcem_with_path_cap(model; test_label, path_cap=MAX_PATHS_PER_SUBJECT, kwargs...)` which:
  - Calls `fit` on semi-Markov models with `return_convergence_records=true`.
  - Examines `fitted.ConvergenceRecords.ess_trace`.
  - Throws an error if `maximum(ess_trace) > path_cap`.

With this wrapper, the run produces:

- Warnings from `src/sampling.jl`:
  - `More than 600 sample paths are required to obtain ess>30 for individual i.` for many subject IDs.
- Error from the wrapper for `MCEM Gompertz - No Covariates`:
  - `MCEM path explosion in test 'MCEM Gompertz - No Covariates': implied paths per subject exceeded 200 (max ESS≈610)`.

## Model and Design Details

### Panel Data Generator

For the illness-death longtests, panel data are generated via
`generate_panel_data_illness_death` in `test/longtest_mcem.jl`:

- Default observation times (currently):
  - `obs_times = collect(0.0:2.0:MAX_TIME)` with `MAX_TIME = 15.0`.
  - So each subject has panel intervals [0,2), [2,4), …, [14,15].
- Model structure: illness-death with hazards 1→2, 2→3, 1→3.
- Data simulated from the true hazards with `simulate(...; paths=false, data=true, autotmax=false)`.

### MCEM Configuration (semi-Markov)

From `fit(::MultistateSemiMarkovModel, ...)` in `src/modelfitting.jl`:

- `ess_target_initial = 30` (in tests).
- `max_ess = 500`.
- `max_sampling_effort = 20`.
- `npaths_additional = 10`.
- Proposal: `proposal=:markov` (Markov surrogate).
- `return_convergence_records=true` so `ess_trace` is recorded.

### Sampling Logic

From `src/sampling.jl` (adaptive sampler snippet):

- For each subject `i`, paths are incrementally sampled until:
  - `ess_cur[i] >= ess_target` **or**
  - `length(samplepaths[i]) > n_path_max`, which triggers:
    - `@warn "More than $n_path_max sample paths are required to obtain ess>$ess_target for individual $i."`

The warnings observed in the failing Gompertz test come from this code path.

## Observations So Far

1. **Exponential panel tests** (now correctly labeled as Markov solver tests) behave as expected and do not use MCEM.
2. **Weibull MCEM tests** currently pass both parameter recovery and distributional checks with the path cap in place.
3. **Gompertz MCEM (no covariate)** hits path explosion:
   - Many subjects require >600 paths to reach ESS>30.
   - `ess_trace` shows maximum ESS ≈ 610, triggering the diagnostic cap.
4. Parameter values (shape/scale) and panel grid are moderate; this behavior is not expected from model difficulty alone.

## Hypotheses

- **State / cache reuse issues**:
  - Arrays `samplepaths`, `_logImportanceWeights`, `ImportanceWeights`, and related buffers are preallocated per subject in `fit`. If they are not correctly reset between MCEM iterations or proposal updates, ESS may be computed over inconsistent path sets, forcing unnecessary path augmentation.
- **Numerical instability in importance weights**:
  - Extremely imbalanced log-weights could concentrate mass on a tiny subset of paths, driving ESS to remain low regardless of path count.
- **Mismatch between ESS and actual path counts**:
  - For Markov proposals, ESS is expected to reflect the effective number of paths. If ESS is computed incorrectly (for example, with mis-normalized weights after augmentation), MCEM may over-sample.

## Root Cause Identified

**Bug Location**: `src/sampling.jl` `DrawSamplePaths!` function + MCEM iteration loop in `src/modelfitting.jl`

**The Issue**:

1. Path arrays (`samplepaths`, `ImportanceWeights`, `_logImportanceWeights`, etc.) are **never cleared between MCEM iterations**
2. Each call to `DrawSamplePaths!` **appends** new paths to existing arrays (line 99-108 in sampling.jl)
3. After each M-step, `ComputeImportanceWeightsESS!` recomputes weights over **ALL accumulated paths** (line ~1339 in modelfitting.jl)
4. When `ess_target` increases, `DrawSamplePaths!` is called again and appends **more paths** to the already-large arrays

**Why ESS Explodes**:

- For subjects where importance weights are relatively uniform (common with Gompertz in the test scenario):
  - ESS ≈ number of paths
  - As paths accumulate across iterations (50 + 50 + 50 + ...), ESS grows linearly
  - Eventually hits 600+ paths across multiple MCEM iterations

**Why min/median ESS = 30 but max ≈ 610**:

- Most subjects: weights become concentrated quickly, ESS saturates near target even as paths accumulate
- Some subjects (Gompertz-specific): nearly uniform weights → ESS = path count → explosion

**Evidence**:

- ESS diagnostics showed: `min=30, median=30, max≈610` across 3 iterations
- Warnings: "More than 600 sample paths required..." for many subjects
- Path accumulation is iteration-cumulative, not per-iteration

## Proposed Fix

**Option 1: Clear paths at start of each MCEM iteration (safest)**

In `src/modelfitting.jl`, before calling `DrawSamplePaths!` in the iteration loop, reset:

```julia
# Clear accumulated paths before resampling for new ess_target
for i in eachindex(model.subjectindices)
    empty!(samplepaths[i])
    empty!(loglik_surrog[i])
    empty!(loglik_target_cur[i])
    empty!(loglik_target_prop[i])
    empty!(_logImportanceWeights[i])
    empty!(ImportanceWeights[i])
    ess_cur[i] = 0.0
end
```

**Option 2: Track iteration and prevent re-entry for satisfied subjects**

Add logic to skip subjects that have already met `ess_target` in the current iteration.

**Option 3: Cap per-subject path count explicitly**

Add a hard cap in `DrawSamplePaths!` based on `max_sampling_effort * ess_target` **per iteration**, not cumulative.
