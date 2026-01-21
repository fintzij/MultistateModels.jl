# MCEM PhaseType Proposal Parameter Recovery Bug — Handoff Document

**Date**: 2025-06-09  
**Status**: BUG FIX APPLIED, BUT MCEM TEST STILL FAILS WITH IDENTICAL VALUES  
**Urgency**: High — indicates deeper issue than originally identified

---

## 1. Executive Summary

### Problem Statement
The MCEM algorithm with PhaseType proposal systematically biases h23 (second transition) parameter estimates in a progressive illness-death model:

| Parameter | True Value | Markov Proposal | PhaseType Proposal | PhaseType Error |
|-----------|------------|-----------------|-------------------|-----------------|
| shape_23 | 1.1 | 1.2569 (14.3% error) | **1.5910** | **44.6%** |
| scale_23 | 0.12 | 0.0944 (21.3% error) | **0.0708** | **41.0%** |

### Fix Status
- **Two bugs identified and fixed** in `convert_expanded_path_to_censored_data`
- **Fix verified active** via diagnostic showing correct emission matrix and statefrom values
- **MCEM test still fails with IDENTICAL values** — returns exactly 1.5910/0.0708 before and after fix
- **This means another bug exists elsewhere in the PhaseType importance sampling pipeline**

---

## 2. Complete Simulation Setup

### Test File Location
`MultistateModelsTests/longtests/longtest_mcem.jl`, lines 520-650

### Model Structure
```
3-state progressive illness-death model:
  State 1 (healthy) → State 2 (diseased) → State 3 (dead/absorbing)

Transitions:
  h12: 1→2  Weibull(shape=1.3, scale=0.15)
  h23: 2→3  Weibull(shape=1.1, scale=0.12)
```

### Data Generation Process

1. **Panel Data Template** (`generate_panel_data_progressive`, lines 140-180):
   - `N_SUBJECTS = 1000`
   - `obs_times = 0:2:14` (observation times: 0, 2, 4, 6, 8, 10, 12, 14)
   - Initial template: all subjects start in state 1, all intervals have obstype=2 (panel)

2. **Simulation Call** (line 178):
   ```julia
   obstype_map = Dict(1 => 2, 2 => 1)  # Transition 1→2 is panel, 2→3 is exact
   sim_result = simulate(model; paths=false, data=true, nsim=1, autotmax=false,
                        obstype_by_transition=obstype_map)
   ```

3. **Observation Types in Generated Data**:
   - Transition 1→2: `obstype=2` (panel — state known only at observation times)
   - Transition 2→3: `obstype=1` (exact — death time known precisely)

### PhaseType Proposal Configuration
```julia
model_pt = multistatemodel(h12_pt, h23_pt; data=panel_data, surrogate=:markov)
fitted_pt = fit_mcem_with_path_cap(model_pt;
    proposal=PhaseTypeProposal(n_phases=3),  # 3 phases per macro-state
    maxiter=50,
    tol=1e-3,
    ess_target_initial=100,
    max_ess=500,
    ...)
```

### PhaseType Expansion Structure
```
Macro-states → Phases:
  State 1 → phases 1, 2, 3  (transient)
  State 2 → phases 4, 5, 6  (transient)
  State 3 → phase 7         (absorbing)

Coxian structure: Within each macro-state, phases form a ladder
  1 → 2 → 3 → (exit to state 2)
  4 → 5 → 6 → (exit to state 3)
```

### Test Constants
```julia
RNG_SEED = 0xABCD1234
N_SUBJECTS = 1000
MAX_TIME = 15.0
PARAM_TOL_REL = 0.35  # 35% relative tolerance
MCEM_TOL = 1e-3
MAX_ITER = 50
```

### Expected Transition Time Distribution

The Weibull hazard in MultistateModels.jl uses parameterization:
- Hazard: h(t) = κ · λ · (λt)^(κ-1)
- Survival: S(t) = exp(-(λt)^κ)
- Median sojourn time: t_median = (log 2)^(1/κ) / λ

**For h12 (1→2): shape=1.3, scale=0.15**
- Median sojourn: (0.693)^(1/1.3) / 0.15 ≈ **5.0 time units**
- 25th percentile: (0.288)^(0.77) / 0.15 ≈ 2.5 time units
- 75th percentile: (1.386)^(0.77) / 0.15 ≈ 8.5 time units
- Probability still in state 1 at t=14: exp(-(0.15 × 14)^1.3) ≈ 3%

**For h23 (2→3): shape=1.1, scale=0.12**
- Median sojourn: (0.693)^(1/1.1) / 0.12 ≈ **5.4 time units**
- 25th percentile: (0.288)^(0.91) / 0.12 ≈ 2.7 time units
- 75th percentile: (1.386)^(0.91) / 0.12 ≈ 9.3 time units

**Implication for Panel Data**:
- Observation times: 0, 2, 4, 6, 8, 10, 12, 14
- Typical subject: Transitions 1→2 around t≈5, then 2→3 around t≈10
- Many 1→2 transitions fall BETWEEN observation times (latent)
- The 2→3 transition is exact (obstype=1), so timing is known
- **Key**: PhaseType proposal must correctly handle subjects observed in state 1 at t=4, then state 2 at t=6 (transition happened somewhere in [4,6])

**Why h23 Bias Matters**:
The h23 transition is exact, but its START TIME depends on when 1→2 occurred.
If the PhaseType proposal systematically samples 1→2 at wrong times, this biases the 2→3 sojourn time estimate.

---

## 3. Bugs Identified and Fixed

### Location
`src/inference/sampling_phasetype.jl`, function `convert_expanded_path_to_censored_data`

### Bug 1: Emission Matrix for Transition Rows (lines 398-420)

**Before (WRONG)**:
```julia
# For transition rows, only the specific sampled destination phase was allowed
emat_row[dest_phase] = 1.0  # e.g., [0,0,0,1,0,0,0] if sampled phase 4
```

**After (CORRECT)**:
```julia
# For transition rows, allow ALL phases of the destination macro-state
dest_phases = state_phase_map[dest_state]
for dp in dest_phases
    emat_row[dp] = 1.0  # e.g., [0,0,0,1,1,1,0] for state 2 (phases 4,5,6)
end
```

**Rationale**: When computing the forward likelihood, we observe only the macro-state, not the specific phase. The emission matrix should allow any phase consistent with the observed macro-state.

### Bug 2: statefrom Re-initialization After Transitions (line 384, 476)

**Before (WRONG)**:
```julia
# For survival rows after a transition, re-initialized α to specific phase
statefrom_row[dest_phase] = 1  # e.g., [0,0,0,1,0,0,0] 
```

**After (CORRECT)**:
```julia
# For survival rows after a transition, set statefrom=0 to preserve marginalized distribution
statefrom_row .= 0  # Forward algorithm continues with marginalized distribution from previous row
```

**Rationale**: The forward algorithm maintains a marginal distribution over phases. After a transition row, we shouldn't re-initialize to a specific phase — we should continue with the marginalized distribution from the previous step.

---

## 4. Evidence Fix Is Active (But Doesn't Work)

### Diagnostic Test
```julia
# Test case: Subject observed at t=0,2 in state 1, transition at t=2.5, state 2 at t=4
# Path: State 1 → State 2 with specific sampled phase 4

# VERIFIED OUTPUT:
statefrom: [1, 0, 0, 0]        # ✓ Only first row initializes
row 2 emat: [0, 0, 0, 1, 1, 1, 0]  # ✓ Allows phases 4, 5, 6 (not just 4)
```

### BUT: MCEM Returns IDENTICAL Values
```
BEFORE fix: shape_23=1.5910, scale_23=0.0708
AFTER fix:  shape_23=1.5910, scale_23=0.0708  (IDENTICAL!)
```

**This is highly suspicious** — the fix changes the forward likelihood computation, which should change importance weights, which should change the MCEM estimates.

---

## 5. Key Code Paths for Investigation

### Importance Weight Computation
`src/inference/sampling_markov.jl`, lines 195-255:
```julia
# For PhaseType proposal:
#   target_ll = loglik(target_pars, collapsed_path, model.hazards, model)
#   surrogate_ll = compute_forward_loglik(censored_data, emat_path, ...)
#   log_weight = target_ll - surrogate_ll
```

### Forward Algorithm
`src/inference/sampling_phasetype.jl`, function `compute_forward_loglik`:
- Uses censored_data from `convert_expanded_path_to_censored_data`
- Computes marginal likelihood over phase sequences

### Censored Data Conversion (FIXED)
`src/inference/sampling_phasetype.jl`, function `convert_expanded_path_to_censored_data`:
- Converts sampled expanded path to censored data format
- Builds emission matrices for forward algorithm
- **This is where fixes were applied**

---

## 6. Hypotheses for Why Fix Doesn't Resolve Bias

### Hypothesis A: Fix Applied to Wrong Code Path
The fix is in `convert_expanded_path_to_censored_data`, but there may be another code path that computes the surrogate likelihood. Check:
- Are there multiple versions of this function?
- Is there caching that's using old values?
- Is the PhaseType proposal using a different function entirely?

### Hypothesis B: Forward Algorithm Called Differently
The emission matrix fix only matters if the forward algorithm actually uses those rows. Check:
- How does `compute_forward_loglik` iterate through rows?
- Does it handle transition rows differently?
- Is there early exit logic that skips the fixed rows?

### Hypothesis C: Bug in Other Direction
The fix addresses *computing* the surrogate likelihood given a sampled path. But what if the bug is in *sampling* paths from the proposal? Check:
- `sample_paths_phasetype` function
- How are paths sampled from the expanded state space?
- Are the sampled paths correct?

### Hypothesis D: Parameter Transformation Issue
The Weibull parameters might be on different scales between target and surrogate:
- Target: Weibull(shape=κ, scale=λ) 
- Surrogate: Phase-type with progression/exit rates
- Check: Are transformations consistent?

### Hypothesis E: IDENTICAL VALUES = Deterministic Bug
The fact that values are EXACTLY identical suggests the bug is deterministic, not stochastic. This could mean:
- RNG is seeded identically (it is: `RNG_SEED + 10`)
- A branch is always taken/not taken
- A variable is always overwritten to the same value

---

## 7. Files Modified

| File | Change |
|------|--------|
| `src/inference/sampling_phasetype.jl` | Fixed emission matrix and statefrom initialization in `convert_expanded_path_to_censored_data` |

---

## 8. Recommended Investigation Path

### Priority 1: Trace Actual Code Execution
1. Add print statements to `convert_expanded_path_to_censored_data` during MCEM run
2. Verify the function is actually called
3. Verify the fixed logic branches are executed
4. Print emission matrices and statefrom values for real data

### Priority 2: Compare Likelihood Values
1. For a single sampled path, manually compute:
   - Target log-likelihood (collapsed path under Weibull hazards)
   - Surrogate log-likelihood (forward algorithm on expanded path)
2. Compare values between Markov and PhaseType proposals
3. The log-weights should differ if proposals are different

### Priority 3: Check Path Sampling
1. Verify that PhaseType proposal actually samples from the expanded state space
2. Compare sampled paths between Markov and PhaseType proposals
3. The key question: Are the sampled paths themselves biased?

### Priority 4: Examine Pareto-k Values
The test prints Pareto-k diagnostics. High Pareto-k (>0.7) indicates poor proposal overlap:
```
Markov:    median=?, max=?
PhaseType: median=?, max=?
```
If PhaseType has much higher Pareto-k, the proposal is poor despite PSIS correction.

---

## 9. Test Commands

### Run the Failing Test
```bash
cd "/Users/fintzij/Library/CloudStorage/OneDrive-BristolMyersSquibb/Documents/Julia packages/MultistateModels.jl"
julia --project -e '
using MultistateModels, MultistateModelsTests, Test
include("MultistateModelsTests/longtests/longtest_mcem.jl")
' 2>&1 | tee mcem_test_output.txt
```

### Run Unit Tests (Should Pass)
```bash
julia --project -e 'using Pkg; Pkg.test()'
```

### Quick Diagnostic
```julia
using MultistateModels
using MultistateModels: convert_expanded_path_to_censored_data, state_phase_map

# Add diagnostic prints to the function, then:
# Run a simple MCEM fit and observe output
```

---

## 10. Context for Fresh Session

### Essential Background
- MultistateModels.jl is a Julia package for continuous-time multistate models
- MCEM = Monte Carlo Expectation-Maximization for panel data with unobserved transitions
- PhaseType proposal uses Coxian phase-type expansion to approximate semi-Markov hazards
- The forward algorithm computes marginal likelihood over latent phase sequences
- Importance sampling corrects for proposal/target mismatch

### Key Insight
The fix addresses emission matrix construction for the forward algorithm, but **MCEM returns IDENTICAL VALUES**. This means either:
1. The fixed code is not being executed during MCEM
2. The fixed code is executed but doesn't affect the result (unlikely given the math)
3. There's caching or memoization somewhere
4. There's another bug that dominates

### Pointers to Relevant Code
- `src/inference/sampling_phasetype.jl` — PhaseType-specific sampling and likelihood
- `src/inference/sampling_markov.jl` — Main MCEM infrastructure, importance weight computation
- `src/surrogate/surrogate.jl` — Surrogate initialization and parameter setting
- `MultistateModelsTests/longtests/longtest_mcem.jl` — Failing test

---

## 11. Questions for Next Session

1. **Is the fix actually executed during MCEM?** Add prints to verify.
2. **What are the Pareto-k values?** High values would indicate fundamental proposal/target mismatch.
3. **Are the sampled paths correct?** Compare paths between Markov and PhaseType proposals.
4. **Is there another surrogate likelihood computation path?** Search for alternatives.
5. **Why IDENTICAL values?** This is the key mystery — deterministic bugs have deterministic causes.

---

*End of handoff document*
