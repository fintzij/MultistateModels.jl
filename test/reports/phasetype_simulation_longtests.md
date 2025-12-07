# Phase-Type Simulation Long Tests Report

**Generated:** 2025-12-06T15:42:19.689

## Overview

These tests validate that the phase-type model simulation produces
identical results to a manually-expanded Markov model with explicit
exponential hazards on the expanded state space.

### Model Structure
- **Observed states:** 1 → 2 → 3 (progressive, state 3 absorbing)
- **Expanded states:** Phase expansion for Coxian phase-type distributions

## Summary

| Test | Path Equivalence | Prevalence Diff | CumIncid Diff | Status |
|------|------------------|-----------------|---------------|--------|
| ptsim_2phase_nocov | 100.0% | 0.0 | 0.0 | ✅ PASS |
| ptsim_3phase_nocov | 100.0% | 0.0 | 0.0 | ✅ PASS |
| ptsim_2phase_allequal | 100.0% | 0.0 | 0.0 | ✅ PASS |

---

## ptsim_2phase_nocov

### Results
- **Path Equivalence Rate:** 100.0%
- **Max Prevalence Difference:** 0.0
- **Max Cumulative Incidence Difference:** 0.0
- **Status:** ✅ PASSED

### State Prevalence Curves

![](assets/phasetype_simulation/ptsim_2phase_nocov_prevalence.png)

### Cumulative Incidence Curves

![](assets/phasetype_simulation/ptsim_2phase_nocov_cumincid.png)

### Log-Likelihood Distributions

The log-likelihood proxy is computed as the negative total sojourn time
(i.e., time from start until reaching the absorbing state or censoring).

![](assets/phasetype_simulation/ptsim_2phase_nocov_loglik_hist.png)

### Log-Likelihood Scatter (Path-by-Path)

![](assets/phasetype_simulation/ptsim_2phase_nocov_loglik_scatter.png)

### Log-Likelihood Statistics

| Metric | PhaseType | Manual | Difference |
|--------|-----------|--------|------------|
| Mean | -1.4455 | -1.4455 | 0.0 |
| Std Dev | 1.0172 | 1.0172 | 0.0 |
| Min | -9.6097 | -9.6097 | 0.0 |
| Max | -0.0038 | -0.0038 | 0.0 |
| Correlation | | | 1.0 |

---

## ptsim_3phase_nocov

### Results
- **Path Equivalence Rate:** 100.0%
- **Max Prevalence Difference:** 0.0
- **Max Cumulative Incidence Difference:** 0.0
- **Status:** ✅ PASSED

### State Prevalence Curves

![](assets/phasetype_simulation/ptsim_3phase_nocov_prevalence.png)

### Cumulative Incidence Curves

![](assets/phasetype_simulation/ptsim_3phase_nocov_cumincid.png)

### Log-Likelihood Distributions

The log-likelihood proxy is computed as the negative total sojourn time
(i.e., time from start until reaching the absorbing state or censoring).

![](assets/phasetype_simulation/ptsim_3phase_nocov_loglik_hist.png)

### Log-Likelihood Scatter (Path-by-Path)

![](assets/phasetype_simulation/ptsim_3phase_nocov_loglik_scatter.png)

### Log-Likelihood Statistics

| Metric | PhaseType | Manual | Difference |
|--------|-----------|--------|------------|
| Mean | -1.5124 | -1.5124 | 0.0 |
| Std Dev | 1.054 | 1.054 | 0.0 |
| Min | -8.9205 | -8.9205 | 0.0 |
| Max | -0.0025 | -0.0025 | 0.0 |
| Correlation | | | 1.0 |

---

## ptsim_2phase_allequal

### Results
- **Path Equivalence Rate:** 100.0%
- **Max Prevalence Difference:** 0.0
- **Max Cumulative Incidence Difference:** 0.0
- **Status:** ✅ PASSED

### State Prevalence Curves

![](assets/phasetype_simulation/ptsim_2phase_allequal_prevalence.png)

### Cumulative Incidence Curves

![](assets/phasetype_simulation/ptsim_2phase_allequal_cumincid.png)

### Log-Likelihood Distributions

The log-likelihood proxy is computed as the negative total sojourn time
(i.e., time from start until reaching the absorbing state or censoring).

![](assets/phasetype_simulation/ptsim_2phase_allequal_loglik_hist.png)

### Log-Likelihood Scatter (Path-by-Path)

![](assets/phasetype_simulation/ptsim_2phase_allequal_loglik_scatter.png)

### Log-Likelihood Statistics

| Metric | PhaseType | Manual | Difference |
|--------|-----------|--------|------------|
| Mean | -1.5192 | -1.5192 | 0.0 |
| Std Dev | 1.0623 | 1.0623 | 0.0 |
| Min | -8.1377 | -8.1377 | 0.0 |
| Max | -0.0188 | -0.0188 | 0.0 |
| Correlation | | | 1.0 |

---

## Conclusion

✅ **All 3 tests passed.** The phase-type model simulation
produces results that are identical to the manually-expanded Markov model.
