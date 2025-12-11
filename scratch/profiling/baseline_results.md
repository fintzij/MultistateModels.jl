# Phase 0 Baseline Results

**Date:** 2025-12-10  
**Julia Version:** 1.11.x  
**System:** macOS

---

## Summary

| Category | Metric | Baseline | Notes |
|----------|--------|----------|-------|
| **Simulation** | Single path (2-state Markov) | 2.1 ms | 45k allocs, 1.3 MiB |
| | Single path (2-state Semi-Markov) | 2.7 ms | 66k allocs, 1.8 MiB |
| | 100 paths (n=50, Markov) | 71 ms | 2.1M allocs, 44 MiB |
| | 100 paths (n=50, Semi-Markov) | 99 ms | 3.0M allocs, 66 MiB |
| **Likelihood** | Markov panel (n=100) | 79 μs | 2.4k allocs, 59 KiB |
| | Markov panel (n=500) | 381 μs | 12k allocs, 291 KiB |
| | Semi-Markov (n=50) | 314 μs | 5.3k allocs, 204 KiB |
| | ExactData (n=50) | 314 μs | 5.4k allocs, 210 KiB |
| **MCEM** | Single iteration (n=20) | 0.20 s | |
| | Single iteration (n=50) | 0.46 s | |
| | Single iteration (n=100) | 1.04 s | |
| | 3 iterations (n=30) | 12.9 s | ~4.3 s/iter |

---

## Detailed Results

### 1. Simulation

#### Single Path Simulation
| Model Type | Time (μs) | Allocs | Memory (KiB) |
|------------|-----------|--------|--------------|
| 2-state Markov | 2,090 | 45,141 | 1,294 |
| 3-state Markov | 3,462 | 80,769 | 2,077 |
| 2-state Semi-Markov (Weibull) | 2,682 | 66,438 | 1,779 |
| 3-state Semi-Markov + Covariates | 3,448 | 73,964 | 2,315 |

**Observation:** Semi-Markov adds ~30% overhead vs Markov for 2-state model.

#### Multi-Path Simulation (nsim=100, n=50)
| Model Type | Time (ms) | Allocs | Memory (MiB) |
|------------|-----------|--------|--------------|
| 2-state Markov | 70.7 | 2,091,227 | 44.0 |
| 3-state Markov | 142.1 | 4,186,233 | 88.3 |
| 2-state Semi-Markov | 99.2 | 3,023,161 | 65.6 |

**Observation:** Memory scales roughly linearly with paths (~0.4-0.9 MiB per path).

#### Scaling with nsim (2-state Semi-Markov, n=50)
| nsim | Time (ms) | Allocs | Memory (MiB) | Allocs/path |
|------|-----------|--------|--------------|-------------|
| 10 | 10.0 | 302,647 | 6.7 | 6,053 |
| 50 | 49.9 | 1,507,080 | 32.8 | 6,028 |
| 100 | 99.5 | 3,021,293 | 65.5 | 6,043 |
| 200 | 199.9 | 6,050,121 | 131.0 | 6,050 |

**Observation:** Perfect linear scaling with nsim. ~6k allocs per path is constant.

---

### 2. Likelihood

#### Markov Panel Likelihood
| Configuration | Time (μs) | Allocs | Memory (KiB) | Allocs/subj |
|---------------|-----------|--------|--------------|-------------|
| 2-state (n=100) | 79 | 2,358 | 58.7 | 24 |
| 3-state (n=100) | 108 | 2,904 | 70.4 | 29 |
| 2-state (n=500) | 381 | 12,048 | 291.4 | 24 |

**Observation:** ~24-29 allocs per subject. Time scales sub-linearly with n.

#### Semi-Markov Likelihood
| Configuration | Time (μs) | Allocs | Memory (KiB) |
|---------------|-----------|--------|--------------|
| 2-state (n=50) | 314 | 5,329 | 203.6 |
| 3-state + Cov (n=50) | 717 | 11,640 | 421.4 |

**Observation:** Semi-Markov likelihood is ~4x slower than Markov for similar n.

#### Exact Data Likelihood
| Configuration | Time (μs) | Allocs | Memory (KiB) |
|---------------|-----------|--------|--------------|
| ExactData (n=50) | 314 | 5,444 | 209.7 |
| ExactDataAD (single path) | 29 | 280 | 12.1 |

**Observation:** ExactDataAD is 10x faster for single path (used in variance estimation).

---

### 3. MCEM

#### Full Fit (3 iterations, n=30)
- Total time: 12.9 s
- Time per iteration: 4.3 s
- Memory usage: Not measured (dominated by path storage)

#### Profiling Breakdown (MCEM hot path)
From profile data (relative counts):

| Component | Counts | % of MCEM |
|-----------|--------|-----------|
| Ipopt optimization | 367 | 79% |
| ├─ Objective (loglik) | 105 | 23% |
| ├─ Gradient (ForwardDiff) | 117 | 25% |
| └─ Hessian (ForwardDiff) | 114 | 25% |
| loglik_semi_markov | ~220 | 47% |
| _compute_path_loglik_fused | ~78 | 17% |

**Key finding:** M-step (Ipopt optimization) dominates at 79% of MCEM time.
The likelihood evaluation is called multiple times per Ipopt iteration.

#### Scaling with nsubj
| nsubj | Time (s) | Time/subj (ms) |
|-------|----------|----------------|
| 20 | 0.20 | 10.0 |
| 50 | 0.46 | 9.2 |
| 100 | 1.04 | 10.4 |

**Observation:** Linear scaling with nsubj (~10ms per subject per iteration).

---

## Hot Paths Identified

### Simulation Hot Paths
1. **IntervalNonlinearProblem construction** - Called for each jump time root-finding
2. **BracketingNonlinearSolve** - Root-finding for jump times
3. **Path construction** - Allocating vectors for times/states

### Likelihood Hot Paths
1. **loglik_markov** - Main panel likelihood
2. **loglik_semi_markov** - Main semi-Markov likelihood (4x slower)
3. **_compute_path_loglik_fused** - Per-path likelihood in MCEM

### MCEM Hot Paths
1. **M-step optimization** - 79% of MCEM time (Ipopt + derivatives)
2. **Gradient computation** - 25% (ForwardDiff through loglik_semi_markov)
3. **Hessian computation** - 25% (ForwardDiff)

---

## Optimization Priorities

Based on profiling results:

### High Priority (Impact on MCEM)
1. **Reduce loglik_semi_markov allocations** - Called many times per iteration
2. **Optimize _compute_path_loglik_fused** - 17% of MCEM time
3. **Consider Mooncake for reverse-mode AD** - Could speed up gradient/Hessian

### Medium Priority (Impact on simulation)
4. **Reduce per-path allocations** - 6k allocs per path is high
5. **Pre-allocate path storage** - Avoid vcat/push! patterns
6. **Cache IntervalNonlinearProblem** - Avoid reconstruction overhead

### Lower Priority
7. **Hazard evaluation** - Need proper benchmark setup
8. **Memory layout** - After other optimizations

---

## Baseline Files

Results saved to:
- `scratch/profiling/simulation_baseline.json`
- `scratch/profiling/likelihood_baseline.json`
- `scratch/profiling/mcem_baseline.json`

These files serve as the reference for measuring improvement after optimizations.
