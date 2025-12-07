# Phase-Type Forward-Backward Sampling (FFBS)

This document describes the mathematical details of the Forward-Filtering Backward-Sampling (FFBS) algorithm for phase-type surrogates in MultistateModels.jl.

## Overview

When fitting multistate models with panel or interval-censored data, we use Monte Carlo EM (MCEM) which requires sampling latent paths consistent with the observed data. Phase-type surrogates provide a flexible proposal distribution by expanding each observed state into multiple latent "phases."

The FFBS algorithm samples paths from the posterior distribution:

$$P(\text{path} \mid \text{observations})$$

For phase-type models, we must sample paths in the **expanded phase space** and then collapse them to the observed state space.

## State Space Expansion

### Coxian Phase-Type Structure

For a Coxian phase-type distribution with $k$ phases, the sojourn time in an observed state follows a mixture of Erlang distributions. The expanded state space has:

- **Phases 1, 2, ..., k** for the observed state
- Transitions follow a Coxian structure: phase $i$ can either advance to phase $i+1$ or exit to the next observed state

The generator matrix $Q$ for a 2-phase Coxian has the form:

$$Q = \begin{pmatrix} -\lambda_1 & p\lambda_1 & (1-p)\lambda_1 \\ 0 & -\lambda_2 & \lambda_2 \\ 0 & 0 & 0 \end{pmatrix}$$

where:
- $\lambda_1$ = exit rate from phase 1
- $\lambda_2$ = exit rate from phase 2  
- $p$ = probability of advancing to phase 2 (vs. exiting directly)
- Row 3 is the absorbing state

### State Mappings

For a model with observed states $\{1, 2, \ldots, S\}$, the phase-type expansion creates:

- `state_to_phases[s]`: range of phase indices for observed state $s$
- `phase_to_state[p]`: observed state corresponding to phase $p$
- `n_expanded`: total number of phases across all states

**Example:** With 2 observed states where state 1 has 2 phases and state 2 has 1 phase:
- `state_to_phases = [1:2, 3:3]`
- `phase_to_state = [1, 1, 2]`
- `n_expanded = 3`

## Forward-Backward Algorithm

### Notation

- $n$ = number of observation times
- $t_1 < t_2 < \cdots < t_n$ = observation times
- $y_s$ = observation at time $t_s$ (which phases are compatible)
- $P_{s}$ = transition probability matrix from $t_{s-1}$ to $t_s$
- $E_s$ = emission matrix row for observation $s$

### Emission Matrix Construction

The emission matrix $E$ has dimensions $(n \times n_{\text{expanded}})$. Entry $E_{s,p}$ represents the probability of observing $y_s$ given the process is in phase $p$ at time $t_s$.

For **exact observations** (obstype = 1) or **panel observations** (obstype = 2):
$$E_{s,p} = \begin{cases} 1/k_s & \text{if phase } p \text{ belongs to observed state} \\ 0 & \text{otherwise} \end{cases}$$

where $k_s$ is the number of phases for the observed state.

For **censored observations** (obstype = 3 with censoring pattern):

If the censoring pattern specifies probabilities $\pi_1, \pi_2, \ldots$ for each observed state, then:

$$E_{s,p} = \frac{\pi_{s(p)}}{k_{s(p)}}$$

where $s(p)$ is the observed state containing phase $p$ and $k_{s(p)}$ is the number of phases in that state.

**Key insight:** The probability mass for each observed state must be divided equally among its phases to preserve the correct marginal distribution over observed states.

### Forward Filtering

The forward matrices $F_s$ have dimensions $(n_{\text{expanded}} \times n_{\text{expanded}})$. Entry $F_s(i,j)$ represents:

$$F_s(i,j) = P(\text{phase } i \text{ at } t_{s-1}, \text{ phase } j \text{ at } t_s \mid y_1, \ldots, y_s)$$

**Initialization (s = 1):**

$$F_1(i,j) = p_0(i) \cdot E_1(j) \cdot P_1(i,j)$$

where $p_0$ is the initial phase distribution. For panel data with uncertain initial phase:
$$p_0(i) = \frac{1}{k_1} \mathbf{1}[\text{phase } i \in \text{initial observed state}]$$

**Important:** The TPM $P_1(i,j)$ must be included in the first step to ensure that only reachable (start, end) phase pairs have non-zero probability. This is critical for phase-type models where some transitions are impossible (e.g., phase 2 → phase 1 in Coxian).

**Recursion (s > 1):**

$$F_s(i,j) = \left(\sum_k F_{s-1}(k,i)\right) \cdot E_s(j) \cdot P_s(i,j)$$

Each $F_s$ is normalized to prevent numerical underflow.

### Backward Sampling

Given the forward matrices, we sample a phase sequence $(\phi_1, \phi_2, \ldots, \phi_n)$ backwards:

**Sample final phase:**
$$\phi_n \sim \text{Categorical}\left(\frac{\sum_i F_n(i,j)}{\sum_{i,j} F_n(i,j)}\right)$$

**Sample earlier phases (t = n-1, ..., 1):**
$$\phi_t \mid \phi_{t+1} \sim \text{Categorical}\left(\frac{F_{t+1}(i, \phi_{t+1})}{\sum_k F_{t+1}(k, \phi_{t+1})}\right)$$

**Sample initial phase:**

For panel data where the initial phase is uncertain, we must sample the phase at $t_0$ conditioned on the sampled phase at $t_1$:

$$\phi_0 \mid \phi_1 \sim \text{Categorical}\left(\frac{F_1(i, \phi_1)}{\sum_k F_1(k, \phi_1)}\right)$$

**Critical:** The initial phase must be sampled **conditioned on** $\phi_1$, not from the marginal distribution. Otherwise, we may sample impossible paths (e.g., starting in phase 2 but ending in phase 1 when $P(2 \to 1) = 0$).

### Endpoint-Conditioned Path Sampling

After backward sampling gives us phases at observation times, we sample the detailed path between consecutive observations using endpoint-conditioned CTMC sampling (uniformization method).

For each interval $[t_{s-1}, t_s]$ with start phase $\phi_{s-1}$ and end phase $\phi_s$:

1. Use uniformization to sample a path from the endpoint-conditioned distribution
2. The path respects both the start and end phase constraints
3. Paths are then collapsed to observed states using `phase_to_state`

## Importance Sampling

When the phase-type surrogate differs from the target model, importance weights correct for the proposal distribution:

$$w = \frac{p_{\text{target}}(\text{path})}{p_{\text{surrogate}}(\text{path})}$$

The log-weight is:
$$\log w = \ell_{\text{target}}(\text{path}) - \ell_{\text{surrogate}}(\text{path})$$

For valid importance sampling:
1. The surrogate must have support wherever the target has support
2. Weights should have finite variance for stable estimates

When the target and surrogate are identical, weights equal 1 exactly.

## Censoring Pattern Expansion

User-specified censoring patterns on the observed state space must be expanded to the phase space. If the original pattern assigns probability $\pi_s$ to observed state $s$, the expanded pattern assigns:

$$\pi_p^{\text{expanded}} = \frac{\pi_s}{k_s}$$

to each phase $p$ belonging to state $s$, where $k_s$ is the number of phases for state $s$.

**Rationale:** This preserves the total probability mass for each observed state:
$$\sum_{p \in \text{phases}(s)} \pi_p^{\text{expanded}} = \sum_{p \in \text{phases}(s)} \frac{\pi_s}{k_s} = k_s \cdot \frac{\pi_s}{k_s} = \pi_s$$

## Validation

### Uninformative Data Test

When observations are completely uninformative (all states possible at all times), the FFBS posterior should match the prior (forward simulation). This test validates:

1. Forward filtering correctly propagates phase distributions
2. Backward sampling correctly conditions on future observations  
3. Censoring pattern expansion preserves state probabilities
4. Initial phase sampling respects reachability constraints

**Test setup:**
- 2-state model: 1 → 2 (absorbing)
- Phase-type surrogate with 2 phases for state 1
- All observations use censoring pattern allowing both states
- Compare state prevalence from `simulate()` vs `draw_paths()`

**Results:** State prevalence matches theoretical $P(\text{state } 1 \text{ at } t) = e^{-\lambda t}$ for both methods within sampling error.

| Time | Theoretical | simulate | draw_paths |
|------|-------------|----------|------------|
| 0.5 | 0.607 | 0.607 | 0.583 |
| 1.0 | 0.368 | 0.359 | 0.348 |
| 1.5 | 0.223 | 0.220 | 0.215 |
| 2.0 | 0.135 | 0.131 | 0.130 |
| 3.0 | 0.050 | 0.049 | 0.050 |

### Phase-Type Simulation Tests

Separate tests validate that phase-type simulation matches manually-expanded Markov models. See `test/reports/phasetype_simulation_longtests.md` for details.

## Implementation Notes

### Key Functions

- `ForwardFiltering!()`: Computes forward matrices with TPM inclusion at first step
- `BackwardSampling_expanded()`: Samples phase sequence from forward matrices
- `_draw_samplepath_phasetype_original()`: Full FFBS for phase-type models
- `build_phasetype_emat_expanded()`: Constructs emission matrix with proper phase weighting
- `_merge_censoring_patterns()`: Expands user censoring patterns to phase space

### Bug Fixes (December 2025)

1. **Forward filtering first step**: Added TPM multiplication to ensure unreachable (start, end) pairs have zero probability

2. **Initial phase sampling**: Changed from marginal sampling to conditioning on the sampled endpoint phase

3. **Censoring pattern expansion**: Added division by number of phases to preserve observed state probabilities

## References

- Bladt, M., & Nielsen, B. F. (2017). *Matrix-Exponential Distributions in Applied Probability*. Springer.
- Hobolth, A., & Jensen, J. L. (2005). Statistical inference in evolutionary models of DNA sequences via the EM algorithm. *Statistical Applications in Genetics and Molecular Biology*, 4(1).
- Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE*, 77(2), 257-286.
