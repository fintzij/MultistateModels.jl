# Identifiability in Phase-Type Models with Covariates

## Overview

This document provides a rigorous analysis of identifiability options for Coxian phase-type hazard models in MultistateModels.jl. We consider a spectrum from completely unstructured (maximum flexibility, potential non-identifiability) to highly structured (reduced flexibility, guaranteed identifiability), examining both **hard constraints** (enforced via optimization) and **soft constraints** (enforced via penalties/priors).

---

## 1. Model Setup and Notation

### 1.1 Coxian Phase-Type Structure

Consider a transition from observed state $r$ to observed state $d$ (destination). The sojourn time in state $r$ follows a **Coxian phase-type distribution** with $n$ latent phases.

**Latent state space expansion:**
- Observed state $r$ expands to phases $r_1, r_2, \ldots, r_n$
- A subject enters in phase $r_1$ and progresses sequentially: $r_1 \to r_2 \to \cdots \to r_n$
- Exit to destination $d$ can occur from any phase

**Transition intensities:**

| Transition Type | Notation | Description |
|-----------------|----------|-------------|
| Progression | $\lambda_j$ | Rate from phase $r_j$ to phase $r_{j+1}$ for $j = 1, \ldots, n-1$ |
| Exit | $\mu_{j,d}$ | Rate from phase $r_j$ to destination $d$ |

### 1.2 Eigenvalues vs. Component Rates

**Important distinction:** The identifiability literature (Lindqvist, Cumani) discusses ordering constraints on **eigenvalues** of the sub-generator matrix. For an upper-triangular Coxian generator, the eigenvalues equal the **total exit rates** from each phase (the diagonal elements), not the individual component rates.

Define the **total rate out of phase $j$**:
$$
\nu_j = \lambda_j + \mu_j^{\text{total}} = \lambda_j + \sum_{d} \mu_{j,d}
$$

where $\nu_n = \mu_n^{\text{total}}$ for the final phase (no progression).

These $\nu_j$ are the eigenvalues of the sub-generator and appear on the diagonal. The canonical representation theorem orders these **total rates**, not the progression rates alone:
$$
\nu_1 \geq \nu_2 \geq \cdots \geq \nu_n
$$

This couples progression and exit rates in the identifiability constraint.

The generator matrix for the expanded state space (for a single destination) is:

$$
\mathbf{Q} = \begin{pmatrix}
-(\lambda_1 + \mu_{1,d}) & \lambda_1 & 0 & \cdots & 0 & \mu_{1,d} \\
0 & -(\lambda_2 + \mu_{2,d}) & \lambda_2 & \cdots & 0 & \mu_{2,d} \\
\vdots & & \ddots & & \vdots & \vdots \\
0 & 0 & \cdots & -(\lambda_{n-1} + \mu_{n-1,d}) & \lambda_{n-1} & \mu_{n-1,d} \\
0 & 0 & \cdots & 0 & -\mu_{n,d} & \mu_{n,d} \\
0 & 0 & \cdots & 0 & 0 & 0
\end{pmatrix}
$$

### 1.3 Multiple Destinations

When state $r$ has $K$ possible destination states $d_1, \ldots, d_K$, the total exit rate from phase $j$ is:

$$
\mu_j^{\text{total}} = \sum_{k=1}^{K} \mu_{j,d_k}
$$

The probability of exiting to destination $d_k$ given exit from phase $j$ is:

$$
P(d_k \mid \text{exit from phase } j) = \frac{\mu_{j,d_k}}{\mu_j^{\text{total}}}
$$

### 1.4 Covariate Effects

For a subject with covariate vector $\mathbf{x}$, we model rates via log-linear effects:

**Progression rates:**
$$
\lambda_j(\mathbf{x}) = \lambda_j^{(0)} \exp(\boldsymbol{\beta}^{(\lambda_j)\top} \mathbf{x})
$$

**Exit rates:**
$$
\mu_{j,d}(\mathbf{x}) = \mu_{j,d}^{(0)} \exp(\boldsymbol{\beta}^{(\mu_{j,d})\top} \mathbf{x})
$$

where superscript $(0)$ denotes baseline parameters.

### 1.5 Parameter Count (Unstructured Model)

For $n$ phases and $K$ destinations:

| Component | Count |
|-----------|-------|
| Progression baselines $\lambda_j^{(0)}$ | $n - 1$ |
| Exit baselines $\mu_{j,d}^{(0)}$ | $n \times K$ |
| **Total baseline** | $(K+1)n - 1$ |
| Progression covariates $\boldsymbol{\beta}^{(\lambda_j)}$ | $(n-1) \times p$ |
| Exit covariates $\boldsymbol{\beta}^{(\mu_{j,d})}$ | $n \times K \times p$ |
| **Total covariate** | $((K+1)n - 1) \times p$ |

where $p$ is the number of covariates.

---

## 2. The Identifiability Problem

### 2.1 Non-Uniqueness of Representations (Lindqvist 2022)

**Theorem (Telek & Horváth 2007):** Two nonredundant phase-type representations $(\mathbf{p}^{(a)}, \mathbf{Q}^{(a)})$ and $(\mathbf{p}^{(b)}, \mathbf{Q}^{(b)})$ of the same dimension yield the same distribution if and only if there exists an invertible matrix $\mathbf{B}$ with $\mathbf{B}\mathbf{1} = \mathbf{1}$ such that:

$$
\mathbf{p}^{(b)} = \mathbf{p}^{(a)\top} \mathbf{B}, \quad \mathbf{Q}^{(b)} = \mathbf{B}^{-1} \mathbf{Q}^{(a)} \mathbf{B}
$$

**Implication:** Different parameter values can yield identical likelihoods, making MLEs non-unique.

### 2.2 Coxian Identifiability (Cumani 1982, Rizk et al. 2019)

**Theorem (Canonical Representation):** Any upper-triangular phase-type distribution has a unique canonical representation with **eigenvalues** (total rates out of each phase) ordered as:

$$
\nu_1 \geq \nu_2 \geq \cdots \geq \nu_n
$$

where $\nu_j = \lambda_j + \mu_j^{\text{total}}$ is the total rate out of phase $j$.

**Theorem (Rizk et al. 2019):** For Coxian models with fixed initial distribution $\mathbf{p} = (1, 0, \ldots, 0)^\top$ and the same diagonal ordering, the representation is unique.

**Note on initial distribution:** The $\mathbf{p} = (1, 0, \ldots, 0)^\top$ assumption holds when transitions into the state are observed exactly. With panel data where entry times are unknown, the initial phase distribution must be inferred, complicating identifiability.

### 2.3 Why Covariates Complicate Matters

Covariates introduce subject-specific rates. For identifiability:
1. The ordering $\nu_1(\mathbf{x}) \geq \nu_2(\mathbf{x}) \geq \cdots$ must hold **for all** $\mathbf{x}$
2. Different covariate effects on component rates can violate the total-rate ordering for some subjects
3. Proportional hazards effects (same $\boldsymbol{\beta}$ for all rates) preserve ordering

---

## 3. Spectrum of Model Structures

We organize constraints along two dimensions: **baseline hazards** and **covariate effects**.

### 3.1 Baseline Hazard Structures

#### Level B0: Completely Unstructured

**Parameters:** All $\lambda_j^{(0)}$ and $\mu_{j,d}^{(0)}$ are free.

**Count:** $(K+1)n - 1$ baseline parameters.

**Identifiability:** ❌ Not guaranteed. Equivalent representations exist.

**Implementation:** None required — this is the default.

**Practicality:** ✅ Trivial. Current MultistateModels.jl default for `:pt` hazards.

**Use case:** Exploratory analysis with abundant data.

---

#### Level B1: Eigenvalue Ordering (Lindqvist Canonical)

**Constraint:** The **total rates** (eigenvalues) must be ordered:
$$
\nu_1^{(0)} \geq \nu_2^{(0)} \geq \cdots \geq \nu_n^{(0)}
$$

where $\nu_j^{(0)} = \lambda_j^{(0)} + \mu_j^{\text{total},(0)}$ for $j < n$ and $\nu_n^{(0)} = \mu_n^{\text{total},(0)}$.

**Important:** This constraint couples progression rates $\lambda_j$ with exit rates $\mu_{j,d}$. It is **not** simply an ordering on the progression rates alone.

**Count:** $(K+1)n - 1$ parameters (unchanged, but constrained).

**Identifiability:** ✅ Guaranteed for baseline (single-covariate-value).

**Implementation options:**

| Method | How | Complexity |
|--------|-----|------------|
| **Inequality constraints** | Add $\nu_j \leq \nu_{j-1}$ to optimizer | Moderate — requires constrained optimization |
| **Reparameterization** | Work with $\nu_j$ directly, derive $\lambda_j = \nu_j - \mu_j^{\text{total}}$ | High — complex bookkeeping |
| **Soft penalty** | Add $\rho \sum_j \max(0, \nu_j - \nu_{j-1})^2$ to objective | Low — unconstrained optimization |

**Practicality:** ⚠️ Moderate difficulty. The constraint couples all parameters, making it awkward to implement cleanly. The reparameterization approach requires tracking derived quantities. **Recommend soft penalty as simplest option.**

---

#### Level B2: SCTP (Stationary Conditional Transition Probabilities)

**Constraint:** Exit rates from each phase scale by a common **phase effect** $\tau_j$:

$$
\mu_{j,d}^{(0)} = \tau_j \cdot \mu_{1,d}^{(0)} \quad \text{for all } d, \quad \tau_1 \equiv 1
$$

**Equivalent formulation (log scale):**
$$
\log \mu_{j,d}^{(0)} - \log \mu_{1,d}^{(0)} = \log \tau_j \quad \text{(constant across } d \text{)}
$$

**Parameter reduction:**

| Before SCTP | After SCTP |
|-------------|------------|
| $nK$ exit baselines | $K$ baseline rates + $(n-1)$ phase effects |

**New count:** $(n-1) + K + (n-1) = 2n + K - 2$ baseline parameters.

**Identifiability:** ✅ Better conditioned. Preserves exit probability stationarity.

**Implementation options:**

| Method | How | Complexity |
|--------|-----|------------|
| **Equality constraints** | Add $(n-1)(K-1)$ constraints: $\log \mu_{j,d_1} - \log \mu_{1,d_1} = \log \mu_{j,d_2} - \log \mu_{1,d_2}$ | Moderate |
| **Reparameterization** | Store $\mu_{1,d}$ and $\tau_j$; compute $\mu_{j,d} = \tau_j \mu_{1,d}$ | Low — **recommended** |
| **Soft penalty** | Penalize variance of $\log(\mu_{j,d}/\mu_{1,d})$ across $d$ | Low |

**Practicality:** ✅ **Already implemented** in MultistateModels.jl via `_generate_sctp_constraints()` in `expansion_constraints.jl`. Uses equality constraints approach.

**Interpretation:** The probability of exiting to each destination is constant regardless of phase (hence "stationary" in sojourn time).

---

#### Level B3: B1 + B2 Combined (Ordered + SCTP)

**Constraints:**
1. $\nu_1^{(0)} \geq \nu_2^{(0)} \geq \cdots \geq \nu_n^{(0)}$ (total rate ordering)
2. $\mu_{j,d}^{(0)} = \tau_j \cdot \mu_{1,d}^{(0)}$ (SCTP)

**Count:** $(n-1) + K + (n-1) = 2n + K - 2$ baseline parameters (ordered + SCTP).

**Identifiability:** ✅ Strong.

**Implementation:** Combine B1 and B2 approaches. With SCTP reparameterization, the total rate simplifies to:
$$
\nu_j = \lambda_j + \tau_j \sum_d \mu_{1,d}
$$
so ordering constraint becomes a function of $\lambda_j$, $\tau_j$, and the phase-1 exit rates.

**Practicality:** ⚠️ Moderate. SCTP is implemented; eigenvalue ordering adds complexity. For soft constraint version, add ordering penalty to existing SCTP implementation.

---

#### Level B4: Erlang Structure (Equal Progression Rates)

**Constraint:** All progression rates equal:
$$
\lambda_1^{(0)} = \lambda_2^{(0)} = \cdots = \lambda_{n-1}^{(0)} \equiv \lambda^{(0)}
$$

**Count:** $1 + nK$ baseline parameters (or $1 + K + (n-1) = n + K$ with SCTP).

**Identifiability:** ✅ Strongly identified. The sojourn distribution becomes a generalized Erlang mixture.

**Implementation:** 
- **Reparameterization:** Use single $\lambda$ parameter; all progression hazards reference same parameter name.
- In `_build_progression_hazard()`, use shared parameter name `h{from}_lambda` instead of `h{from}_{a}{b}_rate`.

**Practicality:** ✅ Easy. Simple change to parameter naming in hazard expansion. Automatically satisfies eigenvalue ordering (all $\lambda_j$ equal, so ordering determined by exit rates only).

**Note:** Very restrictive; may not fit data well. Use when data are limited or for hypothesis testing.

---

### 3.2 Covariate Effect Structures

#### Level C0: Completely Unstructured

**Parameters:** Each rate has its own covariate vector:
- $\boldsymbol{\beta}^{(\lambda_j)}$ for $j = 1, \ldots, n-1$
- $\boldsymbol{\beta}^{(\mu_{j,d})}$ for $j = 1, \ldots, n$, $d = 1, \ldots, K$

**Count:** $((K+1)n - 1) \times p$ covariate parameters.

**Identifiability:** ❌ Poor. Covariates can break eigenvalue ordering.

**Implementation:** None required — this is what happens with independent parameter names per phase.

**Practicality:** ✅ Trivial. **Partially implemented**: Current MultistateModels.jl puts independent covariate effects on exit rates (`h12_a_x`, `h12_b_x`), but does **not** put covariates on progression rates. This is the default behavior (`:per_phase`).

**Current state:** Exit rates have phase-specific covariates (C0 for exits); progression rates have no covariates.

---

#### Level C1: Shared Covariate Effects Across Phases (per destination)

**Constraint:** All phases share the same covariate effect for a given destination:
$$
\boldsymbol{\beta}^{(\mu_{j,d})} = \boldsymbol{\beta}^{(d)} \quad \text{for all } j
$$

**Exit rate model:**
$$
\mu_{j,d}(\mathbf{x}) = \mu_{j,d}^{(0)} \exp(\boldsymbol{\beta}^{(d)\top} \mathbf{x})
$$

**Count:** $K \times p$ covariate parameters for exits (no covariates on progression).

**Identifiability:** ✅ Good. Substantial improvement over C0.

**Preserves:** SCTP property (if baseline satisfies SCTP) — the phase effect ratios $\tau_j = \mu_{j,d}/\mu_{1,d}$ remain constant across $\mathbf{x}$.

**Does NOT preserve:** Eigenvalue ordering. Different $\boldsymbol{\beta}^{(d)}$ for different destinations can reorder total rates $\nu_j(\mathbf{x})$ for some subjects. See Section 7.4 for details.

**Implementation:** Available as option via `covariate_constraints=:C1`:
```julia
# C0 (default): phase-specific covariates
h12 = Hazard(@formula(0 ~ x), :pt, 1, 2; n_phases=2)
# Parameter names: h12_a_x, h12_b_x (independent per phase)

# C1: destination-specific shared covariates
h12 = Hazard(@formula(0 ~ x), :pt, 1, 2; n_phases=2, covariate_constraints=:C1)
# Parameter names: h12_x (shared across phases a,b for destination 2)
```

In `_build_exit_hazard()`, this changes parameter naming:
```julia
# C0 (covariate_constraints=:C0, default): phase-specific
covar_parnames = [Symbol("h$(from)$(to)_$(phase_letter)_$(c)") for c in covars]
# e.g., h12_a_x, h12_b_x

# C1 (covariate_constraints=:C1): destination-specific  
covar_parnames = [Symbol("h$(from)$(to)_$(c)") for c in covars]
# e.g., h12_x (shared across phases a,b for destination 2)
#       h13_x (shared across phases a,b for destination 3)
```

**Practicality:** ✅ **Easy to implement and recommended for most applications.** Simple change to parameter naming. Reduces parameters from $nKp$ to $Kp$. Allows biologically meaningful destination-specific effects (e.g., different covariate effects for recovery vs death).

---

#### Level C2: Shared Covariate Effects Across Destinations (per phase)

**Constraint:** All destinations share the same covariate effect for a given phase:
$$
\boldsymbol{\beta}^{(\mu_{j,d})} = \boldsymbol{\beta}^{(j)} \quad \text{for all } d
$$

**Count:** $n \times p$ covariate parameters for exits.

**Implementation:** Use phase-based covariate parameter names:
```julia
# Proposed (C2): phase-specific, destination-agnostic
covar_parnames = [Symbol("h$(from)_$(phase_letter)_$(c)") for c in covars]
# e.g., h1_a_x (shared across destinations 2,3 for phase a)
#       h1_b_x (shared across destinations 2,3 for phase b)
```

**Practicality:** ✅ Easy. Same implementation pattern as C1 with different naming scheme.

**Note:** This is a different structure from C1. May be appropriate when phase represents disease severity affecting all outcomes similarly. Less common in practice than C1 or C3.

---

#### Level C3: Shared Covariate Effects Across All Exit Rates

**Constraint:** A single covariate vector for all exit rates:
$$
\boldsymbol{\beta}^{(\mu_{j,d})} = \boldsymbol{\beta}^{(\mu)} \quad \text{for all } j, d
$$

**Exit rate model:**
$$
\mu_{j,d}(\mathbf{x}) = \mu_{j,d}^{(0)} \exp(\boldsymbol{\beta}^{(\mu)\top} \mathbf{x})
$$

**Count:** $p$ covariate parameters for exits.

**Identifiability:** ✅ Strong. This is the Titman & Sharples approach.

**Preserves:** SCTP property (if baseline satisfies SCTP).

**Does NOT preserve:** Eigenvalue ordering. Since progression rates have no covariates, the ordering $\nu_j(\mathbf{x}) = \lambda_j + e^{\boldsymbol{\beta}^\top \mathbf{x}} \mu_j^{\text{total},(0)}$ can still be violated.

**Implementation:** Use state-based covariate parameter names (no phase, no destination):
```julia
# Proposed (C3): fully shared across phases and destinations
covar_parnames = [Symbol("h$(from)_exit_$(c)") for c in covars]
# e.g., h1_exit_x (shared across all exit hazards from state 1)
```

**Practicality:** ✅ Easy. Dramatically reduces parameters ($nKp \to p$).

**Note:** More restrictive than C1 — assumes all destinations have identical covariate effects. May be too strong if destinations represent qualitatively different outcomes.

---

#### Level C4: Fully Shared (Proportional Hazards on All Rates)

**Constraint:** A single covariate vector affects **all** rates (progression + exit):
$$
\lambda_j(\mathbf{x}) = \lambda_j^{(0)} \exp(\boldsymbol{\beta}^\top \mathbf{x}), \quad
\mu_{j,d}(\mathbf{x}) = \mu_{j,d}^{(0)} \exp(\boldsymbol{\beta}^\top \mathbf{x})
$$

**Count:** $p$ covariate parameters total.

**Interpretation:** Accelerated failure time on the phase-type time scale. Covariates speed up or slow down the entire Coxian process.

**Identifiability:** ✅ Strongest. Total-rate ordering $\nu_1(\mathbf{x}) \geq \cdots \geq \nu_n(\mathbf{x})$ preserved for all subjects because all rates scale by the same factor.

**Implementation:** Requires adding covariates to progression hazards (not currently supported) with shared parameter names:
```julia
# All hazards (progression and exit) use same covariate names
covar_parnames = [Symbol("h$(from)_$(c)") for c in covars]
# e.g., h1_x (shared across ALL hazards originating from state 1)
```

**Practicality:** ⚠️ **Requires new functionality.** Current implementation does not put covariates on progression rates. Would need to:
1. Add covariate support to `_build_progression_hazard()`
2. Use shared parameter naming across progression and exit hazards

**When to use:** When you believe covariates uniformly accelerate/decelerate the entire disease process.

---

#### Level C5: Separate Progression and Exit Effects

**Constraint:** One effect for all progression, one for all exits:
$$
\lambda_j(\mathbf{x}) = \lambda_j^{(0)} \exp(\boldsymbol{\beta}^{(\lambda)\top} \mathbf{x}), \quad
\mu_{j,d}(\mathbf{x}) = \mu_{j,d}^{(0)} \exp(\boldsymbol{\beta}^{(\mu)\top} \mathbf{x})
$$

**Count:** $2p$ covariate parameters.

**Interpretation:** Covariates can differentially affect "how fast you progress through phases" vs "how fast you exit."

**Identifiability:** ✅ Good. Does **not** automatically preserve total-rate ordering because different $\boldsymbol{\beta}^{(\lambda)}$ and $\boldsymbol{\beta}^{(\mu)}$ can reorder $\nu_j(\mathbf{x}) = \lambda_j(\mathbf{x}) + \mu_j^{\text{total}}(\mathbf{x})$ for different $\mathbf{x}$.

**Implementation:**
```julia
# Progression hazards use one set of names
progression_covar_parnames = [Symbol("h$(from)_prog_$(c)") for c in covars]
# e.g., h1_prog_x

# Exit hazards use another set
exit_covar_parnames = [Symbol("h$(from)_exit_$(c)") for c in covars]  
# e.g., h1_exit_x
```

**Practicality:** ⚠️ **Requires new functionality** (covariates on progression). Less restrictive than C4 but still requires the same infrastructure changes.

**When to use:** When you believe progression speed and exit propensity respond differently to covariates (e.g., treatment delays progression but doesn't affect exit probability).

---

### 3.3 Summary Table: Model Structures

| Baseline | Covariate | Params | Identifiability | Practicality |
|----------|-----------|--------|-----------------|--------------|
| B0 (free) | C0 (free) | $(K+1)n - 1 + nKp$ | ❌ Poor | Option: `:unstructured` |
| B0 (free) | C1 (shared/dest) | $(K+1)n - 1 + Kp$ | ⚠️ Marginal | ✅ Easy |
| B2 (SCTP) | C0 (free) | $n + K - 1 + nKp$ | ⚠️ Marginal | Option: `:sctp` |
| B2 (SCTP) | C1 (shared/dest) | $n + K - 1 + Kp$ | ✅ Good | Option: `:sctp` |
| **B3 (ordered+SCTP)** | **C1 (shared/dest)** | $n + K - 1 + Kp$ | ✅ **Strong** | **✅ DEFAULT** (`:eigorder_sctp`) |
| B4 (Erlang+SCTP) | C1 (shared/dest) | $K + 1 + Kp$ | ✅ Strongest | ✅ Easy |

**Notes:**
- Parameter counts assume $n$ phases, $K$ destinations, $p$ covariates, no covariates on progression
- **Default (v0.4.0+)**: B3 (SCTP + eigenvalue ordering) + C1 (homogeneous covariates) via `coxian_structure=:eigorder_sctp`
- **Recommended**: `:eigorder_sctp` (B3 + C1) — strong identifiability, interpretable, practical
- Use `:sctp` for SCTP only (no eigenvalue ordering) if ordering constraints cause optimization issues
- Use `:unstructured` for maximum flexibility (not recommended—poor identifiability)
- `:ordered_sctp` is an alias for `:eigorder_sctp` (deprecated)

---

## 4. Hard vs. Soft Constraints

### 4.1 Hard Constraints (Optimization)

Hard constraints are enforced exactly via constrained optimization (e.g., Optimization.jl with constraints).

**Equality constraints:**
$$
c(\boldsymbol{\theta}) = 0
$$

**Inequality constraints:**
$$
g(\boldsymbol{\theta}) \leq 0
$$

#### Eigenvalue Ordering (B1)

**Implementation:** Transform to unconstrained space:
$$
\nu_j^{(0)} = \nu_{j+1}^{(0)} + \exp(\delta_j), \quad \delta_j \in \mathbb{R}
$$

where $\nu_j = \lambda_j + \mu_j^{\text{total}}$. Or use inequality constraints: $\nu_j \leq \nu_{j-1}$.

**Note:** This is more complex than constraining progression rates alone because it couples $\lambda_j$ and $\mu_{j,d}$.

#### SCTP (B2)

**Implementation:** Equality constraints on log differences:
$$
\log \mu_{j,d_1}^{(0)} - \log \mu_{1,d_1}^{(0)} - \log \mu_{j,d_2}^{(0)} + \log \mu_{1,d_2}^{(0)} = 0
$$

This is $(n-1)(K-1)$ constraints.

#### Shared Covariates (C1-C4)

**Implementation:** Use the same parameter symbols in the model; no separate constraints needed.

### 4.2 Soft Constraints (Penalties)

Soft constraints add penalty terms to the log-likelihood:
$$
\ell_{\text{pen}}(\boldsymbol{\theta}) = \ell(\boldsymbol{\theta}) - \text{Penalty}(\boldsymbol{\theta})
$$

#### Eigenvalue Ordering Penalty

$$
P_{\text{order}}(\boldsymbol{\nu}) = \rho \sum_{j=2}^{n} \left[ \max(0, \nu_j^{(0)} - \nu_{j-1}^{(0)}) \right]^2
$$

where $\nu_j = \lambda_j + \mu_j^{\text{total}}$.

**Behavior:** Zero when total-rate ordering is satisfied; quadratic penalty when violated.

#### SCTP Deviation Penalty

Let $\hat{\tau}_j^{(d)} = \mu_{j,d}^{(0)} / \mu_{1,d}^{(0)}$. Under SCTP, $\hat{\tau}_j^{(d)}$ should be constant across $d$.

$$
P_{\text{SCTP}}(\boldsymbol{\mu}) = \rho \sum_{j=2}^{n} \sum_{d_1 < d_2} \left( \log \hat{\tau}_j^{(d_1)} - \log \hat{\tau}_j^{(d_2)} \right)^2
$$

#### Covariate Similarity Penalty

Encourage similar covariate effects across phases:
$$
P_{\text{cov}}(\boldsymbol{\beta}) = \rho \sum_{j=2}^{n} \sum_{d} \left\| \boldsymbol{\beta}^{(\mu_{j,d})} - \boldsymbol{\beta}^{(\mu_{1,d})} \right\|^2
$$

#### Progression Rate Stability (Titman & Sharples Appendix)

Penalize extreme $\lambda$ values to ensure identifiability when testing Markov vs semi-Markov:
$$
P_{\lambda}(\boldsymbol{\lambda}) = \sum_{j} \left( C_j \log \lambda_j - C_j \lambda_j \alpha_j \right)
$$

This is equivalent to a Gamma prior on $\lambda_j$ with shape $C_j$ and rate $\alpha_j^{-1}$.

### 4.3 Trade-offs

| Approach | Pros | Cons |
|----------|------|------|
| **Hard constraints** | Exact enforcement; guarantees structure | Constrained optimization harder; may reduce convergence |
| **Soft constraints** | Unconstrained optimization; smooth gradients | Structure only approximate; requires tuning $\rho$ |

---

## 5. Implementation in MultistateModels.jl

### 5.1 Current Capabilities

**Baseline hazards:**
- ✅ Unstructured (B0): Via `coxian_structure=:unstructured`
- ✅ SCTP (B2): Via `coxian_structure=:sctp` — implemented in `expansion_constraints.jl`
- ✅ Eigenvalue ordering + SCTP (B3): Via `coxian_structure=:eigorder_sctp` **(DEFAULT)**
- ❌ Erlang (B4): Not implemented

**Covariate effects:**
- ✅ Homogeneous (C1): All phases share covariate effects per destination **(DEFAULT)**, `covariate_constraints=:homogeneous`
- ✅ Unstructured (C0): Each phase × destination has independent effects, `covariate_constraints=:unstructured`
- ❌ Shared per phase (C2): Not implemented
- ❌ Shared across all exits (C3): Not implemented
- ❌ Fully shared (C4): Not implemented (requires covariates on progression)

**Constraint infrastructure:**
- ✅ Optimization.jl integration supports constraints
- ✅ `constraints` field in `ProposalConfig`
- ⚠️ Penalty infrastructure partially available (spline smoothing penalties exist)

### 5.2 Implemented Options

#### Baseline Structure (`coxian_structure`)

| Option | Symbol | Description |
|--------|--------|-------------|
| **Eigenvalue ordered + SCTP** | `:eigorder_sctp` | SCTP + ν₁ ≥ ν₂ ≥ ... ≥ νₙ **(DEFAULT, recommended)** |
| SCTP only | `:sctp` | P(dest \| leaving) constant across phases |
| Unstructured | `:unstructured` | All rates free (not recommended) |
| *(deprecated)* | `:ordered_sctp` | Alias for `:eigorder_sctp` |

```julia
# Default: eigorder_sctp (SCTP + eigenvalue ordering) — recommended
h12 = Hazard(@formula(0 ~ x), :pt, 1, 2; n_phases=2)

# SCTP only (no eigenvalue ordering)
h12 = Hazard(@formula(0 ~ x), :pt, 1, 2; n_phases=2, coxian_structure=:sctp)

# Unstructured (not recommended — poor identifiability)
h12 = Hazard(@formula(0 ~ x), :pt, 1, 2; n_phases=2, coxian_structure=:unstructured)
```

#### Covariate Constraints (`covariate_constraints`)

| Option | Symbol | Parameter Count | Description |
|--------|--------|-----------------|-------------|
| **Homogeneous (C1)** | `:homogeneous` | $Kp$ | Shared across phases, different per destination **(DEFAULT)** |
| Unstructured (C0) | `:unstructured` | $nKp$ | Each phase has independent covariate effects |

```julia
# Default: homogeneous (C1) — shared covariates per destination
h12 = Hazard(@formula(0 ~ x), :pt, 1, 2; n_phases=2)
# Parameter names: h12_x (shared across phases a,b)

# Unstructured (C0): phase-specific covariates
h12 = Hazard(@formula(0 ~ x), :pt, 1, 2; n_phases=2, covariate_constraints=:unstructured)
# Parameter names: h12_a_x, h12_b_x (independent per phase)
```

**Recommendation:** Use `:homogeneous` (default) for most applications — better identifiability, fewer parameters, more interpretable.

### 5.3 Default Configuration

As of v0.4.0, the **default configuration** is:
- **Baseline**: `:eigorder_sctp` (SCTP + eigenvalue ordering)
- **Covariates**: `:homogeneous` (shared per destination)

This provides **strong identifiability** while remaining practical for most applications.

```julia
# These are all equivalent (using defaults):
h12 = Hazard(:pt, 1, 2; n_phases=2)
h12 = Hazard(:pt, 1, 2; n_phases=2, coxian_structure=:eigorder_sctp, covariate_constraints=:homogeneous)

# Model-level override
model = multistatemodel(h12; data=df, coxian_structure=:sctp)  # SCTP only
```

### 5.4 Future Implementations

#### Priority 1: Erlang Structure (B4)

**Constraint:** All progression rates equal: λ₁ = λ₂ = ... = λₙ₋₁

```julia
# Proposed API
h12 = Hazard(:pt, 1, 2; n_phases=3, coxian_structure=:erlang_sctp)
```

#### Priority 2: Penalty Infrastructure

Add generic penalty support to `fit()`:

```julia
fit(model; 
    penalty = (
        ordering = (type=:soft, weight=10.0),
        sctp = (type=:soft, weight=5.0),
        lambda_prior = (type=:gamma, shape=1.0, rate=1.0)
    )
)
```

### 5.5 Structure Options Reference

| `coxian_structure` | Baseline Constraints | Status |
|-------------|---------------------|--------|
| `:unstructured` | None (B0) | ✅ Implemented |
| `:sctp` | SCTP only (B2) | ✅ Implemented |
| `:eigorder_sctp` | SCTP + ordering (B3) | ✅ **DEFAULT** |
| `:ordered_sctp` | Alias for `:eigorder_sctp` | ✅ Deprecated |
| `:erlang_sctp` | Erlang + SCTP (B4) | ❌ Future |

| `covariate_constraints` | Covariate Structure | Status |
|---------------------|---------------------|--------|
| `:homogeneous` | Shared per destination (C1) | ✅ **DEFAULT** |
| `:unstructured` | All independent (C0) | ✅ Implemented |

---

## 6. Practical Recommendations

### 6.1 Default Recommendation

For most applications, use the defaults:
- **Baseline:** `:ordered_sctp` (B3)
- **Covariates:** `:across_phases` (C1) or `:exits` (C3)

This provides:
- Identifiability guarantees (canonical representation)
- Interpretable phase effects (SCTP)
- Reasonable parameter count
- Preserved ordering for all subjects

### 6.2 Exploratory Analysis

Start with:
- **Baseline:** `:eigorder_sctp` (SCTP + eigenvalue ordering) — **this is the default**
- **Covariates:** `:homogeneous` (shared per destination) — **this is the default**

This provides:
- Identifiability guarantees (canonical representation via eigenvalue ordering)
- Interpretable phase effects (SCTP)
- Reasonable parameter count
- Preserved ordering for all subjects (with `:homogeneous` covariates)

```julia
# Just use the defaults:
h12 = Hazard(:pt, 1, 2; n_phases=3)
model = multistatemodel(h12; data=df)
```

### 6.2 Exploratory Analysis

Start with defaults, then relax if needed:
- **Baseline:** `:sctp` (if eigenvalue ordering causes issues)
- **Covariates:** `:homogeneous` (keep this for identifiability)

Test whether data support more structure via likelihood ratio tests (using modified LRT from Titman & Sharples if needed).

### 6.3 Limited Data

Use maximal structure:
- **Baseline:** `:eigorder_sctp` (default) or future `:erlang_sctp`
- **Covariates:** `:homogeneous` (default)

### 6.4 Testing Markov vs Semi-Markov

Use Titman & Sharples modified LRT:
1. Fit with penalty on $\lambda$ (Gamma prior)
2. Compare penalized likelihoods
3. Use χ² approximation for degrees of freedom (excluding $\lambda$ parameters)

---

## 7. Mathematical Proofs and Details

### 7.1 SCTP Preserves Exit Probabilities

**Claim:** Under SCTP ($\mu_{j,d} = \tau_j \mu_{1,d}$), the conditional exit probability is phase-invariant.

**Proof:**
$$
P(d \mid \text{exit from phase } j) = \frac{\mu_{j,d}}{\sum_{d'} \mu_{j,d'}} = \frac{\tau_j \mu_{1,d}}{\sum_{d'} \tau_j \mu_{1,d'}} = \frac{\mu_{1,d}}{\sum_{d'} \mu_{1,d'}}
$$

which is independent of $j$. ∎

### 7.2 Shared Covariates Preserve SCTP

**Claim:** If baseline satisfies SCTP and covariates are shared across phases (C1), then SCTP holds for all $\mathbf{x}$.

**Proof:** Let $\mu_{j,d}(\mathbf{x}) = \mu_{j,d}^{(0)} \exp(\boldsymbol{\beta}^{(d)\top} \mathbf{x})$.

The phase effect ratio:
$$
\frac{\mu_{j,d}(\mathbf{x})}{\mu_{1,d}(\mathbf{x})} = \frac{\mu_{j,d}^{(0)} \exp(\boldsymbol{\beta}^{(d)\top} \mathbf{x})}{\mu_{1,d}^{(0)} \exp(\boldsymbol{\beta}^{(d)\top} \mathbf{x})} = \frac{\mu_{j,d}^{(0)}}{\mu_{1,d}^{(0)}} = \tau_j
$$

which is independent of $\mathbf{x}$ and $d$. ∎

### 7.3 Shared Covariates Preserve Total-Rate Ordering

**Claim:** If baseline satisfies $\nu_1^{(0)} \geq \nu_2^{(0)} \geq \cdots$ (where $\nu_j = \lambda_j + \mu_j^{\text{total}}$) and covariates are shared across **all** rates (C4), then the total-rate ordering holds for all $\mathbf{x}$.

**Proof:** Under C4, all rates scale by the same factor:
$$
\lambda_j(\mathbf{x}) = \lambda_j^{(0)} \exp(\boldsymbol{\beta}^\top \mathbf{x}), \quad
\mu_{j,d}(\mathbf{x}) = \mu_{j,d}^{(0)} \exp(\boldsymbol{\beta}^\top \mathbf{x})
$$

Therefore:
$$
\nu_j(\mathbf{x}) = \lambda_j(\mathbf{x}) + \mu_j^{\text{total}}(\mathbf{x}) = \left( \lambda_j^{(0)} + \mu_j^{\text{total},(0)} \right) \exp(\boldsymbol{\beta}^\top \mathbf{x}) = \nu_j^{(0)} \exp(\boldsymbol{\beta}^\top \mathbf{x})
$$

For any $j$:
$$
\nu_{j-1}(\mathbf{x}) - \nu_j(\mathbf{x}) = \exp(\boldsymbol{\beta}^\top \mathbf{x}) \left( \nu_{j-1}^{(0)} - \nu_j^{(0)} \right) \geq 0
$$

since $\exp(\cdot) > 0$ and $\nu_{j-1}^{(0)} \geq \nu_j^{(0)}$ by assumption. ∎

**Note:** This proof requires C4 (all rates share the same $\boldsymbol{\beta}$). Under C5 (separate $\boldsymbol{\beta}^{(\lambda)}$ and $\boldsymbol{\beta}^{(\mu)}$), the total rates $\nu_j(\mathbf{x})$ can become reordered for some $\mathbf{x}$ even if baseline is ordered.

### 7.4 C1 Does Not Preserve Eigenvalue Ordering

**Claim:** Under C1 (shared covariates per destination), eigenvalue ordering can be violated even if baseline is ordered.

**Setup:** Under C1 with no covariates on progression:
$$
\nu_j(\mathbf{x}) = \lambda_j + \sum_d \mu_{j,d}^{(0)} \exp(\boldsymbol{\beta}^{(d)\top} \mathbf{x})
$$

**Counterexample:** Consider 2 phases, 2 destinations $(d_1, d_2)$, 1 binary covariate $x \in \{0, 1\}$:
- $\lambda_1 = 1, \lambda_2 = 0.5$ (no final progression)
- $\mu_{1,d_1}^{(0)} = 0.3, \mu_{2,d_1}^{(0)} = 0.8$ (phase 2 exits faster to $d_1$)
- $\mu_{1,d_2}^{(0)} = 0.5, \mu_{2,d_2}^{(0)} = 0.2$ (phase 1 exits faster to $d_2$)
- $\beta^{(d_1)} = 2, \beta^{(d_2)} = -1$

At baseline ($x = 0$):
$$
\nu_1^{(0)} = 1 + 0.3 + 0.5 = 1.8, \quad \nu_2^{(0)} = 0.5 + 0.8 + 0.2 = 1.5
$$
Ordering satisfied: $\nu_1^{(0)} > \nu_2^{(0)}$ ✓

At $x = 1$:
$$
\nu_1(1) = 1 + 0.3 e^2 + 0.5 e^{-1} \approx 1 + 2.22 + 0.18 = 3.40
$$
$$
\nu_2(1) = 0.5 + 0.8 e^2 + 0.2 e^{-1} \approx 0.5 + 5.91 + 0.07 = 6.48
$$
Ordering violated: $\nu_2(1) > \nu_1(1)$ ✗

**Conclusion:** C1 preserves SCTP (Theorem 7.2) but does **not** guarantee eigenvalue ordering. This is acceptable in practice because:
1. SCTP is the more important constraint for interpretability
2. Eigenvalue ordering primarily matters for theoretical identifiability at the boundary
3. The added flexibility of destination-specific effects (C1 vs C3/C4) is often substantively important

---

## 8. References

1. **Lindqvist, B.H.** (2022). Phase-type models for competing risks, with emphasis on identifiability issues. *Lifetime Data Analysis*, 29, 318-341.

2. **Titman, A.C. & Sharples, L.D.** (2010). Semi-Markov models with phase-type sojourn distributions. *Biometrics*, 66, 742-752.

3. **Cumani, A.** (1982). On the canonical representation of homogeneous Markov processes modelling failure-time distributions. *Microelectronics Reliability*, 22, 583-602.

4. **Rizk, J., Burke, K. & Walsh, D.** (2019). Identifiability and estimation of Coxian phase-type models. *arXiv preprint*.

5. **Telek, M. & Horváth, G.** (2007). A minimal representation of Markov arrival processes and a moments matching method. *Performance Evaluation*, 64, 1153-1168.

---

## Appendix A: MultistateModels.jl Parameter Naming

### Baseline Parameters

Naming convention for $n$-phase Coxian hazard from state 1 to states 2, 3:

| Parameter | Name | Description |
|-----------|------|-------------|
| $\lambda_1$ | `h1_ab_rate` | Progression 1→2 (phases a→b) |
| $\lambda_2$ | `h1_bc_rate` | Progression 2→3 (phases b→c) |
| $\mu_{1,2}$ | `h12_a_rate` | Exit from phase a to state 2 |
| $\mu_{2,2}$ | `h12_b_rate` | Exit from phase b to state 2 |
| $\mu_{3,2}$ | `h12_c_rate` | Exit from phase c to state 2 |

### Covariate Parameters

**`:homogeneous` (default):** Destination-specific shared effects (C1)

| Parameter | Name | Description |
|-----------|------|-------------|
| $\beta^{(d=2)}$ | `h12_age` | Age effect on all exits to state 2 (shared across phases a,b) |
| $\beta^{(d=3)}$ | `h13_age` | Age effect on all exits to state 3 (shared across phases a,b) |

**`:unstructured`:** Phase-specific (C0)

| Parameter | Name | Description |
|-----------|------|-------------|
| $\beta^{(\mu_{1,2})}$ | `h12_a_age` | Age effect on exit from phase a to state 2 |
| $\beta^{(\mu_{2,2})}$ | `h12_b_age` | Age effect on exit from phase b to state 2 |
| $\beta^{(\mu_{1,3})}$ | `h13_a_age` | Age effect on exit from phase a to state 3 |
| $\beta^{(\mu_{2,3})}$ | `h13_b_age` | Age effect on exit from phase b to state 3 |

### API Example

```julia
# Default: eigorder_sctp + homogeneous covariates (recommended)
h = Hazard(@formula(0 ~ age), :pt, 1, 2; n_phases=2)
# coxian_structure=:eigorder_sctp, covariate_constraints=:homogeneous
# Parameters: h12_age (shared across phases a,b)

# SCTP only (no eigenvalue ordering):
h = Hazard(@formula(0 ~ age), :pt, 1, 2; n_phases=2, coxian_structure=:sctp)

# Unstructured covariates (phase-specific):
h = Hazard(@formula(0 ~ age), :pt, 1, 2; n_phases=2, covariate_constraints=:unstructured)
# Parameters: h12_a_age, h12_b_age (one per phase)

# Fully unstructured (not recommended):
h = Hazard(@formula(0 ~ age), :pt, 1, 2; n_phases=2, 
           coxian_structure=:unstructured, covariate_constraints=:unstructured)

# Model-level override:
model = multistatemodel(h12; data=df, coxian_structure=:sctp)  # SCTP only at model level
```

