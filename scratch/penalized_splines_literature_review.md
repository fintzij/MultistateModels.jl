# Penalized Splines for Multistate Models: Design Document

## Executive Summary

This document guides the implementation of penalized splines for baseline hazards in `MultistateModels.jl`. We address two fundamental questions:

1. **What to penalize?** — Which functions should have smoothness penalties, and how should those penalties be constructed?
2. **How to set tuning parameters?** — Given a penalty structure, how do we select the smoothing parameters $\lambda$?

Our approach synthesizes methods from four key papers:
- **Machado et al. (2018)** and **Eletti et al. (2024)** for penalized multistate models
- **Wood (2016)** for penalty matrix construction
- **Wood (2024)** for smoothing parameter selection via Neighbourhood Cross-Validation (NCV)

---

## The Two Key Design Questions

### Question 1: What to Penalize?

We have flexibility in what quantities we penalize for smoothness:

| Target | Formula | Verdict |
|:-------|:--------|:--------|
| **Individual hazards** $h_{rs}(t)$ | $\lambda_{rs} \int [h_{rs}^{(m)}]^2 dt$ | ✅ Standard, necessary |
| **Total hazard** $H_r(t) = \sum_s h_{rs}(t)$ | $\mu_r \int [H_r^{(m)}]^2 dt$ | ✅ Our addition for competing risks |
| **Cumulative hazard** $\Lambda_{rs}(t)$ | Equivalent to $(m-1)$-th derivative of hazard | 🔍 Explore |
| **Transition probabilities** $P_{rs}(t)$ | Non-quadratic in parameters | 🔍 Explore (All models) |
| **Sojourn times** $S_r(t)$ | Penalize density/survival curvature | 🔍 Explore (Observable) |
| **Transition densities** $f_{rs}(t)$ | Penalize event density curvature | 🔍 Explore (Observable) |

**Strategy:** Penalize both individual hazards and total hazards. We will also explore penalizing cumulative hazards and transition probabilities, as the latter is particularly relevant for panel data (path likelihoods). We are also interested in the general effect of penalizing observable quantities (sojourn times, transition densities) vs. latent parameters.

**Additional choices:**
- **Parametrization:** Log scale (standard, ensures positivity)
- **Penalty order:** $m=2$ (penalize curvature/"wiggliness")
- **Penalty construction:** Wood (2016) derivative-based algorithm

### Question 2: How to Set Tuning Parameters?

Many criteria exist for selecting smoothing parameters $\lambda$:

| Method | Assumption | Works for MSM? |
|:-------|:-----------|:---------------|
| **GCV/UBRE/GACV** | Independent subjects | ⚠️ Possible (with deviance residuals) |
| **REML** | Smooths as random effects | ✅ Yes — good benchmark |
| **NCV (PIJCV)** | Subject-level independence | ✅ Yes — natural choice for panel data |

**Recommendation:** Use NCV (specifically, leave-one-subject-out CV implemented via the Predictive Infinitesimal Jackknife).

**Why not just UBRE?** Machado et al. (2018) successfully used UBRE by constructing a working linear model from the likelihood derivatives. However, standard UBRE assumes that the data points (transitions) are independent. In panel data, observations within a subject are correlated. Treating them as independent often leads to **undersmoothing** (fitting the within-subject noise). NCV avoids this by leaving out entire subjects.

---

## Decision 1: What to Penalize

### The Penalized Log-Likelihood

The penalized log-likelihood has the form:

$$\ell_p(\theta; \boldsymbol{\lambda}) = \ell(\theta) - \frac{1}{2} \mathcal{P}(\theta; \boldsymbol{\lambda})$$

where $\mathcal{P}$ is a quadratic penalty. Our combined penalty structure is:

$$\mathcal{P}(\theta) = \underbrace{\sum_{(r,s)} \lambda_{rs} \theta_{rs}^T S_{rs} \theta_{rs}}_{\text{individual hazard penalties}} + \underbrace{\sum_r \mu_r \left(\sum_s \theta_{rs}\right)^T S_r \left(\sum_s \theta_{rs}\right)}_{\text{total hazard penalties}}$$

where:
- $\theta_{rs}$ = spline coefficients for log-hazard $\log h_{rs}(t)$
- $S_{rs}$ = penalty matrix for transition $(r,s)$
- $\lambda_{rs}$ = smoothing parameter for transition $(r,s)$
- $\mu_r$ = smoothing parameter for total hazard from state $r$

### Candidate Penalty Targets

#### 1. Individual Hazards $h_{rs}(t)$ — Standard Approach

Each transition has its own penalty:

$$\mathcal{P}_{\text{ind}} = \sum_{(r,s) \in \mathcal{A}} \lambda_{rs} \int_0^T \left[h_{rs}^{(m)}(t)\right]^2 dt$$

**Pros:** Simple, well-understood, standard in the literature

**Cons:** Total hazard can be wiggly even if individual hazards are smooth

#### 2. Total Hazard $H_r(t) = \sum_s h_{rs}(t)$ — Our Addition

The total hazard governs the overall rate of leaving state $r$:

$$\mathcal{P}_{\text{total}} = \sum_{r} \mu_r \int_0^T \left[\sum_s h_{rs}^{(m)}(t)\right]^2 dt$$

**Pros:** 
- Ensures survival function $S_r(t) = \exp(-\int_0^t H_r(u) du)$ is smooth
- Important for competing risks where we care about overall event rate

**Cons:**
- Introduces coupling between transitions (not separable)
- More complex Hessian structure

#### 3. Cumulative Hazard $\Lambda_{rs}(t)$ — Explore

While penalizing the $m$-th derivative of the cumulative hazard is mathematically equivalent to penalizing the $(m-1)$-th derivative of the hazard, the numerical properties and interpretation may differ. We will explore this as an alternative target.

#### 4. Transition Probabilities $P_{rs}(t)$ — Explore

For panel data, the likelihood is constructed directly from transition probabilities $P(s(t_{j+1}) | s(t_j))$, making this a natural target for smoothness.

**Pros:**
- Directly smooths the quantity of interest for panel data
- Aligns with the structure of the path likelihood

**Cons:**
- Non-quadratic in parameters (requires iterative approximation)
- Computationally expensive (requires matrix exponentials and derivatives)

**Feasibility:** Eletti et al. (2024) provide the analytical derivatives for Markov models. For semi-Markov models, these quantities are also computable (e.g., via numerical integration or simulation), allowing us to explore this direction for all model types.

#### 5. Sojourn Time Distributions — Explore

The sojourn time in state $r$ has survival function $S_r(t) = \exp(-\Lambda_r(t))$. Penalizing the roughness of the sojourn time distribution (e.g., its density $f_r(t) = H_r(t)S_r(t)$) targets an **observable quantity**—the time spent in a state—rather than the latent hazard rate.

**Hypothesis:** Penalizing observables might lead to better behaved predictions in data-sparse regions compared to penalizing latent parameters.

#### 6. Transition Densities — Explore

The density of observing a transition from $r \to s$ at time $t$ is $f_{rs}(t) = h_{rs}(t) S_r(t)$. This represents the distribution of actual event times.

**Pros:**
- Directly smooths the distribution of observed events
- May prevent artifacts where hazard spikes in regions with low survival probability (few people at risk)

**Cons:**
- Highly non-linear in parameters
- Depends on total hazard (coupling)

### Philosophy: Observables vs. Parameters

A key open question is the effect of penalizing **observable quantities** (transition probabilities, sojourn times, event densities) versus **model parameters** (hazards, log-hazards).

- **Parameter Penalties (Standard):** Smooth the underlying generator.
  - *Pros:* Computationally simpler (often quadratic), ensures the "engine" of the model is well-behaved.
  - *Cons:* Can allow wild behavior in observables if the transformation is non-linear (e.g., small hazard wiggles can be magnified or suppressed in $P(t)$).

- **Observable Penalties:** Smooth the resulting data distribution.
  - *Pros:* Aligns smoothness with what we see and measure. Prevents "invisible wiggliness" (e.g., wiggly hazards in regions where no one survives).
  - *Cons:* Computationally expensive (non-quadratic, dense Hessians), complex dependencies.

We intend to explore whether shifting the penalty target to observables yields better predictive performance, particularly for panel data.

### The Coupled Hessian Structure

With total hazard penalties, the penalty Hessian has off-diagonal blocks:

$$\frac{\partial^2 \mathcal{P}}{\partial \theta_{rs} \partial \theta_{rs}} = 2\lambda_{rs} S_{rs} + 2\mu_r S_r$$

$$\frac{\partial^2 \mathcal{P}}{\partial \theta_{rs} \partial \theta_{rs'}} = 2\mu_r S_r \quad \text{(transitions from same state)}$$

$$\frac{\partial^2 \mathcal{P}}{\partial \theta_{rs} \partial \theta_{r's'}} = 0 \quad \text{(transitions from different states)}$$

**Key insight:** Total hazard penalties couple transitions from the same origin state.

### Penalty Matrix Construction: Wood (2016) Algorithm

For B-splines of order $m_1$ with penalty on the $m_2$-th derivative:

**Algorithm:**
1. Set $p = m_1 - m_2$
2. For each knot interval, generate $p+1$ Gauss-Legendre points
3. Compute $G$: maps coefficients to derivative values
4. Compute $W$: integration weights (adapts to knot spacing)
5. Return $S = G^T W G$

**Properties:**
- $S$ is banded with $2(m_1 - 1) + 1$ diagonals
- Adapts automatically to uneven knot spacing
- Cost: $O(bk)$ where $b$ = bandwidth, $k$ = basis size

### The Parametrization Question

| Option | Model | Penalty | Verdict |
|:-------|:------|:--------|:--------|
| **A: Natural scale** | $h = \sum_k \beta_k B_k$ | $\lambda \beta^T S \beta$ | ❌ Positivity issues |
| **B: Log scale** | $\log h = \sum_k \theta_k B_k$ | $\lambda \theta^T S \theta$ | ✅ Standard, use this |
| **C: Hybrid** | $h = \exp(\sum_k \theta_k B_k)$ | $\lambda \int [h^{(m)}]^2$ | ⏳ Future extension |

**Recommendation:** Option B (log scale). Standard, ensures positivity, keeps penalty quadratic.

### Summary: Penalty Design Decisions

| Decision | Recommendation | Rationale |
|:---------|:---------------|:----------|
| Individual hazards | ✅ Yes | Standard, necessary |
| Total hazards | ✅ Yes | Important for competing risks |
| Cumulative hazards | 🔍 Explore | Alternative to hazard penalty |
| Transition probabilities | 🔍 Explore | Natural for panel data |
| Sojourn/Transition densities | 🔍 Explore | Smooth observables vs parameters |
| Parametrization | Log scale | Positivity, quadratic penalty |
| Penalty order | $m=2$ | Penalize curvature |

---

## Decision 2: How to Set Tuning Parameters

### The Nested Optimization Problem

Given the penalty structure, we need to find optimal $\boldsymbol{\lambda} = (\lambda_{rs}, \mu_r)$:

$$\min_{\boldsymbol{\lambda}} \text{Criterion}(\boldsymbol{\lambda}) \quad \text{where} \quad \hat{\theta}(\boldsymbol{\lambda}) = \arg\max_\theta \ell_p(\theta; \boldsymbol{\lambda})$$

**Inner optimization:** Newton/trust-region for $\theta$ given $\boldsymbol{\lambda}$

**Outer optimization:** BFGS or similar for $\boldsymbol{\lambda}$

### Why Classical CV Methods Are Suboptimal for MSM

**GCV, UBRE, GACV** are derived under the assumption of **independent subjects** (or independent sampling units).

**Machado et al. (2018)** applied UBRE to multistate models by:
1.  Approximating the likelihood with a penalized least squares problem (IRLS).
2.  Defining a "working response" vector $z$ and "hat matrix" $A$.
3.  Applying the standard UBRE formula.

**The Issue (Wood 2024):**
In panel data, observations from the same subject are **correlated**.
- Standard implementations often treat every row (transition) as independent, which leads to **undersmoothing** (fitting the within-subject noise).
- However, these methods **can** be adapted if we treat the **subject** as the independent unit.
- A key challenge is defining "residuals" for MSMs, which lack the clear $y - \hat{y}$ structure of GLMs.
- **Solution:** We can use **deviance residuals** (aggregated at the subject level) as a proxy for the squared error term in these criteria. This allows GCV/UBRE/GACV to be used, provided the independence assumption is applied at the subject level.

**NCV Solution:**
NCV explicitly handles this by leaving out **independent units** (subjects) rather than single observations.

### NCV: The Natural Choice for Multistate Models

Leave out an entire **neighbourhood** (= subject) instead of single observations:

$$\text{NCV}(\boldsymbol{\lambda}) = \sum_k D\left(y_k, \hat{\theta}^{-k}(\boldsymbol{\lambda})\right)$$

where $\hat{\theta}^{-k}$ is the estimate with subject $k$ removed.

**Why this is right:**
- Matches how likelihoods factorize (at subject level)
- Natural generalization of CV for panel data
- Not an approximation—it's the correct CV criterion

### Efficient Computation: PIJCV

Exact NCV requires $m$ refits (one per subject). Wood's key contribution: approximate via a single Newton step.

$$\hat{\theta}^{-k} \approx \hat{\theta} + H_{\lambda,k}^{-1} g_k$$

where:
- $H_{\lambda,k} = H_\lambda - H_k$ (downdated Hessian)
- $g_k$ = gradient contribution from subject $k$

This is the **Predictive Infinitesimal Jackknife Cross-Validation (PIJCV)**.

| Aspect | Standard IJ | PIJCV |
|:-------|:------------|:------|
| Leave-out unit | Observation | Neighbourhood (subject) |
| Hessian | Full $H$ | Downdated $H_{\lambda,k}$ |
| Accuracy | $O(n^{-1})$ | $O(n^{-2})$ |

**Cost:** $O(mp^2)$ via Cholesky downdating (already implemented in `NCVState`).

### REML as Alternative

REML treats smoothing parameters as variance components:

$$\ell_{REML}(\lambda) = \ell_p(\hat{\theta}) - \frac{1}{2}\log|H_p| + \frac{1}{2}\log|S_\lambda| + \text{const}$$

**Pros:** More stable optimization landscape, widely used

**Cons:** Different interpretation (likelihood-based vs prediction-based)

Implement as secondary method for comparison.

### How Many Smoothing Parameters?

| Option | # Parameters | Use When |
|:-------|:-------------|:---------|
| Global $\lambda$ | 1 | Debugging, simple models |
| Per-transition $\lambda_{rs}$ | $|\mathcal{A}|$ | Default |
| Per-origin-state $\lambda_r$ | # states | Many transitions |
| + Total hazard $\mu_r$ | + # competing risk states | Competing risks |

**Tradeoffs:**
- Fewer parameters → more stable, faster
- More parameters → more flexible, risk of overfitting $\lambda$

**Strategy:** Start simple (global), add complexity as needed.

### Comparison of Smoothing Selection Methods

| Method | Scale Known? | Assumption | Works for MSM? |
|:-------|:-------------|:-----------|:---------------|
| **GCV** | No | Independent subjects | ⚠️ Possible (with deviance residuals) |
| **UBRE** | Yes | Independent subjects | ⚠️ Possible (with deviance residuals) |
| **GACV** | No | Independent subjects | ⚠️ Possible (with deviance residuals) |
| **REML** | No | Random effects view | ✅ |
| **NCV (PIJCV)** | Either | Subject-level factorization | ✅ |

### Summary: Smoothing Parameter Selection

| Decision | Recommendation | Rationale |
|:---------|:---------------|:----------|
| Primary method | NCV (PIJCV) | Natural for subject-level likelihoods |
| Secondary method | REML | Good benchmark |
| Reporting | AIC/BIC + edf | Model comparison |
| Default structure | Per-transition | Balance flexibility/stability |

---

## The Wood-Marra Computational Framework

The methods in Eletti (2024) build on a computational framework developed by Marra and Wood:

### Key Papers

1. **Marra & Radice (2020)** — Trust region + automatic smoothing selection
2. **Marra & Wood (2012)** — Coverage properties of Bayesian CIs
3. **Marra et al. (2017)** — Full gradient/Hessian approach

### The Connection

Both REML (Marra) and NCV (Wood) require:
- **Exact analytical Hessian** (not approximations)
- **Efficient Cholesky factorization**
- **Per-subject gradient/Hessian contributions**

**Key insight:** The same infrastructure supports both methods. We build once, get both.

### Implications for Implementation

1. **Follow Marra & Radice (2020)** for optimization:
   - Trust region (not line search)
   - Analytical Hessian via Kosorok & Chao (1996)

2. **Use NCV instead of REML** as primary:
   - Same computational infrastructure
   - More robust for panel data

3. **Eletti (2024) is our template**, with NCV replacing REML.

---

## Handling Uneven Knot Spacing

### Univariate Baseline Hazards (Current Work)

**Quantile-based knot placement:** Already implemented in `place_interior_knots()`.

$$x_j = F^{-1}\left(\frac{j}{n_k + 1}\right)$$

Wood's algorithm adapts automatically via the weight matrix:
$$W = \bigoplus_{q=1}^{n_k} \frac{h_q}{2} \tilde{W}$$

where $h_q$ is knot interval width.

### Tensor Products (Future Work)

When data don't fill the domain (e.g., age × time with triangular support):

1. Identify unsupported basis functions
2. Drop corresponding rows from penalty matrix $\tilde{D}_j$
3. Constrain unsupported coefficients to zero

This is needed for time-varying covariate effects—future work.

---

## Implementation Roadmap

### Phase 1: Penalty Matrix Construction
- [ ] `compute_derivative_penalty_matrix(basis, order)` — Wood's algorithm
- [ ] Test banding structure
- [ ] Verify against known results

### Phase 2: Update Hazard Structures  
- [ ] Add `penalty_matrix` to `RuntimeSplineHazard`
- [ ] Add `smoothing_param` field
- [ ] Add `total_hazard_group_id` for coupling

### Phase 3: Penalized Likelihood
- [ ] `penalized_loglik` wrapper
- [ ] Individual + total hazard penalties
- [ ] ForwardDiff compatibility

### Phase 4: Gradient and Hessian
- [ ] Penalty gradient
- [ ] Penalty Hessian (with coupling)
- [ ] Integration with optimizer

### Phase 5: Smoothing Parameter Selection
- [ ] NCV via `NCVState`
- [ ] REML (secondary)
- [ ] AIC/BIC reporting

### Phase 6: Testing
- [ ] Unit tests for penalty matrices
- [ ] Integration tests for fitting
- [ ] Simulation studies

---

## Open Questions

1. **Total vs individual penalty strength:** How to set default ratio $\mu_r / \lambda_{rs}$?

2. **Shared smoothing parameters:** Should transitions from same state share $\lambda$?

3. **Initialization:** Good starting values for $\lambda$?

4. **Computational cost:** How much overhead does NCV add vs REML?

---

## Appendix A: Detailed Paper Summaries

### A.1 Machado, van den Hout & Marra (2018)
**"Penalised maximum likelihood estimation in multistate models for interval-censored data"**

**Context:** Frequentist penalized maximum likelihood for multistate models with interval-censored transition times.

#### Hazard Representation
The baseline hazard for transition $r \to s$ is modeled using B-splines on the log scale:

$$q_{rs,0}(t) = \exp\left(\sum_{k=1}^{K_{rs}} \alpha_{rs,k} B_k(t)\right)$$

where $B_k(t)$ are cubic regression spline basis functions and $\alpha_{rs,k}$ are coefficients to be estimated.

#### Penalty Structure
Each transition hazard has its own quadratic penalty:

$$\lambda_{rs} \alpha_{rs}^T S_{rs} \alpha_{rs}$$

where $S_{rs}$ is the penalty matrix (integrated squared second derivative). The full penalty matrix $S_\lambda$ is **block-diagonal** with blocks $\lambda_{rs} S_{rs}$ for each transition and zeros elsewhere.

#### Penalized Log-Likelihood

$$\ell_p(\theta) = \ell(\theta) - \frac{1}{2} \theta^T S_\lambda \theta$$

#### Smoothing Parameter Selection
They use **UBRE** (Unbiased Risk Estimator):

$$V(\lambda) = \|z - A_\lambda z\|^2 - c + 2\text{tr}(A_\lambda)$$

where $A_\lambda = \sqrt{I}(I + S_\lambda)^{-1}\sqrt{I}$ is the "hat" matrix. This is minimized using a general-purpose optimizer.

#### Estimation Algorithm (Two-Step Iteration)
1. **Step 1:** For fixed $\lambda$, update $\theta$ using Newton-type scoring:
   $$\theta^{[a+1]} = (I^{[a]} + S_{\hat{\lambda}})^{-1} \sqrt{I^{[a]}} z^{[a]}$$
   where $z^{[a]} = \sqrt{I^{[a]}} \theta^{[a]} + \epsilon^{[a]}$

2. **Step 2:** Given $\theta$, update $\lambda$ by minimizing UBRE.

3. Iterate until convergence: $\max|\theta^{[a+1]} - \theta^{[a]}| < \delta$

#### Fisher Information Approximation
They use an **approximation** to the Fisher information matrix that only requires **first-order derivatives** of the log-likelihood. This is because computing second derivatives of transition probabilities is intractable via their approach.

#### Confidence Intervals
Use the Bayesian posterior approximation $\hat{\theta} \sim N(\theta, V_\theta)$ where $V_\theta$ is the inverse of the penalized information matrix. For nonlinear functions (hazards, transition probabilities):
1. Draw $n$ samples from $N(\hat{\theta}, V_\theta)$
2. Evaluate the function at each sample
3. Take quantiles for confidence intervals

#### Limitations
- Uses approximate Fisher information (1st derivatives only)
- UBRE assumes independent residuals → undersmoothing with correlated data
- No support for tensor products or time-varying effects

---

### A.2 Eletti, Marra & Radice (2024)
**"Spline-Based Multi-State Models for Analyzing Disease Progression"** (R package `flexmsm`)

**Context:** Frequentist penalized maximum likelihood with **exact analytical Hessian** for general multistate models.

#### Hazard Representation
Additive predictor framework:

$$q^{(rr')}(t_\iota) = \exp\left(\eta_\iota^{(rr')}(t_\iota, x_\iota; \beta^{(rr')})\right)$$

where the additive predictor can include:
- Overall intercept $\beta_0^{(rr')}$
- Smooth functions of time $s_k(t)$
- Parametric covariate effects
- **Tensor product interactions** (e.g., time-varying effects $s_k(\text{age}, t)$)

Each smooth term is represented as:
$$s_k^{(rr')}(\tilde{x}_{k\iota}) = b_k^{(rr')}(\tilde{x}_{k\iota})^T \beta_k^{(rr')}$$

#### Penalty Structure
Each smooth term has a quadratic penalty:
$$\lambda_k^{(rr')} \beta_k^{(rr')T} D_k^{(rr')} \beta_k^{(rr')}$$

The overall penalty matrix is block-diagonal:
$$S_\lambda = \text{diag}\{S_{\lambda^{(rr')}} : (r, r') \in A\}$$

where each block is:
$$S_{\lambda^{(rr')}} = \text{diag}(0, \lambda_1^{(rr')} D_1^{(rr')}, \ldots, \lambda_{K^{(rr')}}^{(rr')} D_{K^{(rr')}}^{(rr')})$$

#### Key Innovation: Analytical Hessian
Unlike Machado, they derive and implement the **exact analytical Hessian** using results from Kosorok & Chao (1996).

For transition probability matrix via eigendecomposition $Q = A\Gamma A^{-1}$:

$$P(\delta t) = A \text{diag}(\exp(\gamma_1 \delta t), \ldots, \exp(\gamma_C \delta t)) A^{-1}$$

$$\frac{\partial P(\delta t)}{\partial \theta_w} = A U_w A^{-1}$$

$$\frac{\partial^2 P(\delta t)}{\partial \theta_w \partial \theta_{w'}} = A(\check{U}_{ww'} + \dot{U}_{ww'} + \dot{U}_{w'w}) A^{-1}$$

where:
- $U_w = G^{(w)} \circ E$ with $G^{(w)} = A^{-1} \frac{\partial Q}{\partial \theta_w} A$
- $E[l,m] = \frac{\exp(\gamma_l \delta t) - \exp(\gamma_m \delta t)}{\gamma_l - \gamma_m}$ when $\gamma_l \neq \gamma_m$
- $E[l,m] = \delta t \exp(\gamma_l \delta t)$ when $\gamma_l = \gamma_m$

The second derivative terms $\check{U}_{ww'}$ and $\dot{U}_{ww'}$ have more complex expressions involving products of first-derivative quantities.

#### Estimation
Uses a **trust region algorithm** (not line search) which "significantly outperforms its line search counterparts" when given analytical first and second derivatives.

#### Smoothing Parameter Selection
Uses REML-like methods following Marra & Radice (2020), requiring the analytical observed information matrix.

#### Confidence Intervals
Same Bayesian approximation: $\theta \sim N(\hat{\theta}, V_\theta)$ where $V_\theta = -H_p(\hat{\theta})^{-1}$.

This gives "close to across-the-function frequentist coverage probabilities because it accounts for both sampling variability and smoothing bias."

#### Model Selection
- AIC: $-2\ell(\theta) + 2 \cdot \text{edf}$
- BIC: $-2\ell(\theta) + \log(\check{n}) \cdot \text{edf}$
- Effective degrees of freedom: $\text{edf} = \text{tr}((-H(\theta))(-H_p(\theta))^{-1}(-H(\theta))^{1/2})$

#### Advantages over Machado
- Exact Hessian enables stable convergence for complex models
- Supports tensor products for time-varying effects
- Trust region optimization is more robust
- Can fit models that fail to converge with approximate methods

---

### A.3 Wood (2016)
**"P-splines with derivative based penalties and tensor product smoothing of unevenly distributed data"**

**Context:** Efficient computation of derivative-based penalties for B-splines, with application to tensor product smoothing.

#### Motivation
P-splines (Eilers & Marx 1996) use discrete difference penalties on coefficients. However:
1. B-spline bases arise from variational problems with **derivative-based** penalties
2. Discrete penalties are "less interpretable in terms of function shape"
3. Discrete penalties only approximate derivative penalties "in the limit of large basis size" — but P-splines intentionally use small bases

Wood shows that derivative-based penalties can be computed with the **same efficiency** as P-splines.

#### Derivative-Based Penalty
For a spline $f(x) = \sum_{j=1}^k \beta_j B_{m_1,j}(x)$ with penalty based on the $m_2$-th derivative:

$$J = \int_a^b [f^{(m_2)}(x)]^2 dx = \beta^T S \beta$$

where $S$ is a **banded** (sparse) penalty matrix.

#### Algorithm for Computing $S$

Given B-splines of order $m_1$ with penalty order $m_2$ (where $m_2 \leq m_1$):

Let $p = m_1 - m_2$ (order of piecewise polynomial for the derivative).

1. **Generate evaluation points:** For each knot interval $[x_j, x_{j+1}]$, generate $p+1$ evenly spaced points.

2. **Compute derivative matrix $G$:** Maps spline coefficients $\beta$ to the $m_2$-th derivative at evaluation points. (Use standard B-spline derivative recursions.)

3. **Compute weight matrix $W$:**
   - If $p = 0$: $W = \text{diag}(h)$ where $h_j = x_{j+1} - x_j$
   - If $p > 0$: $\tilde{W} = P^{-T} H P^{-1}$ where:
     - $P_{ij} = (-1 + 2(i-1)/p)^j$
     - $H_{ij} = (1 + (-1)^{i+j-2})/(i+j-1)$
   - Then $W = \bigoplus_q (h_q/2) \tilde{W}$ (block diagonal)

4. **Penalty matrix:** $S = G^T W G$ (banded, sparse)

5. **Optional square root:** Compute $R^T R = W$ (Cholesky), then $D = RG$, so $S = D^T D$.

#### Key Properties
- $S$ is **banded** with $2(m_1 - 1) + 1$ non-zero diagonals
- **Mix-and-match:** Basis order $m_1$ and penalty order $m_2$ are independent
- Cost: $O(bk)$ where $b$ = bandwidth, $k$ = basis dimension

#### Tensor Product Smoothing

For a 3D smooth $f(z_1, z_2, z_3)$:

$$f(z) = \sum_{ijl} \beta_{ijl} B_{1i}(z_1) B_{2j}(z_2) B_{3l}(z_3)$$

With coefficients in column-major order, the three penalties are $\beta^T S_j \beta$ where:

$$S_j = \tilde{D}_j^T \tilde{D}_j$$

with:
- $\tilde{D}_1 = D_1 \otimes I_{k_2} \otimes I_{k_3}$
- $\tilde{D}_2 = I_{k_1} \otimes D_2 \otimes I_{k_3}$
- $\tilde{D}_3 = I_{k_1} \otimes I_{k_2} \otimes D_3$

#### Sparse Handling for Partial Domains

When data only covers part of the tensor product domain:
1. Identify coefficients $\beta_\iota$ where the basis function is zero at all data points
2. Drop rows $\kappa$ from $\tilde{D}_j$ where $\tilde{D}_{j,\kappa\iota} \neq 0$

This preserves the penalty structure while reducing dimensionality.

#### Handling Unevenly Spaced Knots

The algorithm naturally handles non-uniform knot spacing. The weight matrix $W$ accounts for variable knot intervals:

For penalty order $m_2 = m_1$ (so $p = 0$):
$$W = \text{diag}(h_1, h_2, \ldots, h_{n_k})$$

where $h_q = x_{q+1} - x_q$ is the width of knot interval $q$.

For $p > 0$, the block-diagonal structure becomes:
$$W = \bigoplus_{q=1}^{n_k} \frac{h_q}{2} \tilde{W}$$

**Key insight:** The penalty matrix $S = G^T W G$ automatically adapts to non-uniform knot spacing.

---

### A.4 Wood (2024)
**"On Neighbourhood Cross Validation"**

**Context:** Robust smoothing parameter selection for penalized regression when data has short-range autocorrelation.

#### The Problem
Standard CV/GCV/REML/UBRE assume that "leave-one-out" residuals are approximately independent. In longitudinal/clustered data:
- Observations within a subject are correlated
- Leaving out one observation still "leaks" information from correlated neighbors
- Result: **Undersmoothing** (fitting noise as signal)

#### NCV Solution
Leave out an entire **neighbourhood** (e.g., all observations from one subject) instead of single observations.

The NCV criterion:
$$V = \sum_k \sum_{i \in \delta(k)} D(y_i, \hat{\theta}_i^{-\alpha(k)})$$

where:
- $\delta(k)$ is neighbourhood $k$
- $\alpha(k)$ is the set of indices for neighbourhood $k$
- $\hat{\theta}^{-\alpha(k)}$ is the estimate with neighbourhood $k$ removed

#### Efficient Computation via PIJCV

Exact NCV would require refitting the model $m$ times (once per neighbourhood). Wood's key contribution is an efficient **approximation** using a single Newton step from the full-data estimate.

The "leave-out" parameter perturbation is approximated as:

$$\hat{\theta}^{-\alpha(k)} \approx \hat{\theta} + \Delta_{-\alpha(k)}$$

where:

$$\Delta_{-\alpha(k)} = H_{\lambda,\alpha(k)}^{-1} g_{\alpha(k)}$$

with:
- $H_{\lambda,\alpha(k)} = H_\lambda - H_{\alpha(k),\alpha(k)}$ (penalized Hessian minus neighbourhood contribution)
- $g_{\alpha(k)}$ is the gradient contribution from neighbourhood $k$

This is the **Predictive Infinitesimal Jackknife Cross-Validation (PIJCV)**.

#### Relationship to Standard Infinitesimal Jackknife

| Aspect | Standard IJ | PIJCV |
|:-------|:------------|:------|
| Leave-out unit | Single observation | Neighbourhood (subject) |
| Approximation | $\hat{\theta}^{[-i]} \approx \hat{\theta} - H^{-1} \psi_i$ | $\hat{\theta}^{-\alpha(k)} \approx \hat{\theta} + H_{\lambda,\alpha(k)}^{-1} g_{\alpha(k)}$ |
| Hessian used | Full $H$ | **Downdated** $H_{\lambda,\alpha(k)}$ |
| Accuracy | $O(n^{-1})$ | $O(n^{-2})$ |

**Key refinement:** The downdated Hessian is closer to the true leave-out Hessian, improving approximation accuracy.

Using **Cholesky downdating**, PIJCV can be computed in $O(mp^2)$ time rather than $O(m)$ refits.

#### Requirements for Implementation
1. **Penalized Hessian $H_\lambda$:** Must be exact (not approximate)
2. **Per-neighbourhood gradients $g_{\alpha(k)}$:** Must be extractable
3. **Per-neighbourhood Hessian contributions $H_{\alpha(k),\alpha(k)}$:** For the downdate

#### Degeneracy Protections
- Cholesky downdate with indefiniteness detection
- Woodbury identity fallback for indefinite cases
- Quadratic approximation (QNCV) for finite deviance robustness

---

## Appendix B: Detailed Explanation of Classical CV Methods

### The Starting Point: Leave-One-Out Cross-Validation

The ideal criterion would be true leave-one-out CV:
$$\text{CV} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{f}^{[-i]}(x_i))^2$$

where $\hat{f}^{[-i]}$ is the fit with observation $i$ removed. This requires $n$ refits—expensive. GCV, UBRE, and GACV are efficient approximations.

### UBRE (Unbiased Risk Estimator)

**When scale $\sigma^2$ is known**, there's a shortcut. The expected prediction error is:

$$\text{EPE} = E\|y - \hat{f}\|^2 = \sigma^2(n - 2\cdot\text{tr}(A) + \text{tr}(A^TA))$$

where $A$ is the "hat matrix" ($\hat{y} = Ay$). This leads to:

$$\text{UBRE} = \frac{1}{n}\|y - \hat{f}\|^2 - \sigma^2 + \frac{2\sigma^2}{n}\text{tr}(A)$$

This is algebraically equivalent to **Mallows' $C_p$** and to **AIC** (up to constants):
$$\text{UBRE} = \frac{\text{RSS}}{n} + \frac{2\sigma^2 \cdot \text{edf}}{n} - \sigma^2$$

**Key insight**: The $2\cdot\text{edf}$ term corrects for the optimism of training error.

### GCV (Generalized Cross-Validation)

**When scale $\sigma^2$ is unknown**, UBRE doesn't work. Craven & Wahba (1979) proposed:

$$\text{GCV} = \frac{n \cdot \text{RSS}}{(n - \text{tr}(A))^2} = \frac{\|y - \hat{f}\|^2 / n}{(1 - \text{edf}/n)^2}$$

**Where does this come from?**

1. True leave-one-out CV can be computed without refitting using:
   $$\text{CV} = \frac{1}{n}\sum_{i=1}^n \left(\frac{y_i - \hat{f}(x_i)}{1 - A_{ii}}\right)^2$$
   
2. GCV replaces each $A_{ii}$ with the average $\text{tr}(A)/n$:
   $$\text{GCV} = \frac{1}{n}\sum_{i=1}^n \left(\frac{y_i - \hat{f}(x_i)}{1 - \text{tr}(A)/n}\right)^2$$

**Why "generalized"?** The averaging makes GCV invariant to rotations of the coordinate system.

### GACV (Generalized Approximate CV)

For **non-Gaussian likelihoods** (Poisson, binomial, etc.), there's no simple hat matrix. GACV extends GCV:

$$\text{GACV} = \frac{\sum_i D(y_i, \hat{\mu}_i)}{n - \text{tr}(\tilde{A})}$$

where $D$ is the deviance contribution and $\tilde{A}$ is an "influence matrix."

### Summary

| Method | Scale | Distribution | Formula |
|:-------|:------|:-------------|:--------|
| **UBRE** | Known | Gaussian | $\text{RSS}/n + 2\sigma^2\cdot\text{edf}/n$ |
| **GCV** | Unknown | Gaussian | $\text{RSS}/n \cdot (1 - \text{edf}/n)^{-2}$ |
| **GACV** | Unknown | Any (GLM) | Deviance/(n - edf) |

### Why These Are Suboptimal for Multistate Models

All three assume:
1. **Observation-level independence**: Each $y_i$ contributes independently.
2. **Hat matrix exists**: $\hat{y} = Ay$.

In multistate models with panel data:
- **Correlation:** Observations within a subject are correlated.
- **Undersmoothing:** Treating correlated observations as independent leads to undersmoothing (fitting noise).

Machado et al. (2018) showed that you *can* mechanically apply UBRE by constructing a working linear model. However, Wood (2024) argues that for correlated data, criteria that respect the correlation structure (like NCV) are theoretically superior.

---

## Appendix C: Comparison Table

| Aspect | Machado (2018) | Eletti (2024) | Wood (2016) | Wood (2024) NCV |
|:-------|:---------------|:--------------|:------------|:----------------|
| **Focus** | Multistate models | Multistate models | Penalty computation | Smoothing selection |
| **Penalty Type** | Unspecified | Flexible | **Derivative-based** | Any |
| **Penalty Matrix** | Block-diagonal | Block-diagonal | Banded/sparse | Any |
| **Tensor Products** | No | Yes | **Yes** | Compatible |
| **Hessian** | Approximate (1st order) | **Exact** (2nd order) | N/A | Requires exact |
| **Smoothing Selection** | UBRE | REML | N/A | **NCV** |
| **Autocorrelation Robust** | No | No | N/A | **Yes** |
| **Optimization** | Scoring | Trust region | N/A | N/A |
| **Key Marra Ref** | Marra et al. (2017) | Marra & Radice (2020) | N/A | Marra & Wood (2012) |
