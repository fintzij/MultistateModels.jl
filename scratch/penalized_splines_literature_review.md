# Penalized Splines for Multistate Models: Design Document

## Document Status

> **Version**: Final Design Document (Ready for Implementation)
> 
> **Companion Document**: See [penalized_splines_plan.md](penalized_splines_plan.md) for implementation roadmap.
> 
> **Consistency Check**: All major design decisions in this document are consistent with 
> the implementation plan. Key clarifications made during adversarial review:
> - The **likelihood-motivated penalty decomposition** (Section 2) grounds penalty structure
>   in the competing risks likelihood factorization
> - Total hazard penalty is motivated by the survival component of the likelihood
> - Deviation penalty provides indirect control of cause allocation smoothness
> - Varying coefficient models are marked as **FUTURE WORK**
> - REML is marked as **EXPLICITLY DEFERRED** (NCV is primary method)
> - Penalty info stored in `ConvergenceRecords`, not new struct field (R8-1)

## Executive Summary

This document guides the implementation of penalized splines in `MultistateModels.jl`. The scope has been expanded based on adversarial review to include:

1. **Baseline hazard smoothing**: Penalized splines for $\log h_{rs}(t)$
2. **Covariate splines**: Smooth nonlinear covariate effects via `s(covariate)` syntax
3. **Likelihood-motivated penalty decomposition**: Total hazard and deviation penalties grounded in the likelihood factorization
4. **Covariate tensor products**: Interactions between covariates and time-varying effects

We address three fundamental questions:

1. **What to penalize?** — Which functions should have smoothness penalties?
2. **How to set tuning parameters?** — How do we select smoothing parameters $\lambda$?
3. **How to structure penalties?** — How do we structure penalties across related transitions?

Our approach synthesizes methods from four key papers:
- **Machado et al. (2018)** and **Eletti et al. (2024)** for penalized multistate models
- **Wood (2016)** for penalty matrix construction and tensor products
- **Wood (2024)** for smoothing parameter selection via Neighbourhood Cross-Validation (NCV)

### Key Design Decisions (Summary)

| Decision | Choice | Rationale |
|:---------|:-------|:----------|
| What to penalize | Hazards $h_{rs}(t)$ on natural scale | Closed-form cumulative hazard; positivity via constraints |
| Penalty type | Derivative-based (Wood 2016) | Adapts to uneven knots |
| Penalty order | $m = 2$ (curvature), configurable | Standard; penalizes roughness |
| Smoothing selection | NCV via `PIJCVState` + criterion function | Accounts for within-subject correlation |
| Competing risks | Likelihood-motivated decomposition | Total hazard + deviation penalties |
| Error handling | `ArgumentError` for user input | Package convention |
| Penalty storage | `PenaltyConfig` struct | Keep hazard types immutable |
| Initial scope | Markov + exact data | MCEM compatibility deferred |

---

## Tensor Product API Design

### Three Distinct Use Cases

Tensor products serve different purposes depending on what dimensions are involved:

| Use Case | API Location | Description | λ parameters |
|:---------|:-------------|:------------|:-------------|
| **Shared-origin baselines** | Model-level kwarg | Jointly penalize competing risk baselines | 2 ($\lambda_t$, $\lambda_d$) |
| **Covariate × covariate** | Formula `te(x1, x2)` | Smooth interaction surface | 2 per term |
| **Covariate × time** | Formula `te(x, t)` | Time-varying covariate effect | 2 per term |

### 1. Shared-Origin Tensor Products (Baseline Hazards)

For competing risks from state $r$ to destinations $\{s_1, \ldots, s_D\}$:

```julia
# Model-level specification (NOT in hazard formulas)
model = multistatemodel(h12, h13, h14;
    data = data,
    shared_origin_tensor = [1]  # State 1's baselines share structure
)

# Explicit configuration
model = multistatemodel(...;
    shared_origin_tensor = Dict(
        1 => SharedOriginConfig(
            dest_penalty = :ridge,      # :ridge, :difference, or Matrix
            monotone_time = false,
            increasing = true
        )
    )
)
```

**Why model-level?** This restructures the baseline hazard parameters into a tensor, which is a structural model choice, not a hazard-level formula term.

### 2. Covariate × Covariate Interactions

Smooth interaction surface between two covariates:

```julia
# Full tensor product (includes main effects implicitly)
h12 = Hazard(@formula(0 ~ te(age, bmi)), :wei, 1, 2)

# Main effects + pure interaction (ANOVA decomposition)
h12 = Hazard(@formula(0 ~ s(age) + s(bmi) + ti(age, bmi)), :wei, 1, 2)
```

**Semantics:**
- `te(x1, x2)`: Full tensor product $f(x_1, x_2) = \sum_{ij} \beta_{ij} B_{1i}(x_1) B_{2j}(x_2)$
- `ti(x1, x2)`: Interaction only (constrained to sum to zero over margins)

### 3. Time-Varying Covariate Effects

Covariate effect that changes smoothly over time:

```julia
# Requires :sp family (time basis needed)
h12 = Hazard(@formula(0 ~ te(age, t)), :sp, 1, 2)

# With additional linear terms
h12 = Hazard(@formula(0 ~ te(age, t) + trt), :sp, 1, 2)
```

**Constraint:** `te(x, t)` requires `:sp` family because it needs a time spline basis.

### API Summary

| Syntax | Level | Family Constraint | Parameters |
|:-------|:------|:------------------|:-----------|
| `shared_origin_tensor=[r]` | Model | `:sp` required | $K_t \times D$ |
| `s(x)` | Formula | Any | $K_x$ |
| `te(x1, x2)` | Formula | Any | $K_1 \times K_2$ |
| `te(x, t)` | Formula | `:sp` only | $K_x \times K_t$ |
| `ti(x1, x2)` | Formula | Any | $(K_1-1) \times (K_2-1)$ |

---

## The Three Key Design Questions

### Question 1: What to Penalize?

We have flexibility in what quantities we penalize for smoothness:

| Target | Formula | Status |
|:-------|:--------|:-------|
| **Individual hazards** $h_{rs}(t)$ | $\lambda_{rs} \int [h_{rs}^{(m)}]^2 dt$ | ✅ Primary target |
| **Shared-origin tensor** | $\lambda_t (I_D \otimes S_t) + \lambda_d (S_d \otimes I)$ | ✅ Replaces total hazard penalty |
| **Covariate smooths** | $\lambda_j \gamma_j^T S_j \gamma_j$ | ✅ New feature |
| **Total hazard** $H_r(t)$ | $\mu_r \int [H_r^{(m)}]^2 dt$ | ❌ Superseded by tensor products |
| **Transition probabilities** | Non-quadratic | 🔍 Future exploration |

**Parametrization:** Log scale (standard, ensures positivity)
**Penalty order:** $m=2$ (penalize curvature), user-configurable
**Penalty construction:** Wood (2016) derivative-based algorithm

### Question 2: How to Set Tuning Parameters?

| Method | Assumption | Works for MSM? |
|:-------|:-----------|:---------------|
| **GCV/UBRE/GACV** | Independent subjects | ⚠️ Possible (with deviance residuals) |
| **REML** | Smooths as random effects | ✅ Yes — benchmark method |
| **NCV (PIJCV)** | Subject-level independence | ✅ Yes — primary method |

**Recommendation:** Use NCV (leave-one-subject-out CV via Predictive Infinitesimal Jackknife).
Existing infrastructure: `PIJCVState` in `src/output/variance.jl`.

### Question 3: How to Structure Penalties Across Transitions?

For competing risks from state $r$ to destinations $\{s_1, \ldots, s_D\}$:

**Original approach (superseded)**:
$$\mathcal{P}_{total} = \mu_r \left(\sum_s \theta_{rs}\right)^T S_r \left(\sum_s \theta_{rs}\right)$$

**New approach (tensor products)**:
$$\mathcal{P}_{tensor} = \lambda_t \boldsymbol{\beta}^T (I_D \otimes S_t) \boldsymbol{\beta} + \lambda_d \boldsymbol{\beta}^T (S_d \otimes I_{K_t}) \boldsymbol{\beta}$$

The tensor product approach:
- Decomposes curvature into mean and deviation components
- Allows separate control of total hazard wiggliness and relative shape divergence
- Is more flexible (configurable destination penalty)

---

## Decision 1: What to Penalize

### The Penalized Log-Likelihood

The penalized log-likelihood has the form:

$$\ell_p(\boldsymbol{\beta}; \lambda_H, \lambda_\pi) = \ell(\boldsymbol{\beta}) - \frac{1}{2} \mathcal{P}(\boldsymbol{\beta}; \lambda_H, \lambda_\pi)$$

where $\mathcal{P}$ is a quadratic penalty. Based on the **likelihood-motivated decomposition** (see Section 2), our penalty structure for competing risks is:

$$\mathcal{P}(\boldsymbol{\beta}) = \lambda_H \boldsymbol{\beta}^T (\mathbf{1}_D\mathbf{1}_D^T \otimes S) \boldsymbol{\beta} + \lambda_\pi \boldsymbol{\beta}^T (C_D \otimes S) \boldsymbol{\beta}$$

where:
- $\boldsymbol{\beta}$ = vectorized spline coefficients for all hazards from a shared origin
- $S$ = curvature penalty matrix for the shared B-spline basis
- $\lambda_H$ = smoothing parameter for total hazard (survival smoothness)
- $\lambda_\pi$ = smoothing parameter for deviation (cause allocation smoothness)
- $\mathbf{1}_D\mathbf{1}_D^T$ and $C_D = I_D - \frac{1}{D}\mathbf{1}_D\mathbf{1}_D^T$ are the Kronecker factors

### Candidate Penalty Targets

#### 1. Individual Hazards $h_{rs}(t)$ — Standard Approach

Each transition has its own penalty:

$$\mathcal{P}_{\text{ind}} = \sum_{(r,s) \in \mathcal{A}} \lambda_{rs} \int_0^T \left[h_{rs}^{(m)}(t)\right]^2 dt$$

**Pros:** Simple, well-understood, standard in the literature

**Cons:** Total hazard can be wiggly even if individual hazards are smooth

**New insight (Section 2)**: The independent penalty *implicitly* decomposes into:
$$\mathcal{P}_{\text{ind}} = \frac{1}{D}\mathcal{P}_H + \mathcal{P}_\pi$$

This means the standard approach already penalizes both survival and cause allocation smoothness, with relative weight $1/D$ on survival.

#### 2. Total Hazard $H_r(t) = \sum_s h_{rs}(t)$ — Likelihood-Motivated

> **✅ RECOMMENDED: Part of the likelihood-motivated decomposition**
> 
> The total hazard penalty is now motivated by the competing risks likelihood factorization.
> See Section 2 (Rigorous Mathematical Framework) for full details.

The total hazard governs the overall rate of leaving state $r$ and directly controls survival smoothness:

$$\mathcal{P}_H = \int_0^T (H''(t))^2 dt = \boldsymbol{\beta}^T (\mathbf{1}_D\mathbf{1}_D^T \otimes S) \boldsymbol{\beta}$$

**Pros:** 
- Ensures survival function $S_r(t) = \exp(-\int_0^t H_r(u) du)$ is smooth
- Directly motivated by the survival component of the likelihood
- Separable control of survival vs. cause allocation smoothness

**Implementation**: Combined with deviation penalty via two-parameter penalty:
$$\mathcal{P} = \lambda_H \mathcal{P}_H + \lambda_\pi \mathcal{P}_\pi$$

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

**Cost:** $O(mp^2)$ via Cholesky downdating (already implemented in `PIJCVState`).

**Implementation note**: The existing `PIJCVState` infrastructure computes LOO parameter perturbations $\hat{\theta}^{-k}$ but does NOT compute the NCV criterion itself. We must add `compute_ncv_criterion(state, model, data)` to evaluate the deviance at LOO parameters.

### REML as Alternative

REML treats smoothing parameters as variance components:

$$\ell_{REML}(\lambda) = \ell_p(\hat{\theta}) - \frac{1}{2}\log|H_p| + \frac{1}{2}\log|S_\lambda| + \text{const}$$

**Pros:** More stable optimization landscape, widely used

**Cons:** Different interpretation (likelihood-based vs prediction-based)

Implement as secondary method for comparison (may defer to reduce initial scope).

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

### Tensor Products for Partial Domains

When data don't fill the domain (e.g., age × time with triangular support):

1. Identify unsupported basis functions
2. Drop corresponding rows from penalty matrix $\tilde{D}_j$
3. Constrain unsupported coefficients to zero

---

## Extended Scope: Covariate Splines

### Motivation

Support smooth nonlinear covariate effects via `s(covariate)` syntax in hazard formulas:
```julia
h12 = Hazard(@formula(0 ~ s(age) + trt), :sp, 1, 2)
```

### Model Structure

For covariate $x$ with B-spline basis $\{B_{cj}(x)\}_{j=1}^{K_c}$:
$$\eta = \beta_0 + \sum_{j=1}^{K_c} \gamma_j B_{cj}(x) + \text{linear terms}$$

### Penalty

Each smooth term has its own penalty:
$$\mathcal{P}_{cov}(\gamma) = \sum_{j} \lambda_j \gamma_j^T S_j \gamma_j$$

### Identifiability

Smooth + intercept are non-identifiable. Resolve via **sum-to-zero constraint**:
$$\sum_i s(x_i) = 0$$

Implemented by centering the basis matrix: $\tilde{B} = B - \bar{B}$

### Centering and Back-Transformation

#### Basis Centering (REQUIRED)

**Centered basis**: $\tilde{B} = B - \bar{B}$ ensures $\sum_i f(x_i) = 0$.

**Relationship**:
- Centered: $f(x) = \tilde{B}(x)^T \tilde{\gamma}$
- Uncentered: $f(x) = B(x)^T \gamma - c$ where $c = \bar{B}^T \gamma$
- Coefficients are the same ($\gamma = \tilde{\gamma}$), but intercept absorbs $c$

**For prediction at new $x_{new}$**: Must store $\bar{B}$ to compute $\tilde{B}(x_{new}) = B(x_{new}) - \bar{B}$.

#### Covariate Scaling: NOT RECOMMENDED

After reviewing established packages:
- **MixedModels.jl**: Does NOT scale or center fixed effects covariates
- **survival (R)**: Does NOT scale design matrix; centers linear predictor post-hoc only

**Decision**: Do NOT scale covariates.
- Modern optimizers handle scale differences via Hessian adaptation
- Eliminates complexity of storing/applying scale factors
- Users can pre-transform if numerical issues arise
- Time is never scaled (consistency with existing APIs)

#### Storage for Prediction

```julia
struct SmoothTermInfo
    term::SmoothTerm
    basis::BSplineBasis
    centering_vec::Vector{Float64}  # B̄ for basis centering
    x_range::Tuple{Float64, Float64}  # Data range for plotting
end
```

---

## StatsModels.jl Formula Integration

### The Shadowed @formula Macro Approach

**Critical limitation**: StatsModels.jl's `@formula` macro rejects keyword arguments at parse time. It checks the AST for `:kw` and `:parameters` nodes and throws `ArgumentError` before any dispatch occurs. This means `@formula(0 ~ s(age; k=5))` fails immediately.

**Solution**: MultistateModels exports a **shadowed `@formula` macro** that preprocesses smooth term syntax before delegating to `StatsModels.@formula`. This is implemented in `src/construction/formula.jl`.

```julia
# File: src/construction/formula.jl

import StatsModels

# Declare smooth term functions (must exist for FunctionTerm{typeof(s)})
function s end
function te end  
function ti end

"""
Transform smooth terms with kwargs to positional args before StatsModels parsing.
s(age; k=5, penalty_order=3) → s(age, 5, 3)
te(age, bmi; k=(8, 6)) → te(age, bmi, 8, 6, 2)
"""
function transform_smooth_terms(ex)
    if ex isa Expr
        if ex.head == :call && length(ex.args) >= 1 && ex.args[1] in (:s, :te, :ti)
            return transform_smooth_call(ex)
        else
            return Expr(ex.head, [transform_smooth_terms(a) for a in ex.args]...)
        end
    else
        return ex
    end
end

function transform_smooth_call(ex)
    func = ex.args[1]
    positional = Symbol[]
    k = func == :s ? 10 : (10, 10)  # Default k
    penalty_order = 2               # Default penalty order
    
    for i in 2:length(ex.args)
        arg = ex.args[i]
        if arg isa Expr && arg.head == :parameters
            # Extract kwargs from parameters block
            for kwexpr in arg.args
                if kwexpr isa Expr && kwexpr.head == :kw
                    name, val = kwexpr.args[1], kwexpr.args[2]
                    if name == :k
                        k = val
                    elseif name == :penalty_order
                        penalty_order = val
                    end
                end
            end
        elseif arg isa Symbol
            push!(positional, arg)
        end
    end
    
    # Build positional-only call
    if func == :s
        return Expr(:call, :s, positional[1], k, penalty_order)
    else  # te or ti
        k1, k2 = k isa Tuple ? k : (k, k)
        return Expr(:call, func, positional[1], positional[2], k1, k2, penalty_order)
    end
end

"""
    @formula(ex)

Extended formula macro supporting smooth term kwargs.
Transforms s(x; k=5) syntax to s(x, 5, 2) before calling StatsModels.@formula.
"""
macro formula(ex)
    transformed = transform_smooth_terms(ex)
    return :(StatsModels.@formula(\$transformed))
end

# Export (in MultistateModels.jl, not here)
# export @formula, s, te, ti
```

### User-Facing API

```julia
using MultistateModels

# All these now work:
h12 = Hazard(@formula(0 ~ s(age; k=5)), :wei, 1, 2)
h13 = Hazard(@formula(0 ~ te(age, bmi; k=(8, 6))), :sp, 1, 3)
h14 = Hazard(@formula(0 ~ s(age; k=5, penalty_order=3) + trt), :wei, 1, 4)
```

### Term Registration via apply_schema

After the macro transforms kwargs to positional args, StatsModels creates `FunctionTerm` objects with positional arguments as `ConstantTerm` in `ft.args`. We dispatch on these:

```julia
# Register s() term conversion
# After transformation: s(age, 5, 2) → ft.args = [Term(:age), ConstantTerm(5), ConstantTerm(2)]
# NOTE (R9-1): Second argument MUST be schema::StatsModels.Schema for dispatch to work
function StatsModels.apply_schema(ft::FunctionTerm{typeof(s)}, schema::StatsModels.Schema, Mod::Type)
    var = ft.args[1].sym
    k = ft.args[2].n              # ConstantTerm has .n accessor
    penalty_order = ft.args[3].n
    SmoothTerm(var, k, penalty_order, :natural)  # extrapolation default
end

# Register te() term conversion  
function StatsModels.apply_schema(ft::FunctionTerm{typeof(te)}, schema::StatsModels.Schema, Mod::Type)
    v1, v2 = ft.args[1].sym, ft.args[2].sym
    k1, k2 = ft.args[3].n, ft.args[4].n
    penalty_order = ft.args[5].n
    TensorTerm((v1, v2), (k1, k2), penalty_order, (false, false), (true, true), false)
end

# Register ti() term conversion
function StatsModels.apply_schema(ft::FunctionTerm{typeof(ti)}, schema::StatsModels.Schema, Mod::Type)
    v1, v2 = ft.args[1].sym, ft.args[2].sym
    k1, k2 = ft.args[3].n, ft.args[4].n
    penalty_order = ft.args[5].n
    TensorTerm((v1, v2), (k1, k2), penalty_order, (false, false), (true, true), true)
end
```

### Interaction Pattern Summary

| Pattern | Syntax | Semantics | λ count |
|:--------|:-------|:----------|:--------|
| Smooth main | `s(age)` | $f(\text{age})$ | 1 |
| Varying coefficient | `s(age, by=trt)` | $\text{trt} \cdot f(\text{age})$ | 1 per level |
| Linear × smooth | `s(age) * trt` | Main + interaction | 2 |
| Smooth × smooth | `te(age, bmi)` | Tensor product | 2 |
| Pure interaction | `s(age) + s(bmi) + ti(age, bmi)` | ANOVA decomposition | 3 |
| Time-varying | `te(age, t)` | Modifies baseline | 2 |

### The `t` Symbol: Time Axis Reference

In formulas like `te(age, t)`, the symbol `t` does **not** refer to a data column. It references the hazard's internal time axis:

- For `:sp` family: `t` uses the same B-spline basis as the baseline hazard
- For parametric families (`:wei`, `:exp`, `:gom`): `te(..., t)` is **invalid**

This coupling ensures the time-varying effect shares the same knots and boundary conditions as the baseline.

### Constraints: What's Not Allowed

| Invalid syntax | Reason | Use instead |
|:---------------|:-------|:------------|
| `s(age) * s(bmi)` | Smooth × smooth ambiguous | `te(age, bmi)` or `ti(age, bmi)` |
| `te(age, t)` with `:wei` | No time basis for parametric | Use `:sp` family |
| `s(t)` | Time is not a covariate | `:sp` family handles baseline |
| `te(age, bmi, dose)` | Only 2D tensors | Nest: `te(te(age, bmi), dose)` (future) |

### Varying Coefficient Models

> **⚠️ SCOPE: DEFERRED TO FUTURE WORK**
> 
> The `by` argument for varying coefficient models is documented here for reference
> but is **NOT included in the initial implementation scope**. See 
> `penalized_splines_plan.md` Section "Varying Coefficient Models — FUTURE WORK".

The `by` argument would enable smooth effects that vary across levels of a factor:

```julia
# FUTURE SYNTAX (NOT YET IMPLEMENTED):
# Separate age effect per treatment arm
h12 = Hazard(@formula(0 ~ s(age, by=trt)), :wei, 1, 2)

# Model: log h(t|x) = baseline + f_trt(age)
# where f_0(age), f_1(age) are separate smooth functions
```

**Penalty structure** (future): Each level gets its own smooth with **shared** smoothing parameter (penalizes deviation from mean shape).

---

## Extended Scope: Shared-Origin Tensor Product Splines

### Motivation

For competing risks from state $r$, the baseline hazards to different destinations are often related:
- Similar time patterns (e.g., all hazards increase after diagnosis)
- Correlated uncertainty when some destinations have sparse data
- Total hazard should be smooth

Rather than independent splines with ad-hoc penalties, tensor products provide a principled approach.

### Mathematical Formulation

For state $r$ with destinations $\{s_1, \ldots, s_D\}$, model:
$$\log h_{r,d}(t) = f_r(t, d) = \sum_{i=1}^{K_t} \sum_{j=1}^{D} \beta_{ij} B_i(t) \mathbf{1}_{d=j}$$

This is a tensor product of:
1. **Time basis**: $B_1(t), \ldots, B_{K_t}(t)$ (B-splines as before)
2. **Destination factor**: $\mathbf{1}_{d=s_1}, \ldots, \mathbf{1}_{d=s_D}$

Coefficient matrix $\boldsymbol{\beta}$ is $K_t \times D$.

### Penalty Structure (ANOVA Decomposition)

**Penalty 1: Smoothness in time** (within each destination)
$$\mathcal{P}_1 = \lambda_t \sum_{j=1}^{D} \boldsymbol{\beta}_{\cdot j}^T S_t \boldsymbol{\beta}_{\cdot j} = \lambda_t \text{tr}(\boldsymbol{\beta}^T S_t \boldsymbol{\beta})$$

**Penalty 2: Shrinkage across destinations** (at each time point)
$$\mathcal{P}_2 = \lambda_d \sum_{i=1}^{K_t} \boldsymbol{\beta}_{i \cdot}^T S_d \boldsymbol{\beta}_{i \cdot}$$

where $S_d$ is a penalty on destination dimension:
- **Ridge** (default): $S_d = I_D$ — shrink destinations toward each other
- **First difference**: $S_d = D_1^T D_1$ — adjacent destinations similar (if ordered)
- **Graph Laplacian**: $S_d = L$ — destinations connected by clinical similarity

**Combined penalty (vectorized)**:
$$\mathcal{P}(\boldsymbol{\beta}) = \lambda_t \boldsymbol{\beta}^T (I_D \otimes S_t) \boldsymbol{\beta} + \lambda_d \boldsymbol{\beta}^T (S_d \otimes I_{K_t}) \boldsymbol{\beta}$$

where $\boldsymbol{\beta} = \text{vec}(B)$ is column-vectorized.

### Advantages Over Total Hazard Penalty

The original plan used:
$$\mathcal{P}_{total} = \mu_r \left(\sum_s \theta_{rs}\right)^T S_r \left(\sum_s \theta_{rs}\right)$$

which penalizes the **sum** of log-hazards.

The tensor product penalty instead:
- Shrinks destinations **toward each other** (not toward their sum)
- Has cleaner probabilistic interpretation (random effects on destinations)
- More flexible (can specify structure on destination dimension)
- Borrows strength from data-rich transitions to sparse ones

**Recommendation**: Replace total hazard penalty with tensor products for shared-origin states.

---

## Rigorous Mathematical Framework: Likelihood-Motivated Penalty Decomposition

### Theoretical Foundation

This section provides the rigorous mathematical underpinnings for penalty structures in competing risks models. Rather than motivating penalties through abstract function spaces or latent failure times, we ground our approach in the **likelihood factorization** itself.

**Guiding principle**: The likelihood for competing risks data factors into components that govern *when* events occur and *which type* of event occurs. We design penalties to control the smoothness of each component.

---

### 2.1 The Competing Risks Likelihood Factorization

#### 2.1.1 Setting

Consider competing risks from origin state $r$ to $D$ destinations $\{1, \ldots, D\}$. For a subject observed until time $t$ with event of type $d$:

**Cause-specific hazard**: $h_d(t) = \lim_{\Delta t \to 0} \frac{P(\text{event type } d \text{ in } [t, t+\Delta t) \mid T \geq t)}{\Delta t}$

**Total hazard**: $H(t) = \sum_{d=1}^D h_d(t)$

**Cause allocation probability**: $\pi_d(t) = \frac{h_d(t)}{H(t)}$

#### 2.1.2 Likelihood Factorization

**Theorem 2.1** (Likelihood Factorization). *For a subject with event at time $t$ of type $d$, the likelihood factors as:*

$$L = \underbrace{\exp\left(-\int_0^t H(u) du\right) \cdot H(t)}_{\text{Total event time density}} \cdot \underbrace{\pi_d(t)}_{\text{Cause allocation}}$$

*Proof.* The competing risks likelihood contribution is:
$$L = S(t) \cdot h_d(t)$$
where $S(t) = \exp(-\int_0^t H(u) du)$ is the overall survival function. Rewriting:
$$L = S(t) \cdot H(t) \cdot \frac{h_d(t)}{H(t)} = S(t) \cdot H(t) \cdot \pi_d(t) \;\;\square$$

For a censored observation at time $t$, the contribution is $S(t) = \exp(-\int_0^t H(u) du)$, which involves only the total hazard.

#### 2.1.3 Interpretation

The factorization reveals that the competing risks likelihood has **two distinct components**:

| Component | Expression | Governs |
|:----------|:-----------|:--------|
| **Survival/Event-time** | $\exp(-\int H) \cdot H(t)$ | *When* events occur |
| **Cause allocation** | $\pi_d(t) = h_d(t)/H(t)$ | *Which type* of event |

This motivates controlling:
1. **Smoothness of $H(t)$** → smooth survival curves
2. **Smoothness of $\pi_d(t)$** → stable cause allocation over time

---

### 2.2 B-Spline Representation

#### 2.2.1 Shared Knots Requirement

With shared knots across all $D$ causes, each hazard is represented as:
$$h_d(t) = \sum_{k=1}^K \beta_{kd} B_k(t), \quad \beta_{kd} > 0$$

where $\{B_k(t)\}_{k=1}^K$ is a B-spline basis of order $m_1$ (typically $m_1 = 4$ for cubic splines).

The coefficients form a matrix $\mathbf{B} \in \mathbb{R}^{K \times D}$. Vectorizing column-wise: $\boldsymbol{\beta} = \text{vec}(\mathbf{B}) \in \mathbb{R}^{KD}$.

**Why shared knots?** The Kronecker product structure $(P_D \otimes S)$ requires that $S$ be the **same matrix** for all destinations. If bases differ (different knots), the penalty cannot be expressed as a Kronecker product, and the mathematical framework collapses.

#### 2.2.2 Curvature Penalty Matrix

The curvature penalty matrix $S \in \mathbb{R}^{K \times K}$ (for penalty order $m_2 = 2$) is:
$$S_{ij} = \int_0^\tau B_i''(t) B_j''(t) \, dt$$

**Properties of $S$**:
- Symmetric positive semidefinite
- Banded with $2(m_1 - 1) + 1 = 7$ diagonals (for cubic splines)
- Null space consists of coefficients for linear functions

The penalty matrix is computed using Wood's (2016) derivative-based algorithm, which handles uneven knot spacing correctly.

---

### 2.3 Total Hazard Curvature Penalty

#### 2.3.1 Definition

The total hazard $H(t) = \sum_{d=1}^D h_d(t)$ can be written:
$$H(t) = \sum_{k=1}^K \underbrace{\left(\sum_{d=1}^D \beta_{kd}\right)}_{=: \beta_{H,k}} B_k(t)$$

Let $\boldsymbol{\beta}_H = \sum_{d=1}^D \boldsymbol{\beta}_d \in \mathbb{R}^K$ be the coefficient vector for the total hazard.

**Definition** (Total Hazard Curvature Penalty).
$$\mathcal{P}_H = \int_0^\tau (H''(t))^2 dt$$

#### 2.3.2 Kronecker Form

**Proposition 2.2** (Total Hazard Penalty in Kronecker Form).
$$\mathcal{P}_H = \boldsymbol{\beta}_H^T S \boldsymbol{\beta}_H = \boldsymbol{\beta}^T (\mathbf{1}_D \mathbf{1}_D^T \otimes S) \boldsymbol{\beta}$$

*where $\mathbf{1}_D = (1, \ldots, 1)^T \in \mathbb{R}^D$.*

*Proof.* Let $J = \mathbf{1}_D \mathbf{1}_D^T$ be the $D \times D$ matrix of all ones. For the vectorized coefficients $\boldsymbol{\beta} = [\boldsymbol{\beta}_1; \ldots; \boldsymbol{\beta}_D]$:

\begin{align}
\boldsymbol{\beta}^T (J \otimes S) \boldsymbol{\beta} &= \sum_{d=1}^D \sum_{d'=1}^D J_{dd'} \cdot \boldsymbol{\beta}_d^T S \boldsymbol{\beta}_{d'} \\
&= \sum_{d=1}^D \sum_{d'=1}^D \boldsymbol{\beta}_d^T S \boldsymbol{\beta}_{d'} \quad (\text{since } J_{dd'} = 1 \text{ for all } d, d') \\
&= \left(\sum_{d=1}^D \boldsymbol{\beta}_d\right)^T S \left(\sum_{d'=1}^D \boldsymbol{\beta}_{d'}\right) \\
&= \boldsymbol{\beta}_H^T S \boldsymbol{\beta}_H \;\;\square
\end{align}

#### 2.3.3 Interpretation

The total hazard penalty $\mathcal{P}_H$ directly controls the wiggliness of the survival component of the likelihood. A small $\mathcal{P}_H$ ensures that the survival curve $S(t) = \exp(-\int_0^t H(u) du)$ is smooth.

---

### 2.4 Cause Allocation Smoothness (Indirect Control)

#### 2.4.1 The Problem

The cause allocation probabilities $\pi_d(t) = h_d(t)/H(t)$ are **nonlinear** in the coefficients $\boldsymbol{\beta}$. There is no exact quadratic penalty for controlling $\int (\pi_d'')^2 dt$.

**Question**: Can we construct a quadratic penalty that *indirectly* encourages smooth $\pi_d(t)$?

#### 2.4.2 When Is Cause Allocation Smooth?

**Lemma 2.3** (Identical Shapes Imply Constant Allocation). *If all hazards have the same shape up to scale, i.e., $h_d(t) = c_d \cdot g(t)$ for positive constants $c_d$ and a common shape function $g(t)$, then $\pi_d(t)$ is constant:*
$$\pi_d(t) = \frac{c_d \cdot g(t)}{\sum_{d'} c_{d'} \cdot g(t)} = \frac{c_d}{\sum_{d'} c_{d'}}$$

*Proof.* Direct substitution. The shape $g(t)$ cancels from numerator and denominator. $\square$

**Corollary 2.4**. *When hazards share identical curvature patterns, the cause allocation is maximally smooth (constant in time).*

#### 2.4.3 Deviation Curvature Penalty

Define the **mean hazard**:
$$\bar{h}(t) = \frac{1}{D}\sum_{d=1}^D h_d(t)$$

and the **deviation from mean**:
$$\varepsilon_d(t) = h_d(t) - \bar{h}(t)$$

**Definition** (Deviation Curvature Penalty).
$$\mathcal{P}_\pi = \sum_{d=1}^D \int_0^\tau (\varepsilon_d''(t))^2 dt = \sum_{d=1}^D \int_0^\tau ((h_d - \bar{h})''(t))^2 dt$$

**Proposition 2.5** (Deviation Penalty in Kronecker Form).
$$\mathcal{P}_\pi = \boldsymbol{\beta}^T (C_D \otimes S) \boldsymbol{\beta}$$

*where $C_D = I_D - \frac{1}{D}\mathbf{1}_D\mathbf{1}_D^T$ is the centering matrix.*

*Proof.* Let $\bar{\boldsymbol{\beta}} = \frac{1}{D}\sum_{d=1}^D \boldsymbol{\beta}_d$ and $\boldsymbol{\varepsilon}_d = \boldsymbol{\beta}_d - \bar{\boldsymbol{\beta}}$. Then:
$$\varepsilon_d(t) = \sum_{k=1}^K (\beta_{kd} - \bar{\beta}_k) B_k(t) = \sum_{k=1}^K \varepsilon_{kd} B_k(t)$$

The deviation curvature penalty is:
\begin{align}
\mathcal{P}_\pi &= \sum_{d=1}^D \boldsymbol{\varepsilon}_d^T S \boldsymbol{\varepsilon}_d
\end{align}

Now, the centering matrix $C_D$ acts on the destination dimension. For the vectorized coefficients:
\begin{align}
(C_D \otimes I_K) \boldsymbol{\beta} &= [\boldsymbol{\varepsilon}_1; \ldots; \boldsymbol{\varepsilon}_D]
\end{align}

Therefore:
\begin{align}
\boldsymbol{\beta}^T (C_D \otimes S) \boldsymbol{\beta} &= \boldsymbol{\beta}^T (C_D \otimes I_K)^T (I_D \otimes S) (C_D \otimes I_K) \boldsymbol{\beta} \\
&= \sum_{d=1}^D \boldsymbol{\varepsilon}_d^T S \boldsymbol{\varepsilon}_d = \mathcal{P}_\pi
\end{align}

where we used $C_D^T = C_D$ and $C_D^2 = C_D$ (idempotent). $\square$

#### 2.4.4 Properties of the Centering Matrix

**Proposition 2.6** (Properties of $C_D$).
1. *Symmetric*: $C_D = C_D^T$
2. *Idempotent*: $C_D^2 = C_D$
3. *Rank*: $\text{rank}(C_D) = D - 1$
4. *Kernel*: $\ker(C_D) = \text{span}(\mathbf{1}_D)$
5. *Row sums*: $C_D \mathbf{1}_D = \mathbf{0}$

*Proof.* Let $P_1 = \frac{1}{D}\mathbf{1}_D\mathbf{1}_D^T$. Then $P_1$ is the orthogonal projection onto $\text{span}(\mathbf{1}_D)$:
- $P_1^T = P_1$ (symmetric)
- $P_1^2 = \frac{1}{D^2}\mathbf{1}_D(\mathbf{1}_D^T\mathbf{1}_D)\mathbf{1}_D^T = \frac{1}{D^2} \cdot D \cdot \mathbf{1}_D\mathbf{1}_D^T = P_1$ (idempotent)

Thus $C_D = I_D - P_1$ is the complementary projection. Properties 1-5 follow from standard projection theory. $\square$

#### 2.4.5 Connection to Cause Allocation Smoothness

**Proposition 2.7** (Deviation Penalty Controls Ratio Smoothness). *The deviation penalty provides an indirect control on cause allocation smoothness:*
1. *If $\mathcal{P}_\pi = 0$, then all hazards have identical curvature, and $\pi_d(t)$ is constant.*
2. *When $\mathcal{P}_\pi$ is small, hazards have similar curvature, and $\pi_d(t)$ changes slowly.*

*Heuristic argument for (2).* Differentiating $\pi_d = h_d/H$:
$$\pi_d'(t) = \frac{h_d'(t) - \pi_d(t) H'(t)}{H(t)} = \frac{h_d'(t) - \pi_d(t) \sum_{d'} h_{d'}'(t)}{H(t)}$$

When all hazards have similar derivatives (i.e., $h_d'(t) \approx \pi_d(t) H'(t)$), the numerator is small, so $\pi_d'(t)$ is small. By extension, when all hazards have similar second derivatives, $\pi_d''(t)$ is also controlled. $\square$

**Caveat**: The relationship is *sufficient but not necessary*. One can construct examples where $\mathcal{P}_\pi$ is large but $\pi_d(t)$ is smooth (e.g., if deviations in curvature happen to cancel in the ratio). However, in practice, penalizing deviation curvature effectively stabilizes cause allocation.

---

### 2.5 The Fundamental ANOVA Decomposition

#### 2.5.1 Spectral Decomposition of $I_D$

**Lemma 2.8** (Spectral Decomposition). *The identity matrix admits the orthogonal decomposition:*
$$I_D = \frac{1}{D} \mathbf{1}_D \mathbf{1}_D^T + C_D$$

*Proof.* By definition, $C_D = I_D - \frac{1}{D}\mathbf{1}_D\mathbf{1}_D^T$. Rearranging gives the result. Orthogonality follows from $(\frac{1}{D}\mathbf{1}_D\mathbf{1}_D^T) C_D = \frac{1}{D}\mathbf{1}_D\mathbf{1}_D^T - \frac{1}{D^2}\mathbf{1}_D(\mathbf{1}_D^T\mathbf{1}_D)\mathbf{1}_D^T = 0$. $\square$

#### 2.5.2 Independent Penalty

**Definition** (Independent Penalty). The standard penalty that treats each hazard independently:
$$\mathcal{P}_{\text{ind}} = \sum_{d=1}^D \int_0^\tau (h_d''(t))^2 dt = \sum_{d=1}^D \boldsymbol{\beta}_d^T S \boldsymbol{\beta}_d = \boldsymbol{\beta}^T (I_D \otimes S) \boldsymbol{\beta}$$

#### 2.5.3 Main Decomposition Theorem

**Theorem 2.9** (Likelihood-Motivated Penalty Decomposition). *The independent penalty decomposes into total hazard and deviation components:*
$$\mathcal{P}_{\text{ind}} = \frac{1}{D}\mathcal{P}_H + \mathcal{P}_\pi$$

*Equivalently, in matrix form:*
$$I_D \otimes S = \frac{1}{D}(\mathbf{1}_D\mathbf{1}_D^T \otimes S) + (C_D \otimes S)$$

*Proof.* Using Lemma 2.8 and the distributive property of Kronecker products:
\begin{align}
I_D \otimes S &= \left(\frac{1}{D}\mathbf{1}_D\mathbf{1}_D^T + C_D\right) \otimes S \\
&= \frac{1}{D}(\mathbf{1}_D\mathbf{1}_D^T \otimes S) + (C_D \otimes S) \;\;\square
\end{align}

*Corollary* (Integral Form):
$$\sum_{d=1}^D \int (h_d'')^2 = \frac{1}{D} \int (H'')^2 + \sum_{d=1}^D \int ((h_d - \bar{h})'')^2$$

*Proof.* Apply the matrix identity to $\boldsymbol{\beta}^T (\cdot) \boldsymbol{\beta}$ and use Propositions 2.2 and 2.5. $\square$

#### 2.5.4 Interpretation

The decomposition reveals that the standard "independent" penalty on individual hazard curvatures **implicitly** penalizes both likelihood components:

| Component | Weight | Controls |
|:----------|:-------|:---------|
| Total hazard curvature | $1/D$ | Survival smoothness |
| Deviation curvature | $1$ | Cause allocation smoothness |

The relative weight $1/D$ on the total hazard component means that with more competing risks, the independent penalty emphasizes deviation control over survival smoothness.

---

### 2.6 The Three Penalty Modes

#### 2.6.1 Summary Table

| Mode | Matrix | Quadratic Form | Controls |
|:-----|:-------|:---------------|:---------|
| **Independent** | $I_D \otimes S$ | $\sum_d \int (h_d'')^2$ | Both (balanced) |
| **Total Hazard** | $\mathbf{1}_D\mathbf{1}_D^T \otimes S$ | $\int (H'')^2$ | Survival only |
| **Deviation** | $C_D \otimes S$ | $\sum_d \int ((h_d - \bar{h})'')^2$ | Cause allocation (indirect) |

#### 2.6.2 Matrix Properties

**Proposition 2.10** (Properties of Penalty Matrices).

| Matrix | Rank | Eigenvalues |
|:-------|:-----|:------------|
| $\mathbf{1}_D\mathbf{1}_D^T \otimes S$ | $\text{rank}(S)$ | $D \cdot \lambda_i(S)$ (with multiplicity 1 each) and 0 |
| $C_D \otimes S$ | $(D-1) \cdot \text{rank}(S)$ | $\lambda_i(S)$ (with multiplicity $D-1$ each) and 0 |
| $I_D \otimes S$ | $D \cdot \text{rank}(S)$ | $\lambda_i(S)$ (with multiplicity $D$ each) |

*Proof.* For Kronecker products $A \otimes B$, the eigenvalues are products of eigenvalues of $A$ and $B$. The eigenvalues of $\mathbf{1}_D\mathbf{1}_D^T$ are $\{D, 0, \ldots, 0\}$; the eigenvalues of $C_D$ are $\{1, \ldots, 1, 0\}$ with $D-1$ ones. $\square$

---

### 2.7 General Penalty Formulation

#### 2.7.1 Two-Parameter Penalty

The general penalty allows separate control of the two likelihood components:

$$\mathcal{P}(\boldsymbol{\beta}; \lambda_H, \lambda_\pi) = \lambda_H \mathcal{P}_H + \lambda_\pi \mathcal{P}_\pi$$

$$= \lambda_H \boldsymbol{\beta}^T (\mathbf{1}_D\mathbf{1}_D^T \otimes S) \boldsymbol{\beta} + \lambda_\pi \boldsymbol{\beta}^T (C_D \otimes S) \boldsymbol{\beta}$$

$$= \boldsymbol{\beta}^T \left(\lambda_H (\mathbf{1}_D\mathbf{1}_D^T \otimes S) + \lambda_\pi (C_D \otimes S)\right) \boldsymbol{\beta}$$

#### 2.7.2 Special Cases

| Setting | Penalty Structure | Interpretation |
|:--------|:------------------|:---------------|
| $\lambda_H = \lambda/D$, $\lambda_\pi = \lambda$ | $\lambda(I_D \otimes S)$ | Independent (standard) |
| $\lambda_H > 0$, $\lambda_\pi = 0$ | Pure total hazard | Max flexibility in cause allocation |
| $\lambda_H = 0$, $\lambda_\pi > 0$ | Pure deviation | No constraint on survival smoothness |
| $\lambda_H \gg \lambda_\pi$ | Emphasize survival | Prioritize smooth survival curve |
| $\lambda_\pi \gg \lambda_H$ | Emphasize allocation | Prioritize stable cause mix |

#### 2.7.3 Penalized Log-Likelihood

The penalized log-likelihood is:
$$\ell_p(\boldsymbol{\beta}; \lambda_H, \lambda_\pi) = \ell(\boldsymbol{\beta}) - \frac{1}{2}\mathcal{P}(\boldsymbol{\beta}; \lambda_H, \lambda_\pi)$$

where $\ell(\boldsymbol{\beta})$ is the unpenalized competing risks log-likelihood.

---

### 2.8 Explicit Quadratic Forms

#### 2.8.1 Direct Computation

For implementation, it's useful to express penalties in terms of coefficient vectors.

**Proposition 2.11** (Explicit Quadratic Forms). *Let $\bar{\boldsymbol{\beta}} = \frac{1}{D}\sum_{d=1}^D \boldsymbol{\beta}_d$ and $\boldsymbol{\varepsilon}_d = \boldsymbol{\beta}_d - \bar{\boldsymbol{\beta}}$.*

1. **Independent**: $\mathcal{P}_{\text{ind}} = \sum_{d=1}^D \boldsymbol{\beta}_d^T S \boldsymbol{\beta}_d$

2. **Total Hazard**: $\mathcal{P}_H = (D\bar{\boldsymbol{\beta}})^T S (D\bar{\boldsymbol{\beta}}) = D^2 \bar{\boldsymbol{\beta}}^T S \bar{\boldsymbol{\beta}}$

3. **Deviation**: $\mathcal{P}_\pi = \sum_{d=1}^D \boldsymbol{\varepsilon}_d^T S \boldsymbol{\varepsilon}_d$

4. **Verification**: $\mathcal{P}_{\text{ind}} = \frac{1}{D}\mathcal{P}_H + \mathcal{P}_\pi$

#### 2.8.2 Numerical Verification

The decomposition can be verified numerically. For $D = 3$ competing risks with random coefficients:

```
1. TOTAL HAZARD PENALTY: P_H = ∫(H'')² where H = Σh_d
   Direct: β_H' S β_H = 800.807...
   Kronecker: β'(11'⊗S)β = 800.807...
   Match: ✓

2. DEVIATION PENALTY: P_π = Σ∫((h_d - h̄)'')²
   Direct: Σεd'Sεd = 608.806...
   Kronecker: β'(C_D⊗S)β = 608.806...
   Match: ✓

3. INDEPENDENT PENALTY: P_ind = Σ∫(h_d'')²
   Direct: Σβd'Sβd = 875.742...
   Kronecker: β'(I_D⊗S)β = 875.742...
   Match: ✓

4. DECOMPOSITION: P_ind = (1/D)P_H + P_π
   LHS (P_ind): 875.742...
   RHS ((1/D)P_H + P_π): 875.742...
   Match: ✓
```

---

### 2.9 Adversarial Review

#### 2.9.1 Is the Deviation Penalty Really About Cause Allocation?

**Challenge**: The claim that $\mathcal{P}_\pi$ controls cause allocation smoothness is heuristic. Can we make it precise?

**Analysis**: The relationship is *indirect* because $\pi_d(t)$ is nonlinear in $\boldsymbol{\beta}$.

**Precise statement**: The deviation penalty provides a **sufficient but not necessary** condition for smooth cause allocation:
- If $\mathcal{P}_\pi = 0$ (all hazards have identical curvature), then $\pi_d(t)$ is exactly constant.
- If $\mathcal{P}_\pi$ is small, hazards have similar shapes, which *typically* implies smooth $\pi_d(t)$.
- The converse fails: one can construct examples where $\mathcal{P}_\pi$ is large but $\pi_d(t)$ is smooth due to fortuitous cancellation.

**Practical implication**: The deviation penalty is a computationally tractable proxy that works well in practice.

#### 2.9.2 Why Not Penalize $\pi_d''$ Directly?

**Challenge**: Why not compute $\int (\pi_d'')^2$ and penalize that?

**Analysis**: Direct penalization would require:
1. **Nonlinear penalty** (not quadratic in $\boldsymbol{\beta}$)
2. **Numerical integration** at each optimization step
3. **Complex gradients/Hessians** involving quotient rules

The computational cost would be prohibitive. The deviation penalty provides an effective quadratic approximation.

#### 2.9.3 Scale Sensitivity

**Challenge**: If hazard magnitudes differ greatly (e.g., $h_1 \sim O(1)$, $h_2 \sim O(100)$), does the penalty structure break down?

**Analysis**: Yes, the penalties become dominated by the larger hazard:
- $\mathcal{P}_H = \int (H'')^2 \approx \int (h_2'')^2$ when $h_2 \gg h_1$
- $\mathcal{P}_\pi$: Deviations from mean are driven by the large hazard

**Recommendations for highly unequal hazards**:
1. Model on log scale: $\log h_d(t) = \sum_k \theta_{kd} B_k(t)$
2. Use per-cause smoothing parameters
3. Consider standardization (though this changes interpretation)

#### 2.9.4 Ordering Invariance

**Challenge**: Is the decomposition invariant to the labeling of causes?

**Analysis**: Yes. Both $\mathbf{1}_D\mathbf{1}_D^T$ and $C_D$ are symmetric under permutation of indices. The all-ones matrix is trivially permutation-invariant; the centering matrix $C_D = I_D - \frac{1}{D}\mathbf{1}_D\mathbf{1}_D^T$ has equal off-diagonal entries $-1/D$ and equal diagonal entries $(D-1)/D$.

#### 2.9.5 Relationship to ANOVA

**Challenge**: How does this relate to classical ANOVA decomposition?

**Analysis**: The decomposition $I_D = \frac{1}{D}\mathbf{1}_D\mathbf{1}_D^T + C_D$ is the variance decomposition for a one-way layout:
- $\frac{1}{D}\mathbf{1}_D\mathbf{1}_D^T$ projects onto the **grand mean** (between-group mean)
- $C_D$ projects onto the **deviations from mean** (within-group variation)

In our context:
- Total hazard penalty ↔ "between-cause" component (the shared pattern)
- Deviation penalty ↔ "within-cause" component (the heterogeneity)

---

### 2.10 Connection to Smoothing Spline ANOVA (SS-ANOVA)

#### 2.10.1 Background

The penalty decomposition is a special case of the **Smoothing Spline ANOVA** framework (Wahba 1990, Gu 2002). For a function $f$ on a product domain $\mathcal{X}_1 \times \cdots \times \mathcal{X}_D$, SS-ANOVA decomposes:

$$f(\mathbf{x}) = f_0 + \sum_d f_d(x_d) + \sum_{d < d'} f_{dd'}(x_d, x_{d'}) + \cdots$$

with orthogonality constraints. Each component has its own smoothing parameter.

#### 2.10.2 Our Specialization

In our setting:
- The "domain" is $[0, \tau] \times \{1, \ldots, D\}$ (time × cause)
- The function $h(t, d) = h_d(t)$ is **additive** by construction (no time × cause interaction)
- The penalty decomposes into:
  - **Mean component**: $\bar{h}(t)$ — the average hazard over causes
  - **Cause components**: $h_d(t) - \bar{h}(t)$ — deviations from mean

The total hazard penalty controls the mean component; the deviation penalty controls the cause components.

#### 2.10.3 References

1. **Gu, C. (2002)**. *Smoothing Spline ANOVA Models*. Springer.
2. **Wahba, G. (1990)**. *Spline Models for Observational Data*. SIAM.
3. **Wood, S.N. (2017)**. *Generalized Additive Models* (2nd ed.). CRC Press. Chapter 5.
4. **Gu, C. & Wahba, G. (1993)**. JRSS-B 55(2), 353–368.

---

### Design Decisions

| Decision | Options | Recommendation |
|----------|---------|----------------|
| Destination penalty $S_d$ | Ridge, Difference, Custom | Ridge (default), user-configurable |
| Shared $\lambda_t$ across destinations | Yes / No | Yes (tensor structure implies sharing) |
| Number of $\lambda$ parameters | 2 or more | Start with 2 ($\lambda_t$, $\lambda_d$) |

---

## Extended Scope: General Tensor Product Splines

### Use Cases

| Use Case | Syntax | Example |
|----------|--------|---------|
| Time-varying effect | `te(age, t)` | Effect of age changes over follow-up |
| Spatial effects | `te(lat, lon)` | Geographic variation |
| Pure interaction | `ti(age, bmi)` | Interaction excluding main effects |

### Implementation Structs

```julia
# Formula-level term for te() and ti()
struct TensorTerm <: StatsModels.AbstractTerm
    vars::Tuple{Symbol, Symbol}
    k::Tuple{Int, Int}
    penalty_order::Int            # Derivative order for penalty (default 2)
    monotone::Tuple{Bool, Bool}
    increasing::Tuple{Bool, Bool}
    interaction_only::Bool  # true for ti(), false for te()
end

# Helper constructors (for reference - actual construction via apply_schema)
te(v1::Symbol, v2::Symbol; k=(10, 10), penalty_order=2, monotone=(false, false)) = 
    TensorTerm((v1, v2), k, penalty_order, monotone, (true, true), false)

ti(v1::Symbol, v2::Symbol; k=(10, 10), penalty_order=2) = 
    TensorTerm((v1, v2), k, penalty_order, (false, false), (true, true), true)
```

### Basis Construction

For `te(x1, x2)`:
$$f(x_1, x_2) = \sum_{i,j} \beta_{ij} B_{1i}(x_1) B_{2j}(x_2)$$

**Efficient evaluation** (avoid full Kronecker product):
```julia
function eval_tensor_term(beta_mat, B1_vals, B2_vals)
    # O(n * k1 * k2) instead of O(n * k1² * k2²)
    return sum((B1_vals * beta_mat) .* B2_vals, dims=2)
end
```

### Penalty Structure

Marginal penalties from Wood (2016):
- $S_1 = D_1^T D_1 \otimes I_{k_2}$: Roughness in $x_1$
- $S_2 = I_{k_1} \otimes D_2^T D_2$: Roughness in $x_2$

**Efficient penalty application** (no full matrix formation):
```julia
function apply_penalty(beta_vec, D1, D2, λ1, λ2)
    beta_mat = reshape(beta_vec, k1, k2)
    S1_beta = vec(D1' * (D1 * beta_mat))       # S1 penalty
    S2_beta = vec(beta_mat * (D2' * D2))       # S2 penalty
    return λ1 * S1_beta + λ2 * S2_beta
end
```

### The ti() Reparametrization: Pure Interaction

The `ti()` term (tensor interaction) represents **pure interaction** that excludes main effects. This is essential for ANOVA-style decompositions:

$$f(x_1, x_2) = f_1(x_1) + f_2(x_2) + f_{12}(x_1, x_2)$$

where `s(x1) + s(x2) + ti(x1, x2)` models the three components.

#### Constraint Mechanism

For `ti()` to represent pure interaction, it must satisfy **centered constraints**:
$$\sum_i f_{12}(x_{1i}, x_2) = 0 \quad \forall x_2$$
$$\sum_j f_{12}(x_1, x_{2j}) = 0 \quad \forall x_1$$

This is achieved via **basis reparametrization** using Helmert-style contrasts (Wood, 2017 §5.6.3):

1. Build sum-to-zero contrast matrix $C_1$ ($k_1 \times (k_1-1)$) for dimension 1
2. Build sum-to-zero contrast matrix $C_2$ ($k_2 \times (k_2-1)$) for dimension 2
3. Center each marginal basis: $B_1^c = B_1 C_1$, $B_2^c = B_2 C_2$
4. Form interaction basis: row-wise Kronecker of centered bases
5. Result: $n \times (k_1-1)(k_2-1)$ basis satisfying constraints

#### Implementation

```julia
"""
Build sum-to-zero contrast matrix using Helmert-style contrasts.
Returns: k × (k-1) matrix where each column sums to zero.
"""
function build_sum_to_zero_contrast(k::Int)
    # Returns k × (k-1) orthonormal matrix where each column sums to zero
    # Based on Helmert contrasts, then orthonormalized via QR for numerical stability
    # (QR preserves column space, so sum-to-zero property is maintained)
    C = zeros(Float64, k, k-1)
    for j in 1:(k-1)
        C[j, j] = 1.0
        C[j+1:k, j] .= -1.0 / (k - j)
    end
    # Orthonormalize: QR preserves the property that columns span sum-to-zero subspace
    return Matrix(qr(C).Q)[:, 1:(k-1)]
end

"""
Build the centered interaction basis for ti() terms.
Returns: n × ((k1-1)*(k2-1)) constrained interaction basis.
"""
function build_ti_basis(B1::Matrix{Float64}, B2::Matrix{Float64})
    n = size(B1, 1)
    k1, k2 = size(B1, 2), size(B2, 2)
    
    # Get contrast matrices
    C1 = build_sum_to_zero_contrast(k1)  # k1 × (k1-1)
    C2 = build_sum_to_zero_contrast(k2)  # k2 × (k2-1)
    
    # Centered marginal bases
    B1_c = B1 * C1  # n × (k1-1)
    B2_c = B2 * C2  # n × (k2-1)
    
    # Row-wise Kronecker product for interaction
    # Result: n × ((k1-1)*(k2-1))
    p_new = (k1-1) * (k2-1)
    B_ti = zeros(Float64, n, p_new)
    for i in 1:n
        B_ti[i, :] = kron(B1_c[i, :], B2_c[i, :])
    end
    
    return B_ti
end

"""
Penalty matrix for ti() term in the reduced parameterization.
"""
function build_ti_penalty(S1::Matrix{Float64}, S2::Matrix{Float64}, 
                          k1::Int, k2::Int, λ1::Real, λ2::Real)
    C1 = build_sum_to_zero_contrast(k1)
    C2 = build_sum_to_zero_contrast(k2)
    
    # Transform penalties: C' S C
    S1_ti = C1' * S1 * C1  # (k1-1) × (k1-1)
    S2_ti = C2' * S2 * C2  # (k2-1) × (k2-1)
    
    # Kronecker structure on reduced space
    I1 = Matrix{Float64}(I, k1-1, k1-1)
    I2 = Matrix{Float64}(I, k2-1, k2-1)
    
    S_ti = λ1 * kron(I2, S1_ti) + λ2 * kron(S2_ti, I1)
    return S_ti
end
```

#### Why Reparametrize Rather Than Constrain?

| Approach | Pros | Cons |
|----------|------|------|
| **Soft constraint** (penalty) | Simple implementation | Doesn't exactly enforce |
| **Hard constraint** (Lagrange) | Exact | Complicated optimization |
| **Reparametrization** | Exact + unconstrained optimization | Reduced basis dimension |

We use **reparametrization** because:
1. Optimization stays unconstrained (no Lagrange multipliers)
2. Constraints are exactly satisfied by construction
3. Basis dimension matches degrees of freedom: $(K_1-1) \times (K_2-1)$

---

## Monotone Tensor Products

### Motivation

In survival analysis, shape constraints are often substantively motivated:
- **Monotone in time**: Hazards that increase or decrease over follow-up (e.g., post-transplant mortality)
- **Monotone in covariate**: Risk that increases with biomarker level
- **Mixed**: Monotone in one dimension but flexible in another

Tensor products allow **per-dimension monotonicity**, unlike univariate splines where combining monotonicity with penalization creates conflicts.

### I-Spline Basis via Cumulative Differencing

The existing `spline_ests2coefs()` in `spline.jl` implements I-spline transformation via cumulative differencing:

$$\text{coef}_i = \text{coef}_{i-1} + \exp(\tilde{\beta}_i) \cdot \frac{t_{i+k} - t_i}{k}$$

where:
- $\tilde{\beta}_i$ is the estimation-scale parameter (unconstrained)
- $\exp(\tilde{\beta}_i)$ ensures non-negative increments
- $(t_{i+k} - t_i)/k$ is the knot-spacing weight for proper I-spline integration
- $k$ is the spline order

This guarantees monotonicity through cumulative sums of non-negative quantities.

### Tensor Product with Partial Monotonicity

For `te(x1, x2, monotone=(false, true))`:
- $x_1$: Standard B-spline basis $B_1(x_1)$ with unconstrained coefficients
- $x_2$: I-spline transformation applied along columns of coefficient matrix

The transformation operates on the coefficient matrix $\tilde{B}$ (estimation scale):

```julia
# For tensor product with monotonicity in dimension 2 (columns)
function ests_to_coefs_tensor_monotone_dim2(ests_mat, knots2, order2)
    k1, k2 = size(ests_mat)
    coefs_mat = similar(ests_mat)
    for i in 1:k1
        # Apply cumulative I-spline transform to each row
        coefs_mat[i, 1] = exp(ests_mat[i, 1])  # Intercept
        for j in 2:k2
            coefs_mat[i, j] = coefs_mat[i, j-1] + 
                exp(ests_mat[i, j]) * (knots2[j + order2] - knots2[j]) / order2
        end
    end
    return coefs_mat
end
```

### Penalty Structure for Monotone Dimensions

**Key insight**: Penalization operates on the **estimation-scale parameters** $\tilde{\beta}$, not the transformed coefficients.

For monotone dimension with I-splines:
1. Coefficients stored as $\tilde{\beta}$ (log-scale, unconstrained)
2. Forward pass applies cumulative sum: $\beta = \text{cumsum}(\exp(\tilde{\beta}) \cdot w)$
3. Penalty matrix applied to $\tilde{\beta}$ directly

This induces shrinkage toward **constant (flat)** monotone surfaces, not toward zero.

**Penalty formulation**:
$$\text{Penalty} = \lambda_1 \tilde{\beta}^T S_1 \tilde{\beta} + \lambda_2 \tilde{\beta}^T S_2 \tilde{\beta}$$

where $S_2$ (monotone dimension) penalizes roughness in the estimation-scale parameters.

### Implementation Details

```julia
struct MonotoneTensorConfig
    monotone::Tuple{Bool, Bool}  # (dim1_monotone, dim2_monotone)
    increasing::Tuple{Bool, Bool}  # Direction (true=increasing)
end

function transform_tensor_coefficients(ests_mat, knots1, knots2, order1, order2, config)
    coefs_mat = copy(ests_mat)
    
    # Apply cumulative I-spline transform along monotone dimensions
    if config.monotone[1]
        for j in axes(coefs_mat, 2)
            coefs_mat[:, j] = cumulative_ispline_transform(
                coefs_mat[:, j], knots1, order1, config.increasing[1]
            )
        end
    end
    if config.monotone[2]
        for i in axes(coefs_mat, 1)
            coefs_mat[i, :] = cumulative_ispline_transform(
                coefs_mat[i, :], knots2, order2, config.increasing[2]
            )
        end
    end
    return coefs_mat
end

function cumulative_ispline_transform(ests, knots, order, increasing)
    coefs = similar(ests)
    coefs[1] = exp(ests[1])
    for i in 2:length(coefs)
        coefs[i] = coefs[i-1] + exp(ests[i]) * (knots[i + order] - knots[i]) / order
    end
    return increasing ? coefs : reverse(coefs)
end
```

### Shared-Origin Tensor Products with Monotonicity

For competing risks with monotone temporal baseline:
```julia
model = multistatemodel(h12, h13, h14;
    data = data,
    shared_origin_tensor = Dict(
        1 => SharedOriginConfig(monotone_time=true, increasing=true)
    )
)
```

This specifies:
- Shared time basis across destinations with I-spline transformation
- Destination effects via ridge-penalized coefficients
- Hazards guaranteed to be monotonically increasing over follow-up

**Use case**: Immediate post-operative period where all-cause mortality decreases over time.

---

## Implementation Roadmap (Revised)

### Phase 0: Infrastructure Preparation
- [ ] Add penalty constants to `constants.jl`
- [ ] Create `PenaltyConfig` struct (model-level)
- [ ] Update error handling (`ArgumentError` convention)

### Phase 1: Penalty Matrix Construction
- [ ] `compute_derivative_penalty_matrix(basis, order)` — Wood's algorithm
- [ ] Handle `RecombinedBSplineBasis` transformation
- [ ] Test banding structure
- [ ] Verify against R's `mgcv::smoothCon`

### Phase 2: Penalized Likelihood
- [ ] `penalized_loglik` wrapper
- [ ] Individual transition penalties
- [ ] Shared-origin tensor penalties
- [ ] ForwardDiff compatibility

### Phase 3: Smoothing Parameter Selection
- [ ] `compute_ncv_criterion(state, model, data)` — evaluate deviance at LOO parameters
- [ ] Multi-λ optimization for tensor products (BFGS on log scale)
- [ ] REML (secondary benchmark, may defer)
- [ ] AIC/BIC with effective degrees of freedom
- [ ] Subject weights support
- [ ] Indefinite Hessian fallback

### Phase 4: Confidence Intervals
- [ ] Bayesian posterior CIs
- [ ] Simulation-based CIs for functions
- [ ] Integration with `predict` interface

### Phase 5: Covariate Splines
- [ ] `SmoothTerm <: StatsModels.AbstractTerm` struct for `s()` syntax
- [ ] Formula parsing via term interception
- [ ] Basis construction from data quantiles
- [ ] Identifiability constraints (sum-to-zero centering)

### Phase 6: Tensor Product Infrastructure
- [ ] `SharedOriginConfig` and `SharedOriginTensorConfig` structs
- [ ] `TensorTerm <: StatsModels.AbstractTerm` for `te()` and `ti()`
- [ ] Destination penalty matrices (:ridge, :difference, custom)
- [ ] Kronecker-efficient penalty application
- [ ] Partial domain handling
- [ ] Model construction integration for `shared_origin_tensor` kwarg

### Phase 7: Testing & Documentation
- [ ] Unit tests for penalty matrices (verify against R mgcv)
- [ ] RecombinedBSplineBasis penalty transformation tests
- [ ] Coverage simulation studies
- [ ] Shared-origin tensor tests
- [ ] R/flexsurv/flexmsm comparison benchmarks

**Estimated timeline: 35-42 days**

---

## Compatibility Notes

### MCEM / Semi-Markov Models (Deferred)

Penalization with MCEM-fitted semi-Markov models presents challenges:

1. **Stochastic gradients**: MCEM uses importance-weighted gradient estimates, not exact gradients
2. **Hessian estimation**: The penalized Hessian $H_\lambda$ includes the (constant) penalty term, but the likelihood Hessian is stochastic
3. **NCV computation**: Leave-one-out parameter perturbations rely on exact subject-level contributions

**Potential approach** (future work):
- Use converged MCEM Hessian estimate for penalty selection
- Require convergence diagnostics before smoothing selection
- Consider REML as alternative (less sensitive to gradient noise)

**Initial scope**: Markov models and exact-observation semi-Markov models only.

### Constrained Optimization

When penalties are combined with parameter constraints (e.g., monotone 1D splines, positivity), the optimization becomes:
$$\min_\theta -\ell_p(\theta; \lambda) \quad \text{subject to } g(\theta) \leq 0$$

This requires constrained trust region methods, which are more complex than unconstrained penalized optimization.

**Initial scope**: Penalization available only for unconstrained problems. Monotonicity via tensor products (per-dimension) rather than 1D constrained splines.

---

## Resolved Design Questions

| Question | Resolution |
|:---------|:-----------|
| Total vs individual penalty | Replace with shared-origin tensor products |
| Shared smoothing parameters | Via tensor product structure |
| Penalty storage location | `PenaltyConfig` stored in `ConvergenceRecords` NamedTuple (R8-1, R8-4, R9-4) |
| PIJCVState usage | Use existing `pijcv_criterion()` for NCV; provide appropriate loss_fn (R8-13) |
| RecombinedBSplineBasis | Transform penalty: $S_{recombined} = R^T S_{original} R$ (R is rectangular) |
| Initialization | $\log\lambda_0 = -3 - \log\|S\|_F$ (heuristic, should validate) |
| Smoothing param bounds | Log scale: $[-20, 20]$ |
| Monotone + penalization (1D) | Disallow (conflicting objectives) |
| Monotone tensor products | Supported via cumulative I-spline transformation per dimension |
| MCEM compatibility | Deferred; initial scope is Markov + exact data |
| Absorbing states | Treat same as other destinations in shared-origin tensors |
| `apply_schema` signature | Must use `schema::StatsModels.Schema` (R9-1) |

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
Same Bayesian approximation: $\theta \sim N(\hat{\theta}, V_\theta)$ where $V_\theta = I_p(\hat{\theta})^{-1}$ and $I_p = -H_p$ is the penalized Fisher information.

This gives "close to across-the-function frequentist coverage probabilities because it accounts for both sampling variability and smoothing bias."

#### Model Selection
- AIC: $-2\ell(\theta) + 2 \cdot \text{edf}$
- BIC: $-2\ell(\theta) + \log(\check{n}) \cdot \text{edf}$
- Effective degrees of freedom (simple): $\text{edf} = \text{tr}(I \cdot I_p^{-1})$ where $I = -H$ is unpenalized Fisher info
- Note: Eletti uses a more complex formula; we implement the simple version initially

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
