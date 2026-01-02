# Implementation Specification: Penalized Splines in MultistateModels.jl

## 1. Executive Summary

This document specifies penalized B-splines for baseline hazard estimation in `MultistateModels.jl`. Covariate splines (`s(x)`, `te(x, y)`) are deferred to a future phase.

### 1.1 Scope (Phase 1)

1. **Baseline Hazard Smoothing**: B-spline representation of hazards on the natural scale.
2. **Competing Risks**: Shared knot locations, configurable penalty sharing.
3. **Total Hazard Penalty**: Optional penalty on smoothness of total hazard out of each origin.

### 1.2 Core Design Decisions

| Component | Decision | Rationale |
|:---|:---|:---|
| **Model Scale** | Natural hazard scale: $h(t) = \sum_i \beta_i B_i(t)$, $\beta_i > 0$ | Closed-form cumulative hazard; positivity via constraints. |
| **Penalty Scale** | Natural scale: $\int (h''(t))^2 dt = \beta^T S \beta$ | Quadratic form; stable; compatible with adaptive smooths. |
| **Penalty Basis** | Derivative-based (Wood 2016) | Robust to uneven knot spacing. |
| **Smoothing Selection** | PIJCV (Wood 2024) with GCV fallback | Efficient; accounts for within-subject correlation. |
| **Spline Library** | `BSplineKit.jl` | Pure Julia; supports derivatives. |

---

## 2. Mathematical Framework

### 2.1 Likelihood-Motivated Penalty Decomposition

#### 2.1.1 Competing Risks Likelihood Factorization

For an observation with event at time $t$ of type $d$, the competing risks likelihood factors as:

$$L = \underbrace{\exp\left(-\int_0^t H(u) du\right) \cdot H(t)}_{\text{Total event time density}} \cdot \underbrace{\pi_d(t)}_{\text{Cause allocation}}$$

where:
- $H(t) = \sum_{d=1}^D h_d(t)$ is the **total hazard** (governs the survival component)
- $\pi_d(t) = h_d(t) / H(t)$ is the **cause allocation probability** (time-varying multinomial)

This factorization reveals two distinct sources of variation:
1. **When** events occur (governed by $H(t)$)
2. **Which type** of event occurs (governed by $\pi_d(t)$)

#### 2.1.2 What We Want to Control

From the likelihood factorization, we want to control:

1. **Wiggliness of the total hazard** $H(t) = \sum_d h_d(t)$
   - This affects the survival curve $S(t) = \exp(-\int_0^t H(u) du)$
   - Wiggly $H(t)$ produces wiggly survival estimates

2. **Wiggliness of the cause allocation probabilities** $\pi_d(t) = h_d(t)/H(t)$
   - These are the time-varying multinomial probabilities
   - Wiggly $\pi_d(t)$ means the cause mix changes erratically over time

### 2.2 B-Spline Representation

With shared knots across all $D$ causes, each hazard is represented as:
$$h_d(t) = \sum_{k=1}^K \beta_{kd} B_k(t), \quad \beta_{kd} > 0$$

The coefficients form a matrix $\mathbf{B} \in \mathbb{R}^{K \times D}$. Vectorizing column-wise: $\boldsymbol{\beta} = \text{vec}(\mathbf{B}) \in \mathbb{R}^{KD}$.

The curvature penalty matrix $S \in \mathbb{R}^{K \times K}$ is:
$$S_{ij} = \int_0^\tau B_i''(t) B_j''(t) \, dt$$

### 2.3 Total Hazard Curvature Penalty

The total hazard is:
$$H(t) = \sum_{d=1}^D h_d(t) = \sum_{k=1}^K \left(\sum_{d=1}^D \beta_{kd}\right) B_k(t)$$

Let $\boldsymbol{\beta}_H = \sum_{d=1}^D \boldsymbol{\beta}_d$ be the coefficient vector for the total hazard.

**Proposition 2.1** (Total Hazard Curvature Penalty).
$$\mathcal{P}_H = \int_0^\tau (H''(t))^2 dt = \boldsymbol{\beta}_H^T S \boldsymbol{\beta}_H = \boldsymbol{\beta}^T (\mathbf{1}_D \mathbf{1}_D^T \otimes S) \boldsymbol{\beta}$$

*Proof*: The matrix $\mathbf{1}_D \mathbf{1}_D^T$ has all entries equal to 1, so:
$$\boldsymbol{\beta}^T (\mathbf{1}_D \mathbf{1}_D^T \otimes S) \boldsymbol{\beta} = \sum_{d,d'} \boldsymbol{\beta}_d^T S \boldsymbol{\beta}_{d'} = \left(\sum_d \boldsymbol{\beta}_d\right)^T S \left(\sum_d \boldsymbol{\beta}_d\right) = \boldsymbol{\beta}_H^T S \boldsymbol{\beta}_H \;\;\square$$

### 2.4 Cause Allocation Smoothness (Indirect Control)

The cause allocation $\pi_d(t) = h_d(t)/H(t)$ is **nonlinear** in $\boldsymbol{\beta}$, so there is no exact quadratic penalty for its curvature.

**Key insight**: When are $\pi_d(t)$ smooth?

If all hazards have the **same shape** up to scale, i.e., $h_d(t) = c_d \cdot g(t)$ for constants $c_d$ and common shape $g(t)$, then:
$$\pi_d(t) = \frac{c_d \cdot g(t)}{\sum_{d'} c_{d'} \cdot g(t)} = \frac{c_d}{\sum_{d'} c_{d'}} = \text{constant}$$

The cause allocation is **maximally smooth** (constant in time) when hazards share a common shape.

**Deviation curvature penalty**: To control shape heterogeneity, we penalize deviations from the mean hazard:
$$\mathcal{P}_\pi = \sum_{d=1}^D \int_0^\tau ((h_d - \bar{h})''(t))^2 dt = \boldsymbol{\beta}^T (C_D \otimes S) \boldsymbol{\beta}$$

where $\bar{h}(t) = \frac{1}{D}\sum_d h_d(t)$ is the mean hazard and $C_D = I_D - \frac{1}{D}\mathbf{1}_D\mathbf{1}_D^T$ is the centering matrix.

**Proposition 2.2**. *When $\mathcal{P}_\pi$ is small, hazards have similar curvature, which implies the ratios $\pi_d(t)$ change slowly.*

*Heuristic argument*: Differentiating $\pi_d = h_d/H$:
$$\pi_d'(t) = \frac{h_d'(t) - \pi_d(t) H'(t)}{H(t)}$$
This is small when $h_d'(t) \approx \pi_d(t) H'(t)$, i.e., when each hazard changes proportionally to the total. Similarly for higher derivatives—when all hazards wiggle together, the ratios are stable.

**Note**: This is an *indirect* control. The deviation penalty penalizes shape heterogeneity, which typically correlates with ratio smoothness, but the relationship is not exact.

### 2.5 The Fundamental Decomposition

**Lemma 2.3** (Spectral Decomposition of $I_D$).
$$I_D = \frac{1}{D} \mathbf{1}_D \mathbf{1}_D^T + C_D$$

**Theorem 2.4** (Likelihood-Motivated Penalty Decomposition).
*The independent penalty decomposes as:*
$$\mathcal{P}_{\text{ind}} = \frac{1}{D}\mathcal{P}_H + \mathcal{P}_\pi$$

*Equivalently:*
$$\boldsymbol{\beta}^T (I_D \otimes S) \boldsymbol{\beta} = \frac{1}{D}\boldsymbol{\beta}^T (\mathbf{1}_D \mathbf{1}_D^T \otimes S) \boldsymbol{\beta} + \boldsymbol{\beta}^T (C_D \otimes S) \boldsymbol{\beta}$$

*Or in integral form:*
$$\sum_d \int (h_d'')^2 = \frac{1}{D} \int (H'')^2 + \sum_d \int ((h_d - \bar{h})'')^2$$

**Interpretation**: The standard "independent" penalty on individual hazard curvatures implicitly penalizes **both**:
- The curvature of the total hazard (survival smoothness)
- The curvature of deviations (cause allocation smoothness)

with relative weight $1/D$ on the total hazard component.

### 2.6 The Three Penalty Modes

| Mode | Penalty Matrix | Controls | Likelihood Component |
|:-----|:---------------|:---------|:--------------------|
| **Independent** | $I_D \otimes S$ | $\sum_d \int (h_d'')^2$ | Both survival and cause allocation (balanced) |
| **Total Hazard** | $\mathbf{1}_D\mathbf{1}_D^T \otimes S$ | $\int (H'')^2$ | Survival smoothness only |
| **Deviation** | $C_D \otimes S$ | $\sum_d \int ((h_d - \bar{h})'')^2$ | Cause allocation smoothness (indirect) |

**Decomposition**: $I_D \otimes S = \frac{1}{D}(\mathbf{1}_D\mathbf{1}_D^T \otimes S) + (C_D \otimes S)$

### 2.7 General Penalty Formulation

The general penalty allows separate control of the two components:

$$\mathcal{P} = \lambda_H \int (H''(t))^2 dt + \lambda_\pi \sum_d \int ((h_d - \bar{h})''(t))^2 dt$$

$$= \lambda_H \boldsymbol{\beta}^T (\mathbf{1}_D\mathbf{1}_D^T \otimes S) \boldsymbol{\beta} + \lambda_\pi \boldsymbol{\beta}^T (C_D \otimes S) \boldsymbol{\beta}$$

**Special cases**:
- $\lambda_H = \lambda/D$, $\lambda_\pi = \lambda$: Recovers independent penalty $\lambda(I_D \otimes S)$
- $\lambda_H > 0$, $\lambda_\pi = 0$: Pure total hazard penalty (maximum flexibility in cause allocation)
- $\lambda_H = 0$, $\lambda_\pi > 0$: Pure deviation penalty (no constraint on total hazard smoothness)

### 2.8 Adversarial Review

#### 2.8.1 Is the Deviation Penalty Really About Cause Allocation?

**Challenge**: The claim that $\mathcal{P}_\pi$ controls cause allocation smoothness is heuristic. Can we make it precise?

**Analysis**: No exact relationship exists because $\pi_d(t)$ is nonlinear. However:
- If $\mathcal{P}_\pi = 0$, all hazards have identical curvature → $\pi_d(t)$ is exactly constant
- As $\mathcal{P}_\pi$ increases, hazards can have different curvatures → $\pi_d(t)$ can vary more rapidly

The deviation penalty provides a **sufficient but not necessary** condition for smooth cause allocation.

#### 2.8.2 Why Not Penalize $\pi_d''$ Directly?

**Challenge**: Why not compute $\int (\pi_d'')^2$ and penalize that?

**Analysis**: This would require:
1. Nonlinear penalty term (not quadratic in $\boldsymbol{\beta}$)
2. Numerical integration at each optimization step
3. Complex gradient/Hessian computation

The deviation penalty provides a computationally tractable approximation that captures the key behavior.

#### 2.8.3 Scale Invariance

**Challenge**: If $h_1 \sim O(1)$ and $h_2 \sim O(100)$, does the penalty structure break down?

**Analysis**: The total hazard penalty $\int (H'')^2$ will be dominated by the large hazard. The deviation penalty penalizes $(h_d - \bar{h})''$, so:
- For the large hazard: $h_2 - \bar{h} \approx h_2(1 - 1/D)$ (still large)
- For the small hazard: $h_1 - \bar{h} \approx -h_2/D$ (also driven by large hazard)

**Recommendation**: With highly unequal hazards, consider per-cause smoothing parameters or log-scale modeling.

#### 2.8.4 Ordering Invariance

**Challenge**: Is the decomposition invariant to cause ordering?

**Analysis**: Yes. Both $\mathbf{1}_D\mathbf{1}_D^T$ and $C_D$ are permutation-invariant (all-ones matrix and centering matrix are symmetric under index permutation).

### 2.9 Baseline Hazard Model

For transition $r \to s$:

$$h_{rs}(t) = \sum_{i=1}^k \beta_{is} B_i(t), \quad \beta_{is} > 0$$

where:
- $B_i(t)$: B-spline basis functions (order 4 = cubic)
- $\beta_{is}$: Positive coefficients (enforced via box constraints or reparameterization)
- $k$: Number of basis functions

**Cumulative hazard** (closed form):
$$H_{rs}(t) = \sum_{i=1}^k \beta_{is} \int_0^t B_i(u) \, du$$

The integrated basis is computed via `BSplineKit.integral()`, which returns an order-(k+1) spline whose evaluation gives the antiderivative.

### 2.10 Knot Placement

For each origin state $r$:
1. Pool event times from all transitions $r \to s_1, r \to s_2, \ldots$
2. Place interior knots at quantiles of pooled times
3. All competing hazards from $r$ share the same knots

This ensures a consistent time grid without assuming similar hazard shapes.

**Rationale for Shared Knots**: The Kronecker product structure $(P_D \otimes S)$ requires that $S$ be the **same matrix** for all destinations. If bases differ, the penalty cannot be expressed as a single Kronecker product, breaking the mathematical framework.

**Implementation note**: The existing knot placement helpers need to be updated to support this pooling behavior.

---

## 3. User API

### 3.1 Hazard Specification

```julia
# Spline baseline hazard (intercept-only formula)
h12 = Hazard(@formula(0 ~ 1), :sp, 1, 2)
h13 = Hazard(@formula(0 ~ 1), :sp, 1, 3)
h21 = Hazard(@formula(0 ~ 1), :sp, 2, 1)

model = MultistateModel(h12, h13, h21, ...)
```

The `:sp` family indicates a spline baseline hazard.

### 3.2 Penalty Configuration

Penalties are configured at the model level using a **rule-based API**. Rules are applied from general to specific.

**Validation Note**: When `share_lambda=true` or `total_hazard=true` is used for a set of competing risks, the constructor **must enforce** that all involved hazards share the exact same knot locations. If knots differ, the constructor will throw an error.

```julia
"""
    SplinePenalty(selector=:all; order=2, total_hazard=false, share_lambda=false)

Configure spline penalty for baseline hazards.

# Arguments
- `selector`: Which hazards this rule applies to.
  - `:all` — All spline hazards (default for global settings)
  - `r::Int` — All hazards from origin state `r`
  - `(r, s)::Tuple{Int,Int}` — Specific transition `r → s`

- `order::Int=2`: Derivative to penalize (1=slope, 2=curvature, 3=change in curvature).
- `total_hazard::Bool=false`: Penalize smoothness of total hazard out of this origin.
- `share_lambda::Bool=false`: Share λ across competing hazards from same origin.
"""
struct SplinePenalty
    selector::Union{Symbol, Int, Tuple{Int,Int}}
    order::Int
    total_hazard::Bool
    share_lambda::Bool
end

# Convenience constructors
SplinePenalty(; kwargs...) = SplinePenalty(:all; kwargs...)
SplinePenalty(sel; order=2, total_hazard=false, share_lambda=false) = ...
```

### 3.3 Usage Examples

**Example 1: Defaults (just works)**
```julia
fit(model, data)
# Equivalent to: penalty = SplinePenalty()
# - Order 2 (curvature penalty)
# - Independent λ per hazard
# - No total hazard penalty
```

**Example 2: Global change**
```julia
fit(model, data; penalty = SplinePenalty(order=3))
# Stiffer penalty (penalize 3rd derivative) for all hazards
```

**Example 3: Per-origin configuration**
```julia
fit(model, data; penalty = [
    SplinePenalty(1, share_lambda=true, total_hazard=true),
    SplinePenalty(2, share_lambda=false)
])
# Origin 1: Share λ across 1→2 and 1→3; penalize total hazard
# Origin 2: Independent λ; no total hazard penalty
```

**Example 4: Fine-grained control**
```julia
fit(model, data; penalty = [
    SplinePenalty(order=2),                   # Global default
    SplinePenalty(1, total_hazard=true),      # Origin 1: add total hazard penalty
    SplinePenalty((1, 2), order=1)            # Transition 1→2: penalize 1st derivative (flatness)
])
```

### 3.4 Resolution Order

For a transition $r \to s$, settings are resolved by specificity:
1. **Transition rule** `(r, s)` — highest priority
2. **Origin rule** `r`
3. **Global rule** `:all`
4. **System defaults** — `order=2, total_hazard=false, share_lambda=false`

---

## 4. Data Structures

### 4.1 SplinePenalty (User-Facing)

```julia
struct SplinePenalty
    selector::Union{Symbol, Int, Tuple{Int,Int}}
    order::Int
    total_hazard::Bool
    share_lambda::Bool
    
    function SplinePenalty(selector=:all; order=2, total_hazard=false, share_lambda=false)
        order >= 1 || throw(ArgumentError("order must be >= 1"))
        new(selector, order, total_hazard, share_lambda)
    end
end
```

**Note**: `order` can be any positive integer supported by BSplineKit. For cubic splines (order 4), meaningful values are 1-3; higher orders give zero penalty.

### 4.2 SplineHazardInfo (Internal)

Stored per hazard after model construction.

```julia
struct SplineHazardInfo
    origin::Int
    dest::Int
    k::Int                          # Number of basis functions
    breakpoints::Vector{Float64}    # Knot locations (shared by origin)
    basis::BSplineBasis             # BSplineKit basis object
    S::Matrix{Float64}              # Penalty matrix ∫B''B''ᵀdt
    integrated_basis::Function      # ∫₀ᵗ B(u)du for cumulative hazard
end
```

### 4.3 PenaltyConfig (Internal)

Resolved penalty configuration per origin.

```julia
struct OriginPenaltyConfig
    origin::Int
    order::Int
    total_hazard::Bool
    share_lambda::Bool
    transitions::Vector{Tuple{Int,Int}}  # (origin, dest) pairs
    lambda_indices::Vector{Int}          # Index into global λ vector
    total_lambda_index::Union{Int, Nothing}
end
```

---

## 5. Implementation Phases

### Phase 0: Dependencies & Setup

- Add `BSplineKit.jl` to `Project.toml`
- Verify compatibility with existing hazard infrastructure

### Phase 1: Baseline Hazard Splines

**Goal**: Working `:sp` hazard family for Markov models.

**Tasks**:
1. Define `SplineHazard` type and integrate with hazard dispatch
2. Implement `place_knots(origin, event_times; k)` — pools by origin
3. Implement `build_penalty_matrix(basis, order)` — computes $\int B^{(m)} B^{(m)T} dt$
4. Implement `integrated_basis(basis)` — for cumulative hazard
5. Implement positivity constraints (box constraints or $\beta = \exp(\gamma)$)
6. Update `fit!` to handle `SplinePenalty` rules and penalized objective
7. Implement PIJCV for λ selection

**Deliverable**: Fit spline hazards with configurable penalties.

### Phase 2: MCEM & Semi-Markov Support

- Integrate penalty into MCEM M-step
- Validate on panel data simulations

### Phase 3: Covariate Splines (Future)

- `s(x)` syntax for smooth covariate effects
- `te(x, y)` tensor products
- Separate specification document

---

## 6. Validation Strategy

### 6.1 Unit Tests

- **Basis correctness**: Verify B-spline evaluation against known functions
- **Penalty matrix**: Check $S$ is positive semi-definite with correct null space (polynomials of degree < m)
- **Cumulative hazard**: Verify $H(t) = \int_0^t h(u) du$ via numerical integration
- **Positivity**: Confirm $h(t) > 0$ for all $t$ when $\beta > 0$
- **Edge cases**: $k=4$ (minimum cubic), $\lambda \to \infty$ (converges to polynomial)

### 6.2 Simulation Studies

- **Scenario A**: Recover Weibull hazard using `:sp` — compare to parametric fit
- **Scenario B**: Recover non-monotonic hazard (bathtub curve)
- **Scenario C**: Competing risks with shared vs independent λ
- **Scenario D**: Total hazard penalty effect on cause-specific estimates

### 6.3 Performance

- Benchmark likelihood evaluation (critical for MCEM)
- Monitor allocations in hot paths

---

## 7. Resolved Design Decisions

1. **Positivity enforcement**: Log-scale parameterization (existing approach)
   - Optimization uses $\gamma$ (unconstrained)
   - `ests2coefs` transforms to positive $\beta = \exp(\gamma)$ for spline evaluation
   - Penalty applies to natural-scale coefficients: $\mathcal{P} = \lambda \beta^T S \beta$

2. **Derivative order**: Any $m \geq 1$ supported
   - BSplineKit `Derivative(m)` verified for $m \in \{1, 2, 3\}$ on cubic splines
   - Null space dimension = $m$ (polynomials of degree $< m$)

3. **Cumulative hazard**: `BSplineKit.integral()` for closed form
   - Verified: returns antiderivative spline
   - $H(t) = \texttt{integral}(h)(t) - \texttt{integral}(h)(0)$

## 8. Open Questions

1. **Boundary behavior**: Natural spline constraints (force $h''(0) = h''(T) = 0$)?
   - Pro: Reduces boundary wiggle
   - Con: Additional constraints to implement
   - Current implementation supports `natural_spline=true` option

2. **Default k**: How many basis functions by default?
   - mgcv uses k=10 for most smooths
   - Current: `floor(n^(1/5))` based on sieve estimation theory

3. **fit() integration**: How to pass penalty to likelihood and select λ?
   - See Sections 9-10 (fit() Integration & PIJCV Algorithm)

---

## 9. fit() Integration Design

### 9.1 Penalty LocatB — penalty stored in model at construction, overridable at fit time.

```julia
# Model construction with default penalty
model = multistatemodel(h12, h13; data=data, penalty=SplinePenalty())

# Use model penalty
fit(model)

# Override at fit time (returns model with updated penalty config)
fit(model; penalty=SplinePenalty(order=3))
```

**Storage**: The `MultistateModel` struct will gain a `penalty_config` field.
**Type Stability**: Spline coefficients will be stored within the existing `VectorOfVectors` or `ComponentArray` parameter structure to ensure type stability and compatibility with `ElasticArrays`.verride at fit time
fit(model; penalty=SplinePenalty(order=3))
```

### 9.2 Overloaded loglik Functions

Extend existing loglik functions with optional `penalty` argument:

```julia
# Current signature
loglik_exact(parameters, data::ExactData; neg=true, ...)

# Extended signature  
loglik_exact(parameters, data::ExactData; neg=true, penalty=nothing, ...)
```

When `penalty !== nothing`, the penalized negative log-likelihood is:
$$-\ell_p(\beta; \lambda) = -\ell(\beta) + \frac{1}{2}\sum_j \lambda_j \beta^T S_j \beta$$

### 9.3 PenaltyConfig Struct

```julia
struct PenaltyConfig
    terms::Vector{PenaltyTerm}                    # One per λ
    total_hazard_terms::Vector{TotalHazardTerm}   # Optional
end

struct PenaltyTerm
    hazard_indices::UnitRange{Int}   # Indices into flat parameter vector
    S::Matrix{Float64}               # Penalty matrix
    λ::Float64                       # Current smoothing parameter
    order::Int                       # Derivative order
end
```

### 9.4 Penalty Computation

```julia
function compute_penalty(β_natural, penalty::PenaltyConfig)
    total = 0.0
    for term in penalty.terms
        βj = β_natural[term.hazard_indices]
        total += term.λ * dot(βj, term.S * βj)
    end
    return total / 2
end
```

---

## 10. PIJCV Algorithm (Wood 2024)

Predictive Infinitesimal Jackknife Cross-Validation (PIJCV) efficiently selects smoothing parameters using a Newton-step approximation to leave-one-out CV. This is Wood's "Neighbourhood Cross-Validation" applied to the subject-level (leave-one-subject-out).

### 10.1 Key Distinction: Penalized vs Unpenalized

**Critical insight from Wood (2024):**

| Optimization | Objective | Purpose |
|:-------------|:----------|:--------|
| **Inner** (fit β given λ) | **Penalized** loss: $\sum_i D(y_i, \theta_i) + \sum_j \lambda_j \beta^T S_j \beta$ | Regularized estimation |
| **Outer** (select λ) | **Unpenalized** loss on held-out: $V = \sum_i D(y_i, \theta_i^{-i})$ | Prediction quality |

The PIJCV criterion $V$ evaluates **unpenalized** prediction error using coefficients estimated from the **penalized** model. This is because:
- The penalty is a regularizer during estimation
- Model quality is judged by prediction of new data (no penalty involved)

### 10.2 Core Formula

For subject $i$, the coefficient change on omission is approximated by one Newton step:
$$\Delta^{-i} = H_{\lambda,-i}^{-1} g_i$$

where:
- $H_{\lambda,-i} = H_\lambda - H_i$ is the **penalized** Hessian with subject $i$'s contribution removed
- $g_i$ is the **unpenalized** gradient contribution from subject $i$
- $H_\lambda = H + \sum_j \lambda_j S_j$ is the full **penalized** Hessian

The approximation $\hat\beta^{-i} = \hat\beta - \Delta^{-i}$ has error $O(p^3/n^2)$.

### 10.3 PIJCV Criterion

$$V(\lambda) = \sum_{i=1}^n D(y_i, \theta_i^{-i})$$

where $D$ is the **unpenalized** negative log-likelihood and $\theta_i^{-i}$ is the prediction for subject $i$ using $\hat\beta^{-i}$.

### 10.4 Algorithm

**Cost**: $O(np^2)$ — same as single model fit.

```
outer loop: BFGS on ρ = log(λ) to minimize V(λ)
    for each trial ρ:
        1. Inner optimization: minimize penalized loss → β̂
           (warm start from previous β̂)
        2. Compute penalized Hessian H_λ = H + Σ_j λ_j S_j
        3. Cholesky factorize: R^T R = H_λ
        4. For each subject i:
           a. Compute unpenalized gradient g_i and Hessian H_i
           b. Downdate Cholesky: R_i from H_{λ,-i} = H_λ - H_i
           c. Solve: Δ^{-i} = H_{λ,-i}^{-1} g_i
           d. Evaluate: D(y_i, θ_i^{-i}) using β̂ - Δ^{-i}
        5. Sum: V(λ) = Σ_i D(y_i, θ_i^{-i})
        6. Compute ∂V/∂ρ for BFGS
```

**Key derivative**: $\partial\hat\beta/\partial\rho_j = -\lambda_j H_\lambda^{-1} S_j \hat\beta$

### 10.5 Survival Data Adaptation

For multistate models:
- **Neighbourhood** $\alpha(i)$ = subject $i$ (all observations for that subject)
- **Unpenalized loss** $D$ = subject $i$'s contribution to negative log-likelihood
- **Prediction** $\theta_i^{-i}$ = hazard/survival predictions using $\hat\beta^{-i}$

This naturally handles within-subject correlation in panel data.

### 10.6 Implementation Requirements

### 10.7 Fallback Strategy (GCV)

If PIJCV fails to converge or produces degenerate smoothing parameters (e.g., due to sparse data or flat likelihoods), the optimization routine will fall back to **Generalized Cross-Validation (GCV)**.

**GCV Criterion**:
$$V_{GCV}(\lambda) = \frac{n \|\mathbf{y} - \hat{\boldsymbol{\mu}}\| ^2}{(n - \text{tr}(\mathbf{A}))^2}$$

where $\mathbf{A}$ is the influence (hat) matrix. For non-Gaussian models, this is approximated using the deviance and the effective degrees of freedom ($\text{edf} = \text{tr}(\mathbf{A})$).

**Implementation**:
1. Attempt PIJCV optimization.
2. If failure detected (NaNs, non-convergence, or $\lambda \to 0$ with wiggly fit):
   - Switch objective function to GCV.
   - Re-optimize $\lambda$.
3. If GCV also fails, warn user and default to a stiff penalty (high $\lambda$).
2. **Subject-level Hessians**: Already have `compute_subject_hessians()`
3. **Cholesky downdate**: Need to implement rank-k update/downdate
4. **Separate penalized/unpenalized evaluation**: Need both in loglik functions

---

## 11. References

- Wood, S.N. (2017). *Generalized Additive Models: An Introduction with R* (2nd ed.). CRC Press.
- Wood, S.N. (2024). Neighbourhood Cross-Validation. *arXiv:2404.16490*.
- Eilers, P.H.C. & Marx, B.D. (1996). Flexible smoothing with B-splines and penalties. *Statistical Science*.
- Royston, P. & Parmar, M.K.B. (2002). Flexible parametric proportional-hazards and proportional-odds models. *Statistics in Medicine*.
