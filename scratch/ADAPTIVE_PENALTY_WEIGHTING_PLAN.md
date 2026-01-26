# Adaptive Penalty Weighting Implementation Plan

**Document Version**: 2.0  
**Created**: 2026-01-25  
**Updated**: 2026-01-25  
**Status**: Decisions Finalized  
**Branch**: `penalized_splines`

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Mathematical Formulation](#2-mathematical-formulation)
3. [API Design](#3-api-design)
4. [Implementation Architecture](#4-implementation-architecture)
5. [Function Specifications](#5-function-specifications)
6. [MCEM Considerations](#6-mcem-considerations)
7. [Data Flow](#7-data-flow)
8. [File Changes](#8-file-changes)
9. [Testing Strategy](#9-testing-strategy)
10. [Implementation Phases](#10-implementation-phases)
11. [Adversarial Review](#11-adversarial-review)
12. [Open Questions](#12-open-questions)

---

## 1. Executive Summary

### Goal

Implement **adaptive penalty weighting** that allows the penalty strength to vary across time, with weights learned from data characteristics (at-risk counts, Fisher information).

### Key Features

1. **At-Risk Weighting**: Penalize more heavily where fewer subjects are at risk
2. **Single Parameter**: One scalar α controls adaptation strength (w(t) = Y(t)^(-α))
3. **All Fitting Modes**: Works with exact data, Markov panel, and MCEM
4. **Per-Transition**: Different α per transition, with share_lambda implying share_alpha

### Core Formula

Instead of uniform penalty:
$$P(\beta) = \lambda \int_0^T [(\log h)''(t)]^2 dt = \lambda \beta^T S \beta$$

Use weighted penalty:
$$P(\beta; \alpha) = \lambda \int_0^T w(t; \alpha) [(\log h)''(t)]^2 dt = \lambda \beta^T S_w(\alpha) \beta$$

where $w(t; \alpha)$ is parameterized by regression coefficients $\alpha$.

---

## 2. Mathematical Formulation

### 2.1 Weight Function Parameterization

**At-Risk Weighting (single parameter):**
$$w(t; \alpha) = Y(t)^{-\alpha}$$

where:
- $Y(t)$ is the number at risk at time $t$
- $\alpha \geq 0$ controls adaptation strength:
  - $\alpha = 0$: Uniform weighting (standard P-spline)
  - $\alpha = 1$: Weight proportional to $1/Y(t)$
  - $\alpha > 1$: Stronger adaptation (rarely needed)

**Why single parameter?** The baseline weight (previously $\alpha_0$) is absorbed by $\lambda$, so only the power $\alpha$ matters.

**Default**: $\alpha = 1.0$ when `adaptive_weight=:atrisk` is specified.

### 2.2 Weighted Penalty Matrix Construction

For B-splines $B_1(t), ..., B_K(t)$ with penalty on second derivative:

$$S_w = \int_0^T w(t) \cdot B''(t) B''(t)^T dt$$

**GPS-style construction** (preferred):

The GPS algorithm computes $D_m = W_m^{-1} \Delta W_{m-1}^{-1} \Delta \cdots$ using knot spacing weights. For adaptive weighting:

1. Compute $w_j = w(t_j^*)$ at knot midpoints $t_j^* = (t_j + t_{j+1})/2$
2. Modify GPS weights: $\tilde{W}_j = W_j / \sqrt{w_j}$ (distribute weight across iterations)
3. Or: Post-multiply $S_w = W_{diag} \cdot D^T D \cdot W_{diag}$ with $W_{diag} = \text{diag}(\sqrt{w_1}, ..., \sqrt{w_K})$

**Integral construction** (more accurate):

$$S_w[i,j] = \int_0^T w(t) \cdot B_i''(t) B_j''(t) dt$$

Use Gauss-Legendre quadrature on each knot interval with weight function evaluated at quadrature points.

### 2.3 At-Risk Count Computation

**For exact data** (obstype=1):
$$Y(t) = \sum_{i=1}^n \mathbf{1}[\text{subject } i \text{ at risk for transition at time } t]$$

For a specific transition $r \to s$:
$$Y_{r \to s}(t) = \sum_{i=1}^n \mathbf{1}[\text{subj } i \text{ in state } r \text{ at time } t]$$

**For panel data** (obstype=2):

At-risk counts are not directly observable—we only observe states at discrete times. Options:

1. **Upper bound**: Count subjects who *could* be in the origin state
2. **Expected counts**: Use surrogate model to compute $E[Y(t) | \text{observed data}]$
3. **Interpolation**: Linear interpolation between observation times

### 2.4 MCEM Path-Weighted At-Risk Counts

In MCEM, we have sampled paths $\{Z_{ij}\}$ with importance weights $\{w_{ij}\}$.

**Weighted at-risk count at time $t$:**
$$\tilde{Y}(t) = \sum_{i=1}^n \sum_{j=1}^{J_i} w_{ij} \cdot \mathbf{1}[\text{path } Z_{ij} \text{ in origin state at time } t]$$

where $w_{ij}$ are normalized importance weights for subject $i$, path $j$.

**Issue**: Weights change each MCEM iteration as surrogate is updated.

**Solution**: Recompute $S_w$ at each MCEM iteration (adds overhead) OR use fixed weights from final surrogate.

---

## 3. API Design

### 3.1 User-Facing API

```julia
# Current API (unchanged for backward compatibility)
SplinePenalty()  # Uniform weighting
SplinePenalty(order=2)

# New: at-risk adaptive weighting
SplinePenalty(adaptive_weight=:atrisk)              # At-risk weighting, α=1.0
SplinePenalty(adaptive_weight=:atrisk, alpha=0.5)   # Custom α value
SplinePenalty(adaptive_weight=:atrisk, learn_alpha=true)  # Estimate α from data
SplinePenalty(adaptive_weight=:none)                # Explicit uniform (default)

# Per-transition specification (uses selector)
SplinePenalty((1,2), adaptive_weight=:atrisk, alpha=1.0)  # Only 1→2 transition
SplinePenalty((1,3), adaptive_weight=:atrisk, alpha=0.5)  # Different α for 1→3

# Interaction with share_lambda
SplinePenalty(share_lambda=true, adaptive_weight=:atrisk)  # Shared λ implies shared α
```

**Design choice**: Extend SplinePenalty for simplicity.

### 3.2 Internal Types

```julia
"""
    PenaltyWeighting

Abstract type for penalty weight specifications.
"""
abstract type PenaltyWeighting end

"""
    UniformWeighting <: PenaltyWeighting

No time-varying weights (standard P-spline).
"""
struct UniformWeighting <: PenaltyWeighting end

"""
    AtRiskWeighting <: PenaltyWeighting

Weight penalty by inverse at-risk count: w(t) = Y(t)^(-α)

# Fields
- `alpha::Float64`: Power on at-risk count (default 1.0)
- `learn::Bool`: Whether to estimate α from data via marginal likelihood
"""
struct AtRiskWeighting <: PenaltyWeighting
    alpha::Float64
    learn::Bool
    
    function AtRiskWeighting(; alpha=1.0, learn=false)
        alpha >= 0 || throw(ArgumentError("alpha must be non-negative"))
        new(alpha, learn)
    end
end
```

### 3.3 Modified SplinePenalty

```julia
struct SplinePenalty
    selector::Union{Symbol, Int, Tuple{Int,Int}}
    order::Int
    total_hazard::Bool
    share_lambda::Bool
    share_covariate_lambda::Union{Bool, Symbol}
    weighting::PenaltyWeighting  # NEW FIELD
    
    function SplinePenalty(selector=:all; 
                           order=2, 
                           total_hazard=false,
                           share_lambda=false,
                           share_covariate_lambda=false,
                           adaptive_weight::Symbol=:none,  # :none or :atrisk
                           alpha::Float64=1.0,             # Power for at-risk weighting
                           learn_alpha::Bool=false)        # Estimate α from data
        # Construct appropriate PenaltyWeighting
        weighting = if adaptive_weight == :none
            UniformWeighting()
        elseif adaptive_weight == :atrisk
            AtRiskWeighting(alpha=alpha, learn=learn_alpha)
        else
            throw(ArgumentError("adaptive_weight must be :none or :atrisk"))
        end
        
        new(selector, order, total_hazard, share_lambda, share_covariate_lambda, weighting)
    end
end
```

**Note**: When `share_lambda=true`, penalties that share λ also share α.

---

## 4. Implementation Architecture

### 4.1 Component Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        User API                                  │
│  SplinePenalty(adaptive_weight=:atrisk, learn_weights=true)     │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                  build_penalty_config()                          │
│  - Resolves SplinePenalty rules                                  │
│  - Creates PenaltyWeighting objects                              │
│  - Computes at-risk counts (for :atrisk)                        │
│  - Builds weighted penalty matrix S_w                            │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              WeightedPenaltyTerm (new type)                      │
│  - Stores S_w (weighted penalty matrix)                          │
│  - Stores weighting specification                                │
│  - Stores at-risk function Y(t) or precomputed values            │
│  - Method: recompute_weights!(term, beta, model) for iterative   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Fitting Functions                               │
│  _fit_exact_penalized()  - For exact data                       │
│  _fit_markov_penalized() - For Markov panel data                │
│  _fit_mcem_penalized()   - For semi-Markov MCEM                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Key Design Decisions

1. **Penalty matrix is recomputable**: Store ingredients, not just final $S_w$
2. **At-risk counts computed once at setup** (for fixed α mode)
3. **Iterative recomputation for learn_alpha=true**
4. **MCEM: Recompute $S_w$ each iteration** with updated path weights
5. **share_lambda implies share_alpha**: Penalties grouped by λ share the same α
6. **Per-transition α**: Different transitions can have different adaptation strength

---

## 5. Function Specifications

### 5.1 At-Risk Count Functions

```julia
"""
    compute_atrisk_counts(model::MultistateProcess, 
                          eval_times::Vector{Float64},
                          transition::Tuple{Int,Int}) -> Vector{Float64}

Compute number at risk for a specific transition at given times.

# Arguments
- `model`: MultistateProcess with data
- `eval_times`: Times at which to evaluate Y(t)
- `transition`: (origin, destination) state pair

# Returns
- Vector of at-risk counts at each eval_time

# For Exact Data (obstype=1)
Counts subjects whose observed path has them in origin state at time t.

# For Panel Data (obstype=2)
Uses upper bound: counts subjects who *could* be in origin state based on
last observed state before t.
"""
function compute_atrisk_counts(model::MultistateProcess,
                               eval_times::Vector{Float64},
                               transition::Tuple{Int,Int})
    # Implementation details...
end

"""
    compute_atrisk_counts_mcem(paths::Vector{Vector{SamplePath}},
                               weights::Vector{Vector{Float64}},
                               eval_times::Vector{Float64},
                               transition::Tuple{Int,Int}) -> Vector{Float64}

Compute importance-weighted at-risk counts from MCEM sampled paths.

# Arguments
- `paths`: paths[i][j] = j-th sampled path for subject i
- `weights`: weights[i][j] = normalized importance weight for path i,j
- `eval_times`: Times at which to evaluate Y(t)
- `transition`: (origin, destination) state pair

# Returns
- Vector of weighted at-risk counts: Ỹ(t) = Σᵢ Σⱼ wᵢⱼ · 1[Zᵢⱼ in origin at t]
"""
function compute_atrisk_counts_mcem(paths::Vector{Vector{SamplePath}},
                                     weights::Vector{Vector{Float64}},
                                     eval_times::Vector{Float64},
                                     transition::Tuple{Int,Int})
    # Implementation details...
end
```

### 5.2 Weighted Penalty Matrix Construction

```julia
"""
    build_weighted_penalty_matrix(basis, order::Int, 
                                   weighting::AtRiskWeighting,
                                   atrisk_at_knots::Vector{Float64})
                                   -> Matrix{Float64}

Construct weighted penalty matrix S_w with at-risk weighting.

# Arguments
- `basis`: BSplineKit basis
- `order`: Penalty order (1=slope, 2=curvature)
- `weighting`: AtRiskWeighting with α parameter
- `atrisk_at_knots`: At-risk counts Y(t) at knot midpoints

# Returns
- S_w: Weighted penalty matrix (K × K)

# Method
Compute weights w_j = Y(t_j)^(-α) at knot midpoints, then use integral formulation
with Gauss-Legendre quadrature:
  S_w[i,j] = ∫ w(t) B_i^(m)(t) B_j^(m)(t) dt

Weight function w(t) is interpolated from knot midpoint values.

# Edge case
When Y(t) = 0, use Y(t) + 1 to avoid division by zero.
"""
function build_weighted_penalty_matrix(basis, order::Int,
                                        weighting::AtRiskWeighting,
                                        atrisk_at_knots::Vector{Float64})
    # Implementation details...
end
```

### 5.3 Alpha Parameter Learning

```julia
"""
    learn_alpha(model::MultistateProcess,
                penalty_config::QuadraticPenalty,
                beta::Vector{Float64},
                transition::Tuple{Int,Int};
                method::Symbol=:marginal_likelihood) -> Float64

Estimate optimal α for a specific transition by maximizing marginal likelihood.

# Arguments
- `model`: Fitted model
- `penalty_config`: Current penalty configuration  
- `beta`: Current coefficient estimates
- `transition`: (origin, dest) state pair
- `method`: `:marginal_likelihood` (REML-like criterion)

# Returns
- Optimal α value (scalar)

# Method
For fixed λ and β, maximize over α:
  p(y|λ,α) ∝ |H + λS_w(α)|^(-1/2) exp(-ℓ(β) - λ/2 β'S_w(α)β)

Uses 1D optimization (e.g., Brent's method) since α is scalar.

# Constraints
- α ∈ [0, 2] (rarely need α > 2)
- When share_lambda=true, returns single α for the group
"""
function learn_alpha(model::MultistateProcess,
                     penalty_config::QuadraticPenalty,
                     beta::Vector{Float64},
                     transition::Tuple{Int,Int};
                     method::Symbol=:marginal_likelihood)
    # Implementation details...
end
```

---

## 6. MCEM Considerations

### 6.1 Problem Statement

In MCEM, the observed data likelihood is:
$$p(Y|\theta) = \int p(Y,Z|\theta) dZ$$

We approximate the E-step expectation with importance sampling:
$$Q(\theta|\theta') \approx \sum_i \sum_j w_{ij} \log p(Y_i, Z_{ij}|\theta)$$

**Challenge**: At-risk counts depend on the latent paths $Z$, which are sampled.

### 6.2 Options for MCEM

**Option 1: Use expected at-risk from surrogate (Recommended)**

- Compute $E[Y(t)]$ under the Markov surrogate
- This is tractable via matrix exponentials
- Penalty matrix is fixed per MCEM iteration
- Recompute after surrogate update

**Option 2: Weighted average over sampled paths**

- $\tilde{Y}(t) = \sum_i \sum_j w_{ij} \cdot Y_{ij}(t)$
- Recompute each E-step
- Computationally expensive but more accurate

**Option 3: Upper bound from observed data**

- Ignore latent paths entirely
- Use panel observation intervals
- Least accurate but fastest

### 6.3 Implementation for MCEM

```julia
"""
    update_penalty_weights_mcem!(penalty_config::QuadraticPenalty,
                                  model::MultistateProcess,
                                  mcem_state::MCEMState)

Update weighted penalty matrices using MCEM E-step results.

Called after each E-step (every iteration) when using adaptive weighting with MCEM.
"""
function update_penalty_weights_mcem!(penalty_config::QuadraticPenalty,
                                       model::MultistateProcess,
                                       mcem_state::MCEMState)
    # Get paths and weights from MCEM state
    paths = mcem_state.sampled_paths
    weights = mcem_state.importance_weights
    
    for term in penalty_config.terms
        if term.weighting isa AtRiskWeighting
            # Compute weighted at-risk at knot midpoints
            knot_midpoints = get_knot_midpoints(term)
            transition = (term.origin, term.dest)
            
            atrisk = compute_atrisk_counts_mcem(paths, weights, knot_midpoints, transition)
            
            # Rebuild weighted penalty matrix
            term.S = build_weighted_penalty_matrix(term.basis, term.order, 
                                                    term.weighting, atrisk)
        end
    end
end
```

### 6.4 Computational Cost Analysis

| Operation | Uniform | At-Risk (fixed) | At-Risk (MCEM) |
|-----------|---------|-----------------|----------------|
| Setup | O(K³) | O(nK + K³) | O(nJK + K³) |
| Per iteration | 0 | 0 | O(nJK + K³) |
| Memory | O(K²) | O(K² + K) | O(K² + nJ) |

Where: K = basis size, n = subjects, J = paths per subject

For typical values (K=10, n=500, J=50), the per-iteration MCEM overhead is ~250K operations—likely acceptable.

---

## 7. Data Flow

### 7.1 Exact Data Flow

```
User: fit(model; penalty=SplinePenalty(adaptive_weight=:atrisk))
                    │
                    ▼
        _resolve_penalty(penalty, model)
                    │
                    ▼
        build_penalty_config(model, resolved_penalty)
                    │
        ┌───────────┴────────────┐
        │ For each spline hazard │
        └───────────┬────────────┘
                    │
                    ▼
        compute_atrisk_counts(model, knot_midpoints, transition)
                    │
                    ▼
        build_weighted_penalty_matrix(basis, order, weighting, atrisk)
                    │
                    ▼
        Create WeightedPenaltyTerm with S_w
                    │
                    ▼
        QuadraticPenalty(terms, ...)
                    │
                    ▼
        _fit_exact_penalized(model, penalty_config, ...)
                    │
                    ▼
        [If learn_weights=true: iterate until α converges]
```

### 7.2 MCEM Data Flow

```
fit(model; penalty=SplinePenalty(adaptive_weight=:atrisk))
                    │
                    ▼
        _fit_mcem(model, penalty_config)
                    │
        ┌───────────┴────────────┐
        │    MCEM Iteration k    │
        └───────────┬────────────┘
                    │
            ┌───────┴───────┐
            │   E-step      │
            │ Sample paths  │
            │ Compute wᵢⱼ   │
            └───────┬───────┘
                    │
                    ▼
        update_penalty_weights_mcem!(penalty_config, model, mcem_state)
                    │
                    ▼
            ┌───────────────┐
            │   M-step      │
            │ Optimize β    │
            │ with new S_w  │
            └───────┬───────┘
                    │
                    ▼
        [Repeat until convergence]
```

---

## 8. File Changes

### 8.1 New Files

| File | Purpose |
|------|---------|
| `src/utilities/penalty_weighting.jl` | PenaltyWeighting types, at-risk computation |
| `MultistateModelsTests/unit/test_penalty_weighting.jl` | Unit tests |

### 8.2 Modified Files

| File | Changes |
|------|---------|
| `src/types/penalties.jl` | Add `weighting` field to SplinePenalty |
| `src/utilities/penalty_config.jl` | Handle weighting in build_penalty_config |
| `src/utilities/spline_utils.jl` | Add weighted penalty matrix construction |
| `src/inference/fit_exact.jl` | Support weight learning iteration |
| `src/inference/fit_mcem.jl` | Call update_penalty_weights_mcem! |
| `src/inference/smoothing_selection.jl` | Handle weighted penalties in λ selection |
| `src/MultistateModels.jl` | Add include, exports |

### 8.3 Estimated Lines of Code

| Component | New Lines | Modified Lines |
|-----------|-----------|----------------|
| Types (penalty_weighting.jl) | 150 | 0 |
| At-risk computation | 200 | 0 |
| Weighted penalty matrix | 150 | 50 |
| MCEM integration | 100 | 50 |
| Weight learning | 200 | 0 |
| Tests | 400 | 0 |
| **Total** | **1200** | **100** |

---

## 9. Testing Strategy

### 9.1 Unit Tests

**At-risk count computation:**
```julia
@testset "compute_atrisk_counts" begin
    # Test 1: All subjects at risk at t=0
    # Test 2: Decreasing at-risk over time (censoring)
    # Test 3: Panel data upper bound
    # Test 4: MCEM weighted counts
end
```

**Weighted penalty matrix:**
```julia
@testset "build_weighted_penalty_matrix" begin
    # Test 1: Uniform weights recover standard GPS
    # Test 2: Higher weight = larger penalty contribution
    # Test 3: Symmetry and positive semi-definiteness
    # Test 4: Correct null space dimension
end
```

### 9.2 Integration Tests

**Exact data fitting:**
```julia
@testset "fit with adaptive weighting - exact" begin
    # Simulate Weibull data with heavy late censoring
    # Fit with uniform vs at-risk weighting
    # At-risk weighting should have narrower CI at late times
end
```

**MCEM fitting:**
```julia
@testset "fit with adaptive weighting - MCEM" begin
    # Panel data with semi-Markov process
    # Verify penalty matrix updates each iteration
    # Check convergence is not affected
end
```

### 9.3 Statistical Validation (Long Tests)

**Coverage simulation:**
```julia
# 500 replicates
# True Weibull hazard
# Heavy right censoring (50% censored after t=5)
# Compare 95% CI coverage: uniform vs at-risk weighting
# Expectation: at-risk improves late-time coverage without sacrificing early-time
```

### 9.4 Regression Tests

- Uniform weighting must reproduce existing behavior exactly
- No change to non-spline hazards
- Backward compatibility with old SplinePenalty() calls

---

## 10. Implementation Phases

### Phase 1: Core Types and At-Risk Computation (2-3 days)

1. Add PenaltyWeighting type hierarchy to `penalties.jl`
2. Add `weighting` field to SplinePenalty (default: UniformWeighting)
3. Implement `compute_atrisk_counts()` for exact data
4. Write unit tests for at-risk computation

**Deliverable**: At-risk counts work for exact data

### Phase 2: Weighted Penalty Matrix (2-3 days)

1. Implement `build_weighted_penalty_matrix()` in `spline_utils.jl`
2. Modify `build_penalty_config()` to handle weighting
3. Add WeightedPenaltyTerm or extend PenaltyTerm
4. Write unit tests for weighted penalty matrix

**Deliverable**: Weighted penalties work for exact data fitting

### Phase 3: Markov Panel Integration (1-2 days)

1. Implement `compute_atrisk_counts_panel()` for panel data (upper bound approach)
2. Modify `_fit_markov_penalized()` to support adaptive weighting
3. Write integration tests for Markov panel + adaptive weighting

**Deliverable**: Adaptive weighting works with Markov panel data

### Phase 4: MCEM Integration (2-3 days)

1. Implement `compute_atrisk_counts_mcem()` for path-weighted counts
2. Add `update_penalty_weights_mcem!()` to MCEM loop (every iteration)
3. Modify `_fit_mcem()` to call update function
4. Write integration tests for MCEM + adaptive weighting

**Deliverable**: Adaptive weighting works with MCEM

### Phase 5: Alpha Learning (3-4 days)

1. Implement `learn_alpha()` via marginal likelihood (1D optimization)
2. Add iteration loop to fitting functions
3. Handle convergence criteria and share_lambda grouping
4. Write tests for alpha learning

**Deliverable**: Full learn_alpha=true functionality

### Phase 6: Testing and Documentation (2-3 days)

1. Run full test suite
2. Long test for coverage validation  
3. Documentation strings
4. Example in docs/

**Total Estimated Time**: 13-19 days

---

## 11. Adversarial Review

### 11.1 Potential Issues Identified

**Issue 1: MCEM weight degeneracy affects at-risk estimates**

- Problem: If importance weights are degenerate (high Pareto-k), weighted at-risk may be unstable
- Risk: Penalty matrix becomes unreliable
- Mitigation:
  - Check Pareto-k before using weights
  - Fall back to uniform weights if k > 0.7
  - Use PSIS-smoothed weights (already implemented)

**Issue 2: At-risk = 0 at some times**

- Problem: $w(t) = Y(t)^{-\alpha}$ is infinite if $Y(t) = 0$
- Risk: Penalty matrix has infinities
- Mitigation:
  - Use $Y(t) + 1$ when $Y(t) = 0$ (floor at 1)
  - Document this edge case handling

**Issue 3: learn_alpha=true with MCEM is complex**

- Problem: Need to optimize over $(\beta, \lambda, \alpha)$ simultaneously
- Risk: Implementation complexity, convergence issues
- Mitigation:
  - α is scalar per transition → 1D optimization is fast
  - Alternating optimization: fix α, fit (β,λ), update α, repeat
  - Max 5 iterations typically sufficient

**Issue 4: Different transitions have different at-risk patterns**

- Problem: Transition 1→2 may have very different at-risk pattern than 1→3
- Solution: **Per-transition α is the default**
  - `SplinePenalty((1,2), adaptive_weight=:atrisk, alpha=1.0)`
  - `SplinePenalty((1,3), adaptive_weight=:atrisk, alpha=0.5)`
  - share_lambda=true groups share α automatically

**Issue 5: Smoothing parameter selection with weighted penalty**

- Problem: PIJCV/EFS formulas assume standard penalty structure
- Risk: Selection may be biased with weighted penalties
- Mitigation:
  - EFS/REML formulas still valid—they use the actual $S_w$ matrix
  - PIJCV uses LOO approximation—should work but needs verification
  - Add tests to confirm λ selection is sensible with weighted penalties

### 11.2 Design Alternatives Considered

**Alternative A: Weight as part of $\lambda$ grid search**

Instead of fixed $\alpha$, search over $(\lambda, \alpha)$ grid.

- Pro: No new optimization loop
- Con: Exponential growth in grid size
- Decision: Rejected—use optimization for $\alpha$

**Alternative B: Bayesian approach**

Put prior on $\alpha$ and integrate out.

- Pro: Fully principled uncertainty
- Con: Much more complex, requires MCMC
- Decision: Rejected for MVP—future work

**Alternative C: Non-parametric weight function**

Learn $w(t)$ as a separate spline.

- Pro: Maximum flexibility
- Con: Too many parameters, identifiability issues
- Decision: Rejected—regression parameterization is more robust

### 11.3 Missing Items in Plan

1. **Error messages**: Need clear messages when adaptive weighting fails
2. **Verbose output**: Should print α value and weight function diagnostics
3. **Plotting**: Utility to visualize $w(t) = Y(t)^{-\alpha}$ and $Y(t)$
4. **Parameter bounds**: α ∈ [0, 2] enforced (rarely need α > 2)

---

## 12. Decisions Made

### Finalized Design Choices

| Question | Decision |
|----------|----------|
| Parameter structure | Single scalar α (not vector) |
| Default α | 1.0 when `adaptive_weight=:atrisk` |
| MCEM update frequency | Every iteration |
| Scope | Exact + Markov panel + MCEM |
| Per-transition | Yes, different α per transition |
| share_lambda behavior | Implies share_alpha |
| Information-based weighting | **Removed** — not implementing |

---

## Appendix A: Future Work (Out of Scope)

### Aalen-Johansen Based Weighting

Once MultistateModels.jl supports Aalen-Johansen (AJ) and conditional Aalen-Johansen (cAJ) estimators, consider:

1. **AJ-based E[Y(t)]**: Weight by expected at-risk under AJ state occupation probabilities instead of observed counts. Potentially useful if modeling informative censoring in the future.

2. **cAJ-based weighting**: Conditional on covariate strata, may give better weighting for heterogeneous populations.

3. **Comparison study**: Empirically compare observed Y(t) vs AJ-based E[Y(t)] weighting in terms of coverage and MSE.

**Current decision**: Use observed Y(t) because it directly reflects actual information in the data, and the package does not currently model informative censoring.

---

## Appendix B: References

1. Wood, S.N. (2017). *Generalized Additive Models: An Introduction with R*. 2nd ed. CRC Press.
   - Section 4.1.3: Adaptive smoothing

2. Ruppert, D., Wand, M.P., & Carroll, R.J. (2003). *Semiparametric Regression*. Cambridge.
   - Chapter 13: Adaptive penalty methods

3. Li, Z. & Cao, J. (2022). "General P-Splines for Non-Uniform B-Splines." arXiv:2201.06808
   - GPS penalty matrix construction

4. Krivobokova, T. (2013). "Smoothing Parameter Selection in Two Frameworks for Penalized Splines." *JRSS-B* 75(4), 725-741.
   - Marginal likelihood for penalty parameter selection

---

**Document Status**: Decisions finalized, ready for implementation

**Next Steps**: 
1. ✅ Decisions finalized (2026-01-25)
2. Approve Phase 1 start
3. Begin implementation

