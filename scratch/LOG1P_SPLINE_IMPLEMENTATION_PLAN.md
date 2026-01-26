# Log(1+t) Time-Transformed Spline Implementation Plan

**Author:** GitHub Copilot  
**Date:** 2026-01-25  
**Status:** Draft for Review  
**Branch:** `penalized_splines`

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Mathematical Formulation](#2-mathematical-formulation)
3. [API Design](#3-api-design)
4. [Core Implementation](#4-core-implementation)
5. [Monotonicity Constraints](#5-monotonicity-constraints)
6. [Post-Processing and Output](#6-post-processing-and-output)
7. [Testing Infrastructure](#7-testing-infrastructure)
8. [Documentation Updates](#8-documentation-updates)
9. [Implementation Sequence](#9-implementation-sequence)
10. [Validation Criteria](#10-validation-criteria)
11. [File Change Summary](#11-file-change-summary)

---

## 1. Executive Summary

### Motivation

The Royston-Parmar flexible parametric survival model parameterizes splines on log-time, achieving parsimony for hazards that are linear on the log-time scale (e.g., Weibull). Currently, MultistateModels.jl models splines on linear time: $h(t) = \mathbf{B}(t)'\boldsymbol{\beta}$.

We propose adding a `time_scale` option that evaluates the spline basis at a transformed time $\tau = g(t)$, specifically $g(t) = \log(1+t)$. This transformation:

- Is well-defined at $t = 0$ (unlike $\log t$)
- Approximates $\log t$ for $t \gg 1$
- Preserves closed-form cumulative hazard computation
- Requires minimal changes to existing infrastructure

### Key Insight

For a monotonic transformation $\tau = g(t)$ with $g(0) = 0$, if we model hazard in transformed time as $h^*(\tau) = \mathbf{B}(\tau)'\boldsymbol{\beta}$, then:

$$h(t) = h^*(g(t)) \cdot g'(t)$$
$$H(t) = H^*(g(t))$$

The cumulative hazard is simply the integrated basis evaluated at $g(t)$—**no numerical quadrature required**.

---

## 2. Mathematical Formulation

### 2.1 Time Transformation Framework

Let $\tau = g(t)$ be a strictly increasing transformation with $g(0) = 0$.

**Definition (Transformed-Time Spline Hazard):**

In the transformed scale:
$$h^*(\tau) = \sum_{j=1}^{K} \beta_j B_j(\tau) = \mathbf{B}(\tau)'\boldsymbol{\beta}$$

where $B_j$ are B-spline basis functions and $\beta_j \geq 0$ are coefficients.

The cumulative hazard in transformed scale:
$$H^*(\tau) = \int_0^\tau h^*(u)\,du = \sum_{j=1}^{K} \beta_j I_j(\tau) = \mathbf{I}(\tau)'\boldsymbol{\beta}$$

where $I_j(\tau) = \int_0^\tau B_j(u)\,du$ is the integrated B-spline basis (computed analytically by BSplineKit).

### 2.2 Transformation to Original Time Scale

**Proposition:** For $h(t)$ the hazard in original time and $H(t)$ the cumulative hazard:

$$h(t) = h^*(g(t)) \cdot g'(t) = \mathbf{B}(g(t))'\boldsymbol{\beta} \cdot g'(t)$$

$$H(t) = H^*(g(t)) = \mathbf{I}(g(t))'\boldsymbol{\beta}$$

**Proof:** By the change of variables formula:
$$H(t) = \int_0^t h(u)\,du = \int_0^t h^*(g(u)) \cdot g'(u)\,du$$

Let $v = g(u)$, so $dv = g'(u)\,du$. When $u = 0$, $v = g(0) = 0$. When $u = t$, $v = g(t)$.

$$H(t) = \int_0^{g(t)} h^*(v)\,dv = H^*(g(t)) \quad \blacksquare$$

### 2.3 The log(1+t) Transformation

**Choice:** $g(t) = \log(1 + t)$

**Properties:**
- $g(0) = \log(1) = 0$ ✓ (well-defined at origin)
- $g'(t) = \frac{1}{1+t} > 0$ ✓ (strictly increasing)
- $g(t) \approx \log t$ for $t \gg 1$ ✓ (approximates Royston-Parmar for large $t$)
- $g(t) \approx t$ for $t \ll 1$ (linear near origin)

**Hazard in original time:**
$$h(t) = \frac{\mathbf{B}(\log(1+t))'\boldsymbol{\beta}}{1+t}$$

**Cumulative hazard in original time (closed form):**
$$H(t) = \mathbf{I}(\log(1+t))'\boldsymbol{\beta}$$

**Survival function:**
$$S(t) = \exp(-H(t)) = \exp\left(-\mathbf{I}(\log(1+t))'\boldsymbol{\beta}\right)$$

### 2.4 Covariate Effects

#### 2.4.1 Proportional Hazards (`:ph`)

The proportional hazards model multiplies the baseline hazard by $\exp(\boldsymbol{\gamma}'\mathbf{x})$:

$$h(t \mid \mathbf{x}) = h_0(t) \cdot \exp(\boldsymbol{\gamma}'\mathbf{x}) = \frac{\mathbf{B}(\log(1+t))'\boldsymbol{\beta}}{1+t} \cdot \exp(\boldsymbol{\gamma}'\mathbf{x})$$

$$H(t \mid \mathbf{x}) = H_0(t) \cdot \exp(\boldsymbol{\gamma}'\mathbf{x}) = \mathbf{I}(\log(1+t))'\boldsymbol{\beta} \cdot \exp(\boldsymbol{\gamma}'\mathbf{x})$$

**Implementation:** No change from linear-time splines—multiply hazard/cumhaz by $\exp(\boldsymbol{\gamma}'\mathbf{x})$.

#### 2.4.2 Accelerated Failure Time (`:aft`)

The AFT model scales time by $\phi = \exp(-\boldsymbol{\gamma}'\mathbf{x})$:

$$h(t \mid \mathbf{x}) = h_0(t \cdot \phi) \cdot \phi$$

With $g(t) = \log(1+t)$:

$$h(t \mid \mathbf{x}) = \frac{\mathbf{B}(\log(1 + t\phi))'\boldsymbol{\beta}}{1 + t\phi} \cdot \phi$$

**Key observation:** $\log(1 + t\phi) \neq \log(1+t) + \log\phi$, so AFT does NOT simplify to a shift in transformed time. This is different from the $\log t$ case where $\log(t\phi) = \log t + \log\phi$.

$$H(t \mid \mathbf{x}) = H_0(t\phi) = \mathbf{I}(\log(1 + t\phi))'\boldsymbol{\beta}$$

**Implementation:** Replace $t$ with $t \cdot \exp(-\boldsymbol{\gamma}'\mathbf{x})$ before applying transformation.

### 2.5 Knot Specification

**User interface:** Users specify knots on the **original time scale** (e.g., `knots = [1.0, 2.0, 5.0]`).

**Internal transformation:** Knots are transformed to the log(1+t) scale:
- Interior knots: $\xi_j^* = \log(1 + \xi_j)$
- Boundary knots: $[0, \log(1 + t_{\max})]$ where the lower bound is always 0

**Example:**
```
User specifies:     knots = [1.0, 2.0, 5.0], boundaryknots = [0.0, 10.0]
Internal log1p:     knots* = [0.693, 1.099, 1.792], boundaryknots* = [0.0, 2.398]
```

**Note:** The lower boundary in transformed space is always 0 since $\log(1+0) = 0$.

### 2.6 Why This Works for Weibull

The Weibull hazard is $h(t) = \kappa \lambda t^{\kappa-1}$, so:
$$\log h(t) = \log(\kappa\lambda) + (\kappa - 1)\log t$$

For large $t$, $\log(1+t) \approx \log t$, so:
$$h(t) \cdot (1+t) \approx h(t) \cdot t = \kappa\lambda t^\kappa$$

Taking logs:
$$\log(h(t) \cdot (1+t)) \approx \log(\kappa\lambda) + \kappa\log t \approx \log(\kappa\lambda) + \kappa\log(1+t)$$

Thus, the hazard in transformed scale $h^*(\tau) = h(t)(1+t)$ is approximately exponential in $\tau = \log(1+t)$, which a low-degree spline can capture efficiently.

### 2.7 Boundary Behavior

At $t = 0$:
- $g(0) = 0$
- $g'(0) = 1$
- $h(0) = h^*(0) \cdot 1 = \mathbf{B}(0)'\boldsymbol{\beta}$

The hazard at $t = 0$ equals the spline evaluated at $\tau = 0$—no singularity or special handling needed.

---

## 3. API Design

### 3.1 User-Facing API

**New keyword argument:** `time_scale::Symbol`

```julia
# Linear time (current default, unchanged behavior)
h12 = Hazard(@formula(0 ~ age), :sp, 1, 2;
    degree = 3,
    knots = [1.0, 2.0, 5.0],
    boundaryknots = [0.0, 10.0],
    time_scale = :linear   # DEFAULT - current behavior
)

# Log(1+t) transformed time (new)
h12 = Hazard(@formula(0 ~ age), :sp, 1, 2;
    degree = 3,
    knots = [1.0, 2.0, 5.0],        # User specifies on ORIGINAL time scale
    boundaryknots = [0.0, 10.0],    # Internally transformed to [0, log(11)]
    time_scale = :log1p             # NEW OPTION
)
```

**Allowed values:**
- `:linear` (default) — Current behavior, $h(t) = \mathbf{B}(t)'\boldsymbol{\beta}$
- `:log1p` — New option, $h(t) = \mathbf{B}(\log(1+t))'\boldsymbol{\beta} / (1+t)$

### 3.2 SplineHazard Type Update

**File:** `src/types/hazard_specs.jl`

```julia
struct SplineHazard <: HazardFunction
    hazard::StatsModels.FormulaTerm
    family::Symbol
    statefrom::Int64
    stateto::Int64
    degree::Int64
    knots::Union{Nothing,Float64,Vector{Float64}}
    boundaryknots::Union{Nothing,Vector{Float64}}
    extrapolation::String
    natural_spline::Bool
    monotone::Int64
    time_scale::Symbol              # NEW FIELD: :linear or :log1p
    metadata::HazardMetadata
end
```

### 3.3 RuntimeSplineHazard Type Update

**File:** `src/types/hazard_structs.jl`

```julia
struct RuntimeSplineHazard <: _SplineHazard
    hazname::Symbol
    statefrom::Int64
    stateto::Int64
    family::Symbol
    parnames::Vector{Symbol}
    npar_baseline::Int64
    npar_total::Int64
    hazard_fn::Function
    cumhaz_fn::Function
    has_covariates::Bool
    covar_names::Vector{Symbol}
    degree::Int64
    knots::Vector{Float64}          # Original-scale knots (user-facing)
    knots_transformed::Vector{Float64}  # NEW: Transformed-scale knots (internal)
    natural_spline::Bool
    monotone::Int64
    extrapolation::String
    time_scale::Symbol              # NEW FIELD
    metadata::HazardMetadata
    shared_baseline_key::Union{Nothing,SharedBaselineKey}
    smooth_info::Vector{SmoothTermInfo}
end
```

### 3.4 Hazard Constructor Update

**File:** `src/construction/hazard_constructors.jl`

Add `time_scale` keyword argument with validation:

```julia
function Hazard(
    hazard::StatsModels.FormulaTerm,
    family::Union{AbstractString,Symbol},
    statefrom::Int,
    stateto::Int;
    # ... existing kwargs ...
    time_scale::Symbol = :linear    # NEW
)
    # Validation for splines
    if family_key == :sp
        time_scale in (:linear, :log1p) || throw(ArgumentError(
            "time_scale must be :linear or :log1p, got :$time_scale"
        ))
    end
    # ... rest of constructor ...
end
```

### 3.5 calibrate_splines! Update

**File:** `src/hazard/spline.jl`

When `time_scale = :log1p`, knot placement should consider the transformed scale:

```julia
function calibrate_splines!(model::MultistateProcess; ...)
    for haz in model.hazards
        if haz isa RuntimeSplineHazard && haz.time_scale == :log1p
            # Place knots at quantiles of log1p(event_times)
            # Transform back to original scale for storage
        end
    end
end
```

---

## 4. Core Implementation

### 4.1 Spline Builder Modifications

**File:** `src/construction/spline_builder.jl`

#### Task 4.1.1: Update `_build_spline_hazard`

```julia
function _build_spline_hazard(ctx::HazardBuildContext)
    hazard = ctx.hazard::SplineHazard
    time_scale = hazard.time_scale  # NEW: Extract time_scale
    
    # ... existing boundary/knot extraction ...
    
    # NEW: Transform knots if time_scale == :log1p
    if time_scale == :log1p
        # Store original knots for user display
        original_knots = copy(allknots)
        original_bknots = copy(bknots)
        
        # Transform to log1p scale
        allknots_transformed = log1p.(allknots)
        bknots_transformed = log1p.(bknots)
        
        # Build basis on transformed scale
        B = BSplineBasis(BSplineOrder(hazard.degree + 1), allknots_transformed)
    else
        # Linear time: use knots directly
        original_knots = allknots
        allknots_transformed = allknots
        B = BSplineBasis(BSplineOrder(hazard.degree + 1), allknots)
    end
    
    # ... boundary condition recombination (unchanged) ...
    
    # Generate hazard functions with time_scale awareness
    hazard_fn, cumhaz_fn = _generate_spline_hazard_fns(
        B, extrap_method, hazard.monotone, nbasis, parnames, 
        ctx.metadata.linpred_effect, time_scale  # NEW: pass time_scale
    )
    
    # Build RuntimeSplineHazard with both original and transformed knots
    haz_struct = RuntimeSplineHazard(
        # ... existing fields ...
        original_knots,           # knots (user-facing, original scale)
        allknots_transformed,     # knots_transformed (internal)
        # ... rest of fields ...
        time_scale,               # NEW field
        # ...
    )
end
```

#### Task 4.1.2: Update `_generate_spline_hazard_fns`

**Linear time (`:linear`)** — Existing implementation:
```julia
# hazard_fn(t, pars, covars)
h0 = spline_ext(t)
return h0 * exp(linear_pred)  # PH

# cumhaz_fn(lb, ub, pars, covars)
H0 = cumhaz_spline(ub) - cumhaz_spline(lb)
return H0 * exp(linear_pred)  # PH
```

**Log1p time (`:log1p`)** — New implementation:
```julia
# hazard_fn(t, pars, covars)
tau = log1p(t)           # Transform time
jacobian = 1 / (1 + t)   # g'(t)
h_star = spline_ext(tau) # Hazard in transformed scale
h0 = h_star * jacobian   # Hazard in original scale
return h0 * exp(linear_pred)  # PH

# cumhaz_fn(lb, ub, pars, covars)
tau_lb = log1p(lb)
tau_ub = log1p(ub)
# Closed form! No quadrature needed.
H0 = cumhaz_spline(tau_ub) - cumhaz_spline(tau_lb)
return H0 * exp(linear_pred)  # PH
```

**AFT covariate effect:**

For `:linear`:
```julia
scale = exp(-linear_pred)
h0 = spline_ext(t * scale) * scale
H0 = cumhaz_spline(ub * scale) - cumhaz_spline(lb * scale)
```

For `:log1p`:
```julia
scale = exp(-linear_pred)
t_scaled = t * scale
tau = log1p(t_scaled)
jacobian = scale / (1 + t_scaled)
h_star = spline_ext(tau)
h0 = h_star * jacobian

# Cumulative hazard
tau_lb = log1p(lb * scale)
tau_ub = log1p(ub * scale)
H0 = cumhaz_spline(tau_ub) - cumhaz_spline(tau_lb)
```

#### Task 4.1.3: Extrapolation Handling

The extrapolation handling in `_eval_cumhaz_with_extrap` needs updating for `:log1p`.

**For constant extrapolation beyond boundaries:**

In transformed scale $[\tau_{\text{lo}}, \tau_{\text{hi}}]$, the hazard extends constantly.

In original scale, if $t > t_{\text{hi}}$ where $\tau_{\text{hi}} = \log(1 + t_{\text{hi}})$:
- $h^*(\tau) = h^*(\tau_{\text{hi}})$ for $\tau > \tau_{\text{hi}}$
- $h(t) = h^*(\tau_{\text{hi}}) / (1 + t)$

The cumulative hazard contribution for $t > t_{\text{hi}}$:
$$\int_{t_{\text{hi}}}^{t} \frac{h^*(\tau_{\text{hi}})}{1 + u}\,du = h^*(\tau_{\text{hi}}) \cdot [\log(1+t) - \log(1+t_{\text{hi}})]$$

This is still closed-form.

---

## 5. Monotonicity Constraints

### 5.1 Monotonicity in Transformed Scale

The existing monotonicity constraint ensures $h^*(\tau)$ is monotone in $\tau$. For `:log1p`, this translates to monotonicity of $h^*(\log(1+t))$.

**Question:** Does monotonicity of $h^*(\tau)$ imply monotonicity of $h(t) = h^*(\tau) / (1+t)$?

**Answer:** No, not in general.

**Example:** Suppose $h^*(\tau) = c$ (constant). Then:
$$h(t) = \frac{c}{1+t}$$
which is strictly decreasing even though $h^*$ is constant.

### 5.2 Derivative Analysis

For $h(t) = h^*(g(t)) \cdot g'(t)$ with $g(t) = \log(1+t)$:

$$h(t) = \frac{h^*(\tau)}{1+t} \quad \text{where } \tau = \log(1+t)$$

Taking the derivative:
$$h'(t) = \frac{d}{dt}\left[\frac{h^*(\tau)}{1+t}\right]$$

Using the chain rule and quotient rule:
$$h'(t) = \frac{(h^*)'(\tau) \cdot \tau'(t) \cdot (1+t) - h^*(\tau)}{(1+t)^2}$$

Since $\tau'(t) = 1/(1+t)$:
$$h'(t) = \frac{(h^*)'(\tau) - h^*(\tau)}{(1+t)^2}$$

For $h(t)$ to be increasing ($h'(t) \geq 0$):
$$(h^*)'(\tau) \geq h^*(\tau)$$

This is a **stronger condition** than just $(h^*)'(\tau) \geq 0$.

### 5.3 Implementation Options for Monotonicity

**Option A: Constrain in original time (complex)**

Require $(h^*)'(\tau) \geq h^*(\tau)$ at all knot points. This couples the I-spline increments in a nonlinear way. Complex to implement.

**Option B: Constrain in transformed time (simple, document limitation)**

Keep existing I-spline monotonicity in $\tau$-space. Document that this ensures $h^*(\tau)$ is monotone but does NOT guarantee $h(t)$ is monotone in $t$ for `:log1p`.

**Recommendation:** Option B. The primary use case for `:log1p` is Weibull-like hazards that are naturally monotone. Users needing strict monotonicity in $t$ can use `:linear` time scale.

### 5.4 Monotone Parameter Transformation

The existing `_spline_ests2coefs` and `_spline_coefs2ests` functions apply to coefficients in the spline space and are independent of time transformation. No changes needed to these functions—they operate on $\boldsymbol{\beta}$ regardless of whether the basis is evaluated at $t$ or $\log(1+t)$.

---

## 6. Post-Processing and Output

### 6.1 Parameter Accessors

**No changes needed.** The parameters $\boldsymbol{\beta}$ have the same interpretation regardless of time scale—they are spline coefficients in the (possibly transformed) time domain.

The accessor functions `get_parameters`, `get_parameters_flat`, `get_parameters_natural` all work unchanged.

### 6.2 Hazard Evaluation API

**File:** `src/hazard/api.jl`

The existing `compute_hazard` and `compute_cumulative_hazard` functions dispatch to hazard-specific evaluation:

```julia
function compute_hazard(t, model::MultistateProcess, hazard::Symbol, subj::Int=1)
    # Dispatches to hazard.hazard_fn(t, pars, covars)
    # hazard_fn already handles time_scale internally
end
```

**No changes needed**—the transformation is encapsulated in `hazard_fn`.

### 6.3 Fitted Model Display

When printing a fitted model with `:log1p` splines, the display should indicate the time scale.

**Enhancement:** Update `show` method for `RuntimeSplineHazard` to display time scale:

```julia
function Base.show(io::IO, h::RuntimeSplineHazard)
    print(io, "RuntimeSplineHazard(")
    print(io, "$(h.statefrom)→$(h.stateto), ")
    print(io, "degree=$(h.degree), ")
    if h.time_scale == :log1p
        print(io, "time_scale=:log1p, ")
        print(io, "knots=$(round.(h.knots, digits=3)) [log1p: $(round.(h.knots_transformed, digits=3))]")
    else
        print(io, "knots=$(round.(h.knots, digits=3))")
    end
    print(io, ")")
end
```

### 6.4 Knot Reporting in calibrate_splines!

The `calibrate_splines!` return value should report knots on the **original time scale** (user-facing) with an optional indication of transformed values:

```julia
function calibrate_splines!(model; ...)
    # Returns NamedTuple with:
    # - interior_knots: Vector{Float64} on original scale
    # - boundary_knots: Vector{Float64} on original scale
    # - interior_knots_transformed: Vector{Float64} (only for :log1p)
    # - boundary_knots_transformed: Vector{Float64} (only for :log1p)
end
```

---

## 7. Testing Infrastructure

### 7.1 Unit Tests

**File:** `MultistateModelsTests/unit/test_log1p_splines.jl` (NEW)

#### Test 7.1.1: Knot Transformation Correctness

```julia
@testset "Knot transformation to log1p scale" begin
    # Test that user-specified knots are correctly transformed
    knots = [1.0, 2.0, 5.0]
    expected = log1p.(knots)  # [0.693, 1.099, 1.792]
    
    h12 = Hazard(@formula(0 ~ 1), :sp, 1, 2;
        knots = knots,
        boundaryknots = [0.0, 10.0],
        time_scale = :log1p
    )
    
    data = DataFrame(id=1, tstart=0.0, tstop=1.0, statefrom=1, stateto=2, obstype=1)
    model = multistatemodel(h12; data=data)
    
    haz = model.hazards[1]
    @test haz.knots ≈ [0.0, knots..., 10.0]  # Original scale
    @test haz.knots_transformed ≈ log1p.([0.0, knots..., 10.0])  # Transformed
end
```

#### Test 7.1.2: Hazard Includes Jacobian Factor

```julia
@testset "Hazard includes 1/(1+t) Jacobian" begin
    # For constant spline in transformed scale, h(t) = c/(1+t)
    h12 = Hazard(@formula(0 ~ 1), :sp, 1, 2;
        degree = 0,  # Constant spline
        knots = Float64[],
        boundaryknots = [0.0, 10.0],
        time_scale = :log1p
    )
    
    data = DataFrame(id=1, tstart=0.0, tstop=5.0, statefrom=1, stateto=2, obstype=1)
    model = multistatemodel(h12; data=data)
    set_parameters!(model, (h12 = [1.0],))  # h*(τ) = 1
    
    # h(t) = 1 / (1+t)
    @test compute_hazard(0.0, model, :h12) ≈ 1.0
    @test compute_hazard(1.0, model, :h12) ≈ 0.5
    @test compute_hazard(4.0, model, :h12) ≈ 0.2
end
```

#### Test 7.1.3: Cumulative Hazard is Closed Form

```julia
@testset "Cumulative hazard is closed form (no quadrature)" begin
    # For h(t) = c/(1+t), H(t) = c * log(1+t)
    h12 = Hazard(@formula(0 ~ 1), :sp, 1, 2;
        degree = 0,
        knots = Float64[],
        boundaryknots = [0.0, 10.0],
        time_scale = :log1p
    )
    
    data = DataFrame(id=1, tstart=0.0, tstop=5.0, statefrom=1, stateto=2, obstype=1)
    model = multistatemodel(h12; data=data)
    set_parameters!(model, (h12 = [1.0],))
    
    # H(t) = 1 * log(1+t)
    @test compute_cumulative_hazard(0.0, 1.0, model, :h12) ≈ log(2) rtol=1e-10
    @test compute_cumulative_hazard(0.0, 4.0, model, :h12) ≈ log(5) rtol=1e-10
    @test compute_cumulative_hazard(1.0, 4.0, model, :h12) ≈ log(5) - log(2) rtol=1e-10
end
```

#### Test 7.1.4: Consistency Check: H'(t) = h(t)

```julia
@testset "Derivative of cumhaz equals hazard" begin
    h12 = Hazard(@formula(0 ~ 1), :sp, 1, 2;
        degree = 3,
        knots = [1.0, 2.0, 4.0],
        boundaryknots = [0.0, 10.0],
        time_scale = :log1p
    )
    
    data = DataFrame(id=1, tstart=0.0, tstop=5.0, statefrom=1, stateto=2, obstype=1)
    model = multistatemodel(h12; data=data)
    set_parameters!(model, (h12 = [0.5, 0.8, 1.0, 0.7, 0.6, 0.5, 0.4],))
    
    # Numerical derivative of H(t) should equal h(t)
    for t in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
        ε = 1e-6
        H_deriv = (compute_cumulative_hazard(0.0, t+ε, model, :h12) - 
                   compute_cumulative_hazard(0.0, t-ε, model, :h12)) / (2ε)
        h_t = compute_hazard(t, model, :h12)
        @test H_deriv ≈ h_t rtol=1e-4
    end
end
```

#### Test 7.1.5: AD Compatibility

```julia
@testset "ForwardDiff compatibility for log1p splines" begin
    h12 = Hazard(@formula(0 ~ 1), :sp, 1, 2;
        degree = 3,
        knots = [2.0],
        boundaryknots = [0.0, 10.0],
        time_scale = :log1p
    )
    
    data = DataFrame(id=[1,1], tstart=[0.0,0.0], tstop=[3.0,5.0], 
                     statefrom=[1,1], stateto=[2,1], obstype=[1,1])
    model = multistatemodel(h12; data=data)
    
    # Verify gradient computation doesn't error
    params = get_parameters_flat(model)
    
    # Log-likelihood function
    function negloglik(θ)
        set_parameters!(model, θ)
        return -loglik_exact(model)  # Internal function
    end
    
    grad = ForwardDiff.gradient(negloglik, params)
    @test all(isfinite.(grad))
end
```

### 7.2 Integration Tests

**File:** `MultistateModelsTests/integration/test_log1p_splines_exact.jl` (NEW)

#### Test 7.2.1: Weibull Recovery with Log1p Splines

```julia
@testset "Log1p spline recovers Weibull hazard" begin
    # Weibull DGP: h(t) = κλt^(κ-1)
    κ, λ = 1.5, 0.3
    n = 500
    
    # Simulate Weibull data
    Random.seed!(12345)
    U = rand(n)
    T = (-log.(U) ./ λ) .^ (1/κ)
    censored = T .> 10.0
    T[censored] .= 10.0
    
    data = DataFrame(
        id = 1:n,
        tstart = zeros(n),
        tstop = T,
        statefrom = ones(Int, n),
        stateto = ifelse.(censored, 1, 2),
        obstype = ones(Int, n)
    )
    
    # Fit log1p spline with few knots (should be parsimonious)
    h12 = Hazard(@formula(0 ~ 1), :sp, 1, 2;
        degree = 3,
        knots = [2.0, 5.0],  # Just 2 interior knots
        boundaryknots = [0.0, 10.0],
        time_scale = :log1p
    )
    
    model = multistatemodel(h12; data=data)
    fitted = fit(model; penalty=:none)
    
    # Compare fitted hazard to true Weibull
    weibull_hazard(t) = κ * λ * t^(κ-1)
    
    test_times = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
    for t in test_times
        h_true = weibull_hazard(t)
        h_fitted = compute_hazard(t, fitted, :h12)
        @test h_fitted ≈ h_true rtol=0.15  # 15% tolerance
    end
end
```

#### Test 7.2.2: Covariate Effects (PH and AFT)

```julia
@testset "Log1p spline with PH covariates" begin
    # Weibull DGP with covariate effect
    κ, λ, β = 1.3, 0.2, 0.5
    n = 500
    
    Random.seed!(54321)
    x = rand(0:1, n)
    U = rand(n)
    T = (-log.(U) ./ (λ .* exp.(β .* x))) .^ (1/κ)
    censored = T .> 10.0
    T[censored] .= 10.0
    
    data = DataFrame(
        id = 1:n,
        tstart = zeros(n),
        tstop = T,
        statefrom = ones(Int, n),
        stateto = ifelse.(censored, 1, 2),
        obstype = ones(Int, n),
        x = x
    )
    
    h12 = Hazard(@formula(0 ~ x), :sp, 1, 2;
        degree = 3,
        knots = [2.0, 5.0],
        boundaryknots = [0.0, 10.0],
        time_scale = :log1p,
        linpred_effect = :ph
    )
    
    model = multistatemodel(h12; data=data)
    fitted = fit(model; penalty=:none)
    
    # Extract covariate coefficient
    params = get_parameters(fitted)
    β_hat = params.h12[end]  # Last parameter is covariate
    @test β_hat ≈ β atol=0.2
end

@testset "Log1p spline with AFT covariates" begin
    # Similar setup with linpred_effect = :aft
    # ...
end
```

### 7.3 Long Tests

**File:** `MultistateModelsTests/longtests/longtest_log1p_splines.jl` (NEW)

Structure parallel to `longtest_spline_exact.jl`:

```julia
# Test Configuration
const RNG_SEED_LOG1P = 0xL0G1P001
const TRUE_WEIBULL_SHAPE = 1.4
const TRUE_WEIBULL_SCALE = 0.15
const TRUE_BETA = 0.5
const HAZARD_RTOL = 0.20  # 20% relative tolerance for hazard curve

# Tests:
# - log1p_sp_exact_nocov: No covariates, Weibull DGP
# - log1p_sp_exact_tfc: Time-fixed covariate
# - log1p_sp_exact_tvc: Time-varying covariate
# - log1p_sp_aft_exact_nocov: AFT covariate effect
# - log1p_sp_aft_exact_tfc: AFT with time-fixed covariate
```

**File:** `MultistateModelsTests/longtests/longtest_log1p_comparison.jl` (NEW)

```julia
# Compare linear vs log1p splines for different DGPs

@testset "Linear vs Log1p: Weibull DGP" begin
    # Log1p should achieve lower RMSE with fewer EDF
end

@testset "Linear vs Log1p: Gompertz DGP" begin
    # Linear may be competitive
end

@testset "Linear vs Log1p: Bathtub DGP" begin
    # Linear should be better for non-monotonic hazards
end
```

### 7.4 MCEM Tests Extension

**File:** `MultistateModelsTests/longtests/longtest_mcem_splines.jl` (EXTEND)

Add parallel test cases with `time_scale = :log1p`:

```julia
@testset "MCEM Log1p Spline vs Exponential" begin
    # Same as existing test but with time_scale = :log1p
    h12_sp = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
        degree=1, 
        knots=Float64[],
        boundaryknots=[0.0, 5.0],
        time_scale = :log1p  # NEW
    )
    # ...
end
```

---

## 8. Documentation Updates

### 8.1 Smoothing Splines Skill File

**File:** `.github/skills/smoothing-splines/SKILL.md`

Add new section:

```markdown
## Time-Transformed Splines (Log1p Scale)

### Overview

For hazards that are well-behaved on the log-time scale (e.g., Weibull), evaluating 
the spline basis at τ = log(1+t) can achieve better parsimony.

### Mathematical Formulation

With time transformation g(t) = log(1+t):

**Hazard:** h(t) = B(log(1+t))'β / (1+t)

**Cumulative hazard (closed form):** H(t) = I(log(1+t))'β

### Usage

```julia
h12 = Hazard(@formula(0 ~ 1), :sp, 1, 2;
    degree = 3,
    knots = [1.0, 2.0, 5.0],     # Specified on ORIGINAL time scale
    boundaryknots = [0.0, 10.0],
    time_scale = :log1p          # Evaluate basis at log(1+t)
)
```

### When to Use

| Scenario | Recommended time_scale |
|----------|------------------------|
| Weibull-like hazard | `:log1p` |
| Gompertz-like hazard | `:linear` |
| Unknown hazard shape | Try both, compare EDF |
| Strict monotonicity in t required | `:linear` |

### Monotonicity Note

With `time_scale = :log1p`, the monotonicity constraint applies to h*(τ) in transformed
scale, which does NOT guarantee monotonicity of h(t) in original scale. For strict
monotonicity in t, use `time_scale = :linear`.
```

### 8.2 Codebase Knowledge Skill File

**File:** `.github/skills/codebase-knowledge/SKILL.md`

Update the SplineHazard documentation in the type system map:

```markdown
### Spline Time Scale Options

| time_scale | Basis Evaluation | Hazard | Cumulative Hazard | Use Case |
|------------|------------------|--------|-------------------|----------|
| `:linear` | B(t) | B(t)'β | I(t)'β | Default, general purpose |
| `:log1p` | B(log(1+t)) | B(log(1+t))'β / (1+t) | I(log(1+t))'β | Weibull-like hazards |
```

### 8.3 User Documentation

**File:** `docs/src/splines.md` (NEW or extend existing)

```markdown
# Spline Hazard Models

## Time Scale Options

By default, spline hazards are evaluated on linear time: h(t) = B(t)'β.

For hazards that are linear on the log-time scale (like Weibull), you can use
`time_scale = :log1p` to evaluate the basis at log(1+t):

```julia
h12 = Hazard(@formula(0 ~ age), :sp, 1, 2;
    knots = [1.0, 2.0, 5.0],
    time_scale = :log1p
)
```

This often achieves comparable fit with fewer effective degrees of freedom for
Weibull-like hazard shapes.

### Knot Specification

Knots are always specified on the **original time scale**, regardless of `time_scale`.
The transformation is applied internally.
```

---

## 9. Implementation Sequence

### Phase 1: Core Infrastructure (Days 1-2)

| Task | Description | Files |
|------|-------------|-------|
| 1.1 | Add `time_scale` field to `SplineHazard` | `src/types/hazard_specs.jl` |
| 1.2 | Add `time_scale` and `knots_transformed` to `RuntimeSplineHazard` | `src/types/hazard_structs.jl` |
| 1.3 | Update `Hazard` constructor validation | `src/construction/hazard_constructors.jl` |
| 1.4 | Update `_build_spline_hazard` for knot transformation | `src/construction/spline_builder.jl` |
| 1.5 | Update `_generate_spline_hazard_fns` for log1p | `src/construction/spline_builder.jl` |
| 1.6 | Update extrapolation handling for log1p | `src/construction/spline_builder.jl` |

### Phase 2: Supporting Functions (Day 2)

| Task | Description | Files |
|------|-------------|-------|
| 2.1 | Update `calibrate_splines!` for log1p knot placement | `src/hazard/spline.jl` |
| 2.2 | Update display/show methods | `src/types/hazard_structs.jl` |
| 2.3 | Verify accessor functions work unchanged | (verification only) |

### Phase 3: Unit Tests (Day 3)

| Task | Description | Files |
|------|-------------|-------|
| 3.1 | Knot transformation tests | `MultistateModelsTests/unit/test_log1p_splines.jl` |
| 3.2 | Jacobian factor tests | `MultistateModelsTests/unit/test_log1p_splines.jl` |
| 3.3 | Closed-form cumhaz tests | `MultistateModelsTests/unit/test_log1p_splines.jl` |
| 3.4 | H'(t) = h(t) consistency tests | `MultistateModelsTests/unit/test_log1p_splines.jl` |
| 3.5 | AD compatibility tests | `MultistateModelsTests/unit/test_log1p_splines.jl` |

### Phase 4: Integration and Long Tests (Days 4-5)

| Task | Description | Files |
|------|-------------|-------|
| 4.1 | Weibull recovery test | `MultistateModelsTests/integration/test_log1p_splines_exact.jl` |
| 4.2 | PH covariate test | `MultistateModelsTests/integration/test_log1p_splines_exact.jl` |
| 4.3 | AFT covariate test | `MultistateModelsTests/integration/test_log1p_splines_exact.jl` |
| 4.4 | Long test: no covariates | `MultistateModelsTests/longtests/longtest_log1p_splines.jl` |
| 4.5 | Long test: TFC/TVC | `MultistateModelsTests/longtests/longtest_log1p_splines.jl` |
| 4.6 | Comparison long test | `MultistateModelsTests/longtests/longtest_log1p_comparison.jl` |
| 4.7 | Extend MCEM tests | `MultistateModelsTests/longtests/longtest_mcem_splines.jl` |

### Phase 5: Documentation (Day 5)

| Task | Description | Files |
|------|-------------|-------|
| 5.1 | Update smoothing-splines skill | `.github/skills/smoothing-splines/SKILL.md` |
| 5.2 | Update codebase-knowledge skill | `.github/skills/codebase-knowledge/SKILL.md` |
| 5.3 | User documentation | `docs/src/splines.md` or equivalent |

---

## 10. Validation Criteria

### 10.1 Correctness Criteria

| Criterion | Test Method | Tolerance |
|-----------|-------------|-----------|
| $H'(t) = h(t)$ numerically | Finite difference | rtol=1e-4 |
| Weibull hazard recovery | Compare to analytical | rtol=0.15 |
| Weibull cumhaz recovery | Compare to analytical | rtol=0.10 |
| Covariate effect recovery | Parameter estimate | atol=0.20 |

### 10.2 Implementation Criteria

| Criterion | Verification |
|-----------|--------------|
| No numerical quadrature in cumhaz | Code inspection |
| Closed-form cumhaz | Code inspection |
| AD compatible | ForwardDiff.gradient returns finite |
| Backward compatible | All existing tests pass |

### 10.3 Performance Criteria

| Criterion | Threshold |
|-----------|-----------|
| Fitting time vs linear splines | ≤1.1× (no quadrature overhead) |
| Memory usage vs linear splines | ≤1.05× (only extra knot storage) |

---

## 11. File Change Summary

### New Files

| File | Purpose |
|------|---------|
| `MultistateModelsTests/unit/test_log1p_splines.jl` | Unit tests |
| `MultistateModelsTests/integration/test_log1p_splines_exact.jl` | Integration tests |
| `MultistateModelsTests/longtests/longtest_log1p_splines.jl` | Statistical validation |
| `MultistateModelsTests/longtests/longtest_log1p_comparison.jl` | Linear vs log1p comparison |

### Modified Files

| File | Changes |
|------|---------|
| `src/types/hazard_specs.jl` | Add `time_scale` field to `SplineHazard` |
| `src/types/hazard_structs.jl` | Add `time_scale`, `knots_transformed` to `RuntimeSplineHazard` |
| `src/construction/hazard_constructors.jl` | Accept and validate `time_scale` kwarg |
| `src/construction/spline_builder.jl` | Knot transformation, hazard/cumhaz generation |
| `src/hazard/spline.jl` | Update `calibrate_splines!` for log1p |
| `MultistateModelsTests/longtests/longtest_mcem_splines.jl` | Add log1p test cases |
| `.github/skills/smoothing-splines/SKILL.md` | Document log1p option |
| `.github/skills/codebase-knowledge/SKILL.md` | Update type documentation |

### Estimated Lines of Code

| Category | Lines |
|----------|-------|
| Type definitions | ~20 |
| Spline builder | ~100 |
| Hazard constructor | ~10 |
| calibrate_splines! | ~20 |
| Unit tests | ~200 |
| Integration tests | ~150 |
| Long tests | ~400 |
| Documentation | ~100 |
| **Total** | **~1000** |

---

## Appendix: Alternative Transformations Considered

### A.1 log(t) (Royston-Parmar)

**Pros:** Exact match to RP model, Weibull is exactly linear

**Cons:** log(0) = -∞, requires t > 0, boundary handling complex

**Decision:** Rejected in favor of log(1+t) for numerical stability at origin

### A.2 log(t + ε) for small ε

**Pros:** Approximates log(t), finite at t=0

**Cons:** Introduces arbitrary ε, discontinuous derivative at t=0

**Decision:** Rejected; log(1+t) is cleaner

### A.3 arcsinh(t)

**Pros:** Similar to log(t) for large t, linear near 0, symmetric

**Cons:** Less interpretable, no theoretical motivation

**Decision:** Not considered primary; could add later if needed
