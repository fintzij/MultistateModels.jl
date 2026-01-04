# Hazard Parameterization Consistency Audit Report

## Executive Summary

This report documents a comprehensive review of hazard parameterizations in MultistateModels.jl, comparing documentation against implementation. **The implementation appears internally consistent**, but there is a **critical discrepancy between copilot-instructions.md and the actual code for Weibull parameterization**.

---

## 1. Documentation Sources

### 1.1 copilot-instructions.md ([.github/copilot-instructions.md](.github/copilot-instructions.md))

| Family | Parameters | Storage (estimation scale) |
|--------|------------|---------------------------|
| Exponential | rate λ | `[log(λ)]` |
| Weibull | shape κ, scale λ | `[κ, log(λ)]` |
| Gompertz | shape a, rate b | `[a, log(b)]` |
| Spline | coefficients β | `[β...]` (log-hazard scale) |

**Key claim**: Weibull shape is stored on **identity scale** (not log-transformed)

### 1.2 docs/src/index.md ([docs/src/index.md](docs/src/index.md#L49-L86))

#### Exponential (`:exp`)
- **Hazard**: h(t) = rate
- **Cumulative hazard**: H(t) = rate × t
- **Parameters**: rate > 0

#### Weibull (`:wei`)
- **Hazard**: h(t) = shape × scale × t^(shape-1)
- **Cumulative hazard**: H(t) = scale × t^shape
- **Parameters**: shape > 0, scale > 0

#### Gompertz (`:gom`)
- **Hazard**: h(t) = rate × exp(shape × t)
- **Cumulative hazard**: H(t) = (rate/shape) × (exp(shape×t) - 1) for shape ≠ 0
- **Parameters**: shape ∈ ℝ (unconstrained), rate > 0

---

## 2. Code Implementation Analysis

### 2.1 Parameter Transformations ([src/utilities/transforms.jl](src/utilities/transforms.jl))

#### `transform_baseline_to_natural` (lines 42-61)

```julia
if family == :sp
    # Splines: parameters stay on log scale
    return baseline
elseif family == :gom
    # Gompertz: shape is unconstrained (first param), rate is positive (second param)
    transformed_values = ntuple(i -> i == 1 ? vs[i] : exp(vs[i]), length(vs))
elseif family in (:exp, :wei)
    # Exponential and Weibull: all baseline parameters are positive (exp)
    transformed_values = map(v -> exp(v), values(baseline))
```

**Actual behavior for Weibull**: 
- `pars[1]` (shape) → `exp(pars[1])` 
- `pars[2]` (scale) → `exp(pars[2])`

⚠️ **INCONSISTENCY**: Code applies `exp()` to **both** Weibull parameters, but copilot-instructions.md says shape is on identity scale.

#### `transform_baseline_to_estimation` (lines 69-99)
Inverse transformation: applies `log()` to both Weibull parameters.

### 2.2 Parameter Initialization ([src/utilities/initialization.jl](src/utilities/initialization.jl#L557-L592))

```julia
function init_par(hazard::Union{MarkovHazard,SemiMarkovHazard,_SplineHazard}, crude_log_rate=0.0)
    if family == :exp
        return has_covs ? vcat(crude_log_rate, zeros(ncovar)) : [crude_log_rate]
        
    elseif family == :wei
        # Weibull: [log_shape, log_scale] or [log_shape, log_scale, β1, β2, ...]
        baseline = [0.0, crude_log_rate]  # log(shape=1), log_scale
        
    elseif family == :gom
        # Gompertz: [shape, log_rate] or [shape, log_rate, β1, β2, ...]
        baseline = [0.0, crude_log_rate]  # shape=0 (not log-transformed), log_rate
```

**Initialization patterns**:
- **Exponential**: `[log(rate)]`
- **Weibull**: `[log(shape), log(scale)]` — initializes log_shape=0 so shape=1
- **Gompertz**: `[shape, log(rate)]` — initializes shape=0 (on identity scale)

### 2.3 Hazard Generators ([src/hazard/generators.jl](src/hazard/generators.jl))

#### Exponential (lines 11-60)
```julia
# pars.baseline is on NATURAL scale (no exp needed)
return pars.baseline.$(baseline_parname) * exp(linear_pred)  # PH
return pars.baseline.$(baseline_parname) * exp(-linear_pred)  # AFT
```
**Formula**: h(t) = rate × exp(±β'x)

#### Weibull (lines 64-130)
```julia
# pars.baseline is on NATURAL scale (no exp needed)
shape = pars.baseline.$(shape_parname)
scale = pars.baseline.$(scale_parname)

# h(t) = shape * scale * t^(shape-1) * exp(linear_pred)
haz = shape * scale * exp(linear_pred)
if shape != 1.0
    haz *= t^(shape - 1)
end
```
**Formula**: h(t) = κ × λ × t^(κ-1) × exp(β'x) for PH

#### Gompertz (lines 133-245)
```julia
# flexsurv parameterization: h(t) = rate * exp(shape * t)
return rate * exp(shape * t + linear_pred)  # PH
return rate * exp(shape * t * time_scale) * time_scale  # AFT (time_scale = exp(-linpred))
```
**Formula**: h(t) = b × exp(a×t + β'x) for PH

### 2.4 Time Transform Functions ([src/hazard/time_transform.jl](src/hazard/time_transform.jl))

#### Weibull (lines 100-139)
```julia
# pars are on NATURAL scale (shape and scale already positive)
shape, scale = pars[1], pars[2]
# h(t) = shape * scale * t^(shape-1)
haz = shape * scale
if shape != 1.0
    haz *= t^(shape - 1)
end
```

#### Gompertz (lines 142-210)
```julia
# pars are on NATURAL scale
# pars[1] = shape (can be positive, negative, or zero)
# pars[2] = rate (positive)
if effect == :aft
    time_scale = exp(-linpred)
    scaled_shape = shape * time_scale
    return rate * exp(scaled_shape * t) * time_scale
else
    return rate * exp(shape * t + linpred)
end
```

---

## 3. Test File Analysis

### 3.1 Unit Tests ([MultistateModelsTests/unit/test_hazards.jl](MultistateModelsTests/unit/test_hazards.jl))

#### Weibull Tests (lines 160-210)
```julia
# Parameters: log(shape) = -0.25, log(scale) = 0.2
MultistateModels.set_parameters!(model, (h21 = [-0.25, 0.2],))

# Baseline params are now on natural scale (exp already applied)
shape = pars[1]  # exp(-0.25)
scale = pars[2]  # exp(0.2)

# Analytical log-hazard: log(λκt^{κ-1}) = log(λ) + log(κ) + (κ-1)log(t)
log_hazard = log_scale + log_shape + (shape - 1) * log(t_eval)
```

**Test confirms**: Both shape and scale are stored on log scale, then exp() applied.

#### Gompertz Tests (lines 437-520)
```julia
# Parameters: log(shape) = log(1.5), log(scale) = log(0.5)
MultistateModels.set_parameters!(model, (h12 = [log(1.5), log(0.5)], ...))

# flexsurv: log h(1) = log(rate) + shape * 1
shape = pars_h12[1]   # Already natural scale
rate = pars_h12[2]    # scale = rate in flexsurv terminology
expected_log_haz = log(rate) + shape * 1.0
```

⚠️ **ISSUE IN TEST**: Test sets `[log(1.5), log(0.5)]` for Gompertz, but according to the transformation code:
- `pars[1]` (shape) stays as-is (identity transform)
- `pars[2]` (rate) gets `exp()` applied

So setting `[log(1.5), log(0.5)]` gives:
- shape = log(1.5) ≈ 0.405 (not 1.5!)
- rate = exp(log(0.5)) = 0.5 ✓

### 3.2 Long Tests ([MultistateModelsTests/longtests/longtest_parametric_suite.jl](MultistateModelsTests/longtests/longtest_parametric_suite.jl))

```julia
elseif family == "wei"
    # Weibull: [log(shape), log(scale)] or [log(shape), log(scale), beta]
    h12 = [log(TRUE_WEIBULL_SHAPE_12), log(TRUE_WEIBULL_SCALE_12), TRUE_BETA]
    
elseif family == "gom"
    # Gompertz: [shape, log(rate)] or [shape, log(rate), beta]
    # Note: shape is NOT log-transformed for Gompertz (can be negative)
    h12 = [TRUE_GOMPERTZ_SHAPE_12, log(TRUE_RATE_12), TRUE_BETA]
```

**Longtest confirms Gompertz**: shape on identity scale, rate on log scale.

### 3.3 AFT Test Suite ([MultistateModelsTests/longtests/longtest_aft_suite.jl](MultistateModelsTests/longtests/longtest_aft_suite.jl))

```julia
if scenario.family == "wei"
    # shape, log(scale), beta
    true_params = (h12 = [log(1.2), log(0.5), 0.5], ...)
else # gom
    # shape, log(rate), beta
    true_params = (h12 = [0.1, log(0.5), 0.5], ...)
```

**AFT suite also confirms**: Weibull uses log(shape), Gompertz uses identity(shape).

---

## 4. Inconsistency Summary

### 4.1 ⚠️ CRITICAL: Weibull Shape Transform

| Source | Weibull Shape Storage |
|--------|----------------------|
| copilot-instructions.md | `κ` (identity scale) |
| Code (transforms.jl) | `log(κ)` (log scale, exp applied) |
| Initialization (init_par) | `log(κ)` (comment says "log_shape") |
| Unit tests | `log(κ)` (tests use log values) |
| Long tests | `log(κ)` (explicitly `log(TRUE_WEIBULL_SHAPE)`) |

**Conclusion**: The **code is internally consistent** — Weibull shape is stored on **log scale** everywhere. The **copilot-instructions.md is WRONG**.

### 4.2 Gompertz Test Bug

In [test_hazards.jl](MultistateModelsTests/unit/test_hazards.jl#L437-L440):
```julia
MultistateModels.set_parameters!(model, (h12 = [log(1.5), log(0.5)], ...))
```

This incorrectly uses `log(1.5)` for the shape parameter. Since Gompertz shape is NOT log-transformed, this sets shape = 0.405 instead of the intended shape = 1.5.

The test still passes because it computes expected values from the retrieved parameters (which are correct), not from the intended parameters.

---

## 5. Detailed Formula Comparison

### 5.1 Exponential

| Aspect | Documentation | Code | Consistent? |
|--------|---------------|------|-------------|
| Storage | `[log(λ)]` | `[log(λ)]` | ✅ Yes |
| Formula | h(t) = λ | h(t) = rate | ✅ Yes |
| Transform | exp() to rate | exp() applied in transforms.jl | ✅ Yes |

### 5.2 Weibull

| Aspect | copilot-instructions | Code | docs/index.md |
|--------|---------------------|------|---------------|
| Shape storage | `κ` (identity) | `log(κ)` | Not specified |
| Scale storage | `log(λ)` | `log(λ)` | Not specified |
| Formula | Not specified | h(t) = κλt^(κ-1) | h(t) = κλt^(κ-1) |

**Code-internal consistency**: ✅ Yes  
**copilot-instructions accuracy**: ❌ No (shape should be `log(κ)`)

### 5.3 Gompertz

| Aspect | Documentation | Code | Consistent? |
|--------|---------------|------|-------------|
| Shape storage | `a` (identity) | `a` (identity) | ✅ Yes |
| Rate storage | `log(b)` | `log(b)` | ✅ Yes |
| Formula | h(t) = b·exp(a·t) | h(t) = rate·exp(shape·t) | ✅ Yes |
| Transform | shape: identity, rate: exp | Correct in transforms.jl | ✅ Yes |

### 5.4 Spline

| Aspect | Documentation | Code | Consistent? |
|--------|---------------|------|-------------|
| Storage | `[β...]` (log-hazard) | Log-scale coefficients | ✅ Yes |
| Transform | Internal via _spline_ests2coefs | No transform in transforms.jl | ✅ Yes |

---

## 6. Recommended Actions

### 6.1 HIGH PRIORITY: Fix copilot-instructions.md

Update the hazard parameterizations table:

```markdown
| Family | Parameters | Storage (estimation scale) |
|--------|------------|---------------------------|
| Exponential | rate λ | `[log(λ)]` |
| Weibull | shape κ, scale λ | `[log(κ), log(λ)]` |
| Gompertz | shape a, rate b | `[a, log(b)]` |
| Spline | coefficients β | `[β...]` (log-hazard scale) |
```

Also update Common Gotchas:
```markdown
2. **Weibull shape**: Stored on **log scale** (both shape and scale are log-transformed)
```

### 6.2 MEDIUM PRIORITY: Fix Gompertz test

In [test_hazards.jl](MultistateModelsTests/unit/test_hazards.jl#L437):

Change:
```julia
MultistateModels.set_parameters!(model, (h12 = [log(1.5), log(0.5)], ...))
```
To:
```julia
MultistateModels.set_parameters!(model, (h12 = [1.5, log(0.5)], ...))
```

### 6.3 LOW PRIORITY: Update WEIBULL_BUG_ANALYSIS.md

The file [WEIBULL_BUG_ANALYSIS.md](MultistateModelsTests/fixtures/WEIBULL_BUG_ANALYSIS.md) appears to be outdated analysis. Either:
- Delete it if the "bug" was a misunderstanding
- Update it to reflect the actual (consistent) parameterization

---

## 7. Verification Checklist

| Item | Status |
|------|--------|
| Exponential: log(rate) storage | ✅ Consistent |
| Exponential: exp() transform | ✅ Consistent |
| Weibull: log(shape) storage | ✅ Code consistent, docs wrong |
| Weibull: log(scale) storage | ✅ Consistent |
| Weibull: exp() to both params | ✅ Consistent |
| Weibull: h(t) = κλt^(κ-1) formula | ✅ Consistent |
| Gompertz: identity(shape) storage | ✅ Consistent |
| Gompertz: log(rate) storage | ✅ Consistent |
| Gompertz: shape identity, rate exp | ✅ Consistent |
| Gompertz: h(t) = b·exp(a·t) formula | ✅ Consistent |
| Spline: log-scale coefficients | ✅ Consistent |

---

## 8. Files Reviewed

### Source Code
- [src/hazard/generators.jl](src/hazard/generators.jl) - Hazard function generation
- [src/hazard/time_transform.jl](src/hazard/time_transform.jl) - Time transform optimization
- [src/hazard/evaluation.jl](src/hazard/evaluation.jl) - Hazard evaluation API
- [src/utilities/transforms.jl](src/utilities/transforms.jl) - Parameter scale transformations
- [src/utilities/initialization.jl](src/utilities/initialization.jl) - Parameter initialization

### Documentation
- [.github/copilot-instructions.md](.github/copilot-instructions.md) - Project conventions
- [docs/src/index.md](docs/src/index.md) - User documentation

### Tests
- [MultistateModelsTests/unit/test_hazards.jl](MultistateModelsTests/unit/test_hazards.jl) - Unit tests
- [MultistateModelsTests/longtests/longtest_parametric_suite.jl](MultistateModelsTests/longtests/longtest_parametric_suite.jl) - Integration tests
- [MultistateModelsTests/longtests/longtest_aft_suite.jl](MultistateModelsTests/longtests/longtest_aft_suite.jl) - AFT tests

---

*Report generated: January 3, 2026*
