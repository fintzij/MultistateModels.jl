# AFT + TVC Remediation Plan

## Executive Summary

**Bug Location**: Test validation formulas in `longtest_simulation_tvc.jl`  
**Root Cause**: Incorrect mathematical formula for AFT + TVC cumulative hazard  
**Package Status**: Implementation is CORRECT  
**Test Status**: Validation formulas are WRONG  

## Mathematical Background

For AFT with time-varying covariates, the correct approach is:

1. **Effective time**: τ(t) = ∫₀ᵗ exp(-β x(s)) ds
2. **Cumulative hazard**: H(t) = H₀(τ(t))

For piecewise-constant covariates with change points {t₁, t₂, ...}:
- τ(t) = Σₖ exp(-β xₖ) × Δtₖ (sum up to current segment)
- H(t) = H₀(τ)

### Current WRONG Weibull AFT formula:
```
cumhaz += scale * exp(-shape * beta * x[i]) * (t^shape - prev_t^shape)
```
This incorrectly computes segment-wise integrals with time-scaling applied to each segment.

### CORRECT Weibull AFT formula:
```
tau = 0.0
for each segment:
    tau += exp(-beta * x[i]) * (segment_length)
H = scale * tau^shape
```

### Current WRONG Gompertz AFT formula:
Uses segment-wise scaling of both shape and rate parameters.

### CORRECT Gompertz AFT formula:
```
tau = accumulated effective time
H = (rate/shape) * (exp(shape * tau) - 1)
```

## Remediation Tasks

### Phase 1: Fix Incorrect Test Formulas (CRITICAL)

**Task 1.1**: Replace `piecewise_cumhaz_wei_aft` (lines 119-132)
- Old: Segment-wise integral with time-scaling
- New: Effective time accumulation, then H₀(τ)

**Task 1.2**: Replace `piecewise_cumhaz_gom_aft` (lines 143-165)  
- Old: Segment-wise shape/rate scaling
- New: Effective time accumulation, then H₀(τ)

### Phase 2: Add Unit Tests for AFT + TVC

**Task 2.1**: Add machine-precision tests to `test_reversible_tvc_loglik.jl`
- Test Weibull AFT likelihood with TVC
- Test cumulative hazard matches manual calculation (rtol ≤ 1e-6)

### Phase 3: Verify Existing AFT Test in MCEM

**Task 3.1**: Review Test 7 in `longtest_mcem_tvc.jl` (lines 653-704)
- Verify it uses correct model specification
- Confirm parameter recovery tolerance is appropriate

### Phase 4: Run Verification

**Task 4.1**: Run corrected simulation distribution tests
**Task 4.2**: Run AFT likelihood unit tests  
**Task 4.3**: Run MCEM AFT test

## Detailed Fixes

### Fix 1: piecewise_cumhaz_wei_aft

```julia
function piecewise_cumhaz_wei_aft(t, shape, scale, beta, t_changes, x_values)
    # AFT: Accumulate effective time τ, then H(τ) = scale * τ^shape
    tau = 0.0
    prev_t = 0.0
    for (i, tc) in enumerate(t_changes)
        if t <= tc
            tau += exp(-beta * x_values[i]) * (t - prev_t)
            return scale * tau^shape
        else
            tau += exp(-beta * x_values[i]) * (tc - prev_t)
            prev_t = tc
        end
    end
    tau += exp(-beta * x_values[end]) * (t - prev_t)
    return scale * tau^shape
end
```

### Fix 2: piecewise_cumhaz_gom_aft

```julia
function piecewise_cumhaz_gom_aft(t, shape, scale, beta, t_changes, x_values)
    # AFT: Accumulate effective time τ, then H(τ) = (scale/shape) * (exp(shape*τ) - 1)
    tau = 0.0
    prev_t = 0.0
    for (i, tc) in enumerate(t_changes)
        if t <= tc
            tau += exp(-beta * x_values[i]) * (t - prev_t)
            return (scale / shape) * (exp(shape * tau) - 1.0)
        else
            tau += exp(-beta * x_values[i]) * (tc - prev_t)
            prev_t = tc
        end
    end
    tau += exp(-beta * x_values[end]) * (t - prev_t)
    return (scale / shape) * (exp(shape * tau) - 1.0)
end
```

## Verification Evidence

Weibull test case (verified in Julia):
- shape=1.5, scale=0.8, beta=0.5
- t=3.0, segments at [1.0, 2.0], x=[0.0, 1.0, 0.5]
- τ = 1.0*exp(0) + 1.0*exp(-0.5) + 1.0*exp(-0.25) = 1.0 + 0.6065 + 0.7788 = 2.3853
- H = 0.8 * 2.3853^1.5 = 2.9456

Package result: 2.945607 ✓ (matches to 6 decimal places)

---

## Execution Status

### Phase 1: Fix Incorrect Test Formulas ✅ COMPLETE

**Task 1.1**: ✅ Fixed `piecewise_cumhaz_wei_aft` 
- Changed from segment-wise `scale * exp(-shape*beta*x) * (t^shape - prev_t^shape)` 
- To effective time: `tau += exp(-beta*x) * (segment_length)`, then `scale * tau^shape`

**Task 1.2**: ✅ Fixed `piecewise_cumhaz_gom_aft`
- Changed from segment-wise scaled shape/rate
- To effective time: `tau += exp(-beta*x) * (segment_length)`, then `(scale/shape) * (exp(shape*tau) - 1)`

### Phase 2: Add Unit Tests for AFT + TVC ✅ COMPLETE

**Task 2.1**: ✅ Added machine-precision tests to `test_reversible_tvc_loglik.jl`
- Weibull AFT + TVC likelihood test
- Gompertz AFT + TVC likelihood test  
- Exponential AFT vs PH equivalence test
- All tests pass with rtol ≤ 1e-6

### Phase 3: Verify Existing AFT Test in MCEM ✅ VERIFIED

**Task 3.1**: ✅ Reviewed Test 7 in `longtest_mcem_tvc.jl` (lines 653-704)
- Uses correct model specification with `linpred_effect=:aft`
- Parameter recovery with 15% tolerance

### Phase 4: Run Verification ✅ COMPLETE

**Task 4.1**: ✅ All 9702 TVC simulation tests pass
**Task 4.2**: ✅ All 5 AFT likelihood unit tests pass  
**Task 4.3**: ✅ MCEM AFT test verified (part of existing suite)

## Summary

All identified issues have been rectified:

1. **Bug Fixed**: Test validation formulas for Weibull AFT and Gompertz AFT in 
   `longtest_simulation_tvc.jl` now correctly use effective time accumulation

2. **Tests Added**: Machine-precision unit tests for AFT + TVC in 
   `test_reversible_tvc_loglik.jl` verify the package implementation matches 
   manual mathematical calculation

3. **Verification**: All test suites pass, confirming the package implementation 
   was correct and only the test formulas needed fixing
