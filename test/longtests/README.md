# Inference Long Tests Plan

## Overview

This document describes the comprehensive 50-test inference validation suite for MultistateModels.jl. The tests validate parameter estimation across all hazard families, parameterizations, covariate types, and data observation types.

## Test Grid

| Dimension | Values |
|-----------|--------|
| **Families** | Exponential, Weibull, Gompertz, Phase-Type, Spline |
| **Parameterizations** | None (baseline only), PH, AFT |
| **Covariates** | None, Time-Fixed (TFC), Time-Varying (TVC) |
| **Data Types** | Exact (obstype=1), Panel (obstype=2) |

### Covariate Settings (5 total)
1. No covariates (baseline only)
2. PH + Time-Fixed Covariate
3. AFT + Time-Fixed Covariate  
4. PH + Time-Varying Covariate
5. AFT + Time-Varying Covariate

### Total Tests
5 families × 5 covariate settings × 2 data types = **50 tests**

---

## Model Specification

### State Structure
- **3-state progressive model**: State 1 → State 2 → State 3 (absorbing)
- **2 transitions**: h₁₂ (1→2), h₂₃ (2→3)

### True Parameters

| Family | Transition | Baseline Parameters | Covariate Effect |
|--------|------------|---------------------|------------------|
| Exponential | 1→2 | λ = 0.2 | β = 0.5 |
| Exponential | 2→3 | λ = 0.3 | β = 0.3 |
| Weibull | 1→2 | shape α = 1.5, scale λ = 0.15 | β = 0.5 |
| Weibull | 2→3 | shape α = 1.3, scale λ = 0.2 | β = 0.3 |
| Gompertz | 1→2 | scale η = 0.1, shape γ = 0.15 | β = 0.5 |
| Gompertz | 2→3 | scale η = 0.15, shape γ = 0.1 | β = 0.3 |
| Phase-Type | 1→2 | 2-phase Coxian (see below) | β = 0.5 |
| Phase-Type | 2→3 | 2-phase Coxian (see below) | β = 0.3 |
| Spline | 1→2 | degree=1, knots at sojourn quantiles | β = 0.5 |
| Spline | 2→3 | degree=1, knots at sojourn quantiles | β = 0.3 |

### Phase-Type Configuration
- **2-phase Coxian** per transient state (entry to first phase only)
- **5 phases total**: phases 1-2 → observed state 1, phases 3-4 → observed state 2, phase 5 → absorbing state 3
- Exits allowed from any phase to next observed state
- Phase transition rates: λ₁ = 0.4, λ₂ = 0.3 (state 1); λ₃ = 0.5, λ₄ = 0.35 (state 2)
- Exit probabilities: p₁ = 0.6, p₂ = 1.0 (from state 1); p₃ = 0.5, p₄ = 1.0 (from state 2)

### Spline Configuration
- **Degree**: 1 (linear spline)
- **Interior knots**: Placed at 20th, 40th, 60th, 80th percentiles of the true sojourn time distribution
- Knots computed from the baseline hazard (e.g., for Weibull with shape=1.5, scale=0.15)

---

## Simulation & Fitting Settings

### Sample Size
- **N = 1000** subjects per test

### Time Settings
- **MAX_TIME**: 15.0
- **PANEL_TIMES**: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
- **EVAL_TIMES**: [0.5, 1.0, 1.5, ..., 14.5, 15.0] (every 0.5 units)
- **TVC_CHANGEPOINT**: 5.0 (covariate x changes from 0 to 1 at t=5)

### Fitting Methods
| Data Type | Family | Method |
|-----------|--------|--------|
| Exact | All | MLE (`fit()`) |
| Panel | Exponential | Markov likelihood (`fit()`) |
| Panel | Phase-Type | Markov likelihood (`fit()`) |
| Panel | Weibull | MCEM with Markov proposal |
| Panel | Gompertz | MCEM with Markov proposal |
| Panel | Spline | MCEM with Markov proposal |

### MCEM Settings
```julia
MCEM_TOL = 0.01
MCEM_ESS_INITIAL = 100
MCEM_ESS_MAX = 2000
MCEM_MAX_ITER = 100
MCEM_PROPOSAL = :markov
```

### Pass Criterion
- **Max |relative error| ≤ 10%** for all estimated parameters

---

## File Structure

```
test/
├── longtest_config.jl          # Constants, TestResult struct, ALL_RESULTS
├── longtest_helpers.jl         # Shared utility functions
└── longtests/
    ├── README.md               # This file
    ├── exponential_tests.jl    # Tests 1-10
    ├── weibull_tests.jl        # Tests 11-20
    ├── gompertz_tests.jl       # Tests 21-30
    ├── phasetype_tests.jl      # Tests 31-40
    └── spline_tests.jl         # Tests 41-50
```

---

## Test Index

### Exponential Family (Tests 1-10)

| # | Test Name | Covariates | Data Type | Fitting Method |
|---|-----------|------------|-----------|----------------|
| 1 | `exp_nocov_exact` | None | Exact | MLE |
| 2 | `exp_nocov_panel` | None | Panel | Markov |
| 3 | `exp_ph_tfc_exact` | PH + TFC | Exact | MLE |
| 4 | `exp_ph_tfc_panel` | PH + TFC | Panel | Markov |
| 5 | `exp_aft_tfc_exact` | AFT + TFC | Exact | MLE |
| 6 | `exp_aft_tfc_panel` | AFT + TFC | Panel | Markov |
| 7 | `exp_ph_tvc_exact` | PH + TVC | Exact | MLE |
| 8 | `exp_ph_tvc_panel` | PH + TVC | Panel | Markov |
| 9 | `exp_aft_tvc_exact` | AFT + TVC | Exact | MLE |
| 10 | `exp_aft_tvc_panel` | AFT + TVC | Panel | Markov |

### Weibull Family (Tests 11-20)

| # | Test Name | Covariates | Data Type | Fitting Method |
|---|-----------|------------|-----------|----------------|
| 11 | `wei_nocov_exact` | None | Exact | MLE |
| 12 | `wei_nocov_panel` | None | Panel | MCEM (Markov) |
| 13 | `wei_ph_tfc_exact` | PH + TFC | Exact | MLE |
| 14 | `wei_ph_tfc_panel` | PH + TFC | Panel | MCEM (Markov) |
| 15 | `wei_aft_tfc_exact` | AFT + TFC | Exact | MLE |
| 16 | `wei_aft_tfc_panel` | AFT + TFC | Panel | MCEM (Markov) |
| 17 | `wei_ph_tvc_exact` | PH + TVC | Exact | MLE |
| 18 | `wei_ph_tvc_panel` | PH + TVC | Panel | MCEM (Markov) |
| 19 | `wei_aft_tvc_exact` | AFT + TVC | Exact | MLE |
| 20 | `wei_aft_tvc_panel` | AFT + TVC | Panel | MCEM (Markov) |

### Gompertz Family (Tests 21-30)

| # | Test Name | Covariates | Data Type | Fitting Method |
|---|-----------|------------|-----------|----------------|
| 21 | `gom_nocov_exact` | None | Exact | MLE |
| 22 | `gom_nocov_panel` | None | Panel | MCEM (Markov) |
| 23 | `gom_ph_tfc_exact` | PH + TFC | Exact | MLE |
| 24 | `gom_ph_tfc_panel` | PH + TFC | Panel | MCEM (Markov) |
| 25 | `gom_aft_tfc_exact` | AFT + TFC | Exact | MLE |
| 26 | `gom_aft_tfc_panel` | AFT + TFC | Panel | MCEM (Markov) |
| 27 | `gom_ph_tvc_exact` | PH + TVC | Exact | MLE |
| 28 | `gom_ph_tvc_panel` | PH + TVC | Panel | MCEM (Markov) |
| 29 | `gom_aft_tvc_exact` | AFT + TVC | Exact | MLE |
| 30 | `gom_aft_tvc_panel` | AFT + TVC | Panel | MCEM (Markov) |

### Phase-Type Family (Tests 31-40)

| # | Test Name | Covariates | Data Type | Fitting Method |
|---|-----------|------------|-----------|----------------|
| 31 | `phasetype_nocov_exact` | None | Exact | MLE |
| 32 | `phasetype_nocov_panel` | None | Panel | Markov |
| 33 | `phasetype_ph_tfc_exact` | PH + TFC | Exact | MLE |
| 34 | `phasetype_ph_tfc_panel` | PH + TFC | Panel | Markov |
| 35 | `phasetype_aft_tfc_exact` | AFT + TFC | Exact | MLE |
| 36 | `phasetype_aft_tfc_panel` | AFT + TFC | Panel | Markov |
| 37 | `phasetype_ph_tvc_exact` | PH + TVC | Exact | MLE |
| 38 | `phasetype_ph_tvc_panel` | PH + TVC | Panel | Markov |
| 39 | `phasetype_aft_tvc_exact` | AFT + TVC | Exact | MLE |
| 40 | `phasetype_aft_tvc_panel` | AFT + TVC | Panel | Markov |

### Spline Family (Tests 41-50)

| # | Test Name | Covariates | Data Type | Fitting Method |
|---|-----------|------------|-----------|----------------|
| 41 | `spline_nocov_exact` | None | Exact | MLE |
| 42 | `spline_nocov_panel` | None | Panel | MCEM (Markov) |
| 43 | `spline_ph_tfc_exact` | PH + TFC | Exact | MLE |
| 44 | `spline_ph_tfc_panel` | PH + TFC | Panel | MCEM (Markov) |
| 45 | `spline_aft_tfc_exact` | AFT + TFC | Exact | MLE |
| 46 | `spline_aft_tfc_panel` | AFT + TFC | Panel | MCEM (Markov) |
| 47 | `spline_ph_tvc_exact` | PH + TVC | Exact | MLE |
| 48 | `spline_ph_tvc_panel` | PH + TVC | Panel | MCEM (Markov) |
| 49 | `spline_aft_tvc_exact` | AFT + TVC | Exact | MLE |
| 50 | `spline_aft_tvc_panel` | AFT + TVC | Panel | MCEM (Markov) |

---

## Test Function Template

Each test function follows this structure:

```julia
function run_<family>_<covtype>_<datatype>()
    test_name = "<family>_<covtype>_<datatype>"
    @info "Running $test_name"
    
    # 1. Define true parameters
    
    # 2. Create hazard specifications
    h12 = Hazard(@formula(...), "<family>", 1, 2; link=:<link>)
    h23 = Hazard(@formula(...), "<family>", 2, 3; link=:<link>)
    
    # 3. Create data template (baseline, TFC, or TVC)
    dat = create_<template>_template(N_SUBJECTS; max_time=MAX_TIME)
    
    # 4. Create and parameterize simulation model
    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim, (...))
    
    # 5. Simulate data
    paths = simulate(msm_sim; paths=true, data=false)
    data = simulate(msm_sim; paths=false, data=true)  # or convert to panel
    
    # 6. Create and fit model
    msm_fit = multistatemodel(h12, h23; data=data)
    fitted = fit(msm_fit)  # or fit_mcem(msm_fit; proposal=:markov, ...)
    
    # 7. Extract estimates and compute relative errors
    
    # 8. Compute prevalence and cumulative incidence
    #    - True (from simulation paths)
    #    - Observed (from data)
    #    - Fitted (from fitted model simulation)
    
    # 9. Create TestResult and finalize
    result = TestResult(...)
    finalize_result!(result)
    push!(ALL_RESULTS, result)
    
    return result
end
```

---

## Diagnostics Output

Each test produces:

1. **Parameter estimates** with relative errors
2. **State prevalence curves** (3 states × 30 time points)
   - True (simulated with true parameters)
   - Observed (computed from data)
   - Fitted (simulated with fitted parameters)
3. **Cumulative incidence curves** (2 transitions × 30 time points)
   - True, Observed, Fitted

### Plots (per test)
- State prevalence over time (3 panels: State 1, State 2, State 3)
- Cumulative incidence (2 panels: 1→2, 2→3)

Each plot shows:
- **Black solid line**: Expected (true parameters)
- **Gray scatter points**: Observed (from data)
- **Blue dashed line**: Fitted (fitted parameters)

---

## Execution

### Run All Tests
```julia
include("test/longtest_config.jl")
include("test/longtest_helpers.jl")
include("test/longtests/exponential_tests.jl")
include("test/longtests/weibull_tests.jl")
include("test/longtests/gompertz_tests.jl")
include("test/longtests/phasetype_tests.jl")
include("test/longtests/spline_tests.jl")

# Run all
run_all_exponential_tests()
run_all_weibull_tests()
run_all_gompertz_tests()
run_all_phasetype_tests()
run_all_spline_tests()

# Generate report
generate_inference_report(ALL_RESULTS)
```

### Run Single Family
```julia
run_all_exponential_tests()
```

### Run Single Test
```julia
run_exp_nocov_exact()
```

---

## Report Structure

The generated HTML report includes:

1. **Executive Summary**
   - Overall pass rate (X/50)
   - Pass rate by family
   - Pass rate by data type
   - Pass rate by covariate setting

2. **Detailed Results Table**
   - All 50 tests with parameters, estimates, relative errors, pass/fail

3. **Diagnostic Plots**
   - Prevalence and cumulative incidence for each test
   - Visual comparison of expected vs observed vs fitted

4. **Failure Analysis**
   - List of failed tests with details
   - Potential causes and recommendations

---

## Implementation Status

| Component | Status |
|-----------|--------|
| `longtest_config.jl` | ✅ Created |
| `longtest_helpers.jl` | ✅ Created |
| `longtests/README.md` | ✅ This file |
| `longtests/exponential_tests.jl` | ✅ Created (10 tests) |
| `longtests/weibull_tests.jl` | ⏳ Pending |
| `longtests/gompertz_tests.jl` | ⏳ Pending |
| `longtests/phasetype_tests.jl` | ⏳ Pending |
| `longtests/spline_tests.jl` | ⏳ Pending |
| Report generator | ⏳ Pending |

---

## Notes

### AFT vs PH for Exponential
For exponential hazards, AFT and PH parameterizations are mathematically equivalent:
- AFT coefficient = -PH coefficient
- Both tests are included for API completeness

### Phase-Type Observations
- Phase-type models track internal phases but report observed states
- Panel data records observed state at panel times
- Phase-to-state mapping: `phase_to_state = Dict(1=>1, 2=>1, 3=>2, 4=>2, 5=>3)`

### Spline Knot Selection
Knots are placed at quantiles of the true sojourn distribution to ensure the spline can approximate the true hazard well. For tests, we use Weibull-like baseline truth and compute knots accordingly.

### Time-Varying Covariates
- All subjects have x=0 for t < 5.0
- All subjects have x=1 for t ≥ 5.0
- This is a deterministic changepoint, not random per subject
