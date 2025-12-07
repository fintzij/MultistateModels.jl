---
title: "Inference Long Tests - Illness-Death Model"
format:
    html:
        theme:
            light: flatly
            dark: darkly
        highlight-style: atom-one-dark
---

# Inference Long Tests - Illness-Death Model

_Generated: 2025-12-06  (updated non-MCEM tests)_

_Julia: 1.12.2, Threads: 10_

This document contains results from inference validation tests using a 3-state 
illness-death model:

```
Healthy (1) ──→ Ill (2) ──→ Dead (3)
     │                         ↑
     └─────────────────────────┘
```

All tests use n = 1000 subjects.

---

## Summary (Non-MCEM Inference Longtests)

| Test | Family | N | Status |
|------|--------|---|--------|
| Exponential (exact) | exp | 1000 | ✅ PASS |
| Weibull (exact) | wei | 1000 | ✅ PASS |
| Gompertz (exact) | gom | 1000 | ✅ PASS |
| Exponential + Covariate (exact) | exp | 1000 | ✅ PASS |
| Weibull AFT (exact) | wei-aft | 1000 | ✅ PASS |
| Phase-Type Exact (2-phase) | phasetype | 1000 | ✅ PASS |
| Phase-Type Panel & Mixed (2-phase) | phasetype | 1000 | ✅ PASS |

---

## Parameter Estimates and Relative Errors

| Test | Parameter | True | Estimate | Rel. Error (%) | SE | 95% CI |
|------|-----------|------|----------|----------------|-----|--------|
| Exponential | h12 (Healthy→Ill) | -1.8971 | -1.8907 | -0.34 | 0.0380 | (-1.9651, -1.8162) |
|  | h13 (Healthy→Dead) | -2.9957 | -2.8826 | -3.78 | 0.0624 | (-3.0049, -2.7604) |
|  | h23 (Ill→Dead) | -1.6094 | -1.5351 | -4.62 | 0.0407 | (-1.6148, -1.4554) |
| Weibull | h12_scale | -2.1203 | -2.1867 | 3.13 | 0.0748 | (-2.3333, -2.0401) |
|  | h12_shape | 0.5878 | 0.6167 | 4.92 | 0.0273 | (0.5631, 0.6703) |
|  | h13_scale | -3.2189 | -3.1670 | -1.61 | 0.1304 | (-3.4226, -2.9114) |
|  | h13_shape | 0.4055 | 0.4003 | -1.27 | 0.0592 | (0.2843, 0.5163) |
|  | h23_scale | -1.8971 | -1.8862 | -0.57 | 0.0728 | (-2.0289, -1.7435) |
|  | h23_shape | 0.6931 | 0.6875 | -0.82 | 0.0277 | (0.6333, 0.7417) |
| Gompertz | h12_scale | 0.4055 | 0.4120 | 1.62 | 0.0786 | (0.2580, 0.5661) |
|  | h12_shape | -2.3026 | -2.3086 | 0.26 | 0.0564 | (-2.4190, -2.1981) |
|  | h13_scale | -0.4700 | -0.4176 | -11.15 | 0.1708 | (-0.7524, -0.0827) |
|  | h13_shape | -2.5257 | -2.5540 | 1.12 | 0.1324 | (-2.8136, -2.2945) |
|  | h23_scale | 0.6931 | 0.6994 | 0.91 | 0.0890 | (0.5250, 0.8738) |
|  | h23_shape | -2.3026 | -2.3131 | 0.46 | 0.0656 | (-2.4416, -2.1846) |
| Exponential + Covariate | h12_beta | 0.5000 | 0.5450 | 9.00 | 0.0534 | (0.4404, 0.6496) |
|  | h12_intercept | -2.1203 | -2.1586 | 1.81 | 0.0382 | (-2.2335, -2.0837) |
|  | h13_beta | 0.5000 | 0.5529 | 10.57 | 0.0919 | (0.3727, 0.7330) |
|  | h13_intercept | -3.2189 | -3.2500 | 0.97 | 0.0659 | (-3.3792, -3.1207) |
|  | h23_beta | 0.4000 | 0.3351 | -16.23 | 0.0589 | (0.2197, 0.4505) |
|  | h23_intercept | -1.8971 | -1.8471 | -2.64 | 0.0438 | (-1.9329, -1.7612) |
| Weibull AFT | h12_beta | 0.3000 | 0.2885 | -3.82 | 0.0271 | (0.2355, 0.3416) |
|  | h12_scale | -2.1203 | -2.2056 | 4.02 | 0.0582 | (-2.3197, -2.0915) |
|  | h12_shape | 0.5878 | 0.6085 | 3.53 | 0.0192 | (0.5709, 0.6461) |
|  | h13_beta | 0.4000 | 0.3617 | -9.57 | 0.0676 | (0.2292, 0.4943) |
|  | h13_scale | -3.2189 | -3.1772 | -1.29 | 0.1053 | (-3.3836, -2.9709) |
|  | h13_shape | 0.4055 | 0.4163 | 2.67 | 0.0415 | (0.3350, 0.4975) |
|  | h23_beta | 0.2000 | 0.2263 | 13.14 | 0.0259 | (0.1754, 0.2771) |
|  | h23_scale | -1.8971 | -1.7703 | -6.68 | 0.0551 | (-1.8783, -1.6624) |
|  | h23_shape | 0.6931 | 0.6515 | -6.00 | 0.0193 | (0.6136, 0.6895) |
| Phase-Type Exact (2-phase) | λ₁ (Healthy 1→2) | -0.2231 | -0.1853 | -16.98 | 0.0385 | (-0.2608, -0.1097) |
|  | λ₂ (Ill 3→4) | -0.5108 | -0.5037 | -1.39 | 0.0433 | (-0.5885, -0.4189) |
|  | μ₁₂_p1 (H1→Ill) | -1.2040 | -1.1363 | -5.62 | 0.0620 | (-1.2579, -1.0148) |
|  | μ₁₂_p2 (H2→Ill) | -0.9163 | -0.8816 | -3.79 | 0.0450 | (-0.9698, -0.7933) |
|  | μ₁₃_p1 (H1→Dead) | -2.3026 | -2.4923 | 8.24 | 0.1222 | (-2.7318, -2.2529) |
|  | μ₁₃_p2 (H2→Dead) | -1.8971 | -1.8891 | -0.42 | 0.0745 | (-2.0352, -1.7430) |
|  | μ₂₃_p3 (I3→Dead) | -1.3863 | -1.3951 | 0.63 | 0.0676 | (-1.5275, -1.2626) |
|  | μ₂₃_p4 (I4→Dead) | -1.0498 | -1.0871 | 3.55 | 0.0439 | (-1.1730, -1.0011) |
| Phase-Type Panel (2-phase) | h12 (Healthy→Ill) | NaN | -0.9663 | NaN | 0.0533 | (-1.0707, -0.8619) |
|  | h13 (Healthy→Dead) | NaN | -2.0856 | NaN | 0.1216 | (-2.3240, -1.8472) |
|  | h23 (Ill→Dead) | NaN | -1.1832 | NaN | 0.0509 | (-1.2830, -1.0834) |

---

## Diagnostic Plots

### Exponential

#### State Prevalence
![](assets/diagnostics/prevalence_exponential.png)

#### Cumulative Incidence
![](assets/diagnostics/cumincid_exponential.png)

### Weibull

#### State Prevalence
![](assets/diagnostics/prevalence_weibull.png)

#### Cumulative Incidence
![](assets/diagnostics/cumincid_weibull.png)

### Gompertz

#### State Prevalence
![](assets/diagnostics/prevalence_gompertz.png)

#### Cumulative Incidence
![](assets/diagnostics/cumincid_gompertz.png)

### Exponential + Covariate

#### State Prevalence
![](assets/diagnostics/prevalence_exponential_covariate.png)

#### Cumulative Incidence
![](assets/diagnostics/cumincid_exponential_covariate.png)

### Weibull AFT

#### State Prevalence
![](assets/diagnostics/prevalence_weibull_aft.png)

#### Cumulative Incidence
![](assets/diagnostics/cumincid_weibull_aft.png)

### Phase-Type Exact (2-phase)

#### State Prevalence
![](assets/diagnostics/prevalence_phase-type_exact_(2-phase).png)

#### Cumulative Incidence
![](assets/diagnostics/cumincid_phase-type_exact_(2-phase).png)

### Phase-Type Panel (2-phase)

#### State Prevalence
![](assets/diagnostics/prevalence_phase-type_panel_(2-phase).png)

#### Cumulative Incidence
![](assets/diagnostics/cumincid_phase-type_panel_(2-phase).png)


---

## Test Configuration

- RNG Seed: 2882347045
- Sample Size: 1000
- Simulation Trajectories: 5000
- Max Follow-up Time: 15.0
- Parallel Enabled: Yes (10 threads)

