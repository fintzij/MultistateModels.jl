# MultistateModels.jl Infrastructure Testing Walkthrough
**Date:** October 30, 2025  
**Branch:** infrastructure_changes  
**Purpose:** Systematically validate all new infrastructure through complete workflow

---

## Overview

This document provides a step-by-step plan to walk through the entire package workflow with the new infrastructure, testing each component in isolation before proceeding to the next. We'll build from the ground up, validating at each step.

**Approach:** 
- Start simple (2-state exponential, no covariates)
- Add complexity incrementally (covariates, more states, different families)
- Test each step thoroughly before moving forward
- Document any issues encountered

---

## Phase 1: Basic Model Creation (No Covariates)

### Step 1.1: Create Simple 2-State Model
**Goal:** Verify basic hazard creation and model construction work

**Test case:**
```julia
using MultistateModels
using DataFrames

# Simple 2-state model: Healthy (1) → Diseased (2)
# Exponential hazard, no covariates

# Create hazard specification
haz_1_to_2 = hazard(1, 2, family="exp")

# Inspect hazard object
println("Hazard type: ", typeof(haz_1_to_2))
println("Hazard family: ", haz_1_to_2.family)
println("Has covariates: ", haz_1_to_2.has_covariates)
println("Parameter names: ", haz_1_to_2.parnames)
println("Baseline params: ", haz_1_to_2.npar_baseline)
println("Total params: ", haz_1_to_2.npar_total)
```

**Expected output:**
- Type: `MarkovHazard`
- Family: `"exp"`
- Has covariates: `false`
- Parameter names: `[:intercept]` (or similar)
- Baseline params: `1`
- Total params: `1`

**Validation:**
- [ ] Hazard object created without error
- [ ] Correct type returned
- [ ] Fields populated correctly
- [ ] hazard_fn and cumhaz_fn are callable functions

### Step 1.2: Test Hazard Functions
**Goal:** Verify runtime-generated functions work correctly

**Test case:**
```julia
# Test hazard function evaluation
params = (intercept = -1.0,)  # Named parameters
t = 1.0
covars = NamedTuple()  # Empty for no covariates

# Call hazard function
h_val = haz_1_to_2.hazard_fn(t, params, covars; give_log=false)
log_h_val = haz_1_to_2.hazard_fn(t, params, covars; give_log=true)

println("h(1.0) = ", h_val)
println("log h(1.0) = ", log_h_val)
println("Expected: h = exp(-1.0) = ", exp(-1.0))

# Test cumulative hazard
cumhaz_val = haz_1_to_2.cumhaz_fn(0.0, 1.0, params, covars)
println("Cumulative hazard [0, 1] = ", cumhaz_val)
println("Expected: ", 1.0 * exp(-1.0))
```

**Expected output:**
- `h(1.0) ≈ 0.368` (exp(-1))
- `log h(1.0) ≈ -1.0`
- `cumhaz ≈ 0.368`

**Validation:**
- [ ] Hazard evaluates without error
- [ ] Correct numerical values
- [ ] log vs non-log consistent
- [ ] Cumulative hazard correct

### Step 1.3: Initialize Parameters
**Goal:** Verify parameter initialization works

**Test case:**
```julia
# Test init_par
crude_rate = 0.0
initial_params = init_par(haz_1_to_2, crude_rate)

println("Initial parameters: ", initial_params)
println("Expected: [0.0] (crude log rate)")
```

**Validation:**
- [ ] init_par returns vector
- [ ] Correct length (1 for exponential)
- [ ] Reasonable values

### Step 1.4: Create Minimal Dataset
**Goal:** Create smallest possible dataset for testing

**Test case:**
```julia
# Minimal 2-state data: 10 subjects, all transition
dat = DataFrame(
    id = 1:10,
    tstart = zeros(10),
    tstop = rand(10) .* 2.0,  # Random event times 0-2
    statefrom = ones(Int, 10),
    stateto = fill(2, 10)
)

println("Sample data:")
println(first(dat, 3))
```

**Validation:**
- [ ] Data created without error
- [ ] Correct structure (id, tstart, tstop, statefrom, stateto)
- [ ] Valid transitions (all 1 → 2)

### Step 1.5: Build Model Object
**Goal:** Construct full MultistateModel

**Test case:**
```julia
# Build model
model = multistatemodel(haz_1_to_2; data=dat)

println("Model type: ", typeof(model))
println("Number of hazards: ", length(model.hazards))
println("Number of subjects: ", model.nsubj)
println("Total parameters: ", model.npar)
```

**Expected output:**
- Model type: `MultistateModel`
- 1 hazard
- 10 subjects
- 1 parameter

**Validation:**
- [ ] Model created without error
- [ ] Correct number of hazards
- [ ] Correct number of subjects
- [ ] Correct total parameters
- [ ] Model fields populated correctly

**STOP HERE IF ANY ISSUES - Debug before proceeding**

---

## Phase 2: Model Fitting (No Covariates)

### Step 2.1: Test Likelihood Evaluation
**Goal:** Verify likelihood computation works

**Test case:**
```julia
# Test likelihood at crude estimates
crude_params = init_par(haz_1_to_2, log(1.0))
println("Testing at parameters: ", crude_params)

# NEED TO CHECK: How are parameters structured for likelihood?
# Is it flat vector or ParameterHandling structure?
# This step will reveal the interface

# Try calling likelihood (exact function name to be determined)
# loglik_val = loglik(model, crude_params)
# println("Log-likelihood: ", loglik_val)
```

**Questions to answer:**
- [ ] What is the likelihood function called?
- [ ] Does it expect flat vector or ParameterHandling structure?
- [ ] Does it return scalar log-likelihood?
- [ ] Are there helper functions for parameter structuring?

**Validation:**
- [ ] Likelihood evaluates without error
- [ ] Returns finite value
- [ ] Reasonable magnitude

### Step 2.2: Test Parameter Handling Structure
**Goal:** Understand ParameterHandling integration

**Test case:**
```julia
# How do we build parameter structure?
# Expected something like:
# params = (
#     hazard_1to2 = (intercept = 0.0,)
# )

# How do we flatten/unflatten?
# flat, unflatten = ParameterHandling.value_flatten(params)

# Need to identify actual functions in codebase
```

**Questions to answer:**
- [ ] How are parameter structures built from model?
- [ ] What functions handle flatten/unflatten?
- [ ] Are there model methods for this?
- [ ] How are constraints applied?

### Step 2.3: Manual Optimization
**Goal:** Verify we can optimize parameters

**Test case:**
```julia
using Optim

# Define objective function
function objective(flat_params)
    # Unflatten parameters
    params = unflatten(flat_params)  # Need to determine unflatten function
    
    # Compute negative log-likelihood
    -loglik(model, params)  # Need to determine loglik function
end

# Initial parameters
initial_flat = init_par(haz_1_to_2, 0.0)  # Or flattened version?

# Optimize
result = optimize(objective, initial_flat, BFGS())

println("Optimization converged: ", Optim.converged(result))
println("Final parameters: ", result.minimizer)
println("Final log-likelihood: ", -result.minimum)
```

**Validation:**
- [ ] Optimization runs without error
- [ ] Converges successfully
- [ ] Parameters reasonable (log hazard around -1 to 1)
- [ ] Likelihood improved from initial

### Step 2.4: Use Built-in Fitting Function
**Goal:** Test high-level fitting interface

**Test case:**
```julia
# Fit model using package function
# (Need to identify actual fitting function)
# fitted_model = fit(model) or similar?

# Check what's returned
# println("Fitted parameters: ", coef(fitted_model))
# println("Log-likelihood: ", loglikelihood(fitted_model))
```

**Questions to answer:**
- [ ] What is the model fitting function?
- [ ] What does it return?
- [ ] How do we extract results (coef, vcov, loglik, etc.)?
- [ ] Are there print/show methods?

**STOP HERE IF ANY ISSUES - Debug before proceeding**

---

## Phase 3: Add Covariates (2-State Model)

### Step 3.1: Create Model with Covariates
**Goal:** Test name-based covariate matching

**Test case:**
```julia
# Create hazard with covariates
haz_1_to_2_cov = hazard(1, 2, family="exp", formula=@formula(0 ~ age + sex))

println("Parameter names: ", haz_1_to_2_cov.parnames)
println("Expected: [:intercept, :age, :sex] or similar")
println("Covariate names: ", extract_covar_names(haz_1_to_2_cov.parnames))
```

**Expected output:**
- Parameter names include intercept + covariates
- Covariate extraction works

**Validation:**
- [ ] Hazard created with formula
- [ ] Parameter names include covariates
- [ ] Still MarkovHazard type
- [ ] has_covariates = true

### Step 3.2: Create Data with Covariates
**Goal:** Add covariate columns

**Test case:**
```julia
# Add covariates to data
dat_cov = copy(dat)
dat_cov.age = rand(10) .* 50 .+ 20  # Ages 20-70
dat_cov.sex = rand([0, 1], 10)      # Binary

println("Data with covariates:")
println(first(dat_cov, 3))
```

**Validation:**
- [ ] Covariate columns added
- [ ] Column names match formula

### Step 3.3: Test Covariate Extraction
**Goal:** Verify name-based matching

**Test case:**
```julia
# Test helper function
using DataFrames

row = dat_cov[1, :]
covar_names = [:age, :sex]
covars = extract_covariates(row, covar_names)

println("Row: ", row)
println("Extracted covariates: ", covars)
println("Type: ", typeof(covars))
println("Expected: NamedTuple with fields age, sex")

# Test accessing by name
println("Age: ", covars.age)
println("Sex: ", covars.sex)
```

**Validation:**
- [ ] Covariates extracted as NamedTuple
- [ ] Correct values
- [ ] Can access by name

### Step 3.4: Test Hazard Evaluation with Covariates
**Goal:** Verify runtime functions handle covariates

**Test case:**
```julia
# Parameters for exponential with covariates
params_cov = (intercept = -1.0, age = 0.02, sex = 0.5)
t = 1.0
row = dat_cov[1, :]
covars = extract_covariates(row, [:age, :sex])

# Evaluate hazard
h_val = haz_1_to_2_cov.hazard_fn(t, params_cov, covars; give_log=false)

# Manual calculation
expected_log_h = -1.0 + 0.02 * covars.age + 0.5 * covars.sex
expected_h = exp(expected_log_h)

println("Computed h: ", h_val)
println("Expected h: ", expected_h)
println("Match: ", isapprox(h_val, expected_h))
```

**Validation:**
- [ ] Hazard evaluates with covariates
- [ ] Matches manual calculation
- [ ] No index errors

### Step 3.5: Build and Fit Model with Covariates
**Goal:** Full workflow with covariates

**Test case:**
```julia
# Build model
model_cov = multistatemodel(haz_1_to_2_cov; data=dat_cov)

println("Model parameters: ", model_cov.npar)
println("Expected: 3 (intercept + age + sex)")

# Fit model
# fitted_model_cov = fit(model_cov)
# println("Fitted parameters: ", coef(fitted_model_cov))
```

**Validation:**
- [ ] Model created with covariates
- [ ] Correct parameter count
- [ ] Fitting works
- [ ] Results interpretable

**STOP HERE IF ANY ISSUES**

---

## Phase 4: Multi-State Models (3+ States)

### Step 4.1: Create 3-State Illness-Death Model
**Goal:** Multiple transitions, no covariates

**Test case:**
```julia
# States: 1=Healthy, 2=Ill, 3=Dead
# Transitions: 1→2, 1→3, 2→3

haz_1_to_2 = hazard(1, 2, family="exp")
haz_1_to_3 = hazard(1, 3, family="exp")
haz_2_to_3 = hazard(2, 3, family="exp")

println("Created 3 hazards")
```

**Validation:**
- [ ] All hazards created
- [ ] Different hazard names/IDs

### Step 4.2: Create Multi-State Data
**Goal:** Data with multiple transition types

**Test case:**
```julia
# Simple illness-death data
# 5 subjects: 1→2→3
# 5 subjects: 1→3 (skip illness)

dat_3state = DataFrame(
    id = [1,1, 2,2, 3,3, 4,4, 5,5, 6, 7, 8, 9, 10],
    tstart = [0.0,1.0, 0.0,1.5, 0.0,2.0, 0.0,1.2, 0.0,0.8,  0.0, 0.0, 0.0, 0.0, 0.0],
    tstop =  [1.0,2.0, 1.5,3.0, 2.0,3.5, 1.2,2.5, 0.8,2.0,  1.0, 1.5, 2.0, 1.8, 2.2],
    statefrom = [1,2, 1,2, 1,2, 1,2, 1,2,  1, 1, 1, 1, 1],
    stateto =   [2,3, 2,3, 2,3, 2,3, 2,3,  3, 3, 3, 3, 3]
)

println("Multi-state data:")
println(dat_3state)
```

**Validation:**
- [ ] Data structure correct
- [ ] Multiple rows per subject
- [ ] All transition types represented

### Step 4.3: Build 3-State Model
**Goal:** Model with multiple hazards

**Test case:**
```julia
model_3state = multistatemodel(
    haz_1_to_2,
    haz_1_to_3,
    haz_2_to_3;
    data = dat_3state
)

println("Number of hazards: ", length(model_3state.hazards))
println("Total parameters: ", model_3state.npar)
println("Number of subjects: ", model_3state.nsubj)
```

**Expected:**
- 3 hazards
- 3 parameters (one per hazard)
- 10 subjects

**Validation:**
- [ ] Model created
- [ ] Correct hazard count
- [ ] Correct parameter count
- [ ] All transitions represented

### Step 4.4: Fit 3-State Model
**Goal:** Optimization with multiple hazards

**Test case:**
```julia
# Fit model
# fitted_3state = fit(model_3state)

# Check results
# println("Hazard 1→2: ", fitted_3state.params.hazard_1to2)
# println("Hazard 1→3: ", fitted_3state.params.hazard_1to3)
# println("Hazard 2→3: ", fitted_3state.params.hazard_2to3)
```

**Validation:**
- [ ] Optimization converges
- [ ] All hazards estimated
- [ ] Results reasonable

**STOP HERE IF ANY ISSUES**

---

## Phase 5: Different Hazard Families

### Step 5.1: Weibull Hazard (Semi-Markov)
**Goal:** Test SemiMarkovHazard type

**Test case:**
```julia
# Create Weibull hazard
haz_wei = hazard(1, 2, family="wei")

println("Type: ", typeof(haz_wei))
println("Expected: SemiMarkovHazard")
println("Parameters: ", haz_wei.parnames)
println("Expected: shape and scale (or similar)")
```

**Validation:**
- [ ] Correct type (SemiMarkovHazard)
- [ ] 2 baseline parameters
- [ ] Functions generated

### Step 5.2: Test Weibull Evaluation
**Goal:** Verify Weibull hazard math

**Test case:**
```julia
# Weibull: h(t) = (shape/scale) * (t/scale)^(shape-1)
params_wei = (shape = 2.0, scale = 1.0)
t = 0.5
covars = NamedTuple()

h_val = haz_wei.hazard_fn(t, params_wei, covars; give_log=false)

# Manual calculation
expected_h = (2.0/1.0) * (0.5/1.0)^(2.0-1.0)
println("Computed: ", h_val)
println("Expected: ", expected_h)
println("Match: ", isapprox(h_val, expected_h))
```

**Validation:**
- [ ] Weibull evaluation correct
- [ ] Cumulative hazard correct

### Step 5.3: Gompertz Hazard
**Goal:** Test another SemiMarkovHazard

**Test case:**
```julia
haz_gom = hazard(1, 2, family="gom")

println("Type: ", typeof(haz_gom))
println("Parameters: ", haz_gom.parnames)

# Test evaluation
params_gom = (shape = 0.1, scale = 1.0)
h_val = haz_gom.hazard_fn(1.0, params_gom, NamedTuple(); give_log=false)
println("Gompertz h(1.0): ", h_val)
```

**Validation:**
- [ ] Gompertz creates SemiMarkovHazard
- [ ] Evaluation works

### Step 5.4: Mixed Model (Different Families)
**Goal:** One model with multiple families

**Test case:**
```julia
# Illness-death with different families
haz_1_to_2_exp = hazard(1, 2, family="exp")
haz_1_to_3_wei = hazard(1, 3, family="wei")
haz_2_to_3_gom = hazard(2, 3, family="gom")

model_mixed = multistatemodel(
    haz_1_to_2_exp,
    haz_1_to_3_wei,
    haz_2_to_3_gom;
    data = dat_3state
)

println("Total parameters: ", model_mixed.npar)
println("Expected: 1 + 2 + 2 = 5")
```

**Validation:**
- [ ] Mixed model creates successfully
- [ ] Correct parameter count
- [ ] Each hazard has correct type

### Step 5.5: Fit Mixed Model
**Goal:** Optimization with different families

**Test case:**
```julia
# Fit mixed model
# fitted_mixed = fit(model_mixed)

# Check parameter structure
# Should have different structure for each hazard
```

**Validation:**
- [ ] Converges
- [ ] Each hazard has appropriate parameters

**STOP HERE IF ANY ISSUES**

---

## Phase 6: Different Covariates per Hazard

### Step 6.1: Critical Name-Based Test
**Goal:** Verify name-based matching prevents bugs

**Test case:**
```julia
# Create data with multiple covariates
dat_multi_cov = DataFrame(
    id = 1:10,
    tstart = zeros(10),
    tstop = rand(10) .* 2.0,
    statefrom = ones(Int, 10),
    stateto = fill(2, 10),
    age = rand(10) .* 50 .+ 20,
    sex = rand([0, 1], 10),
    treatment = rand([0, 1], 10)
)

# Different covariates per hazard - THIS IS THE KEY TEST
haz_1_to_2_age_sex = hazard(1, 2, family="exp", formula=@formula(0 ~ age + sex))
haz_2_to_3_treatment = hazard(2, 3, family="exp", formula=@formula(0 ~ treatment))

println("Hazard 1→2 covariates: ", extract_covar_names(haz_1_to_2_age_sex.parnames))
println("Hazard 2→3 covariates: ", extract_covar_names(haz_2_to_3_treatment.parnames))
```

**Validation:**
- [ ] Each hazard has different covariates
- [ ] Covariate names extracted correctly

### Step 6.2: Test Evaluation with Different Covariates
**Goal:** Ensure no index confusion

**Test case:**
```julia
row = dat_multi_cov[1, :]

# Extract for hazard 1→2 (age, sex)
covars_12 = extract_covariates(row, [:age, :sex])
params_12 = (intercept = 0.0, age = 0.02, sex = 0.5)
h_12 = haz_1_to_2_age_sex.hazard_fn(1.0, params_12, covars_12; give_log=false)

# Extract for hazard 2→3 (treatment only)
covars_23 = extract_covariates(row, [:treatment])
params_23 = (intercept = 0.0, treatment = 0.3)
h_23 = haz_2_to_3_treatment.hazard_fn(1.0, params_23, covars_23; give_log=false)

println("h_12 evaluated: ", h_12)
println("h_23 evaluated: ", h_23)

# These should use DIFFERENT covariate values
# Old index-based approach would fail here!
```

**Validation:**
- [ ] Both hazards evaluate correctly
- [ ] No mixing of covariate values
- [ ] Named access prevents bugs

### Step 6.3: Build and Fit Model
**Goal:** Full workflow with different covariates

**Test case:**
```julia
# Extend data for 3-state
dat_multi_cov.statefrom = vcat(ones(Int, 5), fill(2, 5))
dat_multi_cov.stateto = vcat(fill(2, 5), fill(3, 5))

model_diff_cov = multistatemodel(
    haz_1_to_2_age_sex,
    haz_2_to_3_treatment;
    data = dat_multi_cov
)

println("Parameters: ", model_diff_cov.npar)
println("Expected: 3 + 2 = 5")

# Fit
# fitted_diff_cov = fit(model_diff_cov)
```

**Validation:**
- [ ] Model builds correctly
- [ ] Parameter count correct
- [ ] Fitting works
- [ ] Covariate effects interpretable

**STOP HERE IF ANY ISSUES**

---

## Phase 7: Simulation

### Step 7.1: Simulate from Fitted Model
**Goal:** Test path simulation

**Test case:**
```julia
# Use simple 2-state model
# fitted_simple = fit(model)  # From Phase 2

# Simulate paths
# simulated_paths = simulate(fitted_simple, n=100)

# Check output structure
# println("Simulated data structure:")
# println(first(simulated_paths, 5))
```

**Questions to answer:**
- [ ] What is simulation function called?
- [ ] What arguments does it take?
- [ ] What format is output?
- [ ] Can we specify subject covariates?

### Step 7.2: Simulate with Covariates
**Goal:** Verify covariate handling in simulation

**Test case:**
```julia
# Create new covariate data for simulation
sim_covars = DataFrame(
    id = 1:50,
    age = rand(50) .* 50 .+ 20,
    sex = rand([0, 1], 50)
)

# Simulate using model with covariates
# sim_paths_cov = simulate(fitted_model_cov, newdata=sim_covars)

# Verify covariates used correctly
```

**Validation:**
- [ ] Simulation accepts covariate data
- [ ] Uses name-based matching
- [ ] Paths vary by covariates

### Step 7.3: Check survprob Function
**Goal:** Verify survival probability calculation

**Test case:**
```julia
# This should now use subjdat parameter
# Test that it's being called correctly in simulation

# May need to call directly
# subj_dat = dat[1, :]
# s_prob = survprob(0.0, 1.0, params, 1, totalhaz, hazards, subj_dat)
# println("Survival probability: ", s_prob)
```

**Validation:**
- [ ] survprob works with subjdat
- [ ] No missing parameter errors
- [ ] Reasonable probabilities (0-1)

**STOP HERE IF ANY ISSUES**

---

## Phase 8: Splines (If Implemented)

### Step 8.1: Create Spline Hazard
**Goal:** Test SplineHazard type

**Test case:**
```julia
# Create spline hazard
# haz_spline = hazard(1, 2, family="sp", degree=3, knots=[0.5, 1.0, 1.5, 2.0])

# println("Type: ", typeof(haz_spline))
# println("Expected: SplineHazard")
# println("Parameters: ", haz_spline.parnames)
```

**Validation:**
- [ ] SplineHazard created
- [ ] Correct number of parameters (degree + internal knots + 1?)
- [ ] Functions generated

### Step 8.2: Test Spline Evaluation
**Goal:** Verify spline hazard works

**Test case:**
```julia
# Test evaluation at various time points
# times = [0.1, 0.5, 1.0, 1.5, 2.0]
# for t in times
#     h = haz_spline.hazard_fn(t, params, covars; give_log=false)
#     println("h($t) = ", h)
# end
```

**Validation:**
- [ ] Evaluates at all time points
- [ ] Smooth behavior
- [ ] Positive hazard values

### Step 8.3: Fit Model with Splines
**Goal:** Full spline workflow

**Test case:**
```julia
# model_spline = multistatemodel(haz_spline; data=dat)
# fitted_spline = fit(model_spline)
```

**Validation:**
- [ ] Converges
- [ ] Spline coefficients reasonable
- [ ] Hazard shape interpretable

**STOP HERE IF ANY ISSUES**

---

## Phase 9: Integration Tests

### Step 9.1: Full Illness-Death Example
**Goal:** Realistic complete workflow

**Test case:**
```julia
# Create realistic illness-death data
# - 1000 subjects
# - Covariates: age, sex, treatment
# - Different hazard families
# - Some censoring

# Build model with:
# - 1→2: Weibull with age + sex
# - 1→3: Exponential with age
# - 2→3: Gompertz with treatment

# Fit model
# Check convergence
# Examine results
# Simulate from fitted model
# Compare to original data
```

**Validation:**
- [ ] Entire workflow completes
- [ ] Results reasonable
- [ ] Simulation produces realistic data

### Step 9.2: Compare with Test Suite
**Goal:** Ensure walkthrough matches test expectations

**Test case:**
```julia
# Run existing tests
# using Pkg
# Pkg.test()

# Compare results from manual walkthrough with test suite
```

**Validation:**
- [ ] Tests pass
- [ ] Manual results consistent with tests
- [ ] No unexpected differences

---

## Issues Tracking

### Issues Encountered

**Phase 1:**
- [ ] Issue 1: [Description]
  - Error message:
  - Steps to reproduce:
  - Proposed fix:

**Phase 2:**
- [ ] Issue 2: [Description]

**Phase 3:**
- [ ] Issue 3: [Description]

[Continue for each phase]

### Unresolved Questions

1. **Parameter structure interface:**
   - How exactly are ParameterHandling structures built?
   - Where is flatten/unflatten handled?
   - What are the actual function names?

2. **Likelihood function:**
   - What is it called?
   - What arguments does it take?
   - Where is it defined?

3. **Fitting function:**
   - High-level fitting interface?
   - Return type?
   - Accessor functions for results?

4. **Simulation:**
   - Function name and signature?
   - How to specify new covariate data?
   - Output format?

5. **Splines:**
   - Is SplineHazard fully implemented?
   - How are knots specified?
   - Is this ready for testing?

---

## Success Criteria

### Phase 1-2: Basic Functionality
- [x] Create simple exponential model
- [x] Evaluate hazard functions correctly
- [ ] Compute likelihood
- [ ] Optimize parameters
- [ ] Results numerically sensible

### Phase 3: Covariates
- [ ] Model with covariates builds
- [ ] Name-based matching works
- [ ] Covariate effects estimated correctly

### Phase 4-5: Complex Models
- [ ] Multi-state models work
- [ ] Different hazard families work
- [ ] Mixed models converge

### Phase 6: Critical Test
- [ ] Different covariates per hazard works correctly
- [ ] No index-based bugs
- [ ] Name-based matching prevents errors

### Phase 7: Simulation
- [ ] Can simulate paths
- [ ] Covariate handling in simulation works
- [ ] Results realistic

### Overall
- [ ] All phases complete without errors
- [ ] Documentation matches implementation
- [ ] Infrastructure changes validated
- [ ] Ready to merge to main

---

## Next Steps After Walkthrough

1. **If all phases pass:**
   - Complete Phase 3 Task 3.8 (documentation)
   - Run full test suite
   - Update CHANGELOG
   - Create PR

2. **If issues found:**
   - Document each issue thoroughly
   - Prioritize fixes
   - Fix critical issues first
   - Re-run affected phases

3. **Documentation:**
   - Create user guide based on walkthrough
   - Add examples to docs
   - Update API reference

---

## Notes

**Important discoveries:**
- [To be filled during walkthrough]

**Things that surprised us:**
- [To be filled during walkthrough]

**Things to improve:**
- [To be filled during walkthrough]

**Performance observations:**
- [To be filled during walkthrough]

---

*This is a living document - update as we progress through the walkthrough*
