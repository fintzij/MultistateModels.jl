# MultistateModels.jl Infrastructure Testing Walkthrough
**Date:** October 30, 2025 (Updated: November 7, 2025)  
**Branch:** infrastructure_changes  
**Purpose:** Systematically validate all new infrastructure through complete workflow

---

## Critical Understanding: Two-Stage Type System

**Stage 1: User-Facing Specification (Input)**
- `Hazard()` constructor returns `ParametricHazard` or `SplineHazard`
- These are **specification objects only**
- Fields: `hazard` (formula), `family`, `statefrom`, `stateto`
- Do NOT have: `hazard_fn`, `cumhaz_fn`, `parnames`, `npar_total`, etc.

**Stage 2: Internal Execution Types (After Model Building)**
- `multistatemodel()` calls `build_hazards()` which creates:
  - `MarkovHazard` (for exponential family)
  - `SemiMarkovHazard` (for Weibull, Gompertz families)
  - Internal `SplineHazard` (for spline hazards)
- These internal types have: `hazname`, `parnames`, `npar_total`, `hazard_fn`, `cumhaz_fn`, etc.
- Access via: `model.hazards[i]` after building the model

**Parameter Naming Convention:**
- Parameters are prefixed with hazard identifier: `h{from}{to}_`
- Example for hazard 1→2: `h12_Intercept`, `h12_age`, `h12_sex`
- Example for hazard 1→3: `h13_Intercept`, `h13_age`, `h13_sex`
- This ensures **uniqueness** across all hazards in multi-state models

**Parameter Passing to Hazard Functions:**
- Hazard functions (`hazard_fn`, `cumhaz_fn`) expect parameters as **Vector**, not NamedTuple
- Example: `params = [-1.0, 0.02, 0.5]` for [log_baseline, coef1, coef2]
- Baseline parameters (intercept, shape, scale) are on **LOG SCALE**
- Covariate coefficients are on **NATURAL SCALE**
- Functions return hazard values on **NATURAL SCALE** (not log)
- Covariates are passed as NamedTuple: `covars = (age=45, sex=1)`

**Testing Implications:**
- Create hazard specification with `Hazard()`
- Build model with `multistatemodel()`
- Access internal hazard via `model.hazards[1]`
- Then test fields and functions on internal hazard

---

## Important: Hazard Constructor Syntax

**Correct syntax:**
```julia
# Intercept-only (no covariates)
haz = Hazard(@formula(0 ~ 1), "exp", 1, 2)

# With covariates
haz = Hazard(@formula(0 ~ age + sex), "exp", 1, 2)
```

**Key points:**

- Formula comes FIRST: `@formula(0 ~ ...)` 
- Family is SECOND: `"exp"`, `"wei"`, `"gom"`, or `"sp"`
- Then state transitions: `from::Int`, `to::Int`
- Use `@formula(0 ~ 1)` for intercept-only (no covariates)

---

## Required Data Format

**All datasets must have these first 6 columns in order:**

1. `id::Int` - Subject identifier (must be 1, 2, 3, ...)
2. `tstart::Float64` - Start time of interval
3. `tstop::Float64` - Stop time of interval
4. `statefrom::Int` - State at start (can be `missing` for first observation)
5. `stateto::Int` - State at end (0 for censored, `missing` for interval censored)
6. `obstype::Int` - Observation type:
   - `1` = Exact observation (transition observed at exact time)
   - `2` = Interval censored (transition occurred between tstart and tstop)
   - `>2` = Custom censoring pattern ID

**Additional columns** (after the first 6) can contain covariates with names matching the formula.

**Example:**
```julia
dat = DataFrame(
    id = [1, 2, 3],
    tstart = [0.0, 0.0, 0.0],
    tstop = [1.5, 2.3, 1.8],
    statefrom = [1, 1, 1],
    stateto = [2, 2, 2],
    obstype = [1, 1, 1],  # All exact observations
    age = [45, 52, 38],   # Covariate columns after first 6
    sex = [0, 1, 0]
)
```

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

**Important Note on Type System:**
- `Hazard()` creates a **ParametricHazard** (user-facing specification)
- Only has fields: `hazard`, `family`, `statefrom`, `stateto`
- `multistatemodel()` converts to internal types (MarkovHazard, SemiMarkovHazard, etc.)
- Internal types have: `hazname`, `parnames`, `npar_total`, `hazard_fn`, `cumhaz_fn`, etc.
- Access internal hazards via `model.hazards[i]` after building the model

### Step 1.1: Create Simple Hazard Specification
**Goal:** Verify basic hazard specification creation works

**Test case:**
```julia
using MultistateModels
using DataFrames

# Simple 2-state model: Healthy (1) → Diseased (2)
# Exponential hazard, no covariates

# Create hazard specification
haz_1_to_2 = Hazard(@formula(0 ~ 1), "exp", 1, 2)

# Inspect ParametricHazard object (input specification)
println("Hazard type: ", typeof(haz_1_to_2))
println("Hazard family: ", haz_1_to_2.family)
println("From state: ", haz_1_to_2.statefrom)
println("To state: ", haz_1_to_2.stateto)
```

**Expected output:**
- Type: `ParametricHazard` (not MarkovHazard yet!)
- Family: `"exp"`
- From state: `1`
- To state: `2`

**Validation:**
- [ ] Hazard specification created without error
- [ ] Correct type returned (ParametricHazard)
- [ ] Fields populated correctly
- [ ] Formula stored in `hazard` field

### Step 1.2: Build Model to Access Internal Hazard
**Goal:** Verify model construction creates internal hazard types

**Test case:**
```julia
# Create minimal dataset first
# Note: Must include 'obstype' column (1 = exact observation, 2 = interval censored)
dat = DataFrame(
    id = 1:10,
    tstart = zeros(10),
    tstop = rand(10) .* 2.0,
    statefrom = ones(Int, 10),
    stateto = fill(2, 10),
    obstype = ones(Int, 10)  # 1 = exact observation
)

# Build model - this creates the internal hazard structures
model = multistatemodel(haz_1_to_2; data=dat)

println("Model type: ", typeof(model))
println("Number of hazards: ", length(model.hazards))

# Now we can access the internal hazard structure
internal_haz = model.hazards[1]
println("\nInternal hazard structure:")
println("  Type: ", typeof(internal_haz))
println("  Family: ", internal_haz.family)
println("  Has covariates: ", internal_haz.has_covariates)
println("  Parameter names: ", internal_haz.parnames)
println("  Baseline params: ", internal_haz.npar_baseline)
println("  Total params: ", internal_haz.npar_total)
```

**Expected output:**
- Model type: `MultistateModel`
- Internal hazard type: `MarkovHazard` (not ParametricHazard!)
- Family: `"exp"`
- Has covariates: `false`
- Parameter names: `["h12_Intercept"]` (with hazard prefix!)
- Baseline params: `1`
- Total params: `1`

**Validation:**
- [ ] Model object created without error
- [ ] Correct type for internal hazard (MarkovHazard)
- [ ] Fields populated correctly
- [ ] Parameter names include hazard prefix (h12_)
- [ ] hazard_fn and cumhaz_fn are callable functions

### Step 1.3: Test Hazard Functions
**Goal:** Verify runtime-generated functions work correctly

**Test case:**
```julia
### Step 1.3: Test Hazard Functions
**Goal:** Verify runtime-generated functions work correctly

**Test case:**
```julia
# Test hazard function evaluation
# Use the internal hazard from the model
# Parameters are passed as a VECTOR (not NamedTuple)
# Parameters are on LOG SCALE for baseline parameters
params = [-1.0]  # Log rate = -1.0, so hazard rate = exp(-1.0) ≈ 0.368
t = 1.0
covars = NamedTuple()  # Empty for no covariates

# Call hazard function (returns natural scale)
h_val = internal_haz.hazard_fn(t, params, covars)
log_h_val = log(h_val)  # Compute log manually

println("h(1.0) = ", h_val)
println("log h(1.0) = ", log_h_val)
println("Expected: h = exp(-1.0) = ", exp(-1.0))

# Test cumulative hazard
cumhaz_val = internal_haz.cumhaz_fn(0.0, 1.0, params, covars)
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
- [ ] Hazard functions return natural scale (not log)
- [ ] Cumulative hazard correct
- [ ] Parameters passed as Vector, not NamedTuple

### Step 1.4: Initialize Parameters
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

### Step 1.4: Initialize Parameters
**Goal:** Verify parameter initialization works

**Test case:**
```julia
# Test init_par on internal hazard
crude_rate = 0.0
initial_params = MultistateModels.init_par(internal_haz, crude_rate)

println("Initial parameters: ", initial_params)
println("Expected: [0.0] (crude log rate)")

# Model's initialized parameters
println("\nModel's initialized parameters:")
println("  ", model.parameters)
```

**Validation:**
- [ ] init_par returns vector
- [ ] Correct length (1 for exponential)
- [ ] Reasonable values
- [ ] Model parameters initialized

**Phase 1 Complete!**

**Key Findings:**
- ✓ Hazard() creates ParametricHazard specification
- ✓ multistatemodel() converts to internal MarkovHazard
- ✓ Internal hazard has runtime-generated functions
- ✓ Parameter names include hazard prefix (h12_Intercept)

**STOP HERE IF ANY ISSUES - Debug before proceeding**

---

## Phase 2: Model Fitting (No Covariates)

### Step 2.1: Prepare for Fitting
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

### Step 3.1: Create Hazard Specification with Covariates
**Goal:** Test formula-based covariate specification

**Test case:**
```julia
# Create hazard specification with covariates
haz_1_to_2_cov = Hazard(@formula(0 ~ age + sex), "exp", 1, 2)

println("Type: ", typeof(haz_1_to_2_cov))
println("Family: ", haz_1_to_2_cov.family)
println("Formula: ", haz_1_to_2_cov.hazard)
```

**Expected output:**
- Type: `ParametricHazard` (input specification)
- Family: `"exp"`
- Formula stored in hazard field

**Validation:**
- [ ] Hazard specification created with formula
- [ ] Still ParametricHazard type
- [ ] Formula preserved

### Step 3.2: Create Data with Covariates
**Goal:** Add covariate columns matching formula

**Test case:**
```julia
# Add covariates to data
dat_cov = DataFrame(
    id = 1:50,
    tstart = zeros(50),
    tstop = rand(50) .* 5.0,
    statefrom = ones(Int, 50),
    stateto = fill(2, 50),
    obstype = ones(Int, 50),     # 1 = exact observation
    age = rand(50) .* 50 .+ 20,  # Ages 20-70
    sex = rand([0, 1], 50)       # Binary
)

println("Data with covariates:")
println(first(dat_cov, 3))
```

**Validation:**
- [ ] Covariate columns added
- [ ] Column names match formula (age, sex)
- [ ] More subjects for stability

### Step 3.3: Build Model and Check Internal Hazard
**Goal:** Verify covariate processing during model building

**Test case:**
```julia
# Build model with covariates
model_cov = multistatemodel(haz_1_to_2_cov; data=dat_cov)

# Access internal hazard
internal_haz_cov = model_cov.hazards[1]

println("\nInternal hazard with covariates:")
println("  Type: ", typeof(internal_haz_cov))
println("  Has covariates: ", internal_haz_cov.has_covariates)
println("  Parameter names: ", internal_haz_cov.parnames)
println("  Baseline params: ", internal_haz_cov.npar_baseline)
println("  Total params: ", internal_haz_cov.npar_total)
```

**Expected output:**
- Internal type: `MarkovHazard`
- Has covariates: `true`
- Parameter names: `["h12_Intercept", "h12_age", "h12_sex"]` (with hazard prefix!)
- Baseline params: `1` (intercept only)
- Total params: `3` (intercept + 2 covariates)

**Validation:**
- [ ] Model created with covariate data
- [ ] Internal hazard has_covariates = true
- [ ] Parameter names include all terms with prefix
- [ ] Correct parameter counts

### Step 3.4: Test Covariate Extraction
**Goal:** Verify name-based matching works

**Test case:**
```julia
# Test helper function
row = dat_cov[1, :]
covar_names = [:age, :sex]
covars = MultistateModels.extract_covariates(row, covar_names)

println("Row: ", row)
println("Extracted covariates: ", covars)
println("Type: ", typeof(covars))

# Test accessing by name
println("Age: ", covars.age)
println("Sex: ", covars.sex)
```

**Validation:**
- [ ] Covariates extracted as NamedTuple
- [ ] Correct values from data row
- [ ] Can access by name
- [ ] Names match exactly

### Step 3.5: Test Hazard Evaluation with Covariates
**Goal:** Verify runtime functions handle covariates

**Test case:**
```julia
# Parameters for exponential with covariates (as VECTOR, not NamedTuple)
# Order: [log_baseline, coef_age, coef_sex]
params_cov = [-1.0, 0.02, 0.5]
t = 1.0
row = dat_cov[1, :]
covar_names = [:age, :sex]
covars = MultistateModels.extract_covariates(row, covar_names)

# Evaluate hazard using internal hazard
h_val = internal_haz_cov.hazard_fn(t, params_cov, covars)

println("Hazard with covariates: ", h_val)
println("Covariates used: age=", covars.age, ", sex=", covars.sex)

# Expected: exp(-1.0 + 0.02*age + 0.5*sex)
expected = exp(-1.0 + 0.02*covars.age + 0.5*covars.sex)
println("Expected: ", expected)
println("Match: ", isapprox(h_val, expected))
```

**Validation:**
- [ ] Hazard evaluates with covariates
- [ ] Correct value (matches manual calculation)
- [ ] Parameters passed as Vector (not NamedTuple)
- [ ] Covariates passed as NamedTuple for name-based matching
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

### Step 4.1: Create 3-State Illness-Death Hazard Specifications
**Goal:** Multiple transitions, no covariates

**Test case:**
```julia
# States: 1=Healthy, 2=Ill, 3=Dead
# Transitions: 1→2, 1→3, 2→3

haz_1_to_2 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
haz_1_to_3 = Hazard(@formula(0 ~ 1), "exp", 1, 3)
haz_2_to_3 = Hazard(@formula(0 ~ 1), "exp", 2, 3)

println("Created 3 hazard specifications")
println("Types: ", typeof(haz_1_to_2), ", ", typeof(haz_1_to_3), ", ", typeof(haz_2_to_3))
```

**Validation:**
- [ ] All hazard specifications created (ParametricHazard)
- [ ] Different transitions specified

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
    stateto =   [2,3, 2,3, 2,3, 2,3, 2,3,  3, 3, 3, 3, 3],
    obstype = ones(Int, 15)  # All exact observations
)

println("Multi-state data:")
println(dat_3state)
```

**Validation:**
- [ ] Data structure correct
- [ ] Multiple rows per subject
- [ ] All transition types represented (1→2, 1→3, 2→3)

### Step 4.3: Build 3-State Model and Inspect Internal Hazards
**Goal:** Model with multiple hazards, verify parameter naming

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

# Check internal hazards and parameter names
println("\nInternal hazard structures:")
for (i, haz) in enumerate(model_3state.hazards)
    println("Hazard $i: $(haz.from)→$(haz.to)")
    println("  Type: ", typeof(haz))
    println("  Name: ", haz.hazname)
    println("  Parameters: ", haz.parnames)
end
```

**Expected:**
- 3 hazards (all MarkovHazard)
- 3 parameters total (one per hazard)
- 10 subjects
- Parameter names: `["h12_Intercept", "h13_Intercept", "h23_Intercept"]` (all unique!)

**Validation:**
- [ ] Model created
- [ ] Correct hazard count (3)
- [ ] Correct parameter count (3)
- [ ] All transitions represented
- [ ] Parameter names are unique with hazard-specific prefixes

### Step 4.4: Verify Parameter Uniqueness
**Goal:** Confirm no name collisions

**Test case:**
```julia
# Extract all parameter names
all_params = model_3state.parameters

println("\nAll model parameters:")
for (name, val) in pairs(all_params)
    println("  $name = $val")
end

# Check uniqueness
param_names = keys(all_params) |> collect
println("\nParameter names unique: ", length(param_names) == length(unique(param_names)))
println("Total parameters: ", length(param_names))
```

**Validation:**
- [ ] All parameter names unique
- [ ] Names follow pattern: h{from}{to}_{term}
- [ ] No collisions between hazards
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
haz_wei = Hazard(@formula(0 ~ 1), "wei", 1, 2)

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
haz_gom = Hazard(@formula(0 ~ 1), "gom", 1, 2)

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
haz_1_to_2_exp = Hazard(@formula(0 ~ 1), "exp", 1, 2)
haz_1_to_3_wei = Hazard(@formula(0 ~ 1), "wei", 1, 3)
haz_2_to_3_gom = Hazard(@formula(0 ~ 1), "gom", 2, 3)

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

# Create data with multiple covariates
# Different covariates per hazard - THIS IS THE KEY TEST
haz_1_to_2_age_sex = Hazard(@formula(0 ~ age + sex), "exp", 1, 2)
haz_2_to_3_treatment = Hazard(@formula(0 ~ treatment), "exp", 2, 3)

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
# haz_spline = Hazard(@formula(0 ~ 1), "sp", 1, 2; degree=3, knots=[0.5, 1.0, 1.5, 2.0])

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
#   haz_1_2 = Hazard(@formula(0 ~ age + sex), "wei", 1, 2)
# - 1→3: Exponential with age
#   haz_1_3 = Hazard(@formula(0 ~ age), "exp", 1, 3)
# - 2→3: Gompertz with treatment
#   haz_2_3 = Hazard(@formula(0 ~ treatment), "gom", 2, 3)

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
