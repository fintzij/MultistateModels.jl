# Critical Bugs Found During Infrastructure Validation

**Date:** November 7, 2025  
**Branch:** infrastructure_changes  
**Context:** Creating complete workflow (setup → simulate → fit)

## Bug 1: Parameter Names Include Data Values (CRITICAL)

**Severity:** ⚠️  **CRITICAL** - Breaks fitting with covariates

**Location:** `src/modelgeneration.jl` lines ~160-166

**Problem:**
When creating a model with covariates, the parameter names include the actual data values instead of just the covariate names.

**Example:**
```julia
# Expected parameter names:
[:h12_Intercept, :h12_age, :h12_sex]

# Actual parameter names:
[:h12_age: 40.65182737072878, :h12_age: 40.916380207305885, ..., :h12_sex: 1]
```

**Impact:**
- Makes it impossible to map parameter estimates back to parameters
- Breaks any code that tries to look up parameters by name
- Creates hundreds/thousands of "parameters" (one per data row!)

**Root Cause:**
```julia
# Line ~165 in modelgeneration.jl
rhs_names = coefnames(hazschema)[2]
```

The `hazschema` is created by applying the schema to the data:
```julia
# Line ~140
hazschema = apply_schema(hazards[h].hazard, StatsModels.schema(hazards[h].hazard, data))
```

This causes `coefnames()` to return the actual data values, not just the column names.

**Proposed Fix:**
Extract covariate names from the formula BEFORE applying to data:
```julia
# Get formula terms without data
formula_terms = StatsModels.termnames(hazards[h].hazard.rhs)
rhs_names_vec = [String(t) for t in formula_terms if t != Symbol("1")]

# Then create parameter names
parnames = replace.(hazname * "_" .* rhs_names_vec, "(Intercept)" => "Intercept")
```

**Files to Check:**
- `src/modelgeneration.jl` - lines ~140-170 (Markov exponential)
- `src/modelgeneration.jl` - lines ~190-210 (Markov Weibull)  
- `src/modelgeneration.jl` - lines ~230-250 (Semi-Markov)

---

## Bug 2: `loglik` Function Not Exported

**Severity:** 🟡 **MODERATE** - Workaround exists

**Location:** `src/MultistateModels.jl`

**Problem:**
The `loglik_exact`, `loglik_markov`, etc. functions are not exported, making it hard for users to compute log-likelihood for custom optimization.

**Impact:**
- Users can't easily write custom fitting procedures
- Need to use `MultistateModels.loglik_exact()` with full namespace
- Not clear which `loglik_*` function to use for which model type

**Example Error:**
```julia
ll = loglik(params, model.totalhazards, model)
# ERROR: UndefVarError: loglik
```

**Workaround:**
```julia
ll = MultistateModels.loglik_exact(params, model.totalhazards, model)
```

**Proposed Fix:**
1. Export appropriate `loglik_*` functions
2. OR create a generic `loglik()` dispatcher that calls the right one based on model type
3. Add to exports in `src/MultistateModels.jl`

---

## Bug 4: Simulation Requires Multi-Row Subject Data

**Severity:** 🟡 **MODERATE** - Limits simulation use cases

**Location:** `src/simulation.jl`

**Problem:**
The `simulate()` function assumes each subject has multiple rows of data (panel data format), but fails when subjects have only a single row (which is natural for simulation from scratch).

**Example Error:**
```julia
# Create simple single-row-per-subject data
init_data = DataFrame(
    id = 1:100,
    tstart = zeros(100),
    tstop = fill(10.0, 100),
    statefrom = ones(Int, 100),
    stateto = ones(Int, 100),
    obstype = ones(Int, 100)
)

model = multistatemodel(haz; data = init_data)
simulate(model; nsim = 1)  # ERROR: BoundsError: attempt to access 1×6 SubDataFrame at index [2, :]
```

**Root Cause:**
In `src/simulation.jl`, the `simulate_path` function increments `rowind` expecting multiple rows per subject:
```julia
# Line ~176
row  += 1
ind  += 1
```

But with single-row data, there is no row 2.

**Impact:**
- Cannot simulate "from scratch" with simple initial states
- Must provide artificial panel data structure even for simulation
- Confusing user experience

**Workaround:**
Create multi-row data even though it's redundant for simulation purposes.

**Proposed Fix:**
Modify `simulate_path` to handle single-row subjects by treating end of available data as automatic censoring.

---

## Bug 5: Interaction Terms Not Supported

**Severity:** 🟡 **MODERATE** - Workaround exists (don't use interactions)

**Location:** `src/hazards.jl`, `src/modelgeneration.jl`

**Problem:**
Formula interactions like `trt*age` create parameter names like `"trt & age"` which cannot be matched to data columns.

**Example Error:**
```julia
h13 = Hazard(@formula(0 ~ 1 + trt*age), "exp", 1, 3)
# Later: ArgumentError: column name :trt & age not found in the data frame
```

**Root Cause:**
StatsModels creates interaction terms with names like `"trt & age"`, but `extract_covariates()` tries to find a column with that exact name instead of computing the interaction.

**Impact:**
- Interactions must be pre-computed and added as columns
- Cannot use natural formula syntax for interactions
- Inconsistent with typical statistical modeling packages

**Workaround:**
Pre-compute interactions:
```julia
data.trt_age = data.trt .* data.age
h13 = Hazard(@formula(0 ~ 1 + trt + age + trt_age), "exp", 1, 3)
```

**Proposed Fix:**
1. Detect interaction terms in parameter names
2. Compute interactions from base columns
3. OR use StatsModels' model matrix functionality

---

## Testing Status

**What Works:**
- ✅ Intercept-only models can be created
- ✅ Model structure is correct (hazards, parameters, etc.)
- ✅ All 9 walkthrough scripts pass (but they don't do fitting or simulation)

**What Doesn't Work:**
- ❌ Cannot fit models with covariates (Bug #1 - parameter naming)
- ❌ Cannot easily compute log-likelihood for optimization (Bug #2 - not exported)
- ❌ Unclear how to set up custom fitting (Bug #3 - documentation)
- ❌ Cannot simulate from single-row data (Bug #4 - simulation assumes panels)
- ❌ Cannot use interaction terms (Bug #5 - formula interactions)

**Test Scripts:**
- `complete_workflow.jl` - FAILS due to Bug #1 (parameter naming)
- `simple_workflow.jl` - INCOMPLETE due to Bug #2 (loglik not exported)
- `test/test_simulation_validation.jl` - FAILS due to Bug #4 (simulation data structure)
- `test/runtests.jl` - FAILS due to Bug #5 (interaction terms in setup files)

---

## Priority

1. **Fix Bug #1 IMMEDIATELY** - Blocks all covariate model fitting
2. **Fix Bug #4** - Blocks simulation testing and use
3. Export/document likelihood functions (Bugs #2, #3)
4. Fix interaction term support (Bug #5) - or document as unsupported

---

## Next Steps

1. Fix parameter naming in `src/modelgeneration.jl` (Bug #1)
2. Fix simulation to handle single-row data (Bug #4)
3. Export or create unified `loglik()` function
4. Document likelihood function interface
5. Fix or document interaction term limitations
6. Re-test all workflows once fixes are in place
