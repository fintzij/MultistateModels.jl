# Walkthrough Phase 1: Basic Model Creation
# Goal: Verify basic hazard creation and model construction work
# Simple 2-state model: Healthy (1) → Diseased (2)
# Exponential hazard, no covariates

using MultistateModels
using DataFrames

println("=" ^ 70)
println("PHASE 1: BASIC MODEL CREATION (NO COVARIATES)")
println("=" ^ 70)

# Step 1.1: Create Simple 2-State Model
println("\n--- Step 1.1: Create Simple Hazard ---")
haz_1_to_2 = MultistateModels.Hazard(1, 2, family="exp")

println("✓ Hazard created successfully")
println("  Type: ", typeof(haz_1_to_2))
println("  Family: ", haz_1_to_2.family)
println("  Has covariates: ", haz_1_to_2.has_covariates)
println("  Parameter names: ", haz_1_to_2.parnames)
println("  Baseline params: ", haz_1_to_2.npar_baseline)
println("  Total params: ", haz_1_to_2.npar_total)

# Expected: MarkovHazard, family="exp", has_covariates=false, 1 parameter

# Step 1.2: Test Hazard Functions
println("\n--- Step 1.2: Test Hazard Function Evaluation ---")
params = (intercept = -1.0,)  # Named parameters
t = 1.0
covars = NamedTuple()  # Empty for no covariates

# Call hazard function
h_val = haz_1_to_2.hazard_fn(t, params, covars; give_log=false)
log_h_val = haz_1_to_2.hazard_fn(t, params, covars; give_log=true)

println("h(1.0) = ", h_val)
println("log h(1.0) = ", log_h_val)
println("Expected: h ≈ ", exp(-1.0), " (exp(-1))")
println("Match: ", isapprox(h_val, exp(-1.0)))

# Test cumulative hazard
cumhaz_val = haz_1_to_2.cumhaz_fn(0.0, 1.0, params, covars)
println("\nCumulative hazard [0, 1] = ", cumhaz_val)
println("Expected: ", 1.0 * exp(-1.0))
println("Match: ", isapprox(cumhaz_val, 1.0 * exp(-1.0)))

# Step 1.3: Initialize Parameters
println("\n--- Step 1.3: Initialize Parameters ---")
crude_rate = 0.0
initial_params = MultistateModels.init_par(haz_1_to_2, crude_rate)

println("Initial parameters: ", initial_params)
println("Expected: [0.0] (crude log rate)")
println("Length correct: ", length(initial_params) == 1)

# Step 1.4: Create Minimal Dataset
println("\n--- Step 1.4: Create Minimal Dataset ---")
dat = DataFrame(
    id = 1:10,
    tstart = zeros(10),
    tstop = rand(10) .* 2.0,  # Random event times 0-2
    statefrom = ones(Int, 10),
    stateto = fill(2, 10)
)

println("Sample data (first 3 rows):")
println(first(dat, 3))

# Step 1.5: Build Model Object
println("\n--- Step 1.5: Build Model Object ---")
model = multistatemodel(haz_1_to_2; data=dat)

println("✓ Model created successfully")
println("  Model type: ", typeof(model))
println("  Number of hazards: ", length(model.hazards))
println("  Number of subjects: ", model.nsubj)
println("  Total parameters: ", model.npar)

# Expected: 1 hazard, 10 subjects, 1 parameter

println("\n" * "=" ^ 70)
println("PHASE 1 COMPLETE - Basic model creation successful!")
println("=" ^ 70)
println("\nNext: Run walkthrough_02_model_fitting.jl")
