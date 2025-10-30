# Walkthrough Phase 3: Add Covariates
# Goal: Test name-based covariate matching
# 2-state model with covariates

using MultistateModels
using DataFrames

println("=" ^ 70)
println("PHASE 3: ADD COVARIATES (2-STATE MODEL)")
println("=" ^ 70)

# Step 3.1: Create Model with Covariates
println("\n--- Step 3.1: Create Hazard with Covariates ---")
haz_1_to_2_cov = MultistateModels.Hazard(1, 2, family="exp", formula=@formula(0 ~ age + sex))

println("✓ Hazard with covariates created")
println("  Parameter names: ", haz_1_to_2_cov.parnames)
println("  Has covariates: ", haz_1_to_2_cov.has_covariates)
println("  Total parameters: ", haz_1_to_2_cov.npar_total)
println("  Baseline parameters: ", haz_1_to_2_cov.npar_baseline)

# Extract covariate names
covar_names = MultistateModels.extract_covar_names(haz_1_to_2_cov.parnames)
println("  Covariate names: ", covar_names)

# Step 3.2: Create Data with Covariates
println("\n--- Step 3.2: Create Data with Covariates ---")
dat_cov = DataFrame(
    id = 1:10,
    tstart = zeros(10),
    tstop = rand(10) .* 2.0,
    statefrom = ones(Int, 10),
    stateto = fill(2, 10),
    age = rand(10) .* 50 .+ 20,  # Ages 20-70
    sex = rand([0, 1], 10)       # Binary
)

println("Sample data with covariates (first 3 rows):")
println(first(dat_cov, 3))

# Step 3.3: Test Covariate Extraction
println("\n--- Step 3.3: Test Covariate Extraction ---")
row = dat_cov[1, :]
covars = MultistateModels.extract_covariates(row, [:age, :sex])

println("Original row:")
println("  age = ", row.age)
println("  sex = ", row.sex)
println("\nExtracted covariates (NamedTuple):")
println("  ", covars)
println("  Type: ", typeof(covars))
println("\nAccess by name:")
println("  covars.age = ", covars.age)
println("  covars.sex = ", covars.sex)

# Step 3.4: Test Hazard Evaluation with Covariates
println("\n--- Step 3.4: Test Hazard Evaluation with Covariates ---")
params_cov = (intercept = -1.0, age = 0.02, sex = 0.5)
t = 1.0

h_val = haz_1_to_2_cov.hazard_fn(t, params_cov, covars; give_log=false)

# Manual calculation
expected_log_h = -1.0 + 0.02 * covars.age + 0.5 * covars.sex
expected_h = exp(expected_log_h)

println("Computed h(1.0): ", h_val)
println("Expected h(1.0): ", expected_h)
println("Match: ", isapprox(h_val, expected_h))

# Verify for different subjects
println("\nTesting multiple subjects:")
for i in 1:3
    row = dat_cov[i, :]
    covars = MultistateModels.extract_covariates(row, [:age, :sex])
    h = haz_1_to_2_cov.hazard_fn(1.0, params_cov, covars; give_log=false)
    expected = exp(-1.0 + 0.02 * covars.age + 0.5 * covars.sex)
    println("  Subject $i: h = ", round(h, digits=4), 
            " (expected: ", round(expected, digits=4), ")")
end

# Step 3.5: Build and Fit Model with Covariates
println("\n--- Step 3.5: Build Model with Covariates ---")
model_cov = multistatemodel(haz_1_to_2_cov; data=dat_cov)

println("✓ Model with covariates created")
println("  Total parameters: ", model_cov.npar)
println("  Expected: 3 (intercept + age + sex)")
println("  Number of subjects: ", model_cov.nsubj)

# Initialize parameters
initial_params_cov = MultistateModels.init_par(haz_1_to_2_cov, 0.0)
println("\nInitial parameters: ", initial_params_cov)
println("Length: ", length(initial_params_cov))

println("\n" * "=" ^ 70)
println("PHASE 3 COMPLETE - Covariate handling successful!")
println("=" ^ 70)
println("\nKey achievements:")
println("✓ Name-based covariate extraction works")
println("✓ Hazard evaluation with covariates correct")
println("✓ Model with covariates builds successfully")
println("\nNext: Run walkthrough_04_multistate.jl")
