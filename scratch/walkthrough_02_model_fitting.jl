# Walkthrough Phase 2: Model Fitting
# Goal: Verify likelihood computation and optimization work
# Using simple 2-state exponential model from Phase 1

using MultistateModels
using DataFrames
using Optim

println("=" ^ 70)
println("PHASE 2: MODEL FITTING (NO COVARIATES)")
println("=" ^ 70)

# Recreate model from Phase 1
println("\n--- Setup: Recreate Basic Model ---")
haz_1_to_2 = MultistateModels.Hazard(1, 2, family="exp")
dat = DataFrame(
    id = 1:10,
    tstart = zeros(10),
    tstop = rand(10) .* 2.0,
    statefrom = ones(Int, 10),
    stateto = fill(2, 10)
)
model = multistatemodel(haz_1_to_2; data=dat)
println("✓ Model recreated")

# Step 2.1: Test Likelihood Evaluation
println("\n--- Step 2.1: Test Likelihood Evaluation ---")
println("Exploring likelihood interface...")

# Initialize parameters
crude_params = MultistateModels.init_par(haz_1_to_2, log(1.0))
println("Testing at parameters: ", crude_params)

# Try to find and call likelihood function
# NOTE: This will help us discover the actual interface
println("\nAttempting to evaluate likelihood...")
println("(This step will reveal the actual likelihood function signature)")

# Possibilities to test:
# 1. loglik(model, params)
# 2. loglikelihood(model, params) 
# 3. compute_loglik(model, params)
# 4. model.loglik(params)

# Let's check what methods are available
println("\nModel fields:")
for field in fieldnames(typeof(model))
    println("  - ", field)
end

# Step 2.2: Test Parameter Handling Structure
println("\n--- Step 2.2: Explore Parameter Handling ---")
println("Model has ", model.npar, " total parameters")

# Check if there's a parameter structure builder
println("\nLooking for parameter structure functions...")

# Expected workflow:
# params = build_parameter_structure(model) or similar
# flat, unflatten = ParameterHandling.value_flatten(params)

# Step 2.3: Manual Optimization (Template)
println("\n--- Step 2.3: Manual Optimization Template ---")
println("Once we identify the likelihood function, we can optimize like:")
println("""
using Optim

function objective(flat_params)
    params = unflatten(flat_params)
    -loglik(model, params)  # Negative for minimization
end

initial_flat = MultistateModels.init_par(haz_1_to_2, 0.0)  # Or flattened version?
result = optimize(objective, initial_flat, BFGS())

println("Converged: ", Optim.converged(result))
println("Parameters: ", result.minimizer)
println("Log-likelihood: ", -result.minimum)
""")

# Step 2.4: Look for Built-in Fitting Function
println("\n--- Step 2.4: Look for High-Level Fitting Function ---")
println("Checking for fitting functions...")
println("Possible names: fit, fit_model, estimate, mle, etc.")

println("\n" * "=" ^ 70)
println("PHASE 2 EXPLORATION COMPLETE")
println("=" ^ 70)
println("\nNEXT STEPS:")
println("1. Identify actual likelihood function name")
println("2. Identify parameter structure interface")
println("3. Test optimization workflow")
println("\nThen run: walkthrough_03_with_covariates.jl")
