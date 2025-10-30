# Walkthrough Phase 7: Simulation
# Goal: Test path simulation with new infrastructure
# Verify survprob works with subjdat parameter

using MultistateModels
using DataFrames
using Random

println("=" ^ 70)
println("PHASE 7: SIMULATION")
println("=" ^ 70)

Random.seed!(123)

# Step 7.1: Simple Simulation Setup
println("\n--- Step 7.1: Setup Simple Model for Simulation ---")

# Create simple 2-state model
haz = MultistateModels.Hazard(1, 2, family="exp")
dat = DataFrame(
    id = 1:10,
    tstart = zeros(10),
    tstop = rand(10) .* 2.0,
    statefrom = ones(Int, 10),
    stateto = fill(2, 10)
)
model = multistatemodel(haz; data=dat)

println("✓ Simple model created for simulation")
println("  States: 1 → 2")
println("  Hazard family: exponential")

# Step 7.2: Test survprob Function
println("\n--- Step 7.2: Test survprob Function ---")
println("Testing survival probability calculation...")
println("(This function now requires subjdat parameter)")

# survprob signature should be:
# survprob(lb, ub, params, rowind, totalhaz, hazards, subjdat)

println("\nNote: survprob is typically called internally during simulation")
println("We updated it to use subjdat parameter in Phase 5 of infrastructure changes")

# Step 7.3: Explore Simulation Functions
println("\n--- Step 7.3: Explore Simulation Functions ---")
println("Looking for simulation functions in package...")

# Possible function names:
# - simulate(model, ...)
# - simulate_paths(model, ...)
# - sample_paths(model, ...)

println("\nExpected signature:")
println("  simulate(model; n=100, params=..., tmax=...)")
println("  OR")
println("  simulate(model, params; n=100, tmax=...)")

# Step 7.4: Test with Covariates (Template)
println("\n--- Step 7.4: Simulation with Covariates (Template) ---")

haz_cov = MultistateModels.Hazard(1, 2, family="exp", formula=@formula(0 ~ age + sex))
dat_cov = DataFrame(
    id = 1:10,
    tstart = zeros(10),
    tstop = rand(10) .* 2.0,
    statefrom = ones(Int, 10),
    stateto = fill(2, 10),
    age = rand(10) .* 50 .+ 20,
    sex = rand([0, 1], 10)
)
model_cov = multistatemodel(haz_cov; data=dat_cov)

println("✓ Model with covariates created")
println("  For simulation testing")

println("\nTo test simulation with covariates:")
println("""
# Create new subject data
new_subjects = DataFrame(
    id = 1:50,
    age = rand(50) .* 50 .+ 20,
    sex = rand([0, 1], 50)
)

# Simulate (once we know the function signature)
# simulated_data = simulate(model_cov, params; newdata=new_subjects, ...)
""")

# Step 7.5: Check for Sample Path Functions
println("\n--- Step 7.5: Check Internal Simulation Functions ---")
println("The simulation.jl file should contain:")
println("  - simulate_path() - single subject path simulation")
println("  - Uses survprob() with subjdat parameter")
println("  - Uses name-based covariate matching")

println("\n" * "=" ^ 70)
println("PHASE 7 EXPLORATION COMPLETE")
println("=" ^ 70)
println("\nNEXT STEPS:")
println("1. Identify simulation function interface")
println("2. Test simple simulation (no covariates)")
println("3. Test simulation with covariates")
println("4. Verify subjdat parameter is used correctly")
println("\nAfter identifying functions, we can complete simulation testing")
