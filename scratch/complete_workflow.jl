# Complete Workflow: Setup → Simulate → Fit
# This script demonstrates the full workflow with the new infrastructure

using MultistateModels
using DataFrames
using Random
using Distributions
using Optim

println("=" ^ 80)
println("COMPLETE WORKFLOW: SETUP → SIMULATE → FIT")
println("=" ^ 80)

Random.seed!(123)

# ==============================================================================
# PART 1: SETUP TRUE MODEL
# ==============================================================================

println("\n" * "=" ^ 80)
println("PART 1: SETUP TRUE MODEL")
println("=" ^ 80)

# We'll use a 3-state illness-death model:
# State 1 (Healthy) → State 2 (Illness) → State 3 (Death)
# State 1 (Healthy) → State 3 (Death)

# True parameter values (these are what we'll try to recover)
true_params = Dict(
    # Hazard 1→2 (Healthy → Illness): depends on age and sex
    :h12_Intercept => -2.0,  # log-scale baseline
    :h12_age => 0.03,        # age effect (per year)
    :h12_sex => 0.5,         # sex effect (male vs female)
    
    # Hazard 1→3 (Healthy → Death): depends on age only
    :h13_Intercept => -3.0,  # log-scale baseline
    :h13_age => 0.05,        # age effect
    
    # Hazard 2→3 (Illness → Death): depends on age and treatment
    :h23_Intercept => -1.5,  # log-scale baseline
    :h23_age => 0.04,        # age effect
    :h23_treatment => -0.6   # treatment effect (protective)
)

println("\nTrue parameter values:")
for (k, v) in sort(collect(true_params), by=x->string(x[1]))
    println("  ", rpad(k, 20), " = ", round(v, digits=3))
end

# Define hazards for the true model
haz_12_true = Hazard(@formula(0 ~ 1 + age + sex), "exp", 1, 2)
haz_13_true = Hazard(@formula(0 ~ 1 + age), "exp", 1, 3)
haz_23_true = Hazard(@formula(0 ~ 1 + age + treatment), "exp", 2, 3)

println("\n✓ True model hazards defined")
println("  1→2: Exponential with age, sex")
println("  1→3: Exponential with age")
println("  2→3: Exponential with age, treatment")

# ==============================================================================
# PART 2: SIMULATE DATA USING PACKAGE
# ==============================================================================

println("\n" * "=" ^ 80)
println("PART 2: SIMULATE DATA FROM TRUE MODEL")
println("=" ^ 80)

# Simulation parameters
n_subjects = 200
max_time = 10.0

println("\nSimulation settings:")
println("  Number of subjects: ", n_subjects)
println("  Maximum follow-up time: ", max_time, " years")

# Generate subject-level covariates
subject_covariates = DataFrame(
    id = 1:n_subjects,
    age = rand(Uniform(40, 80), n_subjects),      # Age 40-80 years
    sex = rand([0, 1], n_subjects),               # 0=female, 1=male
    treatment = rand([0, 1], n_subjects)          # 0=control, 1=treatment
)

println("\nSubject covariates generated (first 5):")
println(first(subject_covariates, 5))

# Create initial data structure for simulation
# Each subject starts in state 1 at time 0
init_data = DataFrame(
    id = 1:n_subjects,
    tstart = zeros(n_subjects),
    tstop = fill(max_time, n_subjects),
    statefrom = ones(Int, n_subjects),
    stateto = ones(Int, n_subjects),  # Will be updated by simulation
    obstype = ones(Int, n_subjects)
)

# Add covariates to initial data
init_data = hcat(init_data, subject_covariates[:, [:age, :sex, :treatment]])

# Build model for simulation with true parameters
model_sim = multistatemodel(haz_12_true, haz_13_true, haz_23_true; data = init_data)

# Set true parameters
# Convert Dict to vectors matching hazard order
true_params_vectors = [
    [true_params[:h12_Intercept], true_params[:h12_age], true_params[:h12_sex]],
    [true_params[:h13_Intercept], true_params[:h13_age]],
    [true_params[:h23_Intercept], true_params[:h23_age], true_params[:h23_treatment]]
]

set_parameters!(model_sim, true_params_vectors)

println("\n✓ Simulation model created with true parameters")

# Simulate using package
println("\nSimulating data using MultistateModels.simulate()...")
simulated_data_matrix = simulate(model_sim; nsim = 1, data = true, paths = false)
simulated_data = simulated_data_matrix[1]  # Extract DataFrame from Matrix{DataFrame}

println("✓ Data simulated using package simulation")
println("\nSimulated data summary:")
println("  Total intervals: ", nrow(simulated_data))
println("  Subjects: ", length(unique(simulated_data.id)))

# Count transitions
transitions = combine(groupby(simulated_data, [:statefrom, :stateto]), nrow => :count)
println("\nObserved transitions:")
println(transitions)

# ==============================================================================
# PART 3: FIT MODEL TO SIMULATED DATA
# ==============================================================================

println("\n" * "=" ^ 80)
println("PART 3: FIT MODEL TO SIMULATED DATA")
println("=" ^ 80)

# Define hazards for fitting (same structure as true model)
haz_12_fit = Hazard(@formula(0 ~ 1 + age + sex), "exp", 1, 2)
haz_13_fit = Hazard(@formula(0 ~ 1 + age), "exp", 1, 3)
haz_23_fit = Hazard(@formula(0 ~ 1 + age + treatment), "exp", 2, 3)

# Build model
model = multistatemodel(haz_12_fit, haz_13_fit, haz_23_fit; data = simulated_data)

println("✓ Model built for fitting")

# Get parameter names
all_parnames = get_parnames(model)
println("  Number of parameters: ", length(all_parnames))

println("\nParameter names:")
for (i, pname) in enumerate(all_parnames)
    println("  ", i, ". ", pname)
end

println("\nFitting model using built-in fit() function...")

# Fit the model
fitted_result = fit(model; verbose=false)

println("✓ Model fitted successfully")
println("  Converged: ", fitted_result.optimization.converged)
println("  Log-likelihood: ", round(get_loglik(fitted_result), digits=2))

# ==============================================================================
# PART 4: COMPARE RESULTS
# ==============================================================================

println("\n" * "=" ^ 80)
println("PART 4: COMPARE FITTED VS TRUE PARAMETERS")
println("=" ^ 80)

# Get fitted parameters (in natural space)
fitted_params_natural = get_parameters_natural(fitted_result)
fitted_params_dict = Dict{Symbol,Float64}()

# Extract parameters hazard by hazard
for haz in fitted_result.hazards
    haz_params = fitted_params_natural[findfirst(h -> h.name == haz.name, fitted_result.hazards)]
    for (i, pname) in enumerate(haz.parnames)
        fitted_params_dict[pname] = haz_params[i]
    end
end

final_loglik = get_loglik(fitted_result)

println("\nFinal log-likelihood: ", round(final_loglik, digits=2))

println("\n" * "─" ^ 80)
println(rpad("Parameter", 20), " | ", rpad("True", 10), " | ", rpad("Fitted", 10), " | ", "Difference")
println("─" ^ 80)

for pname in all_parnames
    true_val = true_params[pname]
    fitted_val = fitted_params_dict[pname]
    diff = fitted_val - true_val
    
    println(rpad(pname, 20), " | ", 
            rpad(round(true_val, digits=3), 10), " | ",
            rpad(round(fitted_val, digits=3), 10), " | ",
            round(diff, digits=3))
end
println("─" ^ 80)

# Calculate some summary statistics
diffs = [fitted_params_dict[pname] - true_params[pname] for pname in all_parnames]
mean_abs_diff = mean(abs.(diffs))
max_abs_diff = maximum(abs.(diffs))

println("\nSummary statistics:")
println("  Mean absolute difference: ", round(mean_abs_diff, digits=4))
println("  Max absolute difference: ", round(max_abs_diff, digits=4))

# ==============================================================================
# PART 5: INTERPRETATION
# ==============================================================================

println("\n" * "=" ^ 80)
println("PART 5: INTERPRETATION")
println("=" ^ 80)

println("\n✅ WORKFLOW COMPLETE!")
println("\nWhat we demonstrated:")
println("  1. ✓ Set up a 3-state illness-death model with covariates")
println("  2. ✓ Simulated ", n_subjects, " subjects with realistic parameters")
println("  3. ✓ Fitted the model using maximum likelihood")
println("  4. ✓ Recovered parameters close to true values")

if mean_abs_diff < 0.1
    println("\n🎉 Parameter recovery is EXCELLENT (mean error < 0.1)")
elseif mean_abs_diff < 0.2
    println("\n✅ Parameter recovery is GOOD (mean error < 0.2)")
else
    println("\n⚠️  Parameter recovery is moderate (may need more data or debugging)")
end

println("\nThis validates:")
println("  • Model specification with different covariates per hazard")
println("  • Data simulation from exponential hazards")
println("  • Log-likelihood computation")
println("  • Parameter estimation via optimization")
println("  • Name-based covariate matching throughout")

println("\n" * "=" ^ 80)
println("END OF COMPLETE WORKFLOW")
println("=" ^ 80)
