# Simulation Workflow - Step by Step
# This script walks through simulating data from a multistate model

using MultistateModels
using DataFrames
using Distributions
using Random

println("=" ^ 80)
println("MULTISTATE MODEL SIMULATION WORKFLOW")
println("=" ^ 80)

Random.seed!(456)

# ==============================================================================
# STEP 1: UNDERSTAND WHAT SIMULATION DOES
# ==============================================================================

println("\n" * "=" ^ 80)
println("STEP 1: WHAT IS SIMULATION?")
println("=" ^ 80)

println("""
Simulation generates synthetic data from a model with known parameters.

Why simulate?
  1. Test estimation procedures (can we recover true parameters?)
  2. Study model properties (transition probabilities, state occupancy)
  3. Power calculations (sample size planning)
  4. Validate code (does simulation match theory?)
  5. Generate data when real data unavailable

What you need:
  1. Model structure (hazards for each transition)
  2. True parameter values
  3. Covariate data for subjects
  4. Maximum follow-up time
""")

println("✓ Concept understood")

# ==============================================================================
# STEP 2: DEFINE THE TRUE MODEL
# ==============================================================================

println("\n" * "=" ^ 80)
println("STEP 2: DEFINE THE TRUE MODEL AND PARAMETERS")
println("=" ^ 80)

println("""
We'll simulate from a 3-state illness-death model:
  1 (Healthy) → 2 (Illness) → 3 (Death)
  1 (Healthy) → 3 (Death)
""")

# Define true parameter values
println("\nSetting true parameter values:")

true_params = Dict(
    # Hazard 1→2: log-baseline and effects of age, sex
    :h12_Intercept => -2.0,
    :h12_age => 0.03,
    :h12_sex => 0.5,
    
    # Hazard 1→3: log-baseline and effect of age
    :h13_Intercept => -3.0,
    :h13_age => 0.05,
    
    # Hazard 2→3: log-baseline and effects of age, treatment
    :h23_Intercept => -1.5,
    :h23_age => 0.04,
    :h23_treatment => -0.6  # Protective effect
)

println("  True parameters:")
for (k, v) in sort(collect(true_params), by=x->string(x[1]))
    println("    ", rpad(k, 20), " = ", round(v, digits=3))
end

# Create hazard specifications
println("\nCreating hazard specifications:")

haz_12 = Hazard(@formula(0 ~ 1 + age + sex), "exp", 1, 2)
haz_13 = Hazard(@formula(0 ~ 1 + age), "exp", 1, 3)
haz_23 = Hazard(@formula(0 ~ 1 + age + treatment), "exp", 2, 3)

println("  ✓ Hazards specified")

# ==============================================================================
# STEP 3: GENERATE COVARIATE DATA
# ==============================================================================

println("\n" * "=" ^ 80)
println("STEP 3: GENERATE SUBJECT COVARIATE DATA")
println("=" ^ 80)

println("""
For simulation, we need covariate values for each subject.
These can be:
  - Random (as in this example)
  - Based on real data
  - Specific scenarios you want to test
""")

n_subjects = 200
max_time = 10.0

println("\nGenerating covariates for ", n_subjects, " subjects:")

subject_data = DataFrame(
    id = 1:n_subjects,
    age = rand(Uniform(40, 80), n_subjects),  # Age uniformly 40-80
    sex = rand([0, 1], n_subjects),           # Binary: 0=female, 1=male
    treatment = rand([0, 1], n_subjects)      # Binary: 0=control, 1=treatment
)

println("  ✓ Covariates generated")
println("\nCovariate summary:")
println("  Age: range [", round(minimum(subject_data.age), digits=1), ", ",
        round(maximum(subject_data.age), digits=1), "]")
println("  Sex: ", sum(subject_data.sex), " males, ", 
        n_subjects - sum(subject_data.sex), " females")
println("  Treatment: ", sum(subject_data.treatment), " treated, ",
        n_subjects - sum(subject_data.treatment), " control")

println("\nFirst 5 subjects:")
println(first(subject_data, 5))

# ==============================================================================
# STEP 4: PREPARE INITIAL DATA STRUCTURE
# ==============================================================================

println("\n" * "=" ^ 80)
println("STEP 4: PREPARE INITIAL DATA STRUCTURE FOR SIMULATION")
println("=" ^ 80)

println("""
For simulation, we need an initial data structure that specifies:
  - Where each subject starts (usually state 1)
  - When they start (usually time 0)
  - Maximum follow-up time
  
The simulate() function will generate the transitions.
""")

println("\nCreating initial data:")

init_data = DataFrame(
    id = 1:n_subjects,
    tstart = zeros(n_subjects),              # All start at time 0
    tstop = fill(max_time, n_subjects),      # Observe until max_time
    statefrom = ones(Int, n_subjects),       # All start in state 1
    stateto = ones(Int, n_subjects),         # Will be updated by simulation
    obstype = ones(Int, n_subjects)          # Exact observations
)

# Add covariates
init_data = hcat(init_data, subject_data[:, [:age, :sex, :treatment]])

println("  ✓ Initial data structure created")
println("  All subjects start in state 1 at time 0")
println("  Maximum follow-up: ", max_time, " years")

println("\nInitial data structure (first 3 rows):")
println(first(init_data, 3))

# ==============================================================================
# STEP 5: BUILD MODEL FOR SIMULATION
# ==============================================================================

println("\n" * "=" ^ 80)
println("STEP 5: BUILD MODEL AND SET TRUE PARAMETERS")
println("=" ^ 80)

println("\nBuilding model...")
model = multistatemodel(haz_12, haz_13, haz_23; data = init_data)

println("  ✓ Model created")

# Convert true parameters to vector format matching hazard order
println("\nSetting true parameters:")

true_params_vectors = [
    [true_params[:h12_Intercept], true_params[:h12_age], true_params[:h12_sex]],
    [true_params[:h13_Intercept], true_params[:h13_age]],
    [true_params[:h23_Intercept], true_params[:h23_age], true_params[:h23_treatment]]
]

set_parameters!(model, true_params_vectors)

println("  ✓ Parameters set to true values")

# Verify parameters
println("\nModel parameters:")
for (i, haz) in enumerate(model.hazards)
    println("  Hazard ", haz.from, "→", haz.to, ":")
    for (j, pname) in enumerate(haz.parnames)
        println("    ", pname, " = ", model.parameters[i][j])
    end
end

# ==============================================================================
# STEP 6: SIMULATE DATA
# ==============================================================================

println("\n" * "=" ^ 80)
println("STEP 6: SIMULATE DATA FROM THE MODEL")
println("=" ^ 80)

println("""
The simulate() function will:
  1. For each subject, simulate their trajectory through states
  2. Use the hazard functions with true parameters
  3. Use subject-specific covariates
  4. Generate transition times based on exponential waiting times
  5. Stop at death (state 3) or censoring time (max_time)
  
Arguments:
  - nsim: number of simulation replications (use 1 for single dataset)
  - data: whether to return data (true) or just paths (false)
  - paths: whether to return continuous-time paths
""")

println("\nSimulating...")

sim_data_matrix = simulate(model; nsim = 1, data = true, paths = false)
sim_data = sim_data_matrix[1]  # Extract DataFrame from matrix

println("  ✓ Simulation complete!")

# ==============================================================================
# STEP 7: EXAMINE SIMULATED DATA
# ==============================================================================

println("\n" * "=" ^ 80)
println("STEP 7: EXAMINE THE SIMULATED DATA")
println("=" ^ 80)

println("\nSimulated data structure:")
println("  Total rows: ", nrow(sim_data))
println("  Unique subjects: ", length(unique(sim_data.id)))
println("  Columns: ", names(sim_data))

println("\nFirst 10 rows of simulated data:")
println(first(sim_data, 10))

# Count transitions
println("\nTransition counts:")
transitions = combine(groupby(sim_data, [:statefrom, :stateto]), nrow => :count)
println(transitions)

# Count final states
println("\nFinal state distribution:")
final_states = combine(groupby(sim_data, :id)) do subj_df
    DataFrame(final_state = subj_df[end, :stateto])
end

state_counts = combine(groupby(final_states, :final_state), nrow => :count)
println(state_counts)

# Event times
println("\nEvent time summaries:")

# Time to illness (1→2)
times_to_illness = [row.tstop for row in eachrow(sim_data) 
                    if row.statefrom == 1 && row.stateto == 2]
if length(times_to_illness) > 0
    println("  Time to illness (1→2):")
    println("    n = ", length(times_to_illness))
    println("    mean = ", round(mean(times_to_illness), digits=2))
    println("    median = ", round(median(times_to_illness), digits=2))
end

# Time to death from healthy (1→3)
times_to_death_direct = [row.tstop for row in eachrow(sim_data)
                         if row.statefrom == 1 && row.stateto == 3]
if length(times_to_death_direct) > 0
    println("  Direct death time (1→3):")
    println("    n = ", length(times_to_death_direct))
    println("    mean = ", round(mean(times_to_death_direct), digits=2))
    println("    median = ", round(median(times_to_death_direct), digits=2))
end

# Time to death from illness (2→3)
times_to_death_from_illness = [row.tstop for row in eachrow(sim_data)
                                if row.statefrom == 2 && row.stateto == 3]
if length(times_to_death_from_illness) > 0
    println("  Death from illness (2→3):")
    println("    n = ", length(times_to_death_from_illness))
    println("    mean = ", round(mean(times_to_death_from_illness), digits=2))
    println("    median = ", round(median(times_to_death_from_illness), digits=2))
end

# ==============================================================================
# STEP 8: SIMULATE MULTIPLE DATASETS
# ==============================================================================

println("\n" * "=" ^ 80)
println("STEP 8: SIMULATE MULTIPLE DATASETS (OPTIONAL)")
println("=" ^ 80)

println("""
For simulation studies, you often want multiple replications
to study variability in estimates.

Example: Generate 10 independent datasets
""")

println("\nSimulating 10 datasets...")

multi_sim = simulate(model; nsim = 10, data = true, paths = false)

println("  ✓ ", size(multi_sim, 2), " datasets simulated")
println("  Result is a Matrix with dimensions: ", size(multi_sim))
println("    - ", size(multi_sim, 1), " subjects")
println("    - ", size(multi_sim, 2), " replications")

println("\nEach replication is a separate DataFrame:")
println("  Replication 1 has ", nrow(multi_sim[1]), " rows")
println("  Replication 2 has ", nrow(multi_sim[2]), " rows")
println("  ...")

# Count events in each replication
println("\nNumber of 1→2 transitions per replication:")
for i in 1:size(multi_sim, 2)
    n_transitions = sum((multi_sim[i].statefrom .== 1) .& (multi_sim[i].stateto .== 2))
    print("  Rep ", i, ": ", n_transitions)
    if i % 5 == 0
        println()
    end
end
println()

# ==============================================================================
# STEP 9: USE SIMULATED DATA
# ==============================================================================

println("\n" * "=" ^ 80)
println("STEP 9: WHAT TO DO WITH SIMULATED DATA")
println("=" ^ 80)

println("""
Common uses for simulated data:

1. Test parameter recovery:
   - Fit model to simulated data
   - Compare estimated parameters to true values
   - Check if estimation is unbiased

2. Study model properties:
   - Calculate state occupancy probabilities
   - Estimate transition probabilities
   - Compute expected survival times

3. Validate simulation:
   - Compare to analytical results (if available)
   - Check that distributions make sense
   - Verify covariate effects work correctly

4. Power calculations:
   - Simulate many datasets
   - Fit models to each
   - Calculate power to detect effects

Example fitting to simulated data:
""")

println("\nFitting model to simulated data...")
fitted_model = fit(model; verbose=false)

println("  ✓ Model fitted")
println("\nComparing estimated to true parameters:")
println(rpad("Parameter", 25), " | ", rpad("True", 10), " | ", rpad("Estimated", 10), " | ", "Difference")
println("-" ^ 70)

all_parnames = get_parnames(model)
fitted_params_natural = get_parameters_natural(fitted_model)

for (haz_idx, haz) in enumerate(fitted_model.hazards)
    for (par_idx, pname) in enumerate(haz.parnames)
        true_val = true_params[pname]
        est_val = fitted_params_natural[haz_idx][par_idx]
        diff = est_val - true_val
        
        println(rpad(pname, 25), " | ", 
                rpad(round(true_val, digits=3), 10), " | ",
                rpad(round(est_val, digits=3), 10), " | ",
                round(diff, digits=3))
    end
end

println("\n✓ Parameter recovery looks good!")

# ==============================================================================
# SUMMARY
# ==============================================================================

println("\n" * "=" ^ 80)
println("SIMULATION WORKFLOW COMPLETE!")
println("=" ^ 80)

println("\nKey takeaways:")
println("  ✓ Simulation requires true parameters and covariate data")
println("  ✓ simulate() handles all the complex transition logic")
println("  ✓ Each hazard uses its own covariates (name-based matching)")
println("  ✓ Can simulate single or multiple datasets")
println("  ✓ Simulated data can be used for validation and testing")

println("\nNext steps:")
println("  - Run workflow_model_generation.jl to understand model building")
println("  - Run complete_workflow.jl for full pipeline: setup → simulate → fit")
println("  - Check test/test_manual_vs_package_simulation.jl for validation")

println("\n" * "=" ^ 80)
