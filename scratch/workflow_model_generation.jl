# Model Generation Workflow - Step by Step
# This script walks through building a multistate model from scratch

using MultistateModels
using DataFrames
using Random

println("=" ^ 80)
println("MULTISTATE MODEL GENERATION WORKFLOW")
println("=" ^ 80)

Random.seed!(123)

# ==============================================================================
# STEP 1: UNDERSTAND THE PROBLEM
# ==============================================================================

println("\n" * "=" ^ 80)
println("STEP 1: DEFINE THE MULTISTATE PROCESS")
println("=" ^ 80)

println("""
We want to model a 3-state illness-death process:
  
  State 1 (Healthy) ──┐
         │            │
         │            │
         ▼            ▼
  State 2 (Illness) ─► State 3 (Death)
  
Possible transitions:
  1 → 2: Healthy to Illness
  1 → 3: Healthy to Death (direct)
  2 → 3: Illness to Death
  
State 3 (Death) is absorbing - no transitions out.
""")

println("✓ Model structure defined")

# ==============================================================================
# STEP 2: SPECIFY HAZARDS
# ==============================================================================

println("\n" * "=" ^ 80)
println("STEP 2: SPECIFY HAZARD FUNCTIONS FOR EACH TRANSITION")
println("=" ^ 80)

println("""
For each transition, we need to specify:
  1. Hazard family (exponential, Weibull, Gompertz, splines)
  2. Covariates that affect the hazard
  3. Origin and destination states

Formula syntax: @formula(0 ~ 1 + covariate1 + covariate2 + ...)
  - Use `1 +` for intercept (required for exponential hazards with covariates)
  - Covariates must exist in your data
""")

# Transition 1→2: Healthy to Illness
# Depends on age and sex
println("\nCreating hazard 1→2 (Healthy → Illness):")
println("  Family: Exponential (constant hazard)")
println("  Covariates: age, sex")
println("  Formula: @formula(0 ~ 1 + age + sex)")

haz_12 = Hazard(@formula(0 ~ 1 + age + sex), "exp", 1, 2)

println("  ✓ Hazard specification created: ", typeof(haz_12))

# Transition 1→3: Healthy to Death
# Depends only on age
println("\nCreating hazard 1→3 (Healthy → Death):")
println("  Family: Exponential")
println("  Covariates: age")
println("  Formula: @formula(0 ~ 1 + age)")

haz_13 = Hazard(@formula(0 ~ 1 + age), "exp", 1, 3)

println("  ✓ Hazard specification created: ", typeof(haz_13))

# Transition 2→3: Illness to Death
# Depends on age and treatment
println("\nCreating hazard 2→3 (Illness → Death):")
println("  Family: Exponential")
println("  Covariates: age, treatment")
println("  Formula: @formula(0 ~ 1 + age + treatment)")

haz_23 = Hazard(@formula(0 ~ 1 + age + treatment), "exp", 2, 3)

println("  ✓ Hazard specification created: ", typeof(haz_23))

println("\n✓ All hazard specifications created")
println("\nNote: Different transitions can use different covariates!")
println("  1→2 uses: age, sex")
println("  1→3 uses: age")
println("  2→3 uses: age, treatment")

# ==============================================================================
# STEP 3: PREPARE DATA
# ==============================================================================

println("\n" * "=" ^ 80)
println("STEP 3: PREPARE DATA IN REQUIRED FORMAT")
println("=" ^ 80)

println("""
Data must be in interval format with required columns:
  - id: Subject identifier
  - tstart: Interval start time
  - tstop: Interval stop time
  - statefrom: State at tstart
  - stateto: State at tstop (same as statefrom if no transition)
  - obstype: Observation type (1 = exact, 2 = right-censored, 3 = interval-censored)
  - [covariates]: All covariates used in any hazard
""")

# Create sample data
n_subjects = 100

# Generate covariate data
println("\nGenerating subject-level covariates:")
covariate_data = DataFrame(
    id = 1:n_subjects,
    age = rand(40.0:80.0, n_subjects),
    sex = rand([0, 1], n_subjects),
    treatment = rand([0, 1], n_subjects)
)

println("  Covariates generated for ", n_subjects, " subjects")
println("  Columns: ", names(covariate_data))

# Create interval data structure
# For this example, we'll create a simple structure where each subject
# has one observation starting in state 1
println("\nCreating interval-format data structure:")

interval_data = DataFrame(
    id = 1:n_subjects,
    tstart = zeros(n_subjects),
    tstop = rand(1.0:10.0, n_subjects),  # Random observation times
    statefrom = ones(Int, n_subjects),   # All start in state 1
    stateto = rand([1, 2, 3], n_subjects),  # Random observed states
    obstype = ones(Int, n_subjects)      # All exact observations
)

println("  Interval structure created")

# Combine with covariates
model_data = hcat(interval_data, covariate_data[:, [:age, :sex, :treatment]])

println("  Combined with covariates")
println("\nSample data (first 5 rows):")
println(first(model_data, 5))

println("\n✓ Data prepared in required format")

# ==============================================================================
# STEP 4: BUILD THE MODEL
# ==============================================================================

println("\n" * "=" ^ 80)
println("STEP 4: BUILD THE MULTISTATE MODEL")
println("=" ^ 80)

println("""
The multistatemodel() function combines:
  - Hazard specifications
  - Data
  
It will:
  1. Create internal hazard structures
  2. Extract parameter names from formulas
  3. Initialize parameters
  4. Build transition matrix
  5. Set up data structures for fitting
""")

println("\nCalling multistatemodel()...")
model = multistatemodel(haz_12, haz_13, haz_23; data = model_data)

println("✓ Model created successfully!")

# ==============================================================================
# STEP 5: INSPECT THE MODEL
# ==============================================================================

println("\n" * "=" ^ 80)
println("STEP 5: INSPECT THE MODEL STRUCTURE")
println("=" ^ 80)

println("\nModel type: ", typeof(model))
println("Number of hazards: ", length(model.hazards))
println("Number of subjects: ", length(unique(model.data.id)))

println("\nHazard details:")
for (i, haz) in enumerate(model.hazards)
    println("  Hazard ", i, " (", haz.statefrom, "→", haz.stateto, "):")
    println("    Family: ", haz.family)
    println("    Number of parameters: ", length(model.parameters[i]))
end

println("\nParameter names (in order):")
all_parnames = get_parnames(model)
for (i, pname) in enumerate(all_parnames)
    println("  ", i, ". ", pname)
end

println("\nCurrent parameter values (initialized to 0.0):")
for (i, haz) in enumerate(model.hazards)
    println("  Hazard ", haz.statefrom, "→", haz.stateto, ": ", model.parameters[i])
end

# ==============================================================================
# STEP 6: UNDERSTAND NAME-BASED MATCHING
# ==============================================================================

println("\n" * "=" ^ 80)
println("STEP 6: HOW NAME-BASED COVARIATE MATCHING WORKS")
println("=" ^ 80)

println("""
Each hazard extracts only its own covariates by name:
  
  Hazard 1→2: Needs (age, sex)
    - Looks for columns 'age' and 'sex' in data
    - Ignores 'treatment' column
  
  Hazard 1→3: Needs (age)
    - Looks for column 'age' in data
    - Ignores 'sex' and 'treatment' columns
  
  Hazard 2→3: Needs (age, treatment)
    - Looks for columns 'age' and 'treatment' in data
    - Ignores 'sex' column

This prevents bugs where:
  - Wrong covariates are used for a hazard
  - Column order matters
  - Hazards interfere with each other
""")

# Demonstrate covariate extraction
sample_row = model_data[1, :]

println("\nSample data row:")
println("  id = ", sample_row.id)
println("  age = ", sample_row.age)
println("  sex = ", sample_row.sex)
println("  treatment = ", sample_row.treatment)

# Show what each hazard extracts
haz_12_internal = model.hazards[1]
haz_13_internal = model.hazards[2]
haz_23_internal = model.hazards[3]

covar_names_12 = MultistateModels.extract_covar_names(haz_12_internal.parnames)
covar_names_13 = MultistateModels.extract_covar_names(haz_13_internal.parnames)
covar_names_23 = MultistateModels.extract_covar_names(haz_23_internal.parnames)

covars_12 = MultistateModels.extract_covariates(sample_row, covar_names_12)
covars_13 = MultistateModels.extract_covariates(sample_row, covar_names_13)
covars_23 = MultistateModels.extract_covariates(sample_row, covar_names_23)

println("\nExtracted covariates:")
println("  Hazard 1→2 extracts: ", covars_12)
println("  Hazard 1→3 extracts: ", covars_13)
println("  Hazard 2→3 extracts: ", covars_23)

println("\n✓ Each hazard correctly extracts only its covariates by name!")

# ==============================================================================
# STEP 7: NEXT STEPS
# ==============================================================================

println("\n" * "=" ^ 80)
println("STEP 7: WHAT YOU CAN DO WITH THE MODEL")
println("=" ^ 80)

println("""
Now that you have a model, you can:

1. Set parameters manually:
   set_parameters!(model, [[par1, par2, ...], [par3, ...], ...])

2. Fit the model to data:
   fitted_model = fit(model)

3. Simulate from the model:
   sim_data = simulate(model; nsim=1, data=true)

4. Extract parameter estimates:
   params = get_parameters(fitted_model)
   params_natural = get_parameters_natural(fitted_model)

5. Get log-likelihood:
   ll = get_loglik(fitted_model)

6. Get standard errors and confidence intervals:
   se = get_se(fitted_model)
   ci = get_confint(fitted_model)
""")

println("\n" * "=" ^ 80)
println("MODEL GENERATION WORKFLOW COMPLETE!")
println("=" ^ 80)

println("\nKey takeaways:")
println("  ✓ Each transition needs a hazard specification")
println("  ✓ Different hazards can use different covariates")
println("  ✓ Data must be in interval format with required columns")
println("  ✓ Model automatically handles covariate matching by name")
println("  ✓ Parameters are initialized and ready for fitting")

println("\nNext: See workflow_simulation.jl for simulation details")
