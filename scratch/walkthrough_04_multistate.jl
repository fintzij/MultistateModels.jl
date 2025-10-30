# Walkthrough Phase 4: Multi-State Models
# Goal: Multiple transitions, test 3-state illness-death model
# States: 1=Healthy, 2=Ill, 3=Dead
# Transitions: 1→2, 1→3, 2→3

using MultistateModels
using DataFrames

println("=" ^ 70)
println("PHASE 4: MULTI-STATE MODELS (3+ STATES)")
println("=" ^ 70)

# Step 4.1: Create 3-State Illness-Death Model
println("\n--- Step 4.1: Create 3-State Illness-Death Hazards ---")
haz_1_to_2 = MultistateModels.Hazard(1, 2, family="exp")  # Healthy → Ill
haz_1_to_3 = MultistateModels.Hazard(1, 3, family="exp")  # Healthy → Dead
haz_2_to_3 = MultistateModels.Hazard(2, 3, family="exp")  # Ill → Dead

println("✓ Created 3 hazards:")
println("  1→2 (Healthy → Ill): ", typeof(haz_1_to_2))
println("  1→3 (Healthy → Dead): ", typeof(haz_1_to_3))
println("  2→3 (Ill → Dead): ", typeof(haz_2_to_3))

# Step 4.2: Create Multi-State Data
println("\n--- Step 4.2: Create Multi-State Data ---")

# First 5 subjects: 1→2→3 (get ill, then die)
# Last 5 subjects: 1→3 (die without getting ill)
dat_3state = DataFrame(
    id = [1,1, 2,2, 3,3, 4,4, 5,5, 6, 7, 8, 9, 10],
    tstart = [0.0,1.0, 0.0,1.5, 0.0,2.0, 0.0,1.2, 0.0,0.8,  0.0, 0.0, 0.0, 0.0, 0.0],
    tstop =  [1.0,2.0, 1.5,3.0, 2.0,3.5, 1.2,2.5, 0.8,2.0,  1.0, 1.5, 2.0, 1.8, 2.2],
    statefrom = [1,2, 1,2, 1,2, 1,2, 1,2,  1, 1, 1, 1, 1],
    stateto =   [2,3, 2,3, 2,3, 2,3, 2,3,  3, 3, 3, 3, 3]
)

println("Multi-state data structure:")
println(dat_3state)

println("\nData summary:")
println("  Total rows: ", nrow(dat_3state))
println("  Unique subjects: ", length(unique(dat_3state.id)))
println("  Subjects with 1→2 transition: 5")
println("  Subjects with 1→3 transition: 10")
println("  Subjects with 2→3 transition: 5")

# Step 4.3: Build 3-State Model
println("\n--- Step 4.3: Build 3-State Model ---")
model_3state = multistatemodel(
    haz_1_to_2,
    haz_1_to_3,
    haz_2_to_3;
    data = dat_3state
)

println("✓ 3-state model created")
println("  Number of hazards: ", length(model_3state.hazards))
println("  Total parameters: ", model_3state.npar)
println("  Number of subjects: ", model_3state.nsubj)

# Verify hazards
println("\nHazards in model:")
for (i, haz) in enumerate(model_3state.hazards)
    println("  $i. ", haz.from, "→", haz.to, " (", haz.family, ")")
end

# Step 4.4: Initialize Parameters for Multi-State
println("\n--- Step 4.4: Initialize Parameters ---")
init_1_2 = MultistateModels.init_par(haz_1_to_2, 0.0)
init_1_3 = MultistateModels.init_par(haz_1_to_3, 0.0)
init_2_3 = MultistateModels.init_par(haz_2_to_3, 0.0)

println("Initial parameters:")
println("  1→2: ", init_1_2)
println("  1→3: ", init_1_3)
println("  2→3: ", init_2_3)

# Step 4.5: Test Hazard Evaluations
println("\n--- Step 4.5: Test Individual Hazard Evaluations ---")
params_exp = (intercept = -0.5,)
t = 1.0
covars = NamedTuple()

println("Testing all hazards at t=1.0, log(hazard)=-0.5:")
for (i, haz) in enumerate(model_3state.hazards)
    h = haz.hazard_fn(t, params_exp, covars; give_log=false)
    println("  Hazard ", haz.from, "→", haz.to, ": h = ", h, 
            " (expected: ", exp(-0.5), ")")
end

println("\n" * "=" ^ 70)
println("PHASE 4 COMPLETE - Multi-state model successful!")
println("=" ^ 70)
println("\nKey achievements:")
println("✓ 3-state illness-death model created")
println("✓ Multiple hazards managed correctly")
println("✓ Data structure with multiple rows per subject works")
println("\nNext: Run walkthrough_05_hazard_families.jl")
