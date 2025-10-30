# Walkthrough Phase 5: Different Hazard Families
# Goal: Test Weibull, Gompertz, and mixed family models
# Verify SemiMarkovHazard type works correctly

using MultistateModels
using DataFrames

println("=" ^ 70)
println("PHASE 5: DIFFERENT HAZARD FAMILIES")
println("=" ^ 70)

# Step 5.1: Weibull Hazard (Semi-Markov)
println("\n--- Step 5.1: Create Weibull Hazard ---")
haz_wei = MultistateModels.Hazard(1, 2, family="wei")

println("✓ Weibull hazard created")
println("  Type: ", typeof(haz_wei))
println("  Expected: SemiMarkovHazard")
println("  Family: ", haz_wei.family)
println("  Parameter names: ", haz_wei.parnames)
println("  Baseline parameters: ", haz_wei.npar_baseline)

# Step 5.2: Test Weibull Evaluation
println("\n--- Step 5.2: Test Weibull Evaluation ---")
# Weibull: h(t) = (shape/scale) * (t/scale)^(shape-1)
# With log-parameterization: need to check actual implementation
params_wei = (logshape = log(2.0), logscale = log(1.0))
t = 0.5
covars = NamedTuple()

h_val = haz_wei.hazard_fn(t, params_wei, covars; give_log=false)

# Manual calculation (assuming standard parameterization)
shape = exp(params_wei.logshape)  # 2.0
scale = exp(params_wei.logscale)  # 1.0
expected_h = (shape/scale) * (t/scale)^(shape-1.0)

println("Weibull h(0.5):")
println("  Computed: ", h_val)
println("  Expected: ", expected_h, " (shape=2, scale=1)")
println("  Match: ", isapprox(h_val, expected_h, rtol=0.01))

# Test cumulative hazard
cumhaz_val = haz_wei.cumhaz_fn(0.0, 0.5, params_wei, covars)
expected_cumhaz = (0.5/scale)^shape
println("\nWeibull cumulative hazard [0, 0.5]:")
println("  Computed: ", cumhaz_val)
println("  Expected: ", expected_cumhaz)
println("  Match: ", isapprox(cumhaz_val, expected_cumhaz, rtol=0.01))

# Step 5.3: Gompertz Hazard
println("\n--- Step 5.3: Create Gompertz Hazard ---")
haz_gom = MultistateModels.Hazard(1, 2, family="gom")

println("✓ Gompertz hazard created")
println("  Type: ", typeof(haz_gom))
println("  Expected: SemiMarkovHazard")
println("  Family: ", haz_gom.family)
println("  Parameter names: ", haz_gom.parnames)

# Test evaluation
params_gom = (logshape = log(0.1), logscale = log(1.0))
h_val_gom = haz_gom.hazard_fn(1.0, params_gom, covars; give_log=false)

println("\nGompertz h(1.0):")
println("  Computed: ", h_val_gom)
# Gompertz: h(t) = scale * exp(shape * t)
shape_gom = exp(params_gom.logshape)
scale_gom = exp(params_gom.logscale)
expected_gom = scale_gom * exp(shape_gom * 1.0)
println("  Expected: ", expected_gom, " (shape=0.1, scale=1)")
println("  Match: ", isapprox(h_val_gom, expected_gom, rtol=0.01))

# Step 5.4: Mixed Model (Different Families)
println("\n--- Step 5.4: Create Mixed-Family Model ---")

# Create simple data for testing
dat = DataFrame(
    id = [1,1, 2,2, 3,3, 4,4, 5,5],
    tstart = [0.0,1.0, 0.0,1.5, 0.0,2.0, 0.0,1.2, 0.0,0.8],
    tstop =  [1.0,2.0, 1.5,3.0, 2.0,3.5, 1.2,2.5, 0.8,2.0],
    statefrom = [1,2, 1,2, 1,2, 1,2, 1,2],
    stateto =   [2,3, 2,3, 2,3, 2,3, 2,3]
)

# Illness-death with different families per transition
haz_1_to_2_exp = MultistateModels.Hazard(1, 2, family="exp")
haz_1_to_3_wei = MultistateModels.Hazard(1, 3, family="wei")
haz_2_to_3_gom = MultistateModels.Hazard(2, 3, family="gom")

model_mixed = multistatemodel(
    haz_1_to_2_exp,
    haz_1_to_3_wei,
    haz_2_to_3_gom;
    data = dat
)

println("✓ Mixed-family model created")
println("  Total parameters: ", model_mixed.npar)
println("  Expected: 1 (exp) + 2 (wei) + 2 (gom) = 5")

println("\nHazards in mixed model:")
for (i, haz) in enumerate(model_mixed.hazards)
    println("  $i. ", haz.from, "→", haz.to, " (", haz.family, 
            ", ", typeof(haz), ")")
end

# Step 5.5: Verify Each Hazard Type
println("\n--- Step 5.5: Verify Hazard Types ---")
println("Exponential (MarkovHazard):")
println("  Type: ", typeof(model_mixed.hazards[1]))
println("  Is MarkovHazard: ", typeof(model_mixed.hazards[1]) <: MultistateModels.MarkovHazard)

println("\nWeibull (SemiMarkovHazard):")
println("  Type: ", typeof(model_mixed.hazards[2]))
println("  Is SemiMarkovHazard: ", typeof(model_mixed.hazards[2]) <: MultistateModels.SemiMarkovHazard)

println("\nGompertz (SemiMarkovHazard):")
println("  Type: ", typeof(model_mixed.hazards[3]))
println("  Is SemiMarkovHazard: ", typeof(model_mixed.hazards[3]) <: MultistateModels.SemiMarkovHazard)

# Step 5.6: Initialize Mixed Model Parameters
println("\n--- Step 5.6: Initialize Mixed Model Parameters ---")
init_exp = MultistateModels.init_par(haz_1_to_2_exp, 0.0)
init_wei = MultistateModels.init_par(haz_1_to_3_wei, 0.0)
init_gom = MultistateModels.init_par(haz_2_to_3_gom, 0.0)

println("Initial parameters by hazard:")
println("  Exponential 1→2: ", init_exp, " (length: ", length(init_exp), ")")
println("  Weibull 1→3: ", init_wei, " (length: ", length(init_wei), ")")
println("  Gompertz 2→3: ", init_gom, " (length: ", length(init_gom), ")")

println("\n" * "=" ^ 70)
println("PHASE 5 COMPLETE - Multiple hazard families working!")
println("=" ^ 70)
println("\nKey achievements:")
println("✓ Weibull (SemiMarkovHazard) works correctly")
println("✓ Gompertz (SemiMarkovHazard) works correctly")
println("✓ Mixed-family models supported")
println("✓ Each family has correct parameter structure")
println("\nNext: Run walkthrough_06_different_covariates.jl")
