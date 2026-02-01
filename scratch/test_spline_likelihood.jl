# Direct likelihood comparison test WITH SPLINE HAZARDS
using MultistateModels
using DataFrames
using Random
using BSplineKit

println("=== Direct Spline Likelihood Comparison ===")

# Create exact data with ONE known path
# Subject 1: 1→2 at t=1.0, 2→3 at t=2.0 (sojourn in state 2 = 1.0)
data = DataFrame(
    id = [1, 1],
    tstart = [0.0, 1.0],
    tstop = [1.0, 2.0],
    statefrom = [1, 2],
    stateto = [2, 3],
    obstype = [1, 1]
)

println("Data:")
println(data)

# Create model with CONSTANT spline hazards
# Use just intercept in splines so we can verify manually
# Note: Spline on semi-Markov transition h23 should use sojourn time
h12 = Hazard(@formula(0 ~ 1), :sp, 1, 2; degree=3, knots=[2.5])
h13 = Hazard(@formula(0 ~ 1), :sp, 1, 3; degree=3, knots=[2.5])
h23 = Hazard(@formula(0 ~ 1), :sp, 2, 3; degree=3, knots=[0.5])

model = multistatemodel(h12, h13, h23; data=data)

# Check parameter structure
println("\nParameter structure:")
println("Flat parameters: ", model.parameters.flat)
println("Number of params: ", length(model.parameters.flat))
println("Nested structure: ", model.parameters.nested)

# Get the hazard structures to understand spline basis
println("\n=== Hazard Structures ===")
for (name, idx) in model.hazkeys
    haz = model.hazards[idx]
    println("$name: family=$(haz.family)")
    if haz.family == :sp
        println("  degree: $(haz.degree)")
        println("  knots: $(haz.knots)")
    end
end

# Now compute the likelihood
import MultistateModels: extract_paths, ExactData, loglik_exact, set_parameters_flat!

samplepaths = extract_paths(model)
println("\nExtracted path: times=$(samplepaths[1].times), states=$(samplepaths[1].states)")

# Set all params to 1.0 for simplicity
new_flat = ones(Float64, length(model.parameters.flat))
set_parameters_flat!(model, new_flat)
println("Set all params to 1.0: ", model.parameters.flat)

data_exact = ExactData(model, samplepaths)
ll_computed = loglik_exact(model.parameters.flat, data_exact; neg=false)

println("\n=== MultistateModels Computation ===")
println("Computed log-likelihood: $ll_computed")

# If we can figure out what the hazard values are, we can compute manually
println("\n=== Manual Check ===")
println("Path: 1→2 at t=1.0, then 2→3 at t=2.0")
println("  Interval 1: [0, 1.0] in state 1")
println("    Needs: cumhaz h12([0,1]), cumhaz h13([0,1]), log(h12(1.0))")
println("  Interval 2: sojourn [0, 1.0] in state 2")
println("    Needs: cumhaz h23([0,1] sojourn), log(h23(1.0) sojourn)")
