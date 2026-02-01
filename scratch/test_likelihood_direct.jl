# Direct likelihood comparison test
using MultistateModels
using DataFrames
using Random

println("=== Direct Likelihood Comparison ===")

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

# Create model with CONSTANT hazards for easy manual verification
# We'll use the exponential family which has h(t) = exp(param)
h12 = Hazard(@formula(0 ~ 1), :exp, 1, 2)
h13 = Hazard(@formula(0 ~ 1), :exp, 1, 3)
h23 = Hazard(@formula(0 ~ 1), :exp, 2, 3)

model = multistatemodel(h12, h13, h23; data=data)

# Set specific parameter values
# v0.3.0+ uses NATURAL scale: the parameter IS the rate (not log(rate))
# Let's use h12 = 0.5, h13 = 0.3, h23 = 0.4
h12_rate = 0.5
h13_rate = 0.3
h23_rate = 0.4

# Get current parameters
println("\nCurrent flat parameters: ", model.parameters.flat)
println("Current nested parameters: ", model.parameters.nested)

# Set parameters using the proper API
import MultistateModels: set_parameters_flat!, get_parameters_flat
new_flat = [h12_rate, h13_rate, h23_rate]  # Direct rates, NOT log!
set_parameters_flat!(model, new_flat)

# Verify
println("\nAfter setting parameters:")
println("Flat: ", model.parameters.flat)

# Manual likelihood calculation
# Path: 1→2 at t=1.0, 2→3 at t=2.0
#
# Interval 1: [0, 1.0] in state 1
#   - h12(t) = 0.5 (constant rate on natural scale)
#   - h13(t) = 0.3 (constant rate on natural scale)
#   - Cumhaz h12([0,1]) = 0.5 * 1.0 = 0.5
#   - Cumhaz h13([0,1]) = 0.3 * 1.0 = 0.3
#   - Transition 1→2: log(h12(1.0)) = log(0.5)
#
# Interval 2: [0, 1.0] sojourn in state 2 (NOT [1.0, 2.0]!)
#   - h23(s) = 0.4 (constant in sojourn time s)
#   - Cumhaz h23([0,1]) = 0.4 * 1.0 = 0.4  (sojourn time!)
#   - Transition 2→3: log(h23(1.0)) = log(0.4)  (sojourn time!)
#
# Total log-likelihood = -(0.5 + 0.3) + log(0.5) - 0.4 + log(0.4)

h12_rate = 0.5
h13_rate = 0.3
h23_rate = 0.4

cumhaz_12 = h12_rate * 1.0
cumhaz_13 = h13_rate * 1.0
cumhaz_23 = h23_rate * 1.0  # sojourn = 1.0

ll_manual = -cumhaz_12 - cumhaz_13 + log(h12_rate) - cumhaz_23 + log(h23_rate)

println("\n=== Manual Calculation ===")
println("Cumhaz h12([0,1]): $cumhaz_12")
println("Cumhaz h13([0,1]): $cumhaz_13")
println("Cumhaz h23([0,1] sojourn): $cumhaz_23")
println("log(h12(1.0)): $(log(h12_rate))")
println("log(h23(1.0) sojourn): $(log(h23_rate))")
println("Manual log-likelihood: $ll_manual")

# Now compute using MultistateModels
import MultistateModels: extract_paths, ExactData, loglik_exact

samplepaths = extract_paths(model)
println("\nExtracted path: times=$(samplepaths[1].times), states=$(samplepaths[1].states)")

flat_pars = model.parameters.flat
println("Flat parameters: $flat_pars")

data_exact = ExactData(model, samplepaths)
ll_computed = loglik_exact(flat_pars, data_exact; neg=false)

println("\n=== MultistateModels Computation ===")
println("Computed log-likelihood: $ll_computed")

println("\n=== Comparison ===")
println("Manual:   $ll_manual")
println("Computed: $ll_computed")
println("Difference: $(ll_computed - ll_manual)")

if abs(ll_computed - ll_manual) < 1e-6
    println("✓ MATCH!")
else
    println("✗ MISMATCH - indicates a bug!")
    println("\nLet's debug more...")
    
    # Check what the extracted path looks like
    path = samplepaths[1]
    println("\nPath details:")
    println("  times: $(path.times)")
    println("  states: $(path.states)")
    
    # Calculate what the package should compute
    println("\nExpected computation:")
    println("  Interval [0, 1.0], state 1→2:")
    println("    Subject enters state 1 at time 0")
    println("    Subject transitions to state 2 at time 1")
    println("    h12 and h13 active over [0, 1]")
    println("  Interval [1.0, 2.0], state 2→3:")
    println("    Subject enters state 2 at time 1 (sojourn starts)")
    println("    Subject transitions to state 3 at time 2")
    println("    Sojourn = 2 - 1 = 1")
    println("    h23 active over sojourn [0, 1]")
end
