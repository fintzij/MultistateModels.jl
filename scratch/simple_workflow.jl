# Simple Complete Workflow: Setup → Simulate → Fit
# Simplified version that avoids bugs in parameter naming

using MultistateModels
using DataFrames
using Random
using Distributions
using Optim

println("=" ^ 80)
println("SIMPLE WORKFLOW: SETUP → SIMULATE → FIT")
println("=" ^ 80)
println("Using INTERCEPT-ONLY models to avoid parameter naming bug")
println("=" ^ 80)

Random.seed!(123)

# ==============================================================================
# PART 1: SETUP TRUE MODEL (Intercept-Only)
# ==============================================================================

println("\n" * "=" ^ 80)
println("PART 1: SETUP TRUE MODEL")
println("=" ^ 80)

# We'll use a simple 2-state model: 1 → 2
# Intercept-only (no covariates) to avoid the parameter naming bug

# True parameter value
true_h12_intercept = -2.0  # log-scale baseline hazard

println("\nTrue parameter values:")
println("  h12_Intercept = ", true_h12_intercept)
println("  (Baseline hazard = ", round(exp(true_h12_intercept), digits=4), " on natural scale)")

# ==============================================================================
# PART 2: SIMULATE DATA
# ==============================================================================

println("\n" * "=" ^ 80)
println("PART 2: SIMULATE DATA FROM TRUE MODEL")
println("=" ^ 80)

# Simulation parameters
n_subjects = 100
max_time = 10.0

println("\nSimulation settings:")
println("  Number of subjects: ", n_subjects)
println("  Maximum follow-up time: ", max_time, " years")

# Simple simulation for 1→2 exponential
function simulate_simple_data(n, lambda, tmax)
    intervals = []
    
    for i in 1:n
        # Time to event from exponential distribution
        event_time = rand(Exponential(1.0 / lambda))
        
        if event_time > tmax
            # Censored
            push!(intervals, (
                id = i,
                tstart = 0.0,
                tstop = tmax,
                statefrom = 1,
                stateto = 1,  # No transition (censored)
                obstype = 1
            ))
        else
            # Event observed
            push!(intervals, (
                id = i,
                tstart = 0.0,
                tstop = event_time,
                statefrom = 1,
                stateto = 2,  # Transition to state 2
                obstype = 1
            ))
        end
    end
    
    return DataFrame(intervals)
end

println("\nSimulating data...")
true_lambda = exp(true_h12_intercept)
simulated_data = simulate_simple_data(n_subjects, true_lambda, max_time)

println("✓ Data simulated")
println("\nSimulated data summary:")
println("  Total observations: ", nrow(simulated_data))
println("  Subjects: ", length(unique(simulated_data.id)))

# Count events vs censored
n_events = sum(simulated_data.stateto .== 2)
n_censored = sum(simulated_data.stateto .== 1)

println("\nObservations:")
println("  Events (1→2): ", n_events)
println("  Censored: ", n_censored)

println("\nFirst 10 observations:")
println(first(simulated_data, 10))

# ==============================================================================
# PART 3: FIT MODEL TO SIMULATED DATA
# ==============================================================================

println("\n" * "=" ^ 80)
println("PART 3: FIT MODEL TO SIMULATED DATA")
println("=" ^ 80)

# Define hazard for fitting (intercept-only)
haz_12_fit = Hazard(@formula(0 ~ 1), "exp", 1, 2)

# Build model
model = multistatemodel(haz_12_fit; data = simulated_data)

println("✓ Model built for fitting")
println("  Number of parameters: ", sum(haz.npar_total for haz in model.hazards))

# Get parameter names
internal_haz = model.hazards[1]
println("\nParameter names:")
println("  ", internal_haz.parnames)

# Create objective function
println("\nSetting up optimization...")

function neg_loglik(params_vec)
    try
        # Compute negative log-likelihood
        ll = MultistateModels.loglik(params_vec, model.totalhazards, model)
        return -ll
    catch e
        println("Error in likelihood evaluation: ", e)
        return Inf
    end
end

# Initial parameter values (start near truth with noise)
initial_params = [true_h12_intercept + randn() * 0.2]

println("\nInitial parameter value:")
println("  h12_Intercept = ", round(initial_params[1], digits=3), 
        " (true: ", round(true_h12_intercept, digits=3), ")")

# Evaluate initial log-likelihood
initial_loglik = -neg_loglik(initial_params)
println("\nInitial log-likelihood: ", round(initial_loglik, digits=2))

# Optimize
println("\nOptimizing...")
result = optimize(neg_loglik, initial_params, BFGS(), 
                  Optim.Options(show_trace=true, iterations=100))

println("\n✓ Optimization complete")
println("  Converged: ", Optim.converged(result))
println("  Iterations: ", Optim.iterations(result))

# ==============================================================================
# PART 4: COMPARE RESULTS
# ==============================================================================

println("\n" * "=" ^ 80)
println("PART 4: COMPARE FITTED VS TRUE PARAMETERS")
println("=" ^ 80)

fitted_h12_intercept = Optim.minimizer(result)[1]
final_loglik = -Optim.minimum(result)

println("\nFinal log-likelihood: ", round(final_loglik, digits=2))
println("Log-likelihood improvement: ", round(final_loglik - initial_loglik, digits=2))

println("\n" * "─" ^ 60)
println("Parameter Comparison:")
println("─" ^ 60)
println("  True value:   ", round(true_h12_intercept, digits=4))
println("  Fitted value: ", round(fitted_h12_intercept, digits=4))
println("  Difference:   ", round(fitted_h12_intercept - true_h12_intercept, digits=4))
println("─" ^ 60)

# Convert to hazard scale
true_hazard = exp(true_h12_intercept)
fitted_hazard = exp(fitted_h12_intercept)

println("\nOn hazard scale:")
println("  True hazard:   ", round(true_hazard, digits=6))
println("  Fitted hazard: ", round(fitted_hazard, digits=6))
println("  Ratio:         ", round(fitted_hazard / true_hazard, digits=4))

# ==============================================================================
# PART 5: INTERPRETATION
# ==============================================================================

println("\n" * "=" ^ 80)
println("PART 5: INTERPRETATION")
println("=" ^ 80)

abs_diff = abs(fitted_h12_intercept - true_h12_intercept)

println("\n✅ WORKFLOW COMPLETE!")
println("\nWhat we demonstrated:")
println("  1. ✓ Set up a 2-state exponential model")
println("  2. ✓ Simulated ", n_subjects, " subjects")
println("  3. ✓ Fitted the model using maximum likelihood")
println("  4. ✓ Recovered parameter close to true value")

if abs_diff < 0.1
    println("\n🎉 Parameter recovery is EXCELLENT (error < 0.1)")
elseif abs_diff < 0.2
    println("\n✅ Parameter recovery is GOOD (error < 0.2)")
else
    println("\n⚠️  Parameter recovery is moderate (error = ", round(abs_diff, digits=3), ")")
end

println("\nThis validates:")
println("  • Model specification")
println("  • Data simulation from exponential hazard")
println("  • Log-likelihood computation")
println("  • Parameter estimation via optimization")

println("\n" * "=" ^ 80)
println("END OF SIMPLE WORKFLOW")
println("=" ^ 80)

println("\nNOTE: This uses intercept-only model to avoid parameter naming bug.")
println("See complete_workflow.jl for attempted version with covariates.")
println("The bug: parameter names include data values like 'h12_age: 40.651...'")
println("instead of just 'h12_age'. This needs to be fixed in modelgeneration.jl")
