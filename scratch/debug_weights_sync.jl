# Debug script: Break immediately on path explosion, trace parameter/weight sync
using MultistateModels, DataFrames, Random, Statistics

Random.seed!(52787)

# Setup reversible model with TVC (from longtest)
statefrom = fill(1, 400)
datTVC = DataFrame(
    id = repeat(1:200, inner=2),
    tstart = repeat([0.0, 1.0], 200),
    tstop = repeat([1.0, 5.0], 200),
    statefrom = statefrom,
    stateto = fill(1, 400),
    obstype = fill(2, 400),
    x1 = repeat([0, 1], 200))

model = multistatemodel(
    Hazard(@formula(0 ~ 1 + x1), "wei", 1, 2),
    Hazard(@formula(0 ~ 1 + x1), "wei", 2, 1);
    data=datTVC, surrogate=:markov
)

# Set true parameters and simulate
true_h12 = [log(1.0), log(0.5), 0.5]
true_h21 = [log(1.0), log(0.3), -0.2] 
MultistateModels.set_parameters!(model, (h12=true_h12, h21=true_h21))

simdat = simulate(model; paths=false, data=true)[1]

# Create fitting model  
model_fit = multistatemodel(
    Hazard(@formula(0 ~ 1 + x1), "wei", 1, 2),
    Hazard(@formula(0 ~ 1 + x1), "wei", 2, 1);
    data=simdat, surrogate=:markov
)

println("=" ^ 70)
println("DEBUG: Comparing target vs surrogate parameters")
println("=" ^ 70)

# 1. Check initial parameters of target model
init_params_target = MultistateModels.get_parameters_flat(model_fit)
println("\n1. INITIAL TARGET MODEL PARAMS")
println("   Flat params: ", round.(init_params_target, digits=6))

# 2. Check surrogate model parameters
surrog = model_fit.markovsurrogate
println("\n2. MARKOV SURROGATE")
println("   Type: ", typeof(surrog))
println("   Has parameters: ", hasfield(typeof(surrog), :parameters))
surrog_params = surrog.parameters
println("   Parameters type: ", typeof(surrog_params))
println("   Parameters keys: ", keys(surrog_params))

# Get surrogate params via get_log_scale_params
surrog_log_pars = MultistateModels.get_log_scale_params(surrog_params)
println("   Surrogate log-scale params: ", [collect(p) for p in surrog_log_pars])

# 3. Check if they are the same
println("\n3. COMPARISON TARGET vs SURROGATE")
target_log_pars = MultistateModels.get_log_scale_params(model_fit.parameters)
println("   Target log-scale params: ", [collect(p) for p in target_log_pars])
println("   Surrogate log-scale params: ", [collect(p) for p in surrog_log_pars])

# 4. Set target to true params and check again  
println("\n4. AFTER SETTING TARGET TO TRUE PARAMS")
MultistateModels.set_parameters!(model_fit, (h12=true_h12, h21=true_h21))
updated_target_pars = MultistateModels.get_log_scale_params(model_fit.parameters)
println("   Target log-scale params: ", [collect(p) for p in updated_target_pars])
updated_surrog_pars = MultistateModels.get_log_scale_params(model_fit.markovsurrogate.parameters)
println("   Surrogate log-scale params: ", [collect(p) for p in updated_surrog_pars])
println("   ARE THEY DIFFERENT? ", updated_target_pars != updated_surrog_pars)

# 5. What is the actual Markov surrogate doing?
println("\n5. MARKOV SURROGATE DETAILS")
println("   Type: ", typeof(surrog))
println("   Hazards: ", length(surrog.hazards), " hazards")
for (i, h) in enumerate(surrog.hazards)
    println("   Hazard $i: ", h.statefrom, " → ", h.stateto, ", family: ", h.family)
end

# 6. This is the key issue: when we draw paths from surrogate,
#    what log-likelihood do we compute?
println("\n6. SIMULATING WHAT HAPPENS IN DrawSamplePaths!")

# Get Markov surrogate params (used for sampling & surrogate LL)
surrog_pars_for_ll = MultistateModels.get_log_scale_params(surrog.parameters)
println("   Surrogate params for LL: ", [collect(p) for p in surrog_pars_for_ll])

# Get target params (used for target LL in importance weights)  
target_pars_for_ll = MultistateModels.nest_params(updated_params_target, model_fit.parameters)
println("   Target params for LL: ", [collect(p) for p in target_pars_for_ll])

# 7. KEY QUESTION: Does set_parameters! update the surrogate?
println("\n7. KEY TEST: Does set_parameters! update the surrogate?")
# Check before and after
MultistateModels.set_parameters!(model_fit, (h12=[0.5, 0.5, 0.5], h21=[0.5, 0.5, 0.5]))
test_target = MultistateModels.get_log_scale_params(model_fit.parameters)
test_surrog = MultistateModels.get_log_scale_params(model_fit.markovsurrogate.parameters)
println("   After setting target to [0.5, 0.5, 0.5]:")
println("   Target: ", [collect(p) for p in test_target])
println("   Surrogate: ", [collect(p) for p in test_surrog])
println("   Surrogate was updated: ", test_target == test_surrog)

println("\n" * "=" ^ 70)
println("CRITICAL: If surrogate is NOT updated when target changes,")
println("importance weights will be computed incorrectly!")
println("=" ^ 70)
