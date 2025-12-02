using MultistateModels
using Random
using DataFrames
using Statistics

Random.seed!(12345)

println("="^70)
println("Testing phase-type importance sampling")
println("="^70)

# Simple illness-death model with panel observations
h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)  
h13 = Hazard(@formula(0 ~ 1), "wei", 1, 3)  
h23 = Hazard(@formula(0 ~ 1), "wei", 2, 3)  

dat = DataFrame(
    id = [1, 1, 1, 1],
    tstart = [0.0, 0.5, 1.0, 1.5],
    tstop = [0.5, 1.0, 1.5, 2.0],
    statefrom = [1, 1, 1, 1],
    stateto = [1, 1, 2, 3],
    obstype = [2, 2, 2, 2]  # Panel observations
)

model = multistatemodel(h12, h13, h23; data=dat)

# Build phase-type infrastructure
tmat = model.tmat
phasetype_config = MultistateModels.PhaseTypeConfig(n_phases=[2, 2, 1])
phasetype_surrogate = MultistateModels.build_phasetype_surrogate(tmat, phasetype_config)
emat_ph = MultistateModels.build_phasetype_emat_expanded(model, phasetype_surrogate)

println("\nPhase-type surrogate:")
println("  Observed states -> phases:")
for (s, phases) in enumerate(phasetype_surrogate.state_to_phases)
    println("    State $s: phases $phases")
end

# Compute phase-type marginal likelihood
ll_ph_marginal = MultistateModels.compute_phasetype_marginal_loglik(model, phasetype_surrogate, emat_ph)
println("\nPhase-type marginal log-likelihood r(Y|θ'): $(round(ll_ph_marginal, digits=4))")

# Build sampling infrastructure
books = MultistateModels.build_tpm_mapping(model.data)
absorbingstates = findall([isa(h, MultistateModels._TotalHazardAbsorbing) for h in model.totalhazards])
tpm_book_ph, hazmat_book_ph = MultistateModels.build_phasetype_tpm_book(phasetype_surrogate, books, model.data)
fbmats_ph = MultistateModels.build_fbmats_phasetype(model, phasetype_surrogate)

# Sample paths and compute importance weights
println("\n--- Sample paths ---")
n_paths_show = 3
for p in 1:n_paths_show
    path_result = MultistateModels.draw_samplepath_phasetype(1, model, tpm_book_ph, hazmat_book_ph, 
                                                             books[2], fbmats_ph, emat_ph, 
                                                             phasetype_surrogate, absorbingstates)
    
    loglik_target = MultistateModels.loglik(model.parameters, path_result.collapsed, model.hazards, model)
    loglik_surrog = MultistateModels.loglik_phasetype_expanded(path_result.expanded, phasetype_surrogate)
    
    println("\nPath $p:")
    println("  Collapsed: states=$(path_result.collapsed.states)")
    println("  Expanded: states=$(path_result.expanded.states)")
    println("  log f(Z|θ): $(round(loglik_target, digits=4))")
    println("  log h(Z|θ'): $(round(loglik_surrog, digits=4))")
    println("  log weight: $(round(loglik_target - loglik_surrog, digits=4))")
end

# Full IS estimate
println("\n--- Monte Carlo estimate (n=1000) ---")
n_paths = 1000
log_weights = Float64[]

for p in 1:n_paths
    path_result = MultistateModels.draw_samplepath_phasetype(1, model, tpm_book_ph, hazmat_book_ph, 
                                                             books[2], fbmats_ph, emat_ph, 
                                                             phasetype_surrogate, absorbingstates)
    loglik_target = MultistateModels.loglik(model.parameters, path_result.collapsed, model.hazards, model)
    loglik_surrog = MultistateModels.loglik_phasetype_expanded(path_result.expanded, phasetype_surrogate)
    push!(log_weights, loglik_target - loglik_surrog)
end

unnorm_weights = exp.(log_weights)
log_ml_is = ll_ph_marginal + log(mean(unnorm_weights))

println("Log weight stats: mean=$(round(mean(log_weights), digits=3)), std=$(round(std(log_weights), digits=3))")
println("Phase-type IS estimate: $(round(log_ml_is, digits=4))")
println("  = r(Y|θ') + log(mean(ν))")
println("  = $(round(ll_ph_marginal, digits=4)) + $(round(log(mean(unnorm_weights)), digits=4))")

# Compare to Markov surrogate
surrogate_fitted = MultistateModels.fit_surrogate(model; verbose=false)
println("\nMarkov surrogate log-likelihood: $(round(surrogate_fitted.loglik.loglik, digits=4))")

# They should be similar if the target equals the Markov surrogate
println("\nExpected: Phase-type IS estimate ≈ Markov surrogate loglik (within MC error)")
