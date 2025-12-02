using MultistateModels
using Random
using DataFrames
using Statistics

Random.seed!(12345)

# Simple setup
h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)  
h13 = Hazard(@formula(0 ~ 1), "wei", 1, 3)  
h23 = Hazard(@formula(0 ~ 1), "wei", 2, 3)  

dat = DataFrame(
    id = [1, 1, 1, 1],
    tstart = [0.0, 0.5, 1.0, 1.5],
    tstop = [0.5, 1.0, 1.5, 2.0],
    statefrom = [1, 1, 1, 1],
    stateto = [1, 1, 2, 3],
    obstype = [2, 2, 2, 2]
)

model = multistatemodel(h12, h13, h23; data=dat)

# Build infrastructure
tmat = model.tmat
phasetype_config = MultistateModels.PhaseTypeConfig(n_phases=[2, 2, 1])
phasetype_surrogate = MultistateModels.build_phasetype_surrogate(tmat, phasetype_config)
emat_ph = MultistateModels.build_phasetype_emat_expanded(model, phasetype_surrogate)

# NormConstantProposal
ll_ph_marginal = MultistateModels.compute_phasetype_marginal_loglik(model, phasetype_surrogate, emat_ph)
println("Phase-type NormConstantProposal (r(Y|θ')): ", round(ll_ph_marginal, digits=3))

# Build books
books = MultistateModels.build_tpm_mapping(model.data)
absorbingstates = findall([isa(h, MultistateModels._TotalHazardAbsorbing) for h in model.totalhazards])
tpm_book_ph, hazmat_book_ph = MultistateModels.build_phasetype_tpm_book(phasetype_surrogate, books, model.data)
fbmats_ph = MultistateModels.build_fbmats_phasetype(model, phasetype_surrogate)

# Sample many paths and compute importance weights
n_paths = 100
log_weights = Float64[]

for p in 1:n_paths
    path_result = MultistateModels.draw_samplepath_phasetype(1, model, tpm_book_ph, hazmat_book_ph, 
                                                             books[2], fbmats_ph, emat_ph, 
                                                             phasetype_surrogate, absorbingstates)
    loglik_target = MultistateModels.loglik(model.parameters, path_result.collapsed, model.hazards, model)
    loglik_surrog = MultistateModels.loglik_phasetype_expanded(path_result.expanded, phasetype_surrogate)
    push!(log_weights, loglik_target - loglik_surrog)
end

println("\nLog importance weights statistics:")
println("  Mean: ", round(mean(log_weights), digits=3))
println("  Std: ", round(std(log_weights), digits=3))
println("  Min: ", round(minimum(log_weights), digits=3))
println("  Max: ", round(maximum(log_weights), digits=3))

# The proper IS estimate is:
# log f̂(Y|θ) = log r(Y|θ') + log(mean(exp(log_weights)))
unnorm_weights = exp.(log_weights)
println("\nUnnormalized weights statistics:")
println("  Mean: ", round(mean(unnorm_weights), digits=3))
println("  Max: ", round(maximum(unnorm_weights), digits=3))

# Monte Carlo estimate
log_ml_is = ll_ph_marginal + log(mean(unnorm_weights))
println("\nMonte Carlo log-likelihood estimate:")
println("  log r(Y|θ') + log(mean(ν)): ", round(log_ml_is, digits=3))
println("  = ", round(ll_ph_marginal, digits=3), " + ", round(log(mean(unnorm_weights)), digits=3))

# For comparison, what does Markov give?
surrogate_fitted = MultistateModels.fit_surrogate(model; verbose=false)
println("\nMarkov surrogate log-likelihood: ", round(surrogate_fitted.loglik.loglik, digits=3))
