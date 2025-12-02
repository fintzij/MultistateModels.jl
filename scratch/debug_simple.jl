using MultistateModels
using Random
using DataFrames
using Statistics

Random.seed!(12345)

# ============================================================================
# SIMPLE TEST: Single subject, 2-state model with exact observations
# This lets us manually verify the importance sampling formula
# ============================================================================

println("="^70)
println("Simple 2-state test with Weibull hazards")
println("="^70)

# Single transition 1 -> 2 at exact time t=1.0
h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)

dat = DataFrame(
    id = [1],
    tstart = [0.0],
    tstop = [1.0],
    statefrom = [1],
    stateto = [2],
    obstype = [1]  # Exact observation
)

model = multistatemodel(h12; data=dat)

println("\nModel initial parameters:")
println("  log_λ = ", model.parameters[1][1])
println("  log_α = ", model.parameters[1][2])

# Build phase-type infrastructure
tmat = model.tmat
phasetype_config = MultistateModels.PhaseTypeConfig(n_phases=[2, 1])
phasetype_surrogate = MultistateModels.build_phasetype_surrogate(tmat, phasetype_config)

println("\nPhase-type surrogate:")
println("  State 1 has phases: ", phasetype_surrogate.state_to_phases[1])
println("  State 2 has phases: ", phasetype_surrogate.state_to_phases[2])
println("  Expanded Q matrix:\n", round.(phasetype_surrogate.expanded_Q, digits=3))

emat_ph = MultistateModels.build_phasetype_emat_expanded(model, phasetype_surrogate)

# Compute phase-type marginal likelihood
ll_ph_marginal = MultistateModels.compute_phasetype_marginal_loglik(model, phasetype_surrogate, emat_ph)
println("\nPhase-type marginal log-likelihood r(Y|θ'): ", round(ll_ph_marginal, digits=4))

# Build sampling infrastructure
books = MultistateModels.build_tpm_mapping(model.data)
absorbingstates = findall([isa(h, MultistateModels._TotalHazardAbsorbing) for h in model.totalhazards])
tpm_book_ph, hazmat_book_ph = MultistateModels.build_phasetype_tpm_book(phasetype_surrogate, books, model.data)
fbmats_ph = MultistateModels.build_fbmats_phasetype(model, phasetype_surrogate)

println("\n" * "="^70)
println("Sampling paths and computing importance weights")
println("="^70)

n_paths = 5
for p in 1:n_paths
    println("\n--- Path $p ---")
    
    path_result = MultistateModels.draw_samplepath_phasetype(1, model, tpm_book_ph, hazmat_book_ph, 
                                                             books[2], fbmats_ph, emat_ph, 
                                                             phasetype_surrogate, absorbingstates)
    
    println("Collapsed path:")
    println("  times: ", path_result.collapsed.times)
    println("  states: ", path_result.collapsed.states)
    
    println("Expanded path:")
    println("  times: ", path_result.expanded.times)
    println("  states: ", path_result.expanded.states)
    
    # Compute likelihoods
    loglik_target = MultistateModels.loglik(model.parameters, path_result.collapsed, model.hazards, model)
    loglik_surrog = MultistateModels.loglik_phasetype_expanded(path_result.expanded, phasetype_surrogate)
    log_weight = loglik_target - loglik_surrog
    
    println("Log-likelihoods:")
    println("  f(Z|θ) target:     ", round(loglik_target, digits=4))
    println("  h(Z|θ') surrogate: ", round(loglik_surrog, digits=4))
    println("  log weight:        ", round(log_weight, digits=4))
    
    # Manual calculation of surrogate density for verification
    Q = phasetype_surrogate.expanded_Q
    println("\nManual verification of h(Z|θ'):")
    for i in 1:length(path_result.expanded.times)-1
        t0 = path_result.expanded.times[i]
        t1 = path_result.expanded.times[i + 1]
        dt = t1 - t0
        s = path_result.expanded.states[i]
        d = path_result.expanded.states[i + 1]
        q_s = -Q[s, s]
        q_sd = s != d ? Q[s, d] : 0.0
        surv_term = -q_s * dt
        trans_term = s != d ? log(q_sd) : 0.0
        println("  Transition $i: s=$s->d=$d, dt=$(round(dt, digits=3)), q_s=$(round(q_s, digits=3)), q_sd=$(round(q_sd, digits=3))")
        println("    survival: $(round(surv_term, digits=4)), transition: $(round(trans_term, digits=4))")
    end
end

# Now test the full importance sampling estimate
println("\n" * "="^70)
println("Full importance sampling estimate")
println("="^70)

n_paths_full = 1000
log_weights = Float64[]

for p in 1:n_paths_full
    path_result = MultistateModels.draw_samplepath_phasetype(1, model, tpm_book_ph, hazmat_book_ph, 
                                                             books[2], fbmats_ph, emat_ph, 
                                                             phasetype_surrogate, absorbingstates)
    loglik_target = MultistateModels.loglik(model.parameters, path_result.collapsed, model.hazards, model)
    loglik_surrog = MultistateModels.loglik_phasetype_expanded(path_result.expanded, phasetype_surrogate)
    push!(log_weights, loglik_target - loglik_surrog)
end

println("\nLog weight statistics (n=$n_paths_full):")
println("  Mean:  ", round(mean(log_weights), digits=4))
println("  Std:   ", round(std(log_weights), digits=4))
println("  Min:   ", round(minimum(log_weights), digits=4))
println("  Max:   ", round(maximum(log_weights), digits=4))

# IS estimate
unnorm_weights = exp.(log_weights)
log_ml_is = ll_ph_marginal + log(mean(unnorm_weights))

println("\nImportance sampling estimate:")
println("  r(Y|θ') = $(round(ll_ph_marginal, digits=4))")
println("  log(mean(exp(log_weights))) = $(round(log(mean(unnorm_weights)), digits=4))")
println("  log f̂(Y|θ) = $(round(log_ml_is, digits=4))")

# Compare to Markov
println("\n" * "="^70)
println("Comparison to Markov proposal")
println("="^70)

surrogate_fitted = MultistateModels.fit_surrogate(model; verbose=false)
println("Markov surrogate log-likelihood: $(round(surrogate_fitted.loglik.loglik, digits=4))")

# Analytical target (for 2-state Weibull)
# f(t) = α * λ * t^(α-1) * exp(-λ * t^α)
# At default params: log_λ ≈ 0, log_α ≈ 0 means λ=1, α=1 (exponential)
# f(1) = exp(-1) ≈ 0.368, log f(1) ≈ -1.0
α = exp(model.parameters[1][2])
λ = exp(model.parameters[1][1])
t = 1.0
log_f_analytical = log(α) + log(λ) + (α-1)*log(t) - λ*t^α
println("Analytical target log-likelihood: $(round(log_f_analytical, digits=4))")
