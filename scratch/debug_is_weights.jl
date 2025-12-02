# Debug script for IS weight mismatch
using MultistateModels
using Random
using DataFrames

Random.seed!(12345)

# Simple 2-state model with exponential hazards (Markov)
h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)

# Panel data
dat = DataFrame(
    id = [1, 1, 1],
    tstart = [0.0, 1.0, 2.0],
    tstop = [1.0, 2.0, 3.0],
    statefrom = [1, 1, 1],
    stateto = [1, 1, 2],
    obstype = [2, 2, 2]
)

model = multistatemodel(h12; data=dat)

# Set known parameters
MultistateModels.set_parameters!(model, [[-0.5]])
rate = exp(-0.5)
println("Rate: $rate")

# Build phase-type with 1 phase per state
tmat = model.tmat
phasetype_config = MultistateModels.PhaseTypeConfig(n_phases=[1, 1])
surrogate = MultistateModels.build_phasetype_surrogate(tmat, phasetype_config)

println("\nPhase-type surrogate Q matrix:")
display(surrogate.expanded_Q)

println("\nModel parameters:")
println(model.parameters)

# Build infrastructure
emat_ph = MultistateModels.build_phasetype_emat_expanded(model, surrogate)
books = MultistateModels.build_tpm_mapping(model.data)
absorbingstates = findall([isa(h, MultistateModels._TotalHazardAbsorbing) for h in model.totalhazards])
tpm_book_ph, hazmat_book_ph = MultistateModels.build_phasetype_tpm_book(surrogate, books, model.data)
fbmats_ph = MultistateModels.build_fbmats_phasetype(model, surrogate)

# Sample a few paths and compute likelihoods
println("\n" * "="^60)
println("Sampling paths and computing likelihoods")
println("="^60)

for p in 1:5
    path_result = MultistateModels.draw_samplepath_phasetype(
        1, model, tpm_book_ph, hazmat_book_ph, books[2],
        fbmats_ph, emat_ph, surrogate, absorbingstates)
    
    println("\nPath $p:")
    println("  Collapsed: times=$(path_result.collapsed.times), states=$(path_result.collapsed.states)")
    println("  Expanded:  times=$(path_result.expanded.times), states=$(path_result.expanded.states)")
    
    # Target log-likelihood (semi-Markov)
    ll_target = MultistateModels.loglik(model.parameters, path_result.collapsed, model.hazards, model)
    
    # Surrogate log-likelihood (phase-type CTMC)
    ll_surrog = MultistateModels.loglik_phasetype_expanded(path_result.expanded, surrogate)
    
    println("  log f(Z|θ) [target]: $ll_target")
    println("  log h(Z|θ') [surrog]: $ll_surrog")
    println("  log weight: $(ll_target - ll_surrog)")
    
    # Manual calculation for the expanded path
    Q = surrogate.expanded_Q
    ll_manual = 0.0
    times_exp = path_result.expanded.times
    states_exp = path_result.expanded.states
    
    for i in 1:(length(times_exp)-1)
        dt = times_exp[i+1] - times_exp[i]
        s = states_exp[i]
        d = states_exp[i+1]
        q_s = -Q[s, s]
        
        # Survival
        ll_manual += -q_s * dt
        
        # Transition (if state changes)
        if s != d
            ll_manual += log(Q[s, d])
        end
        
        println("    Interval $i: state $s → $d, dt=$dt, -q_s*dt=$(-q_s*dt), log(q_sd)=$(s != d ? log(Q[s,d]) : 0.0)")
    end
    println("  Manual h(Z): $ll_manual")
end
