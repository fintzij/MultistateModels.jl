# Trace through actual path sampling to see collapsed path structure

using MultistateModels
using DataFrames

import MultistateModels: _build_phasetype_from_markov, build_phasetype_tpm_book,
    build_phasetype_emat_expanded, ForwardFiltering!, BackwardSampling_expanded,
    draw_samplepath_phasetype, collapse_phasetype_path, SamplePath,
    build_tpm_mapping, loglik_phasetype_collapsed_path, ProposalConfig

# Simple test: one subject, panel then exact
data = DataFrame(
    id = [1, 1],
    tstart = [0.0, 4.0],
    tstop = [4.0, 6.0],
    statefrom = [1, 2],
    stateto = [2, 3],
    obstype = [2, 1]
)

println("Original data:")
println(data)

h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
h23 = Hazard(@formula(0 ~ 1), "wei", 2, 3)
model = multistatemodel(h12, h23; data=data, surrogate=:markov)

# Set crude rates
MultistateModels.set_crude_init!(model)

# Build phase-type surrogate using the correct API
config = ProposalConfig(n_phases=3)  # 3 phases per transient state
pt_surrogate = _build_phasetype_from_markov(model, model.markovsurrogate; 
                                             config=config, verbose=true)

println("\nPhase mapping:")
println("  State 1 -> phases: ", pt_surrogate.state_to_phases[1])
println("  State 2 -> phases: ", pt_surrogate.state_to_phases[2])
println("  State 3 -> phases: ", pt_surrogate.state_to_phases[3])

# Build infrastructure
books = build_tpm_mapping(model.data)
tpm_book_ph, hazmat_book_ph = build_phasetype_tpm_book(pt_surrogate, model.markovsurrogate, books, model.data)
emat_ph = build_phasetype_emat_expanded(model, pt_surrogate)

# Build tpm_map
n_rows = nrow(model.data)
tpm_map = zeros(Int, n_rows, 2)
for i in 1:n_rows
    tpm_map[i, 1] = 1
    tpm_map[i, 2] = i
end

# Initialize fbmats
n_expanded = pt_surrogate.n_expanded_states
fbmats_ph = [zeros(Float64, 2, n_expanded, n_expanded)]

# Sample paths and examine structure
absorbingstates = [3]
println("\nSampling paths and examining structure...")
for trial in 1:5
    result = draw_samplepath_phasetype(1, model, tpm_book_ph, hazmat_book_ph,
                                        tpm_map, fbmats_ph, emat_ph, pt_surrogate, absorbingstates)
    
    println("\n--- Trial $trial ---")
    println("  Expanded path:")
    println("    times:  ", round.(result.expanded.times, digits=3))
    println("    states: ", result.expanded.states)
    
    println("  Collapsed path:")
    println("    times:  ", round.(result.collapsed.times, digits=3))
    println("    states: ", result.collapsed.states)
    
    # Compute what loglik_phasetype_collapsed_path would compute
    collapsed = result.collapsed
    println("  loglik_phasetype_collapsed_path computes:")
    for i in 1:(length(collapsed.times)-1)
        t0, t1 = collapsed.times[i], collapsed.times[i+1]
        s0, s1 = collapsed.states[i], collapsed.states[i+1]
        tau = t1 - t0
        if s0 == s1
            println("    Segment $i: survival in state $s0 for τ=$(round(tau, digits=3))")
        else
            println("    Segment $i: transition $s0→$s1 with sojourn τ=$(round(tau, digits=3))")
        end
    end
    
    ll = loglik_phasetype_collapsed_path(collapsed, pt_surrogate)
    println("  Total log-likelihood: ", round(ll, digits=4))
end

println("\n" * "="^60)
println("KEY QUESTION:")
println("="^60)
println("Does the collapsed path include the panel observation time t=4?")
println("")
println("For data [0,4] panel, [4,6] exact:")
println("  - We know state 1 at t=0, state 2 at t=4, state 3 at t=6")
println("  - The 1→2 transition happened sometime in (0, 4)")
println("  - The 2→3 transition happened exactly at t=6")
println("")
println("If collapsed path is [0, t_12, 6], [1, 2, 3]:")
println("  - loglik computes f(τ₁=t_12; 1→2) × f(τ₂=6-t_12; 2→3)")
println("  - This ASSUMES sojourn in state 2 is (6-t_12)")
println("")
println("But the ACTUAL sojourn in state 2 should be:")
println("  - From state entry (at t_12) to state exit (at t=6)")
println("  - = 6 - t_12")
println("")
println("So the math is correct IF t_12 is the actual transition time!")
println("="^60)

