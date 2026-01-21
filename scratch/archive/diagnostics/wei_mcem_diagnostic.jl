# Targeted diagnostic for Weibull MCEM h23 bias
# 
# INSIGHT: Phase-type INFERENCE works, so the bug must be in the SURROGATE
# calibration or usage, not in the phase-type likelihood computation itself.

using MultistateModels
using DataFrames
using Random
using Statistics
using Printf
using LinearAlgebra

import MultistateModels: 
    Hazard, multistatemodel, fit, set_parameters!, simulate,
    get_parameters_flat, get_parameters, PhaseTypeProposal,
    SamplePath, @formula, expand_data_for_phasetype,
    fit_surrogate, PhaseTypeSurrogate, MarkovSurrogate,
    build_tpm_mapping, build_hazmat_book, build_tpm_book,
    build_phasetype_tpm_book, compute_expanded_subject_indices,
    draw_samplepath_phasetype, collapse_phasetype_path,
    _compute_path_loglik_fused, loglik_phasetype_collapsed_path,
    loglik, unflatten_parameters

println("="^70)
println("PhaseType Surrogate Diagnostic")
println("="^70)

# True parameters
const TRUE_SHAPE_12 = 1.3
const TRUE_SCALE_12 = 0.15
const TRUE_SHAPE_23 = 1.1
const TRUE_SCALE_23 = 0.12

# ============================================================================
# STEP 1: Create minimal test case 
# ============================================================================
println("\n1. Creating minimal test case...")

# Single subject with:
# - Panel observation: state 1 → 2 in [0, 4]  
# - Exact observation: state 2 → 3 at t=6

test_data = DataFrame(
    id = [1, 1],
    tstart = [0.0, 4.0],
    tstop = [4.0, 6.0],
    statefrom = [1, 2],
    stateto = [2, 3],
    obstype = [2, 1]  # Panel, Exact
)

println("   Original data:")
println(test_data)

h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
h23 = Hazard(@formula(0 ~ 1), "wei", 2, 3)

model = multistatemodel(h12, h23; data=test_data, surrogate=:markov)
set_parameters!(model, (h12 = [TRUE_SHAPE_12, TRUE_SCALE_12], 
                        h23 = [TRUE_SHAPE_23, TRUE_SCALE_23]))

# ============================================================================
# STEP 2: Examine the surrogate calibration
# ============================================================================
println("\n2. Examining surrogate calibration...")

markov_surrogate = fit_surrogate(model; type=:markov, method=:heuristic, verbose=false)
phasetype_surrogate = fit_surrogate(model; type=:phasetype, n_phases=3, method=:heuristic, verbose=false)

println("\n   Markov surrogate rates (exponential means):")
for h in markov_surrogate.hazards
    rate = markov_surrogate.parameters.nested[h.hazname].baseline[Symbol("$(h.hazname)_rate")]
    println("   $(h.hazname): rate = $(@sprintf("%.4f", rate)), mean = $(@sprintf("%.4f", 1/rate))")
end

println("\n   Phase-type surrogate expanded Q matrix:")
Q_pt = phasetype_surrogate.expanded_Q
display(round.(Q_pt, digits=4))

println("\n   Phase-to-state mapping:")
println("   ", phasetype_surrogate.phase_to_state)

# ============================================================================
# STEP 3: Compare sampling Q to likelihood Q
# ============================================================================
println("\n3. Comparing sampling Q vs likelihood Q...")

# Build the tpm_book_ph and hazmat_book_ph used for sampling
import MultistateModels: build_tpm_mapping

books = build_tpm_mapping(model.data)
tpm_book_ph, hazmat_book_ph = build_phasetype_tpm_book(phasetype_surrogate, markov_surrogate, books, model.data)

println("\n   Q used for SAMPLING (hazmat_book_ph[1]):")
display(round.(hazmat_book_ph[1], digits=4))

println("\n   Q used for LIKELIHOOD (surrogate.expanded_Q):")
display(round.(phasetype_surrogate.expanded_Q, digits=4))

println("\n   Are they equal? ", hazmat_book_ph[1] ≈ phasetype_surrogate.expanded_Q)

if !(hazmat_book_ph[1] ≈ phasetype_surrogate.expanded_Q)
    println("\n   *** MISMATCH DETECTED! ***")
    println("   Difference:")
    display(round.(hazmat_book_ph[1] - phasetype_surrogate.expanded_Q, digits=6))
end

# ============================================================================
# STEP 4: Compute sojourn time densities from both Q matrices
# ============================================================================
println("\n4. Sojourn time density comparison for state 2...")

# Extract sub-intensity matrix for state 2 (phases 4-6)
state2_phases = phasetype_surrogate.state_to_phases[2]
n_ph = length(state2_phases)

S_sampling = zeros(n_ph, n_ph)
S_likelihood = zeros(n_ph, n_ph)
exit_rate_sampling = zeros(n_ph)
exit_rate_likelihood = zeros(n_ph)

for (ii, pi) in enumerate(state2_phases)
    for (jj, pj) in enumerate(state2_phases)
        S_sampling[ii, jj] = hazmat_book_ph[1][pi, pj]
        S_likelihood[ii, jj] = phasetype_surrogate.expanded_Q[pi, pj]
    end
    # Exit rate to state 3 (phase 7)
    exit_rate_sampling[ii] = hazmat_book_ph[1][pi, 7]
    exit_rate_likelihood[ii] = phasetype_surrogate.expanded_Q[pi, 7]
end

println("\n   S_sampling (within-state rates for state 2):")
display(round.(S_sampling, digits=4))

println("\n   S_likelihood:")
display(round.(S_likelihood, digits=4))

println("\n   Exit rates to state 3:")
println("   Sampling:   ", round.(exit_rate_sampling, digits=4))
println("   Likelihood: ", round.(exit_rate_likelihood, digits=4))

# Compute density at several sojourn times
# f(τ; 2→3) = π' * exp(S*τ) * r where π = (1,0,0), r = exit rates
println("\n   Phase-type density of h23 sojourn time:")
println("   " * "-"^50)
println("   τ       f_sampling      f_likelihood    ratio")
println("   " * "-"^50)

for τ in [1.0, 2.0, 3.0, 4.0, 5.0, 5.5]
    expS_samp = exp(S_sampling * τ)
    expS_lik = exp(S_likelihood * τ)
    
    # Density = first row of exp(S*τ) dotted with exit rates
    f_samp = dot(expS_samp[1, :], exit_rate_sampling)
    f_lik = dot(expS_lik[1, :], exit_rate_likelihood)
    
    ratio = f_samp / f_lik
    println("   $(@sprintf("%.1f", τ))     $(@sprintf("%.6f", f_samp))    $(@sprintf("%.6f", f_lik))    $(@sprintf("%.4f", ratio))")
end
println("   " * "-"^50)

# ============================================================================
# STEP 5: Direct comparison on actual paths
# ============================================================================
println("\n5. Direct likelihood comparison on example paths...")

example_paths = [
    SamplePath(1, [0.0, 0.5, 6.0], [1, 2, 3]),  # Early 1→2 at t=0.5
    SamplePath(1, [0.0, 2.0, 6.0], [1, 2, 3]),  # Middle 1→2 at t=2.0
    SamplePath(1, [0.0, 3.5, 6.0], [1, 2, 3]),  # Late 1→2 at t=3.5
]

target_pars = unflatten_parameters(get_parameters_flat(model), model)

println("   " * "-"^80)
println("   Path                    τ_1    τ_2    Target LL    Surrog LL    IW")
println("   " * "-"^80)

for path in example_paths
    τ_1 = path.times[2] - path.times[1]  # Time in state 1
    τ_2 = path.times[3] - path.times[2]  # Time in state 2
    
    target_ll = loglik(target_pars, path, model.hazards, model)
    surrogate_ll = loglik_phasetype_collapsed_path(path, phasetype_surrogate)
    iw = target_ll - surrogate_ll
    
    desc = "1→2 at t=$(path.times[2])"
    println("   $(rpad(desc, 25)) $(@sprintf("%.1f", τ_1))    $(@sprintf("%.1f", τ_2))    $(@sprintf("%.4f", target_ll))    $(@sprintf("%.4f", surrogate_ll))    $(@sprintf("%.4f", iw))")
end
println("   " * "-"^80)

# ============================================================================
# STEP 6: What SHOULD the likelihood be using the sampling Q?
# ============================================================================
println("\n6. Computing surrogate likelihood USING SAMPLING Q matrix...")

function loglik_with_Q(path::SamplePath, Q::Matrix{Float64}, state_to_phases::Vector{UnitRange{Int}})
    loglik = 0.0
    n_transitions = length(path.times) - 1
    
    for i in 1:n_transitions
        t0 = path.times[i]
        t1 = path.times[i + 1]
        τ = t1 - t0
        
        s_obs = path.states[i]
        d_obs = path.states[i + 1]
        
        s_phases = state_to_phases[s_obs]
        d_phases = state_to_phases[d_obs]
        n_phases_s = length(s_phases)
        
        # Extract sub-intensity for state s
        S_within = zeros(Float64, n_phases_s, n_phases_s)
        for (ii, pi) in enumerate(s_phases)
            for (jj, pj) in enumerate(s_phases)
                S_within[ii, jj] = Q[pi, pj]
            end
        end
        
        # Exit rates to destination state d
        exit_to_dest = zeros(Float64, n_phases_s)
        for (ii, pi) in enumerate(s_phases)
            exit_to_dest[ii] = sum(Q[pi, dp] for dp in d_phases)
        end
        
        # Density: π' * exp(S*τ) * r
        expSτ = exp(S_within * τ)
        density = dot(expSτ[1, :], exit_to_dest)
        
        if density <= 0
            return -Inf
        end
        
        loglik += log(density)
    end
    
    return loglik
end

println("   " * "-"^90)
println("   Path                    Surrog LL (stored Q)    Surrog LL (sampling Q)    Diff")
println("   " * "-"^90)

for path in example_paths
    ll_stored = loglik_phasetype_collapsed_path(path, phasetype_surrogate)
    ll_sampling = loglik_with_Q(path, hazmat_book_ph[1], phasetype_surrogate.state_to_phases)
    diff = ll_stored - ll_sampling
    
    desc = "1→2 at t=$(path.times[2])"
    println("   $(rpad(desc, 25)) $(@sprintf("%.6f", ll_stored))               $(@sprintf("%.6f", ll_sampling))               $(@sprintf("%.6f", diff))")
end
println("   " * "-"^90)

println("\n" * "="^70)
println("DIAGNOSIS:")
println("="^70)
println("If 'Surrog LL (stored Q)' ≠ 'Surrog LL (sampling Q)', then the bug is that")
println("paths are sampled from one Q but likelihood computed with different Q.")
println("="^70)

# ============================================================================
# STEP 7: Verify phase-type likelihood formula
# ============================================================================
println("\n7. Verifying phase-type likelihood formula...")

# For a 3-phase Coxian, the density of sojourn τ followed by exit is:
# f(τ) = π' exp(S τ) r
# where π = [1, 0, 0], S is the sub-intensity, r is exit rates

# Let's verify manually for state 2 (phases 4-6)
state2_phases = phasetype_surrogate.state_to_phases[2]
Q = phasetype_surrogate.expanded_Q

# Sub-intensity for state 2
S2 = Q[state2_phases, state2_phases]
println("   S (sub-intensity for state 2):")
display(round.(S2, digits=4))

# Exit rates from state 2 to state 3 (phase 7)
r_to_3 = [Q[p, 7] for p in state2_phases]
println("\n   Exit rates to state 3: ", round.(r_to_3, digits=4))

# Now verify the density at τ=4 (the sojourn in state 2 for path "1→2 at t=2")
τ = 4.0
expSτ = exp(S2 * τ)
manual_density_23 = dot(expSτ[1, :], r_to_3)
println("\n   Manual calculation for τ_2 = $τ:")
println("   exp(S*τ)[1,:] = ", round.(expSτ[1, :], digits=6))
println("   Density f(τ; 2→3) = ", round(manual_density_23, digits=6))
println("   Log density = ", round(log(manual_density_23), digits=4))

# Now do the same for state 1 (phases 1-3)
state1_phases = phasetype_surrogate.state_to_phases[1]
S1 = Q[state1_phases, state1_phases]
println("\n   S (sub-intensity for state 1):")
display(round.(S1, digits=4))

# Exit rates from state 1 to state 2 (phase 4)
r_to_2 = [Q[p, 4] for p in state1_phases]
println("\n   Exit rates to state 2: ", round.(r_to_2, digits=4))

# Verify for τ_1=2 (the sojourn in state 1 for path "1→2 at t=2")
τ1 = 2.0
expSτ1 = exp(S1 * τ1)
manual_density_12 = dot(expSτ1[1, :], r_to_2)
println("\n   Manual calculation for τ_1 = $τ1:")
println("   exp(S*τ)[1,:] = ", round.(expSτ1[1, :], digits=6))
println("   Density f(τ; 1→2) = ", round(manual_density_12, digits=6))
println("   Log density = ", round(log(manual_density_12), digits=4))

# Total manual log-lik for path 1→2 at t=2, 2→3 at t=6
manual_total = log(manual_density_12) + log(manual_density_23)
println("\n   Total manual log-lik: ", round(manual_total, digits=4))

# Compare with loglik_phasetype_collapsed_path
test_path = SamplePath(1, [0.0, 2.0, 6.0], [1, 2, 3])
auto_ll = loglik_phasetype_collapsed_path(test_path, phasetype_surrogate)
println("   loglik_phasetype_collapsed_path: ", round(auto_ll, digits=4))
println("   Match? ", isapprox(manual_total, auto_ll, rtol=1e-6))

# ============================================================================
# STEP 8: Check if the issue is with how time is counted
# ============================================================================
println("\n8. Checking time conventions...")

# The path times are [0, 2, 6], meaning:
# - At t=0, in state 1
# - At t=2, transition to state 2
# - At t=6, transition to state 3

# The sojourn in state 2 starts at t=2 and ends at t=6, so duration = 4
# But for a semi-Markov model with clock reset, the hazard h23 is evaluated
# at times starting from 0 (since clock resets on entry to state 2)

# For the Weibull target, h23 density for sojourn τ=4:
κ23, λ23 = TRUE_SHAPE_23, TRUE_SCALE_23
wei_h_at_4 = κ23 * λ23 * (λ23 * 4.0)^(κ23 - 1)
wei_H_0_4 = (λ23 * 4.0)^κ23
wei_f_4 = wei_h_at_4 * exp(-wei_H_0_4)
println("   Weibull h23 density at τ=4:")
println("   h(4) = ", round(wei_h_at_4, digits=6))
println("   H(0,4) = ", round(wei_H_0_4, digits=6))  
println("   f(4) = h(4)*S(4) = ", round(wei_f_4, digits=6))
println("   log f(4) = ", round(log(wei_f_4), digits=4))

# For h12 density at τ=2:
κ12, λ12 = TRUE_SHAPE_12, TRUE_SCALE_12
wei_h12_at_2 = κ12 * λ12 * (λ12 * 2.0)^(κ12 - 1)
wei_H12_0_2 = (λ12 * 2.0)^κ12
wei_f12_2 = wei_h12_at_2 * exp(-wei_H12_0_2)
println("\n   Weibull h12 density at τ=2:")
println("   h(2) = ", round(wei_h12_at_2, digits=6))
println("   H(0,2) = ", round(wei_H12_0_2, digits=6))  
println("   f(2) = h(2)*S(2) = ", round(wei_f12_2, digits=6))
println("   log f(2) = ", round(log(wei_f12_2), digits=4))

wei_total = log(wei_f12_2) + log(wei_f_4)
println("\n   Total Weibull log-lik (manual): ", round(wei_total, digits=4))

# Compare with what loglik() returns
auto_wei_ll = loglik(target_pars, test_path, model.hazards, model)
println("   loglik() for Weibull: ", round(auto_wei_ll, digits=4))
println("   Match? ", isapprox(wei_total, auto_wei_ll, rtol=1e-2))

# ============================================================================
# STEP 9: The ratio is what matters for bias
# ============================================================================
println("\n9. Summary of likelihood ratio...")
println("   Target (Weibull) LL: ", round(auto_wei_ll, digits=4))
println("   Surrogate (PT) LL:   ", round(auto_ll, digits=4))
println("   Importance weight:   ", round(auto_wei_ll - auto_ll, digits=4))
println("")

# ============================================================================
# STEP 10: Investigate the discrepancy in Weibull LL
# ============================================================================
println("\n10. Investigating Weibull likelihood discrepancy...")

# The path is 1→2 at t=2, 2→3 at t=6
# Clock resets on entry to each state

# For h12: sojourn τ=2 in state 1
# For h23: sojourn τ=4 in state 2

# But wait - are there competing risks?
# In state 1, the only exit is to state 2 (h12)
# In state 2, the only exit is to state 3 (h23)
# So no competing risks correction needed

# Let me trace through what _compute_path_loglik_fused actually does
println("   Path transitions:")
for i in 1:(length(test_path.states)-1)
    println("   $i: $(test_path.states[i]) → $(test_path.states[i+1]) at t=$(test_path.times[i+1]), sojourn=$(test_path.times[i+1]-test_path.times[i])")
end

# Now let's manually compute using the same logic as the code
println("\n   Manual computation using code logic:")

# For segment 1: state 1 (t=0 to t=2)
# - Cumulative hazard H12(0, 2)
# - Transition hazard h12(2)
statefrom1, stateto1 = 1, 2
lb1, ub1 = 0.0, 2.0

# h12 Weibull: h(t) = κλ(λt)^(κ-1), H(t) = (λt)^κ
κ12, λ12 = TRUE_SHAPE_12, TRUE_SCALE_12
H12_0_2 = (λ12 * ub1)^κ12 - (λ12 * lb1)^κ12
h12_at_2 = κ12 * λ12 * (λ12 * ub1)^(κ12 - 1)

println("   Segment 1: state $statefrom1 → $stateto1, [$(lb1), $(ub1)]")
println("   H12(0,2) = $(round(H12_0_2, digits=4))")
println("   h12(2) = $(round(h12_at_2, digits=4))")
println("   log(h12(2)) = $(round(log(h12_at_2), digits=4))")

# For segment 2: state 2 (t=2 to t=6, but clock resets so sojourn is 0 to 4)
# Wait - does the code use absolute time or sojourn time?
statefrom2, stateto2 = 2, 3

# Let me check what the code path tracking does...
# The path.times are [0, 2, 6]
# For the i=2 transition (index 2), path.times[2]=2, path.times[3]=6
# So increment = 6 - 2 = 4
# But is lb = 0 (sojourn time) or lb = 2 (clock time)?

println("\n   Checking time conventions in _compute_path_loglik_fused...")

# Looking at the code:
# sojourn = 0.0 (initialized)
# for i in 1:n_transitions
#     increment = path.times[i+1] - path.times[i]
#     lb = sojourn
#     ub = sojourn + increment

# So for i=1: lb=0, ub=2, then sojourn stays 0 (no reset shown?)
# Wait, there's no clock reset in that snippet I saw earlier
# Let me check what happens after a transition

println("\n   Let me check if the code resets the clock...")
# Looking more carefully at the code logic:
# The key question is: does sojourn get reset to 0 after a transition?

# From the code I read earlier, it looks like sojourn is NOT explicitly reset.
# But wait, the code has:
#   if lb == 0.0
#       fill!(effective_times, zero(T))
# This only happens when lb == 0, not when entering a new state.

# Actually, I think I misread. Let me trace through more carefully.
# For i=1: increment = 2, lb = 0, ub = 2
# For i=2: increment = 4, lb = 2, ub = 6

# Hmm, so it seems like the code is using ABSOLUTE time, not sojourn time!
# But that doesn't match the description of "clock reset" for semi-Markov models.

# Let me verify by computing the "wrong" way (absolute time):
println("\n   If using ABSOLUTE time (no clock reset):")
H12_0_2_abs = (λ12 * 2.0)^κ12
h12_at_2_abs = κ12 * λ12 * (λ12 * 2.0)^(κ12 - 1)
ll_seg1_abs = log(h12_at_2_abs) - H12_0_2_abs
println("   Segment 1: log(h12(2)) - H12(0,2) = $(round(ll_seg1_abs, digits=4))")

κ23, λ23 = TRUE_SHAPE_23, TRUE_SCALE_23
H23_2_6_abs = (λ23 * 6.0)^κ23 - (λ23 * 2.0)^κ23  # H23(2,6) with absolute times
h23_at_6_abs = κ23 * λ23 * (λ23 * 6.0)^(κ23 - 1)
ll_seg2_abs = log(h23_at_6_abs) - H23_2_6_abs
println("   Segment 2: log(h23(6)) - H23(2,6) = $(round(ll_seg2_abs, digits=4))")

ll_total_abs = ll_seg1_abs + ll_seg2_abs
println("   Total (absolute time): $(round(ll_total_abs, digits=4))")

println("\n   If using SOJOURN time (with clock reset):")
H12_0_2_soj = (λ12 * 2.0)^κ12  # H12(0, 2)
h12_at_2_soj = κ12 * λ12 * (λ12 * 2.0)^(κ12 - 1)
ll_seg1_soj = log(h12_at_2_soj) - H12_0_2_soj
println("   Segment 1: log(h12(2)) - H12(0,2) = $(round(ll_seg1_soj, digits=4))")

H23_0_4_soj = (λ23 * 4.0)^κ23  # H23(0, 4) with clock reset
h23_at_4_soj = κ23 * λ23 * (λ23 * 4.0)^(κ23 - 1)
ll_seg2_soj = log(h23_at_4_soj) - H23_0_4_soj
println("   Segment 2: log(h23(4)) - H23(0,4) = $(round(ll_seg2_soj, digits=4))")

ll_total_soj = ll_seg1_soj + ll_seg2_soj
println("   Total (sojourn time): $(round(ll_total_soj, digits=4))")

println("\n   Actual loglik() result: $(round(auto_wei_ll, digits=4))")
println("   Matches absolute time? $(isapprox(ll_total_abs, auto_wei_ll, rtol=0.01))")
println("   Matches sojourn time? $(isapprox(ll_total_soj, auto_wei_ll, rtol=0.01))")

# ============================================================================
# STEP 11: Check the actual Weibull parameterization used by the code
# ============================================================================
println("\n11. Checking Weibull parameterization...")

println("   The code uses: h(t) = shape * scale * t^(shape-1)")
println("   The code uses: H(lb, ub) = scale * (ub^shape - lb^shape)")
println("")
println("   This is DIFFERENT from the standard Weibull parameterization!")
println("   Standard: h(t) = κλ(λt)^(κ-1), H(t) = (λt)^κ")
println("   Code:     h(t) = κλ t^(κ-1),    H(t) = λ t^κ")
println("")

# Recalculate with the code's parameterization
κ12, λ12 = TRUE_SHAPE_12, TRUE_SCALE_12  # shape=1.3, scale=0.15
κ23, λ23 = TRUE_SHAPE_23, TRUE_SCALE_23  # shape=1.1, scale=0.12

println("   Recalculating with code's parameterization:")
println("")

# h12: sojourn τ=2 in state 1
H12_code = λ12 * (2.0^κ12 - 0.0^κ12)
h12_code = κ12 * λ12 * 2.0^(κ12 - 1)
ll_h12_code = log(h12_code) - H12_code
println("   h12 at τ=2:")
println("   H12(0,2) = $λ12 * (2^$κ12 - 0^$κ12) = $(round(H12_code, digits=4))")
println("   h12(2) = $κ12 * $λ12 * 2^$(κ12-1) = $(round(h12_code, digits=4))")
println("   log(h12) - H12 = $(round(ll_h12_code, digits=4))")

# h23: sojourn τ=4 in state 2
H23_code = λ23 * (4.0^κ23 - 0.0^κ23)
h23_code = κ23 * λ23 * 4.0^(κ23 - 1)
ll_h23_code = log(h23_code) - H23_code
println("\n   h23 at τ=4:")
println("   H23(0,4) = $λ23 * (4^$κ23 - 0^$κ23) = $(round(H23_code, digits=4))")
println("   h23(4) = $κ23 * $λ23 * 4^$(κ23-1) = $(round(h23_code, digits=4))")
println("   log(h23) - H23 = $(round(ll_h23_code, digits=4))")

ll_total_code = ll_h12_code + ll_h23_code
println("\n   Total log-lik (code parameterization): $(round(ll_total_code, digits=4))")
println("   Actual loglik() result: $(round(auto_wei_ll, digits=4))")
println("   Match? $(isapprox(ll_total_code, auto_wei_ll, rtol=0.01))")

# ============================================================================
# STEP 12: Final analysis - compare densities at multiple sojourn times
# ============================================================================
println("\n12. FINAL ANALYSIS: Comparing Weibull vs Phase-Type densities...")

println("\n   For h23 (2→3 transition), comparing sojourn time densities:")
println("   " * "-"^70)
println("   τ       Weibull f(τ)    PT f(τ)      log(Weibull/PT)  Interpretation")
println("   " * "-"^70)

κ23, λ23 = TRUE_SHAPE_23, TRUE_SCALE_23

for τ in [1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 5.5]
    # Weibull density (code parameterization)
    h_wei = κ23 * λ23 * τ^(κ23 - 1)
    H_wei = λ23 * τ^κ23
    f_wei = h_wei * exp(-H_wei)
    
    # Phase-type density
    expSτ = exp(S_likelihood * τ)
    f_pt = dot(expSτ[1, :], exit_rate_likelihood)
    
    log_ratio = log(f_wei) - log(f_pt)
    interp = log_ratio > 0 ? "Wei > PT (upweight)" : "Wei < PT (downweight)"
    
    println("   $(@sprintf("%.1f", τ))     $(@sprintf("%.6f", f_wei))    $(@sprintf("%.6f", f_pt))    $(@sprintf("%+.4f", log_ratio))         $interp")
end
println("   " * "-"^70)

println("\n   KEY INSIGHT:")
println("   The importance weights favor LONGER sojourns because:")
println("   - At short τ (e.g., 1-2): Phase-type > Weibull → paths downweighted")
println("   - At long τ (e.g., 5+): Weibull > Phase-type → paths upweighted")
println("")
println("   The phase-type surrogate has a DIFFERENT shape than the Weibull:")
println("   - Phase-type has a mode near 0 (like exponential)")
println("   - Weibull with shape 1.1 has a mode > 0 (increasing hazard)")
println("")
println("   This causes systematic bias in importance sampling!")

# ============================================================================
# STEP 13: Compute the actual effect on expected sojourn
# ============================================================================
println("\n13. Expected sojourn under different distributions...")

# True Weibull expected sojourn for h23
# E[T] = Γ(1 + 1/κ) / λ^(1/κ) for standard Weibull
# But for code parameterization h(t) = κλt^(κ-1), we need to derive:
# H(t) = λt^κ, so S(t) = exp(-λt^κ)
# Mean = ∫₀^∞ S(t) dt = λ^(-1/κ) Γ(1 + 1/κ)
using SpecialFunctions
wei_mean = (1/λ23)^(1/κ23) * gamma(1 + 1/κ23)
println("   Weibull h23 mean sojourn (code param): $(round(wei_mean, digits=4))")

# Phase-type expected sojourn: E[T] = π' (-S)^{-1} 1
S_inv = -inv(S_likelihood)
pt_mean = sum(S_inv[1, :])  # Since π = [1, 0, 0]
println("   Phase-type h23 mean sojourn: $(round(pt_mean, digits=4))")

println("\n   The Markov surrogate rate was:")
println("   h23_rate = 0.5, meaning exp mean = 2.0")
println("")
println("   But the Weibull with shape=1.1, scale=0.12 has mean ≈ $(round(wei_mean, digits=2))")
println("   The phase-type was calibrated to the MARKOV surrogate (mean=2), not the Weibull!")
println("")
println("   This is NOT a bug in importance sampling - it's working correctly.")
println("   The bias in h23 estimates must come from somewhere else!")

