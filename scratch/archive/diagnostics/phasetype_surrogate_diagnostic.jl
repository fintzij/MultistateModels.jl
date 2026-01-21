# =============================================================================
# Phase-Type Surrogate Diagnostic Script
# =============================================================================
#
# Diagnostic to isolate why MCEM with PhaseType proposal converges to different
# parameter estimates than Markov proposal.
#
# This script tests whether:
# 1. Fitting a phase-type hazard directly to data gives expected parameters
# 2. Building a phase-type surrogate from a Markov surrogate preserves rates
# 3. The importance weight calculation (loglik_surrog) is correct
#
# Reference: longtest_mcem_splines.jl Test 1 showed:
#   - Markov proposal: hazard ~0.34-0.39 (close to true 0.30)
#   - PhaseType proposal: hazard 0.43 to 1.41 (steeply increasing, WRONG)
#
# Created: 2026-01-16
# =============================================================================

using Test
using MultistateModels
using DataFrames
using Random
using Statistics
using Printf
using LinearAlgebra

# Import internal functions needed for diagnostic
import MultistateModels: 
    Hazard, multistatemodel, fit, set_parameters!, simulate,
    get_parameters_flat, get_parameters, get_hazard_params,
    MarkovSurrogate, PhaseTypeSurrogate, PhaseTypeProposal,
    _build_phasetype_from_markov, build_tpm_mapping,
    build_phasetype_tpm_book, fit_surrogate,
    loglik_phasetype_collapsed_path, loglik,
    build_coxian_intensity, subintensity, absorption_rates,
    SamplePath, cumulative_hazard, @formula,
    draw_samplepath, draw_samplepath_phasetype,
    build_fbmats, build_hazmat_book, build_tpm_book, compute_hazmat!, compute_tmat!,
    unflatten_parameters, build_fbmats_phasetype_with_indices,
    build_phasetype_emat_expanded, collapse_phasetype_path,
    ForwardFiltering!

using ExponentialUtilities

println("="^70)
println("Phase-Type Surrogate Diagnostic")
println("="^70)
println()

# =============================================================================
# Part 1: Generate Simple Panel Data
# =============================================================================

println("Part 1: Generate Panel Data from Exponential(λ=0.3) Hazard")
println("-"^70)

Random.seed!(0xABCD5678)
true_rate = 0.3
n_subjects = 500

# Create exponential model for data generation
h12_exp = Hazard(@formula(0 ~ 1), "exp", 1, 2)

# Panel observation times
obs_times = [0.0, 2.0, 4.0]
nobs = length(obs_times) - 1

# Create data template
sim_data = DataFrame(
    id = repeat(1:n_subjects, inner=nobs),
    tstart = repeat(obs_times[1:end-1], n_subjects),
    tstop = repeat(obs_times[2:end], n_subjects),
    statefrom = ones(Int, n_subjects * nobs),
    stateto = ones(Int, n_subjects * nobs),
    obstype = fill(2, n_subjects * nobs)
)

# Build model and set true rate
model_dgp = multistatemodel(h12_exp; data=sim_data, initialize=false)
set_parameters!(model_dgp, (h12 = [true_rate],))

# Simulate panel data (1→2 absorbing transition is exact by default)
obstype_map = Dict(1 => 1)
sim_result = simulate(model_dgp; paths=false, data=true, nsim=1, autotmax=false,
                     obstype_by_transition=obstype_map)
panel_data = sim_result[1, 1]

# Calculate summary statistics
n_transitioned = sum(panel_data.stateto .== 2 .&& panel_data.statefrom .== 1)
n_total_obs = nrow(panel_data)
println("  True rate: $true_rate")
println("  N subjects: $n_subjects")
println("  N observations: $n_total_obs")
println("  N transitions (1→2): $n_transitioned")
println()

# =============================================================================
# Part 2: Fit Markov Model (Exponential) - Baseline
# =============================================================================

println("Part 2: Fit Markov Model (Exponential Hazard)")
println("-"^70)

h12_markov = Hazard(@formula(0 ~ 1), "exp", 1, 2)
model_markov = multistatemodel(h12_markov; data=panel_data)

fitted_markov = fit(model_markov; verbose=false)
# get_parameters with hazard index returns a vector on :natural scale
params_natural = get_parameters(fitted_markov, 1, scale=:natural)
rate_markov = params_natural[1]  # First (and only) parameter is the rate
println("  Fitted rate (Markov/Exponential): $(@sprintf("%.4f", rate_markov))")
println("  True rate: $true_rate")
println("  Relative error: $(@sprintf("%.1f%%", 100*abs(rate_markov - true_rate)/true_rate))")
println()

# =============================================================================
# Part 3: Build PhaseType Surrogate from Markov MLE
# =============================================================================

println("Part 3: Build PhaseType Surrogate from Markov Surrogate")
println("-"^70)

# Create spline model for MCEM (same as in longtest)
h12_sp = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                degree=1, 
                knots=Float64[],
                boundaryknots=[0.0, 5.0],
                extrapolation="linear")

model_spline = multistatemodel(h12_sp; data=panel_data, surrogate=:markov)

# Fit Markov surrogate explicitly
fit_surrogate(model_spline; type=:markov, method=:mle, verbose=true)
markov_surrogate = model_spline.markovsurrogate

println("\n  Markov Surrogate Rate:")
surrog_rate = markov_surrogate.parameters.nested.h12.baseline.h12_rate
println("    h12 rate: $(@sprintf("%.4f", surrog_rate))")

# Build PhaseType surrogate (same as MCEM internally does)
n_phases = 3  # Match longtest
phasetype_surrogate = _build_phasetype_from_markov(
    model_spline, markov_surrogate;
    config=PhaseTypeProposal(n_phases=n_phases),
    verbose=true
)

# Examine the expanded Q matrix
Q = phasetype_surrogate.expanded_Q
println("\n  Expanded Q matrix ($(size(Q, 1)) × $(size(Q, 2))):")
display(round.(Q, digits=4))
println()

# Verify total exit rate from state 1 phases
state1_phases = phasetype_surrogate.state_to_phases[1]
total_rate_state1 = sum(-Q[first(state1_phases), first(state1_phases)])
println("  Total exit rate from state 1 (phase 1): $(@sprintf("%.4f", total_rate_state1))")
println("  (Should match Markov surrogate rate: $(@sprintf("%.4f", surrog_rate)))")

# Check the mean sojourn time of the phase-type distribution
# For Coxian starting in phase 1: E[T] = π' (-S)⁻¹ 𝟙
# where S is the sub-intensity matrix
n_phases_state1 = length(state1_phases)
S = Q[state1_phases, state1_phases]
println("\n  Sub-intensity matrix S (state 1):")
display(round.(S, digits=4))
pi0 = zeros(n_phases_state1); pi0[1] = 1.0
mean_sojourn = -pi0' * inv(S) * ones(n_phases_state1)
println("\n  Mean sojourn time (phase-type): $(@sprintf("%.4f", mean_sojourn))")
println("  Mean sojourn time (exponential): $(@sprintf("%.4f", 1/surrog_rate))")
println("  Ratio (should be ~1.0): $(@sprintf("%.4f", mean_sojourn * surrog_rate))")
println()

# =============================================================================
# Part 4: Check PhaseType TPM vs Markov TPM
# =============================================================================

println("Part 4: Compare Transition Probability Matrices")
println("-"^70)

# Build TPM books
books = build_tpm_mapping(panel_data)
tpm_book_ph, hazmat_book_ph = build_phasetype_tpm_book(
    phasetype_surrogate, markov_surrogate, books, panel_data
)

println("  Number of unique TPM book entries: $(length(tpm_book_ph))")
println("  Number of unique hazmat book entries: $(length(hazmat_book_ph))")

# Compare collapsed transition probabilities for a standard 2-unit interval
dt = 2.0  # typical interval length

# Markov TPM: P(t) = exp(Q*dt) for simple 1→2 model
# P[1,1] = exp(-λ*dt), P[1,2] = 1 - exp(-λ*dt)
p_markov_11 = exp(-surrog_rate * dt)
p_markov_12 = 1 - p_markov_11
println("\n  For dt = $dt:")
println("    Markov P[1→1]: $(@sprintf("%.4f", p_markov_11))")
println("    Markov P[1→2]: $(@sprintf("%.4f", p_markov_12))")

# PhaseType TPM: compute from expanded Q
P_ph = exp(Matrix(Q * dt))
# Collapse to observed states:
# P_collapsed[1,1] = sum over phase pairs in state 1 weighted by initial phase dist
# For Coxian, we start in phase 1, so:
# P[1→1] = sum_{j ∈ phases(state1)} P[phase1, j]
# P[1→2] = P[phase1, absorbing_state]
state2_phases = phasetype_surrogate.state_to_phases[2]
p_ph_11 = sum(P_ph[first(state1_phases), j] for j in state1_phases)
p_ph_12 = sum(P_ph[first(state1_phases), j] for j in state2_phases)
println("\n    PhaseType P[1→1] (collapsed): $(@sprintf("%.4f", p_ph_11))")
println("    PhaseType P[1→2] (collapsed): $(@sprintf("%.4f", p_ph_12))")
println("    Sum check (should be 1.0): $(@sprintf("%.4f", p_ph_11 + p_ph_12))")

println("\n  Discrepancy in P[1→2]: $(@sprintf("%.1f%%", 100*abs(p_ph_12 - p_markov_12)/p_markov_12))")
println()

# =============================================================================
# Part 5: Test loglik_phasetype_collapsed_path
# =============================================================================

println("Part 5: Test Collapsed Path Log-likelihood")
println("-"^70)

# Create a simple test path: state 1 for time τ, then transition to state 2
test_times = [0.0, 1.5]
test_path = SamplePath(1, test_times, [1, 2])

ll_collapsed = loglik_phasetype_collapsed_path(test_path, phasetype_surrogate)
println("  Test path: state 1 for 1.5 time units, then → state 2")
println("  Log-lik (collapsed): $(@sprintf("%.4f", ll_collapsed))")

# Compare to Markov log-likelihood
# For exponential: log f(τ) = log(λ) - λ*τ
ll_markov = log(surrog_rate) - surrog_rate * 1.5
println("  Log-lik (Markov/exponential): $(@sprintf("%.4f", ll_markov))")
println("  Difference: $(@sprintf("%.4f", ll_collapsed - ll_markov))")

# For a well-calibrated phase-type surrogate, these should be close
# The phase-type approximates the exponential
println()

# =============================================================================
# Part 5b: Compare Target Likelihoods for Paths from Both Proposals
# =============================================================================

println("Part 5b: Path Likelihood Comparison")
println("-"^70)

# We need to compare the log-likelihood of paths under both target and surrogate
# The issue might be that paths sampled from PhaseType have different characteristics

# Build MCEM infrastructure manually to examine paths
import MultistateModels: draw_samplepath, draw_samplepath_phasetype,
    build_fbmats, build_hazmat_book, build_tpm_book, compute_hazmat!, compute_tmat!,
    unflatten_parameters, build_fbmats_phasetype_with_indices,
    build_phasetype_emat_expanded, collapse_phasetype_path

println("\n  Building sampling infrastructure...")

# Use the spline model
h12_test = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                  degree=1, knots=Float64[], boundaryknots=[0.0, 5.0], 
                  extrapolation="linear")
model_test = multistatemodel(h12_test; data=panel_data, surrogate=:markov)
fit_surrogate(model_test; type=:markov, method=:mle, verbose=false)

# Get surrogate
markov_surrog = model_test.markovsurrogate

# Build PhaseType surrogate
pt_surrog = _build_phasetype_from_markov(
    model_test, markov_surrog;
    config=PhaseTypeProposal(n_phases=3),
    verbose=false
)

# Build TPM infrastructure for Markov proposal
books_test = build_tpm_mapping(panel_data)
hazmat_book_mk = build_hazmat_book(Float64, model_test.tmat, books_test[1])
tpm_book_mk = build_tpm_book(Float64, model_test.tmat, books_test[1])
cache = ExponentialUtilities.alloc_mem(similar(hazmat_book_mk[1]), ExpMethodGeneric())

surrogate_pars = get_hazard_params(markov_surrog.parameters, markov_surrog.hazards)
for t in eachindex(books_test[1])
    compute_hazmat!(hazmat_book_mk[t], surrogate_pars, markov_surrog.hazards, books_test[1][t], panel_data)
    compute_tmat!(tpm_book_mk[t], hazmat_book_mk[t], books_test[1][t], cache)
end

# Build PhaseType infrastructure
tpm_book_pt, hazmat_book_pt = build_phasetype_tpm_book(pt_surrog, markov_surrog, books_test, panel_data)
fbmats_pt = build_fbmats_phasetype_with_indices(model_test.subjectindices, pt_surrog)
emat_pt = build_phasetype_emat_expanded(model_test, pt_surrog; expanded_data=nothing, censoring_patterns=nothing)

# Build Markov FFBS infrastructure
fbmats_mk = build_fbmats(model_test)
absorbingstates = findall(vec(sum(model_test.tmat, dims=2)) .== 0)

# Sample paths for first 5 subjects and compare
println("\n  Sampling paths and comparing likelihoods...")
println("  " * "-"^60)
println("  Subj  Markov LogL(sur)  Markov LogL(tar)  PT LogL(sur)  PT LogL(tar)  Weight(MK)  Weight(PT)")
println("  " * "-"^60)

# Get current model parameters (use initial values)
params_cur = get_parameters_flat(model_test)
target_pars = unflatten_parameters(params_cur, model_test)

for i in 1:min(5, length(model_test.subjectindices))
    # Sample Markov path
    # First compute fbmats for subject i
    subj_inds = model_test.subjectindices[i]
    subj_dat = view(panel_data, subj_inds, :)
    subj_tpm_map = view(books_test[2], subj_inds, :)
    subj_emat = view(model_test.emat, subj_inds, :)
    
    if any(subj_dat.obstype .∉ Ref([1, 2]))
        ForwardFiltering!(fbmats_mk[i], subj_dat, tpm_book_mk, subj_tpm_map, subj_emat;
                         hazmat_book=hazmat_book_mk)
    end
    path_mk = draw_samplepath(i, model_test, tpm_book_mk, hazmat_book_mk, books_test[2], fbmats_mk, absorbingstates)
    
    # Sample PhaseType path
    path_result_pt = draw_samplepath_phasetype(i, model_test, tpm_book_pt, hazmat_book_pt,
                                                books_test[2], fbmats_pt, emat_pt, pt_surrog, absorbingstates)
    path_pt = path_result_pt.collapsed
    
    # Compute surrogate log-likelihoods
    ll_surrog_mk = loglik(surrogate_pars, path_mk, markov_surrog.hazards, model_test)
    ll_surrog_pt = loglik_phasetype_collapsed_path(path_pt, pt_surrog)
    
    # Compute target log-likelihoods (using model hazards)
    ll_target_mk = loglik(target_pars, path_mk, model_test.hazards, model_test)
    ll_target_pt = loglik(target_pars, path_pt, model_test.hazards, model_test)
    
    # Importance weights
    weight_mk = ll_target_mk - ll_surrog_mk
    weight_pt = ll_target_pt - ll_surrog_pt
    
    println("  $(@sprintf("%4d", i))   $(@sprintf("%12.4f", ll_surrog_mk))   $(@sprintf("%14.4f", ll_target_mk))   $(@sprintf("%10.4f", ll_surrog_pt))   $(@sprintf("%11.4f", ll_target_pt))   $(@sprintf("%9.4f", weight_mk))   $(@sprintf("%9.4f", weight_pt))")
end
println("  " * "-"^60)
println()

# =============================================================================
# Part 6: Run MCEM with Both Proposals
# =============================================================================

println("Part 6: Compare MCEM with Markov vs PhaseType Proposals")
println("-"^70)

# Fit with Markov proposal
println("\n  Fitting with Markov proposal...")
h12_markov_mcem = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                         degree=1, knots=Float64[], boundaryknots=[0.0, 5.0], 
                         extrapolation="linear")
model_markov_mcem = multistatemodel(h12_markov_mcem; data=panel_data, surrogate=:markov)

fitted_markov_mcem = fit(model_markov_mcem;
    proposal=:markov,
    penalty=:none,
    verbose=false,
    maxiter=20,
    tol=0.05,
    ess_target_initial=30,
    max_ess=200,
    compute_vcov=false,
    compute_ij_vcov=false)

pars_markov = get_parameters(fitted_markov_mcem, 1, scale=:log)
println("  Markov proposal - Spline coefficients: $(round.(get_parameters_flat(fitted_markov_mcem), digits=4))")

# Evaluate hazard at several times
println("\n  Hazard evaluation (Markov proposal):")
println("    Time    h(t)      True")
for t in [0.5, 1.5, 2.5, 3.5, 4.5]
    h_fitted = fitted_markov_mcem.hazards[1](t, pars_markov, NamedTuple())
    println("    $(@sprintf("%.1f", t))     $(@sprintf("%.4f", h_fitted))    $(@sprintf("%.4f", true_rate))")
end

# Fit with PhaseType proposal
println("\n  Fitting with PhaseType proposal...")
h12_pt_mcem = Hazard(@formula(0 ~ 1), "sp", 1, 2; 
                     degree=1, knots=Float64[], boundaryknots=[0.0, 5.0], 
                     extrapolation="linear")
model_pt_mcem = multistatemodel(h12_pt_mcem; data=panel_data, surrogate=:markov)

fitted_pt_mcem = fit(model_pt_mcem;
    proposal=PhaseTypeProposal(n_phases=3),
    penalty=:none,
    verbose=false,
    maxiter=20,
    tol=0.05,
    ess_target_initial=30,
    max_ess=200,
    compute_vcov=false,
    compute_ij_vcov=false)

pars_pt = get_parameters(fitted_pt_mcem, 1, scale=:log)
println("  PhaseType proposal - Spline coefficients: $(round.(get_parameters_flat(fitted_pt_mcem), digits=4))")

println("\n  Hazard evaluation (PhaseType proposal):")
println("    Time    h(t)      True     Markov h(t)")
for t in [0.5, 1.5, 2.5, 3.5, 4.5]
    h_pt = fitted_pt_mcem.hazards[1](t, pars_pt, NamedTuple())
    h_mk = fitted_markov_mcem.hazards[1](t, pars_markov, NamedTuple())
    println("    $(@sprintf("%.1f", t))     $(@sprintf("%.4f", h_pt))    $(@sprintf("%.4f", true_rate))     $(@sprintf("%.4f", h_mk))")
end

# =============================================================================
# Part 7: Diagnostic Summary
# =============================================================================

println("\n" * "="^70)
println("DIAGNOSTIC SUMMARY")
println("="^70)

h_markov_mean = mean([fitted_markov_mcem.hazards[1](t, pars_markov, NamedTuple()) for t in 0.5:0.5:4.5])
h_pt_mean = mean([fitted_pt_mcem.hazards[1](t, pars_pt, NamedTuple()) for t in 0.5:0.5:4.5])

println("\n  True rate: $(@sprintf("%.4f", true_rate))")
println("  Markov surrogate rate: $(@sprintf("%.4f", surrog_rate))")
println("  Direct Markov fit rate: $(@sprintf("%.4f", rate_markov))")
println()
println("  MCEM (Markov proposal) mean hazard: $(@sprintf("%.4f", h_markov_mean))")
println("  MCEM (PhaseType proposal) mean hazard: $(@sprintf("%.4f", h_pt_mean))")
println("  Discrepancy: $(@sprintf("%.1f%%", 100*abs(h_pt_mean - h_markov_mean)/h_markov_mean))")
println()

if abs(h_pt_mean - h_markov_mean) / h_markov_mean > 0.35
    println("  ⚠️  WARNING: PhaseType and Markov proposals differ by >35%")
    println("      This indicates a bug in the importance weight calculation")
    println("      or the phase-type surrogate construction.")
else
    println("  ✓ PhaseType and Markov proposals agree within tolerance")
end

println()
println("="^70)
