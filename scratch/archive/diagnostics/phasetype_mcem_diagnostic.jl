#=
Diagnostic script for PhaseType MCEM proposal bug - v2

This script traces through the PhaseType importance sampling pipeline to identify
why MCEM returns IDENTICAL parameter values with PhaseType proposal.
=#

using Test
using MultistateModels
using DataFrames
using Random
using Statistics
using LinearAlgebra
using Printf

# Import internal functions
import MultistateModels: Hazard, multistatemodel, fit, set_parameters!, simulate,
    get_parameters_flat, get_parameters, MarkovProposal, PhaseTypeProposal,
    fit_surrogate, build_tpm_mapping, build_hazmat_book, build_tpm_book,
    compute_hazmat!, compute_tmat!, ExpMethodGeneric, ExponentialUtilities,
    needs_phasetype_proposal, resolve_proposal_config, SamplePath, @formula,
    MarkovSurrogate, PhaseTypeSurrogate,
    # Sampling functions
    DrawSamplePaths!, draw_samplepath, draw_samplepath_phasetype,
    ForwardFiltering!, BackwardSampling_expanded,
    convert_expanded_path_to_censored_data, compute_forward_loglik,
    collapse_phasetype_path, get_hazard_params, loglik, unflatten_parameters,
    build_phasetype_emat_expanded, build_phasetype_tpm_book,
    # PhaseType construction
    _build_phasetype_from_markov, PhaseTypeConfig,
    build_fbmats_phasetype_with_indices

const RNG_SEED = 0xABCD1234
const N_SUBJECTS = 100  # Small sample for debugging

println("="^80)
println("PhaseType MCEM Diagnostic v2")
println("="^80)

# ============================================================================
# 1. Generate test data (same setup as longtest)
# ============================================================================

println("\n[1] Generating test data...")

# Weibull hazards (semi-Markov) - progressive 3-state model (1→2→3)
true_shape_12, true_scale_12 = 1.3, 0.15
true_shape_23, true_scale_23 = 1.1, 0.12

true_params = (
    h12 = [true_shape_12, true_scale_12],
    h23 = [true_shape_23, true_scale_23]
)

h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
h23 = Hazard(@formula(0 ~ 1), "wei", 2, 3)

# Generate panel data template
obs_times = collect(0.0:2.0:14.0)
nobs = length(obs_times) - 1

template = DataFrame(
    id = repeat(1:N_SUBJECTS, inner=nobs),
    tstart = repeat(obs_times[1:end-1], N_SUBJECTS),
    tstop = repeat(obs_times[2:end], N_SUBJECTS),
    statefrom = ones(Int, N_SUBJECTS * nobs),
    stateto = ones(Int, N_SUBJECTS * nobs),
    obstype = fill(2, N_SUBJECTS * nobs)
)

# Simulate panel data
Random.seed!(RNG_SEED + 10)
model_template = multistatemodel(h12, h23; data=template, initialize=false)
for (haz_idx, haz_name) in enumerate(keys(true_params))
    set_parameters!(model_template, haz_idx, true_params[haz_name])
end

# Use obstype_map: 1→2 is panel (obstype=2), 2→3 is exact (obstype=1)
obstype_map = Dict(1 => 2, 2 => 1)
sim_result = simulate(model_template; paths=false, data=true, nsim=1, autotmax=false,
                     obstype_by_transition=obstype_map)
panel_data = sim_result[1, 1]

println("  - Generated panel data with $(nrow(panel_data)) observations for $N_SUBJECTS subjects")
println("  - Observation types: ", sort(unique(panel_data.obstype)))

# ============================================================================
# 2. Create model with PhaseType proposal
# ============================================================================

println("\n[2] Creating model with PhaseType proposal...")

h12_pt = Hazard(@formula(0 ~ 1), "wei", 1, 2)
h23_pt = Hazard(@formula(0 ~ 1), "wei", 2, 3)
model_pt = multistatemodel(h12_pt, h23_pt; data=panel_data, surrogate=:markov)

println("  - Model created with $(length(model_pt.hazards)) hazards")
println("  - Subject indices: $(length(model_pt.subjectindices)) subjects")

# ============================================================================
# 3. Build PhaseType infrastructure (same as fit_mcem)
# ============================================================================

println("\n[3] Building PhaseType infrastructure...")

# Get Markov surrogate
markov_surrogate = model_pt.markovsurrogate
println("  - Markov surrogate parameters: ", markov_surrogate.parameters.flat)

# Build PhaseType surrogate using the same API as fit_mcem
n_phases = 3
proposal_config = PhaseTypeProposal(n_phases=n_phases)
phasetype_surrogate = _build_phasetype_from_markov(model_pt, markov_surrogate; 
                                                    config=proposal_config, verbose=false)
println("  - PhaseType surrogate created with $n_phases phases")
println("  - Expanded states: $(phasetype_surrogate.n_expanded_states)")
println("  - Observed states: $(phasetype_surrogate.n_observed_states)")

# Build books (TPM mapping)
books = build_tpm_mapping(model_pt.data)
println("  - Books built")

# Build PhaseType TPM and hazmat books
tpm_book_ph, hazmat_book_ph = build_phasetype_tpm_book(phasetype_surrogate, markov_surrogate, books, model_pt.data)
println("  - PhaseType TPM book built, $(length(tpm_book_ph)) covariate combos")
println("  - Hazmat book has $(length(hazmat_book_ph)) Q matrices")

# Build emission matrix  
emat_ph = build_phasetype_emat_expanded(model_pt, phasetype_surrogate)
println("  - Emission matrix shape: $(size(emat_ph))")

# Build forward/backward matrices
fbmats_ph = build_fbmats_phasetype_with_indices(model_pt.subjectindices, phasetype_surrogate)
println("  - Forward/backward matrices allocated for $(length(fbmats_ph)) subjects")

# Get absorbing states
absorbingstates = [s for s in 1:size(model_pt.tmat, 1) if all(model_pt.tmat[s, :] .== 0)]
println("  - Absorbing states: ", absorbingstates)

# ============================================================================
# 4. Sample a single path and trace the likelihood computation
# ============================================================================

println("\n[4] Sampling a single path and tracing likelihood computation...")

# Pick a subject that has some transitions
# Find a subject that's not just staying in state 1
subj = 1
for s in 1:N_SUBJECTS
    subj_inds = model_pt.subjectindices[s]
    subj_dat = model_pt.data[subj_inds, :]
    if any(subj_dat.statefrom .!= 1) || any(subj_dat.stateto .!= 1)
        subj = s
        break
    end
end

subj_inds = model_pt.subjectindices[subj]
subj_dat = model_pt.data[subj_inds, :]

println("\n  Subject $subj data:")
for i in 1:nrow(subj_dat)
    println("    Row $i: t=$(subj_dat.tstart[i])-$(subj_dat.tstop[i]), s=$(subj_dat.statefrom[i])→$(subj_dat.stateto[i]), obstype=$(subj_dat.obstype[i])")
end

# Sample a path using PhaseType proposal
Random.seed!(12345)
path_result = draw_samplepath_phasetype(subj, model_pt, tpm_book_ph, hazmat_book_ph,
                                         books[2], fbmats_ph, emat_ph, 
                                         phasetype_surrogate, absorbingstates)

println("\n  Collapsed path:")
println("    times:  ", path_result.collapsed.times)
println("    states: ", path_result.collapsed.states)

println("\n  Expanded path:")
println("    times:  ", path_result.expanded.times)
println("    states: ", path_result.expanded.states)

# ============================================================================
# 5. Compute surrogate log-likelihood via forward algorithm
# ============================================================================

println("\n[5] Computing surrogate log-likelihood...")

censored_data, emat_path, tpm_map_path, tpm_book_path, hazmat_book_path = 
    convert_expanded_path_to_censored_data(
        path_result.expanded, phasetype_surrogate;
        hazmat = phasetype_surrogate.expanded_Q,
        schur_cache = nothing
    )

println("\n  Censored data from convert_expanded_path_to_censored_data:")
for i in 1:nrow(censored_data)
    println("    Row $i: t=$(round(censored_data.tstart[i], digits=3))-$(round(censored_data.tstop[i], digits=3)), " *
            "s=$(censored_data.statefrom[i])→$(censored_data.stateto[i]), " *
            "obstype=$(censored_data.obstype[i])")
end

println("\n  Emission matrix from convert:")
for i in 1:size(emat_path, 1)
    println("    Row $i: ", emat_path[i, :])
end

# Compute forward log-likelihood
loglik_forward = compute_forward_loglik(
    censored_data, emat_path, tpm_map_path, tpm_book_path, 
    hazmat_book_path, phasetype_surrogate.n_expanded_states
)
println("\n  Forward algorithm log-likelihood: $loglik_forward")

# ============================================================================
# 6. Compare with Markov surrogate path log-likelihood
# ============================================================================

println("\n[6] Computing Markov surrogate path log-likelihood...")

# Collapsed path likelihood under Markov surrogate
surrogate_pars = get_hazard_params(markov_surrogate.parameters, markov_surrogate.hazards)
loglik_markov_path = loglik(surrogate_pars, path_result.collapsed, markov_surrogate.hazards, model_pt)
println("  Markov path log-likelihood: $loglik_markov_path")

# Target likelihood under Weibull
target_pars = (h12 = [true_shape_12, true_scale_12], h23 = [true_shape_23, true_scale_23])
loglik_target = loglik(target_pars, path_result.collapsed, model_pt.hazards, model_pt)
println("  Target (Weibull) path log-likelihood: $loglik_target")

# ============================================================================
# 7. Compare importance weights
# ============================================================================

println("\n[7] Comparing importance weights...")

log_weight_markov = loglik_target - loglik_markov_path
log_weight_phasetype = loglik_target - loglik_forward

println("  Log importance weight (Markov):    $log_weight_markov")
println("  Log importance weight (PhaseType): $log_weight_phasetype")
println("  Difference: $(log_weight_phasetype - log_weight_markov)")

# ============================================================================
# 8. Sample multiple paths and check for variation
# ============================================================================

println("\n[8] Sampling multiple paths to check for variation...")

n_paths = 20
ll_forwards = Float64[]
ll_markovs = Float64[]
ll_targets = Float64[]

for j in 1:n_paths
    Random.seed!(12345 + j)
    path_res = draw_samplepath_phasetype(subj, model_pt, tpm_book_ph, hazmat_book_ph,
                                          books[2], fbmats_ph, emat_ph, 
                                          phasetype_surrogate, absorbingstates)
    
    # Forward likelihood
    cd, em, tm, tb, hb = convert_expanded_path_to_censored_data(
        path_res.expanded, phasetype_surrogate;
        hazmat = phasetype_surrogate.expanded_Q,
        schur_cache = nothing
    )
    ll_fwd = compute_forward_loglik(cd, em, tm, tb, hb, phasetype_surrogate.n_expanded_states)
    push!(ll_forwards, ll_fwd)
    
    # Markov path likelihood
    ll_mk = loglik(surrogate_pars, path_res.collapsed, markov_surrogate.hazards, model_pt)
    push!(ll_markovs, ll_mk)
    
    # Target likelihood
    ll_tgt = loglik(target_pars, path_res.collapsed, model_pt.hazards, model_pt)
    push!(ll_targets, ll_tgt)
end

println("\n  Summary statistics:")
println("  Forward likelihoods:  mean=$(round(mean(ll_forwards), digits=3)), std=$(round(std(ll_forwards), digits=3)), range=$(round(minimum(ll_forwards), digits=3)) to $(round(maximum(ll_forwards), digits=3))")
println("  Markov likelihoods:   mean=$(round(mean(ll_markovs), digits=3)), std=$(round(std(ll_markovs), digits=3)), range=$(round(minimum(ll_markovs), digits=3)) to $(round(maximum(ll_markovs), digits=3))")
println("  Target likelihoods:   mean=$(round(mean(ll_targets), digits=3)), std=$(round(std(ll_targets), digits=3)), range=$(round(minimum(ll_targets), digits=3)) to $(round(maximum(ll_targets), digits=3))")

println("\n  Log importance weights:")
weights_mk = ll_targets .- ll_markovs
weights_ph = ll_targets .- ll_forwards
println("  Markov weights:    mean=$(round(mean(weights_mk), digits=3)), std=$(round(std(weights_mk), digits=3))")
println("  PhaseType weights: mean=$(round(mean(weights_ph), digits=3)), std=$(round(std(weights_ph), digits=3))")

# ============================================================================
# 9. Check if forward likelihoods are all equal (key diagnostic)
# ============================================================================

println("\n[9] KEY DIAGNOSTIC: Are forward likelihoods all equal?")
if allequal(ll_forwards)
    println("  ⚠️  ALL FORWARD LIKELIHOODS ARE EQUAL! This is a bug.")
    println("      Value: $(first(ll_forwards))")
else
    println("  ✓ Forward likelihoods vary (as expected)")
end

# ============================================================================
# 10. Check the expanded Q matrix structure
# ============================================================================

println("\n[10] Expanded Q matrix structure:")
Q = phasetype_surrogate.expanded_Q
n_exp = size(Q, 1)
println("  Dimension: $n_exp × $n_exp")
println("  Diagonal: ", diag(Q))
println("  Row sums: ", sum(Q, dims=2)[:])

println("\n" * "="^80)
println("Diagnostic complete")
println("="^80)
